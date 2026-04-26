"""Run the thesis ablation: filtered evidence is sent directly to the final 8B."""

from __future__ import annotations

import argparse
import copy
import sqlite3
from pathlib import Path
from typing import Tuple

from llm_long_memory.baselines.run_baseline import run_one_dataset_with_config
from llm_long_memory.experiments.build_eval_subset import build_subset
from llm_long_memory.experiments.run_thesis_eval import _parse_csv, _resolve_dataset_path
from llm_long_memory.evaluation.eval_store import EvalStore
from llm_long_memory.utils.helpers import (
    dataset_display_name,
    load_config,
    resolve_project_path,
    sanitize_filename_part,
)


MODE_NAME = "ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the filter-only ablation baseline.")
    parser.add_argument(
        "--config",
        default="llm_long_memory/config/config.yaml",
        help="Main config path. The runner overrides it into filter-only mode.",
    )
    parser.add_argument("--dataset", default="", help="Explicit dataset path.")
    parser.add_argument("--split", default="", help="Configured dataset split name.")
    parser.add_argument("--max-total", type=int, default=0, help="Cap total instances.")
    parser.add_argument("--per-type", type=int, default=0, help="Cap instances per question type.")
    parser.add_argument("--seed", type=int, default=42, help="Subset sampling seed.")
    parser.add_argument("--keep-types", default="", help="Comma-separated question types to keep.")
    parser.add_argument("--include-types", default="", help="Alias of --keep-types.")
    parser.add_argument("--drop-types", default="", help="Comma-separated question types to drop.")
    parser.add_argument("--model", default="", help="Optional model override.")
    parser.add_argument("--subset-output", default="", help="Optional subset JSON output path.")
    parser.add_argument("--resume-run-id", default="", help="Resume an existing run id.")
    return parser.parse_args()


def _prepare_dataset(args: argparse.Namespace, config: dict) -> Tuple[str, str]:
    source_dataset = _resolve_dataset_path(config, args.dataset.strip() or None, args.split.strip() or None)
    source_name = dataset_display_name(source_dataset)
    keep_types = _parse_csv(args.keep_types) or _parse_csv(args.include_types)
    drop_types = _parse_csv(args.drop_types)
    needs_subset = int(args.max_total) > 0 or int(args.per_type) > 0 or bool(keep_types) or bool(drop_types)
    if not needs_subset:
        print(f"dataset: {source_dataset} (no extra subset built)")
        return source_name, source_dataset

    if args.subset_output.strip():
        subset_path = str(resolve_project_path(args.subset_output))
    else:
        subset_dir = resolve_project_path(
            str(config["evaluation"].get("thesis_subset_dir", "data/raw/LongMemEval/thesis_subsets"))
        )
        subset_dir.mkdir(parents=True, exist_ok=True)
        source_stem = sanitize_filename_part(Path(source_dataset).stem)
        keep_tag = f"__keep-{sanitize_filename_part('-'.join(sorted(keep_types)))}" if keep_types else ""
        drop_tag = f"__drop-{sanitize_filename_part('-'.join(sorted(drop_types)))}" if drop_types else ""
        subset_name = (
            f"{source_stem}__max{int(args.max_total)}__per{int(args.per_type)}"
            f"__seed{int(args.seed)}{keep_tag}{drop_tag}.json"
        )
        subset_path = str(subset_dir / subset_name)
    build_subset(
        source_path=source_dataset,
        output_path=subset_path,
        max_total=int(args.max_total),
        per_type=int(args.per_type),
        seed=int(args.seed),
        keep_types=keep_types,
        drop_types=drop_types,
    )
    print(f"subset_dataset: {subset_path}")
    return source_name, subset_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = copy.deepcopy(config)
    config.setdefault("retrieval", {})
    config["retrieval"]["execution_mode"] = "filter_only"
    answering_cfg = dict(config["retrieval"].get("answering", {}))
    answering_cfg["final_answer_guard_enabled"] = False
    answering_cfg["final_answer_second_pass_enabled"] = False
    answering_cfg["toolkit_direct_answer_enabled"] = False
    config["retrieval"]["answering"] = answering_cfg
    dataset_name, dataset_path = _prepare_dataset(args, config)
    model_name = args.model.strip() or str(config["llm"]["default_model"])
    run_id = run_one_dataset_with_config(
        config=config,
        dataset_path=dataset_path,
        sample_limit=int(args.max_total) if int(args.max_total) > 0 else 0,
        model_name=model_name,
        resume_run_id=(args.resume_run_id.strip() or None),
    )

    db_path = resolve_project_path(str(config["evaluation"].get("database_file", "data/processed/thesis_eval.db")))
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        store = EvalStore(conn=conn, eval_cfg=dict(config["evaluation"]))
        store.create_tables()
        store.ensure_schema_compat()
        store.log_thesis_mode_run(
            dataset_name=dataset_name,
            mode=MODE_NAME,
            run_id=run_id,
            model_name=model_name,
            judge_model="",
            commit=True,
        )
    finally:
        conn.close()

    print(f"[{MODE_NAME}] run_id={run_id}")


if __name__ == "__main__":
    main()
