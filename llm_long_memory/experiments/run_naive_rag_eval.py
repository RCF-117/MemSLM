"""Run a standalone classic retrieve-then-generate baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from llm_long_memory.experiments.build_eval_subset import build_subset
from llm_long_memory.experiments.run_thesis_eval import _parse_csv, _resolve_dataset_path
from llm_long_memory.experiments.direct_eval_runner import build_naive_rag_prompt, run_direct_mode_eval
from llm_long_memory.utils.helpers import (
    dataset_display_name,
    load_config,
    resolve_project_path,
    sanitize_filename_part,
)


MODE_NAME = "naive rag"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a naive RAG thesis baseline.")
    parser.add_argument("--config", default="llm_long_memory/config/config.yaml", help="Config path.")
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
    dataset_name, dataset_path = _prepare_dataset(args, config)
    model_name = args.model.strip() or str(config["llm"]["default_model"])
    top_k = int(config["retrieval"].get("top_k", 5))
    run_id = run_direct_mode_eval(
        mode_name=MODE_NAME,
        config=config,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        model_name=model_name,
        prompt_builder=lambda instance, _cfg: build_naive_rag_prompt(instance, top_k=top_k),
        resume_run_id=(args.resume_run_id.strip() or None),
    )
    print(f"[{MODE_NAME}] run_id={run_id}")


if __name__ == "__main__":
    main()
