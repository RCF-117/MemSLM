"""One-shot thesis evaluation runner for compact, reproducible experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from llm_long_memory.baselines.run_baseline import run_one_dataset
from llm_long_memory.experiments.build_eval_subset import build_subset
from llm_long_memory.experiments.export_eval_report import export_report
from llm_long_memory.utils.helpers import (
    dataset_display_name,
    load_config,
    resolve_project_path,
    sanitize_filename_part,
)


def _parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _resolve_dataset_path(config: dict, dataset: str | None, split: str | None) -> str:
    if dataset:
        return str(resolve_project_path(dataset))
    dataset_cfg = config["dataset"]
    split_map = dataset_cfg.get("eval_splits", {})
    if split:
        key = split.strip().lower()
    else:
        key = str(dataset_cfg.get("default_eval_split", "")).strip().lower()
    if not key or key not in split_map:
        known = ", ".join(sorted(str(k) for k in split_map.keys()))
        raise ValueError(f"Unknown dataset split '{key}'. Available: {known}")
    return str(resolve_project_path(str(split_map[key])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a thesis-friendly eval workflow.")
    parser.add_argument("--config", default="llm_long_memory/config/config.yaml", help="Config path.")
    parser.add_argument("--dataset", default="", help="Explicit dataset path.")
    parser.add_argument("--split", default="", help="Configured dataset split name.")
    parser.add_argument("--max-total", type=int, default=0, help="Cap total instances.")
    parser.add_argument("--per-type", type=int, default=0, help="Cap instances per question type.")
    parser.add_argument("--seed", type=int, default=42, help="Subset sampling seed.")
    parser.add_argument(
        "--keep-types",
        default="",
        help="Comma-separated question types to keep. Empty keeps all.",
    )
    parser.add_argument(
        "--include-types",
        default="",
        help="Alias of --keep-types for backward compatibility.",
    )
    parser.add_argument(
        "--drop-types",
        default="",
        help="Comma-separated question types to drop after keep filtering.",
    )
    parser.add_argument(
        "--report-dir",
        default="",
        help="Where to write the final report artifacts.",
    )
    parser.add_argument(
        "--graph-output-dir",
        default="",
        help="Where to write the full-run graph visualization artifacts.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Use a local LLM judge when exporting the final report.",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="Optional judge model override. Defaults to config.llm.default_model.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional experiment model override. Defaults to config.llm.default_model.",
    )
    parser.add_argument(
        "--subset-output",
        default="",
        help="Optional path to persist the sampled subset JSON.",
    )
    parser.add_argument(
        "--resume-run-id",
        default="",
        help="Optional existing eval run_id to resume.",
    )
    parser.add_argument(
        "--swap-roles",
        action="store_true",
        help="Swap the experiment model and judge model roles for the run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    default_model = str(config["llm"]["default_model"])
    source_dataset = _resolve_dataset_path(config, args.dataset.strip() or None, args.split.strip() or None)
    swap_roles = bool(args.swap_roles)
    report_dir = args.report_dir.strip() or str(
        config["evaluation"].get("thesis_report_dir", "data/processed/thesis_reports_debug_analysis")
    )
    graph_output_dir = args.graph_output_dir.strip() or str(
        config["evaluation"].get("thesis_graph_dir", "data/graphs_thesis_debug_analysis")
    )

    subset_path = source_dataset
    keep_types = _parse_csv(args.keep_types) or _parse_csv(args.include_types)
    drop_types = _parse_csv(args.drop_types)
    needs_subset = int(args.max_total) > 0 or int(args.per_type) > 0 or bool(keep_types) or bool(drop_types)
    if needs_subset:
        if args.subset_output.strip():
            subset_path = str(resolve_project_path(args.subset_output))
        else:
            subset_dir = resolve_project_path(
                str(config["evaluation"].get("thesis_subset_dir", "data/raw/LongMemEval/thesis_subsets"))
            )
            subset_dir.mkdir(parents=True, exist_ok=True)
            source_stem = sanitize_filename_part(Path(source_dataset).stem)
            keep_tag = (
                f"__keep-{sanitize_filename_part('-'.join(sorted(keep_types)))}"
                if keep_types
                else ""
            )
            drop_tag = (
                f"__drop-{sanitize_filename_part('-'.join(sorted(drop_types)))}"
                if drop_types
                else ""
            )
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
    else:
        print(f"subset_dataset: {source_dataset} (no extra subset built)")

    model_override = args.model.strip() or default_model
    judge_override = args.judge_model.strip() or default_model
    if swap_roles:
        model_override, judge_override = judge_override, model_override

    run_id = run_one_dataset(
        args.config,
        subset_path,
        int(args.max_total) if int(args.max_total) > 0 else 0,
        model_name=model_override,
        resume_run_id=(args.resume_run_id.strip() or None),
    )
    dataset_name = dataset_display_name(source_dataset)
    dataset_artifact_name = Path(source_dataset).name
    artifact_prefix = "__".join(
        [
            sanitize_filename_part(run_id),
            sanitize_filename_part(Path(dataset_artifact_name).stem),
            f"model-{sanitize_filename_part(model_override)}",
            f"judge-{sanitize_filename_part(judge_override)}",
        ]
    )
    report_db_path = str(config["evaluation"]["database_file"])
    graph_json_path = ""
    node_graph_json_path = ""
    export_report(
        db_path=report_db_path,
        output_dir=report_dir,
        run_id=(run_id if run_id else (args.resume_run_id.strip() or None)),
        dataset_name=dataset_name,
        model_name=model_override,
        judge_model=judge_override,
        artifact_prefix=artifact_prefix,
        graph_json_path=(graph_json_path or None),
        node_graph_json_path=(node_graph_json_path or None),
        judge_enabled=bool(args.judge),
    )
    # Register the latest memslm run so report builders can resolve it without rerunning eval.
    try:
        from llm_long_memory.evaluation.eval_store import EvalStore
        import sqlite3

        db_file = resolve_project_path(report_db_path)
        conn = sqlite3.connect(str(db_file))
        conn.row_factory = sqlite3.Row
        try:
            store = EvalStore(conn=conn, eval_cfg=dict(config["evaluation"]))
            store.create_tables()
            store.ensure_schema_compat()
            if run_id:
                store.log_thesis_mode_run(
                    dataset_name=dataset_name,
                    mode="memslm",
                    run_id=run_id,
                    model_name=model_override,
                    judge_model=judge_override,
                    commit=True,
                )
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover - best effort registration
        print(f"[warn] failed to register memslm mode metadata: {exc}")


if __name__ == "__main__":
    main()
