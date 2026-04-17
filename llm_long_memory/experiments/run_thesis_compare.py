"""Build a consolidated thesis comparison report from existing mode runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_long_memory.experiments.run_thesis_eval import _resolve_dataset_path
from llm_long_memory.experiments.thesis_report_builder import build_consolidated_report
from llm_long_memory.utils.helpers import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a consolidated thesis report.")
    parser.add_argument("--config", default="llm_long_memory/config/config.yaml", help="Main config path.")
    parser.add_argument("--dataset", default="", help="Optional dataset path used to infer the dataset name.")
    parser.add_argument("--split", default="", help="Configured dataset split name.")
    parser.add_argument(
        "--report-dir",
        default="",
        help="Directory for the final consolidated report.",
    )
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Dataset display name used to resolve the latest mode runs.",
    )
    parser.add_argument("--model-name", default="", help="Main model display name for the report.")
    parser.add_argument("--judge-model", default="", help="Judge model display name for the report.")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Use a local LLM judge when rebuilding the report.",
    )
    parser.add_argument("--model-only-run-id", default="", help="Explicit model-only run id.")
    parser.add_argument("--naive-rag-run-id", default="", help="Explicit naive-rag run id.")
    parser.add_argument("--memslm-run-id", default="", help="Explicit memslm run id.")
    parser.add_argument("--ablation-run-id", default="", help="Explicit ablation run id.")
    parser.add_argument(
        "--db-path",
        default="",
        help="Evaluation SQLite database path. Defaults to config evaluation.db.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    db_path = args.db_path.strip() or str(config["evaluation"].get("database_file", "data/processed/thesis_eval.db"))
    report_dir = args.report_dir.strip() or str(
        config["evaluation"].get("thesis_report_dir", "data/processed/thesis_reports_debug_analysis")
    )
    dataset_name = args.dataset_name.strip()
    if not dataset_name and (args.dataset.strip() or args.split.strip()):
        dataset_name = Path(
            _resolve_dataset_path(config, args.dataset.strip() or None, args.split.strip() or None)
        ).name
    model_name = args.model_name.strip() or str(config["llm"]["default_model"])
    judge_model = args.judge_model.strip() or str(config["llm"]["default_model"])
    mode_run_ids = {
        "model-only": args.model_only_run_id.strip() or None,
        "naive rag": args.naive_rag_run_id.strip() or None,
        "memslm": args.memslm_run_id.strip() or None,
        "ablation": args.ablation_run_id.strip() or None,
    }

    result = build_consolidated_report(
        db_path=db_path,
        output_dir=report_dir,
        dataset_name=dataset_name,
        model_name=model_name,
        judge_model=judge_model,
        judge_enabled=bool(args.judge),
        mode_run_ids=mode_run_ids,
        report_dir=report_dir,
    )
    print("Comparison report:")
    print(f"- json: {result['json']}")
    print(f"- markdown: {result['markdown']}")
    print(f"- csv: {result['csv']}")


if __name__ == "__main__":
    main()
