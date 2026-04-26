"""One-shot thesis evaluation runner for compact, reproducible experiments."""

from __future__ import annotations

import argparse

from llm_long_memory.experiments.cli_utils import (
    load_runtime_config,
    parse_csv,
    prepare_eval_dataset,
    register_mode_run,
)
from llm_long_memory.experiments.eval_launcher import run_one_dataset
from llm_long_memory.experiments.export_eval_report import export_report
from llm_long_memory.utils.helpers import sanitize_filename_part


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
    config = load_runtime_config(args.config)
    default_model = str(config["llm"]["default_model"])
    prepared = prepare_eval_dataset(
        config=config,
        dataset=(args.dataset.strip() or None),
        split=(args.split.strip() or None),
        max_total=int(args.max_total),
        per_type=int(args.per_type),
        seed=int(args.seed),
        keep_types=(parse_csv(args.keep_types) or parse_csv(args.include_types)),
        drop_types=parse_csv(args.drop_types),
        subset_output=args.subset_output.strip() or None,
    )
    swap_roles = bool(args.swap_roles)
    report_dir = args.report_dir.strip() or str(
        config["evaluation"].get("thesis_report_dir", "data/processed/thesis_reports_debug_analysis")
    )
    graph_output_dir = args.graph_output_dir.strip() or str(
        config["evaluation"].get("thesis_graph_dir", "data/graphs_thesis_debug_analysis")
    )
    print(
        f"subset_dataset: {prepared.effective_dataset}"
        if prepared.subset_built
        else f"dataset: {prepared.source_dataset} (no extra subset built)"
    )

    model_override = args.model.strip() or default_model
    judge_override = args.judge_model.strip() or default_model
    if swap_roles:
        model_override, judge_override = judge_override, model_override

    run_id = run_one_dataset(
        args.config,
        prepared.effective_dataset,
        int(args.max_total) if int(args.max_total) > 0 else 0,
        model_name=model_override,
        resume_run_id=(args.resume_run_id.strip() or None),
    )
    dataset_name = prepared.dataset_name
    dataset_artifact_name = prepared.source_dataset.rsplit("/", 1)[-1]
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
    try:
        register_mode_run(
            config=config,
            dataset_name=dataset_name,
            mode="memslm",
            run_id=run_id,
            model_name=model_override,
            judge_model=judge_override,
        )
    except Exception as exc:  # pragma: no cover - best effort registration
        print(f"[warn] failed to register memslm mode metadata: {exc}")


if __name__ == "__main__":
    main()
