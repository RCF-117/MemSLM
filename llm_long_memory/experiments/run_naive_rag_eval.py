"""Run a standalone classic retrieve-then-generate baseline."""

from __future__ import annotations

import argparse

from llm_long_memory.experiments.cli_utils import (
    load_runtime_config,
    parse_csv,
    prepare_eval_dataset,
)
from llm_long_memory.experiments.direct_eval_runner import build_naive_rag_prompt, run_direct_mode_eval


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


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config)
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
    print(
        f"subset_dataset: {prepared.effective_dataset}"
        if prepared.subset_built
        else f"dataset: {prepared.source_dataset} (no extra subset built)"
    )
    model_name = args.model.strip() or str(config["llm"]["default_model"])
    top_k = int(config["retrieval"].get("top_k", 5))
    run_id = run_direct_mode_eval(
        mode_name=MODE_NAME,
        config=config,
        dataset_path=prepared.effective_dataset,
        dataset_name=prepared.dataset_name,
        model_name=model_name,
        prompt_builder=lambda instance, _cfg: build_naive_rag_prompt(instance, top_k=top_k),
        resume_run_id=(args.resume_run_id.strip() or None),
    )
    print(f"[{MODE_NAME}] run_id={run_id}")


if __name__ == "__main__":
    main()
