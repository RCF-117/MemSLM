"""Compatibility wrapper for legacy baseline batch entrypoints.

The repository's active evaluation surfaces now live under
``llm_long_memory.experiments``. This module is kept as a thin wrapper so old
commands do not break abruptly, while all shared implementation stays in the
main experiment launcher.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_long_memory.experiments.eval_launcher import (
    run_one_dataset,
    run_one_dataset_with_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the legacy two-dataset baseline batch using the active launcher."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="llm_long_memory/config/config.yaml",
        help="Config file.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=10,
        help="Number of instances per dataset.",
    )
    parser.add_argument(
        "--sample20_path",
        type=str,
        default="llm_long_memory/data/raw/LongMemEval/longmemeval_s_sample_20.json",
        help="Path to LongMemEval sample-20 file.",
    )
    parser.add_argument(
        "--oracle_path",
        type=str,
        default="llm_long_memory/data/raw/LongMemEval/longmemeval_oracle.json",
        help="Path to LongMemEval oracle file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional LLM model override. Defaults to config.llm.default_model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample20 = str(Path(args.sample20_path))
    oracle = str(Path(args.oracle_path))
    model_override = args.model.strip() or None

    print(f"[Baseline] sample20: {sample20}")
    run_one_dataset(args.config, sample20, args.sample_limit, model_name=model_override)

    print(f"[Baseline] oracle: {oracle}")
    run_one_dataset(args.config, oracle, args.sample_limit, model_name=model_override)


__all__ = ["run_one_dataset", "run_one_dataset_with_config"]


if __name__ == "__main__":
    main()
