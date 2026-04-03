"""Run frozen Mid-RAG baseline evaluations with fixed protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_long_memory.evaluation.eval_runner import run_eval
from llm_long_memory.llm.ollama_client import LLM
from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.utils.helpers import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen baseline evaluations.")
    parser.add_argument(
        "--config",
        type=str,
        default="llm_long_memory/baselines/baseline_midrag_v1.yaml",
        help="Baseline config file.",
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
        default="llm_long_memory/data/raw/longmemeval_s_sample_20.json",
        help="Path to LongMemEval sample-20 file.",
    )
    parser.add_argument(
        "--oracle_path",
        type=str,
        default="llm_long_memory/data/raw/longmemeval_oracle.json",
        help="Path to LongMemEval oracle file.",
    )
    return parser.parse_args()


def run_one_dataset(config_path: str, dataset_path: str, sample_limit: int) -> None:
    config = load_config(config_path)
    config["dataset"]["eval_max_instances"] = sample_limit

    llm_cfg = config["llm"]
    llm = LLM(model_name=str(llm_cfg["default_model"]), host=str(llm_cfg["host"]))
    manager = MemoryManager(llm=llm, config=config)
    try:
        run_eval(manager, dataset_path, config)
    finally:
        manager.close()


def main() -> None:
    args = parse_args()
    sample20 = str(Path(args.sample20_path))
    oracle = str(Path(args.oracle_path))

    print(f"[Baseline] sample20: {sample20}")
    run_one_dataset(args.config, sample20, args.sample_limit)

    print(f"[Baseline] oracle: {oracle}")
    run_one_dataset(args.config, oracle, args.sample_limit)


if __name__ == "__main__":
    main()
