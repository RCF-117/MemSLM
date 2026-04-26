"""Run frozen Mid-RAG baseline evaluations with fixed protocol."""

from __future__ import annotations

import argparse
import copy
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


def run_one_dataset(
    config_path: str,
    dataset_path: str,
    sample_limit: int,
    model_name: str | None = None,
    resume_run_id: str | None = None,
) -> str:
    config = load_config(config_path)
    config["dataset"]["eval_max_instances"] = sample_limit
    return run_one_dataset_with_config(
        config=config,
        dataset_path=dataset_path,
        sample_limit=sample_limit,
        model_name=model_name,
        resume_run_id=resume_run_id,
    )


def run_one_dataset_with_config(
    *,
    config: dict,
    dataset_path: str,
    sample_limit: int,
    model_name: str | None = None,
    resume_run_id: str | None = None,
) -> str:
    config = copy.deepcopy(config)
    config["dataset"]["eval_max_instances"] = sample_limit

    llm_cfg = config["llm"]
    selected_model = (model_name or str(llm_cfg["default_model"])).strip() or str(llm_cfg["default_model"])
    llm = LLM(model_name=selected_model, host=str(llm_cfg["host"]))
    manager = MemoryManager(llm=llm, config=config)
    try:
        return run_eval(manager, dataset_path, config, resume_run_id=resume_run_id)
    finally:
        manager.close()


def main() -> None:
    args = parse_args()
    sample20 = str(Path(args.sample20_path))
    oracle = str(Path(args.oracle_path))
    model_override = args.model.strip() or None

    print(f"[Baseline] sample20: {sample20}")
    run_one_dataset(args.config, sample20, args.sample_limit, model_name=model_override)

    print(f"[Baseline] oracle: {oracle}")
    run_one_dataset(args.config, oracle, args.sample_limit, model_name=model_override)


if __name__ == "__main__":
    main()
