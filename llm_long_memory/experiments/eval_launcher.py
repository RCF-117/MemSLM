"""Shared evaluation launcher for MemoryManager-backed experiment runners."""

from __future__ import annotations

import copy

from llm_long_memory.evaluation.eval_runner import run_eval
from llm_long_memory.llm.ollama_client import LLM
from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.utils.helpers import load_config


def run_one_dataset(
    config_path: str,
    dataset_path: str,
    sample_limit: int,
    model_name: str | None = None,
    resume_run_id: str | None = None,
) -> str:
    """Load config from disk and run one dataset through the active runtime."""
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
    """Run one dataset with an already-materialized config mapping."""
    config = copy.deepcopy(config)
    config["dataset"]["eval_max_instances"] = sample_limit

    llm_cfg = config["llm"]
    selected_model = (model_name or str(llm_cfg["default_model"])).strip() or str(
        llm_cfg["default_model"]
    )
    llm = LLM(model_name=selected_model, host=str(llm_cfg["host"]))
    manager = MemoryManager(llm=llm, config=config)
    try:
        return run_eval(manager, dataset_path, config, resume_run_id=resume_run_id)
    finally:
        manager.close()
