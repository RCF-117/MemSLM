"""Utility helpers for configuration and common text operations."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


def project_root() -> Path:
    """Return package project root: llm_long_memory/."""
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str) -> Path:
    """Resolve absolute/relative path with llm_long_memory as base."""
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return project_root() / raw


def _resolve_config_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    cwd_candidate = Path.cwd() / raw_path
    package_candidate = resolve_project_path(path)
    return cwd_candidate if cwd_candidate.exists() else package_candidate


def _require(mapping: Dict[str, Any], key: str, parent: str) -> Dict[str, Any]:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {parent}{key}")
    value = mapping[key]
    if not isinstance(value, dict):
        raise ValueError(f"Config key must be a mapping: {parent}{key}")
    return value


def _validate_config(data: Dict[str, Any]) -> None:
    """Validate minimal required config schema for runtime safety."""
    required_top = {
        "memory",
        "retrieval",
        "embedding",
        "logging",
        "llm",
        "dataset",
        "evaluation",
    }
    missing = sorted(required_top.difference(set(data.keys())))
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(missing)}")

    memory = _require(data, "memory", "")
    if "short_memory_size" not in memory:
        raise ValueError("Missing required config key: memory.short_memory_size")
    _require(memory, "mid_memory", "memory.")

    retrieval = _require(data, "retrieval", "")
    for key in ("top_k", "chunks_per_topic", "hybrid_alpha", "keyword_weight"):
        if key not in retrieval:
            raise ValueError(f"Missing required config key: retrieval.{key}")
    answering = _require(retrieval, "answering", "retrieval.")
    for key in (
        "context_only",
        "log_decision_details",
        "evidence_top_n_chunks",
        "evidence_top_n_sentences",
        "evidence_sentence_max_chars",
        "candidate_top_n",
        "llm_fallback_to_top_candidate",
        "fallback_min_score",
        "response_evidence_min_token_overlap",
        "response_evidence_min_shared_tokens",
        "span_min_tokens",
        "span_max_tokens",
        "span_top_n_per_sentence",
    ):
        if key not in answering:
            raise ValueError(f"Missing required config key: retrieval.answering.{key}")
    _require(answering, "intent_extraction", "retrieval.answering.")
    _require(answering, "candidate_scoring", "retrieval.answering.")

    embedding = _require(data, "embedding", "")
    for key in ("dim", "model"):
        if key not in embedding:
            raise ValueError(f"Missing required config key: embedding.{key}")

    logging_cfg = _require(data, "logging", "")
    for key in ("log_file", "level", "console_enabled"):
        if key not in logging_cfg:
            raise ValueError(f"Missing required config key: logging.{key}")

    llm = _require(data, "llm", "")
    for key in ("default_model", "supported_models", "host", "temperature", "request_timeout_sec"):
        if key not in llm:
            raise ValueError(f"Missing required config key: llm.{key}")

    dataset = _require(data, "dataset", "")
    if "stream_mode" not in dataset:
        raise ValueError("Missing required config key: dataset.stream_mode")
    eval_splits = dataset.get("eval_splits")
    if eval_splits is not None:
        if not isinstance(eval_splits, dict):
            raise ValueError("Config key must be a mapping: dataset.eval_splits")
        for key, value in eval_splits.items():
            if not str(key).strip():
                raise ValueError("Config key dataset.eval_splits contains empty split name.")
            if not str(value).strip():
                raise ValueError(
                    f"Config key dataset.eval_splits.{key} must be a non-empty path."
                )
    default_eval_split = dataset.get("default_eval_split")
    if default_eval_split is not None:
        if not isinstance(default_eval_split, str) or not default_eval_split.strip():
            raise ValueError("Config key dataset.default_eval_split must be a non-empty string.")
        if isinstance(eval_splits, dict) and default_eval_split not in eval_splits:
            raise ValueError(
                "Config key dataset.default_eval_split must reference dataset.eval_splits."
            )

    evaluation = _require(data, "evaluation", "")
    for key in ("save_to_db", "run_table", "result_table", "group_table"):
        if key not in evaluation:
            raise ValueError(f"Missing required config key: evaluation.{key}")


@lru_cache(maxsize=8)
def _load_config_cached(path_str: str) -> Dict[str, Any]:
    config_path = Path(path_str)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping.")
    _validate_config(data)
    return data


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration and return it as a dictionary."""
    config_path = _resolve_config_path(path).resolve()
    return copy.deepcopy(_load_config_cached(str(config_path)))
