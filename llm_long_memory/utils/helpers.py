"""Utility helpers for configuration, dataset naming, and common text operations."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Set

import yaml


_DATASET_DISPLAY_NAMES = {
    "sample20": "LongMemEval Sample-20 Split",
    "longmemeval_s_sample_20": "LongMemEval Sample-20 Split",
    "oracle": "LongMemEval Oracle Split",
    "longmemeval_oracle": "LongMemEval Oracle Split",
    "ragdebug10": "LongMemEval Diagnostic Split",
    "longmemeval_ragdebug10_rebuilt": "LongMemEval Diagnostic Split",
    "diagnostic_heldout20": "LongMemEval Held-Out Matched Split",
    "longmemeval_eval_subset_matched_to_diagnostic_split": "LongMemEval Held-Out Matched Split",
    "quick5": "LongMemEval Diagnostic Mini Split",
    "locomo10": "LoCoMo Evaluation Subset",
    "locomo_matched20": "LoCoMo Matched-Distribution 20-QA Subset",
    "locomo20_matched_distribution": "LoCoMo Matched-Distribution 20-QA Subset",
    "locomo": "LoCoMo Evaluation Set",
}


def project_root() -> Path:
    """Return package project root: llm_long_memory/."""
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str) -> Path:
    """Resolve absolute/relative path with llm_long_memory as base.

    Accepts paths already prefixed with ``llm_long_memory/`` and strips the
    redundant package prefix so callers can use either ``data/...`` or
    ``llm_long_memory/data/...`` without creating duplicate nested paths.
    """
    raw = Path(path)
    if raw.is_absolute():
        return raw
    parts = raw.parts
    root_name = project_root().name
    if parts and parts[0] == root_name:
        raw = Path(*parts[1:]) if len(parts) > 1 else Path('.')
    return project_root() / raw


def sanitize_filename_part(value: str) -> str:
    """Return a filesystem-friendly filename component."""
    text = " ".join(str(value).split()).strip()
    if not text:
        return "unknown"
    chars: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("_")
    cleaned = "".join(chars).strip("._")
    return cleaned or "unknown"


def dataset_name_aliases(value: str | None) -> Set[str]:
    """Return normalized aliases for a dataset identifier or path."""
    raw = str(value or "").strip()
    if not raw:
        return set()
    path = Path(raw)
    aliases = {
        raw.lower(),
        path.name.lower(),
        path.stem.lower(),
    }
    for item in list(aliases):
        if item.endswith(".json"):
            aliases.add(item[:-5])
    return {alias for alias in aliases if alias}


def dataset_display_name(value: str | None) -> str:
    """Return a paper-facing dataset display name while preserving raw filenames internally."""
    aliases = dataset_name_aliases(value)
    for alias in aliases:
        if alias in _DATASET_DISPLAY_NAMES:
            return _DATASET_DISPLAY_NAMES[alias]
    joined = " ".join(sorted(aliases))
    if "ragdebug" in joined and "quick5" in joined:
        return "LongMemEval Diagnostic Mini Split"
    if "diagnostic_heldout20" in joined or "matched_to_diagnostic_split" in joined:
        return "LongMemEval Held-Out Matched Split"
    if "ragdebug" in joined:
        return "LongMemEval Diagnostic Split"
    if "sample20" in joined or "sample_20" in joined:
        return "LongMemEval Sample-20 Split"
    if "oracle" in joined and "longmemeval" in joined:
        return "LongMemEval Oracle Split"
    if "locomo10" in joined:
        return "LoCoMo Evaluation Subset"
    if "locomo_matched20" in joined or "locomo20_matched_distribution" in joined:
        return "LoCoMo Matched-Distribution 20-QA Subset"
    if "locomo" in joined:
        return "LoCoMo Evaluation Set"
    if "longmemeval" in joined:
        return "LongMemEval Evaluation Set"
    raw = str(value or "").strip()
    if not raw:
        return "Unknown Dataset"
    return Path(raw).name or raw


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
    for key in ("top_k", "hybrid_alpha", "keyword_weight"):
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
        "response_evidence_min_token_overlap",
        "response_evidence_min_shared_tokens",
        "response_evidence_relaxed_overlap_enabled",
        "response_evidence_relaxed_min_token_overlap",
        "response_evidence_relaxed_min_shared_tokens",
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
