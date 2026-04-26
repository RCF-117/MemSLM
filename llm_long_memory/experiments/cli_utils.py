"""Shared CLI helpers for experiment runners.

These utilities keep dataset resolution, subset materialization, and run
registration consistent across the public experiment entrypoints.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from llm_long_memory.evaluation.eval_store import EvalStore
from llm_long_memory.experiments.build_eval_subset import build_subset
from llm_long_memory.utils.helpers import (
    dataset_display_name,
    load_config,
    resolve_project_path,
    sanitize_filename_part,
)


@dataclass(frozen=True)
class PreparedEvalDataset:
    """Resolved dataset paths for one eval runner invocation."""

    source_dataset: str
    effective_dataset: str
    dataset_name: str
    subset_built: bool


def parse_csv(value: str | None) -> List[str]:
    """Parse a comma-separated CLI value into trimmed tokens."""
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def resolve_dataset_path(config: dict, dataset: str | None, split: str | None) -> str:
    """Resolve a dataset path from an explicit path or configured split key."""
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


def prepare_eval_dataset(
    *,
    config: dict,
    dataset: str | None,
    split: str | None,
    max_total: int,
    per_type: int,
    seed: int,
    keep_types: Sequence[str] | None,
    drop_types: Sequence[str] | None,
    subset_output: str | None = None,
) -> PreparedEvalDataset:
    """Resolve the source dataset and optionally build a filtered subset."""
    source_dataset = resolve_dataset_path(config, dataset, split)
    dataset_name = dataset_display_name(source_dataset)
    keep = [str(x).strip() for x in list(keep_types or []) if str(x).strip()]
    drop = [str(x).strip() for x in list(drop_types or []) if str(x).strip()]
    needs_subset = int(max_total) > 0 or int(per_type) > 0 or bool(keep) or bool(drop)
    if not needs_subset:
        return PreparedEvalDataset(
            source_dataset=source_dataset,
            effective_dataset=source_dataset,
            dataset_name=dataset_name,
            subset_built=False,
        )

    if str(subset_output or "").strip():
        effective_dataset = str(resolve_project_path(str(subset_output).strip()))
    else:
        subset_dir = resolve_project_path(
            str(config["evaluation"].get("thesis_subset_dir", "data/raw/LongMemEval/thesis_subsets"))
        )
        subset_dir.mkdir(parents=True, exist_ok=True)
        source_stem = sanitize_filename_part(Path(source_dataset).stem)
        keep_tag = f"__keep-{sanitize_filename_part('-'.join(sorted(keep)))}" if keep else ""
        drop_tag = f"__drop-{sanitize_filename_part('-'.join(sorted(drop)))}" if drop else ""
        subset_name = (
            f"{source_stem}__max{int(max_total)}__per{int(per_type)}"
            f"__seed{int(seed)}{keep_tag}{drop_tag}.json"
        )
        effective_dataset = str(subset_dir / subset_name)

    build_subset(
        source_path=source_dataset,
        output_path=effective_dataset,
        max_total=int(max_total),
        per_type=int(per_type),
        seed=int(seed),
        keep_types=list(keep),
        drop_types=list(drop),
    )
    return PreparedEvalDataset(
        source_dataset=source_dataset,
        effective_dataset=effective_dataset,
        dataset_name=dataset_name,
        subset_built=True,
    )


def register_mode_run(
    *,
    config: dict,
    dataset_name: str,
    mode: str,
    run_id: str,
    model_name: str,
    judge_model: str = "",
) -> None:
    """Register one exported run inside the shared thesis-mode table."""
    db_file = resolve_project_path(str(config["evaluation"]["database_file"]))
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        store = EvalStore(conn=conn, eval_cfg=dict(config["evaluation"]))
        store.create_tables()
        store.ensure_schema_compat()
        store.log_thesis_mode_run(
            dataset_name=dataset_name,
            mode=mode,
            run_id=run_id,
            model_name=model_name,
            judge_model=judge_model,
            commit=True,
        )
    finally:
        conn.close()


def load_runtime_config(path: str) -> dict:
    """Thin wrapper so runners all use the same config loader import site."""
    return load_config(path)
