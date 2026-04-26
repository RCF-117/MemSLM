"""Helpers for attaching answer-source audit summaries to thesis reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Set

from llm_long_memory.utils.helpers import dataset_display_name, dataset_name_aliases


AUDIT_SUMMARY_KEYS = [
    "quality_high",
    "quality_medium",
    "quality_low",
    "avg_quality_score",
    "avg_noisy_ratio",
    "avg_long_plan_ratio",
    "avg_best_f1",
    "avg_best_rec",
    "coverage_f1_pos",
    "coverage_rec50",
]


def _dataset_aliases(dataset_name: str | None) -> Set[str]:
    aliases = set(dataset_name_aliases(dataset_name))
    display = dataset_display_name(dataset_name)
    aliases.update(dataset_name_aliases(display))
    return aliases


def _payload_dataset_aliases(payload: Dict[str, Any]) -> Set[str]:
    aliases: Set[str] = set()
    for value in (
        str(payload.get("dataset", "")).strip(),
        str(payload.get("source_json", "")).strip(),
    ):
        if not value:
            continue
        path = Path(value)
        aliases.update({value.lower(), path.name.lower(), path.stem.lower()})
        aliases.update(dataset_name_aliases(dataset_display_name(value)))
    return {x for x in aliases if x}


def load_latest_source_audit_summary(
    report_dir: str | Path,
    dataset_name: str | None = None,
) -> Dict[str, Any]:
    root = Path(report_dir)
    if not root.exists():
        return {}
    wanted = _dataset_aliases(dataset_name)
    candidates = sorted(
        root.glob("answer_source_audit*__summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = dict(payload.get("metrics", {}) or {})
        if not metrics:
            continue
        if wanted and not (wanted & _payload_dataset_aliases(payload)):
            continue
        return {
            "file": path.name,
            "dataset": str(payload.get("dataset", "")).strip(),
            "source_json": str(payload.get("source_json", "")).strip(),
            "metrics": {key: metrics.get(key) for key in AUDIT_SUMMARY_KEYS if key in metrics},
        }
    return {}


def iter_audit_summary_lines(summary: Dict[str, Any]) -> Iterable[str]:
    metrics = dict(summary.get("metrics", {}) or {})
    if not metrics:
        return []
    lines = [
        "## Answer Source Audit Summary",
        "",
        f"- audit_summary_file: `{summary.get('file', '')}`",
        f"- audit_dataset: `{summary.get('dataset', '')}`",
    ]
    for key in AUDIT_SUMMARY_KEYS:
        if key not in metrics:
            continue
        value = metrics.get(key)
        if isinstance(value, int):
            lines.append(f"- {key}: `{value}`")
        elif isinstance(value, float):
            lines.append(f"- {key}: `{value:.4f}`")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    return lines
