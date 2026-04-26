"""Build evaluation subsets for thesis experiments.

Supports:
- simple head subsets
- balanced per-type subsets
- exact type-distribution matching against a reference dataset
- exclusion by question id overlap from another dataset
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from llm_long_memory.evaluation.dataset_loader import load_stream
from llm_long_memory.utils.helpers import resolve_project_path


EvalInstance = Dict[str, Any]


def _parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_type_count_spec(value: str | None) -> Dict[str, int]:
    if not value:
        return {}
    out: Dict[str, int] = {}
    for raw in str(value).split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid type-count item '{item}'. Expected format question_type=count."
            )
        key, count = item.split("=", 1)
        qtype = str(key).strip().lower()
        if not qtype:
            raise ValueError(f"Invalid empty question_type in '{item}'.")
        out[qtype] = int(str(count).strip())
    return out


def _filter_instances(
    instances: Sequence[EvalInstance],
    *,
    keep_types: Sequence[str],
    drop_types: Sequence[str],
    exclude_question_ids: Sequence[str],
) -> List[Tuple[int, EvalInstance]]:
    keep = {str(x).strip().lower() for x in keep_types if str(x).strip()}
    drop = {str(x).strip().lower() for x in drop_types if str(x).strip()}
    exclude_ids = {str(x).strip() for x in exclude_question_ids if str(x).strip()}
    grouped: List[Tuple[int, EvalInstance]] = []
    for idx, inst in enumerate(instances):
        qtype = str(inst.get("question_type", "")).strip().lower()
        qid = str(inst.get("question_id", "")).strip()
        if qid and qid in exclude_ids:
            continue
        if keep and qtype not in keep:
            continue
        if drop and qtype in drop:
            continue
        grouped.append((idx, dict(inst)))
    return grouped


def _balanced_subset(
    grouped: Sequence[Tuple[int, EvalInstance]],
    *,
    per_type: int,
    max_total: int,
    seed: int,
) -> List[EvalInstance]:
    rng = random.Random(seed)
    by_type: Dict[str, List[Tuple[int, EvalInstance]]] = defaultdict(list)
    for idx, inst in grouped:
        qtype = str(inst.get("question_type", "")).strip().lower() or "unknown"
        by_type[qtype].append((idx, inst))

    selected: List[Tuple[int, EvalInstance]] = []
    for qtype in sorted(by_type.keys()):
        bucket = list(by_type[qtype])
        rng.shuffle(bucket)
        if per_type > 0:
            selected.extend(bucket[:per_type])
        else:
            selected.extend(bucket)

    if max_total > 0 and len(selected) > max_total:
        selected = selected[:max_total]

    selected.sort(key=lambda item: item[0])
    return [inst for _, inst in selected]


def _exact_type_count_subset(
    grouped: Sequence[Tuple[int, EvalInstance]],
    *,
    target_type_counts: Dict[str, int],
    seed: int,
) -> List[EvalInstance]:
    rng = random.Random(seed)
    by_type: Dict[str, List[Tuple[int, EvalInstance]]] = defaultdict(list)
    for idx, inst in grouped:
        qtype = str(inst.get("question_type", "")).strip().lower() or "unknown"
        by_type[qtype].append((idx, inst))

    selected: List[Tuple[int, EvalInstance]] = []
    for qtype, required in sorted(target_type_counts.items()):
        need = int(required)
        if need <= 0:
            continue
        bucket = list(by_type.get(qtype, []))
        if len(bucket) < need:
            raise ValueError(
                f"Not enough candidates for question_type='{qtype}': need {need}, available {len(bucket)}."
            )
        rng.shuffle(bucket)
        selected.extend(bucket[:need])

    selected.sort(key=lambda item: item[0])
    return [inst for _, inst in selected]


def _head_subset(
    grouped: Sequence[Tuple[int, EvalInstance]],
    *,
    max_total: int,
) -> List[EvalInstance]:
    selected = list(grouped)
    if max_total > 0:
        selected = selected[:max_total]
    selected.sort(key=lambda item: item[0])
    return [inst for _, inst in selected]


def _count_types(instances: Sequence[EvalInstance]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for inst in instances:
        counts[str(inst.get("question_type", "")).strip() or "unknown"] += 1
    return dict(sorted(counts.items()))


def _load_target_type_counts(reference_path: str | None) -> Dict[str, int]:
    if not reference_path:
        return {}
    reference_instances = list(load_stream(str(reference_path)))
    raw_counts = _count_types(reference_instances)
    return {str(k).strip().lower(): int(v) for k, v in raw_counts.items() if int(v) > 0}


def _load_exclude_question_ids(exclude_dataset_path: str | None) -> List[str]:
    if not exclude_dataset_path:
        return []
    out: List[str] = []
    for inst in load_stream(str(exclude_dataset_path)):
        qid = str(inst.get("question_id", "")).strip()
        if qid:
            out.append(qid)
    return out


def build_subset(
    source_path: str,
    output_path: str,
    *,
    max_total: int,
    per_type: int,
    seed: int,
    keep_types: Sequence[str],
    drop_types: Sequence[str],
    reference_distribution_path: str | None = None,
    exclude_dataset_path: str | None = None,
    target_type_counts: Dict[str, int] | None = None,
) -> Path:
    instances = list(load_stream(str(source_path)))
    exclude_question_ids = _load_exclude_question_ids(exclude_dataset_path)
    grouped = _filter_instances(
        instances,
        keep_types=keep_types,
        drop_types=drop_types,
        exclude_question_ids=exclude_question_ids,
    )
    resolved_target_type_counts = {
        str(k).strip().lower(): int(v)
        for k, v in (target_type_counts or {}).items()
        if str(k).strip() and int(v) > 0
    }
    if not resolved_target_type_counts:
        resolved_target_type_counts = _load_target_type_counts(reference_distribution_path)
    if resolved_target_type_counts:
        subset = _exact_type_count_subset(
            grouped,
            target_type_counts=resolved_target_type_counts,
            seed=seed,
        )
    elif per_type > 0:
        subset = _balanced_subset(grouped, per_type=per_type, max_total=max_total, seed=seed)
    else:
        subset = _head_subset(grouped, max_total=max_total)

    out_path = resolve_project_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = _count_types(subset)

    print(f"source={source_path}")
    print(f"output={out_path}")
    print(f"total={len(subset)}")
    if resolved_target_type_counts:
        print(f"target_type_counts={json.dumps(resolved_target_type_counts, ensure_ascii=False)}")
    if reference_distribution_path:
        print(f"reference_distribution={reference_distribution_path}")
    if exclude_dataset_path:
        print(f"exclude_dataset={exclude_dataset_path}")
        print(f"excluded_question_ids={len(exclude_question_ids)}")
    for qtype in sorted(counts.keys()):
        print(f"- {qtype}: {counts[qtype]}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a thesis-friendly eval subset.")
    parser.add_argument("--source", required=True, help="Source dataset path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "--max-total",
        type=int,
        default=20,
        help="Maximum number of instances to keep.",
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=2,
        help="Maximum instances per question type. Set 0 to disable balancing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when balancing types.",
    )
    parser.add_argument(
        "--keep-types",
        type=str,
        default="",
        help="Comma-separated question types to keep. Empty keeps all.",
    )
    parser.add_argument(
        "--include-types",
        type=str,
        default="",
        help="Alias of --keep-types for backward compatibility.",
    )
    parser.add_argument(
        "--drop-types",
        type=str,
        default="",
        help="Comma-separated question types to drop after keep filtering.",
    )
    parser.add_argument(
        "--reference-distribution",
        type=str,
        default="",
        help="Optional dataset path whose question_type distribution should be matched exactly.",
    )
    parser.add_argument(
        "--exclude-dataset",
        type=str,
        default="",
        help="Optional dataset path whose question_ids should be excluded from the source pool.",
    )
    parser.add_argument(
        "--target-type-counts",
        type=str,
        default="",
        help="Optional exact question_type counts, e.g. 'type_a=3,type_b=2'. Overrides --reference-distribution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_types = _parse_csv(args.keep_types) or _parse_csv(args.include_types)
    build_subset(
        source_path=args.source,
        output_path=args.output,
        max_total=int(args.max_total),
        per_type=int(args.per_type),
        seed=int(args.seed),
        keep_types=keep_types,
        drop_types=_parse_csv(args.drop_types),
        reference_distribution_path=(args.reference_distribution.strip() or None),
        exclude_dataset_path=(args.exclude_dataset.strip() or None),
        target_type_counts=_parse_type_count_spec(args.target_type_counts),
    )


if __name__ == "__main__":
    main()
