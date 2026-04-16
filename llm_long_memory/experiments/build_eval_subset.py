"""Build small, balanced evaluation subsets for thesis experiments."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from llm_long_memory.evaluation.dataset_loader import load_stream
from llm_long_memory.utils.helpers import resolve_project_path


EvalInstance = Dict[str, Any]


def _parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _filter_instances(
    instances: Sequence[EvalInstance],
    *,
    keep_types: Sequence[str],
    drop_types: Sequence[str],
) -> List[Tuple[int, EvalInstance]]:
    keep = {str(x).strip().lower() for x in keep_types if str(x).strip()}
    drop = {str(x).strip().lower() for x in drop_types if str(x).strip()}
    grouped: List[Tuple[int, EvalInstance]] = []
    for idx, inst in enumerate(instances):
        qtype = str(inst.get("question_type", "")).strip().lower()
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


def build_subset(
    source_path: str,
    output_path: str,
    *,
    max_total: int,
    per_type: int,
    seed: int,
    keep_types: Sequence[str],
    drop_types: Sequence[str],
) -> Path:
    instances = list(load_stream(str(source_path)))
    grouped = _filter_instances(instances, keep_types=keep_types, drop_types=drop_types)
    if per_type > 0:
        subset = _balanced_subset(grouped, per_type=per_type, max_total=max_total, seed=seed)
    else:
        subset = _head_subset(grouped, max_total=max_total)

    out_path = resolve_project_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")

    counts: Dict[str, int] = defaultdict(int)
    for inst in subset:
        counts[str(inst.get("question_type", "")).strip() or "unknown"] += 1

    print(f"source={source_path}")
    print(f"output={out_path}")
    print(f"total={len(subset)}")
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
    )


if __name__ == "__main__":
    main()
