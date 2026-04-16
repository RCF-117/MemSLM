"""Build stratified debug/test splits for thesis experiments."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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


def _stratified_split(
    grouped: Sequence[Tuple[int, EvalInstance]],
    *,
    debug_ratio: float,
    seed: int,
) -> tuple[List[EvalInstance], List[EvalInstance]]:
    rng = random.Random(seed)
    by_type: Dict[str, List[Tuple[int, EvalInstance]]] = defaultdict(list)
    for idx, inst in grouped:
        qtype = str(inst.get("question_type", "")).strip().lower() or "unknown"
        by_type[qtype].append((idx, inst))

    debug_items: List[Tuple[int, EvalInstance]] = []
    test_items: List[Tuple[int, EvalInstance]] = []
    ratio = max(0.0, min(1.0, float(debug_ratio)))

    for qtype in sorted(by_type.keys()):
        bucket = list(by_type[qtype])
        rng.shuffle(bucket)
        if len(bucket) == 1:
            debug_count = 1 if ratio >= 0.5 else 0
        else:
            debug_count = int(round(len(bucket) * ratio))
            debug_count = max(1 if ratio > 0 else 0, min(len(bucket) - 1, debug_count))
        debug_bucket = bucket[:debug_count]
        test_bucket = bucket[debug_count:]
        debug_items.extend(debug_bucket)
        test_items.extend(test_bucket)

    debug_items.sort(key=lambda item: item[0])
    test_items.sort(key=lambda item: item[0])
    return [inst for _, inst in debug_items], [inst for _, inst in test_items]


def build_split(
    source_path: str,
    *,
    debug_output: str,
    test_output: str,
    manifest_output: str | None,
    debug_ratio: float,
    seed: int,
    keep_types: Sequence[str],
    drop_types: Sequence[str],
) -> tuple[Path, Path]:
    instances = list(load_stream(str(source_path)))
    grouped = _filter_instances(instances, keep_types=keep_types, drop_types=drop_types)
    debug_items, test_items = _stratified_split(grouped, debug_ratio=debug_ratio, seed=seed)

    debug_path = resolve_project_path(debug_output)
    test_path = resolve_project_path(test_output)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(json.dumps(debug_items, ensure_ascii=False, indent=2), encoding="utf-8")
    test_path.write_text(json.dumps(test_items, ensure_ascii=False, indent=2), encoding="utf-8")

    def _print_counts(label: str, items: Sequence[EvalInstance]) -> None:
        counts: Dict[str, int] = defaultdict(int)
        for inst in items:
            counts[str(inst.get("question_type", "")).strip() or "unknown"] += 1
        print(f"[{label}] total={len(items)}")
        for qtype in sorted(counts.keys()):
            print(f"- {qtype}: {counts[qtype]}")

    print(f"source={source_path}")
    print(f"debug_output={debug_path}")
    print(f"test_output={test_path}")
    _print_counts("debug", debug_items)
    _print_counts("test", test_items)

    def _count_types(items: Sequence[EvalInstance]) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for inst in items:
            counts[str(inst.get("question_type", "")).strip() or "unknown"] += 1
        return dict(sorted(counts.items()))

    manifest = {
        "source": str(source_path),
        "debug_output": str(debug_path),
        "test_output": str(test_path),
        "debug_ratio": float(debug_ratio),
        "seed": int(seed),
        "keep_types": list(keep_types),
        "drop_types": list(drop_types),
        "debug_total": len(debug_items),
        "test_total": len(test_items),
        "debug_type_counts": _count_types(debug_items),
        "test_type_counts": _count_types(test_items),
    }
    if manifest_output:
        manifest_path = resolve_project_path(manifest_output)
    else:
        manifest_path = debug_path.parent / "split_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest={manifest_path}")
    return debug_path, test_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stratified debug/test eval splits.")
    parser.add_argument("--source", required=True, help="Source dataset path.")
    parser.add_argument("--debug-output", required=True, help="Debug split JSON path.")
    parser.add_argument("--test-output", required=True, help="Test split JSON path.")
    parser.add_argument(
        "--manifest-output",
        default="",
        help="Optional manifest JSON path. Defaults to sibling split_manifest.json.",
    )
    parser.add_argument(
        "--debug-ratio",
        type=float,
        default=0.3,
        help="Fraction of each question type reserved for the debug split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when splitting within each question type.",
    )
    parser.add_argument(
        "--keep-types",
        type=str,
        default="",
        help="Comma-separated question types to keep. Empty keeps all.",
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
    build_split(
        source_path=args.source,
        debug_output=args.debug_output,
        test_output=args.test_output,
        manifest_output=(args.manifest_output.strip() or None),
        debug_ratio=float(args.debug_ratio),
        seed=int(args.seed),
        keep_types=_parse_csv(args.keep_types),
        drop_types=_parse_csv(args.drop_types),
    )


if __name__ == "__main__":
    main()
