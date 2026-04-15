"""Utility helpers for MemoryManager retrieval and prompt-context plumbing."""

from __future__ import annotations

from typing import Dict, List, Set

from llm_long_memory.memory.answering_temporal import extract_choice_options, parse_choice_query


def dedup_chunks_keep_best(chunks: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Deduplicate chunks by chunk_id/text and keep best score."""
    by_key: Dict[str, Dict[str, object]] = {}
    for item in chunks:
        cid = item.get("chunk_id")
        text = str(item.get("text", "")).strip().lower()
        key = f"id:{cid}" if cid is not None else f"text:{text}"
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = dict(item)
            continue
        if float(item.get("score", 0.0)) > float(prev.get("score", 0.0)):
            by_key[key] = dict(item)
    out = list(by_key.values())
    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out


def is_temporal_query(query: str, cue_keywords: Set[str]) -> bool:
    """Simple temporal-cue classifier for retrieval branching."""
    text = str(query).strip().lower()
    if not text:
        return False
    if not cue_keywords:
        return False
    return any(k in text for k in cue_keywords)


def build_temporal_anchor_queries(
    *,
    query: str,
    temporal_anchor_enabled: bool,
    temporal_anchor_require_temporal_cue: bool,
    temporal_anchor_cue_keywords: Set[str],
    temporal_anchor_max_options: int,
    temporal_anchor_extra_queries_per_option: int,
) -> List[str]:
    """Build extra retrieval queries for temporal choice questions."""
    if not temporal_anchor_enabled:
        return []
    if temporal_anchor_require_temporal_cue and (
        not is_temporal_query(query, temporal_anchor_cue_keywords)
    ):
        return []
    parsed = parse_choice_query(
        query,
        max_options=max(2, temporal_anchor_max_options),
        default_target_k=temporal_anchor_max_options,
    )
    options = parsed if parsed is not None else extract_choice_options(
        query, max_options=max(2, temporal_anchor_max_options)
    )
    out: List[str] = []
    seen: Set[str] = set()
    for opt in options[: max(1, temporal_anchor_max_options)]:
        normalized = " ".join(str(opt).split())
        if not normalized:
            continue
        q1 = normalized
        q2 = f"{query} {normalized}"
        for candidate in (q1, q2):
            c = " ".join(candidate.split()).strip()
            if (not c) or c.lower() in seen:
                continue
            seen.add(c.lower())
            out.append(c)
            if len(out) >= max(
                1,
                temporal_anchor_max_options * max(1, temporal_anchor_extra_queries_per_option),
            ):
                return out
    return out


def merge_anchor_chunks(
    *,
    base_chunks: List[Dict[str, object]],
    extra_chunks: List[Dict[str, object]],
    additive_limit: int,
) -> List[Dict[str, object]]:
    """Merge additional anchor chunks and deduplicate globally."""
    if additive_limit <= 0 or not extra_chunks:
        return list(base_chunks)
    existing_ids = {int(x["chunk_id"]) for x in base_chunks if x.get("chunk_id") is not None}
    merged = list(base_chunks)
    added = 0
    for item in extra_chunks:
        cid = item.get("chunk_id")
        if cid is not None and int(cid) in existing_ids:
            continue
        merged.append(dict(item))
        if cid is not None:
            existing_ids.add(int(cid))
        added += 1
        if added >= additive_limit:
            break
    return dedup_chunks_keep_best(merged)

