"""Prompt context assembly utilities for MemoryManager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PromptContextLimits:
    """Limits for prompt context construction."""

    max_chunks: int
    max_chars_per_chunk: int
    max_total_chars: int


def _norm_text_key(text: str) -> str:
    """Normalize text for robust deduplication."""
    return " ".join(str(text).lower().split())


def build_generation_context(
    *,
    reranked_chunks: List[Dict[str, object]],
    evidence_sentences: List[Dict[str, object]],
    fallback_context: str,
    limits: PromptContextLimits,
) -> str:
    """Build compact high-signal prompt context from reranked chunks."""
    if limits.max_chunks <= 0 or limits.max_total_chars <= 0:
        return fallback_context

    chunk_map: Dict[int, Dict[str, object]] = {}
    for item in reranked_chunks:
        raw_id = item.get("chunk_id")
        if raw_id is None:
            continue
        chunk_id = int(raw_id)
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = item

    selected_ids: List[int] = []
    for evidence in evidence_sentences:
        raw_id = evidence.get("chunk_id")
        if raw_id is None:
            continue
        chunk_id = int(raw_id)
        if chunk_id in chunk_map and chunk_id not in selected_ids:
            selected_ids.append(chunk_id)
        if len(selected_ids) >= limits.max_chunks:
            break

    if len(selected_ids) < limits.max_chunks:
        for item in sorted(
            reranked_chunks, key=lambda x: float(x.get("score", 0.0)), reverse=True
        ):
            raw_id = item.get("chunk_id")
            if raw_id is None:
                continue
            chunk_id = int(raw_id)
            if chunk_id in selected_ids:
                continue
            selected_ids.append(chunk_id)
            if len(selected_ids) >= limits.max_chunks:
                break

    parts: List[str] = []
    seen_text: set[str] = set()
    total_chars = 0
    for chunk_id in selected_ids:
        item = chunk_map.get(chunk_id)
        if item is None:
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        key = _norm_text_key(text)
        if key in seen_text:
            continue
        seen_text.add(key)
        clipped = text[: limits.max_chars_per_chunk].strip()
        if not clipped:
            continue
        topic_id = str(item.get("topic_id", "")).strip()
        block = f"[Topic: {topic_id}]\n{clipped}" if topic_id else clipped
        projected = total_chars + len(block) + (2 if parts else 0)
        if projected > limits.max_total_chars:
            break
        parts.append(block)
        total_chars = projected

    if parts:
        return "\n\n".join(parts)
    return fallback_context
