"""Chunking helpers for MidMemory."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

import numpy as np


def split_long_text(
    text: str,
    message_chunk_max_chars: int,
    message_chunk_min_chars: int,
    message_chunk_overlap_chars: int,
    message_chunk_boundary_window_chars: int,
) -> List[str]:
    """Split long text into overlap chunks with sentence-first strategy."""
    clean = text.strip()
    if not clean:
        return []
    if len(clean) <= message_chunk_max_chars:
        return [clean]

    max_chars = message_chunk_max_chars
    min_chars = message_chunk_min_chars
    overlap = min(message_chunk_overlap_chars, max_chars - 1)
    boundary_window = max(0, int(message_chunk_boundary_window_chars))

    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", clean)
    sentences = [s.strip() for s in sentences if s and s.strip()]
    if not sentences:
        sentences = [clean]

    parts: List[str] = []
    current = ""
    for sent in sentences:
        if not current:
            current = sent
            continue
        if len(current) + 1 + len(sent) <= max_chars:
            current = f"{current} {sent}".strip()
            continue
        parts.append(current.strip())
        current = sent
    if current.strip():
        parts.append(current.strip())

    if overlap > 0 and len(parts) >= 2:
        overlapped: List[str] = [parts[0]]
        for nxt in parts[1:]:
            tail = overlapped[-1][-overlap:].strip()
            if tail and not nxt.startswith(tail):
                overlapped.append(f"{tail} {nxt}".strip())
            else:
                overlapped.append(nxt)
        parts = overlapped

    final_parts: List[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final_parts.append(part)
            continue
        step = max(1, max_chars - overlap)
        start = 0
        while start < len(part):
            end = min(start + max_chars, len(part))
            if end < len(part) and boundary_window > 0:
                low = max(start + 1, end - boundary_window)
                high = min(len(part), end + boundary_window)
                scan = part[low:high]
                rel = max(
                    (
                        idx
                        for idx, ch in enumerate(scan)
                        if ch in {".", "!", "?", "。", "！", "？", "\n", ";", "；"}
                    ),
                    default=-1,
                )
                if rel >= 0:
                    end = low + rel + 1
            piece = part[start:end].strip()
            if piece:
                final_parts.append(piece)
            if end >= len(part):
                break
            start += step

    if len(final_parts) >= 2 and len(final_parts[-1]) < min_chars:
        final_parts[-2] = f"{final_parts[-2]} {final_parts[-1]}".strip()
        final_parts.pop()
    return final_parts


def split_sentences(text: str) -> List[str]:
    """Split text into sentence-like units and keep non-empty spans."""
    clean = str(text or "").strip()
    if not clean:
        return []
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", clean)
    return [s.strip() for s in sentences if s and s.strip()]


def extract_time_terms(text: str, time_regexes: Sequence[re.Pattern[str]]) -> List[str]:
    """Extract normalized time expressions using configured regex patterns."""
    if not text:
        return []
    terms: List[str] = []
    for pattern in time_regexes:
        for match in pattern.findall(text):
            m = str(match).strip().lower()
            if m and m not in terms:
                terms.append(m)
    return terms


def dominant_text(values: Sequence[str]) -> str:
    """Return most frequent non-empty value; ties resolved by first appearance."""
    counts: Dict[str, int] = {}
    first_index: Dict[str, int] = {}
    for idx, value in enumerate(values):
        v = str(value or "").strip()
        if not v:
            continue
        counts[v] = counts.get(v, 0) + 1
        if v not in first_index:
            first_index[v] = idx
    if not counts:
        return ""
    return max(counts, key=lambda x: (counts[x], -first_index[x]))


def dominant_role(roles: Sequence[str]) -> str:
    """Return majority role; ties resolved by first appearance."""
    counts: Dict[str, int] = {}
    first_index: Dict[str, int] = {}
    for idx, role in enumerate(roles):
        counts[role] = counts.get(role, 0) + 1
        if role not in first_index:
            first_index[role] = idx
    return max(counts, key=lambda r: (counts[r], -first_index[r]))


def buffer_centroid_embedding(
    buffer: Sequence[Dict[str, Any]],
    embedding_dim: int,
) -> np.ndarray:
    """Compute mean embedding from current message buffer."""
    if not buffer:
        return np.zeros(embedding_dim, dtype=np.float32)
    vectors = [x["embedding"] for x in buffer]
    return np.mean(np.stack(vectors), axis=0).astype(np.float32)
