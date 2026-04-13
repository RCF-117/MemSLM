"""Anchor sentence scoring and selection utilities for offline graph building."""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Set


def sentence_feature_score(text: str) -> float:
    """Heuristic sentence score emphasizing time/fact anchors."""
    value = str(text).strip()
    if not value:
        return 0.0
    s = value.lower()
    score = 0.0
    if re.search(r"\b\d{1,4}\b", s):
        score += 1.2
    if re.search(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|am|pm|today|yesterday|tomorrow)\b",
        s,
    ):
        score += 1.0
    if re.search(r"\b(bought|moved|met|scheduled|booked|paid|started|finished|visited)\b", s):
        score += 1.0
    if re.search(r"\b(at|in|on)\s+[A-Z][a-z]+", value):
        score += 0.8
    if len(value) >= 45:
        score += 0.4
    return score


def select_anchor_sentences(
    *,
    text: str,
    split_sentences_fn: Callable[[str], List[str]],
    tokenize_fn: Callable[[str], List[str]],
    fact_keywords: Set[str],
    hint_keywords: Set[str],
    min_sentence_chars: int,
    min_score: float,
    top_k: int,
    max_chars: int,
) -> List[str]:
    """Select high-anchor sentences from one chunk for event extraction."""
    sentences = split_sentences_fn(text)
    scored: List[Dict[str, object]] = []
    for idx, sent in enumerate(sentences):
        s = str(sent).strip()
        if len(s) < int(min_sentence_chars):
            continue
        tokens = set(tokenize_fn(s))
        event_overlap = len(tokens.intersection(fact_keywords))
        hint_overlap = len(tokens.intersection(hint_keywords))
        score = float(sentence_feature_score(s))
        if event_overlap > 0:
            score += 1.0
        if hint_overlap > 0:
            score += 0.5
        if score < float(min_score):
            continue
        scored.append({"idx": idx, "text": s, "score": score})

    if not scored:
        return []

    scored.sort(key=lambda x: float(x["score"]), reverse=True)
    picked = scored[: max(1, int(top_k))]
    picked.sort(key=lambda x: int(x["idx"]))
    limit = max(32, int(max_chars))
    return [str(x["text"])[:limit] for x in picked]

