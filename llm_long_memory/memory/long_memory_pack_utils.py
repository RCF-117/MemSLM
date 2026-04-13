"""Text packing helpers for long-memory offline graph ingestion."""

from __future__ import annotations

from typing import Any, Callable, Dict, List


def build_evidence_packs(
    *,
    texts: List[str],
    split_sentences_fn: Callable[[str], List[str]],
    normalize_space_fn: Callable[[str], str],
    sentence_score_fn: Callable[[str], float],
    min_sentence_chars: int,
    top_sentences_per_chunk: int,
    max_packs: int,
    max_chars: int,
) -> List[str]:
    packs: List[str] = []
    for text in texts:
        sentences = split_sentences_fn(text)
        candidates: List[Dict[str, Any]] = []
        for idx, sent in enumerate(sentences):
            s = normalize_space_fn(sent)
            if len(s) < int(min_sentence_chars):
                continue
            score = float(sentence_score_fn(s))
            candidates.append({"idx": idx, "text": s, "score": score})
        if not candidates:
            continue
        candidates.sort(key=lambda x: float(x["score"]), reverse=True)
        top = candidates[: max(1, int(top_sentences_per_chunk))]
        top_sorted = sorted(top, key=lambda x: int(x["idx"]))
        cur = ""
        for item in top_sorted:
            segment = str(item["text"])
            if not cur:
                cur = segment
                continue
            candidate = f"{cur} {segment}"
            if len(candidate) <= int(max_chars):
                cur = candidate
            else:
                packs.append(cur)
                cur = segment
        if cur:
            packs.append(cur)
    uniq: List[str] = []
    seen = set()
    for p in packs:
        key = normalize_space_fn(p).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq[: max(1, int(max_packs))]

