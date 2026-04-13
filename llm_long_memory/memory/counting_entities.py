"""Entity and lexical utilities for counting resolver."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Set


def normalize_text(text: str) -> str:
    """Collapse spaces and trim."""
    return " ".join(str(text).split()).strip()


def quoted_phrases(text: str) -> List[str]:
    """Extract short quoted phrases from text."""
    return [
        x.strip()
        for x in re.findall(r"'([^']{1,80})'|\"([^\"]{1,80})\"", text)
        for x in x
        if x.strip()
    ]


def normalize_entity(
    text: str,
    max_entity_tokens: int,
    stopwords: Set[str],
) -> str:
    """Normalize one candidate entity phrase."""
    lowered = normalize_text(text).lower()
    lowered = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", lowered)
    if not lowered:
        return ""
    tokens = [t for t in re.findall(r"[a-z0-9]+", lowered) if t]
    if not tokens:
        return ""
    if len(tokens) > max_entity_tokens:
        return ""
    if all(t in stopwords for t in tokens):
        return ""
    if any(re.fullmatch(r"\d+", t) for t in tokens):
        return ""
    return " ".join(tokens)


def extract_list_entities(
    text: str,
    max_entity_tokens: int,
    stopwords: Set[str],
) -> List[str]:
    """Extract comma/and lists and quoted entities."""
    entities: List[str] = []
    for q in quoted_phrases(text):
        norm = normalize_entity(q, max_entity_tokens=max_entity_tokens, stopwords=stopwords)
        if norm:
            entities.append(norm)

    normalized_text = normalize_text(text)
    if "," in normalized_text or " and " in normalized_text.lower():
        tmp = re.sub(r"\band\b", ",", normalized_text, flags=re.IGNORECASE)
        for part in [x.strip() for x in tmp.split(",")]:
            norm = normalize_entity(part, max_entity_tokens=max_entity_tokens, stopwords=stopwords)
            if norm:
                entities.append(norm)
    return entities


def extract_query_focus_tokens(query: str, focus_stopwords: Set[str]) -> List[str]:
    """Extract coarse focus terms from count questions to filter noisy evidence."""
    lowered = normalize_text(query).lower()
    fragments: List[str] = []
    patterns = [
        r"how many\s+(.+?)(?:\?|$)",
        r"number of\s+(.+?)(?:\?|$)",
        r"count of\s+(.+?)(?:\?|$)",
    ]
    for pat in patterns:
        m = re.search(pat, lowered, flags=re.IGNORECASE)
        if m:
            fragments.append(str(m.group(1)))
    if not fragments:
        fragments = [lowered]
    tokens: List[str] = []
    for frag in fragments:
        for tok in re.findall(r"[a-z0-9]+", frag):
            if tok in focus_stopwords:
                continue
            tokens.append(tok)
    uniq: List[str] = []
    for tok in tokens:
        if tok not in uniq:
            uniq.append(tok)
    return uniq


def sentence_focus_overlap(
    text: str,
    focus_tokens: Sequence[str],
    irregular_forms: Dict[str, List[str]],
) -> int:
    """Count focus-token overlap with light morphology and irregular forms."""
    if not focus_tokens:
        return 0
    raw_tokens = set(re.findall(r"[a-z0-9]+", str(text).lower()))
    expanded: Set[str] = set(raw_tokens)
    for tok in list(raw_tokens):
        if tok.endswith("ed") and len(tok) > 3:
            expanded.add(tok[:-2])
        if tok.endswith("ing") and len(tok) > 4:
            expanded.add(tok[:-3])
        if tok.endswith("s") and len(tok) > 2:
            expanded.add(tok[:-1])
    for base, forms in irregular_forms.items():
        if any(form in raw_tokens for form in forms):
            expanded.add(base)
    return sum(1 for tok in focus_tokens if tok in expanded)
