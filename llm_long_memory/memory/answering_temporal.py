"""Temporal-choice decision helpers for answering pipeline."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable, Dict, List, Optional


TokenizeFn = Callable[[str], List[str]]


def extract_quoted_options(query: str) -> List[str]:
    """Extract quoted options like 'A' or \"B\" from query."""
    return [
        x.strip()
        for x in re.findall(r"'([^']{2,120})'|\"([^\"]{2,120})\"", query)
        for x in x
        if x.strip()
    ]


def extract_or_options(query: str) -> List[str]:
    """Extract basic A-or-B options from query tail."""
    match = re.search(
        r"(?:the\s+)?([a-z0-9][a-z0-9\\s\\-]{1,50}?)\\s+or\\s+(?:the\\s+)?([a-z0-9][a-z0-9\\s\\-]{1,50}?)\\??$",
        query.strip().lower(),
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    left = " ".join(match.group(1).split()).strip(" .,;:!?")
    right = " ".join(match.group(2).split()).strip(" .,;:!?")
    if not left or not right:
        return []
    if len(left.split()) > 8 or len(right.split()) > 8:
        return []
    return [left, right]


def parse_date_token(token: str) -> Optional[datetime]:
    """Parse common date tokens into datetime."""
    clean = token.strip().lower().replace(",", "")
    clean = re.sub(r"(\\d)(st|nd|rd|th)\\b", r"\\1", clean)
    formats = [
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m/%d",
        "%b %d %Y",
        "%b %d",
        "%B %d %Y",
        "%B %d",
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(clean, fmt)
            if fmt in {"%m/%d", "%b %d", "%B %d"}:
                parsed = parsed.replace(year=2000)
            return parsed
        except ValueError:
            continue
    return None


def extract_dates_from_text(text: str, time_patterns: List[re.Pattern[str]]) -> List[datetime]:
    """Extract date tokens from text and parse into datetime."""
    out: List[datetime] = []
    for pattern in time_patterns:
        for token in pattern.findall(text):
            parsed = parse_date_token(str(token))
            if parsed is not None:
                out.append(parsed)
    return out


def choose_temporal_option(
    *,
    query: str,
    evidence_sentences: List[Dict[str, object]],
    enabled: bool,
    min_confidence_gap: float,
    require_both_options: bool,
    time_patterns: List[re.Pattern[str]],
) -> Optional[Dict[str, str]]:
    """Choose between two options when query asks temporal comparison."""
    if not enabled:
        return None
    query_lower = query.lower()
    if (" first" not in query_lower) and (" earlier" not in query_lower) and (" before " not in query_lower):
        if (" last" not in query_lower) and (" later" not in query_lower) and (" after " not in query_lower):
            return None
    prefer_earliest = (" first" in query_lower) or (" earlier" in query_lower) or (" before " in query_lower)
    prefer_latest = (" last" in query_lower) or (" later" in query_lower) or (" after " in query_lower)
    options = extract_quoted_options(query)
    if len(options) < 2:
        options = extract_or_options(query)
    if len(options) < 2:
        return None
    left, right = options[0], options[1]

    left_hits = 0.0
    right_hits = 0.0
    left_dates: List[datetime] = []
    right_dates: List[datetime] = []
    left_mentions = 0
    right_mentions = 0
    for item in evidence_sentences:
        text = str(item.get("text", ""))
        score = float(item.get("score", 0.0))
        low = text.lower()
        if left.lower() in low:
            left_hits += score
            left_mentions += 1
            left_dates.extend(extract_dates_from_text(text, time_patterns))
        if right.lower() in low:
            right_hits += score
            right_mentions += 1
            right_dates.extend(extract_dates_from_text(text, time_patterns))

    if require_both_options and (left_mentions == 0 or right_mentions == 0):
        return None
    if left_hits <= 0.0 and right_hits <= 0.0:
        return None

    if left_dates and right_dates:
        left_anchor = min(left_dates) if prefer_earliest or (not prefer_latest) else max(left_dates)
        right_anchor = min(right_dates) if prefer_earliest or (not prefer_latest) else max(right_dates)
        if left_anchor != right_anchor:
            if prefer_latest:
                answer = left if left_anchor > right_anchor else right
            else:
                answer = left if left_anchor < right_anchor else right
            return {"answer": answer, "reason": "temporal_choice_by_date"}

    gap = abs(left_hits - right_hits)
    if gap < min_confidence_gap:
        return None
    answer = left if left_hits > right_hits else right
    return {"answer": answer, "reason": "temporal_choice_by_score"}
