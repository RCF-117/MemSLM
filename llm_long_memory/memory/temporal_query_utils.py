"""Temporal-choice parsing helpers for answering pipeline."""

from __future__ import annotations

import re
from datetime import datetime
from typing import List


def extract_quoted_choice_candidates(query: str) -> List[str]:
    """Extract quoted options like 'A' or \"B\" from query."""
    return [
        x.strip()
        for x in re.findall(r"'([^']{2,120})'|\"([^\"]{2,120})\"", query)
        for x in x
        if x.strip()
    ]


def extract_binary_choice_candidates(query: str) -> List[str]:
    """Extract A/B options using the last 'or' in query."""
    text = " ".join(str(query).split()).strip(" .?!")
    lower = text.lower()
    idx = lower.rfind(" or ")
    if idx <= 0:
        return []
    left_raw = text[:idx].strip(" ,;:!?")
    right_raw = text[idx + 4 :].strip(" ,;:!?")
    if not left_raw or not right_raw:
        return []
    left = re.split(r"[,;:]\s*", left_raw)[-1].strip()
    right = right_raw.strip()
    left = re.sub(r"^(the|a|an)\s+", "", left, flags=re.IGNORECASE)
    right = re.sub(r"^(the|a|an)\s+", "", right, flags=re.IGNORECASE)
    left = " ".join(left.lower().split())
    right = " ".join(right.lower().split())
    if not left or not right:
        return []
    if left == right:
        return []
    if len(left.split()) > 10 or len(right.split()) > 10:
        return []
    return [left, right]


def extract_listed_choice_candidates(query: str, max_options: int) -> List[str]:
    """Extract option list from comma/or separated query segments."""
    text = " ".join(str(query).split()).strip(" .?!")
    if not text:
        return []
    if " or " not in text.lower():
        return []
    segments = re.split(r"\s*,\s*|\s+or\s+", text, flags=re.IGNORECASE)
    options: List[str] = []
    for seg in segments:
        s = str(seg).strip(" ,;:!?")
        if not s:
            continue
        s = re.sub(
            r"^(which|what|who|did|do|does|is|are|was|were|among|between|choose|select|pick)\s+",
            "",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(r"^(the|a|an)\s+", "", s, flags=re.IGNORECASE)
        s = " ".join(s.lower().split())
        if not s:
            continue
        if len(s.split()) > 12:
            continue
        if s not in options:
            options.append(s)
        if len(options) >= int(max_options):
            break
    return options


def extract_choice_candidates(query: str, max_options: int) -> List[str]:
    """Extract options for 2-way / N-way choices from query text."""
    quoted = extract_quoted_choice_candidates(query)
    if len(quoted) >= 2:
        normalized: List[str] = []
        for option in quoted:
            s = " ".join(str(option).lower().split())
            if s and s not in normalized:
                normalized.append(s)
            if len(normalized) >= int(max_options):
                break
        return normalized

    pair = extract_binary_choice_candidates(query)
    if len(pair) >= 2:
        return pair[: int(max_options)]

    listed = extract_listed_choice_candidates(query, max_options=max_options)
    if len(listed) >= 2:
        return listed
    return []


def infer_choice_target_k(query: str, option_count: int, default_target_k: int) -> int:
    """Infer how many options should be selected in N-way choice questions."""
    if option_count <= 0:
        return 0
    match = re.search(r"\b(?:choose|select|pick|which)\s+(\d+)\b", str(query), flags=re.IGNORECASE)
    if match:
        value = int(match.group(1))
        return max(1, min(option_count, value))
    if re.search(r"\bwhich\s+(?:two|2)\b", str(query), flags=re.IGNORECASE):
        return max(1, min(option_count, 2))
    if re.search(r"\bwhich\s+(?:three|3)\b", str(query), flags=re.IGNORECASE):
        return max(1, min(option_count, 3))
    return max(1, min(option_count, int(default_target_k)))


def parse_choice_targets(
    query: str,
    *,
    max_options: int,
    default_target_k: int,
) -> List[str] | None:
    """Parse query into option list + selection target for choice-style questions."""
    options = extract_choice_candidates(query, max_options=max_options)
    if len(options) < 2:
        return None
    target_k = infer_choice_target_k(
        query,
        option_count=len(options),
        default_target_k=default_target_k,
    )
    return options[: max(1, target_k)]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _overlap(anchor: str, text: str) -> float:
    a = set(_tokenize(anchor))
    t = set(_tokenize(text))
    if not a or not t:
        return 0.0
    return float(len(a.intersection(t))) / float(len(a))


def parse_date_token(token: str) -> datetime | None:
    """Parse common date tokens into datetime."""
    clean = token.strip().lower().replace(",", "")
    clean = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", clean)
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


def parse_session_date(text: str) -> datetime | None:
    """Parse session date like '2023/05/30 (Tue) 21:40' into datetime."""
    token = str(text).strip().split(" ")[0]
    if not token:
        return None
    for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(token, fmt)
        except ValueError:
            continue
    return None
