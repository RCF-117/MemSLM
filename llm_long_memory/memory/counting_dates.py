"""Date and temporal utilities for counting resolver."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Sequence


def parse_date_token(token: str) -> Optional[datetime]:
    """Parse a single date token into datetime when possible."""
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


def extract_dates(text: str, date_patterns: Sequence[re.Pattern[str]]) -> List[datetime]:
    """Extract absolute dates from text by configured regex patterns."""
    out: List[datetime] = []
    for pat in date_patterns:
        for m in pat.findall(text):
            token = str(m).strip()
            if not token:
                continue
            dt = parse_date_token(token)
            if dt is not None:
                out.append(dt)
    return out


def parse_session_date(session_date: str) -> Optional[datetime]:
    """Parse session-level date used for relative date resolution."""
    token = str(session_date).strip().split(" ")[0]
    if not token:
        return None
    for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(token, fmt)
        except ValueError:
            continue
    return None


def relative_weekday_date(reference: datetime, weekday: int, is_last: bool) -> datetime:
    """Resolve relative weekday against a reference date."""
    delta = (reference.weekday() - weekday) % 7
    if is_last:
        if delta == 0:
            delta = 7
        return reference - timedelta(days=delta)
    return reference - timedelta(days=delta)


def extract_relative_dates(text: str, session_date: str) -> List[datetime]:
    """Extract relative date mentions like today/yesterday/last monday."""
    out: List[datetime] = []
    ref = parse_session_date(session_date)
    if ref is None:
        return out
    lowered = str(text).lower()
    if "today" in lowered:
        out.append(ref)
    if "yesterday" in lowered:
        out.append(ref - timedelta(days=1))

    weekday_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    for name, idx in weekday_map.items():
        if re.search(rf"\blast\s+{name}\b", lowered):
            out.append(relative_weekday_date(ref, idx, is_last=True))
        elif re.search(rf"\bthis\s+{name}\b", lowered):
            out.append(relative_weekday_date(ref, idx, is_last=False))
    return out


def unique_dates(values: Iterable[datetime]) -> List[datetime]:
    """Deduplicate dates at day granularity."""
    uniq = {}
    for d in values:
        key = d.strftime("%Y-%m-%d")
        if key not in uniq:
            uniq[key] = d
    return list(uniq.values())


def extract_dates_with_context(
    text: str,
    session_date: str,
    date_patterns: Sequence[re.Pattern[str]],
) -> List[datetime]:
    """Extract absolute + relative dates and deduplicate."""
    dates = extract_dates(text, date_patterns)
    dates.extend(extract_relative_dates(text=text, session_date=session_date))
    return unique_dates(dates)


def anchor_tokens(text: str) -> List[str]:
    """Tokenize anchor text while dropping generic stop words."""
    stop = {
        "the",
        "a",
        "an",
        "of",
        "to",
        "for",
        "with",
        "and",
        "at",
        "in",
        "on",
        "my",
        "me",
        "i",
        "did",
        "it",
        "take",
        "days",
        "day",
        "after",
        "before",
        "between",
        "had",
        "passed",
        "was",
        "were",
    }
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return [t for t in tokens if t and t not in stop]


def anchor_match_score(anchor: str, text: str) -> float:
    """Compute overlap ratio between anchor tokens and sentence tokens."""
    at = set(anchor_tokens(anchor))
    if not at:
        return 0.0
    tt = set(anchor_tokens(text))
    if not tt:
        return 0.0
    return float(len(at.intersection(tt))) / float(len(at))
