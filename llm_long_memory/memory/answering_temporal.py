"""Temporal-choice decision helpers for answering pipeline."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple


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


def extract_list_options(query: str, max_options: int) -> List[str]:
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


def extract_choice_options(query: str, max_options: int) -> List[str]:
    """Extract options for 2-way / N-way choices from query text."""
    quoted = extract_quoted_options(query)
    if len(quoted) >= 2:
        normalized: List[str] = []
        for option in quoted:
            s = " ".join(str(option).lower().split())
            if s and s not in normalized:
                normalized.append(s)
            if len(normalized) >= int(max_options):
                break
        return normalized

    pair = extract_or_options(query)
    if len(pair) >= 2:
        return pair[: int(max_options)]

    listed = extract_list_options(query, max_options=max_options)
    if len(listed) >= 2:
        return listed
    return []


def infer_selection_target_k(query: str, option_count: int, default_target_k: int) -> int:
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


def parse_choice_query(
    query: str,
    *,
    max_options: int,
    default_target_k: int,
) -> Optional[Tuple[List[str], int]]:
    """Parse query into option list + selection target for choice-style questions."""
    options = extract_choice_options(query, max_options=max_options)
    if len(options) < 2:
        return None
    target_k = infer_selection_target_k(
        query,
        option_count=len(options),
        default_target_k=default_target_k,
    )
    return options, target_k


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _overlap(anchor: str, text: str) -> float:
    a = set(_tokenize(anchor))
    t = set(_tokenize(text))
    if not a or not t:
        return 0.0
    return float(len(a.intersection(t))) / float(len(a))


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


def parse_session_date(text: str) -> Optional[datetime]:
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


def choose_temporal_option(
    *,
    query: str,
    evidence_sentences: List[Dict[str, object]],
    enabled: bool,
    min_confidence_gap: float,
    require_both_options: bool,
    time_patterns: List[re.Pattern[str]],
    overlap_floor: float,
    contains_bonus: float,
    date_bonus: float,
    event_anchor_enabled: bool,
    event_anchor_min_overlap: float,
    event_anchor_min_score: float,
    event_anchor_pair_min_score: float,
    event_anchor_use_session_date_fallback: bool,
    event_anchor_fallback_to_sentence_score: bool,
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
    left_lower = left.lower()
    right_lower = right.lower()

    if event_anchor_enabled:
        left_anchors: List[Dict[str, object]] = []
        right_anchors: List[Dict[str, object]] = []
        for item in evidence_sentences:
            text = str(item.get("text", ""))
            score = float(item.get("score", 0.0))
            session_date = str(item.get("session_date", ""))
            low = text.lower()
            for option_lower, bucket in (
                (left_lower, left_anchors),
                (right_lower, right_anchors),
            ):
                overlap = _overlap(option_lower, low)
                contains = option_lower in low
                if (not contains) and overlap < event_anchor_min_overlap:
                    continue
                mention_score = score * max(overlap_floor, overlap)
                if contains:
                    mention_score += contains_bonus * score
                if mention_score < event_anchor_min_score:
                    continue
                explicit_dates = extract_dates_from_text(text, time_patterns)
                dates = list(explicit_dates)
                if (not dates) and event_anchor_use_session_date_fallback:
                    session_dt = parse_session_date(session_date)
                    if session_dt is not None:
                        dates = [session_dt]
                for date_value in dates:
                    bucket.append(
                        {
                            "date": date_value,
                            "score": mention_score,
                            "explicit": bool(explicit_dates),
                        }
                    )

        if left_anchors and right_anchors:
            best_pair: Optional[Dict[str, object]] = None
            for left_anchor in left_anchors:
                left_date = left_anchor["date"]
                if not isinstance(left_date, datetime):
                    continue
                for right_anchor in right_anchors:
                    right_date = right_anchor["date"]
                    if not isinstance(right_date, datetime):
                        continue
                    if left_date == right_date:
                        continue
                    if prefer_latest:
                        answer = left if left_date > right_date else right
                    else:
                        answer = left if left_date < right_date else right
                    pair_score = float(left_anchor["score"]) + float(right_anchor["score"])
                    if bool(left_anchor.get("explicit")):
                        pair_score += date_bonus
                    if bool(right_anchor.get("explicit")):
                        pair_score += date_bonus
                    if best_pair is None or pair_score > float(best_pair["pair_score"]):
                        best_pair = {"answer": answer, "pair_score": pair_score}
            if best_pair is not None and float(best_pair["pair_score"]) >= event_anchor_pair_min_score:
                return {"answer": str(best_pair["answer"]), "reason": "temporal_choice_by_event_anchor"}
            if require_both_options and (not event_anchor_fallback_to_sentence_score):
                return None
        elif require_both_options and (not event_anchor_fallback_to_sentence_score):
            return None

    left_hits = 0.0
    right_hits = 0.0
    left_max = 0.0
    right_max = 0.0
    left_dates: List[datetime] = []
    right_dates: List[datetime] = []
    left_mentions = 0
    right_mentions = 0
    for item in evidence_sentences:
        text = str(item.get("text", ""))
        score = float(item.get("score", 0.0))
        low = text.lower()
        left_overlap = _overlap(left_lower, low)
        right_overlap = _overlap(right_lower, low)
        left_contains = left_lower in low
        right_contains = right_lower in low
        if left_contains or left_overlap > 0.0:
            base = score * max(overlap_floor, left_overlap)
            if left_contains:
                base += contains_bonus * score
            left_mentions += 1
            left_hits += base
            left_max = max(left_max, base)
            left_dates.extend(extract_dates_from_text(text, time_patterns))
        if right_contains or right_overlap > 0.0:
            base = score * max(overlap_floor, right_overlap)
            if right_contains:
                base += contains_bonus * score
            right_mentions += 1
            right_hits += base
            right_max = max(right_max, base)
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

    left_score = left_hits + (date_bonus * float(len(left_dates))) + (0.2 * left_max)
    right_score = right_hits + (date_bonus * float(len(right_dates))) + (0.2 * right_max)
    gap = abs(left_score - right_score)
    if gap < min_confidence_gap:
        return None
    answer = left if left_score > right_score else right
    return {"answer": answer, "reason": "temporal_choice_by_score"}
