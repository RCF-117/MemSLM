"""Utility helpers for MemoryManager retrieval and prompt-context plumbing."""

from __future__ import annotations

import re
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


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _dedup_texts(values: List[str], limit: int) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for raw in values:
        text = _normalize_space(raw).strip(" ,.;:!?\"'")
        if not text:
            continue
        low = text.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(text)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _extract_target_object(query: str) -> str:
    q = _normalize_space(query)
    patterns = [
        r"(?:how many|number of|count(?:\s+of)?)\s+([a-z][a-z0-9\-\s]{1,48}?)(?:\?|$|\b(?:did|do|does|are|is|was|were|have|has|in|on|for|during)\b)",
        r"\b(?:how much)\s+([a-z][a-z0-9\-\s]{1,48}?)(?:\?|$|\b(?:did|do|does|are|is|was|were|have|has|in|on|for|during)\b)",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if not m:
            continue
        val = _normalize_space(str(m.group(1))).strip(" ,.;:!?\"'")
        val = re.sub(
            r"^(?:the|a|an|my|our|your|their|his|her)\s+",
            "",
            val,
            flags=re.IGNORECASE,
        )
        if len(_tokenize(val)) >= 1:
            return val
    return ""


def _extract_count_unit(query: str) -> str:
    q = _normalize_space(query).lower()
    m = re.search(r"\bhow many\s+([a-z][a-z0-9\-]{1,24})\b", q)
    if not m:
        return ""
    unit = str(m.group(1)).strip().lower()
    unit = unit.rstrip("s")
    return unit


def _extract_count_subject(query: str, count_unit: str) -> str:
    q = _normalize_space(query)
    unit = re.escape(str(count_unit or "").strip())
    if not unit:
        return ""
    patterns = [
        rf"\bhow many\s+{unit}s?\b[^?]*?\b(?:in|into|for|of|with|about)\s+(?:the\s+)?([A-Za-z0-9][A-Za-z0-9'\- ]{{2,70}}?)(?:\b(?:when|while|where|who|that|which)\b|\?|$)",
        rf"\bhow many\s+{unit}s?\b[^?]*?\b(?:accepted|enrolled|registered)\s+(?:in|into|for)\s+(?:the\s+)?([A-Za-z0-9][A-Za-z0-9'\- ]{{2,70}}?)(?:\b(?:when|while|where|who|that|which)\b|\?|$)",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if not m:
            continue
        val = _normalize_space(str(m.group(1))).strip(" ,.;:!?\"'")
        if len(_tokenize(val)) >= 1:
            return val
    return ""


def _extract_entities(query: str, limit: int = 6) -> List[str]:
    text = str(query or "")
    entities: List[str] = []
    qword = {
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "did",
        "do",
        "does",
        "is",
        "are",
        "was",
        "were",
    }
    blocked_single_token = qword.union(
        {
            "can",
            "any",
            "for",
            "please",
            "should",
            "would",
            "could",
            "tell",
            "remind",
            "me",
            "my",
            "our",
            "your",
        }
    )

    for m in re.findall(r"\"([^\"]{2,80})\"", text):
        entities.append(str(m))
    for m in re.findall(r"'([^']{2,80})'", text):
        entities.append(str(m))

    caps = re.findall(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b",
        text,
    )
    entities.extend(caps)

    for m in re.findall(
        r"\b(?:for|about|regarding|on)\s+([A-Za-z0-9][A-Za-z0-9'\- ]{2,60}?)(?:[,.!?]|$)",
        text,
        flags=re.IGNORECASE,
    ):
        entities.append(str(m))
    for m in re.findall(
        r"\b(?:in|into|with|from|at)\s+(?:the\s+)?([A-Za-z0-9][A-Za-z0-9'\- ]{2,50}?)(?:\b(?:when|where|who|what|how|did|do|does|is|are|was|were|have|has|had)\b|[,.!?]|$)",
        text,
        flags=re.IGNORECASE,
    ):
        entities.append(str(m))

    cleaned: List[str] = []
    for ent in entities:
        norm = _normalize_space(ent).strip(" ,.;:!?\"'")
        norm = re.sub(
            r"^(?:can|any|please|for|tell me|remind me)\s+",
            "",
            norm,
            flags=re.IGNORECASE,
        ).strip(" ,.;:!?\"'")
        toks = _tokenize(norm)
        if not toks:
            continue
        if len(toks) == 1 and toks[0] in blocked_single_token:
            continue
        if all(t in blocked_single_token for t in toks):
            continue
        cleaned.append(norm)
    return _dedup_texts(cleaned, limit)


def _extract_time_terms(query: str, limit: int = 4) -> List[str]:
    q = str(query or "")
    out: List[str] = []
    for m in re.findall(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
        q,
        flags=re.IGNORECASE,
    ):
        out.append(str(m))
    for m in re.findall(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", q):
        out.append(str(m))
    for m in re.findall(
        r"\b(?:first|earlier|later|before|after|when|during|between|since|until)\b",
        q,
        flags=re.IGNORECASE,
    ):
        out.append(str(m))
    return _dedup_texts(out, limit)


def _extract_state_keys(query: str, limit: int = 4) -> List[str]:
    q = _normalize_space(query)
    keys: List[str] = []
    for m in re.findall(
        r"\b(?:for|about|regarding)\s+([A-Za-z0-9][A-Za-z0-9'\- ]{2,80}?)(?:,|\?|$)",
        q,
        flags=re.IGNORECASE,
    ):
        keys.append(str(m))
    if re.search(r"\bwhere\s+is\b|\bwhere\s+are\b", q, flags=re.IGNORECASE):
        for m in re.findall(
            r"\bwhere\s+(?:is|are)\s+([A-Za-z0-9][A-Za-z0-9'\- ]{2,60}?)(?:\?|$)",
            q,
            flags=re.IGNORECASE,
        ):
            keys.append(str(m))
    return _dedup_texts(keys, limit)


def _extract_plan_keywords(query: str, limit: int = 6) -> List[str]:
    q = str(query or "")
    toks = _tokenize(q)
    stop = {
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "many",
        "much",
        "did",
        "do",
        "does",
        "is",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "they",
        "their",
        "he",
        "she",
        "it",
        "its",
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "from",
        "at",
        "this",
        "that",
        "these",
        "those",
        "any",
        "can",
        "could",
        "would",
        "should",
    }
    out: List[str] = []
    seen: Set[str] = set()
    for t in toks:
        low = str(t).lower().strip()
        if len(low) < 3 or low in stop:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(low)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _focus_stopwords() -> Set[str]:
    return {
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "did",
        "do",
        "does",
        "is",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "can",
        "could",
        "would",
        "should",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "they",
        "their",
        "he",
        "she",
        "it",
        "its",
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "from",
        "at",
        "this",
        "that",
        "these",
        "those",
        "first",
        "later",
        "earlier",
        "before",
        "after",
        "during",
        "between",
        "current",
        "currently",
        "latest",
        "now",
        "more",
        "less",
        "many",
        "much",
        "number",
        "count",
    }


def _is_valid_focus_phrase(text: str) -> bool:
    toks = _tokenize(text)
    if not toks:
        return False
    stop = _focus_stopwords()
    starter_block = {
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "do",
        "did",
        "does",
        "is",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "can",
        "could",
        "would",
        "should",
        "i",
        "we",
        "you",
    }
    if len(toks) == 1 and toks[0] in stop:
        return False
    if all(t in stop for t in toks):
        return False
    if toks[0] in starter_block and len(toks) <= 4:
        return False
    non_stop = [t for t in toks if t not in stop]
    if not non_stop:
        return False
    return True


def _extract_focus_phrases(
    *,
    raw: str,
    answer_type: str,
    compare_options: List[str],
    target_object: str,
    entities: List[str],
    time_terms: List[str],
    state_keys: List[str],
    limit: int = 8,
) -> List[str]:
    out: List[str] = []
    if target_object:
        out.append(target_object)
    out.extend(compare_options[:2])
    out.extend(entities[:6])
    if answer_type.startswith("temporal"):
        out.extend(time_terms[:2])
    if answer_type == "update":
        out.extend(state_keys[:2])

    # Generic "X of/for/about Y" tail phrase capture (non-dataset-specific).
    for pat in [
        r"\b(?:more|less|higher|lower)\s+(?:than\s+)?([A-Za-z0-9][A-Za-z0-9'\- ]{2,80})",
        r"\b(?:issue|problem|update|status|details?|information)\s+(?:on|about|for|with)\s+([A-Za-z0-9][A-Za-z0-9'\- ]{2,80})",
        r"\b(?:for|about|regarding)\s+([A-Za-z0-9][A-Za-z0-9'\- ]{2,80})",
    ]:
        m = re.search(pat, raw, flags=re.IGNORECASE)
        if m:
            out.append(str(m.group(1)))

    # Add one compact content phrase from non-stop tokens.
    tokens = [t for t in _tokenize(raw) if t not in _focus_stopwords()]
    if len(tokens) >= 2:
        out.append(" ".join(tokens[: min(5, len(tokens))]))

    cleaned: List[str] = []
    for item in out:
        norm = _normalize_space(item).strip(" ,.;:!?\"'")
        norm = re.sub(
            r"^(?:what|which|who|where|when|why|how|do|did|does|is|are|was|were|have|has|had|can|could|would|should)\s+",
            "",
            norm,
            flags=re.IGNORECASE,
        )
        norm = re.sub(r"^(?:i|we|you)\s+", "", norm, flags=re.IGNORECASE)
        norm = re.sub(r"^(?:and|or)\s+", "", norm, flags=re.IGNORECASE)
        if not norm:
            continue
        if not _is_valid_focus_phrase(norm):
            continue
        cleaned.append(norm)
    deduped = _dedup_texts(cleaned, limit * 2)
    compact: List[str] = []
    seen_low: List[str] = []
    for phrase in deduped:
        low = phrase.lower()
        if any((low in prev) or (prev in low and len(low) - len(prev) <= 6) for prev in seen_low):
            continue
        compact.append(phrase)
        seen_low.append(low)
        if len(compact) >= max(1, int(limit)):
            break
    return compact


def _build_keyword_query(
    *,
    focus_phrases: List[str],
    answer_type: str,
    count_unit: str,
    time_terms: List[str],
) -> str:
    terms: List[str] = []
    for phrase in focus_phrases[:4]:
        toks = _tokenize(phrase)
        if len(toks) >= 2:
            terms.extend(toks[:2])
        elif toks:
            terms.extend(toks[:1])
    terms.extend(_tokenize(" ".join(time_terms[:2])))
    if count_unit:
        terms.append(count_unit)
    if answer_type.startswith("temporal"):
        terms.extend(["first", "before", "after", "date", "time"])
    elif answer_type == "update":
        terms.extend(["current", "latest", "updated", "now"])
    elif answer_type == "count":
        terms.extend(["number", "count"])
    terms = _dedup_texts(terms, 10)
    return _normalize_space(" ".join(terms))


def _compose_sub_queries(
    *,
    raw: str,
    focus_phrases: List[str],
    answer_type: str,
    keyword_query: str,
    max_sub_queries: int,
) -> List[str]:
    out: List[str] = [raw]

    kq = _normalize_space(keyword_query)
    if len(_tokenize(kq)) >= 2 and kq.lower() != raw.lower():
        out.append(kq)

    focus_pair = _normalize_space(" ".join(focus_phrases[:2]))
    if len(_tokenize(focus_pair)) >= 2 and focus_pair.lower() != raw.lower():
        out.append(focus_pair)

    if answer_type == "temporal_comparison" and len(focus_phrases) >= 2:
        cmp_q = _normalize_space(
            f"{focus_phrases[0]} {focus_phrases[1]} first before after"
        )
        if len(_tokenize(cmp_q)) >= 2:
            out.append(cmp_q)
    elif answer_type == "update" and focus_phrases:
        upd_q = _normalize_space(f"{focus_phrases[0]} current latest update status")
        if len(_tokenize(upd_q)) >= 2:
            out.append(upd_q)
    elif answer_type.startswith("temporal") and focus_phrases:
        tmp_q = _normalize_space(f"{focus_phrases[0]} date time first before after")
        if len(_tokenize(tmp_q)) >= 2:
            out.append(tmp_q)

    return _dedup_texts(out, max_sub_queries)


def build_query_plan(query: str, max_sub_queries: int = 4) -> Dict[str, object]:
    """Create a lightweight structured plan for retrieval-time query expansion."""
    raw = _normalize_space(query)
    q = raw.lower()

    answer_type = "factoid"
    intent = "lookup"
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|\btotal\b", q):
        answer_type = "count"
        intent = "count"
    elif re.search(
        r"\bwhen\b|\bfirst\b|\bearlier\b|\blater\b|\bbefore\b|\bafter\b|\bhow long\b|\bduration\b|\bdays?\b|\bweeks?\b|\bmonths?\b|\byears?\b",
        q,
    ):
        answer_type = "temporal"
        intent = "temporal"
    elif re.search(
        r"\bcurrently\b|\bcurrent\b|\bnow\b|\bswitch\b|\bchanged\b|\bupdate(?:d)?\b|\bmore or less\b|\bstill\b",
        q,
    ):
        answer_type = "update"
        intent = "update"
    elif re.search(
        r"\bprefer(?:ence)?\b|\bfavorite\b|\brecommend\b|\bsuggest(?:ion)?\b|\bwhat should\b",
        q,
    ):
        answer_type = "preference"
        intent = "preference"

    compare_options = parse_choice_query(raw, max_options=3, default_target_k=2)
    if not compare_options:
        compare_options = extract_choice_options(raw, max_options=3)
    compare_options = _dedup_texts([str(x) for x in compare_options], 3)
    if len(compare_options) >= 2 and answer_type == "temporal":
        answer_type = "temporal_comparison"
        intent = "temporal"

    target_object = _extract_target_object(raw)
    entities = _extract_entities(raw, limit=6)
    count_unit = _extract_count_unit(raw)
    count_units = {
        "day",
        "week",
        "month",
        "year",
        "hour",
        "minute",
        "second",
    }
    if answer_type == "count" and count_unit in count_units:
        target_low = str(target_object or "").strip().lower().rstrip("s")
        if (not target_object) or target_low == count_unit:
            for ent in entities:
                ent_low = str(ent).strip().lower()
                if not ent_low:
                    continue
                if count_unit in ent_low:
                    continue
                if re.search(
                    r"\b(first|last|earlier|later|before|after|when|current|currently)\b",
                    ent_low,
                ):
                    continue
                target_object = ent
                break
        if (not target_object) or str(target_object).strip().lower().rstrip("s") == count_unit:
            subject = _extract_count_subject(raw, count_unit)
            if subject:
                target_object = subject
    time_terms = _extract_time_terms(raw, limit=4)
    state_keys = _extract_state_keys(raw, limit=4)
    plan_keywords = _extract_plan_keywords(raw, limit=6)
    constraints = _dedup_texts(time_terms, limit=4)
    need_latest_state = bool(
        answer_type == "update"
        or re.search(r"\bcurrent|currently|latest|now|switch|changed?|updated?\b", q)
    )

    focus_phrases = _extract_focus_phrases(
        raw=raw,
        answer_type=answer_type,
        compare_options=compare_options,
        target_object=target_object,
        entities=entities,
        time_terms=time_terms,
        state_keys=state_keys,
        limit=8,
    )
    if (not target_object) and answer_type == "count" and focus_phrases:
        target_object = focus_phrases[0]
    if answer_type == "update" and (not state_keys) and focus_phrases:
        state_keys = [focus_phrases[0]]

    must_keywords: List[str] = []
    for phrase in focus_phrases[:2]:
        must_keywords.extend(_tokenize(phrase))
    if not must_keywords:
        must_keywords.extend(plan_keywords[:2])
    must_keywords = _dedup_texts(must_keywords, 6)

    constraint_keywords: List[str] = []
    for term in time_terms[:2]:
        constraint_keywords.extend(_tokenize(term))
    if answer_type == "update":
        constraint_keywords.extend(["current", "latest"])
    if count_unit:
        constraint_keywords.append(count_unit)
    constraint_keywords = _dedup_texts(constraint_keywords, 6)

    keyword_query = _build_keyword_query(
        focus_phrases=focus_phrases,
        answer_type=answer_type,
        count_unit=count_unit,
        time_terms=time_terms,
    )
    sub_queries = _compose_sub_queries(
        raw=raw,
        focus_phrases=focus_phrases,
        answer_type=answer_type,
        keyword_query=keyword_query,
        max_sub_queries=max_sub_queries,
    )

    time_range = ""
    if time_terms:
        time_range = ", ".join(time_terms[:2])

    return {
        "intent": intent,
        "answer_type": answer_type,
        "entities": entities,
        "focus_phrases": focus_phrases,
        "time_range": time_range,
        "constraints": constraints,
        "need_latest_state": need_latest_state,
        "sub_queries": sub_queries,
        "target_object": target_object,
        "compare_options": compare_options,
        "time_terms": time_terms,
        "state_keys": state_keys,
        "count_unit": count_unit,
        "plan_keywords": plan_keywords,
        "must_keywords": must_keywords,
        "constraint_keywords": constraint_keywords,
        "keyword_query": keyword_query,
    }


def slot_coverage_score(text: str, plan: Dict[str, object]) -> float:
    """Measure whether a retrieved unit covers plan-critical slots."""
    low = _normalize_space(text).lower()
    if not low:
        return 0.0
    score = 0.0

    txt_tokens = set(_tokenize(low))

    focus_phrases = [str(x).strip().lower() for x in list(plan.get("focus_phrases", []))]
    for phrase in focus_phrases[:3]:
        if not phrase:
            continue
        p_tokens = set(_tokenize(phrase))
        if not p_tokens:
            continue
        overlap = len(p_tokens.intersection(txt_tokens)) / float(len(p_tokens))
        score += 0.22 * overlap

    compare_options = [str(x).strip().lower() for x in list(plan.get("compare_options", []))]
    for option in compare_options[:2]:
        if option and option in low:
            score += 0.16

    if bool(plan.get("need_latest_state", False)):
        if re.search(r"\b(now|currently|latest|updated?|switched?|changed?)\b", low):
            score += 0.2

    answer_type = str(plan.get("answer_type", "factoid")).strip().lower()
    if answer_type.startswith("temporal"):
        if re.search(r"\b(before|after|first|later|earlier|date|time|week|month|year|day)\b", low):
            score += 0.1

    entities = [str(x).strip().lower() for x in list(plan.get("entities", []))]
    if any(e and e in low for e in entities[:2]):
        score += 0.08

    return min(1.0, float(score))


def detect_missing_slots(
    plan: Dict[str, object],
    units: List[Dict[str, object]],
    top_n: int = 12,
) -> List[str]:
    """Detect missing critical slots from current evidence pool."""
    joined = " \n ".join(
        _normalize_space(str(x.get("text", ""))).lower()
        for x in units[: max(1, int(top_n))]
        if str(x.get("text", "")).strip()
    )
    missing: List[str] = []
    answer_type = str(plan.get("answer_type", "factoid")).strip().lower()

    focus_phrases = [str(x).strip().lower() for x in list(plan.get("focus_phrases", []))]
    for phrase in focus_phrases[:2]:
        if phrase and phrase not in joined:
            missing.append(f"focus_phrase:{phrase}")

    compare_options = [str(x).strip().lower() for x in list(plan.get("compare_options", []))]
    if answer_type == "temporal_comparison" and len(compare_options) >= 2:
        for option in compare_options[:2]:
            if option and option not in joined:
                missing.append(f"compare_option:{option}")

    if answer_type.startswith("temporal"):
        if not re.search(
            r"\b(before|after|first|later|earlier|date|time|week|month|year|day)\b",
            joined,
        ):
            missing.append("time_anchor")

    if bool(plan.get("need_latest_state", False)):
        if not re.search(r"\b(now|currently|latest|updated?|switched?|changed?)\b", joined):
            missing.append("state_key")

    return _dedup_texts(missing, 4)


def build_gap_queries(
    query: str,
    plan: Dict[str, object],
    missing_slots: List[str],
    max_queries: int = 2,
) -> List[str]:
    """Build at most one extra retrieval round query set for missing slots."""
    out: List[str] = []
    base = _normalize_space(query)
    focus_phrases = [str(x).strip() for x in list(plan.get("focus_phrases", []))]
    stop = {
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "did",
        "do",
        "does",
        "is",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "they",
        "their",
        "the",
        "a",
        "an",
        "of",
        "to",
        "for",
        "with",
        "in",
        "on",
        "at",
    }

    def _compact_query(text: str, limit: int = 10) -> str:
        toks: List[str] = []
        seen: set[str] = set()
        for token in _tokenize(text):
            if len(token) < 2 or token in stop or token in seen:
                continue
            seen.add(token)
            toks.append(token)
            if len(toks) >= max(1, int(limit)):
                break
        return _normalize_space(" ".join(toks))

    def _merge_query(primary: str, context: str, context_limit: int = 4) -> str:
        merged: List[str] = []
        seen: set[str] = set()
        for token in _tokenize(primary):
            if len(token) < 2 or token in stop or token in seen:
                continue
            seen.add(token)
            merged.append(token)
        extra = 0
        for token in _tokenize(context):
            if len(token) < 2 or token in stop or token in seen:
                continue
            seen.add(token)
            merged.append(token)
            extra += 1
            if extra >= max(1, int(context_limit)):
                break
        return _normalize_space(" ".join(merged))

    base_compact = _compact_query(base, limit=12)
    base_context = _compact_query(base, limit=6)

    for item in missing_slots:
        if item.startswith("focus_phrase:"):
            phrase = item.split(":", 1)[1].strip()
            phrase_compact = _compact_query(phrase, limit=10)
            if phrase_compact:
                out.append(phrase_compact)
                if base_context:
                    out.append(_merge_query(phrase_compact, base_context, context_limit=4))
        elif item.startswith("compare_option:"):
            opt = item.split(":", 1)[1].strip()
            if opt:
                opt_compact = _compact_query(opt, limit=6) or _normalize_space(opt)
                out.append(opt_compact)
                if base_context:
                    out.append(_merge_query(opt_compact, base_context, context_limit=4))
        elif item == "time_anchor":
            out.append(_merge_query(base_compact or "date time", "date time when", context_limit=3))
        elif item == "state_key":
            for key in focus_phrases[:1]:
                key_compact = _compact_query(key, limit=8) or _normalize_space(key)
                if key_compact:
                    out.append(key_compact)
                    out.append(_merge_query(key_compact, "current latest recent", context_limit=3))

    return _dedup_texts(out, max_queries)
