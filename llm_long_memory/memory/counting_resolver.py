"""Deterministic counting resolver for retrieval-grounded QA."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_long_memory.memory.counting_dates import (
    anchor_tokens,
    anchor_match_score,
    extract_dates_with_context,
    parse_date_token,
    parse_session_date,
    unique_dates,
)
from llm_long_memory.memory.counting_entities import (
    extract_list_entities,
    extract_query_focus_tokens,
    normalize_entity,
    normalize_text,
    quoted_phrases,
    sentence_focus_overlap,
)

Evidence = Dict[str, object]
Candidate = Dict[str, object]
Chunk = Dict[str, object]


class CountingResolver:
    """Resolve count-style questions from evidence without free-form generation."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = dict(cfg or {})
        self.enabled = bool(self.cfg.get("enabled", False))
        self.count_query_keywords = [
            str(x).strip().lower() for x in list(self.cfg.get("count_query_keywords", []))
        ]
        self.day_count_keywords = [
            str(x).strip().lower() for x in list(self.cfg.get("day_count_keywords", []))
        ]
        self.date_patterns = [
            re.compile(str(x), flags=re.IGNORECASE) for x in list(self.cfg.get("date_regexes", []))
        ]
        self.list_count_enabled = bool(self.cfg.get("list_count_enabled", False))
        self.list_count_query_keywords = [
            str(x).strip().lower() for x in list(self.cfg.get("list_count_query_keywords", []))
        ]
        self.list_count_min_unique_entities = int(self.cfg.get("list_count_min_unique_entities", 2))
        self.list_count_max_entity_tokens = int(self.cfg.get("list_count_max_entity_tokens", 4))
        self.list_count_stopwords = {
            str(x).strip().lower() for x in list(self.cfg.get("list_count_stopwords", []))
        }
        self.list_count_focus_stopwords = {
            str(x).strip().lower()
            for x in list(self.cfg.get("list_count_focus_stopwords", []))
            if str(x).strip()
        }
        self.list_count_focus_min_overlap = int(self.cfg.get("list_count_focus_min_overlap", 1))
        irregular_cfg = dict(self.cfg.get("list_count_irregular_forms", {}))
        self.list_count_irregular_forms = {
            str(k).strip().lower(): [str(x).strip().lower() for x in list(v)]
            for k, v in irregular_cfg.items()
        }
        numeric_cfg = dict(
            self.cfg.get(
                "numeric_answer_filter",
                {
                    "enabled": False,
                    "focus_min_overlap": 1,
                    "allowed_units": [],
                    "disallowed_units": [],
                },
            )
        )
        self.numeric_answer_enabled = bool(numeric_cfg.get("enabled", False))
        self.numeric_focus_min_overlap = int(numeric_cfg.get("focus_min_overlap", 1))
        self.numeric_allowed_units = {
            str(x).strip().lower() for x in list(numeric_cfg.get("allowed_units", []))
        }
        self.numeric_disallowed_units = {
            str(x).strip().lower() for x in list(numeric_cfg.get("disallowed_units", []))
        }

    @staticmethod
    def _norm(text: str) -> str:
        return normalize_text(text)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text).lower())

    @staticmethod
    def _extract_quoted_options(query: str) -> List[str]:
        return [x.strip() for x in re.findall(r"'([^']{2,120})'|\"([^\"]{2,120})\"", query) for x in x if x.strip()]

    @staticmethod
    def _extract_between_options(query: str) -> List[str]:
        """Extract two anchors from 'between X and Y' style query."""
        q = " ".join(str(query).split())
        m = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)", q, flags=re.IGNORECASE)
        if not m:
            return []
        left = str(m.group(1)).strip(" ,.;:!?\"'")
        right = str(m.group(2)).strip(" ,.;:!?\"'")
        if not left or not right:
            return []
        return [left, right]

    @staticmethod
    def _extract_after_options(query: str) -> List[str]:
        """Extract target/reference anchors from '... X after Y ...' style query."""
        q = " ".join(str(query).split())
        m = re.search(
            r"\b(?:how many days|days)\b.*?\bto\s+(.+?)\s+after\s+(.+?)(?:[?.!]|$)",
            q,
            flags=re.IGNORECASE,
        )
        if not m:
            return []
        target = str(m.group(1)).strip(" ,.;:!?\"'")
        ref = str(m.group(2)).strip(" ,.;:!?\"'")
        if not target or not ref:
            return []
        return [target, ref]

    def _is_count_query(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in self.count_query_keywords)

    def _is_day_count_query(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in self.day_count_keywords)

    def _is_list_count_query(self, query: str) -> bool:
        q = " ".join(str(query).split()).lower()
        if self._is_day_count_query(query):
            return False
        if any(k in q for k in self.list_count_query_keywords):
            return True
        return bool(
            re.search(r"\bhow many\b", q)
            or re.search(r"\bnumber of\b", q)
            or re.search(r"\bcount of\b", q)
            or re.search(r"\bhow much\b", q)
        )

    def _extract_numeric_phrase(self, text: str) -> Optional[Tuple[str, str]]:
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*([a-zA-Z%$]+)?\b", text, flags=re.IGNORECASE)
        if not m:
            return None
        unit = str(m.group(2) or "").strip().lower()
        phrase = self._norm(m.group(0))
        return phrase, unit

    @staticmethod
    def _parse_date_token(token: str) -> Optional[datetime]:
        return parse_date_token(token)

    def _extract_dates(self, text: str) -> List[datetime]:
        return extract_dates_with_context(text=text, session_date="", date_patterns=self.date_patterns)

    @staticmethod
    def _parse_session_date(session_date: str) -> Optional[datetime]:
        return parse_session_date(session_date)

    def _extract_dates_with_context(self, text: str, session_date: str) -> List[datetime]:
        return extract_dates_with_context(
            text=text, session_date=session_date, date_patterns=self.date_patterns
        )

    @staticmethod
    def _unique_dates(values: Sequence[datetime]) -> List[datetime]:
        return unique_dates(values)

    @staticmethod
    def _anchor_tokens(text: str) -> List[str]:
        return anchor_tokens(text)

    def _anchor_match_score(self, anchor: str, text: str) -> float:
        return anchor_match_score(anchor, text)

    @staticmethod
    def _quoted_phrases(text: str) -> List[str]:
        return quoted_phrases(text)

    def _normalize_entity(self, text: str) -> str:
        return normalize_entity(
            text=text,
            max_entity_tokens=self.list_count_max_entity_tokens,
            stopwords=self.list_count_stopwords,
        )

    def _extract_list_entities(self, text: str) -> List[str]:
        return extract_list_entities(
            text=text,
            max_entity_tokens=self.list_count_max_entity_tokens,
            stopwords=self.list_count_stopwords,
        )

    def _extract_query_focus_tokens(self, query: str) -> List[str]:
        return extract_query_focus_tokens(
            query=query, focus_stopwords=self.list_count_focus_stopwords
        )

    @staticmethod
    def _line_has_list_shape(text: str) -> bool:
        low = " ".join(str(text or "").split()).lower()
        return (
            "," in low
            or " and " in low
            or " or " in low
            or " plus " in low
            or "+" in low
        )

    def _sentence_focus_overlap(self, text: str, focus_tokens: Sequence[str]) -> int:
        return sentence_focus_overlap(
            text=text,
            focus_tokens=focus_tokens,
            irregular_forms=self.list_count_irregular_forms,
        )

    def _resolve_list_count(
        self,
        query: str,
        evidence: Sequence[Evidence],
        candidates: Sequence[Candidate],
    ) -> Optional[Tuple[str, str]]:
        if not self.list_count_enabled:
            return None
        if self._is_day_count_query(query):
            return None
        if not self._is_list_count_query(query):
            return None

        focus_tokens = self._extract_query_focus_tokens(query)
        uniq: List[str] = []
        seen: set[str] = set()

        for item in evidence:
            text = str(item.get("text", ""))
            list_entities = self._extract_list_entities(text)
            line_has_shape = self._line_has_list_shape(text)
            focus_overlap = self._sentence_focus_overlap(text, focus_tokens)
            if (
                focus_overlap < self.list_count_focus_min_overlap
                and not (line_has_shape and len(list_entities) >= 2)
            ):
                continue
            for ent in list_entities:
                if ent not in seen:
                    seen.add(ent)
                    uniq.append(ent)

        # candidate phrases are a weak signal but can help when evidence is short.
        for cand in candidates:
            ctext = str(cand.get("text", ""))
            focus_overlap = self._sentence_focus_overlap(ctext, focus_tokens)
            if focus_overlap < self.list_count_focus_min_overlap and not self._line_has_list_shape(ctext):
                continue
            ent = self._normalize_entity(ctext)
            if ent and ent not in seen:
                seen.add(ent)
                uniq.append(ent)

        if len(uniq) < self.list_count_min_unique_entities:
            return None
        return (str(len(uniq)), "list_entity_count")

    def _resolve_numeric_count(
        self,
        query: str,
        evidence: Sequence[Evidence],
        candidates: Sequence[Candidate],
        reranked_chunks: Sequence[Chunk],
    ) -> Optional[Tuple[str, str]]:
        focus_tokens = self._extract_query_focus_tokens(query)
        if not focus_tokens:
            return None
        scored: List[Tuple[float, int, int, str]] = []

        def _collect(
            items: Sequence[Dict[str, object]],
            source_bonus: int,
        ) -> None:
            for item in items:
                text = str(item.get("text", "")).strip()
                numeric = self._extract_numeric_phrase(text)
                if numeric is None:
                    continue
                number, unit = numeric
                if not self._accept_numeric_phrase(number, unit, query, text):
                    continue
                overlap = float(self._sentence_focus_overlap(text, focus_tokens))
                if overlap <= 0.0 and not any(ch.isdigit() for ch in text):
                    continue
                scored.append((overlap, source_bonus, len(self._tokenize(text)), number))

        _collect(reranked_chunks, 3)
        _collect(evidence, 2)
        _collect(candidates, 1)

        if not scored:
            return None
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        answer = scored[0][3]
        return (answer, "numeric_focus_count")

    def _resolve_day_diff(
        self,
        query: str,
        evidence: Sequence[Evidence],
        reranked_chunks: Sequence[Chunk],
    ) -> Optional[Tuple[str, str]]:
        if not self._is_day_count_query(query):
            return None
        options = self._extract_quoted_options(query)
        if len(options) < 2:
            options = self._extract_between_options(query)
        if len(options) < 2:
            options = self._extract_after_options(query)
        if len(options) >= 2:
            left, right = options[0].lower(), options[1].lower()
            left_scored: List[Tuple[float, datetime]] = []
            right_scored: List[Tuple[float, datetime]] = []

            for item in evidence:
                text = str(item.get("text", ""))
                dates = self._extract_dates_with_context(text=text, session_date="")
                if not dates:
                    continue
                l_score = self._anchor_match_score(left, text)
                r_score = self._anchor_match_score(right, text)
                for d in dates:
                    if l_score > 0.0:
                        left_scored.append((l_score, d))
                    if r_score > 0.0:
                        right_scored.append((r_score, d))

            for item in reranked_chunks:
                text = str(item.get("text", ""))
                session_date = str(item.get("session_date", ""))
                dates = self._extract_dates_with_context(text=text, session_date=session_date)
                if not dates:
                    continue
                l_score = self._anchor_match_score(left, text)
                r_score = self._anchor_match_score(right, text)
                for d in dates:
                    if l_score > 0.0:
                        left_scored.append((l_score, d))
                    if r_score > 0.0:
                        right_scored.append((r_score, d))

            left_scored = sorted(left_scored, key=lambda x: x[0], reverse=True)[:8]
            right_scored = sorted(right_scored, key=lambda x: x[0], reverse=True)[:8]
            left_dates = self._unique_dates([d for _s, d in left_scored])
            right_dates = self._unique_dates([d for _s, d in right_scored])
            if left_dates and right_dates:
                left_map = {d.strftime("%Y-%m-%d"): s for s, d in left_scored}
                right_map = {d.strftime("%Y-%m-%d"): s for s, d in right_scored}
                after_mode = " after " in query.lower()
                pair_scores: List[Tuple[float, int]] = []
                for ld in left_dates:
                    for rd in right_dates:
                        l_key = ld.strftime("%Y-%m-%d")
                        r_key = rd.strftime("%Y-%m-%d")
                        score = float(left_map.get(l_key, 0.0) + right_map.get(r_key, 0.0))
                        if after_mode:
                            delta = (ld - rd).days
                            if delta >= 0:
                                pair_scores.append((score, delta))
                        else:
                            delta = abs((ld - rd).days)
                            pair_scores.append((score, delta))

                if pair_scores:
                    non_zero = [x for x in pair_scores if x[1] > 0]
                    chosen = non_zero if non_zero else pair_scores
                    chosen.sort(key=lambda x: (-x[0], x[1]))
                    return (f"{chosen[0][1]} days", "day_diff_from_option_dates")
            # For explicit two-event questions, do not fall back to global date spread.
            # Returning None here lets the main LLM path answer instead of forcing wrong "0 days".
            return None

        all_dates: List[datetime] = []
        for item in evidence:
            all_dates.extend(self._extract_dates_with_context(str(item.get("text", "")), session_date=""))
        for item in reranked_chunks:
            all_dates.extend(
                self._extract_dates_with_context(
                    str(item.get("text", "")),
                    session_date=str(item.get("session_date", "")),
                )
            )
        uniq_dates = self._unique_dates(all_dates)
        # Avoid forced numeric answer when only one unique date is present.
        if len(uniq_dates) >= 2:
            dates_sorted = sorted(uniq_dates)
            delta = abs((dates_sorted[-1] - dates_sorted[0]).days)
            return (f"{delta} days", "day_diff_from_evidence_dates")
        return None

    def _accept_numeric_phrase(self, phrase: str, unit: str, query: str, text: str) -> bool:
        if not self.numeric_answer_enabled:
            return True
        unit_low = unit.lower()
        if unit_low and unit_low in self.numeric_disallowed_units:
            return False
        if self._is_day_count_query(query):
            return ("day" in unit_low) or (unit_low == "")
        if self.numeric_allowed_units and unit_low and unit_low not in self.numeric_allowed_units:
            return False
        focus_tokens = self._extract_query_focus_tokens(query)
        if self._sentence_focus_overlap(text, focus_tokens) < self.numeric_focus_min_overlap:
            return False
        return True

    def resolve(
        self,
        query: str,
        evidence: Sequence[Evidence],
        candidates: Sequence[Candidate],
        reranked_chunks: Optional[Sequence[Chunk]] = None,
    ) -> Optional[Dict[str, str]]:
        """Return deterministic count answer when possible."""
        if (not self.enabled) or (not self._is_count_query(query)):
            return None

        day_result = self._resolve_day_diff(query, evidence, reranked_chunks or [])
        if day_result is not None:
            answer, reason = day_result
            return {"answer": answer, "reason": reason}

        list_result = self._resolve_list_count(query, evidence, candidates)
        if list_result is not None:
            answer, reason = list_result
            return {"answer": answer, "reason": reason}

        numeric_result = self._resolve_numeric_count(
            query=query,
            evidence=evidence,
            candidates=candidates,
            reranked_chunks=reranked_chunks or [],
        )
        if numeric_result is not None:
            answer, reason = numeric_result
            return {"answer": answer, "reason": reason}

        for cand in candidates:
            text = str(cand.get("text", "")).strip()
            numeric = self._extract_numeric_phrase(text)
            if numeric is None:
                continue
            number, unit = numeric
            if self._accept_numeric_phrase(number, unit, query, text):
                return {"answer": number, "reason": "numeric_candidate"}

        for item in evidence:
            text = str(item.get("text", "")).strip()
            numeric = self._extract_numeric_phrase(text)
            if numeric is None:
                continue
            number, unit = numeric
            if self._accept_numeric_phrase(number, unit, query, text):
                return {"answer": number, "reason": "numeric_evidence_sentence"}
        return None
