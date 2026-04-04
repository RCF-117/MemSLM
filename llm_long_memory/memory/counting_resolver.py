"""Deterministic counting resolver for retrieval-grounded QA."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
            str(x).strip().lower() for x in list(self.cfg["list_count_focus_stopwords"])
        }
        self.list_count_focus_min_overlap = int(self.cfg["list_count_focus_min_overlap"])
        irregular_cfg = dict(self.cfg.get("list_count_irregular_forms", {}))
        self.list_count_irregular_forms = {
            str(k).strip().lower(): [str(x).strip().lower() for x in list(v)]
            for k, v in irregular_cfg.items()
        }
        numeric_cfg = dict(self.cfg["numeric_answer_filter"])
        self.numeric_answer_enabled = bool(numeric_cfg["enabled"])
        self.numeric_focus_min_overlap = int(numeric_cfg["focus_min_overlap"])
        self.numeric_allowed_units = {
            str(x).strip().lower() for x in list(numeric_cfg["allowed_units"])
        }
        self.numeric_disallowed_units = {
            str(x).strip().lower() for x in list(numeric_cfg["disallowed_units"])
        }

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(str(text).split()).strip()

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
        q = query.lower()
        return any(k in q for k in self.list_count_query_keywords)

    def _extract_numeric_phrase(self, text: str) -> Optional[Tuple[str, str]]:
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*([a-zA-Z%$]+)?\b", text, flags=re.IGNORECASE)
        if not m:
            return None
        number = str(m.group(1)).strip()
        unit = str(m.group(2) or "").strip().lower()
        phrase = self._norm(m.group(0))
        return phrase, unit

    @staticmethod
    def _parse_date_token(token: str) -> Optional[datetime]:
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

    def _extract_dates(self, text: str) -> List[datetime]:
        out: List[datetime] = []
        for pat in self.date_patterns:
            for m in pat.findall(text):
                token = str(m).strip()
                if not token:
                    continue
                dt = self._parse_date_token(token)
                if dt is not None:
                    out.append(dt)
        return out

    @staticmethod
    def _parse_session_date(session_date: str) -> Optional[datetime]:
        token = str(session_date).strip().split(" ")[0]
        if not token:
            return None
        for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(token, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _relative_weekday_date(reference: datetime, weekday: int, is_last: bool) -> datetime:
        """Resolve relative weekday against reference date."""
        delta = (reference.weekday() - weekday) % 7
        if is_last:
            if delta == 0:
                delta = 7
            return reference - timedelta(days=delta)
        return reference - timedelta(days=delta)

    def _extract_relative_dates(self, text: str, session_date: str) -> List[datetime]:
        out: List[datetime] = []
        ref = self._parse_session_date(session_date)
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
                out.append(self._relative_weekday_date(ref, idx, is_last=True))
            elif re.search(rf"\bthis\s+{name}\b", lowered):
                out.append(self._relative_weekday_date(ref, idx, is_last=False))
        return out

    def _extract_dates_with_context(self, text: str, session_date: str) -> List[datetime]:
        dates = self._extract_dates(text)
        dates.extend(self._extract_relative_dates(text=text, session_date=session_date))
        return self._unique_dates(dates)

    @staticmethod
    def _unique_dates(values: Sequence[datetime]) -> List[datetime]:
        """Deduplicate dates at day granularity to avoid fake 0-day deltas."""
        uniq: Dict[str, datetime] = {}
        for d in values:
            key = d.strftime("%Y-%m-%d")
            if key not in uniq:
                uniq[key] = d
        return list(uniq.values())

    @staticmethod
    def _anchor_tokens(text: str) -> List[str]:
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

    def _anchor_match_score(self, anchor: str, text: str) -> float:
        at = set(self._anchor_tokens(anchor))
        if not at:
            return 0.0
        tt = set(self._anchor_tokens(text))
        if not tt:
            return 0.0
        return float(len(at.intersection(tt))) / float(len(at))

    @staticmethod
    def _quoted_phrases(text: str) -> List[str]:
        return [
            x.strip()
            for x in re.findall(r"'([^']{1,80})'|\"([^\"]{1,80})\"", text)
            for x in x
            if x.strip()
        ]

    def _normalize_entity(self, text: str) -> str:
        lowered = self._norm(text).lower()
        lowered = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", lowered)
        if not lowered:
            return ""
        tokens = [t for t in re.findall(r"[a-z0-9]+", lowered) if t]
        if not tokens:
            return ""
        if len(tokens) > self.list_count_max_entity_tokens:
            return ""
        if all(t in self.list_count_stopwords for t in tokens):
            return ""
        if any(re.fullmatch(r"\d+", t) for t in tokens):
            return ""
        return " ".join(tokens)

    def _extract_list_entities(self, text: str) -> List[str]:
        entities: List[str] = []
        # 1) quoted entities are usually the cleanest signal.
        for q in self._quoted_phrases(text):
            norm = self._normalize_entity(q)
            if norm:
                entities.append(norm)

        # 2) comma/and-separated list fragments.
        normalized_text = self._norm(text)
        if "," in normalized_text or " and " in normalized_text.lower():
            tmp = re.sub(r"\band\b", ",", normalized_text, flags=re.IGNORECASE)
            for part in [x.strip() for x in tmp.split(",")]:
                norm = self._normalize_entity(part)
                if norm:
                    entities.append(norm)
        return entities

    def _extract_query_focus_tokens(self, query: str) -> List[str]:
        """Extract coarse focus terms from count questions to filter noisy evidence."""
        lowered = self._norm(query).lower()
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
                if tok in self.list_count_focus_stopwords:
                    continue
                tokens.append(tok)
        uniq: List[str] = []
        for tok in tokens:
            if tok not in uniq:
                uniq.append(tok)
        return uniq

    def _sentence_focus_overlap(self, text: str, focus_tokens: Sequence[str]) -> int:
        if not focus_tokens:
            return 0
        raw_tokens = set(re.findall(r"[a-z0-9]+", str(text).lower()))
        expanded: set[str] = set(raw_tokens)
        for tok in list(raw_tokens):
            if tok.endswith("ed") and len(tok) > 3:
                expanded.add(tok[:-2])
            if tok.endswith("ing") and len(tok) > 4:
                expanded.add(tok[:-3])
            if tok.endswith("s") and len(tok) > 2:
                expanded.add(tok[:-1])
        for base, forms in self.list_count_irregular_forms.items():
            if any(form in raw_tokens for form in forms):
                expanded.add(base)
        return sum(1 for tok in focus_tokens if tok in expanded)

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
            if self._sentence_focus_overlap(text, focus_tokens) < self.list_count_focus_min_overlap:
                continue
            for ent in self._extract_list_entities(text):
                if ent not in seen:
                    seen.add(ent)
                    uniq.append(ent)

        # candidate phrases are a weak signal but can help when evidence is short.
        for cand in candidates:
            ctext = str(cand.get("text", ""))
            if self._sentence_focus_overlap(ctext, focus_tokens) < self.list_count_focus_min_overlap:
                continue
            ent = self._normalize_entity(ctext)
            if ent and ent not in seen:
                seen.add(ent)
                uniq.append(ent)

        if len(uniq) < self.list_count_min_unique_entities:
            return None
        return (str(len(uniq)), "list_entity_count")

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
