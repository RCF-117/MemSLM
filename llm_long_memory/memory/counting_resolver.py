"""Deterministic counting resolver for retrieval-grounded QA."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple


Evidence = Dict[str, object]
Candidate = Dict[str, object]


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

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(str(text).split()).strip()

    @staticmethod
    def _extract_quoted_options(query: str) -> List[str]:
        return [x.strip() for x in re.findall(r"'([^']{2,120})'|\"([^\"]{2,120})\"", query) for x in x if x.strip()]

    def _is_count_query(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in self.count_query_keywords)

    def _is_day_count_query(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in self.day_count_keywords)

    def _is_list_count_query(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in self.list_count_query_keywords)

    def _extract_numeric_phrase(self, text: str) -> Optional[str]:
        m = re.search(
            r"\b\d+(?:\.\d+)?\s*(?:days?|hours?|weeks?|months?|years?|times?|items?|people|persons|events?)?\b",
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        return self._norm(m.group(0))

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

        uniq: List[str] = []
        seen: set[str] = set()

        for item in evidence:
            text = str(item.get("text", ""))
            for ent in self._extract_list_entities(text):
                if ent not in seen:
                    seen.add(ent)
                    uniq.append(ent)

        # candidate phrases are a weak signal but can help when evidence is short.
        for cand in candidates:
            ent = self._normalize_entity(str(cand.get("text", "")))
            if ent and ent not in seen:
                seen.add(ent)
                uniq.append(ent)

        if len(uniq) < self.list_count_min_unique_entities:
            return None
        return (str(len(uniq)), "list_entity_count")

    def _resolve_day_diff(
        self, query: str, evidence: Sequence[Evidence]
    ) -> Optional[Tuple[str, str]]:
        if not self._is_day_count_query(query):
            return None
        options = self._extract_quoted_options(query)
        if len(options) >= 2:
            left, right = options[0].lower(), options[1].lower()
            left_dates: List[datetime] = []
            right_dates: List[datetime] = []
            for item in evidence:
                text = str(item.get("text", ""))
                dates = self._extract_dates(text)
                if not dates:
                    continue
                low = text.lower()
                if left in low:
                    left_dates.extend(dates)
                if right in low:
                    right_dates.extend(dates)
            if left_dates and right_dates:
                delta = abs((min(right_dates) - min(left_dates)).days)
                return (f"{delta} days", "day_diff_from_option_dates")
            # For explicit two-event questions, do not fall back to global date spread.
            # Returning None here lets the main LLM path answer instead of forcing wrong "0 days".
            return None

        all_dates: List[datetime] = []
        for item in evidence:
            all_dates.extend(self._extract_dates(str(item.get("text", ""))))
        if len(all_dates) >= 2:
            dates_sorted = sorted(all_dates)
            delta = abs((dates_sorted[-1] - dates_sorted[0]).days)
            return (f"{delta} days", "day_diff_from_evidence_dates")
        return None

    def resolve(
        self,
        query: str,
        evidence: Sequence[Evidence],
        candidates: Sequence[Candidate],
    ) -> Optional[Dict[str, str]]:
        """Return deterministic count answer when possible."""
        if (not self.enabled) or (not self._is_count_query(query)):
            return None

        day_result = self._resolve_day_diff(query, evidence)
        if day_result is not None:
            answer, reason = day_result
            return {"answer": answer, "reason": reason}

        list_result = self._resolve_list_count(query, evidence, candidates)
        if list_result is not None:
            answer, reason = list_result
            return {"answer": answer, "reason": reason}

        for cand in candidates:
            text = str(cand.get("text", "")).strip()
            number = self._extract_numeric_phrase(text)
            if number:
                return {"answer": number, "reason": "numeric_candidate"}

        for item in evidence:
            text = str(item.get("text", "")).strip()
            number = self._extract_numeric_phrase(text)
            if number:
                return {"answer": number, "reason": "numeric_evidence_sentence"}
        return None
