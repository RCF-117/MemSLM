"""Text and keyword utilities for long-memory extraction pipeline."""

from __future__ import annotations

import re
from typing import Iterable, List


class LongMemoryTextUtils:
    """Pure text helpers used by long-memory extraction and indexing."""

    NUMBER_WORDS = {
        "zero": "zero",
        "one": "one",
        "two": "two",
        "three": "three",
        "four": "four",
        "five": "five",
        "six": "six",
        "seven": "seven",
        "eight": "eight",
        "nine": "nine",
        "ten": "ten",
        "eleven": "eleven",
        "twelve": "twelve",
        "thirteen": "thirteen",
        "fourteen": "fourteen",
        "fifteen": "fifteen",
        "sixteen": "sixteen",
        "seventeen": "seventeen",
        "eighteen": "eighteen",
        "nineteen": "nineteen",
        "twenty": "twenty",
        "thirty": "thirty",
        "forty": "forty",
        "fifty": "fifty",
        "sixty": "sixty",
        "seventy": "seventy",
        "eighty": "eighty",
        "ninety": "ninety",
        "hundred": "hundred",
        "thousand": "thousand",
    }

    def __init__(
        self,
        *,
        stopwords: Iterable[str],
        keyword_max_count: int,
        no_source_keyword_fallback: bool,
        sentence_overlap: int,
        sentence_min_chars: int,
    ) -> None:
        self.stopwords = {str(x).strip().lower() for x in stopwords if str(x).strip()}
        self.keyword_max_count = int(max(1, keyword_max_count))
        self.no_source_keyword_fallback = bool(no_source_keyword_fallback)
        self.sentence_overlap = int(max(0, sentence_overlap))
        self.sentence_min_chars = int(max(1, sentence_min_chars))

    @staticmethod
    def normalize_space(text: str) -> str:
        return " ".join(str(text).strip().split())

    def tokenize(self, text: str) -> List[str]:
        out: List[str] = []
        for raw in str(text).lower().split():
            cleaned = "".join(ch for ch in raw if ch.isalnum())
            if cleaned:
                out.append(cleaned)
        return out

    def is_noise_keyword(self, token: str) -> bool:
        t = str(token).strip().lower()
        if not t:
            return True
        if t in self.stopwords:
            return True
        if t in self.ROLE_AND_META_KEYWORDS:
            return True
        if len(t) <= 1:
            return True
        if t.isdigit() and len(t) <= 2:
            return True
        return False

    def keyword_candidates(self, text: str) -> List[str]:
        out: List[str] = []
        for tok in self.tokenize(text):
            if self.is_noise_keyword(tok):
                continue
            out.append(tok)
        return out

    def first_number_token(self, text: str) -> str:
        value = self.normalize_space(text).lower()
        if not value:
            return ""
        for raw in value.split():
            token = "".join(ch for ch in raw if ch.isalnum())
            if not token:
                continue
            if token.isdigit():
                return token
            if token in self.NUMBER_WORDS:
                return self.NUMBER_WORDS[token]
        return ""

    def normalize_value_text(
        self,
        text: str,
        *,
        fact_slot: str,
        value_type: str,
    ) -> str:
        value = self.normalize_space(text)
        if not value:
            return ""
        slot = self.normalize_space(fact_slot).lower()
        vtype = self.normalize_space(value_type).lower()
        if (slot != "count") and (vtype != "number"):
            return value
        num = self.first_number_token(value)
        return num or value

    def build_keywords(
        self,
        *,
        model_keywords: List[str],
        subject: str,
        action: str,
        obj: str,
        event_text: str,
        raw_span: str,
        source_content: str,
        time_text: str,
        location_text: str,
    ) -> List[str]:
        merged: List[str] = []
        seen = set()

        def push(tok: str) -> None:
            t = str(tok).strip().lower()
            if self.is_noise_keyword(t):
                return
            if t in seen:
                return
            seen.add(t)
            merged.append(t)

        for tok in self.keyword_candidates(time_text):
            push(tok)
        for tok in self.keyword_candidates(location_text):
            push(tok)
        for tok in model_keywords:
            for part in self.keyword_candidates(tok):
                push(part)
        for src in (subject, action, obj, event_text, raw_span):
            for tok in self.keyword_candidates(src):
                push(tok)
        if not self.no_source_keyword_fallback:
            for tok in self.keyword_candidates(source_content):
                push(tok)

        return merged[: self.keyword_max_count]

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        cleaned = str(text).replace("\n", " ").strip()
        if not cleaned:
            return []
        parts = re.split(r"(?<=[。！？!?\.])\s+", cleaned)
        out = [p.strip() for p in parts if p.strip()]
        return out if out else [cleaned]

    def pack_sentences(self, text: str, max_chars: int) -> List[str]:
        sentences = self.split_sentences(text)
        if not sentences:
            return []
        max_len = max(64, int(max_chars))
        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0

        for s in sentences:
            s2 = self.normalize_space(s)
            if not s2:
                continue
            add_len = len(s2) + (1 if cur else 0)
            if cur and (cur_len + add_len > max_len):
                merged = " ".join(cur).strip()
                if len(merged) >= self.sentence_min_chars:
                    chunks.append(merged)
                cur = cur[-self.sentence_overlap :] if self.sentence_overlap > 0 else []
                cur_len = len(" ".join(cur))
            if len(s2) > max_len:
                start = 0
                while start < len(s2):
                    piece = s2[start : start + max_len].strip()
                    if piece:
                        chunks.append(piece)
                    start += max_len
                cur = []
                cur_len = 0
                continue
            cur.append(s2)
            cur_len += add_len

        if cur:
            merged = " ".join(cur).strip()
            if len(merged) >= self.sentence_min_chars:
                chunks.append(merged)
        return chunks

    def normalize_fact_component(self, text: str) -> str:
        toks = [t for t in self.tokenize(text) if not self.is_noise_keyword(t)]
        return " ".join(toks[:8]).strip()

    def build_fact_key(self, subject: str, action: str, obj: str) -> str:
        s = self.normalize_fact_component(subject)
        a = self.normalize_fact_component(action)
        o = self.normalize_fact_component(obj)
        return f"{s}|{a}|{o}".strip("|")
    ROLE_AND_META_KEYWORDS = {
        "user",
        "assistant",
        "system",
        "conversation",
        "chat",
        "response",
        "suggestion",
        "suggestions",
        "recommend",
        "recommendation",
        "recommendations",
        "advice",
    }
