"""Evidence sentence ranking and extractive span helpers for the active pipeline."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from llm_long_memory.memory.lexical_resources import BASIC_STOPWORDS


class EvidenceCandidateExtractor:
    """Encapsulate evidence sentence ranking and candidate span extraction."""

    def __init__(self, grounding_cfg: Dict[str, Any]) -> None:
        self.grounding_cfg = dict(grounding_cfg)
        self.evidence_top_n_chunks = int(self.grounding_cfg["evidence_top_n_chunks"])
        self.evidence_top_n_sentences = int(self.grounding_cfg["evidence_top_n_sentences"])
        self.evidence_sentence_max_chars = int(self.grounding_cfg["evidence_sentence_max_chars"])
        self.candidate_top_n = int(self.grounding_cfg["candidate_top_n"])
        self.span_min_tokens = int(self.grounding_cfg["span_min_tokens"])
        self.span_max_tokens = int(self.grounding_cfg["span_max_tokens"])
        self.span_top_n_per_sentence = int(self.grounding_cfg["span_top_n_per_sentence"])
        evidence_candidate_cfg = dict(self.grounding_cfg["evidence_candidate"])
        self.evidence_candidate_enabled = bool(evidence_candidate_cfg["enabled"])
        self.evidence_candidate_min_score = float(evidence_candidate_cfg["min_score"])
        self.evidence_candidate_max_tokens = int(evidence_candidate_cfg["max_tokens"])
        candidate_filter_cfg = dict(self.grounding_cfg["candidate_filter"])
        self.candidate_filter_enabled = bool(candidate_filter_cfg["enabled"])
        self.candidate_filter_min_token_count = int(candidate_filter_cfg["min_token_count"])
        self.candidate_filter_role_prefixes = {
            str(x).strip().lower() for x in list(candidate_filter_cfg["role_prefixes"])
        }
        self.candidate_filter_reject_prefixes = [
            str(x).strip().lower() for x in list(candidate_filter_cfg["reject_prefixes"])
        ]
        self.candidate_filter_reject_contains = [
            str(x).strip().lower() for x in list(candidate_filter_cfg["reject_contains"])
        ]

        intent_cfg = dict(self.grounding_cfg["intent_extraction"])
        self.intent_extraction_enabled = bool(intent_cfg["enabled"])
        self.intent_time_keywords = {
            str(x).strip().lower() for x in list(intent_cfg["time_keywords"])
        }
        self.intent_number_keywords = {
            str(x).strip().lower() for x in list(intent_cfg["number_keywords"])
        }
        self.intent_location_keywords = {
            str(x).strip().lower() for x in list(intent_cfg["location_keywords"])
        }
        self.intent_name_keywords = {
            str(x).strip().lower() for x in list(intent_cfg["name_keywords"])
        }
        self.intent_time_patterns = [
            re.compile(str(x), flags=re.IGNORECASE) for x in list(intent_cfg["time_regexes"])
        ]
        self.intent_number_patterns = [
            re.compile(str(x), flags=re.IGNORECASE) for x in list(intent_cfg["number_regexes"])
        ]
        self.intent_capitalized_phrase_max_tokens = int(intent_cfg["capitalized_phrase_max_tokens"])

        scoring_cfg = dict(self.grounding_cfg["candidate_scoring"])
        self.cand_min_score = float(scoring_cfg["min_score"])
        self.cand_reject_tokens = {
            str(x).strip().lower() for x in list(scoring_cfg["reject_tokens"])
        }
        self.answer_stopwords = set(BASIC_STOPWORDS).union(
            {
                "and",
                "or",
                "at",
                "by",
                "from",
                "into",
                "about",
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
                "we",
                "our",
                "your",
            }
        )
        self.number_words = {
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "once",
            "twice",
        }

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text).lower())

    @staticmethod
    def normalize_space(text: str) -> str:
        return " ".join(str(text).split())

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", str(text))
        return [s.strip() for s in sentences if s and s.strip()]

    def sentence_overlap_score(self, query: str, sentence: str, chunk_score: float) -> float:
        q_tokens = set(self.tokenize(query))
        s_tokens = set(self.tokenize(sentence))
        if not q_tokens or not s_tokens:
            overlap = 0.0
        else:
            overlap = float(len(q_tokens.intersection(s_tokens))) / float(len(q_tokens))
        return (0.7 * overlap) + (0.3 * float(chunk_score))

    def candidate_overlap(self, query: str, candidate: str) -> float:
        q_tokens = set(self.tokenize(query))
        c_tokens = set(self.tokenize(candidate))
        if not q_tokens or not c_tokens:
            return 0.0
        return float(len(q_tokens.intersection(c_tokens))) / float(len(q_tokens))

    def text_overlap(self, left: str, right: str) -> float:
        left_tokens = set(self.tokenize(left))
        right_tokens = set(self.tokenize(right))
        if not left_tokens or not right_tokens:
            return 0.0
        return float(len(left_tokens.intersection(right_tokens))) / float(len(left_tokens))

    def generate_spans(self, sentence: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'&._/-]*", sentence)
        spans: List[str] = []
        if not tokens:
            return spans
        max_len = max(self.span_min_tokens, self.span_max_tokens)
        for start in range(len(tokens)):
            for length in range(self.span_min_tokens, max_len + 1):
                end = start + length
                if end > len(tokens):
                    break
                span = self.normalize_space(" ".join(tokens[start:end]).strip(".,;:!?"))
                if span:
                    spans.append(span)
        return spans

    def infer_answer_intent(self, query: str) -> str:
        if not self.intent_extraction_enabled:
            return "generic"
        q_tokens = {tok for tok in self.tokenize(query)}
        if q_tokens.intersection(self.intent_time_keywords):
            return "time"
        if q_tokens.intersection(self.intent_number_keywords):
            return "number"
        if q_tokens.intersection(self.intent_location_keywords):
            return "location"
        if q_tokens.intersection(self.intent_name_keywords):
            return "name"
        return "generic"

    def extract_intent_candidates(self, sentence: str, intent: str) -> List[str]:
        candidates: List[str] = []
        for matched in re.findall(r"\"([^\"]{2,80})\"", sentence):
            c = self.normalize_space(str(matched))
            if c:
                candidates.append(c)

        if intent == "time":
            for pattern in self.intent_time_patterns:
                for m in pattern.findall(sentence):
                    c = self.normalize_space(str(m).strip(".,;:!?"))
                    if c:
                        candidates.append(c)
        elif intent == "number":
            for pattern in self.intent_number_patterns:
                for m in pattern.findall(sentence):
                    c = self.normalize_space(str(m).strip(".,;:!?"))
                    if c:
                        candidates.append(c)
        elif intent in {"name", "location"}:
            max_extra = max(0, self.intent_capitalized_phrase_max_tokens - 1)
            pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0," + str(max_extra) + r"}\b")
            for m in pattern.findall(sentence):
                c = self.normalize_space(str(m).strip(".,;:!?"))
                if c:
                    candidates.append(c)

        unique: List[str] = []
        for c in candidates:
            low = c.lower()
            if low in self.cand_reject_tokens:
                continue
            if c not in unique:
                unique.append(c)
        return unique

    @staticmethod
    def _extract_copula_spans(sentence: str) -> List[str]:
        spans: List[str] = []
        for pattern in [
            r"\b(?:was|were|is|are)\s+(?:the\s+|a\s+|an\s+|my\s+|your\s+|their\s+|our\s+|his\s+|her\s+)?(.+?)(?:[,.!?;:]|$)",
            r"\b(?:was|were|is|are)\s+(.+?)(?:[,.!?;:]|$)",
        ]:
            for m in re.finditer(pattern, sentence, flags=re.IGNORECASE):
                span = " ".join(str(m.group(1)).split()).strip(" ,.;:!?\"'")
                if span:
                    spans.append(span)
        unique: List[str] = []
        seen: set[str] = set()
        for span in spans:
            key = span.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(span)
        return unique

    def is_noisy_candidate(self, value: str) -> bool:
        if not self.candidate_filter_enabled:
            return False
        normalized = self.normalize_space(value)
        low = normalized.lower()
        if not low:
            return True
        tokens = self.tokenize(low)
        if len(tokens) < self.candidate_filter_min_token_count:
            return True
        if tokens and tokens[0] in self.candidate_filter_role_prefixes:
            return True
        for prefix in self.candidate_filter_reject_prefixes:
            if low.startswith(prefix + " ") or low == prefix:
                return True
        for bad in self.candidate_filter_reject_contains:
            if bad and bad in low:
                return True
        if low.startswith("by the way "):
            return True
        if low in {"by the way", "by the", "the way i"}:
            return True
        return False

    def _query_content_tokens(self, query: str) -> set[str]:
        return {tok for tok in self.tokenize(query) if tok and tok not in self.answer_stopwords}

    def _is_clause_like_candidate(self, value: str) -> bool:
        text = self.normalize_space(value)
        if not text:
            return False
        low = text.lower()
        if re.match(
            r"^(?:i|we|you|he|she|they|it)\b\s+(?:am|are|was|were|have|has|had|got|do|did|can|could|should|would|will)\b",
            low,
        ):
            return True
        if re.match(r"^(?:i|we|you|he|she|they|it)\b", low):
            return len(self.tokenize(low)) >= 4
        return False

    def _matches_intent_shape(self, value: str, intent: str) -> bool:
        text = self.normalize_space(value)
        if not text:
            return False
        low = text.lower()
        tokens = self.tokenize(low)
        if not tokens:
            return False
        if intent == "number":
            if re.search(r"\b\d+(?:\.\d+)?\b", low):
                return True
            if set(tokens).intersection(self.number_words):
                return True
            for pattern in self.intent_number_patterns + self.intent_time_patterns:
                if pattern.search(text):
                    return True
            return False
        if intent == "time":
            for pattern in self.intent_time_patterns:
                if pattern.search(text):
                    return True
            return bool(
                set(tokens).intersection({"today", "yesterday", "tomorrow", "current", "latest"})
            )
        if intent == "location":
            if low.startswith(("in ", "at ", "on ")):
                return True
            if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$", text):
                return True
            return not self._is_clause_like_candidate(text) and len(tokens) <= 5
        if intent == "name":
            if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$", text):
                return True
            return not self._is_clause_like_candidate(text) and len(tokens) <= 5
        return not self._is_clause_like_candidate(text)

    def _answer_shape_score(self, query: str, value: str, intent: str) -> float:
        text = self.normalize_space(value)
        if not text:
            return -1.0
        low = text.lower()
        tokens = self.tokenize(low)
        if not tokens:
            return -1.0
        score = 0.0
        if self._matches_intent_shape(text, intent):
            score += 0.30
        if self._is_clause_like_candidate(text):
            score -= 0.55
        token_count = len(tokens)
        if 1 <= token_count <= self.evidence_candidate_max_tokens:
            score += 0.08
        elif token_count > self.evidence_candidate_max_tokens:
            score -= 0.20
        query_content = self._query_content_tokens(query)
        cand_tokens = set(tokens)
        overlap = len(query_content.intersection(cand_tokens))
        novelty = len(cand_tokens.difference(query_content).difference(self.answer_stopwords))
        if query_content and overlap >= max(2, len(query_content) - 1) and novelty <= 1:
            score -= 0.30
        if intent == "generic" and novelty >= 1 and token_count <= 6:
            score += 0.10
        if intent in {"number", "time"} and not self._matches_intent_shape(text, intent):
            score -= 0.45
        return score

    def collect_evidence_sentences(
        self, query: str, reranked_chunks: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        query_tokens = set(self.tokenize(query))

        def _sentence_noise_penalty(sentence: str, overlap: float) -> float:
            low = self.normalize_space(sentence).lower()
            if not low:
                return 1.0
            penalty = 0.0
            if (
                "large language model" in low
                or "as an ai" in low
                or "i'm just an ai" in low
                or "don't have personal experiences" in low
                or "do not have personal experiences" in low
                or "don't have access" in low
                or "do not have access" in low
            ):
                penalty += 0.38
            if re.search(r"\b(tips?|how to|guide|tutorial|best practices?)\b", low):
                penalty += 0.18
            if low.endswith("?"):
                penalty += 0.10
            if re.search(r"\b(you can|consider|try|should|recommended?)\b", low):
                penalty += 0.08
            tok_count = len(self.tokenize(low))
            if tok_count <= 3 and overlap < 0.40:
                penalty += 0.10
            if overlap >= 0.45:
                penalty *= 0.5
            elif overlap >= 0.30:
                penalty *= 0.75
            return penalty

        top_chunks = sorted(
            reranked_chunks,
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True,
        )[: self.evidence_top_n_chunks]
        evidence: List[Dict[str, object]] = []
        for chunk in top_chunks:
            text = str(chunk.get("text", ""))
            chunk_score = float(chunk.get("score", 0.0))
            for sentence in self.split_sentences(text):
                clipped = sentence[: self.evidence_sentence_max_chars].strip()
                if not clipped:
                    continue
                stoks = set(self.tokenize(clipped))
                overlap = (
                    float(len(query_tokens.intersection(stoks))) / float(len(query_tokens))
                    if query_tokens and stoks
                    else 0.0
                )
                penalty = _sentence_noise_penalty(clipped, overlap)
                if penalty >= 0.55 and overlap < 0.24:
                    continue
                score = self.sentence_overlap_score(query, clipped, chunk_score) - penalty
                if score <= 0.0:
                    continue
                evidence.append(
                    {
                        "text": clipped,
                        "score": score,
                        "topic_id": str(chunk.get("topic_id", "")),
                        "chunk_id": int(chunk.get("chunk_id", 0) or 0),
                        "session_date": str(chunk.get("session_date", "")),
                    }
                )
        evidence.sort(key=lambda x: float(x["score"]), reverse=True)
        return evidence[: self.evidence_top_n_sentences]

    def extract_candidates(
        self, query: str, evidence: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        candidates: Dict[str, Dict[str, object]] = {}
        evidence_size = max(1, len(evidence))
        intent = self.infer_answer_intent(query)

        for idx, item in enumerate(evidence):
            text = str(item.get("text", ""))
            if not text:
                continue
            intent_spans = self.extract_intent_candidates(text, intent)
            sentence_score = float(item.get("score", 0.0))
            generated_spans = self.generate_spans(text)
            copula_spans = self._extract_copula_spans(text) if intent == "generic" else []
            scored_spans: List[tuple[float, str, float, str, float]] = []
            for origin, values in (
                ("intent", intent_spans),
                ("copula", copula_spans),
                ("span", generated_spans),
            ):
                for value in values:
                    if not value:
                        continue
                    if value.lower() in self.cand_reject_tokens:
                        continue
                    if self.is_noisy_candidate(value):
                        continue
                    overlap_score = self.candidate_overlap(query, value)
                    token_count = len(self.tokenize(value))
                    length_bonus = min(0.15, float(max(0, token_count - 1)) * 0.03)
                    answer_shape = self._answer_shape_score(query, value, intent)
                    origin_bonus = (
                        0.18 if origin == "intent" else 0.12 if origin == "copula" else 0.0
                    )
                    ranked = (0.60 * overlap_score) + length_bonus + answer_shape + origin_bonus
                    scored_spans.append((ranked, value, overlap_score, origin, answer_shape))
            dedup_scored: List[tuple[float, str, float, str, float]] = []
            seen_values: set[str] = set()
            for ranked, value, overlap_score, origin, answer_shape in sorted(
                scored_spans,
                key=lambda x: x[0],
                reverse=True,
            ):
                key = self.normalize_space(value).lower()
                if key in seen_values:
                    continue
                seen_values.add(key)
                dedup_scored.append((ranked, value, overlap_score, origin, answer_shape))
            spans = dedup_scored[: self.span_top_n_per_sentence]
            for _, value, overlap_score, origin, answer_shape in spans:
                if overlap_score <= 0.0 and intent == "generic":
                    continue
                position_score = 1.0 / float(1 + idx)
                prev = candidates.get(value)
                support = 1 if prev is None else int(prev.get("support", 1)) + 1
                support_score = float(support) / float(evidence_size)
                total_score = (
                    (0.40 * overlap_score)
                    + (0.25 * sentence_score)
                    + (0.15 * support_score)
                    + (0.05 * position_score)
                    + (0.15 * max(-1.0, min(1.0, answer_shape)))
                )
                if total_score < self.cand_min_score:
                    continue
                if prev is None:
                    candidates[value] = {
                        "text": value,
                        "score": total_score,
                        "support": support,
                        "origin": origin,
                        "answer_shape": answer_shape,
                    }
                else:
                    prev["score"] = max(float(prev["score"]), total_score)
                    prev["support"] = support
                    prev["answer_shape"] = max(
                        float(prev.get("answer_shape", 0.0) or 0.0), answer_shape
                    )
                    if origin in {"intent", "copula"}:
                        prev["origin"] = origin
        ranked = sorted(
            candidates.values(),
            key=lambda x: (
                float(x["score"]),
                float(x.get("answer_shape", 0.0) or 0.0),
                int(x.get("support", 0)),
            ),
            reverse=True,
        )
        return ranked[: self.candidate_top_n]

    def extract_evidence_candidate(
        self,
        query: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
    ) -> Optional[Dict[str, str]]:
        """Extract a concise answer directly from evidence before/after LLM response."""
        if not self.evidence_candidate_enabled:
            return None
        intent = self.infer_answer_intent(query)
        for cand in candidates:
            score = float(cand.get("score", 0.0))
            text = self.normalize_space(str(cand.get("text", "")))
            token_count = len(self.tokenize(text))
            answer_shape = float(cand.get("answer_shape", 0.0) or 0.0)
            if score < self.evidence_candidate_min_score:
                continue
            if token_count == 0 or token_count > self.evidence_candidate_max_tokens:
                continue
            if self.is_noisy_candidate(text):
                continue
            if not self._matches_intent_shape(text, intent):
                continue
            if answer_shape < 0.0:
                continue
            return {
                "answer": text,
                "source": f"candidate_{str(cand.get('origin', 'ranked'))}",
                "score": f"{score:.4f}",
            }

        for item in evidence_sentences:
            score = float(item.get("score", 0.0))
            if score < self.evidence_candidate_min_score:
                continue
            sentence = str(item.get("text", ""))
            if not sentence:
                continue
            spans = self.extract_intent_candidates(sentence, intent)
            for span in spans:
                normalized = self.normalize_space(span)
                token_count = len(self.tokenize(normalized))
                if token_count == 0 or token_count > self.evidence_candidate_max_tokens:
                    continue
                if self.is_noisy_candidate(normalized):
                    continue
                if not self._matches_intent_shape(normalized, intent):
                    continue
                return {
                    "answer": normalized,
                    "source": "intent_span",
                    "score": f"{score:.4f}",
                }
            if intent == "generic":
                for span in self._extract_copula_spans(sentence):
                    normalized = self.normalize_space(span)
                    token_count = len(self.tokenize(normalized))
                    if token_count == 0 or token_count > self.evidence_candidate_max_tokens:
                        continue
                    if self.is_noisy_candidate(normalized):
                        continue
                    if self._is_clause_like_candidate(normalized):
                        continue
                    return {
                        "answer": normalized,
                        "source": "copula_span",
                        "score": f"{score:.4f}",
                    }
        return None
