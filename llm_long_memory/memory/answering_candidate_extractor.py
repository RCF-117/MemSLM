"""Candidate and evidence extraction component for answer pipeline."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class AnswerCandidateExtractor:
    """Encapsulate evidence sentence ranking and candidate span extraction."""

    def __init__(self, answering_cfg: Dict[str, Any]) -> None:
        self.answering_cfg = dict(answering_cfg)
        self.evidence_top_n_chunks = int(self.answering_cfg["evidence_top_n_chunks"])
        self.evidence_top_n_sentences = int(self.answering_cfg["evidence_top_n_sentences"])
        self.evidence_sentence_max_chars = int(self.answering_cfg["evidence_sentence_max_chars"])
        self.candidate_top_n = int(self.answering_cfg["candidate_top_n"])
        self.span_min_tokens = int(self.answering_cfg["span_min_tokens"])
        self.span_max_tokens = int(self.answering_cfg["span_max_tokens"])
        self.span_top_n_per_sentence = int(self.answering_cfg["span_top_n_per_sentence"])
        evidence_candidate_cfg = dict(self.answering_cfg["evidence_candidate"])
        self.evidence_candidate_enabled = bool(evidence_candidate_cfg["enabled"])
        self.evidence_candidate_min_score = float(evidence_candidate_cfg["min_score"])
        self.evidence_candidate_max_tokens = int(evidence_candidate_cfg["max_tokens"])
        candidate_filter_cfg = dict(self.answering_cfg["candidate_filter"])
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

        intent_cfg = dict(self.answering_cfg["intent_extraction"])
        self.intent_extraction_enabled = bool(intent_cfg["enabled"])
        self.intent_time_keywords = {str(x).strip().lower() for x in list(intent_cfg["time_keywords"])}
        self.intent_number_keywords = {
            str(x).strip().lower() for x in list(intent_cfg["number_keywords"])
        }
        self.intent_location_keywords = {
            str(x).strip().lower() for x in list(intent_cfg["location_keywords"])
        }
        self.intent_name_keywords = {str(x).strip().lower() for x in list(intent_cfg["name_keywords"])}
        self.intent_time_patterns = [
            re.compile(str(x), flags=re.IGNORECASE) for x in list(intent_cfg["time_regexes"])
        ]
        self.intent_number_patterns = [
            re.compile(str(x), flags=re.IGNORECASE) for x in list(intent_cfg["number_regexes"])
        ]
        self.intent_capitalized_phrase_max_tokens = int(intent_cfg["capitalized_phrase_max_tokens"])

        scoring_cfg = dict(self.answering_cfg["candidate_scoring"])
        self.cand_min_score = float(scoring_cfg["min_score"])
        self.cand_reject_tokens = {
            str(x).strip().lower() for x in list(scoring_cfg["reject_tokens"])
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
            pattern = re.compile(
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0," + str(max_extra) + r"}\b"
            )
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

    def collect_evidence_sentences(
        self, query: str, reranked_chunks: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
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
                score = self.sentence_overlap_score(query, clipped, chunk_score)
                evidence.append(
                    {
                        "text": clipped,
                        "score": score,
                        "topic_id": str(chunk.get("topic_id", "")),
                        "chunk_id": int(chunk.get("chunk_id", 0)),
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
            scored_spans: List[tuple[float, str, float]] = []
            for value in intent_spans + generated_spans:
                if not value:
                    continue
                if value.lower() in self.cand_reject_tokens:
                    continue
                if self.is_noisy_candidate(value):
                    continue
                overlap_score = self.candidate_overlap(query, value)
                token_count = len(self.tokenize(value))
                length_bonus = min(0.15, float(max(0, token_count - 1)) * 0.03)
                ranked = (0.75 * overlap_score) + length_bonus
                scored_spans.append((ranked, value, overlap_score))
            scored_spans.sort(key=lambda x: x[0], reverse=True)
            spans = scored_spans[: self.span_top_n_per_sentence]
            for _, value, overlap_score in spans:
                if overlap_score <= 0.0 and intent == "generic":
                    continue
                position_score = 1.0 / float(1 + idx)
                prev = candidates.get(value)
                support = 1 if prev is None else int(prev.get("support", 1)) + 1
                support_score = float(support) / float(evidence_size)
                total_score = (
                    (0.55 * overlap_score)
                    + (0.25 * sentence_score)
                    + (0.15 * support_score)
                    + (0.05 * position_score)
                )
                if total_score < self.cand_min_score:
                    continue
                if prev is None:
                    candidates[value] = {"text": value, "score": total_score, "support": support}
                else:
                    prev["score"] = max(float(prev["score"]), total_score)
                    prev["support"] = support
        ranked = sorted(
            candidates.values(),
            key=lambda x: (float(x["score"]), int(x.get("support", 0))),
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
        if candidates:
            top = candidates[0]
            score = float(top.get("score", 0.0))
            text = self.normalize_space(str(top.get("text", "")))
            token_count = len(self.tokenize(text))
            if (
                score >= self.evidence_candidate_min_score
                and token_count > 0
                and token_count <= self.evidence_candidate_max_tokens
                and not self.is_noisy_candidate(text)
            ):
                return {"answer": text, "source": "candidate_top1", "score": f"{score:.4f}"}

        intent = self.infer_answer_intent(query)
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
                    return {
                        "answer": normalized,
                        "source": "copula_span",
                        "score": f"{score:.4f}",
                    }
        return None
