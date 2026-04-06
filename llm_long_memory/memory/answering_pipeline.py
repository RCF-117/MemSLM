"""Answer decision pipeline for retrieval-grounded responses."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from llm_long_memory.memory.counting_resolver import CountingResolver
from llm_long_memory.memory.answering_response import AnswerResponseHandler
from llm_long_memory.memory.answering_temporal import (
    choose_temporal_option,
    extract_dates_from_text,
    parse_choice_query,
    parse_session_date,
)
from llm_long_memory.utils.logger import logger


class AnsweringPipeline:
    """Encapsulate evidence extraction, candidate scoring, and answer fallback."""

    def __init__(self, answering_cfg: Dict[str, Any]) -> None:
        self.answering_cfg = dict(answering_cfg)
        self.answer_context_only = bool(self.answering_cfg["context_only"])
        self.log_decision_details = bool(self.answering_cfg["log_decision_details"])
        self.evidence_top_n_chunks = int(self.answering_cfg["evidence_top_n_chunks"])
        self.evidence_top_n_sentences = int(self.answering_cfg["evidence_top_n_sentences"])
        self.evidence_sentence_max_chars = int(self.answering_cfg["evidence_sentence_max_chars"])
        self.candidate_top_n = int(self.answering_cfg["candidate_top_n"])
        self.short_circuit_enabled = bool(self.answering_cfg["short_circuit_enabled"])
        self.short_circuit_min_sentence_score = float(
            self.answering_cfg["short_circuit_min_sentence_score"]
        )
        self.llm_fallback_to_top_candidate = bool(
            self.answering_cfg["llm_fallback_to_top_candidate"]
        )
        self.fallback_min_score = float(self.answering_cfg["fallback_min_score"])
        self.response_evidence_min_token_overlap = float(
            self.answering_cfg["response_evidence_min_token_overlap"]
        )
        self.response_evidence_min_shared_tokens = int(
            self.answering_cfg["response_evidence_min_shared_tokens"]
        )
        self.not_found_top_evidence_score_threshold = float(
            self.answering_cfg["not_found_top_evidence_score_threshold"]
        )
        self.second_pass_llm_enabled = bool(self.answering_cfg["second_pass_llm_enabled"])
        self.second_pass_use_evidence_candidate = bool(
            self.answering_cfg["second_pass_use_evidence_candidate"]
        )
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
        decision_cfg = dict(self.answering_cfg.get("decision", {}))
        self.decision_temporal_choice_enabled = bool(
            decision_cfg.get("temporal_choice_enabled", True)
        )
        self.decision_temporal_min_confidence_gap = float(
            decision_cfg.get("temporal_min_confidence_gap", 0.0)
        )
        self.decision_temporal_require_both_options = bool(
            decision_cfg.get("temporal_require_both_options", True)
        )
        self.decision_temporal_overlap_floor = float(
            decision_cfg["temporal_overlap_floor"]
        )
        self.decision_temporal_contains_bonus = float(
            decision_cfg["temporal_contains_bonus"]
        )
        self.decision_temporal_date_bonus = float(
            decision_cfg["temporal_date_bonus"]
        )
        self.decision_temporal_event_anchor_enabled = bool(
            decision_cfg["temporal_event_anchor_enabled"]
        )
        self.decision_temporal_event_anchor_min_overlap = float(
            decision_cfg["temporal_event_anchor_min_overlap"]
        )
        self.decision_temporal_event_anchor_min_score = float(
            decision_cfg["temporal_event_anchor_min_score"]
        )
        self.decision_temporal_event_anchor_pair_min_score = float(
            decision_cfg["temporal_event_anchor_pair_min_score"]
        )
        self.decision_temporal_event_anchor_use_session_date_fallback = bool(
            decision_cfg["temporal_event_anchor_use_session_date_fallback"]
        )
        self.decision_temporal_event_anchor_fallback_to_sentence_score = bool(
            decision_cfg["temporal_event_anchor_fallback_to_sentence_score"]
        )
        self.decision_option_chain_enabled = bool(decision_cfg["option_chain_enabled"])
        self.decision_option_chain_max_options = int(decision_cfg["option_chain_max_options"])
        self.decision_option_chain_default_target_k = int(
            decision_cfg["option_chain_default_target_k"]
        )
        self.decision_option_chain_min_option_overlap = float(
            decision_cfg["option_chain_min_option_overlap"]
        )
        self.decision_option_chain_top_evidence_per_option = int(
            decision_cfg["option_chain_top_evidence_per_option"]
        )
        self.decision_option_chain_top_time_evidence_per_option = int(
            decision_cfg["option_chain_top_time_evidence_per_option"]
        )
        self.decision_option_chain_top_candidates_per_option = int(
            decision_cfg["option_chain_top_candidates_per_option"]
        )
        self.decision_temporal_delegate_to_llm_when_option_chain = bool(
            decision_cfg["temporal_delegate_to_llm_when_option_chain"]
        )
        self.not_found_force_evidence_candidate_when_available = bool(
            self.answering_cfg["not_found_force_evidence_candidate_when_available"]
        )
        post_cfg = dict(self.answering_cfg["postprocess"])
        self.postprocess_enabled = bool(post_cfg["enabled"])
        self.postprocess_strip_prefixes = [
            str(x).strip().lower() for x in list(post_cfg["strip_prefixes"])
        ]
        self.postprocess_issue_with_pattern_enabled = bool(
            post_cfg["issue_with_pattern_enabled"]
        )
        self.counting = CountingResolver(dict(self.answering_cfg.get("counting", {})))
        self.response_handler = AnswerResponseHandler(
            answer_context_only=self.answer_context_only,
            llm_fallback_to_top_candidate=self.llm_fallback_to_top_candidate,
            fallback_min_score=self.fallback_min_score,
            response_evidence_min_token_overlap=self.response_evidence_min_token_overlap,
            response_evidence_min_shared_tokens=self.response_evidence_min_shared_tokens,
            not_found_top_evidence_score_threshold=self.not_found_top_evidence_score_threshold,
            second_pass_llm_enabled=self.second_pass_llm_enabled,
            second_pass_use_evidence_candidate=self.second_pass_use_evidence_candidate,
            not_found_force_evidence_candidate_when_available=self.not_found_force_evidence_candidate_when_available,
            postprocess_enabled=self.postprocess_enabled,
            postprocess_strip_prefixes=self.postprocess_strip_prefixes,
            postprocess_issue_with_pattern_enabled=self.postprocess_issue_with_pattern_enabled,
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text).lower())

    @staticmethod
    def _normalize_space(text: str) -> str:
        return " ".join(str(text).split())

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", str(text))
        return [s.strip() for s in sentences if s and s.strip()]

    def _sentence_overlap_score(self, query: str, sentence: str, chunk_score: float) -> float:
        q_tokens = set(self._tokenize(query))
        s_tokens = set(self._tokenize(sentence))
        if not q_tokens or not s_tokens:
            overlap = 0.0
        else:
            overlap = float(len(q_tokens.intersection(s_tokens))) / float(len(q_tokens))
        return (0.7 * overlap) + (0.3 * float(chunk_score))

    def _candidate_overlap(self, query: str, candidate: str) -> float:
        q_tokens = set(self._tokenize(query))
        c_tokens = set(self._tokenize(candidate))
        if not q_tokens or not c_tokens:
            return 0.0
        return float(len(q_tokens.intersection(c_tokens))) / float(len(q_tokens))

    def _text_overlap(self, left: str, right: str) -> float:
        left_tokens = set(self._tokenize(left))
        right_tokens = set(self._tokenize(right))
        if not left_tokens or not right_tokens:
            return 0.0
        return float(len(left_tokens.intersection(right_tokens))) / float(len(left_tokens))

    def _generate_spans(self, sentence: str) -> List[str]:
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
                span = self._normalize_space(" ".join(tokens[start:end]).strip(".,;:!?"))
                if span:
                    spans.append(span)
        return spans

    def _infer_answer_intent(self, query: str) -> str:
        if not self.intent_extraction_enabled:
            return "generic"
        q_tokens = {tok for tok in self._tokenize(query)}
        if q_tokens.intersection(self.intent_time_keywords):
            return "time"
        if q_tokens.intersection(self.intent_number_keywords):
            return "number"
        if q_tokens.intersection(self.intent_location_keywords):
            return "location"
        if q_tokens.intersection(self.intent_name_keywords):
            return "name"
        return "generic"

    def _extract_intent_candidates(self, sentence: str, intent: str) -> List[str]:
        candidates: List[str] = []
        for matched in re.findall(r"\"([^\"]{2,80})\"", sentence):
            c = self._normalize_space(str(matched))
            if c:
                candidates.append(c)

        if intent == "time":
            for pattern in self.intent_time_patterns:
                for m in pattern.findall(sentence):
                    c = self._normalize_space(str(m).strip(".,;:!?"))
                    if c:
                        candidates.append(c)
        elif intent == "number":
            for pattern in self.intent_number_patterns:
                for m in pattern.findall(sentence):
                    c = self._normalize_space(str(m).strip(".,;:!?"))
                    if c:
                        candidates.append(c)
        elif intent in {"name", "location"}:
            max_extra = max(0, self.intent_capitalized_phrase_max_tokens - 1)
            pattern = re.compile(
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0," + str(max_extra) + r"}\b"
            )
            for m in pattern.findall(sentence):
                c = self._normalize_space(str(m).strip(".,;:!?"))
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

    def _is_noisy_candidate(self, value: str) -> bool:
        if not self.candidate_filter_enabled:
            return False
        normalized = self._normalize_space(value)
        low = normalized.lower()
        if not low:
            return True
        tokens = self._tokenize(low)
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
            for sentence in self._split_sentences(text):
                clipped = sentence[: self.evidence_sentence_max_chars].strip()
                if not clipped:
                    continue
                score = self._sentence_overlap_score(query, clipped, chunk_score)
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
        intent = self._infer_answer_intent(query)

        for idx, item in enumerate(evidence):
            text = str(item.get("text", ""))
            if not text:
                continue
            intent_spans = self._extract_intent_candidates(text, intent)
            sentence_score = float(item.get("score", 0.0))
            generated_spans = self._generate_spans(text)
            scored_spans: List[tuple[float, str, float]] = []
            for value in intent_spans + generated_spans:
                if not value:
                    continue
                if value.lower() in self.cand_reject_tokens:
                    continue
                if self._is_noisy_candidate(value):
                    continue
                overlap_score = self._candidate_overlap(query, value)
                token_count = len(self._tokenize(value))
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
            text = self._normalize_space(str(top.get("text", "")))
            token_count = len(self._tokenize(text))
            if (
                score >= self.evidence_candidate_min_score
                and token_count > 0
                and token_count <= self.evidence_candidate_max_tokens
                and not self._is_noisy_candidate(text)
            ):
                return {"answer": text, "source": "candidate_top1", "score": f"{score:.4f}"}

        intent = self._infer_answer_intent(query)
        for item in evidence_sentences:
            score = float(item.get("score", 0.0))
            if score < self.evidence_candidate_min_score:
                continue
            sentence = str(item.get("text", ""))
            if not sentence:
                continue
            spans = self._extract_intent_candidates(sentence, intent)
            for span in spans:
                normalized = self._normalize_space(span)
                token_count = len(self._tokenize(normalized))
                if token_count == 0 or token_count > self.evidence_candidate_max_tokens:
                    continue
                if self._is_noisy_candidate(normalized):
                    continue
                return {
                    "answer": normalized,
                    "source": "intent_span",
                    "score": f"{score:.4f}",
                }
        return None

    def log_decision_snapshot(
        self,
        query: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
    ) -> None:
        if not self.log_decision_details:
            return
        top_evidence = [
            {
                "score": round(float(item.get("score", 0.0)), 4),
                "text": str(item.get("text", ""))[:120],
            }
            for item in evidence_sentences[:3]
        ]
        top_candidates = [
            {
                "text": str(item.get("text", "")),
                "score": round(float(item.get("score", 0.0)), 4),
                "support": int(item.get("support", 0)),
            }
            for item in candidates[:3]
        ]
        logger.info(
            "MemoryManager.decision: "
            f"query='{query[:120]}', top_evidence={top_evidence}, top_candidates={top_candidates}"
        )

    def maybe_short_circuit(
        self,
        candidates: List[Dict[str, object]],
        evidence_sentences: List[Dict[str, object]],
    ) -> Optional[str]:
        if (
            self.short_circuit_enabled
            and candidates
            and evidence_sentences
            and float(evidence_sentences[0]["score"]) >= self.short_circuit_min_sentence_score
        ):
            return str(candidates[0]["text"])
        return None

    def build_answer_prompt(
        self,
        input_text: str,
        retrieved_context: str,
        recent_context: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        option_evidence_chains: Optional[Dict[str, object]] = None,
    ) -> str:
        return self.response_handler.build_answer_prompt(
            input_text=input_text,
            retrieved_context=retrieved_context,
            recent_context=recent_context,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            option_evidence_chains=option_evidence_chains,
        )

    def response_in_evidence(self, response: str, evidence_sentences: List[Dict[str, object]]) -> bool:
        return self.response_handler.response_in_evidence(response, evidence_sentences)

    def response_supported_by_evidence(
        self, response: str, evidence_sentences: List[Dict[str, object]]
    ) -> bool:
        return self.response_handler.response_supported_by_evidence(
            response, evidence_sentences
        )

    def evaluate_response_fallback(
        self,
        response: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        evidence_candidate: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        return self.response_handler.evaluate_response_fallback(
            response=response,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            evidence_candidate=evidence_candidate,
        )

    def apply_response_fallback(
        self,
        response: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        evidence_candidate: Optional[Dict[str, str]] = None,
    ) -> str:
        """Backward-compatible response fallback returning only final answer text."""
        result = self.evaluate_response_fallback(
            response=response,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            evidence_candidate=evidence_candidate,
        )
        return str(result.get("response", ""))

    def build_second_pass_prompt(
        self,
        input_text: str,
        evidence_sentences: List[Dict[str, object]],
        evidence_candidate: Optional[Dict[str, str]],
        option_evidence_chains: Optional[Dict[str, object]] = None,
    ) -> str:
        return self.response_handler.build_second_pass_prompt(
            input_text=input_text,
            evidence_sentences=evidence_sentences,
            evidence_candidate=evidence_candidate,
            option_evidence_chains=option_evidence_chains,
        )

    def build_option_evidence_chains(
        self,
        query: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
    ) -> Optional[Dict[str, object]]:
        """Build independent evidence pools per option for choice-style questions."""
        if not self.decision_option_chain_enabled:
            return None

        parsed = parse_choice_query(
            query,
            max_options=self.decision_option_chain_max_options,
            default_target_k=self.decision_option_chain_default_target_k,
        )
        if parsed is None:
            return None
        options, target_k = parsed

        option_rows: List[Dict[str, object]] = []
        for option in options:
            option_low = str(option).lower()
            evidence_pool: List[Dict[str, object]] = []
            time_pool: List[Dict[str, object]] = []

            for item in evidence_sentences:
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                overlap = self._text_overlap(option_low, text.lower())
                contains = option_low in text.lower()
                if (not contains) and overlap < self.decision_option_chain_min_option_overlap:
                    continue
                sentence_score = float(item.get("score", 0.0))
                option_score = sentence_score * max(self.decision_temporal_overlap_floor, overlap)
                if contains:
                    option_score += self.decision_temporal_contains_bonus * sentence_score
                evidence_pool.append(
                    {
                        "text": text,
                        "score": option_score,
                        "session_date": str(item.get("session_date", "")),
                    }
                )
                parsed_dates = extract_dates_from_text(text, self.intent_time_patterns)
                if not parsed_dates:
                    session_dt = parse_session_date(str(item.get("session_date", "")))
                    if session_dt is not None:
                        parsed_dates = [session_dt]
                for dt in parsed_dates:
                    time_pool.append(
                        {
                            "text": text,
                            "score": option_score,
                            "date": dt.isoformat(),
                        }
                    )

            evidence_pool.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            time_pool.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

            option_candidates: List[Dict[str, object]] = []
            for cand in candidates:
                c_text = str(cand.get("text", "")).strip()
                if not c_text:
                    continue
                overlap = self._text_overlap(option_low, c_text.lower())
                if overlap < self.decision_option_chain_min_option_overlap:
                    continue
                option_candidates.append(
                    {
                        "text": c_text,
                        "score": float(cand.get("score", 0.0)) * overlap,
                        "support": int(cand.get("support", 0)),
                    }
                )
            option_candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

            option_rows.append(
                {
                    "option": option,
                    "time_evidence": time_pool[: self.decision_option_chain_top_time_evidence_per_option],
                    "other_evidence": evidence_pool[: self.decision_option_chain_top_evidence_per_option],
                    "candidates": option_candidates[: self.decision_option_chain_top_candidates_per_option],
                }
            )

        # Require at least two options with usable evidence.
        usable = [
            row
            for row in option_rows
            if row["other_evidence"] or row["time_evidence"] or row["candidates"]
        ]
        if len(usable) < 2:
            return None
        return {
            "selection_target_k": int(target_k),
            "options": usable,
        }

    def decide_answer(
        self,
        query: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        reranked_chunks: Optional[List[Dict[str, object]]] = None,
        option_evidence_chains: Optional[Dict[str, object]] = None,
    ) -> Optional[Dict[str, str]]:
        """Intent-specific decision before LLM generation (counting/temporal only)."""

        counted = self.counting.resolve(
            query,
            evidence_sentences,
            candidates,
            reranked_chunks=reranked_chunks or [],
        )
        if counted is not None:
            return counted

        if not (
            self.decision_temporal_delegate_to_llm_when_option_chain
            and option_evidence_chains is not None
        ):
            temporal = choose_temporal_option(
                query=query,
                evidence_sentences=evidence_sentences,
                enabled=self.decision_temporal_choice_enabled,
                min_confidence_gap=self.decision_temporal_min_confidence_gap,
                require_both_options=self.decision_temporal_require_both_options,
                time_patterns=self.intent_time_patterns,
                overlap_floor=self.decision_temporal_overlap_floor,
                contains_bonus=self.decision_temporal_contains_bonus,
                date_bonus=self.decision_temporal_date_bonus,
                event_anchor_enabled=self.decision_temporal_event_anchor_enabled,
                event_anchor_min_overlap=self.decision_temporal_event_anchor_min_overlap,
                event_anchor_min_score=self.decision_temporal_event_anchor_min_score,
                event_anchor_pair_min_score=self.decision_temporal_event_anchor_pair_min_score,
                event_anchor_use_session_date_fallback=self.decision_temporal_event_anchor_use_session_date_fallback,
                event_anchor_fallback_to_sentence_score=self.decision_temporal_event_anchor_fallback_to_sentence_score,
            )
            if temporal is not None:
                return temporal

        return None

    def postprocess_final_answer(
        self,
        answer: str,
        query: str,
        evidence_candidate: Optional[Dict[str, str]] = None,
    ) -> str:
        return self.response_handler.postprocess_final_answer(
            answer=answer,
            query=query,
            evidence_candidate=evidence_candidate,
        )
