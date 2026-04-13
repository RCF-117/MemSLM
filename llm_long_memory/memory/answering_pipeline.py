"""Answer decision pipeline for retrieval-grounded responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm_long_memory.memory.answering_candidate_extractor import AnswerCandidateExtractor
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
        self.candidate_extractor = AnswerCandidateExtractor(self.answering_cfg)
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
        # Keep temporal parsing patterns aligned with candidate extractor.
        self.intent_time_patterns = list(self.candidate_extractor.intent_time_patterns)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return AnswerCandidateExtractor.tokenize(text)

    @staticmethod
    def _normalize_space(text: str) -> str:
        return AnswerCandidateExtractor.normalize_space(text)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return AnswerCandidateExtractor.split_sentences(text)

    def _sentence_overlap_score(self, query: str, sentence: str, chunk_score: float) -> float:
        return self.candidate_extractor.sentence_overlap_score(query, sentence, chunk_score)

    def _candidate_overlap(self, query: str, candidate: str) -> float:
        return self.candidate_extractor.candidate_overlap(query, candidate)

    def _text_overlap(self, left: str, right: str) -> float:
        return self.candidate_extractor.text_overlap(left, right)

    def _generate_spans(self, sentence: str) -> List[str]:
        return self.candidate_extractor.generate_spans(sentence)

    def _infer_answer_intent(self, query: str) -> str:
        return self.candidate_extractor.infer_answer_intent(query)

    def _extract_intent_candidates(self, sentence: str, intent: str) -> List[str]:
        return self.candidate_extractor.extract_intent_candidates(sentence, intent)

    def _is_noisy_candidate(self, value: str) -> bool:
        return self.candidate_extractor.is_noisy_candidate(value)

    def collect_evidence_sentences(
        self, query: str, reranked_chunks: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        return self.candidate_extractor.collect_evidence_sentences(query, reranked_chunks)

    def extract_candidates(
        self, query: str, evidence: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        return self.candidate_extractor.extract_candidates(query, evidence)

    def extract_evidence_candidate(
        self,
        query: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
    ) -> Optional[Dict[str, str]]:
        return self.candidate_extractor.extract_evidence_candidate(
            query,
            evidence_sentences,
            candidates,
        )

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
