"""Active answer-grounding pipeline for evidence-grounded MemSLM answering."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm_long_memory.memory.evidence_candidate_extractor import EvidenceCandidateExtractor
from llm_long_memory.memory.answer_response_guard import AnswerResponseGuard
from llm_long_memory.utils.logger import logger


class AnswerGroundingPipeline:
    """Encapsulate evidence extraction, candidate scoring, and response guarding."""

    def __init__(self, grounding_cfg: Dict[str, Any]) -> None:
        self.grounding_cfg = dict(grounding_cfg)
        self.answer_context_only = bool(self.grounding_cfg["context_only"])
        self.reasoning_fallback_enabled = bool(
            self.grounding_cfg.get("reasoning_fallback_enabled", True)
        )
        self.log_decision_details = bool(self.grounding_cfg["log_decision_details"])
        self.response_evidence_min_token_overlap = float(
            self.grounding_cfg["response_evidence_min_token_overlap"]
        )
        self.response_evidence_min_shared_tokens = int(
            self.grounding_cfg["response_evidence_min_shared_tokens"]
        )
        self.response_evidence_relaxed_overlap_enabled = bool(
            self.grounding_cfg.get("response_evidence_relaxed_overlap_enabled", False)
        )
        self.response_evidence_relaxed_min_token_overlap = float(
            self.grounding_cfg.get("response_evidence_relaxed_min_token_overlap", 0.25)
        )
        self.response_evidence_relaxed_min_shared_tokens = int(
            self.grounding_cfg.get("response_evidence_relaxed_min_shared_tokens", 1)
        )
        self.not_found_top_evidence_score_threshold = float(
            self.grounding_cfg["not_found_top_evidence_score_threshold"]
        )
        self.second_pass_llm_enabled = bool(self.grounding_cfg["second_pass_llm_enabled"])
        self.candidate_extractor = EvidenceCandidateExtractor(self.grounding_cfg)
        post_cfg = dict(
            self.grounding_cfg.get(
                "postprocess",
                {
                    "enabled": False,
                    "strip_prefixes": [],
                    "issue_with_pattern_enabled": False,
                },
            )
        )
        self.postprocess_enabled = bool(post_cfg["enabled"])
        self.postprocess_strip_prefixes = [
            str(x).strip().lower() for x in list(post_cfg["strip_prefixes"])
        ]
        self.postprocess_issue_with_pattern_enabled = bool(
            post_cfg["issue_with_pattern_enabled"]
        )
        self.response_guard = AnswerResponseGuard(
            answer_context_only=self.answer_context_only,
            response_evidence_min_token_overlap=self.response_evidence_min_token_overlap,
            response_evidence_min_shared_tokens=self.response_evidence_min_shared_tokens,
            response_evidence_relaxed_overlap_enabled=self.response_evidence_relaxed_overlap_enabled,
            response_evidence_relaxed_min_token_overlap=self.response_evidence_relaxed_min_token_overlap,
            response_evidence_relaxed_min_shared_tokens=self.response_evidence_relaxed_min_shared_tokens,
            not_found_top_evidence_score_threshold=self.not_found_top_evidence_score_threshold,
            second_pass_llm_enabled=self.second_pass_llm_enabled,
            postprocess_enabled=self.postprocess_enabled,
            postprocess_strip_prefixes=self.postprocess_strip_prefixes,
            postprocess_issue_with_pattern_enabled=self.postprocess_issue_with_pattern_enabled,
        )
        # Keep temporal parsing patterns aligned with candidate extractor.
        self.intent_time_patterns = list(self.candidate_extractor.intent_time_patterns)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return EvidenceCandidateExtractor.tokenize(text)

    @staticmethod
    def _normalize_space(text: str) -> str:
        return EvidenceCandidateExtractor.normalize_space(text)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return EvidenceCandidateExtractor.split_sentences(text)

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

    def response_in_evidence(self, response: str, evidence_sentences: List[Dict[str, object]]) -> bool:
        return self.response_guard.response_in_evidence(response, evidence_sentences)

    def response_supported_by_evidence(
        self, response: str, evidence_sentences: List[Dict[str, object]]
    ) -> bool:
        return self.response_guard.response_supported_by_evidence(
            response, evidence_sentences
        )

    def response_supported_by_sources(
        self, response: str, support_sources: List[Dict[str, object]]
    ) -> bool:
        return self.response_guard.response_supported_by_sources(
            response, support_sources
        )

    def evaluate_response_guard(
        self,
        response: str,
        evidence_sentences: List[Dict[str, object]],
        support_sources: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, str]:
        return self.response_guard.evaluate_response_guard(
            response=response,
            evidence_sentences=evidence_sentences,
            support_sources=support_sources,
        )

    def apply_response_guard(
        self,
        response: str,
        evidence_sentences: List[Dict[str, object]],
        support_sources: Optional[List[Dict[str, object]]] = None,
    ) -> str:
        """Return only the final guarded answer text."""
        result = self.evaluate_response_guard(
            response=response,
            evidence_sentences=evidence_sentences,
            support_sources=support_sources,
        )
        return str(result.get("response", ""))

    def build_second_pass_retry_prompt(
        self,
        prompt_text: str,
        first_answer: str = "",
    ) -> str:
        return self.response_guard.build_second_pass_retry_prompt(
            prompt_text=prompt_text,
            first_answer=first_answer,
        )

    def normalize_final_answer(
        self,
        answer: str,
        query: str,
    ) -> str:
        return self.response_guard.normalize_final_answer(
            answer=answer,
            query=query,
        )
