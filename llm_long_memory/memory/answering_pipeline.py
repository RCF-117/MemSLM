"""Answer decision pipeline for retrieval-grounded responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm_long_memory.memory.answering_candidate_extractor import AnswerCandidateExtractor
from llm_long_memory.memory.answering_response import AnswerResponseHandler
from llm_long_memory.utils.logger import logger


class AnsweringPipeline:
    """Encapsulate evidence extraction, candidate scoring, and answer fallback."""

    def __init__(self, answering_cfg: Dict[str, Any]) -> None:
        self.answering_cfg = dict(answering_cfg)
        self.answer_context_only = bool(self.answering_cfg["context_only"])
        self.reasoning_fallback_enabled = bool(
            self.answering_cfg.get("reasoning_fallback_enabled", True)
        )
        self.log_decision_details = bool(self.answering_cfg["log_decision_details"])
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
        self.not_found_force_evidence_candidate_when_available = bool(
            self.answering_cfg.get("not_found_force_evidence_candidate_when_available", False)
        )
        post_cfg = dict(
            self.answering_cfg.get(
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

    def build_answer_prompt(
        self,
        input_text: str,
        graph_context: str,
        query_plan: str = "",
        graph_tool_hints: str = "",
        rag_evidence: str = "",
        fallback_answer: str = "",
    ) -> str:
        return self.response_handler.build_answer_prompt(
            input_text=input_text,
            graph_context=graph_context,
            query_plan=query_plan,
            graph_tool_hints=graph_tool_hints,
            rag_evidence=rag_evidence,
            fallback_answer=fallback_answer,
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
        fallback_answer: Optional[str] = None,
    ) -> Dict[str, str]:
        return self.response_handler.evaluate_response_fallback(
            response=response,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            evidence_candidate=evidence_candidate,
            fallback_answer=fallback_answer,
        )

    def apply_response_fallback(
        self,
        response: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        evidence_candidate: Optional[Dict[str, str]] = None,
        fallback_answer: Optional[str] = None,
    ) -> str:
        """Backward-compatible response fallback returning only final answer text."""
        result = self.evaluate_response_fallback(
            response=response,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            evidence_candidate=evidence_candidate,
            fallback_answer=fallback_answer,
        )
        return str(result.get("response", ""))

    def build_second_pass_prompt(
        self,
        prompt_text: str,
        evidence_candidate: Optional[Dict[str, str]],
    ) -> str:
        return self.response_handler.build_second_pass_prompt(
            prompt_text=prompt_text,
            evidence_candidate=evidence_candidate,
        )

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
