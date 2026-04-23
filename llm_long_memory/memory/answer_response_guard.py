"""Response guard and post-processing for retrieval-grounded answering."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


class AnswerResponseGuard:
    """Handle evidence support checks, guarded retries, and answer normalization."""

    def __init__(
        self,
        *,
        answer_context_only: bool,
        response_evidence_min_token_overlap: float,
        response_evidence_min_shared_tokens: int,
        response_evidence_relaxed_overlap_enabled: bool,
        response_evidence_relaxed_min_token_overlap: float,
        response_evidence_relaxed_min_shared_tokens: int,
        not_found_top_evidence_score_threshold: float,
        second_pass_llm_enabled: bool,
        postprocess_enabled: bool,
        postprocess_strip_prefixes: List[str],
        postprocess_issue_with_pattern_enabled: bool,
    ) -> None:
        self.answer_context_only = bool(answer_context_only)
        self.response_evidence_min_token_overlap = float(
            response_evidence_min_token_overlap
        )
        self.response_evidence_min_shared_tokens = int(response_evidence_min_shared_tokens)
        self.response_evidence_relaxed_overlap_enabled = bool(
            response_evidence_relaxed_overlap_enabled
        )
        self.response_evidence_relaxed_min_token_overlap = float(
            response_evidence_relaxed_min_token_overlap
        )
        self.response_evidence_relaxed_min_shared_tokens = int(
            response_evidence_relaxed_min_shared_tokens
        )
        self.not_found_top_evidence_score_threshold = float(
            not_found_top_evidence_score_threshold
        )
        self.second_pass_llm_enabled = bool(second_pass_llm_enabled)
        self.postprocess_enabled = bool(postprocess_enabled)
        self.postprocess_strip_prefixes = [
            str(x).strip().lower() for x in postprocess_strip_prefixes
        ]
        self.postprocess_issue_with_pattern_enabled = bool(
            postprocess_issue_with_pattern_enabled
        )

    @property
    def _effective_overlap_threshold(self) -> Tuple[float, int]:
        if self.response_evidence_relaxed_overlap_enabled:
            return (
                float(self.response_evidence_relaxed_min_token_overlap),
                int(self.response_evidence_relaxed_min_shared_tokens),
            )
        return (
            float(self.response_evidence_min_token_overlap),
            int(self.response_evidence_min_shared_tokens),
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text).lower())

    @staticmethod
    def _normalize_space(text: str) -> str:
        return " ".join(str(text).split())

    def response_in_evidence(self, response: str, evidence_sentences: List[Dict[str, object]]) -> bool:
        return self.response_in_support_sources(response, evidence_sentences)

    def response_in_support_sources(
        self,
        response: str,
        support_sources: List[Dict[str, object]],
    ) -> bool:
        ans = self._normalize_space(response).lower()
        if not ans:
            return False
        for item in support_sources:
            sentence = self._normalize_space(str(item.get("text", ""))).lower()
            if ans in sentence:
                return True
        return False

    def response_supported_by_evidence(
        self, response: str, evidence_sentences: List[Dict[str, object]]
    ) -> bool:
        return self.response_supported_by_sources(response, evidence_sentences)

    def response_supported_by_sources(
        self,
        response: str,
        support_sources: List[Dict[str, object]],
    ) -> bool:
        response_tokens = self._tokenize(response)
        if not response_tokens:
            return False
        response_token_set = set(response_tokens)
        response_token_len = float(max(1, len(response_token_set)))
        min_overlap, min_shared = self._effective_overlap_threshold
        for item in support_sources:
            sentence_tokens = set(self._tokenize(str(item.get("text", ""))))
            if not sentence_tokens:
                continue
            shared = response_token_set.intersection(sentence_tokens)
            shared_count = len(shared)
            overlap_ratio = float(shared_count) / response_token_len
            if shared_count >= min_shared and overlap_ratio >= min_overlap:
                return True
        return False

    def _single_event_high_confidence_supported(
        self,
        response: str,
        support_sources: List[Dict[str, object]],
    ) -> bool:
        response_tokens = self._tokenize(response)
        if not response_tokens:
            return False
        response_token_set = set(response_tokens)
        token_count = len(response_token_set)
        if token_count < 2 or token_count > 8:
            return False
        response_token_len = float(max(1, token_count))
        for item in support_sources:
            section = str(item.get("section", "")).strip().lower()
            bucket = str(item.get("bucket", "")).strip().lower()
            score = float(item.get("score", 0.0) or 0.0)
            if section in {"light_graph", "evidence_pack"}:
                trusted = True
            elif section == "filtered_evidence" and bucket in {"core", "support"}:
                trusted = score >= 0.80
            else:
                trusted = False
            if not trusted:
                continue
            sentence_tokens = set(self._tokenize(str(item.get("text", ""))))
            if not sentence_tokens:
                continue
            shared = response_token_set.intersection(sentence_tokens)
            shared_count = len(shared)
            overlap_ratio = float(shared_count) / response_token_len
            if shared_count >= 2 and overlap_ratio >= 0.34:
                return True
        return False

    def evaluate_response_guard(
        self,
        response: str,
        evidence_sentences: List[Dict[str, object]],
        support_sources: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, str]:
        if not self.answer_context_only:
            return {"response": response, "fallback_path": "context_free"}
        active_support_sources = list(evidence_sentences)
        for item in list(support_sources or []):
            text = self._normalize_space(str(item.get("text", "")))
            if not text:
                continue
            active_support_sources.append(dict(item))
        top_evidence_score = (
            max(
                (
                    float(item.get("score", 0.0) or 0.0)
                    for item in active_support_sources
                ),
                default=0.0,
            )
        )
        normalized_response = self._normalize_space(response).lower()
        if normalized_response == "not found in retrieved context.":
            if (
                active_support_sources
                and top_evidence_score >= self.not_found_top_evidence_score_threshold
            ):
                if self.second_pass_llm_enabled:
                    return {
                        "response": response,
                        "fallback_path": "retry_due_to_guarded_not_found",
                        "not_found_reason": "guarded_by_high_evidence_score",
                    }
            return {
                "response": response,
                "fallback_path": "llm_not_found_accepted",
                "not_found_reason": (
                    "empty_evidence" if not active_support_sources else "low_top_evidence_score"
                ),
            }
        if self.response_in_support_sources(response, active_support_sources) or self.response_supported_by_sources(
            response, active_support_sources
        ):
            return {"response": response, "fallback_path": "llm_supported_by_evidence"}
        if self._single_event_high_confidence_supported(response, active_support_sources):
            return {
                "response": response,
                "fallback_path": "llm_supported_by_high_confidence_single_event",
            }
        return {
            "response": "Not found in retrieved context.",
            "fallback_path": "fallback_to_not_found",
            "not_found_reason": "llm_response_not_supported_and_no_fallback",
        }

    def build_second_pass_retry_prompt(
        self,
        prompt_text: str,
        first_answer: str = "",
    ) -> str:
        guidance = (
            "Adjudicate the candidate packet against the first answer.\n"
            "If the first answer is supported, return it in the shortest form.\n"
            "If a stronger supported candidate is present, return that candidate.\n"
            "Return Not found in retrieved context only when no candidate is supported.\n"
            "Return only the final answer."
        )
        first_answer = self._normalize_space(first_answer).strip()
        return (
            "[First Answer]\n"
            f"{first_answer or 'Not provided.'}\n\n"
            "[Candidate Evidence Packet]\n"
            f"{prompt_text}\n\n"
            "[Adjudication Task]\n"
            f"{guidance}\n\n"
            "Question: Return the final answer."
        )

    def normalize_final_answer(
        self,
        answer: str,
        query: str,
    ) -> str:
        out = self._normalize_space(answer).strip(" \"'`")
        if not self.postprocess_enabled or not out:
            return out
        low = out.lower()
        for prefix in self.postprocess_strip_prefixes:
            if low.startswith(prefix + " "):
                out = self._normalize_space(out[len(prefix) :]).strip(" ,.:;!?")
                low = out.lower()
        if self.postprocess_issue_with_pattern_enabled:
            match = re.search(r"\bissue\s+with\s+([^,.!?;]+)", out, flags=re.IGNORECASE)
            if match:
                candidate = self._normalize_space(str(match.group(1))).strip(" ,.:;!?")
                if candidate:
                    out = candidate
        _ = query
        return out
