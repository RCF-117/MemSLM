"""Response guard and post-processing for retrieval-grounded answering."""

from __future__ import annotations

import re
from typing import Dict, List, Optional


class AnswerResponseHandler:
    """Handle prompt construction, evidence support checks, and fallback decisions."""

    def __init__(
        self,
        *,
        answer_context_only: bool,
        llm_fallback_to_top_candidate: bool,
        fallback_min_score: float,
        response_evidence_min_token_overlap: float,
        response_evidence_min_shared_tokens: int,
        not_found_top_evidence_score_threshold: float,
        second_pass_llm_enabled: bool,
        second_pass_use_evidence_candidate: bool,
        not_found_force_evidence_candidate_when_available: bool,
        postprocess_enabled: bool,
        postprocess_strip_prefixes: List[str],
        postprocess_issue_with_pattern_enabled: bool,
    ) -> None:
        self.answer_context_only = bool(answer_context_only)
        self.llm_fallback_to_top_candidate = bool(llm_fallback_to_top_candidate)
        self.fallback_min_score = float(fallback_min_score)
        self.response_evidence_min_token_overlap = float(response_evidence_min_token_overlap)
        self.response_evidence_min_shared_tokens = int(response_evidence_min_shared_tokens)
        self.not_found_top_evidence_score_threshold = float(
            not_found_top_evidence_score_threshold
        )
        self.second_pass_llm_enabled = bool(second_pass_llm_enabled)
        self.second_pass_use_evidence_candidate = bool(second_pass_use_evidence_candidate)
        self.not_found_force_evidence_candidate_when_available = bool(
            not_found_force_evidence_candidate_when_available
        )
        self.postprocess_enabled = bool(postprocess_enabled)
        self.postprocess_strip_prefixes = [
            str(x).strip().lower() for x in postprocess_strip_prefixes
        ]
        self.postprocess_issue_with_pattern_enabled = bool(postprocess_issue_with_pattern_enabled)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text).lower())

    @staticmethod
    def _normalize_space(text: str) -> str:
        return " ".join(str(text).split())

    def build_answer_prompt(
        self,
        input_text: str,
        retrieved_context: str,
        recent_context: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
    ) -> str:
        evidence_text = "\n".join(
            f"- {str(item.get('text', ''))}" for item in evidence_sentences
        )
        candidate_text = "\n".join(
            f"- {str(item.get('text', ''))} (score={float(item.get('score', 0.0)):.2f})"
            for item in candidates
        )
        rules = (
            "Answer using only the retrieved context and evidence sentences.\n"
            "If the answer is not in the evidence, say exactly: Not found in retrieved context.\n"
            "Keep key qualifiers (for example: each way, round trip, per day).\n"
            "Final answer must be answer of query.\n"
            "Return only the final answer."
            if self.answer_context_only
            else "Return only the final answer."
        )
        return (
            "[Retrieved Context]\n"
            f"{retrieved_context}\n\n"
            "[Evidence Sentences]\n"
            f"{evidence_text}\n\n"
            "[Candidate Answers]\n"
            f"{candidate_text}\n\n"
            "[Recent Context]\n"
            f"{recent_context}\n\n"
            "[Answer Rules]\n"
            f"{rules}\n\n"
            f"User: {input_text}"
        )

    def response_in_evidence(self, response: str, evidence_sentences: List[Dict[str, object]]) -> bool:
        ans = self._normalize_space(response).lower()
        if not ans:
            return False
        for item in evidence_sentences:
            sentence = self._normalize_space(str(item.get("text", ""))).lower()
            if ans in sentence:
                return True
        return False

    def response_supported_by_evidence(
        self, response: str, evidence_sentences: List[Dict[str, object]]
    ) -> bool:
        response_tokens = self._tokenize(response)
        if not response_tokens:
            return False
        response_token_set = set(response_tokens)
        response_token_len = float(max(1, len(response_token_set)))
        for item in evidence_sentences:
            sentence_tokens = set(self._tokenize(str(item.get("text", ""))))
            if not sentence_tokens:
                continue
            shared = response_token_set.intersection(sentence_tokens)
            shared_count = len(shared)
            overlap_ratio = float(shared_count) / response_token_len
            if (
                shared_count >= self.response_evidence_min_shared_tokens
                and overlap_ratio >= self.response_evidence_min_token_overlap
            ):
                return True
        return False

    def evaluate_response_fallback(
        self,
        response: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        evidence_candidate: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        if not self.answer_context_only:
            return {"response": response, "fallback_path": "context_free"}
        top_evidence_score = (
            float(evidence_sentences[0].get("score", 0.0)) if evidence_sentences else 0.0
        )
        normalized_response = self._normalize_space(response).lower()
        if normalized_response == "not found in retrieved context.":
            if (
                self.not_found_force_evidence_candidate_when_available
                and evidence_candidate is not None
            ):
                return {
                    "response": evidence_candidate["answer"],
                    "fallback_path": "not_found_to_evidence_candidate",
                    "not_found_reason": "candidate_available",
                }
            if (
                evidence_sentences
                and top_evidence_score >= self.not_found_top_evidence_score_threshold
            ):
                if self.second_pass_llm_enabled:
                    return {
                        "response": response,
                        "fallback_path": "retry_due_to_guarded_not_found",
                        "not_found_reason": "guarded_by_high_evidence_score",
                    }
                if evidence_candidate is not None:
                    return {
                        "response": evidence_candidate["answer"],
                        "fallback_path": "guarded_not_found_to_evidence_candidate",
                        "not_found_reason": "guarded_by_high_evidence_score",
                    }
            return {
                "response": response,
                "fallback_path": "llm_not_found_accepted",
                "not_found_reason": (
                    "empty_evidence" if not evidence_sentences else "low_top_evidence_score"
                ),
            }
        if self.response_in_evidence(response, evidence_sentences) or self.response_supported_by_evidence(
            response, evidence_sentences
        ):
            if evidence_candidate is not None:
                candidate_answer = self._normalize_space(str(evidence_candidate.get("answer", "")))
                candidate_low = candidate_answer.lower()
                response_low = self._normalize_space(response).lower()
                if candidate_answer and candidate_low in response_low:
                    candidate_tokens = len(self._tokenize(candidate_answer))
                    response_tokens = len(self._tokenize(response))
                    if response_tokens > candidate_tokens:
                        return {
                            "response": candidate_answer,
                            "fallback_path": "compress_supported_response_to_evidence_candidate",
                        }
            return {"response": response, "fallback_path": "llm_supported_by_evidence"}
        if self.second_pass_use_evidence_candidate and evidence_candidate is not None:
            return {
                "response": evidence_candidate["answer"],
                "fallback_path": "fallback_to_evidence_candidate",
            }
        if (
            self.llm_fallback_to_top_candidate
            and candidates
            and float(candidates[0]["score"]) >= self.fallback_min_score
        ):
            fallback = str(candidates[0]["text"])
            return {"response": fallback, "fallback_path": "fallback_to_top_candidate"}
        return {
            "response": "Not found in retrieved context.",
            "fallback_path": "fallback_to_not_found",
            "not_found_reason": "llm_response_not_supported_and_no_fallback",
        }

    def build_second_pass_prompt(
        self,
        input_text: str,
        evidence_sentences: List[Dict[str, object]],
        evidence_candidate: Optional[Dict[str, str]],
    ) -> str:
        evidence_text = "\n".join(f"- {str(item.get('text', ''))}" for item in evidence_sentences)
        candidate_text = (
            str(evidence_candidate.get("answer", "")) if evidence_candidate is not None else ""
        )
        guidance = (
            "You must answer using only the evidence sentences.\n"
            "Do not say Not found unless evidence is empty.\n"
            "Prefer the shortest exact phrase from evidence.\n"
            "Return only the final answer."
        )
        if candidate_text:
            guidance += f"\nPreferred evidence candidate: {candidate_text}"
        return (
            "[Evidence Sentences]\n"
            f"{evidence_text}\n\n"
            "[Rules]\n"
            f"{guidance}\n\n"
            f"Question: {input_text}"
        )

    def postprocess_final_answer(
        self,
        answer: str,
        query: str,
        evidence_candidate: Optional[Dict[str, str]] = None,
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
        if evidence_candidate is not None:
            candidate_answer = self._normalize_space(str(evidence_candidate.get("answer", "")))
            if candidate_answer:
                c_low = candidate_answer.lower()
                if c_low in out.lower() and len(self._tokenize(out)) > len(self._tokenize(candidate_answer)):
                    out = candidate_answer
        _ = query
        return out
