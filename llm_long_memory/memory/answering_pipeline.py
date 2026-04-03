"""Answer decision pipeline for retrieval-grounded responses."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from llm_long_memory.memory.counting_resolver import CountingResolver
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
        self.counting = CountingResolver(dict(self.answering_cfg.get("counting", {})))

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
            spans = intent_spans + self._generate_spans(text)
            if len(spans) > self.span_top_n_per_sentence:
                spans = spans[: self.span_top_n_per_sentence]
            sentence_score = float(item.get("score", 0.0))
            for value in spans:
                if not value:
                    continue
                if value.lower() in self.cand_reject_tokens:
                    continue
                if self._is_noisy_candidate(value):
                    continue
                overlap_score = self._candidate_overlap(query, value)
                position_score = 1.0 / float(1 + idx)
                prev = candidates.get(value)
                support = 1 if prev is None else int(prev.get("support", 1)) + 1
                support_score = float(support) / float(evidence_size)
                total_score = (overlap_score + sentence_score + position_score + support_score) / 4.0
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
            "Return the smallest complete answer phrase from evidence.\n"
            "Keep key qualifiers (for example: each way, round trip, per day).\n"
            "Final answer must be an exact substring of one evidence sentence.\n"
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
        """Allow semantically grounded response via token overlap when exact substring fails."""
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
                    "empty_evidence"
                    if not evidence_sentences
                    else "low_top_evidence_score"
                ),
            }
        if self.response_in_evidence(response, evidence_sentences) or self.response_supported_by_evidence(
            response, evidence_sentences
        ):
            if evidence_candidate is not None:
                candidate_answer = self._normalize_space(
                    str(evidence_candidate.get("answer", ""))
                )
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
        # Only retry with a second LLM call when the first pass returns
        # "Not found..." but evidence confidence is strong (handled above).
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
    ) -> str:
        """Build a strict second-pass prompt for answer extraction from evidence."""
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

    @staticmethod
    def _extract_quoted_options(query: str) -> List[str]:
        return [
            x.strip()
            for x in re.findall(r"'([^']{2,120})'|\"([^\"]{2,120})\"", query)
            for x in x
            if x.strip()
        ]

    @staticmethod
    def _extract_or_options(query: str) -> List[str]:
        # Match common patterns like: "the bike or the car"
        m = re.search(
            r"(?:the\s+)?([a-z0-9][a-z0-9\\s\\-]{1,50}?)\\s+or\\s+(?:the\\s+)?([a-z0-9][a-z0-9\\s\\-]{1,50}?)\\??$",
            query.strip().lower(),
            flags=re.IGNORECASE,
        )
        if not m:
            return []
        left = " ".join(m.group(1).split()).strip(" .,;:!?")
        right = " ".join(m.group(2).split()).strip(" .,;:!?")
        if not left or not right:
            return []
        if len(left.split()) > 8 or len(right.split()) > 8:
            return []
        return [left, right]

    @staticmethod
    def _parse_date_token(token: str) -> Optional[datetime]:
        clean = token.strip().lower().replace(",", "")
        clean = re.sub(r"(\\d)(st|nd|rd|th)\\b", r"\\1", clean)
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

    def _extract_dates_from_text(self, text: str) -> List[datetime]:
        out: List[datetime] = []
        # Reuse existing time regexes for extraction, then parse tokens.
        for pat in self.intent_time_patterns:
            for token in pat.findall(text):
                dt = self._parse_date_token(str(token))
                if dt is not None:
                    out.append(dt)
        return out

    def _choose_temporal_option(
        self,
        query: str,
        evidence_sentences: List[Dict[str, object]],
    ) -> Optional[Dict[str, str]]:
        if not self.decision_temporal_choice_enabled:
            return None
        q = query.lower()
        if (" first" not in q) and (" earlier" not in q) and (" before " not in q):
            if (" last" not in q) and (" later" not in q) and (" after " not in q):
                return None
        prefer_earliest = (" first" in q) or (" earlier" in q) or (" before " in q)
        prefer_latest = (" last" in q) or (" later" in q) or (" after " in q)
        options = self._extract_quoted_options(query)
        if len(options) < 2:
            options = self._extract_or_options(query)
        if len(options) < 2:
            return None
        left, right = options[0], options[1]
        left_hits = 0.0
        right_hits = 0.0
        left_dates: List[datetime] = []
        right_dates: List[datetime] = []
        left_mentions = 0
        right_mentions = 0
        for item in evidence_sentences:
            text = str(item.get("text", ""))
            score = float(item.get("score", 0.0))
            low = text.lower()
            if left.lower() in low:
                left_hits += score
                left_mentions += 1
                left_dates.extend(self._extract_dates_from_text(text))
            if right.lower() in low:
                right_hits += score
                right_mentions += 1
                right_dates.extend(self._extract_dates_from_text(text))

        if self.decision_temporal_require_both_options and (
            left_mentions == 0 or right_mentions == 0
        ):
            return None
        if left_hits <= 0.0 and right_hits <= 0.0:
            return None

        # Prefer explicit date comparison when both options have date evidence.
        if left_dates and right_dates:
            left_anchor = min(left_dates) if prefer_earliest or (not prefer_latest) else max(left_dates)
            right_anchor = min(right_dates) if prefer_earliest or (not prefer_latest) else max(right_dates)
            if left_anchor != right_anchor:
                if prefer_latest:
                    answer = left if left_anchor > right_anchor else right
                else:
                    answer = left if left_anchor < right_anchor else right
                return {"answer": answer, "reason": "temporal_choice_by_date"}

        gap = abs(left_hits - right_hits)
        if gap < self.decision_temporal_min_confidence_gap:
            return None
        answer = left if left_hits > right_hits else right
        return {"answer": answer, "reason": "temporal_choice_by_score"}

    def decide_answer(
        self,
        query: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
    ) -> Optional[Dict[str, str]]:
        """Intent-specific decision before LLM generation (counting/temporal only)."""

        counted = self.counting.resolve(query, evidence_sentences, candidates)
        if counted is not None:
            return counted

        temporal = self._choose_temporal_option(query, evidence_sentences)
        if temporal is not None:
            return temporal

        return None
