"""Chat orchestration runtime extracted from MemoryManager."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from llm_long_memory.memory.prompt_context import PromptContextLimits, build_generation_context
from llm_long_memory.utils.logger import logger


class MemoryManagerChatRuntime:
    """Keep MemoryManager.chat-related orchestration out of the main class body."""

    def __init__(self, manager: Any) -> None:
        self.m = manager

    def prepare_answer_inputs(
        self,
        query: str,
        precomputed_context: Optional[
            Tuple[str, List[Dict[str, object]], List[Dict[str, object]]]
        ],
    ) -> Tuple[
        str,
        List[Dict[str, object]],
        List[Dict[str, object]],
        List[Dict[str, object]],
        List[Dict[str, object]],
        Optional[Dict[str, object]],
        Optional[Dict[str, str]],
        str,
        str,
    ]:
        if precomputed_context is not None:
            retrieved_context, topics, chunks = precomputed_context
        else:
            retrieved_context, topics, chunks = self.m.retrieve_context(query)
        retrieved_ids = [str(topic["topic_id"]) for topic in topics]
        logger.info(f"MemoryManager.chat: retrieved topics={retrieved_ids}")

        evidence_sentences = self.m.answering.collect_evidence_sentences(query, chunks)
        candidates = self.m.answering.extract_candidates(query, evidence_sentences)
        option_evidence_chains = self.m.answering.build_option_evidence_chains(
            query,
            evidence_sentences,
            candidates,
        )
        self.m.answering.log_decision_snapshot(query, evidence_sentences, candidates)
        evidence_candidate = self.m.answering.extract_evidence_candidate(
            query, evidence_sentences, candidates
        )
        best_evidence = str(evidence_sentences[0].get("text", ""))[:160] if evidence_sentences else ""
        best_candidate = str(candidates[0].get("text", "")) if candidates else ""
        return (
            retrieved_context,
            topics,
            chunks,
            evidence_sentences,
            candidates,
            option_evidence_chains,
            evidence_candidate,
            best_evidence,
            best_candidate,
        )

    def try_fast_paths(
        self,
        *,
        input_text: str,
        query: str,
        chunks: List[Dict[str, object]],
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        option_evidence_chains: Optional[Dict[str, object]],
        evidence_candidate: Optional[Dict[str, str]],
    ) -> Optional[str]:
        decided = self.m.answering.decide_answer(
            query,
            evidence_sentences,
            candidates,
            reranked_chunks=chunks,
            option_evidence_chains=option_evidence_chains,
        )
        if decided is not None:
            final = str(decided.get("answer", "")).strip()
            reason = str(decided.get("reason", "deterministic"))
            if final:
                final = self.m.answering.postprocess_final_answer(
                    final, query, evidence_candidate=evidence_candidate
                )
                self.m._set_evidence_only_eval_chunks(evidence_sentences)
                logger.info(
                    "MemoryManager.chat: deterministic decision "
                    f"(reason={reason}, answer='{final}')."
                )
                self.m._record_turn(input_text, final)
                return final

        short_answer = self.m.answering.maybe_short_circuit(candidates, evidence_sentences)
        if short_answer is not None:
            self.m._set_evidence_only_eval_chunks(evidence_sentences)
            top_score = f"{float(candidates[0]['score']):.4f}" if candidates else "n/a"
            logger.info(
                "MemoryManager.chat: short-circuit answer from extracted candidates "
                f"(candidate='{short_answer}', score={top_score})."
            )
            self.m._record_turn(input_text, short_answer)
            return short_answer
        return None

    def build_generation_prompt(
        self,
        *,
        input_text: str,
        retrieved_context: str,
        chunks: List[Dict[str, object]],
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        option_evidence_chains: Optional[Dict[str, object]],
    ) -> str:
        recent_context = self.m._build_recent_context()
        graph_context = self.build_graph_context(query=input_text, chunks=chunks)
        refined_context = self.build_refined_context(query=input_text, chunks=chunks)
        fallback_context = refined_context if refined_context else retrieved_context
        if graph_context:
            fallback_context = (
                f"{graph_context}\n\n{fallback_context}" if fallback_context else graph_context
            )
        generation_context = build_generation_context(
            reranked_chunks=chunks,
            evidence_sentences=evidence_sentences,
            fallback_context=fallback_context,
            limits=PromptContextLimits(
                max_chunks=self.m.prompt_context_max_chunks,
                max_chars_per_chunk=self.m.prompt_context_max_chars_per_chunk,
                max_total_chars=self.m.prompt_context_max_total_chars,
            ),
        )
        self.m._set_prompt_eval_chunks(generation_context, evidence_sentences)
        return self.m.answering.build_answer_prompt(
            input_text=input_text,
            retrieved_context=generation_context,
            recent_context=recent_context,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            option_evidence_chains=option_evidence_chains,
        )

    def build_graph_context(self, query: str, chunks: List[Dict[str, object]]) -> str:
        if not self.m.graph_refiner_enabled:
            return ""
        if self.m.graph_context_from_store_enabled:
            snippets = self.m.long_memory.build_context_snippets(query)
            if not snippets:
                return ""
            return "[Long Memory Graph]\n" + "\n".join(f"- {line}" for line in snippets)
        snippets = self.m.long_memory.retrieve_from_chunks(
            query=query,
            chunks=[dict(x) for x in chunks],
            top_chunks=self.m.graph_refiner_top_chunks,
            max_chars_per_chunk=self.m.graph_refiner_chunk_max_chars,
            top_events=self.m.graph_refiner_top_events,
        )
        if not snippets:
            return ""
        return "[Long Memory Graph]\n" + "\n".join(f"- {line}" for line in snippets)

    def build_refined_context(self, query: str, chunks: List[Dict[str, object]]) -> str:
        if (not self.m.refiner_enabled) or (self.m.refiner_llm is None):
            return ""
        if not chunks:
            return ""
        selected = chunks[: max(1, self.m.refiner_top_chunks)]
        compact: List[str] = []
        for i, item in enumerate(selected, start=1):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            compact.append(f"{i}. {text[: self.m.refiner_max_chars_per_chunk]}")
        if not compact:
            return ""
        prompt = (
            "You are an evidence refiner.\n"
            "Given a question and retrieved text snippets, extract only the most relevant factual evidence.\n"
            "Return strict JSON only with format:\n"
            '{"evidence":["...", "..."]}\n'
            f"Maximum evidence items: {self.m.refiner_output_items}\n"
            "No explanation.\n\n"
            f"Question: {query}\n"
            "Snippets:\n"
            + "\n".join(compact)
        )
        try:
            raw = self.m.refiner_llm.chat([{"role": "user", "content": prompt}])
            payload = json.loads(str(raw).strip())
            rows = payload.get("evidence", [])
            if not isinstance(rows, list):
                return ""
            picked = [str(x).strip() for x in rows if str(x).strip()]
            if not picked:
                return ""
            return "[Refined Retrieved Evidence]\n" + "\n".join(
                f"- {line}" for line in picked[: self.m.refiner_output_items]
            )
        except (RuntimeError, ValueError, TypeError, json.JSONDecodeError):
            return ""

    def generate_with_fallback(
        self,
        *,
        input_text: str,
        query: str,
        prompt_text: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        option_evidence_chains: Optional[Dict[str, object]],
        evidence_candidate: Optional[Dict[str, str]],
    ) -> Tuple[str, str, str]:
        prompt_messages: List[Dict[str, str]] = [{"role": "user", "content": prompt_text}]
        model_name = str(getattr(self.m.llm, "model_name", "unknown"))
        logger.info(
            "MemoryManager.chat: invoking main LLM "
            f"(model={model_name}, prompt_chars={len(prompt_text)})."
        )
        response = self.m.llm.chat(prompt_messages)
        fallback_result = self.m.answering.evaluate_response_fallback(
            response=response,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            evidence_candidate=evidence_candidate,
        )
        ai_response = str(fallback_result.get("response", "")).strip()
        fallback_path = str(fallback_result.get("fallback_path", "none"))
        not_found_reason = str(fallback_result.get("not_found_reason", ""))

        normalized_ai_response = ai_response.strip().lower()
        should_retry_second_pass = (
            fallback_path.startswith("retry_due_to_")
            or (
                self.m.answering.second_pass_llm_enabled
                and evidence_sentences
                and normalized_ai_response == "not found in retrieved context."
                and fallback_path in {"fallback_to_not_found", "llm_not_found_accepted"}
            )
        )

        if should_retry_second_pass:
            model_name = str(getattr(self.m.llm, "model_name", "unknown"))
            logger.info(
                "MemoryManager.chat: invoking second-pass LLM "
                f"(model={model_name})."
            )
            second_prompt = self.m.answering.build_second_pass_prompt(
                input_text=input_text,
                evidence_sentences=evidence_sentences,
                evidence_candidate=evidence_candidate,
                option_evidence_chains=option_evidence_chains,
            )
            second_response = self.m.llm.chat([{"role": "user", "content": second_prompt}])
            second_result = self.m.answering.evaluate_response_fallback(
                response=second_response,
                evidence_sentences=evidence_sentences,
                candidates=candidates,
                evidence_candidate=evidence_candidate,
            )
            second_path = str(second_result.get("fallback_path", "none"))
            ai_response = str(second_result.get("response", "")).strip()
            if second_path.startswith("retry_due_to_"):
                ai_response = "Not found in retrieved context."
                second_path = "fallback_to_not_found"
            fallback_path = "second_pass:" + second_path
            not_found_reason = str(second_result.get("not_found_reason", not_found_reason))

        ai_response = self.m.answering.postprocess_final_answer(
            ai_response, query, evidence_candidate=evidence_candidate
        )
        return ai_response, fallback_path, not_found_reason

