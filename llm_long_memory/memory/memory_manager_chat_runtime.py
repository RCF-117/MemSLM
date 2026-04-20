"""Chat orchestration runtime extracted from MemoryManager."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from llm_long_memory.utils.logger import logger


class MemoryManagerChatRuntime:
    """Keep MemoryManager.chat-related orchestration out of the main class body."""

    def __init__(self, manager: Any) -> None:
        self.m = manager
        self._last_specialist_payload: Dict[str, object] = {}

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
        str,
        Optional[Dict[str, str]],
        str,
        str,
    ]:
        if precomputed_context is not None:
            context_text, topics, chunks = precomputed_context
        else:
            context_text, topics, chunks = self.m.retrieve_context(query)
        logger.info(f"MemoryManager.chat: retrieved chunks={len(chunks)}")

        if self.m.retrieval_execution_mode in {"model_only", "naive_rag"}:
            evidence_sentences: List[Dict[str, object]] = []
            candidates: List[Dict[str, object]] = []
            fallback_answer = ""
            evidence_candidate = None
            best_evidence = ""
            best_candidate = ""
            self._last_specialist_payload = {}
        else:
            evidence_sentences = self.m.answering.collect_evidence_sentences(query, chunks)
            candidates = self.m.answering.extract_candidates(query, evidence_sentences)
            self.m.answering.log_decision_snapshot(query, evidence_sentences, candidates)
            fallback_answer = ""
            evidence_candidate = self.m.answering.extract_evidence_candidate(
                query, evidence_sentences, candidates
            )
            if bool(getattr(self.m.answering, "reasoning_fallback_enabled", True)):
                # Base fallback comes from candidate extractor path only.
                fallback_answer = (
                    str((evidence_candidate or {}).get("answer", "")).strip()
                    if evidence_candidate is not None
                    else ""
                )
            best_evidence = (
                str(evidence_sentences[0].get("text", ""))[:160] if evidence_sentences else ""
            )
            best_candidate = str(candidates[0].get("text", "")) if candidates else ""
            graph_context = self.build_graph_context(query=query, chunks=chunks)
            self._last_specialist_payload = self.m.specialist_layer.run(
                query=query,
                graph_context=graph_context,
                evidence_sentences=evidence_sentences,
                candidates=candidates,
                chunks=chunks,
            )
            specialist_fallback = str(
                self._last_specialist_payload.get("fallback_answer", "")
            ).strip()
            if specialist_fallback:
                fallback_answer = specialist_fallback
        return (
            context_text,
            topics,
            chunks,
            evidence_sentences,
            candidates,
            fallback_answer,
            evidence_candidate,
            best_evidence,
            best_candidate,
        )

    def resolve_prompt_fallback(
        self,
        fallback_answer: str,
        evidence_candidate: Optional[Dict[str, str]],
        candidates: List[Dict[str, object]],
        best_evidence: str,
    ) -> str:
        fallback_text = str(fallback_answer or "").strip()
        if not fallback_text and evidence_candidate is not None:
            fallback_text = str(evidence_candidate.get("answer", "")).strip()
        if not fallback_text and candidates:
            fallback_text = str(candidates[0].get("text", "")).strip()
        if not fallback_text and best_evidence.strip():
            fallback_text = best_evidence.strip()
        return fallback_text

    def build_generation_prompt(
        self,
        *,
        input_text: str,
        retrieved_context_text: str,
        evidence_sentences: List[Dict[str, object]],
        chunks: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        best_evidence: str,
        fallback_answer: str,
        evidence_candidate: Optional[Dict[str, str]],
    ) -> str:
        execution_mode = str(getattr(self.m, "retrieval_execution_mode", "memslm")).strip().lower()
        graph_context = self.build_graph_context(query=input_text, chunks=chunks)
        retrieved_context_text = str(retrieved_context_text or "").strip()

        if execution_mode == "model_only":
            prompt_sections: List[Dict[str, str]] = [
                {
                    "section": "answer_rules",
                    "text": "Return only the final answer.",
                }
            ]
            compact_prompt = self.m.answering.build_answer_prompt(
                input_text=input_text,
                graph_context="",
                graph_tool_hints="",
                rag_evidence="",
                fallback_answer="",
            )
            self.m._set_prompt_eval_chunks(prompt_sections)
            return compact_prompt

        if execution_mode == "naive_rag":
            retrieved_text = retrieved_context_text or "None"
            prompt_sections = [
                {"section": "retrieved_context", "text": retrieved_text},
                {
                    "section": "answer_rules",
                    "text": (
                        "Use only the retrieved context.\n"
                        "Do not add graph reasoning or fallback heuristics.\n"
                        "Return only the final answer."
                    ),
                },
            ]
            compact_prompt = (
                "[Retrieved Context]\n"
                f"{retrieved_text}\n\n"
                "[Answer Rules]\n"
                "Use only the retrieved context.\n"
                "Return only the final answer.\n\n"
                f"User: {input_text}"
            )
            self.m._set_prompt_eval_chunks(prompt_sections)
            return compact_prompt

        graph_tool_hints = self.build_graph_tool_hints(
            query=input_text,
            graph_context=graph_context,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            chunks=chunks,
        )

        fallback_text = self.resolve_prompt_fallback(
            fallback_answer=fallback_answer,
            evidence_candidate=evidence_candidate,
            candidates=candidates,
            best_evidence=best_evidence,
        )
        prompt_sections: List[Dict[str, str]] = []
        if graph_context.strip():
            prompt_sections.append({"section": "graph_evidence", "text": graph_context.strip()})
        if graph_tool_hints.strip():
            prompt_sections.append({"section": "graph_tool_hints", "text": graph_tool_hints.strip()})
        if best_evidence.strip():
            prompt_sections.append({"section": "rag_evidence", "text": best_evidence.strip()})
        if fallback_text:
            prompt_sections.append({"section": "fallback_answer", "text": fallback_text})
        prompt_sections.append(
            {
                "section": "answer_rules",
                "text": (
                    "Use Graph Evidence first.\n"
                    "If Graph Evidence is weak or empty, use the Fallback Answer as the compact backup clue.\n"
                    "Do not repeat long evidence blocks.\n"
                    "Do not say Not found unless both Graph Evidence and the fallback cues are insufficient.\n"
                    "Keep key qualifiers (for example: each way, round trip, per day).\n"
                    "Return only the final answer."
                    if self.m.answering.answer_context_only
                    else "Return only the final answer."
                ),
            }
        )
        compact_prompt = self.m.answering.build_answer_prompt(
            input_text=input_text,
            graph_context=graph_context,
            graph_tool_hints=graph_tool_hints,
            rag_evidence=best_evidence,
            fallback_answer=fallback_text,
        )
        self.m._set_prompt_eval_chunks(prompt_sections)
        return compact_prompt

    def build_graph_tool_hints(
        self,
        *,
        query: str,
        graph_context: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        chunks: List[Dict[str, object]],
    ) -> str:
        # Kept method name for backward compatibility with prompt-builder call sites.
        # In the current architecture, specialist hints come from the unified specialist layer.
        hints = str(self._last_specialist_payload.get("hints", "")).strip()
        if hints:
            return hints
        payload = self.m.specialist_layer.run(
            query=query,
            graph_context=graph_context,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            chunks=chunks,
        )
        self._last_specialist_payload = payload
        return str(payload.get("hints", "")).strip()

    def build_graph_context(self, query: str, chunks: List[Dict[str, object]]) -> str:
        if not self.m.graph_refiner_enabled:
            return ""
        if not bool(getattr(self.m, "long_memory_enabled", False)):
            return ""
        if not bool(getattr(self.m, "offline_graph_build_enabled", False)):
            return ""
        # Enforce offline-first long-memory usage:
        # graph extraction is completed before answering; chat only reads stored graph evidence.
        if not self.m.graph_context_from_store_enabled:
            return ""
        snippets: List[str] = []
        engine = getattr(self.m, "graph_query_engine", None)
        if engine is not None:
            try:
                pack = engine.query(
                    query=query,
                    max_items=min(4, max(1, int(getattr(self.m.long_memory, "context_max_items", 4)))),
                )
                raw_snippets = list(pack.get("snippets", [])) if isinstance(pack, dict) else []
                snippets = [str(x).strip() for x in raw_snippets if str(x).strip()]
            except (RuntimeError, ValueError, TypeError):
                snippets = []
        # Safe fallback to legacy snippet generation if graph query yielded nothing.
        if not snippets:
            snippets = self.m.long_memory.build_context_snippets(query)
        if not snippets:
            return ""
        return "[Long Memory Graph]\n" + "\n".join(f"- {line}" for line in snippets[:4])

    def generate_with_fallback(
        self,
        *,
        input_text: str,
        query: str,
        prompt_text: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        fallback_answer: str,
        evidence_candidate: Optional[Dict[str, str]],
    ) -> Tuple[str, str, str]:
        prompt_messages: List[Dict[str, str]] = [{"role": "user", "content": prompt_text}]
        model_name = str(getattr(self.m.llm, "model_name", "unknown"))
        logger.info(
            "MemoryManager.chat: invoking main LLM "
            f"(model={model_name}, prompt_chars={len(prompt_text)})."
        )
        response = self.m.llm.chat(prompt_messages)
        execution_mode = str(getattr(self.m, "retrieval_execution_mode", "memslm")).strip().lower()
        if execution_mode in {"model_only", "naive_rag"}:
            ai_response = self.m.answering.postprocess_final_answer(
                response, query, evidence_candidate=None
            )
            return ai_response, f"{execution_mode}_direct", ""
        fallback_result = self.m.answering.evaluate_response_fallback(
            response=response,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            evidence_candidate=evidence_candidate,
            fallback_answer=fallback_answer,
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
                prompt_text=prompt_text,
                evidence_candidate=evidence_candidate,
            )
            second_response = self.m.llm.chat([{"role": "user", "content": second_prompt}])
            second_result = self.m.answering.evaluate_response_fallback(
                response=second_response,
                evidence_sentences=evidence_sentences,
                candidates=candidates,
                evidence_candidate=evidence_candidate,
                fallback_answer=fallback_answer,
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
