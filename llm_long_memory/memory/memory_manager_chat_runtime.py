"""Chat orchestration runtime extracted from MemoryManager."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

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
        str,
        Optional[Dict[str, str]],
        str,
        str,
    ]:
        if precomputed_context is not None:
            context_text, topics, chunks = precomputed_context
        else:
            context_text, topics, chunks = self.m.retrieve_context(query)
        retrieved_ids = [str(topic["topic_id"]) for topic in topics]
        logger.info(f"MemoryManager.chat: retrieved topics={retrieved_ids}")

        if self.m.retrieval_execution_mode in {"model_only", "naive_rag"}:
            evidence_sentences: List[Dict[str, object]] = []
            candidates: List[Dict[str, object]] = []
            fallback_answer = ""
            evidence_candidate = None
            best_evidence = ""
            best_candidate = ""
            graph_tool_answer = ""
        else:
            evidence_sentences = self.m.answering.collect_evidence_sentences(query, chunks)
            candidates = self.m.answering.extract_candidates(query, evidence_sentences)
            self.m.answering.log_decision_snapshot(query, evidence_sentences, candidates)
            fallback_answer = self.m.answering.resolve_fallback_answer(
                query,
                evidence_sentences,
                candidates,
                chunks,
            )
            evidence_candidate = self.m.answering.extract_evidence_candidate(
                query, evidence_sentences, candidates
            )
            best_evidence = (
                str(evidence_sentences[0].get("text", ""))[:160] if evidence_sentences else ""
            )
            best_candidate = str(candidates[0].get("text", "")) if candidates else ""
            graph_tool_answer = self.m.graph_toolkit.build_tool_answer(
                query=query,
                graph_context=self.build_graph_context(query=query, chunks=chunks),
                evidence_sentences=evidence_sentences,
                candidates=candidates,
                chunks=chunks,
            )
            if graph_tool_answer.strip():
                fallback_answer = graph_tool_answer.strip()
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

        fallback_text = str(fallback_answer).strip()
        if not fallback_text and evidence_candidate is not None:
            fallback_text = str(evidence_candidate.get("answer", "")).strip()
        if not fallback_text and candidates:
            fallback_text = str(candidates[0].get("text", "")).strip()
        if not fallback_text and best_evidence.strip():
            fallback_text = best_evidence.strip()
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
        if not self.m.graph_refiner_enabled:
            return ""
        if not bool(getattr(self.m, "long_memory_enabled", False)):
            return ""
        if not bool(getattr(self.m, "offline_graph_build_enabled", False)):
            return ""
        if not self.m.graph_context_from_store_enabled:
            return ""
        toolkit = getattr(self.m, "graph_toolkit", None)
        if toolkit is None:
            return ""
        return str(
            toolkit.build_tool_hints(
                query=query,
                graph_context=graph_context,
                evidence_sentences=evidence_sentences,
                candidates=candidates,
                chunks=chunks,
            )
        ).strip()

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
        snippets = self.m.long_memory.build_context_snippets(query)
        if not snippets:
            return ""
        query_tokens = {
            tok
            for tok in re.findall(r"[a-z0-9]+", str(query).lower())
            if tok and tok not in {"the", "a", "an", "to", "of", "and", "or", "in", "on", "my"}
        }
        if query_tokens:
            filtered: List[str] = []
            for snippet in snippets:
                snippet_tokens = set(re.findall(r"[a-z0-9]+", str(snippet).lower()))
                if not snippet_tokens:
                    continue
                shared = len(query_tokens.intersection(snippet_tokens))
                overlap_ratio = float(shared) / float(max(1, len(query_tokens)))
                if shared >= 2 or overlap_ratio >= 0.20:
                    filtered.append(snippet)
            max_items = min(3, max(1, int(getattr(self.m.long_memory, "context_max_items", 4))))
            snippets = (filtered[:max_items] if filtered else snippets[:max_items])
        return "[Long Memory Graph]\n" + "\n".join(f"- {line}" for line in snippets)

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
