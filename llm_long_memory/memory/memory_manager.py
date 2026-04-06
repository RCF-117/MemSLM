"""Memory manager that orchestrates short memory, mid memory, and LLM calls."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json

from llm_long_memory.llm.ollama_client import LLM
from llm_long_memory.memory.answering_pipeline import AnsweringPipeline
from llm_long_memory.memory.long_memory import LongMemory
from llm_long_memory.memory.mid_memory import MidMemory
from llm_long_memory.memory.prompt_context import (
    PromptContextLimits,
    build_generation_context,
)
from llm_long_memory.memory.short_memory import ShortMemory
from llm_long_memory.utils.helpers import load_config
from llm_long_memory.utils.logger import logger


Message = Dict[str, Any]


class MemoryManager:
    """Central controller for retrieval-augmented chat with persistent mid memory."""

    def __init__(self, llm: LLM, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize manager with config-driven memory sizes and modules."""
        self.config = config or load_config()
        self.llm = llm
        short_size = int(self.config["memory"]["short_memory_size"])
        self.short_memory = ShortMemory(max_turns=short_size, config=self.config)
        self.mid_memory = MidMemory(config=self.config)
        self.long_memory = LongMemory(config=self.config)
        answering_cfg = dict(self.config["retrieval"]["answering"])
        self.answering = AnsweringPipeline(answering_cfg)
        self.prompt_context_max_chunks = int(answering_cfg["prompt_context_max_chunks"])
        self.prompt_context_max_chars_per_chunk = int(answering_cfg["prompt_context_max_chars_per_chunk"])
        self.prompt_context_max_total_chars = int(answering_cfg["prompt_context_max_total_chars"])
        self.prompt_recent_max_messages = int(answering_cfg["prompt_recent_max_messages"])
        self.last_prompt_eval_chunks: List[Dict[str, str]] = []
        logger.info("MemoryManager initialized.")

    def _rewrite_query_for_long_memory(self, query: str) -> List[str]:
        """Use main LLM once to create a few retrieval-friendly rewrites."""
        if not bool(getattr(self.long_memory, "query_rewrite_enabled", False)):
            return [query]
        max_rewrites = max(0, int(getattr(self.long_memory, "query_rewrite_max", 2)))
        if max_rewrites <= 0:
            return [query]
        prompt = (
            "Rewrite the query into short retrieval-oriented paraphrases.\n"
            "Return strict JSON only with format: "
            '{"rewrites":["...","..."]}\n'
            f"max_rewrites={max_rewrites}\n"
            f"query={query}"
        )
        try:
            raw = self.llm.chat([{"role": "user", "content": prompt}])
            data = json.loads(str(raw).strip())
            rewrites = [
                str(x).strip()
                for x in list(data.get("rewrites", []))
                if str(x).strip()
            ]
            uniq: List[str] = [query]
            for rw in rewrites:
                if rw not in uniq:
                    uniq.append(rw)
                if len(uniq) >= (max_rewrites + 1):
                    break
            return uniq
        except (RuntimeError, ValueError, TypeError, json.JSONDecodeError):
            return [query]

    def retrieve_context(
        self, query: str
    ) -> Tuple[str, List[Dict[str, object]], List[Dict[str, object]]]:
        """Retrieve top topics and reranked chunks for final prompt context."""
        topics = self.mid_memory.search(query)
        topic_score_map = {
            str(topic.get("topic_id", "")): float(topic.get("score", 0.0)) for topic in topics
        }
        global_chunk_enabled = bool(
            getattr(self.mid_memory, "global_chunk_retrieval_enabled", False)
        )
        if global_chunk_enabled and hasattr(self.mid_memory, "search_chunks_global"):
            reranked_chunks = self.mid_memory.search_chunks_global(query, topic_score_map)
            topic_expansion_enabled = bool(
                getattr(self.mid_memory, "topic_expansion_enabled", False)
            )
            if topic_expansion_enabled and topics:
                primary_topic_ids = {str(item.get("topic_id", "")) for item in reranked_chunks}
                missing_topics = [
                    topic for topic in topics if str(topic.get("topic_id", "")) not in primary_topic_ids
                ]
                if missing_topics:
                    expanded = self.mid_memory.rerank_chunks(query, missing_topics)
                    per_topic_limit = max(
                        0, int(getattr(self.mid_memory, "topic_expansion_per_topic_limit", 0))
                    )
                    if per_topic_limit > 0:
                        existing_chunk_ids = {
                            int(c.get("chunk_id"))
                            for c in reranked_chunks
                            if c.get("chunk_id") is not None
                        }
                        topic_counts: Dict[str, int] = {}
                        for chunk in expanded:
                            tid = str(chunk.get("topic_id", ""))
                            count = topic_counts.get(tid, 0)
                            if count >= per_topic_limit:
                                continue
                            cid_raw = chunk.get("chunk_id")
                            if cid_raw is not None and int(cid_raw) in existing_chunk_ids:
                                continue
                            reranked_chunks.append(chunk)
                            if cid_raw is not None:
                                existing_chunk_ids.add(int(cid_raw))
                            topic_counts[tid] = count + 1
        else:
            reranked_chunks = self.mid_memory.rerank_chunks(query, topics)
        grouped: Dict[str, List[str]] = {}
        for item in reranked_chunks:
            topic_id = str(item["topic_id"])
            grouped.setdefault(topic_id, []).append(str(item["text"]))

        context_parts: List[str] = []
        ordered_topic_ids: List[str] = []
        for topic in topics:
            topic_id = str(topic["topic_id"])
            ordered_topic_ids.append(topic_id)
        for item in reranked_chunks:
            topic_id = str(item.get("topic_id", ""))
            if topic_id and topic_id not in ordered_topic_ids:
                ordered_topic_ids.append(topic_id)

        for topic_id in ordered_topic_ids:
            chunks = grouped.get(topic_id, [])
            if chunks:
                context_parts.append(f"[Topic: {topic_id}]\n" + "\n".join(chunks))
        context_text = "\n\n".join(context_parts)

        lm_ctx_cfg = dict(self.config["retrieval"].get("long_memory_context", {}))
        if bool(lm_ctx_cfg.get("enabled", False)):
            rewritten_queries = self._rewrite_query_for_long_memory(query)
            if len(rewritten_queries) > 1:
                long_snippets = self.long_memory.build_context_snippets_multi(rewritten_queries)
            else:
                long_snippets = self.long_memory.build_context_snippets(query)
            if long_snippets:
                long_block = "[Long Memory]\n" + "\n".join(long_snippets)
                prepend_long = bool(lm_ctx_cfg.get("prepend_before_mid_context", True))
                if prepend_long and context_text:
                    context_text = f"{long_block}\n\n{context_text}"
                elif prepend_long:
                    context_text = long_block
                elif context_text:
                    context_text = f"{context_text}\n\n{long_block}"
                else:
                    context_text = long_block
                logger.info(
                    "MemoryManager.retrieve_context: "
                    f"long_memory_hits={len(long_snippets)}."
                )
        return context_text, topics, reranked_chunks

    def ingest_message(self, message: Message) -> None:
        """Ingest one dataset message into memory without calling the LLM."""
        role = str(message.get("role", "user")).strip().lower() or "user"
        content = str(message.get("content", "")).strip()
        if not content:
            return
        normalized: Message = {"role": role, "content": content}
        for key in ("session_id", "session_date", "turn_index"):
            if key in message:
                normalized[key] = message.get(key)
        if bool(message.get("has_answer", False)):
            normalized["has_answer"] = True
        logger.debug(f"MemoryManager.ingest_message: role={role}, content_len={len(content)}")
        self.short_memory.add(normalized)
        self.short_memory.flush_to_mid_memory(self.mid_memory)
        self.long_memory.enqueue_message(normalized)

    def finalize_ingest(self) -> None:
        """Flush pending dynamic chunk buffer after dataset ingestion."""
        self.mid_memory.flush_pending()
        logger.info("MemoryManager.finalize_ingest: flushed pending mid-memory buffer.")

    def archive_short_to_mid(self, clear_short: bool = True) -> int:
        """Persist all current short-memory messages into mid memory."""
        pending = self.short_memory.get()
        moved = 0
        for message in pending:
            self.mid_memory.add(message)
            moved += 1
        self.mid_memory.flush_pending()
        if clear_short:
            self.short_memory.clear()
        logger.info(
            "MemoryManager.archive_short_to_mid: "
            f"moved={moved}, clear_short={clear_short}."
        )
        return moved

    def reset_for_new_instance(self) -> None:
        """Reset short and mid memory for isolated per-instance evaluation."""
        self.short_memory.clear()
        self.mid_memory.clear_all()
        self.long_memory.clear_all()
        logger.info("MemoryManager.reset_for_new_instance: memory reset completed.")

    def chat(
        self,
        input_text: str,
        retrieval_query: Optional[str] = None,
        precomputed_context: Optional[Tuple[str, List[Dict[str, object]], List[Dict[str, object]]]] = None,
    ) -> str:
        """Handle one user message with retrieval, LLM call, and memory updates."""
        self.last_prompt_eval_chunks = []
        logger.info(f"MemoryManager.chat: user input='{input_text}'")
        query = retrieval_query if retrieval_query is not None else input_text
        (
            retrieved_context,
            topics,
            chunks,
            evidence_sentences,
            candidates,
            option_evidence_chains,
            evidence_candidate,
            best_evidence,
            best_candidate,
        ) = self._prepare_answer_inputs(query, precomputed_context)
        fast_answer = self._try_fast_paths(
            input_text=input_text,
            query=query,
            chunks=chunks,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            option_evidence_chains=option_evidence_chains,
            evidence_candidate=evidence_candidate,
        )
        if fast_answer is not None:
            return fast_answer

        prompt_text = self._build_generation_prompt(
            input_text=input_text,
            retrieved_context=retrieved_context,
            chunks=chunks,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            option_evidence_chains=option_evidence_chains,
        )
        ai_response, fallback_path, not_found_reason = self._generate_with_fallback(
            input_text=input_text,
            query=query,
            prompt_text=prompt_text,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            option_evidence_chains=option_evidence_chains,
            evidence_candidate=evidence_candidate,
        )
        logger.info(
            "MemoryManager.answer_debug: "
            f"fallback_path={fallback_path}, "
            f"not_found_reason={not_found_reason or 'none'}, "
            f"best_evidence='{best_evidence}', "
            f"best_candidate='{best_candidate}', "
            f"evidence_candidate='{(evidence_candidate or {}).get('answer', '')}'."
        )
        logger.info(f"MemoryManager.chat: LLM response='{ai_response}'")
        self._record_turn(input_text, ai_response)
        return ai_response

    def _prepare_answer_inputs(
        self,
        query: str,
        precomputed_context: Optional[Tuple[str, List[Dict[str, object]], List[Dict[str, object]]]],
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
            retrieved_context, topics, chunks = self.retrieve_context(query)
        retrieved_ids = [str(topic["topic_id"]) for topic in topics]
        logger.info(f"MemoryManager.chat: retrieved topics={retrieved_ids}")

        evidence_sentences = self.answering.collect_evidence_sentences(query, chunks)
        candidates = self.answering.extract_candidates(query, evidence_sentences)
        option_evidence_chains = self.answering.build_option_evidence_chains(
            query,
            evidence_sentences,
            candidates,
        )
        self.answering.log_decision_snapshot(query, evidence_sentences, candidates)
        evidence_candidate = self.answering.extract_evidence_candidate(
            query, evidence_sentences, candidates
        )
        best_evidence = (
            str(evidence_sentences[0].get("text", ""))[:160] if evidence_sentences else ""
        )
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

    def _set_prompt_eval_chunks(self, generation_context: str, evidence_sentences: List[Dict[str, object]]) -> None:
        self.last_prompt_eval_chunks = []
        if generation_context.strip():
            self.last_prompt_eval_chunks.append({"text": generation_context})
        self.last_prompt_eval_chunks.extend(
            {"text": str(item.get("text", ""))}
            for item in evidence_sentences
            if str(item.get("text", "")).strip()
        )

    def _set_evidence_only_eval_chunks(self, evidence_sentences: List[Dict[str, object]]) -> None:
        self.last_prompt_eval_chunks = [
            {"text": str(item.get("text", ""))}
            for item in evidence_sentences
            if str(item.get("text", "")).strip()
        ]

    def _try_fast_paths(
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
        decided = self.answering.decide_answer(
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
                final = self.answering.postprocess_final_answer(
                    final, query, evidence_candidate=evidence_candidate
                )
                self._set_evidence_only_eval_chunks(evidence_sentences)
                logger.info(
                    "MemoryManager.chat: deterministic decision "
                    f"(reason={reason}, answer='{final}')."
                )
                self._record_turn(input_text, final)
                return final

        short_answer = self.answering.maybe_short_circuit(candidates, evidence_sentences)
        if short_answer is not None:
            self._set_evidence_only_eval_chunks(evidence_sentences)
            top_score = (
                f"{float(candidates[0]['score']):.4f}" if candidates else "n/a"
            )
            logger.info(
                "MemoryManager.chat: short-circuit answer from extracted candidates "
                f"(candidate='{short_answer}', score={top_score})."
            )
            self._record_turn(input_text, short_answer)
            return short_answer
        return None

    def _build_recent_context(self) -> str:
        recent_messages = self.short_memory.get()
        if self.prompt_recent_max_messages > 0:
            recent_messages = recent_messages[-self.prompt_recent_max_messages :]
        return "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in recent_messages
        )

    def _build_generation_prompt(
        self,
        *,
        input_text: str,
        retrieved_context: str,
        chunks: List[Dict[str, object]],
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        option_evidence_chains: Optional[Dict[str, object]],
    ) -> str:
        recent_context = self._build_recent_context()
        generation_context = build_generation_context(
            reranked_chunks=chunks,
            evidence_sentences=evidence_sentences,
            fallback_context=retrieved_context,
            limits=PromptContextLimits(
                max_chunks=self.prompt_context_max_chunks,
                max_chars_per_chunk=self.prompt_context_max_chars_per_chunk,
                max_total_chars=self.prompt_context_max_total_chars,
            ),
        )
        self._set_prompt_eval_chunks(generation_context, evidence_sentences)
        return self.answering.build_answer_prompt(
            input_text=input_text,
            retrieved_context=generation_context,
            recent_context=recent_context,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            option_evidence_chains=option_evidence_chains,
        )

    def _generate_with_fallback(
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
        prompt_messages: List[Message] = [{"role": "user", "content": prompt_text}]
        model_name = str(getattr(self.llm, "model_name", "unknown"))
        logger.info(
            "MemoryManager.chat: invoking main LLM "
            f"(model={model_name}, prompt_chars={len(prompt_text)})."
        )
        response = self.llm.chat(prompt_messages)
        fallback_result = self.answering.evaluate_response_fallback(
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
                self.answering.second_pass_llm_enabled
                and evidence_sentences
                and normalized_ai_response == "not found in retrieved context."
                and fallback_path in {"fallback_to_not_found", "llm_not_found_accepted"}
            )
        )

        if should_retry_second_pass:
            model_name = str(getattr(self.llm, "model_name", "unknown"))
            logger.info(
                "MemoryManager.chat: invoking second-pass LLM "
                f"(model={model_name})."
            )
            second_prompt = self.answering.build_second_pass_prompt(
                input_text=input_text,
                evidence_sentences=evidence_sentences,
                evidence_candidate=evidence_candidate,
                option_evidence_chains=option_evidence_chains,
            )
            second_response = self.llm.chat([{"role": "user", "content": second_prompt}])
            second_result = self.answering.evaluate_response_fallback(
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

        ai_response = self.answering.postprocess_final_answer(
            ai_response, query, evidence_candidate=evidence_candidate
        )
        return ai_response, fallback_path, not_found_reason

    def get_last_prompt_eval_chunks(self) -> List[Dict[str, str]]:
        """Return prompt-grounded chunks used by the most recent chat call."""
        return [{"text": str(item.get("text", ""))} for item in self.last_prompt_eval_chunks]

    def close(self) -> None:
        """Close owned resources."""
        self.long_memory.close()
        self.mid_memory.close()

    def _record_turn(self, user_input: str, assistant_output: str) -> None:
        """Persist one user-assistant turn into short memory and flush overflow."""
        user_message: Message = {"role": "user", "content": user_input}
        assistant_message: Message = {"role": "assistant", "content": assistant_output}
        self.short_memory.add(user_message)
        self.short_memory.add(assistant_message)
        self.short_memory.flush_to_mid_memory(self.mid_memory)
        self.long_memory.enqueue_message(user_message)
        self.long_memory.enqueue_message(assistant_message)
