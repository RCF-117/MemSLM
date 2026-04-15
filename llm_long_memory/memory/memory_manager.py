"""Memory manager that orchestrates short memory, mid memory, and LLM calls."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from llm_long_memory.llm.ollama_client import LLM
from llm_long_memory.memory.answering_pipeline import AnsweringPipeline
from llm_long_memory.memory.long_memory import LongMemory
from llm_long_memory.memory.memory_manager_chat_runtime import MemoryManagerChatRuntime
from llm_long_memory.memory.memory_manager_utils import (
    build_temporal_anchor_queries,
    dedup_chunks_keep_best,
    is_temporal_query,
    merge_anchor_chunks,
)
from llm_long_memory.memory.mid_memory import MidMemory
from llm_long_memory.memory.short_memory import ShortMemory
from llm_long_memory.utils.helpers import load_config
from llm_long_memory.utils.logger import logger


Message = Dict[str, Any]


class _NoOpLongMemory:
    """Disabled long-memory stub to avoid runtime overhead and code-path noise."""

    query_rewrite_enabled = False
    query_rewrite_max = 0

    def query(self, query_text: str) -> List[Dict[str, Any]]:
        return []

    def query_multi(self, query_texts: List[str]) -> List[Dict[str, Any]]:
        return []

    def build_context_snippets(self, query_text: str) -> List[str]:
        return []

    def build_context_snippets_multi(self, query_texts: List[str]) -> List[str]:
        return []

    def retrieve_from_chunks(
        self,
        *,
        query: str,
        chunks: List[Dict[str, Any]],
        top_chunks: int,
        max_chars_per_chunk: int,
        top_events: int,
    ) -> List[str]:
        return []

    def ingest_from_chunks(
        self,
        *,
        chunks: List[Dict[str, Any]],
        top_chunks: int,
        max_chars_per_chunk: int,
    ) -> int:
        return 0

    def debug_stats(self) -> Dict[str, int]:
        return {
            "nodes": 0,
            "edges": 0,
            "events": 0,
            "details": 0,
            "active_events": 0,
            "superseded_events": 0,
            "queued_updates": 0,
            "applied_updates": 0,
            "ingest_event_total": 0,
            "ingest_event_accepted": 0,
            "ingest_event_rejected": 0,
            "extractor_calls": 0,
            "extractor_success": 0,
            "extractor_failures": 0,
            "extractor_seen_messages": 0,
            "candidate_events": 0,
            "reject_reason_low_confidence": 0,
            "reject_reason_few_keywords": 0,
            "reject_reason_short_object": 0,
            "reject_reason_few_entities": 0,
            "reject_reason_short_sentence": 0,
            "reject_reason_long_sentence": 0,
            "reject_reason_missing_time_or_location": 0,
            "reject_reason_rejected_phrase": 0,
            "reject_reason_generic_subject_action": 0,
            "reject_reason_generic_action_disabled": 0,
            "reject_reason_empty_key_component": 0,
        }

    def clear_all(self) -> None:
        return None

    def close(self) -> None:
        return None


class MemoryManager:
    """Central controller for retrieval-augmented chat with persistent mid memory."""

    def __init__(self, llm: LLM, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize manager with config-driven memory sizes and modules."""
        self.config = config or load_config()
        self.llm = llm
        short_size = int(self.config["memory"]["short_memory_size"])
        self.short_memory = ShortMemory(max_turns=short_size, config=self.config)
        self.mid_memory = MidMemory(config=self.config)
        lm_ctx_cfg = dict(self.config["retrieval"].get("long_memory_context", {}))
        long_mem_cfg = dict(self.config["memory"]["long_memory"])
        self.long_memory_enabled = bool(long_mem_cfg.get("enabled", False))
        self.long_memory_query_graph_enabled = bool(
            lm_ctx_cfg.get("query_graph_enabled", False)
        )
        self.long_memory = (
            LongMemory(config=self.config)
            if (self.long_memory_enabled or self.long_memory_query_graph_enabled)
            else _NoOpLongMemory()
        )
        answering_cfg = dict(self.config["retrieval"]["answering"])
        self.answering = AnsweringPipeline(answering_cfg)
        self.graph_refiner_enabled = bool(answering_cfg.get("graph_refiner_enabled", False))
        self.graph_context_from_store_enabled = bool(
            answering_cfg.get("graph_context_from_store_enabled", False)
        )
        offline_graph_cfg = dict(self.config["memory"]["long_memory"].get("offline_graph", {}))
        self.offline_graph_build_enabled = bool(offline_graph_cfg.get("enabled", False))
        self.offline_graph_build_top_chunks = int(offline_graph_cfg.get("build_top_chunks", 6))
        self.offline_graph_build_chunk_max_chars = int(
            offline_graph_cfg.get("build_chunk_max_chars", 260)
        )

        temporal_anchor_cfg = dict(self.config["retrieval"].get("temporal_anchor_retrieval", {}))
        self.temporal_anchor_enabled = bool(temporal_anchor_cfg.get("enabled", False))
        self.temporal_anchor_require_temporal_cue = bool(
            temporal_anchor_cfg.get("require_temporal_cue", True)
        )
        self.temporal_anchor_max_options = int(temporal_anchor_cfg.get("max_options", 3))
        self.temporal_anchor_extra_queries_per_option = int(
            temporal_anchor_cfg.get("extra_queries_per_option", 1)
        )
        self.temporal_anchor_top_n_per_query = int(
            temporal_anchor_cfg.get("top_n_per_query", 10)
        )
        self.temporal_anchor_additive_limit = int(
            temporal_anchor_cfg.get("additive_limit", 8)
        )
        self.temporal_anchor_cue_keywords = [
            str(x).strip().lower()
            for x in list(temporal_anchor_cfg.get("cue_keywords", []))
            if str(x).strip()
        ]

        graph_retry_cfg = dict(self.config["retrieval"].get("graph_build_retry", {}))
        self.graph_build_retry_enabled = bool(graph_retry_cfg.get("enabled", False))
        self.graph_build_retry_expanded_top_n = int(
            graph_retry_cfg.get("expanded_top_n", 48)
        )
        self.graph_build_retry_ingest_top_chunks = int(
            graph_retry_cfg.get("ingest_top_chunks", 10)
        )
        self.graph_build_retry_use_temporal_anchors = bool(
            graph_retry_cfg.get("use_temporal_anchors", True)
        )
        self.graph_build_retry_anchor_query_limit = int(
            graph_retry_cfg.get("anchor_query_limit", 4)
        )
        self.graph_build_retry_anchor_top_n_per_query = int(
            graph_retry_cfg.get("anchor_top_n_per_query", 8)
        )
        graph_build_role_cfg = dict(self.config["retrieval"].get("graph_build_role_weights", {}))
        self.graph_build_role_weights = {
            str(k).strip().lower(): float(v)
            for k, v in graph_build_role_cfg.items()
            if str(k).strip()
        }
        self.chat_runtime = MemoryManagerChatRuntime(self)
        self.last_prompt_eval_chunks: List[Dict[str, str]] = []
        logger.info("MemoryManager initialized.")

    @staticmethod
    def _dedup_chunks_keep_best(chunks: List[Dict[str, object]]) -> List[Dict[str, object]]:
        return dedup_chunks_keep_best(chunks)

    def _graph_build_role_weight(self, role: str) -> float:
        normalized = str(role).strip().lower()
        if normalized in self.graph_build_role_weights:
            return float(self.graph_build_role_weights[normalized])
        return 1.0

    def _rank_chunks_for_graph_build(
        self,
        chunks: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        ranked = [dict(x) for x in chunks]
        ranked.sort(
            key=lambda x: float(x.get("score", 0.0))
            * self._graph_build_role_weight(str(x.get("role", "user"))),
            reverse=True,
        )
        return ranked

    def _is_temporal_query(self, query: str) -> bool:
        return is_temporal_query(query, self.temporal_anchor_cue_keywords)

    def _build_temporal_anchor_queries(self, query: str) -> List[str]:
        return build_temporal_anchor_queries(
            query=query,
            temporal_anchor_enabled=self.temporal_anchor_enabled,
            temporal_anchor_require_temporal_cue=self.temporal_anchor_require_temporal_cue,
            temporal_anchor_cue_keywords=self.temporal_anchor_cue_keywords,
            temporal_anchor_max_options=self.temporal_anchor_max_options,
            temporal_anchor_extra_queries_per_option=self.temporal_anchor_extra_queries_per_option,
        )

    def _merge_anchor_chunks(
        self,
        *,
        base_chunks: List[Dict[str, object]],
        extra_chunks: List[Dict[str, object]],
        additive_limit: int,
    ) -> List[Dict[str, object]]:
        return merge_anchor_chunks(
            base_chunks=base_chunks,
            extra_chunks=extra_chunks,
            additive_limit=additive_limit,
        )

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

        if self.temporal_anchor_enabled and hasattr(self.mid_memory, "search_chunks_global_with_limit"):
            anchor_queries = self._build_temporal_anchor_queries(query)
            if anchor_queries:
                anchor_chunks: List[Dict[str, object]] = []
                for aq in anchor_queries:
                    anchor_chunks.extend(
                        self.mid_memory.search_chunks_global_with_limit(
                            aq,
                            topic_score_map=topic_score_map,
                            top_n=self.temporal_anchor_top_n_per_query,
                        )
                    )
                if anchor_chunks:
                    reranked_chunks = self._merge_anchor_chunks(
                        base_chunks=reranked_chunks,
                        extra_chunks=self._dedup_chunks_keep_best(anchor_chunks),
                        additive_limit=self.temporal_anchor_additive_limit,
                    )
                    logger.info(
                        "MemoryManager.retrieve_context: "
                        f"temporal_anchor_queries={len(anchor_queries)}, "
                        f"anchor_candidates={len(anchor_chunks)}."
                    )
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
            _topics,
            chunks,
            evidence_sentences,
            candidates,
            fallback_answer,
            evidence_candidate,
            best_evidence,
            best_candidate,
        ) = self._prepare_answer_inputs(query, precomputed_context)

        prompt_text = self._build_generation_prompt(
            input_text=input_text,
            chunks=chunks,
            candidates=candidates,
            best_evidence=best_evidence,
            fallback_answer=fallback_answer,
            evidence_candidate=evidence_candidate,
        )
        ai_response, fallback_path, not_found_reason = self._generate_with_fallback(
            input_text=input_text,
            query=query,
            prompt_text=prompt_text,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            fallback_answer=fallback_answer,
            evidence_candidate=evidence_candidate,
        )
        logger.info(
            "MemoryManager.answer_debug: "
            f"fallback_path={fallback_path}, "
            f"not_found_reason={not_found_reason or 'none'}, "
            f"best_evidence='{best_evidence}', "
            f"best_candidate='{best_candidate}', "
            f"fallback_answer='{fallback_answer}', "
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
        List[Dict[str, object]],
        List[Dict[str, object]],
        List[Dict[str, object]],
        str,
        Optional[Dict[str, str]],
        str,
        str,
    ]:
        return self.chat_runtime.prepare_answer_inputs(
            query=query,
            precomputed_context=precomputed_context,
        )

    def _set_prompt_eval_chunks(
        self, generation_context: List[Dict[str, str]] | str
    ) -> None:
        if isinstance(generation_context, str):
            text = str(generation_context).strip()
            self.last_prompt_eval_chunks = [{"section": "prompt", "text": text}] if text else []
            return
        sections: List[Dict[str, str]] = []
        for item in generation_context:
            section = str(item.get("section", "")).strip()
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            payload = {"text": text}
            if section:
                payload["section"] = section
            sections.append(payload)
        self.last_prompt_eval_chunks = sections

    def _build_generation_prompt(
        self,
        *,
        input_text: str,
        chunks: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        best_evidence: str,
        fallback_answer: str,
        evidence_candidate: Optional[Dict[str, str]],
    ) -> str:
        return self.chat_runtime.build_generation_prompt(
            input_text=input_text,
            chunks=chunks,
            candidates=candidates,
            best_evidence=best_evidence,
            fallback_answer=fallback_answer,
            evidence_candidate=evidence_candidate,
        )

    def _build_graph_context(self, query: str, chunks: List[Dict[str, object]]) -> str:
        return self.chat_runtime.build_graph_context(query, chunks)

    def offline_build_long_graph_from_chunks(
        self,
        chunks: List[Dict[str, object]],
        query: Optional[str] = None,
    ) -> int:
        """Offline stage: build long-memory graph from retrieved chunks via 4B extractor."""
        if not self.offline_graph_build_enabled:
            return 0
        accepted = int(
            self.long_memory.ingest_from_chunks(
                chunks=self._rank_chunks_for_graph_build(list(chunks)),
                top_chunks=self.offline_graph_build_top_chunks,
                max_chars_per_chunk=self.offline_graph_build_chunk_max_chars,
            )
        )
        if accepted > 0:
            return accepted
        if (not self.graph_build_retry_enabled) or (not query):
            return accepted
        if not hasattr(self.mid_memory, "search_chunks_global_with_limit"):
            return accepted

        topic_score_map: Dict[str, float] = {}
        for item in chunks:
            topic_id = str(item.get("topic_id", "")).strip()
            if not topic_id:
                continue
            topic_score_map[topic_id] = max(
                float(item.get("score", 0.0)),
                float(topic_score_map.get(topic_id, 0.0)),
            )

        expanded = self.mid_memory.search_chunks_global_with_limit(
            str(query),
            topic_score_map=topic_score_map,
            top_n=self.graph_build_retry_expanded_top_n,
        )
        combined = self._dedup_chunks_keep_best(list(chunks) + list(expanded))

        if self.graph_build_retry_use_temporal_anchors:
            anchor_queries = self._build_temporal_anchor_queries(str(query))
            for aq in anchor_queries[: max(0, self.graph_build_retry_anchor_query_limit)]:
                combined.extend(
                    self.mid_memory.search_chunks_global_with_limit(
                        aq,
                        topic_score_map=topic_score_map,
                        top_n=self.graph_build_retry_anchor_top_n_per_query,
                    )
                )
            combined = self._dedup_chunks_keep_best(combined)

        retry_accepted = int(
            self.long_memory.ingest_from_chunks(
                chunks=self._rank_chunks_for_graph_build(combined),
                top_chunks=self.graph_build_retry_ingest_top_chunks,
                max_chars_per_chunk=self.offline_graph_build_chunk_max_chars,
            )
        )
        logger.info(
            "MemoryManager.offline_graph_retry: "
            f"base_chunks={len(chunks)}, expanded_chunks={len(combined)}, "
            f"accepted_before={accepted}, accepted_after={retry_accepted}."
        )
        return retry_accepted

    def _generate_with_fallback(
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
        return self.chat_runtime.generate_with_fallback(
            input_text=input_text,
            query=query,
            prompt_text=prompt_text,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            fallback_answer=fallback_answer,
            evidence_candidate=evidence_candidate,
        )

    def get_last_prompt_eval_chunks(self) -> List[Dict[str, str]]:
        """Return prompt-grounded chunks used by the most recent chat call."""
        return [
            {
                "section": str(item.get("section", "")),
                "text": str(item.get("text", "")),
            }
            for item in self.last_prompt_eval_chunks
        ]

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
