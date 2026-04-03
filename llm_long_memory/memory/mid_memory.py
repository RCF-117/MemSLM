"""Topic-aware mid-term memory with SQLite and two-stage hybrid retrieval."""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from llm_long_memory.memory import mid_memory_chunking as chunking
from llm_long_memory.memory import mid_memory_retrieval as retrieval
from llm_long_memory.memory import mid_memory_summary as summary
from llm_long_memory.memory.mid_memory_store import MidMemoryStore
from llm_long_memory.memory import mid_memory_topicing as topicing
from llm_long_memory.utils.embedding import embed
from llm_long_memory.utils.helpers import load_config
from llm_long_memory.utils.logger import logger


Message = Dict[str, Any]
Topic = Dict[str, Any]
Chunk = Dict[str, Any]


class MidMemory:
    """Persistent topic memory with dynamic chunking and temporal-aware retrieval."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize config, storage, and runtime buffers."""
        self.config = config or load_config()
        self.memory_cfg = self.config["memory"]["mid_memory"]
        self.retrieval_cfg = self.config["retrieval"]
        self.llm_cfg = self.config["llm"]
        self.eval_cfg = self.config["evaluation"]
        self.embedding_dim = int(self.config["embedding"]["dim"])
        self.time_cfg = dict(self.config.get("time_extraction", {}))
        self.time_regexes = [re.compile(str(x), flags=re.IGNORECASE) for x in self.time_cfg.get("regexes", [])]

        self.max_size = int(self.memory_cfg["max_size"])
        self.min_chunk_size = int(self.memory_cfg["min_chunk_size"])
        self.max_chunk_size = int(self.memory_cfg["max_chunk_size"])
        self.message_chunk_max_chars = int(self.memory_cfg["message_chunk_max_chars"])
        self.message_chunk_overlap_chars = int(self.memory_cfg["message_chunk_overlap_chars"])
        self.message_chunk_min_chars = int(self.memory_cfg["message_chunk_min_chars"])
        self.message_chunk_boundary_window_chars = int(
            self.memory_cfg["message_chunk_boundary_window_chars"]
        )
        self.chunk_similarity_threshold = float(self.memory_cfg["chunk_similarity_threshold"])
        self.embedding_min_length = int(self.memory_cfg["embedding_min_length"])
        self.topic_similarity_threshold = float(self.memory_cfg["topic_similarity_threshold"])
        self.topic_multi_assign_top_k = int(self.memory_cfg["topic_multi_assign_top_k"])
        self.topic_inactive_steps = int(self.memory_cfg["topic_inactive_steps"])
        self.topic_merge_threshold = float(self.memory_cfg["topic_merge_threshold"])
        self.enable_topic_merge = bool(self.memory_cfg["enable_topic_merge"])
        self.topic_min_chunks_before_merge = int(self.memory_cfg.get("topic_min_chunks_before_merge", 1))
        self.summary_update_every = int(self.memory_cfg["summary_update_every"])
        self.summary_min_chunk_count = int(self.memory_cfg["summary_min_chunk_count"])
        self.summary_cooldown_steps = int(self.memory_cfg["summary_cooldown_steps"])
        self.summary_on_new_topic = bool(self.memory_cfg["summary_on_new_topic"])
        self.summary_enabled = bool(self.memory_cfg["summary_enabled"])
        self.sqlite_busy_timeout_ms = int(self.memory_cfg["sqlite_busy_timeout_ms"])
        self.sqlite_journal_mode = str(self.memory_cfg["sqlite_journal_mode"])
        self.sqlite_synchronous = str(self.memory_cfg["sqlite_synchronous"])

        self.role_enabled = bool(self.memory_cfg["role"]["enable"])
        self.role_weights = {
            "user": float(self.memory_cfg["role"]["user_weight"]),
            "assistant": float(self.memory_cfg["role"]["assistant_weight"]),
            "system": float(self.memory_cfg["role"]["system_weight"]),
        }
        self.centroid_roles = {
            str(x).strip().lower() for x in self.memory_cfg["role"]["centroid_roles"]
        }
        centroid_role_weights_cfg = dict(self.memory_cfg["role"].get("centroid_role_weights", {}))
        self.centroid_role_weights = {
            str(k).strip().lower(): float(v) for k, v in centroid_role_weights_cfg.items()
        }
        self.non_user_threshold_multiplier = float(
            self.memory_cfg["role"]["non_user_threshold_multiplier"]
        )
        self.role_boundary_flush = bool(self.memory_cfg["chunking"]["role_boundary_flush"])
        self.use_buffer_centroid_similarity = bool(
            self.memory_cfg["chunking"]["use_buffer_centroid_similarity"]
        )
        self.stopwords = {str(x).lower() for x in self.memory_cfg["stopwords"]}

        self.top_k = int(self.retrieval_cfg["top_k"])
        self.chunks_per_topic = int(self.retrieval_cfg["chunks_per_topic"])
        self.hybrid_alpha = float(self.retrieval_cfg["hybrid_alpha"])
        self.keyword_weight = float(self.retrieval_cfg["keyword_weight"])
        self.summary_weight = float(self.retrieval_cfg["summary_weight"])
        self.recent_topic_weight = float(self.retrieval_cfg["recent_topic_weight"])
        self.recent_topic_apply_margin = float(self.retrieval_cfg["recent_topic_apply_margin"])
        self.chunk_topic_weight = float(self.retrieval_cfg["chunk_topic_weight"])
        self.time_weight = float(self.retrieval_cfg["time_weight"])
        lexical_cfg = dict(self.retrieval_cfg.get("lexical_search", {}))
        self.lexical_search_enabled = bool(lexical_cfg.get("enabled", True))
        self.lexical_bm25_top_n = int(lexical_cfg.get("bm25_top_n", 40))
        fusion_cfg = dict(self.retrieval_cfg.get("fusion", {}))
        self.fusion_rrf_k = int(fusion_cfg.get("rrf_k", 60))
        self.fusion_dense_weight = float(fusion_cfg.get("dense_weight", 1.0))
        self.fusion_lexical_weight = float(fusion_cfg.get("lexical_weight", 1.0))
        self.fusion_topic_weight = float(fusion_cfg.get("topic_weight", 0.15))
        self.temporal_boost_max = float(self.retrieval_cfg["temporal_boost_max"])
        self.retrieval_diversity_enabled = bool(self.retrieval_cfg["diversity"]["enabled"])
        self.retrieval_diversity_similarity_threshold = float(
            self.retrieval_cfg["diversity"]["similarity_threshold"]
        )
        global_chunk_cfg = dict(self.retrieval_cfg["global_chunk_retrieval"])
        self.global_chunk_retrieval_enabled = bool(global_chunk_cfg["enabled"])
        self.global_chunk_top_n = int(global_chunk_cfg["top_n"])
        self.global_chunk_rrf_k = int(global_chunk_cfg["rrf_k"])
        self.global_chunk_dense_weight = float(global_chunk_cfg["dense_weight"])
        self.global_chunk_lexical_weight = float(global_chunk_cfg["lexical_weight"])
        self.global_chunk_keyword_weight = float(global_chunk_cfg["keyword_weight"])
        self.global_chunk_topic_prior_weight = float(global_chunk_cfg["topic_prior_weight"])
        self.global_chunk_prefilter_enabled = bool(global_chunk_cfg["prefilter_enabled"])
        self.global_chunk_prefilter_top_n = int(global_chunk_cfg["prefilter_top_n"])
        self.global_chunk_lexical_prefilter_top_n = int(
            global_chunk_cfg["lexical_prefilter_top_n"]
        )
        self.global_chunk_prefilter_topic_top_k = int(global_chunk_cfg["prefilter_topic_top_k"])
        self.global_chunk_recent_fallback_n = int(global_chunk_cfg["recent_fallback_n"])
        self.global_chunk_dedup_by_text = bool(global_chunk_cfg["dedup_by_text"])
        topic_expansion_cfg = dict(self.retrieval_cfg["topic_expansion"])
        self.topic_expansion_enabled = bool(topic_expansion_cfg["enabled"])
        self.topic_expansion_per_topic_limit = int(topic_expansion_cfg["per_topic_limit"])

        self.summary_model = str(self.llm_cfg["summary_model"])
        self.llm_host = str(self.llm_cfg["host"]).rstrip("/")
        self.temperature = float(self.llm_cfg["temperature"])
        self.request_timeout_sec = int(self.llm_cfg["request_timeout_sec"])
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        self.buffer: List[Dict[str, Any]] = []
        self.current_step = 0
        self.temporal_weight_disabled = False

        self.store = MidMemoryStore(
            database_file=str(self.memory_cfg["database_file"]),
            sqlite_busy_timeout_ms=self.sqlite_busy_timeout_ms,
            sqlite_journal_mode=self.sqlite_journal_mode,
            sqlite_synchronous=self.sqlite_synchronous,
            lexical_search_enabled=self.lexical_search_enabled,
            eval_cfg=self.eval_cfg,
        )
        self.conn = self.store.conn
        self.eval_store = self.store.eval_store
        self.lexical_search_enabled = self.store.lexical_search_enabled
        self.db_path = self.store.db_path
        self.current_step = self.store.load_current_step()
        logger.info(f"MidMemory initialized with SQLite at {self.db_path}.")

    @staticmethod
    def _arr_to_blob(arr: np.ndarray) -> bytes:
        return arr.astype(np.float32).tobytes()

    def _blob_to_arr(self, blob: bytes | None) -> np.ndarray:
        if not blob:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size == self.embedding_dim:
            return arr
        fixed = np.zeros(self.embedding_dim, dtype=np.float32)
        limit = min(arr.size, self.embedding_dim)
        fixed[:limit] = arr[:limit]
        return fixed

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _temporal_weight_base(delta: int) -> float:
        return 1.0 / (1.0 + float(max(0, delta)))

    def _temporal_weight(self, delta: int) -> float:
        if self.temporal_weight_disabled:
            return 0.0
        return self._temporal_weight_base(delta)

    def _index_chunk_fts(self, chunk_id: int, topic_id: str, text: str) -> None:
        self.store.index_chunk_fts(chunk_id=chunk_id, topic_id=topic_id, text=text)

    def _delete_chunk_fts(self, chunk_ids: Sequence[int]) -> None:
        self.store.delete_chunk_fts(chunk_ids=chunk_ids)

    def _rebuild_chunk_fts(self) -> None:
        self.store.rebuild_chunk_fts()

    def _lexical_rank_map(self, topic_id: str, query: str) -> Dict[int, int]:
        return self.store.lexical_rank_map(
            topic_id=topic_id,
            query=query,
            tokenize=self._tokenize,
            bm25_top_n=self.lexical_bm25_top_n,
        )

    def _lexical_rank_map_global(self, query: str, top_n: int) -> Dict[int, int]:
        return self.store.lexical_rank_map_global(
            query=query,
            tokenize=self._tokenize,
            bm25_top_n=top_n,
        )

    def _role_weight(self, role: str) -> float:
        if not self.role_enabled:
            return 1.0
        return float(self.role_weights.get(role, self.role_weights["user"]))

    def _normalize_token(self, token: str) -> str:
        cleaned = "".join(ch for ch in token.lower().strip() if ch.isalnum())
        if not cleaned or cleaned in self.stopwords:
            return ""
        return cleaned

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in (self._normalize_token(x) for x in text.split()) if t]

    def _split_long_text(self, text: str) -> List[str]:
        return chunking.split_long_text(
            text=text,
            message_chunk_max_chars=self.message_chunk_max_chars,
            message_chunk_min_chars=self.message_chunk_min_chars,
            message_chunk_overlap_chars=self.message_chunk_overlap_chars,
            message_chunk_boundary_window_chars=self.message_chunk_boundary_window_chars,
        )

    def _extract_time_terms(self, text: str) -> List[str]:
        return chunking.extract_time_terms(text=text, time_regexes=self.time_regexes)

    def add(self, message: Message) -> None:
        """Add a role-aware message and dynamically chunk by semantic shift."""
        role = str(message.get("role", "user")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            return
        if role not in self.role_weights:
            role = "user"
        session_id = str(message.get("session_id", "")).strip()
        session_date = str(message.get("session_date", "")).strip()
        has_answer = bool(message.get("has_answer", False))

        parts = self._split_long_text(content)
        for index, part in enumerate(parts):
            self.current_step += 1
            msg_embedding = embed(part, self.embedding_dim)
            msg_times = self._extract_time_terms(part) or self._extract_time_terms(session_date)

            # Avoid mixing different sessions into one semantic chunk.
            if self.buffer and session_id:
                last_sid = str(self.buffer[-1].get("session_id", ""))
                if last_sid and last_sid != session_id:
                    self._flush_buffer(force=True)

            if self.buffer:
                if (
                    self.role_boundary_flush
                    and role != str(self.buffer[-1].get("role", "user"))
                    and len(self.buffer) >= self.min_chunk_size
                ):
                    self._flush_buffer()
                centroid = self._buffer_centroid_embedding() if self.use_buffer_centroid_similarity else self.buffer[-1]["embedding"]
                sim = self._cosine(centroid, msg_embedding)
                sim_final = sim * self._role_weight(role)
                if (
                    sim_final < self.chunk_similarity_threshold
                    and len(self.buffer) >= self.min_chunk_size
                ):
                    self._flush_buffer()

            self.buffer.append(
                {
                    "role": role,
                    "text": part,
                    "embedding": msg_embedding,
                    "session_id": session_id,
                    "session_date": session_date,
                    "has_answer": has_answer,
                    "times": msg_times,
                }
            )
            logger.debug(
                f"MidMemory.add: step={self.current_step}, role={role}, "
                f"part={index + 1}/{len(parts)}, part_len={len(part)}, buffer_size={len(self.buffer)}."
            )

            if len(self.buffer) >= self.max_chunk_size:
                self._flush_buffer()

    def flush_pending(self) -> None:
        """Flush pending buffer if it satisfies minimum chunk size."""
        if self.buffer:
            self._flush_buffer(force=True)

    def _flush_buffer(self, force: bool = False) -> None:
        if not self.buffer:
            return
        if (not force) and len(self.buffer) < self.min_chunk_size:
            return
        chunk_text = "\n".join(str(m["text"]) for m in self.buffer)
        chunk_embedding = np.mean(
            np.stack([m["embedding"] for m in self.buffer]), axis=0
        ).astype(np.float32)
        dominant_role = self._dominant_role([m["role"] for m in self.buffer])
        dominant_session_id = self._dominant_text([str(m.get("session_id", "")) for m in self.buffer])
        dominant_session_date = self._dominant_text([str(m.get("session_date", "")) for m in self.buffer])
        chunk_has_answer = 1 if any(bool(m.get("has_answer", False)) for m in self.buffer) else 0
        times: List[str] = []
        for m in self.buffer:
            for t in list(m.get("times", []) or []):
                tt = str(t).strip().lower()
                if tt and tt not in times:
                    times.append(tt)
        self.buffer.clear()
        self._add_chunk(
            chunk_text,
            chunk_embedding,
            dominant_role,
            dominant_session_id,
            dominant_session_date,
            chunk_has_answer,
            times,
        )

    @staticmethod
    def _dominant_text(values: Sequence[str]) -> str:
        return chunking.dominant_text(values)

    @staticmethod
    def _dominant_role(roles: Sequence[str]) -> str:
        return chunking.dominant_role(roles)

    def _buffer_centroid_embedding(self) -> np.ndarray:
        return chunking.buffer_centroid_embedding(
            buffer=self.buffer,
            embedding_dim=self.embedding_dim,
        )

    def _add_chunk(
        self,
        chunk_text: str,
        chunk_embedding: np.ndarray,
        chunk_role: str,
        chunk_session_id: str,
        chunk_session_date: str,
        chunk_has_answer: int,
        chunk_times: Sequence[str],
    ) -> None:
        topic_ids = self._assign_topics(chunk_embedding, chunk_role)
        created_new_topic = False

        if not topic_ids:
            topic_ids = [self._create_topic(chunk_embedding)]
            created_new_topic = True
            logger.info(f"MidMemory._add_chunk: created topic {topic_ids[0]}.")
        else:
            logger.info("MidMemory._add_chunk: matched topics=" + ",".join(topic_ids))

        for topic_id in topic_ids:
            cursor = self.conn.execute(
                """
                INSERT INTO chunks(
                  topic_id, text, chunk_embedding, chunk_role,
                  chunk_session_id, chunk_session_date, chunk_has_answer, chunk_times
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    topic_id,
                    chunk_text,
                    self._arr_to_blob(chunk_embedding),
                    chunk_role,
                    chunk_session_id,
                    chunk_session_date,
                    int(chunk_has_answer),
                    json.dumps(list(chunk_times)),
                ),
            )
            chunk_id = int(cursor.lastrowid)
            self._index_chunk_fts(chunk_id, topic_id, chunk_text)
            self._update_topic(topic_id, created_new_topic)

        self._merge_topics()
        self._deactivate_topics()
        self._enforce_fifo_limit()
        self.conn.commit()

    def _assign_topics(self, chunk_embedding: np.ndarray, chunk_role: str) -> List[str]:
        return topicing.assign_topics(self, chunk_embedding, chunk_role)

    def _create_topic(self, topic_embedding: np.ndarray) -> str:
        return topicing.create_topic(self, topic_embedding)

    def _update_topic(self, topic_id: str, created_new_topic: bool) -> None:
        self._recompute_topic_embedding(topic_id)
        self.conn.execute(
            "UPDATE topics SET last_updated_step = ?, active = 1 WHERE topic_id = ?",
            (self.current_step, topic_id),
        )
        self._maybe_update_summary_keywords(topic_id, created_new_topic)

    def _recompute_topic_embedding(self, topic_id: str) -> None:
        topicing.recompute_topic_embedding(self, topic_id)

    def _maybe_update_summary_keywords(self, topic_id: str, created_new_topic: bool) -> None:
        summary.maybe_update_summary_keywords(self, topic_id, created_new_topic)

    def _summarize_with_model(self, raw_text: str) -> str:
        return summary.summarize_with_model(self, raw_text)

    @staticmethod
    def _extractive_clip(summary: str, source: str) -> str:
        return summary.extractive_clip(summary, source)

    def _extract_keywords(self, text: str) -> List[str]:
        return summary.extract_keywords(self, text)

    def _merge_topics(self) -> None:
        topicing.merge_topics(self)

    def _merge_topic_pair(self, target_topic_id: str, source_topic_id: str) -> None:
        topicing.merge_topic_pair(self, target_topic_id, source_topic_id)

    def _deduplicate_topic_chunks(self, topic_id: str) -> None:
        topicing.deduplicate_topic_chunks(self, topic_id)

    @staticmethod
    def _merge_text(left: str, right: str) -> str:
        return topicing.merge_text(left, right)

    def _merge_keywords(self, left: Sequence[str], right: Sequence[str]) -> List[str]:
        return topicing.merge_keywords(self, left, right)

    def _deactivate_topics(self) -> None:
        topicing.deactivate_topics(self)

    def _enforce_fifo_limit(self) -> None:
        topicing.enforce_fifo_limit(self)

    @staticmethod
    def _keyword_overlap(query_tokens: Sequence[str], target_tokens: Sequence[str]) -> float:
        return retrieval.keyword_overlap(query_tokens, target_tokens)

    def _select_diverse_topics(self, scored: Sequence[Topic]) -> List[Topic]:
        return retrieval.select_diverse_topics(self, scored)

    def search(self, query: str) -> List[Topic]:
        """Hybrid level 1: topic retrieval."""
        return retrieval.search_topics(self, query)

    def rerank_chunks(self, query: str, topics: Sequence[Topic]) -> List[Chunk]:
        """Hybrid level 2: chunk rerank via dense+lexical rank fusion (RRF)."""
        return retrieval.rerank_chunks(self, query, topics)

    def search_chunks_global(
        self,
        query: str,
        topic_score_map: Optional[Dict[str, float]] = None,
    ) -> List[Chunk]:
        """Global chunk retrieval without topic gate (dense + lexical + keyword + topic prior)."""
        return retrieval.rerank_chunks_global(
            self,
            query=query,
            topic_score_map=topic_score_map or {},
        )

    def set_temporal_weight_disabled(self, disabled: bool) -> None:
        """Set temporal weighting behavior at runtime (useful for oracle eval)."""
        self.temporal_weight_disabled = bool(disabled)
        logger.info(
            f"MidMemory temporal weighting disabled={self.temporal_weight_disabled}."
        )

    def get_topic_chunks(self, topic_id: str) -> List[str]:
        """Return all chunk texts for one topic."""
        rows = self.conn.execute(
            "SELECT text FROM chunks WHERE topic_id = ? ORDER BY chunk_id ASC",
            (topic_id,),
        ).fetchall()
        return [str(r["text"]) for r in rows]

    def debug_stats(self) -> Dict[str, int]:
        """Return topic/chunk statistics."""
        return self.store.debug_stats()

    def clear_all(self) -> None:
        """Clear all persisted mid-memory records and in-memory buffers."""
        self.store.clear_all()
        self.buffer.clear()
        self.current_step = 0
        logger.info("MidMemory.clear_all: cleared topics, chunks, and runtime buffer.")

    def log_eval_run_start(
        self, run_id: str, dataset_path: str, isolated: bool, commit: bool = True
    ) -> None:
        """Insert evaluation run start metadata."""
        self.eval_store.log_eval_run_start(run_id, dataset_path, isolated, commit=commit)

    def log_eval_result(
        self,
        run_id: str,
        question_id: str,
        question_type: str,
        question: str,
        expected_answer: str,
        prediction: str,
        is_match: bool,
        evidence_hit: bool | None = None,
        evidence_recall: float | None = None,
        answer_span_hit: bool | None = None,
        support_sentence_hit: bool | None = None,
        graph_retrieval_hit: bool | None = None,
        retrieved_session_ids: Sequence[str] | None = None,
        commit: bool = True,
    ) -> None:
        """Insert one per-instance eval result row."""
        self.eval_store.log_eval_result(
            run_id=run_id,
            question_id=question_id,
            question_type=question_type,
            question=question,
            expected_answer=expected_answer,
            prediction=prediction,
            is_match=is_match,
            evidence_hit=evidence_hit,
            evidence_recall=evidence_recall,
            answer_span_hit=answer_span_hit,
            support_sentence_hit=support_sentence_hit,
            graph_retrieval_hit=graph_retrieval_hit,
            retrieved_session_ids=retrieved_session_ids,
            commit=commit,
        )

    def log_eval_group_result(
        self,
        run_id: str,
        group_key: str,
        total: int,
        matched: int,
        accuracy: float,
        commit: bool = True,
    ) -> None:
        """Insert one grouped eval summary row."""
        self.eval_store.log_eval_group_result(
            run_id=run_id,
            group_key=group_key,
            total=total,
            matched=matched,
            accuracy=accuracy,
            commit=commit,
        )

    def log_eval_run_finish(
        self,
        run_id: str,
        total: int,
        matched: int,
        accuracy: float,
        retrieval_answer_span_hit_rate: float | None = None,
        retrieval_support_sentence_hit_rate: float | None = None,
        retrieval_evidence_hit_rate: float | None = None,
        commit: bool = True,
    ) -> None:
        """Update final metrics and finished_at for one eval run."""
        self.eval_store.log_eval_run_finish(
            run_id=run_id,
            total=total,
            matched=matched,
            accuracy=accuracy,
            retrieval_answer_span_hit_rate=retrieval_answer_span_hit_rate,
            retrieval_support_sentence_hit_rate=retrieval_support_sentence_hit_rate,
            retrieval_evidence_hit_rate=retrieval_evidence_hit_rate,
            commit=commit,
        )

    def commit(self) -> None:
        """Commit current transaction."""
        self.store.commit()

    def close(self) -> None:
        """Flush pending state and close SQLite connection safely."""
        try:
            self.flush_pending()
        finally:
            try:
                self.store.commit()
            finally:
                self.store.close()

    def __enter__(self) -> "MidMemory":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
