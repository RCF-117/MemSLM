"""Chunk-first mid-term memory with SQLite and hybrid retrieval."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from llm_long_memory.memory import mid_memory_chunking as chunking
from llm_long_memory.memory import mid_memory_retrieval as retrieval
from llm_long_memory.memory.mid_memory_store import MidMemoryStore
from llm_long_memory.utils.embedding import embed
from llm_long_memory.utils.helpers import load_config
from llm_long_memory.utils.logger import logger


Message = Dict[str, Any]
Chunk = Dict[str, Any]


class MidMemory:
    """Persistent chunk memory with dynamic chunking and temporal-aware retrieval."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize config, storage, and runtime buffers."""
        self.config = config or load_config()
        self.memory_cfg = self.config["memory"]["mid_memory"]
        self.retrieval_cfg = self.config["retrieval"]
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
        self.sqlite_busy_timeout_ms = int(self.memory_cfg["sqlite_busy_timeout_ms"])
        self.sqlite_journal_mode = str(self.memory_cfg["sqlite_journal_mode"])
        self.sqlite_synchronous = str(self.memory_cfg["sqlite_synchronous"])
        self.sqlite_checkpoint_on_commit = bool(self.memory_cfg.get("sqlite_checkpoint_on_commit", False))
        self.sqlite_checkpoint_mode = str(self.memory_cfg.get("sqlite_checkpoint_mode", "PASSIVE"))

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
        self.hybrid_alpha = float(self.retrieval_cfg["hybrid_alpha"])
        self.keyword_weight = float(self.retrieval_cfg["keyword_weight"])
        self.time_weight = float(self.retrieval_cfg["time_weight"])
        lexical_cfg = dict(self.retrieval_cfg["lexical_search"])
        self.lexical_search_enabled = bool(lexical_cfg["enabled"])
        self.lexical_bm25_top_n = int(lexical_cfg["bm25_top_n"])
        fusion_cfg = dict(self.retrieval_cfg["fusion"])
        self.fusion_rrf_k = int(fusion_cfg["rrf_k"])
        self.fusion_dense_weight = float(fusion_cfg["dense_weight"])
        self.fusion_lexical_weight = float(fusion_cfg["lexical_weight"])
        self.temporal_boost_max = float(self.retrieval_cfg["temporal_boost_max"])
        global_chunk_cfg = dict(self.retrieval_cfg["global_chunk_retrieval"])
        self.global_chunk_retrieval_enabled = bool(global_chunk_cfg["enabled"])
        self.global_chunk_top_n = int(global_chunk_cfg["top_n"])
        self.global_chunk_rrf_k = int(global_chunk_cfg["rrf_k"])
        self.global_chunk_dense_weight = float(global_chunk_cfg["dense_weight"])
        self.global_chunk_lexical_weight = float(global_chunk_cfg["lexical_weight"])
        self.global_chunk_keyword_weight = float(global_chunk_cfg["keyword_weight"])
        self.global_chunk_prefilter_enabled = bool(global_chunk_cfg["prefilter_enabled"])
        self.global_chunk_prefilter_top_n = int(global_chunk_cfg["prefilter_top_n"])
        self.global_chunk_lexical_prefilter_top_n = int(
            global_chunk_cfg["lexical_prefilter_top_n"]
        )
        self.global_chunk_recent_fallback_n = int(global_chunk_cfg["recent_fallback_n"])
        self.global_chunk_dedup_by_text = bool(global_chunk_cfg["dedup_by_text"])
        global_sentence_cfg = dict(self.retrieval_cfg.get("global_sentence_retrieval", {}))
        self.global_sentence_retrieval_enabled = bool(
            global_sentence_cfg.get("enabled", True)
        )
        self.global_sentence_top_n = int(global_sentence_cfg.get("top_n", 48))
        self.global_sentence_rrf_k = int(global_sentence_cfg.get("rrf_k", 60))
        self.global_sentence_dense_weight = float(
            global_sentence_cfg.get("dense_weight", 1.2)
        )
        self.global_sentence_lexical_weight = float(
            global_sentence_cfg.get("lexical_weight", 1.0)
        )
        self.global_sentence_keyword_weight = float(
            global_sentence_cfg.get("keyword_weight", 0.1)
        )
        self.global_sentence_prefilter_enabled = bool(
            global_sentence_cfg.get("prefilter_enabled", True)
        )
        self.global_sentence_prefilter_top_n = int(
            global_sentence_cfg.get("prefilter_top_n", 384)
        )
        self.global_sentence_lexical_prefilter_top_n = int(
            global_sentence_cfg.get("lexical_prefilter_top_n", 384)
        )
        self.global_sentence_recent_fallback_n = int(
            global_sentence_cfg.get("recent_fallback_n", 128)
        )
        self.global_sentence_dedup_by_text = bool(
            global_sentence_cfg.get("dedup_by_text", True)
        )
        self.global_sentence_max_chars = int(
            global_sentence_cfg.get("sentence_max_chars", 320)
        )
        self.global_sentence_min_chars = int(
            global_sentence_cfg.get("sentence_min_chars", 24)
        )

        self.buffer: List[Dict[str, Any]] = []
        self.current_step = 0
        self.temporal_weight_disabled = False

        self.store = MidMemoryStore(
            database_file=str(self.memory_cfg["database_file"]),
            eval_database_file=str(self.config["evaluation"]["database_file"]),
            sqlite_busy_timeout_ms=self.sqlite_busy_timeout_ms,
            sqlite_journal_mode=self.sqlite_journal_mode,
            sqlite_synchronous=self.sqlite_synchronous,
            sqlite_checkpoint_on_commit=self.sqlite_checkpoint_on_commit,
            sqlite_checkpoint_mode=self.sqlite_checkpoint_mode,
            lexical_search_enabled=self.lexical_search_enabled,
            eval_cfg=self.eval_cfg,
        )
        self.conn = self.store.conn
        self.eval_store = self.store.eval_store
        self.lexical_search_enabled = self.store.lexical_search_enabled
        self.db_path = self.store.db_path
        self.eval_db_path = self.store.eval_db_path
        self.current_step = self.store.load_current_step()
        logger.info(
            f"MidMemory initialized with SQLite at {self.db_path}; eval db at {self.eval_db_path}."
        )

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

    def _index_chunk_fts(self, chunk_id: int, text: str) -> None:
        self.store.index_chunk_fts(chunk_id=chunk_id, text=text)

    def _delete_chunk_fts(self, chunk_ids: Sequence[int]) -> None:
        self.store.delete_chunk_fts(chunk_ids=chunk_ids)

    def _rebuild_chunk_fts(self) -> None:
        self.store.rebuild_chunk_fts()

    def _index_sentence_fts(self, sentence_id: int, text: str) -> None:
        self.store.index_sentence_fts(sentence_id=sentence_id, text=text)

    def _delete_sentence_fts(self, sentence_ids: Sequence[int]) -> None:
        self.store.delete_sentence_fts(sentence_ids=sentence_ids)

    def _rebuild_sentence_fts(self) -> None:
        self.store.rebuild_sentence_fts()

    def _lexical_rank_map_global(self, query: str, top_n: int) -> Dict[int, int]:
        return self.store.lexical_rank_map_global(
            query=query,
            tokenize=self._tokenize,
            bm25_top_n=top_n,
        )

    def _lexical_rank_map_sentences(self, query: str, top_n: int) -> Dict[int, int]:
        return self.store.lexical_rank_map_sentences(
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
                    "source_part_index": int(index),
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
        weighted_embeddings: List[np.ndarray] = []
        weights: List[float] = []
        for m in self.buffer:
            vec = np.asarray(m["embedding"], dtype=np.float32)
            token_len = max(1, len(self._tokenize(str(m.get("text", "")))))
            role_weight = self._role_weight(str(m.get("role", "user")))
            weight = max(1e-6, float(token_len) * float(role_weight))
            weighted_embeddings.append(vec)
            weights.append(weight)
        if weighted_embeddings and weights:
            matrix = np.stack(weighted_embeddings)
            chunk_embedding = np.average(
                matrix,
                axis=0,
                weights=np.asarray(weights, dtype=np.float32),
            ).astype(np.float32)
        else:
            chunk_embedding = np.mean(
                np.stack([m["embedding"] for m in self.buffer]), axis=0
            ).astype(np.float32)
        dominant_role = self._dominant_role([m["role"] for m in self.buffer])
        role_hist = self._role_hist([str(m.get("role", "user")) for m in self.buffer])
        dominant_session_id = self._dominant_text([str(m.get("session_id", "")) for m in self.buffer])
        dominant_session_date = self._dominant_text([str(m.get("session_date", "")) for m in self.buffer])
        secondary_session_dates = self._secondary_values(
            values=[str(m.get("session_date", "")) for m in self.buffer],
            primary=dominant_session_date,
        )
        chunk_has_answer_count = sum(1 for m in self.buffer if bool(m.get("has_answer", False)))
        chunk_answer_density = float(chunk_has_answer_count) / float(max(1, len(self.buffer)))
        chunk_has_answer = 1 if chunk_has_answer_count > 0 else 0
        times: List[str] = []
        time_sources: List[Dict[str, Any]] = []
        for buffer_index, m in enumerate(self.buffer):
            for t in list(m.get("times", []) or []):
                tt = str(t).strip().lower()
                if not tt:
                    continue
                if tt not in times:
                    times.append(tt)
                time_sources.append(
                    {
                        "time": tt,
                        "buffer_index": int(buffer_index),
                        "source_part_index": int(m.get("source_part_index", -1)),
                    }
                )
        self.buffer.clear()
        self._add_chunk(
            chunk_text,
            chunk_embedding,
            dominant_role,
            role_hist,
            dominant_session_id,
            dominant_session_date,
            secondary_session_dates,
            chunk_has_answer,
            chunk_has_answer_count,
            chunk_answer_density,
            times,
            time_sources,
        )

    @staticmethod
    def _secondary_values(values: Sequence[str], primary: str) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        p = str(primary).strip()
        for value in values:
            v = str(value or "").strip()
            if not v or v == p:
                continue
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
        return out

    @staticmethod
    def _role_hist(roles: Sequence[str]) -> List[str]:
        counts: Dict[str, int] = {}
        first_index: Dict[str, int] = {}
        for idx, role in enumerate(roles):
            r = str(role or "").strip().lower() or "user"
            counts[r] = counts.get(r, 0) + 1
            if r not in first_index:
                first_index[r] = idx
        ordered = sorted(
            counts.keys(),
            key=lambda x: (-counts[x], first_index[x]),
        )
        return ordered

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
        chunk_role_hist: Sequence[str],
        chunk_session_id: str,
        chunk_session_date: str,
        chunk_session_dates: Sequence[str],
        chunk_has_answer: int,
        chunk_has_answer_count: int,
        chunk_answer_density: float,
        chunk_times: Sequence[str],
        chunk_time_sources: Sequence[Dict[str, Any]],
    ) -> None:
        cursor = self.conn.execute(
            """
            INSERT INTO chunks(
              text, chunk_embedding, chunk_role, chunk_role_hist,
              chunk_session_id, chunk_session_date, chunk_session_dates,
              chunk_has_answer, chunk_has_answer_count, chunk_answer_density,
              chunk_times, chunk_time_sources
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_text,
                self._arr_to_blob(chunk_embedding),
                chunk_role,
                json.dumps(list(chunk_role_hist)),
                chunk_session_id,
                chunk_session_date,
                json.dumps(list(chunk_session_dates)),
                int(chunk_has_answer),
                int(chunk_has_answer_count),
                float(chunk_answer_density),
                json.dumps(list(chunk_times)),
                json.dumps(list(chunk_time_sources)),
            ),
        )
        chunk_id = int(cursor.lastrowid)
        self._index_chunk_fts(chunk_id, chunk_text)
        self._add_sentences_for_chunk(
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            chunk_role=chunk_role,
            chunk_session_id=chunk_session_id,
            chunk_session_date=chunk_session_date,
        )
        self._enforce_fifo_limit()
        self.conn.commit()

    def _add_sentences_for_chunk(
        self,
        *,
        chunk_id: int,
        chunk_text: str,
        chunk_role: str,
        chunk_session_id: str,
        chunk_session_date: str,
    ) -> None:
        units = chunking.split_sentences(chunk_text)
        for idx, sentence in enumerate(units):
            text = str(sentence).strip()
            if not text:
                continue
            if len(text) < int(self.global_sentence_min_chars):
                continue
            if len(text) > int(self.global_sentence_max_chars):
                text = text[: int(self.global_sentence_max_chars)].strip()
            sent_embedding = embed(text, self.embedding_dim)
            cursor = self.conn.execute(
                """
                INSERT INTO sentences(
                  chunk_id, text, sentence_embedding, sentence_role,
                  sentence_session_id, sentence_session_date, source_part_index
                )
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(chunk_id),
                    text,
                    self._arr_to_blob(sent_embedding),
                    chunk_role,
                    chunk_session_id,
                    chunk_session_date,
                    int(idx),
                ),
            )
            sentence_id = int(cursor.lastrowid)
            self._index_sentence_fts(sentence_id, text)

    def _enforce_fifo_limit(self) -> None:
        overflow = max(0, int(self.debug_stats().get("chunks", 0)) - int(self.max_size))
        if overflow <= 0:
            return
        rows = self.conn.execute(
            "SELECT chunk_id FROM chunks ORDER BY chunk_id ASC LIMIT ?",
            (overflow,),
        ).fetchall()
        chunk_ids = [int(r["chunk_id"]) for r in rows]
        if not chunk_ids:
            return
        marks = ",".join(["?"] * len(chunk_ids))
        sent_rows = self.conn.execute(
            f"SELECT sentence_id FROM sentences WHERE chunk_id IN ({marks})",
            tuple(chunk_ids),
        ).fetchall()
        sentence_ids = [int(r["sentence_id"]) for r in sent_rows]
        placeholders = ",".join(["?"] * len(chunk_ids))
        self.conn.execute(f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", tuple(chunk_ids))
        self.conn.execute(
            f"DELETE FROM sentences WHERE chunk_id IN ({placeholders})",
            tuple(chunk_ids),
        )
        self._delete_chunk_fts(chunk_ids)
        self._delete_sentence_fts(sentence_ids)

    def search_chunks_global(
        self,
        query: str,
    ) -> List[Chunk]:
        """Global chunk retrieval (dense + lexical + keyword)."""
        return retrieval.rerank_chunks_global(
            self,
            query=query,
        )

    def search_chunks_global_with_limit(
        self,
        query: str,
        *,
        top_n: Optional[int] = None,
    ) -> List[Chunk]:
        """Global chunk retrieval with optional temporary top-N override."""
        return retrieval.rerank_chunks_global(
            self,
            query=query,
            top_n_override=top_n,
        )

    def search_sentences_global(self, query: str) -> List[Chunk]:
        """Global sentence retrieval (dense + lexical + keyword)."""
        return retrieval.rerank_sentences_global(
            self,
            query=query,
        )

    def search_sentences_global_with_limit(
        self,
        query: str,
        *,
        top_n: Optional[int] = None,
    ) -> List[Chunk]:
        """Global sentence retrieval with optional temporary top-N override."""
        return retrieval.rerank_sentences_global(
            self,
            query=query,
            top_n_override=top_n,
        )

    def set_temporal_weight_disabled(self, disabled: bool) -> None:
        """Set temporal weighting behavior at runtime (useful for oracle eval)."""
        self.temporal_weight_disabled = bool(disabled)
        logger.info(
            f"MidMemory temporal weighting disabled={self.temporal_weight_disabled}."
        )

    def debug_stats(self) -> Dict[str, int]:
        """Return chunk statistics."""
        return self.store.debug_stats()

    def clear_all(self) -> None:
        """Clear all persisted mid-memory records and in-memory buffers."""
        self.store.clear_all()
        self.buffer.clear()
        self.current_step = 0
        logger.info("MidMemory.clear_all: cleared chunks and runtime buffer.")

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
