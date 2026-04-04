"""Asynchronous long-term memory orchestrator on top of LongMemoryStore."""

from __future__ import annotations

import hashlib
import queue
import sqlite3
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from llm_long_memory.memory.long_memory_ingest import LongMemoryIngestor
from llm_long_memory.memory.long_memory_retrieval import LongMemoryRetriever
from llm_long_memory.memory.long_memory_store import LongMemoryStore
from llm_long_memory.utils.embedding import embed
from llm_long_memory.utils.helpers import load_config
from llm_long_memory.utils.logger import logger


Message = Dict[str, Any]


class LongMemory:
    """Async long memory using event-centric storage with sleep-time maintenance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or load_config()
        self.cfg = dict(self.config["memory"]["long_memory"])
        self.enabled = bool(self.cfg["enabled"])
        self.worker_poll_timeout_sec = float(self.cfg["worker_poll_timeout_sec"])
        self.queue_max_size = int(self.cfg["queue_max_size"])
        self.queue_put_timeout_sec = float(self.cfg["queue_put_timeout_sec"])
        self.queue_overflow_policy = str(self.cfg["queue_overflow_policy"]).strip().lower()

        self.consolidation_every_updates = int(self.cfg["consolidation_every_updates"])
        self.forgetting_every_updates = int(self.cfg["forgetting_every_updates"])
        self.forget_decay = float(self.cfg["forget_decay"])
        self.forget_threshold = float(self.cfg["forget_threshold"])
        self.forget_min_age_steps = int(self.cfg["forget_min_age_steps"])
        self.max_events = int(self.cfg["max_events"])

        self.retrieval_top_k = int(self.cfg["retrieval_top_k"])
        self.retrieval_min_score = float(self.cfg["retrieval_min_score"])
        self.details_per_event = int(self.cfg["details_per_event"])
        self.context_max_items = int(self.cfg["context_max_items"])
        self.context_max_chars_per_item = int(self.cfg["context_max_chars_per_item"])
        ingest_cfg = dict(self.cfg.get("ingest", {}))
        self.ingest_allow_roles = {
            str(x).strip().lower()
            for x in list(ingest_cfg.get("allow_roles", ["user", "assistant", "system"]))
            if str(x).strip()
        }
        self.ingest_min_confidence = float(ingest_cfg.get("min_confidence", 0.7))
        self.ingest_assistant_min_confidence = float(
            ingest_cfg.get("assistant_min_confidence", self.ingest_min_confidence)
        )
        self.ingest_min_keywords = int(ingest_cfg.get("min_keywords", 2))
        self.ingest_min_object_tokens = int(ingest_cfg.get("min_object_tokens", 1))
        self.ingest_min_entity_count = int(ingest_cfg.get("min_entity_count", 1))
        self.ingest_min_sentence_tokens = int(ingest_cfg.get("min_sentence_tokens", 4))
        self.ingest_max_sentence_tokens = int(ingest_cfg.get("max_sentence_tokens", 60))
        self.ingest_require_time_or_location = bool(
            ingest_cfg.get("require_time_or_location", False)
        )
        self.ingest_object_key_tokens = int(ingest_cfg.get("object_key_tokens", 4))
        self.ingest_detail_min_chars = int(ingest_cfg.get("detail_min_chars", 8))
        self.ingest_max_details_per_event = int(ingest_cfg.get("max_details_per_event", 6))
        self.ingest_allow_generic_action = bool(ingest_cfg.get("allow_generic_action", True))
        self.ingest_bootstrap_min_accepts = int(ingest_cfg.get("bootstrap_min_accepts", 8))
        self.ingest_bootstrap_min_confidence = float(
            ingest_cfg.get("bootstrap_min_confidence", 0.35)
        )
        self.ingest_bootstrap_allow_generic_action = bool(
            ingest_cfg.get("bootstrap_allow_generic_action", True)
        )
        self.ingest_generic_actions = {
            str(x).strip().lower()
            for x in list(
                ingest_cfg.get(
                    "generic_actions",
                    ["states", "is", "are", "was", "were", "be", "do", "did"],
                )
            )
            if str(x).strip()
        }
        self.ingest_reject_subjects = {
            str(x).strip().lower()
            for x in list(
                ingest_cfg.get(
                    "reject_subjects",
                    ["assistant", "user", "this", "that", "it", "here", "there"],
                )
            )
            if str(x).strip()
        }
        self.ingest_reject_phrases = [
            str(x).strip().lower()
            for x in list(
                ingest_cfg.get(
                    "reject_phrases",
                    [
                        "how can i help",
                        "let me know",
                        "good luck",
                        "i can help",
                        "feel free",
                    ],
                )
            )
            if str(x).strip()
        ]
        self.ingest_detail_kinds = [
            str(x).strip().lower()
            for x in list(ingest_cfg.get("detail_kinds", ["sentence", "time", "location"]))
            if str(x).strip()
        ]

        retrieval_cfg = dict(self.cfg["retrieval_scoring"])
        self.retrieval_use_embedding = bool(retrieval_cfg["use_embedding"])
        self.retrieval_lexical_weight = float(retrieval_cfg["lexical_weight"])
        self.retrieval_embedding_weight = float(retrieval_cfg["embedding_weight"])
        self.retrieval_recency_weight = float(retrieval_cfg["recency_weight"])
        self.retrieval_fact_type_boost = float(retrieval_cfg["fact_type_boost"])
        self.retrieval_fact_type_mismatch_penalty = float(
            retrieval_cfg["fact_type_mismatch_penalty"]
        )
        self.retrieval_role_weights = {
            str(k).strip().lower(): float(v)
            for k, v in dict(retrieval_cfg.get("role_weights", {})).items()
        }
        rewrite_cfg = dict(self.cfg.get("query_rewrite", {}))
        fusion_cfg = dict(rewrite_cfg.get("fusion", {}))
        self.query_rewrite_enabled = bool(rewrite_cfg.get("enabled", False))
        self.query_rewrite_max = int(rewrite_cfg.get("max_rewrites", 2))
        self.rewrite_fusion_mode = str(fusion_cfg.get("mode", "rrf")).strip().lower()
        self.rewrite_fusion_rrf_k = int(fusion_cfg.get("rrf_k", 60))

        self.embedding_dim = int(self.config["embedding"]["dim"])
        self.event_cfg = dict(self.cfg["event"])

        self._queue: queue.Queue[Message] = queue.Queue(maxsize=self.queue_max_size)
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        self.current_step = 0
        self._ingest_event_total = 0
        self._ingest_event_accepted = 0
        self._ingest_event_rejected = 0
        self._ingest_reject_reasons: Dict[str, int] = {}

        self.store = LongMemoryStore(
            database_file=str(self.cfg["database_file"]),
            sqlite_busy_timeout_ms=int(self.cfg["sqlite_busy_timeout_ms"]),
            sqlite_journal_mode=str(self.cfg["sqlite_journal_mode"]),
            sqlite_synchronous=str(self.cfg["sqlite_synchronous"]),
            embedding_dim=self.embedding_dim,
        )
        self.conn = self.store.conn
        self.db_path = self.store.db_path
        self.current_step = self.store.load_current_step()
        self.ingestor = LongMemoryIngestor(self)
        self.retriever = LongMemoryRetriever(self)

        self._worker = threading.Thread(target=self._worker_loop, name="long-memory-worker", daemon=True)
        if self.enabled:
            self._worker.start()
        logger.info(
            "LongMemory initialized "
            f"(enabled={self.enabled}, database={self.db_path})."
        )

    @staticmethod
    def _stable_id(prefix: str, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"{prefix}:{digest}"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        out: List[str] = []
        for raw in str(text).lower().split():
            cleaned = "".join(ch for ch in raw if ch.isalnum())
            if cleaned:
                out.append(cleaned)
        return out

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        parts = []
        for raw in str(text).lower().split():
            tok = "".join(ch for ch in raw if ch.isalnum())
            if tok:
                parts.append(tok)
        return " ".join(parts).strip()

    def _canonical_subject(self, subject: str, role: str) -> str:
        s = self._normalize_phrase(subject)
        if s in {"i", "me", "my", "mine"}:
            if role == "user":
                return "user_self"
            if role == "assistant":
                return "assistant_self"
        if s in {"you", "your", "yours"}:
            if role == "user":
                return "assistant"
            if role == "assistant":
                return "user"
        return s

    def _canonical_object_key(self, obj: str) -> str:
        o = self._normalize_phrase(obj)
        if not o:
            return ""
        tokens = o.split()
        return " ".join(tokens[: self.ingest_object_key_tokens]).strip()

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _safe_embed(self, text: str) -> np.ndarray:
        if not self.retrieval_use_embedding:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        try:
            return embed(text, self.embedding_dim)
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.warn(f"LongMemory embedding fallback to zeros: {exc}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def enqueue_message(self, message: Message) -> bool:
        if not self.enabled:
            return False
        try:
            self._queue.put(dict(message), timeout=self.queue_put_timeout_sec)
            return True
        except queue.Full:
            if self.queue_overflow_policy == "sync_process":
                logger.warn("LongMemory queue full; processing message synchronously.")
                try:
                    self._process_message(dict(message))
                    return True
                except (RuntimeError, ValueError, TypeError, OSError, sqlite3.DatabaseError) as exc:
                    logger.error(f"LongMemory sync overflow processing failed: {exc}")
                    return False
            logger.warn("LongMemory queue full; dropped one message update.")
            return False

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                message = self._queue.get(timeout=self.worker_poll_timeout_sec)
            except queue.Empty:
                continue
            try:
                self._process_message(message)
            except (RuntimeError, ValueError, TypeError, OSError, sqlite3.DatabaseError) as exc:
                logger.error(f"LongMemory worker failed to process message: {exc}")
            finally:
                self._queue.task_done()

    def _process_message(self, message: Message) -> None:
        with self._lock:
            self.ingestor.process_message(message)

    def query(self, query_text: str) -> List[Dict[str, Any]]:
        return self.retriever.query(query_text)

    def query_multi(self, query_texts: List[str]) -> List[Dict[str, Any]]:
        return self.retriever.query_multi(query_texts)

    def build_context_snippets(self, query_text: str) -> List[str]:
        return self.retriever.build_context_snippets(query_text)

    def build_context_snippets_multi(self, query_texts: List[str]) -> List[str]:
        return self.retriever.build_context_snippets_multi(query_texts)

    def debug_stats(self) -> Dict[str, int]:
        with self._lock:
            counts = self.store.debug_counts()
        return {
            "nodes": int(counts["events"]),
            "edges": int(counts["details"]),
            "events": int(counts["events"]),
            "details": int(counts["details"]),
            "active_events": int(counts["active_events"]),
            "superseded_events": int(counts["superseded_events"]),
            "queued_updates": int(self._queue.qsize()) if self.enabled else 0,
            "applied_updates": int(self.current_step),
            "ingest_event_total": int(self._ingest_event_total),
            "ingest_event_accepted": int(self._ingest_event_accepted),
            "ingest_event_rejected": int(self._ingest_event_rejected),
            "candidate_events": 0,
            "reject_reason_low_confidence": int(self._ingest_reject_reasons.get("low_confidence", 0)),
            "reject_reason_few_keywords": int(self._ingest_reject_reasons.get("few_keywords", 0)),
            "reject_reason_short_object": int(self._ingest_reject_reasons.get("short_object", 0)),
            "reject_reason_few_entities": int(self._ingest_reject_reasons.get("few_entities", 0)),
            "reject_reason_short_sentence": int(self._ingest_reject_reasons.get("short_sentence", 0)),
            "reject_reason_long_sentence": int(self._ingest_reject_reasons.get("long_sentence", 0)),
            "reject_reason_missing_time_or_location": int(
                self._ingest_reject_reasons.get("missing_time_or_location", 0)
            ),
            "reject_reason_rejected_phrase": int(self._ingest_reject_reasons.get("rejected_phrase", 0)),
            "reject_reason_generic_subject_action": int(
                self._ingest_reject_reasons.get("generic_subject_action", 0)
            ),
            "reject_reason_generic_action_disabled": int(
                self._ingest_reject_reasons.get("generic_action_disabled", 0)
            ),
            "reject_reason_empty_key_component": int(
                self._ingest_reject_reasons.get("empty_key_component", 0)
            ),
        }

    def clear_all(self) -> None:
        with self._lock:
            self.store.clear_all()
            self.current_step = 0
            self.store.save_current_step(self.current_step)
            self.store.commit()
            self._ingest_event_total = 0
            self._ingest_event_accepted = 0
            self._ingest_event_rejected = 0
            self._ingest_reject_reasons = {}
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        logger.info("LongMemory.clear_all: cleared SQLite event/detail tables and queue.")

    def close(self) -> None:
        if self.enabled:
            self._stop_event.set()
            self._worker.join(timeout=max(1.0, self.worker_poll_timeout_sec * 5.0))
        with self._lock:
            self.store.save_current_step(self.current_step)
            self.store.commit()
            self.store.close()
