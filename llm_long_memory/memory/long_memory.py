"""Asynchronous long-term memory orchestrator on top of LongMemoryStore."""

from __future__ import annotations

import hashlib
import json
import queue
import sqlite3
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from llm_long_memory.memory.long_memory_store import LongMemoryStore
from llm_long_memory.processing.graph_builder import extract_events_from_message
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
        self.ingest_object_key_tokens = int(ingest_cfg.get("object_key_tokens", 4))
        self.ingest_detail_min_chars = int(ingest_cfg.get("detail_min_chars", 8))
        self.ingest_max_details_per_event = int(ingest_cfg.get("max_details_per_event", 6))
        self.ingest_allow_generic_action = bool(ingest_cfg.get("allow_generic_action", True))
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

    def _should_accept_event(
        self,
        role: str,
        subject: str,
        action: str,
        obj: str,
        sentence: str,
        confidence: float,
        keywords: List[str],
    ) -> bool:
        if role not in self.ingest_allow_roles:
            return False
        min_conf = self.ingest_assistant_min_confidence if role == "assistant" else self.ingest_min_confidence
        if confidence < min_conf:
            return False
        if len(keywords) < self.ingest_min_keywords:
            return False
        if len(self._tokenize(obj)) < self.ingest_min_object_tokens:
            return False

        lowered_sentence = str(sentence).strip().lower()
        for phrase in self.ingest_reject_phrases:
            if phrase and phrase in lowered_sentence:
                return False

        norm_subject = self._canonical_subject(subject, role)
        norm_action = self._normalize_phrase(action)
        if norm_subject in self.ingest_reject_subjects and norm_action in self.ingest_generic_actions:
            return False
        if (not self.ingest_allow_generic_action) and (norm_action in self.ingest_generic_actions):
            return False
        return True

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
            self._queue.put_nowait(dict(message))
            return True
        except queue.Full:
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
        events = extract_events_from_message(message, self.event_cfg)
        if not events:
            return

        with self._lock:
            for ev in events:
                self._ingest_event_total += 1
                self.current_step += 1

                subject = str(ev.get("subject", "")).strip()
                action = str(ev.get("action", "states")).strip() or "states"
                obj = str(ev.get("object", "")).strip()
                location = str(ev.get("location", "")).strip()
                time_value = str(ev.get("time", "")).strip()
                sentence = str(ev.get("sentence", "")).strip()
                role = str(ev.get("role", "user")).strip().lower() or "user"
                confidence = float(ev.get("confidence", 0.0) or 0.0)

                normalized_keywords = self._tokenize(" ".join([subject, action, obj, location, time_value]))
                if not self._should_accept_event(
                    role=role,
                    subject=subject,
                    action=action,
                    obj=obj,
                    sentence=sentence,
                    confidence=confidence,
                    keywords=normalized_keywords,
                ):
                    self._ingest_event_rejected += 1
                    continue

                subject_key = self._canonical_subject(subject, role)
                action_key = self._normalize_phrase(action)
                object_key = self._canonical_object_key(obj)
                if not subject_key or not action_key or not object_key:
                    self._ingest_event_rejected += 1
                    continue

                skeleton_text = f"{subject_key} {action_key} {object_key}".strip()
                self._ingest_event_accepted += 1

                fact_key = f"{subject_key}|{action_key}|{object_key}"
                event_id = self._stable_id("event", fact_key)
                keywords = normalized_keywords
                skeleton_embedding = self._safe_embed(skeleton_text)

                self.store.upsert_event(
                    event_id=event_id,
                    fact_key=fact_key,
                    skeleton_text=skeleton_text,
                    skeleton_embedding=skeleton_embedding,
                    keywords=keywords,
                    role=role,
                    current_step=self.current_step,
                )

                for kind, value in (
                    ("sentence", sentence),
                    ("location", location),
                    ("time", time_value),
                    ("object", obj),
                ):
                    if kind not in self.ingest_detail_kinds:
                        continue
                    clean = str(value).strip()
                    if not clean or len(clean) < self.ingest_detail_min_chars:
                        continue
                    detail_id = self._stable_id("detail", f"{event_id}|{kind}|{clean}")
                    self.store.insert_detail(
                        detail_id=detail_id,
                        event_id=event_id,
                        kind=kind,
                        text=clean,
                        current_step=self.current_step,
                        max_per_event=self.ingest_max_details_per_event,
                    )

            if self.consolidation_every_updates > 0 and (
                self.current_step % self.consolidation_every_updates == 0
            ):
                self.store.resolve_conflicts()

            if self.forgetting_every_updates > 0 and (
                self.current_step % self.forgetting_every_updates == 0
            ):
                self.store.apply_forgetting(
                    current_step=self.current_step,
                    forget_decay=self.forget_decay,
                    forget_threshold=self.forget_threshold,
                    forget_min_age_steps=self.forget_min_age_steps,
                    max_events=self.max_events,
                )

            self.store.save_current_step(self.current_step)
            self.store.commit()

    def query(self, query_text: str) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        q_tokens = set(self._tokenize(query_text))
        if not q_tokens:
            return []
        q_vec = self._safe_embed(query_text)

        with self._lock:
            rows = self.store.fetch_active_events()
            now_step = max(1, int(self.current_step))

        scored: List[Dict[str, Any]] = []
        for row in rows:
            skeleton_text = str(row["skeleton_text"])
            try:
                keywords = list(json.loads(str(row["keywords"])))
            except (ValueError, TypeError):
                keywords = self._tokenize(skeleton_text)
            key_set = {str(k).strip().lower() for k in keywords if str(k).strip()}
            overlap = len(q_tokens.intersection(key_set))
            lexical = float(overlap) / float(max(1, len(q_tokens)))

            emb_score = 0.0
            if self.retrieval_use_embedding:
                emb = self.store.blob_to_arr(row["skeleton_embedding"])
                emb_score = max(0.0, self._cosine(q_vec, emb))

            delta = max(0, now_step - int(row["last_seen_step"] or 0))
            recency = 1.0 / (1.0 + float(delta))

            score = (
                self.retrieval_lexical_weight * lexical
                + self.retrieval_embedding_weight * emb_score
                + self.retrieval_recency_weight * recency
            )
            role = str(row["role"]).strip().lower()
            role_weight = float(self.retrieval_role_weights.get(role, 1.0))
            score *= role_weight
            if score < self.retrieval_min_score:
                continue

            scored.append(
                {
                    "event_id": str(row["event_id"]),
                    "type": "event",
                    "text": skeleton_text,
                    "score": float(score),
                    "lexical_score": float(lexical),
                    "embedding_score": float(emb_score),
                    "recency_score": float(recency),
                    "role": str(row["role"]),
                }
            )

        scored.sort(key=lambda x: (float(x["score"]), len(str(x["text"]))), reverse=True)
        return scored[: self.retrieval_top_k]

    def query_multi(self, query_texts: List[str]) -> List[Dict[str, Any]]:
        """Fuse results from multiple rewritten queries."""
        normalized = [str(x).strip() for x in query_texts if str(x).strip()]
        if not normalized:
            return []
        if len(normalized) == 1:
            return self.query(normalized[0])

        rank_acc: Dict[str, float] = {}
        meta: Dict[str, Dict[str, Any]] = {}
        for q in normalized:
            hits = self.query(q)
            for idx, item in enumerate(hits):
                event_id = str(item.get("event_id", "")).strip()
                if not event_id:
                    continue
                rank = idx + 1
                if self.rewrite_fusion_mode == "rrf":
                    add = 1.0 / float(self.rewrite_fusion_rrf_k + rank)
                else:
                    add = float(item.get("score", 0.0))
                rank_acc[event_id] = rank_acc.get(event_id, 0.0) + add
                old = meta.get(event_id)
                if old is None or float(item.get("score", 0.0)) > float(old.get("score", 0.0)):
                    meta[event_id] = dict(item)

        merged: List[Dict[str, Any]] = []
        for event_id, fused_score in rank_acc.items():
            row = dict(meta.get(event_id, {}))
            row["fused_score"] = float(fused_score)
            row["score"] = max(float(row.get("score", 0.0)), float(fused_score))
            merged.append(row)
        merged.sort(
            key=lambda x: (
                float(x.get("fused_score", 0.0)),
                float(x.get("score", 0.0)),
            ),
            reverse=True,
        )
        return merged[: self.retrieval_top_k]

    def build_context_snippets(self, query_text: str) -> List[str]:
        hits = self.query(query_text)
        return self._build_context_from_hits(hits)

    def build_context_snippets_multi(self, query_texts: List[str]) -> List[str]:
        """Build context snippets from multi-query fused retrieval."""
        hits = self.query_multi(query_texts)
        return self._build_context_from_hits(hits)

    def _build_context_from_hits(self, hits: List[Dict[str, Any]]) -> List[str]:
        snippets: List[str] = []
        with self._lock:
            for hit in hits[: self.context_max_items]:
                event_id = str(hit["event_id"])
                detail_rows = self.store.fetch_event_details(event_id, self.details_per_event)
                details = [
                    f"{str(r['kind'])}: {str(r['text'])}"
                    for r in detail_rows
                    if str(r["text"]).strip()
                ]
                base = f"[event] {str(hit['text'])}"
                if details:
                    base += " | " + " ; ".join(details)
                snippets.append(base[: self.context_max_chars_per_item])
        return snippets

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
