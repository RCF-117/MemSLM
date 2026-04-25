"""Predictive light-graph cache for amortizing claims/light-graph latency.

This module implements a strict, default-off cache layer:

- Offline: generate anticipated queries for bounded mid-memory windows, then run
  the normal retrieval -> filter -> claims -> light-graph pipeline and persist
  the resulting graph bundle.
- Online: embed the live query, recall top-k predicted queries, and apply a
  conservative compatibility gate before reusing the cached graph bundle.

The cache is intentionally an acceleration layer only. Misses always fall back
to the existing online pipeline. Hits reuse the cached graph bundle but still
flow through toolkit and final answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from llm_long_memory.llm.ollama_client import ollama_generate_with_retry
from llm_long_memory.utils.embedding import embed
from llm_long_memory.utils.helpers import resolve_project_path
from llm_long_memory.utils.logger import logger


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _tokenize(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(value))


def _token_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = {str(x).strip().lower() for x in left if str(x).strip()}
    right_set = {str(x).strip().lower() for x in right if str(x).strip()}
    if not left_set or not right_set:
        return 0.0
    return float(len(left_set & right_set)) / float(max(1, min(len(left_set), len(right_set))))


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _json_loads(raw: str, default: Any) -> Any:
    text = str(raw or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def _extract_first_json_container(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return "{}"
    if (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    ):
        return text
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj >= 0 and end_obj > start_obj:
        return text[start_obj : end_obj + 1]
    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr >= 0 and end_arr > start_arr:
        return text[start_arr : end_arr + 1]
    return "{}"


def _embedding_to_json(arr: np.ndarray) -> str:
    return _json_dumps([float(x) for x in arr.astype(np.float32).reshape(-1).tolist()])


def _embedding_from_json(raw: str, dim: int) -> np.ndarray:
    data = _json_loads(raw, [])
    if not isinstance(data, list):
        return np.zeros(dim, dtype=np.float32)
    arr = np.asarray([float(x) for x in data], dtype=np.float32)
    if arr.size != int(dim):
        fixed = np.zeros(int(dim), dtype=np.float32)
        limit = min(int(dim), int(arr.size))
        if limit > 0:
            fixed[:limit] = arr[:limit]
        arr = fixed
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = (arr / norm).astype(np.float32)
    return arr


def _dot_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return 0.0
    return float(np.dot(left.astype(np.float32), right.astype(np.float32)))


def _answer_type_ttl_hours(answer_type: str, cfg: Dict[str, Any]) -> int:
    atype = str(answer_type or "").strip().lower()
    if atype in {"update", "temporal", "temporal_comparison", "temporal_count"}:
        return int(cfg.get("short_ttl_hours", 72))
    return int(cfg.get("default_ttl_hours", 168))


@dataclass
class PredictiveGraphCacheEntry:
    cache_id: str
    window_id: str
    anticipated_query: str
    anticipated_query_embedding: List[float]
    answer_type: str
    query_plan_summary: Dict[str, Any]
    retrieval_snapshot: Dict[str, Any]
    filtered_pack: Dict[str, Any]
    claim_result: Dict[str, Any]
    light_graph: Dict[str, Any]
    toolkit_payload: Dict[str, Any]
    quality_signals: Dict[str, Any]
    cache_stats: Dict[str, Any]
    expiry_metadata: Dict[str, Any]


@dataclass
class PredictiveGraphCacheHit:
    cache_id: str
    bundle: Dict[str, Any]
    chunks: List[Dict[str, Any]]
    context_text: str
    toolkit_payload: Dict[str, Any]
    latency_saved_sec: float
    answer_type: str


class PredictiveGraphCacheStore:
    """SQLite-backed store for anticipated-query graph cache entries."""

    def __init__(self, database_file: str, embedding_dim: int) -> None:
        self.db_path = resolve_project_path(database_file)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = int(embedding_dim)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictive_graph_cache_entries(
              cache_id TEXT PRIMARY KEY,
              window_id TEXT NOT NULL,
              anticipated_query TEXT NOT NULL,
              anticipated_query_embedding TEXT NOT NULL,
              answer_type TEXT NOT NULL,
              query_plan_summary TEXT NOT NULL,
              retrieval_snapshot TEXT NOT NULL,
              filtered_pack TEXT NOT NULL,
              claim_result TEXT NOT NULL,
              light_graph TEXT NOT NULL,
              toolkit_payload TEXT NOT NULL,
              quality_signals TEXT NOT NULL,
              cache_stats TEXT NOT NULL,
              expiry_metadata TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictive_cache_window_id "
            "ON predictive_graph_cache_entries(window_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictive_cache_answer_type "
            "ON predictive_graph_cache_entries(answer_type)"
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def _row_to_entry(self, row: sqlite3.Row) -> PredictiveGraphCacheEntry:
        return PredictiveGraphCacheEntry(
            cache_id=str(row["cache_id"]),
            window_id=str(row["window_id"]),
            anticipated_query=str(row["anticipated_query"]),
            anticipated_query_embedding=_json_loads(
                str(row["anticipated_query_embedding"]), []
            ),
            answer_type=str(row["answer_type"]),
            query_plan_summary=_json_loads(str(row["query_plan_summary"]), {}),
            retrieval_snapshot=_json_loads(str(row["retrieval_snapshot"]), {}),
            filtered_pack=_json_loads(str(row["filtered_pack"]), {}),
            claim_result=_json_loads(str(row["claim_result"]), {}),
            light_graph=_json_loads(str(row["light_graph"]), {}),
            toolkit_payload=_json_loads(str(row["toolkit_payload"]), {}),
            quality_signals=_json_loads(str(row["quality_signals"]), {}),
            cache_stats=_json_loads(str(row["cache_stats"]), {}),
            expiry_metadata=_json_loads(str(row["expiry_metadata"]), {}),
        )

    def upsert(self, entry: PredictiveGraphCacheEntry) -> None:
        now = _utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO predictive_graph_cache_entries(
              cache_id, window_id, anticipated_query, anticipated_query_embedding,
              answer_type, query_plan_summary, retrieval_snapshot, filtered_pack,
              claim_result, light_graph, toolkit_payload, quality_signals,
              cache_stats, expiry_metadata, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_id) DO UPDATE SET
              window_id=excluded.window_id,
              anticipated_query=excluded.anticipated_query,
              anticipated_query_embedding=excluded.anticipated_query_embedding,
              answer_type=excluded.answer_type,
              query_plan_summary=excluded.query_plan_summary,
              retrieval_snapshot=excluded.retrieval_snapshot,
              filtered_pack=excluded.filtered_pack,
              claim_result=excluded.claim_result,
              light_graph=excluded.light_graph,
              toolkit_payload=excluded.toolkit_payload,
              quality_signals=excluded.quality_signals,
              cache_stats=excluded.cache_stats,
              expiry_metadata=excluded.expiry_metadata,
              updated_at=excluded.updated_at
            """,
            (
                entry.cache_id,
                entry.window_id,
                entry.anticipated_query,
                _json_dumps(entry.anticipated_query_embedding),
                entry.answer_type,
                _json_dumps(entry.query_plan_summary),
                _json_dumps(entry.retrieval_snapshot),
                _json_dumps(entry.filtered_pack),
                _json_dumps(entry.claim_result),
                _json_dumps(entry.light_graph),
                _json_dumps(entry.toolkit_payload),
                _json_dumps(entry.quality_signals),
                _json_dumps(entry.cache_stats),
                _json_dumps(entry.expiry_metadata),
                now,
                now,
            ),
        )
        self.conn.commit()

    def fetch_all(self) -> List[PredictiveGraphCacheEntry]:
        rows = self.conn.execute(
            "SELECT * FROM predictive_graph_cache_entries ORDER BY updated_at DESC"
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def fetch_topk(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int,
        answer_type: str = "",
    ) -> List[Tuple[PredictiveGraphCacheEntry, float]]:
        entries = self.fetch_all()
        target_type = str(answer_type or "").strip().lower()
        scored: List[Tuple[PredictiveGraphCacheEntry, float]] = []
        for entry in entries:
            if target_type and str(entry.answer_type).strip().lower() != target_type:
                continue
            emb = _embedding_from_json(
                _json_dumps(entry.anticipated_query_embedding), self.embedding_dim
            )
            scored.append((entry, _dot_similarity(query_embedding, emb)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, int(top_k))]

    def mark_hit(self, cache_id: str) -> None:
        rows = self.conn.execute(
            "SELECT cache_stats FROM predictive_graph_cache_entries WHERE cache_id = ?",
            (str(cache_id),),
        ).fetchone()
        if rows is None:
            return
        stats = _json_loads(str(rows["cache_stats"]), {})
        stats["hit_count"] = int(stats.get("hit_count", 0) or 0) + 1
        stats["last_hit_time"] = _utcnow().isoformat()
        self.conn.execute(
            """
            UPDATE predictive_graph_cache_entries
            SET cache_stats = ?, updated_at = ?
            WHERE cache_id = ?
            """,
            (_json_dumps(stats), _utcnow().isoformat(), str(cache_id)),
        )
        self.conn.commit()

    def invalidate(self, cache_id: str, reason: str) -> None:
        row = self.conn.execute(
            "SELECT expiry_metadata FROM predictive_graph_cache_entries WHERE cache_id = ?",
            (str(cache_id),),
        ).fetchone()
        if row is None:
            return
        metadata = _json_loads(str(row["expiry_metadata"]), {})
        metadata["invalidated"] = True
        metadata["invalidation_reason"] = str(reason or "").strip() or "manual"
        self.conn.execute(
            """
            UPDATE predictive_graph_cache_entries
            SET expiry_metadata = ?, updated_at = ?
            WHERE cache_id = ?
            """,
            (_json_dumps(metadata), _utcnow().isoformat(), str(cache_id)),
        )
        self.conn.commit()

    def forget_low_value_entries(
        self,
        *,
        max_entries_per_window: int,
        stale_before_iso: str,
    ) -> int:
        rows = self.conn.execute(
            "SELECT cache_id, window_id, quality_signals, cache_stats FROM predictive_graph_cache_entries"
        ).fetchall()
        by_window: Dict[str, List[Tuple[str, float, int, str]]] = {}
        for row in rows:
            quality = _json_loads(str(row["quality_signals"]), {})
            stats = _json_loads(str(row["cache_stats"]), {})
            quality_score = float(quality.get("top_evidence_score", 0.0) or 0.0)
            hits = int(stats.get("hit_count", 0) or 0)
            last_hit = str(stats.get("last_hit_time", "") or "")
            by_window.setdefault(str(row["window_id"]), []).append(
                (str(row["cache_id"]), quality_score, hits, last_hit)
            )
        deleted = 0
        for _window_id, items in by_window.items():
            if len(items) <= max(1, int(max_entries_per_window)):
                continue
            ranked = sorted(
                items,
                key=lambda item: (item[2], item[1], item[3] >= stale_before_iso),
                reverse=True,
            )
            keep_ids = {item[0] for item in ranked[: max(1, int(max_entries_per_window))]}
            drop_ids = [item[0] for item in items if item[0] not in keep_ids]
            for cache_id in drop_ids:
                self.conn.execute(
                    "DELETE FROM predictive_graph_cache_entries WHERE cache_id = ?",
                    (cache_id,),
                )
                deleted += 1
        if deleted:
            self.conn.commit()
        return deleted


class PredictiveGraphCacheIndex:
    """Strict online lookup + compatibility gate for predictive graph cache."""

    def __init__(self, manager: Any, store: PredictiveGraphCacheStore, cfg: Dict[str, Any]) -> None:
        self.m = manager
        self.store = store
        self.cfg = dict(cfg)
        self.top_k = int(self.cfg.get("top_k", 5))
        self.entity_overlap_min = float(self.cfg.get("entity_overlap_min", 0.34))
        self.focus_overlap_min = float(self.cfg.get("focus_overlap_min", 0.34))
        self.semantic_similarity_min = float(self.cfg.get("semantic_similarity_min", 0.52))
        self.quality_min_top_evidence_score = float(
            self.cfg.get("quality_min_top_evidence_score", 0.30)
        )
        self.quality_min_selected_evidence = int(
            self.cfg.get("quality_min_selected_evidence", 2)
        )
        self.quality_min_selected_claims = int(
            self.cfg.get("quality_min_selected_claims", 1)
        )
        self.quality_min_structural_edges = int(
            self.cfg.get("quality_min_structural_edges", 1)
        )

    def lookup(self, query: str) -> Optional[PredictiveGraphCacheHit]:
        live_plan = self.m._build_query_plan(query)
        answer_type = str(live_plan.get("answer_type", "")).strip().lower()
        query_embedding = embed(str(query or ""), int(self.m.mid_memory.embedding_dim))
        candidates = self.store.fetch_topk(
            query_embedding,
            top_k=max(1, int(self.top_k)),
            answer_type=answer_type,
        )
        metrics = getattr(self.m, "predictive_graph_cache_metrics", {})
        metrics["lookups"] = int(metrics.get("lookups", 0) or 0) + 1
        for entry, similarity in candidates:
            reason = self._compatibility_reject_reason(
                entry=entry,
                query=query,
                live_plan=live_plan,
                similarity=similarity,
            )
            if reason:
                metrics["false_hit_rejections"] = int(
                    metrics.get("false_hit_rejections", 0) or 0
                ) + 1
                continue
            hit = self._build_hit(entry=entry, query=query, live_plan=live_plan)
            if hit is None:
                metrics["false_hit_rejections"] = int(
                    metrics.get("false_hit_rejections", 0) or 0
                ) + 1
                continue
            metrics["hits"] = int(metrics.get("hits", 0) or 0) + 1
            hit_by_type = dict(metrics.get("hit_by_answer_type", {}) or {})
            hit_by_type[answer_type] = int(hit_by_type.get(answer_type, 0) or 0) + 1
            metrics["hit_by_answer_type"] = hit_by_type
            self.m.predictive_graph_cache_metrics = metrics
            self.store.mark_hit(entry.cache_id)
            return hit
        metrics["misses"] = int(metrics.get("misses", 0) or 0) + 1
        self.m.predictive_graph_cache_metrics = metrics
        return None

    def _compatibility_reject_reason(
        self,
        *,
        entry: PredictiveGraphCacheEntry,
        query: str,
        live_plan: Dict[str, Any],
        similarity: float,
    ) -> str:
        if float(similarity) < float(self.semantic_similarity_min):
            return "low_similarity"
        if self._entry_expired_or_invalid(entry):
            return "expired_or_invalid"
        live_answer_type = str(live_plan.get("answer_type", "")).strip().lower()
        if live_answer_type != str(entry.answer_type or "").strip().lower():
            return "answer_type_mismatch"
        summary = dict(entry.query_plan_summary or {})
        entity_overlap = _token_overlap(
            list(live_plan.get("entities", [])),
            list(summary.get("entities", [])),
        )
        focus_overlap = _token_overlap(
            list(live_plan.get("focus_phrases", [])),
            list(summary.get("focus_phrases", [])),
        )
        if entity_overlap < self.entity_overlap_min and focus_overlap < self.focus_overlap_min:
            return "focus_entity_mismatch"
        if not self._temporal_compatible(live_plan=live_plan, summary=summary, query=query):
            return "temporal_mismatch"
        if not self._quality_ok(entry):
            return "quality_below_threshold"
        return ""

    def _entry_expired_or_invalid(self, entry: PredictiveGraphCacheEntry) -> bool:
        expiry = dict(entry.expiry_metadata or {})
        if bool(expiry.get("invalidated", False)):
            return True
        ttl_deadline = str(expiry.get("ttl_deadline", "") or "")
        if ttl_deadline:
            try:
                if datetime.fromisoformat(ttl_deadline) <= _utcnow():
                    self.store.invalidate(entry.cache_id, "ttl_expired")
                    self.m.predictive_graph_cache_metrics["stale_cache_invalidations"] = int(
                        self.m.predictive_graph_cache_metrics.get("stale_cache_invalidations", 0)
                        or 0
                    ) + 1
                    return True
            except ValueError:
                self.store.invalidate(entry.cache_id, "invalid_ttl_deadline")
                return True
        chunk_ids = [
            int(x)
            for x in list(dict(entry.retrieval_snapshot or {}).get("source_chunk_ids", []))
            if int(x)
        ]
        if chunk_ids:
            existing = set(self.m.mid_memory.get_existing_chunk_ids(chunk_ids))
            if any(cid not in existing for cid in chunk_ids):
                self.store.invalidate(entry.cache_id, "source_chunk_missing")
                self.m.predictive_graph_cache_metrics["stale_cache_invalidations"] = int(
                    self.m.predictive_graph_cache_metrics.get("stale_cache_invalidations", 0)
                    or 0
                ) + 1
                return True
        return False

    def _temporal_compatible(
        self,
        *,
        live_plan: Dict[str, Any],
        summary: Dict[str, Any],
        query: str,
    ) -> bool:
        live_terms = {
            str(x).strip().lower()
            for x in list(live_plan.get("time_terms", []))
            if str(x).strip()
        }
        cached_terms = {
            str(x).strip().lower()
            for x in list(summary.get("temporal_anchors", []))
            if str(x).strip()
        }
        query_low = _normalize_text(query)
        if any(term in query_low for term in {"current", "latest", "now"}):
            if not bool(summary.get("need_latest_state", False)):
                return False
        if any(term in query_low for term in {"before", "after", "when"}):
            if not cached_terms and not live_terms:
                return False
        if live_terms and cached_terms and not (live_terms & cached_terms):
            return False
        return True

    def _quality_ok(self, entry: PredictiveGraphCacheEntry) -> bool:
        quality = dict(entry.quality_signals or {})
        top_score = float(quality.get("top_evidence_score", 0.0) or 0.0)
        selected_evidence = int(quality.get("selected_evidence_count", 0) or 0)
        selected_claims = int(quality.get("selected_claim_count", 0) or 0)
        edge_count = int(quality.get("graph_structural_edge_count", 0) or 0)
        toolkit_activated = bool(quality.get("toolkit_activated", False))
        if top_score < self.quality_min_top_evidence_score:
            return False
        if selected_evidence < self.quality_min_selected_evidence:
            return False
        if (
            selected_claims < self.quality_min_selected_claims
            and edge_count < self.quality_min_structural_edges
            and not toolkit_activated
        ):
            return False
        return True

    def _build_hit(
        self,
        *,
        entry: PredictiveGraphCacheEntry,
        query: str,
        live_plan: Dict[str, Any],
    ) -> Optional[PredictiveGraphCacheHit]:
        snapshot = dict(entry.retrieval_snapshot or {})
        chunk_ids = [
            int(x)
            for x in list(snapshot.get("source_chunk_ids", []))
            if int(x)
        ]
        if not chunk_ids:
            return None
        chunks = self.m.mid_memory.get_chunks_by_ids(chunk_ids)
        if not chunks:
            return None
        context_text = self.m._chunks_to_context_text(chunks)
        bundle = {
            "query": str(query or ""),
            "query_plan": dict(live_plan),
            "stage_flags": {
                "filter_enabled": True,
                "claims_enabled": True,
                "light_graph_enabled": True,
                "predictive_cache_hit": True,
            },
            "stage_latency_sec": {
                "filter": 0.0,
                "claims": 0.0,
                "light_graph": 0.0,
                "graph_total": 0.0,
                "predictive_cache_lookup": 0.0,
                "predictive_cache_saved": float(
                    dict(entry.quality_signals or {}).get("graph_total_latency_sec", 0.0) or 0.0
                ),
            },
            "retrieved_context_text": context_text,
            "retrieval_snapshot": snapshot,
            "unified_source": [],
            "filtered_pack": dict(entry.filtered_pack or {}),
            "claim_result": dict(entry.claim_result or {}),
            "light_graph": dict(entry.light_graph or {}),
            "cache_metadata": {
                "cache_id": entry.cache_id,
                "window_id": entry.window_id,
                "anticipated_query": entry.anticipated_query,
                "answer_type": entry.answer_type,
                "cache_hit": True,
            },
        }
        return PredictiveGraphCacheHit(
            cache_id=entry.cache_id,
            bundle=bundle,
            chunks=chunks,
            context_text=context_text,
            toolkit_payload=dict(entry.toolkit_payload or {}),
            latency_saved_sec=float(
                dict(entry.quality_signals or {}).get("graph_total_latency_sec", 0.0) or 0.0
            ),
            answer_type=str(entry.answer_type or ""),
        )


class PredictiveQueryGenerator:
    """Generate anticipated queries with type-conditioned structured plans."""

    ALLOWED_ANSWER_TYPES = {
        "factoid",
        "update",
        "count",
        "temporal",
        "temporal_comparison",
    }
    ALLOWED_TEMPORAL_MODES = {
        "current",
        "past",
        "first",
        "latest",
        "duration",
        "before_after",
        "compare",
        "unspecified",
    }
    ALLOWED_ANSWER_SHAPES = {
        "entity",
        "value",
        "number",
        "date",
        "duration",
        "ordered_choice",
    }

    def __init__(self, manager: Any, cfg: Dict[str, Any]) -> None:
        self.m = manager
        self.cfg = dict(cfg)
        self.enabled = bool(self.cfg.get("enabled", True))
        self.max_queries_per_window = int(self.cfg.get("max_queries_per_window", 3))
        self.max_candidates_per_type = int(self.cfg.get("max_candidates_per_type", 2))
        self.max_window_chars = int(self.cfg.get("max_window_chars", 2400))
        self.model = str(self.cfg.get("model", getattr(self.m.llm, "model_name", "")))
        self.temperature = float(self.cfg.get("temperature", 0.0))
        self.timeout_sec = int(self.cfg.get("timeout_sec", 120))
        self.max_output_tokens = int(self.cfg.get("max_output_tokens", 320))
        self.force_json_output = bool(self.cfg.get("force_json_output", True))
        self.think = bool(self.cfg.get("think", False))
        self.target_answer_types = [
            str(x).strip().lower()
            for x in list(
                self.cfg.get(
                    "target_answer_types",
                    ["update", "count", "temporal", "temporal_comparison", "factoid"],
                )
            )
            if str(x).strip().lower() in self.ALLOWED_ANSWER_TYPES
        ]
        if not self.target_answer_types:
            self.target_answer_types = ["update", "count", "temporal", "factoid"]

    def generate_for_window(
        self,
        *,
        window_id: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not chunks:
            return []
        if (not hasattr(self.m.llm, "host")) or (not hasattr(self.m.llm, "model_name")):
            return []
        lines: List[str] = []
        chars = 0
        for idx, chunk in enumerate(chunks, start=1):
            text = re.sub(r"\s+", " ", str(chunk.get("text", "")).strip())
            if not text:
                continue
            line = f"{idx}. {text}"
            if chars + len(line) > self.max_window_chars and lines:
                break
            lines.append(line)
            chars += len(line)
        if not lines:
            return []
        window_text = "\n".join(lines)
        all_plans: List[Dict[str, Any]] = []
        for answer_type in self.target_answer_types:
            all_plans.extend(
                self._generate_plans_for_type(
                    window_id=window_id,
                    window_text=window_text,
                    answer_type=answer_type,
                )
            )
        return self._plans_to_queries(all_plans)

    def _generate_plans_for_type(
        self,
        *,
        window_id: str,
        window_text: str,
        answer_type: str,
    ) -> List[Dict[str, Any]]:
        try:
            host = str(getattr(self.m.llm, "host", self.m.config["llm"]["host"])).rstrip("/")
            opener = getattr(self.m.llm, "_opener", None)
            if opener is None:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            prompt = self._build_type_prompt(
                window_id=window_id,
                window_text=window_text,
                answer_type=answer_type,
            )
            raw = ollama_generate_with_retry(
                host=host,
                model=self.model or str(getattr(self.m.llm, "model_name", "")),
                prompt=prompt,
                temperature=self.temperature,
                timeout_sec=self.timeout_sec,
                opener=opener,
                max_attempts=1,
                backoff_sec=0.0,
                retry_on_timeout=False,
                retry_on_http_502=False,
                retry_on_url_error=False,
                max_output_tokens=self.max_output_tokens,
                think=self.think,
                response_format="json" if self.force_json_output else None,
            )
            data = _json_loads(_extract_first_json_container(raw), {})
        except (RuntimeError, ValueError, TypeError, OSError):
            return []
        raw_plans = list(dict(data).get("plans", []))
        out: List[Dict[str, Any]] = []
        for raw_plan in raw_plans:
            plan = self._normalize_plan(raw_plan, answer_type=answer_type)
            if plan is None:
                continue
            out.append(plan)
        return out

    def _build_type_prompt(
        self,
        *,
        window_id: str,
        window_text: str,
        answer_type: str,
    ) -> str:
        answer_shape_hint = {
            "count": "single numeric count only",
            "update": "single current state or changed state",
            "temporal": "single date/time/duration answer",
            "temporal_comparison": "ordered choice between two events or entities",
            "factoid": "single factual entity/value answer",
        }.get(answer_type, "single factual answer")
        temporal_mode_hint = {
            "count": "current or latest count; avoid duration questions",
            "update": "current or latest state update",
            "temporal": "first, latest, duration, or dated event",
            "temporal_comparison": "before/after or which happened first",
            "factoid": "unspecified or current factual lookup",
        }.get(answer_type, "unspecified")
        return (
            "You are generating anticipated memory-retrieval query plans for one bounded memory window.\n"
            "Return JSON only in the form:\n"
            '{"plans":[{"answer_type":"","target_object":"","state_key":"","temporal_mode":"","answer_shape":"","compare_target":"","entities":[],"focus_phrases":[]}]}'
            "\nThis is a controlled generation task. Generate only cache-worthy factual queries.\n"
            "Requirements:\n"
            "- Generate questions answerable with a single short factual answer.\n"
            "- Do not generate preference, recommendation, best-practice, guide, advice, brainstorming, or long-summary questions.\n"
            "- Do not generate open-ended assistant synthesis questions.\n"
            "- Prefer user-state, event-state, count, update, or temporal lookup questions.\n"
            f"- answer_type must be exactly: {answer_type}\n"
            f"- preferred answer shape: {answer_shape_hint}\n"
            f"- preferred temporal mode: {temporal_mode_hint}\n"
            f"- generate at most {max(1, self.max_candidates_per_type)} plans\n"
            f"window_id: {window_id}\n"
            "memory window:\n"
            f"{window_text}\n"
        )

    def _normalize_plan(self, raw_plan: Any, *, answer_type: str) -> Optional[Dict[str, Any]]:
        if not isinstance(raw_plan, dict):
            return None
        target_object = re.sub(r"\s+", " ", str(raw_plan.get("target_object", "")).strip(" ,.;:!?"))
        state_key = re.sub(r"\s+", " ", str(raw_plan.get("state_key", "")).strip(" ,.;:!?"))
        compare_target = re.sub(
            r"\s+", " ", str(raw_plan.get("compare_target", "")).strip(" ,.;:!?")
        )
        temporal_mode = str(raw_plan.get("temporal_mode", "unspecified")).strip().lower()
        answer_shape = str(raw_plan.get("answer_shape", "")).strip().lower()
        if temporal_mode not in self.ALLOWED_TEMPORAL_MODES:
            temporal_mode = "unspecified"
        if answer_shape not in self.ALLOWED_ANSWER_SHAPES:
            answer_shape = self._default_answer_shape(answer_type)
        entities = self._clean_string_list(raw_plan.get("entities", []), limit=6)
        focus_phrases = self._clean_string_list(raw_plan.get("focus_phrases", []), limit=6)
        if not target_object and focus_phrases:
            target_object = str(focus_phrases[0])
        if not target_object:
            return None
        if answer_type == "temporal_comparison" and not compare_target:
            return None
        if answer_type == "update" and not state_key:
            return None
        if answer_type == "count" and answer_shape == "duration":
            return None
        return {
            "answer_type": answer_type,
            "target_object": target_object,
            "state_key": state_key,
            "temporal_mode": temporal_mode,
            "answer_shape": answer_shape,
            "compare_target": compare_target,
            "entities": entities,
            "focus_phrases": focus_phrases,
        }

    @staticmethod
    def _clean_string_list(values: Any, *, limit: int) -> List[str]:
        if not isinstance(values, list):
            values = [values] if str(values or "").strip() else []
        out: List[str] = []
        seen: set[str] = set()
        for raw in values:
            text = re.sub(r"\s+", " ", str(raw or "").strip(" ,.;:!?"))
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
            if len(out) >= max(1, int(limit)):
                break
        return out

    @staticmethod
    def _default_answer_shape(answer_type: str) -> str:
        return {
            "count": "number",
            "update": "entity",
            "temporal": "date",
            "temporal_comparison": "ordered_choice",
            "factoid": "entity",
        }.get(answer_type, "entity")

    def _plans_to_queries(self, plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen_query: set[str] = set()
        seen_signature: set[str] = set()
        for plan in plans:
            query = self._realize_query_from_plan(plan)
            if not query:
                continue
            query_key = _normalize_text(query)
            signature = self._plan_signature(plan)
            if query_key in seen_query or signature in seen_signature:
                continue
            seen_query.add(query_key)
            seen_signature.add(signature)
            candidate = dict(plan)
            candidate["query"] = query
            candidate["score"] = self._score_plan_candidate(candidate)
            candidates.append(candidate)
        ranked = self._rerank_candidates(candidates)
        out: List[Dict[str, Any]] = []
        for item in ranked[: max(1, self.max_queries_per_window)]:
            out.append(
                {
                    "query": str(item.get("query", "")),
                    "answer_type": str(item.get("answer_type", "")),
                    "focus_phrases": list(item.get("focus_phrases", [])),
                    "entities": list(item.get("entities", [])),
                    "temporal_cues": self._derive_temporal_cues(item),
                    "structured_plan": {
                        "target_object": str(item.get("target_object", "")),
                        "state_key": str(item.get("state_key", "")),
                        "temporal_mode": str(item.get("temporal_mode", "")),
                        "answer_shape": str(item.get("answer_shape", "")),
                        "compare_target": str(item.get("compare_target", "")),
                    },
                }
            )
        return out

    @staticmethod
    def _derive_temporal_cues(plan: Dict[str, Any]) -> List[str]:
        cues: List[str] = []
        temporal_mode = str(plan.get("temporal_mode", "")).strip().lower()
        if temporal_mode and temporal_mode != "unspecified":
            cues.append(temporal_mode)
        answer_shape = str(plan.get("answer_shape", "")).strip().lower()
        if answer_shape in {"date", "duration"}:
            cues.append(answer_shape)
        return cues[:4]

    @staticmethod
    def _plan_signature(plan: Dict[str, Any]) -> str:
        return "||".join(
            [
                str(plan.get("answer_type", "")),
                _normalize_text(str(plan.get("target_object", ""))),
                _normalize_text(str(plan.get("state_key", ""))),
                _normalize_text(str(plan.get("compare_target", ""))),
                _normalize_text(str(plan.get("temporal_mode", ""))),
            ]
        )

    def _realize_query_from_plan(self, plan: Dict[str, Any]) -> str:
        answer_type = str(plan.get("answer_type", "")).strip().lower()
        target_object = str(plan.get("target_object", "")).strip()
        state_key = str(plan.get("state_key", "")).strip()
        temporal_mode = str(plan.get("temporal_mode", "")).strip().lower()
        answer_shape = str(plan.get("answer_shape", "")).strip().lower()
        compare_target = str(plan.get("compare_target", "")).strip()
        if not target_object:
            return ""
        if answer_type == "update":
            if temporal_mode in {"current", "latest"}:
                return f"What is the current {state_key} of {target_object}?"
            return f"What changed about the {state_key} of {target_object}?"
        if answer_type == "count":
            prefix = "currently " if temporal_mode in {"current", "latest"} else ""
            return f"How many {target_object} do I {prefix}have?"
        if answer_type == "temporal":
            if answer_shape == "duration" or temporal_mode == "duration":
                return f"How long did {target_object} last?"
            if temporal_mode == "first":
                return f"When did {target_object} first happen?"
            if temporal_mode == "latest":
                return f"When did {target_object} most recently happen?"
            return f"When did {target_object} happen?"
        if answer_type == "temporal_comparison":
            if not compare_target:
                return ""
            return f"Which happened first: {target_object} or {compare_target}?"
        if state_key:
            if temporal_mode in {"current", "latest"}:
                return f"What is the current {state_key} of {target_object}?"
            return f"What is the {state_key} of {target_object}?"
        return f"What is the key fact about {target_object}?"

    def _score_plan_candidate(self, plan: Dict[str, Any]) -> float:
        answer_type = str(plan.get("answer_type", "")).strip().lower()
        target_object = str(plan.get("target_object", "")).strip()
        state_key = str(plan.get("state_key", "")).strip()
        compare_target = str(plan.get("compare_target", "")).strip()
        answer_shape = str(plan.get("answer_shape", "")).strip().lower()
        focus_phrases = list(plan.get("focus_phrases", []))
        entities = list(plan.get("entities", []))
        score = 0.0
        score += 1.5
        score += min(1.0, len(_tokenize(target_object)) * 0.18)
        if state_key:
            score += 0.45
        if compare_target:
            score += 0.55
        if len(entities) >= 1:
            score += 0.25
        if len(focus_phrases) >= 1:
            score += 0.20
        if answer_type == "count" and answer_shape == "number":
            score += 0.55
        if answer_type == "temporal" and answer_shape in {"date", "duration"}:
            score += 0.55
        if answer_type == "temporal_comparison" and answer_shape == "ordered_choice":
            score += 0.55
        if answer_type == "update" and answer_shape in {"entity", "value"}:
            score += 0.40
        if self._looks_open_ended(target_object) or self._looks_open_ended(state_key):
            score -= 1.15
        if len(_tokenize(target_object)) > 10:
            score -= 0.50
        return float(score)

    @staticmethod
    def _looks_open_ended(text: str) -> bool:
        toks = _tokenize(text)
        if not toks:
            return False
        open_terms = {
            "advice",
            "tips",
            "recommendation",
            "recommendations",
            "guide",
            "guidance",
            "best",
            "practices",
            "practice",
            "strategy",
            "strategies",
            "ways",
            "ideas",
            "suggestions",
            "summary",
            "overview",
        }
        return any(tok in open_terms for tok in toks)

    def _rerank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked = sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        selected: List[Dict[str, Any]] = []
        used_types: Dict[str, int] = {}
        for item in ranked:
            answer_type = str(item.get("answer_type", "")).strip().lower()
            if used_types.get(answer_type, 0) >= 1:
                continue
            too_similar = False
            item_focus = set(_tokenize(" ".join(list(item.get("focus_phrases", [])))))
            for chosen in selected:
                chosen_focus = set(_tokenize(" ".join(list(chosen.get("focus_phrases", [])))))
                if item_focus and chosen_focus and len(item_focus & chosen_focus) >= min(
                    len(item_focus), len(chosen_focus)
                ):
                    too_similar = True
                    break
            if too_similar:
                continue
            selected.append(item)
            used_types[answer_type] = used_types.get(answer_type, 0) + 1
            if len(selected) >= max(1, self.max_queries_per_window):
                break
        if len(selected) < max(1, self.max_queries_per_window):
            seen_queries = {_normalize_text(str(item.get("query", ""))) for item in selected}
            for item in ranked:
                q = _normalize_text(str(item.get("query", "")))
                if q in seen_queries:
                    continue
                selected.append(item)
                seen_queries.add(q)
                if len(selected) >= max(1, self.max_queries_per_window):
                    break
        return selected


class PredictiveGraphCacheBuilder:
    """Offline builder for anticipated-query graph cache entries."""

    def __init__(self, manager: Any, store: PredictiveGraphCacheStore, cfg: Dict[str, Any]) -> None:
        self.m = manager
        self.store = store
        self.cfg = dict(cfg)
        self.window_max_chunks = int(self.cfg.get("window_max_chunks", 8))
        self.max_windows = int(self.cfg.get("max_windows", 0) or 0)
        self.include_toolkit_payload = bool(self.cfg.get("include_toolkit_payload", True))
        self.generator = PredictiveQueryGenerator(manager, dict(self.cfg.get("query_generator", {})))

    def build(self, *, max_windows: Optional[int] = None) -> Dict[str, int]:
        windows = self._build_windows()
        if max_windows is None:
            max_windows = self.max_windows
        if max_windows and max_windows > 0:
            windows = windows[: int(max_windows)]
        stored = 0
        discarded = 0
        for window_id, chunks in windows:
            anticipated_queries = self.generator.generate_for_window(
                window_id=window_id,
                chunks=chunks,
            )
            for item in anticipated_queries:
                entry = self._build_entry(window_id=window_id, query_item=item)
                if entry is None:
                    discarded += 1
                    continue
                self.store.upsert(entry)
                stored += 1
        return {"windows": len(windows), "stored_entries": stored, "discarded_entries": discarded}

    def _build_windows(self) -> List[Tuple[str, List[Dict[str, Any]]]]:
        chunks = self.m.mid_memory.get_all_chunks()
        by_session: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            session_id = str(chunk.get("session_id", "") or chunk.get("chunk_session_id", "")).strip()
            if not session_id:
                session_id = f"sessionless-{int(chunk.get('chunk_id', 0) or 0)}"
            by_session.setdefault(session_id, []).append(chunk)
        windows: List[Tuple[str, List[Dict[str, Any]]]] = []
        for session_id, items in by_session.items():
            items = sorted(items, key=lambda x: int(x.get("chunk_id", 0) or 0))
            for start in range(0, len(items), max(1, self.window_max_chunks)):
                subset = items[start : start + max(1, self.window_max_chunks)]
                first_id = int(subset[0].get("chunk_id", 0) or 0)
                last_id = int(subset[-1].get("chunk_id", 0) or 0)
                window_id = f"{session_id}:{first_id}-{last_id}"
                windows.append((window_id, subset))
        return sorted(windows, key=lambda item: item[0])

    def _build_entry(
        self,
        *,
        window_id: str,
        query_item: Dict[str, Any],
    ) -> Optional[PredictiveGraphCacheEntry]:
        query = str(query_item.get("query", "")).strip()
        if not query:
            return None
        self.m.last_query_plan = {}
        bundle = self.m.build_evidence_graph_bundle(query)
        toolkit_payload: Dict[str, Any] = {}
        if self.include_toolkit_payload:
            toolkit_payload = self.m.specialist_layer.run(query=query, graph_bundle=bundle)
        filtered_pack = dict(bundle.get("filtered_pack", {}) or {})
        claim_result = dict(bundle.get("claim_result", {}) or {})
        light_graph = dict(bundle.get("light_graph", {}) or {})
        quality = self._build_quality_signals(
            bundle=bundle,
            filtered_pack=filtered_pack,
            claim_result=claim_result,
            light_graph=light_graph,
            toolkit_payload=toolkit_payload,
        )
        if not self._passes_quality_gate(
            filtered_pack=filtered_pack,
            claim_result=claim_result,
            light_graph=light_graph,
            toolkit_payload=toolkit_payload,
            quality=quality,
        ):
            return None
        plan = dict(bundle.get("query_plan", {}) or self.m._build_query_plan(query))
        cache_id = hashlib.sha1(
            f"{window_id}\n{_normalize_text(query)}\n{str(plan.get('answer_type', ''))}".encode(
                "utf-8"
            )
        ).hexdigest()
        now = _utcnow()
        ttl_hours = _answer_type_ttl_hours(str(plan.get("answer_type", "")), self.cfg)
        retrieval_snapshot = dict(bundle.get("retrieval_snapshot", {}) or {})
        retrieval_snapshot.setdefault("build_timestamp", now.isoformat())
        retrieval_snapshot.setdefault("source_version", int(self.m.mid_memory.get_current_step()))
        anticipated_query_embedding = embed(query, int(self.m.mid_memory.embedding_dim))
        return PredictiveGraphCacheEntry(
            cache_id=cache_id,
            window_id=window_id,
            anticipated_query=query,
            anticipated_query_embedding=[
                float(x) for x in anticipated_query_embedding.astype(np.float32).reshape(-1).tolist()
            ],
            answer_type=str(plan.get("answer_type", "")),
            query_plan_summary={
                "focus_phrases": list(plan.get("focus_phrases", [])),
                "entities": list(plan.get("entities", [])),
                "state_keys": list(plan.get("state_keys", [])),
                "temporal_anchors": list(plan.get("time_terms", [])),
                "time_range": str(plan.get("time_range", "")),
                "need_latest_state": bool(plan.get("need_latest_state", False)),
                "target_object": str(plan.get("target_object", "")),
            },
            retrieval_snapshot=retrieval_snapshot,
            filtered_pack=filtered_pack,
            claim_result=claim_result,
            light_graph=light_graph,
            toolkit_payload=dict(toolkit_payload or {}),
            quality_signals=quality,
            cache_stats={
                "hit_count": 0,
                "last_hit_time": "",
                "last_build_time": now.isoformat(),
                "source_version": int(self.m.mid_memory.get_current_step()),
            },
            expiry_metadata={
                "ttl_deadline": (now + timedelta(hours=max(1, ttl_hours))).isoformat(),
                "invalidated": False,
                "invalidation_reason": "",
            },
        )

    @staticmethod
    def _build_quality_signals(
        *,
        bundle: Dict[str, Any],
        filtered_pack: Dict[str, Any],
        claim_result: Dict[str, Any],
        light_graph: Dict[str, Any],
        toolkit_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        selected: List[Dict[str, Any]] = []
        for key in ("core_evidence", "supporting_evidence", "conflict_evidence", "backup_evidence"):
            selected.extend(list(filtered_pack.get(key, []) or []))
        top_score = 0.0
        for item in selected:
            top_score = max(top_score, float(item.get("score", 0.0) or 0.0))
        graph_stats = dict(light_graph.get("stats", {}) or {})
        claim_stats = dict(claim_result.get("stats", {}) or {})
        return {
            "top_evidence_score": float(top_score),
            "selected_evidence_count": len(selected),
            "selected_claim_count": int(
                claim_stats.get("claims", len(list(claim_result.get("claims", []) or []))) or 0
            ),
            "graph_structural_edge_count": int(graph_stats.get("edge_count", 0) or 0),
            "graph_total_latency_sec": float(
                dict(bundle.get("stage_latency_sec", {}) or {}).get("graph_total", 0.0) or 0.0
            ),
            "toolkit_activated": bool(dict(toolkit_payload or {}).get("tool_payload")),
        }

    @staticmethod
    def _passes_quality_gate(
        *,
        filtered_pack: Dict[str, Any],
        claim_result: Dict[str, Any],
        light_graph: Dict[str, Any],
        toolkit_payload: Dict[str, Any],
        quality: Dict[str, Any],
    ) -> bool:
        if not filtered_pack:
            return False
        has_claims = bool(claim_result and (claim_result.get("claims") or claim_result.get("support_units")))
        has_graph = bool(
            light_graph
            and (
                list(light_graph.get("edges", []) or [])
                or int(dict(light_graph.get("stats", {}) or {}).get("edge_count", 0) or 0) > 0
            )
        )
        has_toolkit = bool(dict(toolkit_payload or {}).get("tool_payload"))
        if not has_claims and not has_graph and not has_toolkit:
            return False
        return (
            float(quality.get("top_evidence_score", 0.0) or 0.0) > 0.0
            and int(quality.get("selected_evidence_count", 0) or 0) > 0
        )


class PredictiveGraphCacheMaintenance:
    """TTL invalidation and low-value eviction worker."""

    def __init__(self, store: PredictiveGraphCacheStore, cfg: Dict[str, Any]) -> None:
        self.store = store
        self.cfg = dict(cfg)

    def run(self) -> Dict[str, int]:
        now = _utcnow()
        max_entries_per_window = int(self.cfg.get("max_entries_per_window", 8))
        stale_before = (now - timedelta(hours=int(self.cfg.get("low_value_max_age_hours", 336)))).isoformat()
        deleted = self.store.forget_low_value_entries(
            max_entries_per_window=max_entries_per_window,
            stale_before_iso=stale_before,
        )
        return {"deleted_entries": deleted}
