"""Retrieval helper for LongMemory."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np


class LongMemoryRetriever:
    """Handle query scoring, fusion, and snippet building for long memory."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def infer_query_fact_type(self, query_text: str) -> str:
        q = str(query_text).strip().lower()
        if not q:
            return "generic"
        if any(k in q for k in ("when", "date", "time", "how many days", "before", "after")):
            return "time"
        if any(k in q for k in ("where", "location", "city", "country", "place")):
            return "location"
        if any(k in q for k in ("who", "name", "person")):
            return "person"
        if any(k in q for k in ("what", "which", "did", "happen")):
            return "event"
        return "generic"

    def query(self, query_text: str) -> List[Dict[str, Any]]:
        o = self.owner
        if not o.enabled:
            return []
        q_tokens = set(o._tokenize(query_text))
        if not q_tokens:
            return []
        q_vec = o._safe_embed(query_text)
        query_fact_type = self.infer_query_fact_type(query_text)

        with o._lock:
            rows = o.store.fetch_active_events()
            now_step = max(1, int(o.current_step))

        scored: List[Dict[str, Any]] = []
        for row in rows:
            skeleton_text = str(row["skeleton_text"])
            try:
                keywords = list(json.loads(str(row["keywords"])))
            except (ValueError, TypeError):
                keywords = o._tokenize(skeleton_text)
            key_set = {str(k).strip().lower() for k in keywords if str(k).strip()}
            overlap = len(q_tokens.intersection(key_set))
            lexical = float(overlap) / float(max(1, len(q_tokens)))

            emb_score = 0.0
            if o.retrieval_use_embedding:
                emb = o.store.blob_to_arr(row["skeleton_embedding"])
                emb_score = max(0.0, o._cosine(q_vec, emb))

            delta = max(0, now_step - int(row["last_seen_step"] or 0))
            recency = 1.0 / (1.0 + float(delta))

            score = (
                o.retrieval_lexical_weight * lexical
                + o.retrieval_embedding_weight * emb_score
                + o.retrieval_recency_weight * recency
            )
            event_fact_type = str(row["fact_type"] or "").strip().lower() or "event"
            if query_fact_type != "generic":
                if event_fact_type == query_fact_type:
                    score += o.retrieval_fact_type_boost
                else:
                    score = max(0.0, score - o.retrieval_fact_type_mismatch_penalty)
            role = str(row["role"]).strip().lower()
            role_weight = float(o.retrieval_role_weights.get(role, 1.0))
            score *= role_weight
            if score < o.retrieval_min_score:
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
                    "fact_type": event_fact_type,
                }
            )

        scored.sort(key=lambda x: (float(x["score"]), len(str(x["text"]))), reverse=True)
        return scored[: o.retrieval_top_k]

    def query_multi(self, query_texts: List[str]) -> List[Dict[str, Any]]:
        o = self.owner
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
                if o.rewrite_fusion_mode == "rrf":
                    add = 1.0 / float(o.rewrite_fusion_rrf_k + rank)
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
        return merged[: o.retrieval_top_k]

    def build_context_from_hits(self, hits: List[Dict[str, Any]]) -> List[str]:
        o = self.owner
        snippets: List[str] = []
        with o._lock:
            for hit in hits[: o.context_max_items]:
                event_id = str(hit["event_id"])
                detail_rows = o.store.fetch_event_details(event_id, o.details_per_event)
                details = [
                    f"{str(r['kind'])}: {str(r['text'])}"
                    for r in detail_rows
                    if str(r["text"]).strip()
                ]
                base = f"[event] {str(hit['text'])}"
                fact_type = str(hit.get("fact_type", "")).strip().lower()
                if fact_type:
                    base = f"[{fact_type}] {str(hit['text'])}"
                if details:
                    base += " | " + " ; ".join(details)
                snippets.append(base[: o.context_max_chars_per_item])
        return snippets

    def build_context_snippets(self, query_text: str) -> List[str]:
        return self.build_context_from_hits(self.query(query_text))

    def build_context_snippets_multi(self, query_texts: List[str]) -> List[str]:
        return self.build_context_from_hits(self.query_multi(query_texts))
