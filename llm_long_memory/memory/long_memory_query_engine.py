"""Query/ranking engine extracted from LongMemory to reduce class bloat."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set


class LongMemoryQueryEngine:
    """Run long-memory retrieval and graph-neighbor boosting."""

    def __init__(self, memory: Any) -> None:
        self.m = memory

    def _parse_keywords(self, raw_keywords: Any) -> List[str]:
        try:
            return [
                str(x).strip().lower()
                for x in list(json.loads(str(raw_keywords or "[]")))
                if str(x).strip()
            ]
        except (TypeError, ValueError, json.JSONDecodeError):
            return self.m._tokenize(str(raw_keywords or ""))

    def _build_detail_index(self, details: List[Any]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for row in details:
            kind = str(row["kind"] or "").strip().lower()
            text = str(row["text"] or "").strip()
            if (not kind) or (not text):
                continue
            out.setdefault(kind, [])
            if text not in out[kind]:
                out[kind].append(text)
        return out

    @staticmethod
    def _tokenize_time(text: str) -> Set[str]:
        value = str(text or "").strip().lower()
        if not value:
            return set()
        out: Set[str] = set()
        for pat in (
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
            r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b",
            r"\b\d{4}\b",
        ):
            for m in re.findall(pat, value):
                token = str(m).strip().lower()
                if token:
                    out.add(token)
        return out

    def _extract_time_constraints(self, query: str) -> Dict[str, Any]:
        q = str(query or "").strip().lower()
        time_tokens = self._tokenize_time(q)
        has_temporal_terms = any(
            t in q
            for t in (
                "first",
                "earlier",
                "later",
                "before",
                "after",
                "when",
                "date",
                "day",
                "month",
                "year",
            )
        )
        return {
            "has_explicit_time_tokens": bool(time_tokens),
            "time_tokens": time_tokens,
            "has_temporal_terms": bool(has_temporal_terms),
        }

    def _event_time_tokens(self, detail_idx: Dict[str, List[str]]) -> Set[str]:
        out: Set[str] = set()
        for kind in ("time", "time_start", "time_end", "source_date"):
            for value in detail_idx.get(kind, []):
                out.update(self._tokenize_time(value))
        return out

    def _compose_context_text(
        self,
        *,
        skeleton: str,
        node_rows: List[Any],
        details: List[Any],
    ) -> str:
        base = self.m._build_compact_node_context(
            skeleton=skeleton,
            node_rows=node_rows,
            max_chars=max(80, int(self.m.context_max_chars_per_item * 0.45)),
        )
        detail_idx = self._build_detail_index(details)
        parts: List[str] = [base]

        evidence = detail_idx.get("evidence", [])
        if evidence:
            parts.append(f"evidence={evidence[0][: self.m.context_evidence_max_chars]}")

        if self.m.context_include_source:
            source = detail_idx.get("source", [])
            if source:
                parts.append(f"source={source[0][: self.m.context_source_max_chars]}")

        time_rows = detail_idx.get("time", [])
        if time_rows:
            parts.append(f"time={time_rows[0][:64]}")
        time_start = detail_idx.get("time_start", [])
        time_end = detail_idx.get("time_end", [])
        if time_start:
            parts.append(f"time_start={time_start[0][:32]}")
        if time_end:
            parts.append(f"time_end={time_end[0][:32]}")
        source_date = detail_idx.get("source_date", [])
        if source_date:
            parts.append(f"source_date={source_date[0][:32]}")
        location_rows = detail_idx.get("location", [])
        if location_rows:
            parts.append(f"location={location_rows[0][:64]}")
        return " || ".join([x for x in parts if x])[: self.m.context_max_chars_per_item]

    def _score_event_row(
        self,
        *,
        row: Any,
        channel: str,
        channel_weight: float,
        q_tokens: Set[str],
        q_emb: Any,
        time_constraints: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        event_id = str(row["event_id"])
        skeleton = str(row["skeleton_text"] or "").strip()
        if not skeleton:
            return None
        fact_type = str(row["fact_type"] or "event").strip().lower()
        if (not self.m.retrieval_include_hints) and fact_type == "hint":
            return None
        keywords = self._parse_keywords(row["keywords"])

        node_rows = self.m.store.fetch_event_nodes(
            event_id,
            self.m.node_context_per_event if self.m.node_graph_enabled else 0,
        )
        node_edge_rows = self.m.store.fetch_event_node_edges(
            event_id,
            (self.m.node_context_per_event * 2) if self.m.node_graph_enabled else 0,
        )
        node_texts = [str(n["node_text"]).strip() for n in node_rows if str(n["node_text"]).strip()]
        edge_texts: List[str] = []
        edge_weight_total = 0.0
        edge_weight_count = 0
        for e in node_edge_rows:
            rel = str(e["relation"] or "").strip()
            ft = str(e["from_text"] or "").strip()
            tt = str(e["to_text"] or "").strip()
            if rel or ft or tt:
                edge_texts.append(" ".join([x for x in [rel, ft, tt] if x]))
            try:
                edge_weight_total += float(e["weight"] or 0.0)
                edge_weight_count += 1
            except (TypeError, ValueError):
                pass

        node_corpus = set(self.m._tokenize(" ".join(node_texts)))
        edge_corpus = set(self.m._tokenize(" ".join(edge_texts)))
        corpus = set(keywords) | set(self.m._tokenize(skeleton)) | node_corpus | edge_corpus
        overlap_count, keyword_score = self.m._keyword_overlap_features(q_tokens, corpus)
        _, node_keyword_score = self.m._keyword_overlap_features(q_tokens, node_corpus)
        _, node_edge_keyword_score = self.m._keyword_overlap_features(q_tokens, edge_corpus)
        emb_score = self.m._cosine(q_emb, self.m.store.blob_to_arr(row["skeleton_embedding"]))
        delta = max(0, self.m.current_step - int(row["last_seen_step"] or 0))
        recency = 1.0 / (1.0 + float(delta))

        if overlap_count >= self.m.keyword_primary_min_overlap:
            base_score = (
                self.m.lexical_weight * keyword_score
                + self.m.embedding_weight * emb_score
                + self.m.recency_weight * recency
            )
        else:
            base_score = (
                self.m.embedding_fallback_weight * emb_score
                + self.m.recency_weight * recency
            )
        if self.m.node_graph_enabled:
            base_score += self.m.node_boost_weight * node_keyword_score
            edge_strength = (
                edge_weight_total / float(edge_weight_count)
                if edge_weight_count > 0
                else 0.0
            )
            base_score += self.m.node_edge_boost_weight * node_edge_keyword_score * edge_strength

        score = float(channel_weight) * float(base_score)
        if score < self.m.retrieval_min_score:
            return None

        details = self.m.store.fetch_event_details(event_id, self.m.details_per_event)
        detail_idx = self._build_detail_index(details)
        if self.m.temporal_filter_enabled:
            query_time_tokens = set(time_constraints.get("time_tokens", set()))
            event_time_tokens = self._event_time_tokens(detail_idx)
            if bool(time_constraints.get("has_explicit_time_tokens", False)):
                if query_time_tokens:
                    if query_time_tokens.intersection(event_time_tokens):
                        score *= float(self.m.temporal_query_time_boost)
                    else:
                        score *= float(self.m.temporal_query_no_time_penalty)
            elif bool(time_constraints.get("has_temporal_terms", False)):
                if event_time_tokens:
                    score *= float(self.m.temporal_query_time_boost)
                else:
                    score *= float(self.m.temporal_query_no_time_penalty)
            if score < self.m.retrieval_min_score:
                return None
        text = self._compose_context_text(
            skeleton=skeleton,
            node_rows=node_rows,
            details=details,
        )
        return {
            "event_id": event_id,
            "text": text,
            "score": float(score),
            "keywords": keywords,
            "fact_type": fact_type,
            "channel": channel,
            "status": "active" if channel == "active" else "superseded",
            "fact_key": str(row["fact_key"] or "").strip(),
        }

    def _add_rows(
        self,
        *,
        rows: List[Any],
        channel: str,
        channel_weight: float,
        q_tokens: Set[str],
        q_emb: Any,
        time_constraints: Dict[str, Any],
        score_map: Dict[str, Dict[str, Any]],
    ) -> None:
        for row in rows:
            item = self._score_event_row(
                row=row,
                channel=channel,
                channel_weight=channel_weight,
                q_tokens=q_tokens,
                q_emb=q_emb,
                time_constraints=time_constraints,
            )
            if item is None:
                continue
            event_id = str(item["event_id"])
            prev = score_map.get(event_id)
            if (prev is None) or (float(item["score"]) > float(prev["score"])):
                score_map[event_id] = item

    def _inject_neighbor_events(self, score_map: Dict[str, Dict[str, Any]]) -> None:
        base_ranked = sorted(
            [x for x in score_map.values() if str(x.get("channel")) == "active"],
            key=lambda x: float(x["score"]),
            reverse=True,
        )
        seed_ids = [str(x["event_id"]) for x in base_ranked[: self.m.graph_neighbor_limit]]
        for seed_id in seed_ids:
            for edge in self.m.store.fetch_edges_from(seed_id, self.m.graph_neighbor_limit):
                neighbor_id = str(edge["to_event_id"])
                edge_weight = float(edge["weight"] or 0.0)
                if neighbor_id in score_map:
                    score_map[neighbor_id]["score"] = float(score_map[neighbor_id]["score"]) + (
                        self.m.graph_neighbor_weight * edge_weight
                    )
                    continue
                rows = self.m.store.fetch_events_by_ids([neighbor_id])
                if not rows:
                    continue
                row = rows[0]
                skeleton = str(row["skeleton_text"] or "").strip()
                if not skeleton:
                    continue
                keywords = self._parse_keywords(row["keywords"])
                details = self.m.store.fetch_event_details(neighbor_id, self.m.details_per_event)
                node_rows = self.m.store.fetch_event_nodes(
                    neighbor_id,
                    self.m.node_context_per_event if self.m.node_graph_enabled else 0,
                )
                text = self._compose_context_text(
                    skeleton=skeleton,
                    node_rows=node_rows,
                    details=details,
                )
                score_map[neighbor_id] = {
                    "event_id": neighbor_id,
                    "text": text,
                    "score": self.m.graph_neighbor_weight * edge_weight,
                    "keywords": keywords,
                    "fact_type": str(row["fact_type"] or "event"),
                    "channel": "active",
                    "status": "active",
                }

    def query(self, query_text: str) -> List[Dict[str, Any]]:
        query = str(query_text).strip()
        if not query:
            return []
        q_struct = self.m._extract_query_struct(query)
        q_tokens = set(self.m._keyword_candidates_from_text(" ".join(list(q_struct.get("keywords", [])))))
        if not q_tokens:
            q_tokens = set(self.m._tokenize(query))
        time_constraints = self._extract_time_constraints(query)
        q_skeleton = str(q_struct.get("skeleton", "")).strip() or query
        q_emb = self.m._safe_embed(q_skeleton)
        score_map: Dict[str, Dict[str, Any]] = {}

        self._add_rows(
            rows=self.m.store.fetch_active_events(),
            channel="active",
            channel_weight=1.0,
            q_tokens=q_tokens,
            q_emb=q_emb,
            time_constraints=time_constraints,
            score_map=score_map,
        )
        if self.m.history_enabled:
            self._add_rows(
                rows=self.m.store.fetch_superseded_events(self.m.history_max_candidates),
                channel="history",
                channel_weight=self.m.history_weight,
                q_tokens=q_tokens,
                q_emb=q_emb,
                time_constraints=time_constraints,
                score_map=score_map,
            )

        self._inject_neighbor_events(score_map)
        ranked = sorted(score_map.values(), key=lambda x: float(x["score"]), reverse=True)
        return ranked[: self.m.retrieval_top_k]
