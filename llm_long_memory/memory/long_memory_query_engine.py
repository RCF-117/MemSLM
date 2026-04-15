"""Fact-oriented query/ranking engine for LongMemory."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set


class LongMemoryQueryEngine:
    """Run fact retrieval using atomic-value scoring and minimal graph boosting."""

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
        for pattern in (
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
            r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b",
            r"\b\d{4}\b",
        ):
            for match in re.findall(pattern, value):
                token = str(match).strip().lower()
                if token:
                    out.add(token)
        return out

    def _extract_time_constraints(self, query: str) -> Dict[str, Any]:
        q = str(query or "").strip().lower()
        time_tokens = self._tokenize_time(q)
        has_temporal_terms = any(
            token in q
            for token in (
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
                "last",
                "current",
                "currently",
            )
        )
        return {
            "has_explicit_time_tokens": bool(time_tokens),
            "time_tokens": time_tokens,
            "has_temporal_terms": bool(has_temporal_terms),
        }

    def _extract_query_intent(self, query: str) -> Dict[str, bool]:
        lowered = str(query or "").strip().lower()
        return {
            "asks_where": "where" in lowered or "location" in lowered,
            "asks_when": any(t in lowered for t in ("when", "date", "time", "year", "month", "day")),
            "asks_how_many": any(t in lowered for t in ("how many", "number of", "count", "total")),
            "asks_current": any(t in lowered for t in ("current", "currently", "latest", "now")),
        }

    def _event_time_tokens(self, detail_idx: Dict[str, List[str]]) -> Set[str]:
        out: Set[str] = set()
        for kind in ("time", "time_start", "time_end", "source_date"):
            for value in detail_idx.get(kind, []):
                out.update(self._tokenize_time(value))
        return out

    def _collect_node_texts(self, node_rows: List[Any]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for row in node_rows:
            kind = str(row["node_kind"] or "").strip().lower()
            text = str(row["node_text"] or "").strip()
            if (not kind) or (not text):
                continue
            out.setdefault(kind, [])
            if text not in out[kind]:
                out[kind].append(text)
        return out

    def _compose_context_text(
        self,
        *,
        skeleton: str,
        detail_idx: Dict[str, List[str]],
        node_texts: Dict[str, List[str]],
    ) -> str:
        canonical_fact = (detail_idx.get("canonical_fact", []) or [skeleton])[0]
        value_text = detail_idx.get("value", []) or node_texts.get("value", []) or node_texts.get("object", [])
        time_text = detail_idx.get("time", []) or node_texts.get("time", [])
        location_text = detail_idx.get("location", []) or node_texts.get("location", [])
        evidence = detail_idx.get("evidence", []) or node_texts.get("evidence", [])
        parts: List[str] = [canonical_fact]
        if value_text:
            parts.append(f"value={value_text[0][:80]}")
        if location_text:
            parts.append(f"location={location_text[0][:80]}")
        if time_text:
            parts.append(f"time={time_text[0][:80]}")
        if evidence:
            parts.append(f"evidence={evidence[0][: self.m.context_evidence_max_chars]}")
        if self.m.context_include_source:
            source = detail_idx.get("source", [])
            if source:
                parts.append(f"source={source[0][: self.m.context_source_max_chars]}")
        return " || ".join([part for part in parts if part])[: self.m.context_max_chars_per_item]

    def _semantic_text(
        self,
        *,
        skeleton: str,
        detail_idx: Dict[str, List[str]],
        node_texts: Dict[str, List[str]],
    ) -> str:
        fields: List[str] = [skeleton]
        for kind in ("canonical_fact", "subject", "predicate", "value", "location", "time", "fact_slot"):
            fields.extend(detail_idx.get(kind, []))
        for kind in ("subject", "predicate", "action", "value", "object", "location", "time", "keyword"):
            fields.extend(node_texts.get(kind, []))
        return " ".join([field for field in fields if field]).strip()

    def _score_event_row(
        self,
        *,
        row: Any,
        channel: str,
        channel_weight: float,
        q_tokens: Set[str],
        q_emb: Any,
        time_constraints: Dict[str, Any],
        query_intent: Dict[str, bool],
    ) -> Optional[Dict[str, Any]]:
        event_id = str(row["event_id"])
        skeleton = str(row["skeleton_text"] or "").strip()
        if not skeleton:
            return None
        fact_type = str(row["fact_type"] or "event").strip().lower()
        if (not self.m.retrieval_include_hints) and fact_type == "hint":
            return None
        keywords = self._parse_keywords(row["keywords"])
        details = self.m.store.fetch_event_details(event_id, self.m.details_per_event)
        detail_idx = self._build_detail_index(details)
        node_rows = self.m.store.fetch_event_nodes(
            event_id,
            self.m.node_context_per_event if self.m.node_graph_enabled else 0,
        )
        node_texts = self._collect_node_texts(node_rows)

        semantic_text = self._semantic_text(
            skeleton=skeleton,
            detail_idx=detail_idx,
            node_texts=node_texts,
        )
        corpus = set(keywords) | set(self.m._tokenize(semantic_text))
        overlap_count, keyword_score = self.m._keyword_overlap_features(q_tokens, corpus)
        value_text = " ".join(detail_idx.get("value", []) + detail_idx.get("value_norm", []))
        evidence_text = " ".join(
            detail_idx.get("answer_span_raw", [])
            + detail_idx.get("evidence", [])
            + detail_idx.get("raw_value", [])
        )
        value_tokens = set(self.m._tokenize(value_text))
        evidence_tokens = set(self.m._tokenize(evidence_text))
        value_overlap_count = len(q_tokens.intersection(value_tokens))
        evidence_overlap_count = len(q_tokens.intersection(evidence_tokens))
        value_overlap_score = (
            float(value_overlap_count) / float(max(1, len(q_tokens))) if q_tokens else 0.0
        )
        evidence_overlap_score = (
            float(evidence_overlap_count) / float(max(1, len(q_tokens))) if q_tokens else 0.0
        )
        emb_score = self.m._cosine(q_emb, self.m.store.blob_to_arr(row["skeleton_embedding"]))

        delta = max(0, self.m.current_step - int(row["last_seen_step"] or 0))
        recency = 1.0 / (1.0 + float(delta))
        if overlap_count >= self.m.keyword_primary_min_overlap:
            score = (
                self.m.lexical_weight * keyword_score
                + self.m.embedding_weight * emb_score
                + self.m.recency_weight * recency
            )
        else:
            score = (
                self.m.embedding_fallback_weight * emb_score
                + self.m.recency_weight * recency
            )
        if value_overlap_count >= int(self.m.value_overlap_min_tokens):
            score += float(self.m.value_overlap_weight) * float(value_overlap_score)
        if evidence_overlap_count >= int(self.m.evidence_overlap_min_tokens):
            score += float(self.m.evidence_overlap_weight) * float(evidence_overlap_score)

        value_type = (detail_idx.get("value_type", []) or [""])[0].strip().lower()
        has_location = bool(detail_idx.get("location", []) or node_texts.get("location", []))
        has_time = bool(self._event_time_tokens(detail_idx))
        has_value = bool(detail_idx.get("value", []) or node_texts.get("value", []))
        if query_intent.get("asks_where") and has_location:
            score += self.m.node_boost_weight
        if query_intent.get("asks_when") and has_time:
            score += self.m.node_boost_weight
        if query_intent.get("asks_how_many") and has_value and value_type == "number":
            score += self.m.node_boost_weight
        if query_intent.get("asks_current") and channel == "active" and fact_type == "state_fact":
            score += self.m.node_edge_boost_weight

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

        score = float(channel_weight) * float(score)
        if score < self.m.retrieval_min_score:
            return None

        text = self._compose_context_text(
            skeleton=skeleton,
            detail_idx=detail_idx,
            node_texts=node_texts,
        )
        return {
            "event_id": event_id,
            "text": text,
            "score": float(score),
            "keywords": keywords,
            "fact_type": fact_type,
            "channel": channel,
            "status": "active" if channel == "active" else "history",
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
        query_intent: Dict[str, bool],
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
                query_intent=query_intent,
            )
            if item is None:
                continue
            event_id = str(item["event_id"])
            prev = score_map.get(event_id)
            if (prev is None) or (float(item["score"]) > float(prev["score"])):
                score_map[event_id] = item

    def _inject_neighbor_events(self, score_map: Dict[str, Dict[str, Any]]) -> None:
        base_ranked = sorted(
            [item for item in score_map.values() if str(item.get("channel")) == "active"],
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        seed_ids = [str(item["event_id"]) for item in base_ranked[: self.m.graph_neighbor_limit]]
        for seed_id in seed_ids:
            for edge in self.m.store.fetch_edges_from(seed_id, self.m.graph_neighbor_limit):
                relation = str(edge["relation"] or "").strip().lower()
                if relation not in {"updates", "extends"}:
                    continue
                neighbor_id = str(edge["to_event_id"])
                edge_weight = float(edge["weight"] or 0.0)
                if neighbor_id in score_map:
                    score_map[neighbor_id]["score"] = float(score_map[neighbor_id]["score"]) + (
                        self.m.graph_neighbor_weight * edge_weight
                    )

    def query(self, query_text: str) -> List[Dict[str, Any]]:
        query = str(query_text).strip()
        if not query:
            return []
        q_struct = self.m._extract_query_struct(query)
        q_tokens = set(
            self.m._keyword_candidates_from_text(" ".join(list(q_struct.get("keywords", []))))
        )
        if not q_tokens:
            q_tokens = set(self.m._tokenize(query))
        q_skeleton = str(q_struct.get("skeleton", "")).strip() or query
        q_emb = self.m._safe_embed(q_skeleton)
        time_constraints = self._extract_time_constraints(query)
        query_intent = self._extract_query_intent(query)

        score_map: Dict[str, Dict[str, Any]] = {}
        self._add_rows(
            rows=self.m.store.fetch_active_events(),
            channel="active",
            channel_weight=1.0,
            q_tokens=q_tokens,
            q_emb=q_emb,
            time_constraints=time_constraints,
            query_intent=query_intent,
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
                query_intent=query_intent,
                score_map=score_map,
            )

        self._inject_neighbor_events(score_map)
        ranked = sorted(score_map.values(), key=lambda item: float(item["score"]), reverse=True)
        return ranked[: self.m.retrieval_top_k]
