"""Relation-aware graph retrieval on top of long-memory events/edges."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple


class GraphQueryEngine:
    """Build graph evidence packs with seed retrieval + relation expansion."""

    def __init__(self, long_memory: Any) -> None:
        self.m = long_memory

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    def _tokenize(self, text: str) -> Set[str]:
        return {
            t
            for t in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if t and t not in {"the", "a", "an", "to", "of", "and", "or", "in", "on", "my"}
        }

    def _infer_intent(self, query: str) -> str:
        q = str(query or "").lower()
        if any(x in q for x in ("how many", "number of", "count")):
            return "count"
        if any(x in q for x in ("who did", "first", "before", "after", "earlier", "later", "when")):
            return "temporal"
        if any(x in q for x in ("current", "currently", "latest", "switch", "updated", "now")):
            return "update"
        if any(x in q for x in ("prefer", "preference", "suggest", "recommend")):
            return "preference"
        return "lookup"

    def _extract_anchors(self, query: str) -> List[str]:
        text = self._norm(query)
        if not text:
            return []
        stop = {
            "the",
            "a",
            "an",
            "to",
            "of",
            "and",
            "or",
            "in",
            "on",
            "my",
            "is",
            "are",
            "was",
            "were",
            "what",
            "who",
            "where",
            "when",
            "how",
            "many",
            "did",
            "do",
            "does",
            "i",
            "me",
            "we",
            "our",
            "with",
            "for",
            "currently",
            "latest",
        }
        anchors: List[str] = []
        for pat in (r"'([^']{2,80})'", r"\"([^\"]{2,80})\""):
            for match in re.findall(pat, text):
                value = self._norm(match)
                if value:
                    anchors.append(value)
        if not anchors:
            keywords = list(self.m._keyword_candidates_from_text(text))
            anchors.extend([k for k in keywords if len(k) >= 3 and k not in stop][:6])
        out: List[str] = []
        seen: Set[str] = set()
        for anchor in anchors:
            low = anchor.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(anchor)
        return out[:8]

    def _build_plan(self, query: str) -> Dict[str, Any]:
        intent = self._infer_intent(query)
        anchors = self._extract_anchors(query)
        need_latest = intent in {"update", "temporal"}
        relation_pref = ["updates", "extends", "same_subject"]
        if intent == "count":
            relation_pref = ["same_subject", "extends", "updates"]
        return {
            "intent": intent,
            "anchors": anchors,
            "need_latest_state": need_latest,
            "relation_preference": relation_pref,
        }

    def _seed_events(
        self, query: str, anchors: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        seeds: Dict[str, float] = {}
        debug_rows: Dict[str, Dict[str, Any]] = {}
        for item in self.m.query(query)[: max(4, int(self.m.retrieval_top_k))]:
            event_id = str(item.get("event_id", "")).strip()
            if not event_id:
                continue
            value = float(item.get("score", 0.0))
            seeds[event_id] = max(float(seeds.get(event_id, 0.0)), value)
            row = debug_rows.setdefault(
                event_id,
                {"event_id": event_id, "seed_score": 0.0, "seed_sources": [], "anchor_hits": []},
            )
            row["seed_score"] = max(float(row.get("seed_score", 0.0)), value)
            sources = list(row.get("seed_sources", []))
            if "semantic_seed" not in sources:
                sources.append("semantic_seed")
            row["seed_sources"] = sources

        for anchor in anchors:
            for row in self.m.store.search_event_nodes(anchor, limit=20):
                event_id = str(row["event_id"] or "").strip()
                if not event_id:
                    continue
                text = str(row["node_text"] or "")
                overlap = len(self._tokenize(anchor).intersection(self._tokenize(text)))
                bonus = 0.15 + min(0.35, 0.08 * float(overlap))
                seeds[event_id] = max(float(seeds.get(event_id, 0.0)), bonus)
                dbg = debug_rows.setdefault(
                    event_id,
                    {"event_id": event_id, "seed_score": 0.0, "seed_sources": [], "anchor_hits": []},
                )
                dbg["seed_score"] = max(float(dbg.get("seed_score", 0.0)), bonus)
                sources = list(dbg.get("seed_sources", []))
                if "node_anchor_seed" not in sources:
                    sources.append("node_anchor_seed")
                dbg["seed_sources"] = sources
                hits = list(dbg.get("anchor_hits", []))
                hit_text = self._norm(anchor)
                if hit_text and hit_text not in hits:
                    hits.append(hit_text)
                dbg["anchor_hits"] = hits
        return seeds, debug_rows

    def _detail_index(self, event_id: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for row in self.m.store.fetch_event_details(event_id, 64):
            kind = str(row["kind"] or "").strip().lower()
            text = self._norm(str(row["text"] or ""))
            if kind and text and kind not in out:
                out[kind] = text
        return out

    def _relation_bonus(self, relation: str, intent: str) -> float:
        rel = str(relation or "").strip().lower()
        if intent == "update":
            if rel == "updates":
                return 0.45
            if rel == "same_subject":
                return 0.18
        if intent == "temporal":
            if rel == "updates":
                return 0.25
            if rel == "extends":
                return 0.20
        if intent == "count":
            if rel == "same_subject":
                return 0.25
            if rel == "extends":
                return 0.12
        return 0.10

    def _slot_bonus(self, event_id: str, intent: str) -> float:
        rows = self.m.store.fetch_event_node_edges(event_id, 12)
        rels = {str(row["relation"] or "").strip().lower() for row in rows}
        bonus = 0.0
        if intent == "count" and "has_value" in rels:
            bonus += 0.20
        if intent == "temporal" and "has_time" in rels:
            bonus += 0.20
        if intent in {"update", "lookup"} and "has_location" in rels:
            bonus += 0.12
        if "supported_by" in rels:
            bonus += 0.06
        return bonus

    def _expand_with_relations(
        self,
        seed_scores: Dict[str, float],
        plan: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        ranked: Dict[str, Dict[str, Any]] = {}
        anchors = set(self._tokenize(" ".join(list(plan.get("anchors", [])))))
        intent = str(plan.get("intent", "lookup"))

        # Keep top seeds to avoid noisy graph explosions.
        seed_items = sorted(seed_scores.items(), key=lambda x: float(x[1]), reverse=True)[
            : max(4, int(self.m.graph_neighbor_limit) * 2)
        ]
        for event_id, seed_score in seed_items:
            event = self.m.store.fetch_event_by_id(event_id)
            if event is None:
                continue
            detail = self._detail_index(event_id)
            text = self._norm(str(event["skeleton_text"] or ""))
            evidence = self._norm(detail.get("evidence", detail.get("answer_span_raw", "")))
            tokens = self._tokenize(text + " " + evidence)
            cover = 0.0
            if anchors:
                cover = float(len(tokens.intersection(anchors))) / float(max(1, len(anchors)))
            score = float(seed_score) + (0.25 * cover) + self._slot_bonus(event_id, intent)
            ranked[event_id] = {
                "event_id": event_id,
                "score": score,
                "path": [f"seed:{event_id}"],
                "text": text,
                "detail": detail,
            }

            for edge in self.m.store.fetch_edges_from(event_id, self.m.graph_neighbor_limit):
                rel = str(edge["relation"] or "").strip().lower()
                if rel not in {"updates", "extends", "same_subject"}:
                    continue
                nbr = str(edge["to_event_id"] or "").strip()
                if not nbr:
                    continue
                nbr_row = self.m.store.fetch_event_by_id(nbr)
                if nbr_row is None:
                    continue
                nbr_detail = self._detail_index(nbr)
                nbr_text = self._norm(str(nbr_row["skeleton_text"] or ""))
                base = float(seed_score)
                rel_bonus = self._relation_bonus(rel, intent) * float(edge["weight"] or 0.0)
                nbr_score = base + rel_bonus + self._slot_bonus(nbr, intent)
                prev = ranked.get(nbr)
                if prev is None or float(prev["score"]) < nbr_score:
                    ranked[nbr] = {
                        "event_id": nbr,
                        "score": nbr_score,
                        "path": [f"seed:{event_id}", f"{event_id}-[{rel}]->{nbr}"],
                        "text": nbr_text,
                        "detail": nbr_detail,
                    }
        return ranked

    def _render_snippet(self, item: Dict[str, Any]) -> str:
        text = self._norm(str(item.get("text", "")))
        detail = dict(item.get("detail", {}))
        value = self._norm(detail.get("value", ""))
        time_text = self._norm(detail.get("time", detail.get("time_start", "")))
        location = self._norm(detail.get("location", ""))
        evidence = self._norm(detail.get("evidence", detail.get("answer_span_raw", "")))
        path = " -> ".join([self._norm(p) for p in list(item.get("path", [])) if self._norm(p)])
        parts: List[str] = []
        if path:
            parts.append(f"path={path}")
        if text:
            parts.append(f"fact={text[:150]}")
        if value:
            parts.append(f"value={value[:80]}")
        if time_text:
            parts.append(f"time={time_text[:60]}")
        if location:
            parts.append(f"location={location[:60]}")
        if evidence:
            parts.append(f"evidence={evidence[:140]}")
        return " || ".join(parts)

    def query(self, query: str, max_items: int = 4, include_debug: bool = False) -> Dict[str, Any]:
        plan = self._build_plan(query)
        seed_scores, seed_debug = self._seed_events(query, list(plan.get("anchors", [])))
        if not seed_scores:
            out = {"plan": plan, "snippets": []}
            if include_debug:
                out["seed_debug"] = []
                out["ranked_debug"] = []
            return out
        ranked_map = self._expand_with_relations(seed_scores, plan)
        ranked = sorted(ranked_map.values(), key=lambda x: float(x["score"]), reverse=True)
        snippets = [self._render_snippet(item) for item in ranked[: max(1, int(max_items))]]
        out: Dict[str, Any] = {"plan": plan, "snippets": [s for s in snippets if s]}
        if include_debug:
            seed_rows: List[Dict[str, Any]] = []
            for event_id, item in seed_debug.items():
                event = self.m.store.fetch_event_by_id(event_id)
                seed_rows.append(
                    {
                        "event_id": event_id,
                        "seed_score": float(item.get("seed_score", 0.0)),
                        "seed_sources": list(item.get("seed_sources", [])),
                        "anchor_hits": list(item.get("anchor_hits", [])),
                        "seed_text": self._norm(str((event or {}).get("skeleton_text", "")))[:160],
                    }
                )
            seed_rows.sort(key=lambda x: float(x.get("seed_score", 0.0)), reverse=True)
            ranked_rows: List[Dict[str, Any]] = []
            for item in ranked[: max(1, int(max_items) * 2)]:
                ranked_rows.append(
                    {
                        "event_id": str(item.get("event_id", "")),
                        "score": float(item.get("score", 0.0)),
                        "path": list(item.get("path", [])),
                        "text": self._norm(str(item.get("text", "")))[:200],
                        "detail": dict(item.get("detail", {})),
                    }
                )
            out["seed_debug"] = seed_rows
            out["ranked_debug"] = ranked_rows
        return out
