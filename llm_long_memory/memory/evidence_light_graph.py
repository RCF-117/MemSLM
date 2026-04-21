"""Deterministic light evidence-graph builder for filtered claim sets."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple


class EvidenceLightGraph:
    """Build a compact question-scoped graph from extracted claims."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(cfg or {})
        self.max_edges_per_subject = max(2, int(cfg.get("graph_max_edges_per_subject", 16)))

    @staticmethod
    def _normalize_space(text: str) -> str:
        return " ".join(str(text or "").split())

    @staticmethod
    def _text_key(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

    @staticmethod
    def _stable_id(prefix: str, text: str) -> str:
        digest = hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()[:12]
        return f"{prefix}_{digest}"

    def _parse_time(self, value: str) -> Optional[datetime]:
        text = self._normalize_space(value)
        if not text:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        lower = text.lower()
        m = re.match(r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+(\d{1,2})$", lower)
        if m:
            try:
                return datetime.strptime(f"{text}, 2000", "%b %d, %Y")
            except ValueError:
                return None
        return None

    def build_graph(
        self,
        *,
        query: str,
        filtered_pack: Dict[str, Any],
        claims: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        node_ids = set()
        edge_ids = set()
        entity_nodes: Dict[str, str] = {}
        claim_nodes: Dict[str, str] = {}
        query_node_id = "query_root"

        def add_node(node: Dict[str, Any]) -> None:
            node_id = str(node.get("id", "")).strip()
            if not node_id or node_id in node_ids:
                return
            node_ids.add(node_id)
            nodes.append(node)

        def add_edge(src: str, dst: str, edge_type: str, meta: Optional[Dict[str, Any]] = None) -> None:
            payload = dict(meta or {})
            edge_id = self._stable_id("edge", f"{src}|{dst}|{edge_type}|{payload}")
            if edge_id in edge_ids:
                return
            edge_ids.add(edge_id)
            edges.append(
                {
                    "id": edge_id,
                    "source": src,
                    "target": dst,
                    "type": edge_type,
                    **payload,
                }
            )

        add_node(
            {
                "id": query_node_id,
                "type": "query",
                "label": self._normalize_space(query),
                "meta": {
                    "answer_type": str(filtered_pack.get("answer_type", "")),
                    "intent": str(filtered_pack.get("intent", "")),
                },
            }
        )

        for claim in list(claims):
            subject = self._normalize_space(str(claim.get("subject", "")))
            if not subject:
                continue
            entity_key = self._text_key(subject)
            entity_id = entity_nodes.get(entity_key)
            if entity_id is None:
                entity_id = self._stable_id("entity", entity_key)
                entity_nodes[entity_key] = entity_id
                add_node({"id": entity_id, "type": "entity", "label": subject})

            claim_type = str(claim.get("claim_type", "fact_statement")).strip().lower()
            node_type = "fact"
            if claim_type == "state_snapshot":
                node_type = "state"
            elif claim_type == "event_record":
                node_type = "event"
            claim_label = f"{subject} | {claim.get('predicate', '')} | {claim.get('value', '')}"
            claim_id = self._stable_id("claim", claim_label + "|" + str(claim.get("time_anchor", "")))
            claim_nodes[str(claim.get("claim_id", claim_id))] = claim_id
            add_node(
                {
                    "id": claim_id,
                    "type": node_type,
                    "label": claim_label,
                    "meta": dict(claim),
                }
            )
            add_edge(entity_id, claim_id, "mentions", {"evidence_ids": list(claim.get("evidence_ids", []))})
            add_edge(claim_id, query_node_id, "supports_query", {"weight": round(float(claim.get("confidence", 0.0)), 4)})

        claims_by_subject: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for claim in list(claims):
            subject_key = self._text_key(str(claim.get("subject", "")))
            claim_ref = claim_nodes.get(str(claim.get("claim_id", "")))
            if not subject_key or not claim_ref:
                continue
            claims_by_subject.setdefault(subject_key, []).append((claim_ref, dict(claim)))

        for _subject_key, subject_claims in claims_by_subject.items():
            subject_claims = subject_claims[: self.max_edges_per_subject]
            for idx, (left_id, left_claim) in enumerate(subject_claims):
                for right_id, right_claim in subject_claims[idx + 1 :]:
                    add_edge(left_id, right_id, "same_subject")
                    left_state = self._text_key(str(left_claim.get("state_key", "")))
                    right_state = self._text_key(str(right_claim.get("state_key", "")))
                    left_value = self._text_key(str(left_claim.get("value", "")))
                    right_value = self._text_key(str(right_claim.get("value", "")))
                    if left_state and left_state == right_state and left_value and right_value and left_value != right_value:
                        add_edge(
                            left_id,
                            right_id,
                            "updates",
                            {"state_key": str(left_claim.get("state_key", ""))},
                        )
                    left_time = self._parse_time(str(left_claim.get("time_anchor", "")))
                    right_time = self._parse_time(str(right_claim.get("time_anchor", "")))
                    if left_time is not None and right_time is not None and left_time != right_time:
                        if left_time < right_time:
                            add_edge(left_id, right_id, "before")
                            add_edge(right_id, left_id, "after")
                        else:
                            add_edge(right_id, left_id, "before")
                            add_edge(left_id, right_id, "after")

        return {
            "query": self._normalize_space(query),
            "answer_type": str(filtered_pack.get("answer_type", "")),
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "entity_count": sum(1 for n in nodes if str(n.get("type", "")) == "entity"),
                "claim_count": sum(
                    1 for n in nodes if str(n.get("type", "")) in {"fact", "state", "event"}
                ),
            },
        }
