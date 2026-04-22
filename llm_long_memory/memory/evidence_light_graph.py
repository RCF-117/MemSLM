"""Deterministic light evidence-graph builder for filtered claim sets."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple


class EvidenceLightGraph:
    """Build a compact question-scoped graph from extracted claims."""

    _CONTENT_TOKEN_RE = re.compile(r"[a-z0-9]+")
    _NAMED_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9'/-]+\b")
    _CONTENT_STOPWORDS = {
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "get",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "should",
        "some",
        "than",
        "that",
        "the",
        "their",
        "them",
        "these",
        "this",
        "to",
        "use",
        "using",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "would",
        "you",
        "your",
    }
    _NAMED_TOKEN_STOPWORDS = {
        "A",
        "An",
        "And",
        "Any",
        "Can",
        "Could",
        "Do",
        "Does",
        "How",
        "I",
        "If",
        "In",
        "My",
        "Of",
        "Or",
        "The",
        "This",
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "Would",
        "You",
        "Your",
    }
    _PREFERENCE_QUERY_RE = re.compile(
        r"\b("
        r"recommend|suggest(?:ion|ions)?|advice|tips?|"
        r"what should i|which .* should i|"
        r"ways to|resources? where i can learn|learn more about|"
        r"any suggestions|ideas? for"
        r")\b"
    )
    _PREFERENCE_PREDICATES = {"preferred_direction", "supported_reason"}
    _GENERIC_SUBJECTS = {"resource", "resources", "recipe", "recipes", "option", "options", "activity", "activities"}

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

    @classmethod
    def _extract_content_tokens(cls, *texts: str) -> List[str]:
        seen = set()
        out: List[str] = []
        for text in texts:
            for token in cls._CONTENT_TOKEN_RE.findall(str(text or "").lower()):
                if len(token) <= 2 or token in cls._CONTENT_STOPWORDS or token in seen:
                    continue
                seen.add(token)
                out.append(token)
        return out

    @classmethod
    def _extract_named_tokens(cls, *texts: str) -> List[str]:
        seen = set()
        out: List[str] = []
        for text in texts:
            for token in cls._NAMED_TOKEN_RE.findall(str(text or "")):
                if token in cls._NAMED_TOKEN_STOPWORDS or token in seen:
                    continue
                seen.add(token)
                out.append(token)
        return out

    @classmethod
    def _effective_answer_type(cls, query: str, filtered_pack: Dict[str, Any]) -> str:
        base = str(filtered_pack.get("answer_type", "")).strip().lower()
        if base in {"count", "temporal", "temporal_comparison", "update"}:
            return base
        if cls._PREFERENCE_QUERY_RE.search(str(query or "").lower()):
            return "preference"
        return base

    @staticmethod
    def _claim_text_parts(claim: Dict[str, Any]) -> Tuple[str, str, str]:
        return (
            str(claim.get("subject", "")),
            str(claim.get("predicate", "")),
            str(claim.get("value", "")),
        )

    def _query_support_weight(
        self,
        *,
        claim: Dict[str, Any],
        query_tokens: Sequence[str],
        query_named_tokens: Sequence[str],
        focus_tokens: Sequence[str],
        target_tokens: Sequence[str],
        answer_type: str,
    ) -> float:
        subject, predicate, value = self._claim_text_parts(claim)
        subject_tokens = set(self._extract_content_tokens(subject))
        predicate_tokens = set(self._extract_content_tokens(predicate))
        value_tokens = set(self._extract_content_tokens(value))
        claim_named_tokens = set(self._extract_named_tokens(subject, value))
        query_token_set = set(query_tokens)
        focus_token_set = set(focus_tokens)
        target_token_set = set(target_tokens)
        query_named_set = {str(t) for t in query_named_tokens}

        subject_overlap = len(subject_tokens & query_token_set)
        predicate_overlap = len(predicate_tokens & query_token_set)
        value_overlap = len(value_tokens & query_token_set)
        focus_overlap = len((subject_tokens | predicate_tokens | value_tokens) & focus_token_set)
        target_overlap = len((subject_tokens | value_tokens) & target_token_set)
        named_overlap = len(claim_named_tokens & query_named_set)

        score = 0.0
        if subject_overlap:
            score += min(0.36, 0.18 * subject_overlap)
        if predicate_overlap:
            score += min(0.16, 0.08 * predicate_overlap)
        if value_overlap:
            score += min(0.16, 0.04 * value_overlap)
        if focus_overlap:
            score += min(0.20, 0.05 * focus_overlap)
        if target_overlap:
            score += min(0.18, 0.09 * target_overlap)
        if named_overlap:
            score += min(0.24, 0.12 * named_overlap)

        state_key = self._text_key(str(claim.get("state_key", "")))
        predicate_key = self._text_key(str(claim.get("predicate", "")))
        if answer_type == "preference" and (
            state_key == "preference" or predicate_key in self._PREFERENCE_PREDICATES
        ):
            score += 0.18
        elif answer_type in {"update", "knowledge-update"} and state_key in {"location", "time", "status", "schedule"}:
            score += 0.12
        elif answer_type in {"temporal", "temporal_comparison"} and claim.get("time_anchor"):
            score += 0.10

        confidence = float(claim.get("confidence", 0.0) or 0.0)
        score += min(0.10, confidence * 0.10)
        return round(min(1.0, max(0.0, score)), 4)

    def _project_claims(
        self,
        *,
        claims: Sequence[Dict[str, Any]],
        answer_type: str,
        query_tokens: Sequence[str],
        query_named_tokens: Sequence[str],
        focus_tokens: Sequence[str],
        target_tokens: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if answer_type != "preference":
            return [dict(c) for c in claims]

        kept: List[Tuple[float, Dict[str, Any]]] = []
        summaries: List[Tuple[float, Dict[str, Any]]] = []
        for claim in claims:
            claim_dict = dict(claim)
            weight = self._query_support_weight(
                claim=claim_dict,
                query_tokens=query_tokens,
                query_named_tokens=query_named_tokens,
                focus_tokens=focus_tokens,
                target_tokens=target_tokens,
                answer_type=answer_type,
            )
            claim_dict["_graph_support_weight"] = weight
            predicate_key = self._text_key(str(claim_dict.get("predicate", "")))
            state_key = self._text_key(str(claim_dict.get("state_key", "")))
            subject_key = self._text_key(str(claim_dict.get("subject", "")))
            value_key = self._text_key(str(claim_dict.get("value", "")))
            if state_key == "preference" or predicate_key in self._PREFERENCE_PREDICATES:
                summaries.append((weight, claim_dict))
                continue
            if subject_key in self._GENERIC_SUBJECTS and value_key in self._GENERIC_SUBJECTS and weight < 0.30:
                continue
            kept.append((weight, claim_dict))

        kept.sort(key=lambda item: (item[0], float(item[1].get("confidence", 0.0) or 0.0)), reverse=True)
        summaries.sort(
            key=lambda item: (item[0], float(item[1].get("confidence", 0.0) or 0.0), len(str(item[1].get("value", "")))),
            reverse=True,
        )
        selected: List[Dict[str, Any]] = []
        seen = set()
        for _weight, claim in summaries + kept[:4]:
            claim_key = str(claim.get("claim_id", "")) or self._text_key(
                "|".join([str(claim.get("subject", "")), str(claim.get("predicate", "")), str(claim.get("value", ""))])
            )
            if claim_key in seen:
                continue
            seen.add(claim_key)
            selected.append(claim)
        return selected

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
            if not src or not dst or src == dst:
                return
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

        answer_type = self._effective_answer_type(query, filtered_pack)
        focus_phrases = list(filtered_pack.get("focus_phrases") or [])
        target_object = str(filtered_pack.get("target_object", "") or "")
        query_tokens = self._extract_content_tokens(query)
        query_named_tokens = self._extract_named_tokens(query)
        focus_tokens = self._extract_content_tokens(*focus_phrases)
        target_tokens = self._extract_content_tokens(target_object)
        projected_claims = self._project_claims(
            claims=claims,
            answer_type=answer_type,
            query_tokens=query_tokens,
            query_named_tokens=query_named_tokens,
            focus_tokens=focus_tokens,
            target_tokens=target_tokens,
        )

        add_node(
            {
                "id": query_node_id,
                "type": "query",
                "label": self._normalize_space(query),
                "meta": {
                    "answer_type": answer_type,
                    "intent": str(filtered_pack.get("intent", "")),
                },
            }
        )

        for claim in list(projected_claims):
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
            support_weight = self._query_support_weight(
                claim=claim,
                query_tokens=query_tokens,
                query_named_tokens=query_named_tokens,
                focus_tokens=focus_tokens,
                target_tokens=target_tokens,
                answer_type=answer_type,
            )
            add_edge(claim_id, query_node_id, "supports_query", {"weight": support_weight})

        claims_by_subject: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for claim in list(projected_claims):
            subject_key = self._text_key(str(claim.get("subject", "")))
            claim_ref = claim_nodes.get(str(claim.get("claim_id", "")))
            if not subject_key or not claim_ref:
                continue
            claims_by_subject.setdefault(subject_key, []).append((claim_ref, dict(claim)))

        for _subject_key, subject_claims in claims_by_subject.items():
            deduped_subject_claims: List[Tuple[str, Dict[str, Any]]] = []
            seen_refs = set()
            for claim_ref, claim in subject_claims:
                if claim_ref in seen_refs:
                    continue
                seen_refs.add(claim_ref)
                deduped_subject_claims.append((claim_ref, claim))
            subject_claims = deduped_subject_claims[: self.max_edges_per_subject]
            for idx, (left_id, left_claim) in enumerate(subject_claims):
                for right_id, right_claim in subject_claims[idx + 1 :]:
                    add_edge(left_id, right_id, "same_subject")
                    left_state = self._text_key(str(left_claim.get("state_key", "")))
                    right_state = self._text_key(str(right_claim.get("state_key", "")))
                    left_value = self._text_key(str(left_claim.get("value", "")))
                    right_value = self._text_key(str(right_claim.get("value", "")))
                    left_predicate = self._text_key(str(left_claim.get("predicate", "")))
                    right_predicate = self._text_key(str(right_claim.get("predicate", "")))
                    left_type = self._text_key(str(left_claim.get("claim_type", "")))
                    right_type = self._text_key(str(right_claim.get("claim_type", "")))
                    can_update = (
                        left_state
                        and left_state == right_state
                        and left_value
                        and right_value
                        and left_value != right_value
                        and left_state != "preference"
                        and left_predicate not in self._PREFERENCE_PREDICATES
                        and right_predicate not in self._PREFERENCE_PREDICATES
                        and left_type == "state_snapshot"
                        and right_type == "state_snapshot"
                    )
                    if can_update:
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
            "answer_type": answer_type,
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
