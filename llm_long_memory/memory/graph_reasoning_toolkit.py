"""Graph-only toolkit for question-scoped light-graph reasoning."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

from llm_long_memory.memory.query_intent import extract_query_intent
from llm_long_memory.memory.temporal_query_utils import parse_choice_targets


class GraphReasoningToolkit:
    """Produce compact graph-grounded reasoning payloads.

    The toolkit only consumes question-scoped light graphs. It performs:
    query routing -> query-aware subgraph projection -> task solver ->
    structured tool payload.
    """

    def __init__(self, manager: Any) -> None:
        self.m = manager
        self._count_stopwords = {
            "how",
            "many",
            "number",
            "count",
            "total",
            "of",
            "the",
            "a",
            "an",
            "did",
            "do",
            "does",
            "i",
            "we",
            "you",
            "my",
            "our",
            "your",
            "in",
            "on",
            "for",
            "with",
        }
        self._content_stopwords = {
            "what",
            "which",
            "who",
            "when",
            "where",
            "how",
            "did",
            "do",
            "does",
            "is",
            "are",
            "was",
            "were",
            "have",
            "has",
            "had",
            "i",
            "we",
            "you",
            "my",
            "our",
            "your",
            "the",
            "a",
            "an",
            "of",
            "to",
            "for",
            "in",
            "on",
            "with",
            "after",
            "before",
            "between",
            "and",
            "or",
            "me",
            "it",
            "that",
            "this",
            "there",
            "be",
            "been",
            "being",
        }

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text or "").lower())

    @staticmethod
    def _singular(token: str) -> str:
        raw = str(token or "").strip().lower()
        if len(raw) <= 3:
            return raw
        if raw.endswith("ies") and len(raw) > 4:
            return raw[:-3] + "y"
        if raw.endswith("es") and len(raw) > 4 and raw[-3] in {"s", "x", "z"}:
            return raw[:-2]
        if raw.endswith("s") and len(raw) > 3:
            return raw[:-1]
        return raw

    def _content_tokens(self, text: str, *, extra_stopwords: Sequence[str] = ()) -> List[str]:
        stopwords = set(self._content_stopwords).union({str(x).strip().lower() for x in extra_stopwords})
        out: List[str] = []
        for token in self._tokenize(text):
            norm = self._singular(token)
            if len(norm) <= 2:
                continue
            if norm in stopwords:
                continue
            out.append(norm)
        return list(dict.fromkeys(out))

    def _classify_intent(self, query: str, light_graph: Dict[str, object]) -> str:
        answer_type = str(dict(light_graph or {}).get("answer_type", "")).strip().lower()
        if answer_type in {"temporal_count", "count", "update"}:
            return answer_type
        if answer_type == "temporal_comparison":
            return "temporal_compare"
        flags = extract_query_intent(query)
        lowered = str(query or "").lower()
        if flags.get("asks_how_many") and (
            flags.get("asks_when")
            or "how long" in lowered
            or any(unit in lowered for unit in ("week", "weeks", "month", "months", "day", "days", "year", "years", "hour", "hours", "minute", "minutes"))
        ):
            return "temporal_count"
        if flags.get("asks_how_many"):
            return "count"
        if flags.get("asks_compare") and len(parse_choice_targets(query, max_options=4, default_target_k=2) or []) >= 2:
            return "temporal_compare"
        if flags.get("asks_current") or flags.get("asks_where"):
            return "update"
        return "generic"

    def _query_object_heads(self, query: str) -> List[str]:
        heads: List[str] = []
        for token in self._tokenize(query):
            if token in self._count_stopwords:
                continue
            norm = self._singular(token)
            if len(norm) <= 2:
                continue
            heads.append(norm)
        return list(dict.fromkeys(heads))

    @staticmethod
    def _extract_spelled_number(text: str) -> int:
        words = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
        }
        for token in re.findall(r"[a-z]+", str(text or "").lower()):
            if token in words:
                return words[token]
        return -1

    @staticmethod
    def _claim_text(item: Dict[str, object]) -> str:
        subject = str(item.get("subject", "")).strip()
        predicate = str(item.get("predicate", "")).strip()
        value = str(item.get("value", "")).strip()
        if subject and predicate and value:
            return f"{subject} | {predicate} | {value}"
        return " | ".join(x for x in [subject, predicate, value] if x).strip()

    @staticmethod
    def _parse_date(text: str) -> List[datetime]:
        out: List[datetime] = []
        raw = str(text or "")
        for match in re.finditer(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", raw):
            try:
                out.append(datetime(int(match.group(1)), int(match.group(2)), int(match.group(3))))
            except ValueError:
                pass
        for match in re.finditer(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", raw):
            month = int(match.group(1))
            day = int(match.group(2))
            year = int(match.group(3)) if match.group(3) else 2000
            if year < 100:
                year += 2000
            try:
                out.append(datetime(year, month, day))
            except ValueError:
                pass
        months = {
            "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
            "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
            "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
            "nov": 11, "november": 11, "dec": 12, "december": 12,
        }
        for match in re.finditer(
            r"\b("
            r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
            r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
            raw,
            flags=re.IGNORECASE,
        ):
            month = months.get(match.group(1).lower(), 1)
            day = int(match.group(2))
            year = int(match.group(3)) if match.group(3) else 2000
            try:
                out.append(datetime(year, month, day))
            except ValueError:
                pass
        return out

    @staticmethod
    def _format_duration(delta_days: int, query: str) -> str:
        lowered = str(query or "").lower()
        if "year" in lowered and delta_days >= 365:
            years = max(1, round(delta_days / 365))
            return f"{years} year" if years == 1 else f"{years} years"
        if "month" in lowered and delta_days >= 28:
            months = max(1, round(delta_days / 30))
            return f"{months} month" if months == 1 else f"{months} months"
        if delta_days % 7 == 0 and delta_days >= 7:
            weeks = max(1, delta_days // 7)
            return f"{weeks} week" if weeks == 1 else f"{weeks} weeks"
        return f"{delta_days} day" if delta_days == 1 else f"{delta_days} days"

    def _graph_claim_items(self, light_graph: Dict[str, object]) -> List[Dict[str, object]]:
        graph = dict(light_graph or {})
        nodes = list(graph.get("nodes", []))
        edges = list(graph.get("edges", []))
        support_weights: Dict[str, float] = {}
        for edge in edges:
            if str(edge.get("type", "")).strip() != "supports_query":
                continue
            src = str(edge.get("source", "")).strip()
            if not src:
                continue
            try:
                weight = float(edge.get("weight", 0.0) or 0.0)
            except (TypeError, ValueError):
                weight = 0.0
            support_weights[src] = max(weight, support_weights.get(src, 0.0))
        items: List[Dict[str, object]] = []
        for node in nodes:
            node_type = str(node.get("type", "")).strip().lower()
            if node_type not in {"fact", "state", "event"}:
                continue
            node_id = str(node.get("id", "")).strip()
            meta = dict(node.get("meta", {}) or {})
            item = dict(meta)
            item["node_id"] = node_id
            item["node_type"] = node_type
            item["support_weight"] = float(support_weights.get(node_id, 0.0))
            try:
                item["confidence"] = float(item.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                item["confidence"] = 0.0
            items.append(item)
        items.sort(
            key=lambda x: (
                float(x.get("support_weight", 0.0)),
                float(x.get("confidence", 0.0)),
                len(str(x.get("value", ""))),
            ),
            reverse=True,
        )
        return items

    def _claim_overlap(self, item: Dict[str, object], tokens: Sequence[str]) -> float:
        token_set = {self._singular(tok) for tok in tokens if tok}
        if not token_set:
            return 0.0
        claim_tokens = {self._singular(tok) for tok in self._tokenize(self._claim_text(item))}
        if not claim_tokens:
            return 0.0
        return float(len(token_set.intersection(claim_tokens))) / float(len(token_set))

    def _claim_projection_score(
        self,
        item: Dict[str, object],
        *,
        query: str,
        intent: str,
        query_tokens: Sequence[str],
        object_heads: Sequence[str],
        option_tokens: Sequence[Sequence[str]],
    ) -> float:
        score = 0.65 * float(item.get("support_weight", 0.0))
        score += 0.35 * float(item.get("confidence", 0.0))
        overlap = self._claim_overlap(item, query_tokens)
        score += 1.10 * overlap
        dates = self._parse_date(str(item.get("time_anchor", ""))) or self._parse_date(self._claim_text(item))
        predicate = self._normalize(str(item.get("predicate", ""))).lower()
        claim_type = self._normalize(str(item.get("claim_type", ""))).lower()

        if intent == "count":
            head_overlap = self._claim_overlap(item, object_heads)
            score += 1.25 * head_overlap
            if predicate in {"count", "number", "total", "item", "member"}:
                score += 0.35
        elif intent in {"temporal_count", "temporal_compare"}:
            if dates:
                score += 0.35
        elif intent == "update":
            if claim_type == "state_snapshot":
                score += 0.40
            if predicate in {"location", "status", "state", "count", "ratio", "time"}:
                score += 0.25
        elif intent == "preference":
            if predicate in {"preferred_direction", "supported_reason"}:
                score += 0.30

        if option_tokens:
            best_option_overlap = max((self._claim_overlap(item, toks) for toks in option_tokens), default=0.0)
            score += 0.90 * best_option_overlap
        return score

    def _project_subgraph(
        self,
        *,
        query: str,
        light_graph: Dict[str, object],
        claims: Sequence[Dict[str, object]],
        intent: str,
    ) -> Dict[str, object]:
        query_tokens = self._content_tokens(query)
        object_heads = self._query_object_heads(query)
        options = parse_choice_targets(query, max_options=4, default_target_k=2) or []
        option_token_groups = [self._content_tokens(option) for option in options]

        scored_claims: List[Dict[str, object]] = []
        for raw in claims:
            claim = dict(raw)
            claim["_projection_score"] = self._claim_projection_score(
                claim,
                query=query,
                intent=intent,
                query_tokens=query_tokens,
                object_heads=object_heads,
                option_tokens=option_token_groups,
            )
            scored_claims.append(claim)

        scored_claims.sort(key=lambda item: float(item.get("_projection_score", 0.0)), reverse=True)
        selected: List[Dict[str, object]] = []
        selected_ids: set[str] = set()

        def _add_claim(claim: Dict[str, object]) -> None:
            node_id = str(claim.get("node_id", "")).strip()
            if not node_id or node_id in selected_ids:
                return
            selected_ids.add(node_id)
            selected.append(claim)

        for claim in scored_claims:
            if float(claim.get("_projection_score", 0.0)) < 0.35:
                continue
            _add_claim(claim)
            if len(selected) >= 8:
                break
        if intent == "temporal_compare" and len(options) >= 2:
            for option_tokens in option_token_groups[:2]:
                option_claims = sorted(
                    scored_claims,
                    key=lambda item: self._claim_overlap(item, option_tokens),
                    reverse=True,
                )
                for claim in option_claims[:2]:
                    if self._claim_overlap(claim, option_tokens) >= 0.50:
                        _add_claim(claim)
        if not selected and scored_claims:
            for claim in scored_claims[:4]:
                _add_claim(claim)

        relevant_edge_types = {"supports_query", "same_subject", "updates", "before", "after"}
        selected_edges: List[Dict[str, object]] = []
        for edge in list(dict(light_graph or {}).get("edges", [])):
            edge_type = str(edge.get("type", "")).strip()
            if edge_type not in relevant_edge_types:
                continue
            src = str(edge.get("source", "")).strip()
            dst = str(edge.get("target", "")).strip()
            if edge_type == "supports_query":
                if src in selected_ids:
                    selected_edges.append(dict(edge))
                continue
            if src in selected_ids and dst in selected_ids:
                selected_edges.append(dict(edge))

        return {
            "claims": sorted(selected, key=lambda item: float(item.get("_projection_score", 0.0)), reverse=True),
            "edges": selected_edges,
            "query_tokens": query_tokens,
            "object_heads": object_heads,
            "options": options,
            "stats": {
                "input_claims": len(list(claims)),
                "selected_claims": len(selected),
                "selected_edges": len(selected_edges),
            },
        }

    def _abstain_payload(
        self,
        *,
        intent: str,
        reason: str,
        subgraph: Dict[str, object],
        structured_result: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        return {
            "intent": intent,
            "activated": False,
            "answer_candidate": "",
            "raw_candidate": "",
            "verified": False,
            "verified_candidate": "",
            "verification_reason": "",
            "verified_used_claim_ids": [],
            "confidence": 0.0,
            "used_claim_ids": [],
            "rationale_lines": [],
            "structured_result": structured_result or {},
            "abstain_reason": reason,
            "subgraph_stats": dict(subgraph.get("stats", {}) or {}),
        }

    def _verify_payload(
        self,
        *,
        intent: str,
        answer_candidate: str,
        structured_result: Dict[str, object],
        subgraph: Dict[str, object],
        used_claim_ids: Sequence[str],
    ) -> Tuple[bool, str]:
        candidate = self._normalize(answer_candidate)
        if not candidate:
            return False, "missing_candidate"
        claims = list(subgraph.get("claims", []))
        if intent == "count":
            explicit = [self._normalize(str(x)) for x in list(structured_result.get("explicit_count_candidates", [])) if self._normalize(str(x))]
            enumerated = [self._normalize(str(x)) for x in list(structured_result.get("enumerated_items", [])) if self._normalize(str(x))]
            if len(explicit) != 1 or explicit[0] != candidate:
                return False, "count_requires_unique_explicit_signal"
            supporting_count_claims = 0
            for claim in claims:
                predicate = self._normalize(str(claim.get("predicate", ""))).lower()
                text = self._claim_text(dict(claim))
                numeric = self._extract_numeric_candidates(text)
                value = self._normalize(str(claim.get("value", "")))
                if predicate in {"count", "number", "total"} and (candidate in numeric or value == candidate):
                    supporting_count_claims += 1
            if enumerated:
                try:
                    if len(enumerated) == int(candidate) and int(candidate) >= 2:
                        return True, "count_verified_by_enumeration"
                except ValueError:
                    return False, "count_candidate_not_integer"
            if supporting_count_claims >= 2:
                return True, "count_verified_by_duplicate_count_claims"
            return False, "count_missing_second_support"
        if intent == "temporal_count":
            anchor_a = self._normalize(str(structured_result.get("anchor_a_date", "")))
            anchor_b = self._normalize(str(structured_result.get("anchor_b_date", "")))
            if anchor_a and anchor_b and anchor_a != anchor_b and len([x for x in used_claim_ids if str(x).strip()]) >= 2:
                return True, "temporal_count_dual_anchor_verified"
            return False, "temporal_count_missing_dual_anchor_verification"
        if intent == "temporal_compare":
            mode = self._normalize(str(structured_result.get("resolution_mode", ""))).lower()
            if mode == "graph_edge":
                return (len([x for x in used_claim_ids if str(x).strip()]) >= 2, "temporal_compare_graph_edge_verified")
            if mode == "date_compare":
                left = self._normalize(str(structured_result.get("option_a_best_anchor", "")))
                right = self._normalize(str(structured_result.get("option_b_best_anchor", "")))
                if left and right and left != right:
                    return True, "temporal_compare_dual_dates_verified"
            return False, "temporal_compare_not_verified"
        if intent == "update":
            resolution_mode = self._normalize(str(structured_result.get("resolution_mode", ""))).lower()
            state_key = self._normalize(str(structured_result.get("state_key", ""))).lower()
            trusted_state_keys = {"location", "status", "state", "count", "ratio", "time", "amount"}
            if resolution_mode == "update_edge" and state_key in trusted_state_keys:
                return True, "update_edge_verified"
            return False, "update_requires_trusted_update_edge"
        return False, "intent_not_verifiable"

    def _finalize_payload(
        self,
        *,
        intent: str,
        answer_candidate: str,
        confidence: float,
        used_claim_ids: Sequence[str],
        rationale_lines: Sequence[str],
        structured_result: Dict[str, object],
        subgraph: Dict[str, object],
        abstain_reason: str = "",
    ) -> Dict[str, object]:
        clean_lines = [self._normalize(str(line)) for line in rationale_lines if self._normalize(str(line))]
        clean_candidate = self._normalize(answer_candidate)
        clean_used_claim_ids = [str(x).strip() for x in used_claim_ids if str(x).strip()]
        verified, verification_reason = self._verify_payload(
            intent=intent,
            answer_candidate=clean_candidate,
            structured_result=dict(structured_result or {}),
            subgraph=subgraph,
            used_claim_ids=clean_used_claim_ids,
        )
        payload = {
            "intent": intent,
            "activated": bool(clean_candidate),
            "answer_candidate": clean_candidate,
            "raw_candidate": clean_candidate,
            "verified": bool(verified and clean_candidate),
            "verified_candidate": clean_candidate if verified and clean_candidate else "",
            "verification_reason": self._normalize(verification_reason),
            "verified_used_claim_ids": list(clean_used_claim_ids if verified and clean_candidate else []),
            "confidence": float(confidence),
            "used_claim_ids": clean_used_claim_ids,
            "rationale_lines": clean_lines[:6],
            "structured_result": dict(structured_result or {}),
            "abstain_reason": self._normalize(abstain_reason),
            "subgraph_stats": dict(subgraph.get("stats", {}) or {}),
        }
        payload["summary_lines"] = list(payload["rationale_lines"])
        payload["summary_text"] = "\n".join(payload["summary_lines"]).strip()
        return payload

    def _extract_numeric_candidates(self, text: str) -> List[str]:
        out: List[str] = []
        for match in re.finditer(r"\b(\d+)\b", str(text or "")):
            out.append(str(int(match.group(1))))
        spelled = self._extract_spelled_number(text)
        if spelled >= 0:
            out.append(str(spelled))
        return list(dict.fromkeys(out))

    def _is_count_item_value(
        self,
        *,
        value: str,
        predicate: str,
        subject: str,
        object_heads: Sequence[str],
    ) -> bool:
        value_norm = self._normalize(value).strip(" ,.;:!?\"'")
        if not value_norm:
            return False
        lowered = value_norm.lower()
        if lowered in {"yes", "no", "count", "number", "total", "item", "items"}:
            return False
        value_tokens = {self._singular(tok) for tok in self._tokenize(value_norm)}
        subject_tokens = {self._singular(tok) for tok in self._tokenize(subject)}
        predicate_norm = self._normalize(predicate).lower()
        enumerative_predicates = {"item", "member", "entity", "object", "entry", "component", "name"}
        if predicate_norm in {"count", "number", "total"}:
            return False
        if object_heads:
            object_set = {self._singular(tok) for tok in object_heads}
            if value_tokens.intersection(object_set):
                return True
            if predicate_norm in enumerative_predicates and subject_tokens.intersection(object_set):
                return True
            return False
        return predicate_norm in enumerative_predicates

    def _solve_count(
        self,
        *,
        query: str,
        subgraph: Dict[str, object],
    ) -> Dict[str, object]:
        claims = list(subgraph.get("claims", []))
        object_heads = list(subgraph.get("object_heads", []))
        if not claims:
            return self._abstain_payload(intent="count", reason="insufficient_subgraph", subgraph=subgraph)

        explicit_counts: List[str] = []
        enumerated_items: List[str] = []
        used_claim_ids: List[str] = []
        seen_values: set[str] = set()
        for claim in claims:
            text = self._claim_text(dict(claim))
            claim_tokens = {self._singular(tok) for tok in self._tokenize(text)}
            if object_heads and not any(head in claim_tokens for head in object_heads):
                predicate = self._normalize(str(claim.get("predicate", ""))).lower()
                if predicate not in {"count", "number", "total", "item", "member"}:
                    continue
            used_claim_ids.append(str(claim.get("claim_id", "")).strip())
            predicate = self._normalize(str(claim.get("predicate", ""))).lower()
            subject = self._normalize(str(claim.get("subject", "")))
            value = self._normalize(str(claim.get("value", ""))).strip(" ,.;:!?\"'")
            numeric_signals = self._extract_numeric_candidates(text)
            if predicate in {"count", "number", "total"} or re.fullmatch(r"\d+", value):
                explicit_counts.extend(numeric_signals or ([value] if value else []))
            if (
                value
                and not re.fullmatch(r"\d+", value)
                and len(value.split()) <= 8
                and self._is_count_item_value(
                    value=value,
                    predicate=predicate,
                    subject=subject,
                    object_heads=object_heads,
                )
            ):
                low = value.lower()
                if low not in seen_values:
                    seen_values.add(low)
                    enumerated_items.append(value)
        explicit_counts = list(dict.fromkeys([x for x in explicit_counts if x]))
        unique_counts = list(dict.fromkeys(explicit_counts))
        final_count = ""
        abstain_reason = ""
        confidence = 0.0

        if len(unique_counts) == 1:
            final_count = unique_counts[0]
            if enumerated_items and int(final_count) != len(enumerated_items):
                abstain_reason = "conflicting_count_signals"
                final_count = ""
            else:
                confidence = 0.74 if enumerated_items else 0.68
        elif len(unique_counts) > 1:
            abstain_reason = "conflicting_count_signals"
        elif enumerated_items:
            if len(enumerated_items) >= 2:
                final_count = str(len(enumerated_items))
                confidence = 0.66
            else:
                abstain_reason = "insufficient_object_list"
        else:
            abstain_reason = "missing_count_signal"

        structured = {
            "target_object": object_heads[0] if object_heads else "",
            "explicit_count_candidates": unique_counts[:4],
            "enumerated_items": enumerated_items[:8],
            "final_count": final_count,
        }
        if not final_count:
            return self._abstain_payload(
                intent="count",
                reason=abstain_reason or "insufficient_subgraph",
                subgraph=subgraph,
                structured_result=structured,
            )
        rationale = []
        if object_heads:
            rationale.append(f"count_object_type={object_heads[0]}")
        if unique_counts:
            rationale.append("count_explicit_candidates=" + " | ".join(unique_counts[:4]))
        if enumerated_items:
            rationale.append("count_graph_items=" + " | ".join(enumerated_items[:8]))
        return self._finalize_payload(
            intent="count",
            answer_candidate=final_count,
            confidence=confidence,
            used_claim_ids=used_claim_ids[:8],
            rationale_lines=rationale,
            structured_result=structured,
            subgraph=subgraph,
        )

    def _solve_temporal_count(
        self,
        *,
        query: str,
        subgraph: Dict[str, object],
    ) -> Dict[str, object]:
        claims = list(subgraph.get("claims", []))
        anchors: List[Tuple[datetime, str, str]] = []
        used_claim_ids: List[str] = []
        for claim in claims:
            text = self._claim_text(dict(claim))
            dates = self._parse_date(str(claim.get("time_anchor", ""))) or self._parse_date(text)
            if not dates:
                continue
            claim_id = str(claim.get("claim_id", "")).strip()
            if claim_id:
                used_claim_ids.append(claim_id)
            for dt in dates:
                anchors.append((dt, text, claim_id))
        if len(anchors) < 2:
            return self._abstain_payload(intent="temporal_count", reason="missing_dual_anchors", subgraph=subgraph)
        anchors.sort(key=lambda item: item[0])
        start_dt, start_text, _ = anchors[0]
        end_dt, end_text, _ = anchors[-1]
        delta_days = abs((end_dt - start_dt).days)
        duration_answer = self._format_duration(delta_days, query)
        structured = {
            "anchor_a": start_text,
            "anchor_b": end_text,
            "anchor_a_date": start_dt.strftime("%Y-%m-%d"),
            "anchor_b_date": end_dt.strftime("%Y-%m-%d"),
            "duration_days": delta_days,
            "duration_answer": duration_answer,
            "resolution_mode": "dual_anchor_duration",
        }
        rationale = [
            f"duration_answer={duration_answer}",
            f"temporal_points={start_dt.strftime('%Y-%m-%d')} | {end_dt.strftime('%Y-%m-%d')}",
        ]
        return self._finalize_payload(
            intent="temporal_count",
            answer_candidate=duration_answer,
            confidence=0.82,
            used_claim_ids=list(dict.fromkeys([x for x in used_claim_ids if x]))[:6],
            rationale_lines=rationale,
            structured_result=structured,
            subgraph=subgraph,
        )

    def _solve_temporal_compare(
        self,
        *,
        query: str,
        subgraph: Dict[str, object],
    ) -> Dict[str, object]:
        claims = list(subgraph.get("claims", []))
        edges = list(subgraph.get("edges", []))
        options = list(subgraph.get("options", []))
        if len(options) < 2:
            return self._abstain_payload(intent="temporal_compare", reason="missing_choice_targets", subgraph=subgraph)

        def _display_option(option: str) -> str:
            for item in claims:
                subject = self._normalize(str(item.get("subject", "")))
                if subject and subject.lower() == self._normalize(option).lower():
                    return subject
            return option

        def _pool(option: str) -> List[Dict[str, object]]:
            option_tokens = self._content_tokens(option)
            ranked = sorted(
                claims,
                key=lambda item: self._claim_overlap(item, option_tokens),
                reverse=True,
            )
            return [item for item in ranked if self._claim_overlap(item, option_tokens) >= 0.50][:4]

        left_pool = _pool(options[0])
        right_pool = _pool(options[1])
        if not left_pool or not right_pool:
            return self._abstain_payload(
                intent="temporal_compare",
                reason="missing_option_anchor",
                subgraph=subgraph,
                structured_result={
                    "option_a": _display_option(options[0]),
                    "option_b": _display_option(options[1]),
                    "option_a_pool": len(left_pool),
                    "option_b_pool": len(right_pool),
                },
            )

        left_ids = {str(item.get("node_id", "")).strip() for item in left_pool}
        right_ids = {str(item.get("node_id", "")).strip() for item in right_pool}
        for edge in edges:
            edge_type = str(edge.get("type", "")).strip()
            if edge_type not in {"before", "after"}:
                continue
            src = str(edge.get("source", "")).strip()
            dst = str(edge.get("target", "")).strip()
            if src in left_ids and dst in right_ids:
                answer = _display_option(options[0] if edge_type == "before" else options[1])
                if "later" in query.lower() or "after" in query.lower():
                    answer = _display_option(options[1] if edge_type == "before" else options[0])
                return self._finalize_payload(
                    intent="temporal_compare",
                    answer_candidate=answer,
                    confidence=0.86,
                    used_claim_ids=[str(item.get("claim_id", "")).strip() for item in left_pool + right_pool],
                    rationale_lines=[f"graph_edge={_display_option(options[0])} {edge_type} {_display_option(options[1])}"],
                    structured_result={
                        "option_a": _display_option(options[0]),
                        "option_b": _display_option(options[1]),
                        "comparison_result": answer,
                        "resolution_mode": "graph_edge",
                    },
                    subgraph=subgraph,
                )
            if src in right_ids and dst in left_ids:
                answer = _display_option(options[1] if edge_type == "before" else options[0])
                if "later" in query.lower() or "after" in query.lower():
                    answer = _display_option(options[0] if edge_type == "before" else options[1])
                return self._finalize_payload(
                    intent="temporal_compare",
                    answer_candidate=answer,
                    confidence=0.86,
                    used_claim_ids=[str(item.get("claim_id", "")).strip() for item in left_pool + right_pool],
                    rationale_lines=[f"graph_edge={_display_option(options[1])} {edge_type} {_display_option(options[0])}"],
                    structured_result={
                        "option_a": _display_option(options[0]),
                        "option_b": _display_option(options[1]),
                        "comparison_result": answer,
                        "resolution_mode": "graph_edge",
                    },
                    subgraph=subgraph,
                )

        def _best_date(pool: Sequence[Dict[str, object]]) -> datetime | None:
            out: List[datetime] = []
            for item in pool:
                dates = self._parse_date(str(item.get("time_anchor", ""))) or self._parse_date(self._claim_text(dict(item)))
                out.extend(dates)
            return sorted(out)[0] if out else None

        left_date = _best_date(left_pool)
        right_date = _best_date(right_pool)
        if left_date is None or right_date is None:
            return self._abstain_payload(intent="temporal_compare", reason="missing_comparison_dates", subgraph=subgraph)

        answer = _display_option(options[0] if left_date <= right_date else options[1])
        if "later" in query.lower() or "after" in query.lower():
            answer = _display_option(options[0] if left_date > right_date else options[1])
        return self._finalize_payload(
            intent="temporal_compare",
            answer_candidate=answer,
            confidence=0.78,
            used_claim_ids=[str(item.get("claim_id", "")).strip() for item in left_pool + right_pool],
            rationale_lines=[
                f"temporal_points={_display_option(options[0])}:{left_date.strftime('%Y-%m-%d')} | "
                f"{_display_option(options[1])}:{right_date.strftime('%Y-%m-%d')}"
            ],
            structured_result={
                "option_a": _display_option(options[0]),
                "option_b": _display_option(options[1]),
                "option_a_best_anchor": left_date.strftime("%Y-%m-%d"),
                "option_b_best_anchor": right_date.strftime("%Y-%m-%d"),
                "comparison_result": answer,
                "resolution_mode": "date_compare",
            },
            subgraph=subgraph,
        )

    def _solve_update(
        self,
        *,
        subgraph: Dict[str, object],
    ) -> Dict[str, object]:
        claims = list(subgraph.get("claims", []))
        edges = [edge for edge in list(subgraph.get("edges", [])) if str(edge.get("type", "")).strip() == "updates"]
        items_by_node = {str(item.get("node_id", "")).strip(): item for item in claims}
        if edges:
            ranked = sorted(
                edges,
                key=lambda edge: float(items_by_node.get(str(edge.get("target", "")), {}).get("_projection_score", 0.0)),
                reverse=True,
            )
            best = ranked[0]
            old_item = dict(items_by_node.get(str(best.get("source", "")), {}) or {})
            new_item = dict(items_by_node.get(str(best.get("target", "")), {}) or {})
            final_value = self._normalize(str(new_item.get("value", "")))
            if final_value:
                structured = {
                    "subject": self._normalize(str(new_item.get("subject", ""))),
                    "state_key": self._normalize(str(best.get("state_key", "")) or str(new_item.get("predicate", ""))),
                    "old_value": self._normalize(str(old_item.get("value", ""))),
                    "new_value": final_value,
                    "final_value": final_value,
                }
                return self._finalize_payload(
                    intent="update",
                    answer_candidate=final_value,
                    confidence=0.84,
                    used_claim_ids=[str(old_item.get("claim_id", "")).strip(), str(new_item.get("claim_id", "")).strip()],
                    rationale_lines=[f"state_update={self._claim_text(old_item)} -> {self._claim_text(new_item)}"],
                    structured_result={**structured, "resolution_mode": "update_edge"},
                    subgraph=subgraph,
                )

        ranked_claims = [
            item
            for item in sorted(claims, key=lambda x: float(x.get("_projection_score", 0.0)), reverse=True)
            if self._normalize(str(item.get("value", "")))
            and self._normalize(str(item.get("claim_type", ""))).lower() == "state_snapshot"
            and self._normalize(str(item.get("predicate", ""))).lower()
            not in {"preferred_direction", "supported_reason"}
        ]
        if not ranked_claims:
            return self._abstain_payload(intent="update", reason="missing_state_snapshot", subgraph=subgraph)
        best = ranked_claims[0]
        top_value = self._normalize(str(best.get("value", "")))
        if len(ranked_claims) > 1:
            second = ranked_claims[1]
            second_value = self._normalize(str(second.get("value", "")))
            if second_value and second_value.lower() != top_value.lower():
                top_score = float(best.get("_projection_score", 0.0))
                second_score = float(second.get("_projection_score", 0.0))
                if abs(top_score - second_score) < 0.15:
                    return self._abstain_payload(
                        intent="update",
                        reason="conflicting_state_values",
                        subgraph=subgraph,
                        structured_result={
                            "candidate_values": [top_value, second_value],
                            "state_key": self._normalize(str(best.get("predicate", ""))),
                        },
                    )
        structured = {
            "subject": self._normalize(str(best.get("subject", ""))),
            "state_key": self._normalize(str(best.get("predicate", ""))),
            "old_value": "",
            "new_value": top_value,
            "final_value": top_value,
            "resolution_mode": "state_snapshot",
        }
        return self._finalize_payload(
            intent="update",
            answer_candidate=top_value,
            confidence=0.70,
            used_claim_ids=[str(best.get("claim_id", "")).strip()],
            rationale_lines=[f"state_answer={self._claim_text(best)}"],
            structured_result=structured,
            subgraph=subgraph,
        )

    def _solve_preference(
        self,
        *,
        subgraph: Dict[str, object],
    ) -> Dict[str, object]:
        claims = list(subgraph.get("claims", []))
        direction = ""
        reason = ""
        support_items: List[str] = []
        used: List[str] = []
        for claim in claims:
            predicate = self._normalize(str(claim.get("predicate", ""))).lower()
            value = self._normalize(str(claim.get("value", "")))
            if not value:
                continue
            if predicate == "preferred_direction" and not direction:
                direction = value
                used.append(str(claim.get("claim_id", "")).strip())
            elif predicate == "supported_reason" and not reason:
                reason = value
                used.append(str(claim.get("claim_id", "")).strip())
            elif len(support_items) < 2 and float(claim.get("_projection_score", 0.0)) >= 0.65:
                support_items.append(value)
        if not direction and not reason:
            return self._abstain_payload(intent="preference", reason="missing_preference_summary", subgraph=subgraph)
        rationale: List[str] = []
        if direction:
            rationale.append("preference_direction=" + direction)
        if reason:
            rationale.append("preference_reason=" + reason)
        if support_items:
            rationale.append("preference_support=" + " | ".join(support_items[:2]))
        answer = f"Preference: {direction}" if direction else ""
        if answer and reason:
            answer += f" | Reason: {reason}"
        return self._finalize_payload(
            intent="preference",
            answer_candidate=answer,
            confidence=0.60 if direction else 0.0,
            used_claim_ids=[x for x in used if x],
            rationale_lines=rationale,
            structured_result={
                "preferred_direction": direction,
                "supported_reason": reason,
                "support_items": support_items[:2],
            },
            subgraph=subgraph,
        )

    def build_light_graph_tool_payload(
        self,
        *,
        query: str,
        light_graph: Dict[str, object],
    ) -> Dict[str, object]:
        graph = dict(light_graph or {})
        claims = self._graph_claim_items(graph)
        if not claims:
            return {}

        intent = self._classify_intent(query, graph)
        if intent == "generic":
            return {}

        subgraph = self._project_subgraph(
            query=query,
            light_graph=graph,
            claims=claims,
            intent=intent,
        )

        if intent == "count":
            payload = self._solve_count(query=query, subgraph=subgraph)
        elif intent == "temporal_count":
            payload = self._solve_temporal_count(query=query, subgraph=subgraph)
        elif intent == "temporal_compare":
            payload = self._solve_temporal_compare(query=query, subgraph=subgraph)
        elif intent == "update":
            payload = self._solve_update(subgraph=subgraph)
        elif intent == "preference":
            payload = self._solve_preference(subgraph=subgraph)
        else:
            payload = {}

        if not payload:
            return {}

        if not bool(payload.get("activated")):
            payload["summary_lines"] = []
            payload["summary_text"] = ""
        else:
            payload["summary_lines"] = [self._normalize(str(line)) for line in list(payload.get("summary_lines", [])) if self._normalize(str(line))][:6]
            payload["summary_text"] = "\n".join(payload["summary_lines"]).strip()
        return payload
