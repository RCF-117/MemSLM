"""Graph-only toolkit for question-scoped light-graph reasoning."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_long_memory.memory.query_intent import extract_query_intent
from llm_long_memory.memory.temporal_query_utils import parse_choice_targets


class GraphReasoningToolkit:
    """Produce compact graph-grounded reasoning payloads.

    This toolkit intentionally consumes only the question-scoped light graph.
    It does not read raw retrieval text, graph_context strings, or candidate
    lists from the old answering chain.
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

    def _classify_intent(self, query: str, light_graph: Dict[str, object]) -> str:
        answer_type = str(dict(light_graph or {}).get("answer_type", "")).strip().lower()
        if answer_type == "temporal_count":
            return "temporal_count"
        if answer_type == "count":
            return "count"
        if answer_type == "temporal_comparison":
            return "temporal_compare"
        if answer_type == "update":
            return "update"
        if answer_type == "preference":
            return "preference"
        flags = extract_query_intent(query)
        if flags.get("asks_how_many") and (flags.get("asks_when") or "how long" in str(query or "").lower()):
            return "temporal_count"
        if flags.get("asks_how_many"):
            return "count"
        if flags.get("asks_compare") and len(parse_choice_targets(query, max_options=4, default_target_k=2)) >= 2:
            return "temporal_compare"
        if flags.get("asks_current") or flags.get("asks_where"):
            return "update"
        if flags.get("asks_preference"):
            return "preference"
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
        # Keep order, remove duplicates.
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

    def _graph_edges(self, light_graph: Dict[str, object], edge_type: str) -> List[Dict[str, object]]:
        return [
            dict(edge)
            for edge in list(dict(light_graph or {}).get("edges", []))
            if str(edge.get("type", "")).strip() == edge_type
        ]

    def _graph_count_payload(self, query: str, claims: Sequence[Dict[str, object]]) -> Dict[str, object]:
        query_heads = self._query_object_heads(query)
        if not query_heads:
            return {}
        explicit_numbers: List[str] = []
        enumerated_values: List[str] = []
        matched_ids: List[str] = []
        seen_values: set[str] = set()
        for claim in claims:
            text = self._claim_text(dict(claim))
            tokens = {self._singular(tok) for tok in self._tokenize(text)}
            if not tokens:
                continue
            if not any(head in tokens or any(head in token or token in head for token in tokens) for head in query_heads):
                continue
            claim_id = str(claim.get("claim_id", "")).strip()
            if claim_id:
                matched_ids.append(claim_id)
            for match in re.finditer(r"\b(\d+)\b", text):
                explicit_numbers.append(str(int(match.group(1))))
            spelled = self._extract_spelled_number(text)
            if spelled >= 0:
                explicit_numbers.append(str(spelled))
            value = self._normalize(str(claim.get("value", ""))).strip(" ,.;:!?\"'")
            if value and value.lower() not in seen_values and len(value.split()) <= 8:
                seen_values.add(value.lower())
                enumerated_values.append(value)
        explicit_numbers = list(dict.fromkeys(explicit_numbers))
        answer_candidate = ""
        if len(explicit_numbers) == 1:
            answer_candidate = explicit_numbers[0]
        elif enumerated_values:
            answer_candidate = str(len(enumerated_values))
        summary_lines: List[str] = []
        if query_heads:
            summary_lines.append(f"count_object_type={query_heads[0]}")
        if explicit_numbers:
            summary_lines.append("count_explicit_candidates=" + " | ".join(explicit_numbers[:4]))
        if enumerated_values:
            summary_lines.append("count_graph_items=" + " | ".join(enumerated_values[:8]))
        return {
            "intent": "count",
            "summary_lines": summary_lines,
            "answer_candidate": answer_candidate,
            "confidence": 0.78 if answer_candidate else 0.0,
            "used_claim_ids": matched_ids[:8],
        }

    def _graph_temporal_count_payload(self, query: str, claims: Sequence[Dict[str, object]]) -> Dict[str, object]:
        dates: List[datetime] = []
        used_claim_ids: List[str] = []
        for claim in claims:
            claim_dates = self._parse_date(str(claim.get("time_anchor", ""))) or self._parse_date(self._claim_text(dict(claim)))
            if not claim_dates:
                continue
            dates.extend(claim_dates)
            claim_id = str(claim.get("claim_id", "")).strip()
            if claim_id:
                used_claim_ids.append(claim_id)
        if len(dates) < 2:
            return {}
        dates = sorted(dates)
        delta_days = abs((dates[-1] - dates[0]).days)
        answer = self._format_duration(delta_days, query)
        return {
            "intent": "temporal_count",
            "summary_lines": [
                f"duration_answer={answer}",
                f"temporal_points={dates[0].strftime('%Y-%m-%d')} | {dates[-1].strftime('%Y-%m-%d')}",
            ],
            "answer_candidate": answer,
            "confidence": 0.82,
            "used_claim_ids": list(dict.fromkeys(used_claim_ids))[:6],
        }

    def _graph_temporal_compare_payload(
        self,
        *,
        query: str,
        claims: Sequence[Dict[str, object]],
        light_graph: Dict[str, object],
    ) -> Dict[str, object]:
        options = parse_choice_targets(query, max_options=4, default_target_k=2)
        if len(options) < 2:
            return {}
        items_by_node = {str(item.get("node_id", "")): item for item in claims}
        before_edges = self._graph_edges(light_graph, "before")
        after_edges = self._graph_edges(light_graph, "after")

        def _display_option(option: str) -> str:
            for item in claims:
                subject = self._normalize(str(item.get("subject", "")))
                if subject and self._normalize(subject).lower() == self._normalize(option).lower():
                    return subject
            return option

        def _matches(item: Dict[str, object], option: str) -> bool:
            text = self._claim_text(item).lower()
            option_tokens = set(self._tokenize(option))
            text_tokens = set(self._tokenize(text))
            if not option_tokens or not text_tokens:
                return False
            overlap = len(option_tokens.intersection(text_tokens)) / float(len(option_tokens))
            return overlap >= 0.66

        summary_lines: List[str] = []
        for edge in before_edges + after_edges:
            left = items_by_node.get(str(edge.get("source", "")), {})
            right = items_by_node.get(str(edge.get("target", "")), {})
            if not left or not right:
                continue
            if _matches(left, options[0]) and _matches(right, options[1]):
                edge_type = str(edge.get("type", ""))
                summary_lines.append(f"graph_edge={options[0]} {edge_type} {options[1]}")
                if edge_type == "before":
                    answer = options[0] if "later" not in query.lower() and "after" not in query.lower() else options[1]
                else:
                    answer = options[1] if "later" not in query.lower() and "after" not in query.lower() else options[0]
                answer = _display_option(answer)
                return {
                    "intent": "temporal_compare",
                    "summary_lines": summary_lines,
                    "answer_candidate": answer,
                    "confidence": 0.84,
                    "used_claim_ids": [],
                }

        option_dates: Dict[str, List[datetime]] = {options[0]: [], options[1]: []}
        used_claim_ids: List[str] = []
        for claim in claims:
            for option in options[:2]:
                if not _matches(claim, option):
                    continue
                dates = self._parse_date(str(claim.get("time_anchor", ""))) or self._parse_date(self._claim_text(dict(claim)))
                option_dates[option].extend(dates)
                claim_id = str(claim.get("claim_id", "")).strip()
                if claim_id:
                    used_claim_ids.append(claim_id)
        if option_dates[options[0]] and option_dates[options[1]]:
            left_best = sorted(option_dates[options[0]])[0]
            right_best = sorted(option_dates[options[1]])[0]
            answer = options[0] if left_best <= right_best else options[1]
            if "later" in query.lower() or "after" in query.lower():
                answer = options[0] if left_best > right_best else options[1]
            answer = _display_option(answer)
            return {
                "intent": "temporal_compare",
                "summary_lines": [
                    f"temporal_points={options[0]}:{left_best.strftime('%Y-%m-%d')} | {options[1]}:{right_best.strftime('%Y-%m-%d')}"
                ],
                "answer_candidate": answer,
                "confidence": 0.78,
                "used_claim_ids": list(dict.fromkeys(used_claim_ids))[:6],
            }
        return {}

    def _graph_update_payload(
        self,
        *,
        query: str,
        claims: Sequence[Dict[str, object]],
        light_graph: Dict[str, object],
    ) -> Dict[str, object]:
        items_by_node = {str(item.get("node_id", "")): item for item in claims}
        update_edges = self._graph_edges(light_graph, "updates")
        if update_edges:
            ranked = sorted(
                update_edges,
                key=lambda edge: float(items_by_node.get(str(edge.get("target", "")), {}).get("support_weight", 0.0)),
                reverse=True,
            )
            best_edge = ranked[0]
            old_item = items_by_node.get(str(best_edge.get("source", "")), {})
            new_item = items_by_node.get(str(best_edge.get("target", "")), {})
            answer_candidate = self._normalize(str(new_item.get("value", "")))
            if answer_candidate:
                return {
                    "intent": "update",
                    "summary_lines": [
                        f"state_update={self._claim_text(old_item)} -> {self._claim_text(new_item)}"
                    ],
                    "answer_candidate": answer_candidate,
                    "confidence": 0.82,
                    "used_claim_ids": [
                        str(old_item.get("claim_id", "")),
                        str(new_item.get("claim_id", "")),
                    ],
                }
        ranked_claims = sorted(
            claims,
            key=lambda item: (
                float(item.get("support_weight", 0.0)),
                float(item.get("confidence", 0.0)),
            ),
            reverse=True,
        )
        if not ranked_claims:
            return {}
        best = ranked_claims[0]
        answer_candidate = self._normalize(str(best.get("value", "")))
        if not answer_candidate:
            return {}
        return {
            "intent": "update",
            "summary_lines": [f"state_answer={self._claim_text(best)}"],
            "answer_candidate": answer_candidate,
            "confidence": 0.70,
            "used_claim_ids": [str(best.get("claim_id", ""))] if str(best.get("claim_id", "")) else [],
        }

    def _graph_preference_payload(self, claims: Sequence[Dict[str, object]]) -> Dict[str, object]:
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
                used.append(str(claim.get("claim_id", "")))
            elif predicate == "supported_reason" and not reason:
                reason = value
                used.append(str(claim.get("claim_id", "")))
            elif len(support_items) < 2 and float(claim.get("support_weight", 0.0)) >= 0.30:
                support_items.append(value)
        if not direction and not reason and not support_items:
            return {}
        summary_lines: List[str] = []
        if direction:
            summary_lines.append("preference_direction=" + direction)
        if reason:
            summary_lines.append("preference_reason=" + reason)
        if support_items:
            summary_lines.append("preference_support=" + " | ".join(support_items[:2]))
        answer_candidate = ""
        if direction:
            answer_candidate = f"Preference: {direction}" + (f" | Reason: {reason}" if reason else "")
        return {
            "intent": "preference",
            "summary_lines": summary_lines,
            "answer_candidate": answer_candidate,
            "confidence": 0.76 if direction else 0.0,
            "used_claim_ids": [claim_id for claim_id in used if claim_id],
        }

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
        if intent == "count":
            payload = self._graph_count_payload(query, claims)
        elif intent == "temporal_count":
            payload = self._graph_temporal_count_payload(query, claims)
        elif intent == "temporal_compare":
            payload = self._graph_temporal_compare_payload(query=query, claims=claims, light_graph=graph)
        elif intent == "update":
            payload = self._graph_update_payload(query=query, claims=claims, light_graph=graph)
        elif intent == "preference":
            payload = self._graph_preference_payload(claims)
        else:
            payload = {}
        if not payload:
            return {}
        summary_lines = [self._normalize(str(line)) for line in list(payload.get("summary_lines", [])) if self._normalize(str(line))]
        payload["summary_lines"] = summary_lines[:6]
        payload["summary_text"] = "\n".join(summary_lines[:6]).strip()
        return payload
