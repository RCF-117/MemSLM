"""Program-level routing for final answer generation.

This module turns structured intermediate artifacts into one of three runtime
answering modes:

- toolkit-first
- graph-first
- evidence-heavy

The router also chooses a prompt schema:

- source-direct: a single primary source in compact mode
- primary-plus-check: a primary source plus a tiny cross-check block
"""

from __future__ import annotations

from typing import Any, Dict


class FinalAnswerRouter:
    """Decide which answer-construction mode should drive final generation."""

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        cfg = dict(cfg or {})
        self.toolkit_verified_min_confidence = max(
            0.0,
            float(
                cfg.get(
                    "router_toolkit_verified_min_confidence",
                    cfg.get("composer_min_tool_confidence", 0.80),
                )
            ),
        )
        self.graph_min_top_support = max(
            0.0, float(cfg.get("router_graph_min_top_support", 0.45))
        )
        self.graph_min_supported_claims = max(
            1, int(cfg.get("router_graph_min_supported_claims", 2))
        )
        self.graph_min_structural_edges = max(
            1, int(cfg.get("router_graph_min_structural_edges", 1))
        )
        self.graph_min_top_support_strict = max(
            self.graph_min_top_support,
            float(cfg.get("router_graph_min_top_support_strict", 0.60)),
        )
        self.graph_min_supported_claims_strict = max(
            self.graph_min_supported_claims,
            int(cfg.get("router_graph_min_supported_claims_strict", 3)),
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    def _inspect_toolkit(self, toolkit_payload: Dict[str, object]) -> Dict[str, object]:
        payload = dict(toolkit_payload or {})
        tool_payload = dict(payload.get("tool_payload", {}) or {})
        activated = bool(tool_payload.get("activated", False))
        verified = bool(tool_payload.get("verified", False))
        confidence = float(tool_payload.get("confidence", 0.0) or 0.0)
        answer_candidate = self._normalize(
            str(tool_payload.get("verified_candidate", ""))
            or str(tool_payload.get("answer_candidate", ""))
        )
        used_claim_ids = [
            self._normalize(str(x))
            for x in list(
                tool_payload.get("verified_used_claim_ids", [])
                or tool_payload.get("used_claim_ids", [])
            )
            if self._normalize(str(x))
        ]
        eligible = bool(
            activated
            and verified
            and confidence >= self.toolkit_verified_min_confidence
            and answer_candidate
            and used_claim_ids
        )
        return {
            "eligible": eligible,
            "activated": activated,
            "verified": verified,
            "confidence": confidence,
            "answer_candidate": answer_candidate,
            "used_claim_ids": used_claim_ids,
        }

    def _inspect_graph(self, light_graph: Dict[str, object]) -> Dict[str, object]:
        graph = dict(light_graph or {})
        supported_claims = 0
        structural_edges = 0
        top_support = 0.0
        support_weights: Dict[str, float] = {}

        for edge in list(graph.get("edges", [])):
            edge_type = str(edge.get("type", "")).strip()
            if edge_type == "supports_query":
                src_id = str(edge.get("source", "")).strip()
                if not src_id:
                    continue
                try:
                    weight = float(edge.get("weight", 0.0) or 0.0)
                except Exception:
                    weight = 0.0
                support_weights[src_id] = weight
                top_support = max(top_support, weight)
                continue
            if edge_type in {"updates", "before", "after"}:
                structural_edges += 1

        for node in list(graph.get("nodes", [])):
            node_id = str(node.get("id", "")).strip()
            node_type = str(node.get("type", "")).strip().lower()
            if node_type not in {"fact", "state", "event"}:
                continue
            if float(support_weights.get(node_id, 0.0) or 0.0) >= self.graph_min_top_support:
                supported_claims += 1

        eligible = bool(
            structural_edges >= self.graph_min_structural_edges
            or (
                supported_claims >= self.graph_min_supported_claims
                and top_support >= self.graph_min_top_support
            )
        )
        strength = "weak"
        if structural_edges >= self.graph_min_structural_edges:
            strength = "strong"
        elif eligible:
            strength = "medium"
        return {
            "eligible": eligible,
            "strength": strength,
            "top_support": top_support,
            "supported_claims": supported_claims,
            "structural_edges": structural_edges,
        }

    def _inspect_filter(self, filtered_pack: Dict[str, object]) -> Dict[str, object]:
        filtered = dict(filtered_pack or {})
        core = list(filtered.get("core_evidence", []))
        supporting = list(filtered.get("supporting_evidence", []))
        conflict = list(filtered.get("conflict_evidence", []))
        combined = core + supporting + conflict

        def _looks_assistant_text(item: Dict[str, object]) -> bool:
            text = self._normalize(str(item.get("text", "")) or str(item.get("prompt_text", ""))).lower()
            if not text:
                return False
            return text.startswith("(assistant)") or text.startswith("assistant:")

        assistant_like_count = sum(1 for item in combined if _looks_assistant_text(dict(item)))
        total_count = len(combined)
        return {
            "core_count": len(core),
            "supporting_count": len(supporting),
            "conflict_count": len(conflict),
            "top_core_score": float(core[0].get("score", 0.0) or 0.0) if core else 0.0,
            "assistant_like_count": assistant_like_count,
            "assistant_like_ratio": (assistant_like_count / total_count) if total_count else 0.0,
        }

    def route(
        self,
        *,
        query: str,
        filtered_pack: Dict[str, object],
        claim_result: Dict[str, object],
        light_graph: Dict[str, object],
        toolkit_payload: Dict[str, object],
    ) -> Dict[str, object]:
        _ = query
        _ = claim_result
        toolkit_stats = self._inspect_toolkit(toolkit_payload)
        graph_stats = self._inspect_graph(light_graph)
        filter_stats = self._inspect_filter(filtered_pack)
        answer_type = self._normalize(str(dict(light_graph or {}).get("answer_type", ""))).lower()
        has_assistant_surface = bool(
            int(filter_stats.get("assistant_like_count", 0) or 0) > 0
            and float(filter_stats.get("assistant_like_ratio", 0.0) or 0.0) >= 0.50
        )
        structured_types = {"update", "count", "temporal", "temporal_comparison"}

        if bool(toolkit_stats["eligible"]):
            mode = "toolkit-first"
            reason = "verified_toolkit_candidate"
        else:
            graph_structural = int(graph_stats["structural_edges"]) >= self.graph_min_structural_edges
            graph_supported = (
                int(graph_stats["supported_claims"]) >= self.graph_min_supported_claims_strict
                and float(graph_stats["top_support"]) >= self.graph_min_top_support_strict
            )
            if answer_type == "preference":
                graph_eligible = False
            elif answer_type in {"temporal", "temporal_comparison"}:
                graph_eligible = graph_structural
            elif answer_type == "count":
                graph_eligible = graph_supported
            elif answer_type == "update":
                graph_eligible = graph_structural or graph_supported
            elif answer_type == "factoid":
                graph_eligible = graph_supported
            else:
                graph_eligible = bool(graph_stats["eligible"]) and (
                    graph_structural or graph_supported
                )

            if graph_eligible:
                mode = "graph-first"
                reason = "graph_has_strong_query_aligned_structure"
            else:
                mode = "evidence-heavy"
                reason = "fallback_to_filtered_evidence"

        prompt_schema = "source-direct"
        compact_sections = {
            "toolkit-first": ["toolkit_output", "answer_rules"],
            "graph-first": ["light_graph", "answer_rules"],
            "evidence-heavy": ["filtered_evidence", "answer_rules"],
        }[mode]
        expanded_sections = ["filtered_evidence", "answer_rules"]
        section_roles = {
            name: "primary"
            for name in compact_sections
            if name != "answer_rules"
        }
        expanded_section_roles = {"filtered_evidence": "primary"}
        if mode != "toolkit-first" and answer_type in structured_types:
            prompt_schema = "primary-plus-check"
            if mode == "graph-first":
                compact_sections = ["light_graph", "filtered_evidence", "answer_rules"]
                section_roles = {
                    "light_graph": "primary",
                    "filtered_evidence": "cross_check",
                }
            else:
                compact_sections = ["filtered_evidence", "light_graph", "answer_rules"]
                section_roles = {
                    "filtered_evidence": "primary",
                    "light_graph": "cross_check",
                }
            expanded_sections = ["filtered_evidence", "light_graph", "answer_rules"]
            expanded_section_roles = {
                "filtered_evidence": "primary",
                "light_graph": "cross_check",
            }
        presentation_mode = {
            "toolkit-first": "verdict-led",
            "graph-first": "structure-led",
            "evidence-heavy": "evidence-led",
        }[mode]
        evidence_present = bool(
            int(filter_stats["core_count"]) + int(filter_stats["supporting_count"]) > 0
            or bool(graph_stats["eligible"])
            or bool(toolkit_stats["eligible"])
        )
        abstention_policy = "standard"
        if evidence_present and (
            answer_type in structured_types
            or answer_type == "preference"
            or has_assistant_surface
        ):
            abstention_policy = "missing-only"

        return {
            "mode": mode,
            "reason": reason,
            "schema_sections": compact_sections,
            "compact_sections": compact_sections,
            "expanded_sections": expanded_sections,
            "section_roles": section_roles,
            "expanded_section_roles": expanded_section_roles,
            "presentation_mode": presentation_mode,
            "prompt_schema": prompt_schema,
            "abstention_policy": abstention_policy,
            "assistant_surface": has_assistant_surface,
            "primary_source": (
                "toolkit"
                if mode == "toolkit-first"
                else "light_graph"
                if mode == "graph-first"
                else "filtered_evidence"
            ),
            "toolkit": toolkit_stats,
            "graph": graph_stats,
            "filtered": filter_stats,
            "answer_type": answer_type,
        }

    def build_answer_rules(
        self, route_packet: Dict[str, object], *, prompt_mode: str = "compact"
    ) -> str:
        mode = str(route_packet.get("mode", "evidence-heavy")).strip().lower()
        prompt_mode = str(prompt_mode or "compact").strip().lower()
        prompt_schema = str(route_packet.get("prompt_schema", "source-direct") or "source-direct").strip().lower()
        answer_type = str(route_packet.get("answer_type", "") or "").strip().lower()
        abstention_policy = str(route_packet.get("abstention_policy", "standard") or "standard").strip().lower()
        assistant_surface = bool(route_packet.get("assistant_surface", False))

        def _missing_only_suffix() -> str:
            return (
                "Only answer Not found in retrieved context if the evidence does not address the question at all. "
                "Return only the final answer."
            )

        if prompt_mode == "expanded":
            if prompt_schema == "primary-plus-check":
                return (
                    "Use the Primary Evidence as the main source. "
                    "Use the Cross-check only to resolve count, ordering, latest/current, or before/after differences. "
                    "If both sections address the question, prefer the more specific or later-supported answer. "
                    + _missing_only_suffix()
                )
            if abstention_policy == "missing-only":
                if answer_type == "preference" or assistant_surface:
                    return (
                        "Use the filtered evidence below to synthesize the most specific answer it supports. "
                        "Do not give generic advice beyond the evidence. "
                        + _missing_only_suffix()
                    )
                return (
                    "Use only the filtered evidence below. "
                    "Answer directly from it whenever it addresses the question. "
                    + _missing_only_suffix()
                )
            return (
                "Use only the filtered evidence below. "
                "If it is insufficient, answer Not found in retrieved context. "
                "Return only the final answer."
            )
        if mode == "toolkit-first":
            return (
                "Use the toolkit result below as the answer source. "
                "Return only the final answer."
            )
        if mode == "graph-first":
            if prompt_schema == "primary-plus-check":
                return (
                    "Use the Primary Evidence as the answer source. "
                    "Use the Cross-check only to confirm count, ordering, or latest/current details. "
                    "If the cross-check is more specific, directly comparative, or clearly later in time, prefer it. "
                    + _missing_only_suffix()
                )
            return (
                "Use the light graph below as the answer source. "
                "Return only the final answer."
            )
        if prompt_schema == "primary-plus-check":
            return (
                "Use the Primary Evidence as the main source. "
                "Use the Cross-check only to resolve count, ordering, or latest/current differences. "
                "If the evidence addresses the question, answer from it instead of defaulting to Not found. "
                + _missing_only_suffix()
            )
        if abstention_policy == "missing-only":
            if answer_type == "preference" or assistant_surface:
                return (
                    "Use the filtered evidence below. "
                    "Synthesize the most specific answer supported by it, instead of giving generic advice. "
                    + _missing_only_suffix()
                )
            return (
                "Use the filtered evidence below. "
                "Answer directly from it whenever it addresses the question. "
                + _missing_only_suffix()
            )
        return (
            "Use the filtered evidence below. "
            "If it is insufficient, answer Not found in retrieved context. "
            "Return only the final answer."
        )
