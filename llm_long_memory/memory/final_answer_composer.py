"""Final answer composer for the active MemSLM runtime.

This module turns structured intermediate artifacts into the only prompt that
the final 8B answering model sees:

filtered evidence + light graph + toolkit

Claims remain an internal middle layer for graph/toolkit construction and are
not exposed as a first-class final-answer prompt section.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple


class FinalAnswerComposer:
    """Compose the final answering prompt from structured pipeline outputs."""

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        cfg = dict(cfg or {})
        self.default_prompt_mode = str(
            cfg.get("composer_default_prompt_mode", "compact")
        ).strip().lower() or "compact"
        self.expanded_limits = {
            "core": max(1, int(cfg.get("composer_max_core_evidence", 6))),
            "supporting": max(0, int(cfg.get("composer_max_supporting_evidence", 4))),
            "conflict": max(0, int(cfg.get("composer_max_conflict_evidence", 2))),
            "graph_lines": max(1, int(cfg.get("composer_max_graph_lines", 6))),
            "tool_lines": max(1, int(cfg.get("composer_max_tool_lines", 6))),
        }
        self.compact_limits = {
            "core": max(
                1,
                int(
                    cfg.get(
                        "composer_compact_max_core_evidence",
                        min(3, self.expanded_limits["core"]),
                    )
                ),
            ),
            "supporting": max(
                0,
                int(
                    cfg.get(
                        "composer_compact_max_supporting_evidence",
                        min(1, self.expanded_limits["supporting"]),
                    )
                ),
            ),
            "conflict": max(
                0,
                int(
                    cfg.get(
                        "composer_compact_max_conflict_evidence",
                        min(1, self.expanded_limits["conflict"]),
                    )
                ),
            ),
            "graph_lines": max(
                1,
                int(
                    cfg.get(
                        "composer_compact_max_graph_lines",
                        min(3, self.expanded_limits["graph_lines"]),
                    )
                ),
            ),
            "tool_lines": max(
                1,
                int(
                    cfg.get(
                        "composer_compact_max_tool_lines",
                        min(4, self.expanded_limits["tool_lines"]),
                    )
                ),
            ),
        }
        self.expanded_anchor_counts = {
            "core": max(
                0,
                int(cfg.get("composer_expanded_anchor_core_evidence", 1)),
            ),
            "supporting": max(
                0,
                int(cfg.get("composer_expanded_anchor_supporting_evidence", 0)),
            ),
            "conflict": max(
                0,
                int(cfg.get("composer_expanded_anchor_conflict_evidence", 0)),
            ),
            "graph_lines": max(
                0,
                int(cfg.get("composer_expanded_anchor_graph_lines", 1)),
            ),
        }
        self.expanded_backfill_compact = bool(
            cfg.get("composer_expanded_backfill_compact", False)
        )
        self.min_tool_confidence = max(0.0, float(cfg.get("composer_min_tool_confidence", 0.80)))
        self.min_graph_claim_support = max(
            0.0, float(cfg.get("composer_min_graph_claim_support", 0.20))
        )

    def _limits_for_mode(self, prompt_mode: str | None) -> Dict[str, int]:
        mode = str(prompt_mode or self.default_prompt_mode).strip().lower()
        if mode == "expanded":
            return dict(self.expanded_limits)
        return dict(self.compact_limits)

    @staticmethod
    def _normalize_space(text: str) -> str:
        return " ".join(str(text or "").split())

    def _item_prompt_text(self, item: Dict[str, object]) -> str:
        prompt_text = self._normalize_space(str(item.get("prompt_text", "")))
        if prompt_text:
            return prompt_text
        return self._normalize_space(str(item.get("text", "")))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text or "").lower())

    def _select_mode_items(
        self,
        items: Sequence[Dict[str, object]] | Sequence[str],
        *,
        prompt_mode: str | None,
        compact_limit: int,
        expanded_limit: int,
        anchor_count: int = 0,
    ) -> List[object]:
        values = list(items)
        if not values:
            return []
        mode = str(prompt_mode or self.default_prompt_mode).strip().lower()
        if mode != "expanded":
            return values[: max(0, int(compact_limit))]

        expanded_limit = max(0, int(expanded_limit))
        if expanded_limit <= 0:
            return []
        compact_limit = max(0, int(compact_limit))
        anchor_count = max(0, min(int(anchor_count), compact_limit, expanded_limit, len(values)))

        selected: List[object] = []
        selected_ids: set[int] = set()

        for idx in range(anchor_count):
            selected.append(values[idx])
            selected_ids.add(idx)

        next_start = min(len(values), compact_limit)
        for idx in range(next_start, len(values)):
            if len(selected) >= expanded_limit:
                break
            if idx in selected_ids:
                continue
            selected.append(values[idx])
            selected_ids.add(idx)

        if self.expanded_backfill_compact:
            for idx in range(anchor_count, min(compact_limit, len(values))):
                if len(selected) >= expanded_limit:
                    break
                if idx in selected_ids:
                    continue
                selected.append(values[idx])
                selected_ids.add(idx)

        return selected[:expanded_limit]

    def _claim_to_text(self, claim: Dict[str, Any]) -> str:
        subject = self._normalize_space(str(claim.get("subject", "")))
        predicate = self._normalize_space(str(claim.get("predicate", "")))
        value = self._normalize_space(str(claim.get("value", "")))
        parts: List[str] = []
        if subject:
            parts.append(subject)
        if predicate:
            parts.append(predicate)
        if value:
            parts.append(value)
        time_anchor = self._normalize_space(str(claim.get("time_anchor", "")))
        if time_anchor:
            parts.append(f"time={time_anchor}")
        status = self._normalize_space(str(claim.get("status", "")))
        if status and status != "unknown":
            parts.append(f"status={status}")
        return " | ".join(parts)

    def bundle_to_evidence_sentences(
        self,
        bundle: Dict[str, object],
        *,
        raw_fallback: Sequence[Dict[str, object]] = (),
    ) -> List[Dict[str, object]]:
        filtered = dict(bundle.get("filtered_pack", {}) or {})
        out: List[Dict[str, object]] = []
        seen: set[str] = set()

        def _append(items: Sequence[Dict[str, object]], limit: int, score_bias: float = 0.0) -> None:
            for item in list(items)[: max(0, int(limit))]:
                text = self._normalize_space(str(item.get("text", "")))
                if not text:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                base = float(item.get("score", 0.0) or 0.0)
                out.append(
                    {
                        "text": text,
                        "score": base + score_bias,
                        "chunk_id": int(item.get("chunk_id", 0) or 0),
                        "session_date": str(item.get("session_date", "")),
                        "channel": str(item.get("channel", "")),
                    }
                )

        limits = self._limits_for_mode("expanded")
        _append(list(filtered.get("core_evidence", [])), limits["core"], 0.30)
        _append(list(filtered.get("supporting_evidence", [])), limits["supporting"], 0.10)
        _append(list(filtered.get("conflict_evidence", [])), limits["conflict"], 0.05)
        if not out:
            _append(list(raw_fallback), min(8, len(list(raw_fallback))), 0.0)
        if out:
            out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return out

    def _format_filtered_pack(
        self,
        filtered_pack: Dict[str, object],
        *,
        limits: Dict[str, int],
        prompt_mode: str | None,
    ) -> str:
        filtered = dict(filtered_pack or {})
        lines: List[str] = []
        compact = self.compact_limits
        core_items = self._select_mode_items(
            list(filtered.get("core_evidence", [])),
            prompt_mode=prompt_mode,
            compact_limit=compact["core"],
            expanded_limit=limits["core"],
            anchor_count=self.expanded_anchor_counts["core"],
        )
        for item in core_items:
            text = self._item_prompt_text(dict(item))
            if text:
                lines.append(f"- core: {text}")
        supporting_items = self._select_mode_items(
            list(filtered.get("supporting_evidence", [])),
            prompt_mode=prompt_mode,
            compact_limit=compact["supporting"],
            expanded_limit=limits["supporting"],
            anchor_count=self.expanded_anchor_counts["supporting"],
        )
        for item in supporting_items:
            text = self._item_prompt_text(dict(item))
            if text:
                lines.append(f"- support: {text}")
        conflict_items = self._select_mode_items(
            list(filtered.get("conflict_evidence", [])),
            prompt_mode=prompt_mode,
            compact_limit=compact["conflict"],
            expanded_limit=limits["conflict"],
            anchor_count=self.expanded_anchor_counts["conflict"],
        )
        for item in conflict_items:
            text = self._item_prompt_text(dict(item))
            if text:
                lines.append(f"- conflict: {text}")
        return "[Filtered Evidence]\n" + "\n".join(lines) if lines else ""

    def _selected_filtered_support_sources(
        self,
        filtered_pack: Dict[str, object],
        *,
        limits: Dict[str, int],
        prompt_mode: str | None,
    ) -> List[Dict[str, object]]:
        filtered = dict(filtered_pack or {})
        support_sources: List[Dict[str, object]] = []
        selection_plan = [
            (
                "core_evidence",
                "filtered_evidence",
                "core",
                self.compact_limits["core"],
                limits["core"],
                self.expanded_anchor_counts["core"],
                0.90,
            ),
            (
                "supporting_evidence",
                "filtered_evidence",
                "support",
                self.compact_limits["supporting"],
                limits["supporting"],
                self.expanded_anchor_counts["supporting"],
                0.70,
            ),
            (
                "conflict_evidence",
                "filtered_evidence",
                "conflict",
                self.compact_limits["conflict"],
                limits["conflict"],
                self.expanded_anchor_counts["conflict"],
                0.60,
            ),
        ]
        for bucket_name, section, bucket, compact_limit, expanded_limit, anchor_count, default_score in selection_plan:
            items = self._select_mode_items(
                list(filtered.get(bucket_name, [])),
                prompt_mode=prompt_mode,
                compact_limit=compact_limit,
                expanded_limit=expanded_limit,
                anchor_count=anchor_count,
            )
            for item in items:
                text = self._item_prompt_text(dict(item))
                if not text:
                    continue
                score = float(item.get("score", default_score) or default_score)
                support_sources.append(
                    {
                        "text": text,
                        "score": score,
                        "section": section,
                        "bucket": bucket,
                    }
                )
        return support_sources

    def _selected_light_graph_support_sources(
        self,
        light_graph: Dict[str, object],
        *,
        limits: Dict[str, int],
        prompt_mode: str | None,
    ) -> List[Dict[str, object]]:
        graph = dict(light_graph or {})
        node_map = {str(node.get("id", "")): dict(node) for node in list(graph.get("nodes", []))}
        structural_lines: List[Tuple[float, str]] = []
        support_weights: Dict[str, float] = {}
        for edge in list(graph.get("edges", [])):
            edge_type = str(edge.get("type", "")).strip()
            if edge_type == "supports_query":
                src_id = str(edge.get("source", "")).strip()
                if src_id:
                    try:
                        support_weights[src_id] = float(edge.get("weight", 0.0) or 0.0)
                    except Exception:
                        support_weights[src_id] = 0.0
                continue
            if edge_type not in {"updates", "before", "after"}:
                continue
            src = dict(node_map.get(str(edge.get("source", "")), {}) or {})
            dst = dict(node_map.get(str(edge.get("target", "")), {}) or {})
            src_meta = dict(src.get("meta", {}) or {})
            dst_meta = dict(dst.get("meta", {}) or {})
            left = self._normalize_space(self._claim_to_text(src_meta))
            right = self._normalize_space(self._claim_to_text(dst_meta))
            if not left or not right:
                continue
            if edge_type == "updates":
                state_key = self._normalize_space(str(edge.get("state_key", "")))
                prefix = f"update[{state_key}]" if state_key else "update"
                structural_lines.append((0.95, f"- {prefix}: {left} -> {right}"))
            else:
                structural_lines.append((0.90, f"- {edge_type}: {left} -> {right}"))

        selected_texts: List[str] = []
        selected_scores: Dict[str, float] = {}
        if structural_lines:
            lines = [line for _score, line in structural_lines]
            selected = self._select_mode_items(
                lines,
                prompt_mode=prompt_mode,
                compact_limit=self.compact_limits["graph_lines"],
                expanded_limit=limits["graph_lines"],
                anchor_count=self.expanded_anchor_counts["graph_lines"],
            )
            selected_texts = [str(x) for x in selected]
            score_lookup = {line: score for score, line in structural_lines}
            selected_scores = {line: float(score_lookup.get(line, 0.90)) for line in selected_texts}
        else:
            fallback_claims: List[Tuple[float, str]] = []
            for node in list(graph.get("nodes", [])):
                node_id = str(node.get("id", "")).strip()
                node_type = str(node.get("type", "")).strip().lower()
                if node_type not in {"fact", "state", "event"}:
                    continue
                weight = float(support_weights.get(node_id, 0.0) or 0.0)
                if weight < self.min_graph_claim_support:
                    continue
                meta = dict(node.get("meta", {}) or {})
                text = self._normalize_space(self._claim_to_text(meta))
                if not text:
                    continue
                fallback_claims.append((weight, f"- claim[{node_type}]: {text}"))
            fallback_claims.sort(key=lambda item: item[0], reverse=True)
            deduped_lines: List[str] = []
            line_scores: Dict[str, float] = {}
            seen = set()
            for weight, line in fallback_claims:
                key = line.lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped_lines.append(line)
                line_scores[line] = weight
            selected = self._select_mode_items(
                deduped_lines,
                prompt_mode=prompt_mode,
                compact_limit=self.compact_limits["graph_lines"],
                expanded_limit=limits["graph_lines"],
                anchor_count=self.expanded_anchor_counts["graph_lines"],
            )
            selected_texts = [str(x) for x in selected]
            selected_scores = {line: float(line_scores.get(line, self.min_graph_claim_support)) for line in selected_texts}

        return [
            {
                "text": line,
                "score": float(selected_scores.get(line, 0.5)),
                "section": "light_graph",
                "bucket": "graph",
            }
            for line in selected_texts
            if self._normalize_space(line)
        ]

    def _selected_toolkit_support_sources(
        self,
        toolkit_payload: Dict[str, object],
        *,
        limits: Dict[str, int],
    ) -> List[Dict[str, object]]:
        payload = dict(toolkit_payload or {})
        tool_payload = dict(payload.get("tool_payload", {}) or {})
        activated = bool(tool_payload.get("activated", False))
        verified = bool(tool_payload.get("verified", False))
        confidence = float(tool_payload.get("confidence", 0.0) or 0.0)
        used_claim_ids = [
            self._normalize_space(str(x))
            for x in list(tool_payload.get("verified_used_claim_ids", []) or tool_payload.get("used_claim_ids", []))
            if self._normalize_space(str(x))
        ]
        if not activated or not verified or not used_claim_ids:
            return []
        support_sources: List[Dict[str, object]] = []
        verification_reason = self._normalize_space(str(tool_payload.get("verification_reason", "")))
        answer_candidate = self._normalize_space(
            str(tool_payload.get("verified_candidate", "")) or str(tool_payload.get("answer_candidate", ""))
        )
        summary_lines = [
            self._normalize_space(str(line))
            for line in list(tool_payload.get("summary_lines", []))[: limits["tool_lines"]]
            if self._normalize_space(str(line))
        ]
        if answer_candidate:
            support_sources.append(
                {
                    "text": answer_candidate,
                    "score": confidence,
                    "section": "toolkit_output",
                    "bucket": "tool_answer",
                }
            )
        for line in summary_lines:
            support_sources.append(
                {
                    "text": line,
                    "score": confidence,
                    "section": "toolkit_output",
                    "bucket": "tool_summary",
                }
            )
        if verification_reason:
            support_sources.append(
                {
                    "text": verification_reason,
                    "score": confidence,
                    "section": "toolkit_output",
                    "bucket": "tool_verification",
                }
            )
        return support_sources

    def _format_light_graph(
        self,
        light_graph: Dict[str, object],
        *,
        limits: Dict[str, int],
        prompt_mode: str | None,
    ) -> str:
        graph = dict(light_graph or {})
        node_map = {str(node.get("id", "")): dict(node) for node in list(graph.get("nodes", []))}
        lines: List[str] = []
        support_weights: Dict[str, float] = {}
        for edge in list(graph.get("edges", [])):
            edge_type = str(edge.get("type", "")).strip()
            if edge_type == "supports_query":
                src_id = str(edge.get("source", "")).strip()
                if src_id:
                    try:
                        support_weights[src_id] = float(edge.get("weight", 0.0) or 0.0)
                    except Exception:
                        support_weights[src_id] = 0.0
                continue
            if edge_type not in {"updates", "before", "after"}:
                continue
            src = dict(node_map.get(str(edge.get("source", "")), {}) or {})
            dst = dict(node_map.get(str(edge.get("target", "")), {}) or {})
            src_meta = dict(src.get("meta", {}) or {})
            dst_meta = dict(dst.get("meta", {}) or {})
            left = self._normalize_space(self._claim_to_text(src_meta))
            right = self._normalize_space(self._claim_to_text(dst_meta))
            if not left or not right:
                continue
            if edge_type == "updates":
                state_key = self._normalize_space(str(edge.get("state_key", "")))
                prefix = f"update[{state_key}]" if state_key else "update"
                lines.append(f"- {prefix}: {left} -> {right}")
            else:
                lines.append(f"- {edge_type}: {left} -> {right}")
        if not lines:
            fallback_claims: List[Tuple[float, str]] = []
            for node in list(graph.get("nodes", [])):
                node_id = str(node.get("id", "")).strip()
                node_type = str(node.get("type", "")).strip().lower()
                if node_type not in {"fact", "state", "event"}:
                    continue
                weight = float(support_weights.get(node_id, 0.0) or 0.0)
                if weight < self.min_graph_claim_support:
                    continue
                meta = dict(node.get("meta", {}) or {})
                text = self._normalize_space(self._claim_to_text(meta))
                if not text:
                    continue
                fallback_claims.append((weight, f"- claim[{node_type}]: {text}"))
            if fallback_claims:
                fallback_claims.sort(key=lambda item: item[0], reverse=True)
                seen = set()
                deduped: List[str] = []
                for _weight, line in fallback_claims:
                    key = line.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(line)
                lines.extend(deduped)
        selected_lines = self._select_mode_items(
            lines,
            prompt_mode=prompt_mode,
            compact_limit=self.compact_limits["graph_lines"],
            expanded_limit=limits["graph_lines"],
            anchor_count=self.expanded_anchor_counts["graph_lines"],
        )
        return "[Light Graph]\n" + "\n".join([str(x) for x in selected_lines]) if selected_lines else ""

    def _format_toolkit(self, toolkit_payload: Dict[str, object], *, limits: Dict[str, int]) -> str:
        payload = dict(toolkit_payload or {})
        tool_payload = dict(payload.get("tool_payload", {}) or {})
        activated = bool(tool_payload.get("activated", False))
        verified = bool(tool_payload.get("verified", False))
        confidence = float(tool_payload.get("confidence", 0.0) or 0.0)
        used_claim_ids = [
            self._normalize_space(str(x))
            for x in list(tool_payload.get("verified_used_claim_ids", []) or tool_payload.get("used_claim_ids", []))
            if self._normalize_space(str(x))
        ]
        summary_lines = list(tool_payload.get("summary_lines", []))
        answer_candidate = self._normalize_space(
            str(tool_payload.get("verified_candidate", "")) or str(tool_payload.get("answer_candidate", ""))
        )
        if not activated:
            return ""
        if not verified:
            return ""
        if not used_claim_ids:
            return ""
        if not summary_lines and not answer_candidate:
            return ""
        lines: List[str] = []
        intent = self._normalize_space(str(tool_payload.get("intent", "")))
        if intent:
            lines.append(f"intent={intent}")
        verification_reason = self._normalize_space(str(tool_payload.get("verification_reason", "")))
        if verification_reason:
            lines.append(f"tool_verification={verification_reason}")
        lines.append(f"tool_confidence={confidence:.2f}")
        for line in summary_lines[: limits["tool_lines"]]:
            text = self._normalize_space(str(line))
            if text:
                lines.append(text)
        if answer_candidate:
            lines.append(f"tool_answer_candidate={answer_candidate}")
        return "[Toolkit Analysis]\n" + "\n".join(lines) if lines else ""

    def build_support_sources(
        self,
        *,
        filtered_pack: Dict[str, object],
        claim_result: Dict[str, object],
        light_graph: Dict[str, object],
        toolkit_payload: Dict[str, object],
        prompt_mode: str | None = None,
        route_packet: Dict[str, object] | None = None,
    ) -> List[Dict[str, object]]:
        limits = self._limits_for_mode(prompt_mode)
        _ = claim_result
        section_order = [
            str(x).strip()
            for x in list((route_packet or {}).get("schema_sections", []))
            if str(x).strip() and str(x).strip() != "answer_rules"
        ]
        if not section_order:
            section_order = ["toolkit_output", "light_graph", "filtered_evidence"]
        section_to_sources = {
            "toolkit_output": self._selected_toolkit_support_sources(
                toolkit_payload, limits=limits
            ),
            "light_graph": self._selected_light_graph_support_sources(
                light_graph,
                limits=limits,
                prompt_mode=prompt_mode,
            ),
            "filtered_evidence": self._selected_filtered_support_sources(
                filtered_pack,
                limits=limits,
                prompt_mode=prompt_mode,
            ),
        }
        support_sources: List[Dict[str, object]] = []
        for section_name in section_order:
            support_sources.extend(list(section_to_sources.get(section_name, [])))
        return support_sources

    def build_prompt(
        self,
        *,
        input_text: str,
        filtered_pack: Dict[str, object],
        claim_result: Dict[str, object],
        light_graph: Dict[str, object],
        toolkit_payload: Dict[str, object],
        prompt_mode: str | None = None,
        route_packet: Dict[str, object] | None = None,
        answer_rules_text: str | None = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        sections: List[Dict[str, str]] = []
        limits = self._limits_for_mode(prompt_mode)
        filtered_text = self._format_filtered_pack(
            filtered_pack,
            limits=limits,
            prompt_mode=prompt_mode,
        )
        light_graph_text = self._format_light_graph(
            light_graph,
            limits=limits,
            prompt_mode=prompt_mode,
        )
        toolkit_text = self._format_toolkit(toolkit_payload, limits=limits)
        _ = claim_result

        section_order = [
            str(x).strip()
            for x in list((route_packet or {}).get("schema_sections", []))
            if str(x).strip() and str(x).strip() != "answer_rules"
        ]
        if not section_order:
            section_order = ["toolkit_output", "light_graph", "filtered_evidence"]
        section_to_text = {
            "toolkit_output": toolkit_text,
            "light_graph": light_graph_text,
            "filtered_evidence": filtered_text,
        }
        for name in section_order:
            text = str(section_to_text.get(name, "") or "")
            if text:
                sections.append({"section": name, "text": text})

        rules = str(answer_rules_text or "").strip()
        if not rules:
            rules = (
                "Use Toolkit Analysis first when it provides a grounded answer candidate.\n"
                "Then use Light Graph relations for structure and resolution.\n"
                "Use Filtered Evidence to fill missing detail or resolve ambiguity.\n"
                "Do not use hidden fallback heuristics or raw noisy retrieval.\n"
                "If evidence conflicts, prefer explicit graph-grounded state updates or better query-aligned evidence.\n"
                "If the available evidence is still insufficient, answer Not found in retrieved context.\n"
                "Return only the final answer."
            )
        sections.append({"section": "answer_rules", "text": "[Answer Rules]\n" + rules})

        prompt_parts = [section["text"] for section in sections]
        prompt_parts.append(f"User: {input_text}")
        return "\n\n".join(prompt_parts), sections
