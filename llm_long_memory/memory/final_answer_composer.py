"""Final answer composer for the active MemSLM runtime.

This module turns structured intermediate artifacts into the only prompt that
the final 8B answering model sees:

filtered evidence + claims + light graph + toolkit
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple


class FinalAnswerComposer:
    """Compose the final answering prompt from structured pipeline outputs."""

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        cfg = dict(cfg or {})
        self.max_core = max(1, int(cfg.get("composer_max_core_evidence", 6)))
        self.max_supporting = max(0, int(cfg.get("composer_max_supporting_evidence", 4)))
        self.max_conflict = max(0, int(cfg.get("composer_max_conflict_evidence", 2)))
        self.max_claims = max(1, int(cfg.get("composer_max_claims", 6)))
        self.max_support_units = max(1, int(cfg.get("composer_max_support_units", 4)))
        self.max_graph_lines = max(1, int(cfg.get("composer_max_graph_lines", 6)))
        self.max_tool_lines = max(1, int(cfg.get("composer_max_tool_lines", 6)))

    @staticmethod
    def _normalize_space(text: str) -> str:
        return " ".join(str(text or "").split())

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text or "").lower())

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

        _append(list(filtered.get("core_evidence", [])), self.max_core, 0.30)
        _append(list(filtered.get("supporting_evidence", [])), self.max_supporting, 0.10)
        _append(list(filtered.get("conflict_evidence", [])), self.max_conflict, 0.05)
        if not out:
            _append(list(raw_fallback), min(8, len(list(raw_fallback))), 0.0)
        if out:
            out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return out

    def _format_filtered_pack(self, filtered_pack: Dict[str, object]) -> str:
        filtered = dict(filtered_pack or {})
        lines: List[str] = []
        for item in list(filtered.get("core_evidence", []))[: self.max_core]:
            text = self._normalize_space(str(item.get("text", "")))
            if text:
                lines.append(f"- core: {text}")
        for item in list(filtered.get("supporting_evidence", []))[: self.max_supporting]:
            text = self._normalize_space(str(item.get("text", "")))
            if text:
                lines.append(f"- support: {text}")
        for item in list(filtered.get("conflict_evidence", []))[: self.max_conflict]:
            text = self._normalize_space(str(item.get("text", "")))
            if text:
                lines.append(f"- conflict: {text}")
        return "[Filtered Evidence]\n" + "\n".join(lines) if lines else ""

    def _format_claim_result(self, claim_result: Dict[str, object]) -> str:
        payload = dict(claim_result or {})
        claims = list(payload.get("claims", []))
        support_units = list(payload.get("support_units", []))
        lines: List[str] = []
        for claim in claims[: self.max_claims]:
            text = self._normalize_space(self._claim_to_text(dict(claim)))
            if text:
                lines.append(f"- claim: {text}")
        if not lines:
            for unit in support_units[: self.max_support_units]:
                text = self._normalize_space(
                    str(unit.get("verbatim_span", "")) or str(unit.get("text", ""))
                )
                if text:
                    lines.append(f"- support_unit: {text}")
        return "[Graph Claims]\n" + "\n".join(lines) if lines else ""

    def _format_light_graph(self, light_graph: Dict[str, object]) -> str:
        graph = dict(light_graph or {})
        node_map = {str(node.get("id", "")): dict(node) for node in list(graph.get("nodes", []))}
        lines: List[str] = []
        for edge in list(graph.get("edges", [])):
            edge_type = str(edge.get("type", "")).strip()
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
            if len(lines) >= self.max_graph_lines:
                break
        return "[Light Graph]\n" + "\n".join(lines) if lines else ""

    def _format_toolkit(self, toolkit_payload: Dict[str, object]) -> str:
        payload = dict(toolkit_payload or {})
        tool_payload = dict(payload.get("tool_payload", {}) or {})
        lines: List[str] = []
        intent = self._normalize_space(str(tool_payload.get("intent", "")))
        if intent:
            lines.append(f"intent={intent}")
        for line in list(tool_payload.get("summary_lines", []))[: self.max_tool_lines]:
            text = self._normalize_space(str(line))
            if text:
                lines.append(text)
        answer_candidate = self._normalize_space(str(tool_payload.get("answer_candidate", "")))
        if answer_candidate:
            lines.append(f"tool_answer_candidate={answer_candidate}")
        return "[Toolkit Analysis]\n" + "\n".join(lines) if lines else ""

    def build_prompt(
        self,
        *,
        input_text: str,
        filtered_pack: Dict[str, object],
        claim_result: Dict[str, object],
        light_graph: Dict[str, object],
        toolkit_payload: Dict[str, object],
    ) -> Tuple[str, List[Dict[str, str]]]:
        sections: List[Dict[str, str]] = []
        filtered_text = self._format_filtered_pack(filtered_pack)
        claims_text = self._format_claim_result(claim_result)
        light_graph_text = self._format_light_graph(light_graph)
        toolkit_text = self._format_toolkit(toolkit_payload)

        for name, text in [
            ("filtered_evidence", filtered_text),
            ("graph_claims", claims_text),
            ("light_graph", light_graph_text),
            ("toolkit_output", toolkit_text),
        ]:
            if text:
                sections.append({"section": name, "text": text})

        rules = (
            "Use Toolkit Analysis first when it provides a grounded answer candidate.\n"
            "Then use Light Graph relations and Graph Claims for structure and resolution.\n"
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
