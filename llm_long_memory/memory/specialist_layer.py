"""Unified specialist-layer orchestrator.

This layer coordinates optional specialist helpers and returns compact
graph-grounded tool output for the final answer composer.
"""

from __future__ import annotations

from typing import Any, Dict, List


class SpecialistLayer:
    """Lightweight orchestrator for specialist modules."""

    def __init__(self, manager: Any, config: Dict[str, Any] | None = None) -> None:
        self.m = manager
        self.cfg = dict(config or {})
        self.enabled = bool(self.cfg.get("enabled", False))
        modules_raw = [str(x).strip().lower() for x in list(self.cfg.get("modules", []))]
        self.modules = [x for x in modules_raw if x]
        # Counting is merged into GraphReasoningToolkit; keep this flag only for compatibility.
        self.counting_enabled = bool(self.cfg.get("counting_enabled", True)) and (
            (not self.modules) or ("counting" in self.modules)
        )
        self.graph_toolkit_enabled = bool(self.cfg.get("graph_toolkit_enabled", True)) and (
            (not self.modules) or ("graph_toolkit" in self.modules)
        )
        self.min_confidence = float(self.cfg.get("min_confidence", 0.0))
        self.allow_fallback_override = bool(self.cfg.get("allow_fallback_override", False))
        self.max_hint_lines = max(0, int(self.cfg.get("max_hint_lines", 8)))

    def _trim_hints(self, text: str) -> str:
        lines = [str(x).strip() for x in str(text or "").splitlines() if str(x).strip()]
        if self.max_hint_lines > 0:
            lines = lines[: self.max_hint_lines]
        return "\n".join(lines).strip()

    def run(
        self,
        *,
        query: str,
        graph_bundle: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        """Run enabled specialist modules and return a compact payload."""
        payload: Dict[str, object] = {
            "hints": "",
            "fallback_answer": "",
            "sources": [],
            "tool_payload": {},
        }
        if not self.enabled:
            return payload

        hint_lines: List[str] = []
        fallback_answer = ""
        sources: List[str] = []
        tool_payload: Dict[str, object] = {}

        if self.graph_toolkit_enabled:
            toolkit = getattr(self.m, "graph_toolkit", None)
            if toolkit is not None:
                try:
                    light_graph = dict(dict(graph_bundle or {}).get("light_graph", {}) or {})
                    tool_payload = dict(
                        toolkit.build_light_graph_tool_payload(
                            query=query,
                            light_graph=light_graph,
                        )
                        or {}
                    )
                    gh = str(tool_payload.get("summary_text", "") or "").strip()
                    ga = str(tool_payload.get("answer_candidate", "") or "").strip()
                    if gh:
                        hint_lines.extend([x for x in gh.splitlines() if x.strip()])
                    # Final answering now consumes structured tool payload directly.
                    # Keep fallback_answer empty so toolkit does not bypass the composer.
                    _ = ga
                    if gh or ga:
                        sources.append("graph_toolkit")
                except Exception:
                    # Keep specialist layer fail-safe; never break main answer path.
                    pass

        payload["hints"] = self._trim_hints("\n".join(hint_lines))
        payload["fallback_answer"] = str(fallback_answer).strip()
        payload["sources"] = sources
        payload["tool_payload"] = tool_payload
        return payload
