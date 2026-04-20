"""Unified specialist-layer orchestrator.

This layer coordinates optional specialist helpers (counting / graph toolkit)
and returns compact prompt hints plus an optional fallback answer clue.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


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
        graph_context: str,
        evidence_sentences: Sequence[Dict[str, object]],
        candidates: Sequence[Dict[str, object]],
        chunks: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        """Run enabled specialist modules and return a compact payload."""
        payload: Dict[str, object] = {
            "hints": "",
            "fallback_answer": "",
            "sources": [],
        }
        if not self.enabled:
            return payload

        hint_lines: List[str] = []
        fallback_answer = ""
        sources: List[str] = []

        if self.graph_toolkit_enabled:
            toolkit = getattr(self.m, "graph_toolkit", None)
            if toolkit is not None:
                try:
                    gh = str(
                        toolkit.build_tool_hints(
                            query=query,
                            graph_context=graph_context,
                            evidence_sentences=evidence_sentences,
                            candidates=candidates,
                            chunks=chunks,
                        )
                        or ""
                    ).strip()
                    ga = str(
                        toolkit.build_tool_answer(
                            query=query,
                            graph_context=graph_context,
                            evidence_sentences=evidence_sentences,
                            candidates=candidates,
                            chunks=chunks,
                        )
                        or ""
                    ).strip()
                    if gh:
                        hint_lines.extend([x for x in gh.splitlines() if x.strip()])
                    if self.allow_fallback_override and ga:
                        fallback_answer = ga
                    if gh or ga:
                        sources.append("graph_toolkit")
                except Exception:
                    # Keep specialist layer fail-safe; never break main answer path.
                    pass

        payload["hints"] = self._trim_hints("\n".join(hint_lines))
        payload["fallback_answer"] = str(fallback_answer).strip()
        payload["sources"] = sources
        return payload
