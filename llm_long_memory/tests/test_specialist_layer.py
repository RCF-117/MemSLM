"""Unit tests for the active specialist-layer orchestration."""

from __future__ import annotations

from types import SimpleNamespace

from llm_long_memory.memory.specialist_layer import SpecialistLayer


class _FakeGraphToolkit:
    def build_light_graph_tool_payload(self, *, query, light_graph):
        _ = query
        _ = light_graph
        return {
            "intent": "count",
            "summary_lines": ["count_object_type=bike", "count_graph_items=road bike | commuter bike"],
            "summary_text": "count_object_type=bike\ncount_graph_items=road bike | commuter bike",
            "answer_candidate": "2",
            "confidence": 0.8,
            "used_claim_ids": ["c1", "c2"],
        }


def _manager(graph_toolkit=None):
    return SimpleNamespace(graph_toolkit=graph_toolkit)


def test_specialist_layer_disabled_returns_empty_payload():
    layer = SpecialistLayer(_manager(), {"enabled": False})
    payload = layer.run(query="how many bikes", graph_bundle={})
    assert payload["hints"] == ""
    assert payload["sources"] == []
    assert payload["tool_payload"] == {}


def test_specialist_layer_returns_graph_only_tool_payload():
    layer = SpecialistLayer(
        _manager(graph_toolkit=_FakeGraphToolkit()),
        {
            "enabled": True,
            "modules": ["graph_toolkit"],
            "graph_toolkit_enabled": True,
            "max_hint_lines": 8,
        },
    )
    payload = layer.run(
        query="how many bikes do I own?",
        graph_bundle={"light_graph": {"nodes": [], "edges": []}},
    )
    assert "count_object_type=bike" in payload["hints"]
    assert payload["tool_payload"]["answer_candidate"] == "2"
    assert payload["sources"] == ["graph_toolkit"]
    assert payload["fallback_answer"] == ""


def test_specialist_layer_is_fail_safe_when_toolkit_raises():
    class _BrokenToolkit:
        def build_light_graph_tool_payload(self, *, query, light_graph):
            _ = query
            _ = light_graph
            raise RuntimeError("boom")

    layer = SpecialistLayer(
        _manager(graph_toolkit=_BrokenToolkit()),
        {"enabled": True, "modules": ["graph_toolkit"], "graph_toolkit_enabled": True},
    )
    payload = layer.run(
        query="what changed?",
        graph_bundle={"light_graph": {"nodes": [], "edges": []}},
    )
    assert payload["hints"] == ""
    assert payload["tool_payload"] == {}
    assert payload["sources"] == []
