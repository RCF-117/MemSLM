"""Unit tests for specialist-layer orchestration."""

from __future__ import annotations

from types import SimpleNamespace

from llm_long_memory.memory.specialist_layer import SpecialistLayer


class _FakeGraphToolkit:
    def build_tool_hints(self, **kwargs):
        return "intent=count\ncount_hint=4"

    def build_tool_answer(self, **kwargs):
        return "4"


def _manager(graph_toolkit=None, counting=None):
    answering = SimpleNamespace(counting=counting)
    return SimpleNamespace(graph_toolkit=graph_toolkit, answering=answering)


def test_specialist_layer_disabled_returns_empty_payload():
    layer = SpecialistLayer(_manager(), {"enabled": False})
    payload = layer.run(
        query="how many bikes",
        graph_context="",
        evidence_sentences=[],
        candidates=[],
        chunks=[],
    )
    assert payload["hints"] == ""
    assert payload["fallback_answer"] == ""
    assert payload["sources"] == []


def test_specialist_layer_prefers_graph_fallback_when_allowed():
    layer = SpecialistLayer(
        _manager(graph_toolkit=_FakeGraphToolkit()),
        {
            "enabled": True,
            "modules": ["graph_toolkit"],
            "graph_toolkit_enabled": True,
            "allow_fallback_override": True,
            "max_hint_lines": 8,
        },
    )
    payload = layer.run(
        query="how many bikes",
        graph_context="",
        evidence_sentences=[],
        candidates=[],
        chunks=[],
    )
    assert "intent=count" in payload["hints"]
    assert payload["fallback_answer"] == "4"
    assert "graph_toolkit" in payload["sources"]


def test_specialist_layer_no_separate_counting_path():
    layer = SpecialistLayer(
        _manager(counting=SimpleNamespace(resolve=lambda **kwargs: {"answer": "3"})),
        {
            "enabled": True,
            "modules": ["counting"],
            "counting_enabled": True,
            "allow_fallback_override": True,
        },
    )
    payload = layer.run(
        query="how many bikes",
        graph_context="",
        evidence_sentences=[{"text": "I have three bikes.", "score": 1.0}],
        candidates=[],
        chunks=[],
    )
    assert payload["hints"] == ""
    assert payload["fallback_answer"] == ""
    assert payload["sources"] == []
