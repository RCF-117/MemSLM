"""Tests for the graph-only reasoning toolkit."""

from __future__ import annotations

import types
import unittest

from llm_long_memory.memory.graph_reasoning_toolkit import GraphReasoningToolkit
from llm_long_memory.memory.query_intent import extract_query_intent


def _claim_node(
    node_id: str,
    *,
    claim_type: str,
    subject: str,
    predicate: str,
    value: str,
    time_anchor: str = "",
    confidence: float = 0.8,
    support_weight: float = 0.7,
) -> dict:
    return {
        "id": node_id,
        "type": {"fact_statement": "fact", "state_snapshot": "state", "event_record": "event"}[claim_type],
        "label": f"{subject} | {predicate} | {value}",
        "meta": {
            "claim_id": node_id,
            "claim_type": claim_type,
            "subject": subject,
            "predicate": predicate,
            "value": value,
            "time_anchor": time_anchor,
            "confidence": confidence,
            "support_weight": support_weight,
        },
    }


def _supports_query(node_id: str, weight: float = 0.8) -> dict:
    return {"source": node_id, "target": "query", "type": "supports_query", "weight": weight}


class TestGraphReasoningToolkit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manager = types.SimpleNamespace()
        cls.toolkit = GraphReasoningToolkit(cls.manager)

    def test_build_light_graph_tool_payload_count(self) -> None:
        graph = {
            "answer_type": "count",
            "nodes": [
                {"id": "query", "type": "query", "label": "Q1", "meta": {}},
                _claim_node("c1", claim_type="fact_statement", subject="bike collection", predicate="count", value="3"),
                _claim_node("c2", claim_type="fact_statement", subject="bike collection", predicate="item", value="road bike"),
                _claim_node("c3", claim_type="fact_statement", subject="bike collection", predicate="item", value="mountain bike"),
                _claim_node("c4", claim_type="fact_statement", subject="bike collection", predicate="item", value="commuter bike"),
            ],
            "edges": [
                _supports_query("c1", 0.9),
                _supports_query("c2", 0.8),
                _supports_query("c3", 0.8),
                _supports_query("c4", 0.8),
            ],
        }
        payload = self.toolkit.build_light_graph_tool_payload(
            query="How many bikes do I own?",
            light_graph=graph,
        )
        self.assertEqual(payload["intent"], "count")
        self.assertEqual(payload["answer_candidate"], "3")
        self.assertTrue(any("count_graph_items=" in line for line in payload["summary_lines"]))

    def test_build_light_graph_tool_payload_temporal_count(self) -> None:
        graph = {
            "answer_type": "temporal_count",
            "nodes": [
                {"id": "query", "type": "query", "label": "Q2", "meta": {}},
                _claim_node(
                    "c1",
                    claim_type="event_record",
                    subject="exchange program",
                    predicate="accepted",
                    value="accepted",
                    time_anchor="March 20",
                ),
                _claim_node(
                    "c2",
                    claim_type="event_record",
                    subject="exchange program",
                    predicate="orientation",
                    value="started",
                    time_anchor="March 27",
                ),
            ],
            "edges": [_supports_query("c1"), _supports_query("c2")],
        }
        payload = self.toolkit.build_light_graph_tool_payload(
            query="How many weeks passed between acceptance and orientation?",
            light_graph=graph,
        )
        self.assertEqual(payload["intent"], "temporal_count")
        self.assertEqual(payload["answer_candidate"], "1 week")

    def test_build_light_graph_tool_payload_temporal_compare(self) -> None:
        graph = {
            "answer_type": "temporal_comparison",
            "nodes": [
                {"id": "query", "type": "query", "label": "Q3", "meta": {}},
                _claim_node(
                    "a",
                    claim_type="event_record",
                    subject="Mark and Sarah",
                    predicate="met",
                    value="a month ago",
                    time_anchor="2026-03-01",
                ),
                _claim_node(
                    "b",
                    claim_type="event_record",
                    subject="Tom",
                    predicate="met",
                    value="a few months ago",
                    time_anchor="2026-01-01",
                ),
            ],
            "edges": [
                _supports_query("a", 0.8),
                _supports_query("b", 0.8),
                {"source": "b", "target": "a", "type": "before", "weight": 0.9},
            ],
        }
        payload = self.toolkit.build_light_graph_tool_payload(
            query="Who did I meet first, Mark and Sarah or Tom?",
            light_graph=graph,
        )
        self.assertEqual(payload["intent"], "temporal_compare")
        self.assertEqual(payload["answer_candidate"], "Tom")

    def test_build_light_graph_tool_payload_update(self) -> None:
        graph = {
            "answer_type": "update",
            "nodes": [
                {"id": "query", "type": "query", "label": "Q4", "meta": {}},
                _claim_node(
                    "old",
                    claim_type="state_snapshot",
                    subject="Ethereal Dreams painting",
                    predicate="location",
                    value="living room sofa",
                    time_anchor="before",
                ),
                _claim_node(
                    "new",
                    claim_type="state_snapshot",
                    subject="Ethereal Dreams painting",
                    predicate="location",
                    value="bedroom",
                    time_anchor="latest",
                ),
            ],
            "edges": [
                _supports_query("old", 0.6),
                _supports_query("new", 0.9),
                {"source": "old", "target": "new", "type": "updates", "weight": 0.9, "state_key": "location"},
            ],
        }
        payload = self.toolkit.build_light_graph_tool_payload(
            query="Where is the Ethereal Dreams painting now?",
            light_graph=graph,
        )
        self.assertEqual(payload["intent"], "update")
        self.assertEqual(payload["answer_candidate"], "bedroom")

    def test_build_light_graph_tool_payload_preference(self) -> None:
        graph = {
            "answer_type": "preference",
            "nodes": [
                {"id": "query", "type": "query", "label": "Q5", "meta": {}},
                _claim_node(
                    "c1",
                    claim_type="fact_statement",
                    subject="workflow advice",
                    predicate="preferred_direction",
                    value="short checklists and time blocking",
                ),
                _claim_node(
                    "c2",
                    claim_type="fact_statement",
                    subject="workflow advice",
                    predicate="supported_reason",
                    value="stays organized with compact, repeatable routines",
                ),
            ],
            "edges": [_supports_query("c1", 0.9), _supports_query("c2", 0.7)],
        }
        payload = self.toolkit.build_light_graph_tool_payload(
            query="What advice fits my workflow preference?",
            light_graph=graph,
        )
        self.assertEqual(payload["intent"], "preference")
        self.assertIn("Preference:", payload["answer_candidate"])
        self.assertTrue(any("preference_direction=" in line for line in payload["summary_lines"]))

    def test_query_intent_still_does_not_overtrigger_compare(self) -> None:
        intent = extract_query_intent("What was the first issue I had with my new car?")
        self.assertFalse(intent["asks_compare"])


if __name__ == "__main__":
    unittest.main()
