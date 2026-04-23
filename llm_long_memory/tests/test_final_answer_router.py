"""Tests for final answer routing modes."""

from __future__ import annotations

import unittest

from llm_long_memory.memory.final_answer_router import FinalAnswerRouter


class TestFinalAnswerRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.router = FinalAnswerRouter(
            {
                "router_toolkit_verified_min_confidence": 0.80,
                "router_graph_min_top_support": 0.45,
                "router_graph_min_supported_claims": 2,
                "router_graph_min_structural_edges": 1,
            }
        )

    def test_route_prefers_verified_toolkit(self) -> None:
        packet = self.router.route(
            query="Where is the painting now?",
            filtered_pack={"core_evidence": [{"text": "I moved it to the bedroom.", "score": 0.8}]},
            claim_result={},
            light_graph={"nodes": [], "edges": []},
            toolkit_payload={
                "tool_payload": {
                    "activated": True,
                    "verified": True,
                    "confidence": 0.91,
                    "verified_candidate": "bedroom",
                    "verified_used_claim_ids": ["c1"],
                }
            },
        )
        self.assertEqual(packet["mode"], "toolkit-first")
        self.assertEqual(packet["primary_source"], "toolkit")

    def test_route_uses_graph_first_when_update_graph_is_structural(self) -> None:
        packet = self.router.route(
            query="Where is the painting now?",
            filtered_pack={"core_evidence": [{"text": "I moved it to the bedroom.", "score": 0.8}]},
            claim_result={},
            light_graph={
                "answer_type": "update",
                "nodes": [
                    {"id": "old", "type": "state", "meta": {"subject": "painting", "predicate": "location", "value": "living room"}},
                    {"id": "new", "type": "state", "meta": {"subject": "painting", "predicate": "location", "value": "bedroom"}},
                ],
                "edges": [{"source": "old", "target": "new", "type": "updates", "state_key": "location"}],
            },
            toolkit_payload={"tool_payload": {"activated": False}},
        )
        self.assertEqual(packet["mode"], "graph-first")
        self.assertEqual(packet["primary_source"], "light_graph")
        self.assertEqual(packet["presentation_mode"], "structure-led")
        self.assertEqual(
            packet["compact_sections"],
            ["light_graph", "filtered_evidence", "answer_rules"],
        )
        self.assertEqual(packet["expanded_sections"], ["filtered_evidence", "light_graph", "answer_rules"])
        self.assertEqual(packet["section_roles"]["light_graph"], "primary")
        self.assertEqual(packet["section_roles"]["filtered_evidence"], "cross_check")

    def test_route_keeps_preference_out_of_graph_first(self) -> None:
        packet = self.router.route(
            query="Can you recommend some resources for video editing?",
            filtered_pack={"core_evidence": [{"text": "You asked about learning resources.", "score": 0.8}]},
            claim_result={},
            light_graph={
                "answer_type": "preference",
                "nodes": [
                    {"id": "claim_a", "type": "fact", "meta": {"subject": "resource", "predicate": "type", "value": "video tutorials"}},
                    {"id": "query_root", "type": "query", "meta": {}},
                ],
                "edges": [
                    {"source": "claim_a", "target": "query_root", "type": "supports_query", "weight": 0.95},
                ],
            },
            toolkit_payload={"tool_payload": {"activated": False}},
        )
        self.assertEqual(packet["mode"], "evidence-heavy")

    def test_route_keeps_temporal_without_structure_out_of_graph_first(self) -> None:
        packet = self.router.route(
            query="Who did I meet first, Mark and Sarah or Tom?",
            filtered_pack={"core_evidence": [{"text": "I met Tom in January.", "score": 0.8}]},
            claim_result={},
            light_graph={
                "answer_type": "temporal_comparison",
                "nodes": [
                    {"id": "claim_a", "type": "event", "meta": {"subject": "I", "predicate": "met", "value": "Tom"}},
                    {"id": "query_root", "type": "query", "meta": {}},
                ],
                "edges": [
                    {"source": "claim_a", "target": "query_root", "type": "supports_query", "weight": 0.95},
                ],
            },
            toolkit_payload={"tool_payload": {"activated": False}},
        )
        self.assertEqual(packet["mode"], "evidence-heavy")

    def test_route_falls_back_to_evidence_heavy_when_toolkit_and_graph_are_weak(self) -> None:
        packet = self.router.route(
            query="Where is the painting now?",
            filtered_pack={"core_evidence": [{"text": "I moved it to the bedroom.", "score": 0.8}]},
            claim_result={},
            light_graph={"nodes": [], "edges": []},
            toolkit_payload={"tool_payload": {"activated": False}},
        )
        self.assertEqual(packet["mode"], "evidence-heavy")
        self.assertEqual(packet["primary_source"], "filtered_evidence")
        self.assertEqual(packet["presentation_mode"], "evidence-led")
        self.assertEqual(packet["compact_sections"], ["filtered_evidence", "light_graph", "answer_rules"])
        self.assertEqual(packet["expanded_sections"], ["filtered_evidence", "light_graph", "answer_rules"])


if __name__ == "__main__":
    unittest.main()
