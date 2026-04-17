"""Tests for graph reasoning hint builder."""

from __future__ import annotations

import types
import unittest
from pathlib import Path

from llm_long_memory.memory.counting_resolver import CountingResolver
from llm_long_memory.memory.graph_reasoning_toolkit import GraphReasoningToolkit
from llm_long_memory.utils.helpers import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class TestGraphReasoningToolkit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cfg = load_config(str(CONFIG_PATH))
        counting = CountingResolver(dict(cfg["retrieval"]["answering"].get("counting", {})))
        answering = types.SimpleNamespace(counting=counting)
        cls.manager = types.SimpleNamespace(
            answering=answering,
            graph_refiner_enabled=True,
            long_memory_enabled=True,
            offline_graph_build_enabled=True,
            graph_context_from_store_enabled=True,
        )
        cls.toolkit = GraphReasoningToolkit(cls.manager)

    def test_build_tool_hints_count(self) -> None:
        hints = self.toolkit.build_tool_hints(
            query="How many items did I buy?",
            graph_context="[Long Memory Graph]\n- I bought 4 items yesterday.",
            evidence_sentences=[
                {"text": "I bought a jacket, a shirt, a scarf, and a pair of boots yesterday.", "score": 1.0}
            ],
            candidates=[{"text": "jacket", "score": 1.0}],
            chunks=[{"text": "I bought a jacket, a shirt, a scarf, and a pair of boots yesterday."}],
        )
        self.assertIn("intent=count", hints)
        self.assertIn("count_items=", hints)
        self.assertTrue("count_hint=" in hints or "support=" in hints)

    def test_build_tool_answer_count(self) -> None:
        answer = self.toolkit.build_tool_answer(
            query="How many bikes do I own?",
            graph_context="[Long Memory Graph]\n- I've got three of them: a road bike, a mountain bike, and a commuter bike.",
            evidence_sentences=[
                {
                    "text": "I've got three of them: a road bike, a mountain bike, and a commuter bike.",
                    "score": 1.0,
                }
            ],
            candidates=[],
            chunks=[{"text": "I've got three of them: a road bike, a mountain bike, and a commuter bike."}],
        )
        self.assertEqual(answer, "3")

    def test_build_tool_answer_count_ignores_unrelated_numeric_noise(self) -> None:
        answer = self.toolkit.build_tool_answer(
            query="How many bikes do I own?",
            graph_context="[Long Memory Graph]\n- By the way, speaking of my bikes, I've got three of them - a road bike, a mountain bike, and a commuter bike - and I've been using them for different types of rides.",
            evidence_sentences=[
                {
                    "text": "By the way, speaking of my bikes, I've got three of them - a road bike, a mountain bike, and a commuter bike - and I've been using them for different types of rides.",
                    "score": 1.0,
                }
            ],
            candidates=[{"text": "The project has 4 sections.", "score": 0.9}],
            chunks=[
                {
                    "text": "By the way, speaking of my bikes, I've got three of them - a road bike, a mountain bike, and a commuter bike - and I've been using them for different types of rides."
                }
            ],
        )
        self.assertEqual(answer, "3")

    def test_build_tool_answer_count_keeps_object_list_completeness_without_exact_focus_overlap(self) -> None:
        answer = self.toolkit.build_tool_answer(
            query="How many things did I buy?",
            graph_context="[Long Memory Graph]\n- I bought a jacket, a shirt, a scarf, and a pair of boots yesterday.",
            evidence_sentences=[
                {
                    "text": "I bought a jacket, a shirt, a scarf, and a pair of boots yesterday.",
                    "score": 1.0,
                }
            ],
            candidates=[],
            chunks=[{"text": "I bought a jacket, a shirt, a scarf, and a pair of boots yesterday."}],
        )
        self.assertEqual(answer, "4")

    def test_counting_entities_drop_discourse_fillers(self) -> None:
        entities = self.toolkit.counting._extract_list_entities(
            "By the way, speaking of my bikes, I've got three of them - a road bike, a mountain bike, and a commuter bike."
        )
        self.assertNotIn("by the way", entities)
        self.assertNotIn("speaking of my bikes", entities)
        self.assertIn("a mountain bike", entities)
        self.assertIn("a commuter bike", entities)

    def test_build_tool_hints_temporal_count(self) -> None:
        hints = self.toolkit.build_tool_hints(
            query="How many weeks have I been accepted into the exchange program?",
            graph_context="[Long Memory Graph]\n- I was accepted on March 20.\n- I started orientation on March 27.",
            evidence_sentences=[
                {"text": "I was accepted on March 20.", "score": 1.0},
                {"text": "I started orientation on March 27.", "score": 0.8},
            ],
            candidates=[],
            chunks=[{"text": "I was accepted on March 20."}, {"text": "I started orientation on March 27."}],
        )
        self.assertIn("intent=temporal_count", hints)
        self.assertIn("duration_hint=", hints)
        self.assertIn("temporal_points=", hints)

    def test_build_tool_answer_temporal_count(self) -> None:
        answer = self.toolkit.build_tool_answer(
            query="How many weeks have I been accepted into the exchange program?",
            graph_context="[Long Memory Graph]\n- I was accepted on March 20.\n- I started orientation on March 27.",
            evidence_sentences=[
                {"text": "I was accepted on March 20.", "score": 1.0},
                {"text": "I started orientation on March 27.", "score": 0.8},
            ],
            candidates=[],
            chunks=[{"text": "I was accepted on March 20."}, {"text": "I started orientation on March 27."}],
        )
        self.assertEqual(answer, "1 week")

    def test_extract_temporal_dates_supports_bare_month_day(self) -> None:
        dates = self.toolkit._extract_temporal_dates(
            [
                "I was accepted on March 20.",
                "I started orientation on 3/27.",
            ]
        )
        self.assertIn("2000-03-20", dates)
        self.assertIn("2000-03-27", dates)

    def test_build_tool_hints_preference(self) -> None:
        hints = self.toolkit.build_tool_hints(
            query="Can you recommend some resources for video editing?",
            graph_context="[Long Memory Graph]\n- I prefer Adobe Premiere Pro for editing.",
            evidence_sentences=[{"text": "I prefer Adobe Premiere Pro for editing.", "score": 1.0}],
            candidates=[{"text": "Adobe Premiere Pro", "score": 1.0}],
            chunks=[{"text": "I prefer Adobe Premiere Pro for editing."}],
        )
        self.assertIn("intent=preference", hints)
        self.assertIn("preference_summary=", hints)
        self.assertIn("preference_hint=", hints)
        self.assertIn("resource_hint=", hints)
        self.assertIn("reason:", hints)

    def test_build_tool_answer_preference(self) -> None:
        answer = self.toolkit.build_tool_answer(
            query="Can you recommend some resources for video editing?",
            graph_context="[Long Memory Graph]\n- I prefer Adobe Premiere Pro for editing.",
            evidence_sentences=[{"text": "I prefer Adobe Premiere Pro for editing.", "score": 1.0}],
            candidates=[{"text": "Adobe Premiere Pro", "score": 1.0}],
            chunks=[{"text": "I prefer Adobe Premiere Pro for editing."}],
        )
        self.assertIn("Adobe Premiere Pro", answer)
        self.assertIn("Prefer", answer)
        self.assertIn("resource suggestions:", answer)
        self.assertIn("reason:", answer)


if __name__ == "__main__":
    unittest.main()
