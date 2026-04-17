"""Tests for graph reasoning hint builder."""

from __future__ import annotations

import types
import unittest
from pathlib import Path

from llm_long_memory.memory.counting_resolver import CountingResolver
from llm_long_memory.memory.answering_candidate_extractor import AnswerCandidateExtractor
from llm_long_memory.memory.graph_reasoning_toolkit import GraphReasoningToolkit
from llm_long_memory.memory.long_memory_query_engine import LongMemoryQueryEngine
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
        cls.candidate_extractor = AnswerCandidateExtractor(dict(cfg["retrieval"]["answering"]))
        cls.query_engine = LongMemoryQueryEngine(types.SimpleNamespace())

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

    def test_build_tool_answer_count_generalizes_to_other_object_types(self) -> None:
        answer = self.toolkit.build_tool_answer(
            query="How many kitchen items do I have?",
            graph_context="[Long Memory Graph]\n- I have a bowl, a spoon, a plate, and a mug.",
            evidence_sentences=[
                {
                    "text": "I have a bowl, a spoon, a plate, and a mug.",
                    "score": 1.0,
                }
            ],
            candidates=[],
            chunks=[{"text": "I have a bowl, a spoon, a plate, and a mug."}],
        )
        self.assertEqual(answer, "4")

    def test_counting_resolver_counts_list_items_without_query_word_overlap(self) -> None:
        result = self.toolkit.counting.resolve(
            query="How many books do I own?",
            evidence=[
                {
                    "text": "I have a cookbook, a novel, and a notebook.",
                    "score": 1.0,
                }
            ],
            candidates=[],
            reranked_chunks=[
                {
                    "text": "I have a cookbook, a novel, and a notebook.",
                    "score": 1.0,
                    "session_date": "",
                }
            ],
        )
        self.assertIsNotNone(result)
        self.assertEqual(str(result.get("answer", "")).strip(), "3")

    def test_counting_entities_drop_discourse_fillers(self) -> None:
        entities = self.toolkit.counting._extract_list_entities(
            "By the way, speaking of my bikes, I've got three of them - a road bike, a mountain bike, and a commuter bike."
        )
        self.assertNotIn("by the way", entities)
        self.assertNotIn("speaking of my bikes", entities)
        self.assertIn("a mountain bike", entities)
        self.assertIn("a commuter bike", entities)

    def test_counting_entities_drop_aggregate_count_statements(self) -> None:
        entities = self.toolkit.counting._extract_list_entities(
            "I have three bikes: a road bike, a mountain bike, and a commuter bike."
        )
        self.assertNotIn("i have three bikes", entities)
        self.assertIn("a road bike", entities)
        self.assertIn("a mountain bike", entities)
        self.assertIn("a commuter bike", entities)

    def test_counting_entities_drop_generic_summary_clauses(self) -> None:
        entities = self.toolkit.counting._extract_list_entities(
            "We own two books, a notebook, and a pen."
        )
        self.assertNotIn("we own two books", entities)
        self.assertIn("a notebook", entities)
        self.assertIn("a pen", entities)

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
            query="What advice do you have for improving my workflow?",
            graph_context="[Long Memory Graph]\n- I prefer short checklists and time blocking to stay organized.",
            evidence_sentences=[
                {"text": "I prefer short checklists and time blocking to stay organized.", "score": 1.0}
            ],
            candidates=[{"text": "short checklists", "score": 1.0}],
            chunks=[{"text": "I prefer short checklists and time blocking to stay organized."}],
        )
        self.assertIn("intent=preference", hints)
        self.assertIn("preference_summary=", hints)
        self.assertIn("preference_hint=", hints)
        self.assertIn("reason:", hints)

    def test_classify_intent_does_not_over_trigger_temporal_compare(self) -> None:
        intent = self.toolkit._classify_intent(
            "What was the first issue I had with my new car after its first service?"
        )
        self.assertEqual(intent, "generic")

    def test_build_tool_hints_temporal_compare_requires_real_comparison(self) -> None:
        hints = self.toolkit.build_tool_hints(
            query="What was the first issue I had with my new car after its first service?",
            graph_context="[Long Memory Graph]\n- The first issue was the GPS system not functioning correctly.",
            evidence_sentences=[
                {"text": "The first issue was the GPS system not functioning correctly.", "score": 1.0}
            ],
            candidates=[],
            chunks=[{"text": "The first issue was the GPS system not functioning correctly."}],
        )
        self.assertNotIn("intent=temporal_compare", hints)
        self.assertNotIn("options=", hints)

    def test_extract_evidence_candidate_generic_copula_span(self) -> None:
        candidate = self.candidate_extractor.extract_evidence_candidate(
            "What was the first issue I had with my new car after its first service?",
            [{"text": "The first issue was the GPS system not functioning correctly.", "score": 1.0}],
            [],
        )
        self.assertIsNotNone(candidate)
        self.assertIn("GPS system not functioning correctly", str(candidate.get("answer", "")))

    def test_build_tool_answer_preference(self) -> None:
        answer = self.toolkit.build_tool_answer(
            query="What advice do you have for improving my workflow?",
            graph_context="[Long Memory Graph]\n- I prefer short checklists and time blocking to stay organized.",
            evidence_sentences=[
                {"text": "I prefer short checklists and time blocking to stay organized.", "score": 1.0}
            ],
            candidates=[{"text": "short checklists", "score": 1.0}],
            chunks=[{"text": "I prefer short checklists and time blocking to stay organized."}],
        )
        self.assertIn("Prefer", answer)
        self.assertIn("reason:", answer)

    def test_build_tool_answer_temporal_compare_generalizes(self) -> None:
        answer = self.toolkit.build_tool_answer(
            query="Who did I meet first, Alice and Bob or Charlie?",
            graph_context="[Long Memory Graph]\n- I met Alice and Bob on Monday.\n- I met Charlie on Wednesday.",
            evidence_sentences=[
                {"text": "I met Alice and Bob on Monday.", "score": 1.0},
                {"text": "I met Charlie on Wednesday.", "score": 0.9},
            ],
            candidates=[],
            chunks=[
                {"text": "I met Alice and Bob on Monday."},
                {"text": "I met Charlie on Wednesday."},
            ],
        )
        self.assertIn("Alice and Bob", answer)

    def test_build_tool_answer_temporal_count_generalizes(self) -> None:
        answer = self.toolkit.build_tool_answer(
            query="How many days passed between April 3 and April 17?",
            graph_context="[Long Memory Graph]\n- I started the project on April 3.\n- I finished on April 17.",
            evidence_sentences=[
                {"text": "I started the project on April 3.", "score": 1.0},
                {"text": "I finished on April 17.", "score": 0.9},
            ],
            candidates=[],
            chunks=[
                {"text": "I started the project on April 3."},
                {"text": "I finished on April 17."},
            ],
        )
        self.assertEqual(answer, "14 days")

    def test_query_engine_does_not_use_sample_specific_preference_cues(self) -> None:
        intent = self.query_engine._extract_query_intent(
            "Can you recommend some resources to improve my workflow?"
        )
        self.assertTrue(intent["asks_preference"])
        self.assertFalse(intent["asks_compare"])

    def test_query_engine_does_not_over_trigger_compare_on_first_issue(self) -> None:
        intent = self.query_engine._extract_query_intent(
            "What was the first issue I had with my new car?"
        )
        self.assertFalse(intent["asks_compare"])


if __name__ == "__main__":
    unittest.main()
