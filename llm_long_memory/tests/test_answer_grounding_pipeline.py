"""Tests for the answer-grounding pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path

from llm_long_memory.utils.helpers import load_config
from llm_long_memory.memory.answer_grounding_pipeline import AnswerGroundingPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class TestAnswerGroundingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cfg = load_config(str(CONFIG_PATH))
        cls.pipeline = AnswerGroundingPipeline(cfg["retrieval"]["answering"])

    def test_collect_evidence_sentences_sorted(self) -> None:
        chunks = [
            {"text": "(user) Alice graduated in Business Administration.", "score": 0.4, "topic_id": "t1"},
            {"text": "(assistant) She moved to Boston in 2023.", "score": 0.8, "topic_id": "t2"},
        ]
        evidence = self.pipeline.collect_evidence_sentences("where did she move", chunks)
        self.assertTrue(len(evidence) > 0)
        self.assertGreaterEqual(float(evidence[0]["score"]), float(evidence[-1]["score"]))

    def test_extract_candidates_returns_ranked(self) -> None:
        evidence = [
            {"text": "She moved to Boston in 2023.", "score": 0.9, "topic_id": "t1"},
            {"text": "Her degree was Business Administration.", "score": 0.7, "topic_id": "t1"},
        ]
        candidates = self.pipeline.extract_candidates("where did she move", evidence)
        self.assertIsInstance(candidates, list)
        if candidates:
            self.assertIn("text", candidates[0])
            self.assertIn("score", candidates[0])

    def test_apply_response_guard_to_not_found(self) -> None:
        result = self.pipeline.apply_response_guard(
            response="Completely unrelated answer",
            evidence_sentences=[{"text": "She moved to Boston in 2023.", "score": 0.8}],
            candidates=[],
        )
        if self.pipeline.answer_context_only:
            self.assertEqual(result, "Not found in retrieved context.")
        else:
            self.assertEqual(result, "Completely unrelated answer")

    def test_build_second_pass_retry_prompt_uses_structured_language(self) -> None:
        prompt = self.pipeline.build_second_pass_retry_prompt(
            prompt_text="[Filtered Evidence]\n- She moved to Boston in 2023.",
            evidence_candidate={"answer": "Boston"},
        )
        self.assertIn("[Original Prompt]", prompt)
        self.assertIn("Toolkit Analysis", prompt)
        self.assertIn("Light Graph", prompt)
        self.assertIn("Filtered Evidence", prompt)
        self.assertNotIn("Graph Claims", prompt)
        self.assertIn("Preferred evidence candidate: Boston", prompt)

if __name__ == "__main__":
    unittest.main()
