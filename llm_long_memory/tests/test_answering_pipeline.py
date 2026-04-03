"""Tests for answering decision pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path

from llm_long_memory.utils.helpers import load_config
from llm_long_memory.memory.answering_pipeline import AnsweringPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class TestAnsweringPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cfg = load_config(str(CONFIG_PATH))
        cls.pipeline = AnsweringPipeline(cfg["retrieval"]["answering"])

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

    def test_short_circuit_disabled_returns_none(self) -> None:
        # Config currently disables short-circuit by default.
        out = self.pipeline.maybe_short_circuit(
            candidates=[{"text": "Boston", "score": 0.9}],
            evidence_sentences=[{"text": "She moved to Boston.", "score": 0.95}],
        )
        self.assertIsNone(out)

    def test_apply_response_fallback_to_not_found(self) -> None:
        result = self.pipeline.apply_response_fallback(
            response="Completely unrelated answer",
            evidence_sentences=[{"text": "She moved to Boston in 2023.", "score": 0.8}],
            candidates=[],
        )
        if self.pipeline.answer_context_only:
            self.assertEqual(result, "Not found in retrieved context.")
        else:
            self.assertEqual(result, "Completely unrelated answer")


if __name__ == "__main__":
    unittest.main()
