from __future__ import annotations

import json
import unittest

from llm_long_memory.memory.evidence_graph_extractor import EvidenceGraphExtractor


class _FakeLLM:
    host = "http://127.0.0.1:11434"


class _FakeManager:
    def __init__(self) -> None:
        self.llm = _FakeLLM()
        self.config = {"llm": {"default_model": "qwen3:8b", "host": "http://127.0.0.1:11434"}}


class TestEvidenceGraphExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = _FakeManager()

    def test_empty_fallback_retries_once_when_enabled(self) -> None:
        extractor = EvidenceGraphExtractor(
            self.manager,
            {
                "enabled": True,
                "extractor_max_claims_per_batch": 4,
                "extractor_max_support_units_per_batch": 4,
            },
        )
        calls: list[str] = []

        def _call_model(prompt: str) -> str:
            calls.append(prompt)
            return json.dumps({"support_units": [], "claims": []})

        extractor._call_model = _call_model  # type: ignore[method-assign]

        pack = {
            "query": "Where is the painting now?",
            "answer_type": "update",
            "focus_phrases": [],
            "target_object": "painting",
            "core_evidence": [
                {
                    "evidence_id": "ev_001",
                    "text": "I moved the painting to my bedroom. It used to be above the sofa.",
                    "channel": "plan_combined_evidence",
                    "score": 0.9,
                    "chunk_id": 1,
                    "session_date": "2023/01/01",
                    "structured_format": True,
                    "window_backup": True,
                }
            ],
            "supporting_evidence": [],
            "conflict_evidence": [],
        }
        result = extractor.extract_claims(pack)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(result["raw_batches"][0]["attempts"]), 2)


if __name__ == "__main__":
    unittest.main()
