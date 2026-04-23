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

    def test_prompt_prefers_prompt_text_when_available(self) -> None:
        extractor = EvidenceGraphExtractor(self.manager, {"enabled": True})
        prompt = extractor._prompt(
            filtered_pack={
                "query": "Where is the painting now?",
                "intent": "update",
                "answer_type": "update",
                "focus_phrases": ["painting"],
                "target_object": "painting",
            },
            batch=[
                {
                    "evidence_id": "ev_001",
                    "channel": "rag_evidence",
                    "bucket": "core",
                    "session_date": "2023/01/01",
                    "text": "I moved the painting to my bedroom. Here is a lot of generic extra explanation that should not be fed to the extractor prompt.",
                    "prompt_text": "I moved the painting to my bedroom.",
                    "signals": ["update_signal"],
                }
            ],
        )
        self.assertIn("I moved the painting to my bedroom.", prompt)
        self.assertNotIn("generic extra explanation", prompt)

    def test_extract_claims_synthesizes_missing_claims_from_support_units(self) -> None:
        extractor = EvidenceGraphExtractor(
            self.manager,
            {
                "enabled": True,
                "extractor_max_claims_per_batch": 4,
                "extractor_max_support_units_per_batch": 4,
            },
        )

        def _call_model(_prompt: str) -> str:
            return json.dumps(
                {
                    "support_units": [
                        {
                            "unit_type": "state_span",
                            "text": "The painting is now in my bedroom.",
                            "subject_hint": "painting",
                            "predicate_hint": "location",
                            "value_hint": "bedroom",
                            "state_key": "location",
                            "status": "current",
                            "confidence": 0.9,
                            "evidence_ids": ["ev_001"],
                            "verbatim_span": "in my bedroom",
                        }
                    ],
                    "claims": [
                        {
                            "claim_type": "fact_statement",
                            "subject": "car",
                            "predicate": "color",
                            "value": "silver",
                            "time_anchor": "",
                            "state_key": "",
                            "status": "current",
                            "modality": "reported",
                            "compare_role": "",
                            "numeric_value": "",
                            "unit": "",
                            "confidence": 0.8,
                            "evidence_ids": ["ev_002"],
                            "verbatim_span": "silver",
                        }
                    ],
                }
            )

        extractor._call_model = _call_model  # type: ignore[method-assign]

        pack = {
            "query": "Where is the painting now?",
            "answer_type": "update",
            "focus_phrases": [],
            "target_object": "painting",
            "core_evidence": [
                {
                    "evidence_id": "ev_001",
                    "text": "The painting is now in my bedroom.",
                    "prompt_text": "The painting is now in my bedroom.",
                    "channel": "plan_combined_evidence",
                    "score": 0.9,
                    "chunk_id": 1,
                    "session_date": "2023/01/01",
                    "structured_format": False,
                    "window_backup": False,
                },
                {
                    "evidence_id": "ev_002",
                    "text": "My car is silver.",
                    "prompt_text": "My car is silver.",
                    "channel": "rag_evidence",
                    "score": 0.8,
                    "chunk_id": 2,
                    "session_date": "2023/01/01",
                    "structured_format": False,
                    "window_backup": False,
                },
            ],
            "supporting_evidence": [],
            "conflict_evidence": [],
        }
        result = extractor.extract_claims(pack)
        values = {(claim["subject"], claim["predicate"], claim["value"]) for claim in result["claims"]}
        self.assertIn(("car", "color", "silver"), values)
        self.assertIn(("painting", "location", "bedroom"), values)


if __name__ == "__main__":
    unittest.main()
