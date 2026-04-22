"""Tests for the final structured answer composer."""

from __future__ import annotations

import unittest

from llm_long_memory.memory.final_answer_composer import FinalAnswerComposer


class TestFinalAnswerComposer(unittest.TestCase):
    def setUp(self) -> None:
        self.composer = FinalAnswerComposer(
            {
                "composer_default_prompt_mode": "compact",
                "composer_compact_max_core_evidence": 2,
                "composer_compact_max_supporting_evidence": 1,
                "composer_compact_max_conflict_evidence": 0,
                "composer_compact_max_graph_lines": 2,
                "composer_compact_max_tool_lines": 2,
                "composer_max_core_evidence": 4,
                "composer_max_supporting_evidence": 2,
                "composer_max_conflict_evidence": 1,
                "composer_max_graph_lines": 4,
                "composer_max_tool_lines": 4,
            }
        )

    def test_bundle_to_evidence_sentences_prefers_filtered_pack(self) -> None:
        bundle = {
            "filtered_pack": {
                "core_evidence": [{"text": "She moved to Boston in 2023.", "score": 0.9}],
                "supporting_evidence": [{"text": "The move happened after graduation.", "score": 0.5}],
                "conflict_evidence": [{"text": "Old location: Chicago.", "score": 0.2}],
            }
        }
        out = self.composer.bundle_to_evidence_sentences(bundle)
        self.assertEqual(out[0]["text"], "She moved to Boston in 2023.")
        self.assertEqual(len(out), 3)

    def test_build_prompt_uses_only_structured_sections(self) -> None:
        prompt, sections = self.composer.build_prompt(
            input_text="Where did she move?",
            filtered_pack={
                "core_evidence": [{"text": "She moved to Boston in 2023."}],
                "supporting_evidence": [],
                "conflict_evidence": [],
            },
            claim_result={
                "claims": [
                    {"subject": "she", "predicate": "location", "value": "Boston", "time_anchor": "2023"}
                ],
                "support_units": [],
            },
            light_graph={
                "nodes": [
                    {
                        "id": "old",
                        "type": "state",
                        "meta": {"subject": "she", "predicate": "location", "value": "Chicago"},
                    },
                    {
                        "id": "new",
                        "type": "state",
                        "meta": {"subject": "she", "predicate": "location", "value": "Boston"},
                    },
                ],
                "edges": [{"source": "old", "target": "new", "type": "updates", "state_key": "location"}],
            },
            toolkit_payload={
                "tool_payload": {
                    "intent": "update",
                    "activated": True,
                    "confidence": 0.92,
                    "verified": True,
                    "verified_candidate": "Boston",
                    "verification_reason": "update_edge_verified",
                    "verified_used_claim_ids": ["c1"],
                    "used_claim_ids": ["c1"],
                    "summary_lines": ["state_update=she | location | Chicago -> she | location | Boston"],
                    "answer_candidate": "Boston",
                }
            },
        )
        section_names = [section["section"] for section in sections]
        self.assertEqual(
            section_names,
            ["filtered_evidence", "light_graph", "toolkit_output", "answer_rules"],
        )
        self.assertIn("[Filtered Evidence]", prompt)
        self.assertIn("[Light Graph]", prompt)
        self.assertIn("[Toolkit Analysis]", prompt)
        self.assertNotIn("[Query Plan]", prompt)
        self.assertNotIn("[Graph Claims]", prompt)
        self.assertIn("tool_verification=update_edge_verified", prompt)

    def test_build_prompt_omits_unverified_toolkit_even_if_activated(self) -> None:
        prompt, sections = self.composer.build_prompt(
            input_text="Where is the painting now?",
            filtered_pack={
                "core_evidence": [{"text": "I moved the painting to my bedroom."}],
                "supporting_evidence": [],
                "conflict_evidence": [],
            },
            claim_result={"claims": [], "support_units": []},
            light_graph={"nodes": [], "edges": []},
            toolkit_payload={
                "tool_payload": {
                    "intent": "update",
                    "activated": True,
                    "confidence": 0.91,
                    "verified": False,
                    "verified_candidate": "",
                    "verification_reason": "update_requires_trusted_update_edge",
                    "used_claim_ids": ["c1"],
                    "summary_lines": ["state_answer=painting | location | bedroom"],
                    "answer_candidate": "bedroom",
                }
            },
        )
        section_names = [section["section"] for section in sections]
        self.assertNotIn("toolkit_output", section_names)
        self.assertNotIn("[Toolkit Analysis]", prompt)

    def test_build_prompt_compact_is_shorter_than_expanded(self) -> None:
        filtered_pack = {
            "core_evidence": [
                {"text": "Core evidence one."},
                {"text": "Core evidence two."},
                {"text": "Core evidence three."},
                {"text": "Core evidence four."},
                {"text": "Core evidence five."},
            ],
            "supporting_evidence": [
                {"text": "Supporting evidence one."},
                {"text": "Supporting evidence two."},
                {"text": "Supporting evidence three."},
            ],
            "conflict_evidence": [{"text": "Conflict evidence one."}],
        }
        light_graph = {
            "nodes": [
                {"id": "a", "meta": {"subject": "A", "predicate": "p", "value": "1"}},
                {"id": "b", "meta": {"subject": "A", "predicate": "p", "value": "2"}},
                {"id": "c", "meta": {"subject": "B", "predicate": "p", "value": "3"}},
            ],
            "edges": [
                {"source": "a", "target": "b", "type": "updates", "state_key": "p"},
                {"source": "b", "target": "c", "type": "before"},
            ],
        }
        toolkit_payload = {
            "tool_payload": {
                "intent": "update",
                "activated": True,
                "confidence": 0.92,
                "verified": True,
                "verified_candidate": "2",
                "verification_reason": "update_edge_verified",
                "verified_used_claim_ids": ["c1"],
                "summary_lines": [
                    "state_update=A | p | 1 -> A | p | 2",
                    "supporting line",
                    "extra line",
                ],
            }
        }
        compact_prompt, _ = self.composer.build_prompt(
            input_text="What is the latest value?",
            filtered_pack=filtered_pack,
            claim_result={"claims": [], "support_units": []},
            light_graph=light_graph,
            toolkit_payload=toolkit_payload,
            prompt_mode="compact",
        )
        expanded_prompt, _ = self.composer.build_prompt(
            input_text="What is the latest value?",
            filtered_pack=filtered_pack,
            claim_result={"claims": [], "support_units": []},
            light_graph=light_graph,
            toolkit_payload=toolkit_payload,
            prompt_mode="expanded",
        )
        self.assertLess(len(compact_prompt), len(expanded_prompt))
        self.assertNotIn("Core evidence three.", compact_prompt)
        self.assertNotIn("Core evidence two.", expanded_prompt)
        self.assertIn("Core evidence three.", expanded_prompt)
        self.assertIn("Core evidence four.", expanded_prompt)
        self.assertIn("Core evidence five.", expanded_prompt)
        self.assertIn("Supporting evidence two.", expanded_prompt)
        self.assertNotIn("Supporting evidence one.", expanded_prompt)


if __name__ == "__main__":
    unittest.main()
