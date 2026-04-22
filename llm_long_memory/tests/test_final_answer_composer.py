"""Tests for the final structured answer composer."""

from __future__ import annotations

import unittest

from llm_long_memory.memory.final_answer_composer import FinalAnswerComposer


class TestFinalAnswerComposer(unittest.TestCase):
    def setUp(self) -> None:
        self.composer = FinalAnswerComposer(
            {
                "composer_max_core_evidence": 4,
                "composer_max_supporting_evidence": 2,
                "composer_max_conflict_evidence": 1,
                "composer_max_claims": 4,
                "composer_max_support_units": 2,
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
                    "summary_lines": ["state_update=she | location | Chicago -> she | location | Boston"],
                    "answer_candidate": "Boston",
                }
            },
        )
        section_names = [section["section"] for section in sections]
        self.assertEqual(
            section_names,
            ["filtered_evidence", "graph_claims", "light_graph", "toolkit_output", "answer_rules"],
        )
        self.assertIn("[Filtered Evidence]", prompt)
        self.assertIn("[Graph Claims]", prompt)
        self.assertIn("[Light Graph]", prompt)
        self.assertIn("[Toolkit Analysis]", prompt)
        self.assertNotIn("[Query Plan]", prompt)


if __name__ == "__main__":
    unittest.main()
