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
            ["toolkit_output", "light_graph", "filtered_evidence", "answer_rules"],
        )
        self.assertLess(prompt.index("[Toolkit Analysis]"), prompt.index("[Light Graph]"))
        self.assertLess(prompt.index("[Light Graph]"), prompt.index("[Filtered Evidence]"))
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
        self.assertIn("Supporting evidence three.", expanded_prompt)
        self.assertNotIn("Supporting evidence one.", expanded_prompt)

    def test_build_prompt_keeps_long_filtered_evidence_verbatim(self) -> None:
        long_text = (
            "I just exchanged a pair of boots from Zara on 2/5 and still need to pick up the new pair. "
            "Here are several general tips about using reminders and organizing returns that are not central. "
            "More generic advice about notes apps and organization follows for a long time."
        )
        prompt, _ = self.composer.build_prompt(
            input_text="How many items of clothing do I need to pick up from the store?",
            filtered_pack={
                "query": "How many items of clothing do I need to pick up from the store?",
                "answer_type": "count",
                "focus_phrases": ["items of clothing", "pick up from the store"],
                "target_object": "items of clothing",
                "core_evidence": [{"text": long_text}],
                "supporting_evidence": [],
                "conflict_evidence": [],
            },
            claim_result={"claims": [], "support_units": []},
            light_graph={"nodes": [], "edges": []},
            toolkit_payload={},
            prompt_mode="compact",
        )
        self.assertIn("boots from Zara on 2/5", prompt)
        self.assertIn("More generic advice about notes apps", prompt)

    def test_build_prompt_projects_graph_claims_when_no_structural_edges(self) -> None:
        prompt, sections = self.composer.build_prompt(
            input_text="How many projects have I led or am currently leading?",
            filtered_pack={
                "core_evidence": [{"text": "I led a market research project."}],
                "supporting_evidence": [],
                "conflict_evidence": [],
            },
            claim_result={"claims": [], "support_units": []},
            light_graph={
                "nodes": [
                    {
                        "id": "claim_a",
                        "type": "fact",
                        "meta": {
                            "subject": "I",
                            "predicate": "led",
                            "value": "market research project",
                            "time_anchor": "2023",
                        },
                    },
                    {
                        "id": "claim_b",
                        "type": "fact",
                        "meta": {
                            "subject": "I",
                            "predicate": "leading",
                            "value": "analytics project",
                            "time_anchor": "2024",
                        },
                    },
                ],
                "edges": [
                    {"source": "claim_a", "target": "query_root", "type": "supports_query", "weight": 0.72},
                    {"source": "claim_b", "target": "query_root", "type": "supports_query", "weight": 0.65},
                ],
            },
            toolkit_payload={},
            prompt_mode="compact",
        )
        section_names = [section["section"] for section in sections]
        self.assertIn("light_graph", section_names)
        self.assertIn("[Light Graph]", prompt)
        self.assertIn("claim[fact]: I | led | market research project | time=2023", prompt)

    def test_build_prompt_prefers_prompt_text_for_filtered_evidence(self) -> None:
        long_text = (
            "I just exchanged a pair of boots from Zara on 2/5 and still need to pick up the new pair. "
            "Here are several general tips about using reminders and organizing returns that are not central. "
            "More generic advice about notes apps and organization follows for a long time."
        )
        prompt, _ = self.composer.build_prompt(
            input_text="How many items of clothing do I need to pick up from the store?",
            filtered_pack={
                "query": "How many items of clothing do I need to pick up from the store?",
                "answer_type": "count",
                "focus_phrases": ["items of clothing", "pick up from the store"],
                "target_object": "items of clothing",
                "core_evidence": [
                    {
                        "text": long_text,
                        "prompt_text": "I just exchanged a pair of boots from Zara on 2/5 and still need to pick up the new pair.",
                    }
                ],
                "supporting_evidence": [],
                "conflict_evidence": [],
            },
            claim_result={"claims": [], "support_units": []},
            light_graph={"nodes": [], "edges": []},
            toolkit_payload={},
            prompt_mode="compact",
        )
        self.assertIn("boots from Zara on 2/5", prompt)
        self.assertNotIn("More generic advice about notes apps", prompt)

    def test_build_prompt_obeys_graph_first_route_schema(self) -> None:
        prompt, sections = self.composer.build_prompt(
            input_text="Where is the painting now?",
            filtered_pack={
                "core_evidence": [{"text": "I moved the painting to my bedroom."}],
                "supporting_evidence": [],
                "conflict_evidence": [],
            },
            claim_result={"claims": [], "support_units": []},
            light_graph={
                "nodes": [
                    {
                        "id": "old",
                        "type": "state",
                        "meta": {"subject": "painting", "predicate": "location", "value": "living room"},
                    },
                    {
                        "id": "new",
                        "type": "state",
                        "meta": {"subject": "painting", "predicate": "location", "value": "bedroom"},
                    },
                ],
                "edges": [{"source": "old", "target": "new", "type": "updates", "state_key": "location"}],
            },
            toolkit_payload={},
            route_packet={
                "mode": "graph-first",
                "schema_sections": ["light_graph", "filtered_evidence", "answer_rules"],
            },
            answer_rules_text="Use Light Graph as the primary answer source.\nReturn only the final answer.",
        )
        section_names = [section["section"] for section in sections]
        self.assertEqual(section_names, ["light_graph", "filtered_evidence", "answer_rules"])
        self.assertLess(prompt.index("[Light Graph]"), prompt.index("[Filtered Evidence]"))
        self.assertIn("Use Light Graph as the primary answer source.", prompt)

    def test_build_support_sources_includes_toolkit_graph_and_filtered(self) -> None:
        support_sources = self.composer.build_support_sources(
            filtered_pack={
                "core_evidence": [{"text": "She moved to Boston in 2023.", "score": 0.9}],
                "supporting_evidence": [],
                "conflict_evidence": [],
            },
            claim_result={"claims": [], "support_units": []},
            light_graph={
                "nodes": [
                    {
                        "id": "state_1",
                        "type": "state",
                        "meta": {"subject": "she", "predicate": "location", "value": "Chicago"},
                    },
                    {
                        "id": "state_2",
                        "type": "state",
                        "meta": {"subject": "she", "predicate": "location", "value": "Boston"},
                    },
                ],
                "edges": [
                    {"source": "state_1", "target": "state_2", "type": "updates", "state_key": "location"}
                ],
            },
            toolkit_payload={
                "tool_payload": {
                    "intent": "update",
                    "activated": True,
                    "verified": True,
                    "confidence": 0.91,
                    "verified_candidate": "Boston",
                    "verification_reason": "update_edge_verified",
                    "verified_used_claim_ids": ["c1"],
                    "summary_lines": ["state_update=she | location | Chicago -> she | location | Boston"],
                }
            },
            prompt_mode="compact",
        )
        joined = "\n".join(str(item.get("text", "")) for item in support_sources)
        self.assertIn("Boston", joined)
        self.assertIn("update[location]", joined)
        self.assertIn("She moved to Boston in 2023.", joined)


if __name__ == "__main__":
    unittest.main()
