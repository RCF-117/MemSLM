"""Integration-style tests for MemoryManager orchestration (with mocks)."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.utils.helpers import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class FakeLLM:
    def __init__(self, response: str | list[str] = "mock-llm-response") -> None:
        self.response = response
        self.calls = 0
        self.last_messages = None
        self.history = []

    def chat(self, messages):
        self.calls += 1
        self.last_messages = messages
        self.history.append(messages)
        if isinstance(self.response, list):
            idx = min(self.calls - 1, len(self.response) - 1)
            return self.response[idx]
        return self.response


class FakeShortMemory:
    def __init__(self, max_turns=None, config=None) -> None:
        self.max_turns = int(max_turns or 10)
        self.buffer = []
        self.flush_calls = 0

    def add(self, message):
        self.buffer.append(dict(message))

    def get(self):
        return list(self.buffer)

    def flush_to_mid_memory(self, mid_memory):
        self.flush_calls += 1
        while len(self.buffer) > self.max_turns:
            mid_memory.add(self.buffer.pop(0))

    def clear(self):
        self.buffer.clear()


class FakeMidMemory:
    def __init__(self, config=None) -> None:
        self.search_calls = 0
        self.rerank_calls = 0
        self.global_search_calls = 0
        self.added = []
        self.closed = False
        self.cleared = False
        self.global_chunk_retrieval_enabled = True
        self._chunks = [
            {
                "chunk_id": 1,
                "role": "user",
                "text": "(user) She moved to Boston in 2023.",
                "session_id": "s1",
                "session_date": "2023/01/01",
                "has_answer": 1,
                "score": 0.9,
            }
        ]

    def search(self, query):
        self.search_calls += 1
        return []

    def rerank_chunks(self, query, topics):
        self.rerank_calls += 1
        return list(self._chunks)

    def search_chunks_global_with_limit(self, query, topic_score_map=None, top_n=5):
        self.global_search_calls += 1
        return list(self._chunks)[: max(1, int(top_n))]

    def search_chunks_global(self, query, topic_score_map=None):
        self.global_search_calls += 1
        return list(self._chunks)

    def add(self, message):
        self.added.append(dict(message))

    def flush_pending(self):
        return None

    def clear_all(self):
        self.cleared = True
        self.added.clear()

    def close(self):
        self.closed = True


class TestMemoryManager(unittest.TestCase):
    def _config(self):
        cfg = load_config(str(CONFIG_PATH))
        return copy.deepcopy(cfg)

    def _build_manager(self, cfg=None, llm=None):
        cfg = cfg or self._config()
        llm = llm or FakeLLM("mock-llm-response")
        with patch("llm_long_memory.memory.memory_manager.ShortMemory", FakeShortMemory):
            with patch("llm_long_memory.memory.memory_manager.MidMemory", FakeMidMemory):
                manager = MemoryManager(llm=llm, config=cfg)
        return manager, llm

    def test_retrieve_context_groups_by_topic(self):
        manager, _ = self._build_manager()
        context, topics, chunks = manager.retrieve_context("where moved")
        self.assertEqual(len(topics), 0)
        self.assertEqual(len(chunks), 1)
        self.assertIn("[Chunk 1]", context)
        self.assertIn("Boston", context)
        self.assertEqual(manager.mid_memory.search_calls, 0)
        self.assertEqual(manager.mid_memory.rerank_calls, 0)
        self.assertEqual(manager.mid_memory.global_search_calls, 1)

    def test_chat_uses_precomputed_context_without_extra_retrieval(self):
        manager, llm = self._build_manager()
        pre = (
            "[Chunk 1]\n(user) She moved to Boston in 2023.",
            [],
            [
                {
                    "chunk_id": 1,
                    "role": "user",
                    "text": "(user) She moved to Boston in 2023.",
                    "score": 0.9,
                    "session_id": "s1",
                    "session_date": "2023/01/01",
                    "has_answer": 1,
                }
            ],
        )
        out = manager.chat("where did she move?", precomputed_context=pre)
        self.assertGreaterEqual(llm.calls, 1)
        self.assertIsInstance(out, str)
        self.assertEqual(manager.mid_memory.search_calls, 0)
        self.assertEqual(manager.mid_memory.rerank_calls, 0)

    def test_chat_context_only_fallback(self):
        cfg = self._config()
        cfg["retrieval"]["answering"]["context_only"] = True
        cfg["retrieval"]["answering"]["llm_fallback_to_top_candidate"] = False
        llm = FakeLLM("unrelated answer not in evidence")
        manager, _ = self._build_manager(cfg=cfg, llm=llm)
        out = manager.chat("where did she move?")
        self.assertEqual(out, "Not found in retrieved context.")
        self.assertGreaterEqual(llm.calls, 1)

    def test_chat_uses_prompt_fallback_for_not_found(self):
        manager, llm = self._build_manager(llm=FakeLLM("Not found in retrieved context."))

        def _prepare_answer_inputs(query, precomputed_context):
            return (
                "[Chunk 1]\n(assistant) The first issue was the GPS system not functioning correctly.",
                [],
                [
                    {
                        "chunk_id": 1,
                        "role": "assistant",
                        "text": "(assistant) The first issue was the GPS system not functioning correctly.",
                        "score": 0.9,
                        "session_id": "s1",
                        "session_date": "2023/01/01",
                        "has_answer": 1,
                    }
                ],
                [
                    {
                        "text": "The first issue was the GPS system not functioning correctly.",
                        "score": 0.9,
                        "chunk_id": 1,
                        "session_date": "2023/01/01",
                    }
                ],
                [],
                "",
                {"answer": "GPS system not functioning correctly", "source": "intent_span", "score": "0.9000"},
                "The first issue was the GPS system not functioning correctly.",
                "",
            )

        manager._prepare_answer_inputs = _prepare_answer_inputs  # type: ignore[method-assign]
        out = manager.chat("What was the first issue with my car?")
        self.assertEqual(out, "GPS system not functioning correctly")
        self.assertIsNotNone(llm.last_messages)
        self.assertNotIn("[Query Plan]", llm.last_messages[0]["content"])
        self.assertIn("[Answer Rules]", llm.last_messages[0]["content"])
        self.assertNotEqual(out, "Not found in retrieved context.")

    def test_counting_fallback_is_not_forced_without_evidence_support(self):
        manager, llm = self._build_manager(llm=FakeLLM("unrelated answer"))
        manager.answer_grounding.reasoning_fallback_enabled = True
        out = manager.chat("How many items did I buy?")
        self.assertEqual(out, "Not found in retrieved context.")
        self.assertGreaterEqual(llm.calls, 1)

    def test_chat_always_invokes_llm(self):
        cfg = self._config()
        llm = FakeLLM("this should not be used")
        manager, _ = self._build_manager(cfg=cfg, llm=llm)
        out = manager.chat("question")
        self.assertIsInstance(out, str)
        self.assertGreaterEqual(llm.calls, 1)

    def test_second_pass_uses_expanded_prompt_variant(self):
        llm = FakeLLM(["Not found in retrieved context.", "Boston"])
        manager, _ = self._build_manager(llm=llm)
        manager.last_evidence_graph_bundle = {
            "filtered_pack": {
                "core_evidence": [
                    {"text": "Core evidence one."},
                    {"text": "Core evidence two."},
                    {"text": "Core evidence three."},
                ],
                "supporting_evidence": [
                    {"text": "Supporting evidence one."},
                    {"text": "Supporting evidence two."},
                ],
                "conflict_evidence": [{"text": "Conflict evidence one."}],
            },
            "claim_result": {"claims": [], "support_units": []},
            "light_graph": {"nodes": [], "edges": []},
        }
        manager.chat_runtime._last_specialist_payload = {}
        prompt_text = manager._build_generation_prompt(
            input_text="Where did she move?",
            retrieved_context_text="",
            evidence_sentences=[{"text": "She moved to Boston in 2023.", "score": 0.9}],
            chunks=[],
            candidates=[],
            best_evidence="",
            fallback_answer="",
            evidence_candidate=None,
        )
        ai_response, fallback_path, _ = manager._generate_final_answer(
            input_text="Where did she move?",
            query="Where did she move?",
            prompt_text=prompt_text,
            evidence_sentences=[{"text": "She moved to Boston in 2023.", "score": 0.9}],
            candidates=[],
            fallback_answer="",
            evidence_candidate=None,
        )
        self.assertEqual(ai_response, "Boston")
        self.assertTrue(fallback_path.startswith("second_pass:"))
        self.assertEqual(llm.calls, 2)
        first_prompt = llm.history[0][0]["content"]
        second_prompt = llm.history[1][0]["content"]
        self.assertNotIn("Supporting evidence two.", first_prompt)
        self.assertIn("Supporting evidence two.", second_prompt)

    def test_reset_and_close(self):
        manager, _ = self._build_manager()
        manager.reset_for_new_instance()
        self.assertTrue(manager.mid_memory.cleared)
        self.assertTrue(manager.long_memory.cleared)
        manager.close()
        self.assertTrue(manager.mid_memory.closed)
        self.assertTrue(manager.long_memory.closed)


if __name__ == "__main__":
    unittest.main()
