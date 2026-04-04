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
    def __init__(self, response: str = "mock-llm-response") -> None:
        self.response = response
        self.calls = 0
        self.last_messages = None

    def chat(self, messages):
        self.calls += 1
        self.last_messages = messages
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
        self.added = []
        self.closed = False
        self.cleared = False
        self._topics = [{"topic_id": "topic_1", "score": 1.0}]
        self._chunks = [
            {
                "topic_id": "topic_1",
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
        return list(self._topics)

    def rerank_chunks(self, query, topics):
        self.rerank_calls += 1
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


class FakeLongMemory:
    def __init__(self, config=None) -> None:
        self.enqueued = []
        self.cleared = False
        self.closed = False

    def enqueue_message(self, message):
        self.enqueued.append(dict(message))
        return True

    def build_context_snippets(self, query):
        return []

    def query(self, query_text):
        return []

    def debug_stats(self):
        return {
            "nodes": 0,
            "edges": 0,
            "queued_updates": 0,
            "applied_updates": 0,
            "ingest_event_total": 0,
            "ingest_event_accepted": 0,
            "ingest_event_rejected": 0,
            "candidate_events": 0,
        }

    def clear_all(self):
        self.cleared = True

    def close(self):
        self.closed = True


class TestMemoryManager(unittest.TestCase):
    def _config(self):
        cfg = load_config(str(CONFIG_PATH))
        cfg["retrieval"]["answering"]["decision"]["deterministic_enabled"] = False
        return copy.deepcopy(cfg)

    def _build_manager(self, cfg=None, llm=None):
        cfg = cfg or self._config()
        llm = llm or FakeLLM("mock-llm-response")
        with patch("llm_long_memory.memory.memory_manager.ShortMemory", FakeShortMemory):
            with patch("llm_long_memory.memory.memory_manager.MidMemory", FakeMidMemory):
                with patch("llm_long_memory.memory.memory_manager.LongMemory", FakeLongMemory):
                    manager = MemoryManager(llm=llm, config=cfg)
        return manager, llm

    def test_retrieve_context_groups_by_topic(self):
        manager, _ = self._build_manager()
        context, topics, chunks = manager.retrieve_context("where moved")
        self.assertEqual(len(topics), 1)
        self.assertEqual(len(chunks), 1)
        self.assertIn("[Topic: topic_1]", context)
        self.assertIn("Boston", context)
        self.assertEqual(manager.mid_memory.search_calls, 1)
        self.assertEqual(manager.mid_memory.rerank_calls, 1)

    def test_chat_uses_precomputed_context_without_extra_retrieval(self):
        manager, llm = self._build_manager()
        pre = (
            "[Topic: topic_1]\n(user) She moved to Boston in 2023.",
            [{"topic_id": "topic_1", "score": 1.0}],
            [
                {
                    "topic_id": "topic_1",
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
        expected_calls = 2 if manager.answering.second_pass_llm_enabled else 1
        self.assertEqual(llm.calls, expected_calls)
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

    def test_chat_short_circuit_skips_llm(self):
        cfg = self._config()
        cfg["retrieval"]["answering"]["short_circuit_enabled"] = True
        cfg["retrieval"]["answering"]["short_circuit_min_sentence_score"] = 0.0
        llm = FakeLLM("this should not be used")
        manager, _ = self._build_manager(cfg=cfg, llm=llm)
        manager.answering.maybe_short_circuit = lambda candidates, evidence: "short-circuit-answer"
        out = manager.chat("question")
        self.assertEqual(out, "short-circuit-answer")
        self.assertEqual(llm.calls, 0)

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
