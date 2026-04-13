"""Unit tests for long-term graph memory lifecycle and retrieval."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from llm_long_memory.memory.long_memory import LongMemory
from llm_long_memory.utils.helpers import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class TestLongMemory(unittest.TestCase):
    def _config(self, db_path: Path, enabled: bool = True):
        cfg = load_config(str(CONFIG_PATH))
        cfg = copy.deepcopy(cfg)
        cfg["memory"]["long_memory"]["enabled"] = bool(enabled)
        cfg["memory"]["long_memory"]["database_file"] = str(db_path)
        cfg["memory"]["long_memory"]["extractor"]["enabled"] = True
        cfg["memory"]["long_memory"]["extractor"]["min_confidence"] = 0.0
        cfg["memory"]["long_memory"]["extractor"]["gating"]["quality_threshold"] = 0.0
        cfg["memory"]["long_memory"]["retrieval_scoring"]["use_embedding"] = False
        cfg["memory"]["long_memory"]["offline_graph"]["use_full_chunk_text"] = True
        return cfg

    @staticmethod
    def _fake_events(content: str):
        return [
            {
                "subject": "Alice",
                "action": "moved to",
                "object": "Boston",
                "event_text": "Alice moved to Boston",
                "keywords": ["alice", "moved", "boston"],
                "time": "2023",
                "location": "Boston",
                "confidence": 0.9,
                "role": "user",
                "source_model": "unit-test",
                "raw_span": content,
                "source_content": content,
            }
        ]

    def test_process_message_and_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "long_memory.db"
            lm = LongMemory(config=self._config(db_path=db_path, enabled=True))
            try:
                lm.extract_events_structured = lambda message, force=False: self._fake_events(
                    str(message.get("content", ""))
                )
                accepted = lm.ingest_from_chunks(
                    chunks=[
                        {
                            "text": "Alice moved to Boston in 2023 and works at Acme.",
                            "session_id": "s1",
                        }
                    ],
                    top_chunks=1,
                    max_chars_per_chunk=220,
                )
                self.assertGreaterEqual(accepted, 1)
                snippets = lm.build_context_snippets("Where did Alice move")
                self.assertGreaterEqual(len(snippets), 1)
                self.assertTrue(any("alice" in x.lower() or "boston" in x.lower() for x in snippets))
            finally:
                lm.close()

    def test_ingest_applies_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "long_memory_enqueue.db"
            lm = LongMemory(config=self._config(db_path=db_path, enabled=True))
            try:
                lm.extract_events_structured = lambda message, force=False: self._fake_events(
                    str(message.get("content", ""))
                )
                accepted = lm.ingest_from_chunks(
                    chunks=[{"text": "Bob studied Business Administration.", "session_id": "s2"}],
                    top_chunks=1,
                    max_chars_per_chunk=200,
                )
                self.assertGreaterEqual(accepted, 1)
                self.assertGreater(int(lm.debug_stats()["applied_updates"]), 0)
            finally:
                lm.close()

    def test_clear_all_resets_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "long_memory_clear.db"
            lm = LongMemory(config=self._config(db_path=db_path, enabled=True))
            try:
                lm.extract_events_structured = lambda message, force=False: self._fake_events(
                    str(message.get("content", ""))
                )
                lm.ingest_from_chunks(
                    chunks=[{"text": "Alice moved to Shanghai in 2024.", "session_id": "s3"}],
                    top_chunks=1,
                    max_chars_per_chunk=220,
                )
                before = lm.debug_stats()
                self.assertGreater(before["nodes"], 0)
                lm.clear_all()
                after = lm.debug_stats()
                self.assertEqual(after["nodes"], 0)
                self.assertEqual(after["edges"], 0)
            finally:
                lm.close()


if __name__ == "__main__":
    unittest.main()
