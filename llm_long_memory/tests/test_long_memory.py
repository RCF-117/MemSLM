"""Unit tests for long-term graph memory lifecycle and retrieval."""

from __future__ import annotations

import copy
import tempfile
import time
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
        cfg["memory"]["long_memory"]["consolidation_every_updates"] = 100000
        cfg["memory"]["long_memory"]["forgetting_every_updates"] = 100000
        cfg["memory"]["long_memory"]["worker_poll_timeout_sec"] = 0.05
        cfg["memory"]["long_memory"]["retrieval_scoring"]["use_embedding"] = False
        return cfg

    def test_process_message_and_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "long_memory.db"
            lm = LongMemory(config=self._config(db_path=db_path, enabled=True))
            try:
                lm._process_message(
                    {
                        "role": "user",
                        "content": "Alice moved to Boston in 2023 and works at Acme.",
                        "session_date": "2023/01/01",
                        "session_id": "s1",
                    }
                )
                snippets = lm.build_context_snippets("Where did Alice move")
                self.assertGreaterEqual(len(snippets), 1)
                self.assertTrue(any("alice" in x.lower() or "boston" in x.lower() for x in snippets))
            finally:
                lm.close()

    def test_async_enqueue_applies_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "long_memory_async.db"
            lm = LongMemory(config=self._config(db_path=db_path, enabled=True))
            try:
                ok = lm.enqueue_message(
                    {
                        "role": "user",
                        "content": "Bob studied Business Administration.",
                        "session_date": "2023/05/30",
                        "session_id": "s2",
                    }
                )
                self.assertTrue(ok)
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    stats = lm.debug_stats()
                    if int(stats["applied_updates"]) > 0:
                        break
                    time.sleep(0.05)
                self.assertGreater(int(lm.debug_stats()["applied_updates"]), 0)
            finally:
                lm.close()

    def test_clear_all_resets_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "long_memory_clear.db"
            lm = LongMemory(config=self._config(db_path=db_path, enabled=False))
            try:
                lm._process_message(
                    {
                        "role": "user",
                        "content": "Alice moved to Shanghai in 2024.",
                        "session_date": "2024/08/01",
                        "session_id": "s3",
                    }
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
