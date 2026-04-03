"""Tests for SQLite storage layer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_long_memory.utils.helpers import load_config
from llm_long_memory.memory.mid_memory_store import MidMemoryStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class TestMidMemoryStore(unittest.TestCase):
    def _build_store(self, tmpdir: str) -> MidMemoryStore:
        cfg = load_config(str(CONFIG_PATH))
        return MidMemoryStore(
            database_file=str(Path(tmpdir) / "test_mid_memory.db"),
            sqlite_busy_timeout_ms=int(cfg["memory"]["mid_memory"]["sqlite_busy_timeout_ms"]),
            sqlite_journal_mode=str(cfg["memory"]["mid_memory"]["sqlite_journal_mode"]),
            sqlite_synchronous=str(cfg["memory"]["mid_memory"]["sqlite_synchronous"]),
            lexical_search_enabled=True,
            eval_cfg=cfg["evaluation"],
        )

    def test_create_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._build_store(tmp)
            stats = store.debug_stats()
            self.assertEqual(stats["topics"], 0)
            self.assertEqual(stats["chunks"], 0)
            store.close()

    def test_fts_index_roundtrip_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._build_store(tmp)
            if not store.lexical_search_enabled:
                store.close()
                self.skipTest("SQLite FTS5 not available in this runtime.")
            store.conn.execute(
                """
                INSERT INTO topics(topic_id, topic_embedding, summary, summary_embedding, keywords, topic_times, last_updated_step, last_summary_step, active)
                VALUES('t1', ?, '', ?, '[]', '[]', 1, 0, 1)
                """,
                (b"\x00" * 16, b"\x00" * 16),
            )
            cursor = store.conn.execute(
                """
                INSERT INTO chunks(topic_id, text, chunk_embedding, chunk_role, chunk_session_id, chunk_session_date, chunk_has_answer, chunk_times)
                VALUES('t1', 'alice moved to boston', ?, 'user', '', '', 0, '[]')
                """,
                (b"\x00" * 16,),
            )
            chunk_id = int(cursor.lastrowid)
            store.index_chunk_fts(chunk_id, "t1", "alice moved to boston")
            rank_map = store.lexical_rank_map(
                topic_id="t1",
                query="alice boston",
                tokenize=lambda s: s.split(),
                bm25_top_n=10,
            )
            self.assertIn(chunk_id, rank_map)
            store.delete_chunk_fts([chunk_id])
            rank_map_after = store.lexical_rank_map(
                topic_id="t1",
                query="alice boston",
                tokenize=lambda s: s.split(),
                bm25_top_n=10,
            )
            self.assertNotIn(chunk_id, rank_map_after)
            store.close()


if __name__ == "__main__":
    unittest.main()
