"""Tests for LongMemoryStore schema creation and compat path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_long_memory.memory.long_memory_store import LongMemoryStore


class TestLongMemoryStore(unittest.TestCase):
    def test_create_and_schema_columns_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "long_memory_store.db"
            store = LongMemoryStore(
                database_file=str(db_path),
                sqlite_busy_timeout_ms=5000,
                sqlite_journal_mode="WAL",
                sqlite_synchronous="NORMAL",
                embedding_dim=16,
            )
            try:
                tables = {
                    str(r["name"])
                    for r in store.conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                self.assertIn("events", tables)
                self.assertIn("details", tables)
                self.assertIn("meta", tables)

                event_cols = {
                    str(r["name"])
                    for r in store.conn.execute("PRAGMA table_info(events)").fetchall()
                }
                self.assertIn("event_id", event_cols)
                self.assertIn("status", event_cols)
                self.assertIn("last_seen_step", event_cols)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
