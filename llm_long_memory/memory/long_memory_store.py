"""SQLite storage layer for minimal long memory."""

from __future__ import annotations

import sqlite3

import numpy as np

from llm_long_memory.memory.long_memory_store_read import LongMemoryStoreReadMixin
from llm_long_memory.memory.long_memory_store_write import LongMemoryStoreWriteMixin
from llm_long_memory.utils.helpers import resolve_project_path


class LongMemoryStore(LongMemoryStoreWriteMixin, LongMemoryStoreReadMixin):
    """Persist extracted long-memory events in SQLite."""

    def __init__(
        self,
        database_file: str,
        sqlite_busy_timeout_ms: int,
        sqlite_journal_mode: str,
        sqlite_synchronous: str,
        embedding_dim: int,
    ) -> None:
        self.db_path = resolve_project_path(database_file)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.embedding_dim = int(embedding_dim)

        self.conn.execute(f"PRAGMA busy_timeout={int(sqlite_busy_timeout_ms)}")
        self.conn.execute(f"PRAGMA journal_mode={str(sqlite_journal_mode)}")
        self.conn.execute(f"PRAGMA synchronous={str(sqlite_synchronous)}")
        self._create_tables()
        self._create_indexes()
        self.conn.commit()

    def _create_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta(
              key TEXT PRIMARY KEY,
              value TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events(
              event_id TEXT PRIMARY KEY,
              fact_key TEXT,
              subject_action_key TEXT,
              fact_type TEXT,
              skeleton_text TEXT,
              skeleton_embedding BLOB,
              keywords TEXT,
              role TEXT,
              boundary_flag INTEGER,
              extract_confidence REAL,
              source_model TEXT,
              raw_span TEXT,
              status TEXT,
              salience REAL,
              first_seen_step INTEGER,
              last_seen_step INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events_staging(
              staging_id TEXT PRIMARY KEY,
              fact_key TEXT,
              skeleton_text TEXT,
              skeleton_embedding BLOB,
              keywords TEXT,
              role TEXT,
              extract_confidence REAL,
              source_model TEXT,
              raw_span TEXT,
              source_content TEXT,
              reject_reason TEXT,
              created_step INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS details(
              detail_id TEXT PRIMARY KEY,
              event_id TEXT,
              kind TEXT,
              text TEXT,
              created_step INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_nodes(
              node_id TEXT PRIMARY KEY,
              event_id TEXT,
              node_kind TEXT,
              node_text TEXT,
              is_core INTEGER,
              node_embedding BLOB,
              created_step INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_node_edges(
              node_edge_id TEXT PRIMARY KEY,
              event_id TEXT,
              from_node_id TEXT,
              to_node_id TEXT,
              relation TEXT,
              weight REAL,
              created_step INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS edges(
              edge_id TEXT PRIMARY KEY,
              from_event_id TEXT,
              to_event_id TEXT,
              relation TEXT,
              weight REAL,
              created_step INTEGER
            )
            """
        )

    def _create_indexes(self) -> None:
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_last_seen ON events(last_seen_step)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_fact_key ON events(fact_key)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_staging_created ON events_staging(created_step)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_details_event ON details(event_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_event_nodes_event ON event_nodes(event_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_event_nodes_kind ON event_nodes(node_kind)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_event_node_edges_event ON event_node_edges(event_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_event_node_edges_from ON event_node_edges(from_node_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_event_node_edges_to ON event_node_edges(to_node_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_event_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_event_id)")

    @staticmethod
    def _arr_to_blob(arr: np.ndarray) -> bytes:
        return arr.astype(np.float32).tobytes()

    def blob_to_arr(self, blob: bytes | None) -> np.ndarray:
        if not blob:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        arr = np.frombuffer(blob, dtype=np.float32)
        out = np.zeros(self.embedding_dim, dtype=np.float32)
        n = min(arr.size, self.embedding_dim)
        out[:n] = arr[:n]
        return out

    def load_current_step(self) -> int:
        row = self.conn.execute("SELECT value FROM meta WHERE key='current_step'").fetchone()
        if row is None:
            self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('current_step','0')")
            self.conn.commit()
            return 0
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return 0

    def save_current_step(self, step: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('current_step', ?)",
            (str(int(step)),),
        )

    def clear_all(self) -> None:
        self.conn.execute("DELETE FROM edges")
        self.conn.execute("DELETE FROM event_node_edges")
        self.conn.execute("DELETE FROM event_nodes")
        self.conn.execute("DELETE FROM details")
        self.conn.execute("DELETE FROM events_staging")
        self.conn.execute("DELETE FROM events")

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

