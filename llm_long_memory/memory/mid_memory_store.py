"""SQLite storage layer for MidMemory."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Callable, Dict, List, Sequence

from llm_long_memory.evaluation.eval_store import EvalStore
from llm_long_memory.utils.helpers import resolve_project_path
from llm_long_memory.utils.logger import logger


class MidMemoryStore:
    """Encapsulate SQLite setup, schema compatibility, and FTS operations."""

    def __init__(
        self,
        database_file: str,
        sqlite_busy_timeout_ms: int,
        sqlite_journal_mode: str,
        sqlite_synchronous: str,
        sqlite_checkpoint_on_commit: bool,
        sqlite_checkpoint_mode: str,
        lexical_search_enabled: bool,
        eval_cfg: dict,
    ) -> None:
        self.db_path = self._resolve_path(database_file)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        self.lexical_search_enabled = bool(lexical_search_enabled)
        self.sqlite_busy_timeout_ms = int(sqlite_busy_timeout_ms)
        self.sqlite_journal_mode = str(sqlite_journal_mode)
        self.sqlite_synchronous = str(sqlite_synchronous)
        self.sqlite_checkpoint_on_commit = bool(sqlite_checkpoint_on_commit)
        self.sqlite_checkpoint_mode = str(sqlite_checkpoint_mode).upper()

        self._configure_sqlite()
        self._create_tables()
        self._ensure_schema_compat()
        self._create_indexes()

        self.eval_store = EvalStore(conn=self.conn, eval_cfg=eval_cfg)
        self.eval_store.create_tables()
        self.eval_store.ensure_schema_compat()

    @staticmethod
    def _resolve_path(path: str) -> Path:
        return resolve_project_path(path)

    def _configure_sqlite(self) -> None:
        self.conn.execute(f"PRAGMA journal_mode={self.sqlite_journal_mode}")
        self.conn.execute(f"PRAGMA synchronous={self.sqlite_synchronous}")
        self.conn.execute(f"PRAGMA busy_timeout={self.sqlite_busy_timeout_ms}")
        self.conn.commit()

    def _checkpoint_if_enabled(self) -> None:
        if not self.sqlite_checkpoint_on_commit:
            return
        if self.sqlite_journal_mode.upper() != "WAL":
            return
        mode = self.sqlite_checkpoint_mode if self.sqlite_checkpoint_mode in {
            "PASSIVE",
            "FULL",
            "RESTART",
            "TRUNCATE",
        } else "TRUNCATE"
        try:
            self.conn.execute(f"PRAGMA wal_checkpoint({mode})")
        except sqlite3.OperationalError as exc:
            logger.warn(f"MidMemoryStore.wal_checkpoint failed: {exc}")

    def _create_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS topics(
              topic_id TEXT PRIMARY KEY,
              topic_embedding BLOB,
              summary TEXT,
              summary_embedding BLOB,
              keywords TEXT,
              topic_times TEXT,
              last_updated_step INTEGER,
              last_summary_step INTEGER,
              active INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks(
              chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
              topic_id TEXT,
              text TEXT,
              chunk_embedding BLOB,
              chunk_role TEXT,
              chunk_session_id TEXT,
              chunk_session_date TEXT,
              chunk_has_answer INTEGER,
              chunk_times TEXT
            )
            """
        )
        if self.lexical_search_enabled:
            try:
                self.conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(text, topic_id UNINDEXED)
                    """
                )
            except sqlite3.OperationalError:
                self.lexical_search_enabled = False
                logger.warn("MidMemory: SQLite FTS5 unavailable, lexical retrieval disabled.")
        self.conn.commit()

    def _create_indexes(self) -> None:
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_topics_last_updated ON topics(last_updated_step)"
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_topic_id ON chunks(topic_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id)")
        self.conn.commit()

    def _ensure_schema_compat(self) -> None:
        chunk_cols = self.conn.execute("PRAGMA table_info(chunks)").fetchall()
        chunk_names = {str(row["name"]) for row in chunk_cols}
        topic_cols = self.conn.execute("PRAGMA table_info(topics)").fetchall()
        topic_names = {str(row["name"]) for row in topic_cols}
        if "chunk_role" not in chunk_names:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN chunk_role TEXT")
        if "chunk_session_id" not in chunk_names:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN chunk_session_id TEXT")
        if "chunk_session_date" not in chunk_names:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN chunk_session_date TEXT")
        if "chunk_has_answer" not in chunk_names:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN chunk_has_answer INTEGER")
        if "chunk_times" not in chunk_names:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN chunk_times TEXT")
        if "last_summary_step" not in topic_names:
            self.conn.execute("ALTER TABLE topics ADD COLUMN last_summary_step INTEGER DEFAULT 0")
        if "topic_times" not in topic_names:
            self.conn.execute("ALTER TABLE topics ADD COLUMN topic_times TEXT")
        if self.lexical_search_enabled:
            try:
                self.conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(text, topic_id UNINDEXED)
                    """
                )
            except sqlite3.OperationalError:
                self.lexical_search_enabled = False
                logger.warn("MidMemory: SQLite FTS5 unavailable, lexical retrieval disabled.")
        self.conn.commit()

    def load_current_step(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(last_updated_step), 0) AS step FROM topics"
        ).fetchone()
        return int(row["step"]) if row else 0

    def index_chunk_fts(self, chunk_id: int, topic_id: str, text: str) -> None:
        if not self.lexical_search_enabled:
            return
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO chunks_fts(rowid, text, topic_id) VALUES(?, ?, ?)",
                (int(chunk_id), str(text), str(topic_id)),
            )
        except sqlite3.OperationalError as exc:
            logger.warn(f"MidMemory._index_chunk_fts failed: {exc}")

    def delete_chunk_fts(self, chunk_ids: Sequence[int]) -> None:
        if not self.lexical_search_enabled or not chunk_ids:
            return
        self.conn.executemany(
            "DELETE FROM chunks_fts WHERE rowid = ?",
            [(int(cid),) for cid in chunk_ids],
        )

    def rebuild_chunk_fts(self) -> None:
        if not self.lexical_search_enabled:
            return
        rows = self.conn.execute("SELECT chunk_id, topic_id, text FROM chunks").fetchall()
        self.conn.execute("DELETE FROM chunks_fts")
        for row in rows:
            self.conn.execute(
                "INSERT INTO chunks_fts(rowid, text, topic_id) VALUES(?, ?, ?)",
                (int(row["chunk_id"]), str(row["text"]), str(row["topic_id"])),
            )

    def lexical_rank_map(
        self,
        topic_id: str,
        query: str,
        tokenize: Callable[[str], List[str]],
        bm25_top_n: int,
    ) -> Dict[int, int]:
        if not self.lexical_search_enabled:
            return {}
        query_tokens = tokenize(query)
        if not query_tokens:
            return {}
        fts_query = " OR ".join(query_tokens)
        try:
            rows = self.conn.execute(
                """
                SELECT rowid AS chunk_id
                FROM chunks_fts
                WHERE chunks_fts MATCH ? AND topic_id = ?
                ORDER BY bm25(chunks_fts) ASC
                LIMIT ?
                """,
                (fts_query, topic_id, int(bm25_top_n)),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        return {int(row["chunk_id"]): idx + 1 for idx, row in enumerate(rows)}

    def lexical_rank_map_global(
        self,
        query: str,
        tokenize: Callable[[str], List[str]],
        bm25_top_n: int,
    ) -> Dict[int, int]:
        """Return global lexical rank map across all chunks."""
        if not self.lexical_search_enabled:
            return {}
        query_tokens = tokenize(query)
        if not query_tokens:
            return {}
        fts_query = " OR ".join(query_tokens)
        try:
            rows = self.conn.execute(
                """
                SELECT rowid AS chunk_id
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts) ASC
                LIMIT ?
                """,
                (fts_query, int(bm25_top_n)),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        return {int(row["chunk_id"]): idx + 1 for idx, row in enumerate(rows)}

    def debug_stats(self) -> Dict[str, int]:
        t = self.conn.execute("SELECT COUNT(*) AS cnt FROM topics").fetchone()
        c = self.conn.execute("SELECT COUNT(*) AS cnt FROM chunks").fetchone()
        a = self.conn.execute("SELECT COUNT(*) AS cnt FROM topics WHERE active = 1").fetchone()
        topics = int(t["cnt"]) if t else 0
        chunks = int(c["cnt"]) if c else 0
        active = int(a["cnt"]) if a else 0
        return {
            "topics": topics,
            "chunks": chunks,
            "active_topics": active,
            "inactive_topics": max(0, topics - active),
        }

    def clear_all(self) -> None:
        self.conn.execute("DELETE FROM chunks")
        if self.lexical_search_enabled:
            self.conn.execute("DELETE FROM chunks_fts")
        self.conn.execute("DELETE FROM topics")
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()
        self._checkpoint_if_enabled()

    def close(self) -> None:
        try:
            self._checkpoint_if_enabled()
        finally:
            self.conn.close()
