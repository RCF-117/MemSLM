"""SQLite storage layer for MidMemory (chunk + sentence dual-granularity)."""

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
        eval_database_file: str | None = None,
        *,
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
        self.eval_db_path = self._resolve_path(eval_database_file or database_file)
        self.eval_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.eval_conn = sqlite3.connect(str(self.eval_db_path))
        self.eval_conn.row_factory = sqlite3.Row

        self.lexical_search_enabled = bool(lexical_search_enabled)
        self.sqlite_busy_timeout_ms = int(sqlite_busy_timeout_ms)
        self.sqlite_journal_mode = str(sqlite_journal_mode)
        self.sqlite_synchronous = str(sqlite_synchronous)
        self.sqlite_checkpoint_on_commit = bool(sqlite_checkpoint_on_commit)
        self.sqlite_checkpoint_mode = str(sqlite_checkpoint_mode).upper()

        self._configure_sqlite()
        self._configure_sqlite_conn(self.eval_conn)
        self._create_tables()
        self._ensure_schema_compat()
        self._create_indexes()

        self.eval_store = EvalStore(conn=self.eval_conn, eval_cfg=eval_cfg)
        self.eval_store.create_tables()
        self.eval_store.ensure_schema_compat()

    @staticmethod
    def _resolve_path(path: str) -> Path:
        return resolve_project_path(path)

    def _configure_sqlite_conn(self, conn: sqlite3.Connection) -> None:
        conn.execute(f"PRAGMA journal_mode={self.sqlite_journal_mode}")
        conn.execute(f"PRAGMA synchronous={self.sqlite_synchronous}")
        conn.execute(f"PRAGMA busy_timeout={self.sqlite_busy_timeout_ms}")
        conn.commit()

    def _configure_sqlite(self) -> None:
        self._configure_sqlite_conn(self.conn)

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
            message = str(exc).lower()
            if "locked" not in message:
                logger.warn(f"MidMemoryStore.wal_checkpoint failed: {exc}")

    def _create_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks(
              chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
              text TEXT,
              chunk_embedding BLOB,
              chunk_role TEXT,
              chunk_role_hist TEXT,
              chunk_session_id TEXT,
              chunk_session_date TEXT,
              chunk_session_dates TEXT,
              chunk_has_answer INTEGER,
              chunk_has_answer_count INTEGER,
              chunk_answer_density REAL,
              chunk_times TEXT,
              chunk_time_sources TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sentences(
              sentence_id INTEGER PRIMARY KEY AUTOINCREMENT,
              chunk_id INTEGER,
              text TEXT,
              sentence_embedding BLOB,
              sentence_role TEXT,
              sentence_session_id TEXT,
              sentence_session_date TEXT,
              source_part_index INTEGER
            )
            """
        )
        if self.lexical_search_enabled:
            try:
                self.conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(text)
                    """
                )
                self.conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS sentences_fts
                    USING fts5(text)
                    """
                )
            except sqlite3.OperationalError:
                self.lexical_search_enabled = False
                logger.warn("MidMemory: SQLite FTS5 unavailable, lexical retrieval disabled.")
        self.conn.commit()

    def _create_indexes(self) -> None:
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_sentence_id ON sentences(sentence_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_chunk_id ON sentences(chunk_id)")
        self.conn.commit()

    def _ensure_schema_compat(self) -> None:
        chunk_cols = self.conn.execute("PRAGMA table_info(chunks)").fetchall()
        chunk_names = {str(row["name"]) for row in chunk_cols}
        if "chunk_role" not in chunk_names:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN chunk_role TEXT")
        if "chunk_role_hist" not in chunk_names:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN chunk_role_hist TEXT")
        if "chunk_session_id" not in chunk_names:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN chunk_session_id TEXT")
        if "chunk_session_date" not in chunk_names:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN chunk_session_date TEXT")
        if "chunk_session_dates" not in chunk_names:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN chunk_session_dates TEXT")
        if "chunk_has_answer" not in chunk_names:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN chunk_has_answer INTEGER")
        if "chunk_has_answer_count" not in chunk_names:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN chunk_has_answer_count INTEGER")
        if "chunk_answer_density" not in chunk_names:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN chunk_answer_density REAL")
        if "chunk_times" not in chunk_names:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN chunk_times TEXT")
        if "chunk_time_sources" not in chunk_names:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN chunk_time_sources TEXT")
        sentence_cols = self.conn.execute("PRAGMA table_info(sentences)").fetchall()
        sentence_names = {str(row["name"]) for row in sentence_cols}
        if not sentence_cols:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sentences(
                  sentence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  chunk_id INTEGER,
                  text TEXT,
                  sentence_embedding BLOB,
                  sentence_role TEXT,
                  sentence_session_id TEXT,
                  sentence_session_date TEXT,
                  source_part_index INTEGER
                )
                """
            )
            sentence_names = {
                "sentence_id",
                "chunk_id",
                "text",
                "sentence_embedding",
                "sentence_role",
                "sentence_session_id",
                "sentence_session_date",
                "source_part_index",
            }
        if "sentence_role" not in sentence_names:
            self.conn.execute("ALTER TABLE sentences ADD COLUMN sentence_role TEXT")
        if "sentence_session_id" not in sentence_names:
            self.conn.execute("ALTER TABLE sentences ADD COLUMN sentence_session_id TEXT")
        if "sentence_session_date" not in sentence_names:
            self.conn.execute("ALTER TABLE sentences ADD COLUMN sentence_session_date TEXT")
        if "source_part_index" not in sentence_names:
            self.conn.execute("ALTER TABLE sentences ADD COLUMN source_part_index INTEGER")
        if self.lexical_search_enabled:
            try:
                self.conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(text)
                    """
                )
                self.conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS sentences_fts
                    USING fts5(text)
                    """
                )
            except sqlite3.OperationalError:
                self.lexical_search_enabled = False
                logger.warn("MidMemory: SQLite FTS5 unavailable, lexical retrieval disabled.")
        self.conn.commit()

    def load_current_step(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(chunk_id), 0) AS step FROM chunks"
        ).fetchone()
        return int(row["step"]) if row else 0

    def index_chunk_fts(self, chunk_id: int, text: str) -> None:
        if not self.lexical_search_enabled:
            return
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO chunks_fts(rowid, text) VALUES(?, ?)",
                (int(chunk_id), str(text)),
            )
        except sqlite3.OperationalError as exc:
            logger.warn(f"MidMemory._index_chunk_fts failed: {exc}")

    def index_sentence_fts(self, sentence_id: int, text: str) -> None:
        if not self.lexical_search_enabled:
            return
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO sentences_fts(rowid, text) VALUES(?, ?)",
                (int(sentence_id), str(text)),
            )
        except sqlite3.OperationalError as exc:
            logger.warn(f"MidMemory._index_sentence_fts failed: {exc}")

    def delete_chunk_fts(self, chunk_ids: Sequence[int]) -> None:
        if not self.lexical_search_enabled or not chunk_ids:
            return
        self.conn.executemany(
            "DELETE FROM chunks_fts WHERE rowid = ?",
            [(int(cid),) for cid in chunk_ids],
        )

    def delete_sentence_fts(self, sentence_ids: Sequence[int]) -> None:
        if not self.lexical_search_enabled or not sentence_ids:
            return
        self.conn.executemany(
            "DELETE FROM sentences_fts WHERE rowid = ?",
            [(int(sid),) for sid in sentence_ids],
        )

    def rebuild_chunk_fts(self) -> None:
        if not self.lexical_search_enabled:
            return
        rows = self.conn.execute("SELECT chunk_id, text FROM chunks").fetchall()
        self.conn.execute("DELETE FROM chunks_fts")
        for row in rows:
            self.conn.execute(
                "INSERT INTO chunks_fts(rowid, text) VALUES(?, ?)",
                (int(row["chunk_id"]), str(row["text"])),
            )

    def rebuild_sentence_fts(self) -> None:
        if not self.lexical_search_enabled:
            return
        rows = self.conn.execute("SELECT sentence_id, text FROM sentences").fetchall()
        self.conn.execute("DELETE FROM sentences_fts")
        for row in rows:
            self.conn.execute(
                "INSERT INTO sentences_fts(rowid, text) VALUES(?, ?)",
                (int(row["sentence_id"]), str(row["text"])),
            )

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

    def lexical_rank_map_sentences(
        self,
        query: str,
        tokenize: Callable[[str], List[str]],
        bm25_top_n: int,
    ) -> Dict[int, int]:
        """Return lexical rank map across all sentence units."""
        if not self.lexical_search_enabled:
            return {}
        query_tokens = tokenize(query)
        if not query_tokens:
            return {}
        fts_query = " OR ".join(query_tokens)
        try:
            rows = self.conn.execute(
                """
                SELECT rowid AS sentence_id
                FROM sentences_fts
                WHERE sentences_fts MATCH ?
                ORDER BY bm25(sentences_fts) ASC
                LIMIT ?
                """,
                (fts_query, int(bm25_top_n)),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        return {int(row["sentence_id"]): idx + 1 for idx, row in enumerate(rows)}

    def debug_stats(self) -> Dict[str, int]:
        c = self.conn.execute("SELECT COUNT(*) AS cnt FROM chunks").fetchone()
        chunks = int(c["cnt"]) if c else 0
        s = self.conn.execute("SELECT COUNT(*) AS cnt FROM sentences").fetchone()
        sentences = int(s["cnt"]) if s else 0
        return {
            "chunks": chunks,
            "sentences": sentences,
        }

    def clear_all(self) -> None:
        self.conn.execute("DELETE FROM chunks")
        self.conn.execute("DELETE FROM sentences")
        if self.lexical_search_enabled:
            self.conn.execute("DELETE FROM chunks_fts")
            self.conn.execute("DELETE FROM sentences_fts")
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()
        self.eval_conn.commit()
        self._checkpoint_if_enabled()

    def close(self) -> None:
        try:
            self._checkpoint_if_enabled()
        finally:
            self.conn.close()
            try:
                self.eval_conn.commit()
            finally:
                self.eval_conn.close()
