"""SQLite storage layer for LongMemory."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from llm_long_memory.utils.helpers import resolve_project_path


class LongMemoryStore:
    """Encapsulate SQLite setup, schema compatibility, and CRUD for long memory."""

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

        self.sqlite_busy_timeout_ms = int(sqlite_busy_timeout_ms)
        self.sqlite_journal_mode = str(sqlite_journal_mode)
        self.sqlite_synchronous = str(sqlite_synchronous)
        self.embedding_dim = int(embedding_dim)

        self._configure_sqlite()
        self._create_tables()
        self._ensure_schema_compat()
        self._create_indexes()

    def _configure_sqlite(self) -> None:
        self.conn.execute(f"PRAGMA busy_timeout={self.sqlite_busy_timeout_ms}")
        self.conn.execute(f"PRAGMA journal_mode={self.sqlite_journal_mode}")
        self.conn.execute(f"PRAGMA synchronous={self.sqlite_synchronous}")
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
              skeleton_text TEXT,
              skeleton_embedding BLOB,
              keywords TEXT,
              role TEXT,
              status TEXT,
              salience REAL,
              first_seen_step INTEGER,
              last_seen_step INTEGER
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
        self.conn.commit()

    def _create_indexes(self) -> None:
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_fact_key ON events(fact_key)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_last_seen ON events(last_seen_step)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_details_event ON details(event_id)")
        self.conn.commit()

    def _ensure_schema_compat(self) -> None:
        event_cols = self.conn.execute("PRAGMA table_info(events)").fetchall()
        event_names = {str(row["name"]) for row in event_cols}
        for col, ddl in (
            ("fact_key", "ALTER TABLE events ADD COLUMN fact_key TEXT"),
            ("skeleton_text", "ALTER TABLE events ADD COLUMN skeleton_text TEXT"),
            ("skeleton_embedding", "ALTER TABLE events ADD COLUMN skeleton_embedding BLOB"),
            ("keywords", "ALTER TABLE events ADD COLUMN keywords TEXT"),
            ("role", "ALTER TABLE events ADD COLUMN role TEXT"),
            ("status", "ALTER TABLE events ADD COLUMN status TEXT"),
            ("salience", "ALTER TABLE events ADD COLUMN salience REAL"),
            ("first_seen_step", "ALTER TABLE events ADD COLUMN first_seen_step INTEGER"),
            ("last_seen_step", "ALTER TABLE events ADD COLUMN last_seen_step INTEGER"),
        ):
            if col not in event_names:
                self.conn.execute(ddl)

        detail_cols = self.conn.execute("PRAGMA table_info(details)").fetchall()
        detail_names = {str(row["name"]) for row in detail_cols}
        for col, ddl in (
            ("kind", "ALTER TABLE details ADD COLUMN kind TEXT"),
            ("text", "ALTER TABLE details ADD COLUMN text TEXT"),
            ("created_step", "ALTER TABLE details ADD COLUMN created_step INTEGER"),
        ):
            if col not in detail_names:
                self.conn.execute(ddl)

        self.conn.commit()

    @staticmethod
    def _arr_to_blob(arr: np.ndarray) -> bytes:
        return arr.astype(np.float32).tobytes()

    def blob_to_arr(self, blob: bytes | None) -> np.ndarray:
        if not blob:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size == self.embedding_dim:
            return arr
        out = np.zeros(self.embedding_dim, dtype=np.float32)
        n = min(arr.size, self.embedding_dim)
        out[:n] = arr[:n]
        return out

    def load_current_step(self) -> int:
        row = self.conn.execute("SELECT value FROM meta WHERE key='current_step'").fetchone()
        if not row:
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

    def upsert_event(
        self,
        event_id: str,
        fact_key: str,
        skeleton_text: str,
        skeleton_embedding: np.ndarray,
        keywords: List[str],
        role: str,
        current_step: int,
    ) -> None:
        row = self.conn.execute(
            "SELECT salience FROM events WHERE event_id=?",
            (event_id,),
        ).fetchone()
        if row is None:
            self.conn.execute(
                """
                INSERT INTO events(
                  event_id, fact_key, skeleton_text, skeleton_embedding, keywords,
                  role, status, salience, first_seen_step, last_seen_step
                ) VALUES(?, ?, ?, ?, ?, ?, 'active', 1.0, ?, ?)
                """,
                (
                    event_id,
                    fact_key,
                    skeleton_text,
                    self._arr_to_blob(skeleton_embedding),
                    json.dumps(keywords),
                    role,
                    int(current_step),
                    int(current_step),
                ),
            )
            return
        self.conn.execute(
            """
            UPDATE events
            SET fact_key = ?,
                skeleton_text = ?,
                skeleton_embedding = ?,
                keywords = ?,
                role = ?,
                salience = ?,
                last_seen_step = ?,
                status='active'
            WHERE event_id = ?
            """,
            (
                fact_key,
                skeleton_text,
                self._arr_to_blob(skeleton_embedding),
                json.dumps(keywords),
                role,
                float(row["salience"] or 0.0) + 1.0,
                int(current_step),
                event_id,
            ),
        )

    def insert_detail(
        self,
        detail_id: str,
        event_id: str,
        kind: str,
        text: str,
        current_step: int,
        max_per_event: int,
    ) -> None:
        if max_per_event > 0:
            row = self.conn.execute(
                "SELECT COUNT(*) AS cnt FROM details WHERE event_id=?",
                (event_id,),
            ).fetchone()
            existing = int(row["cnt"] if row else 0)
            if existing >= int(max_per_event):
                return
        self.conn.execute(
            """
            INSERT OR IGNORE INTO details(detail_id, event_id, kind, text, created_step)
            VALUES(?, ?, ?, ?, ?)
            """,
            (detail_id, event_id, kind, text, int(current_step)),
        )

    def resolve_conflicts(self) -> None:
        rows = self.conn.execute(
            """
            SELECT fact_key, COUNT(*) AS cnt
            FROM events
            WHERE status='active' AND fact_key <> ''
            GROUP BY fact_key
            HAVING COUNT(*) > 1
            """
        ).fetchall()
        for row in rows:
            fact_key = str(row["fact_key"])
            keep = self.conn.execute(
                """
                SELECT event_id
                FROM events
                WHERE fact_key=? AND status='active'
                ORDER BY last_seen_step DESC, salience DESC
                LIMIT 1
                """,
                (fact_key,),
            ).fetchone()
            if not keep:
                continue
            keep_id = str(keep["event_id"])
            self.conn.execute(
                """
                UPDATE events
                SET status='superseded'
                WHERE fact_key=? AND status='active' AND event_id<>?
                """,
                (fact_key, keep_id),
            )

    def apply_forgetting(
        self,
        current_step: int,
        forget_decay: float,
        forget_threshold: float,
        forget_min_age_steps: int,
        max_events: int,
    ) -> None:
        self.conn.execute(
            "UPDATE events SET salience = salience * ? WHERE status='active'",
            (float(forget_decay),),
        )
        self.conn.execute(
            """
            DELETE FROM details
            WHERE event_id IN (
              SELECT event_id FROM events
              WHERE status='active'
                AND salience < ?
                AND (? - last_seen_step) > ?
            )
            """,
            (float(forget_threshold), int(current_step), int(forget_min_age_steps)),
        )
        self.conn.execute(
            """
            DELETE FROM events
            WHERE status='active'
              AND salience < ?
              AND (? - last_seen_step) > ?
            """,
            (float(forget_threshold), int(current_step), int(forget_min_age_steps)),
        )

        row = self.conn.execute("SELECT COUNT(*) AS cnt FROM events WHERE status='active'").fetchone()
        total = int(row["cnt"] if row else 0)
        if total <= int(max_events):
            return
        overflow = total - int(max_events)
        old_rows = self.conn.execute(
            """
            SELECT event_id
            FROM events
            WHERE status='active'
            ORDER BY salience ASC, last_seen_step ASC
            LIMIT ?
            """,
            (int(overflow),),
        ).fetchall()
        for r in old_rows:
            event_id = str(r["event_id"])
            self.conn.execute("DELETE FROM details WHERE event_id=?", (event_id,))
            self.conn.execute("DELETE FROM events WHERE event_id=?", (event_id,))

    def fetch_active_events(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT event_id, skeleton_text, skeleton_embedding, keywords, role, salience, last_seen_step
            FROM events
            WHERE status='active'
            """
        ).fetchall()

    def fetch_event_details(self, event_id: str, limit: int) -> List[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT kind, text
            FROM details
            WHERE event_id=?
            ORDER BY created_step DESC
            LIMIT ?
            """,
            (event_id, int(limit)),
        ).fetchall()

    def debug_counts(self) -> Dict[str, int]:
        event_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM events").fetchone()
        detail_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM details").fetchone()
        active_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM events WHERE status='active'").fetchone()
        superseded_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM events WHERE status='superseded'").fetchone()
        return {
            "events": int(event_row["cnt"] if event_row else 0),
            "details": int(detail_row["cnt"] if detail_row else 0),
            "active_events": int(active_row["cnt"] if active_row else 0),
            "superseded_events": int(superseded_row["cnt"] if superseded_row else 0),
        }

    def clear_all(self) -> None:
        self.conn.execute("DELETE FROM details")
        self.conn.execute("DELETE FROM events")

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
