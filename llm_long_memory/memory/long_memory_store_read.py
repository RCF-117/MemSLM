"""Read/query mixin for LongMemoryStore."""

from __future__ import annotations

import sqlite3
from typing import Dict, List


class LongMemoryStoreReadMixin:
    """SQLite read operations for long-memory persistence."""

    def fetch_recent_events(
        self,
        limit: int,
        exclude_event_id: str | None = None,
    ) -> List[sqlite3.Row]:
        params: List[object] = []
        sql = """
            SELECT event_id, fact_key, subject_action_key, fact_type, skeleton_text,
                   keywords, role, boundary_flag, extract_confidence, source_model,
                   raw_span, status, is_latest, salience, first_seen_step, last_seen_step
            FROM events
        """
        where_clauses = []
        if exclude_event_id:
            where_clauses.append("event_id <> ?")
            params.append(str(exclude_event_id))
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += " ORDER BY is_latest DESC, last_seen_step DESC, salience DESC"
        if int(limit) > 0:
            sql += " LIMIT ?"
            params.append(int(limit))
        return self.conn.execute(sql, tuple(params)).fetchall()

    def fetch_active_events(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT event_id, fact_key, skeleton_text, skeleton_embedding, keywords, role, salience, last_seen_step, fact_type
            FROM events
            WHERE is_latest=1
            """
        ).fetchall()

    def fetch_superseded_events(self, limit: int = 0) -> List[sqlite3.Row]:
        if int(limit) > 0:
            return self.conn.execute(
                """
                SELECT event_id, fact_key, skeleton_text, skeleton_embedding, keywords, role, salience, last_seen_step, fact_type
                FROM events
                WHERE is_latest=0
                ORDER BY last_seen_step DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return self.conn.execute(
            """
            SELECT event_id, fact_key, skeleton_text, skeleton_embedding, keywords, role, salience, last_seen_step, fact_type
            FROM events
            WHERE is_latest=0
            """
        ).fetchall()

    def fetch_event_details(self, event_id: str, limit: int) -> List[sqlite3.Row]:
        if int(limit) <= 0:
            return self.conn.execute(
                """
                SELECT kind, text
                FROM details
                WHERE event_id=?
                ORDER BY created_step DESC
                """,
                (event_id,),
            ).fetchall()
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

    def fetch_event_nodes(self, event_id: str, limit: int = 0) -> List[sqlite3.Row]:
        if int(limit) > 0:
            return self.conn.execute(
                """
                SELECT node_id, event_id, node_kind, node_text, is_core, node_embedding, created_step
                FROM event_nodes
                WHERE event_id=?
                ORDER BY is_core DESC, created_step DESC
                LIMIT ?
                """,
                (event_id, int(limit)),
            ).fetchall()
        return self.conn.execute(
            """
            SELECT node_id, event_id, node_kind, node_text, is_core, node_embedding, created_step
            FROM event_nodes
            WHERE event_id=?
            ORDER BY is_core DESC, created_step DESC
            """,
            (event_id,),
        ).fetchall()

    def fetch_event_node_edges(self, event_id: str, limit: int = 0) -> List[sqlite3.Row]:
        if int(limit) > 0:
            return self.conn.execute(
                """
                SELECT e.relation, e.weight, fn.node_text AS from_text, tn.node_text AS to_text
                FROM event_node_edges e
                LEFT JOIN event_nodes fn ON fn.node_id=e.from_node_id
                LEFT JOIN event_nodes tn ON tn.node_id=e.to_node_id
                WHERE e.event_id=?
                ORDER BY e.weight DESC, e.created_step DESC
                LIMIT ?
                """,
                (event_id, int(limit)),
            ).fetchall()
        return self.conn.execute(
            """
            SELECT e.relation, e.weight, fn.node_text AS from_text, tn.node_text AS to_text
            FROM event_node_edges e
            LEFT JOIN event_nodes fn ON fn.node_id=e.from_node_id
            LEFT JOIN event_nodes tn ON tn.node_id=e.to_node_id
            WHERE e.event_id=?
            ORDER BY e.weight DESC, e.created_step DESC
            """,
            (event_id,),
        ).fetchall()

    def fetch_edges_from(self, event_id: str, limit: int) -> List[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT from_event_id, to_event_id, relation, weight
            FROM edges
            WHERE from_event_id=?
            ORDER BY weight DESC, created_step DESC
            LIMIT ?
            """,
            (event_id, int(limit)),
        ).fetchall()

    def fetch_edges_to(self, event_id: str, limit: int) -> List[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT from_event_id, to_event_id, relation, weight
            FROM edges
            WHERE to_event_id=?
            ORDER BY weight DESC, created_step DESC
            LIMIT ?
            """,
            (event_id, int(limit)),
        ).fetchall()

    def fetch_event_by_id(self, event_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT event_id, fact_key, skeleton_text, skeleton_embedding, keywords, role, salience, last_seen_step, fact_type
            FROM events
            WHERE event_id=?
            LIMIT 1
            """,
            (event_id,),
        ).fetchone()

    def search_event_nodes(self, query_text: str, limit: int = 24) -> List[sqlite3.Row]:
        q = str(query_text or "").strip()
        if not q:
            return []
        pattern = f"%{q}%"
        return self.conn.execute(
            """
            SELECT n.event_id, n.node_kind, n.node_text, n.is_core, n.created_step
            FROM event_nodes n
            JOIN events e ON e.event_id=n.event_id
            WHERE e.is_latest=1 AND lower(n.node_text) LIKE lower(?)
            ORDER BY n.is_core DESC, n.created_step DESC
            LIMIT ?
            """,
            (pattern, int(limit)),
        ).fetchall()

    def fetch_events_by_ids(self, event_ids: List[str]) -> List[sqlite3.Row]:
        if not event_ids:
            return []
        marks = ",".join("?" for _ in event_ids)
        return self.conn.execute(
            f"""
            SELECT event_id, fact_key, skeleton_text, skeleton_embedding, keywords, role, salience, last_seen_step, fact_type
            FROM events
            WHERE is_latest=1 AND event_id IN ({marks})
            """,
            tuple(event_ids),
        ).fetchall()

    def fetch_all_events(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT event_id, fact_key, subject_action_key, fact_type, skeleton_text,
                   skeleton_embedding, keywords, role, boundary_flag, extract_confidence,
                   source_model, raw_span, status, is_latest, salience, first_seen_step, last_seen_step
            FROM events
            ORDER BY is_latest DESC, last_seen_step DESC, salience DESC
            """
        ).fetchall()

    def fetch_all_details(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT detail_id, event_id, kind, text, created_step
            FROM details
            ORDER BY created_step DESC
            """
        ).fetchall()

    def fetch_all_event_nodes(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT node_id, event_id, node_kind, node_text, is_core, node_embedding, created_step
            FROM event_nodes
            ORDER BY created_step DESC
            """
        ).fetchall()

    def fetch_all_event_node_edges(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT node_edge_id, event_id, from_node_id, to_node_id, relation, weight, created_step
            FROM event_node_edges
            ORDER BY created_step DESC
            """
        ).fetchall()

    def fetch_all_edges(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT edge_id, from_event_id, to_event_id, relation, weight, created_step
            FROM edges
            ORDER BY created_step DESC
            """
        ).fetchall()

    def debug_counts(self) -> Dict[str, int]:
        event_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM events").fetchone()
        detail_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM details").fetchone()
        edge_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM edges").fetchone()
        node_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM event_nodes").fetchone()
        node_edge_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM event_node_edges").fetchone()
        staging_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM events_staging").fetchone()
        active_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM events WHERE is_latest=1").fetchone()
        superseded_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM events WHERE is_latest=0").fetchone()
        return {
            "events": int(event_row["cnt"] if event_row else 0),
            "event_nodes": int(node_row["cnt"] if node_row else 0),
            "event_node_edges": int(node_edge_row["cnt"] if node_edge_row else 0),
            "details": int(detail_row["cnt"] if detail_row else 0),
            "relations": int(edge_row["cnt"] if edge_row else 0),
            "staging_events": int(staging_row["cnt"] if staging_row else 0),
            "active_events": int(active_row["cnt"] if active_row else 0),
            "superseded_events": int(superseded_row["cnt"] if superseded_row else 0),
        }
