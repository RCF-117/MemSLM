"""Read/query mixin for LongMemoryStore."""

from __future__ import annotations

import sqlite3
from typing import Dict, List


class LongMemoryStoreReadMixin:
    """SQLite read operations for long-memory persistence."""

    def fetch_active_events(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT event_id, fact_key, skeleton_text, skeleton_embedding, keywords, role, salience, last_seen_step, fact_type
            FROM events
            WHERE status='active'
            """
        ).fetchall()

    def fetch_superseded_events(self, limit: int = 0) -> List[sqlite3.Row]:
        if int(limit) > 0:
            return self.conn.execute(
                """
                SELECT event_id, fact_key, skeleton_text, skeleton_embedding, keywords, role, salience, last_seen_step, fact_type
                FROM events
                WHERE status='superseded'
                ORDER BY last_seen_step DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return self.conn.execute(
            """
            SELECT event_id, fact_key, skeleton_text, skeleton_embedding, keywords, role, salience, last_seen_step, fact_type
            FROM events
            WHERE status='superseded'
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

    def fetch_events_by_ids(self, event_ids: List[str]) -> List[sqlite3.Row]:
        if not event_ids:
            return []
        marks = ",".join("?" for _ in event_ids)
        return self.conn.execute(
            f"""
            SELECT event_id, fact_key, skeleton_text, skeleton_embedding, keywords, role, salience, last_seen_step, fact_type
            FROM events
            WHERE status='active' AND event_id IN ({marks})
            """,
            tuple(event_ids),
        ).fetchall()

    def debug_counts(self) -> Dict[str, int]:
        event_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM events").fetchone()
        detail_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM details").fetchone()
        edge_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM edges").fetchone()
        node_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM event_nodes").fetchone()
        node_edge_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM event_node_edges").fetchone()
        staging_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM events_staging").fetchone()
        active_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM events WHERE status='active'").fetchone()
        superseded_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM events WHERE status='superseded'").fetchone()
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
