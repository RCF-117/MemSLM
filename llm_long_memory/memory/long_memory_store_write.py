"""Write/update mixin for LongMemoryStore."""

from __future__ import annotations

import json
import re
from typing import List

import numpy as np


class LongMemoryStoreWriteMixin:
    """SQLite write operations for long-memory persistence."""

    def upsert_event(
        self,
        event_id: str,
        fact_key: str,
        subject_action_key: str,
        fact_type: str,
        skeleton_text: str,
        skeleton_embedding: np.ndarray,
        keywords: List[str],
        role: str,
        boundary_flag: int,
        extract_confidence: float,
        source_model: str,
        raw_span: str,
        current_step: int,
    ) -> None:
        row = self.conn.execute("SELECT event_id FROM events WHERE event_id=?", (event_id,)).fetchone()
        if row is None:
            self.conn.execute(
                """
                INSERT INTO events(
                  event_id, fact_key, subject_action_key, fact_type, skeleton_text, skeleton_embedding, keywords,
                  role, boundary_flag, extract_confidence, source_model, raw_span,
                  is_latest, salience, first_seen_step, last_seen_step
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1.0, ?, ?)
                """,
                (
                    event_id,
                    fact_key,
                    subject_action_key,
                    fact_type,
                    skeleton_text,
                    self._arr_to_blob(skeleton_embedding),
                    json.dumps(keywords, ensure_ascii=False),
                    role,
                    int(boundary_flag),
                    float(extract_confidence),
                    source_model,
                    raw_span,
                    int(current_step),
                    int(current_step),
                ),
            )
            return
        self.conn.execute(
            """
            UPDATE events
            SET fact_key=?, subject_action_key=?, fact_type=?, skeleton_text=?, skeleton_embedding=?,
                keywords=?, role=?, boundary_flag=?, extract_confidence=?, source_model=?, raw_span=?,
                is_latest=1, salience=salience+1.0, last_seen_step=?
            WHERE event_id=?
            """,
            (
                fact_key,
                subject_action_key,
                fact_type,
                skeleton_text,
                self._arr_to_blob(skeleton_embedding),
                json.dumps(keywords, ensure_ascii=False),
                role,
                int(boundary_flag),
                float(extract_confidence),
                source_model,
                raw_span,
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
        if int(max_per_event) > 0:
            row = self.conn.execute(
                "SELECT COUNT(*) AS cnt FROM details WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if int(row["cnt"] if row else 0) >= int(max_per_event):
                return
        self.conn.execute(
            """
            INSERT OR IGNORE INTO details(detail_id, event_id, kind, text, created_step)
            VALUES(?, ?, ?, ?, ?)
            """,
            (detail_id, event_id, kind, text, int(current_step)),
        )

    def insert_edge(
        self,
        edge_id: str,
        from_event_id: str,
        to_event_id: str,
        relation: str,
        weight: float,
        current_step: int,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO edges(edge_id, from_event_id, to_event_id, relation, weight, created_step)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                edge_id,
                from_event_id,
                to_event_id,
                relation,
                float(weight),
                int(current_step),
            ),
        )

    def upsert_event_node(
        self,
        *,
        node_id: str,
        event_id: str,
        node_kind: str,
        node_text: str,
        is_core: bool,
        node_embedding: np.ndarray,
        current_step: int,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO event_nodes(
              node_id, event_id, node_kind, node_text, is_core, node_embedding, created_step
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node_id,
                event_id,
                node_kind,
                node_text,
                int(bool(is_core)),
                self._arr_to_blob(node_embedding),
                int(current_step),
            ),
        )

    def upsert_event_node_edge(
        self,
        *,
        node_edge_id: str,
        event_id: str,
        from_node_id: str,
        to_node_id: str,
        relation: str,
        weight: float,
        current_step: int,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO event_node_edges(
              node_edge_id, event_id, from_node_id, to_node_id, relation, weight, created_step
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node_edge_id,
                event_id,
                from_node_id,
                to_node_id,
                relation,
                float(weight),
                int(current_step),
            ),
        )

    def clear_event_node_edges_for_event(self, event_id: str) -> None:
        self.conn.execute(
            "DELETE FROM event_node_edges WHERE event_id=?",
            (event_id,),
        )

    def insert_staging_event(
        self,
        *,
        staging_id: str,
        fact_key: str,
        skeleton_text: str,
        skeleton_embedding: np.ndarray,
        keywords: List[str],
        role: str,
        extract_confidence: float,
        source_model: str,
        raw_span: str,
        source_content: str,
        reject_reason: str,
        current_step: int,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO events_staging(
              staging_id, fact_key, skeleton_text, skeleton_embedding, keywords, role,
              extract_confidence, source_model, raw_span, source_content, reject_reason, created_step
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                staging_id,
                fact_key,
                skeleton_text,
                self._arr_to_blob(skeleton_embedding),
                json.dumps(keywords, ensure_ascii=False),
                role,
                float(extract_confidence),
                source_model,
                raw_span,
                source_content,
                reject_reason,
                int(current_step),
            ),
        )

    def supersede_active_by_fact_key(
        self,
        *,
        fact_key: str,
        keep_event_id: str,
        new_time_text: str,
        new_raw_span: str,
        min_evidence_overlap: float,
        current_step: int,
    ) -> List[str]:
        if not fact_key:
            return []
        normalized_new_time = str(new_time_text).strip().lower()
        if not normalized_new_time:
            return []

        def norm_tokens(text: str) -> set[str]:
            return {
                t
                for t in re.findall(r"[a-z0-9]+", str(text).lower())
                if t
            }

        new_span_tokens = norm_tokens(new_raw_span)
        rows = self.conn.execute(
            """
            SELECT event_id, raw_span
            FROM events
            WHERE is_latest=1 AND fact_key=? AND event_id<>?
            """,
            (fact_key, keep_event_id),
        ).fetchall()
        old_ids: List[str] = []
        for row in rows:
            event_id = str(row["event_id"]).strip()
            if not event_id:
                continue
            time_row = self.conn.execute(
                """
                SELECT text
                FROM details
                WHERE event_id=? AND kind='time'
                ORDER BY created_step DESC
                LIMIT 1
                """,
                (event_id,),
            ).fetchone()
            old_time = str(time_row["text"]).strip().lower() if time_row else ""
            if (not old_time) or (old_time == normalized_new_time):
                continue
            if new_span_tokens:
                old_tokens = norm_tokens(str(row["raw_span"] or ""))
                if old_tokens:
                    overlap = float(len(new_span_tokens.intersection(old_tokens))) / float(
                        max(1, len(new_span_tokens))
                    )
                    if overlap < float(min_evidence_overlap):
                        continue
            old_ids.append(event_id)
        if old_ids:
            marks = ",".join("?" for _ in old_ids)
            self.conn.execute(
                f"""
                UPDATE events
                SET is_latest=0, last_seen_step=?
                WHERE event_id IN ({marks})
                """,
                (int(current_step), *old_ids),
            )
        return old_ids
