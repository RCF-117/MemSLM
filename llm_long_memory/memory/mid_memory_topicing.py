"""Topic assignment and lifecycle helpers for MidMemory."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Protocol, Sequence

import numpy as np

from llm_long_memory.utils.logger import logger


class TopicOwner(Protocol):
    conn: Any
    topic_similarity_threshold: float
    non_user_threshold_multiplier: float
    current_step: int
    topic_multi_assign_top_k: int
    embedding_dim: int
    centroid_roles: set[str]
    centroid_role_weights: Dict[str, float]
    topic_min_chunks_before_merge: int
    topic_merge_threshold: float
    enable_topic_merge: bool
    topic_inactive_steps: int
    max_size: int

    def _blob_to_arr(self, blob: bytes | None) -> np.ndarray: ...
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float: ...
    def _temporal_weight(self, delta: int) -> float: ...
    def _arr_to_blob(self, arr: np.ndarray) -> bytes: ...
    def _normalize_token(self, token: str) -> str: ...
    def _delete_chunk_fts(self, chunk_ids: Sequence[int]) -> None: ...
    def _rebuild_chunk_fts(self) -> None: ...


def assign_topics(owner: TopicOwner, chunk_embedding: np.ndarray, chunk_role: str) -> List[str]:
    """Assign chunk to top-K topics using semantic similarity and thresholds."""
    rows = owner.conn.execute(
        """
        SELECT
          t.topic_id,
          t.topic_embedding,
          t.last_updated_step
        FROM topics t
        """
    ).fetchall()
    similarity_threshold = owner.topic_similarity_threshold
    if chunk_role != "user":
        similarity_threshold *= owner.non_user_threshold_multiplier

    scored: List[tuple[str, float, float]] = []
    for row in rows:
        topic_id = str(row["topic_id"])
        topic_embedding = owner._blob_to_arr(row["topic_embedding"])
        sim_semantic = owner._cosine(chunk_embedding, topic_embedding)
        delta = owner.current_step - int(row["last_updated_step"] or 0)
        temporal_weight = owner._temporal_weight(delta)
        sim_rank = sim_semantic * temporal_weight
        scored.append((topic_id, sim_rank, sim_semantic))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: owner.topic_multi_assign_top_k]
    return [tid for tid, _rank, sem in top if sem >= similarity_threshold]


def create_topic(owner: TopicOwner, topic_embedding: np.ndarray) -> str:
    """Create a new topic row and return topic id."""
    topic_id = f"topic_{owner.current_step}_{uuid.uuid4().hex[:8]}"
    owner.conn.execute(
        """
        INSERT INTO topics(
          topic_id, topic_embedding, keywords, topic_times, last_updated_step, active
        )
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        (
            topic_id,
            owner._arr_to_blob(topic_embedding),
            json.dumps([]),
            json.dumps([]),
            owner.current_step,
            1,
        ),
    )
    return topic_id


def recompute_topic_embedding(owner: TopicOwner, topic_id: str) -> None:
    """Recompute topic centroid and topic time terms from chunk members."""
    rows = owner.conn.execute(
        "SELECT chunk_embedding, chunk_role, chunk_times FROM chunks WHERE topic_id = ?",
        (topic_id,),
    ).fetchall()
    if not rows:
        owner.conn.execute("UPDATE topics SET active = 0 WHERE topic_id = ?", (topic_id,))
        return
    centroid_rows = [
        row
        for row in rows
        if str(row["chunk_role"] or "user").strip().lower() in owner.centroid_roles
    ]
    source_rows = centroid_rows if centroid_rows else rows
    weighted_vectors: List[np.ndarray] = []
    weights: List[float] = []
    for row in source_rows:
        role = str(row["chunk_role"] or "user").strip().lower()
        weight = float(owner.centroid_role_weights.get(role, 1.0))
        if weight <= 0.0:
            continue
        weighted_vectors.append(owner._blob_to_arr(row["chunk_embedding"]))
        weights.append(weight)
    if weighted_vectors and weights:
        matrix = np.stack(weighted_vectors)
        mean_emb = np.average(matrix, axis=0, weights=np.asarray(weights, dtype=np.float32)).astype(
            np.float32
        )
    else:
        vectors = [owner._blob_to_arr(row["chunk_embedding"]) for row in rows]
        mean_emb = np.mean(np.stack(vectors), axis=0).astype(np.float32)
    times: List[str] = []
    for row in rows:
        try:
            chunk_times = json.loads(row["chunk_times"] or "[]")
        except (TypeError, ValueError):
            chunk_times = []
        for t in chunk_times:
            tt = str(t).strip().lower()
            if tt and tt not in times:
                times.append(tt)
    owner.conn.execute(
        "UPDATE topics SET topic_embedding = ?, topic_times = ? WHERE topic_id = ?",
        (owner._arr_to_blob(mean_emb), json.dumps(times), topic_id),
    )


def merge_keywords(owner: TopicOwner, left: Sequence[str], right: Sequence[str]) -> List[str]:
    """Merge keyword lists with normalization and deduplication."""
    merged: List[str] = []
    for tok in list(left) + list(right):
        norm = owner._normalize_token(str(tok))
        if norm and norm not in merged:
            merged.append(norm)
    return merged


def deduplicate_topic_chunks(owner: TopicOwner, topic_id: str) -> None:
    """Remove duplicate chunks inside one topic."""
    owner.conn.execute(
        """
        DELETE FROM chunks
        WHERE chunk_id IN (
          SELECT c.chunk_id
          FROM chunks c
          JOIN (
            SELECT MIN(chunk_id) AS keep_id, text, chunk_role, hex(chunk_embedding) AS emb_hex
            FROM chunks
            WHERE topic_id = ?
            GROUP BY text, chunk_role, emb_hex
          ) d
            ON c.text = d.text
           AND c.chunk_role = d.chunk_role
           AND hex(c.chunk_embedding) = d.emb_hex
          WHERE c.topic_id = ?
            AND c.chunk_id <> d.keep_id
        )
        """,
        (topic_id, topic_id),
    )
    owner._rebuild_chunk_fts()


def merge_topic_pair(owner: TopicOwner, target_topic_id: str, source_topic_id: str) -> None:
    """Merge source topic into target topic and update metadata."""
    owner.conn.execute(
        "UPDATE chunks SET topic_id = ? WHERE topic_id = ?",
        (target_topic_id, source_topic_id),
    )
    deduplicate_topic_chunks(owner, target_topic_id)

    source = owner.conn.execute(
        "SELECT keywords, last_updated_step FROM topics WHERE topic_id = ?",
        (source_topic_id,),
    ).fetchone()
    target = owner.conn.execute(
        "SELECT keywords, last_updated_step FROM topics WHERE topic_id = ?",
        (target_topic_id,),
    ).fetchone()
    if source and target:
        merged_keywords = merge_keywords(
            owner,
            json.loads(target["keywords"] or "[]"),
            json.loads(source["keywords"] or "[]"),
        )
        merged_step = max(
            int(target["last_updated_step"] or 0),
            int(source["last_updated_step"] or 0),
        )
        owner.conn.execute(
            """
            UPDATE topics
            SET keywords = ?, last_updated_step = ?, active = 1
            WHERE topic_id = ?
            """,
            (
                json.dumps(merged_keywords),
                merged_step,
                target_topic_id,
            ),
        )
    owner.conn.execute("DELETE FROM topics WHERE topic_id = ?", (source_topic_id,))
    recompute_topic_embedding(owner, target_topic_id)


def merge_topics(owner: TopicOwner) -> None:
    """Merge highly similar topics when merge feature is enabled."""
    if not owner.enable_topic_merge:
        return

    rows = owner.conn.execute("SELECT topic_id, topic_embedding FROM topics").fetchall()
    topics = [(str(r["topic_id"]), owner._blob_to_arr(r["topic_embedding"])) for r in rows]
    chunk_count_rows = owner.conn.execute(
        "SELECT topic_id, COUNT(*) AS cnt FROM chunks GROUP BY topic_id"
    ).fetchall()
    chunk_counts = {str(r["topic_id"]): int(r["cnt"]) for r in chunk_count_rows}
    removed: set[str] = set()
    for i in range(len(topics)):
        left_id, left_emb = topics[i]
        if left_id in removed:
            continue
        if chunk_counts.get(left_id, 0) < owner.topic_min_chunks_before_merge:
            continue
        for j in range(i + 1, len(topics)):
            right_id, right_emb = topics[j]
            if right_id in removed:
                continue
            if chunk_counts.get(right_id, 0) < owner.topic_min_chunks_before_merge:
                continue
            if owner._cosine(left_emb, right_emb) > owner.topic_merge_threshold:
                merge_topic_pair(owner, left_id, right_id)
                removed.add(right_id)
    if removed:
        owner._rebuild_chunk_fts()


def deactivate_topics(owner: TopicOwner) -> None:
    """Deactivate stale topics by inactivity step threshold."""
    threshold = owner.current_step - owner.topic_inactive_steps
    owner.conn.execute(
        "UPDATE topics SET active = 0 WHERE last_updated_step < ?",
        (threshold,),
    )


def enforce_fifo_limit(owner: TopicOwner) -> None:
    """Apply global FIFO deletion when chunk count exceeds max_size."""
    row = owner.conn.execute("SELECT COUNT(*) AS cnt FROM chunks").fetchone()
    total = int(row["cnt"]) if row else 0
    overflow = total - owner.max_size
    if overflow <= 0:
        return
    old = owner.conn.execute(
        "SELECT chunk_id, topic_id FROM chunks ORDER BY chunk_id ASC LIMIT ?",
        (overflow,),
    ).fetchall()
    old_ids = [int(r["chunk_id"]) for r in old]
    topics = {str(r["topic_id"]) for r in old}
    owner.conn.executemany("DELETE FROM chunks WHERE chunk_id = ?", [(cid,) for cid in old_ids])
    owner._delete_chunk_fts(old_ids)
    for topic_id in topics:
        recompute_topic_embedding(owner, topic_id)
        left = owner.conn.execute(
            "SELECT COUNT(*) AS cnt FROM chunks WHERE topic_id = ?",
            (topic_id,),
        ).fetchone()
        if int(left["cnt"]) == 0:
            owner.conn.execute("DELETE FROM topics WHERE topic_id = ?", (topic_id,))
    logger.warn(f"MidMemory._enforce_fifo_limit: removed {overflow} chunks.")
