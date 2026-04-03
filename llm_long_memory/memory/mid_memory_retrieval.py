"""Retrieval helpers for MidMemory (topic search + chunk rerank)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Protocol, Sequence

from llm_long_memory.utils.embedding import embed

Topic = Dict[str, Any]
Chunk = Dict[str, Any]


class RetrievalOwner(Protocol):
    embedding_dim: int
    retrieval_diversity_enabled: bool
    top_k: int
    retrieval_diversity_similarity_threshold: float
    summary_enabled: bool
    hybrid_alpha: float
    keyword_weight: float
    summary_weight: float
    time_weight: float
    temporal_boost_max: float
    recent_topic_apply_margin: float
    recent_topic_weight: float
    chunks_per_topic: int
    fusion_dense_weight: float
    fusion_rrf_k: int
    fusion_lexical_weight: float
    fusion_topic_weight: float
    global_chunk_top_n: int
    global_chunk_rrf_k: int
    global_chunk_dense_weight: float
    global_chunk_lexical_weight: float
    global_chunk_keyword_weight: float
    global_chunk_topic_prior_weight: float
    global_chunk_prefilter_enabled: bool
    global_chunk_prefilter_top_n: int
    global_chunk_lexical_prefilter_top_n: int
    global_chunk_prefilter_topic_top_k: int
    global_chunk_recent_fallback_n: int
    global_chunk_dedup_by_text: bool
    current_step: int
    conn: Any

    def _cosine(self, a: Any, b: Any) -> float: ...
    def _tokenize(self, text: str) -> List[str]: ...
    def _extract_time_terms(self, text: str) -> List[str]: ...
    def _blob_to_arr(self, blob: bytes | None): ...
    def _temporal_weight(self, delta: int) -> float: ...
    def _lexical_rank_map(self, topic_id: str, query: str) -> Dict[int, int]: ...
    def _lexical_rank_map_global(self, query: str, top_n: int) -> Dict[int, int]: ...


def keyword_overlap(query_tokens: Sequence[str], target_tokens: Sequence[str]) -> float:
    """Compute keyword overlap ratio between query and target tokens."""
    if not query_tokens:
        return 0.0
    target = {str(x).lower() for x in target_tokens}
    hit = sum(1 for q in query_tokens if q in target)
    return float(hit) / float(len(query_tokens))


def select_diverse_topics(owner: RetrievalOwner, scored: Sequence[Topic]) -> List[Topic]:
    """Select topic list with optional embedding diversity filter."""
    if not owner.retrieval_diversity_enabled:
        return list(scored[: owner.top_k])
    selected: List[Topic] = []
    deferred: List[Topic] = []
    for topic in scored:
        emb = topic.get("_topic_embedding")
        if emb is None:
            deferred.append(topic)
            continue
        is_diverse = True
        for chosen in selected:
            chosen_emb = chosen.get("_topic_embedding")
            if chosen_emb is not None:
                if owner._cosine(emb, chosen_emb) >= owner.retrieval_diversity_similarity_threshold:
                    is_diverse = False
                    break
        if is_diverse:
            selected.append(topic)
        else:
            deferred.append(topic)
        if len(selected) >= owner.top_k:
            break
    if len(selected) < owner.top_k:
        for topic in deferred:
            if len(selected) >= owner.top_k:
                break
            selected.append(topic)
    return selected[: owner.top_k]


def search_topics(owner: RetrievalOwner, query: str) -> List[Topic]:
    """Hybrid level-1 retrieval over topics."""
    q = query.strip()
    if not q:
        return []
    q_emb = embed(q, owner.embedding_dim)
    q_tokens = owner._tokenize(q)
    q_times = owner._extract_time_terms(q)

    rows = owner.conn.execute(
        """
        SELECT topic_id, topic_embedding, summary_embedding, keywords, topic_times, last_updated_step, active
        FROM topics
        """
    ).fetchall()
    if not rows:
        return []

    newest = owner.conn.execute(
        "SELECT topic_id FROM topics ORDER BY last_updated_step DESC LIMIT 1"
    ).fetchone()
    recent_topic = str(newest["topic_id"]) if newest else ""

    scored: List[Topic] = []
    for row in rows:
        topic_id = str(row["topic_id"])
        topic_emb = owner._blob_to_arr(row["topic_embedding"])
        summary_emb = owner._blob_to_arr(row["summary_embedding"])
        keywords = json.loads(row["keywords"] or "[]")
        topic_times = json.loads(row["topic_times"] or "[]")
        last = int(row["last_updated_step"] or 0)

        embedding_score = owner._cosine(q_emb, topic_emb)
        keyword_score = keyword_overlap(q_tokens, keywords)
        summary_score = owner._cosine(q_emb, summary_emb) if owner.summary_enabled else 0.0
        time_score = keyword_overlap(q_times, topic_times)
        base_score = (
            owner.hybrid_alpha * embedding_score
            + owner.keyword_weight * keyword_score
            + owner.summary_weight * summary_score
            + owner.time_weight * time_score
        )
        temporal = owner._temporal_weight(owner.current_step - last)
        temporal = min(temporal, owner.temporal_boost_max)
        score = base_score * (1.0 + temporal)

        scored.append(
            {
                "topic_id": topic_id,
                "score": score,
                "base_score": base_score,
                "last_updated_step": last,
                "active": int(row["active"] or 0),
                "_topic_embedding": topic_emb,
            }
        )
    best_base_score = max((float(x["base_score"]) for x in scored), default=0.0)
    margin = best_base_score * owner.recent_topic_apply_margin
    for item in scored:
        if str(item["topic_id"]) != recent_topic:
            continue
        if (best_base_score - float(item["base_score"])) <= margin:
            item["score"] = float(item["score"]) * owner.recent_topic_weight
    scored.sort(key=lambda x: x["score"], reverse=True)
    selected = select_diverse_topics(owner, scored)
    for item in selected:
        item.pop("_topic_embedding", None)
        item.pop("base_score", None)
    return selected


def rerank_chunks(owner: RetrievalOwner, query: str, topics: Sequence[Topic]) -> List[Chunk]:
    """Hybrid level-2 chunk reranking by dense+lexical RRF."""
    q_emb = embed(query, owner.embedding_dim)
    selected: List[Chunk] = []
    topic_score_map: Dict[str, float] = {
        str(topic.get("topic_id", "")): float(topic.get("score", 0.0))
        for topic in topics
    }
    max_topic_score = max(topic_score_map.values(), default=0.0)

    for topic in topics:
        topic_id = str(topic["topic_id"])
        topic_score_raw = float(topic_score_map.get(topic_id, 0.0))
        topic_score_norm = max(0.0, topic_score_raw) / max_topic_score if max_topic_score > 0.0 else 0.0
        rows = owner.conn.execute(
            """
            SELECT chunk_id, text, chunk_embedding, chunk_role, chunk_session_id, chunk_session_date, chunk_has_answer, chunk_times
            FROM chunks
            WHERE topic_id = ?
            ORDER BY chunk_id ASC
            """,
            (topic_id,),
        ).fetchall()
        dense_ranked: List[tuple[int, float]] = []
        payload: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            chunk_id = int(row["chunk_id"])
            text = str(row["text"])
            role = str(row["chunk_role"] or "user").lower()
            session_id = str(row["chunk_session_id"] or "")
            session_date = str(row["chunk_session_date"] or "")
            has_answer = int(row["chunk_has_answer"] or 0)
            emb = owner._blob_to_arr(row["chunk_embedding"])
            emb_score = owner._cosine(q_emb, emb)
            dense_ranked.append((chunk_id, emb_score))
            payload[chunk_id] = {
                "topic_id": topic_id,
                "chunk_id": chunk_id,
                "role": role,
                "text": f"({role}) {text}",
                "session_id": session_id,
                "session_date": session_date,
                "has_answer": has_answer,
            }
        dense_ranked.sort(key=lambda x: x[1], reverse=True)
        dense_ranks = {chunk_id: rank + 1 for rank, (chunk_id, _s) in enumerate(dense_ranked)}
        lexical_ranks = owner._lexical_rank_map(topic_id, query)

        ranked: List[Chunk] = []
        for chunk_id, item in payload.items():
            score = 0.0
            dense_rank = dense_ranks.get(chunk_id)
            if dense_rank is not None:
                score += owner.fusion_dense_weight / float(owner.fusion_rrf_k + dense_rank)
            lexical_rank = lexical_ranks.get(chunk_id)
            if lexical_rank is not None:
                score += owner.fusion_lexical_weight / float(owner.fusion_rrf_k + lexical_rank)
            score += owner.fusion_topic_weight * topic_score_norm
            out: Chunk = dict(item)
            out["score"] = score
            ranked.append(out)
        ranked.sort(key=lambda x: float(x["score"]), reverse=True)
        selected.extend(ranked[: owner.chunks_per_topic])
    return selected


def rerank_chunks_global(
    owner: RetrievalOwner,
    query: str,
    topic_score_map: Dict[str, float],
) -> List[Chunk]:
    """Primary retrieval path: global chunk rerank without topic gating."""
    q = query.strip()
    if not q:
        return []
    q_emb = embed(q, owner.embedding_dim)
    q_tokens = owner._tokenize(q)
    lexical_ranks = owner._lexical_rank_map_global(
        query=q,
        top_n=owner.global_chunk_lexical_prefilter_top_n,
    )

    candidate_ids: set[int] = set()
    if owner.global_chunk_prefilter_enabled:
        candidate_ids.update(int(x) for x in lexical_ranks.keys())

        if topic_score_map and owner.global_chunk_prefilter_topic_top_k > 0:
            top_topic_ids = [
                str(tid)
                for tid, _score in sorted(
                    topic_score_map.items(),
                    key=lambda item: float(item[1]),
                    reverse=True,
                )[: owner.global_chunk_prefilter_topic_top_k]
            ]
            if top_topic_ids:
                placeholders = ",".join(["?"] * len(top_topic_ids))
                topic_rows = owner.conn.execute(
                    f"""
                    SELECT chunk_id
                    FROM chunks
                    WHERE topic_id IN ({placeholders})
                    ORDER BY chunk_id DESC
                    LIMIT ?
                    """,
                    (*top_topic_ids, int(owner.global_chunk_prefilter_top_n)),
                ).fetchall()
                candidate_ids.update(int(row["chunk_id"]) for row in topic_rows)

        if owner.global_chunk_recent_fallback_n > 0:
            recent_rows = owner.conn.execute(
                """
                SELECT chunk_id
                FROM chunks
                ORDER BY chunk_id DESC
                LIMIT ?
                """,
                (int(owner.global_chunk_recent_fallback_n),),
            ).fetchall()
            candidate_ids.update(int(row["chunk_id"]) for row in recent_rows)

    if owner.global_chunk_prefilter_enabled and candidate_ids:
        ids = sorted(candidate_ids)[: owner.global_chunk_prefilter_top_n]
        placeholders = ",".join(["?"] * len(ids))
        rows = owner.conn.execute(
            f"""
            SELECT chunk_id, topic_id, text, chunk_embedding, chunk_role, chunk_session_id,
                   chunk_session_date, chunk_has_answer
            FROM chunks
            WHERE chunk_id IN ({placeholders})
            """,
            tuple(ids),
        ).fetchall()
    else:
        rows = owner.conn.execute(
            """
            SELECT chunk_id, topic_id, text, chunk_embedding, chunk_role, chunk_session_id,
                   chunk_session_date, chunk_has_answer
            FROM chunks
            """
        ).fetchall()
    if not rows:
        return []

    dense_ranked: List[tuple[int, float]] = []
    payload: Dict[int, Dict[str, Any]] = {}
    max_topic_score = max((float(v) for v in topic_score_map.values()), default=0.0)

    for row in rows:
        chunk_id = int(row["chunk_id"])
        topic_id = str(row["topic_id"])
        role = str(row["chunk_role"] or "user").lower()
        text_raw = str(row["text"] or "")
        emb = owner._blob_to_arr(row["chunk_embedding"])
        dense_score = owner._cosine(q_emb, emb)
        dense_ranked.append((chunk_id, dense_score))

        keyword_score = keyword_overlap(q_tokens, owner._tokenize(text_raw))
        topic_prior = float(topic_score_map.get(topic_id, 0.0))
        if max_topic_score > 0.0:
            topic_prior = max(0.0, topic_prior) / max_topic_score
        payload[chunk_id] = {
            "topic_id": topic_id,
            "chunk_id": chunk_id,
            "role": role,
            "text": f"({role}) {text_raw}",
            "session_id": str(row["chunk_session_id"] or ""),
            "session_date": str(row["chunk_session_date"] or ""),
            "has_answer": int(row["chunk_has_answer"] or 0),
            "_keyword_score": keyword_score,
            "_topic_prior": topic_prior,
        }

    dense_ranked.sort(key=lambda x: x[1], reverse=True)
    dense_ranks = {chunk_id: idx + 1 for idx, (chunk_id, _score) in enumerate(dense_ranked)}

    scored: List[Chunk] = []
    for chunk_id, item in payload.items():
        score = 0.0
        dense_rank = dense_ranks.get(chunk_id)
        if dense_rank is not None:
            score += owner.global_chunk_dense_weight / float(owner.global_chunk_rrf_k + dense_rank)
        lexical_rank = lexical_ranks.get(chunk_id)
        if lexical_rank is not None:
            score += owner.global_chunk_lexical_weight / float(
                owner.global_chunk_rrf_k + lexical_rank
            )
        score += owner.global_chunk_keyword_weight * float(item["_keyword_score"])
        score += owner.global_chunk_topic_prior_weight * float(item["_topic_prior"])

        out: Chunk = {
            "topic_id": item["topic_id"],
            "chunk_id": item["chunk_id"],
            "role": item["role"],
            "text": item["text"],
            "session_id": item["session_id"],
            "session_date": item["session_date"],
            "has_answer": item["has_answer"],
            "score": score,
        }
        scored.append(out)

    scored.sort(key=lambda x: float(x["score"]), reverse=True)
    selected = scored[: owner.global_chunk_top_n]
    if not owner.global_chunk_dedup_by_text:
        return selected

    deduped: List[Chunk] = []
    seen_text: set[str] = set()
    for item in selected:
        normalized = re.sub(r"\s+", " ", str(item.get("text", "")).strip().lower())
        if not normalized:
            continue
        if normalized in seen_text:
            continue
        seen_text.add(normalized)
        deduped.append(item)
    return deduped
