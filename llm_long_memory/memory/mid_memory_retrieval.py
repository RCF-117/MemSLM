"""Chunk-first retrieval helpers for MidMemory."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Protocol, Sequence

from llm_long_memory.utils.embedding import embed

Chunk = Dict[str, Any]


class RetrievalOwner(Protocol):
    embedding_dim: int
    global_chunk_top_n: int
    global_chunk_rrf_k: int
    global_chunk_dense_weight: float
    global_chunk_lexical_weight: float
    global_chunk_keyword_weight: float
    global_chunk_prefilter_enabled: bool
    global_chunk_prefilter_top_n: int
    global_chunk_lexical_prefilter_top_n: int
    global_chunk_recent_fallback_n: int
    global_chunk_dedup_by_text: bool
    global_sentence_top_n: int
    global_sentence_rrf_k: int
    global_sentence_dense_weight: float
    global_sentence_lexical_weight: float
    global_sentence_keyword_weight: float
    global_sentence_prefilter_enabled: bool
    global_sentence_prefilter_top_n: int
    global_sentence_lexical_prefilter_top_n: int
    global_sentence_recent_fallback_n: int
    global_sentence_dedup_by_text: bool
    conn: Any

    def _cosine(self, a: Any, b: Any) -> float: ...
    def _tokenize(self, text: str) -> List[str]: ...
    def _blob_to_arr(self, blob: bytes | None): ...
    def _lexical_rank_map_global(self, query: str, top_n: int) -> Dict[int, int]: ...
    def _lexical_rank_map_sentences(self, query: str, top_n: int) -> Dict[int, int]: ...


def keyword_overlap(query_tokens: Sequence[str], target_tokens: Sequence[str]) -> float:
    """Compute keyword overlap ratio between query and target tokens."""
    if not query_tokens:
        return 0.0
    target = {str(x).lower() for x in target_tokens}
    hit = sum(1 for q in query_tokens if q in target)
    return float(hit) / float(len(query_tokens))


def rerank_chunks_global(
    owner: RetrievalOwner,
    query: str,
    top_n_override: int | None = None,
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
            SELECT chunk_id, text, chunk_embedding, chunk_role, chunk_session_id,
                   chunk_session_date, chunk_has_answer, chunk_times, chunk_role_hist,
                   chunk_session_dates, chunk_has_answer_count, chunk_answer_density,
                   chunk_time_sources
            FROM chunks
            WHERE chunk_id IN ({placeholders})
            """,
            tuple(ids),
        ).fetchall()
    else:
        rows = owner.conn.execute(
            """
            SELECT chunk_id, text, chunk_embedding, chunk_role, chunk_session_id,
                   chunk_session_date, chunk_has_answer, chunk_times, chunk_role_hist,
                   chunk_session_dates, chunk_has_answer_count, chunk_answer_density,
                   chunk_time_sources
            FROM chunks
            """
        ).fetchall()
    if not rows:
        return []

    dense_ranked: List[tuple[int, float]] = []
    payload: Dict[int, Dict[str, Any]] = {}

    for row in rows:
        chunk_id = int(row["chunk_id"])
        role = str(row["chunk_role"] or "user").lower()
        text_raw = str(row["text"] or "")
        emb = owner._blob_to_arr(row["chunk_embedding"])
        dense_score = owner._cosine(q_emb, emb)
        dense_ranked.append((chunk_id, dense_score))

        keyword_score = keyword_overlap(q_tokens, owner._tokenize(text_raw))
        try:
            role_hist = json.loads(row["chunk_role_hist"] or "[]")
        except (TypeError, ValueError):
            role_hist = []
        try:
            session_dates = json.loads(row["chunk_session_dates"] or "[]")
        except (TypeError, ValueError):
            session_dates = []
        try:
            time_sources = json.loads(row["chunk_time_sources"] or "[]")
        except (TypeError, ValueError):
            time_sources = []
        payload[chunk_id] = {
            "chunk_id": chunk_id,
            "role": role,
            "text": f"({role}) {text_raw}",
            "session_id": str(row["chunk_session_id"] or ""),
            "session_date": str(row["chunk_session_date"] or ""),
            "has_answer": int(row["chunk_has_answer"] or 0),
            "role_hist": role_hist,
            "session_dates": session_dates,
            "has_answer_count": int(row["chunk_has_answer_count"] or 0),
            "answer_density": float(row["chunk_answer_density"] or 0.0),
            "time_sources": time_sources,
            "_keyword_score": keyword_score,
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

        out: Chunk = {
            "chunk_id": item["chunk_id"],
            "role": item["role"],
            "text": item["text"],
            "session_id": item["session_id"],
            "session_date": item["session_date"],
            "has_answer": item["has_answer"],
            "role_hist": item["role_hist"],
            "session_dates": item["session_dates"],
            "has_answer_count": item["has_answer_count"],
            "answer_density": item["answer_density"],
            "time_sources": item["time_sources"],
            "score": score,
        }
        scored.append(out)

    scored.sort(key=lambda x: float(x["score"]), reverse=True)
    if top_n_override is not None and int(top_n_override) > 0:
        effective_top_n = int(top_n_override)
    else:
        effective_top_n = int(owner.global_chunk_top_n)
    selected = scored[: max(1, effective_top_n)]
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


def rerank_sentences_global(
    owner: RetrievalOwner,
    query: str,
    top_n_override: int | None = None,
) -> List[Chunk]:
    """Secondary retrieval path: sentence-level rerank for fine-grained evidence."""
    q = query.strip()
    if not q:
        return []
    q_emb = embed(q, owner.embedding_dim)
    q_tokens = owner._tokenize(q)
    lexical_ranks = owner._lexical_rank_map_sentences(
        query=q,
        top_n=owner.global_sentence_lexical_prefilter_top_n,
    )

    candidate_ids: set[int] = set()
    if owner.global_sentence_prefilter_enabled:
        candidate_ids.update(int(x) for x in lexical_ranks.keys())
        if owner.global_sentence_recent_fallback_n > 0:
            recent_rows = owner.conn.execute(
                """
                SELECT sentence_id
                FROM sentences
                ORDER BY sentence_id DESC
                LIMIT ?
                """,
                (int(owner.global_sentence_recent_fallback_n),),
            ).fetchall()
            candidate_ids.update(int(row["sentence_id"]) for row in recent_rows)

    if owner.global_sentence_prefilter_enabled and candidate_ids:
        ids = sorted(candidate_ids)[: owner.global_sentence_prefilter_top_n]
        placeholders = ",".join(["?"] * len(ids))
        rows = owner.conn.execute(
            f"""
            SELECT sentence_id, chunk_id, text, sentence_embedding, sentence_role,
                   sentence_session_id, sentence_session_date, source_part_index
            FROM sentences
            WHERE sentence_id IN ({placeholders})
            """,
            tuple(ids),
        ).fetchall()
    else:
        rows = owner.conn.execute(
            """
            SELECT sentence_id, chunk_id, text, sentence_embedding, sentence_role,
                   sentence_session_id, sentence_session_date, source_part_index
            FROM sentences
            """
        ).fetchall()
    if not rows:
        return []

    dense_ranked: List[tuple[int, float]] = []
    payload: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        sentence_id = int(row["sentence_id"])
        text_raw = str(row["text"] or "")
        role = str(row["sentence_role"] or "user").lower()
        emb = owner._blob_to_arr(row["sentence_embedding"])
        dense_score = owner._cosine(q_emb, emb)
        dense_ranked.append((sentence_id, dense_score))
        keyword_score = keyword_overlap(q_tokens, owner._tokenize(text_raw))
        payload[sentence_id] = {
            "sentence_id": sentence_id,
            "parent_chunk_id": int(row["chunk_id"] or 0),
            "role": role,
            "text": f"({role}) {text_raw}",
            "session_id": str(row["sentence_session_id"] or ""),
            "session_date": str(row["sentence_session_date"] or ""),
            "source_part_index": int(row["source_part_index"] or 0),
            "_keyword_score": keyword_score,
        }

    dense_ranked.sort(key=lambda x: x[1], reverse=True)
    dense_ranks = {sid: idx + 1 for idx, (sid, _score) in enumerate(dense_ranked)}

    scored: List[Chunk] = []
    for sentence_id, item in payload.items():
        score = 0.0
        dense_rank = dense_ranks.get(sentence_id)
        if dense_rank is not None:
            score += owner.global_sentence_dense_weight / float(
                owner.global_sentence_rrf_k + dense_rank
            )
        lexical_rank = lexical_ranks.get(sentence_id)
        if lexical_rank is not None:
            score += owner.global_sentence_lexical_weight / float(
                owner.global_sentence_rrf_k + lexical_rank
            )
        score += owner.global_sentence_keyword_weight * float(item["_keyword_score"])
        out: Chunk = {
            "unit_type": "sentence",
            "sentence_id": sentence_id,
            "chunk_id": None,
            "parent_chunk_id": item["parent_chunk_id"],
            "role": item["role"],
            "text": item["text"],
            "session_id": item["session_id"],
            "session_date": item["session_date"],
            "source_part_index": item["source_part_index"],
            "score": score,
        }
        scored.append(out)

    scored.sort(key=lambda x: float(x["score"]), reverse=True)
    if top_n_override is not None and int(top_n_override) > 0:
        effective_top_n = int(top_n_override)
    else:
        effective_top_n = int(owner.global_sentence_top_n)
    selected = scored[: max(1, effective_top_n)]
    if not owner.global_sentence_dedup_by_text:
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
