"""Summary and keyword update helpers for MidMemory."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, List, Protocol

from llm_long_memory.utils.embedding import embed
from llm_long_memory.utils.logger import logger


class SummaryOwner(Protocol):
    summary_enabled: bool
    summary_min_chunk_count: int
    summary_update_every: int
    summary_on_new_topic: bool
    summary_cooldown_steps: int
    current_step: int
    embedding_dim: int
    summary_model: str
    llm_host: str
    temperature: float
    request_timeout_sec: int
    embedding_min_length: int
    max_chunk_size: int
    conn: Any
    _opener: Any

    def _arr_to_blob(self, arr: Any) -> bytes: ...
    def _normalize_token(self, token: str) -> str: ...


def maybe_update_summary_keywords(owner: SummaryOwner, topic_id: str, created_new_topic: bool) -> None:
    """Update topic summary/keywords at configured cadence."""
    if not owner.summary_enabled:
        return
    row = owner.conn.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM chunks WHERE topic_id = ?) AS cnt,
          (SELECT COALESCE(last_summary_step, 0) FROM topics WHERE topic_id = ?) AS last_summary_step
        """,
        (topic_id, topic_id),
    ).fetchone()
    chunk_count = int(row["cnt"]) if row else 0
    last_summary_step = int(row["last_summary_step"]) if row else 0

    if chunk_count < owner.summary_min_chunk_count:
        return

    periodic_trigger = chunk_count % owner.summary_update_every == 0
    new_topic_trigger = created_new_topic and owner.summary_on_new_topic
    cooldown_ok = (owner.current_step - last_summary_step) >= owner.summary_cooldown_steps
    if not ((periodic_trigger or new_topic_trigger) and cooldown_ok):
        return

    chunk_rows = owner.conn.execute(
        "SELECT text FROM chunks WHERE topic_id = ? ORDER BY chunk_id ASC",
        (topic_id,),
    ).fetchall()
    raw_text = "\n".join(str(r["text"]) for r in chunk_rows)
    summary = summarize_with_model(owner, raw_text)
    keywords = extract_keywords(owner, summary)
    owner.conn.execute(
        """
        UPDATE topics
        SET summary = ?, summary_embedding = ?, keywords = ?, last_summary_step = ?
        WHERE topic_id = ?
        """,
        (
            summary,
            owner._arr_to_blob(embed(summary, owner.embedding_dim)),
            json.dumps(keywords),
            owner.current_step,
            topic_id,
        ),
    )


def summarize_with_model(owner: SummaryOwner, raw_text: str) -> str:
    """Generate extractive summary with local summary model, fallback to truncation."""
    prompt = (
        "You are an extractive summarizer.\n"
        "Return only content from source text. No hallucination.\n"
        f"Source:\n{raw_text}"
    )
    payload = {
        "model": owner.summary_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": owner.temperature},
    }
    req = urllib.request.Request(
        url=f"{owner.llm_host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with owner._opener.open(req, timeout=owner.request_timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        out = str(data.get("response", "")).strip()
        if out:
            return extractive_clip(out, raw_text)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        logger.warn(f"MidMemory._summarize_with_model failed: {exc}")

    fallback_len = max(owner.embedding_min_length, owner.max_chunk_size * owner.embedding_min_length)
    return raw_text[:fallback_len].strip()


def extractive_clip(summary: str, source: str) -> str:
    """Constrain summary to extractive content when possible."""
    if summary in source:
        return summary
    lines = [x.strip() for x in source.split("\n") if x.strip()]
    selected = [line for line in lines if line in summary or summary in line]
    return "\n".join(selected) if selected else summary


def extract_keywords(owner: SummaryOwner, text: str) -> List[str]:
    """Extract normalized keyword list from summary text."""
    out: List[str] = []
    for tok in text.lower().split():
        norm = owner._normalize_token(tok)
        if norm and norm not in out:
            out.append(norm)
    return out
