"""Standalone thesis baseline runners for direct model-only and naive RAG evals."""

from __future__ import annotations

import re
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from llm_long_memory.evaluation.dataset_loader import load_stream
from llm_long_memory.evaluation.eval_reporting import EvalCounters, finalize_eval_run
from llm_long_memory.evaluation.eval_store import EvalStore
from llm_long_memory.evaluation.metrics_runtime import (
    compute_answer_span_hit,
    compute_support_sentence_hit,
    eval_group_key,
    evaluate_match,
    update_group_stats,
)
from llm_long_memory.llm.ollama_client import LLM
from llm_long_memory.utils.helpers import resolve_project_path, sanitize_filename_part
from llm_long_memory.utils.logger import logger


EvalPromptResult = Tuple[str, List[Dict[str, object]], List[str]]
PromptBuilder = Callable[[Dict[str, Any], Dict[str, Any]], EvalPromptResult]


class _NoOpLongMemory:
    """Minimal long-memory stub so finalize_eval_run can reuse existing code."""

    def debug_stats(self) -> Dict[str, int]:
        return {
            "nodes": 0,
            "edges": 0,
            "events": 0,
            "details": 0,
            "active_events": 0,
            "superseded_events": 0,
            "queued_updates": 0,
            "applied_updates": 0,
            "ingest_event_total": 0,
            "ingest_event_accepted": 0,
            "ingest_event_rejected": 0,
            "extractor_calls": 0,
            "extractor_success": 0,
            "extractor_failures": 0,
            "extractor_seen_messages": 0,
            "candidate_events": 0,
            "reject_reason_low_confidence": 0,
            "reject_reason_few_keywords": 0,
            "reject_reason_short_object": 0,
            "reject_reason_few_entities": 0,
            "reject_reason_short_sentence": 0,
            "reject_reason_long_sentence": 0,
            "reject_reason_missing_time_or_location": 0,
            "reject_reason_rejected_phrase": 0,
            "reject_reason_generic_subject_action": 0,
            "reject_reason_generic_action_disabled": 0,
            "reject_reason_empty_key_component": 0,
        }


def _parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _default_run_id(mode_name: str) -> str:
    tag = sanitize_filename_part(mode_name).replace(".", "_")
    return f"run_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _turn_text(turn: Dict[str, Any]) -> str:
    role = str(turn.get("role", "user")).strip().lower() or "user"
    content = str(turn.get("content", "")).strip()
    if role == "assistant":
        prefix = "Assistant"
    elif role == "system":
        prefix = "System"
    else:
        prefix = "User"
    return f"{prefix}: {content}"


def build_session_passages(instance: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn one eval instance into passage-like session blocks."""
    sessions = instance.get("haystack_sessions", [])
    session_ids = list(instance.get("haystack_session_ids", []))
    session_dates = list(instance.get("haystack_dates", []))
    passages: List[Dict[str, Any]] = []
    if not isinstance(sessions, list):
        return passages
    for index, session in enumerate(sessions):
        if not isinstance(session, list):
            continue
        lines: List[str] = []
        for turn in session:
            if not isinstance(turn, dict):
                continue
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            lines.append(_turn_text(turn))
        if not lines:
            continue
        sid = str(session_ids[index]) if index < len(session_ids) else f"session_{index + 1}"
        sdate = str(session_dates[index]) if index < len(session_dates) else ""
        header = f"[Session {index + 1} | {sid}"
        if sdate:
            header += f" | {sdate}"
        header += "]"
        passages.append(
            {
                "session_id": sid,
                "session_date": sdate,
                "text": header + "\n" + "\n".join(lines),
            }
        )
    return passages


def _tokenize(text: str) -> List[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "to",
        "of",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "my",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "i",
        "me",
        "you",
        "it",
        "that",
        "this",
        "do",
        "does",
        "did",
    }
    return [tok for tok in re.findall(r"[a-z0-9]+", str(text).lower()) if tok and tok not in stopwords]


def _score_passage(query_tokens: Sequence[str], passage_text: str) -> float:
    passage_tokens = _tokenize(passage_text)
    if not query_tokens or not passage_tokens:
        return 0.0
    query_set = set(query_tokens)
    passage_set = set(passage_tokens)
    overlap = len(query_set.intersection(passage_set))
    if overlap <= 0:
        return 0.0
    coverage = float(overlap) / float(max(1, len(query_set)))
    length_penalty = max(1.0, float(len(passage_tokens)) / 80.0)
    return ((coverage * 2.0) + float(overlap) * 0.25) / length_penalty


def build_model_only_prompt(instance: Dict[str, Any]) -> EvalPromptResult:
    """Build a bare-model prompt that directly includes the whole conversation."""
    question = str(instance.get("question", "")).strip()
    passages = build_session_passages(instance)
    history_text = "\n\n".join(str(p["text"]) for p in passages)
    prompt = (
        "You are answering a memory benchmark question.\n"
        "Use only the conversation history below.\n"
        "Return the final answer only.\n"
        "If the answer cannot be found, say 'Not found in conversation.'\n\n"
        "[Conversation History]\n"
        f"{history_text or 'None'}\n\n"
        "[Question]\n"
        f"{question}"
    )
    prompt_chunks = [{"section": "conversation_history", "text": text} for text in (p["text"] for p in passages)]
    retrieved_ids = [str(p.get("session_id", "")).strip() for p in passages if str(p.get("session_id", "")).strip()]
    return prompt, prompt_chunks, retrieved_ids


def build_naive_rag_prompt(instance: Dict[str, Any], *, top_k: int) -> EvalPromptResult:
    """Build a classic retrieve-then-generate prompt over session passages."""
    question = str(instance.get("question", "")).strip()
    passages = build_session_passages(instance)
    query_tokens = _tokenize(question)
    ranked = [
        {**passage, "score": _score_passage(query_tokens, str(passage.get("text", "")))}
        for passage in passages
    ]
    ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    selected = ranked[: max(1, int(top_k))]
    context_text = "\n\n".join(
        f"[Chunk {index} | {chunk.get('session_id', '')}]\n{chunk.get('text', '')}"
        for index, chunk in enumerate(selected, start=1)
        if str(chunk.get("text", "")).strip()
    )
    prompt = (
        "You are answering a memory benchmark question.\n"
        "Use only the retrieved context below.\n"
        "Return the final answer only.\n"
        "If the answer cannot be found, say 'Not found in retrieved context.'\n\n"
        "[Retrieved Context]\n"
        f"{context_text or 'None'}\n\n"
        "[Question]\n"
        f"{question}"
    )
    prompt_chunks = [{"section": "retrieved_context", "text": str(chunk.get("text", "")).strip()} for chunk in selected if str(chunk.get("text", "")).strip()]
    retrieved_ids = [str(chunk.get("session_id", "")).strip() for chunk in selected if str(chunk.get("session_id", "")).strip()]
    return prompt, prompt_chunks, retrieved_ids


def _make_store_proxy(store: EvalStore) -> SimpleNamespace:
    return SimpleNamespace(
        mid_memory=SimpleNamespace(eval_store=store, commit=store.conn.commit),
        long_memory=_NoOpLongMemory(),
    )


def run_direct_mode_eval(
    *,
    mode_name: str,
    config: Dict[str, Any],
    dataset_path: str,
    dataset_name: str,
    model_name: str,
    prompt_builder: PromptBuilder,
    resume_run_id: str | None = None,
) -> str:
    """Run a standalone mode and persist results to the shared eval DB."""
    dataset_cfg = config["dataset"]
    eval_cfg = config["evaluation"]
    llm_cfg = config["llm"]
    max_instances = int(dataset_cfg["eval_max_instances"])
    preview_chars = int(dataset_cfg["eval_print_prediction_chars"])
    group_by_type = bool(eval_cfg["group_by_question_type"])
    compute_answer_span_hit_enabled = bool(eval_cfg["compute_answer_span_hit"])
    compute_support_sentence_hit_enabled = bool(eval_cfg["compute_support_sentence_hit"])

    db_path = resolve_project_path(str(eval_cfg["database_file"]))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    store = EvalStore(conn=conn, eval_cfg=dict(eval_cfg))
    store.create_tables()
    store.ensure_schema_compat()

    selected_model = model_name.strip() or str(llm_cfg["default_model"])
    llm = LLM(model_name=selected_model, host=str(llm_cfg["host"]))

    run_id = str(resume_run_id or "").strip() or _default_run_id(mode_name)
    if not store.run_exists(run_id):
        store.log_eval_run_start(run_id, dataset_path, isolated=True, commit=True)
        logger.info(f"[{mode_name}] eval run id: {run_id} (new)")
    else:
        logger.info(f"[{mode_name}] eval run id: {run_id} (resuming existing run)")

    processed_question_ids = store.get_existing_question_ids(run_id)
    if max_instances > 0:
        remaining_budget = max(0, int(max_instances) - len(processed_question_ids))
    else:
        remaining_budget = None

    counters = EvalCounters()
    grouped: Dict[str, Dict[str, int]] = {}
    seen = 0
    processed_now = 0
    eval_error: Exception | None = None

    try:
        for instance in load_stream(dataset_path):
            if remaining_budget is not None and processed_now >= remaining_budget:
                break
            seen += 1
            qid = str(instance.get("question_id", "")).strip()
            qtype = str(instance.get("question_type", "")).strip()
            question = str(instance.get("question", "")).strip()
            expected = str(instance.get("answer", "")).strip()

            if qid and qid in processed_question_ids:
                logger.info(f"[{mode_name}] instance {seen}: question_id={qid} already processed, skipped.")
                continue
            if not question:
                logger.warn(f"[{mode_name}] instance {seen}: empty question, skipped.")
                continue

            prompt_text, prompt_chunks, retrieved_session_ids = prompt_builder(instance, config)
            logger.info(
                f"[{mode_name}] instance {seen}: question_id={qid}, question_type={qtype}, prompt_chars={len(prompt_text)}"
            )
            started = time.perf_counter()
            prediction = llm.chat(prompt_text)
            latency_sec = time.perf_counter() - started
            preview = prediction[:preview_chars].replace("\n", " ")

            answer_span_hit = (
                compute_answer_span_hit(expected, prompt_chunks, eval_cfg)
                if compute_answer_span_hit_enabled
                else None
            )
            support_sentence_hit = (
                compute_support_sentence_hit(expected, prompt_chunks, eval_cfg)
                if compute_support_sentence_hit_enabled
                else None
            )
            if compute_answer_span_hit_enabled or compute_support_sentence_hit_enabled:
                counters.retrieval_total += 1
            if answer_span_hit:
                counters.retrieval_span_hits += 1
            if support_sentence_hit:
                counters.retrieval_support_hits += 1

            match_result = evaluate_match(prediction, expected, eval_cfg)
            is_match = bool(match_result["is_match"])
            if is_match:
                counters.matched += 1
            counters.total += 1
            if group_by_type:
                update_group_stats(grouped, eval_group_key(qid, qtype, eval_cfg), is_match)

            store.log_eval_result(
                run_id=run_id,
                question_id=qid,
                question_type=qtype,
                question=question,
                expected_answer=expected,
                prediction=prediction,
                is_match=is_match,
                evidence_hit=None,
                evidence_recall=None,
                answer_span_hit=answer_span_hit,
                support_sentence_hit=support_sentence_hit,
                graph_answer_span_hit=None,
                graph_support_sentence_hit=None,
                latency_sec=latency_sec,
                retrieved_session_ids=retrieved_session_ids,
                commit=True,
            )

            processed_now += 1
            print(
                f"[{mode_name} {counters.total}] question_id={qid} | type={qtype} | match={is_match}\n"
                f"Score: em={match_result['em']:.2f}, f1={match_result['f1']:.2f}, "
                f"substring={match_result['substring']:.0f}, numeric={match_result['numeric']:.0f}\n"
                f"Latency: {latency_sec:.3f}s\n"
                f"Prompt chunks: {len(prompt_chunks)}\n"
                f"Q: {question}\n"
                f"Gold: {expected}\n"
                f"MatchedRef: {match_result['best_reference']}\n"
                f"Pred: {preview}\n"
            )
    except (RuntimeError, ValueError, TypeError, OSError, KeyboardInterrupt) as exc:
        eval_error = exc
        logger.error(f"[{mode_name}] eval aborted by exception: {exc}")
    finally:
        proxy = _make_store_proxy(store)
        finalize_eval_run(
            manager=proxy,
            run_id=run_id,
            grouped=grouped,
            group_by_type=group_by_type,
            counters=counters,
        )
        store.log_thesis_mode_run(
            dataset_name=dataset_name,
            mode=mode_name,
            run_id=run_id,
            model_name=selected_model,
            judge_model="",
            commit=True,
        )
        conn.close()

    if eval_error is not None:
        raise eval_error
    return run_id
