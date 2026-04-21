"""Evaluation runner for dataset-based memory-RAG benchmarks."""

from __future__ import annotations

from datetime import datetime
import time
import uuid
from typing import Any, Dict, List

from llm_long_memory.evaluation.dataset_loader import iter_history_messages, load_stream
from llm_long_memory.evaluation.metrics_runtime import (
    compute_answer_span_hit,
    compute_support_sentence_hit,
    eval_group_key,
    evaluate_match,
    update_group_stats,
)
from llm_long_memory.evaluation.eval_reporting import EvalCounters, finalize_eval_run
from llm_long_memory.memory.long_memory_store import LongMemoryStore
from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.utils.helpers import resolve_project_path
from llm_long_memory.utils.logger import logger


def _create_graph_accumulator_store(config: Dict[str, Any]) -> LongMemoryStore:
    long_cfg = dict(config["memory"]["long_memory"])
    graph_root = resolve_project_path("data/processed/thesis_graph_runs")
    graph_root.mkdir(parents=True, exist_ok=True)
    db_path = graph_root / "thesis_graph_runs.db"
    return LongMemoryStore(
        database_file=str(db_path),
        sqlite_busy_timeout_ms=int(long_cfg.get("sqlite_busy_timeout_ms", 5000)),
        sqlite_journal_mode=str(long_cfg.get("sqlite_journal_mode", "WAL")),
        sqlite_synchronous=str(long_cfg.get("sqlite_synchronous", "NORMAL")),
        embedding_dim=int(config["embedding"]["dim"]),
    )


def run_eval(
    manager: MemoryManager,
    dataset_path: str,
    config: Dict[str, Any],
    *,
    resume_run_id: str | None = None,
) -> str:
    """Run eval mode: ingest history, then ask question and compare with answer."""
    dataset_cfg = config["dataset"]
    eval_cfg = config["evaluation"]
    stream_mode = bool(dataset_cfg["stream_mode"])
    max_instances = int(dataset_cfg["eval_max_instances"])
    preview_chars = int(dataset_cfg["eval_print_prediction_chars"])
    isolated = bool(eval_cfg["isolated_per_instance"])
    group_by_type = bool(eval_cfg["group_by_question_type"])
    use_short_context_in_eval = bool(eval_cfg["use_short_context_in_eval"])
    compute_evidence_recall = bool(eval_cfg.get("compute_evidence_recall", False))
    compute_answer_span_hit_enabled = bool(eval_cfg["compute_answer_span_hit"])
    compute_support_sentence_hit_enabled = bool(eval_cfg["compute_support_sentence_hit"])
    answer_style = str(eval_cfg.get("eval_answer_style", "")).strip()
    oracle_temporal_cfg = dict(eval_cfg.get("oracle_temporal", {}))
    oracle_temporal_disable = bool(oracle_temporal_cfg.get("disable_temporal_weight", False))
    oracle_dataset_keyword = str(oracle_temporal_cfg.get("dataset_keyword", "")).strip().lower()
    offline_graph_build_enabled = bool(eval_cfg.get("offline_graph_build_enabled", False))
    graph_metric_enabled = bool(
        offline_graph_build_enabled
        or bool(getattr(manager, "evidence_graph_enabled", False))
        or bool(getattr(manager, "long_memory_enabled", False))
    )

    prev_temporal_disabled = bool(getattr(manager.mid_memory, "temporal_weight_disabled", False))
    should_disable_temporal = (
        oracle_temporal_disable
        and bool(oracle_dataset_keyword)
        and (oracle_dataset_keyword in str(dataset_path).lower())
    )
    if should_disable_temporal:
        manager.mid_memory.set_temporal_weight_disabled(True)

    logger.info(
        "Eval mode started: "
        f"path={dataset_path}, stream_mode={stream_mode}, max_instances={max_instances}, isolated={isolated}"
    )
    run_id = str(resume_run_id or "").strip() or (
        datetime.now().strftime("run_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    )
    store = manager.mid_memory.eval_store
    if not store.run_exists(run_id):
        store.log_eval_run_start(run_id, dataset_path, isolated, commit=True)
        logger.info(f"Eval run id: {run_id} (new)")
    else:
        logger.info(f"Eval run id: {run_id} (resuming existing run)")
    processed_question_ids = store.get_existing_question_ids(run_id)
    remaining_budget = None
    if max_instances > 0:
        remaining_budget = max(0, int(max_instances) - len(processed_question_ids))
        logger.info(
            "Eval resume budget: "
            f"target_total={max_instances}, already_done={len(processed_question_ids)}, "
            f"remaining={remaining_budget}"
        )
    if processed_question_ids:
        logger.info(
            f"Eval resume state: already_processed={len(processed_question_ids)} question_ids."
        )

    instances = load_stream(dataset_path)
    seen = 0
    processed_now = 0
    counters = EvalCounters()
    grouped: Dict[str, Dict[str, int]] = {}
    graph_accumulator_store: LongMemoryStore | None = None
    if offline_graph_build_enabled and bool(getattr(manager, "long_memory_enabled", False)):
        graph_accumulator_store = _create_graph_accumulator_store(config)
        if not store.run_exists(run_id):
            graph_accumulator_store.clear_all()
            graph_accumulator_store.commit()

    eval_error: Exception | None = None
    try:
        for instance in instances:
            if remaining_budget is not None and processed_now >= remaining_budget:
                break

            seen += 1
            qid = str(instance.get("question_id", ""))
            qtype = str(instance.get("question_type", ""))
            question = str(instance.get("question", "")).strip()
            expected = str(instance.get("answer", "")).strip()

            if qid and qid in processed_question_ids:
                logger.info(f"Eval instance {seen}: question_id={qid} already processed, skipped.")
                continue

            logger.info(f"Eval instance {seen}: question_id={qid}, question_type={qtype}")
            if isolated:
                manager.reset_for_new_instance()

            for message in iter_history_messages(instance):
                manager.ingest_message(message)
            manager.finalize_ingest()

            if not question:
                logger.warn(f"Eval instance {seen}: empty question, skipped.")
                continue

            # Prevent evaluation leakage from short-memory recency unless explicitly enabled.
            if not use_short_context_in_eval:
                manager.archive_short_to_mid(clear_short=True)

            evidence_hit = None
            evidence_recall = None
            answer_span_hit = None
            support_sentence_hit = None
            graph_answer_span_hit = None
            graph_support_sentence_hit = None
            retrieved_session_ids: List[str] = []
            precomputed_context = None
            graph_context_chunks: List[Dict[str, str]] = []
            if (
                compute_evidence_recall
                or compute_answer_span_hit_enabled
                or compute_support_sentence_hit_enabled
                or offline_graph_build_enabled
                or graph_metric_enabled
            ):
                precomputed_context = manager.retrieve_context(question)
                _ctx, _topics, chunks = precomputed_context
                if offline_graph_build_enabled:
                    built = manager.offline_build_long_graph_from_chunks(
                        chunks,
                        query=question,
                    )
                    logger.info(
                        "Eval offline graph build: "
                            f"question_id={qid}, accepted_events={built}."
                        )
                    if graph_accumulator_store is not None and hasattr(
                        manager.long_memory, "export_snapshot_to_store"
                    ):
                        manager.long_memory.export_snapshot_to_store(graph_accumulator_store)
                if bool(getattr(manager, "evidence_graph_enabled", False)):
                    graph_bundle = manager.build_evidence_graph_bundle(
                        question,
                        precomputed_context=precomputed_context,
                    )
                    graph_context_chunks = manager.chat_runtime.graph_bundle_to_chunks(graph_bundle)
                elif graph_metric_enabled and hasattr(manager.long_memory, "build_context_snippets"):
                    graph_snippets = manager.long_memory.build_context_snippets(question)
                    graph_context_chunks = [{"text": snippet} for snippet in graph_snippets if str(snippet).strip()]
                if compute_evidence_recall:
                    retrieved_session_ids = sorted(
                        {
                            str(c.get("session_id") or "").strip()
                            for c in chunks
                            if str(c.get("session_id") or "").strip()
                        }
                    )
                    evidence_sessions = {
                        str(x).strip()
                        for x in list(instance.get("answer_session_ids", []))
                        if str(x).strip()
                    }
                    hit_session = (
                        bool(evidence_sessions.intersection(retrieved_session_ids))
                        if evidence_sessions
                        else False
                    )
                    hit_turn = any(int(c.get("has_answer") or 0) == 1 for c in chunks)
                    evidence_hit = bool(hit_session or hit_turn)
                    evidence_recall = (
                        float(len(evidence_sessions.intersection(retrieved_session_ids)))
                        / float(len(evidence_sessions))
                        if evidence_sessions
                        else (1.0 if hit_turn else 0.0)
                    )
                    counters.retrieval_evidence_hits += int(bool(evidence_hit))
                if graph_metric_enabled:
                    graph_answer_span_hit = compute_answer_span_hit(expected, graph_context_chunks, eval_cfg)
                    counters.graph_span_hits += int(bool(graph_answer_span_hit))
                if graph_metric_enabled:
                    graph_support_sentence_hit = compute_support_sentence_hit(expected, graph_context_chunks, eval_cfg)
                    counters.graph_support_hits += int(bool(graph_support_sentence_hit))
                if graph_context_chunks:
                    counters.graph_retrieval_total += 1

            processed_now += 1
            counters.total += 1
            eval_question = f"{answer_style}\n{question}" if answer_style else question
            started = time.perf_counter()
            prediction = manager.chat(
                eval_question,
                retrieval_query=question,
                precomputed_context=precomputed_context,
            )
            latency_sec = time.perf_counter() - started
            prompt_chunks = manager.get_last_prompt_eval_chunks()
            if compute_answer_span_hit_enabled:
                answer_span_hit = compute_answer_span_hit(expected, prompt_chunks, eval_cfg)
                counters.retrieval_span_hits += int(bool(answer_span_hit))
            if compute_support_sentence_hit_enabled:
                support_sentence_hit = compute_support_sentence_hit(expected, prompt_chunks, eval_cfg)
                counters.retrieval_support_hits += int(bool(support_sentence_hit))
            if compute_answer_span_hit_enabled or compute_support_sentence_hit_enabled:
                counters.retrieval_total += 1
            match_result = evaluate_match(prediction, expected, eval_cfg)
            is_match = bool(match_result["is_match"])
            if is_match:
                counters.matched += 1
            if group_by_type:
                update_group_stats(grouped, eval_group_key(qid, qtype, eval_cfg), is_match)
            if bool(getattr(manager, "long_memory_enabled", False)):
                stats = manager.long_memory.debug_stats()
                cur_total = int(stats.get("ingest_event_total", 0))
                cur_accepted = int(stats.get("ingest_event_accepted", 0))
                prev_total = int(counters.ingest_prev_total)
                prev_accepted = int(counters.ingest_prev_accepted)
                if cur_total < prev_total:
                    delta_total = cur_total
                else:
                    delta_total = cur_total - prev_total
                if cur_accepted < prev_accepted:
                    delta_accepted = cur_accepted
                else:
                    delta_accepted = cur_accepted - prev_accepted
                counters.ingest_event_total += max(0, int(delta_total))
                counters.ingest_event_accepted += max(0, int(delta_accepted))
                counters.ingest_prev_total = cur_total
                counters.ingest_prev_accepted = cur_accepted
            manager.mid_memory.eval_store.log_eval_result(
                run_id=run_id,
                question_id=qid,
                question_type=qtype,
                question=question,
                expected_answer=expected,
                prediction=prediction,
                is_match=is_match,
                evidence_hit=evidence_hit,
                evidence_recall=evidence_recall,
                answer_span_hit=answer_span_hit,
                support_sentence_hit=support_sentence_hit,
                graph_answer_span_hit=graph_answer_span_hit,
                graph_support_sentence_hit=graph_support_sentence_hit,
                latency_sec=latency_sec,
                retrieved_session_ids=retrieved_session_ids,
                commit=True,
            )

            preview = prediction[:preview_chars].replace("\n", " ")
            print(
                f"[Eval {counters.total}] question_id={qid} | type={qtype} | match={is_match}\n"
                f"Score: em={match_result['em']:.2f}, f1={match_result['f1']:.2f}, "
                f"substring={match_result['substring']:.0f}, numeric={match_result['numeric']:.0f}\n"
                f"Latency: {latency_sec:.3f}s\n"
                "Retrieval Quality: "
                f"support_sentence_hit={support_sentence_hit}, answer_span_hit={answer_span_hit}\n"
                "Graph Quality: "
                f"support_sentence_hit={graph_support_sentence_hit}, answer_span_hit={graph_answer_span_hit}\n"
                "Coverage (aux): "
                f"evidence_hit={evidence_hit}, recall={evidence_recall}\n"
                f"Q: {question}\n"
                f"Gold: {expected}\n"
                f"MatchedRef: {match_result['best_reference']}\n"
                f"Pred: {preview}\n"
            )
    except (RuntimeError, ValueError, TypeError, OSError, KeyboardInterrupt) as exc:
        eval_error = exc
        logger.error(f"Eval aborted by exception: {exc}")
    finally:
        if should_disable_temporal:
            manager.mid_memory.set_temporal_weight_disabled(prev_temporal_disabled)
        if graph_accumulator_store is not None:
            graph_accumulator_store.commit()
            graph_accumulator_store.close()
        finalize_eval_run(
            manager=manager,
            run_id=run_id,
            grouped=grouped,
            group_by_type=group_by_type,
            counters=counters,
        )

    if eval_error is not None:
        raise eval_error
    return run_id
