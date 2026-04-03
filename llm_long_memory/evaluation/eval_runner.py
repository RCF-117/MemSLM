"""Evaluation runner for dataset-based memory-RAG benchmarks."""

from __future__ import annotations

from datetime import datetime
import uuid
from typing import Any, Dict, List

from llm_long_memory.evaluation.longmemeval_loader import iter_history_messages, load_stream
from llm_long_memory.evaluation.metrics_runtime import (
    compute_answer_span_hit,
    compute_support_sentence_hit,
    eval_group_key,
    evaluate_match,
    update_group_stats,
)
from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.utils.logger import logger


def run_eval(manager: MemoryManager, dataset_path: str, config: Dict[str, Any]) -> None:
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
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    manager.mid_memory.log_eval_run_start(run_id, dataset_path, isolated, commit=False)
    logger.info(f"Eval run id: {run_id}")

    instances = load_stream(dataset_path)
    seen = 0
    total = 0
    matched = 0
    grouped: Dict[str, Dict[str, int]] = {}
    retrieval_total = 0
    retrieval_span_hits = 0
    retrieval_support_hits = 0
    retrieval_evidence_hits = 0
    graph_retrieval_total = 0
    graph_retrieval_hits = 0

    eval_error: Exception | None = None
    try:
        for instance in instances:
            if max_instances > 0 and seen >= max_instances:
                break

            seen += 1
            qid = str(instance.get("question_id", ""))
            qtype = str(instance.get("question_type", ""))
            question = str(instance.get("question", "")).strip()
            expected = str(instance.get("answer", "")).strip()

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
            graph_retrieval_hit = None
            retrieved_session_ids: List[str] = []
            precomputed_context = None
            if (
                compute_evidence_recall
                or compute_answer_span_hit_enabled
                or compute_support_sentence_hit_enabled
            ):
                precomputed_context = manager.retrieve_context(question)
                _ctx, _topics, chunks = precomputed_context
                graph_nodes = manager.long_memory.query(question)
                graph_retrieval_total += 1
                graph_chunks = [{"text": str(item.get("text", ""))} for item in graph_nodes]
                graph_retrieval_hit = bool(
                    compute_answer_span_hit(expected, graph_chunks, eval_cfg)
                    or compute_support_sentence_hit(expected, graph_chunks, eval_cfg)
                )
                graph_retrieval_hits += int(bool(graph_retrieval_hit))
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
                    retrieval_evidence_hits += int(bool(evidence_hit))
                if compute_answer_span_hit_enabled:
                    answer_span_hit = compute_answer_span_hit(expected, chunks, eval_cfg)
                    retrieval_span_hits += int(bool(answer_span_hit))
                if compute_support_sentence_hit_enabled:
                    support_sentence_hit = compute_support_sentence_hit(expected, chunks, eval_cfg)
                    retrieval_support_hits += int(bool(support_sentence_hit))
                retrieval_total += 1

            total += 1
            eval_question = f"{answer_style}\n{question}" if answer_style else question
            prediction = manager.chat(
                eval_question,
                retrieval_query=question,
                precomputed_context=precomputed_context,
            )
            match_result = evaluate_match(prediction, expected, eval_cfg)
            is_match = bool(match_result["is_match"])
            if is_match:
                matched += 1
            if group_by_type:
                update_group_stats(grouped, eval_group_key(qid, qtype, eval_cfg), is_match)
            manager.mid_memory.log_eval_result(
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
                graph_retrieval_hit=graph_retrieval_hit,
                retrieved_session_ids=retrieved_session_ids,
                commit=False,
            )

            preview = prediction[:preview_chars].replace("\n", " ")
            print(
                f"[Eval {total}] question_id={qid} | type={qtype} | match={is_match}\n"
                f"Score: em={match_result['em']:.2f}, f1={match_result['f1']:.2f}, "
                f"substring={match_result['substring']:.0f}, numeric={match_result['numeric']:.0f}\n"
                "Retrieval Quality: "
                f"support_sentence_hit={support_sentence_hit}, answer_span_hit={answer_span_hit}\n"
                "Coverage (aux): "
                f"evidence_hit={evidence_hit}, recall={evidence_recall}\n"
                f"Q: {question}\n"
                f"Gold: {expected}\n"
                f"MatchedRef: {match_result['best_reference']}\n"
                f"Pred: {preview}\n"
            )
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        eval_error = exc
        logger.error(f"Eval aborted by exception: {exc}")
    finally:
        if should_disable_temporal:
            manager.mid_memory.set_temporal_weight_disabled(prev_temporal_disabled)
        accuracy = (float(matched) / float(total)) if total else 0.0
        retrieval_span_rate = (
            (float(retrieval_span_hits) / float(retrieval_total)) if retrieval_total else 0.0
        )
        retrieval_support_rate = (
            float(retrieval_support_hits) / float(retrieval_total)
        ) if retrieval_total else 0.0
        retrieval_evidence_rate = (
            float(retrieval_evidence_hits) / float(retrieval_total)
        ) if retrieval_total else 0.0
        graph_retrieval_hit_rate = (
            float(graph_retrieval_hits) / float(graph_retrieval_total)
        ) if graph_retrieval_total else 0.0
        long_stats = manager.long_memory.debug_stats()
        ingest_total = int(long_stats.get("ingest_event_total", 0))
        ingest_accepted = int(long_stats.get("ingest_event_accepted", 0))
        graph_ingest_accept_rate = (
            float(ingest_accepted) / float(ingest_total)
            if ingest_total > 0
            else 0.0
        )
        logger.info(f"Eval mode completed: total={total}, matched={matched}, accuracy={accuracy:.4f}")
        print(f"Eval summary: total={total}, matched={matched}, final_answer_acc={accuracy:.4f}")
        print(
            "Retrieval summary: "
            f"answer_span_hit_rate={retrieval_span_rate:.4f}, "
            f"support_sentence_hit_rate={retrieval_support_rate:.4f}, "
            f"evidence_hit_rate={retrieval_evidence_rate:.4f}, "
            f"graph_retrieval_hit_rate={graph_retrieval_hit_rate:.4f}"
        )
        print(
            "Graph ingest summary: "
            f"accepted={ingest_accepted}, total={ingest_total}, "
            f"graph_ingest_accept_rate={graph_ingest_accept_rate:.4f}"
        )
        if group_by_type and grouped:
            print("Eval by question_type:")
            for key in sorted(grouped.keys()):
                g_total = grouped[key]["total"]
                g_matched = grouped[key]["matched"]
                g_acc = (float(g_matched) / float(g_total)) if g_total else 0.0
                manager.mid_memory.log_eval_group_result(
                    run_id, key, g_total, g_matched, g_acc, commit=False
                )
                print(f"- {key}: total={g_total}, matched={g_matched}, accuracy={g_acc:.4f}")
        manager.mid_memory.log_eval_run_finish(
            run_id,
            total,
            matched,
            accuracy,
            retrieval_answer_span_hit_rate=retrieval_span_rate,
            retrieval_support_sentence_hit_rate=retrieval_support_rate,
            retrieval_evidence_hit_rate=retrieval_evidence_rate,
            graph_retrieval_hit_rate=graph_retrieval_hit_rate,
            graph_ingest_accept_rate=graph_ingest_accept_rate,
            commit=False,
        )
        manager.mid_memory.commit()

    if eval_error is not None:
        raise eval_error
