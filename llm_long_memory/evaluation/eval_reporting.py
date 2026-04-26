"""Evaluation reporting and run-finalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List

from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.utils.logger import logger


@dataclass
class EvalCounters:
    """Mutable counters collected during eval loop."""

    total: int = 0
    matched: int = 0
    retrieval_total: int = 0
    retrieval_span_hits: int = 0
    retrieval_support_hits: int = 0
    retrieval_evidence_hits: int = 0
    graph_retrieval_total: int = 0
    graph_span_hits: int = 0
    graph_support_hits: int = 0
    graph_ingest_selected: int = 0
    graph_ingest_accepted: int = 0


def finalize_eval_run(
    *,
    manager: MemoryManager,
    run_id: str,
    grouped: Dict[str, Dict[str, int]],
    group_by_type: bool,
    counters: EvalCounters,
) -> None:
    """Compute final metrics, print summary, and persist run-level results."""
    store = manager.mid_memory.eval_store
    result_rows: List[Dict[str, object]] = [dict(row) for row in store.get_eval_result_rows(run_id)]
    total = len(result_rows)
    matched = sum(1 for row in result_rows if int(row.get("is_match") or 0) == 1)
    accuracy = (float(matched) / float(total)) if total else 0.0

    def _avg(column: str) -> float:
        values = []
        for row in result_rows:
            value = row.get(column)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        return (sum(values) / float(len(values))) if values else 0.0

    retrieval_span_rate = _avg("answer_span_hit")
    retrieval_support_rate = _avg("support_sentence_hit")
    retrieval_evidence_rate = _avg("evidence_hit")
    graph_span_rate = _avg("graph_answer_span_hit")
    graph_support_rate = _avg("graph_support_sentence_hit")
    answer_density_rate = _avg("answer_token_density")
    noise_density_rate = _avg("noise_density")
    avg_latency_sec = _avg("latency_sec")

    grouped_from_db: Dict[str, Dict[str, int]] = {}
    grouped_latency: Dict[str, float] = {}
    if group_by_type and result_rows:
        buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in result_rows:
            key = str(row.get("question_type") or "unknown").strip() or "unknown"
            buckets[key].append(row)
        for key, bucket in buckets.items():
            grouped_from_db[key] = {
                "total": len(bucket),
                "matched": sum(1 for row in bucket if int(row.get("is_match") or 0) == 1),
            }
            latency_vals = [
                float(row.get("latency_sec"))
                for row in bucket
                if row.get("latency_sec") is not None
            ]
            answer_density_vals = [
                float(row.get("answer_token_density"))
                for row in bucket
                if row.get("answer_token_density") is not None
            ]
            noise_density_vals = [
                float(row.get("noise_density"))
                for row in bucket
                if row.get("noise_density") is not None
            ]
            grouped_latency[key] = (
                (sum(latency_vals) / float(len(latency_vals))) if latency_vals else 0.0
            )
            grouped_from_db[key]["type_answer_token_density"] = (
                (sum(answer_density_vals) / float(len(answer_density_vals)))
                if answer_density_vals
                else 0.0
            )
            grouped_from_db[key]["type_noise_density"] = (
                (sum(noise_density_vals) / float(len(noise_density_vals)))
                if noise_density_vals
                else 0.0
            )

    ingest_total = int(counters.graph_ingest_selected)
    ingest_accepted = int(counters.graph_ingest_accepted)
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
        f"graph_answer_span_hit_rate={graph_span_rate:.4f}, "
        f"graph_support_sentence_hit_rate={graph_support_rate:.4f}"
    )
    print(
        "Prompt density summary: "
        f"answer_token_density={answer_density_rate:.4f}, "
        f"noise_density={noise_density_rate:.4f}"
    )
    print(
        "Graph ingest summary: "
        f"accepted={ingest_accepted}, total={ingest_total}, "
        f"graph_ingest_accept_rate={graph_ingest_accept_rate:.4f}"
    )
    print(f"Latency summary: avg_latency_sec={avg_latency_sec:.4f}")

    store.delete_group_results(run_id)
    if group_by_type and grouped_from_db:
        print("Eval by question_type:")
        for key in sorted(grouped_from_db.keys()):
            g_total = grouped_from_db[key]["total"]
            g_matched = grouped_from_db[key]["matched"]
            g_acc = (float(g_matched) / float(g_total)) if g_total else 0.0
            manager.mid_memory.eval_store.log_eval_group_result(
                run_id, key, g_total, g_matched, g_acc, commit=False
            )
            manager.mid_memory.eval_store.log_thesis_type_metric(
                run_id,
                key,
                type_answer_acc=g_acc,
                type_answer_token_density=float(grouped_from_db[key].get("type_answer_token_density", 0.0)),
                type_noise_density=float(grouped_from_db[key].get("type_noise_density", 0.0)),
                type_latency_sec=float(grouped_latency.get(key, 0.0)),
                commit=False,
            )
            print(
                f"- {key}: total={g_total}, matched={g_matched}, "
                f"accuracy={g_acc:.4f}, "
                f"type_answer_token_density={float(grouped_from_db[key].get('type_answer_token_density', 0.0)):.4f}, "
                f"type_noise_density={float(grouped_from_db[key].get('type_noise_density', 0.0)):.4f}, "
                f"type_latency_sec={float(grouped_latency.get(key, 0.0)):.4f}"
            )

    manager.mid_memory.eval_store.log_eval_run_finish(
        run_id,
        total,
        matched,
        accuracy,
        retrieval_answer_span_hit_rate=retrieval_span_rate,
        retrieval_support_sentence_hit_rate=retrieval_support_rate,
        retrieval_evidence_hit_rate=retrieval_evidence_rate,
        graph_answer_span_hit_rate=graph_span_rate,
        graph_support_sentence_hit_rate=graph_support_rate,
        graph_ingest_accept_rate=graph_ingest_accept_rate,
        avg_answer_token_density=answer_density_rate,
        avg_noise_density=noise_density_rate,
        avg_latency_sec=avg_latency_sec,
        commit=False,
    )
    manager.mid_memory.eval_store.log_thesis_run_metrics(
        run_id,
        final_answer_acc=float(accuracy),
        retrieval_answer_span_hit_rate=retrieval_span_rate,
        retrieval_support_sentence_hit_rate=retrieval_support_rate,
        graph_answer_span_hit_rate=graph_span_rate,
        graph_support_sentence_hit_rate=graph_support_rate,
        graph_ingest_accept_rate=graph_ingest_accept_rate,
        avg_answer_token_density=answer_density_rate,
        avg_noise_density=noise_density_rate,
        avg_latency_sec=avg_latency_sec,
        commit=False,
    )
    manager.mid_memory.commit()
