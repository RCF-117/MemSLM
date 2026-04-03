"""Evaluation reporting and run-finalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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


def finalize_eval_run(
    *,
    manager: MemoryManager,
    run_id: str,
    grouped: Dict[str, Dict[str, int]],
    group_by_type: bool,
    counters: EvalCounters,
) -> None:
    """Compute final metrics, print summary, and persist run-level results."""
    total = int(counters.total)
    matched = int(counters.matched)
    retrieval_total = int(counters.retrieval_total)
    graph_retrieval_total = int(counters.graph_retrieval_total)

    accuracy = (float(matched) / float(total)) if total else 0.0
    retrieval_span_rate = (
        (float(counters.retrieval_span_hits) / float(retrieval_total)) if retrieval_total else 0.0
    )
    retrieval_support_rate = (
        float(counters.retrieval_support_hits) / float(retrieval_total)
    ) if retrieval_total else 0.0
    retrieval_evidence_rate = (
        float(counters.retrieval_evidence_hits) / float(retrieval_total)
    ) if retrieval_total else 0.0
    graph_span_rate = (
        float(counters.graph_span_hits) / float(graph_retrieval_total)
    ) if graph_retrieval_total else 0.0
    graph_support_rate = (
        float(counters.graph_support_hits) / float(graph_retrieval_total)
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
        f"graph_answer_span_hit_rate={graph_span_rate:.4f}, "
        f"graph_support_sentence_hit_rate={graph_support_rate:.4f}"
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
        graph_answer_span_hit_rate=graph_span_rate,
        graph_support_sentence_hit_rate=graph_support_rate,
        graph_ingest_accept_rate=graph_ingest_accept_rate,
        commit=False,
    )
    manager.mid_memory.commit()
