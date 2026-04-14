"""Offline graph evaluation with per-instance checkpoint persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_long_memory.evaluation.dataset_loader import iter_history_messages, load_stream
from llm_long_memory.evaluation.metrics_runtime import (
    compute_answer_span_hit,
    compute_support_sentence_hit,
)
from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.utils.logger import logger


def _build_summary(rows: List[Dict[str, Any]], started_at: datetime) -> Dict[str, float]:
    total = len(rows)
    elapsed_sec = (datetime.now() - started_at).total_seconds()
    if total <= 0:
        return {
            "total": 0,
            "mid_answer_span_hit_rate": 0.0,
            "mid_support_sentence_hit_rate": 0.0,
            "graph_answer_span_hit_rate": 0.0,
            "graph_support_sentence_hit_rate": 0.0,
            "graph_non_empty_ratio": 0.0,
            "graph_accept_nonzero_ratio": 0.0,
            "avg_accepted_events": 0.0,
            "avg_graph_snippets": 0.0,
            "avg_extractor_failures": 0.0,
            "avg_extractor_json_success": 0.0,
            "avg_extractor_schema_pass": 0.0,
            "avg_extractor_empty_payload": 0.0,
            "avg_extractor_retry_compact": 0.0,
            "elapsed_sec": elapsed_sec,
        }
    return {
        "total": total,
        "mid_answer_span_hit_rate": sum(1 for row in rows if row["mid_span"]) / float(total),
        "mid_support_sentence_hit_rate": sum(1 for row in rows if row["mid_support"]) / float(total),
        "graph_answer_span_hit_rate": sum(1 for row in rows if row["graph_span"]) / float(total),
        "graph_support_sentence_hit_rate": sum(1 for row in rows if row["graph_support"]) / float(total),
        "graph_non_empty_ratio": sum(1 for row in rows if row["snippets"] > 0) / float(total),
        "graph_accept_nonzero_ratio": sum(1 for row in rows if row["accepted"] > 0) / float(total),
        "avg_accepted_events": sum(int(row["accepted"]) for row in rows) / float(total),
        "avg_graph_snippets": sum(int(row["snippets"]) for row in rows) / float(total),
        "avg_extractor_failures": sum(int(row["extractor_failures"]) for row in rows) / float(total),
        "avg_extractor_json_success": sum(int(row["extractor_json_success"]) for row in rows) / float(total),
        "avg_extractor_schema_pass": sum(int(row["extractor_schema_pass"]) for row in rows) / float(total),
        "avg_extractor_empty_payload": sum(int(row["extractor_empty_payload"]) for row in rows) / float(total),
        "avg_extractor_retry_compact": sum(int(row["extractor_retry_compact"]) for row in rows) / float(total),
        "elapsed_sec": elapsed_sec,
    }


def _write_checkpoint(
    *,
    output_path: Path,
    dataset_path: str,
    rows: List[Dict[str, Any]],
    failure_buckets: Dict[str, int],
    started_at: datetime,
    interrupted: bool,
) -> None:
    payload = {
        "dataset": dataset_path,
        "summary": _build_summary(rows, started_at),
        "failure_buckets": failure_buckets,
        "rows": rows,
        "checkpoint": {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "interrupted": bool(interrupted),
            "completed_instances": len(rows),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_offline_graph_eval(
    *,
    manager: MemoryManager,
    dataset_path: str,
    config: Dict[str, Any],
    output_path: Path,
    max_instances: int = 0,
    checkpoint_every: int = 1,
) -> Path:
    """Run offline graph evaluation and persist checkpoint JSON after each batch."""
    started_at = datetime.now()
    rows: List[Dict[str, Any]] = []
    failure_buckets = {"mid_miss": 0, "graph_miss": 0, "both_miss": 0}
    eval_cfg = config["evaluation"]
    checkpoint_stride = max(1, int(checkpoint_every))
    limit = max(0, int(max_instances))

    logger.info(
        "Offline graph eval started: "
        f"path={dataset_path}, max_instances={limit or 'all'}, checkpoint_every={checkpoint_stride}."
    )

    interrupted = False
    try:
        for idx, instance in enumerate(load_stream(dataset_path), start=1):
            if limit > 0 and idx > limit:
                break

            manager.reset_for_new_instance()
            for message in iter_history_messages(instance):
                manager.ingest_message(message)
            manager.finalize_ingest()
            manager.archive_short_to_mid(clear_short=True)

            question = str(instance.get("question", "")).strip()
            expected = str(instance.get("answer", "")).strip()
            _ctx, _topics, chunks = manager.retrieve_context(question)
            accepted = int(manager.offline_build_long_graph_from_chunks(chunks, query=question))
            snippets = manager.long_memory.build_context_snippets(question)
            stats = manager.long_memory.debug_stats()

            mid_eval_chunks = [
                {"text": str(chunk.get("text", ""))}
                for chunk in chunks
                if str(chunk.get("text", "")).strip()
            ]
            graph_eval_chunks = [{"text": snippet} for snippet in snippets if str(snippet).strip()]
            mid_span = bool(compute_answer_span_hit(expected, mid_eval_chunks, eval_cfg))
            mid_support = bool(compute_support_sentence_hit(expected, mid_eval_chunks, eval_cfg))
            graph_span = bool(compute_answer_span_hit(expected, graph_eval_chunks, eval_cfg))
            graph_support = bool(compute_support_sentence_hit(expected, graph_eval_chunks, eval_cfg))

            if not mid_span:
                failure_buckets["mid_miss"] += 1
            if mid_span and (not graph_span):
                failure_buckets["graph_miss"] += 1
            if (not mid_span) and (not graph_span):
                failure_buckets["both_miss"] += 1

            row = {
                "idx": idx,
                "qid": str(instance.get("question_id", "")),
                "qtype": str(instance.get("question_type", "")),
                "chunks": len(chunks),
                "accepted": accepted,
                "events": int(stats.get("events", 0)),
                "snippets": len(snippets),
                "mid_span": mid_span,
                "mid_support": mid_support,
                "graph_span": graph_span,
                "graph_support": graph_support,
                "extractor_failures": int(stats.get("extractor_failures", 0)),
                "extractor_json_success": int(stats.get("extractor_json_success", 0)),
                "extractor_schema_pass": int(stats.get("extractor_schema_pass", 0)),
                "extractor_empty_payload": int(stats.get("extractor_empty_payload", 0)),
                "extractor_retry_compact": int(stats.get("extractor_retry_compact", 0)),
                "preview": snippets[0] if snippets else "",
            }
            rows.append(row)

            logger.info(
                "Offline graph eval row: "
                f"idx={idx}, qid={row['qid']}, accepted={accepted}, snippets={row['snippets']}, "
                f"mid_span={mid_span}, graph_span={graph_span}."
            )

            if (idx % checkpoint_stride) == 0:
                _write_checkpoint(
                    output_path=output_path,
                    dataset_path=dataset_path,
                    rows=rows,
                    failure_buckets=failure_buckets,
                    started_at=started_at,
                    interrupted=False,
                )
    except KeyboardInterrupt:
        interrupted = True
        logger.warn("Offline graph eval interrupted; writing checkpoint before exit.")
        raise
    finally:
        _write_checkpoint(
            output_path=output_path,
            dataset_path=dataset_path,
            rows=rows,
            failure_buckets=failure_buckets,
            started_at=started_at,
            interrupted=interrupted,
        )

    logger.info(
        "Offline graph eval finished: "
        f"rows={len(rows)}, output={output_path}."
    )
    return output_path
