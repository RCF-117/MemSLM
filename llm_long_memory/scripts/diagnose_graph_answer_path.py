from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import sqlite3

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_long_memory.evaluation.dataset_loader import iter_history_messages, load_stream
from llm_long_memory.evaluation.metrics_runtime import (
    compute_answer_span_hit,
    compute_support_sentence_hit,
)
from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.utils.helpers import load_config, resolve_project_path


class DryRunLLM:
    model_name = "dry-run-no-final-8b"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        raise RuntimeError("Final 8B should not be called in diagnose mode")


def _read_graph_corpus(db_path: Path) -> List[str]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    out: List[str] = []
    try:
        rows = conn.execute(
            """
            SELECT skeleton_text AS text FROM events
            UNION ALL
            SELECT text AS text FROM details
            UNION ALL
            SELECT node_text AS text FROM event_nodes
            """
        ).fetchall()
        for row in rows:
            text = str(row["text"] or "").strip()
            if text:
                out.append(text)
    finally:
        conn.close()
    return out


def main() -> None:
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    tmp_root = Path(tempfile.mkdtemp(prefix="graph_path_diag_"))
    mid_db = tmp_root / "mid.db"
    long_db = tmp_root / "long.db"
    cfg["memory"]["mid_memory"]["database_file"] = str(mid_db)
    cfg["memory"]["long_memory"]["database_file"] = str(long_db)
    cfg["memory"]["long_memory"]["enabled"] = True
    cfg["memory"]["long_memory"]["offline_graph"]["enabled"] = True
    cfg["memory"]["long_memory"]["extractor"]["model"] = "qwen3:8b"
    cfg["memory"]["long_memory"]["extractor"]["timeout_sec"] = 180
    cfg["memory"]["long_memory"]["extractor"]["retry_max_attempts"] = 1
    cfg["retrieval"]["answering"]["graph_refiner_enabled"] = True
    cfg["retrieval"]["answering"]["graph_context_from_store_enabled"] = True
    cfg["retrieval"]["answering"]["second_pass_llm_enabled"] = False
    cfg["retrieval"]["answering"]["llm_fallback_to_top_candidate"] = False

    dataset_path = resolve_project_path("data/raw/LongMemEval/longmemeval_ragdebug10_rebuilt.json")
    eval_cfg = cfg["evaluation"]

    rows: List[Dict[str, Any]] = []
    manager = MemoryManager(llm=DryRunLLM(), config=cfg)
    try:
        for idx, inst in enumerate(load_stream(str(dataset_path)), start=1):
            qid = str(inst.get("question_id", ""))
            qtype = str(inst.get("question_type", ""))
            question = str(inst.get("question", "")).strip()
            expected = str(inst.get("answer", "")).strip()
            manager.reset_for_new_instance()
            for msg in iter_history_messages(inst):
                manager.ingest_message(msg)
            manager.finalize_ingest()
            manager.archive_short_to_mid(clear_short=True)

            _ctx, _topics, chunks = manager.retrieve_context(question)
            accepted = int(manager.offline_build_long_graph_from_chunks(chunks, query=question))

            graph_store_chunks = [{"text": t} for t in _read_graph_corpus(long_db)]
            graph_query_items = manager.long_memory.query(question)
            graph_query_chunks = [{"text": str(item.get("text", ""))} for item in graph_query_items]
            graph_prompt_text = manager.chat_runtime.build_graph_context(question, chunks)
            graph_prompt_chunks = [{"text": graph_prompt_text}]
            rag_chunks = [{"text": str(x.get("text", ""))} for x in chunks]

            row = {
                "idx": idx,
                "qid": qid,
                "qtype": qtype,
                "accepted_events": accepted,
                "rag_span_hit": bool(compute_answer_span_hit(expected, rag_chunks, eval_cfg)),
                "rag_support_hit": bool(compute_support_sentence_hit(expected, rag_chunks, eval_cfg)),
                "graph_store_span_hit": bool(
                    compute_answer_span_hit(expected, graph_store_chunks, eval_cfg)
                ),
                "graph_store_support_hit": bool(
                    compute_support_sentence_hit(expected, graph_store_chunks, eval_cfg)
                ),
                "graph_query_span_hit": bool(
                    compute_answer_span_hit(expected, graph_query_chunks, eval_cfg)
                ),
                "graph_query_support_hit": bool(
                    compute_support_sentence_hit(expected, graph_query_chunks, eval_cfg)
                ),
                "graph_prompt_span_hit": bool(
                    compute_answer_span_hit(expected, graph_prompt_chunks, eval_cfg)
                ),
                "graph_prompt_support_hit": bool(
                    compute_support_sentence_hit(expected, graph_prompt_chunks, eval_cfg)
                ),
                "graph_query_top1_preview": str(graph_query_items[0].get("text", ""))[:220]
                if graph_query_items
                else "",
                "graph_prompt_preview": graph_prompt_text[:220],
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "idx": idx,
                        "qid": qid,
                        "qtype": qtype,
                        "accepted": accepted,
                        "rag_span": row["rag_span_hit"],
                        "store_span": row["graph_store_span_hit"],
                        "query_span": row["graph_query_span_hit"],
                        "prompt_span": row["graph_prompt_span_hit"],
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        manager.close()

    def _rate(key: str) -> float:
        return sum(1 for r in rows if bool(r[key])) / float(max(1, len(rows)))

    summary = {
        "dataset": str(dataset_path),
        "tmp_dir": str(tmp_root),
        "total": len(rows),
        "rates": {
            "rag_span_hit_rate": _rate("rag_span_hit"),
            "graph_store_span_hit_rate": _rate("graph_store_span_hit"),
            "graph_query_span_hit_rate": _rate("graph_query_span_hit"),
            "graph_prompt_span_hit_rate": _rate("graph_prompt_span_hit"),
            "rag_support_hit_rate": _rate("rag_support_hit"),
            "graph_store_support_hit_rate": _rate("graph_store_support_hit"),
            "graph_query_support_hit_rate": _rate("graph_query_support_hit"),
            "graph_prompt_support_hit_rate": _rate("graph_prompt_support_hit"),
        },
        "rows": rows,
    }
    out_path = resolve_project_path("data/processed/graph_answer_path_diag_ragdebug10.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(json.dumps(summary["rates"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

