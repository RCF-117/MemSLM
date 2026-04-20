"""Run toolkit-only evaluation on rag/debug-style datasets.

This script intentionally does NOT call final answer LLM generation.
It runs ingestion + mid/long retrieval + offline long-graph build, then
uses graph_reasoning_toolkit outputs for per-question inspection.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_long_memory.llm.ollama_client import LLM
from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.utils.helpers import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Toolkit-only judge eval (no final answer LLM call).")
    p.add_argument(
        "--config",
        default="llm_long_memory/config/config.yaml",
        help="Config file path.",
    )
    p.add_argument(
        "--dataset",
        default="llm_long_memory/data/raw/LongMemEval/longmemeval_ragdebug10_rebuilt.json",
        help="Dataset JSON path.",
    )
    p.add_argument(
        "--output-dir",
        default="llm_long_memory/data/processed/thesis_reports_debug_analysis",
        help="Output directory for toolkit-only artifacts.",
    )
    p.add_argument(
        "--output-prefix",
        default="",
        help="Optional output filename prefix.",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Optional max items to run (0 means all).",
    )
    return p.parse_args()


def _to_str(value: Any) -> str:
    return str(value) if value is not None else ""


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_path = resolve_project_path(str(args.dataset))
    output_dir = resolve_project_path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    items = json.loads(dataset_path.read_text(encoding="utf-8"))
    if int(args.max_items) > 0:
        items = items[: int(args.max_items)]

    llm = LLM()
    manager = MemoryManager(llm=llm, config=config)

    rows: list[dict[str, Any]] = []
    total = len(items)
    for i, item in enumerate(items, 1):
        manager.reset_for_new_instance()

        sessions = item.get("haystack_sessions", [])
        session_ids = list(item.get("haystack_session_ids", []))
        session_dates = list(item.get("haystack_dates", []))

        for sidx, session in enumerate(sessions):
            sid = _to_str(session_ids[sidx]) if sidx < len(session_ids) else f"s{sidx}"
            sdate = _to_str(session_dates[sidx]) if sidx < len(session_dates) else ""
            for tidx, msg in enumerate(session):
                manager.ingest_message(
                    {
                        "role": _to_str(msg.get("role", "user")),
                        "content": _to_str(msg.get("content", "")),
                        "has_answer": bool(msg.get("has_answer", False)),
                        "session_id": sid,
                        "session_date": sdate,
                        "turn_index": tidx,
                    }
                )

        manager.finalize_ingest()
        manager.archive_short_to_mid(clear_short=True)

        query = _to_str(item.get("question", ""))

        _, _, base_chunks = manager.retrieve_context(query)
        try:
            manager.offline_build_long_graph_from_chunks(chunks=base_chunks, query=query)
        except Exception:
            # Keep eval resilient; long-graph build failure should not block toolkit-only judging.
            pass

        _, _, chunks = manager.retrieve_context(query)
        evidence = manager.answering.collect_evidence_sentences(query, chunks)
        candidates = manager.answering.extract_candidates(query, evidence)
        graph_context = manager.chat_runtime.build_graph_context(query=query, chunks=chunks)

        prediction = ""
        tool_hints = ""
        if manager.graph_toolkit is not None:
            prediction = _to_str(
                manager.graph_toolkit.build_tool_answer(
                    query=query,
                    graph_context=graph_context,
                    evidence_sentences=evidence,
                    candidates=candidates,
                    chunks=chunks,
                )
            ).strip()
            tool_hints = _to_str(
                manager.graph_toolkit.build_tool_hints(
                    query=query,
                    graph_context=graph_context,
                    evidence_sentences=evidence,
                    candidates=candidates,
                    chunks=chunks,
                )
            )

        row = {
            "idx": i,
            "question_id": item.get("question_id", ""),
            "question_type": item.get("question_type", ""),
            "question": query,
            "gold": _to_str(item.get("answer", "")),
            "prediction": prediction,
            "tool_hints": tool_hints,
            "top_evidence": [_to_str(x.get("text", "")) for x in evidence[:5]],
            "retrieved_chunks": len(chunks),
        }
        rows.append(row)

        print(
            f"[{i}/{total}] qid={row['question_id']} type={row['question_type']} "
            f"pred={row['prediction'][:80]!r} gold={row['gold'][:80]!r}",
            flush=True,
        )

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.output_prefix.strip() or "toolkit_only_judge"
    stem = Path(dataset_path).stem

    out_json = output_dir / f"{prefix}_{tag}__{stem}.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = output_dir / f"{prefix}_{tag}__{stem}.md"
    lines = [
        "# Toolkit-only Judge Run",
        f"- dataset: {Path(dataset_path).name}",
        f"- total: {len(rows)}",
        f"- output_json: {out_json.name}",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['idx']:02d} | {row['question_id']} | {row['question_type']}",
                f"- Q: {row['question']}",
                f"- Pred: {row['prediction']}",
                f"- Gold: {row['gold']}",
                f"- Retrieved Chunks: {row['retrieved_chunks']}",
                "- Top Evidence:",
            ]
        )
        for ev in row["top_evidence"]:
            lines.append(f"  - {ev}")
        lines.extend(["", "- Tool Hints:", f"  - {row['tool_hints']}", ""])
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved_json {out_json}")
    print(f"saved_md {out_md}")

    manager.close()


if __name__ == "__main__":
    main()
