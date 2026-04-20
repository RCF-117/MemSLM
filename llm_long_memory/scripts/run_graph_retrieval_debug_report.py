"""Export per-question graph-retrieval debug report (no final answer generation)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from llm_long_memory.llm.ollama_client import LLM
from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.utils.helpers import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph retrieval debug report.")
    parser.add_argument(
        "--config",
        default="llm_long_memory/config/config.yaml",
        help="Config file path.",
    )
    parser.add_argument(
        "--dataset",
        default="llm_long_memory/data/raw/LongMemEval/longmemeval_ragdebug10_rebuilt.json",
        help="Dataset JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        default="llm_long_memory/data/processed/thesis_reports_debug_analysis",
        help="Output directory.",
    )
    parser.add_argument(
        "--output-prefix",
        default="graph_retrieval_debug",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Max instances to process (0 means all).",
    )
    return parser.parse_args()


def _as_text(value: Any) -> str:
    return str(value) if value is not None else ""


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_path = resolve_project_path(str(args.dataset))
    output_dir = resolve_project_path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if int(args.max_items) > 0:
        data = data[: int(args.max_items)]

    llm = LLM()
    manager = MemoryManager(llm=llm, config=cfg)

    rows: List[Dict[str, object]] = []
    total = len(data)
    for i, item in enumerate(data, start=1):
        manager.reset_for_new_instance()
        sessions = item.get("haystack_sessions", [])
        session_ids = list(item.get("haystack_session_ids", []))
        session_dates = list(item.get("haystack_dates", []))

        for sidx, session in enumerate(sessions):
            sid = _as_text(session_ids[sidx]) if sidx < len(session_ids) else f"s{sidx}"
            sdate = _as_text(session_dates[sidx]) if sidx < len(session_dates) else ""
            for tidx, msg in enumerate(session):
                manager.ingest_message(
                    {
                        "role": _as_text(msg.get("role", "user")).strip().lower() or "user",
                        "content": _as_text(msg.get("content", "")),
                        "has_answer": bool(msg.get("has_answer", False)),
                        "session_id": sid,
                        "session_date": sdate,
                        "turn_index": tidx,
                    }
                )
        manager.finalize_ingest()
        manager.archive_short_to_mid(clear_short=True)

        question = _as_text(item.get("question", ""))
        _ctx_before, _topics_before, chunks_before = manager.retrieve_context(question)
        try:
            manager.offline_build_long_graph_from_chunks(chunks=chunks_before, query=question)
        except Exception:
            pass
        _ctx, _topics, chunks = manager.retrieve_context(question)

        graph_pack: Dict[str, Any] = {}
        graph_snippets: List[str] = []
        if getattr(manager, "graph_query_engine", None) is not None:
            try:
                graph_pack = manager.graph_query_engine.query(
                    query=question,
                    max_items=min(4, max(1, int(getattr(manager.long_memory, "context_max_items", 4)))),
                    include_debug=True,
                )
                graph_snippets = [str(x).strip() for x in list(graph_pack.get("snippets", [])) if str(x).strip()]
            except Exception:
                graph_pack = {}
                graph_snippets = []

        graph_context = manager.chat_runtime.build_graph_context(query=question, chunks=chunks)
        rows.append(
            {
                "idx": i,
                "question_id": _as_text(item.get("question_id", "")),
                "question_type": _as_text(item.get("question_type", "")),
                "question": question,
                "gold": _as_text(item.get("answer", "")),
                "retrieved_chunks": int(len(chunks)),
                "graph_plan": graph_pack.get("plan", {}),
                "graph_seed_debug": graph_pack.get("seed_debug", []),
                "graph_ranked_debug": graph_pack.get("ranked_debug", []),
                "graph_snippets": graph_snippets,
                "graph_context": graph_context,
            }
        )
        print(
            f"[{i}/{total}] id={_as_text(item.get('question_id', ''))} "
            f"type={_as_text(item.get('question_type', ''))} "
            f"chunks={len(chunks)} snippets={len(graph_snippets)}"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_stem = dataset_path.stem
    base = f"{args.output_prefix}_{ts}__{dataset_stem}"
    out_json = output_dir / f"{base}.json"
    out_md = output_dir / f"{base}.md"

    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Graph Retrieval Debug Report")
    lines.append(f"- dataset: {dataset_path.name}")
    lines.append(f"- total: {len(rows)}")
    lines.append(f"- output_json: {out_json.name}")
    lines.append("")
    for row in rows:
        lines.append(
            f"## {int(row['idx']):02d} | {row['question_id']} | {row['question_type']}"
        )
        lines.append(f"- Q: {row['question']}")
        lines.append(f"- Gold: {row['gold']}")
        lines.append(f"- Retrieved Chunks: {row['retrieved_chunks']}")
        lines.append(f"- Graph Plan: {json.dumps(row['graph_plan'], ensure_ascii=False)}")
        seed_rows = list(row["graph_seed_debug"])[:4]
        lines.append("- Graph Seeds:")
        if seed_rows:
            for seed in seed_rows:
                lines.append(
                    "  - "
                    f"({float(seed.get('seed_score', 0.0)):.4f}) "
                    f"{_as_text(seed.get('event_id', ''))} | "
                    f"sources={_as_text(seed.get('seed_sources', []))} | "
                    f"text={_as_text(seed.get('seed_text', ''))[:140]}"
                )
        else:
            lines.append("  - (none)")
        ranked_rows = list(row["graph_ranked_debug"])[:4]
        lines.append("- Graph Paths:")
        if ranked_rows:
            for path_row in ranked_rows:
                lines.append(
                    "  - "
                    f"({float(path_row.get('score', 0.0)):.4f}) "
                    f"path={_as_text(path_row.get('path', []))} | "
                    f"text={_as_text(path_row.get('text', ''))[:140]}"
                )
        else:
            lines.append("  - (none)")
        snippets = list(row["graph_snippets"])[:3]
        lines.append("- Graph Snippets:")
        if snippets:
            for s in snippets:
                lines.append(f"  - {s[:220]}")
        else:
            lines.append("  - (none)")
        lines.append(f"- Final Graph Context: {_as_text(row['graph_context'])[:280]}")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved_json {out_json}")
    print(f"saved_md {out_md}")


if __name__ == "__main__":
    main()
