"""Audit answer-source quality without calling final answer generation LLM.

This script runs ingestion + retrieval (+ optional offline long-graph build) and
exports six source channels per question so we can inspect what should be used
as primary evidence vs fallback.
"""

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
    parser = argparse.ArgumentParser(description="Answer-source audit (no final answer generation).")
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
        default="answer_source_audit",
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


def _top_texts(items: List[Dict[str, object]], limit: int = 5) -> List[Dict[str, object]]:
    ranked = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    out: List[Dict[str, object]] = []
    for item in ranked[: max(0, limit)]:
        out.append(
            {
                "text": _as_text(item.get("text", "")).strip(),
                "score": float(item.get("score", 0.0)),
                "chunk_id": int(item.get("chunk_id", 0) or 0),
                "session_date": _as_text(item.get("session_date", "")),
            }
        )
    return out


def _score_row_quality(row: Dict[str, object]) -> Dict[str, object]:
    score = 0
    reasons: List[str] = []
    rag = list(row.get("source_1_rag_evidence_top", []))
    spans = list(row.get("source_2_span_candidates", []))
    ec = dict(row.get("source_3_evidence_candidate", {}) or {})
    graph_ctx = str(row.get("source_4_long_graph_context", "") or "").strip()
    graph_snips = list(row.get("graph_snippets", []))
    graph_paths = list(row.get("graph_ranked_debug", []))
    hints = str(row.get("source_5_toolkit_hints", "") or "").strip()

    if rag:
        top = float(rag[0].get("score", 0.0))
        if top >= 0.40:
            score += 2
            reasons.append("rag_top_score>=0.40")
        elif top >= 0.28:
            score += 1
            reasons.append("rag_top_score>=0.28")
    if spans:
        score += 1
        reasons.append("has_span_candidates")
    if ec and str(ec.get("answer", "")).strip():
        score += 1
        reasons.append("has_evidence_candidate")
    if graph_ctx:
        score += 1
        reasons.append("has_graph_context")
    if graph_snips:
        score += 1
        reasons.append("has_graph_snippets")
    if graph_paths:
        score += 1
        reasons.append("has_graph_paths")
    if hints:
        score += 1
        reasons.append("has_toolkit_hints")

    tier = "low"
    if score >= 6:
        tier = "high"
    elif score >= 3:
        tier = "medium"
    return {"quality_tier": tier, "quality_score": score, "quality_reasons": reasons}


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

        evidence_sentences = manager.answering.collect_evidence_sentences(question, chunks)
        candidates = manager.answering.extract_candidates(question, evidence_sentences)
        evidence_candidate = manager.answering.extract_evidence_candidate(
            question, evidence_sentences, candidates
        )

        graph_context = manager.chat_runtime.build_graph_context(query=question, chunks=chunks)
        graph_pack: Dict[str, object] = {}
        if getattr(manager, "graph_query_engine", None) is not None:
            try:
                graph_pack = dict(
                    manager.graph_query_engine.query(
                        query=question,
                        max_items=min(4, max(1, int(getattr(manager.long_memory, "context_max_items", 4)))),
                        include_debug=True,
                    )
                )
            except Exception:
                graph_pack = {}
        specialist_payload = manager.specialist_layer.run(
            query=question,
            graph_context=graph_context,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            chunks=chunks,
        )
        graph_tool_hints = _as_text(specialist_payload.get("hints", "")).strip()
        specialist_fallback = _as_text(specialist_payload.get("fallback_answer", "")).strip()
        prompt_fallback = manager.chat_runtime.resolve_prompt_fallback(
            fallback_answer=specialist_fallback,
            evidence_candidate=evidence_candidate,
            candidates=candidates,
            best_evidence=_as_text(evidence_sentences[0]["text"]) if evidence_sentences else "",
        )

        tool_answer = ""
        if manager.graph_toolkit is not None:
            try:
                tool_answer = _as_text(
                    manager.graph_toolkit.build_tool_answer(
                        query=question,
                        graph_context=graph_context,
                        evidence_sentences=evidence_sentences,
                        candidates=candidates,
                        chunks=chunks,
                    )
                ).strip()
            except Exception:
                tool_answer = ""

        row: Dict[str, object] = {
            "idx": i,
            "question_id": _as_text(item.get("question_id", "")),
            "question_type": _as_text(item.get("question_type", "")),
            "question": question,
            "gold": _as_text(item.get("answer", "")),
            # Six source channels
            "source_1_rag_evidence_top": _top_texts(evidence_sentences, limit=5),
            "source_2_span_candidates": _top_texts(candidates, limit=3),
            "source_3_evidence_candidate": dict(evidence_candidate or {}),
            "source_4_long_graph_context": graph_context,
            "source_5_toolkit_hints": graph_tool_hints,
            "source_6_toolkit_fallback": specialist_fallback or tool_answer,
            # Graph debug channels
            "graph_plan": dict(graph_pack.get("plan", {}) or {}),
            "graph_seed_debug": list(graph_pack.get("seed_debug", []) or []),
            "graph_ranked_debug": list(graph_pack.get("ranked_debug", []) or []),
            "graph_snippets": list(graph_pack.get("snippets", []) or []),
            # Extra diagnostics
            "prompt_fallback": prompt_fallback,
            "retrieved_chunks": len(chunks),
        }
        row.update(_score_row_quality(row))
        rows.append(row)

        print(
            f"[{i}/{total}] {row['question_id']} | {row['question_type']} | "
            f"rag={len(row['source_1_rag_evidence_top'])} cand={len(row['source_2_span_candidates'])} "
            f"ec={'yes' if row['source_3_evidence_candidate'] else 'no'} "
            f"graph={'yes' if graph_context else 'no'} "
            f"graph_path={'yes' if row['graph_ranked_debug'] else 'no'} "
            f"hint={'yes' if graph_tool_hints else 'no'} "
            f"fb={'yes' if (specialist_fallback or tool_answer) else 'no'} "
            f"quality={row['quality_tier']}:{row['quality_score']}",
            flush=True,
        )

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(dataset_path).stem
    prefix = str(args.output_prefix or "answer_source_audit").strip()

    out_json = output_dir / f"{prefix}_{tag}__{stem}.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = output_dir / f"{prefix}_{tag}__{stem}.md"
    lines: List[str] = [
        "# Answer Source Audit",
        f"- dataset: {Path(dataset_path).name}",
        f"- total: {len(rows)}",
        f"- output_json: {out_json.name}",
        "",
    ]
    for row in rows:
        lines.append(f"## {int(row['idx']):02d} | {row['question_id']} | {row['question_type']}")
        lines.append(f"- Q: {row['question']}")
        lines.append(f"- Gold: {row['gold']}")
        lines.append(f"- Retrieved Chunks: {row['retrieved_chunks']}")
        lines.append(f"- quality: {row['quality_tier']} (score={row['quality_score']})")
        lines.append(f"- quality_reasons: {_as_text(row['quality_reasons'])}")
        lines.append("- source_1_rag_evidence_top:")
        for ev in list(row["source_1_rag_evidence_top"])[:3]:
            lines.append(f"  - ({float(ev.get('score', 0.0)):.4f}) {_as_text(ev.get('text', ''))}")
        lines.append("- source_2_span_candidates:")
        for c in list(row["source_2_span_candidates"])[:3]:
            lines.append(f"  - ({float(c.get('score', 0.0)):.4f}) {_as_text(c.get('text', ''))}")
        lines.append(f"- source_3_evidence_candidate: {_as_text(row['source_3_evidence_candidate'])}")
        lines.append(f"- source_4_long_graph_context: {_as_text(row['source_4_long_graph_context'])[:280]}")
        lines.append(f"- graph_plan: {_as_text(row['graph_plan'])}")
        lines.append("- graph_seed_debug:")
        for seed in list(row["graph_seed_debug"])[:3]:
            lines.append(
                "  - "
                f"({float(seed.get('seed_score', 0.0)):.4f}) {_as_text(seed.get('event_id', ''))} "
                f"| {_as_text(seed.get('seed_sources', []))} | {_as_text(seed.get('seed_text', ''))[:120]}"
            )
        lines.append("- graph_ranked_debug:")
        for path_row in list(row["graph_ranked_debug"])[:3]:
            lines.append(
                "  - "
                f"({float(path_row.get('score', 0.0)):.4f}) path={_as_text(path_row.get('path', []))} "
                f"| {_as_text(path_row.get('text', ''))[:120]}"
            )
        lines.append("- graph_snippets:")
        for s in list(row["graph_snippets"])[:3]:
            lines.append(f"  - {_as_text(s)[:220]}")
        lines.append(f"- source_5_toolkit_hints: {_as_text(row['source_5_toolkit_hints'])[:280]}")
        lines.append(f"- source_6_toolkit_fallback: {_as_text(row['source_6_toolkit_fallback'])}")
        lines.append(f"- prompt_fallback: {_as_text(row['prompt_fallback'])}")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved_json {out_json}")
    print(f"saved_md {out_md}")
    manager.close()


if __name__ == "__main__":
    main()
