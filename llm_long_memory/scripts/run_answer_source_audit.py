"""Audit answer-source quality without calling final answer generation LLM.

This script runs ingestion + retrieval (+ optional offline long-graph build) and
exports six source channels per question so we can inspect what should be used
as primary evidence vs fallback.
"""

from __future__ import annotations

import argparse
import json
import re
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
    parser.add_argument(
        "--disable-long-memory",
        action="store_true",
        help="Disable long memory/graph path for this audit run.",
    )
    parser.add_argument(
        "--disable-toolkit",
        action="store_true",
        help="Disable specialist toolkit hints/fallback for this audit run.",
    )
    parser.add_argument(
        "--enable-evidence-graph",
        action="store_true",
        help="Also export filtered evidence, extracted claims, and light graph.",
    )
    parser.add_argument(
        "--enable-evidence-filter",
        action="store_true",
        help="Export filtered evidence pack derived from unified RAG source.",
    )
    parser.add_argument(
        "--enable-evidence-claims",
        action="store_true",
        help="Run 8B fixed-schema claim extraction on filtered evidence.",
    )
    parser.add_argument(
        "--enable-evidence-light-graph",
        action="store_true",
        help="Build deterministic light graph from extracted claims.",
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


def _normalize_text_key(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _pack_lines_to_items(evidence_pack: Dict[str, object], limit: int = 8) -> List[Dict[str, object]]:
    lines = [str(x).strip() for x in list(evidence_pack.get("lines", [])) if str(x).strip()]
    out: List[Dict[str, object]] = []
    for raw in lines[: max(1, int(limit))]:
        text = raw
        if text.startswith("- "):
            text = text[2:].strip()
        out.append(
            {
                "text": text,
                "score": 0.0,
                "chunk_id": 0,
                "session_date": "",
            }
        )
    return out


def _label_items(
    items: List[Dict[str, object]],
    channel: str,
    limit: int,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for item in items[: max(1, int(limit))]:
        out.append(
            {
                "text": _as_text(item.get("text", "")).strip(),
                "score": float(item.get("score", 0.0)),
                "chunk_id": int(item.get("chunk_id", 0) or 0),
                "session_date": _as_text(item.get("session_date", "")),
                "channel": channel,
            }
        )
    return out


def _unified_rag_source(
    *,
    evidence_sentences: List[Dict[str, object]],
    evidence_pack: Dict[str, object],
    plan_keywords: List[Dict[str, object]],
    plan_evidence: List[Dict[str, object]],
    rag_limit: int = 5,
    pack_limit: int = 8,
    plan_kw_limit: int = 6,
    plan_ev_limit: int = 5,
    total_limit: int = 20,
) -> List[Dict[str, object]]:
    primary = _label_items(_top_texts(evidence_sentences, limit=rag_limit), "rag_evidence", rag_limit)
    pack_items = _label_items(
        _pack_lines_to_items(evidence_pack, limit=pack_limit),
        "evidence_pack",
        pack_limit,
    )
    plan_kw_items = _label_items(plan_keywords, "plan_keywords", plan_kw_limit)
    plan_ev_items = _label_items(plan_evidence, "plan_combined_evidence", plan_ev_limit)
    merged: List[Dict[str, object]] = []
    seen: set[str] = set()
    for item in primary + pack_items + plan_ev_items + plan_kw_items:
        text = _as_text(item.get("text", "")).strip()
        if not text:
            continue
        key = _normalize_text_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "text": text,
                "score": float(item.get("score", 0.0)),
                "chunk_id": int(item.get("chunk_id", 0) or 0),
                "session_date": _as_text(item.get("session_date", "")),
                "channel": _as_text(item.get("channel", "")),
            }
        )
        if len(merged) >= max(1, int(total_limit)):
            break
    return merged


def _keyword_items(plan: Dict[str, object], limit: int = 6) -> List[Dict[str, object]]:
    kws = [str(x).strip() for x in list(plan.get("plan_keywords", [])) if str(x).strip()]
    must = [str(x).strip() for x in list(plan.get("must_keywords", [])) if str(x).strip()]
    constraints = [str(x).strip() for x in list(plan.get("constraint_keywords", [])) if str(x).strip()]
    merged: List[str] = []
    seen: set[str] = set()
    for k in must + constraints + kws:
        low = k.lower()
        if low in seen:
            continue
        seen.add(low)
        merged.append(k)
        if len(merged) >= max(1, int(limit)):
            break
    out: List[Dict[str, object]] = []
    for k in merged:
        out.append({"text": k, "score": 1.0, "chunk_id": 0, "session_date": ""})
    return out


def _keyword_combined_evidence(
    chunks: List[Dict[str, object]],
    plan: Dict[str, object],
    limit: int = 5,
) -> List[Dict[str, object]]:
    must = [str(x).strip().lower() for x in list(plan.get("must_keywords", [])) if str(x).strip()]
    constraints = [
        str(x).strip().lower() for x in list(plan.get("constraint_keywords", [])) if str(x).strip()
    ]
    keys = [str(x).strip().lower() for x in list(plan.get("plan_keywords", [])) if str(x).strip()]
    ranked: List[Dict[str, object]] = []
    for item in chunks:
        text = _as_text(item.get("text", "")).strip()
        if not text:
            continue
        low = text.lower()
        must_hit = sum(1 for k in must if k and k in low)
        if must and must_hit <= 0:
            continue
        constraint_hit = sum(1 for k in constraints if k and k in low)
        key_hit = sum(1 for k in keys if k and k in low)
        base = float(item.get("score", 0.0))
        score = base + (0.6 * must_hit) + (0.25 * constraint_hit) + (0.15 * key_hit)
        ranked.append(
            {
                "text": text,
                "score": score,
                "chunk_id": int(item.get("chunk_id", 0) or 0),
                "session_date": _as_text(item.get("session_date", "")),
            }
        )
    ranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    out: List[Dict[str, object]] = []
    seen: set[str] = set()
    for item in ranked:
        norm = " ".join(_as_text(item.get("text", "")).lower().split())
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(item)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _score_row_quality(row: Dict[str, object]) -> Dict[str, object]:
    score = 0
    reasons: List[str] = []
    rag = list(row.get("source_1_rag_evidence_top", []))

    if rag:
        top = float(rag[0].get("score", 0.0))
        if top >= 0.40:
            score += 2
            reasons.append("rag_top_score>=0.40")
        elif top >= 0.28:
            score += 1
            reasons.append("rag_top_score>=0.28")
    channels = {_as_text(x.get("channel", "")) for x in rag}
    if "plan_keywords" in channels:
        score += 1
        reasons.append("has_plan_keywords")
    if "plan_combined_evidence" in channels:
        score += 1
        reasons.append("has_keyword_combined_evidence")
    if "evidence_pack" in channels:
        score += 1
        reasons.append("has_evidence_pack")

    tier = "low"
    if score >= 4:
        tier = "high"
    elif score >= 2:
        tier = "medium"
    return {"quality_tier": tier, "quality_score": score, "quality_reasons": reasons}


_AUDIT_TOKEN_RE = re.compile(r"[a-z0-9]+")
_AUDIT_NOISE_RE = re.compile(
    r"\b("
    r"tips?|how to|guide|tutorial|example script|i'm a large language model|"
    r"i do not have|don't have personal|recommendations?"
    r")\b"
)


def _audit_tokens(text: str) -> List[str]:
    return _AUDIT_TOKEN_RE.findall(_as_text(text).lower())


def _row_audit_metrics(row: Dict[str, object]) -> Dict[str, float]:
    items = list(row.get("source_1_rag_evidence_top", []))
    total = max(1, len(items))
    noisy = 0
    long_plan = 0
    gold_tokens = set(_audit_tokens(_as_text(row.get("gold", ""))))
    best_f1 = 0.0
    best_rec = 0.0

    for item in items:
        text = _as_text(item.get("text", ""))
        if _AUDIT_NOISE_RE.search(text.lower()):
            noisy += 1
        if _as_text(item.get("channel", "")) == "plan_combined_evidence" and len(text) > 600:
            long_plan += 1
        cand_tokens = set(_audit_tokens(text))
        if gold_tokens and cand_tokens:
            overlap = len(gold_tokens.intersection(cand_tokens))
            if overlap > 0:
                precision = float(overlap) / float(len(cand_tokens))
                recall = float(overlap) / float(len(gold_tokens))
                f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                best_f1 = max(best_f1, f1)
                best_rec = max(best_rec, recall)

    return {
        "noisy_ratio": float(noisy) / float(total),
        "long_plan_ratio": float(long_plan) / float(total),
        "best_f1": best_f1,
        "best_rec": best_rec,
    }


def _summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    total = max(1, len(rows))
    quality_high = 0
    quality_medium = 0
    quality_low = 0
    sum_quality = 0.0
    sum_noisy = 0.0
    sum_long_plan = 0.0
    sum_best_f1 = 0.0
    sum_best_rec = 0.0
    coverage_f1_pos = 0
    coverage_rec50 = 0

    for row in rows:
        tier = _as_text(row.get("quality_tier", ""))
        if tier == "high":
            quality_high += 1
        elif tier == "medium":
            quality_medium += 1
        else:
            quality_low += 1
        sum_quality += float(row.get("quality_score", 0.0))

        metrics = dict(row.get("audit_metrics", {}) or {})
        noisy_ratio = float(metrics.get("noisy_ratio", 0.0))
        long_plan_ratio = float(metrics.get("long_plan_ratio", 0.0))
        best_f1 = float(metrics.get("best_f1", 0.0))
        best_rec = float(metrics.get("best_rec", 0.0))
        sum_noisy += noisy_ratio
        sum_long_plan += long_plan_ratio
        sum_best_f1 += best_f1
        sum_best_rec += best_rec
        if best_f1 > 0.0:
            coverage_f1_pos += 1
        if best_rec >= 0.5:
            coverage_rec50 += 1

    return {
        "quality_high": quality_high,
        "quality_medium": quality_medium,
        "quality_low": quality_low,
        "avg_quality_score": sum_quality / float(total),
        "avg_noisy_ratio": sum_noisy / float(total),
        "avg_long_plan_ratio": sum_long_plan / float(total),
        "avg_best_f1": sum_best_f1 / float(total),
        "avg_best_rec": sum_best_rec / float(total),
        "coverage_f1_pos": coverage_f1_pos,
        "coverage_rec50": coverage_rec50,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage_filter = bool(args.enable_evidence_filter or args.enable_evidence_claims or args.enable_evidence_light_graph)
    stage_claims = bool(args.enable_evidence_claims or args.enable_evidence_light_graph)
    stage_light_graph = bool(args.enable_evidence_light_graph)
    if args.enable_evidence_graph:
        stage_filter = True
        stage_claims = True
        stage_light_graph = True
    if args.disable_long_memory:
        cfg["memory"]["long_memory"]["enabled"] = False
        cfg["memory"]["long_memory"].setdefault("offline_graph", {})["enabled"] = False
        cfg["retrieval"]["answering"]["graph_refiner_enabled"] = False
        cfg["retrieval"]["answering"]["graph_context_from_store_enabled"] = False
        cfg["retrieval"].setdefault("long_memory_context", {})["enabled"] = False
    if args.disable_toolkit:
        sp = cfg["retrieval"]["answering"].setdefault("specialist_layer", {})
        sp["enabled"] = False
        sp["modules"] = []
        sp["counting_enabled"] = False
        sp["graph_toolkit_enabled"] = False
        sp["allow_fallback_override"] = False
    evidence_graph_cfg = cfg["retrieval"].setdefault("evidence_graph", {})
    evidence_graph_cfg["enabled"] = bool(stage_filter or stage_claims or stage_light_graph)
    evidence_graph_cfg["filter_enabled"] = bool(stage_filter)
    evidence_graph_cfg["claims_enabled"] = bool(stage_claims)
    evidence_graph_cfg["light_graph_enabled"] = bool(stage_light_graph)
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
        _ctx, _topics, chunks = manager.retrieve_context(question)
        sentence_units = sum(1 for c in chunks if str(c.get("unit_type", "")) == "sentence")
        chunk_units = sum(1 for c in chunks if str(c.get("unit_type", "")) != "sentence")

        evidence_sentences = manager.answering.collect_evidence_sentences(question, chunks)
        evidence_pack = manager.chat_runtime._build_evidence_pack(
            query=question,
            evidence_sentences=evidence_sentences,
            chunks=chunks,
        )
        candidates = manager.answering.extract_candidates(question, evidence_sentences)
        evidence_candidate = manager.answering.extract_evidence_candidate(
            question, evidence_sentences, candidates
        )

        plan = dict(getattr(manager, "last_query_plan", {}) or {})
        plan_keywords = _keyword_items(plan, limit=6)
        combined_evidence = _keyword_combined_evidence(chunks, plan, limit=5)
        unified_source_1 = _unified_rag_source(
            evidence_sentences=evidence_sentences,
            evidence_pack=dict(evidence_pack or {}),
            plan_keywords=plan_keywords,
            plan_evidence=combined_evidence,
            rag_limit=5,
            pack_limit=8,
            plan_kw_limit=6,
            plan_ev_limit=5,
            total_limit=20,
        )

        row: Dict[str, object] = {
            "idx": i,
            "question_id": _as_text(item.get("question_id", "")),
            "question_type": _as_text(item.get("question_type", "")),
            "question": question,
            "gold": _as_text(item.get("answer", "")),
            # Unified single source output
            "source_1_rag_evidence_top": unified_source_1,
            "retrieved_chunks": len(chunks),
            "retrieved_chunk_units": int(chunk_units),
            "retrieved_sentence_units": int(sentence_units),
            "query_plan": {
                "intent": _as_text(plan.get("intent", "")),
                "answer_type": _as_text(plan.get("answer_type", "")),
                "focus_phrases": [str(x) for x in list(plan.get("focus_phrases", []))[:6]],
                "sub_queries": [str(x) for x in list(plan.get("sub_queries", []))[:6]],
            },
        }
        if stage_filter or stage_claims or stage_light_graph:
            graph_bundle = manager.build_evidence_graph_bundle(
                question,
                precomputed_context=(_ctx, _topics, chunks),
                enable_filter=stage_filter,
                enable_claims=stage_claims,
                enable_light_graph=stage_light_graph,
            )
            claim_result = dict(graph_bundle.get("claim_result", {}) or {})
            row.update(
                {
                    "evidence_graph_stage_flags": dict(graph_bundle.get("stage_flags", {}) or {}),
                    "evidence_graph_source": list(graph_bundle.get("unified_source", [])),
                    "filtered_evidence_pack": dict(graph_bundle.get("filtered_pack", {}) or {}),
                    "evidence_graph_claim_result": {
                        "enabled": bool(claim_result.get("enabled", False)),
                        "model": _as_text(claim_result.get("model", "")),
                        "claims": list(claim_result.get("claims", [])),
                        "stats": dict(claim_result.get("stats", {}) or {}),
                        "raw_batches": list(claim_result.get("raw_batches", [])),
                    },
                    "evidence_light_graph": dict(graph_bundle.get("light_graph", {}) or {}),
                }
            )
        row.update(_score_row_quality(row))
        row["audit_metrics"] = _row_audit_metrics(row)
        rows.append(row)

        claim_count = 0
        if stage_claims:
            claim_count = int(
                dict(row.get("evidence_graph_claim_result", {}) or {})
                .get("stats", {})
                .get("claims", 0)
            )
        print(
            f"[{i}/{total}] {row['question_id']} | {row['question_type']} | "
            f"unified_source={len(row['source_1_rag_evidence_top'])} "
            f"units=chunk:{row['retrieved_chunk_units']}/sent:{row['retrieved_sentence_units']} "
            f"quality={row['quality_tier']}:{row['quality_score']}"
            + (f" claims={claim_count}" if stage_claims else ""),
            flush=True,
        )

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(dataset_path).stem
    prefix = str(args.output_prefix or "answer_source_audit").strip()

    out_json = output_dir / f"{prefix}_{tag}__{stem}.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = _summarize_rows(rows)
    out_summary = output_dir / f"{prefix}_{tag}__{stem}__summary.json"
    out_summary.write_text(
        json.dumps(
            {
                "dataset": Path(dataset_path).name,
                "total": len(rows),
                "source_json": out_json.name,
                "metrics": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_md = output_dir / f"{prefix}_{tag}__{stem}.md"
    lines: List[str] = [
        "# Answer Source Audit",
        f"- dataset: {Path(dataset_path).name}",
        f"- total: {len(rows)}",
        f"- output_json: {out_json.name}",
        f"- output_summary_json: {out_summary.name}",
        f"- evidence_filter_enabled: {bool(stage_filter)}",
        f"- evidence_claims_enabled: {bool(stage_claims)}",
        f"- evidence_light_graph_enabled: {bool(stage_light_graph)}",
        "",
        "## Summary Metrics",
        f"- quality_high: {int(summary['quality_high'])}",
        f"- quality_medium: {int(summary['quality_medium'])}",
        f"- quality_low: {int(summary['quality_low'])}",
        f"- avg_quality_score: {float(summary['avg_quality_score']):.4f}",
        f"- avg_noisy_ratio: {float(summary['avg_noisy_ratio']):.4f}",
        f"- avg_long_plan_ratio: {float(summary['avg_long_plan_ratio']):.4f}",
        f"- avg_best_f1: {float(summary['avg_best_f1']):.4f}",
        f"- avg_best_rec: {float(summary['avg_best_rec']):.4f}",
        f"- coverage_f1_pos: {int(summary['coverage_f1_pos'])}",
        f"- coverage_rec50: {int(summary['coverage_rec50'])}",
        "",
    ]
    for row in rows:
        lines.append(f"## {int(row['idx']):02d} | {row['question_id']} | {row['question_type']}")
        lines.append(f"- Q: {row['question']}")
        lines.append(f"- Gold: {row['gold']}")
        lines.append(f"- Retrieved Chunks: {row['retrieved_chunks']}")
        lines.append(
            f"- Retrieved Units: chunk={row['retrieved_chunk_units']}, sentence={row['retrieved_sentence_units']}"
        )
        lines.append(f"- query_plan: {_as_text(row.get('query_plan', {}))}")
        lines.append(f"- quality: {row['quality_tier']} (score={row['quality_score']})")
        lines.append(f"- quality_reasons: {_as_text(row['quality_reasons'])}")
        if stage_filter or stage_claims or stage_light_graph:
            graph_stats = dict(dict(row.get("evidence_light_graph", {}) or {}).get("stats", {}) or {})
            filter_pack = dict(row.get("filtered_evidence_pack", {}) or {})
            claim_stats = dict(dict(row.get("evidence_graph_claim_result", {}) or {}).get("stats", {}) or {})
            lines.append(f"- evidence_graph_stage_flags: {_as_text(row.get('evidence_graph_stage_flags', {}))}")
            lines.append(
                "- evidence_graph_stats: "
                + json.dumps(
                    {
                        "source_count": len(list(row.get("evidence_graph_source", []))),
                        "core": len(list(filter_pack.get("core_evidence", []))),
                        "supporting": len(list(filter_pack.get("supporting_evidence", []))),
                        "conflict": len(list(filter_pack.get("conflict_evidence", []))),
                        "claims": int(claim_stats.get("claims", 0)),
                        "graph_nodes": int(graph_stats.get("node_count", 0)),
                        "graph_edges": int(graph_stats.get("edge_count", 0)),
                    },
                    ensure_ascii=False,
                )
            )
        lines.append("- source_1_rag_evidence_top:")
        for ev in list(row["source_1_rag_evidence_top"])[:10]:
            lines.append(
                "  - "
                f"[{_as_text(ev.get('channel', ''))}] "
                f"({float(ev.get('score', 0.0)):.4f}) "
                f"{_as_text(ev.get('text', ''))[:220]}"
            )
        if stage_filter or stage_claims or stage_light_graph:
            lines.append("- filtered_core_evidence:")
            for ev in list(dict(row.get("filtered_evidence_pack", {}) or {}).get("core_evidence", []))[:8]:
                lines.append("  - " + json.dumps(ev, ensure_ascii=False))
            lines.append("- filtered_supporting_evidence:")
            for ev in list(dict(row.get("filtered_evidence_pack", {}) or {}).get("supporting_evidence", []))[:6]:
                lines.append("  - " + json.dumps(ev, ensure_ascii=False))
            if stage_claims:
                lines.append("- evidence_graph_claims:")
                for claim in list(dict(row.get("evidence_graph_claim_result", {}) or {}).get("claims", []))[:8]:
                    lines.append("  - " + json.dumps(claim, ensure_ascii=False))
            if stage_light_graph:
                lines.append(
                    "- evidence_light_graph_stats: "
                    + json.dumps(
                        dict(dict(row.get("evidence_light_graph", {}) or {}).get("stats", {}) or {}),
                        ensure_ascii=False,
                    )
                )
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved_json {out_json}")
    print(f"saved_summary_json {out_summary}")
    print(f"saved_md {out_md}")
    manager.close()


if __name__ == "__main__":
    main()
