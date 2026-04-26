"""Audit stage-by-stage source quality without calling final answer generation LLM.

The audit follows the active MemSLM pipeline:
RAG -> filter -> claims -> light graph -> toolkit.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from llm_long_memory.llm.ollama_client import LLM
from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.evaluation.metrics_runtime import (
    compute_answer_token_density_from_texts,
    compute_noise_density_from_texts,
)
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
        help="Reserved no-op switch kept for CLI compatibility.",
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


def _stage_best_metrics(texts: List[str], gold: str, eval_cfg: Dict[str, object]) -> Dict[str, float]:
    gold_tokens = set(_audit_tokens(gold))
    best_f1 = 0.0
    best_rec = 0.0
    for text in texts:
        cand_tokens = set(_audit_tokens(text))
        if not gold_tokens or not cand_tokens:
            continue
        overlap = len(gold_tokens.intersection(cand_tokens))
        if overlap <= 0:
            continue
        precision = float(overlap) / float(len(cand_tokens))
        recall = float(overlap) / float(len(gold_tokens))
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        best_f1 = max(best_f1, f1)
        best_rec = max(best_rec, recall)
    return {
        "best_f1": best_f1,
        "best_rec": best_rec,
        "coverage_f1_pos": 1 if best_f1 > 0.0 else 0,
        "coverage_rec50": 1 if best_rec >= 0.5 else 0,
        "answer_token_density": compute_answer_token_density_from_texts(gold, texts, eval_cfg),
        "noise_density": compute_noise_density_from_texts(gold, texts, eval_cfg),
    }


def _claim_to_text(claim: Dict[str, object]) -> str:
    subject = _as_text(claim.get("subject", "")).strip()
    predicate = _as_text(claim.get("predicate", "")).strip()
    value = _as_text(claim.get("value", "")).strip()
    return " | ".join([x for x in [subject, predicate, value] if x]).strip()


def _stage_texts_from_row(row: Dict[str, object]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {
        "rag": [],
        "filter": [],
        "claims": [],
        "light_graph": [],
        "toolkit": [],
        "final_prompt": [],
    }
    for item in list(row.get("source_1_rag_evidence_top", [])):
        text = _as_text(item.get("text", "")).strip()
        if text:
            out["rag"].append(text)

    filtered = dict(row.get("filtered_evidence_pack", {}) or {})
    for key in ("core_evidence", "supporting_evidence", "conflict_evidence"):
        for item in list(filtered.get(key, [])):
            text = _as_text(item.get("text", "")).strip()
            if text:
                out["filter"].append(text)

    claim_result = dict(row.get("evidence_graph_claim_result", {}) or {})
    for unit in list(claim_result.get("support_units", [])):
        text = _as_text(unit.get("verbatim_span", "")).strip() or _as_text(unit.get("text", "")).strip()
        if text:
            out["claims"].append(text)
    for claim in list(claim_result.get("claims", [])):
        text = _claim_to_text(dict(claim))
        if text:
            out["claims"].append(text)
        value = _as_text(claim.get("value", "")).strip()
        if value:
            out["claims"].append(value)

    graph = dict(row.get("evidence_light_graph", {}) or {})
    for node in list(graph.get("nodes", [])):
        if _as_text(node.get("type", "")).strip() not in {"fact", "state", "event"}:
            continue
        label = _as_text(node.get("label", "")).strip()
        if label:
            out["light_graph"].append(label)
        meta = dict(node.get("meta", {}) or {})
        value = _as_text(meta.get("value", "")).strip()
        if value:
            out["light_graph"].append(value)

    toolkit = dict(row.get("toolkit_output", {}) or {})
    tool_payload = dict(toolkit.get("tool_payload", {}) or {})
    answer_candidate = _as_text(tool_payload.get("answer_candidate", "")).strip()
    if answer_candidate:
        out["toolkit"].append(answer_candidate)
    for line in list(tool_payload.get("summary_lines", [])):
        text = _as_text(line).strip()
        if text:
            out["toolkit"].append(text)
    hints = _as_text(toolkit.get("hints", "")).strip()
    if hints:
        out["toolkit"].extend([x.strip() for x in hints.splitlines() if x.strip()])
    final_prompt = _as_text(row.get("final_answer_prompt", "")).strip()
    if final_prompt:
        out["final_prompt"].extend([x.strip() for x in final_prompt.splitlines() if x.strip()])
    return out


def _row_stage_metrics(
    row: Dict[str, object],
    eval_cfg: Dict[str, object],
) -> Dict[str, Dict[str, float]]:
    texts = _stage_texts_from_row(row)
    gold = _as_text(row.get("gold", ""))
    return {stage: _stage_best_metrics(items, gold, eval_cfg) for stage, items in texts.items()}


def _summarize_stage_metrics(rows: List[Dict[str, object]]) -> Dict[str, object]:
    stage_names = ["rag", "filter", "claims", "light_graph", "toolkit", "final_prompt"]
    overall: Dict[str, Dict[str, float]] = {}
    by_type: Dict[str, Dict[str, Dict[str, float]]] = {}

    def _aggregate(bucket: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
        total = max(1, len(bucket))
        out: Dict[str, Dict[str, float]] = {}
        for stage in stage_names:
            sum_f1 = 0.0
            sum_rec = 0.0
            sum_answer_density = 0.0
            sum_noise_density = 0.0
            cov_f1 = 0
            cov_rec50 = 0
            for row in bucket:
                metrics = dict(dict(row.get("stage_metrics", {}) or {}).get(stage, {}) or {})
                sum_f1 += float(metrics.get("best_f1", 0.0))
                sum_rec += float(metrics.get("best_rec", 0.0))
                sum_answer_density += float(metrics.get("answer_token_density", 0.0))
                sum_noise_density += float(metrics.get("noise_density", 0.0))
                cov_f1 += int(metrics.get("coverage_f1_pos", 0))
                cov_rec50 += int(metrics.get("coverage_rec50", 0))
            out[stage] = {
                "avg_best_f1": sum_f1 / float(total),
                "avg_best_rec": sum_rec / float(total),
                "avg_answer_token_density": sum_answer_density / float(total),
                "avg_noise_density": sum_noise_density / float(total),
                "coverage_f1_pos": cov_f1,
                "coverage_rec50": cov_rec50,
            }
        return out

    overall = _aggregate(rows)
    buckets: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        qtype = _as_text(row.get("question_type", "")).strip() or "unknown"
        buckets.setdefault(qtype, []).append(row)
    for qtype, bucket in sorted(buckets.items()):
        by_type[qtype] = _aggregate(bucket)
    return {"overall": overall, "by_type": by_type}


def _summarize_stage_latency(rows: List[Dict[str, object]]) -> Dict[str, object]:
    stage_names = ["rag", "filter", "claims", "light_graph", "toolkit", "composer", "total"]

    def _aggregate(bucket: List[Dict[str, object]]) -> Dict[str, float]:
        total = max(1, len(bucket))
        out: Dict[str, float] = {}
        for stage in stage_names:
            out[stage] = (
                sum(
                    float(dict(row.get("stage_latency_sec", {}) or {}).get(stage, 0.0) or 0.0)
                    for row in bucket
                )
                / float(total)
            )
        return out

    overall = _aggregate(rows)
    by_type: Dict[str, Dict[str, float]] = {}
    buckets: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        qtype = _as_text(row.get("question_type", "")).strip() or "unknown"
        buckets.setdefault(qtype, []).append(row)
    for qtype, bucket in sorted(buckets.items()):
        by_type[qtype] = _aggregate(bucket)
    return {"overall": overall, "by_type": by_type}


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
        cfg["retrieval"]["answering"]["graph_refiner_enabled"] = False
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
        total_started = time.perf_counter()
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
        rag_started = time.perf_counter()
        _ctx, _topics, chunks = manager.retrieve_context(question)
        sentence_units = sum(1 for c in chunks if str(c.get("unit_type", "")) == "sentence")
        chunk_units = sum(1 for c in chunks if str(c.get("unit_type", "")) != "sentence")

        evidence_sentences = manager.answer_grounding.collect_evidence_sentences(question, chunks)
        evidence_pack = manager.chat_runtime._build_evidence_pack(
            query=question,
            evidence_sentences=evidence_sentences,
            chunks=chunks,
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
        stage_latency_sec: Dict[str, float] = {
            "rag": time.perf_counter() - rag_started,
            "filter": 0.0,
            "claims": 0.0,
            "light_graph": 0.0,
            "toolkit": 0.0,
            "composer": 0.0,
            "total": 0.0,
        }

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
            bundle_stage_latency = dict(graph_bundle.get("stage_latency_sec", {}) or {})
            for key in ("filter", "claims", "light_graph"):
                stage_latency_sec[key] = float(bundle_stage_latency.get(key, 0.0) or 0.0)
            toolkit_started = time.perf_counter()
            claim_result = dict(graph_bundle.get("claim_result", {}) or {})
            toolkit_output = manager.specialist_layer.run(
                query=question,
                graph_bundle=graph_bundle,
            )
            toolkit_latency = float(dict(toolkit_output or {}).get("latency_sec", 0.0) or 0.0)
            stage_latency_sec["toolkit"] = toolkit_latency or (time.perf_counter() - toolkit_started)
            composer_started = time.perf_counter()
            route_packet = {}
            router = getattr(manager, "final_answer_router", None)
            if router is not None:
                route_packet = router.route(
                    query=question,
                    filtered_pack=dict(graph_bundle.get("filtered_pack", {}) or {}),
                    claim_result=claim_result,
                    light_graph=dict(graph_bundle.get("light_graph", {}) or {}),
                    toolkit_payload=dict(toolkit_output or {}),
                )
            answer_rules_text = (
                router.build_answer_rules(route_packet, prompt_mode="compact")
                if router is not None and route_packet
                else None
            )
            final_prompt, prompt_sections = manager.final_answer_composer.build_prompt(
                input_text=question,
                filtered_pack=dict(graph_bundle.get("filtered_pack", {}) or {}),
                claim_result=claim_result,
                light_graph=dict(graph_bundle.get("light_graph", {}) or {}),
                toolkit_payload=dict(toolkit_output or {}),
                prompt_mode="compact",
                route_packet=route_packet,
                answer_rules_text=answer_rules_text,
            )
            stage_latency_sec["composer"] = time.perf_counter() - composer_started
            row.update(
                {
                    "evidence_graph_stage_flags": dict(graph_bundle.get("stage_flags", {}) or {}),
                    "evidence_graph_source": list(graph_bundle.get("unified_source", [])),
                    "filtered_evidence_pack": dict(graph_bundle.get("filtered_pack", {}) or {}),
                    "evidence_graph_claim_result": {
                        "enabled": bool(claim_result.get("enabled", False)),
                        "model": _as_text(claim_result.get("model", "")),
                        "support_units": list(claim_result.get("support_units", [])),
                        "claims": list(claim_result.get("claims", [])),
                        "stats": dict(claim_result.get("stats", {}) or {}),
                        "raw_batches": list(claim_result.get("raw_batches", [])),
                    },
                    "evidence_light_graph": dict(graph_bundle.get("light_graph", {}) or {}),
                    "toolkit_output": dict(toolkit_output or {}),
                    "final_answer_route": dict(route_packet or {}),
                    "final_answer_prompt": final_prompt,
                    "final_answer_prompt_sections": list(prompt_sections),
                    "final_answer_prompt_char_count": len(final_prompt),
                }
            )
        stage_latency_sec["total"] = time.perf_counter() - total_started
        row["stage_latency_sec"] = stage_latency_sec
        row.update(_score_row_quality(row))
        row["audit_metrics"] = _row_audit_metrics(row)
        row["stage_metrics"] = _row_stage_metrics(row, cfg["evaluation"])
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
    stage_summary = _summarize_stage_metrics(rows)
    latency_summary = _summarize_stage_latency(rows)
    out_summary = output_dir / f"{prefix}_{tag}__{stem}__summary.json"
    out_summary.write_text(
        json.dumps(
            {
                "dataset": Path(dataset_path).name,
                "total": len(rows),
                "source_json": out_json.name,
                "metrics": summary,
                "stage_metrics": stage_summary,
                "stage_latency_sec": latency_summary,
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
        "## Stage Metrics (Overall)",
    ]
    for stage, metrics in stage_summary["overall"].items():
        lines.append(
            f"- {stage}: avg_best_f1={float(metrics['avg_best_f1']):.4f}, "
            f"avg_best_rec={float(metrics['avg_best_rec']):.4f}, "
            f"avg_answer_token_density={float(metrics['avg_answer_token_density']):.4f}, "
            f"avg_noise_density={float(metrics['avg_noise_density']):.4f}, "
            f"coverage_f1_pos={int(metrics['coverage_f1_pos'])}, "
            f"coverage_rec50={int(metrics['coverage_rec50'])}"
        )
    lines.extend(
        [
            "",
            "## Stage Latency Sec (Overall)",
        ]
    )
    for stage, avg_latency in latency_summary["overall"].items():
        lines.append(f"- {stage}: {float(avg_latency):.4f}s")
    lines.extend(
        [
            "",
            "## Stage Latency Sec (By Type)",
        ]
    )
    for qtype, latency_map in latency_summary["by_type"].items():
        lines.append(f"- {qtype}:")
        for stage, avg_latency in latency_map.items():
            lines.append(f"  - {stage}: {float(avg_latency):.4f}s")
    lines.extend(
        [
            "",
            "## Stage Metrics (By Type)",
        ]
    )
    for qtype, stage_map in stage_summary["by_type"].items():
        lines.append(f"- {qtype}:")
        for stage, metrics in stage_map.items():
            lines.append(
                f"  - {stage}: avg_best_f1={float(metrics['avg_best_f1']):.4f}, "
                f"avg_best_rec={float(metrics['avg_best_rec']):.4f}, "
                f"avg_answer_token_density={float(metrics['avg_answer_token_density']):.4f}, "
                f"avg_noise_density={float(metrics['avg_noise_density']):.4f}, "
                f"coverage_f1_pos={int(metrics['coverage_f1_pos'])}, "
                f"coverage_rec50={int(metrics['coverage_rec50'])}"
            )
    lines.extend(
        [
            "",
        ]
    )
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
        lines.append(f"- stage_metrics: {_as_text(row.get('stage_metrics', {}))}")
        lines.append(f"- stage_latency_sec: {_as_text(row.get('stage_latency_sec', {}))}")
        if stage_filter or stage_claims or stage_light_graph:
            graph_stats = dict(dict(row.get("evidence_light_graph", {}) or {}).get("stats", {}) or {})
            filter_pack = dict(row.get("filtered_evidence_pack", {}) or {})
            claim_stats = dict(dict(row.get("evidence_graph_claim_result", {}) or {}).get("stats", {}) or {})
            lines.append(
                f"- final_answer_prompt: chars={int(row.get('final_answer_prompt_char_count', 0) or 0)}, "
                f"sections={_as_text([x.get('section', '') for x in list(row.get('final_answer_prompt_sections', []))])}"
            )
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
            toolkit = dict(row.get("toolkit_output", {}) or {})
            tool_payload = dict(toolkit.get("tool_payload", {}) or {})
            if toolkit or tool_payload:
                lines.append("- toolkit_output:")
                lines.append("  - " + json.dumps(tool_payload, ensure_ascii=False))
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved_json {out_json}")
    print(f"saved_summary_json {out_summary}")
    print(f"saved_md {out_md}")
    manager.close()


if __name__ == "__main__":
    main()
