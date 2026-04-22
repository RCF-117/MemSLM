"""Consolidated thesis report builder from existing eval DB runs."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from llm_long_memory.evaluation.eval_store import EvalStore
from llm_long_memory.experiments.report_audit_utils import (
    iter_audit_summary_lines,
    load_latest_source_audit_summary,
)
from llm_long_memory.experiments.local_llm_judge import LocalLLMJudge
from llm_long_memory.utils.helpers import resolve_project_path, sanitize_filename_part


MODE_ORDER = ["model-only", "naive rag", "memslm", "ablation"]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_to_dict(row: sqlite3.Row | None) -> Dict[str, Any]:
    if row is None:
        return {}
    return {str(k): row[k] for k in row.keys()}


def _run_exists(conn: sqlite3.Connection, run_table: str, run_id: str | None) -> bool:
    candidate = str(run_id or "").strip()
    if not candidate:
        return False
    row = conn.execute(
        f"SELECT 1 FROM {run_table} WHERE run_id=? LIMIT 1",
        (candidate,),
    ).fetchone()
    return row is not None


def _collect_type_metrics(payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for row in payload.get("question_type_metrics", []):
        qtype = str(row.get("question_type", "")).strip()
        if not qtype:
            continue
        result[qtype] = {
            "type_answer_acc": float(row.get("type_answer_acc") or 0.0),
            "type_latency_sec": float(row.get("type_latency_sec") or 0.0),
        }
    return result


def _latest_compare_report(report_dir: Path, dataset_name: str | None = None) -> Dict[str, str]:
    candidates = sorted(report_dir.glob("*_comparison.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if dataset_name and str(payload.get("dataset", "")).strip() != str(dataset_name).strip():
            continue
        mapping: Dict[str, str] = {}
        for row in list(payload.get("modes", [])):
            mode = str(row.get("mode", "")).strip().lower()
            run_id = str(row.get("run_id", "")).strip()
            if mode and run_id:
                mapping[mode] = run_id
        if mapping:
            return mapping
    return {}


def _resolve_mode_run_id(
    *,
    conn: sqlite3.Connection,
    eval_store: EvalStore,
    mode: str,
    dataset_name: str | None,
    explicit_run_id: str | None,
    report_dir: Path,
) -> str:
    if explicit_run_id:
        candidate = str(explicit_run_id).strip()
        if _run_exists(conn, eval_store.run_table, candidate):
            return candidate
        raise ValueError(f"Explicit run_id not found in eval db for mode={mode}: {candidate}")
    if dataset_name:
        latest = eval_store.get_latest_thesis_mode_run(dataset_name=dataset_name, mode=mode)
        if latest and _run_exists(conn, eval_store.run_table, latest):
            return latest
    row = conn.execute(
        f"""
        SELECT run_id
        FROM {eval_store.thesis_mode_table}
        WHERE mode = ?
        ORDER BY datetime(created_at) DESC, run_id DESC
        LIMIT 1
        """,
        (mode,),
    ).fetchone()
    if row is not None and _run_exists(conn, eval_store.run_table, str(row["run_id"])):
        return str(row["run_id"])
    legacy = _latest_compare_report(report_dir, dataset_name=dataset_name)
    if mode in legacy and _run_exists(conn, eval_store.run_table, legacy[mode]):
        return legacy[mode]
    raise ValueError(
        f"Unable to resolve a valid run_id for mode={mode}. "
        f"Current eval db has no matching run in {eval_store.run_table}; "
        "old comparison artifacts are ignored unless their run_id still exists."
    )


def _load_mode_payload(
    *,
    db_path: str,
    run_id: str,
    eval_cfg: Dict[str, Any],
    judge_enabled: bool,
    judge_model: str | None,
) -> Dict[str, Any]:
    db_file = resolve_project_path(db_path)
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        store = EvalStore(conn=conn, eval_cfg=dict(eval_cfg))
        store.create_tables()
        store.ensure_schema_compat()
        run_table = store.run_table
        type_table = store.result_table
        run_row = conn.execute(f"SELECT * FROM {run_table} WHERE run_id=?", (run_id,)).fetchone()
        if run_row is None:
            raise ValueError(f"Run not found in eval db: {run_id}")
        type_rows = conn.execute(
            f"SELECT * FROM {type_table} WHERE run_id=? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        enriched_rows: List[Dict[str, Any]] = [{str(k): row[k] for k in row.keys()} for row in type_rows]
        if judge_enabled and enriched_rows:
            judge = LocalLLMJudge(model_name=judge_model or str(eval_cfg.get("judge_model", "")) or "qwen3:8b")
            judge_cache: Dict[tuple[str, str, str], Dict[str, Any]] = {}
            for row in enriched_rows:
                question = str(row.get("question", "")).strip()
                gold = str(row.get("expected_answer", "")).strip()
                prediction = str(row.get("prediction", "")).strip()
                cache_key = (question, gold, prediction)
                verdict = judge_cache.get(cache_key)
                if verdict is None:
                    result = judge.judge(question, gold, prediction)
                    verdict = {
                        "judge_is_correct": int(bool(result.is_correct)),
                        "judge_verdict": result.verdict,
                        "judge_reason": result.reason,
                    }
                    judge_cache[cache_key] = verdict
                row.update(verdict)
        else:
            for row in enriched_rows:
                row.setdefault("judge_is_correct", None)
                row.setdefault("judge_verdict", "")
                row.setdefault("judge_reason", "")

        summary = {str(k): run_row[k] for k in run_row.keys()}
        total = len(enriched_rows)
        matched = sum(1 for row in enriched_rows if int(row["is_match"] or 0) == 1)
        judge_matched = sum(1 for row in enriched_rows if int(row.get("judge_is_correct") or 0) == 1) if judge_enabled else None
        summary.update(
            {
                "run_id": run_id,
                "total": total,
                "matched": matched,
                "accuracy": (matched / total) if total else 0.0,
                "final_answer_acc": ((judge_matched / total) if (judge_enabled and total) else ((matched / total) if total else 0.0)),
            }
        )
        type_metrics = []
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for row in enriched_rows:
            key = str(row.get("question_type", "") or "unknown").strip() or "unknown"
            buckets.setdefault(key, []).append(row)
        for key in sorted(buckets.keys()):
            bucket = buckets[key]
            total_bucket = len(bucket)
            matched_bucket = sum(1 for row in bucket if int(row["is_match"] or 0) == 1)
            judge_bucket = (
                sum(1 for row in bucket if int(row.get("judge_is_correct") or 0) == 1)
                if judge_enabled
                else None
            )
            latency_vals = [float(row["latency_sec"]) for row in bucket if row.get("latency_sec") is not None]
            type_metrics.append(
                {
                    "question_type": key,
                    "total": total_bucket,
                    "matched": matched_bucket,
                    "type_answer_acc": (
                        (judge_bucket / total_bucket)
                        if (judge_enabled and total_bucket)
                        else ((matched_bucket / total_bucket) if total_bucket else None)
                    ),
                    "type_latency_sec": (
                        (sum(latency_vals) / float(len(latency_vals))) if latency_vals else None
                    ),
                }
            )
        return {"run": summary, "question_type_metrics": type_metrics}
    finally:
        conn.close()


def _write_comparison_report(
    *,
    output_dir: str,
    artifact_prefix: str,
    dataset_name: str,
    model_name: str,
    judge_model: str,
    mode_payloads: Dict[str, Dict[str, Any]],
    source_audit_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    out_dir = resolve_project_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_types = sorted(
        {
            qtype
            for payload in mode_payloads.values()
            for qtype in _collect_type_metrics(payload).keys()
        }
    )
    type_answer_rows: List[Dict[str, Any]] = []
    type_latency_rows: List[Dict[str, Any]] = []
    for qtype in all_types:
        answer_row = {"question_type": qtype}
        latency_row = {"question_type": qtype}
        for mode in MODE_ORDER:
            metrics = _collect_type_metrics(mode_payloads.get(mode, {})).get(qtype, {})
            answer_row[mode] = metrics.get("type_answer_acc", None)
            latency_row[mode] = metrics.get("type_latency_sec", None)
        type_answer_rows.append(answer_row)
        type_latency_rows.append(latency_row)

    summary_rows: List[Dict[str, Any]] = []
    for mode in MODE_ORDER:
        summary = dict(mode_payloads[mode].get("run", {}))
        summary_rows.append(
            {
                "mode": mode,
                "run_id": summary.get("run_id", ""),
                "final_answer_acc": float(summary.get("final_answer_acc") or 0.0),
                "avg_latency_sec": float(summary.get("avg_latency_sec") or 0.0),
                "retrieval_answer_span_hit_rate": float(summary.get("retrieval_answer_span_hit_rate") or 0.0),
                "retrieval_support_sentence_hit_rate": float(summary.get("retrieval_support_sentence_hit_rate") or 0.0),
                "graph_answer_span_hit_rate": float(summary.get("graph_answer_span_hit_rate") or 0.0),
                "graph_support_sentence_hit_rate": float(summary.get("graph_support_sentence_hit_rate") or 0.0),
                "graph_ingest_accept_rate": float(summary.get("graph_ingest_accept_rate") or 0.0),
            }
        )

    comparison = {
        "dataset": dataset_name,
        "model": model_name,
        "judge_model": judge_model,
        "primary_mode": "memslm",
        "modes": summary_rows,
        "type_answer_acc": type_answer_rows,
        "type_latency_sec": type_latency_rows,
    }
    if source_audit_summary:
        comparison["source_audit_summary"] = source_audit_summary

    json_path = out_dir / f"{artifact_prefix}_comparison.json"
    md_path = out_dir / f"{artifact_prefix}_comparison.md"
    csv_path = out_dir / f"{artifact_prefix}_comparison.csv"
    json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "run_id",
                "final_answer_acc",
                "avg_latency_sec",
                "retrieval_answer_span_hit_rate",
                "retrieval_support_sentence_hit_rate",
                "graph_answer_span_hit_rate",
                "graph_support_sentence_hit_rate",
                "graph_ingest_accept_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    md_lines = [
        f"# Thesis Comparison Report: {artifact_prefix}",
        "",
        f"- dataset: `{dataset_name}`",
        f"- model: `{model_name}`",
        f"- judge_model: `{judge_model}`",
        "- primary_mode: `memslm`",
        "- note: this report is a single consolidated comparison; no per-mode reports are emitted.",
        "",
    ]
    if source_audit_summary:
        md_lines.extend(list(iter_audit_summary_lines(source_audit_summary)))
    md_lines.extend(
        [
            "## Run Summary",
            "",
            "| mode | run_id | final_answer_acc | avg_latency_sec | retrieval_answer_span_hit_rate | retrieval_support_sentence_hit_rate | graph_answer_span_hit_rate | graph_support_sentence_hit_rate | graph_ingest_accept_rate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        md_lines.append(
            "| {mode} | {run_id} | {final_answer_acc:.4f} | {avg_latency_sec:.4f} | {retrieval_answer_span_hit_rate:.4f} | {retrieval_support_sentence_hit_rate:.4f} | {graph_answer_span_hit_rate:.4f} | {graph_support_sentence_hit_rate:.4f} | {graph_ingest_accept_rate:.4f} |".format(
                **row
            )
        )

    md_lines.extend(
        [
            "",
            "## Type Answer Acc",
            "",
            "| question_type | model-only | naive rag | memslm | ablation | avg |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in type_answer_rows:
        values = [row.get(mode) for mode in MODE_ORDER]
        numeric = [float(v) for v in values if v is not None]
        avg = sum(numeric) / float(len(numeric)) if numeric else 0.0
        md_lines.append(
            "| {question_type} | {model_only} | {naive_rag} | {memslm} | {ablation} | {avg:.4f} |".format(
                question_type=row["question_type"],
                model_only=("" if row.get("model-only") is None else f"{float(row.get('model-only')):.4f}"),
                naive_rag=("" if row.get("naive rag") is None else f"{float(row.get('naive rag')):.4f}"),
                memslm=("" if row.get("memslm") is None else f"{float(row.get('memslm')):.4f}"),
                ablation=("" if row.get("ablation") is None else f"{float(row.get('ablation')):.4f}"),
                avg=avg,
            )
        )
    overall_answer_values = [row["final_answer_acc"] for row in summary_rows]
    md_lines.append(
        "| overall | "
        + " | ".join(f"{float(row['final_answer_acc']):.4f}" for row in summary_rows)
        + f" | {sum(overall_answer_values) / float(len(overall_answer_values)):.4f} |"
    )

    md_lines.extend(
        [
            "",
            "## Type Latency Sec",
            "",
            "| question_type | model-only | naive rag | memslm | ablation | avg |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in type_latency_rows:
        values = [row.get(mode) for mode in MODE_ORDER]
        numeric = [float(v) for v in values if v is not None]
        avg = sum(numeric) / float(len(numeric)) if numeric else 0.0
        md_lines.append(
            "| {question_type} | {model_only} | {naive_rag} | {memslm} | {ablation} | {avg:.4f} |".format(
                question_type=row["question_type"],
                model_only=("" if row.get("model-only") is None else f"{float(row.get('model-only')):.4f}"),
                naive_rag=("" if row.get("naive rag") is None else f"{float(row.get('naive rag')):.4f}"),
                memslm=("" if row.get("memslm") is None else f"{float(row.get('memslm')):.4f}"),
                ablation=("" if row.get("ablation") is None else f"{float(row.get('ablation')):.4f}"),
                avg=avg,
            )
        )
    overall_latency_values = [row["avg_latency_sec"] for row in summary_rows]
    md_lines.append(
        "| overall | "
        + " | ".join(f"{float(row['avg_latency_sec']):.4f}" for row in summary_rows)
        + f" | {sum(overall_latency_values) / float(len(overall_latency_values)):.4f} |"
    )

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "csv": str(csv_path), "comparison": comparison}


def build_consolidated_report(
    *,
    db_path: str,
    output_dir: str,
    dataset_name: str,
    model_name: str,
    judge_model: str,
    judge_enabled: bool,
    mode_run_ids: Dict[str, str | None],
    report_dir: str,
) -> Dict[str, Any]:
    db_file = resolve_project_path(db_path)
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        from llm_long_memory.utils.helpers import load_config

        config = load_config()
        run_table = str(config["evaluation"]["run_table"])
        result_table = str(config["evaluation"]["result_table"])
        eval_store = EvalStore(conn=conn, eval_cfg=dict(config["evaluation"]))
        eval_store.create_tables()
        eval_store.ensure_schema_compat()

        resolved_run_ids: Dict[str, str] = {}
        report_root = resolve_project_path(report_dir)
        for mode in MODE_ORDER:
            explicit = mode_run_ids.get(mode)
            run_id = _resolve_mode_run_id(
                conn=conn,
                eval_store=eval_store,
                mode=mode,
                dataset_name=dataset_name,
                explicit_run_id=explicit,
                report_dir=report_root,
            )
            resolved_run_ids[mode] = run_id

        mode_payloads: Dict[str, Dict[str, Any]] = {}
        for mode in MODE_ORDER:
            run_id = resolved_run_ids[mode]
            mode_payloads[mode] = _load_mode_payload(
                db_path=str(db_file),
                run_id=run_id,
                eval_cfg=config["evaluation"],
                judge_enabled=judge_enabled,
                judge_model=judge_model,
            )

        resolved_dataset_name = dataset_name.strip() if str(dataset_name).strip() else ""
        if not resolved_dataset_name:
            for mode in MODE_ORDER:
                run = mode_payloads.get(mode, {}).get("run", {})
                ds = str(run.get("dataset_path", "")).strip()
                if ds:
                    resolved_dataset_name = Path(ds).name
                    break
        if not resolved_dataset_name:
            resolved_dataset_name = "unknown"
        source_audit_summary = load_latest_source_audit_summary(report_root, resolved_dataset_name)

        artifact_prefix = "__".join(
            [
                sanitize_filename_part(Path(resolved_dataset_name).stem),
                f"model-{sanitize_filename_part(model_name)}",
                f"judge-{sanitize_filename_part(judge_model)}",
                "memslm-centered",
            ]
        )
        return _write_comparison_report(
            output_dir=output_dir,
            artifact_prefix=artifact_prefix,
            dataset_name=resolved_dataset_name,
            model_name=model_name,
            judge_model=judge_model,
            mode_payloads=mode_payloads,
            source_audit_summary=source_audit_summary,
        )
    finally:
        conn.close()
