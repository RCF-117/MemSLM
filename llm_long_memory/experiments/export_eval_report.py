"""Export a thesis-ready evaluation report from the eval SQLite database."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from llm_long_memory.evaluation.eval_store import EvalStore
from llm_long_memory.experiments.report_audit_utils import (
    iter_audit_summary_lines,
    load_latest_source_audit_summary,
)
from llm_long_memory.experiments.local_llm_judge import LocalLLMJudge
from llm_long_memory.utils.helpers import (
    dataset_display_name,
    load_config,
    resolve_project_path,
    sanitize_filename_part,
)


def _row_to_dict(row: sqlite3.Row | None) -> Dict[str, Any]:
    if row is None:
        return {}
    return {str(k): row[k] for k in row.keys()}


def _latest_run(conn: sqlite3.Connection, run_table: str) -> str:
    row = conn.execute(
        f"""
        SELECT run_id
        FROM {run_table}
        ORDER BY datetime(started_at) DESC, run_id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise ValueError(f"No runs found in table {run_table}.")
    return str(row["run_id"])


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_metrics(rows: Sequence[sqlite3.Row], *, judge_enabled: bool = False) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        key = str(row["question_type"] or "unknown").strip() or "unknown"
        buckets[key].append(row)

    grouped: List[Dict[str, Any]] = []
    for key in sorted(buckets.keys()):
        bucket = buckets[key]
        total = len(bucket)
        matched = sum(1 for row in bucket if int(row["is_match"] or 0) == 1)
        judge_matched = sum(1 for row in bucket if int(row["judge_is_correct"] or 0) == 1) if judge_enabled else None

        def avg(col: str) -> float | None:
            vals = [_safe_float(row.get(col)) for row in bucket if _safe_float(row.get(col)) is not None]
            if not vals:
                return None
            return sum(vals) / float(len(vals))

        grouped.append(
            {
                "question_type": key,
                "total": total,
                "matched": matched,
                "type_answer_acc": (
                    (judge_matched / total) if (judge_enabled and total) else ((matched / total) if total else None)
                ),
                "answer_span_hit_rate": avg("answer_span_hit"),
                "support_sentence_hit_rate": avg("support_sentence_hit"),
                "graph_answer_span_hit_rate": avg("graph_answer_span_hit"),
                "graph_support_sentence_hit_rate": avg("graph_support_sentence_hit"),
                "type_answer_token_density": avg("answer_token_density"),
                "type_noise_density": avg("noise_density"),
                "type_latency_sec": avg("latency_sec"),
            }
        )
    return grouped


def export_report(
    *,
    db_path: str,
    output_dir: str,
    run_id: str | None = None,
    dataset_name: str | None = None,
    model_name: str | None = None,
    judge_model: str | None = None,
    artifact_prefix: str | None = None,
    graph_json_path: str | None = None,
    node_graph_json_path: str | None = None,
    judge_enabled: bool = False,
) -> Dict[str, Any]:
    db_file = resolve_project_path(db_path)
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        config = load_config()
        run_table = str(config["evaluation"]["run_table"])
        result_table = str(config["evaluation"]["result_table"])
        resolved_run_id = run_id or _latest_run(conn, run_table)
        run_row = conn.execute(
            f"SELECT * FROM {run_table} WHERE run_id=?",
            (resolved_run_id,),
        ).fetchone()
        if run_row is None:
            raise ValueError(f"Run not found: {resolved_run_id}")
        result_rows = conn.execute(
            f"SELECT * FROM {result_table} WHERE run_id=? ORDER BY id ASC",
            (resolved_run_id,),
        ).fetchall()
        eval_store = EvalStore(conn=conn, eval_cfg=dict(config["evaluation"]))
        eval_store.create_tables()
        eval_store.ensure_schema_compat()

        enriched_rows: List[Dict[str, Any]] = [_row_to_dict(row) for row in result_rows]
        if judge_enabled and enriched_rows:
            judge = LocalLLMJudge(model_name=judge_model or str(config["llm"]["default_model"]))
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

        total = len(enriched_rows)
        matched = sum(1 for row in enriched_rows if int(row["is_match"] or 0) == 1)
        judge_matched = sum(1 for row in enriched_rows if int(row.get("judge_is_correct") or 0) == 1) if judge_enabled else None
        summary = _row_to_dict(run_row)
        resolved_dataset_name = (
            str(dataset_name).strip()
            if dataset_name and str(dataset_name).strip()
            else Path(str(summary.get("dataset_path", ""))).name
        )
        resolved_dataset_display_name = dataset_display_name(resolved_dataset_name)
        resolved_model_name = str(model_name).strip() if model_name and str(model_name).strip() else ""
        resolved_judge_model = str(judge_model).strip() if judge_model and str(judge_model).strip() else ""
        summary["dataset_path"] = resolved_dataset_name
        summary["dataset_display_name"] = resolved_dataset_display_name
        summary.update(
            {
                "run_id": resolved_run_id,
                "dataset_name": resolved_dataset_display_name,
                "model_name": resolved_model_name,
                "judge_model": resolved_judge_model,
                "total_results": total,
                "matched_results": matched,
                "exact_match_acc": (matched / total) if total else 0.0,
                "final_answer_acc": ((judge_matched / total) if (judge_enabled and total) else ((matched / total) if total else 0.0)),
                "judge_matched_results": judge_matched,
                "avg_latency_sec": _safe_float(summary.get("avg_latency_sec")),
            }
        )

        grouped = _group_metrics(enriched_rows, judge_enabled=judge_enabled)
        thesis_final_answer_acc = ((judge_matched / total) if (judge_enabled and total) else ((matched / total) if total else 0.0))
        if grouped:
            for row in grouped:
                eval_store.log_thesis_type_metric(
                    resolved_run_id,
                    str(row["question_type"]),
                    type_answer_acc=float(row.get("type_answer_acc") or 0.0),
                    type_answer_token_density=_safe_float(row.get("type_answer_token_density")),
                    type_noise_density=_safe_float(row.get("type_noise_density")),
                    type_latency_sec=float(row.get("type_latency_sec") or 0.0),
                    commit=False,
                )
        eval_store.log_thesis_run_metrics(
            resolved_run_id,
            final_answer_acc=float(thesis_final_answer_acc),
            retrieval_answer_span_hit_rate=_safe_float(summary.get("retrieval_answer_span_hit_rate")),
            retrieval_support_sentence_hit_rate=_safe_float(summary.get("retrieval_support_sentence_hit_rate")),
            graph_answer_span_hit_rate=_safe_float(summary.get("graph_answer_span_hit_rate")),
            graph_support_sentence_hit_rate=_safe_float(summary.get("graph_support_sentence_hit_rate")),
            graph_ingest_accept_rate=_safe_float(summary.get("graph_ingest_accept_rate")),
            avg_answer_token_density=_safe_float(summary.get("avg_answer_token_density")),
            avg_noise_density=_safe_float(summary.get("avg_noise_density")),
            avg_latency_sec=_safe_float(summary.get("avg_latency_sec")),
            commit=False,
        )
        conn.commit()
        graph_payload: Dict[str, Any] = {}
        if graph_json_path:
            graph_file = resolve_project_path(graph_json_path)
            if graph_file.exists():
                graph_payload = json.loads(graph_file.read_text(encoding="utf-8"))
        node_graph_payload: Dict[str, Any] = {}
        if node_graph_json_path:
            node_graph_file = resolve_project_path(node_graph_json_path)
            if node_graph_file.exists():
                node_graph_payload = json.loads(node_graph_file.read_text(encoding="utf-8"))

        out_dir = resolve_project_path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        source_audit_summary = load_latest_source_audit_summary(out_dir, resolved_dataset_display_name)
        payload = {
            "run": summary,
            "question_type_metrics": grouped,
            "rows": enriched_rows,
        }
        if source_audit_summary:
            payload["source_audit_summary"] = source_audit_summary
        if graph_payload:
            payload["graph_eval"] = graph_payload
        if node_graph_payload:
            payload["node_graph_eval"] = node_graph_payload
        file_prefix = artifact_prefix or "__".join(
            [
                sanitize_filename_part(resolved_run_id),
                sanitize_filename_part(Path(resolved_dataset_name).stem),
                f"model-{sanitize_filename_part(resolved_model_name)}" if resolved_model_name else "model-unknown",
                f"judge-{sanitize_filename_part(resolved_judge_model)}" if resolved_judge_model else "judge-unknown",
            ]
        )
        json_path = out_dir / f"{file_prefix}_report.json"
        md_path = out_dir / f"{file_prefix}_report.md"
        csv_path = out_dir / f"{file_prefix}_rows.csv"

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(payload["rows"][0].keys()) if payload["rows"] else [])
            if payload["rows"]:
                writer.writeheader()
                writer.writerows(payload["rows"])

        md_lines = [
            f"# Eval Report: {resolved_run_id}",
            "",
            f"- dataset: `{resolved_dataset_display_name}`",
            f"- eval_db: `{Path(str(db_file)).name}`",
            f"- model: `{resolved_model_name or 'unknown'}`",
            f"- judge_model: `{resolved_judge_model or 'unknown'}`",
            f"- total: `{summary.get('total', 0)}`",
            f"- final_answer_acc: `{summary.get('final_answer_acc', 0.0):.4f}`",
            f"- retrieval_answer_span_hit_rate: `{_safe_float(summary.get('retrieval_answer_span_hit_rate')) or 0.0:.4f}`",
            f"- retrieval_support_sentence_hit_rate: `{_safe_float(summary.get('retrieval_support_sentence_hit_rate')) or 0.0:.4f}`",
            f"- graph_answer_span_hit_rate: `{_safe_float(summary.get('graph_answer_span_hit_rate')) or 0.0:.4f}`",
            f"- graph_support_sentence_hit_rate: `{_safe_float(summary.get('graph_support_sentence_hit_rate')) or 0.0:.4f}`",
            f"- graph_ingest_accept_rate: `{_safe_float(summary.get('graph_ingest_accept_rate')) or 0.0:.4f}`",
            f"- avg_answer_token_density: `{_safe_float(summary.get('avg_answer_token_density')) or 0.0:.4f}`",
            f"- avg_noise_density: `{_safe_float(summary.get('avg_noise_density')) or 0.0:.4f}`",
            f"- avg_latency_sec: `{_safe_float(summary.get('avg_latency_sec')) or 0.0:.4f}`",
        ]
        if source_audit_summary:
            md_lines.extend([""] + list(iter_audit_summary_lines(source_audit_summary)))
        if judge_enabled:
            md_lines.extend(
                [
                    "",
                    "## Type Answer Acc",
                    "",
                    "| question_type | total | type_answer_acc |",
                    "| --- | ---: | ---: |",
                ]
            )
            for row in grouped:
                type_answer_acc = row.get("type_answer_acc")
                if type_answer_acc is None:
                    continue
                md_lines.append(
                    "| {question_type} | {total} | {type_answer_acc:.4f} |".format(
                        question_type=row["question_type"],
                        total=row["total"],
                        type_answer_acc=float(type_answer_acc or 0.0),
                    )
                )
        md_lines.extend(
            [
                "",
                "## Type Prompt Density",
                "",
                "| question_type | total | type_answer_token_density | type_noise_density |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in grouped:
            md_lines.append(
                "| {question_type} | {total} | {type_answer_token_density:.4f} | {type_noise_density:.4f} |".format(
                    question_type=row["question_type"],
                    total=row["total"],
                    type_answer_token_density=float(row.get("type_answer_token_density") or 0.0),
                    type_noise_density=float(row.get("type_noise_density") or 0.0),
                )
            )
        md_lines.extend(
            [
                "",
                "## Type Latency Sec",
                "",
                "| question_type | total | type_latency_sec |",
                "| --- | ---: | ---: |",
            ]
        )
        for row in grouped:
            md_lines.append(
                "| {question_type} | {total} | {type_latency_sec:.4f} |".format(
                    question_type=row["question_type"],
                    total=row["total"],
                    type_latency_sec=float(row.get("type_latency_sec") or 0.0),
                )
            )
        if graph_payload:
            graph_summary = graph_payload.get("summary", {})
            md_lines.extend(
                [
                    "",
                    "## Graph Eval Summary",
                    "",
                    f"- total: `{graph_summary.get('total', 0)}`",
                    f"- mid_answer_span_hit_rate: `{_safe_float(graph_summary.get('mid_answer_span_hit_rate')) or 0.0:.4f}`",
                    f"- mid_support_sentence_hit_rate: `{_safe_float(graph_summary.get('mid_support_sentence_hit_rate')) or 0.0:.4f}`",
                    f"- graph_answer_span_hit_rate: `{_safe_float(graph_summary.get('graph_answer_span_hit_rate')) or 0.0:.4f}`",
                    f"- graph_support_sentence_hit_rate: `{_safe_float(graph_summary.get('graph_support_sentence_hit_rate')) or 0.0:.4f}`",
                    f"- graph_non_empty_ratio: `{_safe_float(graph_summary.get('graph_non_empty_ratio')) or 0.0:.4f}`",
                    f"- graph_accept_nonzero_ratio: `{_safe_float(graph_summary.get('graph_accept_nonzero_ratio')) or 0.0:.4f}`",
                    f"- avg_accepted_events: `{_safe_float(graph_summary.get('avg_accepted_events')) or 0.0:.4f}`",
                ]
            )
        if node_graph_payload:
            node_graph_summary = node_graph_payload.get("summary", {})
            md_lines.extend(
                [
                    "",
                    "## Node Graph Summary",
                    "",
                    f"- event_nodes: `{node_graph_summary.get('event_nodes', 0)}`",
                    f"- event_edges: `{node_graph_summary.get('event_edges', 0)}`",
                    f"- node_nodes: `{node_graph_summary.get('node_nodes', 0)}`",
                    f"- node_edges: `{node_graph_summary.get('node_edges', 0)}`",
                ]
            )
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        print("Thesis eval summary:")
        print(f"- final_answer_acc: {summary.get('final_answer_acc', 0.0):.4f}")
        if judge_enabled:
            print("- type_answer_acc:")
            for row in grouped:
                type_answer_acc = row.get("type_answer_acc")
                if type_answer_acc is None:
                    continue
                print(
                    f"  - {row['question_type']}: {float(type_answer_acc or 0.0):.4f}"
                )
        print(
            "- retrieval_answer_span_hit_rate: "
            f"{_safe_float(summary.get('retrieval_answer_span_hit_rate')) or 0.0:.4f}"
        )
        print(
            "- retrieval_support_sentence_hit_rate: "
            f"{_safe_float(summary.get('retrieval_support_sentence_hit_rate')) or 0.0:.4f}"
        )
        print(
            "- graph_answer_span_hit_rate: "
            f"{_safe_float(summary.get('graph_answer_span_hit_rate')) or 0.0:.4f}"
        )
        print(
            "- graph_support_sentence_hit_rate: "
            f"{_safe_float(summary.get('graph_support_sentence_hit_rate')) or 0.0:.4f}"
        )
        print(
            "- graph_ingest_accept_rate: "
            f"{_safe_float(summary.get('graph_ingest_accept_rate')) or 0.0:.4f}"
        )
        print(f"- avg_latency_sec: {_safe_float(summary.get('avg_latency_sec')) or 0.0:.4f}")
        if source_audit_summary:
            print("- answer_source_audit_summary:")
            for line in iter_audit_summary_lines(source_audit_summary):
                clean = str(line).strip()
                if not clean or clean.startswith("## "):
                    continue
                print(f"  {clean}")
        print("- type_latency_sec:")
        for row in grouped:
            print(
                f"  - {row['question_type']}: {float(row.get('type_latency_sec') or 0.0):.4f}"
            )
        if node_graph_payload:
            node_graph_summary = node_graph_payload.get("summary", {})
            print("- node_graph_summary:")
            print(
                f"  - event_nodes: {node_graph_summary.get('event_nodes', 0)} | "
                f"event_edges: {node_graph_summary.get('event_edges', 0)} | "
                f"node_nodes: {node_graph_summary.get('node_nodes', 0)} | "
                f"node_edges: {node_graph_summary.get('node_edges', 0)}"
            )
        print(f"run_id: {resolved_run_id}")
        print(f"json: {json_path}")
        print(f"markdown: {md_path}")
        print(f"csv: {csv_path}")
        return payload
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a thesis-ready eval report.")
    parser.add_argument(
        "--db-path",
        default="",
        help="Evaluation SQLite database path.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory where report artifacts are written.",
    )
    parser.add_argument("--run-id", default="", help="Specific run id. Defaults to latest run.")
    parser.add_argument("--dataset-name", default="", help="Dataset display name for the report.")
    parser.add_argument("--model-name", default="", help="Main model display name for the report.")
    parser.add_argument("--judge-model", default="", help="Judge model display name for the report.")
    parser.add_argument(
        "--artifact-prefix",
        default="",
        help="Optional filename prefix for the exported report artifacts.",
    )
    parser.add_argument(
        "--graph-json",
        default="",
        help="Optional offline graph eval JSON to merge into the report.",
    )
    parser.add_argument(
        "--node-graph-json",
        default="",
        help="Optional node graph JSON to merge into the report.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Use a local LLM judge for the final answer accuracy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    db_path = args.db_path.strip() or str(config["evaluation"].get("database_file", "data/processed/thesis_eval.db"))
    output_dir = args.output_dir.strip() or str(
        config["evaluation"].get("thesis_report_dir", "data/processed/thesis_reports_debug_analysis")
    )
    export_report(
        db_path=db_path,
        output_dir=output_dir,
        run_id=(args.run_id.strip() or None),
        dataset_name=(args.dataset_name.strip() or None),
        model_name=(args.model_name.strip() or None),
        judge_model=(args.judge_model.strip() or None),
        artifact_prefix=(args.artifact_prefix.strip() or None),
        graph_json_path=(args.graph_json.strip() or None),
        node_graph_json_path=(args.node_graph_json.strip() or None),
        judge_enabled=bool(args.judge),
    )


if __name__ == "__main__":
    main()
