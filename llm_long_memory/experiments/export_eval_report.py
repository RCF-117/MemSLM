"""Export a thesis-ready evaluation report from the eval SQLite database."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from llm_long_memory.experiments.llm_judge import LLMJudge
from llm_long_memory.utils.helpers import load_config, resolve_project_path


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
            vals = [_safe_float(row[col]) for row in bucket if _safe_float(row[col]) is not None]
            if not vals:
                return None
            return sum(vals) / float(len(vals))

        grouped.append(
            {
                "question_type": key,
                "total": total,
                "matched": matched,
                "type_answer_acc": ((judge_matched / total) if (judge_enabled and total) else None),
                "answer_span_hit_rate": avg("answer_span_hit"),
                "support_sentence_hit_rate": avg("support_sentence_hit"),
                "graph_answer_span_hit_rate": avg("graph_answer_span_hit"),
                "graph_support_sentence_hit_rate": avg("graph_support_sentence_hit"),
                "type_latency_sec": avg("latency_sec"),
            }
        )
    return grouped


def export_report(
    *,
    db_path: str,
    output_dir: str,
    run_id: str | None = None,
    graph_json_path: str | None = None,
    judge_enabled: bool = False,
    judge_model: str | None = None,
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

        enriched_rows: List[Dict[str, Any]] = [_row_to_dict(row) for row in result_rows]
        if judge_enabled and enriched_rows:
            judge = LLMJudge(model_name=judge_model or str(config["llm"]["default_model"]))
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
        summary.update(
            {
                "run_id": resolved_run_id,
                "total_results": total,
                "matched_results": matched,
                "exact_match_acc": (matched / total) if total else 0.0,
                "final_answer_acc": ((judge_matched / total) if (judge_enabled and total) else ((matched / total) if total else 0.0)),
                "judge_matched_results": judge_matched,
                "judge_enabled": bool(judge_enabled),
                "avg_latency_sec": _safe_float(summary.get("avg_latency_sec")),
            }
        )

        grouped = _group_metrics(enriched_rows, judge_enabled=judge_enabled)
        graph_payload: Dict[str, Any] = {}
        if graph_json_path:
            graph_file = resolve_project_path(graph_json_path)
            if graph_file.exists():
                graph_payload = json.loads(graph_file.read_text(encoding="utf-8"))

        payload = {
            "run": summary,
            "question_type_metrics": grouped,
            "rows": enriched_rows,
        }
        if graph_payload:
            payload["graph_eval"] = graph_payload

        out_dir = resolve_project_path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{resolved_run_id}_report.json"
        md_path = out_dir / f"{resolved_run_id}_report.md"
        csv_path = out_dir / f"{resolved_run_id}_rows.csv"

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(payload["rows"][0].keys()) if payload["rows"] else [])
            if payload["rows"]:
                writer.writeheader()
                writer.writerows(payload["rows"])

        md_lines = [
            f"# Eval Report: {resolved_run_id}",
            "",
            f"- dataset: `{summary.get('dataset_path', '')}`",
            f"- total: `{summary.get('total', 0)}`",
            f"- final_answer_acc: `{summary.get('final_answer_acc', 0.0):.4f}`",
            f"- judge_enabled: `{bool(summary.get('judge_enabled', False))}`",
            f"- retrieval_answer_span_hit_rate: `{_safe_float(summary.get('retrieval_answer_span_hit_rate')) or 0.0:.4f}`",
            f"- retrieval_support_sentence_hit_rate: `{_safe_float(summary.get('retrieval_support_sentence_hit_rate')) or 0.0:.4f}`",
            f"- graph_answer_span_hit_rate: `{_safe_float(summary.get('graph_answer_span_hit_rate')) or 0.0:.4f}`",
            f"- graph_support_sentence_hit_rate: `{_safe_float(summary.get('graph_support_sentence_hit_rate')) or 0.0:.4f}`",
            f"- graph_ingest_accept_rate: `{_safe_float(summary.get('graph_ingest_accept_rate')) or 0.0:.4f}`",
            f"- avg_latency_sec: `{_safe_float(summary.get('avg_latency_sec')) or 0.0:.4f}`",
        ]
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
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

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
        default="data/processed/mid_memory.db",
        help="Evaluation SQLite database path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/thesis_reports",
        help="Directory where report artifacts are written.",
    )
    parser.add_argument("--run-id", default="", help="Specific run id. Defaults to latest run.")
    parser.add_argument(
        "--graph-json",
        default="",
        help="Optional offline graph eval JSON to merge into the report.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Use a local LLM judge for the final answer accuracy.",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="Optional judge model override. Defaults to config.llm.default_model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_report(
        db_path=args.db_path,
        output_dir=args.output_dir,
        run_id=(args.run_id.strip() or None),
        graph_json_path=(args.graph_json.strip() or None),
        judge_enabled=bool(args.judge),
        judge_model=(args.judge_model.strip() or None),
    )


if __name__ == "__main__":
    main()
