"""Run thesis-style comparison experiments and export a wide comparison report."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Sequence

from llm_long_memory.baselines.run_baseline import run_one_dataset_with_config
from llm_long_memory.experiments.build_eval_subset import build_subset
from llm_long_memory.experiments.llm_judge import LLMJudge
from llm_long_memory.experiments.run_thesis_eval import _parse_csv, _resolve_dataset_path
from llm_long_memory.evaluation.eval_store import EvalStore
from llm_long_memory.utils.helpers import load_config, resolve_project_path, sanitize_filename_part


MODE_ORDER = ["model-only", "naive rag", "memslm", "ablation"]


def _parse_modes(value: str | None) -> List[str]:
    modes = [item.strip().lower() for item in _parse_csv(value) if item.strip()]
    return modes or list(MODE_ORDER)


def _mode_config(
    base_config: Dict[str, Any],
    *,
    mode: str,
    baseline_config: Dict[str, Any],
) -> Dict[str, Any]:
    config = copy.deepcopy(baseline_config if mode == "ablation" else base_config)
    evaluation_cfg = dict(config.get("evaluation", {}))
    retrieval_cfg = dict(config.get("retrieval", {}))
    long_mem_cfg = dict(config.get("memory", {}).get("long_memory", {}))

    evaluation_cfg["offline_graph_build_enabled"] = bool(mode == "memslm")
    config["evaluation"] = evaluation_cfg

    if mode == "model-only":
        retrieval_cfg["execution_mode"] = "model_only"
        retrieval_cfg["model_only_enabled"] = True
        retrieval_cfg["classic_rag_enabled"] = False
        retrieval_cfg.setdefault("global_chunk_retrieval", {})["enabled"] = False
        retrieval_cfg.setdefault("topic_expansion", {})["enabled"] = False
        retrieval_cfg.setdefault("long_memory_context", {})["enabled"] = False
        long_mem_cfg["enabled"] = False
    elif mode == "naive rag":
        retrieval_cfg["execution_mode"] = "naive_rag"
        retrieval_cfg["model_only_enabled"] = False
        retrieval_cfg["classic_rag_enabled"] = True
        retrieval_cfg.setdefault("global_chunk_retrieval", {})["enabled"] = False
        retrieval_cfg.setdefault("topic_expansion", {})["enabled"] = False
        retrieval_cfg.setdefault("long_memory_context", {})["enabled"] = False
        long_mem_cfg["enabled"] = False
    elif mode == "memslm":
        retrieval_cfg["execution_mode"] = "memslm"
        retrieval_cfg["model_only_enabled"] = False
        retrieval_cfg["classic_rag_enabled"] = False
    elif mode == "ablation":
        retrieval_cfg["execution_mode"] = "memslm"
        retrieval_cfg["model_only_enabled"] = False
        retrieval_cfg["classic_rag_enabled"] = False
        retrieval_cfg.setdefault("global_chunk_retrieval", {})["enabled"] = False
        retrieval_cfg.setdefault("topic_expansion", {})["enabled"] = False
        retrieval_cfg.setdefault("long_memory_context", {})["enabled"] = False
        long_mem_cfg["enabled"] = False
    else:
        raise ValueError(f"Unknown compare mode: {mode}")

    config["retrieval"] = retrieval_cfg
    config.setdefault("memory", {})["long_memory"] = long_mem_cfg
    return config


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
        run_row = conn.execute(
            f"SELECT * FROM {run_table} WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if run_row is None:
            raise ValueError(f"Run not found in eval db: {run_id}")
        type_rows = conn.execute(
            f"SELECT * FROM {type_table} WHERE run_id=? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        enriched_rows: List[Dict[str, Any]] = [{str(k): row[k] for k in row.keys()} for row in type_rows]
        if judge_enabled and enriched_rows:
            judge = LLMJudge(model_name=judge_model or str(eval_cfg.get("judge_model", "")) or "qwen3:8b")
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
        "## Run Summary",
        "",
        "| mode | run_id | final_answer_acc | avg_latency_sec | retrieval_answer_span_hit_rate | retrieval_support_sentence_hit_rate | graph_answer_span_hit_rate | graph_support_sentence_hit_rate | graph_ingest_accept_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thesis comparison experiments.")
    parser.add_argument("--config", default="llm_long_memory/config/config.yaml", help="Main config path.")
    parser.add_argument(
        "--baseline-config",
        default="llm_long_memory/baselines/baseline_midrag_v1.yaml",
        help="Frozen baseline config path.",
    )
    parser.add_argument("--dataset", default="", help="Explicit dataset path.")
    parser.add_argument("--split", default="", help="Configured dataset split name.")
    parser.add_argument("--max-total", type=int, default=0, help="Cap total instances.")
    parser.add_argument("--per-type", type=int, default=0, help="Cap instances per question type.")
    parser.add_argument("--seed", type=int, default=42, help="Subset sampling seed.")
    parser.add_argument("--keep-types", default="", help="Comma-separated question types to keep.")
    parser.add_argument("--drop-types", default="", help="Comma-separated question types to drop.")
    parser.add_argument("--model", default="", help="Main model override.")
    parser.add_argument("--judge-model", default="", help="Judge model override.")
    parser.add_argument("--judge", action="store_true", help="Use local LLM judge.")
    parser.add_argument(
        "--report-dir",
        default="",
        help="Directory for the single comparison report.",
    )
    parser.add_argument(
        "--modes",
        default="model-only,naive rag,memslm,ablation",
        help="Comma-separated experiment modes in report order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    baseline_config = load_config(args.baseline_config)
    default_model = str(base_config["llm"]["default_model"])
    default_judge = str(base_config["llm"]["default_model"])
    source_dataset = _resolve_dataset_path(base_config, args.dataset.strip() or None, args.split.strip() or None)
    report_dir = args.report_dir.strip() or str(
        base_config["evaluation"].get("thesis_report_dir", "data/processed/thesis_reports_debug_analysis")
    )

    keep_types = _parse_csv(args.keep_types)
    drop_types = _parse_csv(args.drop_types)
    needs_subset = int(args.max_total) > 0 or int(args.per_type) > 0 or bool(keep_types) or bool(drop_types)
    subset_path = source_dataset
    if needs_subset:
        subset_dir = resolve_project_path(
            str(base_config["evaluation"].get("thesis_subset_dir", "data/raw/LongMemEval/thesis_subsets"))
        )
        subset_dir.mkdir(parents=True, exist_ok=True)
        source_stem = sanitize_filename_part(Path(source_dataset).stem)
        subset_name = (
            f"{source_stem}__max{int(args.max_total)}__per{int(args.per_type)}"
            f"__seed{int(args.seed)}.json"
        )
        subset_path = str(subset_dir / subset_name)
        build_subset(
            source_path=source_dataset,
            output_path=subset_path,
            max_total=int(args.max_total),
            per_type=int(args.per_type),
            seed=int(args.seed),
            keep_types=keep_types,
            drop_types=drop_types,
        )
        print(f"subset_dataset: {subset_path}")
    else:
        print(f"subset_dataset: {source_dataset} (no extra subset built)")

    modes = [mode.strip().lower() for mode in _parse_csv(args.modes)]
    if not modes:
        modes = list(MODE_ORDER)
    ordered_modes = [mode for mode in MODE_ORDER if mode in modes]
    unknown_modes = [mode for mode in modes if mode not in MODE_ORDER]
    if unknown_modes:
        raise ValueError(f"Unknown modes: {', '.join(unknown_modes)}")

    model_override = args.model.strip() or default_model
    judge_override = args.judge_model.strip() or default_judge
    artifact_seed = "__".join(
        [
            sanitize_filename_part(Path(source_dataset).stem),
            f"model-{sanitize_filename_part(model_override)}",
            f"judge-{sanitize_filename_part(judge_override)}",
        ]
    )

    mode_payloads: Dict[str, Dict[str, Any]] = {}
    for mode in ordered_modes:
        mode_config = _mode_config(base_config, mode=mode, baseline_config=baseline_config)
        run_id = run_one_dataset_with_config(
            config=mode_config,
            dataset_path=subset_path,
            sample_limit=int(args.max_total) if int(args.max_total) > 0 else 0,
            model_name=model_override,
            resume_run_id=None,
        )
        mode_payloads[mode] = _load_mode_payload(
            db_path=str(mode_config["evaluation"]["database_file"]),
            run_id=run_id,
            eval_cfg=mode_config["evaluation"],
            judge_enabled=bool(args.judge),
            judge_model=judge_override,
        )

    compare_prefix = "__".join([artifact_seed, "compare"])
    compare_report = _write_comparison_report(
        output_dir=report_dir,
        artifact_prefix=compare_prefix,
        dataset_name=Path(source_dataset).name,
        model_name=model_override,
        judge_model=judge_override,
        mode_payloads=mode_payloads,
    )

    print("Comparison report:")
    print(f"- json: {compare_report['json']}")
    print(f"- markdown: {compare_report['markdown']}")
    print(f"- csv: {compare_report['csv']}")


if __name__ == "__main__":
    main()
