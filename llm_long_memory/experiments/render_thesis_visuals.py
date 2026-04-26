"""Render standalone thesis visuals from audit and comparison artifacts.

This script is intentionally separate from evaluation and report generation.
It converts already-saved JSON artifacts into reusable paper-facing figures and
CSV/Markdown tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llm_long_memory.utils.helpers import resolve_project_path


STAGE_ORDER = ["rag", "filter", "claims", "light_graph", "toolkit", "final_prompt"]
MODE_ORDER = ["model-only", "naive rag", "memslm", "ablation"]
STAGE_LABELS = {
    "rag": "RAG",
    "filter": "Filter",
    "claims": "Claims",
    "light_graph": "Light Graph",
    "toolkit": "Toolkit",
    "final_prompt": "Final Prompt",
}
MODE_LABELS = {
    "model-only": "Model-Only",
    "naive rag": "Naive RAG",
    "memslm": "MemSLM",
    "ablation": "Filter-Only Ablation",
}
PALETTE = [
    "#4C6272",
    "#5B8E7D",
    "#A06C5B",
    "#7D6C91",
    "#B08AA4",
    "#C79D4F",
    "#607D8B",
    "#8D99AE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render thesis figures from existing JSON artifacts.")
    parser.add_argument("--audit-json", default="", help="Path to answer_source_audit JSON.")
    parser.add_argument("--comparison-json", default="", help="Path to *_comparison.json.")
    parser.add_argument(
        "--output-dir",
        default="llm_long_memory/data/processed/thesis_visuals",
        help="Output directory for figures and tables.",
    )
    parser.add_argument("--prefix", default="", help="Optional artifact prefix override.")
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: str | Path) -> Any:
    p = resolve_project_path(str(path))
    return json.loads(p.read_text(encoding="utf-8"))


def _ensure_dir(path: str | Path) -> Path:
    p = resolve_project_path(str(path))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sanitize(name: str) -> str:
    return "_".join(str(name or "artifact").strip().replace("/", "_").split())


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md_table(path: Path, title: str, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_multi_line(
    *,
    x_labels: Sequence[str],
    series: Dict[str, Sequence[float]],
    title: str,
    ylabel: str,
    output_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for idx, (label, values) in enumerate(series.items()):
        ax.plot(
            range(len(x_labels)),
            list(values),
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            color=PALETTE[idx % len(PALETTE)],
            label=label,
        )
    ax.set_title(title, fontsize=13, fontweight="semibold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(list(x_labels), fontsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _plot_bar(
    *,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    output_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    ax.bar(range(len(labels)), list(values), color=colors, width=0.65)
    ax.set_title(title, fontsize=13, fontweight="semibold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(list(labels), fontsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(
    *,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    values: Sequence[Sequence[float]],
    title: str,
    output_path: Path,
    cmap: str = "Blues",
    value_format: str = ".3f",
) -> None:
    plt.close("all")
    width = max(8.5, 1.6 * len(x_labels) + 2.4)
    height = max(4.8, 0.5 * len(y_labels) + 2.4)
    fig, ax = plt.subplots(figsize=(width, height))
    mat = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=13, fontweight="semibold")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(list(x_labels), fontsize=10)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(list(y_labels), fontsize=10)
    for i, row in enumerate(values):
        for j, val in enumerate(row):
            ax.text(j, i, format(float(val), value_format), ha="center", va="center", fontsize=8, color="#1f2933")
    cbar = fig.colorbar(mat, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _aggregate_audit_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    stage_names = [stage for stage in STAGE_ORDER if any(stage in dict(r.get("stage_metrics", {}) or {}) for r in rows)]
    type_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        qtype = str(row.get("question_type", "") or "unknown").strip() or "unknown"
        type_buckets[qtype].append(row)

    answerability_rows: List[Dict[str, Any]] = []
    noise_rows: List[Dict[str, Any]] = []
    latency_rows: List[Dict[str, Any]] = []
    overall_answerability: Dict[str, Any] = {"question_type": "overall", "n": len(rows)}
    overall_noise: Dict[str, Any] = {"question_type": "overall", "n": len(rows)}
    overall_latency: Dict[str, Any] = {"question_type": "overall", "n": len(rows)}

    def _stage_rate(bucket: Sequence[Dict[str, Any]], stage: str) -> float:
        if not bucket:
            return 0.0
        count = 0
        for row in bucket:
            metrics = dict(dict(row.get("stage_metrics", {}) or {}).get(stage, {}) or {})
            count += int(metrics.get("coverage_rec50", 0) or 0)
        return float(count) / float(len(bucket))

    def _stage_latency(bucket: Sequence[Dict[str, Any]], stage: str) -> float:
        if not bucket:
            return 0.0
        vals = [
            _safe_float(dict(row.get("stage_latency_sec", {}) or {}).get(stage, 0.0), 0.0)
            for row in bucket
        ]
        return sum(vals) / float(len(vals)) if vals else 0.0

    def _stage_noise(bucket: Sequence[Dict[str, Any]], stage: str) -> float:
        if not bucket:
            return 0.0
        vals = [
            _safe_float(
                dict(dict(row.get("stage_metrics", {}) or {}).get(stage, {}) or {}).get("noise_density", None),
                0.0,
            )
            for row in bucket
        ]
        return sum(vals) / float(len(vals)) if vals else 0.0

    for stage in stage_names:
        overall_answerability[stage] = _stage_rate(rows, stage)
        overall_noise[stage] = _stage_noise(rows, stage)
    for stage in [s for s in ["rag", "filter", "claims", "light_graph", "toolkit", "composer", "total"] if any(s in dict(r.get("stage_latency_sec", {}) or {}) for r in rows)]:
        overall_latency[stage] = _stage_latency(rows, stage)

    for qtype in sorted(type_buckets.keys()):
        bucket = type_buckets[qtype]
        arow: Dict[str, Any] = {"question_type": qtype, "n": len(bucket)}
        for stage in stage_names:
            arow[stage] = _stage_rate(bucket, stage)
        answerability_rows.append(arow)

        nrow: Dict[str, Any] = {"question_type": qtype, "n": len(bucket)}
        for stage in stage_names:
            nrow[stage] = _stage_noise(bucket, stage)
        noise_rows.append(nrow)

        lrow: Dict[str, Any] = {"question_type": qtype, "n": len(bucket)}
        for stage in [s for s in ["rag", "filter", "claims", "light_graph", "toolkit", "composer", "total"] if any(s in dict(r.get("stage_latency_sec", {}) or {}) for r in rows)]:
            lrow[stage] = _stage_latency(bucket, stage)
        latency_rows.append(lrow)

    return {
        "stage_names": stage_names,
        "answerability_rows": answerability_rows,
        "answerability_overall": overall_answerability,
        "noise_rows": noise_rows,
        "noise_overall": overall_noise,
        "latency_rows": latency_rows,
        "latency_overall": overall_latency,
    }


def render_audit_visuals(*, audit_json: str | Path, output_dir: str | Path, prefix: str = "") -> Dict[str, str]:
    rows = _load_json(audit_json)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Audit JSON must be a non-empty row list.")
    out_dir = _ensure_dir(output_dir)
    base = prefix or Path(str(audit_json)).stem
    base = _sanitize(base)
    agg = _aggregate_audit_rows(rows)

    answerability_rows = list(agg["answerability_rows"])
    noise_rows = list(agg["noise_rows"])
    latency_rows = list(agg["latency_rows"])
    stage_names = list(agg["stage_names"])

    answer_csv = out_dir / f"{base}__stage_answerability_by_type.csv"
    answer_md = out_dir / f"{base}__stage_answerability_by_type.md"
    noise_csv = out_dir / f"{base}__stage_noise_density_by_type.csv"
    noise_md = out_dir / f"{base}__stage_noise_density_by_type.md"
    latency_csv = out_dir / f"{base}__stage_latency_by_type.csv"
    latency_md = out_dir / f"{base}__stage_latency_by_type.md"
    answer_fields = ["question_type", "n"] + stage_names
    noise_fields = ["question_type", "n"] + stage_names
    latency_fields = ["question_type", "n"] + [k for k in ["rag", "filter", "claims", "light_graph", "toolkit", "composer", "total"] if latency_rows and k in latency_rows[0]]

    _write_csv(answer_csv, answerability_rows + [agg["answerability_overall"]], answer_fields)
    _write_md_table(answer_md, "Stage Answerability By Type", answerability_rows + [agg["answerability_overall"]], answer_fields)
    _write_csv(noise_csv, noise_rows + [agg["noise_overall"]], noise_fields)
    _write_md_table(noise_md, "Stage Noise Density By Type", noise_rows + [agg["noise_overall"]], noise_fields)
    _write_csv(latency_csv, latency_rows + [agg["latency_overall"]], latency_fields)
    _write_md_table(latency_md, "Stage Latency By Type", latency_rows + [agg["latency_overall"]], latency_fields)

    answer_series = {
        row["question_type"]: [float(row.get(stage, 0.0) or 0.0) for stage in stage_names]
        for row in answerability_rows
    }
    answer_svg = out_dir / f"{base}__stage_answerability_by_type.svg"
    _plot_multi_line(
        x_labels=[STAGE_LABELS.get(x, x) for x in stage_names],
        series=answer_series,
        title="Stage-Wise Answerability by Type",
        ylabel="Coverage@rec50",
        output_path=answer_svg,
        ylim=(0.0, 1.02),
    )

    noise_series = {
        row["question_type"]: [float(row.get(stage, 0.0) or 0.0) for stage in stage_names]
        for row in noise_rows
    }
    noise_svg = out_dir / f"{base}__stage_noise_density_by_type.svg"
    _plot_multi_line(
        x_labels=[STAGE_LABELS.get(x, x) for x in stage_names],
        series=noise_series,
        title="Stage-Wise Noise Density by Type",
        ylabel="Noise Density",
        output_path=noise_svg,
        ylim=(0.0, 1.02),
    )

    latency_stage_names = [k for k in ["rag", "filter", "claims", "light_graph", "toolkit", "composer", "total"] if latency_rows and k in latency_rows[0]]
    latency_series = {
        row["question_type"]: [float(row.get(stage, 0.0) or 0.0) for stage in latency_stage_names]
        for row in latency_rows
    }
    latency_svg = out_dir / f"{base}__stage_latency_by_type.svg"
    _plot_multi_line(
        x_labels=[STAGE_LABELS.get(x, x.title()) if x in STAGE_LABELS else x.title() for x in latency_stage_names],
        series=latency_series,
        title="Stage-Wise Latency by Type",
        ylabel="Seconds",
        output_path=latency_svg,
    )

    overall_answer_svg = out_dir / f"{base}__stage_answerability_overall.svg"
    _plot_bar(
        labels=[STAGE_LABELS.get(x, x) for x in stage_names],
        values=[float(agg["answerability_overall"].get(stage, 0.0) or 0.0) for stage in stage_names],
        title="Overall Stage Answerability",
        ylabel="Coverage@rec50",
        output_path=overall_answer_svg,
        ylim=(0.0, 1.02),
    )
    overall_latency_svg = out_dir / f"{base}__stage_latency_overall.svg"
    _plot_bar(
        labels=[STAGE_LABELS.get(x, x.title()) if x in STAGE_LABELS else x.title() for x in latency_stage_names],
        values=[float(agg["latency_overall"].get(stage, 0.0) or 0.0) for stage in latency_stage_names],
        title="Overall Stage Latency",
        ylabel="Seconds",
        output_path=overall_latency_svg,
    )
    overall_noise_svg = out_dir / f"{base}__stage_noise_density_overall.svg"
    _plot_bar(
        labels=[STAGE_LABELS.get(x, x) for x in stage_names],
        values=[float(agg["noise_overall"].get(stage, 0.0) or 0.0) for stage in stage_names],
        title="Overall Stage Noise Density",
        ylabel="Noise Density",
        output_path=overall_noise_svg,
        ylim=(0.0, 1.02),
    )

    return {
        "answerability_csv": str(answer_csv),
        "answerability_md": str(answer_md),
        "noise_csv": str(noise_csv),
        "noise_md": str(noise_md),
        "latency_csv": str(latency_csv),
        "latency_md": str(latency_md),
        "answerability_svg": str(answer_svg),
        "noise_svg": str(noise_svg),
        "latency_svg": str(latency_svg),
        "overall_answerability_svg": str(overall_answer_svg),
        "overall_noise_svg": str(overall_noise_svg),
        "overall_latency_svg": str(overall_latency_svg),
    }


def render_comparison_visuals(*, comparison_json: str | Path, output_dir: str | Path, prefix: str = "") -> Dict[str, str]:
    payload = _load_json(comparison_json)
    if not isinstance(payload, dict):
        raise ValueError("Comparison JSON must be an object payload.")
    out_dir = _ensure_dir(output_dir)
    base = prefix or Path(str(comparison_json)).stem
    base = _sanitize(base)

    modes = list(payload.get("modes", []))
    type_answer_rows = list(payload.get("type_answer_acc", []))
    type_latency_rows = list(payload.get("type_latency_sec", []))

    summary_csv = out_dir / f"{base}__run_summary.csv"
    summary_md = out_dir / f"{base}__run_summary.md"
    answer_csv = out_dir / f"{base}__type_answer_acc.csv"
    answer_md = out_dir / f"{base}__type_answer_acc.md"
    latency_csv = out_dir / f"{base}__type_latency_sec.csv"
    latency_md = out_dir / f"{base}__type_latency_sec.md"

    summary_fields = [
        "mode",
        "run_id",
        "final_answer_acc",
        "avg_latency_sec",
        "avg_answer_token_density",
        "avg_noise_density",
        "retrieval_answer_span_hit_rate",
        "retrieval_support_sentence_hit_rate",
        "graph_answer_span_hit_rate",
        "graph_support_sentence_hit_rate",
        "graph_ingest_accept_rate",
    ]
    _write_csv(summary_csv, modes, summary_fields)
    _write_md_table(summary_md, "Comparison Run Summary", modes, summary_fields)

    answer_fields = ["question_type"] + MODE_ORDER
    _write_csv(answer_csv, type_answer_rows, answer_fields)
    _write_md_table(answer_md, "Type Answer Accuracy", type_answer_rows, answer_fields)
    latency_fields = ["question_type"] + MODE_ORDER
    _write_csv(latency_csv, type_latency_rows, latency_fields)
    _write_md_table(latency_md, "Type Latency Sec", type_latency_rows, latency_fields)

    summary_acc_svg = out_dir / f"{base}__overall_accuracy.svg"
    _plot_bar(
        labels=[MODE_LABELS.get(str(row.get("mode", "")), str(row.get("mode", ""))) for row in modes],
        values=[_safe_float(row.get("final_answer_acc"), 0.0) for row in modes],
        title="Overall Accuracy by Mode",
        ylabel="Final Answer Acc",
        output_path=summary_acc_svg,
        ylim=(0.0, 1.02),
    )
    summary_latency_svg = out_dir / f"{base}__overall_latency.svg"
    _plot_bar(
        labels=[MODE_LABELS.get(str(row.get("mode", "")), str(row.get("mode", ""))) for row in modes],
        values=[_safe_float(row.get("avg_latency_sec"), 0.0) for row in modes],
        title="Overall Latency by Mode",
        ylabel="Seconds",
        output_path=summary_latency_svg,
    )

    if type_answer_rows:
        answer_heatmap_svg = out_dir / f"{base}__type_answer_acc_heatmap.svg"
        _plot_heatmap(
            x_labels=[MODE_LABELS.get(m, m) for m in MODE_ORDER],
            y_labels=[str(row.get("question_type", "")) for row in type_answer_rows],
            values=[[_safe_float(row.get(mode), 0.0) for mode in MODE_ORDER] for row in type_answer_rows],
            title="Type Answer Accuracy",
            output_path=answer_heatmap_svg,
            cmap="Blues",
        )
    else:
        answer_heatmap_svg = None

    if type_latency_rows:
        latency_heatmap_svg = out_dir / f"{base}__type_latency_sec_heatmap.svg"
        _plot_heatmap(
            x_labels=[MODE_LABELS.get(m, m) for m in MODE_ORDER],
            y_labels=[str(row.get("question_type", "")) for row in type_latency_rows],
            values=[[_safe_float(row.get(mode), 0.0) for mode in MODE_ORDER] for row in type_latency_rows],
            title="Type Latency (sec)",
            output_path=latency_heatmap_svg,
            cmap="YlGnBu",
        )
    else:
        latency_heatmap_svg = None

    return {
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "type_answer_csv": str(answer_csv),
        "type_answer_md": str(answer_md),
        "type_latency_csv": str(latency_csv),
        "type_latency_md": str(latency_md),
        "overall_accuracy_svg": str(summary_acc_svg),
        "overall_latency_svg": str(summary_latency_svg),
        "type_answer_heatmap_svg": str(answer_heatmap_svg) if answer_heatmap_svg else "",
        "type_latency_heatmap_svg": str(latency_heatmap_svg) if latency_heatmap_svg else "",
    }


def main() -> None:
    args = parse_args()
    out_dir = _ensure_dir(args.output_dir)
    outputs: Dict[str, Any] = {}
    if args.audit_json:
        outputs["audit"] = render_audit_visuals(
            audit_json=args.audit_json,
            output_dir=out_dir,
            prefix=args.prefix or "audit_visuals",
        )
    if args.comparison_json:
        outputs["comparison"] = render_comparison_visuals(
            comparison_json=args.comparison_json,
            output_dir=out_dir,
            prefix=args.prefix or "comparison_visuals",
        )
    if not outputs:
        raise SystemExit("Please provide --audit-json and/or --comparison-json")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
