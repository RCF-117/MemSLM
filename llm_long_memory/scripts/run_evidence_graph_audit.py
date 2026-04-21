"""Run question-scoped evidence filtering + claim extraction + light-graph export."""

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
    parser = argparse.ArgumentParser(description="Evidence graph audit.")
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
        default="evidence_graph_audit",
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
        help="Disable long memory / graph store retrieval.",
    )
    parser.add_argument(
        "--disable-toolkit",
        action="store_true",
        help="Disable specialist toolkit side-effects.",
    )
    parser.add_argument(
        "--enable-evidence-graph",
        action="store_true",
        help="Force-enable the evidence graph extractor path for this run.",
    )
    return parser.parse_args()


def _as_text(value: Any) -> str:
    return str(value) if value is not None else ""


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
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
    if args.enable_evidence_graph:
        cfg["retrieval"].setdefault("evidence_graph", {})["enabled"] = True

    dataset_path = resolve_project_path(str(args.dataset))
    output_dir = resolve_project_path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if int(args.max_items) > 0:
        data = data[: int(args.max_items)]

    llm = LLM()
    manager = MemoryManager(llm=llm, config=cfg)

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(data, start=1):
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
        bundle = manager.build_evidence_graph_bundle(question)
        rows.append(
            {
                "index": idx,
                "qid": _as_text(item.get("qid", item.get("id", idx))),
                "question": question,
                "gold": _as_text(item.get("answer", item.get("gold", ""))),
                "query_plan": bundle.get("query_plan", {}),
                "unified_source": bundle.get("unified_source", []),
                "filtered_pack": bundle.get("filtered_pack", {}),
                "claim_result": {
                    "enabled": bool(bundle.get("claim_result", {}).get("enabled", False)),
                    "model": _as_text(bundle.get("claim_result", {}).get("model", "")),
                    "support_units": list(bundle.get("claim_result", {}).get("support_units", [])),
                    "claims": list(bundle.get("claim_result", {}).get("claims", [])),
                    "stats": dict(bundle.get("claim_result", {}).get("stats", {})),
                },
                "light_graph": bundle.get("light_graph", {}),
            }
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = Path(dataset_path).stem
    base = output_dir / f"{args.output_prefix}_{stamp}__{dataset_name}"
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json_path)


if __name__ == "__main__":
    main()
