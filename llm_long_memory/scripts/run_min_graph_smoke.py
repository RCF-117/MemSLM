"""Run a minimal offline graph-ingest smoke test on one instance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm_long_memory.evaluation.dataset_loader import iter_history_messages, load_stream
from llm_long_memory.llm.ollama_client import LLM
from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.utils.helpers import load_config, resolve_project_path


def _apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    cfg["memory"]["long_memory"]["enabled"] = True
    cfg["memory"]["long_memory"]["online_ingest_enabled"] = False
    cfg["memory"]["long_memory"]["offline_graph"]["enabled"] = True
    cfg["memory"]["long_memory"]["extractor"]["model"] = args.extractor_model
    cfg["memory"]["long_memory"]["extractor"]["timeout_sec"] = args.timeout_sec
    cfg["memory"]["long_memory"]["extractor"]["input_max_chars"] = args.input_max_chars
    cfg["memory"]["long_memory"]["extractor"]["warmup_enabled"] = False
    cfg["memory"]["long_memory"]["offline_graph"]["build_top_chunks"] = args.build_top_chunks
    cfg["memory"]["long_memory"]["offline_graph"]["build_chunk_max_chars"] = args.build_chunk_max_chars
    cfg["retrieval"]["answering"]["graph_refiner_enabled"] = True
    cfg["retrieval"]["answering"]["graph_context_from_store_enabled"] = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal offline graph smoke test")
    parser.add_argument("--dataset", default="data/raw/LongMemEval/longmemeval_oracle_first_20.json")
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--max-history", type=int, default=15)
    parser.add_argument("--extractor-model", default="qwen3:8b")
    parser.add_argument("--timeout-sec", type=int, default=45)
    parser.add_argument("--input-max-chars", type=int, default=180)
    parser.add_argument("--build-top-chunks", type=int, default=1)
    parser.add_argument("--build-chunk-max-chars", type=int, default=160)
    args = parser.parse_args()

    cfg = load_config()
    _apply_overrides(cfg, args)

    dataset_path = resolve_project_path(args.dataset)
    instances = list(load_stream(str(dataset_path)))
    if args.instance_index < 0 or args.instance_index >= len(instances):
        raise IndexError(f"instance_index out of range: {args.instance_index} (size={len(instances)})")
    inst = instances[args.instance_index]

    llm = LLM(model_name=cfg["llm"]["default_model"])
    manager = MemoryManager(llm=llm, config=cfg)
    manager.reset_for_new_instance()

    ingested = 0
    for message in iter_history_messages(inst):
        manager.ingest_message(message)
        ingested += 1
        if ingested >= args.max_history:
            break
    manager.finalize_ingest()
    manager.archive_short_to_mid(clear_short=True)

    question = str(inst.get("question", "")).strip()
    _, topics, chunks = manager.retrieve_context(question)
    accepted = manager.offline_build_long_graph_from_chunks(chunks)
    snippets = manager.long_memory.build_context_snippets(question)
    stats = manager.long_memory.debug_stats()

    print("=== OFFLINE GRAPH SMOKE ===")
    print(f"dataset={dataset_path}")
    print(f"question_id={inst.get('question_id', '')}")
    print(f"ingested_history_messages={ingested}")
    print(f"topics={len(topics)} chunks={len(chunks)}")
    print(f"accepted_events={accepted}")
    print(
        f"events={stats.get('events', 0)} edges={stats.get('edges', 0)} "
        f"details={stats.get('details', 0)}"
    )
    print(
        "extractor: "
        f"calls={int(stats.get('extractor_calls', 0))}, "
        f"success={int(stats.get('extractor_success', 0))}, "
        f"failures={int(stats.get('extractor_failures', 0))}, "
        f"seen={int(stats.get('extractor_seen_messages', 0))}"
    )
    reject_keys = [
        "reject_reason_hard_low_confidence",
        "reject_reason_hard_missing_action",
        "reject_reason_hard_missing_subject_object",
        "reject_reason_quality_below_threshold",
        "reject_reason_legacy_gate_reject",
        "reject_reason_empty_event_text",
    ]
    reject_summary = ", ".join(f"{k}={int(stats.get(k, 0))}" for k in reject_keys)
    print(f"rejects: {reject_summary}")
    print(f"snippets={len(snippets)}")
    for idx, text in enumerate(snippets[:5], start=1):
        print(f"SNIP{idx}: {text[:220]}")

    manager.close()


if __name__ == "__main__":
    main()
