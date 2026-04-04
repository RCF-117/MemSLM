"""CLI for interactive and dataset-driven memory-RAG execution."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List
import urllib.error
import urllib.request

from llm_long_memory.evaluation.eval_runner import run_eval
from llm_long_memory.evaluation.dataset_loader import iter_history_messages, load_stream
from llm_long_memory.llm.ollama_client import LLM
from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.utils.helpers import load_config
from llm_long_memory.utils.logger import logger


Message = Dict[str, str]
EvalInstance = Dict[str, Any]


def parse_args(config: Dict[str, Any]) -> argparse.Namespace:
    """Parse CLI arguments for model and host."""
    llm_cfg = config["llm"]
    supported_models = list(llm_cfg["supported_models"])
    parser = argparse.ArgumentParser(description="Memory-RAG system CLI.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(llm_cfg["default_model"]),
        choices=supported_models,
        help="Ollama model name.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=str(llm_cfg["host"]),
        help="Ollama host URL.",
    )
    return parser.parse_args()


def run_dataset(manager: MemoryManager, dataset_path: str, config: Dict[str, Any]) -> None:
    """Run dataset mode and ingest messages into memory without LLM generation."""
    stream_mode = bool(config["dataset"]["stream_mode"])
    logger.info(f"Dataset mode started: path={dataset_path}, stream_mode={stream_mode}")
    instance_index = 0
    message_count = 0
    instances = load_stream(dataset_path) if stream_mode else _load_non_stream(dataset_path)
    for instance in instances:
        instance_index += 1
        qid = str(instance.get("question_id", ""))
        qtype = str(instance.get("question_type", ""))
        logger.info(
            f"Processing instance {instance_index}: question_id={qid}, question_type={qtype}"
        )
        for message in iter_history_messages(instance):
            message_count += 1
            manager.ingest_message(message)
        print(f"Processed instance {instance_index}")
    manager.finalize_ingest()
    logger.info(
        f"Dataset mode completed: instances={instance_index}, messages={message_count}"
    )


def _load_non_stream(dataset_path: str) -> List[EvalInstance]:
    """Load full dataset into memory when stream_mode is disabled."""
    return list(load_stream(dataset_path))


def _fetch_ollama_models(host: str, timeout_sec: int) -> List[str]:
    """Fetch local Ollama model names from /api/tags."""
    req = urllib.request.Request(
        url=f"{host.rstrip('/')}/api/tags",
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    models = payload.get("models", [])
    names: List[str] = []
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if name:
                    names.append(name)
    return names


def run_health(manager: MemoryManager, config: Dict[str, Any]) -> None:
    """Run quick health checks for Ollama connectivity and local DB status."""
    llm_cfg = config["llm"]
    emb_cfg = config["embedding"]
    host = str(llm_cfg["host"]).rstrip("/")
    timeout_sec = int(llm_cfg["request_timeout_sec"])
    main_model = str(llm_cfg["default_model"])
    embedding_model = str(emb_cfg["model"])

    print("Health Check:")
    print(f"host: {host}")
    print(f"main_model: {main_model}")
    print(f"embedding_model: {embedding_model}")

    try:
        available_models = _fetch_ollama_models(host=host, timeout_sec=timeout_sec)
        print("ollama_api: OK")
        print(f"main_model_pulled: {main_model in available_models}")
        print(f"embedding_model_pulled: {embedding_model in available_models}")
        print(f"ollama_models_count: {len(available_models)}")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        print(f"ollama_api: FAIL ({exc})")
        print("main_model_pulled: unknown")
        print("embedding_model_pulled: unknown")

    stats = manager.mid_memory.debug_stats()
    print(f"db_path: {manager.mid_memory.db_path}")
    print(f"topics: {stats['topics']}")
    print(f"chunks: {stats['chunks']}")
    print(f"active_topics: {stats['active_topics']}")
    print(f"inactive_topics: {stats['inactive_topics']}")
    row = manager.mid_memory.conn.execute(
        "SELECT COUNT(*) AS cnt FROM eval_runs"
    ).fetchone()
    eval_runs = int(row["cnt"]) if row else 0
    print(f"eval_runs: {eval_runs}")


def run_interactive(manager: MemoryManager, config: Dict[str, Any]) -> None:
    """Run interactive chat loop with dataset and debug commands."""
    logger.info("Interactive mode started.")
    print("Commands:")
    print("/run_dataset path/to/file.json")
    print("/run_eval path/to/file.json")
    print("/debug")
    print("/health")
    print("exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Session ended.")
            break

        if user_input.startswith("/run_dataset "):
            dataset_path = user_input[len("/run_dataset ") :].strip()
            run_dataset(manager, dataset_path, config)
            continue

        if user_input.startswith("/run_eval "):
            dataset_path = user_input[len("/run_eval ") :].strip()
            run_eval(manager, dataset_path, config)
            continue

        if user_input == "/debug":
            stats = manager.mid_memory.debug_stats()
            long_stats = manager.long_memory.debug_stats()
            print(f"topics: {stats['topics']}")
            print(f"total_chunks: {stats['chunks']}")
            print(f"active_topics: {stats['active_topics']}")
            print(f"inactive_topics: {stats['inactive_topics']}")
            print(f"long_nodes: {long_stats['nodes']}")
            print(f"long_edges: {long_stats['edges']}")
            print(f"long_events: {long_stats['events']}")
            print(f"long_details: {long_stats['details']}")
            print(f"long_active_events: {long_stats['active_events']}")
            print(f"long_superseded_events: {long_stats['superseded_events']}")
            print(f"long_queue: {long_stats['queued_updates']}")
            print(f"long_ingest_total: {long_stats['ingest_event_total']}")
            print(f"long_ingest_accepted: {long_stats['ingest_event_accepted']}")
            print(f"long_ingest_rejected: {long_stats['ingest_event_rejected']}")
            continue

        if user_input == "/health":
            run_health(manager, config)
            continue

        try:
            answer = manager.chat(user_input)
            print(f"Assistant: {answer}\n")
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.error(f"Chat failed: {exc}")
            print(f"Assistant: [Error] {exc}\n")


def main() -> None:
    """Program entry point."""
    config = load_config()
    args = parse_args(config)
    llm = LLM(model_name=args.model, host=args.host)
    manager = MemoryManager(llm=llm, config=config)
    try:
        run_interactive(manager, config)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
