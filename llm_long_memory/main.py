"""CLI for interactive and dataset-driven memory-RAG execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import urllib.error
import urllib.request

from llm_long_memory.evaluation.eval_runner import run_eval
from llm_long_memory.evaluation.dataset_loader import iter_history_messages, load_stream
from llm_long_memory.llm.ollama_client import LLM
from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.utils.helpers import load_config, resolve_project_path
from llm_long_memory.utils.logger import logger


Message = Dict[str, str]
EvalInstance = Dict[str, Any]


def _parse_config_path() -> str:
    """Parse config path early so runtime options can depend on it."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to runtime config file.",
    )
    args, _ = parser.parse_known_args()
    return str(args.config)


def parse_args(config: Dict[str, Any], default_config_path: str) -> argparse.Namespace:
    """Parse CLI arguments for model and host."""
    llm_cfg = config["llm"]
    supported_models = list(llm_cfg["supported_models"])
    parser = argparse.ArgumentParser(description="Memory-RAG system CLI.")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        help="Path to runtime config file.",
    )
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


def _resolve_input_path(path: str) -> Path:
    """Resolve user path from absolute, CWD-relative, or project-relative inputs."""
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return resolve_project_path(path).resolve()


def _require_existing_file(path: str) -> Path:
    resolved = _resolve_input_path(path)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {resolved}")
    return resolved


def _resolve_eval_split_path(config: Dict[str, Any], split_name: str) -> Path:
    """Resolve dataset path by configured split name."""
    dataset_cfg = config["dataset"]
    split_map = dataset_cfg.get("eval_splits", {})
    if not isinstance(split_map, dict):
        raise ValueError("Config key dataset.eval_splits must be a mapping.")
    key = split_name.strip().lower()
    if key not in split_map:
        known = ", ".join(sorted(str(k) for k in split_map.keys()))
        raise ValueError(f"Unknown eval split '{split_name}'. Available: {known}")
    split_path = str(split_map[key]).strip()
    if not split_path:
        raise ValueError(f"Configured path for split '{split_name}' is empty.")
    return _require_existing_file(split_path)


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


def _print_commands() -> None:
    """Print interactive CLI command help."""
    print("Commands:")
    print("/run_dataset path/to/file.json")
    print("/run_eval path/to/file.json")
    print("/run_eval_split split_name")
    print("/debug")
    print("/health")
    print("exit\n")


def _print_debug(manager: MemoryManager) -> None:
    """Print mid/long memory debug stats."""
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


def _handle_dataset_command(
    manager: MemoryManager, config: Dict[str, Any], user_input: str
) -> bool:
    if not user_input.startswith("/run_dataset "):
        return False
    dataset_path = user_input[len("/run_dataset ") :].strip()
    try:
        resolved = _require_existing_file(dataset_path)
        run_dataset(manager, str(resolved), config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        print(f"[Error] {exc}")
    return True


def _handle_eval_command(
    manager: MemoryManager, config: Dict[str, Any], user_input: str
) -> bool:
    if not user_input.startswith("/run_eval "):
        return False
    dataset_path = user_input[len("/run_eval ") :].strip()
    try:
        resolved = _require_existing_file(dataset_path)
        run_eval(manager, str(resolved), config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        print(f"[Error] {exc}")
    return True


def _handle_eval_split_command(
    manager: MemoryManager, config: Dict[str, Any], user_input: str
) -> bool:
    if not user_input.startswith("/run_eval_split "):
        return False
    split_name = user_input[len("/run_eval_split ") :].strip()
    try:
        resolved = _resolve_eval_split_path(config, split_name)
        run_eval(manager, str(resolved), config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        print(f"[Error] {exc}")
    return True


def _handle_builtin_command(
    manager: MemoryManager, config: Dict[str, Any], user_input: str
) -> bool:
    """Handle one built-in command; return True when consumed."""
    if _handle_dataset_command(manager, config, user_input):
        return True
    if _handle_eval_command(manager, config, user_input):
        return True
    if _handle_eval_split_command(manager, config, user_input):
        return True
    if user_input == "/debug":
        _print_debug(manager)
        return True
    if user_input == "/health":
        run_health(manager, config)
        return True
    return False


def run_interactive(manager: MemoryManager, config: Dict[str, Any]) -> None:
    """Run interactive chat loop with dataset and debug commands."""
    logger.info("Interactive mode started.")
    _print_commands()

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

        if _handle_builtin_command(manager, config, user_input):
            continue

        try:
            answer = manager.chat(user_input)
            print(f"Assistant: {answer}\n")
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.error(f"Chat failed: {exc}")
            print(f"Assistant: [Error] {exc}\n")


def main() -> None:
    """Program entry point."""
    config_path = _parse_config_path()
    config = load_config(config_path)
    args = parse_args(config, default_config_path=config_path)
    llm = LLM(model_name=args.model, host=args.host)
    manager = MemoryManager(llm=llm, config=config)
    try:
        run_interactive(manager, config)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
