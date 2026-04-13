"""CLI entrypoint for interactive and dataset-driven memory-RAG execution."""

from __future__ import annotations

import argparse
from typing import Any, Dict

from llm_long_memory.cli.runtime import run_interactive
from llm_long_memory.llm.ollama_client import LLM
from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.utils.helpers import load_config


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

