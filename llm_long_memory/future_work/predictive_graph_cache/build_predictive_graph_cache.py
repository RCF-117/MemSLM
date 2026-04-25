"""Offline builder for predictive light-graph cache entries."""

from __future__ import annotations

import argparse
import json

from llm_long_memory.llm.ollama_client import LLM
from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.future_work.predictive_graph_cache.predictive_graph_cache import (
    PredictiveGraphCacheBuilder,
    PredictiveGraphCacheStore,
)
from llm_long_memory.utils.helpers import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build predictive light-graph cache.")
    parser.add_argument("--config", default="llm_long_memory/config/config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-windows", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    llm = LLM(model_name=args.model or cfg["llm"]["default_model"])
    manager = MemoryManager(llm=llm, config=cfg)
    try:
        answer_cfg = dict(cfg["retrieval"]["answering"])
        cache_cfg = dict(answer_cfg.get("predictive_graph_cache", {}))
        if not bool(cache_cfg.get("enabled", False)):
            cache_cfg["enabled"] = True
        store = PredictiveGraphCacheStore(
            database_file=str(
                cache_cfg.get(
                    "database_file",
                    "data/processed/predictive_graph_cache/predictive_graph_cache.sqlite",
                )
            ),
            embedding_dim=int(cfg["embedding"]["dim"]),
        )
        try:
            builder = PredictiveGraphCacheBuilder(
                manager,
                store,
                dict(cache_cfg.get("builder", {}), **cache_cfg),
            )
            result = builder.build(
                max_windows=int(args.max_windows) if int(args.max_windows) > 0 else None
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            store.close()
    finally:
        manager.close()


if __name__ == "__main__":
    main()
