"""Maintenance worker for predictive light-graph cache entries."""

from __future__ import annotations

import argparse
import json

from llm_long_memory.future_work.predictive_graph_cache.predictive_graph_cache import (
    PredictiveGraphCacheMaintenance,
    PredictiveGraphCacheStore,
)
from llm_long_memory.utils.helpers import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Maintain predictive graph cache.")
    parser.add_argument("--config", default="llm_long_memory/config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    answer_cfg = dict(cfg["retrieval"]["answering"])
    cache_cfg = dict(answer_cfg.get("predictive_graph_cache", {}))
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
        worker = PredictiveGraphCacheMaintenance(store, cache_cfg)
        result = worker.run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
