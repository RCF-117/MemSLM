# Future Work

This directory contains exploratory modules that are intentionally **not**
part of the active MemSLM runtime path.

Design intent:

- keep the thesis mainline pipeline stable and reproducible
- preserve promising research prototypes without polluting the active chain
- make it easy to revisit failed or partial ideas later

Current contents:

- `predictive_graph_cache/`
  - offline anticipated-query graph-cache prototype
  - validated as an architectural prototype
  - not integrated into the active runtime because online hit-rate was not yet
    high enough to justify the added offline cost

Rule of thumb:

- `llm_long_memory/memory/` is the active runtime
- `llm_long_memory/experiments/` is the active evaluation/reporting surface
- `llm_long_memory/future_work/` is for bounded research prototypes and
  negative-result directions that should remain inspectable but isolated
