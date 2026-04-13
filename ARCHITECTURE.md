# MemSLM Architecture

This repository uses a layered memory-RAG design with configurable runtime behavior.

## Runtime Flow

1. User input enters `ShortMemory` (recent-turn FIFO window).
2. Overflow/history is persisted into `MidMemory` (SQLite topic/chunk store).
3. Retrieval is hybrid (dense + lexical + topic prior), with chunk-level rerank.
4. Optional long-memory graph context is injected before final generation.
5. `AnsweringPipeline` performs evidence extraction, deterministic helpers (temporal/count), and final response control.
6. Evaluation stores run metadata and per-question results into SQLite.

## Modules

- `llm_long_memory/main.py`: CLI entrypoint.
- `llm_long_memory/cli/runtime.py`: interactive/eval command routing.
- `llm_long_memory/memory/memory_manager.py`: orchestration root.
- `llm_long_memory/memory/memory_manager_chat_runtime.py`: chat-path runtime logic.
- `llm_long_memory/memory/mid_memory.py`: topic/chunk persistence + retrieval APIs.
- `llm_long_memory/memory/long_memory.py`: long-memory graph retrieval/ingest facade.
- `llm_long_memory/memory/answering_pipeline.py`: answer decision and fallback policies.
- `llm_long_memory/evaluation/eval_runner.py`: dataset benchmark loop.
- `llm_long_memory/evaluation/metrics_runtime.py`: match and retrieval-quality metrics.

## Data Stores

- Mid memory DB: `llm_long_memory/data/processed/mid_memory.db`
- Long memory DB: `llm_long_memory/data/processed/long_memory.db`
- Logs: `llm_long_memory/logs/system.log`

## Key Principles

- Config-first behavior: runtime knobs live in `llm_long_memory/config/config.yaml`.
- Separation of concerns: orchestration, retrieval, decision, and persistence are split.
- Evaluation reproducibility: run-level metadata and per-instance outputs are persisted.
- Safe fallback path: deterministic modules provide evidence-grounded guardrails.

## Current Baseline Scope

- Mid-memory hybrid retrieval is the verified baseline path.
- Long-memory graph is integrated but can be switched off in config for isolated RAG tests.
- Final LLM answering can be bypassed in custom retrieval-only tests.
