# Reproducibility Guide

This document describes the active paper-facing protocol for reproducing the
main MemSLM results.

## Environment

- Python: `3.11+`
- Local Ollama-compatible models:
  - `qwen3:8b`
  - `deepseek-r1:8b`
  - `nomic-embed-text`

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sanity checks:

```bash
make compile
make test
```

## Active Evaluation Protocol

### Main datasets

- `ragdebug10`
  - paper-facing name: `LongMemEval Diagnostic Split`
- `diagnostic_heldout20`
  - paper-facing name: `LongMemEval Held-Out Matched Split`

### Main methods

- `model-only`
- `naive rag`
- `memslm`
- `filter-only ablation`

### Main model setup

- answer model: `qwen3:8b`
- judge model: `deepseek-r1:8b`

## Core Commands

### MemSLM

```bash
python3 -m llm_long_memory.experiments.run_thesis_eval \
  --config llm_long_memory/config/config.yaml \
  --split ragdebug10 \
  --model qwen3:8b \
  --judge \
  --judge-model deepseek-r1:8b
```

### Model-only

```bash
python3 -m llm_long_memory.experiments.run_model_only_eval \
  --config llm_long_memory/config/config.yaml \
  --split ragdebug10 \
  --model qwen3:8b
```

### Naive RAG

```bash
python3 -m llm_long_memory.experiments.run_naive_rag_eval \
  --config llm_long_memory/config/config.yaml \
  --split ragdebug10 \
  --model qwen3:8b
```

### Filter-only Ablation

```bash
python3 -m llm_long_memory.experiments.run_ablation_eval \
  --config llm_long_memory/config/config.yaml \
  --split ragdebug10 \
  --model qwen3:8b
```

### Consolidated comparison

```bash
python3 -m llm_long_memory.experiments.run_thesis_compare \
  --config llm_long_memory/config/config.yaml \
  --split ragdebug10 \
  --model qwen3:8b \
  --judge-model deepseek-r1:8b
```

## Mainline Reporting Scope

The repository treats the following as the active thesis mainline:

1. two LongMemEval splits:
   - `LongMemEval Diagnostic Split`
   - `LongMemEval Held-Out Matched Split`
2. four-way comparison:
   - `model-only`
   - `naive rag`
   - `memslm`
   - `filter-only ablation`
3. extension checks:
   - swapped answer/judge roles
   - LoCoMo external-dataset check

Exploratory prototypes under `llm_long_memory/future_work/` are intentionally
excluded from the mainline protocol unless they are explicitly promoted back
into the active runtime.

## Artifact Locations

Main generated artifacts:

- reports:
  - `llm_long_memory/data/processed/thesis_reports_debug_analysis/`
- stage-wise visuals:
  - `llm_long_memory/data/processed/thesis_visuals/`
- graph exports:
  - `llm_long_memory/data/graphs_thesis_debug_analysis/`

Stable documentation copies used by the repository homepage live under:

- `docs/assets/`

## Notes on Repeatability

- The evaluation DB supports resuming runs via `run_id`.
- The active runtime is the code under `llm_long_memory/memory/`.
- `future_work/` modules are preserved for research continuity but are not part
  of the reproducibility contract for the current thesis results.
