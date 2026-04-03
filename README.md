# MemSLM (Research Prototype)

A modular, local memory-RAG prototype for long-conversation research.

## Repo Goals
- Keep a stable **mid-RAG baseline** branch for reproducible comparisons.
- Develop new memory modules (especially long-term memory) on `main`.
- Keep runtime artifacts and large datasets **out of Git history**.

## Branch Strategy
- `main`: active development (long-memory / architecture upgrades).
- `baseline/midrag_v1`: frozen mid-memory baseline for controlled A/B.

## Project Layout
- `llm_long_memory/main.py`: CLI entry (`interactive`, `/run_dataset`, `/run_eval`, `/debug`, `/health`).
- `llm_long_memory/memory/`: short/mid/long memory orchestration.
- `llm_long_memory/evaluation/`: loaders, metrics runtime, eval persistence.
- `llm_long_memory/baselines/`: baseline protocol, baseline config and runner.
- `llm_long_memory/config/config.yaml`: active full-system config.

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s llm_long_memory/tests -v
python llm_long_memory/main.py
```

## Data Policy
Ignored by Git:
- `llm_long_memory/data/raw/*.json` (LongMemEval raw files)
- `llm_long_memory/data/processed/*.db*` (runtime sqlite)
- `llm_long_memory/logs/`

This keeps repo size clean while preserving reproducibility scripts/config.

## Notes
- This is a thesis/research prototype, not production software.
- Prioritize modularity, observability, and controlled experiments.
