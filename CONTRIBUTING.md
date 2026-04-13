# Contributing Guide

## Workflow
1. Keep `baseline/midrag_v1` frozen unless intentionally updating baseline protocol.
2. Implement new features on `main`.
3. Commit in small, scoped units:
   - one concern per commit
   - clear commit message (`type(scope): summary`)
4. Run tests before commit:
```bash
pytest -q llm_long_memory/tests
```

## What not to commit
- Large datasets in `data/raw/*.json`
- SQLite DB files in `data/processed/*.db*`
- Logs and caches

## Branch Naming (recommended)
- `feat/...`
- `fix/...`
- `chore/...`

## Baseline Discipline
For fair evaluation:
- do not change baseline algorithm/params in ad-hoc experiments
- keep baseline config in `llm_long_memory/baselines/baseline_midrag_v1.yaml`

## Config Discipline
- Treat `llm_long_memory/config/config.yaml` as the single runtime source of truth.
- Avoid hidden constants in code paths that impact retrieval, prompting, or evaluation.
