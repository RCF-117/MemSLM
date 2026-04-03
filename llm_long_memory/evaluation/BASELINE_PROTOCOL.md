# Frozen Baseline Protocol: midrag_v1

This file defines the reproducible baseline protocol for the Mid-RAG stage.

## Branch and Config
- Branch: `baseline/midrag_v1`
- Frozen config: `llm_long_memory/config/baseline_midrag_v1.yaml`

## Datasets and Default Scope
- Sample set: `llm_long_memory/data/raw/longmemeval_s_sample_20.json`
- Oracle set: `llm_long_memory/data/raw/longmemeval_oracle.json`
- Default run size: first 10 instances per dataset

## Run Command
```bash
python -m llm_long_memory.evaluation.run_baseline \
  --config llm_long_memory/config/baseline_midrag_v1.yaml \
  --sample_limit 10
```

## Notes
- Use this frozen protocol to compare future changes.
- Keep this file unchanged for baseline reproducibility.
