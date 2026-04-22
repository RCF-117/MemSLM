# Experiments

This directory contains the experiment-facing entrypoints for MemSLM.

The active experiment workflow is centered on:
- reproducible eval runs
- report export from SQLite
- stage-wise auditing for retrieval / filter / claims / graph / toolkit

## Main Entry Points

### Dataset construction

- `build_eval_subset.py`
  - build compact diagnostic subsets
- `build_eval_split.py`
  - build frozen debug/test splits

### Evaluation

- `run_model_only_eval.py`
  - bare-model baseline
- `run_naive_rag_eval.py`
  - classic retrieve-then-generate baseline
- `run_ablation_eval.py`
  - frozen ablation / baseline protocol
- `run_thesis_eval.py`
  - main MemSLM eval runner
- `run_thesis_compare.py`
  - consolidated comparison report

### Reporting

- `export_eval_report.py`
  - export eval DB rows into report artifacts
- `local_llm_judge.py`
  - optional judge helper used by report export

### Shared helpers

- `direct_eval_runner.py`
  - shared direct-baseline runner for `model-only` and `naive rag`
- `report_audit_utils.py`
  - shared source-audit summary loader for reports

## Recommended Workflow

1. build or select a debug split
2. run source audit to inspect stage-wise evidence quality
3. refresh `memslm`
4. refresh fixed baselines only when needed
5. export the per-run report
6. export the consolidated comparison report

## Graph Utility

- `export_graph.py`
  - active light-graph export from audit artifacts
  - writes one combined HTML / JSON overview canvas for multi-question screenshots
