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
  - filter-only ablation built from the active mainline retrieval+filter path
- `run_thesis_eval.py`
  - main MemSLM eval runner
- `run_thesis_compare.py`
  - consolidated comparison report

### Reporting

- `export_eval_report.py`
  - export eval DB rows into report artifacts
- `local_llm_judge.py`
  - optional judge helper used by report export
- `render_thesis_visuals.py`
  - standalone paper-figure/table renderer from saved audit or comparison JSON
  - recommended when you want stage-wise answerability / latency plots without rerunning eval

### Shared helpers

- `direct_eval_runner.py`
  - shared direct-baseline runner for `model-only` and `naive rag`
- `report_audit_utils.py`
  - shared source-audit summary loader for reports

## Recommended Workflow

1. build or select a debug split
   Note: in paper-facing text, the main `ragdebug10` split is referred to as the `LongMemEval Diagnostic Split`.
   The matched held-out evaluation subset is exposed as `diagnostic_heldout20`, with paper-facing name `LongMemEval Held-Out Matched Split`.
   A lightweight LoCoMo generalization subset is exposed as `locomo_matched20`, with paper-facing name `LoCoMo Matched-Distribution 20-QA Subset`.
2. run source audit to inspect stage-wise evidence quality
3. refresh `memslm`
4. refresh fixed baselines only when needed
5. export the per-run report
6. export the consolidated comparison report

## Graph Utility

- `export_graph.py`
  - active light-graph export from audit artifacts
  - writes one combined HTML / JSON overview canvas for multi-question screenshots

## Typical Usage

### Render stage-wise thesis visuals from an audit JSON

```bash
python -m llm_long_memory.experiments.render_thesis_visuals \
  --audit-json llm_long_memory/data/processed/thesis_reports_debug_analysis/answer_source_audit_*.json \
  --output-dir llm_long_memory/data/processed/thesis_visuals \
  --prefix ragdebug_memslm_visuals
```

### Render comparison tables and figures from a comparison JSON

```bash
python -m llm_long_memory.experiments.render_thesis_visuals \
  --comparison-json llm_long_memory/data/processed/thesis_reports_debug_analysis/*_comparison.json \
  --output-dir llm_long_memory/data/processed/thesis_visuals \
  --prefix thesis_compare_visuals
```
