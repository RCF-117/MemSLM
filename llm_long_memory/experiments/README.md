# Experiments

This directory contains the thesis-oriented experiment entry points for MemSLM.

The scripts are intentionally split by responsibility so the research workflow stays auditable:

## 1. Dataset Subset and Split Builders

- `build_eval_subset.py`
  - builds a compact balanced subset from a larger benchmark file
  - useful for quick A/B checks and small diagnostic runs

- `build_eval_split.py`
  - builds a stratified `debug` / `test` split
  - optionally writes a manifest for traceability
  - useful when you want a frozen final test set after iterating on debug

## 2. Thesis Evaluation Runner

- `run_thesis_eval.py`
  - one-shot workflow for subset construction, evaluation, and report export
  - supports model override, judge override, and checkpoint resume

- `run_thesis_compare.py`
  - runs the four thesis comparison protocols in a fixed order:
    - `model-only`
    - `naive rag`
    - `memslm`
    - `ablation`
  - exports one consolidated wide comparison report with per-type accuracy and latency columns
  - keeps MemSLM as the focal column, with the other three modes serving as reference protocols

## 3. Report and Graph Export

- `export_eval_report.py`
  - exports SQLite evaluation results into JSON / Markdown / CSV
  - optionally uses a separate LLM judge for final answer accuracy

- `export_graph.py`
  - exports the long-memory graph into inspectable formats
  - useful for Gephi, browser preview, or paper figures

## 4. Judge Helper

- `llm_judge.py`
  - local judge wrapper used by the report exporter
  - kept separate from generation so the scoring phase remains isolated

## Practical Use

Recommended order:
1. build a debug/test split
2. tune on `debug`
3. freeze `test`
4. run thesis evaluation
5. run thesis comparison
6. export report
7. export graph artifacts

That separation keeps experimentation clean and makes the repository easier to maintain over time.
