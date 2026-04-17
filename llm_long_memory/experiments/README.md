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

- `run_model_only_eval.py`
  - runs the bare-model baseline directly on the raw dataset
  - bypasses retrieval, graph, and long-memory code paths entirely

- `run_naive_rag_eval.py`
  - runs a classic retrieve-then-generate baseline
  - uses a simple passage retrieval stage without the richer MemSLM stack

- `run_ablation_eval.py`
  - runs the frozen Mid-RAG baseline with long memory disabled
  - acts as the ablation/reference protocol for MemSLM

- `run_thesis_eval.py`
  - one-shot workflow for subset construction, evaluation, and report export
  - supports model override, judge override, and checkpoint resume

- `run_thesis_compare.py`
  - builds one consolidated wide comparison report from already stored runs
  - resolves `model-only`, `naive rag`, `memslm`, and `ablation` from the mode registry or the latest legacy comparison artifact
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
4. run `model-only` / `naive rag` once when needed
5. run `ablation` once when needed
6. run thesis evaluation for `memslm`
7. build the consolidated comparison report
8. export report
9. export graph artifacts

That separation keeps experimentation clean and makes the repository easier to maintain over time.
