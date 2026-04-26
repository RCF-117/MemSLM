# Thesis Asset Checklist

This file collects the most useful figures, tables, and source artifacts for the current thesis draft.

## Recommended Main Tables

### Table 1. Four-Way Comparison on LongMemEval Diagnostic Split

Purpose:

- main benchmark table for the diagnostic split
- compare `model-only`, `naive rag`, `memslm`, and `filter-only ablation`

Source files:

- [comparison.md](llm_long_memory/data/processed/thesis_reports_debug_analysis/LongMemEval_Diagnostic_Split__model-qwen3_8b__judge-deepseek-r1_8b__memslm-centered_comparison.md)
- [comparison.csv](llm_long_memory/data/processed/thesis_reports_debug_analysis/LongMemEval_Diagnostic_Split__model-qwen3_8b__judge-deepseek-r1_8b__memslm-centered_comparison.csv)

Recommended columns:

- method
- judged accuracy
- exact match
- average latency
- answer density
- noise density

### Table 2. Four-Way Comparison on LongMemEval Held-Out Matched Split

Purpose:

- held-out generalization table
- same four methods, matched type distribution, non-overlapping questions

Source files:

- [comparison.md](llm_long_memory/data/processed/thesis_reports_debug_analysis/LongMemEval_Held-Out_Matched_Split__model-qwen3_8b__judge-deepseek-r1_8b__memslm-centered_comparison.md)
- [comparison.csv](llm_long_memory/data/processed/thesis_reports_debug_analysis/LongMemEval_Held-Out_Matched_Split__model-qwen3_8b__judge-deepseek-r1_8b__memslm-centered_comparison.csv)

### Table 3. Type Breakdown on Diagnostic Split

Purpose:

- show where gains come from
- especially `knowledge-update`, `single-session-user`, and weak `multi-session`

Source files:

- [type_answer_acc.md](llm_long_memory/data/processed/thesis_visuals/diagnostic_split_compare__type_answer_acc.md)
- [type_answer_acc.csv](llm_long_memory/data/processed/thesis_visuals/diagnostic_split_compare__type_answer_acc.csv)

### Table 4. Type Breakdown on Held-Out Split

Purpose:

- held-out robustness by question type

Source files:

- [type_answer_acc.md](llm_long_memory/data/processed/thesis_visuals/heldout_split_compare__type_answer_acc.md)
- [type_answer_acc.csv](llm_long_memory/data/processed/thesis_visuals/heldout_split_compare__type_answer_acc.csv)

## Recommended Main Figures

### Figure 1. Stage-Wise Answerability on Diagnostic Split

Use:

- primary pipeline analysis figure
- shows where answer-bearing signal survives across stages

Preferred asset:

- [diagnostic_stage_answerability_by_type.svg](docs/assets/diagnostic_stage_answerability_by_type.svg)

Raw source:

- [stage_answerability_by_type.svg](llm_long_memory/data/processed/thesis_visuals/diagnostic_split_audit__stage_answerability_by_type.svg)

### Figure 2. Stage-Wise Answerability on Held-Out Split

Use:

- held-out structural robustness figure

Preferred asset:

- [heldout_stage_answerability_by_type.svg](docs/assets/heldout_stage_answerability_by_type.svg)

Raw source:

- [stage_answerability_by_type.svg](llm_long_memory/data/processed/thesis_visuals/heldout_split_audit__stage_answerability_by_type.svg)

### Figure 3. Combined Light Graph Overview on Diagnostic Split

Use:

- main structural visualization
- useful in method and analysis sections

Preferred asset:

- [diagnostic_light_graph_overview.svg](docs/assets/diagnostic_light_graph_overview.svg)

Interactive/raw sources:

- [combined.svg](llm_long_memory/data/graphs_thesis_debug_analysis/ragdebug10_memslm_light_graph__combined.svg)
- [combined.html](llm_long_memory/data/graphs_thesis_debug_analysis/ragdebug10_memslm_light_graph__combined.html)

### Figure 4. Combined Light Graph Overview on Held-Out Split

Use:

- held-out structural visualization

Preferred asset:

- [heldout_light_graph_overview.svg](docs/assets/heldout_light_graph_overview.svg)

Interactive/raw sources:

- [combined.svg](llm_long_memory/data/graphs_thesis_debug_analysis/diagnostic_heldout20_memslm_light_graph__combined.svg)
- [combined.html](llm_long_memory/data/graphs_thesis_debug_analysis/diagnostic_heldout20_memslm_light_graph__combined.html)

## Recommended Secondary Figures

### Overall Accuracy Bar Charts

- [diagnostic_overall_accuracy.svg](llm_long_memory/data/processed/thesis_visuals/diagnostic_split_compare__overall_accuracy.svg)
- [heldout_overall_accuracy.svg](llm_long_memory/data/processed/thesis_visuals/heldout_split_compare__overall_accuracy.svg)

### Overall Latency Bar Charts

- [diagnostic_overall_latency.svg](llm_long_memory/data/processed/thesis_visuals/diagnostic_split_compare__overall_latency.svg)
- [heldout_overall_latency.svg](llm_long_memory/data/processed/thesis_visuals/heldout_split_compare__overall_latency.svg)

### Stage-Wise Noise Density

- [diagnostic_noise_by_type.svg](llm_long_memory/data/processed/thesis_visuals/diagnostic_split_audit__stage_noise_density_by_type.svg)
- [heldout_noise_by_type.svg](llm_long_memory/data/processed/thesis_visuals/heldout_split_audit__stage_noise_density_by_type.svg)

### Stage-Wise Latency

- [diagnostic_latency_by_type.svg](llm_long_memory/data/processed/thesis_visuals/diagnostic_split_audit__stage_latency_by_type.svg)
- [heldout_latency_by_type.svg](llm_long_memory/data/processed/thesis_visuals/heldout_split_audit__stage_latency_by_type.svg)

## Suggested Chapter Placement

### Method

- Figure 3 or Figure 4
- system overview diagram from README / ARCHITECTURE

### Main Results

- Table 1
- Table 2
- one overall accuracy figure

### Analysis

- Figure 1
- Figure 2
- type breakdown tables
- noise density / latency stage plots

### Qualitative Structural Analysis

- combined light-graph overview figures

## Extension Experiments

These are useful as additional generalization checks, but not required for the current two-split core presentation.

### Extension A. Swapped Answer/Judge Roles

Purpose:

- verify that the framework does not depend on a single answer-model / judge ordering

Source files:

- [swapped-role report](llm_long_memory/data/processed/thesis_reports_debug_analysis/run_20260426_103908_dc8d2874__longmemeval_eval_subset_matched_to_diagnostic_split__model-deepseek-r1_8b__judge-qwen3_8b_report.md)
- [swapped-role json](llm_long_memory/data/processed/thesis_reports_debug_analysis/run_20260426_103908_dc8d2874__longmemeval_eval_subset_matched_to_diagnostic_split__model-deepseek-r1_8b__judge-qwen3_8b_report.json)

Recommended use:

- short paragraph in the generalization or robustness section
- report the overall `final_answer_acc` and average latency only

### Extension B. LoCoMo External-Dataset Check

Purpose:

- show that the framework can run on a different long-conversation QA dataset format without replacing the main pipeline

Source files:

- [locomo model-only report](llm_long_memory/data/processed/thesis_reports_debug_analysis/locomo_model_only_matched20_report.md)
- [locomo model-only json](llm_long_memory/data/processed/thesis_reports_debug_analysis/locomo_model_only_matched20_report.json)
- [locomo report](llm_long_memory/data/processed/thesis_reports_debug_analysis/run_20260426_113651_62a381bc__locomo20_matched_distribution__model-qwen3_8b__judge-deepseek-r1_8b_report.md)
- [locomo json](llm_long_memory/data/processed/thesis_reports_debug_analysis/run_20260426_113651_62a381bc__locomo20_matched_distribution__model-qwen3_8b__judge-deepseek-r1_8b_report.json)

Recommended use:

- short external-validity subsection
- compare `memslm` against `model-only`, not against the full four-way LongMemEval protocol
- emphasize compatibility and transfer, not headline SOTA-style performance

These should be presented as external-validity checks rather than part of the core four-way comparison grid.
