# Results Snapshot

This page records the current mainline repository snapshot used in the public-facing README.

## Mainline Diagnostic Result

Reference run:

- `run_20260425_150142_207e3577`

Source report:

- `llm_long_memory/data/processed/thesis_reports_debug_analysis/run_20260425_150142_207e3577__longmemeval_ragdebug10_rebuilt__model-qwen3_8b__judge-deepseek-r1_8b_report.json`

Configuration summary:

| Field | Value |
| --- | --- |
| Dataset | `longmemeval_ragdebug10_rebuilt.json` |
| Main answer model | `qwen3:8b` |
| Judge model | `deepseek-r1:8b` |
| Total questions | `20` |
| Judged final accuracy | `45.0%` |
| Exact match | `25.0%` |
| Average latency | `33.65s` |
| Retrieval answer-span hit rate | `40.0%` |
| Retrieval support-sentence hit rate | `40.0%` |
| Retrieval evidence hit rate | `100.0%` |
| Light-graph answer-span hit rate | `40.0%` |
| Light-graph support-sentence hit rate | `30.0%` |

## Type Breakdown

| Question type | Accuracy |
| --- | --- |
| `knowledge-update` | `75.0%` |
| `multi-session` | `0.0%` |
| `single-session-assistant` | `33.3%` |
| `single-session-preference` | `33.3%` |
| `single-session-user` | `100.0%` |
| `temporal-reasoning` | `33.3%` |

## Stage-Wise Answerability Figure

![Stage answerability by type](assets/ragdebug_stage_answerability_by_type.svg)

Interpretation:

- retrieval and filtering preserve answer-bearing evidence much more often than the final answer layer can fully exploit
- `single-session-user` is already very strong in the current mainline
- `multi-session` remains the weakest category and is the clearest future optimization target

## Light-Graph Overview Figure

![Combined light graph overview](assets/light_graph_overview.svg)

Interpretation:

- the light graph is useful as a compact structural organizer across questions
- it is strongest as an intermediate representation and analysis surface
- it should not be treated as an automatic answer oracle

## Notes

- These numbers describe the current mainline code path, not exploratory `future_work` prototypes.
- The repository intentionally keeps only one active runtime chain and isolates experimental directions outside the mainline.
