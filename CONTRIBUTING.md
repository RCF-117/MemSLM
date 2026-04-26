# Contributing Guide

This repository is a research-grade engineering project. Contributions should improve clarity, reproducibility, or experimental control without weakening the reproducibility contract of the active mainline.

## Working Principles

- Keep the baseline branch and baseline config frozen unless you are intentionally updating the comparison protocol.
- Make changes on `main` unless a task explicitly targets the baseline.
- Prefer small, focused commits.
- Keep runtime behavior config-driven.
- Avoid introducing hidden constants into retrieval, prompting, evaluation, or graph construction.

## Branch Discipline

- `main`: active development and thesis experiments

Recommended commit style:
- one concern per commit
- one change set per commit
- clear message format such as `feat(scope): summary`

## Testing

Before merging or pushing a meaningful change, run the relevant checks:

```bash
make compile
make test
```

If the change touches evaluation or experiment runners, also verify the help output and a minimal dry run when possible.

## Experiment Hygiene

- Use `debug` splits for iteration.
- Freeze the `test` split before final reporting.
- Keep judge evaluation separate from generation.
- Resume long runs with `run_id` instead of rerunning everything.
- When you crop question types for experiments, keep the rule explicit and documented.

## Data and Artifact Hygiene

Do not commit:
- raw benchmark JSON files
- SQLite databases
- logs
- generated reports
- graph exports

These are runtime artifacts and should remain outside version control.

Tracked placeholders keep the directory structure stable:
- `llm_long_memory/data/raw/.gitkeep`
- `llm_long_memory/data/raw/LongMemEval/.gitkeep`
- `llm_long_memory/data/raw/LoCoMo/.gitkeep`
- `llm_long_memory/data/processed/.gitkeep`
- `llm_long_memory/data/graphs/.gitkeep`

## Config Discipline

- Treat `llm_long_memory/config/config.yaml` as the single runtime source of truth.
- If a parameter affects retrieval, answering, graph construction, or evaluation, it should be explicit in config.
- Prefer removing unused knobs over silently keeping them.

## Recommended Review Standard

Before merging a change, ask:

1. Does it keep the baseline reproducible?
2. Does it make the repository easier to inspect or resume?
3. Does it avoid coupling judge, generation, and evaluation too tightly?
4. Does it preserve the ability to run controlled ablations?

If the answer to most of these is yes, the change is probably aligned with the repository’s goals.
