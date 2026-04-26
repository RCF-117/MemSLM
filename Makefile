PYTHON ?= python3
CONFIG ?= llm_long_memory/config/config.yaml
MODEL ?= qwen3:8b
JUDGE ?= deepseek-r1:8b
SPLIT ?= ragdebug10

.PHONY: help test compile lint format eval-memslm eval-model-only eval-naive-rag eval-ablation compare

help:
	@echo "MemSLM maintenance targets"
	@echo ""
	@echo "  make test             Run unit tests"
	@echo "  make compile          Compile-check Python modules"
	@echo "  make lint             Run a lightweight lint pass (compile + unittest)"
	@echo "  make format           Print the formatting tools expected by this repo"
	@echo "  make eval-memslm      Run the main MemSLM evaluation"
	@echo "  make eval-model-only  Run the model-only baseline"
	@echo "  make eval-naive-rag   Run the naive RAG baseline"
	@echo "  make eval-ablation    Run the filter-only ablation"
	@echo "  make compare          Export the consolidated thesis comparison report"

test:
	$(PYTHON) -m unittest discover -s llm_long_memory/tests -v

compile:
	$(PYTHON) -m compileall llm_long_memory

lint: compile test

format:
	@echo "This repository uses black/isort/ruff conventions via pyproject.toml."
	@echo "Suggested commands:"
	@echo "  black llm_long_memory"
	@echo "  isort llm_long_memory"
	@echo "  ruff check llm_long_memory"

eval-memslm:
	$(PYTHON) -m llm_long_memory.experiments.run_thesis_eval --config $(CONFIG) --split $(SPLIT) --model $(MODEL) --judge --judge-model $(JUDGE)

eval-model-only:
	$(PYTHON) -m llm_long_memory.experiments.run_model_only_eval --config $(CONFIG) --split $(SPLIT) --model $(MODEL)

eval-naive-rag:
	$(PYTHON) -m llm_long_memory.experiments.run_naive_rag_eval --config $(CONFIG) --split $(SPLIT) --model $(MODEL)

eval-ablation:
	$(PYTHON) -m llm_long_memory.experiments.run_ablation_eval --config $(CONFIG) --split $(SPLIT) --model $(MODEL)

compare:
	$(PYTHON) -m llm_long_memory.experiments.run_thesis_compare --config $(CONFIG) --split $(SPLIT) --model $(MODEL) --judge-model $(JUDGE)
