# Convenience targets for the Agentic Research Assistant.
# Usage: make <target>

.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-dev demo ingest research test lint format clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (runtime deps)
	$(PY) -m pip install -e .

install-dev: ## Install with dev extras (pytest, ruff)
	$(PY) -m pip install -e ".[dev]"

demo: ## Run the offline end-to-end demo (no API key needed)
	$(PY) -m research_assistant demo

ingest: ## Ingest the bundled sample papers into data/index
	$(PY) -m research_assistant ingest examples/sample_papers --dry-run

research: ## Ask a question against the built index (offline stub)
	$(PY) -m research_assistant research "What are the main research gaps?" --dry-run

test: ## Run the test suite
	$(PY) -m pytest

lint: ## Lint with ruff
	ruff check src tests

format: ## Auto-format with ruff
	ruff check --fix src tests
	ruff format src tests

clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf data/index data/index_demo
