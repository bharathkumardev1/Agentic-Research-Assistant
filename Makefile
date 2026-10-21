# Usage: make <target>. The eval targets call the API and spend money; everything else is offline.

.DEFAULT_GOAL := help
.PHONY: help install test lint demo qasper eval-dev eval-test check-replay labels agreement replay lock image

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-13s %s\n", $$1, $$2}'

install: ## Locked dependencies, then the package in editable mode with the dev tools
	pip install --require-hashes -r requirements.lock
	pip install --no-deps -e . && pip install pytest ruff mypy

test: ## Tests, lint and types; offline
	pytest -q && ruff check src tests && mypy

demo: ## The loop on the bundled sample papers with the offline stub
	python -m research_assistant demo

qasper: ## Download the QASPER dev and test sets and record the tarballs' SHA-256
	research-assistant qasper-download

eval-dev: ## QASPER dev sample, 150 questions; needs OPENAI_API_KEY, $25 cap across the recording
	research-assistant eval --split dev

eval-test: ## QASPER test sample, 150 questions; run once, after dev settled every choice
	research-assistant eval --split test

check-replay: ## Rerun every results file from the recording with no key and compare
	for f in results/qasper-*.json; do research-assistant check-replay $$f || exit 1; done

labels: ## 40 judged test answers for you to grade, verdicts hidden
	research-assistant label-judge

agreement: ## How often the judge agreed with your grades
	research-assistant judge-agreement

replay: ## Recorded test runs as JSON for the site, no key
	research-assistant export-replay

lock: ## Re-resolve requirements.lock for every platform, with hashes
	uv pip compile pyproject.toml --extra service -c constraints.txt -o requirements.lock \
	  --universal --python-version 3.12 --generate-hashes --custom-compile-command "make lock"

image: ## Build the service image
	docker build -t research-assistant .
