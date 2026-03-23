.DEFAULT_GOAL := help
AGENT_DIR     := mistral-it-agent

# ── Colours ──────────────────────────────────────────────────────────────────
BOLD  := \033[1m
RESET := \033[0m

# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install test test-v test-k lint clean app notebook

help:
	@printf "$(BOLD)mistral-it-agent — available targets$(RESET)\n\n"
	@printf "  %-18s %s\n" "install"    "uv sync --dev (install all dependencies)"
	@printf "  %-18s %s\n" "test"       "run the full test suite"
	@printf "  %-18s %s\n" "test-v"     "run tests with verbose output"
	@printf "  %-18s %s\n" "test-k K=…" "run tests matching a pattern, e.g. make test-k K=tools"
	@printf "  %-18s %s\n" "lint"       "run ruff check (skipped if ruff is not installed)"
	@printf "  %-18s %s\n" "clean"      "remove Python cache files and pytest artefacts"
	@printf "  %-18s %s\n" "notebook"   "launch Jupyter in $(AGENT_DIR)/"
	@printf "  %-18s %s\n" "app"        "launch the Streamlit demo (requires trained classifier)"
	@echo ""

install:
	uv sync --dev

test:
	uv run pytest

test-v:
	uv run pytest -v

test-k:
	@test -n "$(K)" || (echo "Usage: make test-k K=<pattern>"; exit 1)
	uv run pytest -k "$(K)" -v

lint:
	@uv run ruff check . 2>/dev/null || echo "ruff not installed — skipping lint (add ruff to dev dependencies to enable)"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

notebook:
	cd $(AGENT_DIR) && uv run jupyter notebook

app:
	cd $(AGENT_DIR) && uv run streamlit run app.py
