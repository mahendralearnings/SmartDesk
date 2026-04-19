# ============================================================================
# SmartDesk Makefile — all commands use uv
# ============================================================================
.PHONY: help setup sync lint format typecheck test test-fast run clean docker-build docker-up

# Default target
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────────────
setup: ## First-time setup: install uv, Python, all deps
	@echo "Installing uv..."
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "Installing Python 3.12..."
	uv python install 3.12
	uv python pin 3.12
	@echo "Syncing all dependencies..."
	uv sync
	@echo "Installing pre-commit hooks..."
	uv run pre-commit install
	@echo "\n✅ Setup complete. Run 'make test' to verify."

sync: ## Sync dependencies from lockfile
	uv sync --frozen

# ── Quality ────────────────────────────────────────────────────────────────
lint: ## Lint with ruff
	uv run ruff check src/ tests/ scripts/

format: ## Format with ruff
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

typecheck: ## Type check with mypy
	uv run mypy src/smartdesk/

# ── Testing ────────────────────────────────────────────────────────────────
test: ## Run all tests with coverage
	uv run pytest tests/ -v --cov=src/smartdesk --cov-report=term-missing

test-fast: ## Run tests without coverage (faster)
	uv run pytest tests/ -v --no-cov -x

# ── Run ────────────────────────────────────────────────────────────────────
run: ## Start the FastAPI dev server
	uv run uvicorn smartdesk.api.main:app --reload --port 8000

demo-phase1: ## Run Phase 1 demo
	uv run python scripts/demo_phase1.py

demo-phase2: ## Run Phase 2 demo
	uv run python scripts/demo_phase2.py

demo-phase3: ## Run Phase 3 demo
	uv run python scripts/demo_phase3.py

# ── Dependencies ───────────────────────────────────────────────────────────
add: ## Add a dependency: make add pkg=requests
	uv add $(pkg)

add-dev: ## Add a dev dependency: make add-dev pkg=pytest
	uv add $(pkg) --dev

upgrade: ## Upgrade all dependencies
	uv lock --upgrade
	uv sync

tree: ## Show dependency tree
	uv tree

# ── Docker ─────────────────────────────────────────────────────────────────
docker-build: ## Build Docker image
	docker build -f docker/Dockerfile -t smartdesk:latest .

docker-up: ## Start all services with docker-compose
	docker compose -f docker/docker-compose.yml up -d

docker-down: ## Stop all services
	docker compose -f docker/docker-compose.yml down

# ── Cleanup ────────────────────────────────────────────────────────────────
clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info

nuke: ## Full reset: remove .venv, lockfile, and reinstall
	rm -rf .venv uv.lock
	uv sync
	@echo "✅ Fresh environment created."
