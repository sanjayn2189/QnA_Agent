.PHONY: install ingest run test lint format clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -r requirements.txt

ingest: ## Run Confluence ingestion pipeline
	python scripts/ingest.py

ingest-force: ## Clear vector store and re-ingest from scratch
	python scripts/ingest.py --force

run: ## Start the FastAPI server with hot reload
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test: ## Run all tests
	pytest tests/ -v --tb=short

lint: ## Lint code with ruff
	ruff check src/ tests/

format: ## Format code with ruff
	ruff format src/ tests/

clean: ## Remove Python cache files and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage
