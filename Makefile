.PHONY: help setup lint test up down reproduce clean

help:
	@echo "GlucoTwin Development and Operations"
	@echo "  setup      - Install dependencies and pre-commit hooks"
	@echo "  lint       - Run ruff linter and type-checking"
	@echo "  test       - Run pytest test suite"
	@echo "  up         - Start all services with docker compose"
	@echo "  down       - Stop all running docker compose services"
	@echo "  reproduce  - Run end-to-end data generation and pipeline check"
	@echo "  clean      - Clean cache and temporary build files"

setup:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

test:
	pytest tests/ -v --cov=src/glucotwin

up:
	docker compose up --build -d

down:
	docker compose down

reproduce:
	bash scripts/reproduce.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
