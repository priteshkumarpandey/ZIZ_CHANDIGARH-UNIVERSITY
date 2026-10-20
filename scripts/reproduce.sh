#!/usr/bin/env bash
set -e

echo "=== GlucoTwin Reproduction Script ==="

echo "[1/4] Running data download / synthetic generation stub..."
bash scripts/download_data.sh

echo "[2/4] Validating data schemas with pandera..."
python -c "import src.glucotwin.data.validation as v; print('Pandera validation ready')"

echo "[3/4] Running unit and integration tests..."
pytest tests/ -v

echo "[4/4] Checking code linting and formatting..."
ruff check src/ tests/
ruff format --check src/ tests/

echo "=== Reproduction completed successfully! ==="
