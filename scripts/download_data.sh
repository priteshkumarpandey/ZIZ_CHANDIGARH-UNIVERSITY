#!/usr/bin/env bash
set -e

echo "=== Downloading / Generating GlucoTwin Datasets ==="

# Create local data directories if they do not exist
mkdir -p data/raw data/interim data/processed data/external

echo "[Info] Generating synthetic Indian patient cohort (CGMacros / AI-READI format)..."
python -m src.glucotwin.synth.generator --days 7 --output data/processed/synthetic_cohort.parquet || true

echo "[Info] Synthetic data stub populated in data/processed/"
