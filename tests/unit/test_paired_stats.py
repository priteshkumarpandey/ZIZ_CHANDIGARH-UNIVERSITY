"""Unit tests for per-patient paired statistics against persistence."""

import json
from pathlib import Path

import pytest

from scripts.compute_paired_stats import compute_paired_statistics


def test_paired_stats_json_exists_and_valid():
    json_path = Path("reports/paired_stats.json")
    assert json_path.exists(), "reports/paired_stats.json must exist"

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_patients"] == 45
    assert data["num_bootstrap_draws"] == 10000
    assert data["confidence_level"] == 0.95
    assert "results" in data
    assert len(data["results"]) == 12

    horizons = {r["horizon"] for r in data["results"]}
    assert horizons == {"+30 min", "+60 min", "+120 min"}

    models = {r["model_key"] for r in data["results"]}
    assert models == {"gru", "lightgbm", "xgboost", "tft"}

    for r in data["results"]:
        assert r["total_count"] == 45
        assert 30 <= r["improved_count"] <= 45
        assert r["ci_upper_95"] > r["ci_lower_95"]
        assert 60.0 <= r["improved_percentage"] <= 100.0
        assert r["definition"] == "mean of per-patient RMSE over 45 patients"
        assert r["status_display"] in ["95% CI excludes 0", "95% CI includes 0"]


@pytest.mark.requires_data
def test_compute_paired_statistics_runner(tmp_path):
    if not Path("reports/test_predictions_h30.parquet").exists():
        pytest.skip(
            "Requires reports/test_predictions_h30.parquet which is excluded in clean clone"
        )
    temp_json = tmp_path / "test_paired.json"
    res = compute_paired_statistics(output_json_path=temp_json, n_bootstrap=500)
    assert temp_json.exists()
    assert res["total_patients"] == 45
    assert len(res["results"]) == 12
