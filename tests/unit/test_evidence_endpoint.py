import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.glucotwin.serving.app import app

client = TestClient(app)


def test_evidence_endpoint_returns_data_matching_reports():
    response = client.get("/api/v1/evidence")
    assert response.status_code == 200
    data = response.json()

    assert "cv_metrics" in data
    assert "operating_thresholds" in data
    assert "whatif_validation" in data
    assert "data_licenses" in data

    # Verify against disk files
    cv_path = Path("reports/metrics_cv.json")
    if cv_path.exists():
        with open(cv_path, encoding="utf-8") as f:
            cv_disk = json.load(f)
        assert data["cv_metrics"].get("lora_personalization") == cv_disk.get("lora_personalization")
        pers_h30 = data["cv_metrics"]["models"]["persistence"]["h30"]
        assert "rmse_std" in pers_h30
        assert pers_h30["rmse_std"] == cv_disk["models"]["persistence"]["h30"]["rmse_std"]
        assert pers_h30["rmse_std"] == 0.89
        assert "mae_std" in pers_h30
        assert pers_h30["mae_std"] == 0.44
        assert "mard_std" in pers_h30
        assert pers_h30["mard_std"] == 0.38
        assert "zone_a_std" in pers_h30
        assert pers_h30["zone_a_std"] == 1.26
        assert "zone_ab_std" in pers_h30
        assert pers_h30["zone_ab_std"] == 0.12

    thresh_path = Path("reports/operating_thresholds.json")
    if thresh_path.exists():
        with open(thresh_path, encoding="utf-8") as f:
            thresh_disk = json.load(f)
        assert data["operating_thresholds"].get("decision_threshold") == thresh_disk.get(
            "decision_threshold"
        )

    whatif_path = Path("reports/whatif_validation.json")
    if whatif_path.exists():
        with open(whatif_path, encoding="utf-8") as f:
            whatif_disk = json.load(f)
        assert data["whatif_validation"].get("t2d_cohort", {}).get(
            "mean_predicted_rise_mgdl"
        ) == whatif_disk.get("t2d_cohort", {}).get("mean_predicted_rise_mgdl")

    # Verify data licenses content
    licenses_path = Path("DATA_LICENSES.md")
    if licenses_path.exists():
        assert len(data["data_licenses"]) > 0
        assert "CGMacros" in data["data_licenses"]
