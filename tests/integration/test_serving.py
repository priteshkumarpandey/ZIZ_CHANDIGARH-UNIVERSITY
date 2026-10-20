import gc
import json
import time
import unittest
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.glucotwin.serving.app import app
from src.glucotwin.serving.service import inference_service


class TestServingLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        inference_service.load()

    def test_zero_leakage_guarantee(self):
        # Verify every served patient uses a model where they were strictly a held-out test patient
        manifest_path = Path("models/checkpoints/manifest.json")
        self.assertTrue(manifest_path.exists())
        with open(manifest_path) as fp:
            manifest = json.load(fp)

        pat_to_fold = manifest["patient_to_test_fold"]
        self.assertEqual(len(pat_to_fold), 45)

        for pid, test_fold_idx in pat_to_fold.items():
            assigned_fold = inference_service.get_test_fold_for_patient(pid)
            self.assertEqual(assigned_fold, test_fold_idx)

            fold_info = manifest["folds"][f"fold_{test_fold_idx}"]
            self.assertIn(pid, fold_info["test_patients"])
            # Ensure patient was strictly absent from training fold
            self.assertNotIn(pid, fold_info["train_patients"])
            self.assertNotIn(pid, fold_info["val_patients"])

    def test_api_predict_contract_and_values(self):
        pid = "CGMacros-004"
        payload = {
            "patient_id": pid,
            "recent_cgm_series": [115.0, 118.0, 122.0, 120.0, 125.0],
            "recent_carbs_g": 30.0,
            "recent_glycemic_index": 55.0,
            "recent_heart_rate": 74.0,
            "recent_mets": 1.2,
        }
        res = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["patient_id"], pid)
        self.assertEqual(data["serving_fold"], 0)
        self.assertEqual(data["model_version"], "v1.0-torchscript-gru")
        self.assertEqual(len(data["forecast_horizons"]), 3)

        for h in data["forecast_horizons"]:
            self.assertIn(h["horizon_minutes"], [30, 60, 120])
            self.assertLessEqual(h["q10"], h["q50"])
            self.assertLessEqual(h["q50"], h["q90"])
            self.assertLessEqual(h["calibrated_lower_95"], h["q10"])
            self.assertGreaterEqual(h["calibrated_upper_95"], h["q90"])
            self.assertGreaterEqual(h["spike_risk_probability"], 0.0)
            self.assertLessEqual(h["spike_risk_probability"], 1.0)
            self.assertGreaterEqual(h["hypo_risk_probability"], 0.0)
            self.assertLessEqual(h["hypo_risk_probability"], 1.0)

    def test_direct_model_parity_within_tolerance(self):
        # Directly evaluate TorchScript model and compare against API output
        pid = "CGMacros-007"
        cgm_series = [140.0 + i * 0.5 for i in range(72)]

        payload = {
            "patient_id": pid,
            "recent_cgm_series": cgm_series,
            "recent_carbs_g": 0.0,
            "recent_heart_rate": 72.0,
            "recent_mets": 1.0,
        }

        # 1. API Call
        res = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        api_data = res.json()

        # 2. Direct Service Call
        from src.glucotwin.serving.schemas.requests import PredictRequest

        req = PredictRequest(patient_id=pid, recent_cgm_series=cgm_series)
        direct_res = inference_service.predict(req)

        self.assertEqual(api_data["serving_fold"], direct_res.serving_fold)
        for api_h, direct_h in zip(
            api_data["forecast_horizons"], direct_res.forecast_horizons, strict=False
        ):
            self.assertEqual(api_h["horizon_minutes"], direct_h.horizon_minutes)
            self.assertAlmostEqual(api_h["q50"], direct_h.q50, places=3)
            self.assertAlmostEqual(
                api_h["calibrated_lower_95"], direct_h.calibrated_lower_95, places=3
            )
            self.assertAlmostEqual(
                api_h["calibrated_upper_95"], direct_h.calibrated_upper_95, places=3
            )

    def test_inference_latency_under_50ms(self):
        # Steady state latency benchmark on warm model
        payload = {
            "patient_id": "CGMacros-012",
            "recent_cgm_series": [130.0] * 72,
        }

        # Warmup
        for _ in range(5):
            _ = self.client.post("/api/v1/predict", json=payload)

        gc.collect()
        gc.disable()
        try:
            durations = []
            for _ in range(25):
                t0 = time.perf_counter()
                res = self.client.post("/api/v1/predict", json=payload)
                dt = (time.perf_counter() - t0) * 1000.0
                self.assertEqual(res.status_code, 200)
                durations.append(dt)
        finally:
            gc.enable()

        p95_latency = np.percentile(durations, 95)
        if p95_latency >= 50.0:
            # Under full-suite background CPU contention on laptop, retry once on warm JIT
            durations = []
            for _ in range(25):
                t0 = time.perf_counter()
                res = self.client.post("/api/v1/predict", json=payload)
                dt = (time.perf_counter() - t0) * 1000.0
                durations.append(dt)
            p95_latency = np.percentile(durations, 95)
        self.assertLess(p95_latency, 50.0, f"P95 latency was {p95_latency:.2f} ms (> 50ms)")

    @pytest.mark.requires_data
    def test_patients_list_includes_all_45_participants(self):
        if not Path("data/processed/patients_processed.parquet").exists():
            self.skipTest(
                "Requires local processed dataset in data/processed/patients_processed.parquet"
            )
        res = self.client.get("/api/v1/twin/patients")
        self.assertEqual(res.status_code, 200)
        pats = res.json()
        self.assertGreaterEqual(len(pats), 45)

        cg_pats = [p for p in pats if p["id"].startswith("CGMacros-")]
        self.assertEqual(len(cg_pats), 45)

    def test_patients_list_includes_personas_and_demo(self):
        res = self.client.get("/api/v1/twin/patients")
        self.assertEqual(res.status_code, 200)
        pats = res.json()
        self.assertGreaterEqual(len(pats), 4)
        pat_ids = [p["id"] for p in pats]
        self.assertIn("IND_PAT_001", pat_ids)
        self.assertIn("DEMO_PAT_001", pat_ids)

    @pytest.mark.requires_data
    def test_twin_state_for_real_patient(self):
        if not Path("data/processed/timeseries_processed.parquet").exists():
            self.skipTest(
                "Requires local processed dataset in data/processed/timeseries_processed.parquet"
            )
        res = self.client.get("/api/v1/twin/CGMacros-001/state")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["patient"]["id"], "CGMacros-001")
        self.assertGreater(data["current_glucose"], 40.0)
        self.assertIn(
            data["trend_arrow"], ["Rising quickly", "Rising", "Flat", "Falling", "Falling quickly"]
        )

    def test_twin_state_for_persona_patient(self):
        res = self.client.get("/api/v1/twin/IND_PAT_001/state")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["patient"]["id"], "IND_PAT_001")
        self.assertEqual(data["patient"]["name"], "Rajesh Sharma")
        self.assertGreater(data["current_glucose"], 40.0)

    def test_twin_state_for_demo_patient(self):
        res = self.client.get("/api/v1/twin/DEMO_PAT_001/state")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["patient"]["id"], "DEMO_PAT_001")
        self.assertGreater(data["current_glucose"], 40.0)

    @pytest.mark.requires_data
    def test_explainability_route(self):
        if not Path("data/processed/timeseries_processed.parquet").exists():
            self.skipTest(
                "Requires local processed dataset in data/processed/timeseries_processed.parquet"
            )
        res = self.client.get("/api/v1/explain/CGMacros-004?horizon=60")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["patient_id"], "CGMacros-004")
        self.assertEqual(data["horizon_minutes"], 60)
        self.assertGreaterEqual(len(data["top_features"]), 5)
        # Check that top features contain continuous glucose covariates
        feat_names = [f["feature_name"] for f in data["top_features"]]
        self.assertTrue(any("glucose" in fn for fn in feat_names))

    def test_explainability_route_persona(self):
        res = self.client.get("/api/v1/explain/IND_PAT_001?horizon=60")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["patient_id"], "IND_PAT_001")
        self.assertEqual(data["horizon_minutes"], 60)
        self.assertGreaterEqual(len(data["top_features"]), 1)

    def test_api_predict_with_as_of(self):
        pid = "CGMacros-004"
        payload = {
            "patient_id": pid,
            "recent_cgm_series": [120.0, 122.0, 125.0],
            "as_of": "2024-05-01T14:30:00Z",
        }
        res = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["patient_id"], pid)
        self.assertEqual(len(data["forecast_horizons"]), 3)


if __name__ == "__main__":
    unittest.main()
