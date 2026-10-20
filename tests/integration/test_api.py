import unittest
from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient

    from src.glucotwin.serving.app import app

    HAS_TESTCLIENT = True
except Exception:
    HAS_TESTCLIENT = False


class TestAPI(unittest.TestCase):
    def setUp(self):
        if HAS_TESTCLIENT:
            self.client = TestClient(app)

    def test_health_route(self):
        if not HAS_TESTCLIENT:
            self.skipTest("fastapi TestClient/httpx not available")
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["project"], "GlucoTwin")

    @pytest.mark.requires_data
    def test_v1_patients_route(self):
        if not HAS_TESTCLIENT:
            self.skipTest("fastapi TestClient/httpx not available")
        if not Path("data/processed/patients_processed.parquet").exists():
            self.skipTest(
                "Requires local processed dataset in data/processed/patients_processed.parquet"
            )
        res = self.client.get("/api/v1/twin/patients")
        self.assertEqual(res.status_code, 200)
        patients = res.json()
        self.assertGreaterEqual(len(patients), 45)

    def test_v1_patients_route_personas_and_demo(self):
        if not HAS_TESTCLIENT:
            self.skipTest("fastapi TestClient/httpx not available")
        res = self.client.get("/api/v1/twin/patients")
        self.assertEqual(res.status_code, 200)
        patients = res.json()
        # Personas and demo patients are always available even without raw/processed data
        self.assertGreaterEqual(len(patients), 4)
        pat_ids = [p["id"] for p in patients]
        self.assertIn("IND_PAT_001", pat_ids)

    def test_v1_twin_state(self):
        if not HAS_TESTCLIENT:
            self.skipTest("fastapi TestClient/httpx not available")
        res = self.client.get("/api/v1/twin/IND_PAT_001/state")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["patient"]["name"], "Rajesh Sharma")

    def test_v1_predict_route(self):
        if not HAS_TESTCLIENT:
            self.skipTest("fastapi TestClient/httpx not available")
        payload = {
            "patient_id": "IND_PAT_001",
            "recent_cgm_series": [130.0, 135.0, 140.0, 145.0, 150.0, 155.0],
            "recent_carbs_g": 45.0,
            "recent_glycemic_index": 65.0,
        }
        res = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["forecast_horizons"]), 3)
        self.assertEqual(data["patient_id"], "IND_PAT_001")

    def test_v1_simulate_route(self):
        if not HAS_TESTCLIENT:
            self.skipTest("fastapi TestClient/httpx not available")
        res_foods = self.client.get("/api/v1/simulate/foods")
        self.assertEqual(res_foods.status_code, 200)
        self.assertGreater(len(res_foods.json()), 0)

        sim_payload = {"patient_id": "IND_PAT_001", "simulation_duration_minutes": 120}
        res_sim = self.client.post("/api/v1/simulate", json=sim_payload)
        self.assertEqual(res_sim.status_code, 200)
        self.assertGreater(len(res_sim.json()["baseline_trajectory"]), 0)

    def test_v1_explain_route(self):
        if not HAS_TESTCLIENT:
            self.skipTest("fastapi TestClient/httpx not available")
        res = self.client.get("/api/v1/explain/IND_PAT_001?horizon=60")
        self.assertEqual(res.status_code, 200)
        self.assertGreater(len(res.json()["top_features"]), 0)

    def test_v1_summary_route(self):
        if not HAS_TESTCLIENT:
            self.skipTest("fastapi TestClient/httpx not available")
        res = self.client.get("/api/v1/summary/IND_PAT_001")
        self.assertEqual(res.status_code, 200)
        self.assertIn("metrics", res.json())

    def test_v1_cgm_route(self):
        if not HAS_TESTCLIENT:
            self.skipTest("fastapi TestClient/httpx not available")
        res = self.client.get("/api/v1/twin/DEMO_PAT_001/cgm?days=1")
        self.assertEqual(res.status_code, 200)
        readings = res.json()
        self.assertGreater(len(readings), 0)
        self.assertEqual(readings[0]["patient_id"], "DEMO_PAT_001")
