import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.glucotwin.serving.schemas.requests import PredictRequest
from src.glucotwin.serving.service import inference_service
from src.glucotwin.twin.state import TwinStateManager
from src.glucotwin.utils.io import as_pandas_df, read_parquet


@pytest.mark.requires_data
class TestTwinStateIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if not Path("data/processed/timeseries_processed.parquet").exists():
            self.skipTest(
                "Requires local processed dataset at data/processed/timeseries_processed.parquet"
            )
        inference_service.load()
        from sqlalchemy import delete

        from src.glucotwin.db.models import (
            PersonalizationAdapter,
            TwinAlertEpisode,
            TwinPrediction,
            TwinWindowReading,
        )
        from src.glucotwin.db.session import AsyncSessionLocal, init_db

        await init_db()
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(TwinWindowReading).where(TwinWindowReading.patient_id == "CGMacros-001")
            )
            await session.execute(
                delete(TwinPrediction).where(TwinPrediction.patient_id == "CGMacros-001")
            )
            await session.execute(
                delete(TwinAlertEpisode).where(TwinAlertEpisode.patient_id == "CGMacros-001")
            )
            await session.execute(
                delete(PersonalizationAdapter).where(
                    PersonalizationAdapter.patient_id == "CGMacros-001"
                )
            )
            await session.commit()

    async def test_patient_replay_and_twin_lifecycle(self):
        # 1. Load real CGMacros patient data
        ts_path = Path("data/processed/timeseries_processed.parquet")
        self.assertTrue(ts_path.exists())
        df_ts = as_pandas_df(read_parquet(ts_path))

        test_patient = "CGMacros-001"
        pat_df = df_ts[df_ts["patient_id"] == test_patient].sort_values("timestamp")
        self.assertGreater(len(pat_df), 500)

        # Take first 4 days (or up to 1152 steps)
        replay_df = pat_df.head(1152).copy()

        manager = TwinStateManager()

        # Ingest 30 steps through twin state manager (covering past + scoring pending)
        step_results = []
        for i in range(30):
            row = replay_df.iloc[i]
            dt = pd.to_datetime(row["timestamp"]).to_pydatetime()
            g = float(row["glucose_mgdl"])
            hr = float(row["heart_rate"]) if pd.notna(row.get("heart_rate")) else 72.0
            st = int(row["steps"]) if pd.notna(row.get("steps")) else 0

            res = await manager.ingest_cgm_reading(
                patient_id=test_patient,
                timestamp=dt,
                glucose_mgdl=g,
                heart_rate_bpm=hr,
                steps=st,
                mets=1.0,
                carbs_g=0.0,
            )
            step_results.append(res)

        # Check latest step forecast
        last_step = step_results[-1]
        forecast = last_step["forecast"]
        self.assertEqual(forecast.patient_id, test_patient)
        self.assertEqual(len(forecast.forecast_horizons), 3)

        # 2. Invariant Check: predictions must equal direct model call on identical 6-hour history
        # Retrieve recent readings matching what was ingested
        recent_cgm = replay_df.iloc[:30]["glucose_mgdl"].values.tolist()
        direct_req = PredictRequest(
            patient_id=test_patient,
            recent_cgm_series=recent_cgm,
            carbs_consumed_g=0.0,
        )
        direct_forecast = inference_service.predict(direct_req)

        for p_twin, p_direct in zip(
            forecast.forecast_horizons, direct_forecast.forecast_horizons, strict=False
        ):
            self.assertEqual(p_twin.horizon_minutes, p_direct.horizon_minutes)
            self.assertAlmostEqual(p_twin.q50, p_direct.q50, delta=1e-3)
            self.assertAlmostEqual(p_twin.q10, p_direct.q10, delta=1e-3)
            self.assertAlmostEqual(p_twin.q90, p_direct.q90, delta=1e-3)

        # 3. Test Personalization Loop:
        # Populate scored predictions covering 3 days (864 steps) plus 1 day eval (288 steps)
        from src.glucotwin.db.models import TwinPrediction
        from src.glucotwin.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            t0 = datetime.utcnow() - timedelta(days=5)
            for idx in range(1152):
                step_dt = t0 + timedelta(minutes=idx * 5)
                sim_true = 120.0 + np.sin(idx / 10.0) * 20.0
                sim_pred = sim_true + (
                    np.random.normal(0, 8.0) if idx < 864 else np.random.normal(0, 15.0)
                )
                p_item = TwinPrediction(
                    patient_id=test_patient,
                    timestamp=step_dt,
                    horizon_minutes=60,
                    target_timestamp=step_dt + timedelta(minutes=60),
                    model_version="gru_v1_fold0",
                    serving_fold=0,
                    q10=sim_pred - 15.0,
                    q50=sim_pred,
                    q90=sim_pred + 15.0,
                    calibrated_lower_95=sim_pred - 20.0,
                    calibrated_upper_95=sim_pred + 20.0,
                    realised_glucose=sim_true,
                    prediction_error=sim_pred - sim_true,
                    absolute_error=abs(sim_pred - sim_true),
                    relative_error_pct=(abs(sim_pred - sim_true) / sim_true) * 100.0,
                )
                session.add(p_item)
            await session.commit()

        # Run personalization training and validation evaluation
        train_res = await manager.train_and_evaluate_personalization(
            test_patient, min_scored_days=3.0
        )
        self.assertEqual(train_res["patient_id"], test_patient)
        self.assertIn("promoted", train_res)
        self.assertTrue(train_res["promoted"])
        self.assertLess(train_res["adapter_rmse"], train_res["baseline_rmse"])
        self.assertLess(train_res["adapter_mard"], train_res["baseline_mard"])

        # Check history API method
        hist = await manager.get_patient_personalization_history(test_patient)
        self.assertEqual(hist["patient_id"], test_patient)
        self.assertGreater(hist["total_scored_samples"], 1000)
        self.assertEqual(hist["adapter_version"], train_res["version"])
        self.assertTrue(hist["is_active"])

        # Check rollback functionality
        rb_res = await manager.rollback_adapter(test_patient)
        self.assertEqual(rb_res["status"], "rolled_back")

        # After rollback, active status should be False
        hist_after = await manager.get_patient_personalization_history(test_patient)
        self.assertFalse(hist_after["is_active"])
