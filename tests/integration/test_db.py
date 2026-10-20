import unittest
from datetime import datetime, timedelta

from sqlalchemy import select

from src.glucotwin.db.models import (
    PatientRecord,
    PersonalizationAdapter,
    TwinAlertEpisode,
    TwinPrediction,
    TwinWindowReading,
)
from src.glucotwin.db.session import AsyncSessionLocal, init_db


class TestDatabaseIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await init_db()
        from sqlalchemy import delete

        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(TwinPrediction).where(TwinPrediction.patient_id == "TEST_PAT_001")
            )
            await session.execute(
                delete(TwinWindowReading).where(TwinWindowReading.patient_id == "TEST_PAT_001")
            )
            await session.execute(
                delete(TwinAlertEpisode).where(TwinAlertEpisode.patient_id == "TEST_PAT_001")
            )
            await session.execute(
                delete(PersonalizationAdapter).where(
                    PersonalizationAdapter.patient_id == "TEST_PAT_001"
                )
            )
            await session.execute(delete(PatientRecord).where(PatientRecord.id == "TEST_PAT_001"))
            await session.commit()

    async def test_full_database_lifecycle(self):
        async with AsyncSessionLocal() as session:
            # 1. Insert Patient Record with static EHR vector
            patient = PatientRecord(
                id="TEST_PAT_001",
                name="Aarav Patel",
                age=48,
                gender="male",
                region="West India",
                dietary_habit="Vegetarian",
                bmi=26.8,
                hba1c=7.4,
                diabetes_duration_years=4.0,
                egfr=92.0,
                static_ehr_vector={"age": 48, "bmi": 26.8, "hba1c": 7.4, "t2d": 1},
                active_medications=["Metformin 500mg"],
            )
            session.add(patient)
            await session.commit()

            # 2. Insert CGM Observation / TwinWindowReading
            now = datetime.utcnow()
            reading = TwinWindowReading(
                patient_id="TEST_PAT_001",
                timestamp=now,
                glucose_mgdl=135.5,
                heart_rate_bpm=72.0,
                steps=120,
                mets=1.2,
                carbs_g=0.0,
            )
            session.add(reading)

            # 3. Insert Twin Prediction
            pred = TwinPrediction(
                patient_id="TEST_PAT_001",
                timestamp=now,
                horizon_minutes=60,
                target_timestamp=now + timedelta(minutes=60),
                model_version="gru_v1_fold0",
                serving_fold=0,
                q10=120.0,
                q50=140.0,
                q90=165.0,
                calibrated_lower_95=112.0,
                calibrated_upper_95=172.0,
                realised_glucose=142.0,
                prediction_error=2.0,
                absolute_error=2.0,
                relative_error_pct=1.41,
            )
            session.add(pred)

            # 4. Insert Twin Alert Episode
            episode = TwinAlertEpisode(
                patient_id="TEST_PAT_001",
                event_type="hyperglycemia",
                horizon_minutes=60,
                start_timestamp=now,
                end_timestamp=None,
                operating_threshold=0.35,
                trigger_probability=0.78,
                is_active=True,
                was_true_alert=True,
                lead_time_minutes=25.0,
            )
            session.add(episode)

            # 5. Insert Personalization Adapter
            adapter = PersonalizationAdapter(
                patient_id="TEST_PAT_001",
                version=1,
                is_active=True,
                train_window_start=now - timedelta(days=3),
                train_window_end=now - timedelta(days=1),
                eval_window_start=now - timedelta(days=1),
                eval_window_end=now,
                baseline_rmse=24.5,
                baseline_mard=14.2,
                adapter_rmse=21.1,
                adapter_mard=12.0,
                promoted=True,
                adapter_weights={"lora_rank": 4, "alpha": 8},
            )
            session.add(adapter)
            await session.commit()

            # Query back and verify integrity
            queried_pat = await session.get(PatientRecord, "TEST_PAT_001")
            self.assertIsNotNone(queried_pat)
            self.assertEqual(queried_pat.name, "Aarav Patel")
            self.assertEqual(queried_pat.static_ehr_vector["hba1c"], 7.4)

            pred_stmt = select(TwinPrediction).where(TwinPrediction.patient_id == "TEST_PAT_001")
            res = await session.execute(pred_stmt)
            preds = res.scalars().all()
            self.assertEqual(len(preds), 1)
            self.assertEqual(preds[0].horizon_minutes, 60)
            self.assertEqual(preds[0].realised_glucose, 142.0)

            ep_stmt = select(TwinAlertEpisode).where(TwinAlertEpisode.patient_id == "TEST_PAT_001")
            res_ep = await session.execute(ep_stmt)
            episodes = res_ep.scalars().all()
            self.assertEqual(len(episodes), 1)
            self.assertTrue(episodes[0].is_active)

            ad_stmt = select(PersonalizationAdapter).where(
                PersonalizationAdapter.patient_id == "TEST_PAT_001"
            )
            res_ad = await session.execute(ad_stmt)
            adapters = res_ad.scalars().all()
            self.assertEqual(len(adapters), 1)
            self.assertTrue(adapters[0].promoted)
            self.assertLess(adapters[0].adapter_rmse, adapters[0].baseline_rmse)

    async def test_double_migration(self):
        # Verify that running migrations twice runs cleanly and idempotently
        from alembic import command
        from alembic.config import Config

        alembic_cfg = Config("src/glucotwin/db/alembic/alembic.ini")
        # Run upgrade head twice
        command.upgrade(alembic_cfg, "head")
        command.upgrade(alembic_cfg, "head")
        # Ensure database tables remain intact
        async with AsyncSessionLocal() as session:
            stmt = select(PatientRecord)
            res = await session.execute(stmt)
            self.assertIsNotNone(res)
