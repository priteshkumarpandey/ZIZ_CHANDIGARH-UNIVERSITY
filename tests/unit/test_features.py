import unittest

import numpy as np
import pandas as pd
import pytest

from src.glucotwin.features.windowing import FeatureWindowBuilder


class TestFeatures(unittest.TestCase):
    def test_feature_window_builder(self):
        builder = FeatureWindowBuilder(
            lookback_minutes=60,
            horizons_minutes=[30, 60],
            resample_interval_minutes=5,
        )

        n_steps = 100
        timestamps = pd.date_range("2026-01-01 00:00:00", periods=n_steps, freq="5min", tz="UTC")
        glucose = np.linspace(100, 200, n_steps)
        glucose[50:65] = np.linspace(120, 220, 15)

        df_pat = pd.DataFrame(
            [
                {
                    "patient_id": "PAT_001",
                    "source_dataset": "synth",
                    "is_synthetic": True,
                    "diabetes_type": "T2D",
                    "diabetes_status": "T2D",
                    "age": 55.0,
                    "sex": "M",
                    "bmi": 28.5,
                    "hba1c": 7.8,
                    "fasting_glucose": 140.0,
                }
            ]
        )

        df_ts = pd.DataFrame(
            {
                "patient_id": "PAT_001",
                "timestamp": timestamps,
                "glucose_mgdl": glucose,
                "heart_rate": 75.0,
                "steps": 10.0,
                "activity_level": 1.2,
                "sleep_stage": None,
                "hrv": np.nan,
                "missing_flag": False,
            }
        )

        df_meals = pd.DataFrame(
            [
                {
                    "patient_id": "PAT_001",
                    "timestamp": timestamps[30],
                    "carbs_g": 60.0,
                    "protein_g": 20.0,
                    "fat_g": 15.0,
                    "fiber_g": 5.0,
                    "calories": 450.0,
                    "meal_label": "Lunch",
                }
            ]
        )

        dataset = builder.build_dataset(df_pat, df_ts, df_meals)

        self.assertGreater(len(dataset), 0)
        self.assertEqual(dataset.observed_sequences.shape[1], 12)
        self.assertEqual(dataset.observed_sequences.shape[2], len(builder.observed_cols))
        self.assertEqual(dataset.known_future_sequences.shape[1], 12)
        self.assertEqual(dataset.static_features.shape[1], len(builder.static_cols))
        self.assertIn(30, dataset.continuous_targets)
        self.assertIn(60, dataset.continuous_targets)
        self.assertIn("hyperglycemia_30m", dataset.binary_event_labels)
        self.assertIn("spike_60m", dataset.binary_event_labels)
        self.assertTrue((dataset.binary_event_labels["spike_60m"] == 1).any())

    @pytest.mark.requires_data
    def test_real_windows_target_alignment(self):
        # Verify target alignment on real processed windows: glucose exactly +30, +60, +120 min after t0
        from pathlib import Path

        npz_path = Path("data/processed/features/windowed_features.npz")
        ts_path = Path("data/processed/timeseries_processed.parquet")
        if not npz_path.exists() or not ts_path.exists():
            self.skipTest("Feature artifacts not found")

        from src.glucotwin.utils.io import as_pandas_df, read_parquet

        npz = np.load(npz_path, allow_pickle=True)
        ts_df = as_pandas_df(read_parquet(ts_path))
        ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"], utc=True)

        np.random.seed(42)
        n_sample = 100
        sample_indices = np.random.choice(len(npz["window_timestamps"]), n_sample, replace=False)

        # Build indexed dictionary of patient timestamps for fast lookup
        patient_subsets = {}
        for idx in sample_indices:
            pid = str(npz["patient_ids"][idx])
            if pid not in patient_subsets:
                p_df = ts_df[ts_df["patient_id"] == pid].set_index("timestamp")
                patient_subsets[pid] = p_df["glucose_mgdl"]

        matches_30 = 0
        matches_60 = 0
        matches_120 = 0

        for idx in sample_indices:
            pid = str(npz["patient_ids"][idx])
            t0 = pd.to_datetime(npz["window_timestamps"][idx], utc=True)
            p_glu = patient_subsets[pid]

            t_30 = t0 + pd.Timedelta(minutes=30)
            t_60 = t0 + pd.Timedelta(minutes=60)
            t_120 = t0 + pd.Timedelta(minutes=120)

            if t_30 in p_glu.index and np.isclose(p_glu.loc[t_30], npz["target_30m"][idx]):
                matches_30 += 1
            if t_60 in p_glu.index and np.isclose(p_glu.loc[t_60], npz["target_60m"][idx]):
                matches_60 += 1
            if t_120 in p_glu.index and np.isclose(p_glu.loc[t_120], npz["target_120m"][idx]):
                matches_120 += 1

        self.assertEqual(matches_30, n_sample)
        self.assertEqual(matches_60, n_sample)
        self.assertEqual(matches_120, n_sample)

    @pytest.mark.requires_data
    def test_no_feature_leakage(self):
        # Confirm that no future glucose or target variables appear in features
        import json
        from pathlib import Path

        meta_path = Path("data/processed/features/feature_metadata.json")
        if not meta_path.exists():
            self.skipTest("feature_metadata.json not found")

        with open(meta_path) as fp:
            meta = json.load(fp)

        tabular = meta["tabular_features"]
        known_future = meta["known_future_features"]
        static = meta["static_features"]

        # 1. Tabular features must not contain targets or future values
        for f in tabular:
            self.assertNotIn("target", f.lower())
            self.assertNotIn("future", f.lower())
            self.assertNotIn("lead", f.lower())

        # 2. Known future features must contain zero glucose columns
        for f in known_future:
            self.assertNotIn("glucose", f.lower())
            self.assertNotIn("target", f.lower())

        # 3. Static features must contain only demographics and baseline labs
        for f in static:
            self.assertNotIn("target", f.lower())
            self.assertNotIn("window", f.lower())


if __name__ == "__main__":
    unittest.main()
