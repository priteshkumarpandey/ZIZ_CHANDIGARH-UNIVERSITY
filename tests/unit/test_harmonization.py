import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.glucotwin.data.harmonize import Harmonizer
from src.glucotwin.utils.io import as_pandas_df, read_parquet


class TestHarmonization(unittest.TestCase):
    def test_harmonizer_gap_interpolation_and_outlier_filtering(self):
        harmonizer = Harmonizer(
            resample_interval_minutes=5,
            max_interpolation_minutes=30,
            glucose_min_mgdl=40.0,
            glucose_max_mgdl=500.0,
            heart_rate_min=30.0,
            heart_rate_max=220.0,
        )

        timestamps = pd.date_range("2026-01-01 00:00:00", periods=20, freq="5min", tz="UTC")
        glucose = (
            [100.0, 105.0, np.nan, np.nan, 110.0, 30.0, 550.0, 120.0]
            + [np.nan] * 8
            + [140.0, 142.0, 144.0, 146.0]
        )
        hr = [70.0] * 19 + [250.0]

        df = pd.DataFrame(
            {
                "patient_id": "TEST_PAT_01",
                "timestamp": timestamps,
                "glucose_mgdl": glucose,
                "heart_rate": hr,
                "steps": [0.0] * 20,
                "activity_level": [1.0] * 20,
                "sleep_stage": [None] * 20,
                "hrv": [np.nan] * 20,
                "missing_flag": [False] * 20,
            }
        )

        harm_df, stats = harmonizer.harmonize_timeseries(df)

        self.assertEqual(stats["dropped_glucose_out_of_range"], 2)
        self.assertEqual(stats["dropped_hr_out_of_range"], 1)
        self.assertGreater(stats["interpolated_values_count"], 0)
        self.assertGreaterEqual(stats["gap_count_short_le_30min"], 1)
        self.assertGreaterEqual(stats["gap_count_long_gt_30min"], 1)

        # Short gap (index 2 and 3) should be interpolated
        self.assertTrue(pd.notna(harm_df.loc[2, "glucose_mgdl"]))
        self.assertTrue(pd.notna(harm_df.loc[3, "glucose_mgdl"]))

        # Uninterpolated trailing NaNs in large gap (> 30 mins) retain missing_flag = True
        self.assertTrue(harm_df.loc[14, "missing_flag"])
        self.assertTrue(pd.isna(harm_df.loc[14, "glucose_mgdl"]))

    def test_harmonizer_quality_report(self):
        harmonizer = Harmonizer()
        df_pat = pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "source_dataset": "synth",
                    "is_synthetic": True,
                    "diabetes_type": "T2D",
                    "diabetes_status": "T2D",
                },
                {
                    "patient_id": "P2",
                    "source_dataset": "synth",
                    "is_synthetic": True,
                    "diabetes_type": "prediabetes",
                    "diabetes_status": "prediabetes",
                },
            ]
        )
        df_ts = pd.DataFrame(
            columns=[
                "patient_id",
                "timestamp",
                "glucose_mgdl",
                "heart_rate",
                "steps",
                "activity_level",
                "sleep_stage",
                "hrv",
                "missing_flag",
            ]
        )
        df_meals = pd.DataFrame(
            columns=[
                "patient_id",
                "timestamp",
                "carbs_g",
                "protein_g",
                "fat_g",
                "fiber_g",
                "calories",
                "meal_label",
            ]
        )

        report = harmonizer.generate_quality_report(
            dataset_name="synth",
            df_patients=df_pat,
            df_timeseries=df_ts,
            df_meals=df_meals,
            ts_stats={"raw_rows": 0, "harmonized_rows": 0},
        )

        self.assertEqual(report["dataset_name"], "synth")
        self.assertEqual(report["patient_count"], 2)
        self.assertEqual(report["status_breakdown"]["T2D"], 1)
        self.assertEqual(report["status_breakdown"]["prediabetes"], 1)

    @pytest.mark.requires_data
    def test_sources_do_not_share_identical_timeseries_quality(self):
        report_path = Path("data/processed/data_quality_report.json")
        if not report_path.exists():
            self.skipTest("data_quality_report.json not yet generated")

        with open(report_path) as fp:
            report_data = json.load(fp)

        # Collect non-null timeseries_quality dictionaries
        ts_qualities = {}
        for src, data in report_data.items():
            tq = data.get("timeseries_quality")
            if tq is not None:
                ts_qualities[src] = tq

        # Verify at least two sources with time series exist and their quality stats differ
        self.assertGreaterEqual(len(ts_qualities), 2)
        sources = list(ts_qualities.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                s1, s2 = sources[i], sources[j]
                self.assertNotEqual(
                    ts_qualities[s1],
                    ts_qualities[s2],
                    f"Sources {s1} and {s2} must not share identical timeseries_quality dictionaries",
                )

    @pytest.mark.requires_data
    def test_ai_readi_has_null_timeseries_quality(self):
        report_path = Path("data/processed/data_quality_report.json")
        if not report_path.exists():
            self.skipTest("data_quality_report.json not yet generated")

        with open(report_path) as fp:
            report_data = json.load(fp)

        self.assertIn("ai_readi_synthetic", report_data)
        ai_rep = report_data["ai_readi_synthetic"]
        self.assertIsNone(ai_rep.get("timeseries_quality"))
        self.assertIn("timeseries_quality_note", ai_rep)

    @pytest.mark.requires_data
    def test_hupa_ucm_quality_report_entry(self):
        report_path = Path("data/processed/data_quality_report.json")
        if not report_path.exists():
            self.skipTest("data_quality_report.json not yet generated")

        with open(report_path) as fp:
            report_data = json.load(fp)

        self.assertIn("hupa_ucm", report_data)
        hupa_rep = report_data["hupa_ucm"]
        self.assertEqual(hupa_rep["patient_count"], 25)
        self.assertEqual(hupa_rep["patients_with_windows"], 19)
        self.assertEqual(len(hupa_rep["zero_window_patients_reasons"]), 6)
        self.assertGreater(hupa_rep["skipped_or_empty_files_count"], 0)
        self.assertIsNotNone(hupa_rep["timeseries_quality"])

    @pytest.mark.requires_data
    def test_patient_quality_parquet(self):
        pq_path = Path("data/processed/patient_quality.parquet")
        if not pq_path.exists():
            self.skipTest("patient_quality.parquet not yet generated")

        df = as_pandas_df(read_parquet(pq_path))
        self.assertGreater(len(df), 0)
        required_cols = [
            "patient_id",
            "source_dataset",
            "raw_rows",
            "harmonized_rows",
            "missingness_pct",
            "short_gaps_le_30m",
            "long_gaps_gt_30m",
            "window_count",
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)

        # Confirm all window counts and raw rows are non-negative
        self.assertTrue((df["raw_rows"] >= 0).all())
        self.assertTrue((df["window_count"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
