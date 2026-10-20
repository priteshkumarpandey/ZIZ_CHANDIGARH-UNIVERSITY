import unittest
from pathlib import Path

import pytest

from src.glucotwin.data.adapters.ai_readi_synthetic import AIREADISyntheticAdapter
from src.glucotwin.data.adapters.cgmacros import CGMacrosAdapter
from src.glucotwin.data.adapters.hupa_ucm import HUPAAdapter

RAW_DIR = Path("data/raw")


def _has_raw_data() -> bool:
    if not RAW_DIR.exists():
        return False
    # Check if there is at least one subfolder with data
    subdirs = [d for d in RAW_DIR.iterdir() if d.is_dir() and d.name != ".git"]
    return len(subdirs) > 0 and any(any(d.iterdir()) for d in subdirs)


@pytest.mark.requires_data
class TestAdapters(unittest.TestCase):
    def setUp(self):
        if not _has_raw_data():
            self.skipTest("Requires local raw clinical dataset in data/raw (missing dataset files)")

    @pytest.mark.requires_data
    def test_cgmacros_adapter_patients(self):

        adapter = CGMacrosAdapter()
        df_pat = adapter.load_patients(RAW_DIR)
        self.assertFalse(df_pat.empty)
        self.assertIn("patient_id", df_pat.columns)
        self.assertIn("hba1c", df_pat.columns)
        self.assertTrue((df_pat["source_dataset"] == "cgmacros").all())
        self.assertTrue((~df_pat["is_synthetic"]).all())
        self.assertTrue(
            set(df_pat["diabetes_status"].unique()).issubset(
                {"T2D", "prediabetes", "healthy", "unknown"}
            )
        )

    @pytest.mark.requires_data
    def test_cgmacros_adapter_timeseries_and_meals(self):
        adapter = CGMacrosAdapter()
        df_ts = adapter.load_timeseries(RAW_DIR, patient_ids=["CGMacros-001"])
        self.assertFalse(df_ts.empty)
        self.assertIn("glucose_mgdl", df_ts.columns)
        self.assertIn("timestamp", df_ts.columns)
        self.assertEqual(df_ts["patient_id"].iloc[0], "CGMacros-001")

        df_meals = adapter.load_meals(RAW_DIR, patient_ids=["CGMacros-001"])
        self.assertFalse(df_meals.empty)
        self.assertIn("carbs_g", df_meals.columns)
        self.assertTrue((df_meals["carbs_g"] >= 0).all())

    @pytest.mark.requires_data
    def test_cgmacros_validation_glucose(self):
        adapter = CGMacrosAdapter()
        df_val = adapter.load_validation_glucose(RAW_DIR)
        self.assertFalse(df_val.empty)
        self.assertIn("fingerstick_glucose_mgdl", df_val.columns)
        self.assertIn("measurement_number", df_val.columns)

    @pytest.mark.requires_data
    def test_ai_readi_synthetic_adapter(self):
        adapter = AIREADISyntheticAdapter()
        df_pat = adapter.load_patients(RAW_DIR)
        self.assertFalse(df_pat.empty)
        self.assertTrue(df_pat["is_synthetic"].all())
        self.assertTrue((df_pat["source_dataset"] == "ai_readi_synthetic").all())
        self.assertIn("bmi", df_pat.columns)
        self.assertIn("systolic_bp", df_pat.columns)

        # AI-READI dynamic timeseries is empty by design as it contains cross-sectional summaries
        df_ts = adapter.load_timeseries(RAW_DIR, patient_ids=["AIREADI-00000"])
        self.assertTrue(df_ts.empty)
        self.assertIn("glucose_mgdl", df_ts.columns)

        df_summary = adapter.load_cross_sectional_summary(RAW_DIR, patient_ids=["AIREADI-00000"])
        self.assertFalse(df_summary.empty)
        self.assertEqual(df_summary["patient_id"].iloc[0], "AIREADI-00000")

        df_meals = adapter.load_meals(RAW_DIR)
        self.assertTrue(df_meals.empty)
        self.assertIn("carbs_g", df_meals.columns)

    @pytest.mark.requires_data
    def test_hupa_adapter_empty_skipping_and_carb_conversion(self):
        adapter = HUPAAdapter()
        df_pat = adapter.load_patients(RAW_DIR)
        self.assertFalse(df_pat.empty)
        self.assertTrue((df_pat["diabetes_type"] == "T1D").all())
        self.assertTrue((df_pat["source_dataset"] == "hupa_ucm").all())

        df_meals = adapter.load_meals(RAW_DIR, patient_ids=["HUPA0003P"])
        self.assertFalse(df_meals.empty)
        self.assertTrue((df_meals["carbs_g"] > 0).all())


if __name__ == "__main__":
    unittest.main()
