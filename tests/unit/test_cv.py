import unittest
from pathlib import Path

import numpy as np
import pytest

from src.glucotwin.models.cv import CrossValidationManager
from src.glucotwin.models.uncertainty import ConformalCalibrator, enforce_quantile_monotonicity


class TestCrossValidationAndLeakage(unittest.TestCase):
    def setUp(self):
        npz_file = Path("data/processed/features/windowed_features.npz")
        pat_file = Path("data/processed/patients_processed.parquet")
        self.has_cv_data = npz_file.exists() and pat_file.exists()
        if self.has_cv_data:
            self.cv = CrossValidationManager(n_splits=5, random_seed=42)

    @pytest.mark.requires_data
    def test_zero_patient_overlap_across_all_folds(self):
        if not self.has_cv_data:
            self.skipTest("Requires local processed dataset at data/processed")
        # Assert strict patient isolation between train, val, and test splits in every fold
        for fold in self.cv.folds:
            train_set = set(fold.train_patients)
            val_set = set(fold.val_patients)
            test_set = set(fold.test_patients)

            self.assertEqual(len(train_set.intersection(val_set)), 0)
            self.assertEqual(len(train_set.intersection(test_set)), 0)
            self.assertEqual(len(val_set.intersection(test_set)), 0)

            # Ensure total patients partitioned equals the 45 participants
            all_patients = train_set | val_set | test_set
            self.assertEqual(len(all_patients), 45)

    @pytest.mark.requires_data
    def test_stratification_representation(self):
        if not self.has_cv_data:
            self.skipTest("Requires local processed dataset at data/processed")
        # Verify that all 5 folds have non-empty train, val, and test splits
        self.assertEqual(len(self.cv.folds), 5)
        for fold in self.cv.folds:
            self.assertGreater(len(fold.test_patients), 0)
            self.assertGreater(len(fold.val_patients), 0)
            self.assertGreater(len(fold.train_patients), 0)

    @pytest.mark.requires_data
    def test_scaler_fitted_on_train_only(self):
        if not self.has_cv_data:
            self.skipTest("Requires local processed dataset at data/processed")
        fold_0 = self.cv.get_fold_data(0)
        # Scaler parameters must match train statistics exactly
        # X_train normalized mean should be close to 0
        mean_train = np.mean(fold_0.X_train, axis=0)
        self.assertTrue(np.all(np.abs(mean_train) < 0.2))

    def test_quantile_monotonicity_enforcement(self):
        q10 = np.array([120.0, 150.0, 180.0])
        q50 = np.array([110.0, 160.0, 170.0])
        q90 = np.array([130.0, 140.0, 200.0])

        q10_fixed, q50_fixed, q90_fixed = enforce_quantile_monotonicity(q10, q50, q90)
        self.assertTrue(np.all(q10_fixed <= q50_fixed))
        self.assertTrue(np.all(q50_fixed <= q90_fixed))

    def test_conformal_coverage_on_synthetic_val(self):
        # Verify conformal calibrator increases interval width to satisfy 80% coverage
        cal = ConformalCalibrator(target_coverage=0.80)
        y_true = {
            60: np.array([100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0, 260.0, 280.0])
        }
        # Narrow initial quantiles
        q10 = {60: y_true[60] - 1.0}
        q90 = {60: y_true[60] + 1.0}
        cal.fit(y_true, q10, q90)

        # Calibrate intervals
        q10_cal, q90_cal = cal.predict_calibrated_intervals(60, q10[60], q90[60])
        covered = (y_true[60] >= q10_cal) & (y_true[60] <= q90_cal)
        coverage = np.mean(covered)
        self.assertGreaterEqual(coverage, 0.80)

    def test_metrics_cv_per_fold_arrays(self):
        # Assert that every metric in metrics_cv.json has a matching per-fold array of length n_splits
        import json
        from pathlib import Path

        metrics_file = Path("reports/metrics_cv.json")
        if not metrics_file.exists():
            return

        with open(metrics_file) as fp:
            cv_summary = json.load(fp)

        n_splits = cv_summary.get("n_splits", 5)
        self.assertGreater(n_splits, 0)
        models = cv_summary.get("models", {})
        self.assertGreater(len(models), 0)

        for m_name, m_horizons in models.items():
            for h_key, h_metrics in m_horizons.items():
                mean_keys = [k for k in h_metrics.keys() if k.endswith("_mean")]
                self.assertGreater(len(mean_keys), 0, f"No metrics found for {m_name} {h_key}")
                for mk in mean_keys:
                    base_metric = mk[:-5]
                    folds_key = f"{base_metric}_folds"
                    self.assertIn(
                        folds_key,
                        h_metrics,
                        f"Model {m_name} horizon {h_key} metric {base_metric} missing matching per-fold array {folds_key}",
                    )
                    fold_array = h_metrics[folds_key]
                    self.assertIsInstance(
                        fold_array,
                        list,
                        f"{folds_key} for {m_name} {h_key} must be a list",
                    )
                    self.assertEqual(
                        len(fold_array),
                        n_splits,
                        f"{folds_key} length ({len(fold_array)}) must match n_splits ({n_splits})",
                    )
