import unittest

import numpy as np

from src.glucotwin.models.metrics import (
    clarke_error_grid,
    compute_calibration_metrics,
    compute_classification_metrics,
    compute_episode_alert_metrics,
    compute_regression_metrics,
    paired_significance_test,
    tune_operating_threshold,
)


class TestMetrics(unittest.TestCase):
    def test_clarke_error_grid_known_points(self):
        # Exact agreement -> 100% Zone A
        y_true = np.array([70.0, 100.0, 150.0, 200.0, 300.0])
        y_pred = np.array([70.0, 100.0, 150.0, 200.0, 300.0])
        _, percentages = clarke_error_grid(y_true, y_pred)
        self.assertEqual(percentages["zone_a"], 100.0)
        self.assertEqual(percentages["zone_b"], 0.0)
        self.assertEqual(percentages["zone_ab"], 100.0)

        # Severe opposite treatment during hypoglycemia -> Zone E
        y_true_crit = np.array([50.0])
        y_pred_crit = np.array([200.0])
        _, pct_e = clarke_error_grid(y_true_crit, y_pred_crit)
        self.assertEqual(pct_e["zone_e"], 100.0)

        # Failure to detect hyperglycemia (True = 250, Pred = 100) -> Zone D
        y_true_hyper = np.array([250.0])
        y_pred_hyper = np.array([100.0])
        _, pct_d = clarke_error_grid(y_true_hyper, y_pred_hyper)
        self.assertEqual(pct_d["zone_d"], 100.0)

    def test_regression_metrics(self):
        y_true = np.array([100.0, 150.0, 200.0])
        y_pred = np.array([110.0, 140.0, 200.0])
        # Errors: [10, -10, 0] -> MAE = 20/3 = 6.67
        # MSE = (100 + 100 + 0)/3 = 66.6667 -> RMSE = 8.16
        metrics = compute_regression_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["mae"], 6.67, places=2)
        self.assertAlmostEqual(metrics["rmse"], 8.16, places=2)
        self.assertIn("zone_ab", metrics)

    def test_classification_metrics(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.4])
        metrics = compute_classification_metrics(
            y_true, y_prob, threshold=0.5, lead_time_minutes=30
        )
        self.assertIn("auroc", metrics)
        self.assertIn("auprc", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertEqual(metrics["lead_time_mins"], 30.0)
        self.assertGreater(metrics["auroc"], 0.8)

    def test_calibration_metrics(self):
        y_true = np.array([100.0, 120.0, 150.0, 180.0, 210.0])
        q10 = np.array([90.0, 110.0, 140.0, 170.0, 200.0])
        q90 = np.array([110.0, 130.0, 160.0, 190.0, 220.0])
        cal = compute_calibration_metrics(y_true, q10, q90, target_coverage=0.80)
        self.assertEqual(cal["empirical_coverage"], 100.0)
        self.assertEqual(cal["mean_interval_width"], 20.0)

    def test_paired_significance_test(self):
        err_a = np.array([2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 6.0, 5.0, 7.0, 6.0])
        err_b = np.array([10.0, 12.0, 15.0, 14.0, 20.0, 18.0, 22.0, 25.0, 30.0, 28.0])
        res = paired_significance_test(err_a, err_b)
        self.assertTrue(res["is_significant"])
        self.assertLess(res["p_value"], 0.05)
        self.assertLess(res["mean_mae_difference"], 0.0)

    def test_episode_alert_metrics_hand_built_series(self):
        from src.glucotwin.models.metrics import compute_episode_alert_metrics

        pids = np.array(["P1"] * 150)

        # 1. Alert active for hours before crossing: must be a miss
        y1 = np.zeros(150, dtype=int)
        y1[88] = 1  # onset at j=88, horizon=60m -> crossing at 100
        p1 = np.zeros(150)
        p1[40:110] = 0.85  # active for hours before window starts at 76
        res1 = compute_episode_alert_metrics(pids, y1, p1, threshold=0.50, horizon_minutes=60)
        self.assertEqual(res1["total_events"], 1.0)
        self.assertEqual(res1["detected_events"], 0.0)
        self.assertEqual(res1["recall"], 0.0)

        # 2. Alert begins exactly 45m early: detected with lead time = 45m
        y2 = np.zeros(150, dtype=int)
        y2[88] = 1  # crossing at 100
        p2 = np.zeros(150)
        p2[91:105] = 0.85  # starts at step 91 (9 steps = 45m early)
        res2 = compute_episode_alert_metrics(pids, y2, p2, threshold=0.50, horizon_minutes=60)
        self.assertEqual(res2["total_events"], 1.0)
        self.assertEqual(res2["detected_events"], 1.0)
        self.assertEqual(res2["lead_time_median"], 45.0)
        self.assertEqual(res2["share_detected_30m_early"], 100.0)

        # 3. False alert episode: alert with no subsequent event
        y3 = np.zeros(150, dtype=int)
        p3 = np.zeros(150)
        p3[20:30] = 0.85
        res3 = compute_episode_alert_metrics(pids, y3, p3, threshold=0.50, horizon_minutes=60)
        self.assertEqual(res3["total_events"], 0.0)
        self.assertEqual(res3["false_episodes"], 1.0)
        self.assertEqual(res3["true_episodes"], 0.0)
        self.assertEqual(res3["precision"], 0.0)

        # 4. Dip under threshold < 30m does not start new episode
        p4 = np.zeros(150)
        p4[20:30] = 0.85
        p4[30:32] = 0.20  # only 10m below
        p4[32:40] = 0.85
        res4 = compute_episode_alert_metrics(pids, y3, p4, threshold=0.50, horizon_minutes=60)
        self.assertEqual(res4["total_episodes"], 1.0)

    def test_tune_operating_threshold_with_budget(self):
        # 10 days of synthetic data (2880 steps)
        pids = np.zeros(2880, dtype=int)
        y = np.zeros(2880, dtype=int)
        p = np.zeros(2880, dtype=float)

        # Place 4 true events preceded by clean history
        for day in [2, 4, 6, 8]:
            idx = day * 288
            y[idx : idx + 12] = 1
            # Strong alert probability 6 steps (30 min) ahead
            p[idx - 6 : idx] = 0.75

        # Place 3 false alert bursts with probability 0.35 (which would yield 3 false episodes in 10 days = 0.3/day)
        p[300:305] = 0.35
        p[900:905] = 0.35
        p[1500:1505] = 0.35

        # Place 20 noisy false alert bursts with probability 0.15 (yielding 2.0 false alerts/day)
        for i in range(20):
            p[100 + i * 100 : 104 + i * 100] = 0.15

        # If budget B <= 1.0, threshold must be tuned higher than 0.15 to reject the noisy bursts
        th_budget_1 = tune_operating_threshold(
            y_true=y,
            y_prob=p,
            patient_ids=pids,
            horizon_minutes=60,
            max_false_alerts_per_day=1.0,
        )
        self.assertGreater(th_budget_1, 0.15)
        # Verify that under this tuned threshold, false alerts per day is <= 1.0
        m = compute_episode_alert_metrics(pids, y, p, threshold=th_budget_1, horizon_minutes=60)
        self.assertLessEqual(m["false_alerts_per_patient_day"], 1.0)
