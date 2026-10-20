import unittest

import numpy as np
import torch

from src.glucotwin.models.baselines import (
    LightGBMForecaster,
    LinearTrendBaseline,
    PersistenceBaseline,
)
from src.glucotwin.models.gru import MultimodalGRUForecaster, PinballLoss
from src.glucotwin.models.personalize import LoRALinear
from src.glucotwin.models.tft import TemporalFusionTransformer


class TestModels(unittest.TestCase):
    def test_persistence_baseline(self):
        pers = PersistenceBaseline(horizons=[30, 60, 120])
        cgm_curr = np.array([120.0, 150.0, 180.0])
        preds = pers.predict(cgm_curr)
        self.assertIn("quantiles", preds)
        self.assertIn(30, preds["quantiles"])
        self.assertIn(60, preds["quantiles"])
        self.assertIn(120, preds["quantiles"])
        np.testing.assert_array_equal(preds["quantiles"][30]["q50"], cgm_curr)
        # Check that wider horizon has wider uncertainty spread
        spread_30 = preds["quantiles"][30]["q90"] - preds["quantiles"][30]["q10"]
        spread_120 = preds["quantiles"][120]["q90"] - preds["quantiles"][120]["q10"]
        self.assertTrue(np.all(spread_120 > spread_30))

    def test_persistence_exact_synthetic_error(self):
        pers = PersistenceBaseline(horizons=[30])
        # Synthetic CGM readings at prediction time t0
        g0 = np.array([100.0, 120.0, 140.0, 160.0])
        # True future glucose at t0 + 30m with constant offset of 10.0 mg/dL
        g30_true = np.array([110.0, 130.0, 150.0, 170.0])
        preds = pers.predict(g0)
        q50 = preds["quantiles"][30]["q50"]

        from src.glucotwin.models.metrics import compute_regression_metrics

        metrics = compute_regression_metrics(g30_true, q50)

        # Error is exactly 10.0 mg/dL for every point: RMSE=10, MAE=10, Zone A=100%
        self.assertAlmostEqual(metrics["mae"], 10.0, places=2)
        self.assertAlmostEqual(metrics["rmse"], 10.0, places=2)
        expected_mard = float(np.mean(10.0 / g30_true) * 100.0)
        self.assertAlmostEqual(metrics["mard"], round(expected_mard, 2), places=2)
        self.assertEqual(metrics["zone_a"], 100.0)

    def test_linear_trend_baseline(self):
        lin = LinearTrendBaseline(horizons=[30, 60, 120])
        cgm_hist = np.array([[100.0, 102.0, 104.0, 106.0, 108.0, 110.0]])
        preds = lin.predict(cgm_hist)
        self.assertAlmostEqual(preds["quantiles"][30]["q50"][0], 122.0, places=1)

    def test_gru_model_forward_and_loss(self):
        model = MultimodalGRUForecaster(
            time_series_dim=11,
            future_dim=6,
            static_dim=16,
            hidden_dim=32,
            num_layers=1,
            horizons=[30, 60, 120],
        )
        x_past = torch.randn(4, 72, 11)
        x_fut = torch.randn(4, 24, 6)
        x_stat = torch.randn(4, 16)

        quantiles, events = model(x_past, x_fut, x_stat)
        self.assertIn(30, quantiles)
        self.assertEqual(quantiles[30].shape, (4, 3))
        self.assertIn("spike_60m", events)

        # Check pinball loss calculation
        loss_fn = PinballLoss(quantiles=[0.10, 0.50, 0.90])
        target = torch.tensor([120.0, 130.0, 140.0, 150.0])
        loss = loss_fn(quantiles[30], target)
        self.assertGreater(loss.item(), 0.0)

    def test_tft_model_forward_and_attention(self):
        model = TemporalFusionTransformer(
            num_dynamic_vars=11,
            num_future_vars=6,
            num_static_vars=16,
            hidden_dim=32,
            horizons=[30, 60, 120],
        )
        x_past = torch.randn(4, 72, 11)
        x_fut = torch.randn(4, 24, 6)
        x_stat = torch.randn(4, 16)

        quantiles, events, interp = model(x_past, x_fut, x_stat)
        self.assertIn(60, quantiles)
        self.assertEqual(quantiles[60].shape, (4, 3))
        self.assertIn("temporal_attention", interp)
        self.assertEqual(interp["temporal_attention"].shape[0], 4)

    def test_lora_adapter(self):
        base_layer = torch.nn.Linear(32, 16)
        lora_layer = LoRALinear(base_layer, rank=4, alpha=8.0)

        # Base weights must be frozen, only lora_a and lora_b trainable
        self.assertFalse(lora_layer.base_linear.weight.requires_grad)
        self.assertTrue(lora_layer.lora_a.requires_grad)
        self.assertTrue(lora_layer.lora_b.requires_grad)

        x = torch.randn(2, 32)
        out = lora_layer(x)
        self.assertEqual(out.shape, (2, 16))

    def test_lightgbm_fit_predict(self):
        np.random.seed(42)
        X = np.random.randn(50, 20)
        y = {
            30: np.random.randn(50) * 20 + 130,
            60: np.random.randn(50) * 20 + 130,
            120: np.random.randn(50) * 20 + 130,
        }
        events = {"spike_60m": (y[60] > 140).astype(int), "hypo_30m": (y[30] < 100).astype(int)}

        lgb_model = LightGBMForecaster(horizons=[30], n_estimators=5)
        lgb_model.fit(X, y, events)
        preds = lgb_model.predict(X)
        self.assertIn(30, preds["quantiles"])
        self.assertEqual(len(preds["quantiles"][30]["q50"]), 50)
