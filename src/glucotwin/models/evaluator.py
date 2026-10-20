import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.glucotwin.models.baselines import (
    LightGBMForecaster,
    LinearTrendBaseline,
    PersistenceBaseline,
    XGBoostForecaster,
)
from src.glucotwin.models.cv import CrossValidationManager
from src.glucotwin.models.explain import ModelExplainer
from src.glucotwin.models.gru import GRUTrainer, MultimodalDataset, MultimodalGRUForecaster
from src.glucotwin.models.metrics import (
    compute_calibration_metrics,
    compute_classification_metrics,
    compute_patient_level_lead_times,
    compute_regression_metrics,
)
from src.glucotwin.models.personalize import PersonalizationEngine
from src.glucotwin.models.tft import TemporalFusionTransformer, TFTTrainer
from src.glucotwin.models.uncertainty import ConformalCalibrator
from src.glucotwin.utils.io import as_pandas_df, read_parquet, write_parquet
from src.glucotwin.utils.logging import logger
from src.glucotwin.utils.seeds import set_global_seed


class CrossValidationEvaluator:
    def __init__(
        self,
        cv_manager: CrossValidationManager,
        checkpoints_dir: Path = Path("models/checkpoints"),
        reports_dir: Path = Path("reports"),
        device: str | None = None,
        quick_mode: bool = False,
        params_path: Path = Path("params.yaml"),
    ):
        self.cv = cv_manager
        self.checkpoints_dir = Path(checkpoints_dir)
        self.reports_dir = Path(reports_dir)
        self.figures_dir = self.reports_dir / "figures"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quick_mode = quick_mode
        self.params_path = Path(params_path)

        # Default alert configuration
        self.alert_threshold = 0.50
        self.alert_rule = "sustained_threshold"
        self.preceding_window_steps = 24
        self.min_early_minutes = 30.0

        if self.params_path.exists():
            with open(self.params_path) as fp:
                cfg = yaml.safe_load(fp) or {}
            alerts_cfg = cfg.get("alerts", {})
            self.alert_threshold = float(alerts_cfg.get("operating_threshold", 0.50))
            self.alert_rule = str(alerts_cfg.get("rule", "sustained_threshold"))
            preceding_mins = int(alerts_cfg.get("preceding_window_minutes", 120))
            self.preceding_window_steps = max(1, preceding_mins // 5)
            self.min_early_minutes = float(alerts_cfg.get("min_early_minutes", 30.0))

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def run_evaluation(
        self,
        lgb_params: dict[str, Any] | None = None,
        gru_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        horizons = [30, 60, 120]
        model_names = ["persistence", "linear_trend", "lightgbm", "xgboost", "gru", "tft"]

        fold_results: dict[str, list[dict[str, Any]]] = {m: [] for m in model_names}
        all_patient_records: list[dict[str, Any]] = []
        all_h30_records: list[dict[str, Any]] = []

        best_test_actuals: list[float] = []
        best_test_preds: list[float] = []
        best_test_pids: list[str] = []
        sample_attention_matrix: np.ndarray | None = None
        best_lgb_model: LightGBMForecaster | None = None

        personalization_records: list[dict[str, Any]] = []

        num_folds_to_run = 1 if self.quick_mode else len(self.cv.folds)

        for f_idx in range(num_folds_to_run):
            logger.info(f"Running Cross-Validation Fold {f_idx + 1}/{num_folds_to_run}...")
            fold_data = self.cv.get_fold_data(f_idx)

            # 1. Persistence Baseline
            pers = PersistenceBaseline(horizons=horizons)
            # Use raw unnormalized glucose (mg/dL) for continuous evaluation
            cgm_curr_test = fold_data.raw_cgm_curr_test
            preds_pers = pers.predict(cgm_curr_test)

            # 2. Linear Trend Baseline
            lin = LinearTrendBaseline(horizons=horizons)
            # Extract last 6 steps (30 min) of raw unnormalized glucose (mg/dL)
            cgm_hist_test = fold_data.raw_cgm_hist_test
            preds_lin = lin.predict(cgm_hist_test)

            # 3. LightGBM Forecaster
            lgb_model = LightGBMForecaster(
                horizons=horizons,
                quantiles=[0.10, 0.50, 0.90],
                random_state=42 + f_idx,
                **(lgb_params or {}),
            )
            lgb_model.fit(
                fold_data.X_train,
                fold_data.y_train_cont,
                fold_data.events_train,
                fold_data.X_val,
                fold_data.y_val_cont,
                fold_data.events_val,
            )
            preds_lgb_val = lgb_model.predict(fold_data.X_val)
            preds_lgb = lgb_model.predict(fold_data.X_test)
            if f_idx == 0:
                best_lgb_model = lgb_model

            # 4. XGBoost Forecaster
            xgb_model = XGBoostForecaster(
                horizons=horizons,
                random_state=42 + f_idx,
            )
            xgb_model.fit(
                fold_data.X_train,
                fold_data.y_train_cont,
                fold_data.events_train,
                fold_data.X_val,
                fold_data.y_val_cont,
                fold_data.events_val,
            )
            preds_xgb = xgb_model.predict(fold_data.X_test)

            # 5. Multimodal GRU Forecaster
            set_global_seed(42 + f_idx * 100)
            gru_model = MultimodalGRUForecaster(
                time_series_dim=fold_data.seq_train.shape[-1],
                future_dim=fold_data.future_train.shape[-1],
                static_dim=fold_data.static_train.shape[-1],
                hidden_dim=gru_params.get("hidden_dim", 64) if gru_params else 64,
                num_layers=gru_params.get("num_layers", 2) if gru_params else 2,
                dropout=gru_params.get("dropout", 0.1) if gru_params else 0.1,
                horizons=horizons,
            )
            gru_trainer = GRUTrainer(
                gru_model,
                device=self.device,
                lr=gru_params.get("lr", 1e-3) if gru_params else 1e-3,
            )
            train_ds = MultimodalDataset(
                fold_data.seq_train,
                fold_data.future_train,
                fold_data.static_train,
                fold_data.y_train_cont,
                fold_data.events_train,
            )
            tr_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

            epochs = 5 if self.quick_mode else 12
            for _ep in range(epochs):
                gru_trainer.train_epoch(tr_loader)

            preds_gru_val = gru_trainer.predict(
                fold_data.seq_val, fold_data.future_val, fold_data.static_val
            )
            preds_gru = gru_trainer.predict(
                fold_data.seq_test, fold_data.future_test, fold_data.static_test
            )

            # Save GRU checkpoint
            torch.save(gru_model.state_dict(), self.checkpoints_dir / f"gru_fold_{f_idx}.pt")

            # 6. Temporal Fusion Transformer
            # Reset deterministic seed before TFT to ensure complete reproducibility across folds and runs
            set_global_seed(42 + f_idx * 100 + 50)
            tft_model = TemporalFusionTransformer(
                num_dynamic_vars=fold_data.seq_train.shape[-1],
                num_future_vars=fold_data.future_train.shape[-1],
                num_static_vars=fold_data.static_train.shape[-1],
                hidden_dim=48,
                horizons=horizons,
            )
            tft_trainer = TFTTrainer(tft_model, device=self.device, lr=1e-3)
            tr_loader_tft = DataLoader(train_ds, batch_size=256, shuffle=True)
            tft_epochs = 4 if self.quick_mode else 8
            for _ep in range(tft_epochs):
                tft_trainer.train_epoch(tr_loader_tft)

            preds_tft_val, _ = tft_trainer.predict(
                fold_data.seq_val, fold_data.future_val, fold_data.static_val
            )
            preds_tft, attn_map = tft_trainer.predict(
                fold_data.seq_test, fold_data.future_test, fold_data.static_test
            )
            if sample_attention_matrix is None and attn_map is not None:
                sample_attention_matrix = attn_map

            # Conformal Calibration (Fitted strictly on validation fold)
            calibrator_lgb = ConformalCalibrator(target_coverage=0.80)
            calibrator_lgb.fit(
                fold_data.y_val_cont,
                {h: preds_lgb_val["quantiles"][h]["q10"] for h in horizons},
                {h: preds_lgb_val["quantiles"][h]["q90"] for h in horizons},
            )

            calibrator_gru = ConformalCalibrator(target_coverage=0.80)
            calibrator_gru.fit(
                fold_data.y_val_cont,
                {h: preds_gru_val["quantiles"][h]["q10"] for h in horizons},
                {h: preds_gru_val["quantiles"][h]["q90"] for h in horizons},
            )

            # Evaluate each model on test fold
            predictions_map = {
                "persistence": preds_pers,
                "linear_trend": preds_lin,
                "lightgbm": preds_lgb,
                "xgboost": preds_xgb,
                "gru": preds_gru,
                "tft": preds_tft,
            }

            for m_name, p_dict in predictions_map.items():
                m_fold_metrics: dict[str, Any] = {"fold": f_idx}
                for h in horizons:
                    y_true = fold_data.y_test_cont[h]
                    q10 = p_dict["quantiles"][h]["q10"]
                    q50 = p_dict["quantiles"][h]["q50"]
                    q90 = p_dict["quantiles"][h]["q90"]

                    reg_m = compute_regression_metrics(y_true, q50)
                    raw_calib_m = compute_calibration_metrics(
                        y_true, q10, q90, target_coverage=0.80
                    )

                    # Apply conformal calibration
                    calib_obj = calibrator_gru if m_name in ["gru", "tft"] else calibrator_lgb
                    cal_q10, cal_q90 = calib_obj.predict_calibrated_intervals(h, q10, q90)
                    post_calib_m = compute_calibration_metrics(
                        y_true, cal_q10, cal_q90, target_coverage=0.80
                    )

                    # Adverse Event Metrics with Real Lead Times
                    # 1. Hyperglycemia (>180 mg/dL)
                    hyper_true = fold_data.events_test.get(
                        f"hyperglycemia_{h}m", (y_true >= 180).astype(int)
                    )
                    hyper_prob = p_dict["events"].get(
                        f"hyperglycemia_{h}m",
                        p_dict["events"].get(
                            f"hyper_{h}m", 1.0 / (1.0 + np.exp(-(q50 - 180.0) / 15.0))
                        ),
                    )
                    hyper_m = compute_classification_metrics(
                        hyper_true, hyper_prob, threshold=self.alert_threshold
                    )
                    hyper_lt = compute_patient_level_lead_times(
                        fold_data.test_patient_ids,
                        hyper_true,
                        hyper_prob,
                        operating_threshold=self.alert_threshold,
                        preceding_window_steps=self.preceding_window_steps,
                        min_early_minutes=self.min_early_minutes,
                    )

                    # 2. Severe Hyperglycemia (>250 mg/dL)
                    sev_hyper_true = fold_data.events_test.get(
                        f"severe_hyperglycemia_{h}m", (y_true >= 250).astype(int)
                    )
                    sev_hyper_prob = p_dict["events"].get(
                        f"severe_hyperglycemia_{h}m", 1.0 / (1.0 + np.exp(-(q50 - 250.0) / 15.0))
                    )
                    sev_hyper_m = compute_classification_metrics(
                        sev_hyper_true, sev_hyper_prob, threshold=self.alert_threshold
                    )
                    sev_hyper_lt = compute_patient_level_lead_times(
                        fold_data.test_patient_ids,
                        sev_hyper_true,
                        sev_hyper_prob,
                        operating_threshold=self.alert_threshold,
                        preceding_window_steps=self.preceding_window_steps,
                        min_early_minutes=self.min_early_minutes,
                    )

                    # 3. Hypoglycemia (<=70 mg/dL)
                    hypo_true = fold_data.events_test.get(
                        f"hypoglycemia_{h}m", (y_true <= 70).astype(int)
                    )
                    hypo_prob = p_dict["events"].get(
                        f"hypoglycemia_{h}m",
                        p_dict["events"].get(
                            f"hypo_{h}m", 1.0 / (1.0 + np.exp(-(70.0 - q50) / 10.0))
                        ),
                    )
                    hypo_m = compute_classification_metrics(
                        hypo_true, hypo_prob, threshold=self.alert_threshold
                    )
                    hypo_lt = compute_patient_level_lead_times(
                        fold_data.test_patient_ids,
                        hypo_true,
                        hypo_prob,
                        operating_threshold=self.alert_threshold,
                        preceding_window_steps=self.preceding_window_steps,
                        min_early_minutes=self.min_early_minutes,
                    )

                    # 4. Spike excursion
                    spike_true = fold_data.events_test.get(
                        f"spike_{h}m",
                        fold_data.events_test.get("spike_60m", (y_true >= 180).astype(int)),
                    )
                    spike_prob = p_dict["events"].get(
                        f"spike_{h}m",
                        p_dict["events"].get(
                            "spike_60m", 1.0 / (1.0 + np.exp(-(q50 - 180.0) / 15.0))
                        ),
                    )
                    spike_m = compute_classification_metrics(
                        spike_true, spike_prob, threshold=self.alert_threshold
                    )
                    spike_lt = compute_patient_level_lead_times(
                        fold_data.test_patient_ids,
                        spike_true,
                        spike_prob,
                        operating_threshold=self.alert_threshold,
                        preceding_window_steps=self.preceding_window_steps,
                        min_early_minutes=self.min_early_minutes,
                    )

                    m_fold_metrics[f"h{h}"] = {
                        **reg_m,
                        "raw_coverage": raw_calib_m["empirical_coverage"],
                        "calibrated_coverage": post_calib_m["empirical_coverage"],
                        "raw_interval_width": raw_calib_m["mean_interval_width"],
                        "calibrated_interval_width": post_calib_m["mean_interval_width"],
                        "hyper_auroc": hyper_m["auroc"],
                        "hyper_auprc": hyper_m["auprc"],
                        "hyper_precision": hyper_m["precision"],
                        "hyper_recall": hyper_m["recall"],
                        "hyper_lead_time": hyper_lt["lead_time_median"],
                        "hyper_lead_time_median": hyper_lt["lead_time_median"],
                        "hyper_lead_time_iqr": hyper_lt["lead_time_iqr"],
                        "hyper_share_30m_early": hyper_lt["share_detected_30m_early"],
                        "severe_hyper_auroc": sev_hyper_m["auroc"],
                        "severe_hyper_auprc": sev_hyper_m["auprc"],
                        "severe_hyper_precision": sev_hyper_m["precision"],
                        "severe_hyper_recall": sev_hyper_m["recall"],
                        "severe_hyper_lead_time": sev_hyper_lt["lead_time_median"],
                        "severe_hyper_lead_time_median": sev_hyper_lt["lead_time_median"],
                        "severe_hyper_lead_time_iqr": sev_hyper_lt["lead_time_iqr"],
                        "severe_hyper_share_30m_early": sev_hyper_lt["share_detected_30m_early"],
                        "hypo_auroc": hypo_m["auroc"],
                        "hypo_auprc": hypo_m["auprc"],
                        "hypo_precision": hypo_m["precision"],
                        "hypo_recall": hypo_m["recall"],
                        "hypo_lead_time": hypo_lt["lead_time_median"],
                        "hypo_lead_time_median": hypo_lt["lead_time_median"],
                        "hypo_lead_time_iqr": hypo_lt["lead_time_iqr"],
                        "hypo_share_30m_early": hypo_lt["share_detected_30m_early"],
                        "spike_auroc": spike_m["auroc"],
                        "spike_auprc": spike_m["auprc"],
                        "spike_precision": spike_m["precision"],
                        "spike_recall": spike_m["recall"],
                        "spike_lead_time": spike_lt["lead_time_median"],
                        "spike_lead_time_median": spike_lt["lead_time_median"],
                        "spike_lead_time_iqr": spike_lt["lead_time_iqr"],
                        "spike_share_30m_early": spike_lt["share_detected_30m_early"],
                    }

                fold_results[m_name].append(m_fold_metrics)

            # Record +30m test predictions for independent evaluation
            y_true_30 = fold_data.y_test_cont[30]
            p_pers_30 = preds_pers["quantiles"][30]["q50"]
            p_lin_30 = preds_lin["quantiles"][30]["q50"]
            p_lgb_30 = preds_lgb["quantiles"][30]["q50"]
            p_xgb_30 = preds_xgb["quantiles"][30]["q50"]
            p_gru_30 = preds_gru["quantiles"][30]["q50"]
            p_tft_30 = preds_tft["quantiles"][30]["q50"]

            for i in range(len(fold_data.test_patient_ids)):
                all_h30_records.append(
                    {
                        "patient_id": str(fold_data.test_patient_ids[i]),
                        "fold": int(f_idx),
                        "y_true_30m": float(y_true_30[i]),
                        "pred_persistence_30m": float(p_pers_30[i]),
                        "pred_linear_trend_30m": float(p_lin_30[i]),
                        "pred_lightgbm_30m": float(p_lgb_30[i]),
                        "pred_xgboost_30m": float(p_xgb_30[i]),
                        "pred_gru_30m": float(p_gru_30[i]),
                        "pred_tft_30m": float(p_tft_30[i]),
                    }
                )

            # Collect test actuals and predictions for plotting
            best_test_actuals.extend(fold_data.y_test_cont[60].tolist())
            best_test_preds.extend(preds_gru["quantiles"][60]["q50"].tolist())
            best_test_pids.extend(fold_data.test_patient_ids.tolist())

            # Collect per-patient detailed record for reports
            for pid in np.unique(fold_data.test_patient_ids):
                p_mask = fold_data.test_patient_ids == pid
                pid_actual = fold_data.y_test_cont[60][p_mask]
                pid_pred_gru = preds_gru["quantiles"][60]["q50"][p_mask]
                pid_pred_lgb = preds_lgb["quantiles"][60]["q50"][p_mask]
                pid_pred_xgb = preds_xgb["quantiles"][60]["q50"][p_mask]

                p_m_gru = compute_regression_metrics(pid_actual, pid_pred_gru)
                p_m_lgb = compute_regression_metrics(pid_actual, pid_pred_lgb)
                p_m_xgb = compute_regression_metrics(pid_actual, pid_pred_xgb)

                all_patient_records.append(
                    {
                        "patient_id": pid,
                        "fold": f_idx,
                        "num_windows": int(np.sum(p_mask)),
                        "mean_glucose": round(float(np.mean(pid_actual)), 2),
                        "gru_rmse_60m": p_m_gru["rmse"],
                        "gru_mae_60m": p_m_gru["mae"],
                        "gru_mard_60m": p_m_gru["mard"],
                        "gru_zone_a_60m": p_m_gru["zone_a"],
                        "lgb_rmse_60m": p_m_lgb["rmse"],
                        "lgb_mae_60m": p_m_lgb["mae"],
                        "lgb_mard_60m": p_m_lgb["mard"],
                        "xgb_rmse_60m": p_m_xgb["rmse"],
                        "xgb_mae_60m": p_m_xgb["mae"],
                        "xgb_mard_60m": p_m_xgb["mard"],
                    }
                )

            # Run personalization on ALL fold test patients (all 45 cohort participants)
            pers_engine = PersonalizationEngine(gru_model, device=self.device)
            for test_pid in self.cv.folds[f_idx].test_patients:
                p_mask = fold_data.test_patient_ids == test_pid
                if np.sum(p_mask) > 100:
                    pers_res = pers_engine.personalize_patient(
                        test_pid,
                        fold_data.seq_test[p_mask],
                        fold_data.future_test[p_mask],
                        fold_data.static_test[p_mask],
                        {h: fold_data.y_test_cont[h][p_mask] for h in horizons},
                        {k: fold_data.events_test[k][p_mask] for k in fold_data.events_test},
                        days_for_training=3,
                        num_epochs=10,
                    )
                    personalization_records.append(pers_res)

        # Aggregate CV metrics across folds (mean +/- std and per-fold arrays)
        cv_summary: dict[str, Any] = {
            "n_splits": num_folds_to_run,
            "models": {},
        }

        metric_keys = [
            "rmse",
            "mae",
            "mard",
            "zone_a",
            "zone_b",
            "zone_ab",
            "zone_cde",
            "raw_coverage",
            "calibrated_coverage",
            "raw_interval_width",
            "calibrated_interval_width",
            "hyper_auroc",
            "hyper_auprc",
            "hyper_precision",
            "hyper_recall",
            "hyper_lead_time",
            "hyper_lead_time_median",
            "hyper_lead_time_iqr",
            "hyper_share_30m_early",
            "severe_hyper_auroc",
            "severe_hyper_auprc",
            "severe_hyper_precision",
            "severe_hyper_recall",
            "severe_hyper_lead_time",
            "severe_hyper_lead_time_median",
            "severe_hyper_lead_time_iqr",
            "severe_hyper_share_30m_early",
            "hypo_auroc",
            "hypo_auprc",
            "hypo_precision",
            "hypo_recall",
            "hypo_lead_time",
            "hypo_lead_time_median",
            "hypo_lead_time_iqr",
            "hypo_share_30m_early",
            "spike_auroc",
            "spike_auprc",
            "spike_precision",
            "spike_recall",
            "spike_lead_time",
            "spike_lead_time_median",
            "spike_lead_time_iqr",
            "spike_share_30m_early",
        ]

        for m_name in model_names:
            cv_summary["models"][m_name] = {}
            for h in horizons:
                h_key = f"h{h}"
                cv_summary["models"][m_name][h_key] = {}
                for k in metric_keys:
                    values = [
                        f[h_key][k]
                        for f in fold_results[m_name]
                        if not np.isnan(f[h_key].get(k, float("nan")))
                    ]
                    if values:
                        mean_val = float(np.mean(values))
                        std_val = float(np.std(values))
                        cv_summary["models"][m_name][h_key][f"{k}_mean"] = round(mean_val, 2)
                        cv_summary["models"][m_name][h_key][f"{k}_std"] = round(std_val, 2)
                        cv_summary["models"][m_name][h_key][f"{k}_folds"] = [
                            round(float(v), 4) for v in values
                        ]
                    else:
                        cv_summary["models"][m_name][h_key][f"{k}_mean"] = 0.0
                        cv_summary["models"][m_name][h_key][f"{k}_std"] = 0.0
                        cv_summary["models"][m_name][h_key][f"{k}_folds"] = [0.0] * num_folds_to_run

        # Personalization Summary: LoRA vs Simple Bias Correction across all 45 patients
        pers_patient_summary = []
        for r in personalization_records:
            pid = r["patient_id"]
            rmse_before = r["base_metrics"]["rmse"]
            rmse_bias = r["bias_corrected_metrics"]["rmse"]
            rmse_lora = r["personalized_metrics"]["rmse"]

            mard_before = r["base_metrics"]["mard"]
            mard_bias = r["bias_corrected_metrics"]["mard"]
            mard_lora = r["personalized_metrics"]["mard"]

            rmse_gain_lora = round(rmse_before - rmse_lora, 2)
            mard_gain_lora = round(mard_before - mard_lora, 2)
            rmse_gain_bias = round(rmse_before - rmse_bias, 2)
            mard_gain_bias = round(mard_before - mard_bias, 2)

            improved_lora = (rmse_lora < rmse_before) or (mard_lora < mard_before)
            improved_bias = (rmse_bias < rmse_before) or (mard_bias < mard_before)

            pers_patient_summary.append(
                {
                    "patient_id": pid,
                    "rmse_before": rmse_before,
                    "rmse_bias": rmse_bias,
                    "rmse_lora": rmse_lora,
                    "rmse_gain_lora": rmse_gain_lora,
                    "rmse_gain_bias": rmse_gain_bias,
                    "mard_before": mard_before,
                    "mard_bias": mard_bias,
                    "mard_lora": mard_lora,
                    "mard_gain_lora": mard_gain_lora,
                    "mard_gain_bias": mard_gain_bias,
                    "is_improved_lora": bool(improved_lora),
                    "is_improved_bias": bool(improved_bias),
                    # Compatibility fields
                    "rmse_after": rmse_lora,
                    "rmse_gain": rmse_gain_lora,
                    "mard_after": mard_lora,
                    "mard_gain": mard_gain_lora,
                    "is_improved": bool(improved_lora),
                }
            )

        total_p = len(pers_patient_summary)
        lora_imp_cnt = sum(1 for p in pers_patient_summary if p["is_improved_lora"])
        bias_imp_cnt = sum(1 for p in pers_patient_summary if p["is_improved_bias"])

        cv_summary["personalization"] = {
            "forecast_horizon_minutes": 60,
            "total_patients_evaluated": total_p,
            "bias_correction": {
                "improved_patients_count": bias_imp_cnt,
                "improvement_percentage": round(bias_imp_cnt / total_p * 100.0, 1)
                if total_p > 0
                else 0.0,
                "mean_rmse_before": round(
                    float(np.mean([p["rmse_before"] for p in pers_patient_summary])), 2
                )
                if total_p > 0
                else 0.0,
                "mean_rmse_after": round(
                    float(np.mean([p["rmse_bias"] for p in pers_patient_summary])), 2
                )
                if total_p > 0
                else 0.0,
                "mean_rmse_gain": round(
                    float(np.mean([p["rmse_gain_bias"] for p in pers_patient_summary])), 2
                )
                if total_p > 0
                else 0.0,
                "mean_mard_before": round(
                    float(np.mean([p["mard_before"] for p in pers_patient_summary])), 2
                )
                if total_p > 0
                else 0.0,
                "mean_mard_after": round(
                    float(np.mean([p["mard_bias"] for p in pers_patient_summary])), 2
                )
                if total_p > 0
                else 0.0,
                "mean_mard_gain": round(
                    float(np.mean([p["mard_gain_bias"] for p in pers_patient_summary])), 2
                )
                if total_p > 0
                else 0.0,
            },
            "lora_adapters": {
                "improved_patients_count": lora_imp_cnt,
                "improvement_percentage": round(lora_imp_cnt / total_p * 100.0, 1)
                if total_p > 0
                else 0.0,
                "mean_rmse_before": round(
                    float(np.mean([p["rmse_before"] for p in pers_patient_summary])), 2
                )
                if total_p > 0
                else 0.0,
                "mean_rmse_after": round(
                    float(np.mean([p["rmse_lora"] for p in pers_patient_summary])), 2
                )
                if total_p > 0
                else 0.0,
                "mean_rmse_gain": round(
                    float(np.mean([p["rmse_gain_lora"] for p in pers_patient_summary])), 2
                )
                if total_p > 0
                else 0.0,
                "mean_mard_before": round(
                    float(np.mean([p["mard_before"] for p in pers_patient_summary])), 2
                )
                if total_p > 0
                else 0.0,
                "mean_mard_after": round(
                    float(np.mean([p["mard_lora"] for p in pers_patient_summary])), 2
                )
                if total_p > 0
                else 0.0,
                "mean_mard_gain": round(
                    float(np.mean([p["mard_gain_lora"] for p in pers_patient_summary])), 2
                )
                if total_p > 0
                else 0.0,
            },
            "patient_records": pers_patient_summary,
        }
        cv_summary["lora_personalization"] = {
            "forecast_horizon_minutes": 60,
            "total_patients_evaluated": total_p,
            "improved_patients_count": lora_imp_cnt,
            "improvement_percentage": round(lora_imp_cnt / total_p * 100.0, 1)
            if total_p > 0
            else 0.0,
            "mean_rmse_before": cv_summary["personalization"]["lora_adapters"]["mean_rmse_before"],
            "mean_rmse_after": cv_summary["personalization"]["lora_adapters"]["mean_rmse_after"],
            "mean_rmse_gain": cv_summary["personalization"]["lora_adapters"]["mean_rmse_gain"],
            "mean_mard_before": cv_summary["personalization"]["lora_adapters"]["mean_mard_before"],
            "mean_mard_after": cv_summary["personalization"]["lora_adapters"]["mean_mard_after"],
            "mean_mard_gain": cv_summary["personalization"]["lora_adapters"]["mean_mard_gain"],
            "patient_records": pers_patient_summary,
        }

        # Save test predictions (+30m) for independent verification
        df_h30 = pd.DataFrame(all_h30_records)
        write_parquet(df_h30, self.reports_dir / "test_predictions_h30.parquet")

        # Save metrics_cv.json
        with open(self.reports_dir / "metrics_cv.json", "w") as fp:
            json.dump(cv_summary, fp, indent=2)

        # Per-patient results parquet
        df_patient_results = pd.DataFrame(all_patient_records)
        write_parquet(df_patient_results, self.reports_dir / "per_patient_results.parquet")

        # Breakdown by diabetes status (T2D, prediabetes, healthy) for best model (GRU)
        df_patients_meta = as_pandas_df(
            read_parquet(self.cv.processed_dir / "patients_processed.parquet")
        )
        merged_pat = df_patient_results.merge(
            df_patients_meta[["patient_id", "diabetes_status"]], on="patient_id", how="left"
        )
        status_grouped = (
            merged_pat.groupby("diabetes_status")
            .agg(
                {
                    "gru_rmse_60m": ["mean", "std"],
                    "gru_mae_60m": ["mean", "std"],
                    "gru_mard_60m": ["mean", "std"],
                    "gru_zone_a_60m": ["mean", "std"],
                    "patient_id": "count",
                }
            )
            .reset_index()
        )
        status_grouped.columns = [
            "diabetes_status",
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "mae_std",
            "mard_mean",
            "mard_std",
            "zone_a_mean",
            "zone_a_std",
            "patient_count",
        ]
        status_grouped.to_csv(self.reports_dir / "metrics_by_status.csv", index=False)

        # Generate Figures
        self._generate_clarke_scatter_plot(np.array(best_test_actuals), np.array(best_test_preds))
        self._generate_forecast_traces(
            best_test_pids, np.array(best_test_actuals), np.array(best_test_preds)
        )
        self._generate_calibration_plot(cv_summary)
        if personalization_records:
            self._generate_personalization_plot(personalization_records)

        # Explainability plots
        if best_lgb_model is not None:
            explainer = ModelExplainer(best_lgb_model.quantile_models.get("h60_q50"))
            explainer.generate_shap_summary_plot(
                fold_data.X_test, self.figures_dir / "shap_summary.png"
            )

        if sample_attention_matrix is not None:
            explainer = ModelExplainer()
            explainer.generate_attention_heatmap(
                sample_attention_matrix, self.figures_dir / "tft_attention.png"
            )

        return cv_summary

    def _generate_clarke_scatter_plot(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        plt.figure(figsize=(7, 7))
        # Downsample for crisp scatter plot visualization
        idx = np.random.choice(len(y_true), min(2500, len(y_true)), replace=False)
        yt_sample = y_true[idx]
        yp_sample = y_pred[idx]

        plt.scatter(yt_sample, yp_sample, alpha=0.35, s=8, color="#2563eb", label="60m Forecasts")
        plt.plot([0, 500], [0, 500], "k--", alpha=0.7, label="Perfect Agreement")
        plt.plot([0, 500], [0, 600], "g:", alpha=0.5, label="+20% Boundary")
        plt.plot([0, 500], [0, 400], "g:", alpha=0.5, label="-20% Boundary")

        # Clarke Critical Thresholds
        plt.axhline(70, color="r", linestyle="--", alpha=0.3, label="Hypoglycemia (70 mg/dL)")
        plt.axvline(70, color="r", linestyle="--", alpha=0.3)
        plt.axhline(
            180, color="darkorange", linestyle="--", alpha=0.3, label="Hyperglycemia (180 mg/dL)"
        )
        plt.axvline(180, color="darkorange", linestyle="--", alpha=0.3)

        plt.xlim(30, 450)
        plt.ylim(30, 450)
        plt.xlabel("Reference CGM Glucose (mg/dL)", fontsize=10)
        plt.ylabel("Forecasted Glucose (mg/dL)", fontsize=10)
        plt.title("Clarke Error Grid Analysis (60-Minute Horizon)", fontsize=12, fontweight="bold")
        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "clarke_error_grid.png", dpi=200)
        plt.close()

    def _generate_forecast_traces(
        self, pids: list[str], y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        pids_arr = np.array(pids)
        unique_pids, counts = np.unique(pids_arr, return_counts=True)
        if len(unique_pids) < 3:
            return

        # Pick 3 representative test patients: lowest error (good), median error, and highest error
        errors = [
            float(np.mean(np.abs(y_true[pids_arr == p] - y_pred[pids_arr == p])))
            for p in unique_pids
        ]
        sorted_indices = np.argsort(errors)
        good_pid = unique_pids[sorted_indices[0]]
        median_pid = unique_pids[sorted_indices[len(sorted_indices) // 2]]
        challenging_pid = unique_pids[sorted_indices[-1]]

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
        selected_pids = [
            (good_pid, "Good Control / Stable Glycemia", axes[0]),
            (median_pid, "Median Excursions / Moderate Variability", axes[1]),
            (challenging_pid, "High Glycemic Volatility / Postprandial Spikes", axes[2]),
        ]

        for pid, label, ax in selected_pids:
            p_mask = pids_arr == pid
            yt = y_true[p_mask][:288]  # Show 24-hour trace (288 steps)
            yp = y_pred[p_mask][:288]
            time_hours = np.arange(len(yt)) * 5.0 / 60.0

            ax.plot(time_hours, yt, color="#0f172a", label="Actual CGM", linewidth=1.8)
            ax.plot(
                time_hours, yp, color="#2563eb", linestyle="--", label="60m Forecast", linewidth=1.5
            )
            ax.axhspan(70, 180, color="#22c55e", alpha=0.12, label="Target Range (70-180 mg/dL)")

            ax.set_ylabel("Glucose (mg/dL)", fontsize=9)
            ax.set_title(f"Patient {pid} ({label})", fontsize=10, fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.6)
            if ax == axes[0]:
                ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("Time (Hours)", fontsize=10)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "forecast_traces.png", dpi=200)
        plt.close()

    def _generate_calibration_plot(self, cv_summary: dict[str, Any]) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        horizons = [30, 60, 120]
        x = np.arange(len(horizons))
        width = 0.35

        # Coverage comparison
        raw_cov = [cv_summary["models"]["gru"][f"h{h}"]["raw_coverage_mean"] for h in horizons]
        cal_cov = [
            cv_summary["models"]["gru"][f"h{h}"]["calibrated_coverage_mean"] for h in horizons
        ]

        ax1.bar(x - width / 2, raw_cov, width, label="Raw Quantile Coverage", color="#94a3b8")
        ax1.bar(
            x + width / 2, cal_cov, width, label="Conformal Calibrated Coverage", color="#2563eb"
        )
        ax1.axhline(80.0, color="r", linestyle="--", label="Target Coverage (80%)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(["+30 min", "+60 min", "+120 min"])
        ax1.set_ylabel("Empirical Coverage (%)", fontsize=10)
        ax1.set_title("Quantile vs Calibrated Coverage", fontsize=11, fontweight="bold")
        ax1.set_ylim(50, 100)
        ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # Interval width comparison
        raw_w = [cv_summary["models"]["gru"][f"h{h}"]["raw_interval_width_mean"] for h in horizons]
        cal_w = [
            cv_summary["models"]["gru"][f"h{h}"]["calibrated_interval_width_mean"] for h in horizons
        ]

        ax2.bar(x - width / 2, raw_w, width, label="Raw Interval Width", color="#94a3b8")
        ax2.bar(x + width / 2, cal_w, width, label="Calibrated Interval Width", color="#2563eb")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["+30 min", "+60 min", "+120 min"])
        ax2.set_ylabel("Mean Interval Width (mg/dL)", fontsize=10)
        ax2.set_title("Forecast Uncertainty Interval Width", fontsize=11, fontweight="bold")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "calibration_curves.png", dpi=200)
        plt.close()

    def _generate_personalization_plot(self, records: list[dict[str, Any]]) -> None:
        plt.figure(figsize=(8, 4.5))
        # Aggregate day-by-day trajectories
        all_trajs = [r["daily_trajectories"] for r in records if r["daily_trajectories"]]
        if not all_trajs:
            return

        max_len = max(len(t) for t in all_trajs)
        base_mard_by_day: list[list[float]] = [[] for _ in range(max_len)]
        pers_mard_by_day: list[list[float]] = [[] for _ in range(max_len)]

        for t in all_trajs:
            for i, d in enumerate(t):
                base_mard_by_day[i].append(d["base_mard"])
                pers_mard_by_day[i].append(d["pers_mard"])

        days = np.arange(1, max_len + 1) + 3  # fine-tuned on days 1-3
        mean_base = [np.mean(vals) for vals in base_mard_by_day]
        mean_pers = [np.mean(vals) for vals in pers_mard_by_day]

        plt.plot(
            days, mean_base, "o--", color="#64748b", label="Population Base GRU", linewidth=2.0
        )
        plt.plot(
            days,
            mean_pers,
            "s-",
            color="#16a34a",
            label="Personalized GRU (LoRA Adapter)",
            linewidth=2.2,
        )

        plt.xlabel("Patient Monitoring Day (Post-Calibration)", fontsize=10)
        plt.ylabel("MARD (%)", fontsize=10)
        plt.title(
            "Personalization Trajectory: Error Reduction Over Time", fontsize=11, fontweight="bold"
        )
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "personalization_trend.png", dpi=200)
        plt.close()
