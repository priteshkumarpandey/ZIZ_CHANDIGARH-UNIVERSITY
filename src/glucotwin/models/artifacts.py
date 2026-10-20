import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
import yaml

from src.glucotwin.models.baselines import LightGBMForecaster
from src.glucotwin.models.cv import CrossValidationManager
from src.glucotwin.models.gru import GRUTrainer, MultimodalDataset, MultimodalGRUForecaster
from src.glucotwin.models.metrics import (
    compute_operating_point_metrics,
    tune_operating_threshold,
)
from src.glucotwin.models.uncertainty import ConformalCalibrator
from src.glucotwin.utils.logging import logger


class ExportableGRU(nn.Module):
    def __init__(self, base_model: MultimodalGRUForecaster):
        super().__init__()
        self.model = base_model

    def forward(
        self,
        x_past: torch.Tensor,
        x_future: torch.Tensor,
        x_static: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        q, ev = self.model(x_past, x_future, x_static)
        return {
            "h30": q[30],
            "h60": q[60],
            "h120": q[120],
            "spike_60m": ev["spike_60m"],
            "hypo_30m": ev["hypo_30m"],
            "hypo_60m": ev["hypo_60m"],
            "hypo_120m": ev["hypo_120m"],
            "hyper_60m": ev["hyper_60m"],
        }


def get_git_commit_hash() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return commit
    except Exception:
        return "unknown"


def build_and_save_serving_artifacts(
    checkpoints_dir: Path = Path("models/checkpoints"),
    reports_dir: Path = Path("reports"),
    params_path: Path = Path("params.yaml"),
    device: str | None = None,
) -> dict[str, Any]:
    checkpoints_dir = Path(checkpoints_dir)
    reports_dir = Path(reports_dir)
    params_path = Path(params_path)

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    cfg: dict[str, Any] = {}
    if params_path.exists():
        with open(params_path) as fp:
            cfg = yaml.safe_load(fp) or {}

    alerts_cfg = cfg.get("alerts", {})
    target_sensitivity = float(alerts_cfg.get("target_sensitivity", 0.80))
    false_alert_budget = float(alerts_cfg.get("false_alert_budget", 1.0))
    preceding_mins = int(alerts_cfg.get("preceding_window_minutes", 120))
    preceding_window_steps = max(1, preceding_mins // 5)
    min_early_minutes = float(alerts_cfg.get("min_early_minutes", 30.0))

    cv_manager = CrossValidationManager()
    n_splits = len(cv_manager.folds)
    horizons = [30, 60, 120]
    target_events = ["hyperglycemia", "severe_hyperglycemia", "hypoglycemia", "spike"]

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Building serving artifacts across {n_splits} folds on {dev}...")

    patient_to_test_fold: dict[str, int] = {}
    folds_manifest: dict[str, Any] = {}
    per_fold_operating_results: list[dict[str, Any]] = []
    cached_test_data: list[dict[int, Any]] = []

    for f_idx in range(n_splits):
        logger.info(f"Processing Fold {f_idx + 1}/{n_splits} artifacts...")
        fold = cv_manager.folds[f_idx]
        for pid in fold.test_patients:
            patient_to_test_fold[pid] = f_idx

        fold_data = cv_manager.get_fold_data(f_idx)

        # 1. Load GRU Checkpoint
        gru_pt_path = checkpoints_dir / f"gru_fold_{f_idx}.pt"
        gru_model = MultimodalGRUForecaster(
            time_series_dim=fold_data.seq_train.shape[-1],
            future_dim=fold_data.future_train.shape[-1],
            static_dim=fold_data.static_train.shape[-1],
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
            horizons=horizons,
        )

        if gru_pt_path.exists():
            gru_model.load_state_dict(torch.load(gru_pt_path, map_location="cpu"))
        else:
            logger.warning(f"GRU checkpoint {gru_pt_path} missing; re-training fold {f_idx}...")
            gru_trainer = GRUTrainer(gru_model, device=dev, lr=1e-3)
            train_ds = MultimodalDataset(
                fold_data.seq_train,
                fold_data.future_train,
                fold_data.static_train,
                fold_data.y_train_cont,
                fold_data.events_train,
            )
            tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
            for _ in range(12):
                gru_trainer.train_epoch(tr_loader)
            torch.save(gru_model.state_dict(), gru_pt_path)

        gru_model.eval()

        # 2. Export and Verify TorchScript GRU
        exportable = ExportableGRU(gru_model)
        exportable.eval()

        dummy_past = torch.randn(1, 72, fold_data.seq_train.shape[-1])
        dummy_fut = torch.randn(1, 24, fold_data.future_train.shape[-1])
        dummy_stat = torch.randn(1, fold_data.static_train.shape[-1])

        traced_gru = torch.jit.trace(exportable, (dummy_past, dummy_fut, dummy_stat), strict=False)

        # Numerical tolerance check between PyTorch and TorchScript
        py_out = exportable(dummy_past, dummy_fut, dummy_stat)
        ts_out = traced_gru(dummy_past, dummy_fut, dummy_stat)
        max_diff = max(float((py_out[k] - ts_out[k]).abs().max()) for k in py_out)
        assert max_diff < 1e-4, f"Fold {f_idx} TorchScript output discrepancy: {max_diff}"

        ts_path = checkpoints_dir / f"gru_fold_{f_idx}.torchscript.pt"
        torch.jit.save(traced_gru, ts_path)

        # 3. Fit and Save LightGBM Forecaster for this fold
        lgb_path = checkpoints_dir / f"lightgbm_fold_{f_idx}.joblib"
        lgb_model = LightGBMForecaster(
            horizons=horizons,
            quantiles=[0.10, 0.50, 0.90],
            random_state=42 + f_idx,
        )
        lgb_model.fit(
            fold_data.X_train,
            fold_data.y_train_cont,
            fold_data.events_train,
            fold_data.X_val,
            fold_data.y_val_cont,
            fold_data.events_val,
        )
        joblib.dump(lgb_model, lgb_path)

        # Save fold scalers
        scalers_path = checkpoints_dir / f"scalers_fold_{f_idx}.joblib"
        joblib.dump(
            {
                "tabular_scaler": fold_data.tabular_scaler,
                "static_scaler": fold_data.static_scaler,
                "seq_mean": fold_data.seq_mean,
                "seq_std": fold_data.seq_std,
            },
            scalers_path,
        )

        # 4. Generate Validation and Test Predictions for GRU and LightGBM
        gru_trainer = GRUTrainer(gru_model, device=dev)
        preds_gru_val = gru_trainer.predict(
            fold_data.seq_val, fold_data.future_val, fold_data.static_val
        )
        preds_gru_test = gru_trainer.predict(
            fold_data.seq_test, fold_data.future_test, fold_data.static_test
        )

        preds_lgb_val = lgb_model.predict(fold_data.X_val)

        # 5. Conformal Calibrators (fitted strictly on validation set)
        calibrator_gru = ConformalCalibrator(target_coverage=0.80)
        calibrator_gru.fit(
            fold_data.y_val_cont,
            {h: preds_gru_val["quantiles"][h]["q10"] for h in horizons},
            {h: preds_gru_val["quantiles"][h]["q90"] for h in horizons},
        )

        calibrator_lgb = ConformalCalibrator(target_coverage=0.80)
        calibrator_lgb.fit(
            fold_data.y_val_cont,
            {h: preds_lgb_val["quantiles"][h]["q10"] for h in horizons},
            {h: preds_lgb_val["quantiles"][h]["q90"] for h in horizons},
        )

        calib_dict = {
            "gru": calibrator_gru.to_dict(),
            "lightgbm": calibrator_lgb.to_dict(),
        }
        with open(checkpoints_dir / f"calibrator_fold_{f_idx}.json", "w") as fp:
            json.dump(calib_dict, fp, indent=2)

        # 6. Tune Operating Thresholds on Validation Patients
        # Targets: reach target_sensitivity (0.80) on validation fold
        fold_operating_points: dict[str, dict[str, Any]] = {}
        fold_eval_metrics: dict[str, dict[str, Any]] = {}

        for h in horizons:
            y_val_h = fold_data.y_val_cont[h]
            y_test_h = fold_data.y_test_cont[h]
            q50_val = preds_gru_val["quantiles"][h]["q50"]
            q50_test = preds_gru_test["quantiles"][h]["q50"]

            # Ground truth binary event vectors
            events_dict_val = {
                "hyperglycemia": fold_data.events_val.get(
                    f"hyperglycemia_{h}m", (y_val_h >= 180.0).astype(int)
                ),
                "severe_hyperglycemia": fold_data.events_val.get(
                    f"severe_hyperglycemia_{h}m", (y_val_h >= 250.0).astype(int)
                ),
                "hypoglycemia": fold_data.events_val.get(
                    f"hypoglycemia_{h}m", (y_val_h <= 70.0).astype(int)
                ),
                "spike": fold_data.events_val.get(
                    f"spike_{h}m",
                    fold_data.events_val.get("spike_60m", (y_val_h >= 180.0).astype(int)),
                ),
            }
            events_dict_test = {
                "hyperglycemia": fold_data.events_test.get(
                    f"hyperglycemia_{h}m", (y_test_h >= 180.0).astype(int)
                ),
                "severe_hyperglycemia": fold_data.events_test.get(
                    f"severe_hyperglycemia_{h}m", (y_test_h >= 250.0).astype(int)
                ),
                "hypoglycemia": fold_data.events_test.get(
                    f"hypoglycemia_{h}m", (y_test_h <= 70.0).astype(int)
                ),
                "spike": fold_data.events_test.get(
                    f"spike_{h}m",
                    fold_data.events_test.get("spike_60m", (y_test_h >= 180.0).astype(int)),
                ),
            }

            # Probability predictions
            probs_val = {
                "hyperglycemia": preds_gru_val["events"].get(
                    f"hyper_{h}m", 1.0 / (1.0 + np.exp(-(q50_val - 180.0) / 15.0))
                ),
                "severe_hyperglycemia": preds_gru_val["events"].get(
                    f"severe_hyper_{h}m", 1.0 / (1.0 + np.exp(-(q50_val - 250.0) / 15.0))
                ),
                "hypoglycemia": preds_gru_val["events"].get(
                    f"hypo_{h}m", 1.0 / (1.0 + np.exp(-(70.0 - q50_val) / 10.0))
                ),
                "spike": preds_gru_val["events"].get(
                    f"spike_{h}m",
                    preds_gru_val["events"].get(
                        "spike_60m", 1.0 / (1.0 + np.exp(-(q50_val - 180.0) / 15.0))
                    ),
                ),
            }
            probs_test = {
                "hyperglycemia": preds_gru_test["events"].get(
                    f"hyper_{h}m", 1.0 / (1.0 + np.exp(-(q50_test - 180.0) / 15.0))
                ),
                "severe_hyperglycemia": preds_gru_test["events"].get(
                    f"severe_hyper_{h}m", 1.0 / (1.0 + np.exp(-(q50_test - 250.0) / 15.0))
                ),
                "hypoglycemia": preds_gru_test["events"].get(
                    f"hypo_{h}m", 1.0 / (1.0 + np.exp(-(70.0 - q50_test) / 10.0))
                ),
                "spike": preds_gru_test["events"].get(
                    f"spike_{h}m",
                    preds_gru_test["events"].get(
                        "spike_60m", 1.0 / (1.0 + np.exp(-(q50_test - 180.0) / 15.0))
                    ),
                ),
            }

            for ev in target_events:
                key = f"{ev}_{h}m"
                th = tune_operating_threshold(
                    y_true=events_dict_val[ev],
                    y_prob=probs_val[ev],
                    patient_ids=fold_data.val_patient_ids,
                    horizon_minutes=h,
                    target_sensitivity=target_sensitivity,
                    max_false_alerts_per_day=false_alert_budget,
                )
                fold_operating_points[key] = {
                    "event": ev,
                    "horizon_minutes": h,
                    "threshold": th,
                    "target_sensitivity": target_sensitivity,
                    "false_alert_budget": false_alert_budget,
                }

                # Evaluate on test fold using the tuned threshold
                test_op_metrics = compute_operating_point_metrics(
                    patient_ids=fold_data.test_patient_ids,
                    y_true=events_dict_test[ev],
                    y_prob=probs_test[ev],
                    threshold=th,
                    horizon_minutes=h,
                    preceding_window_steps=preceding_window_steps,
                    min_early_minutes=min_early_minutes,
                )
                fold_eval_metrics[key] = test_op_metrics

        per_fold_operating_results.append(
            {
                "fold_idx": f_idx,
                "operating_points": fold_operating_points,
                "test_metrics": fold_eval_metrics,
            }
        )

        # Cache fold test evaluation components for operating curves
        fold_test_ev_cache = {}
        for h in horizons:
            y_test_h = fold_data.y_test_cont[h]
            q50_test = preds_gru_test["quantiles"][h]["q50"]
            fold_test_ev_cache[h] = {
                "events_dict_test": {
                    "hyperglycemia": fold_data.events_test.get(
                        f"hyperglycemia_{h}m", (y_test_h >= 180.0).astype(int)
                    ),
                    "severe_hyperglycemia": fold_data.events_test.get(
                        f"severe_hyperglycemia_{h}m", (y_test_h >= 250.0).astype(int)
                    ),
                    "hypoglycemia": fold_data.events_test.get(
                        f"hypoglycemia_{h}m", (y_test_h <= 70.0).astype(int)
                    ),
                    "spike": fold_data.events_test.get(
                        f"spike_{h}m",
                        fold_data.events_test.get("spike_60m", (y_test_h >= 180.0).astype(int)),
                    ),
                },
                "probs_test": {
                    "hyperglycemia": preds_gru_test["events"].get(
                        f"hyper_{h}m", 1.0 / (1.0 + np.exp(-(q50_test - 180.0) / 15.0))
                    ),
                    "severe_hyperglycemia": preds_gru_test["events"].get(
                        f"severe_hyper_{h}m", 1.0 / (1.0 + np.exp(-(q50_test - 250.0) / 15.0))
                    ),
                    "hypoglycemia": preds_gru_test["events"].get(
                        f"hypo_{h}m", 1.0 / (1.0 + np.exp(-(70.0 - q50_test) / 10.0))
                    ),
                    "spike": preds_gru_test["events"].get(
                        f"spike_{h}m",
                        preds_gru_test["events"].get(
                            "spike_60m", 1.0 / (1.0 + np.exp(-(q50_test - 180.0) / 15.0))
                        ),
                    ),
                },
                "patient_ids": fold_data.test_patient_ids,
            }
        cached_test_data.append(fold_test_ev_cache)

        with open(checkpoints_dir / f"operating_thresholds_fold_{f_idx}.json", "w") as fp:
            json.dump(
                {
                    "operating_points": fold_operating_points,
                    "test_metrics": fold_eval_metrics,
                },
                fp,
                indent=2,
            )

        folds_manifest[f"fold_{f_idx}"] = {
            "fold_idx": f_idx,
            "train_patients": fold.train_patients,
            "val_patients": fold.val_patients,
            "test_patients": fold.test_patients,
            "train_windows": fold.train_windows,
            "val_windows": fold.val_windows,
            "test_windows": fold.test_windows,
            "torchscript_path": str(ts_path.relative_to(checkpoints_dir.parent.parent)),
            "lightgbm_path": str(lgb_path.relative_to(checkpoints_dir.parent.parent)),
            "scalers_path": str(scalers_path.relative_to(checkpoints_dir.parent.parent)),
            "calibrator_path": f"models/checkpoints/calibrator_fold_{f_idx}.json",
            "operating_thresholds_path": f"models/checkpoints/operating_thresholds_fold_{f_idx}.json",
        }

    # Aggregate operating point metrics across folds (mean +/- std)
    summary_operating_points: dict[str, Any] = {
        "target_sensitivity": target_sensitivity,
        "false_alert_budget": false_alert_budget,
        "events": {},
    }

    all_keys = list(per_fold_operating_results[0]["test_metrics"].keys())
    metric_fields = [
        "operating_threshold",
        "precision",
        "recall",
        "false_alerts_per_patient_day",
        "lead_time_median",
        "lead_time_iqr",
        "share_detected_30m_early",
        "total_events",
        "detected_events",
        "total_episodes",
    ]

    for k in all_keys:
        summary_operating_points["events"][k] = {}
        events_per_fold = [
            int(f["test_metrics"][k]["total_events"]) for f in per_fold_operating_results
        ]
        summary_operating_points["events"][k]["events_per_fold"] = events_per_fold
        for mf in metric_fields:
            vals = [f["test_metrics"][k][mf] for f in per_fold_operating_results]
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            summary_operating_points["events"][k][f"{mf}_mean"] = round(mean_v, 2)
            summary_operating_points["events"][k][f"{mf}_std"] = round(std_v, 2)
            summary_operating_points["events"][k][f"{mf}_folds"] = [
                round(float(v), 4) for v in vals
            ]

        # Tag events with < 10 test events in any fold as low reliability
        # Also mark hypoglycemia as low reliability if mean precision < 0.20
        prec_mean = summary_operating_points["events"][k]["precision_mean"]
        rec_mean = summary_operating_points["events"][k]["recall_mean"]
        min_events = min(events_per_fold)
        is_hypo = "hypoglycemia" in k
        low_count = min_events < 10
        low_prec = is_hypo and (prec_mean < 0.20)

        is_reliable = not (low_count or low_prec)
        unreliable_reasons = []
        if low_count:
            unreliable_reasons.append(f"event count per fold < 10 (min={min_events})")
        if low_prec:
            unreliable_reasons.append(f"hypoglycemia precision {prec_mean} < 0.20")

        # Alert Usefulness Gate: true only if mean test recall >= 0.40, precision >= 0.30, and is_reliable is true
        alert_usable = bool(is_reliable and rec_mean >= 0.40 and prec_mean >= 0.30)
        unusable_reasons = []
        if not is_reliable:
            unusable_reasons.append("unreliable event")
        if rec_mean < 0.40:
            unusable_reasons.append(f"recall {rec_mean} < 0.40")
        if prec_mean < 0.30:
            unusable_reasons.append(f"precision {prec_mean} < 0.30")

        summary_operating_points["events"][k]["is_reliable"] = is_reliable
        summary_operating_points["events"][k]["reliability_notes"] = (
            "; ".join(unreliable_reasons) if unreliable_reasons else "Reliable"
        )
        summary_operating_points["events"][k]["alert_usable"] = alert_usable
        summary_operating_points["events"][k]["alert_usability_notes"] = (
            "Usable alert" if alert_usable else "; ".join(unusable_reasons)
        )

    # Save operating_thresholds.json
    with open(checkpoints_dir / "operating_thresholds.json", "w") as fp:
        json.dump(summary_operating_points, fp, indent=2)
    with open(reports_dir / "operating_thresholds.json", "w") as fp:
        json.dump(summary_operating_points, fp, indent=2)

    # Compute and save operating curves across all folds
    operating_curves: dict[str, Any] = {}
    curve_candidates = [round(x * 0.05, 2) for x in range(1, 19)]
    for ev in target_events:
        for h in horizons:
            k = f"{ev}_{h}m"
            curve_points: list[dict[str, Any]] = []
            for th in curve_candidates:
                th_metrics = []
                for f_idx in range(n_splits):
                    c_data = cached_test_data[f_idx][h]
                    m = compute_operating_point_metrics(
                        patient_ids=c_data["patient_ids"],
                        y_true=c_data["events_dict_test"][ev],
                        y_prob=c_data["probs_test"][ev],
                        threshold=th,
                        horizon_minutes=h,
                    )
                    th_metrics.append(m)
                avg_rec = round(float(np.mean([m["recall"] for m in th_metrics])), 4)
                avg_prec = round(float(np.mean([m["precision"] for m in th_metrics])), 4)
                avg_fa = round(
                    float(np.mean([m["false_alerts_per_patient_day"] for m in th_metrics])), 2
                )
                curve_points.append(
                    {
                        "threshold": th,
                        "recall": avg_rec,
                        "precision": avg_prec,
                        "false_alerts_per_patient_day": avg_fa,
                    }
                )
            operating_curves[k] = curve_points

    with open(reports_dir / "operating_curves.json", "w") as fp:
        json.dump(operating_curves, fp, indent=2)

    # Plot operating curves
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for idx, ev in enumerate(target_events):
            ax = axes[idx // 2, idx % 2]
            for h in horizons:
                k = f"{ev}_{h}m"
                c_pts = operating_curves.get(k, [])
                recalls = [p["recall"] for p in c_pts]
                fa_rates = [p["false_alerts_per_patient_day"] for p in c_pts]
                ax.plot(fa_rates, recalls, marker="o", label=f"+{h} min")
            ax.axvline(
                false_alert_budget,
                color="red",
                linestyle="--",
                label=f"Budget (B={false_alert_budget}/day)",
            )
            ax.set_title(f"Operating Curve: {ev.replace('_', ' ').capitalize()}")
            ax.set_xlabel("False Alerts / Patient-Day")
            ax.set_ylabel("Recall")
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "operating_curves.png", dpi=300)
        plt.close()
        logger.info(
            f"Operating curves saved to {reports_dir / 'operating_curves.json'} and {figures_dir / 'operating_curves.png'}"
        )
    except Exception as exc:
        logger.warning(f"Could not render operating curve plot: {exc}")

    # Master Manifest
    git_hash = get_git_commit_hash()
    manifest_data = {
        "model_version": "v1.0.0-serving",
        "git_commit": git_hash,
        "total_participants": len(patient_to_test_fold),
        "target_sensitivity": target_sensitivity,
        "false_alert_budget": false_alert_budget,
        "patient_to_test_fold": patient_to_test_fold,
        "folds": folds_manifest,
        "operating_summary": summary_operating_points,
    }

    with open(checkpoints_dir / "manifest.json", "w") as fp:
        json.dump(manifest_data, fp, indent=2)

    logger.info(f"Serving artifacts and manifest generated successfully. Git commit: {git_hash}")
    return manifest_data
