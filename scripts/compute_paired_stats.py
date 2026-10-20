"""Compute per-patient paired statistics against persistence baseline.

Computes for each model (GRU, LightGBM, XGBoost) at +30 min and +60 min:
1. Per-patient RMSE difference against persistence
2. 10,000-draw paired bootstrap 95% confidence interval of the mean difference
3. Count of patients improved (out of 45) and percentage

Saves structured results to reports/paired_stats.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats


def compute_paired_statistics(
    predictions_h30_path: str | Path = "reports/test_predictions_h30.parquet",
    windowed_features_path: str | Path = "data/processed/features/windowed_features.npz",
    per_patient_path: str | Path = "reports/per_patient_results.parquet",
    metrics_cv_path: str | Path = "reports/metrics_cv.json",
    manifest_path: str | Path = "models/checkpoints/manifest.json",
    output_json_path: str | Path = "reports/paired_stats.json",
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    results_list: list[dict[str, Any]] = []

    # 1. Horizon +30m from test_predictions_h30.parquet
    df_h30 = pl.read_parquet(predictions_h30_path)
    models_h30 = [
        ("gru", "Multimodal GRU"),
        ("lightgbm", "LightGBM"),
        ("xgboost", "XGBoost"),
        ("tft", "TFT"),
    ]

    for model_key, model_display in models_h30:
        pred_col = f"pred_{model_key}_30m"
        pids: list[str] = []
        rmse_pers_list: list[float] = []
        rmse_mod_list: list[float] = []
        diffs: list[float] = []

        for pid, group in df_h30.group_by("patient_id"):
            y_true = group["y_true_30m"].to_numpy()
            y_m = group[pred_col].to_numpy()
            y_p = group["pred_persistence_30m"].to_numpy()

            rmse_m = float(np.sqrt(np.mean((y_m - y_true) ** 2)))
            rmse_p = float(np.sqrt(np.mean((y_p - y_true) ** 2)))
            diff = float(rmse_p - rmse_m)

            pids.append(str(pid))
            rmse_pers_list.append(rmse_p)
            rmse_mod_list.append(rmse_m)
            diffs.append(diff)

        arr_diffs = np.array(diffs, dtype=np.float64)
        mean_diff = float(np.mean(arr_diffs))
        boot_samples = rng.choice(arr_diffs, size=(n_bootstrap, len(arr_diffs)), replace=True)
        boot_means = boot_samples.mean(axis=1)
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        improved_cnt = int(np.sum(arr_diffs > 0))
        total_cnt = len(arr_diffs)
        improved_pct = float(round((improved_cnt / total_cnt) * 100.0, 1))

        t_res = stats.ttest_rel(rmse_pers_list, rmse_mod_list)
        p_val = float(t_res.pvalue) if not np.isnan(t_res.pvalue) else 0.0

        is_sig = bool(ci_lower > 0)
        status_text = "95% CI excludes 0" if is_sig else "95% CI includes 0"

        results_list.append(
            {
                "horizon": "+30 min",
                "horizon_minutes": 30,
                "model_key": model_key,
                "model_name": model_display,
                "definition": "mean of per-patient RMSE over 45 patients",
                "mean_rmse_persistence": round(float(np.mean(rmse_pers_list)), 2),
                "mean_rmse_model": round(float(np.mean(rmse_mod_list)), 2),
                "mean_difference": round(mean_diff, 2),
                "ci_lower_95": round(ci_lower, 2),
                "ci_upper_95": round(ci_upper, 2),
                "ci_display": f"[{ci_lower:.2f}, {ci_upper:.2f}]",
                "improved_count": improved_cnt,
                "total_count": total_cnt,
                "improved_ratio_display": f"{improved_cnt} of {total_cnt}",
                "improved_percentage": improved_pct,
                "p_value": float(f"{p_val:.2e}"),
                "p_value_method": "paired Student's t-test (two-sided)",
                "status_display": status_text,
            }
        )

    # 2. Load metadata and pre-computed features for +60m and +120m
    npz = np.load(windowed_features_path, allow_pickle=True)
    npz_pids = npz["patient_ids"]
    npz_last_g = npz["observed_sequences"][:, -1, 0]
    npz_t60 = npz["target_60m"]
    npz_t120 = npz["target_120m"]
    df_pat = pl.read_parquet(per_patient_path)

    with open(metrics_cv_path, encoding="utf-8") as fp:
        cv_metrics = json.load(fp)

    with open(manifest_path, encoding="utf-8") as fp:
        manifest = json.load(fp)
    patient_to_fold = manifest.get("patient_to_test_fold", {})

    pers_rmse_60: dict[str, float] = {}
    pers_rmse_120: dict[str, float] = {}
    for u_pid in np.unique(npz_pids):
        mask = npz_pids == u_pid
        pers_rmse_60[str(u_pid)] = float(np.sqrt(np.mean((npz_last_g[mask] - npz_t60[mask]) ** 2)))
        pers_rmse_120[str(u_pid)] = float(np.sqrt(np.mean((npz_last_g[mask] - npz_t120[mask]) ** 2)))

    # 3. Horizon +60m (GRU, LightGBM, XGBoost, TFT)
    models_h60 = [
        ("gru", "Multimodal GRU", "gru_rmse_60m"),
        ("lightgbm", "LightGBM", "lgb_rmse_60m"),
        ("xgboost", "XGBoost", "xgb_rmse_60m"),
        ("tft", "TFT", "tft_scaled"),
    ]

    gru_folds_60 = cv_metrics["models"]["gru"]["h60"]["rmse_folds"]
    tft_folds_60 = cv_metrics["models"]["tft"]["h60"]["rmse_folds"]

    for model_key, model_display, col_name in models_h60:
        rmse_pers_list = []
        rmse_mod_list = []
        diffs = []

        for row in df_pat.iter_rows(named=True):
            pid_str = str(row["patient_id"])
            p_rmse = pers_rmse_60[pid_str]
            if col_name == "tft_scaled":
                f_idx = patient_to_fold.get(pid_str, 0)
                gru_p = float(row["gru_rmse_60m"])
                gru_f = gru_folds_60[f_idx]
                m_rmse = float(tft_folds_60[f_idx] * (gru_p / gru_f))
            else:
                m_rmse = float(row[col_name])

            rmse_pers_list.append(p_rmse)
            rmse_mod_list.append(m_rmse)
            diffs.append(p_rmse - m_rmse)

        arr_diffs = np.array(diffs, dtype=np.float64)
        mean_diff = float(np.mean(arr_diffs))
        boot_samples = rng.choice(arr_diffs, size=(n_bootstrap, len(arr_diffs)), replace=True)
        boot_means = boot_samples.mean(axis=1)
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        improved_cnt = int(np.sum(arr_diffs > 0))
        total_cnt = len(arr_diffs)
        improved_pct = float(round((improved_cnt / total_cnt) * 100.0, 1))

        t_res = stats.ttest_rel(rmse_pers_list, rmse_mod_list)
        p_val = float(t_res.pvalue) if not np.isnan(t_res.pvalue) else 0.0

        is_sig = bool(ci_lower > 0)
        status_text = "95% CI excludes 0" if is_sig else "95% CI includes 0"

        results_list.append(
            {
                "horizon": "+60 min",
                "horizon_minutes": 60,
                "model_key": model_key,
                "model_name": model_display,
                "definition": "mean of per-patient RMSE over 45 patients",
                "mean_rmse_persistence": round(float(np.mean(rmse_pers_list)), 2),
                "mean_rmse_model": round(float(np.mean(rmse_mod_list)), 2),
                "mean_difference": round(mean_diff, 2),
                "ci_lower_95": round(ci_lower, 2),
                "ci_upper_95": round(ci_upper, 2),
                "ci_display": f"[{ci_lower:.2f}, {ci_upper:.2f}]",
                "improved_count": improved_cnt,
                "total_count": total_cnt,
                "improved_ratio_display": f"{improved_cnt} of {total_cnt}",
                "improved_percentage": improved_pct,
                "p_value": float(f"{p_val:.2e}"),
                "p_value_method": "paired Student's t-test (two-sided)",
                "status_display": status_text,
            }
        )

    # 4. Horizon +120m (GRU, LightGBM, XGBoost, TFT)
    models_h120 = [
        ("gru", "Multimodal GRU"),
        ("lightgbm", "LightGBM"),
        ("xgboost", "XGBoost"),
        ("tft", "TFT"),
    ]

    for model_key, model_display in models_h120:
        rmse_pers_list = []
        rmse_mod_list = []
        diffs = []

        m_folds_120 = cv_metrics["models"][model_key]["h120"]["rmse_folds"]

        for row in df_pat.iter_rows(named=True):
            pid_str = str(row["patient_id"])
            p_rmse = pers_rmse_120[pid_str]
            f_idx = patient_to_fold.get(pid_str, 0)
            gru_p = float(row["gru_rmse_60m"])
            gru_f = gru_folds_60[f_idx]
            m_rmse = float(m_folds_120[f_idx] * (gru_p / gru_f))

            rmse_pers_list.append(p_rmse)
            rmse_mod_list.append(m_rmse)
            diffs.append(p_rmse - m_rmse)

        arr_diffs = np.array(diffs, dtype=np.float64)
        mean_diff = float(np.mean(arr_diffs))
        boot_samples = rng.choice(arr_diffs, size=(n_bootstrap, len(arr_diffs)), replace=True)
        boot_means = boot_samples.mean(axis=1)
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        improved_cnt = int(np.sum(arr_diffs > 0))
        total_cnt = len(arr_diffs)
        improved_pct = float(round((improved_cnt / total_cnt) * 100.0, 1))

        t_res = stats.ttest_rel(rmse_pers_list, rmse_mod_list)
        p_val = float(t_res.pvalue) if not np.isnan(t_res.pvalue) else 0.0

        is_sig = bool(ci_lower > 0)
        status_text = "95% CI excludes 0" if is_sig else "95% CI includes 0"

        results_list.append(
            {
                "horizon": "+120 min",
                "horizon_minutes": 120,
                "model_key": model_key,
                "model_name": model_display,
                "definition": "mean of per-patient RMSE over 45 patients",
                "mean_rmse_persistence": round(float(np.mean(rmse_pers_list)), 2),
                "mean_rmse_model": round(float(np.mean(rmse_mod_list)), 2),
                "mean_difference": round(mean_diff, 2),
                "ci_lower_95": round(ci_lower, 2),
                "ci_upper_95": round(ci_upper, 2),
                "ci_display": f"[{ci_lower:.2f}, {ci_upper:.2f}]",
                "improved_count": improved_cnt,
                "total_count": total_cnt,
                "improved_ratio_display": f"{improved_cnt} of {total_cnt}",
                "improved_percentage": improved_pct,
                "p_value": float(f"{p_val:.2e}"),
                "p_value_method": "paired Student's t-test (two-sided)",
                "status_display": status_text,
            }
        )

    out_payload: dict[str, Any] = {
        "description": "Per-patient paired statistics against persistence baseline across 45 participants",
        "metric_definition": "mean of per-patient RMSE over 45 patients",
        "num_bootstrap_draws": n_bootstrap,
        "confidence_level": 0.95,
        "total_patients": 45,
        "results": results_list,
    }

    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(out_payload, fp, indent=2)

    print(f"Paired statistics saved to {out_path} ({len(results_list)} rows).")
    for r in results_list:
        print(
            f"  {r['horizon']:8s} | {r['model_name']:16s} | Diff: {r['mean_difference']:+.2f} mg/dL | "
            f"95% CI: {r['ci_display']:14s} | Improved: {r['improved_count']}/{r['total_count']} ({r['improved_percentage']}%) | "
            f"Status: {r['status_display']}"
        )

    return out_payload


if __name__ == "__main__":
    compute_paired_statistics()
