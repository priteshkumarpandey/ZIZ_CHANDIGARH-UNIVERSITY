import sys

sys.path.insert(0, ".")
import json
import numpy as np
import pandas as pd
from src.glucotwin.serving.service import inference_service
from src.glucotwin.twin.simulator import CounterfactualTwinSimulator

inference_service.load()
meals = pd.read_parquet("data/processed/meals_processed.parquet")
meals["timestamp"] = pd.to_datetime(meals["timestamp"])
ts = pd.read_parquet("data/processed/timeseries_processed.parquet")
ts["timestamp"] = pd.to_datetime(ts["timestamp"])
chk_manifest = json.load(open("models/checkpoints/manifest.json"))
p2f = chk_manifest["patient_to_test_fold"]

records = []
for f_idx in range(5):
    fold_pids = [p for p, f in p2f.items() if f == f_idx]
    fold_meals = meals[(meals["patient_id"].isin(fold_pids)) & (meals["carbs_g"] >= 10.0)]
    sampled_meals = fold_meals.sample(min(len(fold_meals), 100), random_state=42 + f_idx)
    for _, m_row in sampled_meals.iterrows():
        pid = str(m_row["patient_id"])
        t_m = m_row["timestamp"]
        c_g = float(m_row["carbs_g"])
        p_ts = ts[ts["patient_id"] == pid].sort_values("timestamp")
        past_ts = p_ts[p_ts["timestamp"] <= t_m].tail(72)
        post_ts = p_ts[
            (p_ts["timestamp"] >= t_m) & (p_ts["timestamp"] <= t_m + pd.Timedelta(minutes=120))
        ]
        if len(past_ts) < 24 or len(post_ts) < 6:
            continue
        cgm_raw = past_ts["glucose_mgdl"].tolist()
        cgm_clean = inference_service._prepare_cgm_series(pid, cgm_raw)
        obs_clean = post_ts["glucose_mgdl"].dropna().values
        if len(obs_clean) == 0:
            continue
        g_base = float(cgm_clean[-1])
        obs_pk = float(np.max(obs_clean))
        obs_rise_gbase = max(0.0, obs_pk - g_base)

        sim = CounterfactualTwinSimulator(pid)
        b_preds = sim._run_model_inference(cgm_clean, carbs_g=0.0)
        s_preds = sim._run_model_inference(cgm_clean, carbs_g=c_g)

        b_pk = max(b_preds[h]["q50"] for h in [30, 60, 120])
        s_pk = max(s_preds[h]["q50"] for h in [30, 60, 120])

        # Observed readings specifically at +30, +60, +120
        h_obs = []
        for h_m in [30, 60, 120]:
            h_time = t_m + pd.Timedelta(minutes=h_m)
            sub = p_ts[
                (p_ts["timestamp"] >= h_time - pd.Timedelta(minutes=7))
                & (p_ts["timestamp"] <= h_time + pd.Timedelta(minutes=7))
            ]
            if len(sub) > 0:
                h_obs.append(float(sub["glucose_mgdl"].iloc[0]))
        obs_pk_3h = max(h_obs) if h_obs else obs_pk
        obs_rise_3h = max(0.0, obs_pk_3h - g_base)

        pred_rise_cf = max(0.0, s_pk - b_pk)
        pred_rise_gbase = max(0.0, s_pk - g_base)

        records.append(
            {
                "fold": f_idx,
                "pid": pid,
                "carbs_g": c_g,
                "g_base": g_base,
                "obs_pk": obs_pk,
                "obs_rise_gbase": obs_rise_gbase,
                "obs_rise_3h": obs_rise_3h,
                "s_pk": s_pk,
                "b_pk": b_pk,
                "pred_rise_cf": pred_rise_cf,
                "pred_rise_gbase": pred_rise_gbase,
            }
        )

df = pd.DataFrame(records)
print(f"Total records: {len(df)}")
print(f"Mean obs_rise_gbase (observed peak in 120m - g_base): {df['obs_rise_gbase'].mean():.2f}")
print(f"Mean obs_rise_3h (observed peak at 30,60,120 - g_base): {df['obs_rise_3h'].mean():.2f}")
print(f"Mean pred_rise_cf (s_pk - b_pk): {df['pred_rise_cf'].mean():.2f}")
print(f"Mean pred_rise_gbase (s_pk - g_base): {df['pred_rise_gbase'].mean():.2f}")

# Cross-validation linear calibration evaluation:
# For each held-out fold, fit linear regression (y_obs = a * x_pred + b) on training folds (the other 4 folds),
# then evaluate on held-out fold:
for target_name, pred_name in [
    ("obs_rise_gbase", "pred_rise_cf"),
    ("obs_rise_gbase", "pred_rise_gbase"),
    ("obs_rise_3h", "pred_rise_cf"),
    ("obs_rise_3h", "pred_rise_gbase"),
]:
    print(f"\n--- Calibration: target={target_name}, pred={pred_name} ---")
    maes_raw = []
    biases_raw = []
    maes_cal = []
    biases_cal = []
    slopes = []
    intercepts = []

    for f_idx in range(5):
        train_df = df[df["fold"] != f_idx]
        test_df = df[df["fold"] == f_idx]

        x_tr = train_df[pred_name].values
        y_tr = train_df[target_name].values

        x_te = test_df[pred_name].values
        y_te = test_df[target_name].values

        # Linear fit: y = a * x + b
        poly = np.polyfit(x_tr, y_tr, 1)
        a, b = float(poly[0]), float(poly[1])
        slopes.append(a)
        intercepts.append(b)

        # Before calibration
        err_raw = x_te - y_te
        mae_raw = float(np.mean(np.abs(err_raw)))
        bias_raw = float(np.mean(err_raw))
        maes_raw.append(mae_raw)
        biases_raw.append(bias_raw)

        # After calibration: pred_cal = a * x_te + b (clipped at 0)
        pred_cal = np.maximum(0.0, a * x_te + b)
        err_cal = pred_cal - y_te
        mae_cal = float(np.mean(np.abs(err_cal)))
        bias_cal = float(np.mean(err_cal))
        maes_cal.append(mae_cal)
        biases_cal.append(bias_cal)

    print(f"Mean slope a: {np.mean(slopes):.3f}, mean intercept b: {np.mean(intercepts):.3f}")
    print(f"Raw: MAE = {np.mean(maes_raw):.2f} mg/dL, Bias = {np.mean(biases_raw):.2f} mg/dL")
    print(
        f"Calibrated: MAE = {np.mean(maes_cal):.2f} mg/dL, Bias = {np.mean(biases_cal):.2f} mg/dL"
    )
    improvement = np.mean(maes_raw) - np.mean(maes_cal)
    print(
        f"MAE Improvement: {improvement:+.2f} mg/dL ({'IMPROVED' if improvement > 0 else 'WORSENED'})"
    )
