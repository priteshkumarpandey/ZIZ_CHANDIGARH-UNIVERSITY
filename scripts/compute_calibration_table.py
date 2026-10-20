import json
import numpy as np
import pandas as pd
from src.glucotwin.serving.service import inference_service
from src.glucotwin.twin.simulator import CounterfactualTwinSimulator

inference_service.load()

# Re-evaluate all meals exactly as in validate_whatif.py to get raw, intercept, and proportional MAE per fold
df_pat = pd.read_parquet("data/processed/patients_processed.parquet")
cg_pat = df_pat[df_pat["source_dataset"] == "cgmacros"]
pat_status = dict(zip(cg_pat["patient_id"], cg_pat["diabetes_status"]))

df_meals = pd.read_parquet("data/processed/meals_processed.parquet")
df_ts = pd.read_parquet("data/processed/timeseries_processed.parquet")
ts_by_pid = {p: g.sort_values("timestamp") for p, g in df_ts.groupby("patient_id")}

manifest = json.load(open("models/checkpoints/manifest.json"))
pat_to_fold = manifest["patient_to_test_fold"]

all_records = []
for pid, f_idx in pat_to_fold.items():
    p_meals = df_meals[df_meals["patient_id"] == pid]
    pts = ts_by_pid.get(pid)
    if pts is None:
        continue
    for _, row in p_meals.iterrows():
        t_meal = pd.to_datetime(row["timestamp"])
        c_g = float(row.get("carbs_g", 0.0))
        if c_g <= 5.0:
            continue
        hist = pts[pts["timestamp"] <= t_meal].tail(72)
        if len(hist) < 36:
            continue
        post = pts[
            (pts["timestamp"] > t_meal) & (pts["timestamp"] <= t_meal + pd.Timedelta(minutes=120))
        ]
        if len(post) < 6:
            continue
        cgm_clean = hist["glucose_mgdl"].interpolate().bfill().ffill().values.tolist()
        g_0 = cgm_clean[-1]
        obs_rise = max(0.0, float(post["glucose_mgdl"].max() - g_0))
        sim = CounterfactualTwinSimulator(pid)
        b_preds = sim._run_model_inference(cgm_clean, carbs_g=0.0)
        s_preds = sim._run_model_inference(cgm_clean, carbs_g=c_g)
        b_pk = max(b_preds[h]["q50"] for h in [30, 60, 120])
        s_pk = max(s_preds[h]["q50"] for h in [30, 60, 120])
        pred_rise = max(0.0, s_pk - b_pk)
        all_records.append({"fold": f_idx, "pred_rise": pred_rise, "obs_rise": obs_rise})

df = pd.DataFrame(all_records)
print("=" * 66)
print(f"{'Fold':<6} {'Raw MAE':<12} {'Intercept MAE':<16} {'Proportional MAE':<18} {'Winner':<10}")
print("=" * 66)
raw_maes, int_maes, prop_maes = [], [], []
for f_idx in range(5):
    tr = df[df["fold"] != f_idx]
    te = df[df["fold"] == f_idx]
    x_tr, y_tr = tr["pred_rise"].values, tr["obs_rise"].values
    x_te, y_te = te["pred_rise"].values, te["obs_rise"].values

    # Raw
    mae_raw = float(np.mean(np.abs(x_te - y_te)))
    # Intercept (fit y = a*x + b)
    p = np.polyfit(x_tr, y_tr, 1)
    pred_int = np.maximum(0.0, p[0] * x_te + p[1])
    mae_int = float(np.mean(np.abs(pred_int - y_te)))
    # Proportional (fit y = a*x through origin: a = sum(x*y)/sum(x^2))
    a_prop = float(np.sum(x_tr * y_tr) / np.sum(x_tr**2))
    pred_prop = np.maximum(0.0, a_prop * x_te)
    mae_prop = float(np.mean(np.abs(pred_prop - y_te)))

    winner = "Intercept" if mae_int < mae_prop else "Proportional"
    print(f"{f_idx:<6} {mae_raw:<12.2f} {mae_int:<16.2f} {mae_prop:<18.2f} {winner:<10}")
    raw_maes.append(mae_raw)
    int_maes.append(mae_int)
    prop_maes.append(mae_prop)

print("=" * 66)
print(
    f"{'Mean':<6} {np.mean(raw_maes):<12.2f} {np.mean(int_maes):<16.2f} {np.mean(prop_maes):<18.2f}"
)
print("=" * 66)
