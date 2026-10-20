from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def clarke_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    zones = np.empty(len(y_true), dtype="<U1")

    for i in range(len(y_true)):
        ref = y_true[i]
        pred = y_pred[i]

        # Clarke Error Grid standard piecewise boundaries (mg/dL)
        if (ref <= 70 and pred <= 70) or (pred <= 1.2 * ref and pred >= 0.8 * ref):
            zones[i] = "A"
        elif (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
            zones[i] = "E"
        elif ((ref >= 70 and ref <= 290) and (pred >= ref + 110)) or (
            (ref >= 130 and ref <= 180) and (pred <= (7 / 5) * ref - 182)
        ):
            zones[i] = "C"
        elif (
            (ref >= 240 and (pred >= 70 and pred <= 180))
            or (ref <= 70 and (pred >= 70 and pred <= 180))
            or (ref >= 70 and ref <= 180 and pred >= 240)
        ):
            zones[i] = "D"
        else:
            zones[i] = "B"

    total = len(zones)
    counts = {z: int(np.sum(zones == z)) for z in ["A", "B", "C", "D", "E"]}
    percentages = {
        "zone_a": round(float(counts["A"] / total * 100.0), 2) if total > 0 else 0.0,
        "zone_b": round(float(counts["B"] / total * 100.0), 2) if total > 0 else 0.0,
        "zone_ab": round(float((counts["A"] + counts["B"]) / total * 100.0), 2)
        if total > 0
        else 0.0,
        "zone_c": round(float(counts["C"] / total * 100.0), 2) if total > 0 else 0.0,
        "zone_d": round(float(counts["D"] / total * 100.0), 2) if total > 0 else 0.0,
        "zone_e": round(float(counts["E"] / total * 100.0), 2) if total > 0 else 0.0,
        "zone_cde": round(float((counts["C"] + counts["D"] + counts["E"]) / total * 100.0), 2)
        if total > 0
        else 0.0,
    }
    return zones, percentages


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]

    if len(y_t) == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "mard": float("nan"),
            "zone_a": float("nan"),
            "zone_b": float("nan"),
            "zone_ab": float("nan"),
            "zone_cde": float("nan"),
        }

    errors = y_p - y_t
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    # Avoid zero division in relative difference computation
    nonzero_mask = y_t > 0
    mard = (
        float(np.mean(np.abs(errors[nonzero_mask] / y_t[nonzero_mask])) * 100.0)
        if np.any(nonzero_mask)
        else 0.0
    )

    _, clarke = clarke_error_grid(y_t, y_p)

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mard": round(mard, 2),
        "zone_a": clarke["zone_a"],
        "zone_b": clarke["zone_b"],
        "zone_ab": clarke["zone_ab"],
        "zone_cde": clarke["zone_cde"],
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    lead_time_minutes: int | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_t = y_true[mask]
    y_p = y_prob[mask]

    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))

    if len(y_t) == 0 or n_pos == 0 or n_neg == 0:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "lead_time_mins": float(lead_time_minutes) if lead_time_minutes is not None else 0.0,
            "pos_rate": round(float(n_pos / len(y_t) * 100.0), 2) if len(y_t) > 0 else 0.0,
        }

    y_pred_bin = (y_p >= threshold).astype(int)
    try:
        auroc = float(roc_auc_score(y_t, y_p))
    except Exception:
        auroc = float("nan")

    try:
        auprc = float(average_precision_score(y_t, y_p))
    except Exception:
        auprc = float("nan")

    prec = float(precision_score(y_t, y_pred_bin, zero_division=0))
    rec = float(recall_score(y_t, y_pred_bin, zero_division=0))

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "lead_time_mins": float(lead_time_minutes) if lead_time_minutes is not None else 0.0,
        "pos_rate": round(float(n_pos / len(y_t) * 100.0), 2),
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
    target_coverage: float = 0.80,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    q10 = np.asarray(q10, dtype=float)
    q90 = np.asarray(q90, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(q10) & ~np.isnan(q90)
    y_t = y_true[mask]
    q_low = q10[mask]
    q_high = q90[mask]

    if len(y_t) == 0:
        return {
            "empirical_coverage": float("nan"),
            "target_coverage": target_coverage * 100.0,
            "mean_interval_width": float("nan"),
            "coverage_error": float("nan"),
        }

    inside = (y_t >= q_low) & (y_t <= q_high)
    empirical_coverage = float(np.mean(inside) * 100.0)
    interval_widths = q_high - q_low
    mean_width = float(np.mean(interval_widths))

    return {
        "empirical_coverage": round(empirical_coverage, 2),
        "target_coverage": round(target_coverage * 100.0, 1),
        "mean_interval_width": round(mean_width, 2),
        "coverage_error": round(empirical_coverage - (target_coverage * 100.0), 2),
    }


def paired_significance_test(
    model_errors: np.ndarray,
    baseline_errors: np.ndarray,
) -> dict[str, float]:
    m_err = np.abs(np.asarray(model_errors, dtype=float))
    b_err = np.abs(np.asarray(baseline_errors, dtype=float))

    diff = m_err - b_err
    mask = ~np.isnan(diff)
    diff = diff[mask]

    if len(diff) < 10 or np.all(diff == 0):
        return {
            "mean_mae_difference": float(np.mean(diff)) if len(diff) > 0 else 0.0,
            "p_value": 1.0,
            "is_significant": False,
        }

    try:
        stat, p_val = stats.wilcoxon(diff)
        p_val = float(p_val)
    except Exception:
        p_val = 1.0

    return {
        "mean_mae_difference": round(float(np.mean(diff)), 4),
        "p_value": float(p_val),
        "is_significant": bool(p_val < 0.05),
    }


def compute_episode_alert_metrics(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    horizon_minutes: int,
    step_minutes: int = 5,
    preceding_window_minutes: int = 120,
    min_clean_minutes: int = 60,
    min_below_minutes: int = 30,
) -> dict[str, Any]:
    p_ids = np.asarray(patient_ids)
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    horizon_steps = horizon_minutes // step_minutes
    window_steps = preceding_window_minutes // step_minutes
    clean_steps = min_clean_minutes // step_minutes
    below_steps = min_below_minutes // step_minutes

    all_lead_times: list[float] = []
    total_events = 0
    detected_events = 0
    total_episodes = 0
    true_episodes = 0
    false_episodes = 0
    total_patient_steps = len(y_t)

    unique_patients = np.unique(p_ids)
    for pid in unique_patients:
        p_mask = p_ids == pid
        pat_true = y_t[p_mask]
        pat_prob = y_p[p_mask]
        n_pat = len(pat_true)
        if n_pat == 0:
            continue

        # 1. True event onsets: glucose crossing event threshold with >= 60m of non-event before it
        onsets: list[int] = []
        for j in range(clean_steps, n_pat):
            if pat_true[j] == 1 and np.all(pat_true[j - clean_steps : j] == 0):
                onsets.append(j)

        # 2. Alert episodes: probability crosses threshold after being below for >= 30m
        ep_starts: list[int] = []
        for k in range(below_steps, n_pat):
            if pat_prob[k] >= threshold and np.all(pat_prob[k - below_steps : k] < threshold):
                ep_starts.append(k)

        total_episodes += len(ep_starts)
        detected_episodes_set = set()

        # 3. Match onsets to alert episodes
        for j in onsets:
            total_events += 1
            s_cross = j + horizon_steps
            s_win_start = s_cross - window_steps

            # Check if alert was already active at start of 120m window
            already_active = False
            if s_win_start >= 0 and s_win_start < n_pat:
                if pat_prob[s_win_start] >= threshold:
                    already_active = True
                else:
                    chk_start = max(0, s_win_start - below_steps)
                    if np.any(pat_prob[chk_start:s_win_start] >= threshold):
                        already_active = True

            if already_active:
                # Event is a miss because alert was already active before lookback window
                continue

            # Find alert episode that began within 120m before crossing
            qualifying = [k for k in ep_starts if s_win_start <= k <= min(s_cross, n_pat - 1)]
            if qualifying:
                k_first = qualifying[0]
                lt = float((s_cross - k_first) * step_minutes)
                all_lead_times.append(lt)
                detected_events += 1
                detected_episodes_set.add(k_first)

        # Count true and false alert episodes
        for k in ep_starts:
            if k in detected_episodes_set:
                true_episodes += 1
            else:
                warned = False
                for j in onsets:
                    s_cross = j + horizon_steps
                    if k <= s_cross <= k + window_steps:
                        warned = True
                        break
                if warned:
                    true_episodes += 1
                else:
                    false_episodes += 1

    patient_days = float(total_patient_steps * step_minutes / 1440.0)
    prec = float(true_episodes / total_episodes) if total_episodes > 0 else 0.0
    rec = float(detected_events / total_events) if total_events > 0 else 0.0
    false_per_day = float(false_episodes / patient_days) if patient_days > 0 else 0.0

    if all_lead_times:
        lt_arr = np.array(all_lead_times)
        median_lt = float(np.median(lt_arr))
        q25 = float(np.percentile(lt_arr, 25))
        q75 = float(np.percentile(lt_arr, 75))
        iqr_lt = float(q75 - q25)
        share_30m = float(np.mean(lt_arr >= 30.0) * 100.0)
    else:
        median_lt = 0.0
        iqr_lt = 0.0
        share_30m = 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "false_alerts_per_patient_day": round(false_per_day, 2),
        "lead_time_median": round(median_lt, 2),
        "lead_time_iqr": round(iqr_lt, 2),
        "share_detected_30m_early": round(share_30m, 2),
        "total_events": float(total_events),
        "detected_events": float(detected_events),
        "total_episodes": float(total_episodes),
        "true_episodes": float(true_episodes),
        "false_episodes": float(false_episodes),
    }


def compute_patient_level_lead_times(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    operating_threshold: float = 0.50,
    preceding_window_steps: int = 24,
    min_early_minutes: float = 30.0,
    step_minutes: int = 5,
    horizon_minutes: int = 0,
) -> dict[str, float]:
    return compute_episode_alert_metrics(
        patient_ids=patient_ids,
        y_true=y_true,
        y_prob=y_prob,
        threshold=operating_threshold,
        horizon_minutes=horizon_minutes,
        step_minutes=step_minutes,
        preceding_window_minutes=preceding_window_steps * step_minutes,
        min_clean_minutes=60,
        min_below_minutes=30,
    )


def tune_operating_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: np.ndarray | None = None,
    horizon_minutes: int = 60,
    target_sensitivity: float = 0.80,
    max_false_alerts_per_day: float = 1.0,
    candidate_thresholds: list[float] | None = None,
) -> float:
    candidates = candidate_thresholds or [round(x * 0.02, 2) for x in range(1, 46)]
    p_ids = patient_ids if patient_ids is not None else np.zeros(len(y_true))

    # We evaluate all candidates and choose threshold that:
    # 1. Satisfies false_alerts_per_patient_day <= max_false_alerts_per_day
    # 2. Among valid candidates, maximizes recall (closest to or exceeding target_sensitivity)
    # 3. Breaks ties with precision
    # If no candidate satisfies budget, picks candidate with lowest false_alerts_per_day
    best_th = 0.50
    best_score = (-1.0, -1.0)  # (recall, precision)
    min_false_alerts = float("inf")
    th_min_false_alerts = 0.50

    for th in candidates:
        res = compute_operating_point_metrics(
            patient_ids=p_ids,
            y_true=y_true,
            y_prob=y_prob,
            threshold=th,
            horizon_minutes=horizon_minutes,
        )
        rec = res["recall"]
        prec = res["precision"]
        false_per_day = res["false_alerts_per_patient_day"]

        if false_per_day < min_false_alerts:
            min_false_alerts = false_per_day
            th_min_false_alerts = th

        if false_per_day <= max_false_alerts_per_day:
            score = (rec, prec)
            if score > best_score:
                best_score = score
                best_th = th

    if best_score[0] < 0:
        return th_min_false_alerts

    return best_th


def compute_operating_curve(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    horizon_minutes: int,
    candidate_thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    candidates = candidate_thresholds or [round(x * 0.05, 2) for x in range(1, 19)]
    curve = []
    for th in candidates:
        res = compute_operating_point_metrics(
            patient_ids=patient_ids,
            y_true=y_true,
            y_prob=y_prob,
            threshold=th,
            horizon_minutes=horizon_minutes,
        )
        curve.append(
            {
                "threshold": th,
                "precision": res["precision"],
                "recall": res["recall"],
                "false_alerts_per_patient_day": res["false_alerts_per_patient_day"],
                "lead_time_median": res["lead_time_median"],
                "lead_time_iqr": res["lead_time_iqr"],
                "share_detected_30m_early": res["share_detected_30m_early"],
            }
        )
    return curve


def compute_operating_point_metrics(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    horizon_minutes: int,
    preceding_window_steps: int = 24,
    min_early_minutes: float = 30.0,
    step_minutes: int = 5,
) -> dict[str, float]:
    ep_metrics = compute_episode_alert_metrics(
        patient_ids=patient_ids,
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        horizon_minutes=horizon_minutes,
        step_minutes=step_minutes,
        preceding_window_minutes=preceding_window_steps * step_minutes,
        min_clean_minutes=60,
        min_below_minutes=30,
    )
    ep_metrics["operating_threshold"] = round(threshold, 4)
    return ep_metrics
