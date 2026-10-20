from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_safety_floor_params() -> dict[str, Any]:
    params_path = Path("params.yaml")
    if params_path.exists():
        try:
            with open(params_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
                if isinstance(data, dict) and "safety_floor_alerts" in data:
                    res = data["safety_floor_alerts"]
                    if isinstance(res, dict):
                        return res
        except Exception:
            pass
    return {
        "low_threshold_mgdl": 70.0,
        "urgent_low_threshold_mgdl": 54.0,
        "rapid_fall_rate_mgdl_per_min": 2.0,
        "rapid_fall_duration_minutes": 15,
        "rapid_fall_ceiling_mgdl": 120.0,
    }


def evaluate_safety_floor_rules(
    readings: list[float] | list[dict[str, Any]] | list[Any] | np.ndarray,
    interval_minutes: int = 5,
) -> list[dict[str, Any]]:
    params = load_safety_floor_params()
    low_th = float(params.get("low_threshold_mgdl", 70.0))
    urgent_low_th = float(params.get("urgent_low_threshold_mgdl", 54.0))
    rapid_rate_th = float(params.get("rapid_fall_rate_mgdl_per_min", 2.0))
    rapid_dur_min = int(params.get("rapid_fall_duration_minutes", 15))
    rapid_ceiling_th = float(params.get("rapid_fall_ceiling_mgdl", 120.0))

    vals: list[float] = []
    for r in readings:
        if r is None:
            continue
        if isinstance(r, dict):
            raw_v = r.get("glucose_mgdl", r.get("glucose"))
        elif hasattr(r, "glucose_mgdl"):
            raw_v = r.glucose_mgdl
        else:
            raw_v = r
        try:
            fv = float(raw_v)
            if not np.isnan(fv):
                vals.append(fv)
        except (ValueError, TypeError):
            continue

    if not vals:
        return []

    curr_g = vals[-1]
    alerts: list[dict[str, Any]] = []

    # Rule 1: Urgent Low (observed reading < 54 mg/dL)
    if curr_g < urgent_low_th:
        alerts.append(
            {
                "alert_type": "urgent_low",
                "severity": "critical",
                "title": f"Urgent Low Glucose (<{urgent_low_th:.0f} mg/dL)",
                "message": (
                    f"Observed glucose is {curr_g:.1f} mg/dL, crossing below the critical "
                    f"{urgent_low_th:.0f} mg/dL safety floor."
                ),
                "observed_glucose": round(curr_g, 1),
                "threshold_mgdl": urgent_low_th,
                "is_rule_based": True,
                "provenance": "Rule-based alert (observed, not predicted)",
            }
        )

    # Rule 2: Low (observed reading < 70 mg/dL and >= 54 mg/dL)
    elif curr_g < low_th:
        alerts.append(
            {
                "alert_type": "low",
                "severity": "warning",
                "title": f"Low Glucose (<{low_th:.0f} mg/dL)",
                "message": (
                    f"Observed glucose is {curr_g:.1f} mg/dL, crossing below the "
                    f"{low_th:.0f} mg/dL safety floor."
                ),
                "observed_glucose": round(curr_g, 1),
                "threshold_mgdl": low_th,
                "is_rule_based": True,
                "provenance": "Rule-based alert (observed, not predicted)",
            }
        )

    # Rule 3: Rapid Fall (a drop of at least 2 mg/dL per minute sustained over 15 minutes while below 120)
    steps_needed = max(2, int(round(rapid_dur_min / interval_minutes))) + 1
    if len(vals) >= steps_needed:
        window = vals[-steps_needed:]
        total_drop = float(window[0] - window[-1])
        actual_rate = total_drop / float(rapid_dur_min)
        is_below_ceiling = curr_g < rapid_ceiling_th
        is_sustained_drop = all(window[i] <= window[i - 1] for i in range(1, len(window)))

        if is_below_ceiling and actual_rate >= rapid_rate_th and is_sustained_drop:
            alerts.append(
                {
                    "alert_type": "rapid_fall",
                    "severity": "warning",
                    "title": f"Rapid Glucose Fall (>={rapid_rate_th:.0f} mg/dL/min)",
                    "message": (
                        f"Observed glucose dropped by {total_drop:.1f} mg/dL over {rapid_dur_min} minutes "
                        f"({actual_rate:.1f} mg/dL/min) while below {rapid_ceiling_th:.0f} mg/dL."
                    ),
                    "observed_glucose": round(curr_g, 1),
                    "threshold_mgdl": rapid_ceiling_th,
                    "rate_mgdl_per_min": round(actual_rate, 2),
                    "is_rule_based": True,
                    "provenance": "Rule-based alert (observed, not predicted)",
                }
            )

    return alerts
