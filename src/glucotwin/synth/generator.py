import argparse
import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from src.glucotwin.synth.personas import SYNTHETIC_PERSONAS, get_persona_by_id
from src.glucotwin.utils.io import write_parquet


def generate_patient_cgm_series(
    patient_id: str, days: int = 14, sampling_interval_minutes: int = 5, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    np.random.seed(seed + abs(hash(patient_id)) % 10000)
    persona = get_persona_by_id(patient_id)
    if not persona:
        raise ValueError(f"Patient {patient_id} not found")

    baseline_glucose = persona.fasting_glucose
    total_steps = (days * 24 * 60) // sampling_interval_minutes
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = [
        start_time + timedelta(minutes=i * sampling_interval_minutes) for i in range(total_steps)
    ]

    glucose_values: list[float] = []
    heart_rates: list[float] = []
    steps_list: list[int] = []
    mets_list: list[float] = []
    carbs_list: list[float] = []
    gi_list: list[float] = []

    current_g = baseline_glucose + float(np.random.normal(0, 4))
    active_carb_flux = 0.0

    meal_schedules: list[dict[str, Any]] = [
        {"hour": 8, "minute": 30, "prob": 0.95, "carb_range": (35.0, 60.0), "gi": 65.0},
        {"hour": 13, "minute": 30, "prob": 0.98, "carb_range": (50.0, 85.0), "gi": 70.0},
        {"hour": 21, "minute": 30, "prob": 0.95, "carb_range": (45.0, 80.0), "gi": 68.0},
    ]

    for ts in timestamps:
        hour_frac = ts.hour + ts.minute / 60.0

        dawn_effect = 0.0
        if 4.0 <= hour_frac <= 7.0 and persona.id in ["IND_PAT_002", "IND_PAT_003"]:
            dawn_effect = 15.0 * math.sin((hour_frac - 4.0) / 3.0 * math.pi)

        nocturnal_dip = 0.0
        if 2.0 <= hour_frac <= 5.0 and "Basal Insulin" in [
            m.value for m in persona.medication_classes
        ]:
            nocturnal_dip = -20.0 * math.sin((hour_frac - 2.0) / 3.0 * math.pi)

        carb_input = 0.0
        meal_gi = 0.0
        for sched in meal_schedules:
            if ts.hour == sched["hour"] and ts.minute == sched["minute"]:
                prob = float(sched["prob"])
                if np.random.rand() < prob:
                    cr = sched["carb_range"]
                    carb_input = float(np.random.uniform(cr[0], cr[1]))
                    meal_gi = float(sched["gi"])
                    active_carb_flux += (carb_input * (meal_gi / 55.0)) * 0.45

        # Two-compartment decay and mean reversion
        reversion = (baseline_glucose + dawn_effect + nocturnal_dip - current_g) * 0.035
        carb_impact = active_carb_flux * 0.09
        active_carb_flux = max(0.0, active_carb_flux * 0.96)
        noise = float(np.random.normal(0, 1.5))

        current_g = max(40.0, min(450.0, current_g + reversion + carb_impact + noise))

        glucose_values.append(round(current_g, 1))
        heart_rates.append(72.0)
        steps_list.append(10)
        mets_list.append(1.1)
        carbs_list.append(round(carb_input, 1))
        gi_list.append(round(meal_gi, 1))

    ts_dict = {
        "timestamp": timestamps,
        "patient_id": [patient_id] * total_steps,
        "glucose_mgdl": glucose_values,
        "heart_rate_bpm": heart_rates,
        "steps": steps_list,
        "mets": mets_list,
        "carbs_g": carbs_list,
        "glycemic_index": gi_list,
    }

    st_dict = {
        "patient_id": [persona.id],
        "name": [persona.name],
        "age": [persona.age],
        "gender": [
            persona.gender.value if hasattr(persona.gender, "value") else str(persona.gender)
        ],
        "region": [
            persona.region.value
            if persona.region and hasattr(persona.region, "value")
            else (str(persona.region) if persona.region else None)
        ],
        "bmi": [persona.bmi],
        "hba1c": [persona.hba1c],
        "diabetes_duration_years": [persona.diabetes_duration_years],
        "egfr": [persona.egfr],
        "systolic_bp": [persona.systolic_bp],
        "diastolic_bp": [persona.diastolic_bp],
        "fasting_glucose": [persona.fasting_glucose],
        "postprandial_glucose": [persona.postprandial_glucose],
    }

    if pl is not None:
        return pl.DataFrame(ts_dict), pl.DataFrame(st_dict)
    return pd.DataFrame(ts_dict), pd.DataFrame(st_dict)


def generate_cohort(days: int = 7, seed: int = 42):
    all_ts = []
    all_st = []
    for persona in SYNTHETIC_PERSONAS:
        ts, st = generate_patient_cgm_series(persona.id, days=days, seed=seed)
        all_ts.append(ts)
        all_st.append(st)
    if pl is not None:
        return pl.concat(all_ts), pl.concat(all_st)
    return pd.concat(all_ts, ignore_index=True), pd.concat(all_st, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--output", type=str, default="data/processed/synthetic_cohort.parquet")
    args = parser.parse_args()

    ts, st = generate_cohort(days=args.days)
    merged = ts.join(st, on="patient_id", how="left")
    write_parquet(merged, args.output)
