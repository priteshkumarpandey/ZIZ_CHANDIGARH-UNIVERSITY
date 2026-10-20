"""Generate synthetic demo patients and time series for GlucoTwin.

This script creates 3 realistic, purely synthetic demo patients
with 5-minute CGM time series, gut carbohydrate absorption dynamics,
circadian rhythms, meal markers, and short sensor dropout intervals.
None of the data is derived from real clinical datasets (CGMacros, HUPA, AI-READI).
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Ensure repository root is on sys.path for direct execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.glucotwin.serving.schemas.fhir import Gender  # noqa: E402
from src.glucotwin.synth.personas import get_demo_personas  # noqa: E402


def generate_synthetic_patient_timeseries(
    patient_id: str,
    base_glucose: float,
    carb_sensitivity: float,
    clearance_rate: float,
    daily_meals: list[tuple[float, float, str]],
    start_time: datetime,
    days: int = 3,
    sensor_gaps: list[tuple[int, int]] | None = None,
    seed: int = 42,
    low_episode: tuple[int, list[float]] | None = None,
) -> list[dict[str, Any]]:
    """Generate 5-minute time series with circadian rhythms, gut absorption, meals, and activities."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    records: list[dict[str, Any]] = []
    total_steps = days * 288  # 288 5-min intervals per day
    curr_glucose = base_glucose
    gut_carbs = 0.0

    gap_steps = set()
    if sensor_gaps:
        for start_step, length in sensor_gaps:
            for g in range(start_step, start_step + length):
                gap_steps.add(g)

    ep_start: int = -1
    ep_vals: list[float] = []
    if low_episode:
        ep_start, ep_vals = low_episode

    for step_idx in range(total_steps):
        step_dt = start_time + timedelta(minutes=step_idx * 5)
        hour_of_day = step_dt.hour + step_dt.minute / 60.0

        # Circadian baseline oscillation: lower nocturnal minimum (03:00-04:00), dawn phenomenon (06:00-08:00)
        circadian = 8.0 * math.sin(2 * math.pi * (hour_of_day - 6.0) / 24.0)
        target_baseline = base_glucose + circadian

        # Check for meals ingested in this 5-minute window
        carbs = 0.0
        meal_desc = "none"
        for meal_h, meal_carbs, name in daily_meals:
            if abs(hour_of_day - meal_h) < (2.5 / 60.0):
                carbs = meal_carbs
                meal_desc = name
                gut_carbs += carbs
                break

        # Gut absorption: realistic 2-compartment absorption decay (halflife ~40-50 mins)
        absorbed = gut_carbs * 0.08
        gut_carbs -= absorbed
        meal_rise = absorbed * carb_sensitivity

        # Clearance towards circadian baseline
        clearance = clearance_rate * (curr_glucose - target_baseline)

        # Check for physical activity
        # Morning walk 07:30 - 08:00, Evening walk 18:30 - 19:00
        is_walking = (7.5 <= hour_of_day < 8.0) or (18.5 <= hour_of_day < 19.0)
        if is_walking:
            steps = rng.randint(90, 140)
            mets = round(rng.uniform(3.0, 3.8), 2)
            heart_rate = round(rng.uniform(95.0, 115.0), 1)
            activity_type = "walking"
            exercise_clearance = 0.65 * (mets - 1.0)
        elif 23.0 <= hour_of_day or hour_of_day < 6.5:
            # Sleeping / nocturnal rest
            steps = 0
            mets = 1.0
            heart_rate = round(rng.uniform(56.0, 68.0), 1)
            activity_type = "sedentary"
            exercise_clearance = 0.0
        else:
            # Daytime light activity
            steps = rng.randint(5, 30)
            mets = round(rng.uniform(1.1, 1.4), 2)
            heart_rate = round(rng.uniform(70.0, 84.0), 1)
            activity_type = "sedentary"
            exercise_clearance = 0.0

        # Stochastic small noise
        noise = float(np_rng.normal(0.0, 0.65))

        if 0 <= ep_start <= step_idx < ep_start + len(ep_vals):
            curr_glucose = ep_vals[step_idx - ep_start]
        else:
            curr_glucose = max(
                40.0, min(340.0, curr_glucose + meal_rise - clearance - exercise_clearance + noise)
            )

        is_gap = step_idx in gap_steps
        glucose_val = np.nan if is_gap else round(curr_glucose, 1)

        records.append(
            {
                "patient_id": patient_id,
                "timestamp": step_dt,
                "glucose": glucose_val,
                "glucose_mgdl": glucose_val,
                "heart_rate": heart_rate,
                "steps": steps,
                "mets": mets,
                "carbohydrates": round(carbs, 1),
                "bolus_insulin": 0.0,
                "basal_insulin": 0.0,
                "meal_type": meal_desc,
                "activity_type": activity_type,
                "is_synthetic": True,
            }
        )

    return records


def generate_demo_data(output_dir: str | Path = "data/demo") -> dict[str, Any]:
    """Generate 3 synthetic demo patients and their time series."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sim_params = {
        "DEMO_PAT_001": {
            "carb_sens": 1.05,
            "clearance_rate": 0.035,
            "meals": [
                (8.5, 55.0, "Breakfast"),
                (13.0, 75.0, "Lunch"),
                (17.5, 30.0, "Evening Snack"),
                (20.5, 65.0, "Dinner"),
            ],
            "gaps": [(250, 3), (620, 2)],
            "low_episode": (420, [118.0, 106.0, 93.0, 79.0, 65.0, 53.0, 51.0, 68.0, 88.0, 110.0]),
        },
        "DEMO_PAT_002": {
            "carb_sens": 0.65,
            "clearance_rate": 0.050,
            "meals": [
                (8.5, 45.0, "Breakfast"),
                (13.0, 60.0, "Lunch"),
                (20.0, 50.0, "Dinner"),
            ],
            "gaps": [(310, 2), (700, 2)],
            "low_episode": (480, [116.0, 104.0, 91.0, 77.0, 63.0, 52.0, 50.0, 66.0, 84.0, 108.0]),
        },
        "DEMO_PAT_003": {
            "carb_sens": 0.40,
            "clearance_rate": 0.075,
            "meals": [
                (8.0, 35.0, "Breakfast"),
                (13.0, 50.0, "Lunch"),
                (19.5, 45.0, "Dinner"),
            ],
            "gaps": [(180, 2)],
        },
    }

    demo_personas = get_demo_personas()
    demo_patients = []
    for persona in demo_personas:
        sp = sim_params.get(persona.id, {})
        status = (
            "T2D"
            if persona.id == "DEMO_PAT_001"
            else ("prediabetes" if persona.id == "DEMO_PAT_002" else "healthy")
        )
        reg_val = (
            persona.region.value
            if persona.region and hasattr(persona.region, "value")
            else (str(persona.region) if persona.region else "West India")
        )
        diet_val = (
            persona.dietary_habit.value
            if persona.dietary_habit and hasattr(persona.dietary_habit, "value")
            else (str(persona.dietary_habit) if persona.dietary_habit else "Vegetarian")
        )
        demo_patients.append(
            {
                "patient_id": persona.id,
                "name": persona.name,
                "age": persona.age,
                "sex": "M" if persona.gender == Gender.MALE else "F",
                "bmi": persona.bmi,
                "hba1c": persona.hba1c,
                "fasting_glucose": persona.fasting_glucose,
                "systolic_bp": persona.systolic_bp,
                "diastolic_bp": persona.diastolic_bp,
                "diabetes_status": status,
                "diabetes_duration_years": persona.diabetes_duration_years,
                "egfr": persona.egfr,
                "region": reg_val,
                "dietary_habit": diet_val,
                "occupation": persona.occupation,
                "active_medications": persona.active_medications,
                "source_dataset": "synthetic_demo",
                "is_synthetic": True,
                "base_glucose": persona.fasting_glucose,
                "carb_sens": sp.get("carb_sens", 0.8),
                "clearance_rate": sp.get("clearance_rate", 0.05),
                "meals": sp.get("meals", []),
                "gaps": sp.get("gaps", []),
                "low_episode": sp.get("low_episode"),
            }
        )

    # Time series start timestamp: exactly 3 days ago from now, rounded to 5 mins
    now = datetime.now(UTC)
    start_time = (now - timedelta(days=3)).replace(second=0, microsecond=0)
    minute_remainder = start_time.minute % 5
    start_time = start_time - timedelta(minutes=minute_remainder)

    all_timeseries: list[dict[str, Any]] = []

    for idx, p in enumerate(demo_patients):
        daily_meals: Any = p["meals"]
        sensor_gaps: Any = p["gaps"]
        low_ep: Any = p.get("low_episode")
        ts = generate_synthetic_patient_timeseries(
            patient_id=str(p["patient_id"]),
            base_glucose=float(str(p["base_glucose"])),
            carb_sensitivity=float(str(p["carb_sens"])),
            clearance_rate=float(str(p["clearance_rate"])),
            daily_meals=daily_meals,
            start_time=start_time,
            days=3,
            sensor_gaps=sensor_gaps,
            seed=42 + idx,
            low_episode=low_ep,
        )
        all_timeseries.extend(ts)

    dropped_keys = {
        "base_glucose",
        "carb_sens",
        "clearance_rate",
        "meals",
        "gaps",
        "low_episode",
    }
    clean_patients = [{k: v for k, v in p.items() if k not in dropped_keys} for p in demo_patients]

    # Save to Parquet using polars for zero-dependency parquet writing
    pat_file = out_path / "patients_demo.parquet"
    ts_file = out_path / "timeseries_demo.parquet"
    json_file = out_path / "patients_demo.json"

    df_patients = pl.DataFrame(clean_patients)
    df_timeseries = pl.DataFrame(all_timeseries)

    df_patients.write_parquet(str(pat_file))
    df_timeseries.write_parquet(str(ts_file))

    with open(json_file, "w", encoding="utf-8") as fp:
        json.dump(demo_patients, fp, indent=2, default=str)

    print(f"Generated {len(df_patients)} demo patients -> {pat_file}")
    print(f"Generated {len(df_timeseries)} demo timeseries readings -> {ts_file}")

    return {
        "patients_count": len(df_patients),
        "timeseries_count": len(df_timeseries),
        "patients_file": str(pat_file),
        "timeseries_file": str(ts_file),
    }


if __name__ == "__main__":
    generate_demo_data()
