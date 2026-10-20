import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.glucotwin.serving.schemas.fhir import ActivityRecord, MealRecord, PatientProfile
from src.glucotwin.serving.schemas.responses import SimulationResponse
from src.glucotwin.serving.service import inference_service
from src.glucotwin.synth.meals import get_food_by_name
from src.glucotwin.synth.personas import get_matched_participant
from src.glucotwin.utils.io import as_pandas_df, read_parquet

# Load held-out calibration validation evidence if present
_WHATIF_REPORT_PATH = Path("reports/whatif_validation.json")
_CALIBRATION_EVIDENCE: dict[str, Any] = {}
if _WHATIF_REPORT_PATH.exists():
    try:
        with open(_WHATIF_REPORT_PATH, encoding="utf-8") as _fp:
            _report_data = json.load(_fp)
            _CALIBRATION_EVIDENCE = _report_data.get("calibration_evaluation", {})
    except Exception:
        pass


class CounterfactualTwinSimulator:
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        inference_service.load()
        self.fold_idx = inference_service.get_test_fold_for_patient(patient_id)
        self.ts_model = inference_service.torchscript_models[self.fold_idx]
        self.scalers = inference_service.scalers[self.fold_idx]

    def _prepare_cgm_window(
        self,
        recent_cgm: list[float] | None = None,
        initial_glucose: float | None = None,
        as_of: datetime | None = None,
    ) -> np.ndarray:
        if recent_cgm and len(recent_cgm) > 0:
            return inference_service._prepare_cgm_series(self.patient_id, recent_cgm)

        if as_of is not None:
            candidate_paths = [
                Path("data/demo/timeseries_demo.parquet"),
                Path("data/processed/timeseries_processed.parquet"),
            ]
            for c_ts in candidate_paths:
                if c_ts.exists():
                    df_ts = as_pandas_df(read_parquet(c_ts))
                    target_pid = self.patient_id
                    if target_pid not in df_ts["patient_id"].values:
                        matched_info = get_matched_participant(self.patient_id)
                        if matched_info and not self.patient_id.startswith("DEMO_PAT"):
                            target_pid = matched_info.get("matched_participant_id", self.patient_id)
                    rows = df_ts[df_ts["patient_id"] == target_pid]
                    if not rows.empty:
                        rows_sorted = rows.sort_values("timestamp")
                        cutoff = pd.to_datetime(as_of)
                        ts_col = pd.to_datetime(rows_sorted["timestamp"])
                        if ts_col.dt.tz is not None and cutoff.tzinfo is None:
                            cutoff = cutoff.tz_localize("UTC")
                        elif ts_col.dt.tz is None and cutoff.tzinfo is not None:
                            cutoff = cutoff.tz_localize(None)
                        elif ts_col.dt.tz is not None and cutoff.tzinfo is not None:
                            cutoff = cutoff.tz_convert(ts_col.dt.tz)
                        rows_sorted["timestamp"] = ts_col
                        rows_filtered = rows_sorted[rows_sorted["timestamp"] <= cutoff]
                        g_col = (
                            "glucose_mgdl" if "glucose_mgdl" in rows_filtered.columns else "glucose"
                        )
                        s_vals = rows_filtered[g_col].dropna().values
                        if len(s_vals) > 0:
                            cgm_slice = [float(x) for x in s_vals[-72:]]
                            return inference_service._prepare_cgm_series(self.patient_id, cgm_slice)

        if initial_glucose is not None:
            return np.full(72, float(initial_glucose), dtype=np.float32)

        cached = inference_service.patient_latest_cgm.get(self.patient_id, {}).get(
            "recent_series", []
        )
        if len(cached) > 0:
            return inference_service._prepare_cgm_series(self.patient_id, cached)

        return np.full(72, 135.0, dtype=np.float32)

    def _run_model_inference(
        self,
        cgm_72: np.ndarray,
        carbs_g: float = 0.0,
        meal_offset_mins: int = 0,
        as_of: datetime | None = None,
    ) -> dict[int, dict[str, float]]:
        now = as_of or datetime.utcnow()
        T = len(cgm_72)

        # 1. Observed sequence features
        diff_5m = np.zeros(T, dtype=np.float32)
        diff_15m = np.zeros(T, dtype=np.float32)
        diff_30m = np.zeros(T, dtype=np.float32)
        diff_60m = np.zeros(T, dtype=np.float32)
        roll_mean = np.zeros(T, dtype=np.float32)
        roll_std = np.zeros(T, dtype=np.float32)

        for t in range(T):
            diff_5m[t] = cgm_72[t] - cgm_72[max(0, t - 1)]
            diff_15m[t] = cgm_72[t] - cgm_72[max(0, t - 3)]
            diff_30m[t] = cgm_72[t] - cgm_72[max(0, t - 6)]
            diff_60m[t] = cgm_72[t] - cgm_72[max(0, t - 12)]
            w = cgm_72[max(0, t - 5) : t + 1]
            roll_mean[t] = float(np.mean(w))
            roll_std[t] = float(np.std(w))

        hr_series = np.full(T, 72.0, dtype=np.float32)
        steps_series = np.zeros(T, dtype=np.float32)
        act_series = np.full(T, 1.0, dtype=np.float32)
        carbs_series = np.zeros(T, dtype=np.float32)

        if carbs_g > 0.0 and meal_offset_mins <= 0:
            step_idx = max(0, min(T - 1, (T - 1) + int(meal_offset_mins / 5.0)))
            carbs_series[step_idx] = carbs_g

        seq = np.stack(
            [
                cgm_72,
                diff_5m,
                diff_15m,
                diff_30m,
                diff_60m,
                roll_mean,
                roll_std,
                hr_series,
                steps_series,
                act_series,
                carbs_series,
            ],
            axis=-1,
        )

        # 2. Known future features
        fut = np.zeros((24, 6), dtype=np.float32)
        for step in range(24):
            dt = now + timedelta(minutes=(step + 1) * 5)
            h = dt.hour + dt.minute / 60.0
            d = dt.weekday()
            fut[step, 0] = np.sin(2.0 * np.pi * h / 24.0)
            fut[step, 1] = np.cos(2.0 * np.pi * h / 24.0)
            fut[step, 2] = np.sin(2.0 * np.pi * d / 7.0)
            fut[step, 3] = np.cos(2.0 * np.pi * d / 7.0)

            t_min = (step + 1) * 5.0
            if carbs_g > 0.0:
                if t_min >= meal_offset_mins:
                    fut[step, 4] = t_min - meal_offset_mins
                    fut[step, 5] = carbs_g
                else:
                    fut[step, 4] = 120.0
                    fut[step, 5] = 0.0
            else:
                fut[step, 4] = t_min
                fut[step, 5] = 0.0

        # 3. Static features
        stat = inference_service.patient_static_features.get(
            self.patient_id, np.zeros(16, dtype=np.float32)
        ).copy()

        # Normalization
        seq_mean = self.scalers["seq_mean"]
        seq_std = self.scalers["seq_std"]
        seq_norm = (seq - seq_mean) / seq_std

        stat_scaler = self.scalers["static_scaler"]
        stat_norm = stat_scaler.transform(stat.reshape(1, -1))[0]

        seq_t = torch.from_numpy(seq_norm).unsqueeze(0).float()
        fut_t = torch.from_numpy(fut).unsqueeze(0).float()
        stat_t = torch.from_numpy(stat_norm).unsqueeze(0).float()

        with torch.no_grad():
            out = self.ts_model(seq_t, fut_t, stat_t)

        preds: dict[int, dict[str, float]] = {}
        for horizon in [30, 60, 120]:
            qh = out[f"h{horizon}"][0].numpy()
            preds[horizon] = {
                "q10": float(np.clip(qh[0], 40.0, 400.0)),
                "q50": float(np.clip(qh[1], 40.0, 400.0)),
                "q90": float(np.clip(qh[2], 40.0, 400.0)),
            }
        return preds

    @staticmethod
    def project_isotonic_trajectory(
        peaks: list[float], baseline_peak: float
    ) -> tuple[list[float], bool, float]:
        iso_peaks: list[float] = []
        running_max = float(baseline_peak)
        total_correction = 0.0
        constraint_applied = False

        for p in peaks:
            val = float(p)
            if val < running_max:
                corr = running_max - val
                total_correction += corr
                constraint_applied = True
                iso_peaks.append(round(running_max, 4))
            else:
                running_max = val
                iso_peaks.append(round(val, 4))

        return iso_peaks, constraint_applied, round(total_correction, 4)

    def simulate(
        self,
        as_of: datetime | None = None,
        recent_cgm: list[float] | None = None,
        initial_glucose: float | None = None,
        meal_carbs_g: float = 0.0,
        meal_offset_mins: int = 0,
        activity_mets: float = 1.0,
        activity_duration_mins: int = 0,
        activity_offset_mins: int = 0,
        food_lookups: list[dict[str, Any]] | None = None,
        has_activity_intervention: bool = False,
        calibrate_meal_response: bool = False,
    ) -> dict[str, Any]:
        cgm_72 = self._prepare_cgm_window(recent_cgm, initial_glucose, as_of=as_of)
        curr_g = float(cgm_72[-1])
        now = as_of or datetime.utcnow()

        # 1. Baseline counterfactual (zero intervention)
        base_preds = self._run_model_inference(
            cgm_72=cgm_72,
            carbs_g=0.0,
            meal_offset_mins=0,
            as_of=now,
        )

        constraint_applied = False
        total_correction_mgdl = 0.0

        # 2. Scenario counterfactual with optional calibration and physiology constraint
        CALIBRATION_PARAMS = {
            0: {"slope": 1.0761, "intercept": 19.5370},
            1: {"slope": 0.8154, "intercept": 30.4285},
            2: {"slope": 1.0022, "intercept": 26.2698},
            3: {"slope": 0.9619, "intercept": 24.5038},
            4: {"slope": 0.8466, "intercept": 26.5541},
        }
        cal_cfg = CALIBRATION_PARAMS.get(self.fold_idx, {"slope": 0.9404, "intercept": 25.4586})

        if meal_carbs_g > 0.0:
            scen_preds = self._run_model_inference(
                cgm_72=cgm_72,
                carbs_g=meal_carbs_g,
                meal_offset_mins=meal_offset_mins,
                as_of=now,
            )

            # Apply linear calibration mapping fit on training folds ONLY if requested
            if calibrate_meal_response:
                raw_deltas = {h: scen_preds[h]["q50"] - base_preds[h]["q50"] for h in [30, 60, 120]}
                raw_peak_delta = max(raw_deltas.values())
                if raw_peak_delta > 0.0:
                    carb_factor = min(1.0, max(0.1, meal_carbs_g / 50.0))
                    cal_target_delta = max(
                        0.0, cal_cfg["slope"] * raw_peak_delta + cal_cfg["intercept"] * carb_factor
                    )
                    cal_ratio = cal_target_delta / max(1e-3, raw_peak_delta)
                    for h in [30, 60, 120]:
                        if raw_deltas[h] > 0.0:
                            scaled_shift = raw_deltas[h] * cal_ratio
                            scen_preds[h]["q50"] = base_preds[h]["q50"] + scaled_shift
                            scen_preds[h]["q10"] = float(
                                np.clip(
                                    scen_preds[h]["q10"] + (scaled_shift - raw_deltas[h]),
                                    40.0,
                                    400.0,
                                )
                            )
                            scen_preds[h]["q90"] = float(
                                np.clip(
                                    scen_preds[h]["q90"] + (scaled_shift - raw_deltas[h]),
                                    40.0,
                                    400.0,
                                )
                            )

            # Physiology constraint: added carbohydrates must produce non-negative delta from baseline
            for h in [30, 60, 120]:
                delta_h = scen_preds[h]["q50"] - base_preds[h]["q50"]
                if delta_h < 0.0:
                    corr = -delta_h
                    total_correction_mgdl += corr
                    constraint_applied = True
                    scen_preds[h]["q50"] = base_preds[h]["q50"]
                    scen_preds[h]["q10"] = float(np.clip(scen_preds[h]["q10"] + corr, 40.0, 400.0))
                    scen_preds[h]["q90"] = float(np.clip(scen_preds[h]["q90"] + corr, 40.0, 400.0))

            # Non-decreasing peak constraint relative to baseline peak
            base_pk_val = max(base_preds[h]["q50"] for h in [30, 60, 120])
            scen_pk_val = max(scen_preds[h]["q50"] for h in [30, 60, 120])
            if scen_pk_val < base_pk_val:
                peak_corr = base_pk_val - scen_pk_val
                total_correction_mgdl += peak_corr
                constraint_applied = True
                scen_preds[60]["q50"] = base_pk_val
        else:
            # Exact zero meal invariance
            scen_preds = {h: dict(base_preds[h]) for h in [30, 60, 120]}

        # 3. Bounded activity correction (assumption-based and unvalidated)
        activity_deductions: dict[int, float] = {30: 0.0, 60: 0.0, 120: 0.0}
        if has_activity_intervention and activity_duration_mins > 0 and activity_mets > 1.0:
            delta_mets = max(0.0, activity_mets - 1.0)
            clearance_rate = 0.38  # mg/dL per active minute per net MET
            act_start = max(0, activity_offset_mins)
            act_end = activity_offset_mins + activity_duration_mins

            for h in [30, 60, 120]:
                active_mins = max(0, min(h, act_end) - max(0, act_start))
                # Bounded physiological ceiling of 50 mg/dL deduction
                reduction = min(50.0, delta_mets * clearance_rate * float(active_mins))
                activity_deductions[h] = reduction
                scen_preds[h]["q10"] = float(np.clip(scen_preds[h]["q10"] - reduction, 40.0, 400.0))
                scen_preds[h]["q50"] = float(np.clip(scen_preds[h]["q50"] - reduction, 40.0, 400.0))
                scen_preds[h]["q90"] = float(np.clip(scen_preds[h]["q90"] - reduction, 40.0, 400.0))

        # 4. Deltas and peak computations
        deltas = {
            f"+{h}m": round(scen_preds[h]["q50"] - base_preds[h]["q50"], 1) for h in [30, 60, 120]
        }
        base_peak = max(base_preds[h]["q50"] for h in [30, 60, 120])
        scen_peak = max(scen_preds[h]["q50"] for h in [30, 60, 120])
        peak_change = round(scen_peak - base_peak, 1)
        peak_reduction = max(0.0, round(base_peak - scen_peak, 1))

        # 5. Build continuous trajectory (5-min intervals over 180 min)
        num_steps = int(180 / 5) + 1
        timestamps = [now + timedelta(minutes=i * 5) for i in range(num_steps)]

        # Interpolate baseline trajectory through curr_g, h30, h60, h120, h180
        h_vals_base = [
            curr_g,
            base_preds[30]["q50"],
            base_preds[60]["q50"],
            base_preds[120]["q50"],
            base_preds[120]["q50"],
        ]
        h_times = [0, 6, 12, 24, 36]
        base_traj = [
            round(float(np.clip(np.interp(step, h_times, h_vals_base), 40.0, 400.0)), 1)
            for step in range(num_steps)
        ]

        # Interpolate scenario trajectory
        h_vals_scen = [
            curr_g,
            scen_preds[30]["q50"],
            scen_preds[60]["q50"],
            scen_preds[120]["q50"],
            scen_preds[120]["q50"],
        ]
        scen_traj = [
            round(float(np.clip(np.interp(step, h_times, h_vals_scen), 40.0, 400.0)), 1)
            for step in range(num_steps)
        ]

        h_vals_q10 = [
            curr_g,
            scen_preds[30]["q10"],
            scen_preds[60]["q10"],
            scen_preds[120]["q10"],
            scen_preds[120]["q10"],
        ]
        scen_q10_traj = [
            round(float(np.clip(np.interp(step, h_times, h_vals_q10), 40.0, 400.0)), 1)
            for step in range(num_steps)
        ]

        h_vals_q90 = [
            curr_g,
            scen_preds[30]["q90"],
            scen_preds[60]["q90"],
            scen_preds[120]["q90"],
            scen_preds[120]["q90"],
        ]
        scen_q90_traj = [
            round(float(np.clip(np.interp(step, h_times, h_vals_q90), 40.0, 400.0)), 1)
            for step in range(num_steps)
        ]

        # 6. Provenance metadata
        calibration_meta = {
            "calibrate_meal_response": calibrate_meal_response,
            "status": "applied"
            if calibrate_meal_response and meal_carbs_g > 0.0
            else ("identity_zero_meal" if meal_carbs_g == 0.0 else "raw_model_default"),
            "fold": self.fold_idx,
            "slope": cal_cfg["slope"],
            "intercept": cal_cfg["intercept"],
            "held_out_rule": "4 of 5 folds must improve held-out MAE",
            "held_out_rule_satisfied": False,
            "held_out_evidence": _CALIBRATION_EVIDENCE,
            "note": "Simulator defaults to raw model output because intercept calibration improved MAE in only 3 of 5 folds (rule required 4 of 5).",
        }

        provenance = {
            "meal_response": "model_based",
            "calibrate_meal_response": calibrate_meal_response,
            "served_fold": self.fold_idx,
            "zero_leakage_guarantee": f"Evaluated using held-out fold {self.fold_idx} model for patient {self.patient_id}",
            "calibration": calibration_meta,
            "activity_clearance": "assumption_based_unvalidated"
            if has_activity_intervention
            else "none",
            "activity_assumptions": {
                "status": "assumption-based and unvalidated",
                "baseline_resting_mets": 1.0,
                "clearance_rate_mgdl_per_min": 0.38,
                "maximum_clearance_mgdl": 50.0,
                "origin_of_coefficient": "The clearance coefficient (0.38 mg/dL per active minute per net MET) is an illustrative assumption chosen by the team, with no source and no validation. The effect is capped at 50 mg/dL.",
                "formula": "delta_glucose = -min(50.0, max(0.0, mets - 1.0) * 0.38 * active_minutes)",
            }
            if has_activity_intervention
            else None,
            "food_table_lookups": food_lookups or [],
            "physiology_constraint": {
                "constraint_applied": constraint_applied,
                "correction_mgdl": round(total_correction_mgdl, 2),
                "rule": "Isotonic non-negative postprandial glucose excursion for added carbohydrates",
            },
        }

        # Format horizons output
        baseline_horizons = {
            f"+{h}m": {k: round(v, 1) for k, v in base_preds[h].items()} for h in [30, 60, 120]
        }
        scenario_horizons = {
            f"+{h}m": {k: round(v, 1) for k, v in scen_preds[h].items()} for h in [30, 60, 120]
        }

        takeaway = (
            f"What-If counterfactual simulation indicates a peak glucose shift of {peak_change:+.1f} mg/dL "
            f"(baseline peak {base_peak:.1f} mg/dL vs scenario peak {scen_peak:.1f} mg/dL). "
            f"Deltas: +30m: {deltas['+30m']:+.1f} mg/dL, +60m: {deltas['+60m']:+.1f} mg/dL, +120m: {deltas['+120m']:+.1f} mg/dL."
        )
        if constraint_applied:
            takeaway += (
                f" Physiology constraint applied (+{total_correction_mgdl:.1f} mg/dL correction)."
            )
        if has_activity_intervention:
            takeaway += (
                " Exercise clearance effect is assumption-based and unvalidated on trial data."
            )

        return {
            "patient_id": self.patient_id,
            "simulated_at": now,
            "timestamps": timestamps,
            "baseline_trajectory": base_traj,
            "scenario_trajectory": scen_traj,
            "scenario_q10": scen_q10_traj,
            "scenario_q90": scen_q90_traj,
            "baseline_peak_glucose": round(base_peak, 1),
            "scenario_peak_glucose": round(scen_peak, 1),
            "peak_reduction_mgdl": peak_reduction,
            "peak_change_mgdl": peak_change,
            "baseline_horizons": baseline_horizons,
            "scenario_horizons": scenario_horizons,
            "deltas": deltas,
            "provenance": provenance,
            "constraint_applied": constraint_applied,
            "constraint_correction_mgdl": round(total_correction_mgdl, 2),
            "clinical_takeaway": takeaway,
            "as_of": now.isoformat() if hasattr(now, "isoformat") else str(now),
            "starting_glucose": round(curr_g, 1),
        }


def run_twin_simulation(
    patient_id: str,
    as_of: datetime | None = None,
    recent_cgm_series: list[float] | None = None,
    interventions: list[Any] | None = None,
    initial_glucose: float | None = None,
    meal: MealRecord | None = None,
    activity: ActivityRecord | None = None,
    duration_minutes: int = 180,
    calibrate_meal_response: bool = False,
) -> SimulationResponse:
    sim = CounterfactualTwinSimulator(patient_id=patient_id)

    total_carbs = 0.0
    meal_offset = 0
    total_mets = 1.0
    act_duration = 0
    act_offset = 0
    has_activity = False
    food_lookups: list[dict[str, Any]] = []
    applied_interventions: list[dict[str, Any]] = []

    # Parse modern structured interventions
    if interventions:
        for it in interventions:
            it_dict = (
                it.model_dump()
                if hasattr(it, "model_dump")
                else (it if isinstance(it, dict) else {})
            )
            applied_interventions.append(it_dict)
            itype = it_dict.get("intervention_type", "meal").lower()

            if itype == "meal":
                food_name = it_dict.get("food_name")
                c_g = float(it_dict.get("carbs_g", 0.0))
                offset_m = int(it_dict.get("offset_minutes", 0))

                if food_name:
                    food_obj = get_food_by_name(food_name)
                    if food_obj:
                        c_g = food_obj.carbohydrates_g
                        food_lookups.append(
                            {
                                "food": food_obj.name,
                                "carbs_g": food_obj.carbohydrates_g,
                                "glycemic_index": food_obj.glycemic_index,
                                "source": food_obj.source,
                                "composition_source": food_obj.composition_source,
                                "gi_source": food_obj.gi_source,
                                "is_approximate": food_obj.is_approximate,
                                "notes": food_obj.notes,
                                "provenance": "food_table",
                            }
                        )
                    else:
                        food_lookups.append(
                            {
                                "food": food_name,
                                "carbs_g": c_g,
                                "provenance": "user_input",
                            }
                        )
                else:
                    food_lookups.append(
                        {
                            "food": "Custom meal entry",
                            "carbs_g": c_g,
                            "provenance": "user_input",
                        }
                    )

                total_carbs += c_g
                meal_offset = offset_m

            elif itype == "activity":
                has_activity = True
                act_mets = float(it_dict.get("mets", 3.0))
                act_dur = int(it_dict.get("duration_minutes", 30))
                act_off = int(it_dict.get("offset_minutes", 15))
                total_mets = max(total_mets, act_mets)
                act_duration = max(act_duration, act_dur)
                act_offset = act_off

    # Parse legacy parameters for full backwards compatibility
    if meal is not None:
        total_carbs += meal.total_carbohydrates_g
        food_lookups.append(
            {
                "food": meal.name,
                "carbs_g": meal.total_carbohydrates_g,
                "provenance": "legacy_meal_record",
            }
        )
    if activity is not None:
        has_activity = True
        total_mets = max(total_mets, activity.mets)
        act_duration = max(act_duration, activity.duration_minutes)

    sim_result = sim.simulate(
        as_of=as_of,
        recent_cgm=recent_cgm_series,
        initial_glucose=initial_glucose,
        meal_carbs_g=total_carbs,
        meal_offset_mins=meal_offset,
        activity_mets=total_mets,
        activity_duration_mins=act_duration,
        activity_offset_mins=act_offset,
        food_lookups=food_lookups,
        has_activity_intervention=has_activity,
        calibrate_meal_response=calibrate_meal_response,
    )

    return SimulationResponse(
        patient_id=sim_result["patient_id"],
        simulated_at=sim_result["simulated_at"],
        timestamps=sim_result["timestamps"],
        baseline_trajectory=sim_result["baseline_trajectory"],
        scenario_trajectory=sim_result["scenario_trajectory"],
        scenario_q10=sim_result["scenario_q10"],
        scenario_q90=sim_result["scenario_q90"],
        baseline_peak_glucose=sim_result["baseline_peak_glucose"],
        scenario_peak_glucose=sim_result["scenario_peak_glucose"],
        peak_reduction_mgdl=sim_result["peak_reduction_mgdl"],
        peak_change_mgdl=sim_result["peak_change_mgdl"],
        baseline_horizons=sim_result["baseline_horizons"],
        scenario_horizons=sim_result["scenario_horizons"],
        deltas=sim_result["deltas"],
        provenance=sim_result["provenance"],
        interventions_applied=applied_interventions,
        constraint_applied=sim_result.get("constraint_applied", False),
        constraint_correction_mgdl=sim_result.get("constraint_correction_mgdl", 0.0),
        clinical_takeaway=sim_result["clinical_takeaway"],
        as_of=sim_result.get("as_of"),
        starting_glucose=sim_result.get("starting_glucose"),
    )


# Legacy alias for backward compatibility
class PhysiologySimulator:
    def __init__(self, patient: PatientProfile):
        self.patient = patient
        self.sim = CounterfactualTwinSimulator(patient.id)

    def simulate(
        self,
        initial_glucose: float,
        duration_minutes: int = 180,
        meal: MealRecord | None = None,
        activity: ActivityRecord | None = None,
        dt: float = 5.0,
    ) -> list[float]:
        res = run_twin_simulation(
            patient_id=self.patient.id,
            initial_glucose=initial_glucose,
            meal=meal,
            activity=activity,
            duration_minutes=duration_minutes,
        )
        return res.scenario_trajectory
