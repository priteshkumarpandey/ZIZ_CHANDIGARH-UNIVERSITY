from datetime import datetime
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import pandas as pd

from src.glucotwin.serving.schemas.fhir import CGMReading, PatientProfile
from src.glucotwin.serving.schemas.requests import PredictRequest
from src.glucotwin.serving.schemas.responses import (
    ClinicalMetrics,
    ClinicalSummaryResponse,
    ForecastPoint,
)
from src.glucotwin.serving.service import inference_service
from src.glucotwin.synth.personas import get_matched_participant, get_persona_by_id
from src.glucotwin.utils.io import as_pandas_df, read_parquet


class ClinicalSummaryGenerator:
    @overload
    @staticmethod
    def _fetch_patient_series(
        patient_id: str,
        window_hours: int = 24,
        recent_cgm: list[CGMReading] | None = None,
        as_of: str | None = None,
        return_timestamps: Literal[False] = False,
    ) -> tuple[np.ndarray, PatientProfile | None]: ...

    @overload
    @staticmethod
    def _fetch_patient_series(
        patient_id: str,
        window_hours: int = 24,
        recent_cgm: list[CGMReading] | None = None,
        as_of: str | None = None,
        return_timestamps: Literal[True] = True,
    ) -> tuple[np.ndarray, list[datetime], PatientProfile | None]: ...

    @staticmethod
    def _fetch_patient_series(
        patient_id: str,
        window_hours: int = 24,
        recent_cgm: list[CGMReading] | None = None,
        as_of: str | None = None,
        return_timestamps: bool = False,
    ) -> Any:
        persona = get_persona_by_id(patient_id)
        if not persona:
            inference_service.load()
            persona = inference_service.patient_profiles.get(patient_id)

        if recent_cgm and len(recent_cgm) > 0:
            vals = np.array([float(r.glucose_mgdl) for r in recent_cgm], dtype=np.float32)
            ts_list = [
                pd.to_datetime(r.timestamp).to_pydatetime()
                if not isinstance(r.timestamp, datetime)
                else r.timestamp
                for r in recent_cgm
            ]
            if return_timestamps:
                return vals, ts_list, persona
            return vals, persona

        max_readings = window_hours * 12
        candidate_paths = [
            Path("data/demo/timeseries_demo.parquet"),
            Path("data/processed/timeseries_processed.parquet"),
        ]

        for ts_path in candidate_paths:
            if ts_path.exists():
                df_ts = as_pandas_df(read_parquet(ts_path))
                target_pid = patient_id
                if target_pid not in df_ts["patient_id"].values:
                    matched_info = get_matched_participant(patient_id)
                    if matched_info and not patient_id.startswith("DEMO_PAT"):
                        target_pid = matched_info.get("matched_participant_id", patient_id)

                rows = df_ts[df_ts["patient_id"] == target_pid]
                if not rows.empty:
                    rows_sorted = rows.sort_values("timestamp")
                    if as_of:
                        cutoff = pd.to_datetime(as_of)
                        if cutoff.tzinfo is not None:
                            rows_sorted["timestamp"] = pd.to_datetime(
                                rows_sorted["timestamp"], utc=True
                            )
                            cutoff = cutoff.tz_convert("UTC")
                        else:
                            rows_sorted["timestamp"] = pd.to_datetime(rows_sorted["timestamp"])
                        rows_filtered = rows_sorted[rows_sorted["timestamp"] <= cutoff]
                    else:
                        rows_filtered = rows_sorted

                    g_col = "glucose_mgdl" if "glucose_mgdl" in rows_filtered.columns else "glucose"
                    cgm_slice = rows_filtered[g_col].dropna().values[-max_readings:]
                    ts_slice = [
                        pd.to_datetime(t).to_pydatetime()
                        for t in rows_filtered["timestamp"].dropna().values[-max_readings:]
                    ]
                    if len(cgm_slice) > 0:
                        vals = np.array(cgm_slice, dtype=np.float32)
                        if return_timestamps:
                            return vals, ts_slice, persona
                        return vals, persona

        cached = inference_service.patient_latest_cgm.get(patient_id, {}).get("recent_series", [])
        if len(cached) > 0 and (len(cached) < 2 or float(np.std(cached)) > 0.1):
            vals = np.array(cached[-max_readings:], dtype=np.float32)
            end_ts = pd.to_datetime(as_of).to_pydatetime() if as_of else datetime.utcnow()
            ts_slice = [
                end_ts - pd.Timedelta(minutes=5 * (len(vals) - 1 - i)) for i in range(len(vals))
            ]
            if return_timestamps:
                return vals, ts_slice, persona
            return vals, persona

        if persona:
            try:
                from src.glucotwin.synth.generator import generate_patient_cgm_series

                days_needed = max(1, int(np.ceil(window_hours / 24.0)))
                ts_df, _ = generate_patient_cgm_series(patient_id, days=days_needed)
                vals = ts_df["glucose_mgdl"].to_numpy().astype(np.float32)
                if len(vals) > 0:
                    cgm_slice = vals[-max_readings:]
                    ts_col = ts_df["timestamp"]
                    raw_ts = (
                        ts_col.tail(max_readings).to_list()
                        if hasattr(ts_col, "tail")
                        else list(ts_col)[-max_readings:]
                    )
                    ts_slice = [pd.to_datetime(t).to_pydatetime() for t in raw_ts]
                    if return_timestamps:
                        return cgm_slice, ts_slice, persona
                    return cgm_slice, persona
            except Exception:
                pass

        default_g = float(persona.fasting_glucose) if persona else 125.0
        vals = np.full(max_readings, default_g, dtype=np.float32)
        end_ts = pd.to_datetime(as_of).to_pydatetime() if as_of else datetime.utcnow()
        ts_slice = [
            end_ts - pd.Timedelta(minutes=5 * (max_readings - 1 - i)) for i in range(max_readings)
        ]
        if return_timestamps:
            return vals, ts_slice, persona
        return vals, persona

    @staticmethod
    def generate(
        patient_id: str,
        window_hours: int = 24,
        recent_cgm: list[CGMReading] | None = None,
        as_of: str | None = None,
    ) -> ClinicalSummaryResponse:
        inference_service.load()
        glucose_vals, timestamps, persona = ClinicalSummaryGenerator._fetch_patient_series(
            patient_id=patient_id,
            window_hours=window_hours,
            recent_cgm=recent_cgm,
            as_of=as_of,
            return_timestamps=True,
        )

        actual_h = round(len(glucose_vals) * 5.0 / 60.0, 1)
        window_text = (
            f"last {int(actual_h) if actual_h.is_integer() else actual_h} h (all data so far)"
            if actual_h < window_hours
            else f"last {window_hours}h"
        )

        n_pts = len(glucose_vals)
        if n_pts == 0:
            glucose_vals = np.array([125.0] * (window_hours * 12), dtype=np.float32)
            n_pts = len(glucose_vals)

        mean_g = float(np.mean(glucose_vals))
        sd_g = float(np.std(glucose_vals))
        cv_g = float((sd_g / mean_g) * 100.0) if mean_g > 0 else 0.0

        tir = float(np.mean((glucose_vals >= 70.0) & (glucose_vals <= 180.0)) * 100.0)
        tar_l1 = float(np.mean(glucose_vals > 180.0) * 100.0)
        tar_l2 = float(np.mean(glucose_vals > 250.0) * 100.0)
        tbr_l1 = float(np.mean(glucose_vals < 70.0) * 100.0)
        tbr_l2 = float(np.mean(glucose_vals < 54.0) * 100.0)

        # Standard GMI formula: 3.31 + 0.02392 * mean glucose
        gmi_val = float(3.31 + (0.02392 * mean_g))

        metrics = ClinicalMetrics(
            time_in_range_percentage=round(tir, 1),
            time_above_range_percentage=round(tar_l1, 1),
            time_above_range_level2_percentage=round(tar_l2, 1),
            time_below_range_percentage=round(tbr_l1, 1),
            time_below_range_level2_percentage=round(tbr_l2, 1),
            gmi=round(gmi_val, 2),
            mean_glucose=round(mean_g, 1),
            standard_deviation=round(sd_g, 1),
            coefficient_of_variation=round(cv_g, 1),
        )

        # Generate model forecast with prediction intervals
        forecast_horizons: list[ForecastPoint] = []
        usable_alerts: dict[str, Any] = {}
        try:
            pred_req = PredictRequest(
                patient_id=patient_id, recent_cgm_series=glucose_vals[-72:].tolist()
            )
            forecast_resp = inference_service.predict(pred_req)
            forecast_horizons = forecast_resp.forecast_horizons

            # Strictly gate adverse events: filter to alert_usable == True only
            for fp in forecast_horizons:
                h_key = f"+{fp.horizon_minutes}m"
                usable_events_at_h: dict[str, Any] = {}
                alert_usable_map = fp.alert_usable or {}
                thresholds_map = fp.operating_thresholds or {}
                for ev_name, is_usable in alert_usable_map.items():
                    if is_usable:
                        prob = getattr(fp, f"{ev_name}_risk_probability", None)
                        if prob is None and ev_name == "spike":
                            prob = fp.spike_risk_probability
                        threshold = thresholds_map.get(ev_name, 0.50)
                        usable_events_at_h[ev_name] = {
                            "risk_probability": prob,
                            "operating_threshold": threshold,
                            "alert_active": bool(prob is not None and prob >= threshold),
                            "actionability": "actionable alert threshold",
                        }
                if usable_events_at_h:
                    usable_alerts[h_key] = usable_events_at_h
        except Exception:
            pass

        # Determine personalization status
        if patient_id.startswith("DEMO_PAT"):
            p_status = "calibrated synthetic demo"
        elif persona and persona.is_synthetic:
            p_status = "calibrated synthetic persona"
        elif patient_id.startswith("CGMacros-"):
            p_status = "calibrated (held-out test participant)"
        else:
            p_status = "population baseline"

        findings: list[str] = []
        recommendations: list[str] = []

        if tir < 70.0:
            findings.append(
                f"Suboptimal Time in Range (TIR {metrics.time_in_range_percentage}%, clinical target >70%)."
            )
        else:
            findings.append(
                f"Adequate Time in Range (TIR {metrics.time_in_range_percentage}% meets clinical target)."
            )

        if tar_l1 > 25.0:
            findings.append(
                f"Excessive hyperglycemia: {metrics.time_above_range_percentage}% >180 mg/dL (target <25%), with {metrics.time_above_range_level2_percentage}% >250 mg/dL."
            )
        if tbr_l1 > 4.0:
            findings.append(
                f"Elevated hypoglycemia risk: {metrics.time_below_range_percentage}% <70 mg/dL (target <4%), with {metrics.time_below_range_level2_percentage}% <54 mg/dL."
            )

        if cv_g > 36.0:
            findings.append(
                f"Elevated glycemic variability with CV of {metrics.coefficient_of_variation}% (clinical target <=36%)."
            )
        else:
            findings.append(
                f"Stable glycemic variability with CV of {metrics.coefficient_of_variation}% (within target <=36%)."
            )

        pat_name = persona.name if persona else f"Patient {patient_id}"
        region_str = (
            persona.region.value
            if persona and persona.region and hasattr(persona.region, "value")
            else (str(persona.region) if persona and persona.region else "General")
        )

        # Rule 1: Nocturnal hypoglycemia advice
        # Triggered if window includes night hours (23:00 to 06:00) and nocturnal reading falls below 70 mg/dL
        has_night_window = any((t.hour < 6 or t.hour >= 23) for t in timestamps)
        night_lows = [
            g
            for g, t in zip(glucose_vals, timestamps, strict=False)
            if (t.hour < 6 or t.hour >= 23) and g < 70.0
        ]
        if has_night_window and (len(night_lows) > 0 or tbr_l1 > 1.0):
            recommendations.append(
                "Consider bedtime complex protein snack (such as paneer or besan chilla) or clinical review of nocturnal insulin/medication to prevent nocturnal hypoglycemia."
            )

        # Rule 2: Walk advice / Postprandial excursion advice
        # Triggered if TAR > 15% or a rapid postprandial rise (>40 mg/dL over 30 min) is detected
        has_rapid_rise = False
        if len(glucose_vals) >= 6:
            for idx in range(len(glucose_vals) - 6):
                if float(glucose_vals[idx + 6] - glucose_vals[idx]) >= 40.0:
                    has_rapid_rise = True
                    break
        if tar_l1 > 15.0 or has_rapid_rise:
            recommendations.append(
                "Incorporate a 15-minute brisk walk within 30 minutes following meals with significant glycemic excursions (>180 mg/dL)."
            )

        # Rule 3: Dietary staple substitution
        # Triggered if hyperglycemia is substantial (TAR > 25% or mean glucose > 160 mg/dL)
        if tar_l1 > 25.0 or mean_g > 160.0:
            if "South India" in region_str:
                recommendations.append(
                    "Consider substituting polished white rice with finger millet (ragi) or whole pulse batters to blunt postprandial spikes."
                )
            elif "North India" in region_str:
                recommendations.append(
                    "Advise earlier dinner hours and portion moderation for whole wheat rotis to reduce postprandial glycemic excursions."
                )
            elif "East India" in region_str:
                recommendations.append(
                    "Moderate intake of traditional refined carbohydrate sweets; pair rice meals with high-fiber greens and lean protein."
                )
            elif "West India" in region_str:
                recommendations.append(
                    "Incorporate complex whole pulses and portion moderation for refined wheat preparations."
                )
            else:
                recommendations.append(
                    "Consider portion moderation for high glycemic index carbohydrate staples to reduce postprandial excursions."
                )

        # Rule 4: Daytime frequent hypoglycemia advice
        if tbr_l1 > 4.0:
            recommendations.append(
                "Frequent daytime hypoglycemia observed (<70 mg/dL exceeds 4% clinical target); review sulfonylurea or insulin dosing with care team."
            )

        narrative = (
            f"Clinical Summary for {pat_name} (ID: {patient_id}, {region_str}): "
            f"Over the evaluated {window_text} monitoring window, mean sensor glucose was {metrics.mean_glucose} mg/dL "
            f"with an estimated Glucose Management Indicator (GMI) of {metrics.gmi}% and glycemic variability (CV) of {metrics.coefficient_of_variation}%. "
            f"Time in Range (70-180 mg/dL) was recorded at {metrics.time_in_range_percentage}%, with {metrics.time_above_range_percentage}% spent in hyperglycemia "
            f"(including {metrics.time_above_range_level2_percentage}% severe >250 mg/dL) and {metrics.time_below_range_percentage}% spent in hypoglycemia "
            f"(including {metrics.time_below_range_level2_percentage}% clinically significant <54 mg/dL). "
            f"Digital twin state is currently operating in {p_status} mode."
        )

        return ClinicalSummaryResponse(
            patient_id=patient_id,
            patient_name=pat_name,
            generated_at=datetime.utcnow(),
            metrics=metrics,
            adverse_event_count_24h={
                "hyperglycemia_level_1": int(round(tar_l1 / 10.0)),
                "severe_hyperglycemia": int(round(tar_l2 / 10.0)),
                "hypoglycemia_level_1": int(round(tbr_l1 / 10.0)),
            },
            latest_forecast=forecast_horizons,
            adverse_events_usable=usable_alerts,
            personalization_status=p_status,
            key_findings=findings,
            actionable_recommendations=recommendations,
            plain_language_narrative=narrative,
            is_llm_enhanced=False,
            monitoring_window_text=window_text,
            disclaimer="Decision support only. Not intended as medical diagnosis or direct prescription.",
        )
