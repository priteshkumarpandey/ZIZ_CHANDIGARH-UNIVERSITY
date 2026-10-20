from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.glucotwin.serving.schemas.fhir import CGMReading, PatientProfile
from src.glucotwin.serving.schemas.responses import TwinStateResponse
from src.glucotwin.serving.service import inference_service
from src.glucotwin.synth.generator import generate_patient_cgm_series
from src.glucotwin.synth.personas import get_persona_by_id
from src.glucotwin.utils.io import as_pandas_df, read_parquet

router = APIRouter(prefix="/twin", tags=["Digital Twin & Patients"])


@router.get("/patients", response_model=list[PatientProfile])
def list_patients():
    return inference_service.list_patients()


@router.get("/{patient_id}/state", response_model=TwinStateResponse)
def get_patient_twin_state(patient_id: str):
    try:
        return inference_service.get_patient_state(patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{patient_id}/cgm", response_model=list[CGMReading])
def get_patient_cgm(patient_id: str, days: int = Query(default=2, ge=1, le=14)):
    candidate_paths = [
        Path("data/processed/timeseries_processed.parquet"),
        Path("data/demo/timeseries_demo.parquet"),
    ]
    for ts_file in candidate_paths:
        if ts_file.exists():
            df_ts = as_pandas_df(read_parquet(ts_file))
            patient_rows = df_ts[df_ts["patient_id"] == patient_id]
            if not patient_rows.empty:
                patient_rows = patient_rows.sort_values("timestamp")
                max_readings = days * 288
                subset = patient_rows.tail(max_readings)
                readings = []
                for _, row in subset.iterrows():
                    dt = pd.to_datetime(row["timestamp"])
                    g_val = (
                        float(row["glucose_mgdl"])
                        if "glucose_mgdl" in row and pd.notna(row["glucose_mgdl"])
                        else float(row.get("glucose", 120.0))
                    )
                    hr_val = (
                        float(row["heart_rate"])
                        if "heart_rate" in row and pd.notna(row.get("heart_rate"))
                        else (
                            float(row["heart_rate_bpm"])
                            if "heart_rate_bpm" in row and pd.notna(row.get("heart_rate_bpm"))
                            else None
                        )
                    )
                    step_val = (
                        int(row["steps"]) if "steps" in row and pd.notna(row.get("steps")) else 0
                    )
                    met_val = (
                        float(row["mets"]) if "mets" in row and pd.notna(row.get("mets")) else 1.0
                    )
                    readings.append(
                        CGMReading(
                            patient_id=str(row["patient_id"]),
                            timestamp=dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt,
                            glucose_mgdl=g_val,
                            heart_rate_bpm=hr_val,
                            steps=step_val,
                            mets=met_val,
                        )
                    )
                return readings

    persona = get_persona_by_id(patient_id)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    ts_df, _ = generate_patient_cgm_series(patient_id, days=days)
    readings = []
    for row in ts_df.iter_rows(named=True):
        readings.append(
            CGMReading(
                patient_id=row["patient_id"],
                timestamp=row["timestamp"],
                glucose_mgdl=row["glucose_mgdl"],
                heart_rate_bpm=row["heart_rate_bpm"],
                steps=row["steps"],
                mets=row["mets"],
            )
        )
    return readings


@router.get("/{patient_id}/personalization")
async def get_patient_personalization(patient_id: str):
    from src.glucotwin.twin.state import twin_state_manager

    try:
        return await twin_state_manager.get_patient_personalization_history(patient_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{patient_id}/personalization/train")
async def trigger_personalization_train(patient_id: str):
    from src.glucotwin.twin.state import twin_state_manager

    try:
        return await twin_state_manager.train_and_evaluate_personalization(patient_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{patient_id}/personalization/rollback")
async def rollback_personalization_adapter(patient_id: str):
    from src.glucotwin.twin.state import twin_state_manager

    try:
        return await twin_state_manager.rollback_adapter(patient_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
