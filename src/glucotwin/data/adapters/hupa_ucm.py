import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.glucotwin.data.adapters.base import BaseDatasetAdapter
from src.glucotwin.data.schema import (
    validate_meals_df,
    validate_patients_df,
    validate_timeseries_df,
)
from src.glucotwin.utils.logging import logger


class HUPAAdapter(BaseDatasetAdapter):
    def __init__(self):
        self.skipped_empty_files_count = 0

    @property
    def dataset_name(self) -> str:
        return "hupa_ucm"

    def _find_raw_data_dir(self, raw_dir: Path) -> Path:
        paths = [
            raw_dir / "hupa_ucm" / "Raw_Data",
            raw_dir / "Raw_Data",
            raw_dir / "hupa_ucm",
            raw_dir,
        ]
        for p in paths:
            if p.exists() and any(p.glob("HUPA*P")):
                return p
        return raw_dir

    def load_patients(self, raw_dir: Path) -> pd.DataFrame:
        raw_data_dir = self._find_raw_data_dir(raw_dir)
        patient_folders = sorted(
            [d for d in raw_data_dir.iterdir() if d.is_dir() and re.match(r"HUPA\d+P", d.name)]
        )

        patients_list: list[dict[str, Any]] = []
        for pdir in patient_folders:
            pid = pdir.name
            # HUPA-UCM dataset represents Type 1 Diabetes cohort from Hospital Universitario Principe de Asturias
            patients_list.append(
                {
                    "patient_id": pid,
                    "source_dataset": self.dataset_name,
                    "is_synthetic": False,
                    "diabetes_type": "T1D",
                    "diabetes_status": "T1D",
                    "age": np.nan,
                    "sex": None,
                    "bmi": np.nan,
                    "hba1c": np.nan,
                    "fasting_glucose": np.nan,
                    "medications": ["Insulin"],
                    "systolic_bp": np.nan,
                    "diastolic_bp": np.nan,
                    "cholesterol": np.nan,
                    "triglycerides": np.nan,
                    "hdl": np.nan,
                    "ldl": np.nan,
                    "creatinine": np.nan,
                    "insulin": np.nan,
                    "height_cm": np.nan,
                    "weight_kg": np.nan,
                }
            )

        if not patients_list:
            empty_df = pd.DataFrame(
                columns=[
                    "patient_id",
                    "source_dataset",
                    "is_synthetic",
                    "diabetes_type",
                    "diabetes_status",
                    "age",
                    "sex",
                    "bmi",
                    "hba1c",
                    "fasting_glucose",
                    "medications",
                    "systolic_bp",
                    "diastolic_bp",
                    "cholesterol",
                    "triglycerides",
                    "hdl",
                    "ldl",
                    "creatinine",
                    "insulin",
                    "height_cm",
                    "weight_kg",
                ]
            )
            return validate_patients_df(empty_df)

        df_patients = pd.DataFrame(patients_list)
        return validate_patients_df(df_patients)

    def _read_cgm_file(self, cgm_file: Path) -> pd.DataFrame | None:
        if cgm_file.stat().st_size < 25:
            self.skipped_empty_files_count += 1
            return None

        with open(cgm_file, encoding="latin-1") as fp:
            lines = [fp.readline(), fp.readline()]

        if not lines or not lines[0].strip():
            self.skipped_empty_files_count += 1
            return None

        sample = lines[1] if len(lines) > 1 and lines[1].strip() else lines[0]
        sep = "\t" if "\t" in sample else ";"

        try:
            df = pd.read_csv(cgm_file, skiprows=1, sep=sep, encoding="latin-1", on_bad_lines="skip")
        except Exception as e:
            logger.warning(f"Error reading {cgm_file}: {e}")
            return None

        if df.empty or "Hora" not in df.columns:
            return None
        return df

    def load_timeseries(self, raw_dir: Path, patient_ids: list[str] | None = None) -> pd.DataFrame:
        raw_data_dir = self._find_raw_data_dir(raw_dir)
        patient_folders = sorted(
            [d for d in raw_data_dir.iterdir() if d.is_dir() and re.match(r"HUPA\d+P", d.name)]
        )

        ts_frames: list[pd.DataFrame] = []

        for pdir in patient_folders:
            pid = pdir.name
            if patient_ids is not None and pid not in patient_ids:
                continue

            cgm_dir = pdir / "free_style_sensor"
            if not cgm_dir.exists():
                continue

            cgm_files = sorted(cgm_dir.glob("*.csv"))
            for cgm_file in cgm_files:
                df_cgm = self._read_cgm_file(cgm_file)
                if df_cgm is None or df_cgm.empty:
                    continue

                hist_cols = [c for c in df_cgm.columns if "Hist" in c and "glucosa" in c]
                scan_cols = [
                    c
                    for c in df_cgm.columns
                    if "Glucosa" in c and "mg/dL" in c and "Hist" not in c and "tira" not in c
                ]

                if hist_cols and scan_cols:
                    glucose = pd.to_numeric(df_cgm[hist_cols[0]], errors="coerce").combine_first(
                        pd.to_numeric(df_cgm[scan_cols[0]], errors="coerce")
                    )
                elif hist_cols:
                    glucose = pd.to_numeric(df_cgm[hist_cols[0]], errors="coerce")
                elif scan_cols:
                    glucose = pd.to_numeric(df_cgm[scan_cols[0]], errors="coerce")
                else:
                    glucose = pd.Series(np.nan, index=df_cgm.index)

                timestamps = pd.to_datetime(
                    df_cgm["Hora"], format="mixed", utc=True, errors="coerce"
                )

                valid_mask = timestamps.notna()
                sub_ts = pd.DataFrame(
                    {
                        "patient_id": pid,
                        "timestamp": timestamps[valid_mask],
                        "glucose_mgdl": glucose[valid_mask].astype(float),
                        "heart_rate": np.nan,
                        "steps": np.nan,
                        "activity_level": np.nan,
                        "sleep_stage": None,
                        "hrv": np.nan,
                        "missing_flag": glucose[valid_mask].isna(),
                    }
                )
                ts_frames.append(sub_ts)

        if not ts_frames:
            empty_df = pd.DataFrame(
                columns=[
                    "patient_id",
                    "timestamp",
                    "glucose_mgdl",
                    "heart_rate",
                    "steps",
                    "activity_level",
                    "sleep_stage",
                    "hrv",
                    "missing_flag",
                ]
            )
            return validate_timeseries_df(empty_df)

        full_ts = pd.concat(ts_frames, ignore_index=True)
        return validate_timeseries_df(full_ts)

    def load_meals(self, raw_dir: Path, patient_ids: list[str] | None = None) -> pd.DataFrame:
        raw_data_dir = self._find_raw_data_dir(raw_dir)
        patient_folders = sorted(
            [d for d in raw_data_dir.iterdir() if d.is_dir() and re.match(r"HUPA\d+P", d.name)]
        )

        meal_frames: list[pd.DataFrame] = []

        for pdir in patient_folders:
            pid = pdir.name
            if patient_ids is not None and pid not in patient_ids:
                continue

            cgm_dir = pdir / "free_style_sensor"
            if not cgm_dir.exists():
                continue

            for cgm_file in sorted(cgm_dir.glob("*.csv")):
                df_cgm = self._read_cgm_file(cgm_file)
                if df_cgm is None or df_cgm.empty:
                    continue

                carb_cols = [c for c in df_cgm.columns if "Carbohidratos" in c]
                if not carb_cols:
                    continue

                carb_col = carb_cols[0]
                raw_carb = pd.to_numeric(
                    df_cgm[carb_col].astype(str).str.replace(",", "."), errors="coerce"
                )

                meal_mask = raw_carb.notna() & (raw_carb > 0)
                if not meal_mask.any():
                    continue

                sub_meals_df = df_cgm[meal_mask].copy()
                timestamps = pd.to_datetime(
                    sub_meals_df["Hora"], format="mixed", utc=True, errors="coerce"
                )

                # Spanish diabetic exchange units: 1 ración corresponds to 10 grams of carbohydrate
                if "raciones" in carb_col.lower():
                    carbs_g = raw_carb[meal_mask] * 10.0
                else:
                    carbs_g = raw_carb[meal_mask]

                valid_ts = timestamps.notna()
                sub_res = pd.DataFrame(
                    {
                        "patient_id": pid,
                        "timestamp": timestamps[valid_ts],
                        "carbs_g": carbs_g[valid_ts].astype(float),
                        "protein_g": np.nan,
                        "fat_g": np.nan,
                        "fiber_g": np.nan,
                        "calories": np.nan,
                        "meal_label": "Recorded Meal",
                    }
                )
                meal_frames.append(sub_res)

        if not meal_frames:
            empty_df = pd.DataFrame(
                columns=[
                    "patient_id",
                    "timestamp",
                    "carbs_g",
                    "protein_g",
                    "fat_g",
                    "fiber_g",
                    "calories",
                    "meal_label",
                ]
            )
            return validate_meals_df(empty_df)

        full_meals = pd.concat(meal_frames, ignore_index=True)
        return validate_meals_df(full_meals)
