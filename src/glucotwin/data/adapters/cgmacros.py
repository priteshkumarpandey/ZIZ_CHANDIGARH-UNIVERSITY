from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.glucotwin.data.adapters.base import BaseDatasetAdapter
from src.glucotwin.data.schema import (
    validate_meals_df,
    validate_patients_df,
    validate_timeseries_df,
    validate_validation_glucose_df,
)
from src.glucotwin.utils.logging import logger


class CGMacrosAdapter(BaseDatasetAdapter):
    def __init__(self, t2d_hba1c_cutoff: float = 6.5, prediabetes_hba1c_cutoff: float = 5.7):
        self.t2d_cutoff = t2d_hba1c_cutoff
        self.prediabetes_cutoff = prediabetes_hba1c_cutoff

    @property
    def dataset_name(self) -> str:
        return "cgmacros"

    def _find_bio_csv(self, raw_dir: Path) -> Path:
        paths = [
            raw_dir / "cgmacros" / "CGMacros" / "bio.csv",
            raw_dir / "cgmacros" / "bio.csv",
            raw_dir / "CGMacros" / "bio.csv",
            raw_dir / "bio.csv",
        ]
        for p in paths:
            if p.exists():
                return p
        raise FileNotFoundError(f"bio.csv not found in {raw_dir}")

    def _find_cgm_dir(self, raw_dir: Path) -> Path:
        paths = [
            raw_dir / "cgmacros" / "CGMacros",
            raw_dir / "cgmacros",
            raw_dir / "CGMacros",
            raw_dir,
        ]
        for p in paths:
            if (p / "CGMacros-001").exists():
                return p
        return raw_dir

    def load_patients(self, raw_dir: Path) -> pd.DataFrame:
        bio_path = self._find_bio_csv(raw_dir)
        df_raw = pd.read_csv(bio_path)

        # Normalize column names: strip whitespace and resolve duplicates
        cols: list[str] = []
        counts: dict[str, int] = {}
        for c in df_raw.columns:
            clean_c = c.strip()
            if clean_c in counts:
                counts[clean_c] += 1
                cols.append(f"{clean_c}_{counts[clean_c]}")
            else:
                counts[clean_c] = 0
                cols.append(clean_c)
        df_raw.columns = cols

        patients_list: list[dict[str, Any]] = []
        for _, row in df_raw.iterrows():
            subj_int = int(row["subject"])
            patient_id = f"CGMacros-{subj_int:03d}"

            hba1c_val = (
                float(row["A1c PDL (Lab)"]) if pd.notna(row.get("A1c PDL (Lab)")) else np.nan
            )

            # Clinical HbA1c classification for South Asian and international standard priors
            if pd.notna(hba1c_val):
                if hba1c_val >= self.t2d_cutoff:
                    status = "T2D"
                    dtype = "T2D"
                elif hba1c_val >= self.prediabetes_cutoff:
                    status = "prediabetes"
                    dtype = "prediabetes"
                else:
                    status = "healthy"
                    dtype = "none"
            else:
                status = "unknown"
                dtype = "unknown"

            # Height inches to cm: 1 inch = 2.54 cm
            height_in = float(row["Height"]) if pd.notna(row.get("Height")) else np.nan
            height_cm = height_in * 2.54 if pd.notna(height_in) else np.nan

            # Weight lbs to kg: 1 lb = 0.453592 kg
            weight_lbs = float(row["Body weight"]) if pd.notna(row.get("Body weight")) else np.nan
            weight_kg = weight_lbs * 0.453592 if pd.notna(weight_lbs) else np.nan

            pat = {
                "patient_id": patient_id,
                "source_dataset": self.dataset_name,
                "is_synthetic": False,
                "diabetes_type": dtype,
                "diabetes_status": status,
                "age": float(row["Age"]) if pd.notna(row.get("Age")) else np.nan,
                "sex": str(row["Gender"]).strip().upper() if pd.notna(row.get("Gender")) else None,
                "bmi": float(row["BMI"]) if pd.notna(row.get("BMI")) else np.nan,
                "hba1c": hba1c_val,
                "fasting_glucose": float(row["Fasting GLU - PDL (Lab)"])
                if pd.notna(row.get("Fasting GLU - PDL (Lab)"))
                else np.nan,
                "medications": None,
                "systolic_bp": np.nan,
                "diastolic_bp": np.nan,
                "cholesterol": float(row["Cholesterol"])
                if pd.notna(row.get("Cholesterol"))
                else np.nan,
                "triglycerides": float(row["Triglycerides"])
                if pd.notna(row.get("Triglycerides"))
                else np.nan,
                "hdl": float(row["HDL"]) if pd.notna(row.get("HDL")) else np.nan,
                "ldl": float(row["LDL (Cal)"]) if pd.notna(row.get("LDL (Cal)")) else np.nan,
                "creatinine": np.nan,
                "insulin": float(row["Insulin"]) if pd.notna(row.get("Insulin")) else np.nan,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
            }
            patients_list.append(pat)

        df_patients = pd.DataFrame(patients_list)
        return validate_patients_df(df_patients)

    def load_validation_glucose(self, raw_dir: Path) -> pd.DataFrame:
        bio_path = self._find_bio_csv(raw_dir)
        df_raw = pd.read_csv(bio_path)

        cols: list[str] = []
        counts: dict[str, int] = {}
        for c in df_raw.columns:
            clean_c = c.strip()
            if clean_c in counts:
                counts[clean_c] += 1
                cols.append(f"{clean_c}_{counts[clean_c]}")
            else:
                counts[clean_c] = 0
                cols.append(clean_c)
        df_raw.columns = cols

        val_records: list[dict[str, Any]] = []
        for _, row in df_raw.iterrows():
            subj_int = int(row["subject"])
            patient_id = f"CGMacros-{subj_int:03d}"

            for num in [1, 2, 3]:
                col_name = f"#{num} Contour Fingerstick GLU"
                if col_name in df_raw.columns and pd.notna(row[col_name]):
                    val_records.append(
                        {
                            "patient_id": patient_id,
                            "timestamp": pd.to_datetime("2020-01-01 08:00:00", utc=True),
                            "fingerstick_glucose_mgdl": float(row[col_name]),
                            "measurement_number": num,
                        }
                    )

        df_val = pd.DataFrame(val_records)
        return validate_validation_glucose_df(df_val)

    def load_timeseries(self, raw_dir: Path, patient_ids: list[str] | None = None) -> pd.DataFrame:
        cgm_dir = self._find_cgm_dir(raw_dir)
        patient_folders = sorted(
            [d for d in cgm_dir.iterdir() if d.is_dir() and d.name.startswith("CGMacros-")]
        )

        ts_frames: list[pd.DataFrame] = []
        for pfolder in patient_folders:
            pid = pfolder.name
            if patient_ids is not None and pid not in patient_ids:
                continue

            csv_file = pfolder / f"{pid}.csv"
            if not csv_file.exists():
                logger.warning(f"File {csv_file} missing for patient {pid}")
                continue

            df = pd.read_csv(csv_file)
            clean_cols = {c: c.strip() for c in df.columns}
            df = df.rename(columns=clean_cols)

            timestamps = pd.to_datetime(df["Timestamp"], utc=True)

            # Prefer Dexcom G6 over Libre Pro when available
            if "Dexcom GL" in df.columns and "Libre GL" in df.columns:
                glucose = df["Dexcom GL"].combine_first(df["Libre GL"])
            elif "Dexcom GL" in df.columns:
                glucose = df["Dexcom GL"]
            elif "Libre GL" in df.columns:
                glucose = df["Libre GL"]
            else:
                glucose = pd.Series(np.nan, index=df.index)

            hr = df["HR"] if "HR" in df.columns else pd.Series(np.nan, index=df.index)

            # DataDictionary notes METs is multiplied by 10; divide by 10 to recover physiological METs
            if "METs" in df.columns:
                activity = df["METs"] / 10.0
            elif "Calories (Activity)" in df.columns:
                activity = df["Calories (Activity)"]
            else:
                activity = pd.Series(np.nan, index=df.index)

            sub_ts = pd.DataFrame(
                {
                    "patient_id": pid,
                    "timestamp": timestamps,
                    "glucose_mgdl": glucose.astype(float),
                    "heart_rate": hr.astype(float),
                    "steps": np.nan,
                    "activity_level": activity.astype(float),
                    "sleep_stage": None,
                    "hrv": np.nan,
                    "missing_flag": glucose.isna(),
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
        cgm_dir = self._find_cgm_dir(raw_dir)
        patient_folders = sorted(
            [d for d in cgm_dir.iterdir() if d.is_dir() and d.name.startswith("CGMacros-")]
        )

        meal_frames: list[pd.DataFrame] = []
        for pfolder in patient_folders:
            pid = pfolder.name
            if patient_ids is not None and pid not in patient_ids:
                continue

            csv_file = pfolder / f"{pid}.csv"
            if not csv_file.exists():
                continue

            df = pd.read_csv(csv_file)
            clean_cols = {c: c.strip() for c in df.columns}
            df = df.rename(columns=clean_cols)

            meal_mask = df["Meal Type"].notna() | (df["Carbs"].notna() & (df["Carbs"] > 0))
            meal_rows = df[meal_mask].copy()

            if meal_rows.empty:
                continue

            timestamps = pd.to_datetime(meal_rows["Timestamp"], utc=True)
            carbs = meal_rows["Carbs"].fillna(0.0).astype(float)
            protein = (
                meal_rows["Protein"].astype(float) if "Protein" in meal_rows.columns else np.nan
            )
            fat = meal_rows["Fat"].astype(float) if "Fat" in meal_rows.columns else np.nan
            fiber = meal_rows["Fiber"].astype(float) if "Fiber" in meal_rows.columns else np.nan
            cals = (
                meal_rows["Calories"].astype(float) if "Calories" in meal_rows.columns else np.nan
            )
            label = meal_rows["Meal Type"].astype(str) if "Meal Type" in meal_rows.columns else None

            sub_meals = pd.DataFrame(
                {
                    "patient_id": pid,
                    "timestamp": timestamps,
                    "carbs_g": carbs,
                    "protein_g": protein,
                    "fat_g": fat,
                    "fiber_g": fiber,
                    "calories": cals,
                    "meal_label": label,
                }
            )
            meal_frames.append(sub_meals)

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
