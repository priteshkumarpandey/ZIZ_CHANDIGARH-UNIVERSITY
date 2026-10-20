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

CONCEPT_MAP = {
    4245997: "bmi",
    3012888: "diastolic_bp",
    3004249: "systolic_bp",
    3036277: "height_cm",
    3025315: "weight_kg",
    3004410: "hba1c",
    3004501: "fasting_glucose",
    3007070: "hdl",
    3028288: "ldl",
    3027114: "cholesterol",
    3022192: "triglycerides",
    3016723: "creatinine",
    3016244: "insulin",
}


class AIREADISyntheticAdapter(BaseDatasetAdapter):
    def __init__(self, t2d_hba1c_cutoff: float = 6.5, prediabetes_hba1c_cutoff: float = 5.7):
        self.t2d_cutoff = t2d_hba1c_cutoff
        self.prediabetes_cutoff = prediabetes_hba1c_cutoff

    @property
    def dataset_name(self) -> str:
        return "ai_readi_synthetic"

    def _find_file(self, raw_dir: Path, filename: str) -> Path:
        paths = [
            raw_dir / "ai_readi_synthetic" / filename,
            raw_dir / filename,
        ]
        for p in paths:
            if p.exists():
                return p
        raise FileNotFoundError(f"{filename} not found in {raw_dir}")

    def load_patients(self, raw_dir: Path) -> pd.DataFrame:
        meas_file = self._find_file(raw_dir, "measurement.csv")
        cond_file = self._find_file(raw_dir, "condition_occurrence.csv")

        cond_df = pd.read_csv(
            cond_file, usecols=["person_id", "condition_concept_id", "condition_source_value"]
        )

        t2d_persons: set[int] = set(cond_df[cond_df["condition_concept_id"] == 201826]["person_id"])
        predm_persons: set[int] = set(
            cond_df[cond_df["condition_concept_id"] == 37018196]["person_id"]
        )

        meas_chunks = pd.read_csv(
            meas_file,
            usecols=["person_id", "measurement_concept_id", "value_as_number"],
            chunksize=100000,
        )

        filtered_chunks = []
        for chunk in meas_chunks:
            sub = chunk[chunk["measurement_concept_id"].isin(CONCEPT_MAP.keys())]
            filtered_chunks.append(sub)

        meas_filtered = pd.concat(filtered_chunks, ignore_index=True)
        meas_filtered["feature_name"] = meas_filtered["measurement_concept_id"].map(CONCEPT_MAP)

        pivoted = meas_filtered.pivot_table(
            index="person_id", columns="feature_name", values="value_as_number", aggfunc="mean"
        ).reset_index()

        all_person_ids = sorted(set(cond_df["person_id"]).union(set(pivoted["person_id"])))
        patients_list: list[dict[str, Any]] = []

        pivoted_indexed = pivoted.set_index("person_id")

        for pid in all_person_ids:
            row_data = pivoted_indexed.loc[pid].to_dict() if pid in pivoted_indexed.index else {}
            hba1c_val = float(row_data.get("hba1c", np.nan))

            # Prioritize explicit OMOP clinical diagnosis concept over HbA1c measurement
            if pid in t2d_persons:
                status = "T2D"
                dtype = "T2D"
            elif pid in predm_persons:
                status = "prediabetes"
                dtype = "prediabetes"
            elif pd.notna(hba1c_val):
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

            patient_record = {
                "patient_id": f"AIREADI-{pid:05d}",
                "source_dataset": self.dataset_name,
                "is_synthetic": True,
                "diabetes_type": dtype,
                "diabetes_status": status,
                "age": np.nan,
                "sex": None,
                "bmi": float(row_data.get("bmi", np.nan)),
                "hba1c": hba1c_val,
                "fasting_glucose": float(row_data.get("fasting_glucose", np.nan)),
                "medications": None,
                "systolic_bp": float(row_data.get("systolic_bp", np.nan)),
                "diastolic_bp": float(row_data.get("diastolic_bp", np.nan)),
                "cholesterol": float(row_data.get("cholesterol", np.nan)),
                "triglycerides": float(row_data.get("triglycerides", np.nan)),
                "hdl": float(row_data.get("hdl", np.nan)),
                "ldl": float(row_data.get("ldl", np.nan)),
                "creatinine": float(row_data.get("creatinine", np.nan)),
                "insulin": float(row_data.get("insulin", np.nan)),
                "height_cm": float(row_data.get("height_cm", np.nan)),
                "weight_kg": float(row_data.get("weight_kg", np.nan)),
            }
            patients_list.append(patient_record)

        df_patients = pd.DataFrame(patients_list)
        return validate_patients_df(df_patients)

    def load_timeseries(self, raw_dir: Path, patient_ids: list[str] | None = None) -> pd.DataFrame:
        # AI-READI synthetic contains cross-sectional wearable summaries and no continuous dynamic timeseries
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

    def load_cross_sectional_summary(
        self, raw_dir: Path, patient_ids: list[str] | None = None
    ) -> pd.DataFrame:
        tsv_file = self._find_file(raw_dir, "synthetic_wearable_activity_glucose.tsv")
        df_tsv = pd.read_csv(tsv_file, sep="\t")
        df_tsv["patient_id"] = df_tsv["participant_id"].apply(lambda pid: f"AIREADI-{int(pid):05d}")
        if patient_ids is not None:
            df_tsv = df_tsv[df_tsv["patient_id"].isin(patient_ids)]
        return df_tsv

    def load_meals(self, raw_dir: Path, patient_ids: list[str] | None = None) -> pd.DataFrame:
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
