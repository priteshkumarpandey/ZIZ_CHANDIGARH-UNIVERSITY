from typing import Any

import numpy as np
import pandas as pd

from src.glucotwin.data.schema import (
    validate_timeseries_df,
)


class Harmonizer:
    def __init__(
        self,
        resample_interval_minutes: int = 5,
        max_interpolation_minutes: int = 30,
        glucose_min_mgdl: float = 40.0,
        glucose_max_mgdl: float = 500.0,
        heart_rate_min: float = 30.0,
        heart_rate_max: float = 220.0,
    ):
        self.resample_mins = resample_interval_minutes
        self.max_interp_mins = max_interpolation_minutes
        self.glucose_min = glucose_min_mgdl
        self.glucose_max = glucose_max_mgdl
        self.hr_min = heart_rate_min
        self.hr_max = heart_rate_max
        self.max_consecutive_interp = max_interpolation_minutes // resample_interval_minutes

    def harmonize_timeseries(
        self,
        df_ts: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if df_ts.empty:
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
            return validate_timeseries_df(empty_df), {
                "raw_rows": 0,
                "harmonized_rows": 0,
                "dropped_glucose_out_of_range": 0,
                "dropped_hr_out_of_range": 0,
                "duplicate_timestamps_removed": 0,
                "interpolated_values_count": 0,
                "final_missingness_pct": 0.0,
                "gap_count_short_le_30min": 0,
                "gap_count_long_gt_30min": 0,
            }

        df = df_ts.copy()
        raw_rows = len(df)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Drop physiologically impossible values
        glucose_out_mask = (df["glucose_mgdl"].notna()) & (
            (df["glucose_mgdl"] < self.glucose_min) | (df["glucose_mgdl"] > self.glucose_max)
        )
        dropped_glu = int(glucose_out_mask.sum())
        df.loc[glucose_out_mask, "glucose_mgdl"] = np.nan

        hr_out_mask = (df["heart_rate"].notna()) & (
            (df["heart_rate"] < self.hr_min) | (df["heart_rate"] > self.hr_max)
        )
        dropped_hr = int(hr_out_mask.sum())
        df.loc[hr_out_mask, "heart_rate"] = np.nan

        harmonized_patients: list[pd.DataFrame] = []
        dup_removed_total = 0
        interpolated_total = 0
        short_gaps_total = 0
        long_gaps_total = 0

        grouped = df.groupby("patient_id")
        for pid, group in grouped:
            # Sort and deduplicate timestamps
            initial_count = len(group)
            group_sorted = group.sort_values("timestamp")
            group_dedup = group_sorted.drop_duplicates(subset=["timestamp"], keep="last")
            dup_removed_total += initial_count - len(group_dedup)

            if group_dedup.empty:
                continue

            # Check if this is a single static summary timestamp dataset
            if len(group_dedup) == 1:
                row = group_dedup.iloc[0].to_dict()
                row["missing_flag"] = pd.isna(row["glucose_mgdl"])
                harmonized_patients.append(pd.DataFrame([row]))
                continue

            t_start = group_dedup["timestamp"].min().floor(f"{self.resample_mins}min")
            t_end = group_dedup["timestamp"].max().ceil(f"{self.resample_mins}min")

            regular_index = pd.date_range(
                start=t_start,
                end=t_end,
                freq=f"{self.resample_mins}min",
                tz="UTC",
                name="timestamp",
            )

            # Reindex to regular 5-minute grid
            # Match nearest observed readings within the resampling window tolerance
            group_indexed = group_dedup.set_index("timestamp")

            # Resample numerical and categorical channels
            resampled_glu = (
                group_indexed["glucose_mgdl"].resample(f"{self.resample_mins}min").mean()
            )
            resampled_hr = group_indexed["heart_rate"].resample(f"{self.resample_mins}min").mean()
            resampled_steps = (
                group_indexed["steps"].resample(f"{self.resample_mins}min").sum(min_count=1)
            )
            resampled_act = (
                group_indexed["activity_level"].resample(f"{self.resample_mins}min").mean()
            )
            resampled_hrv = group_indexed["hrv"].resample(f"{self.resample_mins}min").mean()
            resampled_sleep = (
                group_indexed["sleep_stage"].resample(f"{self.resample_mins}min").last()
            )

            patient_grid = pd.DataFrame(index=regular_index)
            patient_grid["glucose_mgdl"] = resampled_glu
            patient_grid["heart_rate"] = resampled_hr
            patient_grid["steps"] = resampled_steps
            patient_grid["activity_level"] = resampled_act
            patient_grid["sleep_stage"] = resampled_sleep
            patient_grid["hrv"] = resampled_hrv

            # Calculate gap statistics on glucose
            isna_series = patient_grid["glucose_mgdl"].isna()
            gap_blocks = (~isna_series).cumsum()[isna_series]
            if not gap_blocks.empty:
                gap_lengths = gap_blocks.value_counts()
                short_gaps_total += int((gap_lengths <= self.max_consecutive_interp).sum())
                long_gaps_total += int((gap_lengths > self.max_consecutive_interp).sum())

            # Clinical safety rule: interpolate only physiological gaps up to 30 minutes
            pre_interp_nans = int(patient_grid["glucose_mgdl"].isna().sum())
            interpolated_glu = patient_grid["glucose_mgdl"].interpolate(
                method="linear",
                limit=self.max_consecutive_interp,
                limit_direction="forward",
            )
            post_interp_nans = int(interpolated_glu.isna().sum())
            interpolated_total += pre_interp_nans - post_interp_nans

            patient_grid["glucose_mgdl"] = interpolated_glu
            patient_grid["missing_flag"] = patient_grid["glucose_mgdl"].isna()
            patient_grid["patient_id"] = str(pid)
            patient_grid = patient_grid.reset_index()

            harmonized_patients.append(patient_grid)

        if not harmonized_patients:
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
            return validate_timeseries_df(empty_df), {}

        full_harm = pd.concat(harmonized_patients, ignore_index=True)
        full_harm = full_harm[
            [
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
        ]

        final_missing_pct = float(full_harm["glucose_mgdl"].isna().mean() * 100.0)

        stats = {
            "raw_rows": raw_rows,
            "harmonized_rows": len(full_harm),
            "dropped_glucose_out_of_range": dropped_glu,
            "dropped_hr_out_of_range": dropped_hr,
            "duplicate_timestamps_removed": dup_removed_total,
            "interpolated_values_count": interpolated_total,
            "final_missingness_pct": round(final_missing_pct, 2),
            "gap_count_short_le_30min": short_gaps_total,
            "gap_count_long_gt_30min": long_gaps_total,
        }

        return validate_timeseries_df(full_harm), stats

    def generate_quality_report(
        self,
        dataset_name: str,
        df_patients: pd.DataFrame,
        df_timeseries: pd.DataFrame,
        df_meals: pd.DataFrame,
        ts_stats: dict[str, Any] | None,
        skipped_files: int = 0,
        patients_with_windows: int | None = None,
        zero_window_patients_reasons: dict[str, str] | None = None,
        timeseries_quality_note: str | None = None,
    ) -> dict[str, Any]:
        status_counts = (
            df_patients["diabetes_status"].value_counts().to_dict() if not df_patients.empty else {}
        )
        dtype_counts = (
            df_patients["diabetes_type"].value_counts().to_dict() if not df_patients.empty else {}
        )

        report: dict[str, Any] = {
            "dataset_name": dataset_name,
            "patient_count": len(df_patients),
            "status_breakdown": {
                "T2D": int(status_counts.get("T2D", 0)),
                "prediabetes": int(status_counts.get("prediabetes", 0)),
                "healthy": int(status_counts.get("healthy", 0)),
                "T1D": int(status_counts.get("T1D", 0)),
                "unknown": int(status_counts.get("unknown", 0)),
            },
            "diabetes_type_breakdown": {k: int(v) for k, v in dtype_counts.items()},
            "meals_count": len(df_meals),
            "skipped_or_empty_files_count": skipped_files,
        }

        if patients_with_windows is not None:
            report["patients_with_windows"] = patients_with_windows
        if zero_window_patients_reasons is not None:
            report["zero_window_patients_reasons"] = zero_window_patients_reasons
        if timeseries_quality_note is not None:
            report["timeseries_quality_note"] = timeseries_quality_note

        report["timeseries_quality"] = ts_stats
        return report


def compute_patient_quality_dataframe(
    patient_sources: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    patient_window_counts: dict[str, int],
    max_consecutive_interp: int = 6,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for src_name, df_pat, df_raw_ts, df_harm_ts in patient_sources:
        for pid in df_pat["patient_id"]:
            p_raw = (
                df_raw_ts[df_raw_ts["patient_id"] == pid] if not df_raw_ts.empty else pd.DataFrame()
            )
            p_harm = (
                df_harm_ts[df_harm_ts["patient_id"] == pid]
                if not df_harm_ts.empty
                else pd.DataFrame()
            )

            raw_rows = len(p_raw)
            harm_rows = len(p_harm)

            if harm_rows > 0:
                missing_pct = float(round(p_harm["glucose_mgdl"].isna().mean() * 100.0, 2))
                isna_series = p_harm["glucose_mgdl"].isna()
                gap_blocks = (~isna_series).cumsum()[isna_series]
                if not gap_blocks.empty:
                    gap_lengths = gap_blocks.value_counts()
                    short_gaps = int((gap_lengths <= max_consecutive_interp).sum())
                    long_gaps = int((gap_lengths > max_consecutive_interp).sum())
                else:
                    short_gaps = 0
                    long_gaps = 0
            else:
                missing_pct = 100.0
                short_gaps = 0
                long_gaps = 0

            window_count = int(patient_window_counts.get(pid, 0))

            records.append(
                {
                    "patient_id": str(pid),
                    "source_dataset": str(src_name),
                    "raw_rows": raw_rows,
                    "harmonized_rows": harm_rows,
                    "missingness_pct": missing_pct,
                    "short_gaps_le_30m": short_gaps,
                    "long_gaps_gt_30m": long_gaps,
                    "window_count": window_count,
                }
            )

    return pd.DataFrame(records)
