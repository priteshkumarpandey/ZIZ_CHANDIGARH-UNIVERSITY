from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindowedDataset:
    patient_ids: np.ndarray
    window_timestamps: np.ndarray
    observed_sequences: np.ndarray
    observed_feature_names: list[str]
    known_future_sequences: np.ndarray
    known_future_feature_names: list[str]
    static_features: np.ndarray
    static_feature_names: list[str]
    tabular_features: np.ndarray
    tabular_feature_names: list[str]
    continuous_targets: dict[int, np.ndarray]
    binary_event_labels: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.patient_ids)


class FeatureWindowBuilder:
    def __init__(
        self,
        lookback_minutes: int = 360,
        horizons_minutes: list[int] | None = None,
        resample_interval_minutes: int = 5,
        hyperglycemia_threshold: float = 180.0,
        severe_hyperglycemia_threshold: float = 250.0,
        hypoglycemia_threshold: float = 70.0,
        severe_hypoglycemia_threshold: float = 54.0,
        spike_threshold_mgdl: float = 40.0,
        spike_window_minutes: int = 60,
        max_missing_ratio_in_lookback: float = 0.25,
    ):
        self.lookback_minutes = lookback_minutes
        self.horizons_minutes = sorted(
            horizons_minutes if horizons_minutes is not None else [30, 60, 120]
        )
        self.step_mins = resample_interval_minutes
        self.lookback_steps = lookback_minutes // resample_interval_minutes
        self.horizon_steps = {h: h // resample_interval_minutes for h in self.horizons_minutes}
        self.max_horizon_steps = max(self.horizon_steps.values())
        self.spike_steps = spike_window_minutes // resample_interval_minutes

        self.hyper_thresh = hyperglycemia_threshold
        self.severe_hyper_thresh = severe_hyperglycemia_threshold
        self.hypo_thresh = hypoglycemia_threshold
        self.severe_hypo_thresh = severe_hypoglycemia_threshold
        self.spike_thresh = spike_threshold_mgdl
        self.max_missing_ratio = max_missing_ratio_in_lookback

        self.static_cols = [
            "age",
            "bmi",
            "hba1c",
            "fasting_glucose",
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
            "sex_encoded",
            "is_synthetic_val",
        ]

        self.observed_cols = [
            "glucose_mgdl",
            "glucose_diff_5m",
            "glucose_diff_15m",
            "glucose_diff_30m",
            "glucose_diff_60m",
            "glucose_roll_mean_30m",
            "glucose_roll_std_30m",
            "heart_rate",
            "steps",
            "activity_level",
            "carbs_g",
        ]

        self.known_future_cols = [
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "time_since_meal_mins",
            "carbs_last_2h",
        ]

    def _prepare_patient_timeseries(
        self,
        df_patient_ts: pd.DataFrame,
        df_patient_meals: pd.DataFrame,
    ) -> pd.DataFrame:
        df = df_patient_ts.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if not df_patient_meals.empty:
            meals = df_patient_meals.copy()
            meals["timestamp"] = pd.to_datetime(meals["timestamp"], utc=True)
            meals = meals.sort_values("timestamp")
            meals["carbs_meal"] = meals["carbs_g"].fillna(0.0)

            meals_5m = (
                meals.set_index("timestamp")["carbs_meal"].resample(f"{self.step_mins}min").sum()
            )
            df = df.merge(meals_5m.rename("carbs_g"), on="timestamp", how="left")
            df["carbs_g"] = df["carbs_g"].fillna(0.0)
        else:
            df["carbs_g"] = 0.0

        df["glucose_diff_5m"] = df["glucose_mgdl"].diff(1)
        df["glucose_diff_15m"] = df["glucose_mgdl"].diff(3)
        df["glucose_diff_30m"] = df["glucose_mgdl"].diff(6)
        df["glucose_diff_60m"] = df["glucose_mgdl"].diff(12)

        df["glucose_roll_mean_30m"] = df["glucose_mgdl"].rolling(window=6, min_periods=1).mean()
        df["glucose_roll_std_30m"] = (
            df["glucose_mgdl"].rolling(window=6, min_periods=1).std().fillna(0.0)
        )

        df["heart_rate"] = df["heart_rate"].ffill().bfill().fillna(72.0)
        df["steps"] = df["steps"].fillna(0.0)
        df["activity_level"] = df["activity_level"].fillna(0.0)

        hours = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        days = df["timestamp"].dt.dayofweek
        df["day_sin"] = np.sin(2 * np.pi * days / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * days / 7.0)

        has_meal = df["carbs_g"] > 0
        meal_ts_series = df["timestamp"].where(has_meal).ffill()
        elapsed_mins = (df["timestamp"] - meal_ts_series).dt.total_seconds() / 60.0
        df["time_since_meal_mins"] = elapsed_mins.fillna(720.0).clip(upper=720.0)

        df["carbs_last_2h"] = df["carbs_g"].rolling(window=24, min_periods=1).sum()

        return df

    def _extract_static_vector(self, patient_row: pd.Series | None) -> np.ndarray:
        if patient_row is None or patient_row.empty:
            return np.zeros(len(self.static_cols), dtype=np.float32)

        sex_val = 1.0 if str(patient_row.get("sex", "")).upper().startswith("M") else 0.0
        is_synth = 1.0 if bool(patient_row.get("is_synthetic", False)) else 0.0

        vec = [
            float(patient_row.get("age", np.nan)),
            float(patient_row.get("bmi", np.nan)),
            float(patient_row.get("hba1c", np.nan)),
            float(patient_row.get("fasting_glucose", np.nan)),
            float(patient_row.get("systolic_bp", np.nan)),
            float(patient_row.get("diastolic_bp", np.nan)),
            float(patient_row.get("cholesterol", np.nan)),
            float(patient_row.get("triglycerides", np.nan)),
            float(patient_row.get("hdl", np.nan)),
            float(patient_row.get("ldl", np.nan)),
            float(patient_row.get("creatinine", np.nan)),
            float(patient_row.get("insulin", np.nan)),
            float(patient_row.get("height_cm", np.nan)),
            float(patient_row.get("weight_kg", np.nan)),
            sex_val,
            is_synth,
        ]
        return np.nan_to_num(np.array(vec, dtype=np.float32), nan=0.0)

    def build_dataset(
        self,
        df_patients: pd.DataFrame,
        df_timeseries: pd.DataFrame,
        df_meals: pd.DataFrame,
    ) -> WindowedDataset:
        patients_dict = (
            {row["patient_id"]: row for _, row in df_patients.iterrows()}
            if not df_patients.empty
            else {}
        )

        all_patient_ids: list[str] = []
        all_timestamps: list[np.datetime64] = []
        all_observed_seqs: list[np.ndarray] = []
        all_known_future_seqs: list[np.ndarray] = []
        all_static_vecs: list[np.ndarray] = []
        all_tabular_rows: list[np.ndarray] = []

        targets_dict: dict[int, list[float]] = {h: [] for h in self.horizons_minutes}
        events_dict: dict[str, list[int]] = {}
        for h in self.horizons_minutes:
            events_dict[f"hyperglycemia_{h}m"] = []
            events_dict[f"severe_hyperglycemia_{h}m"] = []
            events_dict[f"hypoglycemia_{h}m"] = []
        events_dict["spike_60m"] = []

        min_required_steps = self.lookback_steps + self.max_horizon_steps
        grouped_ts = df_timeseries.groupby("patient_id")

        for pid, group in grouped_ts:
            if len(group) < min_required_steps:
                continue

            p_meals = (
                df_meals[df_meals["patient_id"] == pid] if not df_meals.empty else pd.DataFrame()
            )
            patient_row = patients_dict.get(pid, None)
            static_vec = self._extract_static_vector(patient_row)

            p_df = self._prepare_patient_timeseries(group, p_meals)
            n_rows = len(p_df)

            if n_rows < min_required_steps:
                continue

            obs_matrix = np.nan_to_num(p_df[self.observed_cols].to_numpy(dtype=np.float32), nan=0.0)
            fut_matrix = np.nan_to_num(
                p_df[self.known_future_cols].to_numpy(dtype=np.float32), nan=0.0
            )
            glucose_arr = p_df["glucose_mgdl"].to_numpy(dtype=np.float32)
            timestamps_arr = p_df["timestamp"].to_numpy()

            for i in range(self.lookback_steps - 1, n_rows - self.max_horizon_steps):
                current_glucose = glucose_arr[i]
                if np.isnan(current_glucose):
                    continue

                lookback_glu = glucose_arr[i - self.lookback_steps + 1 : i + 1]
                missing_ratio = np.isnan(lookback_glu).mean()
                if missing_ratio > self.max_missing_ratio:
                    continue

                future_valid = True
                curr_targets = {}
                for h in self.horizons_minutes:
                    step = self.horizon_steps[h]
                    future_val = glucose_arr[i + step]
                    if np.isnan(future_val):
                        future_valid = False
                        break
                    curr_targets[h] = future_val

                if not future_valid:
                    continue

                obs_window = obs_matrix[i - self.lookback_steps + 1 : i + 1, :]
                fut_window = fut_matrix[i + 1 : i + 1 + self.max_horizon_steps, :]

                tab_row = np.concatenate(
                    [
                        obs_matrix[i, :],
                        obs_matrix[i - 6, :3] if i >= 6 else obs_matrix[i, :3],
                        obs_matrix[i - 12, :3] if i >= 12 else obs_matrix[i, :3],
                        static_vec,
                    ]
                )

                all_patient_ids.append(pid)
                all_timestamps.append(timestamps_arr[i])
                all_observed_seqs.append(obs_window)
                all_known_future_seqs.append(fut_window)
                all_static_vecs.append(static_vec)
                all_tabular_rows.append(tab_row)

                for h in self.horizons_minutes:
                    targets_dict[h].append(curr_targets[h])
                    gh = curr_targets[h]
                    events_dict[f"hyperglycemia_{h}m"].append(1 if gh >= self.hyper_thresh else 0)
                    events_dict[f"severe_hyperglycemia_{h}m"].append(
                        1 if gh >= self.severe_hyper_thresh else 0
                    )
                    events_dict[f"hypoglycemia_{h}m"].append(1 if gh <= self.hypo_thresh else 0)

                spike_window = glucose_arr[i + 1 : i + 1 + self.spike_steps]
                is_spike = (
                    1 if (np.nanmax(spike_window) - current_glucose) >= self.spike_thresh else 0
                )
                events_dict["spike_60m"].append(is_spike)

        tabular_feature_names = (
            self.observed_cols
            + [f"{c}_lag30m" for c in self.observed_cols[:3]]
            + [f"{c}_lag60m" for c in self.observed_cols[:3]]
            + self.static_cols
        )

        return WindowedDataset(
            patient_ids=np.array(all_patient_ids, dtype=object),
            window_timestamps=np.array(all_timestamps),
            observed_sequences=np.array(all_observed_seqs, dtype=np.float32)
            if all_observed_seqs
            else np.zeros((0, self.lookback_steps, len(self.observed_cols)), dtype=np.float32),
            observed_feature_names=self.observed_cols,
            known_future_sequences=np.array(all_known_future_seqs, dtype=np.float32)
            if all_known_future_seqs
            else np.zeros(
                (0, self.max_horizon_steps, len(self.known_future_cols)), dtype=np.float32
            ),
            known_future_feature_names=self.known_future_cols,
            static_features=np.array(all_static_vecs, dtype=np.float32)
            if all_static_vecs
            else np.zeros((0, len(self.static_cols)), dtype=np.float32),
            static_feature_names=self.static_cols,
            tabular_features=np.array(all_tabular_rows, dtype=np.float32)
            if all_tabular_rows
            else np.zeros((0, len(tabular_feature_names)), dtype=np.float32),
            tabular_feature_names=tabular_feature_names,
            continuous_targets={
                h: np.array(targets_dict[h], dtype=np.float32) for h in self.horizons_minutes
            },
            binary_event_labels={k: np.array(v, dtype=np.int64) for k, v in events_dict.items()},
        )
