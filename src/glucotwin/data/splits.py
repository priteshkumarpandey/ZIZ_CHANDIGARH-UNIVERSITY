import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.glucotwin.features.windowing import WindowedDataset


class PatientSplitter:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def split_patients_stratified(
        self,
        df_patients: pd.DataFrame,
        patient_window_counts: dict[str, int] | None = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
        rng = np.random.RandomState(self.random_seed)

        # Restrict split population strictly to patients having at least one feature window
        if patient_window_counts is not None:
            valid_pids = {p for p, c in patient_window_counts.items() if c > 0}
            df_eligible = df_patients[df_patients["patient_id"].isin(valid_pids)].copy()
        else:
            df_eligible = df_patients.copy()

        train_pids: list[str] = []
        val_pids: list[str] = []
        test_pids: list[str] = []
        small_strata_reports: list[dict[str, Any]] = []

        # Stratify jointly by source_dataset and diabetes_status
        group_cols = [c for c in ["source_dataset", "diabetes_status"] if c in df_eligible.columns]
        if group_cols:
            grouped = df_eligible.groupby(group_cols)
            for group_key, group in grouped:
                key_desc = str(group_key)
                pids = list(group["patient_id"].astype(str))
                rng.shuffle(pids)
                n = len(pids)

                if n == 0:
                    continue
                elif n == 1:
                    train_pids.extend(pids)
                    small_strata_reports.append(
                        {
                            "stratum": key_desc,
                            "count": 1,
                            "action": "Assigned single patient to train partition; stratum too small for val/test split.",
                        }
                    )
                elif n == 2:
                    train_pids.append(pids[0])
                    val_pids.append(pids[1])
                    small_strata_reports.append(
                        {
                            "stratum": key_desc,
                            "count": 2,
                            "action": "Split across train and val; stratum too small for 3-way partition.",
                        }
                    )
                else:
                    n_val = max(1, int(round(n * val_ratio)))
                    n_test = max(1, int(round(n * test_ratio)))
                    n_train = n - n_val - n_test

                    if n_train < 1:
                        n_train = 1
                        if n_val > 1:
                            n_val -= 1
                        elif n_test > 1:
                            n_test -= 1

                    train_pids.extend(pids[:n_train])
                    val_pids.extend(pids[n_train : n_train + n_val])
                    test_pids.extend(pids[n_train + n_val :])
        else:
            pids = list(df_eligible["patient_id"].astype(str))
            rng.shuffle(pids)
            n = len(pids)
            n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
            n_test = max(1, int(round(n * test_ratio))) if n >= 3 else (1 if n == 2 else 0)
            n_train = n - n_val - n_test
            train_pids = pids[:n_train]
            val_pids = pids[n_train : n_train + n_val]
            test_pids = pids[n_train + n_val :]

        s_train = set(train_pids)
        s_val = set(val_pids)
        s_test = set(test_pids)
        assert s_train.isdisjoint(s_val), "Patient overlap detected between train and val"
        assert s_train.isdisjoint(s_test), "Patient overlap detected between train and test"
        assert s_val.isdisjoint(s_test), "Patient overlap detected between val and test"

        splits = {
            "train": sorted(train_pids),
            "val": sorted(val_pids),
            "test": sorted(test_pids),
        }
        return splits, small_strata_reports

    def split_cross_dataset(
        self,
        df_patients: pd.DataFrame,
        test_source: str,
        patient_window_counts: dict[str, int] | None = None,
        val_ratio: float = 0.20,
    ) -> dict[str, list[str]]:
        rng = np.random.RandomState(self.random_seed)

        if patient_window_counts is not None:
            valid_pids = {p for p, c in patient_window_counts.items() if c > 0}
            df_eligible = df_patients[df_patients["patient_id"].isin(valid_pids)].copy()
        else:
            df_eligible = df_patients.copy()

        test_mask = df_eligible["source_dataset"] == test_source
        test_pids = sorted(df_eligible[test_mask]["patient_id"].astype(str).tolist())

        train_val_pids = list(df_eligible[~test_mask]["patient_id"].astype(str))
        rng.shuffle(train_val_pids)

        n_val = int(round(len(train_val_pids) * val_ratio))
        val_pids = sorted(train_val_pids[:n_val])
        train_pids = sorted(train_val_pids[n_val:])

        return {
            "train": train_pids,
            "val": val_pids,
            "test": test_pids,
        }

    def leave_k_patients_out_cv(
        self,
        df_patients: pd.DataFrame,
        n_splits: int = 5,
        patient_window_counts: dict[str, int] | None = None,
    ) -> Generator[tuple[list[str], list[str]], None, None]:
        rng = np.random.RandomState(self.random_seed)

        if patient_window_counts is not None:
            valid_pids = {p for p, c in patient_window_counts.items() if c > 0}
            df_eligible = df_patients[df_patients["patient_id"].isin(valid_pids)].copy()
        else:
            df_eligible = df_patients.copy()

        pids = list(df_eligible["patient_id"].astype(str))
        rng.shuffle(pids)

        folds = np.array_split(pids, n_splits)
        for fold_idx in range(n_splits):
            val_fold = sorted(folds[fold_idx].tolist())
            train_fold = sorted([p for p in pids if p not in val_fold])
            yield train_fold, val_fold

    def compute_split_breakdown(
        self,
        splits: dict[str, list[str]],
        df_patients: pd.DataFrame,
        patient_window_counts: dict[str, int],
    ) -> dict[str, Any]:
        patient_map = (
            df_patients.set_index("patient_id") if not df_patients.empty else pd.DataFrame()
        )

        breakdown: dict[str, Any] = {}
        for split_name, pids in splits.items():
            split_info: dict[str, Any] = {
                "total_patients": int(len(pids)),
                "total_windows": int(sum(int(patient_window_counts.get(p, 0)) for p in pids)),
                "by_source": {},
                "by_diabetes_status": {},
                "by_source_and_status": {},
            }

            if not patient_map.empty:
                sub = patient_map.loc[patient_map.index.intersection(pids)]
                for src in sub["source_dataset"].unique():
                    src_pids = set(sub[sub["source_dataset"] == src].index)
                    split_info["by_source"][str(src)] = {
                        "patients": int(len(src_pids)),
                        "windows": int(sum(int(patient_window_counts.get(p, 0)) for p in src_pids)),
                    }
                for st in sub["diabetes_status"].unique():
                    st_pids = set(sub[sub["diabetes_status"] == st].index)
                    split_info["by_diabetes_status"][str(st)] = {
                        "patients": int(len(st_pids)),
                        "windows": int(sum(int(patient_window_counts.get(p, 0)) for p in st_pids)),
                    }
                for (src, st), grp in sub.groupby(["source_dataset", "diabetes_status"]):
                    grp_pids = set(grp.index)
                    split_info["by_source_and_status"][f"{src}__{st}"] = {
                        "patients": int(len(grp_pids)),
                        "windows": int(sum(int(patient_window_counts.get(p, 0)) for p in grp_pids)),
                    }

            breakdown[split_name] = split_info

        return breakdown

    def save_split_manifest(
        self,
        splits: dict[str, list[str]],
        df_patients: pd.DataFrame,
        patient_window_counts: dict[str, int],
        output_path: Path,
        small_strata_reports: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        breakdown = self.compute_split_breakdown(splits, df_patients, patient_window_counts)

        manifest = {
            "random_seed": int(self.random_seed),
            "patient_counts": {k: int(len(v)) for k, v in splits.items()},
            "window_counts": {
                k: int(sum(int(patient_window_counts.get(p, 0)) for p in v))
                for k, v in splits.items()
            },
            "split_breakdown": breakdown,
            "small_strata_reports": small_strata_reports or [],
            "splits": splits,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fp:
            json.dump(manifest, fp, indent=2)

        return manifest

    def subset_windowed_dataset(
        self,
        dataset: WindowedDataset,
        pids: list[str],
    ) -> WindowedDataset:
        pid_set = set(pids)
        mask = np.array([p in pid_set for p in dataset.patient_ids])

        if not np.any(mask):
            return WindowedDataset(
                patient_ids=np.array([], dtype=object),
                window_timestamps=np.array([]),
                observed_sequences=np.zeros(
                    (0, dataset.observed_sequences.shape[1], dataset.observed_sequences.shape[2]),
                    dtype=np.float32,
                ),
                observed_feature_names=dataset.observed_feature_names,
                known_future_sequences=np.zeros(
                    (
                        0,
                        dataset.known_future_sequences.shape[1],
                        dataset.known_future_sequences.shape[2],
                    ),
                    dtype=np.float32,
                ),
                known_future_feature_names=dataset.known_future_feature_names,
                static_features=np.zeros((0, dataset.static_features.shape[1]), dtype=np.float32),
                static_feature_names=dataset.static_feature_names,
                tabular_features=np.zeros((0, dataset.tabular_features.shape[1]), dtype=np.float32),
                tabular_feature_names=dataset.tabular_feature_names,
                continuous_targets={
                    h: np.array([], dtype=np.float32) for h in dataset.continuous_targets.keys()
                },
                binary_event_labels={
                    k: np.array([], dtype=np.int64) for k in dataset.binary_event_labels.keys()
                },
            )

        return WindowedDataset(
            patient_ids=dataset.patient_ids[mask],
            window_timestamps=dataset.window_timestamps[mask],
            observed_sequences=dataset.observed_sequences[mask],
            observed_feature_names=dataset.observed_feature_names,
            known_future_sequences=dataset.known_future_sequences[mask],
            known_future_feature_names=dataset.known_future_feature_names,
            static_features=dataset.static_features[mask],
            static_feature_names=dataset.static_feature_names,
            tabular_features=dataset.tabular_features[mask],
            tabular_feature_names=dataset.tabular_feature_names,
            continuous_targets={h: arr[mask] for h, arr in dataset.continuous_targets.items()},
            binary_event_labels={k: arr[mask] for k, arr in dataset.binary_event_labels.items()},
        )
