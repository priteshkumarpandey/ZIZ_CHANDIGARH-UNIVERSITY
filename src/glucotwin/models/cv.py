import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.glucotwin.utils.io import as_pandas_df, read_parquet


@dataclass
class FoldSplit:
    fold_idx: int
    train_patients: list[str]
    val_patients: list[str]
    test_patients: list[str]
    train_windows: int
    val_windows: int
    test_windows: int
    status_counts: dict[str, dict[str, int]]


@dataclass
class FoldData:
    fold_idx: int
    train_patient_ids: np.ndarray
    val_patient_ids: np.ndarray
    test_patient_ids: np.ndarray
    # Tabular features (X)
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    # Sequences (72 steps past, 24 steps future, 16 static)
    seq_train: np.ndarray
    seq_val: np.ndarray
    seq_test: np.ndarray
    future_train: np.ndarray
    future_val: np.ndarray
    future_test: np.ndarray
    static_train: np.ndarray
    static_val: np.ndarray
    static_test: np.ndarray
    # Targets per horizon (30, 60, 120)
    y_train_cont: dict[int, np.ndarray]
    y_val_cont: dict[int, np.ndarray]
    y_test_cont: dict[int, np.ndarray]
    # Binary events (spike_60m, hypo_30m, hypo_60m, hypo_120m, hyper_30m, hyper_60m, hyper_120m)
    events_train: dict[str, np.ndarray]
    events_val: dict[str, np.ndarray]
    events_test: dict[str, np.ndarray]
    # Raw unnormalized CGM for persistence and linear trend baselines
    raw_cgm_curr_val: np.ndarray
    raw_cgm_curr_test: np.ndarray
    raw_cgm_hist_val: np.ndarray
    raw_cgm_hist_test: np.ndarray
    # Scalers
    tabular_scaler: StandardScaler
    static_scaler: StandardScaler
    seq_mean: np.ndarray
    seq_std: np.ndarray


class CrossValidationManager:
    def __init__(
        self,
        processed_dir: Path = Path("data/processed"),
        n_splits: int = 5,
        random_seed: int = 42,
    ):
        self.processed_dir = Path(processed_dir)
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.folds: list[FoldSplit] = []
        self._load_and_prepare_cohort()

    def _load_and_prepare_cohort(self) -> None:
        pat_file = self.processed_dir / "patients_processed.parquet"
        npz_file = self.processed_dir / "features" / "windowed_features.npz"

        if not pat_file.exists() or not npz_file.exists():
            return

        df_pat = as_pandas_df(read_parquet(pat_file))
        cg_pat = df_pat[df_pat["source_dataset"] == "cgmacros"].copy()
        cg_pat.sort_values("patient_id", inplace=True)

        patient_ids = cg_pat["patient_id"].values
        statuses = cg_pat["diabetes_status"].values

        npz_data = np.load(npz_file, allow_pickle=True)
        window_pids = npz_data["patient_ids"]

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)

        self.folds = []
        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(patient_ids, statuses)):
            test_pids = patient_ids[test_idx].tolist()
            tv_pids = patient_ids[train_val_idx]
            tv_statuses = statuses[train_val_idx]

            # Internal validation split (stratified hold-out for early stopping and calibration)
            skf_val = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_seed + fold_idx + 1
            )
            for tr_idx, val_idx in skf_val.split(tv_pids, tv_statuses):
                val_pids = tv_pids[val_idx].tolist()
                train_pids = tv_pids[tr_idx].tolist()
                break

            # Verify patient-level independence
            assert len(set(train_pids) & set(val_pids)) == 0
            assert len(set(train_pids) & set(test_pids)) == 0
            assert len(set(val_pids) & set(test_pids)) == 0

            tr_mask = np.isin(window_pids, train_pids)
            val_mask = np.isin(window_pids, val_pids)
            te_mask = np.isin(window_pids, test_pids)

            def get_counts(pids_list: list[str]) -> dict[str, int]:
                sub = cg_pat[cg_pat["patient_id"].isin(pids_list)]
                return {str(k): int(v) for k, v in sub["diabetes_status"].value_counts().items()}

            status_counts = {
                "train": get_counts(train_pids),
                "val": get_counts(val_pids),
                "test": get_counts(test_pids),
            }

            fold_split = FoldSplit(
                fold_idx=fold_idx,
                train_patients=train_pids,
                val_patients=val_pids,
                test_patients=test_pids,
                train_windows=int(np.sum(tr_mask)),
                val_windows=int(np.sum(val_mask)),
                test_windows=int(np.sum(te_mask)),
                status_counts=status_counts,
            )
            self.folds.append(fold_split)

    def save_manifest(self, output_path: Path | None = None) -> None:
        target = output_path or (self.processed_dir / "cv_manifest.json")
        manifest_data = {
            "n_splits": self.n_splits,
            "random_seed": self.random_seed,
            "total_participants": 45,
            "folds": [
                {
                    "fold_idx": f.fold_idx,
                    "train_patients": f.train_patients,
                    "val_patients": f.val_patients,
                    "test_patients": f.test_patients,
                    "train_windows": f.train_windows,
                    "val_windows": f.val_windows,
                    "test_windows": f.test_windows,
                    "status_counts": f.status_counts,
                }
                for f in self.folds
            ],
        }
        with open(target, "w") as fp:
            json.dump(manifest_data, fp, indent=2)

    def get_fold_data(
        self,
        fold_idx: int,
        normalize: bool = True,
    ) -> FoldData:
        npz_file = self.processed_dir / "features" / "windowed_features.npz"
        data = np.load(npz_file, allow_pickle=True)

        window_pids = data["patient_ids"]
        fold = self.folds[fold_idx]

        train_mask = np.isin(window_pids, fold.train_patients)
        val_mask = np.isin(window_pids, fold.val_patients)
        test_mask = np.isin(window_pids, fold.test_patients)

        X_train = data["tabular_features"][train_mask].copy()
        X_val = data["tabular_features"][val_mask].copy()
        X_test = data["tabular_features"][test_mask].copy()

        static_train = data["static_features"][train_mask].copy()
        static_val = data["static_features"][val_mask].copy()
        static_test = data["static_features"][test_mask].copy()

        seq_train = data["observed_sequences"][train_mask].copy()
        seq_val = data["observed_sequences"][val_mask].copy()
        seq_test = data["observed_sequences"][test_mask].copy()

        fut_train = data["known_future_sequences"][train_mask].copy()
        fut_val = data["known_future_sequences"][val_mask].copy()
        fut_test = data["known_future_sequences"][test_mask].copy()

        # Extract raw unnormalized CGM values (mg/dL) before z-score scaling
        raw_cgm_curr_val = seq_val[:, -1, 0].copy()
        raw_cgm_curr_test = seq_test[:, -1, 0].copy()
        raw_cgm_hist_val = seq_val[:, -6:, 0].copy()
        raw_cgm_hist_test = seq_test[:, -6:, 0].copy()

        tab_scaler = StandardScaler()
        stat_scaler = StandardScaler()
        seq_mean = np.zeros(seq_train.shape[-1], dtype=np.float32)
        seq_std = np.ones(seq_train.shape[-1], dtype=np.float32)

        if normalize:
            # Scalers fit on training patients only to avoid lookahead or distribution leakage
            X_train = tab_scaler.fit_transform(X_train)
            X_val = tab_scaler.transform(X_val)
            X_test = tab_scaler.transform(X_test)

            static_train = stat_scaler.fit_transform(static_train)
            static_val = stat_scaler.transform(static_val)
            static_test = stat_scaler.transform(static_test)

            # Sequence normalization on 3D arrays
            N, T, F = seq_train.shape
            seq_train_flat = seq_train.reshape(-1, F)
            seq_mean = np.nanmean(seq_train_flat, axis=0)
            seq_std = np.nanstd(seq_train_flat, axis=0)
            seq_std = np.where(seq_std < 1e-6, 1.0, seq_std)

            seq_train = (seq_train - seq_mean) / seq_std
            seq_val = (seq_val - seq_mean) / seq_std
            seq_test = (seq_test - seq_mean) / seq_std

        y_train_cont = {
            30: data["target_30m"][train_mask],
            60: data["target_60m"][train_mask],
            120: data["target_120m"][train_mask],
        }
        y_val_cont = {
            30: data["target_30m"][val_mask],
            60: data["target_60m"][val_mask],
            120: data["target_120m"][val_mask],
        }
        y_test_cont = {
            30: data["target_30m"][test_mask],
            60: data["target_60m"][test_mask],
            120: data["target_120m"][test_mask],
        }

        event_keys = [
            "spike_60m",
            "hyperglycemia_30m",
            "severe_hyperglycemia_30m",
            "hypoglycemia_30m",
            "hyperglycemia_60m",
            "severe_hyperglycemia_60m",
            "hypoglycemia_60m",
            "hyperglycemia_120m",
            "severe_hyperglycemia_120m",
            "hypoglycemia_120m",
        ]
        events_train = {
            k: data[f"event_{k}"][train_mask] for k in event_keys if f"event_{k}" in data
        }
        events_val = {k: data[f"event_{k}"][val_mask] for k in event_keys if f"event_{k}" in data}
        events_test = {k: data[f"event_{k}"][test_mask] for k in event_keys if f"event_{k}" in data}

        return FoldData(
            fold_idx=fold_idx,
            train_patient_ids=window_pids[train_mask],
            val_patient_ids=window_pids[val_mask],
            test_patient_ids=window_pids[test_mask],
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            seq_train=seq_train,
            seq_val=seq_val,
            seq_test=seq_test,
            future_train=fut_train,
            future_val=fut_val,
            future_test=fut_test,
            static_train=static_train,
            static_val=static_val,
            static_test=static_test,
            y_train_cont=y_train_cont,
            y_val_cont=y_val_cont,
            y_test_cont=y_test_cont,
            events_train=events_train,
            events_val=events_val,
            events_test=events_test,
            raw_cgm_curr_val=raw_cgm_curr_val,
            raw_cgm_curr_test=raw_cgm_curr_test,
            raw_cgm_hist_val=raw_cgm_hist_val,
            raw_cgm_hist_test=raw_cgm_hist_test,
            tabular_scaler=tab_scaler,
            static_scaler=stat_scaler,
            seq_mean=seq_mean,
            seq_std=seq_std,
        )
