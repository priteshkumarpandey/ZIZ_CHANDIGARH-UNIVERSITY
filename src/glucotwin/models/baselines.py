from typing import Any

import lightgbm as lgb
import numpy as np
import xgboost as xgb


class PersistenceBaseline:
    def __init__(self, horizons: list[int] | None = None):
        self.horizons = horizons if horizons is not None else [30, 60, 120]

    def predict(
        self,
        current_cgm: np.ndarray,
    ) -> dict[str, Any]:
        cgm = np.asarray(current_cgm, dtype=float)
        # Baseline persistence assumes constant trajectory equal to current CGM reading
        quantiles_dict: dict[int, dict[str, np.ndarray]] = {}
        events_dict: dict[str, np.ndarray] = {}

        for h in self.horizons:
            # Empirical spread grows with horizon uncertainty
            std_spread = 8.0 * np.sqrt(h / 30.0)
            q10 = np.clip(cgm - 1.28 * std_spread, 30.0, 500.0)
            q50 = np.clip(cgm, 30.0, 500.0)
            q90 = np.clip(cgm + 1.28 * std_spread, 30.0, 500.0)
            quantiles_dict[h] = {"q10": q10, "q50": q50, "q90": q90}

            # Event probability heuristics for persistence
            events_dict[f"spike_{h}m"] = (cgm >= 180.0).astype(float)
            events_dict[f"hypoglycemia_{h}m"] = (cgm <= 70.0).astype(float)
            events_dict[f"hyperglycemia_{h}m"] = (cgm >= 180.0).astype(float)

        events_dict["spike_60m"] = events_dict.get("spike_60m", (cgm >= 180.0).astype(float))
        return {"quantiles": quantiles_dict, "events": events_dict}


class LinearTrendBaseline:
    def __init__(self, horizons: list[int] | None = None):
        self.horizons = horizons if horizons is not None else [30, 60, 120]

    def predict(
        self,
        cgm_history_30m: np.ndarray,
    ) -> dict[str, Any]:
        # Extrapolate linear slope from last 30 minutes (6 observations at 5-minute intervals)
        cgm_seq = np.asarray(cgm_history_30m, dtype=float)
        N, T = cgm_seq.shape
        t_steps = np.arange(T, dtype=float)
        t_mean = np.mean(t_steps)
        t_var = np.sum((t_steps - t_mean) ** 2)

        quantiles_dict: dict[int, dict[str, np.ndarray]] = {}
        events_dict: dict[str, np.ndarray] = {}

        # Vectorized slope calculation
        y_mean = np.mean(cgm_seq, axis=1, keepdims=True)
        cov = np.sum((cgm_seq - y_mean) * (t_steps - t_mean), axis=1)
        slopes = cov / (t_var + 1e-8)
        current_cgm = cgm_seq[:, -1]

        for h in self.horizons:
            forward_steps = h / 5.0
            pred_point = np.clip(current_cgm + slopes * forward_steps, 30.0, 500.0)

            std_spread = 10.0 * np.sqrt(h / 30.0)
            q10 = np.clip(pred_point - 1.28 * std_spread, 30.0, 500.0)
            q50 = pred_point
            q90 = np.clip(pred_point + 1.28 * std_spread, 30.0, 500.0)
            quantiles_dict[h] = {"q10": q10, "q50": q50, "q90": q90}

            events_dict[f"spike_{h}m"] = (pred_point >= 180.0).astype(float)
            events_dict[f"hypoglycemia_{h}m"] = (pred_point <= 70.0).astype(float)
            events_dict[f"hyperglycemia_{h}m"] = (pred_point >= 180.0).astype(float)

        events_dict["spike_60m"] = events_dict.get(
            "spike_60m", (current_cgm >= 180.0).astype(float)
        )
        return {"quantiles": quantiles_dict, "events": events_dict}


class LightGBMForecaster:
    def __init__(
        self,
        horizons: list[int] | None = None,
        quantiles: list[float] | None = None,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.horizons = horizons if horizons is not None else [30, 60, 120]
        self.quantiles = quantiles if quantiles is not None else [0.10, 0.50, 0.90]
        self.params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": -1,
        }
        self.quantile_models: dict[str, lgb.LGBMRegressor] = {}
        self.event_models: dict[str, lgb.LGBMClassifier] = {}
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train_cont: dict[int, np.ndarray],
        events_train: dict[str, np.ndarray],
        X_val: np.ndarray | None = None,
        y_val_cont: dict[int, np.ndarray] | None = None,
        events_val: dict[str, np.ndarray] | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        if lgb is None:
            raise ImportError("lightgbm is not installed")

        self.feature_names = feature_names or [f"f_{i}" for i in range(X_train.shape[1])]

        for h in self.horizons:
            y_tr = y_train_cont[h]
            y_v = y_val_cont[h] if y_val_cont is not None else None

            for q in self.quantiles:
                key = f"h{h}_q{int(q * 100)}"
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    **self.params,
                )
                if X_val is not None and y_v is not None:
                    model.fit(
                        X_train,
                        y_tr,
                        eval_set=[(X_val, y_v)],
                        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
                    )
                else:
                    model.fit(X_train, y_tr)
                self.quantile_models[key] = model

        # Fit binary classifiers for key adverse glycemic events
        target_events = [
            "spike_60m",
            "hypoglycemia_30m",
            "hypoglycemia_60m",
            "hypoglycemia_120m",
            "hyperglycemia_60m",
        ]
        for ev in target_events:
            if ev in events_train:
                y_ev_tr = events_train[ev]
                y_ev_val = events_val[ev] if events_val is not None and ev in events_val else None
                clf = lgb.LGBMClassifier(
                    objective="binary",
                    is_unbalance=True,
                    **self.params,
                )
                if X_val is not None and y_ev_val is not None:
                    clf.fit(
                        X_train,
                        y_ev_tr,
                        eval_set=[(X_val, y_ev_val)],
                        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
                    )
                else:
                    clf.fit(X_train, y_ev_tr)
                self.event_models[ev] = clf

    def predict(self, X: np.ndarray) -> dict[str, Any]:
        quantiles_dict: dict[int, dict[str, np.ndarray]] = {}
        for h in self.horizons:
            h_dict: dict[str, np.ndarray] = {}
            for q in self.quantiles:
                key = f"h{h}_q{int(q * 100)}"
                if key in self.quantile_models:
                    h_dict[f"q{int(q * 100)}"] = np.asarray(self.quantile_models[key].predict(X))

            if "q10" in h_dict and "q50" in h_dict and "q90" in h_dict:
                q10 = h_dict["q10"]
                q50 = h_dict["q50"]
                q90 = h_dict["q90"]
                q10_fixed = np.minimum(q10, q50)
                q90_fixed = np.maximum(q90, q50)
                h_dict["q10"] = np.clip(q10_fixed, 30.0, 500.0)
                h_dict["q50"] = np.clip(q50, 30.0, 500.0)
                h_dict["q90"] = np.clip(q90_fixed, 30.0, 500.0)
            elif "q50" in h_dict:
                h_dict["q50"] = np.clip(h_dict["q50"], 30.0, 500.0)

            quantiles_dict[h] = h_dict

        events_dict: dict[str, np.ndarray] = {}
        for ev, clf in self.event_models.items():
            prob = np.asarray(clf.predict_proba(X))[:, 1]
            events_dict[ev] = prob

        return {"quantiles": quantiles_dict, "events": events_dict}


class XGBoostForecaster:
    def __init__(
        self,
        horizons: list[int] | None = None,
        quantiles: list[float] | None = None,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.horizons = horizons if horizons is not None else [30, 60, 120]
        self.quantiles = quantiles if quantiles is not None else [0.10, 0.50, 0.90]
        self.params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": -1,
        }
        self.quantile_models: dict[str, xgb.XGBRegressor] = {}
        self.event_models: dict[str, xgb.XGBClassifier] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train_cont: dict[int, np.ndarray],
        events_train: dict[str, np.ndarray],
        X_val: np.ndarray | None = None,
        y_val_cont: dict[int, np.ndarray] | None = None,
        events_val: dict[str, np.ndarray] | None = None,
    ) -> None:
        if xgb is None:
            raise ImportError("xgboost is not installed")

        for h in self.horizons:
            y_tr = y_train_cont[h]
            y_v = y_val_cont[h] if y_val_cont is not None else None

            for q in self.quantiles:
                key = f"h{h}_q{int(q * 100)}"
                model = xgb.XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=q,
                    early_stopping_rounds=15 if X_val is not None else None,
                    **self.params,
                )
                if X_val is not None and y_v is not None:
                    model.fit(X_train, y_tr, eval_set=[(X_val, y_v)], verbose=False)
                else:
                    model.fit(X_train, y_tr, verbose=False)
                self.quantile_models[key] = model

        target_events = [
            "spike_60m",
            "hypoglycemia_30m",
            "hypoglycemia_60m",
            "hypoglycemia_120m",
            "hyperglycemia_60m",
        ]
        for ev in target_events:
            if ev in events_train:
                y_ev_tr = events_train[ev]
                y_ev_val = events_val[ev] if events_val is not None and ev in events_val else None
                clf = xgb.XGBClassifier(
                    objective="binary:logistic",
                    scale_pos_weight=3.0,
                    early_stopping_rounds=15 if X_val is not None else None,
                    **self.params,
                )
                if X_val is not None and y_ev_val is not None:
                    clf.fit(X_train, y_ev_tr, eval_set=[(X_val, y_ev_val)], verbose=False)
                else:
                    clf.fit(X_train, y_ev_tr, verbose=False)
                self.event_models[ev] = clf

    def predict(self, X: np.ndarray) -> dict[str, Any]:
        quantiles_dict: dict[int, dict[str, np.ndarray]] = {}
        for h in self.horizons:
            h_dict: dict[str, np.ndarray] = {}
            for q in self.quantiles:
                key = f"h{h}_q{int(q * 100)}"
                if key in self.quantile_models:
                    h_dict[f"q{int(q * 100)}"] = np.asarray(self.quantile_models[key].predict(X))

            if "q10" in h_dict and "q50" in h_dict and "q90" in h_dict:
                q10 = h_dict["q10"]
                q50 = h_dict["q50"]
                q90 = h_dict["q90"]
                q10_fixed = np.minimum(q10, q50)
                q90_fixed = np.maximum(q90, q50)
                h_dict["q10"] = np.clip(q10_fixed, 30.0, 500.0)
                h_dict["q50"] = np.clip(q50, 30.0, 500.0)
                h_dict["q90"] = np.clip(q90_fixed, 30.0, 500.0)
            elif "q50" in h_dict:
                h_dict["q50"] = np.clip(h_dict["q50"], 30.0, 500.0)

            quantiles_dict[h] = h_dict

        events_dict: dict[str, np.ndarray] = {}
        for ev, clf in self.event_models.items():
            prob = np.asarray(clf.predict_proba(X))[:, 1]
            events_dict[ev] = prob

        return {"quantiles": quantiles_dict, "events": events_dict}
