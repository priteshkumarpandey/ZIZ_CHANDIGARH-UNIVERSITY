from typing import Any

import numpy as np


def enforce_quantile_monotonicity(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10_arr = np.asarray(q10, dtype=float).copy()
    q50_arr = np.asarray(q50, dtype=float).copy()
    q90_arr = np.asarray(q90, dtype=float).copy()

    # Monotonic sorting across quantiles per sample
    stacked = np.stack([q10_arr, q50_arr, q90_arr], axis=-1)
    stacked_sorted = np.sort(stacked, axis=-1)

    return stacked_sorted[..., 0], stacked_sorted[..., 1], stacked_sorted[..., 2]


class ConformalCalibrator:
    def __init__(self, target_coverage: float = 0.80):
        self.target_coverage = target_coverage
        self.nonconformity_scores: dict[int, np.ndarray] = {}
        self.q_hat: dict[int, float] = {}

    def fit(
        self,
        y_true_dict: dict[int, np.ndarray],
        y_pred_lower_dict: dict[int, np.ndarray],
        y_pred_upper_dict: dict[int, np.ndarray],
    ) -> None:
        for horizon in y_true_dict.keys():
            y_t = np.asarray(y_true_dict[horizon], dtype=float)
            lower = np.asarray(y_pred_lower_dict[horizon], dtype=float)
            upper = np.asarray(y_pred_upper_dict[horizon], dtype=float)

            mask = ~np.isnan(y_t) & ~np.isnan(lower) & ~np.isnan(upper)
            y_t_val = y_t[mask]
            lower_val = lower[mask]
            upper_val = upper[mask]

            if len(y_t_val) == 0:
                self.q_hat[horizon] = 0.0
                continue

            scores = np.maximum(lower_val - y_t_val, y_t_val - upper_val)
            self.nonconformity_scores[horizon] = scores

            n = len(scores)
            alpha = 1.0 - self.target_coverage
            # Finite sample conformal correction factor
            quantile_level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
            self.q_hat[horizon] = float(np.quantile(scores, quantile_level))

    def calibrate(
        self,
        y_true_dict: dict[int, np.ndarray],
        y_pred_lower_dict: dict[int, np.ndarray],
        y_pred_upper_dict: dict[int, np.ndarray],
    ) -> None:
        self.fit(y_true_dict, y_pred_lower_dict, y_pred_upper_dict)

    def predict_calibrated_intervals(
        self,
        horizon: int,
        raw_lower: np.ndarray,
        raw_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        q_val = self.q_hat.get(horizon, 0.0)
        lower_arr = np.asarray(raw_lower, dtype=float)
        upper_arr = np.asarray(raw_upper, dtype=float)

        calibrated_lower = np.maximum(30.0, lower_arr - q_val)
        calibrated_upper = np.minimum(500.0, upper_arr + q_val)
        return calibrated_lower, calibrated_upper

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_coverage": float(self.target_coverage),
            "q_hat": {str(k): float(v) for k, v in self.q_hat.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConformalCalibrator":
        obj = cls(target_coverage=float(data.get("target_coverage", 0.80)))
        obj.q_hat = {int(k): float(v) for k, v in data.get("q_hat", {}).items()}
        return obj
