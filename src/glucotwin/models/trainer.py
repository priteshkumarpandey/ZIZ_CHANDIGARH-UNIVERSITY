from typing import Any

import numpy as np
import optuna
from torch.utils.data import DataLoader

from src.glucotwin.models.baselines import LightGBMForecaster
from src.glucotwin.models.cv import FoldData
from src.glucotwin.models.gru import GRUTrainer, MultimodalDataset, MultimodalGRUForecaster

optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner:
    def __init__(self, fold_data: FoldData, n_trials: int = 15, seed: int = 42):
        self.data = fold_data
        self.n_trials = n_trials
        self.seed = seed

    def tune_lightgbm(self) -> dict[str, Any]:
        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
            num_leaves = trial.suggest_int("num_leaves", 15, 63)
            min_child_samples = trial.suggest_int("min_child_samples", 10, 50)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
            colsample = trial.suggest_float("colsample_bytree", 0.6, 1.0)

            model = LightGBMForecaster(
                horizons=[60],
                quantiles=[0.50],
                n_estimators=80,
                learning_rate=lr,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                subsample=subsample,
                colsample_bytree=colsample,
                random_state=self.seed,
            )
            model.fit(
                self.data.X_train,
                {60: self.data.y_train_cont[60]},
                {},
                self.data.X_val,
                {60: self.data.y_val_cont[60]},
            )
            preds = model.predict(self.data.X_val)["quantiles"][60]["q50"]
            mae = float(np.mean(np.abs(preds - self.data.y_val_cont[60])))
            return mae

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        best_params["n_estimators"] = 150
        return best_params

    def tune_gru(self, device: str = "cuda") -> dict[str, Any]:
        def objective(trial: optuna.Trial) -> float:
            hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 96])
            num_layers = trial.suggest_int("num_layers", 1, 2)
            dropout = trial.suggest_float("dropout", 0.05, 0.25)
            lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)

            model = MultimodalGRUForecaster(
                time_series_dim=self.data.seq_train.shape[-1],
                future_dim=self.data.future_train.shape[-1],
                static_dim=self.data.static_train.shape[-1],
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                horizons=[60],
                quantiles=[0.50],
            )
            trainer = GRUTrainer(model, device=device, lr=lr)

            train_ds = MultimodalDataset(
                self.data.seq_train,
                self.data.future_train,
                self.data.static_train,
                {60: self.data.y_train_cont[60]},
                {},
            )
            val_ds = MultimodalDataset(
                self.data.seq_val,
                self.data.future_val,
                self.data.static_val,
                {60: self.data.y_val_cont[60]},
                {},
            )
            train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

            for _ in range(5):
                trainer.train_epoch(train_loader)

            val_loss = trainer.evaluate(val_loader)
            return val_loss

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params
