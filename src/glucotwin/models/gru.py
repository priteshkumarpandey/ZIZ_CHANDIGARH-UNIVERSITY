from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class MultimodalDataset(Dataset):
    def __init__(
        self,
        past_sequences: np.ndarray,
        future_sequences: np.ndarray,
        static_features: np.ndarray,
        targets_dict: dict[int, np.ndarray],
        events_dict: dict[str, np.ndarray],
    ):
        self.past = torch.tensor(past_sequences, dtype=torch.float32)
        self.future = torch.tensor(future_sequences, dtype=torch.float32)
        self.static = torch.tensor(static_features, dtype=torch.float32)
        self.targets = {h: torch.tensor(v, dtype=torch.float32) for h, v in targets_dict.items()}
        self.events = {k: torch.tensor(v, dtype=torch.float32) for k, v in events_dict.items()}

    def __len__(self) -> int:
        return len(self.past)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict[int, torch.Tensor], dict[str, torch.Tensor]
    ]:
        tgt = {h: self.targets[h][idx] for h in self.targets}
        ev = {k: self.events[k][idx] for k in self.events}
        return self.past[idx], self.future[idx], self.static[idx], tgt, ev


class PinballLoss(nn.Module):
    def __init__(self, quantiles: list[float] | None = None):
        super().__init__()
        self.quantiles = quantiles if quantiles is not None else [0.10, 0.50, 0.90]

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds shape: (batch_size, num_quantiles), targets shape: (batch_size,)
        targets_exp = targets.unsqueeze(-1)
        errors = targets_exp - preds
        losses = []
        for i, q in enumerate(self.quantiles):
            err = errors[:, i]
            loss = torch.max((q - 1) * err, q * err)
            losses.append(loss.mean())
        return torch.stack(losses).mean()


class MultimodalGRUForecaster(nn.Module):
    def __init__(
        self,
        time_series_dim: int = 11,
        future_dim: int = 6,
        static_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizons: list[int] | None = None,
        quantiles: list[float] | None = None,
    ):
        super().__init__()
        self.horizons = horizons if horizons is not None else [30, 60, 120]
        self.quantiles = quantiles if quantiles is not None else [0.10, 0.50, 0.90]
        self.num_quantiles = len(self.quantiles)

        # Dynamic lookback encoder
        self.gru_past = nn.GRU(
            input_size=time_series_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Known future covariate encoder (meals, diurnal harmonics)
        self.gru_future = nn.GRU(
            input_size=future_dim,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )

        # Static EHR covariate encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
        )

        fused_dim = hidden_dim + 32 + 32
        self.horizon_quantile_heads = nn.ModuleDict(
            {
                f"h_{h}": nn.Sequential(
                    nn.Linear(fused_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, self.num_quantiles),
                )
                for h in self.horizons
            }
        )

        # Multi-task adverse event heads
        self.event_heads = nn.ModuleDict(
            {
                "spike_60m": nn.Sequential(nn.Linear(fused_dim, 32), nn.ReLU(), nn.Linear(32, 1)),
                "hypo_30m": nn.Sequential(nn.Linear(fused_dim, 32), nn.ReLU(), nn.Linear(32, 1)),
                "hypo_60m": nn.Sequential(nn.Linear(fused_dim, 32), nn.ReLU(), nn.Linear(32, 1)),
                "hypo_120m": nn.Sequential(nn.Linear(fused_dim, 32), nn.ReLU(), nn.Linear(32, 1)),
                "hyper_60m": nn.Sequential(nn.Linear(fused_dim, 32), nn.ReLU(), nn.Linear(32, 1)),
            }
        )

    def forward(
        self,
        x_past: torch.Tensor,
        x_future: torch.Tensor,
        x_static: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[str, torch.Tensor]]:
        out_past, _ = self.gru_past(x_past)
        past_rep = out_past[:, -1, :]

        out_fut, _ = self.gru_future(x_future)
        fut_rep = out_fut[:, -1, :]

        static_rep = self.static_encoder(x_static)

        fused = torch.cat([past_rep, fut_rep, static_rep], dim=-1)

        quantiles: dict[int, torch.Tensor] = {}
        for h in self.horizons:
            raw_q = self.horizon_quantile_heads[f"h_{h}"](fused)
            quantiles[h] = raw_q

        events: dict[str, torch.Tensor] = {}
        for ev, head in self.event_heads.items():
            events[ev] = head(fused).squeeze(-1)

        return quantiles, events


class GRUTrainer:
    def __init__(
        self,
        model: MultimodalGRUForecaster,
        device: str | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.pinball = PinballLoss(model.quantiles)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(self.device))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for past, fut, static, targets, events in dataloader:
            past = past.to(self.device)
            fut = fut.to(self.device)
            static = static.to(self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast(
                device_type="cuda" if "cuda" in self.device else "cpu",
                enabled="cuda" in self.device,
            ):
                q_preds, ev_preds = self.model(past, fut, static)

                loss: torch.Tensor = torch.tensor(0.0, device=self.device)
                for h, tgt in targets.items():
                    loss = loss + self.pinball(q_preds[h], tgt.to(self.device))

                # Event losses
                if "spike_60m" in events:
                    loss += 0.5 * self.bce(
                        ev_preds["spike_60m"], events["spike_60m"].to(self.device)
                    )
                if "hypoglycemia_60m" in events:
                    loss += 0.5 * self.bce(
                        ev_preds["hypo_60m"], events["hypoglycemia_60m"].to(self.device)
                    )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(1, n_batches)

    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for past, fut, static, targets, events in dataloader:
                past = past.to(self.device)
                fut = fut.to(self.device)
                static = static.to(self.device)

                with torch.amp.autocast(
                    device_type="cuda" if "cuda" in self.device else "cpu",
                    enabled="cuda" in self.device,
                ):
                    q_preds, ev_preds = self.model(past, fut, static)

                    loss: torch.Tensor = torch.tensor(0.0, device=self.device)
                    for h, tgt in targets.items():
                        loss = loss + self.pinball(q_preds[h], tgt.to(self.device))

                    if "spike_60m" in events:
                        loss += 0.5 * self.bce(
                            ev_preds["spike_60m"], events["spike_60m"].to(self.device)
                        )
                    if "hypoglycemia_60m" in events:
                        loss += 0.5 * self.bce(
                            ev_preds["hypo_60m"], events["hypoglycemia_60m"].to(self.device)
                        )

                total_loss += float(loss.item())
                n_batches += 1

        return total_loss / max(1, n_batches)

    def predict(
        self,
        past_sequences: np.ndarray,
        future_sequences: np.ndarray,
        static_features: np.ndarray,
        batch_size: int = 512,
    ) -> dict[str, Any]:
        self.model.eval()
        dataset = MultimodalDataset(
            past_sequences=past_sequences,
            future_sequences=future_sequences,
            static_features=static_features,
            targets_dict={
                30: np.zeros(len(past_sequences)),
                60: np.zeros(len(past_sequences)),
                120: np.zeros(len(past_sequences)),
            },
            events_dict={},
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        q_outputs: dict[int, list[np.ndarray]] = {h: [] for h in self.model.horizons}
        ev_outputs: dict[str, list[np.ndarray]] = {ev: [] for ev in self.model.event_heads}

        with torch.no_grad():
            for past, fut, static, _, _ in loader:
                past = past.to(self.device)
                fut = fut.to(self.device)
                static = static.to(self.device)

                q_preds, ev_preds = self.model(past, fut, static)
                for h in self.model.horizons:
                    q_outputs[h].append(q_preds[h].cpu().numpy())
                for ev in self.model.event_heads:
                    ev_outputs[ev].append(torch.sigmoid(ev_preds[ev]).cpu().numpy())

        quantiles_dict: dict[int, dict[str, np.ndarray]] = {}
        for h in self.model.horizons:
            raw_mat = np.concatenate(q_outputs[h], axis=0)
            q10 = raw_mat[:, 0]
            q50 = raw_mat[:, 1]
            q90 = raw_mat[:, 2]

            # Enforce monotonic quantile bounds
            q10_fixed = np.minimum(q10, q50)
            q90_fixed = np.maximum(q90, q50)

            quantiles_dict[h] = {
                "q10": np.clip(q10_fixed, 30.0, 500.0),
                "q50": np.clip(q50, 30.0, 500.0),
                "q90": np.clip(q90_fixed, 30.0, 500.0),
            }

        events_dict: dict[str, np.ndarray] = {}
        for ev in self.model.event_heads:
            events_dict[ev] = np.concatenate(ev_outputs[ev], axis=0)

        # Standardize event keys for downstream evaluator
        events_dict["spike_60m"] = events_dict.get("spike_60m", np.zeros(len(past_sequences)))
        events_dict["hypoglycemia_30m"] = events_dict.get("hypo_30m", np.zeros(len(past_sequences)))
        events_dict["hypoglycemia_60m"] = events_dict.get("hypo_60m", np.zeros(len(past_sequences)))
        events_dict["hypoglycemia_120m"] = events_dict.get(
            "hypo_120m", np.zeros(len(past_sequences))
        )
        events_dict["hyperglycemia_60m"] = events_dict.get(
            "hyper_60m", np.zeros(len(past_sequences))
        )

        return {"quantiles": quantiles_dict, "events": events_dict}
