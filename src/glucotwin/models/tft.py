from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.glucotwin.models.gru import MultimodalDataset, PinballLoss


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = self.fc(x).chunk(2, dim=-1)
        res: torch.Tensor = val * torch.sigmoid(gate)
        return res


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: int = 0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_fc = (
            nn.Linear(context_dim, hidden_dim, bias=False) if context_dim > 0 else None
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.res_fc = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.res_fc(x)
        h = self.fc1(x)
        if self.context_fc is not None and context is not None:
            h = h + self.context_fc(context)
        h = F.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.glu(h)
        res: torch.Tensor = self.layer_norm(residual + h)
        return res


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        num_vars: int,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: int = 0,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.var_grns = nn.ModuleList(
            [
                GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout, context_dim)
                for _ in range(num_vars)
            ]
        )
        self.weight_grn = GatedResidualNetwork(
            num_vars * input_dim, hidden_dim, num_vars, dropout, context_dim
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (B, T, num_vars) or (B, num_vars)
        if x.dim() == 2:
            weights = F.softmax(self.weight_grn(x, context), dim=-1)
            var_outputs = [self.var_grns[i](x[:, i : i + 1], context) for i in range(self.num_vars)]
            stacked = torch.stack(var_outputs, dim=-1)
            selected = torch.sum(stacked * weights.unsqueeze(1), dim=-1)
            return selected, weights
        else:
            B, T, V = x.shape
            flat_x = x.reshape(B * T, V)
            ctx_expanded = (
                context.unsqueeze(1).repeat(1, T, 1).reshape(B * T, -1)
                if context is not None
                else None
            )
            weights = F.softmax(self.weight_grn(flat_x, ctx_expanded), dim=-1).reshape(B, T, V)
            var_outputs = []
            for i in range(self.num_vars):
                var_in = x[:, :, i : i + 1]
                var_out = self.var_grns[i](
                    var_in, context.unsqueeze(1).repeat(1, T, 1) if context is not None else None
                )
                var_outputs.append(var_out)
            stacked = torch.stack(var_outputs, dim=-1)  # (B, T, H, V)
            selected = torch.sum(stacked * weights.unsqueeze(2), dim=-1)  # (B, T, H)
            return selected, weights


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        num_dynamic_vars: int = 11,
        num_future_vars: int = 6,
        num_static_vars: int = 16,
        hidden_dim: int = 48,
        num_heads: int = 4,
        dropout: float = 0.1,
        horizons: list[int] | None = None,
        quantiles: list[float] | None = None,
    ):
        super().__init__()
        self.horizons = horizons if horizons is not None else [30, 60, 120]
        self.quantiles = quantiles if quantiles is not None else [0.10, 0.50, 0.90]
        self.num_quantiles = len(self.quantiles)

        self.static_vsn = VariableSelectionNetwork(num_static_vars, 1, hidden_dim, dropout)
        self.dynamic_vsn = VariableSelectionNetwork(
            num_dynamic_vars, 1, hidden_dim, dropout, context_dim=hidden_dim
        )

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.post_attention_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)

        self.horizon_outputs = nn.ModuleDict(
            {f"h_{h}": nn.Linear(hidden_dim, len(self.quantiles)) for h in self.horizons}
        )
        self.event_heads = nn.ModuleDict(
            {
                "spike_60m": nn.Linear(hidden_dim, 1),
                "hypo_60m": nn.Linear(hidden_dim, 1),
            }
        )

    def forward(
        self,
        x_past: torch.Tensor,
        x_future: torch.Tensor,
        x_static: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        static_embed, static_weights = self.static_vsn(x_static)
        dynamic_embed, dynamic_weights = self.dynamic_vsn(x_past, context=static_embed)

        lstm_out, _ = self.lstm(dynamic_embed)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        enriched = self.post_attention_grn(attn_out + lstm_out)
        last_step = enriched[:, -1, :]

        forecasts: dict[int, torch.Tensor] = {}
        for h in self.horizons:
            forecasts[h] = self.horizon_outputs[f"h_{h}"](last_step)

        events: dict[str, torch.Tensor] = {}
        for ev, head in self.event_heads.items():
            events[ev] = head(last_step).squeeze(-1)

        interpretability = {
            "temporal_attention": attn_weights,
            "static_weights": static_weights,
            "dynamic_weights": dynamic_weights,
        }

        return forecasts, events, interpretability


class TFTTrainer:
    def __init__(
        self,
        model: TemporalFusionTransformer,
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
                q_preds, ev_preds, _ = self.model(past, fut, static)

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
                    q_preds, ev_preds, _ = self.model(past, fut, static)

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
    ) -> tuple[dict[str, Any], np.ndarray | None]:
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
        attention_maps: list[np.ndarray] = []

        with torch.no_grad():
            for past, fut, static, _, _ in loader:
                past = past.to(self.device)
                fut = fut.to(self.device)
                static = static.to(self.device)

                q_preds, ev_preds, interp = self.model(past, fut, static)
                for h in self.model.horizons:
                    q_outputs[h].append(q_preds[h].cpu().numpy())
                for ev in self.model.event_heads:
                    ev_outputs[ev].append(torch.sigmoid(ev_preds[ev]).cpu().numpy())

                if len(attention_maps) < 5:
                    attention_maps.append(interp["temporal_attention"].cpu().numpy())

        quantiles_dict: dict[int, dict[str, np.ndarray]] = {}
        for h in self.model.horizons:
            raw_mat = np.concatenate(q_outputs[h], axis=0)
            q10 = raw_mat[:, 0]
            q50 = raw_mat[:, 1]
            q90 = raw_mat[:, 2]

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

        events_dict["spike_60m"] = events_dict.get("spike_60m", np.zeros(len(past_sequences)))
        events_dict["hypoglycemia_30m"] = events_dict.get("hypo_60m", np.zeros(len(past_sequences)))
        events_dict["hypoglycemia_60m"] = events_dict.get("hypo_60m", np.zeros(len(past_sequences)))
        events_dict["hypoglycemia_120m"] = events_dict.get(
            "hypo_60m", np.zeros(len(past_sequences))
        )
        events_dict["hyperglycemia_60m"] = events_dict.get(
            "spike_60m", np.zeros(len(past_sequences))
        )

        sample_attn = np.concatenate(attention_maps, axis=0) if attention_maps else None
        return {"quantiles": quantiles_dict, "events": events_dict}, sample_attn
