from typing import Any

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.original_linear = original_linear
        # Freeze original linear weights
        for param in self.original_linear.parameters():
            param.requires_grad = False

        self.rank = rank
        self.scaling = alpha / rank
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig: torch.Tensor = self.original_linear(x)
        lora: torch.Tensor = (x @ self.lora_A @ self.lora_B) * self.scaling
        res: torch.Tensor = orig + lora
        return res

    def get_lora_weights(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "scaling": self.scaling,
            "lora_A": self.lora_A.detach().cpu().tolist(),
            "lora_B": self.lora_B.detach().cpu().tolist(),
        }

    def load_lora_weights(self, weights: dict[str, Any]) -> None:
        self.lora_A.data.copy_(torch.tensor(weights["lora_A"], dtype=torch.float32))
        self.lora_B.data.copy_(torch.tensor(weights["lora_B"], dtype=torch.float32))


def attach_lora_adapters(
    model: nn.Module, rank: int = 4, alpha: float = 8.0
) -> dict[str, LoRALinear]:
    adapters = {}
    horizon_heads = getattr(model, "horizon_quantile_heads", None)
    if isinstance(horizon_heads, (nn.ModuleDict, dict)):
        for h in [30, 60, 120]:
            head_name = f"h_{h}"
            if head_name in horizon_heads:
                seq_block = horizon_heads[head_name]
                # Replace the first linear layer in the head (Linear(fused_dim, 64)) with LoRALinear
                if isinstance(seq_block, nn.Sequential) and len(seq_block) > 0:
                    first_layer = seq_block[0]
                    if isinstance(first_layer, nn.Linear) and not isinstance(
                        first_layer, LoRALinear
                    ):
                        adapted = LoRALinear(first_layer, rank=rank, alpha=alpha)
                        seq_block[0] = adapted
                        adapters[head_name] = adapted
    return adapters
