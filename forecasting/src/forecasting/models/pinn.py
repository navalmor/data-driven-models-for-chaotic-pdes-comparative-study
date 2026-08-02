from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class StatePredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        depth: int,
        activation: str,
        residual_prediction: bool,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = []
        in_dim = dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_activation(activation))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)
        self.residual_prediction = residual_prediction

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        delta_or_next = self.net(z)
        if self.residual_prediction:
            return z + delta_or_next
        return delta_or_next
