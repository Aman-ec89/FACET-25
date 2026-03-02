"""Additive attention layer."""
from __future__ import annotations

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        e = torch.tanh(self.proj(x))
        a = torch.softmax(self.score(e), dim=1)  # (B, T, 1)
        context = torch.sum(a * x, dim=1)
        return context, a.squeeze(-1)
