"""Frequency + Temporal Dual Attention Model (FINAL CLEAN VERSION)"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn

from attention import AdditiveAttention


# ==========================================
# 🔥 FREQUENCY ATTENTION
# ==========================================
class FrequencyAttention(nn.Module):
    def __init__(self, freq_bins=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(freq_bins, freq_bins // 2),
            nn.ReLU(),
            nn.Linear(freq_bins // 2, freq_bins),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = x.mean(dim=3)       # (B,C,F)
        w = self.fc(w)
        w = w.unsqueeze(-1)
        return x * w


# ==========================================
# 🔥 TEMPORAL SELF-ATTENTION
# ==========================================
class TemporalSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        return torch.matmul(attn, V)


# ==========================================
# CONV STACK
# ==========================================
class ConvStack(nn.Module):
    def __init__(self, in_ch, channels, layers, kernel, dropout):
        super().__init__()

        mods = []
        c = in_ch
        pad = (kernel[0] // 2, kernel[1] // 2)

        for _ in range(layers):
            mods += [
                nn.Conv2d(c, channels, kernel_size=kernel, padding=pad),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            c = channels

        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


# ==========================================
# TEMPORAL MODEL
# ==========================================
class TemporalTCN(nn.Module):
    def __init__(self, in_dim, hidden=128, dilations=(1, 2, 4, 8), dropout=0.3):
        super().__init__()

        layers = []
        c = in_dim

        for d in dilations:
            layers += [
                nn.Conv1d(c, hidden, kernel_size=3, padding=d, dilation=d),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            c = hidden

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.net(x)
        return y.transpose(1, 2)


# ==========================================
# CONFIG
# ==========================================
@dataclass
class ModelConfig:
    dropout: float = 0.5
    hidden: int = 256
    use_attention: bool = True
    use_subbands: bool = True
    use_freq_attention: bool = False
    use_temporal_self_attention: bool = False
    temporal: str = "bilstm"
    remove_b1: bool = False
    remove_b3: bool = False


# ==========================================
# MAIN MODEL
# ==========================================
class FrequencyAwareMultiTaskNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.freq_attn = FrequencyAttention() if cfg.use_freq_attention else None

        self.band_modules = nn.ModuleDict({
            "B1": ConvStack(5, 16, 2, (7, 3), cfg.dropout),
            "B2": ConvStack(5, 24, 3, (5, 3), cfg.dropout),
            "B3": ConvStack(5, 32, 4, (3, 3), cfg.dropout),
            "B4": ConvStack(5, 32, 4, (3, 1), cfg.dropout),
        })

        feat_dim = 16 + 24 + 32 + 32

        if cfg.remove_b1:
            feat_dim -= 16
        if cfg.remove_b3:
            feat_dim -= 32

        # TEMPORAL
        if cfg.temporal == "bilstm":
            self.temporal = nn.LSTM(
                feat_dim,
                cfg.hidden,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            tdim = cfg.hidden * 2
        else:
            self.temporal = TemporalTCN(feat_dim, hidden=cfg.hidden, dropout=cfg.dropout)
            tdim = cfg.hidden

        self.self_attn = TemporalSelfAttention(tdim) if cfg.use_temporal_self_attention else None
        self.attn = AdditiveAttention(tdim) if cfg.use_attention else None

        self.det_head = nn.Linear(tdim, 2)
        self.tex_head = nn.Linear(tdim, 4)

    def _band_forward(self, xb, key):

        if self.freq_attn is not None:
            xb = self.freq_attn(xb)

        z = self.band_modules[key](xb)
        z = z.mean(dim=2)
        return z.transpose(1, 2)

    def forward(self, x):

        band_feats = []

        for i, k in enumerate(["B1", "B2", "B3", "B4"]):

            if (k == "B1" and self.cfg.remove_b1) or (k == "B3" and self.cfg.remove_b3):
                continue

            xb = x[:, i * 5:(i + 1) * 5, :, :]
            band_feats.append(self._band_forward(xb, k))

        min_t = min(b.shape[1] for b in band_feats)
        band_feats = [b[:, :min_t, :] for b in band_feats]

        feats = torch.cat(band_feats, dim=-1)

        if self.cfg.temporal == "bilstm":
            seq, _ = self.temporal(feats)
        else:
            seq = self.temporal(feats)

        if self.self_attn is not None:
            seq = self.self_attn(seq)

        if self.attn is not None:
            ctx, attn = self.attn(seq)
        else:
            ctx = seq.mean(dim=1)
            attn = None

        det_logits = self.det_head(seq)
        tex_logits = self.tex_head(ctx)

        return {
            "det_logits": det_logits,
            "tex_logits": tex_logits,
            "attn": attn,
        }
