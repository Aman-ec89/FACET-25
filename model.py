"""Frequency-aware multi-task chewing classification models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn

from attention import AdditiveAttention


class ConvStack(nn.Module):
    def __init__(self, in_ch: int, channels: int, layers: int, kernel: tuple[int, int], dropout: float):
        super().__init__()
        mods: List[nn.Module] = []
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


class TemporalTCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dilations=(1, 2, 4, 8), dropout: float = 0.3):
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
        # x (B,T,D)
        x = x.transpose(1, 2)
        y = self.net(x)
        return y.transpose(1, 2)


@dataclass
class ModelConfig:
    dropout: float = 0.3
    hidden: int = 128
    use_attention: bool = True
    use_subbands: bool = True
    temporal: str = "bilstm"  # or tcn
    remove_b1: bool = False
    remove_b3: bool = False


class FrequencyAwareMultiTaskNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.band_modules = nn.ModuleDict(
            {
                "B1": ConvStack(1, 16, 2, (7, 3), cfg.dropout),
                "B2": ConvStack(1, 24, 3, (5, 3), cfg.dropout),
                "B3": ConvStack(1, 32, 4, (3, 3), cfg.dropout),
                "B4": ConvStack(1, 32, 4, (3, 1), cfg.dropout),
            }
        )
        feat_dim = 16 + 24 + 32 + 32
        if cfg.remove_b1:
            feat_dim -= 16
        if cfg.remove_b3:
            feat_dim -= 32

        if cfg.temporal == "bilstm":
            self.temporal = nn.LSTM(feat_dim, cfg.hidden, num_layers=2, batch_first=True, bidirectional=True)
            tdim = cfg.hidden * 2
        else:
            self.temporal = TemporalTCN(feat_dim, hidden=cfg.hidden, dropout=cfg.dropout)
            tdim = cfg.hidden

        self.attn = AdditiveAttention(tdim) if cfg.use_attention else None
        det_in = tdim
        tex_in = tdim
        self.det_head = nn.Linear(det_in, 2)
        self.tex_head = nn.Linear(tex_in, 4)

    def _band_forward(self, xb: torch.Tensor, key: str):
        z = self.band_modules[key](xb)
        z = z.mean(dim=2)  # average over mel bins -> (B,C,T)
        return z.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B,4,64,T)
        if not self.cfg.use_subbands:
            merged = x.mean(dim=1, keepdim=True)
            fake = self._band_forward(merged, "B2")
            feats = torch.cat([fake, fake, fake], dim=-1)
        else:
            # band_feats = []
            # for i, k in enumerate(["B1", "B2", "B3", "B4"]):
            #     if (k == "B1" and self.cfg.remove_b1) or (k == "B3" and self.cfg.remove_b3):
            #         continue
            #     xb = x[:, i : i + 1, :, :]
            #     # band_feats.append(self._band_forward(xb, k))
            #     min_t = min(b.shape[1] for b in band_feats)
            #     band_feats = [b[:, :min_t, :] for b in band_feats]
            # feats = torch.cat(band_feats, dim=-1)
            band_feats = []
            for i, k in enumerate(["B1", "B2", "B3", "B4"]):
                if (k == "B1" and self.cfg.remove_b1) or (k == "B3" and self.cfg.remove_b3):
                    continue
                xb = x[:, i : i + 1, :, :]
                band_feats.append(self._band_forward(xb, k))

            # safety check
            if len(band_feats) == 0:
                raise RuntimeError("No active subbands available in model configuration")

            min_t = min(b.shape[1] for b in band_feats)
            band_feats = [b[:, :min_t, :] for b in band_feats]

            feats = torch.cat(band_feats, dim=-1)

        if self.cfg.temporal == "bilstm":
            seq, _ = self.temporal(feats)
        else:
            seq = self.temporal(feats)

        if self.attn is not None:
            ctx, attn = self.attn(seq)
        else:
            ctx = seq.mean(dim=1)
            attn = torch.ones(seq.size(0), seq.size(1), device=seq.device) / seq.size(1)

        det_logits = self.det_head(seq)
        tex_logits = self.tex_head(ctx)
        return {"det_logits": det_logits, "tex_logits": tex_logits, "attn": attn}
