"""FINAL MAIN (DIRECT TRAINING, NO LOSO/LOOCV)"""

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt 
import pandas as pd
import os

from data_loader import make_loader
# from model import FrequencyAwareMultiTaskNet, ModelConfig
from training import TrainConfig, train_model


# ==========================================
# MAIN
# ==========================================
def run(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = list(Path(args.recorded_dir).glob("*.npy"))
    print("Total samples:", len(files))

    os.makedirs("outputs", exist_ok=True)

    # ==========================================
    # 🔴 REMOVE LOSO / LOOCV (COMMENTED)
    # ==========================================
    # subjects = sorted(set([f.stem.split("_")[0] for f in files]))
    # results = {}
    # for subject in subjects:
    #     ...

    # ==========================================
    # ✅ DIRECT TRAIN / VAL SPLIT
    # ==========================================
    np.random.seed(42)
    np.random.shuffle(files)

    split = int(0.8 * len(files))
    train_files = files[:split]
    val_files = files[split:]

    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # ==========================================
    # CLASS WEIGHTS
    # ==========================================
    counts = np.array([410, 660, 290, 490], dtype=np.float32)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # ==========================================
    # LOADERS
    # ==========================================
    train_loader = make_loader(train_files, args.batch_size, True, class_weights)
    val_loader   = make_loader(val_files, args.batch_size, False)

    # ==========================================
    # MODEL
    # ==========================================
    cfg_model = ModelConfig()
    model = FrequencyAwareMultiTaskNet(cfg_model).to(device)

    # ==========================================
    # TRAIN CONFIG
    # ==========================================
    cfg_train = TrainConfig()
    cfg_train.epochs = args.epochs

    # ==========================================
    # TRAIN
    # ==========================================
    model, history, cm, report = train_model(
        model, train_loader, val_loader, device, cfg_train
    )

    # ==========================================
    # SAVE PLOTS
    # ==========================================
    plot_dir = "outputs/plots"
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.legend()
    plt.title("Loss")
    plt.savefig(f"{plot_dir}/loss.png")
    plt.close()

    plt.figure()
    plt.plot(history["val_acc"])
    plt.title("Accuracy")
    plt.savefig(f"{plot_dir}/accuracy.png")
    plt.close()

    # ==========================================
    # SAVE METRICS
    # ==========================================
    best_acc = max(history["val_acc"])

    summary = {
        "best_acc": best_acc
    }

    pd.DataFrame([summary]).to_csv("outputs/final_summary.csv", index=False)

    print(f"\n✅ Final Accuracy: {best_acc:.3f}")
    print("\nSaved all results in outputs/")


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--recorded_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)

    run(ap.parse_args())

# """Frequency + Temporal Dual Attention Model (FINAL CLEAN VERSION)"""

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, List

# import torch
# from torch import nn

# from attention import AdditiveAttention


# # ==========================================
# # 🔥 FREQUENCY ATTENTION
# # ==========================================
# class FrequencyAttention(nn.Module):
#     def __init__(self, freq_bins=64):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(freq_bins, freq_bins // 2),
#             nn.ReLU(),
#             nn.Linear(freq_bins // 2, freq_bins),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # x: (B,C,F,T)
#         w = x.mean(dim=3)       # (B,C,F)
#         w = self.fc(w)          # (B,C,F)
#         w = w.unsqueeze(-1)     # (B,C,F,1)
#         return x * w


# # ==========================================
# # 🔥 TEMPORAL SELF-ATTENTION
# # ==========================================
# class TemporalSelfAttention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.q = nn.Linear(dim, dim)
#         self.k = nn.Linear(dim, dim)
#         self.v = nn.Linear(dim, dim)
#         self.scale = dim ** 0.5

#     def forward(self, x):
#         # x: (B,T,D)
#         Q = self.q(x)
#         K = self.k(x)
#         V = self.v(x)

#         attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
#         out = torch.matmul(attn, V)

#         return out


# # ==========================================
# # CONV STACK
# # ==========================================
# class ConvStack(nn.Module):
#     def __init__(self, in_ch: int, channels: int, layers: int, kernel: tuple[int, int], dropout: float):
#         super().__init__()
#         mods: List[nn.Module] = []
#         c = in_ch
#         pad = (kernel[0] // 2, kernel[1] // 2)

#         for _ in range(layers):
#             mods += [
#                 nn.Conv2d(c, channels, kernel_size=kernel, padding=pad),
#                 nn.BatchNorm2d(channels),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(dropout),
#             ]
#             c = channels

#         self.net = nn.Sequential(*mods)

#     def forward(self, x):
#         return self.net(x)


# # ==========================================
# # TEMPORAL MODEL
# # ==========================================
# class TemporalTCN(nn.Module):
#     def __init__(self, in_dim: int, hidden: int = 128, dilations=(1, 2, 4, 8), dropout: float = 0.3):
#         super().__init__()
#         layers = []
#         c = in_dim

#         for d in dilations:
#             layers += [
#                 nn.Conv1d(c, hidden, kernel_size=3, padding=d, dilation=d),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(dropout),
#             ]
#             c = hidden

#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         x = x.transpose(1, 2)
#         y = self.net(x)
#         return y.transpose(1, 2)


# # ==========================================
# # CONFIG
# # ==========================================
# @dataclass
# class ModelConfig:
#     dropout: float = 0.5
#     hidden: int = 256
#     use_attention: bool = True
#     use_subbands: bool = True
#     use_freq_attention: bool = False
#     use_temporal_self_attention: bool = False   # 🔥 NEW
#     temporal: str = "bilstm"
#     remove_b1: bool = False
#     remove_b3: bool = False


# # ==========================================
# # MAIN MODEL
# # ==========================================
# class FrequencyAwareMultiTaskNet(nn.Module):
#     def __init__(self, cfg: ModelConfig):
#         super().__init__()
#         self.cfg = cfg

#         # 🔥 Frequency attention
#         self.freq_attn = FrequencyAttention() if cfg.use_freq_attention else None

#         # BAND MODELS
#         self.band_modules = nn.ModuleDict(
#             {
#                 "B1": ConvStack(5, 16, 2, (7, 3), cfg.dropout),
#                 "B2": ConvStack(5, 24, 3, (5, 3), cfg.dropout),
#                 "B3": ConvStack(5, 32, 4, (3, 3), cfg.dropout),
#                 "B4": ConvStack(5, 32, 4, (3, 1), cfg.dropout),
#             }
#         )

#         feat_dim = 16 + 24 + 32 + 32

#         if cfg.remove_b1:
#             feat_dim -= 16
#         if cfg.remove_b3:
#             feat_dim -= 32

#         # TEMPORAL
#         if cfg.temporal == "bilstm":
#             self.temporal = nn.LSTM(
#                 feat_dim,
#                 cfg.hidden,
#                 num_layers=2,
#                 batch_first=True,
#                 bidirectional=True,
#             )
#             tdim = cfg.hidden * 2
#         else:
#             self.temporal = TemporalTCN(feat_dim, hidden=cfg.hidden, dropout=cfg.dropout)
#             tdim = cfg.hidden

#         # 🔥 Temporal self-attention
#         self.self_attn = TemporalSelfAttention(tdim) if cfg.use_temporal_self_attention else None

#         # Additive attention (kept)
#         self.attn = AdditiveAttention(tdim) if cfg.use_attention else None

#         # HEADS
#         self.det_head = nn.Linear(tdim, 2)
#         self.tex_head = nn.Linear(tdim, 4)

#     # ==========================================
#     # BAND FORWARD
#     # ==========================================
#     def _band_forward(self, xb: torch.Tensor, key: str):

#         if self.freq_attn is not None:
#             xb = self.freq_attn(xb)

#         z = self.band_modules[key](xb)
#         z = z.mean(dim=2)
#         return z.transpose(1, 2)

#     # ==========================================
#     # FORWARD
#     # ==========================================
#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

#         band_feats = []

#         for i, k in enumerate(["B1", "B2", "B3", "B4"]):

#             if (k == "B1" and self.cfg.remove_b1) or (k == "B3" and self.cfg.remove_b3):
#                 continue

#             xb = x[:, i * 5:(i + 1) * 5, :, :]
#             band_feats.append(self._band_forward(xb, k))

#         min_t = min(b.shape[1] for b in band_feats)
#         band_feats = [b[:, :min_t, :] for b in band_feats]

#         feats = torch.cat(band_feats, dim=-1)

#         # TEMPORAL
#         if self.cfg.temporal == "bilstm":
#             seq, _ = self.temporal(feats)
#         else:
#             seq = self.temporal(feats)

#         # 🔥 SELF ATTENTION
#         if self.self_attn is not None:
#             seq = self.self_attn(seq)

#         # ADDITIVE ATTENTION
#         if self.attn is not None:
#             ctx, attn = self.attn(seq)
#         else:
#             ctx = seq.mean(dim=1)
#             attn = None

#         det_logits = self.det_head(seq)
#         tex_logits = self.tex_head(ctx)

#         return {
#             "det_logits": det_logits,
#             "tex_logits": tex_logits,
#             "attn": attn,
#         }

# # """Frequency-aware multi-task chewing classification models (FINAL FIXED + FREQ ATTENTION WORKING)."""

# # from __future__ import annotations

# # from dataclasses import dataclass
# # from typing import Dict, List

# # import torch
# # from torch import nn

# # from attention import AdditiveAttention


# # # ==========================================
# # # 🔥 FREQUENCY ATTENTION
# # # ==========================================
# # class FrequencyAttention(nn.Module):
# #     def __init__(self, freq_bins=64):
# #         super().__init__()
# #         self.fc = nn.Sequential(
# #             nn.Linear(freq_bins, freq_bins // 2),
# #             nn.ReLU(),
# #             nn.Linear(freq_bins // 2, freq_bins),
# #             nn.Sigmoid()
# #         )

# #     def forward(self, x):
# #         # x: (B,C,F,T)

# #         w = x.mean(dim=3)       # (B,C,F)
# #         w = self.fc(w)          # (B,C,F)
# #         w = w.unsqueeze(-1)     # (B,C,F,1)

# #         return x * w


# # # ==========================================
# # # CONV STACK
# # # ==========================================
# # class ConvStack(nn.Module):
# #     def __init__(self, in_ch: int, channels: int, layers: int, kernel: tuple[int, int], dropout: float):
# #         super().__init__()
# #         mods: List[nn.Module] = []
# #         c = in_ch
# #         pad = (kernel[0] // 2, kernel[1] // 2)

# #         for _ in range(layers):
# #             mods += [
# #                 nn.Conv2d(c, channels, kernel_size=kernel, padding=pad),
# #                 nn.BatchNorm2d(channels),
# #                 nn.ReLU(inplace=True),
# #                 nn.Dropout(dropout),
# #             ]
# #             c = channels

# #         self.net = nn.Sequential(*mods)

# #     def forward(self, x):
# #         return self.net(x)


# # # ==========================================
# # # TEMPORAL MODEL
# # # ==========================================
# # class TemporalTCN(nn.Module):
# #     def __init__(self, in_dim: int, hidden: int = 128, dilations=(1, 2, 4, 8), dropout: float = 0.3):
# #         super().__init__()
# #         layers = []
# #         c = in_dim

# #         for d in dilations:
# #             layers += [
# #                 nn.Conv1d(c, hidden, kernel_size=3, padding=d, dilation=d),
# #                 nn.ReLU(inplace=True),
# #                 nn.Dropout(dropout),
# #             ]
# #             c = hidden

# #         self.net = nn.Sequential(*layers)

# #     def forward(self, x):
# #         x = x.transpose(1, 2)
# #         y = self.net(x)
# #         return y.transpose(1, 2)


# # # ==========================================
# # # CONFIG
# # # ==========================================
# # @dataclass
# # class ModelConfig:
# #     dropout: float = 0.5
# #     hidden: int = 256
# #     use_attention: bool = True
# #     use_subbands: bool = True
# #     use_freq_attention: bool = False   # 🔥 ENABLE SWITCH
# #     temporal: str = "bilstm"
# #     remove_b1: bool = False
# #     remove_b3: bool = False


# # # ==========================================
# # # MAIN MODEL
# # # ==========================================
# # class FrequencyAwareMultiTaskNet(nn.Module):
# #     def __init__(self, cfg: ModelConfig):
# #         super().__init__()
# #         self.cfg = cfg

# #         # 🔥 FREQUENCY ATTENTION MODULE
# #         self.freq_attn = FrequencyAttention() if cfg.use_freq_attention else None

# #         # BAND MODELS
# #         self.band_modules = nn.ModuleDict(
# #             {
# #                 "B1": ConvStack(5, 16, 2, (7, 3), cfg.dropout),
# #                 "B2": ConvStack(5, 24, 3, (5, 3), cfg.dropout),
# #                 "B3": ConvStack(5, 32, 4, (3, 3), cfg.dropout),
# #                 "B4": ConvStack(5, 32, 4, (3, 1), cfg.dropout),
# #             }
# #         )

# #         feat_dim = 16 + 24 + 32 + 32

# #         if cfg.remove_b1:
# #             feat_dim -= 16
# #         if cfg.remove_b3:
# #             feat_dim -= 32

# #         # TEMPORAL
# #         if cfg.temporal == "bilstm":
# #             self.temporal = nn.LSTM(
# #                 feat_dim,
# #                 cfg.hidden,
# #                 num_layers=2,
# #                 batch_first=True,
# #                 bidirectional=True,
# #             )
# #             tdim = cfg.hidden * 2
# #         else:
# #             self.temporal = TemporalTCN(feat_dim, hidden=cfg.hidden, dropout=cfg.dropout)
# #             tdim = cfg.hidden

# #         # ATTENTION
# #         self.attn = AdditiveAttention(tdim) if cfg.use_attention else None

# #         # HEADS
# #         self.det_head = nn.Linear(tdim, 2)
# #         self.tex_head = nn.Linear(tdim, 4)

# #     # ==========================================
# #     # BAND FORWARD (🔥 FIXED)
# #     # ==========================================
# #     def _band_forward(self, xb: torch.Tensor, key: str):

# #         # 🔥 APPLY FREQUENCY ATTENTION HERE
# #         if self.freq_attn is not None:
# #             xb = self.freq_attn(xb)

# #         z = self.band_modules[key](xb)  # (B,C,F,T)
# #         z = z.mean(dim=2)               # (B,C,T)
# #         return z.transpose(1, 2)        # (B,T,C)

# #     # ==========================================
# #     # FORWARD
# #     # ==========================================
# #     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

# #         if not self.cfg.use_subbands:
# #             merged = x.mean(dim=1, keepdim=True)
# #             fake = self._band_forward(merged, "B2")
# #             feats = torch.cat([fake, fake, fake], dim=-1)

# #         else:
# #             band_feats = []

# #             for i, k in enumerate(["B1", "B2", "B3", "B4"]):

# #                 if (k == "B1" and self.cfg.remove_b1) or (k == "B3" and self.cfg.remove_b3):
# #                     continue

# #                 start = i * 5
# #                 end = (i + 1) * 5
# #                 xb = x[:, start:end, :, :]  # (B,5,64,T)

# #                 band_feats.append(self._band_forward(xb, k))

# #             if len(band_feats) == 0:
# #                 raise RuntimeError("No active subbands available")

# #             min_t = min(b.shape[1] for b in band_feats)
# #             band_feats = [b[:, :min_t, :] for b in band_feats]

# #             feats = torch.cat(band_feats, dim=-1)

# #         # TEMPORAL
# #         if self.cfg.temporal == "bilstm":
# #             seq, _ = self.temporal(feats)
# #         else:
# #             seq = self.temporal(feats)

# #         # ATTENTION
# #         if self.attn is not None:
# #             ctx, attn = self.attn(seq)
# #         else:
# #             ctx = seq.mean(dim=1)
# #             attn = torch.ones(seq.size(0), seq.size(1), device=seq.device) / seq.size(1)

# #         # OUTPUTS
# #         det_logits = self.det_head(seq)
# #         tex_logits = self.tex_head(ctx)

# #         return {
# #             "det_logits": det_logits,
# #             "tex_logits": tex_logits,
# #             "attn": attn,
# #         }
