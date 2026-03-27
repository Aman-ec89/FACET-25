"""Training and validation loops for multi-task learning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import time
from torch import nn
from torch.optim import AdamW
# from torch.optim.lr_scheduler import ReduceLROnPlateau   # ❌ disabled


# ==========================================
# CONFIG
# ==========================================
@dataclass
class TrainConfig:
    lr: float = 3e-4   # 🔥 FIXED (was 1e-4, 1e-3)
    batch_size: int = 128
    epochs: int = 30
    patience: int = 10
    grad_clip: float = 1.5
    alpha_det: float = 0.0
    alpha_tex: float = 1.0


# ==========================================
# CLASS WEIGHTS (DISABLED)
# ==========================================
def get_class_weights(device):
    # ❌ NOT USED ANYMORE (balanced dataset)
    class_counts = torch.tensor([449, 610, 361, 240], dtype=torch.float32)
    weights = class_counts.sum() / class_counts
    return weights.to(device)


# ==========================================
# LOSS FUNCTION (FIXED)
# ==========================================
def multitask_loss(outputs, det_y, tex_y, tex_weight=None):

    # ❌ OLD (hurting learning)
    # tex_ce = nn.CrossEntropyLoss(weight=tex_weight)

    # ✅ NEW (correct)
    tex_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    tex = tex_ce(outputs["tex_logits"], tex_y)

    return tex, {
        "det_loss": 0.0,
        "tex_loss": float(tex.item()),
    }


# ==========================================
# EPOCH RUNNER
# ==========================================
def run_epoch(model, loader, optimizer, device, train: bool, cfg: TrainConfig, tex_weights):

    model.train(train)

    losses = []
    tex_logits_all, tex_y_all = [], []

    total_batches = len(loader)

    for i, batch in enumerate(loader):

        x = batch["x"].to(device, non_blocking=True)
        det_y = batch["det_y"].to(device, non_blocking=True)
        tex_y = batch["tex_y"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):

            out = model(x)

            loss, _ = multitask_loss(
                out,
                det_y,
                tex_y,
                tex_weight=tex_weights,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()

                # ==========================================
                # ❌ GRAD DEBUG DISABLED (was noisy)
                # ==========================================
                # total_norm = 0.0
                # for p in model.parameters():
                #     if p.grad is not None:
                #         total_norm += p.grad.norm().item()
                # print("Grad norm:", round(total_norm, 4))

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

        losses.append(loss.item())

        tex_logits_all.append(out["tex_logits"].detach().cpu())
        tex_y_all.append(tex_y.detach().cpu())

        # ==========================================
        # ✅ PROGRESS %
        # ==========================================
        progress = (i + 1) / total_batches * 100
        print(f"\rProgress: {progress:5.1f}%", end="", flush=True)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "det_logits": None,
        "det_y": None,
        "tex_logits": torch.cat(tex_logits_all, 0),
        "tex_y": torch.cat(tex_y_all, 0),
    }


# ==========================================
# TRAIN LOOP
# ==========================================
def train_model(model, train_loader, val_loader, device, cfg: TrainConfig):

    model = model.to(device)

    # ❌ NOT NEEDED FOR FINAL OUTPUT
    # print("Model running on:", next(model.parameters()).device)

    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # ❌ DISABLED
    # sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    tex_weights = get_class_weights(device)

    best = float("inf")
    best_state = None
    stale = 0

    for epoch in range(cfg.epochs):

        epoch_start = time.time()

        # ❌ REMOVE EXTRA HEADERS
        # print(f"\n\n===== Epoch {epoch+1}/{cfg.epochs} =====")

        train_out = run_epoch(model, train_loader, opt, device, True, cfg, tex_weights)
        print()  # newline after progress

        val_out = run_epoch(model, val_loader, opt, device, False, cfg, tex_weights)
        print()

        val_pred = torch.argmax(val_out["tex_logits"], dim=1)

        # ❌ DEBUG REMOVED
        # print("Pred classes:", torch.unique(val_pred))

        val_acc = (val_pred == val_out["tex_y"]).float().mean().item()

        # ❌ GPU DEBUG REMOVED
        # if torch.cuda.is_available():
        #     print("GPU mem (GB):", round(torch.cuda.memory_allocated() / 1e9, 3))

        epoch_time = time.time() - epoch_start

        if val_out["loss"] < best:
            best = val_out["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        # ✅ FINAL OUTPUT (ONLY THIS WILL SHOW)
        print(
            f"Epoch {epoch+1:03d} | "
            f"train_loss={train_out['loss']:.4f} | "
            f"val_loss={val_out['loss']:.4f} | "
            f"val_acc={val_acc:.3f} | "
            f"time={epoch_time:.2f}s"
        )

        if stale >= cfg.patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
