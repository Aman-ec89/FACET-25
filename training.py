"""Training and validation loops for multi-task learning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import time
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


@dataclass
class TrainConfig:
    lr: float = 1e-4
    batch_size: int = 32
    epochs: int = 80
    patience: int = 10
    grad_clip: float = 5.0
    alpha_det: float = 0.6
    alpha_tex: float = 0.4


def multitask_loss(outputs, det_y, tex_y, det_weight=None, tex_weight=None, alpha_det=0.6, alpha_tex=0.4):
    det_ce = nn.CrossEntropyLoss(weight=det_weight)
    tex_ce = nn.CrossEntropyLoss(weight=tex_weight)
    det = det_ce(outputs["det_logits"].reshape(-1, 2), det_y.reshape(-1))
    tex = tex_ce(outputs["tex_logits"], tex_y)
    return alpha_det * det + alpha_tex * tex, {"det_loss": float(det.item()), "tex_loss": float(tex.item())}


def run_epoch(model, loader, optimizer, device, train: bool, cfg: TrainConfig):
    model.train(train)
    losses = []
    det_logits_all, det_y_all = [], []
    tex_logits_all, tex_y_all = [], []

    for batch in loader:
        x = batch["x"].to(device)
        det_y = batch["det_y"].to(device)
        tex_y = batch["tex_y"].to(device)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss, _ = multitask_loss(out, det_y, tex_y, alpha_det=cfg.alpha_det, alpha_tex=cfg.alpha_tex)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

        losses.append(loss.item())
        det_logits_all.append(out["det_logits"].detach().cpu())
        tex_logits_all.append(out["tex_logits"].detach().cpu())
        det_y_all.append(det_y.detach().cpu())
        tex_y_all.append(tex_y.detach().cpu())

    # return {
        # "loss": float(np.mean(losses)),
        # "det_logits": torch.cat(det_logits_all, 0),
        # "det_y": torch.cat(det_y_all, 0),
        # "tex_logits": torch.cat(tex_logits_all, 0),
        # "tex_y": torch.cat(tex_y_all, 0),
    # }
    return {
        "loss": float(np.mean(losses)),
        "det_logits": torch.cat([x.reshape(-1, x.shape[-1]) for x in det_logits_all], 0),
        "det_y": torch.cat([x.reshape(-1) for x in det_y_all], 0),
        "tex_logits": torch.cat(tex_logits_all, 0),
        "tex_y": torch.cat(tex_y_all, 0),
    }

def train_model(model, train_loader, val_loader, device, cfg: TrainConfig):
    opt = AdamW(model.parameters(), lr=cfg.lr)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    best = float("inf")
    best_state = None
    stale = 0

    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        train_out = run_epoch(model, train_loader, opt, device, True, cfg)
        val_out = run_epoch(model, val_loader, opt, device, False, cfg)
        # Compute validation accuracy (texture classification)
        val_pred = torch.argmax(val_out["tex_logits"], dim=1)
        val_acc = (val_pred == val_out["tex_y"]).float().mean().item()
        sched.step(val_out["loss"])
        epoch_time = time.time() - epoch_start

        if val_out["loss"] < best:
            best = val_out["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break

        # print(f"Epoch {epoch+1:03d} | train={train_out['loss']:.4f} val={val_out['loss']:.4f}")
        print(
            f"Epoch {epoch+1:03d} | "
            f"train_loss={train_out['loss']:.4f} | "
            f"val_loss={val_out['loss']:.4f} | "
            f"val_acc={val_acc:.3f} | "
            f"time={epoch_time:.2f}s"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
