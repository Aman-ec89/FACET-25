"""Training and validation loops for multi-task learning (FINAL + METRICS + IMBALANCE FIX)."""

from dataclasses import dataclass
import numpy as np
import torch
import time
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, classification_report


# ==========================================
# CONFIG
# ==========================================
@dataclass
class TrainConfig:
    lr: float = 1e-4
    batch_size: int = 64
    epochs: int = 5
    patience: int = 3
    grad_clip: float = 1.5


# ==========================================
# LOSS (UPDATED: FOCAL + CLASS WEIGHTS)
# ==========================================
def multitask_loss(outputs, det_y, tex_y, class_weights):

    import torch.nn.functional as F

    logits = outputs["tex_logits"]

    # ---- Logit Adjustment (fix class bias) ----
    log_prior = torch.log(class_weights + 1e-8)
    # logits = logits + log_prior
    

    # ---- ORIGINAL LOSS (kept for reference) ----
    # tex_ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    # tex = tex_ce(logits, tex_y)
    # return tex, {"tex_loss": float(tex.item())}

    # ---- FOCAL LOSS + CLASS WEIGHTS ----
    ce = F.cross_entropy(
        logits,
        tex_y,
        weight=class_weights,
        reduction='none',
        label_smoothing=0.05
    )

    # pt = torch.exp(-ce)
    probs = torch.softmax(logits, dim=1)
    pt = probs.gather(1, tex_y.unsqueeze(1)).squeeze()
    focal = ((1 - pt) ** 2.0) * ce   # gamma = 2

    loss = focal.mean()

    return loss, {"tex_loss": float(loss.item())}


# ==========================================
# EPOCH
# ==========================================
def run_epoch(model, loader, optimizer, device, train, cfg, class_weights):

    model.train(train)

    losses = []
    tex_logits_all, tex_y_all = [], []

    for i, batch in enumerate(loader):

        x = batch["x"].to(device)
        tex_y = batch["tex_y"].to(device)

        with torch.set_grad_enabled(train):

            out = model(x)

            # ---- UPDATED LOSS CALL ----
            loss, _ = multitask_loss(out, None, tex_y, class_weights)

            if train:
                optimizer.zero_grad()
                loss.backward()

                # ---- Gradient Clipping (kept) ----
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                optimizer.step()

        losses.append(loss.item())

        tex_logits_all.append(out["tex_logits"].detach().cpu())
        tex_y_all.append(tex_y.detach().cpu())

        print(f"\rProgress: {(i+1)/len(loader)*100:5.1f}%", end="")

    tex_logits_all = torch.cat(tex_logits_all)
    tex_y_all = torch.cat(tex_y_all)

    return {
        "loss": np.mean(losses),
        "tex_logits": tex_logits_all,
        "tex_y": tex_y_all,
    }


# ==========================================
# TRAIN LOOP
# ==========================================
def train_model(model, train_loader, val_loader, device, cfg):

    model = model.to(device)

    # ---- CLASS WEIGHTS (GLOBAL, FIXED) ----
    counts = np.array([410, 660, 290, 490], dtype=np.float32)  # [soft, crunchy, brittle, fibrous]

    # weights = 1.0 / counts
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    weights = weights / weights.sum() * len(counts)

    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    print("Class Weights:", class_weights)

    # ---- OPTIMIZER ----
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)  # reduced from 1e-3

    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    best = float("inf")
    best_state = None
    stale = 0

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    final_cm = None
    final_report = None

    for epoch in range(cfg.epochs):

        start = time.time()

        # ---- PASS CLASS WEIGHTS ----
        train_out = run_epoch(model, train_loader, opt, device, True, cfg, class_weights)
        print()
        val_out = run_epoch(model, val_loader, opt, device, False, cfg, class_weights)
        print()

        pred = val_out["tex_logits"].argmax(1).numpy()
        true = val_out["tex_y"].numpy()

        acc = (pred == true).mean()

        # ==========================================
        # METRICS
        # ==========================================
        cm = confusion_matrix(true, pred)

        report = classification_report(
            true, pred, output_dict=True, zero_division=0
        )

        final_cm = cm
        final_report = report

        sched.step(val_out["loss"])

        history["train_loss"].append(train_out["loss"])
        history["val_loss"].append(val_out["loss"])
        history["val_acc"].append(acc)

        print(
            f"Epoch {epoch+1:03d} | "
            f"train={train_out['loss']:.4f} | "
            f"val={val_out['loss']:.4f} | "
            f"acc={acc:.3f} | "
            f"time={time.time()-start:.1f}s"
        )

        # ==========================================
        # EARLY STOPPING
        # ==========================================
        if val_out["loss"] < best:
            best = val_out["loss"]
            best_state = model.state_dict()
            stale = 0
        else:
            stale += 1

        if stale >= cfg.patience:
            print("Early stopping")
            break

    model.load_state_dict(best_state)

    return model, history, final_cm, final_report

# """Training and validation loops for multi-task learning (FINAL + METRICS)."""

# from dataclasses import dataclass
# import numpy as np
# import torch
# import time
# from torch import nn
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# from sklearn.metrics import confusion_matrix, classification_report


# # ==========================================
# # CONFIG
# # ==========================================
# @dataclass
# class TrainConfig:
#     lr: float = 1e-4
#     batch_size: int = 64
#     epochs: int = 5
#     patience: int = 3
#     grad_clip: float = 1.5


# # ==========================================
# # LOSS
# # ==========================================
# def multitask_loss(outputs, det_y, tex_y, tex_weight=None):

#     tex_ce = nn.CrossEntropyLoss(label_smoothing=0.05)
#     tex = tex_ce(outputs["tex_logits"], tex_y)

#     return tex, {"tex_loss": float(tex.item())}


# # ==========================================
# # EPOCH
# # ==========================================
# def run_epoch(model, loader, optimizer, device, train, cfg):

#     model.train(train)

#     losses = []
#     tex_logits_all, tex_y_all = [], []

#     for i, batch in enumerate(loader):

#         x = batch["x"].to(device)
#         tex_y = batch["tex_y"].to(device)

#         with torch.set_grad_enabled(train):

#             out = model(x)
#             loss, _ = multitask_loss(out, None, tex_y)

#             if train:
#                 optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
#                 optimizer.step()

#         losses.append(loss.item())

#         tex_logits_all.append(out["tex_logits"].detach().cpu())
#         tex_y_all.append(tex_y.detach().cpu())

#         print(f"\rProgress: {(i+1)/len(loader)*100:5.1f}%", end="")

#     tex_logits_all = torch.cat(tex_logits_all)
#     tex_y_all = torch.cat(tex_y_all)

#     return {
#         "loss": np.mean(losses),
#         "tex_logits": tex_logits_all,
#         "tex_y": tex_y_all,
#     }


# # ==========================================
# # TRAIN LOOP
# # ==========================================
# def train_model(model, train_loader, val_loader, device, cfg):

#     model = model.to(device)

#     opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-3)
#     sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

#     best = float("inf")
#     best_state = None
#     stale = 0

#     history = {"train_loss": [], "val_loss": [], "val_acc": []}

#     final_cm = None
#     final_report = None

#     for epoch in range(cfg.epochs):

#         start = time.time()

#         train_out = run_epoch(model, train_loader, opt, device, True, cfg)
#         print()
#         val_out = run_epoch(model, val_loader, opt, device, False, cfg)
#         print()

#         pred = val_out["tex_logits"].argmax(1).numpy()
#         true = val_out["tex_y"].numpy()

#         acc = (pred == true).mean()

#         # 🔥 CONFUSION MATRIX
#         cm = confusion_matrix(true, pred)

#         # 🔥 METRICS
#         report = classification_report(
#             true, pred, output_dict=True, zero_division=0
#         )

#         final_cm = cm
#         final_report = report

#         sched.step(val_out["loss"])

#         history["train_loss"].append(train_out["loss"])
#         history["val_loss"].append(val_out["loss"])
#         history["val_acc"].append(acc)

#         print(f"Epoch {epoch+1:03d} | train={train_out['loss']:.4f} | val={val_out['loss']:.4f} | acc={acc:.3f} | time={time.time()-start:.1f}s")

#         if val_out["loss"] < best:
#             best = val_out["loss"]
#             best_state = model.state_dict()
#             stale = 0
#         else:
#             stale += 1

#         if stale >= cfg.patience:
#             print("Early stopping")
#             break

#     model.load_state_dict(best_state)

#     return model, history, final_cm, final_report
