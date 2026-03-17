"""Entry point for end-to-end chewing classification experiments (Colab-ready)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ablation import ablation_variants
from data_loader import kaggle_pretrain_split, load_kaggle_records, load_recorded_records, loso_splits, make_loader
from evaluation import (
    compute_wilcoxon,
    make_ablation_plot,
    make_confusion_plot,
    make_rate_scatter,
    make_roc_plot,
    make_attention_plot,
    make_psd_subband_plot,
    summarize_fold_metrics,
)
from metrics import chewing_rate_metrics, detection_metrics, texture_confusion, texture_metrics
from model import FrequencyAwareMultiTaskNet, ModelConfig
from preprocessing import PreprocessConfig
from rate_estimation import estimate_chewing_rate_bpm
from training import TrainConfig, train_model
from utils import count_parameters, device_auto, ensure_dir, set_seed


# =========================================================
# SAFE MERGE FUNCTION (FIXED)
# =========================================================
def safe_merge(out_list):
    merged = {}

    for k in out_list[0]:

        if k in ["file", "signal"]:
            merged[k] = sum([o[k] for o in out_list], [])

        else:
            tensors = [o[k] for o in out_list]

            try:
                merged[k] = torch.cat(tensors, dim=0)

            except RuntimeError:
                # pad along LAST dimension only
                max_t = max(t.shape[-1] for t in tensors)

                padded = []
                for t in tensors:
                    pad_size = max_t - t.shape[-1]

                    if pad_size > 0:
                        t = F.pad(t, (0, pad_size))  # pad last dim

                    padded.append(t)

                merged[k] = torch.cat(padded, dim=0)

    return merged


# =========================================================
# LOGIT PROCESSING
# =========================================================
def _eval_logits(out):
    det_prob = torch.softmax(out["det_logits"], dim=-1)[..., 1].cpu().numpy().reshape(-1)
    det_pred = out["det_logits"].argmax(-1).cpu().numpy().reshape(-1)
    det_true = out["det_y"].cpu().numpy().reshape(-1)

    tex_pred = out["tex_logits"].argmax(-1).cpu().numpy().reshape(-1)
    tex_true = out["tex_y"].cpu().numpy().reshape(-1)

    n = min(len(tex_true), len(tex_pred))
    return det_true, det_pred, det_prob, tex_true[:n], tex_pred[:n]


# =========================================================
# MAIN RUN
# =========================================================
def run(args):
    set_seed(args.seed)
    device = device_auto()
    out_dir = ensure_dir(args.output_dir)

    p_cfg = PreprocessConfig()
    t_cfg = TrainConfig(batch_size=args.batch_size, epochs=args.epochs)
    m_cfg = ModelConfig()

    kaggle_records = load_kaggle_records(args.kaggle_dir)
    recorded_records = load_recorded_records(args.recorded_dir)

    ktr, kva = kaggle_pretrain_split(kaggle_records, seed=args.seed)
    pretrain_train = make_loader(ktr, p_cfg, t_cfg.batch_size, True)
    pretrain_val = make_loader(kva, p_cfg, t_cfg.batch_size, False)

    # ========================
    # Stage 1
    # ========================
    base_model = FrequencyAwareMultiTaskNet(m_cfg).to(device)
    print("[Stage 1] pretraining on Kaggle")
    base_model = train_model(base_model, pretrain_train, pretrain_val, device, t_cfg)

    fold_det, fold_tex, fold_rate = [], [], []
    roc_true_all, roc_prob_all = [], []
    cm_sum = np.zeros((4, 4), dtype=int)

    # ========================
    # Stage 2 (LOSO)
    # ========================
    for tr_records, te_records, sid in loso_splits(recorded_records):
        print(f"[Stage 2] LOSO subject={sid}")

        tr_loader = make_loader(tr_records, p_cfg, t_cfg.batch_size, True, rate_csv=args.rate_csv)
        te_loader = make_loader(te_records, p_cfg, t_cfg.batch_size, False, rate_csv=args.rate_csv)

        model = FrequencyAwareMultiTaskNet(m_cfg).to(device)
        model.load_state_dict(base_model.state_dict(), strict=False)
        model = train_model(model, tr_loader, te_loader, device, t_cfg)

        # -------- Evaluation --------
        out = []
        model.eval()
        with torch.no_grad():
            for b in te_loader:
                x = b["x"].to(device)
                pred = model(x)
                out.append(
                    {
                        "det_logits": pred["det_logits"].cpu(),
                        "tex_logits": pred["tex_logits"].cpu(),
                        "det_y": b["det_y"],
                        "tex_y": b["tex_y"],
                    }
                )

        merged = safe_merge(out)

        det_true, det_pred, det_prob, tex_true, tex_pred = _eval_logits(merged)

        dmet = detection_metrics(det_true, det_pred, det_prob)
        tmet = texture_metrics(tex_true, tex_pred)

        dmet["fold"] = sid
        tmet["fold"] = sid

        fold_det.append(dmet)
        fold_tex.append(tmet)

        roc_true_all.extend(det_true.tolist())
        roc_prob_all.extend(det_prob.tolist())

        cm_sum += texture_confusion(tex_true, tex_pred)

    # ========================
    # Results
    # ========================
    det_df, _ = summarize_fold_metrics(fold_det, out_dir, "detection_results")
    tex_df, _ = summarize_fold_metrics(fold_tex, out_dir, "texture_results")

    make_roc_plot(np.array(roc_true_all), np.array(roc_prob_all), out_dir / "roc_curve.png")
    make_confusion_plot(cm_sum, out_dir / "texture_confusion_matrix.png")

    print(f"Done. Outputs saved to: {out_dir}")


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorded_dir", type=str, default="Recorded audio")
    ap.add_argument("--kaggle_dir", type=str, default="Kaggle audio")
    ap.add_argument("--rate_csv", type=str, default="chewing_rate.csv")
    ap.add_argument("--output_dir", type=str, default="outputs")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    run(ap.parse_args())
