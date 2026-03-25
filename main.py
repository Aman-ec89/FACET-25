"""Entry point for end-to-end chewing classification experiments (Colab-ready)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_loader import (
    kaggle_pretrain_split,
    load_kaggle_records,
    load_recorded_records,
    loso_splits,
    make_loader,
)
from evaluation import (
    make_confusion_plot,
    summarize_fold_metrics,
)
from metrics import texture_confusion, texture_metrics
from model import FrequencyAwareMultiTaskNet, ModelConfig
from preprocessing import PreprocessConfig
from training import TrainConfig, train_model
from utils import device_auto, ensure_dir, set_seed


# =========================================================
# SAFE MERGE
# =========================================================
def safe_merge(out_list):

    merged = {}

    for k in out_list[0]:
        tensors = [o[k] for o in out_list if o[k] is not None]

        if len(tensors) == 0:
            continue

        merged[k] = torch.cat(tensors, dim=0)

    return merged


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

    # ========================
    # LOAD DATA
    # ========================
    kaggle_records = load_kaggle_records(args.kaggle_dir)
    recorded_records = load_recorded_records(args.recorded_dir)

    ktr, kva = kaggle_pretrain_split(kaggle_records, seed=args.seed)

    pretrain_train = make_loader(ktr, p_cfg, t_cfg.batch_size, True)
    pretrain_val = make_loader(kva, p_cfg, t_cfg.batch_size, False)

    # ========================
    # 🔍 LABEL CHECK (CRITICAL)
    # ========================
    print("\n🔍 Checking label distribution...")

    from collections import Counter
    all_labels = []

    for batch in pretrain_train:
        all_labels.extend(batch["tex_y"].cpu().numpy())

    print("Train label distribution:", Counter(all_labels))

    # ========================
    # Stage 1 (Pretraining)
    # ========================
    print("\n[Stage 1] Pretraining on Kaggle")

    base_model = FrequencyAwareMultiTaskNet(m_cfg).to(device)
    base_model = train_model(base_model, pretrain_train, pretrain_val, device, t_cfg)

    # ========================
    # Stage 2 (LOSO)
    # ========================
    print("\n[Stage 2] LOSO Training")

    fold_tex = []
    cm_sum = np.zeros((4, 4), dtype=int)

    for tr_records, te_records, sid in loso_splits(recorded_records):

        print(f"\nLOSO subject={sid}")

        tr_loader = make_loader(tr_records, p_cfg, t_cfg.batch_size, True, rate_csv=args.rate_csv)
        te_loader = make_loader(te_records, p_cfg, t_cfg.batch_size, False, rate_csv=args.rate_csv)

        model = FrequencyAwareMultiTaskNet(m_cfg).to(device)
        model.load_state_dict(base_model.state_dict(), strict=False)

        model = train_model(model, tr_loader, te_loader, device, t_cfg)

        # -------- Evaluation --------
        outputs = []

        model.eval()
        with torch.no_grad():
            for b in te_loader:
                x = b["x"].to(device)
                pred = model(x)

                outputs.append({
                    "tex_logits": pred["tex_logits"].cpu(),
                    "tex_y": b["tex_y"],
                })

        if len(outputs) == 0:
            print(f"⚠️ Skipping subject {sid}")
            continue

        merged = safe_merge(outputs)

        tex_pred = merged["tex_logits"].argmax(dim=1).numpy()
        tex_true = merged["tex_y"].numpy()

        tmet = texture_metrics(tex_true, tex_pred)
        tmet["fold"] = sid

        fold_tex.append(tmet)

        cm_sum += texture_confusion(tex_true, tex_pred)

    # ========================
    # RESULTS
    # ========================
    tex_df, _ = summarize_fold_metrics(fold_tex, out_dir, "texture_results")

    make_confusion_plot(cm_sum, out_dir / "texture_confusion_matrix.png")

    print(f"\nDone. Outputs saved to: {out_dir}")


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--recorded_dir", type=str, required=True)
    ap.add_argument("--kaggle_dir", type=str, required=True)
    ap.add_argument("--rate_csv", type=str, default="")
    ap.add_argument("--output_dir", type=str, default="outputs")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    run(ap.parse_args())
