"""Entry point for end-to-end chewing classification experiments (RECORDED-ONLY LOSO + PLOTS)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_loader import load_recorded_records, loso_splits, make_loader
from evaluation import make_confusion_plot, summarize_fold_metrics
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

    print("\n📂 Loading RECORDED dataset only...")
    recorded_records = load_recorded_records(args.recorded_dir)
    print("Total recorded samples:", len(recorded_records))

    print("\n⚠️ Pretraining disabled (pure LOSO setup)")

    print("\n[Stage 2] LOSO Training (RECORDED ONLY)")

    fold_tex = []
    cm_sum = np.zeros((4, 4), dtype=int)

    # ==========================================
    # 🔥 GLOBAL COLLECTION
    # ==========================================
    all_preds = []
    all_labels = []
    subject_acc = []

    for tr_records, te_records, sid in loso_splits(recorded_records):

        print(f"\nLOSO subject={sid}")

        tr_loader = make_loader(tr_records, p_cfg, t_cfg.batch_size, True, rate_csv=args.rate_csv)
        te_loader = make_loader(te_records, p_cfg, t_cfg.batch_size, False, rate_csv=args.rate_csv)

        model = FrequencyAwareMultiTaskNet(m_cfg).to(device)

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

        print("Pred classes:", np.unique(tex_pred))

        # ==========================================
        # METRICS
        # ==========================================
        tmet = texture_metrics(tex_true, tex_pred)
        tmet["fold"] = sid
        fold_tex.append(tmet)

        acc = tmet.get("accuracy", 0)
        subject_acc.append(acc)

        cm = texture_confusion(tex_true, tex_pred)
        cm_sum += cm

        all_preds.extend(tex_pred)
        all_labels.extend(tex_true)

    # ==========================================
    # RESULTS
    # ==========================================
    print("\n========== FINAL RESULTS ==========")

    tex_df, _ = summarize_fold_metrics(fold_tex, out_dir, "texture_results")

    # ==========================================
    # 🔥 CONFUSION MATRIX
    # ==========================================
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    # ==========================================
    # 🔥 LOSO BAR PLOT
    # ==========================================
    plt.figure()
    plt.bar(range(len(subject_acc)), subject_acc)
    plt.xlabel("Subject")
    plt.ylabel("Accuracy")
    plt.title("LOSO Accuracy per Subject")
    plt.savefig(out_dir / "loso_accuracy.png")
    plt.close()

    # ==========================================
    # 🔥 SAVE METRICS CSV
    # ==========================================
    df = pd.DataFrame({
        "subject": list(range(1, len(subject_acc)+1)),
        "accuracy": subject_acc
    })
    df.to_csv(out_dir / "loso_metrics.csv", index=False)

    # ==========================================
    # PRINT SUMMARY
    # ==========================================
    if len(subject_acc) > 0:
        print("Per-subject accuracy:", subject_acc)
        print(f"Mean LOSO Accuracy: {np.mean(subject_acc):.4f}")

    print(f"\nDone. Outputs saved to: {out_dir}")


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--recorded_dir", type=str, required=True)
    ap.add_argument("--rate_csv", type=str, default="")
    ap.add_argument("--output_dir", type=str, default="outputs")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    run(ap.parse_args())
