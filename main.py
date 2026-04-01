"""FINAL CONSOLIDATED MAIN: LOSO + Curves + Attention + Ablation (CLEAN VERSION)"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_loader import load_recorded_records, loso_splits, make_loader
from evaluation import summarize_fold_metrics
from metrics import texture_metrics
from model import FrequencyAwareMultiTaskNet, ModelConfig
from preprocessing import PreprocessConfig
from training import TrainConfig, train_model
from utils import device_auto, ensure_dir, set_seed


# =========================================================
# SAFE MERGE
# =========================================================
def safe_merge(outputs):
    merged = {}
    for k in outputs[0]:
        tensors = [o[k] for o in outputs if o[k] is not None]
        if tensors:
            merged[k] = torch.cat(tensors, dim=0)
    return merged


# =========================================================
# ATTENTION PLOT
# =========================================================
def plot_attention(attn, save_path):
    attn = np.array(attn)
    # 🔥 FIX SHAPE
    if attn.ndim == 1:
        attn = attn.reshape(1, -1)
    plt.figure(figsize=(10, 3))
    plt.imshow(attn, aspect='auto', cmap='viridis')
    plt.title("Attention Map")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# def plot_attention(attn, save_path):
#     plt.figure()
#     plt.imshow(attn, aspect='auto')
#     plt.colorbar()
#     plt.title("Attention Map")
#     plt.xlabel("Time")
#     plt.ylabel("Weight")
#     plt.savefig(save_path)
#     plt.close()



# =========================================================
# RUN SINGLE CONFIG (CORE EXPERIMENT)
# =========================================================
def run_experiment(args, m_cfg, tag):

    device = device_auto()
    out_dir = ensure_dir(Path(args.output_dir) / tag)

    p_cfg = PreprocessConfig()
    t_cfg = TrainConfig(batch_size=args.batch_size, epochs=args.epochs)

    print(f"\n📂 Loading data for [{tag}]...")
    records = load_recorded_records(args.recorded_dir)
    print("Total samples:", len(records))

    # ==========================================
    # COLLECTION
    # ==========================================
    all_preds, all_labels = [], []
    subject_acc = []

    all_train_loss, all_val_loss, all_val_acc = [], [], []

    fold_tex = []

    # ==========================================
    # LOSO LOOP
    # ==========================================
    for tr, te, sid in loso_splits(records):

        print(f"\n[{tag}] LOSO subject={sid}")

        tr_loader = make_loader(tr, p_cfg, t_cfg.batch_size, True, rate_csv=args.rate_csv)
        te_loader = make_loader(te, p_cfg, t_cfg.batch_size, False, rate_csv=args.rate_csv)

        model = FrequencyAwareMultiTaskNet(m_cfg).to(device)

        # TRAIN
        model, history = train_model(model, tr_loader, te_loader, device, t_cfg)

        # STORE CURVES
        all_train_loss.extend(history["train_loss"])
        all_val_loss.extend(history["val_loss"])
        all_val_acc.extend(history["val_acc"])

        # ==========================================
        # EVALUATION
        # ==========================================
        outputs = []

        model.eval()
        with torch.no_grad():
            for b in te_loader:
                x = b["x"].to(device)
                pred = model(x)

                outputs.append({
                    "tex_logits": pred["tex_logits"].cpu(),
                    "tex_y": b["tex_y"],
                    "attn": pred["attn"].cpu()
                })

        if len(outputs) == 0:
            print(f"⚠️ Skipping subject {sid}")
            continue

        merged = safe_merge(outputs)

        pred = merged["tex_logits"].argmax(dim=1).numpy()
        true = merged["tex_y"].numpy()

        # METRICS
        tmet = texture_metrics(true, pred)
        tmet["fold"] = sid
        fold_tex.append(tmet)

        acc = tmet.get("accuracy", 0)
        subject_acc.append(acc)

        all_preds.extend(pred)
        all_labels.extend(true)

        # ATTENTION (1 sample per subject)
        plot_attention(merged["attn"][0], out_dir / f"attn_subject_{sid}.png")

    # ==========================================
    # SAVE METRICS
    # ==========================================
    summarize_fold_metrics(fold_tex, out_dir, "texture_results")

    # ==========================================
    # PLOTS
    # ==========================================

    # LOSS CURVE
    plt.figure()
    plt.plot(all_train_loss, label="Train")
    plt.plot(all_val_loss, label="Val")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(out_dir / "loss.png")
    plt.close()

    # ACCURACY CURVE
    plt.figure()
    plt.plot(all_val_acc)
    plt.title("Accuracy Curve")
    plt.savefig(out_dir / "accuracy.png")
    plt.close()

    # CONFUSION MATRIX
    cm = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig(out_dir / "confusion.png")
    plt.close()

    # LOSO BAR
    plt.figure()
    plt.bar(range(len(subject_acc)), subject_acc)
    plt.xlabel("Subject")
    plt.ylabel("Accuracy")
    plt.title("LOSO Accuracy")
    plt.savefig(out_dir / "loso.png")
    plt.close()

    # CSV
    pd.DataFrame({
        "subject": list(range(1, len(subject_acc)+1)),
        "accuracy": subject_acc
    }).to_csv(out_dir / "loso_metrics.csv", index=False)

    return np.mean(subject_acc)


# =========================================================
# MAIN (ABLATION DRIVER)
# =========================================================
def run(args):

    set_seed(args.seed)

    configs = [
        ("baseline", ModelConfig()),
        ("no_attention", ModelConfig(use_attention=False)),
        ("no_B1", ModelConfig(remove_b1=True)),
        ("no_B3", ModelConfig(remove_b3=True)),
    ]

    results = []

    for name, cfg in configs:
        acc = run_experiment(args, cfg, name)
        results.append({"config": name, "accuracy": acc})

    df = pd.DataFrame(results)
    df.to_csv(Path(args.output_dir) / "ablation.csv", index=False)

    print("\n===== ABLATION RESULTS =====")
    print(df)


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--recorded_dir", required=True)
    ap.add_argument("--rate_csv", default="")
    ap.add_argument("--output_dir", default="outputs")

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)

    run(ap.parse_args())
