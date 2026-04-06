"""FINAL MAIN (LOSO + ABLATION + CSV EXPORT)"""

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

from data_loader import make_loader
from model import FrequencyAwareMultiTaskNet, ModelConfig
from training import TrainConfig, train_model


# ==========================================
# MAIN
# ==========================================
def run(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = list(Path(args.recorded_dir).glob("*.npy"))
    print("Total samples:", len(files))

    # ==========================================
    # SUBJECT LIST (LOSO)
    # ==========================================
    subjects = sorted(set([f.stem.split("_")[0] for f in files]))

    os.makedirs("outputs", exist_ok=True)

    results = {}

    # ==========================================
    # ABLATION MODES
    # ==========================================
    for mode in ["baseline", "no_attention", "freq_attention"]:

        print(f"\n📂 Running mode: {mode}")
        results[mode] = []

        subject_records = []

        for subject in subjects:

            print(f"\n[LOSO] Subject: {subject}")

            train_files = [f for f in files if subject not in f.stem]
            val_files = [f for f in files if subject in f.stem]

            train_loader = make_loader(train_files, args.batch_size, True)
            val_loader = make_loader(val_files, args.batch_size, False)

            # ==========================================
            # MODEL CONFIG
            # ==========================================
            cfg_model = ModelConfig()

            if mode == "no_attention":
                cfg_model.use_attention = False

            model = FrequencyAwareMultiTaskNet(cfg_model).to(device)

            # ==========================================
            # TRAIN CONFIG
            # ==========================================
            cfg_train = TrainConfig()
            cfg_train.epochs = args.epochs

            model, history = train_model(
                model, train_loader, val_loader, device, cfg_train
            )

            # ==========================================
            # BEST ACC
            # ==========================================
            best_acc = max(history["val_acc"])
            results[mode].append(best_acc)

            subject_records.append({
                "subject": subject,
                "best_acc": best_acc
            })

            print(f"Subject {subject} best acc: {best_acc:.3f}")

        # ==========================================
        # SAVE SUBJECT CSV
        # ==========================================
        df = pd.DataFrame(subject_records)
        df.to_csv(f"outputs/{mode}_loso_results.csv", index=False)

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    summary = []

    print("\n========== FINAL RESULTS ==========")

    for mode in results:
        accs = results[mode]

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)

        print(f"{mode}: mean={mean_acc:.3f}, std={std_acc:.3f}")

        summary.append({
            "mode": mode,
            "mean_acc": mean_acc,
            "std_acc": std_acc
        })

    pd.DataFrame(summary).to_csv("outputs/final_summary.csv", index=False)

    print("\nSaved all results in outputs/")


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--recorded_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)

    run(ap.parse_args())
