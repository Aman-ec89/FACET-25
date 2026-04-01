"""FINAL MAIN (CLEAN + ACCURACY IMPROVED)"""

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from data_loader import make_loader
from model import FrequencyAwareMultiTaskNet, ModelConfig
from training import TrainConfig, train_model


# ==========================================
# ATTENTION FIX
# ==========================================
def plot_attention(attn, save_path):
    attn = np.array(attn)
    if attn.ndim == 1:
        attn = attn.reshape(1, -1)

    plt.imshow(attn, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()


# ==========================================
# MAIN
# ==========================================
def run(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = list(Path(args.recorded_dir).glob("*.npy"))

    print("Total samples:", len(files))

    split = int(0.8 * len(files))
    train_files = files[:split]
    val_files = files[split:]

    train_loader = make_loader(train_files, args.batch_size, True)
    val_loader = make_loader(val_files, args.batch_size, False)

    model = FrequencyAwareMultiTaskNet(ModelConfig()).to(device)

    cfg = TrainConfig()

    model, history = train_model(model, train_loader, val_loader, device, cfg)

    # ==========================================
    # PLOTS
    # ==========================================
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.legend()
    plt.savefig("loss.png")
    plt.close()

    plt.plot(history["val_acc"])
    plt.savefig("acc.png")
    plt.close()

    print("Training complete")


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--recorded_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=64)

    run(ap.parse_args())
