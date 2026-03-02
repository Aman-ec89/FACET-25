"""Evaluation and figure/table generation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.metrics import RocCurveDisplay, roc_curve

from metrics import chewing_rate_metrics, detection_metrics, texture_confusion, texture_metrics
from utils import confidence_interval, ensure_dir, to_latex_table


def summarize_fold_metrics(rows: List[Dict], out_dir: str | Path, name: str):
    out = ensure_dir(out_dir)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{name}.csv", index=False)
    summary = []
    for c in df.columns:
        if c == "fold":
            continue
        vals = df[c].astype(float).values
        lo, hi = confidence_interval(vals)
        summary.append({"metric": c, "mean": vals.mean(), "std": vals.std(ddof=1), "ci_low": lo, "ci_high": hi})
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out / f"{name}_summary.csv", index=False)
    (out / f"{name}.tex").write_text(to_latex_table(sdf, caption=name, label=f"tab:{name}"))
    return df, sdf


def make_roc_plot(y_true, y_prob, path: str | Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_confusion_plot(cm: np.ndarray, path: str | Path):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_rate_scatter(gt, pred, path: str | Path):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.scatter(gt, pred, alpha=0.6)
    lo, hi = min(gt + pred), max(gt + pred)
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("GT rate (bpm)")
    ax.set_ylabel("Predicted rate (bpm)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_ablation_plot(df: pd.DataFrame, path: str | Path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.barplot(data=df, x="variant", y="macro_f1", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def compute_wilcoxon(base: np.ndarray, variant: np.ndarray) -> Dict[str, float]:
    stat, p = wilcoxon(base, variant)
    return {"stat": float(stat), "p": float(p)}


def make_attention_plot(attn_weights: np.ndarray, path: str | Path):
    fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
    sns.heatmap(attn_weights, cmap="viridis", ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Sample")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_psd_subband_plot(sig: np.ndarray, fs: int, path: str | Path):
    from scipy.signal import welch

    f, pxx = welch(sig, fs=fs, nperseg=min(4096, len(sig)))
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.semilogy(f, pxx)
    for lo, hi, c in [(20, 200, "#d73027"), (200, 800, "#fc8d59"), (800, 2000, "#91bfdb"), (2000, 4500, "#4575b4")]:
        ax.axvspan(lo, hi, alpha=0.15, color=c)
    ax.set_xlim(0, 5000)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
