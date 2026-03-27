"""Signal preprocessing pipeline for chewing audio."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import librosa
import numpy as np
from scipy.signal import butter, lfilter, stft


@dataclass
class PreprocessConfig:
    sr: int = 44_000
    frame_ms: float = 50.0
    overlap: float = 0.10
    n_mels: int = 64
    min_freq: int = 20
    max_freq: int = 4_500


# ==========================================
# BANDPASS
# ==========================================
def butter_bandpass(low: float, high: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return b, a


def apply_bandpass(sig: np.ndarray, low: float, high: float, fs: int, order: int = 4) -> np.ndarray:
    b, a = butter_bandpass(low, high, fs, order=order)

    filtered = lfilter(b, a, sig)

    # ✅ FIX: fallback if signal collapses
    if np.abs(filtered).mean() < 1e-5:
        return sig

    return filtered


# ==========================================
# SILENCE REMOVAL
# ==========================================
def adaptive_silence_removal(sig: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    frame_len = int(cfg.frame_ms * cfg.sr / 1000)
    hop = int(frame_len * (1 - cfg.overlap))

    energies = []
    starts = []

    for i in range(0, max(1, len(sig) - frame_len + 1), max(1, hop)):
        frame = sig[i : i + frame_len]
        energies.append(np.mean(frame**2) + 1e-10)
        starts.append(i)

    energies = np.asarray(energies)

    # ❌ OLD (too aggressive)
    # thresh = 0.01 * np.mean(energies)

    # ✅ NEW (safer)
    thresh = 0.005 * np.mean(energies)

    keep = np.zeros_like(sig, dtype=bool)

    for e, i in zip(energies, starts):
        if e >= thresh:
            keep[i : i + frame_len] = True

    if not keep.any():
        return sig

    return sig[keep]


# ==========================================
# STFT + MEL
# ==========================================
def stft_logmel(sig: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:

    frame_len = int(cfg.frame_ms * cfg.sr / 1000)
    hop = int(frame_len * (1 - cfg.overlap))
    window = np.bartlett(frame_len)

    _, _, zxx = stft(sig, fs=cfg.sr, window=window,
                     nperseg=frame_len,
                     noverlap=frame_len - hop)

    mag = np.abs(zxx) + 1e-8

    mel_basis = librosa.filters.mel(
        sr=cfg.sr,
        n_fft=frame_len,
        n_mels=cfg.n_mels,
        fmin=cfg.min_freq,
        fmax=cfg.max_freq,
    )

    mel = mel_basis @ mag
    mel = np.maximum(mel, 1e-8)

    # ✅ FIX: log stability
    mel = np.log(mel + 1e-6)

    # ✅ ADD: energy boost
    energy = np.mean(mel, axis=0, keepdims=True)
    mel = mel + 0.1 * energy

    return mel


# ==========================================
# NORMALIZATION
# ==========================================
def zscore_global(x: np.ndarray, mean=None, std=None):

    if mean is None:
        mean = float(x.mean())

    if std is None:
        std = float(x.std())

    # ✅ FIX: avoid collapse
    if std < 1e-6:
        std = 1.0

    return (x - mean) / std, mean, std


# ==========================================
# SUBBANDS
# ==========================================
def compute_subbands(sig: np.ndarray, fs: int) -> Dict[str, np.ndarray]:
    bands = {
        "B1": (20, 200),
        "B2": (200, 800),
        "B3": (800, 2000),
        "B4": (2000, 4500),
    }
    return {k: apply_bandpass(sig, lo, hi, fs, order=4) for k, (lo, hi) in bands.items()}


# ==========================================
# MAIN PIPELINE
# ==========================================
def preprocess_audio(path: str, cfg: PreprocessConfig) -> Tuple[Dict[str, np.ndarray], np.ndarray]:

    sig, _ = librosa.load(path, sr=cfg.sr, mono=True)

    # -------------------------
    # Data augmentation
    # -------------------------
    if np.random.rand() < 0.5:
        sig = sig + 0.003 * np.random.randn(len(sig))

    if np.random.rand() < 0.5:
        sig = sig * np.random.uniform(0.8, 1.2)

    # -------------------------
    # Processing
    # -------------------------
    sig = adaptive_silence_removal(sig, cfg)

    subbands = compute_subbands(sig, cfg.sr)

    feats = {k: stft_logmel(v, cfg) for k, v in subbands.items()}

    # ❌ OLD (BROKEN → caused zero features)
    # stacked = np.concatenate(list(feats.values()), axis=0)
    # normed, mean, std = zscore_global(stacked)
    # splits = np.array_split(normed, 4, axis=0)
    # out = {k: s for k, s in zip(["B1", "B2", "B3", "B4"], splits)}

    # ✅ NEW (CORRECT → per-band normalization)
    out = {}
    for k, v in feats.items():
        v, _, _ = zscore_global(v)
        out[k] = np.nan_to_num(v)

    return out, sig
