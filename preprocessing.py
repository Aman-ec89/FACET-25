"""Signal preprocessing pipeline for chewing audio (FINAL STABLE VERSION)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import librosa
import numpy as np
from scipy.signal import butter, lfilter

# ==========================================
# GPU SUPPORT
# ==========================================
import torch
import torchaudio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# CONFIG
# ==========================================
@dataclass
class PreprocessConfig:
    sr: int = 44_000
    frame_ms: float = 50.0
    overlap: float = 0.10
    n_mels: int = 64
    min_freq: int = 20
    max_freq: int = 4_500


# ==========================================
# BANDPASS FILTER
# ==========================================
def butter_bandpass(low: float, high: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return b, a


def apply_bandpass(sig: np.ndarray, low: float, high: float, fs: int, order: int = 4) -> np.ndarray:
    b, a = butter_bandpass(low, high, fs, order=order)
    return lfilter(b, a, sig)


# ==========================================
# SILENCE REMOVAL
# ==========================================
def adaptive_silence_removal(sig: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    frame_len = int(cfg.frame_ms * cfg.sr / 1000)
    hop = int(frame_len * (1 - cfg.overlap))

    energies = []
    starts = []

    for i in range(0, max(1, len(sig) - frame_len + 1), max(1, hop)):
        frame = sig[i:i + frame_len]
        energies.append(np.mean(frame**2))
        starts.append(i)

    energies = np.asarray(energies)
    thresh = 0.01 * np.mean(energies)

    keep = np.zeros_like(sig, dtype=bool)

    for e, i in zip(energies, starts):
        if e >= thresh:
            keep[i:i + frame_len] = True

    return sig if not keep.any() else sig[keep]


# ==========================================
# GPU MEL-SPECTROGRAM
# ==========================================
def stft_logmel_gpu(sig: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:

    sig = np.nan_to_num(sig)  # 🔥 critical safety

    sig_tensor = torch.tensor(sig, dtype=torch.float32).to(DEVICE)

    n_fft = int(cfg.frame_ms * cfg.sr / 1000)
    hop = int(n_fft * (1 - cfg.overlap))

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=cfg.n_mels,
        f_min=cfg.min_freq,
        f_max=cfg.max_freq,
        power=2.0,
    ).to(DEVICE)

    spec = mel_spec(sig_tensor)

    # stable log
    spec = torch.log1p(spec)

    return spec.cpu().numpy()


# ==========================================
# FEATURE EXTRACTION (ENHANCED)
# ==========================================
def extract_features(sig: np.ndarray, cfg: PreprocessConfig):

    mel = stft_logmel_gpu(sig, cfg)  # (64, T)

    mel = np.nan_to_num(mel)

    # DELTA FEATURES
    delta = librosa.feature.delta(mel)
    delta2 = librosa.feature.delta(mel, order=2)

    # ENERGY
    energy = np.mean(mel**2, axis=0, keepdims=True)
    energy = np.repeat(energy, mel.shape[0], axis=0)

    # SPECTRAL CENTROID
    centroid = librosa.feature.spectral_centroid(S=mel)
    centroid = np.repeat(centroid, mel.shape[0], axis=0)

    # STACK FEATURES → (5, 64, T)
    out = np.stack([mel, delta, delta2, energy, centroid], axis=0)

    # GLOBAL NORMALIZATION
    mean = out.mean()
    std = out.std() + 1e-8
    out = (out - mean) / std

    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

    return out


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

    subbands = {}
    for k, (lo, hi) in bands.items():
        x = apply_bandpass(sig, lo, hi, fs)
        x = np.nan_to_num(x)  # 🔥 critical fix
        subbands[k] = x

    return subbands


# ==========================================
# MAIN PIPELINE
# ==========================================
def preprocess_audio(path: str, cfg: PreprocessConfig) -> Tuple[Dict[str, np.ndarray], np.ndarray]:

    sig, _ = librosa.load(path, sr=cfg.sr, mono=True)

    # ==========================================
    # 🔥 CRITICAL CLEANING
    # ==========================================
    sig = np.nan_to_num(sig)

    max_val = np.max(np.abs(sig)) + 1e-8
    sig = sig / max_val

    # ==========================================
    # AUGMENTATION
    # ==========================================
    if np.random.rand() < 0.5:
        sig = sig + 0.003 * np.random.randn(len(sig))

    if np.random.rand() < 0.5:
        sig = sig * np.random.uniform(0.8, 1.2)

    sig = np.nan_to_num(sig)

    # ==========================================
    # SILENCE REMOVAL
    # ==========================================
    sig = adaptive_silence_removal(sig, cfg)

    # ==========================================
    # SUBBANDS
    # ==========================================
    subbands = compute_subbands(sig, cfg.sr)

    # ==========================================
    # FEATURE EXTRACTION
    # ==========================================
    feats = {k: extract_features(v, cfg) for k, v in subbands.items()}

    return feats, sig
