"""Signal preprocessing pipeline for chewing audio (GPU ENABLED + FIXED)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import librosa
import numpy as np
from scipy.signal import butter, lfilter

# ==========================================
# ✅ GPU SUPPORT
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
# BANDPASS FILTER (CPU – OK)
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
        frame = sig[i : i + frame_len]
        energies.append(np.mean(frame**2))
        starts.append(i)

    energies = np.asarray(energies)
    thresh = 0.01 * np.mean(energies)

    keep = np.zeros_like(sig, dtype=bool)

    for e, i in zip(energies, starts):
        if e >= thresh:
            keep[i : i + frame_len] = True

    return sig if not keep.any() else sig[keep]


# ==========================================
# ❌ OLD CPU STFT (KEPT FOR REFERENCE)
# ==========================================
# def stft_logmel(sig, cfg):
#     ...


# ==========================================
# ✅ GPU MEL-SPECTROGRAM
# ==========================================
def stft_logmel_gpu(sig: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:

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

    # ❌ OLD (unstable)
    # spec = torch.log(spec + 1e-8)

    # ✅ FIXED
    spec = torch.log1p(spec)

    return spec.cpu().numpy()


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
    return {k: apply_bandpass(sig, lo, hi, fs) for k, (lo, hi) in bands.items()}


# ==========================================
# MAIN PIPELINE
# ==========================================
def preprocess_audio(path: str, cfg: PreprocessConfig) -> Tuple[Dict[str, np.ndarray], np.ndarray]:

    sig, _ = librosa.load(path, sr=cfg.sr, mono=True)

    # ==========================================
    # DATA AUGMENTATION
    # ==========================================
    if np.random.rand() < 0.5:
        sig = sig + 0.003 * np.random.randn(len(sig))

    if np.random.rand() < 0.5:
        sig = sig * np.random.uniform(0.8, 1.2)

    # ==========================================
    # SILENCE REMOVAL
    # ==========================================
    sig = adaptive_silence_removal(sig, cfg)

    # ==========================================
    # SUBBANDS
    # ==========================================
    subbands = compute_subbands(sig, cfg.sr)

    # ==========================================
    # FEATURE EXTRACTION (GPU)
    # ==========================================
    feats = {k: stft_logmel_gpu(v, cfg) for k, v in subbands.items()}

    # ==========================================
    # ❌ OLD GLOBAL NORMALIZATION (BROKEN)
    # ==========================================
    # stacked = np.concatenate(list(feats.values()), axis=0)
    # normed = (stacked - stacked.mean()) / (stacked.std() + 1e-8)
    # splits = np.array_split(normed, 4, axis=0)
    # out = {k: s for k, s in zip(["B1","B2","B3","B4"], splits)}

    # ==========================================
    # ✅ NEW PER-BAND NORMALIZATION (FIX)
    # ==========================================
    out = {}

    for k in ["B1", "B2", "B3", "B4"]:
        x = feats[k]

        mean = x.mean()
        std = x.std() + 1e-8

        x = (x - mean) / std

        # ❌ OLD
        # x = np.nan_to_num(x)

        # ✅ SAFE VERSION
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        out[k] = x

    return out, sig"""Signal preprocessing pipeline for chewing audio (GPU ENABLED + FIXED)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import librosa
import numpy as np
from scipy.signal import butter, lfilter

# ==========================================
# ✅ GPU SUPPORT
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
# BANDPASS FILTER (CPU – OK)
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
        frame = sig[i : i + frame_len]
        energies.append(np.mean(frame**2))
        starts.append(i)

    energies = np.asarray(energies)
    thresh = 0.01 * np.mean(energies)

    keep = np.zeros_like(sig, dtype=bool)

    for e, i in zip(energies, starts):
        if e >= thresh:
            keep[i : i + frame_len] = True

    return sig if not keep.any() else sig[keep]


# ==========================================
# ❌ OLD CPU STFT (KEPT FOR REFERENCE)
# ==========================================
# def stft_logmel(sig, cfg):
#     ...


# ==========================================
# ✅ GPU MEL-SPECTROGRAM
# ==========================================
def stft_logmel_gpu(sig: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:

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

    # ❌ OLD (unstable)
    # spec = torch.log(spec + 1e-8)

    # ✅ FIXED
    spec = torch.log1p(spec)

    return spec.cpu().numpy()


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
    return {k: apply_bandpass(sig, lo, hi, fs) for k, (lo, hi) in bands.items()}


# ==========================================
# MAIN PIPELINE
# ==========================================
def preprocess_audio(path: str, cfg: PreprocessConfig) -> Tuple[Dict[str, np.ndarray], np.ndarray]:

    sig, _ = librosa.load(path, sr=cfg.sr, mono=True)

    # ==========================================
    # DATA AUGMENTATION
    # ==========================================
    if np.random.rand() < 0.5:
        sig = sig + 0.003 * np.random.randn(len(sig))

    if np.random.rand() < 0.5:
        sig = sig * np.random.uniform(0.8, 1.2)

    # ==========================================
    # SILENCE REMOVAL
    # ==========================================
    sig = adaptive_silence_removal(sig, cfg)

    # ==========================================
    # SUBBANDS
    # ==========================================
    subbands = compute_subbands(sig, cfg.sr)

    # ==========================================
    # FEATURE EXTRACTION (GPU)
    # ==========================================
    feats = {k: stft_logmel_gpu(v, cfg) for k, v in subbands.items()}

    # ==========================================
    # ❌ OLD GLOBAL NORMALIZATION (BROKEN)
    # ==========================================
    # stacked = np.concatenate(list(feats.values()), axis=0)
    # normed = (stacked - stacked.mean()) / (stacked.std() + 1e-8)
    # splits = np.array_split(normed, 4, axis=0)
    # out = {k: s for k, s in zip(["B1","B2","B3","B4"], splits)}

    # ==========================================
    # ✅ NEW PER-BAND NORMALIZATION (FIX)
    # ==========================================
    out = {}

    for k in ["B1", "B2", "B3", "B4"]:
        x = feats[k]

        mean = x.mean()
        std = x.std() + 1e-8

        x = (x - mean) / std

        # ❌ OLD
        # x = np.nan_to_num(x)

        # ✅ SAFE VERSION
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        out[k] = x

    return out, sig"""Signal preprocessing pipeline for chewing audio (GPU ENABLED)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import librosa
import numpy as np
from scipy.signal import butter, lfilter

# ==========================================
# ✅ GPU SUPPORT
# ==========================================
import torch
import torchaudio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PreprocessConfig:
    sr: int = 44_000
    frame_ms: float = 50.0
    overlap: float = 0.10
    n_mels: int = 64
    min_freq: int = 20
    max_freq: int = 4_500


# ==========================================
# BANDPASS FILTER (CPU – OK)
# ==========================================
def butter_bandpass(low: float, high: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return b, a


def apply_bandpass(sig: np.ndarray, low: float, high: float, fs: int, order: int = 4) -> np.ndarray:
    b, a = butter_bandpass(low, high, fs, order=order)
    return lfilter(b, a, sig)


# ==========================================
# SILENCE REMOVAL (CPU – OK)
# ==========================================
def adaptive_silence_removal(sig: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    frame_len = int(cfg.frame_ms * cfg.sr / 1000)
    hop = int(frame_len * (1 - cfg.overlap))

    energies = []
    starts = []

    for i in range(0, max(1, len(sig) - frame_len + 1), max(1, hop)):
        frame = sig[i : i + frame_len]
        energies.append(np.mean(frame**2))
        starts.append(i)

    energies = np.asarray(energies)
    thresh = 0.01 * np.mean(energies)

    keep = np.zeros_like(sig, dtype=bool)

    for e, i in zip(energies, starts):
        if e >= thresh:
            keep[i : i + frame_len] = True

    return sig if not keep.any() else sig[keep]


# ==========================================
# ❌ OLD CPU VERSION (KEPT)
# ==========================================
# def stft_logmel(sig: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
#     ...


# ==========================================
# ✅ GPU VERSION (TORCH)
# ==========================================
def stft_logmel_gpu(sig: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:

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
    spec = torch.log(spec + 1e-8)

    return spec.cpu().numpy()


# ==========================================
# NORMALIZATION
# ==========================================
def zscore_global(x: np.ndarray):
    mean = x.mean()
    std = x.std() + 1e-8
    return (x - mean) / std


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
    return {k: apply_bandpass(sig, lo, hi, fs) for k, (lo, hi) in bands.items()}


# ==========================================
# MAIN PIPELINE
# ==========================================
def preprocess_audio(path: str, cfg: PreprocessConfig) -> Tuple[Dict[str, np.ndarray], np.ndarray]:

    sig, _ = librosa.load(path, sr=cfg.sr, mono=True)

    # ==========================================
    # DATA AUGMENTATION
    # ==========================================
    if np.random.rand() < 0.5:
        sig = sig + 0.003 * np.random.randn(len(sig))

    if np.random.rand() < 0.5:
        sig = sig * np.random.uniform(0.8, 1.2)

    # ==========================================
    sig = adaptive_silence_removal(sig, cfg)

    # ==========================================
    # SUBBANDS
    # ==========================================
    subbands = compute_subbands(sig, cfg.sr)

    # ==========================================
    # ✅ GPU FEATURE EXTRACTION
    # ==========================================
    feats = {k: stft_logmel_gpu(v, cfg) for k, v in subbands.items()}

    # ==========================================
    # NORMALIZATION
    # ==========================================
    stacked = np.concatenate(list(feats.values()), axis=0)
    normed = zscore_global(stacked)

    splits = np.array_split(normed, 4, axis=0)

    out = {k: s for k, s in zip(["B1", "B2", "B3", "B4"], splits)}

    for k in out:
        out[k] = np.nan_to_num(out[k])

    return out, sig
