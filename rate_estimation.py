"""Chewing rate estimation from waveform peaks."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, lfilter


def sliding_rms(sig: np.ndarray, fs: int, frame_ms: float = 50.0) -> np.ndarray:
    n = max(1, int(frame_ms * fs / 1000.0))
    window = np.ones(n) / n
    return np.sqrt(np.convolve(sig**2, window, mode="same"))


def lowpass(sig: np.ndarray, fs: int, cutoff_hz: float = 5.0, order: int = 4) -> np.ndarray:
    b, a = butter(order, cutoff_hz / (0.5 * fs), btype="low")
    return lfilter(b, a, sig)


def estimate_chewing_rate_bpm(sig: np.ndarray, fs: int, min_peak_distance_s: float = 0.35) -> float:
    rms = sliding_rms(sig, fs=fs)
    smooth = lowpass(rms, fs=fs)
    peaks, _ = find_peaks(smooth, distance=max(1, int(min_peak_distance_s * fs)))
    dur = max(len(sig) / fs, 1e-8)
    return (len(peaks) / dur) * 60.0
