"""Utility helpers for reproducible audio ML experiments."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


# ---------------------------------------------------------
# ✅ CORRECT TEXTURE MAPPING (MATCHES YOUR DATASET)
# ---------------------------------------------------------
TEXTURE_TO_ID: Dict[str, int] = {
    "brittle": 0,
    "crunchy": 1,
    "fibrous": 2,
    "soft": 3,
}

ID_TO_TEXTURE = {v: k for k, v in TEXTURE_TO_ID.items()}


# ---------------------------------------------------------
# ✅ KAGGLE → RECORDED ALIGNMENT
# ---------------------------------------------------------
KAGGLE_TO_TEXTURE: Dict[str, str] = {
    "cabbage": "fibrous",
    "carrots": "crunchy",
    "noodles": "soft",
    "chocolate": "brittle",
}


# ---------------------------------------------------------
# DATA CLASS
# ---------------------------------------------------------
@dataclass
class AudioRecord:
    path: Path
    texture: str
    texture_id: int
    subject_id: int | None = None


# ---------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# FILE HELPERS
# ---------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------
# STATS
# ---------------------------------------------------------
def confidence_interval(values: Iterable[float], z: float = 1.96) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)

    if arr.size == 0:
        return np.nan, np.nan

    mu = arr.mean()
    std = arr.std(ddof=1) if arr.size > 1 else 0.0
    margin = z * (std / np.sqrt(max(arr.size, 1)))

    return float(mu - margin), float(mu + margin)


# ---------------------------------------------------------
# ✅ FIXED RECORDED PARSER (CRITICAL FIX)
# ---------------------------------------------------------
def parse_recorded_filename(path: Path) -> AudioRecord:
    """
    Expected format:
    sub01_brittle_01.wav
    sub01_crunchy_02.wav
    """

    stem = path.stem.lower()
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Invalid filename format: {path.name}")

    # subject
    subject = parts[0]
    if not subject.startswith("sub"):
        raise ValueError(f"Invalid subject format: {path.name}")

    sid = int(subject.replace("sub", ""))

    # texture
    texture = parts[1]

    if texture not in TEXTURE_TO_ID:
        raise ValueError(f"Unknown texture '{texture}' in file {path.name}")

    texture_id = TEXTURE_TO_ID[texture]

    return AudioRecord(
        path=path,
        texture=texture,
        texture_id=texture_id,
        subject_id=sid,
    )


# ---------------------------------------------------------
# ✅ FIXED KAGGLE PARSER
# ---------------------------------------------------------
def parse_kaggle_filename(path: Path) -> AudioRecord:
    """
    Expected format:
    cabbage_2_01.wav
    """

    prefix = path.stem.split("_")[0].lower()

    if prefix not in KAGGLE_TO_TEXTURE:
        raise KeyError(f"Unknown Kaggle class: {prefix}")

    texture = KAGGLE_TO_TEXTURE[prefix]

    if texture not in TEXTURE_TO_ID:
        raise ValueError(f"Texture mapping missing for: {texture}")

    return AudioRecord(
        path=path,
        texture=texture,
        texture_id=TEXTURE_TO_ID[texture],
    )


# ---------------------------------------------------------
# LATEX EXPORT
# ---------------------------------------------------------
def to_latex_table(df, caption: str, label: str) -> str:
    return df.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4f}",
        caption=caption,
        label=label,
    )


# ---------------------------------------------------------
# MODEL UTILS
# ---------------------------------------------------------
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
