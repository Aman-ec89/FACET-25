"""Utility helpers for reproducible audio ML experiments."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


TEXTURE_TO_ID: Dict[str, int] = {
    "fibrous": 0,
    "crunchy": 1,
    "soft": 2,
    "brittle": 3,
}
ID_TO_TEXTURE = {v: k for k, v in TEXTURE_TO_ID.items()}
KAGGLE_TO_TEXTURE: Dict[str, str] = {
    "cabbage": "fibrous",
    "carrot": "crunchy",
    "noodles": "soft",
    "chocolate": "brittle",
}


@dataclass
class AudioRecord:
    path: Path
    texture: str
    texture_id: int
    subject_id: int | None = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def confidence_interval(values: Iterable[float], z: float = 1.96) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    mu = arr.mean()
    std = arr.std(ddof=1) if arr.size > 1 else 0.0
    margin = z * (std / np.sqrt(max(arr.size, 1)))
    return float(mu - margin), float(mu + margin)


def parse_recorded_filename(path: Path) -> AudioRecord:
    # sub01_crunchy_05.wav
    stem = path.stem
    subject, texture, _idx = stem.split("_")
    sid = int(subject.replace("sub", ""))
    return AudioRecord(path=path, texture=texture, texture_id=TEXTURE_TO_ID[texture], subject_id=sid)


def parse_kaggle_filename(path: Path) -> AudioRecord:
    # cabbage_2_01.wav
    prefix = path.stem.split("_")[0].lower()
    texture = KAGGLE_TO_TEXTURE[prefix]
    return AudioRecord(path=path, texture=texture, texture_id=TEXTURE_TO_ID[texture])


def to_latex_table(df, caption: str, label: str) -> str:
    return df.to_latex(index=False, float_format=lambda x: f"{x:.4f}", caption=caption, label=label)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
