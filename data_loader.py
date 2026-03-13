"""Dataset and split utilities for Mandible/Kaggle audio corpora."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from preprocessing import PreprocessConfig
from utils import AudioRecord, parse_kaggle_filename, parse_recorded_filename


class ChewingDataset(Dataset):

    def __init__(
        self,
        records: Sequence[AudioRecord],
        cfg: PreprocessConfig,
        rate_csv: str | None = None,
    ):
        self.records = list(records)
        self.cfg = cfg

        self.rate_map = {}
        if rate_csv:
            df = pd.read_csv(rate_csv)
            self.rate_map = dict(zip(df["filename"], df["rate_bpm"]))

        # root feature directory
        self.feature_root = Path("features")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        rec = self.records[idx]

        # ----------------------------
        # determine feature folder
        # ----------------------------
        if "Kaggle" in rec.path.parent.name:
            feature_path = self.feature_root / "kaggle" / (rec.path.stem + ".npy")
        else:
            feature_path = self.feature_root / "recorded" / (rec.path.stem + ".npy")

        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feature_path}")

        # ----------------------------
        # load cached feature
        # ----------------------------
        x = np.load(feature_path)
        # normalize features
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        # ------------------------------
        # SpecAugment style masking
        # ------------------------------
        if np.random.rand() < 0.5:
            t = x.shape[-1]
            mask_len = np.random.randint(5, 15)
            start = np.random.randint(0, max(1, t - mask_len))
            x[:, :, start:start+mask_len] = 0

        t = x.shape[-1]
        sig = None

        detection_target = np.ones((t,), dtype=np.int64)
        texture_target = rec.texture_id
        rate_target = self.rate_map.get(rec.path.name, np.nan)

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "det_y": torch.tensor(detection_target, dtype=torch.long),
            "tex_y": torch.tensor(texture_target, dtype=torch.long),
            "rate_y": torch.tensor(rate_target, dtype=torch.float32),
            "subject_id": rec.subject_id if rec.subject_id is not None else -1,
            "file": rec.path.name,
            "signal": None
        }


# --------------------------------------------------------
# Batch collation
# --------------------------------------------------------

def _collate(batch):

    t = min(b["x"].shape[-1] for b in batch)

    x = torch.stack([b["x"][..., :t] for b in batch], dim=0)
    det = torch.stack([b["det_y"][:t] for b in batch], dim=0)
    tex = torch.stack([b["tex_y"] for b in batch], dim=0)
    rate = torch.stack([b["rate_y"] for b in batch], dim=0)

    return {
        "x": x,
        "det_y": det,
        "tex_y": tex,
        "rate_y": rate,
        "file": [b["file"] for b in batch],
        "signal": [b["signal"] for b in batch],
    }


# --------------------------------------------------------
# Dataset loaders
# --------------------------------------------------------

def load_recorded_records(root: str | Path) -> List[AudioRecord]:

    root = Path(root)
    return [parse_recorded_filename(p) for p in sorted(root.glob("*.wav"))]


def load_kaggle_records(root: str | Path) -> List[AudioRecord]:

    root = Path(root)
    out = []

    for p in sorted(root.glob("*.wav")):
        try:
            out.append(parse_kaggle_filename(p))
        except KeyError:
            continue

    return out


# --------------------------------------------------------
# Leave-One-Subject-Out splits
# --------------------------------------------------------

def loso_splits(records: Sequence[AudioRecord]):

    subjects = sorted({r.subject_id for r in records if r.subject_id is not None})

    folds = []

    for sid in subjects:
        train = [r for r in records if r.subject_id != sid]
        test = [r for r in records if r.subject_id == sid]

        folds.append((train, test, sid))

    return folds


# --------------------------------------------------------
# Kaggle pretraining split
# --------------------------------------------------------

def kaggle_pretrain_split(records: Sequence[AudioRecord], seed: int = 42):

    idx = np.arange(len(records))

    tr_idx, va_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=seed,
        stratify=[r.texture_id for r in records],
    )

    return [records[i] for i in tr_idx], [records[i] for i in va_idx]


# --------------------------------------------------------
# DataLoader
# --------------------------------------------------------

def make_loader(
    records: Sequence[AudioRecord],
    cfg: PreprocessConfig,
    batch_size: int,
    shuffle: bool,
    rate_csv: str | None = None,
):

    ds = ChewingDataset(records, cfg, rate_csv=rate_csv)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        collate_fn=_collate,
    )
