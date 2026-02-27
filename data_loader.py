"""Dataset and split utilities for Mandible/Kaggle audio corpora."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from preprocessing import PreprocessConfig, preprocess_audio
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

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        feats, sig = preprocess_audio(str(rec.path), self.cfg)
        t = min(v.shape[1] for v in feats.values())
        x = np.stack([feats[k][:, :t] for k in ["B1", "B2", "B3", "B4"]], axis=0)
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
            "signal": torch.tensor(sig, dtype=torch.float32),
        }


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


def loso_splits(records: Sequence[AudioRecord]) -> List[Tuple[List[AudioRecord], List[AudioRecord], int]]:
    subjects = sorted({r.subject_id for r in records if r.subject_id is not None})
    folds = []
    for sid in subjects:
        train = [r for r in records if r.subject_id != sid]
        test = [r for r in records if r.subject_id == sid]
        folds.append((train, test, sid))
    return folds


def kaggle_pretrain_split(records: Sequence[AudioRecord], seed: int = 42):
    idx = np.arange(len(records))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=[r.texture_id for r in records])
    return [records[i] for i in tr_idx], [records[i] for i in va_idx]


def make_loader(records: Sequence[AudioRecord], cfg: PreprocessConfig, batch_size: int, shuffle: bool, rate_csv: str | None = None):
    ds = ChewingDataset(records, cfg, rate_csv=rate_csv)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=_collate)
