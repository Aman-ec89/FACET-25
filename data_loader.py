"""Dataset and split utilities for Mandible/Kaggle audio corpora."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler

from preprocessing import PreprocessConfig
from utils import AudioRecord, parse_kaggle_filename, parse_recorded_filename

from collections import defaultdict


# ========================================================
# DATASET
# ========================================================
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

        self.feature_root = Path(
            "/content/drive/MyDrive/PhD Phase 3/Paper 7/chewing project/features"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        rec = self.records[idx]

        # feature path
        if "Kaggle" in rec.path.parent.name:
            feature_path = self.feature_root / "kaggle" / (rec.path.stem + ".npy")
        else:
            feature_path = self.feature_root / "recorded" / (rec.path.stem + ".npy")

        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feature_path}")

        x = np.load(feature_path)
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        # SpecAugment masking
        if np.random.rand() < 0.5:
            t = x.shape[-1]
            mask_len = np.random.randint(5, 15)
            start = np.random.randint(0, max(1, t - mask_len))
            x[:, :, start:start + mask_len] = 0

        t = x.shape[-1]

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "det_y": torch.ones((t,), dtype=torch.long),
            "tex_y": torch.tensor(rec.texture_id, dtype=torch.long),
            "rate_y": torch.tensor(
                self.rate_map.get(rec.path.name, np.nan), dtype=torch.float32
            ),
            "subject_id": rec.subject_id if rec.subject_id is not None else -1,
            "file": rec.path.name,
            "signal": None,
        }


# ========================================================
# COLLATE
# ========================================================
def _collate(batch):

    if len(batch) == 0:
        raise ValueError("Empty batch received in collate")

    t = min(b["x"].shape[-1] for b in batch)

    x = torch.stack([b["x"][..., :t] for b in batch])
    det = torch.stack([b["det_y"][:t] for b in batch])
    tex = torch.stack([b["tex_y"] for b in batch])
    rate = torch.stack([b["rate_y"] for b in batch])

    return {
        "x": x,
        "det_y": det,
        "tex_y": tex,
        "rate_y": rate,
        "file": [b["file"] for b in batch],
        "signal": [b["signal"] for b in batch],
    }


# ========================================================
# LOADERS
# ========================================================
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


# ========================================================
# LOSO SPLIT
# ========================================================
def loso_splits(records: Sequence[AudioRecord]):
    subjects = sorted({r.subject_id for r in records if r.subject_id is not None})

    for sid in subjects:
        train = [r for r in records if r.subject_id != sid]
        test = [r for r in records if r.subject_id == sid]
        yield train, test, sid


# ========================================================
# KAGGLE SPLIT
# ========================================================
def kaggle_pretrain_split(records: Sequence[AudioRecord], seed: int = 42):

    idx = np.arange(len(records))

    tr_idx, va_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=seed,
        stratify=[r.texture_id for r in records],
    )

    return [records[i] for i in tr_idx], [records[i] for i in va_idx]


# ========================================================
# BALANCED SAMPLER (FINAL FIXED VERSION)
# ========================================================
class BalancedBatchSampler(Sampler):

    def __init__(self, dataset, batch_size):

        self.dataset = dataset
        self.batch_size = batch_size

        self.class_indices = defaultdict(list)

        for i, rec in enumerate(dataset.records):
            self.class_indices[rec.texture_id].append(i)

        self.classes = list(self.class_indices.keys())
        self.num_classes = len(self.classes)

        self.samples_per_class = max(1, batch_size // self.num_classes)

        # ALWAYS safe number of batches
        self.num_batches = max(1, len(dataset) // batch_size)

    def __iter__(self):

        for _ in range(self.num_batches):

            batch = []

            for c in self.classes:
                indices = self.class_indices[c]

                if len(indices) == 0:
                    continue

                chosen = np.random.choice(
                    indices,
                    self.samples_per_class,
                    replace=True  # 🔥 critical fix
                )

                batch.extend(chosen.tolist())

            if len(batch) == 0:
                continue

            batch = batch[:self.batch_size]

            yield batch

    def __len__(self):
        return self.num_batches


# ========================================================
# DATALOADER
# ========================================================
def make_loader(
    records: Sequence[AudioRecord],
    cfg: PreprocessConfig,
    batch_size: int,
    shuffle: bool,
    rate_csv: str | None = None,
):

    ds = ChewingDataset(records, cfg, rate_csv=rate_csv)

    if shuffle:
        sampler = BalancedBatchSampler(ds, batch_size)

        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=2,
            pin_memory=True,
            collate_fn=_collate,
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=_collate,
    )
