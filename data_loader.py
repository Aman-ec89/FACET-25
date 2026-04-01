"""Dataset and split utilities for RECORDED chewing features (.npy based, LOSO-ready)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from preprocessing import PreprocessConfig
from utils import AudioRecord, parse_recorded_filename

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

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        rec = self.records[idx]

        # ==========================================
        # LOAD FEATURE FILE
        # ==========================================
        feature_path = Path(rec.path).with_suffix(".npy")

        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feature_path}")

        x = np.load(feature_path)

        # ==========================================
        # 🔥 CRITICAL FIX: RESHAPE
        # (4,5,64,200) → (20,64,200)
        # ==========================================
        x = x.reshape(-1, x.shape[-2], x.shape[-1])

        # ==========================================
        # NORMALIZATION
        # ==========================================
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        # ==========================================
        # AUGMENTATION (TIME MASK)
        # ==========================================
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
# COLLATE FUNCTION
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
# LOAD RECORDED RECORDS (NPY BASED)
# ========================================================
def load_recorded_records(root: str | Path) -> List[AudioRecord]:

    root = Path(root)

    records = []
    for p in sorted(root.glob("*.npy")):
        fake_path = p.with_suffix(".wav")  # parser compatibility
        records.append(parse_recorded_filename(fake_path))

    return records


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
# BALANCED SAMPLER
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
                    replace=True
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
