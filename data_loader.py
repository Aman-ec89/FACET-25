"""Improved dataloader (better normalization + augmentation)."""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler


class ChewingDataset(Dataset):

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        x = np.load(self.files[idx])

        # 🔥 reshape (CRITICAL)
        x = x.reshape(-1, x.shape[-2], x.shape[-1])

        # 🔥 BETTER NORMALIZATION (per-channel)
        mean = x.mean(axis=(1,2), keepdims=True)
        std = x.std(axis=(1,2), keepdims=True) + 1e-8
        x = (x - mean) / std

        # 🔥 TIME MASK
        if np.random.rand() < 0.5:
            t = x.shape[-1]
            l = np.random.randint(5, 15)
            s = np.random.randint(0, t-l)
            x[:, :, s:s+l] = 0

        # 🔥 FREQ MASK (NEW)
        if np.random.rand() < 0.3:
            f = x.shape[1]
            l = np.random.randint(3, 10)
            s = np.random.randint(0, f-l)
            x[:, s:s+l, :] = 0

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "tex_y": torch.tensor(self._label(self.files[idx]), dtype=torch.long)
        }

    def _label(self, path):
        name = Path(path).stem
        if "soft" in name: return 0
        if "crunchy" in name: return 1
        if "brittle" in name: return 2
        return 3


def make_loader(files, batch_size, shuffle, class_weights=None):

    ds = ChewingDataset(files)

    # ---- TRAIN LOADER (WITH SAMPLER) ----
    if shuffle and class_weights is not None:

        labels = []
        for f in files:
            name = Path(f).stem
            if "soft" in name:
                labels.append(0)
            elif "crunchy" in name:
                labels.append(1)
            elif "brittle" in name:
                labels.append(2)
            else:
                labels.append(3)

        labels = np.array(labels)

        # sample_weights = class_weights.cpu().numpy()[labels]
        class_counts = np.bincount(labels)
        sample_weights = 1.0 / class_counts[labels]

        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,   # MUST be False with sampler
            num_workers=2,
            pin_memory=True
        )

    # ---- VALIDATION LOADER (NO SAMPLER) ----
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )
