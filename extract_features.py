import os
import numpy as np
from pathlib import Path
from preprocessing import preprocess_audio, PreprocessConfig

RECORDED_DIR = "Recorded audio"
KAGGLE_DIR = "Kaggle audio"

OUT_DIR = "features"
os.makedirs(OUT_DIR, exist_ok=True)

cfg = PreprocessConfig()

def process_folder(audio_dir, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    files = list(Path(audio_dir).glob("*.wav"))

    for i, f in enumerate(files):

        feats, _ = preprocess_audio(str(f), cfg)

        t = min(v.shape[1] for v in feats.values())
        x = np.stack([feats[k][:, :t] for k in ["B1","B2","B3","B4"]], axis=0)

        np.save(os.path.join(out_dir, f.stem + ".npy"), x)

        if i % 100 == 0:
            print(i, "/", len(files))

process_folder(RECORDED_DIR, os.path.join(OUT_DIR, "recorded"))
process_folder(KAGGLE_DIR, os.path.join(OUT_DIR, "kaggle"))
