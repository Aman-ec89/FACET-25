import os
import numpy as np
from pathlib import Path
from preprocessing import preprocess_audio, PreprocessConfig

# ----------------------------
# DATASET PATHS (EDIT THESE)
# ----------------------------

RECORDED_DIR = "/content/drive/MyDrive/PhD Phase 3/Paper 7/chewing project/Recorded audio"
KAGGLE_DIR = "/content/drive/MyDrive/PhD Phase 3/Paper 7/chewing project/Kaggle audio"

# save features in project folder
OUT_DIR = "features"

cfg = PreprocessConfig()

def process_folder(audio_dir, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    files = sorted(Path(audio_dir).glob("*.wav"))

    print("\nProcessing:", audio_dir)
    print("Total files:", len(files))

    for i, f in enumerate(files):

        try:

            feats, _ = preprocess_audio(str(f), cfg)

            t = min(v.shape[1] for v in feats.values())

            x = np.stack(
                [feats[k][:, :t] for k in ["B1", "B2", "B3", "B4"]],
                axis=0
            )

            np.save(os.path.join(out_dir, f.stem + ".npy"), x)

            if i % 100 == 0:
                print(i, "/", len(files))

        except Exception as e:
            print("ERROR:", f.name, e)

# ----------------------------
# RUN EXTRACTION
# ----------------------------

os.makedirs(OUT_DIR, exist_ok=True)

process_folder(RECORDED_DIR, os.path.join(OUT_DIR, "recorded"))
process_folder(KAGGLE_DIR, os.path.join(OUT_DIR, "kaggle"))

print("\nFeature extraction complete.")
