#extract features


import os
import numpy as np
from pathlib import Path
from preprocessing import preprocess_audio, PreprocessConfig

# ----------------------------
# DATASET PATHS (EDIT THESE)
# ----------------------------

RECORDED_DIR = "/content/drive/MyDrive/PhD Phase 3/Paper 7/chewing project/Recorded audio"
KAGGLE_DIR = "/content/drive/MyDrive/PhD Phase 3/Paper 7/chewing project/Kaggle audio"

OUT_DIR = "features"

cfg = PreprocessConfig()

# 🔧 FIXED TIME LENGTH
FIXED_T = 200


def process_folder(audio_dir, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    files = sorted(Path(audio_dir).glob("*.wav"))

    total_files = len(files)

    print("\nProcessing:", audio_dir)
    print("Total files:", total_files)

    # ==========================================
    # 🔥 PROGRESS INIT
    # ==========================================
    if total_files == 0:
        print("⚠️ No files found")
        return

    for i, f in enumerate(files):

        try:
            feats, _ = preprocess_audio(str(f), cfg)

            processed = []

            for k in ["B1", "B2", "B3", "B4"]:
                x = feats[k]

                # 🔧 FIX TIME DIMENSION (MAIN FIX)
                if x.shape[1] > FIXED_T:
                    x = x[:, :FIXED_T]
                else:
                    pad = FIXED_T - x.shape[1]
                    x = np.pad(x, ((0, 0), (0, pad)))

                processed.append(x)

            # shape → (4, 64, 200)
            x = np.stack(processed, axis=0)

            np.save(os.path.join(out_dir, f.stem + ".npy"), x)

        except Exception as e:
            print("ERROR:", f.name, e)

        # ==========================================
        # 🔥 PROGRESS DISPLAY (REAL %)
        # ==========================================
        progress = (i + 1) / total_files * 100
        print(f"\rProgress: {progress:6.2f}% ({i+1}/{total_files})", end="", flush=True)

    print()  # newline after loop


# ----------------------------
# RUN EXTRACTION
# ----------------------------

# ❗ IMPORTANT: remove old features first
if os.path.exists(OUT_DIR):
    import shutil
    shutil.rmtree(OUT_DIR)

os.makedirs(OUT_DIR, exist_ok=True)

process_folder(RECORDED_DIR, os.path.join(OUT_DIR, "recorded"))
process_folder(KAGGLE_DIR, os.path.join(OUT_DIR, "kaggle"))

print("\nFeature extraction complete.")
