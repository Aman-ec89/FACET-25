# ==========================================
# 1. Imports
# ==========================================

import os
import numpy as np
from pathlib import Path
from preprocessing import preprocess_audio, PreprocessConfig

# ==========================================
# 2. PATHS (EDIT IF NEEDED)
# ==========================================

RECORDED_DIR = "/content/drive/MyDrive/PhD Phase 3/Paper 7/chewing project/Recorded audio"
KAGGLE_DIR  = "/content/drive/MyDrive/PhD Phase 3/Paper 7/chewing project/Kaggle audio"

OUT_DIR = "/content/features"   # output root

# ==========================================
# 3. CONFIG
# ==========================================

cfg = PreprocessConfig()

FIXED_T = 200   # time steps (tune if needed)
FIXED_F = 64    # feature dimension (must be same for all)

# ==========================================
# 4. CORE FUNCTION
# ==========================================

def process_dataset(input_root, output_root):

    input_root = Path(input_root)
    output_root = Path(output_root)

    files = list(input_root.rglob("*.wav"))

    print(f"\nProcessing: {input_root}")
    print(f"Total files: {len(files)}")

    for i, f in enumerate(files):

        try:
            # ----------------------------------
            # Maintain folder structure
            # ----------------------------------
            relative_path = f.relative_to(input_root).with_suffix(".npy")
            out_path = output_root / relative_path

            out_path.parent.mkdir(parents=True, exist_ok=True)

            # ----------------------------------
            # Feature extraction
            # ----------------------------------
            feats, _ = preprocess_audio(str(f), cfg)

            processed = []

            for k in ["B1", "B2", "B3", "B4"]:
                x = feats[k]

                # ------------------------------
                # FIX FEATURE DIMENSION
                # ------------------------------
                if x.shape[0] > FIXED_F:
                    x = x[:FIXED_F, :]
                else:
                    pad_f = FIXED_F - x.shape[0]
                    x = np.pad(x, ((0, pad_f), (0, 0)))

                # ------------------------------
                # FIX TIME DIMENSION
                # ------------------------------
                if x.shape[1] > FIXED_T:
                    x = x[:, :FIXED_T]
                else:
                    pad_t = FIXED_T - x.shape[1]
                    x = np.pad(x, ((0, 0), (0, pad_t)))

                processed.append(x)

            # shape → (4, FIXED_F, FIXED_T)
            x = np.stack(processed, axis=0)

            # ----------------------------------
            # NORMALIZATION (important)
            # ----------------------------------
            x = (x - np.mean(x)) / (np.std(x) + 1e-6)

            # ----------------------------------
            # NaN safety
            # ----------------------------------
            if np.isnan(x).any():
                print("⚠️ NaN skipped:", f.name)
                continue

            # ----------------------------------
            # Save
            # ----------------------------------
            np.save(out_path, x)

            if i % 100 == 0:
                print(f"{i}/{len(files)} processed")

        except Exception as e:
            print("❌ ERROR:", f.name, e)


# ==========================================
# 5. RUN
# ==========================================

os.makedirs(OUT_DIR, exist_ok=True)

process_dataset(RECORDED_DIR, os.path.join(OUT_DIR, "recorded"))
process_dataset(KAGGLE_DIR,  os.path.join(OUT_DIR, "kaggle"))

print("\n✅ Feature extraction complete.")
