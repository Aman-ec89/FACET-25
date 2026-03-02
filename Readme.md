# Frequency-Aware Multi-Task Chewing Classification (PyTorch)

Research-grade, modular implementation for:
1. Frame-level chewing detection
2. Bout-level food texture classification
3. Chewing rate estimation

Designed for Google Colab (GPU optional).

## Project files
- `data_loader.py` dataset parsing, LOSO split, and dataloaders
- `preprocessing.py` silence removal, STFT/log-Mel, z-score, subband filtering
- `model.py` subband CNN + BiLSTM/TCN + attention + multitask heads
- `attention.py` additive attention layer
- `training.py` training loop, weighted multitask loss, early stopping
- `evaluation.py` metrics summaries, plots, CSV/LaTeX tables, Wilcoxon
- `ablation.py` ablation experiment variants
- `rate_estimation.py` RMS/peak-based chewing rate estimation
- `metrics.py` all task metrics
- `utils.py` seed, parsing, CI, helpers
- `main.py` complete experiment pipeline (pretrain + LOSO + ablation + analysis)

## Colab setup
```bash
pip install -r requirements.txt
python main.py \
  --recorded_dir "Recorded audio" \
  --kaggle_dir "Kaggle audio" \
  --rate_csv "chewing_rate.csv" \
  --output_dir outputs
```

## Notes
- Recorded data must follow `subXX_texture_YY.wav`.
- Kaggle data must follow `food_x_y.wav` where `food` maps to texture.
- `chewing_rate.csv` expected columns: `filename,rate_bpm`.
- Outputs include figures (300 dpi), CSVs, and `.tex` tables.
