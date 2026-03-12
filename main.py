"""Entry point for end-to-end chewing classification experiments (Colab-ready)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ablation import ablation_variants
from data_loader import kaggle_pretrain_split, load_kaggle_records, load_recorded_records, loso_splits, make_loader
from evaluation import (
    compute_wilcoxon,
    make_ablation_plot,
    make_confusion_plot,
    make_rate_scatter,
    make_roc_plot,
    make_attention_plot,
    make_psd_subband_plot,
    summarize_fold_metrics,
)
from metrics import chewing_rate_metrics, detection_metrics, texture_confusion, texture_metrics
from model import FrequencyAwareMultiTaskNet, ModelConfig
from preprocessing import PreprocessConfig
from rate_estimation import estimate_chewing_rate_bpm
from training import TrainConfig, train_model
from utils import count_parameters, device_auto, ensure_dir, set_seed


def _eval_logits(out):
    det_prob = torch.softmax(out["det_logits"], dim=-1)[..., 1].reshape(-1).numpy()
    det_pred = out["det_logits"].argmax(-1).reshape(-1).numpy()
    det_true = out["det_y"].reshape(-1).numpy()
    tex_pred = out["tex_logits"].argmax(-1).numpy()
    tex_true = out["tex_y"].numpy()
    return det_true, det_pred, det_prob, tex_true, tex_pred


def run(args):
    set_seed(args.seed)
    device = device_auto()
    out_dir = ensure_dir(args.output_dir)

    p_cfg = PreprocessConfig()
    t_cfg = TrainConfig(batch_size=args.batch_size, epochs=args.epochs)
    m_cfg = ModelConfig()

    kaggle_records = load_kaggle_records(args.kaggle_dir)
    recorded_records = load_recorded_records(args.recorded_dir)

    ktr, kva = kaggle_pretrain_split(kaggle_records, seed=args.seed)
    pretrain_train = make_loader(ktr, p_cfg, t_cfg.batch_size, True)
    pretrain_val = make_loader(kva, p_cfg, t_cfg.batch_size, False)

    base_model = FrequencyAwareMultiTaskNet(m_cfg).to(device)
    print("[Stage 1] pretraining on Kaggle")
    base_model = train_model(base_model, pretrain_train, pretrain_val, device, t_cfg)

    fold_det, fold_tex, fold_rate = [], [], []
    roc_true_all, roc_prob_all = [], []
    cm_sum = np.zeros((4, 4), dtype=int)

    for tr_records, te_records, sid in loso_splits(recorded_records):
        print(f"[Stage 2] LOSO subject={sid}")
        tr_loader = make_loader(tr_records, p_cfg, t_cfg.batch_size, True, rate_csv=args.rate_csv)
        te_loader = make_loader(te_records, p_cfg, t_cfg.batch_size, False, rate_csv=args.rate_csv)

        model = FrequencyAwareMultiTaskNet(m_cfg).to(device)
        model.load_state_dict(base_model.state_dict(), strict=False)
        model = train_model(model, tr_loader, te_loader, device, t_cfg)

        out = []
        model.eval()
        with torch.no_grad():
            for b in te_loader:
                x = b["x"].to(device)
                pred = model(x)
                out.append(
                    {
                        "det_logits": pred["det_logits"].cpu(),
                        "tex_logits": pred["tex_logits"].cpu(),
                        "det_y": b["det_y"],
                        "tex_y": b["tex_y"],
                    }
                )
        merged = {k: torch.cat([o[k] for o in out], dim=0) for k in out[0]}
        det_true, det_pred, det_prob, tex_true, tex_pred = _eval_logits(merged)

        dmet = detection_metrics(det_true, det_pred, det_prob)
        tmet = texture_metrics(tex_true, tex_pred)
        dmet["fold"] = sid
        tmet["fold"] = sid
        fold_det.append(dmet)
        fold_tex.append(tmet)
        roc_true_all.extend(det_true.tolist())
        roc_prob_all.extend(det_prob.tolist())
        cm_sum += texture_confusion(tex_true, tex_pred)

        gt_rate, pr_rate = [], []
        for b in te_loader:
            for sig, gt in zip(b["signal"], b["rate_y"]):
                if torch.isnan(gt):
                    continue
                gt_rate.append(float(gt.item()))
                pr_rate.append(estimate_chewing_rate_bpm(sig.numpy(), fs=p_cfg.sr))
        if gt_rate:
            rmet = chewing_rate_metrics(gt_rate, pr_rate)
            rmet["fold"] = sid
            fold_rate.append(rmet)

    det_df, _ = summarize_fold_metrics(fold_det, out_dir, "detection_results")
    tex_df, _ = summarize_fold_metrics(fold_tex, out_dir, "texture_results")
    if fold_rate:
        rate_df, _ = summarize_fold_metrics(fold_rate, out_dir, "rate_results")
    else:
        rate_df = pd.DataFrame(columns=["fold", "mae", "rmse", "r2"])

    make_roc_plot(np.array(roc_true_all), np.array(roc_prob_all), out_dir / "roc_curve.png")
    make_confusion_plot(cm_sum, out_dir / "texture_confusion_matrix.png")
    # Attention map from one validation mini-batch
    
    # for b in pretrain_val:
    #     with torch.no_grad():
    #         attn = base_model(b["x"].to(device))["attn"].cpu().numpy()
    #     make_attention_plot(attn, out_dir / "attention_weights.png")
    #     # make_psd_subband_plot(b["signal"][0].numpy(), fs=p_cfg.sr, path=out_dir / "psd_subbands.png")
    #     # if b["signal"] is not None:
    #     #     make_psd_subband_plot(b["signal"][0].numpy(), fs=p_cfg.sr, path=out_dir / "psd_subbands.png")
    #     try:
    #         if b.get("signal") is not None and len(b["signal"]) > 0:
    #             make_psd_subband_plot(b["signal"][0].numpy(),fs=p_cfg.sr,path=out_dir / "psd_subbands.png")
    #         except Exception:
    #             pass
    #         break

    # Attention map from one validation mini-batch
    for b in pretrain_val:
        with torch.no_grad():
            attn = base_model(b["x"].to(device))["attn"].cpu().numpy()
            make_attention_plot(attn, out_dir / "attention_weights.png")
            try:
                if b.get("signal") is not None and len(b["signal"]) > 0:
                    make_psd_subband_plot(b["signal"][0].numpy(),fs=p_cfg.sr,path=out_dir / "psd_subbands.png")
            except Exception:
                pass
            break
        if not rate_df.empty:
            make_rate_scatter(rate_df["mae"].tolist(), rate_df["rmse"].tolist(), out_dir / "rate_scatter.png")
   
    # Ablations
    ab_rows = []
    variants = ablation_variants(m_cfg)
    for name, cfg in variants.items():
        model = FrequencyAwareMultiTaskNet(cfg).to(device)
        model.load_state_dict(base_model.state_dict(), strict=False)
        model = train_model(model, pretrain_train, pretrain_val, device, t_cfg)
        out = []
        with torch.no_grad():
            for b in pretrain_val:
                pred = model(b["x"].to(device))
                out.append({"det_logits": pred["det_logits"].cpu(), "tex_logits": pred["tex_logits"].cpu(), "det_y": b["det_y"], "tex_y": b["tex_y"]})
        merged = {k: torch.cat([o[k] for o in out], dim=0) for k in out[0]}
        _, _, _, tex_true, tex_pred = _eval_logits(merged)
        tm = texture_metrics(tex_true, tex_pred)
        tm["variant"] = name
        ab_rows.append(tm)
    ab_df = pd.DataFrame(ab_rows)
    ab_df.to_csv(out_dir / "ablation_results.csv", index=False)
    (out_dir / "ablation_results.tex").write_text(ab_df.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))
    make_ablation_plot(ab_df, out_dir / "ablation_bar.png")

    base_scores = ab_df.loc[ab_df.variant == "full", "macro_f1"].values
    wil_rows = []
    for variant in ab_df.variant.unique():
        if variant == "full":
            continue
        v = ab_df.loc[ab_df.variant == variant, "macro_f1"].values
        if len(base_scores) == len(v):
            stat = compute_wilcoxon(base_scores, v)
            stat["variant"] = variant
            wil_rows.append(stat)
    if wil_rows:
        pd.DataFrame(wil_rows).to_csv(out_dir / "wilcoxon_results.csv", index=False)

    # Computational analysis
    params = count_parameters(base_model)
    flops = None
    try:
        from ptflops import get_model_complexity_info

        flops, _ = get_model_complexity_info(base_model, (4, 64, 100), as_strings=False, print_per_layer_stat=False)
    except Exception:
        flops = np.nan

    dummy = torch.randn(1, 4, 64, 100).to(device)
    n = 30
    starter = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            _ = base_model(dummy)
    latency = (time.perf_counter() - starter) / n
    rtf = latency / 5.0

    pd.DataFrame(
        [{"params": params, "flops": flops, "latency_s_per_5s": latency, "real_time_factor": rtf}]
    ).to_csv(out_dir / "computational_analysis.csv", index=False)

    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorded_dir", type=str, default="Recorded audio")
    ap.add_argument("--kaggle_dir", type=str, default="Kaggle audio")
    ap.add_argument("--rate_csv", type=str, default="chewing_rate.csv")
    ap.add_argument("--output_dir", type=str, default="outputs")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    run(ap.parse_args())
