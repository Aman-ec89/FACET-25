# =========================================================
# MAIN RUN
# =========================================================
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

    # ========================
    # 🔍 FIXED LABEL CHECK
    # ========================
    print("\n🔍 Checking label distribution...")

    from collections import Counter
    all_labels = []

    for batch in pretrain_train:   # ✅ FIXED
        all_labels.extend(batch["tex_y"].cpu().numpy())

    print("Train label distribution:", Counter(all_labels))


    # ========================
    # Stage 1
    # ========================
    print("[Stage 1] pretraining on Kaggle")

    base_model = FrequencyAwareMultiTaskNet(m_cfg).to(device)
    base_model = train_model(base_model, pretrain_train, pretrain_val, device, t_cfg)

    # ========================
    # Stage 2 (LOSO)
    # ========================
    fold_tex = []
    cm_sum = np.zeros((4, 4), dtype=int)

    for tr_records, te_records, sid in loso_splits(recorded_records):

        print(f"[Stage 2] LOSO subject={sid}")

        tr_loader = make_loader(tr_records, p_cfg, t_cfg.batch_size, True, rate_csv=args.rate_csv)
        te_loader = make_loader(te_records, p_cfg, t_cfg.batch_size, False, rate_csv=args.rate_csv)

        model = FrequencyAwareMultiTaskNet(m_cfg).to(device)
        model.load_state_dict(base_model.state_dict(), strict=False)

        model = train_model(model, tr_loader, te_loader, device, t_cfg)

        # -------- Evaluation --------
        out = []

        model.eval()
        with torch.no_grad():
            for b in te_loader:
                x = b["x"].to(device)
                pred = model(x)

                out.append({
                    "tex_logits": pred["tex_logits"].cpu(),
                    "tex_y": b["tex_y"],
                })

        if len(out) == 0:
            print(f"⚠️ Skipping subject {sid} (no data)")
            continue

        merged = safe_merge(out)

        tex_pred = merged["tex_logits"].argmax(-1).cpu().numpy()
        tex_true = merged["tex_y"].cpu().numpy()

        tmet = texture_metrics(tex_true, tex_pred)
        tmet["fold"] = sid

        fold_tex.append(tmet)

        cm_sum += texture_confusion(tex_true, tex_pred)

    # ========================
    # RESULTS
    # ========================
    tex_df, _ = summarize_fold_metrics(fold_tex, out_dir, "texture_results")

    make_confusion_plot(cm_sum, out_dir / "texture_confusion_matrix.png")

    print(f"Done. Outputs saved to: {out_dir}")
