from .shared import *
from .data import *
from .modeling import *
from .losses import (
    build_reversal_sample_weight,
    compute_total_task_loss,
    get_rev_aux_target,
)
from .metrics import (
    collect_structured_metrics_from_loader,
    _score_value_or_default,
    has_reversal_np,
)
from .evaluation import evaluate_and_plot

# =========================
# Main
# =========================
def main():
    # =========================
    # 本次运行输出目录（程序运行结果/时间戳）
    # =========================
    refresh_runtime_training_config()
    apply_smoke_overrides()

    RUN_DIR = make_run_dir(prefix="TRAIN_V5_4_STATECOND_REV")
    CKPT_DIR = RUN_DIR / "checkpoints"
    FIG_DIR = RUN_DIR / "figures"
    LOG_DIR = RUN_DIR / "logs"

    orig_stdout = sys.stdout
    tee = TeeStdout(LOG_DIR / "train.log", console_stream=orig_stdout)
    sys.stdout = tee
    try:
        try_copy_self(RUN_DIR)

        print("RUN_DIR:", str(RUN_DIR))
        print("设备:", DEVICE)
        print("时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
        print("========================================")

        protocol_config, split_subjects = load_protocol_split(PROTOCOL_CONFIG_PATH, FROZEN_SPLIT_PATH)
        print("Protocol version:", protocol_config.get("protocol_version"))
        print("Protocol split source:", str(FROZEN_SPLIT_PATH))

        style_map = load_driver_style_map(STYLE_CSV)
        X_pool, y_pool, curve_pool, ctx_pool, base_pool, sample_meta_df, feature_names, input_qc_summary = build_all_samples(style_map)
        INPUT_QC_DIR = RUN_DIR / "input_qc"
        INPUT_QC_DIR.mkdir(parents=True, exist_ok=True)
        selected_feature_columns_path = INPUT_QC_DIR / "selected_feature_columns.json"
        speed_source_report_path = INPUT_QC_DIR / "speed_source_report.json"
        if INPUT_PIPELINE_VERSION == "legacy_v1":
            speed_source_policy = "legacy_v1: zx|vx divided by 3.6 when present"
        else:
            speed_source_policy = "fixed_v20260421: prefer zx1|v_km/h / 3.6, fallback zx|vx as m/s"
        speed_report_payload = {
            "input_pipeline_version": INPUT_PIPELINE_VERSION,
            "speed_source_policy": speed_source_policy,
            "speed_source_counts": input_qc_summary.get("speed_source_counts", {}),
            "speed_warning_vehicle_files": input_qc_summary.get("speed_warning_vehicle_files", []),
            "vehicle_speed_reports": [
                {
                    "vehicle_file": rec.get("vehicle_file"),
                    **dict(rec.get("speed_source_report", {})),
                }
                for rec in input_qc_summary.get("records", [])
            ],
        }
        save_json(selected_feature_columns_path, input_qc_summary)
        save_json(speed_source_report_path, speed_report_payload)
        print(
            f"Input pipeline: version={INPUT_PIPELINE_VERSION} | "
            f"use_pedals={int(USE_PEDALS)} | use_vy={int(USE_VY)} | use_vroll={int(USE_VROLL)} | "
            f"use_mu={int(USE_MU)} | use_z={int(USE_Z)} | use_is_curve_ctx={int(USE_IS_CURVE_CTX)}"
        )
        print(
            f"Speed source policy: {speed_source_policy} | "
            f"counts={input_qc_summary.get('speed_source_counts', {})} | "
            f"warnings={len(input_qc_summary.get('speed_warning_vehicle_files', []))}"
        )
        print(
            f"Input QC artifacts: selected={selected_feature_columns_path} | "
            f"speed={speed_source_report_path}"
        )
        # ---- dump feature names for verification ----
        try:
            save_json(RUN_DIR / "feature_names.json", {"n_features": int(len(feature_names)), "feature_names": feature_names})
            print("🧩 已保存特征列表:", RUN_DIR / "feature_names.json")
        except Exception as e:
            print("⚠ 保存特征列表失败:", e)

        total = len(X_pool)
        if total == 0:
            print("❌ 没有有效事件样本")
            return

        if len(sample_meta_df) != total:
            raise ValueError(f"sample meta length mismatch: total={total}, meta={len(sample_meta_df)}")

        split_indices = build_subject_split_indices(sample_meta_df, split_subjects)
        smoke_sampling_policy = "disabled"
        if SMOKE_MODE:
            rng = np.random.default_rng(SEED)
            split_indices, smoke_counts = choose_smoke_indices(split_indices, SMOKE_MAX_SAMPLES, rng)
            smoke_sampling_policy = f"protocol-first per-split subsample with guaranteed non-empty splits; counts={smoke_counts}"
            print(f"[SMOKE] {smoke_sampling_policy}")

        train_idx = np.asarray(split_indices["train"], dtype=np.int64)
        val_idx = np.asarray(split_indices["val"], dtype=np.int64)
        test_idx = np.asarray(split_indices["test"], dtype=np.int64)
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                f"Protocol split must keep train/val/test non-empty, got train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
            )

        selected_idx = np.concatenate([train_idx, val_idx, test_idx])
        selected_meta_df = sample_meta_df.loc[selected_idx].copy()
        selected_meta_df["protocol_split_applied"] = ""
        selected_meta_df.loc[train_idx, "protocol_split_applied"] = "train"
        selected_meta_df.loc[val_idx, "protocol_split_applied"] = "val"
        selected_meta_df.loc[test_idx, "protocol_split_applied"] = "test"
        selected_meta_df.to_csv(str(RUN_DIR / "selected_samples_with_split.csv"), index=False, encoding="utf-8-sig")

        split_audit, _, split_sample_counts_df = export_split_audit(
            RUN_DIR,
            sample_meta_df,
            {"train": train_idx, "val": val_idx, "test": test_idx},
            split_subjects,
            protocol_config,
            SMOKE_MODE,
            smoke_sampling_policy,
        )
        print("Split audit saved:", RUN_DIR / "split_audit.json")

        # ---- road-type split (straight/curve) using ONLY history-window curvature stats ----
        # ---- NEW(v5.4): reversal label from GT future steer (aux training + stratified eval) ----
        # ---- reversal label (weak & strong) computed from FUTURE steer (GT) ----
        rev_gt_weak = np.array([has_reversal_np(y[:, 0], eps=REV_EPS_WEAK) for y in y_pool], dtype=np.float32)

        # strong reversal: requires crossing both +REV_EPS_STRONG and -REV_EPS_STRONG, AND a sufficient peak magnitude
        rev_gt_strong = []
        for y in y_pool:
            steer_f = y[:, 0]
            peak_abs = float(np.max(np.abs(steer_f))) if steer_f.size else 0.0
            r = has_reversal_np(steer_f, eps=REV_EPS_STRONG)
            if r > 0.5 and peak_abs >= STRONG_PEAK_THR:
                rev_gt_strong.append(1.0)
            else:
                rev_gt_strong.append(0.0)
        rev_gt_strong = np.asarray(rev_gt_strong, dtype=np.float32)

        # label used for rev_head training
        rev_gt = rev_gt_strong if USE_STRONG_REV_LOSS else rev_gt_weak

        try:
            print(f"🔁 reversal labels: weak_rate={float(np.mean(rev_gt_weak)):.3f}, strong_rate={float(np.mean(rev_gt_strong)):.3f}, used={'strong' if USE_STRONG_REV_LOSS else 'weak'}")
        except Exception:
            pass


        curve_feat_name, curve_feat_idx = find_feature_in_list(feature_names, ["lanecurvature", "curvature"])
        if curve_feat_idx is None:
            print("⚠️ 未找到曲率特征列（lanecurvature/curvature），将默认全部视为直道。")
            curve_scores = np.zeros((total,), dtype=np.float32)
            curve_thr = 0.0
            is_curve = np.zeros((total,), dtype=np.int64)
        else:
            curve_scores = np.array(
                [float(np.mean(np.abs(x[:, curve_feat_idx]))) for x in X_pool],
                dtype=np.float32
            )
            curve_thr = auto_curve_threshold(curve_scores[train_idx])
            is_curve = (curve_scores > curve_thr).astype(np.int64)

            ratio_curve = float(np.mean(is_curve))
            print(f"🛣 road_type: 使用历史 3s 平均|curvature| 分割直/弯")
            print(f"   曲率列: {curve_feat_name}")
            print(f"   curve_thr = {curve_thr:.3e}  (train auto)")
            print(f"   curve_ratio(all) = {ratio_curve*100:.1f}%  |  straight_ratio = {(1-ratio_curve)*100:.1f}%")

        # ---- standardize encoder src features ----
        all_X_concat = np.concatenate([X_pool[int(i)] for i in train_idx], axis=0)
        feat_mean = all_X_concat.mean(axis=0)
        feat_std = all_X_concat.std(axis=0)
        feat_std[feat_std < 1e-6] = 1e-6
        for i in range(len(X_pool)):
            X_pool[i] = (X_pool[i] - feat_mean) / feat_std
    
        # ---- standardize outputs ----
        all_y_concat = np.concatenate([y_pool[int(i)].reshape(-1, 3) for i in train_idx], axis=0)
        y_mean = all_y_concat.mean(axis=0)
        y_std  = all_y_concat.std(axis=0)
        y_std[y_std < 1e-6] = 1e-6

        y_mean_t = torch.tensor(y_mean, device=DEVICE, dtype=torch.float32)
        y_std_t = torch.tensor(y_std, device=DEVICE, dtype=torch.float32)
    
        # ---- curve std ----
        all_curve_concat = np.concatenate([curve_pool[int(i)] for i in train_idx], axis=0)
        curve_mean = all_curve_concat.mean()
        curve_std = all_curve_concat.std()
        if curve_std < 1e-6:
            curve_std = 1e-6
    
        # ---- ctx std ----
        ctx_array = np.stack([ctx_pool[int(i)] for i in train_idx], axis=0)
        ctx_mean = ctx_array.mean(axis=0)
        ctx_std  = ctx_array.std(axis=0)
        ctx_std[ctx_std < 1e-6] = 1e-6
    
        # ---- teacher base feat z-score (train stats only) ----
        base_train = np.stack([base_pool[int(i)] for i in train_idx], axis=0)  # (Ntr,12)
        teacher_base_names = [
            "hr", "eda_tonic", "eda_phasic", "emg_rms",
            "alpha_asym", "occ_ta_beta", "frontal_ta_beta", "temporal_ta_beta",
            "occ_alpha_abs", "temporal_gamma_rel", "occ_gamma_rel", "frontal_gamma_rel",
        ]
        finite_count = np.isfinite(base_train).sum(axis=0)
        missing_count = (~np.isfinite(base_train)).sum(axis=0)
        valid_ratio = (finite_count / max(1, len(train_idx))).astype(np.float32)
        all_missing_mask = (finite_count == 0)

        base_mu = np.zeros((base_train.shape[1],), dtype=np.float32)
        base_sd = np.ones((base_train.shape[1],), dtype=np.float32)
        valid_stat_mask = ~all_missing_mask
        if np.any(valid_stat_mask):
            base_mu[valid_stat_mask] = np.nanmean(base_train[:, valid_stat_mask], axis=0).astype(np.float32)
            base_sd[valid_stat_mask] = np.nanstd(base_train[:, valid_stat_mask], axis=0).astype(np.float32)
        base_sd[base_sd < 1e-6] = 1e-6

        teacher_base_stats = []
        for i, name in enumerate(teacher_base_names):
            teacher_base_stats.append({
                "index": int(i),
                "name": name,
                "finite_count": int(finite_count[i]),
                "missing_count": int(missing_count[i]),
                "valid_ratio": float(valid_ratio[i]),
                "all_missing": bool(all_missing_mask[i]),
                "mean": float(base_mu[i]),
                "std": float(base_sd[i]),
            })
        save_json(RUN_DIR / "teacher_base_missing_stats.json", {
            "fit_split": "train",
            "fit_sample_count": int(len(train_idx)),
            "all_missing_indices": [int(i) for i in np.where(all_missing_mask)[0]],
            "all_missing_names": [teacher_base_names[int(i)] for i in np.where(all_missing_mask)[0]],
            "stats": teacher_base_stats,
        })
        print(
            f"Teacher-base missing dims: {int(all_missing_mask.sum())}/{int(len(all_missing_mask))} | "
            f"all-missing={ [teacher_base_names[int(i)] for i in np.where(all_missing_mask)[0]] }"
        )

        def zscore_base(x12):
            x = x12.copy()
            # NaN -> mean（等价于 z=0），避免污染
            nan_mask = ~np.isfinite(x)
            x[nan_mask] = np.take(base_mu, np.where(nan_mask)[0])
            return (x - base_mu) / base_sd

        base_z_all = np.stack([zscore_base(x) for x in base_pool], axis=0)  # (N,12)
        z_phys_raw, teacher_state_meta = build_teacher_state(
            base_z_all,
            mode=TEACHER_STATE_MODE,
            state_dim=TEACHER_STATE_DIM,
            fit_indices=train_idx,
        )

        # 进一步把 teacher latent 再标准化（train stats）
        z_tr = z_phys_raw[train_idx]
        z_mu = np.mean(z_tr, axis=0)
        z_sd = np.std(z_tr, axis=0)
        z_sd[z_sd < 1e-6] = 1e-6
        z_phys = ((z_phys_raw - z_mu) / z_sd).astype(np.float32)
        teacher_state_meta["fit_split"] = "train"
        teacher_state_meta["fit_sample_count"] = int(len(train_idx))
        teacher_state_meta["z_mu"] = z_mu.astype(np.float32).tolist()
        teacher_state_meta["z_sd"] = z_sd.astype(np.float32).tolist()
        teacher_state_meta["state_dim"] = int(z_phys.shape[1])
        teacher_state_meta["base_feature_names"] = teacher_base_names
        teacher_state_meta["base_all_missing_indices"] = [int(i) for i in np.where(all_missing_mask)[0]]
        teacher_state_meta["base_all_missing_names"] = [teacher_base_names[int(i)] for i in np.where(all_missing_mask)[0]]
        teacher_state_meta["base_valid_ratio"] = valid_ratio.astype(np.float32).tolist()
        teacher_state_meta["base_mu"] = base_mu.astype(np.float32).tolist()
        teacher_state_meta["base_sd"] = base_sd.astype(np.float32).tolist()
        teacher_state_meta["base_missing_stats_file"] = "teacher_base_missing_stats.json"
        teacher_state_meta["base_valid_stats_count"] = int(valid_stat_mask.sum())
        teacher_state_meta["base_all_missing_count"] = int(all_missing_mask.sum())
        save_json(RUN_DIR / "teacher_state_meta.json", teacher_state_meta)
        print(
            f"Teacher-state mode={teacher_state_meta['mode']} | "
            f"state_dim={teacher_state_meta['state_dim']} | "
            f"components={teacher_state_meta['component_names']}"
        )
        state_dim = int(z_phys.shape[1])
        context_dim = int(ctx_pool[0].shape[0] + state_dim)
    
        def build_dataset(indices):
            return MultiTaskFutureWithCurveDataset(
                subset_list(X_pool, indices),
                subset_list(y_pool, indices),
                subset_list(curve_pool, indices),
                subset_list(ctx_pool, indices),
                subset_array(z_phys, indices),
                subset_array(rev_gt, indices),
                subset_array(rev_gt_weak, indices),
                subset_array(rev_gt_strong, indices),
                y_mean, y_std, curve_mean, curve_std, ctx_mean, ctx_std,
                subset_array(curve_scores, indices),
                subset_array(is_curve, indices),
            )

        train_dataset = build_dataset(train_idx)
        val_dataset = build_dataset(val_idx)
        test_dataset = build_dataset(test_idx)
    
        def collate_fn(batch):
            src = torch.stack([torch.from_numpy(b["src"]).float() for b in batch], dim=0)
            y_norm = torch.stack([torch.from_numpy(b["y_norm"]).float() for b in batch], dim=0)
            curve_norm = torch.stack([torch.from_numpy(b["curve_norm"]).float() for b in batch], dim=0)
            ctx = torch.stack([torch.from_numpy(b["ctx"]).float() for b in batch], dim=0)
            z_phys = torch.stack([torch.from_numpy(b["z_phys"]).float() for b in batch], dim=0)
            z_mask = torch.stack([torch.from_numpy(b["z_mask"]).float() for b in batch], dim=0)  # (B,1)
            rev_gt = torch.stack([torch.from_numpy(b["rev_gt"]).float() for b in batch], dim=0)  # (B,1)
            rev_gt_weak = torch.stack([torch.from_numpy(b["rev_gt_weak"]).float() for b in batch], dim=0)  # (B,1)
            rev_gt_strong = torch.stack([torch.from_numpy(b["rev_gt_strong"]).float() for b in batch], dim=0)  # (B,1)
            idx = torch.stack([torch.from_numpy(b["idx"]).long() for b in batch], dim=0).squeeze(1)
            curve_score = torch.stack([torch.from_numpy(b["curve_score"]).float() for b in batch], dim=0).squeeze(1)
            is_curve = torch.stack([torch.from_numpy(b["is_curve"]).long() for b in batch], dim=0).squeeze(1)
            return {"src": src, "y_norm": y_norm, "curve_norm": curve_norm, "ctx": ctx, "z_phys": z_phys, "z_mask": z_mask,
            "rev_gt": rev_gt,
            "rev_gt_weak": rev_gt_weak,
            "rev_gt_strong": rev_gt_strong,
                    "idx": idx, "curve_score": curve_score, "is_curve": is_curve}
    
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
        val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  collate_fn=collate_fn, pin_memory=torch.cuda.is_available())

        # ---- class balancing for reversal aux loss ----
        try:
            pos_cnt = float(np.sum(rev_gt[train_idx] > 0.5))
            neg_cnt = float(len(train_idx) - pos_cnt)
            pw = neg_cnt / max(1.0, pos_cnt)
            rev_pos_weight = torch.tensor(pw, device=DEVICE)
            print(f"🔁 rev_head pos_weight={pw:.3f}  (pos={pos_cnt:.0f}, neg={neg_cnt:.0f})")
        except Exception:
            rev_pos_weight = torch.tensor(1.0, device=DEVICE)
        try:
            strong_pos_cnt = float(np.sum(rev_gt_strong[train_idx] > 0.5))
            strong_pos_neg_cnt = float(len(train_idx) - strong_pos_cnt)
            spw = strong_pos_neg_cnt / max(1.0, strong_pos_cnt)
            strong_pos_gate_pos_weight = torch.tensor(spw, device=DEVICE)
            print(f"strong_pos_gate pos_weight={spw:.3f}  (pos={strong_pos_cnt:.0f}, neg={strong_pos_neg_cnt:.0f})")
        except Exception:
            strong_pos_gate_pos_weight = torch.tensor(1.0, device=DEVICE)
    
        # model
        model = Past2FutureMultiTaskRoadPreview(
            input_dim=len(feature_names),
            context_dim=context_dim,
            future_len=FUTURE_LEN,
            out_dim=3,
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_layers_enc=NUM_LAYERS_ENC,
            num_layers_dec=NUM_LAYERS_DEC,
            dim_feedforward=FFN_DIM,
            dropout=DROPOUT,
            max_len_enc=WIN_LEN,
            max_len_dec=FUTURE_LEN,
            state_dim=state_dim,
            enable_steer_coarse_fine=ENABLE_STEER_COARSE_FINE,
            enable_manual_coarse_upsample=ENABLE_MANUAL_COARSE_UPSAMPLE,
            trend_pool_kernel=TREND_POOL_KERNEL,
            trend_pool_stride=TREND_POOL_STRIDE,
            enable_late_residual_head=ENABLE_LATE_RESIDUAL_HEAD,
            late_residual_start_sec=LATE_RESIDUAL_START_SEC,
            enable_late_reversal_gate=ENABLE_LATE_REV_GATE,
            late_rev_gate_start_sec=LATE_REV_GATE_START_SEC,
            late_rev_gate_scale=LATE_REV_GATE_SCALE,
            late_rev_gate_ramp_power=LATE_REV_GATE_RAMP_POWER,
            enable_strong_pos_gate=ENABLE_STRONG_POS_GATE,
            strong_pos_gate_start_sec=STRONG_POS_GATE_START_SEC,
            strong_pos_gate_scale=STRONG_POS_GATE_SCALE,
            strong_pos_gate_ramp_power=STRONG_POS_GATE_RAMP_POWER,
            strong_pos_gate_prob_center=STRONG_POS_GATE_PROB_CENTER,
        ).to(DEVICE)
    
        optim = build_optimizer(model)
        print(f"Split counts | train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")
    
        print(f"训练集样本数: {len(train_dataset)} | 测试集样本数: {len(test_dataset)}")
        print(f"历史窗口: {WIN_SEC:.1f}s({WIN_LEN}) 未来窗口: {FUTURE_SEC:.1f}s({FUTURE_LEN})")
        print(
            f"Response-state v1: enabled={ENABLE_RESPONSE_STATE_V1} | "
            f"state={ENABLE_STATE_DISTILL} | reversal={ENABLE_REVERSAL_AUX} | "
            f"peaktime={ENABLE_PEAKTIME_AUX} | peakintensity={ENABLE_PEAKINTENSITY_AUX}"
        )
        print(
            f"Teacher-state config: mode={TEACHER_STATE_MODE} | requested_dim={TEACHER_STATE_DIM} | "
            f"actual_dim={state_dim}"
        )
        print(f"Distill: lambda_state={LAMBDA_STATE} | lambda_rev={LAMBDA_REV} | REV_EPS={REV_EPS}\n")
        print(
            f"Steer unit: source={STEER_SOURCE_UNIT} -> target={STEER_ANGLE_UNIT} | "
            f"scale={STEER_ANGLE_SCALE:.6f} | plot={STEER_PLOT_UNIT}"
        )
        print(
            f"Reversal bridge: mode={REV_BRIDGE_MODE} | "
            f"hybrid_weak_coef={REV_HYBRID_WEAK_COEF:.3f} | hybrid_strong_coef={REV_HYBRID_STRONG_COEF:.3f}"
        )
        if ENABLE_LATE_REV_GATE:
            print(
                f"Late rev gate: enabled=True | start_sec={LATE_REV_GATE_START_SEC:.2f} | "
                f"scale={LATE_REV_GATE_SCALE:.2f} | ramp_power={LATE_REV_GATE_RAMP_POWER:.2f}"
            )
        if ENABLE_STRONG_POS_GATE:
            print(
                f"Strong-pos gate: enabled=True | start_sec={STRONG_POS_GATE_START_SEC:.2f} | "
                f"scale={STRONG_POS_GATE_SCALE:.2f} | ramp_power={STRONG_POS_GATE_RAMP_POWER:.2f} | "
                f"prob_center={STRONG_POS_GATE_PROB_CENTER:.2f} | lambda={LAMBDA_STRONG_POS_GATE:.3f}"
            )
        if ENABLE_LATE_RESIDUAL_HEAD:
            print(
                f"Late residual head: enabled=True | start_sec={LATE_RESIDUAL_START_SEC:.2f} | "
                f"w_late_residual={W_LATE_RESIDUAL:.3f}"
            )
        if ENABLE_STEER_COARSE_FINE:
            print(f"Manual coarse upsample: enabled={ENABLE_MANUAL_COARSE_UPSAMPLE}")
        print(
            f"Reversal config: aux_target={REV_AUX_TARGET} | sample_weight_mode={REV_SAMPLE_WEIGHT_MODE} | "
            f"w_firstrev_local={W_FIRSTREV_LOCAL:.4f} | firstrev_radius={FIRSTREV_LOCAL_RADIUS}"
        )
        print(
            f"Optimizer config: optimizer={OPTIMIZER_NAME} | lr={LR:.6g} | weight_decay={WEIGHT_DECAY:.6g} | "
            f"scheduler={SCHEDULER_NAME} | warmup_epochs={WARMUP_EPOCHS} | grad_clip_norm={GRAD_CLIP_NORM:.4f}"
        )

        # ---- persist run config (for reproducibility) ----
        run_config = {
            "MODEL_VER": "v5_8_response_state_v1_protocol_safe",
            "protocol_config_path": str(PROTOCOL_CONFIG_PATH),
            "protocol_version": protocol_config.get("protocol_version"),
            "split_policy_expected": "subject-level fixed split",
            "split_policy_applied": "subject-level fixed split",
            "split_source": str(FROZEN_SPLIT_PATH),
            "train_subjects": list(split_subjects["train"]),
            "val_subjects": list(split_subjects["val"]),
            "test_subjects": list(split_subjects["test"]),
            "train_subject_count": int(len(split_subjects["train"])),
            "val_subject_count": int(len(split_subjects["val"])),
            "test_subject_count": int(len(split_subjects["test"])),
            "train_sample_count": int(len(train_idx)),
            "val_sample_count": int(len(val_idx)),
            "test_sample_count": int(len(test_idx)),
            "smoke_mode": bool(SMOKE_MODE),
            "smoke_sampling_policy": smoke_sampling_policy,
            "teacher_state_fit_split": "train",
            "teacher_state_fit_sample_count": int(len(train_idx)),
            "standardization_fit_split": "train",
            "curve_threshold_fit_split": "train",
            "anchor_source_expected": protocol_config.get("anchor_source"),
            "anchor_source_applied": "curve->roll_peak; straight->steer_rate_peak80_first",
            "maintained_anchor_policy": "curve->roll_peak; straight->steer_rate_peak80_first",
            "ENABLE_RESPONSE_STATE_V1": bool(ENABLE_RESPONSE_STATE_V1),
            "ENABLE_STATE_DISTILL": bool(ENABLE_STATE_DISTILL),
            "ENABLE_REVERSAL_AUX": bool(ENABLE_REVERSAL_AUX),
            "ENABLE_PEAKTIME_AUX": bool(ENABLE_PEAKTIME_AUX),
            "ENABLE_PEAKINTENSITY_AUX": bool(ENABLE_PEAKINTENSITY_AUX),
            "TEACHER_STATE_MODE": TEACHER_STATE_MODE,
            "TEACHER_STATE_DIM": int(TEACHER_STATE_DIM),
            "ACTUAL_STATE_DIM": int(state_dim),
            "TEACHER_STATE_COMPONENTS": teacher_state_meta["component_names"],
            "LAMBDA_REV": float(LAMBDA_REV),
            "REV_AUX_TARGET": REV_AUX_TARGET,
            "REV_SAMPLE_WEIGHT_MODE": REV_SAMPLE_WEIGHT_MODE,
            "USE_STRONG_REV_LOSS": bool(USE_STRONG_REV_LOSS),
            "REV_EPS": float(REV_EPS),
            "REV_HYBRID_WEAK_COEF": float(REV_HYBRID_WEAK_COEF),
            "REV_HYBRID_STRONG_COEF": float(REV_HYBRID_STRONG_COEF),
            "REV_BRIDGE_MODE": REV_BRIDGE_MODE,
            "STEER_SOURCE_UNIT": STEER_SOURCE_UNIT,
            "STEER_ANGLE_UNIT": STEER_ANGLE_UNIT,
            "STEER_ANGLE_SCALE": float(STEER_ANGLE_SCALE),
            "STEER_PLOT_UNIT": STEER_PLOT_UNIT,
            "STEER_PLOT_SCALE": float(STEER_PLOT_SCALE),
            "STEER_PLOT_FROM_TARGET_SCALE": float(STEER_PLOT_FROM_TARGET_SCALE),
            "STEER_ONSET_THR_ABS": float(STEER_ONSET_THR_ABS),
            "INPUT_PIPELINE_VERSION": INPUT_PIPELINE_VERSION,
            "USE_PEDALS": bool(USE_PEDALS),
            "USE_VY": bool(USE_VY),
            "USE_VROLL": bool(USE_VROLL),
            "USE_MU": bool(USE_MU),
            "USE_Z": bool(USE_Z),
            "USE_IS_CURVE_CTX": bool(USE_IS_CURVE_CTX),
            "SPEED_SOURCE_POLICY": speed_source_policy,
            "INPUT_QC_SELECTED_FEATURE_COLUMNS_PATH": str(selected_feature_columns_path),
            "INPUT_QC_SPEED_SOURCE_REPORT_PATH": str(speed_source_report_path),
            "INPUT_QC_SPEED_SOURCE_COUNTS": input_qc_summary.get("speed_source_counts", {}),
            "INPUT_QC_SPEED_WARNING_FILE_COUNT": int(len(input_qc_summary.get("speed_warning_vehicle_files", []))),
            "ROOT": ROOT,
            "STYLE_CSV": STYLE_CSV,
            "FS": FS,
            "WIN_SEC": WIN_SEC,
            "FUTURE_SEC": FUTURE_SEC,
            "WIN_LEN": WIN_LEN,
            "FUTURE_LEN": FUTURE_LEN,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "LR": LR,
            "OPTIMIZER": OPTIMIZER_NAME,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "SCHEDULER": SCHEDULER_NAME,
            "WARMUP_EPOCHS": WARMUP_EPOCHS,
            "GRAD_CLIP_NORM": GRAD_CLIP_NORM,
            "DEVICE": DEVICE,
            "D_MODEL": D_MODEL,
            "N_HEAD": N_HEAD,
            "ENC_LAYERS": NUM_LAYERS_ENC,
            "DEC_LAYERS": NUM_LAYERS_DEC,
            "FFN_DIM": FFN_DIM,
            "DROPOUT": DROPOUT,
            "W_DIFF1": W_DIFF1,
            "W_DIFF2": W_DIFF2,
            "W_REVSEQ": W_REVSEQ,
            "W_PEAKTIME": W_PEAKTIME,
            "REVSEQ_ALPHA_FRAC": REVSEQ_ALPHA_FRAC,
            "PEAK_TEMP_FRAC": PEAK_TEMP_FRAC,
            "W_STEER_WT": W_STEER_WT,
            "W_STEER_RATE": W_STEER_RATE,
            "W_STEER_REV": W_STEER_REV,
            "W_FIRSTREV_LOCAL": W_FIRSTREV_LOCAL,
            "FIRSTREV_LOCAL_RADIUS": FIRSTREV_LOCAL_RADIUS,
            "STEER_WT_MAX": STEER_WT_MAX,
            "W_TREND": W_TREND,
            "TREND_POOL_KERNEL": TREND_POOL_KERNEL,
            "TREND_POOL_STRIDE": TREND_POOL_STRIDE,
            "TREND_SIGN_EPS": TREND_SIGN_EPS,
            "TREND_LOSS_MODE": TREND_LOSS_MODE,
            "TREND_LEVEL_WEIGHT": TREND_LEVEL_WEIGHT,
            "TREND_DELTA_WEIGHT": TREND_DELTA_WEIGHT,
            "TREND_DIR_WEIGHT": TREND_DIR_WEIGHT,
            "ENABLE_STEER_COARSE_FINE": bool(ENABLE_STEER_COARSE_FINE),
            "ENABLE_MANUAL_COARSE_UPSAMPLE": bool(ENABLE_MANUAL_COARSE_UPSAMPLE),
            "ENABLE_LATE_RESIDUAL_HEAD": bool(ENABLE_LATE_RESIDUAL_HEAD),
            "LATE_RESIDUAL_START_SEC": LATE_RESIDUAL_START_SEC,
            "W_LATE_RESIDUAL": W_LATE_RESIDUAL,
            "W_TREND_COARSE": W_TREND_COARSE,
            "W_FINE_DC": W_FINE_DC,
            "ENABLE_PHASE_ADAPTIVE_TREND": bool(ENABLE_PHASE_ADAPTIVE_TREND),
            "TREND_EARLY_BINS": TREND_EARLY_BINS,
            "TREND_LATE_STRAIGHT_DOWN": TREND_LATE_STRAIGHT_DOWN,
            "TREND_LATE_STRONGREV_DOWN": TREND_LATE_STRONGREV_DOWN,
            "ENABLE_LATE_REV_GATE": bool(ENABLE_LATE_REV_GATE),
            "LATE_REV_GATE_START_SEC": LATE_REV_GATE_START_SEC,
            "LATE_REV_GATE_SCALE": LATE_REV_GATE_SCALE,
            "LATE_REV_GATE_RAMP_POWER": LATE_REV_GATE_RAMP_POWER,
            "ENABLE_STRONG_POS_GATE": bool(ENABLE_STRONG_POS_GATE),
            "STRONG_POS_GATE_START_SEC": STRONG_POS_GATE_START_SEC,
            "STRONG_POS_GATE_SCALE": STRONG_POS_GATE_SCALE,
            "STRONG_POS_GATE_RAMP_POWER": STRONG_POS_GATE_RAMP_POWER,
            "STRONG_POS_GATE_PROB_CENTER": STRONG_POS_GATE_PROB_CENTER,
            "ENABLE_HARD_LATE_FINE": bool(ENABLE_HARD_LATE_FINE),
            "W_HARD_LATE_FINE": W_HARD_LATE_FINE,
            "HARD_LATE_START_SEC": HARD_LATE_START_SEC,
            "HARD_TAIL_START_SEC": HARD_TAIL_START_SEC,
            "HARD_PEAK_QUANTILE": HARD_PEAK_QUANTILE,
            "HARD_TAIL_QUANTILE": HARD_TAIL_QUANTILE,
            "LAMBDA_STATE": LAMBDA_STATE,
            "LAMBDA_STRONG_POS_GATE": LAMBDA_STRONG_POS_GATE,
            "EEG_HIST_SEC": EEG_HIST_SEC,
            "SEED": SEED,
            "N_TRAIN": int(len(train_dataset)),
            "N_VAL": int(len(val_dataset)),
            "N_TEST": int(len(test_dataset)),
            "split_audit_path": str(RUN_DIR / "split_audit.json"),
            "split_sample_counts_path": str(RUN_DIR / "split_sample_counts.csv"),
            "val_structured_history_path": str(RUN_DIR / "val_structured_history.csv"),
            "best_checkpoint_by_loss_path": str(CKPT_DIR / "best_model_v5_8_by_loss.pth"),
            "best_checkpoint_by_structured_path": str(CKPT_DIR / "best_model_v5_8_by_structured.pth"),
        }
        save_json(RUN_DIR / "run_config.json", run_config)
    
        # ---- training history ----
        history = []
        history_csv = RUN_DIR / "loss_history.csv"
        structured_history = []
        structured_history_csv = RUN_DIR / "val_structured_history.csv"

        best_val = np.inf
        best_structured_score = np.inf
        start_all = time.time()
    
        for epoch in range(1, EPOCHS + 1):
            current_lr = compute_epoch_lr(
                epoch=epoch,
                total_epochs=EPOCHS,
                base_lr=LR,
                scheduler_name=SCHEDULER_NAME,
                warmup_epochs=WARMUP_EPOCHS,
            )
            set_optimizer_lr(optim, current_lr)
            model.train()
            epoch_weak_coef, epoch_strong_coef = resolve_reversal_weight_blend(epoch, EPOCHS)
            loss_sum, loss_task_sum, loss_state_sum, loss_rev_sum, loss_trend_sum, loss_trend_coarse_sum, loss_fine_dc_sum, loss_hard_late_sum, loss_late_residual_sum, loss_firstrev_local_sum, loss_strong_pos_gate_sum, n_batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
            grad_norm_preclip_sum = 0.0
            grad_norm_batches = 0
    
            for batch in train_loader:
                src = batch["src"].to(DEVICE, non_blocking=True)
                y_true = batch["y_norm"].to(DEVICE, non_blocking=True)
                curve_norm = batch["curve_norm"].to(DEVICE, non_blocking=True)
                ctx = batch["ctx"].to(DEVICE, non_blocking=True)
                z_phys_b = batch["z_phys"].to(DEVICE, non_blocking=True)
                z_mask = batch["z_mask"].to(DEVICE, non_blocking=True)  # (B,1)
                is_curve_b = batch["is_curve"].to(DEVICE, non_blocking=True)
                rev_gt_b = batch["rev_gt"].to(DEVICE, non_blocking=True).squeeze(1)  # (B,)
                rev_gt_weak_b = batch.get("rev_gt_weak", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)
                rev_gt_strong_b = batch.get("rev_gt_strong", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)
    
                optim.zero_grad()
                y_hat, z_veh, rev_logit, forward_aux = unpack_model_output(model(src, ctx, curve_norm))

                rev_aux_target_b = get_rev_aux_target(rev_gt_weak_b, rev_gt_strong_b)
                sample_weight = build_reversal_sample_weight(
                    rev_gt_b,
                    rev_gt_weak=rev_gt_weak_b,
                    rev_gt_strong=rev_gt_strong_b,
                    weak_coef=epoch_weak_coef,
                    strong_coef=epoch_strong_coef,
                )
                loss_task, loss_amp, loss_d1, loss_d2, loss_revseq, loss_peaktime, loss_steer_wt, loss_trend, loss_trend_coarse, loss_fine_dc, loss_hard_late_fine, loss_late_residual, loss_firstrev_local = compute_total_task_loss(
                    y_hat, y_true, y_mean_t, y_std_t, sample_weight=sample_weight, use_reversal_local_weight=True, forward_aux=forward_aux,
                    is_curve=is_curve_b, rev_gt_weak=rev_gt_weak_b, rev_gt_strong=rev_gt_strong_b
                )

                # train 侧也使用 GT soft reversal 作为局部加权依据，避免 hard case 被均值化
                # state loss supports missing physio: if z_mask=0, ignore
                mse_state = ((z_veh - z_phys_b) ** 2).mean(dim=1, keepdim=True)  # (B,1)
                loss_state = (mse_state * z_mask).sum() / (z_mask.sum() + EPS)
    
                # reversal aux loss (binary): whether future steer changes sign
                if ENABLE_REVERSAL_AUX:
                    loss_rev = F.binary_cross_entropy_with_logits(rev_logit, rev_aux_target_b.float(), pos_weight=rev_pos_weight)
                else:
                    loss_rev = torch.tensor(0.0, device=DEVICE)
                strong_pos_gate_logit = forward_aux.get("strong_pos_gate_logit")
                if ENABLE_STRONG_POS_GATE and strong_pos_gate_logit is not None:
                    loss_strong_pos_gate = F.binary_cross_entropy_with_logits(
                        strong_pos_gate_logit,
                        rev_gt_strong_b.float(),
                        pos_weight=strong_pos_gate_pos_weight,
                    )
                else:
                    loss_strong_pos_gate = torch.tensor(0.0, device=DEVICE)

                loss = loss_task + LAMBDA_STATE * loss_state + LAMBDA_REV * loss_rev + LAMBDA_STRONG_POS_GATE * loss_strong_pos_gate
                loss.backward()
                if GRAD_CLIP_NORM > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    if torch.is_tensor(grad_norm):
                        grad_norm = float(grad_norm.detach().cpu().item())
                    else:
                        grad_norm = float(grad_norm)
                    grad_norm_preclip_sum += grad_norm
                    grad_norm_batches += 1
                optim.step()
    
                loss_sum += float(loss.item())
                loss_task_sum += float(loss_task.item())
                loss_state_sum += float(loss_state.item())
                loss_rev_sum += float(loss_rev.item())
                loss_trend_sum += float(loss_trend.item())
                loss_trend_coarse_sum += float(loss_trend_coarse.item())
                loss_fine_dc_sum += float(loss_fine_dc.item())
                loss_hard_late_sum += float(loss_hard_late_fine.item())
                loss_late_residual_sum += float(loss_late_residual.item())
                loss_firstrev_local_sum += float(loss_firstrev_local.item())
                loss_strong_pos_gate_sum += float(loss_strong_pos_gate.item())
                n_batch += 1
    
            train_loss = loss_sum / max(1, n_batch)
            train_loss_rev = loss_rev_sum / max(1, n_batch)
            grad_norm_preclip_mean = grad_norm_preclip_sum / max(1, grad_norm_batches) if GRAD_CLIP_NORM > 0 else 0.0
    
            # val
            model.eval()
            val_sum, val_trend_sum, val_trend_coarse_sum, val_fine_dc_sum, val_hard_late_sum, val_late_residual_sum, val_firstrev_local_sum, val_strong_pos_gate_sum, val_n = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    src = batch["src"].to(DEVICE, non_blocking=True)
                    y_true = batch["y_norm"].to(DEVICE, non_blocking=True)
                    curve_norm = batch["curve_norm"].to(DEVICE, non_blocking=True)
                    ctx = batch["ctx"].to(DEVICE, non_blocking=True)
                    z_phys_b = batch["z_phys"].to(DEVICE, non_blocking=True)
                    z_mask = batch["z_mask"].to(DEVICE, non_blocking=True)
                    is_curve_b = batch["is_curve"].to(DEVICE, non_blocking=True)
                    rev_gt_b = batch["rev_gt"].to(DEVICE, non_blocking=True).squeeze(1)
                    rev_gt_weak_b = batch.get("rev_gt_weak", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)
                    rev_gt_strong_b = batch.get("rev_gt_strong", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)

                    y_hat, z_veh, rev_logit, forward_aux = unpack_model_output(model(src, ctx, curve_norm))
                    rev_aux_target_b = get_rev_aux_target(rev_gt_weak_b, rev_gt_strong_b)
                    sample_weight = build_reversal_sample_weight(
                        rev_gt_b,
                        rev_gt_weak=rev_gt_weak_b,
                        rev_gt_strong=rev_gt_strong_b,
                        weak_coef=epoch_weak_coef,
                        strong_coef=epoch_strong_coef,
                    )
                    loss_task, loss_amp, loss_d1, loss_d2, loss_revseq, loss_peaktime, loss_steer_wt, loss_trend, loss_trend_coarse, loss_fine_dc, loss_hard_late_fine, loss_late_residual, loss_firstrev_local = compute_total_task_loss(
                        y_hat, y_true, y_mean_t, y_std_t, sample_weight=sample_weight, use_reversal_local_weight=True, forward_aux=forward_aux,
                        is_curve=is_curve_b, rev_gt_weak=rev_gt_weak_b, rev_gt_strong=rev_gt_strong_b
                    )

                    # val 侧与 train 使用同一套加权目标，避免选择标准错位
                    mse_state = ((z_veh - z_phys_b) ** 2).mean(dim=1, keepdim=True)
                    loss_state = (mse_state * z_mask).sum() / (z_mask.sum() + EPS)
                    # reversal aux loss (binary): whether future steer changes sign
                    if ENABLE_REVERSAL_AUX:
                        loss_rev = F.binary_cross_entropy_with_logits(rev_logit, rev_aux_target_b.float(), pos_weight=rev_pos_weight)
                    else:
                        loss_rev = torch.tensor(0.0, device=DEVICE)
                    strong_pos_gate_logit = forward_aux.get("strong_pos_gate_logit")
                    if ENABLE_STRONG_POS_GATE and strong_pos_gate_logit is not None:
                        loss_strong_pos_gate = F.binary_cross_entropy_with_logits(
                            strong_pos_gate_logit,
                            rev_gt_strong_b.float(),
                            pos_weight=strong_pos_gate_pos_weight,
                        )
                    else:
                        loss_strong_pos_gate = torch.tensor(0.0, device=DEVICE)
                    loss = loss_task + LAMBDA_STATE * loss_state + LAMBDA_REV * loss_rev + LAMBDA_STRONG_POS_GATE * loss_strong_pos_gate

                    val_sum += float(loss.item())
                    val_trend_sum += float(loss_trend.item())
                    val_trend_coarse_sum += float(loss_trend_coarse.item())
                    val_fine_dc_sum += float(loss_fine_dc.item())
                    val_hard_late_sum += float(loss_hard_late_fine.item())
                    val_late_residual_sum += float(loss_late_residual.item())
                    val_firstrev_local_sum += float(loss_firstrev_local.item())
                    val_strong_pos_gate_sum += float(loss_strong_pos_gate.item())
                    val_n += 1
            val_loss = val_sum / max(1, val_n)
            val_structured = collect_structured_metrics_from_loader(
                model,
                val_loader,
                y_mean,
                y_std,
                fs=FS,
            )
            val_structured_rmse = _score_value_or_default(val_structured["rmse_steer"], float("nan"))
            val_structured_tail = _score_value_or_default(val_structured["tail_rmse_steer"], float("nan"))
            val_structured_late_peak = _score_value_or_default(val_structured["late_peak_recall"], float("nan"))
            val_structured_firstrev = _score_value_or_default(val_structured["first_reversal_time_mae_sec"], float("nan"))
            val_structured_rev_match = _score_value_or_default(val_structured["reversal_count_exact_match_rate"], float("nan"))

            blend_msg = (
                "rev_blend=n/a"
                if epoch_weak_coef is None or epoch_strong_coef is None
                else f"rev_blend=({epoch_weak_coef:.3f},{epoch_strong_coef:.3f})"
            )
            print(f"[Epoch {epoch:02d}/{EPOCHS:02d}] "
                  f"LR={current_lr:.6g} | "
                  f"Train={train_loss:.6f} (task={loss_task_sum/max(1,n_batch):.6f}, trend={loss_trend_sum/max(1,n_batch):.6f}, trend_cf={loss_trend_coarse_sum/max(1,n_batch):.6f}, fine_dc={loss_fine_dc_sum/max(1,n_batch):.6f}, hard_late={loss_hard_late_sum/max(1,n_batch):.6f}, late_res={loss_late_residual_sum/max(1,n_batch):.6f}, firstrev={loss_firstrev_local_sum/max(1,n_batch):.6f}, strong_gate={loss_strong_pos_gate_sum/max(1,n_batch):.6f}, state={loss_state_sum/max(1,n_batch):.6f}) | "
                  f"Val={val_loss:.6f} (trend={val_trend_sum/max(1,val_n):.6f}, trend_cf={val_trend_coarse_sum/max(1,val_n):.6f}, fine_dc={val_fine_dc_sum/max(1,val_n):.6f}, hard_late={val_hard_late_sum/max(1,val_n):.6f}, late_res={val_late_residual_sum/max(1,val_n):.6f}, firstrev={val_firstrev_local_sum/max(1,val_n):.6f}, strong_gate={val_strong_pos_gate_sum/max(1,val_n):.6f}) | "
                  f"ValStruct(score={val_structured['structured_score']:.6f}, rmse={val_structured_rmse:.6f}, tail={val_structured_tail:.6f}, late_peak={val_structured_late_peak:.6f}, firstrev={val_structured_firstrev:.6f}, rev_match={val_structured_rev_match:.6f}) | "
                  f"{blend_msg}")
    
            # ---- write history (CSV) ----
            task_avg = float(loss_task_sum / max(1, n_batch))
            trend_avg = float(loss_trend_sum / max(1, n_batch))
            trend_coarse_avg = float(loss_trend_coarse_sum / max(1, n_batch))
            fine_dc_avg = float(loss_fine_dc_sum / max(1, n_batch))
            hard_late_avg = float(loss_hard_late_sum / max(1, n_batch))
            late_residual_avg = float(loss_late_residual_sum / max(1, n_batch))
            firstrev_local_avg = float(loss_firstrev_local_sum / max(1, n_batch))
            strong_pos_gate_avg = float(loss_strong_pos_gate_sum / max(1, n_batch))
            state_avg = float(loss_state_sum / max(1, n_batch))
            history.append({
                "epoch": int(epoch),
                "lr": float(current_lr),
                "grad_clip_norm_applied": float(GRAD_CLIP_NORM),
                "train_grad_norm_preclip_mean": float(grad_norm_preclip_mean),
                "train_loss": float(train_loss),
                "train_task": task_avg,
                "train_trend": trend_avg,
                "train_trend_coarse": trend_coarse_avg,
                "train_fine_dc": fine_dc_avg,
                "train_hard_late": hard_late_avg,
                "train_late_residual": late_residual_avg,
                "train_firstrev_local": firstrev_local_avg,
                "train_strong_pos_gate": strong_pos_gate_avg,
                "train_state": state_avg,
                "rev_hybrid_weak_coef": epoch_weak_coef,
                "rev_hybrid_strong_coef": epoch_strong_coef,
                "rev_bridge_mode": REV_BRIDGE_MODE,
                "val_loss": float(val_loss),
                "val_trend": float(val_trend_sum / max(1, val_n)),
                "val_trend_coarse": float(val_trend_coarse_sum / max(1, val_n)),
                "val_fine_dc": float(val_fine_dc_sum / max(1, val_n)),
                "val_hard_late": float(val_hard_late_sum / max(1, val_n)),
                "val_late_residual": float(val_late_residual_sum / max(1, val_n)),
                "val_firstrev_local": float(val_firstrev_local_sum / max(1, val_n)),
                "val_strong_pos_gate": float(val_strong_pos_gate_sum / max(1, val_n)),
                "val_structured_score": float(val_structured["structured_score"]),
                "val_structured_rmse_steer": val_structured_rmse,
                "val_structured_tail_rmse_steer": val_structured_tail,
                "val_structured_late_peak_recall": val_structured_late_peak,
                "val_structured_first_reversal_time_mae_sec": val_structured_firstrev,
                "val_structured_reversal_count_exact_match_rate": val_structured_rev_match,
            })
            structured_history.append({
                "epoch": int(epoch),
                "lr": float(current_lr),
                "rev_hybrid_weak_coef": epoch_weak_coef,
                "rev_hybrid_strong_coef": epoch_strong_coef,
                "rev_bridge_mode": REV_BRIDGE_MODE,
                **val_structured,
            })
            try:
                pd.DataFrame(history).to_csv(str(history_csv), index=False, encoding="utf-8-sig")
                pd.DataFrame(structured_history).to_csv(str(structured_history_csv), index=False, encoding="utf-8-sig")
            except Exception:
                pass
    
            if val_loss < best_val:
                best_val = val_loss
                best_path = CKPT_DIR / "best_model_v5_8_protocol_safe.pth"
                best_loss_path = CKPT_DIR / "best_model_v5_8_by_loss.pth"
                torch.save(model.state_dict(), str(best_path))
                torch.save(model.state_dict(), str(best_loss_path))
                print(f"  🌟 New best by loss -> {best_path}")
                print(f"  🌟 Synced best_by_loss -> {best_loss_path}\n")
            if float(val_structured["structured_score"]) < best_structured_score:
                best_structured_score = float(val_structured["structured_score"])
                best_structured_path = CKPT_DIR / "best_model_v5_8_by_structured.pth"
                torch.save(model.state_dict(), str(best_structured_path))
                print(
                    "  🌟 New best by structured -> "
                    f"{best_structured_path} | score={best_structured_score:.6f} | "
                    f"late_peak={val_structured_late_peak:.6f} | "
                    f"firstrev={val_structured_firstrev:.6f}\n"
                )
    
        print(f"\n⌛ 总训练耗时: {(time.time()-start_all)/60:.2f} min\n")
    
        # ---- save checkpoint with norms ----
        ckpt = {
            "state_dict": model.state_dict(),
            "feature_names": feature_names,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "curve_mean": curve_mean,
            "curve_std": curve_std,
            "ctx_mean": ctx_mean,
            "ctx_std": ctx_std,
            "teacher_base_mu": base_mu,
            "teacher_base_sd": base_sd,
            "teacher_z_mu": z_mu,
            "teacher_z_sd": z_sd,
            "config": {
                "MODEL_VER": "v5_8_response_state_v1_protocol_safe",
                "WIN_SEC": WIN_SEC,
                "FUTURE_SEC": FUTURE_SEC,
                "WIN_LEN": WIN_LEN,
                "FUTURE_LEN": FUTURE_LEN,
                "BATCH_SIZE": BATCH_SIZE,
                "EPOCHS": EPOCHS,
                "LR": LR,
                "OPTIMIZER": OPTIMIZER_NAME,
                "WEIGHT_DECAY": WEIGHT_DECAY,
                "SCHEDULER": SCHEDULER_NAME,
                "WARMUP_EPOCHS": WARMUP_EPOCHS,
                "GRAD_CLIP_NORM": GRAD_CLIP_NORM,
                "D_MODEL": D_MODEL,
                "N_HEAD": N_HEAD,
                "ENC_LAYERS": NUM_LAYERS_ENC,
                "DEC_LAYERS": NUM_LAYERS_DEC,
                "FFN_DIM": FFN_DIM,
                "DROPOUT": DROPOUT,
                "W_DIFF1": W_DIFF1,
                "W_DIFF2": W_DIFF2,
                "W_REVSEQ": W_REVSEQ,
                "W_PEAKTIME": W_PEAKTIME,
                "REVSEQ_ALPHA_FRAC": REVSEQ_ALPHA_FRAC,
                "PEAK_TEMP_FRAC": PEAK_TEMP_FRAC,
                "REV_AUX_TARGET": REV_AUX_TARGET,
                "REV_SAMPLE_WEIGHT_MODE": REV_SAMPLE_WEIGHT_MODE,
                "USE_STRONG_REV_LOSS": USE_STRONG_REV_LOSS,
                "W_STEER_WT": W_STEER_WT,
                "W_STEER_RATE": W_STEER_RATE,
                "W_STEER_REV": W_STEER_REV,
                "W_FIRSTREV_LOCAL": W_FIRSTREV_LOCAL,
                "FIRSTREV_LOCAL_RADIUS": FIRSTREV_LOCAL_RADIUS,
                "STEER_WT_MAX": STEER_WT_MAX,
                "W_TREND": W_TREND,
                "TREND_POOL_KERNEL": TREND_POOL_KERNEL,
                "TREND_POOL_STRIDE": TREND_POOL_STRIDE,
                "TREND_SIGN_EPS": TREND_SIGN_EPS,
                "STEER_SOURCE_UNIT": STEER_SOURCE_UNIT,
                "STEER_ANGLE_UNIT": STEER_ANGLE_UNIT,
                "STEER_ANGLE_SCALE": float(STEER_ANGLE_SCALE),
                "STEER_PLOT_UNIT": STEER_PLOT_UNIT,
                "STEER_PLOT_SCALE": float(STEER_PLOT_SCALE),
                "STEER_PLOT_FROM_TARGET_SCALE": float(STEER_PLOT_FROM_TARGET_SCALE),
                "STEER_ONSET_THR_ABS": float(STEER_ONSET_THR_ABS),
                "TREND_LOSS_MODE": TREND_LOSS_MODE,
                "TREND_LEVEL_WEIGHT": TREND_LEVEL_WEIGHT,
                "TREND_DELTA_WEIGHT": TREND_DELTA_WEIGHT,
                "TREND_DIR_WEIGHT": TREND_DIR_WEIGHT,
                "ENABLE_STEER_COARSE_FINE": ENABLE_STEER_COARSE_FINE,
                "ENABLE_MANUAL_COARSE_UPSAMPLE": ENABLE_MANUAL_COARSE_UPSAMPLE,
                "ENABLE_LATE_RESIDUAL_HEAD": ENABLE_LATE_RESIDUAL_HEAD,
                "LATE_RESIDUAL_START_SEC": LATE_RESIDUAL_START_SEC,
                "W_LATE_RESIDUAL": W_LATE_RESIDUAL,
                "W_TREND_COARSE": W_TREND_COARSE,
                "W_FINE_DC": W_FINE_DC,
                "ENABLE_PHASE_ADAPTIVE_TREND": ENABLE_PHASE_ADAPTIVE_TREND,
                "TREND_EARLY_BINS": TREND_EARLY_BINS,
                "TREND_LATE_STRAIGHT_DOWN": TREND_LATE_STRAIGHT_DOWN,
                "TREND_LATE_STRONGREV_DOWN": TREND_LATE_STRONGREV_DOWN,
                "ENABLE_LATE_REV_GATE": ENABLE_LATE_REV_GATE,
                "LATE_REV_GATE_START_SEC": LATE_REV_GATE_START_SEC,
                "LATE_REV_GATE_SCALE": LATE_REV_GATE_SCALE,
                "LATE_REV_GATE_RAMP_POWER": LATE_REV_GATE_RAMP_POWER,
                "ENABLE_STRONG_POS_GATE": ENABLE_STRONG_POS_GATE,
                "STRONG_POS_GATE_START_SEC": STRONG_POS_GATE_START_SEC,
                "STRONG_POS_GATE_SCALE": STRONG_POS_GATE_SCALE,
                "STRONG_POS_GATE_RAMP_POWER": STRONG_POS_GATE_RAMP_POWER,
                "STRONG_POS_GATE_PROB_CENTER": STRONG_POS_GATE_PROB_CENTER,
                "ENABLE_HARD_LATE_FINE": ENABLE_HARD_LATE_FINE,
                "W_HARD_LATE_FINE": W_HARD_LATE_FINE,
                "HARD_LATE_START_SEC": HARD_LATE_START_SEC,
                "HARD_TAIL_START_SEC": HARD_TAIL_START_SEC,
                "HARD_PEAK_QUANTILE": HARD_PEAK_QUANTILE,
                "HARD_TAIL_QUANTILE": HARD_TAIL_QUANTILE,
                "LAMBDA_STATE": LAMBDA_STATE,
                "LAMBDA_REV": LAMBDA_REV,
                "LAMBDA_STRONG_POS_GATE": LAMBDA_STRONG_POS_GATE,
                "W_AMP": W_AMP,
                "ENABLE_RESPONSE_STATE_V1": ENABLE_RESPONSE_STATE_V1,
                "ENABLE_STATE_DISTILL": ENABLE_STATE_DISTILL,
                "ENABLE_REVERSAL_AUX": ENABLE_REVERSAL_AUX,
                "ENABLE_PEAKTIME_AUX": ENABLE_PEAKTIME_AUX,
                "ENABLE_PEAKINTENSITY_AUX": ENABLE_PEAKINTENSITY_AUX,
                "EEG_HIST_SEC": EEG_HIST_SEC,
            }
        }
        ckpt_path = CKPT_DIR / "model_rollpeak_transformer_v5_8_protocol_safe.pth"
        torch.save(ckpt, str(ckpt_path))
        print(f"💾 已保存 checkpoint: {ckpt_path}\n")
    
        # ---- save training curves ----
        try:
            df_h = pd.DataFrame(history)
            if len(df_h) > 0:
                plt.figure()
                plt.plot(df_h["epoch"], df_h["train_loss"], label="train")
                plt.plot(df_h["epoch"], df_h["val_loss"], label="val")
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fig_path = FIG_DIR / "loss_curve.png"
                plt.savefig(str(fig_path), dpi=200)
                plt.close()
                print(f"📈 已保存曲线: {fig_path}\n")
        except Exception:
            pass
    

        # ---- export test prediction plots + state inspection ----
        try:
            best_path = CKPT_DIR / "best_model_v5_8_protocol_safe.pth"
            if best_path.exists():
                model.load_state_dict(torch.load(str(best_path), map_location=DEVICE))
                print("✅ 已加载 best 权重用于评估画图:", best_path)
            evaluate_and_plot(
                model,
                test_loader,
                y_mean,
                y_std,
                FIG_DIR,
                curve_thr=curve_thr,
                fs=FS,
                n_examples=8,
                state_component_names=teacher_state_meta["component_names"],
                teacher_state_mode=teacher_state_meta["mode"],
            )
        except Exception as e:
            print("⚠ 评估画图阶段失败:", repr(e))

        print("✅ 本次运行已完成。")
    
    finally:
        # restore stdout & close file handle
        sys.stdout = orig_stdout
        try:
            tee.close()
        except Exception:
            pass
