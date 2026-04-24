from .shared import *
from .data import make_state_column_names, summarize_state_vector
from .modeling import unpack_model_output
from .metrics import (
    _binary_metrics,
    _denorm_y,
    _head_metrics,
    _peak_metrics,
    _structured_reversal_metrics,
    _tail_metrics,
    _trend_metrics,
)

def evaluate_and_plot(model: nn.Module, test_loader: DataLoader,
                      y_mean: np.ndarray, y_std: np.ndarray,
                      fig_dir: Path, curve_thr: float = None, fs: int = 200, n_examples: int = 8,
                      state_component_names=None, teacher_state_mode: str = "old_ac"):
    """Export:
      - figures/pred_vs_gt_example_*.png
      - figures/test_metrics.json
      - figures/test_state_dump.csv (A/C from veh & teacher + mask)
      - figures/state_vs_peak_*.png (quick relationship views)
    """
    model.eval()

    preds, trues = [], []
    zveh_all, zphys_all, zmask_all = [], [], []
    idx_all, curve_score_all, is_curve_all = [], [], []
    rev_gt_all, rev_gt_weak_all, rev_gt_strong_all, rev_prob_all = [], [], [], []
    strong_pos_gate_prob_all = []
    late_residual_norm_all = []
    late_residual_mask_ref = None

    with torch.no_grad():
        for batch in test_loader:
            src = batch["src"].to(DEVICE, non_blocking=True)
            y_true_norm = batch["y_norm"].to(DEVICE, non_blocking=True)
            curve_norm = batch["curve_norm"].to(DEVICE, non_blocking=True)
            ctx = batch["ctx"].to(DEVICE, non_blocking=True)
            z_phys = batch["z_phys"].to(DEVICE, non_blocking=True)
            z_mask = batch["z_mask"].to(DEVICE, non_blocking=True)  # (B,1)
            rev_gt_b = batch["rev_gt"].to(DEVICE, non_blocking=True).squeeze(1)  # (B,)
            rev_gt_weak_b = batch.get("rev_gt_weak", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)
            rev_gt_strong_b = batch.get("rev_gt_strong", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)

            y_hat_norm, z_veh, rev_logit, forward_aux = unpack_model_output(model(src, ctx, curve_norm))

            preds.append(y_hat_norm.cpu().numpy())
            trues.append(y_true_norm.cpu().numpy())
            zveh_all.append(z_veh.cpu().numpy())
            zphys_all.append(z_phys.cpu().numpy())
            zmask_all.append(z_mask.cpu().numpy())
            idx_all.append(batch.get("idx", torch.full((src.shape[0],), -1, dtype=torch.long)).cpu().numpy())
            rev_gt_all.append(rev_gt_b.detach().cpu().numpy())
            rev_gt_weak_all.append(rev_gt_weak_b.detach().cpu().numpy())
            rev_gt_strong_all.append(rev_gt_strong_b.detach().cpu().numpy())
            rev_prob_all.append(torch.sigmoid(rev_logit).detach().cpu().numpy())
            if forward_aux.get("strong_pos_gate_prob") is not None:
                strong_pos_gate_prob_all.append(forward_aux["strong_pos_gate_prob"].detach().cpu().numpy())
            if forward_aux.get("steer_late_residual_norm") is not None:
                late_residual_norm_all.append(forward_aux["steer_late_residual_norm"].detach().cpu().numpy())
                if late_residual_mask_ref is None and forward_aux.get("steer_late_residual_mask") is not None:
                    late_residual_mask_ref = forward_aux["steer_late_residual_mask"].detach().cpu().numpy()
            curve_score_all.append(batch.get("curve_score", torch.full((src.shape[0],), float('nan'))).cpu().numpy())
            is_curve_all.append(batch.get("is_curve", torch.full((src.shape[0],), -1, dtype=torch.long)).cpu().numpy())

    pred_norm = np.concatenate(preds, axis=0)
    true_norm = np.concatenate(trues, axis=0)
    zveh_all = np.concatenate(zveh_all, axis=0)
    zphys_all = np.concatenate(zphys_all, axis=0)
    zmask_all = np.concatenate(zmask_all, axis=0).reshape(-1)  # (N,)
    state_dim = int(zveh_all.shape[1]) if zveh_all.ndim == 2 else 0
    veh_state_cols = make_state_column_names("veh", state_dim, state_component_names)
    teacher_state_cols = make_state_column_names("teacher", state_dim, state_component_names)

    pred = _denorm_y(pred_norm, y_mean, y_std)
    true = _denorm_y(true_norm, y_mean, y_std)
    late_residual_den = None
    if len(late_residual_norm_all) > 0:
        late_residual_den = np.concatenate(late_residual_norm_all, axis=0).astype(np.float32) * float(y_std[0])

    err = pred - true
    rmse_all = float(np.sqrt(np.mean(err ** 2)))
    rmse_ch = np.sqrt(np.mean(err ** 2, axis=(0, 1))).astype(float)
    mae_ch = np.mean(np.abs(err), axis=(0, 1)).astype(float)

    metrics = {
        "rmse_all": rmse_all,
        "rmse_steer": float(rmse_ch[0]),
        "rmse_yawrate": float(rmse_ch[1]),
        "rmse_ay": float(rmse_ch[2]),
        "mae_steer": float(mae_ch[0]),
        "mae_yawrate": float(mae_ch[1]),
        "mae_ay": float(mae_ch[2]),
        "n_test": int(pred.shape[0]),
        "future_len": int(pred.shape[1]),
    }
    head_metrics = _head_metrics(pred, true, fs=fs)
    tail_metrics = _tail_metrics(pred, true, fs=fs)
    peak_metrics = _peak_metrics(pred, true, fs=fs)
    trend_metrics = _trend_metrics(pred, true, fs=fs)
    metrics.update({
        "head_metrics": head_metrics,
        "tail_metrics": tail_metrics,
        "peak_metrics": peak_metrics,
        "trend_metrics": trend_metrics,
    })
    save_json(fig_dir / "test_metrics.json", metrics)
    save_json(fig_dir / "test_metrics_head.json", head_metrics)
    save_json(fig_dir / "test_metrics_tail.json", tail_metrics)
    save_json(fig_dir / "test_metrics_peak.json", peak_metrics)
    save_json(fig_dir / "test_metrics_trend.json", trend_metrics)
    print("📌 Test 指标:", metrics)


    # ---- road-type metrics (curve vs straight) ----
    if is_curve_all is not None and len(is_curve_all) > 0:
        is_curve_vec = np.concatenate(is_curve_all, axis=0).astype(np.int64)
        curve_score_vec = np.concatenate(curve_score_all, axis=0).astype(np.float32)
        # is_curve: 1=curve, 0=straight
        mask_curve = (is_curve_vec == 1)
        mask_straight = (is_curve_vec == 0)

        def rmse_by_mask(err, mask):
            if mask is None or mask.sum() == 0:
                return None
            ee = err[mask, :, :]
            out = np.sqrt(np.mean(ee ** 2, axis=(0, 1)))
            return out.tolist()

        err = pred - true
        road_metrics = {
            "curve_thr": float(curve_thr) if curve_thr is not None else None,
            "curve_ratio_test": float(mask_curve.mean()) if mask_curve.size else None,
            "rmse_curve": rmse_by_mask(err, mask_curve),
            "rmse_straight": rmse_by_mask(err, mask_straight),
        }
        save_json(fig_dir / "test_metrics_by_roadtype.json", road_metrics)
        print("🛣 RoadType 指标:", road_metrics)

    # ---- reversal metrics (weak & strong; and the label actually used for training) ----
    rev_prob_vec = np.concatenate(rev_prob_all, axis=0).astype(np.float32) if len(rev_prob_all)>0 else None
    is_curve_vec = np.concatenate(is_curve_all, axis=0).astype(np.int64) if (is_curve_all is not None and len(is_curve_all)>0) else None

    rev_gt_used_vec = np.concatenate(rev_gt_all, axis=0).astype(np.int64) if len(rev_gt_all)>0 else None
    rev_gt_weak_vec = np.concatenate(rev_gt_weak_all, axis=0).astype(np.int64) if len(rev_gt_weak_all)>0 else None
    rev_gt_strong_vec = np.concatenate(rev_gt_strong_all, axis=0).astype(np.int64) if len(rev_gt_strong_all)>0 else None

    def _rmse_steer_mask(mask):
        if mask is None or mask.sum() == 0:
            return None
        ee = err[mask, :, 0]
        return float(np.sqrt(np.mean(ee ** 2)))

    def _compute_rev_metrics(label_vec):
        if label_vec is None or rev_prob_vec is None:
            return None
        pred_vec = (rev_prob_vec >= 0.5).astype(np.int64)
        met_all = _binary_metrics(label_vec, pred_vec)

        met_straight = None
        rmse_straight_pos = None
        rmse_straight_neg = None
        if is_curve_vec is not None:
            mask_straight = (is_curve_vec == 0)
            met_straight = _binary_metrics(label_vec[mask_straight], pred_vec[mask_straight])
            rmse_straight_pos = _rmse_steer_mask(mask_straight & (label_vec == 1))
            rmse_straight_neg = _rmse_steer_mask(mask_straight & (label_vec == 0))
        return {
            "metrics_all": met_all,
            "metrics_straight": met_straight,
            "rmse_steer_straight_pos": rmse_straight_pos,
            "rmse_steer_straight_neg": rmse_straight_neg,
        }

    structured_rev_metrics = _structured_reversal_metrics(
        pred,
        true,
        rev_gt_weak_vec=rev_gt_weak_vec,
        rev_gt_strong_vec=rev_gt_strong_vec,
        fs=fs,
    )
    rev_metrics = {
        "STEER_SOURCE_UNIT": STEER_SOURCE_UNIT,
        "STEER_ANGLE_UNIT": STEER_ANGLE_UNIT,
        "STEER_ANGLE_SCALE": float(STEER_ANGLE_SCALE),
        "REV_EPS_WEAK": float(REV_EPS_WEAK),
        "REV_EPS_STRONG": float(REV_EPS_STRONG),
        "STRONG_PEAK_THR": float(STRONG_PEAK_THR),
        "used_label": REV_AUX_TARGET,
        "rate_used": float(np.mean(rev_gt_used_vec)) if rev_gt_used_vec is not None else None,
        "rate_weak": float(np.mean(rev_gt_weak_vec)) if rev_gt_weak_vec is not None else None,
        "rate_strong": float(np.mean(rev_gt_strong_vec)) if rev_gt_strong_vec is not None else None,
        "used": _compute_rev_metrics(rev_gt_used_vec),
        "weak": _compute_rev_metrics(rev_gt_weak_vec),
        "strong": _compute_rev_metrics(rev_gt_strong_vec),
        "structured": structured_rev_metrics,
    }
    save_json(fig_dir / "test_metrics_by_reversal.json", rev_metrics)
    save_json(fig_dir / "test_metrics_reversal_structure.json", structured_rev_metrics)
    print("🔁 Reversal 指标:", rev_metrics)

    late_residual_metrics = {
        "enabled": bool(late_residual_den is not None),
        "start_sec": float(LATE_RESIDUAL_START_SEC) if late_residual_den is not None else None,
        "start_idx": int(round(float(LATE_RESIDUAL_START_SEC) * float(fs))) if late_residual_den is not None else None,
        "w_late_residual": float(W_LATE_RESIDUAL) if late_residual_den is not None else None,
    }
    if late_residual_den is not None:
        late_abs_mean = np.mean(np.abs(late_residual_den), axis=1)
        late_peak_abs = np.max(np.abs(late_residual_den), axis=1)
        late_tail_amp = np.ptp(late_residual_den, axis=1)
        late_residual_metrics.update({
            "mean_abs": float(np.mean(late_abs_mean)),
            "median_abs": float(np.median(late_abs_mean)),
            "peak_abs_mean": float(np.mean(late_peak_abs)),
            "tail_amp_mean": float(np.mean(late_tail_amp)),
            "nonzero_rate": float(np.mean(late_peak_abs > 1e-6)),
            "mask_support": late_residual_mask_ref.astype(np.float32).reshape(-1).tolist() if late_residual_mask_ref is not None else None,
        })
        if rev_gt_strong_vec is not None and rev_gt_strong_vec.size == late_abs_mean.size:
            strong_mask = rev_gt_strong_vec == 1
            weak_mask = ~strong_mask
            late_residual_metrics["strong_pos_mean_abs"] = float(np.mean(late_abs_mean[strong_mask])) if np.any(strong_mask) else None
            late_residual_metrics["strong_pos_peak_abs"] = float(np.mean(late_peak_abs[strong_mask])) if np.any(strong_mask) else None
            late_residual_metrics["non_strong_mean_abs"] = float(np.mean(late_abs_mean[weak_mask])) if np.any(weak_mask) else None
            late_residual_metrics["non_strong_peak_abs"] = float(np.mean(late_peak_abs[weak_mask])) if np.any(weak_mask) else None
    save_json(fig_dir / "test_late_residual_metrics.json", late_residual_metrics)
    print("🧩 Late residual 指标:", late_residual_metrics)


    # ---- state dump (event-level) ----
    component_names = state_component_names or [f"d{i}" for i in range(state_dim)]
    has_semantic_ac = bool(teacher_state_mode == "old_ac" and state_dim >= 2)
    state_dump = {
        "teacher_mask": zmask_all,
        "teacher_state_mode": teacher_state_mode,
        "is_curve": np.concatenate(is_curve_all, axis=0).astype(np.int64) if len(is_curve_all)>0 else -1,
        "curve_score": np.concatenate(curve_score_all, axis=0).astype(np.float32) if len(curve_score_all)>0 else np.nan,
        "rev_gt": np.concatenate(rev_gt_all, axis=0).astype(np.int64) if len(rev_gt_all)>0 else -1,
        "rev_gt_weak": np.concatenate(rev_gt_weak_all, axis=0).astype(np.int64) if len(rev_gt_weak_all)>0 else -1,
        "rev_gt_strong": np.concatenate(rev_gt_strong_all, axis=0).astype(np.int64) if len(rev_gt_strong_all)>0 else -1,
        "rev_prob": np.concatenate(rev_prob_all, axis=0).astype(np.float32) if len(rev_prob_all)>0 else np.nan,
        "strong_pos_gate_prob": np.concatenate(strong_pos_gate_prob_all, axis=0).astype(np.float32) if len(strong_pos_gate_prob_all)>0 else np.nan,
        "idx": np.concatenate(idx_all, axis=0).astype(np.int64) if len(idx_all)>0 else -1,
    }
    for j, col in enumerate(veh_state_cols):
        state_dump[col] = zveh_all[:, j]
    for j, col in enumerate(teacher_state_cols):
        state_dump[col] = zphys_all[:, j]
    if state_dim >= 2:
        state_dump["A_veh"] = zveh_all[:, 0]
        state_dump["C_veh"] = zveh_all[:, 1]
        state_dump["A_teacher"] = zphys_all[:, 0]
        state_dump["C_teacher"] = zphys_all[:, 1]
    if late_residual_den is not None:
        state_dump["late_residual_abs_mean"] = np.mean(np.abs(late_residual_den), axis=1).astype(np.float32)
        state_dump["late_residual_peak_abs"] = np.max(np.abs(late_residual_den), axis=1).astype(np.float32)
        state_dump["late_residual_tail_amp"] = np.ptp(late_residual_den, axis=1).astype(np.float32)
    df_state = pd.DataFrame(state_dump)
    df_state.to_csv(str(fig_dir / "test_state_dump.csv"), index=False, encoding="utf-8-sig")

    meta_out = {
        "teacher_state_mode": teacher_state_mode,
        "state_dim": int(state_dim),
        "component_names": component_names,
        "veh_state_cols": veh_state_cols,
        "teacher_state_cols": teacher_state_cols,
        "has_semantic_ac": has_semantic_ac,
    }
    save_json(fig_dir / "test_state_meta.json", meta_out)
    print("🧠 State dump meta:", meta_out)

    def _state_label(j):
        if has_semantic_ac and j == 0:
            return "A"
        if has_semantic_ac and j == 1:
            return "C"
        return component_names[j] if j < len(component_names) else f"d{j}"

    primary_plot_dims = min(2, state_dim)
    plot_dim_labels = [_state_label(j) for j in range(primary_plot_dims)]

    def _student_state_title(i):
        return summarize_state_vector(zveh_all[i], component_names)

    def _teacher_state_title(i):
        if zmask_all[i] <= 0.5:
            return "teacher=NA"
        return summarize_state_vector(zphys_all[i], component_names)

    print(f"🧾 已保存 test 状态/行为汇总: {fig_dir / 'test_state_dump.csv'}")

    peak_abs_steer = np.max(np.abs(true[:, :, 0]), axis=1)
    peak_abs_steer_plot = steer_array_for_plot(peak_abs_steer)
    peak_abs_yaw = np.max(np.abs(true[:, :, 1]), axis=1)
    peak_abs_ay = np.max(np.abs(true[:, :, 2]), axis=1)

    rmse_steer_evt = np.sqrt(np.mean((pred[:, :, 0] - true[:, :, 0]) ** 2, axis=1))
    rmse_yaw_evt = np.sqrt(np.mean((pred[:, :, 1] - true[:, :, 1]) ** 2, axis=1))
    rmse_ay_evt = np.sqrt(np.mean((pred[:, :, 2] - true[:, :, 2]) ** 2, axis=1))

    df_state["peak_abs_steer_gt"] = peak_abs_steer
    df_state["peak_abs_steer_gt_plot"] = peak_abs_steer_plot
    df_state["peak_abs_yaw_gt"] = peak_abs_yaw
    df_state["peak_abs_ay_gt"] = peak_abs_ay
    df_state["rmse_steer_evt"] = rmse_steer_evt
    df_state["rmse_yaw_evt"] = rmse_yaw_evt
    df_state["rmse_ay_evt"] = rmse_ay_evt
    df_state.to_csv(str(fig_dir / "test_state_dump.csv"), index=False, encoding="utf-8-sig")

    # ---- quick relationship plots (state vs peak) ----
    def _scatter(x, y, xlabel, ylabel, outname):
        plt.figure(figsize=(7.2, 5.0))
        plt.scatter(x, y, s=10, alpha=0.6)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(fig_dir / outname), dpi=200)
        plt.close()

    for j in range(primary_plot_dims):
        veh_col = veh_state_cols[j]
        label = plot_dim_labels[j]
        _scatter(df_state[veh_col].values, peak_abs_steer_plot, f"{label}_veh (student)", STEER_PEAK_PLOT_LABEL, f"state_vs_peak_steer_{label}.png")
        _scatter(df_state[veh_col].values, peak_abs_ay, f"{label}_veh (student)", "peak|ay| (GT)", f"state_vs_peak_ay_{label}.png")

    # ---- per-sample pred-vs-gt plots with state annotation ----
    n = pred.shape[0]
    if n == 0:
        print("⚠ test 集为空，无法画图")
        return

    t = np.arange(pred.shape[1], dtype=np.float32) / float(fs)
    pick = np.linspace(0, n - 1, num=min(n_examples, n), dtype=int)
    pred_plot = pred.copy()
    true_plot = true.copy()
    pred_plot[:, :, 0] = steer_array_for_plot(pred[:, :, 0])
    true_plot[:, :, 0] = steer_array_for_plot(true[:, :, 0])

    for k, idx in enumerate(pick):
        title = (
            f"Test sample #{idx} | Future {t[-1]:.2f}s | "
            f"veh[{_student_state_title(idx)}] | { _teacher_state_title(idx) }"
        )

        fig = plt.figure(figsize=(12, 8))

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(t, true_plot[idx, :, 0], label="GT", linewidth=1.2)
        ax1.plot(t, pred_plot[idx, :, 0], label="Pred", linewidth=1.2, linestyle="--")
        ax1.set_ylabel(STEER_PLOT_LABEL)
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(t, true[idx, :, 1], linewidth=1.2)
        ax2.plot(t, pred[idx, :, 1], linewidth=1.2, linestyle="--")
        ax2.set_ylabel("yawrate")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(t, true[idx, :, 2], linewidth=1.2)
        ax3.plot(t, pred[idx, :, 2], linewidth=1.2, linestyle="--")
        ax3.set_ylabel("ay")
        ax3.set_xlabel("time (s)")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = fig_dir / f"pred_vs_gt_example_{k:02d}_idx{idx}.png"
        plt.savefig(str(out_path), dpi=200)
        plt.close()

    print(f"🖼 已保存预测效果图到: {fig_dir} (pred_vs_gt_example_*.png)")

    return

    peak_abs_steer = np.max(np.abs(true[:, :, 0]), axis=1)
    peak_abs_yaw = np.max(np.abs(true[:, :, 1]), axis=1)
    peak_abs_ay = np.max(np.abs(true[:, :, 2]), axis=1)

    rmse_steer_evt = np.sqrt(np.mean((pred[:, :, 0] - true[:, :, 0]) ** 2, axis=1))
    rmse_yaw_evt = np.sqrt(np.mean((pred[:, :, 1] - true[:, :, 1]) ** 2, axis=1))
    rmse_ay_evt = np.sqrt(np.mean((pred[:, :, 2] - true[:, :, 2]) ** 2, axis=1))

    df_state["peak_abs_steer_gt"] = peak_abs_steer
    df_state["peak_abs_yaw_gt"] = peak_abs_yaw
    df_state["peak_abs_ay_gt"] = peak_abs_ay
    df_state["rmse_steer_evt"] = rmse_steer_evt
    df_state["rmse_yaw_evt"] = rmse_yaw_evt
    df_state["rmse_ay_evt"] = rmse_ay_evt
    df_state.to_csv(str(fig_dir / "test_state_dump.csv"), index=False, encoding="utf-8-sig")

    print(f"🧾 已保存 test 状态/行为汇总: {fig_dir / 'test_state_dump.csv'}")

    # ---- quick relationship plots (state vs peak) ----
    def _scatter(x, y, xlabel, ylabel, outname):
        plt.figure(figsize=(7.2, 5.0))
        plt.scatter(x, y, s=10, alpha=0.6)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(fig_dir / outname), dpi=200)
        plt.close()

    for j in range(primary_plot_dims):
        veh_col = veh_state_cols[j]
        label = plot_dim_labels[j]
        _scatter(df_state[veh_col].values, peak_abs_steer, f"{label}_veh (student)", STEER_PEAK_PLOT_LABEL, f"state_vs_peak_steer_{label}.png")
        _scatter(df_state[veh_col].values, peak_abs_ay, f"{label}_veh (student)", "peak|ay| (GT)", f"state_vs_peak_ay_{label}.png")

    # ---- per-sample pred-vs-gt plots with state annotation ----
    n = pred.shape[0]
    if n == 0:
        print("⚠ test 集为空，无法画图")
        return

    t = np.arange(pred.shape[1], dtype=np.float32) / float(fs)
    pick = np.linspace(0, n - 1, num=min(n_examples, n), dtype=int)

    for k, idx in enumerate(pick):
        title = (
            f"Test sample #{idx} | Future {t[-1]:.2f}s | "
            f"veh[{_student_state_title(idx)}] | {_teacher_state_title(idx)}"
        )

        fig = plt.figure(figsize=(12, 8))

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(t, true[idx, :, 0], label="GT", linewidth=1.2)
        ax1.plot(t, pred[idx, :, 0], label="Pred", linewidth=1.2, linestyle="--")
        ax1.set_ylabel(STEER_PLOT_LABEL)
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(t, true[idx, :, 1], linewidth=1.2)
        ax2.plot(t, pred[idx, :, 1], linewidth=1.2, linestyle="--")
        ax2.set_ylabel("yawrate")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(t, true[idx, :, 2], linewidth=1.2)
        ax3.plot(t, pred[idx, :, 2], linewidth=1.2, linestyle="--")
        ax3.set_ylabel("ay")
        ax3.set_xlabel("time (s)")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = fig_dir / f"pred_vs_gt_example_{k:02d}_idx{idx}.png"
        plt.savefig(str(out_path), dpi=200)
        plt.close()

    print(f"🖼 已保存预测效果图到: {fig_dir} (pred_vs_gt_example_*.png)")

    return
