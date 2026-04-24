from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from event_conditioned_eval_support import (
    annotate_event_meta,
    build_primary_selection_bundle,
    structure_aware_selection_key,
)
from event_conditioned_baseline_model import (
    EventConditionedDataset,
    EventConditionedTrajectoryModel,
    build_event_schema_targets,
    build_event_teacher_from_batch,
    compute_event_loss,
    count_parameters,
    masked_mse,
    subset_array_dict,
)
from future_steer_speed_subjectsplit_masked import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    FUTURE_LEN,
    RESULT_ROOT,
    _make_sample,
    normalize_inputs,
    save_json,
)


THIS_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = THIS_DIR / "protocol_allphase_control_v2_context_full2s"
DEFAULT_MANIFEST = PROTOCOL_DIR / "sample_manifest.csv"
RUN_ROOT = RESULT_ROOT.parent / "event_conditioned_runs"

# Tail amplitude penalty
TAIL_START = 200          # step index where tail begins (1.0 s at 200 Hz)
W_TAIL_AMP = 0.3          # penalty weight; adjust after seeing results


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _sample_by_split(meta_df: pd.DataFrame, split: str, n_keep: int | None, seed: int) -> pd.DataFrame:
    split_df = meta_df[meta_df["split"].astype(str) == split]
    if n_keep is None or n_keep <= 0 or len(split_df) <= n_keep:
        return split_df
    return split_df.sample(n=n_keep, random_state=seed)


def subset_manifest(
    manifest_df: pd.DataFrame,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    seed: int,
) -> pd.DataFrame:
    if not {"train", "val", "test"}.issubset(set(manifest_df["split"].astype(str).unique())):
        return manifest_df.reset_index(drop=True)
    out = pd.concat(
        [
            _sample_by_split(manifest_df, "train", max_train_samples, seed + 11),
            _sample_by_split(manifest_df, "val", max_val_samples, seed + 13),
            _sample_by_split(manifest_df, "test", max_test_samples, seed + 17),
        ],
        axis=0,
        ignore_index=True,
    )
    return out.reset_index(drop=True)


def build_sample_bundle_from_manifest(
    manifest_path: str | Path,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int]:
    meta_df = pd.read_csv(manifest_path)
    meta_df = subset_manifest(meta_df, max_train_samples, max_val_samples, max_test_samples, seed=seed)

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    curve_list: list[np.ndarray] = []
    ctx_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    keep_rows: list[int] = []
    dropped = 0

    for i, row in meta_df.iterrows():
        try:
            x_win, y_seq, curve_future, ctx, future_mask = _make_sample(row)
        except Exception:
            dropped += 1
            continue
        x_list.append(x_win)
        y_list.append(y_seq)
        curve_list.append(curve_future)
        ctx_list.append(ctx)
        mask_list.append(future_mask)
        keep_rows.append(i)

    if not x_list:
        raise RuntimeError("No valid samples were built from manifest; check manifest path and data files.")

    kept_meta = meta_df.iloc[keep_rows].reset_index(drop=True).copy()
    return (
        np.stack(x_list).astype(np.float32),
        np.stack(y_list).astype(np.float32),
        np.stack(curve_list).astype(np.float32),
        np.stack(ctx_list).astype(np.float32),
        np.stack(mask_list).astype(np.float32),
        kept_meta,
        dropped,
    )


def make_loader(dataset: EventConditionedDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key.endswith("_has"):
                out[key] = value.to(device=device, dtype=torch.float32)
            elif key in {"src", "y_true", "curve_norm", "ctx", "event_mask"}:
                out[key] = value.to(device=device, dtype=torch.float32)
            else:
                out[key] = value.to(device=device)
    return out


def _compute_event_metrics(batch: dict[str, torch.Tensor], event_logits: dict[str, torch.Tensor]) -> dict[str, float]:
    turn_has_pred = (torch.sigmoid(event_logits["first_major_turn_onset_has_logit"]) >= 0.5).to(dtype=torch.float32)
    reversal_has_pred = (torch.sigmoid(event_logits["first_reversal_has_logit"]) >= 0.5).to(dtype=torch.float32)
    peak_idx_pred = torch.argmax(event_logits["main_peak_idx_logits"], dim=1)

    turn_has_acc = (turn_has_pred == batch["first_major_turn_onset_has"]).to(dtype=torch.float32).mean().item()
    reversal_has_acc = (reversal_has_pred == batch["first_reversal_has"]).to(dtype=torch.float32).mean().item()

    valid_peak = (batch["event_mask"].sum(dim=1) > 0)
    if valid_peak.any():
        peak_mae = (peak_idx_pred[valid_peak] - batch["main_peak_idx"][valid_peak]).abs().to(dtype=torch.float32).mean().item()
    else:
        peak_mae = 0.0
    return {
        "turn_has_acc": float(turn_has_acc),
        "reversal_has_acc": float(reversal_has_acc),
        "main_peak_idx_mae": float(peak_mae),
    }


def evaluate_epoch(
    model: EventConditionedTrajectoryModel,
    loader: DataLoader,
    meta_df: pd.DataFrame,
    split_name: str,
    seed: int,
    device: str,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
    event_loss_weight: float,
    use_privileged_teacher: bool = False,
) -> dict[str, float]:
    model.eval()
    traj_loss_total = 0.0
    event_loss_total = 0.0
    total_loss = 0.0
    rmse_steer_num = 0.0
    rmse_speed_num = 0.0
    rmse_den = 0.0
    metric_accum = {"turn_has_acc": 0.0, "reversal_has_acc": 0.0, "main_peak_idx_mae": 0.0}
    n_batch = 0
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ctxs_raw: list[np.ndarray] = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch_to_device(raw_batch, device=device)
            privileged_teacher = None
            if use_privileged_teacher and "privileged_event_teacher" in batch:
                privileged_teacher = batch["privileged_event_teacher"]
            y_hat, extras = model(
                src=batch["src"],
                ctx=batch["ctx"],
                curve_norm=batch["curve_norm"],
                event_teacher=None,
                privileged_event_teacher=privileged_teacher,
            )
            traj_mask = batch["event_mask"].unsqueeze(-1)
            traj_loss = masked_mse(y_hat, batch["y_true"], traj_mask)
            event_breakdown = compute_event_loss(batch, extras["event_logits"])
            loss = traj_loss + event_loss_weight * event_breakdown.total

            y_hat_den = y_hat * y_std_t.view(1, 1, -1) + y_mean_t.view(1, 1, -1)
            y_true_den = batch["y_true"] * y_std_t.view(1, 1, -1) + y_mean_t.view(1, 1, -1)
            mask = batch["event_mask"]
            rmse_steer_num += float((((y_hat_den[:, :, 0] - y_true_den[:, :, 0]) ** 2) * mask).sum().item())
            rmse_speed_num += float((((y_hat_den[:, :, 1] - y_true_den[:, :, 1]) ** 2) * mask).sum().item())
            rmse_den += float(mask.sum().item())
            preds.append(y_hat_den.cpu().numpy())
            trues.append(y_true_den.cpu().numpy())
            masks.append(mask.cpu().numpy())
            ctxs_raw.append(raw_batch["ctx_raw"].cpu().numpy())

            metrics = _compute_event_metrics(batch, extras["event_logits"])
            for key, value in metrics.items():
                metric_accum[key] += value

            traj_loss_total += float(traj_loss.item())
            event_loss_total += float(event_breakdown.total.item())
            total_loss += float(loss.item())
            n_batch += 1

    denom = max(n_batch, 1)
    rmse_den = max(rmse_den, 1.0)
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    mask = np.concatenate(masks, axis=0)
    ctx_raw = np.concatenate(ctxs_raw, axis=0)
    selection_bundle = build_primary_selection_bundle(
        pred=pred,
        true=true,
        mask=mask,
        ctx_raw=ctx_raw,
        meta_df=meta_df,
        split_name=split_name,
        seed=seed,
    )
    out = {
        "loss": total_loss / denom,
        "traj_loss": traj_loss_total / denom,
        "event_loss": event_loss_total / denom,
        "steer_rmse": float(np.sqrt(rmse_steer_num / rmse_den)),
        "speed_rmse": float(np.sqrt(rmse_speed_num / rmse_den)),
        "turn_has_acc": metric_accum["turn_has_acc"] / denom,
        "reversal_has_acc": metric_accum["reversal_has_acc"] / denom,
        "main_peak_idx_mae": metric_accum["main_peak_idx_mae"] / denom,
    }
    out.update(
        {
            "selection_summary": selection_bundle["selection_summary"],
            "trajectory_sample_df": selection_bundle["sample_df"],
            "primary_trajectory_sample_df": selection_bundle["primary_sample_df"],
            "weighted_metrics": selection_bundle["weighted"],
            "primary_weighted_metrics": selection_bundle["primary_weighted"],
            "interaction_sample_count": int(
                (selection_bundle["sample_df"].get("interaction_slice", pd.Series([], dtype=object)).astype(str) == "interaction").sum()
            ),
        }
    )
    return out


def _compact_eval_summary(eval_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": float(eval_metrics["loss"]),
        "traj_loss": float(eval_metrics["traj_loss"]),
        "event_loss": float(eval_metrics["event_loss"]),
        "steer_rmse": float(eval_metrics["steer_rmse"]),
        "speed_rmse": float(eval_metrics["speed_rmse"]),
        "turn_has_acc": float(eval_metrics["turn_has_acc"]),
        "reversal_has_acc": float(eval_metrics["reversal_has_acc"]),
        "main_peak_idx_mae": float(eval_metrics["main_peak_idx_mae"]),
        "interaction_sample_count": int(eval_metrics.get("interaction_sample_count", 0)),
        "selection_summary": {
            key: float(value) if isinstance(value, (int, float, np.floating)) else value
            for key, value in eval_metrics["selection_summary"].items()
        },
    }


def train_one_run(args: argparse.Namespace) -> dict[str, Any]:
    set_determinism(seed=int(args.seed))
    run_name = f"{args.run_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_root = RUN_ROOT / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    max_train = args.max_train_samples
    max_val = args.max_val_samples
    max_test = args.max_test_samples
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    if bool(args.smoke_test):
        epochs = int(args.smoke_epochs)
        batch_size = int(args.smoke_batch_size)
        max_train = int(args.smoke_train_samples)
        max_val = int(args.smoke_val_samples)
        max_test = int(args.smoke_test_samples)

    sample_bundle = build_sample_bundle_from_manifest(
        manifest_path=args.manifest,
        max_train_samples=max_train,
        max_val_samples=max_val,
        max_test_samples=max_test,
        seed=int(args.seed),
    )
    X_pool, y_pool, curve_pool, ctx_pool, mask_pool, meta_df, dropped_count = sample_bundle
    meta_df = annotate_event_meta(meta_df, y_pool, mask_pool)

    split_series = meta_df["split"].astype(str).reset_index(drop=True)
    train_idx = split_series.index[split_series == "train"].tolist()
    val_idx = split_series.index[split_series == "val"].tolist()
    test_idx = split_series.index[split_series == "test"].tolist()
    if not train_idx or not val_idx or not test_idx:
        raise RuntimeError("Split samples are incomplete after filtering; check manifest subset settings.")

    X_norm, norm_stats = normalize_inputs(X_pool, y_pool, curve_pool, ctx_pool, train_idx)
    event_targets = build_event_schema_targets(
        y_pool=y_pool,
        mask_pool=mask_pool,
        future_len=FUTURE_LEN,
        event_bin_size=int(args.event_bin_size),
    )

    train_ds = EventConditionedDataset(
        X_norm=X_norm[train_idx],
        y_pool=y_pool[train_idx],
        curve_pool=curve_pool[train_idx],
        ctx_pool=ctx_pool[train_idx],
        mask_pool=mask_pool[train_idx],
        norm_stats=norm_stats,
        event_targets=subset_array_dict(event_targets, train_idx),
        meta_df=meta_df.iloc[train_idx].reset_index(drop=True),
    )
    val_ds = EventConditionedDataset(
        X_norm=X_norm[val_idx],
        y_pool=y_pool[val_idx],
        curve_pool=curve_pool[val_idx],
        ctx_pool=ctx_pool[val_idx],
        mask_pool=mask_pool[val_idx],
        norm_stats=norm_stats,
        event_targets=subset_array_dict(event_targets, val_idx),
        meta_df=meta_df.iloc[val_idx].reset_index(drop=True),
    )
    test_ds = EventConditionedDataset(
        X_norm=X_norm[test_idx],
        y_pool=y_pool[test_idx],
        curve_pool=curve_pool[test_idx],
        ctx_pool=ctx_pool[test_idx],
        mask_pool=mask_pool[test_idx],
        norm_stats=norm_stats,
        event_targets=subset_array_dict(event_targets, test_idx),
        meta_df=meta_df.iloc[test_idx].reset_index(drop=True),
    )

    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True, seed=int(args.seed) + 101)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False, seed=int(args.seed) + 103)
    test_loader = make_loader(test_ds, batch_size=batch_size, shuffle=False, seed=int(args.seed) + 107)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EventConditionedTrajectoryModel(
        input_dim=int(train_ds.src.shape[-1]),
        context_dim=int(train_ds.ctx.shape[-1]),
        future_len=FUTURE_LEN,
        event_bin_size=int(args.event_bin_size),
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        enc_layers=int(args.enc_layers),
        dec_layers=int(args.dec_layers),
        ffn_dim=int(args.ffn_dim),
        dropout=float(args.dropout),
        event_embed_dim=int(args.event_embed_dim),
        out_dim=2,
        conditioning_mode=str(args.conditioning_mode),
        structure_width=float(args.structure_width),
        gate_temperature=float(args.gate_temperature),
        event_residual_scale=float(args.event_residual_scale),
    ).to(device)
    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(init_ckpt["model_state"], strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    y_mean_t = torch.tensor(norm_stats["y_mean"], dtype=torch.float32, device=device)
    y_std_t = torch.tensor(norm_stats["y_std"], dtype=torch.float32, device=device)

    best_val = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    best_ckpt = run_root / "best_model.pt"
    legacy_best_ckpt = run_root / "best_model_legacy.pt"
    structure_best_ckpt = run_root / "best_model_structure.pt"
    teacher_rng = random.Random(int(args.seed) + 999)
    selection_mode = str(args.selection_mode)
    patience = int(args.patience)
    min_epochs = int(args.min_epochs)
    best_structure_key: tuple[float, ...] | None = None
    best_legacy_key: tuple[float, ...] | None = None
    best_structure_epoch = 0
    best_legacy_epoch = 0
    best_structure_summary: dict[str, Any] | None = None
    best_legacy_summary: dict[str, Any] | None = None
    active_best_key: tuple[float, ...] | None = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        traj_sum = 0.0
        event_sum = 0.0
        n_batch = 0
        teacher_steps = 0

        for raw_batch in train_loader:
            batch = _move_batch_to_device(raw_batch, device=device)
            use_teacher = False
            if float(args.teacher_forcing_ratio) >= 1.0:
                use_teacher = True
            elif float(args.teacher_forcing_ratio) > 0.0:
                use_teacher = teacher_rng.random() < float(args.teacher_forcing_ratio)

            teacher_events = build_event_teacher_from_batch(batch, device=device) if use_teacher else None
            if use_teacher:
                teacher_steps += 1
            privileged_teacher = None
            if bool(args.use_privileged_teacher) and "privileged_event_teacher" in batch:
                privileged_teacher = batch["privileged_event_teacher"]

            optimizer.zero_grad()
            y_hat, extras = model(
                src=batch["src"],
                ctx=batch["ctx"],
                curve_norm=batch["curve_norm"],
                event_teacher=teacher_events,
                privileged_event_teacher=privileged_teacher,
            )
            traj_mask = batch["event_mask"].unsqueeze(-1)
            traj_loss = masked_mse(y_hat, batch["y_true"], traj_mask)
            event_breakdown = compute_event_loss(batch, extras["event_logits"])
            # Tail amplitude penalty (steer channel only, steps >= TAIL_START).
            tail_mask = traj_mask[:, TAIL_START:, :]
            pred_amp = y_hat[:, TAIL_START:, 0:1].abs()
            true_amp = batch["y_true"][:, TAIL_START:, 0:1].abs()
            tail_amp_loss = masked_mse(pred_amp, true_amp, tail_mask)
            loss = (
                traj_loss
                + float(args.event_loss_weight) * event_breakdown.total
                + W_TAIL_AMP * tail_amp_loss
            )
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            loss_sum += float(loss.item())
            traj_sum += float(traj_loss.item())
            event_sum += float(event_breakdown.total.item())
            n_batch += 1

        train_metrics = {
            "loss": loss_sum / max(n_batch, 1),
            "traj_loss": traj_sum / max(n_batch, 1),
            "event_loss": event_sum / max(n_batch, 1),
            "teacher_step_ratio": teacher_steps / max(n_batch, 1),
        }
        val_metrics = evaluate_epoch(
            model=model,
            loader=val_loader,
            meta_df=val_ds.meta_df,
            split_name="val",
            seed=int(args.seed),
            device=device,
            y_mean_t=y_mean_t,
            y_std_t=y_std_t,
            event_loss_weight=float(args.event_loss_weight),
            use_privileged_teacher=bool(args.use_privileged_teacher),
        )
        selection_summary = val_metrics["selection_summary"]
        structure_key = structure_aware_selection_key(selection_summary)
        legacy_key = (float(val_metrics["steer_rmse"]),)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_traj_loss": train_metrics["traj_loss"],
            "train_event_loss": train_metrics["event_loss"],
            "train_teacher_step_ratio": train_metrics["teacher_step_ratio"],
            "val_loss": val_metrics["loss"],
            "val_steer_rmse": val_metrics["steer_rmse"],
            "val_speed_rmse": val_metrics["speed_rmse"],
            "val_turn_has_acc": val_metrics["turn_has_acc"],
            "val_reversal_has_acc": val_metrics["reversal_has_acc"],
            "val_main_peak_idx_mae": val_metrics["main_peak_idx_mae"],
            "val_selection_mode": selection_mode,
            "val_selection_score": float(selection_summary["selection_score"]),
            "val_primary_rmse_score": float(selection_summary["primary_rmse_score"]),
            "val_trajectory_score": float(selection_summary["trajectory_score"]),
            "val_tail_score": float(selection_summary["tail_score"]),
            "val_trend_score": float(selection_summary["trend_score"]),
            "val_turning_score": float(selection_summary["turning_score"]),
            "val_continuity_score": float(selection_summary["continuity_score"]),
            "val_tail_rmse": float(selection_summary["rmse_tail_abs_steer"]),
            "val_tail_pre_ratio": float(selection_summary["tail_pre_ratio_abs_steer"]),
            "val_tail_trend_corr": float(selection_summary["tail_trend_corr"]),
            "val_turning_count_abs_err": float(selection_summary["turning_count_abs_err"]),
            "val_peak_time_abs_err_s": float(selection_summary["peak_time_abs_err_s"]),
            "val_boundary_shift_abs_err": float(selection_summary["boundary_shift_abs_err"]),
            "val_interaction_sample_count": int(val_metrics["interaction_sample_count"]),
        }
        history.append(epoch_log)

        checkpoint_payload = {
            "model_state": model.state_dict(),
            "args": vars(args),
            "norm_stats": norm_stats,
            "epoch": int(epoch),
            "selection_summary": selection_summary,
        }
        if best_structure_key is None or structure_key < best_structure_key:
            best_structure_key = structure_key
            best_structure_epoch = int(epoch)
            best_structure_summary = _compact_eval_summary(val_metrics)
            torch.save(checkpoint_payload, structure_best_ckpt)
        if best_legacy_key is None or legacy_key < best_legacy_key:
            best_legacy_key = legacy_key
            best_legacy_epoch = int(epoch)
            best_legacy_summary = _compact_eval_summary(val_metrics)
            torch.save(checkpoint_payload, legacy_best_ckpt)

        active_key = structure_key if selection_mode == "structure_aware_primary" else legacy_key
        if active_best_key is None or active_key < active_best_key:
            active_best_key = active_key
            best_val = float(val_metrics["steer_rmse"])
            best_epoch = int(epoch)
            bad_epochs = 0
            torch.save(checkpoint_payload, best_ckpt)
        else:
            bad_epochs += 1

        if epoch >= min_epochs and bad_epochs >= patience:
            break

    active_ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(active_ckpt["model_state"])
    val_metrics = evaluate_epoch(
        model=model,
        loader=val_loader,
        meta_df=val_ds.meta_df,
        split_name="val",
        seed=int(args.seed),
        device=device,
        y_mean_t=y_mean_t,
        y_std_t=y_std_t,
        event_loss_weight=float(args.event_loss_weight),
        use_privileged_teacher=bool(args.use_privileged_teacher),
    )
    test_metrics = evaluate_epoch(
        model=model,
        loader=test_loader,
        meta_df=test_ds.meta_df,
        split_name="test",
        seed=int(args.seed),
        device=device,
        y_mean_t=y_mean_t,
        y_std_t=y_std_t,
        event_loss_weight=float(args.event_loss_weight),
        use_privileged_teacher=bool(args.use_privileged_teacher),
    )

    selection_compare_rows: list[dict[str, Any]] = []
    compare_payload: dict[str, Any] = {}
    for tag, ckpt_path in (("legacy", legacy_best_ckpt), ("structure", structure_best_ckpt), ("active", best_ckpt)):
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        val_eval = evaluate_epoch(
            model=model,
            loader=val_loader,
            meta_df=val_ds.meta_df,
            split_name="val",
            seed=int(args.seed),
            device=device,
            y_mean_t=y_mean_t,
            y_std_t=y_std_t,
            event_loss_weight=float(args.event_loss_weight),
            use_privileged_teacher=bool(args.use_privileged_teacher),
        )
        test_eval = evaluate_epoch(
            model=model,
            loader=test_loader,
            meta_df=test_ds.meta_df,
            split_name="test",
            seed=int(args.seed),
            device=device,
            y_mean_t=y_mean_t,
            y_std_t=y_std_t,
            event_loss_weight=float(args.event_loss_weight),
            use_privileged_teacher=bool(args.use_privileged_teacher),
        )
        selection_summary_eval = val_eval["selection_summary"]
        selection_compare_rows.append(
            {
                "selection_tag": tag,
                "epoch": int(ckpt.get("epoch", 0)),
                "val_steer_rmse": float(val_eval["steer_rmse"]),
                "val_selection_score": float(selection_summary_eval["selection_score"]),
                "val_trajectory_score": float(selection_summary_eval["trajectory_score"]),
                "val_turning_score": float(selection_summary_eval["turning_score"]),
                "val_tail_trend_corr": float(selection_summary_eval["tail_trend_corr"]),
                "val_tail_rmse": float(selection_summary_eval["rmse_tail_abs_steer"]),
                "val_boundary_shift_abs_err": float(selection_summary_eval["boundary_shift_abs_err"]),
                "test_steer_rmse": float(test_eval["steer_rmse"]),
                "test_selection_score": float(test_eval["selection_summary"]["selection_score"]),
                "test_tail_trend_corr": float(test_eval["selection_summary"]["tail_trend_corr"]),
                "test_tail_rmse": float(test_eval["selection_summary"]["rmse_tail_abs_steer"]),
            }
        )
        compare_payload[tag] = {
            "epoch": int(ckpt.get("epoch", 0)),
            "val": _compact_eval_summary(val_eval),
            "test": _compact_eval_summary(test_eval),
        }

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_root / "loss_history.csv", index=False)
    pd.DataFrame(selection_compare_rows).to_csv(run_root / "selection_comparison.csv", index=False)
    meta_df.to_csv(run_root / "sample_manifest_used.csv", index=False)
    save_json(
        run_root / "run_summary.json",
        {
            "run_root": str(run_root),
            "smoke_test": bool(args.smoke_test),
            "manifest": str(args.manifest),
            "dropped_samples": int(dropped_count),
            "device": str(device),
            "parameter_count": count_parameters(model),
            "selection_mode": selection_mode,
            "best_epoch": int(best_epoch),
            "best_val_steer_rmse": float(best_val),
            "best_structure_epoch": int(best_structure_epoch),
            "best_legacy_epoch": int(best_legacy_epoch),
            "final_val_metrics": _compact_eval_summary(val_metrics),
            "final_test_metrics": _compact_eval_summary(test_metrics),
            "selection_compare": compare_payload,
            "config": vars(args),
        },
    )
    save_json(
        run_root / "metrics.json",
        {
            "val": _compact_eval_summary(val_metrics),
            "test": _compact_eval_summary(test_metrics),
        },
    )
    return {
        "run_root": str(run_root),
        "best_epoch": int(best_epoch),
        "best_val_steer_rmse": float(best_val),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "selection_mode": selection_mode,
        "best_structure_epoch": int(best_structure_epoch),
        "best_legacy_epoch": int(best_legacy_epoch),
        "selection_compare_path": str(run_root / "selection_comparison.csv"),
        "parameter_count": count_parameters(model),
        "dropped_samples": int(dropped_count),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--run-prefix", default="EXP_EVENT_CONDITIONED_TRAJECTORY_BASELINE")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--init-checkpoint", default=None)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--event-loss-weight", type=float, default=0.50)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=1.0)
    parser.add_argument(
        "--selection-mode",
        default="legacy_rmse",
        choices=["legacy_rmse", "structure_aware_primary"],
    )

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--enc-layers", type=int, default=2)
    parser.add_argument("--dec-layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--event-embed-dim", type=int, default=96)
    parser.add_argument("--event-bin-size", type=int, default=20)
    parser.add_argument("--conditioning-mode", default="baseline", choices=["baseline", "structured_v2"])
    parser.add_argument("--structure-width", type=float, default=0.065)
    parser.add_argument("--gate-temperature", type=float, default=0.040)
    parser.add_argument("--event-residual-scale", type=float, default=1.0)
    parser.add_argument("--use-privileged-teacher", action="store_true")

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)

    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-batch-size", type=int, default=16)
    parser.add_argument("--smoke-train-samples", type=int, default=96)
    parser.add_argument("--smoke-val-samples", type=int, default=32)
    parser.add_argument("--smoke-test-samples", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_one_run(args)
    print(result["run_root"])
    print(result["best_val_steer_rmse"])
    print(result["test_metrics"]["steer_rmse"])


if __name__ == "__main__":
    main()
