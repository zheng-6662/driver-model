from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    loader = SourceFileLoader(module_name, str(module_path))
    spec = importlib.util.spec_from_loader(module_name, loader)
    if spec is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    loader.exec_module(module)
    return module


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def try_load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def resolve_steer_angle_unit(model_config: dict[str, Any]) -> str:
    unit = str(model_config.get("STEER_ANGLE_UNIT", "rad")).strip().lower()
    if unit not in {"rad", "deg"}:
        raise ValueError(f"Unsupported STEER_ANGLE_UNIT={unit!r}; expected 'rad' or 'deg'")
    return unit


def subset_list(items, indices: np.ndarray):
    return [items[int(i)] for i in indices]


def subset_array(items, indices: np.ndarray):
    arr = np.asarray(items)
    return arr[indices]


def build_split_indices(
    module,
    sample_meta_df: pd.DataFrame,
    split_mode: str,
    smoke_max_samples: int,
    seed: int,
    protocol_config_path: str | None = None,
    frozen_split_path: str | None = None,
):
    if split_mode == "protocol_safe":
        if not hasattr(module, "load_protocol_split") or not hasattr(module, "build_subject_split_indices"):
            raise RuntimeError("protocol_safe split requested but module does not expose protocol split helpers")
        load_kwargs = {}
        if protocol_config_path is not None:
            load_kwargs["protocol_config_path"] = protocol_config_path
        if frozen_split_path is not None:
            load_kwargs["frozen_split_path"] = frozen_split_path
        protocol_config, split_subjects = module.load_protocol_split(**load_kwargs)
        raw = module.build_subject_split_indices(sample_meta_df, split_subjects)
        train_idx = np.asarray(raw["train"], dtype=np.int64)
        val_idx = np.asarray(raw["val"], dtype=np.int64)
        test_idx = np.asarray(raw["test"], dtype=np.int64)
        ordered = np.concatenate([train_idx, val_idx, test_idx])
        extras = {
            "split_policy": "protocol_safe_subject_split",
            "protocol_version": protocol_config.get("protocol_version"),
            "train_subjects": list(split_subjects["train"]),
            "val_subjects": list(split_subjects["val"]),
            "test_subjects": list(split_subjects["test"]),
            "smoke_max_samples": None,
        }
        return ordered, len(train_idx), len(val_idx), len(test_idx), extras

    if split_mode == "smoke_random80":
        total = min(len(sample_meta_df), int(smoke_max_samples))
        rng = np.random.RandomState(int(seed))
        ordered = np.arange(total, dtype=np.int64)
        rng.shuffle(ordered)
        n_train = int(total * 0.8)
        n_test = total - n_train
        extras = {
            "split_policy": "smoke_first_n_then_random_80_20",
            "protocol_version": None,
            "train_subjects": None,
            "val_subjects": None,
            "test_subjects": None,
            "smoke_max_samples": int(total),
        }
        return ordered, n_train, 0, n_test, extras

    raise ValueError(f"Unsupported split mode: {split_mode}")


def build_teacher_state_with_module(module, base_z_all: np.ndarray, mode: str, state_dim: int, n_train: int):
    sig = inspect.signature(module.build_teacher_state)
    if "fit_indices" in sig.parameters:
        fit_indices = np.arange(n_train, dtype=np.int64)
        return module.build_teacher_state(base_z_all, mode=mode, state_dim=state_dim, fit_indices=fit_indices)
    if "train_count" in sig.parameters:
        return module.build_teacher_state(base_z_all, mode=mode, state_dim=state_dim, train_count=n_train)
    raise RuntimeError("Unsupported build_teacher_state signature")


def build_collate_fn():
    def collate_fn(batch):
        out = {
            "src": torch.stack([torch.from_numpy(b["src"]).float() for b in batch], dim=0),
            "y_norm": torch.stack([torch.from_numpy(b["y_norm"]).float() for b in batch], dim=0),
            "curve_norm": torch.stack([torch.from_numpy(b["curve_norm"]).float() for b in batch], dim=0),
            "ctx": torch.stack([torch.from_numpy(b["ctx"]).float() for b in batch], dim=0),
            "z_phys": torch.stack([torch.from_numpy(b["z_phys"]).float() for b in batch], dim=0),
            "z_mask": torch.stack([torch.from_numpy(b["z_mask"]).float() for b in batch], dim=0),
            "rev_gt": torch.stack([torch.from_numpy(b["rev_gt"]).float() for b in batch], dim=0),
            "rev_gt_weak": torch.stack([torch.from_numpy(b["rev_gt_weak"]).float() for b in batch], dim=0),
            "rev_gt_strong": torch.stack([torch.from_numpy(b["rev_gt_strong"]).float() for b in batch], dim=0),
            "idx": torch.stack([torch.from_numpy(b["idx"]).long() for b in batch], dim=0).squeeze(1),
            "curve_score": torch.stack([torch.from_numpy(b["curve_score"]).float() for b in batch], dim=0).squeeze(1),
            "is_curve": torch.stack([torch.from_numpy(b["is_curve"]).long() for b in batch], dim=0).squeeze(1),
        }
        return out

    return collate_fn


def unpack_model_output(output):
    if not isinstance(output, tuple):
        raise TypeError(f"Unexpected model output type: {type(output)!r}")
    if len(output) == 3:
        y_hat, z_veh, rev_logit = output
        return y_hat, z_veh, rev_logit, {}
    if len(output) == 4:
        y_hat, z_veh, rev_logit, aux = output
        return y_hat, z_veh, rev_logit, (aux or {})
    raise ValueError(f"Unexpected model output length: {len(output)}")


def compute_basic_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred - true
    return {
        "rmse_all": float(np.sqrt(np.mean(err ** 2))),
        "rmse_steer": float(np.sqrt(np.mean(err[:, :, 0] ** 2))),
        "rmse_yawrate": float(np.sqrt(np.mean(err[:, :, 1] ** 2))),
        "rmse_ay": float(np.sqrt(np.mean(err[:, :, 2] ** 2))),
        "mae_steer": float(np.mean(np.abs(err[:, :, 0]))),
        "mae_yawrate": float(np.mean(np.abs(err[:, :, 1]))),
        "mae_ay": float(np.mean(np.abs(err[:, :, 2]))),
        "n_test": int(pred.shape[0]),
        "future_len": int(pred.shape[1]),
    }


def build_case_rows(metrics_module, pred: np.ndarray, true: np.ndarray, test_meta_df: pd.DataFrame, fs: int):
    rows: list[dict[str, Any]] = []
    tail_start_idx = int(round(pred.shape[1] * 0.75))
    for i in range(pred.shape[0]):
        meta = test_meta_df.iloc[i].to_dict()
        pred_steer = np.asarray(pred[i, :, 0], dtype=np.float64)
        true_steer = np.asarray(true[i, :, 0], dtype=np.float64)

        true_base = float(true_steer[0])
        pred_base = float(pred_steer[0])
        true_peak_delta = float(np.max(np.abs(true_steer - true_base)))
        onset_thr_abs = float(getattr(metrics_module, "STEER_ONSET_THR_ABS", 0.02))
        onset_thr = max(onset_thr_abs, 0.15 * true_peak_delta)

        gt_onset_idx = metrics_module._first_threshold_crossing_idx_np(true_steer, threshold=onset_thr, ref_value=true_base)
        pred_onset_idx = metrics_module._first_threshold_crossing_idx_np(pred_steer, threshold=onset_thr, ref_value=pred_base)
        gt_peak_idx = int(np.argmax(np.abs(true_steer)))
        pred_peak_idx = int(np.argmax(np.abs(pred_steer)))
        gt_tail = true_steer[tail_start_idx:]
        pred_tail = pred_steer[tail_start_idx:]
        gt_tail_amp = float(np.ptp(gt_tail)) if gt_tail.size else float("nan")
        pred_tail_amp = float(np.ptp(pred_tail)) if pred_tail.size else float("nan")
        gt_first_rev_sec = metrics_module._first_reversal_time_np(true_steer, eps=metrics_module.REV_EPS_WEAK, fs=fs)
        pred_first_rev_sec = metrics_module._first_reversal_time_np(pred_steer, eps=metrics_module.REV_EPS_WEAK, fs=fs)
        head_amp_gt = float(np.ptp(true_steer[:100])) if true_steer.size >= 100 else float(np.ptp(true_steer))
        head_amp_pred = float(np.ptp(pred_steer[:100])) if pred_steer.size >= 100 else float(np.ptp(pred_steer))

        row = {
            "sample_key": meta.get("sample_key"),
            "subject_id": meta.get("subject_id"),
            "vehicle_file": meta.get("vehicle_file"),
            "event_idx": meta.get("event_idx"),
            "event_level": meta.get("event_level"),
            "split": meta.get("protocol_split_applied", meta.get("split", "test")),
            "anchor_idx": meta.get("anchor_idx"),
            "anchor_source_applied": meta.get("anchor_source_applied"),
            "maintained_anchor_policy": meta.get("maintained_anchor_policy"),
            "is_curve_applied": meta.get("is_curve_applied"),
            "curve_score_event_mean_abs": meta.get("curve_score_event_mean_abs"),
            "trigger_idx": meta.get("trigger_idx"),
            "gt_onset_idx": gt_onset_idx,
            "gt_onset_sec": None if gt_onset_idx is None else float(gt_onset_idx / fs),
            "pred_onset_idx": pred_onset_idx,
            "pred_onset_sec": None if pred_onset_idx is None else float(pred_onset_idx / fs),
            "gt_main_peak_idx": gt_peak_idx,
            "gt_main_peak_sec": float(gt_peak_idx / fs),
            "pred_main_peak_idx": pred_peak_idx,
            "pred_main_peak_sec": float(pred_peak_idx / fs),
            "gt_first_reversal_sec": None if gt_first_rev_sec is None else float(gt_first_rev_sec),
            "pred_first_reversal_sec": None if pred_first_rev_sec is None else float(pred_first_rev_sec),
            "gt_tail_amp": gt_tail_amp,
            "pred_tail_amp": pred_tail_amp,
            "tail_amp_ratio_pred_over_gt": None if abs(gt_tail_amp) < 1e-6 else float(pred_tail_amp / gt_tail_amp),
            "gt_head_amp": head_amp_gt,
            "pred_head_amp": head_amp_pred,
            "head_amp_ratio_pred_over_gt": None if abs(head_amp_gt) < 1e-6 else float(head_amp_pred / head_amp_gt),
            "peak_time_error_sec": float((pred_peak_idx - gt_peak_idx) / fs),
            "onset_delay_sec": None if (gt_onset_idx is None or pred_onset_idx is None) else float((pred_onset_idx - gt_onset_idx) / fs),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_repro_dataset(eval_module, split_mode: str, smoke_max_samples: int, seed: int):
    style_map = eval_module.load_driver_style_map(eval_module.STYLE_CSV)
    X_pool, y_pool, curve_pool, ctx_pool, base_pool, sample_meta_df, feature_names = eval_module.build_all_samples(style_map)
    ordered_idx, n_train, n_val, n_test, split_meta = build_split_indices(
        eval_module,
        sample_meta_df,
        split_mode=split_mode,
        smoke_max_samples=smoke_max_samples,
        seed=seed,
        protocol_config_path=getattr(eval_module, "PROTOCOL_CONFIG_PATH", None),
        frozen_split_path=getattr(eval_module, "FROZEN_SPLIT_PATH", None),
    )

    X_pool = subset_list(X_pool, ordered_idx)
    y_pool = subset_list(y_pool, ordered_idx)
    curve_pool = subset_list(curve_pool, ordered_idx)
    ctx_pool = subset_list(ctx_pool, ordered_idx)
    base_pool = subset_list(base_pool, ordered_idx)
    sample_meta_df = sample_meta_df.iloc[ordered_idx].reset_index(drop=True)

    if split_mode == "protocol_safe":
        sample_meta_df["protocol_split_applied"] = (
            ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        )
    else:
        sample_meta_df["protocol_split_applied"] = ["train"] * n_train + ["test"] * n_test

    train_idx = np.arange(n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64)

    rev_gt_weak = np.array([eval_module.has_reversal_np(y[:, 0], eps=eval_module.REV_EPS_WEAK) for y in y_pool], dtype=np.float32)
    rev_gt_strong = []
    for y in y_pool:
        steer_f = y[:, 0]
        peak_abs = float(np.max(np.abs(steer_f))) if steer_f.size else 0.0
        r = eval_module.has_reversal_np(steer_f, eps=eval_module.REV_EPS_STRONG)
        rev_gt_strong.append(1.0 if (r > 0.5 and peak_abs >= eval_module.STRONG_PEAK_THR) else 0.0)
    rev_gt_strong = np.asarray(rev_gt_strong, dtype=np.float32)
    rev_gt = rev_gt_strong if getattr(eval_module, "USE_STRONG_REV_LOSS", True) else rev_gt_weak

    curve_feat_name, curve_feat_idx = eval_module.find_feature_in_list(feature_names, ["lanecurvature", "curvature"])
    if curve_feat_idx is None:
        curve_scores = np.zeros((len(X_pool),), dtype=np.float32)
        curve_thr = 0.0
        is_curve = np.zeros((len(X_pool),), dtype=np.int64)
    else:
        curve_scores = np.array([float(np.mean(np.abs(x[:, curve_feat_idx]))) for x in X_pool], dtype=np.float32)
        curve_thr = float(eval_module.auto_curve_threshold(curve_scores[train_idx]))
        is_curve = (curve_scores > curve_thr).astype(np.int64)

    all_X_concat = np.concatenate([X_pool[int(i)] for i in train_idx], axis=0)
    feat_mean = all_X_concat.mean(axis=0)
    feat_std = all_X_concat.std(axis=0)
    feat_std[feat_std < 1e-6] = 1e-6
    for i in range(len(X_pool)):
        X_pool[i] = (X_pool[i] - feat_mean) / feat_std

    all_y_concat = np.concatenate([y_pool[int(i)].reshape(-1, 3) for i in train_idx], axis=0)
    y_mean = all_y_concat.mean(axis=0)
    y_std = all_y_concat.std(axis=0)
    y_std[y_std < 1e-6] = 1e-6

    all_curve_concat = np.concatenate([curve_pool[int(i)] for i in train_idx], axis=0)
    curve_mean = float(all_curve_concat.mean())
    curve_std = float(all_curve_concat.std())
    if curve_std < 1e-6:
        curve_std = 1e-6

    ctx_array = np.stack([ctx_pool[int(i)] for i in train_idx], axis=0)
    ctx_mean = ctx_array.mean(axis=0)
    ctx_std = ctx_array.std(axis=0)
    ctx_std[ctx_std < 1e-6] = 1e-6

    base_train = np.stack([base_pool[int(i)] for i in train_idx], axis=0)
    teacher_base_names = [
        "hr",
        "eda_tonic",
        "eda_phasic",
        "emg_rms",
        "alpha_asym",
        "occ_ta_beta",
        "frontal_ta_beta",
        "temporal_ta_beta",
        "occ_alpha_abs",
        "temporal_gamma_rel",
        "occ_gamma_rel",
        "frontal_gamma_rel",
    ]
    finite_count = np.isfinite(base_train).sum(axis=0)
    all_missing_mask = finite_count == 0
    base_mu = np.zeros((base_train.shape[1],), dtype=np.float32)
    base_sd = np.ones((base_train.shape[1],), dtype=np.float32)
    valid_stat_mask = ~all_missing_mask
    if np.any(valid_stat_mask):
        base_mu[valid_stat_mask] = np.nanmean(base_train[:, valid_stat_mask], axis=0).astype(np.float32)
        base_sd[valid_stat_mask] = np.nanstd(base_train[:, valid_stat_mask], axis=0).astype(np.float32)
    base_sd[base_sd < 1e-6] = 1e-6

    def zscore_base(x12):
        x = x12.copy()
        nan_mask = ~np.isfinite(x)
        x[nan_mask] = np.take(base_mu, np.where(nan_mask)[0])
        return (x - base_mu) / base_sd

    base_z_all = np.stack([zscore_base(x) for x in base_pool], axis=0)
    teacher_state_mode = getattr(eval_module, "TEACHER_STATE_MODE", "pca_latent")
    teacher_state_dim = int(getattr(eval_module, "TEACHER_STATE_DIM", 4))
    z_phys_raw, teacher_state_meta = build_teacher_state_with_module(
        eval_module,
        base_z_all,
        mode=teacher_state_mode,
        state_dim=teacher_state_dim,
        n_train=n_train,
    )
    z_tr = z_phys_raw[:n_train]
    z_mu = np.mean(z_tr, axis=0)
    z_sd = np.std(z_tr, axis=0)
    z_sd[z_sd < 1e-6] = 1e-6
    z_phys = ((z_phys_raw - z_mu) / z_sd).astype(np.float32)

    def build_dataset(indices):
        return eval_module.MultiTaskFutureWithCurveDataset(
            subset_list(X_pool, indices),
            subset_list(y_pool, indices),
            subset_list(curve_pool, indices),
            subset_list(ctx_pool, indices),
            subset_array(z_phys, indices),
            subset_array(rev_gt, indices),
            subset_array(rev_gt_weak, indices),
            subset_array(rev_gt_strong, indices),
            y_mean,
            y_std,
            curve_mean,
            curve_std,
            ctx_mean,
            ctx_std,
            subset_array(curve_scores, indices),
            subset_array(is_curve, indices),
        )

    datasets = {
        "train": build_dataset(train_idx),
        "test": build_dataset(test_idx),
    }
    if n_val > 0:
        datasets["val"] = build_dataset(val_idx)

    meta = {
        "split_meta": split_meta,
        "feature_names": feature_names,
        "sample_meta_df": sample_meta_df,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "rev_gt_weak": rev_gt_weak,
        "rev_gt_strong": rev_gt_strong,
        "teacher_state_meta": teacher_state_meta,
        "teacher_base_names": teacher_base_names,
        "curve_thr": curve_thr,
        "curve_feat_name": curve_feat_name,
        "y_mean": y_mean,
        "y_std": y_std,
        "state_dim": int(z_phys.shape[1]),
    }
    return datasets, meta


def instantiate_model(module, input_dim: int, context_dim: int, future_len: int, state_dim: int, device: str, model_config: dict[str, Any] | None = None):
    model_config = model_config or {}
    d_model = int(getattr(module, "D_MODEL", 128))
    nhead = int(getattr(module, "N_HEAD", getattr(module, "NHEAD", 2)))
    num_layers_enc = int(getattr(module, "NUM_LAYERS_ENC", getattr(module, "ENC_LAYERS", 2)))
    num_layers_dec = int(getattr(module, "NUM_LAYERS_DEC", getattr(module, "DEC_LAYERS", 2)))
    dim_feedforward = int(getattr(module, "FFN_DIM", 256))
    dropout = float(getattr(module, "DROPOUT", 0.1))
    max_len_enc = int(getattr(module, "WIN_LEN", 600))
    max_len_dec = int(getattr(module, "FUTURE_LEN", future_len))
    enable_steer_coarse_fine = bool(model_config.get("ENABLE_STEER_COARSE_FINE", getattr(module, "ENABLE_STEER_COARSE_FINE", False)))
    trend_pool_kernel = int(model_config.get("TREND_POOL_KERNEL", getattr(module, "TREND_POOL_KERNEL", 20)))
    trend_pool_stride = int(model_config.get("TREND_POOL_STRIDE", getattr(module, "TREND_POOL_STRIDE", 20)))
    enable_late_reversal_gate = bool(model_config.get("ENABLE_LATE_REV_GATE", getattr(module, "ENABLE_LATE_REV_GATE", False)))
    late_rev_gate_start_sec = float(model_config.get("LATE_REV_GATE_START_SEC", getattr(module, "LATE_REV_GATE_START_SEC", 1.05)))
    late_rev_gate_scale = float(model_config.get("LATE_REV_GATE_SCALE", getattr(module, "LATE_REV_GATE_SCALE", 0.60)))
    late_rev_gate_ramp_power = float(model_config.get("LATE_REV_GATE_RAMP_POWER", getattr(module, "LATE_REV_GATE_RAMP_POWER", 1.50)))
    enable_strong_pos_gate = bool(model_config.get("ENABLE_STRONG_POS_GATE", getattr(module, "ENABLE_STRONG_POS_GATE", False)))
    strong_pos_gate_start_sec = float(model_config.get("STRONG_POS_GATE_START_SEC", getattr(module, "STRONG_POS_GATE_START_SEC", 1.20)))
    strong_pos_gate_scale = float(model_config.get("STRONG_POS_GATE_SCALE", getattr(module, "STRONG_POS_GATE_SCALE", 0.45)))
    strong_pos_gate_ramp_power = float(model_config.get("STRONG_POS_GATE_RAMP_POWER", getattr(module, "STRONG_POS_GATE_RAMP_POWER", 1.75)))
    strong_pos_gate_prob_center = float(model_config.get("STRONG_POS_GATE_PROB_CENTER", getattr(module, "STRONG_POS_GATE_PROB_CENTER", 0.60)))
    model = module.Past2FutureMultiTaskRoadPreview(
        input_dim=input_dim,
        context_dim=context_dim,
        future_len=future_len,
        out_dim=3,
        d_model=d_model,
        nhead=nhead,
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len_enc=max_len_enc,
        max_len_dec=max_len_dec,
        state_dim=state_dim,
        enable_steer_coarse_fine=enable_steer_coarse_fine,
        trend_pool_kernel=trend_pool_kernel,
        trend_pool_stride=trend_pool_stride,
        enable_late_reversal_gate=enable_late_reversal_gate,
        late_rev_gate_start_sec=late_rev_gate_start_sec,
        late_rev_gate_scale=late_rev_gate_scale,
        late_rev_gate_ramp_power=late_rev_gate_ramp_power,
        enable_strong_pos_gate=enable_strong_pos_gate,
        strong_pos_gate_start_sec=strong_pos_gate_start_sec,
        strong_pos_gate_scale=strong_pos_gate_scale,
        strong_pos_gate_ramp_power=strong_pos_gate_ramp_power,
        strong_pos_gate_prob_center=strong_pos_gate_prob_center,
    ).to(device)
    return model


def run_eval(args):
    script_path = Path(args.script_path).resolve()
    metrics_script_path = Path(args.metrics_script_path).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_obj = torch.load(str(checkpoint_path), map_location="cpu")
    model_config = raw_obj.get("config", {}) if isinstance(raw_obj, dict) else {}
    if not model_config:
        run_config_path = checkpoint_path.parent.parent / "run_config.json"
        model_config = try_load_json(run_config_path) or {}
    os.environ["DRIVER_MODEL_STEER_ANGLE_UNIT"] = resolve_steer_angle_unit(model_config)

    eval_module = load_module(script_path, f"eval_module_{abs(hash(str(script_path)))}")
    metrics_module = load_module(metrics_script_path, f"metrics_module_{abs(hash(str(metrics_script_path)))}")

    if args.protocol_config_path:
        setattr(eval_module, "PROTOCOL_CONFIG_PATH", str(Path(args.protocol_config_path).resolve()))
    if args.frozen_split_path:
        setattr(eval_module, "FROZEN_SPLIT_PATH", str(Path(args.frozen_split_path).resolve()))

    seed = int(args.seed if args.seed is not None else getattr(eval_module, "SEED", 2025))
    datasets, meta = build_repro_dataset(
        eval_module,
        split_mode=args.split_mode,
        smoke_max_samples=int(args.smoke_max_samples),
        seed=seed,
    )

    sample_meta_df = meta["sample_meta_df"]
    test_idx = meta["test_idx"]
    test_meta_df = sample_meta_df.iloc[test_idx].reset_index(drop=True)
    test_rev_gt_weak = meta["rev_gt_weak"][test_idx]
    test_rev_gt_strong = meta["rev_gt_strong"][test_idx]
    context_dim = int(datasets["test"].ctx[0].shape[0] + meta["state_dim"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = instantiate_model(
        eval_module,
        input_dim=len(meta["feature_names"]),
        context_dim=context_dim,
        future_len=int(getattr(eval_module, "FUTURE_LEN", 400)),
        state_dim=meta["state_dim"],
        device=device,
        model_config=model_config,
    )

    state_dict = raw_obj["state_dict"] if isinstance(raw_obj, dict) and "state_dict" in raw_obj else raw_obj
    model.load_state_dict(state_dict)
    model.eval()

    test_loader = DataLoader(
        datasets["test"],
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        collate_fn=build_collate_fn(),
        pin_memory=torch.cuda.is_available(),
    )

    y_mean = np.asarray(meta["y_mean"], dtype=np.float32)
    y_std = np.asarray(meta["y_std"], dtype=np.float32)
    pred_norm_all = []
    true_norm_all = []
    batch_local_idx_all = []
    with torch.no_grad():
        for batch in test_loader:
            src = batch["src"].to(device, non_blocking=True)
            y_true = batch["y_norm"].to(device, non_blocking=True)
            curve_norm = batch["curve_norm"].to(device, non_blocking=True)
            ctx = batch["ctx"].to(device, non_blocking=True)
            y_hat, _, _, _ = unpack_model_output(model(src, ctx, curve_norm))
            pred_norm_all.append(y_hat.detach().cpu().numpy())
            true_norm_all.append(y_true.detach().cpu().numpy())
            batch_local_idx_all.append(batch["idx"].detach().cpu().numpy())

    pred_norm = np.concatenate(pred_norm_all, axis=0)
    true_norm = np.concatenate(true_norm_all, axis=0)
    batch_local_idx = np.concatenate(batch_local_idx_all, axis=0)
    pred = metrics_module._denorm_y(pred_norm, y_mean, y_std)
    true = metrics_module._denorm_y(true_norm, y_mean, y_std)

    basic_metrics = compute_basic_metrics(pred, true)
    head_metrics = metrics_module._head_metrics(pred, true, fs=int(getattr(eval_module, "FS", 200)))
    tail_metrics = metrics_module._tail_metrics(pred, true, fs=int(getattr(eval_module, "FS", 200)))
    peak_metrics = metrics_module._peak_metrics(pred, true, fs=int(getattr(eval_module, "FS", 200)))
    trend_metrics = (
        metrics_module._trend_metrics(pred, true, fs=int(getattr(eval_module, "FS", 200)))
        if hasattr(metrics_module, "_trend_metrics")
        else None
    )
    reversal_metrics = metrics_module._structured_reversal_metrics(
        pred,
        true,
        rev_gt_weak_vec=test_rev_gt_weak,
        rev_gt_strong_vec=test_rev_gt_strong,
        fs=int(getattr(eval_module, "FS", 200)),
    )

    batch_order_meta = test_meta_df.iloc[batch_local_idx].reset_index(drop=True)
    case_df = build_case_rows(metrics_module, pred, true, batch_order_meta, fs=int(getattr(eval_module, "FS", 200)))
    case_df["combined_bad_score"] = (
        case_df["onset_delay_sec"].fillna(0.0).clip(lower=0.0)
        + case_df["peak_time_error_sec"].abs()
        + (1.0 - case_df["tail_amp_ratio_pred_over_gt"].fillna(0.0))
    )
    case_df = case_df.sort_values(
        by=["combined_bad_score", "tail_amp_ratio_pred_over_gt"],
        ascending=[False, True],
    ).reset_index(drop=True)

    summary = {
        **basic_metrics,
        "head_metrics": head_metrics,
        "tail_metrics": tail_metrics,
        "peak_metrics": peak_metrics,
        "trend_metrics": trend_metrics,
        "reversal_structure_metrics": reversal_metrics,
        "recalc_config": {
            "script_path": str(script_path),
            "metrics_script_path": str(metrics_script_path),
            "checkpoint_path": str(checkpoint_path),
            "split_mode": args.split_mode,
            "smoke_max_samples": None if args.split_mode == "protocol_safe" else int(args.smoke_max_samples),
            "seed": int(seed),
            "curve_feat_name": meta["curve_feat_name"],
            "curve_threshold": float(meta["curve_thr"]),
            "split_meta": meta["split_meta"],
        },
    }

    prefix = args.output_prefix.strip()
    summary_path = output_dir / f"{prefix}_summary.json"
    cases_path = output_dir / f"{prefix}_cases.csv"
    top_cases_path = output_dir / f"{prefix}_top_bad_cases.csv"
    save_json(summary_path, summary)
    case_df.to_csv(cases_path, index=False, encoding="utf-8-sig")
    case_df.head(int(args.top_k)).to_csv(top_cases_path, index=False, encoding="utf-8-sig")

    print(json.dumps({"summary_path": str(summary_path), "cases_path": str(cases_path), "top_cases_path": str(top_cases_path)}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Recalculate v5.8 checkpoint metrics with current head/tail/peak/reversal definitions.")
    parser.add_argument("--script-path", required=True, help="Training script used by the target run.")
    parser.add_argument("--metrics-script-path", required=True, help="Script exposing current metric functions.")
    parser.add_argument("--checkpoint-path", required=True, help="Checkpoint path (.pth).")
    parser.add_argument("--output-dir", required=True, help="Directory for summary/case outputs.")
    parser.add_argument("--output-prefix", required=True, help="Prefix for generated output files.")
    parser.add_argument("--split-mode", choices=["protocol_safe", "smoke_random80"], required=True)
    parser.add_argument("--protocol-config-path", default=None)
    parser.add_argument("--frozen-split-path", default=None)
    parser.add_argument("--smoke-max-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
