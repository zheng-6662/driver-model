#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v310 severe-error targeted curve model.

目的：
- 在不使用 test severe 错例做训练/选模的前提下，针对 v309 暴露出的三类严重错误做一次小步改造：
  1. 方向峰值反向；
  2. 真实近似无大动作但模型预测大动作；
  3. 真实极端动作被模型大幅低估。
- 训练仍然只使用 train split；候选选择仍然只看 validation。
- v309 的严重错误表只用于训练结束后的诊断分组，不能进入训练 loss 或 validation 选择。

方法：
- 复用 v307 的 coarse-scene conditioned decoder 和 v307 selected checkpoint 初始化；
- 在 train/val 目标曲线上构造 target-shape 权重；
- 在原 curve loss 之外加入三个轻量形状约束：
  - direction loss：真实峰值方向明确时，惩罚预测在真实峰值时刻反向；
  - amplitude-under loss：真实极端动作时，惩罚峰值幅值严重低估；
  - flat false-large loss：真实近似平直时，惩罚预测凭空大幅摆动。
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import pickle
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


SEED = 20260704
ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "03_baselines"
SCRIPTS = BASELINES / "scripts"

V307_SCRIPT = SCRIPTS / "stage03_v307_coarse_scene_label_conditioned_curve_model_20260704.py"
V307_OUT = BASELINES / "v307_coarse_scene_label_conditioned_curve_model_20260704"
V307_PRED = V307_OUT / "v307_coarse_scene_label_conditioned_predictions.npz"
V307_GUARDRAIL = V307_OUT / "logs" / "guardrail_check.json"
V309_SEVERE = (
    BASELINES
    / "v309_recent_best_prediction_effect_gallery_20260704"
    / "tables"
    / "v309_severe_direction_or_intent_errors.csv"
)

OUT = BASELINES / "v310_severe_error_targeted_curve_model_20260704"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
MODELS = OUT / "models"


def import_module_from_path(module_name: str, path: Path):
    """按路径导入前序脚本，复用已经通过审计的数据构造和评估口径。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少依赖脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V307 = import_module_from_path("stage03_v307_for_v310_severe_targeted", V307_SCRIPT)
V304 = V307.V304
FUTURE_GRID = V307.FUTURE_GRID


def patch_output_globals() -> None:
    """让复用的 v307/v304 helper 写入 v310 输出目录。"""

    V307.SEED = SEED
    V307.OUT = OUT
    V307.TABLES = TABLES
    V307.FIGURES = FIGURES
    V307.REPORTS = REPORTS
    V307.LOGS = LOGS
    V307.MODELS = MODELS
    V307.patch_v304_output_globals()


def ensure_dirs() -> None:
    """创建 v310 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v310 自己的输出。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 使用 utf-8-sig，方便 Windows/Excel 直接查看中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, object], path: Path) -> None:
    """保存 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """计算输入或产物文件哈希。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    """固定随机种子，保证可复跑。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def load_v307_prediction_all(manifest: pd.DataFrame) -> Tuple[np.ndarray, str, Dict[str, object]]:
    """读取 v307 selected 预测，并校验顺序与当前 manifest 一致。"""

    if not V307_PRED.exists():
        raise FileNotFoundError(f"缺少 v307 预测文件：{V307_PRED}")
    with np.load(V307_PRED, allow_pickle=True) as z:
        pred = z["pred_v307_selected"].astype(np.float32)
        selected = str(z["best_v307_model"][0])
        event_uid = z["event_uid"].astype(str)
        delay_ms = z["delay_ms"].astype(int)
    if not np.array_equal(manifest["event_uid"].astype(str).to_numpy(), event_uid):
        raise AssertionError("v307 event_uid 与当前 manifest 不一致")
    if not np.array_equal(manifest["delay_ms"].astype(int).to_numpy(), delay_ms):
        raise AssertionError("v307 delay_ms 与当前 manifest 不一致")
    guard = json.loads(V307_GUARDRAIL.read_text(encoding="utf-8")) if V307_GUARDRAIL.exists() else {}
    return pred, selected, guard


def signed_peak(curve: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算每个样本在 valid 范围内的有符号绝对峰值。"""

    masked_abs = np.where(valid, np.abs(curve), -1.0)
    idx = np.argmax(masked_abs, axis=1)
    peak = curve[np.arange(curve.shape[0]), idx]
    peak_abs = np.abs(peak)
    return peak, peak_abs


def build_target_shape_multiplier(
    prepared,
    y_true_curve: np.ndarray,
    base_hard_event_extra: float,
    target_shape_extra: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    从 train/val/test 各自的目标曲线形态生成 loss 权重。

    这里使用的是当前样本自己的训练目标形态，不使用 v309 test 错误标签；
    对 test 行生成 multiplier 只是为了完整审计，预测和 test 指标不会使用它。
    """

    manifest = prepared.data.manifest.reset_index(drop=True)
    n_samples = int(len(manifest))
    n_steps = int(y_true_curve.shape[1])
    valid = prepared.prepared.point_data.valid_original_remaining_all.reshape(n_samples, n_steps).astype(bool)
    peak, peak_abs = signed_peak(y_true_curve, valid)

    base = V304.build_curve_sample_multiplier(
        manifest,
        prepared.event_label_name,
        hard_event_extra=float(base_hard_event_extra),
    ).astype(np.float32)
    vehicle_strong = (
        manifest["vehicle_strong"].astype(bool).to_numpy()
        if "vehicle_strong" in manifest.columns
        else np.zeros(n_samples, dtype=bool)
    )
    zero_cross = (
        manifest["zero_cross"].astype(bool).to_numpy()
        if "zero_cross" in manifest.columns
        else np.zeros(n_samples, dtype=bool)
    )
    extreme_peak_flag = (
        manifest["extreme_peak"].astype(bool).to_numpy()
        if "extreme_peak" in manifest.columns
        else np.zeros(n_samples, dtype=bool)
    )

    extreme_target = peak_abs >= 2.0
    flat_vehicle_risk = (peak_abs < 0.40) & vehicle_strong
    direction_fragile = (peak_abs >= 0.40) & zero_cross
    manifest_extreme = extreme_peak_flag & (peak_abs >= 1.0)

    mult = base.copy()
    mult += float(target_shape_extra) * extreme_target.astype(np.float32) * 1.00
    mult += float(target_shape_extra) * flat_vehicle_risk.astype(np.float32) * 0.90
    mult += float(target_shape_extra) * direction_fragile.astype(np.float32) * 0.55
    mult += float(target_shape_extra) * manifest_extreme.astype(np.float32) * 0.35
    mult = np.clip(mult, 0.50, 2.80).astype(np.float32)

    audit = manifest[
        [
            "event_uid",
            "split",
            "delay_ms",
            "scene_type",
            "route_event",
            "strong_steer",
            "vehicle_strong",
            "zero_cross",
            "extreme_peak",
        ]
    ].copy()
    audit["coarse_scene_label"] = prepared.event_label_name.astype(str)
    audit["true_peak_signed"] = peak.astype(float)
    audit["true_peak_abs"] = peak_abs.astype(float)
    audit["extreme_target"] = extreme_target.astype(int)
    audit["flat_vehicle_risk"] = flat_vehicle_risk.astype(int)
    audit["direction_fragile"] = direction_fragile.astype(int)
    audit["manifest_extreme"] = manifest_extreme.astype(int)
    audit["base_multiplier"] = base.astype(float)
    audit["target_shape_multiplier"] = mult.astype(float)
    return mult, audit


def shape_guard_loss(
    pred_scaled: torch.Tensor,
    target_scaled: torch.Tensor,
    valid: torch.Tensor,
    weight_seq: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    direction_weight: float,
    amplitude_weight: float,
    flat_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    三个形状约束都在原始 steering_delta 空间计算。

    pred_scaled/target_scaled 是模型训练空间；转换回原始空间后再判断方向和幅值，
    避免阈值受 scaler 影响。
    """

    pred = pred_scaled * y_std + y_mean
    target = target_scaled * y_std + y_mean
    valid_bool = valid > 0.5
    masked_abs = torch.where(valid_bool, torch.abs(target), torch.full_like(target, -1.0))
    peak_idx = torch.argmax(masked_abs, dim=1)
    batch_idx = torch.arange(target.shape[0], device=target.device)
    true_peak = target[batch_idx, peak_idx]
    pred_at_true_peak = pred[batch_idx, peak_idx]
    true_abs = torch.abs(true_peak)
    true_sign = torch.sign(true_peak)
    sample_weight = torch.sum(valid * weight_seq, dim=1) / torch.clamp(torch.sum(valid, dim=1), min=1.0)

    def weighted_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = sample_weight * mask.float()
        return torch.sum(values * w) / torch.clamp(torch.sum(w), min=1e-6)

    direction_mask = true_abs >= 0.40
    signed_pred = pred_at_true_peak * true_sign
    direction_loss = weighted_mean(torch.square(torch.relu(0.10 - signed_pred)), direction_mask)

    amplitude_mask = true_abs >= 1.50
    amplitude_floor = 0.60 * true_abs
    amplitude_loss = weighted_mean(torch.square(torch.relu(amplitude_floor - signed_pred)), amplitude_mask)

    true_max_abs = torch.max(torch.where(valid_bool, torch.abs(target), torch.zeros_like(target)), dim=1).values
    flat_mask = true_max_abs < 0.40
    pred_energy = torch.sum(torch.square(pred) * valid, dim=1) / torch.clamp(torch.sum(valid, dim=1), min=1.0)
    flat_loss = weighted_mean(pred_energy, flat_mask)

    total = (
        float(direction_weight) * direction_loss
        + float(amplitude_weight) * amplitude_loss
        + float(flat_weight) * flat_loss
    )
    stats = {
        "shape_loss": float(total.detach().cpu().item()),
        "direction_loss": float(direction_loss.detach().cpu().item()),
        "amplitude_loss": float(amplitude_loss.detach().cpu().item()),
        "flat_loss": float(flat_loss.detach().cpu().item()),
        "direction_case_rate": float(direction_mask.float().mean().detach().cpu().item()),
        "amplitude_case_rate": float(amplitude_mask.float().mean().detach().cpu().item()),
        "flat_case_rate": float(flat_mask.float().mean().detach().cpu().item()),
    }
    return total, stats


def run_epoch_v310(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    smooth_weight: float,
    aux_weight: float,
    class_weight: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    direction_weight: float,
    amplitude_weight: float,
    flat_weight: float,
) -> Dict[str, float]:
    """运行一个 v310 训练或验证 epoch。"""

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_curve = 0.0
    total_aux = 0.0
    total_shape = 0.0
    total_direction = 0.0
    total_amplitude = 0.0
    total_flat = 0.0
    total_weight = 0.0
    total_samples = 0
    correct = 0
    for batch in loader:
        hist = batch["hist"].to(device=device, dtype=torch.float32)
        road = batch["road"].to(device=device, dtype=torch.float32)
        phase = batch["phase"].to(device=device, dtype=torch.float32)
        point_seq = batch["point_seq"].to(device=device, dtype=torch.float32)
        roll = batch["roll"].to(device=device, dtype=torch.float32)
        event_label = batch["event_label"].to(device=device, dtype=torch.long)
        y_seq = batch["y_seq"].to(device=device, dtype=torch.float32)
        valid_seq = batch["valid_seq"].to(device=device, dtype=torch.float32)
        weight_seq = batch["weight_seq"].to(device=device, dtype=torch.float32)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
        pred, logits = model(hist, road, phase, point_seq, roll, event_label)
        curve_loss = V304.V242.masked_curve_loss(pred, y_seq, valid_seq, weight_seq, smooth_weight)
        aux_loss = F.cross_entropy(logits, event_label, weight=class_weight)
        shape_loss, shape_stats = shape_guard_loss(
            pred,
            y_seq,
            valid_seq,
            weight_seq,
            y_mean,
            y_std,
            direction_weight,
            amplitude_weight,
            flat_weight,
        )
        loss = curve_loss + float(aux_weight) * aux_loss + shape_loss
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        wsum = float(torch.sum(valid_seq * weight_seq).detach().cpu().item())
        bsz = int(event_label.shape[0])
        total_loss += float(loss.detach().cpu().item()) * wsum
        total_curve += float(curve_loss.detach().cpu().item()) * wsum
        total_aux += float(aux_loss.detach().cpu().item()) * bsz
        total_shape += shape_stats["shape_loss"] * wsum
        total_direction += shape_stats["direction_loss"] * wsum
        total_amplitude += shape_stats["amplitude_loss"] * wsum
        total_flat += shape_stats["flat_loss"] * wsum
        total_weight += wsum
        total_samples += bsz
        correct += int((torch.argmax(logits, dim=1) == event_label).sum().detach().cpu().item())
    return {
        "loss": total_loss / max(total_weight, 1e-6),
        "curve_loss": total_curve / max(total_weight, 1e-6),
        "aux_loss": total_aux / max(total_samples, 1),
        "shape_loss": total_shape / max(total_weight, 1e-6),
        "direction_loss": total_direction / max(total_weight, 1e-6),
        "amplitude_loss": total_amplitude / max(total_weight, 1e-6),
        "flat_loss": total_flat / max(total_weight, 1e-6),
        "event_acc": correct / max(total_samples, 1),
    }


def load_v307_checkpoint_state(model: nn.Module, checkpoint_path: Path) -> Dict[str, object]:
    """从 v307 selected checkpoint 初始化同结构模型。"""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"缺少 v307 checkpoint：{checkpoint_path}")
    # 该 checkpoint 由本项目 v307 脚本在本机生成，里面包含 numpy 标量和配置字典；
    # PyTorch 2.6 默认 weights_only=True 会拒绝这些对象，因此这里显式按可信本地文件加载。
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        raise AssertionError(f"v307 checkpoint 结构不匹配：missing={missing}, unexpected={unexpected}")
    return {
        "init_from_v307": True,
        "v307_checkpoint": str(checkpoint_path),
        "v307_model_name": str(ckpt.get("model_name", checkpoint_path.stem)),
        "v307_best_epoch": int(ckpt.get("best_epoch", -1)),
    }


def train_v310_candidate(
    model_name: str,
    config: Dict[str, object],
    prepared,
    device: torch.device,
    seed: int,
    v307_checkpoint: Path,
):
    """训练一个 v310 候选。"""

    set_seed(seed)
    sample_masks = prepared.prepared.sample_masks
    train_dataset = V304.RollAuxCurveDataset(
        prepared.prepared.arrays,
        prepared.prepared.point_data,
        sample_masks["train"],
        prepared.roll_scaled,
        prepared.event_label,
        prepared.curve_sample_multiplier,
    )
    val_dataset = V304.RollAuxCurveDataset(
        prepared.prepared.arrays,
        prepared.prepared.point_data,
        sample_masks["val"],
        prepared.roll_scaled,
        prepared.event_label,
        prepared.curve_sample_multiplier,
    )
    batch_size = int(config["batch_size"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = V304.FixedEventConditionedDecoder(
        hist_dim=prepared.data.x_hist.shape[-1],
        road_dim=prepared.data.x_road.shape[-1],
        phase_dim=prepared.data.x_phase.shape[-1],
        point_dim=len(V304.V238.POINT_EXTRA_FEATURE_NAMES),
        roll_dim=prepared.roll_scaled.shape[1],
        class_n=len(prepared.class_names),
        hist_len=prepared.data.x_hist.shape[1],
        road_len=prepared.data.x_road.shape[1],
        n_steps=len(FUTURE_GRID),
        hidden_dim=int(config["hidden_dim"]),
        n_heads=int(config["n_heads"]),
        n_layers=int(config["n_layers"]),
        mixer_layers=int(config["mixer_layers"]),
        mlp_hidden=int(config["mlp_hidden"]),
        roll_hidden=int(config["roll_hidden"]),
        event_embed_dim=int(config["event_embed_dim"]),
        dropout=float(config["dropout"]),
        film_scale=float(config["film_scale"]),
    )
    init_info = load_v307_checkpoint_state(model, v307_checkpoint)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, int(config["patience"]) // 3),
        min_lr=float(config["min_lr"]),
    )
    class_weight = torch.from_numpy(prepared.class_weight).to(device=device, dtype=torch.float32)
    y_mean = torch.as_tensor(prepared.prepared.scalers.y_mean, device=device, dtype=torch.float32)
    y_std = torch.as_tensor(prepared.prepared.scalers.y_std, device=device, dtype=torch.float32)

    max_epochs = int(config["max_epochs"])
    patience = int(config["patience"])
    best_val = math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        train_stat = run_epoch_v310(
            model,
            train_loader,
            device,
            optimizer,
            smooth_weight=float(config["smooth_weight"]),
            aux_weight=float(config["aux_weight"]),
            class_weight=class_weight,
            y_mean=y_mean,
            y_std=y_std,
            direction_weight=float(config["direction_weight"]),
            amplitude_weight=float(config["amplitude_weight"]),
            flat_weight=float(config["flat_weight"]),
        )
        val_stat = run_epoch_v310(
            model,
            val_loader,
            device,
            None,
            smooth_weight=float(config["smooth_weight"]),
            aux_weight=float(config["aux_weight"]),
            class_weight=class_weight,
            y_mean=y_mean,
            y_std=y_std,
            direction_weight=float(config["direction_weight"]),
            amplitude_weight=float(config["amplitude_weight"]),
            flat_weight=float(config["flat_weight"]),
        )
        scheduler.step(val_stat["loss"])
        history.append(
            {
                "model_name": model_name,
                "epoch": epoch,
                "train_loss": train_stat["loss"],
                "train_curve_loss": train_stat["curve_loss"],
                "train_aux_loss": train_stat["aux_loss"],
                "train_shape_loss": train_stat["shape_loss"],
                "train_direction_loss": train_stat["direction_loss"],
                "train_amplitude_loss": train_stat["amplitude_loss"],
                "train_flat_loss": train_stat["flat_loss"],
                "train_event_acc": train_stat["event_acc"],
                "val_loss": val_stat["loss"],
                "val_curve_loss": val_stat["curve_loss"],
                "val_aux_loss": val_stat["aux_loss"],
                "val_shape_loss": val_stat["shape_loss"],
                "val_direction_loss": val_stat["direction_loss"],
                "val_amplitude_loss": val_stat["amplitude_loss"],
                "val_flat_loss": val_stat["flat_loss"],
                "val_event_acc": val_stat["event_acc"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                **init_info,
            }
        )
        if val_stat["loss"] < best_val - 1e-5:
            best_val = val_stat["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise AssertionError(f"{model_name} 没有生成 best_state")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    pred_curve, logits = V304.predict_curves_and_logits(
        model,
        prepared.prepared.arrays,
        prepared.roll_scaled,
        prepared.event_label,
        prepared.prepared.scalers,
        device,
        batch_size=batch_size * 4,
    )
    return V304.V304Run(
        model_name=model_name,
        config=config,
        state_dict=best_state,
        pred_curve=pred_curve.astype(np.float32),
        event_logits=logits.astype(np.float32),
        event_pred_class=np.argmax(logits, axis=1).astype(np.int64),
        training_history=pd.DataFrame(history),
        training_seconds=float(time.time() - start_time),
        best_epoch=int(best_epoch),
        best_val_loss=float(best_val),
    )


def metric_row(summary: pd.DataFrame, model_name: str, split: str, group: str) -> pd.Series | None:
    """从 delay0 summary 取一行。"""

    one = summary[
        summary["model_name"].astype(str).eq(model_name)
        & summary["split"].astype(str).eq(split)
        & summary["group"].astype(str).eq(group)
    ]
    if one.empty:
        return None
    return one.iloc[0]


def summarize_delay0_group(per_sample: pd.DataFrame, model_name: str, event_uids: set[str], group_name: str) -> Dict[str, object]:
    """汇总 delay0 事件集合的 per-sample 指标。"""

    one = per_sample[
        per_sample["model_name"].astype(str).eq(model_name)
        & per_sample["delay_ms"].astype(int).eq(0)
        & per_sample["event_uid"].astype(str).isin(event_uids)
    ].copy()
    if one.empty:
        return {
            "model_name": model_name,
            "diagnostic_group": group_name,
            "n": 0,
            "sample_rmse_mean": math.nan,
            "sample_rmse_median": math.nan,
            "tail_rmse_mean": math.nan,
            "direction_acc": math.nan,
            "strong_under_rate": math.nan,
            "peak_ratio_mean": math.nan,
        }
    return {
        "model_name": model_name,
        "diagnostic_group": group_name,
        "n": int(len(one)),
        "sample_rmse_mean": float(one["sample_rmse"].astype(float).mean()),
        "sample_rmse_median": float(one["sample_rmse"].astype(float).median()),
        "tail_rmse_mean": float(one["tail_rmse"].astype(float).mean()),
        "direction_acc": float(one["direction_ok"].astype(bool).mean()),
        "strong_under_rate": float(one["strong_under"].astype(bool).mean()),
        "peak_ratio_mean": float(one["peak_ratio"].astype(float).replace([np.inf, -np.inf], np.nan).mean()),
    }


def build_v309_severe_diagnostics(per_sample: pd.DataFrame, model_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    用 v309 severe 表生成 test-only 诊断组。

    注意：这一步在训练和 validation 选模之后执行，只用于回答“差样本有没有改善”。
    """

    if not V309_SEVERE.exists():
        raise FileNotFoundError(f"缺少 v309 severe 表：{V309_SEVERE}")
    severe = pd.read_csv(V309_SEVERE, encoding="utf-8-sig")
    severe["event_uid"] = severe["event_uid"].astype(str)
    severe["error_tags"] = severe["error_tags"].astype(str)
    groups = {
        "v309_severe_all37": set(severe["event_uid"]),
        "user_screenshot_5": set(severe.loc[severe["error_tags"].str.contains("shown_in_user_screenshot"), "event_uid"]),
        "opposite_peak_direction": set(severe.loc[severe["error_tags"].str.contains("opposite_peak_direction"), "event_uid"]),
        "false_large_maneuver": set(severe.loc[severe["error_tags"].str.contains("false_large_maneuver"), "event_uid"]),
        "missed_extreme_amplitude": set(severe.loc[severe["error_tags"].str.contains("missed_extreme_amplitude"), "event_uid"]),
        "regression_vs_v300": set(severe.loc[severe["error_tags"].str.contains("regression_vs_v300"), "event_uid"]),
    }
    rows = []
    for group_name, event_uids in groups.items():
        for model_name in model_names:
            rows.append(summarize_delay0_group(per_sample, model_name, event_uids, group_name))
    summary = pd.DataFrame(rows)

    keep_cols = [
        "severe_rank",
        "screenshot_rank",
        "gallery_rank",
        "event_uid",
        "coarse_scene_label_cn",
        "v307_rmse",
        "v300_rmse",
        "delta_v307_minus_v300",
        "true_peak",
        "v307_peak",
        "error_tags",
        "error_reason_cn",
    ]
    keep_cols = [c for c in keep_cols if c in severe.columns]
    severe_keep = severe[keep_cols].copy()
    rows = []
    delay0 = per_sample[per_sample["delay_ms"].astype(int).eq(0)].copy()
    for _, sev in severe_keep.iterrows():
        event_uid = str(sev["event_uid"])
        for model_name in model_names:
            one = delay0[
                delay0["model_name"].astype(str).eq(model_name) & delay0["event_uid"].astype(str).eq(event_uid)
            ]
            if one.empty:
                continue
            row = sev.to_dict()
            metric = one.iloc[0]
            row.update(
                {
                    "model_name": model_name,
                    "sample_rmse": float(metric["sample_rmse"]),
                    "tail_rmse": float(metric["tail_rmse"]),
                    "peak_ratio": float(metric["peak_ratio"]),
                    "direction_ok": bool(metric["direction_ok"]),
                    "strong_under": bool(metric["strong_under"]),
                }
            )
            rows.append(row)
    event_compare = pd.DataFrame(rows)
    return summary, event_compare


def plot_v309_severe_group_bars(summary: pd.DataFrame, selected_name: str, v307_name: str, v300_name: str) -> Path:
    """画严重诊断组 RMSE 对比图。"""

    groups = ["v309_severe_all37", "user_screenshot_5", "opposite_peak_direction", "false_large_maneuver", "missed_extreme_amplitude"]
    models = [v300_name, v307_name, selected_name]
    one = summary[summary["diagnostic_group"].isin(groups) & summary["model_name"].isin(models)].copy()
    pivot = one.pivot(index="diagnostic_group", columns="model_name", values="sample_rmse_mean").reindex(groups)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(groups))
    width = 0.25
    colors = {v300_name: "#f97316", v307_name: "#2563eb", selected_name: "#16a34a"}
    for j, model_name in enumerate(models):
        values = pivot[model_name].to_numpy(dtype=float)
        ax.bar(x + (j - 1) * width, values, width=width, label=model_name, color=colors.get(model_name))
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel("delay0 sample RMSE mean")
    ax.set_title("v310 diagnostic only: v309 severe groups")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGURES / "v310_v309_severe_group_rmse.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_report(
    selection: pd.DataFrame,
    delay0_summary: pd.DataFrame,
    severe_summary: pd.DataFrame,
    severe_event_compare: pd.DataFrame,
    guardrail: Dict[str, object],
    selected_name: str,
    v307_name: str,
    v300_name: str,
) -> Path:
    """写 v310 中文报告。"""

    path = REPORTS / "v310_severe_error_targeted_curve_model_cn.md"

    def group_value(model_name: str, group: str) -> str:
        row = metric_row(delay0_summary, model_name, "test", group)
        if row is None:
            return "NA"
        return f"{float(row['sample_rmse_mean']):.4f}"

    selected_rows = delay0_summary[
        delay0_summary["model_name"].isin([v300_name, v307_name, selected_name])
        & delay0_summary["split"].eq("test")
        & delay0_summary["group"].isin(["all", "within_bad_top10", "within_bad_top20", "strong_steer", "vehicle_ambiguous"])
    ][["model_name", "split", "group", "n", "sample_rmse_mean", "sample_rmse_median", "sample_rmse_p90"]]
    severe_show = severe_summary[
        severe_summary["model_name"].isin([v300_name, v307_name, selected_name])
        & severe_summary["diagnostic_group"].isin(
            [
                "v309_severe_all37",
                "user_screenshot_5",
                "opposite_peak_direction",
                "false_large_maneuver",
                "missed_extreme_amplitude",
            ]
        )
    ].copy()
    screenshot = severe_event_compare[
        severe_event_compare["error_tags"].astype(str).str.contains("shown_in_user_screenshot")
        & severe_event_compare["model_name"].isin([v300_name, v307_name, selected_name])
    ].copy()
    lines = [
        "# v310 severe-error targeted curve model",
        "",
        "## 这一步做了什么",
        "",
        "v310 针对 v309 图册中暴露的严重方向/意图错误做小步改造，但没有把 test severe 错例拿去训练或选模。",
        "",
        "- 训练初始化：v307 selected checkpoint。",
        "- 训练改动：train/val 目标曲线形态权重 + 方向/幅值/平直三类轻量形状约束。",
        "- 选模规则：仍然只看 validation，不看 test。",
        "- v309 severe 表用途：训练结束后诊断 hard cases，不参与训练和选模。",
        "",
        "## validation-only 选择",
        "",
        selection.to_markdown(index=False),
        "",
        f"validation 选出的 v310 候选：`{selected_name}`。",
        f"v307 参照模型：`{v307_name}`。",
        f"v300 参照模型：`{v300_name}`。",
        "",
        "## test delay0 常规分组",
        "",
        selected_rows.to_markdown(index=False),
        "",
        "简表：",
        "",
        f"- test/all：v300 `{group_value(v300_name, 'all')}` -> v307 `{group_value(v307_name, 'all')}` -> v310 `{group_value(selected_name, 'all')}`",
        f"- test/within_bad_top10：v300 `{group_value(v300_name, 'within_bad_top10')}` -> v307 `{group_value(v307_name, 'within_bad_top10')}` -> v310 `{group_value(selected_name, 'within_bad_top10')}`",
        f"- test/within_bad_top20：v300 `{group_value(v300_name, 'within_bad_top20')}` -> v307 `{group_value(v307_name, 'within_bad_top20')}` -> v310 `{group_value(selected_name, 'within_bad_top20')}`",
        "",
        "## v309 severe 诊断分组",
        "",
        severe_show.to_markdown(index=False),
        "",
        "## 用户截图 5 个事件逐模型对比",
        "",
        screenshot.to_markdown(index=False),
        "",
        "## 当前判断",
        "",
        "- 如果 v310 在 `v309_severe_all37` 或 `user_screenshot_5` 上下降，但常规 test/all 明显变差，说明 hard-case 约束过强，不能直接作为主线。",
        "- 如果 v310 常规 test/all 基本持平，同时 severe 组改善，才值得作为下一轮主线。",
        "- 如果 severe 组没有改善，说明单纯 loss/权重不足，需要回到事件标签或候选轨迹结构，而不是继续加权。",
        "",
        "## guardrail",
        "",
        "```json",
        json.dumps(guardrail, ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_file_inventory() -> pd.DataFrame:
    """记录输出文件清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)})
    inv = pd.DataFrame(rows)
    write_csv(inv, LOGS / "file_inventory.csv")
    return inv


def make_zip_package() -> Tuple[Path, bool]:
    """打包并校验 v310 产物。"""

    zip_path = OUT / "v310_severe_error_targeted_curve_model_20260704.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path == zip_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(zip_path, "r") as zf:
        ok = zf.testzip() is None
    return zip_path, ok


def main() -> None:
    start_time = time.time()
    patch_output_globals()
    clean_out_dir()
    torch.set_num_threads(1)
    set_seed(SEED)

    print("[v310] 读取 v307 数据构造和 coarse-scene 条件输入")
    prepared_base = V307.prepare_v307_data(hard_event_extra=0.0)
    split_audit = V304.V300.build_split_audit(prepared_base.data.manifest, prepared_base.event_table)
    write_csv(split_audit, TABLES / "v310_within_subject_split_audit.csv")

    y_true_curve = prepared_base.data.y_future[:, :, 0].astype(np.float32)
    pred_v300, v300_name, v300_guard = V304.load_v300_prediction_all(prepared_base.data.manifest)
    pred_v307, v307_name, v307_guard = load_v307_prediction_all(prepared_base.data.manifest)
    v307_ckpt = V307_OUT / "models" / f"{v307_name}.pt"

    input_hashes = pd.DataFrame(
        [
            {"input_name": "v307_script_reused", "path": str(V307_SCRIPT), "sha256": file_sha256(V307_SCRIPT)},
            {"input_name": "v307_predictions", "path": str(V307_PRED), "sha256": file_sha256(V307_PRED)},
            {"input_name": "v307_selected_checkpoint", "path": str(v307_ckpt), "sha256": file_sha256(v307_ckpt)},
            {"input_name": "v309_severe_diagnostic_table", "path": str(V309_SEVERE), "sha256": file_sha256(V309_SEVERE)},
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v310] 使用设备：{device}")

    base_config = {
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 3,
        "mixer_layers": 2,
        "mlp_hidden": 112,
        "roll_hidden": 128,
        "event_embed_dim": 64,
        "dropout": 0.06,
        "film_scale": 0.05,
        "smooth_weight": 0.02,
        "aux_weight": 0.03,
        "batch_size": 384,
        "min_lr": 5e-6,
        "weight_decay": 3e-4,
        "max_epochs": 34,
        "patience": 7,
    }
    configs: List[Tuple[str, Dict[str, object]]] = []
    for suffix, overrides in [
        (
            "lo",
            {
                "lr": 6e-5,
                "base_hard_event_extra": 0.15,
                "target_shape_extra": 0.20,
                "direction_weight": 0.015,
                "amplitude_weight": 0.030,
                "flat_weight": 0.012,
            },
        ),
        (
            "mid",
            {
                "lr": 6e-5,
                "base_hard_event_extra": 0.25,
                "target_shape_extra": 0.40,
                "direction_weight": 0.030,
                "amplitude_weight": 0.060,
                "flat_weight": 0.025,
            },
        ),
        (
            "hi",
            {
                "lr": 5e-5,
                "base_hard_event_extra": 0.35,
                "target_shape_extra": 0.65,
                "direction_weight": 0.050,
                "amplitude_weight": 0.100,
                "flat_weight": 0.040,
            },
        ),
    ]:
        cfg = dict(base_config)
        cfg.update(overrides)
        configs.append((f"v310_v307init_shape_guard_{suffix}", cfg))

    runs = []
    multiplier_audits = []
    for idx, (model_name, config) in enumerate(configs):
        prepared = copy.copy(prepared_base)
        mult, audit = build_target_shape_multiplier(
            prepared_base,
            y_true_curve,
            base_hard_event_extra=float(config["base_hard_event_extra"]),
            target_shape_extra=float(config["target_shape_extra"]),
        )
        prepared.curve_sample_multiplier = mult
        audit["model_name"] = model_name
        audit["base_hard_event_extra"] = float(config["base_hard_event_extra"])
        audit["target_shape_extra"] = float(config["target_shape_extra"])
        multiplier_audits.append(audit)
        print(
            "[v310] training "
            f"{model_name} | dir={config['direction_weight']} amp={config['amplitude_weight']} flat={config['flat_weight']} "
            f"shape_extra={config['target_shape_extra']}"
        )
        run = train_v310_candidate(model_name, config, prepared, device, seed=SEED + idx, v307_checkpoint=v307_ckpt)
        runs.append(run)
        write_csv(run.training_history, TABLES / f"{model_name}_training_history.csv")
        torch.save(
            {
                "model_name": run.model_name,
                "state_dict": run.state_dict,
                "config": run.config,
                "roll_feature_names": prepared.roll_feature_names,
                "roll_impute_mean": prepared.roll_impute_mean,
                "roll_scale_mean": prepared.roll_scale_mean,
                "roll_scale_std": prepared.roll_scale_std,
                "class_names": prepared.class_names,
                "class_weight": prepared.class_weight,
                "best_epoch": run.best_epoch,
                "best_val_loss": run.best_val_loss,
                "training_seconds": run.training_seconds,
                "seed": SEED + idx,
                "initialized_from_v307_model": v307_name,
            },
            MODELS / f"{model_name}.pt",
        )
        print(f"[v310] {model_name} best_epoch={run.best_epoch} best_val_loss={run.best_val_loss:.6f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    multiplier_audit = pd.concat(multiplier_audits, ignore_index=True)
    write_csv(multiplier_audit, TABLES / "v310_target_shape_multiplier_audit.csv")
    write_csv(
        multiplier_audit.groupby(["model_name", "split"], as_index=False).agg(
            rows=("event_uid", "size"),
            event_n=("event_uid", "nunique"),
            multiplier_mean=("target_shape_multiplier", "mean"),
            multiplier_p90=("target_shape_multiplier", lambda s: float(np.quantile(s.astype(float), 0.90))),
            extreme_target_n=("extreme_target", "sum"),
            flat_vehicle_risk_n=("flat_vehicle_risk", "sum"),
            direction_fragile_n=("direction_fragile", "sum"),
        ),
        TABLES / "v310_target_shape_multiplier_summary.csv",
    )

    print("[v310] 计算指标和 validation-only 选择")
    pred_by_model: Dict[str, np.ndarray] = {
        v300_name: pred_v300.astype(np.float32),
        v307_name: pred_v307.astype(np.float32),
    }
    for run in runs:
        pred_by_model[run.model_name] = run.pred_curve.astype(np.float32)

    metrics = V304.V238.compute_metrics_table(
        y_true_curve=y_true_curve,
        pred_by_model=pred_by_model,
        manifest=prepared_base.data.manifest,
        eval_modes=["original_remaining", "receding_2s_diagnostic"],
    )
    write_csv(metrics, TABLES / "v310_metrics_by_delay_and_bucket.csv")

    per_tables = []
    for model_name, pred_curve in pred_by_model.items():
        per = V304.V238.build_per_sample_metrics(
            y_true_curve=y_true_curve,
            pred_curve=pred_curve,
            manifest=prepared_base.data.manifest,
            model_name=model_name,
        )
        per_tables.append(per)
    per_sample = pd.concat(per_tables, ignore_index=True)
    per_sample = V304.V300.attach_v299_event_labels(per_sample, prepared_base.event_table)
    write_csv(per_sample, TABLES / "v310_per_sample_metrics_original_remaining.csv")

    delay0_summary = V304.V300.build_delay0_group_summary(per_sample)
    write_csv(delay0_summary, TABLES / "v310_delay0_group_summary.csv")

    selection = V304.build_selection_from_metrics(metrics, delay0_summary, runs, v300_name)
    write_csv(selection, TABLES / "v310_model_selection_validation.csv")
    selected_name = str(selection.iloc[0]["model_name"])

    event_metrics = V304.build_event_aux_metrics(prepared_base, runs)
    write_csv(event_metrics, TABLES / "v310_coarse_scene_aux_metrics.csv")

    model_names_for_diag = [v300_name, v307_name] + [run.model_name for run in runs]
    severe_summary, severe_event_compare = build_v309_severe_diagnostics(per_sample, model_names_for_diag)
    write_csv(severe_summary, TABLES / "v310_v309_severe_group_summary.csv")
    write_csv(severe_event_compare, TABLES / "v310_v309_severe_event_comparison.csv")

    print("[v310] 保存预测数组和图像")
    original_remaining_valid, _ = V304.V238.build_original_remaining_mask(prepared_base.data.manifest)
    npz_payload = {
        "y_true_steering_delta": y_true_curve.astype(np.float32),
        "pred_v300_reference": pred_v300.astype(np.float32),
        "v300_reference_model": np.array([v300_name]),
        "pred_v307_reference": pred_v307.astype(np.float32),
        "v307_reference_model": np.array([v307_name]),
        "pred_v310_selected": pred_by_model[selected_name].astype(np.float32),
        "best_v310_model": np.array([selected_name]),
        "delay_ms": prepared_base.data.manifest["delay_ms"].astype(int).to_numpy(dtype=np.int32),
        "split": prepared_base.data.manifest["split"].astype(str).to_numpy(),
        "event_uid": prepared_base.data.manifest["event_uid"].astype(str).to_numpy(),
        "subject": prepared_base.data.manifest["subject"].astype(str).to_numpy(),
        "future_grid_s": FUTURE_GRID.astype(np.float32),
        "original_remaining_valid": original_remaining_valid.astype(bool),
        "coarse_scene_label": prepared_base.event_label_name.astype(str),
        "coarse_scene_class_index": prepared_base.event_label.astype(np.int64),
        "coarse_scene_class_names": np.array(prepared_base.class_names),
    }
    for run in runs:
        npz_payload[f"pred_{run.model_name}"] = run.pred_curve.astype(np.float32)
        npz_payload[f"event_logits_{run.model_name}"] = run.event_logits.astype(np.float32)
    np.savez_compressed(OUT / "v310_severe_error_targeted_predictions.npz", **npz_payload)

    with (MODELS / "v310_scalers_and_selection.pkl").open("wb") as f:
        pickle.dump(
            {
                "selection": selection.to_dict(orient="records"),
                "selected_name": selected_name,
                "v300_reference_model": v300_name,
                "v307_reference_model": v307_name,
                "roll_feature_names": prepared_base.roll_feature_names,
                "roll_impute_mean": prepared_base.roll_impute_mean,
                "roll_scale_mean": prepared_base.roll_scale_mean,
                "roll_scale_std": prepared_base.roll_scale_std,
                "class_names": prepared_base.class_names,
                "v300_guardrail": v300_guard,
                "v307_guardrail": v307_guard,
            },
            f,
        )

    figure_paths = [
        V304.plot_training_history(runs),
        V304.plot_delay0_group_bars(delay0_summary, selected_name, v300_name),
        V304.plot_event_aux(event_metrics, selected_name),
        plot_v309_severe_group_bars(severe_summary, selected_name, v307_name, v300_name),
    ]

    event_split_n = prepared_base.data.manifest.groupby("event_uid")["split"].nunique()
    event_delay_n = prepared_base.data.manifest.groupby("event_uid")["delay_ms"].nunique()
    selected_row = selection.iloc[0].to_dict()
    selected_test_all = metric_row(delay0_summary, selected_name, "test", "all")
    selected_test_bad10 = metric_row(delay0_summary, selected_name, "test", "within_bad_top10")
    v307_test_all = metric_row(delay0_summary, v307_name, "test", "all")
    v307_test_bad10 = metric_row(delay0_summary, v307_name, "test", "within_bad_top10")
    severe_selected = severe_summary[
        severe_summary["model_name"].eq(selected_name) & severe_summary["diagnostic_group"].eq("v309_severe_all37")
    ].iloc[0]
    severe_v307 = severe_summary[
        severe_summary["model_name"].eq(v307_name) & severe_summary["diagnostic_group"].eq("v309_severe_all37")
    ].iloc[0]

    guardrail = {
        "pass": bool((event_split_n <= 1).all() and (event_delay_n == 6).all()),
        "version": "v310_severe_error_targeted_curve_model_20260704",
        "model_structure_changed": False,
        "loss_changed": True,
        "output_target_unchanged": "21_point_steering_delta_curve",
        "initialized_from_v307_selected": True,
        "v307_reference_model": v307_name,
        "v300_reference_model": v300_name,
        "selected_v310_model": selected_name,
        "uses_coarse_scene_labels_as_features": True,
        "uses_v309_severe_table_for_training": False,
        "uses_v309_severe_table_for_validation_selection": False,
        "uses_v309_severe_table_for_diagnostic_only": True,
        "uses_test_error_as_features": False,
        "candidate_selection_uses_test": False,
        "candidate_selection_uses_validation_only": True,
        "same_event_never_repeated_across_splits": bool((event_split_n <= 1).all()),
        "event_in_multiple_splits_n": int((event_split_n > 1).sum()),
        "event_without_6_delay_rows_n": int((event_delay_n != 6).sum()),
        "event_n": int(prepared_base.data.manifest["event_uid"].nunique()),
        "rolling_sample_n": int(len(prepared_base.data.manifest)),
        "v309_severe_candidate_n": int(pd.read_csv(V309_SEVERE, encoding="utf-8-sig").shape[0]),
        "selected_passes_v304_noharm_gate": bool(selected_row.get("passes_v304_noharm_gate", False)),
        "selected_val_all_delta_vs_v300": float(selected_row.get("delay0_val_all_delta_vs_v300", math.nan)),
        "selected_val_bad10_delta_vs_v300": float(selected_row.get("delay0_val_bad10_delta_vs_v300", math.nan)),
        "selected_test_all_rmse": float(selected_test_all["sample_rmse_mean"]) if selected_test_all is not None else math.nan,
        "v307_test_all_rmse": float(v307_test_all["sample_rmse_mean"]) if v307_test_all is not None else math.nan,
        "selected_test_bad10_rmse": float(selected_test_bad10["sample_rmse_mean"]) if selected_test_bad10 is not None else math.nan,
        "v307_test_bad10_rmse": float(v307_test_bad10["sample_rmse_mean"]) if v307_test_bad10 is not None else math.nan,
        "selected_v309_severe_all37_rmse": float(severe_selected["sample_rmse_mean"]),
        "v307_v309_severe_all37_rmse": float(severe_v307["sample_rmse_mean"]),
        "device": str(device),
        "runtime_seconds": float(time.time() - start_time),
        "figure_paths": [str(p) for p in figure_paths],
    }
    write_json(guardrail, LOGS / "guardrail_check.json")
    report_path = write_report(
        selection,
        delay0_summary,
        severe_summary,
        severe_event_compare,
        guardrail,
        selected_name,
        v307_name,
        v300_name,
    )
    guardrail["report_path"] = str(report_path)
    write_json(guardrail, LOGS / "guardrail_check.json")

    write_file_inventory()
    zip_path, zip_ok = make_zip_package()
    guardrail["zip_path"] = str(zip_path)
    guardrail["zip_testzip"] = bool(zip_ok)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()

    print("[v310] 完成")
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
