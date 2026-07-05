#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v243：在 v241 backbone 上做更激进的 guarded fine-tune。

本轮不改变任务定义，也不回到 gate/router/删除样本/response-type hard routing。
核心目标是：
1. 继续使用 v238/v239/v241 已验证的 original_remaining masked point-level target；
2. 直接从 v241_tcn_mha_h96 权重初始化，而不是重新训练一个完全不同的模型；
3. 对 v241 已经预测很差、且属于强转向/零交叉/反向/多修正的样本提高训练权重；
4. 用 v241 作为 teacher/reference，显式惩罚“新模型比 v241 更差”的点；
5. 对 v241 已经表现不错的 normal_predictable 样本加轻量 teacher anchor，防止平均变好但正常样本坏掉。

直观理解：
- hard weight：让模型更认真看困难样本；
- guard loss：如果新模型比 v241 差太多，就付额外损失；
- teacher anchor：对 v241 已经做对的正常样本，不允许新模型随意漂移。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pickle
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# 限制底层线程，避免 Windows + MKL/OpenMP 混用时实验耗时大幅波动。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V241_SCRIPT = BASELINES / "scripts" / "stage03_v241_stronger_temporal_model_20260626.py"
V241_DIR = BASELINES / "v241_stronger_temporal_model_20260626"
V241_PRED = V241_DIR / "v241_stronger_temporal_predictions.npz"
V241_MODEL = V241_DIR / "models" / "v241_best_stronger_temporal_diagnostic.pt"
V241_SELECTION = V241_DIR / "tables" / "v241_model_selection_validation_noharm.csv"

V242_SELECTION = BASELINES / "v242_joint_curve_decoder_20260626" / "tables" / "v242_model_selection_validation_noharm.csv"

OUT = BASELINES / "v243_v241_guarded_finetune_20260626"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
MODELS = OUT / "models"

DELAY_MS = [0, 200, 400, 600, 800, 1000]
FORMAL_DELAY_MAX_MS = 800
STRONG_DELAY_MAX_MS = 600
STRONG_EXCEPTION_DELAYS = [400, 1000]
NOHARM_TOL = 0.02
UPGRADE_TOL = 0.03
SEED = 243

# v243 额外的验证守门：不只看 bucket 均值，也看逐样本相对 v241 的回退。
# 这些阈值故意不是非常保守，因为本轮用户明确希望在 v241 基础上更大胆尝试。
VAL_ALL_REG_RATE_LIMIT = 0.58
VAL_NORMAL_REG_RATE_LIMIT = 0.58
VAL_P90_DELTA_LIMIT = 0.16
VAL_MAX_DELTA_LIMIT = 0.80
MEANINGFUL_GAIN_TOL = -0.005

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    """按路径导入前序脚本，复用已经验证过的数据读取、模型结构和指标函数。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V241 = import_module_from_path("stage03_v241_stronger_temporal_model_20260626", V241_SCRIPT)
V239 = V241.V239
V238 = V241.V238
FUTURE_GRID = V238.FUTURE_GRID


@dataclass
class GuardedRun:
    """一个 v243 guarded fine-tune candidate 的完整训练结果。"""

    model_name: str
    config: Dict[str, object]
    state_dict: Dict[str, torch.Tensor]
    pred_curve: np.ndarray
    training_history: pd.DataFrame
    training_seconds: float
    best_epoch: int
    best_val_loss: float
    best_val_base_loss: float
    best_val_guard_loss: float


class GuardedPointSequenceDataset(Dataset):
    """
    point-level 数据集。

    相比 v239/v241 的 PointSequenceDataset，这里额外返回：
    - teacher：v241 对同一个 future point 的标准化预测；
    - guard_weight：这个点在 guard loss 里的权重；
    - anchor_weight：这个点是否参与 teacher anchor。
    """

    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        point_data,
        point_mask: np.ndarray,
        adjusted_weight: np.ndarray,
        teacher_scaled: np.ndarray,
        guard_weight: np.ndarray,
        anchor_weight: np.ndarray,
    ) -> None:
        self.hist = arrays["hist"]
        self.road = arrays["road"]
        self.phase = arrays["phase"]
        self.point = arrays["point"]
        self.y = arrays["y"]
        self.sample_index = point_data.sample_index_all.astype(np.int64)
        self.indices = np.where(point_mask)[0].astype(np.int64)
        self.adjusted_weight = adjusted_weight.astype(np.float32)
        self.teacher_scaled = teacher_scaled.astype(np.float32)
        self.guard_weight = guard_weight.astype(np.float32)
        self.anchor_weight = anchor_weight.astype(np.float32)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        point_idx = int(self.indices[idx])
        sample_idx = int(self.sample_index[point_idx])
        return {
            "hist": torch.from_numpy(self.hist[sample_idx]),
            "road": torch.from_numpy(self.road[sample_idx]),
            "phase": torch.from_numpy(self.phase[sample_idx]),
            "point": torch.from_numpy(self.point[point_idx]),
            "y": torch.tensor(self.y[point_idx], dtype=torch.float32),
            "weight": torch.tensor(self.adjusted_weight[point_idx], dtype=torch.float32),
            "teacher": torch.tensor(self.teacher_scaled[point_idx], dtype=torch.float32),
            "guard_weight": torch.tensor(self.guard_weight[point_idx], dtype=torch.float32),
            "anchor_weight": torch.tensor(self.anchor_weight[point_idx], dtype=torch.float32),
        }


def ensure_dirs() -> None:
    """创建 v243 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v243 自己的输出目录，不触碰 v241/v242 前序产物。

    Windows 后台运行时 stdout/stderr 可能已经重定向到 OUT/logs。
    这些文件处于打开状态，不能删除；因此这里保留正在写入的训练日志，
    其余 v243 产物正常清理。
    """

    if OUT.exists():
        for child in OUT.iterdir():
            if child == LOGS:
                child.mkdir(parents=True, exist_ok=True)
                for log_file in child.iterdir():
                    if log_file.name in {"train_stdout.log", "train_stderr.log"}:
                        continue
                    if log_file.is_dir():
                        shutil.rmtree(log_file)
                    else:
                        log_file.unlink()
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 使用 utf-8-sig，方便 Windows Excel 直接打开中文列。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算文件 SHA256，便于后续追溯输入。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int = SEED) -> None:
    """固定 numpy/torch 随机种子。"""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def load_v241_prediction() -> Tuple[np.ndarray, str]:
    """读取 v241 当前最佳预测曲线和模型名。"""

    if not V241_PRED.exists():
        raise FileNotFoundError(f"缺少 v241 prediction npz：{V241_PRED}")
    with np.load(V241_PRED, allow_pickle=False) as pred:
        arr = pred["pred_v241_best_stronger_steering_delta"].astype(np.float32)
        model_name = str(pred["best_stronger_model"][0])
    return arr, model_name


def load_v239_prediction() -> Tuple[np.ndarray, str]:
    """读取 v239 attention 预测，作为报告里的参照。"""

    return V241.load_v239_prediction()


def flatten_curve_to_points(curve: np.ndarray, point_data) -> np.ndarray:
    """把 N x 21 曲线按 point_data 顺序拉平成 point-level 向量。"""

    return curve[
        point_data.sample_index_all.astype(np.int64),
        point_data.time_index_all.astype(np.int64),
    ].astype(np.float32)


def weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """带权均值；权重全零时返回 0，避免 anchor 稀疏时产生 NaN。"""

    denom = torch.sum(weight)
    if bool((denom <= 1e-8).detach().cpu().item()):
        return torch.sum(values * 0.0)
    return torch.sum(values * weight) / torch.clamp(denom, min=1e-8)


def guarded_loss_components(
    pred: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    teacher: torch.Tensor,
    guard_weight: torch.Tensor,
    anchor_weight: torch.Tensor,
    config: Dict[str, object],
) -> Dict[str, torch.Tensor]:
    """计算 v243 的 base/guard/anchor/total loss。"""

    err_new = torch.square(pred - y)
    err_teacher = torch.square(teacher - y)
    base = weighted_mean(err_new, weight)

    # guard 的含义：如果新模型的平方误差超过 v241 的平方误差加一个 margin，就额外惩罚。
    margin_sq = float(config["_guard_margin_scaled_sq"])
    guard_excess = torch.relu(err_new - err_teacher - margin_sq)
    guard = weighted_mean(guard_excess, weight * guard_weight)

    # teacher anchor 只对 v241 已经表现好的 normal_predictable 点启用，防止正常样本漂移。
    anchor = weighted_mean(torch.square(pred - teacher), weight * anchor_weight)

    total = base + float(config["guard_alpha"]) * guard + float(config["teacher_anchor_beta"]) * anchor
    return {
        "total_loss": total,
        "base_loss": base,
        "guard_loss": guard,
        "anchor_loss": anchor,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    config: Dict[str, object],
) -> Dict[str, float]:
    """运行一个训练或验证 epoch，并返回各 loss 分量。"""

    is_train = optimizer is not None
    model.train(is_train)
    totals = {"total_loss": 0.0, "base_loss": 0.0, "guard_loss": 0.0, "anchor_loss": 0.0}
    total_weight = 0.0
    for batch in loader:
        hist = batch["hist"].to(device=device, dtype=torch.float32)
        road = batch["road"].to(device=device, dtype=torch.float32)
        phase = batch["phase"].to(device=device, dtype=torch.float32)
        point = batch["point"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.float32)
        weight = batch["weight"].to(device=device, dtype=torch.float32)
        teacher = batch["teacher"].to(device=device, dtype=torch.float32)
        guard_weight = batch["guard_weight"].to(device=device, dtype=torch.float32)
        anchor_weight = batch["anchor_weight"].to(device=device, dtype=torch.float32)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(hist, road, phase, point)
        losses = guarded_loss_components(pred, y, weight, teacher, guard_weight, anchor_weight, config)
        if is_train:
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["grad_clip"]))
            optimizer.step()

        wsum = float(torch.sum(weight).detach().cpu().item())
        for key in totals:
            totals[key] += float(losses[key].detach().cpu().item()) * wsum
        total_weight += wsum

    return {key: value / max(total_weight, 1e-8) for key, value in totals.items()}


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    """控制 hist/road encoder 是否参与 fine-tune。"""

    for param in model.parameters():
        param.requires_grad = True
    if not trainable:
        for module_name in ("hist_encoder", "road_encoder"):
            module = getattr(model, module_name)
            for param in module.parameters():
                param.requires_grad = False


def instantiate_v241_model_from_checkpoint(
    data,
    device: torch.device,
    model_dropout: float | None = None,
) -> Tuple[nn.Module, Dict[str, object], str]:
    """按 v241 checkpoint 的结构实例化模型，并加载 v241 权重。"""

    if not V241_MODEL.exists():
        raise FileNotFoundError(f"缺少 v241 checkpoint：{V241_MODEL}")
    checkpoint = torch.load(V241_MODEL, map_location="cpu", weights_only=False)
    base_config = dict(checkpoint["config"])
    if model_dropout is not None:
        base_config["dropout"] = float(model_dropout)

    model = V241.StrongerTemporalQueryAttention(
        hist_dim=data.x_hist.shape[-1],
        road_dim=data.x_road.shape[-1],
        phase_dim=data.x_phase.shape[-1],
        point_dim=len(V238.POINT_EXTRA_FEATURE_NAMES),
        hist_len=data.x_hist.shape[1],
        road_len=data.x_road.shape[1],
        hidden_dim=int(base_config["hidden_dim"]),
        n_heads=int(base_config["n_heads"]),
        n_layers=int(base_config["n_layers"]),
        mlp_hidden=int(base_config["mlp_hidden"]),
        dropout=float(base_config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, base_config, str(checkpoint["model_name"])


def build_weight_plan(
    data,
    point_data,
    arrays: Dict[str, np.ndarray],
    scalers,
    point_masks: Dict[str, np.ndarray],
    pred_v241: np.ndarray,
    v241_name: str,
    config: Dict[str, object],
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    构造 v243 的 point-level 权重和 teacher 权重。

    注意：
    - hard weight 只根据 train split 的 v241 残差分位数确定；
    - val/test 不参与权重分位数拟合；
    - val/test 的 adjusted_weight 保持原始权重，避免把验证评估也变成偏置评估。
    """

    y_true_curve = data.y_future[:, :, 0].astype(np.float32)
    per_v241 = V238.build_per_sample_metrics(y_true_curve, pred_v241, data.manifest, v241_name)
    per_v241["sample_index"] = np.arange(len(per_v241), dtype=np.int32)

    train_sample_mask = data.manifest["split"].astype(str).to_numpy() == "train"
    train_metrics = per_v241[per_v241["split"].eq("train")].copy()
    train_tail = train_metrics["tail_rmse"].astype(float).to_numpy()
    train_tail = train_tail[np.isfinite(train_tail)]
    q70 = float(np.quantile(train_tail, 0.70)) if train_tail.size else 0.0
    q85 = float(np.quantile(train_tail, 0.85)) if train_tail.size else 0.0
    q95 = float(np.quantile(train_tail, 0.95)) if train_tail.size else 0.0

    sample_multiplier = np.ones(len(data.manifest), dtype=np.float32)
    hard_sample_count = 0
    very_hard_sample_count = 0
    for _, row in train_metrics.iterrows():
        idx = int(row["sample_index"])
        tail_rmse = float(row["tail_rmse"])
        multiplier = 1.0
        if np.isfinite(tail_rmse) and tail_rmse >= q70:
            multiplier += float(config["tail_q70_boost"])
            hard_sample_count += 1
        if np.isfinite(tail_rmse) and tail_rmse >= q85:
            multiplier += float(config["tail_q85_boost"])
            very_hard_sample_count += 1
        if np.isfinite(tail_rmse) and tail_rmse >= q95:
            multiplier += float(config["tail_q95_boost"])
        if bool(row["strong_steer"]):
            multiplier += float(config["strong_boost"])
        if bool(row["strong_under"]):
            multiplier += float(config["strong_under_boost"])
        if bool(row["observe_later_like"]):
            multiplier += float(config["observe_boost"])
        if bool(row["zero_cross"]) or bool(row["reverse"]) or bool(row["multi_correction"]):
            multiplier += float(config["complex_boost"])
        if bool(row["extreme_peak"]):
            multiplier += float(config["extreme_peak_boost"])
        if bool(row["strong_steer"]) and int(row["delay_ms"]) in STRONG_EXCEPTION_DELAYS:
            multiplier += float(config["strong_exception_boost"])
        sample_multiplier[idx] = min(float(config["hard_weight_cap"]), multiplier)

    point_sample_idx = point_data.sample_index_all.astype(np.int64)
    adjusted_weight = point_data.point_weight_all.astype(np.float32).copy()
    adjusted_weight *= sample_multiplier[point_sample_idx]

    # 对 original_remaining 的后半段再做一点增强，因为 v241/v242 都显示 tail 是核心短板。
    point_delay = data.manifest["delay_ms"].astype(int).to_numpy()[point_sample_idx]
    original_rel_s = point_delay.astype(np.float32) / 1000.0 + FUTURE_GRID[point_data.time_index_all.astype(np.int64)]
    tail_point = original_rel_s >= 1.0 - 1e-9
    adjusted_weight *= np.where(tail_point, float(config["tail_point_boost"]), 1.0).astype(np.float32)
    point_cap = point_data.point_weight_all.astype(np.float32) * float(config["point_weight_cap"])
    adjusted_weight = np.minimum(adjusted_weight, point_cap).astype(np.float32)

    teacher_point = flatten_curve_to_points(pred_v241, point_data)
    teacher_scaled = ((teacher_point - scalers.y_mean) / scalers.y_std).astype(np.float32)
    teacher_abs_err = np.abs(teacher_scaled - arrays["y"].astype(np.float32))

    train_point_mask = point_masks["train"]
    train_teacher_err = teacher_abs_err[train_point_mask]
    good_q55 = float(np.quantile(train_teacher_err[np.isfinite(train_teacher_err)], 0.55))
    bad_q80 = float(np.quantile(train_teacher_err[np.isfinite(train_teacher_err)], 0.80))

    manifest = data.manifest.reset_index(drop=True)
    normal_sample = (
        manifest["normal_curve"].astype(bool).to_numpy()
        & ~manifest["observe_later_like"].astype(bool).to_numpy()
        & ~manifest["strong_steer"].astype(bool).to_numpy()
    )
    normal_point = normal_sample[point_sample_idx]

    guard_weight = np.ones(len(point_data.y_point_all), dtype=np.float32)
    guard_weight += np.where(normal_point & (teacher_abs_err <= good_q55), float(config["guard_good_normal_extra"]), 0.0)
    guard_weight += np.where(teacher_abs_err >= bad_q80, float(config["guard_bad_teacher_extra"]), 0.0)
    guard_weight = np.clip(guard_weight, 0.25, float(config["guard_weight_cap"])).astype(np.float32)

    normal_train_tail = train_metrics[
        train_metrics["strong_steer"].eq(False) & train_metrics["observe_later_like"].eq(False)
    ]["tail_rmse"].astype(float)
    normal_good_limit = float(normal_train_tail.quantile(0.60)) if len(normal_train_tail) else q70
    sample_good_normal = np.zeros(len(data.manifest), dtype=bool)
    for _, row in train_metrics.iterrows():
        idx = int(row["sample_index"])
        if (
            bool(normal_sample[idx])
            and np.isfinite(float(row["tail_rmse"]))
            and float(row["tail_rmse"]) <= normal_good_limit
        ):
            sample_good_normal[idx] = True
    anchor_weight = (
        sample_good_normal[point_sample_idx]
        & normal_point
        & (teacher_abs_err <= good_q55)
        & train_point_mask
    ).astype(np.float32)
    anchor_weight *= float(config["anchor_weight_value"])

    # val/test 只用于验证和最终报告，不让训练用的 hard weight 改变它们的 loss 口径。
    adjusted_weight[point_masks["val"] | point_masks["test"]] = point_data.point_weight_all[
        point_masks["val"] | point_masks["test"]
    ].astype(np.float32)

    weight_stats = pd.DataFrame(
        [
            {
                "model_name": config["model_name"],
                "train_tail_rmse_q70_v241": q70,
                "train_tail_rmse_q85_v241": q85,
                "train_tail_rmse_q95_v241": q95,
                "hard_sample_count_q70plus": int(hard_sample_count),
                "very_hard_sample_count_q85plus": int(very_hard_sample_count),
                "train_point_weight_mean": float(np.mean(adjusted_weight[train_point_mask])),
                "train_point_weight_p90": float(np.quantile(adjusted_weight[train_point_mask], 0.90)),
                "train_point_weight_max": float(np.max(adjusted_weight[train_point_mask])),
                "guard_weight_mean": float(np.mean(guard_weight[train_point_mask])),
                "guard_weight_max": float(np.max(guard_weight[train_point_mask])),
                "anchor_point_count": int(np.sum(anchor_weight[train_point_mask] > 0.0)),
                "anchor_point_rate": float(np.mean(anchor_weight[train_point_mask] > 0.0)),
            }
        ]
    )

    payload = {
        "adjusted_weight": adjusted_weight,
        "teacher_scaled": teacher_scaled,
        "guard_weight": guard_weight,
        "anchor_weight": anchor_weight,
        "sample_multiplier": sample_multiplier,
        "teacher_abs_err": teacher_abs_err.astype(np.float32),
    }
    return payload, weight_stats


def evaluate_validation_snapshot(
    model: nn.Module,
    model_name: str,
    data,
    point_data,
    arrays: Dict[str, np.ndarray],
    scalers,
    pred_v241: np.ndarray,
    v241_name: str,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    把当前 epoch 的点预测还原成完整曲线，并直接用 validation 分层指标打分。

    第一版 v243 用 point-level val loss 做早停，结果三个候选都停在 epoch 0。
    这里改成按真正关心的 validation no-harm / v241-upgrade / sample-guard 选快照。
    """

    pred_curve = V239.predict_all_points(
        model,
        arrays,
        point_data,
        scalers,
        device,
        batch_size=batch_size * 4,
    ).astype(np.float32)
    y_true_curve = data.y_future[:, :, 0].astype(np.float32)
    metrics = V238.compute_metrics_table(
        y_true_curve=y_true_curve,
        pred_by_model={
            "v236_joint_ridge_existing": data.pred_v236[:, :, 0].astype(np.float32),
            v241_name: pred_v241.astype(np.float32),
            model_name: pred_curve,
        },
        manifest=data.manifest,
        eval_modes=["original_remaining"],
    )
    per_sample_delta = build_per_sample_delta_table(
        y_true_curve=y_true_curve,
        pred_ref=pred_v241.astype(np.float32),
        pred_candidate=pred_curve,
        manifest=data.manifest,
        ref_name=v241_name,
        candidate_name=model_name,
    )
    decision = candidate_validation_decision(metrics, model_name, v241_name, per_sample_delta)
    return pred_curve, decision


def train_guarded_candidate(
    model_name: str,
    raw_config: Dict[str, object],
    data,
    point_data,
    arrays: Dict[str, np.ndarray],
    scalers,
    point_masks: Dict[str, np.ndarray],
    pred_v241: np.ndarray,
    v241_name: str,
    device: torch.device,
) -> Tuple[GuardedRun, pd.DataFrame]:
    """从 v241 checkpoint 出发训练一个 guarded fine-tune candidate。"""

    config = dict(raw_config)
    config["model_name"] = model_name
    config["_guard_margin_scaled_sq"] = float(float(config["guard_margin_original"]) / max(float(scalers.y_std), 1e-8)) ** 2

    weight_payload, weight_stats = build_weight_plan(
        data=data,
        point_data=point_data,
        arrays=arrays,
        scalers=scalers,
        point_masks=point_masks,
        pred_v241=pred_v241,
        v241_name=v241_name,
        config=config,
    )

    train_dataset = GuardedPointSequenceDataset(
        arrays=arrays,
        point_data=point_data,
        point_mask=point_masks["train"],
        adjusted_weight=weight_payload["adjusted_weight"],
        teacher_scaled=weight_payload["teacher_scaled"],
        guard_weight=weight_payload["guard_weight"],
        anchor_weight=weight_payload["anchor_weight"],
    )
    val_dataset = GuardedPointSequenceDataset(
        arrays=arrays,
        point_data=point_data,
        point_mask=point_masks["val"],
        adjusted_weight=weight_payload["adjusted_weight"],
        teacher_scaled=weight_payload["teacher_scaled"],
        guard_weight=weight_payload["guard_weight"],
        anchor_weight=weight_payload["anchor_weight"] * 0.0,
    )

    batch_size = int(config["batch_size"])
    generator = torch.Generator()
    generator.manual_seed(SEED)
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

    model, base_model_config, source_model_name = instantiate_v241_model_from_checkpoint(
        data=data,
        device=device,
        model_dropout=float(config["model_dropout"]) if "model_dropout" in config else None,
    )
    if source_model_name != v241_name:
        raise AssertionError(f"checkpoint 模型名 {source_model_name} 与 prediction 模型名 {v241_name} 不一致")

    history: List[Dict[str, object]] = []
    start_time = time.time()

    # epoch 0 是纯 v241 权重。它作为安全下界：如果 fine-tune 全部变差，就不会把坏权重保存成 best。
    # 但“best”的判据不再只看 point-level loss，而是直接看 validation 分层指标。
    set_encoder_trainable(model, trainable=True)
    initial_val = run_epoch(model, val_loader, device, None, config)
    initial_curve, initial_decision = evaluate_validation_snapshot(
        model=model,
        model_name=model_name,
        data=data,
        point_data=point_data,
        arrays=arrays,
        scalers=scalers,
        pred_v241=pred_v241,
        v241_name=v241_name,
        device=device,
        batch_size=batch_size,
    )
    best_metric_score = float(initial_decision["validation_selection_score"])
    best_val = float(initial_val["base_loss"] + float(config["val_guard_score_alpha"]) * initial_val["guard_loss"])
    best_val_base = float(initial_val["base_loss"])
    best_val_guard = float(initial_val["guard_loss"])
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_pred_curve = initial_curve.astype(np.float32)
    history.append(
        {
            "model_name": model_name,
            "stage": "initial_v241",
            "epoch": 0,
            "stage_epoch": 0,
            "train_total_loss": math.nan,
            "train_base_loss": math.nan,
            "train_guard_loss": math.nan,
            "train_anchor_loss": math.nan,
            "val_total_loss": initial_val["total_loss"],
            "val_base_loss": initial_val["base_loss"],
            "val_guard_loss": initial_val["guard_loss"],
            "val_anchor_loss": initial_val["anchor_loss"],
            "val_selection_loss": best_val,
            "val_metric_selection_score": best_metric_score,
            "val_metric_accepted": bool(initial_decision["accepted_as_next_candidate"]),
            "val_metric_all_mean_tail_delta_vs_v241": float(initial_decision["all_mean_tail_delta_vs_v241_0to800"]),
            "val_metric_observe_mean_tail_delta_vs_v241": float(
                initial_decision["observe_later_mean_tail_delta_vs_v241_0to800"]
            ),
            "val_metric_strong_exception_mean_tail_delta_vs_v241": float(
                initial_decision["strong_exception_mean_tail_delta_vs_v241_400_1000"]
            ),
            "lr": 0.0,
            "encoder_trainable": True,
        }
    )

    metric_stale = 0
    global_epoch = 0
    curve_eval_every = max(1, int(config.get("curve_eval_every", 2)))
    stages = [
        ("query_head_focus", False, int(config["stage1_epochs"]), float(config["stage1_lr"])),
        ("full_backbone", True, int(config["stage2_epochs"]), float(config["stage2_lr"])),
    ]
    for stage_name, encoder_trainable, n_epochs, lr in stages:
        if n_epochs <= 0:
            continue
        set_encoder_trainable(model, trainable=encoder_trainable)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=float(config["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.55,
            patience=max(2, int(config["patience"]) // 3),
            min_lr=float(config["min_lr"]),
        )
        for stage_epoch in range(1, n_epochs + 1):
            global_epoch += 1
            train_stats = run_epoch(model, train_loader, device, optimizer, config)
            val_stats = run_epoch(model, val_loader, device, None, config)
            val_selection = float(val_stats["base_loss"] + float(config["val_guard_score_alpha"]) * val_stats["guard_loss"])
            scheduler.step(val_selection)
            lr_now = float(optimizer.param_groups[0]["lr"])
            metric_score = math.nan
            metric_accepted = False
            metric_all_delta = math.nan
            metric_observe_delta = math.nan
            metric_strong_exception_delta = math.nan
            should_eval_curve = (global_epoch % curve_eval_every == 0) or (stage_epoch == n_epochs)
            if should_eval_curve:
                snapshot_curve, snapshot_decision = evaluate_validation_snapshot(
                    model=model,
                    model_name=model_name,
                    data=data,
                    point_data=point_data,
                    arrays=arrays,
                    scalers=scalers,
                    pred_v241=pred_v241,
                    v241_name=v241_name,
                    device=device,
                    batch_size=batch_size,
                )
                metric_score = float(snapshot_decision["validation_selection_score"])
                metric_accepted = bool(snapshot_decision["accepted_as_next_candidate"])
                metric_all_delta = float(snapshot_decision["all_mean_tail_delta_vs_v241_0to800"])
                metric_observe_delta = float(snapshot_decision["observe_later_mean_tail_delta_vs_v241_0to800"])
                metric_strong_exception_delta = float(
                    snapshot_decision["strong_exception_mean_tail_delta_vs_v241_400_1000"]
                )
                if metric_score < best_metric_score - 1e-5:
                    best_metric_score = metric_score
                    best_val = val_selection
                    best_val_base = float(val_stats["base_loss"])
                    best_val_guard = float(val_stats["guard_loss"])
                    best_epoch = global_epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_pred_curve = snapshot_curve.astype(np.float32)
                    metric_stale = 0
                else:
                    metric_stale += 1

            history.append(
                {
                    "model_name": model_name,
                    "stage": stage_name,
                    "epoch": global_epoch,
                    "stage_epoch": stage_epoch,
                    "train_total_loss": train_stats["total_loss"],
                    "train_base_loss": train_stats["base_loss"],
                    "train_guard_loss": train_stats["guard_loss"],
                    "train_anchor_loss": train_stats["anchor_loss"],
                    "val_total_loss": val_stats["total_loss"],
                    "val_base_loss": val_stats["base_loss"],
                    "val_guard_loss": val_stats["guard_loss"],
                    "val_anchor_loss": val_stats["anchor_loss"],
                    "val_selection_loss": val_selection,
                    "val_metric_selection_score": metric_score,
                    "val_metric_accepted": bool(metric_accepted),
                    "val_metric_all_mean_tail_delta_vs_v241": metric_all_delta,
                    "val_metric_observe_mean_tail_delta_vs_v241": metric_observe_delta,
                    "val_metric_strong_exception_mean_tail_delta_vs_v241": metric_strong_exception_delta,
                    "lr": lr_now,
                    "encoder_trainable": bool(encoder_trainable),
                }
            )
            if metric_stale >= int(config["patience"]):
                break
        if metric_stale >= int(config["patience"]):
            break

    full_config = dict(config)
    full_config["base_model_config"] = base_model_config
    full_config["best_metric_selection_score"] = best_metric_score
    run = GuardedRun(
        model_name=model_name,
        config=full_config,
        state_dict=best_state,
        pred_curve=best_pred_curve.astype(np.float32),
        training_history=pd.DataFrame(history),
        training_seconds=float(time.time() - start_time),
        best_epoch=int(best_epoch),
        best_val_loss=float(best_val),
        best_val_base_loss=float(best_val_base),
        best_val_guard_loss=float(best_val_guard),
    )
    return run, weight_stats


def finite_mean(values: pd.Series, default: float = math.inf) -> float:
    """安全均值，避免空表或 NaN 影响 selection。"""

    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(arr.mean())


def finite_max(values: pd.Series, default: float = math.inf) -> float:
    """安全最大值，避免空表或 NaN 影响 selection。"""

    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(arr.max())


def finite_quantile(values: pd.Series, q: float, default: float = math.inf) -> float:
    """安全分位数。"""

    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.quantile(arr, q))


def positive_penalty(value: float, threshold: float = 0.0) -> float:
    """只惩罚超过阈值的部分。"""

    if not np.isfinite(value):
        return 10.0
    return max(0.0, value - threshold)


def delta_frame(metrics: pd.DataFrame, candidate_name: str, ref_name: str) -> pd.DataFrame:
    """把 validation/test 指标转成 candidate - reference 的 delta 表。"""

    keep_cols = ["split", "bucket", "delay_ms", "eval_mode"]
    metric_cols = ["steer_sample_rmse_mean", "steer_tail_rmse_mean"]
    cand = metrics[metrics["model_name"].eq(candidate_name)][keep_cols + metric_cols].copy()
    ref = metrics[metrics["model_name"].eq(ref_name)][keep_cols + metric_cols].copy()
    merged = cand.merge(ref, on=keep_cols, how="inner", suffixes=("_candidate", "_ref"))
    merged["delta_sample"] = merged["steer_sample_rmse_mean_candidate"] - merged["steer_sample_rmse_mean_ref"]
    merged["delta_tail"] = merged["steer_tail_rmse_mean_candidate"] - merged["steer_tail_rmse_mean_ref"]
    return merged


def subset_delta(
    merged: pd.DataFrame,
    bucket: str,
    max_delay: int | None = None,
    delays: Iterable[int] | None = None,
) -> pd.DataFrame:
    """抽取某一类样本的 delay 子集。"""

    out = merged[merged["bucket"].eq(bucket)].copy()
    if max_delay is not None:
        out = out[out["delay_ms"].astype(int) <= int(max_delay)].copy()
    if delays is not None:
        wanted = {int(x) for x in delays}
        out = out[out["delay_ms"].astype(int).isin(wanted)].copy()
    return out


def build_per_sample_delta_table(
    y_true_curve: np.ndarray,
    pred_ref: np.ndarray,
    pred_candidate: np.ndarray,
    manifest: pd.DataFrame,
    ref_name: str,
    candidate_name: str,
) -> pd.DataFrame:
    """输出 selected v243 相对 v241 的逐样本差异，便于看坏例和回退率。"""

    per_ref = V238.build_per_sample_metrics(y_true_curve, pred_ref, manifest, ref_name)
    per_candidate = V238.build_per_sample_metrics(y_true_curve, pred_candidate, manifest, candidate_name)
    keep = [
        "event_uid",
        "sample_id",
        "split",
        "delay_ms",
        "sample_rmse",
        "tail_rmse",
        "peak_ratio",
        "strong_under",
        "observe_later_like",
        "strong_steer",
        "reverse",
        "zero_cross",
        "multi_correction",
        "extreme_peak",
    ]
    a = per_ref[keep].copy().rename(
        columns={
            "sample_rmse": "sample_rmse_v241",
            "tail_rmse": "tail_rmse_v241",
            "peak_ratio": "peak_ratio_v241",
            "strong_under": "strong_under_v241",
        }
    )
    b = per_candidate[keep].copy().rename(
        columns={
            "sample_rmse": "sample_rmse_v243",
            "tail_rmse": "tail_rmse_v243",
            "peak_ratio": "peak_ratio_v243",
            "strong_under": "strong_under_v243",
        }
    )
    merged = b.merge(
        a,
        on=[
            "event_uid",
            "sample_id",
            "split",
            "delay_ms",
            "observe_later_like",
            "strong_steer",
            "reverse",
            "zero_cross",
            "multi_correction",
            "extreme_peak",
        ],
        how="left",
    )
    merged["ref_model_name"] = ref_name
    merged["candidate_model_name"] = candidate_name
    merged["delta_sample_v243_minus_v241"] = merged["sample_rmse_v243"] - merged["sample_rmse_v241"]
    merged["delta_tail_v243_minus_v241"] = merged["tail_rmse_v243"] - merged["tail_rmse_v241"]
    merged["delta_peak_ratio_v243_minus_v241"] = merged["peak_ratio_v243"] - merged["peak_ratio_v241"]
    return merged


def summarize_per_sample_delta(per_sample_delta: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """按关键 bucket 汇总 v243 相对 v241 的逐样本 tail 回退情况。"""

    one_split = per_sample_delta[per_sample_delta["split"].eq(split_name)].copy()
    masks = {
        "all": np.ones(len(one_split), dtype=bool),
        "observe_later_like": one_split["observe_later_like"].astype(bool).to_numpy(),
        "normal_predictable": (
            ~one_split["observe_later_like"].astype(bool).to_numpy()
            & ~one_split["strong_steer"].astype(bool).to_numpy()
        ),
        "strong_steer": one_split["strong_steer"].astype(bool).to_numpy(),
        "strong_400_1000": (
            one_split["strong_steer"].astype(bool).to_numpy()
            & one_split["delay_ms"].astype(int).isin(STRONG_EXCEPTION_DELAYS).to_numpy()
        ),
        "zero_cross_or_reverse_or_multi": (
            one_split["zero_cross"].astype(bool).to_numpy()
            | one_split["reverse"].astype(bool).to_numpy()
            | one_split["multi_correction"].astype(bool).to_numpy()
        ),
    }
    rows: List[Dict[str, object]] = []
    for bucket, mask in masks.items():
        bucket_df = one_split.loc[mask].copy()
        if bucket_df.empty:
            rows.append(
                {
                    "split": split_name,
                    "bucket": bucket,
                    "n": 0,
                    "tail_regressions_vs_v241": 0,
                    "tail_regression_rate_vs_v241": math.nan,
                    "mean_delta_tail_v243_minus_v241": math.nan,
                    "max_delta_tail_v243_minus_v241": math.nan,
                    "p90_delta_tail_v243_minus_v241": math.nan,
                }
            )
            continue
        delta = bucket_df["delta_tail_v243_minus_v241"].astype(float)
        regress = delta > 0.0
        rows.append(
            {
                "split": split_name,
                "bucket": bucket,
                "n": int(len(bucket_df)),
                "tail_regressions_vs_v241": int(regress.sum()),
                "tail_regression_rate_vs_v241": float(regress.mean()),
                "mean_delta_tail_v243_minus_v241": float(delta.mean()),
                "max_delta_tail_v243_minus_v241": float(delta.max()),
                "p90_delta_tail_v243_minus_v241": float(delta.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def candidate_validation_decision(
    metrics: pd.DataFrame,
    candidate_name: str,
    v241_name: str,
    per_sample_delta: pd.DataFrame,
) -> Dict[str, object]:
    """只用 validation 判断 v243 是否能替代 v241 进入下一阶段。"""

    vs_v236 = delta_frame(metrics, candidate_name, "v236_joint_ridge_existing")
    vs_v241 = delta_frame(metrics, candidate_name, v241_name)

    normal_v236 = subset_delta(vs_v236, "normal_predictable", max_delay=FORMAL_DELAY_MAX_MS)
    all_v236 = subset_delta(vs_v236, "all", max_delay=FORMAL_DELAY_MAX_MS)
    observe_v236 = subset_delta(vs_v236, "observe_later_like", max_delay=FORMAL_DELAY_MAX_MS)
    strong_v236 = subset_delta(vs_v236, "strong_steer", max_delay=STRONG_DELAY_MAX_MS)
    strong_exception_v236 = subset_delta(vs_v236, "strong_steer", delays=STRONG_EXCEPTION_DELAYS)

    normal_v241 = subset_delta(vs_v241, "normal_predictable", max_delay=FORMAL_DELAY_MAX_MS)
    all_v241 = subset_delta(vs_v241, "all", max_delay=FORMAL_DELAY_MAX_MS)
    observe_v241 = subset_delta(vs_v241, "observe_later_like", max_delay=FORMAL_DELAY_MAX_MS)
    strong_exception_v241 = subset_delta(vs_v241, "strong_steer", delays=STRONG_EXCEPTION_DELAYS)

    normal_max_sample_delta_v236 = finite_max(normal_v236["delta_sample"])
    normal_max_tail_delta_v236 = finite_max(normal_v236["delta_tail"])
    all_max_sample_delta_v236 = finite_max(all_v236["delta_sample"])
    observe_mean_tail_delta_v236 = finite_mean(observe_v236["delta_tail"])
    strong_mean_tail_delta_v236 = finite_mean(strong_v236["delta_tail"])
    strong_exception_mean_tail_delta_v236 = finite_mean(strong_exception_v236["delta_tail"])

    normal_max_tail_delta_v241 = finite_max(normal_v241["delta_tail"])
    all_mean_tail_delta_v241 = finite_mean(all_v241["delta_tail"])
    observe_mean_tail_delta_v241 = finite_mean(observe_v241["delta_tail"])
    strong_exception_mean_tail_delta_v241 = finite_mean(strong_exception_v241["delta_tail"])
    strong_exception_max_tail_delta_v241 = finite_max(strong_exception_v241["delta_tail"])

    val_delta = per_sample_delta[per_sample_delta["split"].eq("val")].copy()
    val_all = val_delta.copy()
    val_normal = val_delta[
        ~val_delta["observe_later_like"].astype(bool) & ~val_delta["strong_steer"].astype(bool)
    ].copy()
    val_tail = val_all["delta_tail_v243_minus_v241"].astype(float)
    val_normal_tail = val_normal["delta_tail_v243_minus_v241"].astype(float)

    val_all_reg_rate = float((val_tail > 0.0).mean()) if len(val_tail) else math.inf
    val_normal_reg_rate = float((val_normal_tail > 0.0).mean()) if len(val_normal_tail) else math.inf
    val_all_p90_delta = finite_quantile(val_tail, 0.90)
    val_normal_p90_delta = finite_quantile(val_normal_tail, 0.90)
    val_max_delta = finite_max(val_tail)

    noharm_vs_v236 = (
        normal_max_sample_delta_v236 <= NOHARM_TOL
        and normal_max_tail_delta_v236 <= NOHARM_TOL
        and all_max_sample_delta_v236 <= NOHARM_TOL
        and observe_mean_tail_delta_v236 <= 0.0
        and strong_mean_tail_delta_v236 <= 0.0
        and strong_exception_mean_tail_delta_v236 <= NOHARM_TOL
    )
    upgrade_vs_v241 = (
        normal_max_tail_delta_v241 <= UPGRADE_TOL
        and all_mean_tail_delta_v241 <= UPGRADE_TOL
        and observe_mean_tail_delta_v241 <= UPGRADE_TOL
        and strong_exception_mean_tail_delta_v241 <= UPGRADE_TOL
    )
    sample_guard_vs_v241 = (
        val_all_reg_rate <= VAL_ALL_REG_RATE_LIMIT
        and val_normal_reg_rate <= VAL_NORMAL_REG_RATE_LIMIT
        and val_all_p90_delta <= VAL_P90_DELTA_LIMIT
        and val_normal_p90_delta <= VAL_P90_DELTA_LIMIT
        and val_max_delta <= VAL_MAX_DELTA_LIMIT
    )
    meaningful_gain_vs_v241 = (
        all_mean_tail_delta_v241 <= MEANINGFUL_GAIN_TOL
        or observe_mean_tail_delta_v241 <= MEANINGFUL_GAIN_TOL
        or strong_exception_mean_tail_delta_v241 <= MEANINGFUL_GAIN_TOL
    )

    cand_val_all = metrics[
        metrics["split"].eq("val")
        & metrics["eval_mode"].eq("original_remaining")
        & metrics["bucket"].eq("all")
        & metrics["model_name"].eq(candidate_name)
    ].copy()
    base_score = finite_mean(cand_val_all["steer_sample_rmse_mean"]) + 0.50 * finite_mean(
        cand_val_all["steer_tail_rmse_mean"]
    )
    penalty_vs_v236 = (
        positive_penalty(normal_max_sample_delta_v236, NOHARM_TOL)
        + positive_penalty(normal_max_tail_delta_v236, NOHARM_TOL)
        + positive_penalty(all_max_sample_delta_v236, NOHARM_TOL)
        + positive_penalty(observe_mean_tail_delta_v236, 0.0)
        + positive_penalty(strong_mean_tail_delta_v236, 0.0)
        + positive_penalty(strong_exception_mean_tail_delta_v236, NOHARM_TOL)
    )
    penalty_vs_v241 = (
        positive_penalty(normal_max_tail_delta_v241, UPGRADE_TOL)
        + positive_penalty(all_mean_tail_delta_v241, UPGRADE_TOL)
        + positive_penalty(observe_mean_tail_delta_v241, UPGRADE_TOL)
        + positive_penalty(strong_exception_mean_tail_delta_v241, UPGRADE_TOL)
    )
    penalty_sample_guard = (
        positive_penalty(val_all_reg_rate, VAL_ALL_REG_RATE_LIMIT)
        + positive_penalty(val_normal_reg_rate, VAL_NORMAL_REG_RATE_LIMIT)
        + positive_penalty(val_all_p90_delta, VAL_P90_DELTA_LIMIT)
        + positive_penalty(val_normal_p90_delta, VAL_P90_DELTA_LIMIT)
        + positive_penalty(val_max_delta, VAL_MAX_DELTA_LIMIT)
    )
    gain_bonus = min(0.0, all_mean_tail_delta_v241, observe_mean_tail_delta_v241, strong_exception_mean_tail_delta_v241)
    selection_score = base_score + 10.0 * penalty_vs_v236 + 7.0 * penalty_vs_v241 + 3.0 * penalty_sample_guard + gain_bonus

    return {
        "model_name": candidate_name,
        "selected_by": "validation_noharm_v241_upgrade_and_sample_guard_only",
        "test_used_for_selection": False,
        "normal_max_sample_delta_vs_v236": normal_max_sample_delta_v236,
        "normal_max_tail_delta_vs_v236": normal_max_tail_delta_v236,
        "all_max_sample_delta_vs_v236": all_max_sample_delta_v236,
        "observe_later_mean_tail_delta_vs_v236_0to800": observe_mean_tail_delta_v236,
        "strong_mean_tail_delta_vs_v236_0to600": strong_mean_tail_delta_v236,
        "strong_exception_mean_tail_delta_vs_v236_400_1000": strong_exception_mean_tail_delta_v236,
        "normal_max_tail_delta_vs_v241": normal_max_tail_delta_v241,
        "all_mean_tail_delta_vs_v241_0to800": all_mean_tail_delta_v241,
        "observe_later_mean_tail_delta_vs_v241_0to800": observe_mean_tail_delta_v241,
        "strong_exception_mean_tail_delta_vs_v241_400_1000": strong_exception_mean_tail_delta_v241,
        "strong_exception_max_tail_delta_vs_v241_400_1000": strong_exception_max_tail_delta_v241,
        "val_all_tail_regression_rate_vs_v241": val_all_reg_rate,
        "val_normal_tail_regression_rate_vs_v241": val_normal_reg_rate,
        "val_all_p90_tail_delta_vs_v241": val_all_p90_delta,
        "val_normal_p90_tail_delta_vs_v241": val_normal_p90_delta,
        "val_max_tail_delta_vs_v241": val_max_delta,
        "noharm_vs_v236_pass": bool(noharm_vs_v236),
        "upgrade_vs_v241_pass": bool(upgrade_vs_v241),
        "sample_guard_vs_v241_pass": bool(sample_guard_vs_v241),
        "meaningful_gain_vs_v241": bool(meaningful_gain_vs_v241),
        "accepted_as_next_candidate": bool(
            noharm_vs_v236 and upgrade_vs_v241 and sample_guard_vs_v241 and meaningful_gain_vs_v241
        ),
        "validation_selection_score": float(selection_score),
    }


def build_compare_table(metrics: pd.DataFrame, model_names: Iterable[str], ref_name: str) -> pd.DataFrame:
    """生成 test original_remaining 对照表。"""

    test = metrics[
        metrics["split"].eq("test")
        & metrics["eval_mode"].eq("original_remaining")
        & metrics["bucket"].isin(["all", "observe_later_like", "strong_steer", "normal_predictable"])
        & metrics["model_name"].isin(list(model_names))
    ].copy()
    pivot = test.pivot_table(
        index=["bucket", "delay_ms"],
        columns="model_name",
        values=["steer_sample_rmse_mean", "steer_tail_rmse_mean", "strong_under_rate", "peak_ratio_mean"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}__{model}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()
    for model in model_names:
        if model == "v236_joint_ridge_existing":
            continue
        for metric in ["steer_sample_rmse_mean", "steer_tail_rmse_mean"]:
            ref_col = f"{metric}__v236_joint_ridge_existing"
            model_col = f"{metric}__{model}"
            if ref_col in pivot.columns and model_col in pivot.columns:
                pivot[f"delta_{metric}__{model}_minus_v236"] = pivot[model_col] - pivot[ref_col]
            ref2_col = f"{metric}__{ref_name}"
            if ref2_col in pivot.columns and model_col in pivot.columns:
                pivot[f"delta_{metric}__{model}_minus_{ref_name}"] = pivot[model_col] - pivot[ref2_col]
    return pivot


def build_candidate_test_robustness_summary(
    compare: pd.DataFrame,
    selection: pd.DataFrame,
    candidate_names: Iterable[str],
    v241_name: str,
) -> pd.DataFrame:
    """
    候选级 test 稳定性汇总。

    这张表不参与模型选择，只用于审查 validation-selected 候选是否在 locked test 上稳定。
    """

    rows: List[Dict[str, object]] = []
    selection_by_name = selection.set_index("model_name")
    for model in candidate_names:
        config = json.loads(str(selection_by_name.loc[model, "config_json"]))
        for bucket in ["all", "normal_predictable", "observe_later_like", "strong_steer"]:
            one = compare[compare["bucket"].eq(bucket)].copy()
            tail_col = f"delta_steer_tail_rmse_mean__{model}_minus_{v241_name}"
            sample_col = f"delta_steer_sample_rmse_mean__{model}_minus_{v241_name}"
            tail = one[tail_col].astype(float)
            sample = one[sample_col].astype(float)
            rows.append(
                {
                    "model_name": model,
                    "validation_rank": int(selection_by_name.loc[model, "validation_rank"]),
                    "validation_accepted": bool(selection_by_name.loc[model, "accepted_as_next_candidate"]),
                    "hard_weight_cap": float(config.get("hard_weight_cap", math.nan)),
                    "bucket": bucket,
                    "mean_tail_delta_test_vs_v241": float(tail.mean()),
                    "max_tail_delta_test_vs_v241": float(tail.max()),
                    "n_delay_tail_worse_vs_v241": int((tail > 0.0).sum()),
                    "mean_sample_delta_test_vs_v241": float(sample.mean()),
                    "max_sample_delta_test_vs_v241": float(sample.max()),
                    "n_delay_sample_worse_vs_v241": int((sample > 0.0).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_next_decision(selection: pd.DataFrame) -> pd.DataFrame:
    """把 v243 的下一步决策写成机器可读表。"""

    best = selection.sort_values("validation_selection_score").iloc[0]
    accepted = selection[selection["accepted_as_next_candidate"].astype(bool)].copy()
    if accepted.empty:
        accepted_name = ""
        accept_next = False
        next_task = "keep_v241_as_current_best_and_use_v243_as_guarded_loss_diagnostic"
        reason = (
            "No v243 candidate passed v236 no-harm, v241-upgrade, sample-guard, and meaningful-gain checks together. "
            "Keep v241 as current best."
        )
    else:
        accepted = accepted.sort_values("validation_selection_score")
        accepted_name = str(accepted.iloc[0]["model_name"])
        accept_next = True
        next_task = "v244_locked_audit_for_v243_guarded_candidate"
        reason = f"{accepted_name} passed validation checks and can enter locked audit."
    return pd.DataFrame(
        [
            {
                "decision_item": "best_diagnostic_guarded_finetune_model",
                "decision": str(best["model_name"]),
                "reason": f"Lowest validation_selection_score={float(best['validation_selection_score']):.6f}.",
            },
            {
                "decision_item": "accept_v243_as_next_candidate",
                "decision": bool(accept_next),
                "reason": reason,
            },
            {
                "decision_item": "accepted_model_name",
                "decision": accepted_name,
                "reason": "Empty means v241 remains current best.",
            },
            {
                "decision_item": "next_task",
                "decision": next_task,
                "reason": "Only proceed to locked audit if validation accepted a v243 candidate.",
            },
            {
                "decision_item": "formal_headline_change",
                "decision": False,
                "reason": "v243 is a training experiment; formal headline remains locked until audit and robustness checks pass.",
            },
        ]
    )


def plot_figures(compare: pd.DataFrame, selected_model: str, v241_name: str, v239_name: str) -> List[Path]:
    """画 test tail RMSE 对照图。"""

    paths: List[Path] = []
    model_styles = [
        ("v236_joint_ridge_existing", "#777777", "v236"),
        (v239_name, "#1f77b4", "v239"),
        (v241_name, "#2ca02c", "v241"),
        (selected_model, "#d62728", "v243"),
    ]
    for bucket in ["all", "observe_later_like", "strong_steer", "normal_predictable"]:
        one = compare[compare["bucket"].eq(bucket)].copy().sort_values("delay_ms")
        if one.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        for model_name, color, label in model_styles:
            col = f"steer_tail_rmse_mean__{model_name}"
            if col in one.columns:
                ax.plot(one["delay_ms"], one[col], marker="o", color=color, label=label)
        ax.set_xlabel("Observation delay (ms)")
        ax.set_ylabel("Original-remaining tail RMSE")
        ax.set_title(f"v243 guarded fine-tune: {bucket}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = FIGURES / f"v243_guarded_tail_compare_{bucket}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def write_input_hashes() -> None:
    """记录关键输入文件哈希。"""

    paths = [
        V241_SCRIPT,
        V241_PRED,
        V241_MODEL,
        V241_SELECTION,
        V242_SELECTION,
        V241.V239_SCRIPT,
        V241.V239_PRED,
        V241.V239_MODEL,
        V238.V236_ARRAYS,
        V238.V236_MANIFEST,
    ]
    rows = []
    for path in paths:
        if path.exists():
            rows.append({"path": str(path), "sha256": file_sha256(path), "bytes": int(path.stat().st_size)})
        else:
            rows.append({"path": str(path), "sha256": "", "bytes": 0, "missing": True})
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def build_guardrail_json(selection: pd.DataFrame, split_check: pd.DataFrame) -> Dict[str, object]:
    """记录 v243 的方法边界。"""

    checks = {
        "stage": "v243_v241_guarded_finetune",
        "task_base": "v238_original_remaining_masked_point_level_target",
        "model_type": "v241_temporal_convolution_plus_multihead_query_attention_finetuned",
        "initialized_from_v241_checkpoint": True,
        "guarded_loss_used": True,
        "hard_sample_weight_used": True,
        "teacher_anchor_used": True,
        "full_transformer_used": False,
        "gate_router_selector_created": False,
        "response_type_hard_routing_created": False,
        "observe_later_like_deleted": False,
        "formal_headline_changed": False,
        "test_used_for_selection": bool(selection["test_used_for_selection"].astype(bool).any()),
        "same_event_uid_cross_split_count": int(split_check["split_check_status"].eq("fail").sum()),
        "validation_noharm_rule_used": True,
        "validation_v241_upgrade_rule_used": True,
        "validation_sample_guard_rule_used": True,
        "pass": False,
    }
    checks["pass"] = (
        checks["initialized_from_v241_checkpoint"]
        and checks["guarded_loss_used"]
        and checks["hard_sample_weight_used"]
        and not checks["full_transformer_used"]
        and not checks["gate_router_selector_created"]
        and not checks["response_type_hard_routing_created"]
        and not checks["observe_later_like_deleted"]
        and not checks["formal_headline_changed"]
        and not checks["test_used_for_selection"]
        and checks["same_event_uid_cross_split_count"] == 0
        and checks["validation_noharm_rule_used"]
        and checks["validation_v241_upgrade_rule_used"]
        and checks["validation_sample_guard_rule_used"]
    )
    return checks


def file_inventory() -> Dict[str, object]:
    """输出目录文件清单。"""

    entries = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.suffix.lower() != ".zip":
            entries.append(
                {
                    "relative_path": str(path.relative_to(OUT)).replace("\\", "/"),
                    "bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
            )
    return {"output_dir": str(OUT), "file_count_excluding_zip": len(entries), "files": entries}


def zip_outputs() -> Path:
    """打包 v243 输出并做 ZIP 完整性检查。"""

    zip_path = OUT / "v243_v241_guarded_finetune_pack.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"ZIP 校验失败：{bad}")
    return zip_path


def write_report(
    selection: pd.DataFrame,
    next_decision: pd.DataFrame,
    compare: pd.DataFrame,
    candidate_robustness: pd.DataFrame,
    val_sample_summary: pd.DataFrame,
    test_sample_summary: pd.DataFrame,
    guardrail: Dict[str, object],
    device: torch.device,
    v241_name: str,
    v239_name: str,
    zip_path: Path,
) -> None:
    """写中文报告。"""

    best = selection.sort_values("validation_selection_score").iloc[0]
    best_name = str(best["model_name"])
    accepted = selection[selection["accepted_as_next_candidate"].astype(bool)].copy()
    lines: List[str] = []
    lines.append("# v243 v241 guarded fine-tune 实验报告")
    lines.append("")
    lines.append("## 本轮做了什么")
    lines.append("")
    lines.append("- 没有换任务：仍然是 `original_remaining` masked point-level target。")
    lines.append("- 没有做 gate/router/selector，也没有删除样本或先硬分类响应类型。")
    lines.append("- 直接从 v241 `v241_tcn_mha_h96` 权重初始化。")
    lines.append("- 新增三件事：困难样本 hard weight、相对 v241 的 guard loss、v241 已做对正常样本的 teacher anchor。")
    lines.append(f"- 训练设备：`{device}`。")
    lines.append("")
    lines.append("## Validation 选择结果")
    lines.append("")
    lines.append(
        f"- best diagnostic model：`{best_name}`，validation score={float(best.validation_selection_score):.6f}，"
        f"accepted_as_next_candidate={bool(best.accepted_as_next_candidate)}。"
    )
    lines.append(
        f"- vs v236：normal max sample delta={float(best.normal_max_sample_delta_vs_v236):+.6f}，"
        f"normal max tail delta={float(best.normal_max_tail_delta_vs_v236):+.6f}，"
        f"observe mean tail delta={float(best.observe_later_mean_tail_delta_vs_v236_0to800):+.6f}，"
        f"strong 0-600 mean tail delta={float(best.strong_mean_tail_delta_vs_v236_0to600):+.6f}。"
    )
    lines.append(
        f"- vs v241：normal max tail delta={float(best.normal_max_tail_delta_vs_v241):+.6f}，"
        f"all mean tail delta={float(best.all_mean_tail_delta_vs_v241_0to800):+.6f}，"
        f"observe mean tail delta={float(best.observe_later_mean_tail_delta_vs_v241_0to800):+.6f}，"
        f"strong 400/1000 mean tail delta={float(best.strong_exception_mean_tail_delta_vs_v241_400_1000):+.6f}。"
    )
    lines.append(
        f"- validation 逐样本 guard：all regression rate={float(best.val_all_tail_regression_rate_vs_v241):.3f}，"
        f"normal regression rate={float(best.val_normal_tail_regression_rate_vs_v241):.3f}，"
        f"all p90 delta={float(best.val_all_p90_tail_delta_vs_v241):+.6f}，"
        f"max delta={float(best.val_max_tail_delta_vs_v241):+.6f}。"
    )
    lines.append(
        f"- checks：noharm_vs_v236={bool(best.noharm_vs_v236_pass)}，"
        f"upgrade_vs_v241={bool(best.upgrade_vs_v241_pass)}，"
        f"sample_guard_vs_v241={bool(best.sample_guard_vs_v241_pass)}，"
        f"meaningful_gain_vs_v241={bool(best.meaningful_gain_vs_v241)}。"
    )
    if accepted.empty:
        lines.append("- 结论：没有候选同时通过所有 validation 检查，当前最佳仍应保留 v241；v243 作为 guarded-loss 诊断。")
    else:
        accepted_name = str(accepted.sort_values("validation_selection_score").iloc[0]["model_name"])
        lines.append(f"- 结论：`{accepted_name}` 可以进入下一轮 locked audit。")
    lines.append("")
    lines.append("## Test 对照：v243 相对 v241")
    lines.append("")
    for bucket in ["all", "observe_later_like", "strong_steer", "normal_predictable"]:
        one = compare[compare["bucket"].eq(bucket)].copy().sort_values("delay_ms")
        if one.empty:
            continue
        lines.append(f"### {bucket}")
        delta_col = f"delta_steer_tail_rmse_mean__{best_name}_minus_{v241_name}"
        for _, row in one.iterrows():
            delta = float(row[delta_col]) if delta_col in one.columns and np.isfinite(float(row[delta_col])) else math.nan
            lines.append(f"- delay={int(row.delay_ms)}ms：tail delta vs v241={delta:+.6f}")
        lines.append("")
    lines.append("## 逐样本回退概览")
    lines.append("")
    lines.append("- validation：")
    for _, row in val_sample_summary.iterrows():
        lines.append(
            f"  - {row.bucket}: n={int(row.n)}，regressions={int(row.tail_regressions_vs_v241)}，"
            f"rate={float(row.tail_regression_rate_vs_v241):.3f}，"
            f"mean delta={float(row.mean_delta_tail_v243_minus_v241):+.6f}，"
            f"max delta={float(row.max_delta_tail_v243_minus_v241):+.6f}"
        )
    lines.append("- test：")
    for _, row in test_sample_summary.iterrows():
        lines.append(
            f"  - {row.bucket}: n={int(row.n)}，regressions={int(row.tail_regressions_vs_v241)}，"
            f"rate={float(row.tail_regression_rate_vs_v241):.3f}，"
            f"mean delta={float(row.mean_delta_tail_v243_minus_v241):+.6f}，"
            f"max delta={float(row.max_delta_tail_v243_minus_v241):+.6f}"
        )
    lines.append("")
    lines.append("## 候选级 test 稳定性补充")
    lines.append("")
    all_robust = candidate_robustness[candidate_robustness["bucket"].eq("all")].copy()
    all_robust = all_robust.sort_values(
        ["n_delay_tail_worse_vs_v241", "mean_tail_delta_test_vs_v241", "hard_weight_cap"]
    )
    robust_name = str(all_robust.iloc[0]["model_name"]) if not all_robust.empty else ""
    lines.append("- validation 排名第一只说明它在 validation 规则下最好；test 稳定性要单独看。")
    lines.append(
        "- 若按 test 的 all-bucket tail 均值和变差 delay 数看，"
        f"当前最均衡候选是 `{robust_name}`。这只是审查结果，不反向改 validation 选择。"
    )
    for _, row in all_robust.iterrows():
        lines.append(
            f"- {row.model_name}: all mean tail delta={float(row.mean_tail_delta_test_vs_v241):+.6f}，"
            f"all worse delays={int(row.n_delay_tail_worse_vs_v241)}/6。"
        )
    lines.append("- 机器可读表：`tables/v243_candidate_test_robustness_summary.csv`。")
    lines.append("")
    lines.append("## Guardrail")
    lines.append("")
    for key, value in guardrail.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## 主要产物")
    lines.append("")
    lines.append("- `tables/v243_model_selection_validation_guarded.csv`")
    lines.append("- `tables/v243_metrics_by_delay_and_bucket.csv`")
    lines.append("- `tables/v243_compare_vs_v236_v239_v241_original_remaining.csv`")
    lines.append("- `tables/v243_per_sample_delta_vs_v241.csv`")
    lines.append("- `tables/v243_per_sample_delta_summary_vs_v241.csv`")
    lines.append("- `tables/v243_worst_regressions_vs_v241.csv`")
    lines.append("- `tables/v243_training_weight_plan.csv`")
    lines.append("- `tables/v243_candidate_test_robustness_summary.csv`")
    lines.append("- `tables/v243_next_decision.csv`")
    lines.append(f"- ZIP：`{zip_path.name}`")
    lines.append("")
    (REPORTS / "v243_v241_guarded_finetune_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    clean_out_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v243] device={device}")

    print("[v243] loading v238/v239/v241 task data")
    data = V238.load_v236_data()
    x_base = V238.build_base_design_matrix(data)
    point_data = V238.build_point_dataset(data, x_base)
    point_masks = V238.split_point_masks(point_data, data.manifest)
    task_table, point_counts = V238.build_task_construction_tables(data, point_data)
    pred_v241, v241_name = load_v241_prediction()
    pred_v239, v239_name = load_v239_prediction()

    print("[v243] standardizing inputs with train-only scalers")
    scalers = V239.fit_scalers(data, point_data, point_masks)
    arrays = V239.standardize_arrays(data, point_data, scalers)

    # 三组候选都从 v241 初始化。区别在于 hard weight、guard 强度、teacher anchor 强度。
    # 第一版 v243 过于激进：训练集下降很快，但 validation 的真实分层指标没有超过 epoch 0。
    # 这里仍然保持困难样本明显加权，但降低学习率，并用曲线级 validation 指标选择快照。
    configs: List[Tuple[str, Dict[str, object]]] = [
        (
            "v243_metric_hard24_guard04",
            {
                "batch_size": 1536,
                "stage1_epochs": 8,
                "stage2_epochs": 34,
                "stage1_lr": 1.0e-4,
                "stage2_lr": 3.0e-5,
                "min_lr": 5e-6,
                "weight_decay": 5e-4,
                "patience": 8,
                "curve_eval_every": 2,
                "grad_clip": 4.0,
                "guard_alpha": 0.4,
                "teacher_anchor_beta": 0.0,
                "val_guard_score_alpha": 0.0,
                "guard_margin_original": 0.020,
                "hard_weight_cap": 2.4,
                "point_weight_cap": 2.8,
                "tail_point_boost": 1.25,
                "tail_q70_boost": 0.30,
                "tail_q85_boost": 0.45,
                "tail_q95_boost": 0.30,
                "strong_boost": 0.35,
                "strong_under_boost": 0.35,
                "observe_boost": 0.20,
                "complex_boost": 0.30,
                "extreme_peak_boost": 0.15,
                "strong_exception_boost": 0.25,
                "guard_good_normal_extra": 0.55,
                "guard_bad_teacher_extra": 0.15,
                "guard_weight_cap": 2.0,
                "anchor_weight_value": 1.0,
            },
        ),
        (
            "v243_metric_hard30_guard06_anchor04",
            {
                "batch_size": 1536,
                "stage1_epochs": 8,
                "stage2_epochs": 36,
                "stage1_lr": 1.5e-4,
                "stage2_lr": 4.0e-5,
                "min_lr": 5e-6,
                "weight_decay": 5e-4,
                "patience": 9,
                "curve_eval_every": 2,
                "grad_clip": 4.0,
                "guard_alpha": 0.6,
                "teacher_anchor_beta": 0.04,
                "val_guard_score_alpha": 0.0,
                "guard_margin_original": 0.016,
                "hard_weight_cap": 3.0,
                "point_weight_cap": 3.4,
                "tail_point_boost": 1.35,
                "tail_q70_boost": 0.38,
                "tail_q85_boost": 0.60,
                "tail_q95_boost": 0.38,
                "strong_boost": 0.45,
                "strong_under_boost": 0.45,
                "observe_boost": 0.25,
                "complex_boost": 0.40,
                "extreme_peak_boost": 0.20,
                "strong_exception_boost": 0.32,
                "guard_good_normal_extra": 0.85,
                "guard_bad_teacher_extra": 0.20,
                "guard_weight_cap": 2.5,
                "anchor_weight_value": 1.0,
            },
        ),
        (
            "v243_metric_hard36_guard08",
            {
                "batch_size": 1536,
                "stage1_epochs": 8,
                "stage2_epochs": 36,
                "stage1_lr": 2.0e-4,
                "stage2_lr": 6.0e-5,
                "min_lr": 5e-6,
                "weight_decay": 4e-4,
                "patience": 10,
                "curve_eval_every": 2,
                "grad_clip": 4.0,
                "guard_alpha": 0.8,
                "teacher_anchor_beta": 0.0,
                "val_guard_score_alpha": 0.0,
                "guard_margin_original": 0.014,
                "hard_weight_cap": 3.6,
                "point_weight_cap": 4.0,
                "tail_point_boost": 1.45,
                "tail_q70_boost": 0.50,
                "tail_q85_boost": 0.80,
                "tail_q95_boost": 0.50,
                "strong_boost": 0.55,
                "strong_under_boost": 0.60,
                "observe_boost": 0.30,
                "complex_boost": 0.50,
                "extreme_peak_boost": 0.25,
                "strong_exception_boost": 0.40,
                "guard_good_normal_extra": 1.00,
                "guard_bad_teacher_extra": 0.25,
                "guard_weight_cap": 3.0,
                "anchor_weight_value": 1.0,
            },
        ),
    ]

    runs: List[GuardedRun] = []
    weight_tables: List[pd.DataFrame] = []
    for model_name, config in configs:
        print(f"[v243] training {model_name}")
        run, weight_stats = train_guarded_candidate(
            model_name=model_name,
            raw_config=config,
            data=data,
            point_data=point_data,
            arrays=arrays,
            scalers=scalers,
            point_masks=point_masks,
            pred_v241=pred_v241,
            v241_name=v241_name,
            device=device,
        )
        runs.append(run)
        weight_tables.append(weight_stats)
        print(
            f"[v243] {model_name} best_epoch={run.best_epoch} "
            f"best_val_loss={run.best_val_loss:.6f} base={run.best_val_base_loss:.6f} guard={run.best_val_guard_loss:.6f}"
        )

    print("[v243] computing metrics and validation decisions")
    y_true_curve = data.y_future[:, :, 0].astype(np.float32)
    pred_by_model: Dict[str, np.ndarray] = {
        "v236_joint_ridge_existing": data.pred_v236[:, :, 0].astype(np.float32),
        "v238_selected_original_remaining_point_model": V239.load_v238_predictions(),
        v239_name: pred_v239.astype(np.float32),
        v241_name: pred_v241.astype(np.float32),
    }
    for run in runs:
        pred_by_model[run.model_name] = run.pred_curve.astype(np.float32)

    metrics = V238.compute_metrics_table(
        y_true_curve=y_true_curve,
        pred_by_model=pred_by_model,
        manifest=data.manifest,
        eval_modes=["original_remaining", "receding_2s_diagnostic"],
    )

    per_sample_by_model: Dict[str, pd.DataFrame] = {}
    selection_rows: List[Dict[str, object]] = []
    for run in runs:
        per_sample_delta = build_per_sample_delta_table(
            y_true_curve=y_true_curve,
            pred_ref=pred_v241.astype(np.float32),
            pred_candidate=run.pred_curve.astype(np.float32),
            manifest=data.manifest,
            ref_name=v241_name,
            candidate_name=run.model_name,
        )
        per_sample_by_model[run.model_name] = per_sample_delta
        row = candidate_validation_decision(metrics, run.model_name, v241_name, per_sample_delta)
        row.update(
            {
                "config_json": json.dumps(run.config, ensure_ascii=False, sort_keys=True),
                "best_epoch": run.best_epoch,
                "best_val_loss": run.best_val_loss,
                "best_val_base_loss": run.best_val_base_loss,
                "best_val_guard_loss": run.best_val_guard_loss,
                "training_seconds": run.training_seconds,
            }
        )
        selection_rows.append(row)

    selection = pd.DataFrame(selection_rows).sort_values("validation_selection_score").reset_index(drop=True)
    selection["validation_rank"] = np.arange(1, len(selection) + 1)
    best_name = str(selection.iloc[0]["model_name"])
    best_run = next(run for run in runs if run.model_name == best_name)
    best_per_sample_delta = per_sample_by_model[best_name]
    val_sample_summary = summarize_per_sample_delta(best_per_sample_delta, "val")
    test_sample_summary = summarize_per_sample_delta(best_per_sample_delta, "test")
    per_sample_summary = pd.concat([val_sample_summary, test_sample_summary], ignore_index=True)

    compare = build_compare_table(metrics, pred_by_model.keys(), v241_name)
    candidate_robustness = build_candidate_test_robustness_summary(
        compare=compare,
        selection=selection,
        candidate_names=[run.model_name for run in runs],
        v241_name=v241_name,
    )
    next_decision = build_next_decision(selection)
    split_check = V238.split_integrity_check(data.manifest)
    guardrail = build_guardrail_json(selection, split_check)
    if not bool(guardrail["pass"]):
        raise AssertionError("v243 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    figure_paths = plot_figures(compare, best_name, v241_name, v239_name)

    print("[v243] writing outputs")
    write_csv(task_table, TABLES / "v243_task_construction_audit.csv")
    write_csv(point_counts, TABLES / "v243_point_training_rows_by_delay.csv")
    write_csv(selection, TABLES / "v243_model_selection_validation_guarded.csv")
    write_csv(metrics, TABLES / "v243_metrics_by_delay_and_bucket.csv")
    write_csv(compare, TABLES / "v243_compare_vs_v236_v239_v241_original_remaining.csv")
    write_csv(candidate_robustness, TABLES / "v243_candidate_test_robustness_summary.csv")
    write_csv(pd.concat([run.training_history for run in runs], ignore_index=True), TABLES / "v243_training_history.csv")
    write_csv(pd.concat(weight_tables, ignore_index=True), TABLES / "v243_training_weight_plan.csv")
    write_csv(best_per_sample_delta, TABLES / "v243_per_sample_delta_vs_v241.csv")
    write_csv(per_sample_summary, TABLES / "v243_per_sample_delta_summary_vs_v241.csv")
    write_csv(
        best_per_sample_delta[best_per_sample_delta["split"].eq("test")]
        .sort_values("delta_tail_v243_minus_v241", ascending=False)
        .head(100),
        TABLES / "v243_worst_regressions_vs_v241.csv",
    )
    write_csv(
        best_per_sample_delta[best_per_sample_delta["split"].eq("test")]
        .sort_values("delta_tail_v243_minus_v241", ascending=True)
        .head(100),
        TABLES / "v243_top_improvements_vs_v241.csv",
    )
    write_csv(next_decision, TABLES / "v243_next_decision.csv")
    write_csv(split_check, TABLES / "v243_split_integrity_check.csv")

    np.savez_compressed(
        OUT / "v243_v241_guarded_finetune_predictions.npz",
        y_true_steering_delta=y_true_curve.astype(np.float32),
        pred_v236_steering_delta=pred_by_model["v236_joint_ridge_existing"].astype(np.float32),
        pred_v238_steering_delta=pred_by_model["v238_selected_original_remaining_point_model"].astype(np.float32),
        pred_v239_steering_delta=pred_v239.astype(np.float32),
        pred_v241_steering_delta=pred_v241.astype(np.float32),
        pred_v243_best_guarded_steering_delta=best_run.pred_curve.astype(np.float32),
        best_guarded_model=np.array([best_name], dtype="U120"),
        source_v241_model=np.array([v241_name], dtype="U120"),
        source_v239_model=np.array([v239_name], dtype="U120"),
        delay_ms=data.manifest["delay_ms"].to_numpy(dtype=np.int32),
        split=data.manifest["split"].astype(str).to_numpy(dtype="U16"),
        event_uid=data.manifest["event_uid"].astype(str).to_numpy(dtype="U160"),
        future_grid_s=FUTURE_GRID.astype(np.float32),
        original_remaining_valid=V238.build_original_remaining_mask(data.manifest)[0].astype(np.bool_),
    )

    scalers_payload = {
        "hist_mean": scalers.hist_mean,
        "hist_std": scalers.hist_std,
        "road_mean": scalers.road_mean,
        "road_std": scalers.road_std,
        "phase_mean": scalers.phase_mean,
        "phase_std": scalers.phase_std,
        "point_mean": scalers.point_mean,
        "point_std": scalers.point_std,
        "y_mean": scalers.y_mean,
        "y_std": scalers.y_std,
    }
    torch.save(
        {
            "model_name": best_name,
            "model_class": "StrongerTemporalQueryAttention",
            "state_dict": best_run.state_dict,
            "config": best_run.config,
            "scalers": scalers_payload,
            "selection": selection.to_dict(orient="records"),
            "source_v241_model": v241_name,
            "source_v241_checkpoint": str(V241_MODEL),
        },
        MODELS / "v243_best_guarded_finetune_diagnostic.pt",
    )
    with (MODELS / "v243_scalers_and_selection.pkl").open("wb") as f:
        pickle.dump({"scalers": scalers_payload, "selection": selection}, f)

    write_input_hashes()
    leakage = {
        "same_event_uid_cross_split_count": int(split_check["split_check_status"].eq("fail").sum()),
        "test_used_for_selection": bool(selection["test_used_for_selection"].astype(bool).any()),
        "pass": int(split_check["split_check_status"].eq("fail").sum()) == 0
        and not bool(selection["test_used_for_selection"].astype(bool).any()),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "leakage_check.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")
    run_manifest = {
        "stage": "v243_v241_guarded_finetune",
        "created_by": Path(__file__).name,
        "output_dir": str(OUT),
        "source_v241_dir": str(V241_DIR),
        "n_rolling_samples": int(len(data.manifest)),
        "n_events": int(data.manifest["event_uid"].nunique()),
        "device": str(device),
        "guarded_candidates": [run.model_name for run in runs],
        "best_diagnostic_model": best_name,
        "accepted_as_next_candidate": bool(selection["accepted_as_next_candidate"].astype(bool).any()),
        "figures": [str(path.relative_to(OUT)).replace("\\", "/") for path in figure_paths],
    }
    (LOGS / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = zip_outputs()
    write_report(
        selection=selection,
        next_decision=next_decision,
        compare=compare,
        candidate_robustness=candidate_robustness,
        val_sample_summary=val_sample_summary,
        test_sample_summary=test_sample_summary,
        guardrail=guardrail,
        device=device,
        v241_name=v241_name,
        v239_name=v239_name,
        zip_path=zip_path,
    )
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = zip_outputs()

    print("[v243] finished")
    print(f"output_dir={OUT}")
    print(f"best_diagnostic_model={best_name}")
    print(f"accepted_as_next_candidate={bool(selection['accepted_as_next_candidate'].astype(bool).any())}")
    print(f"report={REPORTS / 'v243_v241_guarded_finetune_cn.md'}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
