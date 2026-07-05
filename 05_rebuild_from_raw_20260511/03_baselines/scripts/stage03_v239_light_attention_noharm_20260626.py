#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v239 轻量 temporal attention + no-harm 约束实验。

本轮承接 v238 的结论：
- 保留 original_remaining masked point-level target；
- 不把 response type 做成硬路由；
- 不扩大到完整 Transformer；
- 只新增同一个模型内部的轻量时间注意力，让模型学习历史和道路预瞄中哪些时刻更重要；
- 模型是否可作为下一步候选，只由 validation no-harm 规则决定，test 只做固定后报告。

注意：
v239 的目标是检查 attention 是否能在不伤普通样本的前提下继续改善难例。
如果 attention 改善难例但 normal_predictable 被伤害，则只能作为诊断原型，不能作为 formal replacement。
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

# 限制底层线程，避免 Windows + MKL/OpenMP 在 sklearn/torch 混用时出现不稳定。
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

V238_SCRIPT = BASELINES / "scripts" / "stage03_v238_task_model_redesign_20260626.py"
V238_DIR = BASELINES / "v238_task_model_redesign_20260626"
V238_PRED = V238_DIR / "v238_original_remaining_predictions.npz"
V238_SELECTION = V238_DIR / "tables" / "v238_model_selection_validation_only.csv"
V238_DECISION = V238_DIR / "tables" / "v238_next_model_decision.csv"

OUT = BASELINES / "v239_light_attention_noharm_20260626"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
MODELS = OUT / "models"

DELAY_MS = [0, 200, 400, 600, 800, 1000]
FORMAL_DELAY_MAX_MS = 800
STRONG_DELAY_MAX_MS = 600
NOHARM_TOL = 0.02
SEED = 239

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_v238_module():
    """导入 v238 脚本，复用其数据读取、任务构造和指标函数。"""

    if not V238_SCRIPT.exists():
        raise FileNotFoundError(f"找不到 v238 脚本：{V238_SCRIPT}")
    spec = importlib.util.spec_from_file_location("stage03_v238_task_model_redesign_20260626", V238_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 v238 脚本：{V238_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V238 = import_v238_module()
FUTURE_GRID = V238.FUTURE_GRID


@dataclass
class SequenceScalers:
    """序列与 point 特征的标准化参数。"""

    hist_mean: np.ndarray
    hist_std: np.ndarray
    road_mean: np.ndarray
    road_std: np.ndarray
    phase_mean: np.ndarray
    phase_std: np.ndarray
    point_mean: np.ndarray
    point_std: np.ndarray
    y_mean: float
    y_std: float


@dataclass
class AttentionRun:
    """一个 attention 候选的训练结果。"""

    model_name: str
    config: Dict[str, object]
    state_dict: Dict[str, torch.Tensor]
    pred_curve: np.ndarray
    training_history: pd.DataFrame
    training_seconds: float
    best_epoch: int
    best_val_loss: float


def ensure_dirs() -> None:
    """创建 v239 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v239 自己的输出目录。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 统一使用 utf-8-sig，方便 Windows 中文环境直接打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算文件 SHA256。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int = SEED) -> None:
    """固定 numpy / torch 随机种子。"""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def safe_mean_std(values: np.ndarray, axis: int | Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """基于 train 数据计算均值和标准差，并处理全 NaN/零方差。"""

    mean = np.nanmean(values, axis=axis)
    std = np.nanstd(values, axis=axis)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    mean[~np.isfinite(mean)] = 0.0
    std[~np.isfinite(std)] = 1.0
    std[std < 1e-6] = 1.0
    return mean, std


def apply_standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """填补 NaN 并标准化。"""

    out = values.astype(np.float32, copy=True)
    if out.ndim == 3:
        fill = mean.reshape(1, 1, -1)
        scale = std.reshape(1, 1, -1)
    elif out.ndim == 2:
        fill = mean.reshape(1, -1)
        scale = std.reshape(1, -1)
    else:
        raise ValueError(f"不支持的标准化维度：{out.shape}")
    out = np.where(np.isfinite(out), out, fill)
    return ((out - fill) / scale).astype(np.float32)


def fit_scalers(data, point_data, point_masks: Dict[str, np.ndarray]) -> SequenceScalers:
    """只用 train split 拟合序列、phase、point 和 target 标准化。"""

    train_sample = data.manifest["split"].astype(str).to_numpy() == "train"
    train_point = point_masks["train"]
    point_extra = point_data.x_point_all[:, -len(V238.POINT_EXTRA_FEATURE_NAMES) :].astype(np.float32)

    hist_mean, hist_std = safe_mean_std(data.x_hist[train_sample].reshape(-1, data.x_hist.shape[-1]), axis=0)
    road_mean, road_std = safe_mean_std(data.x_road[train_sample].reshape(-1, data.x_road.shape[-1]), axis=0)
    phase_mean, phase_std = safe_mean_std(data.x_phase[train_sample], axis=0)
    point_mean, point_std = safe_mean_std(point_extra[train_point], axis=0)

    y_train = point_data.y_point_all[train_point].astype(np.float32)
    y_mean = float(np.nanmean(y_train))
    y_std = float(np.nanstd(y_train))
    if not np.isfinite(y_mean):
        y_mean = 0.0
    if not np.isfinite(y_std) or y_std < 1e-6:
        y_std = 1.0

    return SequenceScalers(
        hist_mean=hist_mean,
        hist_std=hist_std,
        road_mean=road_mean,
        road_std=road_std,
        phase_mean=phase_mean,
        phase_std=phase_std,
        point_mean=point_mean,
        point_std=point_std,
        y_mean=y_mean,
        y_std=y_std,
    )


def standardize_arrays(data, point_data, scalers: SequenceScalers) -> Dict[str, np.ndarray]:
    """生成 attention 模型训练所需的标准化数组。"""

    point_extra = point_data.x_point_all[:, -len(V238.POINT_EXTRA_FEATURE_NAMES) :].astype(np.float32)
    arrays = {
        "hist": apply_standardize(data.x_hist, scalers.hist_mean, scalers.hist_std),
        "road": apply_standardize(data.x_road, scalers.road_mean, scalers.road_std),
        "phase": apply_standardize(data.x_phase, scalers.phase_mean, scalers.phase_std),
        "point": apply_standardize(point_extra, scalers.point_mean, scalers.point_std),
        "y": ((point_data.y_point_all.astype(np.float32) - scalers.y_mean) / scalers.y_std).astype(np.float32),
    }
    return arrays


class PointSequenceDataset(Dataset):
    """point-level 数据集：每个样本返回一个 rolling sample 的序列输入和一个 future point。"""

    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        point_data,
        point_mask: np.ndarray,
    ) -> None:
        self.hist = arrays["hist"]
        self.road = arrays["road"]
        self.phase = arrays["phase"]
        self.point = arrays["point"]
        self.y = arrays["y"]
        self.sample_index = point_data.sample_index_all.astype(np.int64)
        self.point_weight = point_data.point_weight_all.astype(np.float32)
        self.indices = np.where(point_mask)[0].astype(np.int64)

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
            "weight": torch.tensor(self.point_weight[point_idx], dtype=torch.float32),
        }


class LightTemporalAttention(nn.Module):
    """
    轻量时间注意力模型。

    它不是 router/gate，也不先判断响应类型。
    它只在同一个模型内部，根据 phase + future point query，
    对历史序列和道路预瞄序列分别做 soft attention。
    """

    def __init__(
        self,
        hist_dim: int,
        road_dim: int,
        phase_dim: int,
        point_dim: int,
        hidden_dim: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hist_proj = nn.Sequential(nn.Linear(hist_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.road_proj = nn.Sequential(nn.Linear(road_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.query = nn.Sequential(
            nn.Linear(phase_dim + point_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.hist_score = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.road_score = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + phase_dim + point_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, max(16, head_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(16, head_dim // 2), 1),
        )

    def attend(self, seq_emb: torch.Tensor, query: torch.Tensor, scorer: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
        """对一段时间序列做点积 attention。"""

        q = scorer(query).unsqueeze(1)
        scores = (seq_emb * q).sum(dim=-1) / math.sqrt(float(self.hidden_dim))
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(seq_emb * weights.unsqueeze(-1), dim=1)
        return context, weights

    def forward(self, hist: torch.Tensor, road: torch.Tensor, phase: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        hist_emb = self.hist_proj(hist)
        road_emb = self.road_proj(road)
        q_input = torch.cat([phase, point], dim=1)
        query = self.query(q_input)
        hist_ctx, _ = self.attend(hist_emb, query, self.hist_score)
        road_ctx, _ = self.attend(road_emb, query, self.road_score)
        out = self.head(torch.cat([hist_ctx, road_ctx, query, phase, point], dim=1))
        return out.squeeze(1)


def weighted_mse(pred: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """加权 MSE。"""

    return torch.sum(torch.square(pred - y) * weight) / torch.clamp(torch.sum(weight), min=1e-6)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """运行一个训练或验证 epoch。"""

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_weight = 0.0
    for batch in loader:
        hist = batch["hist"].to(device=device, dtype=torch.float32)
        road = batch["road"].to(device=device, dtype=torch.float32)
        phase = batch["phase"].to(device=device, dtype=torch.float32)
        point = batch["point"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.float32)
        weight = batch["weight"].to(device=device, dtype=torch.float32)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(hist, road, phase, point)
        loss = weighted_mse(pred, y, weight)
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        wsum = float(torch.sum(weight).detach().cpu().item())
        total_loss += float(loss.detach().cpu().item()) * wsum
        total_weight += wsum
    return total_loss / max(total_weight, 1e-6)


def predict_all_points(
    model: nn.Module,
    arrays: Dict[str, np.ndarray],
    point_data,
    scalers: SequenceScalers,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """对所有 21 个 future point 生成预测，并还原成 N x 21 曲线。"""

    model.eval()
    n_points = len(point_data.y_point_all)
    pred_scaled = np.empty(n_points, dtype=np.float32)
    sample_idx = point_data.sample_index_all.astype(np.int64)
    with torch.no_grad():
        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            sidx = sample_idx[start:end]
            hist = torch.from_numpy(arrays["hist"][sidx]).to(device=device, dtype=torch.float32)
            road = torch.from_numpy(arrays["road"][sidx]).to(device=device, dtype=torch.float32)
            phase = torch.from_numpy(arrays["phase"][sidx]).to(device=device, dtype=torch.float32)
            point = torch.from_numpy(arrays["point"][start:end]).to(device=device, dtype=torch.float32)
            pred = model(hist, road, phase, point)
            pred_scaled[start:end] = pred.detach().cpu().numpy().astype(np.float32)
    pred = pred_scaled * scalers.y_std + scalers.y_mean
    pred_curve = V238.predict_curve_from_point_predictions(
        point_pred=pred.astype(np.float32),
        sample_index_all=point_data.sample_index_all,
        time_index_all=point_data.time_index_all,
        n_samples=int(point_data.sample_index_all.max()) + 1,
    )
    return pred_curve.astype(np.float32)


def train_attention_candidate(
    model_name: str,
    config: Dict[str, object],
    data,
    point_data,
    arrays: Dict[str, np.ndarray],
    scalers: SequenceScalers,
    point_masks: Dict[str, np.ndarray],
    device: torch.device,
) -> AttentionRun:
    """训练一个轻量 attention 候选。"""

    train_dataset = PointSequenceDataset(arrays, point_data, point_masks["train"])
    val_dataset = PointSequenceDataset(arrays, point_data, point_masks["val"])
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
    model = LightTemporalAttention(
        hist_dim=data.x_hist.shape[-1],
        road_dim=data.x_road.shape[-1],
        phase_dim=data.x_phase.shape[-1],
        point_dim=len(V238.POINT_EXTRA_FEATURE_NAMES),
        hidden_dim=int(config["hidden_dim"]),
        head_dim=int(config["head_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    max_epochs = int(config["max_epochs"])
    patience = int(config["patience"])
    best_val = math.inf
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: List[Dict[str, object]] = []
    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device, None)
        history.append(
            {
                "model_name": model_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise AssertionError(f"{model_name} 未产生 best_state")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    pred_curve = predict_all_points(model, arrays, point_data, scalers, device, batch_size=batch_size * 4)
    return AttentionRun(
        model_name=model_name,
        config=config,
        state_dict=best_state,
        pred_curve=pred_curve,
        training_history=pd.DataFrame(history),
        training_seconds=float(time.time() - start_time),
        best_epoch=int(best_epoch),
        best_val_loss=float(best_val),
    )


def load_v238_predictions() -> np.ndarray:
    """读取 v238 selected MLP 的预测曲线，用于对照。"""

    if not V238_PRED.exists():
        raise FileNotFoundError(f"缺少 v238 prediction npz：{V238_PRED}")
    with np.load(V238_PRED, allow_pickle=False) as pred:
        return pred["pred_v238_steering_delta"].astype(np.float32)


def delay_subset(df: pd.DataFrame, bucket: str, max_delay: int) -> pd.DataFrame:
    """抽取某个 bucket 的 delay 子集。"""

    return df[df["bucket"].eq(bucket) & (df["delay_ms"].astype(int) <= max_delay)].copy()


def candidate_validation_decision(metrics: pd.DataFrame, candidate_name: str) -> Dict[str, object]:
    """基于 validation 指标判断候选 attention 是否满足 no-harm。"""

    val = metrics[metrics["split"].eq("val") & metrics["eval_mode"].eq("original_remaining")].copy()
    cand = val[val["model_name"].eq(candidate_name)].copy()
    ref = val[val["model_name"].eq("v236_joint_ridge_existing")].copy()
    if cand.empty or ref.empty:
        raise AssertionError(f"{candidate_name} 或 v236 validation 指标为空")
    merged = cand.merge(
        ref,
        on=["split", "bucket", "delay_ms", "eval_mode"],
        suffixes=("_candidate", "_ref"),
    )
    merged["delta_sample"] = merged["steer_sample_rmse_mean_candidate"] - merged["steer_sample_rmse_mean_ref"]
    merged["delta_tail"] = merged["steer_tail_rmse_mean_candidate"] - merged["steer_tail_rmse_mean_ref"]

    normal = delay_subset(merged, "normal_predictable", FORMAL_DELAY_MAX_MS)
    all_bucket = delay_subset(merged, "all", FORMAL_DELAY_MAX_MS)
    observe = delay_subset(merged, "observe_later_like", FORMAL_DELAY_MAX_MS)
    strong = delay_subset(merged, "strong_steer", STRONG_DELAY_MAX_MS)

    normal_max_sample_delta = float(normal["delta_sample"].max()) if not normal.empty else math.inf
    normal_max_tail_delta = float(normal["delta_tail"].max()) if not normal.empty else math.inf
    all_max_sample_delta = float(all_bucket["delta_sample"].max()) if not all_bucket.empty else math.inf
    observe_mean_tail_delta = float(observe["delta_tail"].mean()) if not observe.empty else math.inf
    strong_mean_tail_delta = float(strong["delta_tail"].mean()) if not strong.empty else math.inf

    normal_noharm = normal_max_sample_delta <= NOHARM_TOL and normal_max_tail_delta <= NOHARM_TOL
    all_noharm = all_max_sample_delta <= NOHARM_TOL
    observe_gain = observe_mean_tail_delta <= 0.0
    strong_gain = strong_mean_tail_delta <= 0.0
    pass_noharm = bool(normal_noharm and all_noharm and observe_gain and strong_gain)

    penalty = (
        max(0.0, normal_max_sample_delta - NOHARM_TOL)
        + max(0.0, normal_max_tail_delta - NOHARM_TOL)
        + max(0.0, all_max_sample_delta - NOHARM_TOL)
        + max(0.0, observe_mean_tail_delta)
        + max(0.0, strong_mean_tail_delta)
    )
    selection_score = (
        float(cand[cand["bucket"].eq("all")]["steer_sample_rmse_mean"].mean())
        + 0.50 * float(cand[cand["bucket"].eq("all")]["steer_tail_rmse_mean"].mean())
        + 10.0 * penalty
    )
    return {
        "model_name": candidate_name,
        "selected_by": "validation_noharm_only",
        "test_used_for_selection": False,
        "formal_delay_max_ms": FORMAL_DELAY_MAX_MS,
        "normal_noharm_tolerance": NOHARM_TOL,
        "normal_max_sample_delta_vs_v236": normal_max_sample_delta,
        "normal_max_tail_delta_vs_v236": normal_max_tail_delta,
        "all_max_sample_delta_vs_v236": all_max_sample_delta,
        "observe_later_mean_tail_delta_vs_v236_0to800": observe_mean_tail_delta,
        "strong_mean_tail_delta_vs_v236_0to600": strong_mean_tail_delta,
        "normal_noharm_pass": bool(normal_noharm),
        "all_noharm_pass": bool(all_noharm),
        "observe_later_gain_pass": bool(observe_gain),
        "strong_gain_pass": bool(strong_gain),
        "validation_noharm_pass": pass_noharm,
        "validation_selection_score": selection_score,
    }


def build_compare_table(metrics: pd.DataFrame, model_names: Iterable[str]) -> pd.DataFrame:
    """生成 test original_remaining 下相对 v236 的对照表。"""

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
        sample_ref = "steer_sample_rmse_mean__v236_joint_ridge_existing"
        tail_ref = "steer_tail_rmse_mean__v236_joint_ridge_existing"
        sample_col = f"steer_sample_rmse_mean__{model}"
        tail_col = f"steer_tail_rmse_mean__{model}"
        if sample_ref in pivot.columns and sample_col in pivot.columns:
            pivot[f"delta_sample__{model}_minus_v236"] = pivot[sample_col] - pivot[sample_ref]
        if tail_ref in pivot.columns and tail_col in pivot.columns:
            pivot[f"delta_tail__{model}_minus_v236"] = pivot[tail_col] - pivot[tail_ref]
    return pivot


def build_next_decision(selection: pd.DataFrame) -> pd.DataFrame:
    """根据 validation no-harm 结果形成下一步决策表。"""

    passed = selection[selection["validation_noharm_pass"].astype(bool)].copy()
    if passed.empty:
        accepted = False
        selected_name = ""
        reason = "No attention candidate passed validation no-harm; keep attention diagnostic-only."
        next_task = "v240_attention_or_mlp_with_stronger_noharm_or_regularization"
    else:
        accepted = True
        selected_name = str(passed.sort_values("validation_selection_score").iloc[0]["model_name"])
        reason = f"{selected_name} passed validation no-harm and can be treated as the next candidate, still not formal headline."
        next_task = "v240_locked_test_report_for_attention_candidate"
    rows = [
        {
            "decision_item": "accept_attention_as_candidate",
            "decision": accepted,
            "model_name": selected_name,
            "reason": reason,
        },
        {
            "decision_item": "keep_original_remaining_task",
            "decision": True,
            "model_name": "",
            "reason": "v239 keeps v238 original_remaining masked target; this remains the correct task construction.",
        },
        {
            "decision_item": "formal_replacement_allowed",
            "decision": False,
            "model_name": "",
            "reason": "This run is a prototype experiment; formal headline remains locked to v225/v226.",
        },
        {
            "decision_item": "recommended_next_task",
            "decision": next_task,
            "model_name": selected_name,
            "reason": "Continue only through validation-bounded no-harm; do not expand to router/gate or full Transformer.",
        },
    ]
    return pd.DataFrame(rows)


def plot_figures(compare: pd.DataFrame, attention_names: List[str], selected_for_plot: str) -> List[Path]:
    """生成 attention 与 v236/v238 的核心对照图。"""

    paths: List[Path] = []
    models = ["v238_selected_original_remaining_point_model", selected_for_plot]
    for bucket in ["observe_later_like", "strong_steer", "normal_predictable"]:
        one = compare[compare["bucket"].eq(bucket)].copy().sort_values("delay_ms")
        if one.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        ref_col = "steer_tail_rmse_mean__v236_joint_ridge_existing"
        if ref_col in one.columns:
            ax.plot(one["delay_ms"], one[ref_col], marker="o", color="#777777", label="v236 existing")
        for model_name, color in zip(models, ["#1f77b4", "#d62728"]):
            col = f"steer_tail_rmse_mean__{model_name}"
            if col in one.columns:
                ax.plot(one["delay_ms"], one[col], marker="o", label=model_name, color=color)
        ax.set_xlabel("Observation delay (ms)")
        ax.set_ylabel("Original-remaining tail RMSE")
        ax.set_title(f"v239 light attention comparison: {bucket}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = FIGURES / f"v239_attention_tail_compare_{bucket}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def write_input_hashes() -> None:
    """写入关键输入文件哈希。"""

    paths = [
        V238_SCRIPT,
        V238_PRED,
        V238_SELECTION,
        V238_DECISION,
        V238.V236_ARRAYS,
        V238.V236_MANIFEST,
    ]
    rows = [{"path": str(path), "sha256": file_sha256(path), "bytes": int(path.stat().st_size)} for path in paths]
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def split_integrity_check(manifest: pd.DataFrame) -> pd.DataFrame:
    """复用 v238 split 检查，确认同一 event_uid 不跨 split。"""

    return V238.split_integrity_check(manifest)


def build_guardrail_json(selection: pd.DataFrame, split_check: pd.DataFrame) -> Dict[str, object]:
    """记录 v239 guardrail。"""

    checks = {
        "task_base": "v238_original_remaining_masked_point_level_target",
        "attention_type": "light_temporal_attention_inside_single_model",
        "full_transformer_used": False,
        "gate_router_selector_created": False,
        "response_type_hard_routing_created": False,
        "observe_later_like_deleted": False,
        "formal_headline_changed": False,
        "test_used_for_selection": bool(selection["test_used_for_selection"].astype(bool).any()),
        "same_event_uid_cross_split_count": int(split_check["split_check_status"].eq("fail").sum()),
        "validation_noharm_rule_used": True,
        "pass": False,
    }
    checks["pass"] = (
        not checks["full_transformer_used"]
        and not checks["gate_router_selector_created"]
        and not checks["response_type_hard_routing_created"]
        and not checks["observe_later_like_deleted"]
        and not checks["formal_headline_changed"]
        and not checks["test_used_for_selection"]
        and checks["same_event_uid_cross_split_count"] == 0
        and checks["validation_noharm_rule_used"]
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
    """打包 v239 输出。"""

    zip_path = OUT / "v239_light_attention_noharm_pack.zip"
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
    guardrail: Dict[str, object],
    device: torch.device,
    zip_path: Path,
) -> None:
    """写 v239 中文报告。"""

    best = selection.sort_values("validation_selection_score").iloc[0]
    passed = selection[selection["validation_noharm_pass"].astype(bool)].copy()
    accepted = not passed.empty
    lines: List[str] = []
    lines.append("# v239 轻量 temporal attention + no-harm 约束报告")
    lines.append("")
    lines.append("## 本轮做了什么")
    lines.append("")
    lines.append("- 继续保留 v238 的 `original_remaining` masked point-level target。")
    lines.append("- 新增轻量 temporal attention：历史序列和道路预瞄序列分别做 soft attention。")
    lines.append("- attention 只是在同一个模型内部给时间点加权，不是 gate/router/selector，也不是响应类型硬分类。")
    lines.append("- 只用 validation no-harm 规则判断 attention 是否可作为下一步候选；test 不参与选择。")
    lines.append(f"- 训练设备：`{device}`。")
    lines.append("")
    lines.append("## Validation 选择")
    lines.append("")
    lines.append(
        f"- best diagnostic model：`{best.model_name}`，validation score={float(best.validation_selection_score):.6f}，"
        f"no-harm pass={bool(best.validation_noharm_pass)}。"
    )
    if accepted:
        selected = passed.sort_values("validation_selection_score").iloc[0]
        lines.append(f"- 有 attention 候选通过 no-harm：`{selected.model_name}`。")
    else:
        lines.append("- 没有 attention 候选通过 validation no-harm，因此本轮 attention 只能作为诊断原型。")
    lines.append("")
    lines.append("## Test original_remaining 重点对照")
    lines.append("")
    plot_model = str(best.model_name)
    for bucket in ["observe_later_like", "strong_steer", "normal_predictable"]:
        one = compare[compare["bucket"].eq(bucket)].sort_values("delay_ms")
        if one.empty:
            continue
        lines.append(f"### {bucket}")
        delta_col = f"delta_tail__{plot_model}_minus_v236"
        sample_col = f"delta_sample__{plot_model}_minus_v236"
        for row in one.itertuples(index=False):
            delta_tail = getattr(row, delta_col, math.nan)
            delta_sample = getattr(row, sample_col, math.nan)
            lines.append(
                f"- delay={int(row.delay_ms)}ms：attention tail delta={float(delta_tail):+.6f}，"
                f"sample delta={float(delta_sample):+.6f}"
            )
        lines.append("")
    lines.append("## 下一步决策")
    lines.append("")
    for row in next_decision.itertuples(index=False):
        lines.append(f"- `{row.decision_item}`: `{row.decision}`；{row.reason}")
    lines.append("")
    lines.append("## Guardrail")
    lines.append("")
    for key, value in guardrail.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## 输出")
    lines.append("")
    lines.append("- `tables/v239_model_selection_validation_noharm.csv`")
    lines.append("- `tables/v239_metrics_by_delay_and_bucket.csv`")
    lines.append("- `tables/v239_compare_vs_v236_original_remaining.csv`")
    lines.append("- `tables/v239_attention_training_history.csv`")
    lines.append("- `tables/v239_next_model_decision.csv`")
    lines.append(f"- ZIP：`{zip_path.name}`")
    lines.append("")
    (REPORTS / "v239_light_attention_noharm_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    clean_out_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v239] device={device}")

    print("[v239] loading v238/v236 task data")
    data = V238.load_v236_data()
    x_base = V238.build_base_design_matrix(data)
    point_data = V238.build_point_dataset(data, x_base)
    point_masks = V238.split_point_masks(point_data, data.manifest)
    task_table, point_counts = V238.build_task_construction_tables(data, point_data)

    print("[v239] standardizing sequence inputs")
    scalers = fit_scalers(data, point_data, point_masks)
    arrays = standardize_arrays(data, point_data, scalers)

    configs: List[Tuple[str, Dict[str, object]]] = [
        (
            "v239_light_attention_h32",
            {
                "hidden_dim": 32,
                "head_dim": 64,
                "dropout": 0.05,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 2048,
                "max_epochs": 70,
                "patience": 9,
            },
        ),
        (
            "v239_light_attention_h48",
            {
                "hidden_dim": 48,
                "head_dim": 80,
                "dropout": 0.05,
                "lr": 8e-4,
                "weight_decay": 1e-4,
                "batch_size": 2048,
                "max_epochs": 70,
                "patience": 9,
            },
        ),
    ]

    runs: List[AttentionRun] = []
    for model_name, config in configs:
        print(f"[v239] training {model_name}")
        run = train_attention_candidate(model_name, config, data, point_data, arrays, scalers, point_masks, device)
        runs.append(run)
        print(f"[v239] {model_name} best_epoch={run.best_epoch} best_val_loss={run.best_val_loss:.6f}")

    print("[v239] computing metrics and no-harm decisions")
    y_true_curve = data.y_future[:, :, 0].astype(np.float32)
    pred_by_model: Dict[str, np.ndarray] = {
        "v236_joint_ridge_existing": data.pred_v236[:, :, 0].astype(np.float32),
        "v238_selected_original_remaining_point_model": load_v238_predictions(),
    }
    for run in runs:
        pred_by_model[run.model_name] = run.pred_curve.astype(np.float32)

    metrics = V238.compute_metrics_table(
        y_true_curve=y_true_curve,
        pred_by_model=pred_by_model,
        manifest=data.manifest,
        eval_modes=["original_remaining", "receding_2s_diagnostic"],
    )
    selection_rows: List[Dict[str, object]] = []
    for run in runs:
        row = candidate_validation_decision(metrics, run.model_name)
        row.update(
            {
                "config_json": json.dumps(run.config, ensure_ascii=False, sort_keys=True),
                "best_epoch": run.best_epoch,
                "best_val_loss": run.best_val_loss,
                "training_seconds": run.training_seconds,
            }
        )
        selection_rows.append(row)
    selection = pd.DataFrame(selection_rows).sort_values("validation_selection_score").reset_index(drop=True)
    selection["validation_rank"] = np.arange(1, len(selection) + 1)
    next_decision = build_next_decision(selection)
    compare = build_compare_table(metrics, pred_by_model.keys())
    split_check = split_integrity_check(data.manifest)
    guardrail = build_guardrail_json(selection, split_check)
    if not bool(guardrail["pass"]):
        raise AssertionError("v239 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))

    best_diagnostic_name = str(selection.iloc[0]["model_name"])
    best_run = next(run for run in runs if run.model_name == best_diagnostic_name)
    figure_paths = plot_figures(compare, [run.model_name for run in runs], best_diagnostic_name)

    print("[v239] writing outputs")
    write_csv(task_table, TABLES / "v239_task_construction_audit.csv")
    write_csv(point_counts, TABLES / "v239_point_training_rows_by_delay.csv")
    write_csv(selection, TABLES / "v239_model_selection_validation_noharm.csv")
    write_csv(metrics, TABLES / "v239_metrics_by_delay_and_bucket.csv")
    write_csv(compare, TABLES / "v239_compare_vs_v236_original_remaining.csv")
    write_csv(pd.concat([run.training_history for run in runs], ignore_index=True), TABLES / "v239_attention_training_history.csv")
    write_csv(next_decision, TABLES / "v239_next_model_decision.csv")
    write_csv(split_check, TABLES / "v239_split_integrity_check.csv")

    np.savez_compressed(
        OUT / "v239_light_attention_predictions.npz",
        y_true_steering_delta=y_true_curve.astype(np.float32),
        pred_v236_steering_delta=pred_by_model["v236_joint_ridge_existing"].astype(np.float32),
        pred_v238_steering_delta=pred_by_model["v238_selected_original_remaining_point_model"].astype(np.float32),
        pred_v239_best_attention_steering_delta=best_run.pred_curve.astype(np.float32),
        best_attention_model=np.array([best_diagnostic_name], dtype="U80"),
        delay_ms=data.manifest["delay_ms"].to_numpy(dtype=np.int32),
        split=data.manifest["split"].astype(str).to_numpy(dtype="U16"),
        event_uid=data.manifest["event_uid"].astype(str).to_numpy(dtype="U160"),
        future_grid_s=FUTURE_GRID.astype(np.float32),
        original_remaining_valid=V238.build_original_remaining_mask(data.manifest)[0].astype(np.bool_),
    )

    model_payload = {
        "model_name": best_diagnostic_name,
        "state_dict": best_run.state_dict,
        "config": best_run.config,
        "scalers": {
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
        },
        "selection": selection.to_dict(orient="records"),
    }
    torch.save(model_payload, MODELS / "v239_best_light_attention_diagnostic.pt")
    with (MODELS / "v239_scalers_and_selection.pkl").open("wb") as f:
        pickle.dump({"scalers": scalers, "selection": selection}, f)

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
        "stage": "v239_light_attention_noharm",
        "created_by": Path(__file__).name,
        "output_dir": str(OUT),
        "source_v238_dir": str(V238_DIR),
        "n_rolling_samples": int(len(data.manifest)),
        "n_events": int(data.manifest["event_uid"].nunique()),
        "device": str(device),
        "attention_candidates": [run.model_name for run in runs],
        "best_diagnostic_model": best_diagnostic_name,
        "any_attention_noharm_pass": bool(selection["validation_noharm_pass"].astype(bool).any()),
        "figures": [str(path.relative_to(OUT)).replace("\\", "/") for path in figure_paths],
    }
    (LOGS / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = zip_outputs()
    write_report(selection, next_decision, compare, guardrail, device, zip_path)
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = zip_outputs()

    print("[v239] finished")
    print(f"output_dir={OUT}")
    print(f"best_diagnostic_model={best_diagnostic_name}")
    print(f"any_attention_noharm_pass={bool(selection['validation_noharm_pass'].astype(bool).any())}")
    print(f"report={REPORTS / 'v239_light_attention_noharm_cn.md'}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
