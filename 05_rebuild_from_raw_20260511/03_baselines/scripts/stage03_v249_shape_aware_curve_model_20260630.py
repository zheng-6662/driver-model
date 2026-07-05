#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v249 shape-aware curve model。

本轮目标：
- 接受 v248 的结论：锚点已经不是主要矛盾，剩余瓶颈是强变化样本的轨迹形状；
- 继承 v241 的 TCN + multi-head query attention backbone；
- 继承 v238/v241 的 original_remaining masked target；
- 不做 anchor selector，不做 gate/router，不删除样本，不做 response-type hard routing；
- 只训练少数 shape-aware 候选，模型选择只看 validation，locked test 只做最终审查。

三个候选：
- v249a_shape_loss_only：从 v241 checkpoint 初始化，只加入 peak/slope/tail/curvature/excursion shape loss；
- v249b_shape_aux_heads：在 a 的基础上加入 shape auxiliary heads，但不 hard route；
- v249c_shape_conditioned_residual：在 b 的基础上用连续 shape context 产生一个小 residual head，仍然不是类型路由。

额外审查：
- input-neighborhood ambiguity audit：对 hard sample 找 train set 中输入最相似邻居，检查相似输入是否对应分歧很大的未来曲线。
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

V242_SCRIPT = BASELINES / "scripts" / "stage03_v242_joint_curve_decoder_20260626.py"
V241_DIR = BASELINES / "v241_stronger_temporal_model_20260626"
V241_PRED = V241_DIR / "v241_stronger_temporal_predictions.npz"
V241_MODEL = V241_DIR / "models" / "v241_best_stronger_temporal_diagnostic.pt"

OUT = BASELINES / "v249_shape_aware_curve_model_20260630"
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
SEED = 249
K_NEIGHBORS = 10

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    """按路径导入前序脚本，复用已经验证过的数据、模型和指标函数。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V242 = import_module_from_path("stage03_v242_joint_curve_decoder_20260626", V242_SCRIPT)
V241 = V242.V241
V239 = V241.V239
V238 = V241.V238
FUTURE_GRID = V238.FUTURE_GRID.astype(np.float32)
FUTURE_GRID_TORCH = torch.tensor(FUTURE_GRID, dtype=torch.float32)


@dataclass
class V249Run:
    """一个 v249 shape-aware 候选的训练结果。"""

    model_name: str
    config: Dict[str, object]
    state_dict: Dict[str, torch.Tensor]
    pred_curve: np.ndarray
    training_history: pd.DataFrame
    training_seconds: float
    best_epoch: int
    best_val_loss: float


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v249 自己的输出目录，避免触碰前序产物。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 统一使用 utf-8-sig，方便 Windows Excel 打开中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算文件 SHA256，用于输入追溯。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int = SEED) -> None:
    """固定随机种子。"""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def load_v241_checkpoint() -> Dict[str, object]:
    """读取本地可信的 v241 checkpoint。"""

    if not V241_MODEL.exists():
        raise FileNotFoundError(f"缺少 v241 checkpoint：{V241_MODEL}")
    return torch.load(V241_MODEL, map_location="cpu", weights_only=False)


def load_v241_prediction() -> Tuple[np.ndarray, str]:
    """读取 v241 locked prediction。"""

    if not V241_PRED.exists():
        raise FileNotFoundError(f"缺少 v241 prediction：{V241_PRED}")
    with np.load(V241_PRED, allow_pickle=False) as pred:
        arr = pred["pred_v241_best_stronger_steering_delta"].astype(np.float32)
        name = str(pred["best_stronger_model"][0])
    return arr, name


def sample_masks(manifest: pd.DataFrame) -> Dict[str, np.ndarray]:
    """按 split 生成 sample-level mask。"""

    split = manifest["split"].astype(str).to_numpy()
    return {name: split == name for name in ["train", "val", "test"]}


class ShapeCurveDataset(Dataset):
    """sample-level 曲线数据集，每个样本返回 21 点 target 和 v241 teacher。"""

    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        point_data,
        manifest: pd.DataFrame,
        sample_mask: np.ndarray,
        pred_v241_original: np.ndarray,
        scalers,
    ) -> None:
        n_samples = int(arrays["hist"].shape[0])
        n_steps = len(FUTURE_GRID)
        self.hist = arrays["hist"].astype(np.float32)
        self.road = arrays["road"].astype(np.float32)
        self.phase = arrays["phase"].astype(np.float32)
        self.point_seq = arrays["point"].reshape(n_samples, n_steps, -1).astype(np.float32)
        self.y_seq = arrays["y"].reshape(n_samples, n_steps).astype(np.float32)
        self.valid_seq = point_data.valid_original_remaining_all.reshape(n_samples, n_steps).astype(np.float32)
        self.weight_seq = point_data.point_weight_all.reshape(n_samples, n_steps).astype(np.float32)
        self.teacher_seq = ((pred_v241_original - scalers.y_mean) / scalers.y_std).astype(np.float32)
        self.normal_flag = (
            manifest["normal_curve"].astype(bool).to_numpy()
            & ~manifest["strong_steer"].astype(bool).to_numpy()
            & ~manifest["observe_later_like"].astype(bool).to_numpy()
        ).astype(np.float32)
        self.indices = np.where(sample_mask)[0].astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = int(self.indices[idx])
        return {
            "hist": torch.from_numpy(self.hist[sample_idx]),
            "road": torch.from_numpy(self.road[sample_idx]),
            "phase": torch.from_numpy(self.phase[sample_idx]),
            "point_seq": torch.from_numpy(self.point_seq[sample_idx]),
            "y_seq": torch.from_numpy(self.y_seq[sample_idx]),
            "valid_seq": torch.from_numpy(self.valid_seq[sample_idx]),
            "weight_seq": torch.from_numpy(self.weight_seq[sample_idx]),
            "teacher_seq": torch.from_numpy(self.teacher_seq[sample_idx]),
            "normal_flag": torch.tensor(self.normal_flag[sample_idx], dtype=torch.float32),
        }


class ShapeAwareV241CurveModel(nn.Module):
    """
    v241 backbone 的 curve-level 包装。

    模块名保持和 v241 一致，便于直接加载 v241 checkpoint；新增 aux/residual 模块用于 v249b/c。
    """

    def __init__(
        self,
        hist_dim: int,
        road_dim: int,
        phase_dim: int,
        point_dim: int,
        hist_len: int,
        road_len: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        mlp_hidden: int,
        dropout: float,
        use_aux_heads: bool,
        shape_conditioned: bool,
        residual_scale: float,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} 必须能被 n_heads={n_heads} 整除")
        self.hidden_dim = hidden_dim
        self.phase_dim = phase_dim
        self.point_dim = point_dim
        self.use_aux_heads = bool(use_aux_heads)
        self.shape_conditioned = bool(shape_conditioned)
        self.residual_scale = float(residual_scale)

        self.hist_encoder = V241.TemporalConvEncoder(hist_dim, hidden_dim, hist_len, n_layers, dropout)
        self.road_encoder = V241.TemporalConvEncoder(road_dim, hidden_dim, road_len, max(1, n_layers - 1), dropout)
        self.query = nn.Sequential(
            nn.Linear(phase_dim + point_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.hist_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.road_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + phase_dim + point_dim, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, max(32, mlp_hidden // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, mlp_hidden // 2), 1),
        )

        shape_context_dim = hidden_dim * 4 + phase_dim
        self.shape_context = nn.Sequential(
            nn.Linear(shape_context_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.aux_head = nn.Linear(hidden_dim, 6)
        residual_in = hidden_dim * 3 + phase_dim + point_dim + hidden_dim
        self.conditioned_residual = nn.Sequential(
            nn.Linear(residual_in, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        hist: torch.Tensor,
        road: torch.Tensor,
        phase: torch.Tensor,
        point_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch, steps, _ = point_seq.shape
        hist_tokens = self.hist_encoder(hist)
        road_tokens = self.road_encoder(road)

        phase_rep = phase.unsqueeze(1).expand(batch, steps, phase.shape[-1])
        query_in = torch.cat([phase_rep, point_seq], dim=-1).reshape(batch * steps, -1)
        query = self.query(query_in).reshape(batch, steps, self.hidden_dim)

        q_flat = query.reshape(batch * steps, 1, self.hidden_dim)
        hist_rep = hist_tokens.unsqueeze(1).expand(batch, steps, hist_tokens.shape[1], self.hidden_dim)
        road_rep = road_tokens.unsqueeze(1).expand(batch, steps, road_tokens.shape[1], self.hidden_dim)
        hist_rep = hist_rep.reshape(batch * steps, hist_tokens.shape[1], self.hidden_dim)
        road_rep = road_rep.reshape(batch * steps, road_tokens.shape[1], self.hidden_dim)
        hist_ctx, _ = self.hist_attn(q_flat, hist_rep, hist_rep, need_weights=False)
        road_ctx, _ = self.road_attn(q_flat, road_rep, road_rep, need_weights=False)
        hist_ctx = hist_ctx.reshape(batch, steps, self.hidden_dim)
        road_ctx = road_ctx.reshape(batch, steps, self.hidden_dim)

        head_input = torch.cat([hist_ctx, road_ctx, query, phase_rep, point_seq], dim=-1)
        base_pred = self.head(head_input.reshape(batch * steps, -1)).reshape(batch, steps)

        hist_last = hist_tokens[:, -1, :]
        hist_mean = hist_tokens.mean(dim=1)
        road_mean = road_tokens.mean(dim=1)
        query_mean = query.mean(dim=1)
        shape_ctx = self.shape_context(torch.cat([hist_last, hist_mean, road_mean, query_mean, phase], dim=-1))
        aux = self.aux_head(shape_ctx)

        if self.shape_conditioned:
            shape_rep = shape_ctx.unsqueeze(1).expand(batch, steps, self.hidden_dim)
            residual_input = torch.cat([head_input, shape_rep], dim=-1)
            residual = self.conditioned_residual(residual_input.reshape(batch * steps, -1)).reshape(batch, steps)
            pred = base_pred + self.residual_scale * residual
        else:
            pred = base_pred

        return pred, {"shape_aux": aux, "base_pred": base_pred}


def valid_weighted_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """按 mask 求均值，mask 为空时回退到 0。"""

    denom = torch.clamp(mask.sum(dim=1), min=1.0)
    return (value * mask).sum(dim=1) / denom


def curve_shape_features(curve: torch.Tensor, valid: torch.Tensor, time_s: torch.Tensor) -> torch.Tensor:
    """
    从曲线提取连续 shape 特征。

    返回 6 维：
    peak_abs、tail_mean、max_abs_slope、slope_energy、excursion、final_value。
    """

    device = curve.device
    time_s = time_s.to(device=device, dtype=curve.dtype)
    valid = valid.to(dtype=curve.dtype)
    neg_big = torch.full_like(curve, -1.0e6)
    abs_curve = torch.abs(curve)
    peak_abs = torch.max(torch.where(valid > 0.5, abs_curve, neg_big), dim=1).values
    peak_abs = torch.clamp(peak_abs, min=0.0)

    tail_mask = valid * (time_s.view(1, -1) >= 1.0).to(dtype=curve.dtype)
    tail_denom = tail_mask.sum(dim=1)
    all_mean = valid_weighted_mean(curve, valid)
    tail_mean = torch.where(tail_denom > 0, valid_weighted_mean(curve, tail_mask), all_mean)

    diff_valid = valid[:, 1:] * valid[:, :-1]
    dt = torch.clamp(time_s[1:] - time_s[:-1], min=1.0e-6).view(1, -1)
    diff = curve[:, 1:] - curve[:, :-1]
    slope = diff / dt
    abs_slope = torch.abs(slope)
    max_abs_slope = torch.max(torch.where(diff_valid > 0.5, abs_slope, torch.full_like(abs_slope, -1.0e6)), dim=1).values
    max_abs_slope = torch.clamp(max_abs_slope, min=0.0)
    slope_energy = valid_weighted_mean(abs_slope, diff_valid)
    excursion = (torch.abs(diff) * diff_valid).sum(dim=1)

    valid_count = torch.clamp(valid.sum(dim=1).long(), min=1)
    last_index = torch.clamp(valid_count - 1, min=0, max=curve.shape[1] - 1)
    final_value = curve.gather(1, last_index.view(-1, 1)).squeeze(1)
    return torch.stack([peak_abs, tail_mean, max_abs_slope, slope_energy, excursion, final_value], dim=1)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """masked MSE。"""

    return torch.sum(torch.square(pred - target) * mask) / torch.clamp(torch.sum(mask), min=1.0e-6)


def shape_aware_loss(
    pred_scaled: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    target_scaled: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    teacher_scaled: torch.Tensor,
    normal_flag: torch.Tensor,
    config: Dict[str, object],
    y_mean: float,
    y_std: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """计算 v249 shape-aware loss，并返回日志项。"""

    point_weight = valid * weight
    point_loss = masked_mse(pred_scaled, target_scaled, point_weight)
    pred = pred_scaled * float(y_std) + float(y_mean)
    target = target_scaled * float(y_std) + float(y_mean)
    teacher = teacher_scaled * float(y_std) + float(y_mean)
    time_s = FUTURE_GRID_TORCH.to(device=pred.device, dtype=pred.dtype)

    pred_feat = curve_shape_features(pred, valid, time_s)
    true_feat = curve_shape_features(target, valid, time_s)
    peak_loss = torch.mean(torch.square(pred_feat[:, 0] - true_feat[:, 0]))
    tail_loss = torch.mean(torch.square(pred_feat[:, 1] - true_feat[:, 1]))
    slope_loss = torch.mean(torch.square(pred_feat[:, 2] - true_feat[:, 2])) + 0.25 * torch.mean(
        torch.square(pred_feat[:, 3] - true_feat[:, 3])
    )
    excursion_loss = torch.mean(torch.square(pred_feat[:, 4] - true_feat[:, 4]))

    diff_valid = valid[:, 1:] * valid[:, :-1]
    d2_valid = diff_valid[:, 1:] * diff_valid[:, :-1]
    pred_d2 = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    true_d2 = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    curvature_loss = masked_mse(pred_d2, true_d2, d2_valid)

    aux_loss = torch.tensor(0.0, device=pred.device)
    if bool(config.get("use_aux_heads", False)):
        # auxiliary heads 预测原始尺度 shape 特征；只做软监督，不参与 hard route。
        aux_loss = torch.mean(torch.square(aux["shape_aux"] - true_feat.detach()))

    teacher_guard = torch.tensor(0.0, device=pred.device)
    guard_weight = float(config.get("teacher_guard_weight", 0.0))
    if guard_weight > 0:
        normal_mask = normal_flag.view(-1, 1) * valid
        teacher_guard = masked_mse(pred, teacher, normal_mask)

    loss = (
        point_loss
        + float(config.get("peak_weight", 0.0)) * peak_loss
        + float(config.get("tail_weight", 0.0)) * tail_loss
        + float(config.get("slope_weight", 0.0)) * slope_loss
        + float(config.get("curvature_weight", 0.0)) * curvature_loss
        + float(config.get("excursion_weight", 0.0)) * excursion_loss
        + float(config.get("aux_weight", 0.0)) * aux_loss
        + guard_weight * teacher_guard
    )
    logs = {
        "point_loss": float(point_loss.detach().cpu().item()),
        "peak_loss": float(peak_loss.detach().cpu().item()),
        "tail_loss": float(tail_loss.detach().cpu().item()),
        "slope_loss": float(slope_loss.detach().cpu().item()),
        "curvature_loss": float(curvature_loss.detach().cpu().item()),
        "excursion_loss": float(excursion_loss.detach().cpu().item()),
        "aux_loss": float(aux_loss.detach().cpu().item()),
        "teacher_guard": float(teacher_guard.detach().cpu().item()),
        "total_loss": float(loss.detach().cpu().item()),
    }
    return loss, logs


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    config: Dict[str, object],
    y_mean: float,
    y_std: float,
) -> Dict[str, float]:
    """运行一个训练或验证 epoch。"""

    is_train = optimizer is not None
    model.train(is_train)
    totals: Dict[str, float] = {}
    total_weight = 0.0
    for batch in loader:
        hist = batch["hist"].to(device=device, dtype=torch.float32)
        road = batch["road"].to(device=device, dtype=torch.float32)
        phase = batch["phase"].to(device=device, dtype=torch.float32)
        point_seq = batch["point_seq"].to(device=device, dtype=torch.float32)
        y_seq = batch["y_seq"].to(device=device, dtype=torch.float32)
        valid_seq = batch["valid_seq"].to(device=device, dtype=torch.float32)
        weight_seq = batch["weight_seq"].to(device=device, dtype=torch.float32)
        teacher_seq = batch["teacher_seq"].to(device=device, dtype=torch.float32)
        normal_flag = batch["normal_flag"].to(device=device, dtype=torch.float32)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
        pred, aux = model(hist, road, phase, point_seq)
        loss, logs = shape_aware_loss(
            pred,
            aux,
            y_seq,
            valid_seq,
            weight_seq,
            teacher_seq,
            normal_flag,
            config,
            y_mean,
            y_std,
        )
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
        wsum = float(torch.sum(valid_seq * weight_seq).detach().cpu().item())
        total_weight += wsum
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + value * wsum
    return {key: value / max(total_weight, 1.0e-6) for key, value in totals.items()}


def predict_curves(
    model: nn.Module,
    arrays: Dict[str, np.ndarray],
    scalers,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """对所有 rolling samples 输出 21 点曲线，并还原到原始 steering_delta 空间。"""

    model.eval()
    n_samples = int(arrays["hist"].shape[0])
    n_steps = len(FUTURE_GRID)
    point_seq = arrays["point"].reshape(n_samples, n_steps, -1).astype(np.float32)
    pred_scaled = np.empty((n_samples, n_steps), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            hist = torch.from_numpy(arrays["hist"][start:end]).to(device=device, dtype=torch.float32)
            road = torch.from_numpy(arrays["road"][start:end]).to(device=device, dtype=torch.float32)
            phase = torch.from_numpy(arrays["phase"][start:end]).to(device=device, dtype=torch.float32)
            points = torch.from_numpy(point_seq[start:end]).to(device=device, dtype=torch.float32)
            pred, _ = model(hist, road, phase, points)
            pred_scaled[start:end] = pred.detach().cpu().numpy().astype(np.float32)
    return (pred_scaled * scalers.y_std + scalers.y_mean).astype(np.float32)


def build_model_from_v241(data, config: Dict[str, object], checkpoint: Dict[str, object], device: torch.device) -> nn.Module:
    """按 v241 配置建模，并加载 v241 checkpoint 中可复用权重。"""

    base_config = dict(checkpoint["config"])
    model = ShapeAwareV241CurveModel(
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
        dropout=float(config.get("dropout", base_config["dropout"])),
        use_aux_heads=bool(config.get("use_aux_heads", False)),
        shape_conditioned=bool(config.get("shape_conditioned", False)),
        residual_scale=float(config.get("residual_scale", 0.0)),
    )
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    allowed_missing_prefixes = ("shape_context.", "aux_head.", "conditioned_residual.")
    bad_missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]
    if bad_missing or unexpected:
        raise AssertionError(
            f"v241 checkpoint 加载异常，bad_missing={bad_missing[:8]}, unexpected={unexpected[:8]}"
        )
    return model.to(device)


def train_v249_candidate(
    model_name: str,
    config: Dict[str, object],
    data,
    point_data,
    arrays: Dict[str, np.ndarray],
    scalers,
    masks: Dict[str, np.ndarray],
    pred_v241: np.ndarray,
    checkpoint: Dict[str, object],
    device: torch.device,
) -> V249Run:
    """训练一个 v249 候选。"""

    train_dataset = ShapeCurveDataset(arrays, point_data, data.manifest, masks["train"], pred_v241, scalers)
    val_dataset = ShapeCurveDataset(arrays, point_data, data.manifest, masks["val"], pred_v241, scalers)
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
    model = build_model_from_v241(data, config, checkpoint, device)
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

    max_epochs = int(config["max_epochs"])
    patience = int(config["patience"])
    best_val = math.inf
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: List[Dict[str, object]] = []
    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        train_logs = run_epoch(model, train_loader, device, optimizer, config, scalers.y_mean, scalers.y_std)
        val_logs = run_epoch(model, val_loader, device, None, config, scalers.y_mean, scalers.y_std)
        val_loss = float(val_logs["total_loss"])
        scheduler.step(val_loss)
        lr_now = float(optimizer.param_groups[0]["lr"])
        row: Dict[str, object] = {
            "model_name": model_name,
            "epoch": epoch,
            "lr": lr_now,
        }
        row.update({f"train_{k}": v for k, v in train_logs.items()})
        row.update({f"val_{k}": v for k, v in val_logs.items()})
        history.append(row)
        if val_loss < best_val - 1.0e-5:
            best_val = val_loss
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
    pred_curve = predict_curves(model, arrays, scalers, device, batch_size=batch_size * 4)
    return V249Run(
        model_name=model_name,
        config=config,
        state_dict=best_state,
        pred_curve=pred_curve.astype(np.float32),
        training_history=pd.DataFrame(history),
        training_seconds=float(time.time() - start_time),
        best_epoch=int(best_epoch),
        best_val_loss=float(best_val),
    )


def finite_mean(values: pd.Series, default: float = math.inf) -> float:
    """安全均值。"""

    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(arr.mean())


def finite_max(values: pd.Series, default: float = math.inf) -> float:
    """安全最大值。"""

    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(arr.max())


def positive_penalty(value: float, threshold: float) -> float:
    """超过阈值才惩罚。"""

    if not np.isfinite(value):
        return 10.0
    return max(0.0, float(value) - float(threshold))


def delta_frame(metrics: pd.DataFrame, candidate_name: str, ref_name: str) -> pd.DataFrame:
    """validation 原始剩余窗口下，候选相对参考的 bucket/delay delta。"""

    val = metrics[metrics["split"].eq("val") & metrics["eval_mode"].eq("original_remaining")].copy()
    cand = val[val["model_name"].eq(candidate_name)].copy()
    ref = val[val["model_name"].eq(ref_name)].copy()
    if cand.empty or ref.empty:
        raise AssertionError(f"{candidate_name} 或 {ref_name} validation 指标为空")
    merged = cand.merge(
        ref,
        on=["split", "bucket", "delay_ms", "eval_mode"],
        suffixes=("_candidate", "_ref"),
    )
    merged["delta_sample"] = merged["steer_sample_rmse_mean_candidate"] - merged["steer_sample_rmse_mean_ref"]
    merged["delta_tail"] = merged["steer_tail_rmse_mean_candidate"] - merged["steer_tail_rmse_mean_ref"]
    return merged


def subset_delta(
    merged: pd.DataFrame,
    bucket: str,
    max_delay: int | None = None,
    delays: Iterable[int] | None = None,
) -> pd.DataFrame:
    """抽取某个 bucket/delay 子集。"""

    out = merged[merged["bucket"].eq(bucket)].copy()
    if max_delay is not None:
        out = out[out["delay_ms"].astype(int) <= int(max_delay)].copy()
    if delays is not None:
        wanted = {int(x) for x in delays}
        out = out[out["delay_ms"].astype(int).isin(wanted)].copy()
    return out


def candidate_validation_decision(
    metrics: pd.DataFrame,
    candidate_name: str,
    v241_name: str,
    shape_validation: pd.DataFrame,
) -> Dict[str, object]:
    """只用 validation 选择 v249 候选。"""

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

    shape_val = shape_validation[
        shape_validation["split"].eq("val") & shape_validation["model_name"].eq(candidate_name)
    ].copy()
    strong_shape = shape_val[shape_val["event_group"].eq("strong_steer")]
    bad_shape = shape_val[shape_val["event_group"].eq("bad_top10_v241")]
    normal_shape = shape_val[shape_val["event_group"].eq("normal")]
    strong_range_gain = finite_mean(strong_shape["delta_range_ratio_candidate_minus_v241"], default=0.0)
    strong_slope_gain = finite_mean(strong_shape["delta_slope_ratio_candidate_minus_v241"], default=0.0)
    bad_range_gain = finite_mean(bad_shape["delta_range_ratio_candidate_minus_v241"], default=0.0)
    bad_slope_gain = finite_mean(bad_shape["delta_slope_ratio_candidate_minus_v241"], default=0.0)
    normal_shape_rmse_delta = finite_mean(normal_shape["delta_rmse_candidate_minus_v241"], default=0.0)

    noharm_vs_v236 = (
        normal_max_sample_delta_v236 <= NOHARM_TOL
        and normal_max_tail_delta_v236 <= NOHARM_TOL
        and all_max_sample_delta_v236 <= NOHARM_TOL
        and observe_mean_tail_delta_v236 <= NOHARM_TOL
        and strong_mean_tail_delta_v236 <= NOHARM_TOL
        and strong_exception_mean_tail_delta_v236 <= NOHARM_TOL
    )
    upgrade_vs_v241 = (
        normal_max_tail_delta_v241 <= UPGRADE_TOL
        and all_mean_tail_delta_v241 <= UPGRADE_TOL
        and observe_mean_tail_delta_v241 <= UPGRADE_TOL
        and strong_exception_mean_tail_delta_v241 <= UPGRADE_TOL
    )
    shape_gain_pass = (strong_range_gain + strong_slope_gain + bad_range_gain + bad_slope_gain) > 0.05

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
        + positive_penalty(observe_mean_tail_delta_v236, NOHARM_TOL)
        + positive_penalty(strong_mean_tail_delta_v236, NOHARM_TOL)
        + positive_penalty(strong_exception_mean_tail_delta_v236, NOHARM_TOL)
    )
    penalty_vs_v241 = (
        positive_penalty(normal_max_tail_delta_v241, UPGRADE_TOL)
        + positive_penalty(all_mean_tail_delta_v241, UPGRADE_TOL)
        + positive_penalty(observe_mean_tail_delta_v241, UPGRADE_TOL)
        + positive_penalty(strong_exception_mean_tail_delta_v241, UPGRADE_TOL)
    )
    shape_reward = max(0.0, strong_range_gain) + max(0.0, strong_slope_gain) + max(0.0, bad_range_gain) + max(
        0.0, bad_slope_gain
    )
    shape_penalty = positive_penalty(normal_shape_rmse_delta, UPGRADE_TOL)
    selection_score = base_score + 8.0 * penalty_vs_v236 + 5.0 * penalty_vs_v241 + 4.0 * shape_penalty - 0.20 * shape_reward

    return {
        "model_name": candidate_name,
        "selected_by": "validation_noharm_v241_upgrade_and_shape_gain_only",
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
        "val_strong_range_ratio_gain_vs_v241": strong_range_gain,
        "val_strong_slope_ratio_gain_vs_v241": strong_slope_gain,
        "val_bad_top10_range_ratio_gain_vs_v241": bad_range_gain,
        "val_bad_top10_slope_ratio_gain_vs_v241": bad_slope_gain,
        "val_normal_shape_rmse_delta_vs_v241": normal_shape_rmse_delta,
        "noharm_vs_v236_pass": bool(noharm_vs_v236),
        "upgrade_vs_v241_pass": bool(upgrade_vs_v241),
        "shape_gain_pass": bool(shape_gain_pass),
        "accepted_as_shape_candidate": bool(noharm_vs_v236 and upgrade_vs_v241 and shape_gain_pass),
        "validation_selection_score": float(selection_score),
    }


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE。"""

    if a.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(a - b))))


def turning_count(curve: np.ndarray, eps: float = 0.03) -> int:
    """用一阶差分符号变化粗略计算转折次数。"""

    if curve.size < 3:
        return 0
    diff = np.diff(curve)
    diff[np.abs(diff) < eps] = 0.0
    signs = np.sign(diff)
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def shape_metrics_np(true_y: np.ndarray, pred_y: np.ndarray) -> Dict[str, float]:
    """numpy 版 shape 指标，用于报告和 casebook。"""

    y = np.asarray(true_y, dtype=float)
    p = np.asarray(pred_y, dtype=float)
    if y.size == 0 or p.size == 0:
        return {
            "rmse": math.nan,
            "range_ratio": math.nan,
            "excursion_ratio": math.nan,
            "slope_ratio": math.nan,
            "turning_gap": math.nan,
        }
    true_range = float(np.nanmax(y) - np.nanmin(y))
    pred_range = float(np.nanmax(p) - np.nanmin(p))
    true_exc = float(np.nansum(np.abs(np.diff(y))))
    pred_exc = float(np.nansum(np.abs(np.diff(p))))
    true_slope = float(np.nanmax(np.abs(np.diff(y)))) if y.size >= 2 else 0.0
    pred_slope = float(np.nanmax(np.abs(np.diff(p)))) if p.size >= 2 else 0.0

    def div(a: float, b: float) -> float:
        return float(a / b) if abs(b) > 1.0e-8 else math.nan

    return {
        "rmse": rmse(y, p),
        "range_ratio": div(pred_range, true_range),
        "excursion_ratio": div(pred_exc, true_exc),
        "slope_ratio": div(pred_slope, true_slope),
        "turning_gap": float(turning_count(p) - turning_count(y)),
    }


def build_shape_validation_table(
    y_true: np.ndarray,
    pred_v241: np.ndarray,
    pred_by_model: Dict[str, np.ndarray],
    manifest: pd.DataFrame,
    valid_mask: np.ndarray,
    bad_top10_thresholds: Dict[str, float],
) -> pd.DataFrame:
    """按 split/event_group 汇总 shape 指标，供 validation 选择和 test 报告。"""

    rows: List[Dict[str, object]] = []
    groups = {
        "all": np.ones(len(manifest), dtype=bool),
        "normal": (
            manifest["normal_curve"].astype(bool).to_numpy()
            & ~manifest["strong_steer"].astype(bool).to_numpy()
            & ~manifest["observe_later_like"].astype(bool).to_numpy()
        ),
        "strong_steer": manifest["strong_steer"].astype(bool).to_numpy(),
        "observe_later_like": manifest["observe_later_like"].astype(bool).to_numpy(),
    }
    split_values = manifest["split"].astype(str).to_numpy()
    delay_values = manifest["delay_ms"].astype(int).to_numpy()

    # bad_top10_v241 在每个 split 内按 v241 原始剩余窗口 RMSE 定义，避免 test 影响 validation。
    v241_rmse = np.full(len(manifest), np.nan, dtype=float)
    for i in range(len(manifest)):
        valid = valid_mask[i]
        if np.any(valid):
            v241_rmse[i] = rmse(y_true[i, valid], pred_v241[i, valid])
    bad_mask = np.zeros(len(manifest), dtype=bool)
    for split in ["train", "val", "test"]:
        split_mask = split_values == split
        base = v241_rmse[split_mask & np.isfinite(v241_rmse)]
        if base.size == 0:
            continue
        threshold = float(np.quantile(base, 0.90))
        bad_top10_thresholds[split] = threshold
        bad_mask |= split_mask & (v241_rmse >= threshold)
    groups["bad_top10_v241"] = bad_mask

    for split in ["train", "val", "test"]:
        split_mask = split_values == split
        for group_name, group_mask in groups.items():
            base_mask = split_mask & group_mask
            for delay in DELAY_MS:
                mask = base_mask & (delay_values == delay)
                if not np.any(mask):
                    continue
                for model_name, pred in pred_by_model.items():
                    rmse_values: List[float] = []
                    range_values: List[float] = []
                    excursion_values: List[float] = []
                    slope_values: List[float] = []
                    turning_values: List[float] = []
                    rmse_delta_values: List[float] = []
                    range_delta_values: List[float] = []
                    slope_delta_values: List[float] = []
                    for idx in np.where(mask)[0]:
                        valid = valid_mask[idx]
                        if not np.any(valid):
                            continue
                        m = shape_metrics_np(y_true[idx, valid], pred[idx, valid])
                        b = shape_metrics_np(y_true[idx, valid], pred_v241[idx, valid])
                        rmse_values.append(m["rmse"])
                        range_values.append(m["range_ratio"])
                        excursion_values.append(m["excursion_ratio"])
                        slope_values.append(m["slope_ratio"])
                        turning_values.append(m["turning_gap"])
                        rmse_delta_values.append(m["rmse"] - b["rmse"])
                        range_delta_values.append(m["range_ratio"] - b["range_ratio"])
                        slope_delta_values.append(m["slope_ratio"] - b["slope_ratio"])
                    if not rmse_values:
                        continue
                    rows.append(
                        {
                            "split": split,
                            "event_group": group_name,
                            "delay_ms": delay,
                            "model_name": model_name,
                            "n": len(rmse_values),
                            "mean_rmse": float(np.nanmean(rmse_values)),
                            "mean_range_ratio": float(np.nanmean(range_values)),
                            "mean_excursion_ratio": float(np.nanmean(excursion_values)),
                            "mean_slope_ratio": float(np.nanmean(slope_values)),
                            "mean_turning_gap": float(np.nanmean(turning_values)),
                            "delta_rmse_candidate_minus_v241": float(np.nanmean(rmse_delta_values)),
                            "delta_range_ratio_candidate_minus_v241": float(np.nanmean(range_delta_values)),
                            "delta_slope_ratio_candidate_minus_v241": float(np.nanmean(slope_delta_values)),
                        }
                    )
    return pd.DataFrame(rows)


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
        if model in {"v236_joint_ridge_existing", ref_name}:
            continue
        for metric in ["steer_sample_rmse_mean", "steer_tail_rmse_mean"]:
            ref_col = f"{metric}__{ref_name}"
            model_col = f"{metric}__{model}"
            if ref_col in pivot.columns and model_col in pivot.columns:
                pivot[f"delta_{metric}__{model}_minus_v241"] = pivot[model_col] - pivot[ref_col]
    return pivot


def per_sample_shape_delta(
    y_true: np.ndarray,
    pred_v241: np.ndarray,
    pred_candidate: np.ndarray,
    manifest: pd.DataFrame,
    valid_mask: np.ndarray,
    candidate_name: str,
) -> pd.DataFrame:
    """逐样本 shape delta 表。"""

    rows: List[Dict[str, object]] = []
    for idx, row in manifest.iterrows():
        valid = valid_mask[idx]
        if not np.any(valid):
            continue
        base = shape_metrics_np(y_true[idx, valid], pred_v241[idx, valid])
        cand = shape_metrics_np(y_true[idx, valid], pred_candidate[idx, valid])
        rows.append(
            {
                "rolling_sample_index": int(row["rolling_sample_index"]),
                "event_uid": str(row["event_uid"]),
                "split": str(row["split"]),
                "delay_ms": int(row["delay_ms"]),
                "subject": str(row["subject"]),
                "observe_later_like": bool(row["observe_later_like"]),
                "strong_steer": bool(row["strong_steer"]),
                "reverse": bool(row["reverse"]),
                "normal_curve": bool(row["normal_curve"]),
                "model_name": candidate_name,
                "rmse_v241": base["rmse"],
                "rmse_candidate": cand["rmse"],
                "delta_rmse_candidate_minus_v241": cand["rmse"] - base["rmse"],
                "range_ratio_v241": base["range_ratio"],
                "range_ratio_candidate": cand["range_ratio"],
                "delta_range_ratio_candidate_minus_v241": cand["range_ratio"] - base["range_ratio"],
                "slope_ratio_v241": base["slope_ratio"],
                "slope_ratio_candidate": cand["slope_ratio"],
                "delta_slope_ratio_candidate_minus_v241": cand["slope_ratio"] - base["slope_ratio"],
                "turning_gap_v241": base["turning_gap"],
                "turning_gap_candidate": cand["turning_gap"],
            }
        )
    return pd.DataFrame(rows)


def build_neighbor_ambiguity_audit(
    y_true: np.ndarray,
    pred_v241: np.ndarray,
    pred_candidate: np.ndarray,
    arrays: Dict[str, np.ndarray],
    manifest: pd.DataFrame,
    valid_mask: np.ndarray,
    k: int = K_NEIGHBORS,
) -> pd.DataFrame:
    """
    输入近邻不可判别审查。

    对 test delay=0 的 v241 bad_top10 样本，找同 delay train 近邻，检查近邻真实未来曲线是否彼此分歧。
    """

    split = manifest["split"].astype(str).to_numpy()
    delay = manifest["delay_ms"].astype(int).to_numpy()
    feature = np.concatenate(
        [
            arrays["hist"].reshape(arrays["hist"].shape[0], -1),
            arrays["road"].reshape(arrays["road"].shape[0], -1),
            arrays["phase"].reshape(arrays["phase"].shape[0], -1),
        ],
        axis=1,
    ).astype(np.float32)
    # 限制极端维度对距离的影响。
    feature = np.clip(feature, -5.0, 5.0)

    v241_rmse = np.full(len(manifest), np.nan, dtype=float)
    cand_rmse = np.full(len(manifest), np.nan, dtype=float)
    for i in range(len(manifest)):
        valid = valid_mask[i]
        if np.any(valid):
            v241_rmse[i] = rmse(y_true[i, valid], pred_v241[i, valid])
            cand_rmse[i] = rmse(y_true[i, valid], pred_candidate[i, valid])
    test_delay0 = (split == "test") & (delay == 0) & np.isfinite(v241_rmse)
    if not np.any(test_delay0):
        return pd.DataFrame()
    threshold = float(np.quantile(v241_rmse[test_delay0], 0.90))
    query_indices = np.where(test_delay0 & (v241_rmse >= threshold))[0]

    rows: List[Dict[str, object]] = []
    for qi in query_indices:
        train_mask = (split == "train") & (delay == int(delay[qi]))
        train_indices = np.where(train_mask)[0]
        if train_indices.size == 0:
            continue
        dist = np.sqrt(np.mean(np.square(feature[train_indices] - feature[qi]), axis=1))
        order = np.argsort(dist)[: min(k, train_indices.size)]
        nn_idx = train_indices[order]
        nn_dist = dist[order]
        valid = valid_mask[qi]
        nn_curves = [y_true[j, valid] for j in nn_idx]
        pairwise: List[float] = []
        for a in range(len(nn_curves)):
            for b in range(a + 1, len(nn_curves)):
                pairwise.append(rmse(nn_curves[a], nn_curves[b]))
        query_vs_nn = [rmse(y_true[qi, valid], c) for c in nn_curves]
        peaks = np.array([float(np.max(np.abs(c))) if c.size else math.nan for c in nn_curves], dtype=float)
        slopes = np.array([float(np.max(np.abs(np.diff(c)))) if c.size >= 2 else math.nan for c in nn_curves], dtype=float)
        pairwise_mean = float(np.nanmean(pairwise)) if pairwise else math.nan
        peak_std = float(np.nanstd(peaks)) if peaks.size else math.nan
        slope_std = float(np.nanstd(slopes)) if slopes.size else math.nan
        category = "input_ambiguous" if (pairwise_mean >= 0.50 and peak_std >= 0.25) else "neighbor_consistent"
        rows.append(
            {
                "rolling_sample_index": int(manifest.iloc[qi]["rolling_sample_index"]),
                "event_uid": str(manifest.iloc[qi]["event_uid"]),
                "subject": str(manifest.iloc[qi]["subject"]),
                "delay_ms": int(delay[qi]),
                "v241_rmse": float(v241_rmse[qi]),
                "candidate_rmse": float(cand_rmse[qi]),
                "delta_candidate_minus_v241": float(cand_rmse[qi] - v241_rmse[qi]),
                "neighbor_k": int(len(nn_idx)),
                "neighbor_input_distance_mean": float(np.mean(nn_dist)),
                "neighbor_input_distance_min": float(np.min(nn_dist)),
                "neighbor_future_pairwise_rmse_mean": pairwise_mean,
                "neighbor_peak_abs_std": peak_std,
                "neighbor_slope_abs_std": slope_std,
                "query_vs_neighbor_best_rmse": float(np.nanmin(query_vs_nn)),
                "query_vs_neighbor_mean_rmse": float(np.nanmean(query_vs_nn)),
                "ambiguity_category": category,
                "neighbor_event_uids": "|".join(manifest.iloc[nn_idx]["event_uid"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values("v241_rmse", ascending=False).reset_index(drop=True)


def plot_shape_casebook(
    per_sample: pd.DataFrame,
    y_true: np.ndarray,
    pred_v241: np.ndarray,
    pred_candidate: np.ndarray,
    manifest: pd.DataFrame,
    valid_mask: np.ndarray,
    candidate_name: str,
) -> Path:
    """绘制 v249 hard case casebook。"""

    test = per_sample[per_sample["split"].eq("test")].copy()
    # 优先看 v241 很差且 candidate 后仍差/或改善明显的样本。
    selected = test.sort_values(["rmse_v241", "rmse_candidate"], ascending=[False, False]).head(8)
    if selected.empty:
        selected = test.head(8)
    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(3.0, 2.4 * n)), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, selected.iterrows()):
        idx = int(row["rolling_sample_index"])
        valid = valid_mask[idx]
        t = FUTURE_GRID[valid]
        ax.plot(t, y_true[idx, valid], color="#222222", linewidth=2.0, label="真实")
        ax.plot(t, pred_v241[idx, valid], color="#009E73", linestyle="--", linewidth=1.6, label="v241")
        ax.plot(t, pred_candidate[idx, valid], color="#D55E00", linestyle="-.", linewidth=1.8, label=candidate_name)
        title = (
            f"{row['event_uid']} | delay={int(row['delay_ms'])}ms | "
            f"v241={row['rmse_v241']:.3f} -> v249={row['rmse_candidate']:.3f} | "
            f"range {row['range_ratio_v241']:.2f}->{row['range_ratio_candidate']:.2f}, "
            f"slope {row['slope_ratio_v241']:.2f}->{row['slope_ratio_candidate']:.2f}"
        )
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("steering_delta")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("observation 后时间 / s")
    fig.tight_layout()
    out = FIGURES / "v249_shape_casebook_test_hard.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_compare_bars(compare: pd.DataFrame, best_name: str, v241_name: str) -> Path:
    """绘制核心 bucket tail delta。"""

    rows = compare[compare["bucket"].isin(["normal_predictable", "strong_steer", "observe_later_like"])].copy()
    delta_col = f"delta_steer_tail_rmse_mean__{best_name}_minus_v241"
    if delta_col not in rows.columns:
        out = FIGURES / "v249_tail_delta_by_bucket.png"
        return out
    fig, ax = plt.subplots(figsize=(10, 5))
    rows["label"] = rows["bucket"].astype(str) + "_" + rows["delay_ms"].astype(str) + "ms"
    colors = np.where(rows[delta_col].to_numpy(dtype=float) <= 0, "#0072B2", "#D55E00")
    ax.bar(rows["label"], rows[delta_col], color=colors)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_ylabel(f"{best_name} tail RMSE - {v241_name}")
    ax.set_title("v249 locked test tail delta by bucket/delay")
    ax.tick_params(axis="x", labelrotation=75, labelsize=8)
    fig.tight_layout()
    out = FIGURES / "v249_tail_delta_by_bucket.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def build_next_decision(selection: pd.DataFrame) -> pd.DataFrame:
    """输出下一步决策。"""

    best = selection.sort_values("validation_selection_score").iloc[0]
    accepted = selection[selection["accepted_as_shape_candidate"].astype(bool)].copy()
    if accepted.empty:
        accepted_name = ""
        accept_decision = False
        reason = "No v249 candidate passed validation no-harm + v241-upgrade + shape-gain checks."
        next_task = "v249_error_review_or_input_ambiguity_followup"
    else:
        winner = accepted.sort_values("validation_selection_score").iloc[0]
        accepted_name = str(winner["model_name"])
        accept_decision = True
        reason = f"{accepted_name} passed validation checks and can enter locked audit / narrowed follow-up."
        next_task = "v250_locked_shape_model_audit_or_refine_v249"
    return pd.DataFrame(
        [
            {
                "decision_item": "best_diagnostic_shape_model",
                "decision": str(best["model_name"]),
                "reason": "Best by validation selection score; not automatically a formal replacement.",
            },
            {
                "decision_item": "accept_shape_model_as_next_candidate",
                "decision": accept_decision,
                "reason": reason,
            },
            {
                "decision_item": "accepted_model_name",
                "decision": accepted_name,
                "reason": "Empty means v249 remains diagnostic only.",
            },
            {
                "decision_item": "formal_replacement_allowed",
                "decision": False,
                "reason": "v249 is a shape-aware diagnostic experiment; formal claim needs locked audit and robustness checks.",
            },
            {
                "decision_item": "recommended_next_task",
                "decision": next_task,
                "reason": "Do not use test to retune; use ambiguity audit and validation evidence to decide next bounded step.",
            },
        ]
    )


def build_guardrail_json(selection: pd.DataFrame, split_check: pd.DataFrame) -> Dict[str, object]:
    """实验边界检查。"""

    return {
        "pass": bool(
            int(split_check["split_check_status"].eq("fail").sum()) == 0
            and not bool(selection["test_used_for_selection"].astype(bool).any())
        ),
        "same_event_uid_cross_split_count": int(split_check["split_check_status"].eq("fail").sum()),
        "test_used_for_selection": bool(selection["test_used_for_selection"].astype(bool).any()),
        "forbidden_routes": {
            "anchor_selector": False,
            "gate_router_selector": False,
            "response_type_hard_routing": False,
            "sample_deletion": False,
            "oracle_best_anchor_as_policy": False,
        },
        "model_selection": "validation_only",
        "inherits_v241_checkpoint": True,
        "uses_original_remaining_target": True,
    }


def write_report(
    selection: pd.DataFrame,
    compare: pd.DataFrame,
    shape_summary: pd.DataFrame,
    ambiguity: pd.DataFrame,
    next_decision: pd.DataFrame,
    figures: List[Path],
    best_name: str,
    v241_name: str,
) -> Path:
    """写中文报告。"""

    lines: List[str] = []
    lines.append("# v249 shape-aware curve model 报告")
    lines.append("")
    lines.append("## 本轮边界")
    lines.append("")
    lines.append("- 从 v241 checkpoint 初始化，继承 TCN + multi-head query attention backbone。")
    lines.append("- 继续使用 `original_remaining` masked target。")
    lines.append("- 不做 anchor selector、gate/router、response-type hard routing，不删除样本。")
    lines.append("- 只用 validation 选择候选；test 只做 locked report。")
    lines.append("")
    lines.append("## Validation 选择")
    lines.append("")
    for _, r in selection.iterrows():
        lines.append(
            f"- `{r.model_name}`: score={r.validation_selection_score:.4f}, "
            f"noharm_vs_v236={bool(r.noharm_vs_v236_pass)}, upgrade_vs_v241={bool(r.upgrade_vs_v241_pass)}, "
            f"shape_gain={bool(r.shape_gain_pass)}, accepted={bool(r.accepted_as_shape_candidate)}"
        )
    lines.append("")
    lines.append(f"当前 best diagnostic model：`{best_name}`。")
    lines.append("")
    lines.append("## Test 对照摘要")
    lines.append("")
    view_cols = [
        "bucket",
        "delay_ms",
        f"steer_tail_rmse_mean__{v241_name}",
        f"steer_tail_rmse_mean__{best_name}",
        f"delta_steer_tail_rmse_mean__{best_name}_minus_v241",
        f"peak_ratio_mean__{v241_name}",
        f"peak_ratio_mean__{best_name}",
    ]
    existing = [c for c in view_cols if c in compare.columns]
    if existing:
        lines.append(compare[existing].to_markdown(index=False))
    lines.append("")
    lines.append("## Shape 指标摘要")
    lines.append("")
    test_shape = shape_summary[shape_summary["split"].eq("test") & shape_summary["model_name"].eq(best_name)].copy()
    keep_groups = ["all", "normal", "strong_steer", "observe_later_like", "bad_top10_v241"]
    test_shape = test_shape[test_shape["event_group"].isin(keep_groups)]
    if not test_shape.empty:
        show = (
            test_shape.groupby(["event_group"], as_index=False)
            .agg(
                n=("n", "sum"),
                mean_rmse=("mean_rmse", "mean"),
                mean_range_ratio=("mean_range_ratio", "mean"),
                mean_slope_ratio=("mean_slope_ratio", "mean"),
                mean_delta_rmse=("delta_rmse_candidate_minus_v241", "mean"),
                mean_delta_range=("delta_range_ratio_candidate_minus_v241", "mean"),
                mean_delta_slope=("delta_slope_ratio_candidate_minus_v241", "mean"),
            )
            .sort_values("event_group")
        )
        lines.append(show.to_markdown(index=False))
    lines.append("")
    lines.append("## 输入近邻不可判别审查")
    lines.append("")
    if ambiguity.empty:
        lines.append("- 未生成 ambiguity rows。")
    else:
        n_amb = int(ambiguity["ambiguity_category"].eq("input_ambiguous").sum())
        lines.append(
            f"- test delay=0 v241 bad_top10 共审查 `{len(ambiguity)}` 个样本，其中 `{n_amb}` 个被标记为 `input_ambiguous`。"
        )
        cols = [
            "event_uid",
            "v241_rmse",
            "candidate_rmse",
            "neighbor_future_pairwise_rmse_mean",
            "neighbor_peak_abs_std",
            "query_vs_neighbor_best_rmse",
            "ambiguity_category",
        ]
        lines.append(ambiguity[cols].head(12).to_markdown(index=False))
    lines.append("")
    lines.append("## 下一步决策")
    lines.append("")
    lines.append(next_decision.to_markdown(index=False))
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    lines.append("")
    lines.append("## 关键产物")
    lines.append("")
    lines.append("- `tables/v249_model_selection_validation_shape.csv`")
    lines.append("- `tables/v249_metrics_by_delay_and_bucket.csv`")
    lines.append("- `tables/v249_compare_vs_v241_original_remaining.csv`")
    lines.append("- `tables/v249_shape_summary.csv`")
    lines.append("- `tables/v249_per_sample_shape_delta_vs_v241.csv`")
    lines.append("- `tables/v249_input_neighborhood_ambiguity_audit.csv`")
    lines.append("- `figures/v249_shape_casebook_test_hard.png`")
    lines.append("- `figures/v249_tail_delta_by_bucket.png`")
    out = REPORTS / "v249_shape_aware_curve_model_cn.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_input_hashes() -> None:
    """记录关键输入文件 hash。"""

    files = [
        V242_SCRIPT,
        V241_PRED,
        V241_MODEL,
        V241.V239.V238.V236_NPZ if hasattr(V241.V239.V238, "V236_NPZ") else None,
    ]
    rows = []
    for path in files:
        if path is None:
            continue
        path = Path(path)
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
                "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else 0,
            }
        )
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def make_zip() -> Path:
    """打包 v249 关键产物。"""

    zip_path = BASELINES / "v249_shape_aware_curve_model_20260630_pack.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in (TABLES, REPORTS, LOGS, FIGURES):
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(OUT)).replace("\\", "/"))
        for path in MODELS.rglob("*"):
            if path.is_file() and path.suffix in {".pkl", ".json"}:
                zf.write(path, arcname=str(path.relative_to(OUT)).replace("\\", "/"))
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
    if bad is not None:
        raise AssertionError(f"ZIP 校验失败：{bad}")
    return zip_path


def main() -> None:
    set_seed(SEED)
    clean_out_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v249] device={device}")

    print("[v249] load v236/v241 data")
    data = V238.load_v236_data()
    x_base = V238.build_base_design_matrix(data)
    point_data = V238.build_point_dataset(data, x_base)
    point_masks = V238.split_point_masks(point_data, data.manifest)
    task_table, point_counts = V238.build_task_construction_tables(data, point_data)
    masks = sample_masks(data.manifest)
    pred_v241, v241_name = load_v241_prediction()
    checkpoint = load_v241_checkpoint()

    print("[v249] standardize arrays with train-only scalers")
    scalers = V239.fit_scalers(data, point_data, point_masks)
    arrays = V239.standardize_arrays(data, point_data, scalers)

    base_config = checkpoint["config"]
    configs: List[Tuple[str, Dict[str, object]]] = [
        (
            "v249a_shape_loss_only",
            {
                "dropout": float(base_config["dropout"]),
                "use_aux_heads": False,
                "shape_conditioned": False,
                "residual_scale": 0.0,
                "peak_weight": 0.08,
                "tail_weight": 0.04,
                "slope_weight": 0.015,
                "curvature_weight": 0.04,
                "excursion_weight": 0.03,
                "aux_weight": 0.0,
                "teacher_guard_weight": 0.03,
                "lr": 2.0e-4,
                "min_lr": 5.0e-6,
                "weight_decay": 5.0e-4,
                "batch_size": 128,
                "max_epochs": 55,
                "patience": 8,
            },
        ),
        (
            "v249b_shape_aux_heads",
            {
                "dropout": float(base_config["dropout"]),
                "use_aux_heads": True,
                "shape_conditioned": False,
                "residual_scale": 0.0,
                "peak_weight": 0.08,
                "tail_weight": 0.04,
                "slope_weight": 0.015,
                "curvature_weight": 0.04,
                "excursion_weight": 0.03,
                "aux_weight": 0.03,
                "teacher_guard_weight": 0.03,
                "lr": 2.0e-4,
                "min_lr": 5.0e-6,
                "weight_decay": 5.0e-4,
                "batch_size": 128,
                "max_epochs": 55,
                "patience": 8,
            },
        ),
        (
            "v249c_shape_conditioned_residual",
            {
                "dropout": float(base_config["dropout"]),
                "use_aux_heads": True,
                "shape_conditioned": True,
                "residual_scale": 0.20,
                "peak_weight": 0.08,
                "tail_weight": 0.04,
                "slope_weight": 0.015,
                "curvature_weight": 0.04,
                "excursion_weight": 0.03,
                "aux_weight": 0.03,
                "teacher_guard_weight": 0.04,
                "lr": 1.5e-4,
                "min_lr": 5.0e-6,
                "weight_decay": 5.0e-4,
                "batch_size": 128,
                "max_epochs": 55,
                "patience": 8,
            },
        ),
    ]

    runs: List[V249Run] = []
    for model_name, config in configs:
        print(f"[v249] training {model_name}")
        run = train_v249_candidate(
            model_name,
            config,
            data,
            point_data,
            arrays,
            scalers,
            masks,
            pred_v241,
            checkpoint,
            device,
        )
        runs.append(run)
        print(f"[v249] {model_name} best_epoch={run.best_epoch} best_val_loss={run.best_val_loss:.6f}")

    print("[v249] compute metrics")
    y_true_curve = data.y_future[:, :, 0].astype(np.float32)
    valid_mask = V238.build_original_remaining_mask(data.manifest)[0].astype(bool)
    pred_by_model: Dict[str, np.ndarray] = {
        "v236_joint_ridge_existing": data.pred_v236[:, :, 0].astype(np.float32),
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
    bad_thresholds: Dict[str, float] = {}
    shape_summary = build_shape_validation_table(
        y_true_curve,
        pred_v241.astype(np.float32),
        pred_by_model,
        data.manifest,
        valid_mask,
        bad_thresholds,
    )

    selection_rows: List[Dict[str, object]] = []
    for run in runs:
        row = candidate_validation_decision(metrics, run.model_name, v241_name, shape_summary)
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
    best_name = str(selection.iloc[0]["model_name"])
    best_run = next(run for run in runs if run.model_name == best_name)

    compare = build_compare_table(metrics, pred_by_model.keys(), v241_name)
    per_sample = per_sample_shape_delta(
        y_true_curve,
        pred_v241.astype(np.float32),
        best_run.pred_curve.astype(np.float32),
        data.manifest,
        valid_mask,
        best_name,
    )
    ambiguity = build_neighbor_ambiguity_audit(
        y_true_curve,
        pred_v241.astype(np.float32),
        best_run.pred_curve.astype(np.float32),
        arrays,
        data.manifest,
        valid_mask,
        k=K_NEIGHBORS,
    )
    split_check = V238.split_integrity_check(data.manifest)
    guardrail = build_guardrail_json(selection, split_check)
    if not bool(guardrail["pass"]):
        raise AssertionError("v249 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    next_decision = build_next_decision(selection)

    print("[v249] write outputs")
    write_csv(task_table, TABLES / "v249_task_construction_audit.csv")
    write_csv(point_counts, TABLES / "v249_point_training_rows_by_delay.csv")
    write_csv(selection, TABLES / "v249_model_selection_validation_shape.csv")
    write_csv(metrics, TABLES / "v249_metrics_by_delay_and_bucket.csv")
    write_csv(compare, TABLES / "v249_compare_vs_v241_original_remaining.csv")
    write_csv(pd.concat([run.training_history for run in runs], ignore_index=True), TABLES / "v249_training_history.csv")
    write_csv(shape_summary, TABLES / "v249_shape_summary.csv")
    write_csv(per_sample, TABLES / "v249_per_sample_shape_delta_vs_v241.csv")
    write_csv(
        per_sample[per_sample["split"].eq("test")]
        .sort_values("delta_rmse_candidate_minus_v241", ascending=False)
        .head(80),
        TABLES / "v249_worst_regressions_vs_v241.csv",
    )
    write_csv(
        per_sample[per_sample["split"].eq("test")]
        .sort_values("delta_rmse_candidate_minus_v241", ascending=True)
        .head(80),
        TABLES / "v249_top_improvements_vs_v241.csv",
    )
    write_csv(ambiguity, TABLES / "v249_input_neighborhood_ambiguity_audit.csv")
    write_csv(next_decision, TABLES / "v249_next_decision.csv")
    write_csv(split_check, TABLES / "v249_split_integrity_check.csv")
    write_csv(
        pd.DataFrame([{"split": k, "bad_top10_v241_threshold": v} for k, v in bad_thresholds.items()]),
        TABLES / "v249_bad_top10_thresholds.csv",
    )

    np.savez_compressed(
        OUT / "v249_shape_aware_predictions.npz",
        y_true_steering_delta=y_true_curve.astype(np.float32),
        pred_v236_steering_delta=pred_by_model["v236_joint_ridge_existing"].astype(np.float32),
        pred_v241_steering_delta=pred_v241.astype(np.float32),
        pred_v249_best_shape_steering_delta=best_run.pred_curve.astype(np.float32),
        best_shape_model=np.array([best_name], dtype="U120"),
        source_v241_model=np.array([v241_name], dtype="U120"),
        delay_ms=data.manifest["delay_ms"].to_numpy(dtype=np.int32),
        split=data.manifest["split"].astype(str).to_numpy(dtype="U16"),
        event_uid=data.manifest["event_uid"].astype(str).to_numpy(dtype="U160"),
        future_grid_s=FUTURE_GRID.astype(np.float32),
        original_remaining_valid=valid_mask.astype(np.bool_),
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
            "model_class": "ShapeAwareV241CurveModel",
            "state_dict": best_run.state_dict,
            "config": best_run.config,
            "base_v241_config": checkpoint["config"],
            "scalers": scalers_payload,
            "selection": selection.to_dict(orient="records"),
            "source_v241_model": v241_name,
        },
        MODELS / "v249_best_shape_aware_diagnostic.pt",
    )
    with (MODELS / "v249_scalers_and_selection.pkl").open("wb") as f:
        pickle.dump({"scalers": scalers_payload, "selection": selection}, f)

    figures = [
        plot_shape_casebook(per_sample, y_true_curve, pred_v241, best_run.pred_curve, data.manifest, valid_mask, best_name),
        plot_compare_bars(compare, best_name, v241_name),
    ]
    report_path = write_report(selection, compare, shape_summary, ambiguity, next_decision, figures, best_name, v241_name)

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
        "stage": "v249_shape_aware_curve_model",
        "created_by": Path(__file__).name,
        "output_dir": str(OUT),
        "source_v241_dir": str(V241_DIR),
        "n_rolling_samples": int(len(data.manifest)),
        "n_events": int(data.manifest["event_uid"].nunique()),
        "device": str(device),
        "candidates": [run.model_name for run in runs],
        "best_diagnostic_model": best_name,
        "accepted_as_shape_candidate": bool(selection["accepted_as_shape_candidate"].astype(bool).any()),
        "figures": [str(path.relative_to(OUT)).replace("\\", "/") for path in figures],
        "report": str(report_path.relative_to(OUT)).replace("\\", "/"),
    }
    (LOGS / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    inventory_rows = []
    for path in OUT.rglob("*"):
        if path.is_file():
            inventory_rows.append(
                {
                    "relative_path": str(path.relative_to(OUT)).replace("\\", "/"),
                    "size_bytes": int(path.stat().st_size),
                }
            )
    write_csv(pd.DataFrame(inventory_rows), LOGS / "file_inventory.csv")

    zip_path = make_zip()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_bad = zf.testzip()
        zip_count = len(zf.namelist())
    guardrail["zip_testzip"] = zip_bad
    guardrail["zip_items"] = zip_count
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[v249] best={best_name}")
    print(f"[v249] accepted={bool(selection['accepted_as_shape_candidate'].astype(bool).any())}")
    print(f"[v249] report={report_path}")
    print(f"[v249] zip={zip_path}")


if __name__ == "__main__":
    main()
