#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v242 联合曲线解码模型训练。

本轮目标：
- 继续使用 v238/v239/v241 已固定的 original_remaining masked target；
- 从“逐 future point 独立查询”改成“一次解码 21 个未来点”；
- 让模型在同一个前向传播里学习整条 steering_delta 曲线形态；
- 用 validation-only 选择模型，test 只做固定后的报告；
- 不创建 gate/router/selector，不做 response-type hard routing，不删除样本。

方法直观解释：
v241 已经能在 bucket 均值上明显优于 v239，但逐样本仍有回退。
v242 不再只是把网络加宽，而是把未来曲线作为一个整体输出，并在训练损失里加入轻量曲线差分约束，
希望减少点与点之间形态不连贯导致的局部回退。
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

V241_SCRIPT = BASELINES / "scripts" / "stage03_v241_stronger_temporal_model_20260626.py"
V241_DIR = BASELINES / "v241_stronger_temporal_model_20260626"
V241_PRED = V241_DIR / "v241_stronger_temporal_predictions.npz"
V241_MODEL = V241_DIR / "models" / "v241_best_stronger_temporal_diagnostic.pt"
V241_SELECTION = V241_DIR / "tables" / "v241_model_selection_validation_noharm.csv"
V241_DECISION = V241_DIR / "tables" / "v241_next_decision.csv"

OUT = BASELINES / "v242_joint_curve_decoder_20260626"
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
SEED = 242

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    """按文件路径导入前序脚本，复用已经验证过的函数。"""

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
class JointCurveRun:
    """一个联合曲线候选模型的训练结果。"""

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
    """只清理 v242 自己的输出目录。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 统一使用 utf-8-sig。"""

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
    """固定随机种子。"""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


class JointCurveDataset(Dataset):
    """sample-level 数据集：每个样本一次返回 21 个 future point。"""

    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        point_data,
        sample_mask: np.ndarray,
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
        }


class JointCurveDecoder(nn.Module):
    """
    联合曲线解码器。

    历史和道路序列先通过 v241 的 temporal convolution encoder；
    21 个 future point 作为 21 个 query token，一次性读取历史/道路上下文；
    再通过一个轻量曲线 token mixer，让相邻未来点之间可以交换信息；
    最后逐点输出 steering_delta。
    """

    def __init__(
        self,
        hist_dim: int,
        road_dim: int,
        phase_dim: int,
        point_dim: int,
        hist_len: int,
        road_len: int,
        n_steps: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        mixer_layers: int,
        mlp_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} 必须能被 n_heads={n_heads} 整除")
        self.hist_encoder = V241.TemporalConvEncoder(hist_dim, hidden_dim, hist_len, n_layers, dropout)
        self.road_encoder = V241.TemporalConvEncoder(road_dim, hidden_dim, road_len, max(1, n_layers - 1), dropout)
        self.query_input = nn.Sequential(
            nn.Linear(phase_dim + point_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.future_pos = nn.Parameter(torch.zeros(1, n_steps, hidden_dim))
        self.hist_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.road_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3 + phase_dim + point_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.mixers = nn.ModuleList(
            [V241.TemporalConvBlock(hidden_dim, dilation=1 + i, dropout=dropout) for i in range(mixer_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )
        nn.init.normal_(self.future_pos, mean=0.0, std=0.02)

    def forward(
        self,
        hist: torch.Tensor,
        road: torch.Tensor,
        phase: torch.Tensor,
        point_seq: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, _ = point_seq.shape
        hist_tokens = self.hist_encoder(hist)
        road_tokens = self.road_encoder(road)
        phase_rep = phase.unsqueeze(1).expand(batch, steps, phase.shape[-1])
        query = self.query_input(torch.cat([phase_rep, point_seq], dim=-1)) + self.future_pos[:, :steps, :]
        hist_ctx, _ = self.hist_attn(query, hist_tokens, hist_tokens, need_weights=False)
        road_ctx, _ = self.road_attn(query, road_tokens, road_tokens, need_weights=False)
        x = self.fuse(torch.cat([query, hist_ctx, road_ctx, phase_rep, point_seq], dim=-1))
        for mixer in self.mixers:
            x = mixer(x)
        return self.head(x).squeeze(-1)


def masked_curve_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    smooth_weight: float,
) -> torch.Tensor:
    """masked MSE + 轻量差分形态约束。"""

    point_weight = valid * weight
    mse = torch.sum(torch.square(pred - target) * point_weight) / torch.clamp(torch.sum(point_weight), min=1e-6)
    if smooth_weight <= 0:
        return mse
    adj_valid = valid[:, 1:] * valid[:, :-1]
    adj_weight = 0.5 * (weight[:, 1:] + weight[:, :-1]) * adj_valid
    pred_diff = pred[:, 1:] - pred[:, :-1]
    true_diff = target[:, 1:] - target[:, :-1]
    diff_loss = torch.sum(torch.square(pred_diff - true_diff) * adj_weight) / torch.clamp(
        torch.sum(adj_weight),
        min=1e-6,
    )
    return mse + float(smooth_weight) * diff_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    smooth_weight: float,
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
        point_seq = batch["point_seq"].to(device=device, dtype=torch.float32)
        y_seq = batch["y_seq"].to(device=device, dtype=torch.float32)
        valid_seq = batch["valid_seq"].to(device=device, dtype=torch.float32)
        weight_seq = batch["weight_seq"].to(device=device, dtype=torch.float32)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(hist, road, phase, point_seq)
        loss = masked_curve_loss(pred, y_seq, valid_seq, weight_seq, smooth_weight)
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
        wsum = float(torch.sum(valid_seq * weight_seq).detach().cpu().item())
        total_loss += float(loss.detach().cpu().item()) * wsum
        total_weight += wsum
    return total_loss / max(total_weight, 1e-6)


def predict_curves(
    model: nn.Module,
    arrays: Dict[str, np.ndarray],
    scalers,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """对所有 rolling samples 一次输出完整 21 点曲线，并还原到原始 steering_delta 空间。"""

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
            pred = model(hist, road, phase, points)
            pred_scaled[start:end] = pred.detach().cpu().numpy().astype(np.float32)
    return (pred_scaled * scalers.y_std + scalers.y_mean).astype(np.float32)


def sample_masks(manifest: pd.DataFrame) -> Dict[str, np.ndarray]:
    """按 split 生成 sample-level mask。"""

    split = manifest["split"].astype(str).to_numpy()
    return {name: split == name for name in ["train", "val", "test"]}


def train_joint_candidate(
    model_name: str,
    config: Dict[str, object],
    data,
    point_data,
    arrays: Dict[str, np.ndarray],
    scalers,
    masks: Dict[str, np.ndarray],
    device: torch.device,
) -> JointCurveRun:
    """训练一个联合曲线候选模型。"""

    train_dataset = JointCurveDataset(arrays, point_data, masks["train"])
    val_dataset = JointCurveDataset(arrays, point_data, masks["val"])
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
    model = JointCurveDecoder(
        hist_dim=data.x_hist.shape[-1],
        road_dim=data.x_road.shape[-1],
        phase_dim=data.x_phase.shape[-1],
        point_dim=len(V238.POINT_EXTRA_FEATURE_NAMES),
        hist_len=data.x_hist.shape[1],
        road_len=data.x_road.shape[1],
        n_steps=len(FUTURE_GRID),
        hidden_dim=int(config["hidden_dim"]),
        n_heads=int(config["n_heads"]),
        n_layers=int(config["n_layers"]),
        mixer_layers=int(config["mixer_layers"]),
        mlp_hidden=int(config["mlp_hidden"]),
        dropout=float(config["dropout"]),
    ).to(device)
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
    smooth_weight = float(config["smooth_weight"])
    best_val = math.inf
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: List[Dict[str, object]] = []
    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, smooth_weight)
        val_loss = run_epoch(model, val_loader, device, None, smooth_weight)
        scheduler.step(val_loss)
        lr_now = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "model_name": model_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr_now,
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
        raise AssertionError(f"{model_name} 没有生成 best_state")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    pred_curve = predict_curves(model, arrays, scalers, device, batch_size=batch_size * 4)
    return JointCurveRun(
        model_name=model_name,
        config=config,
        state_dict=best_state,
        pred_curve=pred_curve.astype(np.float32),
        training_history=pd.DataFrame(history),
        training_seconds=float(time.time() - start_time),
        best_epoch=int(best_epoch),
        best_val_loss=float(best_val),
    )


def load_v241_prediction() -> Tuple[np.ndarray, str]:
    """读取 v241 已选 stronger temporal 模型预测。"""

    if not V241_PRED.exists():
        raise FileNotFoundError(f"缺少 v241 prediction npz：{V241_PRED}")
    with np.load(V241_PRED, allow_pickle=False) as pred:
        arr = pred["pred_v241_best_stronger_steering_delta"].astype(np.float32)
        model_name = str(pred["best_stronger_model"][0])
    return arr, model_name


def load_v239_prediction() -> Tuple[np.ndarray, str]:
    """读取 v239 attention 预测，作为报告对照。"""

    with np.load(V241.V239_PRED, allow_pickle=False) as pred:
        arr = pred["pred_v239_best_attention_steering_delta"].astype(np.float32)
        model_name = str(pred["best_attention_model"][0])
    return arr, model_name


def finite_mean(values: pd.Series, default: float = math.inf) -> float:
    """安全均值。"""

    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else default


def finite_max(values: pd.Series, default: float = math.inf) -> float:
    """安全最大值。"""

    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size else default


def positive_penalty(value: float, threshold: float = 0.0) -> float:
    """超过阈值的部分才计入 selection penalty。"""

    if not np.isfinite(value):
        return 10.0
    return max(0.0, float(value) - threshold)


def delta_frame(metrics: pd.DataFrame, candidate_name: str, ref_name: str) -> pd.DataFrame:
    """在 validation original_remaining 上生成 candidate-ref 差值。"""

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
    """抽取某一类样本的 delay 子集。"""

    out = merged[merged["bucket"].eq(bucket)].copy()
    if max_delay is not None:
        out = out[out["delay_ms"].astype(int) <= int(max_delay)].copy()
    if delays is not None:
        wanted = {int(x) for x in delays}
        out = out[out["delay_ms"].astype(int).isin(wanted)].copy()
    return out


def candidate_validation_decision(metrics: pd.DataFrame, candidate_name: str, v241_name: str) -> Dict[str, object]:
    """只用 validation 判断 v242 是否能替代 v241 进入下一阶段。"""

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
        and strong_exception_mean_tail_delta_v241 <= 0.0
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
        + positive_penalty(strong_exception_mean_tail_delta_v241, 0.0)
    )
    selection_score = base_score + 10.0 * penalty_vs_v236 + 6.0 * penalty_vs_v241

    return {
        "model_name": candidate_name,
        "selected_by": "validation_noharm_and_v241_upgrade_only",
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
        "noharm_vs_v236_pass": bool(noharm_vs_v236),
        "upgrade_vs_v241_pass": bool(upgrade_vs_v241),
        "accepted_as_next_candidate": bool(noharm_vs_v236 and upgrade_vs_v241),
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
        for metric in ["steer_sample_rmse_mean", "steer_tail_rmse_mean"]:
            ref_col = f"{metric}__{ref_name}"
            model_col = f"{metric}__{model}"
            if model != ref_name and ref_col in pivot.columns and model_col in pivot.columns:
                pivot[f"delta_{metric}__{model}_minus_v241"] = pivot[model_col] - pivot[ref_col]
    return pivot


def build_next_decision(selection: pd.DataFrame) -> pd.DataFrame:
    """输出下一步决策。"""

    accepted = selection[selection["accepted_as_next_candidate"].astype(bool)].copy()
    best = selection.sort_values("validation_selection_score").iloc[0]
    if accepted.empty:
        accepted_name = ""
        accept_decision = False
        reason = (
            "No v242 joint-curve candidate passed both v236 no-harm and v241-upgrade checks. "
            "Keep v241 as the stronger candidate."
        )
        next_task = "v243_manual_review_or_loss_redesign_for_sample_regressions"
    else:
        winner = accepted.sort_values("validation_selection_score").iloc[0]
        accepted_name = str(winner["model_name"])
        accept_decision = True
        reason = f"{accepted_name} passed validation no-harm and v241-upgrade checks; it can enter locked audit."
        next_task = "v243_locked_test_report_for_joint_curve_candidate"
    return pd.DataFrame(
        [
            {
                "decision_item": "best_diagnostic_joint_curve_model",
                "decision": str(best["model_name"]),
                "reason": "Best by validation selection score; not automatically a formal replacement.",
            },
            {
                "decision_item": "accept_joint_curve_model_as_next_candidate",
                "decision": accept_decision,
                "reason": reason,
            },
            {
                "decision_item": "accepted_model_name",
                "decision": accepted_name,
                "reason": "Empty means v242 should remain diagnostic only.",
            },
            {
                "decision_item": "formal_replacement_allowed",
                "decision": False,
                "reason": "v242 is a training experiment; formal headline remains locked until locked audit and robustness checks pass.",
            },
            {
                "decision_item": "recommended_next_task",
                "decision": next_task,
                "reason": "Do not use test to retune; fixed candidate must be locked-audited before any formal claim.",
            },
        ]
    )


def build_per_sample_delta_table(
    y_true_curve: np.ndarray,
    pred_ref: np.ndarray,
    pred_candidate: np.ndarray,
    manifest: pd.DataFrame,
    ref_name: str,
    candidate_name: str,
) -> pd.DataFrame:
    """输出 candidate 相对 reference 的逐样本差异。"""

    per_ref = V238.build_per_sample_metrics(y_true_curve, pred_ref, manifest, ref_name)
    per_cand = V238.build_per_sample_metrics(y_true_curve, pred_candidate, manifest, candidate_name)
    keys = [
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
    ]
    keep = keys + ["sample_rmse", "tail_rmse", "peak_ratio", "strong_under"]
    ref = per_ref[keep].rename(
        columns={
            "sample_rmse": "sample_rmse_ref",
            "tail_rmse": "tail_rmse_ref",
            "peak_ratio": "peak_ratio_ref",
            "strong_under": "strong_under_ref",
        }
    )
    cand = per_cand[keep].rename(
        columns={
            "sample_rmse": "sample_rmse_candidate",
            "tail_rmse": "tail_rmse_candidate",
            "peak_ratio": "peak_ratio_candidate",
            "strong_under": "strong_under_candidate",
        }
    )
    merged = cand.merge(ref, on=keys, how="left")
    merged["delta_sample_candidate_minus_ref"] = merged["sample_rmse_candidate"] - merged["sample_rmse_ref"]
    merged["delta_tail_candidate_minus_ref"] = merged["tail_rmse_candidate"] - merged["tail_rmse_ref"]
    merged["delta_peak_ratio_candidate_minus_ref"] = merged["peak_ratio_candidate"] - merged["peak_ratio_ref"]
    return merged


def build_per_sample_delta_summary(per_sample_delta: pd.DataFrame) -> pd.DataFrame:
    """按关键 bucket 汇总逐样本回退数量。"""

    test = per_sample_delta[per_sample_delta["split"].eq("test")].copy()
    masks = {
        "all": np.ones(len(test), dtype=bool),
        "observe_later_like": test["observe_later_like"].astype(bool).to_numpy(),
        "normal_predictable": (
            ~test["observe_later_like"].astype(bool).to_numpy()
            & ~test["strong_steer"].astype(bool).to_numpy()
        ),
        "strong_steer": test["strong_steer"].astype(bool).to_numpy(),
        "strong_400_1000": (
            test["strong_steer"].astype(bool).to_numpy()
            & test["delay_ms"].astype(int).isin(STRONG_EXCEPTION_DELAYS).to_numpy()
        ),
    }
    rows: List[Dict[str, object]] = []
    for bucket, mask in masks.items():
        one = test.loc[mask].copy()
        if one.empty:
            rows.append(
                {
                    "bucket": bucket,
                    "n": 0,
                    "tail_regressions_vs_ref": 0,
                    "tail_regression_rate_vs_ref": math.nan,
                    "mean_delta_tail_candidate_minus_ref": math.nan,
                    "max_delta_tail_candidate_minus_ref": math.nan,
                    "p90_delta_tail_candidate_minus_ref": math.nan,
                }
            )
            continue
        delta = one["delta_tail_candidate_minus_ref"].astype(float)
        regress = delta > 0.0
        rows.append(
            {
                "bucket": bucket,
                "n": int(len(one)),
                "tail_regressions_vs_ref": int(regress.sum()),
                "tail_regression_rate_vs_ref": float(regress.mean()),
                "mean_delta_tail_candidate_minus_ref": float(delta.mean()),
                "max_delta_tail_candidate_minus_ref": float(delta.max()),
                "p90_delta_tail_candidate_minus_ref": float(delta.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def plot_figures(compare: pd.DataFrame, selected_model: str, v241_name: str, v239_name: str) -> List[Path]:
    """生成 tail RMSE 对照图。"""

    paths: List[Path] = []
    model_styles = [
        ("v236_joint_ridge_existing", "#777777", "v236"),
        (v239_name, "#d62728", "v239 attention"),
        (v241_name, "#2ca02c", "v241 stronger"),
        (selected_model, "#9467bd", "v242 joint curve"),
    ]
    for bucket in ["observe_later_like", "strong_steer", "normal_predictable"]:
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
        ax.set_title(f"v242 joint curve decoder: {bucket}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = FIGURES / f"v242_joint_curve_tail_compare_{bucket}.png"
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
        V241_DECISION,
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
    """记录方法边界。"""

    checks = {
        "stage": "v242_joint_curve_decoder",
        "task_base": "v238_original_remaining_masked_point_level_target",
        "model_type": "sample_level_joint_curve_decoder_with_cross_attention",
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
    """打包输出并校验 ZIP。"""

    zip_path = OUT / "v242_joint_curve_decoder_pack.zip"
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
    per_sample_summary: pd.DataFrame,
    guardrail: Dict[str, object],
    device: torch.device,
    v241_name: str,
    zip_path: Path,
) -> None:
    """写中文报告。"""

    best = selection.sort_values("validation_selection_score").iloc[0]
    best_name = str(best["model_name"])
    accepted = selection[selection["accepted_as_next_candidate"].astype(bool)].copy()
    lines: List[str] = []
    lines.append("# v242 联合曲线解码模型训练报告")
    lines.append("")
    lines.append("## 本轮做了什么")
    lines.append("")
    lines.append("- 保留 `original_remaining` masked target。")
    lines.append("- 把逐点预测改成一次输出 21 个 future points 的联合曲线解码。")
    lines.append("- 增加轻量曲线差分约束，让模型学习曲线形态，而不是只追逐单点误差。")
    lines.append("- 不做 gate/router/selector，不做 response-type hard routing，不删除样本。")
    lines.append("- 只用 validation 选择模型；test 只做固定后的报告。")
    lines.append(f"- 训练设备：`{device}`。")
    lines.append("")
    lines.append("## Validation 选择结果")
    lines.append("")
    lines.append(
        f"- best diagnostic model：`{best_name}`，validation score={float(best.validation_selection_score):.6f}，"
        f"accepted_as_next_candidate={bool(best.accepted_as_next_candidate)}。"
    )
    lines.append(
        f"- vs v236：normal max tail delta={float(best.normal_max_tail_delta_vs_v236):+.6f}，"
        f"observe mean tail delta={float(best.observe_later_mean_tail_delta_vs_v236_0to800):+.6f}，"
        f"strong 0-600 mean tail delta={float(best.strong_mean_tail_delta_vs_v236_0to600):+.6f}。"
    )
    lines.append(
        f"- vs v241：normal max tail delta={float(best.normal_max_tail_delta_vs_v241):+.6f}，"
        f"observe mean tail delta={float(best.observe_later_mean_tail_delta_vs_v241_0to800):+.6f}，"
        f"strong 400/1000 mean tail delta={float(best.strong_exception_mean_tail_delta_vs_v241_400_1000):+.6f}。"
    )
    if accepted.empty:
        lines.append("- 没有 v242 候选同时通过 v236 no-harm 和 v241 upgrade；本轮只作为诊断训练。")
    else:
        winner = accepted.sort_values("validation_selection_score").iloc[0]
        lines.append(f"- 通过下一候选检查的模型：`{winner.model_name}`。")
    lines.append("")
    lines.append("## Test original_remaining 对照")
    lines.append("")
    for bucket in ["observe_later_like", "strong_steer", "normal_predictable"]:
        one = compare[compare["bucket"].eq(bucket)].copy().sort_values("delay_ms")
        if one.empty:
            continue
        lines.append(f"### {bucket}")
        delta_v236_col = f"delta_steer_tail_rmse_mean__{best_name}_minus_v236"
        delta_v241_col = f"delta_steer_tail_rmse_mean__{best_name}_minus_v241"
        for row in one.itertuples(index=False):
            lines.append(
                f"- delay={int(row.delay_ms)}ms：v242-v236 tail delta={float(getattr(row, delta_v236_col, math.nan)):+.6f}，"
                f"v242-v241 tail delta={float(getattr(row, delta_v241_col, math.nan)):+.6f}"
            )
        lines.append("")
    lines.append("## 逐样本回退摘要")
    lines.append("")
    lines.append("- 下面统计的是 test 样本内 v242 相对 v241 的逐样本 tail RMSE 是否变差。")
    for row in per_sample_summary.itertuples(index=False):
        lines.append(
            f"- `{row.bucket}`：n={int(row.n)}，tail 回退 {int(row.tail_regressions_vs_ref)} 条，"
            f"回退率={float(row.tail_regression_rate_vs_ref):.3f}，"
            f"mean delta={float(row.mean_delta_tail_candidate_minus_ref):+.6f}，"
            f"max delta={float(row.max_delta_tail_candidate_minus_ref):+.6f}。"
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
    lines.append("- `tables/v242_model_selection_validation_noharm.csv`")
    lines.append("- `tables/v242_metrics_by_delay_and_bucket.csv`")
    lines.append("- `tables/v242_compare_vs_v236_v239_v241_original_remaining.csv`")
    lines.append("- `tables/v242_per_sample_delta_vs_v241.csv`")
    lines.append("- `tables/v242_per_sample_delta_summary_vs_v241.csv`")
    lines.append("- `tables/v242_next_decision.csv`")
    lines.append(f"- ZIP：`{zip_path.name}`")
    lines.append("")
    (REPORTS / "v242_joint_curve_decoder_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    clean_out_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v242] device={device}")

    print("[v242] loading v236/v239/v241 data")
    data = V238.load_v236_data()
    x_base = V238.build_base_design_matrix(data)
    point_data = V238.build_point_dataset(data, x_base)
    point_masks = V238.split_point_masks(point_data, data.manifest)
    task_table, point_counts = V238.build_task_construction_tables(data, point_data)
    masks = sample_masks(data.manifest)
    pred_v239, v239_name = load_v239_prediction()
    pred_v241, v241_name = load_v241_prediction()

    print("[v242] standardizing with train-only scalers")
    scalers = V239.fit_scalers(data, point_data, point_masks)
    arrays = V239.standardize_arrays(data, point_data, scalers)

    configs: List[Tuple[str, Dict[str, object]]] = [
        (
            "v242_joint_curve_h64_smooth002",
            {
                "hidden_dim": 64,
                "n_heads": 4,
                "n_layers": 3,
                "mixer_layers": 2,
                "mlp_hidden": 96,
                "dropout": 0.08,
                "smooth_weight": 0.02,
                "lr": 6e-4,
                "min_lr": 1e-5,
                "weight_decay": 3e-4,
                "batch_size": 384,
                "max_epochs": 90,
                "patience": 12,
            },
        ),
        (
            "v242_joint_curve_h96_smooth005",
            {
                "hidden_dim": 96,
                "n_heads": 4,
                "n_layers": 4,
                "mixer_layers": 2,
                "mlp_hidden": 128,
                "dropout": 0.10,
                "smooth_weight": 0.05,
                "lr": 5e-4,
                "min_lr": 1e-5,
                "weight_decay": 5e-4,
                "batch_size": 256,
                "max_epochs": 90,
                "patience": 12,
            },
        ),
    ]

    runs: List[JointCurveRun] = []
    for model_name, config in configs:
        print(f"[v242] training {model_name}")
        run = train_joint_candidate(model_name, config, data, point_data, arrays, scalers, masks, device)
        runs.append(run)
        print(f"[v242] {model_name} best_epoch={run.best_epoch} best_val_loss={run.best_val_loss:.6f}")

    print("[v242] computing metrics and decisions")
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
    selection_rows: List[Dict[str, object]] = []
    for run in runs:
        row = candidate_validation_decision(metrics, run.model_name, v241_name)
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
    per_sample_delta = build_per_sample_delta_table(
        y_true_curve=y_true_curve,
        pred_ref=pred_v241.astype(np.float32),
        pred_candidate=best_run.pred_curve.astype(np.float32),
        manifest=data.manifest,
        ref_name=v241_name,
        candidate_name=best_name,
    )
    per_sample_summary = build_per_sample_delta_summary(per_sample_delta)
    split_check = V238.split_integrity_check(data.manifest)
    guardrail = build_guardrail_json(selection, split_check)
    if not bool(guardrail["pass"]):
        raise AssertionError("v242 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))

    next_decision = build_next_decision(selection)
    figure_paths = plot_figures(compare, best_name, v241_name, v239_name)

    print("[v242] writing outputs")
    write_csv(task_table, TABLES / "v242_task_construction_audit.csv")
    write_csv(point_counts, TABLES / "v242_point_training_rows_by_delay.csv")
    write_csv(selection, TABLES / "v242_model_selection_validation_noharm.csv")
    write_csv(metrics, TABLES / "v242_metrics_by_delay_and_bucket.csv")
    write_csv(compare, TABLES / "v242_compare_vs_v236_v239_v241_original_remaining.csv")
    write_csv(pd.concat([run.training_history for run in runs], ignore_index=True), TABLES / "v242_training_history.csv")
    write_csv(per_sample_delta, TABLES / "v242_per_sample_delta_vs_v241.csv")
    write_csv(per_sample_summary, TABLES / "v242_per_sample_delta_summary_vs_v241.csv")
    write_csv(
        per_sample_delta[per_sample_delta["split"].eq("test")]
        .sort_values("delta_tail_candidate_minus_ref", ascending=False)
        .head(80),
        TABLES / "v242_worst_regressions_vs_v241.csv",
    )
    write_csv(
        per_sample_delta[per_sample_delta["split"].eq("test")]
        .sort_values("delta_tail_candidate_minus_ref", ascending=True)
        .head(80),
        TABLES / "v242_top_improvements_vs_v241.csv",
    )
    write_csv(next_decision, TABLES / "v242_next_decision.csv")
    write_csv(split_check, TABLES / "v242_split_integrity_check.csv")

    np.savez_compressed(
        OUT / "v242_joint_curve_predictions.npz",
        y_true_steering_delta=y_true_curve.astype(np.float32),
        pred_v236_steering_delta=pred_by_model["v236_joint_ridge_existing"].astype(np.float32),
        pred_v238_steering_delta=pred_by_model["v238_selected_original_remaining_point_model"].astype(np.float32),
        pred_v239_steering_delta=pred_v239.astype(np.float32),
        pred_v241_steering_delta=pred_v241.astype(np.float32),
        pred_v242_best_joint_curve_steering_delta=best_run.pred_curve.astype(np.float32),
        best_joint_curve_model=np.array([best_name], dtype="U100"),
        source_v241_model=np.array([v241_name], dtype="U100"),
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
            "model_class": "JointCurveDecoder",
            "state_dict": best_run.state_dict,
            "config": best_run.config,
            "scalers": scalers_payload,
            "selection": selection.to_dict(orient="records"),
            "source_v241_model": v241_name,
        },
        MODELS / "v242_best_joint_curve_diagnostic.pt",
    )
    with (MODELS / "v242_scalers_and_selection.pkl").open("wb") as f:
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
        "stage": "v242_joint_curve_decoder",
        "created_by": Path(__file__).name,
        "output_dir": str(OUT),
        "source_v241_dir": str(V241_DIR),
        "n_rolling_samples": int(len(data.manifest)),
        "n_events": int(data.manifest["event_uid"].nunique()),
        "device": str(device),
        "joint_curve_candidates": [run.model_name for run in runs],
        "best_diagnostic_model": best_name,
        "accepted_as_next_candidate": bool(selection["accepted_as_next_candidate"].astype(bool).any()),
        "figures": [str(path.relative_to(OUT)).replace("\\", "/") for path in figure_paths],
    }
    (LOGS / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = zip_outputs()
    write_report(selection, next_decision, compare, per_sample_summary, guardrail, device, v241_name, zip_path)
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = zip_outputs()

    print("[v242] finished")
    print(f"output_dir={OUT}")
    print(f"best_diagnostic_model={best_name}")
    print(f"accepted_as_next_candidate={bool(selection['accepted_as_next_candidate'].astype(bool).any())}")
    print(f"report={REPORTS / 'v242_joint_curve_decoder_cn.md'}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
