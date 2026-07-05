#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v259 生理-车辆 cross-attention 直接轨迹预测实验。

本轮目标：
- 不走 v222a gate / 删除样本 / 轻量 residual 修正；
- 不再把生理压成简单拼接特征，也不只做候选轨迹 tie-break；
- 直接让锚点前 raw 生理序列作为一组时序 token，被未来每个预测点 cross-attention 查询；
- 与同一车辆时序结构的 vehicle-only 模型公平对照，验证生理是否能补足锚点前车辆信息不足。

输入：
- v252/v250 固定 minimal_lateral7 车辆样本、v250 locked 预测、v251 逐样本误差；
- v256 已缓存的锚点前 20s raw 生理序列，形状为 [sample, channel, 400]。

输出：
- subject-disjoint 正式泛化口径；
- subject-aware 个体化诊断口径；
- all / bad_top10 / strong_steer / observe_later_like 分桶指标；
- train-only model selection 和 no-oracle guardrail。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import zipfile
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V252_SCRIPT = BASELINES / "scripts" / "stage03_v252_input_similarity_future_divergence_20260701.py"
V254B_SCRIPT = BASELINES / "scripts" / "stage03_v254b_physio_200hz_event_representation_20260702.py"
V256_SEQ = BASELINES / "v256_raw_physio_cnn_fusion_20260702" / "tensors" / "v256_physio_seq_20s_20hz.npz"
V256_METRICS = BASELINES / "v256_raw_physio_cnn_fusion_20260702" / "tables" / "v256_prediction_metrics_by_bucket.csv"

OUT = BASELINES / "v259_physio_cross_attention_prediction_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
TENSORS = OUT / "tensors"
ZIP_PATH = BASELINES / "v259_physio_cross_attention_prediction_20260702_pack.zip"

SEED = 25902
BATCH_SIZE = 384
EPOCHS = 70
PATIENCE = 10
LR = 7.0e-4
WEIGHT_DECAY = 5.0e-4
HIDDEN_DIM = 96
N_HEADS = 4
TAIL_LOSS_WEIGHT = 2.0
BAD_SAMPLE_WEIGHT = 4.0

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    """按路径导入已经验证过的前序脚本，复用 split、mask 和数据读取逻辑。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V252 = import_module_from_path("stage03_v252_for_v259", V252_SCRIPT)
V254B = import_module_from_path("stage03_v254b_for_v259", V254B_SCRIPT)
FUTURE_GRID = V252.FUTURE_GRID.astype(np.float32)


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, TENSORS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v259 自己的输出目录。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 统一用 utf-8-sig，方便 Windows/Excel 查看中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算输入文件 SHA256，用于追溯复现。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    """固定随机种子，降低重复运行波动。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def finite_nanmedian(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """nanmedian 的安全版本；全空列回退为 0。"""

    with np.errstate(all="ignore"):
        med = np.nanmedian(x, axis=axis)
    med = np.asarray(med, dtype=float)
    med[~np.isfinite(med)] = 0.0
    return med


def standardize_array_by_train(x: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    用训练 split 估计每个位置/维度的填充值、均值和标准差。

    这里对 hist/road/phase 分别 flatten 后标准化，再 reshape 回原形状；这样不会把 val/test 信息泄漏进 scaler。
    """

    original_shape = x.shape
    flat = np.asarray(x, dtype=float).reshape(original_shape[0], -1)
    train_x = flat[train_mask]
    med = finite_nanmedian(train_x, axis=0)
    filled = np.where(np.isfinite(flat), flat, med[None, :])
    mean = np.nanmean(filled[train_mask], axis=0)
    std = np.nanstd(filled[train_mask], axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    z = (filled - mean[None, :]) / std[None, :]
    audit = pd.DataFrame(
        {
            "feature_i": np.arange(flat.shape[1]),
            "train_finite_n": np.isfinite(train_x).sum(axis=0),
            "train_mean_after_fill": mean,
            "train_std_after_fill": std,
        }
    )
    return z.reshape(original_shape).astype(np.float32), audit


def load_physio_sequence() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """读取 v256 已缓存的 raw 生理序列，避免重复抽取 200Hz 原始文件。"""

    if not V256_SEQ.exists():
        raise FileNotFoundError(f"缺少 v256 raw 生理缓存，请先运行 v256：{V256_SEQ}")
    cache = np.load(V256_SEQ, allow_pickle=False)
    seq = cache["physio_seq"].astype(np.float32)
    ok = cache["physio_ok"].astype(np.float32)
    signals = [str(x) for x in cache["signals"]]
    return seq, ok, signals


def build_bad_top10_by_split(reference_tail: np.ndarray, split: np.ndarray) -> np.ndarray:
    """在每个 split 内按 v250 tail RMSE 的 90 分位定义 bad_top10，避免 test 阈值参与训练。"""

    bad = np.zeros(len(reference_tail), dtype=bool)
    for split_name in ["train", "val", "test"]:
        mask = split == split_name
        vals = reference_tail[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        q90 = float(np.quantile(vals, 0.90))
        bad[mask] = reference_tail[mask] >= q90
    return bad


def build_point_weights(
    manifest: pd.DataFrame,
    split: np.ndarray,
    sample_metrics: pd.DataFrame,
    bad_weighted: bool,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    构造训练损失权重。

    权重只来自训练 split 的 v250 误差标签和未来时间位置，不使用 test future。
    val/test 权重只用于保存审计，不参与训练。
    """

    n = len(manifest)
    weights = np.ones((n, len(FUTURE_GRID)), dtype=np.float32)
    for i, delay in enumerate(manifest["delay_ms"].astype(int).to_numpy()):
        tail = V252.future_tail_mask(int(delay))
        weights[i, tail] *= float(TAIL_LOSS_WEIGHT)

    v250_tail = pd.to_numeric(sample_metrics["tail_rmse_v250"], errors="coerce").to_numpy(dtype=float)
    bad = build_bad_top10_by_split(v250_tail, split)
    sample_weight = np.ones(n, dtype=np.float32)
    if bad_weighted:
        sample_weight[(split == "train") & bad] = float(BAD_SAMPLE_WEIGHT)
    weights *= sample_weight[:, None]
    audit = pd.DataFrame(
        {
            "row_index": np.arange(n),
            "split": split,
            "delay_ms": manifest["delay_ms"].astype(int).to_numpy(),
            "bad_top10_v250_by_split": bad,
            "sample_weight": sample_weight,
            "point_weight_mean": weights.mean(axis=1),
        }
    )
    return weights.astype(np.float32), audit


class ConvTokenEncoder(nn.Module):
    """把短时序编码成 token 序列，供后续 cross-attention 查询。"""

    def __init__(self, in_dim: int, hidden_dim: int, long_sequence: bool):
        super().__init__()
        if long_sequence:
            # 生理序列 400 步，需要下采样到较短 token 序列。
            self.net = nn.Sequential(
                nn.Conv1d(in_dim, hidden_dim // 2, kernel_size=9, stride=2, padding=4),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.GELU(),
                nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(in_dim, hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 [B, T, C] 或 [B, C, T]，输出统一为 [B, T_token, H]。
        if x.shape[1] < x.shape[-1]:
            # raw 生理通常是 [B, C, T]。
            z = x
        else:
            z = x.transpose(1, 2)
        tokens = self.net(z).transpose(1, 2)
        return tokens


class RoadQueryEncoder(nn.Module):
    """把未来每个点的 road token 和 sample phase 编成查询 token。"""

    def __init__(self, road_dim: int, phase_dim: int, n_steps: int, hidden_dim: int):
        super().__init__()
        self.road_proj = nn.Linear(road_dim, hidden_dim)
        self.phase_proj = nn.Sequential(nn.Linear(phase_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.pos = nn.Parameter(torch.zeros(1, n_steps, hidden_dim))
        nn.init.normal_(self.pos, std=0.02)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, road: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        phase_token = self.phase_proj(phase).unsqueeze(1)
        query = self.road_proj(road) + phase_token + self.pos[:, : road.shape[1], :]
        return self.norm(query)


class CrossAttentionTrajectoryModel(nn.Module):
    """
    车辆时序 + raw 生理时序 cross-attention 预测模型。

    kind:
    - vehicle_attn：只用车辆 history/road/phase；
    - vehicle_physio_crossattn：未来每个预测点同时 attend 车辆 history token 和生理 token。
    """

    def __init__(
        self,
        kind: str,
        hist_dim: int,
        road_dim: int,
        phase_dim: int,
        physio_channels: int,
        n_steps: int,
        hidden_dim: int = HIDDEN_DIM,
        n_heads: int = N_HEADS,
    ):
        super().__init__()
        self.kind = kind
        self.hidden_dim = int(hidden_dim)
        self.hist_encoder = ConvTokenEncoder(hist_dim, hidden_dim, long_sequence=False)
        self.query_encoder = RoadQueryEncoder(road_dim, phase_dim, n_steps, hidden_dim)
        self.hist_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=0.10, batch_first=True)
        self.use_physio = kind.startswith("vehicle_physio")
        if self.use_physio:
            self.physio_encoder = ConvTokenEncoder(physio_channels, hidden_dim, long_sequence=True)
            self.physio_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=0.10, batch_first=True)
            head_in = hidden_dim * 3 + phase_dim + 1
        else:
            self.physio_encoder = None
            self.physio_attn = None
            head_in = hidden_dim * 2 + phase_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, 192),
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Dropout(0.12),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(96, 1),
        )

    def forward(
        self,
        hist: torch.Tensor,
        road: torch.Tensor,
        phase: torch.Tensor,
        physio: torch.Tensor,
        physio_ok: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_encoder(road, phase)
        hist_tokens = self.hist_encoder(hist)
        hist_ctx, _ = self.hist_attn(query, hist_tokens, hist_tokens, need_weights=False)
        phase_rep = phase.unsqueeze(1).expand(-1, query.shape[1], -1)
        if self.use_physio:
            physio_tokens = self.physio_encoder(physio)
            physio_ctx, _ = self.physio_attn(query, physio_tokens, physio_tokens, need_weights=False)
            ok_rep = physio_ok.view(-1, 1, 1).expand(-1, query.shape[1], 1)
            physio_ctx = physio_ctx * ok_rep
            head_in = torch.cat([query, hist_ctx, physio_ctx, phase_rep, ok_rep], dim=-1)
        else:
            head_in = torch.cat([query, hist_ctx, phase_rep], dim=-1)
        return self.head(head_in).squeeze(-1)


def masked_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    point_weight: torch.Tensor,
) -> torch.Tensor:
    """带 original_remaining mask 和训练点权重的 MSE。"""

    weight = valid * point_weight
    diff2 = torch.square(pred - target) * weight
    return diff2.sum() / torch.clamp(weight.sum(), min=1.0)


def make_loader(
    indices: np.ndarray,
    hist: np.ndarray,
    road: np.ndarray,
    phase: np.ndarray,
    physio: np.ndarray,
    physio_ok: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    point_weight: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    """构造 PyTorch DataLoader。"""

    ds = TensorDataset(
        torch.from_numpy(hist[indices].astype(np.float32)),
        torch.from_numpy(road[indices].astype(np.float32)),
        torch.from_numpy(phase[indices].astype(np.float32)),
        torch.from_numpy(physio[indices].astype(np.float32)),
        torch.from_numpy(physio_ok[indices].astype(np.float32)),
        torch.from_numpy(y[indices].astype(np.float32)),
        torch.from_numpy(valid[indices].astype(np.float32)),
        torch.from_numpy(point_weight[indices].astype(np.float32)),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray]:
    """在一个 loader 上计算 loss 并返回预测。"""

    model.eval()
    losses: List[float] = []
    preds: List[np.ndarray] = []
    for hist, road, phase, physio, ok, y, valid, weight in loader:
        hist = hist.to(device, non_blocking=True)
        road = road.to(device, non_blocking=True)
        phase = phase.to(device, non_blocking=True)
        physio = physio.to(device, non_blocking=True)
        ok = ok.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)
        pred = model(hist, road, phase, physio, ok)
        loss = masked_weighted_mse(pred, y, valid, weight)
        losses.append(float(loss.detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(preds, axis=0)


def train_one_model(
    protocol: str,
    kind: str,
    split: np.ndarray,
    hist: np.ndarray,
    road: np.ndarray,
    phase: np.ndarray,
    physio: np.ndarray,
    physio_ok: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    point_weight: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """训练一个 v259 模型；checkpoint 选择只看 validation loss。"""

    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    all_idx = np.arange(len(split))
    train_loader = make_loader(train_idx, hist, road, phase, physio, physio_ok, y, valid, point_weight, shuffle=True)
    val_loader = make_loader(val_idx, hist, road, phase, physio, physio_ok, y, valid, point_weight, shuffle=False)
    all_loader = make_loader(all_idx, hist, road, phase, physio, physio_ok, y, valid, point_weight, shuffle=False)

    seed_offset = {
        "vehicle_attn": 11,
        "vehicle_physio_crossattn": 23,
        "vehicle_physio_crossattn_badweighted": 37,
    }[kind]
    set_seed(SEED + seed_offset + (0 if protocol == "subject_disjoint" else 1000))
    model = CrossAttentionTrajectoryModel(
        kind=kind,
        hist_dim=hist.shape[-1],
        road_dim=road.shape[-1],
        phase_dim=phase.shape[-1],
        physio_channels=physio.shape[1],
        n_steps=y.shape[1],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4, min_lr=1.0e-5)

    best_state = None
    best_val = math.inf
    bad_epochs = 0
    rows: List[Dict[str, object]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses: List[float] = []
        for hist_b, road_b, phase_b, physio_b, ok_b, y_b, valid_b, weight_b in train_loader:
            hist_b = hist_b.to(device, non_blocking=True)
            road_b = road_b.to(device, non_blocking=True)
            phase_b = phase_b.to(device, non_blocking=True)
            physio_b = physio_b.to(device, non_blocking=True)
            ok_b = ok_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            valid_b = valid_b.to(device, non_blocking=True)
            weight_b = weight_b.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(hist_b, road_b, phase_b, physio_b, ok_b)
            loss = masked_weighted_mse(pred, y_b, valid_b, weight_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss, _ = evaluate_model(model, val_loader, device)
        scheduler.step(val_loss)
        val_rmse = math.sqrt(max(val_loss, 0.0))
        rows.append(
            {
                "protocol": protocol,
                "model_name": f"v259_{kind}",
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "val_rmse_weighted": val_rmse,
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"[v259] {protocol}/{kind} epoch={epoch} weighted_val_rmse={val_rmse:.4f}", flush=True)
        if bad_epochs >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, pred_all = evaluate_model(model, all_loader, device)
    return pred_all.astype(np.float32), pd.DataFrame(rows)


def sample_rmse(pred: np.ndarray, y: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """逐样本 original_remaining RMSE。"""

    diff2 = np.square(pred - y)
    diff2 = np.where(valid_mask, diff2, np.nan)
    with np.errstate(all="ignore"):
        return np.sqrt(np.nanmean(diff2, axis=1))


def sample_tail_rmse(pred: np.ndarray, y: np.ndarray, valid_mask: np.ndarray, delays: np.ndarray) -> np.ndarray:
    """逐样本 tail RMSE。"""

    out = np.full(len(y), np.nan, dtype=float)
    for i, delay in enumerate(delays):
        tail = V252.future_tail_mask(int(delay))
        mask = valid_mask[i] & tail
        if int(mask.sum()) < 2:
            continue
        out[i] = float(np.sqrt(np.mean(np.square(pred[i, mask] - y[i, mask]))))
    return out


def summarize_predictions(
    protocol: str,
    split: np.ndarray,
    manifest: pd.DataFrame,
    sample_metrics: pd.DataFrame,
    y: np.ndarray,
    valid_mask: np.ndarray,
    pred_map: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """汇总 all/bad/strong/observe_later 分桶指标和逐样本指标。"""

    delays = manifest["delay_ms"].astype(int).to_numpy()
    v250_tail = pd.to_numeric(sample_metrics["tail_rmse_v250"], errors="coerce").to_numpy(dtype=float)
    bad_top10 = build_bad_top10_by_split(v250_tail, split)
    bucket_defs = [
        ("all", np.ones(len(split), dtype=bool)),
        ("bad_top10_v250", bad_top10),
        ("strong_steer", sample_metrics["is_strong_steer"].astype(bool).to_numpy()),
        ("observe_later_like", sample_metrics["is_observe_later_like"].astype(bool).to_numpy()),
    ]
    metric_rows: List[Dict[str, object]] = []
    per_sample_rows: List[Dict[str, object]] = []

    for model_name, pred in pred_map.items():
        rmse = sample_rmse(pred, y, valid_mask)
        tail = sample_tail_rmse(pred, y, valid_mask, delays)
        for i in range(len(manifest)):
            per_sample_rows.append(
                {
                    "protocol": protocol,
                    "model_name": model_name,
                    "row_index": i,
                    "event_uid": str(manifest.iloc[i]["event_uid"]),
                    "subject": str(manifest.iloc[i]["subject"]),
                    "recording": str(manifest.iloc[i]["recording"]),
                    "split": str(split[i]),
                    "delay_ms": int(delays[i]),
                    "sample_rmse": float(rmse[i]),
                    "tail_rmse": float(tail[i]),
                    "bad_top10_v250_bucket": bool(bad_top10[i]),
                    "is_strong_steer": bool(sample_metrics.iloc[i]["is_strong_steer"]),
                    "is_observe_later_like": bool(sample_metrics.iloc[i]["is_observe_later_like"]),
                }
            )
        for eval_split in ["val", "test"]:
            split_mask = split == eval_split
            for bucket, bucket_mask in bucket_defs:
                mask = split_mask & bucket_mask
                if int(mask.sum()) == 0:
                    continue
                metric_rows.append(
                    {
                        "protocol": protocol,
                        "eval_split": eval_split,
                        "bucket": bucket,
                        "model_name": model_name,
                        "n": int(mask.sum()),
                        "sample_rmse_mean": float(np.nanmean(rmse[mask])),
                        "sample_rmse_median": float(np.nanmedian(rmse[mask])),
                        "tail_rmse_mean": float(np.nanmean(tail[mask])),
                        "tail_rmse_median": float(np.nanmedian(tail[mask])),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    v250_base = metrics[metrics["model_name"].eq("v250_existing")][
        ["protocol", "eval_split", "bucket", "sample_rmse_mean", "tail_rmse_mean"]
    ].rename(
        columns={
            "sample_rmse_mean": "v250_sample_rmse_mean",
            "tail_rmse_mean": "v250_tail_rmse_mean",
        }
    )
    veh_base = metrics[metrics["model_name"].eq("v259_vehicle_attn")][
        ["protocol", "eval_split", "bucket", "sample_rmse_mean", "tail_rmse_mean"]
    ].rename(
        columns={
            "sample_rmse_mean": "v259_vehicle_sample_rmse_mean",
            "tail_rmse_mean": "v259_vehicle_tail_rmse_mean",
        }
    )
    metrics = metrics.merge(v250_base, on=["protocol", "eval_split", "bucket"], how="left")
    metrics = metrics.merge(veh_base, on=["protocol", "eval_split", "bucket"], how="left")
    metrics["delta_tail_rmse_vs_v250"] = metrics["tail_rmse_mean"] - metrics["v250_tail_rmse_mean"]
    metrics["delta_tail_rmse_vs_v259_vehicle"] = metrics["tail_rmse_mean"] - metrics["v259_vehicle_tail_rmse_mean"]
    metrics["delta_sample_rmse_vs_v250"] = metrics["sample_rmse_mean"] - metrics["v250_sample_rmse_mean"]
    metrics["delta_sample_rmse_vs_v259_vehicle"] = metrics["sample_rmse_mean"] - metrics["v259_vehicle_sample_rmse_mean"]
    return metrics, pd.DataFrame(per_sample_rows)


def choose_by_validation(metrics: pd.DataFrame) -> pd.DataFrame:
    """只用 validation 排名 v259 候选模型，test 只做 locked 报告。"""

    rows: List[Dict[str, object]] = []
    models = [
        "v259_vehicle_attn",
        "v259_vehicle_physio_crossattn",
        "v259_vehicle_physio_crossattn_badweighted",
    ]
    for protocol in metrics["protocol"].drop_duplicates():
        sub = metrics[metrics["protocol"].eq(protocol) & metrics["eval_split"].eq("val")].copy()
        all_base = sub[sub["bucket"].eq("all") & sub["model_name"].eq("v250_existing")]
        for model in models:
            bad = sub[sub["bucket"].eq("bad_top10_v250") & sub["model_name"].eq(model)]
            all_row = sub[sub["bucket"].eq("all") & sub["model_name"].eq(model)]
            if bad.empty or all_row.empty or all_base.empty:
                continue
            all_harm_vs_v250 = max(0.0, float(all_row["tail_rmse_mean"].iloc[0] - all_base["tail_rmse_mean"].iloc[0]))
            score = float(bad["tail_rmse_mean"].iloc[0]) + 2.0 * all_harm_vs_v250
            rows.append(
                {
                    "protocol": protocol,
                    "model_name": model,
                    "val_bad_top10_tail_rmse": float(bad["tail_rmse_mean"].iloc[0]),
                    "val_all_tail_rmse": float(all_row["tail_rmse_mean"].iloc[0]),
                    "val_all_harm_vs_v250": all_harm_vs_v250,
                    "selection_score": score,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["validation_rank"] = out.groupby("protocol")["selection_score"].rank(method="first")
    out["chosen_by_validation"] = out["validation_rank"].eq(1.0)
    return out.sort_values(["protocol", "validation_rank"]).reset_index(drop=True)


def plot_test_bucket_tail(metrics: pd.DataFrame) -> Path:
    """画 test 分桶 tail RMSE 对照图。"""

    path = FIGURES / "v259_test_bucket_tail_rmse.png"
    sub = metrics[
        metrics["eval_split"].eq("test")
        & metrics["bucket"].isin(["all", "bad_top10_v250", "strong_steer", "observe_later_like"])
        & metrics["model_name"].isin(
            [
                "v250_existing",
                "v259_vehicle_attn",
                "v259_vehicle_physio_crossattn",
                "v259_vehicle_physio_crossattn_badweighted",
            ]
        )
    ].copy()
    protocols = list(sub["protocol"].drop_duplicates())
    buckets = ["all", "bad_top10_v250", "strong_steer", "observe_later_like"]
    models = [
        "v250_existing",
        "v259_vehicle_attn",
        "v259_vehicle_physio_crossattn",
        "v259_vehicle_physio_crossattn_badweighted",
    ]
    fig, axes = plt.subplots(len(protocols), 1, figsize=(13.5, 4.6 * max(1, len(protocols))), squeeze=False)
    for ax, protocol in zip(axes[:, 0], protocols):
        g = sub[sub["protocol"].eq(protocol)]
        x = np.arange(len(buckets))
        width = 0.82 / len(models)
        for j, model in enumerate(models):
            vals = []
            for bucket in buckets:
                r = g[g["bucket"].eq(bucket) & g["model_name"].eq(model)]
                vals.append(float(r["tail_rmse_mean"].iloc[0]) if len(r) else np.nan)
            ax.bar(x + (j - (len(models) - 1) / 2) * width, vals, width=width, label=model)
        ax.set_title(f"{protocol}: test tail RMSE")
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)
        ax.set_ylabel("tail RMSE")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def read_v256_focus() -> pd.DataFrame:
    """读取 v256 指标，作为报告里的前序 raw-CNN 参照。"""

    if not V256_METRICS.exists():
        return pd.DataFrame()
    m = pd.read_csv(V256_METRICS, encoding="utf-8-sig")
    return m[
        m["eval_split"].eq("test")
        & m["bucket"].isin(["all", "bad_top10_v250"])
        & m["model_name"].isin(["v256_vehicle_only", "v256_vehicle_physio_cnn"])
    ].copy()


def write_input_hashes() -> None:
    """记录关键输入文件哈希。"""

    rows = []
    for label, path in [
        ("v252_script", V252_SCRIPT),
        ("v254b_script", V254B_SCRIPT),
        ("v256_physio_seq_cache", V256_SEQ),
        ("v256_metrics", V256_METRICS),
    ]:
        rows.append(
            {
                "label": label,
                "path": str(path),
                "exists": path.exists(),
                "sha256": file_sha256(path) if path.exists() else "",
            }
        )
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    """记录输出目录文件清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    """打包输出，较大的中间缓存不进入 zip。"""

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip() is None


def write_report(
    metrics: pd.DataFrame,
    train_log: pd.DataFrame,
    selection: pd.DataFrame,
    physio_signals: List[str],
    physio_ok_rate: float,
    v256_focus: pd.DataFrame,
    figures: List[Path],
) -> None:
    """写中文报告。"""

    lines: List[str] = []
    lines.append("# v259 生理-车辆 cross-attention 直接预测实验")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("- 前序 v254b-v258 说明：手工生理统计、raw-CNN 简单融合、候选重排序、同驾驶员记忆和 anchor selector 都没有形成稳定生理增量。")
    lines.append("- v259 检查一个更强但仍干净的路线：raw 生理序列不再简单拼接，而是作为时序 token 被每个未来预测点 cross-attention 查询。")
    lines.append("- 本轮不做删样本、不做 hard gate/router、不做 residual 修正；输出仍是单条可部署轨迹。")
    lines.append("")
    lines.append("## 输入与模型")
    lines.append("")
    lines.append(f"- 车辆输入：v250 minimal_lateral7 的 history/road/phase，history 形状为 31x7。")
    lines.append(f"- 生理输入：v256 raw cache，通道为 {', '.join(physio_signals)}，20s x 20Hz = 400 步。")
    lines.append(f"- 生理覆盖率：{physio_ok_rate:.4f}。缺失生理时序保留为 0，并显式输入 physio_ok。")
    lines.append("- 模型：vehicle_attn 是车辆时序 attention baseline；vehicle_physio_crossattn 在同一查询上额外 attend 生理 token。")
    lines.append(f"- 损失：tail 点权重 {TAIL_LOSS_WEIGHT:.1f}；badweighted 版本仅在训练 split 对 v250 bad_top10 样本乘 {BAD_SAMPLE_WEIGHT:.1f} 权重。")
    lines.append("")
    lines.append("## Validation 选型")
    lines.append("")
    if selection.empty:
        lines.append("- selection 表为空。")
    else:
        lines.append(selection.to_markdown(index=False))
    lines.append("")
    lines.append("## Test 关键结果")
    lines.append("")
    focus = metrics[
        metrics["eval_split"].eq("test")
        & metrics["bucket"].isin(["all", "bad_top10_v250", "strong_steer", "observe_later_like"])
        & metrics["model_name"].isin(
            [
                "v250_existing",
                "v259_vehicle_attn",
                "v259_vehicle_physio_crossattn",
                "v259_vehicle_physio_crossattn_badweighted",
            ]
        )
    ].copy()
    lines.append(
        focus[
            [
                "protocol",
                "bucket",
                "model_name",
                "n",
                "sample_rmse_mean",
                "tail_rmse_mean",
                "delta_tail_rmse_vs_v250",
                "delta_tail_rmse_vs_v259_vehicle",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")
    if not v256_focus.empty:
        lines.append("## v256 raw-CNN 参照")
        lines.append("")
        lines.append(
            v256_focus[
                [
                    "protocol",
                    "bucket",
                    "model_name",
                    "n",
                    "sample_rmse_mean",
                    "tail_rmse_mean",
                    "delta_tail_rmse_vs_v256_vehicle",
                ]
            ].to_markdown(index=False)
        )
        lines.append("")
    lines.append("## 判读")
    lines.append("")
    for protocol in ["subject_disjoint", "subject_aware"]:
        bad = focus[focus["protocol"].eq(protocol) & focus["bucket"].eq("bad_top10_v250")]
        veh = bad[bad["model_name"].eq("v259_vehicle_attn")]
        phys = bad[bad["model_name"].eq("v259_vehicle_physio_crossattn")]
        badw = bad[bad["model_name"].eq("v259_vehicle_physio_crossattn_badweighted")]
        base = bad[bad["model_name"].eq("v250_existing")]
        if len(base) and len(veh) and len(phys) and len(badw):
            lines.append(
                f"- {protocol} bad_top10：v250 tail={float(base['tail_rmse_mean'].iloc[0]):.4f}；"
                f"v259 vehicle={float(veh['tail_rmse_mean'].iloc[0]):.4f}；"
                f"physio={float(phys['tail_rmse_mean'].iloc[0]):.4f}；"
                f"physio_badweighted={float(badw['tail_rmse_mean'].iloc[0]):.4f}。"
            )
    lines.append("- 如果 vehicle+physio 明显低于 vehicle-only，说明 raw 生理 cross-attention 真的补充了车辆锚点前信息。")
    lines.append("- 如果 vehicle+physio 没有低于 vehicle-only 或 v250，说明瓶颈不是融合结构太弱，而是当前生理片段对未来方向盘曲线的可判别信息仍不足。")
    lines.append("")
    lines.append("## 训练日志摘要")
    lines.append("")
    last = train_log.sort_values("epoch").groupby(["protocol", "model_name"], as_index=False).tail(1)
    lines.append(last[["protocol", "model_name", "epoch", "val_rmse_weighted", "lr"]].to_markdown(index=False))
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v259_physio_cross_attention_prediction_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v259] physio cross-attention prediction", flush=True)
    clean_out_dir()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v259] device={device}", flush=True)

    loaded = V252.load_fixed_inputs()
    data = loaded["data"]
    manifest = data.manifest.copy()
    y = loaded["y_true"].astype(np.float32)
    valid_mask = loaded["valid_mask"].astype(bool)
    sample_metrics = loaded["sample_metrics"].copy()
    pred_v250 = loaded["pred_v250"].astype(np.float32)
    split_disjoint = manifest["split"].astype(str).to_numpy()
    split_aware = V254B.make_subject_aware_split(manifest)

    physio_seq, physio_ok, physio_signals = load_physio_sequence()
    if len(physio_seq) != len(manifest):
        raise AssertionError(f"physio_seq 行数 {len(physio_seq)} 与 manifest {len(manifest)} 不一致")

    all_metrics: List[pd.DataFrame] = []
    all_samples: List[pd.DataFrame] = []
    all_train_logs: List[pd.DataFrame] = []
    all_weight_audit: List[pd.DataFrame] = []
    all_predictions: Dict[str, np.ndarray] = {}

    for protocol, split in [("subject_disjoint", split_disjoint), ("subject_aware", split_aware)]:
        train_mask = split == "train"
        hist, hist_audit = standardize_array_by_train(data.x_hist.astype(np.float32), train_mask)
        road, road_audit = standardize_array_by_train(data.x_road.astype(np.float32), train_mask)
        phase, phase_audit = standardize_array_by_train(data.x_phase.astype(np.float32), train_mask)
        hist_audit["protocol"] = protocol
        hist_audit["block"] = "hist"
        road_audit["protocol"] = protocol
        road_audit["block"] = "road"
        phase_audit["protocol"] = protocol
        phase_audit["block"] = "phase"
        write_csv(pd.concat([hist_audit, road_audit, phase_audit], ignore_index=True), TABLES / f"v259_{protocol}_standardization_audit.csv")

        pred_map: Dict[str, np.ndarray] = {"v250_existing": pred_v250}
        for kind in ["vehicle_attn", "vehicle_physio_crossattn", "vehicle_physio_crossattn_badweighted"]:
            bad_weighted = kind.endswith("_badweighted")
            point_weight, weight_audit = build_point_weights(manifest, split, sample_metrics, bad_weighted=bad_weighted)
            weight_audit["protocol"] = protocol
            weight_audit["model_name"] = f"v259_{kind}"
            all_weight_audit.append(weight_audit)
            print(f"[v259] train {protocol}/{kind}", flush=True)
            pred, log = train_one_model(
                protocol=protocol,
                kind=kind,
                split=split,
                hist=hist,
                road=road,
                phase=phase,
                physio=physio_seq,
                physio_ok=physio_ok,
                y=y,
                valid=valid_mask,
                point_weight=point_weight,
                device=device,
            )
            model_name = f"v259_{kind}"
            pred_map[model_name] = pred
            all_predictions[f"{protocol}__{model_name}"] = pred
            all_train_logs.append(log)

        metrics, samples = summarize_predictions(protocol, split, manifest, sample_metrics, y, valid_mask, pred_map)
        all_metrics.append(metrics)
        all_samples.append(samples)

    metrics = pd.concat(all_metrics, ignore_index=True)
    samples = pd.concat(all_samples, ignore_index=True)
    train_log = pd.concat(all_train_logs, ignore_index=True)
    weight_audit = pd.concat(all_weight_audit, ignore_index=True)
    selection = choose_by_validation(metrics)

    write_csv(metrics, TABLES / "v259_prediction_metrics_by_bucket.csv")
    write_csv(samples, TABLES / "v259_per_sample_prediction_metrics.csv")
    write_csv(train_log, TABLES / "v259_training_log.csv")
    write_csv(selection, TABLES / "v259_validation_model_selection.csv")
    write_csv(weight_audit, TABLES / "v259_training_weight_audit.csv")
    np.savez_compressed(TENSORS / "v259_predictions.npz", **all_predictions)

    figures = [plot_test_bucket_tail(metrics)]
    v256_focus = read_v256_focus()
    write_input_hashes()
    write_file_inventory()
    write_report(
        metrics=metrics,
        train_log=train_log,
        selection=selection,
        physio_signals=physio_signals,
        physio_ok_rate=float(np.mean(physio_ok)),
        v256_focus=v256_focus,
        figures=figures,
    )
    write_file_inventory()
    zip_ok = make_zip()
    guardrail = {
        "pass": bool(zip_ok),
        "zip_testzip": bool(zip_ok),
        "device": str(device),
        "train_only_model_selection": True,
        "no_sample_deletion": True,
        "no_gate_router": True,
        "no_residual_correction": True,
        "physio_seq_shape": list(physio_seq.shape),
        "physio_ok_rate": float(np.mean(physio_ok)),
        "n_metric_rows": int(len(metrics)),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not guardrail["pass"]:
        raise AssertionError("v259 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    focus = metrics[
        metrics["protocol"].eq("subject_disjoint")
        & metrics["eval_split"].eq("test")
        & metrics["bucket"].eq("bad_top10_v250")
        & metrics["model_name"].isin(
            ["v250_existing", "v259_vehicle_attn", "v259_vehicle_physio_crossattn", "v259_vehicle_physio_crossattn_badweighted"]
        )
    ].copy()
    print(f"[v259] report={REPORTS / 'v259_physio_cross_attention_prediction_cn.md'}", flush=True)
    print(f"[v259] zip={ZIP_PATH}", flush=True)
    if len(focus):
        print(
            focus[
                [
                    "model_name",
                    "tail_rmse_mean",
                    "delta_tail_rmse_vs_v250",
                    "delta_tail_rmse_vs_v259_vehicle",
                ]
            ].to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
