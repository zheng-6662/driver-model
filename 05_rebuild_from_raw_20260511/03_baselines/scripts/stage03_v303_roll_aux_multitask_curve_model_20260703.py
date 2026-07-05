#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v303 roll-cause 辅助监督多任务曲线模型。

目的：
- 输出仍然是原任务的 21 点 steering_delta 曲线；
- 输入新增 v302 已验证的因果可见 roll-cause summary；
- v301 自动事件类型标签只作为训练期辅助监督，不作为推理输入；
- 用 validation-only 选择候选，并明确检查 bad_top10 no-harm。

核心结构：
hist/road/phase/point_seq -> joint curve decoder 主干
roll-cause summary -> roll encoder -> 事件类型辅助头 + 对曲线 token 的 FiLM 调制

边界：
- 不使用未来事件标签作为 feature；
- 不使用 test 后验误差作为 feature 或选择规则；
- v301 标签来自未来行为，只作为 supervised auxiliary target / training signal。
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
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


SEED = 20260703
ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "03_baselines"
SCRIPTS = BASELINES / "scripts"
V300_SCRIPT = SCRIPTS / "stage03_v300_within_subject_full_joint_curve_train_20260702.py"
V302_SCRIPT = SCRIPTS / "stage03_v302_roll_cause_input_audit_20260703.py"
V301_LABELS = BASELINES / "v301_event_type_multiclass_label_audit_20260703" / "tables" / "v301_event_type_labels.csv"
V300_PRED = BASELINES / "v300_within_subject_full_joint_curve_train_20260702" / "v300_within_subject_full_predictions.npz"
V300_GUARDRAIL = BASELINES / "v300_within_subject_full_joint_curve_train_20260702" / "logs" / "guardrail_check.json"
V300_MODELS = BASELINES / "v300_within_subject_full_joint_curve_train_20260702" / "models"

OUT = BASELINES / "v303_roll_aux_multitask_curve_model_20260703"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
MODELS = OUT / "models"


HARD_EVENT_TYPES = {
    "复合急制动转向",
    "急左转",
    "急右转",
    "紧急连续变道/避让",
}


@dataclass
class RollPrepared:
    """v303 训练所需的全部数组和元数据。"""

    data: object
    prepared: object
    roll_raw: np.ndarray
    roll_scaled: np.ndarray
    roll_feature_names: List[str]
    roll_impute_mean: np.ndarray
    roll_scale_mean: np.ndarray
    roll_scale_std: np.ndarray
    event_label: np.ndarray
    event_label_name: np.ndarray
    class_names: List[str]
    class_weight: np.ndarray
    curve_sample_multiplier: np.ndarray
    labels_table: pd.DataFrame
    event_table: pd.DataFrame


@dataclass
class V303Run:
    """一个 v303 候选模型的训练结果。"""

    model_name: str
    config: Dict[str, object]
    state_dict: Dict[str, torch.Tensor]
    pred_curve: np.ndarray
    event_logits: np.ndarray
    event_pred_class: np.ndarray
    training_history: pd.DataFrame
    training_seconds: float
    best_epoch: int
    best_val_loss: float


def import_module_from_path(module_name: str, path: Path):
    """按路径导入前序脚本，复用已经通过审计的数据构造逻辑。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少依赖脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V300 = import_module_from_path("stage03_v300_within_subject_full_joint_curve_train_for_v303", V300_SCRIPT)
V302 = import_module_from_path("stage03_v302_roll_cause_input_audit_for_v303", V302_SCRIPT)
V242 = V300.V242
V241 = V242.V241
V238 = V300.V238
V239 = V300.V239
FUTURE_GRID = V238.FUTURE_GRID.astype(np.float32)


def ensure_dirs() -> None:
    """创建 v303 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v303 自己的输出。"""

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
    """计算输入/产物文件哈希。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    """固定随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def load_v300_prediction_all(manifest: pd.DataFrame) -> Tuple[np.ndarray, str, Dict[str, object]]:
    """读取 v300 selected baseline 并校验顺序。"""

    if not V300_PRED.exists():
        raise FileNotFoundError(f"缺少 v300 预测：{V300_PRED}")
    with np.load(V300_PRED, allow_pickle=True) as z:
        pred = z["pred_v300_best_within_subject_full"].astype(np.float32)
        selected = str(z["best_v300_model"][0])
        event_uid = z["event_uid"].astype(str)
        delay_ms = z["delay_ms"].astype(int)
    if not np.array_equal(manifest["event_uid"].astype(str).to_numpy(), event_uid):
        raise AssertionError("v300 event_uid 与当前 manifest 不一致")
    if not np.array_equal(manifest["delay_ms"].astype(int).to_numpy(), delay_ms):
        raise AssertionError("v300 delay_ms 与当前 manifest 不一致")
    guard = json.loads(V300_GUARDRAIL.read_text(encoding="utf-8")) if V300_GUARDRAIL.exists() else {}
    return pred, selected, guard


def fit_transform_roll_features(roll_raw: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """只用 train split 对 roll-cause summary 做缺失填补和标准化。"""

    x = roll_raw.astype(np.float64, copy=True)
    train_values = x[train_mask]
    impute_mean = np.nanmean(train_values, axis=0)
    impute_mean[~np.isfinite(impute_mean)] = 0.0
    bad = ~np.isfinite(x)
    if bad.any():
        row_idx, col_idx = np.where(bad)
        x[row_idx, col_idx] = impute_mean[col_idx]
    mean = np.mean(x[train_mask], axis=0)
    std = np.std(x[train_mask], axis=0)
    mean[~np.isfinite(mean)] = 0.0
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    scaled = ((x - mean) / std).astype(np.float32)
    return scaled, impute_mean.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def load_event_labels_for_all_samples(manifest: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """把 v301 delay0 事件类型标签映射到同一 event 的全部 rolling delay 行。"""

    if not V301_LABELS.exists():
        raise FileNotFoundError(f"缺少 v301 标签表：{V301_LABELS}")
    labels = pd.read_csv(V301_LABELS, encoding="utf-8-sig")
    label_map = labels.set_index("event_uid")["event_primary_type"].astype(str)
    names = manifest["event_uid"].astype(str).map(label_map).astype(str).to_numpy()
    if pd.isna(names).any():
        raise AssertionError("存在无法映射 v301 event_primary_type 的 rolling 样本")
    class_names = sorted(pd.Series(names).unique().tolist())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    y = np.array([class_to_idx[name] for name in names], dtype=np.int64)
    return y, names, class_names, labels


def build_curve_sample_multiplier(manifest: pd.DataFrame, event_label_name: np.ndarray, hard_event_extra: float) -> np.ndarray:
    """构造训练用样本权重乘子；只进入 loss，不作为模型输入。"""

    mult = np.ones(len(manifest), dtype=np.float32)
    hard = np.array([name in HARD_EVENT_TYPES for name in event_label_name], dtype=bool)
    mult += hard.astype(np.float32) * float(hard_event_extra)
    if "strong_steer" in manifest.columns:
        mult += manifest["strong_steer"].astype(bool).to_numpy(dtype=bool).astype(np.float32) * 0.10
    if "observe_later_like" in manifest.columns:
        mult += manifest["observe_later_like"].astype(bool).to_numpy(dtype=bool).astype(np.float32) * 0.05
    return mult.astype(np.float32)


def prepare_v303_data(hard_event_extra: float) -> RollPrepared:
    """读取数据、构造 v302 roll-cause summary、映射辅助标签。"""

    raw_data = V238.load_v236_data()
    data, event_table = V300.apply_v299_within_subject_split(raw_data)
    no_subject = V300.prepare_variant(
        "no_subject",
        data,
        {
            "uses_subject_onehot": False,
            "description_cn": "v303 主线只使用车辆/道路/phase，不拼 subject one-hot",
        },
    )

    x_base = V238.build_base_design_matrix(data)
    roll_raw, roll_feature_names, signal_audit = V302.build_roll_cause_summary(x_base, data.feature_names)
    train_mask = data.manifest["split"].astype(str).to_numpy() == "train"
    roll_scaled, impute_mean, scale_mean, scale_std = fit_transform_roll_features(roll_raw, train_mask)
    event_label, event_label_name, class_names, labels = load_event_labels_for_all_samples(data.manifest)

    train_labels = event_label[train_mask]
    counts = np.bincount(train_labels, minlength=len(class_names)).astype(np.float32)
    counts[counts < 1] = 1.0
    class_weight = counts.sum() / (len(class_names) * counts)
    class_weight = np.clip(class_weight, 0.35, 4.0).astype(np.float32)
    curve_mult = build_curve_sample_multiplier(data.manifest, event_label_name, hard_event_extra)

    write_csv(signal_audit, TABLES / "v303_roll_cause_signal_coverage.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "roll_cause_feature_n": int(roll_scaled.shape[1]),
                    "raw_nan_rate": float(np.mean(~np.isfinite(roll_raw))),
                    "scaled_nan_rate": float(np.mean(~np.isfinite(roll_scaled))),
                    "event_class_n": int(len(class_names)),
                    "hard_event_extra": float(hard_event_extra),
                }
            ]
        ),
        TABLES / "v303_input_audit.csv",
    )
    write_csv(
        pd.DataFrame({"event_primary_type": class_names, "class_index": list(range(len(class_names))), "class_weight": class_weight}),
        TABLES / "v303_event_class_mapping.csv",
    )

    return RollPrepared(
        data=data,
        prepared=no_subject,
        roll_raw=roll_raw.astype(np.float32),
        roll_scaled=roll_scaled.astype(np.float32),
        roll_feature_names=roll_feature_names,
        roll_impute_mean=impute_mean,
        roll_scale_mean=scale_mean,
        roll_scale_std=scale_std,
        event_label=event_label,
        event_label_name=event_label_name,
        class_names=class_names,
        class_weight=class_weight,
        curve_sample_multiplier=curve_mult,
        labels_table=labels,
        event_table=event_table,
    )


class RollAuxCurveDataset(Dataset):
    """样本级曲线数据集，额外返回 roll-cause summary 和事件辅助标签。"""

    def __init__(self, arrays: Dict[str, np.ndarray], point_data, sample_mask: np.ndarray, roll_scaled: np.ndarray, event_label: np.ndarray, curve_multiplier: np.ndarray) -> None:
        n_samples = int(arrays["hist"].shape[0])
        n_steps = len(FUTURE_GRID)
        self.hist = arrays["hist"].astype(np.float32)
        self.road = arrays["road"].astype(np.float32)
        self.phase = arrays["phase"].astype(np.float32)
        self.point_seq = arrays["point"].reshape(n_samples, n_steps, -1).astype(np.float32)
        self.y_seq = arrays["y"].reshape(n_samples, n_steps).astype(np.float32)
        self.valid_seq = point_data.valid_original_remaining_all.reshape(n_samples, n_steps).astype(np.float32)
        base_weight = point_data.point_weight_all.reshape(n_samples, n_steps).astype(np.float32)
        self.weight_seq = base_weight * curve_multiplier.reshape(n_samples, 1).astype(np.float32)
        self.roll = roll_scaled.astype(np.float32)
        self.event_label = event_label.astype(np.int64)
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
            "roll": torch.from_numpy(self.roll[sample_idx]),
            "event_label": torch.tensor(self.event_label[sample_idx], dtype=torch.long),
            "y_seq": torch.from_numpy(self.y_seq[sample_idx]),
            "valid_seq": torch.from_numpy(self.valid_seq[sample_idx]),
            "weight_seq": torch.from_numpy(self.weight_seq[sample_idx]),
        }


class RollCauseAuxDecoder(nn.Module):
    """带 roll-cause 编码器、事件辅助头和 FiLM 调制的曲线解码器。"""

    def __init__(
        self,
        hist_dim: int,
        road_dim: int,
        phase_dim: int,
        point_dim: int,
        roll_dim: int,
        class_n: int,
        hist_len: int,
        road_len: int,
        n_steps: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        mixer_layers: int,
        mlp_hidden: int,
        roll_hidden: int,
        dropout: float,
        film_scale: float,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} 必须能被 n_heads={n_heads} 整除")
        self.film_scale = float(film_scale)
        self.hist_encoder = V241.TemporalConvEncoder(hist_dim, hidden_dim, hist_len, n_layers, dropout)
        self.road_encoder = V241.TemporalConvEncoder(road_dim, hidden_dim, road_len, max(1, n_layers - 1), dropout)
        self.roll_encoder = nn.Sequential(
            nn.Linear(roll_dim, roll_hidden),
            nn.GELU(),
            nn.LayerNorm(roll_hidden),
            nn.Dropout(dropout),
            nn.Linear(roll_hidden, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.event_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, class_n),
        )
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.query_input = nn.Sequential(
            nn.Linear(phase_dim + point_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.future_pos = nn.Parameter(torch.zeros(1, n_steps, hidden_dim))
        self.hist_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.road_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4 + phase_dim + point_dim, hidden_dim),
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

    def forward(self, hist: torch.Tensor, road: torch.Tensor, phase: torch.Tensor, point_seq: torch.Tensor, roll: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = point_seq.shape
        hist_tokens = self.hist_encoder(hist)
        road_tokens = self.road_encoder(road)
        roll_embed = self.roll_encoder(roll)
        event_logits = self.event_head(roll_embed)
        phase_rep = phase.unsqueeze(1).expand(batch, steps, phase.shape[-1])
        roll_rep = roll_embed.unsqueeze(1).expand(batch, steps, roll_embed.shape[-1])
        query = self.query_input(torch.cat([phase_rep, point_seq], dim=-1)) + self.future_pos[:, :steps, :]
        hist_ctx, _ = self.hist_attn(query, hist_tokens, hist_tokens, need_weights=False)
        road_ctx, _ = self.road_attn(query, road_tokens, road_tokens, need_weights=False)
        x = self.fuse(torch.cat([query, hist_ctx, road_ctx, roll_rep, phase_rep, point_seq], dim=-1))
        gamma, beta = self.film(roll_embed).chunk(2, dim=-1)
        x = x * (1.0 + self.film_scale * torch.tanh(gamma).unsqueeze(1)) + self.film_scale * torch.tanh(beta).unsqueeze(1)
        for mixer in self.mixers:
            x = mixer(x)
        curve = self.head(x).squeeze(-1)
        return curve, event_logits


def initialize_from_v300_backbone(model: RollCauseAuxDecoder, v300_model_name: str, hidden_dim: int) -> Dict[str, object]:
    """
    用 v300 checkpoint 初始化共同主干。

    v303 比 v300 多了 roll_rep，因此 fuse.0.weight 的输入维度多一个 hidden block。
    这里把 v300 的 query/hist/road/phase/point 权重复制过来，roll block 置零；
    同时把 FiLM 层置零，使初始化时曲线输出尽量等价于 v300，再让训练学习小幅调制。
    """

    ckpt_path = V300_MODELS / f"{v300_model_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"缺少 v300 checkpoint：{ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    old_state = ckpt["state_dict"]
    new_state = model.state_dict()
    copied = []
    skipped = []

    for key, value in old_state.items():
        if key == "fuse.0.weight":
            continue
        if key in new_state and tuple(new_state[key].shape) == tuple(value.shape):
            new_state[key] = value.detach().clone()
            copied.append(key)
        else:
            skipped.append(key)

    if "fuse.0.weight" in old_state and "fuse.0.weight" in new_state:
        old_w = old_state["fuse.0.weight"].detach().clone()
        new_w = new_state["fuse.0.weight"].detach().clone()
        new_w.zero_()
        h = int(hidden_dim)
        # v300: [query, hist_ctx, road_ctx, phase, point]
        # v303: [query, hist_ctx, road_ctx, roll_rep, phase, point]
        new_w[:, : 3 * h] = old_w[:, : 3 * h]
        new_w[:, 4 * h :] = old_w[:, 3 * h :]
        new_state["fuse.0.weight"] = new_w
        copied.append("fuse.0.weight_partial_with_zero_roll_block")

    if "film.weight" in new_state:
        new_state["film.weight"].zero_()
    if "film.bias" in new_state:
        new_state["film.bias"].zero_()
    model.load_state_dict(new_state)
    return {
        "init_from_v300": True,
        "v300_checkpoint": str(ckpt_path),
        "copied_key_n": len(copied),
        "skipped_key_n": len(skipped),
        "copied_keys_preview": copied[:20],
        "skipped_keys_preview": skipped[:20],
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    smooth_weight: float,
    aux_weight: float,
    class_weight: torch.Tensor,
) -> Dict[str, float]:
    """运行一个训练或验证 epoch。"""

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_curve = 0.0
    total_aux = 0.0
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
        pred, logits = model(hist, road, phase, point_seq, roll)
        curve_loss = V242.masked_curve_loss(pred, y_seq, valid_seq, weight_seq, smooth_weight)
        aux_loss = F.cross_entropy(logits, event_label, weight=class_weight)
        loss = curve_loss + float(aux_weight) * aux_loss
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
        wsum = float(torch.sum(valid_seq * weight_seq).detach().cpu().item())
        bsz = int(event_label.shape[0])
        total_loss += float(loss.detach().cpu().item()) * wsum
        total_curve += float(curve_loss.detach().cpu().item()) * wsum
        total_aux += float(aux_loss.detach().cpu().item()) * bsz
        total_weight += wsum
        total_samples += bsz
        correct += int((torch.argmax(logits, dim=1) == event_label).sum().detach().cpu().item())
    return {
        "loss": total_loss / max(total_weight, 1e-6),
        "curve_loss": total_curve / max(total_weight, 1e-6),
        "aux_loss": total_aux / max(total_samples, 1),
        "event_acc": correct / max(total_samples, 1),
    }


def predict_curves_and_logits(
    model: nn.Module,
    arrays: Dict[str, np.ndarray],
    roll_scaled: np.ndarray,
    scalers,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """预测全量 rolling 样本的曲线和事件 logits。"""

    model.eval()
    n_samples = int(arrays["hist"].shape[0])
    n_steps = len(FUTURE_GRID)
    point_seq = arrays["point"].reshape(n_samples, n_steps, -1).astype(np.float32)
    pred_scaled = np.empty((n_samples, n_steps), dtype=np.float32)
    logits_all = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            hist = torch.from_numpy(arrays["hist"][start:end]).to(device=device, dtype=torch.float32)
            road = torch.from_numpy(arrays["road"][start:end]).to(device=device, dtype=torch.float32)
            phase = torch.from_numpy(arrays["phase"][start:end]).to(device=device, dtype=torch.float32)
            points = torch.from_numpy(point_seq[start:end]).to(device=device, dtype=torch.float32)
            roll = torch.from_numpy(roll_scaled[start:end]).to(device=device, dtype=torch.float32)
            pred, logits = model(hist, road, phase, points, roll)
            pred_scaled[start:end] = pred.detach().cpu().numpy().astype(np.float32)
            logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
    pred_curve = (pred_scaled * scalers.y_std + scalers.y_mean).astype(np.float32)
    return pred_curve, np.concatenate(logits_all, axis=0).astype(np.float32)


def train_v303_candidate(
    model_name: str,
    config: Dict[str, object],
    prepared: RollPrepared,
    device: torch.device,
    seed: int,
) -> V303Run:
    """训练一个 v303 候选模型。"""

    set_seed(seed)
    sample_masks = prepared.prepared.sample_masks
    train_dataset = RollAuxCurveDataset(
        prepared.prepared.arrays,
        prepared.prepared.point_data,
        sample_masks["train"],
        prepared.roll_scaled,
        prepared.event_label,
        prepared.curve_sample_multiplier,
    )
    val_dataset = RollAuxCurveDataset(
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
    model = RollCauseAuxDecoder(
        hist_dim=prepared.data.x_hist.shape[-1],
        road_dim=prepared.data.x_road.shape[-1],
        phase_dim=prepared.data.x_phase.shape[-1],
        point_dim=len(V238.POINT_EXTRA_FEATURE_NAMES),
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
        dropout=float(config["dropout"]),
        film_scale=float(config["film_scale"]),
    )
    init_info: Dict[str, object] = {"init_from_v300": False}
    if bool(config.get("init_from_v300", False)):
        init_info = initialize_from_v300_backbone(model, str(config["v300_init_model_name"]), int(config["hidden_dim"]))
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
    max_epochs = int(config["max_epochs"])
    patience = int(config["patience"])
    smooth_weight = float(config["smooth_weight"])
    aux_weight = float(config["aux_weight"])

    best_val = math.inf
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history = []
    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        train_stat = run_epoch(model, train_loader, device, optimizer, smooth_weight, aux_weight, class_weight)
        val_stat = run_epoch(model, val_loader, device, None, smooth_weight, aux_weight, class_weight)
        scheduler.step(val_stat["loss"])
        lr_now = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "model_name": model_name,
                "epoch": epoch,
                "train_loss": train_stat["loss"],
                "train_curve_loss": train_stat["curve_loss"],
                "train_aux_loss": train_stat["aux_loss"],
                "train_event_acc": train_stat["event_acc"],
                "val_loss": val_stat["loss"],
                "val_curve_loss": val_stat["curve_loss"],
                "val_aux_loss": val_stat["aux_loss"],
                "val_event_acc": val_stat["event_acc"],
                "lr": lr_now,
                "init_from_v300": bool(init_info.get("init_from_v300", False)),
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
    pred_curve, logits = predict_curves_and_logits(
        model,
        prepared.prepared.arrays,
        prepared.roll_scaled,
        prepared.prepared.scalers,
        device,
        batch_size=batch_size * 4,
    )
    return V303Run(
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
    """从 delay0 group summary 里取一行。"""

    one = summary[
        summary["model_name"].astype(str).eq(model_name)
        & summary["split"].astype(str).eq(split)
        & summary["group"].astype(str).eq(group)
    ]
    if one.empty:
        return None
    return one.iloc[0]


def build_selection_from_metrics(metrics: pd.DataFrame, delay0_summary: pd.DataFrame, runs: List[V303Run], v300_name: str) -> pd.DataFrame:
    """正式 selection 表；只看 validation，不看 test。"""

    rows = []
    base_val_all = metric_row(delay0_summary, v300_name, "val", "all")
    base_val_bad10 = metric_row(delay0_summary, v300_name, "val", "within_bad_top10")
    base_val_bad20 = metric_row(delay0_summary, v300_name, "val", "within_bad_top20")
    for run in runs:
        score = V300.validation_score(metrics, run.model_name)
        row = {
            "model_name": run.model_name,
            "test_used_for_selection": False,
            "selected_by": "validation_original_remaining_plus_delay0_noharm",
            "best_epoch": int(run.best_epoch),
            "best_val_loss": float(run.best_val_loss),
            "training_seconds": float(run.training_seconds),
            "config_json": json.dumps(run.config, ensure_ascii=False, sort_keys=True),
        }
        row.update(score)
        cand_val_all = metric_row(delay0_summary, run.model_name, "val", "all")
        cand_val_bad10 = metric_row(delay0_summary, run.model_name, "val", "within_bad_top10")
        cand_val_bad20 = metric_row(delay0_summary, run.model_name, "val", "within_bad_top20")
        if base_val_all is not None and cand_val_all is not None:
            row["delay0_val_all_delta_vs_v300"] = float(cand_val_all["sample_rmse_mean"]) - float(base_val_all["sample_rmse_mean"])
        if base_val_bad10 is not None and cand_val_bad10 is not None:
            row["delay0_val_bad10_delta_vs_v300"] = float(cand_val_bad10["sample_rmse_mean"]) - float(base_val_bad10["sample_rmse_mean"])
        if base_val_bad20 is not None and cand_val_bad20 is not None:
            row["delay0_val_bad20_delta_vs_v300"] = float(cand_val_bad20["sample_rmse_mean"]) - float(base_val_bad20["sample_rmse_mean"])
        row["passes_val_all_noharm_vs_v300"] = bool(row.get("delay0_val_all_delta_vs_v300", math.inf) <= 0.005)
        row["passes_val_bad10_noharm_vs_v300"] = bool(row.get("delay0_val_bad10_delta_vs_v300", math.inf) <= 0.005)
        row["passes_val_bad20_noharm_vs_v300"] = bool(row.get("delay0_val_bad20_delta_vs_v300", math.inf) <= 0.005)
        row["passes_v303_noharm_gate"] = bool(
            row["passes_val_all_noharm_vs_v300"]
            and row["passes_val_bad10_noharm_vs_v300"]
            and row["passes_val_bad20_noharm_vs_v300"]
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if "validation_selection_score" in out.columns:
        out = out.sort_values(["passes_v303_noharm_gate", "validation_selection_score"], ascending=[False, True]).reset_index(drop=True)
    else:
        out = out.sort_values(["passes_v303_noharm_gate", "best_val_loss"], ascending=[False, True]).reset_index(drop=True)
    out["validation_rank"] = np.arange(1, len(out) + 1)
    return out


def build_event_aux_metrics(prepared: RollPrepared, runs: List[V303Run]) -> pd.DataFrame:
    """统计事件辅助头分类效果。"""

    split_values = prepared.data.manifest["split"].astype(str).to_numpy()
    delay0 = prepared.data.manifest["delay_ms"].astype(int).to_numpy() == 0
    rows = []
    for run in runs:
        pred = run.event_pred_class
        for split_name in ["train", "val", "test"]:
            for group_name, extra_mask in [("all_rolling", np.ones(len(split_values), dtype=bool)), ("delay0_only", delay0)]:
                mask = (split_values == split_name) & extra_mask
                if not mask.any():
                    continue
                y = prepared.event_label[mask]
                p = pred[mask]
                rows.append(
                    {
                        "model_name": run.model_name,
                        "split": split_name,
                        "group": group_name,
                        "n": int(mask.sum()),
                        "accuracy": float(accuracy_score(y, p)),
                        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
                        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
                        "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
                    }
                )
    return pd.DataFrame(rows)


def plot_training_history(runs: List[V303Run]) -> Path:
    """绘制训练曲线。"""

    path = FIGURES / "v303_training_history.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    for run in runs:
        hist = run.training_history
        ax.plot(hist["epoch"], hist["val_loss"], label=f"{run.model_name} val")
        ax.plot(hist["epoch"], hist["train_loss"], linestyle="--", alpha=0.45, label=f"{run.model_name} train")
    ax.set_title("v303 roll-cause 多任务模型训练曲线")
    ax.set_xlabel("epoch")
    ax.set_ylabel("curve + aux loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_delay0_group_bars(delay0_summary: pd.DataFrame, selected_name: str, v300_name: str) -> Path:
    """绘制 test delay0 关键组 RMSE。"""

    path = FIGURES / "v303_test_delay0_group_rmse.png"
    groups = ["all", "within_bad_top10", "within_bad_top20", "strong_steer", "vehicle_ambiguous"]
    rows = []
    for model_name in [v300_name, selected_name]:
        for group in groups:
            row = metric_row(delay0_summary, model_name, "test", group)
            if row is not None:
                rows.append({"model_name": model_name, "group": group, "rmse": float(row["sample_rmse_mean"])})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(groups))
    width = 0.36
    for offset, model_name in [(-width / 2, v300_name), (width / 2, selected_name)]:
        vals = []
        for group in groups:
            one = df[df["model_name"].eq(model_name) & df["group"].eq(group)]
            vals.append(float(one["rmse"].iloc[0]) if not one.empty else math.nan)
        ax.bar(x + offset, vals, width, label=model_name)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.set_ylabel("sample RMSE")
    ax.set_title("v303 vs v300：test delay0 分组 RMSE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_event_aux(event_metrics: pd.DataFrame, selected_name: str) -> Path:
    """绘制事件辅助头分类效果。"""

    path = FIGURES / "v303_event_aux_macro_f1.png"
    one = event_metrics[event_metrics["model_name"].eq(selected_name) & event_metrics["group"].eq("delay0_only")]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(one["split"], one["macro_f1"], color="#4c78a8")
    ax.set_ylim(0, max(0.5, float(one["macro_f1"].max()) * 1.15 if not one.empty else 0.5))
    ax.set_title(f"v303 事件辅助头 delay0 macro-F1：{selected_name}")
    ax.set_ylabel("macro-F1")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    selection: pd.DataFrame,
    delay0_summary: pd.DataFrame,
    event_metrics: pd.DataFrame,
    guardrail: Dict[str, object],
    selected_name: str,
    v300_name: str,
) -> Path:
    """写 v303 中文报告。"""

    path = REPORTS / "v303_roll_aux_multitask_curve_model_cn.md"

    def group_line(model_name: str, split: str, group: str) -> str:
        row = metric_row(delay0_summary, model_name, split, group)
        if row is None:
            return "NA"
        return f"{float(row['sample_rmse_mean']):.4f}"

    selected_rows = delay0_summary[
        delay0_summary["model_name"].isin([v300_name, selected_name])
        & delay0_summary["split"].eq("test")
        & delay0_summary["group"].isin(["all", "within_bad_top10", "within_bad_top20", "strong_steer", "vehicle_ambiguous"])
    ][["model_name", "split", "group", "n", "sample_rmse_mean", "sample_rmse_median", "sample_rmse_p90"]]

    selected_event = event_metrics[event_metrics["model_name"].eq(selected_name)]
    lines = [
        "# v303 roll-cause 辅助监督多任务曲线模型",
        "",
        "## 这一步做了什么",
        "",
        "v303 在 v300 的 joint-curve 输出任务上改模型结构：输出仍是 21 点 steering_delta 曲线，但新增 roll-cause summary 编码器，并使用 v301 事件类型作为训练期辅助监督。v301 标签不作为推理输入。",
        "",
        "结构上：历史/道路/phase/point token 仍走曲线解码主干；roll-cause summary 走单独 MLP 编码器；该编码一方面输出事件类型辅助头，另一方面通过 token 拼接和 FiLM 调制影响曲线隐藏表示。",
        "",
        "## validation-only 选择",
        "",
        selection.to_markdown(index=False),
        "",
        f"validation 选择出的 v303 候选：`{selected_name}`。",
        f"v300 参照模型：`{v300_name}`。",
        "",
        "## test delay0 对比",
        "",
        selected_rows.to_markdown(index=False),
        "",
        "简表：",
        "",
        f"- test/all：v300 `{group_line(v300_name, 'test', 'all')}` -> v303 `{group_line(selected_name, 'test', 'all')}`。",
        f"- test/within_bad_top10：v300 `{group_line(v300_name, 'test', 'within_bad_top10')}` -> v303 `{group_line(selected_name, 'test', 'within_bad_top10')}`。",
        f"- test/within_bad_top20：v300 `{group_line(v300_name, 'test', 'within_bad_top20')}` -> v303 `{group_line(selected_name, 'test', 'within_bad_top20')}`。",
        "",
        "## 事件辅助头",
        "",
        selected_event.to_markdown(index=False),
        "",
        "## 当前判断",
        "",
        "- 如果 v303 在事件辅助头上明显好于 v301/v302 的外部分类器，说明 roll-cause 分支确实学到了响应类型信息。",
        "- 是否接受 v303，不看 test 选择，而看 validation no-harm gate 和最终 test 分组报告。",
        "- 如果 test/all 改善但 bad_top10 变差，本轮只能说明结构有方向性，不算达成差样本本质改善。",
        "- 下一步若 v303 bad_top10 仍不够，应在该结构上加入 mixture-of-experts 或不确定性多模态输出，而不是回到删除样本/轻量 residual。",
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
    """打包并校验 v303 产物。"""

    zip_path = OUT / "v303_roll_aux_multitask_curve_model_20260703.zip"
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
    clean_out_dir()
    torch.set_num_threads(1)
    set_seed(SEED)

    print("[v303] 读取数据并构造 roll-cause 多任务输入")
    # hard_event_extra 是训练权重，不是输入。这里先设为 0，具体候选可覆盖。
    prepared_base = prepare_v303_data(hard_event_extra=0.0)
    split_audit = V300.build_split_audit(prepared_base.data.manifest, prepared_base.event_table)
    write_csv(split_audit, TABLES / "v303_within_subject_split_audit.csv")

    y_true_curve = prepared_base.data.y_future[:, :, 0].astype(np.float32)
    pred_v300, v300_name, v300_guard = load_v300_prediction_all(prepared_base.data.manifest)

    input_hashes = pd.DataFrame(
        [
            {"input_name": "v300_script", "path": str(V300_SCRIPT), "sha256": file_sha256(V300_SCRIPT)},
            {"input_name": "v302_script", "path": str(V302_SCRIPT), "sha256": file_sha256(V302_SCRIPT)},
            {"input_name": "v301_labels", "path": str(V301_LABELS), "sha256": file_sha256(V301_LABELS)},
            {"input_name": "v300_predictions", "path": str(V300_PRED), "sha256": file_sha256(V300_PRED)},
            {
                "input_name": "v300_guardrail",
                "path": str(V300_GUARDRAIL),
                "sha256": file_sha256(V300_GUARDRAIL) if V300_GUARDRAIL.exists() else "",
            },
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v303] 使用设备：{device}")

    configs: List[Tuple[str, Dict[str, object]]] = [
        (
            "v303_roll_init_aux003_film005_h64",
            {
                "hidden_dim": 64,
                "n_heads": 4,
                "n_layers": 3,
                "mixer_layers": 2,
                "mlp_hidden": 112,
                "roll_hidden": 128,
                "dropout": 0.06,
                "film_scale": 0.05,
                "smooth_weight": 0.02,
                "aux_weight": 0.03,
                "hard_event_extra": 0.0,
                "init_from_v300": True,
                "v300_init_model_name": v300_name,
                "lr": 2e-4,
                "min_lr": 1e-5,
                "weight_decay": 3e-4,
                "batch_size": 384,
                "max_epochs": 55,
                "patience": 9,
            },
        ),
        (
            "v303_roll_init_aux005_film010_h64",
            {
                "hidden_dim": 64,
                "n_heads": 4,
                "n_layers": 3,
                "mixer_layers": 2,
                "mlp_hidden": 128,
                "roll_hidden": 160,
                "dropout": 0.08,
                "film_scale": 0.10,
                "smooth_weight": 0.025,
                "aux_weight": 0.05,
                "hard_event_extra": 0.0,
                "init_from_v300": True,
                "v300_init_model_name": v300_name,
                "lr": 2e-4,
                "min_lr": 1e-5,
                "weight_decay": 4e-4,
                "batch_size": 384,
                "max_epochs": 60,
                "patience": 10,
            },
        ),
        (
            "v303_roll_init_aux006_film010_hard110_h64",
            {
                "hidden_dim": 64,
                "n_heads": 4,
                "n_layers": 3,
                "mixer_layers": 2,
                "mlp_hidden": 128,
                "roll_hidden": 160,
                "dropout": 0.08,
                "film_scale": 0.10,
                "smooth_weight": 0.025,
                "aux_weight": 0.06,
                "hard_event_extra": 0.10,
                "init_from_v300": True,
                "v300_init_model_name": v300_name,
                "lr": 2e-4,
                "min_lr": 1e-5,
                "weight_decay": 4e-4,
                "batch_size": 384,
                "max_epochs": 60,
                "patience": 10,
            },
        ),
        (
            "v303_roll_scratch_aux010_hard125_h64",
            {
                "hidden_dim": 64,
                "n_heads": 4,
                "n_layers": 3,
                "mixer_layers": 2,
                "mlp_hidden": 128,
                "roll_hidden": 160,
                "dropout": 0.10,
                "film_scale": 0.20,
                "smooth_weight": 0.03,
                "aux_weight": 0.10,
                "hard_event_extra": 0.25,
                "init_from_v300": False,
                "lr": 5.5e-4,
                "min_lr": 1e-5,
                "weight_decay": 5e-4,
                "batch_size": 320,
                "max_epochs": 90,
                "patience": 14,
            },
        ),
    ]

    runs: List[V303Run] = []
    for idx, (model_name, config) in enumerate(configs):
        prepared = copy.copy(prepared_base)
        prepared.curve_sample_multiplier = build_curve_sample_multiplier(
            prepared_base.data.manifest,
            prepared_base.event_label_name,
            hard_event_extra=float(config["hard_event_extra"]),
        )
        print(
            f"[v303] training {model_name} | aux={config['aux_weight']} | film={config['film_scale']} | hard_extra={config['hard_event_extra']}"
        )
        run = train_v303_candidate(model_name, config, prepared, device, seed=SEED + idx)
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
            },
            MODELS / f"{model_name}.pt",
        )
        print(f"[v303] {model_name} best_epoch={run.best_epoch} best_val_loss={run.best_val_loss:.6f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[v303] 计算指标与 validation-only 选择")
    pred_by_model: Dict[str, np.ndarray] = {v300_name: pred_v300.astype(np.float32)}
    for run in runs:
        pred_by_model[run.model_name] = run.pred_curve.astype(np.float32)

    metrics = V238.compute_metrics_table(
        y_true_curve=y_true_curve,
        pred_by_model=pred_by_model,
        manifest=prepared_base.data.manifest,
        eval_modes=["original_remaining", "receding_2s_diagnostic"],
    )
    write_csv(metrics, TABLES / "v303_metrics_by_delay_and_bucket.csv")

    per_tables = []
    for model_name, pred_curve in pred_by_model.items():
        per = V238.build_per_sample_metrics(
            y_true_curve=y_true_curve,
            pred_curve=pred_curve,
            manifest=prepared_base.data.manifest,
            model_name=model_name,
        )
        per_tables.append(per)
    per_sample = pd.concat(per_tables, ignore_index=True)
    per_sample = V300.attach_v299_event_labels(per_sample, prepared_base.event_table)
    write_csv(per_sample, TABLES / "v303_per_sample_metrics_original_remaining.csv")

    delay0_summary = V300.build_delay0_group_summary(per_sample)
    write_csv(delay0_summary, TABLES / "v303_delay0_group_summary.csv")

    selection = build_selection_from_metrics(metrics, delay0_summary, runs, v300_name)
    write_csv(selection, TABLES / "v303_model_selection_validation.csv")
    selected_name = str(selection.iloc[0]["model_name"])

    event_metrics = build_event_aux_metrics(prepared_base, runs)
    write_csv(event_metrics, TABLES / "v303_event_aux_metrics.csv")

    print("[v303] 保存预测数组和图像")
    original_remaining_valid, _ = V238.build_original_remaining_mask(prepared_base.data.manifest)
    npz_payload = {
        "y_true_steering_delta": y_true_curve.astype(np.float32),
        "pred_v300_reference": pred_v300.astype(np.float32),
        "v300_reference_model": np.array([v300_name]),
        "pred_v303_selected": pred_by_model[selected_name].astype(np.float32),
        "best_v303_model": np.array([selected_name]),
        "delay_ms": prepared_base.data.manifest["delay_ms"].astype(int).to_numpy(dtype=np.int32),
        "split": prepared_base.data.manifest["split"].astype(str).to_numpy(),
        "event_uid": prepared_base.data.manifest["event_uid"].astype(str).to_numpy(),
        "subject": prepared_base.data.manifest["subject"].astype(str).to_numpy(),
        "future_grid_s": FUTURE_GRID.astype(np.float32),
        "original_remaining_valid": original_remaining_valid.astype(bool),
        "event_primary_type": prepared_base.event_label_name.astype(str),
        "event_class_index": prepared_base.event_label.astype(np.int64),
        "event_class_names": np.array(prepared_base.class_names),
    }
    for run in runs:
        npz_payload[f"pred_{run.model_name}"] = run.pred_curve.astype(np.float32)
        npz_payload[f"event_logits_{run.model_name}"] = run.event_logits.astype(np.float32)
    np.savez_compressed(OUT / "v303_roll_aux_multitask_predictions.npz", **npz_payload)

    with (MODELS / "v303_scalers_and_selection.pkl").open("wb") as f:
        pickle.dump(
            {
                "selection": selection.to_dict(orient="records"),
                "selected_name": selected_name,
                "v300_reference_model": v300_name,
                "roll_feature_names": prepared_base.roll_feature_names,
                "roll_impute_mean": prepared_base.roll_impute_mean,
                "roll_scale_mean": prepared_base.roll_scale_mean,
                "roll_scale_std": prepared_base.roll_scale_std,
                "class_names": prepared_base.class_names,
                "v300_guardrail": v300_guard,
            },
            f,
        )

    figure_paths = [
        plot_training_history(runs),
        plot_delay0_group_bars(delay0_summary, selected_name, v300_name),
        plot_event_aux(event_metrics, selected_name),
    ]

    event_split_n = prepared_base.data.manifest.groupby("event_uid")["split"].nunique()
    event_delay_n = prepared_base.data.manifest.groupby("event_uid")["delay_ms"].nunique()
    selected_row = selection.iloc[0].to_dict()
    guardrail = {
        "pass": bool((event_split_n <= 1).all() and (event_delay_n == 6).all()),
        "version": "v303_roll_aux_multitask_curve_model_20260703",
        "model_structure_changed": True,
        "output_target_unchanged": "21_point_steering_delta_curve",
        "uses_roll_cause_summary_as_input": True,
        "uses_future_event_labels_as_features": False,
        "uses_event_labels_as_auxiliary_targets": True,
        "uses_test_error_as_features": False,
        "candidate_selection_uses_test": False,
        "same_event_never_repeated_across_splits": bool((event_split_n <= 1).all()),
        "event_in_multiple_splits_n": int((event_split_n > 1).sum()),
        "event_without_6_delay_rows_n": int((event_delay_n != 6).sum()),
        "event_n": int(prepared_base.data.manifest["event_uid"].nunique()),
        "rolling_sample_n": int(len(prepared_base.data.manifest)),
        "roll_cause_feature_n": int(prepared_base.roll_scaled.shape[1]),
        "event_class_n": int(len(prepared_base.class_names)),
        "v300_reference_model": v300_name,
        "selected_v303_model": selected_name,
        "selected_passes_v303_noharm_gate": bool(selected_row.get("passes_v303_noharm_gate", False)),
        "selected_val_all_delta_vs_v300": float(selected_row.get("delay0_val_all_delta_vs_v300", math.nan)),
        "selected_val_bad10_delta_vs_v300": float(selected_row.get("delay0_val_bad10_delta_vs_v300", math.nan)),
        "selected_val_bad20_delta_vs_v300": float(selected_row.get("delay0_val_bad20_delta_vs_v300", math.nan)),
        "device": str(device),
        "runtime_seconds": float(time.time() - start_time),
        "figure_paths": [str(p) for p in figure_paths],
    }
    write_json(guardrail, LOGS / "guardrail_check.json")
    report_path = write_report(selection, delay0_summary, event_metrics, guardrail, selected_name, v300_name)
    guardrail["report_path"] = str(report_path)
    write_json(guardrail, LOGS / "guardrail_check.json")

    write_file_inventory()
    zip_path, zip_ok = make_zip_package()
    guardrail["zip_path"] = str(zip_path)
    guardrail["zip_testzip"] = bool(zip_ok)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()

    print("[v303] 完成")
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
