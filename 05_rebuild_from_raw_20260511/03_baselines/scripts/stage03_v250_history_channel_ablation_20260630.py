#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v250 history-channel ablation。

本轮目标：
- 回答用户提出的问题：当前 18 个车辆历史通道是否过多、是否干扰模型；
- 只做“历史通道精简”，不缩短历史长度，仍使用 -3.0s~0.0s 的 31 个时间点；
- 继承 v241 的 TCN + multi-head query attention 架构和 original_remaining point-level target；
- 每个精简通道组从头训练同一结构，不能加载 v241 checkpoint，因为 hist_dim 已改变；
- 只用 validation 排名候选，test 只做 locked report；
- 不做 anchor selector、gate/router、response-type hard routing，不删除样本，不用 oracle best anchor。

候选通道组：
- drop_attitude_noise13：轻度删噪，去掉 pitch/yaw/pitch_rate/roll_acc/pitch_acc；
- lateral_core10：中度精简，只保留方向盘、速度、横向动力学、油门刹车、道路/横向位置；
- minimal_lateral7：极简横向核心，只保留 steering/speed/ay/yaw_rate/roll/lane_curvature/lateral_distance。
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


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V249_SCRIPT = BASELINES / "scripts" / "stage03_v249_shape_aware_curve_model_20260630.py"
V241_DIR = BASELINES / "v241_stronger_temporal_model_20260626"
V241_PRED = V241_DIR / "v241_stronger_temporal_predictions.npz"
V241_MODEL = V241_DIR / "models" / "v241_best_stronger_temporal_diagnostic.pt"

OUT = BASELINES / "v250_history_channel_ablation_20260630"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
MODELS = OUT / "models"
ZIP_PATH = BASELINES / "v250_history_channel_ablation_20260630_pack.zip"

DELAY_MS = [0, 200, 400, 600, 800, 1000]
FORMAL_DELAY_MAX_MS = 800
STRONG_DELAY_MAX_MS = 600
UPGRADE_TOL = 0.02
SEED = 250
K_NEIGHBORS = 10

HIST_CHANNELS = [
    "steering",
    "speed_kmh",
    "vx",
    "vy",
    "ax",
    "ay",
    "yaw_rate",
    "roll",
    "pitch",
    "yaw",
    "roll_rate",
    "pitch_rate",
    "roll_acc",
    "pitch_acc",
    "accelerator",
    "brake",
    "lane_curvature",
    "lateral_distance",
]

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    """按路径导入已经验证过的前序脚本，复用数据、模型和指标函数。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V249 = import_module_from_path("stage03_v249_shape_aware_curve_model_20260630_for_v250", V249_SCRIPT)
V241 = V249.V241
V239 = V249.V239
V238 = V249.V238
FUTURE_GRID = V238.FUTURE_GRID.astype(np.float32)


@dataclass
class ChannelGroup:
    """一个历史通道精简候选。"""

    model_name: str
    description: str
    channels: List[str]


@dataclass
class ChannelRun:
    """一个通道组训练后的产物。"""

    group: ChannelGroup
    reduced_data: object
    point_data: object
    arrays: Dict[str, np.ndarray]
    scalers: object
    point_masks: Dict[str, np.ndarray]
    run: object
    validation_row: Dict[str, object] | None = None


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v250 自己的输出目录。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 统一使用 utf-8-sig，便于 Windows Excel 打开中文。"""

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


def load_v241_prediction() -> Tuple[np.ndarray, str]:
    """读取 locked v241 预测。"""

    if not V241_PRED.exists():
        raise FileNotFoundError(f"缺少 v241 prediction：{V241_PRED}")
    with np.load(V241_PRED, allow_pickle=False) as pred:
        arr = pred["pred_v241_best_stronger_steering_delta"].astype(np.float32)
        name = str(pred["best_stronger_model"][0])
    return arr, name


def channel_groups() -> List[ChannelGroup]:
    """定义三档通道精简候选。"""

    return [
        ChannelGroup(
            model_name="v250_drop_attitude_noise13",
            description="轻度删噪：去掉 pitch/yaw/pitch_rate/roll_acc/pitch_acc，保留车辆平动、横向动态、控制量和道路/横向位置。",
            channels=[
                "steering",
                "speed_kmh",
                "vx",
                "vy",
                "ax",
                "ay",
                "yaw_rate",
                "roll",
                "roll_rate",
                "accelerator",
                "brake",
                "lane_curvature",
                "lateral_distance",
            ],
        ),
        ChannelGroup(
            model_name="v250_lateral_core10",
            description="中度精简：保留方向盘、速度、横向动力学、控制量和道路/横向位置。",
            channels=[
                "steering",
                "speed_kmh",
                "ay",
                "yaw_rate",
                "roll",
                "roll_rate",
                "accelerator",
                "brake",
                "lane_curvature",
                "lateral_distance",
            ],
        ),
        ChannelGroup(
            model_name="v250_minimal_lateral7",
            description="极简横向核心：只保留最直接的方向盘/速度/横向响应/道路位置通道。",
            channels=[
                "steering",
                "speed_kmh",
                "ay",
                "yaw_rate",
                "roll",
                "lane_curvature",
                "lateral_distance",
            ],
        ),
    ]


def channel_indices(channels: Iterable[str]) -> List[int]:
    """把通道名映射为原 18 通道索引。"""

    idx: List[int] = []
    for name in channels:
        if name not in HIST_CHANNELS:
            raise ValueError(f"未知历史通道：{name}")
        idx.append(HIST_CHANNELS.index(name))
    return idx


def reduced_feature_names(data, idx: List[int]) -> List[str]:
    """构造精简历史通道后的 feature_names，保持 road/phase 名称不变。"""

    hist_len = int(data.x_hist.shape[1])
    hist_dim = int(data.x_hist.shape[2])
    road_dim_total = int(data.x_road.shape[1] * data.x_road.shape[2])
    phase_dim = int(data.x_phase.shape[1])
    if hist_dim != len(HIST_CHANNELS):
        raise AssertionError(f"当前脚本假定原始历史通道为 18，实际为 {hist_dim}")

    names: List[str] = []
    for t in range(hist_len):
        base = t * hist_dim
        for j in idx:
            names.append(str(data.feature_names[base + j]))
    road_start = hist_len * hist_dim
    road_end = road_start + road_dim_total
    names.extend([str(x) for x in data.feature_names[road_start:road_end]])
    names.extend([str(x) for x in data.feature_names[-phase_dim:]])
    return names


def make_reduced_data(data, group: ChannelGroup):
    """
    只裁剪 X_hist 的通道维度，其他输入、目标和 manifest 不变。

    注意：这里不缩短历史长度，仍然保留 31 个时间点；这保证本轮只回答“通道是否过多”。
    """

    idx = channel_indices(group.channels)
    feature_names = reduced_feature_names(data, idx)
    x_hist = data.x_hist[:, :, idx].astype(np.float32)
    expected_n = x_hist.shape[1] * x_hist.shape[2] + data.x_road.shape[1] * data.x_road.shape[2] + data.x_phase.shape[1]
    if len(feature_names) != expected_n:
        raise AssertionError(f"{group.model_name} feature_names 数量不一致：{len(feature_names)} vs {expected_n}")

    return V238.RollingData(
        manifest=data.manifest.copy(),
        x_hist=x_hist,
        x_road=data.x_road.astype(np.float32),
        x_phase=data.x_phase.astype(np.float32),
        y_future=data.y_future.astype(np.float32),
        pred_v236=data.pred_v236.astype(np.float32),
        feature_names=feature_names,
        target_names=list(data.target_names),
    )


def train_config() -> Dict[str, object]:
    """本轮固定使用 v241 h96 配置的轻微缩短版，避免把通道消融变成大规模调参。"""

    return {
        "hidden_dim": 96,
        "n_heads": 4,
        "n_layers": 4,
        "mlp_hidden": 160,
        "dropout": 0.10,
        "lr": 5e-4,
        "min_lr": 1e-5,
        "weight_decay": 5e-4,
        "batch_size": 1024,
        "max_epochs": 70,
        "patience": 9,
    }


def train_one_group(base_data, group: ChannelGroup, device: torch.device) -> ChannelRun:
    """训练一个精简通道候选。"""

    reduced = make_reduced_data(base_data, group)
    x_base = V238.build_base_design_matrix(reduced)
    point_data = V238.build_point_dataset(reduced, x_base)
    point_masks = V238.split_point_masks(point_data, reduced.manifest)
    scalers = V239.fit_scalers(reduced, point_data, point_masks)
    arrays = V239.standardize_arrays(reduced, point_data, scalers)
    run = V241.train_stronger_candidate(
        group.model_name,
        train_config(),
        reduced,
        point_data,
        arrays,
        scalers,
        point_masks,
        device,
    )
    return ChannelRun(
        group=group,
        reduced_data=reduced,
        point_data=point_data,
        arrays=arrays,
        scalers=scalers,
        point_masks=point_masks,
        run=run,
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


def positive_penalty(value: float, threshold: float = 0.0) -> float:
    """只惩罚超过阈值的正向回退。"""

    if not np.isfinite(value):
        return 10.0
    return max(0.0, float(value) - threshold)


def delta_frame(metrics: pd.DataFrame, candidate_name: str, ref_name: str, split: str = "val") -> pd.DataFrame:
    """在 original_remaining 下生成 candidate-ref 的 bucket/delay delta。"""

    sub = metrics[metrics["split"].eq(split) & metrics["eval_mode"].eq("original_remaining")].copy()
    cand = sub[sub["model_name"].eq(candidate_name)].copy()
    ref = sub[sub["model_name"].eq(ref_name)].copy()
    if cand.empty or ref.empty:
        raise AssertionError(f"{candidate_name} 或 {ref_name} 在 {split} 指标为空")
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
    """抽取指定 bucket/delay 范围。"""

    out = merged[merged["bucket"].eq(bucket)].copy()
    if max_delay is not None:
        out = out[out["delay_ms"].astype(int) <= int(max_delay)].copy()
    if delays is not None:
        wanted = {int(x) for x in delays}
        out = out[out["delay_ms"].astype(int).isin(wanted)].copy()
    return out


def validation_decision(
    metrics: pd.DataFrame,
    shape_table: pd.DataFrame,
    channel_run: ChannelRun,
    v241_name: str,
) -> Dict[str, object]:
    """
    validation-only 判断通道精简是否值得保留。

    这里不要求精简模型一定全面超过 v241；但至少不能明显伤 normal/observe/strong，
    并且应在 bad_top10 或 strong shape 上给出可解释收益，否则只记为诊断失败。
    """

    name = channel_run.group.model_name
    vs_v241 = delta_frame(metrics, name, v241_name, split="val")
    normal = subset_delta(vs_v241, "normal_predictable", max_delay=FORMAL_DELAY_MAX_MS)
    all_bucket = subset_delta(vs_v241, "all", max_delay=FORMAL_DELAY_MAX_MS)
    observe = subset_delta(vs_v241, "observe_later_like", max_delay=FORMAL_DELAY_MAX_MS)
    strong = subset_delta(vs_v241, "strong_steer", max_delay=STRONG_DELAY_MAX_MS)

    normal_max_tail_delta = finite_max(normal["delta_tail"])
    all_mean_tail_delta = finite_mean(all_bucket["delta_tail"])
    observe_mean_tail_delta = finite_mean(observe["delta_tail"])
    strong_mean_tail_delta = finite_mean(strong["delta_tail"])

    shape_val = shape_table[shape_table["split"].eq("val") & shape_table["model_name"].eq(name)].copy()
    bad = shape_val[shape_val["event_group"].eq("bad_top10_v241")]
    strong_shape = shape_val[shape_val["event_group"].eq("strong_steer")]
    normal_shape = shape_val[shape_val["event_group"].eq("normal")]

    bad_rmse_delta = finite_mean(bad["delta_rmse_candidate_minus_v241"], default=0.0)
    bad_range_gain = finite_mean(bad["delta_range_ratio_candidate_minus_v241"], default=0.0)
    bad_slope_gain = finite_mean(bad["delta_slope_ratio_candidate_minus_v241"], default=0.0)
    strong_range_gain = finite_mean(strong_shape["delta_range_ratio_candidate_minus_v241"], default=0.0)
    strong_slope_gain = finite_mean(strong_shape["delta_slope_ratio_candidate_minus_v241"], default=0.0)
    normal_rmse_delta = finite_mean(normal_shape["delta_rmse_candidate_minus_v241"], default=0.0)

    no_major_harm = (
        normal_max_tail_delta <= UPGRADE_TOL
        and all_mean_tail_delta <= UPGRADE_TOL
        and observe_mean_tail_delta <= UPGRADE_TOL
        and strong_mean_tail_delta <= UPGRADE_TOL
    )
    hard_gain = bad_rmse_delta <= -0.02 or (bad_range_gain + bad_slope_gain + strong_range_gain + strong_slope_gain) > 0.05
    accepted = bool(no_major_harm and hard_gain)

    val_all = metrics[
        metrics["split"].eq("val")
        & metrics["eval_mode"].eq("original_remaining")
        & metrics["bucket"].eq("all")
        & metrics["model_name"].eq(name)
    ].copy()
    base_score = finite_mean(val_all["steer_sample_rmse_mean"]) + 0.5 * finite_mean(val_all["steer_tail_rmse_mean"])
    harm_penalty = (
        positive_penalty(normal_max_tail_delta, UPGRADE_TOL)
        + positive_penalty(all_mean_tail_delta, UPGRADE_TOL)
        + positive_penalty(observe_mean_tail_delta, UPGRADE_TOL)
        + positive_penalty(strong_mean_tail_delta, UPGRADE_TOL)
        + positive_penalty(normal_rmse_delta, UPGRADE_TOL)
    )
    hard_reward = max(0.0, -bad_rmse_delta) + 0.20 * (
        max(0.0, bad_range_gain)
        + max(0.0, bad_slope_gain)
        + max(0.0, strong_range_gain)
        + max(0.0, strong_slope_gain)
    )
    selection_score = base_score + 6.0 * harm_penalty - 0.25 * hard_reward

    return {
        "model_name": name,
        "selected_by": "validation_only_channel_ablation",
        "test_used_for_selection": False,
        "n_hist_channels": len(channel_run.group.channels),
        "channels": "|".join(channel_run.group.channels),
        "description": channel_run.group.description,
        "normal_max_tail_delta_vs_v241_0to800": normal_max_tail_delta,
        "all_mean_tail_delta_vs_v241_0to800": all_mean_tail_delta,
        "observe_later_mean_tail_delta_vs_v241_0to800": observe_mean_tail_delta,
        "strong_mean_tail_delta_vs_v241_0to600": strong_mean_tail_delta,
        "val_bad_top10_rmse_delta_vs_v241": bad_rmse_delta,
        "val_bad_top10_range_ratio_gain_vs_v241": bad_range_gain,
        "val_bad_top10_slope_ratio_gain_vs_v241": bad_slope_gain,
        "val_strong_range_ratio_gain_vs_v241": strong_range_gain,
        "val_strong_slope_ratio_gain_vs_v241": strong_slope_gain,
        "val_normal_shape_rmse_delta_vs_v241": normal_rmse_delta,
        "no_major_harm_vs_v241_pass": bool(no_major_harm),
        "hard_gain_pass": bool(hard_gain),
        "accepted_as_channel_candidate": accepted,
        "validation_selection_score": float(selection_score),
        "config_json": json.dumps(channel_run.run.config, ensure_ascii=False, sort_keys=True),
        "best_epoch": int(channel_run.run.best_epoch),
        "best_val_loss": float(channel_run.run.best_val_loss),
        "training_seconds": float(channel_run.run.training_seconds),
    }


def build_compare_table(metrics: pd.DataFrame, candidate_names: List[str], v241_name: str) -> pd.DataFrame:
    """生成 test original_remaining 对照宽表。"""

    keep_buckets = ["all", "normal_predictable", "observe_later_like", "strong_steer", "reverse_or_multi_correction"]
    sub = metrics[
        metrics["split"].eq("test")
        & metrics["eval_mode"].eq("original_remaining")
        & metrics["bucket"].isin(keep_buckets)
    ].copy()
    values = [
        "steer_sample_rmse_mean",
        "steer_tail_rmse_mean",
        "peak_ratio_mean",
        "strong_under_rate",
    ]
    wide = sub.pivot_table(index=["bucket", "delay_ms"], columns="model_name", values=values, aggfunc="first")
    wide.columns = [f"{metric}__{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    for name in candidate_names:
        for metric in ["steer_sample_rmse_mean", "steer_tail_rmse_mean"]:
            c = f"{metric}__{name}"
            r = f"{metric}__{v241_name}"
            if c in wide.columns and r in wide.columns:
                wide[f"delta_{metric}__{name}_minus_v241"] = wide[c] - wide[r]
    return wide


def sample_rmse_for_v241(y_true: np.ndarray, pred_v241: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """计算每个 rolling 样本在 original_remaining 上的 v241 RMSE。"""

    out = np.full(len(y_true), np.nan, dtype=float)
    for i in range(len(y_true)):
        valid = valid_mask[i]
        if np.any(valid):
            out[i] = math.sqrt(float(np.mean(np.square(pred_v241[i, valid] - y_true[i, valid]))))
    return out


def pairwise_rmse(curves: np.ndarray) -> float:
    """邻居未来曲线的两两 RMSE 均值。"""

    if len(curves) < 2:
        return math.nan
    vals: List[float] = []
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            vals.append(math.sqrt(float(np.mean(np.square(curves[i] - curves[j])))))
    return float(np.mean(vals)) if vals else math.nan


def flatten_sample_input(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    """把标准化后的 hist/road/phase 拼成 sample-level 输入，用于邻域审计。"""

    n = arrays["hist"].shape[0]
    return np.concatenate(
        [
            arrays["hist"].reshape(n, -1),
            arrays["road"].reshape(n, -1),
            arrays["phase"].reshape(n, -1),
        ],
        axis=1,
    ).astype(np.float32)


def build_neighbor_ambiguity_table(
    channel_runs: List[ChannelRun],
    y_true: np.ndarray,
    pred_v241: np.ndarray,
    manifest: pd.DataFrame,
    valid_mask: np.ndarray,
) -> pd.DataFrame:
    """
    对每个通道组重新做输入邻域审计。

    如果精简通道确实去掉噪声，理论上 hard sample 的近邻未来分歧应下降；
    如果分歧不降，则说明主要不是“通道太多”，而是锚点前可见信息本身不足。
    """

    rows: List[Dict[str, object]] = []
    split = manifest["split"].astype(str).to_numpy()
    delay = manifest["delay_ms"].astype(int).to_numpy()
    v241_rmse = sample_rmse_for_v241(y_true, pred_v241, valid_mask)
    query_pool = (split == "test") & (delay == 0) & np.isfinite(v241_rmse)
    threshold = float(np.quantile(v241_rmse[query_pool], 0.90))
    query_indices = np.where(query_pool & (v241_rmse >= threshold))[0]

    train_pool = np.where((split == "train") & (delay == 0))[0]
    for channel_run in channel_runs:
        x = flatten_sample_input(channel_run.arrays)
        x_train = x[train_pool]
        for sample_idx in query_indices:
            diff = x_train - x[sample_idx][None, :]
            dist = np.sqrt(np.mean(np.square(diff), axis=1))
            order = np.argsort(dist)[: min(K_NEIGHBORS, len(dist))]
            neigh_idx = train_pool[order]
            curves = y_true[neigh_idx][:, valid_mask[sample_idx]]
            query_curve = y_true[sample_idx, valid_mask[sample_idx]]
            if curves.size == 0 or query_curve.size == 0:
                continue
            best_to_query = np.min(np.sqrt(np.mean(np.square(curves - query_curve[None, :]), axis=1)))
            mean_to_query = np.mean(np.sqrt(np.mean(np.square(curves - query_curve[None, :]), axis=1)))
            neighbor_pairwise = pairwise_rmse(curves)
            neighbor_peak_std = float(np.std(np.max(np.abs(curves), axis=1)))
            neighbor_slope_std = float(np.std(np.max(np.abs(np.diff(curves, axis=1)), axis=1))) if curves.shape[1] >= 2 else math.nan
            ambiguous = bool(neighbor_pairwise > 0.50 or neighbor_peak_std > 0.25)
            rows.append(
                {
                    "model_name": channel_run.group.model_name,
                    "n_hist_channels": len(channel_run.group.channels),
                    "rolling_sample_index": int(manifest.iloc[sample_idx]["rolling_sample_index"]),
                    "event_uid": str(manifest.iloc[sample_idx]["event_uid"]),
                    "subject": str(manifest.iloc[sample_idx]["subject"]),
                    "delay_ms": int(delay[sample_idx]),
                    "v241_rmse": float(v241_rmse[sample_idx]),
                    "neighbor_k": int(len(neigh_idx)),
                    "neighbor_input_distance_mean": float(np.mean(dist[order])),
                    "neighbor_input_distance_min": float(np.min(dist[order])),
                    "neighbor_future_pairwise_rmse_mean": float(neighbor_pairwise),
                    "neighbor_peak_abs_std": neighbor_peak_std,
                    "neighbor_slope_abs_std": neighbor_slope_std,
                    "query_vs_neighbor_best_rmse": float(best_to_query),
                    "query_vs_neighbor_mean_rmse": float(mean_to_query),
                    "ambiguity_category": "input_ambiguous" if ambiguous else "less_ambiguous_neighbors",
                    "neighbor_event_uids": "|".join(manifest.iloc[neigh_idx]["event_uid"].astype(str).tolist()),
                }
            )
    return pd.DataFrame(rows)


def ambiguity_summary(ambiguity: pd.DataFrame) -> pd.DataFrame:
    """汇总每个通道组的邻域歧义程度。"""

    if ambiguity.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for name, g in ambiguity.groupby("model_name"):
        rows.append(
            {
                "model_name": name,
                "n_cases": int(len(g)),
                "input_ambiguous_rate": float(g["ambiguity_category"].eq("input_ambiguous").mean()),
                "neighbor_future_pairwise_rmse_mean": float(g["neighbor_future_pairwise_rmse_mean"].mean()),
                "neighbor_peak_abs_std_mean": float(g["neighbor_peak_abs_std"].mean()),
                "neighbor_slope_abs_std_mean": float(g["neighbor_slope_abs_std"].mean()),
                "query_vs_neighbor_best_rmse_mean": float(g["query_vs_neighbor_best_rmse"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("neighbor_future_pairwise_rmse_mean").reset_index(drop=True)


def plot_tail_delta(compare: pd.DataFrame, candidate_names: List[str], best_name: str) -> Path:
    """绘制关键 bucket 的 tail RMSE delta。"""

    keep = compare[
        compare["bucket"].isin(["normal_predictable", "observe_later_like", "strong_steer"])
        & compare["delay_ms"].isin([0, 400, 800, 1000])
    ].copy()
    rows: List[Dict[str, object]] = []
    for _, row in keep.iterrows():
        label = f"{row['bucket']}\n{int(row['delay_ms'])}ms"
        for name in candidate_names:
            col = f"delta_steer_tail_rmse_mean__{name}_minus_v241"
            if col in row.index and np.isfinite(row[col]):
                rows.append({"label": label, "model_name": name, "delta_tail": float(row[col])})
    df = pd.DataFrame(rows)
    path = FIGURES / "v250_tail_delta_by_channel_group.png"
    if df.empty:
        return path

    labels = list(dict.fromkeys(df["label"].tolist()))
    x = np.arange(len(labels))
    width = 0.80 / max(1, len(candidate_names))
    fig, ax = plt.subplots(figsize=(15, 6))
    for i, name in enumerate(candidate_names):
        sub = df[df["model_name"].eq(name)].set_index("label").reindex(labels)
        ax.bar(x + (i - (len(candidate_names) - 1) / 2) * width, sub["delta_tail"], width=width, label=name)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(f"v250 历史通道精简：test tail RMSE delta vs v241（best={best_name}）")
    ax.set_ylabel("tail RMSE delta（负数=优于 v241）")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_ambiguity(amb_summary: pd.DataFrame) -> Path:
    """绘制不同通道组的邻域未来分歧。"""

    path = FIGURES / "v250_neighbor_ambiguity_by_channel_group.png"
    if amb_summary.empty:
        return path
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        amb_summary["model_name"],
        amb_summary["neighbor_future_pairwise_rmse_mean"],
        color=["#4C78A8", "#F58518", "#54A24B"][: len(amb_summary)],
    )
    ax.set_title("v250 精简通道后的输入邻域未来分歧")
    ax.set_ylabel("近邻未来两两 RMSE 均值（越低越可判别）")
    ax.set_xticklabels(amb_summary["model_name"], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build_next_decision(selection: pd.DataFrame, amb_summary: pd.DataFrame) -> pd.DataFrame:
    """生成下一步决策表。"""

    best = selection.iloc[0]
    accepted = selection[selection["accepted_as_channel_candidate"].astype(bool)]
    if accepted.empty:
        decision = "diagnostic_only"
        reason = "No reduced-channel candidate passed validation no-major-harm + hard-gain checks."
        accepted_name = ""
    else:
        decision = "candidate_for_followup"
        accepted_name = str(accepted.iloc[0]["model_name"])
        reason = "At least one reduced-channel candidate passed validation checks; still needs locked robustness audit."

    amb_note = ""
    if not amb_summary.empty:
        best_amb = amb_summary.iloc[0]
        amb_note = (
            f"Lowest neighbor future pairwise RMSE is {best_amb['neighbor_future_pairwise_rmse_mean']:.3f} "
            f"for {best_amb['model_name']}."
        )

    return pd.DataFrame(
        [
            {
                "decision_item": "best_validation_channel_model",
                "decision": str(best["model_name"]),
                "reason": "Best by validation-only channel-ablation score; test was not used for selection.",
            },
            {
                "decision_item": "accept_reduced_channel_as_next_candidate",
                "decision": bool(not accepted.empty),
                "reason": reason,
            },
            {
                "decision_item": "accepted_model_name",
                "decision": accepted_name,
                "reason": "Empty means v250 remains diagnostic only.",
            },
            {
                "decision_item": "formal_replacement_allowed",
                "decision": False,
                "reason": "v250 is an input-channel ablation; formal claim needs robustness and target-line consistency.",
            },
            {
                "decision_item": "input_ambiguity_note",
                "decision": amb_note,
                "reason": "Use this to decide whether reduced channels actually make hard samples more distinguishable.",
            },
            {
                "decision_item": "recommended_next_task",
                "decision": "v250_review_channel_ablation_or_try_multimodal_if_ambiguity_persists",
                "reason": "Do not tune on test; inspect validation + ambiguity evidence before changing model family.",
            },
        ]
    )


def build_guardrail(selection: pd.DataFrame, split_check: pd.DataFrame, zip_test: str | None) -> Dict[str, object]:
    """写出本轮约束检查。"""

    cross = int(split_check["same_event_uid_cross_split"].sum()) if "same_event_uid_cross_split" in split_check.columns else 0
    return {
        "pass": bool(cross == 0 and not bool(selection["test_used_for_selection"].astype(bool).any()) and zip_test is None),
        "same_event_uid_cross_split_count": cross,
        "test_used_for_selection": bool(selection["test_used_for_selection"].astype(bool).any()),
        "forbidden_routes": {
            "anchor_selector": False,
            "gate_router_selector": False,
            "response_type_hard_routing": False,
            "sample_deletion": False,
            "oracle_best_anchor_as_policy": False,
        },
        "model_selection": "validation_only",
        "history_length_changed": False,
        "road_phase_point_inputs_changed": False,
        "zip_testzip": zip_test,
    }


def write_report(
    selection: pd.DataFrame,
    compare: pd.DataFrame,
    shape_table: pd.DataFrame,
    amb_summary: pd.DataFrame,
    next_decision: pd.DataFrame,
    figures: List[Path],
    zip_path: Path,
) -> None:
    """写中文报告。"""

    best_name = str(selection.iloc[0]["model_name"])
    lines: List[str] = []
    lines.append("# v250 历史通道精简消融报告")
    lines.append("")
    lines.append("## 本轮边界")
    lines.append("")
    lines.append("- 只精简 `X_hist` 的 18 个车辆历史通道；历史长度仍为 -3.0s 到 0.0s，共 31 个时间点。")
    lines.append("- 道路预瞄 `X_road`、phase/current 特征和 point query 不变。")
    lines.append("- 每个精简通道组都从头训练 v241 TCN + multi-head query attention；不加载 v241 checkpoint。")
    lines.append("- validation-only 选择；test 只做 locked report。")
    lines.append("- 不做 anchor selector、gate/router、response-type hard routing，不删除样本。")
    lines.append("")
    lines.append("## Validation 选择")
    lines.append("")
    for _, row in selection.iterrows():
        lines.append(
            f"- `{row['model_name']}`: n_channels={int(row['n_hist_channels'])}, "
            f"score={float(row['validation_selection_score']):.4f}, "
            f"no_major_harm={bool(row['no_major_harm_vs_v241_pass'])}, "
            f"hard_gain={bool(row['hard_gain_pass'])}, "
            f"accepted={bool(row['accepted_as_channel_candidate'])}, "
            f"best_epoch={int(row['best_epoch'])}。"
        )
    lines.append("")
    lines.append(f"当前 best validation diagnostic model：`{best_name}`。")
    lines.append("")
    lines.append("## Test 对照摘要")
    lines.append("")
    keep_cols = ["bucket", "delay_ms"]
    for col in compare.columns:
        if col.startswith("steer_tail_rmse_mean__") or col.startswith("delta_steer_tail_rmse_mean__"):
            keep_cols.append(col)
    keep_cols = [c for c in keep_cols if c in compare.columns]
    short = compare[
        compare["bucket"].isin(["all", "normal_predictable", "observe_later_like", "strong_steer"])
        & compare["delay_ms"].isin([0, 600, 1000])
    ][keep_cols]
    lines.append(short.to_markdown(index=False))
    lines.append("")
    lines.append("## Shape 摘要")
    lines.append("")
    shape_short = shape_table[
        shape_table["split"].eq("test")
        & shape_table["event_group"].isin(["all", "normal", "strong_steer", "observe_later_like", "bad_top10_v241"])
        & shape_table["delay_ms"].isin([0, 600, 1000])
        & shape_table["model_name"].eq(best_name)
    ].copy()
    show_cols = [
        "event_group",
        "delay_ms",
        "n",
        "mean_rmse",
        "mean_range_ratio",
        "mean_slope_ratio",
        "delta_rmse_candidate_minus_v241",
        "delta_range_ratio_candidate_minus_v241",
        "delta_slope_ratio_candidate_minus_v241",
    ]
    lines.append(shape_short[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 输入邻域歧义")
    lines.append("")
    if amb_summary.empty:
        lines.append("- 未生成邻域歧义摘要。")
    else:
        lines.append(amb_summary.to_markdown(index=False))
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
    lines.append("- `tables/v250_model_selection_validation_channel_ablation.csv`")
    lines.append("- `tables/v250_compare_vs_v241_original_remaining.csv`")
    lines.append("- `tables/v250_shape_summary.csv`")
    lines.append("- `tables/v250_input_neighborhood_ambiguity_by_channel.csv`")
    lines.append("- `tables/v250_input_neighborhood_ambiguity_summary.csv`")
    lines.append(f"- ZIP：`{zip_path.name}`")
    lines.append("")
    (REPORTS / "v250_history_channel_ablation_cn.md").write_text("\n".join(lines), encoding="utf-8")


def write_input_hashes() -> None:
    """记录关键输入文件哈希。"""

    paths = [
        V249_SCRIPT,
        V241_PRED,
        V241_MODEL,
        V238.V236_ARRAYS,
        V238.V236_MANIFEST,
    ]
    rows = []
    for path in paths:
        if path.exists():
            rows.append({"path": str(path), "sha256": file_sha256(path), "bytes": int(path.stat().st_size)})
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    """记录输出目录文件清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "bytes": int(path.stat().st_size)})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> str | None:
    """打包 v250 关键产物并返回 zipfile.testzip 结果。"""

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(Path(__file__), arcname=f"scripts/{Path(__file__).name}")
        for folder in [TABLES, FIGURES, REPORTS, LOGS]:
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(OUT)))
        for path in MODELS.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".pkl", ".json"}:
                zf.write(path, arcname=str(path.relative_to(OUT)))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip()


def main() -> None:
    set_seed(SEED)
    clean_out_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v250] device={device}")
    print("[v250] 本轮验证：历史长度不变，只做历史通道精简消融。")

    print("[v250] load v236/v241 data")
    base_data = V238.load_v236_data()
    pred_v241, v241_name = load_v241_prediction()
    y_true_curve = base_data.y_future[:, :, 0].astype(np.float32)
    valid_mask, _ = V238.build_original_remaining_mask(base_data.manifest)
    split_check = V238.split_integrity_check(base_data.manifest)

    groups = channel_groups()
    write_csv(
        pd.DataFrame(
            [
                {
                    "model_name": g.model_name,
                    "n_hist_channels": len(g.channels),
                    "channels": "|".join(g.channels),
                    "description": g.description,
                }
                for g in groups
            ]
        ),
        TABLES / "v250_channel_groups.csv",
    )

    runs: List[ChannelRun] = []
    for group in groups:
        print(f"[v250] training {group.model_name} | channels={len(group.channels)}")
        run = train_one_group(base_data, group, device)
        runs.append(run)
        print(
            f"[v250] {group.model_name} best_epoch={run.run.best_epoch} "
            f"best_val_loss={run.run.best_val_loss:.6f}"
        )

    print("[v250] compute metrics")
    pred_by_model: Dict[str, np.ndarray] = {
        "v236_joint_ridge_existing": base_data.pred_v236[:, :, 0].astype(np.float32),
        v241_name: pred_v241.astype(np.float32),
    }
    for channel_run in runs:
        pred_by_model[channel_run.group.model_name] = channel_run.run.pred_curve.astype(np.float32)

    metrics = V238.compute_metrics_table(
        y_true_curve=y_true_curve,
        pred_by_model=pred_by_model,
        manifest=base_data.manifest,
        eval_modes=["original_remaining", "receding_2s_diagnostic"],
    )

    bad_thresholds: Dict[str, float] = {}
    shape_table = V249.build_shape_validation_table(
        y_true=y_true_curve,
        pred_v241=pred_v241.astype(np.float32),
        pred_by_model={run.group.model_name: run.run.pred_curve.astype(np.float32) for run in runs},
        manifest=base_data.manifest,
        valid_mask=valid_mask,
        bad_top10_thresholds=bad_thresholds,
    )

    selection_rows: List[Dict[str, object]] = []
    for channel_run in runs:
        row = validation_decision(metrics, shape_table, channel_run, v241_name)
        channel_run.validation_row = row
        selection_rows.append(row)
    selection = pd.DataFrame(selection_rows).sort_values("validation_selection_score").reset_index(drop=True)
    selection["validation_rank"] = np.arange(1, len(selection) + 1)
    best_name = str(selection.iloc[0]["model_name"])
    best_run = next(run for run in runs if run.group.model_name == best_name)

    compare = build_compare_table(metrics, [r.group.model_name for r in runs], v241_name)
    ambiguity = build_neighbor_ambiguity_table(
        channel_runs=runs,
        y_true=y_true_curve,
        pred_v241=pred_v241.astype(np.float32),
        manifest=base_data.manifest,
        valid_mask=valid_mask,
    )
    amb_summary = ambiguity_summary(ambiguity)
    next_decision = build_next_decision(selection, amb_summary)
    figures = [
        plot_tail_delta(compare, [r.group.model_name for r in runs], best_name),
        plot_ambiguity(amb_summary),
    ]

    print("[v250] write outputs")
    write_csv(selection, TABLES / "v250_model_selection_validation_channel_ablation.csv")
    write_csv(metrics, TABLES / "v250_metrics_by_delay_and_bucket.csv")
    write_csv(compare, TABLES / "v250_compare_vs_v241_original_remaining.csv")
    write_csv(shape_table, TABLES / "v250_shape_summary.csv")
    write_csv(ambiguity, TABLES / "v250_input_neighborhood_ambiguity_by_channel.csv")
    write_csv(amb_summary, TABLES / "v250_input_neighborhood_ambiguity_summary.csv")
    write_csv(next_decision, TABLES / "v250_next_decision.csv")
    write_csv(split_check, TABLES / "v250_split_integrity_check.csv")
    write_csv(pd.concat([r.run.training_history for r in runs], ignore_index=True), TABLES / "v250_training_history.csv")
    write_csv(
        pd.DataFrame([{"split": k, "bad_top10_v241_threshold": v} for k, v in bad_thresholds.items()]),
        TABLES / "v250_bad_top10_thresholds.csv",
    )

    torch.save(
        {
            "model_name": best_name,
            "state_dict": best_run.run.state_dict,
            "config": best_run.run.config,
            "channels": best_run.group.channels,
            "hist_channel_names": HIST_CHANNELS,
            "selection": selection.to_dict(orient="records"),
        },
        MODELS / "v250_best_channel_ablation_diagnostic.pt",
    )
    with (MODELS / "v250_scalers_and_selection.pkl").open("wb") as f:
        pickle.dump(
            {
                "best_model_name": best_name,
                "best_channels": best_run.group.channels,
                "best_scalers": best_run.scalers,
                "selection": selection,
                "channel_groups": groups,
            },
            f,
        )
    np.savez_compressed(
        OUT / "v250_channel_ablation_predictions.npz",
        y_true_steering_delta=y_true_curve.astype(np.float32),
        pred_v236_steering_delta=pred_by_model["v236_joint_ridge_existing"].astype(np.float32),
        pred_v241_steering_delta=pred_v241.astype(np.float32),
        best_channel_model=np.array([best_name]),
        **{f"pred_{run.group.model_name}_steering_delta": run.run.pred_curve.astype(np.float32) for run in runs},
    )

    write_input_hashes()
    write_file_inventory()
    zip_test = make_zip()
    guardrail = build_guardrail(selection, split_check, zip_test)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v250 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()
    write_report(selection, compare, shape_table, amb_summary, next_decision, figures, ZIP_PATH)
    # 报告写入后重新打包一次，确保 zip 内包含最终报告与 guardrail。
    zip_test = make_zip()
    guardrail["zip_testzip"] = zip_test
    guardrail["pass"] = bool(guardrail["same_event_uid_cross_split_count"] == 0 and not guardrail["test_used_for_selection"] and zip_test is None)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()

    print(f"[v250] best={best_name}")
    print(f"[v250] accepted={bool(selection.iloc[0]['accepted_as_channel_candidate'])}")
    print(f"[v250] report={REPORTS / 'v250_history_channel_ablation_cn.md'}")
    print(f"[v250] zip={ZIP_PATH}")


if __name__ == "__main__":
    main()
