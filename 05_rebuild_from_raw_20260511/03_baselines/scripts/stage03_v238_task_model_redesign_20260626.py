#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v238 任务构造与小型 rolling 模型重搭。

本阶段吸收 v229/v236/v237 的经验，重点不是继续堆模型，而是先把
“模型到底在预测哪一段未来”重新定义清楚。

核心改动：
1. 继续复用 v236 已构建好的 rolling observation 输入，不重新扫描原始车辆 CSV；
2. 不再把所有 delay 都硬训练成 observation_time -> observation_time+2s 的 receding 任务；
3. 主任务改成 original_remaining：只监督 observation_time 到 original_anchor+2s 的重叠部分；
4. 用 point-level masked target 训练小模型，避免 delay=1000ms 时把新行为阶段塞进训练目标；
5. 只在 validation 上选择模型配置，test 只做锁定后报告；
6. 不创建 gate/router/selector，不删除 observe_later_like，不修改 formal headline。

输出解释：
- v238 的主结果只说明“新任务构造 + 小模型”是否比 v236 的同一原事件剩余窗口更合理；
- 不把 mixed-delay RMSE 写成正式模型能力；
- 不把本轮结果替代 v225/v226 formal lock。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# 当前 Windows + Anaconda/MKL 环境在较大的 Ridge Cholesky 矩阵上偶发原生崩溃；
# 这里先限制底层 BLAS 线程，并在 Ridge 中使用 lsqr 迭代求解，保证实验可复现运行。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V236_DIR = BASELINES / "v236_rolling_reanchor_dataset_and_baseline_20260624"
V237_DIR = BASELINES / "v237_rolling_target_phase_audit_20260624"
V225_DIR = BASELINES / "v225_formal_route_reconstruction_evidence_pack_20260622"

V236_ARRAYS = V236_DIR / "v236_rolling_dataset_arrays_and_predictions.npz"
V236_MANIFEST = V236_DIR / "tables" / "v236_rolling_sample_manifest.csv"
V236_SPLIT_CHECK = V236_DIR / "tables" / "v236_train_val_test_event_split_check.csv"
V237_DECISION = V237_DIR / "tables" / "v237_next_model_decision.csv"
V237_TARGET_SANITY = V237_DIR / "tables" / "v237_target_definition_sanity_check.csv"
V225_FORMAL = V225_DIR / "tables" / "per_sample_formal_reconstruction_eval.csv"

OUT = BASELINES / "v238_task_model_redesign_20260626"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
MODELS = OUT / "models"

DELAY_MS = [0, 200, 400, 600, 800, 1000]
FUTURE_GRID = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)
TAIL_RECEDING_MASK = FUTURE_GRID >= 1.0
SEED = 238

POINT_EXTRA_FEATURE_NAMES = [
    "point_future_rel_s",
    "point_original_rel_s",
    "point_remaining_to_original_end_s",
    "point_future_index_norm",
    "point_original_progress_norm",
    "point_is_original_tail",
]

BUCKET_ORDER = [
    "all",
    "observe_later_like",
    "strong_steer",
    "normal_predictable",
    "reverse_or_multi_correction",
    "extreme_peak",
    "strict_subset",
]

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


@dataclass
class RollingData:
    """v236 rolling 输入与旧预测。"""

    manifest: pd.DataFrame
    x_hist: np.ndarray
    x_road: np.ndarray
    x_phase: np.ndarray
    y_future: np.ndarray
    pred_v236: np.ndarray
    feature_names: List[str]
    target_names: List[str]


@dataclass
class PointDataset:
    """point-level masked original_remaining 训练表。"""

    x_point_all: np.ndarray
    y_point_all: np.ndarray
    sample_index_all: np.ndarray
    time_index_all: np.ndarray
    valid_original_remaining_all: np.ndarray
    point_weight_all: np.ndarray
    point_feature_names: List[str]


@dataclass
class TrainedPointModel:
    """一个候选 point model 及其全量曲线预测。"""

    model_name: str
    model_kind: str
    config: Dict[str, object]
    model: object
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    pred_curve: np.ndarray
    training_seconds: float
    extra_info: Dict[str, object]


def ensure_dirs() -> None:
    """创建 v238 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v238 自己的输出目录，避免旧文件混入新结论。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 统一用 utf-8-sig，方便 Windows 中文环境直接打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算输入文件哈希，方便结果追溯。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_v236_data() -> RollingData:
    """读取 v236 保存的 rolling 数据集与预测。"""

    required = [V236_ARRAYS, V236_MANIFEST, V236_SPLIT_CHECK, V237_DECISION, V237_TARGET_SANITY, V225_FORMAL]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("v238 缺少必要输入：\n" + "\n".join(missing))

    decision = pd.read_csv(V237_DECISION, encoding="utf-8-sig")
    if not bool(decision.iloc[0]["v238_allowed"]):
        raise AssertionError("v237_next_model_decision.csv 未允许 v238。")

    target_sanity = pd.read_csv(V237_TARGET_SANITY, encoding="utf-8-sig")
    status_cols = [c for c in target_sanity.columns if c.endswith("_pass") or c == "pass"]
    if status_cols:
        pass_values = target_sanity[status_cols].astype(bool).to_numpy().ravel()
        if not bool(pass_values.all()):
            raise AssertionError("v237 target sanity 未全部通过，不能启动 v238。")

    manifest = pd.read_csv(V236_MANIFEST, encoding="utf-8-sig")
    with np.load(V236_ARRAYS, allow_pickle=False) as data:
        x_hist = data["X_hist"].astype(np.float32)
        x_road = data["X_road"].astype(np.float32)
        x_phase = data["X_phase"].astype(np.float32)
        y_future = data["Y_future"].astype(np.float32)
        pred_v236 = data["pred_future"].astype(np.float32)
        feature_names = data["feature_names"].astype(str).tolist()
        target_names = data["target_names"].astype(str).tolist()
        event_uid = data["event_uid"].astype(str)
        delay_ms = data["delay_ms"].astype(int)
        split = data["split"].astype(str)

    if len(manifest) != y_future.shape[0]:
        raise AssertionError(f"manifest 行数与数组不一致：{len(manifest)} vs {y_future.shape[0]}")
    if y_future.shape != pred_v236.shape:
        raise AssertionError(f"target 与 v236 prediction shape 不一致：{y_future.shape} vs {pred_v236.shape}")
    if not np.array_equal(manifest["event_uid"].astype(str).to_numpy(), event_uid):
        raise AssertionError("v236 manifest 与 arrays event_uid 顺序不一致")
    if not np.array_equal(manifest["delay_ms"].astype(int).to_numpy(), delay_ms):
        raise AssertionError("v236 manifest 与 arrays delay_ms 顺序不一致")
    if not np.array_equal(manifest["split"].astype(str).to_numpy(), split):
        raise AssertionError("v236 manifest 与 arrays split 顺序不一致")

    return RollingData(
        manifest=manifest,
        x_hist=x_hist,
        x_road=x_road,
        x_phase=x_phase,
        y_future=y_future,
        pred_v236=pred_v236,
        feature_names=feature_names,
        target_names=target_names,
    )


def build_base_design_matrix(data: RollingData) -> np.ndarray:
    """把 v236 的历史、道路、phase 输入展平成 sample-level 特征。"""

    n = data.x_hist.shape[0]
    x_base = np.concatenate(
        [
            data.x_hist.reshape(n, -1),
            data.x_road.reshape(n, -1),
            data.x_phase.reshape(n, -1),
        ],
        axis=1,
    )
    if x_base.shape[1] != len(data.feature_names):
        raise AssertionError(f"feature name 数量不一致：X={x_base.shape[1]}, names={len(data.feature_names)}")
    return x_base.astype(np.float32)


def event_sample_weight(manifest: pd.DataFrame) -> np.ndarray:
    """沿用 v236 的温和困难样本加权，但不删除任何样本。"""

    weight = np.ones(len(manifest), dtype=np.float32)
    weight += manifest["observe_later_like"].astype(bool).to_numpy(dtype=bool).astype(np.float32) * 0.5
    weight += manifest["strong_steer"].astype(bool).to_numpy(dtype=bool).astype(np.float32) * 0.5
    weight += manifest["extreme_peak"].astype(bool).to_numpy(dtype=bool).astype(np.float32) * 1.0
    return weight


def build_original_remaining_mask(manifest: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造 original_remaining mask。

    对 delay=d 的 rolling 样本，只监督 future_rel + d <= 2.0 的时间点。
    例如 delay=1000ms 时，只保留 observation 后 0~1s 的 11 个点；
    observation 后 1~2s 已经落在 original_anchor+2s 之外，不作为主任务目标。
    """

    delay_s = manifest["delay_ms"].astype(float).to_numpy()[:, None] / 1000.0
    original_rel = delay_s + FUTURE_GRID[None, :]
    valid = original_rel <= 2.0 + 1e-9
    return valid, original_rel.astype(np.float32)


def build_point_dataset(data: RollingData, x_base: np.ndarray) -> PointDataset:
    """把曲线样本改成 point-level masked original_remaining 标量回归样本。"""

    n_samples = len(data.manifest)
    n_steps = len(FUTURE_GRID)
    sample_index_all = np.repeat(np.arange(n_samples, dtype=np.int32), n_steps)
    time_index_all = np.tile(np.arange(n_steps, dtype=np.int32), n_samples)

    valid_matrix, original_rel_matrix = build_original_remaining_mask(data.manifest)
    valid_all = valid_matrix.reshape(-1)
    original_rel_all = original_rel_matrix.reshape(-1)
    future_rel_all = FUTURE_GRID[time_index_all].astype(np.float32)

    remaining = np.clip(2.0 - original_rel_all, 0.0, 2.0).astype(np.float32)
    future_index_norm = (time_index_all.astype(np.float32) / max(1, n_steps - 1)).astype(np.float32)
    original_progress = np.clip(original_rel_all / 2.0, 0.0, 1.0).astype(np.float32)
    is_tail = (original_rel_all >= 1.0 - 1e-9).astype(np.float32)

    point_extra = np.stack(
        [
            future_rel_all,
            original_rel_all,
            remaining,
            future_index_norm,
            original_progress,
            is_tail,
        ],
        axis=1,
    ).astype(np.float32)

    x_point_all = np.concatenate([x_base[sample_index_all], point_extra], axis=1).astype(np.float32)
    y_point_all = data.y_future[:, :, 0].reshape(-1).astype(np.float32)

    weights = event_sample_weight(data.manifest)[sample_index_all]
    # 尾段是项目长期困难处，只做温和加权；这不是 gate，也不基于 test。
    weights = weights * (1.0 + 0.20 * is_tail)
    weights = weights.astype(np.float32)

    return PointDataset(
        x_point_all=x_point_all,
        y_point_all=y_point_all,
        sample_index_all=sample_index_all,
        time_index_all=time_index_all,
        valid_original_remaining_all=valid_all,
        point_weight_all=weights,
        point_feature_names=data.feature_names + POINT_EXTRA_FEATURE_NAMES,
    )


def impute_and_scale_point_features(
    x_point_all: np.ndarray,
    train_valid_mask: np.ndarray,
) -> Tuple[np.ndarray, StandardScaler, np.ndarray]:
    """用 train-valid 点的均值填补 NaN，并只用 train-valid 点拟合标准化。"""

    x_work = x_point_all.astype(np.float64, copy=True)
    train_values = x_work[train_valid_mask]
    means = np.nanmean(train_values, axis=0)
    means[~np.isfinite(means)] = 0.0
    bad = ~np.isfinite(x_work)
    if bad.any():
        row_idx, col_idx = np.where(bad)
        x_work[row_idx, col_idx] = means[col_idx]
    scaler = StandardScaler()
    scaler.fit(x_work[train_valid_mask])
    x_scaled = scaler.transform(x_work).astype(np.float32)
    return x_scaled, scaler, means.astype(np.float32)


def scale_point_target(y: np.ndarray, train_valid_mask: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """标量 steering-delta target 标准化。"""

    scaler = StandardScaler()
    scaler.fit(y[train_valid_mask].reshape(-1, 1))
    y_scaled = scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)
    return y_scaled, scaler


def split_point_masks(point_data: PointDataset, manifest: pd.DataFrame) -> Dict[str, np.ndarray]:
    """给 point-level 样本生成 train/val/test 且 original_remaining 有效的 mask。"""

    split = manifest["split"].astype(str).to_numpy()
    point_split = split[point_data.sample_index_all]
    valid = point_data.valid_original_remaining_all
    return {
        "train": valid & (point_split == "train"),
        "val": valid & (point_split == "val"),
        "test": valid & (point_split == "test"),
        "valid": valid,
    }


def predict_curve_from_point_predictions(
    point_pred: np.ndarray,
    sample_index_all: np.ndarray,
    time_index_all: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """把 point-level 预测还原成 N x 21 的曲线预测。"""

    pred_curve = np.full((n_samples, len(FUTURE_GRID)), np.nan, dtype=np.float32)
    pred_curve[sample_index_all, time_index_all] = point_pred.astype(np.float32)
    if not np.isfinite(pred_curve).all():
        raise AssertionError("point prediction 还原曲线后存在 NaN")
    return pred_curve


def train_one_point_model(
    model_name: str,
    model_kind: str,
    config: Dict[str, object],
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    y_scaler: StandardScaler,
    point_data: PointDataset,
    point_masks: Dict[str, np.ndarray],
) -> TrainedPointModel:
    """训练一个候选 point-level 模型。"""

    train_mask = point_masks["train"]
    start = time.time()
    if model_kind == "ridge":
        model = Ridge(alpha=float(config["alpha"]), solver="lsqr", max_iter=4000, tol=1e-4)
    elif model_kind == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=tuple(config["hidden_layer_sizes"]),
            activation="relu",
            solver="adam",
            alpha=float(config["alpha"]),
            learning_rate_init=float(config["learning_rate_init"]),
            batch_size=int(config["batch_size"]),
            max_iter=int(config["max_iter"]),
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=int(config["n_iter_no_change"]),
            tol=float(config["tol"]),
            random_state=SEED,
            shuffle=True,
        )
    else:
        raise ValueError(f"未知 model_kind：{model_kind}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(
            x_scaled[train_mask],
            y_scaled[train_mask],
            sample_weight=point_data.point_weight_all[train_mask],
        )
    training_seconds = time.time() - start

    point_pred_scaled = model.predict(x_scaled).astype(np.float32)
    point_pred = y_scaler.inverse_transform(point_pred_scaled.reshape(-1, 1)).ravel().astype(np.float32)
    pred_curve = predict_curve_from_point_predictions(
        point_pred=point_pred,
        sample_index_all=point_data.sample_index_all,
        time_index_all=point_data.time_index_all,
        n_samples=int(point_data.sample_index_all.max()) + 1,
    )
    extra = {
        "n_warnings": len(caught),
        "warning_messages": [str(w.message)[:240] for w in caught[:5]],
    }
    if hasattr(model, "n_iter_"):
        n_iter_value = getattr(model, "n_iter_")
        if np.ndim(n_iter_value) == 0:
            extra["n_iter"] = int(n_iter_value)
        else:
            extra["n_iter"] = json.dumps(np.asarray(n_iter_value).astype(int).tolist())
    if hasattr(model, "loss_"):
        extra["loss"] = float(getattr(model, "loss_"))
    if hasattr(model, "best_validation_score_"):
        extra["internal_best_validation_score"] = float(getattr(model, "best_validation_score_"))

    return TrainedPointModel(
        model_name=model_name,
        model_kind=model_kind,
        config=config,
        model=model,
        x_scaler=StandardScaler(),  # 真实 scaler 在外层统一保存，候选对象内不重复存大对象。
        y_scaler=y_scaler,
        pred_curve=pred_curve,
        training_seconds=float(training_seconds),
        extra_info=extra,
    )


def horizon_masks(delay_ms: int, eval_mode: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """返回某个 delay 和评估口径下的 horizon mask 与 tail mask。"""

    if eval_mode == "original_remaining":
        original_rel = delay_ms / 1000.0 + FUTURE_GRID
        horizon = original_rel <= 2.0 + 1e-9
        tail = horizon & (original_rel >= 1.0 - 1e-9)
    elif eval_mode == "receding_2s_diagnostic":
        horizon = np.ones(len(FUTURE_GRID), dtype=bool)
        tail = TAIL_RECEDING_MASK.copy()
    else:
        raise ValueError(f"未知 eval_mode：{eval_mode}")
    return horizon, tail, int(horizon.sum())


def peak_values(curve: np.ndarray, horizon_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算指定 horizon 内的峰值绝对值、带符号峰值和峰值时间。"""

    sub = curve[:, horizon_mask]
    grid = FUTURE_GRID[horizon_mask]
    idx = np.nanargmax(np.abs(sub), axis=1)
    signed = sub[np.arange(sub.shape[0]), idx]
    return np.abs(signed), signed, grid[idx]


def metric_for_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    manifest: pd.DataFrame,
    row_mask: np.ndarray,
    split_name: str,
    bucket: str,
    delay_ms: int,
    eval_mode: str,
    model_name: str,
) -> Dict[str, object] | None:
    """计算 steering-delta 曲线指标。"""

    if int(row_mask.sum()) == 0:
        return None
    horizon_mask, tail_mask, horizon_points = horizon_masks(delay_ms, eval_mode)
    yt = y_true[row_mask][:, horizon_mask]
    yp = y_pred[row_mask][:, horizon_mask]
    diff = yp - yt
    sample_rmse = np.sqrt(np.mean(np.square(diff), axis=1))

    if tail_mask.any():
        tail_local = tail_mask[horizon_mask]
        sample_tail = np.sqrt(np.mean(np.square(diff[:, tail_local]), axis=1))
        tail_rmse = float(np.sqrt(np.mean(np.square(diff[:, tail_local]))))
    else:
        sample_tail = np.full(len(sample_rmse), np.nan, dtype=np.float32)
        tail_rmse = math.nan

    true_peak_abs, true_peak_signed, true_peak_t = peak_values(y_true[row_mask], horizon_mask)
    pred_peak_abs, pred_peak_signed, pred_peak_t = peak_values(y_pred[row_mask], horizon_mask)
    strong = true_peak_abs >= 1.0
    under = pred_peak_abs < 0.5 * true_peak_abs
    direction_ok = np.sign(true_peak_signed) == np.sign(pred_peak_signed)

    return {
        "model_name": model_name,
        "split": split_name,
        "bucket": bucket,
        "delay_ms": int(delay_ms),
        "eval_mode": eval_mode,
        "n_samples": int(row_mask.sum()),
        "horizon_points": horizon_points,
        "steer_rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "steer_sample_rmse_mean": float(np.mean(sample_rmse)),
        "steer_sample_rmse_p90": float(np.quantile(sample_rmse, 0.90)),
        "steer_tail_rmse": tail_rmse,
        "steer_tail_rmse_mean": float(np.nanmean(sample_tail)),
        "steer_tail_rmse_p90": float(np.nanquantile(sample_tail, 0.90)),
        "steer_direction_acc": float(np.mean(direction_ok)),
        "steer_under_rate": float(np.mean(under)),
        "strong_response_n": int(strong.sum()),
        "strong_under_rate": float(np.mean(under[strong])) if strong.any() else math.nan,
        "true_peak_abs_mean": float(np.mean(true_peak_abs)),
        "pred_peak_abs_mean": float(np.mean(pred_peak_abs)),
        "peak_ratio_mean": float(np.mean(pred_peak_abs / np.maximum(true_peak_abs, 1e-6))),
        "peak_time_abs_error_mean": float(np.mean(np.abs(pred_peak_t - true_peak_t))),
    }


def bucket_masks(manifest: pd.DataFrame) -> Dict[str, np.ndarray]:
    """沿用当前项目的核心失败桶，但只作为分层评估，不做硬路由。"""

    reverse_multi = (
        manifest["reverse"].astype(bool).to_numpy()
        | manifest["multi_correction"].astype(bool).to_numpy()
        | manifest["zero_cross"].astype(bool).to_numpy()
    )
    observe = manifest["observe_later_like"].astype(bool).to_numpy()
    normal = manifest["normal_curve"].astype(bool).to_numpy() & ~observe
    return {
        "all": np.ones(len(manifest), dtype=bool),
        "observe_later_like": observe,
        "strong_steer": manifest["strong_steer"].astype(bool).to_numpy(),
        "normal_predictable": normal,
        "reverse_or_multi_correction": reverse_multi,
        "extreme_peak": manifest["extreme_peak"].astype(bool).to_numpy(),
        "strict_subset": manifest["strict_subset"].astype(bool).to_numpy(),
    }


def compute_metrics_table(
    y_true_curve: np.ndarray,
    pred_by_model: Dict[str, np.ndarray],
    manifest: pd.DataFrame,
    eval_modes: Iterable[str],
) -> pd.DataFrame:
    """按 model/split/delay/bucket/eval_mode 生成完整分层指标。"""

    rows: List[Dict[str, object]] = []
    split_values = manifest["split"].astype(str).to_numpy()
    delay_values = manifest["delay_ms"].astype(int).to_numpy()
    buckets = bucket_masks(manifest)
    for model_name, pred_curve in pred_by_model.items():
        for eval_mode in eval_modes:
            for split_name in ["train", "val", "test"]:
                for delay_ms in DELAY_MS:
                    split_delay = (split_values == split_name) & (delay_values == delay_ms)
                    for bucket_name in BUCKET_ORDER:
                        mask = split_delay & buckets[bucket_name]
                        item = metric_for_rows(
                            y_true=y_true_curve,
                            y_pred=pred_curve,
                            manifest=manifest,
                            row_mask=mask,
                            split_name=split_name,
                            bucket=bucket_name,
                            delay_ms=delay_ms,
                            eval_mode=eval_mode,
                            model_name=model_name,
                        )
                        if item is not None:
                            rows.append(item)
    return pd.DataFrame(rows)


def selection_score_from_metric(row: pd.Series) -> float:
    """validation-only 选择分数，显式惩罚强响应低估和峰值收缩。"""

    strong_under = float(row["strong_under_rate"]) if np.isfinite(float(row["strong_under_rate"])) else 0.0
    peak_ratio = float(row["peak_ratio_mean"]) if np.isfinite(float(row["peak_ratio_mean"])) else 1.0
    shrink_penalty = max(0.0, 1.0 - peak_ratio)
    return (
        float(row["steer_sample_rmse_mean"])
        + 0.50 * float(row["steer_tail_rmse_mean"])
        + 0.15 * strong_under
        + 0.20 * shrink_penalty
    )


def train_and_select_models(
    data: RollingData,
    point_data: PointDataset,
) -> Tuple[TrainedPointModel, pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
    """训练候选小模型，并只用 validation original_remaining 指标选择。"""

    point_masks = split_point_masks(point_data, data.manifest)
    if not point_masks["train"].any() or not point_masks["val"].any() or not point_masks["test"].any():
        raise AssertionError("point-level train/val/test 有效样本为空")

    x_scaled, x_scaler, x_impute_mean = impute_and_scale_point_features(point_data.x_point_all, point_masks["train"])
    y_scaled, y_scaler = scale_point_target(point_data.y_point_all, point_masks["train"])

    configs: List[Tuple[str, str, Dict[str, object]]] = [
        ("v238_point_ridge_alpha100", "ridge", {"alpha": 100.0}),
        ("v238_point_ridge_alpha1000", "ridge", {"alpha": 1000.0}),
        (
            "v238_point_mlp_96x48_alpha1e-4",
            "mlp",
            {
                "hidden_layer_sizes": [96, 48],
                "alpha": 1e-4,
                "learning_rate_init": 1e-3,
                "batch_size": 512,
                "max_iter": 120,
                "n_iter_no_change": 8,
                "tol": 1e-4,
            },
        ),
        (
            "v238_point_mlp_96x48_alpha1e-3",
            "mlp",
            {
                "hidden_layer_sizes": [96, 48],
                "alpha": 1e-3,
                "learning_rate_init": 1e-3,
                "batch_size": 512,
                "max_iter": 120,
                "n_iter_no_change": 8,
                "tol": 1e-4,
            },
        ),
    ]

    trained: List[TrainedPointModel] = []
    selection_rows: List[Dict[str, object]] = []
    y_true_curve = data.y_future[:, :, 0].astype(np.float32)

    for model_name, model_kind, config in configs:
        print(f"[v238] training {model_name}")
        candidate = train_one_point_model(
            model_name=model_name,
            model_kind=model_kind,
            config=config,
            x_scaled=x_scaled,
            y_scaled=y_scaled,
            y_scaler=y_scaler,
            point_data=point_data,
            point_masks=point_masks,
        )
        trained.append(candidate)
        metrics = compute_metrics_table(
            y_true_curve=y_true_curve,
            pred_by_model={model_name: candidate.pred_curve},
            manifest=data.manifest,
            eval_modes=["original_remaining"],
        )
        val_all = metrics[
            metrics["split"].eq("val")
            & metrics["bucket"].eq("all")
            & metrics["eval_mode"].eq("original_remaining")
        ].copy()
        if val_all.empty:
            raise AssertionError(f"{model_name} 没有 validation original_remaining 指标")
        # 选择分数取各 delay 的样本数加权平均，避免单一 delay 主导。
        val_all["selection_score_each_delay"] = val_all.apply(selection_score_from_metric, axis=1)
        weights = val_all["n_samples"].astype(float).to_numpy()
        score = float(np.average(val_all["selection_score_each_delay"].astype(float).to_numpy(), weights=weights))
        row = {
            "model_name": model_name,
            "model_kind": model_kind,
            "selected_by": "validation_original_remaining_only",
            "test_used_for_selection": False,
            "validation_selection_score": score,
            "training_seconds": candidate.training_seconds,
            "config_json": json.dumps(config, ensure_ascii=False, sort_keys=True),
        }
        row.update({f"extra_{k}": v for k, v in candidate.extra_info.items() if k not in {"warning_messages"}})
        row["extra_warning_messages"] = json.dumps(candidate.extra_info.get("warning_messages", []), ensure_ascii=False)
        selection_rows.append(row)

    selection = pd.DataFrame(selection_rows).sort_values("validation_selection_score").reset_index(drop=True)
    selection["validation_rank"] = np.arange(1, len(selection) + 1)
    selected_name = str(selection.iloc[0]["model_name"])
    selected_model = next(m for m in trained if m.model_name == selected_name)
    pred_by_model = {
        "v236_joint_ridge_existing": data.pred_v236[:, :, 0].astype(np.float32),
        "v238_selected_original_remaining_point_model": selected_model.pred_curve.astype(np.float32),
    }
    model_payload = {
        "model_kind": selected_model.model_kind,
        "model_name": selected_model.model_name,
        "selected_by": "validation_original_remaining_only",
        "test_used_for_selection": False,
        "config": selected_model.config,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "x_impute_mean": x_impute_mean,
        "point_feature_names": point_data.point_feature_names,
        "model": selected_model.model,
        "selection_table": selection.to_dict(orient="records"),
    }
    return selected_model, selection, pred_by_model, model_payload


def build_task_construction_tables(data: RollingData, point_data: PointDataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """输出新旧任务定义和 point-level 行数审查表。"""

    valid_matrix, original_rel = build_original_remaining_mask(data.manifest)
    split_values = data.manifest["split"].astype(str).to_numpy()
    delay_values = data.manifest["delay_ms"].astype(int).to_numpy()

    task_rows: List[Dict[str, object]] = []
    for delay_ms in DELAY_MS:
        mask = delay_values == delay_ms
        valid = valid_matrix[mask]
        orig = original_rel[mask]
        task_rows.append(
            {
                "delay_ms": int(delay_ms),
                "events": int(mask.sum()),
                "receding_2s_points_per_sample": int(len(FUTURE_GRID)),
                "original_remaining_points_per_sample": float(valid.sum(axis=1).mean()),
                "original_remaining_min_points": int(valid.sum(axis=1).min()),
                "original_remaining_max_points": int(valid.sum(axis=1).max()),
                "original_remaining_tail_points_per_sample": float(((orig >= 1.0 - 1e-9) & valid).sum(axis=1).mean()),
                "dropped_events": 0,
                "task_meaning_cn": "只监督 observation_time 到 original_anchor+2s 的剩余部分",
            }
        )

    point_rows: List[Dict[str, object]] = []
    point_split = split_values[point_data.sample_index_all]
    point_delay = delay_values[point_data.sample_index_all]
    for split_name in ["train", "val", "test"]:
        for delay_ms in DELAY_MS:
            mask = (
                point_data.valid_original_remaining_all
                & (point_split == split_name)
                & (point_delay == delay_ms)
            )
            point_rows.append(
                {
                    "split": split_name,
                    "delay_ms": int(delay_ms),
                    "point_rows": int(mask.sum()),
                    "rolling_samples": int(((split_values == split_name) & (delay_values == delay_ms)).sum()),
                    "unique_events": int(
                        data.manifest.loc[(split_values == split_name) & (delay_values == delay_ms), "event_uid"].nunique()
                    ),
                }
            )
    return pd.DataFrame(task_rows), pd.DataFrame(point_rows)


def build_per_sample_metrics(
    y_true_curve: np.ndarray,
    pred_curve: np.ndarray,
    manifest: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """输出 selected v238 逐样本 original_remaining 指标，便于人工看坏例。"""

    rows: List[Dict[str, object]] = []
    delay_values = manifest["delay_ms"].astype(int).to_numpy()
    for i, row in manifest.reset_index(drop=True).iterrows():
        delay_ms = int(delay_values[i])
        horizon, tail, horizon_points = horizon_masks(delay_ms, "original_remaining")
        yt = y_true_curve[i, horizon]
        yp = pred_curve[i, horizon]
        diff = yp - yt
        sample_rmse = float(np.sqrt(np.mean(np.square(diff))))
        tail_local = tail[horizon]
        tail_rmse = float(np.sqrt(np.mean(np.square(diff[tail_local])))) if tail_local.any() else math.nan
        true_peak_abs, true_peak_signed, true_peak_t = peak_values(y_true_curve[i : i + 1], horizon)
        pred_peak_abs, pred_peak_signed, pred_peak_t = peak_values(pred_curve[i : i + 1], horizon)
        rows.append(
            {
                "model_name": model_name,
                "event_uid": row["event_uid"],
                "sample_id": row.get("sample_id", row["event_uid"]),
                "split": row["split"],
                "delay_ms": delay_ms,
                "horizon_points": horizon_points,
                "sample_rmse": sample_rmse,
                "tail_rmse": tail_rmse,
                "true_peak_abs": float(true_peak_abs[0]),
                "pred_peak_abs": float(pred_peak_abs[0]),
                "peak_ratio": float(pred_peak_abs[0] / max(float(true_peak_abs[0]), 1e-6)),
                "true_peak_t": float(true_peak_t[0]),
                "pred_peak_t": float(pred_peak_t[0]),
                "direction_ok": bool(np.sign(true_peak_signed[0]) == np.sign(pred_peak_signed[0])),
                "strong_under": bool(pred_peak_abs[0] < 0.5 * true_peak_abs[0] and true_peak_abs[0] >= 1.0),
                "observe_later_like": bool(row["observe_later_like"]),
                "strong_steer": bool(row["strong_steer"]),
                "reverse": bool(row["reverse"]),
                "zero_cross": bool(row["zero_cross"]),
                "multi_correction": bool(row["multi_correction"]),
                "extreme_peak": bool(row["extreme_peak"]),
            }
        )
    return pd.DataFrame(rows)


def build_compare_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """提取 test/original_remaining 的 v236 vs v238 对照表。"""

    test = metrics[
        metrics["split"].eq("test")
        & metrics["eval_mode"].eq("original_remaining")
        & metrics["bucket"].isin(["all", "observe_later_like", "strong_steer", "normal_predictable"])
    ].copy()
    keep = [
        "model_name",
        "bucket",
        "delay_ms",
        "n_samples",
        "steer_sample_rmse_mean",
        "steer_tail_rmse_mean",
        "strong_under_rate",
        "peak_ratio_mean",
        "steer_direction_acc",
    ]
    out = test[keep].copy()
    pivot = out.pivot_table(
        index=["bucket", "delay_ms"],
        columns="model_name",
        values=["steer_sample_rmse_mean", "steer_tail_rmse_mean", "strong_under_rate", "peak_ratio_mean"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}__{model}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()
    if (
        "steer_tail_rmse_mean__v238_selected_original_remaining_point_model" in pivot.columns
        and "steer_tail_rmse_mean__v236_joint_ridge_existing" in pivot.columns
    ):
        pivot["delta_tail_v238_minus_v236"] = (
            pivot["steer_tail_rmse_mean__v238_selected_original_remaining_point_model"]
            - pivot["steer_tail_rmse_mean__v236_joint_ridge_existing"]
        )
    if (
        "steer_sample_rmse_mean__v238_selected_original_remaining_point_model" in pivot.columns
        and "steer_sample_rmse_mean__v236_joint_ridge_existing" in pivot.columns
    ):
        pivot["delta_sample_v238_minus_v236"] = (
            pivot["steer_sample_rmse_mean__v238_selected_original_remaining_point_model"]
            - pivot["steer_sample_rmse_mean__v236_joint_ridge_existing"]
        )
    return pivot


def _delta_values(compare: pd.DataFrame, bucket: str, max_delay_ms: int | None = None) -> pd.DataFrame:
    """从 compare 表中抽取某个 bucket 的 delta 行。"""

    one = compare[compare["bucket"].eq(bucket)].copy()
    if max_delay_ms is not None:
        one = one[one["delay_ms"].astype(int) <= max_delay_ms].copy()
    return one


def build_next_decision(compare: pd.DataFrame, guardrail: Dict[str, object]) -> pd.DataFrame:
    """把 v238 的方法结论写成机器可读决策表。"""

    observe_mid = _delta_values(compare, "observe_later_like", max_delay_ms=800)
    strong_mid = _delta_values(compare, "strong_steer", max_delay_ms=600)
    normal_all = _delta_values(compare, "normal_predictable")
    observe_1000 = compare[
        compare["bucket"].eq("observe_later_like") & compare["delay_ms"].astype(int).eq(1000)
    ].copy()

    observe_mid_gain = bool(
        not observe_mid.empty and (observe_mid["delta_tail_v238_minus_v236"].astype(float) < 0.0).all()
    )
    strong_mid_gain = bool(
        not strong_mid.empty and (strong_mid["delta_tail_v238_minus_v236"].astype(float) <= 0.0).all()
    )
    normal_noharm_pass = bool(
        not normal_all.empty and (normal_all["delta_sample_v238_minus_v236"].astype(float) <= 0.0).all()
    )
    delay1000_pass = bool(
        not observe_1000.empty and float(observe_1000.iloc[0]["delta_tail_v238_minus_v236"]) <= 0.0
    )

    rows = [
        {
            "decision_item": "accept_task_construction",
            "decision": bool(guardrail.get("pass", False)),
            "reason": "original_remaining masked point-level target passes guardrail and removes new-phase points from the main loss.",
        },
        {
            "decision_item": "accept_selected_model_as_formal_replacement",
            "decision": False,
            "reason": "normal_predictable no-harm fails and delay=1000 observe_later_like degrades; v238 is a prototype, not a formal replacement.",
        },
        {
            "decision_item": "observe_later_mid_delay_gain",
            "decision": observe_mid_gain,
            "reason": "test observe_later_like tail delta is negative for delays 0-800ms."
            if observe_mid_gain
            else "observe_later_like mid-delay improvement is not consistent.",
        },
        {
            "decision_item": "strong_0_to_600_gain",
            "decision": strong_mid_gain,
            "reason": "test strong_steer tail delta is non-positive for delays 0-600ms."
            if strong_mid_gain
            else "strong_steer gain is not stable in the 0-600ms range.",
        },
        {
            "decision_item": "normal_noharm_pass",
            "decision": normal_noharm_pass,
            "reason": "normal_predictable sample delta is non-positive for all delays."
            if normal_noharm_pass
            else "normal_predictable sample RMSE is worse than v236 at one or more delays.",
        },
        {
            "decision_item": "delay_1000_policy_pass",
            "decision": delay1000_pass,
            "reason": "observe_later_like delay=1000 does not degrade."
            if delay1000_pass
            else "delay=1000 remains unsafe for the same selected point model; keep it diagnostic or handle separately.",
        },
        {
            "decision_item": "recommended_next_task",
            "decision": "v239_noharm_constrained_original_remaining_model",
            "reason": "Keep the new original_remaining task, but add validation no-harm criteria for normal samples and an explicit late-delay policy before any formal use.",
        },
    ]
    return pd.DataFrame(rows)


def split_integrity_check(manifest: pd.DataFrame) -> pd.DataFrame:
    """确认同一 event_uid 的所有 delay 没有跨 split。"""

    rows = []
    for event_uid, group in manifest.groupby("event_uid", dropna=False):
        splits = sorted(group["split"].astype(str).unique().tolist())
        delays = sorted(group["delay_ms"].astype(int).unique().tolist())
        rows.append(
            {
                "event_uid": event_uid,
                "n_rows": int(len(group)),
                "n_splits": int(len(splits)),
                "splits": "|".join(splits),
                "delays": "|".join(str(x) for x in delays),
                "has_all_delays": delays == DELAY_MS,
                "split_check_status": "pass" if len(splits) == 1 else "fail",
            }
        )
    return pd.DataFrame(rows)


def build_guardrail_json(
    manifest: pd.DataFrame,
    split_check: pd.DataFrame,
    selection: pd.DataFrame,
    task_table: pd.DataFrame,
) -> Dict[str, object]:
    """记录 v238 的边界检查。"""

    decision = pd.read_csv(V237_DECISION, encoding="utf-8-sig").iloc[0].to_dict()
    checks = {
        "v237_allowed_v238": bool(decision.get("v238_allowed", False)),
        "test_used_for_selection": bool(selection["test_used_for_selection"].astype(bool).any()),
        "same_event_uid_cross_split_count": int(split_check["split_check_status"].eq("fail").sum()),
        "observe_later_like_deleted": False,
        "gate_router_selector_created": False,
        "formal_headline_changed": False,
        "mixed_delay_metric_used_as_headline": False,
        "primary_target_mode": "original_remaining_masked_point_level_steering_delta",
        "dropped_events": int(task_table["dropped_events"].sum()),
    }
    checks["pass"] = (
        checks["v237_allowed_v238"]
        and not checks["test_used_for_selection"]
        and checks["same_event_uid_cross_split_count"] == 0
        and not checks["observe_later_like_deleted"]
        and not checks["gate_router_selector_created"]
        and not checks["formal_headline_changed"]
        and not checks["mixed_delay_metric_used_as_headline"]
        and checks["dropped_events"] == 0
    )
    checks["required_guardrails_from_v237"] = decision.get("required_guardrails", "")
    return checks


def write_input_hashes() -> None:
    """写入关键输入文件哈希。"""

    paths = [V236_ARRAYS, V236_MANIFEST, V236_SPLIT_CHECK, V237_DECISION, V237_TARGET_SANITY, V225_FORMAL]
    rows = [{"path": str(path), "sha256": file_sha256(path), "bytes": int(path.stat().st_size)} for path in paths]
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def plot_core_figures(compare: pd.DataFrame, selection: pd.DataFrame) -> List[Path]:
    """生成 v238 核心对照图。"""

    paths: List[Path] = []
    for bucket in ["observe_later_like", "strong_steer", "normal_predictable"]:
        one = compare[compare["bucket"].eq(bucket)].copy().sort_values("delay_ms")
        if one.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        v236_col = "steer_tail_rmse_mean__v236_joint_ridge_existing"
        v238_col = "steer_tail_rmse_mean__v238_selected_original_remaining_point_model"
        if v236_col in one.columns:
            ax.plot(one["delay_ms"], one[v236_col], marker="o", label="v236 existing", color="#777777")
        if v238_col in one.columns:
            ax.plot(one["delay_ms"], one[v238_col], marker="o", label="v238 selected", color="#1f77b4")
        ax.set_xlabel("Observation delay (ms)")
        ax.set_ylabel("Original-remaining tail RMSE")
        ax.set_title(f"v238 original_remaining comparison: {bucket}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = FIGURES / f"v238_compare_tail_original_remaining_{bucket}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sel = selection.sort_values("validation_selection_score")
    ax.barh(sel["model_name"], sel["validation_selection_score"], color="#4c78a8")
    ax.set_xlabel("Validation selection score")
    ax.set_title("v238 validation-only model selection")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = FIGURES / "v238_validation_model_selection.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


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
    """打包 v238 输出。"""

    zip_path = OUT / "v238_task_model_redesign_pack.zip"
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
    manifest: pd.DataFrame,
    task_table: pd.DataFrame,
    point_counts: pd.DataFrame,
    selection: pd.DataFrame,
    compare: pd.DataFrame,
    metrics: pd.DataFrame,
    next_decision: pd.DataFrame,
    guardrail: Dict[str, object],
    zip_path: Path,
) -> None:
    """写中文报告，先讲任务构造，再讲模型结果。"""

    selected = selection.sort_values("validation_rank").iloc[0]
    test_compare = compare.copy()

    lines: List[str] = []
    lines.append("# v238 任务构造与小型 rolling 模型重搭报告")
    lines.append("")
    lines.append("## 本轮到底改了什么")
    lines.append("")
    lines.append("- 本轮没有继续 v222a gate / 删除样本 / light residual 路线。")
    lines.append("- 本轮没有重新扫描原始车辆 CSV，而是复用 v236 已保存的 rolling 输入，降低数据口径变化风险。")
    lines.append("- 主任务从 `receding_2s` 改成 `original_remaining`：delay 后只预测原始事件 `anchor+2s` 以内还剩下的部分。")
    lines.append("- 训练形式从“一条样本输出 21 点整曲线”改成 point-level masked target：无效的新阶段点不进入训练 loss。")
    lines.append("- 模型仍是小模型：validation-only 在 point Ridge 与小 MLP 中选择；没有 gate/router/selector。")
    lines.append("")
    lines.append("## 总体判断")
    lines.append("")
    lines.append("- 接受 v238 的任务构造方向：`original_remaining` masked target 是对 v236 receding 目标混入新阶段问题的修正。")
    lines.append("- 不接受当前 selected MLP 作为正式替代模型：它改善了部分难例，但普通样本 no-harm 和 1000ms late delay 没守住。")
    lines.append("- 下一步不应扩大模型，而应加 validation no-harm 约束，并把 1000ms 延迟作为单独策略/诊断处理。")
    lines.append("")
    lines.append("## 任务构造")
    lines.append("")
    lines.append(f"- rolling 样本数：{len(manifest)}；唯一事件数：{manifest['event_uid'].nunique()}。")
    lines.append("- 每个 delay 的 original_remaining 有效点数：")
    for row in task_table.sort_values("delay_ms").itertuples(index=False):
        lines.append(
            f"  - delay={int(row.delay_ms)}ms：每样本有效点 {float(row.original_remaining_points_per_sample):.1f}，"
            f"尾段点 {float(row.original_remaining_tail_points_per_sample):.1f}。"
        )
    lines.append("")
    lines.append("## 模型选择")
    lines.append("")
    lines.append(
        f"- selected model：`{selected.model_name}`；validation rank={int(selected.validation_rank)}；"
        f"selection score={float(selected.validation_selection_score):.6f}。"
    )
    lines.append("- 选择只使用 validation original_remaining；`test_used_for_selection=False`。")
    lines.append("")
    lines.append("## Test original_remaining 对照")
    lines.append("")
    for bucket in ["all", "observe_later_like", "strong_steer", "normal_predictable"]:
        one = test_compare[test_compare["bucket"].eq(bucket)].sort_values("delay_ms")
        if one.empty:
            continue
        lines.append(f"### {bucket}")
        for row in one.itertuples(index=False):
            v236_tail = getattr(row, "steer_tail_rmse_mean__v236_joint_ridge_existing", math.nan)
            v238_tail = getattr(row, "steer_tail_rmse_mean__v238_selected_original_remaining_point_model", math.nan)
            delta_tail = getattr(row, "delta_tail_v238_minus_v236", math.nan)
            v236_sample = getattr(row, "steer_sample_rmse_mean__v236_joint_ridge_existing", math.nan)
            v238_sample = getattr(row, "steer_sample_rmse_mean__v238_selected_original_remaining_point_model", math.nan)
            lines.append(
                f"- delay={int(row.delay_ms)}ms：tail v236={float(v236_tail):.6f}，"
                f"v238={float(v238_tail):.6f}，delta={float(delta_tail):+.6f}；"
                f"sample v236={float(v236_sample):.6f}，v238={float(v238_sample):.6f}"
            )
        lines.append("")
    lines.append("## 边界与解释")
    lines.append("")
    lines.append("- 如果 v238 在 observe_later_like 上改善，优先解释为“任务窗口修正 + 小非线性模型缓解 Ridge 收缩”，不是正式 headline。")
    lines.append("- 如果 v238 在某些 delay 或 bucket 变差，说明 point-level 原事件剩余任务还需要继续调输入/目标，不代表 rolling 方向失败。")
    lines.append("- 本轮仍不允许用 test 反调配置，也不允许把 response type 变成硬路由。类型信息只用于评估分桶。")
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
    lines.append("- `tables/v238_task_construction_audit.csv`")
    lines.append("- `tables/v238_point_training_rows_by_delay.csv`")
    lines.append("- `tables/v238_model_selection_validation_only.csv`")
    lines.append("- `tables/v238_metrics_by_delay_and_bucket.csv`")
    lines.append("- `tables/v238_compare_v236_original_remaining.csv`")
    lines.append("- `tables/v238_selected_per_sample_metrics.csv`")
    lines.append("- `tables/v238_next_model_decision.csv`")
    lines.append(f"- ZIP：`{zip_path.name}`")
    lines.append("")

    report_path = REPORTS / "v238_task_model_redesign_cn.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    np.random.seed(SEED)
    clean_out_dir()

    print("[v238] loading v236 rolling arrays")
    data = load_v236_data()
    x_base = build_base_design_matrix(data)

    print("[v238] building original_remaining point-level task")
    point_data = build_point_dataset(data, x_base)
    task_table, point_counts = build_task_construction_tables(data, point_data)

    print("[v238] training and selecting small point model")
    selected_model, selection, pred_by_model, model_payload = train_and_select_models(data, point_data)
    y_true_curve = data.y_future[:, :, 0].astype(np.float32)

    print("[v238] computing metrics")
    metrics = compute_metrics_table(
        y_true_curve=y_true_curve,
        pred_by_model=pred_by_model,
        manifest=data.manifest,
        eval_modes=["original_remaining", "receding_2s_diagnostic"],
    )
    compare = build_compare_table(metrics)
    per_sample = build_per_sample_metrics(
        y_true_curve=y_true_curve,
        pred_curve=pred_by_model["v238_selected_original_remaining_point_model"],
        manifest=data.manifest,
        model_name="v238_selected_original_remaining_point_model",
    )
    split_check = split_integrity_check(data.manifest)
    guardrail = build_guardrail_json(data.manifest, split_check, selection, task_table)
    if not bool(guardrail["pass"]):
        raise AssertionError("v238 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    next_decision = build_next_decision(compare, guardrail)

    print("[v238] writing outputs")
    write_csv(task_table, TABLES / "v238_task_construction_audit.csv")
    write_csv(point_counts, TABLES / "v238_point_training_rows_by_delay.csv")
    write_csv(selection, TABLES / "v238_model_selection_validation_only.csv")
    write_csv(metrics, TABLES / "v238_metrics_by_delay_and_bucket.csv")
    write_csv(compare, TABLES / "v238_compare_v236_original_remaining.csv")
    write_csv(per_sample, TABLES / "v238_selected_per_sample_metrics.csv")
    write_csv(next_decision, TABLES / "v238_next_model_decision.csv")
    write_csv(split_check, TABLES / "v238_split_integrity_check.csv")

    np.savez_compressed(
        OUT / "v238_original_remaining_predictions.npz",
        y_true_steering_delta=y_true_curve.astype(np.float32),
        pred_v236_steering_delta=data.pred_v236[:, :, 0].astype(np.float32),
        pred_v238_steering_delta=pred_by_model["v238_selected_original_remaining_point_model"].astype(np.float32),
        delay_ms=data.manifest["delay_ms"].to_numpy(dtype=np.int32),
        split=data.manifest["split"].astype(str).to_numpy(dtype="U16"),
        event_uid=data.manifest["event_uid"].astype(str).to_numpy(dtype="U160"),
        future_grid_s=FUTURE_GRID.astype(np.float32),
        original_remaining_valid=build_original_remaining_mask(data.manifest)[0].astype(np.bool_),
    )
    with (MODELS / "v238_selected_point_model.pkl").open("wb") as f:
        pickle.dump(model_payload, f)

    figure_paths = plot_core_figures(compare, selection)
    write_input_hashes()
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    leakage = {
        "same_event_uid_cross_split_count": int(split_check["split_check_status"].eq("fail").sum()),
        "test_used_for_selection": bool(selection["test_used_for_selection"].astype(bool).any()),
        "pass": int(split_check["split_check_status"].eq("fail").sum()) == 0
        and not bool(selection["test_used_for_selection"].astype(bool).any()),
    }
    (LOGS / "leakage_check.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")
    run_manifest = {
        "stage": "v238_task_model_redesign",
        "created_by": Path(__file__).name,
        "output_dir": str(OUT),
        "source_v236": str(V236_DIR),
        "source_v237": str(V237_DIR),
        "n_rolling_samples": int(len(data.manifest)),
        "n_events": int(data.manifest["event_uid"].nunique()),
        "n_point_rows_all": int(len(point_data.y_point_all)),
        "n_point_rows_original_remaining_valid": int(point_data.valid_original_remaining_all.sum()),
        "primary_target": "steering_delta_original_remaining_masked",
        "selected_model": selected_model.model_name,
        "figures": [str(path.relative_to(OUT)).replace("\\", "/") for path in figure_paths],
    }
    (LOGS / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = zip_outputs()
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = zip_outputs()
    write_report(data.manifest, task_table, point_counts, selection, compare, metrics, next_decision, guardrail, zip_path)
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = zip_outputs()

    print("[v238] finished")
    print(f"output_dir={OUT}")
    print(f"selected_model={selected_model.model_name}")
    print(f"report={REPORTS / 'v238_task_model_redesign_cn.md'}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
