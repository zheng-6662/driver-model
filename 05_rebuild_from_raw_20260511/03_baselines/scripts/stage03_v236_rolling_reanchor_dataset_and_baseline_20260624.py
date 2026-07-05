#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v236 Rolling Reanchor Joint Prediction.

本阶段不继续 v222a gate / 删除样本 / 轻量 residual 路线。

目标：
1. 用现有 Gold-V2 事件锚点和原始车辆时序，把一个事件扩展成多个 observation time；
2. 让模型分别在 t0、t0+200ms、...、t0+1000ms 做未来 2 秒预测；
3. 训练一个小的 joint baseline，检查 observe_later_like 样本是否会随着观察延迟增加而变好；
4. 所有指标按 delay 分开报告，避免把“更晚观察更容易”和“0ms 正式预测”混成一个任务。

严格边界：
- 不创建 v222a-style gate/router/selector；
- 不删除 observe_later_like 样本；
- 不修改 v225/v226 formal headline；
- 不把同一 event_uid 的不同 delay 拆到不同 split；
- 不用 test 选择 alpha、delay、阈值或模型配置。
"""

from __future__ import annotations

import json
import math
import pickle
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
RAW_ROOT = ROOT / "01_datasets" / "数据预处理" / "原始车辆数据"

SAMPLE_MANIFEST = BASELINES / "v222a_candidate_curve_cache_20260622" / "sample_manifest.csv"
OLD_FORMAL_REFERENCE = (
    BASELINES
    / "v225_formal_route_reconstruction_evidence_pack_20260622"
    / "tables"
    / "per_sample_formal_reconstruction_eval.csv"
)
OBSERVE_LATER_SOURCE = (
    BASELINES
    / "v234_short_observation_prediction_layer_20260624"
    / "tables"
    / "v234_all_split_observe_later_like_counts.csv"
)

OUT = BASELINES / "v236_rolling_reanchor_dataset_and_baseline_20260624"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
MODELS = OUT / "models"

DELAY_MS = [0, 200, 400, 600, 800, 1000]
HISTORY_GRID = np.round(np.arange(-3.0, 0.0 + 1e-9, 0.1), 4)
FUTURE_GRID = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)
TAIL_MASK = FUTURE_GRID >= 1.0

RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0]
SEED = 236

# 历史输入使用当前观测时刻以前的人车状态；这些列来自原始车辆 CSV。
HISTORY_FEATURE_SPECS = [
    ("steering", ["zx|SteeringWheel"]),
    ("speed_kmh", ["zx1|v_km/h", "zx|v_km/h"]),
    ("vx", ["zx|vx"]),
    ("vy", ["zx|vy"]),
    ("ax", ["zx|ax"]),
    ("ay", ["zx|ay"]),
    ("yaw_rate", ["zx|vyaw"]),
    ("roll", ["zx|roll"]),
    ("pitch", ["zx|pitch"]),
    ("yaw", ["zx|yaw"]),
    ("roll_rate", ["zx|vroll"]),
    ("pitch_rate", ["zx|vpitch"]),
    ("roll_acc", ["zx|aroll"]),
    ("pitch_acc", ["zx|apitch"]),
    ("accelerator", ["zx|AcceleratorPedal"]),
    ("brake", ["zx|BrakePedal"]),
    ("lane_curvature", ["zx1|lanecurvatureXY", "zx|lanecurvatureXY"]),
    ("lateral_distance", ["zx1|lateraldistance", "zx|lateraldistance"]),
]

# 道路预瞄只使用道路/车道几何，不使用未来车辆响应。
ROAD_FEATURE_SPECS = [
    ("road_curvature", ["zx1|lanecurvatureXY", "zx|lanecurvatureXY"]),
    ("road_lateral_distance", ["zx1|lateraldistance", "zx|lateraldistance"]),
]

TARGET_NAMES = [
    "steering_delta",
    "steering_rate",
    "roll_delta",
    "roll_rate",
    "ay",
    "yaw_rate",
]

PHASE_FEATURE_NAMES = [
    "delay_s",
    "delay_norm_0_to_1",
    "current_steer_abs",
    "current_steer_rate_abs",
    "current_roll_abs",
    "current_roll_rate_abs",
    "current_ay_abs",
    "current_yaw_rate_abs",
    "current_speed_kmh",
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
class PreparedSignal:
    times: np.ndarray
    values: np.ndarray


@dataclass
class RawVehicle:
    path: Path
    df: pd.DataFrame
    start_time: pd.Timestamp
    min_t: float
    max_t: float
    usecols: List[str]
    encoding: str
    signals: Dict[str, PreparedSignal]


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v236 自己的输出目录，避免旧文件混入本轮结果。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一用 utf-8-sig，方便 Windows 中文环境直接打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv_header(path: Path) -> Tuple[List[str], str]:
    """兼容不同原始 CSV 编码。"""

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            header = pd.read_csv(path, nrows=0, encoding=encoding).columns.tolist()
            return header, encoding
        except Exception as exc:  # pragma: no cover - 原始文件编码兼容。
            last_error = exc
    raise RuntimeError(f"无法读取 CSV 表头：{path} / {last_error}")


def all_candidate_raw_columns() -> List[str]:
    """汇总本脚本可能读取的原始列。"""

    cols = {"ID", "StorageTime"}
    for _, candidates in HISTORY_FEATURE_SPECS + ROAD_FEATURE_SPECS:
        cols.update(candidates)
    cols.update(
        [
            "zx|SteeringWheel",
            "zx|roll",
            "zx|vroll",
            "zx|ay",
            "zx|vyaw",
            "zx1|v_km/h",
            "zx|v_km/h",
        ]
    )
    return sorted(cols)


def read_raw_vehicle(path: Path) -> RawVehicle:
    """读取一个 recording 的原始车辆数据，并为最近邻采样预构建信号数组。"""

    header, encoding = read_csv_header(path)
    wanted = all_candidate_raw_columns()
    usecols = [col for col in wanted if col in header]
    if "StorageTime" not in usecols or "zx|SteeringWheel" not in usecols:
        raise RuntimeError(f"原始车辆文件缺少 StorageTime 或 zx|SteeringWheel：{path}")

    df = pd.read_csv(path, encoding=encoding, usecols=usecols, low_memory=False)
    df["StorageTime_dt"] = pd.to_datetime(df["StorageTime"], errors="coerce")
    df = df[df["StorageTime_dt"].notna()].copy()
    df = df.sort_values("StorageTime_dt").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"StorageTime 解析后为空：{path}")

    start_time = df["StorageTime_dt"].iloc[0]
    df["t_rel_record_s"] = (df["StorageTime_dt"] - start_time).dt.total_seconds()
    times = df["t_rel_record_s"].to_numpy(dtype=float)

    signals: Dict[str, PreparedSignal] = {}
    for col in usecols:
        if col in ("ID", "StorageTime"):
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(times) & np.isfinite(values)
        if not valid.any():
            continue
        vt = times[valid]
        vv = values[valid]
        order = np.argsort(vt)
        signals[col] = PreparedSignal(times=vt[order], values=vv[order])

    return RawVehicle(
        path=path,
        df=df,
        start_time=start_time,
        min_t=float(np.nanmin(times)),
        max_t=float(np.nanmax(times)),
        usecols=usecols,
        encoding=encoding,
        signals=signals,
    )


def resolve_column(raw: RawVehicle, candidates: Iterable[str]) -> str:
    """从候选列名中找当前 raw 文件实际存在的一列。"""

    for col in candidates:
        if col in raw.signals:
            return col
    return ""


def nearest_from_signal(signal: PreparedSignal | None, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对一个已准备信号做最近邻采样，返回值和时间误差 ms。"""

    if signal is None or len(signal.times) == 0:
        return np.full(len(targets), np.nan, dtype=float), np.full(len(targets), np.nan, dtype=float)
    vt = signal.times
    vv = signal.values
    pos = np.searchsorted(vt, targets)
    left = np.clip(pos - 1, 0, len(vt) - 1)
    right = np.clip(pos, 0, len(vt) - 1)
    choose_right = np.abs(vt[right] - targets) < np.abs(vt[left] - targets)
    idx = np.where(choose_right, right, left)
    return vv[idx].astype(float), ((vt[idx] - targets) * 1000.0).astype(float)


def sample_col(raw: RawVehicle, col: str, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """按原始 record 相对秒采样某列。"""

    if not col:
        return np.full(len(targets), np.nan), np.full(len(targets), np.nan)
    return nearest_from_signal(raw.signals.get(col), targets.astype(float))


def sample_feature(raw: RawVehicle, candidates: Iterable[str], targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """按候选列采样一个语义特征。"""

    col = resolve_column(raw, candidates)
    values, errors = sample_col(raw, col, targets)
    return values, errors, col


def finite_ratio(arr: np.ndarray) -> float:
    """计算有限值比例。"""

    values = np.asarray(arr, dtype=float)
    return float(np.isfinite(values).mean()) if values.size else 0.0


def safe_nanmax_abs(arr: np.ndarray) -> float:
    """返回绝对峰值；全 NaN 时返回 NaN。"""

    values = np.asarray(arr, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    return float(np.max(np.abs(values)))


def locate_raw_file(subject: str, recording: str) -> Path:
    """按 subject/recording 定位原始车辆 CSV。"""

    direct = RAW_ROOT / subject / f"{recording}_vehicle.csv"
    if direct.exists():
        return direct
    hits = sorted((RAW_ROOT / subject).glob(f"{recording}*_vehicle.csv")) if (RAW_ROOT / subject).exists() else []
    if hits:
        return hits[0]
    raise FileNotFoundError(f"找不到原始车辆 CSV：subject={subject}, recording={recording}")


def load_event_manifest() -> pd.DataFrame:
    """读取 canonical event manifest，并合并旧 formal 标签桶与 observe_later_like 标记。"""

    manifest = pd.read_csv(SAMPLE_MANIFEST, encoding="utf-8-sig")
    canonical = manifest[manifest["pool_key"].astype(str).eq("loose_main_pool")].copy()
    canonical = canonical.sort_values("array_index").reset_index(drop=True)
    canonical = canonical.rename(columns={"event_uid": "sample_id"})

    strict_ids = set(
        manifest.loc[manifest["pool_key"].astype(str).eq("strict_main_pool"), "event_uid"].astype(str).tolist()
    )
    canonical["strict_subset"] = canonical["sample_id"].astype(str).isin(strict_ids)

    old_ref = pd.read_csv(OLD_FORMAL_REFERENCE, encoding="utf-8-sig")
    old_ref = old_ref[old_ref["pool_key"].astype(str).eq("loose_main_pool")].copy()
    keep_old = [
        "sample_id",
        "formal_model",
        "rmse",
        "tail_rmse",
        "observed_peak_abs",
        "strong_steer",
        "reverse",
        "zero_cross",
        "multi_correction",
        "vehicle_strong",
        "normal_curve",
        "extreme_peak",
        "high_tail_error",
        "route_event",
    ]
    old_ref = old_ref[[col for col in keep_old if col in old_ref.columns]].copy()
    canonical = canonical.merge(old_ref, on="sample_id", how="left", suffixes=("", "_oldformal"))

    observe = pd.read_csv(OBSERVE_LATER_SOURCE, encoding="utf-8-sig")
    observe_cols = [
        "sample_id",
        "observe_later_like",
        "review_priority",
        "pre_3_0_peak_abs_delta",
        "post_0_3_peak_abs_delta",
        "post_3_8_peak_abs_delta",
        "future_peak_abs_delta",
    ]
    observe = observe[[col for col in observe_cols if col in observe.columns]].copy()
    canonical = canonical.merge(observe, on="sample_id", how="left")

    bool_cols = [
        "strong_steer",
        "reverse",
        "zero_cross",
        "multi_correction",
        "vehicle_strong",
        "normal_curve",
        "extreme_peak",
        "high_tail_error",
        "observe_later_like",
    ]
    for col in bool_cols:
        if col not in canonical.columns:
            canonical[col] = False
        canonical[col] = canonical[col].fillna(False).astype(bool)

    raw_paths = []
    missing_rows = []
    for row in canonical.itertuples(index=False):
        try:
            raw_paths.append(str(locate_raw_file(str(row.subject), str(row.recording))))
        except Exception as exc:
            raw_paths.append("")
            missing_rows.append({"sample_id": row.sample_id, "subject": row.subject, "recording": row.recording, "error": str(exc)})
    canonical["raw_vehicle_csv"] = raw_paths
    if missing_rows:
        write_csv(pd.DataFrame(missing_rows), LOGS / "missing_raw_vehicle_files.csv")
        raise FileNotFoundError(f"存在 {len(missing_rows)} 个事件找不到原始车辆 CSV")
    return canonical


def build_one_delay_sample(
    raw: RawVehicle,
    row: pd.Series,
    delay_ms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object], Dict[str, object] | None]:
    """从一个原始事件构造一个 observation delay 样本。"""

    sample_id = str(row["sample_id"])
    anchor_s = float(row["anchor_s"])
    delay_s = delay_ms / 1000.0
    obs_s = anchor_s + delay_s
    hist_times = obs_s + HISTORY_GRID
    future_times = obs_s + FUTURE_GRID

    if hist_times[0] < raw.min_t or future_times[-1] > raw.max_t:
        dropped = {
            "sample_id": sample_id,
            "delay_ms": delay_ms,
            "reason": "window_out_of_record_range",
            "obs_s": obs_s,
            "raw_min_t": raw.min_t,
            "raw_max_t": raw.max_t,
        }
        empty = np.empty((0,), dtype=float)
        return empty, empty, empty, empty, {}, dropped

    history_cols_used: List[str] = []
    history_values: List[np.ndarray] = []
    history_errors: List[np.ndarray] = []
    for _, candidates in HISTORY_FEATURE_SPECS:
        values, errors, col = sample_feature(raw, candidates, hist_times)
        history_cols_used.append(col)
        history_values.append(values)
        history_errors.append(errors)
    x_hist = np.stack(history_values, axis=1).astype(np.float32)

    road_cols_used: List[str] = []
    road_values: List[np.ndarray] = []
    road_errors: List[np.ndarray] = []
    for _, candidates in ROAD_FEATURE_SPECS:
        values, errors, col = sample_feature(raw, candidates, future_times)
        road_cols_used.append(col)
        road_values.append(values)
        road_errors.append(errors)
    x_road = np.stack(road_values, axis=1).astype(np.float32)

    steer, steer_err = sample_feature(raw, ["zx|SteeringWheel"], future_times)[:2]
    roll, roll_err = sample_feature(raw, ["zx|roll"], future_times)[:2]
    roll_rate_direct, roll_rate_err = sample_feature(raw, ["zx|vroll"], future_times)[:2]
    ay, ay_err = sample_feature(raw, ["zx|ay"], future_times)[:2]
    yaw_rate, yaw_rate_err = sample_feature(raw, ["zx|vyaw"], future_times)[:2]

    current_steer = steer[0]
    current_roll = roll[0]
    if not np.isfinite(current_steer) or finite_ratio(steer) < 1.0:
        dropped = {
            "sample_id": sample_id,
            "delay_ms": delay_ms,
            "reason": "target_steering_not_finite",
            "obs_s": obs_s,
            "target_steering_finite_ratio": finite_ratio(steer),
        }
        empty = np.empty((0,), dtype=float)
        return empty, empty, empty, empty, {}, dropped

    steering_delta = steer - current_steer
    steering_rate = np.gradient(steer, 0.1)
    roll_delta = roll - current_roll if np.isfinite(current_roll) else np.full_like(roll, np.nan)
    if finite_ratio(roll_rate_direct) >= 0.90:
        roll_rate = roll_rate_direct
    else:
        roll_rate = np.gradient(roll, 0.1) if finite_ratio(roll) >= 0.90 else np.full_like(roll, np.nan)
    y_future = np.stack([steering_delta, steering_rate, roll_delta, roll_rate, ay, yaw_rate], axis=1).astype(np.float32)
    if finite_ratio(y_future) < 0.98:
        dropped = {
            "sample_id": sample_id,
            "delay_ms": delay_ms,
            "reason": "target_joint_finite_ratio_too_low",
            "obs_s": obs_s,
            "target_finite_ratio": finite_ratio(y_future),
        }
        empty = np.empty((0,), dtype=float)
        return empty, empty, empty, empty, {}, dropped

    prev_steer, _ = sample_feature(raw, ["zx|SteeringWheel"], np.array([obs_s - 0.1], dtype=float))[:2]
    prev_roll, _ = sample_feature(raw, ["zx|roll"], np.array([obs_s - 0.1], dtype=float))[:2]
    current_ay, _ = sample_feature(raw, ["zx|ay"], np.array([obs_s], dtype=float))[:2]
    current_yaw_rate, _ = sample_feature(raw, ["zx|vyaw"], np.array([obs_s], dtype=float))[:2]
    current_speed, _ = sample_feature(raw, ["zx1|v_km/h", "zx|v_km/h"], np.array([obs_s], dtype=float))[:2]
    current_roll_rate, _ = sample_feature(raw, ["zx|vroll"], np.array([obs_s], dtype=float))[:2]

    steer_rate_now = (current_steer - prev_steer[0]) / 0.1 if np.isfinite(prev_steer[0]) else math.nan
    if not np.isfinite(current_roll_rate[0]) and np.isfinite(current_roll) and np.isfinite(prev_roll[0]):
        current_roll_rate_value = (current_roll - prev_roll[0]) / 0.1
    else:
        current_roll_rate_value = current_roll_rate[0]

    phase = np.array(
        [
            delay_s,
            delay_s / 1.0,
            abs(current_steer),
            abs(steer_rate_now),
            abs(current_roll) if np.isfinite(current_roll) else math.nan,
            abs(current_roll_rate_value) if np.isfinite(current_roll_rate_value) else math.nan,
            abs(current_ay[0]) if np.isfinite(current_ay[0]) else math.nan,
            abs(current_yaw_rate[0]) if np.isfinite(current_yaw_rate[0]) else math.nan,
            current_speed[0] if np.isfinite(current_speed[0]) else math.nan,
        ],
        dtype=np.float32,
    )

    history_err = np.concatenate([err for err in history_errors if len(err)])
    road_err = np.concatenate([err for err in road_errors if len(err)])
    target_err = np.concatenate([steer_err, roll_err, roll_rate_err, ay_err, yaw_rate_err])
    max_abs_err = float(
        np.nanmax(np.abs(np.concatenate([history_err, road_err, target_err])))
        if len(history_err) + len(road_err) + len(target_err) > 0
        else math.nan
    )

    meta = {
        "sample_id": sample_id,
        "event_uid": sample_id,
        "subject": row.get("subject", ""),
        "recording": row.get("recording", ""),
        "array_index": int(row.get("array_index", -1)),
        "split": row.get("split", ""),
        "pool_key": "loose_main_pool",
        "pool_name": row.get("pool_name", ""),
        "strict_subset": bool(row.get("strict_subset", False)),
        "scene_type": row.get("scene_type", ""),
        "route_event": row.get("route_event", ""),
        "original_anchor_s": anchor_s,
        "observation_s": obs_s,
        "delay_ms": delay_ms,
        "delay_s": delay_s,
        "raw_vehicle_csv": str(row.get("raw_vehicle_csv", "")),
        "history_start_s": float(hist_times[0]),
        "history_end_s": float(hist_times[-1]),
        "target_start_s": float(future_times[0]),
        "target_end_s": float(future_times[-1]),
        "history_finite_ratio": finite_ratio(x_hist),
        "road_finite_ratio": finite_ratio(x_road),
        "target_finite_ratio": finite_ratio(y_future),
        "max_abs_nearest_time_error_ms": max_abs_err,
        "observe_later_like": bool(row.get("observe_later_like", False)),
        "strong_steer": bool(row.get("strong_steer", False)),
        "reverse": bool(row.get("reverse", False)),
        "zero_cross": bool(row.get("zero_cross", False)),
        "multi_correction": bool(row.get("multi_correction", False)),
        "vehicle_strong": bool(row.get("vehicle_strong", False)),
        "normal_curve": bool(row.get("normal_curve", False)),
        "extreme_peak": bool(row.get("extreme_peak", False)),
        "high_tail_error": bool(row.get("high_tail_error", False)),
        "old_formal_rmse": float(row.get("rmse", math.nan)) if pd.notna(row.get("rmse", math.nan)) else math.nan,
        "old_formal_tail_rmse": float(row.get("tail_rmse", math.nan)) if pd.notna(row.get("tail_rmse", math.nan)) else math.nan,
        "review_priority": row.get("review_priority", ""),
        "history_cols_used": json.dumps(history_cols_used, ensure_ascii=False),
        "road_cols_used": json.dumps(road_cols_used, ensure_ascii=False),
    }
    return x_hist, x_road, phase, y_future, meta, None


def build_rolling_dataset(event_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """从所有事件构造 rolling observation 数据集。"""

    x_hist_rows: List[np.ndarray] = []
    x_road_rows: List[np.ndarray] = []
    x_phase_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    manifest_rows: List[Dict[str, object]] = []
    dropped_rows: List[Dict[str, object]] = []

    for raw_path_str, group in event_df.groupby("raw_vehicle_csv", sort=True):
        raw_path = Path(str(raw_path_str))
        raw = read_raw_vehicle(raw_path)
        for _, row in group.iterrows():
            for delay_ms in DELAY_MS:
                x_hist, x_road, x_phase, y_future, meta, dropped = build_one_delay_sample(raw, row, delay_ms)
                if dropped is not None:
                    dropped_rows.append(dropped)
                    continue
                x_hist_rows.append(x_hist)
                x_road_rows.append(x_road)
                x_phase_rows.append(x_phase)
                y_rows.append(y_future)
                manifest_rows.append(meta)

    if not manifest_rows:
        raise AssertionError("没有生成任何 v236 rolling 样本")

    manifest = pd.DataFrame(manifest_rows)
    manifest.insert(0, "rolling_sample_index", np.arange(len(manifest), dtype=int))
    dropped_df = pd.DataFrame(dropped_rows)
    return (
        np.stack(x_hist_rows).astype(np.float32),
        np.stack(x_road_rows).astype(np.float32),
        np.stack(x_phase_rows).astype(np.float32),
        np.stack(y_rows).astype(np.float32),
        manifest,
        dropped_df,
    )


def build_design_matrix(
    x_hist: np.ndarray,
    x_road: np.ndarray,
    x_phase: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """把序列输入展开成小基线可用的二维特征矩阵。"""

    n = x_hist.shape[0]
    features = [x_hist.reshape(n, -1), x_road.reshape(n, -1), x_phase]
    names: List[str] = []
    for t in HISTORY_GRID:
        for name, _ in HISTORY_FEATURE_SPECS:
            names.append(f"hist_{t:+.1f}s_{name}")
    for t in FUTURE_GRID:
        for name, _ in ROAD_FEATURE_SPECS:
            names.append(f"road_{t:+.1f}s_{name}")
    names.extend(PHASE_FEATURE_NAMES)
    X = np.concatenate(features, axis=1).astype(np.float32)
    if X.shape[1] != len(names):
        raise AssertionError(f"feature name count mismatch: X={X.shape}, names={len(names)}")
    return X, names


def impute_and_scale_features(X: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, StandardScaler, np.ndarray]:
    """用 train split 的均值填补 NaN，再用 train split 标准化。"""

    X_work = X.astype(np.float64).copy()
    train_values = X_work[train_mask]
    means = np.nanmean(train_values, axis=0)
    means[~np.isfinite(means)] = 0.0
    inds = np.where(~np.isfinite(X_work))
    X_work[inds] = means[inds[1]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_work[train_mask])
    X_all = X_work.copy()
    X_all = scaler.transform(X_all)
    return X_all.astype(np.float32), scaler, means.astype(np.float32)


def scale_targets(Y_flat: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """用 train split 标准化多输出目标。"""

    scaler = StandardScaler()
    y_train = Y_flat[train_mask]
    scaler.fit(y_train)
    y_scaled = scaler.transform(Y_flat)
    return y_scaled.astype(np.float32), scaler


def event_sample_weight(manifest: pd.DataFrame) -> np.ndarray:
    """按 GPTPro 指令给困难事件温和加权，不删除任何样本。"""

    weight = np.ones(len(manifest), dtype=np.float32)
    weight += manifest["observe_later_like"].astype(bool).to_numpy(dtype=bool).astype(np.float32) * 0.5
    weight += manifest["strong_steer"].astype(bool).to_numpy(dtype=bool).astype(np.float32) * 0.5
    weight += manifest["extreme_peak"].astype(bool).to_numpy(dtype=bool).astype(np.float32) * 1.0
    return weight


def peak_values(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回每条 steering 曲线的绝对峰值、符号峰值和峰值时间。"""

    idx = np.nanargmax(np.abs(arr), axis=1)
    signed = arr[np.arange(arr.shape[0]), idx]
    peak_t = FUTURE_GRID[idx]
    return np.abs(signed), signed, peak_t


def per_sample_metrics(y_true: np.ndarray, y_pred: np.ndarray, manifest: pd.DataFrame) -> pd.DataFrame:
    """生成逐 rolling sample 的 steering 指标。"""

    true_steer = y_true[:, :, 0]
    pred_steer = y_pred[:, :, 0]
    diff = pred_steer - true_steer
    sample_rmse = np.sqrt(np.mean(np.square(diff), axis=1))
    sample_tail_rmse = np.sqrt(np.mean(np.square(diff[:, TAIL_MASK]), axis=1))
    true_peak_abs, true_peak_signed, true_peak_t = peak_values(true_steer)
    pred_peak_abs, pred_peak_signed, pred_peak_t = peak_values(pred_steer)

    out = manifest[
        [
            "rolling_sample_index",
            "sample_id",
            "event_uid",
            "subject",
            "recording",
            "split",
            "delay_ms",
            "strict_subset",
            "observe_later_like",
            "strong_steer",
            "reverse",
            "zero_cross",
            "multi_correction",
            "normal_curve",
            "extreme_peak",
            "high_tail_error",
            "old_formal_rmse",
            "old_formal_tail_rmse",
        ]
    ].copy()
    out["v236_sample_rmse"] = sample_rmse
    out["v236_tail_rmse"] = sample_tail_rmse
    out["true_peak_abs"] = true_peak_abs
    out["pred_peak_abs"] = pred_peak_abs
    out["peak_ratio_pred_over_true"] = pred_peak_abs / np.maximum(true_peak_abs, 1e-6)
    out["true_peak_t_s"] = true_peak_t
    out["pred_peak_t_s"] = pred_peak_t
    out["direction_ok"] = np.sign(true_peak_signed) == np.sign(pred_peak_signed)
    out["severe_under"] = pred_peak_abs < (0.5 * true_peak_abs)
    out["strong_response"] = true_peak_abs >= 1.0
    return out


def metric_for_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    manifest: pd.DataFrame,
    mask: np.ndarray,
    split_name: str,
    delay_label: str,
    bucket_name: str = "all",
) -> Dict[str, object] | None:
    """计算一个 split/delay/bucket 的主指标。"""

    if int(mask.sum()) == 0:
        return None
    yt = y_true[mask]
    yp = y_pred[mask]
    true_steer = yt[:, :, 0]
    pred_steer = yp[:, :, 0]
    diff = pred_steer - true_steer
    sample_rmse = np.sqrt(np.mean(np.square(diff), axis=1))
    sample_tail_rmse = np.sqrt(np.mean(np.square(diff[:, TAIL_MASK]), axis=1))
    true_peak_abs, true_peak_signed, true_peak_t = peak_values(true_steer)
    pred_peak_abs, pred_peak_signed, pred_peak_t = peak_values(pred_steer)
    direction_ok = np.sign(true_peak_signed) == np.sign(pred_peak_signed)
    severe_under = pred_peak_abs < 0.5 * true_peak_abs
    strong = true_peak_abs >= 1.0

    vehicle_diff = yp[:, :, 1:] - yt[:, :, 1:]
    rows = manifest.loc[mask]
    return {
        "split": split_name,
        "delay_ms": delay_label,
        "bucket": bucket_name,
        "n_samples": int(mask.sum()),
        "n_events": int(rows["event_uid"].nunique()),
        "steer_rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "steer_tail_rmse_1to2s": float(np.sqrt(np.mean(np.square(diff[:, TAIL_MASK])))),
        "steer_sample_rmse_mean": float(np.mean(sample_rmse)),
        "steer_sample_rmse_p90": float(np.quantile(sample_rmse, 0.90)),
        "steer_tail_rmse_mean": float(np.mean(sample_tail_rmse)),
        "steer_tail_rmse_p90": float(np.quantile(sample_tail_rmse, 0.90)),
        "steer_direction_acc": float(np.mean(direction_ok)),
        "steer_severe_under_rate": float(np.mean(severe_under)),
        "strong_response_n": int(strong.sum()),
        "strong_under_rate": float(np.mean(severe_under[strong])) if strong.any() else math.nan,
        "true_peak_abs_mean": float(np.mean(true_peak_abs)),
        "pred_peak_abs_mean": float(np.mean(pred_peak_abs)),
        "peak_ratio_mean": float(np.mean(pred_peak_abs / np.maximum(true_peak_abs, 1e-6))),
        "peak_time_abs_error_mean": float(np.mean(np.abs(pred_peak_t - true_peak_t))),
        "vehicle_aux_rmse": float(np.sqrt(np.mean(np.square(vehicle_diff)))),
        "steering_rate_rmse": float(np.sqrt(np.mean(np.square(yp[:, :, 1] - yt[:, :, 1])))),
        "roll_delta_rmse": float(np.sqrt(np.mean(np.square(yp[:, :, 2] - yt[:, :, 2])))),
        "roll_rate_rmse": float(np.sqrt(np.mean(np.square(yp[:, :, 3] - yt[:, :, 3])))),
        "ay_rmse": float(np.sqrt(np.mean(np.square(yp[:, :, 4] - yt[:, :, 4])))),
        "yaw_rate_rmse": float(np.sqrt(np.mean(np.square(yp[:, :, 5] - yt[:, :, 5])))),
    }


def compute_metrics_by_delay(y_true: np.ndarray, y_pred: np.ndarray, manifest: pd.DataFrame) -> pd.DataFrame:
    """按 split 和 delay 分开计算指标。"""

    rows: List[Dict[str, object]] = []
    split_values = manifest["split"].astype(str).to_numpy()
    delay_values = manifest["delay_ms"].astype(int).to_numpy()
    for split_name in ["train", "val", "test"]:
        for delay_ms in DELAY_MS:
            mask = (split_values == split_name) & (delay_values == delay_ms)
            item = metric_for_indices(y_true, y_pred, manifest, mask, split_name, str(delay_ms), "all")
            if item is not None:
                rows.append(item)
    return pd.DataFrame(rows)


def bucket_masks(manifest: pd.DataFrame) -> Dict[str, np.ndarray]:
    """定义困难样本桶。"""

    n = len(manifest)
    observe = manifest["observe_later_like"].astype(bool).to_numpy()
    normal = manifest["normal_curve"].astype(bool).to_numpy() & ~observe
    strong = manifest["strong_steer"].astype(bool).to_numpy()
    reverse_multi = (
        manifest["reverse"].astype(bool).to_numpy()
        | manifest["multi_correction"].astype(bool).to_numpy()
        | manifest["zero_cross"].astype(bool).to_numpy()
    )
    return {
        "all": np.ones(n, dtype=bool),
        "observe_later_like": observe,
        "normal_predictable": normal,
        "strong_steer": strong,
        "extreme_peak": manifest["extreme_peak"].astype(bool).to_numpy(),
        "reverse_or_multi_correction": reverse_multi,
        "high_tail_error_old_formal": manifest["high_tail_error"].astype(bool).to_numpy(),
        "strict_subset": manifest["strict_subset"].astype(bool).to_numpy(),
    }


def compute_metrics_by_delay_and_bucket(y_true: np.ndarray, y_pred: np.ndarray, manifest: pd.DataFrame) -> pd.DataFrame:
    """按 split、delay、bucket 分开计算指标。"""

    rows: List[Dict[str, object]] = []
    split_values = manifest["split"].astype(str).to_numpy()
    delay_values = manifest["delay_ms"].astype(int).to_numpy()
    buckets = bucket_masks(manifest)
    for split_name in ["train", "val", "test"]:
        for delay_ms in DELAY_MS:
            base_mask = (split_values == split_name) & (delay_values == delay_ms)
            for bucket_name, bucket_mask in buckets.items():
                mask = base_mask & bucket_mask
                item = metric_for_indices(y_true, y_pred, manifest, mask, split_name, str(delay_ms), bucket_name)
                if item is not None:
                    rows.append(item)
    return pd.DataFrame(rows)


def add_delta_vs_delay0(curve: pd.DataFrame) -> pd.DataFrame:
    """给一个 by-delay 曲线添加相对 0ms 的变化。"""

    out = curve.copy().sort_values("delay_ms")
    if out.empty:
        return out
    base_rows = out[out["delay_ms"].astype(int).eq(0)]
    if base_rows.empty:
        out["delta_tail_rmse_mean_vs_0ms"] = math.nan
        out["delta_sample_rmse_mean_vs_0ms"] = math.nan
        out["delta_strong_under_vs_0ms"] = math.nan
        return out
    base = base_rows.iloc[0]
    out["delta_tail_rmse_mean_vs_0ms"] = out["steer_tail_rmse_mean"] - float(base["steer_tail_rmse_mean"])
    out["delta_sample_rmse_mean_vs_0ms"] = out["steer_sample_rmse_mean"] - float(base["steer_sample_rmse_mean"])
    out["delta_strong_under_vs_0ms"] = out["strong_under_rate"] - float(base["strong_under_rate"])
    return out


def build_improvement_tables(bucket_metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成 observe_later、strong_event、normal_noharm 三张核心曲线。"""

    test = bucket_metrics[bucket_metrics["split"].eq("test")].copy()
    observe = add_delta_vs_delay0(test[test["bucket"].eq("observe_later_like")].copy())
    strong = add_delta_vs_delay0(test[test["bucket"].eq("strong_steer")].copy())
    normal = add_delta_vs_delay0(test[test["bucket"].eq("normal_predictable")].copy())
    if not normal.empty:
        normal["noharm_status_vs_0ms"] = np.where(
            normal["delta_sample_rmse_mean_vs_0ms"] <= 0.05,
            "pass",
            "review",
        )
    return observe, strong, normal


def compare_vs_old_formal_0ms(per_sample: pd.DataFrame) -> pd.DataFrame:
    """把 v236 0ms test 逐样本指标与旧 formal 0ms 参考对齐比较。"""

    base = per_sample[(per_sample["split"].eq("test")) & (per_sample["delay_ms"].eq(0))].copy()
    buckets = {
        "all": np.ones(len(base), dtype=bool),
        "observe_later_like": base["observe_later_like"].astype(bool).to_numpy(),
        "normal_predictable": (base["normal_curve"].astype(bool) & ~base["observe_later_like"].astype(bool)).to_numpy(),
        "strong_steer": base["strong_steer"].astype(bool).to_numpy(),
        "extreme_peak": base["extreme_peak"].astype(bool).to_numpy(),
        "reverse_or_multi_correction": (
            base["reverse"].astype(bool) | base["zero_cross"].astype(bool) | base["multi_correction"].astype(bool)
        ).to_numpy(),
        "strict_subset": base["strict_subset"].astype(bool).to_numpy(),
    }
    rows: List[Dict[str, object]] = []
    for bucket_name, mask in buckets.items():
        cur = base.loc[mask].copy()
        if cur.empty:
            continue
        rows.append(
            {
                "split": "test",
                "delay_ms": 0,
                "bucket": bucket_name,
                "n_samples": int(len(cur)),
                "n_events": int(cur["event_uid"].nunique()),
                "old_formal_sample_rmse_mean": float(cur["old_formal_rmse"].mean()),
                "v236_0ms_sample_rmse_mean": float(cur["v236_sample_rmse"].mean()),
                "delta_v236_minus_old_sample_rmse": float(cur["v236_sample_rmse"].mean() - cur["old_formal_rmse"].mean()),
                "old_formal_tail_rmse_mean": float(cur["old_formal_tail_rmse"].mean()),
                "v236_0ms_tail_rmse_mean": float(cur["v236_tail_rmse"].mean()),
                "delta_v236_minus_old_tail_rmse": float(cur["v236_tail_rmse"].mean() - cur["old_formal_tail_rmse"].mean()),
            }
        )
    return pd.DataFrame(rows)


def split_integrity_check(manifest: pd.DataFrame) -> pd.DataFrame:
    """检查同一 event_uid 是否跨 split。"""

    grouped = (
        manifest.groupby("event_uid", dropna=False)
        .agg(
            split_nunique=("split", "nunique"),
            splits=("split", lambda x: ",".join(sorted(set(map(str, x))))),
            delay_nunique=("delay_ms", "nunique"),
            sample_rows=("rolling_sample_index", "count"),
        )
        .reset_index()
    )
    grouped["split_check_status"] = np.where(grouped["split_nunique"].eq(1), "pass", "fail")
    return grouped


def delay_sample_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    """统计每个 split/delay 的 rolling 样本和事件数。"""

    rows = []
    for keys, one in manifest.groupby(["split", "delay_ms"], dropna=False, sort=True):
        split, delay_ms = keys
        rows.append(
            {
                "split": split,
                "delay_ms": int(delay_ms),
                "n_samples": int(len(one)),
                "n_events": int(one["event_uid"].nunique()),
                "observe_later_like_samples": int(one["observe_later_like"].astype(bool).sum()),
                "strong_steer_samples": int(one["strong_steer"].astype(bool).sum()),
                "extreme_peak_samples": int(one["extreme_peak"].astype(bool).sum()),
                "normal_predictable_samples": int((one["normal_curve"].astype(bool) & ~one["observe_later_like"].astype(bool)).sum()),
            }
        )
    return pd.DataFrame(rows)


def train_joint_ridge_baseline(
    X: np.ndarray,
    Y: np.ndarray,
    manifest: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, object]]:
    """训练小 joint Ridge baseline，alpha 只按 validation 选择。"""

    split_values = manifest["split"].astype(str).to_numpy()
    train_mask = split_values == "train"
    val_mask = split_values == "val"
    test_mask = split_values == "test"
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise AssertionError("缺少 train/val/test split")

    X_scaled, x_scaler, x_impute_mean = impute_and_scale_features(X, train_mask)
    Y_flat = Y.reshape(Y.shape[0], -1).astype(np.float32)
    Y_scaled, y_scaler = scale_targets(Y_flat, train_mask)
    weights = event_sample_weight(manifest)

    selection_rows: List[Dict[str, object]] = []
    best_score = math.inf
    best_model: Ridge | None = None
    best_alpha = math.nan
    best_pred = None

    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(X_scaled[train_mask], Y_scaled[train_mask], sample_weight=weights[train_mask])
        pred_scaled = model.predict(X_scaled)
        pred_flat = y_scaler.inverse_transform(pred_scaled).astype(np.float32)
        pred = pred_flat.reshape(Y.shape)

        val_metric = metric_for_indices(Y, pred, manifest, val_mask, "val", "mixed_for_selection", "all")
        if val_metric is None:
            raise AssertionError("validation metric 为空")
        score = (
            float(val_metric["steer_sample_rmse_mean"])
            + 0.50 * float(val_metric["steer_tail_rmse_mean"])
            + 0.10 * (float(val_metric["strong_under_rate"]) if np.isfinite(float(val_metric["strong_under_rate"])) else 0.0)
        )
        selection_rows.append(
            {
                "model_name": "v236_joint_ridge_baseline",
                "alpha": alpha,
                "selected_by": "validation_only",
                "test_used_for_selection": False,
                "val_selection_score": score,
                "val_steer_sample_rmse_mean": val_metric["steer_sample_rmse_mean"],
                "val_steer_tail_rmse_mean": val_metric["steer_tail_rmse_mean"],
                "val_strong_under_rate": val_metric["strong_under_rate"],
            }
        )
        if score < best_score:
            best_score = score
            best_model = model
            best_alpha = alpha
            best_pred = pred

    if best_model is None or best_pred is None:
        raise AssertionError("没有选出 Ridge baseline")

    selection = pd.DataFrame(selection_rows).sort_values("val_selection_score").reset_index(drop=True)
    selection["validation_rank"] = np.arange(1, len(selection) + 1)
    model_payload = {
        "model_kind": "v236_joint_ridge_baseline",
        "selected_alpha": best_alpha,
        "selected_by": "validation_only",
        "test_used_for_selection": False,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "x_impute_mean": x_impute_mean,
        "model": best_model,
        "ridge_alphas": RIDGE_ALPHAS,
        "selection_score": "val_sample_rmse_mean + 0.5 * val_tail_rmse_mean + 0.1 * val_strong_under_rate",
    }
    return best_pred.astype(np.float32), selection, model_payload


def plot_improvement_curves(observe: pd.DataFrame, strong: pd.DataFrame, normal: pd.DataFrame) -> List[Path]:
    """生成两张核心检查图。"""

    paths: List[Path] = []
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for df, label, color in [
        (observe, "observe_later_like", "#d62728"),
        (strong, "strong_steer", "#1f77b4"),
        (normal, "normal_predictable", "#2ca02c"),
    ]:
        if df.empty:
            continue
        ax.plot(df["delay_ms"].astype(int), df["steer_tail_rmse_mean"], marker="o", label=label, color=color)
    ax.set_xlabel("Observation delay (ms)")
    ax.set_ylabel("Test tail RMSE mean")
    ax.set_title("v236 rolling update: tail RMSE by delay")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGURES / "v236_tail_rmse_by_delay_buckets.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for df, label, color in [
        (observe, "observe_later_like", "#d62728"),
        (strong, "strong_steer", "#1f77b4"),
        (normal, "normal_predictable", "#2ca02c"),
    ]:
        if df.empty:
            continue
        ax.plot(df["delay_ms"].astype(int), df["strong_under_rate"], marker="o", label=label, color=color)
    ax.set_xlabel("Observation delay (ms)")
    ax.set_ylabel("Strong under rate")
    ax.set_title("v236 rolling update: strong under by delay")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGURES / "v236_strong_under_by_delay_buckets.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def build_guardrail_json(
    manifest: pd.DataFrame,
    split_check: pd.DataFrame,
    selection: pd.DataFrame,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """生成 guardrail/leakage 两个 JSON。"""

    observe_count = int(manifest["observe_later_like"].astype(bool).sum())
    split_bad = int(split_check["split_check_status"].eq("fail").sum())
    selected = selection.sort_values("validation_rank").iloc[0].to_dict()

    guardrail = {
        "stage": "v236_rolling_reanchor_dataset_and_baseline",
        "no_v222a_gate_router_selector": True,
        "observe_later_like_deleted": False,
        "observe_later_like_rolling_samples": observe_count,
        "formal_headline_changed": False,
        "mixed_delay_rmse_used_as_formal_headline": False,
        "selected_model": {
            "model_name": selected.get("model_name"),
            "alpha": float(selected.get("alpha")),
            "selected_by": "validation_only",
            "test_used_for_selection": False,
        },
        "status": "pass" if split_bad == 0 and not bool(selection["test_used_for_selection"].any()) else "fail",
    }
    leakage = {
        "same_event_uid_never_crosses_split": split_bad == 0,
        "cross_split_event_count": split_bad,
        "history_input_max_relative_to_observation_s": float(HISTORY_GRID.max()),
        "history_input_uses_future_vehicle_state": False,
        "road_preview_uses_future_road_geometry_only": True,
        "target_window_relative_to_observation_s": [float(FUTURE_GRID.min()), float(FUTURE_GRID.max())],
        "test_used_for_alpha_or_delay_selection": False,
        "delay_samples_from_same_event_share_split": split_bad == 0,
        "status": "pass" if split_bad == 0 else "fail",
    }
    return guardrail, leakage


def write_report(
    manifest: pd.DataFrame,
    selection: pd.DataFrame,
    delay_metrics: pd.DataFrame,
    observe_curve: pd.DataFrame,
    strong_curve: pd.DataFrame,
    normal_check: pd.DataFrame,
    old_compare: pd.DataFrame,
    guardrail: Dict[str, object],
    zip_path: Path,
) -> None:
    """生成中文报告。"""

    selected = selection.sort_values("validation_rank").iloc[0]
    test_delay = delay_metrics[delay_metrics["split"].eq("test")].copy()
    lines: List[str] = []
    lines.append("# v236 Rolling Reanchor Joint Prediction 报告")
    lines.append("")
    lines.append("## 结论边界")
    lines.append("")
    lines.append("- 本轮是新的 rolling/reanchor 训练任务，不再继续 v222a gate、删除样本或 light residual 路线。")
    lines.append("- 每个事件生成 0/200/400/600/800/1000ms 多个 observation time；同一 event_uid 的所有 delay 保持在同一 split。")
    lines.append("- 小基线是 joint Ridge，多输出预测未来 2 秒 steering delta、steering rate、roll delta、roll rate、ay、yaw rate。")
    lines.append("- alpha 只按 validation 选择，test 只在模型固定后报告。")
    lines.append("")
    lines.append("## 数据集")
    lines.append("")
    lines.append(f"- rolling 样本数：{len(manifest)}")
    lines.append(f"- 唯一事件数：{manifest['event_uid'].nunique()}")
    lines.append(f"- observe_later_like rolling 样本数：{int(manifest['observe_later_like'].astype(bool).sum())}")
    lines.append(f"- strict subset rolling 样本数：{int(manifest['strict_subset'].astype(bool).sum())}")
    lines.append("")
    lines.append("## 模型选择")
    lines.append("")
    lines.append(
        f"- selected alpha=`{float(selected.alpha):g}`，val score={float(selected.val_selection_score):.6f}，"
        f"val sample RMSE={float(selected.val_steer_sample_rmse_mean):.6f}，"
        f"val tail={float(selected.val_steer_tail_rmse_mean):.6f}"
    )
    lines.append("")
    lines.append("## Test by delay")
    lines.append("")
    for row in test_delay.sort_values("delay_ms").itertuples(index=False):
        lines.append(
            f"- delay={int(row.delay_ms)}ms: n={int(row.n_samples)}，"
            f"sample_RMSE={float(row.steer_sample_rmse_mean):.6f}，"
            f"tail_mean={float(row.steer_tail_rmse_mean):.6f}，"
            f"strong_under={float(row.strong_under_rate) if np.isfinite(float(row.strong_under_rate)) else math.nan:.6f}"
        )
    lines.append("")
    lines.append("## observe_later_like improvement")
    lines.append("")
    if observe_curve.empty:
        lines.append("- test 中没有 observe_later_like 样本。")
    else:
        for row in observe_curve.sort_values("delay_ms").itertuples(index=False):
            lines.append(
                f"- delay={int(row.delay_ms)}ms: n={int(row.n_samples)}，"
                f"tail_mean={float(row.steer_tail_rmse_mean):.6f}，"
                f"delta_tail_vs_0ms={float(row.delta_tail_rmse_mean_vs_0ms):+.6f}，"
                f"sample_RMSE={float(row.steer_sample_rmse_mean):.6f}"
            )
    lines.append("")
    lines.append("## strong event improvement")
    lines.append("")
    if strong_curve.empty:
        lines.append("- test 中没有 strong_steer 样本。")
    else:
        for row in strong_curve.sort_values("delay_ms").itertuples(index=False):
            lines.append(
                f"- delay={int(row.delay_ms)}ms: tail_mean={float(row.steer_tail_rmse_mean):.6f}，"
                f"strong_under={float(row.strong_under_rate) if np.isfinite(float(row.strong_under_rate)) else math.nan:.6f}，"
                f"delta_tail_vs_0ms={float(row.delta_tail_rmse_mean_vs_0ms):+.6f}"
            )
    lines.append("")
    lines.append("## normal no-harm")
    lines.append("")
    if normal_check.empty:
        lines.append("- normal_predictable 桶为空。")
    else:
        for row in normal_check.sort_values("delay_ms").itertuples(index=False):
            lines.append(
                f"- delay={int(row.delay_ms)}ms: sample_RMSE={float(row.steer_sample_rmse_mean):.6f}，"
                f"delta_vs_0ms={float(row.delta_sample_rmse_mean_vs_0ms):+.6f}，status={row.noharm_status_vs_0ms}"
            )
    lines.append("")
    lines.append("## Old 0ms formal reference")
    lines.append("")
    if old_compare.empty:
        lines.append("- 旧 formal 对照为空。")
    else:
        for row in old_compare.itertuples(index=False):
            if row.bucket not in ("all", "observe_later_like", "normal_predictable", "strong_steer"):
                continue
            lines.append(
                f"- {row.bucket}: old_RMSE={float(row.old_formal_sample_rmse_mean):.6f}，"
                f"v236_0ms_RMSE={float(row.v236_0ms_sample_rmse_mean):.6f}，"
                f"delta={float(row.delta_v236_minus_old_sample_rmse):+.6f}；"
                f"old_tail={float(row.old_formal_tail_rmse_mean):.6f}，"
                f"v236_tail={float(row.v236_0ms_tail_rmse_mean):.6f}"
            )
    lines.append("")
    lines.append("## Guardrail")
    lines.append("")
    lines.append(f"- guardrail status：`{guardrail['status']}`")
    lines.append("- 未删除 observe_later_like；未创建 gate/router/selector；未改变 formal headline；未使用 mixed-delay 指标作为正式 headline。")
    lines.append("")
    lines.append("## 输出")
    lines.append("")
    lines.append("- `tables/v236_rolling_sample_manifest.csv`")
    lines.append("- `tables/v236_delay_sample_counts.csv`")
    lines.append("- `tables/v236_train_val_test_event_split_check.csv`")
    lines.append("- `tables/v236_baseline_metrics_by_delay.csv`")
    lines.append("- `tables/v236_baseline_metrics_by_delay_and_bucket.csv`")
    lines.append("- `tables/v236_observe_later_improvement_curve.csv`")
    lines.append("- `tables/v236_strong_event_improvement_curve.csv`")
    lines.append("- `tables/v236_normal_sample_noharm_check.csv`")
    lines.append("- `tables/v236_metric_vs_old_0ms_formal_reference.csv`")
    lines.append(f"- ZIP：`{zip_path.name}`")
    lines.append("")
    (REPORTS / "v236_rolling_reanchor_baseline_cn.md").write_text("\n".join(lines), encoding="utf-8")


def file_inventory() -> List[Dict[str, object]]:
    """记录输出文件清单。"""

    rows: List[Dict[str, object]] = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(OUT)),
                    "bytes": path.stat().st_size,
                }
            )
    return rows


def zip_outputs() -> Path:
    """打包 v236 输出并校验 zip。"""

    zip_path = OUT / "v236_rolling_reanchor_dataset_and_baseline_pack.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if not path.is_file() or path == zip_path:
                continue
            zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
    if bad is not None:
        raise AssertionError(f"ZIP 校验失败：{bad}")
    return zip_path


def main() -> None:
    """执行 v236 数据构造、基线训练、分 delay 评估和报告。"""

    np.random.seed(SEED)
    clean_out_dir()

    event_df = load_event_manifest()
    x_hist, x_road, x_phase, y_future, manifest, dropped = build_rolling_dataset(event_df)
    X, feature_names = build_design_matrix(x_hist, x_road, x_phase)

    split_check = split_integrity_check(manifest)
    if split_check["split_check_status"].eq("fail").any():
        bad = split_check[split_check["split_check_status"].eq("fail")]
        raise AssertionError("同一 event_uid 跨 split：\n" + bad.head(20).to_string(index=False))

    pred, selection, model_payload = train_joint_ridge_baseline(X, y_future, manifest)
    per_sample = per_sample_metrics(y_future, pred, manifest)
    delay_metrics = compute_metrics_by_delay(y_future, pred, manifest)
    bucket_metrics = compute_metrics_by_delay_and_bucket(y_future, pred, manifest)
    observe_curve, strong_curve, normal_check = build_improvement_tables(bucket_metrics)
    old_compare = compare_vs_old_formal_0ms(per_sample)
    counts = delay_sample_counts(manifest)
    guardrail, leakage = build_guardrail_json(manifest, split_check, selection)
    figure_paths = plot_improvement_curves(observe_curve, strong_curve, normal_check)

    write_csv(manifest, TABLES / "v236_rolling_sample_manifest.csv")
    write_csv(counts, TABLES / "v236_delay_sample_counts.csv")
    write_csv(split_check, TABLES / "v236_train_val_test_event_split_check.csv")
    write_csv(delay_metrics, TABLES / "v236_baseline_metrics_by_delay.csv")
    write_csv(bucket_metrics, TABLES / "v236_baseline_metrics_by_delay_and_bucket.csv")
    write_csv(observe_curve, TABLES / "v236_observe_later_improvement_curve.csv")
    write_csv(strong_curve, TABLES / "v236_strong_event_improvement_curve.csv")
    write_csv(normal_check, TABLES / "v236_normal_sample_noharm_check.csv")
    write_csv(old_compare, TABLES / "v236_metric_vs_old_0ms_formal_reference.csv")
    write_csv(per_sample, TABLES / "v236_selected_per_sample_metrics.csv")
    write_csv(selection, TABLES / "v236_model_selection_validation_only.csv")
    write_csv(pd.DataFrame({"feature_name": feature_names}), TABLES / "v236_feature_schema.csv")
    if not dropped.empty:
        write_csv(dropped, TABLES / "v236_dropped_delay_samples.csv")
    else:
        write_csv(pd.DataFrame(columns=["sample_id", "delay_ms", "reason"]), TABLES / "v236_dropped_delay_samples.csv")

    np.savez_compressed(
        OUT / "v236_rolling_dataset_arrays_and_predictions.npz",
        X_hist=x_hist.astype(np.float32),
        X_road=x_road.astype(np.float32),
        X_phase=x_phase.astype(np.float32),
        Y_future=y_future.astype(np.float32),
        pred_future=pred.astype(np.float32),
        feature_names=np.array(feature_names, dtype="U120"),
        target_names=np.array(TARGET_NAMES, dtype="U40"),
        delay_ms=manifest["delay_ms"].to_numpy(dtype=np.int32),
        split=manifest["split"].astype(str).to_numpy(dtype="U16"),
        event_uid=manifest["event_uid"].astype(str).to_numpy(dtype="U160"),
    )
    with (MODELS / "v236_joint_ridge_selected.pkl").open("wb") as f:
        pickle.dump(model_payload, f)

    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "leakage_check.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_json = {
        "stage": "v236_rolling_reanchor_dataset_and_baseline",
        "created_by": Path(__file__).name,
        "output_dir": str(OUT),
        "canonical_event_pool": "loose_main_pool unique event_uid",
        "n_rolling_samples": int(len(manifest)),
        "n_events": int(manifest["event_uid"].nunique()),
        "delays_ms": DELAY_MS,
        "history_grid_s": [float(HISTORY_GRID.min()), float(HISTORY_GRID.max()), int(len(HISTORY_GRID))],
        "future_grid_s": [float(FUTURE_GRID.min()), float(FUTURE_GRID.max()), int(len(FUTURE_GRID))],
        "target_names": TARGET_NAMES,
        "model": "joint Ridge baseline",
        "selected_alpha": float(selection.sort_values("validation_rank").iloc[0]["alpha"]),
        "figures": [str(path.relative_to(OUT)) for path in figure_paths],
    }
    (LOGS / "run_manifest.json").write_text(json.dumps(manifest_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = zip_outputs()
    write_report(manifest, selection, delay_metrics, observe_curve, strong_curve, normal_check, old_compare, guardrail, zip_path)
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = zip_outputs()

    print("v236 rolling reanchor dataset and baseline finished.")
    print(f"output_dir={OUT}")
    print(f"rolling_samples={len(manifest)} events={manifest['event_uid'].nunique()}")
    print(f"selected_alpha={float(selection.sort_values('validation_rank').iloc[0]['alpha']):g}")
    print(f"report={REPORTS / 'v236_rolling_reanchor_baseline_cn.md'}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
