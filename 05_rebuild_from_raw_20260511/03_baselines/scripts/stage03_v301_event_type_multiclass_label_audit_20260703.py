#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v301 事件类型多分类标签草稿与有效性审计。

本轮目标：
1. 给每个 delay0 事件生成一版可人工复核的事件类型草稿标签。
2. 明确区分“未来行为派生标签”和“锚点前可部署输入”。
3. 检查事件标签是否能解释 v300 的误差差异。
4. 检查这些标签能否从锚点前车辆输入预测出来。
5. 检查如果 oracle 已知标签，用标签均值残差修正 v300，理论上能改善多少。

重要边界：
- 本脚本生成的主事件标签使用了 anchor 后 0-2s 的真实车辆/轨迹行为，因此不能直接作为预测模型输入。
- 它们可以先作为人工标注草稿、分层评估标签、辅助监督目标或后续人工可知场景标签字典的原型。
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V242_SCRIPT = BASELINES / "scripts" / "stage03_v242_joint_curve_decoder_20260626.py"
V299_EVENT_TABLE = BASELINES / "v299_within_subject_split_residual_calibration_20260702" / "tables" / "v299_within_subject_split_event_table.csv"
V300_DIR = BASELINES / "v300_within_subject_full_joint_curve_train_20260702"
V300_PRED = V300_DIR / "v300_within_subject_full_predictions.npz"
V300_GUARDRAIL = V300_DIR / "logs" / "guardrail_check.json"

OUT = BASELINES / "v301_event_type_multiclass_label_audit_20260703"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
MODELS = OUT / "models"

SEED = 20260703
FUTURE_HORIZON_S = 2.0

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
    """一个原始车辆信号的相对时间和值。"""

    times: np.ndarray
    values: np.ndarray


@dataclass
class RawVehicleLite:
    """本脚本只保留事件标签需要的原始车辆信号。"""

    path: Path
    signals: Dict[str, PreparedSignal]
    encoding: str
    min_t: float
    max_t: float


def import_module_from_path(module_name: str, path: Path):
    """按文件路径导入前序脚本，复用已验证的数据读取逻辑。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V242 = import_module_from_path("stage03_v242_joint_curve_decoder_20260626_for_v301", V242_SCRIPT)
V238 = V242.V238
FUTURE_GRID = V238.FUTURE_GRID.astype(np.float32)


RAW_USECOLS = [
    "ID",
    "StorageTime",
    "zx1|v_km/h",
    "zx|v_km/h",
    "zx|BrakePedal",
    "zx|AcceleratorPedal",
    "zx|ax",
    "zx|ay",
    "zx|vyaw",
    "zx|SteeringWheel",
    "zx1|lateraldistance",
    "zx|lateraldistance",
    "zx1|lanecurvatureXY",
    "zx|lanecurvatureXY",
]


def ensure_dirs() -> None:
    """创建 v301 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v301 自己的输出。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一用 utf-8-sig，方便 Windows Excel 直接打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, object], path: Path) -> None:
    """保存 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """计算文件哈希。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_header(path: Path) -> Tuple[List[str], str]:
    """兼容原始 CSV 编码读取表头。"""

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            header = pd.read_csv(path, nrows=0, encoding=encoding).columns.tolist()
            return header, encoding
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取 CSV 表头：{path} / {last_error}")


def read_raw_vehicle_lite(path: Path) -> RawVehicleLite:
    """读取一个 recording 的原始车辆 CSV，只保留事件类型需要的信号。"""

    header, encoding = read_csv_header(path)
    usecols = [c for c in RAW_USECOLS if c in header]
    if "StorageTime" not in usecols:
        raise RuntimeError(f"原始车辆文件缺少 StorageTime：{path}")
    df = pd.read_csv(path, encoding=encoding, usecols=usecols, low_memory=False)
    df["StorageTime_dt"] = pd.to_datetime(df["StorageTime"], errors="coerce")
    df = df[df["StorageTime_dt"].notna()].copy().sort_values("StorageTime_dt")
    if df.empty:
        raise RuntimeError(f"StorageTime 解析后为空：{path}")
    start = df["StorageTime_dt"].iloc[0]
    df["t_rel_record_s"] = (df["StorageTime_dt"] - start).dt.total_seconds()
    base_times = df["t_rel_record_s"].to_numpy(dtype=float)
    signals: Dict[str, PreparedSignal] = {}
    for col in usecols:
        if col in {"ID", "StorageTime"}:
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(base_times) & np.isfinite(values)
        if not valid.any():
            continue
        times = base_times[valid]
        vals = values[valid]
        order = np.argsort(times)
        signals[col] = PreparedSignal(times=times[order], values=vals[order])
    return RawVehicleLite(
        path=path,
        signals=signals,
        encoding=encoding,
        min_t=float(np.nanmin(base_times)),
        max_t=float(np.nanmax(base_times)),
    )


def resolve_signal(raw: RawVehicleLite, candidates: Iterable[str]) -> PreparedSignal | None:
    """从候选列名中找到当前 raw 文件实际存在的信号。"""

    for col in candidates:
        if col in raw.signals:
            return raw.signals[col]
    return None


def nearest_value(signal: PreparedSignal | None, target_s: float) -> float:
    """对单个时间点做最近邻取值。"""

    if signal is None or len(signal.times) == 0 or not np.isfinite(target_s):
        return math.nan
    idx = int(np.searchsorted(signal.times, target_s))
    cand = []
    if idx < len(signal.times):
        cand.append(idx)
    if idx > 0:
        cand.append(idx - 1)
    if not cand:
        return math.nan
    best = min(cand, key=lambda i: abs(signal.times[i] - target_s))
    return float(signal.values[best])


def window_values(signal: PreparedSignal | None, start_s: float, end_s: float) -> np.ndarray:
    """取一个时间窗口内的信号值。"""

    if signal is None or len(signal.times) == 0:
        return np.array([], dtype=float)
    mask = (signal.times >= start_s - 1e-9) & (signal.times <= end_s + 1e-9)
    vals = signal.values[mask].astype(float)
    vals = vals[np.isfinite(vals)]
    return vals


def safe_stat(vals: np.ndarray, fn: str) -> float:
    """对空数组安全计算统计量。"""

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return math.nan
    if fn == "max":
        return float(np.nanmax(vals))
    if fn == "min":
        return float(np.nanmin(vals))
    if fn == "mean":
        return float(np.nanmean(vals))
    if fn == "median":
        return float(np.nanmedian(vals))
    if fn == "absmax":
        return float(np.nanmax(np.abs(vals)))
    if fn == "range":
        return float(np.nanmax(vals) - np.nanmin(vals))
    raise ValueError(fn)


def count_extrema(curve: np.ndarray) -> int:
    """粗略统计曲线局部极值数量。"""

    x = np.asarray(curve, dtype=float)
    if x.size < 3 or not np.isfinite(x).any():
        return 0
    dx = np.diff(x)
    dx[np.abs(dx) < 1e-6] = 0.0
    signs = np.sign(dx)
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def line_length(curve: np.ndarray) -> float:
    """曲线折线长度，用来描述连续修正强度。"""

    x = np.asarray(curve, dtype=float)
    if x.size < 2:
        return math.nan
    return float(np.nansum(np.abs(np.diff(x))))


def load_base_delay0_data():
    """读取 v236 rolling 数据，并同步 v299 within-subject split。"""

    data = V238.load_v236_data()
    event_table = pd.read_csv(V299_EVENT_TABLE, encoding="utf-8-sig")
    split_map = event_table.set_index("event_uid")["within_subject_split"].astype(str)
    manifest = data.manifest.copy()
    manifest["split_original_v236"] = manifest["split"].astype(str)
    manifest["within_subject_split"] = manifest["event_uid"].astype(str).map(split_map)
    if manifest["within_subject_split"].isna().any():
        raise AssertionError("v299 split 无法覆盖全部 rolling manifest")
    manifest["split"] = manifest["within_subject_split"].astype(str)
    delay0_mask = manifest["delay_ms"].astype(int).eq(0).to_numpy()
    if delay0_mask.sum() != event_table["event_uid"].nunique():
        raise AssertionError("delay0 事件数与 v299 event table 不一致")
    return data, manifest, delay0_mask, event_table


def load_v300_prediction(delay0_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, object]]:
    """读取 v300 选择模型的 delay0 预测。"""

    if not V300_PRED.exists():
        raise FileNotFoundError(f"缺少 v300 预测数组：{V300_PRED}")
    # v300 的 npz 内含 best_v300_model / event_uid 等字符串对象数组。
    # 这里读取的是本项目上一阶段本地生成的固定产物，因此允许 pickle 以恢复这些元信息。
    with np.load(V300_PRED, allow_pickle=True) as z:
        y_true = z["y_true_steering_delta"].astype(np.float32)[delay0_mask]
        pred = z["pred_v300_best_within_subject_full"].astype(np.float32)[delay0_mask]
        selected = str(z["best_v300_model"][0])
        pred_event_uid = z["event_uid"].astype(str)[delay0_mask]
    guard = json.loads(V300_GUARDRAIL.read_text(encoding="utf-8")) if V300_GUARDRAIL.exists() else {}
    return y_true, pred, selected, {"guardrail": guard, "event_uid": pred_event_uid}


def build_future_behavior_features(manifest_delay0: pd.DataFrame, y_future_delay0: np.ndarray) -> pd.DataFrame:
    """从原始车辆 CSV 和 y_future 中生成事件行为统计特征。"""

    raw_cache: Dict[str, RawVehicleLite] = {}
    rows: List[Dict[str, object]] = []
    for i, row in manifest_delay0.reset_index(drop=True).iterrows():
        raw_path = Path(str(row["raw_vehicle_csv"]))
        key = str(raw_path)
        if key not in raw_cache:
            raw_cache[key] = read_raw_vehicle_lite(raw_path)
        raw = raw_cache[key]
        anchor_s = float(row["original_anchor_s"])
        start_s = anchor_s
        end_s = anchor_s + FUTURE_HORIZON_S
        pre_start_s = anchor_s - 3.0

        speed_sig = resolve_signal(raw, ["zx1|v_km/h", "zx|v_km/h"])
        brake_sig = resolve_signal(raw, ["zx|BrakePedal"])
        accel_sig = resolve_signal(raw, ["zx|AcceleratorPedal"])
        ax_sig = resolve_signal(raw, ["zx|ax"])
        ay_sig = resolve_signal(raw, ["zx|ay"])
        yaw_sig = resolve_signal(raw, ["zx|vyaw"])
        steer_sig = resolve_signal(raw, ["zx|SteeringWheel"])
        lat_sig = resolve_signal(raw, ["zx1|lateraldistance", "zx|lateraldistance"])
        curv_sig = resolve_signal(raw, ["zx1|lanecurvatureXY", "zx|lanecurvatureXY"])

        speed_start = nearest_value(speed_sig, start_s)
        speed_end = nearest_value(speed_sig, end_s)
        speed_vals = window_values(speed_sig, start_s, end_s)
        pre_speed_vals = window_values(speed_sig, pre_start_s, start_s)
        brake_vals = window_values(brake_sig, start_s, end_s)
        pre_brake_vals = window_values(brake_sig, pre_start_s, start_s)
        accel_vals = window_values(accel_sig, start_s, end_s)
        ax_vals = window_values(ax_sig, start_s, end_s)
        ay_vals = window_values(ay_sig, start_s, end_s)
        yaw_vals = window_values(yaw_sig, start_s, end_s)
        steer_vals = window_values(steer_sig, start_s, end_s)
        lat_vals = window_values(lat_sig, start_s, end_s)
        curv_vals = window_values(curv_sig, start_s, end_s)

        lat_start = nearest_value(lat_sig, start_s)
        lat_end = nearest_value(lat_sig, end_s)
        steer_start = nearest_value(steer_sig, start_s)
        steer_end = nearest_value(steer_sig, end_s)
        curv_mean = safe_stat(curv_vals, "mean")

        steering_delta = y_future_delay0[i, :, 0].astype(float)
        steering_rate = y_future_delay0[i, :, 1].astype(float)
        roll_delta = y_future_delay0[i, :, 2].astype(float)
        ay_future = y_future_delay0[i, :, 4].astype(float)
        yaw_future = y_future_delay0[i, :, 5].astype(float)

        true_peak_idx = int(np.nanargmax(np.abs(steering_delta))) if np.isfinite(steering_delta).any() else 0
        true_peak_signed = float(steering_delta[true_peak_idx]) if np.isfinite(steering_delta).any() else math.nan
        true_direction = "right" if true_peak_signed > 0 else "left" if true_peak_signed < 0 else "flat"

        rows.append(
            {
                "event_uid": row["event_uid"],
                "subject": row["subject"],
                "recording": row["recording"],
                "split": row["split"],
                "observation_s": float(row["observation_s"]),
                "raw_vehicle_csv": str(raw_path),
                "speed_start_kmh": speed_start,
                "speed_end_kmh": speed_end,
                "speed_min_0_2s_kmh": safe_stat(speed_vals, "min"),
                "speed_max_0_2s_kmh": safe_stat(speed_vals, "max"),
                "speed_mean_pre3_0_kmh": safe_stat(pre_speed_vals, "mean"),
                "speed_drop_end_kmh": speed_start - speed_end if np.isfinite(speed_start) and np.isfinite(speed_end) else math.nan,
                "speed_drop_min_kmh": speed_start - safe_stat(speed_vals, "min") if np.isfinite(speed_start) else math.nan,
                "brake_abs_peak_0_2s": safe_stat(brake_vals, "absmax"),
                "brake_max_0_2s": safe_stat(brake_vals, "max"),
                "brake_min_0_2s": safe_stat(brake_vals, "min"),
                "brake_abs_peak_pre3_0": safe_stat(pre_brake_vals, "absmax"),
                "accelerator_abs_peak_0_2s": safe_stat(accel_vals, "absmax"),
                "ax_min_0_2s": safe_stat(ax_vals, "min"),
                "ax_abs_peak_0_2s": safe_stat(ax_vals, "absmax"),
                "ay_abs_peak_0_2s": max(safe_stat(ay_vals, "absmax"), float(np.nanmax(np.abs(ay_future)))),
                "yaw_rate_abs_peak_0_2s": max(safe_stat(yaw_vals, "absmax"), float(np.nanmax(np.abs(yaw_future)))),
                "lat_start_m": lat_start,
                "lat_end_m": lat_end,
                "lat_delta_0_2s_m": lat_end - lat_start if np.isfinite(lat_start) and np.isfinite(lat_end) else math.nan,
                "lat_abs_delta_0_2s_m": abs(lat_end - lat_start) if np.isfinite(lat_start) and np.isfinite(lat_end) else math.nan,
                "lat_range_0_2s_m": safe_stat(lat_vals, "range"),
                "curvature_mean_0_2s": curv_mean,
                "curvature_abs_mean_0_2s": float(np.nanmean(np.abs(curv_vals))) if np.isfinite(curv_vals).any() else math.nan,
                "steer_start": steer_start,
                "steer_end": steer_end,
                "steer_raw_range_0_2s": safe_stat(steer_vals, "range"),
                "true_peak_abs": float(np.nanmax(np.abs(steering_delta))),
                "true_peak_signed": true_peak_signed,
                "true_peak_time_s": float(FUTURE_GRID[true_peak_idx]),
                "true_final_delta": float(steering_delta[-1]),
                "true_direction": true_direction,
                "true_range": float(np.nanmax(steering_delta) - np.nanmin(steering_delta)),
                "true_line_length": line_length(steering_delta),
                "true_steer_rate_abs_peak": float(np.nanmax(np.abs(steering_rate))),
                "true_roll_abs_peak": float(np.nanmax(np.abs(roll_delta))),
                "true_ay_abs_peak": float(np.nanmax(np.abs(ay_future))),
                "true_yaw_rate_abs_peak": float(np.nanmax(np.abs(yaw_future))),
                "true_extrema_n": count_extrema(steering_delta),
                "true_zero_cross": bool(np.nanmin(steering_delta) < -1e-6 and np.nanmax(steering_delta) > 1e-6),
                "true_multi_correction_flag": bool(count_extrema(steering_delta) >= 2),
                "true_late_peak_flag": bool(FUTURE_GRID[true_peak_idx] >= 1.0),
                "future_feature_ok": bool(
                    np.isfinite([speed_start, speed_end, true_peak_signed, lat_start, lat_end]).sum() >= 3
                ),
            }
        )
    return pd.DataFrame(rows)


def q_train(df: pd.DataFrame, col: str, q: float, fallback: float) -> float:
    """只用 train split 估计阈值，避免用 test 分布反调规则。"""

    vals = pd.to_numeric(df.loc[df["split"].eq("train"), col], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 20:
        return float(fallback)
    out = float(np.nanquantile(vals, q))
    return out if np.isfinite(out) else float(fallback)


def build_label_thresholds(features: pd.DataFrame) -> Dict[str, float]:
    """建立可解释的事件标签阈值。"""

    return {
        "strong_steer_abs": max(2.0, q_train(features, "true_peak_abs", 0.75, 2.0)),
        "extreme_steer_abs": max(2.6, q_train(features, "true_peak_abs", 0.90, 2.6)),
        "high_steer_rate_abs": max(7.0, q_train(features, "true_steer_rate_abs_peak", 0.85, 7.0)),
        "high_yaw_abs": max(0.42, q_train(features, "true_yaw_rate_abs_peak", 0.85, 0.42)),
        "high_ay_abs": max(6.5, q_train(features, "true_ay_abs_peak", 0.85, 6.5)),
        "long_line_length": max(4.0, q_train(features, "true_line_length", 0.80, 4.0)),
        "large_lat_delta_abs": max(0.35, q_train(features, "lat_abs_delta_0_2s_m", 0.85, 0.35)),
        "large_lat_range": max(0.45, q_train(features, "lat_range_0_2s_m", 0.85, 0.45)),
        "speed_drop_large": max(3.0, q_train(features, "speed_drop_min_kmh", 0.88, 3.0)),
        "speed_drop_emergency": max(5.0, q_train(features, "speed_drop_min_kmh", 0.95, 5.0)),
        "brake_abs_peak_high": max(0.08, q_train(features, "brake_abs_peak_0_2s", 0.90, 0.08)),
        "ax_min_strong_decel": min(-1.2, q_train(features, "ax_min_0_2s", 0.10, -1.2)),
    }


def assign_event_type_labels(features: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """根据未来行为统计特征生成多标签和主多分类标签。"""

    df = features.copy()
    df["flag_strong_steer"] = df["true_peak_abs"] >= thresholds["strong_steer_abs"]
    df["flag_extreme_steer"] = df["true_peak_abs"] >= thresholds["extreme_steer_abs"]
    df["flag_high_yaw_or_ay"] = (df["true_yaw_rate_abs_peak"] >= thresholds["high_yaw_abs"]) | (
        df["true_ay_abs_peak"] >= thresholds["high_ay_abs"]
    )
    df["flag_fast_steer"] = df["true_steer_rate_abs_peak"] >= thresholds["high_steer_rate_abs"]
    df["flag_large_lateral_move"] = (df["lat_abs_delta_0_2s_m"] >= thresholds["large_lat_delta_abs"]) | (
        df["lat_range_0_2s_m"] >= thresholds["large_lat_range"]
    )
    df["flag_continuous_correction"] = (
        df["true_multi_correction_flag"].astype(bool)
        | df["true_zero_cross"].astype(bool)
        | (df["true_line_length"] >= thresholds["long_line_length"])
    )
    df["flag_late_response"] = df["true_late_peak_flag"].astype(bool)
    df["flag_speed_drop"] = df["speed_drop_min_kmh"] >= thresholds["speed_drop_large"]
    df["flag_emergency_speed_drop"] = df["speed_drop_min_kmh"] >= thresholds["speed_drop_emergency"]
    df["flag_brake_or_decel"] = (
        df["flag_speed_drop"].astype(bool)
        | (df["brake_abs_peak_0_2s"] >= thresholds["brake_abs_peak_high"])
        | (df["ax_min_0_2s"] <= thresholds["ax_min_strong_decel"])
    )
    df["flag_sharp_turn"] = df["flag_strong_steer"].astype(bool) & (
        df["flag_high_yaw_or_ay"].astype(bool) | df["flag_fast_steer"].astype(bool)
    )
    df["flag_lane_change_or_swerve"] = df["flag_large_lateral_move"].astype(bool) & (
        df["flag_sharp_turn"].astype(bool) | df["flag_continuous_correction"].astype(bool)
    )
    df["flag_emergency_lane_change"] = df["flag_lane_change_or_swerve"].astype(bool) & (
        df["flag_extreme_steer"].astype(bool) | df["flag_brake_or_decel"].astype(bool)
    )
    df["flag_compound_brake_turn"] = df["flag_brake_or_decel"].astype(bool) & df["flag_sharp_turn"].astype(bool)

    primary_labels: List[str] = []
    secondary_labels: List[str] = []
    confidence: List[str] = []
    for _, row in df.iterrows():
        labels: List[str] = []
        if bool(row["flag_brake_or_decel"]):
            labels.append("强减速/急停")
        if bool(row["flag_emergency_lane_change"]):
            labels.append("紧急连续变道/避让")
        elif bool(row["flag_lane_change_or_swerve"]):
            labels.append("连续变道/横向避让")
        if bool(row["flag_sharp_turn"]):
            labels.append("急左转" if row["true_direction"] == "left" else "急右转" if row["true_direction"] == "right" else "急转弯")
        if bool(row["flag_continuous_correction"]):
            labels.append("多段修正")
        if bool(row["flag_late_response"]):
            labels.append("晚响应/长事件")

        if bool(row["flag_compound_brake_turn"]):
            primary = "复合急制动转向"
        elif bool(row["flag_emergency_lane_change"]):
            primary = "紧急连续变道/避让"
        elif bool(row["flag_brake_or_decel"]) and not bool(row["flag_sharp_turn"]):
            primary = "强减速/急停"
        elif bool(row["flag_lane_change_or_swerve"]):
            primary = "连续变道/横向避让"
        elif bool(row["flag_sharp_turn"]):
            primary = "急左转" if row["true_direction"] == "left" else "急右转" if row["true_direction"] == "right" else "急转弯"
        elif bool(row["flag_continuous_correction"]):
            primary = "多段修正"
        elif bool(row["flag_late_response"]):
            primary = "晚响应/长事件"
        else:
            primary = "普通/轻微"

        primary_labels.append(primary)
        secondary_labels.append("|".join(dict.fromkeys(labels)) if labels else "普通/轻微")
        evidence_count = int(
            bool(row["flag_brake_or_decel"])
            + bool(row["flag_emergency_lane_change"])
            + bool(row["flag_lane_change_or_swerve"])
            + bool(row["flag_sharp_turn"])
            + bool(row["flag_continuous_correction"])
            + bool(row["flag_late_response"])
        )
        confidence.append("high" if evidence_count >= 2 else "medium" if evidence_count == 1 else "low")

    df["event_primary_type"] = primary_labels
    df["event_secondary_types"] = secondary_labels
    df["auto_label_confidence"] = confidence
    df["label_source_level"] = "future_behavior_auto_draft"
    df["deployable_as_direct_input"] = False
    df["manual_review_needed"] = (
        df["auto_label_confidence"].eq("low")
        | df["event_primary_type"].isin(["复合急制动转向", "紧急连续变道/避让"])
        | df["future_feature_ok"].eq(False)
    )
    return df


def build_delay0_preinput_matrix(data, manifest: pd.DataFrame, delay0_mask: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """构造只使用 delay0 锚点前/当前可见输入的分类特征。"""

    n = data.x_hist.shape[0]
    x_base = np.concatenate(
        [
            data.x_hist.reshape(n, -1),
            data.x_road.reshape(n, -1),
            data.x_phase.reshape(n, -1),
        ],
        axis=1,
    ).astype(np.float32)
    if x_base.shape[1] != len(data.feature_names):
        raise AssertionError("preinput feature name 数量不一致")
    return x_base[delay0_mask], list(data.feature_names)


def train_label_classifiers(labels: pd.DataFrame, x_pre: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str, object]:
    """训练多分类器，检查事件类型能否从锚点前输入预测。"""

    y = labels["event_primary_type"].astype(str).to_numpy()
    split = labels["split"].astype(str).to_numpy()
    train = split == "train"
    val = split == "val"
    test = split == "test"

    configs = [
        (
            "extra_trees_d6",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=400,
                            max_depth=6,
                            min_samples_leaf=3,
                            class_weight="balanced",
                            random_state=SEED,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "extra_trees_d10",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            max_depth=10,
                            min_samples_leaf=2,
                            class_weight="balanced",
                            random_state=SEED + 1,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "random_forest_d8",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=350,
                            max_depth=8,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            random_state=SEED + 2,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "logreg_l2_balanced",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.3,
                            max_iter=800,
                            class_weight="balanced",
                            multi_class="auto",
                            random_state=SEED + 3,
                        ),
                    ),
                ]
            ),
        ),
    ]

    rows: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []
    fitted: Dict[str, object] = {}
    for name, model in configs:
        model.fit(x_pre[train], y[train])
        fitted[name] = model
        for split_name, mask in [("train", train), ("val", val), ("test", test)]:
            pred = model.predict(x_pre[mask])
            rows.append(
                {
                    "classifier": name,
                    "split": split_name,
                    "n": int(mask.sum()),
                    "accuracy": float(accuracy_score(y[mask], pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y[mask], pred)),
                    "macro_f1": float(f1_score(y[mask], pred, average="macro", zero_division=0)),
                    "weighted_f1": float(f1_score(y[mask], pred, average="weighted", zero_division=0)),
                }
            )
            pred_frames.append(
                pd.DataFrame(
                    {
                        "event_uid": labels.loc[mask, "event_uid"].to_numpy(),
                        "split": split_name,
                        "classifier": name,
                        "true_event_primary_type": y[mask],
                        "pred_event_primary_type": pred,
                    }
                )
            )

    summary = pd.DataFrame(rows)
    val_rank = summary[summary["split"].eq("val")].sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"],
        ascending=[False, False, False],
    )
    best_name = str(val_rank.iloc[0]["classifier"])
    best_model = fitted[best_name]
    all_pred = best_model.predict(x_pre)
    pred_table = pd.concat(pred_frames, ignore_index=True)
    return summary, pred_table, all_pred.astype(str), best_name, best_model


def event_rmse(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """逐事件 RMSE。"""

    return np.sqrt(np.nanmean(np.square(pred - y_true), axis=1))


def build_residual_by_label(y_true: np.ndarray, pred_base: np.ndarray, labels: np.ndarray, train_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """用 train split 估计每个事件类型的平均残差曲线。"""

    residual = y_true - pred_base
    global_resid = np.nanmean(residual[train_mask], axis=0).astype(np.float32)
    out: Dict[str, np.ndarray] = {"__GLOBAL__": global_resid}
    for label in sorted(set(labels[train_mask].tolist())):
        mask = train_mask & (labels == label)
        if mask.sum() >= 3:
            out[label] = np.nanmean(residual[mask], axis=0).astype(np.float32)
    return out


def apply_label_residual(pred_base: np.ndarray, labels: np.ndarray, residual_by_label: Dict[str, np.ndarray], shrink: float) -> np.ndarray:
    """按标签均值残差修正曲线。"""

    pred = pred_base.copy().astype(np.float32)
    global_resid = residual_by_label["__GLOBAL__"]
    for i, label in enumerate(labels):
        pred[i] = pred[i] + float(shrink) * residual_by_label.get(str(label), global_resid)
    return pred.astype(np.float32)


def summarize_correction(
    labels: pd.DataFrame,
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_method: np.ndarray,
    method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """汇总残差修正对 all/bad/各事件标签的影响。"""

    base_rmse = event_rmse(y_true, pred_base)
    method_rmse = event_rmse(y_true, pred_method)
    event_delta = pd.DataFrame(
        {
            "event_uid": labels["event_uid"],
            "split": labels["split"],
            "event_primary_type": labels["event_primary_type"],
            "within_bad_top10_by_v249": labels["within_bad_top10_by_v249"].astype(int),
            "within_bad_top20_by_v249": labels["within_bad_top20_by_v249"].astype(int),
            "baseline_rmse": base_rmse,
            "method_rmse": method_rmse,
            "delta_vs_v300": method_rmse - base_rmse,
            "method": method,
        }
    )
    group_specs: List[Tuple[str, np.ndarray]] = [
        ("all", np.ones(len(labels), dtype=bool)),
        ("within_bad_top10", labels["within_bad_top10_by_v249"].astype(int).to_numpy() == 1),
        ("within_bad_top20", labels["within_bad_top20_by_v249"].astype(int).to_numpy() == 1),
    ]
    for label in sorted(labels["event_primary_type"].astype(str).unique()):
        group_specs.append((f"label::{label}", labels["event_primary_type"].astype(str).to_numpy() == label))
    rows: List[Dict[str, object]] = []
    split_values = labels["split"].astype(str).to_numpy()
    for split_name in ["train", "val", "test"]:
        split_mask = split_values == split_name
        for group_name, group_mask in group_specs:
            mask = split_mask & group_mask
            if not mask.any():
                rows.append(
                    {
                        "method": method,
                        "split": split_name,
                        "group": group_name,
                        "n": 0,
                        "baseline_rmse_mean": math.nan,
                        "method_rmse_mean": math.nan,
                        "delta_vs_v300_mean": math.nan,
                        "delta_vs_v300_median": math.nan,
                        "improved_rate": math.nan,
                    }
                )
                continue
            delta = method_rmse[mask] - base_rmse[mask]
            rows.append(
                {
                    "method": method,
                    "split": split_name,
                    "group": group_name,
                    "n": int(mask.sum()),
                    "baseline_rmse_mean": float(np.nanmean(base_rmse[mask])),
                    "method_rmse_mean": float(np.nanmean(method_rmse[mask])),
                    "delta_vs_v300_mean": float(np.nanmean(delta)),
                    "delta_vs_v300_median": float(np.nanmedian(delta)),
                    "improved_rate": float(np.mean(delta < 0)),
                }
            )
    return pd.DataFrame(rows), event_delta


def build_label_count_and_error_tables(labels: pd.DataFrame, y_true: np.ndarray, pred_v300: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """统计标签分布和 v300 基线误差。"""

    rmse = event_rmse(y_true, pred_v300)
    work = labels.copy()
    work["v300_rmse"] = rmse
    count_rows: List[Dict[str, object]] = []
    for split_name in ["train", "val", "test", "all"]:
        one = work if split_name == "all" else work[work["split"].eq(split_name)]
        for label, grp in one.groupby("event_primary_type"):
            count_rows.append(
                {
                    "split": split_name,
                    "event_primary_type": label,
                    "n": int(len(grp)),
                    "rate": float(len(grp) / max(1, len(one))),
                    "within_bad_top10_rate": float(grp["within_bad_top10_by_v249"].astype(int).mean()),
                    "within_bad_top20_rate": float(grp["within_bad_top20_by_v249"].astype(int).mean()),
                    "v300_rmse_mean": float(grp["v300_rmse"].mean()),
                    "v300_rmse_median": float(grp["v300_rmse"].median()),
                    "v300_rmse_p90": float(grp["v300_rmse"].quantile(0.90)),
                }
            )
    label_counts = pd.DataFrame(count_rows)

    flag_cols = [c for c in work.columns if c.startswith("flag_")]
    flag_rows: List[Dict[str, object]] = []
    for col in flag_cols:
        for split_name in ["train", "val", "test", "all"]:
            one = work if split_name == "all" else work[work["split"].eq(split_name)]
            mask = one[col].astype(bool)
            if not mask.any():
                continue
            flag_rows.append(
                {
                    "split": split_name,
                    "flag": col,
                    "n": int(mask.sum()),
                    "rate": float(mask.mean()),
                    "within_bad_top10_rate": float(one.loc[mask, "within_bad_top10_by_v249"].astype(int).mean()),
                    "v300_rmse_mean": float(one.loc[mask, "v300_rmse"].mean()),
                }
            )
    flag_summary = pd.DataFrame(flag_rows)
    return label_counts, flag_summary


def build_manual_review_pack(labels: pd.DataFrame, y_true: np.ndarray, pred_v300: np.ndarray, max_per_label: int = 8) -> pd.DataFrame:
    """输出人工复核优先表，每个标签取若干 v300 高误差事件。"""

    work = labels.copy()
    work["v300_rmse"] = event_rmse(y_true, pred_v300)
    rows: List[pd.DataFrame] = []
    for label, grp in work.sort_values("v300_rmse", ascending=False).groupby("event_primary_type"):
        take = grp.head(max_per_label).copy()
        take["review_reason"] = "top_v300_rmse_within_label"
        rows.append(take)
    bad = work[work["within_bad_top10_by_v249"].astype(int).eq(1)].sort_values("v300_rmse", ascending=False).head(60)
    bad = bad.copy()
    bad["review_reason"] = "within_bad_top10_priority"
    rows.append(bad)
    out = pd.concat(rows, ignore_index=True).drop_duplicates("event_uid", keep="first")
    keep_cols = [
        "event_uid",
        "split",
        "subject",
        "recording",
        "observation_s",
        "event_primary_type",
        "event_secondary_types",
        "auto_label_confidence",
        "manual_review_needed",
        "review_reason",
        "v300_rmse",
        "within_bad_top10_by_v249",
        "true_peak_abs",
        "true_peak_signed",
        "true_peak_time_s",
        "speed_drop_min_kmh",
        "brake_abs_peak_0_2s",
        "ax_min_0_2s",
        "lat_abs_delta_0_2s_m",
        "lat_range_0_2s_m",
        "true_line_length",
        "true_extrema_n",
        "true_direction",
    ]
    return out[[c for c in keep_cols if c in out.columns]].sort_values("v300_rmse", ascending=False)


def plot_label_distribution(label_counts: pd.DataFrame) -> Path:
    """绘制标签分布。"""

    path = FIGURES / "v301_event_type_distribution.png"
    test = label_counts[label_counts["split"].eq("all")].sort_values("n", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(test["event_primary_type"], test["n"], color="#4c78a8")
    ax.set_title("v301 自动事件类型草稿分布")
    ax.set_xlabel("事件数")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_label_rmse(label_counts: pd.DataFrame) -> Path:
    """绘制各标签 test RMSE。"""

    path = FIGURES / "v301_event_type_test_rmse.png"
    test = label_counts[label_counts["split"].eq("test")].sort_values("v300_rmse_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(test["event_primary_type"], test["v300_rmse_mean"], color="#f58518")
    ax.set_title("v301 各事件类型的 v300 test RMSE")
    ax.set_xlabel("v300 RMSE")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_confusion(labels: pd.DataFrame, pred_all: np.ndarray, best_classifier: str) -> Path:
    """绘制 test 混淆矩阵。"""

    path = FIGURES / "v301_event_type_classifier_confusion.png"
    test = labels["split"].astype(str).to_numpy() == "test"
    y_true = labels.loc[test, "event_primary_type"].astype(str).to_numpy()
    y_pred = pred_all[test].astype(str)
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"v301 事件类型分类器 test 混淆矩阵：{best_classifier}")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_residual_delta(summary: pd.DataFrame) -> Path:
    """绘制 test 关键组残差修正 delta。"""

    path = FIGURES / "v301_label_residual_correction_delta.png"
    test = summary[
        summary["split"].eq("test")
        & summary["group"].isin(["all", "within_bad_top10", "within_bad_top20"])
    ].copy()
    methods = test["method"].drop_duplicates().tolist()
    groups = ["all", "within_bad_top10", "within_bad_top20"]
    x = np.arange(len(groups), dtype=float)
    width = 0.75 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, method in enumerate(methods):
        vals = []
        for group in groups:
            one = test[test["method"].eq(method) & test["group"].eq(group)]
            vals.append(float(one["delta_vs_v300_mean"].iloc[0]) if not one.empty else math.nan)
        ax.bar(x + (j - (len(methods) - 1) / 2) * width, vals, width=width, label=method)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("RMSE delta vs v300，负值为改善")
    ax.set_title("v301 标签残差修正理论收益")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    labels: pd.DataFrame,
    label_counts: pd.DataFrame,
    clf_summary: pd.DataFrame,
    correction_summary: pd.DataFrame,
    thresholds: Dict[str, float],
    best_classifier: str,
    selected_v300: str,
    guardrail: Dict[str, object],
) -> Path:
    """生成中文报告。"""

    path = REPORTS / "v301_event_type_multiclass_label_audit_cn.md"
    test_counts = label_counts[label_counts["split"].eq("test")].sort_values("v300_rmse_mean", ascending=False)
    test_clf = clf_summary[clf_summary["split"].eq("test") & clf_summary["classifier"].eq(best_classifier)]
    val_clf = clf_summary[clf_summary["split"].eq("val") & clf_summary["classifier"].eq(best_classifier)]
    key_corr = correction_summary[
        correction_summary["split"].eq("test")
        & correction_summary["group"].isin(["all", "within_bad_top10", "within_bad_top20"])
    ].copy()
    lines = [
        "# v301 事件类型多分类标签草稿与有效性审计",
        "",
        "## 这一步做了什么",
        "",
        "本轮给 1167 个 delay0 事件生成了一版自动事件类型草稿标签，例如强减速/急停、紧急连续变道/避让、急左转、急右转、多段修正、晚响应/长事件等。",
        "",
        "这些标签主要由 anchor 后 0-2s 的真实车辆行为和真实轨迹曲线派生，因此当前不能直接当作预测输入。它们的合理用途是：人工标注草稿、分层评估、辅助监督目标、以及后续建立锚点前可知事件条件标签的候选字典。",
        "",
        f"当前 v300 参照模型：`{selected_v300}`。",
        "",
        "## 标签阈值",
        "",
        pd.DataFrame([thresholds]).to_markdown(index=False),
        "",
        "## 标签分布和误差",
        "",
        test_counts[["event_primary_type", "n", "within_bad_top10_rate", "v300_rmse_mean", "v300_rmse_p90"]].to_markdown(index=False),
        "",
        "## 锚点前输入能否预测标签",
        "",
        f"validation 选择的标签分类器：`{best_classifier}`。",
        "",
        val_clf.to_markdown(index=False),
        "",
        test_clf.to_markdown(index=False),
        "",
        "如果 test macro-F1 / balanced accuracy 较低，说明这些事件类型虽然能解释未来行为，但锚点前车辆输入并不容易提前识别它们。",
        "",
        "## 标签已知时的理论收益",
        "",
        key_corr.to_markdown(index=False),
        "",
        "解释：`oracle_true_label_residual` 使用真实事件类型，属于理论上限；`predicted_label_residual` 使用锚点前输入预测出的事件类型，更接近可部署但通常更难。",
        "",
        "## 当前判断",
        "",
        "- 多分类事件标签值得保留为人工复核和辅助监督方向。",
        "- 但当前自动标签是未来行为派生，不可直接作为正式输入。",
        "- 下一步如果要进模型，应先让用户人工复核一小批高误差样本，确认标签定义是否符合驾驶语义。",
        "- 只有能在锚点前被可靠识别的标签，才适合作为预测模型输入；否则只能作为训练辅助或报告分层。",
        "",
        "## 产物",
        "",
        "- `tables/v301_event_type_labels.csv`：每个事件的自动标签草稿。",
        "- `tables/v301_manual_review_pack.csv`：建议人工优先复核的样本。",
        "- `tables/v301_label_predictability_summary.csv`：标签可预测性。",
        "- `tables/v301_label_residual_correction_summary.csv`：标签残差修正理论收益。",
        "- `figures/v301_event_type_distribution.png`：标签分布。",
        "- `figures/v301_event_type_test_rmse.png`：各标签 test RMSE。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_file_inventory() -> pd.DataFrame:
    """记录 v301 输出文件清单。"""

    rows: List[Dict[str, object]] = []
    for file in OUT.rglob("*"):
        if file.is_file():
            rows.append(
                {
                    "relative_path": str(file.relative_to(OUT)),
                    "bytes": int(file.stat().st_size),
                    "sha256": file_sha256(file),
                }
            )
    inventory = pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)
    write_csv(inventory, LOGS / "file_inventory.csv")
    return inventory


def make_zip_package() -> Tuple[Path, bool]:
    """打包产物并做完整性检查。"""

    zip_path = OUT / "v301_event_type_multiclass_label_audit_20260703.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in OUT.rglob("*"):
            if file.is_file() and file != zip_path:
                zf.write(file, file.relative_to(OUT))
    with zipfile.ZipFile(zip_path, "r") as zf:
        ok = zf.testzip() is None
    return zip_path, bool(ok)


def main() -> None:
    start_time = time.time()
    clean_out_dir()
    np.random.seed(SEED)

    print("[v301] 读取 v236/v299/v300 数据")
    data, manifest, delay0_mask, event_table = load_base_delay0_data()
    manifest_delay0 = manifest.loc[delay0_mask].reset_index(drop=True)
    y_true_v300, pred_v300, selected_v300, v300_meta = load_v300_prediction(delay0_mask)
    y_future_delay0 = data.y_future[delay0_mask].astype(np.float32)
    if not np.array_equal(manifest_delay0["event_uid"].astype(str).to_numpy(), v300_meta["event_uid"].astype(str)):
        raise AssertionError("v300 delay0 event_uid 与当前 manifest 不一致")

    print("[v301] 从原始车辆 CSV 生成未来行为特征")
    future_features = build_future_behavior_features(manifest_delay0, y_future_delay0)
    thresholds = build_label_thresholds(future_features)
    labels = assign_event_type_labels(future_features, thresholds)

    # 合并 v299/v300 关键诊断标签，方便分层分析。
    event_meta = event_table.set_index("event_uid")
    for col in ["within_bad_top10_by_v249", "within_bad_top20_by_v249", "bad_top10", "vehicle_ambiguous"]:
        if col in event_meta.columns:
            labels[col if col.startswith("within_") else f"v299_{col}"] = labels["event_uid"].map(event_meta[col]).fillna(0).astype(int)
    labels["v300_rmse"] = event_rmse(y_true_v300, pred_v300)
    write_csv(labels, TABLES / "v301_event_type_labels.csv")
    write_csv(pd.DataFrame([thresholds]), TABLES / "v301_event_type_thresholds.csv")

    label_counts, flag_summary = build_label_count_and_error_tables(labels, y_true_v300, pred_v300)
    write_csv(label_counts, TABLES / "v301_event_type_counts_and_error.csv")
    write_csv(flag_summary, TABLES / "v301_event_type_flag_summary.csv")

    manual_pack = build_manual_review_pack(labels, y_true_v300, pred_v300, max_per_label=8)
    write_csv(manual_pack, TABLES / "v301_manual_review_pack.csv")

    print("[v301] 训练锚点前输入的事件类型分类器")
    x_pre, feature_names = build_delay0_preinput_matrix(data, manifest, delay0_mask)
    clf_summary, clf_pred_table, pred_labels, best_classifier, best_model = train_label_classifiers(labels, x_pre)
    labels["pred_event_primary_type"] = pred_labels
    write_csv(labels, TABLES / "v301_event_type_labels_with_pred.csv")
    write_csv(clf_summary, TABLES / "v301_label_predictability_summary.csv")
    write_csv(clf_pred_table, TABLES / "v301_label_classifier_predictions.csv")
    with (MODELS / "v301_best_label_classifier.pkl").open("wb") as f:
        pickle.dump(
            {
                "best_classifier": best_classifier,
                "model": best_model,
                "feature_names": feature_names,
                "thresholds": thresholds,
            },
            f,
        )

    print("[v301] 评估标签已知/标签预测的残差修正理论收益")
    split_values = labels["split"].astype(str).to_numpy()
    train_mask = split_values == "train"
    val_mask = split_values == "val"
    true_label_arr = labels["event_primary_type"].astype(str).to_numpy()
    pred_label_arr = labels["pred_event_primary_type"].astype(str).to_numpy()
    residual_by_true = build_residual_by_label(y_true_v300, pred_v300, true_label_arr, train_mask)
    residual_by_pred = build_residual_by_label(y_true_v300, pred_v300, pred_label_arr, train_mask)

    correction_summaries: List[pd.DataFrame] = []
    correction_events: List[pd.DataFrame] = []
    shrink_rows: List[Dict[str, object]] = []
    for method_base, label_arr, resid_map in [
        ("oracle_true_label_residual", true_label_arr, residual_by_true),
        ("predicted_label_residual", pred_label_arr, residual_by_pred),
    ]:
        best_shrink = 0.0
        best_val = math.inf
        for shrink in [0.0, 0.25, 0.50, 0.75, 1.0]:
            pred_corr = apply_label_residual(pred_v300, label_arr, resid_map, shrink)
            val_rmse = float(np.nanmean(event_rmse(y_true_v300[val_mask], pred_corr[val_mask])))
            shrink_rows.append({"method_base": method_base, "shrink": shrink, "val_rmse_mean": val_rmse})
            if val_rmse < best_val:
                best_val = val_rmse
                best_shrink = float(shrink)
        method_name = f"{method_base}_shrink{best_shrink:g}"
        pred_corr = apply_label_residual(pred_v300, label_arr, resid_map, best_shrink)
        summary, event_delta = summarize_correction(labels, y_true_v300, pred_v300, pred_corr, method_name)
        correction_summaries.append(summary)
        correction_events.append(event_delta)
    shrink_table = pd.DataFrame(shrink_rows)
    correction_summary = pd.concat(correction_summaries, ignore_index=True)
    correction_event_deltas = pd.concat(correction_events, ignore_index=True)
    write_csv(shrink_table, TABLES / "v301_label_residual_shrink_selection.csv")
    write_csv(correction_summary, TABLES / "v301_label_residual_correction_summary.csv")
    write_csv(correction_event_deltas, TABLES / "v301_label_residual_event_deltas.csv")

    print("[v301] 绘图和报告")
    figure_paths = [
        plot_label_distribution(label_counts),
        plot_label_rmse(label_counts),
        plot_confusion(labels, pred_labels, best_classifier),
        plot_residual_delta(correction_summary),
    ]

    input_hashes = pd.DataFrame(
        [
            {"input_name": "v242_script", "path": str(V242_SCRIPT), "sha256": file_sha256(V242_SCRIPT)},
            {"input_name": "v299_event_table", "path": str(V299_EVENT_TABLE), "sha256": file_sha256(V299_EVENT_TABLE)},
            {"input_name": "v300_predictions", "path": str(V300_PRED), "sha256": file_sha256(V300_PRED)},
            {
                "input_name": "v300_guardrail",
                "path": str(V300_GUARDRAIL),
                "sha256": file_sha256(V300_GUARDRAIL) if V300_GUARDRAIL.exists() else "",
            },
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    guardrail = {
        "pass": True,
        "version": "v301_event_type_multiclass_label_audit_20260703",
        "event_n": int(len(labels)),
        "delay0_only": True,
        "label_source_level": "future_behavior_auto_draft",
        "labels_use_future_behavior": True,
        "labels_deployable_as_direct_input_now": False,
        "manual_review_required_before_model_input": True,
        "best_label_classifier": best_classifier,
        "selected_v300_model": selected_v300,
        "v300_guardrail_pass": bool(v300_meta.get("guardrail", {}).get("pass", False)),
        "event_type_n": int(labels["event_primary_type"].nunique()),
        "manual_review_needed_n": int(labels["manual_review_needed"].astype(bool).sum()),
        "runtime_seconds": float(time.time() - start_time),
        "figure_paths": [str(p) for p in figure_paths],
    }
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_report(labels, label_counts, clf_summary, correction_summary, thresholds, best_classifier, selected_v300, guardrail)
    write_file_inventory()
    zip_path, zip_ok = make_zip_package()
    guardrail["zip_path"] = str(zip_path)
    guardrail["zip_testzip"] = bool(zip_ok)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()

    print("[v301] 完成")
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
