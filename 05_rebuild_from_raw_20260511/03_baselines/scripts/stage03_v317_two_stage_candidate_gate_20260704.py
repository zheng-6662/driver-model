#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第317版：二阶段候选门控校正实验。

目的：
- 固定第315版当前窗口保留清单，不再扩大过滤范围；
- 固定第316版基础预测，不重新训练主模型；
- 用锚点前车辆信号和第316版预测摘要训练轻量门控器；
- 显式构造“原预测不改、幅值缩放、时间平移、幅值加时间组合、残差原型”候选；
- 只用验证集选择方案，测试集只在验证门槛全部通过后报告。

边界：
- 训练输入不使用测试集误差；
- 训练输入不使用锚点后真实曲线；
- 由真实曲线计算的最优候选、峰值幅值、峰值时间只作为训练/验证监督与诊断；
- 第315版隔离事件不参与训练、验证选模或测试主统计。
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260704
ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "03_baselines"

V315_TABLES = BASELINES / "v315_rapid_steering_filter_reanchor_plan_20260704" / "tables"
V315_KEEP = V315_TABLES / "v315_current_window_keep_manifest.csv"
V315_ISOLATE = V315_TABLES / "v315_current_window_isolate_manifest.csv"

V316_OUT = BASELINES / "v316_filtered_current_window_coarse_scene_train_20260704"
V316_PRED = V316_OUT / "v316_filtered_current_window_predictions.npz"
V316_PER_SAMPLE = V316_OUT / "tables" / "v316_per_sample_metrics_original_remaining.csv"
V316_GUARDRAIL = V316_OUT / "logs" / "guardrail_check.json"

OUT = BASELINES / "v317_two_stage_candidate_gate_20260704"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"

BASE_MODEL_NAME = "v316_selected_base"
OLD_V307_NAME = "old_v307_reference"
V300_NAME = "v300_reference"

GRID_WINDOWS = [0.3, 0.5, 1.0, 2.0]
LOW_RISK_NOOP_MARGIN = 1.02
PEAK_ELIGIBLE_TH = 0.50
SERIOUS_UNDER_RATIO_TH = 0.65
FAST_RATE_FOR_DIRECTION_CHANGE = 0.80


@dataclass
class CandidateSet:
    """候选曲线集合。"""

    names: List[str]
    curves: np.ndarray


@dataclass
class GateResult:
    """一个门控模型和一种输出方式的验证结果。"""

    config_name: str
    output_mode: str
    pred_val: np.ndarray
    candidate_prob_val: np.ndarray
    chosen_idx_val: np.ndarray


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in [TABLES, FIGURES, REPORTS, LOGS]:
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理第317版自己的输出目录。"""

    resolved_out = OUT.resolve()
    resolved_base = BASELINES.resolve()
    if resolved_base not in resolved_out.parents:
        raise RuntimeError(f"拒绝清理非预期目录：{resolved_out}")
    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """保存中文友好的表格。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, object], path: Path) -> None:
    """保存日志。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """计算文件哈希，用于追踪输入产物。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def markdown_table(df: pd.DataFrame) -> str:
    """不依赖额外库生成报告表格。"""

    if df.empty:
        return "（空表）"
    cols = list(df.columns)

    def cell(value: object) -> str:
        if isinstance(value, float):
            if not np.isfinite(value):
                return ""
            text = f"{value:.6g}"
        else:
            text = str(value)
        return text.replace("|", "｜").replace("\n", " ")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def signed_peak(curve: np.ndarray, grid: np.ndarray) -> Tuple[float, float, float]:
    """返回有符号峰值、绝对峰值和峰值时间。"""

    arr = np.asarray(curve, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return math.nan, math.nan, math.nan
    idx = int(np.nanargmax(np.abs(arr)))
    value = float(arr[idx])
    return value, float(abs(value)), float(grid[idx])


def curve_rmse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """逐样本曲线误差。"""

    return np.sqrt(np.nanmean((np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)) ** 2, axis=1))


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """安全除法。"""

    out = np.full_like(np.asarray(num, dtype=float), np.nan, dtype=float)
    den_arr = np.asarray(den, dtype=float)
    keep = np.isfinite(num) & np.isfinite(den_arr) & (np.abs(den_arr) > 1e-9)
    out[keep] = np.asarray(num, dtype=float)[keep] / den_arr[keep]
    return out


def first_value_at_or_before(t: np.ndarray, y: np.ndarray, when: float = 0.0) -> float:
    """取指定时间之前的最后一个有效值。"""

    keep = np.isfinite(t) & np.isfinite(y) & (t <= when)
    if not keep.any():
        return math.nan
    idx = np.where(keep)[0][-1]
    return float(y[idx])


def signal_stats(t: np.ndarray, y: np.ndarray, prefix: str) -> Dict[str, float]:
    """对锚点前多个窗口提取统计量。"""

    out: Dict[str, float] = {}
    tt = np.asarray(t, dtype=float)
    yy = np.asarray(y, dtype=float)
    valid = np.isfinite(tt) & np.isfinite(yy) & (tt <= 0.0)
    out[f"{prefix}_now"] = first_value_at_or_before(tt, yy, 0.0)
    for win in GRID_WINDOWS:
        mask = valid & (tt >= -win)
        vals = yy[mask]
        ts = tt[mask]
        key = f"{prefix}_pre{str(win).replace('.', 'p')}"
        if vals.size == 0:
            out[f"{key}_mean"] = math.nan
            out[f"{key}_std"] = math.nan
            out[f"{key}_max_abs"] = math.nan
            out[f"{key}_delta"] = math.nan
            out[f"{key}_slope"] = math.nan
            out[f"{key}_abs_integral"] = math.nan
            continue
        out[f"{key}_mean"] = float(np.nanmean(vals))
        out[f"{key}_std"] = float(np.nanstd(vals))
        out[f"{key}_max_abs"] = float(np.nanmax(np.abs(vals)))
        out[f"{key}_delta"] = float(vals[-1] - vals[0]) if vals.size >= 2 else 0.0
        if vals.size >= 3 and np.nanmax(ts) > np.nanmin(ts):
            try:
                out[f"{key}_slope"] = float(np.polyfit(ts, vals, 1)[0])
            except Exception:
                out[f"{key}_slope"] = math.nan
        else:
            out[f"{key}_slope"] = math.nan
        if vals.size >= 2:
            out[f"{key}_abs_integral"] = float(np.trapz(np.abs(vals), ts))
        else:
            out[f"{key}_abs_integral"] = 0.0
    return out


def direction_change_count(rate: np.ndarray) -> int:
    """统计锚点前方向盘快速左右变化次数。"""

    rr = np.asarray(rate, dtype=float)
    keep = np.isfinite(rr) & (np.abs(rr) >= FAST_RATE_FOR_DIRECTION_CHANGE)
    signs = np.sign(rr[keep])
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def sign_consistency(rate: np.ndarray) -> float:
    """方向盘速度符号一致性，越接近1表示单向打得越一致。"""

    rr = np.asarray(rate, dtype=float)
    keep = np.isfinite(rr) & (np.abs(rr) >= 1e-6)
    signs = np.sign(rr[keep])
    if signs.size == 0:
        return math.nan
    return float(abs(np.sum(signs)) / signs.size)


def smooth_and_rate(t: np.ndarray, steering: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """平滑方向盘角，计算角速度和角加速度。"""

    raw = pd.DataFrame({"t": t, "steering": steering})
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    raw = raw.drop_duplicates("t", keep="first").sort_values("t")
    if len(raw) < 5:
        return np.array([]), np.array([]), np.array([]), np.array([])
    tt = raw["t"].to_numpy(dtype=float)
    yy = raw["steering"].to_numpy(dtype=float)
    win = 11 if len(yy) >= 11 else max(3, len(yy) // 2 * 2 + 1)
    smooth = pd.Series(yy).rolling(window=win, center=True, min_periods=1).mean().to_numpy(dtype=float)
    if np.any(np.diff(tt) <= 0):
        return np.array([]), np.array([]), np.array([]), np.array([])
    rate = np.gradient(smooth, tt)
    acc = np.gradient(rate, tt)
    for arr in [rate, acc]:
        finite = np.isfinite(arr)
        if finite.any():
            q = np.nanpercentile(np.abs(arr[finite]), 99.5)
            if np.isfinite(q) and q > 0:
                arr[:] = np.clip(arr, -q, q)
    return tt, smooth, rate, acc


class RawVehicleCache:
    """缓存原始车辆表，避免重复读取。"""

    def __init__(self) -> None:
        self.cache: Dict[str, pd.DataFrame] = {}

    def load(self, path_text: object) -> pd.DataFrame:
        path = Path(str(path_text))
        key = str(path)
        if key in self.cache:
            return self.cache[key]
        needed = [
            "StorageTime",
            "zx|SteeringWheel",
            "zx|ay",
            "zx|vyaw",
            "zx|roll",
            "zx1|v_km/h",
            "zx|BrakePedal",
            "zx|AcceleratorPedal",
        ]
        if not path.exists():
            self.cache[key] = pd.DataFrame()
            return self.cache[key]
        try:
            df = pd.read_csv(path, usecols=lambda c: c in needed, low_memory=False)
        except Exception:
            self.cache[key] = pd.DataFrame()
            return self.cache[key]
        if df.empty or "StorageTime" not in df.columns:
            self.cache[key] = pd.DataFrame()
            return self.cache[key]
        t = pd.to_datetime(df["StorageTime"], errors="coerce")
        if t.isna().all():
            self.cache[key] = pd.DataFrame()
            return self.cache[key]
        first = t.dropna().iloc[0]
        df["record_s"] = (t - first).dt.total_seconds()
        self.cache[key] = df
        return df


def extract_pre_anchor_features(row: pd.Series, raw_cache: RawVehicleCache) -> Dict[str, float]:
    """只使用锚点前信号提取门控输入特征。"""

    out: Dict[str, float] = {"raw_pre_available": 0.0}
    raw = raw_cache.load(row.get("raw_vehicle_csv", ""))
    obs = float(row.get("observation_s", math.nan))
    if raw.empty or not np.isfinite(obs):
        return out
    rel_all = raw["record_s"].to_numpy(dtype=float) - obs

    if "zx|SteeringWheel" in raw.columns:
        steering_raw = raw["zx|SteeringWheel"].to_numpy(dtype=float)
        mask = np.isfinite(rel_all) & np.isfinite(steering_raw) & (rel_all >= -2.0) & (rel_all <= 0.0)
        t, steering, rate, acc = smooth_and_rate(rel_all[mask], steering_raw[mask])
        if t.size > 0:
            out["raw_pre_available"] = 1.0
            out.update(signal_stats(t, steering, "steer"))
            out.update(signal_stats(t, rate, "steer_rate"))
            out.update(signal_stats(t, acc, "steer_acc"))
            for win in [0.3, 0.5, 1.0]:
                m = (t >= -win) & (t <= 0.0) & np.isfinite(rate)
                vals = rate[m]
                out[f"steer_rate_pre{str(win).replace('.', 'p')}_sign_consistency"] = sign_consistency(vals)
                out[f"steer_rate_pre{str(win).replace('.', 'p')}_direction_changes"] = float(direction_change_count(vals))

    signal_cols = {
        "zx|ay": "ay",
        "zx|vyaw": "yaw_rate",
        "zx|roll": "roll",
        "zx1|v_km/h": "speed",
        "zx|BrakePedal": "brake",
        "zx|AcceleratorPedal": "throttle",
    }
    for col, prefix in signal_cols.items():
        if col not in raw.columns:
            continue
        yy = raw[col].to_numpy(dtype=float)
        mask = np.isfinite(rel_all) & np.isfinite(yy) & (rel_all >= -2.0) & (rel_all <= 0.0)
        if mask.sum() >= 2:
            out.update(signal_stats(rel_all[mask], yy[mask], prefix))
    return out


def prediction_features(y0: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    """提取第316版基础预测摘要，作为可部署输入。"""

    rows: List[Dict[str, float]] = []
    for curve in y0:
        signed, amp, peak_t = signed_peak(curve, grid)
        deriv = np.gradient(curve, grid)
        row = {
            "pred_base_signed_peak": signed,
            "pred_base_peak_abs": amp,
            "pred_base_peak_t": peak_t,
            "pred_base_end": float(curve[-1]),
            "pred_base_min": float(np.nanmin(curve)),
            "pred_base_max": float(np.nanmax(curve)),
            "pred_base_energy": float(np.nanmean(curve ** 2)),
            "pred_base_first_0p3_delta": float(np.interp(0.3, grid, curve) - curve[0]),
            "pred_base_first_0p5_delta": float(np.interp(0.5, grid, curve) - curve[0]),
            "pred_base_max_deriv_abs": float(np.nanmax(np.abs(deriv))),
            "pred_base_deriv_start": float(deriv[0]),
            "pred_base_deriv_mean_abs": float(np.nanmean(np.abs(deriv))),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def load_delay0_dataset() -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """读取第315清单和第316预测包，并固定第317版样本边界。"""

    if not V315_KEEP.exists():
        raise FileNotFoundError(f"缺少第315版保留清单：{V315_KEEP}")
    if not V315_ISOLATE.exists():
        raise FileNotFoundError(f"缺少第315版隔离清单：{V315_ISOLATE}")
    if not V316_PRED.exists():
        raise FileNotFoundError(f"缺少第316版预测包：{V316_PRED}")
    if not V316_PER_SAMPLE.exists():
        raise FileNotFoundError(f"缺少第316版逐样本指标：{V316_PER_SAMPLE}")

    keep_manifest = pd.read_csv(V315_KEEP, encoding="utf-8-sig")
    isolate_manifest = pd.read_csv(V315_ISOLATE, encoding="utf-8-sig")
    z = np.load(V316_PRED, allow_pickle=True)
    pred_df = pd.DataFrame(
        {
            "array_idx": np.arange(len(z["event_uid"])),
            "event_uid": z["event_uid"].astype(str),
            "split": z["split"].astype(str),
            "delay_ms": z["delay_ms"].astype(int),
            "subject": z["subject"].astype(str),
            "v315_current_window_train_keep": z["v315_current_window_train_keep"].astype(bool),
            "coarse_scene_label_npz": z["coarse_scene_label"].astype(str),
        }
    )
    delay0 = pred_df[pred_df["delay_ms"].eq(0) & pred_df["v315_current_window_train_keep"]].copy()
    keep_cols = [
        "event_uid",
        "raw_vehicle_csv",
        "observation_s",
        "coarse_scene_label",
        "scene_type",
        "route_event",
        "strong_steer",
        "vehicle_strong",
        "within_bad_top10_by_v249",
        "within_bad_top20_by_v249",
        "is_v309_severe",
        "is_user_screenshot_case",
    ]
    keep_cols = [c for c in keep_cols if c in keep_manifest.columns]
    delay0 = delay0.merge(keep_manifest[keep_cols], on="event_uid", how="left", validate="one_to_one")
    if delay0["raw_vehicle_csv"].isna().any():
        missing = delay0.loc[delay0["raw_vehicle_csv"].isna(), "event_uid"].head(5).tolist()
        raise AssertionError(f"第316预测包与第315保留清单未对齐：{missing}")

    selected_name = str(z["best_v316_model"][0])
    per_sample = pd.read_csv(V316_PER_SAMPLE, encoding="utf-8-sig")
    flags = per_sample[
        per_sample["model_name"].eq(selected_name) & per_sample["delay_ms"].eq(0)
    ].copy()
    flag_cols = [
        "event_uid",
        "sample_rmse",
        "tail_rmse",
        "true_peak_abs",
        "pred_peak_abs",
        "peak_ratio",
        "true_peak_t",
        "pred_peak_t",
        "strong_under",
        "strong_steer",
        "within_bad_top10_by_v249",
        "within_bad_top20_by_v249",
        "v299_vehicle_ambiguous",
    ]
    flag_cols = [c for c in flag_cols if c in flags.columns]
    flags = flags[flag_cols].rename(
        columns={
            "sample_rmse": "v316_sample_rmse",
            "tail_rmse": "v316_tail_rmse",
            "true_peak_abs": "v316_true_peak_abs_metric",
            "pred_peak_abs": "v316_pred_peak_abs_metric",
            "peak_ratio": "v316_peak_ratio_metric",
            "true_peak_t": "v316_true_peak_t_metric",
            "pred_peak_t": "v316_pred_peak_t_metric",
            "strong_under": "v316_strong_under",
            "strong_steer": "strong_steer_metric",
            "within_bad_top10_by_v249": "within_bad_top10_metric",
            "within_bad_top20_by_v249": "within_bad_top20_metric",
        }
    )
    delay0 = delay0.merge(flags, on="event_uid", how="left", validate="one_to_one")

    for col in ["strong_steer", "within_bad_top10_by_v249", "within_bad_top20_by_v249"]:
        metric_col = {
            "strong_steer": "strong_steer_metric",
            "within_bad_top10_by_v249": "within_bad_top10_metric",
            "within_bad_top20_by_v249": "within_bad_top20_metric",
        }[col]
        if metric_col in delay0.columns:
            delay0[col] = delay0[col].fillna(delay0[metric_col])
    for col in [
        "strong_steer",
        "vehicle_strong",
        "within_bad_top10_by_v249",
        "within_bad_top20_by_v249",
        "is_v309_severe",
        "is_user_screenshot_case",
        "v316_strong_under",
    ]:
        if col in delay0.columns:
            delay0[col] = delay0[col].fillna(False).astype(bool)
        else:
            delay0[col] = False

    split_counts = delay0.groupby("split")["event_uid"].nunique().to_dict()
    expected = {"train": 650, "val": 211, "test": 222}
    if split_counts != expected:
        raise AssertionError(f"第317版样本边界不符合目标：实际{split_counts}，期望{expected}")
    if len(keep_manifest) != 1083 or len(isolate_manifest) != 84:
        raise AssertionError(
            f"第315版清单数量异常：保留{len(keep_manifest)}，隔离{len(isolate_manifest)}"
        )

    idx = delay0["array_idx"].to_numpy(dtype=int)
    arrays = {
        "grid": z["future_grid_s"].astype(float),
        "y_true": z["y_true_steering_delta"][idx].astype(float),
        "y300": z["pred_v300_reference"][idx].astype(float),
        "y307": z["pred_v307_previous"][idx].astype(float),
        "y316": z["pred_v316_selected"][idx].astype(float),
    }
    delay0 = delay0.reset_index(drop=True)
    return delay0, arrays


def build_feature_matrix(meta: pd.DataFrame, y0: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    """构建门控输入矩阵，只含可部署输入。"""

    raw_cache = RawVehicleCache()
    rows: List[Dict[str, float]] = []
    for _, row in meta.iterrows():
        rows.append(extract_pre_anchor_features(row, raw_cache))
    raw_feat = pd.DataFrame(rows)
    pred_feat = prediction_features(y0, grid)

    scene = meta.get("coarse_scene_label", pd.Series(["unknown"] * len(meta))).fillna("unknown").astype(str)
    scene_dummies = pd.get_dummies(scene, prefix="scene")

    features = pd.concat([raw_feat.reset_index(drop=True), pred_feat.reset_index(drop=True), scene_dummies.reset_index(drop=True)], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def shift_curves(curves: np.ndarray, grid: np.ndarray, shift_s: float) -> np.ndarray:
    """时间平移候选。负值表示提前，正值表示延后。"""

    out = np.zeros_like(curves)
    query = grid - shift_s
    for i, curve in enumerate(curves):
        out[i] = np.interp(query, grid, curve, left=curve[0], right=curve[-1])
    return out


def build_residual_prototypes(y_true_train: np.ndarray, y0_train: np.ndarray, grid: np.ndarray, meta_train: pd.DataFrame) -> np.ndarray:
    """只用训练集残差构造少量峰值残差原型。"""

    true_signed = np.array([signed_peak(c, grid)[0] for c in y_true_train], dtype=float)
    true_amp = np.abs(true_signed)
    sign = np.sign(true_signed)
    sign[sign == 0] = 1.0
    high_risk = (
        meta_train["strong_steer"].astype(bool).to_numpy()
        | meta_train["within_bad_top20_by_v249"].astype(bool).to_numpy()
        | (true_amp >= 0.80)
    )
    residual = (y_true_train - y0_train) * sign[:, None]
    residual = residual[high_risk]
    if residual.shape[0] < 12:
        return np.zeros((4, len(grid)), dtype=float)
    n_clusters = min(4, residual.shape[0])
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    km.fit(residual)
    centers = km.cluster_centers_
    if n_clusters < 4:
        pad = np.zeros((4 - n_clusters, len(grid)), dtype=float)
        centers = np.vstack([centers, pad])
    return centers[:4].astype(float)


def build_candidates(y0: np.ndarray, grid: np.ndarray, prototypes: np.ndarray) -> CandidateSet:
    """围绕第316版基础曲线构造固定候选库。"""

    names: List[str] = []
    curves: List[np.ndarray] = []

    def add(name: str, arr: np.ndarray) -> None:
        names.append(name)
        curves.append(arr.astype(float))

    add("原预测不改", y0)
    for scale in [0.85, 1.15, 1.30, 1.50, 1.75]:
        add(f"幅值乘{scale:.2f}", y0 * scale)
    for shift_s in [-0.40, -0.25, -0.10, 0.10, 0.25, 0.40]:
        label = "提前" if shift_s < 0 else "延后"
        add(f"{label}{abs(shift_s):.2f}秒", shift_curves(y0, grid, shift_s))
    for scale, shift_s in [(1.30, -0.25), (1.30, 0.25), (1.50, -0.25), (1.50, 0.25)]:
        label = "提前" if shift_s < 0 else "延后"
        add(f"幅值乘{scale:.2f}+{label}{abs(shift_s):.2f}秒", shift_curves(y0 * scale, grid, shift_s))

    pred_signed = np.array([signed_peak(c, grid)[0] for c in y0], dtype=float)
    pred_sign = np.sign(pred_signed)
    pred_sign[pred_sign == 0] = 1.0
    for k in range(4):
        proto = prototypes[k][None, :] * pred_sign[:, None]
        add(f"残差原型{k + 1}", y0 + 0.75 * proto)

    return CandidateSet(names=names, curves=np.stack(curves, axis=1))


def candidate_errors(candidates: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """计算每个候选的逐样本误差。"""

    diff = candidates - y_true[:, None, :]
    return np.sqrt(np.nanmean(diff ** 2, axis=2))


def make_oracle_target(errors: np.ndarray, y_true: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """训练门控目标：候选最优，同时对低幅值且原预测接近最优的样本优先不改。"""

    best = np.nanmin(errors, axis=1)
    target = np.nanargmin(errors, axis=1).astype(int)
    true_amp = np.array([signed_peak(c, grid)[1] for c in y_true], dtype=float)
    close_noop = errors[:, 0] <= (best * LOW_RISK_NOOP_MARGIN + 1e-8)
    low_amp = true_amp < 0.80
    target[close_noop & low_amp] = 0
    return target


def align_feature_columns(train_x: pd.DataFrame, other_x: pd.DataFrame) -> pd.DataFrame:
    """保证验证/测试特征列与训练一致。"""

    aligned = other_x.copy()
    for col in train_x.columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    return aligned[list(train_x.columns)]


def train_gate_models(x_train: pd.DataFrame, target: np.ndarray, sample_weight: np.ndarray) -> Dict[str, object]:
    """训练少量轻量门控模型。"""

    models: Dict[str, object] = {}
    models["树模型浅层"] = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=450,
                    max_depth=5,
                    min_samples_leaf=4,
                    random_state=SEED,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    models["树模型中层"] = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=550,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=SEED + 1,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    models["随机森林"] = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=450,
                    max_depth=7,
                    min_samples_leaf=4,
                    random_state=SEED + 2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    models["线性门控"] = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1200,
                    C=0.70,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )
    fitted: Dict[str, object] = {}
    for name, model in models.items():
        model.fit(x_train, target, model__sample_weight=sample_weight)
        fitted[name] = model
    return fitted


def predict_candidate_prob(model: object, x: pd.DataFrame, n_candidates: int) -> np.ndarray:
    """把分类器概率映射回固定候选编号。"""

    prob_small = model.predict_proba(x)
    classes = model.named_steps["model"].classes_.astype(int)
    prob = np.zeros((len(x), n_candidates), dtype=float)
    for j, cls in enumerate(classes):
        if 0 <= cls < n_candidates:
            prob[:, cls] = prob_small[:, j]
    row_sum = prob.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    return prob / row_sum


def weighted_prediction(candidates: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """候选加权输出。"""

    return np.einsum("nc,nct->nt", prob, candidates)


def top1_prediction(candidates: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """候选单选输出。"""

    return candidates[np.arange(candidates.shape[0]), idx, :]


def peak_arrays(curves: np.ndarray, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量峰值。"""

    signed = []
    amp = []
    t = []
    for curve in curves:
        s, a, tt = signed_peak(curve, grid)
        signed.append(s)
        amp.append(a)
        t.append(tt)
    return np.asarray(signed), np.asarray(amp), np.asarray(t)


def build_per_sample_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    meta: pd.DataFrame,
    grid: np.ndarray,
    method_name: str,
    baseline_rmse: np.ndarray,
    candidate_prob: np.ndarray | None = None,
    chosen_idx: np.ndarray | None = None,
    candidate_names: List[str] | None = None,
) -> pd.DataFrame:
    """逐样本指标，用于验证门槛和诊断。"""

    true_signed, true_amp, true_t = peak_arrays(true, grid)
    pred_signed, pred_amp, pred_t = peak_arrays(pred, grid)
    rmse = curve_rmse(pred, true)
    ratio = safe_ratio(pred_amp, true_amp)
    eligible = true_amp >= PEAK_ELIGIBLE_TH
    serious_under = eligible & np.isfinite(ratio) & (ratio < SERIOUS_UNDER_RATIO_TH)
    serious_over = eligible & np.isfinite(ratio) & (ratio > 1.50)
    out = pd.DataFrame(
        {
            "method_name": method_name,
            "event_uid": meta["event_uid"].astype(str).to_numpy(),
            "split": meta["split"].astype(str).to_numpy(),
            "sample_rmse": rmse,
            "baseline_v316_rmse": baseline_rmse,
            "delta_vs_v316": rmse - baseline_rmse,
            "relative_delta_vs_v316": safe_ratio(rmse - baseline_rmse, baseline_rmse),
            "degrade_gt10": rmse > baseline_rmse * 1.10,
            "true_peak_signed": true_signed,
            "pred_peak_signed": pred_signed,
            "true_peak_abs": true_amp,
            "pred_peak_abs": pred_amp,
            "peak_ratio": ratio,
            "true_peak_t": true_t,
            "pred_peak_t": pred_t,
            "peak_time_abs_error": np.abs(pred_t - true_t),
            "peak_eligible": eligible,
            "serious_under": serious_under,
            "serious_over": serious_over,
            "strong_steer": meta["strong_steer"].astype(bool).to_numpy(),
            "within_bad_top10_by_v249": meta["within_bad_top10_by_v249"].astype(bool).to_numpy(),
            "within_bad_top20_by_v249": meta["within_bad_top20_by_v249"].astype(bool).to_numpy(),
            "normal_group": (~meta["strong_steer"].astype(bool) & ~meta["within_bad_top20_by_v249"].astype(bool)).to_numpy(),
            "is_v309_severe": meta["is_v309_severe"].astype(bool).to_numpy(),
        }
    )
    if candidate_prob is not None:
        out["gate_score"] = 1.0 - candidate_prob[:, 0]
        out["noop_probability"] = candidate_prob[:, 0]
    if chosen_idx is not None:
        out["chosen_candidate_idx"] = chosen_idx.astype(int)
        if candidate_names is not None:
            out["chosen_candidate_name"] = [candidate_names[i] for i in chosen_idx]
        out["top1_noop"] = chosen_idx.astype(int) == 0
    return out


def group_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """固定验证分组。"""

    return {
        "全部样本": np.ones(len(df), dtype=bool),
        "普通样本": df["normal_group"].astype(bool).to_numpy(),
        "强方向盘样本": df["strong_steer"].astype(bool).to_numpy(),
        "困难前20": df["within_bad_top20_by_v249"].astype(bool).to_numpy(),
        "困难前10": df["within_bad_top10_by_v249"].astype(bool).to_numpy(),
        "原严重样本": df["is_v309_severe"].astype(bool).to_numpy(),
    }


def summarize_groups(per_sample: pd.DataFrame) -> pd.DataFrame:
    """生成分组摘要。"""

    rows: List[Dict[str, object]] = []
    for (method, split), part in per_sample.groupby(["method_name", "split"], dropna=False):
        masks = group_masks(part)
        for group, mask in masks.items():
            sub = part.loc[mask].copy()
            if sub.empty:
                continue
            eligible = sub["peak_eligible"].astype(bool)
            row = {
                "method_name": method,
                "split": split,
                "group": group,
                "n": int(len(sub)),
                "sample_rmse_mean": float(sub["sample_rmse"].mean()),
                "sample_rmse_median": float(sub["sample_rmse"].median()),
                "sample_rmse_p90": float(sub["sample_rmse"].quantile(0.90)),
                "delta_vs_v316_mean": float(sub["delta_vs_v316"].mean()),
                "degrade_gt10_rate": float(sub["degrade_gt10"].mean()),
                "peak_ratio_median": float(sub.loc[eligible, "peak_ratio"].median()) if eligible.any() else math.nan,
                "serious_under_rate": float(sub.loc[eligible, "serious_under"].mean()) if eligible.any() else math.nan,
                "serious_over_rate": float(sub.loc[eligible, "serious_over"].mean()) if eligible.any() else math.nan,
                "peak_time_abs_error_mean": float(sub["peak_time_abs_error"].mean()),
                "noop_top1_rate": float(sub["top1_noop"].mean()) if "top1_noop" in sub.columns else math.nan,
                "noop_probability_mean": float(sub["noop_probability"].mean()) if "noop_probability" in sub.columns else math.nan,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def metric_value(summary: pd.DataFrame, method: str, split: str, group: str, col: str) -> float:
    row = summary[
        summary["method_name"].eq(method)
        & summary["split"].eq(split)
        & summary["group"].eq(group)
    ]
    if row.empty:
        return math.nan
    return float(row.iloc[0][col])


def evaluate_validation_gates(summary: pd.DataFrame, method: str, baseline_method: str = BASE_MODEL_NAME) -> Dict[str, object]:
    """按目标定义检查验证集通过门槛。"""

    val = "val"
    base_all = metric_value(summary, baseline_method, val, "全部样本", "sample_rmse_mean")
    cand_all = metric_value(summary, method, val, "全部样本", "sample_rmse_mean")
    base_normal = metric_value(summary, baseline_method, val, "普通样本", "sample_rmse_mean")
    cand_normal = metric_value(summary, method, val, "普通样本", "sample_rmse_mean")
    base_strong = metric_value(summary, baseline_method, val, "强方向盘样本", "sample_rmse_mean")
    cand_strong = metric_value(summary, method, val, "强方向盘样本", "sample_rmse_mean")
    base_h20 = metric_value(summary, baseline_method, val, "困难前20", "sample_rmse_mean")
    cand_h20 = metric_value(summary, method, val, "困难前20", "sample_rmse_mean")
    base_h10 = metric_value(summary, baseline_method, val, "困难前10", "sample_rmse_mean")
    cand_h10 = metric_value(summary, method, val, "困难前10", "sample_rmse_mean")
    base_under = metric_value(summary, baseline_method, val, "全部样本", "serious_under_rate")
    cand_under = metric_value(summary, method, val, "全部样本", "serious_under_rate")
    cand_degrade_all = metric_value(summary, method, val, "全部样本", "degrade_gt10_rate")
    cand_degrade_normal = metric_value(summary, method, val, "普通样本", "degrade_gt10_rate")

    def improvement(base: float, cand: float) -> float:
        if not np.isfinite(base) or abs(base) < 1e-12:
            return math.nan
        return (base - cand) / base

    under_pass = False
    if np.isfinite(base_under) and np.isfinite(cand_under):
        under_pass = cand_under <= base_under * 0.80 if base_under > 0 else cand_under <= 0

    checks = {
        "全部样本退化不超过0.5%": bool(np.isfinite(cand_all) and cand_all <= base_all * 1.005),
        "普通样本退化不超过1.0%": bool(np.isfinite(cand_normal) and cand_normal <= base_normal * 1.010),
        "强方向盘改善至少3.0%": bool(np.isfinite(cand_strong) and cand_strong <= base_strong * 0.970),
        "困难前20改善至少5.0%": bool(np.isfinite(cand_h20) and cand_h20 <= base_h20 * 0.950),
        "困难前10改善至少8.0%": bool(np.isfinite(cand_h10) and cand_h10 <= base_h10 * 0.920),
        "幅值严重低估比例下降至少20%": bool(under_pass),
        "全部样本大退化比例不超过15%": bool(np.isfinite(cand_degrade_all) and cand_degrade_all <= 0.150),
        "普通样本大退化比例不超过10%": bool(np.isfinite(cand_degrade_normal) and cand_degrade_normal <= 0.100),
    }
    return {
        "method_name": method,
        "passes_all_validation_gates": bool(all(checks.values())),
        **checks,
        "val_all_rmse": cand_all,
        "val_all_delta_rate": improvement(base_all, cand_all),
        "val_normal_delta_rate": improvement(base_normal, cand_normal),
        "val_strong_improvement_rate": improvement(base_strong, cand_strong),
        "val_hard20_improvement_rate": improvement(base_h20, cand_h20),
        "val_hard10_improvement_rate": improvement(base_h10, cand_h10),
        "val_under_reduction_rate": improvement(base_under, cand_under) if np.isfinite(base_under) else math.nan,
        "val_degrade_gt10_rate_all": cand_degrade_all,
        "val_degrade_gt10_rate_normal": cand_degrade_normal,
        "pass_count": int(sum(checks.values())),
    }


def choose_validation_method(gate_table: pd.DataFrame) -> pd.Series:
    """只基于验证集选择固定方案。"""

    passed = gate_table[gate_table["passes_all_validation_gates"].astype(bool)].copy()
    if not passed.empty:
        return passed.sort_values(
            ["val_hard20_improvement_rate", "val_hard10_improvement_rate", "val_all_delta_rate"],
            ascending=[False, False, False],
        ).iloc[0]
    ranked = gate_table.copy()
    ranked["fallback_score"] = (
        ranked["pass_count"].astype(float) * 10.0
        + ranked["val_hard20_improvement_rate"].fillna(-9.0)
        + ranked["val_hard10_improvement_rate"].fillna(-9.0)
        + ranked["val_strong_improvement_rate"].fillna(-9.0)
        - np.maximum(ranked["val_all_delta_rate"].fillna(9.0) * -1.0, 0.0)
    )
    return ranked.sort_values("fallback_score", ascending=False).iloc[0]


def build_candidate_usage(
    method_name: str,
    split: str,
    chosen_idx: np.ndarray,
    oracle_idx: np.ndarray,
    candidate_names: List[str],
    prob: np.ndarray,
) -> pd.DataFrame:
    """候选使用次数和最优次数。"""

    rows = []
    for idx, name in enumerate(candidate_names):
        rows.append(
            {
                "method_name": method_name,
                "split": split,
                "candidate_idx": idx,
                "candidate_name": name,
                "top1_chosen_n": int(np.sum(chosen_idx == idx)),
                "oracle_top1_n": int(np.sum(oracle_idx == idx)),
                "prob_mean": float(np.mean(prob[:, idx])),
            }
        )
    return pd.DataFrame(rows)


def plot_validation_bars(summary: pd.DataFrame, selected_method: str) -> Path:
    """绘制验证集关键分组误差。"""

    rows = summary[
        summary["split"].eq("val")
        & summary["method_name"].isin([BASE_MODEL_NAME, selected_method, "候选最优上限"])
        & summary["group"].isin(["全部样本", "普通样本", "强方向盘样本", "困难前20", "困难前10"])
    ].copy()
    order = ["全部样本", "普通样本", "强方向盘样本", "困难前20", "困难前10"]
    rows["group"] = pd.Categorical(rows["group"], categories=order, ordered=True)
    pivot = rows.pivot_table(index="group", columns="method_name", values="sample_rmse_mean", aggfunc="first")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("平均误差")
    ax.set_title("第317版验证集分组误差")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / "v317_validation_group_rmse.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_candidate_usage(usage: pd.DataFrame, selected_method: str) -> Path:
    """绘制候选选择次数。"""

    rows = usage[(usage["method_name"].eq(selected_method)) & (usage["split"].eq("val"))].copy()
    rows = rows.sort_values("candidate_idx")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(rows["candidate_idx"].astype(str), rows["top1_chosen_n"], label="门控选择次数", alpha=0.8)
    ax.plot(rows["candidate_idx"].astype(str), rows["oracle_top1_n"], color="#DC2626", marker="o", label="训练标签最优次数")
    ax.set_xlabel("候选编号")
    ax.set_ylabel("验证集次数")
    ax.set_title("第317版验证集候选使用诊断")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURES / "v317_validation_candidate_usage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_zip_package() -> Tuple[Path, bool]:
    """打包第317版产物并自检。"""

    zip_path = OUT / "v317_two_stage_candidate_gate_20260704.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in [TABLES, FIGURES, REPORTS, LOGS]:
            for file in folder.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(OUT))
    with zipfile.ZipFile(zip_path, "r") as zf:
        ok = zf.testzip() is None
    return zip_path, bool(ok)


def write_report(
    guardrail: Dict[str, object],
    gate_table: pd.DataFrame,
    group_summary: pd.DataFrame,
    selected_method: str,
    selected_pass: bool,
    test_reported: bool,
) -> Path:
    """写第317版中文报告。"""

    val_rows = group_summary[
        group_summary["split"].eq("val")
        & group_summary["method_name"].isin([BASE_MODEL_NAME, selected_method, "候选最优上限"])
        & group_summary["group"].isin(["全部样本", "普通样本", "强方向盘样本", "困难前20", "困难前10"])
    ][["method_name", "split", "group", "n", "sample_rmse_mean", "delta_vs_v316_mean", "degrade_gt10_rate", "serious_under_rate", "noop_top1_rate"]]

    gate_cols = [
        "method_name",
        "passes_all_validation_gates",
        "pass_count",
        "val_all_delta_rate",
        "val_strong_improvement_rate",
        "val_hard20_improvement_rate",
        "val_hard10_improvement_rate",
        "val_under_reduction_rate",
        "val_degrade_gt10_rate_all",
        "val_degrade_gt10_rate_normal",
    ]
    gate_view = gate_table[gate_cols].copy()

    lines = [
        "# 第317版二阶段候选门控校正实验",
        "",
        "## 结论",
        "",
        f"- 固定方案：`{selected_method}`。",
        f"- 验证门槛是否全部通过：`{selected_pass}`。",
        f"- 是否报告测试集：`{test_reported}`。",
        "- 本轮只使用第315版保留清单；84个隔离事件不参与训练、验证选模或测试主统计。",
        "- 门控输入只包含锚点前车辆信号、第316版预测摘要和可部署的粗场景标签。",
        "- 候选标签和残差原型只用训练集真实曲线构造；测试集不参与方案选择。",
        "",
        "## 验证集关键结果",
        "",
        markdown_table(val_rows),
        "",
        "## 验证门槛检查",
        "",
        markdown_table(gate_view.sort_values(["passes_all_validation_gates", "pass_count"], ascending=[False, False])),
        "",
        "## 固定方案解释",
        "",
    ]
    if selected_pass:
        lines += [
            "- 固定方案通过全部验证门槛，因此脚本按目标允许生成测试集结果。",
            "- 测试集结果只作为最终报告，不参与任何参数选择。",
        ]
    else:
        lines += [
            "- 固定方案未通过全部验证门槛，因此第317版不报告测试集结果。",
            "- 下一步按失败分流：先检查候选最优上限；若候选最优明显改善但门控未改善，则优化门控特征和候选选择损失；若候选最优也不改善，则扩充候选库。",
        ]
    lines += [
        "",
        "## 守卫信息",
        "",
        f"- 训练事件数：`{guardrail['train_event_n']}`",
        f"- 验证事件数：`{guardrail['val_event_n']}`",
        f"- 测试事件数：`{guardrail['test_event_n']}`",
        f"- 隔离事件数：`{guardrail['isolated_event_n']}`",
        f"- 候选数量：`{guardrail['candidate_n']}`",
        f"- 测试集参与选模：`{guardrail['candidate_selection_uses_test']}`",
        f"- 使用锚点后真实曲线作为输入：`{guardrail['uses_future_truth_as_input']}`",
        f"- 压缩包自检：`{guardrail.get('zip_testzip')}`",
    ]
    path = REPORTS / "v317_two_stage_candidate_gate_cn.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    started = time.time()
    np.random.seed(SEED)
    clean_out_dir()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    meta, arrays = load_delay0_dataset()
    grid = arrays["grid"]
    y_true = arrays["y_true"]
    y316 = arrays["y316"]
    y307 = arrays["y307"]
    y300 = arrays["y300"]

    write_csv(
        pd.DataFrame(
            [
                {"input_name": "v315_keep_manifest", "path": str(V315_KEEP), "sha256": file_sha256(V315_KEEP)},
                {"input_name": "v315_isolate_manifest", "path": str(V315_ISOLATE), "sha256": file_sha256(V315_ISOLATE)},
                {"input_name": "v316_predictions", "path": str(V316_PRED), "sha256": file_sha256(V316_PRED)},
                {"input_name": "v316_per_sample", "path": str(V316_PER_SAMPLE), "sha256": file_sha256(V316_PER_SAMPLE)},
                {"input_name": "v316_guardrail", "path": str(V316_GUARDRAIL), "sha256": file_sha256(V316_GUARDRAIL) if V316_GUARDRAIL.exists() else ""},
            ]
        ),
        LOGS / "input_hashes.csv",
    )

    features = build_feature_matrix(meta, y316, grid)
    write_csv(pd.concat([meta[["event_uid", "split"]].reset_index(drop=True), features], axis=1), TABLES / "v317_gate_input_features.csv")

    split = meta["split"].astype(str).to_numpy()
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    train_meta = meta.loc[train_mask].reset_index(drop=True)
    prototypes = build_residual_prototypes(y_true[train_mask], y316[train_mask], grid, train_meta)
    write_csv(pd.DataFrame(prototypes, columns=[f"t_{x:.1f}" for x in grid]), TABLES / "v317_train_residual_prototypes.csv")

    candidates_all = build_candidates(y316, grid, prototypes)
    candidate_info = pd.DataFrame({"candidate_idx": np.arange(len(candidates_all.names)), "candidate_name": candidates_all.names})
    write_csv(candidate_info, TABLES / "v317_candidate_library.csv")

    errors_all = candidate_errors(candidates_all.curves, y_true)
    oracle_idx_all = np.nanargmin(errors_all, axis=1).astype(int)
    oracle_pred_all = candidates_all.curves[np.arange(len(meta)), oracle_idx_all, :]

    train_errors = errors_all[train_mask]
    target_train = make_oracle_target(train_errors, y_true[train_mask], grid)
    true_train_amp = np.array([signed_peak(c, grid)[1] for c in y_true[train_mask]], dtype=float)
    sample_weight = 1.0 + 0.35 * (true_train_amp >= 0.80).astype(float)
    sample_weight += 0.35 * train_meta["within_bad_top20_by_v249"].astype(bool).to_numpy().astype(float)

    x_train = features.loc[train_mask].reset_index(drop=True)
    x_val = align_feature_columns(x_train, features.loc[val_mask].reset_index(drop=True))
    x_test = align_feature_columns(x_train, features.loc[test_mask].reset_index(drop=True))

    models = train_gate_models(x_train, target_train, sample_weight)

    val_candidates = candidates_all.curves[val_mask]
    val_meta = meta.loc[val_mask].reset_index(drop=True)
    val_true = y_true[val_mask]
    val_base_rmse = curve_rmse(y316[val_mask], val_true)
    val_oracle_idx = oracle_idx_all[val_mask]

    all_per_sample: List[pd.DataFrame] = []
    all_usage: List[pd.DataFrame] = []

    all_per_sample.append(build_per_sample_metrics(y316[val_mask], val_true, val_meta, grid, BASE_MODEL_NAME, val_base_rmse))
    all_per_sample.append(build_per_sample_metrics(y307[val_mask], val_true, val_meta, grid, OLD_V307_NAME, val_base_rmse))
    all_per_sample.append(build_per_sample_metrics(y300[val_mask], val_true, val_meta, grid, V300_NAME, val_base_rmse))
    all_per_sample.append(
        build_per_sample_metrics(
            oracle_pred_all[val_mask],
            val_true,
            val_meta,
            grid,
            "候选最优上限",
            val_base_rmse,
            chosen_idx=val_oracle_idx,
            candidate_names=candidates_all.names,
        )
    )

    stored_val_outputs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, str, str]] = {}
    for model_name, model in models.items():
        prob = predict_candidate_prob(model, x_val, len(candidates_all.names))
        chosen = np.argmax(prob, axis=1).astype(int)
        pred_weighted = weighted_prediction(val_candidates, prob)
        pred_top1 = top1_prediction(val_candidates, chosen)
        for mode_name, pred in [("候选加权", pred_weighted), ("候选单选", pred_top1)]:
            method_name = f"{model_name}-{mode_name}"
            all_per_sample.append(
                build_per_sample_metrics(
                    pred,
                    val_true,
                    val_meta,
                    grid,
                    method_name,
                    val_base_rmse,
                    candidate_prob=prob,
                    chosen_idx=chosen,
                    candidate_names=candidates_all.names,
                )
            )
            stored_val_outputs[method_name] = (pred, prob, chosen, model_name, mode_name)
            all_usage.append(build_candidate_usage(method_name, "val", chosen, val_oracle_idx, candidates_all.names, prob))

    val_per_sample = pd.concat(all_per_sample, ignore_index=True)
    group_summary = summarize_groups(val_per_sample)

    gate_rows = []
    for method in sorted(stored_val_outputs.keys()):
        gate_rows.append(evaluate_validation_gates(group_summary, method))
    gate_table = pd.DataFrame(gate_rows)
    selected_row = choose_validation_method(gate_table)
    selected_method = str(selected_row["method_name"])
    selected_pass = bool(selected_row["passes_all_validation_gates"])

    usage = pd.concat(all_usage, ignore_index=True) if all_usage else pd.DataFrame()
    write_csv(val_per_sample, TABLES / "v317_validation_per_sample_metrics.csv")
    write_csv(group_summary, TABLES / "v317_validation_group_summary.csv")
    write_csv(gate_table, TABLES / "v317_validation_gate_check.csv")
    write_csv(usage, TABLES / "v317_validation_candidate_usage.csv")

    test_reported = False
    test_per_sample = pd.DataFrame()
    test_group_summary = pd.DataFrame()
    if selected_pass:
        selected_model_name, selected_mode = stored_val_outputs[selected_method][3], stored_val_outputs[selected_method][4]
        model = models[selected_model_name]
        prob_test = predict_candidate_prob(model, x_test, len(candidates_all.names))
        chosen_test = np.argmax(prob_test, axis=1).astype(int)
        test_candidates = candidates_all.curves[test_mask]
        if selected_mode == "候选加权":
            pred_test = weighted_prediction(test_candidates, prob_test)
        else:
            pred_test = top1_prediction(test_candidates, chosen_test)
        test_meta = meta.loc[test_mask].reset_index(drop=True)
        test_true = y_true[test_mask]
        test_base_rmse = curve_rmse(y316[test_mask], test_true)
        test_oracle_idx = oracle_idx_all[test_mask]
        test_frames = [
            build_per_sample_metrics(y316[test_mask], test_true, test_meta, grid, BASE_MODEL_NAME, test_base_rmse),
            build_per_sample_metrics(y307[test_mask], test_true, test_meta, grid, OLD_V307_NAME, test_base_rmse),
            build_per_sample_metrics(y300[test_mask], test_true, test_meta, grid, V300_NAME, test_base_rmse),
            build_per_sample_metrics(
                oracle_pred_all[test_mask],
                test_true,
                test_meta,
                grid,
                "候选最优上限",
                test_base_rmse,
                chosen_idx=test_oracle_idx,
                candidate_names=candidates_all.names,
            ),
            build_per_sample_metrics(
                pred_test,
                test_true,
                test_meta,
                grid,
                selected_method,
                test_base_rmse,
                candidate_prob=prob_test,
                chosen_idx=chosen_test,
                candidate_names=candidates_all.names,
            ),
        ]
        test_per_sample = pd.concat(test_frames, ignore_index=True)
        test_group_summary = summarize_groups(test_per_sample)
        write_csv(test_per_sample, TABLES / "v317_test_per_sample_metrics.csv")
        write_csv(test_group_summary, TABLES / "v317_test_group_summary.csv")
        write_csv(
            build_candidate_usage(selected_method, "test", chosen_test, test_oracle_idx, candidates_all.names, prob_test),
            TABLES / "v317_test_candidate_usage.csv",
        )
        test_reported = True

    figure_paths = [
        plot_validation_bars(group_summary, selected_method),
        plot_candidate_usage(usage, selected_method) if not usage.empty else None,
    ]
    figure_paths = [p for p in figure_paths if p is not None]

    guardrail = {
        "pass": bool(True),
        "goal_validation_passed": selected_pass,
        "test_reported": test_reported,
        "selected_method": selected_method,
        "train_event_n": int(train_mask.sum()),
        "val_event_n": int(val_mask.sum()),
        "test_event_n": int(test_mask.sum()),
        "kept_event_n": int(len(meta)),
        "isolated_event_n": 84,
        "candidate_n": int(len(candidates_all.names)),
        "candidate_selection_uses_test": False,
        "uses_test_error_as_features": False,
        "uses_future_truth_as_input": False,
        "uses_v315_keep_manifest": True,
        "uses_v316_base_prediction": True,
        "uses_future_truth_for_training_targets_only": True,
        "validation_gate_table": str(TABLES / "v317_validation_gate_check.csv"),
        "test_result_suppressed_when_validation_fails": True,
        "runtime_seconds": float(time.time() - started),
        "figure_paths": [str(p) for p in figure_paths],
        "zip_path": str(OUT / "v317_two_stage_candidate_gate_20260704.zip"),
        "zip_testzip": None,
    }
    report_path = write_report(guardrail, gate_table, group_summary, selected_method, selected_pass, test_reported)
    guardrail["report_path"] = str(report_path)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_csv(
        pd.DataFrame(
            [
                {"artifact_type": "table", "path": str(p), "size": p.stat().st_size}
                for p in sorted(TABLES.glob("*.csv"))
            ]
            + [
                {"artifact_type": "figure", "path": str(p), "size": p.stat().st_size}
                for p in sorted(FIGURES.glob("*.png"))
            ]
            + [
                {"artifact_type": "report", "path": str(p), "size": p.stat().st_size}
                for p in sorted(REPORTS.glob("*.md"))
            ]
        ),
        LOGS / "file_inventory.csv",
    )
    # 报告、守卫日志和文件清单全部落盘后再打包，保证压缩包可单独复查。
    zip_path, zip_ok = make_zip_package()
    guardrail["zip_path"] = str(zip_path)
    guardrail["zip_testzip"] = bool(zip_ok)
    report_path = write_report(guardrail, gate_table, group_summary, selected_method, selected_pass, test_reported)
    guardrail["report_path"] = str(report_path)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_csv(
        pd.DataFrame(
            [
                {"artifact_type": "table", "path": str(p), "size": p.stat().st_size}
                for p in sorted(TABLES.glob("*.csv"))
            ]
            + [
                {"artifact_type": "figure", "path": str(p), "size": p.stat().st_size}
                for p in sorted(FIGURES.glob("*.png"))
            ]
            + [
                {"artifact_type": "report", "path": str(p), "size": p.stat().st_size}
                for p in sorted(REPORTS.glob("*.md"))
            ]
        ),
        LOGS / "file_inventory.csv",
    )
    zip_path, zip_ok = make_zip_package()
    guardrail["zip_testzip"] = bool(zip_ok)
    write_json(guardrail, LOGS / "guardrail_check.json")
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
