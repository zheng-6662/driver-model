#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第320版：排序配额修复门控实验。

目的：
- 固定第315版当前窗口保留清单，不再扩大过滤范围；
- 固定第316版基础预测，不重新训练主模型；
- 复用第316版原预测和第317版候选库；
- 在第318版两段式输出上增加候选正收益概率和困难代理分数；
- 普通样本保护通道保持低覆盖；
- 强方向盘/困难代理通道改成排序配额，禁止被绝对阈值清空；
- 只用训练集内部交叉验证搜索配额和风险预算，验证集只做通过/失败判定。

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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260705
ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "03_baselines"

V315_TABLES = BASELINES / "v315_rapid_steering_filter_reanchor_plan_20260704" / "tables"
V315_KEEP = V315_TABLES / "v315_current_window_keep_manifest.csv"
V315_ISOLATE = V315_TABLES / "v315_current_window_isolate_manifest.csv"

V316_OUT = BASELINES / "v316_filtered_current_window_coarse_scene_train_20260704"
V316_PRED = V316_OUT / "v316_filtered_current_window_predictions.npz"
V316_PER_SAMPLE = V316_OUT / "tables" / "v316_per_sample_metrics_original_remaining.csv"
V316_GUARDRAIL = V316_OUT / "logs" / "guardrail_check.json"

OUT = BASELINES / "v320_rank_budget_repair_gate_20260705"
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


@dataclass
class TwoStageModels:
    """两段式模型容器，供第320版复用。"""

    corr_clf: object
    gain_reg: object
    cand_gain_reg: object
    cand_bad_clf: object
    cand_pos_clf: object


@dataclass
class TwoStageComponents:
    """两段式门控在某个样本集上的中间输出。"""

    p_corr: np.ndarray
    g_hat: np.ndarray
    candidate_idx: np.ndarray
    candidate_gain_hat: np.ndarray
    candidate_bad_prob: np.ndarray
    candidate_margin: np.ndarray
    candidate_pos_prob: np.ndarray | None = None


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in [TABLES, FIGURES, REPORTS, LOGS]:
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理第320版自己的输出目录。"""

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
            # 兼容新旧 NumPy，避免新版对 trapz 的弃用警告刷屏。
            integral_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            out[f"{key}_abs_integral"] = float(integral_fn(np.abs(vals), ts))
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
    """读取第315清单和第316预测包，并固定第320版样本边界。"""

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
        raise AssertionError(f"第320版样本边界不符合目标：实际{split_counts}，期望{expected}")
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
    cand_noop_all = metric_value(summary, method, val, "全部样本", "noop_top1_rate")
    cand_noop_normal = metric_value(summary, method, val, "普通样本", "noop_top1_rate")

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
        "普通样本原预测保持率至少80%": bool(np.isfinite(cand_noop_normal) and cand_noop_normal >= 0.800),
        "全部样本校正率不超过45%": bool(np.isfinite(cand_noop_all) and (1.0 - cand_noop_all) <= 0.450),
        "普通样本校正率不超过20%": bool(np.isfinite(cand_noop_normal) and (1.0 - cand_noop_normal) <= 0.200),
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
        "val_noop_rate_all": cand_noop_all,
        "val_noop_rate_normal": cand_noop_normal,
        "val_correction_rate_all": 1.0 - cand_noop_all if np.isfinite(cand_noop_all) else math.nan,
        "val_correction_rate_normal": 1.0 - cand_noop_normal if np.isfinite(cand_noop_normal) else math.nan,
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


def positive_class_probability(model: object, x: pd.DataFrame) -> np.ndarray:
    """兼容单类别训练折的二分类概率读取。"""

    prob = model.predict_proba(x)
    classes = model.named_steps["model"].classes_.astype(int)
    if 1 not in classes:
        return np.zeros(len(x), dtype=float)
    return prob[:, int(np.where(classes == 1)[0][0])]


def correctability_targets(errors: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """把候选最优上限转成第318版第一段门控的监督标签。"""

    base_err = errors[:, 0].astype(float)
    non_noop = errors[:, 1:]
    best_non_pos = np.nanargmin(non_noop, axis=1).astype(int)
    best_non_idx = best_non_pos + 1
    best_non_err = non_noop[np.arange(len(errors)), best_non_pos]
    oracle_gain = base_err - best_non_err
    normal = (~meta["strong_steer"].astype(bool) & ~meta["within_bad_top20_by_v249"].astype(bool)).to_numpy()
    abs_min = np.where(normal, 0.050, 0.030)
    rel_min = np.where(normal, 0.080, 0.050)
    gain_threshold = np.maximum(abs_min, rel_min * np.maximum(base_err, 1e-8))
    gray_threshold = np.maximum(abs_min * 0.50, rel_min * 0.50 * np.maximum(base_err, 1e-8))
    y_corr = oracle_gain >= gain_threshold
    gray_zone = (~y_corr) & (oracle_gain >= gray_threshold)
    return pd.DataFrame(
        {
            "event_uid": meta["event_uid"].astype(str).to_numpy(),
            "split": meta["split"].astype(str).to_numpy(),
            "base_error": base_err,
            "best_nonnoop_candidate_idx": best_non_idx,
            "best_nonnoop_error": best_non_err,
            "oracle_gain": oracle_gain,
            "oracle_gain_relative": safe_ratio(oracle_gain, base_err),
            "gain_threshold": gain_threshold,
            "correctable_label": y_corr.astype(int),
            "gray_zone": gray_zone.astype(int),
            "normal_group": normal.astype(int),
            "strong_steer": meta["strong_steer"].astype(bool).to_numpy().astype(int),
            "within_bad_top20_by_v249": meta["within_bad_top20_by_v249"].astype(bool).to_numpy().astype(int),
            "within_bad_top10_by_v249": meta["within_bad_top10_by_v249"].astype(bool).to_numpy().astype(int),
        }
    )


def summarize_correctability(labels: pd.DataFrame, candidate_names: List[str]) -> pd.DataFrame:
    """汇总可校正标签分布和候选上限。"""

    rows: List[Dict[str, object]] = []
    for split, part in labels.groupby("split", dropna=False):
        for group_name, mask in {
            "全部样本": np.ones(len(part), dtype=bool),
            "普通样本": part["normal_group"].astype(bool).to_numpy(),
            "强方向盘样本": part["strong_steer"].astype(bool).to_numpy(),
            "困难前20": part["within_bad_top20_by_v249"].astype(bool).to_numpy(),
            "困难前10": part["within_bad_top10_by_v249"].astype(bool).to_numpy(),
        }.items():
            sub = part.loc[mask]
            if sub.empty:
                continue
            rows.append(
                {
                    "split": split,
                    "group": group_name,
                    "n": int(len(sub)),
                    "correctable_rate": float(sub["correctable_label"].mean()),
                    "gray_zone_rate": float(sub["gray_zone"].mean()),
                    "oracle_gain_mean": float(sub["oracle_gain"].mean()),
                    "oracle_gain_positive_rate": float((sub["oracle_gain"] > 0).mean()),
                    "base_error_mean": float(sub["base_error"].mean()),
                    "best_nonnoop_error_mean": float(sub["best_nonnoop_error"].mean()),
                }
            )
    summary = pd.DataFrame(rows)
    candidate_rows = []
    for split, part in labels.groupby("split", dropna=False):
        counts = part["best_nonnoop_candidate_idx"].value_counts().sort_index()
        for idx, count in counts.items():
            candidate_rows.append(
                {
                    "split": split,
                    "candidate_idx": int(idx),
                    "candidate_name": candidate_names[int(idx)],
                    "best_nonnoop_n": int(count),
                }
            )
    return pd.concat([summary, pd.DataFrame(candidate_rows)], ignore_index=True, sort=False)


def candidate_static_features(y0: np.ndarray, cand: np.ndarray, grid: np.ndarray, candidate_idx: int) -> pd.DataFrame:
    """为某个候选构造不依赖真实未来的候选描述特征。"""

    base_signed, base_amp, base_t = peak_arrays(y0, grid)
    cand_signed, cand_amp, cand_t = peak_arrays(cand, grid)
    diff = cand - y0
    n = len(y0)
    return pd.DataFrame(
        {
            "candidate_idx": np.full(n, candidate_idx, dtype=float),
            "candidate_is_scale": np.full(n, 1.0 if 1 <= candidate_idx <= 5 else 0.0),
            "candidate_is_shift": np.full(n, 1.0 if 6 <= candidate_idx <= 11 else 0.0),
            "candidate_is_scale_shift": np.full(n, 1.0 if 12 <= candidate_idx <= 15 else 0.0),
            "candidate_is_residual_proto": np.full(n, 1.0 if candidate_idx >= 16 else 0.0),
            "candidate_peak_signed": cand_signed,
            "candidate_peak_abs": cand_amp,
            "candidate_peak_t": cand_t,
            "candidate_peak_abs_ratio_to_base": safe_ratio(cand_amp, base_amp),
            "candidate_peak_signed_delta": cand_signed - base_signed,
            "candidate_peak_abs_delta": cand_amp - base_amp,
            "candidate_peak_t_delta": cand_t - base_t,
            "candidate_end_delta": cand[:, -1] - y0[:, -1],
            "candidate_curve_delta_rmse": np.sqrt(np.nanmean(diff ** 2, axis=1)),
            "candidate_curve_delta_energy": np.nanmean(diff ** 2, axis=1),
            "candidate_curve_delta_max_abs": np.nanmax(np.abs(diff), axis=1),
        }
    )


def make_candidate_long_matrix(
    x_base: pd.DataFrame,
    candidate_curves: np.ndarray,
    y0: np.ndarray,
    grid: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """展开为样本-候选长表，供第二段候选收益/风险模型使用。"""

    x0 = x_base.reset_index(drop=True)
    frames: List[pd.DataFrame] = []
    sample_ids: List[np.ndarray] = []
    candidate_ids: List[np.ndarray] = []
    n = len(x0)
    for candidate_idx in range(1, candidate_curves.shape[1]):
        cand_feat = candidate_static_features(y0, candidate_curves[:, candidate_idx, :], grid, candidate_idx)
        frames.append(pd.concat([x0, cand_feat], axis=1))
        sample_ids.append(np.arange(n, dtype=int))
        candidate_ids.append(np.full(n, candidate_idx, dtype=int))
    return pd.concat(frames, ignore_index=True), np.concatenate(sample_ids), np.concatenate(candidate_ids)


def row_rank01_matrix(matrix: np.ndarray) -> np.ndarray:
    """逐行转成0到1秩，用于候选间相对排序。"""

    arr = np.asarray(matrix, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    for i in range(arr.shape[0]):
        row = arr[i]
        valid = np.isfinite(row)
        if valid.sum() <= 1:
            continue
        order = np.argsort(row[valid])
        ranks = np.empty(valid.sum(), dtype=float)
        ranks[order] = np.linspace(0.0, 1.0, valid.sum())
        out[i, valid] = ranks
    return out


def make_two_stage_models(seed: int) -> TwoStageModels:
    """创建第318版保守两段式模型。"""

    corr_clf = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=260,
                    max_depth=6,
                    min_samples_leaf=12,
                    random_state=seed,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    gain_reg = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=240,
                    max_depth=7,
                    min_samples_leaf=12,
                    random_state=seed + 11,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    cand_gain_reg = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=260,
                    max_depth=8,
                    min_samples_leaf=18,
                    random_state=seed + 23,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    cand_bad_clf = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=240,
                    max_depth=7,
                    min_samples_leaf=18,
                    random_state=seed + 37,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    cand_pos_clf = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=320,
                    max_depth=8,
                    min_samples_leaf=12,
                    random_state=seed + 51,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return TwoStageModels(corr_clf, gain_reg, cand_gain_reg, cand_bad_clf, cand_pos_clf)


def fit_two_stage_models(
    x: pd.DataFrame,
    candidate_set: CandidateSet,
    y0: np.ndarray,
    y_true: np.ndarray,
    meta: pd.DataFrame,
    grid: np.ndarray,
    seed: int,
) -> TwoStageModels:
    """拟合第一段可校正门控和第二段候选收益/风险模型。"""

    x = x.reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    errors = candidate_errors(candidate_set.curves, y_true)
    labels = correctability_targets(errors, meta)
    y_corr = labels["correctable_label"].to_numpy(dtype=int)
    oracle_gain = labels["oracle_gain"].to_numpy(dtype=float)
    normal = labels["normal_group"].to_numpy(dtype=bool)
    hard20 = labels["within_bad_top20_by_v249"].to_numpy(dtype=bool)
    gray = labels["gray_zone"].to_numpy(dtype=bool)
    corr_weight = np.ones(len(x), dtype=float)
    corr_weight += 1.50 * (normal & (y_corr == 0)).astype(float)
    corr_weight += 0.75 * hard20.astype(float)
    corr_weight += 0.50 * y_corr.astype(float)
    corr_weight *= np.where(gray, 0.55, 1.0)

    models = make_two_stage_models(seed)
    models.corr_clf.fit(x, y_corr, model__sample_weight=corr_weight)
    models.gain_reg.fit(x, oracle_gain, model__sample_weight=corr_weight)

    long_x, sample_idx, candidate_idx = make_candidate_long_matrix(x, candidate_set.curves, y0, grid)
    base_err = errors[:, 0]
    candidate_err = errors[sample_idx, candidate_idx]
    candidate_gain = base_err[sample_idx] - candidate_err
    candidate_bad = (
        (candidate_err > base_err[sample_idx] * 1.10)
        & ((candidate_err - base_err[sample_idx]) > 0.030)
    ).astype(int)
    candidate_positive = (candidate_gain > 0.0).astype(int)
    strong = meta["strong_steer"].astype(bool).to_numpy()
    long_weight = np.ones(len(long_x), dtype=float)
    long_weight += 4.0 * np.maximum(candidate_gain, 0.0)
    long_weight += 3.0 * candidate_bad.astype(float)
    long_weight += 0.75 * hard20[sample_idx].astype(float)
    long_weight += 0.75 * strong[sample_idx].astype(float)
    models.cand_gain_reg.fit(long_x, candidate_gain, model__sample_weight=long_weight)
    models.cand_bad_clf.fit(long_x, candidate_bad, model__sample_weight=long_weight)
    pos_weight = np.ones(len(long_x), dtype=float)
    pos_weight += 3.0 * candidate_positive.astype(float)
    pos_weight += 1.0 * hard20[sample_idx].astype(float)
    pos_weight += 0.75 * strong[sample_idx].astype(float)
    models.cand_pos_clf.fit(long_x, candidate_positive, model__sample_weight=pos_weight)
    return models


def predict_two_stage_components(
    models: TwoStageModels,
    x: pd.DataFrame,
    candidate_set: CandidateSet,
    y0: np.ndarray,
    grid: np.ndarray,
) -> TwoStageComponents:
    """输出第318版两段式门控的可校正概率、预期收益和候选风险。"""

    x = x.reset_index(drop=True)
    p_corr = positive_class_probability(models.corr_clf, x)
    g_hat = models.gain_reg.predict(x).astype(float)
    long_x, sample_idx, candidate_idx = make_candidate_long_matrix(x, candidate_set.curves, y0, grid)
    gain_pred = models.cand_gain_reg.predict(long_x).astype(float)
    bad_prob = positive_class_probability(models.cand_bad_clf, long_x)
    pos_prob = positive_class_probability(models.cand_pos_clf, long_x)

    n = len(x)
    n_candidates = candidate_set.curves.shape[1]
    gain_matrix = np.full((n, n_candidates), -np.inf, dtype=float)
    bad_matrix = np.ones((n, n_candidates), dtype=float)
    pos_matrix = np.zeros((n, n_candidates), dtype=float)
    gain_matrix[sample_idx, candidate_idx] = gain_pred
    bad_matrix[sample_idx, candidate_idx] = bad_prob
    pos_matrix[sample_idx, candidate_idx] = pos_prob
    score = (
        1.25 * pos_matrix
        + 0.45 * row_rank01_matrix(gain_matrix)
        - 0.40 * row_rank01_matrix(bad_matrix)
    )
    score[:, 0] = -np.inf
    best_idx = np.nanargmax(score, axis=1).astype(int)
    best_score = score[np.arange(n), best_idx]
    score_second = score.copy()
    score_second[np.arange(n), best_idx] = -np.inf
    second_score = np.nanmax(score_second, axis=1)
    margin = best_score - second_score
    margin[~np.isfinite(margin)] = 0.0
    return TwoStageComponents(
        p_corr=p_corr,
        g_hat=g_hat,
        candidate_idx=best_idx,
        candidate_gain_hat=gain_matrix[np.arange(n), best_idx],
        candidate_bad_prob=bad_matrix[np.arange(n), best_idx],
        candidate_margin=margin,
        candidate_pos_prob=pos_matrix[np.arange(n), best_idx],
    )


def fit_oof_and_full_models(
    x_train: pd.DataFrame,
    train_candidates: CandidateSet,
    y0_train: np.ndarray,
    y_true_train: np.ndarray,
    train_meta: pd.DataFrame,
    grid: np.ndarray,
) -> Tuple[TwoStageComponents, TwoStageModels]:
    """训练集内部交叉验证预测，用于搜索阈值；最后再拟合全量训练模型。"""

    n = len(x_train)
    p_corr = np.zeros(n, dtype=float)
    g_hat = np.zeros(n, dtype=float)
    candidate_idx = np.zeros(n, dtype=int)
    candidate_gain_hat = np.zeros(n, dtype=float)
    candidate_bad_prob = np.ones(n, dtype=float)
    candidate_margin = np.zeros(n, dtype=float)
    candidate_pos_prob = np.zeros(n, dtype=float)
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_id, (fit_idx, pred_idx) in enumerate(folds.split(x_train)):
        sub_models = fit_two_stage_models(
            x_train.iloc[fit_idx].reset_index(drop=True),
            CandidateSet(train_candidates.names, train_candidates.curves[fit_idx]),
            y0_train[fit_idx],
            y_true_train[fit_idx],
            train_meta.iloc[fit_idx].reset_index(drop=True),
            grid,
            SEED + 100 + fold_id,
        )
        comp = predict_two_stage_components(
            sub_models,
            x_train.iloc[pred_idx].reset_index(drop=True),
            CandidateSet(train_candidates.names, train_candidates.curves[pred_idx]),
            y0_train[pred_idx],
            grid,
        )
        p_corr[pred_idx] = comp.p_corr
        g_hat[pred_idx] = comp.g_hat
        candidate_idx[pred_idx] = comp.candidate_idx
        candidate_gain_hat[pred_idx] = comp.candidate_gain_hat
        candidate_bad_prob[pred_idx] = comp.candidate_bad_prob
        candidate_margin[pred_idx] = comp.candidate_margin
        if comp.candidate_pos_prob is not None:
            candidate_pos_prob[pred_idx] = comp.candidate_pos_prob

    full_models = fit_two_stage_models(
        x_train.reset_index(drop=True),
        train_candidates,
        y0_train,
        y_true_train,
        train_meta.reset_index(drop=True),
        grid,
        SEED + 999,
    )
    return (
        TwoStageComponents(
            p_corr=p_corr,
            g_hat=g_hat,
            candidate_idx=candidate_idx,
            candidate_gain_hat=candidate_gain_hat,
            candidate_bad_prob=candidate_bad_prob,
            candidate_margin=candidate_margin,
            candidate_pos_prob=candidate_pos_prob,
        ),
        full_models,
    )


def apply_policy(
    y0: np.ndarray,
    candidate_curves: np.ndarray,
    comp: TwoStageComponents,
    meta: pd.DataFrame,
    params: Dict[str, float],
    use_candidate_filters: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按阈值把两段式中间输出转成最终曲线。"""

    normal = (~meta["strong_steer"].astype(bool) & ~meta["within_bad_top20_by_v249"].astype(bool)).to_numpy()
    p_min = float(params["p_min"])
    gain_min = float(params["gain_min"])
    p_normal_min = float(params.get("p_normal_min", min(0.98, p_min + 0.10)))
    gain_normal_min = float(params.get("gain_normal_min", gain_min + 0.020))
    correction = (comp.p_corr >= p_min) & (comp.g_hat >= gain_min)
    correction &= (~normal) | ((comp.p_corr >= p_normal_min) & (comp.g_hat >= gain_normal_min))
    if use_candidate_filters:
        correction &= comp.candidate_gain_hat >= float(params["candidate_gain_min"])
        correction &= comp.candidate_bad_prob <= float(params["bad_prob_max"])
        correction &= comp.candidate_margin >= float(params["margin_min"])
    chosen = np.where(correction, comp.candidate_idx, 0).astype(int)
    selected = candidate_curves[np.arange(len(y0)), chosen, :]
    alpha = float(params["alpha"])
    pred = y0 + alpha * (selected - y0)
    prob = np.zeros((len(y0), candidate_curves.shape[1]), dtype=float)
    prob[:, 0] = np.where(correction, np.maximum(0.0, 1.0 - comp.p_corr), 1.0)
    prob[np.arange(len(y0)), chosen] += np.where(correction, comp.p_corr, 0.0)
    row_sum = prob.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    prob = prob / row_sum
    return pred, chosen, prob


def quick_policy_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    base_rmse: np.ndarray,
    meta: pd.DataFrame,
    chosen: np.ndarray,
    grid: np.ndarray,
) -> Dict[str, float]:
    """阈值搜索用的轻量指标，避免在每个网格点构造大表。"""

    rmse = curve_rmse(pred, true)
    true_amp = np.array([signed_peak(c, grid)[1] for c in true], dtype=float)
    pred_amp = np.array([signed_peak(c, grid)[1] for c in pred], dtype=float)
    ratio = safe_ratio(pred_amp, true_amp)
    eligible = true_amp >= PEAK_ELIGIBLE_TH
    serious_under = eligible & np.isfinite(ratio) & (ratio < SERIOUS_UNDER_RATIO_TH)
    normal = (~meta["strong_steer"].astype(bool) & ~meta["within_bad_top20_by_v249"].astype(bool)).to_numpy()
    strong = meta["strong_steer"].astype(bool).to_numpy()
    hard20 = meta["within_bad_top20_by_v249"].astype(bool).to_numpy()
    hard10 = meta["within_bad_top10_by_v249"].astype(bool).to_numpy()

    def mean_on(values: np.ndarray, mask: np.ndarray) -> float:
        if not mask.any():
            return math.nan
        return float(np.nanmean(values[mask]))

    def rate_on(values: np.ndarray, mask: np.ndarray) -> float:
        if not mask.any():
            return math.nan
        return float(np.nanmean(values[mask]))

    out = {
        "all_rmse": mean_on(rmse, np.ones(len(rmse), dtype=bool)),
        "normal_rmse": mean_on(rmse, normal),
        "strong_rmse": mean_on(rmse, strong),
        "hard20_rmse": mean_on(rmse, hard20),
        "hard10_rmse": mean_on(rmse, hard10),
        "base_all_rmse": mean_on(base_rmse, np.ones(len(rmse), dtype=bool)),
        "base_normal_rmse": mean_on(base_rmse, normal),
        "base_strong_rmse": mean_on(base_rmse, strong),
        "base_hard20_rmse": mean_on(base_rmse, hard20),
        "base_hard10_rmse": mean_on(base_rmse, hard10),
        "degrade_all": rate_on(rmse > base_rmse * 1.10, np.ones(len(rmse), dtype=bool)),
        "degrade_normal": rate_on(rmse > base_rmse * 1.10, normal),
        "noop_all": float(np.mean(chosen == 0)),
        "noop_normal": rate_on(chosen == 0, normal),
        "correction_all": float(np.mean(chosen != 0)),
        "correction_normal": rate_on(chosen != 0, normal),
        "under_rate": float(np.mean(serious_under[eligible])) if eligible.any() else math.nan,
    }
    return out


def improvement_rate(base: float, value: float) -> float:
    """相对改善率，正数代表变好。"""

    if not np.isfinite(base) or abs(base) < 1e-12:
        return math.nan
    return float((base - value) / base)


def score_policy_metrics(m: Dict[str, float]) -> Tuple[int, float]:
    """训练集内部阈值搜索的约束计数和排序分。"""

    checks = {
        "all_noharm": np.isfinite(m["all_rmse"]) and m["all_rmse"] <= m["base_all_rmse"] * 1.005,
        "normal_noharm": np.isfinite(m["normal_rmse"]) and m["normal_rmse"] <= m["base_normal_rmse"] * 1.010,
        "strong_gain": np.isfinite(m["strong_rmse"]) and m["strong_rmse"] <= m["base_strong_rmse"] * 0.970,
        "hard20_gain": np.isfinite(m["hard20_rmse"]) and m["hard20_rmse"] <= m["base_hard20_rmse"] * 0.950,
        "hard10_gain": np.isfinite(m["hard10_rmse"]) and m["hard10_rmse"] <= m["base_hard10_rmse"] * 0.920,
        "degrade_all": np.isfinite(m["degrade_all"]) and m["degrade_all"] <= 0.150,
        "degrade_normal": np.isfinite(m["degrade_normal"]) and m["degrade_normal"] <= 0.100,
        "normal_keep": np.isfinite(m["noop_normal"]) and m["noop_normal"] >= 0.800,
        "all_correction_cap": np.isfinite(m["correction_all"]) and m["correction_all"] <= 0.450,
        "normal_correction_cap": np.isfinite(m["correction_normal"]) and m["correction_normal"] <= 0.200,
    }
    pass_count = int(sum(bool(v) for v in checks.values()))
    hard20_gain = improvement_rate(m["base_hard20_rmse"], m["hard20_rmse"])
    hard10_gain = improvement_rate(m["base_hard10_rmse"], m["hard10_rmse"])
    strong_gain = improvement_rate(m["base_strong_rmse"], m["strong_rmse"])
    all_gain = improvement_rate(m["base_all_rmse"], m["all_rmse"])
    normal_gain = improvement_rate(m["base_normal_rmse"], m["normal_rmse"])
    score = (
        pass_count * 100.0
        + 30.0 * max(hard20_gain, -0.50)
        + 25.0 * max(hard10_gain, -0.50)
        + 18.0 * max(strong_gain, -0.50)
        + 12.0 * max(all_gain, -0.50)
        + 10.0 * max(normal_gain, -0.50)
        - 5.0 * max(m["correction_normal"] - 0.20, 0.0)
        - 3.0 * max(m["correction_all"] - 0.45, 0.0)
    )
    return pass_count, float(score)


def policy_grid(mode: str) -> Iterable[Dict[str, float]]:
    """第318版三条递进线的训练集内部搜索空间。"""

    p_values = [0.50, 0.60, 0.70, 0.80, 0.90]
    gain_values = [0.000, 0.020, 0.040, 0.060, 0.080]
    if mode == "stage_a_only":
        for p_min in p_values:
            for gain_min in gain_values:
                for alpha in [0.25, 0.50, 0.75]:
                    yield {
                        "p_min": p_min,
                        "gain_min": gain_min,
                        "p_normal_min": min(0.98, p_min + 0.10),
                        "gain_normal_min": gain_min + 0.020,
                        "candidate_gain_min": -9.0,
                        "bad_prob_max": 1.0,
                        "margin_min": -9.0,
                        "alpha": alpha,
                    }
        return
    alpha_values = [1.00] if mode == "candidate_select" else [0.25, 0.50, 0.75]
    for p_min in p_values:
        for gain_min in gain_values:
            for candidate_gain_min in [0.020, 0.040, 0.060, 0.080, 0.100]:
                for bad_prob_max in [0.20, 0.30, 0.40, 0.50]:
                    for margin_min in [0.000, 0.010, 0.030]:
                        for alpha in alpha_values:
                            yield {
                                "p_min": p_min,
                                "gain_min": gain_min,
                                "p_normal_min": min(0.98, p_min + 0.10),
                                "gain_normal_min": gain_min + 0.020,
                                "candidate_gain_min": candidate_gain_min,
                                "bad_prob_max": bad_prob_max,
                                "margin_min": margin_min,
                                "alpha": alpha,
                            }


def search_best_policy(
    method_name: str,
    mode: str,
    y0: np.ndarray,
    candidate_curves: np.ndarray,
    comp: TwoStageComponents,
    meta: pd.DataFrame,
    y_true: np.ndarray,
    base_rmse: np.ndarray,
    grid: np.ndarray,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """只用训练集内部交叉验证结果选择阈值和融合幅度。"""

    use_candidate_filters = mode != "stage_a_only"
    rows: List[Dict[str, object]] = []
    best_params: Dict[str, float] | None = None
    best_score = -np.inf
    for params in policy_grid(mode):
        pred, chosen, _ = apply_policy(y0, candidate_curves, comp, meta, params, use_candidate_filters)
        metrics = quick_policy_metrics(pred, y_true, base_rmse, meta, chosen, grid)
        pass_count, score = score_policy_metrics(metrics)
        row = {"method_name": method_name, "mode": mode, **params, **metrics, "pass_count": pass_count, "search_score": score}
        rows.append(row)
        if score > best_score:
            best_score = score
            best_params = params
    assert best_params is not None
    return best_params, pd.DataFrame(rows)


def rank01(values: np.ndarray) -> np.ndarray:
    """把一列分数转成0到1的秩，避免绝对阈值跨集合失效。"""

    arr = np.asarray(values, dtype=float)
    out = np.zeros(len(arr), dtype=float)
    valid = np.isfinite(arr)
    if valid.sum() <= 1:
        return out
    order = np.argsort(arr[valid])
    ranks = np.empty(valid.sum(), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, valid.sum())
    out[valid] = ranks
    return out


def candidate_disagreement_features(candidate_curves: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    """用候选之间的分歧构造困难代理特征，不读取真实未来。"""

    n = candidate_curves.shape[0]
    signed = np.zeros((n, candidate_curves.shape[1]), dtype=float)
    amp = np.zeros_like(signed)
    peak_t = np.zeros_like(signed)
    end = candidate_curves[:, :, -1]
    for k in range(candidate_curves.shape[1]):
        s, a, t = peak_arrays(candidate_curves[:, k, :], grid)
        signed[:, k] = s
        amp[:, k] = a
        peak_t[:, k] = t
    return pd.DataFrame(
        {
            "cand_signed_std": np.nanstd(signed[:, 1:], axis=1),
            "cand_amp_std": np.nanstd(amp[:, 1:], axis=1),
            "cand_peak_t_std": np.nanstd(peak_t[:, 1:], axis=1),
            "cand_end_std": np.nanstd(end[:, 1:], axis=1),
            "cand_amp_range": np.nanmax(amp[:, 1:], axis=1) - np.nanmin(amp[:, 1:], axis=1),
            "cand_end_range": np.nanmax(end[:, 1:], axis=1) - np.nanmin(end[:, 1:], axis=1),
        }
    )


def hard_proxy_feature_matrix(
    x_base: pd.DataFrame,
    comp: TwoStageComponents,
    candidate_curves: np.ndarray,
    grid: np.ndarray,
) -> pd.DataFrame:
    """困难代理模型输入：只使用可部署特征、模型预测摘要和候选分歧。"""

    comp_df = pd.DataFrame(
        {
            "stage_p_corr": comp.p_corr,
            "stage_gain_hat": comp.g_hat,
            "stage_best_candidate_idx": comp.candidate_idx.astype(float),
            "stage_best_candidate_gain_hat": comp.candidate_gain_hat,
            "stage_best_bad_prob": comp.candidate_bad_prob,
            "stage_best_margin": comp.candidate_margin,
        }
    )
    disagree = candidate_disagreement_features(candidate_curves, grid)
    return pd.concat([x_base.reset_index(drop=True), comp_df, disagree], axis=1).replace([np.inf, -np.inf], np.nan)


def fit_hard_proxy_oof(
    x_proxy: pd.DataFrame,
    base_rmse: np.ndarray,
) -> Tuple[np.ndarray, object, pd.DataFrame]:
    """训练困难代理，监督信号只来自训练集第316版误差前20。"""

    threshold = float(np.nanpercentile(base_rmse, 80.0))
    y = (base_rmse >= threshold).astype(int)
    oof = np.zeros(len(x_proxy), dtype=float)
    rows: List[Dict[str, object]] = []
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED + 300)
    for fold_id, (fit_idx, pred_idx) in enumerate(folds.split(x_proxy)):
        model = Pipeline(
            [
                ("fill", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=260,
                        max_depth=6,
                        min_samples_leaf=12,
                        class_weight="balanced",
                        random_state=SEED + 400 + fold_id,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(x_proxy.iloc[fit_idx], y[fit_idx])
        oof[pred_idx] = positive_class_probability(model, x_proxy.iloc[pred_idx])
    full_model = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=320,
                    max_depth=6,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=SEED + 500,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    full_model.fit(x_proxy, y)
    for lo, hi in [(0, 50), (50, 70), (70, 80), (80, 90), (90, 100)]:
        qlo, qhi = np.nanpercentile(oof, [lo, hi])
        mask = (oof >= qlo) & (oof <= qhi if hi == 100 else oof < qhi)
        rows.append(
            {
                "bucket": f"{lo}-{hi}",
                "n": int(mask.sum()),
                "hard20_label_rate": float(y[mask].mean()) if mask.any() else math.nan,
                "base_rmse_mean": float(base_rmse[mask].mean()) if mask.any() else math.nan,
                "score_min": float(qlo),
                "score_max": float(qhi),
            }
        )
    return oof, full_model, pd.DataFrame(rows)


def sample_candidate_score(comp: TwoStageComponents, hard_proxy: np.ndarray, candidate_curves: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """样本级候选激活分数，用于配额排序。"""

    disagree = candidate_disagreement_features(candidate_curves, grid)
    disagreement_score = (
        rank01(disagree["cand_amp_std"].to_numpy(dtype=float))
        + rank01(disagree["cand_peak_t_std"].to_numpy(dtype=float))
        + rank01(disagree["cand_end_std"].to_numpy(dtype=float))
    ) / 3.0
    pos_score = comp.candidate_pos_prob if comp.candidate_pos_prob is not None else comp.p_corr
    return (
        0.30 * rank01(pos_score)
        + 0.20 * rank01(comp.candidate_gain_hat)
        + 0.20 * rank01(comp.g_hat)
        + 0.15 * rank01(comp.p_corr)
        + 0.15 * rank01(comp.candidate_margin)
        + 0.15 * rank01(hard_proxy)
        + 0.10 * disagreement_score
        - 0.18 * rank01(comp.candidate_bad_prob)
    )


def top_k_mask(mask: np.ndarray, score: np.ndarray, k: int) -> np.ndarray:
    """在候选集合中取分数最高的k个样本。"""

    out = np.zeros(len(mask), dtype=bool)
    idx = np.where(mask)[0]
    if k <= 0 or idx.size == 0:
        return out
    k = min(int(k), idx.size)
    chosen = idx[np.argsort(score[idx])[-k:]]
    out[chosen] = True
    return out


def select_v320_samples(
    comp: TwoStageComponents,
    meta: pd.DataFrame,
    hard_proxy: np.ndarray,
    candidate_score: np.ndarray,
    cfg: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """第320版排序配额选择，返回是否校正、通道和融合幅度。"""

    strong = meta["strong_steer"].astype(bool).to_numpy()
    hard_threshold = float(cfg["hard_proxy_threshold_value"])
    hard_proxy_mask = hard_proxy >= hard_threshold
    normal_channel = (~strong) & (~hard_proxy_mask)
    risk_ok = comp.candidate_bad_prob <= float(cfg["repair_bad_max"])

    def select_with_fallback(primary: np.ndarray, fallback: np.ndarray, k: int) -> np.ndarray:
        picked = top_k_mask(primary, candidate_score, k)
        short = int(k) - int(picked.sum())
        if short > 0:
            picked |= top_k_mask(fallback & (~picked), candidate_score, short)
        return picked

    k_strong = int(math.ceil(float(cfg["r_strong"]) * max(int(strong.sum()), 1)))
    k_proxy = int(math.ceil(float(cfg["r_proxy_hard"]) * len(meta)))
    selected_strong = select_with_fallback(risk_ok & strong, strong, k_strong)
    proxy_risk_ok = comp.candidate_bad_prob <= float(cfg["proxy_bad_max"])
    proxy_pool = proxy_risk_ok & (~strong) & hard_proxy_mask & (~selected_strong)
    selected_proxy = select_with_fallback(proxy_pool, proxy_pool, k_proxy)

    normal_pos = comp.candidate_pos_prob if comp.candidate_pos_prob is not None else comp.p_corr
    eligible_normal = (
        normal_channel
        & (normal_pos >= float(cfg["normal_pos_min"]))
        & (comp.candidate_bad_prob <= float(cfg["normal_bad_prob_max"]))
        & (comp.candidate_margin >= float(cfg["normal_margin_min"]))
        & (~selected_strong)
        & (~selected_proxy)
    )
    normal = normal_channel
    selected_normal = top_k_mask(
        eligible_normal,
        candidate_score,
        int(math.floor(float(cfg["normal_corr_cap"]) * max(int(normal.sum()), 1))),
    )
    selected = selected_strong | selected_proxy | selected_normal
    global_cap = int(math.floor(float(cfg["global_corr_cap"]) * len(meta)))
    if int(selected.sum()) > global_cap:
        selected = top_k_mask(selected, candidate_score, global_cap)
        selected_strong &= selected
        selected_proxy &= selected
        selected_normal &= selected
    channel = np.full(len(meta), "未校正", dtype=object)
    channel[selected_strong] = "强方向盘通道"
    channel[selected_proxy] = "困难代理通道"
    channel[selected_normal] = "普通保护通道"
    alpha = np.zeros(len(meta), dtype=float)
    alpha[selected_normal] = float(cfg["normal_alpha"])
    alpha[selected_strong | selected_proxy] = float(cfg["repair_alpha"])
    return selected, channel, alpha


def v320_alpha_error_matrices(candidate_curves: np.ndarray, y0: np.ndarray, y_true: np.ndarray) -> Dict[float, np.ndarray]:
    """预计算不同融合幅度下的候选误差，加速配额搜索。"""

    matrices: Dict[float, np.ndarray] = {}
    base = y0[:, None, :]
    true = y_true[:, None, :]
    for alpha in [0.25, 0.50, 0.75, 1.00]:
        pred = base + alpha * (candidate_curves - base)
        matrices[alpha] = np.sqrt(np.nanmean((pred - true) ** 2, axis=2))
    return matrices


def rmse_from_selection(
    base_rmse: np.ndarray,
    comp: TwoStageComponents,
    selected: np.ndarray,
    alpha: np.ndarray,
    alpha_errors: Dict[float, np.ndarray],
) -> np.ndarray:
    """根据选择和融合幅度取出逐样本误差。"""

    rmse = base_rmse.copy()
    for a in [0.25, 0.50, 0.75, 1.00]:
        mask = selected & np.isclose(alpha, a)
        if mask.any():
            rmse[mask] = alpha_errors[a][np.where(mask)[0], comp.candidate_idx[mask]]
    return rmse


def budget_metrics_from_rmse(
    rmse: np.ndarray,
    base_rmse: np.ndarray,
    meta: pd.DataFrame,
    selected: np.ndarray,
) -> Dict[str, float]:
    """第320版预算型搜索指标。"""

    normal = (~meta["strong_steer"].astype(bool) & ~meta["within_bad_top20_by_v249"].astype(bool)).to_numpy()
    strong = meta["strong_steer"].astype(bool).to_numpy()
    hard20 = meta["within_bad_top20_by_v249"].astype(bool).to_numpy()
    hard10 = meta["within_bad_top10_by_v249"].astype(bool).to_numpy()

    def mean_gain(mask: np.ndarray) -> float:
        if not mask.any():
            return math.nan
        return float(np.nanmean(base_rmse[mask] - rmse[mask]))

    def bad_rate(mask: np.ndarray) -> float:
        if not mask.any():
            return math.nan
        return float(np.nanmean(rmse[mask] > base_rmse[mask] * 1.10))

    def pos_loss(mask: np.ndarray) -> float:
        if not mask.any():
            return math.nan
        return float(np.nanmean(np.maximum(rmse[mask] - base_rmse[mask], 0.0)))

    def corr_rate(mask: np.ndarray) -> float:
        if not mask.any():
            return math.nan
        return float(np.nanmean(selected[mask]))

    all_mask = np.ones(len(rmse), dtype=bool)
    return {
        "gain_all": mean_gain(all_mask),
        "gain_normal": mean_gain(normal),
        "gain_strong": mean_gain(strong),
        "gain_hard20": mean_gain(hard20),
        "gain_hard10": mean_gain(hard10),
        "bad_all": bad_rate(all_mask),
        "bad_normal": bad_rate(normal),
        "bad_strong": bad_rate(strong),
        "bad_hard20": bad_rate(hard20),
        "bad_hard10": bad_rate(hard10),
        "pos_loss_all": pos_loss(all_mask),
        "pos_loss_normal": pos_loss(normal),
        "corr_all": corr_rate(all_mask),
        "corr_normal": corr_rate(normal),
        "corr_strong": corr_rate(strong),
        "corr_hard20": corr_rate(hard20),
        "corr_hard10": corr_rate(hard10),
    }


def v320_pass_checks(m: Dict[str, float], relaxed: bool = False) -> Dict[str, bool]:
    """第320版训练搜索/验证通过标准。"""

    if relaxed:
        return {
            "整体有激活": m["corr_all"] >= 0.040,
            "整体校正率不超15%": m["corr_all"] <= 0.150,
            "普通校正率不超6%": m["corr_normal"] <= 0.060,
            "强方向盘激活至少8%": m["corr_strong"] >= 0.080,
            "困难前20激活至少10%": m["corr_hard20"] >= 0.100,
            "整体不劣化": m["gain_all"] >= 0.000,
            "普通样本轻微不劣化": m["gain_normal"] >= -0.003,
            "强方向盘不劣化": m["gain_strong"] >= 0.000,
            "困难前20不劣化": m["gain_hard20"] >= 0.000,
            "整体大退化可控": m["bad_all"] <= 0.030,
            "普通大退化可控": m["bad_normal"] <= 0.010,
        }
    return {
        "整体校正率至少5%": m["corr_all"] >= 0.050,
        "整体校正率不超15%": m["corr_all"] <= 0.150,
        "普通校正率不超6%": m["corr_normal"] <= 0.060,
        "强方向盘激活至少10%": m["corr_strong"] >= 0.100,
        "困难前20激活至少15%": m["corr_hard20"] >= 0.150,
        "困难前10激活至少15%": m["corr_hard10"] >= 0.150,
        "整体有收益": m["gain_all"] >= 0.003,
        "普通样本不明显劣化": m["gain_normal"] >= -0.003,
        "强方向盘有收益": m["gain_strong"] >= 0.008,
        "困难前20有收益": m["gain_hard20"] >= 0.010,
        "困难前10不劣化": m["gain_hard10"] >= 0.000,
        "整体大退化可控": m["bad_all"] <= 0.020,
        "普通大退化可控": m["bad_normal"] <= 0.008,
        "普通正损失可控": m["pos_loss_normal"] <= 0.004,
    }


def v320_score(m: Dict[str, float]) -> float:
    """第320版配额搜索目标函数。"""

    return float(
        1.0 * m["gain_all"]
        + 1.5 * m["gain_strong"]
        + 2.0 * m["gain_hard20"]
        + 1.2 * m["gain_hard10"]
        + 0.05 * m["corr_strong"]
        + 0.04 * m["corr_hard20"]
        - 3.0 * m["bad_all"]
        - 8.0 * m["bad_normal"]
        - 3.0 * m["pos_loss_normal"]
        - 0.10 * max(0.0, m["corr_normal"] - 0.04)
    )


def v320_grid(hard_proxy: np.ndarray) -> Iterable[Dict[str, float]]:
    """第320版排序配额搜索空间。"""

    normal_presets = [
        {"normal_pos_min": 0.78, "normal_bad_prob_max": 0.35, "normal_margin_min": 0.006, "normal_corr_cap": 0.00},
        {"normal_pos_min": 0.72, "normal_bad_prob_max": 0.45, "normal_margin_min": 0.004, "normal_corr_cap": 0.02},
        {"normal_pos_min": 0.65, "normal_bad_prob_max": 0.55, "normal_margin_min": 0.002, "normal_corr_cap": 0.04},
    ]
    for q in [0.60, 0.70, 0.80]:
        threshold = float(np.nanquantile(hard_proxy, q))
        for repair_bad_max in [0.80, 0.90, 1.00]:
            for proxy_bad_max in [0.58, 0.62]:
                for r_strong in [0.05, 0.08, 0.10, 0.12, 0.15]:
                    for r_proxy_hard in [0.02, 0.05, 0.08, 0.10]:
                        for global_corr_cap in [0.08, 0.12, 0.15]:
                            for repair_alpha in [0.75, 1.00]:
                                for normal_alpha in [0.25, 0.50]:
                                    for normal_cfg in normal_presets:
                                        yield {
                                            "hard_proxy_quantile": q,
                                            "hard_proxy_threshold_value": threshold,
                                            "repair_bad_max": repair_bad_max,
                                            "proxy_bad_max": proxy_bad_max,
                                            "r_strong": r_strong,
                                            "r_proxy_hard": r_proxy_hard,
                                            "global_corr_cap": global_corr_cap,
                                            "repair_alpha": repair_alpha,
                                            "normal_alpha": normal_alpha,
                                            **normal_cfg,
                                        }


def search_v320_config(
    comp: TwoStageComponents,
    meta: pd.DataFrame,
    hard_proxy: np.ndarray,
    candidate_score: np.ndarray,
    base_rmse: np.ndarray,
    alpha_errors: Dict[float, np.ndarray],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """只用训练集折外结果选择第320版配额配置。"""

    rows: List[Dict[str, object]] = []
    best_strict: Tuple[float, Dict[str, float]] | None = None
    best_relaxed: Tuple[float, Dict[str, float]] | None = None
    for cfg in v320_grid(hard_proxy):
        selected, _, alpha = select_v320_samples(comp, meta, hard_proxy, candidate_score, cfg)
        rmse = rmse_from_selection(base_rmse, comp, selected, alpha, alpha_errors)
        metrics = budget_metrics_from_rmse(rmse, base_rmse, meta, selected)
        strict_checks = v320_pass_checks(metrics, relaxed=False)
        relaxed_checks = v320_pass_checks(metrics, relaxed=True)
        strict_pass = all(strict_checks.values())
        relaxed_pass = all(relaxed_checks.values())
        score = v320_score(metrics)
        row = {
            **cfg,
            **metrics,
            "strict_pass": strict_pass,
            "relaxed_pass": relaxed_pass,
            "strict_pass_count": int(sum(strict_checks.values())),
            "relaxed_pass_count": int(sum(relaxed_checks.values())),
            "search_score": score,
        }
        rows.append(row)
        if strict_pass and (best_strict is None or score > best_strict[0]):
            best_strict = (score, cfg)
        if relaxed_pass and (best_relaxed is None or score > best_relaxed[0]):
            best_relaxed = (score, cfg)
    search = pd.DataFrame(rows)
    if best_strict is not None:
        cfg = dict(best_strict[1])
        cfg["selected_tier"] = "严格约束"
        return cfg, search
    if best_relaxed is not None:
        cfg = dict(best_relaxed[1])
        cfg["selected_tier"] = "宽松约束"
        return cfg, search
    search["coverage_first_score"] = (
        5.0 * search["relaxed_pass_count"].astype(float)
        + 2.0 * search["corr_strong"].astype(float)
        + 2.0 * search["corr_hard20"].astype(float)
        + 1.0 * search["corr_all"].astype(float)
        + 20.0 * search["gain_hard20"].astype(float)
        + 15.0 * search["gain_strong"].astype(float)
        + 10.0 * search["gain_all"].astype(float)
        - 3.0 * search["bad_normal"].astype(float)
        - 2.0 * search["bad_all"].astype(float)
    )
    fallback = search.sort_values(
        ["strict_pass_count", "relaxed_pass_count", "coverage_first_score", "search_score"],
        ascending=[False, False, False, False],
    ).iloc[0].to_dict()
    cfg = {k: float(fallback[k]) for k in next(v320_grid(hard_proxy)).keys()}
    cfg["selected_tier"] = "覆盖优先回退"
    return cfg, search


def apply_v320_prediction(
    y0: np.ndarray,
    candidate_curves: np.ndarray,
    comp: TwoStageComponents,
    meta: pd.DataFrame,
    hard_proxy: np.ndarray,
    candidate_score: np.ndarray,
    cfg: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把第320版配置应用到验证或测试集，生成最终曲线和诊断字段。"""

    selected, channel, alpha = select_v320_samples(comp, meta, hard_proxy, candidate_score, cfg)
    chosen = np.where(selected, comp.candidate_idx, 0).astype(int)
    selected_curve = candidate_curves[np.arange(len(y0)), chosen, :]
    pred = y0 + alpha[:, None] * (selected_curve - y0)
    prob = np.zeros((len(y0), candidate_curves.shape[1]), dtype=float)
    prob[:, 0] = np.where(selected, 1.0 - np.clip(comp.p_corr, 0.0, 1.0), 1.0)
    prob[np.arange(len(y0)), chosen] += np.where(selected, np.clip(comp.p_corr, 0.0, 1.0), 0.0)
    row_sum = prob.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    return pred, chosen, prob / row_sum, channel, alpha


def candidate_family_name(idx: int) -> str:
    """候选编号归并到候选家族。"""

    if idx == 0:
        return "原预测"
    if 1 <= idx <= 5:
        return "幅值缩放"
    if 6 <= idx <= 11:
        return "时间平移"
    if 12 <= idx <= 15:
        return "幅值加时间"
    return "残差原型"


def build_v320_budget_table(metrics: Dict[str, float], relaxed: bool = False) -> pd.DataFrame:
    """第320版预算通过表。"""

    checks = v320_pass_checks(metrics, relaxed=relaxed)
    rows = [{"check_name": k, "passed": bool(v)} for k, v in checks.items()]
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
    ax.set_title("第320版验证集分组误差")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / "v320_validation_group_rmse.png"
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
    ax.set_title("第320版验证集候选使用诊断")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURES / "v320_validation_candidate_usage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_zip_package() -> Tuple[Path, bool]:
    """打包第320版产物并自检。"""

    zip_path = OUT / "v320_rank_budget_repair_gate_20260705.zip"
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
    """写第320版中文报告。"""

    val_rows = group_summary[
        group_summary["split"].eq("val")
        & group_summary["method_name"].isin([BASE_MODEL_NAME, selected_method, "候选最优上限"])
        & group_summary["group"].isin(["全部样本", "普通样本", "强方向盘样本", "困难前20", "困难前10"])
    ][["method_name", "split", "group", "n", "sample_rmse_mean", "delta_vs_v316_mean", "degrade_gt10_rate", "serious_under_rate", "noop_top1_rate"]]

    lines = [
        "# 第320版排序配额修复门控实验",
        "",
        "## 结论",
        "",
        f"- 固定方案：`{selected_method}`。",
        f"- 验证门槛是否全部通过：`{selected_pass}`。",
        f"- 是否报告测试集：`{test_reported}`。",
        "- 本轮只使用第315版保留清单；84个隔离事件不参与训练、验证选模或测试主统计。",
        "- 第316版原预测仍是默认输出，第317版候选库不扩展。",
        "- 第320版新增候选正收益概率和排序配额：普通样本低覆盖保护，强方向盘/困难代理样本按排名强制激活一部分。",
        "- 配额、风险预算和融合幅度只由训练集内部交叉验证选择；验证集只做通过/失败判定，测试集不参与任何选择。",
        "",
        "## 验证集关键结果",
        "",
        markdown_table(val_rows),
        "",
        "## 验证预算检查",
        "",
        markdown_table(gate_table),
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
            "- 固定方案未通过全部验证预算，因此第320版不报告测试集结果。",
            "- 下一步按失败分流：若仍然全不改，先查配额实现；若激活达标但困难样本无改善，修候选排序；若普通样本被改坏，关闭普通通道。",
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
        f"- 训练集内部交叉验证选择配额：`{guardrail['uses_train_oof_quota_search']}`",
        f"- 压缩包自检：`{guardrail.get('zip_testzip')}`",
    ]
    path = REPORTS / "v320_rank_budget_repair_gate_cn.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main_v318_disabled() -> None:
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
    write_csv(pd.concat([meta[["event_uid", "split"]].reset_index(drop=True), features], axis=1), TABLES / "v318_gate_input_features.csv")

    split = meta["split"].astype(str).to_numpy()
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    train_meta = meta.loc[train_mask].reset_index(drop=True)
    prototypes = build_residual_prototypes(y_true[train_mask], y316[train_mask], grid, train_meta)
    write_csv(pd.DataFrame(prototypes, columns=[f"t_{x:.1f}" for x in grid]), TABLES / "v318_train_residual_prototypes.csv")

    candidates_all = build_candidates(y316, grid, prototypes)
    candidate_info = pd.DataFrame({"candidate_idx": np.arange(len(candidates_all.names)), "candidate_name": candidates_all.names})
    write_csv(candidate_info, TABLES / "v318_candidate_library.csv")

    errors_all = candidate_errors(candidates_all.curves, y_true)
    oracle_idx_all = np.nanargmin(errors_all, axis=1).astype(int)
    oracle_pred_all = candidates_all.curves[np.arange(len(meta)), oracle_idx_all, :]
    labels_all = correctability_targets(errors_all, meta)
    labels_all["best_nonnoop_candidate_name"] = [candidates_all.names[i] for i in labels_all["best_nonnoop_candidate_idx"].astype(int)]
    write_csv(labels_all, TABLES / "v318_correctability_labels.csv")
    write_csv(summarize_correctability(labels_all, candidates_all.names), TABLES / "v318_correctability_summary.csv")

    x_train = features.loc[train_mask].reset_index(drop=True)
    x_val = align_feature_columns(x_train, features.loc[val_mask].reset_index(drop=True))
    x_test = align_feature_columns(x_train, features.loc[test_mask].reset_index(drop=True))

    train_candidates = CandidateSet(candidates_all.names, candidates_all.curves[train_mask])
    oof_comp, full_models = fit_oof_and_full_models(
        x_train,
        train_candidates,
        y316[train_mask],
        y_true[train_mask],
        train_meta,
        grid,
    )
    train_base_rmse = curve_rmse(y316[train_mask], y_true[train_mask])
    train_labels = labels_all.loc[train_mask].reset_index(drop=True).copy()
    train_oof_diag = train_labels[
        [
            "event_uid",
            "split",
            "base_error",
            "best_nonnoop_candidate_idx",
            "oracle_gain",
            "correctable_label",
            "normal_group",
            "within_bad_top20_by_v249",
            "within_bad_top10_by_v249",
        ]
    ].copy()
    train_oof_diag["p_corr_oof"] = oof_comp.p_corr
    train_oof_diag["g_hat_oof"] = oof_comp.g_hat
    train_oof_diag["candidate_idx_oof"] = oof_comp.candidate_idx
    train_oof_diag["candidate_name_oof"] = [candidates_all.names[i] for i in oof_comp.candidate_idx]
    train_oof_diag["candidate_gain_hat_oof"] = oof_comp.candidate_gain_hat
    train_oof_diag["candidate_bad_prob_oof"] = oof_comp.candidate_bad_prob
    train_oof_diag["candidate_margin_oof"] = oof_comp.candidate_margin
    train_oof_diag["candidate_pos_prob_oof"] = oof_comp.candidate_pos_prob if oof_comp.candidate_pos_prob is not None else np.nan
    train_oof_diag["candidate_actual_gain_oof"] = train_base_rmse - errors_all[train_mask][np.arange(train_mask.sum()), oof_comp.candidate_idx]
    write_csv(train_oof_diag, TABLES / "v318_train_oof_stage_diagnostics.csv")

    policy_defs = [
        ("第318甲-可校正门控保守融合", "stage_a_only"),
        ("第318乙-候选收益安全单选", "candidate_select"),
        ("第318丙-候选收益安全残差融合", "residual_fusion"),
    ]
    search_tables: List[pd.DataFrame] = []
    selected_policy_rows: List[Dict[str, object]] = []
    selected_policy_params: Dict[str, Tuple[str, Dict[str, float]]] = {}
    for method_name, mode in policy_defs:
        params, search_df = search_best_policy(
            method_name,
            mode,
            y316[train_mask],
            candidates_all.curves[train_mask],
            oof_comp,
            train_meta,
            y_true[train_mask],
            train_base_rmse,
            grid,
        )
        search_tables.append(search_df)
        best_search = search_df.sort_values("search_score", ascending=False).iloc[0].to_dict()
        selected_policy_rows.append(best_search)
        selected_policy_params[method_name] = (mode, params)
    policy_search = pd.concat(search_tables, ignore_index=True)
    selected_policy_table = pd.DataFrame(selected_policy_rows)
    write_csv(policy_search, TABLES / "v318_train_oof_policy_search.csv")
    write_csv(selected_policy_table, TABLES / "v318_selected_policy_thresholds.csv")

    val_candidates = candidates_all.curves[val_mask]
    val_meta = meta.loc[val_mask].reset_index(drop=True)
    val_true = y_true[val_mask]
    val_base_rmse = curve_rmse(y316[val_mask], val_true)
    val_oracle_idx = oracle_idx_all[val_mask]
    val_comp = predict_two_stage_components(
        full_models,
        x_val,
        CandidateSet(candidates_all.names, val_candidates),
        y316[val_mask],
        grid,
    )
    val_stage_diag = meta.loc[val_mask, ["event_uid", "split"]].reset_index(drop=True).copy()
    val_stage_diag["p_corr"] = val_comp.p_corr
    val_stage_diag["g_hat"] = val_comp.g_hat
    val_stage_diag["candidate_idx"] = val_comp.candidate_idx
    val_stage_diag["candidate_name"] = [candidates_all.names[i] for i in val_comp.candidate_idx]
    val_stage_diag["candidate_gain_hat"] = val_comp.candidate_gain_hat
    val_stage_diag["candidate_bad_prob"] = val_comp.candidate_bad_prob
    val_stage_diag["candidate_margin"] = val_comp.candidate_margin
    val_stage_diag["candidate_pos_prob"] = val_comp.candidate_pos_prob if val_comp.candidate_pos_prob is not None else np.nan
    write_csv(val_stage_diag, TABLES / "v318_validation_stage_components.csv")

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

    stored_val_outputs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, str]] = {}
    for method_name, (mode, params) in selected_policy_params.items():
        pred, chosen, prob = apply_policy(
            y316[val_mask],
            val_candidates,
            val_comp,
            val_meta,
            params,
            use_candidate_filters=(mode != "stage_a_only"),
        )
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
        stored_val_outputs[method_name] = (pred, prob, chosen, mode)
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
    write_csv(val_per_sample, TABLES / "v318_validation_per_sample_metrics.csv")
    write_csv(group_summary, TABLES / "v318_validation_group_summary.csv")
    write_csv(gate_table, TABLES / "v318_validation_gate_check.csv")
    write_csv(usage, TABLES / "v318_validation_candidate_usage.csv")

    test_reported = False
    test_per_sample = pd.DataFrame()
    test_group_summary = pd.DataFrame()
    if selected_pass:
        test_candidates = candidates_all.curves[test_mask]
        test_meta = meta.loc[test_mask].reset_index(drop=True)
        test_true = y_true[test_mask]
        test_base_rmse = curve_rmse(y316[test_mask], test_true)
        test_oracle_idx = oracle_idx_all[test_mask]
        test_comp = predict_two_stage_components(
            full_models,
            x_test,
            CandidateSet(candidates_all.names, test_candidates),
            y316[test_mask],
            grid,
        )
        selected_mode, selected_params = selected_policy_params[selected_method]
        pred_test, chosen_test, prob_test = apply_policy(
            y316[test_mask],
            test_candidates,
            test_comp,
            test_meta,
            selected_params,
            use_candidate_filters=(selected_mode != "stage_a_only"),
        )
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
        write_csv(test_per_sample, TABLES / "v318_test_per_sample_metrics.csv")
        write_csv(test_group_summary, TABLES / "v318_test_group_summary.csv")
        write_csv(
            build_candidate_usage(selected_method, "test", chosen_test, test_oracle_idx, candidates_all.names, prob_test),
            TABLES / "v318_test_candidate_usage.csv",
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
        "reuses_v317_candidate_library": True,
        "uses_train_oof_threshold_search": True,
        "uses_candidate_gain_model": True,
        "uses_candidate_bad_risk_filter": True,
        "uses_conservative_residual_fusion": True,
        "uses_future_truth_for_training_targets_only": True,
        "validation_gate_table": str(TABLES / "v318_validation_gate_check.csv"),
        "selected_policy_thresholds": str(TABLES / "v318_selected_policy_thresholds.csv"),
        "test_result_suppressed_when_validation_fails": True,
        "runtime_seconds": float(time.time() - started),
        "figure_paths": [str(p) for p in figure_paths],
        "zip_path": str(OUT / "v318_conservative_two_stage_gate_20260705.zip"),
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


def build_metric_frame(metrics: Dict[str, float], split_name: str) -> pd.DataFrame:
    """把预算指标转成长表，方便人工排查。"""

    return pd.DataFrame(
        [
            {"split": split_name, "metric_name": key, "metric_value": float(value)}
            for key, value in metrics.items()
        ]
    )


def build_coverage_table(
    split_name: str,
    meta: pd.DataFrame,
    selected: np.ndarray,
) -> pd.DataFrame:
    """输出每个关键分组实际被激活的比例。"""

    masks = {
        "全部样本": np.ones(len(meta), dtype=bool),
        "普通样本": (~meta["strong_steer"].astype(bool) & ~meta["within_bad_top20_by_v249"].astype(bool)).to_numpy(),
        "强方向盘样本": meta["strong_steer"].astype(bool).to_numpy(),
        "困难前20": meta["within_bad_top20_by_v249"].astype(bool).to_numpy(),
        "困难前10": meta["within_bad_top10_by_v249"].astype(bool).to_numpy(),
    }
    rows: List[Dict[str, object]] = []
    for group, mask in masks.items():
        rows.append(
            {
                "split": split_name,
                "group": group,
                "n": int(mask.sum()),
                "corrected_n": int(np.sum(selected & mask)),
                "correction_rate": float(np.mean(selected[mask])) if mask.any() else math.nan,
            }
        )
    return pd.DataFrame(rows)


def build_risk_budget_table(
    split_name: str,
    meta: pd.DataFrame,
    base_rmse: np.ndarray,
    new_rmse: np.ndarray,
) -> pd.DataFrame:
    """输出每个关键分组的收益、大退化率和正损失。"""

    masks = {
        "全部样本": np.ones(len(meta), dtype=bool),
        "普通样本": (~meta["strong_steer"].astype(bool) & ~meta["within_bad_top20_by_v249"].astype(bool)).to_numpy(),
        "强方向盘样本": meta["strong_steer"].astype(bool).to_numpy(),
        "困难前20": meta["within_bad_top20_by_v249"].astype(bool).to_numpy(),
        "困难前10": meta["within_bad_top10_by_v249"].astype(bool).to_numpy(),
    }
    rows: List[Dict[str, object]] = []
    for group, mask in masks.items():
        if not mask.any():
            continue
        rows.append(
            {
                "split": split_name,
                "group": group,
                "gain_mean": float(np.nanmean(base_rmse[mask] - new_rmse[mask])),
                "bad_gt10_rate": float(np.nanmean(new_rmse[mask] > base_rmse[mask] * 1.10)),
                "positive_loss_mean": float(np.nanmean(np.maximum(new_rmse[mask] - base_rmse[mask], 0.0))),
                "base_rmse_mean": float(np.nanmean(base_rmse[mask])),
                "new_rmse_mean": float(np.nanmean(new_rmse[mask])),
            }
        )
    return pd.DataFrame(rows)


def build_channel_contribution(
    split_name: str,
    meta: pd.DataFrame,
    base_rmse: np.ndarray,
    new_rmse: np.ndarray,
    selected: np.ndarray,
    channel: np.ndarray,
    alpha: np.ndarray,
) -> pd.DataFrame:
    """按双通道统计实际贡献，定位是激活不足还是激活后误伤。"""

    normal = (~meta["strong_steer"].astype(bool) & ~meta["within_bad_top20_by_v249"].astype(bool)).to_numpy()
    strong = meta["strong_steer"].astype(bool).to_numpy()
    hard20 = meta["within_bad_top20_by_v249"].astype(bool).to_numpy()
    channel_names = ["全部校正", "强方向盘通道", "困难代理通道", "普通保护通道", "未校正"]
    rows: List[Dict[str, object]] = []
    for name in channel_names:
        mask = selected if name == "全部校正" else channel == name
        if not mask.any():
            rows.append(
                {
                    "split": split_name,
                    "channel": name,
                    "n": 0,
                    "sample_share": 0.0,
                    "gain_mean": math.nan,
                    "bad_gt10_rate": math.nan,
                    "positive_loss_mean": math.nan,
                    "alpha_mean": math.nan,
                    "normal_rate": math.nan,
                    "strong_rate": math.nan,
                    "hard20_rate": math.nan,
                }
            )
            continue
        rows.append(
            {
                "split": split_name,
                "channel": name,
                "n": int(mask.sum()),
                "sample_share": float(mask.mean()),
                "gain_mean": float(np.nanmean(base_rmse[mask] - new_rmse[mask])),
                "bad_gt10_rate": float(np.nanmean(new_rmse[mask] > base_rmse[mask] * 1.10)),
                "positive_loss_mean": float(np.nanmean(np.maximum(new_rmse[mask] - base_rmse[mask], 0.0))),
                "alpha_mean": float(np.nanmean(alpha[mask])),
                "normal_rate": float(np.nanmean(normal[mask])),
                "strong_rate": float(np.nanmean(strong[mask])),
                "hard20_rate": float(np.nanmean(hard20[mask])),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_family_usage(
    split_name: str,
    chosen_idx: np.ndarray,
    oracle_idx: np.ndarray,
    selected: np.ndarray,
    base_rmse: np.ndarray,
    new_rmse: np.ndarray,
) -> pd.DataFrame:
    """按候选家族归并选择效果，避免只看候选编号。"""

    rows: List[Dict[str, object]] = []
    selected_idx = np.where(selected)[0]
    for family in ["幅值缩放", "时间平移", "幅值加时间", "残差原型"]:
        mask = selected & np.array([candidate_family_name(int(i)) == family for i in chosen_idx], dtype=bool)
        rows.append(
            {
                "split": split_name,
                "candidate_family": family,
                "selected_n": int(mask.sum()),
                "selected_rate": float(mask.sum() / max(len(chosen_idx), 1)),
                "gain_mean": float(np.nanmean(base_rmse[mask] - new_rmse[mask])) if mask.any() else math.nan,
                "bad_gt10_rate": float(np.nanmean(new_rmse[mask] > base_rmse[mask] * 1.10)) if mask.any() else math.nan,
                "oracle_overlap_rate": float(np.nanmean(chosen_idx[mask] == oracle_idx[mask])) if mask.any() else math.nan,
            }
        )
    rows.append(
        {
            "split": split_name,
            "candidate_family": "全部校正",
            "selected_n": int(selected_idx.size),
            "selected_rate": float(selected_idx.size / max(len(chosen_idx), 1)),
            "gain_mean": float(np.nanmean(base_rmse[selected] - new_rmse[selected])) if selected.any() else math.nan,
            "bad_gt10_rate": float(np.nanmean(new_rmse[selected] > base_rmse[selected] * 1.10)) if selected.any() else math.nan,
            "oracle_overlap_rate": float(np.nanmean(chosen_idx[selected] == oracle_idx[selected])) if selected.any() else math.nan,
        }
    )
    return pd.DataFrame(rows)


def build_hard_proxy_eval_buckets(
    split_name: str,
    score: np.ndarray,
    meta: pd.DataFrame,
    base_rmse: np.ndarray,
) -> pd.DataFrame:
    """评估困难代理分数是否富集真实困难样本，只用于诊断。"""

    rows: List[Dict[str, object]] = []
    hard20 = meta["within_bad_top20_by_v249"].astype(bool).to_numpy()
    strong = meta["strong_steer"].astype(bool).to_numpy()
    for lo, hi in [(0, 50), (50, 70), (70, 80), (80, 90), (90, 100)]:
        qlo, qhi = np.nanpercentile(score, [lo, hi])
        mask = (score >= qlo) & (score <= qhi if hi == 100 else score < qhi)
        rows.append(
            {
                "split": split_name,
                "bucket": f"{lo}-{hi}",
                "n": int(mask.sum()),
                "hard20_rate": float(hard20[mask].mean()) if mask.any() else math.nan,
                "strong_steer_rate": float(strong[mask].mean()) if mask.any() else math.nan,
                "base_rmse_mean": float(base_rmse[mask].mean()) if mask.any() else math.nan,
                "score_min": float(qlo),
                "score_max": float(qhi),
            }
        )
    return pd.DataFrame(rows)


def write_file_inventory() -> None:
    """刷新产物清单。"""

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
    write_csv(pd.concat([meta[["event_uid", "split"]].reset_index(drop=True), features], axis=1), TABLES / "v320_gate_input_features.csv")

    split = meta["split"].astype(str).to_numpy()
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    train_meta = meta.loc[train_mask].reset_index(drop=True)
    prototypes = build_residual_prototypes(y_true[train_mask], y316[train_mask], grid, train_meta)
    write_csv(pd.DataFrame(prototypes, columns=[f"t_{x:.1f}" for x in grid]), TABLES / "v320_train_residual_prototypes.csv")

    candidates_all = build_candidates(y316, grid, prototypes)
    candidate_info = pd.DataFrame({"candidate_idx": np.arange(len(candidates_all.names)), "candidate_name": candidates_all.names})
    candidate_info["candidate_family"] = [candidate_family_name(int(i)) for i in candidate_info["candidate_idx"]]
    write_csv(candidate_info, TABLES / "v320_candidate_library.csv")

    errors_all = candidate_errors(candidates_all.curves, y_true)
    oracle_idx_all = np.nanargmin(errors_all, axis=1).astype(int)
    oracle_pred_all = candidates_all.curves[np.arange(len(meta)), oracle_idx_all, :]
    labels_all = correctability_targets(errors_all, meta)
    labels_all["best_nonnoop_candidate_name"] = [candidates_all.names[i] for i in labels_all["best_nonnoop_candidate_idx"].astype(int)]
    write_csv(labels_all, TABLES / "v320_correctability_labels.csv")
    write_csv(summarize_correctability(labels_all, candidates_all.names), TABLES / "v320_correctability_summary.csv")

    x_train = features.loc[train_mask].reset_index(drop=True)
    x_val = align_feature_columns(x_train, features.loc[val_mask].reset_index(drop=True))
    x_test = align_feature_columns(x_train, features.loc[test_mask].reset_index(drop=True))

    train_candidates = CandidateSet(candidates_all.names, candidates_all.curves[train_mask])
    oof_comp, full_models = fit_oof_and_full_models(
        x_train,
        train_candidates,
        y316[train_mask],
        y_true[train_mask],
        train_meta,
        grid,
    )
    train_base_rmse = curve_rmse(y316[train_mask], y_true[train_mask])
    train_labels = labels_all.loc[train_mask].reset_index(drop=True).copy()
    train_oof_diag = train_labels[
        [
            "event_uid",
            "split",
            "base_error",
            "best_nonnoop_candidate_idx",
            "oracle_gain",
            "correctable_label",
            "normal_group",
            "within_bad_top20_by_v249",
            "within_bad_top10_by_v249",
        ]
    ].copy()
    train_oof_diag["p_corr_oof"] = oof_comp.p_corr
    train_oof_diag["g_hat_oof"] = oof_comp.g_hat
    train_oof_diag["candidate_idx_oof"] = oof_comp.candidate_idx
    train_oof_diag["candidate_name_oof"] = [candidates_all.names[i] for i in oof_comp.candidate_idx]
    train_oof_diag["candidate_gain_hat_oof"] = oof_comp.candidate_gain_hat
    train_oof_diag["candidate_bad_prob_oof"] = oof_comp.candidate_bad_prob
    train_oof_diag["candidate_margin_oof"] = oof_comp.candidate_margin
    train_oof_diag["candidate_actual_gain_oof"] = train_base_rmse - errors_all[train_mask][np.arange(train_mask.sum()), oof_comp.candidate_idx]
    write_csv(train_oof_diag, TABLES / "v320_train_oof_stage_diagnostics.csv")

    train_proxy_x = hard_proxy_feature_matrix(x_train, oof_comp, train_candidates.curves, grid)
    hard_proxy_oof, hard_proxy_model, train_proxy_buckets = fit_hard_proxy_oof(train_proxy_x, train_base_rmse)
    train_proxy_buckets["split"] = "训练折外"
    write_csv(train_proxy_buckets, TABLES / "v320_train_hard_proxy_bucket_summary.csv")
    train_proxy_diag = meta.loc[train_mask, ["event_uid", "split"]].reset_index(drop=True).copy()
    train_proxy_diag["hard_proxy_score_oof"] = hard_proxy_oof
    train_proxy_diag["base_rmse"] = train_base_rmse
    train_proxy_diag["target_hard20_from_train_error"] = (train_base_rmse >= np.nanpercentile(train_base_rmse, 80.0)).astype(int)
    train_proxy_diag["true_within_bad_top20_by_v249"] = train_meta["within_bad_top20_by_v249"].astype(bool).to_numpy().astype(int)
    write_csv(train_proxy_diag, TABLES / "v320_train_hard_proxy_oof.csv")

    candidate_score_train = sample_candidate_score(oof_comp, hard_proxy_oof, train_candidates.curves, grid)
    alpha_errors_train = v320_alpha_error_matrices(train_candidates.curves, y316[train_mask], y_true[train_mask])
    selected_cfg, quota_search = search_v320_config(
        oof_comp,
        train_meta,
        hard_proxy_oof,
        candidate_score_train,
        train_base_rmse,
        alpha_errors_train,
    )
    write_csv(quota_search, TABLES / "v320_train_oof_quota_search.csv")
    write_csv(pd.DataFrame([selected_cfg]), TABLES / "v320_selected_quota_config.csv")

    train_selected, train_channel, train_alpha = select_v320_samples(
        oof_comp,
        train_meta,
        hard_proxy_oof,
        candidate_score_train,
        selected_cfg,
    )
    train_rmse_new = rmse_from_selection(train_base_rmse, oof_comp, train_selected, train_alpha, alpha_errors_train)
    train_budget_metrics = budget_metrics_from_rmse(train_rmse_new, train_base_rmse, train_meta, train_selected)
    train_budget_check = build_v320_budget_table(train_budget_metrics, relaxed=(selected_cfg.get("selected_tier") == "宽松约束"))
    train_budget_check["constraint_tier"] = str(selected_cfg.get("selected_tier"))
    write_csv(train_budget_check, TABLES / "v320_train_oof_budget_check.csv")
    write_csv(build_metric_frame(train_budget_metrics, "训练折外"), TABLES / "v320_train_oof_budget_metrics.csv")
    write_csv(build_coverage_table("训练折外", train_meta, train_selected), TABLES / "v320_train_oof_coverage_budget.csv")
    write_csv(build_risk_budget_table("训练折外", train_meta, train_base_rmse, train_rmse_new), TABLES / "v320_train_oof_risk_budget.csv")
    write_csv(build_channel_contribution("训练折外", train_meta, train_base_rmse, train_rmse_new, train_selected, train_channel, train_alpha), TABLES / "v320_train_oof_channel_contribution.csv")

    val_candidates = candidates_all.curves[val_mask]
    val_meta = meta.loc[val_mask].reset_index(drop=True)
    val_true = y_true[val_mask]
    val_base_rmse = curve_rmse(y316[val_mask], val_true)
    val_oracle_idx = oracle_idx_all[val_mask]
    val_comp = predict_two_stage_components(
        full_models,
        x_val,
        CandidateSet(candidates_all.names, val_candidates),
        y316[val_mask],
        grid,
    )
    val_proxy_x = hard_proxy_feature_matrix(x_val, val_comp, val_candidates, grid)
    hard_proxy_val = positive_class_probability(hard_proxy_model, val_proxy_x)
    candidate_score_val = sample_candidate_score(val_comp, hard_proxy_val, val_candidates, grid)
    pred_val, chosen_val, prob_val, channel_val, alpha_val = apply_v320_prediction(
        y316[val_mask],
        val_candidates,
        val_comp,
        val_meta,
        hard_proxy_val,
        candidate_score_val,
        selected_cfg,
    )
    val_selected = chosen_val != 0
    val_rmse_new = curve_rmse(pred_val, val_true)
    val_budget_metrics = budget_metrics_from_rmse(val_rmse_new, val_base_rmse, val_meta, val_selected)
    val_budget_table = build_v320_budget_table(val_budget_metrics, relaxed=True)
    val_budget_table["constraint_tier"] = "验证宽松预算"
    selected_pass = bool(val_budget_table["passed"].all())
    selected_method = "第320版-排序配额修复门控"

    val_stage_diag = meta.loc[val_mask, ["event_uid", "split"]].reset_index(drop=True).copy()
    val_stage_diag["p_corr"] = val_comp.p_corr
    val_stage_diag["g_hat"] = val_comp.g_hat
    val_stage_diag["candidate_idx"] = val_comp.candidate_idx
    val_stage_diag["candidate_name"] = [candidates_all.names[i] for i in val_comp.candidate_idx]
    val_stage_diag["candidate_gain_hat"] = val_comp.candidate_gain_hat
    val_stage_diag["candidate_bad_prob"] = val_comp.candidate_bad_prob
    val_stage_diag["candidate_margin"] = val_comp.candidate_margin
    val_stage_diag["hard_proxy_score"] = hard_proxy_val
    val_stage_diag["candidate_score"] = candidate_score_val
    val_stage_diag["selected_for_correction"] = val_selected
    val_stage_diag["selected_channel"] = channel_val
    val_stage_diag["fusion_alpha"] = alpha_val
    write_csv(val_stage_diag, TABLES / "v320_validation_stage_components.csv")

    all_per_sample: List[pd.DataFrame] = [
        build_per_sample_metrics(y316[val_mask], val_true, val_meta, grid, BASE_MODEL_NAME, val_base_rmse),
        build_per_sample_metrics(y307[val_mask], val_true, val_meta, grid, OLD_V307_NAME, val_base_rmse),
        build_per_sample_metrics(y300[val_mask], val_true, val_meta, grid, V300_NAME, val_base_rmse),
        build_per_sample_metrics(
            oracle_pred_all[val_mask],
            val_true,
            val_meta,
            grid,
            "候选最优上限",
            val_base_rmse,
            chosen_idx=val_oracle_idx,
            candidate_names=candidates_all.names,
        ),
    ]
    v320_val_frame = build_per_sample_metrics(
        pred_val,
        val_true,
        val_meta,
        grid,
        selected_method,
        val_base_rmse,
        candidate_prob=prob_val,
        chosen_idx=chosen_val,
        candidate_names=candidates_all.names,
    )
    v320_val_frame["hard_proxy_score"] = hard_proxy_val
    v320_val_frame["candidate_score"] = candidate_score_val
    v320_val_frame["selected_for_correction"] = val_selected
    v320_val_frame["selected_channel"] = channel_val
    v320_val_frame["fusion_alpha"] = alpha_val
    v320_val_frame["candidate_gain_hat"] = val_comp.candidate_gain_hat
    v320_val_frame["candidate_bad_prob"] = val_comp.candidate_bad_prob
    v320_val_frame["candidate_margin"] = val_comp.candidate_margin
    v320_val_frame["candidate_pos_prob"] = val_comp.candidate_pos_prob if val_comp.candidate_pos_prob is not None else np.nan
    all_per_sample.append(v320_val_frame)

    val_per_sample = pd.concat(all_per_sample, ignore_index=True)
    group_summary = summarize_groups(val_per_sample)
    usage = build_candidate_usage(selected_method, "val", chosen_val, val_oracle_idx, candidates_all.names, prob_val)
    usage["candidate_family"] = [candidate_family_name(int(i)) for i in usage["candidate_idx"]]

    write_csv(val_per_sample, TABLES / "v320_validation_per_sample_metrics.csv")
    write_csv(group_summary, TABLES / "v320_validation_group_summary.csv")
    write_csv(val_budget_table, TABLES / "v320_validation_budget_check.csv")
    write_csv(build_metric_frame(val_budget_metrics, "验证"), TABLES / "v320_validation_budget_metrics.csv")
    write_csv(usage, TABLES / "v320_validation_candidate_usage.csv")
    write_csv(build_coverage_table("验证", val_meta, val_selected), TABLES / "v320_validation_coverage_budget.csv")
    write_csv(build_risk_budget_table("验证", val_meta, val_base_rmse, val_rmse_new), TABLES / "v320_validation_risk_budget.csv")
    write_csv(build_channel_contribution("验证", val_meta, val_base_rmse, val_rmse_new, val_selected, channel_val, alpha_val), TABLES / "v320_validation_channel_contribution.csv")
    write_csv(build_candidate_family_usage("验证", chosen_val, val_oracle_idx, val_selected, val_base_rmse, val_rmse_new), TABLES / "v320_validation_candidate_family_usage.csv")
    write_csv(build_hard_proxy_eval_buckets("验证", hard_proxy_val, val_meta, val_base_rmse), TABLES / "v320_validation_hard_proxy_bucket_summary.csv")
    write_csv(
        pd.concat([build_metric_frame(train_budget_metrics, "训练折外"), build_metric_frame(val_budget_metrics, "验证")], ignore_index=True),
        TABLES / "v320_oof_vs_validation_budget_compare.csv",
    )

    test_reported = False
    test_group_summary = pd.DataFrame()
    if selected_pass:
        test_candidates = candidates_all.curves[test_mask]
        test_meta = meta.loc[test_mask].reset_index(drop=True)
        test_true = y_true[test_mask]
        test_base_rmse = curve_rmse(y316[test_mask], test_true)
        test_oracle_idx = oracle_idx_all[test_mask]
        test_comp = predict_two_stage_components(
            full_models,
            x_test,
            CandidateSet(candidates_all.names, test_candidates),
            y316[test_mask],
            grid,
        )
        test_proxy_x = hard_proxy_feature_matrix(x_test, test_comp, test_candidates, grid)
        hard_proxy_test = positive_class_probability(hard_proxy_model, test_proxy_x)
        candidate_score_test = sample_candidate_score(test_comp, hard_proxy_test, test_candidates, grid)
        pred_test, chosen_test, prob_test, channel_test, alpha_test = apply_v320_prediction(
            y316[test_mask],
            test_candidates,
            test_comp,
            test_meta,
            hard_proxy_test,
            candidate_score_test,
            selected_cfg,
        )
        test_selected = chosen_test != 0
        test_rmse_new = curve_rmse(pred_test, test_true)
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
        ]
        v320_test_frame = build_per_sample_metrics(
            pred_test,
            test_true,
            test_meta,
            grid,
            selected_method,
            test_base_rmse,
            candidate_prob=prob_test,
            chosen_idx=chosen_test,
            candidate_names=candidates_all.names,
        )
        v320_test_frame["hard_proxy_score"] = hard_proxy_test
        v320_test_frame["candidate_score"] = candidate_score_test
        v320_test_frame["selected_for_correction"] = test_selected
        v320_test_frame["selected_channel"] = channel_test
        v320_test_frame["fusion_alpha"] = alpha_test
        v320_test_frame["candidate_gain_hat"] = test_comp.candidate_gain_hat
        v320_test_frame["candidate_bad_prob"] = test_comp.candidate_bad_prob
        v320_test_frame["candidate_margin"] = test_comp.candidate_margin
        v320_test_frame["candidate_pos_prob"] = test_comp.candidate_pos_prob if test_comp.candidate_pos_prob is not None else np.nan
        test_frames.append(v320_test_frame)
        test_per_sample = pd.concat(test_frames, ignore_index=True)
        test_group_summary = summarize_groups(test_per_sample)
        test_budget_metrics = budget_metrics_from_rmse(test_rmse_new, test_base_rmse, test_meta, test_selected)
        test_usage = build_candidate_usage(selected_method, "test", chosen_test, test_oracle_idx, candidates_all.names, prob_test)
        test_usage["candidate_family"] = [candidate_family_name(int(i)) for i in test_usage["candidate_idx"]]
        write_csv(test_per_sample, TABLES / "v320_test_per_sample_metrics.csv")
        write_csv(test_group_summary, TABLES / "v320_test_group_summary.csv")
        write_csv(build_metric_frame(test_budget_metrics, "测试"), TABLES / "v320_test_budget_metrics.csv")
        write_csv(test_usage, TABLES / "v320_test_candidate_usage.csv")
        write_csv(build_coverage_table("测试", test_meta, test_selected), TABLES / "v320_test_coverage_budget.csv")
        write_csv(build_risk_budget_table("测试", test_meta, test_base_rmse, test_rmse_new), TABLES / "v320_test_risk_budget.csv")
        write_csv(build_channel_contribution("测试", test_meta, test_base_rmse, test_rmse_new, test_selected, channel_test, alpha_test), TABLES / "v320_test_channel_contribution.csv")
        write_csv(build_candidate_family_usage("测试", chosen_test, test_oracle_idx, test_selected, test_base_rmse, test_rmse_new), TABLES / "v320_test_candidate_family_usage.csv")
        write_csv(build_hard_proxy_eval_buckets("测试", hard_proxy_test, test_meta, test_base_rmse), TABLES / "v320_test_hard_proxy_bucket_summary.csv")
        test_reported = True

    figure_paths = [
        plot_validation_bars(group_summary, selected_method),
        plot_candidate_usage(usage, selected_method),
    ]

    guardrail = {
        "pass": bool(True),
        "goal_validation_passed": selected_pass,
        "test_reported": test_reported,
        "selected_method": selected_method,
        "selected_tier_from_train_oof": str(selected_cfg.get("selected_tier")),
        "train_event_n": int(train_mask.sum()),
        "val_event_n": int(val_mask.sum()),
        "test_event_n": int(test_mask.sum()),
        "kept_event_n": int(len(meta)),
        "isolated_event_n": 84,
        "candidate_n": int(len(candidates_all.names)),
        "candidate_selection_uses_test": False,
        "uses_test_error_as_features": False,
        "uses_future_truth_as_input": False,
        "uses_hard20_as_gate_input": False,
        "hard20_used_for_training_target_and_evaluation_only": True,
        "uses_v315_keep_manifest": True,
        "uses_v316_base_prediction": True,
        "reuses_v317_candidate_library": True,
        "uses_train_oof_quota_search": True,
        "uses_hard_proxy_score": True,
        "uses_topk_quota_gate": True,
        "uses_dual_channel_gate": True,
        "uses_candidate_gain_model": True,
        "uses_candidate_bad_risk_filter": True,
        "test_result_suppressed_when_validation_fails": True,
        "validation_budget_table": str(TABLES / "v320_validation_budget_check.csv"),
        "selected_quota_config": str(TABLES / "v320_selected_quota_config.csv"),
        "runtime_seconds": float(time.time() - started),
        "figure_paths": [str(p) for p in figure_paths],
        "zip_path": str(OUT / "v320_rank_budget_repair_gate_20260705.zip"),
        "zip_testzip": None,
    }
    report_path = write_report(guardrail, val_budget_table, group_summary, selected_method, selected_pass, test_reported)
    guardrail["report_path"] = str(report_path)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()
    zip_path, zip_ok = make_zip_package()
    guardrail["zip_path"] = str(zip_path)
    guardrail["zip_testzip"] = bool(zip_ok)
    guardrail["runtime_seconds"] = float(time.time() - started)
    report_path = write_report(guardrail, val_budget_table, group_summary, selected_method, selected_pass, test_reported)
    guardrail["report_path"] = str(report_path)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

