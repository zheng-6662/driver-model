#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v248: best-anchor 后残余轨迹形状误差审查。

本脚本承接 v247 的结论：50ms best anchor label 有明显上限收益，
但从图上看，换锚点以后模型仍然经常没有预测出真实轨迹形状。

因此 v248 不训练新模型，也不继续调锚点选择器；它只读取 v247 的
fine-grid v241 预测，回答一个更基础的问题：

    换到 best anchor 后，剩余误差主要是幅值低估、相位滞后、
    平滑均值化/转折缺失，还是方向/反打错误？

输出用于决定下一步是否应该转向 trajectory shape modeling。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 20260630
ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
SCRIPTS = BASELINES / "scripts"
V247_SCRIPT = SCRIPTS / "stage03_v247_multi_resolution_best_anchor_discovery_20260630.py"
V247_OUT = BASELINES / "v247_multi_resolution_best_anchor_discovery_20260630"
OUT = BASELINES / "v248_best_anchor_residual_shape_audit_20260630"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v248_best_anchor_residual_shape_audit_20260630_pack.zip"

PRIMARY_SCORE_NAME = "delay_l05_unstable_m05"
TAIL_START_S = 1.0
TAIL_END_S = 2.0
STILL_BAD_RMSE = 0.65
LARGE_GAIN_RMSE = 0.40


def import_module_from_path(module_name: str, path: Path):
    """按路径导入旧脚本，注册到 sys.modules 以兼容 dataclass。"""

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


V247 = import_module_from_path("stage03_v247_multi_resolution_best_anchor_discovery_20260630", V247_SCRIPT)
V236 = V247.V236
FUTURE_GRID = np.asarray(V247.FUTURE_GRID, dtype=np.float32)


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def ensure_clean_output() -> None:
    """只清理 v248 自己的输出。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 统一用 utf-8-sig，方便 Windows/Excel 打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, Any], path: Path) -> None:
    """写 JSON 日志。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """计算输入文件 hash。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """忽略非有限值计算 RMSE。"""

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if not bool(mask.any()):
        return math.nan
    return float(np.sqrt(np.mean(np.square(aa[mask] - bb[mask]))))


def finite_mae(a: np.ndarray, b: np.ndarray) -> float:
    """忽略非有限值计算 MAE。"""

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if not bool(mask.any()):
        return math.nan
    return float(np.mean(np.abs(aa[mask] - bb[mask])))


def finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    """中心化后计算相关系数。"""

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 3:
        return math.nan
    aa = aa[mask] - float(np.mean(aa[mask]))
    bb = bb[mask] - float(np.mean(bb[mask]))
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    if denom < 1e-9:
        return math.nan
    return float(np.sum(aa * bb) / denom)


def safe_div(num: float, den: float) -> float:
    """安全除法。"""

    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-9:
        return math.nan
    return float(num / den)


def normalize_bool_series(s: pd.Series) -> pd.Series:
    """读取 CSV 后把 bool/字符串/0-1 统一成 bool。"""

    if s.dtype == bool:
        return s.fillna(False).astype(bool)
    return s.fillna(False).astype(str).str.lower().isin(["true", "1", "yes", "y"])


def load_v247_artifacts() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """读取 v247 candidate/best/prediction 产物。"""

    required = [
        V247_OUT / "tables" / "v247_fine_anchor_candidate_table.csv",
        V247_OUT / "tables" / "v247_best_anchor_by_event.csv",
        V247_OUT / "v247_fine_grid_v241_predictions.npz",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("v248 缺少 v247 输入：\n" + "\n".join(missing))

    candidate = pd.read_csv(required[0], encoding="utf-8-sig")
    best = pd.read_csv(required[1], encoding="utf-8-sig")
    primary_best = best[best["score_name"].astype(str).eq(PRIMARY_SCORE_NAME)].copy()
    if primary_best.empty:
        raise AssertionError(f"v247 best table 缺少 primary score: {PRIMARY_SCORE_NAME}")

    with np.load(required[2], allow_pickle=True) as z:
        arrays = {
            "y_true": z["y_true_steering_delta"].astype(np.float32),
            "pred": z["pred_v241_steering_delta"].astype(np.float32),
            "event_uid": z["event_uid"].astype(str),
            "delay_ms": z["delay_ms"].astype(int),
            "split": z["split"].astype(str),
            "future_grid_s": z["future_grid_s"].astype(np.float32),
        }
    return candidate, primary_best, arrays


def row_map_from_arrays(arrays: Dict[str, np.ndarray]) -> Dict[Tuple[str, int], int]:
    """建立 (event_uid, delay_ms) -> npz row index 映射。"""

    return {
        (str(uid), int(delay)): int(i)
        for i, (uid, delay) in enumerate(zip(arrays["event_uid"], arrays["delay_ms"]))
    }


def tail_indices_for_delay(delay_ms: int) -> np.ndarray:
    """返回该 candidate anchor 对应原始事件 1-2s tail 的 future point mask。"""

    original_rel = int(delay_ms) / 1000.0 + FUTURE_GRID.astype(float)
    return (original_rel >= TAIL_START_S - 1e-9) & (original_rel <= TAIL_END_S + 1e-9)


def linear_calibration_rmse(true_y: np.ndarray, pred_y: np.ndarray) -> Tuple[float, float, float]:
    """
    允许 pred 做 y ~= a * pred + b 的线性校准后再算 RMSE。

    如果这个 RMSE 大幅下降，说明主要是幅值/偏置问题，不一定是形状完全错。
    """

    y = np.asarray(true_y, dtype=np.float64)
    p = np.asarray(pred_y, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    if int(mask.sum()) < 3:
        return math.nan, math.nan, math.nan
    yy = y[mask]
    pp = p[mask]
    if float(np.std(pp)) < 1e-9:
        a = 0.0
        b = float(np.mean(yy))
        fit = np.full_like(yy, b)
    else:
        x = np.column_stack([pp, np.ones_like(pp)])
        a, b = np.linalg.lstsq(x, yy, rcond=None)[0]
        fit = a * pp + b
    return finite_rmse(yy, fit), float(a), float(b)


def time_shift_best_rmse(true_y: np.ndarray, pred_y: np.ndarray, max_shift_points: int = 3) -> Tuple[float, int]:
    """
    在 +/- max_shift_points 的离散时间平移内找最低 RMSE。

    如果平移后明显下降，说明更像时间相位错；否则更像幅值/形状错。
    """

    y = np.asarray(true_y, dtype=np.float64)
    p = np.asarray(pred_y, dtype=np.float64)
    best_rmse = math.inf
    best_shift = 0
    n = min(len(y), len(p))
    y = y[:n]
    p = p[:n]
    for shift in range(-max_shift_points, max_shift_points + 1):
        if shift == 0:
            yy, pp = y, p
        elif shift > 0:
            yy, pp = y[:-shift], p[shift:]
        else:
            yy, pp = y[-shift:], p[:shift]
        if len(yy) < 4:
            continue
        rmse = finite_rmse(yy, pp)
        if np.isfinite(rmse) and rmse < best_rmse:
            best_rmse = rmse
            best_shift = shift
    if not np.isfinite(best_rmse):
        return math.nan, 0
    return float(best_rmse), int(best_shift)


def turning_count(curve: np.ndarray, eps: float = 0.03) -> int:
    """粗略统计曲线一阶差分符号变化次数，忽略非常小的抖动。"""

    y = np.asarray(curve, dtype=np.float64)
    dy = np.diff(y)
    signs = np.sign(dy[np.isfinite(dy) & (np.abs(dy) >= eps)])
    if len(signs) <= 1:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def shape_metrics(time_s: np.ndarray, true_y: np.ndarray, pred_y: np.ndarray) -> Dict[str, float]:
    """计算单条曲线的残余形状指标。"""

    x = np.asarray(time_s, dtype=np.float64)
    y = np.asarray(true_y, dtype=np.float64)
    p = np.asarray(pred_y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(p)
    x, y, p = x[mask], y[mask], p[mask]
    if len(y) < 4:
        return {"point_n": int(len(y))}

    raw_rmse = finite_rmse(y, p)
    centered_rmse = finite_rmse(y - float(np.mean(y)), p - float(np.mean(p)))
    linear_rmse, linear_a, linear_b = linear_calibration_rmse(y, p)
    shift_rmse, shift_points = time_shift_best_rmse(y, p, max_shift_points=3)

    y_from_start = y - y[0]
    p_from_start = p - p[0]
    true_range = float(np.max(y) - np.min(y))
    pred_range = float(np.max(p) - np.min(p))
    true_excursion = float(np.max(np.abs(y_from_start)))
    pred_excursion = float(np.max(np.abs(p_from_start)))
    true_final_delta = float(y[-1] - y[0])
    pred_final_delta = float(p[-1] - p[0])

    dt = np.diff(x)
    dy = np.diff(y) / np.where(np.abs(dt) < 1e-9, np.nan, dt)
    dp = np.diff(p) / np.where(np.abs(dt) < 1e-9, np.nan, dt)
    true_max_abs_slope = float(np.nanmax(np.abs(dy))) if len(dy) else math.nan
    pred_max_abs_slope = float(np.nanmax(np.abs(dp))) if len(dp) else math.nan

    true_peak_idx = int(np.nanargmax(np.abs(y_from_start)))
    pred_peak_idx = int(np.nanargmax(np.abs(p_from_start)))
    final_sign_match = bool(np.sign(true_final_delta) == np.sign(pred_final_delta)) if abs(true_final_delta) >= 0.05 else True

    out = {
        "point_n": int(len(y)),
        "raw_rmse": raw_rmse,
        "raw_mae": finite_mae(y, p),
        "bias_mean_pred_minus_true": float(np.mean(p - y)),
        "centered_rmse": centered_rmse,
        "centered_gain_frac": safe_div(raw_rmse - centered_rmse, raw_rmse),
        "linear_calibrated_rmse": linear_rmse,
        "linear_gain_frac": safe_div(raw_rmse - linear_rmse, raw_rmse),
        "linear_scale_a": linear_a,
        "linear_bias_b": linear_b,
        "time_shifted_rmse": shift_rmse,
        "time_shift_gain_frac": safe_div(raw_rmse - shift_rmse, raw_rmse),
        "best_shift_points": float(shift_points),
        "best_shift_ms": float(shift_points * 100),
        "corr_centered": finite_corr(y, p),
        "true_tail_range": true_range,
        "pred_tail_range": pred_range,
        "tail_range_ratio_pred_true": safe_div(pred_range, true_range),
        "true_excursion_from_tail_start": true_excursion,
        "pred_excursion_from_tail_start": pred_excursion,
        "excursion_ratio_pred_true": safe_div(pred_excursion, true_excursion),
        "true_final_delta_from_tail_start": true_final_delta,
        "pred_final_delta_from_tail_start": pred_final_delta,
        "final_direction_match": final_sign_match,
        "true_max_abs_slope": true_max_abs_slope,
        "pred_max_abs_slope": pred_max_abs_slope,
        "slope_ratio_pred_true": safe_div(pred_max_abs_slope, true_max_abs_slope),
        "slope_rmse": finite_rmse(dy, dp),
        "true_turning_count": float(turning_count(y)),
        "pred_turning_count": float(turning_count(p)),
        "turning_count_gap_pred_minus_true": float(turning_count(p) - turning_count(y)),
        "true_peak_time_s": float(x[true_peak_idx]),
        "pred_peak_time_s": float(x[pred_peak_idx]),
        "peak_time_error_ms": float((x[pred_peak_idx] - x[true_peak_idx]) * 1000.0),
    }
    return out


def classify_shape_error(metrics: Dict[str, float]) -> str:
    """基于残余指标给 best-anchor 后错误做粗分类。"""

    raw = float(metrics.get("raw_rmse", math.nan))
    if not np.isfinite(raw):
        return "invalid"
    if raw < 0.35:
        return "mostly_fixed_low_residual"

    corr = float(metrics.get("corr_centered", math.nan))
    final_match = bool(metrics.get("final_direction_match", True))
    if (np.isfinite(corr) and corr < -0.15) or not final_match:
        return "direction_or_reversal_error"

    range_ratio = float(metrics.get("tail_range_ratio_pred_true", math.nan))
    excursion_ratio = float(metrics.get("excursion_ratio_pred_true", math.nan))
    slope_ratio = float(metrics.get("slope_ratio_pred_true", math.nan))
    turning_gap = float(metrics.get("turning_count_gap_pred_minus_true", 0.0))
    linear_gain = float(metrics.get("linear_gain_frac", 0.0))
    time_gain = float(metrics.get("time_shift_gain_frac", 0.0))

    if np.isfinite(range_ratio) and np.isfinite(excursion_ratio) and range_ratio < 0.60 and excursion_ratio < 0.60:
        return "amplitude_underestimation_smoothing"
    if np.isfinite(slope_ratio) and slope_ratio < 0.60 and turning_gap < 0:
        return "shape_smoothing_turning_missing"
    if np.isfinite(range_ratio) and range_ratio < 0.75:
        return "amplitude_underestimation"
    if np.isfinite(time_gain) and time_gain > 0.25:
        return "phase_time_shift"
    if np.isfinite(linear_gain) and linear_gain > 0.25:
        return "calibration_amplitude_bias"
    if np.isfinite(slope_ratio) and slope_ratio < 0.75:
        return "slope_underestimation"
    return "residual_shape_error"


def current_steer_sampler() -> Any:
    """返回带缓存的 observation_s steering 采样函数，用于画绝对 steering 图。"""

    raw_cache: Dict[Path, Any] = {}

    def _sample(row: pd.Series) -> float:
        raw_path = Path(str(row["raw_vehicle_csv"]))
        if raw_path not in raw_cache:
            raw_cache[raw_path] = V236.read_raw_vehicle(raw_path)
        raw = raw_cache[raw_path]
        value, _, _ = V236.sample_feature(
            raw,
            ["zx|SteeringWheel"],
            np.array([float(row["observation_s"])], dtype=float),
        )
        return float(value[0])

    return _sample


def build_residual_decomposition(
    candidate: pd.DataFrame,
    primary_best: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """构建事件级 current-0ms vs best-anchor 残余形状分解表。"""

    row_map = row_map_from_arrays(arrays)
    current0 = candidate[candidate["candidate_delay_ms"].astype(int).eq(0)].copy()
    current0 = current0.set_index("event_uid", drop=False)
    candidate_by_pair = {
        (str(r.event_uid), int(r.candidate_delay_ms)): r
        for r in candidate.itertuples(index=False)
    }

    rows: List[Dict[str, Any]] = []
    for _, best in primary_best.iterrows():
        event_uid = str(best["event_uid"])
        if event_uid not in current0.index:
            continue
        best_delay = int(best["best_delay_ms"])
        if (event_uid, best_delay) not in row_map or (event_uid, 0) not in row_map:
            continue

        i0 = row_map[(event_uid, 0)]
        ib = row_map[(event_uid, best_delay)]
        row0 = current0.loc[event_uid]
        rowb = candidate_by_pair.get((event_uid, best_delay))
        if rowb is None:
            continue

        current_mask = tail_indices_for_delay(0)
        best_mask = tail_indices_for_delay(best_delay)
        x0 = 0.0 + arrays["future_grid_s"][current_mask].astype(float)
        xb = best_delay / 1000.0 + arrays["future_grid_s"][best_mask].astype(float)
        y0 = arrays["y_true"][i0, current_mask].astype(float)
        p0 = arrays["pred"][i0, current_mask].astype(float)
        yb = arrays["y_true"][ib, best_mask].astype(float)
        pb = arrays["pred"][ib, best_mask].astype(float)

        current_metrics = shape_metrics(x0, y0, p0)
        best_metrics = shape_metrics(xb, yb, pb)
        category = classify_shape_error(best_metrics)

        current_rmse = float(current_metrics.get("raw_rmse", math.nan))
        best_rmse = float(best_metrics.get("raw_rmse", math.nan))
        anchor_gain = current_rmse - best_rmse if np.isfinite(current_rmse) and np.isfinite(best_rmse) else math.nan
        row = {
            "event_uid": event_uid,
            "split": str(best["split"]),
            "scene_type": str(best.get("scene_type", "")),
            "pool_key": str(best.get("pool_key", "")),
            "best_delay_ms": best_delay,
            "current_0ms_tail_rmse": current_rmse,
            "best_anchor_tail_rmse": best_rmse,
            "anchor_gain_rmse": anchor_gain,
            "anchor_gain_frac_of_current": safe_div(anchor_gain, current_rmse),
            "still_bad_after_best_anchor": bool(np.isfinite(best_rmse) and best_rmse >= STILL_BAD_RMSE),
            "large_anchor_gain": bool(np.isfinite(anchor_gain) and anchor_gain >= LARGE_GAIN_RMSE),
            "shape_error_category": category,
            "bad_top10_split_v241": bool(best.get("bad_top10_split_v241", False)),
            "very_bad_top5_split_v241": bool(best.get("very_bad_top5_split_v241", False)),
            "normal_curve": bool(best.get("normal_curve", False)),
            "observe_later_like": bool(best.get("observe_later_like", False)),
            "strong_steer": bool(best.get("strong_steer", False)),
            "reverse": bool(best.get("reverse", False)),
            "raw_vehicle_csv": str(getattr(rowb, "raw_vehicle_csv", "")),
        }
        for prefix, metrics in [("current0", current_metrics), ("best", best_metrics)]:
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["split", "best_anchor_tail_rmse"], ascending=[True, False]).reset_index(drop=True)
    write_csv(out, TABLES / "v248_best_anchor_residual_decomposition.csv")
    return out


def group_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """事件级分组。"""

    true = pd.Series(True, index=df.index)
    bad = normalize_bool_series(df["bad_top10_split_v241"]) if "bad_top10_split_v241" in df.columns else ~true
    very_bad = normalize_bool_series(df["very_bad_top5_split_v241"]) if "very_bad_top5_split_v241" in df.columns else ~true
    normal = normalize_bool_series(df["normal_curve"]) if "normal_curve" in df.columns else ~true
    observe = normalize_bool_series(df["observe_later_like"]) if "observe_later_like" in df.columns else ~true
    strong = normalize_bool_series(df["strong_steer"]) if "strong_steer" in df.columns else ~true
    reverse = normalize_bool_series(df["reverse"]) if "reverse" in df.columns else ~true
    still_bad = normalize_bool_series(df["still_bad_after_best_anchor"]) if "still_bad_after_best_anchor" in df.columns else ~true
    large_gain = normalize_bool_series(df["large_anchor_gain"]) if "large_anchor_gain" in df.columns else ~true
    return {
        "all": true,
        "normal": normal & ~bad,
        "bad_top10": bad,
        "very_bad_top5": very_bad,
        "observe_later_like": observe,
        "strong_steer": strong,
        "reverse": reverse,
        "still_bad_after_best_anchor": still_bad,
        "large_anchor_gain": large_gain,
    }


def build_summary_tables(decomp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成 anchor-vs-shape 汇总、类别汇总和峰值低估表。"""

    rows: List[Dict[str, Any]] = []
    for split, split_df in decomp.groupby("split"):
        for group_name, mask in group_masks(split_df).items():
            sub = split_df[mask]
            if sub.empty:
                continue
            rows.append(
                {
                    "split": str(split),
                    "event_group": group_name,
                    "n": int(len(sub)),
                    "mean_current_0ms_rmse": float(sub["current_0ms_tail_rmse"].mean()),
                    "mean_best_anchor_rmse": float(sub["best_anchor_tail_rmse"].mean()),
                    "mean_anchor_gain_rmse": float(sub["anchor_gain_rmse"].mean()),
                    "mean_anchor_gain_frac": float(sub["anchor_gain_frac_of_current"].mean()),
                    "pct_still_bad_after_best_anchor": float(sub["still_bad_after_best_anchor"].astype(bool).mean()),
                    "mean_best_delay_ms": float(sub["best_delay_ms"].astype(float).mean()),
                    "mean_best_range_ratio": float(sub["best_tail_range_ratio_pred_true"].mean()),
                    "mean_best_excursion_ratio": float(sub["best_excursion_ratio_pred_true"].mean()),
                    "mean_best_slope_ratio": float(sub["best_slope_ratio_pred_true"].mean()),
                    "mean_best_corr_centered": float(sub["best_corr_centered"].mean()),
                    "mean_linear_gain_frac": float(sub["best_linear_gain_frac"].mean()),
                    "mean_time_shift_gain_frac": float(sub["best_time_shift_gain_frac"].mean()),
                }
            )
    summary = pd.DataFrame(rows).sort_values(["split", "event_group"]).reset_index(drop=True)
    write_csv(summary, TABLES / "v248_anchor_vs_shape_summary.csv")

    category_rows: List[Dict[str, Any]] = []
    for (split, category), g in decomp.groupby(["split", "shape_error_category"]):
        category_rows.append(
            {
                "split": str(split),
                "shape_error_category": str(category),
                "n": int(len(g)),
                "mean_best_anchor_rmse": float(g["best_anchor_tail_rmse"].mean()),
                "mean_current_0ms_rmse": float(g["current_0ms_tail_rmse"].mean()),
                "mean_anchor_gain_rmse": float(g["anchor_gain_rmse"].mean()),
                "mean_best_delay_ms": float(g["best_delay_ms"].astype(float).mean()),
                "mean_range_ratio": float(g["best_tail_range_ratio_pred_true"].mean()),
                "mean_excursion_ratio": float(g["best_excursion_ratio_pred_true"].mean()),
                "mean_slope_ratio": float(g["best_slope_ratio_pred_true"].mean()),
            }
        )
    category = pd.DataFrame(category_rows).sort_values(["split", "n"], ascending=[True, False]).reset_index(drop=True)
    write_csv(category, TABLES / "v248_shape_error_categories.csv")

    peak_cols = [
        "event_uid",
        "split",
        "best_delay_ms",
        "current_0ms_tail_rmse",
        "best_anchor_tail_rmse",
        "anchor_gain_rmse",
        "shape_error_category",
        "bad_top10_split_v241",
        "very_bad_top5_split_v241",
        "best_true_tail_range",
        "best_pred_tail_range",
        "best_tail_range_ratio_pred_true",
        "best_true_excursion_from_tail_start",
        "best_pred_excursion_from_tail_start",
        "best_excursion_ratio_pred_true",
        "best_true_max_abs_slope",
        "best_pred_max_abs_slope",
        "best_slope_ratio_pred_true",
        "best_true_turning_count",
        "best_pred_turning_count",
        "best_turning_count_gap_pred_minus_true",
        "best_peak_time_error_ms",
    ]
    peak = decomp[[c for c in peak_cols if c in decomp.columns]].copy()
    peak = peak.sort_values(["split", "best_excursion_ratio_pred_true", "best_anchor_tail_rmse"], ascending=[True, True, False])
    write_csv(peak, TABLES / "v248_peak_underestimation_table.csv")
    return summary, category, peak


def configure_plotting() -> None:
    """配置中文字体。"""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_error_decomposition_scatter(decomp: pd.DataFrame) -> Path:
    """画 current RMSE vs best-anchor RMSE，观察锚点解释比例和残余大小。"""

    test = decomp[decomp["split"].eq("test")].copy()
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    color = np.where(normalize_bool_series(test["bad_top10_split_v241"]), "#D55E00", "#0072B2")
    ax.scatter(test["current_0ms_tail_rmse"], test["best_anchor_tail_rmse"], c=color, s=34, alpha=0.78)
    lim = float(np.nanmax([test["current_0ms_tail_rmse"].max(), test["best_anchor_tail_rmse"].max(), 1.0]))
    ax.plot([0, lim], [0, lim], color="0.35", linestyle="--", linewidth=1)
    ax.axhline(STILL_BAD_RMSE, color="#CC79A7", linestyle=":", linewidth=1.5)
    ax.set_xlabel("0ms 原锚点 tail RMSE")
    ax.set_ylabel("v247 best anchor 后 tail RMSE")
    ax.set_title("v248：锚点收益与 best-anchor 后残余误差（test）")
    ax.grid(alpha=0.25)
    out = FIGURES / "v248_error_decomposition_scatter.png"
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_shape_category_summary(category: pd.DataFrame) -> Path:
    """画 test split 的残余错误类型分布。"""

    test = category[category["split"].eq("test")].copy()
    test = test.sort_values("n", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.barh(test["shape_error_category"], test["n"], color="#0072B2")
    ax.set_xlabel("event count")
    ax.set_title("v248：best-anchor 后残余错误类型分布（test）")
    ax.grid(axis="x", alpha=0.25)
    out = FIGURES / "v248_shape_category_summary.png"
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_casebook(
    rows: pd.DataFrame,
    candidate: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
    out_name: str,
    title: str,
    max_cases: int = 8,
) -> Path:
    """画 current 0ms vs best anchor 的 casebook。"""

    row_map = row_map_from_arrays(arrays)
    sampler = current_steer_sampler()
    chosen = rows.head(max_cases).copy()
    if chosen.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "no cases", ha="center", va="center")
        out = FIGURES / out_name
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    fig, axes = plt.subplots(len(chosen), 1, figsize=(15, max(3.0 * len(chosen), 4)), sharex=False)
    if len(chosen) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, chosen.iterrows()):
        event_uid = str(row["event_uid"])
        best_delay = int(row["best_delay_ms"])
        i0 = row_map[(event_uid, 0)]
        ib = row_map[(event_uid, best_delay)]
        row0 = candidate[(candidate["event_uid"].astype(str).eq(event_uid)) & candidate["candidate_delay_ms"].astype(int).eq(0)].iloc[0]
        rowb = candidate[(candidate["event_uid"].astype(str).eq(event_uid)) & candidate["candidate_delay_ms"].astype(int).eq(best_delay)].iloc[0]
        s0 = sampler(row0)
        sb = sampler(rowb)
        x0 = arrays["future_grid_s"].astype(float)
        xb = best_delay / 1000.0 + arrays["future_grid_s"].astype(float)
        true0 = s0 + arrays["y_true"][i0].astype(float)
        pred0 = s0 + arrays["pred"][i0].astype(float)
        trueb = sb + arrays["y_true"][ib].astype(float)
        predb = sb + arrays["pred"][ib].astype(float)

        ax.axvspan(1.0, 2.0, color="0.92", zorder=0, label="原始 1-2s tail" if ax is axes[0] else None)
        ax.plot(x0, true0, color="black", linewidth=2.0, label="真实 steering" if ax is axes[0] else None)
        ax.plot(x0, pred0, color="#009E73", linestyle="--", linewidth=1.8, label="0ms 原锚点预测" if ax is axes[0] else None)
        ax.plot(xb, predb, color="#D55E00", linestyle="-.", linewidth=1.8, label="best anchor 预测" if ax is axes[0] else None)
        ax.plot(xb, trueb, color="0.38", linewidth=1.0, alpha=0.68, label="best anchor 对应真实段" if ax is axes[0] else None)
        ax.axvline(best_delay / 1000.0, color="#D55E00", linestyle=":", linewidth=1.2)
        ax.set_xlim(0, max(2.05, best_delay / 1000.0 + 2.02))
        ax.grid(alpha=0.25)
        ax.set_ylabel("absolute steering")
        ax.set_title(
            f"{event_uid} | best={best_delay}ms | "
            f"0ms={row['current_0ms_tail_rmse']:.3f} -> best={row['best_anchor_tail_rmse']:.3f} | "
            f"cat={row['shape_error_category']} | range_ratio={row['best_tail_range_ratio_pred_true']:.2f}",
            fontsize=9,
            loc="left",
        )
    axes[-1].set_xlabel("原始事件锚点后的时间 / s")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.36), ncol=4, fontsize=9)
    fig.suptitle(title, y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = FIGURES / out_name
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def create_figures(decomp: pd.DataFrame, category: pd.DataFrame, candidate: pd.DataFrame, arrays: Dict[str, np.ndarray]) -> List[Path]:
    """生成 v248 核心图。"""

    configure_plotting()
    paths = [
        plot_error_decomposition_scatter(decomp),
        plot_shape_category_summary(category),
    ]
    test_bad = decomp[decomp["split"].eq("test") & normalize_bool_series(decomp["bad_top10_split_v241"])].copy()
    still_bad = test_bad.sort_values("best_anchor_tail_rmse", ascending=False)
    improved_wrong = test_bad[
        test_bad["anchor_gain_rmse"].ge(LARGE_GAIN_RMSE) & test_bad["best_anchor_tail_rmse"].ge(0.40)
    ].sort_values(["anchor_gain_rmse", "best_anchor_tail_rmse"], ascending=[False, False])
    amplitude_under = test_bad.sort_values(["best_excursion_ratio_pred_true", "best_anchor_tail_rmse"], ascending=[True, False])
    paths.append(
        plot_casebook(
            still_bad,
            candidate,
            arrays,
            "v248_best_anchor_still_bad_casebook.png",
            "v248 test bad_top10：best anchor 后仍然最差的样本",
        )
    )
    paths.append(
        plot_casebook(
            improved_wrong,
            candidate,
            arrays,
            "v248_improved_but_still_wrong_casebook.png",
            "v248 test bad_top10：换锚点明显改善但形状仍不对",
        )
    )
    paths.append(
        plot_casebook(
            amplitude_under,
            candidate,
            arrays,
            "v248_peak_underestimation_casebook.png",
            "v248 test bad_top10：峰值/幅值低估最明显的样本",
        )
    )
    return paths


def metric_row(df: pd.DataFrame, split: str, group: str) -> pd.Series | None:
    """从 summary 表中取一行。"""

    sub = df[df["split"].eq(split) & df["event_group"].eq(group)]
    if sub.empty:
        return None
    return sub.iloc[0]


def fmt(value: Any, digits: int = 3) -> str:
    """安全格式化。"""

    try:
        v = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def write_report(summary: pd.DataFrame, category: pd.DataFrame, decomp: pd.DataFrame, figures: List[Path]) -> Path:
    """写中文报告。"""

    test_all = metric_row(summary, "test", "all")
    test_bad = metric_row(summary, "test", "bad_top10")
    test_very = metric_row(summary, "test", "very_bad_top5")
    test_still = metric_row(summary, "test", "still_bad_after_best_anchor")
    cat_test = category[category["split"].eq("test")].copy()
    top_cat = cat_test.sort_values("n", ascending=False).head(5)
    cat_lines = [
        f"- `{r.shape_error_category}`: n={int(r.n)}, mean best RMSE={fmt(r.mean_best_anchor_rmse)}, range_ratio={fmt(r.mean_range_ratio)}, slope_ratio={fmt(r.mean_slope_ratio)}"
        for r in top_cat.itertuples(index=False)
    ]

    lines = [
        "# v248 best-anchor 后残余轨迹形状误差审查",
        "",
        "## 结论摘要",
        "",
        "- v248 不训练新模型，也不继续调 selector；它只读取 v247 fine-grid + locked v241 预测，量化 best anchor 后剩余错误类型。",
        (
            f"- test/all：0ms 平均 RMSE `{fmt(test_all['mean_current_0ms_rmse'] if test_all is not None else math.nan)}`，"
            f"best-anchor 后 `{fmt(test_all['mean_best_anchor_rmse'] if test_all is not None else math.nan)}`，"
            f"锚点平均解释 `{fmt(test_all['mean_anchor_gain_rmse'] if test_all is not None else math.nan)}`。"
        ),
        (
            f"- test/bad_top10：0ms 平均 RMSE `{fmt(test_bad['mean_current_0ms_rmse'] if test_bad is not None else math.nan)}`，"
            f"best-anchor 后 `{fmt(test_bad['mean_best_anchor_rmse'] if test_bad is not None else math.nan)}`，"
            f"仍高于 `{STILL_BAD_RMSE}` 的比例 `{fmt(test_bad['pct_still_bad_after_best_anchor'] if test_bad is not None else math.nan)}`。"
        ),
        (
            f"- test/very_bad_top5：0ms 平均 RMSE `{fmt(test_very['mean_current_0ms_rmse'] if test_very is not None else math.nan)}`，"
            f"best-anchor 后 `{fmt(test_very['mean_best_anchor_rmse'] if test_very is not None else math.nan)}`。"
        ),
        (
            f"- best-anchor 后仍然很差的 test 样本：n=`{int(test_still['n']) if test_still is not None else 0}`，"
            f"平均 best RMSE `{fmt(test_still['mean_best_anchor_rmse'] if test_still is not None else math.nan)}`。"
        ),
        "",
        "## 残余错误类型",
        "",
        *cat_lines,
        "",
        "## 方法解释",
        "",
        "v247 证明换锚点有上限收益，但图上已经能看到，橙色 best-anchor 预测仍然经常比真实轨迹更平滑、幅值更小，或者错过快速回正/转折。v248 把这种视觉判断量化成幅值比例、斜率比例、转折次数差、线性校准收益和时间平移收益。",
        "",
        "如果 `linear_gain_frac` 高，说明主要是幅值/偏置可校准；如果 `time_shift_gain_frac` 高，说明主要是相位错；如果二者都不高但 RMSE 仍大，通常就是轨迹形状本身没有建好。",
        "",
        "## 关键产物",
        "",
        "- `tables/v248_best_anchor_residual_decomposition.csv`：每个事件 current 0ms 与 best-anchor 后的形状误差分解。",
        "- `tables/v248_peak_underestimation_table.csv`：峰值/幅值低估和斜率低估排序表。",
        "- `tables/v248_shape_error_categories.csv`：残余错误类别汇总。",
        "- `tables/v248_anchor_vs_shape_summary.csv`：按 split/group 的锚点收益与残余形状指标。",
        "- `figures/v248_best_anchor_still_bad_casebook.png`：best anchor 后仍然最差的样本。",
        "- `figures/v248_improved_but_still_wrong_casebook.png`：换锚点改善明显但形状仍不对的样本。",
        "- `figures/v248_peak_underestimation_casebook.png`：峰值/幅值低估最明显样本。",
        "",
        "## 下一步判断",
        "",
        "如果 v248 显示主要残余是 amplitude/shape smoothing，而不是 phase/anchor 错位，那么下一步应从 sequential selector 转向 trajectory shape modeling，例如完整曲线 decoder + peak/slope loss，或基于 v241 的 shape residual corrector。",
    ]
    out = REPORTS / "v248_best_anchor_residual_shape_audit_cn.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_input_hashes() -> pd.DataFrame:
    """记录输入文件 hash。"""

    files = {
        "v248_script": Path(__file__),
        "v247_script": V247_SCRIPT,
        "v247_candidate_table": V247_OUT / "tables" / "v247_fine_anchor_candidate_table.csv",
        "v247_best_anchor_table": V247_OUT / "tables" / "v247_best_anchor_by_event.csv",
        "v247_predictions_npz": V247_OUT / "v247_fine_grid_v241_predictions.npz",
    }
    rows = []
    for name, path in files.items():
        rows.append(
            {
                "input_name": name,
                "path": str(path),
                "exists": bool(path.exists()),
                "sha256": file_sha256(path) if path.exists() else "",
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, LOGS / "input_file_hashes.csv")
    return out


def build_guardrail_json(zip_testzip: str | None) -> Dict[str, Any]:
    """写 guardrail。"""

    payload = {
        "pass": zip_testzip is None,
        "stage": "v248_best_anchor_residual_shape_audit",
        "no_model_training": True,
        "uses_locked_v247_predictions": True,
        "uses_locked_v241_predictions_only": True,
        "no_anchor_selector_training": True,
        "no_test_based_retuning": True,
        "oracle_best_anchor_upper_bound_only": True,
        "primary_score_name": PRIMARY_SCORE_NAME,
        "zip_testzip": zip_testzip,
    }
    write_json(payload, LOGS / "guardrail_check.json")
    return payload


def file_inventory() -> pd.DataFrame:
    """产物清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "bytes": int(path.stat().st_size), "sha256": file_sha256(path)})
    out = pd.DataFrame(rows)
    write_csv(out, LOGS / "file_inventory.csv")
    return out


def zip_outputs() -> Path:
    """打包 v248 输出。"""

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT.parent))
    return ZIP_PATH


def write_run_manifest(decomp: pd.DataFrame, figures: List[Path], report_path: Path, zip_path: Path) -> None:
    """运行元数据。"""

    payload = {
        "stage": "v248_best_anchor_residual_shape_audit",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(OUT),
        "source_v247_out": str(V247_OUT),
        "primary_score_name": PRIMARY_SCORE_NAME,
        "n_events": int(len(decomp)),
        "splits": sorted(decomp["split"].astype(str).unique().tolist()),
        "figures": [str(p) for p in figures],
        "report": str(report_path),
        "zip": str(zip_path),
    }
    write_json(payload, LOGS / "run_manifest.json")


def main() -> None:
    """v248 主流程。"""

    np.random.seed(SEED)
    ensure_clean_output()
    print("[v248] load v247 fine-grid predictions and best-anchor labels")
    write_input_hashes()
    candidate, primary_best, arrays = load_v247_artifacts()

    print("[v248] compute residual shape decomposition after best anchor")
    decomp = build_residual_decomposition(candidate, primary_best, arrays)
    summary, category, peak = build_summary_tables(decomp)

    print("[v248] create figures and report")
    figures = create_figures(decomp, category, candidate, arrays)
    report_path = write_report(summary, category, decomp, figures)
    file_inventory()
    zip_path = zip_outputs()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_testzip = zf.testzip()
    guardrail = build_guardrail_json(zip_testzip)
    write_run_manifest(decomp, figures, report_path, zip_path)
    file_inventory()
    zip_path = zip_outputs()
    with zipfile.ZipFile(zip_path, "r") as zf:
        final_testzip = zf.testzip()
    if final_testzip is not None:
        raise AssertionError(f"ZIP testzip failed: {final_testzip}")
    print(f"[v248] guardrail_check.pass={guardrail['pass']}")
    print(f"[v248] ZIP testzip={final_testzip}")
    print(f"[v248] report={report_path}")
    print(f"[v248] zip={zip_path}")


if __name__ == "__main__":
    main()
