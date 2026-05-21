#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build v1.3 record-level episode dataset after user review.

v1.2 used raw z range too aggressively for suspected off-road / road recovery.
The user found two systematic errors:

1. Some road-edge / off-road-like episodes have small z range but clear speed,
   brake, lateral-offset, and vehicle-dynamic changes.
2. Some curve / road-grade episodes have large raw z range, but the z trend is
   smooth and should not be treated as off-road just because the absolute z
   range is large.

This script keeps v1.2 as input and adds a v1.3 decision layer. It does not
train any model.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

import build_record_episode_dataset_v1_2 as v12


matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
V12_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_2_cleaned"
V12_ALL = V12_ROOT / "tables" / "record_level_episodes_all_v1_2.csv"
OUT_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_3_cleaned"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "review_panels_v1_3"
LOG_DIR = OUT_ROOT / "logs"
REPORT_PATH = ROOT / "09_reports" / "stage02_record_episode_reconstruction_v1_3_user_summary_cn.md"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-21.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"
ROAD_SOURCE_DIR = PROJECT_ROOT / "01_datasets" / "多模态数据" / "被试数据集合" / "道路信息" / "道路"
ROAD_CFG_DIR = ROAD_SOURCE_DIR / "Area2_extracted"
ROAD_CENTERLINE = ROAD_SOURCE_DIR / "road_centerline_generated.csv"


NORMAL_DURATION_MAX_S = 15.0
REVIEW_DURATION_MAX_S = 20.0
EXTREME_DURATION_MAX_S = 30.0

LAT_RANGE_ROADEDGE_FLOOR_M = 3.0
LAT_JUMP_ROADEDGE_FLOOR_M = 1.0
SPEED_DROP_ROADEDGE_FLOOR_KMH = 18.0
BRAKE_ACTIVE_FLOOR = 0.20
Z_RAW_GRADE_FLOOR_M = 0.50
Z_GRADE_RESIDUAL_RATIO_MAX = 0.35
Z_GRADE_MONOTONIC_FRACTION_MIN = 0.68
Z_TRANSIENT_RESIDUAL_FLOOR_M = 0.12
PITCH_TRANSIENT_FLOOR_RAD = 0.08


USER_FEEDBACK_OVERRIDES: dict[str, tuple[str, str, bool, bool, bool, bool]] = {
    "rec_v1_byx_2025_09_28_17_05_51_0002": (
        "defer_roadedge_or_offroad_user_feedback",
        "用户复核指出：该样本更像上下马路/路边恢复，不应作为目标极限工况主训练样本",
        False,
        True,
        False,
        False,
    ),
    "rec_v1_gzj_2025_09_27_12_28_14_0004": (
        "review_long_curve_or_grade_user_feedback",
        "用户复核指出：该样本是弯道/道路趋势，不应仅因高度范围大判为上下马路",
        False,
        True,
        False,
        False,
    ),
}


MAX_FIGURES_PER_BUCKET = {
    "train_target_extreme": 36,
    "train_conservative_extreme": 28,
    "review_duration_15_20s": 24,
    "review_long_curve_or_grade": 36,
    "review_long_curve_or_grade_user_feedback": 8,
    "review_curve_high_dynamics": 30,
    "defer_roadedge_or_offroad": 36,
    "defer_roadedge_or_offroad_long": 36,
    "defer_roadedge_or_offroad_user_feedback": 8,
    "defer_long_merged": 30,
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_PATH.parent, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def finite_numeric(values: Any) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def safe_peak_to_peak(values: np.ndarray) -> float:
    arr = finite_numeric(values)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))


def safe_abs_peak(values: np.ndarray) -> float:
    arr = finite_numeric(values)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(np.abs(arr)))


def safe_max(values: np.ndarray) -> float:
    arr = finite_numeric(values)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(arr))


def safe_min(values: np.ndarray) -> float:
    arr = finite_numeric(values)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmin(arr))


def detrended_residual(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    valid = np.isfinite(values) & np.isfinite(x)
    if valid.sum() < 6:
        return out
    xx = x[valid]
    yy = values[valid]
    if np.nanmax(xx) - np.nanmin(xx) < 1e-6:
        return out
    coef = np.polyfit(xx - xx[0], yy, deg=1)
    trend = np.polyval(coef, xx - xx[0])
    out[valid] = yy - trend
    return out


def monotonic_fraction(values: np.ndarray) -> float:
    arr = finite_numeric(values)
    if arr.size < 5:
        return float("nan")
    diffs = np.diff(arr)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan")
    pos = float(np.mean(diffs >= 0))
    neg = float(np.mean(diffs <= 0))
    return float(max(pos, neg))


def max_adjacent_jump(values: np.ndarray) -> float:
    arr = finite_numeric(values)
    if arr.size < 2:
        return float("nan")
    return float(np.nanmax(np.abs(np.diff(arr))))


def count_large_jumps(values: np.ndarray, threshold: float) -> int:
    arr = finite_numeric(values)
    if arr.size < 2:
        return 0
    return int(np.sum(np.abs(np.diff(arr)) >= threshold))


def read_text_maybe(path: Path) -> str:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def audit_road_height_sources() -> dict[str, Any]:
    cfg_rows = []
    height_pattern = re.compile(r"\b(z|height|elevation|altitude|z0|z1)\s*=", re.IGNORECASE)
    z_value_pattern = re.compile(r"\b(z0|z1|z)\s*=\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
    smooth_pattern = re.compile(r"\b(slope|grade|bank|superelevation)\s*=", re.IGNORECASE)
    if ROAD_CFG_DIR.exists():
        for path in sorted(ROAD_CFG_DIR.glob("*.cfg")):
            text = read_text_maybe(path)
            z_values = [float(m.group(2)) for m in z_value_pattern.finditer(text)]
            cfg_rows.append(
                {
                    "file": str(path),
                    "height_like_assignment_count": len(height_pattern.findall(text)),
                    "z_value_count": len(z_values),
                    "z_value_min": min(z_values) if z_values else np.nan,
                    "z_value_max": max(z_values) if z_values else np.nan,
                    "slope_or_bank_assignment_count": len(smooth_pattern.findall(text)),
                }
            )
    centerline_cols: list[str] = []
    if ROAD_CENTERLINE.exists():
        try:
            centerline_cols = list(pd.read_csv(ROAD_CENTERLINE, nrows=1).columns)
        except Exception:
            centerline_cols = []
    pd.DataFrame(cfg_rows).to_csv(TABLE_DIR / "road_source_height_field_audit_v1_3.csv", index=False, encoding="utf-8-sig")
    return {
        "cfg_dir": str(ROAD_CFG_DIR),
        "cfg_files": len(cfg_rows),
        "cfg_height_like_assignment_total": int(sum(r["height_like_assignment_count"] for r in cfg_rows)),
        "cfg_z_value_total": int(sum(r["z_value_count"] for r in cfg_rows)),
        "cfg_z_value_min": float(np.nanmin([r["z_value_min"] for r in cfg_rows])) if cfg_rows else np.nan,
        "cfg_z_value_max": float(np.nanmax([r["z_value_max"] for r in cfg_rows])) if cfg_rows else np.nan,
        "cfg_slope_or_bank_assignment_total": int(sum(r["slope_or_bank_assignment_count"] for r in cfg_rows)),
        "centerline_path": str(ROAD_CENTERLINE),
        "centerline_columns": centerline_cols,
    }


def compute_episode_features_v1_3(row: pd.Series, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rec = v12.load_vehicle_record(str(row.get("vehicle_file", "")), cache)
    out: dict[str, Any] = {
        "raw_feature_ok_v1_3": False,
        "z_residual_range_v1_3": np.nan,
        "z_residual_rate_peak_v1_3": np.nan,
        "z_trend_ratio_v1_3": np.nan,
        "z_monotonic_fraction_v1_3": np.nan,
        "lat_offset_adjacent_jump_peak_v1_3": np.nan,
        "lat_offset_large_jump_count_v1_3": 0,
        "speed_drop_from_start_v1_3": np.nan,
        "speed_drop_peak_to_peak_v1_3": np.nan,
        "brake_peak_v1_3": np.nan,
        "pitch_residual_range_v1_3": np.nan,
        "road_grade_like_z_v1_3": False,
        "z_transient_suspected_v1_3": False,
        "lat_offset_jump_suspected_v1_3": False,
        "speed_brake_response_suspected_v1_3": False,
        "roadedge_or_offroad_suspected_v1_3": False,
    }
    if rec is None:
        return out
    start = float(row.get("episode_start_s", np.nan))
    end = float(row.get("episode_end_s", np.nan))
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return out
    t = rec["t"]
    mask = np.isfinite(t) & (t >= start) & (t <= end)
    if mask.sum() < 6:
        return out
    x = t[mask] - start
    signals = rec["signals"]
    z = signals["z"][mask]
    pitch = signals["pitch"][mask]
    lat_offset = signals["lat_offset"][mask]
    speed = signals["speed"][mask]
    brake = signals["brake"][mask]

    z_resid = detrended_residual(z, x)
    z_resid_rate = v12.gradient(z_resid, x)
    pitch_resid = detrended_residual(pitch, x)
    z_range = float(row.get("z_range_v1_2", np.nan))
    z_resid_range = safe_peak_to_peak(z_resid)
    z_trend_ratio = z_resid_range / z_range if math.isfinite(z_range) and abs(z_range) > 1e-9 else np.nan
    z_mono = monotonic_fraction(z)
    lat_jump = max_adjacent_jump(lat_offset)
    speed_start = safe_max(speed[: max(2, min(10, len(speed)))])
    speed_min = safe_min(speed)
    speed_drop_start = speed_start - speed_min if math.isfinite(speed_start) and math.isfinite(speed_min) else np.nan
    speed_range = safe_peak_to_peak(speed)
    brake_peak = safe_max(np.abs(brake))
    pitch_resid_range = safe_peak_to_peak(pitch_resid)

    is_curve = bool(row.get("is_curve_context", False)) or "curve" in str(row.get("road_module_names", "")).lower()
    road_grade_like_z = (
        math.isfinite(z_range)
        and z_range >= Z_RAW_GRADE_FLOOR_M
        and math.isfinite(z_trend_ratio)
        and z_trend_ratio <= Z_GRADE_RESIDUAL_RATIO_MAX
        and math.isfinite(z_mono)
        and z_mono >= Z_GRADE_MONOTONIC_FRACTION_MIN
    )
    z_transient = (
        (math.isfinite(z_resid_range) and z_resid_range >= Z_TRANSIENT_RESIDUAL_FLOOR_M)
        or (math.isfinite(pitch_resid_range) and pitch_resid_range >= PITCH_TRANSIENT_FLOOR_RAD)
    )
    lat_jump_suspected = (
        (math.isfinite(float(row.get("lat_offset_range_raw_v1_2", np.nan))) and float(row.get("lat_offset_range_raw_v1_2", np.nan)) >= LAT_RANGE_ROADEDGE_FLOOR_M)
        or (math.isfinite(lat_jump) and lat_jump >= LAT_JUMP_ROADEDGE_FLOOR_M)
    )
    speed_brake = (
        (math.isfinite(speed_drop_start) and speed_drop_start >= SPEED_DROP_ROADEDGE_FLOOR_KMH)
        or (math.isfinite(speed_range) and speed_range >= SPEED_DROP_ROADEDGE_FLOOR_KMH)
        or (math.isfinite(brake_peak) and brake_peak >= BRAKE_ACTIVE_FLOOR)
    )
    context = str(row.get("road_module_names", "")).lower()
    context_suggests_roadedge = any(k in context for k in ["middle", "fix", "long", "stop", "differentmu"])
    curve_grade_exemption = is_curve and road_grade_like_z
    roadedge = (
        not curve_grade_exemption
        and (
            (lat_jump_suspected and speed_brake and context_suggests_roadedge)
            or (lat_jump_suspected and z_transient and speed_brake)
            or (z_transient and speed_brake and not is_curve)
        )
    )

    out.update(
        {
            "raw_feature_ok_v1_3": True,
            "z_residual_range_v1_3": z_resid_range,
            "z_residual_rate_peak_v1_3": safe_abs_peak(z_resid_rate),
            "z_trend_ratio_v1_3": z_trend_ratio,
            "z_monotonic_fraction_v1_3": z_mono,
            "lat_offset_adjacent_jump_peak_v1_3": lat_jump,
            "lat_offset_large_jump_count_v1_3": count_large_jumps(lat_offset, LAT_JUMP_ROADEDGE_FLOOR_M),
            "speed_drop_from_start_v1_3": speed_drop_start,
            "speed_drop_peak_to_peak_v1_3": speed_range,
            "brake_peak_v1_3": brake_peak,
            "pitch_residual_range_v1_3": pitch_resid_range,
            "road_grade_like_z_v1_3": bool(road_grade_like_z),
            "z_transient_suspected_v1_3": bool(z_transient),
            "lat_offset_jump_suspected_v1_3": bool(lat_jump_suspected),
            "speed_brake_response_suspected_v1_3": bool(speed_brake),
            "roadedge_or_offroad_suspected_v1_3": bool(roadedge),
        }
    )
    return out


def classify_v1_3(row: pd.Series) -> tuple[str, str, str, bool, bool, bool, bool]:
    uid = str(row.get("episode_uid", ""))
    if uid in USER_FEEDBACK_OVERRIDES:
        decision, reason, train, deferred, control, discarded = USER_FEEDBACK_OVERRIDES[uid]
        return decision, reason, reason, train, deferred, control, discarded

    if bool(row.get("is_discarded_v1_1", False)):
        return (
            "discard_prior_review",
            "v1.1 已经人工复核为舍弃/暂缓，本轮继续不进入训练",
            "v1.1 已经人工复核为舍弃/暂缓，本轮继续不进入训练",
            False,
            False,
            False,
            True,
        )
    if bool(row.get("is_control_candidate_v1_1", False)):
        return (
            "control_normal_or_curve",
            "正常弯道或普通操控，仅保留为对照样本",
            "正常弯道或普通操控，仅保留为对照样本",
            False,
            False,
            True,
            False,
        )
    if not bool(row.get("is_train_candidate_v1_1", False)):
        return (
            "discard_not_train_source",
            "不是 v1.1 主训练候选，保守暂缓",
            "不是 v1.1 主训练候选，保守暂缓",
            False,
            False,
            False,
            True,
        )

    duration = float(row.get("episode_duration_s", np.nan))
    long30 = math.isfinite(duration) and duration > EXTREME_DURATION_MAX_S
    long20 = math.isfinite(duration) and duration > REVIEW_DURATION_MAX_S
    long15 = math.isfinite(duration) and duration > NORMAL_DURATION_MAX_S
    is_curve = bool(row.get("is_curve_context", False)) or "curve" in str(row.get("road_module_names", "")).lower()
    grade_like = bool(row.get("road_grade_like_z_v1_3", False))
    roadedge = bool(row.get("roadedge_or_offroad_suspected_v1_3", False))
    group = str(row.get("episode_group_id", ""))

    if long30:
        if is_curve and grade_like:
            return (
                "review_long_curve_or_grade",
                "时长超过 30 秒但高度变化更像平滑道路趋势，先按长弯道/坡度复核，不按上下马路处理",
                f"持续时间 {duration:.2f} 秒，且 z 更像道路坡度/弯道趋势，需人工拆分或复核",
                False,
                True,
                False,
                False,
            )
        if roadedge:
            return (
                "defer_roadedge_or_offroad_long",
                "超长片段且存在路边/上下马路风险，暂不进入主训练",
                f"持续时间 {duration:.2f} 秒且存在路边/上下马路风险，暂不进入主训练",
                False,
                True,
                False,
                False,
            )
        return (
            "defer_long_merged",
            "持续时间超过 30 秒，疑似多个过程误合并，需要拆分",
            f"持续时间 {duration:.2f} 秒，疑似多个过程误合并，需要拆分",
            False,
            True,
            False,
            False,
        )

    if long20:
        if is_curve and grade_like:
            return (
                "review_long_curve_or_grade",
                "20 秒以上弯道/坡度趋势片段，不能仅因 z 范围大判为上下马路，先复核是否可拆成单事件",
                f"持续时间 {duration:.2f} 秒，弯道/坡度趋势明显，先复核",
                False,
                True,
                False,
                False,
            )
        if roadedge:
            return (
                "defer_roadedge_or_offroad_long",
                "20 秒以上且存在路边/上下马路风险，暂不进入主训练",
                f"持续时间 {duration:.2f} 秒且存在路边/上下马路风险，暂不进入主训练",
                False,
                True,
                False,
                False,
            )
        return (
            "defer_long_merged",
            "持续时间超过 20 秒，不适合作为单个干净事件直接训练",
            f"持续时间 {duration:.2f} 秒，不适合作为单个干净事件直接训练",
            False,
            True,
            False,
            False,
        )

    if roadedge:
        return (
            "defer_roadedge_or_offroad",
            "车速/制动/横向偏移/高度残差组合提示可能是路边恢复或上下马路，暂不进入主训练",
            "车速/制动/横向偏移/高度残差组合提示可能是路边恢复或上下马路，暂不进入主训练",
            False,
            True,
            False,
            False,
        )

    if is_curve and grade_like and long15:
        return (
            "review_long_curve_or_grade",
            "15 秒以上弯道/坡度趋势片段，先复核是否应拆分",
            f"持续时间 {duration:.2f} 秒，弯道/坡度趋势明显，先复核",
            False,
            True,
            False,
            False,
        )

    if long15:
        return (
            "review_duration_15_20s",
            "持续时间 15-20 秒，先进入复核而不直接训练",
            f"持续时间 {duration:.2f} 秒，先进入复核而不直接训练",
            False,
            True,
            False,
            False,
        )

    if is_curve and bool(row.get("z_transient_suspected_v1_3", False)) and bool(row.get("speed_brake_response_suspected_v1_3", False)):
        return (
            "review_curve_high_dynamics",
            "弯道内高动态样本，不能直接判为上下马路，但也需复核是否为干净极限工况",
            "弯道内高动态样本，不能直接判为上下马路，但也需复核是否为干净极限工况",
            False,
            True,
            False,
            False,
        )

    if group == "conservative_extreme":
        return (
            "train_conservative_extreme",
            "保守/弱操作极限样本，未触发 v1.3 路边/超长风险，保留为训练候选",
            "保守/弱操作极限样本，未触发 v1.3 路边/超长风险，保留为训练候选",
            True,
            False,
            False,
            False,
        )

    return (
        "train_target_extreme",
        "目标极限事件，未触发 v1.3 路边/超长风险，保留为训练候选",
        "目标极限事件，未触发 v1.3 路边/超长风险，保留为训练候选",
        True,
        False,
        False,
        False,
    )


def plot_episode_v1_3(row: pd.Series, out_path: Path, cache: dict[str, dict[str, Any]]) -> None:
    rec = v12.load_vehicle_record(str(row.get("vehicle_file", "")), cache)
    if rec is None:
        return
    t = rec["t"]
    start = float(row.get("episode_start_s", np.nan))
    end = float(row.get("episode_end_s", np.nan))
    if not math.isfinite(start) or not math.isfinite(end):
        return
    pad = 3.0
    left = max(float(np.nanmin(t)), start - pad)
    right = min(float(np.nanmax(t)), end + pad)
    if right <= left:
        return
    mask = np.isfinite(t) & (t >= left) & (t <= right)
    if mask.sum() < 3:
        return
    x = t[mask] - start
    signals = rec["signals"]
    z = signals["z"][mask]
    z_resid = detrended_residual(z, x)
    panels = [
        ("方向盘角", signals["steer"][mask]),
        ("车速", signals["speed"][mask]),
        ("制动踏板", signals["brake"][mask]),
        ("横向加速度", signals["ay"][mask]),
        ("横摆角速度", signals["yaw_rate"][mask]),
        ("横滚角", signals["roll"][mask]),
        ("横滚角速度", signals["roll_rate"][mask]),
        ("高度 z", z),
        ("高度去趋势残差", z_resid),
        ("俯仰角", signals["pitch"][mask]),
        ("横向偏移", signals["lat_offset"][mask]),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 18), sharex=True)
    for ax, (label, values) in zip(axes, panels):
        ax.plot(x, values, lw=1.1)
        ax.axvline(0.0, color="crimson", ls="--", lw=1.0, label="episode 开始")
        for col, color, name in [
            ("driver_action_onset_s", "darkorange", "驾驶员动作"),
            ("vehicle_response_onset_s", "purple", "车辆响应"),
            ("condition_peak_s", "green", "风险峰值"),
            ("episode_end_s", "gray", "episode 结束"),
        ]:
            value = row.get(col, np.nan)
            if pd.notna(value):
                ax.axvline(float(value) - start, color=color, ls=":", lw=0.9, label=name)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), fontsize=8, loc="best")
    title = (
        f"{row.get('episode_uid')} | {row.get('v1_3_decision')} | dur={float(row.get('episode_duration_s', np.nan)):.1f}s\n"
        f"z原始={float(row.get('z_range_v1_2', np.nan)):.3f}, z残差={float(row.get('z_residual_range_v1_3', np.nan)):.3f}, "
        f"lat跳变={float(row.get('lat_offset_adjacent_jump_peak_v1_3', np.nan)):.2f}, "
        f"speed_drop={float(row.get('speed_drop_from_start_v1_3', np.nan)):.1f}, brake={float(row.get('brake_peak_v1_3', np.nan)):.2f}"
    )
    fig.suptitle(title, fontsize=12)
    axes[-1].set_xlabel("相对 episode 开始时间 / s")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_review_figures(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_panel_v1_3_path"] = ""
    cache: dict[str, dict[str, Any]] = {}
    buckets = {
        "train_target_extreme": "01_主训练_目标极限事件",
        "train_conservative_extreme": "02_主训练_保守弱操作极限事件",
        "review_duration_15_20s": "03_15到20秒_需要复核",
        "review_long_curve_or_grade": "04_长弯道或平滑坡度_需要复核",
        "review_long_curve_or_grade_user_feedback": "04_长弯道或平滑坡度_需要复核",
        "review_curve_high_dynamics": "05_弯道高动态_需要复核",
        "defer_roadedge_or_offroad": "06_疑似路边恢复或上下马路_暂缓",
        "defer_roadedge_or_offroad_long": "06_疑似路边恢复或上下马路_暂缓",
        "defer_roadedge_or_offroad_user_feedback": "06_疑似路边恢复或上下马路_暂缓",
        "defer_long_merged": "07_超长误合并_需要拆分",
    }
    selected_indices: list[int] = []
    for decision, folder in buckets.items():
        subset = df[df["v1_3_decision"].astype(str).eq(decision)].copy()
        if subset.empty:
            continue
        if decision.startswith("train"):
            subset = subset.sort_values(["vehicle_score_peak", "condition_score_peak"], ascending=False)
        elif "roadedge" in decision:
            subset = subset.sort_values(
                [
                    "speed_drop_from_start_v1_3",
                    "brake_peak_v1_3",
                    "lat_offset_adjacent_jump_peak_v1_3",
                    "z_residual_range_v1_3",
                ],
                ascending=False,
            )
        elif "curve" in decision:
            subset = subset.sort_values(["episode_duration_s", "z_range_v1_2"], ascending=False)
        else:
            subset = subset.sort_values(["episode_duration_s", "condition_score_peak"], ascending=False)
        max_n = MAX_FIGURES_PER_BUCKET.get(decision, 24)
        for idx, row in subset.head(max_n).iterrows():
            file_name = f"{idx:04d}_{row['episode_uid']}.png"
            out_path = FIG_DIR / folder / file_name
            if not out_path.exists():
                plot_episode_v1_3(row, out_path, cache)
            if out_path.exists():
                df.at[idx, "review_panel_v1_3_path"] = str(out_path)
                selected_indices.append(int(idx))
    pd.DataFrame({"row_index": selected_indices}).to_csv(
        TABLE_DIR / "record_episode_v1_3_review_figure_rows.csv", index=False, encoding="utf-8-sig"
    )
    return df


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "暂无。"
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for v in row.tolist():
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_tables(df: pd.DataFrame, road_audit: dict[str, Any]) -> None:
    df.to_csv(TABLE_DIR / "record_level_episodes_all_v1_3.csv", index=False, encoding="utf-8-sig")
    df[df["is_train_candidate_v1_3"]].to_csv(
        TABLE_DIR / "train_candidate_target_episodes_v1_3.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_3_decision"].astype(str).eq("train_target_extreme")].to_csv(
        TABLE_DIR / "train_target_extreme_episodes_v1_3.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_3_decision"].astype(str).eq("train_conservative_extreme")].to_csv(
        TABLE_DIR / "train_conservative_extreme_episodes_v1_3.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_3_decision"].astype(str).str.contains("roadedge|offroad", na=False)].to_csv(
        TABLE_DIR / "suspected_roadedge_or_offroad_episodes_v1_3.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_3_decision"].astype(str).str.contains("curve|grade", na=False)].to_csv(
        TABLE_DIR / "review_curve_or_grade_episodes_v1_3.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_3_decision"].astype(str).eq("defer_long_merged")].to_csv(
        TABLE_DIR / "long_merged_episodes_v1_3.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_3_decision"].astype(str).eq("review_duration_15_20s")].to_csv(
        TABLE_DIR / "duration_15_20s_review_episodes_v1_3.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_deferred_v1_3"]].to_csv(
        TABLE_DIR / "deferred_or_review_episodes_v1_3.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_control_candidate_v1_3"]].to_csv(
        TABLE_DIR / "control_normal_or_curve_episodes_v1_3.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_discarded_v1_3"]].to_csv(
        TABLE_DIR / "discarded_prior_review_episodes_v1_3.csv", index=False, encoding="utf-8-sig"
    )
    df[df["episode_uid"].isin(USER_FEEDBACK_OVERRIDES)].to_csv(
        TABLE_DIR / "user_feedback_override_examples_v1_3.csv", index=False, encoding="utf-8-sig"
    )
    decision_summary = (
        df.groupby("v1_3_decision", dropna=False)
        .agg(v1_3_decision_cn=("v1_3_decision_cn", "first"), count=("v1_3_decision", "size"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    decision_summary.to_csv(TABLE_DIR / "record_episode_v1_3_decision_summary.csv", index=False, encoding="utf-8-sig")
    subject_summary = (
        df.groupby(["v1_3_decision", "subject"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["v1_3_decision", "subject"])
    )
    subject_summary.to_csv(TABLE_DIR / "record_episode_v1_3_subject_summary.csv", index=False, encoding="utf-8-sig")
    module_summary = (
        df.groupby(["v1_3_decision", "road_module_names"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["v1_3_decision", "count"], ascending=[True, False])
    )
    module_summary.to_csv(TABLE_DIR / "record_episode_v1_3_road_module_summary.csv", index=False, encoding="utf-8-sig")
    duration_bins = pd.cut(
        pd.to_numeric(df["episode_duration_s"], errors="coerce"),
        bins=[0, 5, 10, 15, 20, 30, 60, np.inf],
        labels=["0-5", "5-10", "10-15", "15-20", "20-30", "30-60", ">60"],
        include_lowest=True,
    )
    duration_summary = (
        df.assign(duration_bin=duration_bins)
        .groupby(["v1_3_decision", "duration_bin"], observed=False)
        .size()
        .reset_index(name="count")
    )
    duration_summary.to_csv(TABLE_DIR / "record_episode_v1_3_duration_summary.csv", index=False, encoding="utf-8-sig")
    (TABLE_DIR / "record_episode_v1_3_rule_constants.json").write_text(
        json.dumps(
            {
                "normal_duration_max_s": NORMAL_DURATION_MAX_S,
                "review_duration_max_s": REVIEW_DURATION_MAX_S,
                "extreme_duration_max_s": EXTREME_DURATION_MAX_S,
                "lat_range_roadedge_floor_m": LAT_RANGE_ROADEDGE_FLOOR_M,
                "lat_jump_roadedge_floor_m": LAT_JUMP_ROADEDGE_FLOOR_M,
                "speed_drop_roadedge_floor_kmh": SPEED_DROP_ROADEDGE_FLOOR_KMH,
                "brake_active_floor": BRAKE_ACTIVE_FLOOR,
                "z_raw_grade_floor_m": Z_RAW_GRADE_FLOOR_M,
                "z_grade_residual_ratio_max": Z_GRADE_RESIDUAL_RATIO_MAX,
                "z_grade_monotonic_fraction_min": Z_GRADE_MONOTONIC_FRACTION_MIN,
                "z_transient_residual_floor_m": Z_TRANSIENT_RESIDUAL_FLOOR_M,
                "pitch_transient_floor_rad": PITCH_TRANSIENT_FLOOR_RAD,
                "road_source_audit": road_audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_report(df: pd.DataFrame, road_audit: dict[str, Any]) -> None:
    total = len(df)
    v12_train = int(df["is_train_candidate_v1_2"].fillna(False).astype(bool).sum())
    v13_train = int(df["is_train_candidate_v1_3"].fillna(False).astype(bool).sum())
    deferred = int(df["is_deferred_v1_3"].fillna(False).astype(bool).sum())
    roadedge = int(df["v1_3_decision"].astype(str).str.contains("roadedge|offroad", na=False).sum())
    curve_grade = int(df["v1_3_decision"].astype(str).str.contains("curve|grade", na=False).sum())
    decision = pd.read_csv(TABLE_DIR / "record_episode_v1_3_decision_summary.csv")
    user_examples = df[df["episode_uid"].isin(USER_FEEDBACK_OVERRIDES)][
        [
            "episode_uid",
            "road_module_names",
            "episode_duration_s",
            "v1_2_decision",
            "v1_3_decision",
            "z_range_v1_2",
            "z_residual_range_v1_3",
            "lat_offset_range_raw_v1_2",
            "speed_drop_from_start_v1_3",
            "brake_peak_v1_3",
            "review_panel_v1_3_path",
        ]
    ]
    train_duration = pd.to_numeric(df.loc[df["is_train_candidate_v1_3"], "episode_duration_s"], errors="coerce")
    text = f"""# 完整记录级 episode 样本集 v1.3：修正高度误判与路边恢复误判

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 这次为什么改

用户复核 v1.2 图片后指出两个错误：

1. 有些样本实际像上下马路/路边恢复，但 v1.2 因为高度变化不大，被误放进主训练目标极限事件；
2. 有些样本实际是弯道或道路趋势，但 v1.2 因为 `z_range` 很大，被误判为疑似上下马路。

所以 v1.3 的核心变化是：**不再把原始高度范围 `z_range` 当成上下马路的直接证据**。高度只作为辅助信号，并且要区分“平滑道路趋势”和“短时冲击/跳变残差”。

## 道路源文件检查

- 道路 cfg 目录：`{road_audit.get("cfg_dir")}`
- 扫描 cfg 文件数：{road_audit.get("cfg_files")}
- cfg 中类似 `z/height/elevation/altitude` 的赋值数量：{road_audit.get("cfg_height_like_assignment_total")}
- cfg 中明确 `z0/z1/z` 数值数量：{road_audit.get("cfg_z_value_total")}，范围约 {road_audit.get("cfg_z_value_min")} 到 {road_audit.get("cfg_z_value_max")}
- cfg 中类似坡度/横坡的赋值数量：{road_audit.get("cfg_slope_or_bank_assignment_total")}
- 中心线文件：`{road_audit.get("centerline_path")}`
- 中心线字段：`{road_audit.get("centerline_columns")}`

当前检查说明：`curve1/curve2` 等道路 cfg 中确实存在 `z0/z1` 高程设置，这说明弯道或道路模块本身可能带有明显高度变化；但当前中心线表只有 `s/kappa/x/y`，还不能直接给每个车辆时刻扣除道路高程。因此 v1.3 不把车辆 `zx|z` 的绝对范围当成“上下马路”的唯一依据，而是把它拆成“平滑道路趋势”和“短时异常残差”两部分。

## v1.3 规则变化

- `z_range` 很大，但去掉线性趋势后的 `z_residual_range` 较小、变化方向比较单一时，优先标为“长弯道或平滑坡度，需要复核”，不直接标为上下马路。
- `z_range` 不大，但横向偏移跳变、车速大幅下降、制动明显、并且处在 middle/fix/long/低附着等上下文时，标为“疑似路边恢复或上下马路，暂缓”。
- 20 秒以上片段仍然不直接进入主训练；如果它像平滑弯道/坡度，就进入“长弯道或平滑坡度复核”，如果像路边恢复，就进入“疑似路边恢复或上下马路”。
- 用户点名的两个反例加入人工反馈覆盖规则，防止同类错误继续误导主训练样本。

## 数量变化

- 全量 episode：{total}
- v1.2 主训练候选：{v12_train}
- v1.3 主训练候选：{v13_train}
- v1.3 暂缓/复核：{deferred}
- v1.3 疑似路边恢复或上下马路：{roadedge}
- v1.3 长弯道/平滑坡度/弯道高动态复核：{curve_grade}

v1.3 主训练候选时长中位数：{train_duration.median():.3f} 秒；95% 分位：{train_duration.quantile(0.95):.3f} 秒；最大：{train_duration.max():.3f} 秒。

## v1.3 分类表

{md_table(decision)}

## 用户指出的两个反例在 v1.3 中的位置

{md_table(user_examples)}

## 输出位置

- v1.3 全量表：`{TABLE_DIR / "record_level_episodes_all_v1_3.csv"}`
- v1.3 主训练候选：`{TABLE_DIR / "train_candidate_target_episodes_v1_3.csv"}`
- 疑似路边恢复或上下马路：`{TABLE_DIR / "suspected_roadedge_or_offroad_episodes_v1_3.csv"}`
- 长弯道/平滑坡度/弯道高动态复核：`{TABLE_DIR / "review_curve_or_grade_episodes_v1_3.csv"}`
- 分类统计：`{TABLE_DIR / "record_episode_v1_3_decision_summary.csv"}`
- 复核图目录：`{FIG_DIR}`

## 当前结论

v1.3 比 v1.2 更符合你的反馈：它不再简单用高度范围判断上下马路，同时也能把“小高度变化但车速/制动/横向偏移明显异常”的路边恢复风险样本从主训练候选中分出来。

下一步建议先看两类图：

1. `06_疑似路边恢复或上下马路_暂缓`：确认这里是否主要是应排除/暂缓的样本；
2. `04_长弯道或平滑坡度_需要复核` 和 `05_弯道高动态_需要复核`：确认这些是否应该拆分后保留，还是直接作为弯道高动态样本。

本轮没有训练模型。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
    (OUT_ROOT / "record_episode_dataset_v1_3_summary_cn.md").write_text(text, encoding="utf-8")


def append_notes(df: pd.DataFrame) -> None:
    v13_train = int(df["is_train_candidate_v1_3"].fillna(False).astype(bool).sum())
    deferred = int(df["is_deferred_v1_3"].fillna(False).astype(bool).sum())
    roadedge = int(df["v1_3_decision"].astype(str).str.contains("roadedge|offroad", na=False).sum())
    curve_grade = int(df["v1_3_decision"].astype(str).str.contains("curve|grade", na=False).sum())
    block = (
        "## 2026-05-21 完整记录级 episode 样本集 v1.3\n\n"
        "- 为什么做：用户指出 v1.2 把一个上下马路/路边恢复样本误判为目标极限事件，又把一个弯道样本误判为上下马路。\n"
        "- 本轮动作：加入高度去趋势、平滑坡度识别、横向偏移跳变、车速/制动组合风险和用户反例覆盖规则，生成 v1.3 新样本集；本轮不训练模型。\n"
        f"- v1.3 主训练候选：{v13_train}；暂缓/复核：{deferred}；疑似路边恢复或上下马路：{roadedge}；弯道/坡度复核：{curve_grade}。\n"
        f"- 用户查看版报告：`{REPORT_PATH}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        if "## 2026-05-21 完整记录级 episode 样本集 v1.3" not in raw:
            path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")

    artifact = (
        "## 2026-05-21 完整记录级 episode 样本集 v1.3\n\n"
        f"- 用户查看版报告：`{REPORT_PATH}`\n"
        f"- 全量表：`{TABLE_DIR / 'record_level_episodes_all_v1_3.csv'}`\n"
        f"- 主训练候选：`{TABLE_DIR / 'train_candidate_target_episodes_v1_3.csv'}`\n"
        f"- 疑似路边恢复或上下马路：`{TABLE_DIR / 'suspected_roadedge_or_offroad_episodes_v1_3.csv'}`\n"
        f"- 长弯道/平滑坡度/弯道高动态复核：`{TABLE_DIR / 'review_curve_or_grade_episodes_v1_3.csv'}`\n"
        f"- 复核图目录：`{FIG_DIR}`\n"
    )
    raw = ARTIFACT_INDEX.read_text(encoding="utf-8") if ARTIFACT_INDEX.exists() else ""
    if "## 2026-05-21 完整记录级 episode 样本集 v1.3" not in raw:
        ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not V12_ALL.exists():
        raise FileNotFoundError(V12_ALL)
    df = pd.read_csv(V12_ALL, encoding="utf-8-sig", low_memory=False)
    road_audit = audit_road_height_sources()
    cache: dict[str, dict[str, Any]] = {}
    rows = []
    for i, row in df.iterrows():
        rows.append(compute_episode_features_v1_3(row, cache))
        if (i + 1) % 250 == 0:
            print(f"v1.3 features {i + 1}/{len(df)}", flush=True)
    extra_df = pd.DataFrame(rows)
    df = pd.concat([df.reset_index(drop=True), extra_df], axis=1)

    decisions = df.apply(classify_v1_3, axis=1, result_type="expand")
    decisions.columns = [
        "v1_3_decision",
        "v1_3_decision_cn",
        "v1_3_decision_detail_cn",
        "is_train_candidate_v1_3",
        "is_deferred_v1_3",
        "is_control_candidate_v1_3",
        "is_discarded_v1_3",
    ]
    df = pd.concat([df, decisions], axis=1)
    df = make_review_figures(df)
    write_tables(df, road_audit)
    write_report(df, road_audit)
    append_notes(df)
    print(pd.read_csv(TABLE_DIR / "record_episode_v1_3_decision_summary.csv").to_string(index=False))
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
