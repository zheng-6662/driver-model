#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build v1.2 record-level episode dataset with long/off-road recovery screening.

v1.1 kept most automatically detected record-level episodes after visual review.
The user then pointed out an important failure mode: very long episodes can be
false merges caused by continuous experiments where the driver leaves the road
and drives back, producing z/pitch/body shake signals. This script does not
redetect all episodes from scratch. It adds a v1.2 decision layer on top of
v1.1 by reading raw vehicle CSV files and separating target extreme events from
suspected off-road/road-recovery and over-merged long episodes.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


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
V11_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_1_reviewed"
V11_ALL = V11_ROOT / "tables" / "record_level_episodes_all_reviewed_v1_1.csv"
OUT_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_2_cleaned"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "review_panels_v1_2"
LOG_DIR = OUT_ROOT / "logs"
REPORT_PATH = ROOT / "09_reports" / "stage02_record_episode_reconstruction_v1_2_user_summary_cn.md"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-21.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"

NORMAL_DURATION_MAX_S = 15.0
REVIEW_DURATION_MAX_S = 20.0
OFFROAD_LAT_OFFSET_FLOOR_M = 15.0
Z_RANGE_FLOOR_M = 0.06
Z_RATE_FLOOR_MPS = 0.04
PITCH_RANGE_FLOOR_RAD = 0.08
PITCH_RATE_FLOOR_RADPS = 0.08
MAX_FIGURES_PER_BUCKET = {
    "train_target_extreme": 30,
    "train_conservative_extreme": 24,
    "review_duration_15_20s": 24,
    "defer_offroad_recovery": 36,
    "defer_long_merged": 36,
}

COL_ALIASES = {
    "time": ["StorageTime", "time", "timestamp", "Time"],
    "steer": ["zx|SteeringWheel", "SteeringWheel", "steering_wheel", "steer"],
    "speed": ["zx1|v_km/h", "v_km/h", "speed_kmh"],
    "brake": ["zx|BrakePedal", "BrakePedal", "brake"],
    "ay": ["zx|ay", "ay", "lateral_acceleration"],
    "yaw_rate": ["zx|vyaw", "vyaw", "yaw_rate"],
    "roll": ["zx|roll", "roll", "roll_angle"],
    "roll_rate": ["zx|vroll", "vroll", "roll_rate"],
    "pitch": ["zx|pitch", "pitch"],
    "pitch_rate": ["zx|vpitch", "vpitch", "pitch_rate"],
    "z": ["zx|z", "z", "height", "altitude"],
    "lat_offset": ["zx1|lateraldistance", "zx|lateraldistance", "lateraldistance", "lane_offset"],
    "mu": ["zx1|mu", "mu"],
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_PATH.parent, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv_smart(path: Path, nrows: int | None = None) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def pick_col(df: pd.DataFrame, key: str) -> str | None:
    for col in COL_ALIASES[key]:
        if col in df.columns:
            return col
    lower = {str(c).lower(): c for c in df.columns}
    for col in COL_ALIASES[key]:
        c = lower.get(col.lower())
        if c is not None:
            return str(c)
    return None


def parse_time_seconds(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.8:
        arr = numeric.to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.full(len(series), np.nan)
        return arr - finite[0]
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() == 0:
        return np.full(len(series), np.nan)
    base = parsed.dropna().iloc[0]
    return (parsed - base).dt.total_seconds().to_numpy(dtype=float)


def finite_series(df: pd.DataFrame, col: str | None) -> np.ndarray:
    if not col or col not in df.columns:
        return np.full(len(df), np.nan, dtype=float)
    out = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    idx = np.arange(out.size)
    valid = np.isfinite(out)
    if valid.sum() == 0:
        return np.full(len(df), np.nan, dtype=float)
    if valid.sum() < out.size:
        out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def robust_mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(arr - med)) * 1.4826)


def robust_high_threshold(values: pd.Series | np.ndarray, floor: float, q: float = 0.95) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return floor
    med = float(np.nanmedian(arr))
    mad = robust_mad(arr)
    qv = float(np.nanquantile(arr, q))
    mad_thr = med + 3.0 * mad if math.isfinite(mad) else floor
    return float(max(floor, qv, mad_thr))


def gradient(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & np.isfinite(t)
    if valid.sum() < 3:
        return out
    tt = t[valid]
    vv = values[valid]
    order = np.argsort(tt)
    tt = tt[order]
    vv = vv[order]
    keep = np.r_[True, np.diff(tt) > 1e-6]
    tt = tt[keep]
    vv = vv[keep]
    if tt.size < 3:
        return np.zeros_like(values, dtype=float)
    deriv = np.gradient(vv, tt)
    out = np.interp(t, tt, deriv, left=deriv[0], right=deriv[-1])
    return out


def safe_range(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))


def safe_abs_peak(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(np.abs(arr)))


def load_vehicle_record(path: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if path in cache:
        return cache[path]
    p = Path(str(path))
    if not p.exists():
        cache[path] = None
        return None
    df = read_csv_smart(p)
    time_col = pick_col(df, "time")
    if time_col is None:
        cache[path] = None
        return None
    t = parse_time_seconds(df[time_col])
    cols = {key: pick_col(df, key) for key in COL_ALIASES if key != "time"}
    signals = {key: finite_series(df, col) for key, col in cols.items()}
    if np.isfinite(signals.get("z", np.array([]))).sum() >= 3:
        signals["z_rate"] = gradient(signals["z"], t)
    else:
        signals["z_rate"] = np.full_like(t, np.nan, dtype=float)
    if np.isfinite(signals.get("pitch", np.array([]))).sum() >= 3:
        signals["pitch_rate_calc"] = gradient(signals["pitch"], t)
    else:
        signals["pitch_rate_calc"] = np.full_like(t, np.nan, dtype=float)
    payload = {"path": p, "df": df, "t": t, "cols": cols, "signals": signals}
    cache[path] = payload
    return payload


def compute_episode_features(row: pd.Series, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rec = load_vehicle_record(str(row.get("vehicle_file", "")), cache)
    out: dict[str, Any] = {
        "raw_feature_ok_v1_2": False,
        "z_range_v1_2": np.nan,
        "z_rate_peak_v1_2": np.nan,
        "pitch_range_v1_2": np.nan,
        "pitch_rate_peak_v1_2": np.nan,
        "lat_offset_range_raw_v1_2": np.nan,
        "speed_range_raw_v1_2": np.nan,
        "brake_range_raw_v1_2": np.nan,
        "roll_range_raw_v1_2": np.nan,
        "ay_abs_peak_raw_v1_2": np.nan,
        "yaw_rate_abs_peak_raw_v1_2": np.nan,
    }
    if rec is None:
        return out
    start = float(row.get("episode_start_s", np.nan))
    end = float(row.get("episode_end_s", np.nan))
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return out
    t = rec["t"]
    mask = np.isfinite(t) & (t >= start) & (t <= end)
    if mask.sum() < 3:
        return out
    signals = rec["signals"]
    pitch_rate = signals.get("pitch_rate")
    pitch_rate_calc = signals.get("pitch_rate_calc")
    if pitch_rate is None or not np.isfinite(pitch_rate).any():
        pitch_rate = pitch_rate_calc
    out.update(
        {
            "raw_feature_ok_v1_2": True,
            "z_range_v1_2": safe_range(signals["z"][mask]),
            "z_rate_peak_v1_2": safe_abs_peak(signals["z_rate"][mask]),
            "pitch_range_v1_2": safe_range(signals["pitch"][mask]),
            "pitch_rate_peak_v1_2": safe_abs_peak(pitch_rate[mask]),
            "lat_offset_range_raw_v1_2": safe_range(signals["lat_offset"][mask]),
            "speed_range_raw_v1_2": safe_range(signals["speed"][mask]),
            "brake_range_raw_v1_2": safe_range(signals["brake"][mask]),
            "roll_range_raw_v1_2": safe_range(signals["roll"][mask]),
            "ay_abs_peak_raw_v1_2": safe_abs_peak(signals["ay"][mask]),
            "yaw_rate_abs_peak_raw_v1_2": safe_abs_peak(signals["yaw_rate"][mask]),
        }
    )
    return out


def add_threshold_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    train_mask = df["is_train_candidate_v1_1"].fillna(False).astype(bool)
    base_df = df[train_mask].copy()
    thresholds = {
        "normal_duration_max_s": NORMAL_DURATION_MAX_S,
        "review_duration_max_s": REVIEW_DURATION_MAX_S,
        "z_range_thr": robust_high_threshold(base_df["z_range_v1_2"], Z_RANGE_FLOOR_M),
        "z_rate_thr": robust_high_threshold(base_df["z_rate_peak_v1_2"], Z_RATE_FLOOR_MPS),
        "pitch_range_thr": robust_high_threshold(base_df["pitch_range_v1_2"], PITCH_RANGE_FLOOR_RAD),
        "pitch_rate_thr": robust_high_threshold(base_df["pitch_rate_peak_v1_2"], PITCH_RATE_FLOOR_RADPS),
        "lat_offset_range_thr": robust_high_threshold(base_df["lat_offset_range_raw_v1_2"], OFFROAD_LAT_OFFSET_FLOOR_M),
        "speed_range_thr": robust_high_threshold(base_df["speed_range_raw_v1_2"], 60.0),
    }

    duration = pd.to_numeric(df["episode_duration_s"], errors="coerce")
    z_range = pd.to_numeric(df["z_range_v1_2"], errors="coerce")
    z_rate = pd.to_numeric(df["z_rate_peak_v1_2"], errors="coerce")
    pitch_range = pd.to_numeric(df["pitch_range_v1_2"], errors="coerce")
    pitch_rate = pd.to_numeric(df["pitch_rate_peak_v1_2"], errors="coerce")
    lat_range = pd.to_numeric(df["lat_offset_range_raw_v1_2"], errors="coerce")
    speed_range = pd.to_numeric(df["speed_range_raw_v1_2"], errors="coerce")

    df["duration_over_15s_v1_2"] = duration > NORMAL_DURATION_MAX_S
    df["duration_over_20s_v1_2"] = duration > REVIEW_DURATION_MAX_S
    df["height_jump_suspected_v1_2"] = (z_range >= thresholds["z_range_thr"]) | (z_rate >= thresholds["z_rate_thr"])
    df["pitch_jump_suspected_v1_2"] = (pitch_range >= thresholds["pitch_range_thr"]) | (
        pitch_rate >= thresholds["pitch_rate_thr"]
    )
    df["lat_offset_extreme_suspected_v1_2"] = lat_range >= thresholds["lat_offset_range_thr"]
    df["speed_range_extreme_suspected_v1_2"] = speed_range >= thresholds["speed_range_thr"]
    df["offroad_or_road_recovery_suspected_v1_2"] = (
        (
            df["height_jump_suspected_v1_2"]
            & (df["pitch_jump_suspected_v1_2"] | df["lat_offset_extreme_suspected_v1_2"] | (duration > 12.0))
        )
        | (df["lat_offset_extreme_suspected_v1_2"] & ((duration > 12.0) | df["height_jump_suspected_v1_2"]))
        | (df["duration_over_20s_v1_2"] & (df["height_jump_suspected_v1_2"] | df["pitch_jump_suspected_v1_2"]))
    )
    return df, thresholds


def classify_v1_2(row: pd.Series) -> tuple[str, str, str, bool, bool, bool, bool]:
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
    offroad = bool(row.get("offroad_or_road_recovery_suspected_v1_2", False))
    long20 = bool(row.get("duration_over_20s_v1_2", False))
    long15 = bool(row.get("duration_over_15s_v1_2", False))
    group = str(row.get("episode_group_id", ""))

    if long20 and offroad:
        return (
            "defer_offroad_recovery_long",
            "持续时间超过 20 秒且存在高度/俯仰/横向偏移异常，疑似上下马路或路外恢复误合并",
            f"持续时间 {duration:.2f} 秒且存在高度/俯仰/横向偏移异常，疑似上下马路或路外恢复误合并",
            False,
            True,
            False,
            False,
        )
    if offroad:
        return (
            "defer_offroad_recovery",
            "高度/俯仰/横向偏移特征提示可能是上下马路或路外恢复，先不进入主训练",
            "高度/俯仰/横向偏移特征提示可能是上下马路或路外恢复，先不进入主训练",
            False,
            True,
            False,
            False,
        )
    if long20:
        return (
            "defer_long_merged",
            "持续时间超过 20 秒，不符合单个事件通常只有十几秒的实验逻辑，疑似多个过程误合并",
            f"持续时间 {duration:.2f} 秒，不符合单个事件通常只有十几秒的实验逻辑，疑似多个过程误合并",
            False,
            True,
            False,
            False,
        )
    if long15:
        return (
            "review_duration_15_20s",
            "持续时间 15-20 秒，接近或超过常规事件上限，先进入复核而不直接训练",
            f"持续时间 {duration:.2f} 秒，接近或超过常规事件上限，先进入复核而不直接训练",
            False,
            True,
            False,
            False,
        )
    if group == "conservative_extreme":
        return (
            "train_conservative_extreme",
            "保守/弱操作极限样本，未触发超长或路外恢复风险，保留为训练候选",
            "保守/弱操作极限样本，未触发超长或路外恢复风险，保留为训练候选",
            True,
            False,
            False,
            False,
        )
    return (
        "train_target_extreme",
        "核心/次级目标极限事件，未触发超长或路外恢复风险，保留为训练候选",
        "核心/次级目标极限事件，未触发超长或路外恢复风险，保留为训练候选",
        True,
        False,
        False,
        False,
    )


def vehicle_record_for_plot(row: pd.Series, cache: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    return load_vehicle_record(str(row.get("vehicle_file", "")), cache)


def plot_episode(row: pd.Series, out_path: Path, cache: dict[str, dict[str, Any]]) -> None:
    rec = vehicle_record_for_plot(row, cache)
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
    panels = [
        ("方向盘角", signals["steer"][mask]),
        ("车速", signals["speed"][mask]),
        ("制动踏板", signals["brake"][mask]),
        ("横向加速度", signals["ay"][mask]),
        ("横摆角速度", signals["yaw_rate"][mask]),
        ("横滚角", signals["roll"][mask]),
        ("横滚角速度", signals["roll_rate"][mask]),
        ("高度 z", signals["z"][mask]),
        ("俯仰角", signals["pitch"][mask]),
        ("横向偏移", signals["lat_offset"][mask]),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 16), sharex=True)
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
        f"{row.get('episode_uid')} | {row.get('v1_2_decision')} | dur={float(row.get('episode_duration_s', np.nan)):.1f}s\n"
        f"z_range={float(row.get('z_range_v1_2', np.nan)):.3f}, pitch_range={float(row.get('pitch_range_v1_2', np.nan)):.3f}, "
        f"lat_range={float(row.get('lat_offset_range_raw_v1_2', np.nan)):.2f}"
    )
    fig.suptitle(title, fontsize=12)
    axes[-1].set_xlabel("相对 episode 开始时间 / s")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_review_figures(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_panel_v1_2_path"] = ""
    cache: dict[str, dict[str, Any]] = {}
    buckets = {
        "train_target_extreme": "01_主训练_目标极限事件",
        "train_conservative_extreme": "02_主训练_保守弱操作极限事件",
        "review_duration_15_20s": "03_15到20秒_需要复核",
        "defer_offroad_recovery": "04_疑似上下马路或路外恢复",
        "defer_offroad_recovery_long": "04_疑似上下马路或路外恢复",
        "defer_long_merged": "05_超长误合并_需要拆分",
    }
    selected_indices: list[int] = []
    for decision, folder in buckets.items():
        subset = df[df["v1_2_decision"].astype(str).eq(decision)].copy()
        if subset.empty:
            continue
        if decision.startswith("train"):
            subset = subset.sort_values(["vehicle_score_peak", "condition_score_peak"], ascending=False)
        elif "offroad" in decision:
            subset = subset.sort_values(
                ["z_range_v1_2", "pitch_range_v1_2", "lat_offset_range_raw_v1_2", "episode_duration_s"],
                ascending=False,
            )
        else:
            subset = subset.sort_values(["episode_duration_s", "condition_score_peak"], ascending=False)
        max_n = MAX_FIGURES_PER_BUCKET.get(decision, 24)
        for idx, row in subset.head(max_n).iterrows():
            file_name = f"{idx:04d}_{row['episode_uid']}.png"
            out_path = FIG_DIR / folder / file_name
            plot_episode(row, out_path, cache)
            if out_path.exists():
                df.at[idx, "review_panel_v1_2_path"] = str(out_path)
                selected_indices.append(idx)
    pd.DataFrame({"row_index": selected_indices}).to_csv(
        TABLE_DIR / "record_episode_v1_2_review_figure_rows.csv", index=False, encoding="utf-8-sig"
    )
    return df


def write_tables(df: pd.DataFrame, thresholds: dict[str, float]) -> None:
    df.to_csv(TABLE_DIR / "record_level_episodes_all_v1_2.csv", index=False, encoding="utf-8-sig")
    df[df["is_train_candidate_v1_2"]].to_csv(
        TABLE_DIR / "train_candidate_target_episodes_v1_2.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_2_decision"].astype(str).eq("train_target_extreme")].to_csv(
        TABLE_DIR / "train_target_extreme_episodes_v1_2.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_2_decision"].astype(str).eq("train_conservative_extreme")].to_csv(
        TABLE_DIR / "train_conservative_extreme_episodes_v1_2.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_2_decision"].astype(str).str.contains("offroad", na=False)].to_csv(
        TABLE_DIR / "suspected_offroad_or_road_recovery_episodes_v1_2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    df[df["v1_2_decision"].astype(str).eq("defer_long_merged")].to_csv(
        TABLE_DIR / "long_merged_episodes_v1_2.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_2_decision"].astype(str).eq("review_duration_15_20s")].to_csv(
        TABLE_DIR / "duration_15_20s_review_episodes_v1_2.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_deferred_v1_2"]].to_csv(
        TABLE_DIR / "deferred_or_review_episodes_v1_2.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_control_candidate_v1_2"]].to_csv(
        TABLE_DIR / "control_normal_or_curve_episodes_v1_2.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_discarded_v1_2"]].to_csv(
        TABLE_DIR / "discarded_prior_review_episodes_v1_2.csv", index=False, encoding="utf-8-sig"
    )
    decision_summary = (
        df.groupby(["v1_2_decision", "v1_2_decision_cn"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    decision_summary.to_csv(TABLE_DIR / "record_episode_v1_2_decision_summary.csv", index=False, encoding="utf-8-sig")
    subject_summary = (
        df.groupby(["v1_2_decision", "subject"], dropna=False).size().reset_index(name="count").sort_values(
            ["v1_2_decision", "subject"]
        )
    )
    subject_summary.to_csv(TABLE_DIR / "record_episode_v1_2_subject_summary.csv", index=False, encoding="utf-8-sig")
    duration_bins = pd.cut(
        pd.to_numeric(df["episode_duration_s"], errors="coerce"),
        bins=[0, 5, 10, 15, 20, 30, 60, np.inf],
        labels=["0-5", "5-10", "10-15", "15-20", "20-30", "30-60", ">60"],
        include_lowest=True,
    )
    duration_summary = df.assign(duration_bin=duration_bins).groupby(
        ["v1_2_decision", "duration_bin"], observed=False
    ).size().reset_index(name="count")
    duration_summary.to_csv(TABLE_DIR / "record_episode_v1_2_duration_summary.csv", index=False, encoding="utf-8-sig")
    (TABLE_DIR / "record_episode_v1_2_thresholds.json").write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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


def write_report(df: pd.DataFrame, thresholds: dict[str, float]) -> None:
    total = len(df)
    v11_train = int(df["is_train_candidate_v1_1"].fillna(False).astype(bool).sum())
    v12_train = int(df["is_train_candidate_v1_2"].fillna(False).astype(bool).sum())
    deferred = int(df["is_deferred_v1_2"].fillna(False).astype(bool).sum())
    offroad = int(df["v1_2_decision"].astype(str).str.contains("offroad", na=False).sum())
    long20 = int(df["duration_over_20s_v1_2"].fillna(False).astype(bool).sum())
    decision = pd.read_csv(TABLE_DIR / "record_episode_v1_2_decision_summary.csv")
    duration_train = pd.to_numeric(df.loc[df["is_train_candidate_v1_2"], "episode_duration_s"], errors="coerce")
    duration_v11 = pd.to_numeric(df.loc[df["is_train_candidate_v1_1"], "episode_duration_s"], errors="coerce")
    text = f"""# 完整记录级 episode 样本集 v1.2：超长片段与上下马路/路外恢复筛除

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 这次为什么做

用户指出：真实单个事件通常最长也就十几秒；当前 v1.1 中出现 60 秒、80 秒、105 秒 episode，明显不应理解为真实事件持续这么久。这类片段很可能来自连续实验中驾驶员开下马路、重新开回马路、车身高度变化、车身抖动和恢复驾驶过程，被自动检测误合并为一个长 episode。

因此 v1.2 不训练模型，只在 v1.1 基础上增加一层清洗和分流：

- 保留短时、语义较清楚的目标极限事件；
- 保留保守/弱操作极限事件；
- 将疑似上下马路/路外恢复片段单独分出去；
- 将超过合理事件长度的片段单独分出去，后续需要拆分或人工复核；
- 不直接删除这些风险片段，而是保存表格和复核图。

## 输入与规则

- 输入表：`{V11_ALL}`
- 原始车辆 CSV：来自每个 episode 的 `vehicle_file`
- 新增检测信号：`zx|z` 高度、`zx|pitch` 俯仰角、`zx|vpitch` 俯仰角速度、横向偏移、车速、制动、横滚等。
- 正常主训练时长上限：{NORMAL_DURATION_MAX_S:.1f} 秒。
- 15 到 20 秒：先进入复核。
- 超过 20 秒：默认不是单个干净事件，进入暂缓/拆分类。

## 阈值

```json
{json.dumps(thresholds, ensure_ascii=False, indent=2)}
```

## 数量变化

- v1.1 全量 episode：{total}
- v1.1 主训练候选：{v11_train}
- v1.2 主训练候选：{v12_train}
- v1.2 暂缓/复核：{deferred}
- v1.2 疑似上下马路/路外恢复：{offroad}
- v1.2 超过 20 秒片段：{long20}

## v1.2 分类表

{md_table(decision)}

## 时长变化

- v1.1 主训练候选时长中位数：{duration_v11.median():.3f} 秒；95% 分位：{duration_v11.quantile(0.95):.3f} 秒；最大：{duration_v11.max():.3f} 秒。
- v1.2 主训练候选时长中位数：{duration_train.median():.3f} 秒；95% 分位：{duration_train.quantile(0.95):.3f} 秒；最大：{duration_train.max():.3f} 秒。

## 输出位置

- v1.2 全量表：`{TABLE_DIR / "record_level_episodes_all_v1_2.csv"}`
- v1.2 主训练候选：`{TABLE_DIR / "train_candidate_target_episodes_v1_2.csv"}`
- 疑似上下马路/路外恢复：`{TABLE_DIR / "suspected_offroad_or_road_recovery_episodes_v1_2.csv"}`
- 超长误合并：`{TABLE_DIR / "long_merged_episodes_v1_2.csv"}`
- 15 到 20 秒复核：`{TABLE_DIR / "duration_15_20s_review_episodes_v1_2.csv"}`
- 分类统计：`{TABLE_DIR / "record_episode_v1_2_decision_summary.csv"}`
- 复核图目录：`{FIG_DIR}`

## 当前结论

v1.2 比 v1.1 更适合作为下一轮训练入口，因为它不再把 30 秒、60 秒、100 秒的连续恢复过程直接当成单个目标事件。下一步建议先人工查看 v1.2 的三类图：

1. 主训练目标事件；
2. 疑似上下马路/路外恢复；
3. 超长误合并/需要拆分。

如果这些分类大体符合直觉，再用 `train_candidate_target_episodes_v1_2.csv` 重跑车辆-only。否则继续调整 v1.2 规则。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
    (OUT_ROOT / "record_episode_dataset_v1_2_summary_cn.md").write_text(text, encoding="utf-8")


def append_notes(df: pd.DataFrame) -> None:
    v12_train = int(df["is_train_candidate_v1_2"].fillna(False).astype(bool).sum())
    deferred = int(df["is_deferred_v1_2"].fillna(False).astype(bool).sum())
    offroad = int(df["v1_2_decision"].astype(str).str.contains("offroad", na=False).sum())
    block = (
        "## 2026-05-21 完整记录级 episode 样本集 v1.2\n\n"
        "- 为什么做：用户指出 v1.1 中 60 秒、80 秒、105 秒 episode 不符合真实单事件逻辑，可能是上下马路/路外恢复或连续过程误合并。\n"
        f"- 本轮动作：加入 `zx|z` 高度、俯仰、横向偏移和时长约束，生成 v1.2 新样本集，不训练模型。\n"
        f"- v1.2 主训练候选：{v12_train}；暂缓/复核：{deferred}；疑似上下马路/路外恢复：{offroad}。\n"
        f"- 用户查看版报告：`{REPORT_PATH}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        if "## 2026-05-21 完整记录级 episode 样本集 v1.2" not in raw:
            path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")

    artifact = (
        "## 2026-05-21 完整记录级 episode 样本集 v1.2\n\n"
        f"- 用户查看版报告：`{REPORT_PATH}`\n"
        f"- 全量表：`{TABLE_DIR / 'record_level_episodes_all_v1_2.csv'}`\n"
        f"- 主训练候选：`{TABLE_DIR / 'train_candidate_target_episodes_v1_2.csv'}`\n"
        f"- 疑似上下马路/路外恢复：`{TABLE_DIR / 'suspected_offroad_or_road_recovery_episodes_v1_2.csv'}`\n"
        f"- 超长误合并：`{TABLE_DIR / 'long_merged_episodes_v1_2.csv'}`\n"
        f"- 复核图目录：`{FIG_DIR}`\n"
    )
    raw = ARTIFACT_INDEX.read_text(encoding="utf-8") if ARTIFACT_INDEX.exists() else ""
    if "## 2026-05-21 完整记录级 episode 样本集 v1.2" not in raw:
        ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not V11_ALL.exists():
        raise FileNotFoundError(V11_ALL)
    df = pd.read_csv(V11_ALL, encoding="utf-8-sig", low_memory=False)
    cache: dict[str, dict[str, Any]] = {}
    rows = []
    for i, row in df.iterrows():
        extra = compute_episode_features(row, cache)
        rows.append(extra)
        if (i + 1) % 250 == 0:
            print(f"features {i + 1}/{len(df)}", flush=True)
    extra_df = pd.DataFrame(rows)
    df = pd.concat([df.reset_index(drop=True), extra_df], axis=1)
    df, thresholds = add_threshold_flags(df)

    decisions = df.apply(classify_v1_2, axis=1, result_type="expand")
    decisions.columns = [
        "v1_2_decision",
        "v1_2_decision_cn",
        "v1_2_decision_detail_cn",
        "is_train_candidate_v1_2",
        "is_deferred_v1_2",
        "is_control_candidate_v1_2",
        "is_discarded_v1_2",
    ]
    df = pd.concat([df, decisions], axis=1)
    df = make_review_figures(df)
    write_tables(df, thresholds)
    write_report(df, thresholds)
    append_notes(df)
    print(pd.read_csv(TABLE_DIR / "record_episode_v1_2_decision_summary.csv").to_string(index=False))
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
