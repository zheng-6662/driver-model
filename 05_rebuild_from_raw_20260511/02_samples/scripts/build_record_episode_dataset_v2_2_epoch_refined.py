#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Refine episode boundaries after v2.1 sample recovery.

This script does not train models. It audits and rebuilds episode timing using
raw vehicle signals around each recovered candidate:

- keep old sample identity and v2.1 role;
- refine event start from sustained driver/vehicle activity connected to the
  main local risk peak;
- refine event end from post-peak quiet/stabilization;
- keep model anchor separate from full episode boundaries.
"""

from __future__ import annotations

import math
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

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
REBUILD_ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
SCRIPT_DIR = REBUILD_ROOT / "02_samples" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_record_level_episode_reconstruction_v1_0 as v10  # noqa: E402


V21_ROOT = REBUILD_ROOT / "02_samples" / "record_level_episode_reconstruction_v2_1_reference_height_recovery"
V21_ALL = V21_ROOT / "tables" / "manifest_all_v2_1_reference_height_recovery.csv"
OUT_ROOT = REBUILD_ROOT / "02_samples" / "record_level_episode_reconstruction_v2_2_epoch_refined"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "epoch_boundary_review_v2_2"
REPORT_PATH = REBUILD_ROOT / "09_reports" / "stage02_record_episode_reconstruction_v2_2_epoch_user_summary_cn.md"


PREVIEW_BEFORE_SEC = 4.0
PREVIEW_AFTER_SEC = 8.0
CORE_PEAK_BEFORE_SEC = 1.5
CORE_PEAK_AFTER_SEC = 6.0
QUIET_GAP_SEC = 0.60
MIN_EVENT_DURATION_SEC = 1.20
MAX_EVENT_DURATION_SEC = 15.0
MODEL_PRE_WINDOW_SEC = 2.0
MODEL_EARLY_OBS_SEC = 0.5
MODEL_LABEL_WINDOW_SEC = 5.0


EXTRA_ALIASES = {
    "z": ["zx|z", "z", "pos_z"],
    "pitch": ["zx|pitch", "pitch"],
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, REPORT_PATH.parent]:
        path.mkdir(parents=True, exist_ok=True)


def as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def as_bool(value: object) -> bool:
    return as_text(value).strip().lower() in {"true", "1", "yes", "y", "是"}


def as_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_times(row: pd.Series, names: list[str]) -> list[float]:
    vals: list[float] = []
    for name in names:
        v = as_float(row.get(name))
        if math.isfinite(v):
            vals.append(v)
    return vals


def pick_extra_col(df: pd.DataFrame, key: str) -> str | None:
    for col in EXTRA_ALIASES[key]:
        if col in df.columns:
            return col
    lower = {str(c).lower(): c for c in df.columns}
    for col in EXTRA_ALIASES[key]:
        match = lower.get(col.lower())
        if match is not None:
            return str(match)
    return None


def fill_finite(values: np.ndarray, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    idx = np.arange(arr.size)
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return np.full(arr.shape, default, dtype=float)
    if valid.sum() < arr.size:
        arr = arr.copy()
        arr[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])
    return arr


def record_threshold(values: np.ndarray, floor: float, q: float = 92.0, k_mad: float = 2.5) -> float:
    arr = np.abs(np.asarray(values, dtype=float))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return floor
    med = float(np.nanmedian(arr))
    mad = v10.robust_mad(arr)
    qv = float(np.nanpercentile(arr, q))
    return float(max(floor, qv, med + k_mad * mad if math.isfinite(mad) else floor))


def load_vehicle(path_text: str, cache: dict[str, dict]) -> dict | None:
    path = Path(path_text)
    key = str(path)
    if key in cache:
        return cache[key]
    if not path.exists():
        cache[key] = None
        return None
    df = v10.read_csv_smart(path)
    time_col = v10.pick_col(df, "time")
    if time_col is None:
        cache[key] = None
        return None
    t = v10.parse_time_seconds(df[time_col])
    valid = np.isfinite(t)
    if valid.sum() < 10:
        cache[key] = None
        return None

    cols = {name: v10.pick_col(df, name) for name in v10.COL_ALIASES}
    cols["z"] = pick_extra_col(df, "z")
    cols["pitch"] = pick_extra_col(df, "pitch")

    signals: dict[str, np.ndarray] = {}
    for name in ["steer", "speed_kmh", "brake", "ay", "yaw_rate", "roll_rate", "roll", "lat_offset", "mu", "z", "pitch"]:
        col = cols.get(name)
        signals[name] = fill_finite(v10.finite_series(df, col, default=0.0) if col else np.zeros(len(df)))

    dt = np.diff(t[np.isfinite(t)])
    median_dt = float(np.nanmedian(dt[dt > 1e-6])) if np.any(dt > 1e-6) else 0.02
    fs = 1.0 / median_dt if median_dt > 1e-6 else 50.0
    smooth_width = max(3, int(round(fs * 0.12)))
    if smooth_width % 2 == 0:
        smooth_width += 1
    steer_smooth = v10.moving_average(signals["steer"], smooth_width)
    signals["steer_smooth"] = steer_smooth
    signals["steer_rate"] = fill_finite(v10.gradient(steer_smooth, t))
    signals["speed_rate"] = fill_finite(v10.gradient(signals["speed_kmh"], t))
    signals["lat_offset_rate"] = fill_finite(v10.gradient(signals["lat_offset"], t))

    thresholds = {
        "steer_rate": record_threshold(signals["steer_rate"], floor=2.0, q=94.0, k_mad=2.8),
        "steer_delta": max(0.25, record_threshold(signals["steer_smooth"] - np.nanmedian(signals["steer_smooth"]), floor=0.25, q=85.0, k_mad=2.0)),
        "brake": max(0.05, record_threshold(signals["brake"], floor=0.05, q=90.0, k_mad=2.0)),
        "ay": record_threshold(signals["ay"], floor=1.2, q=90.0, k_mad=2.0),
        "yaw_rate": record_threshold(signals["yaw_rate"], floor=0.05, q=90.0, k_mad=2.0),
        "roll_rate": record_threshold(signals["roll_rate"], floor=0.08, q=90.0, k_mad=2.0),
        "roll_delta": max(0.015, record_threshold(signals["roll"] - np.nanmedian(signals["roll"]), floor=0.015, q=85.0, k_mad=2.0)),
        "lat_offset_rate": record_threshold(signals["lat_offset_rate"], floor=0.5, q=92.0, k_mad=2.5),
    }
    record = {"df": df, "t": t, "cols": cols, "signals": signals, "thresholds": thresholds, "fs": fs}
    cache[key] = record
    return record


def local_baseline(t: np.ndarray, values: np.ndarray, ref_s: float, before: float = 1.5) -> float:
    mask = (t >= ref_s - before) & (t < ref_s - 0.1) & np.isfinite(values)
    if mask.sum() >= 3:
        return float(np.nanmedian(values[mask]))
    mask = (t >= ref_s - 3.0) & (t <= ref_s + 0.2) & np.isfinite(values)
    if mask.sum() >= 3:
        return float(np.nanmedian(values[mask]))
    return float(np.nanmedian(values[np.isfinite(values)])) if np.isfinite(values).any() else 0.0


def score_window(record: dict, row: pd.Series) -> dict:
    t = record["t"]
    sig = record["signals"]
    thr = record["thresholds"]
    start_old = as_float(row.get("episode_start_s"), 0.0)
    end_old = as_float(row.get("episode_end_s"), start_old + 6.0)
    evidences = finite_times(
        row,
        [
            "model_anchor_s_v1_8",
            "driver_action_onset_s",
            "vehicle_response_onset_s",
            "condition_peak_s",
            "vehicle_peak_s",
            "driver_peak_s",
        ],
    )
    if evidences:
        ref = float(np.nanmedian(evidences))
    else:
        ref = start_old
    search_start = max(float(np.nanmin(t)), min([start_old, *evidences]) - PREVIEW_BEFORE_SEC if evidences else start_old - PREVIEW_BEFORE_SEC)
    search_end = min(float(np.nanmax(t)), max([end_old, *evidences]) + PREVIEW_AFTER_SEC if evidences else end_old + PREVIEW_AFTER_SEC)
    if search_end <= search_start + 1.0:
        search_start = max(float(np.nanmin(t)), ref - PREVIEW_BEFORE_SEC)
        search_end = min(float(np.nanmax(t)), ref + PREVIEW_AFTER_SEC)

    steer_base = local_baseline(t, sig["steer_smooth"], ref)
    brake_base = local_baseline(t, sig["brake"], ref)
    roll_base = local_baseline(t, sig["roll"], ref)

    eps = 1e-6
    driver_score = np.maximum.reduce(
        [
            np.abs(sig["steer_rate"]) / (thr["steer_rate"] + eps),
            np.abs(sig["steer_smooth"] - steer_base) / (thr["steer_delta"] + eps),
            np.maximum(0.0, sig["brake"] - brake_base) / (thr["brake"] + eps),
        ]
    )
    vehicle_score = np.maximum.reduce(
        [
            np.abs(sig["ay"]) / (thr["ay"] + eps),
            np.abs(sig["yaw_rate"]) / (thr["yaw_rate"] + eps),
            np.abs(sig["roll_rate"]) / (thr["roll_rate"] + eps),
            np.abs(sig["roll"] - roll_base) / (thr["roll_delta"] + eps),
            0.45 * np.abs(sig["lat_offset_rate"]) / (thr["lat_offset_rate"] + eps),
        ]
    )
    combined = np.maximum(driver_score, vehicle_score)
    combined = v10.moving_average(combined, max(3, int(round(record["fs"] * 0.10))))
    driver_active = driver_score >= 1.0
    vehicle_active = vehicle_score >= 1.0
    combined_active = combined >= 0.95

    return {
        "ref": ref,
        "search_start": search_start,
        "search_end": search_end,
        "steer_base": steer_base,
        "brake_base": brake_base,
        "roll_base": roll_base,
        "driver_score": driver_score,
        "vehicle_score": vehicle_score,
        "combined_score": combined,
        "driver_active": driver_active,
        "vehicle_active": vehicle_active,
        "combined_active": combined_active,
    }


def first_active_time(t: np.ndarray, mask: np.ndarray, start: float, end: float, min_dur: float) -> float:
    region = np.where((t >= start) & (t <= end) & mask)[0]
    if region.size == 0:
        return math.nan
    fs_guess = 1.0 / np.nanmedian(np.diff(t[np.isfinite(t)]))
    min_n = max(1, int(round(min_dur * fs_guess)))
    groups = np.split(region, np.where(np.diff(region) > 1)[0] + 1)
    for g in groups:
        if g.size >= min_n:
            return float(t[g[0]])
    return float(t[region[0]])


def connected_start(t: np.ndarray, active: np.ndarray, peak_idx: int, quiet_gap: float) -> float:
    start_idx = peak_idx
    false_run = 0.0
    last_t = float(t[peak_idx])
    for i in range(peak_idx, -1, -1):
        dt = abs(last_t - float(t[i]))
        last_t = float(t[i])
        if active[i]:
            false_run = 0.0
            start_idx = i
        else:
            false_run += dt
            if false_run >= quiet_gap:
                break
    while start_idx < peak_idx and not active[start_idx]:
        start_idx += 1
    return float(t[start_idx])


def connected_end(t: np.ndarray, active: np.ndarray, peak_idx: int, quiet_gap: float, min_end: float, max_end: float) -> float:
    end_idx = peak_idx
    false_run = 0.0
    last_t = float(t[peak_idx])
    for i in range(peak_idx, len(t)):
        ti = float(t[i])
        if ti < min_end:
            end_idx = i
            last_t = ti
            continue
        dt = abs(ti - last_t)
        last_t = ti
        if active[i]:
            false_run = 0.0
            end_idx = i
        else:
            false_run += dt
            if false_run >= quiet_gap:
                return min(ti, max_end)
        if ti >= max_end:
            return max_end
    return min(float(t[-1]), max_end)


def refine_one(row: pd.Series, record: dict | None) -> pd.Series:
    if record is None:
        return pd.Series(
            {
                "v2_2_epoch_status": "data_missing",
                "v2_2_epoch_quality_cn": "原始车辆文件或时间列不可读取",
                "v2_2_include_boundary_training": False,
            }
        )

    t = record["t"]
    scores = score_window(record, row)
    search_mask = (t >= scores["search_start"]) & (t <= scores["search_end"])
    if search_mask.sum() < 10:
        return pd.Series(
            {
                "v2_2_epoch_status": "window_missing",
                "v2_2_epoch_quality_cn": "可用搜索窗口过短",
                "v2_2_include_boundary_training": False,
            }
        )
    core_mask = (t >= scores["ref"] - CORE_PEAK_BEFORE_SEC) & (t <= scores["ref"] + CORE_PEAK_AFTER_SEC) & search_mask
    if core_mask.sum() < 5:
        core_mask = search_mask
    score = scores["combined_score"]
    core_idx = np.where(core_mask)[0]
    if core_idx.size == 0 or not np.isfinite(score[core_idx]).any():
        peak_idx = int(np.nanargmax(score[search_mask]))
    else:
        peak_idx = int(core_idx[int(np.nanargmax(score[core_idx]))])
    peak_s = float(t[peak_idx])
    peak_score = float(score[peak_idx])

    if peak_score < 0.75:
        return pd.Series(
            {
                "v2_2_epoch_status": "low_activity_unclear",
                "v2_2_epoch_quality_cn": "局部车辆/驾驶员活动分数较弱，边界不可靠",
                "v2_2_condition_peak_s": peak_s,
                "v2_2_peak_score": peak_score,
                "v2_2_include_boundary_training": False,
            }
        )

    active = scores["combined_active"] & search_mask
    start_s = connected_start(t, active, peak_idx, QUIET_GAP_SEC)
    min_end = max(peak_s + 0.8, start_s + MIN_EVENT_DURATION_SEC)
    max_end = min(float(np.nanmax(t)), start_s + MAX_EVENT_DURATION_SEC)
    end_s = connected_end(t, active, peak_idx, QUIET_GAP_SEC, min_end=min_end, max_end=max_end)
    if end_s <= start_s:
        end_s = min(float(np.nanmax(t)), start_s + MIN_EVENT_DURATION_SEC)

    driver_on = first_active_time(t, scores["driver_active"], max(scores["search_start"], start_s - 0.4), min(peak_s + 0.5, end_s), min_dur=0.08)
    vehicle_on = first_active_time(t, scores["vehicle_active"], max(scores["search_start"], start_s - 0.4), min(peak_s + 0.8, end_s), min_dur=0.10)
    if math.isfinite(driver_on) and start_s - 0.5 <= driver_on <= end_s:
        anchor = driver_on
        anchor_source = "驾驶员动作起点"
    elif math.isfinite(vehicle_on) and start_s - 0.5 <= vehicle_on <= end_s:
        anchor = vehicle_on
        anchor_source = "车辆动态起点"
    else:
        anchor = start_s
        anchor_source = "综合活动起点"

    old_start = as_float(row.get("episode_start_s"), start_s)
    old_end = as_float(row.get("episode_end_s"), end_s)
    old_anchor = as_float(row.get("model_anchor_s_v1_8"), anchor)
    start_shift = start_s - old_start
    end_shift = end_s - old_end
    anchor_shift = anchor - old_anchor

    flags: list[str] = []
    if start_shift > 0.8:
        flags.append("old_start_too_early")
    if start_shift < -0.5:
        flags.append("old_start_too_late")
    if end_shift > 1.0:
        flags.append("old_end_too_early")
    if end_shift < -2.0:
        flags.append("old_end_too_late")
    if anchor_shift > 0.8:
        flags.append("old_anchor_too_early")
    if anchor_shift < -0.8:
        flags.append("old_anchor_too_late")
    if not flags:
        flags.append("boundary_ok")

    status = "boundary_ok" if flags == ["boundary_ok"] else "boundary_reworked"
    quality_cn = "；".join(
        {
            "old_start_too_early": "旧开始偏早，前面包含较长平稳段",
            "old_start_too_late": "旧开始偏晚，可能切掉响应前段",
            "old_end_too_early": "旧结束偏早，后续仍有车辆/驾驶员动态",
            "old_end_too_late": "旧结束偏晚，后面可能混入恢复或下一段驾驶",
            "old_anchor_too_early": "旧模型锚点偏早",
            "old_anchor_too_late": "旧模型锚点偏晚",
            "boundary_ok": "旧边界与新边界基本一致",
        }[f]
        for f in flags
    )

    return pd.Series(
        {
            "v2_2_epoch_status": status,
            "v2_2_epoch_quality_flags": ";".join(flags),
            "v2_2_epoch_quality_cn": quality_cn,
            "v2_2_episode_start_s": start_s,
            "v2_2_model_anchor_s": anchor,
            "v2_2_model_anchor_source": anchor_source,
            "v2_2_driver_action_onset_s": driver_on,
            "v2_2_vehicle_response_onset_s": vehicle_on,
            "v2_2_condition_peak_s": peak_s,
            "v2_2_episode_end_s": end_s,
            "v2_2_duration_s": end_s - start_s,
            "v2_2_peak_score": peak_score,
            "v2_2_old_start_shift_s": start_shift,
            "v2_2_old_end_shift_s": end_shift,
            "v2_2_old_anchor_shift_s": anchor_shift,
            "v2_2_obs_start_s": max(0.0, anchor - MODEL_PRE_WINDOW_SEC),
            "v2_2_obs_end_s": anchor + MODEL_EARLY_OBS_SEC,
            "v2_2_label_start_s": anchor + MODEL_EARLY_OBS_SEC,
            "v2_2_label_end_s": anchor + MODEL_EARLY_OBS_SEC + MODEL_LABEL_WINDOW_SEC,
            "v2_2_include_boundary_training": bool(row.get("v2_1_include_training_pool", False)) and (end_s - anchor >= 1.5),
        }
    )


def plot_boundary(row: pd.Series, record: dict, out_path: Path) -> None:
    t = record["t"]
    sig = record["signals"]
    start = as_float(row.get("v2_2_episode_start_s"))
    end = as_float(row.get("v2_2_episode_end_s"))
    anchor = as_float(row.get("v2_2_model_anchor_s"))
    if not all(math.isfinite(v) for v in [start, end, anchor]):
        return
    left = max(float(np.nanmin(t)), start - 2.5)
    right = min(float(np.nanmax(t)), end + 2.0)
    mask = (t >= left) & (t <= right)
    x = t[mask] - anchor
    panels = [
        ("方向盘角", sig["steer_smooth"]),
        ("方向盘角速度", sig["steer_rate"]),
        ("车速", sig["speed_kmh"]),
        ("制动踏板", sig["brake"]),
        ("横向加速度", sig["ay"]),
        ("横摆角速度", sig["yaw_rate"]),
        ("横滚角速度", sig["roll_rate"]),
        ("横滚角", sig["roll"]),
        ("高度 z", sig["z"]),
        ("横向偏移", sig["lat_offset"]),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 2.0 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    lines = [
        (as_float(row.get("episode_start_s")) - anchor, "旧开始", "tab:gray", "--"),
        (as_float(row.get("model_anchor_s_v1_8")) - anchor, "旧模型锚点", "tab:orange", "--"),
        (start - anchor, "新开始", "red", "-"),
        (0.0, "新模型锚点", "black", "-"),
        (as_float(row.get("v2_2_condition_peak_s")) - anchor, "风险峰值", "green", ":"),
        (end - anchor, "新结束", "tab:purple", "-"),
        (as_float(row.get("episode_end_s")) - anchor, "旧结束", "tab:gray", ":"),
    ]
    for ax, (label, values) in zip(axes, panels):
        ax.plot(x, values[mask], linewidth=1.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        for xpos, name, color, style in lines:
            if math.isfinite(xpos) and x.min() <= xpos <= x.max():
                ax.axvline(xpos, color=color, linestyle=style, linewidth=1.0, label=name)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)
    title = (
        f"{row.get('episode_uid')} | {row.get('v2_1_role')} | {row.get('v2_2_epoch_quality_flags')}\n"
        f"old_start_shift={as_float(row.get('v2_2_old_start_shift_s')):.2f}s, "
        f"old_end_shift={as_float(row.get('v2_2_old_end_shift_s')):.2f}s, "
        f"anchor_shift={as_float(row.get('v2_2_old_anchor_shift_s')):.2f}s"
    )
    fig.suptitle(title, fontsize=11)
    axes[-1].set_xlabel("相对新模型锚点时间 / s")
    fig.tight_layout(rect=[0, 0, 0.98, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def select_review_rows(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("00_旧开始偏早", "old_start_too_early", "v2_2_old_start_shift_s", False, 18),
        ("01_旧开始偏晚", "old_start_too_late", "v2_2_old_start_shift_s", True, 18),
        ("02_旧结束偏早", "old_end_too_early", "v2_2_old_end_shift_s", False, 18),
        ("03_旧结束偏晚", "old_end_too_late", "v2_2_old_end_shift_s", True, 18),
        ("04_旧锚点偏晚", "old_anchor_too_late", "v2_2_old_anchor_shift_s", True, 18),
        ("05_边界基本一致", "boundary_ok", "v2_2_peak_score", False, 16),
        ("06_活动弱或不清楚", "low_activity_unclear", "v2_2_peak_score", True, 16),
    ]
    rows = []
    for folder, flag, sort_col, ascending, limit in specs:
        if flag == "low_activity_unclear":
            sub = df[df["v2_2_epoch_status"].eq("low_activity_unclear")].copy()
        else:
            sub = df[df["v2_2_epoch_quality_flags"].fillna("").str.contains(flag, regex=False)].copy()
        if sort_col in sub.columns:
            sub = sub.sort_values(sort_col, ascending=ascending)
        for _, row in sub.head(limit).iterrows():
            rows.append(
                {
                    "episode_uid": row.get("episode_uid"),
                    "folder": folder,
                    "vehicle_file": row.get("vehicle_file"),
                    "v2_2_epoch_quality_flags": row.get("v2_2_epoch_quality_flags"),
                    "v2_2_epoch_quality_cn": row.get("v2_2_epoch_quality_cn"),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["episode_uid", "folder"])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def markdown_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if not df.empty else "_无记录_"


def build_or_load_refined(force_recompute: bool) -> pd.DataFrame:
    refined_path = TABLE_DIR / "record_level_episodes_all_v2_2_epoch_refined.csv"
    if refined_path.exists() and not force_recompute:
        print(f"[INFO] reuse existing refined table: {refined_path}", flush=True)
        return pd.read_csv(refined_path, low_memory=False)

    ensure_dirs()
    df = pd.read_csv(V21_ALL, low_memory=False)
    cache: dict[str, dict] = {}
    refined = []
    for i, row in df.iterrows():
        record = load_vehicle(as_text(row.get("vehicle_file")), cache)
        refined.append(refine_one(row, record))
        if (i + 1) % 250 == 0:
            print(f"[INFO] refined {i + 1}/{len(df)}", flush=True)
    out = pd.concat([df, pd.DataFrame(refined)], axis=1)

    write_csv(out, refined_path)
    write_csv(
        out[out["v2_2_include_boundary_training"].map(as_bool)],
        TABLE_DIR / "training_pool_epoch_refined_v2_2.csv",
    )
    write_csv(
        out[out["v2_2_epoch_quality_flags"].fillna("").str.contains("old_start_too_early|old_start_too_late|old_end_too_early|old_end_too_late|old_anchor_too_late|old_anchor_too_early", regex=True)],
        TABLE_DIR / "epoch_boundary_rework_needed_v2_2.csv",
    )

    status_summary = out.groupby("v2_2_epoch_status", dropna=False).size().reset_index(name="count")
    flag_rows = []
    for flags in out["v2_2_epoch_quality_flags"].fillna(""):
        for flag in str(flags).split(";"):
            if flag:
                flag_rows.append(flag)
    flag_summary = pd.Series(flag_rows, dtype=str).value_counts().rename_axis("flag").reset_index(name="count")
    role_summary = out.groupby(["v2_1_role", "v2_2_epoch_status"], dropna=False).size().reset_index(name="count")
    split_summary = out.groupby(["split", "v2_2_epoch_status"], dropna=False).size().reset_index(name="count") if "split" in out.columns else pd.DataFrame()
    shift_rows = []
    for col, label in [
        ("v2_2_old_start_shift_s", "新开始 - 旧开始"),
        ("v2_2_old_end_shift_s", "新结束 - 旧结束"),
        ("v2_2_old_anchor_shift_s", "新锚点 - 旧锚点"),
        ("v2_2_duration_s", "新episode时长"),
    ]:
        s = pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series(dtype=float)
        shift_rows.append(
            {
                "metric": label,
                "mean": float(s.mean()) if len(s) else math.nan,
                "median": float(s.median()) if len(s) else math.nan,
                "p10": float(s.quantile(0.10)) if len(s) else math.nan,
                "p90": float(s.quantile(0.90)) if len(s) else math.nan,
            }
        )
    shift_summary = pd.DataFrame(shift_rows)
    write_csv(status_summary, TABLE_DIR / "v2_2_epoch_status_summary.csv")
    write_csv(flag_summary, TABLE_DIR / "v2_2_epoch_flag_summary.csv")
    write_csv(role_summary, TABLE_DIR / "v2_2_role_epoch_summary.csv")
    write_csv(shift_summary, TABLE_DIR / "v2_2_shift_summary.csv")
    if not split_summary.empty:
        write_csv(split_summary, TABLE_DIR / "v2_2_split_epoch_summary.csv")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-recompute", action="store_true", help="Recompute v2.2 refined boundaries from raw CSV.")
    parser.add_argument("--skip-figures", action="store_true", help="Only write tables and report.")
    args = parser.parse_args()

    ensure_dirs()
    out = build_or_load_refined(force_recompute=args.force_recompute)

    status_summary = out.groupby("v2_2_epoch_status", dropna=False).size().reset_index(name="count")
    flag_rows = []
    for flags in out["v2_2_epoch_quality_flags"].fillna(""):
        for flag in str(flags).split(";"):
            if flag:
                flag_rows.append(flag)
    flag_summary = pd.Series(flag_rows, dtype=str).value_counts().rename_axis("flag").reset_index(name="count")
    role_summary = out.groupby(["v2_1_role", "v2_2_epoch_status"], dropna=False).size().reset_index(name="count")
    split_summary = out.groupby(["split", "v2_2_epoch_status"], dropna=False).size().reset_index(name="count") if "split" in out.columns else pd.DataFrame()
    shift_rows = []
    for col, label in [
        ("v2_2_old_start_shift_s", "新开始 - 旧开始"),
        ("v2_2_old_end_shift_s", "新结束 - 旧结束"),
        ("v2_2_old_anchor_shift_s", "新锚点 - 旧锚点"),
        ("v2_2_duration_s", "新episode时长"),
    ]:
        s = pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series(dtype=float)
        shift_rows.append(
            {
                "metric": label,
                "mean": float(s.mean()) if len(s) else math.nan,
                "median": float(s.median()) if len(s) else math.nan,
                "p10": float(s.quantile(0.10)) if len(s) else math.nan,
                "p90": float(s.quantile(0.90)) if len(s) else math.nan,
            }
        )
    shift_summary = pd.DataFrame(shift_rows)
    write_csv(status_summary, TABLE_DIR / "v2_2_epoch_status_summary.csv")
    write_csv(flag_summary, TABLE_DIR / "v2_2_epoch_flag_summary.csv")
    write_csv(role_summary, TABLE_DIR / "v2_2_role_epoch_summary.csv")
    write_csv(shift_summary, TABLE_DIR / "v2_2_shift_summary.csv")
    if not split_summary.empty:
        write_csv(split_summary, TABLE_DIR / "v2_2_split_epoch_summary.csv")

    figure_index_path = TABLE_DIR / "epoch_boundary_review_figure_index_v2_2.csv"
    if args.skip_figures and figure_index_path.exists():
        figure_index = pd.read_csv(figure_index_path, low_memory=False)
    else:
        review_index = select_review_rows(out) if not args.skip_figures else pd.DataFrame()
        figure_rows = []
        cache: dict[str, dict] = {}
        for _, r in review_index.iterrows():
            row = out[out["episode_uid"].astype(str).eq(str(r["episode_uid"]))].iloc[0]
            record = load_vehicle(as_text(row.get("vehicle_file")), cache)
            if record is None:
                continue
            folder = as_text(r["folder"])
            out_path = FIG_DIR / folder / f"{row.get('episode_uid')}.png"
            if not out_path.exists():
                plot_boundary(row, record, out_path)
            rr = r.to_dict()
            rr["figure_path"] = str(out_path)
            figure_rows.append(rr)
            if len(figure_rows) % 20 == 0:
                print(f"[INFO] plotted {len(figure_rows)}/{len(review_index)}", flush=True)
        figure_index = pd.DataFrame(figure_rows)
        write_csv(figure_index, figure_index_path)

    total = len(out)
    train_pool = int(out["v2_2_include_boundary_training"].map(as_bool).sum())
    rework = int(out["v2_2_epoch_status"].eq("boundary_reworked").sum())
    low = int(out["v2_2_epoch_status"].eq("low_activity_unclear").sum())
    good = int(out["v2_2_epoch_status"].eq("boundary_ok").sum())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""# v2.2 epoch 边界精修审计

生成时间：{now}

## 这版解决什么

v2.1 解决的是“样本是否被异常规则误删”；v2.2 解决的是“每个 epoch 的开始、结束和模型锚点是否切得合适”。

本轮把三个概念分开：

1. **完整事件段**：从驾驶员或车辆持续活动开始，到主要响应后稳定下来。
2. **模型锚点**：后续模型真正对齐的 `t0`，优先用驾驶员动作起点，其次用车辆动态起点。
3. **建模窗口**：`t0` 前 2 秒作为历史，`t0` 后 0.5 秒可作为早期观察，之后 5 秒作为预测标签。

## 总体结果

| 项目 | 数量 |
|---|---:|
| 全部 episode | {total} |
| v2.2 边界基本一致 | {good} |
| v2.2 需要重划边界 | {rework} |
| 活动弱或边界不清楚 | {low} |
| v2.2 可进入边界训练池 | {train_pool} |
| 复核图数量 | {len(figure_index)} |

## 状态统计

{markdown_table(status_summary)}

## 边界问题统计

{markdown_table(flag_summary)}

## 边界偏移幅度统计

正数表示 v2.2 比旧版本更晚，负数表示 v2.2 比旧版本更早。

{markdown_table(shift_summary.round(3))}

## v2.1 角色与 v2.2 边界状态

{markdown_table(role_summary)}

## 输出文件

- 全量表：`{TABLE_DIR / "record_level_episodes_all_v2_2_epoch_refined.csv"}`
- v2.2 训练池：`{TABLE_DIR / "training_pool_epoch_refined_v2_2.csv"}`
- 需要重划边界表：`{TABLE_DIR / "epoch_boundary_rework_needed_v2_2.csv"}`
- 复核图索引：`{TABLE_DIR / "epoch_boundary_review_figure_index_v2_2.csv"}`
- 复核图目录：`{FIG_DIR}`

## 当前建议

- 后续训练不要再直接使用旧 `episode_start_s` 或旧 `episode_end_s`。
- 训练输入应优先使用 `v2_2_model_anchor_s`、`v2_2_obs_start_s`、`v2_2_obs_end_s`、`v2_2_label_start_s`、`v2_2_label_end_s`。
- 人工复核优先看 `00_旧开始偏早`、`02_旧结束偏早`、`03_旧结束偏晚` 和 `04_旧锚点偏晚` 四类图。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    (OUT_ROOT / "record_episode_dataset_v2_2_epoch_summary_cn.md").write_text(report, encoding="utf-8")
    print(f"[OK] wrote {TABLE_DIR}")
    print(f"[OK] wrote {REPORT_PATH}")
    print(f"[SUMMARY] total={total} good={good} rework={rework} low={low} train_pool={train_pool} figures={len(figure_index)}")


if __name__ == "__main__":
    main()
