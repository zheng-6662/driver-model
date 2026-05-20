# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
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


def find_project_root() -> Path:
    env_root = os.environ.get("DATA_PROCESS_ROOT")
    if env_root:
        p = Path(env_root)
        if p.exists():
            return p
    here = Path(__file__).resolve()
    for p in [Path.cwd(), here, *here.parents, *Path.cwd().parents]:
        if (p / "05_rebuild_from_raw_20260511").exists() and (p / "01_datasets").exists():
            return p
    return Path(r"F:/data_set_process/data_process")


PROJECT_ROOT = find_project_root()
REBUILD_ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
DEFAULT_CONFIG = REBUILD_ROOT / "02_samples" / "configs" / "record_episode_reconstruction_v1_0.json"


COL_ALIASES = {
    "time": ["StorageTime", "time", "timestamp", "Time"],
    "steer": ["zx|SteeringWheel", "SteeringWheel", "steering_wheel", "steer"],
    "ay": ["zx|ay", "ay", "lateral_acceleration"],
    "yaw_rate": ["zx|vyaw", "vyaw", "yaw_rate"],
    "roll_rate": ["zx|vroll", "vroll", "roll_rate"],
    "roll": ["zx|roll", "roll", "roll_angle"],
    "lat_offset": ["zx1|lateraldistance", "zx|lateraldistance", "lateraldistance", "lane_offset"],
    "brake": ["zx|BrakePedal", "BrakePedal", "brake"],
    "accel_pedal": ["zx|AcceleratorPedal", "AcceleratorPedal"],
    "ax": ["zx|ax", "ax", "longitudinal_acceleration"],
    "speed_kmh": ["zx1|v_km/h", "v_km/h", "speed_kmh"],
    "vx": ["zx|vx", "vx"],
    "mu": ["zx1|mu", "mu"],
    "curvature": ["zx1|lanecurvatureXY", "zx|lanecurvatureXY", "lanecurvatureXY", "curvature"],
    "x": ["zx|x", "x", "pos_x"],
    "y": ["zx|y", "y", "pos_y"],
}


@dataclass
class RecordSignals:
    path: Path
    subject: str
    session_stamp: str
    df: pd.DataFrame
    t: np.ndarray
    fs: float
    cols: dict[str, str | None]
    signals: dict[str, np.ndarray]
    thresholds: dict[str, float]
    scores: dict[str, np.ndarray]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["raw_vehicle_root"] = str((PROJECT_ROOT / cfg["raw_vehicle_root"]).resolve())
    cfg["output_dir"] = str((PROJECT_ROOT / cfg["output_dir"]).resolve())
    cfg["road_context_table"] = str((PROJECT_ROOT / cfg["road_context_table"]).resolve())
    return cfg


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "out": out_dir,
        "tables": out_dir / "tables",
        "figures": out_dir / "figures",
        "review": out_dir / "figures" / "review_panels",
        "trajectory3d": out_dir / "figures" / "trajectory_3d_static",
        "reports": REBUILD_ROOT / "09_reports",
        "logs": out_dir / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


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


def parse_subject_session(path: Path) -> tuple[str, str]:
    subject = path.parent.name
    m = re.search(r"Entity_Recording_(.+?)_vehicle", path.name)
    session = m.group(1) if m else path.stem
    return subject, session


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


def finite_series(df: pd.DataFrame, col: str | None, default: float = 0.0) -> np.ndarray:
    if not col or col not in df.columns:
        return np.full(len(df), default, dtype=float)
    out = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    idx = np.arange(out.size)
    valid = np.isfinite(out)
    if valid.sum() == 0:
        return np.full(len(df), default, dtype=float)
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


def robust_threshold(values: np.ndarray, floor: float, q: float = 95.0, k_mad: float = 3.0) -> float:
    arr = np.abs(np.asarray(values, dtype=float))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(floor)
    med = float(np.nanmedian(arr))
    mad = robust_mad(arr)
    qv = float(np.nanpercentile(arr, q))
    return float(np.nanmax([floor, qv, med + k_mad * mad if math.isfinite(mad) else floor]))


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or width <= 1:
        return values.copy()
    idx = np.arange(values.size)
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros_like(values)
    filled = values.copy()
    filled[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(filled, kernel, mode="same")


def gradient(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & np.isfinite(t)
    if valid.sum() < 3:
        return out
    order = np.argsort(t[valid])
    tv = t[valid][order]
    vv = values[valid][order]
    keep = np.r_[True, np.diff(tv) > 1e-6]
    tv = tv[keep]
    vv = vv[keep]
    if tv.size < 3:
        return np.zeros_like(values, dtype=float)
    deriv = np.gradient(vv, tv)
    out = np.interp(t, tv, deriv, left=deriv[0], right=deriv[-1])
    out[~np.isfinite(out)] = 0.0
    return out


def safe_max_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.nanmax(np.abs(arr)))


def safe_range(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.nanmax(arr) - np.nanmin(arr))


def normalize_score(values: np.ndarray, threshold: float, cap: float = 4.0) -> np.ndarray:
    if not math.isfinite(threshold) or threshold <= 1e-9:
        return np.zeros_like(values, dtype=float)
    score = np.abs(values) / threshold
    score[~np.isfinite(score)] = 0.0
    return np.clip(score, 0.0, cap)


def load_record(path: Path, cfg: dict[str, Any]) -> RecordSignals:
    df = read_csv_smart(path)
    subject, session = parse_subject_session(path)
    cols = {k: pick_col(df, k) for k in COL_ALIASES}
    if cols["time"] is None or cols["steer"] is None:
        raise ValueError(f"missing required time/steer column: {path}")
    t = parse_time_seconds(df[cols["time"]])
    finite_t = t[np.isfinite(t)]
    if finite_t.size < 10:
        raise ValueError(f"invalid time column: {path}")
    dt = np.diff(finite_t)
    dt = dt[np.isfinite(dt) & (dt > 1e-5)]
    fs = float(1.0 / np.nanmedian(dt)) if dt.size else 200.0
    width = max(1, int(round(float(cfg["smooth_sec"]) * fs)))
    if width % 2 == 0:
        width += 1

    signals: dict[str, np.ndarray] = {}
    signals["steer"] = moving_average(finite_series(df, cols["steer"]), width)
    signals["steer_rate"] = moving_average(gradient(signals["steer"], t), width)
    signals["ay"] = moving_average(finite_series(df, cols["ay"]), width)
    signals["yaw_rate"] = moving_average(finite_series(df, cols["yaw_rate"]), width)
    signals["roll_rate"] = moving_average(finite_series(df, cols["roll_rate"]), width)
    signals["roll"] = moving_average(finite_series(df, cols["roll"]), width)
    signals["lat_offset"] = moving_average(finite_series(df, cols["lat_offset"]), width)
    signals["lat_rate"] = moving_average(gradient(signals["lat_offset"], t), width)
    signals["brake"] = moving_average(finite_series(df, cols["brake"]), width)
    speed = finite_series(df, cols["speed_kmh"])
    if cols["speed_kmh"] is None and cols["vx"] is not None:
        speed = finite_series(df, cols["vx"]) * 3.6
    signals["speed_kmh"] = moving_average(speed, width)
    signals["speed_rate"] = moving_average(gradient(signals["speed_kmh"], t), width)
    signals["mu"] = finite_series(df, cols["mu"], default=1.0)
    signals["curvature"] = moving_average(finite_series(df, cols["curvature"]), width)
    signals["x"] = finite_series(df, cols["x"])
    signals["y"] = finite_series(df, cols["y"])

    th_cfg = cfg["signal_thresholds"]
    thresholds = {
        "steer_rate": robust_threshold(signals["steer_rate"], th_cfg["steer_rate_floor"], 95.0),
        "ay": robust_threshold(signals["ay"], th_cfg["ay_floor"], 92.0),
        "yaw_rate": robust_threshold(signals["yaw_rate"], th_cfg["yaw_rate_floor"], 92.0),
        "roll_rate": robust_threshold(signals["roll_rate"], th_cfg["roll_rate_floor"], 92.0),
        "roll": robust_threshold(signals["roll"], th_cfg["roll_angle_floor"], 92.0),
        "lat_rate": robust_threshold(signals["lat_rate"], th_cfg["lateral_rate_floor"], 95.0),
        "speed_rate": robust_threshold(signals["speed_rate"], th_cfg["speed_change_floor"], 95.0),
        "brake": robust_threshold(signals["brake"], th_cfg["brake_floor"], 90.0),
        "curvature": robust_threshold(signals["curvature"], th_cfg["curve_floor"], 90.0),
    }

    scores: dict[str, np.ndarray] = {}
    for key in ["steer_rate", "ay", "yaw_rate", "roll_rate", "roll", "lat_rate", "speed_rate", "brake", "curvature"]:
        scores[key] = normalize_score(signals[key], thresholds[key])
    low_mu = (signals["mu"] < float(th_cfg["low_mu_threshold"])).astype(float)
    scores["low_mu"] = low_mu
    vehicle_stack = np.vstack([scores["ay"], scores["yaw_rate"], scores["roll_rate"], scores["roll"], scores["lat_rate"]])
    scores["vehicle_core"] = np.nanmax(vehicle_stack, axis=0)
    scores["vehicle_sum"] = np.nansum(np.clip(vehicle_stack, 0.0, 2.0), axis=0)
    scores["driver"] = np.nanmax(np.vstack([scores["steer_rate"], 0.8 * scores["brake"]]), axis=0)
    scores["context"] = np.nanmax(np.vstack([0.35 * scores["low_mu"], 0.25 * scores["curvature"]]), axis=0)
    scores["condition"] = np.nanmax(
        np.vstack([scores["vehicle_core"], 0.75 * scores["driver"], scores["context"] * np.maximum(scores["vehicle_core"], scores["driver"])]),
        axis=0,
    )
    return RecordSignals(path, subject, session, df, t, fs, cols, signals, thresholds, scores)


def contiguous_true(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    starts: list[int] = []
    ends: list[int] = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            if i - start >= min_len:
                starts.append(start)
                ends.append(i - 1)
            in_run = False
    if in_run and mask.size - start >= min_len:
        starts.append(start)
        ends.append(mask.size - 1)
    return list(zip(starts, ends))


def merge_intervals(intervals: list[tuple[int, int]], t: np.ndarray, gap_s: float) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if float(t[s] - t[last_e]) <= gap_s:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def expand_interval(active: np.ndarray, idx: int, t: np.ndarray, quiet_thr: float, max_duration_s: float) -> tuple[int, int]:
    n = len(active)
    s = idx
    while s > 0 and active[s] > quiet_thr and float(t[idx] - t[s]) < max_duration_s / 2:
        s -= 1
    e = idx
    while e < n - 1 and active[e] > quiet_thr and float(t[e] - t[idx]) < max_duration_s / 2:
        e += 1
    return s, e


def find_candidate_intervals(rs: RecordSignals, cfg: dict[str, Any]) -> list[tuple[int, int]]:
    score = rs.scores["condition"]
    t = rs.t
    finite = np.isfinite(score) & np.isfinite(t)
    if finite.sum() < 10:
        return []
    q = float(np.nanpercentile(score[finite], float(cfg["candidate_peak_percentile"])))
    peak_thr = max(float(cfg["candidate_peak_floor"]), q)
    candidate_idx = np.flatnonzero(score >= peak_thr)
    if candidate_idx.size == 0:
        return []
    order = candidate_idx[np.argsort(score[candidate_idx])[::-1]]
    picked: list[int] = []
    min_gap = float(cfg["episode_min_gap_sec"])
    for idx in order:
        if all(abs(float(t[idx] - t[p])) >= min_gap for p in picked):
            picked.append(int(idx))
    intervals = [
        expand_interval(score, idx, t, float(cfg["quiet_score_threshold"]), float(cfg["episode_max_duration_sec"]))
        for idx in sorted(picked)
    ]
    intervals = merge_intervals(intervals, t, float(cfg["episode_merge_gap_sec"]))
    min_dur = float(cfg["episode_min_duration_sec"])
    return [(s, e) for s, e in intervals if float(t[e] - t[s]) >= min_dur]


def first_crossing(score: np.ndarray, s: int, e: int, thr: float) -> int | None:
    idx = np.flatnonzero(score[s : e + 1] >= thr)
    return int(s + idx[0]) if idx.size else None


def classify_episode(row: dict[str, Any]) -> tuple[str, str, str, str]:
    risk = str(row["vehicle_risk_level"])
    driver = str(row["driver_response_type"])
    quality = str(row["anchor_quality"])
    if quality in {"窗口截断", "坐标风险高"}:
        return "需要复核", "窗口/坐标风险", "人工复核后再决定", "review"
    if risk in {"强", "极强"} and "方向盘" in driver:
        return "核心极限样本", "车辆动态强且驾驶员转向明显", "多输出轨迹预测", "core_extreme"
    if risk in {"强", "极强"} and ("制动" in driver or "弱" in driver or "无明显" in driver):
        return "保守/弱操作极限样本", "车辆动态强但驾驶员转向不强", "车辆状态预测+驾驶员类型分类", "conservative_extreme"
    if risk == "中" and "方向盘" in driver:
        return "次级训练样本", "车辆动态中等且驾驶员有操作", "方向盘/车辆联合预测", "secondary"
    if row.get("is_curve_context") and risk in {"弱", "中"}:
        return "正常弯道或普通操控", "更像正常过弯/普通控制", "对照或分类", "normal_or_curve"
    return "边界复核样本", "强度或语义边界不清", "人工复核", "review"


def road_context_for_interval(road_df: pd.DataFrame, subject: str, session: str, start_s: float, end_s: float) -> dict[str, Any]:
    if road_df.empty:
        return {
            "road_module_names": "",
            "road_design_categories": "",
            "nearest_road_candidate_s": np.nan,
            "nearest_road_candidate_delta_s": np.nan,
        }
    sub = road_df[(road_df["subject"].astype(str) == subject) & (road_df["session_stamp"].astype(str) == session)]
    if sub.empty:
        return {
            "road_module_names": "",
            "road_design_categories": "",
            "nearest_road_candidate_s": np.nan,
            "nearest_road_candidate_delta_s": np.nan,
        }
    cand_t = pd.to_numeric(sub.get("candidate_time_rel_s"), errors="coerce")
    in_interval = sub[(cand_t >= start_s - 2.0) & (cand_t <= end_s + 2.0)]
    mid = 0.5 * (start_s + end_s)
    delta = (cand_t - mid).abs()
    nearest_i = delta.idxmin() if delta.notna().any() else None
    nearest_t = float(cand_t.loc[nearest_i]) if nearest_i is not None and math.isfinite(float(cand_t.loc[nearest_i])) else np.nan
    modules = sorted(set(in_interval.get("module_name", pd.Series(dtype=str)).dropna().astype(str).tolist()))
    cats = sorted(set(in_interval.get("design_category_cn", pd.Series(dtype=str)).dropna().astype(str).tolist()))
    return {
        "road_module_names": "|".join(modules),
        "road_design_categories": "|".join(cats),
        "nearest_road_candidate_s": nearest_t,
        "nearest_road_candidate_delta_s": float(nearest_t - mid) if math.isfinite(nearest_t) else np.nan,
    }


def summarize_episode(rs: RecordSignals, s: int, e: int, episode_idx: int, road_df: pd.DataFrame) -> dict[str, Any]:
    t = rs.t
    seg = slice(s, e + 1)
    vehicle_score = rs.scores["vehicle_core"][seg]
    driver_score = rs.scores["driver"][seg]
    condition_score = rs.scores["condition"][seg]
    local_idx = np.arange(s, e + 1)
    vehicle_peak_i = int(local_idx[int(np.nanargmax(vehicle_score))])
    driver_peak_i = int(local_idx[int(np.nanargmax(driver_score))])
    condition_peak_i = int(local_idx[int(np.nanargmax(condition_score))])
    vehicle_on_i = first_crossing(rs.scores["vehicle_core"], s, e, 1.0)
    driver_on_i = first_crossing(rs.scores["driver"], s, e, 1.0)
    start_s = float(t[s])
    end_s = float(t[e])
    duration = end_s - start_s
    component_peaks = {
        "ay": safe_max_abs(rs.scores["ay"][seg]),
        "yaw": safe_max_abs(rs.scores["yaw_rate"][seg]),
        "roll_rate": safe_max_abs(rs.scores["roll_rate"][seg]),
        "roll": safe_max_abs(rs.scores["roll"][seg]),
        "lat_rate": safe_max_abs(rs.scores["lat_rate"][seg]),
    }
    component_count = int(sum(v >= 1.0 for v in component_peaks.values()))
    max_vehicle = float(np.nanmax(vehicle_score))
    if max_vehicle >= 3.0 or component_count >= 4:
        risk = "极强"
    elif max_vehicle >= 2.0 or component_count >= 2:
        risk = "强"
    elif max_vehicle >= 1.0 or component_count >= 1:
        risk = "中"
    else:
        risk = "弱"

    steer_peak = safe_max_abs(rs.scores["steer_rate"][seg])
    steer_delta = safe_range(rs.signals["steer"][seg])
    brake_delta = safe_range(rs.signals["brake"][seg])
    if steer_peak >= 1.3 and steer_delta >= 0.25:
        driver_type = "方向盘快速操作"
    elif brake_delta >= max(0.08, 0.8 * rs.thresholds["brake"]):
        driver_type = "制动为主"
    elif np.nanmax(driver_score) >= 0.8:
        driver_type = "弱操作"
    else:
        driver_type = "无明显操作"

    if vehicle_on_i is not None and driver_on_i is not None:
        response_order = "驾驶员先动" if t[driver_on_i] <= t[vehicle_on_i] else "车辆先变化"
        response_delay_s = float(t[driver_on_i] - t[vehicle_on_i])
    else:
        response_order = "缺少一方起点"
        response_delay_s = np.nan

    if start_s <= 2.0 or end_s >= float(np.nanmax(t)) - 1.0:
        quality = "窗口截断"
    elif duration < 1.0:
        quality = "事件很短"
    elif component_peaks["lat_rate"] >= 2.0 and component_count <= 1:
        quality = "坐标风险高"
    elif abs(float(t[condition_peak_i] - start_s)) < 0.25 and start_s > 2.0:
        quality = "可能切在事件中段"
    else:
        quality = "正常"

    low_mu = bool(np.nanmin(rs.signals["mu"][seg]) < 0.95)
    curve = bool(safe_max_abs(rs.scores["curvature"][seg]) >= 1.0)
    roll_context = bool(max(component_peaks["roll"], component_peaks["roll_rate"]) >= 1.0)
    lateral_context = bool(max(component_peaks["ay"], component_peaks["yaw"], component_peaks["lat_rate"]) >= 1.0)

    row = {
        "episode_uid": f"rec_v1_{rs.subject}_{rs.session_stamp}_{episode_idx:04d}",
        "subject": rs.subject,
        "session_stamp": rs.session_stamp,
        "vehicle_file": str(rs.path),
        "episode_index_in_record": episode_idx,
        "episode_start_s": start_s,
        "episode_end_s": end_s,
        "episode_duration_s": duration,
        "condition_peak_s": float(t[condition_peak_i]),
        "vehicle_response_onset_s": float(t[vehicle_on_i]) if vehicle_on_i is not None else np.nan,
        "driver_action_onset_s": float(t[driver_on_i]) if driver_on_i is not None else np.nan,
        "vehicle_peak_s": float(t[vehicle_peak_i]),
        "driver_peak_s": float(t[driver_peak_i]),
        "response_order": response_order,
        "driver_minus_vehicle_onset_s": response_delay_s,
        "vehicle_risk_level": risk,
        "vehicle_component_count": component_count,
        "vehicle_score_peak": max_vehicle,
        "condition_score_peak": float(np.nanmax(condition_score)),
        "driver_score_peak": float(np.nanmax(driver_score)),
        "driver_response_type": driver_type,
        "anchor_quality": quality,
        "is_low_mu_context": low_mu,
        "is_curve_context": curve,
        "is_roll_context": roll_context,
        "is_lateral_dynamic_context": lateral_context,
        "min_mu": float(np.nanmin(rs.signals["mu"][seg])),
        "median_speed_kmh": float(np.nanmedian(rs.signals["speed_kmh"][seg])),
        "peak_abs_ay": safe_max_abs(rs.signals["ay"][seg]),
        "peak_abs_yaw_rate": safe_max_abs(rs.signals["yaw_rate"][seg]),
        "peak_abs_roll_rate": safe_max_abs(rs.signals["roll_rate"][seg]),
        "peak_abs_roll": safe_max_abs(rs.signals["roll"][seg]),
        "steer_angle_range": steer_delta,
        "steer_rate_peak": safe_max_abs(rs.signals["steer_rate"][seg]),
        "brake_range": brake_delta,
        "speed_range_kmh": safe_range(rs.signals["speed_kmh"][seg]),
        "lat_offset_range": safe_range(rs.signals["lat_offset"][seg]),
        "recommended_model_window_start_s": max(0.0, start_s - 2.0),
        "recommended_model_window_end_s": min(float(np.nanmax(t)), end_s + 2.0),
    }
    row.update(road_context_for_interval(road_df, rs.subject, rs.session_stamp, start_s, end_s))
    group_cn, reason_cn, task_cn, group_id = classify_episode(row)
    row["episode_group_id"] = group_id
    row["episode_group_cn"] = group_cn
    row["episode_reason_cn"] = reason_cn
    row["recommended_task_cn"] = task_cn
    return row


def plot_episode(rs: RecordSignals, row: dict[str, Any], out_path: Path) -> None:
    start = float(row["episode_start_s"])
    end = float(row["episode_end_s"])
    lo = max(0.0, start - 4.0)
    hi = min(float(np.nanmax(rs.t)), end + 4.0)
    mask = (rs.t >= lo) & (rs.t <= hi)
    x = rs.t[mask] - start
    if x.size < 5:
        return
    step = max(1, int(x.size / 2500))
    x = x[::step]
    m = np.flatnonzero(mask)[::step]
    fig, axes = plt.subplots(9, 1, figsize=(16, 13), sharex=True)
    fig.suptitle(
        f"{row['episode_uid']} | {row['episode_group_cn']} | {row['vehicle_risk_level']} | {row['driver_response_type']}",
        fontsize=14,
    )
    series = [
        ("方向盘角", rs.signals["steer"][m]),
        ("方向盘角速度", rs.signals["steer_rate"][m]),
        ("车速 km/h", rs.signals["speed_kmh"][m]),
        ("制动踏板", rs.signals["brake"][m]),
        ("横向加速度", rs.signals["ay"][m]),
        ("横摆角速度", rs.signals["yaw_rate"][m]),
        ("横滚角速度", rs.signals["roll_rate"][m]),
        ("横滚角", rs.signals["roll"][m]),
        ("综合强度分数", rs.scores["condition"][m]),
    ]
    line_defs = [
        ("事件开始", start, "black", "-"),
        ("驾驶员操作开始", row.get("driver_action_onset_s"), "#d62728", "--"),
        ("车辆响应开始", row.get("vehicle_response_onset_s"), "#1f77b4", "--"),
        ("车辆峰值", row.get("vehicle_peak_s"), "#2ca02c", ":"),
        ("事件结束", end, "black", "-"),
    ]
    for ax, (label, y) in zip(axes, series):
        ax.plot(x, y, color="#2878b5", linewidth=1.2)
        ax.axvspan(0, end - start, color="#fff1b8", alpha=0.30)
        for name, tx, color, style in line_defs:
            try:
                txf = float(tx)
            except Exception:
                continue
            if math.isfinite(txf) and lo <= txf <= hi:
                ax.axvline(txf - start, color=color, linestyle=style, linewidth=1.0, label=name)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=9)
    axes[-1].set_xlabel("相对 episode 开始时间 / s")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_trajectory_3d(rs: RecordSignals, row: dict[str, Any], out_path: Path) -> None:
    if rs.cols.get("x") is None or rs.cols.get("y") is None:
        return
    start = float(row["episode_start_s"])
    end = float(row["episode_end_s"])
    lo = max(0.0, start - 2.0)
    hi = min(float(np.nanmax(rs.t)), end + 2.0)
    mask = (rs.t >= lo) & (rs.t <= hi)
    idx = np.flatnonzero(mask)
    if idx.size < 10:
        return
    step = max(1, int(idx.size / 1200))
    idx = idx[::step]
    x = rs.signals["x"][idx]
    y = rs.signals["y"][idx]
    z = rs.signals["roll"][idx]
    if not (np.isfinite(x).any() and np.isfinite(y).any()):
        return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    c = rs.t[idx] - start
    ax.plot(x, y, z, color="#2878b5", linewidth=1.2)
    sc = ax.scatter(x, y, z, c=c, cmap="viridis", s=5)
    ax.set_title(f"{row['episode_uid']} 3D 轨迹示意：x-y-横滚角")
    ax.set_xlabel("车辆 x")
    ax.set_ylabel("车辆 y")
    ax.set_zlabel("横滚角")
    fig.colorbar(sc, ax=ax, shrink=0.65, label="相对 episode 开始时间 / s")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_road_table(cfg: dict[str, Any]) -> pd.DataFrame:
    path = Path(cfg["road_context_table"])
    if not path.exists():
        return pd.DataFrame()
    keep = [
        "subject",
        "session_stamp",
        "module_name",
        "design_category_cn",
        "candidate_anchor_type_cn",
        "candidate_time_rel_s",
        "segment_entry_time_rel_s",
        "segment_exit_time_rel_s",
    ]
    df = read_csv_smart(path)
    cols = [c for c in keep if c in df.columns]
    return df[cols].copy()


def build_report(all_df: pd.DataFrame, inventory: pd.DataFrame, dirs: dict[str, Path], cfg: dict[str, Any]) -> None:
    summary = all_df["episode_group_cn"].value_counts(dropna=False).reset_index()
    summary.columns = ["episode_group_cn", "count"]
    context_cols = ["is_low_mu_context", "is_curve_context", "is_roll_context", "is_lateral_dynamic_context"]
    context_rows = []
    for col in context_cols:
        if col in all_df.columns:
            context_rows.append({"context": col, "count": int(all_df[col].fillna(False).astype(bool).sum())})
    context_df = pd.DataFrame(context_rows)
    summary.to_csv(dirs["tables"] / "record_episode_group_summary_v1_0.csv", index=False, encoding="utf-8-sig")
    context_df.to_csv(dirs["tables"] / "record_episode_context_summary_v1_0.csv", index=False, encoding="utf-8-sig")
    by_subject = pd.crosstab(all_df["subject"], all_df["episode_group_cn"], dropna=False).reset_index()
    by_subject.to_csv(dirs["tables"] / "record_episode_by_subject_v1_0.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 完整记录级 episode 重建 v1.0 用户说明",
        "",
        f"生成时间：{now_text()}",
        "",
        "## 这一步做了什么",
        "",
        "本流程不训练模型，而是从完整一次实验车辆 CSV 中自动切出多个驾驶 episode。每个 episode 同时记录车辆状态、驾驶员操作、道路/场景上下文和锚点质量。",
        "",
        "## 当前运行结果",
        "",
        f"- 扫描车辆记录数：{len(inventory)}",
        f"- 成功处理记录数：{int((inventory['status'] == 'ok').sum()) if not inventory.empty else 0}",
        f"- 检测到 episode 总数：{len(all_df)}",
        "",
        "### episode 分组",
        "",
        summary.to_markdown(index=False),
        "",
        "### 上下文覆盖",
        "",
        context_df.to_markdown(index=False) if not context_df.empty else "暂无上下文统计。",
        "",
        "## 输出位置",
        "",
        f"- 总表：`{dirs['tables'] / 'record_level_episodes_all_v1_0.csv'}`",
        f"- 分组统计：`{dirs['tables'] / 'record_episode_group_summary_v1_0.csv'}`",
        f"- 复核图目录：`{dirs['review']}`",
        f"- 3D 静态轨迹目录：`{dirs['trajectory3d']}`",
        "",
        "## 解释边界",
        "",
        "- 道路信息只作为工况上下文，不直接当作最终事件锚点。",
        "- 当前是自动初筛，后续需要人工查看复核图后再定最终训练样本。",
        "- 当前系统已经允许一条完整实验记录产生多个 episode，不再默认一条记录只有一个事件。",
    ]
    report = "\n".join(lines)
    (dirs["out"] / "record_episode_summary_v1_0.md").write_text(report, encoding="utf-8")
    (REBUILD_ROOT / "09_reports" / "stage02_record_episode_reconstruction_v1_0_user_summary_cn.md").write_text(
        report, encoding="utf-8"
    )


def append_note(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + text.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-3d", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dirs = ensure_dirs(Path(cfg["output_dir"]))
    road_df = load_road_table(cfg)
    raw_root = Path(cfg["raw_vehicle_root"])
    files = sorted(raw_root.rglob("*_vehicle.csv"))
    if args.max_records is not None:
        files = files[: args.max_records]

    all_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    plot_counts: dict[str, int] = {}
    total_3d = 0
    for file_idx, path in enumerate(files):
        try:
            rs = load_record(path, cfg)
            intervals = find_candidate_intervals(rs, cfg)
            rec_status = "ok"
            rec_error = ""
        except Exception as exc:
            subject, session = parse_subject_session(path)
            inventory_rows.append(
                {
                    "subject": subject,
                    "session_stamp": session,
                    "vehicle_file": str(path),
                    "status": "failed",
                    "error": str(exc),
                    "episode_count": 0,
                }
            )
            continue

        for ep_i, (s, e) in enumerate(intervals):
            row = summarize_episode(rs, s, e, ep_i, road_df)
            all_rows.append(row)
            if cfg.get("make_review_plots", True) and not args.no_plots:
                group = str(row["episode_group_id"])
                count = plot_counts.get(group, 0)
                if count < int(cfg["max_review_plots_per_group"]):
                    group_dir = dirs["review"] / f"{group}_{row['episode_group_cn']}"
                    group_dir.mkdir(parents=True, exist_ok=True)
                    plot_episode(rs, row, group_dir / f"{row['episode_uid']}.png")
                    plot_counts[group] = count + 1
            if cfg.get("make_3d_static_plots", True) and not args.no_3d and total_3d < int(cfg["max_3d_plots"]):
                if row["episode_group_id"] in {"core_extreme", "conservative_extreme", "secondary"}:
                    plot_trajectory_3d(rs, row, dirs["trajectory3d"] / f"{row['episode_uid']}_3d.png")
                    total_3d += 1

        inventory_rows.append(
            {
                "subject": rs.subject,
                "session_stamp": rs.session_stamp,
                "vehicle_file": str(path),
                "status": rec_status,
                "error": rec_error,
                "row_count": len(rs.df),
                "duration_s": float(np.nanmax(rs.t) - np.nanmin(rs.t)),
                "sampling_rate_median_hz": rs.fs,
                "episode_count": len(intervals),
                "available_columns": "|".join([f"{k}:{v}" for k, v in rs.cols.items() if v]),
            }
        )
        if (file_idx + 1) % 10 == 0:
            print(f"[{now_text()}] processed {file_idx + 1}/{len(files)} records, episodes={len(all_rows)}")

    all_df = pd.DataFrame(all_rows)
    inv_df = pd.DataFrame(inventory_rows)
    all_df.to_csv(dirs["tables"] / "record_level_episodes_all_v1_0.csv", index=False, encoding="utf-8-sig")
    inv_df.to_csv(dirs["tables"] / "record_level_file_inventory_v1_0.csv", index=False, encoding="utf-8-sig")
    if not all_df.empty:
        for group_id, sub in all_df.groupby("episode_group_id"):
            sub.to_csv(dirs["tables"] / f"record_level_episodes_{group_id}_v1_0.csv", index=False, encoding="utf-8-sig")
        build_report(all_df, inv_df, dirs, cfg)

    status_text = f"""
## {now_text()} 完整记录级 episode 重建 v1.0

- 已运行完整记录级 episode 重建脚本。
- 输入：`{raw_root}`
- 输出：`{dirs['out']}`
- 处理记录数：{len(inv_df)}
- 检测 episode 数：{len(all_df)}
- 说明：道路/场景信息只作为上下文，不直接作为最终事件锚点；一条完整实验记录允许产生多个 episode。
"""
    append_note(REBUILD_ROOT / "00_project_notes" / "daily_logs" / "2026-05-20.md", status_text)
    append_note(REBUILD_ROOT / "00_project_notes" / "PROJECT_STATUS_CN.md", status_text)
    print(status_text)


if __name__ == "__main__":
    main()
