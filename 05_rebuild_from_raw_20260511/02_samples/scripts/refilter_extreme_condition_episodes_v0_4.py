# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from collections import Counter
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


PROJECT_ROOT = Path(r"F:/data_set_process/data_process")
ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
V03_DIR = ROOT / "02_samples" / "extreme_condition_episodes_v0_3"
V03_TABLE_DIR = V03_DIR / "tables"
EPISODE_TABLE = V03_TABLE_DIR / "extreme_condition_episodes_all_v0_3.csv"
FAST_TABLE = V03_TABLE_DIR / "fast_steer_vehicle_response_split_v0_3.csv"
TIMING_TABLE = V03_TABLE_DIR / "fast_steer_anchor_timing_audit_v0_3.csv"

OUT_DIR = ROOT / "02_samples" / "extreme_condition_episodes_v0_4"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
PANEL_DIR = FIG_DIR / "review_panels"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-19.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"

VEHICLE_COLS = [
    "StorageTime",
    "zx|SteeringWheel",
    "zx|ay",
    "zx|vyaw",
    "zx|vroll",
    "zx|roll",
    "zx1|lateraldistance",
    "zx|lateraldistance",
    "zx|BrakePedal",
    "zx|ax",
    "zx1|v_km/h",
    "zx|vx",
    "zx1|mu",
    "zx1|lanecurvatureXY",
    "zx|lanecurvatureXY",
]

DYNAMIC_FIELDS = [
    ("ay", "zx|ay", "ay_threshold", 0.35, "横向加速度"),
    ("yaw_rate", "zx|vyaw", "yaw_rate_threshold", 0.035, "横摆角速度"),
    ("roll_rate", "zx|vroll", "roll_rate_threshold", 0.030, "横滚角速度"),
    ("roll_angle", "zx|roll", "roll_angle_threshold", 0.020, "横滚角"),
]

PANEL_LIMITS = {
    "01_核心保留_锚点后车辆变化": 45,
    "02_保留_车辆变化但驾驶员操作弱": 45,
    "03_次级保留_快打方向且有弱车辆变化": 35,
    "04_复核_快打方向但车辆变化弱": 35,
    "05_复核_锚点可能在事件中段": 30,
    "06_排除_锚点后车和人都弱": 40,
    "07_排除_锚点偏晚事件已稳定": 35,
    "08_复核_窗口或坐标风险": 30,
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, PANEL_DIR, REPORT_DIR, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "yes", "y", "是"}


def parse_time_seconds(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if float(numeric.notna().mean()) >= 0.8:
        arr = numeric.to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        return arr - finite[0] if finite.size else np.full(len(series), np.nan)
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() == 0:
        return np.full(len(series), np.nan)
    base = parsed.dropna().iloc[0]
    return (parsed - base).dt.total_seconds().to_numpy(dtype=float)


def gradient(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & np.isfinite(times)
    if valid.sum() < 3:
        return out
    idx = np.arange(values.size)
    filled = values.copy()
    filled[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
    dt = np.gradient(times)
    dt[~np.isfinite(dt) | (np.abs(dt) < 1e-6)] = np.nan
    return np.gradient(filled) / dt


def max_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(np.abs(arr)))


def robust_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmedian(arr))


def load_vehicle(path_text: str) -> pd.DataFrame | None:
    path = Path(str(path_text))
    if not path.exists():
        return None
    try:
        header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
        usecols = [col for col in VEHICLE_COLS if col in header.columns]
        if "StorageTime" not in usecols:
            return None
        df = pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    except Exception:
        return None
    df["time_rel_s"] = parse_time_seconds(df["StorageTime"])
    df = df[np.isfinite(df["time_rel_s"])].copy()
    df = df.drop_duplicates("time_rel_s").sort_values("time_rel_s")
    for col in df.columns:
        if col not in {"StorageTime", "time_rel_s"}:
            df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(limit_direction="both")
    if "zx|SteeringWheel" in df.columns:
        df["steer_rate"] = gradient(df["zx|SteeringWheel"].to_numpy(dtype=float), df["time_rel_s"].to_numpy(dtype=float))
    else:
        df["steer_rate"] = np.nan
    lat_col = "zx1|lateraldistance" if "zx1|lateraldistance" in df.columns else "zx|lateraldistance"
    if lat_col in df.columns:
        lat = df[lat_col].to_numpy(dtype=float)
        time = df["time_rel_s"].to_numpy(dtype=float)
        df["lateral_distance_selected"] = lat
        df["lateral_velocity"] = gradient(lat, time)
        df["lateral_step_abs"] = np.r_[0.0, np.abs(np.diff(lat))]
    else:
        df["lateral_distance_selected"] = np.nan
        df["lateral_velocity"] = np.nan
        df["lateral_step_abs"] = np.nan
    return df.reset_index(drop=True)


def w(df: pd.DataFrame, anchor: float, start: float, end: float) -> pd.DataFrame:
    return df[(df["time_rel_s"] >= anchor + start) & (df["time_rel_s"] <= anchor + end)].copy()


def peak_time_abs(seg: pd.DataFrame, col: str) -> tuple[float, float]:
    if col not in seg.columns or seg.empty:
        return float("nan"), float("nan")
    values = seg[col].to_numpy(dtype=float)
    times = seg["time_rel_s"].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(times)
    if not valid.any():
        return float("nan"), float("nan")
    local_values = values[valid]
    local_times = times[valid]
    idx = int(np.nanargmax(np.abs(local_values)))
    return float(abs(local_values[idx])), float(local_times[idx])


def dynamic_metrics(row: pd.Series, vehicle: pd.DataFrame, anchor: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pre = w(vehicle, anchor, -3.0, -0.3)
    post = w(vehicle, anchor, 0.0, 3.0)
    post_full = w(vehicle, anchor, 0.0, 5.0)
    early = w(vehicle, anchor, -0.1, 0.8)

    pre_score_parts: list[float] = []
    post_score_parts: list[float] = []
    post_components = 0
    weak_post_components = 0
    for key, col, threshold_col, floor, _label in DYNAMIC_FIELDS:
        threshold = max(floor, safe_float(row.get(threshold_col), floor))
        pre_peak, pre_t = peak_time_abs(pre, col)
        post_peak, post_t = peak_time_abs(post, col)
        post_full_peak, post_full_t = peak_time_abs(post_full, col)
        out[f"v04_pre_{key}_peak"] = pre_peak
        out[f"v04_pre_{key}_peak_rel_s"] = pre_t - anchor if math.isfinite(pre_t) else float("nan")
        out[f"v04_post_{key}_peak"] = post_peak
        out[f"v04_post_{key}_peak_rel_s"] = post_t - anchor if math.isfinite(post_t) else float("nan")
        out[f"v04_post_full_{key}_peak"] = post_full_peak
        out[f"v04_post_full_{key}_peak_rel_s"] = post_full_t - anchor if math.isfinite(post_full_t) else float("nan")
        if math.isfinite(pre_peak):
            pre_score_parts.append(pre_peak / threshold)
        if math.isfinite(post_peak):
            score = post_peak / threshold
            post_score_parts.append(score)
            if score >= 1.0:
                post_components += 1
            elif score >= 0.65:
                weak_post_components += 1

    lat_pre = max_abs(pre["lateral_velocity"].to_numpy(dtype=float)) if "lateral_velocity" in pre else float("nan")
    lat_post = max_abs(post["lateral_velocity"].to_numpy(dtype=float)) if "lateral_velocity" in post else float("nan")
    lat_step_post = max_abs(post["lateral_step_abs"].to_numpy(dtype=float)) if "lateral_step_abs" in post else float("nan")
    lat_change_post = float("nan")
    if "lateral_distance_selected" in post_full and len(post_full):
        lat_values = post_full["lateral_distance_selected"].to_numpy(dtype=float)
        if np.isfinite(lat_values).any():
            lat_change_post = float(np.nanmax(lat_values) - np.nanmin(lat_values))
    out["v04_pre_lateral_velocity_peak"] = lat_pre
    out["v04_post_lateral_velocity_peak"] = lat_post
    out["v04_post_lateral_step_peak"] = lat_step_post
    out["v04_post_lateral_change"] = abs(lat_change_post) if math.isfinite(lat_change_post) else float("nan")

    if "zx1|mu" in post_full.columns and len(post_full):
        mu = post_full["zx1|mu"].to_numpy(dtype=float)
        out["v04_post_mu_change"] = float(np.nanmax(mu) - np.nanmin(mu)) if np.isfinite(mu).any() else float("nan")
        out["v04_post_min_mu"] = float(np.nanmin(mu)) if np.isfinite(mu).any() else float("nan")
    else:
        out["v04_post_mu_change"] = float("nan")
        out["v04_post_min_mu"] = float("nan")

    if "zx|BrakePedal" in post_full.columns and len(post_full):
        brake = post_full["zx|BrakePedal"].to_numpy(dtype=float)
        base_brake = robust_median(pre["zx|BrakePedal"].to_numpy(dtype=float)) if "zx|BrakePedal" in pre else 0.0
        out["v04_post_brake_delta"] = float(np.nanmax(brake) - base_brake) if np.isfinite(brake).any() else float("nan")
    else:
        out["v04_post_brake_delta"] = float("nan")

    pre_score = float(np.nanmax(pre_score_parts)) if pre_score_parts else float("nan")
    post_score = float(np.nanmax(post_score_parts)) if post_score_parts else float("nan")
    out["v04_pre_vehicle_dyn_score"] = pre_score
    out["v04_post_vehicle_dyn_score"] = post_score
    out["v04_post_vehicle_component_count"] = post_components
    out["v04_post_vehicle_weak_component_count"] = weak_post_components
    out["v04_has_strong_vehicle_after_anchor"] = int(post_components >= 2 or (post_components >= 1 and post_score >= 1.6))
    out["v04_has_weak_vehicle_after_anchor"] = int(
        out["v04_has_strong_vehicle_after_anchor"]
        or post_components >= 1
        or weak_post_components >= 2
        or (math.isfinite(lat_post) and lat_post >= max(0.20, 1.2 * (lat_pre if math.isfinite(lat_pre) else 0.0)))
    )
    out["v04_vehicle_peak_rel_s"] = float("nan")
    peak_candidates = [out.get(f"v04_post_{key}_peak_rel_s") for key, *_ in DYNAMIC_FIELDS]
    peak_candidates = [safe_float(x) for x in peak_candidates if math.isfinite(safe_float(x))]
    if peak_candidates:
        out["v04_vehicle_peak_rel_s"] = float(min(peak_candidates, key=abs))

    if len(early) and "zx|SteeringWheel" in early.columns:
        steer_base = robust_median(pre["zx|SteeringWheel"].to_numpy(dtype=float)) if len(pre) else robust_median(early["zx|SteeringWheel"].to_numpy(dtype=float))
        out["v04_early_steer_delta"] = max_abs(early["zx|SteeringWheel"].to_numpy(dtype=float) - steer_base)
    else:
        out["v04_early_steer_delta"] = float("nan")
    return out


def steering_metrics(row: pd.Series, vehicle: pd.DataFrame, anchor: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pre = w(vehicle, anchor, -3.0, -0.3)
    post = w(vehicle, anchor, 0.0, 2.0)
    post_full = w(vehicle, anchor, 0.0, 5.0)
    steer_thr = max(4.0, 0.70 * safe_float(row.get("steer_rate_threshold"), 6.0))
    amp_thr = max(0.25, 0.70 * safe_float(row.get("steer_amp_threshold"), 0.35))

    pre_rate, pre_rate_t = peak_time_abs(pre, "steer_rate")
    post_rate, post_rate_t = peak_time_abs(post, "steer_rate")
    out["v04_pre_steer_rate_peak"] = pre_rate
    out["v04_pre_steer_rate_peak_rel_s"] = pre_rate_t - anchor if math.isfinite(pre_rate_t) else float("nan")
    out["v04_post_steer_rate_peak"] = post_rate
    out["v04_post_steer_rate_peak_rel_s"] = post_rate_t - anchor if math.isfinite(post_rate_t) else float("nan")
    out["v04_steer_rate_threshold_used"] = steer_thr
    out["v04_steer_amp_threshold_used"] = amp_thr

    if len(pre) and "zx|SteeringWheel" in pre.columns:
        baseline = robust_median(pre["zx|SteeringWheel"].to_numpy(dtype=float))
    elif "zx|SteeringWheel" in vehicle.columns:
        near = w(vehicle, anchor, -0.5, 0.1)
        baseline = robust_median(near["zx|SteeringWheel"].to_numpy(dtype=float))
    else:
        baseline = float("nan")
    out["v04_steer_baseline"] = baseline
    if len(post_full) and "zx|SteeringWheel" in post_full.columns and math.isfinite(baseline):
        post_delta = max_abs(post_full["zx|SteeringWheel"].to_numpy(dtype=float) - baseline)
    else:
        post_delta = float("nan")
    out["v04_post_steer_delta"] = post_delta
    out["v04_has_driver_action_after_anchor"] = int(
        (math.isfinite(post_rate) and post_rate >= steer_thr)
        or (math.isfinite(post_delta) and post_delta >= amp_thr)
    )
    out["v04_has_weak_driver_action_after_anchor"] = int(
        out["v04_has_driver_action_after_anchor"]
        or (math.isfinite(post_rate) and post_rate >= 0.55 * steer_thr)
        or (math.isfinite(post_delta) and post_delta >= 0.55 * amp_thr)
    )
    out["v04_steer_pre_over_post_ratio"] = (
        float(pre_rate / max(post_rate, 1e-6)) if math.isfinite(pre_rate) and math.isfinite(post_rate) else float("nan")
    )
    return out


def classify(row: dict[str, Any]) -> tuple[str, str, str, str]:
    window_ok = safe_bool(row.get("window_complete"))
    coord_ok = safe_bool(row.get("coordinate_continuity_ok"))
    strong_vehicle = int(row.get("v04_has_strong_vehicle_after_anchor", 0)) == 1
    weak_vehicle = int(row.get("v04_has_weak_vehicle_after_anchor", 0)) == 1
    driver = int(row.get("v04_has_driver_action_after_anchor", 0)) == 1
    weak_driver = int(row.get("v04_has_weak_driver_action_after_anchor", 0)) == 1
    pre_vehicle = safe_float(row.get("v04_pre_vehicle_dyn_score"))
    post_vehicle = safe_float(row.get("v04_post_vehicle_dyn_score"))
    pre_rate = safe_float(row.get("v04_pre_steer_rate_peak"))
    post_rate = safe_float(row.get("v04_post_steer_rate_peak"))
    timing = str(row.get("anchor_timing_label", ""))

    post_quiet = (not weak_vehicle) and (not weak_driver)
    late_by_ratio = (
        math.isfinite(pre_vehicle)
        and math.isfinite(post_vehicle)
        and pre_vehicle >= 1.0
        and post_vehicle < 0.65
        and (not weak_driver)
    ) or (
        math.isfinite(pre_rate)
        and math.isfinite(post_rate)
        and pre_rate >= 8.0
        and post_rate < 4.0
        and (not weak_vehicle)
    )
    late_by_old_timing = timing in {"EXCLUDE_LATE_ANCHOR_STABILIZED", "RISK_LATE_ANCHOR_REVIEW"}

    if not window_ok:
        return "REVIEW_WINDOW_OR_COORD_RISK", "复核：窗口不完整", "窗口不完整，不能直接用于训练", "review"
    if not coord_ok and not strong_vehicle:
        return "REVIEW_WINDOW_OR_COORD_RISK", "复核：坐标连续性风险", "横向坐标可能跳变，且没有足够强的非坐标车辆证据", "review"
    if post_quiet:
        return "EXCLUDE_NO_POST_CHANGE", "排除：锚点后车和人都弱", "锚点后车辆状态和方向盘操作都没有明显变化", "exclude"
    if late_by_old_timing or late_by_ratio:
        if not strong_vehicle:
            return "EXCLUDE_LATE_ANCHOR_STABILIZED", "排除：锚点偏晚或事件已稳定", "锚点前已有主要动作，锚点后变化不足", "exclude"
        return "REVIEW_ANCHOR_MID_EVENT", "复核：锚点可能在事件中段", "锚点前已有变化，但锚点后车辆仍明显变化", "review"
    if strong_vehicle and driver:
        return "KEEP_CORE_VEHICLE_AND_DRIVER", "核心保留：车辆变化+驾驶员操作", "锚点后车辆动态增强，同时有方向盘动作", "primary_train"
    if strong_vehicle and not driver:
        return "KEEP_CORE_VEHICLE_DRIVER_WEAK", "核心保留：车辆变化但驾驶员操作弱", "锚点后车辆动态明确，驾驶员操作不强也保留", "primary_train"
    if weak_vehicle and driver:
        return "KEEP_SECONDARY_FAST_STEER_WEAK_VEHICLE", "次级保留：快打方向且有弱车辆变化", "方向盘速度/幅值明显，车辆变化偏弱，适合次级训练或复核", "secondary_train"
    if driver and not weak_vehicle:
        return "REVIEW_FAST_STEER_NO_VEHICLE", "复核：快打方向但车辆变化弱", "可能是直线维持、普通操作或未诱发姿态变化", "review"
    if weak_vehicle and weak_driver:
        return "KEEP_SECONDARY_WEAK_BOTH", "次级保留：车和人都有弱变化", "锚点后车和驾驶员都有弱变化，先作为扩展样本", "secondary_train"
    if weak_vehicle and not weak_driver:
        return "KEEP_CORE_VEHICLE_DRIVER_WEAK", "核心保留：车辆变化但驾驶员操作弱", "锚点后车辆有变化，即便驾驶员不明显操作也保留", "primary_train"
    return "REVIEW_UNCLEAR", "复核：语义不清", "不满足明确保留或排除规则", "review"


def merge_context_tables(episodes: pd.DataFrame) -> pd.DataFrame:
    out = episodes.copy()
    if FAST_TABLE.exists():
        fast_cols = [
            "episode_uid",
            "fast_vehicle_response_split",
            "fast_vehicle_response_reason_cn",
            "fast_vehicle_response_recommended_use",
        ]
        fast = pd.read_csv(FAST_TABLE, encoding="utf-8-sig", low_memory=False)
        out = out.merge(fast[[c for c in fast_cols if c in fast.columns]], on="episode_uid", how="left")
    if TIMING_TABLE.exists():
        timing_cols = [
            "episode_uid",
            "pre_steer_rate_max_m4_to_m03",
            "post_steer_rate_max_m01_to_p2",
            "pre_vehicle_dyn_score_max_m4_to_m03",
            "post_vehicle_dyn_score_max_m01_to_p2",
            "pre_over_post_steer_rate_ratio",
            "pre_over_post_vehicle_dyn_ratio",
            "anchor_timing_label",
            "anchor_timing_reason_cn",
        ]
        timing = pd.read_csv(TIMING_TABLE, encoding="utf-8-sig", low_memory=False)
        out = out.merge(timing[[c for c in timing_cols if c in timing.columns]], on="episode_uid", how="left")
    return out


def score_row(row: pd.Series, cache: dict[str, pd.DataFrame | None]) -> dict[str, Any]:
    base = row.to_dict()
    anchor = safe_float(row.get("t_condition_anchor"))
    if not math.isfinite(anchor):
        base.update(
            {
                "v04_audit_status": "missing_anchor",
                "v04_label": "EXCLUDE_NO_ANCHOR",
                "v04_label_cn": "排除：缺少锚点",
                "v04_reason_cn": "缺少 t_condition_anchor",
                "v04_recommended_use": "exclude",
            }
        )
        return base

    path_text = str(row.get("vehicle_raw_absolute_path", ""))
    if path_text not in cache:
        cache[path_text] = load_vehicle(path_text)
    vehicle = cache[path_text]
    if vehicle is None or vehicle.empty:
        base.update(
            {
                "v04_audit_status": "missing_vehicle",
                "v04_label": "EXCLUDE_MISSING_VEHICLE",
                "v04_label_cn": "排除：车辆文件不可读",
                "v04_reason_cn": "原始车辆 CSV 不可读取",
                "v04_recommended_use": "exclude",
            }
        )
        return base

    tmin = float(vehicle["time_rel_s"].min())
    tmax = float(vehicle["time_rel_s"].max())
    base["v04_has_pre2"] = int(tmin <= anchor - 2.0)
    base["v04_has_post5"] = int(tmax >= anchor + 5.0)
    base["v04_audit_status"] = "ok"
    base.update(dynamic_metrics(row, vehicle, anchor))
    base.update(steering_metrics(row, vehicle, anchor))
    label, label_cn, reason_cn, use = classify(base)
    base["v04_label"] = label
    base["v04_label_cn"] = label_cn
    base["v04_reason_cn"] = reason_cn
    base["v04_recommended_use"] = use
    return base


def group_dir_for_label(label: str) -> str:
    mapping = {
        "KEEP_CORE_VEHICLE_AND_DRIVER": "01_核心保留_锚点后车辆变化",
        "KEEP_CORE_VEHICLE_DRIVER_WEAK": "02_保留_车辆变化但驾驶员操作弱",
        "KEEP_SECONDARY_FAST_STEER_WEAK_VEHICLE": "03_次级保留_快打方向且有弱车辆变化",
        "KEEP_SECONDARY_WEAK_BOTH": "03_次级保留_快打方向且有弱车辆变化",
        "REVIEW_FAST_STEER_NO_VEHICLE": "04_复核_快打方向但车辆变化弱",
        "REVIEW_ANCHOR_MID_EVENT": "05_复核_锚点可能在事件中段",
        "EXCLUDE_NO_POST_CHANGE": "06_排除_锚点后车和人都弱",
        "EXCLUDE_LATE_ANCHOR_STABILIZED": "07_排除_锚点偏晚事件已稳定",
        "REVIEW_WINDOW_OR_COORD_RISK": "08_复核_窗口或坐标风险",
    }
    return mapping.get(label, "09_复核_其他语义不清")


def plot_episode(row: pd.Series, vehicle: pd.DataFrame, out_path: Path) -> None:
    anchor = safe_float(row.get("t_condition_anchor"))
    if not math.isfinite(anchor):
        return
    seg = vehicle[(vehicle["time_rel_s"] >= anchor - 4.0) & (vehicle["time_rel_s"] <= anchor + 6.0)].copy()
    if seg.empty:
        return
    x = seg["time_rel_s"].to_numpy(dtype=float) - anchor
    series = [
        ("方向盘角", "zx|SteeringWheel"),
        ("方向盘角速度", "steer_rate"),
        ("车速", "zx1|v_km/h" if "zx1|v_km/h" in seg.columns else "zx|vx"),
        ("制动踏板", "zx|BrakePedal"),
        ("横向加速度", "zx|ay"),
        ("横摆角速度", "zx|vyaw"),
        ("横滚角速度", "zx|vroll"),
        ("横滚角", "zx|roll"),
    ]
    if "lateral_distance_selected" in seg.columns:
        series.append(("横向偏移", "lateral_distance_selected"))
    if "zx1|mu" in seg.columns:
        series.append(("路面附着", "zx1|mu"))

    fig, axes = plt.subplots(len(series), 1, figsize=(15, 2.0 * len(series)), sharex=True)
    if len(series) == 1:
        axes = [axes]
    for ax, (name, col) in zip(axes, series):
        if col in seg.columns:
            ax.plot(x, seg[col].to_numpy(dtype=float), lw=1.2)
        ax.axvline(0.0, color="crimson", ls="--", lw=1.0, label="当前锚点")
        peak = safe_float(row.get("v04_vehicle_peak_rel_s"))
        if math.isfinite(peak):
            ax.axvline(peak, color="green", ls=":", lw=1.0)
        steer_peak = safe_float(row.get("v04_post_steer_rate_peak_rel_s"))
        if math.isfinite(steer_peak):
            ax.axvline(steer_peak, color="orange", ls=":", lw=1.0)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("相对当前锚点时间 / s")
    title = (
        f"{row.get('episode_uid')} | {row.get('v04_label_cn')} | "
        f"车后={safe_float(row.get('v04_post_vehicle_dyn_score')):.2f} "
        f"盘速={safe_float(row.get('v04_post_steer_rate_peak')):.1f}"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def select_review_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for label, group in df.groupby("v04_label", dropna=False):
        folder = group_dir_for_label(str(label))
        limit = PANEL_LIMITS.get(folder, 20)
        g = group.copy()
        g["plot_score"] = (
            pd.to_numeric(g["v04_post_vehicle_dyn_score"], errors="coerce").fillna(0)
            + 0.15 * pd.to_numeric(g["v04_post_steer_rate_peak"], errors="coerce").fillna(0)
            + 0.1 * pd.to_numeric(g["condition_score_peak"], errors="coerce").fillna(0)
        )
        if str(label).startswith("EXCLUDE"):
            g = g.sort_values(["plot_score"], ascending=False)
        elif "REVIEW" in str(label):
            g = g.assign(boundary=(g["v04_post_vehicle_dyn_score"].astype(float).fillna(0) - 0.9).abs())
            g = g.sort_values(["boundary", "plot_score"], ascending=[True, False])
        else:
            g = g.sort_values(["plot_score"], ascending=False)
        rows.append(g.head(limit))
    return pd.concat(rows, ignore_index=True) if rows else df.head(0)


def write_tables(df: pd.DataFrame) -> None:
    df.to_csv(TABLE_DIR / "extreme_condition_episodes_refiltered_v0_4.csv", index=False, encoding="utf-8-sig")
    df[df["v04_recommended_use"].eq("primary_train")].to_csv(
        TABLE_DIR / "primary_train_episodes_v0_4.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v04_recommended_use"].eq("secondary_train")].to_csv(
        TABLE_DIR / "secondary_train_episodes_v0_4.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v04_recommended_use"].eq("review")].to_csv(
        TABLE_DIR / "manual_review_episodes_v0_4.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v04_recommended_use"].eq("exclude")].to_csv(
        TABLE_DIR / "excluded_episodes_v0_4.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v04_recommended_use"].isin(["primary_train", "secondary_train"])].to_csv(
        TABLE_DIR / "train_candidate_episodes_v0_4.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(Counter(df["v04_label"]).items(), columns=["v04_label", "count"]).sort_values(
        "count", ascending=False
    ).to_csv(TABLE_DIR / "v04_label_counts.csv", index=False, encoding="utf-8-sig")
    by_context = (
        df.pivot_table(index="condition_context_cn", columns="v04_recommended_use", values="episode_uid", aggfunc="count", fill_value=0)
        .reset_index()
    )
    by_context.to_csv(TABLE_DIR / "v04_context_use_counts.csv", index=False, encoding="utf-8-sig")
    by_subject = (
        df.pivot_table(index="subject", columns="v04_recommended_use", values="episode_uid", aggfunc="count", fill_value=0)
        .reset_index()
    )
    by_subject.to_csv(TABLE_DIR / "v04_subject_use_counts.csv", index=False, encoding="utf-8-sig")


def write_report(df: pd.DataFrame, image_index: pd.DataFrame) -> None:
    label_counts = df["v04_label_cn"].value_counts()
    use_counts = df["v04_recommended_use"].value_counts()
    context_counts = pd.crosstab(df["condition_context_cn"], df["v04_recommended_use"])
    primary = int(use_counts.get("primary_train", 0))
    secondary = int(use_counts.get("secondary_train", 0))
    review = int(use_counts.get("review", 0))
    exclude = int(use_counts.get("exclude", 0))
    train_total = primary + secondary
    lines = [
        "# v0.4 极限工况样本重新筛选说明",
        "",
        "## 这次筛选改了什么",
        "",
        "这次不是继续比较 809 个样本版本，而是回到 v0.3 的 1574 个初始 episode，按新的人工判断重新筛：",
        "",
        "- 方向盘转动速度要作为驾驶员紧急操作证据；",
        "- 要检查当前锚点是否偏晚，如果锚点后车辆和驾驶员都已经稳定，则不作为训练样本；",
        "- 如果锚点后车辆状态有明显变化，即使驾驶员没有明显操作，也可以保留，因为这可能代表保守驾驶员、制动为主或车辆扰动主导；",
        "- 如果只有方向盘快打但车辆变化弱，先放入人工复核，不直接作为核心极限样本。",
        "",
        "## 总体数量",
        "",
        f"- 初始 episode 数：{len(df)}",
        f"- 主训练候选：{primary}",
        f"- 次级训练候选：{secondary}",
        f"- 主+次级候选合计：{train_total}",
        f"- 待人工复核：{review}",
        f"- 暂排除：{exclude}",
        "",
        "## 分类数量",
        "",
        label_counts.to_markdown(),
        "",
        "## 按场景/上下文统计",
        "",
        context_counts.to_markdown(),
        "",
        "## 怎么理解",
        "",
        "- 主训练候选不是只看方向盘，也不是只看车身，而是锚点后车辆动态仍然发生或增强的样本。",
        "- 驾驶员操作弱但车辆变化明显的样本被保留，这符合用户最新判断。",
        "- 锚点后车和人都弱的样本被排除，因为它们更像锚点偏晚、事件结束或直线轻微维持。",
        "- 快打方向但车辆变化弱的样本先复核，因为其中一部分可能是直线维持方向盘，一部分可能是真正紧急操作但车辆响应不强。",
        "",
        "## 输出位置",
        "",
        f"- 总表：`{TABLE_DIR / 'extreme_condition_episodes_refiltered_v0_4.csv'}`",
        f"- 主训练候选：`{TABLE_DIR / 'primary_train_episodes_v0_4.csv'}`",
        f"- 次级训练候选：`{TABLE_DIR / 'secondary_train_episodes_v0_4.csv'}`",
        f"- 主+次级训练候选：`{TABLE_DIR / 'train_candidate_episodes_v0_4.csv'}`",
        f"- 待复核：`{TABLE_DIR / 'manual_review_episodes_v0_4.csv'}`",
        f"- 排除：`{TABLE_DIR / 'excluded_episodes_v0_4.csv'}`",
        f"- 复核图索引：`{TABLE_DIR / 'v04_review_figure_index.csv'}`",
        f"- 复核图目录：`{PANEL_DIR}`",
        "",
        f"本轮共生成复核图 {len(image_index)} 张。",
    ]
    (REPORT_DIR / "stage02_extreme_condition_refilter_v0_4_user_summary_cn.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def append_notes(df: pd.DataFrame) -> None:
    use_counts = df["v04_recommended_use"].value_counts()
    primary = int(use_counts.get("primary_train", 0))
    secondary = int(use_counts.get("secondary_train", 0))
    review = int(use_counts.get("review", 0))
    exclude = int(use_counts.get("exclude", 0))
    block = (
        "## 2026-05-19 v0.4 极限工况样本重新筛选\n\n"
        "- 为什么做：用户指出目标不是继续比较 809 样本版本，而是回到 1574 个初始 episode，按方向盘速度、锚点延时性、锚点后车辆/驾驶员是否仍有变化重新筛选。\n"
        "- 本轮规则：锚点后车辆有变化即保留，即使驾驶员操作弱；锚点后车和驾驶员都弱则排除；快打方向但车辆变化弱先复核。\n"
        f"- 当前结果：主训练候选 {primary}，次级训练候选 {secondary}，待复核 {review}，排除 {exclude}。\n"
        f"- 用户查看版报告：`{REPORT_DIR / 'stage02_extreme_condition_refilter_v0_4_user_summary_cn.md'}`。\n"
        f"- 输出目录：`{OUT_DIR}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            if "## 2026-05-19 v0.4 极限工况样本重新筛选" not in raw:
                path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")
    if ARTIFACT_INDEX.exists():
        raw = ARTIFACT_INDEX.read_text(encoding="utf-8")
        artifact = (
            "## 2026-05-19 v0.4 极限工况样本重新筛选\n\n"
            f"- 用户查看版报告：`{REPORT_DIR / 'stage02_extreme_condition_refilter_v0_4_user_summary_cn.md'}`\n"
            f"- 总表：`{TABLE_DIR / 'extreme_condition_episodes_refiltered_v0_4.csv'}`\n"
            f"- 主+次级训练候选：`{TABLE_DIR / 'train_candidate_episodes_v0_4.csv'}`\n"
            f"- 复核图目录：`{PANEL_DIR}`\n"
        )
        if "## 2026-05-19 v0.4 极限工况样本重新筛选" not in raw:
            ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    episodes = pd.read_csv(EPISODE_TABLE, encoding="utf-8-sig", low_memory=False)
    episodes = merge_context_tables(episodes)
    cache: dict[str, pd.DataFrame | None] = {}
    rows: list[dict[str, Any]] = []
    for idx, row in episodes.iterrows():
        if idx % 100 == 0:
            print(f"process {idx}/{len(episodes)}", flush=True)
        rows.append(score_row(row, cache))
    result = pd.DataFrame(rows)
    result = result.sort_values(["v04_recommended_use", "subject", "session_stamp", "t_condition_anchor"]).reset_index(drop=True)
    write_tables(result)

    review_rows = select_review_rows(result)
    image_rows: list[dict[str, Any]] = []
    for i, row in review_rows.reset_index(drop=True).iterrows():
        vehicle = cache.get(str(row.get("vehicle_raw_absolute_path")))
        if vehicle is None:
            vehicle = load_vehicle(str(row.get("vehicle_raw_absolute_path", "")))
        if vehicle is None:
            continue
        folder = group_dir_for_label(str(row.get("v04_label")))
        filename = f"{i:03d}_{str(row.get('episode_uid'))[:80]}.png"
        filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        out_path = PANEL_DIR / folder / filename
        plot_episode(row, vehicle, out_path)
        image_rows.append(
            {
                "episode_uid": row.get("episode_uid"),
                "v04_label": row.get("v04_label"),
                "v04_label_cn": row.get("v04_label_cn"),
                "v04_recommended_use": row.get("v04_recommended_use"),
                "condition_context_cn": row.get("condition_context_cn"),
                "image_path": str(out_path),
            }
        )
    image_index = pd.DataFrame(image_rows)
    image_index.to_csv(TABLE_DIR / "v04_review_figure_index.csv", index=False, encoding="utf-8-sig")
    write_report(result, image_index)
    append_notes(result)
    print(result["v04_recommended_use"].value_counts().to_string())
    print(f"report={REPORT_DIR / 'stage02_extreme_condition_refilter_v0_4_user_summary_cn.md'}")


if __name__ == "__main__":
    main()
