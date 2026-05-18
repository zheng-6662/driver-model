# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import math
import sys
from collections import Counter, defaultdict
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


PROJECT_ROOT = Path(r"F:/data_set_process/data_process")
ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
RAW_VEHICLE_ROOT = PROJECT_ROOT / "01_datasets" / "数据预处理" / "原始车辆数据"

OUT_DIR = ROOT / "02_samples" / "extreme_condition_episodes_v0_3"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
PANEL_DIR = FIG_DIR / "review_panels"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-18.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"

OPTIONAL_CONTEXT_TABLES = [
    ROOT
    / "02_samples"
    / "vehicle_instability_all_raw_rescreen_v0_1"
    / "tables"
    / "all_raw_vehicle_instability_candidates_v0_1.csv",
    ROOT
    / "02_samples"
    / "episode_first_event_v0_6"
    / "tables"
    / "episode_candidates_v0_6.csv",
]

VEHICLE_COLS = [
    "ID",
    "StorageTime",
    "zx|SteeringWheel",
    "zx|ay",
    "zx|vyaw",
    "zx|vroll",
    "zx|roll",
    "zx1|lateraldistance",
    "zx|lateraldistance",
    "zx|BrakePedal",
    "zx|AcceleratorPedal",
    "zx|ax",
    "zx1|v_km/h",
    "zx|vx",
    "zx1|mu",
    "zx1|lanecurvatureXY",
    "zx|lanecurvatureXY",
    "zx|x",
    "zx|y",
]

PRE_WINDOW_S = 2.0
POST_WINDOW_S = 5.0
EPISODE_MIN_GAP_S = 2.5
CONDITION_MIN_DUR_S = 0.15
SMOOTH_S = 0.08
MAX_PANEL_PER_CLASS = {
    "strong_response": 35,
    "weak_or_conservative": 35,
    "delayed_or_no_steer": 35,
    "normal_control": 25,
    "manual_review": 35,
    "excluded": 20,
}


@dataclass
class SignalSpec:
    col: str
    floor: float
    q: float
    label: str


SIGNAL_SPECS = {
    "ay": SignalSpec("zx|ay", 0.35, 92.0, "横向加速度"),
    "yaw_rate": SignalSpec("zx|vyaw", 0.035, 92.0, "横摆角速度"),
    "roll_rate": SignalSpec("zx|vroll", 0.030, 92.0, "横滚角速度"),
    "roll_angle": SignalSpec("zx|roll", 0.018, 92.0, "横滚角"),
    "curvature": SignalSpec("zx1|lanecurvatureXY", 0.0025, 90.0, "道路曲率"),
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, PANEL_DIR, LOG_DIR, REPORT_DIR, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def robust_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmedian(arr))


def robust_mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(arr - med)) * 1.4826)


def robust_threshold(values: np.ndarray, floor: float, q: float = 92.0, k_mad: float = 3.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = np.abs(arr[np.isfinite(arr)])
    if arr.size == 0:
        return float("nan")
    med = float(np.nanmedian(arr))
    mad = robust_mad(arr)
    qv = float(np.nanpercentile(arr, q))
    parts = [floor, qv]
    if math.isfinite(mad):
        parts.append(med + k_mad * mad)
    return float(np.nanmax(parts))


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
    out = np.gradient(filled) / dt
    return out


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if width <= 1 or values.size == 0:
        return values.copy()
    idx = np.arange(values.size)
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros_like(values)
    filled = values.copy()
    filled[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(filled, kernel, mode="same")


def max_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(np.abs(arr)))


def signed_peak(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    idx = int(np.nanargmax(np.abs(arr)))
    return float(arr[idx])


def ratio_bool(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else 0.0


def session_from_path(path: Path) -> tuple[str, str]:
    subject = path.parent.name
    session = path.stem
    if session.startswith("Entity_Recording_"):
        session = session[len("Entity_Recording_") :]
    if session.endswith("_vehicle"):
        session = session[: -len("_vehicle")]
    return subject, session


def load_vehicle_csv(path: Path) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    subject, session_stamp = session_from_path(path)
    meta: dict[str, Any] = {
        "subject": subject,
        "session_stamp": session_stamp,
        "vehicle_raw_absolute_path": str(path),
        "vehicle_raw_relative_path": str(path.relative_to(RAW_VEHICLE_ROOT)).replace("\\", "/")
        if path.is_relative_to(RAW_VEHICLE_ROOT)
        else path.name,
        "vehicle_raw_size_bytes": path.stat().st_size if path.exists() else 0,
        "vehicle_raw_sha256": "",
        "read_status": "failed",
        "read_error": "",
    }
    try:
        meta["vehicle_raw_sha256"] = sha256_file(path)
        header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
        usecols = [col for col in VEHICLE_COLS if col in header.columns]
        meta["available_columns"] = "|".join(usecols)
        meta["missing_required_columns"] = ""
        if "StorageTime" not in usecols:
            meta["missing_required_columns"] = "StorageTime"
            meta["read_error"] = "missing StorageTime"
            return None, meta
        df = pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    except Exception as exc:
        meta["read_error"] = repr(exc)
        return None, meta

    df["time_rel_s"] = parse_time_seconds(df["StorageTime"])
    df = df[np.isfinite(df["time_rel_s"])].copy()
    df = df.drop_duplicates("time_rel_s").sort_values("time_rel_s")
    if len(df) < 20:
        meta["read_error"] = "too few valid time rows"
        return None, meta
    for col in df.columns:
        if col not in {"ID", "StorageTime", "time_rel_s"}:
            df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(limit_direction="both")
    if "zx|SteeringWheel" in df.columns:
        df["steer_rate"] = gradient(df["zx|SteeringWheel"].to_numpy(dtype=float), df["time_rel_s"].to_numpy(dtype=float))
    else:
        df["steer_rate"] = np.nan
    lat_col = "zx1|lateraldistance" if "zx1|lateraldistance" in df.columns else "zx|lateraldistance"
    if lat_col in df.columns:
        lat = df[lat_col].to_numpy(dtype=float)
        df["lateral_distance_selected"] = lat
        df["lateral_step_abs"] = np.r_[0.0, np.abs(np.diff(lat))]
        df["lateral_velocity"] = gradient(lat, df["time_rel_s"].to_numpy(dtype=float))
    else:
        df["lateral_distance_selected"] = np.nan
        df["lateral_step_abs"] = 0.0
        df["lateral_velocity"] = 0.0
    curv_col = "zx1|lanecurvatureXY" if "zx1|lanecurvatureXY" in df.columns else "zx|lanecurvatureXY"
    if curv_col in df.columns:
        df["curvature_selected"] = df[curv_col]
    else:
        df["curvature_selected"] = np.nan
    meta["read_status"] = "ok"
    return df.reset_index(drop=True), meta


def contiguous_segments(mask: np.ndarray, times: np.ndarray, min_dur_s: float, merge_gap_s: float) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    times = np.asarray(times, dtype=float)
    if mask.size == 0:
        return []
    raw: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        if (not flag) and start is not None:
            raw.append((start, idx - 1))
            start = None
    if start is not None:
        raw.append((start, len(mask) - 1))
    kept: list[tuple[int, int]] = []
    for a, b in raw:
        if not math.isfinite(float(times[a])) or not math.isfinite(float(times[b])):
            continue
        if float(times[b] - times[a]) < min_dur_s:
            continue
        if kept and float(times[a] - times[kept[-1][1]]) <= merge_gap_s:
            kept[-1] = (kept[-1][0], b)
        else:
            kept.append((a, b))
    return kept


def nearest_time(times: np.ndarray, target: float) -> int:
    if times.size == 0:
        return 0
    idx = int(np.searchsorted(times, target))
    if idx <= 0:
        return 0
    if idx >= times.size:
        return times.size - 1
    return idx if abs(times[idx] - target) < abs(times[idx - 1] - target) else idx - 1


def window(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return df[(df["time_rel_s"] >= start) & (df["time_rel_s"] <= end)].copy()


def detect_steer_onset(df: pd.DataFrame, anchor: float) -> dict[str, Any]:
    out = {
        "t_steer_onset": np.nan,
        "steer_onset_relation": "none",
        "steer_response_delay_s": np.nan,
        "steer_amp_threshold": np.nan,
        "steer_rate_threshold": np.nan,
        "steer_delta_prepost": np.nan,
        "steer_rate_peak_near": np.nan,
        "steer_response_strength": "none",
        "steer_response_score": 0.0,
        "input_start_already_steering": False,
    }
    if "zx|SteeringWheel" not in df.columns:
        return out
    t = df["time_rel_s"].to_numpy(dtype=float)
    steer = df["zx|SteeringWheel"].to_numpy(dtype=float)
    rate = df["steer_rate"].to_numpy(dtype=float)
    local = window(df, anchor - 3.0, anchor + 3.0)
    if len(local) < 10:
        return out
    pre_far = window(df, anchor - 3.0, anchor - 2.0)
    baseline = robust_median(pre_far["zx|SteeringWheel"].to_numpy(dtype=float)) if len(pre_far) else robust_median(local["zx|SteeringWheel"].to_numpy(dtype=float))
    amp_thr = max(1.0, robust_threshold(local["zx|SteeringWheel"].to_numpy(dtype=float) - baseline, 0.8, q=88.0, k_mad=2.5))
    rate_thr = max(8.0, robust_threshold(local["steer_rate"].to_numpy(dtype=float), 6.0, q=92.0, k_mad=3.0))
    out["steer_amp_threshold"] = amp_thr
    out["steer_rate_threshold"] = rate_thr
    search = window(df, anchor - 2.5, anchor + 2.5)
    if len(search) < 10:
        return out
    cond = (
        (np.abs(search["zx|SteeringWheel"].to_numpy(dtype=float) - baseline) >= amp_thr)
        | (np.abs(search["steer_rate"].to_numpy(dtype=float)) >= rate_thr)
    )
    t_search = search["time_rel_s"].to_numpy(dtype=float)
    onset = np.nan
    for idx, flag in enumerate(cond):
        if not flag:
            continue
        before_start = max(0, idx - 20)
        before = cond[before_start:idx]
        if before.size and before.mean() > 0.25:
            continue
        onset = float(t_search[idx])
        break
    if not math.isfinite(onset):
        peak_rate = max_abs(search["steer_rate"].to_numpy(dtype=float))
        delta = signed_peak(search["zx|SteeringWheel"].to_numpy(dtype=float) - baseline)
        out["steer_delta_prepost"] = delta
        out["steer_rate_peak_near"] = peak_rate
        return out

    idx_anchor = nearest_time(t, anchor)
    idx_onset = nearest_time(t, onset)
    future = window(df, onset, min(onset + 1.2, anchor + 2.5))
    delta = signed_peak(future["zx|SteeringWheel"].to_numpy(dtype=float) - baseline) if len(future) else np.nan
    peak_rate = max_abs(future["steer_rate"].to_numpy(dtype=float)) if len(future) else np.nan
    score = 0.0
    if math.isfinite(abs(delta)):
        score += min(3.0, abs(delta) / max(amp_thr, 1e-6))
    if math.isfinite(peak_rate):
        score += min(3.0, peak_rate / max(rate_thr, 1e-6))
    relation = "before_condition" if onset < anchor - 0.2 else ("near_sync" if onset <= anchor + 0.2 else "after_condition")
    strength = "strong" if score >= 3.5 else ("medium" if score >= 2.2 else ("weak" if score >= 1.0 else "none"))
    input_start = float(anchor - PRE_WINDOW_S)
    input_start_window = window(df, input_start, input_start + 0.4)
    input_start_already = False
    if len(input_start_window):
        input_delta = max_abs(input_start_window["zx|SteeringWheel"].to_numpy(dtype=float) - baseline)
        input_rate = max_abs(input_start_window["steer_rate"].to_numpy(dtype=float))
        input_start_already = bool(input_delta >= amp_thr or input_rate >= rate_thr)
    out.update(
        {
            "t_steer_onset": onset,
            "steer_onset_relation": relation,
            "steer_response_delay_s": float(onset - anchor),
            "steer_delta_prepost": delta,
            "steer_rate_peak_near": peak_rate,
            "steer_response_strength": strength,
            "steer_response_score": float(score),
            "input_start_already_steering": input_start_already and idx_onset > idx_anchor,
        }
    )
    return out


def response_shape(df: pd.DataFrame, anchor: float) -> dict[str, Any]:
    out = {
        "t_response_peak": np.nan,
        "steer_peak_delta": np.nan,
        "has_return": False,
        "has_countersteer": False,
        "response_shape": "no_clear_steer",
    }
    if "zx|SteeringWheel" not in df.columns:
        return out
    pre = window(df, anchor - 1.0, anchor)
    post = window(df, anchor, anchor + POST_WINDOW_S)
    if len(post) < 10:
        return out
    baseline = robust_median(pre["zx|SteeringWheel"].to_numpy(dtype=float)) if len(pre) else robust_median(post["zx|SteeringWheel"].to_numpy(dtype=float))
    delta = post["zx|SteeringWheel"].to_numpy(dtype=float) - baseline
    if not np.isfinite(delta).any():
        return out
    idx_peak = int(np.nanargmax(np.abs(delta)))
    peak_delta = float(delta[idx_peak])
    t_peak = float(post["time_rel_s"].iloc[idx_peak])
    after = delta[idx_peak:]
    has_return = False
    has_counter = False
    if abs(peak_delta) > 1e-6 and after.size >= 5:
        end_abs = float(abs(after[-1]))
        has_return = end_abs <= abs(peak_delta) * 0.55
        has_counter = bool(np.nanmin(after) < -abs(peak_delta) * 0.25) if peak_delta > 0 else bool(np.nanmax(after) > abs(peak_delta) * 0.25)
    if has_counter:
        shape = "countersteer"
    elif has_return:
        shape = "return"
    elif abs(peak_delta) >= 3.0:
        shape = "sustained_same_side"
    else:
        shape = "weak_or_no_clear_steer"
    out.update(
        {
            "t_response_peak": t_peak,
            "steer_peak_delta": peak_delta,
            "has_return": has_return,
            "has_countersteer": has_counter,
            "response_shape": shape,
        }
    )
    return out


def coordinate_ok(df: pd.DataFrame, anchor: float) -> tuple[bool, float, float]:
    local = window(df, anchor - 1.0, anchor + 2.0)
    if len(local) < 5:
        return True, 0.0, 0.0
    steps = local["lateral_step_abs"].to_numpy(dtype=float)
    p99 = float(np.nanpercentile(steps[np.isfinite(steps)], 99)) if np.isfinite(steps).any() else 0.0
    max_step = max_abs(steps)
    return bool(max_step < 1.5 and p99 < 0.8), p99, max_step


def condition_context(row: dict[str, Any]) -> str:
    if row.get("is_low_mu_context"):
        return "低附着"
    if row.get("is_curve_context"):
        return "弯道/曲率"
    if row.get("is_roll_context"):
        return "横滚/姿态"
    if row.get("is_lateral_dynamic_context"):
        return "横向动态"
    if row.get("is_brake_context"):
        return "制动/减速"
    return "综合高动态"


def classify_episode(row: dict[str, Any]) -> tuple[str, str, str, str]:
    if not row.get("window_complete", False) or not row.get("coordinate_continuity_ok", True):
        return "excluded", "排除样本", "窗口不完整或坐标连续性异常", "排除"
    cond_level = row.get("condition_level", "weak")
    steer_strength = row.get("steer_response_strength", "none")
    relation = row.get("steer_onset_relation", "none")
    shape = row.get("response_shape", "no_clear_steer")
    brake = bool(row.get("has_brake_response", False))
    is_normal_curve = bool(row.get("is_curve_context", False)) and cond_level in {"weak", "medium"} and steer_strength in {"weak", "medium"} and not bool(row.get("roll_evidence", False))
    if is_normal_curve:
        return "normal_control", "正常驾驶/普通弯道对照", "更像普通弯道或平滑转向", "正常对照"
    if steer_strength in {"strong", "medium"} and cond_level in {"strong", "extreme"}:
        if relation == "after_condition":
            return "delayed_or_no_steer", "延迟或无明显转向响应", "工况先出现，方向盘响应偏晚", "响应时序分析"
        if shape in {"countersteer", "return", "sustained_same_side", "weak_or_no_clear_steer"}:
            return "strong_response", "强响应型极限工况", "工况明确且方向盘响应较强", "轨迹预测/强响应分析"
    if cond_level in {"strong", "extreme"} and steer_strength in {"weak", "none"}:
        if brake:
            return "weak_or_conservative", "弱响应/保守响应", "工况明确但转向弱，可能以制动或保持为主", "风格/保守响应分析"
        return "delayed_or_no_steer", "延迟或无明显转向响应", "工况明确但未见明显方向盘响应", "响应类型分类"
    if cond_level in {"medium", "strong"} and steer_strength in {"weak", "medium"}:
        return "weak_or_conservative", "弱响应/保守响应", "存在工况压力但驾驶员响应不激烈", "风格/弱响应分析"
    return "manual_review", "待人工复核", "工况或响应强度边界不清", "人工复核"


def normal_control_rows(
    df: pd.DataFrame,
    meta: dict[str, Any],
    path: Path,
    contexts: dict[str, list[dict[str, Any]]],
    condition_mask: np.ndarray,
    condition_score: np.ndarray,
    max_per_record: int = 2,
) -> list[dict[str, Any]]:
    t = df["time_rel_s"].to_numpy(dtype=float)
    if t.size < 20:
        return []
    start = float(t[0] + PRE_WINDOW_S + 1.0)
    end = float(t[-1] - POST_WINDOW_S - 1.0)
    if end <= start:
        return []
    candidate_times = np.arange(start, end, 20.0)
    rows: list[dict[str, Any]] = []
    for anchor in candidate_times:
        if len(rows) >= max_per_record:
            break
        idx = nearest_time(t, float(anchor))
        near = (t >= anchor - 4.0) & (t <= anchor + 6.0)
        if near.sum() < 20:
            continue
        if bool(condition_mask[near].any()):
            continue
        local_score = float(np.nanmax(condition_score[near])) if np.isfinite(condition_score[near]).any() else 0.0
        if local_score >= 2.0:
            continue
        local = window(df, anchor - 1.0, anchor + 3.0)
        if len(local) < 10:
            continue
        speed = robust_median(local["zx1|v_km/h"].to_numpy(dtype=float)) if "zx1|v_km/h" in local.columns else np.nan
        if math.isfinite(speed) and speed < 15.0:
            continue
        coord_ok, lat_p99, lat_max = coordinate_ok(df, anchor)
        if not coord_ok:
            continue
        steer_info = detect_steer_onset(df, anchor)
        shape_info = response_shape(df, anchor)
        row: dict[str, Any] = {
            "episode_uid": f"V03_{meta['subject']}_{meta['session_stamp']}_N{len(rows):03d}",
            "dataset_candidate_version": "extreme_condition_episodes_v0_3_all_raw",
            **meta,
            "event_index_in_session": -1 - len(rows),
            "t_condition_anchor": float(anchor),
            "t_condition_end": float(anchor),
            "t_condition_peak": float(anchor),
            "condition_duration_s": 0.0,
            "condition_score_peak": local_score,
            "condition_score_mean": local_score,
            "condition_level": "normal",
            "condition_context_cn": "普通驾驶对照",
            "window_complete": True,
            "recommended_input_start_s": float(anchor - PRE_WINDOW_S),
            "recommended_input_end_s": float(anchor),
            "recommended_label_start_s": float(anchor),
            "recommended_label_end_s": float(anchor + POST_WINDOW_S),
            "vehicle_component_count": 0,
            "roll_evidence": False,
            "lateral_evidence": False,
            "is_roll_context": False,
            "is_lateral_dynamic_context": False,
            "is_curve_context": bool(max_abs(local["curvature_selected"].to_numpy(dtype=float)) >= 0.0015)
            if "curvature_selected" in local.columns
            else False,
            "is_low_mu_context": False,
            "is_brake_context": False,
            "has_brake_response": False,
            "coordinate_continuity_ok": coord_ok,
            "local_lateral_step_p99": lat_p99,
            "local_lateral_step_max": lat_max,
            "peak_abs_ay_window": max_abs(local["zx|ay"].to_numpy(dtype=float)) if "zx|ay" in local.columns else np.nan,
            "peak_abs_yaw_rate_window": max_abs(local["zx|vyaw"].to_numpy(dtype=float)) if "zx|vyaw" in local.columns else np.nan,
            "peak_abs_roll_rate_window": max_abs(local["zx|vroll"].to_numpy(dtype=float)) if "zx|vroll" in local.columns else np.nan,
            "peak_abs_roll_window": max_abs(local["zx|roll"].to_numpy(dtype=float)) if "zx|roll" in local.columns else np.nan,
            "peak_abs_curvature_window": max_abs(local["curvature_selected"].to_numpy(dtype=float)) if "curvature_selected" in local.columns else np.nan,
            "min_mu_window": float(np.nanmin(local["zx1|mu"].to_numpy(dtype=float))) if "zx1|mu" in local.columns and np.isfinite(local["zx1|mu"]).any() else np.nan,
            "median_speed_kmh_window": speed,
            **steer_info,
            **shape_info,
            **nearest_context(path, float(anchor), contexts),
            "v0_3_category": "normal_control",
            "v0_3_category_cn": "正常驾驶/普通弯道对照",
            "v0_3_reason_cn": "远离高动态工况的普通驾驶对照窗口",
            "recommended_use": "正常对照",
        }
        rows.append(row)
    return rows


def load_context_rows() -> dict[str, list[dict[str, Any]]]:
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for table in OPTIONAL_CONTEXT_TABLES:
        if not table.exists():
            continue
        try:
            df = pd.read_csv(table, encoding="utf-8-sig", low_memory=False)
        except Exception:
            continue
        path_col = "vehicle_raw_absolute_path" if "vehicle_raw_absolute_path" in df.columns else None
        time_col = "anchor_time_rel_s" if "anchor_time_rel_s" in df.columns else None
        if time_col is None and "recommended_vehicle_anchor_s" in df.columns:
            time_col = "recommended_vehicle_anchor_s"
        if path_col is None or time_col is None:
            continue
        for _, row in df.iterrows():
            path = str(row.get(path_col, ""))
            t = safe_float(row.get(time_col))
            if not path or not math.isfinite(t):
                continue
            contexts[path].append(
                {
                    "source_table": str(table.name),
                    "time": t,
                    "road_design_module_name": str(row.get("road_design_module_name", "")),
                    "road_design_risk_class": str(row.get("road_design_risk_class", "")),
                    "source_event_types": str(row.get("source_event_types", "")),
                    "v0_2_category": str(row.get("v0_2_category", "")),
                    "v0_2_category_cn": str(row.get("v0_2_category_cn", "")),
                }
            )
    for rows in contexts.values():
        rows.sort(key=lambda item: item["time"])
    return contexts


def nearest_context(path: Path, anchor: float, contexts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = contexts.get(str(path), [])
    if not rows:
        return {
            "nearest_old_context_table": "",
            "nearest_old_context_time_s": np.nan,
            "delta_to_nearest_old_context_s": np.nan,
            "nearest_old_road_module": "",
            "nearest_old_risk_class": "",
            "nearest_old_event_types": "",
            "nearest_old_v0_2_category": "",
            "nearest_old_v0_2_category_cn": "",
        }
    times = np.array([r["time"] for r in rows], dtype=float)
    idx = nearest_time(times, anchor)
    r = rows[idx]
    return {
        "nearest_old_context_table": r.get("source_table", ""),
        "nearest_old_context_time_s": r.get("time", np.nan),
        "delta_to_nearest_old_context_s": float(anchor - r.get("time", np.nan)),
        "nearest_old_road_module": r.get("road_design_module_name", ""),
        "nearest_old_risk_class": r.get("road_design_risk_class", ""),
        "nearest_old_event_types": r.get("source_event_types", ""),
        "nearest_old_v0_2_category": r.get("v0_2_category", ""),
        "nearest_old_v0_2_category_cn": r.get("v0_2_category_cn", ""),
    }


def build_condition_masks(df: pd.DataFrame) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    t = df["time_rel_s"].to_numpy(dtype=float)
    dt = np.nanmedian(np.diff(t)) if len(t) > 2 else 0.005
    smooth_width = max(1, int(round(SMOOTH_S / max(float(dt), 1e-3))))
    masks: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    strengths: dict[str, np.ndarray] = {}
    for name, spec in SIGNAL_SPECS.items():
        col = spec.col
        if name == "curvature":
            col = "curvature_selected"
        if col not in df.columns:
            arr = np.zeros(len(df), dtype=float)
        else:
            arr = df[col].to_numpy(dtype=float)
        sm = moving_average(arr, smooth_width)
        thr = robust_threshold(sm, spec.floor, q=spec.q)
        if not math.isfinite(thr) or thr <= 0:
            thr = spec.floor
        masks[name] = np.abs(sm) >= thr
        thresholds[f"{name}_threshold"] = thr
        strengths[name] = np.abs(sm) / max(thr, 1e-6)
    mu = df["zx1|mu"].to_numpy(dtype=float) if "zx1|mu" in df.columns else np.full(len(df), np.nan)
    speed = df["zx1|v_km/h"].to_numpy(dtype=float) if "zx1|v_km/h" in df.columns else np.full(len(df), np.nan)
    brake = df["zx|BrakePedal"].to_numpy(dtype=float) if "zx|BrakePedal" in df.columns else np.full(len(df), np.nan)
    ax = df["zx|ax"].to_numpy(dtype=float) if "zx|ax" in df.columns else np.full(len(df), np.nan)
    mu_mask = np.isfinite(mu) & (mu < 0.85)
    brake_thr = robust_threshold(brake, 5.0, q=95.0) if np.isfinite(brake).any() else np.nan
    brake_mask = np.isfinite(brake) & (brake >= max(5.0, brake_thr if math.isfinite(brake_thr) else 5.0))
    decel_mask = np.isfinite(ax) & (ax <= -1.5)
    speed_mask = np.isfinite(speed) & (speed >= max(45.0, np.nanpercentile(speed[np.isfinite(speed)], 75) if np.isfinite(speed).any() else 45.0))
    dynamic_components = (
        masks["ay"].astype(int)
        + masks["yaw_rate"].astype(int)
        + masks["roll_rate"].astype(int)
        + masks["roll_angle"].astype(int)
    )
    condition_mask = (
        (dynamic_components >= 2)
        | (masks["roll_rate"] & (masks["ay"] | masks["yaw_rate"] | masks["curvature"]))
        | (mu_mask & (masks["ay"] | masks["yaw_rate"] | masks["roll_rate"] | speed_mask))
        | (masks["curvature"] & (masks["ay"] | masks["yaw_rate"] | masks["roll_rate"]))
        | ((brake_mask | decel_mask) & (masks["ay"] | masks["yaw_rate"] | masks["roll_rate"]))
    )
    score = (
        np.minimum(strengths["ay"], 3.0)
        + np.minimum(strengths["yaw_rate"], 3.0)
        + np.minimum(strengths["roll_rate"], 3.0)
        + np.minimum(strengths["roll_angle"], 2.0)
        + np.minimum(strengths["curvature"], 2.0) * 0.5
        + mu_mask.astype(float) * 1.0
        + brake_mask.astype(float) * 0.5
        + decel_mask.astype(float) * 0.5
    )
    extras: dict[str, Any] = thresholds
    extras.update(
        {
            "mu_mask": mu_mask,
            "brake_mask": brake_mask,
            "decel_mask": decel_mask,
            "speed_mask": speed_mask,
            "dynamic_component_count": dynamic_components,
            "condition_score_series": score,
            "component_masks": masks,
        }
    )
    return extras, condition_mask, score


def level_from_score(score: float) -> str:
    if score >= 7.0:
        return "extreme"
    if score >= 4.5:
        return "strong"
    if score >= 2.5:
        return "medium"
    return "weak"


def scan_record(path: Path, contexts: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if not path.name.endswith("_vehicle.csv"):
        subject, session_stamp = session_from_path(path)
        meta = {
            "subject": subject,
            "session_stamp": session_stamp,
            "vehicle_raw_absolute_path": str(path),
            "vehicle_raw_relative_path": str(path.relative_to(RAW_VEHICLE_ROOT)).replace("\\", "/")
            if path.is_relative_to(RAW_VEHICLE_ROOT)
            else path.name,
            "vehicle_raw_size_bytes": path.stat().st_size if path.exists() else 0,
            "vehicle_raw_sha256": "",
            "read_status": "skipped_non_subject_csv",
            "read_error": "文件名不是 *_vehicle.csv，作为非被试车辆记录跳过",
            "row_count": 0,
            "duration_s": np.nan,
            "episode_count": 0,
            "success_episode_count": 0,
        }
        excluded = [
            {
                "episode_uid": f"X_{subject}_{session_stamp}_non_subject_csv",
                **meta,
                "t_condition_anchor": np.nan,
                "v0_3_category": "excluded",
                "v0_3_category_cn": "排除样本",
                "v0_3_reason_cn": "非被试车辆记录 CSV，不参与驾驶员 episode 筛选",
                "recommended_use": "排除",
            }
        ]
        return [], meta, excluded
    df, meta = load_vehicle_csv(path)
    file_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    if df is None:
        meta.update(
            {
                "row_count": 0,
                "duration_s": np.nan,
                "episode_count": 0,
                "success_episode_count": 0,
            }
        )
        excluded_rows.append(
            {
                "episode_uid": f"X_{meta['subject']}_{meta['session_stamp']}_file",
                **meta,
                "t_condition_anchor": np.nan,
                "v0_3_category": "excluded",
                "v0_3_category_cn": "排除样本",
                "v0_3_reason_cn": f"文件读取失败：{meta.get('read_error', '')}",
                "recommended_use": "排除",
            }
        )
        return file_rows, meta, excluded_rows
    t = df["time_rel_s"].to_numpy(dtype=float)
    duration = float(np.nanmax(t) - np.nanmin(t)) if len(t) else np.nan
    meta.update(
        {
            "row_count": len(df),
            "duration_s": duration,
            "time_start_s": float(np.nanmin(t)) if len(t) else np.nan,
            "time_end_s": float(np.nanmax(t)) if len(t) else np.nan,
            "sampling_rate_median_hz": float(1.0 / np.nanmedian(np.diff(t))) if len(t) > 2 and np.nanmedian(np.diff(t)) > 0 else np.nan,
            "episode_count": 0,
            "success_episode_count": 0,
        }
    )
    if duration < PRE_WINDOW_S + POST_WINDOW_S + 1.0:
        meta["read_status"] = "too_short"
        return file_rows, meta, excluded_rows
    extras, condition_mask, condition_score = build_condition_masks(df)
    segments = contiguous_segments(condition_mask, t, CONDITION_MIN_DUR_S, EPISODE_MIN_GAP_S)
    meta["episode_count"] = len(segments)
    for event_idx, (a, b) in enumerate(segments):
        seg_t = t[a : b + 1]
        seg_score = condition_score[a : b + 1]
        if seg_t.size == 0:
            continue
        peak_idx_rel = int(np.nanargmax(seg_score)) if np.isfinite(seg_score).any() else 0
        peak_idx = a + peak_idx_rel
        anchor = float(t[a])
        peak_time = float(t[peak_idx])
        score_peak = float(condition_score[peak_idx]) if math.isfinite(float(condition_score[peak_idx])) else 0.0
        score_mean = float(np.nanmean(seg_score)) if np.isfinite(seg_score).any() else 0.0
        w_pre = anchor - PRE_WINDOW_S
        w_post = anchor + POST_WINDOW_S
        win_complete = bool(t[0] <= w_pre and t[-1] >= w_post)
        coord_ok, lat_p99, lat_max = coordinate_ok(df, anchor)
        local = window(df, anchor - 1.0, anchor + 3.0)
        if len(local) < 10:
            continue
        component_masks = extras["component_masks"]
        component_count_local = int(np.nanmax(extras["dynamic_component_count"][a : b + 1])) if b >= a else 0
        roll_evidence = bool(component_masks["roll_rate"][a : b + 1].any() or component_masks["roll_angle"][a : b + 1].any())
        lateral_evidence = bool(component_masks["ay"][a : b + 1].any() or component_masks["yaw_rate"][a : b + 1].any())
        curve_context = bool(component_masks["curvature"][a : b + 1].any())
        low_mu_context = bool(extras["mu_mask"][a : b + 1].any())
        brake_context = bool(extras["brake_mask"][a : b + 1].any() or extras["decel_mask"][a : b + 1].any())
        steer_info = detect_steer_onset(df, anchor)
        shape_info = response_shape(df, anchor)
        row: dict[str, Any] = {
            "episode_uid": f"V03_{meta['subject']}_{meta['session_stamp']}_{event_idx:04d}",
            "dataset_candidate_version": "extreme_condition_episodes_v0_3_all_raw",
            **meta,
            "event_index_in_session": event_idx,
            "t_condition_anchor": anchor,
            "t_condition_end": float(t[b]),
            "t_condition_peak": peak_time,
            "condition_duration_s": float(t[b] - t[a]),
            "condition_score_peak": score_peak,
            "condition_score_mean": score_mean,
            "condition_level": level_from_score(score_peak),
            "condition_context_cn": "",
            "window_complete": win_complete,
            "recommended_input_start_s": w_pre,
            "recommended_input_end_s": anchor,
            "recommended_label_start_s": anchor,
            "recommended_label_end_s": w_post,
            "vehicle_component_count": component_count_local,
            "roll_evidence": roll_evidence,
            "lateral_evidence": lateral_evidence,
            "is_roll_context": roll_evidence,
            "is_lateral_dynamic_context": lateral_evidence,
            "is_curve_context": curve_context,
            "is_low_mu_context": low_mu_context,
            "is_brake_context": brake_context,
            "has_brake_response": brake_context,
            "coordinate_continuity_ok": coord_ok,
            "local_lateral_step_p99": lat_p99,
            "local_lateral_step_max": lat_max,
            "peak_abs_ay_window": max_abs(local["zx|ay"].to_numpy(dtype=float)) if "zx|ay" in local.columns else np.nan,
            "peak_abs_yaw_rate_window": max_abs(local["zx|vyaw"].to_numpy(dtype=float)) if "zx|vyaw" in local.columns else np.nan,
            "peak_abs_roll_rate_window": max_abs(local["zx|vroll"].to_numpy(dtype=float)) if "zx|vroll" in local.columns else np.nan,
            "peak_abs_roll_window": max_abs(local["zx|roll"].to_numpy(dtype=float)) if "zx|roll" in local.columns else np.nan,
            "peak_abs_curvature_window": max_abs(local["curvature_selected"].to_numpy(dtype=float)) if "curvature_selected" in local.columns else np.nan,
            "min_mu_window": float(np.nanmin(local["zx1|mu"].to_numpy(dtype=float))) if "zx1|mu" in local.columns and np.isfinite(local["zx1|mu"]).any() else np.nan,
            "median_speed_kmh_window": robust_median(local["zx1|v_km/h"].to_numpy(dtype=float)) if "zx1|v_km/h" in local.columns else np.nan,
            **{k: v for k, v in extras.items() if k.endswith("_threshold")},
            **steer_info,
            **shape_info,
        }
        row["condition_context_cn"] = condition_context(row)
        row.update(nearest_context(path, anchor, contexts))
        cat, cat_cn, reason, use = classify_episode(row)
        row.update(
            {
                "v0_3_category": cat,
                "v0_3_category_cn": cat_cn,
                "v0_3_reason_cn": reason,
                "recommended_use": use,
            }
        )
        file_rows.append(row)
    file_rows.extend(normal_control_rows(df, meta, path, contexts, condition_mask, condition_score))
    meta["success_episode_count"] = len(file_rows)
    return file_rows, meta, excluded_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def simple_markdown_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "暂无记录。"
    view = df.head(max_rows).copy()
    cols = [str(c) for c in view.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for col in view.columns:
            value = row[col]
            if isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            text = text.replace("\n", " ").replace("|", "/")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        lines.append(f"\n仅显示前 {max_rows} 行，共 {len(df)} 行。")
    return "\n".join(lines)


def choose_panel_rows(df: pd.DataFrame) -> pd.DataFrame:
    selected = []
    if df.empty:
        return df
    for cat, limit in MAX_PANEL_PER_CLASS.items():
        part = df[df["v0_3_category"] == cat].copy()
        if part.empty:
            continue
        if cat == "manual_review":
            part["panel_rank"] = (part["condition_score_peak"].astype(float) - 4.0).abs() + part["steer_response_score"].astype(float).fillna(0.0) * 0.1
            part = part.sort_values("panel_rank", ascending=True)
        else:
            part = part.sort_values(["condition_score_peak", "steer_response_score"], ascending=False)
        selected.append(part.head(limit))
    if not selected:
        return df.head(0)
    return pd.concat(selected, ignore_index=True)


def plot_panel(row: pd.Series, out_path: Path) -> None:
    path = Path(str(row["vehicle_raw_absolute_path"]))
    df, _ = load_vehicle_csv(path)
    if df is None:
        return
    anchor = safe_float(row["t_condition_anchor"])
    start = anchor - 4.0
    end = anchor + 6.0
    view = window(df, start, end)
    if view.empty:
        return
    t = view["time_rel_s"].to_numpy(dtype=float) - anchor
    fig, axes = plt.subplots(8, 1, figsize=(12, 13), sharex=True)
    signals = [
        ("zx|SteeringWheel", "方向盘角"),
        ("steer_rate", "方向盘角速度"),
        ("zx1|v_km/h", "车速"),
        ("zx|BrakePedal", "制动踏板"),
        ("zx|ay", "横向加速度"),
        ("zx|vyaw", "横摆角速度"),
        ("zx|vroll", "横滚角速度"),
        ("zx|roll", "横滚角"),
    ]
    for ax, (col, label) in zip(axes, signals):
        if col in view.columns:
            ax.plot(t, view[col].to_numpy(dtype=float), lw=1.2)
        ax.axvline(0.0, color="crimson", ls="--", lw=1.0, label="工况锚点" if ax is axes[0] else None)
        t_steer = safe_float(row.get("t_steer_onset"))
        if math.isfinite(t_steer):
            ax.axvline(t_steer - anchor, color="royalblue", ls="--", lw=1.0, label="方向盘启动" if ax is axes[0] else None)
        t_peak = safe_float(row.get("t_response_peak"))
        if math.isfinite(t_peak):
            ax.axvline(t_peak - anchor, color="forestgreen", ls=":", lw=1.0, label="响应峰值" if ax is axes[0] else None)
        old_t = safe_float(row.get("nearest_old_context_time_s"))
        if math.isfinite(old_t) and abs(old_t - anchor) <= 5.0:
            ax.axvline(old_t - anchor, color="darkorange", ls="-.", lw=0.9, label="旧候选/上下文" if ax is axes[0] else None)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    title = (
        f"{row['episode_uid']} | {row['v0_3_category_cn']} | {row['condition_context_cn']} | "
        f"{row['subject']} | score={safe_float(row['condition_score_peak']):.2f}"
    )
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("相对工况锚点时间 / s")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_reports(all_df: pd.DataFrame, file_df: pd.DataFrame, panel_df: pd.DataFrame) -> None:
    summary_rows = []
    total = len(all_df)
    for cat, part in all_df.groupby("v0_3_category", dropna=False):
        summary_rows.append(
            {
                "v0_3_category": cat,
                "v0_3_category_cn": part["v0_3_category_cn"].iloc[0] if len(part) else "",
                "count": len(part),
                "ratio": ratio_bool(len(part), total),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("count", ascending=False)
    summary_df.to_csv(TABLE_DIR / "extreme_condition_episode_summary_v0_3.csv", index=False, encoding="utf-8-sig")

    if not all_df.empty:
        by_subject = (
            all_df.groupby(["subject", "v0_3_category_cn"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["subject", "count"], ascending=[True, False])
        )
        by_subject.to_csv(TABLE_DIR / "extreme_condition_episode_by_subject_v0_3.csv", index=False, encoding="utf-8-sig")
        by_context = (
            all_df.groupby(["condition_context_cn", "v0_3_category_cn"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["condition_context_cn", "count"], ascending=[True, False])
        )
        by_context.to_csv(TABLE_DIR / "extreme_condition_episode_by_context_v0_3.csv", index=False, encoding="utf-8-sig")
        by_old_module = (
            all_df.groupby(["nearest_old_road_module", "v0_3_category_cn"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["nearest_old_road_module", "count"], ascending=[True, False])
        )
        by_old_module.to_csv(TABLE_DIR / "extreme_condition_episode_by_old_module_context_v0_3.csv", index=False, encoding="utf-8-sig")

    fig_path = FIG_DIR / "extreme_condition_category_counts_v0_3.png"
    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_df = summary_df.copy()
        ax.bar(plot_df["v0_3_category_cn"], plot_df["count"], color="#4C78A8")
        ax.set_ylabel("样本数")
        ax.set_title("v0.3 全量原始数据极限/近极限工况 episode 分类")
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)

    report = REPORT_DIR / "stage02_extreme_condition_episode_v0_3_user_summary_cn.md"
    total_files = len(file_df)
    ok_files = int((file_df["read_status"] == "ok").sum()) if "read_status" in file_df else 0
    skipped_non_subject = int((file_df["read_status"] == "skipped_non_subject_csv").sum()) if "read_status" in file_df else 0
    too_short_files = int((file_df["read_status"] == "too_short").sum()) if "read_status" in file_df else 0
    report.write_text(
        "\n".join(
            [
                "# 全量原始数据极限工况 episode 重筛 v0.3（用户查看版）",
                "",
                f"生成时间：{now_text()}",
                "",
                "## 这次和 v0.2 最大区别",
                "",
                "- 本次入口是 `原始车辆数据` 下的所有原始车辆 CSV，不再从旧 v0.2/v0.5/v0.6 候选表继续筛。",
                "- 旧候选表只作为最近上下文贴回，用来解释当前 episode 是否靠近旧锚点或旧道路模块。",
                "- 不再要求事件后一定出现明显回正或反打；弱响应、保守响应、延迟响应和无明显转向都保留下来。",
                "",
                "## 全量扫描情况",
                "",
                f"- 扫描 CSV 文件数：{total_files}",
                f"- 成功读取车辆记录数：{ok_files}",
                f"- 非被试 CSV 跳过：{skipped_non_subject}",
                f"- 记录过短跳过：{too_short_files}",
                f"- 检测到 episode 总数：{total}",
                "",
                "## episode 分类结果",
                "",
                simple_markdown_table(summary_df),
                "",
                "## 当前解释边界",
                "",
                "- 这一步仍然不是模型训练结果，只是重新定义样本库。",
                "- 强响应样本适合后续轨迹预测试验；弱响应/保守样本更适合驾驶风格和生理状态差异分析。",
                "- 如果车辆-only 基线在这套新样本上仍然出现方向错侧、幅值压缩或预测图物理意义不对，需要继续回到样本和锚点规则，而不是马上解释生理数据。",
                "",
                "## 推荐优先查看",
                "",
                f"- 总表：`{TABLE_DIR / 'extreme_condition_episodes_all_v0_3.csv'}`",
                f"- 强响应：`{TABLE_DIR / 'strong_response_episodes_v0_3.csv'}`",
                f"- 弱/保守响应：`{TABLE_DIR / 'weak_or_conservative_response_episodes_v0_3.csv'}`",
                f"- 延迟/无明显转向：`{TABLE_DIR / 'delayed_or_no_steer_response_episodes_v0_3.csv'}`",
                f"- 复核图索引：`{TABLE_DIR / 'extreme_condition_review_panel_index_v0_3.csv'}`",
                f"- 复核图目录：`{PANEL_DIR}`",
            ]
        ),
        encoding="utf-8",
    )

    tech_report = REPORT_DIR / "extreme_condition_episode_v0_3_cn.md"
    tech_report.write_text(
        "\n".join(
            [
                "# v0.3 全量原始数据极限工况 episode 技术说明",
                "",
                "本流程直接遍历原始车辆 CSV，以车辆状态、道路曲率、低附着、横向动态、横摆、横滚、制动等信号生成极限/近极限工况候选。",
                "",
                "## 关键规则",
                "",
                "- 工况锚点来自当前记录内的车辆/道路状态异常段起点。",
                "- 方向盘启动、车辆响应峰值、是否回正/反打只作为响应描述字段。",
                "- 保守或弱响应不会因为没有明显纠正而被排除。",
                "- 旧候选表只用于最近上下文贴回，不作为筛选入口。",
                "",
                "## 文件级扫描摘要",
                "",
                simple_markdown_table(file_df, max_rows=200),
            ]
        ),
        encoding="utf-8",
    )

    if ARTIFACT_INDEX.exists():
        with ARTIFACT_INDEX.open("a", encoding="utf-8") as f:
            f.write(
                "\n\n## v0.3 全量原始数据极限工况 episode 重筛\n\n"
                f"- 用户查看版报告：`{report}`\n"
                f"- 技术报告：`{tech_report}`\n"
                f"- 总表：`{TABLE_DIR / 'extreme_condition_episodes_all_v0_3.csv'}`\n"
                f"- 复核图索引：`{TABLE_DIR / 'extreme_condition_review_panel_index_v0_3.csv'}`\n"
                f"- 分类统计图：`{fig_path}`\n"
            )
    with DAILY_LOG.open("a", encoding="utf-8") as f:
        f.write(
            f"\n\n## {now_text()} v0.3 全量原始数据极限工况 episode 重筛\n\n"
            f"- 扫描入口：`{RAW_VEHICLE_ROOT}`\n"
            f"- 扫描 CSV：{total_files} 个，成功读取：{ok_files} 个。\n"
            f"- 检测 episode：{total} 个。\n"
            "- 本轮不训练模型，先输出样本库、分类表、复核图和中文报告。\n"
        )


def main() -> None:
    ensure_dirs()
    contexts = load_context_rows()
    vehicle_paths = sorted(RAW_VEHICLE_ROOT.rglob("*.csv"))
    all_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    excluded_file_rows: list[dict[str, Any]] = []
    log_lines = [f"{now_text()} start v0.3 all raw scan", f"raw_vehicle_root={RAW_VEHICLE_ROOT}", f"csv_count={len(vehicle_paths)}"]
    for idx, path in enumerate(vehicle_paths, start=1):
        rows, meta, excluded = scan_record(path, contexts)
        all_rows.extend(rows)
        file_rows.append(meta)
        excluded_file_rows.extend(excluded)
        log_lines.append(
            f"[{idx}/{len(vehicle_paths)}] {path.name} status={meta.get('read_status')} rows={meta.get('row_count')} episodes={len(rows)} error={meta.get('read_error', '')}"
        )
    all_df = pd.DataFrame(all_rows)
    file_df = pd.DataFrame(file_rows)
    file_df.to_csv(TABLE_DIR / "raw_vehicle_file_scan_report_v0_3.csv", index=False, encoding="utf-8-sig")
    if all_df.empty:
        all_df.to_csv(TABLE_DIR / "extreme_condition_episodes_all_v0_3.csv", index=False, encoding="utf-8-sig")
        (LOG_DIR / "build_extreme_condition_episodes_v0_3.log").write_text("\n".join(log_lines), encoding="utf-8")
        make_reports(all_df, file_df, pd.DataFrame())
        return
    all_df = all_df.sort_values(["subject", "session_stamp", "t_condition_anchor"]).reset_index(drop=True)
    all_df.to_csv(TABLE_DIR / "extreme_condition_episodes_all_v0_3.csv", index=False, encoding="utf-8-sig")
    category_outputs = {
        "strong_response": "strong_response_episodes_v0_3.csv",
        "weak_or_conservative": "weak_or_conservative_response_episodes_v0_3.csv",
        "delayed_or_no_steer": "delayed_or_no_steer_response_episodes_v0_3.csv",
        "normal_control": "normal_driving_controls_v0_3.csv",
        "manual_review": "manual_review_episodes_v0_3.csv",
        "excluded": "excluded_episodes_v0_3.csv",
    }
    for cat, filename in category_outputs.items():
        all_df[all_df["v0_3_category"] == cat].to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")
    if excluded_file_rows:
        pd.DataFrame(excluded_file_rows).to_csv(TABLE_DIR / "excluded_file_level_rows_v0_3.csv", index=False, encoding="utf-8-sig")

    panel_df = choose_panel_rows(all_df)
    panel_rows = []
    for _, row in panel_df.iterrows():
        safe_uid = str(row["episode_uid"]).replace("/", "_").replace("\\", "_")
        out_path = PANEL_DIR / f"{row['v0_3_category']}_{safe_uid}.png"
        try:
            if not out_path.exists():
                plot_panel(row, out_path)
            panel_rows.append({**row.to_dict(), "figure_path": str(out_path)})
        except Exception as exc:
            panel_rows.append({**row.to_dict(), "figure_path": "", "plot_error": repr(exc)})
    panel_out = pd.DataFrame(panel_rows)
    panel_out.to_csv(TABLE_DIR / "extreme_condition_review_panel_index_v0_3.csv", index=False, encoding="utf-8-sig")
    make_reports(all_df, file_df, panel_out)
    (LOG_DIR / "build_extreme_condition_episodes_v0_3.log").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"done: files={len(vehicle_paths)} episodes={len(all_df)} out={OUT_DIR}")


if __name__ == "__main__":
    main()
