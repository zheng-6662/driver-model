# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from collections import Counter
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
INPUT_CANDIDATES = (
    ROOT
    / "02_samples"
    / "vehicle_instability_all_raw_rescreen_v0_1"
    / "tables"
    / "all_raw_vehicle_instability_candidates_v0_1.csv"
)
OUT_DIR = ROOT / "02_samples" / "vehicle_response_presteer_rescreen_v0_2"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
PANEL_DIR = FIG_DIR / "review_panels"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-18.md"

VEHICLE_COLS = [
    "StorageTime",
    "zx|SteeringWheel",
    "zx|ay",
    "zx|vyaw",
    "zx|vroll",
    "zx|roll",
    "zx1|lateraldistance",
    "zx|BrakePedal",
    "zx|ax",
    "zx1|v_km/h",
    "zx1|mu",
    "zx1|lanecurvatureXY",
]

DYNAMIC_SPECS = {
    "ay": {"col": "zx|ay", "floor": 0.35, "label": "横向加速度"},
    "yaw_rate": {"col": "zx|vyaw", "floor": 0.035, "label": "横摆角速度"},
    "roll_rate": {"col": "zx|vroll", "floor": 0.030, "label": "横滚角速度"},
    "roll_angle": {"col": "zx|roll", "floor": 0.020, "label": "横滚角"},
}

CORE_MODULES = {"differentmu_road", "fix_road", "curve1", "curve2"}
CONTINUOUS_OR_CONTEXT_MODULES = {"middle_section", "longstraight", "stop"}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, PANEL_DIR, REPORT_DIR, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)
    for old in PANEL_DIR.glob("*.png"):
        old.unlink()


def finite_float(value: Any, default: float = float("nan")) -> float:
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
    if "zx1|lateraldistance" in df.columns:
        lat = df["zx1|lateraldistance"].to_numpy(dtype=float)
        t = df["time_rel_s"].to_numpy(dtype=float)
        df["lateral_step_abs"] = np.r_[0.0, np.abs(np.diff(lat))]
        df["lateral_velocity"] = gradient(lat, t)
    else:
        df["lateral_step_abs"] = 0.0
        df["lateral_velocity"] = 0.0
    return df.reset_index(drop=True)


def window(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return df[(df["time_rel_s"] >= start) & (df["time_rel_s"] <= end)].copy()


def first_sustained_time(times: np.ndarray, cond: np.ndarray, min_duration_s: float) -> tuple[float, bool]:
    times = np.asarray(times, dtype=float)
    cond = np.asarray(cond, dtype=bool)
    if times.size == 0:
        return float("nan"), False
    dts = np.diff(times)
    dts = dts[np.isfinite(dts) & (dts > 0)]
    dt = float(np.nanmedian(dts)) if dts.size else 0.01
    need = max(3, int(round(min_duration_s / max(dt, 1e-3))))
    run = 0
    start_idx = 0
    for idx, flag in enumerate(cond):
        if flag:
            if run == 0:
                start_idx = idx
            run += 1
            if run >= need:
                return float(times[start_idx]), start_idx == 0
        else:
            run = 0
    return float("nan"), False


def max_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(np.abs(arr)))


def percentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanpercentile(arr, q))


def future_max_abs(values: np.ndarray, times: np.ndarray, horizon_s: float, stop_time: float) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    abs_values = np.abs(values)
    for idx, t in enumerate(times):
        if not math.isfinite(float(t)):
            continue
        end_time = min(float(t) + horizon_s, stop_time)
        right = np.searchsorted(times, end_time, side="right")
        seg = abs_values[idx:right]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[idx] = float(np.nanmax(seg))
    return out


def detect_vehicle_response(row: pd.Series, vehicle: pd.DataFrame) -> dict[str, Any]:
    anchor = finite_float(row.get("anchor_time_rel_s"))
    result: dict[str, Any] = {
        "audit_status": "ok",
        "source_vehicle_anchor_s": anchor,
    }
    if not math.isfinite(anchor) or vehicle is None or vehicle.empty:
        result["audit_status"] = "missing_vehicle_or_anchor"
        return result
    tmin = float(vehicle["time_rel_s"].min())
    tmax = float(vehicle["time_rel_s"].max())
    if tmin > anchor - 3.0 or tmax < anchor + 5.0:
        result["audit_status"] = "window_incomplete"
        result["window_has_pre2"] = int(tmin <= anchor - 2.0)
        result["window_has_post5"] = int(tmax >= anchor + 5.0)
        return result

    base = window(vehicle, anchor - 3.5, anchor - 2.1)
    if len(base) < 30:
        base = window(vehicle, anchor - 4.0, anchor - 1.5)
    search = window(vehicle, anchor - 0.8, anchor + 1.8)
    post = window(vehicle, anchor, anchor + 3.0)
    full = window(vehicle, anchor - 2.0, anchor + 5.0)
    if len(base) < 30 or len(search) < 30 or len(post) < 30:
        result["audit_status"] = "local_window_insufficient"
        return result

    response_onsets: dict[str, float] = {}
    response_left: dict[str, int] = {}
    response_peaks: dict[str, float] = {}
    response_thresholds: dict[str, float] = {}
    response_flags: dict[str, int] = {}
    response_strengths: dict[str, float] = {}

    for key, spec in DYNAMIC_SPECS.items():
        col = str(spec["col"])
        floor = float(spec["floor"])
        if col not in vehicle.columns:
            response_onsets[key] = float("nan")
            response_left[key] = 0
            response_peaks[key] = float("nan")
            response_thresholds[key] = float("nan")
            response_flags[key] = 0
            response_strengths[key] = float("nan")
            continue
        base_values = base[col].to_numpy(dtype=float)
        base_med = robust_median(base_values)
        base_mad = robust_mad(base_values)
        local_thr = max(floor, 3.5 * (base_mad if math.isfinite(base_mad) else 0.0))
        values = search[col].to_numpy(dtype=float) - base_med
        cond = np.abs(values) >= local_thr
        onset, left = first_sustained_time(search["time_rel_s"].to_numpy(dtype=float), cond, min_duration_s=0.06)
        post_peak = max_abs(post[col].to_numpy(dtype=float) - base_med)
        strength = post_peak / local_thr if math.isfinite(post_peak) and local_thr > 0 else float("nan")
        flag = int(math.isfinite(onset) and math.isfinite(strength) and strength >= 1.0)
        response_onsets[key] = onset
        response_left[key] = int(left)
        response_peaks[key] = post_peak
        response_thresholds[key] = local_thr
        response_flags[key] = flag
        response_strengths[key] = strength

    valid_onsets = [v for v in response_onsets.values() if math.isfinite(v)]
    t_vehicle_response = min(valid_onsets) if valid_onsets else float("nan")
    component_count = int(sum(response_flags.values()))
    roll_evidence = bool(response_flags.get("roll_rate", 0) or response_flags.get("roll_angle", 0))
    lateral_evidence = bool(response_flags.get("ay", 0) or response_flags.get("yaw_rate", 0))
    multi_signal_vehicle_response = int(component_count >= 2)
    roll_response_event = int(roll_evidence and lateral_evidence)

    local_lat_step_p99 = percentile(full["lateral_step_abs"].to_numpy(dtype=float), 99.0)
    local_lat_step_max = max_abs(full["lateral_step_abs"].to_numpy(dtype=float))
    coordinate_continuity_ok = int(
        (not math.isfinite(local_lat_step_max))
        or (local_lat_step_max <= 4.0 and (not math.isfinite(local_lat_step_p99) or local_lat_step_p99 <= 1.5))
    )

    result.update(
        {
            "t_vehicle_response_s": t_vehicle_response,
            "vehicle_component_count": component_count,
            "multi_signal_vehicle_response": multi_signal_vehicle_response,
            "roll_evidence": int(roll_evidence),
            "lateral_evidence": int(lateral_evidence),
            "roll_response_event": roll_response_event,
            "coordinate_continuity_ok": coordinate_continuity_ok,
            "local_lateral_step_p99": local_lat_step_p99,
            "local_lateral_step_max": local_lat_step_max,
        }
    )
    for key in DYNAMIC_SPECS:
        result[f"{key}_onset_s"] = response_onsets[key]
        result[f"{key}_left_censored"] = response_left[key]
        result[f"{key}_peak_post3"] = response_peaks[key]
        result[f"{key}_threshold"] = response_thresholds[key]
        result[f"{key}_strength"] = response_strengths[key]
        result[f"{key}_response_flag"] = response_flags[key]
    return result


def detect_presteer_and_correction(row: pd.Series, vehicle: pd.DataFrame, t_vehicle: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "zx|SteeringWheel" not in vehicle.columns or not math.isfinite(t_vehicle):
        result["steer_status"] = "missing_steer_or_vehicle_time"
        return result

    base = window(vehicle, t_vehicle - 3.5, t_vehicle - 2.1)
    if len(base) < 30:
        base = window(vehicle, t_vehicle - 4.0, t_vehicle - 1.5)
    pre_search = window(vehicle, t_vehicle - 2.0, t_vehicle + 0.05)
    near_search = window(vehicle, t_vehicle - 2.0, t_vehicle + 0.25)
    post5 = window(vehicle, t_vehicle, t_vehicle + 5.0)
    if len(base) < 30 or len(pre_search) < 20 or len(post5) < 50:
        result["steer_status"] = "local_window_insufficient"
        return result

    far_base = window(vehicle, t_vehicle - 5.0, t_vehicle - 3.2)
    input_start = window(vehicle, t_vehicle - 2.15, t_vehicle - 1.85)

    steer_base = robust_median(base["zx|SteeringWheel"].to_numpy(dtype=float))
    steer_noise = robust_mad(base["zx|SteeringWheel"].to_numpy(dtype=float) - steer_base)
    rate_noise = robust_mad(base["steer_rate"].to_numpy(dtype=float))
    amp_thr = max(0.08, 4.0 * (steer_noise if math.isfinite(steer_noise) else 0.0))
    rate_thr = max(0.18, 4.0 * (rate_noise if math.isfinite(rate_noise) else 0.0))

    start_active = float("nan")
    start_delta_far = float("nan")
    start_rate_peak = float("nan")
    if len(far_base) >= 30 and len(input_start) >= 5:
        far_baseline = robust_median(far_base["zx|SteeringWheel"].to_numpy(dtype=float))
        far_noise = robust_mad(far_base["zx|SteeringWheel"].to_numpy(dtype=float) - far_baseline)
        start_delta_far = abs(robust_median(input_start["zx|SteeringWheel"].to_numpy(dtype=float)) - far_baseline)
        start_rate_peak = max_abs(input_start["steer_rate"].to_numpy(dtype=float))
        start_active = int(
            (
                math.isfinite(start_delta_far)
                and start_delta_far >= max(0.12, 4.0 * (far_noise if math.isfinite(far_noise) else 0.0))
            )
            or (math.isfinite(start_rate_peak) and start_rate_peak >= rate_thr)
        )

    def find_steer_onset(search_df: pd.DataFrame, stop_time: float) -> tuple[float, bool]:
        times = search_df["time_rel_s"].to_numpy(dtype=float)
        steer_dev = search_df["zx|SteeringWheel"].to_numpy(dtype=float) - steer_base
        rate = search_df["steer_rate"].to_numpy(dtype=float)
        future_amp = future_max_abs(steer_dev, times, 0.35, stop_time=stop_time)
        cond = (np.abs(steer_dev) >= amp_thr) | ((np.abs(rate) >= rate_thr) & (future_amp >= amp_thr * 0.8))
        return first_sustained_time(times, cond, min_duration_s=0.05)

    t_presteer, presteer_left = find_steer_onset(pre_search, stop_time=t_vehicle)
    t_near_steer, near_left = find_steer_onset(near_search, stop_time=t_vehicle + 0.25)
    t_steer = t_presteer if math.isfinite(t_presteer) else t_near_steer
    steer_onset_is_pre_vehicle = int(math.isfinite(t_presteer) and t_presteer <= t_vehicle)
    lead_s = t_vehicle - t_steer if math.isfinite(t_steer) else float("nan")

    pre_to_anchor = window(vehicle, t_vehicle - 2.0, t_vehicle)
    steer_delta_before_vehicle = max_abs(pre_to_anchor["zx|SteeringWheel"].to_numpy(dtype=float) - steer_base)
    steer_rate_peak_before_vehicle = max_abs(pre_to_anchor["steer_rate"].to_numpy(dtype=float))

    post_steer_dev = post5["zx|SteeringWheel"].to_numpy(dtype=float) - steer_base
    post_times = post5["time_rel_s"].to_numpy(dtype=float)
    if np.isfinite(post_steer_dev).any():
        peak_idx = int(np.nanargmax(np.abs(post_steer_dev)))
        t_steer_peak = float(post_times[peak_idx])
        peak_delta = float(post_steer_dev[peak_idx])
        peak_abs = abs(peak_delta)
    else:
        t_steer_peak = float("nan")
        peak_delta = float("nan")
        peak_abs = float("nan")

    after_peak = post_steer_dev[peak_idx:] if math.isfinite(t_steer_peak) else np.array([], dtype=float)
    abs_after_peak = np.abs(after_peak)
    min_after_peak = float(np.nanmin(abs_after_peak)) if abs_after_peak.size else float("nan")
    has_return = int(math.isfinite(peak_abs) and peak_abs >= amp_thr and math.isfinite(min_after_peak) and min_after_peak <= peak_abs * 0.55)
    if math.isfinite(peak_delta) and peak_abs >= amp_thr:
        opposite = after_peak * np.sign(peak_delta) <= -amp_thr * 0.5
        has_countersteer = int(bool(np.any(opposite)))
    else:
        has_countersteer = 0
    post_predictable = int(math.isfinite(peak_abs) and peak_abs >= amp_thr and (has_return or has_countersteer or peak_abs >= amp_thr * 2.0))

    result.update(
        {
            "steer_status": "ok",
            "steer_baseline_pre": steer_base,
            "steer_amp_threshold": amp_thr,
            "steer_rate_threshold": rate_thr,
            "t_presteer_onset_s": t_presteer,
            "presteer_left_censored": int(presteer_left),
            "t_near_steer_onset_s": t_near_steer,
            "near_steer_left_censored": int(near_left),
            "t_selected_steer_onset_s": t_steer,
            "steer_onset_is_pre_vehicle": steer_onset_is_pre_vehicle,
            "lead_vehicle_minus_steer_s": lead_s,
            "steer_onset_in_vehicle_pre2s": int(math.isfinite(t_presteer)),
            "steer_onset_in_vehicle_pre1s": int(math.isfinite(t_presteer) and t_vehicle - 1.0 <= t_presteer <= t_vehicle),
            "steer_delta_before_vehicle": steer_delta_before_vehicle,
            "steer_rate_peak_before_vehicle": steer_rate_peak_before_vehicle,
            "input_start_already_steering": start_active,
            "input_start_delta_from_far_baseline": start_delta_far,
            "input_start_steer_rate_peak": start_rate_peak,
            "t_steer_peak_post_vehicle_s": t_steer_peak,
            "steer_peak_delta_post_vehicle": peak_delta,
            "steer_peak_abs_post_vehicle": peak_abs,
            "has_return_after_vehicle": has_return,
            "has_countersteer_after_vehicle": has_countersteer,
            "post_vehicle_correction_predictable": post_predictable,
        }
    )
    return result


def classify(row: pd.Series) -> tuple[str, str, str]:
    if row.get("audit_status") != "ok" or row.get("steer_status") not in {"ok"}:
        return "X_exclude", "窗口或信号不足", "缺少完整窗口或必要信号，不能可靠审计"
    if int(row.get("coordinate_continuity_ok", 1)) == 0:
        return "X_exclude", "坐标连续性异常", "横向偏移存在非物理跳变，先不作为侧倾失稳核心样本"
    if int(row.get("multi_signal_vehicle_response", 0)) == 0:
        return "U_weak_vehicle_response", "车辆姿态证据不足", "车辆动态不是多信号一致增强，可能是噪声或普通操作"

    module = str(row.get("road_design_module_name", ""))
    lead = finite_float(row.get("lead_vehicle_minus_steer_s"))
    has_presteer = int(row.get("steer_onset_in_vehicle_pre2s", 0)) == 1
    has_post = int(row.get("post_vehicle_correction_predictable", 0)) == 1
    roll_event = int(row.get("roll_response_event", 0)) == 1
    comp_count = int(row.get("vehicle_component_count", 0))
    already_active_value = finite_float(row.get("input_start_already_steering", 0), 0.0)
    already_active_at_input_start = int(already_active_value) == 1

    if module in {"curve1", "curve2"} and not roll_event and comp_count < 3:
        return "C_normal_curve_like", "正常弯道/平滑转向候选", "弯道中车辆横向动态可能是正常过弯，不先作为失稳样本"
    if not has_presteer:
        return "V_vehicle_response_without_presteer", "车辆响应前未找到明确方向盘启动", "不符合方向盘先动引起侧倾的主假设，需要人工复核"
    if not math.isfinite(lead):
        return "U_unclear_timing", "时间差不清楚", "方向盘和车辆响应起点无法稳定计算"
    if lead < -0.10:
        return "V_vehicle_first_or_detection_late", "车辆响应早于方向盘或方向盘检测偏晚", "不符合方向盘先动假设，或方向盘启动点被检测晚了"
    if lead < 0.20:
        return "S_near_sync", "方向盘和车辆响应几乎同步", "适合作短时延续预测，不适合声称有明显提前量"
    if lead > 2.00:
        return "U_steer_too_early", "方向盘过早", "方向盘启动距离车辆响应超过 2 秒，因果关系需要人工复核"
    if not has_post:
        return "U_no_post_correction_target", "后续纠正目标不足", "车辆响应后方向盘可预测变化不足，训练价值有限"
    if already_active_at_input_start:
        return "U_started_before_pre2_window", "输入窗口开始前已在转向", "车辆响应前 2 秒窗口可能没有捕捉到方向盘动作起点，只包含动作中段"
    if module in CONTINUOUS_OR_CONTEXT_MODULES:
        return "R_context_or_continuous_review", "连续/上下文场景复核", "方向盘先动和车辆响应成立，但该场景更像连续任务或上下文事件，暂不放入最干净核心"
    if roll_event:
        return "P1_clean_roll_response_with_presteer", "最干净核心侧倾/姿态响应样本", "方向盘起点落在车辆响应前 2 秒内，随后横滚/横摆/横向动态多信号响应，且后续仍有纠正轨迹"
    return "P2_clean_lateral_response_with_presteer", "最干净次级横向动态响应样本", "方向盘起点落在车辆响应前 2 秒内，随后多信号车辆动态响应，但侧倾证据不如 P1 强"


def quality_score(row: pd.Series) -> float:
    score = 0.0
    score += 10.0 * int(row.get("roll_response_event", 0))
    score += 4.0 * int(row.get("multi_signal_vehicle_response", 0))
    score += 2.0 * int(row.get("vehicle_component_count", 0))
    score += 5.0 * int(row.get("steer_onset_in_vehicle_pre2s", 0))
    score += 4.0 * int(row.get("post_vehicle_correction_predictable", 0))
    lead = finite_float(row.get("lead_vehicle_minus_steer_s"))
    if math.isfinite(lead):
        if 0.2 <= lead <= 1.5:
            score += 5.0
        elif 0.0 <= lead < 0.2:
            score += 2.0
    peak = finite_float(row.get("steer_peak_abs_post_vehicle"))
    if math.isfinite(peak):
        score += min(6.0, peak * 2.0)
    if int(row.get("coordinate_continuity_ok", 1)) == 0:
        score -= 8.0
    return float(score)


def build_rescreen_table() -> pd.DataFrame:
    candidates = pd.read_csv(INPUT_CANDIDATES, encoding="utf-8-sig", low_memory=False)
    vehicle_cache: dict[str, pd.DataFrame | None] = {}
    rows: list[dict[str, Any]] = []
    for idx, row in candidates.iterrows():
        path_text = str(row.get("vehicle_raw_absolute_path", ""))
        if path_text not in vehicle_cache:
            vehicle_cache[path_text] = load_vehicle(path_text)
        vehicle = vehicle_cache[path_text]
        base = row.to_dict()
        if vehicle is None or vehicle.empty:
            base.update({"audit_status": "vehicle_missing", "steer_status": "vehicle_missing"})
        else:
            vehicle_info = detect_vehicle_response(row, vehicle)
            base.update(vehicle_info)
            t_vehicle = finite_float(vehicle_info.get("t_vehicle_response_s"))
            if not math.isfinite(t_vehicle):
                t_vehicle = finite_float(row.get("anchor_time_rel_s"))
            steer_info = detect_presteer_and_correction(row, vehicle, t_vehicle)
            base.update(steer_info)
        category, category_cn, reason = classify(pd.Series(base))
        base["v0_2_category"] = category
        base["v0_2_category_cn"] = category_cn
        base["v0_2_reason_cn"] = reason
        base["v0_2_quality_score"] = quality_score(pd.Series(base))
        base["recommended_vehicle_anchor_s"] = finite_float(base.get("t_vehicle_response_s"), finite_float(base.get("anchor_time_rel_s")))
        base["recommended_input_start_s"] = base["recommended_vehicle_anchor_s"] - 2.0 if math.isfinite(base["recommended_vehicle_anchor_s"]) else float("nan")
        base["recommended_input_end_s"] = base["recommended_vehicle_anchor_s"]
        base["recommended_label_start_s"] = base["recommended_vehicle_anchor_s"]
        base["recommended_label_end_s"] = base["recommended_vehicle_anchor_s"] + 5.0 if math.isfinite(base["recommended_vehicle_anchor_s"]) else float("nan")
        rows.append(base)
        if (idx + 1) % 200 == 0:
            print(f"rescreened {idx + 1}/{len(candidates)}")
    return pd.DataFrame(rows)


def write_tables(df: pd.DataFrame) -> dict[str, Path]:
    all_path = TABLE_DIR / "vehicle_response_presteer_candidates_v0_2.csv"
    p1_path = TABLE_DIR / "primary_roll_presteer_events_P1_v0_2.csv"
    p2_path = TABLE_DIR / "secondary_lateral_presteer_events_P2_v0_2.csv"
    sync_path = TABLE_DIR / "near_sync_events_S_v0_2.csv"
    review_path = TABLE_DIR / "manual_review_events_v0_2.csv"
    exclude_path = TABLE_DIR / "excluded_events_X_v0_2.csv"
    summary_path = TABLE_DIR / "vehicle_response_presteer_summary_v0_2.csv"
    by_module_path = TABLE_DIR / "vehicle_response_presteer_by_module_v0_2.csv"
    quantile_path = TABLE_DIR / "vehicle_response_presteer_latency_quantiles_v0_2.csv"

    df.to_csv(all_path, index=False, encoding="utf-8-sig")
    df[df["v0_2_category"].eq("P1_clean_roll_response_with_presteer")].to_csv(p1_path, index=False, encoding="utf-8-sig")
    df[df["v0_2_category"].eq("P2_clean_lateral_response_with_presteer")].to_csv(p2_path, index=False, encoding="utf-8-sig")
    df[df["v0_2_category"].eq("S_near_sync")].to_csv(sync_path, index=False, encoding="utf-8-sig")
    df[
        df["v0_2_category"].str.startswith("U_")
        | df["v0_2_category"].str.startswith("V_")
        | df["v0_2_category"].eq("C_normal_curve_like")
        | df["v0_2_category"].eq("R_context_or_continuous_review")
    ].to_csv(
        review_path, index=False, encoding="utf-8-sig"
    )
    df[df["v0_2_category"].str.startswith("X_")].to_csv(exclude_path, index=False, encoding="utf-8-sig")

    summary = (
        df.groupby(["v0_2_category", "v0_2_category_cn"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    summary["ratio"] = summary["count"] / max(len(df), 1)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    by_module = (
        df.groupby(["road_design_module_name", "v0_2_category", "v0_2_category_cn"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["road_design_module_name", "count"], ascending=[True, False])
    )
    by_module.to_csv(by_module_path, index=False, encoding="utf-8-sig")

    valid = pd.to_numeric(df["lead_vehicle_minus_steer_s"], errors="coerce")
    quant_rows = []
    for group_name, mask in {
        "all_valid": valid.notna(),
        "P1": df["v0_2_category"].eq("P1_clean_roll_response_with_presteer") & valid.notna(),
        "P2": df["v0_2_category"].eq("P2_clean_lateral_response_with_presteer") & valid.notna(),
        "S_near_sync": df["v0_2_category"].eq("S_near_sync") & valid.notna(),
    }.items():
        values = valid[mask]
        quant_rows.append(
            {
                "group": group_name,
                "n": int(len(values)),
                "mean": float(values.mean()) if len(values) else float("nan"),
                "median": float(values.median()) if len(values) else float("nan"),
                "p10": float(values.quantile(0.10)) if len(values) else float("nan"),
                "p25": float(values.quantile(0.25)) if len(values) else float("nan"),
                "p75": float(values.quantile(0.75)) if len(values) else float("nan"),
                "p90": float(values.quantile(0.90)) if len(values) else float("nan"),
                "lead_ge_0_2_ratio": float((values >= 0.2).mean()) if len(values) else float("nan"),
                "lead_ge_0_5_ratio": float((values >= 0.5).mean()) if len(values) else float("nan"),
            }
        )
    pd.DataFrame(quant_rows).to_csv(quantile_path, index=False, encoding="utf-8-sig")
    return {
        "all": all_path,
        "p1": p1_path,
        "p2": p2_path,
        "sync": sync_path,
        "review": review_path,
        "exclude": exclude_path,
        "summary": summary_path,
        "by_module": by_module_path,
        "quantile": quantile_path,
    }


def plot_summary(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    counts = df["v0_2_category_cn"].value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    counts.iloc[::-1].plot(kind="barh", ax=ax, color="#4e79a7")
    ax.set_title("车辆响应锚点样本重新筛选：类别数量")
    ax.set_xlabel("样本数")
    fig.tight_layout()
    path = FIG_DIR / "vehicle_response_presteer_category_counts_v0_2.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    valid = pd.to_numeric(df["lead_vehicle_minus_steer_s"], errors="coerce").dropna()
    if len(valid):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(valid.clip(-2.0, 2.5), bins=60, color="#59a14f", alpha=0.85)
        ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1)
        ax.axvline(0.2, color="#d62728", linestyle="--", linewidth=1)
        ax.axvline(0.5, color="#ff7f0e", linestyle="--", linewidth=1)
        ax.set_title("车辆响应时刻 - 方向盘启动时刻")
        ax.set_xlabel("秒：正值表示方向盘先动")
        ax.set_ylabel("样本数")
        fig.tight_layout()
        path = FIG_DIR / "vehicle_minus_steer_lead_histogram_v0_2.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def make_panel(row: pd.Series, vehicle: pd.DataFrame, path: Path) -> None:
    t_vehicle = finite_float(row.get("recommended_vehicle_anchor_s"))
    if not math.isfinite(t_vehicle):
        return
    seg = window(vehicle, t_vehicle - 3.0, t_vehicle + 5.0)
    if seg.empty:
        return
    t = seg["time_rel_s"].to_numpy(dtype=float) - t_vehicle
    signals = [
        ("方向盘角", "zx|SteeringWheel"),
        ("方向盘角速度", "steer_rate"),
        ("横向加速度", "zx|ay"),
        ("横摆角速度", "zx|vyaw"),
        ("横滚角速度", "zx|vroll"),
        ("横滚角", "zx|roll"),
        ("横向偏移", "zx1|lateraldistance"),
    ]
    fig, axes = plt.subplots(len(signals), 1, figsize=(11, 10), sharex=True)
    for ax, (label, col) in zip(axes, signals):
        if col in seg.columns:
            ax.plot(t, seg[col].to_numpy(dtype=float), linewidth=1.0)
        ax.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.0, label="车辆响应锚点")
        ts = finite_float(row.get("t_selected_steer_onset_s"))
        if math.isfinite(ts):
            ax.axvline(ts - t_vehicle, color="#1f77b4", linestyle="--", linewidth=1.0, label="方向盘启动")
        tp = finite_float(row.get("t_steer_peak_post_vehicle_s"))
        if math.isfinite(tp):
            ax.axvline(tp - t_vehicle, color="#2ca02c", linestyle=":", linewidth=1.0, label="方向盘峰值")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].set_title(
        f"{row.get('v0_2_category_cn')} | {row.get('subject')} {row.get('session_stamp')} | "
        f"{row.get('road_design_module_name')} | lead={finite_float(row.get('lead_vehicle_minus_steer_s')):.3f}s"
    )
    axes[-1].set_xlabel("相对车辆响应锚点时间 / 秒")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=[0, 0, 0.96, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_review_panels(df: pd.DataFrame) -> pd.DataFrame:
    cache: dict[str, pd.DataFrame | None] = {}
    rows = []
    groups = [
        ("P1_clean_roll_response_with_presteer", 20),
        ("P2_clean_lateral_response_with_presteer", 20),
        ("R_context_or_continuous_review", 20),
        ("S_near_sync", 20),
        ("V_vehicle_response_without_presteer", 15),
        ("V_vehicle_first_or_detection_late", 15),
        ("C_normal_curve_like", 15),
        ("U_no_post_correction_target", 15),
    ]
    for category, limit in groups:
        subset = df[df["v0_2_category"].eq(category)].sort_values("v0_2_quality_score", ascending=False).head(limit)
        for _, row in subset.iterrows():
            path_text = str(row.get("vehicle_raw_absolute_path", ""))
            if path_text not in cache:
                cache[path_text] = load_vehicle(path_text)
            vehicle = cache[path_text]
            if vehicle is None or vehicle.empty:
                continue
            safe_uid = str(row.get("instability_event_uid", "event")).replace("/", "_").replace("\\", "_").replace(":", "_")
            out = PANEL_DIR / f"{category}__{safe_uid}.png"
            make_panel(row, vehicle, out)
            if out.exists():
                rows.append(
                    {
                        "v0_2_category": category,
                        "v0_2_category_cn": row.get("v0_2_category_cn"),
                        "instability_event_uid": row.get("instability_event_uid"),
                        "subject": row.get("subject"),
                        "session_stamp": row.get("session_stamp"),
                        "road_design_module_name": row.get("road_design_module_name"),
                        "lead_vehicle_minus_steer_s": row.get("lead_vehicle_minus_steer_s"),
                        "figure_path": str(out),
                    }
                )
    panel_df = pd.DataFrame(rows)
    panel_df.to_csv(TABLE_DIR / "vehicle_response_presteer_review_panel_index_v0_2.csv", index=False, encoding="utf-8-sig")
    return panel_df


def pct(value: float) -> str:
    if not math.isfinite(float(value)):
        return "NA"
    return f"{value * 100:.1f}%"


def write_reports(df: pd.DataFrame, paths: dict[str, Path], fig_paths: list[Path], panel_df: pd.DataFrame) -> None:
    total = len(df)
    summary = pd.read_csv(paths["summary"], encoding="utf-8-sig")
    count_lines = "\n".join(
        f"- {row.v0_2_category_cn}：{int(row['count'])} 个，占 {row['ratio'] * 100:.1f}%"
        for _, row in summary.iterrows()
    )
    valid_lead = pd.to_numeric(df["lead_vehicle_minus_steer_s"], errors="coerce").dropna()
    lead02 = float((valid_lead >= 0.2).mean()) if len(valid_lead) else float("nan")
    lead05 = float((valid_lead >= 0.5).mean()) if len(valid_lead) else float("nan")
    p1_n = int(df["v0_2_category"].eq("P1_clean_roll_response_with_presteer").sum())
    p2_n = int(df["v0_2_category"].eq("P2_clean_lateral_response_with_presteer").sum())
    sync_n = int(df["v0_2_category"].eq("S_near_sync").sum())
    context_n = int(df["v0_2_category"].eq("R_context_or_continuous_review").sum())
    review_n = int(
        (
            df["v0_2_category"].str.startswith("U_")
            | df["v0_2_category"].str.startswith("V_")
            | df["v0_2_category"].eq("C_normal_curve_like")
            | df["v0_2_category"].eq("R_context_or_continuous_review")
        ).sum()
    )

    user_report = REPORT_DIR / "stage02_vehicle_response_presteer_rescreen_user_summary_cn.md"
    user_report.write_text(
        f"""# 车辆响应锚点前方向盘动作重新筛选 v0.2（用户查看版）

## 这次为什么重新筛选

你提出的核心判断是：实际事件大多不是“车辆先侧倾、驾驶员再纠偏”，而是“驾驶员主动打方向盘，车辆随后出现横向动态、横摆、横滚或侧倾增强”。因此，不能再直接沿用旧的 v0.6 样本分类，也不能只用方向盘动作池。

本次重新筛选的目标是：

- 先从原始车辆动态候选里找车辆响应锚点；
- 再检查车辆响应锚点前 2 秒内是否存在明确方向盘启动；
- 再判断车辆响应后是否还有可预测的回正、反打或纠正轨迹；
- 最后把真正适合“侧倾/姿态响应前早期方向盘信息预测后续纠正”的样本筛出来。

## 这次用的输入

- 输入候选表：`{INPUT_CANDIDATES}`
- 候选数量：{total} 个
- 这些候选来自原始车辆 CSV 的横向加速度、横摆、横滚等非方向盘车辆动态扫描，比旧 v0.6 高置信表更宽。

## 核心筛选原则

这次不是“阈值过了就算侧倾失稳”。核心样本至少要同时满足：

1. 车辆动态不是单一信号异常，而是横向加速度、横摆角速度、横滚角速度、横滚角中至少多个信号有证据；
2. 要有横滚相关证据，才进入 P1 侧倾/姿态响应核心样本；
3. 方向盘启动要落在车辆响应锚点前 2 秒内；
4. 车辆响应锚点后还要有可预测的方向盘回正、反打或继续纠正轨迹；
5. 横向偏移存在明显坐标跳变的样本先排除；
6. 正常弯道平滑转向不直接当作侧倾失稳样本。

## 筛选结果

{count_lines}

其中：

- P1 最干净核心侧倾/姿态响应样本：{p1_n} 个
- P2 最干净次级横向动态响应样本：{p2_n} 个
- 几乎同步样本：{sync_n} 个
- 连续/上下文场景复核样本：{context_n} 个
- 需要人工复核或暂缓样本：{review_n} 个

## 时间差结果

在能够计算方向盘启动和车辆响应时间差的样本里：

- 方向盘至少领先车辆响应 0.2 秒的比例：{pct(lead02)}
- 方向盘至少领先车辆响应 0.5 秒的比例：{pct(lead05)}

这个数字用于判断“车辆响应锚点前 2 秒是否真的包含早期方向盘动作”。如果 P1/P2 复核图能确认方向盘确实在车辆响应前启动，那么侧倾锚点路线可以继续；如果多数只是同步或检测不清，就不能声称有明显提前量。

## 推荐优先查看

- 总表：`{paths['all']}`
- P1 最干净核心样本表：`{paths['p1']}`
- P2 最干净次级样本表：`{paths['p2']}`
- 汇总表：`{paths['summary']}`
- 分场景表：`{paths['by_module']}`
- 时间差分位数表：`{paths['quantile']}`
- 复核图索引：`{TABLE_DIR / 'vehicle_response_presteer_review_panel_index_v0_2.csv'}`

图：
{chr(10).join(f"- `{p}`" for p in fig_paths)}

## 当前结论边界

这一步仍然不是模型训练结果。它只回答：能不能筛出一批“方向盘先动、车辆随后侧倾/横向动态增强、后续仍有纠正轨迹”的样本。

如果人工复核 P1 图后基本认可，下一步才适合基于这些样本构建新预测任务：

> 输入车辆响应锚点前 2 秒的车辆状态和方向盘早期动作，预测车辆响应锚点后的方向盘纠正轨迹。
""",
        encoding="utf-8",
    )

    tech_report = REPORT_DIR / "vehicle_response_presteer_rescreen_v0_2_cn.md"
    tech_report.write_text(
        f"""# 车辆响应锚点前方向盘动作重新筛选 v0.2

## 方法

本脚本从 `{INPUT_CANDIDATES}` 的 1991 个原始车辆动态候选重新开始，不直接继承 v0.6 分类。每个候选重新读取原始车辆 CSV，并在候选锚点附近计算：

- 车辆响应锚点：横向加速度、横摆角速度、横滚角速度、横滚角的局部稳健阈值越界起点；
- 多信号车辆响应：至少两个车辆动态信号成立；
- 侧倾/姿态响应：横滚相关证据与横向/横摆证据同时成立；
- 方向盘启动：车辆响应前 2 秒内方向盘角或方向盘角速度显著离开局部基线；
- 后续纠正目标：车辆响应后方向盘存在峰值、回正、反打或持续纠正；
- 坐标连续性：横向偏移步进过大则排除。

## 输出

- 总表：`{paths['all']}`
- P1：`{paths['p1']}`
- P2：`{paths['p2']}`
- 几乎同步：`{paths['sync']}`
- 人工复核：`{paths['review']}`
- 排除：`{paths['exclude']}`
- 汇总：`{paths['summary']}`
- 分场景：`{paths['by_module']}`
- 分位数：`{paths['quantile']}`
- 复核图索引：`{TABLE_DIR / 'vehicle_response_presteer_review_panel_index_v0_2.csv'}`

## 类别统计

{count_lines}

## 关键数量

- 总候选：{total}
- P1 最干净核心样本：{p1_n}
- P2 最干净次级样本：{p2_n}
- 近同步样本：{sync_n}
- 连续/上下文复核样本：{context_n}
- 复核/暂缓样本：{review_n}
- 复核图数量：{len(panel_df)}

## 时间差

- 有效时间差样本：{len(valid_lead)}
- 方向盘领先 >=0.2 秒比例：{pct(lead02)}
- 方向盘领先 >=0.5 秒比例：{pct(lead05)}
""",
        encoding="utf-8",
    )

    DAILY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DAILY_LOG.open("a", encoding="utf-8") as f:
        f.write(
            f"""## 车辆响应锚点前方向盘动作重新筛选 v0.2

- 为什么做：用户指出所有事件更可能是驾驶员主动打方向导致车辆侧倾/横向动态增强，因此旧 v0.6 样本不能直接沿用，需要重新筛选。
- 做了什么：从 1991 个原始车辆动态候选重新读取车辆 CSV，检查车辆响应锚点前 2 秒方向盘启动、车辆多信号响应、横滚证据、后续纠正目标和坐标连续性。
- 输出：
  - 用户查看版：`{user_report}`
  - 技术报告：`{tech_report}`
  - 总表：`{paths['all']}`
  - P1 最干净核心样本：`{paths['p1']}`
  - 复核图索引：`{TABLE_DIR / 'vehicle_response_presteer_review_panel_index_v0_2.csv'}`
- 结果：P1={p1_n}，P2={p2_n}，近同步={sync_n}，连续/上下文复核={context_n}，复核/暂缓={review_n}。
- 下一步：先人工查看 P1/P2 复核图，再决定是否用车辆响应锚点前 2 秒作为新训练输入。

"""
        )

    artifact = NOTES_DIR / "ARTIFACT_INDEX_CN.md"
    old = artifact.read_text(encoding="utf-8", errors="ignore") if artifact.exists() else "# 产物索引\n\n"
    artifact.write_text(
        f"""## 2026-05-18 车辆响应锚点前方向盘动作重新筛选 v0.2

- 用户查看版：`{user_report}`
- 技术报告：`{tech_report}`
- 总表：`{paths['all']}`
- P1 最干净核心样本：`{paths['p1']}`
- P2 最干净次级样本：`{paths['p2']}`
- 汇总表：`{paths['summary']}`
- 分场景表：`{paths['by_module']}`
- 时间差表：`{paths['quantile']}`
- 复核图索引：`{TABLE_DIR / 'vehicle_response_presteer_review_panel_index_v0_2.csv'}`

""" + old,
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    df = build_rescreen_table()
    paths = write_tables(df)
    fig_paths = plot_summary(df)
    panel_df = make_review_panels(df)
    write_reports(df, paths, fig_paths, panel_df)
    print(f"done: total={len(df)}")
    print(df["v0_2_category_cn"].value_counts().to_string())
    print(f"report={REPORT_DIR / 'stage02_vehicle_response_presteer_rescreen_user_summary_cn.md'}")


if __name__ == "__main__":
    main()
