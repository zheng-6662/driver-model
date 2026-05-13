# -*- coding: utf-8 -*-
from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
PROJECT_ROOT = Path(r"F:/data_set_process/data_process")
RAW_ROOT = PROJECT_ROOT / "01_datasets" / "数据预处理"

HIGHCONF_EVENTS = (
    ROOT
    / "02_samples"
    / "vehicle_instability_all_raw_rescreen_v0_1"
    / "tables"
    / "all_raw_vehicle_instability_primary_high_confidence_v0_1.csv"
)
EVENT_ANCHOR_TABLE = (
    ROOT
    / "02_samples"
    / "vehicle_instability_highconf_v0_1"
    / "tables"
    / "event_anchor_table.csv"
)
SCENE_TRIGGER_TIMES = (
    ROOT
    / "02_samples"
    / "scene_trigger_audit_v0_2"
    / "tables"
    / "scene_trigger_session_times_v0_2.csv"
)
DESIGN_CANDIDATE_SCORES = (
    ROOT
    / "02_samples"
    / "event_candidate_filter_v0_5"
    / "tables"
    / "event_candidate_scores_v0_5.csv"
)

OUT_DIR = ROOT / "02_samples" / "episode_first_event_v0_6"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
PANEL_DIR = FIG_DIR / "episode_review_panels"
REPORT_DIR = ROOT / "09_reports"

VEHICLE_COLS = [
    "StorageTime",
    "zx|SteeringWheel",
    "zx|ay",
    "zx|vyaw",
    "zx|vroll",
    "zx1|lateraldistance",
    "zx|BrakePedal",
    "zx|ax",
    "zx1|v_km/h",
    "zx1|mu",
    "zx1|lanecurvatureXY",
]

CORE_FIRST_MODULES = {"differentmu_road", "fix_road", "curve1", "curve2"}
HOLDOUT_MODULES = {"middle_section", "longstraight", "stop", "curve3", "zd"}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, PANEL_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    for old_png in PANEL_DIR.glob("*.png"):
        old_png.unlink()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def robust_scale(values: np.ndarray, fallback: float = 1.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) * 1.4826
    p90 = float(np.percentile(np.abs(values), 90))
    scale = max(mad * 3.0, p90, fallback)
    return scale if math.isfinite(scale) and scale > 1e-9 else fallback


def load_vehicle(relative_path: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not relative_path:
        return pd.DataFrame()
    if relative_path in cache:
        return cache[relative_path]
    path = RAW_ROOT / relative_path
    if not path.exists():
        cache[relative_path] = pd.DataFrame()
        return cache[relative_path]
    try:
        header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
        usecols = [col for col in VEHICLE_COLS if col in header.columns]
        raw = pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    except Exception:
        cache[relative_path] = pd.DataFrame()
        return cache[relative_path]
    if "StorageTime" not in raw.columns:
        cache[relative_path] = pd.DataFrame()
        return cache[relative_path]

    numeric_time = pd.to_numeric(raw["StorageTime"], errors="coerce")
    if float(numeric_time.notna().mean()) >= 0.8:
        storage_s = numeric_time.astype(float)
    else:
        parsed = pd.to_datetime(raw["StorageTime"], errors="coerce")
        storage_s = pd.Series(np.where(parsed.notna(), parsed.astype("int64") / 1e9, np.nan), index=raw.index)
    raw["storage_time_s"] = storage_s
    for col in raw.columns:
        if col not in {"StorageTime", "storage_time_s"}:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["storage_time_s"]).sort_values("storage_time_s").drop_duplicates("storage_time_s")
    if raw.empty:
        cache[relative_path] = pd.DataFrame()
        return cache[relative_path]
    for col in raw.columns:
        if col not in {"StorageTime", "storage_time_s"}:
            raw[col] = raw[col].interpolate(limit_direction="both")
    raw["time_rel_s"] = raw["storage_time_s"] - float(raw["storage_time_s"].iloc[0])

    if "zx1|lateraldistance" in raw:
        lat = raw["zx1|lateraldistance"].to_numpy(dtype=float)
        t = raw["time_rel_s"].to_numpy(dtype=float)
        raw["lat_vel"] = np.gradient(lat, t, edge_order=1) if len(raw) > 3 else 0.0
    else:
        raw["lat_vel"] = 0.0

    scales = {
        "ay": robust_scale(raw.get("zx|ay", pd.Series(dtype=float)).to_numpy(dtype=float), 0.8),
        "vyaw": robust_scale(raw.get("zx|vyaw", pd.Series(dtype=float)).to_numpy(dtype=float), 0.03),
        "vroll": robust_scale(raw.get("zx|vroll", pd.Series(dtype=float)).to_numpy(dtype=float), 0.05),
        "lat_vel": robust_scale(raw["lat_vel"].to_numpy(dtype=float), 0.2),
    }
    raw["dynamic_score"] = (
        np.abs(raw.get("zx|ay", 0.0)) / scales["ay"]
        + np.abs(raw.get("zx|vyaw", 0.0)) / scales["vyaw"]
        + np.abs(raw.get("zx|vroll", 0.0)) / scales["vroll"]
        + np.abs(raw["lat_vel"]) / scales["lat_vel"]
    ) / 4.0
    raw["dynamic_score_non_lateral"] = (
        np.abs(raw.get("zx|ay", 0.0)) / scales["ay"]
        + np.abs(raw.get("zx|vyaw", 0.0)) / scales["vyaw"]
        + np.abs(raw.get("zx|vroll", 0.0)) / scales["vroll"]
    ) / 3.0
    cache[relative_path] = raw
    return raw


def nearest_value(times: list[float], t: float) -> tuple[float, float]:
    if not times or not math.isfinite(t):
        return float("nan"), float("nan")
    pos = bisect.bisect_left(times, t)
    candidates = []
    if pos < len(times):
        candidates.append(times[pos])
    if pos > 0:
        candidates.append(times[pos - 1])
    nearest = min(candidates, key=lambda x: abs(x - t))
    return nearest, nearest - t


def build_lookup(df: pd.DataFrame, time_col: str, extra_cols: list[str]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if df.empty or time_col not in df:
        return lookup
    for _, row in df.iterrows():
        t = finite_float(row.get(time_col))
        if not math.isfinite(t):
            continue
        key = (str(row.get("subject", "")), str(row.get("session_stamp", "")))
        payload = {"time": t}
        for col in extra_cols:
            payload[col] = row.get(col, "")
        lookup[key].append(payload)
    for key in list(lookup):
        lookup[key] = sorted(lookup[key], key=lambda x: x["time"])
    return lookup


def nearest_payload(lookup: dict[tuple[str, str], list[dict[str, Any]]], key: tuple[str, str], t: float) -> dict[str, Any]:
    items = lookup.get(key, [])
    if not items or not math.isfinite(t):
        return {}
    times = [float(item["time"]) for item in items]
    pos = bisect.bisect_left(times, t)
    candidate_indices = []
    if pos < len(items):
        candidate_indices.append(pos)
    if pos > 0:
        candidate_indices.append(pos - 1)
    idx = min(candidate_indices, key=lambda i: abs(times[i] - t))
    out = dict(items[idx])
    out["delta_s"] = times[idx] - t
    out["abs_delta_s"] = abs(times[idx] - t)
    return out


def time_window(df: pd.DataFrame, center: float, before: float, after: float) -> pd.DataFrame:
    if df.empty or not math.isfinite(center):
        return pd.DataFrame()
    return df[(df["time_rel_s"] >= center - before) & (df["time_rel_s"] <= center + after)].copy()


def first_sustained_time(win: pd.DataFrame, mask: np.ndarray, min_count: int = 12) -> float:
    if win.empty or mask.size == 0:
        return float("nan")
    count = 0
    times = win["time_rel_s"].to_numpy(dtype=float)
    for idx, flag in enumerate(mask):
        if flag:
            count += 1
            if count >= min_count:
                return float(times[max(0, idx - min_count + 1)])
        else:
            count = 0
    return float("nan")


def compute_episode_features(row: pd.Series, vehicle: pd.DataFrame) -> dict[str, Any]:
    t0 = finite_float(row.get("anchor_time_rel_s"))
    event_end = finite_float(row.get("event_end_rel_s"), t0 + 3.0)
    if vehicle.empty or not math.isfinite(t0):
        return {"vehicle_feature_status": "missing_vehicle_or_bad_time"}

    full_min = float(vehicle["time_rel_s"].min())
    full_max = float(vehicle["time_rel_s"].max())
    has_pre3 = int(full_min <= t0 - 3.0)
    has_post5 = int(full_max >= t0 + 5.0)

    pre_base = time_window(vehicle, t0, 5.0, -0.2)
    if pre_base.empty:
        pre_base = time_window(vehicle, t0, 1.0, 0.0)
    post5 = time_window(vehicle, t0, 0.0, 5.0)
    local = time_window(vehicle, t0, 1.0, max(5.0, event_end - t0 + 1.0))
    if post5.empty or local.empty:
        return {"vehicle_feature_status": "window_empty", "has_pre3s": has_pre3, "has_post5s": has_post5}

    steer = vehicle["zx|SteeringWheel"] if "zx|SteeringWheel" in vehicle else pd.Series(dtype=float)
    steer_base = float(np.nanmedian(pre_base["zx|SteeringWheel"])) if "zx|SteeringWheel" in pre_base and not pre_base.empty else 0.0
    pre_noise = pre_base["zx|SteeringWheel"].to_numpy(dtype=float) - steer_base if "zx|SteeringWheel" in pre_base else np.array([])
    steer_noise = robust_scale(pre_noise, 0.02)
    steer_threshold = max(0.04, 2.5 * steer_noise)

    search = time_window(vehicle, t0, 0.8, 3.0)
    steer_delta_search = search["zx|SteeringWheel"].to_numpy(dtype=float) - steer_base if "zx|SteeringWheel" in search else np.array([])
    steer_onset = first_sustained_time(search, np.abs(steer_delta_search) >= steer_threshold)
    pre_steer = time_window(vehicle, t0, 0.8, 0.0)
    pre_steer_delta = pre_steer["zx|SteeringWheel"].to_numpy(dtype=float) - steer_base if "zx|SteeringWheel" in pre_steer else np.array([])
    pre_steer_active = bool(pre_steer_delta.size and np.nanmax(np.abs(pre_steer_delta)) >= steer_threshold)

    peak_win = time_window(vehicle, t0, 0.5, 4.0)
    if "zx|SteeringWheel" in peak_win and not peak_win.empty:
        d = peak_win["zx|SteeringWheel"].to_numpy(dtype=float) - steer_base
        peak_idx = int(np.nanargmax(np.abs(d))) if np.isfinite(d).any() else 0
        steer_peak_time = float(peak_win["time_rel_s"].iloc[peak_idx])
        steer_delta_peak = float(d[peak_idx])
        steer_delta_abs_peak = abs(steer_delta_peak)
    else:
        steer_peak_time = float("nan")
        steer_delta_peak = float("nan")
        steer_delta_abs_peak = float("nan")

    dyn_win = vehicle[(vehicle["time_rel_s"] >= t0) & (vehicle["time_rel_s"] <= max(event_end, t0 + 3.0))].copy()
    dynamic_score_col = "dynamic_score_non_lateral" if "dynamic_score_non_lateral" in dyn_win else "dynamic_score"
    dyn_idx = int(np.nanargmax(dyn_win[dynamic_score_col].to_numpy(dtype=float))) if not dyn_win.empty else 0
    t_dyn_peak = float(dyn_win["time_rel_s"].iloc[dyn_idx]) if not dyn_win.empty else float("nan")
    dynamic_peak_score = float(dyn_win[dynamic_score_col].iloc[dyn_idx]) if not dyn_win.empty else float("nan")

    after_peak = vehicle[(vehicle["time_rel_s"] >= t_dyn_peak + 0.5) & (vehicle["time_rel_s"] <= t_dyn_peak + 2.5)]
    dynamic_tail_median = float(np.nanmedian(after_peak[dynamic_score_col])) if not after_peak.empty else float("nan")
    correction_dynamic_drop_ratio = (
        dynamic_tail_median / dynamic_peak_score
        if math.isfinite(dynamic_tail_median) and math.isfinite(dynamic_peak_score) and dynamic_peak_score > 1e-9
        else float("nan")
    )
    correction_dynamic_drop = bool(math.isfinite(correction_dynamic_drop_ratio) and correction_dynamic_drop_ratio <= 0.72)

    after_steer_peak = vehicle[(vehicle["time_rel_s"] >= steer_peak_time) & (vehicle["time_rel_s"] <= steer_peak_time + 3.0)]
    has_return = False
    has_countersteer = False
    if math.isfinite(steer_delta_peak) and steer_delta_abs_peak > 1e-9 and not after_steer_peak.empty:
        post_delta = after_steer_peak["zx|SteeringWheel"].to_numpy(dtype=float) - steer_base
        has_return = bool(np.nanmin(np.abs(post_delta)) <= max(0.05, steer_delta_abs_peak * 0.5))
        has_countersteer = bool(np.nanmin(post_delta) < -0.25 * steer_delta_abs_peak) if steer_delta_peak > 0 else bool(np.nanmax(post_delta) > 0.25 * steer_delta_abs_peak)

    coord_win = time_window(vehicle, t0, 2.0, 5.0)
    coordinate_continuity_ok = True
    lateral_range = float("nan")
    max_lateral_step = float("nan")
    if "zx1|lateraldistance" in coord_win and len(coord_win) >= 3:
        lat = coord_win["zx1|lateraldistance"].to_numpy(dtype=float)
        lateral_range = float(np.nanmax(lat) - np.nanmin(lat))
        max_lateral_step = float(np.nanmax(np.abs(np.diff(lat))))
        coordinate_continuity_ok = bool(max_lateral_step <= 0.5 and lateral_range <= 12.0)

    post_response_confirmed = bool(
        math.isfinite(steer_delta_abs_peak)
        and steer_delta_abs_peak >= steer_threshold
        and (has_return or has_countersteer or correction_dynamic_drop or steer_delta_abs_peak >= 2.0 * steer_threshold)
    )
    pre_window_clean = bool(not pre_steer_active)

    peak_abs_ay = float(np.nanmax(np.abs(post5["zx|ay"]))) if "zx|ay" in post5 else float("nan")
    peak_abs_yaw = float(np.nanmax(np.abs(post5["zx|vyaw"]))) if "zx|vyaw" in post5 else float("nan")
    peak_abs_roll = float(np.nanmax(np.abs(post5["zx|vroll"]))) if "zx|vroll" in post5 else float("nan")
    lat_range_post5 = float(np.nanmax(post5["zx1|lateraldistance"]) - np.nanmin(post5["zx1|lateraldistance"])) if "zx1|lateraldistance" in post5 else float("nan")
    speed_at_t0 = float(post5["zx1|v_km/h"].iloc[0]) if "zx1|v_km/h" in post5 and not post5.empty else float("nan")
    mu_at_t0 = float(post5["zx1|mu"].iloc[0]) if "zx1|mu" in post5 and not post5.empty else float("nan")

    return {
        "vehicle_feature_status": "ok",
        "has_pre3s": has_pre3,
        "has_post5s": has_post5,
        "t_dyn_onset": t0,
        "t_dyn_peak": t_dyn_peak,
        "dynamic_peak_score_local": dynamic_peak_score,
        "dynamic_score_used_for_peak": dynamic_score_col,
        "dynamic_tail_median_after_peak": dynamic_tail_median,
        "correction_dynamic_drop_ratio": correction_dynamic_drop_ratio,
        "correction_dynamic_drop": int(correction_dynamic_drop),
        "t_steer_onset": steer_onset,
        "t_steer_peak": steer_peak_time,
        "steer_baseline_pre": steer_base,
        "steer_noise_scale_pre": steer_noise,
        "steer_response_threshold": steer_threshold,
        "steer_delta_peak_local": steer_delta_peak,
        "steer_delta_abs_peak_local": steer_delta_abs_peak,
        "steer_onset_delay_from_dyn_s": steer_onset - t0 if math.isfinite(steer_onset) else float("nan"),
        "pre_steer_active": int(pre_steer_active),
        "pre_window_clean": int(pre_window_clean),
        "has_return": int(has_return),
        "has_countersteer": int(has_countersteer),
        "post_response_confirmed": int(post_response_confirmed),
        "coordinate_continuity_ok": int(coordinate_continuity_ok),
        "lateral_range_local": lateral_range,
        "max_lateral_step_local": max_lateral_step,
        "peak_abs_ay_post5": peak_abs_ay,
        "peak_abs_yaw_post5": peak_abs_yaw,
        "peak_abs_roll_post5": peak_abs_roll,
        "lateral_range_post5": lat_range_post5,
        "speed_at_t0": speed_at_t0,
        "mu_at_t0": mu_at_t0,
    }


def classify_episode(row: pd.Series) -> dict[str, Any]:
    module = str(row.get("road_design_module_name", ""))
    role = str(row.get("instability_role", ""))
    feature_ok = str(row.get("vehicle_feature_status", "")) == "ok"
    has_pre3 = int(finite_float(row.get("has_pre3s"), 0)) == 1
    has_post5 = int(finite_float(row.get("has_post5s"), 0)) == 1
    coord_ok = int(finite_float(row.get("coordinate_continuity_ok"), 0)) == 1
    post_resp = int(finite_float(row.get("post_response_confirmed"), 0)) == 1
    pre_clean = int(finite_float(row.get("pre_window_clean"), 0)) == 1
    has_return = int(finite_float(row.get("has_return"), 0)) == 1
    has_counter = int(finite_float(row.get("has_countersteer"), 0)) == 1
    dyn_drop = int(finite_float(row.get("correction_dynamic_drop"), 0)) == 1
    steer_delay = finite_float(row.get("steer_onset_delay_from_dyn_s"))
    steer_peak = finite_float(row.get("steer_delta_abs_peak_local"), 0.0)
    threshold = finite_float(row.get("steer_response_threshold"), 0.05)

    if not feature_ok or not has_post5:
        return {
            "episode_label": "X_exclude",
            "episode_type_cn": "窗口或车辆数据不足",
            "confidence_tier": "D",
            "review_status": "exclude",
            "recommended_table": "holdout_or_excluded",
            "recommended_anchor_basis": "none",
        }
    if not post_resp:
        return {
            "episode_label": "N_vehicle_dynamic_no_steering_response",
            "episode_type_cn": "车辆动态异常但方向盘响应不足",
            "confidence_tier": "B",
            "review_status": "weak_response",
            "recommended_table": "manual_review",
            "recommended_anchor_basis": "t_dyn_onset",
        }

    correction = has_return or has_counter or dyn_drop
    if module in {"middle_section"}:
        return {
            "episode_label": "U_continuous_episode",
            "episode_type_cn": "连续超车/连续任务，需要拆子事件",
            "confidence_tier": "B" if correction else "C",
            "review_status": "continuous",
            "recommended_table": "manual_review",
            "recommended_anchor_basis": "episode_level_or_subevent",
        }
    if module in {"longstraight", "stop", "zd", "curve3"}:
        return {
            "episode_label": "U_unclear_or_holdout_scene",
            "episode_type_cn": "场景被试相关性或语义仍需复核",
            "confidence_tier": "B" if correction else "C",
            "review_status": "unclear",
            "recommended_table": "manual_review",
            "recommended_anchor_basis": "t_exposure_if_available_else_t_dyn_onset",
        }

    if module in {"curve1", "curve2"} and not (has_return or has_counter or dyn_drop):
        return {
            "episode_label": "C_normal_curve",
            "episode_type_cn": "可能是正常弯道转向，不作为失稳纠正核心",
            "confidence_tier": "C",
            "review_status": "normal_curve",
            "recommended_table": "holdout_or_excluded",
            "recommended_anchor_basis": "road_geometry",
        }

    driver_started_first = (not pre_clean) or (math.isfinite(steer_delay) and steer_delay <= 0.15)
    if module == "fix_road" or (module in {"curve1", "curve2"} and driver_started_first):
        label = "P2_driver_initiated_avoidance"
        type_cn = "主动避让/转向后产生高横向动态"
        anchor_basis = "t_exposure_or_trigger_if_available"
    elif module == "differentmu_road" or (math.isfinite(steer_delay) and steer_delay > 0.15):
        label = "P1_vehicle_disturbance_correction"
        type_cn = "车辆扰动/失稳后方向盘纠偏"
        anchor_basis = "t_dyn_onset_or_t_exposure"
    else:
        label = "U_unclear"
        type_cn = "因果顺序不清，需要复核"
        anchor_basis = "manual_review"

    tier = "S" if correction and has_pre3 and has_post5 and steer_peak >= 1.5 * threshold else "A"
    table = "primary_training" if tier in {"S", "A"} and module in CORE_FIRST_MODULES else "manual_review"
    return {
        "episode_label": label,
        "episode_type_cn": type_cn,
        "confidence_tier": tier,
        "review_status": "pass" if table == "primary_training" else "unclear",
        "recommended_table": table,
        "recommended_anchor_basis": anchor_basis,
    }


def attach_nearest_context(episodes: pd.DataFrame) -> pd.DataFrame:
    scene = read_csv(SCENE_TRIGGER_TIMES) if SCENE_TRIGGER_TIMES.exists() else pd.DataFrame()
    design = read_csv(DESIGN_CANDIDATE_SCORES) if DESIGN_CANDIDATE_SCORES.exists() else pd.DataFrame()
    scene_lookup = build_lookup(
        scene,
        "estimated_trigger_time_rel_s",
        ["scene_trigger_uid", "module_name", "trigger_name", "target_title", "target_lane_id", "change_target_lane"],
    )
    design_lookup = build_lookup(
        design,
        "candidate_time_rel_s",
        ["candidate_uid", "module_name", "candidate_anchor_type_cn", "candidate_source_cn", "screening_decision_cn"],
    )
    rows = []
    for _, row in episodes.iterrows():
        key = (str(row.get("subject", "")), str(row.get("session_stamp", "")))
        t0 = finite_float(row.get("anchor_time_rel_s"))
        out = row.to_dict()
        nearest_scene = nearest_payload(scene_lookup, key, t0)
        nearest_design = nearest_payload(design_lookup, key, t0)
        out["nearest_scene_trigger_time_rel_s"] = nearest_scene.get("time", float("nan"))
        out["nearest_scene_trigger_delta_s"] = nearest_scene.get("delta_s", float("nan"))
        out["nearest_scene_trigger_abs_delta_s"] = nearest_scene.get("abs_delta_s", float("nan"))
        out["nearest_scene_trigger_name"] = nearest_scene.get("trigger_name", "")
        out["nearest_scene_target_title"] = nearest_scene.get("target_title", "")
        out["nearest_scene_module_name"] = nearest_scene.get("module_name", "")
        out["nearest_scene_change_target_lane"] = nearest_scene.get("change_target_lane", "")
        out["nearest_design_candidate_time_rel_s"] = nearest_design.get("time", float("nan"))
        out["nearest_design_candidate_delta_s"] = nearest_design.get("delta_s", float("nan"))
        out["nearest_design_candidate_abs_delta_s"] = nearest_design.get("abs_delta_s", float("nan"))
        out["nearest_design_candidate_type_cn"] = nearest_design.get("candidate_anchor_type_cn", "")
        out["nearest_design_candidate_decision_cn"] = nearest_design.get("screening_decision_cn", "")
        rows.append(out)
    return pd.DataFrame(rows)


def choose_train_anchor(row: pd.Series) -> float:
    label = str(row.get("episode_label", ""))
    basis = str(row.get("recommended_anchor_basis", ""))
    t_dyn = finite_float(row.get("t_dyn_onset"))
    scene_t = finite_float(row.get("nearest_scene_trigger_time_rel_s"))
    design_t = finite_float(row.get("nearest_design_candidate_time_rel_s"))
    if "trigger" in basis and math.isfinite(scene_t) and abs(scene_t - t_dyn) <= 5.0 and scene_t <= t_dyn + 0.5:
        return scene_t
    if "t_exposure" in basis and math.isfinite(design_t) and abs(design_t - t_dyn) <= 4.0 and design_t <= t_dyn + 0.5:
        return design_t
    return t_dyn


def build_episode_tables() -> pd.DataFrame:
    base = read_csv(HIGHCONF_EVENTS)
    formal = read_csv(EVENT_ANCHOR_TABLE) if EVENT_ANCHOR_TABLE.exists() else pd.DataFrame()
    if not formal.empty:
        keep_cols = [
            "event_uid",
            "oldcode_usable",
            "history_full_3s_oldcode",
            "future_full_2s_oldcode",
            "random_event_split",
            "session_level_split",
            "subject_level_split",
            "vehicle_available",
            "physio_available",
            "eeg_available",
            "all_three_modalities_available",
        ]
        rename = {"event_uid": "instability_event_uid"}
        formal_small = formal[[c for c in keep_cols if c in formal]].rename(columns=rename)
        base = base.merge(formal_small, on="instability_event_uid", how="left")

    cache: dict[str, pd.DataFrame] = {}
    rows = []
    for _, row in base.iterrows():
        vehicle = load_vehicle(str(row.get("vehicle_raw_relative_path", "")), cache)
        features = compute_episode_features(row, vehicle)
        out = row.to_dict()
        out.update(features)
        rows.append(out)
    episodes = pd.DataFrame(rows)
    episodes = attach_nearest_context(episodes)
    class_rows = [classify_episode(row) for _, row in episodes.iterrows()]
    classified = pd.concat([episodes.reset_index(drop=True), pd.DataFrame(class_rows)], axis=1)
    unclear_mask = classified["episode_label"].eq("U_unclear")
    classified.loc[unclear_mask, "recommended_table"] = "manual_review"
    classified.loc[unclear_mask, "review_status"] = "unclear"
    classified["t_train_anchor"] = classified.apply(choose_train_anchor, axis=1)
    classified["episode_id_v0_6"] = [
        f"episode_v0_6__{s}__{ss}__{i:05d}"
        for i, (s, ss) in enumerate(zip(classified["subject"], classified["session_stamp"]))
    ]
    coord_ok = pd.to_numeric(classified["coordinate_continuity_ok"], errors="coerce").fillna(0).astype(int) == 1
    has_pre3 = pd.to_numeric(classified["has_pre3s"], errors="coerce").fillna(0).astype(int) == 1
    has_post5 = pd.to_numeric(classified["has_post5s"], errors="coerce").fillna(0).astype(int) == 1
    role_text = classified.get("source_event_types", pd.Series("", index=classified.index)).astype(str)
    instability_role_text = classified.get("instability_role", pd.Series("", index=classified.index)).astype(str)
    non_lateral_dynamic_evidence_ok = (
        role_text.str.contains("ay|roll|yaw", case=False, regex=True, na=False)
        | instability_role_text.str.contains("ay|roll|yaw", case=False, regex=True, na=False)
        | (pd.to_numeric(classified.get("peak_abs_ay_post5", 0), errors="coerce").fillna(0.0) >= 1.2)
        | (pd.to_numeric(classified.get("peak_abs_roll_post5", 0), errors="coerce").fillna(0.0) >= 0.08)
        | (pd.to_numeric(classified.get("peak_abs_yaw_post5", 0), errors="coerce").fillna(0.0) >= 0.12)
    )
    core_positive_base = (
        classified["recommended_table"].eq("primary_training")
        & classified["episode_label"].isin(["P1_vehicle_disturbance_correction", "P2_driver_initiated_avoidance"])
        & classified["road_design_module_name"].astype(str).isin(CORE_FIRST_MODULES)
        & classified["confidence_tier"].isin(["S", "A"])
        & has_pre3
        & has_post5
        & non_lateral_dynamic_evidence_ok
    )
    classified["coordinate_issue_needs_review"] = (~coord_ok).astype(int)
    classified["non_lateral_dynamic_evidence_ok"] = non_lateral_dynamic_evidence_ok.astype(int)
    classified["is_first_core_training_candidate"] = (
        core_positive_base
        & coord_ok
    )
    classified["is_coordinate_flagged_core_candidate"] = (
        core_positive_base
        & ~coord_ok
    )
    classified["v0_6_final_bucket"] = "manual_review"
    classified["v0_6_final_bucket_cn"] = "人工复核"
    classified.loc[classified["episode_label"].astype(str).str.startswith("N_"), "v0_6_final_bucket"] = "weak_or_no_steering_response"
    classified.loc[classified["episode_label"].astype(str).str.startswith("N_"), "v0_6_final_bucket_cn"] = "车辆动态明显但方向盘响应不足，可作为弱响应/负样本"
    classified.loc[classified["episode_label"].eq("U_continuous_episode"), "v0_6_final_bucket"] = "continuous_episode_review"
    classified.loc[classified["episode_label"].eq("U_continuous_episode"), "v0_6_final_bucket_cn"] = "连续超车任务，需拆子事件"
    classified.loc[classified["episode_label"].eq("U_unclear_or_holdout_scene"), "v0_6_final_bucket"] = "holdout_scene_review"
    classified.loc[classified["episode_label"].eq("U_unclear_or_holdout_scene"), "v0_6_final_bucket_cn"] = "场景相关性或语义暂缓复核"
    classified.loc[classified["episode_label"].eq("U_unclear"), "v0_6_final_bucket"] = "unclear_review"
    classified.loc[classified["episode_label"].eq("U_unclear"), "v0_6_final_bucket_cn"] = "因果顺序不清，人工复核"
    classified.loc[classified["is_coordinate_flagged_core_candidate"], "v0_6_final_bucket"] = "coordinate_flagged_expansion"
    classified.loc[classified["is_coordinate_flagged_core_candidate"], "v0_6_final_bucket_cn"] = "车辆动态和方向盘响应成立，但横向偏移坐标需复核"
    classified.loc[classified["is_first_core_training_candidate"], "v0_6_final_bucket"] = "strict_clean_primary"
    classified.loc[classified["is_first_core_training_candidate"], "v0_6_final_bucket_cn"] = "第一版最干净核心训练候选"
    return classified


def split_tables(episodes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    primary = episodes[episodes["is_first_core_training_candidate"]].copy()
    coordinate_flagged = episodes[episodes["is_coordinate_flagged_core_candidate"]].copy()
    manual = episodes[episodes["recommended_table"].eq("manual_review")].copy()
    response_only = episodes[
        episodes["episode_label"].isin(["C_normal_curve"])
        | episodes["review_status"].isin(["normal_curve"])
    ].copy()
    holdout = episodes[
        episodes["recommended_table"].eq("holdout_or_excluded")
        | episodes["episode_label"].astype(str).str.startswith("X_")
        | episodes["v0_6_final_bucket"].isin(["holdout_scene_review", "unclear_review"])
    ].copy()
    trigger_no_effect = episodes[episodes["episode_label"].astype(str).str.startswith("N_")].copy()
    return {
        "primary_training_events_v0_6": primary,
        "primary_training_events_v0_6_strict_clean": primary,
        "coordinate_flagged_expansion_events_v0_6": coordinate_flagged,
        "manual_review_events_v0_6": manual,
        "response_confirm_only_v0_6": response_only,
        "holdout_or_excluded_v0_6": holdout,
        "trigger_no_effect_or_weak_response_v0_6": trigger_no_effect,
    }


def summarize(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_label = (
        episodes.groupby(["episode_label", "episode_type_cn", "confidence_tier", "recommended_table"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    by_module = (
        episodes.groupby(["road_design_module_name", "episode_label"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["road_design_module_name", "n"], ascending=[True, False])
    )
    return by_label, by_module


def decision_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    return (
        episodes.groupby(["v0_6_final_bucket", "v0_6_final_bucket_cn", "road_design_module_name", "episode_label"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["v0_6_final_bucket", "road_design_module_name", "n"], ascending=[True, True, False])
    )


def configure_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_summary(episodes: pd.DataFrame, by_module: pd.DataFrame) -> None:
    configure_font()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    label_counts = episodes["episode_label"].value_counts().sort_values()
    axes[0].barh(label_counts.index, label_counts.values, color="#4f7cac")
    axes[0].set_title("episode 类型数量")
    axes[0].set_xlabel("数量")

    pivot = by_module.pivot_table(index="road_design_module_name", columns="episode_label", values="n", fill_value=0)
    pivot.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab20")
    axes[1].set_title("分场景 episode 类型")
    axes[1].set_xlabel("场景")
    axes[1].set_ylabel("数量")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "episode_first_v0_6_summary.png", dpi=180)
    plt.close(fig)


def plot_episode_panel(row: pd.Series, cache: dict[str, pd.DataFrame], out_path: Path) -> None:
    configure_font()
    vehicle = load_vehicle(str(row.get("vehicle_raw_relative_path", "")), cache)
    t0 = finite_float(row.get("t_dyn_onset"))
    if vehicle.empty or not math.isfinite(t0):
        return
    win = vehicle[(vehicle["time_rel_s"] >= t0 - 3.0) & (vehicle["time_rel_s"] <= t0 + 6.0)].copy()
    if win.empty:
        return
    rel = win["time_rel_s"] - t0
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    steer_base = finite_float(row.get("steer_baseline_pre"), 0.0)
    axes[0].plot(rel, win["zx|SteeringWheel"] - steer_base, color="#1f77b4", linewidth=1)
    axes[0].set_ylabel("方向盘变化")
    axes[1].plot(rel, win["zx|ay"], label="横向加速度", color="#d62728", linewidth=1)
    axes[1].plot(rel, win["zx|vyaw"], label="横摆角速度", color="#2ca02c", linewidth=1)
    axes[1].legend(fontsize=8)
    axes[1].set_ylabel("横向动态")
    axes[2].plot(rel, win["zx|vroll"], color="#9467bd", linewidth=1)
    axes[2].set_ylabel("横滚速率")
    if "zx1|lateraldistance" in win:
        axes[3].plot(rel, win["zx1|lateraldistance"] - float(win["zx1|lateraldistance"].iloc[0]), color="#8c564b", linewidth=1)
    axes[3].set_ylabel("横向偏移")
    score_col = "dynamic_score_non_lateral" if "dynamic_score_non_lateral" in win else "dynamic_score"
    axes[4].plot(rel, win[score_col], color="#111111", linewidth=1)
    axes[4].set_ylabel("非横偏动态强度")
    axes[4].set_xlabel("相对 t_dyn_onset 时间/s")

    marker_times = [
        (0.0, "#e63946", "t_dyn_onset"),
        (finite_float(row.get("t_dyn_peak")) - t0, "#f4a261", "t_dyn_peak"),
        (finite_float(row.get("t_steer_onset")) - t0, "#2a9d8f", "t_steer_onset"),
        (finite_float(row.get("t_steer_peak")) - t0, "#264653", "t_steer_peak"),
        (finite_float(row.get("nearest_scene_trigger_time_rel_s")) - t0, "#6c757d", "nearest_trigger"),
    ]
    for ax in axes:
        for mt, color, _label in marker_times:
            if math.isfinite(mt) and -3.0 <= mt <= 6.0:
                ax.axvline(mt, color=color, linestyle="--" if mt != 0 else "-", linewidth=1)
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        f"{row.get('road_design_module_name')} | {row.get('episode_label')} | {row.get('confidence_tier')} | "
        f"strict={row.get('is_first_core_training_candidate', 0)} coord_review={row.get('coordinate_issue_needs_review', 0)} | "
        f"{row.get('subject')} {row.get('session_stamp')}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_representative_panels(episodes: pd.DataFrame) -> pd.DataFrame:
    cache: dict[str, pd.DataFrame] = {}
    rows = []
    selected_parts = []
    pick_specs = [
        ("strict_primary", episodes[episodes["is_first_core_training_candidate"]], 8),
        ("coordinate_flagged_expansion", episodes[episodes["is_coordinate_flagged_core_candidate"]], 8),
        ("weak_or_no_steering_response", episodes[episodes["episode_label"].astype(str).str.startswith("N_")], 5),
        ("continuous_episode", episodes[episodes["episode_label"].eq("U_continuous_episode")], 5),
        ("holdout_scene_review", episodes[episodes["episode_label"].eq("U_unclear_or_holdout_scene")], 5),
        ("unclear_review", episodes[episodes["episode_label"].eq("U_unclear")], 5),
    ]
    for panel_group, group, n in pick_specs:
        if group.empty:
            continue
        picked = group.sort_values("instability_review_score", ascending=False).head(n).copy()
        picked["_panel_group"] = panel_group
        selected_parts.append(picked)
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    for idx, row in selected.iterrows():
        safe_label = str(row.get("episode_label", "episode")).replace("/", "_")
        safe_group = str(row.get("_panel_group", "review")).replace("/", "_")
        out_path = PANEL_DIR / f"{idx + 1:03d}_{safe_group}_{safe_label}.png"
        plot_episode_panel(row, cache, out_path)
        rows.append(
            {
                "episode_id_v0_6": row.get("episode_id_v0_6", ""),
                "panel_group": row.get("_panel_group", ""),
                "episode_label": row.get("episode_label", ""),
                "road_design_module_name": row.get("road_design_module_name", ""),
                "confidence_tier": row.get("confidence_tier", ""),
                "is_first_core_training_candidate": row.get("is_first_core_training_candidate", 0),
                "is_coordinate_flagged_core_candidate": row.get("is_coordinate_flagged_core_candidate", 0),
                "coordinate_issue_needs_review": row.get("coordinate_issue_needs_review", 0),
                "figure_path": str(out_path),
            }
        )
    return pd.DataFrame(rows)


def write_report(episodes: pd.DataFrame, tables: dict[str, pd.DataFrame], by_label: pd.DataFrame, by_module: pd.DataFrame, panel_index: pd.DataFrame) -> None:
    bucket_counts = episodes["v0_6_final_bucket"].value_counts().to_dict()
    lines = [
        "# episode-first 事件样本 v0.6",
        "",
        f"生成时间：{now_str()}",
        "",
        "## 这次做了什么",
        "",
        "本轮按照 GPTPro 的最新建议，不再从设计触发点出发，而是从原始车辆动态出发：先判定是否存在车辆动态 episode，再补方向盘响应、纠正过程和附近场景触发点。",
        "",
        "核心思路是：先找到真实发生的“车辆动态-方向盘-纠正 episode”，再回头解释它和哪个触发或场景有关。",
        "",
        "## 输入",
        "",
        f"- 全原始车辆动态高置信事件：`{HIGHCONF_EVENTS}`",
        f"- 场景触发时间表：`{SCENE_TRIGGER_TIMES}`",
        f"- v0.5 候选触发评分表：`{DESIGN_CANDIDATE_SCORES}`",
        "",
        "## 输出",
        "",
        f"- episode 总表：`{TABLE_DIR / 'episode_candidates_v0_6.csv'}`",
        f"- 第一版可训练核心表：`{TABLE_DIR / 'primary_training_events_v0_6.csv'}`",
        f"- 坐标需复核扩展候选表：`{TABLE_DIR / 'coordinate_flagged_expansion_events_v0_6.csv'}`",
        f"- 人工复核表：`{TABLE_DIR / 'manual_review_events_v0_6.csv'}`",
        f"- 响应确认/正常弯道表：`{TABLE_DIR / 'response_confirm_only_v0_6.csv'}`",
        f"- 暂缓/排除表：`{TABLE_DIR / 'holdout_or_excluded_v0_6.csv'}`",
        f"- 弱响应/触发无效候选表：`{TABLE_DIR / 'trigger_no_effect_or_weak_response_v0_6.csv'}`",
        f"- 概览图：`{FIG_DIR / 'episode_first_v0_6_summary.png'}`",
        f"- 代表图索引：`{TABLE_DIR / 'episode_review_panel_index_v0_6.csv'}`",
        "",
        "## 数量概览",
        "",
        f"- 输入车辆动态 episode：{len(episodes)}",
        f"- 第一版可训练核心：{len(tables['primary_training_events_v0_6'])}",
        f"- 坐标需复核但动态和方向盘响应成立的扩展候选：{len(tables['coordinate_flagged_expansion_events_v0_6'])}",
        f"- 弱响应/负样本候选：{bucket_counts.get('weak_or_no_steering_response', 0)}",
        f"- 连续超车任务复核：{bucket_counts.get('continuous_episode_review', 0)}",
        f"- 场景暂缓复核：{bucket_counts.get('holdout_scene_review', 0)}",
        f"- 因果顺序不清复核：{bucket_counts.get('unclear_review', 0)}",
        "",
        "按 episode 类型：",
        "",
        "| 类型 | 置信级别 | 推荐去向 | 数量 |",
        "|---|---|---|---:|",
    ]
    for _, row in by_label.iterrows():
        lines.append(
            f"| {row['episode_label']} / {row['episode_type_cn']} | {row['confidence_tier']} | {row['recommended_table']} | {int(row['n'])} |"
        )
    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            "1. 这一步已经从“触发点是不是事件”转为“是否真实发生车辆动态-方向盘-纠正 episode”。",
            "2. `middle_section` 和 `longstraight/stop` 当前仍主要进入复核或暂缓，不直接进入第一版核心训练。",
            "3. 第一版可训练核心主要来自 `differentmu_road`、`fix_road`、`curve1/curve2` 中满足方向盘响应和纠正条件的 episode。",
            "4. 这个 v0.6 仍是自动规则版，后续需要看代表图，确认是否存在锚点偏晚、坐标跳变或正常弯道误判。",
            "",
            "## 下一步建议",
            "",
            "1. 先看 episode 代表图，确认 P1/P2 分类是否合理。",
            "2. 若 primary_training_events_v0_6 数量和质量可接受，再用它构建纯车辆/道路 baseline。",
            "3. 如果 primary 数量太少，则先补人工复核，不要急着训练。",
        ]
    )
    (REPORT_DIR / "episode_first_event_v0_6_cn.md").write_text("\n".join(lines), encoding="utf-8")

    user_lines = [
        "# episode-first 事件样本 v0.6 用户版说明",
        "",
        f"生成时间：{now_str()}",
        "",
        "## 这一步为什么做",
        "",
        "GPTPro 指出，我们真正需要的不是“哪个设计触发点是真事件”，而是先判断驾驶过程中是否真实出现了车辆动态异常、方向盘响应和回正/纠正的完整片段。因此本轮先从车辆动态 episode 出发，再回头贴场景触发点。",
        "",
        "## 当前结果",
        "",
        f"本轮输入 {len(episodes)} 个车辆动态高置信 episode，自动分出 {len(tables['primary_training_events_v0_6'])} 个第一版最干净核心候选、{len(tables['coordinate_flagged_expansion_events_v0_6'])} 个坐标需复核扩展候选、{bucket_counts.get('weak_or_no_steering_response', 0)} 个弱响应/负样本、{bucket_counts.get('continuous_episode_review', 0)} 个连续任务复核、{bucket_counts.get('holdout_scene_review', 0)} 个场景暂缓复核和 {bucket_counts.get('unclear_review', 0)} 个因果顺序不清复核。",
        "",
        f"另外，有 {len(tables['coordinate_flagged_expansion_events_v0_6'])} 个片段满足车辆动态、方向盘响应和窗口完整条件，但横向偏移坐标存在跳变风险。它们不能直接混入最干净训练集，但也不能简单判废，建议作为第二批人工复核或扩展候选。",
        "",
        "## 你优先看什么",
        "",
        f"1. 完整报告：`{REPORT_DIR / 'episode_first_event_v0_6_cn.md'}`",
        f"2. 第一版可训练核心表：`{TABLE_DIR / 'primary_training_events_v0_6.csv'}`",
        f"3. 坐标需复核扩展候选表：`{TABLE_DIR / 'coordinate_flagged_expansion_events_v0_6.csv'}`",
        f"4. 分桶汇总表：`{TABLE_DIR / 'episode_decision_summary_v0_6.csv'}`",
        f"5. 人工复核表：`{TABLE_DIR / 'manual_review_events_v0_6.csv'}`",
        f"6. 概览图：`{FIG_DIR / 'episode_first_v0_6_summary.png'}`",
        f"7. 代表图目录：`{PANEL_DIR}`",
        "",
        "## 当前不能直接下的结论",
        "",
        "这还不是最终训练集。它是自动规则版 episode-first 清单。下一步要看代表图，确认 P1/P2 分类、锚点位置和坐标连续性，再决定是否训练纯车辆/道路 baseline。",
    ]
    (REPORT_DIR / "stage02_episode_first_v0_6_user_summary_cn.md").write_text("\n".join(user_lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    episodes = build_episode_tables()
    tables = split_tables(episodes)
    by_label, by_module = summarize(episodes)
    by_decision = decision_summary(episodes)

    write_csv(episodes, TABLE_DIR / "episode_candidates_v0_6.csv")
    for name, df in tables.items():
        write_csv(df, TABLE_DIR / f"{name}.csv")
    write_csv(by_label, TABLE_DIR / "episode_label_summary_v0_6.csv")
    write_csv(by_module, TABLE_DIR / "episode_module_summary_v0_6.csv")
    write_csv(by_decision, TABLE_DIR / "episode_decision_summary_v0_6.csv")
    plot_summary(episodes, by_module)
    panel_index = plot_representative_panels(episodes)
    write_csv(panel_index, TABLE_DIR / "episode_review_panel_index_v0_6.csv")
    write_report(episodes, tables, by_label, by_module, panel_index)

    print(
        {
            "episodes": len(episodes),
            "primary": len(tables["primary_training_events_v0_6"]),
            "coordinate_flagged_expansion": len(tables["coordinate_flagged_expansion_events_v0_6"]),
            "manual": len(tables["manual_review_events_v0_6"]),
            "holdout": len(tables["holdout_or_excluded_v0_6"]),
            "weak_or_no_effect": len(tables["trigger_no_effect_or_weak_response_v0_6"]),
        }
    )


if __name__ == "__main__":
    main()
