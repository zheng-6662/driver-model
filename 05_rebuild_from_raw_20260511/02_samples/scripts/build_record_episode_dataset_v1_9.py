#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build v1.9 record-level episode dataset with coordinate-based road context.

The key correction in this version is user-driven:

- Curve episodes must not be defined by height profile.
- Not every curve is downhill.
- Road/curve context should be determined from road coordinates and road
  geometry. Height is only an auxiliary abnormality signal.

This script reads the v1.8 episode table, maps each episode back to the road
centerline using ego vehicle x/y coordinates, then reclassifies curve vs
non-curve samples. It does not train any model.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except Exception:  # noqa: BLE001
    cKDTree = None

import build_record_episode_dataset_v1_8 as v18


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
V18_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_8_anchor_curve_revised"
V18_ALL = V18_ROOT / "tables" / "record_level_episodes_all_v1_8.csv"
ROAD_LAYOUT = (
    PROJECT_ROOT
    / "01_datasets"
    / "多模态数据"
    / "被试数据集合"
    / "道路信息"
    / "full_centerline_layout.csv"
)

OUT_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_9_coord_curve_revised"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "review_panels_v1_9"
REPORT_PATH = ROOT / "09_reports" / "stage02_record_episode_reconstruction_v1_9_user_summary_cn.md"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-22.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"


CURVE_MODULES = {"curve1", "curve2", "curve3"}
ROAD_COLS = ["StorageTime", "zx|x", "zx|y", "zx1|lanecurvatureXY", "zx1|lateraldistance"]

# These thresholds are only for auxiliary judgment, not for defining curves.
ROLL_ANGLE_CANDIDATE_RAD = 0.08
ROLL_RATE_CANDIDATE_RADPS = 0.60
AY_DYNAMIC_CANDIDATE = 5.0
CURVE_HEIGHT_RISE_ABNORMAL_M = 0.15
CURVE_Z_RESIDUAL_RANGE_ABNORMAL = 3.50
CURVE_Z_RESIDUAL_RATE_ABNORMAL = 3.00
LANE_CURVATURE_HINT = 0.001


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, REPORT_PATH.parent, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def to_seconds(storage_time: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(storage_time, errors="coerce")
    out = np.full(len(storage_time), np.nan, dtype=np.float64)
    valid = parsed.notna().to_numpy()
    if valid.any():
        ns = parsed[valid].astype("datetime64[ns]").astype("int64").to_numpy(dtype=np.float64)
        out[valid] = ns / 1e9
    return out


def mode_text(values: list[str]) -> str:
    cleaned = [str(v) for v in values if str(v) and str(v).lower() != "nan"]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def unique_join(values: list[str]) -> str:
    seen: list[str] = []
    for value in values:
        text = str(value)
        if not text or text.lower() == "nan":
            continue
        if text not in seen:
            seen.append(text)
    return "|".join(seen)


def load_road_mapper() -> tuple[Any | None, pd.DataFrame | None, str]:
    if cKDTree is None:
        return None, None, "scipy_unavailable"
    if not ROAD_LAYOUT.exists():
        return None, None, "layout_missing"
    layout = pd.read_csv(ROAD_LAYOUT, low_memory=False)
    required = {"x", "y", "s", "module_name", "instance_name", "curvature"}
    if not required.issubset(set(layout.columns)):
        return None, None, "layout_missing_required_columns"
    for col in ["x", "y", "s", "curvature"]:
        layout[col] = pd.to_numeric(layout[col], errors="coerce")
    valid = layout[["x", "y"]].notna().all(axis=1)
    layout = layout.loc[valid].reset_index(drop=True)
    if layout.empty:
        return None, None, "layout_no_valid_xy"
    tree = cKDTree(layout[["x", "y"]].to_numpy(dtype=np.float64))
    return tree, layout, "ok"


def load_vehicle_context(path: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if path in cache:
        return cache[path]
    df = pd.read_csv(path, usecols=lambda c: c in ROAD_COLS, low_memory=False)
    if "StorageTime" not in df.columns:
        raise ValueError("missing StorageTime")
    t_abs = to_seconds(df["StorageTime"])
    valid_t = np.isfinite(t_abs)
    if not valid_t.any():
        raise ValueError("no valid StorageTime")
    df = df.loc[valid_t].copy()
    t_abs = t_abs[valid_t]
    df["t_rel_s"] = t_abs - float(t_abs[0])
    for col in ROAD_COLS:
        if col == "StorageTime":
            continue
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    df = df.sort_values("t_rel_s").reset_index(drop=True)
    for col in ["zx1|lanecurvatureXY", "zx1|lateraldistance"]:
        valid = df[["t_rel_s", col]].notna().all(axis=1).to_numpy()
        df.attrs[f"{col}_t"] = df.loc[valid, "t_rel_s"].to_numpy(dtype=np.float64)
        df.attrs[f"{col}_v"] = df.loc[valid, col].to_numpy(dtype=np.float64)
    valid_xy = df[["t_rel_s", "zx|x", "zx|y"]].notna().all(axis=1).to_numpy()
    df.attrs["xy_t"] = df.loc[valid_xy, "t_rel_s"].to_numpy(dtype=np.float64)
    df.attrs["xy_x"] = df.loc[valid_xy, "zx|x"].to_numpy(dtype=np.float64)
    df.attrs["xy_y"] = df.loc[valid_xy, "zx|y"].to_numpy(dtype=np.float64)
    cache[path] = df
    return df


def nearest_value(vehicle: pd.DataFrame, col: str, anchor_s: float) -> tuple[float, float]:
    valid_t = vehicle.attrs.get(f"{col}_t")
    valid_v = vehicle.attrs.get(f"{col}_v")
    if valid_t is None or valid_v is None or len(valid_t) == 0 or not math.isfinite(anchor_s):
        return float("nan"), float("nan")
    pos = int(np.searchsorted(valid_t, anchor_s))
    candidates: list[int] = []
    if pos < len(valid_t):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    if not candidates:
        return float("nan"), float("nan")
    best = min(candidates, key=lambda idx: abs(float(valid_t[idx]) - anchor_s))
    return float(valid_v[best]), abs(float(valid_t[best]) - anchor_s)


def nearest_xy_row(vehicle: pd.DataFrame, anchor_s: float) -> dict[str, float] | None:
    valid_t = vehicle.attrs.get("xy_t")
    valid_x = vehicle.attrs.get("xy_x")
    valid_y = vehicle.attrs.get("xy_y")
    if valid_t is None or valid_x is None or valid_y is None or len(valid_t) == 0 or not math.isfinite(anchor_s):
        return None
    pos = int(np.searchsorted(valid_t, anchor_s))
    candidates: list[int] = []
    if pos < len(valid_t):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda idx: abs(float(valid_t[idx]) - anchor_s))
    return {"t_rel_s": float(valid_t[best]), "zx|x": float(valid_x[best]), "zx|y": float(valid_y[best])}


def candidate_times(row: pd.Series) -> list[tuple[str, float]]:
    raw: list[tuple[str, float]] = [
        ("model_anchor", finite_float(row.get("model_anchor_s_v1_8"))),
        ("episode_start", finite_float(row.get("episode_start_s"))),
        ("driver_action", finite_float(row.get("driver_action_onset_s"))),
        ("vehicle_response", finite_float(row.get("vehicle_response_onset_s"))),
        ("condition_peak", finite_float(row.get("condition_peak_s"))),
        ("vehicle_peak", finite_float(row.get("vehicle_peak_s"))),
        ("episode_end", finite_float(row.get("episode_end_s"))),
    ]
    start = finite_float(row.get("episode_start_s"))
    end = finite_float(row.get("episode_end_s"))
    if math.isfinite(start) and math.isfinite(end) and end > start:
        for frac in [0.25, 0.50, 0.75]:
            raw.append((f"episode_q{int(frac * 100)}", start + (end - start) * frac))
    out: list[tuple[str, float]] = []
    seen: set[int] = set()
    for label, t in raw:
        if not math.isfinite(t):
            continue
        key = int(round(t * 100))
        if key in seen:
            continue
        seen.add(key)
        out.append((label, t))
    return out


def map_time_to_road(
    label: str,
    anchor_s: float,
    vehicle: pd.DataFrame,
    tree: Any | None,
    layout: pd.DataFrame | None,
) -> dict[str, Any]:
    lane_curv, lane_curv_gap = nearest_value(vehicle, "zx1|lanecurvatureXY", anchor_s)
    lat_dist, lat_dist_gap = nearest_value(vehicle, "zx1|lateraldistance", anchor_s)
    base = {
        "label": label,
        "time_s": anchor_s,
        "module": "",
        "instance": "",
        "s": np.nan,
        "curvature": np.nan,
        "nearest_dist": np.nan,
        "vehicle_time_gap_s": np.nan,
        "vehicle_x": np.nan,
        "vehicle_y": np.nan,
        "lane_curvature": lane_curv,
        "lane_curvature_time_gap_s": lane_curv_gap,
        "lateral_distance": lat_dist,
        "lateral_distance_time_gap_s": lat_dist_gap,
        "status": "unmapped",
    }
    if tree is None or layout is None:
        base["status"] = "mapper_unavailable"
        return base
    vehicle_row = nearest_xy_row(vehicle, anchor_s)
    if vehicle_row is None:
        base["status"] = "vehicle_xy_missing"
        return base
    xy = np.array([[float(vehicle_row["zx|x"]), float(vehicle_row["zx|y"])]], dtype=np.float64)
    dist, idx = tree.query(xy, k=1)
    layout_row = layout.iloc[int(idx[0])]
    base.update(
        {
            "module": str(layout_row.get("module_name", "")),
            "instance": str(layout_row.get("instance_name", "")),
            "s": finite_float(layout_row.get("s")),
            "curvature": finite_float(layout_row.get("curvature")),
            "nearest_dist": float(dist[0]),
            "vehicle_time_gap_s": abs(float(vehicle_row["t_rel_s"]) - anchor_s),
            "vehicle_x": finite_float(vehicle_row.get("zx|x")),
            "vehicle_y": finite_float(vehicle_row.get("zx|y")),
            "status": "ok",
        }
    )
    return base


def map_episode_context(
    row: pd.Series,
    tree: Any | None,
    layout: pd.DataFrame | None,
    mapper_status: str,
    vehicle_cache: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    path = str(row.get("vehicle_file", ""))
    base = {
        "road_coord_map_status_v1_9": mapper_status,
        "road_coord_modules_seen_v1_9": "",
        "road_coord_instances_seen_v1_9": "",
        "road_coord_dominant_module_v1_9": "",
        "road_coord_dominant_instance_v1_9": "",
        "road_coord_curve_modules_seen_v1_9": "",
        "road_coord_curve_sample_count_v1_9": 0,
        "road_coord_is_curve_v1_9": False,
        "road_coord_anchor_module_v1_9": "",
        "road_coord_anchor_instance_v1_9": "",
        "road_coord_anchor_curvature_v1_9": np.nan,
        "road_coord_anchor_nearest_dist_v1_9": np.nan,
        "road_coord_anchor_vehicle_x_v1_9": np.nan,
        "road_coord_anchor_vehicle_y_v1_9": np.nan,
        "road_coord_nearest_dist_median_v1_9": np.nan,
        "road_coord_nearest_dist_min_v1_9": np.nan,
        "road_coord_nearest_dist_max_v1_9": np.nan,
        "road_coord_mapping_quality_v1_9": "unmapped",
        "vehicle_lane_curvature_anchor_v1_9": np.nan,
        "vehicle_lane_curvature_abs_max_sampled_v1_9": np.nan,
        "vehicle_lateral_distance_anchor_v1_9": np.nan,
        "vehicle_lateral_distance_abs_max_sampled_v1_9": np.nan,
        "vehicle_lane_curvature_curve_hint_v1_9": False,
        "road_coord_sample_trace_v1_9": "",
    }
    try:
        vehicle = load_vehicle_context(path, vehicle_cache)
    except Exception as exc:  # noqa: BLE001
        base["road_coord_map_status_v1_9"] = f"vehicle_read_error:{type(exc).__name__}"
        return base
    maps = [map_time_to_road(label, t, vehicle, tree, layout) for label, t in candidate_times(row)]
    ok_maps = [m for m in maps if m["status"] == "ok"]
    if not ok_maps:
        base["road_coord_map_status_v1_9"] = "no_mapped_time"
        return base
    modules = [m["module"] for m in ok_maps]
    instances = [m["instance"] for m in ok_maps]
    curve_modules = [m["module"] for m in ok_maps if m["module"] in CURVE_MODULES]
    dist_values = np.array([m["nearest_dist"] for m in ok_maps if math.isfinite(float(m["nearest_dist"]))], dtype=float)
    lane_curvs = np.array(
        [m["lane_curvature"] for m in maps if math.isfinite(float(m["lane_curvature"]))], dtype=float
    )
    lat_dists = np.array(
        [m["lateral_distance"] for m in maps if math.isfinite(float(m["lateral_distance"]))], dtype=float
    )
    anchor_map = maps[0] if maps else {}
    quality = "unmapped"
    if len(dist_values):
        median_dist = float(np.median(dist_values))
        if median_dist <= 50:
            quality = "high"
        elif median_dist <= 500:
            quality = "medium"
        elif median_dist <= 1500:
            quality = "low_but_context_usable"
        else:
            quality = "very_low_review"
    trace = ";".join(
        f"{m['label']}:{m['module']}:{finite_float(m['nearest_dist']):.1f}:{finite_float(m['lane_curvature']):.5f}"
        for m in maps
    )
    base.update(
        {
            "road_coord_map_status_v1_9": "ok",
            "road_coord_modules_seen_v1_9": unique_join(modules),
            "road_coord_instances_seen_v1_9": unique_join(instances),
            "road_coord_dominant_module_v1_9": mode_text(modules),
            "road_coord_dominant_instance_v1_9": mode_text(instances),
            "road_coord_curve_modules_seen_v1_9": unique_join(curve_modules),
            "road_coord_curve_sample_count_v1_9": int(len(curve_modules)),
            "road_coord_is_curve_v1_9": bool(len(curve_modules) > 0),
            "road_coord_anchor_module_v1_9": str(anchor_map.get("module", "")),
            "road_coord_anchor_instance_v1_9": str(anchor_map.get("instance", "")),
            "road_coord_anchor_curvature_v1_9": finite_float(anchor_map.get("curvature")),
            "road_coord_anchor_nearest_dist_v1_9": finite_float(anchor_map.get("nearest_dist")),
            "road_coord_anchor_vehicle_x_v1_9": finite_float(anchor_map.get("vehicle_x")),
            "road_coord_anchor_vehicle_y_v1_9": finite_float(anchor_map.get("vehicle_y")),
            "road_coord_nearest_dist_median_v1_9": float(np.median(dist_values)) if len(dist_values) else np.nan,
            "road_coord_nearest_dist_min_v1_9": float(np.min(dist_values)) if len(dist_values) else np.nan,
            "road_coord_nearest_dist_max_v1_9": float(np.max(dist_values)) if len(dist_values) else np.nan,
            "road_coord_mapping_quality_v1_9": quality,
            "vehicle_lane_curvature_anchor_v1_9": finite_float(anchor_map.get("lane_curvature")),
            "vehicle_lane_curvature_abs_max_sampled_v1_9": float(np.nanmax(np.abs(lane_curvs))) if len(lane_curvs) else np.nan,
            "vehicle_lateral_distance_anchor_v1_9": finite_float(anchor_map.get("lateral_distance")),
            "vehicle_lateral_distance_abs_max_sampled_v1_9": float(np.nanmax(np.abs(lat_dists))) if len(lat_dists) else np.nan,
            "vehicle_lane_curvature_curve_hint_v1_9": bool(
                len(lane_curvs) and float(np.nanmax(np.abs(lane_curvs))) >= LANE_CURVATURE_HINT
            ),
            "road_coord_sample_trace_v1_9": trace,
        }
    )
    return base


def has_vehicle_dynamic_candidate(row: pd.Series) -> bool:
    return (
        finite_float(row.get("peak_abs_roll")) >= ROLL_ANGLE_CANDIDATE_RAD
        or finite_float(row.get("peak_abs_roll_rate")) >= ROLL_RATE_CANDIDATE_RADPS
        or finite_float(row.get("peak_abs_ay")) >= AY_DYNAMIC_CANDIDATE
    )


def curve_height_abnormal(row: pd.Series) -> bool:
    """Height abnormality for curve episodes only; not used to define curve."""
    z_rise = finite_float(row.get("z_rise_from_start_v1_4"), 0.0)
    z_resid_range = finite_float(row.get("z_residual_range_v1_3"), 0.0)
    z_resid_rate = finite_float(row.get("z_residual_rate_peak_v1_3"), 0.0)
    return (
        z_rise >= CURVE_HEIGHT_RISE_ABNORMAL_M
        or z_resid_range >= CURVE_Z_RESIDUAL_RANGE_ABNORMAL
        or z_resid_rate >= CURVE_Z_RESIDUAL_RATE_ABNORMAL
    )


def classify_v1_9(row: pd.Series) -> tuple[str, str, str, bool, bool, bool, bool]:
    coord_curve = bool_value(row.get("road_coord_is_curve_v1_9"))
    dynamic = has_vehicle_dynamic_candidate(row)
    old_decision = str(row.get("v1_8_decision", ""))
    old_curve_train = old_decision.startswith("train_curve_")
    old_noncurve_train = old_decision == "train_noncurve_target_extreme"
    old_control = old_decision == "control_noncurve"
    old_defer = old_decision.startswith("defer_")
    old_discard_curve = old_decision == "discard_curve_height_or_z_abnormal"
    mapping_quality = str(row.get("road_coord_mapping_quality_v1_9", ""))

    if mapping_quality == "very_low_review":
        reason = "道路坐标最近邻距离过大，先不直接决定弯道/非弯道，需要复核道路映射"
        return "review_road_coordinate_mapping_uncertain", reason, reason, False, True, False, False

    if coord_curve:
        if curve_height_abnormal(row):
            reason = "道路坐标确认在弯道，但高度/姿态形态异常，疑似上斜坡、下路边或非正常过弯"
            return "discard_curve_coord_height_or_pose_abnormal", reason, reason, False, False, False, True
        if dynamic:
            reason = "道路坐标确认在弯道，且车辆侧倾/横滚/横向动态明显，可作为弯道极限或近极限候选"
            return "train_curve_coord_valid_roll_candidate", reason, reason, True, False, False, False
        reason = "道路坐标确认在弯道，车辆动态较弱或更像正常过弯，作为弯道普通/弱侧倾训练候选"
        return "train_curve_coord_valid_normal_or_weak", reason, reason, True, False, False, False

    if old_noncurve_train:
        reason = "道路坐标显示非弯道，继承非弯道极限/近极限主训练候选"
        return "train_noncurve_target_extreme", reason, reason, True, False, False, False

    if old_curve_train and dynamic:
        reason = "原先按弯道纳入，但道路坐标显示非弯道；车辆动态明显，改为非弯道动态候选"
        return "train_noncurve_recovered_from_false_curve_dynamic", reason, reason, True, False, False, False

    if old_curve_train:
        reason = "原先按弯道纳入，但道路坐标显示非弯道且动态较弱，改为复核样本"
        return "review_noncurve_false_curve_weak", reason, reason, False, True, False, False

    if old_discard_curve and dynamic:
        reason = "原先按弯道高度异常排除，但道路坐标显示非弯道；车辆动态明显，需复核是否为非弯道极限事件"
        return "review_noncurve_recovered_from_height_rule_conflict", reason, reason, False, True, False, False

    if old_control:
        reason = "道路坐标显示非弯道，继承非弯道对照样本"
        return "control_noncurve", reason, reason, False, False, True, False

    if old_defer:
        reason = "道路坐标显示非弯道，继承历史待复核样本"
        return "defer_noncurve_prior_review", reason, reason, False, True, False, False

    reason = "道路坐标显示非弯道，且不属于当前训练候选"
    return "discard_noncurve_prior_review", reason, reason, False, False, False, True


def plot_episode_v1_9(row: pd.Series, out_path: Path, cache: dict) -> None:
    plot_row = row.copy()
    plot_row["v1_8_decision"] = row.get("v1_9_decision", "")
    v18.plot_episode_v1_8(plot_row, out_path, cache)


def make_review_figures(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_panel_v1_9_path"] = ""
    cache: dict[str, Any] = {}
    specs = [
        (
            "00_道路坐标确认弯道_训练候选",
            df["v1_9_decision"].isin(
                ["train_curve_coord_valid_roll_candidate", "train_curve_coord_valid_normal_or_weak"]
            ),
            ["peak_abs_roll", "peak_abs_roll_rate", "peak_abs_ay"],
            12,
        ),
        (
            "01_道路坐标确认弯道_高度姿态异常排除",
            df["v1_9_decision"].eq("discard_curve_coord_height_or_pose_abnormal"),
            ["z_residual_range_v1_3", "z_rise_from_start_v1_4", "peak_abs_roll"],
            12,
        ),
        (
            "02_原误判弯道_坐标显示非弯道_动态可训练",
            df["v1_9_decision"].eq("train_noncurve_recovered_from_false_curve_dynamic"),
            ["peak_abs_ay", "peak_abs_roll", "peak_abs_roll_rate"],
            12,
        ),
        (
            "03_非弯道主训练候选",
            df["v1_9_decision"].eq("train_noncurve_target_extreme"),
            ["vehicle_score_peak", "peak_abs_ay", "peak_abs_roll"],
            12,
        ),
        (
            "04_道路坐标映射不确定_优先复核",
            df["v1_9_decision"].eq("review_road_coordinate_mapping_uncertain"),
            ["road_coord_nearest_dist_median_v1_9", "peak_abs_ay", "peak_abs_roll"],
            10,
        ),
        (
            "05_道路坐标显示非弯道_历史舍弃抽查",
            df["v1_9_decision"].eq("discard_noncurve_prior_review"),
            ["peak_abs_ay", "peak_abs_roll", "steer_angle_range"],
            10,
        ),
    ]
    for folder, mask, sort_cols, max_n in specs:
        folder_path = FIG_DIR / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        subset = df.loc[mask].copy()
        if subset.empty:
            continue
        for col in sort_cols:
            if col not in subset.columns:
                subset[col] = 0.0
            subset[col] = pd.to_numeric(subset[col], errors="coerce").fillna(0.0).abs()
        subset = subset.sort_values(sort_cols, ascending=False)
        for idx, row in subset.head(max_n).iterrows():
            out_path = folder_path / f"{idx:04d}_{row['episode_uid']}.png"
            if not out_path.exists():
                plot_episode_v1_9(row, out_path, cache)
            if out_path.exists():
                df.at[idx, "review_panel_v1_9_path"] = str(out_path)
    return df


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "（无）"
    return df.to_markdown(index=False)


def write_tables(df: pd.DataFrame) -> None:
    df.to_csv(TABLE_DIR / "record_level_episodes_all_v1_9.csv", index=False, encoding="utf-8-sig")
    df[df["is_train_candidate_v1_9"]].to_csv(
        TABLE_DIR / "train_candidate_all_episodes_v1_9.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_9_decision"].str.startswith("train_curve_", na=False)].to_csv(
        TABLE_DIR / "train_candidate_curve_coord_episodes_v1_9.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_9_decision"].str.startswith("train_noncurve_", na=False)].to_csv(
        TABLE_DIR / "train_candidate_noncurve_episodes_v1_9.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_9_decision"].eq("train_noncurve_recovered_from_false_curve_dynamic")].to_csv(
        TABLE_DIR / "false_curve_recovered_noncurve_dynamic_episodes_v1_9.csv",
        index=False,
        encoding="utf-8-sig",
    )
    df[df["v1_9_decision"].str.startswith("review_", na=False)].to_csv(
        TABLE_DIR / "manual_review_episodes_v1_9.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_discarded_v1_9"]].to_csv(
        TABLE_DIR / "discarded_episodes_v1_9.csv", index=False, encoding="utf-8-sig"
    )
    summary = (
        df.groupby("v1_9_decision", dropna=False)
        .agg(
            v1_9_decision_cn=("v1_9_decision_cn", "first"),
            count=("v1_9_decision", "size"),
            train_count=("is_train_candidate_v1_9", "sum"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    summary.to_csv(TABLE_DIR / "record_episode_v1_9_decision_summary.csv", index=False, encoding="utf-8-sig")
    module_summary = (
        df.groupby(["road_coord_dominant_module_v1_9", "v1_9_decision"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["road_coord_dominant_module_v1_9", "count"], ascending=[True, False])
    )
    module_summary.to_csv(TABLE_DIR / "road_coord_module_summary_v1_9.csv", index=False, encoding="utf-8-sig")
    audit = (
        df.groupby(["v1_8_decision", "road_coord_is_curve_v1_9", "v1_9_decision"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    audit.to_csv(TABLE_DIR / "metadata_vs_coord_curve_audit_v1_9.csv", index=False, encoding="utf-8-sig")


def write_report(df: pd.DataFrame) -> None:
    summary = pd.read_csv(TABLE_DIR / "record_episode_v1_9_decision_summary.csv")
    module_summary = pd.read_csv(TABLE_DIR / "road_coord_module_summary_v1_9.csv")
    audit = pd.read_csv(TABLE_DIR / "metadata_vs_coord_curve_audit_v1_9.csv")
    train_n = int(df["is_train_candidate_v1_9"].fillna(False).astype(bool).sum())
    curve_coord_n = int(df["road_coord_is_curve_v1_9"].fillna(False).astype(bool).sum())
    curve_train_n = int(df["v1_9_decision"].str.startswith("train_curve_", na=False).sum())
    noncurve_train_n = int(df["v1_9_decision"].str.startswith("train_noncurve_", na=False).sum())
    false_curve_recovered_n = int(df["v1_9_decision"].eq("train_noncurve_recovered_from_false_curve_dynamic").sum())
    uncertain_n = int(df["v1_9_decision"].eq("review_road_coordinate_mapping_uncertain").sum())
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    text = f"""# v1.9 道路坐标版 episode 样本重分总结

生成时间：{now}

## 这版为什么要做

用户指出两个关键问题：

1. 弯道不能只凭高度判断，因为不是所有弯道都是下坡。
2. 高度只能用于判断异常，例如疑似上斜坡、下路边或高度跳变；弯道本身应该根据道路坐标、道路中心线、道路曲率或道路模块来判断。

所以 v1.9 废弃了“高度下降约等于弯道”的错误思路。当前规则改为：

- 用车辆原始 `zx|x / zx|y` 在 episode 多个关键时刻匹配 `full_centerline_layout.csv`；
- 根据匹配到的 `curve1 / curve2 / curve3` 判断是否处在弯道道路坐标上；
- 同时保留车辆文件中的 `zx1|lanecurvatureXY` 作为辅助核对；
- 高度 `z`、高度残差、横滚、横向加速度只用于判断弯道内是否异常或是否有明显车身姿态响应。

## 总体数量

- 全部 episode：{len(df)}
- 道路坐标判定为弯道上下文：{curve_coord_n}
- 当前训练候选总数：{train_n}
- 其中弯道训练候选：{curve_train_n}
- 其中非弯道训练候选：{noncurve_train_n}
- 原先误判为弯道、道路坐标显示非弯道但车辆动态明显、被转为非弯道候选：{false_curve_recovered_n}
- 道路坐标映射距离过大、需要复核：{uncertain_n}

## v1.9 决策分布

{md_table(summary)}

## 道路坐标模块分布

{md_table(module_summary.head(40))}

## v1.8 与道路坐标判定的冲突审计

下面这个表用于看“旧规则/旧上下文判为弯道”的样本，在道路坐标下是否仍然是弯道。

{md_table(audit.head(40))}

## 当前解释

1. 弯道判断已经改成道路坐标判断，不再由高度下降或高度起伏决定。
2. 平路弯道会被保留，因为是否弯道来自 `curve1/curve2/curve3` 道路模块，而不是 `z` 是否下降。
3. 下坡直道不会因为高度下降被判为弯道；如果道路坐标显示它不是弯道，它会进入非弯道候选、复核或排除。
4. 高度仍然有用，但用途变了：它只用于判断弯道内是否出现非正常高度变化，例如疑似上斜坡、下路边、坐标/路面异常。
5. 道路中心线最近邻距离有时较大，所以本版保留了 `road_coord_mapping_quality_v1_9`。距离很大的样本不直接作为强结论，进入复核。

## 输出文件

- 全量表：`{TABLE_DIR / "record_level_episodes_all_v1_9.csv"}`
- 全部训练候选：`{TABLE_DIR / "train_candidate_all_episodes_v1_9.csv"}`
- 道路坐标弯道训练候选：`{TABLE_DIR / "train_candidate_curve_coord_episodes_v1_9.csv"}`
- 非弯道训练候选：`{TABLE_DIR / "train_candidate_noncurve_episodes_v1_9.csv"}`
- 原误判弯道但道路坐标显示非弯道的动态候选：`{TABLE_DIR / "false_curve_recovered_noncurve_dynamic_episodes_v1_9.csv"}`
- 待复核表：`{TABLE_DIR / "manual_review_episodes_v1_9.csv"}`
- 舍弃表：`{TABLE_DIR / "discarded_episodes_v1_9.csv"}`
- 冲突审计表：`{TABLE_DIR / "metadata_vs_coord_curve_audit_v1_9.csv"}`
- 道路模块统计表：`{TABLE_DIR / "road_coord_module_summary_v1_9.csv"}`
- 复核图目录：`{FIG_DIR}`

## 下一步建议

先看 v1.9 的复核图，重点看三类：

1. 道路坐标确认弯道的训练候选；
2. 道路坐标确认弯道但高度/姿态异常的排除样本；
3. 原先被当弯道、现在道路坐标显示非弯道但车辆动态明显的样本。

确认这三类的语义后，再决定是否用 v1.9 训练车辆-only 模型。当前没有训练模型。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def append_notes(df: pd.DataFrame) -> None:
    train_n = int(df["is_train_candidate_v1_9"].fillna(False).astype(bool).sum())
    curve_train_n = int(df["v1_9_decision"].str.startswith("train_curve_", na=False).sum())
    noncurve_train_n = int(df["v1_9_decision"].str.startswith("train_noncurve_", na=False).sum())
    curve_coord_n = int(df["road_coord_is_curve_v1_9"].fillna(False).astype(bool).sum())
    block = (
        "## 2026-05-22 完整记录级 episode 样本集 v1.9 道路坐标判弯道\n\n"
        "- 根据用户纠正，本轮废弃“高度下降判弯道”的错误逻辑；弯道改由车辆 `zx|x/zx|y` 匹配道路中心线 `full_centerline_layout.csv` 后的 `curve1/curve2/curve3` 判断。\n"
        "- 高度 z 只作为异常证据，用于判断疑似上斜坡、下路边或非正常高度跳变；不是弯道定义依据。\n"
        f"- 全部 episode `{len(df)}` 个，道路坐标弯道上下文 `{curve_coord_n}` 个，训练候选 `{train_n}` 个，其中弯道候选 `{curve_train_n}` 个，非弯道候选 `{noncurve_train_n}` 个。\n"
        f"- 用户查看版报告：`{REPORT_PATH}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        if "## 2026-05-22 完整记录级 episode 样本集 v1.9 道路坐标判弯道" not in raw:
            path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")
    artifact = (
        "## 2026-05-22 完整记录级 episode 样本集 v1.9 道路坐标判弯道\n\n"
        f"- 用户查看版报告：`{REPORT_PATH}`\n"
        f"- 全量表：`{TABLE_DIR / 'record_level_episodes_all_v1_9.csv'}`\n"
        f"- 全部训练候选：`{TABLE_DIR / 'train_candidate_all_episodes_v1_9.csv'}`\n"
        f"- 道路坐标弯道训练候选：`{TABLE_DIR / 'train_candidate_curve_coord_episodes_v1_9.csv'}`\n"
        f"- 非弯道训练候选：`{TABLE_DIR / 'train_candidate_noncurve_episodes_v1_9.csv'}`\n"
        f"- 冲突审计表：`{TABLE_DIR / 'metadata_vs_coord_curve_audit_v1_9.csv'}`\n"
        f"- 复核图目录：`{FIG_DIR}`\n"
    )
    raw = ARTIFACT_INDEX.read_text(encoding="utf-8") if ARTIFACT_INDEX.exists() else ""
    if "## 2026-05-22 完整记录级 episode 样本集 v1.9 道路坐标判弯道" not in raw:
        ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not V18_ALL.exists():
        raise FileNotFoundError(V18_ALL)
    if not ROAD_LAYOUT.exists():
        raise FileNotFoundError(ROAD_LAYOUT)
    df = pd.read_csv(V18_ALL, encoding="utf-8-sig", low_memory=False)
    tree, layout, mapper_status = load_road_mapper()
    vehicle_cache: dict[str, pd.DataFrame] = {}
    road_contexts = [
        map_episode_context(row, tree, layout, mapper_status, vehicle_cache) for _, row in df.iterrows()
    ]
    road_df = pd.DataFrame(road_contexts)
    df = pd.concat([df.reset_index(drop=True), road_df.reset_index(drop=True)], axis=1)
    decisions = df.apply(classify_v1_9, axis=1, result_type="expand")
    decisions.columns = [
        "v1_9_decision",
        "v1_9_decision_cn",
        "v1_9_decision_detail_cn",
        "is_train_candidate_v1_9",
        "is_deferred_v1_9",
        "is_control_candidate_v1_9",
        "is_discarded_v1_9",
    ]
    df = pd.concat([df, decisions], axis=1)
    df["review_panel_v1_9_path"] = ""
    write_tables(df)
    write_report(df)
    df = make_review_figures(df)
    write_tables(df)
    write_report(df)
    append_notes(df)
    print(pd.read_csv(TABLE_DIR / "record_episode_v1_9_decision_summary.csv").to_string(index=False))
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
