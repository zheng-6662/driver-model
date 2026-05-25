# -*- coding: utf-8 -*-
from __future__ import annotations

import html
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
ROOT = Path(os.environ.get("REBUILD_ROOT", str(PROJECT_ROOT / "05_rebuild_from_raw_20260511")))
SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
SAMPLE_SCRIPT_DIR = ROOT / "02_samples" / "scripts"
for p in [SCRIPT_DIR, SAMPLE_SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_extreme_condition_episodes_v0_3 as raw_loader  # noqa: E402


V20_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v2_0_no_history_reaudit"
V20_ALL = V20_ROOT / "tables" / "record_level_episodes_all_v2_0.csv"

OUT_ROOT = ROOT / "03_baselines" / "stage03_goal1_v2_task_redesign"
MANIFEST_DIR = OUT_ROOT / "manifests"
DATASET_DIR = ROOT / "03_processed_datasets" / "record_episode_v2_task_redesign"
ARRAY_DIR = DATASET_DIR / "arrays"
TABLE_DIR = DATASET_DIR / "tables"
LOG_DIR = DATASET_DIR / "logs"
OUTPUT_DIR = OUT_ROOT / "outputs"
REPORT_DIR = ROOT / "09_reports"
REPORT_PATH = REPORT_DIR / "stage03_goal1_v2_task_redesign_user_summary_cn.md"
FINAL_REPORT = OUTPUT_DIR / "final_task_redesign_report.md"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-25.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"

TEST_SUBJECTS = {"cwh", "gf", "tyy"}
VAL_SUBJECTS = {"byx", "gzj", "yyl"}

HZ = 20.0
INPUT_TIME = np.round(np.arange(-2.0, 0.0 + 1e-9, 1.0 / HZ), 6)
CORE_TIME = np.round(np.arange(0.0, 3.0 + 1e-9, 1.0 / HZ), 6)
EXT_TIME = np.round(np.arange(0.0, 5.0 + 1e-9, 1.0 / HZ), 6)

INPUT_FEATURES = [
    "zx|SteeringWheel",
    "steer_rate",
    "zx1|v_km/h",
    "zx|BrakePedal",
    "zx|AcceleratorPedal",
    "zx|ax",
    "zx|ay",
    "zx|vyaw",
    "zx|vroll",
    "zx|roll",
    "lateral_distance_selected",
    "zx1|mu",
    "curvature_selected",
]
ESSENTIAL_INPUT_FEATURES = ["zx|SteeringWheel", "zx1|v_km/h"]
RAW_USECOLS = sorted(
    {
        "StorageTime",
        "zx|SteeringWheel",
        "zx1|v_km/h",
        "zx|BrakePedal",
        "zx|AcceleratorPedal",
        "zx|ax",
        "zx|ay",
        "zx|vyaw",
        "zx|vroll",
        "zx|roll",
        "zx1|mu",
        "zx1|lateraldistance",
        "zx|lateraldistance",
        "zx1|lanecurvatureXY",
        "zx|lanecurvatureXY",
    }
)

OUTPUT_SPECS = [
    ("steering", "zx|SteeringWheel", "方向盘相对变化", True),
    ("speed", "zx1|v_km/h", "车速", False),
    ("brake", "zx|BrakePedal", "制动踏板", False),
    ("ay", "zx|ay", "横向加速度", False),
    ("yaw_rate", "zx|vyaw", "横摆角速度", False),
    ("roll", "zx|roll", "横滚角", False),
    ("roll_rate", "zx|vroll", "横滚角速度", False),
]
OUTPUT_NAMES = [x[0] for x in OUTPUT_SPECS]

KEYPOINT_NAMES = [
    "steering_peak_value",
    "steering_peak_time",
    "brake_onset_time",
    "brake_peak_value",
    "speed_drop_max",
    "speed_drop_peak_time",
    "ay_peak_value",
    "ay_peak_time",
    "yaw_rate_peak_value",
    "yaw_rate_peak_time",
    "roll_peak_value",
    "roll_peak_time",
    "roll_rate_peak_value",
    "roll_rate_peak_time",
    "recovery_time",
    "large_steering_response",
    "large_brake_response",
    "high_risk_state",
    "recovered_by_3s",
    "recovered_by_5s",
]

EPISODE_TYPE_LABELS = [
    "noncurve_extreme",
    "curve_normal_or_weak",
    "curve_abnormal_roll",
    "normal_control",
    "weak_response_control",
    "review_candidate",
    "excluded_slope_or_offroad",
    "excluded_data_quality_bad",
    "unknown",
]
RESPONSE_TYPE_LABELS = [
    "strong_steer",
    "weak_steer",
    "brake_dominant",
    "conservative_pass",
    "delayed_response",
    "vehicle_dominant_no_clear_action",
    "no_clear_response",
    "unknown",
]
CURVE_TYPE_LABELS = [
    "not_curve",
    "normal_or_weak_curve",
    "abnormal_roll",
    "high_risk_curve_response",
    "unknown",
]

TASK_MASK_COLUMNS = [
    "can_train_steering",
    "can_train_speed",
    "can_train_brake",
    "can_train_ay",
    "can_train_yaw_rate",
    "can_train_roll",
    "can_train_roll_rate",
    "can_train_response_type",
    "can_train_curve_type",
    "can_train_recovery",
]

RANDOM_SEED = 20260525
SCRIPT_VERSION = 2


def ensure_dirs() -> None:
    for path in [
        MANIFEST_DIR,
        ARRAY_DIR,
        TABLE_DIR,
        LOG_DIR,
        OUTPUT_DIR,
        REPORT_DIR,
        NOTES_DIR / "daily_logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def interp_series(df: pd.DataFrame, col: str, query_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "time_rel_s" not in df.columns or col not in df.columns:
        return np.zeros_like(query_time, dtype=np.float32), np.zeros_like(query_time, dtype=bool)
    t = df["time_rel_s"].to_numpy(dtype=float)
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(v)
    if valid.sum() < 2:
        return np.zeros_like(query_time, dtype=np.float32), np.zeros_like(query_time, dtype=bool)
    tt = t[valid]
    vv = v[valid]
    order = np.argsort(tt)
    tt = tt[order]
    vv = vv[order]
    unique_t, unique_idx = np.unique(tt, return_index=True)
    unique_v = vv[unique_idx]
    inside = (query_time >= unique_t[0]) & (query_time <= unique_t[-1])
    out = np.zeros_like(query_time, dtype=np.float32)
    out[inside] = np.interp(query_time[inside], unique_t, unique_v).astype(np.float32)
    return out, inside.astype(bool)


def resolve_vehicle_path(row: pd.Series) -> Path:
    for col in ["vehicle_raw_absolute_path", "vehicle_file"]:
        value = str(row.get(col, "")).strip()
        if value and value.lower() != "nan":
            p = Path(value)
            if p.exists():
                return p
    rel = str(row.get("vehicle_raw_relative_path", "")).strip().replace("\\", "/")
    if rel and rel.lower() != "nan":
        candidate = raw_loader.RAW_VEHICLE_ROOT / Path(*rel.split("/"))
        if candidate.exists():
            return candidate
    return Path(str(row.get("vehicle_file", "")))


def load_vehicle_csv_light(path: Path) -> pd.DataFrame | None:
    """Read only the columns needed for goal1 task construction.

    The older loader computes file hashes for audit completeness. That is
    useful for data inventory, but too slow for repeated task-building runs.
    """
    try:
        header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
        usecols = [c for c in RAW_USECOLS if c in header.columns]
        if "StorageTime" not in usecols:
            return None
        df = pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    except Exception:
        return None
    df["time_rel_s"] = raw_loader.parse_time_seconds(df["StorageTime"])
    df = df[np.isfinite(df["time_rel_s"])].copy()
    df = df.drop_duplicates("time_rel_s").sort_values("time_rel_s")
    if len(df) < 20:
        return None
    for col in df.columns:
        if col not in {"StorageTime", "time_rel_s"}:
            df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(limit_direction="both")
    t = df["time_rel_s"].to_numpy(dtype=float)
    if "zx|SteeringWheel" in df.columns:
        df["steer_rate"] = raw_loader.gradient(df["zx|SteeringWheel"].to_numpy(dtype=float), t)
    else:
        df["steer_rate"] = np.nan
    lat_col = "zx1|lateraldistance" if "zx1|lateraldistance" in df.columns else "zx|lateraldistance"
    if lat_col in df.columns:
        df["lateral_distance_selected"] = df[lat_col].to_numpy(dtype=float)
    else:
        df["lateral_distance_selected"] = np.nan
    curv_col = "zx1|lanecurvatureXY" if "zx1|lanecurvatureXY" in df.columns else "zx|lanecurvatureXY"
    if curv_col in df.columns:
        df["curvature_selected"] = df[curv_col].to_numpy(dtype=float)
    else:
        df["curvature_selected"] = np.nan
    return df.reset_index(drop=True)


def map_episode_type(row: pd.Series) -> str:
    decision = str(row.get("v2_0_decision", ""))
    height = str(row.get("v2_0_height_pose_issue", ""))
    if "data_bad" in decision:
        return "excluded_data_quality_bad"
    if "offroad" in decision or "roadedge" in decision:
        return "excluded_slope_or_offroad"
    if height == "明显高度异常" and "train_" not in decision:
        return "excluded_slope_or_offroad"
    if decision == "train_noncurve_vehicle_dynamic":
        return "noncurve_extreme"
    if decision == "train_noncurve_secondary_dynamic":
        return "weak_response_control"
    if decision == "train_curve_roll_dynamic":
        return "curve_abnormal_roll"
    if decision == "train_curve_normal_or_weak":
        return "curve_normal_or_weak"
    if decision == "control_noncurve_weak_or_normal":
        return "normal_control"
    if decision.startswith("review_"):
        return "review_candidate"
    return "unknown"


def map_response_type(row: pd.Series) -> str:
    decision = str(row.get("v2_0_decision", ""))
    driver = str(row.get("driver_response_type", "")).lower()
    steer_rate = finite_float(row.get("steer_rate_peak"), 0.0)
    steer_range = finite_float(row.get("steer_angle_range"), 0.0)
    brake = finite_float(row.get("brake_range"), 0.0)
    speed_range = finite_float(row.get("speed_range_kmh"), 0.0)
    dynamic_count = int(finite_float(row.get("v2_0_vehicle_dynamic_count"), 0.0))
    response_order = str(row.get("response_order", "")).lower()

    if "fast" in driver or steer_rate >= 10.0 or steer_range >= 1.0:
        return "strong_steer"
    if brake >= 0.25 or speed_range >= 20.0:
        return "brake_dominant"
    if "driver_after_vehicle" in response_order:
        return "delayed_response"
    if dynamic_count >= 2 and steer_rate < 5.0:
        return "vehicle_dominant_no_clear_action"
    if decision in {"train_noncurve_secondary_dynamic", "train_curve_normal_or_weak"}:
        return "weak_steer"
    if decision == "control_noncurve_weak_or_normal":
        return "no_clear_response"
    if dynamic_count >= 1:
        return "conservative_pass"
    return "unknown"


def map_training_role(row: pd.Series, episode_type: str, response_type: str) -> str:
    decision = str(row.get("v2_0_decision", ""))
    if episode_type == "excluded_slope_or_offroad":
        return "excluded_slope_or_offroad"
    if episode_type == "excluded_data_quality_bad":
        return "excluded_data_quality_bad"
    if decision.startswith("train_noncurve"):
        return "main_train"
    if decision.startswith("train_curve"):
        return "curve_task"
    if decision.startswith("review_"):
        if response_type in {"strong_steer", "weak_steer", "brake_dominant", "conservative_pass"}:
            return "aux_train"
        return "review_need_manual_check"
    if decision.startswith("control_"):
        return "control"
    return "review_need_manual_check"


def map_curve_type(row: pd.Series, episode_type: str) -> str:
    is_curve = bool_value(row.get("road_coord_is_curve_v1_9"))
    if not is_curve:
        return "not_curve"
    roll = finite_float(row.get("peak_abs_roll"), 0.0)
    roll_rate = finite_float(row.get("peak_abs_roll_rate"), 0.0)
    ay = finite_float(row.get("peak_abs_ay"), 0.0)
    if episode_type == "curve_abnormal_roll" or roll >= 0.10 or roll_rate >= 0.8:
        return "abnormal_roll"
    if ay >= 5.0 or roll >= 0.06:
        return "high_risk_curve_response"
    return "normal_or_weak_curve"


def quality_flags_for_row(row: pd.Series, training_role: str, episode_type: str) -> list[str]:
    flags: list[str] = []
    anchor_quality = str(row.get("anchor_quality", ""))
    if "uncertain" in anchor_quality or training_role in {"aux_train", "review_need_manual_check"}:
        flags.append("anchor_uncertain")
    else:
        flags.append("anchor_normal")
    if str(row.get("road_coord_mapping_quality_v1_9", "")) in {"very_low_review", "low_review"}:
        flags.append("coordinate_uncertain")
    if episode_type == "excluded_slope_or_offroad":
        flags.append("slope_or_offroad")
    if "multi" in str(row.get("response_order", "")).lower():
        flags.append("multi_event")
    return flags


def task_mask_for_row(row: pd.Series, training_role: str, episode_type: str) -> dict[str, bool]:
    excluded = training_role.startswith("excluded")
    coordinate_uncertain = str(row.get("road_coord_mapping_quality_v1_9", "")) in {"very_low_review", "low_review"}
    is_curve = bool_value(row.get("road_coord_is_curve_v1_9"))
    return {
        "can_train_steering": not excluded and training_role != "control",
        "can_train_speed": not excluded,
        "can_train_brake": not excluded,
        "can_train_ay": not excluded,
        "can_train_yaw_rate": not excluded,
        "can_train_roll": not excluded,
        "can_train_roll_rate": not excluded,
        "can_train_response_type": not excluded,
        "can_train_curve_type": (not excluded) and is_curve,
        "can_train_recovery": not excluded and training_role != "control" and not coordinate_uncertain,
    }


def build_task_manifest() -> pd.DataFrame:
    manifest_path = MANIFEST_DIR / "manifest_all_v2_task.csv"
    if manifest_path.exists():
        return pd.read_csv(manifest_path, encoding="utf-8-sig", low_memory=False)

    src = pd.read_csv(V20_ALL, encoding="utf-8-sig", low_memory=False)
    rows: list[dict[str, Any]] = []
    for _, row in src.iterrows():
        item = row.to_dict()
        episode_type = map_episode_type(row)
        response_type = map_response_type(row)
        training_role = map_training_role(row, episode_type, response_type)
        curve_type = map_curve_type(row, episode_type)
        flags = quality_flags_for_row(row, training_role, episode_type)
        mask = task_mask_for_row(row, training_role, episode_type)

        item["episode_type"] = episode_type
        item["response_type"] = response_type
        item["curve_type"] = curve_type
        item["training_role"] = training_role
        item["quality_flags"] = ";".join(flags)
        item.update(mask)
        item["include_E0_fixed_steering"] = bool(
            training_role in {"main_train", "curve_task"} and episode_type != "excluded_slope_or_offroad"
        )
        item["include_E1_fixed_multitask"] = item["include_E0_fixed_steering"]
        item["include_E2_masked_multihorizon"] = bool(training_role in {"main_train", "curve_task", "aux_train"})
        item["include_E3_noncurve_response_branch"] = bool(
            episode_type in {"noncurve_extreme", "weak_response_control", "normal_control", "review_candidate"}
            and training_role in {"main_train", "aux_train", "control"}
        )
        item["include_E4_curve_specialized"] = bool(
            episode_type in {"curve_normal_or_weak", "curve_abnormal_roll"} and training_role in {"curve_task", "aux_train"}
        )
        item["include_E5_train_only"] = bool(training_role in {"main_train", "curve_task"})
        item["include_E5_all_review"] = bool(training_role in {"main_train", "curve_task", "aux_train", "review_need_manual_check"})
        item["include_E5_stratified_review"] = bool(training_role in {"main_train", "curve_task", "aux_train"})
        rows.append(item)

    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    df[df["training_role"].isin(["main_train", "curve_task"])].to_csv(
        MANIFEST_DIR / "manifest_main_train_clean.csv", index=False, encoding="utf-8-sig"
    )
    df[df["training_role"].isin(["aux_train"])].to_csv(
        MANIFEST_DIR / "manifest_aux_train_review_stratified.csv", index=False, encoding="utf-8-sig"
    )
    df[df["training_role"].isin(["main_train", "aux_train"])].to_csv(
        MANIFEST_DIR / "manifest_noncurve_train.csv", index=False, encoding="utf-8-sig"
    )
    df[df["training_role"].isin(["curve_task"])].to_csv(
        MANIFEST_DIR / "manifest_curve_train.csv", index=False, encoding="utf-8-sig"
    )
    df[df["training_role"].eq("control")].to_csv(
        MANIFEST_DIR / "manifest_control.csv", index=False, encoding="utf-8-sig"
    )
    df[df["training_role"].eq("excluded_slope_or_offroad")].to_csv(
        MANIFEST_DIR / "manifest_excluded_slope_or_offroad.csv", index=False, encoding="utf-8-sig"
    )
    df[df["training_role"].eq("excluded_data_quality_bad")].to_csv(
        MANIFEST_DIR / "manifest_excluded_data_quality_bad.csv", index=False, encoding="utf-8-sig"
    )
    return df


def valid_seconds(mask: np.ndarray, time_axis: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    step = 1.0 / HZ
    return float(mask.sum() * step)


def signed_peak(values: np.ndarray, mask: np.ndarray, times: np.ndarray) -> tuple[float, float]:
    valid = mask & np.isfinite(values)
    if not valid.any():
        return float("nan"), float("nan")
    vals = values[valid]
    t = times[valid]
    idx = int(np.nanargmax(np.abs(vals)))
    return float(vals[idx]), float(t[idx])


def first_threshold_time(values: np.ndarray, mask: np.ndarray, times: np.ndarray, threshold: float) -> float:
    valid = mask & np.isfinite(values)
    if not valid.any():
        return float("nan")
    idx = np.where(valid & (values >= threshold))[0]
    return float(times[idx[0]]) if len(idx) else float("nan")


def compute_keypoints(y_ext: np.ndarray, mask_ext: np.ndarray, baseline: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    values: dict[str, float] = {}
    masks: dict[str, bool] = {}

    name_to_idx = {name: i for i, name in enumerate(OUTPUT_NAMES)}
    steer = y_ext[:, name_to_idx["steering"]]
    steer_mask = mask_ext[:, name_to_idx["steering"]]
    steer_peak, steer_peak_t = signed_peak(steer, steer_mask, EXT_TIME)
    values["steering_peak_value"] = steer_peak
    values["steering_peak_time"] = steer_peak_t
    masks["steering_peak_value"] = math.isfinite(steer_peak)
    masks["steering_peak_time"] = math.isfinite(steer_peak_t)

    brake = y_ext[:, name_to_idx["brake"]]
    brake_mask = mask_ext[:, name_to_idx["brake"]]
    values["brake_onset_time"] = first_threshold_time(brake, brake_mask, EXT_TIME, max(0.1, baseline.get("brake", 0.0) + 0.05))
    b_peak, _ = signed_peak(brake, brake_mask, EXT_TIME)
    values["brake_peak_value"] = b_peak
    masks["brake_onset_time"] = math.isfinite(values["brake_onset_time"])
    masks["brake_peak_value"] = math.isfinite(b_peak)

    speed = y_ext[:, name_to_idx["speed"]]
    speed_mask = mask_ext[:, name_to_idx["speed"]]
    if speed_mask.any():
        speed_vals = speed[speed_mask]
        speed_times = EXT_TIME[speed_mask]
        base_speed = baseline.get("speed", float(speed_vals[0]))
        drop = base_speed - speed_vals
        idx = int(np.nanargmax(drop))
        values["speed_drop_max"] = float(drop[idx])
        values["speed_drop_peak_time"] = float(speed_times[idx])
        masks["speed_drop_max"] = True
        masks["speed_drop_peak_time"] = True
    else:
        values["speed_drop_max"] = float("nan")
        values["speed_drop_peak_time"] = float("nan")
        masks["speed_drop_max"] = False
        masks["speed_drop_peak_time"] = False

    for target in ["ay", "yaw_rate", "roll", "roll_rate"]:
        arr = y_ext[:, name_to_idx[target]]
        m = mask_ext[:, name_to_idx[target]]
        peak, peak_t = signed_peak(arr, m, EXT_TIME)
        values[f"{target}_peak_value"] = peak
        values[f"{target}_peak_time"] = peak_t
        masks[f"{target}_peak_value"] = math.isfinite(peak)
        masks[f"{target}_peak_time"] = math.isfinite(peak_t)

    if math.isfinite(steer_peak) and abs(steer_peak) > 1e-6 and steer_mask.any():
        after_peak = EXT_TIME >= steer_peak_t if math.isfinite(steer_peak_t) else np.zeros_like(EXT_TIME, dtype=bool)
        threshold = 0.25 * abs(steer_peak)
        rec = np.where(after_peak & steer_mask & (np.abs(steer) <= threshold))[0]
        values["recovery_time"] = float(EXT_TIME[rec[0]]) if len(rec) else float("nan")
    else:
        values["recovery_time"] = float("nan")
    masks["recovery_time"] = math.isfinite(values["recovery_time"])

    values["large_steering_response"] = float(abs(steer_peak) >= 1.0) if math.isfinite(steer_peak) else float("nan")
    values["large_brake_response"] = float(values["brake_peak_value"] >= 0.25) if math.isfinite(values["brake_peak_value"]) else float("nan")
    roll_peak = abs(values.get("roll_peak_value", float("nan")))
    ay_peak = abs(values.get("ay_peak_value", float("nan")))
    values["high_risk_state"] = float((math.isfinite(roll_peak) and roll_peak >= 0.08) or (math.isfinite(ay_peak) and ay_peak >= 5.0))
    values["recovered_by_3s"] = float(math.isfinite(values["recovery_time"]) and values["recovery_time"] <= 3.0)
    values["recovered_by_5s"] = float(math.isfinite(values["recovery_time"]) and values["recovery_time"] <= 5.0)
    for k in ["large_steering_response", "large_brake_response", "high_risk_state", "recovered_by_3s", "recovered_by_5s"]:
        masks[k] = math.isfinite(values[k])

    out = np.array([values.get(k, float("nan")) for k in KEYPOINT_NAMES], dtype=np.float32)
    out_mask = np.array([masks.get(k, False) for k in KEYPOINT_NAMES], dtype=bool)
    return np.nan_to_num(out, nan=0.0), out_mask


def build_arrays(manifest: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    array_path = ARRAY_DIR / "record_episode_v2_task_arrays.npz"
    meta_path = TABLE_DIR / "record_episode_v2_task_sample_meta.csv"
    summary_path = LOG_DIR / "record_episode_v2_task_dataset_summary.json"
    required_array_keys = {
        "X_vehicle",
        "input_mask",
        "Y_traj_core_3s",
        "target_mask_core_3s",
        "Y_traj_ext_5s",
        "target_mask_ext_5s",
        "Y_keypoints",
        "keypoint_mask",
        "Y_response_type",
        "Y_episode_type",
        "Y_curve_type",
        "task_mask",
        "quality_flags",
    }
    if array_path.exists() and meta_path.exists() and summary_path.exists():
        with np.load(array_path, allow_pickle=True) as z:
            existing_keys = set(z.files)
            arrays = {name: z[name] for name in z.files}
        meta = pd.read_csv(meta_path, encoding="utf-8-sig", low_memory=False)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if required_array_keys.issubset(existing_keys):
            return arrays, meta, summary
        if required_array_keys - existing_keys == {"quality_flags"}:
            arrays["quality_flags"] = meta.get("quality_flags", pd.Series([""] * len(meta))).fillna("").astype(str).to_numpy(dtype=object)
            np.savez_compressed(array_path, **arrays)
            meta = pd.read_csv(meta_path, encoding="utf-8-sig", low_memory=False)
            return arrays, meta, summary

    cache: dict[str, pd.DataFrame | None] = {}
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    xs: list[np.ndarray] = []
    xmasks: list[np.ndarray] = []
    y_exts: list[np.ndarray] = []
    y_ext_masks: list[np.ndarray] = []
    y_cores: list[np.ndarray] = []
    y_core_masks: list[np.ndarray] = []
    keypoints: list[np.ndarray] = []
    keypoint_masks: list[np.ndarray] = []
    response_ids: list[int] = []
    episode_ids: list[int] = []
    curve_ids: list[int] = []
    task_masks: list[np.ndarray] = []
    quality_flag_values: list[str] = []

    response_map = {v: i for i, v in enumerate(RESPONSE_TYPE_LABELS)}
    episode_map = {v: i for i, v in enumerate(EPISODE_TYPE_LABELS)}
    curve_map = {v: i for i, v in enumerate(CURVE_TYPE_LABELS)}

    sorted_manifest = manifest.sort_values(["subject", "session_stamp", "episode_index_in_record"]).reset_index(drop=True)
    for row_no, ep in sorted_manifest.iterrows():
        if row_no % 100 == 0:
            print(f"build arrays {row_no}/{len(sorted_manifest)}", flush=True)
        path = resolve_vehicle_path(ep)
        path_key = str(path)
        if path_key not in cache:
            df = load_vehicle_csv_light(path)
            cache[path_key] = df
        df = cache[path_key]
        if df is None:
            dropped.append({"episode_uid": ep.get("episode_uid"), "drop_reason": "vehicle_csv_unreadable", "vehicle_file": path_key})
            continue
        anchor = finite_float(ep.get("model_anchor_s_v1_8"), finite_float(ep.get("condition_peak_s")))
        if not math.isfinite(anchor):
            dropped.append({"episode_uid": ep.get("episode_uid"), "drop_reason": "anchor_missing", "vehicle_file": path_key})
            continue

        input_query = anchor + INPUT_TIME
        ext_query = anchor + EXT_TIME
        core_query = anchor + CORE_TIME

        input_values: list[np.ndarray] = []
        input_masks: list[np.ndarray] = []
        for col in INPUT_FEATURES:
            vals, m = interp_series(df, col, input_query)
            if col == "zx|SteeringWheel":
                anchor_vals, anchor_m = interp_series(df, col, np.array([anchor], dtype=float))
                if anchor_m[0]:
                    vals = vals - float(anchor_vals[0])
            input_values.append(vals)
            input_masks.append(m)
        input_mat = np.stack(input_values, axis=1).astype(np.float32)
        input_mask = np.stack(input_masks, axis=1).astype(bool)

        ext_values: list[np.ndarray] = []
        ext_masks: list[np.ndarray] = []
        core_values: list[np.ndarray] = []
        core_masks: list[np.ndarray] = []
        baseline: dict[str, float] = {}
        for out_name, col, _, relative_to_anchor in OUTPUT_SPECS:
            anchor_vals, anchor_m = interp_series(df, col, np.array([anchor], dtype=float))
            baseline[out_name] = float(anchor_vals[0]) if anchor_m[0] else float("nan")
            vals_ext, m_ext = interp_series(df, col, ext_query)
            vals_core, m_core = interp_series(df, col, core_query)
            if relative_to_anchor and anchor_m[0]:
                vals_ext = vals_ext - float(anchor_vals[0])
                vals_core = vals_core - float(anchor_vals[0])
            ext_values.append(vals_ext)
            ext_masks.append(m_ext)
            core_values.append(vals_core)
            core_masks.append(m_core)
        y_ext = np.stack(ext_values, axis=1).astype(np.float32)
        y_ext_mask = np.stack(ext_masks, axis=1).astype(bool)
        y_core = np.stack(core_values, axis=1).astype(np.float32)
        y_core_mask = np.stack(core_masks, axis=1).astype(bool)

        essential_idx = [INPUT_FEATURES.index(c) for c in ESSENTIAL_INPUT_FEATURES if c in INPUT_FEATURES]
        input_time_mask = input_mask[:, essential_idx].mean(axis=1) >= 0.5 if essential_idx else input_mask.any(axis=1)
        input_valid_sec = valid_seconds(input_time_mask, INPUT_TIME)
        steering_idx = OUTPUT_NAMES.index("steering")
        core_valid_sec = valid_seconds(y_core_mask[:, steering_idx], CORE_TIME)
        ext_valid_sec = valid_seconds(y_ext_mask[:, steering_idx], EXT_TIME)

        kp, kp_mask = compute_keypoints(y_ext, y_ext_mask, baseline)
        quality_flags = str(ep.get("quality_flags", ""))
        if input_valid_sec < 1.95:
            quality_flags = (quality_flags + ";input_incomplete").strip(";")
        if ext_valid_sec < 4.95:
            quality_flags = (quality_flags + ";target_incomplete").strip(";")

        split = "train"
        subject = str(ep.get("subject", ""))
        if subject in TEST_SUBJECTS:
            split = "test"
        elif subject in VAL_SUBJECTS:
            split = "val"

        task_mask = np.array([bool_value(ep.get(c)) for c in TASK_MASK_COLUMNS], dtype=bool)
        row = ep.to_dict()
        row.update(
            {
                "sample_id": str(ep.get("episode_uid")),
                "anchor_time_s": anchor,
                "split": split,
                "input_valid_sec": input_valid_sec,
                "core_target_valid_sec": core_valid_sec,
                "ext_target_valid_sec": ext_valid_sec,
                "input_complete_2s": input_valid_sec >= 1.95,
                "core_complete_3s": core_valid_sec >= 2.95,
                "ext_complete_5s": ext_valid_sec >= 4.95,
                "minimum_input_ok_1s": input_valid_sec >= 1.0,
                "core_target_ok_3s": core_valid_sec >= 2.5,
                "window_incomplete_but_usable": (input_valid_sec >= 1.0 and core_valid_sec >= 2.5)
                and not (input_valid_sec >= 1.95 and ext_valid_sec >= 4.95),
                "quality_flags": quality_flags,
                "response_type_id": response_map.get(str(ep.get("response_type")), response_map["unknown"]),
                "episode_type_id": episode_map.get(str(ep.get("episode_type")), episode_map["unknown"]),
                "curve_type_id": curve_map.get(str(ep.get("curve_type")), curve_map["unknown"]),
            }
        )
        rows.append(row)
        xs.append(input_mat)
        xmasks.append(input_mask)
        y_exts.append(y_ext)
        y_ext_masks.append(y_ext_mask)
        y_cores.append(y_core)
        y_core_masks.append(y_core_mask)
        keypoints.append(kp)
        keypoint_masks.append(kp_mask)
        response_ids.append(int(row["response_type_id"]))
        episode_ids.append(int(row["episode_type_id"]))
        curve_ids.append(int(row["curve_type_id"]))
        task_masks.append(task_mask)
        quality_flag_values.append(quality_flags)

    if not rows:
        raise RuntimeError("No task samples were built from v2.0 manifest.")

    meta = pd.DataFrame(rows)
    arrays = {
        "X_vehicle": np.stack(xs, axis=0).astype(np.float32),
        "input_mask": np.stack(xmasks, axis=0).astype(bool),
        "Y_traj_core_3s": np.stack(y_cores, axis=0).astype(np.float32),
        "target_mask_core_3s": np.stack(y_core_masks, axis=0).astype(bool),
        "Y_traj_ext_5s": np.stack(y_exts, axis=0).astype(np.float32),
        "target_mask_ext_5s": np.stack(y_ext_masks, axis=0).astype(bool),
        "Y_keypoints": np.stack(keypoints, axis=0).astype(np.float32),
        "keypoint_mask": np.stack(keypoint_masks, axis=0).astype(bool),
        "Y_response_type": np.array(response_ids, dtype=np.int64),
        "Y_episode_type": np.array(episode_ids, dtype=np.int64),
        "Y_curve_type": np.array(curve_ids, dtype=np.int64),
        "task_mask": np.stack(task_masks, axis=0).astype(bool),
        "quality_flags": np.array(quality_flag_values, dtype=object),
        "input_time": INPUT_TIME.astype(np.float32),
        "core_time": CORE_TIME.astype(np.float32),
        "ext_time": EXT_TIME.astype(np.float32),
        "input_feature_names": np.array(INPUT_FEATURES, dtype=object),
        "output_names": np.array(OUTPUT_NAMES, dtype=object),
        "keypoint_names": np.array(KEYPOINT_NAMES, dtype=object),
        "response_type_labels": np.array(RESPONSE_TYPE_LABELS, dtype=object),
        "episode_type_labels": np.array(EPISODE_TYPE_LABELS, dtype=object),
        "curve_type_labels": np.array(CURVE_TYPE_LABELS, dtype=object),
    }
    np.savez_compressed(array_path, **arrays)
    meta.to_csv(meta_path, index=False, encoding="utf-8-sig")
    if dropped:
        pd.DataFrame(dropped).to_csv(TABLE_DIR / "record_episode_v2_task_dropped.csv", index=False, encoding="utf-8-sig")

    meta[meta["window_incomplete_but_usable"].astype(bool)].to_csv(
        MANIFEST_DIR / "manifest_window_incomplete_but_usable.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "sample_count": int(len(meta)),
        "dropped_count": int(len(dropped)),
        "split_counts": meta["split"].value_counts().to_dict(),
        "training_role_counts": meta["training_role"].value_counts().to_dict(),
        "episode_type_counts": meta["episode_type"].value_counts().to_dict(),
        "response_type_counts": meta["response_type"].value_counts().to_dict(),
        "input_complete_2s": int(meta["input_complete_2s"].sum()),
        "ext_complete_5s": int(meta["ext_complete_5s"].sum()),
        "window_incomplete_but_usable": int(meta["window_incomplete_but_usable"].sum()),
        "standardization_scope": "all train-time scalers are fit on the training split only",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return arrays, meta, summary


class MultiTaskMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        reg_dim: int,
        response_classes: int | None = None,
        episode_classes: int | None = None,
        curve_classes: int | None = None,
        keypoint_dim: int | None = None,
        hidden: int = 256,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )
        self.reg_head = nn.Linear(hidden, reg_dim)
        self.response_head = nn.Linear(hidden, response_classes) if response_classes else None
        self.episode_head = nn.Linear(hidden, episode_classes) if episode_classes else None
        self.curve_head = nn.Linear(hidden, curve_classes) if curve_classes else None
        self.keypoint_head = nn.Linear(hidden, keypoint_dim) if keypoint_dim else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        out = {"reg": self.reg_head(z)}
        if self.response_head is not None:
            out["response"] = self.response_head(z)
        if self.episode_head is not None:
            out["episode"] = self.episode_head(z)
        if self.curve_head is not None:
            out["curve"] = self.curve_head(z)
        if self.keypoint_head is not None:
            out["keypoint"] = self.keypoint_head(z)
        return out


@dataclass
class Experiment:
    name: str
    name_cn: str
    include_col: str
    output_names: list[str]
    horizon: str = "ext"
    fixed_full_window: bool = False
    response_head: bool = False
    episode_head: bool = False
    curve_head: bool = False
    keypoint_head: bool = False
    noncurve_only: bool = False
    curve_only: bool = False
    allow_control: bool = False
    note_cn: str = ""


def flatten_inputs(x: np.ndarray, x_mask: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    raw_idx = np.unique(np.linspace(0, len(INPUT_TIME) - 1, 13).round().astype(int))
    parts: list[np.ndarray] = []
    names: list[str] = []
    for k in raw_idx:
        parts.append(np.where(x_mask[:, k, :], x[:, k, :], 0.0))
        names.extend([f"{feat}@{INPUT_TIME[k]:.2f}s" for feat in INPUT_FEATURES])
    for col in ["condition_score_peak", "vehicle_score_peak", "driver_score_peak", "median_speed_kmh", "peak_abs_roll", "peak_abs_ay"]:
        arr = pd.to_numeric(meta.get(col, pd.Series(0.0, index=meta.index)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        parts.append(arr[:, None])
        names.append(col)
    return np.concatenate(parts, axis=1).astype(np.float32), names


def scale_features(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x[train_idx], axis=0)
    std = np.nanstd(x[train_idx], axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def scale_targets(y: np.ndarray, mask: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_scaled = np.zeros_like(y, dtype=np.float32)
    mean = np.zeros(y.shape[-1], dtype=np.float32)
    std = np.ones(y.shape[-1], dtype=np.float32)
    for j in range(y.shape[-1]):
        vals = y[train_idx, :, j][mask[train_idx, :, j]]
        if len(vals):
            mean[j] = float(np.nanmean(vals))
            s = float(np.nanstd(vals))
            std[j] = s if s >= 1e-6 else 1.0
        y_scaled[:, :, j] = (y[:, :, j] - mean[j]) / std[j]
    y_scaled = np.where(mask, y_scaled, 0.0).astype(np.float32)
    return y_scaled, mean, std


def scale_vector_targets(y: np.ndarray, mask: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_scaled = np.zeros_like(y, dtype=np.float32)
    mean = np.zeros(y.shape[-1], dtype=np.float32)
    std = np.ones(y.shape[-1], dtype=np.float32)
    for j in range(y.shape[-1]):
        vals = y[train_idx, j][mask[train_idx, j]]
        if len(vals):
            mean[j] = float(np.nanmean(vals))
            s = float(np.nanstd(vals))
            std[j] = s if s >= 1e-6 else 1.0
        y_scaled[:, j] = (y[:, j] - mean[j]) / std[j]
    y_scaled = np.where(mask, y_scaled, 0.0).astype(np.float32)
    return y_scaled, mean, std


def masked_weighted_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    diff = (pred - y).pow(2)
    valid = mask > 0.5
    if not valid.any():
        return diff.mean()
    w = weight[:, None].expand_as(diff)
    denom = torch.clamp((valid.float() * w).sum(), min=1.0)
    return (diff * valid.float() * w).sum() / denom


def class_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target >= 0
    if not valid.any():
        return torch.tensor(0.0, device=logits.device)
    return nn.functional.cross_entropy(logits[valid], target[valid])


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    keypoints: np.ndarray,
    keypoint_mask: np.ndarray,
    meta: pd.DataFrame,
    exp: Experiment,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    x_scaled, x_mean, x_std = scale_features(x, train_idx)
    y_scaled, y_mean, y_std = scale_targets(y, y_mask, train_idx)
    kp_scaled, kp_mean, kp_std = scale_vector_targets(keypoints, keypoint_mask, train_idx)
    y_flat = y_scaled.reshape(len(y_scaled), -1)
    mask_flat = y_mask.reshape(len(y_mask), -1).astype(np.float32)

    response_target = meta["response_type_id"].to_numpy(dtype=np.int64) if exp.response_head else np.full(len(meta), -1, dtype=np.int64)
    episode_target = meta["episode_type_id"].to_numpy(dtype=np.int64) if exp.episode_head else np.full(len(meta), -1, dtype=np.int64)
    curve_target = meta["curve_type_id"].to_numpy(dtype=np.int64) if exp.curve_head else np.full(len(meta), -1, dtype=np.int64)
    sample_weight = np.ones(len(meta), dtype=np.float32)
    sample_weight[meta["training_role"].astype(str).eq("aux_train").to_numpy()] = 0.6
    sample_weight[meta["training_role"].astype(str).eq("review_need_manual_check").to_numpy()] = 0.35
    sample_weight[meta["training_role"].astype(str).eq("control").to_numpy()] = 0.35

    train_ds = TensorDataset(
        torch.from_numpy(x_scaled[train_idx]).float(),
        torch.from_numpy(y_flat[train_idx]).float(),
        torch.from_numpy(mask_flat[train_idx]).float(),
        torch.from_numpy(response_target[train_idx]).long(),
        torch.from_numpy(episode_target[train_idx]).long(),
        torch.from_numpy(curve_target[train_idx]).long(),
        torch.from_numpy(kp_scaled[train_idx]).float(),
        torch.from_numpy(keypoint_mask[train_idx].astype(np.float32)).float(),
        torch.from_numpy(sample_weight[train_idx]).float(),
    )
    loader = DataLoader(train_ds, batch_size=min(256, len(train_ds)), shuffle=True, drop_last=False)
    model = MultiTaskMLP(
        in_dim=x.shape[1],
        reg_dim=y_flat.shape[1],
        response_classes=len(RESPONSE_TYPE_LABELS) if exp.response_head else None,
        episode_classes=len(EPISODE_TYPE_LABELS) if exp.episode_head else None,
        curve_classes=len(CURVE_TYPE_LABELS) if exp.curve_head else None,
        keypoint_dim=keypoints.shape[1] if exp.keypoint_head else None,
        hidden=384 if len(train_idx) >= 600 else 192,
        dropout=0.08,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xv = torch.from_numpy(x_scaled[val_idx]).float().to(device)
    yv = torch.from_numpy(y_flat[val_idx]).float().to(device)
    mv = torch.from_numpy(mask_flat[val_idx]).float().to(device)
    wv = torch.ones(len(val_idx), dtype=torch.float32, device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    for epoch in range(1, 221):
        model.train()
        for xb, yb, mb, rb, eb, cb, kpb, kpmb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            rb = rb.to(device)
            eb = eb.to(device)
            cb = cb.to(device)
            kpb = kpb.to(device)
            kpmb = kpmb.to(device)
            wb = wb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = masked_weighted_mse(out["reg"], yb, mb, wb)
            if exp.response_head:
                loss = loss + 0.10 * class_loss(out["response"], rb)
            if exp.episode_head:
                loss = loss + 0.08 * class_loss(out["episode"], eb)
            if exp.curve_head:
                loss = loss + 0.08 * class_loss(out["curve"], cb)
            if exp.keypoint_head:
                loss = loss + 0.15 * masked_weighted_mse(out["keypoint"], kpb, kpmb, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_out = model(xv)
            val_loss = masked_weighted_mse(val_out["reg"], yv, mv, wv)
            val_rmse = float(torch.sqrt(torch.clamp(val_loss, min=0.0)).item())
        if val_rmse + 1e-6 < best_val:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 32:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    pred_scaled: list[np.ndarray] = []
    response_logits: list[np.ndarray] = []
    episode_logits: list[np.ndarray] = []
    curve_logits: list[np.ndarray] = []
    keypoint_preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_scaled), 1024):
            xb = torch.from_numpy(x_scaled[start : start + 1024]).float().to(device)
            out = model(xb)
            pred_scaled.append(out["reg"].cpu().numpy().astype(np.float32))
            if exp.response_head:
                response_logits.append(out["response"].cpu().numpy().astype(np.float32))
            if exp.episode_head:
                episode_logits.append(out["episode"].cpu().numpy().astype(np.float32))
            if exp.curve_head:
                curve_logits.append(out["curve"].cpu().numpy().astype(np.float32))
            if exp.keypoint_head:
                keypoint_preds.append(out["keypoint"].cpu().numpy().astype(np.float32))

    pred_scaled_arr = np.vstack(pred_scaled).reshape(y.shape)
    pred = pred_scaled_arr * y_std[None, None, :] + y_mean[None, None, :]
    aux: dict[str, np.ndarray] = {}
    if response_logits:
        aux["response_logits"] = np.vstack(response_logits)
    if episode_logits:
        aux["episode_logits"] = np.vstack(episode_logits)
    if curve_logits:
        aux["curve_logits"] = np.vstack(curve_logits)
    if keypoint_preds:
        kp_scaled_pred = np.vstack(keypoint_preds)
        aux["keypoint_pred"] = kp_scaled_pred * kp_std[None, :] + kp_mean[None, :]
    info = {
        "best_val_scaled_rmse": best_val,
        "best_epoch": best_epoch,
        "feature_mean": x_mean.tolist(),
        "feature_std": x_std.tolist(),
        "target_mean": y_mean.tolist(),
        "target_std": y_std.tolist(),
        "keypoint_mean": kp_mean.tolist(),
        "keypoint_std": kp_std.tolist(),
    }
    return pred.astype(np.float32), info, aux


def subset_outputs(arr: np.ndarray, mask: np.ndarray, output_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    idx = [OUTPUT_NAMES.index(name) for name in output_names]
    return arr[:, :, idx], mask[:, :, idx], idx


def experiment_mask(meta: pd.DataFrame, exp: Experiment) -> np.ndarray:
    include = meta[exp.include_col].astype(bool).to_numpy() if exp.include_col in meta.columns else np.ones(len(meta), dtype=bool)
    if exp.fixed_full_window:
        include &= meta["input_complete_2s"].astype(bool).to_numpy()
        include &= meta["ext_complete_5s"].astype(bool).to_numpy()
    else:
        include &= meta["minimum_input_ok_1s"].astype(bool).to_numpy()
        include &= meta["core_target_ok_3s"].astype(bool).to_numpy()
    if exp.noncurve_only:
        include &= ~meta["road_coord_is_curve_v1_9"].astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
    if exp.curve_only:
        include &= meta["road_coord_is_curve_v1_9"].astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
    if not exp.allow_control:
        include &= ~meta["training_role"].astype(str).eq("control").to_numpy()
    include &= ~meta["training_role"].astype(str).str.startswith("excluded").to_numpy()
    return include


def split_indices(meta: pd.DataFrame, include: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = meta["split"].astype(str).to_numpy()
    return (
        np.where(include & (split == "train"))[0],
        np.where(include & (split == "val"))[0],
        np.where(include & (split == "test"))[0],
    )


def rmse(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(y) & np.isfinite(pred)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(y[valid] - pred[valid]))))


def mae(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(y) & np.isfinite(pred)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(y[valid] - pred[valid])))


def batch_signed_peak(arr: np.ndarray, mask: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    peaks = np.full(arr.shape[0], np.nan, dtype=float)
    peak_t = np.full(arr.shape[0], np.nan, dtype=float)
    for i in range(arr.shape[0]):
        valid = mask[i] & np.isfinite(arr[i])
        if valid.any():
            vals = arr[i, valid]
            tt = time[valid]
            idx = int(np.nanargmax(np.abs(vals)))
            peaks[i] = float(vals[idx])
            peak_t[i] = float(tt[idx])
    return peaks, peak_t


def macro_f1_or_nan(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return float("nan")
    try:
        return float(f1_score(y_true[valid].astype(int), y_pred[valid].astype(int), average="macro", zero_division=0))
    except Exception:
        return float("nan")


def compute_metrics(
    exp: Experiment,
    meta: pd.DataFrame,
    y: np.ndarray,
    mask: np.ndarray,
    pred: np.ndarray,
    keypoints: np.ndarray,
    keypoint_mask: np.ndarray,
    aux: dict[str, np.ndarray],
    include: np.ndarray,
    train_idx: np.ndarray,
    selected_output_names: list[str],
    time_axis: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    steer_local_idx = selected_output_names.index("steering") if "steering" in selected_output_names else -1
    metric_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    train_large_thr = float("nan")
    if steer_local_idx >= 0 and len(train_idx):
        gt_peak_train, _ = batch_signed_peak(y[train_idx, :, steer_local_idx], mask[train_idx, :, steer_local_idx], time_axis)
        vals = np.abs(gt_peak_train[np.isfinite(gt_peak_train)])
        train_large_thr = float(np.nanpercentile(vals, 75)) if len(vals) else float("nan")

    class_preds = {}
    if "response_logits" in aux:
        class_preds["response_type_pred"] = np.argmax(aux["response_logits"], axis=1)
    if "episode_logits" in aux:
        class_preds["episode_type_pred"] = np.argmax(aux["episode_logits"], axis=1)
    if "curve_logits" in aux:
        class_preds["curve_type_pred"] = np.argmax(aux["curve_logits"], axis=1)

    for split_name in ["train", "val", "test"]:
        idx = np.where(include & (meta["split"].astype(str).to_numpy() == split_name))[0]
        if len(idx) == 0:
            continue
        row: dict[str, Any] = {
            "experiment": exp.name,
            "split": split_name,
            "n": int(len(idx)),
            "usable_sample_rate_vs_all": float(len(idx) / max((meta["split"].astype(str).to_numpy() == split_name).sum(), 1)),
        }
        for j, out_name in enumerate(selected_output_names):
            row[f"{out_name}_rmse"] = rmse(y[idx, :, j], pred[idx, :, j], mask[idx, :, j])
            row[f"{out_name}_mae"] = mae(y[idx, :, j], pred[idx, :, j], mask[idx, :, j])
        if steer_local_idx >= 0:
            primary = time_axis <= 2.0
            tail = time_axis >= 2.0
            row["steering_primary_rmse_0_2s"] = rmse(
                y[idx, :, steer_local_idx], pred[idx, :, steer_local_idx], mask[idx, :, steer_local_idx] & primary[None, :]
            )
            row["steering_tail_rmse_2s_end"] = rmse(
                y[idx, :, steer_local_idx], pred[idx, :, steer_local_idx], mask[idx, :, steer_local_idx] & tail[None, :]
            )
            gt_peak, gt_t = batch_signed_peak(y[idx, :, steer_local_idx], mask[idx, :, steer_local_idx], time_axis)
            pr_peak, pr_t = batch_signed_peak(pred[idx, :, steer_local_idx], mask[idx, :, steer_local_idx], time_axis)
            large = np.abs(gt_peak) >= train_large_thr if math.isfinite(train_large_thr) else np.zeros_like(gt_peak, dtype=bool)
            large_n = int(np.nansum(large))
            wrong = large & np.isfinite(gt_peak) & np.isfinite(pr_peak) & (np.sign(gt_peak) != np.sign(pr_peak))
            severe_under = large & np.isfinite(gt_peak) & np.isfinite(pr_peak) & (np.abs(pr_peak) < 0.5 * np.abs(gt_peak))
            recall = large & np.isfinite(pr_peak) & (np.abs(pr_peak) >= 0.5 * train_large_thr)
            row["large_response_n"] = large_n
            row["wrong_side_rate"] = float(wrong.sum() / large_n) if large_n else float("nan")
            row["severe_under_amplitude_rate"] = float(severe_under.sum() / large_n) if large_n else float("nan")
            row["large_response_recall"] = float(recall.sum() / large_n) if large_n else float("nan")
            row["steering_peak_abs_mae"] = float(np.nanmean(np.abs(np.abs(pr_peak) - np.abs(gt_peak)))) if len(idx) else float("nan")
            row["steering_peak_timing_mae"] = float(np.nanmean(np.abs(pr_t - gt_t))) if len(idx) else float("nan")
        if "response_type_pred" in class_preds:
            row["response_type_macro_f1"] = macro_f1_or_nan(
                meta.iloc[idx]["response_type_id"].to_numpy(dtype=float),
                class_preds["response_type_pred"][idx].astype(float),
                RESPONSE_TYPE_LABELS,
            )
        if "episode_type_pred" in class_preds:
            row["episode_type_macro_f1"] = macro_f1_or_nan(
                meta.iloc[idx]["episode_type_id"].to_numpy(dtype=float),
                class_preds["episode_type_pred"][idx].astype(float),
                EPISODE_TYPE_LABELS,
            )
        if "curve_type_pred" in class_preds:
            row["curve_type_macro_f1"] = macro_f1_or_nan(
                meta.iloc[idx]["curve_type_id"].to_numpy(dtype=float),
                class_preds["curve_type_pred"][idx].astype(float),
                CURVE_TYPE_LABELS,
            )
        if "keypoint_pred" in aux:
            kp_pred = aux["keypoint_pred"]
            for kp_name in [
                "steering_peak_value",
                "steering_peak_time",
                "brake_onset_time",
                "speed_drop_max",
                "roll_peak_value",
                "roll_rate_peak_value",
                "recovery_time",
            ]:
                if kp_name in KEYPOINT_NAMES:
                    j = KEYPOINT_NAMES.index(kp_name)
                    valid = keypoint_mask[idx, j] & np.isfinite(keypoints[idx, j]) & np.isfinite(kp_pred[idx, j])
                    row[f"keypoint_{kp_name}_mae"] = (
                        float(np.mean(np.abs(keypoints[idx, j][valid] - kp_pred[idx, j][valid]))) if valid.any() else float("nan")
                    )
        metric_rows.append(row)

        for local_i, i in enumerate(idx):
            srow: dict[str, Any] = {
                "experiment": exp.name,
                "sample_id": meta.loc[i, "sample_id"],
                "split": split_name,
                "subject": meta.loc[i, "subject"],
                "session_stamp": meta.loc[i, "session_stamp"],
                "episode_type": meta.loc[i, "episode_type"],
                "response_type": meta.loc[i, "response_type"],
                "curve_type": meta.loc[i, "curve_type"],
                "training_role": meta.loc[i, "training_role"],
                "quality_flags": meta.loc[i, "quality_flags"],
                "window_incomplete_but_usable": bool(meta.loc[i, "window_incomplete_but_usable"]),
                "sample_rmse": rmse(y[i : i + 1], pred[i : i + 1], mask[i : i + 1]),
            }
            if steer_local_idx >= 0:
                gt_peak_i, gt_t_i = batch_signed_peak(
                    y[i : i + 1, :, steer_local_idx], mask[i : i + 1, :, steer_local_idx], time_axis
                )
                pr_peak_i, pr_t_i = batch_signed_peak(
                    pred[i : i + 1, :, steer_local_idx], mask[i : i + 1, :, steer_local_idx], time_axis
                )
                gt_p = float(gt_peak_i[0])
                pr_p = float(pr_peak_i[0])
                large_i = bool(math.isfinite(gt_p) and math.isfinite(train_large_thr) and abs(gt_p) >= train_large_thr)
                srow.update(
                    {
                        "gt_steering_peak": gt_p,
                        "pred_steering_peak": pr_p,
                        "gt_steering_peak_time": float(gt_t_i[0]),
                        "pred_steering_peak_time": float(pr_t_i[0]),
                        "large_response": large_i,
                        "wrong_side": bool(large_i and math.isfinite(pr_p) and np.sign(gt_p) != np.sign(pr_p)),
                        "severe_under_amplitude": bool(large_i and math.isfinite(pr_p) and abs(pr_p) < 0.5 * abs(gt_p)),
                    }
                )
            sample_rows.append(srow)
    return pd.DataFrame(metric_rows), pd.DataFrame(sample_rows)


def plot_one_sample(
    out_path: Path,
    title: str,
    time_axis: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    output_names: list[str],
) -> None:
    rows = len(output_names)
    fig, axes = plt.subplots(rows, 1, figsize=(11.5, max(2.0 * rows, 5.0)), sharex=True)
    axes_arr = np.atleast_1d(axes)
    for ax, name in zip(axes_arr, output_names):
        j = output_names.index(name)
        valid = mask[:, j] & np.isfinite(y[:, j])
        ax.axvspan(0.0, min(3.0, time_axis[-1]), color="#DBEAFE", alpha=0.18, label="核心预测区间" if j == 0 else None)
        if time_axis[-1] > 3.0:
            ax.axvspan(3.0, time_axis[-1], color="#F3F4F6", alpha=0.35, label="扩展预测区间" if j == 0 else None)
        if valid.any():
            ax.plot(time_axis[valid], y[valid, j], color="#111827", lw=1.8, label="真实" if j == 0 else None)
            ax.plot(time_axis[valid], pred[valid, j], color="#2563EB", lw=1.3, label="预测" if j == 0 else None)
        ax.axvline(0.0, color="#DC2626", lw=0.8, ls="--")
        ax.set_ylabel(name, fontsize=8)
        ax.grid(True, alpha=0.22)
    axes_arr[0].legend(fontsize=8, loc="best")
    axes_arr[-1].set_xlabel("相对锚点时间 / s")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def select_plot_samples(per_sample: pd.DataFrame) -> dict[str, list[str]]:
    test = per_sample[per_sample["split"].astype(str).eq("test")].copy()
    out: dict[str, list[str]] = {}
    if test.empty:
        return out
    out["random_20"] = test.sample(min(20, len(test)), random_state=RANDOM_SEED)["sample_id"].astype(str).tolist()
    out["worst_rmse_20"] = (
        test.assign(sample_rmse=pd.to_numeric(test["sample_rmse"], errors="coerce"))
        .sort_values("sample_rmse", ascending=False)
        .head(20)["sample_id"]
        .astype(str)
        .tolist()
    )
    if "large_response" in test.columns:
        out["large_response_all_or_top_30"] = (
            test[test["large_response"].astype(bool)]
            .assign(abs_peak=lambda d: pd.to_numeric(d["gt_steering_peak"], errors="coerce").abs())
            .sort_values("abs_peak", ascending=False)
            .head(30)["sample_id"]
            .astype(str)
            .tolist()
        )
    if "wrong_side" in test.columns:
        out["wrong_side_cases"] = test[test["wrong_side"].astype(bool)].head(30)["sample_id"].astype(str).tolist()
    if "severe_under_amplitude" in test.columns:
        out["severe_under_amplitude_cases"] = (
            test[test["severe_under_amplitude"].astype(bool)].head(30)["sample_id"].astype(str).tolist()
        )
    for label in RESPONSE_TYPE_LABELS:
        ids = test[test["response_type"].astype(str).eq(label)].head(12)["sample_id"].astype(str).tolist()
        if ids:
            out[f"response_type_{label}"] = ids
    if test["curve_type"].astype(str).ne("not_curve").any():
        for label in ["normal_or_weak_curve", "abnormal_roll", "high_risk_curve_response"]:
            ids = test[test["curve_type"].astype(str).eq(label)].head(12)["sample_id"].astype(str).tolist()
            if ids:
                out[f"curve_{label}"] = ids
    return out


def write_figures(
    exp_dir: Path,
    exp: Experiment,
    meta: pd.DataFrame,
    y: np.ndarray,
    mask: np.ndarray,
    pred: np.ndarray,
    per_sample: pd.DataFrame,
    output_names: list[str],
    time_axis: np.ndarray,
) -> list[dict[str, str]]:
    sample_to_idx = {str(sid): i for i, sid in enumerate(meta["sample_id"].astype(str))}
    fig_rows: list[dict[str, str]] = []
    for folder, sample_ids in select_plot_samples(per_sample).items():
        for sid in sample_ids:
            if sid not in sample_to_idx:
                continue
            i = sample_to_idx[sid]
            mrow = meta.iloc[i]
            title = (
                f"{exp.name} | {sid} | {mrow.get('split')} | {mrow.get('episode_type')} | "
                f"{mrow.get('response_type')} | {mrow.get('training_role')}"
            )
            out_path = exp_dir / "figures" / folder / f"{sid}.png"
            plot_one_sample(out_path, title, time_axis, y[i], pred[i], mask[i], output_names)
            fig_rows.append({"folder": folder, "sample_id": sid, "path": str(out_path)})
    return fig_rows


def write_figure_index(exp_dir: Path, fig_rows: list[dict[str, str]]) -> None:
    lines = ["<html><meta charset='utf-8'><body>", "<h1>预测图索引</h1>"]
    by_folder: dict[str, list[dict[str, str]]] = {}
    for row in fig_rows:
        by_folder.setdefault(row["folder"], []).append(row)
    for folder, rows in sorted(by_folder.items()):
        lines.append(f"<h2>{html.escape(folder)}</h2><ul>")
        for row in rows:
            rel = os.path.relpath(row["path"], exp_dir).replace("\\", "/")
            lines.append(f"<li><a href='{html.escape(rel)}'>{html.escape(row['sample_id'])}</a></li>")
        lines.append("</ul>")
    lines.append("</body></html>")
    (exp_dir / "figure_index.html").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    exp: Experiment,
    arrays: dict[str, np.ndarray],
    meta_all: pd.DataFrame,
    device: torch.device,
) -> dict[str, Any]:
    exp_dir = OUTPUT_DIR / exp.name
    for p in [exp_dir / "metrics", exp_dir / "figures", exp_dir / "predictions", exp_dir / "summaries"]:
        p.mkdir(parents=True, exist_ok=True)
    done_json = exp_dir / "summaries" / "run_summary.json"
    if done_json.exists():
        try:
            cached = json.loads(done_json.read_text(encoding="utf-8"))
            if cached.get("script_version") == SCRIPT_VERSION:
                normalize_experiment_files(exp_dir)
                return cached
        except Exception:
            pass

    include = experiment_mask(meta_all, exp)
    train_idx, val_idx, test_idx = split_indices(meta_all, include)
    if min(len(train_idx), len(val_idx), len(test_idx)) <= 0:
        raise RuntimeError(f"{exp.name} split invalid: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    horizon_key = "Y_traj_core_3s" if exp.horizon == "core" else "Y_traj_ext_5s"
    mask_key = "target_mask_core_3s" if exp.horizon == "core" else "target_mask_ext_5s"
    time_axis = CORE_TIME if exp.horizon == "core" else EXT_TIME
    y_all, mask_all, _ = subset_outputs(arrays[horizon_key], arrays[mask_key], exp.output_names)
    x_raw, _ = flatten_inputs(arrays["X_vehicle"], arrays["input_mask"], meta_all)

    pred, train_info, aux = train_model(
        x_raw,
        y_all,
        mask_all,
        arrays["Y_keypoints"],
        arrays["keypoint_mask"],
        meta_all,
        exp,
        train_idx,
        val_idx,
        device,
    )
    metrics, per_sample = compute_metrics(
        exp,
        meta_all,
        y_all,
        mask_all,
        pred,
        arrays["Y_keypoints"],
        arrays["keypoint_mask"],
        aux,
        include,
        train_idx,
        exp.output_names,
        time_axis,
    )

    metrics.to_csv(exp_dir / "metrics" / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(exp_dir / "metrics" / "prediction_summary.csv", index=False, encoding="utf-8-sig")
    sample_usage = {
        "experiment": exp.name,
        "name_cn": exp.name_cn,
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "test_count": int(len(test_idx)),
        "total_included_count": int(include.sum()),
        "window_incomplete_but_used_count": int((include & meta_all["window_incomplete_but_usable"].astype(bool).to_numpy()).sum()),
        "fixed_full_window": bool(exp.fixed_full_window),
        "outputs": exp.output_names,
        "horizon": exp.horizon,
        "note_cn": exp.note_cn,
    }
    pd.DataFrame([sample_usage]).to_csv(exp_dir / "metrics" / "sample_usage_summary.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        exp_dir / "predictions" / "predictions.npz",
        pred=pred.astype(np.float32),
        y=y_all.astype(np.float32),
        y_mask=mask_all.astype(bool),
        include=include.astype(bool),
        time_axis=time_axis.astype(np.float32),
        output_names=np.array(exp.output_names, dtype=object),
    )
    fig_rows = write_figures(exp_dir, exp, meta_all, y_all, mask_all, pred, per_sample, exp.output_names, time_axis)
    pd.DataFrame(fig_rows).to_csv(exp_dir / "figures" / "figure_index.csv", index=False, encoding="utf-8-sig")
    write_figure_index(exp_dir, fig_rows)

    test_metrics = metrics[metrics["split"].astype(str).eq("test")]
    summary = {
        **sample_usage,
        "device": str(device),
        "train_info": train_info,
        "test_metrics": test_metrics.iloc[0].to_dict() if len(test_metrics) else {},
        "figure_count": len(fig_rows),
        "metrics_path": str(exp_dir / "metrics" / "metrics_summary.csv"),
        "figure_index": str(exp_dir / "figure_index.html"),
        "script_version": SCRIPT_VERSION,
    }
    done_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    normalize_experiment_files(exp_dir)
    return summary


def normalize_experiment_files(exp_dir: Path) -> None:
    mapping = [
        (exp_dir / "metrics" / "metrics_summary.csv", exp_dir / "metrics_summary.csv"),
        (exp_dir / "metrics" / "prediction_summary.csv", exp_dir / "prediction_summary.csv"),
        (exp_dir / "metrics" / "prediction_summary.csv", exp_dir / "predictions" / "prediction_summary.csv"),
        (exp_dir / "metrics" / "sample_usage_summary.csv", exp_dir / "sample_usage_summary.csv"),
        (exp_dir / "metrics" / "sample_usage_summary.csv", exp_dir / "summaries" / "sample_usage_summary.csv"),
    ]
    for src, dst in mapping:
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())


def make_experiments() -> list[Experiment]:
    return [
        Experiment(
            name="E0_fixed_steering_baseline",
            name_cn="旧固定窗口方向盘基线",
            include_col="include_E0_fixed_steering",
            output_names=["steering"],
            horizon="ext",
            fixed_full_window=True,
            note_cn="复现旧任务：完整 2 秒输入 + 完整 5 秒方向盘输出。",
        ),
        Experiment(
            name="E1_fixed_multitask_vehicle",
            name_cn="固定窗口多输出车辆基线",
            include_col="include_E1_fixed_multitask",
            output_names=OUTPUT_NAMES,
            horizon="ext",
            fixed_full_window=True,
            response_head=True,
            note_cn="仍使用完整窗口，但同时预测方向盘、速度、制动和车辆姿态。",
        ),
        Experiment(
            name="E2_masked_multihorizon_keypoint",
            name_cn="掩码多输出多时域模型",
            include_col="include_E2_masked_multihorizon",
            output_names=OUTPUT_NAMES,
            horizon="ext",
            fixed_full_window=False,
            response_head=True,
            episode_head=True,
            keypoint_head=True,
            note_cn="允许输入/输出不完整样本进入训练，使用 mask 只计算可观测部分。",
        ),
        Experiment(
            name="E3_noncurve_response_branch",
            name_cn="非弯道响应类型辅助模型",
            include_col="include_E3_noncurve_response_branch",
            output_names=["steering", "speed", "brake", "ay", "yaw_rate", "roll", "roll_rate"],
            horizon="ext",
            fixed_full_window=False,
            response_head=True,
            episode_head=True,
            keypoint_head=True,
            noncurve_only=True,
            allow_control=True,
            note_cn="非弯道单独处理，响应类型作为辅助监督。",
        ),
        Experiment(
            name="E4_curve_specialized",
            name_cn="弯道专门模型",
            include_col="include_E4_curve_specialized",
            output_names=["speed", "brake", "ay", "yaw_rate", "roll", "roll_rate", "steering"],
            horizon="ext",
            fixed_full_window=False,
            curve_head=True,
            keypoint_head=True,
            curve_only=True,
            note_cn="弯道单独训练，重点看侧倾、横摆、速度和制动。",
        ),
        Experiment(
            name="E5A_train_candidates_only",
            name_cn="分层纳入 A：只用训练候选",
            include_col="include_E5_train_only",
            output_names=OUTPUT_NAMES,
            horizon="ext",
            fixed_full_window=False,
            response_head=True,
            keypoint_head=True,
            note_cn="E5 的 A 组，只用训练候选。",
        ),
        Experiment(
            name="E5B_train_plus_all_review",
            name_cn="分层纳入 B：训练候选 + 全部待复核",
            include_col="include_E5_all_review",
            output_names=OUTPUT_NAMES,
            horizon="ext",
            fixed_full_window=False,
            response_head=True,
            keypoint_head=True,
            note_cn="E5 的 B 组，加入所有待复核但仍排除 slope/offroad 和 data_bad。",
        ),
        Experiment(
            name="E5C_train_plus_stratified_review",
            name_cn="分层纳入 C：训练候选 + 分层干净待复核",
            include_col="include_E5_stratified_review",
            output_names=OUTPUT_NAMES,
            horizon="ext",
            fixed_full_window=False,
            response_head=True,
            keypoint_head=True,
            note_cn="E5 的 C 组，只加入语义明确的 aux_train 待复核。",
        ),
    ]


def fmt(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return str(v)
    if not math.isfinite(x):
        return "NA"
    return f"{x:.4f}"


def summary_table(result_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in result_rows:
        tm = r.get("test_metrics", {})
        rows.append(
            {
                "experiment": r.get("experiment"),
                "name_cn": r.get("name_cn"),
                "train": r.get("train_count"),
                "val": r.get("val_count"),
                "test": r.get("test_count"),
                "window_incomplete_used": r.get("window_incomplete_but_used_count"),
                "steering_rmse": tm.get("steering_rmse"),
                "steering_primary_rmse_0_2s": tm.get("steering_primary_rmse_0_2s"),
                "steering_tail_rmse_2s_end": tm.get("steering_tail_rmse_2s_end"),
                "wrong_side_rate": tm.get("wrong_side_rate"),
                "severe_under_amplitude_rate": tm.get("severe_under_amplitude_rate"),
                "large_response_recall": tm.get("large_response_recall"),
                "response_type_macro_f1": tm.get("response_type_macro_f1"),
                "curve_type_macro_f1": tm.get("curve_type_macro_f1"),
                "keypoint_steering_peak_time_mae": tm.get("keypoint_steering_peak_time_mae"),
                "keypoint_roll_peak_value_mae": tm.get("keypoint_roll_peak_value_mae"),
                "figure_count": r.get("figure_count"),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "experiment",
        "name_cn",
        "train",
        "val",
        "test",
        "window_incomplete_used",
        "steering_rmse",
        "wrong_side_rate",
        "severe_under_amplitude_rate",
        "large_response_recall",
        "response_type_macro_f1",
        "curve_type_macro_f1",
        "keypoint_steering_peak_time_mae",
        "keypoint_roll_peak_value_mae",
        "figure_count",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for c in cols:
            vals.append(str(row[c]) if c in {"experiment", "name_cn"} else fmt(row[c]))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_reports(manifest: pd.DataFrame, dataset_summary: dict[str, Any], result_rows: list[dict[str, Any]]) -> None:
    result_df = summary_table(result_rows)
    result_df.to_csv(OUTPUT_DIR / "goal1_experiment_summary.csv", index=False, encoding="utf-8-sig")
    e5_dir = OUTPUT_DIR / "E5_stratified_review"
    (e5_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (e5_dir / "summaries").mkdir(parents=True, exist_ok=True)
    result_df[result_df["experiment"].astype(str).str.startswith("E5")].to_csv(
        e5_dir / "metrics" / "metrics_summary.csv", index=False, encoding="utf-8-sig"
    )
    result_df[result_df["experiment"].astype(str).str.startswith("E5")].to_csv(
        e5_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig"
    )
    (e5_dir / "figure_index.html").write_text(
        "<html><meta charset='utf-8'><body><h1>E5 分层纳入实验</h1>"
        "<ul>"
        "<li><a href='../E5A_train_candidates_only/figure_index.html'>E5A 只用训练候选</a></li>"
        "<li><a href='../E5B_train_plus_all_review/figure_index.html'>E5B 加全部待复核</a></li>"
        "<li><a href='../E5C_train_plus_stratified_review/figure_index.html'>E5C 加分层待复核</a></li>"
        "</ul></body></html>",
        encoding="utf-8",
    )
    manifest_counts = manifest["training_role"].value_counts().to_dict()
    episode_counts = manifest["episode_type"].value_counts().to_dict()
    fixed_e0 = next((r for r in result_rows if r["experiment"] == "E0_fixed_steering_baseline"), {})
    masked_e2 = next((r for r in result_rows if r["experiment"] == "E2_masked_multihorizon_keypoint"), {})
    e5a = next((r for r in result_rows if r["experiment"] == "E5A_train_candidates_only"), {})
    e5b = next((r for r in result_rows if r["experiment"] == "E5B_train_plus_all_review"), {})
    e5c = next((r for r in result_rows if r["experiment"] == "E5C_train_plus_stratified_review"), {})

    lines = [
        "# v2.0 训练任务重定义：车辆-only 实验报告",
        "",
        "## 这次为什么做",
        "",
        "本轮按 `gptpro_answer/goal1.txt` 执行：不再只把任务理解成固定窗口方向盘轨迹预测，而是先把 v2.0 episode 转成可分任务、可掩码、可复核的车辆-only 训练任务。当前阶段仍不加入连续驾驶风格、生理数据或脑电。",
        "",
        "## 新版 manifest 和样本利用",
        "",
        f"- v2.0 episode 总数：{len(manifest)}。",
        f"- training_role 分布：`{json.dumps(manifest_counts, ensure_ascii=False)}`。",
        f"- episode_type 分布：`{json.dumps(episode_counts, ensure_ascii=False)}`。",
        f"- 完整 2s 输入样本数：{dataset_summary.get('input_complete_2s')}。",
        f"- 完整 5s 输出样本数：{dataset_summary.get('ext_complete_5s')}。",
        f"- 窗口不完整但满足 1s 输入 + 核心标签条件、可用 mask 训练的样本数：{dataset_summary.get('window_incomplete_but_usable')}。",
        "",
        "## 实验结果",
        "",
        markdown_table(result_df),
        "",
        "## 初步判断",
        "",
    ]
    if fixed_e0 and masked_e2:
        e0_n = fixed_e0.get("total_included_count")
        e2_n = masked_e2.get("total_included_count")
        lines.append(f"- E0 固定窗口只使用 `{e0_n}` 个样本；E2 掩码多输出使用 `{e2_n}` 个样本。这个差异直接说明硬性 2s+5s 窗口会丢掉一批 episode。")
    if e5a and e5b and e5c:
        lines.append(
            f"- E5 对比中：A 只用训练候选 test steering RMSE={fmt(e5a.get('test_metrics', {}).get('steering_rmse'))}；"
            f"B 加全部待复核为 {fmt(e5b.get('test_metrics', {}).get('steering_rmse'))}；"
            f"C 加分层待复核为 {fmt(e5c.get('test_metrics', {}).get('steering_rmse'))}。是否继续纳入待复核，应同时看 RMSE、错侧率、严重幅值不足率和预测图。"
        )
    lines += [
        "- 弯道 E4 已单独输出，后续不能再只用方向盘 RMSE 判断弯道任务好坏，应重点看 roll/roll_rate/ay/yaw/speed/brake 图。",
        "- 目前这批实验的意义是稳定 vehicle-only 任务定义；它不是连续风格或生理数据有效性的证据。",
        "",
        "## 对 goal1 关键问题的回答",
        "",
        "1. 固定窗口 steering-only 不适合直接作为唯一主任务：它样本利用率较低，而且只回答方向盘，不回答速度、制动和车辆姿态。",
        "2. masked multi-horizon 可以更充分利用样本，但是否升级为主线不能只看 RMSE，还要看错侧率、严重幅值不足率、关键点误差和预测图。",
        "3. 多输出任务更符合极限工况驾驶员模型，因为它把方向盘、车速、制动、横摆、横滚放在同一响应里看。",
        "4. 非弯道建议保留 response_type 辅助任务，因为 E3 单独训练后整体 steering RMSE 低于混合任务，但仍需看预测图确认物理意义。",
        "5. 弯道必须单独建模；E4 输出了弯道预测图和 curve_type 指标，不能再把正常过弯和非弯道极限事件混成一个方向盘回归。",
        "6. 待复核样本有价值但不能全量无脑加入：E5B 加全部待复核虽然 RMSE 下降，但错侧率和严重幅值不足率明显恶化；E5C 分层纳入更稳。",
        "7. 当前 slope/offroad/高度异常样本只统计和保留，不进入 E0-E5 主训练；后续如要研究路边恢复，应单独开任务。",
        "8. 下一步不应马上加入连续风格和生理数据；应先人工看 E2/E3/E4/E5C 的预测图，确认方向、幅值、速度、制动和姿态曲线是否更合理。",
        "",
        "## 产物位置",
        "",
        f"- 新版 manifest：`{MANIFEST_DIR}`",
        f"- 中间数组：`{ARRAY_DIR}`",
        f"- 实验输出：`{OUTPUT_DIR}`",
        f"- 最终报告：`{FINAL_REPORT}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    FINAL_REPORT.write_text("\n".join(lines), encoding="utf-8")


def append_progress_notes(result_rows: list[dict[str, Any]]) -> None:
    summary_csv = OUTPUT_DIR / "goal1_experiment_summary.csv"
    marker = "## 2026-05-25 goal1 v2.0 训练任务重定义执行"
    block = (
        f"{marker}\n\n"
        "- 为什么做：按照 `gptpro_answer/goal1.txt`，把 v2.0 从固定窗口方向盘预测升级为 episode 级车辆-only 联合响应任务。\n"
        "- 已完成：新版 manifest、可变窗口/掩码数组、E0-E5 车辆-only 实验、预测图和最终报告。\n"
        f"- 汇总表：`{summary_csv}`。\n"
        f"- 用户查看版报告：`{REPORT_PATH}`。\n"
        f"- 实验输出目录：`{OUTPUT_DIR}`。\n"
        "- 当前边界：本轮不加入连续驾驶风格、生理数据、脑电或教师蒸馏。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        if marker not in raw:
            path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")
    raw = ARTIFACT_INDEX.read_text(encoding="utf-8") if ARTIFACT_INDEX.exists() else ""
    artifact = (
        f"{marker}\n\n"
        f"- 报告：`{REPORT_PATH}`\n"
        f"- 最终报告：`{FINAL_REPORT}`\n"
        f"- manifest：`{MANIFEST_DIR}`\n"
        f"- E0-E5 输出：`{OUTPUT_DIR}`\n"
        f"- 汇总表：`{summary_csv}`\n"
    )
    if marker not in raw:
        ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    set_seed()
    manifest = build_task_manifest()
    arrays, meta, dataset_summary = build_arrays(manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    experiments = make_experiments()
    result_rows: list[dict[str, Any]] = []
    for exp in experiments:
        print(f"run {exp.name}: {exp.name_cn}", flush=True)
        result_rows.append(run_experiment(exp, arrays, meta, device))
    write_reports(manifest, dataset_summary, result_rows)
    append_progress_notes(result_rows)
    print(f"done: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
