# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
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


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
PROJECT_ROOT = Path(r"F:/data_set_process/data_process")
INSTABILITY_TABLE = (
    ROOT
    / "02_samples"
    / "instability_event_review_v0_1"
    / "tables"
    / "instability_reviewed_events_v0_1.csv"
)
CANDIDATE_EVENTS = ROOT / "02_samples" / "tables" / "candidate_events_master.csv"
MANUAL_LABELS = (
    ROOT
    / "02_samples"
    / "manual_event_keyboard_player_v0_1"
    / "tables"
    / "keyboard_instability_event_labels_v0_1.csv"
)
ROAD_LAYOUT = (
    PROJECT_ROOT
    / "01_datasets"
    / "多模态数据"
    / "被试数据集合"
    / "道路信息"
    / "full_centerline_layout.csv"
)

OUT_DIR = ROOT / "02_samples" / "road_guided_instability_v0_1"
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"

OLD_NEAR_PRE_S = 2.0
OLD_NEAR_POST_S = 5.0

ROAD_VEHICLE_COLS = ["StorageTime", "zx|x", "zx|y"]

LEVEL_RANK = {
    "weak": 1,
    "medium": 2,
    "medium_active": 2,
    "strong": 3,
    "strong_active": 3,
    "extreme": 4,
    "extreme_active": 4,
}

HIGH_RISK_INSTANCES = {"mu1", "differentmu_road"}
SPECIAL_INSTANCES = {"fix_road", "stop", "zd"}
CURVE_INSTANCES = {"curve1", "curve2", "curve3"}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def to_seconds(storage_time: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(storage_time, errors="coerce")
    out = np.full(len(storage_time), np.nan, dtype=np.float64)
    valid = parsed.notna().to_numpy()
    if valid.any():
        ns = parsed[valid].astype("datetime64[ns]").astype("int64").to_numpy(dtype=np.float64)
        out[valid] = ns / 1e9
    return out


def clip(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def mode_or_unknown(values: list[str]) -> str:
    cleaned = [str(v) for v in values if str(v) and str(v).lower() != "nan"]
    if not cleaned:
        return "unknown"
    return Counter(cleaned).most_common(1)[0][0]


def max_level_rank(values: list[str]) -> tuple[int, str]:
    best_rank = 0
    best_name = "unknown"
    for value in values:
        name = str(value)
        rank = LEVEL_RANK.get(name, 0)
        if rank > best_rank:
            best_rank = rank
            best_name = name
    return best_rank, best_name


def load_manual_labels() -> pd.DataFrame:
    if not MANUAL_LABELS.exists():
        return pd.DataFrame()
    labels = pd.read_csv(MANUAL_LABELS, low_memory=False)
    if "selected_candidate_event_uid" not in labels.columns:
        return pd.DataFrame()
    labels = labels.dropna(subset=["selected_candidate_event_uid"]).copy()
    labels["selected_candidate_event_uid"] = labels["selected_candidate_event_uid"].astype(str)
    return labels


def prepare_old_events() -> pd.DataFrame:
    events = pd.read_csv(CANDIDATE_EVENTS, low_memory=False)
    old = events[events["anchor_source"].eq("old_v400_context_trigger_idx")].copy()
    if old.empty:
        return old
    for col in [
        "anchor_time_rel_s",
        "event_start_rel_s",
        "event_end_rel_s",
        "old_primary_score",
        "old_trigger_score",
        "curvature_anchor",
    ]:
        if col in old.columns:
            old[col] = pd.to_numeric(old[col], errors="coerce")
    return old


def summarize_old_context(row: pd.Series, grouped_old: dict[tuple[str, str], pd.DataFrame]) -> dict[str, Any]:
    subject = str(row["subject"])
    session = str(row["session_stamp"])
    anchor = finite_float(row.get("anchor_time_rel_s"))
    start = finite_float(row.get("event_start_rel_s"))
    end = finite_float(row.get("event_end_rel_s"))
    group = grouped_old.get((subject, session))
    empty = {
        "old_v400_near_count": 0,
        "old_v400_overlap_count": 0,
        "old_v400_primary_count": 0,
        "old_v400_active_count": 0,
        "old_v400_max_primary_score": np.nan,
        "old_v400_max_trigger_score": np.nan,
        "old_v400_max_level_rank": 0,
        "old_v400_max_level": "none",
        "old_v400_road_type_mode": "none",
        "old_v400_phase_mode": "none",
        "old_v400_min_abs_anchor_gap_s": np.nan,
        "old_v400_nearest_event_uid": "",
    }
    if group is None or group.empty or not math.isfinite(anchor):
        return empty

    near = group[
        (group["anchor_time_rel_s"] >= anchor - OLD_NEAR_PRE_S)
        & (group["anchor_time_rel_s"] <= anchor + OLD_NEAR_POST_S)
    ].copy()
    if near.empty:
        return empty

    if math.isfinite(start) and math.isfinite(end):
        overlap = near[
            (near["event_start_rel_s"].fillna(near["anchor_time_rel_s"]) <= end + 1.0)
            & (near["event_end_rel_s"].fillna(near["anchor_time_rel_s"]) >= start - 1.0)
        ]
    else:
        overlap = pd.DataFrame()

    level_rank, level_name = max_level_rank(near.get("event_level", pd.Series(dtype=str)).astype(str).tolist())
    gaps = (near["anchor_time_rel_s"] - anchor).abs()
    nearest_idx = gaps.idxmin()
    nearest_uid = str(near.loc[nearest_idx, "event_uid"]) if "event_uid" in near.columns else ""
    max_primary = finite_float(near["old_primary_score"].max()) if "old_primary_score" in near.columns else np.nan
    max_trigger = finite_float(near["old_trigger_score"].max()) if "old_trigger_score" in near.columns else np.nan
    levels = near.get("event_level", pd.Series(dtype=str)).astype(str)
    phases = near.get("phase_type", pd.Series(dtype=str)).astype(str)
    roads = near.get("road_type_anchor", pd.Series(dtype=str)).astype(str)

    return {
        "old_v400_near_count": int(len(near)),
        "old_v400_overlap_count": int(len(overlap)),
        "old_v400_primary_count": int((phases == "primary").sum()),
        "old_v400_active_count": int(levels.str.contains("active", na=False).sum()),
        "old_v400_max_primary_score": round(max_primary, 6) if math.isfinite(max_primary) else np.nan,
        "old_v400_max_trigger_score": round(max_trigger, 6) if math.isfinite(max_trigger) else np.nan,
        "old_v400_max_level_rank": int(level_rank),
        "old_v400_max_level": level_name,
        "old_v400_road_type_mode": mode_or_unknown(roads.tolist()),
        "old_v400_phase_mode": mode_or_unknown(phases.tolist()),
        "old_v400_min_abs_anchor_gap_s": round(float(gaps.min()), 6),
        "old_v400_nearest_event_uid": nearest_uid,
    }


def load_road_mapper() -> tuple[Any | None, pd.DataFrame | None, str]:
    if cKDTree is None:
        return None, None, "scipy_unavailable"
    if not ROAD_LAYOUT.exists():
        return None, None, "layout_missing"
    layout = pd.read_csv(ROAD_LAYOUT, low_memory=False)
    required = {"x", "y", "s", "module_name", "instance_name", "curvature"}
    if not required.issubset(set(layout.columns)):
        return None, None, "layout_missing_required_columns"
    xy = layout[["x", "y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(xy).all(axis=1)
    layout = layout.loc[valid].reset_index(drop=True)
    if layout.empty:
        return None, None, "layout_no_valid_xy"
    tree = cKDTree(layout[["x", "y"]].to_numpy(dtype=np.float64))
    return tree, layout, "ok"


def load_vehicle_position(path: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if path in cache:
        return cache[path]
    df = pd.read_csv(path, usecols=lambda c: c in ROAD_VEHICLE_COLS)
    if "StorageTime" not in df.columns:
        raise ValueError("missing StorageTime")
    t_abs = to_seconds(df["StorageTime"])
    valid_t = np.isfinite(t_abs)
    if not valid_t.any():
        raise ValueError("no valid StorageTime")
    df = df.loc[valid_t].copy()
    t_abs = t_abs[valid_t]
    df["t_rel_s"] = t_abs - float(t_abs[0])
    for col in ["zx|x", "zx|y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    df = df.sort_values("t_rel_s").reset_index(drop=True)
    cache[path] = df
    return df


def nearest_finite_vehicle_row(vehicle: pd.DataFrame, anchor_s: float) -> pd.Series | None:
    if vehicle.empty or not math.isfinite(anchor_s):
        return None
    valid = vehicle[["zx|x", "zx|y", "t_rel_s"]].notna().all(axis=1).to_numpy()
    if not valid.any():
        return None
    valid_indices = np.flatnonzero(valid)
    valid_t = vehicle.loc[valid, "t_rel_s"].to_numpy(dtype=np.float64)
    pos = int(np.searchsorted(valid_t, anchor_s))
    candidates: list[int] = []
    if pos < len(valid_indices):
        candidates.append(int(valid_indices[pos]))
    if pos > 0:
        candidates.append(int(valid_indices[pos - 1]))
    if not candidates:
        return None
    best = min(candidates, key=lambda idx: abs(float(vehicle.loc[idx, "t_rel_s"]) - anchor_s))
    return vehicle.loc[best]


def road_reliability(distance_m: float, time_gap_s: float) -> str:
    if not math.isfinite(distance_m) or not math.isfinite(time_gap_s):
        return "unmapped"
    if time_gap_s > 0.05:
        return "low_time_gap"
    if distance_m <= 20.0:
        return "high"
    if distance_m <= 150.0:
        return "medium"
    if distance_m <= 500.0:
        return "low"
    return "very_low"


def road_risk_class(instance: str) -> str:
    instance = str(instance)
    if instance in HIGH_RISK_INSTANCES:
        return "design_high_risk_surface"
    if instance in SPECIAL_INSTANCES:
        return "design_special_event_segment"
    if instance in CURVE_INSTANCES:
        return "design_curve_context"
    if instance in {"section1", "section2", "section3", "section4", "section5", "section6", "section7", "section8", "longstraight"}:
        return "design_regular_road"
    if not instance:
        return "unmapped"
    return "design_other"


def map_road_context(
    row: pd.Series,
    tree: Any | None,
    layout: pd.DataFrame | None,
    mapper_status: str,
    vehicle_cache: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    base = {
        "road_design_map_status": mapper_status,
        "road_design_module_name": "",
        "road_design_instance_name": "",
        "road_design_s": np.nan,
        "road_design_curvature": np.nan,
        "road_design_nearest_dist_m": np.nan,
        "road_design_vehicle_time_gap_s": np.nan,
        "road_design_mapping_reliability": "unmapped",
        "road_design_risk_class": "unmapped",
    }
    if tree is None or layout is None:
        return base
    path = str(row.get("vehicle_raw_absolute_path", ""))
    anchor = finite_float(row.get("anchor_time_rel_s"))
    try:
        vehicle = load_vehicle_position(path, vehicle_cache)
        vehicle_row = nearest_finite_vehicle_row(vehicle, anchor)
        if vehicle_row is None:
            base["road_design_map_status"] = "vehicle_position_missing"
            return base
        xy = np.array([[float(vehicle_row["zx|x"]), float(vehicle_row["zx|y"])]], dtype=np.float64)
        dist, idx = tree.query(xy, k=1)
        layout_row = layout.iloc[int(idx[0])]
        distance_m = float(dist[0])
        time_gap_s = abs(float(vehicle_row["t_rel_s"]) - anchor)
        instance = str(layout_row.get("instance_name", ""))
        base.update(
            {
                "road_design_map_status": "ok",
                "road_design_module_name": str(layout_row.get("module_name", "")),
                "road_design_instance_name": instance,
                "road_design_s": round(finite_float(layout_row.get("s")), 6),
                "road_design_curvature": round(finite_float(layout_row.get("curvature")), 8),
                "road_design_nearest_dist_m": round(distance_m, 6),
                "road_design_vehicle_time_gap_s": round(time_gap_s, 6),
                "road_design_mapping_reliability": road_reliability(distance_m, time_gap_s),
                "road_design_risk_class": road_risk_class(instance),
            }
        )
    except Exception as exc:  # noqa: BLE001
        base["road_design_map_status"] = f"error:{type(exc).__name__}"
    return base


def manual_context(row: pd.Series, manual_by_uid: dict[str, pd.DataFrame]) -> dict[str, Any]:
    uid = str(row["instability_event_uid"])
    labels = manual_by_uid.get(uid)
    if labels is None or labels.empty:
        return {
            "manual_label_count": 0,
            "manual_label_decision_mode": "none",
            "manual_label_anchor_rel_s": np.nan,
            "manual_label_start_rel_s": np.nan,
            "manual_label_end_rel_s": np.nan,
            "manual_label_confidence_max": np.nan,
        }
    decisions = labels.get("decision", pd.Series(dtype=str)).astype(str).tolist()
    confidence = pd.to_numeric(labels.get("confidence_1_5", pd.Series(dtype=float)), errors="coerce")
    latest = labels.iloc[-1]
    return {
        "manual_label_count": int(len(labels)),
        "manual_label_decision_mode": mode_or_unknown(decisions),
        "manual_label_anchor_rel_s": round(finite_float(latest.get("anchor_rel_s")), 6),
        "manual_label_start_rel_s": round(finite_float(latest.get("event_start_rel_s")), 6),
        "manual_label_end_rel_s": round(finite_float(latest.get("event_end_rel_s")), 6),
        "manual_label_confidence_max": round(float(confidence.max()), 6) if confidence.notna().any() else np.nan,
    }


def score_hybrid(row: dict[str, Any]) -> tuple[float, str, str]:
    base = finite_float(row.get("instability_review_score"), 0.0)
    score = base
    reasons: list[str] = [f"base_dynamic_score={base:.2f}"]

    ay = finite_float(row.get("peak_abs_ay_window"), 0.0)
    roll_rate = finite_float(row.get("peak_abs_roll_rate_window"), 0.0)
    yaw_rate = finite_float(row.get("peak_abs_yaw_rate_window"), 0.0)
    lateral = finite_float(row.get("lateral_distance_range_window"), 0.0)
    steering_after = finite_float(row.get("steering_delta_peak_post3s"), 0.0)
    speed = finite_float(row.get("median_speed_kmh_window"), 0.0)
    duration = finite_float(row.get("event_duration_s"), 0.0)
    curvature = finite_float(row.get("peak_abs_curvature_window"), 0.0)

    old_count = int(row.get("old_v400_near_count", 0) or 0)
    old_primary = int(row.get("old_v400_primary_count", 0) or 0)
    old_active = int(row.get("old_v400_active_count", 0) or 0)
    old_level_rank = int(row.get("old_v400_max_level_rank", 0) or 0)
    old_primary_score = finite_float(row.get("old_v400_max_primary_score"), 0.0)
    old_road = str(row.get("old_v400_road_type_mode", "none"))
    risk_class = str(row.get("road_design_risk_class", "unmapped"))
    reliability = str(row.get("road_design_mapping_reliability", "unmapped"))
    original_decision = str(row.get("codex_recommended_decision", ""))
    manual_decision = str(row.get("manual_label_decision_mode", "none"))

    if ay >= 2.0:
        add = clip((ay - 2.0) / 4.0, 0.0, 1.0) * 8.0
        score += add
        reasons.append(f"ay_support=+{add:.1f}")
    if roll_rate >= 0.35:
        add = clip(roll_rate / 1.2, 0.0, 1.0) * 8.0
        score += add
        reasons.append(f"roll_rate_support=+{add:.1f}")
    if yaw_rate >= 0.12:
        score += 5.0
        reasons.append("yaw_rate_support=+5")
    if lateral >= 2.5:
        add = clip((lateral - 2.5) / 5.0, 0.0, 1.0) * 6.0
        score += add
        reasons.append(f"lateral_support=+{add:.1f}")
    if steering_after >= 0.45:
        add = clip((steering_after - 0.45) / 1.2, 0.0, 1.0) * 5.0
        score += add
        reasons.append(f"post_steering_response=+{add:.1f}")
    if duration >= 4.0 and ay >= 1.8:
        score += 4.0
        reasons.append("multi_second_dynamic_episode=+4")

    if old_count > 0:
        add = min(6.0, old_count * 1.5)
        score += add
        reasons.append(f"old_v400_nearby=+{add:.1f}")
    if old_primary > 0:
        score += 5.0
        reasons.append("old_v400_primary=+5")
    if old_active > 0:
        score += 5.0
        reasons.append("old_v400_active=+5")
    if old_level_rank >= 4:
        score += 6.0
        reasons.append("old_v400_extreme=+6")
    elif old_level_rank >= 3:
        score += 4.0
        reasons.append("old_v400_strong=+4")
    if old_primary_score >= 2.0:
        add = clip(old_primary_score / 8.0, 0.0, 1.0) * 5.0
        score += add
        reasons.append(f"old_primary_score=+{add:.1f}")

    reliable_road = reliability in {"high", "medium", "low"}
    if risk_class == "design_high_risk_surface" and reliable_road:
        score += 8.0
        reasons.append("design_high_risk_surface=+8")
    elif risk_class == "design_special_event_segment" and reliable_road:
        score += 4.0
        reasons.append("design_special_segment=+4")

    curve_context = (
        curvature >= 8e-4
        or old_road == "curve"
        or risk_class == "design_curve_context"
    )
    strong_multisignal = (
        (ay >= 3.0 and (yaw_rate >= 0.12 or lateral >= 2.5 or steering_after >= 0.45))
        or roll_rate >= 0.8
        or original_decision == "auto_accept_instability_high"
    )
    medium_multisignal = (
        (ay >= 2.0 and (yaw_rate >= 0.08 or lateral >= 2.0 or steering_after >= 0.35))
        or roll_rate >= 0.45
        or original_decision == "auto_accept_instability_medium"
    )

    if curve_context and not medium_multisignal:
        score -= 10.0
        reasons.append("normal_curve_context_without_enough_instability=-10")
    elif curve_context and strong_multisignal:
        reasons.append("curve_context_kept_because_dynamic_response_is_strong")

    if speed < 5.0 and ay < 1.2 and roll_rate < 0.6:
        score -= 12.0
        reasons.append("low_speed_weak_dynamic=-12")

    if manual_decision in {"accept_candidate", "manual_adjusted"}:
        score = max(score, 72.0)
        reasons.append(f"manual_calibration_label={manual_decision}")

    score = round(clip(score, 0.0, 100.0), 2)

    if manual_decision in {"accept_candidate", "manual_adjusted"}:
        decision = "manual_confirmed_accept"
    elif score >= 82.0 or original_decision == "auto_accept_instability_high":
        decision = "hybrid_accept_high"
    elif score >= 62.0 or (score >= 56.0 and medium_multisignal and (old_count > 0 or reliable_road)):
        decision = "hybrid_accept_medium"
    elif score >= 45.0:
        decision = "hybrid_review_conflict_or_medium"
    else:
        decision = "hybrid_reject_low_evidence"

    return score, decision, "; ".join(reasons)


def build() -> dict[str, Any]:
    ensure_dirs()
    inst = pd.read_csv(INSTABILITY_TABLE, low_memory=False)
    old = prepare_old_events()
    old_groups = {
        (str(subject), str(session)): group.copy()
        for (subject, session), group in old.groupby(["subject", "session_stamp"], sort=False)
    }
    labels = load_manual_labels()
    manual_groups = {
        str(uid): group.copy()
        for uid, group in labels.groupby("selected_candidate_event_uid", sort=False)
    }
    tree, layout, mapper_status = load_road_mapper()
    vehicle_cache: dict[str, pd.DataFrame] = {}

    rows: list[dict[str, Any]] = []
    for _, source_row in inst.iterrows():
        row = source_row.to_dict()
        row.update(summarize_old_context(source_row, old_groups))
        row.update(map_road_context(source_row, tree, layout, mapper_status, vehicle_cache))
        row.update(manual_context(source_row, manual_groups))
        score, decision, reasons = score_hybrid(row)
        row["road_guided_instability_score"] = score
        row["road_guided_recommended_decision"] = decision
        row["road_guided_decision_reasons"] = reasons
        row["dataset_candidate_version"] = "road_guided_vehicle_instability_v0_1"
        row["causal_setting"] = "road_setting_guided_vehicle_instability_onset_predict_future_steering_response"
        row["leakage_note"] = (
            "Road design and old v400 event context are used only as supporting context. "
            "Instability onset remains grounded in non-steering vehicle dynamics; steering metrics are response evidence."
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    accepted_mask = out["road_guided_recommended_decision"].isin(
        ["hybrid_accept_high", "hybrid_accept_medium", "manual_confirmed_accept"]
    )
    review_mask = out["road_guided_recommended_decision"].eq("hybrid_review_conflict_or_medium")

    full_path = TABLE_DIR / "road_guided_instability_events_v0_1.csv"
    accepted_path = TABLE_DIR / "road_guided_auto_accepted_events_v0_1.csv"
    review_path = TABLE_DIR / "road_guided_review_queue_v0_1.csv"
    summary_path = TABLE_DIR / "road_guided_instability_summary_v0_1.csv"
    road_module_path = TABLE_DIR / "road_guided_module_summary_v0_1.csv"
    manual_eval_path = TABLE_DIR / "road_guided_manual_calibration_v0_1.csv"

    out.to_csv(full_path, index=False, encoding="utf-8-sig")
    out.loc[accepted_mask].to_csv(accepted_path, index=False, encoding="utf-8-sig")
    out.loc[review_mask].to_csv(review_path, index=False, encoding="utf-8-sig")

    summary_rows = []
    for name, value in out["road_guided_recommended_decision"].value_counts().items():
        summary_rows.append({"summary_type": "decision_count", "key": name, "value": int(value)})
    for name, value in out["road_design_risk_class"].value_counts().items():
        summary_rows.append({"summary_type": "road_design_risk_class_count", "key": name, "value": int(value)})
    for name, value in out["old_v400_road_type_mode"].value_counts().items():
        summary_rows.append({"summary_type": "old_v400_road_type_count", "key": name, "value": int(value)})
    summary_rows.extend(
        [
            {"summary_type": "total", "key": "all_candidates", "value": int(len(out))},
            {"summary_type": "total", "key": "accepted_candidates", "value": int(accepted_mask.sum())},
            {"summary_type": "total", "key": "review_candidates", "value": int(review_mask.sum())},
            {"summary_type": "total", "key": "manual_label_rows_used_as_calibration", "value": int(len(labels))},
        ]
    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    module_summary = (
        out.groupby(["road_design_mapping_reliability", "road_design_risk_class", "road_design_instance_name", "road_guided_recommended_decision"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["road_design_risk_class", "road_design_instance_name", "road_guided_recommended_decision"])
    )
    module_summary.to_csv(road_module_path, index=False, encoding="utf-8-sig")

    if not labels.empty:
        manual_cols = [
            "instability_event_uid",
            "manual_label_decision_mode",
            "manual_label_count",
            "codex_recommended_decision",
            "road_guided_recommended_decision",
            "instability_review_score",
            "road_guided_instability_score",
            "road_design_instance_name",
            "road_design_risk_class",
            "old_v400_near_count",
            "old_v400_max_level",
            "peak_abs_ay_window",
            "peak_abs_roll_rate_window",
            "peak_abs_yaw_rate_window",
            "lateral_distance_range_window",
            "steering_delta_peak_post3s",
        ]
        manual_eval = out[out["manual_label_count"] > 0][manual_cols].copy()
        manual_eval.to_csv(manual_eval_path, index=False, encoding="utf-8-sig")

    report_path = REPORT_DIR / "road_guided_instability_v0_1_cn.md"
    write_report(
        report_path=report_path,
        out=out,
        full_path=full_path,
        accepted_path=accepted_path,
        review_path=review_path,
        summary_path=summary_path,
        road_module_path=road_module_path,
        manual_eval_path=manual_eval_path,
        mapper_status=mapper_status,
    )

    run_summary = {
        "generated_at": now_str(),
        "input_instability_table": str(INSTABILITY_TABLE).replace("\\", "/"),
        "input_candidate_events": str(CANDIDATE_EVENTS).replace("\\", "/"),
        "input_manual_labels": str(MANUAL_LABELS).replace("\\", "/") if MANUAL_LABELS.exists() else "",
        "input_road_layout": str(ROAD_LAYOUT).replace("\\", "/"),
        "road_mapper_status": mapper_status,
        "all_candidates": int(len(out)),
        "accepted_candidates": int(accepted_mask.sum()),
        "review_candidates": int(review_mask.sum()),
        "decision_counts": {str(k): int(v) for k, v in out["road_guided_recommended_decision"].value_counts().items()},
        "outputs": {
            "full": str(full_path).replace("\\", "/"),
            "accepted": str(accepted_path).replace("\\", "/"),
            "review": str(review_path).replace("\\", "/"),
            "summary": str(summary_path).replace("\\", "/"),
            "report": str(report_path).replace("\\", "/"),
        },
    }
    (LOG_DIR / "road_guided_instability_run_summary_v0_1.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_summary


def write_report(
    report_path: Path,
    out: pd.DataFrame,
    full_path: Path,
    accepted_path: Path,
    review_path: Path,
    summary_path: Path,
    road_module_path: Path,
    manual_eval_path: Path,
    mapper_status: str,
) -> None:
    decision_counts = out["road_guided_recommended_decision"].value_counts().to_string()
    old_road_counts = out["old_v400_road_type_mode"].value_counts().to_string()
    risk_counts = out["road_design_risk_class"].value_counts().to_string()
    reliability_counts = out["road_design_mapping_reliability"].value_counts().to_string()
    accepted = out["road_guided_recommended_decision"].isin(
        ["hybrid_accept_high", "hybrid_accept_medium", "manual_confirmed_accept"]
    )
    manual_count = int((out["manual_label_count"] > 0).sum())
    full_path_s = str(full_path).replace("\\", "/")
    accepted_path_s = str(accepted_path).replace("\\", "/")
    review_path_s = str(review_path).replace("\\", "/")
    summary_path_s = str(summary_path).replace("\\", "/")
    road_module_path_s = str(road_module_path).replace("\\", "/")
    manual_eval_path_s = str(manual_eval_path).replace("\\", "/")
    text = f"""# 道路设定引导的车辆失稳事件自动判定 v0.1

生成时间：{now_str()}

## 为什么做

用户指出，上一版 404 个样本主要是弯道/道路曲率样本，不是项目真正需要的车辆失稳样本。逐个手工标注 1227 个失稳候选也不现实，所以本版改成自动综合判定：用原始车辆动态作为主证据，用旧项目日志和道路设定作为辅助先验。

## 用了哪些证据

1. 主证据：`ay` 和 `roll_rate` 触发的非方向盘车辆动态失稳候选。
2. 车辆响应证据：横摆角速度、横向偏移、事件后 3 秒方向盘修正幅值、车速、片段持续时间。
3. 旧流程上下文：`*_events_v400_context.csv` 在项目日志中被记录为优先事件来源，提供 `road_type_anchor`、`phase_type`、`event_level`、`trigger_idx` 等旧事件上下文。
4. 道路设定先验：从 `full_centerline_layout.csv` 读取道路模块顺序，识别 `curve1/curve2/curve3`、`fix_road`、`stop`、`mu1/differentmu_road`、`zd` 等道路场景。
5. 已有人工抽查：当前 31 条键盘标注只作为校准/确认，不要求继续人工标注全量样本。

## 关键原则

- 弯道不等于失稳。弯道只作为道路上下文，如果车辆动态证据弱，会被降权。
- 方向盘动作不用于定义失稳开始点。方向盘只作为事件之后的响应证据，避免把驾驶员操作结果泄漏进事件锚点。
- 旧 v400 事件不是新真值。它只作为旧道路事件设定和旧锚点上下文，不能替代原始车辆动态证据。
- 道路模块映射不是绝对真值。车辆坐标到道路中心线的最近距离会记录可靠度，高距离映射只作为弱参考。

## 当前结果

候选总数：{len(out)}

自动/已确认采用数：{int(accepted.sum())}

需要复核但不要求用户逐条手工标注的中间候选：{int((out["road_guided_recommended_decision"] == "hybrid_review_conflict_or_medium").sum())}

31 条已有人工抽查命中的候选数：{manual_count}

按最终建议统计：

```text
{decision_counts}
```

旧 v400 道路类型支持统计：

```text
{old_road_counts}
```

道路设计风险类别统计：

```text
{risk_counts}
```

道路中心线映射可靠度：

```text
{reliability_counts}
```

## 产物

- 全量判定表：`{full_path_s}`
- 自动采用表：`{accepted_path_s}`
- 中间复核队列表：`{review_path_s}`
- 汇总表：`{summary_path_s}`
- 道路模块交叉表：`{road_module_path_s}`
- 人工抽查校准表：`{manual_eval_path_s}`

## 目前可以怎么用

本版可以替代“全人工标注”的第一轮失稳样本筛选。下一步不应该回到 404 个弯道样本，也不应该马上训练模型，而是用自动采用表生成 `vehicle_instability` 样本 manifest，并在样本卡里记录每个事件的道路上下文、旧 v400 支持和失稳动态证据。

## 还不能下的结论

- 不能说这些事件已经是完全人工真值。
- 不能说道路模块本身导致失稳，只能说它提供场景先验。
- 不能说生理或连续风格有效，因为这里还没有进入风格/生理建模。
- 不能把旧 v400 锚点继续当作新流程唯一锚点。

## 质量风险

道路映射状态：`{mapper_status}`。

如果某些候选的 `road_design_mapping_reliability` 是 `very_low`，说明车辆坐标到中心线距离过大，模块名称只能作为弱参考。最终样本构建时应优先依赖非方向盘车辆动态证据和旧 v400 近邻上下文，而不是单独依赖该模块名。
"""
    report_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    summary = build()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
