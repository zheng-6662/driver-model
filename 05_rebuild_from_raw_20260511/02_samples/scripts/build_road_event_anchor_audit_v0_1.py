# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except Exception:  # noqa: BLE001
    cKDTree = None


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
PROJECT_ROOT = Path(r"F:/data_set_process/data_process")

AUDIT_DIR = ROOT / "02_samples" / "road_event_anchor_audit_v0_1"
TABLE_DIR = AUDIT_DIR / "tables"
FIGURE_DIR = AUDIT_DIR / "figures"
PANEL_DIR = FIGURE_DIR / "representative_panels"
LOG_DIR = AUDIT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"

RAW_INVENTORY = ROOT / "01_audit" / "tables" / "raw_file_inventory.csv"
CANDIDATE_EVENTS = ROOT / "02_samples" / "tables" / "candidate_events_master.csv"
ANCHOR_SOURCE_COMPARISON = ROOT / "02_samples" / "tables" / "anchor_source_comparison.csv"
ROAD_GUIDED_EVENTS = (
    ROOT
    / "02_samples"
    / "road_guided_instability_v0_1"
    / "tables"
    / "road_guided_instability_events_v0_1.csv"
)

ROAD_VEHICLE_COLS = [
    "StorageTime",
    "zx1|v_km/h",
    "zx1|lanecurvatureXY",
    "zx1|lateraldistance",
    "zx|x",
    "zx|y",
    "zx|SteeringWheel",
    "zx|ay",
    "zx|vyaw",
    "zx|vroll",
]

HIGH_RISK_INSTANCES = {"mu1", "differentmu_road"}
SPECIAL_INSTANCES = {"fix_road", "stop", "zd"}
CURVE_INSTANCES = {"curve1", "curve2", "curve3"}


def configure_matplotlib_fonts() -> None:
    font_candidates = [
        Path(r"C:/Windows/Fonts/msyh.ttc"),
        Path(r"C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path(r"C:/Windows/Fonts/simhei.ttf"),
        Path(r"C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            prop = font_manager.FontProperties(fname=str(font_path))
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIGURE_DIR, PANEL_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


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


def mode_or_unknown(values: pd.Series | list[Any]) -> str:
    if isinstance(values, pd.Series):
        raw = values.dropna().astype(str).tolist()
    else:
        raw = [str(v) for v in values if pd.notna(v)]
    cleaned = [v for v in raw if v and v.lower() not in {"nan", "none", "unknown"}]
    if not cleaned:
        return "unknown"
    return Counter(cleaned).most_common(1)[0][0]


def risk_class(instance: str) -> str:
    name = str(instance)
    if name in HIGH_RISK_INSTANCES:
        return "高风险路面"
    if name in SPECIAL_INSTANCES:
        return "特殊道路段"
    if name in CURVE_INSTANCES:
        return "弯道路段"
    if name == "longstraight":
        return "长直线路段"
    if name.startswith("section"):
        return "普通连接段"
    return "普通道路段"


def reliability_from_distance(distance_m: float, time_gap_s: float = 0.0) -> str:
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


def segment_reliability(median_dist: float, high_ratio: float, medium_ratio: float) -> str:
    if not math.isfinite(median_dist):
        return "unmapped"
    if high_ratio >= 0.7 and median_dist <= 20.0:
        return "high"
    if high_ratio + medium_ratio >= 0.7 and median_dist <= 150.0:
        return "medium"
    if median_dist <= 500.0:
        return "low"
    return "very_low"


def find_road_layout() -> Path:
    candidates = sorted((PROJECT_ROOT / "01_datasets").rglob("full_centerline_layout.csv"))
    usable: list[tuple[int, Path]] = []
    required = {"s", "x", "y", "curvature", "module_name", "instance_name"}
    for path in candidates:
        try:
            head = pd.read_csv(path, nrows=5)
        except Exception:  # noqa: BLE001
            continue
        if required.issubset(set(head.columns)):
            usable.append((path.stat().st_size, path))
    if not usable:
        raise FileNotFoundError("No usable full_centerline_layout.csv was found.")
    return sorted(usable, reverse=True)[0][1]


def load_road_layout() -> pd.DataFrame:
    layout_path = find_road_layout()
    layout = pd.read_csv(layout_path, low_memory=False)
    for col in ["s", "x", "y", "curvature"]:
        layout[col] = pd.to_numeric(layout[col], errors="coerce")
    layout = layout.dropna(subset=["s", "x", "y"]).copy()
    layout["module_name"] = layout["module_name"].fillna("unknown").astype(str)
    layout["instance_name"] = layout["instance_name"].fillna(layout["module_name"]).astype(str)
    layout["direction"] = layout.get("direction", pd.Series(["unknown"] * len(layout))).fillna("unknown").astype(str)
    layout.attrs["source_path"] = str(layout_path)
    return layout.sort_values("s").reset_index(drop=True)


def build_road_event_position_map(layout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (module, instance), group in layout.groupby(["module_name", "instance_name"], sort=False):
        s_min = float(group["s"].min())
        s_max = float(group["s"].max())
        curv = pd.to_numeric(group["curvature"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "module_name": module,
                "instance_name": instance,
                "road_risk_class_cn": risk_class(str(instance)),
                "s_start_m": round(s_min, 3),
                "s_end_m": round(s_max, 3),
                "length_m": round(max(0.0, s_max - s_min), 3),
                "x_min": round(float(group["x"].min()), 3),
                "x_max": round(float(group["x"].max()), 3),
                "y_min": round(float(group["y"].min()), 3),
                "y_max": round(float(group["y"].max()), 3),
                "max_abs_curvature": round(float(curv.abs().max()), 8),
                "mean_abs_curvature": round(float(curv.abs().mean()), 8),
                "nonzero_curvature_points": int((curv.abs() > 1e-8).sum()),
                "direction_mode": mode_or_unknown(group["direction"]),
                "source_layout_path": layout.attrs.get("source_path", ""),
            }
        )
    return pd.DataFrame(rows)


def load_vehicle_inventory() -> pd.DataFrame:
    inv = read_csv(RAW_INVENTORY)
    vehicle = inv[inv["modality"].astype(str).eq("vehicle")].copy()
    vehicle = vehicle[vehicle["absolute_path"].apply(lambda p: Path(str(p)).exists())].copy()
    return vehicle.sort_values(["subject", "session_stamp"]).reset_index(drop=True)


def load_vehicle_track(path: str) -> pd.DataFrame:
    available = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in ROAD_VEHICLE_COLS if c in available]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    t_abs = to_seconds(df["StorageTime"])
    valid_t = np.isfinite(t_abs)
    df = df.loc[valid_t].copy()
    t_abs = t_abs[valid_t]
    if len(df) == 0:
        return pd.DataFrame()
    df["t_abs_s"] = t_abs
    df["t_rel_s"] = t_abs - float(t_abs[0])
    for col in usecols:
        if col != "StorageTime":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("t_rel_s").reset_index(drop=True)
    return df


def downsample_track(track: pd.DataFrame, step_s: float = 0.1) -> pd.DataFrame:
    if track.empty:
        return track
    valid = track.dropna(subset=["zx|x", "zx|y", "t_rel_s"]).copy()
    if valid.empty:
        return valid
    valid["sample_bin"] = np.floor(valid["t_rel_s"] / step_s).astype(int)
    agg: dict[str, str] = {c: "first" for c in valid.columns if c != "sample_bin"}
    return valid.groupby("sample_bin", as_index=False).agg(agg).drop(columns=["sample_bin"])


def build_session_module_entries(layout: pd.DataFrame, vehicle_inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cKDTree is None:
        raise RuntimeError("scipy is required for KD-tree road mapping.")
    xy = layout[["x", "y"]].to_numpy(dtype=np.float64)
    tree = cKDTree(xy)

    segment_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []

    for rec in vehicle_inventory.itertuples(index=False):
        path = str(getattr(rec, "absolute_path"))
        subject = str(getattr(rec, "subject"))
        session = str(getattr(rec, "session_stamp"))
        rel_path = str(getattr(rec, "relative_path"))
        try:
            track = load_vehicle_track(path)
            sampled = downsample_track(track)
            if sampled.empty:
                session_rows.append(
                    {
                        "subject": subject,
                        "session_stamp": session,
                        "vehicle_raw_relative_path": rel_path,
                        "mapping_status": "no_valid_xy",
                        "sampled_point_count": 0,
                    }
                )
                continue
            dist, idx = tree.query(sampled[["zx|x", "zx|y"]].to_numpy(dtype=np.float64), k=1)
            nearest = layout.iloc[idx].reset_index(drop=True)
            mapped = sampled[["t_rel_s", "zx|x", "zx|y"]].reset_index(drop=True).copy()
            mapped["road_nearest_dist_m"] = dist
            mapped["road_s"] = nearest["s"].to_numpy()
            mapped["road_module_name"] = nearest["module_name"].astype(str).to_numpy()
            mapped["road_instance_name"] = nearest["instance_name"].astype(str).to_numpy()
            mapped["road_risk_class_cn"] = [risk_class(v) for v in mapped["road_instance_name"]]
            mapped["point_reliability"] = [reliability_from_distance(float(d)) for d in dist]

            seg_id = 0
            start_idx = 0
            prev_instance = mapped.loc[0, "road_instance_name"]
            prev_t = float(mapped.loc[0, "t_rel_s"])
            for i in range(1, len(mapped)):
                inst = mapped.loc[i, "road_instance_name"]
                t = float(mapped.loc[i, "t_rel_s"])
                if inst != prev_instance or (t - prev_t) > 1.5:
                    add_segment(segment_rows, subject, session, rel_path, seg_id, mapped.iloc[start_idx:i])
                    seg_id += 1
                    start_idx = i
                    prev_instance = inst
                prev_t = t
            add_segment(segment_rows, subject, session, rel_path, seg_id, mapped.iloc[start_idx:])

            reliability_counts = Counter(mapped["point_reliability"].astype(str))
            session_rows.append(
                {
                    "subject": subject,
                    "session_stamp": session,
                    "vehicle_raw_relative_path": rel_path,
                    "mapping_status": "ok",
                    "sampled_point_count": int(len(mapped)),
                    "t_start_rel_s": round(float(mapped["t_rel_s"].min()), 3),
                    "t_end_rel_s": round(float(mapped["t_rel_s"].max()), 3),
                    "median_nearest_dist_m": round(float(np.nanmedian(mapped["road_nearest_dist_m"])), 3),
                    "p90_nearest_dist_m": round(float(np.nanpercentile(mapped["road_nearest_dist_m"], 90)), 3),
                    "high_point_ratio": round(reliability_counts["high"] / len(mapped), 4),
                    "medium_or_better_point_ratio": round((reliability_counts["high"] + reliability_counts["medium"]) / len(mapped), 4),
                    "dominant_instance_name": mode_or_unknown(mapped["road_instance_name"]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            session_rows.append(
                {
                    "subject": subject,
                    "session_stamp": session,
                    "vehicle_raw_relative_path": rel_path,
                    "mapping_status": "error",
                    "mapping_error": str(exc),
                    "sampled_point_count": 0,
                }
            )

    segments = pd.DataFrame(segment_rows)
    sessions = pd.DataFrame(session_rows)
    if not segments.empty:
        segments = segments.sort_values(["subject", "session_stamp", "entry_time_rel_s", "segment_index"]).reset_index(drop=True)
    return segments, sessions


def add_segment(rows: list[dict[str, Any]], subject: str, session: str, rel_path: str, seg_id: int, seg: pd.DataFrame) -> None:
    if seg.empty:
        return
    duration = float(seg["t_rel_s"].max() - seg["t_rel_s"].min())
    if duration < 0.5:
        return
    counts = Counter(seg["point_reliability"].astype(str))
    n = len(seg)
    median_dist = float(np.nanmedian(seg["road_nearest_dist_m"]))
    high_ratio = counts["high"] / n
    medium_ratio = counts["medium"] / n
    rows.append(
        {
            "subject": subject,
            "session_stamp": session,
            "vehicle_raw_relative_path": rel_path,
            "segment_index": int(seg_id),
            "module_name": mode_or_unknown(seg["road_module_name"]),
            "instance_name": mode_or_unknown(seg["road_instance_name"]),
            "road_risk_class_cn": mode_or_unknown(seg["road_risk_class_cn"]),
            "entry_time_rel_s": round(float(seg["t_rel_s"].min()), 3),
            "exit_time_rel_s": round(float(seg["t_rel_s"].max()), 3),
            "duration_s": round(duration, 3),
            "road_s_start_m": round(float(seg["road_s"].iloc[0]), 3),
            "road_s_end_m": round(float(seg["road_s"].iloc[-1]), 3),
            "road_s_min_m": round(float(seg["road_s"].min()), 3),
            "road_s_max_m": round(float(seg["road_s"].max()), 3),
            "median_nearest_dist_m": round(median_dist, 3),
            "p90_nearest_dist_m": round(float(np.nanpercentile(seg["road_nearest_dist_m"], 90)), 3),
            "point_count": int(n),
            "high_point_ratio": round(high_ratio, 4),
            "medium_point_ratio": round(medium_ratio, 4),
            "low_point_ratio": round(counts["low"] / n, 4),
            "very_low_point_ratio": round(counts["very_low"] / n, 4),
            "segment_mapping_reliability": segment_reliability(median_dist, high_ratio, medium_ratio),
        }
    )


def nearest_event(group: pd.DataFrame, anchor: float) -> dict[str, Any]:
    if group.empty or not math.isfinite(anchor):
        return {"event_uid": "", "time_rel_s": np.nan, "delta_s": np.nan, "event_type": "", "event_level": ""}
    times = pd.to_numeric(group["anchor_time_rel_s"], errors="coerce")
    valid = times.notna()
    if not valid.any():
        return {"event_uid": "", "time_rel_s": np.nan, "delta_s": np.nan, "event_type": "", "event_level": ""}
    g = group.loc[valid].copy()
    times = pd.to_numeric(g["anchor_time_rel_s"], errors="coerce")
    idx = (times - anchor).abs().idxmin()
    row = g.loc[idx]
    time_rel = finite_float(row.get("anchor_time_rel_s"))
    return {
        "event_uid": str(row.get("event_uid", row.get("instability_event_uid", ""))),
        "time_rel_s": round(time_rel, 6),
        "delta_s": round(time_rel - anchor, 6),
        "event_type": str(row.get("event_type", row.get("instability_role", ""))),
        "event_level": str(row.get("event_level", row.get("road_guided_recommended_decision", ""))),
    }


def nearest_segment_boundary(segments: pd.DataFrame, anchor: float) -> dict[str, Any]:
    if segments.empty or not math.isfinite(anchor):
        return {
            "nearest_module_boundary_time_rel_s": np.nan,
            "nearest_module_boundary_type": "",
            "nearest_module_boundary_delta_s": np.nan,
            "active_module_name_at_old_anchor": "",
            "active_instance_name_at_old_anchor": "",
            "active_module_reliability_at_old_anchor": "",
            "active_module_risk_class_cn": "",
            "nearest_boundary_instance_name": "",
            "nearest_boundary_reliability": "",
        }
    active = segments[
        (segments["entry_time_rel_s"] <= anchor)
        & (segments["exit_time_rel_s"] >= anchor)
    ]
    if not active.empty:
        active_row = active.sort_values("duration_s", ascending=False).iloc[0]
    else:
        active_row = None

    boundary_rows: list[dict[str, Any]] = []
    for row in segments.itertuples(index=False):
        entry = finite_float(getattr(row, "entry_time_rel_s"))
        exit_t = finite_float(getattr(row, "exit_time_rel_s"))
        boundary_rows.append(
            {
                "time": entry,
                "type": "entry",
                "instance": str(getattr(row, "instance_name")),
                "module": str(getattr(row, "module_name")),
                "risk": str(getattr(row, "road_risk_class_cn")),
                "rel": str(getattr(row, "segment_mapping_reliability")),
            }
        )
        boundary_rows.append(
            {
                "time": exit_t,
                "type": "exit",
                "instance": str(getattr(row, "instance_name")),
                "module": str(getattr(row, "module_name")),
                "risk": str(getattr(row, "road_risk_class_cn")),
                "rel": str(getattr(row, "segment_mapping_reliability")),
            }
        )
    valid = [b for b in boundary_rows if math.isfinite(b["time"])]
    if valid:
        nearest = min(valid, key=lambda b: abs(b["time"] - anchor))
    else:
        nearest = {"time": np.nan, "type": "", "instance": "", "module": "", "risk": "", "rel": ""}

    return {
        "nearest_module_boundary_time_rel_s": round(float(nearest["time"]), 6) if math.isfinite(nearest["time"]) else np.nan,
        "nearest_module_boundary_type": nearest["type"],
        "nearest_module_boundary_delta_s": round(float(nearest["time"]) - anchor, 6) if math.isfinite(nearest["time"]) else np.nan,
        "active_module_name_at_old_anchor": "" if active_row is None else str(active_row["module_name"]),
        "active_instance_name_at_old_anchor": "" if active_row is None else str(active_row["instance_name"]),
        "active_module_reliability_at_old_anchor": "" if active_row is None else str(active_row["segment_mapping_reliability"]),
        "active_module_risk_class_cn": "" if active_row is None else str(active_row["road_risk_class_cn"]),
        "nearest_boundary_instance_name": nearest["instance"],
        "nearest_boundary_reliability": nearest["rel"],
    }


def build_old_anchor_alignment(candidate_events: pd.DataFrame, segments: pd.DataFrame, road_guided: pd.DataFrame) -> pd.DataFrame:
    old = candidate_events[candidate_events["anchor_source"].astype(str).eq("old_v400_context_trigger_idx")].copy()
    raw_curv = candidate_events[candidate_events["anchor_source"].astype(str).eq("raw_road_curvature_onset")].copy()
    raw_dyn_nonsteer = candidate_events[
        candidate_events["anchor_source"].astype(str).eq("raw_vehicle_dynamic_onset")
        & candidate_events["event_type"].astype(str).isin(["ay", "roll_rate"])
    ].copy()

    curv_groups = {(s, sess): g.copy() for (s, sess), g in raw_curv.groupby(["subject", "session_stamp"])}
    dyn_groups = {(s, sess): g.copy() for (s, sess), g in raw_dyn_nonsteer.groupby(["subject", "session_stamp"])}
    seg_groups = {(s, sess): g.copy() for (s, sess), g in segments.groupby(["subject", "session_stamp"])} if not segments.empty else {}
    rg_groups = {(s, sess): g.copy() for (s, sess), g in road_guided.groupby(["subject", "session_stamp"])} if not road_guided.empty else {}

    rows: list[dict[str, Any]] = []
    for row in old.itertuples(index=False):
        subject = str(getattr(row, "subject"))
        session = str(getattr(row, "session_stamp"))
        anchor = finite_float(getattr(row, "anchor_time_rel_s"))
        key = (subject, session)
        nearest_curv = nearest_event(curv_groups.get(key, pd.DataFrame()), anchor)
        nearest_dyn = nearest_event(dyn_groups.get(key, pd.DataFrame()), anchor)
        nearest_rg = nearest_event(rg_groups.get(key, pd.DataFrame()), anchor)
        boundary = nearest_segment_boundary(seg_groups.get(key, pd.DataFrame()), anchor)
        road_gap = finite_float(nearest_curv["delta_s"])
        dyn_gap = finite_float(nearest_dyn["delta_s"])
        boundary_gap = finite_float(boundary["nearest_module_boundary_delta_s"])
        rows.append(
            {
                "old_event_uid": str(getattr(row, "event_uid")),
                "subject": subject,
                "session_stamp": session,
                "old_anchor_time_rel_s": round(anchor, 6),
                "old_phase_type": str(getattr(row, "phase_type", "")),
                "old_event_level": str(getattr(row, "event_level", "")),
                "old_road_type_anchor": str(getattr(row, "road_type_anchor", "")),
                "old_trigger_type": str(getattr(row, "trigger_type", "")),
                "old_curvature_anchor": finite_float(getattr(row, "curvature_anchor", np.nan)),
                "nearest_road_curvature_event_uid": nearest_curv["event_uid"],
                "nearest_road_curvature_time_rel_s": nearest_curv["time_rel_s"],
                "nearest_road_curvature_delta_s": nearest_curv["delta_s"],
                "nearest_nonsteering_dynamic_event_uid": nearest_dyn["event_uid"],
                "nearest_nonsteering_dynamic_time_rel_s": nearest_dyn["time_rel_s"],
                "nearest_nonsteering_dynamic_delta_s": nearest_dyn["delta_s"],
                "nearest_nonsteering_dynamic_type": nearest_dyn["event_type"],
                "nearest_road_guided_event_uid": nearest_rg["event_uid"],
                "nearest_road_guided_time_rel_s": nearest_rg["time_rel_s"],
                "nearest_road_guided_delta_s": nearest_rg["delta_s"],
                **boundary,
                "within_1s_road_curvature": bool(math.isfinite(road_gap) and abs(road_gap) <= 1.0),
                "within_1s_nonsteering_dynamic": bool(math.isfinite(dyn_gap) and abs(dyn_gap) <= 1.0),
                "within_1s_module_boundary": bool(math.isfinite(boundary_gap) and abs(boundary_gap) <= 1.0),
                "old_anchor_audit_bucket": classify_old_anchor_gap(road_gap, dyn_gap, boundary_gap),
            }
        )
    return pd.DataFrame(rows)


def classify_old_anchor_gap(road_gap: float, dyn_gap: float, boundary_gap: float) -> str:
    close_dyn = math.isfinite(dyn_gap) and abs(dyn_gap) <= 1.0
    close_road = math.isfinite(road_gap) and abs(road_gap) <= 1.0
    close_boundary = math.isfinite(boundary_gap) and abs(boundary_gap) <= 1.0
    if close_dyn and close_road:
        return "old_close_to_road_and_body"
    if close_dyn:
        return "old_close_to_body_only"
    if close_road or close_boundary:
        return "old_close_to_road_only"
    if math.isfinite(dyn_gap) and dyn_gap < -1.0:
        return "old_after_body_onset"
    if math.isfinite(dyn_gap) and dyn_gap > 1.0:
        return "old_before_body_onset"
    return "old_unaligned_or_unverified"


def build_road_guided_alignment(road_guided: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    if road_guided.empty:
        return pd.DataFrame()
    seg_groups = {(s, sess): g.copy() for (s, sess), g in segments.groupby(["subject", "session_stamp"])} if not segments.empty else {}
    rows: list[dict[str, Any]] = []
    for row in road_guided.itertuples(index=False):
        subject = str(getattr(row, "subject"))
        session = str(getattr(row, "session_stamp"))
        anchor = finite_float(getattr(row, "anchor_time_rel_s"))
        boundary = nearest_segment_boundary(seg_groups.get((subject, session), pd.DataFrame()), anchor)
        rows.append(
            {
                "instability_event_uid": str(getattr(row, "instability_event_uid")),
                "subject": subject,
                "session_stamp": session,
                "anchor_time_rel_s": round(anchor, 6),
                "recommended_decision": str(getattr(row, "road_guided_recommended_decision", "")),
                "instability_role": str(getattr(row, "instability_role", "")),
                "instability_review_score": finite_float(getattr(row, "instability_review_score", np.nan)),
                "road_guided_instability_score": finite_float(getattr(row, "road_guided_instability_score", np.nan)),
                "old_v400_min_abs_anchor_gap_s": finite_float(getattr(row, "old_v400_min_abs_anchor_gap_s", np.nan)),
                "old_v400_road_type_mode": str(getattr(row, "old_v400_road_type_mode", "")),
                "road_design_instance_name": str(getattr(row, "road_design_instance_name", "")),
                "road_design_risk_class": str(getattr(row, "road_design_risk_class", "")),
                "road_design_mapping_reliability": str(getattr(row, "road_design_mapping_reliability", "")),
                "road_design_nearest_dist_m": finite_float(getattr(row, "road_design_nearest_dist_m", np.nan)),
                "peak_abs_ay_event": finite_float(getattr(row, "peak_abs_ay_event", np.nan)),
                "peak_abs_roll_rate_window": finite_float(getattr(row, "peak_abs_roll_rate_window", np.nan)),
                "peak_abs_yaw_rate_window": finite_float(getattr(row, "peak_abs_yaw_rate_window", np.nan)),
                "steering_delta_peak_post3s": finite_float(getattr(row, "steering_delta_peak_post3s", np.nan)),
                **boundary,
            }
        )
    return pd.DataFrame(rows)


def write_summary_tables(
    road_map: pd.DataFrame,
    sessions: pd.DataFrame,
    segments: pd.DataFrame,
    old_alignment: pd.DataFrame,
    rg_alignment: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"item": "道路模块数", "value": int(len(road_map)), "note": "来自道路中心线布局分组"})
    rows.append({"item": "原始车辆记录数", "value": int(len(sessions)), "note": "来自阶段1原始文件清单"})
    rows.append({"item": "可映射车辆记录数", "value": int((sessions["mapping_status"] == "ok").sum()), "note": "能读取车辆坐标并投影到道路中心线"})
    rows.append({"item": "道路模块经过片段数", "value": int(len(segments)), "note": "每条记录按最近道路模块切段"})
    rows.append({"item": "旧v400锚点数", "value": int(len(old_alignment)), "note": "旧流程事件上下文锚点"})
    rows.append({"item": "旧锚点1秒内贴近道路曲率候选", "value": int(old_alignment["within_1s_road_curvature"].sum()), "note": "道路曲率候选不是最终真值，只是道路侧参考"})
    rows.append({"item": "旧锚点1秒内贴近非方向盘车身动态", "value": int(old_alignment["within_1s_nonsteering_dynamic"].sum()), "note": "只使用ay/roll_rate，不使用方向盘变化率"})
    rows.append({"item": "旧锚点1秒内贴近道路模块边界", "value": int(old_alignment["within_1s_module_boundary"].sum()), "note": "模块进入/离开边界，受道路映射可靠性影响"})
    rows.append({"item": "道路引导候选数", "value": int(len(rg_alignment)), "note": "上一版道路引导车辆失稳候选"})
    if not rg_alignment.empty:
        accept = rg_alignment["recommended_decision"].astype(str).isin(
            ["hybrid_accept_high", "hybrid_accept_medium", "manual_confirmed_accept"]
        )
        rows.append({"item": "道路引导自动/确认采用数", "value": int(accept.sum()), "note": "当前仍是候选，不是人工真值"})
    summary = pd.DataFrame(rows)
    write_csv(summary, TABLE_DIR / "road_event_anchor_audit_summary_v0_1.csv")
    return summary


def plot_road_map(layout: pd.DataFrame, road_map: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    names = road_map["instance_name"].tolist()
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(names))))
    color_map = dict(zip(names, colors))
    for instance, group in layout.groupby("instance_name", sort=False):
        ax.plot(group["x"], group["y"], lw=2.0, color=color_map.get(instance, "gray"), label=instance)
        mid = group.iloc[len(group) // 2]
        ax.text(mid["x"], mid["y"], str(instance), fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("道路中心线模块位置图")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "road_event_position_map_v0_1.png", dpi=220)
    plt.close(fig)


def plot_reliability(sessions: pd.DataFrame, segments: pd.DataFrame, old_alignment: pd.DataFrame, rg_alignment: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    if not sessions.empty and "median_nearest_dist_m" in sessions:
        vals = pd.to_numeric(sessions["median_nearest_dist_m"], errors="coerce").dropna()
        axes[0, 0].hist(vals, bins=30, color="#4C78A8", alpha=0.85)
        axes[0, 0].set_title("每条记录道路映射中位距离")
        axes[0, 0].set_xlabel("距离 m")
        axes[0, 0].set_ylabel("记录数")

    if not segments.empty:
        counts = segments["segment_mapping_reliability"].value_counts().reindex(["high", "medium", "low", "very_low", "unmapped"]).dropna()
        axes[0, 1].bar(counts.index, counts.values, color="#59A14F")
        axes[0, 1].set_title("道路模块片段映射可靠性")
        axes[0, 1].set_ylabel("片段数")

    if not old_alignment.empty:
        for col, label, color in [
            ("nearest_nonsteering_dynamic_delta_s", "旧锚点到车身动态", "#E15759"),
            ("nearest_road_curvature_delta_s", "旧锚点到道路曲率", "#F28E2B"),
            ("nearest_module_boundary_delta_s", "旧锚点到道路模块边界", "#76B7B2"),
        ]:
            vals = pd.to_numeric(old_alignment[col], errors="coerce").dropna()
            vals = vals[vals.abs() <= 20.0]
            axes[1, 0].hist(vals, bins=60, alpha=0.45, label=label, color=color)
        axes[1, 0].axvline(0, color="black", lw=1)
        axes[1, 0].set_title("旧锚点与候选锚点时间差，限制在±20秒")
        axes[1, 0].set_xlabel("候选时间 - 旧锚点时间，秒")
        axes[1, 0].set_ylabel("数量")
        axes[1, 0].legend(fontsize=8)

    if not rg_alignment.empty:
        pivot = pd.crosstab(
            rg_alignment["road_design_mapping_reliability"],
            rg_alignment["recommended_decision"],
        )
        pivot = pivot.reindex(["high", "medium", "low", "very_low", "low_time_gap", "unmapped"]).dropna(how="all")
        pivot.plot(kind="bar", stacked=True, ax=axes[1, 1], colormap="tab20")
        axes[1, 1].set_title("道路引导候选：映射可靠性与采用建议")
        axes[1, 1].set_xlabel("道路映射可靠性")
        axes[1, 1].set_ylabel("候选数")
        axes[1, 1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "road_anchor_audit_overview_v0_1.png", dpi=220)
    plt.close(fig)


def plot_representative_panels(rg_alignment: pd.DataFrame, old_alignment: pd.DataFrame) -> list[str]:
    if rg_alignment.empty:
        return []
    candidates = select_representatives(rg_alignment)
    old_groups = {(s, sess): g.copy() for (s, sess), g in old_alignment.groupby(["subject", "session_stamp"])} if not old_alignment.empty else {}
    outputs: list[str] = []
    for _, event in candidates.iterrows():
        path = ROAD_GUIDED_EVENTS
        full = read_csv(path)
        row = full[full["instability_event_uid"].astype(str).eq(str(event["instability_event_uid"]))]
        if row.empty:
            continue
        vehicle_path = str(row.iloc[0]["vehicle_raw_absolute_path"])
        if not Path(vehicle_path).exists():
            continue
        try:
            track = load_vehicle_track(vehicle_path)
        except Exception:
            continue
        if track.empty:
            continue
        anchor = finite_float(event["anchor_time_rel_s"])
        window = track[(track["t_rel_s"] >= anchor - 8.0) & (track["t_rel_s"] <= anchor + 8.0)].copy()
        if window.empty:
            continue
        fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
        t = window["t_rel_s"] - anchor
        plot_series(axes[0], t, window, "zx|SteeringWheel", "方向盘转角", "#4C78A8")
        plot_series(axes[1], t, window, "zx|ay", "侧向加速度 ay", "#E15759")
        plot_series(axes[2], t, window, "zx|vroll", "横滚角速度", "#59A14F")
        plot_series(axes[2], t, window, "zx|vyaw", "横摆角速度", "#F28E2B", alpha=0.75)
        plot_series(axes[3], t, window, "zx1|v_km/h", "车速", "#76B7B2")
        plot_series(axes[3], t, window, "zx1|lanecurvatureXY", "道路曲率", "#B07AA1", alpha=0.75)

        for ax in axes:
            ax.axvline(0.0, color="black", lw=1.5, label="当前候选锚点")
            old_group = old_groups.get((str(event["subject"]), str(event["session_stamp"])), pd.DataFrame())
            if not old_group.empty:
                old_times = pd.to_numeric(old_group["old_anchor_time_rel_s"], errors="coerce")
                near = old_group.loc[(old_times - anchor).abs() <= 8.0]
                for old_t in pd.to_numeric(near["old_anchor_time_rel_s"], errors="coerce").dropna().head(4):
                    ax.axvline(float(old_t) - anchor, color="#999999", lw=1, ls="--")
            boundary = finite_float(event.get("nearest_module_boundary_time_rel_s", np.nan))
            if math.isfinite(boundary) and abs(boundary - anchor) <= 8.0:
                ax.axvline(boundary - anchor, color="#EDC948", lw=1.2, ls=":")
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8, loc="upper right")
        title = (
            f"{event['subject']} {event['session_stamp']} | {event['recommended_decision']} | "
            f"{event['road_design_instance_name']} | {event['road_design_mapping_reliability']}"
        )
        axes[0].set_title(title)
        axes[-1].set_xlabel("相对当前候选锚点时间，秒")
        fig.tight_layout()
        safe_uid = str(event["instability_event_uid"]).replace(":", "_").replace("/", "_")
        out = PANEL_DIR / f"{safe_uid}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        outputs.append(str(out))
    return outputs


def plot_series(ax: plt.Axes, t: pd.Series, df: pd.DataFrame, col: str, label: str, color: str, alpha: float = 1.0) -> None:
    if col not in df.columns:
        return
    vals = pd.to_numeric(df[col], errors="coerce")
    if vals.notna().sum() < 3:
        return
    ax.plot(t, vals, label=label, color=color, alpha=alpha)
    ax.set_ylabel(label)


def select_representatives(rg_alignment: pd.DataFrame) -> pd.DataFrame:
    selectors = [
        ("recommended_decision", "hybrid_accept_high"),
        ("recommended_decision", "hybrid_accept_medium"),
        ("recommended_decision", "hybrid_review_conflict_or_medium"),
        ("road_design_mapping_reliability", "high"),
        ("road_design_mapping_reliability", "very_low"),
        ("road_design_instance_name", "curve1"),
        ("road_design_instance_name", "mu1"),
        ("road_design_instance_name", "zd"),
    ]
    rows = []
    seen: set[str] = set()
    for col, value in selectors:
        if col not in rg_alignment.columns:
            continue
        subset = rg_alignment[rg_alignment[col].astype(str).eq(value)].copy()
        if subset.empty:
            continue
        subset["sort_score"] = pd.to_numeric(subset["road_guided_instability_score"], errors="coerce").fillna(-1)
        row = subset.sort_values("sort_score", ascending=False).iloc[0]
        uid = str(row["instability_event_uid"])
        if uid not in seen:
            rows.append(row)
            seen.add(uid)
    if not rows:
        return rg_alignment.head(0)
    return pd.DataFrame(rows).head(8)


def write_report(summary: pd.DataFrame, road_map: pd.DataFrame, sessions: pd.DataFrame, segments: pd.DataFrame, old_alignment: pd.DataFrame, rg_alignment: pd.DataFrame, panels: list[str]) -> None:
    old_bucket = old_alignment["old_anchor_audit_bucket"].value_counts().to_string() if not old_alignment.empty else "无"
    session_status = sessions["mapping_status"].value_counts().to_string() if not sessions.empty else "无"
    seg_rel = segments["segment_mapping_reliability"].value_counts().to_string() if not segments.empty else "无"
    rg_decision = rg_alignment["recommended_decision"].value_counts().to_string() if not rg_alignment.empty else "无"
    rg_rel = rg_alignment["road_design_mapping_reliability"].value_counts().to_string() if not rg_alignment.empty else "无"

    old_close_dyn = int(old_alignment["within_1s_nonsteering_dynamic"].sum()) if not old_alignment.empty else 0
    old_close_road = int(old_alignment["within_1s_road_curvature"].sum()) if not old_alignment.empty else 0
    old_close_boundary = int(old_alignment["within_1s_module_boundary"].sum()) if not old_alignment.empty else 0
    old_total = len(old_alignment)

    text = f"""# 道路事件位置与锚点重建审计 v0.1

生成时间：{now_str()}

## 这一步为什么做

当前旧流程模型改来改去提升有限，一个核心怀疑是：样本锚点可能不是“道路事件真正发生的时刻”，而是车辆或驾驶员已经开始响应之后的时刻。这个审计不训练模型，只检查三件事：

1. 道路设计文件里有哪些道路模块，它们在道路中心线上的位置范围是什么；
2. 每条原始车辆记录能否可靠映射到这些道路模块；
3. 旧 v400 锚点与道路曲率候选、非方向盘车身动态候选、道路模块边界之间是否对齐。

## 当前最重要结论

- 已经可以从道路中心线整理出 {len(road_map)} 个道路模块/实例。
- 原始车辆记录共 {len(sessions)} 条，其中可完成道路投影的记录数为 {int((sessions['mapping_status'] == 'ok').sum()) if not sessions.empty else 0} 条。
- 旧 v400 锚点共 {old_total} 个。
- 旧锚点 1 秒内贴近非方向盘车身动态候选的数量为 {old_close_dyn}。
- 旧锚点 1 秒内贴近道路曲率候选的数量为 {old_close_road}。
- 旧锚点 1 秒内贴近道路模块进入/离开边界的数量为 {old_close_boundary}。

这些数字说明：旧锚点不能直接当作“道路事件位置真值”。它有一部分和车身动态很近，但和道路曲率或道路模块边界的直接贴合并不充分。后续如果重新构建样本，应该优先采用“道路位置先验 + 非方向盘车身姿态确认”的锚点，而不是直接继承旧 trigger_idx。

## 道路映射质量

每条记录道路映射状态：

```text
{session_status}
```

道路模块片段映射可靠性：

```text
{seg_rel}
```

解释：

- `high` 表示车辆坐标到道路中心线距离较小，道路模块名称较可信；
- `medium` 可以作为参考，但最好结合车身姿态；
- `low` / `very_low` 说明车辆坐标和道路中心线相距较远，模块名称不能单独作为锚点依据。

## 旧锚点对齐情况

旧锚点分类：

```text
{old_bucket}
```

分类含义：

- `old_close_to_road_and_body`：旧锚点同时接近道路曲率候选和车身动态候选，可信度相对更高；
- `old_close_to_body_only`：旧锚点更像是贴近车辆已经出现动态响应的时刻；
- `old_close_to_road_only`：旧锚点更像贴近道路位置变化，但缺少车身动态支持；
- `old_after_body_onset`：车身动态候选在旧锚点之前出现，说明旧锚点可能偏晚；
- `old_before_body_onset`：旧锚点早于车身动态候选，可能是道路事件提前量，也可能是旧锚点和响应未对齐；
- `old_unaligned_or_unverified`：暂时无法用当前候选解释。

## 道路引导候选情况

道路引导候选采用建议：

```text
{rg_decision}
```

道路引导候选的道路映射可靠性：

```text
{rg_rel}
```

这部分说明：上一版道路引导候选可以作为下一步样本候选，但其中低可靠道路映射比例仍然不能忽略。正式训练前需要把候选分成自动采用、人工复核、只诊断不用训练三类。

## 这一步不能下的结论

- 不能说道路文件已经给出了每条记录的绝对真值锚点。
- 不能说旧 v400 锚点全部错误；只能说旧锚点需要按道路位置和车身姿态重新分级。
- 不能说当前道路引导候选已经是人工真值。
- 不能继续用方向盘未来变化来定义事件锚点；方向盘只能作为事件后的响应标签或后验验证。

## 建议下一步

1. 用 `old_new_anchor_alignment_v0_1.csv` 找出旧锚点明显偏晚、明显偏早、无法对齐的样本。
2. 优先抽查 `old_after_body_onset` 和 `old_unaligned_or_unverified`，确认旧流程坏样本是否集中在这些锚点风险组。
3. 对 `high/medium` 道路映射可靠性的道路引导候选，生成新的样本清单。
4. 对 `low/very_low` 道路映射样本，不直接进入正式训练，先做复核或诊断。
5. 如果用户认可，再进入“新锚点样本 manifest + 强车辆基线”阶段。

## 主要产物

- 道路模块位置表：`{TABLE_DIR / 'road_event_position_map_v0_1.csv'}`
- 每条记录道路映射摘要：`{TABLE_DIR / 'session_road_mapping_summary_v0_1.csv'}`
- 每条记录经过道路模块的时间段：`{TABLE_DIR / 'session_module_entry_exit_v0_1.csv'}`
- 旧锚点对齐表：`{TABLE_DIR / 'old_new_anchor_alignment_v0_1.csv'}`
- 道路引导候选对齐表：`{TABLE_DIR / 'road_guided_anchor_alignment_v0_1.csv'}`
- 审计汇总表：`{TABLE_DIR / 'road_event_anchor_audit_summary_v0_1.csv'}`
- 道路模块位置图：`{FIGURE_DIR / 'road_event_position_map_v0_1.png'}`
- 锚点审计概览图：`{FIGURE_DIR / 'road_anchor_audit_overview_v0_1.png'}`
- 代表样本面板目录：`{PANEL_DIR}`

代表样本图数量：{len(panels)}
"""
    (REPORT_DIR / "road_event_anchor_audit_v0_1_cn.md").write_text(text, encoding="utf-8-sig")

    user_text = f"""# 阶段 2 追加：道路事件位置与锚点审计，用户查看版

生成时间：{now_str()}

## 这一步做了什么

这一步没有训练模型，而是专门检查“事件锚点是不是可能有问题”。我把道路设计文件、原始车辆轨迹、旧 v400 事件锚点和当前道路引导失稳候选放到一起对齐。

## 当前发现

1. 道路设计文件可以整理出 {len(road_map)} 个道路模块/实例，例如弯道、低附着路面、停车/特殊路段、连接段等。
2. 原始车辆记录可以投影到道路中心线，但可靠性不完全一致。部分记录/片段距离道路中心线较远，所以不能只靠道路模块名称直接定锚点。
3. 旧 v400 锚点不能直接当作最终真值。旧锚点中，只有 {old_close_dyn} 个在 1 秒内贴近非方向盘车身动态候选，{old_close_road} 个在 1 秒内贴近道路曲率候选，{old_close_boundary} 个在 1 秒内贴近道路模块边界。
4. 这支持你的怀疑：模型效果卡住，确实可能和样本锚点定义有关。

## 怎么理解

如果旧锚点偏晚，模型训练时看到的“事件后响应”其实已经发生了一部分，模型就容易学成趋势相似但幅值、方向和物理意义不稳定。

如果旧锚点偏早，标签窗口可能还没有覆盖真正响应，模型也会变得很难学。

所以接下来比继续堆模型更重要的是：用道路位置和车身姿态重新定义一批更可信的事件锚点。

## 你可以优先看哪些文件

1. 中文报告：`{REPORT_DIR / 'road_event_anchor_audit_v0_1_cn.md'}`
2. 道路模块位置图：`{FIGURE_DIR / 'road_event_position_map_v0_1.png'}`
3. 锚点审计概览图：`{FIGURE_DIR / 'road_anchor_audit_overview_v0_1.png'}`
4. 旧锚点对齐表：`{TABLE_DIR / 'old_new_anchor_alignment_v0_1.csv'}`
5. 每条记录道路模块进入/离开时间：`{TABLE_DIR / 'session_module_entry_exit_v0_1.csv'}`
6. 代表样本图目录：`{PANEL_DIR}`

## 下一步建议

先不要训练。下一步应该用这张旧锚点对齐表，把旧样本分成：

- 锚点可信，可以保留；
- 旧锚点明显偏晚，需要重选；
- 旧锚点明显偏早，需要重选；
- 道路映射不可靠，只能人工复核或暂时不用。

然后再基于新的高可信锚点生成样本清单和强车辆基线。
"""
    (REPORT_DIR / "stage02_road_anchor_audit_user_summary_cn.md").write_text(user_text, encoding="utf-8-sig")


def write_run_log(extra: dict[str, Any]) -> None:
    (LOG_DIR / "road_event_anchor_audit_run_summary_v0_1.json").write_text(
        json.dumps(extra, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )


def main() -> None:
    configure_matplotlib_fonts()
    ensure_dirs()
    layout = load_road_layout()
    road_map = build_road_event_position_map(layout)
    write_csv(road_map, TABLE_DIR / "road_event_position_map_v0_1.csv")

    vehicle_inventory = load_vehicle_inventory()
    segments, sessions = build_session_module_entries(layout, vehicle_inventory)
    write_csv(sessions, TABLE_DIR / "session_road_mapping_summary_v0_1.csv")
    write_csv(segments, TABLE_DIR / "session_module_entry_exit_v0_1.csv")

    candidate_events = read_csv(CANDIDATE_EVENTS)
    road_guided = read_csv(ROAD_GUIDED_EVENTS) if ROAD_GUIDED_EVENTS.exists() else pd.DataFrame()
    old_alignment = build_old_anchor_alignment(candidate_events, segments, road_guided)
    write_csv(old_alignment, TABLE_DIR / "old_new_anchor_alignment_v0_1.csv")

    rg_alignment = build_road_guided_alignment(road_guided, segments)
    write_csv(rg_alignment, TABLE_DIR / "road_guided_anchor_alignment_v0_1.csv")

    summary = write_summary_tables(road_map, sessions, segments, old_alignment, rg_alignment)

    plot_road_map(layout, road_map)
    plot_reliability(sessions, segments, old_alignment, rg_alignment)
    panels = plot_representative_panels(rg_alignment, old_alignment)

    write_report(summary, road_map, sessions, segments, old_alignment, rg_alignment, panels)
    write_run_log(
        {
            "generated_at": now_str(),
            "road_layout_path": layout.attrs.get("source_path", ""),
            "vehicle_file_count": int(len(vehicle_inventory)),
            "road_module_count": int(len(road_map)),
            "session_mapping_rows": int(len(sessions)),
            "session_module_segment_rows": int(len(segments)),
            "old_anchor_alignment_rows": int(len(old_alignment)),
            "road_guided_alignment_rows": int(len(rg_alignment)),
            "representative_panel_count": int(len(panels)),
            "outputs": {
                "road_event_position_map": str(TABLE_DIR / "road_event_position_map_v0_1.csv"),
                "session_road_mapping_summary": str(TABLE_DIR / "session_road_mapping_summary_v0_1.csv"),
                "session_module_entry_exit": str(TABLE_DIR / "session_module_entry_exit_v0_1.csv"),
                "old_new_anchor_alignment": str(TABLE_DIR / "old_new_anchor_alignment_v0_1.csv"),
                "road_guided_anchor_alignment": str(TABLE_DIR / "road_guided_anchor_alignment_v0_1.csv"),
                "report": str(REPORT_DIR / "road_event_anchor_audit_v0_1_cn.md"),
                "user_summary": str(REPORT_DIR / "stage02_road_anchor_audit_user_summary_cn.md"),
            },
        }
    )


if __name__ == "__main__":
    main()
