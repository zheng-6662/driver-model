# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import build_instability_event_review_v0_1 as instability_review
import build_road_guided_instability_events_v0_1 as road_guided


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
PROJECT_ROOT = Path(r"F:/data_set_process/data_process")
RAW_VEHICLE_ROOT = PROJECT_ROOT / "01_datasets" / "数据预处理" / "原始车辆数据"

OUT_DIR = ROOT / "02_samples" / "vehicle_instability_all_raw_rescreen_v0_1"
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"

FS = 200.0
DT = 1.0 / FS
SEED_MIN_DUR_S = 0.15
SEED_MERGE_GAP_S = 0.35
EPISODE_MERGE_GAP_S = 2.5
REVIEW_PRE_S = 5.0
REVIEW_POST_S = 8.0

AY_THRESHOLD = 1.3
ROLL_RATE_THRESHOLD = 0.05

VEHICLE_COLS = [
    "StorageTime",
    "zx|SteeringWheel",
    "zx|roll",
    "zx|vroll",
    "zx|vyaw",
    "zx|ay",
    "zx1|v_km/h",
    "zx1|lateraldistance",
    "zx|lateraldistance",
    "zx1|lanecurvatureXY",
    "zx|lanecurvatureXY",
    "zx|x",
    "zx|y",
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
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


def to_seconds(storage_time: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(storage_time, errors="coerce")
    out = np.full(len(storage_time), np.nan, dtype=np.float64)
    valid = parsed.notna().to_numpy()
    if valid.any():
        ns = parsed[valid].astype("datetime64[ns]").astype("int64").to_numpy(dtype=np.float64)
        out[valid] = ns / 1e9
    return out


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def clip(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def session_from_path(path: Path) -> tuple[str, str]:
    subject = path.parent.name
    session = path.stem
    if session.startswith("Entity_Recording_"):
        session = session[len("Entity_Recording_") :]
    if session.endswith("_vehicle"):
        session = session[: -len("_vehicle")]
    return subject, session


def collapse_duplicates(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    frame = pd.DataFrame({"x": x, "y": y})
    grouped = frame.groupby("x", sort=True)["y"].mean()
    return grouped.index.to_numpy(dtype=np.float64), grouped.to_numpy(dtype=np.float64)


def interp_to_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x, y = collapse_duplicates(x, y)
    out = np.full(len(grid), np.nan, dtype=np.float64)
    if x.size == 0:
        return out
    if x.size == 1:
        nearest = int(np.nanargmin(np.abs(grid - x[0])))
        out[nearest] = y[0]
        return out
    inside = (grid >= x[0]) & (grid <= x[-1])
    out[inside] = np.interp(grid[inside], x, y)
    return out


def fill_nan_linear(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    out = arr.copy()
    idx = np.arange(len(out))
    valid = np.isfinite(out)
    if valid.all():
        return out
    if not valid.any():
        return np.zeros_like(out)
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def moving_average(arr: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return arr
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(arr, kernel, mode="same")


def find_segments(mask: np.ndarray, t: np.ndarray, min_dur_s: float, merge_gap_s: float) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    starts: list[int] = []
    ends: list[int] = []
    in_seg = False
    start = 0
    for i, flag in enumerate(mask):
        if flag and not in_seg:
            start = i
            in_seg = True
        elif not flag and in_seg:
            starts.append(start)
            ends.append(i - 1)
            in_seg = False
    if in_seg:
        starts.append(start)
        ends.append(len(mask) - 1)

    merged: list[tuple[int, int]] = []
    for a, b in zip(starts, ends):
        if float(t[b] - t[a]) < min_dur_s:
            continue
        if not merged:
            merged.append((a, b))
            continue
        prev_a, prev_b = merged[-1]
        if float(t[a] - t[prev_b]) <= merge_gap_s:
            merged[-1] = (prev_a, b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if float(t[b] - t[a]) >= min_dur_s]


def event_level_from_score(score: float) -> str:
    if score >= 5.0:
        return "extreme"
    if score >= 3.0:
        return "strong"
    if score >= 1.5:
        return "medium"
    return "weak"


def mode_or_unknown(values: list[str]) -> str:
    cleaned = [v for v in values if v and v.lower() != "nan"]
    if not cleaned:
        return "unknown"
    return Counter(cleaned).most_common(1)[0][0]


class VehicleCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.subject, self.session_stamp = session_from_path(path)
        self.vehicle_raw_relative_path = str(path.relative_to(RAW_VEHICLE_ROOT.parent)).replace("\\", "/")
        self.vehicle_raw_absolute_path = str(path)
        self.vehicle_raw_sha256 = ""
        self.read_status = "not_started"
        self.read_error = ""
        self.raw_rows = 0
        self.duration_s = float("nan")
        self.t_grid_rel_s = np.array([], dtype=np.float64)
        self.signals: dict[str, np.ndarray] = {}


def load_vehicle(path: Path) -> VehicleCache:
    cache = VehicleCache(path)
    try:
        cache.vehicle_raw_sha256 = sha256_file(path)
        df = pd.read_csv(path, usecols=lambda c: c in VEHICLE_COLS)
        cache.raw_rows = int(len(df))
        if "StorageTime" not in df.columns:
            raise ValueError("missing StorageTime")
        t_abs = to_seconds(df["StorageTime"])
        valid_t = np.isfinite(t_abs)
        if not valid_t.any():
            raise ValueError("no valid StorageTime")
        df = df.loc[valid_t].copy()
        t_rel = t_abs[valid_t] - float(t_abs[valid_t][0])
        order = np.argsort(t_rel)
        df = df.iloc[order].reset_index(drop=True)
        t_rel = t_rel[order]
        duration = float(np.nanmax(t_rel))
        if duration <= 0:
            raise ValueError("non-positive duration")
        cache.duration_s = duration
        grid = np.arange(0.0, duration + DT * 0.5, DT, dtype=np.float64)
        cache.t_grid_rel_s = grid
        for col in VEHICLE_COLS:
            if col == "StorageTime" or col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            cache.signals[col] = interp_to_grid(t_rel, values, grid)
        cache.read_status = "ok"
    except Exception as exc:  # noqa: BLE001
        cache.read_status = "error"
        cache.read_error = str(exc)
    return cache


def detect_non_steering_seeds(cache: VehicleCache) -> list[dict[str, Any]]:
    if cache.read_status != "ok" or cache.t_grid_rel_s.size < 3:
        return []
    t = cache.t_grid_rel_s
    ay = cache.signals.get("zx|ay")
    roll = cache.signals.get("zx|roll")
    vroll = cache.signals.get("zx|vroll")

    components: list[np.ndarray] = []
    component_names: list[str] = []
    ay_f = None
    if ay is not None and np.isfinite(ay).any():
        ay_f = fill_nan_linear(ay)
        components.append(np.abs(ay_f) / AY_THRESHOLD)
        component_names.append("ay")

    roll_rate = None
    if roll is not None and np.isfinite(roll).any():
        roll_f = moving_average(fill_nan_linear(roll), 5)
        roll_rate = np.gradient(roll_f, DT)
    elif vroll is not None and np.isfinite(vroll).any():
        roll_rate = fill_nan_linear(vroll)
    if roll_rate is not None and np.isfinite(roll_rate).any():
        components.append(np.abs(roll_rate) / ROLL_RATE_THRESHOLD)
        component_names.append("roll_rate")

    if not components:
        return []
    score_stack = np.vstack(components)
    score = np.nanmax(score_stack, axis=0)
    mask = score >= 1.0
    segments = find_segments(mask, t, min_dur_s=SEED_MIN_DUR_S, merge_gap_s=SEED_MERGE_GAP_S)
    seeds: list[dict[str, Any]] = []
    for idx, (a, b) in enumerate(segments, start=1):
        seg_scores = score_stack[:, a : b + 1]
        component_peak_scores = np.nanmax(seg_scores, axis=1)
        main_idx = int(np.nanargmax(component_peak_scores))
        main_component = component_names[main_idx]
        peak_local = int(np.nanargmax(score[a : b + 1]))
        peak_i = a + peak_local

        ay_peak = float(np.nanmax(np.abs(ay_f[a : b + 1]))) if ay_f is not None else float("nan")
        roll_peak = float(np.nanmax(np.abs(roll_rate[a : b + 1]))) if roll_rate is not None else float("nan")
        main_value = ay_peak if main_component == "ay" else roll_peak
        peak_score = float(score[peak_i])
        seeds.append(
            {
                "seed_uid": f"allraw_nonsteering_seed__{cache.subject}__{cache.session_stamp}__{idx:05d}",
                "event_index": idx,
                "event_start_rel_s": float(t[a]),
                "event_end_rel_s": float(t[b]),
                "anchor_time_rel_s": float(t[a]),
                "peak_time_rel_s": float(t[peak_i]),
                "main_component": main_component,
                "event_level": event_level_from_score(peak_score),
                "peak_score": peak_score,
                "anchor_value": main_value,
                "ay_peak_abs": ay_peak,
                "roll_rate_peak_abs": roll_peak,
                "ay_triggered": bool(math.isfinite(ay_peak) and ay_peak >= AY_THRESHOLD),
                "roll_rate_triggered": bool(math.isfinite(roll_peak) and roll_peak >= ROLL_RATE_THRESHOLD),
            }
        )
    return seeds


def merge_seeds(seeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not seeds:
        return []
    seeds = sorted(seeds, key=lambda r: (float(r["event_start_rel_s"]), float(r["anchor_time_rel_s"])))
    episodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for seed in seeds:
        start = float(seed["event_start_rel_s"])
        end = float(seed["event_end_rel_s"])
        if current is None or start > float(current["event_end_rel_s"]) + EPISODE_MERGE_GAP_S:
            if current is not None:
                episodes.append(current)
            current = {
                "event_start_rel_s": start,
                "event_end_rel_s": end,
                "anchor_time_rel_s": float(seed["anchor_time_rel_s"]),
                "seed_uids": [str(seed["seed_uid"])],
                "seed_components": [str(seed["main_component"])],
                "seed_event_levels": [str(seed["event_level"])],
                "seed_peak_scores": [finite_float(seed["peak_score"])],
                "seed_anchor_values": [finite_float(seed["anchor_value"])],
                "ay_seed_count": int(bool(seed["ay_triggered"])),
                "roll_rate_seed_count": int(bool(seed["roll_rate_triggered"])),
            }
            continue
        current["event_end_rel_s"] = max(float(current["event_end_rel_s"]), end)
        current["seed_uids"].append(str(seed["seed_uid"]))
        current["seed_components"].append(str(seed["main_component"]))
        current["seed_event_levels"].append(str(seed["event_level"]))
        current["seed_peak_scores"].append(finite_float(seed["peak_score"]))
        current["seed_anchor_values"].append(finite_float(seed["anchor_value"]))
        current["ay_seed_count"] += int(bool(seed["ay_triggered"]))
        current["roll_rate_seed_count"] += int(bool(seed["roll_rate_triggered"]))
    if current is not None:
        episodes.append(current)
    return episodes


def window_mask(t: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    return (t >= start_s) & (t <= end_s)


def values(cache: VehicleCache, col: str, mask: np.ndarray) -> np.ndarray:
    arr = cache.signals.get(col)
    if arr is None:
        return np.array([], dtype=np.float64)
    return arr[mask]


def choose_signal(cache: VehicleCache, names: list[str]) -> str | None:
    for name in names:
        arr = cache.signals.get(name)
        if arr is not None and np.isfinite(arr).any():
            return name
    return None


def compute_episode_metrics(cache: VehicleCache, episode: dict[str, Any]) -> dict[str, Any]:
    t = cache.t_grid_rel_s
    anchor = float(episode["anchor_time_rel_s"])
    start = float(episode["event_start_rel_s"])
    end = float(episode["event_end_rel_s"])
    review_mask = window_mask(t, anchor - REVIEW_PRE_S, anchor + REVIEW_POST_S)
    event_mask = window_mask(t, start, end)
    pre_mask = window_mask(t, anchor - 1.0, anchor)
    post3_mask = window_mask(t, anchor, anchor + 3.0)

    ay_col = choose_signal(cache, ["zx|ay"])
    yaw_col = choose_signal(cache, ["zx|vyaw"])
    steering_col = choose_signal(cache, ["zx|SteeringWheel"])
    speed_col = choose_signal(cache, ["zx1|v_km/h"])
    lateral_col = choose_signal(cache, ["zx1|lateraldistance", "zx|lateraldistance"])
    curvature_col = choose_signal(cache, ["zx1|lanecurvatureXY", "zx|lanecurvatureXY"])

    roll_arr = cache.signals.get("zx|roll")
    if roll_arr is not None and np.isfinite(roll_arr).any():
        roll_rate = np.gradient(moving_average(fill_nan_linear(roll_arr), 5), DT)
    else:
        vroll = cache.signals.get("zx|vroll")
        roll_rate = fill_nan_linear(vroll) if vroll is not None and np.isfinite(vroll).any() else np.full_like(t, np.nan)

    result: dict[str, Any] = {
        "review_point_count": int(review_mask.sum()),
        "event_point_count": int(event_mask.sum()),
        "peak_abs_ay_window": round(
            instability_review.max_abs(values(cache, ay_col, review_mask)) if ay_col else np.nan,
            6,
        ),
        "signed_peak_ay_window": round(
            instability_review.signed_at_abs_max(values(cache, ay_col, review_mask)) if ay_col else np.nan,
            6,
        ),
        "peak_abs_ay_event": round(
            instability_review.max_abs(values(cache, ay_col, event_mask)) if ay_col else np.nan,
            6,
        ),
        "peak_abs_roll_rate_window": round(instability_review.max_abs(roll_rate[review_mask]), 6),
        "peak_abs_yaw_rate_window": round(
            instability_review.max_abs(values(cache, yaw_col, review_mask)) if yaw_col else np.nan,
            6,
        ),
        "lateral_distance_range_window": round(
            instability_review.robust_range(values(cache, lateral_col, review_mask)) if lateral_col else np.nan,
            6,
        ),
        "peak_abs_curvature_window": round(
            instability_review.max_abs(values(cache, curvature_col, review_mask)) if curvature_col else np.nan,
            8,
        ),
        "median_speed_kmh_window": round(
            float(np.nanmedian(values(cache, speed_col, review_mask))) if speed_col and review_mask.any() else np.nan,
            6,
        ),
    }

    if steering_col and pre_mask.any() and post3_mask.any():
        pre_vals = values(cache, steering_col, pre_mask)
        post_vals = values(cache, steering_col, post3_mask)
        if np.isfinite(pre_vals).any() and np.isfinite(post_vals).any():
            baseline = float(np.nanmedian(pre_vals))
            delta = post_vals - baseline
            result["steering_baseline_pre1s"] = round(baseline, 6)
            result["steering_delta_peak_post3s"] = round(instability_review.max_abs(delta), 6)
            result["steering_signed_delta_peak_post3s"] = round(instability_review.signed_at_abs_max(delta), 6)
        else:
            result["steering_baseline_pre1s"] = np.nan
            result["steering_delta_peak_post3s"] = np.nan
            result["steering_signed_delta_peak_post3s"] = np.nan
    else:
        result["steering_baseline_pre1s"] = np.nan
        result["steering_delta_peak_post3s"] = np.nan
        result["steering_signed_delta_peak_post3s"] = np.nan
    return result


def classify_role(episode: dict[str, Any]) -> str:
    has_ay = int(episode.get("ay_seed_count", 0) or 0) > 0
    has_roll = int(episode.get("roll_rate_seed_count", 0) or 0) > 0
    if has_ay and has_roll:
        return "instability_ay_roll"
    if has_roll:
        return "instability_roll_only"
    return "instability_ay_only"


def manual_context_nearest(row: dict[str, Any], labels: pd.DataFrame) -> dict[str, Any]:
    if labels.empty:
        return {}
    uid = str(row["instability_event_uid"])
    exact = labels[labels["selected_candidate_event_uid"].astype(str).eq(uid)]
    if not exact.empty:
        return {}
    subject = str(row["subject"])
    session = str(row["session_stamp"])
    anchor = finite_float(row["anchor_time_rel_s"])
    same = labels[(labels["subject"].astype(str) == subject) & (labels["session_stamp"].astype(str) == session)].copy()
    if same.empty or not math.isfinite(anchor):
        return {}
    same["anchor_gap"] = (pd.to_numeric(same["anchor_rel_s"], errors="coerce") - anchor).abs()
    same = same[same["anchor_gap"] <= 1.0].sort_values("anchor_gap")
    if same.empty:
        return {}
    latest = same.iloc[0]
    return {
        "manual_label_count": 1,
        "manual_label_decision_mode": str(latest.get("decision", "nearest_manual_within_1s")),
        "manual_label_anchor_rel_s": round(finite_float(latest.get("anchor_rel_s")), 6),
        "manual_label_start_rel_s": round(finite_float(latest.get("event_start_rel_s")), 6),
        "manual_label_end_rel_s": round(finite_float(latest.get("event_end_rel_s")), 6),
        "manual_label_confidence_max": round(finite_float(latest.get("confidence_1_5")), 6),
        "manual_label_match_type": "nearest_within_1s",
    }


def build_episode_row(cache: VehicleCache, episode: dict[str, Any], index: int) -> dict[str, Any]:
    anchor = float(episode["anchor_time_rel_s"])
    peak_scores = [v for v in episode["seed_peak_scores"] if math.isfinite(v)]
    anchor_values = [v for v in episode["seed_anchor_values"] if math.isfinite(v)]
    row: dict[str, Any] = {
        "instability_event_uid": f"vehicle_instability_allraw__{cache.subject}__{cache.session_stamp}__{int(round(anchor * 1000)):09d}",
        "dataset_candidate_version": "vehicle_instability_all_raw_rescreen_v0_1",
        "subject": cache.subject,
        "session_stamp": cache.session_stamp,
        "vehicle_raw_relative_path": cache.vehicle_raw_relative_path,
        "vehicle_raw_absolute_path": cache.vehicle_raw_absolute_path,
        "vehicle_raw_sha256": cache.vehicle_raw_sha256,
        "anchor_time_rel_s": round(anchor, 6),
        "event_start_rel_s": round(float(episode["event_start_rel_s"]), 6),
        "event_end_rel_s": round(float(episode["event_end_rel_s"]), 6),
        "event_duration_s": round(max(0.0, float(episode["event_end_rel_s"]) - float(episode["event_start_rel_s"])), 6),
        "instability_anchor_source": "all_raw_vehicle_dynamic_onset_non_steering",
        "instability_role": classify_role(episode),
        "ay_seed_count": int(episode.get("ay_seed_count", 0) or 0),
        "roll_rate_seed_count": int(episode.get("roll_rate_seed_count", 0) or 0),
        "merged_seed_count": len(episode["seed_uids"]),
        "source_event_uids": ";".join(episode["seed_uids"]),
        "source_event_types": ";".join(episode["seed_components"]),
        "source_event_levels": ";".join(episode["seed_event_levels"]),
        "max_source_anchor_value": round(max(anchor_values), 6) if anchor_values else np.nan,
        "max_source_peak_score": round(max(peak_scores), 6) if peak_scores else np.nan,
        "vehicle_read_status": cache.read_status,
        "vehicle_read_error": cache.read_error,
        "event_index_in_session": index,
    }
    row.update(compute_episode_metrics(cache, episode))
    score, decision = instability_review.score_episode(row)
    row["instability_review_score"] = score
    row["codex_recommended_decision"] = decision
    row["causal_setting"] = "all_raw_detected_vehicle_instability_onset_predict_future_steering_response"
    row["leakage_note"] = (
        "Full raw rescreen. Anchor is derived from non-steering vehicle dynamics (ay/roll_rate). "
        "Steering metrics are only response evidence and must not be used to define onset."
    )
    return row


def build() -> dict[str, Any]:
    ensure_dirs()
    raw_files = sorted(RAW_VEHICLE_ROOT.glob("*/*.csv"))
    old = road_guided.prepare_old_events()
    old_groups = {
        (str(subject), str(session)): group.copy()
        for (subject, session), group in old.groupby(["subject", "session_stamp"], sort=False)
    }
    labels = road_guided.load_manual_labels()
    manual_groups = {
        str(uid): group.copy()
        for uid, group in labels.groupby("selected_candidate_event_uid", sort=False)
    } if not labels.empty else {}
    tree, layout, mapper_status = road_guided.load_road_mapper()
    road_vehicle_cache: dict[str, pd.DataFrame] = {}

    all_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for path in raw_files:
        cache = load_vehicle(path)
        seeds = detect_non_steering_seeds(cache)
        episodes = merge_seeds(seeds)
        status_rows.append(
            {
                "subject": cache.subject,
                "session_stamp": cache.session_stamp,
                "vehicle_raw_relative_path": cache.vehicle_raw_relative_path,
                "vehicle_raw_absolute_path": cache.vehicle_raw_absolute_path,
                "vehicle_raw_sha256": cache.vehicle_raw_sha256,
                "read_status": cache.read_status,
                "read_error": cache.read_error,
                "raw_rows": cache.raw_rows,
                "duration_s": round(cache.duration_s, 6) if math.isfinite(cache.duration_s) else np.nan,
                "non_steering_seed_count": len(seeds),
                "merged_episode_count": len(episodes),
            }
        )
        if cache.read_status != "ok":
            continue
        for idx, episode in enumerate(episodes, start=1):
            row = build_episode_row(cache, episode, idx)
            series = pd.Series(row)
            row.update(road_guided.summarize_old_context(series, old_groups))
            row.update(road_guided.map_road_context(series, tree, layout, mapper_status, road_vehicle_cache))
            manual = road_guided.manual_context(series, manual_groups)
            if int(manual.get("manual_label_count", 0) or 0) == 0:
                manual.update(manual_context_nearest(row, labels))
                if manual:
                    manual.setdefault("manual_label_match_type", "nearest_or_exact")
            else:
                manual["manual_label_match_type"] = "exact_selected_uid"
            row.update(manual)
            score, decision, reasons = road_guided.score_hybrid(row)
            row["road_guided_instability_score"] = score
            row["road_guided_recommended_decision"] = decision
            row["road_guided_decision_reasons"] = reasons
            row["dataset_candidate_version"] = "vehicle_instability_all_raw_rescreen_v0_1"
            all_rows.append(row)

    out = pd.DataFrame(all_rows)
    status = pd.DataFrame(status_rows)

    full_path = TABLE_DIR / "all_raw_vehicle_instability_candidates_v0_1.csv"
    accepted_path = TABLE_DIR / "all_raw_vehicle_instability_auto_accepted_v0_1.csv"
    primary_path = TABLE_DIR / "all_raw_vehicle_instability_primary_high_confidence_v0_1.csv"
    review_path = TABLE_DIR / "all_raw_vehicle_instability_review_queue_v0_1.csv"
    rejected_path = TABLE_DIR / "all_raw_vehicle_instability_rejected_v0_1.csv"
    summary_path = TABLE_DIR / "all_raw_vehicle_instability_summary_v0_1.csv"
    file_status_path = TABLE_DIR / "all_raw_vehicle_rescreen_file_status_v0_1.csv"
    module_path = TABLE_DIR / "all_raw_vehicle_instability_module_summary_v0_1.csv"

    accepted_mask = out["road_guided_recommended_decision"].isin(
        ["hybrid_accept_high", "hybrid_accept_medium", "manual_confirmed_accept"]
    ) if not out.empty else pd.Series([], dtype=bool)
    primary_mask = out["road_guided_recommended_decision"].isin(
        ["hybrid_accept_high", "manual_confirmed_accept"]
    ) if not out.empty else pd.Series([], dtype=bool)
    review_mask = out["road_guided_recommended_decision"].eq("hybrid_review_conflict_or_medium") if not out.empty else pd.Series([], dtype=bool)

    out.to_csv(full_path, index=False, encoding="utf-8-sig")
    out.loc[accepted_mask].to_csv(accepted_path, index=False, encoding="utf-8-sig")
    out.loc[primary_mask].to_csv(primary_path, index=False, encoding="utf-8-sig")
    out.loc[review_mask].to_csv(review_path, index=False, encoding="utf-8-sig")
    out.loc[~accepted_mask & ~review_mask].to_csv(rejected_path, index=False, encoding="utf-8-sig")
    status.to_csv(file_status_path, index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, Any]] = [
        {"summary_type": "total", "key": "raw_vehicle_csv_files", "value": int(len(raw_files))},
        {"summary_type": "total", "key": "read_ok_files", "value": int((status["read_status"] == "ok").sum()) if not status.empty else 0},
        {"summary_type": "total", "key": "all_candidates", "value": int(len(out))},
        {"summary_type": "total", "key": "accepted_candidates", "value": int(accepted_mask.sum()) if len(out) else 0},
        {"summary_type": "total", "key": "primary_high_confidence_candidates", "value": int(primary_mask.sum()) if len(out) else 0},
        {"summary_type": "total", "key": "review_candidates", "value": int(review_mask.sum()) if len(out) else 0},
    ]
    if not out.empty:
        for key, value in out["road_guided_recommended_decision"].value_counts().items():
            summary_rows.append({"summary_type": "decision_count", "key": str(key), "value": int(value)})
        for key, value in out["road_design_risk_class"].value_counts().items():
            summary_rows.append({"summary_type": "road_design_risk_count", "key": str(key), "value": int(value)})
        for key, value in out["subject"].value_counts().sort_index().items():
            summary_rows.append({"summary_type": "subject_candidate_count", "key": str(key), "value": int(value)})
        module_summary = (
            out.groupby(["road_design_mapping_reliability", "road_design_risk_class", "road_design_instance_name", "road_guided_recommended_decision"], dropna=False)
            .size()
            .reset_index(name="count")
        )
    else:
        module_summary = pd.DataFrame()
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    module_summary.to_csv(module_path, index=False, encoding="utf-8-sig")

    report_path = REPORT_DIR / "all_raw_vehicle_instability_rescreen_v0_1_cn.md"
    card_path = REPORT_DIR / "dataset_version_card_all_raw_vehicle_instability_rescreen_v0_1_cn.md"
    write_report(
        report_path,
        card_path,
        out,
        status,
        full_path,
        accepted_path,
        primary_path,
        review_path,
        rejected_path,
        summary_path,
        file_status_path,
        module_path,
    )

    run_summary = {
        "generated_at": now_str(),
        "raw_vehicle_csv_files": int(len(raw_files)),
        "read_ok_files": int((status["read_status"] == "ok").sum()) if not status.empty else 0,
        "all_candidates": int(len(out)),
        "accepted_candidates": int(accepted_mask.sum()) if len(out) else 0,
        "primary_high_confidence_candidates": int(primary_mask.sum()) if len(out) else 0,
        "review_candidates": int(review_mask.sum()) if len(out) else 0,
        "decision_counts": {str(k): int(v) for k, v in out["road_guided_recommended_decision"].value_counts().items()} if not out.empty else {},
        "outputs": {
            "full": str(full_path).replace("\\", "/"),
            "accepted": str(accepted_path).replace("\\", "/"),
            "primary_high_confidence": str(primary_path).replace("\\", "/"),
            "review": str(review_path).replace("\\", "/"),
            "rejected": str(rejected_path).replace("\\", "/"),
            "summary": str(summary_path).replace("\\", "/"),
            "file_status": str(file_status_path).replace("\\", "/"),
            "report": str(report_path).replace("\\", "/"),
            "dataset_card": str(card_path).replace("\\", "/"),
        },
    }
    (LOG_DIR / "all_raw_vehicle_instability_rescreen_run_summary_v0_1.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    return run_summary


def write_report(
    report_path: Path,
    card_path: Path,
    out: pd.DataFrame,
    status: pd.DataFrame,
    full_path: Path,
    accepted_path: Path,
    primary_path: Path,
    review_path: Path,
    rejected_path: Path,
    summary_path: Path,
    file_status_path: Path,
    module_path: Path,
) -> None:
    decision_counts = out["road_guided_recommended_decision"].value_counts().to_string() if not out.empty else "none"
    subject_counts = out.groupby("subject").size().sort_index().to_string() if not out.empty else "none"
    file_counts = status["read_status"].value_counts().to_string() if not status.empty else "none"
    accepted_count = int(out["road_guided_recommended_decision"].isin(["hybrid_accept_high", "hybrid_accept_medium", "manual_confirmed_accept"]).sum()) if not out.empty else 0
    primary_count = int(out["road_guided_recommended_decision"].isin(["hybrid_accept_high", "manual_confirmed_accept"]).sum()) if not out.empty else 0
    review_count = int(out["road_guided_recommended_decision"].eq("hybrid_review_conflict_or_medium").sum()) if not out.empty else 0
    rejected_count = int(len(out) - accepted_count - review_count)
    full_path_s = str(full_path).replace("\\", "/")
    accepted_path_s = str(accepted_path).replace("\\", "/")
    primary_path_s = str(primary_path).replace("\\", "/")
    review_path_s = str(review_path).replace("\\", "/")
    rejected_path_s = str(rejected_path).replace("\\", "/")
    summary_path_s = str(summary_path).replace("\\", "/")
    file_status_path_s = str(file_status_path).replace("\\", "/")
    module_path_s = str(module_path).replace("\\", "/")
    card_path_s = str(card_path).replace("\\", "/")

    report_text = f"""# 全部原始车辆数据失稳样本重筛 v0.1

生成时间：{now_str()}

## 为什么做

用户希望按“道路设定引导 + 原始车辆动态证据”的标准，对所有原始数据重新筛选样本，而不是只在已有候选表上继续人工标注。

本版本直接从 `F:/data_set_process/data_process/01_datasets/数据预处理/原始车辆数据/<被试名>/*.csv` 读取 91 个原始车辆 CSV，重新扫描 `ay` 和 `roll_rate` 非方向盘动态异常，再叠加旧 v400 事件上下文和道路模块先验进行判定。

## 筛选原则

- 主锚点只来自非方向盘车辆动态：`ay` 和 `roll_rate`。
- `steer_rate` 不作为失稳锚点，方向盘只作为事件后响应证据。
- 弯道只作为上下文，不等于失稳。
- `mu1/differentmu_road`、`fix_road`、`stop`、`zd` 等道路模块只作为场景先验，不能单独确认失稳。
- 旧 `events_v400_context` 只作为旧事件上下文，不作为新真值。

## 当前结果

原始车辆 CSV 数：{len(status)}

可读取车辆 CSV 数：{int((status["read_status"] == "ok").sum()) if not status.empty else 0}

重筛候选总数：{len(out)}

自动/已确认采用：{accepted_count}

高置信主清单：{primary_count}

中间复核：{review_count}

低证据剔除：{rejected_count}

按最终建议统计：

```text
{decision_counts}
```

按被试候选数统计：

```text
{subject_counts}
```

文件读取状态：

```text
{file_counts}
```

## 输出

- 全量候选：`{full_path_s}`
- 自动采用：`{accepted_path_s}`
- 高置信主清单：`{primary_path_s}`
- 中间复核：`{review_path_s}`
- 低证据剔除：`{rejected_path_s}`
- 汇总表：`{summary_path_s}`
- 文件状态：`{file_status_path_s}`
- 道路模块交叉表：`{module_path_s}`
- 数据版本卡：`{card_path_s}`

## 下一步

下一步应该用自动采用表生成正式车辆失稳版 `samples_master` 和处理后车辆窗口。之前的 404 个弯道样本、以及旧的道路曲率阶段 3 模型，仍然只能作为历史诊断材料。
"""
    report_path.write_text(report_text, encoding="utf-8")

    card_text = f"""# 数据版本卡：vehicle_instability_all_raw_rescreen_v0_1

生成时间：{now_str()}

## 数据版本定位

这是从所有原始车辆 CSV 直接重筛得到的车辆失稳候选版本。它替代了只基于旧候选表的筛选方式，覆盖 `原始车辆数据/<被试名>/*.csv` 下 91 个原始车辆文件。

## 事件定义

车辆失稳候选事件由非方向盘车辆动态异常触发：

- `|ay| >= {AY_THRESHOLD}`
- `|roll_rate| >= {ROLL_RATE_THRESHOLD}`

相邻动态种子先按 {SEED_MERGE_GAP_S} 秒合并，再按 {EPISODE_MERGE_GAP_S} 秒合并为候选事件片段。

## 证据融合

每个候选事件会补充：

- 横向加速度、横滚速率、横摆角速度、横向偏移、车速、事件后方向盘响应；
- 旧 v400 事件上下文；
- 道路中心线模块和映射可靠度；
- 已有 31 条键盘标注的精确或近邻校准。

## 数量

- 原始车辆 CSV：{len(status)}
- 可读取 CSV：{int((status["read_status"] == "ok").sum()) if not status.empty else 0}
- 候选总数：{len(out)}
- 自动/已确认采用：{accepted_count}
- 高置信主清单：{primary_count}
- 中间复核：{review_count}
- 低证据剔除：{rejected_count}

## 不能下的结论

- 不能把自动采用事件称为完全人工真值。
- 不能把道路模块本身称为失稳原因。
- 不能用本版本证明连续风格或生理有效。
- 不能继续使用旧 404 个弯道样本作为车辆失稳主样本。

## 推荐使用

建议下一步先用 `{primary_path_s}` 作为保守主清单生成车辆失稳样本 manifest；`{accepted_path_s}` 可作为扩展清单。正式训练前必须重新生成处理后车辆窗口和 split 表。
"""
    card_path.write_text(card_text, encoding="utf-8")


if __name__ == "__main__":
    build()
