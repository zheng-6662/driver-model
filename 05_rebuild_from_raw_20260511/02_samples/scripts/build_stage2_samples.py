# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(r"F:/data_set_process/data_process")
REBUILD_ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
RAW_ROOT = PROJECT_ROOT / "01_datasets" / "数据预处理"
OLD_PROCESSED_ROOT = PROJECT_ROOT / "01_datasets" / "多模态数据" / "被试数据集合"
ROAD_DESIGN_ROOT = OLD_PROCESSED_ROOT / "道路信息"

AUDIT_TABLE_DIR = REBUILD_ROOT / "01_audit" / "tables"
OUT_DIR = REBUILD_ROOT / "02_samples"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = REBUILD_ROOT / "09_reports"

FS = 200.0

VEHICLE_USECOLS = {
    "StorageTime",
    "zx|SteeringWheel",
    "zx|roll",
    "zx|vroll",
    "zx|vyaw",
    "zx|ay",
    "zx|vx",
    "zx|vy",
    "zx1|v_km/h",
    "zx1|lanecurvatureXY",
    "zx|lanecurvatureXY",
    "zx1|lateraldistance",
    "zx|lateraldistance",
    "zx1|pointdistance",
    "zx1|pointdistance9",
    "zx1|distance7",
    "zx1|distance8",
    "zx|x",
    "zx|y",
}

WINDOW_CONFIGS = [
    {
        "window_config_id": "pre1_label2_event_trigger",
        "input_start_rel_s": -1.0,
        "input_end_rel_s": 0.0,
        "label_start_rel_s": 0.0,
        "label_end_rel_s": 2.0,
        "causal_setting": "event_trigger_predict_full_future",
        "window_note": "1s event-pre input, 2s future label",
    },
    {
        "window_config_id": "pre2_label2_old_main",
        "input_start_rel_s": -2.0,
        "input_end_rel_s": 0.0,
        "label_start_rel_s": 0.0,
        "label_end_rel_s": 2.0,
        "causal_setting": "event_trigger_predict_full_future",
        "window_note": "2s event-pre input, old-main comparable 2s future label",
    },
    {
        "window_config_id": "pre3_label3_response_coverage",
        "input_start_rel_s": -3.0,
        "input_end_rel_s": 0.0,
        "label_start_rel_s": 0.0,
        "label_end_rel_s": 3.0,
        "causal_setting": "event_trigger_predict_full_future_longer_context",
        "window_note": "3s pre input and 3s future label for response-coverage audit",
    },
    {
        "window_config_id": "pre2_obs0p5_label2_early_observe",
        "input_start_rel_s": -2.0,
        "input_end_rel_s": 0.5,
        "label_start_rel_s": 0.5,
        "label_end_rel_s": 2.5,
        "causal_setting": "early_observation_predict_remaining_response",
        "window_note": "contains 0.5s post-anchor observation; not comparable to pure event-trigger prediction",
    },
]


@dataclass
class VehicleCache:
    subject: str
    session_stamp: str
    raw_relative_path: str
    raw_absolute_path: str
    sha256: str
    t0_abs_s: float
    t_grid_rel_s: np.ndarray
    signals: dict[str, np.ndarray]
    read_status: str
    read_error: str = ""


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def stable_bucket(key: str) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def split_from_key(key: str) -> str:
    bucket = stable_bucket(key)
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def parse_session_stamp(name: str) -> str | None:
    m = re.search(r"Entity_Recording_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", name)
    return m.group(1) if m else None


def safe_rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def to_seconds(storage_time: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(storage_time, errors="coerce")
    out = np.full(len(storage_time), np.nan, dtype=np.float64)
    valid = parsed.notna().to_numpy()
    if valid.any():
        # Force ns before integer conversion. Pandas may infer datetime64[us]
        # for these CSV strings, which would shrink all durations by 1000x.
        ns = parsed[valid].to_numpy(dtype="datetime64[ns]").astype("int64")
        out[valid] = ns.astype(np.float64) / 1e9
    return out


def collapse_duplicates(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    uniq, inv = np.unique(x, return_inverse=True)
    sums = np.bincount(inv, weights=y)
    counts = np.bincount(inv)
    return uniq, sums / np.maximum(counts, 1)


def interp_to_grid(x_rel: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ux, uy = collapse_duplicates(x_rel, y)
    if len(ux) < 2:
        return np.full_like(grid, np.nan, dtype=np.float64)
    return np.interp(grid, ux, uy, left=np.nan, right=np.nan).astype(np.float64)


def fill_nan_linear(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float64, copy=True)
    idx = np.arange(len(out), dtype=np.float64)
    finite = np.isfinite(out)
    if finite.sum() < 2:
        return np.nan_to_num(out, nan=0.0)
    out[~finite] = np.interp(idx[~finite], idx[finite], out[finite])
    return out


def moving_average(arr: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(arr) < width:
        return arr
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(arr, kernel, mode="same")


def read_vehicle_cache(row: pd.Series) -> VehicleCache:
    path = Path(row["absolute_path"])
    try:
        df = pd.read_csv(path, usecols=lambda c: c in VEHICLE_USECOLS)
        if "StorageTime" not in df.columns:
            raise ValueError("missing StorageTime")
        t_abs = to_seconds(df["StorageTime"])
        if not np.isfinite(t_abs).any():
            raise ValueError("no valid StorageTime")
        t0 = float(np.nanmin(t_abs))
        duration = float(np.nanmax(t_abs) - t0)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("non-positive duration")
        grid = np.arange(0.0, duration + 0.5 / FS, 1.0 / FS, dtype=np.float64)
        x_rel = t_abs - t0
        signals: dict[str, np.ndarray] = {}
        for col in df.columns:
            if col == "StorageTime":
                continue
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            signals[col] = interp_to_grid(x_rel, values, grid)
        return VehicleCache(
            subject=str(row["subject"]),
            session_stamp=str(row["session_stamp"]),
            raw_relative_path=str(row["relative_path"]),
            raw_absolute_path=str(row["absolute_path"]),
            sha256=str(row.get("sha256", "")),
            t0_abs_s=t0,
            t_grid_rel_s=grid,
            signals=signals,
            read_status="ok",
        )
    except Exception as exc:
        return VehicleCache(
            subject=str(row["subject"]),
            session_stamp=str(row["session_stamp"]),
            raw_relative_path=str(row["relative_path"]),
            raw_absolute_path=str(row["absolute_path"]),
            sha256=str(row.get("sha256", "")),
            t0_abs_s=float("nan"),
            t_grid_rel_s=np.array([], dtype=np.float64),
            signals={},
            read_status="error",
            read_error=repr(exc),
        )


def scan_road_design_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not ROAD_DESIGN_ROOT.exists():
        return pd.DataFrame()
    for path in sorted(ROAD_DESIGN_ROOT.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        row: dict[str, Any] = {
            "relative_path": safe_rel(path, PROJECT_ROOT),
            "suffix": suffix,
            "size_bytes": int(path.stat().st_size),
            "mtime": pd.Timestamp.fromtimestamp(path.stat().st_mtime).isoformat(),
            "read_status": "not_read",
            "row_count": np.nan,
            "columns": "",
            "s_min": np.nan,
            "s_max": np.nan,
            "curvature_col": "",
            "curvature_nonzero_count": np.nan,
            "module_names": "",
        }
        if suffix == ".csv":
            try:
                df = pd.read_csv(path, nrows=200000)
                row["read_status"] = "ok"
                row["row_count"] = int(len(df))
                row["columns"] = " | ".join(map(str, df.columns[:50]))
                if "s" in df.columns:
                    s = pd.to_numeric(df["s"], errors="coerce")
                    row["s_min"] = float(s.min())
                    row["s_max"] = float(s.max())
                for col in ["curvature", "kappa", "curvature_1pm"]:
                    if col in df.columns:
                        curv = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                        row["curvature_col"] = col
                        row["curvature_nonzero_count"] = int((np.abs(curv) > 1e-8).sum())
                        break
                if "module_name" in df.columns:
                    names = sorted(map(str, df["module_name"].dropna().unique().tolist()))
                    row["module_names"] = " | ".join(names[:30])
            except Exception as exc:
                row["read_status"] = f"error: {exc!r}"
        rows.append(row)
    return pd.DataFrame(rows)


def find_segments(mask: np.ndarray, t: np.ndarray, min_dur_s: float, merge_gap_s: float) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return []
    starts: list[int] = []
    ends: list[int] = []
    in_seg = False
    s = 0
    for i, flag in enumerate(mask):
        if flag and not in_seg:
            s = i
            in_seg = True
        elif not flag and in_seg:
            starts.append(s)
            ends.append(i - 1)
            in_seg = False
    if in_seg:
        starts.append(s)
        ends.append(len(mask) - 1)

    raw = [(a, b) for a, b in zip(starts, ends) if float(t[b] - t[a]) >= min_dur_s]
    if not raw:
        return []
    merged = [raw[0]]
    for a, b in raw[1:]:
        prev_a, prev_b = merged[-1]
        gap = float(t[a] - t[prev_b])
        if gap <= merge_gap_s:
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


def add_raw_road_candidates(cache: VehicleCache) -> list[dict[str, Any]]:
    t = cache.t_grid_rel_s
    if cache.read_status != "ok" or t.size < 3:
        return []
    curv = cache.signals.get("zx1|lanecurvatureXY")
    if curv is None:
        curv = cache.signals.get("zx|lanecurvatureXY")
    if curv is None:
        return []
    curv_f = fill_nan_linear(curv)
    mask = np.abs(curv_f) >= 8e-4
    segments = find_segments(mask, t, min_dur_s=0.8, merge_gap_s=0.8)
    rows: list[dict[str, Any]] = []
    for idx, (a, b) in enumerate(segments, start=1):
        seg = curv_f[a : b + 1]
        if not np.isfinite(seg).any():
            continue
        signed_peak = float(seg[np.nanargmax(np.abs(seg))])
        score = abs(signed_peak) / 8e-4
        direction = "curve_left_or_positive" if signed_peak >= 0 else "curve_right_or_negative"
        rows.append(
            candidate_row_base(
                cache,
                source="raw_road_curvature_onset",
                source_priority=1,
                event_index=idx,
                anchor_time_rel_s=float(t[a]),
                event_start_rel_s=float(t[a]),
                event_end_rel_s=float(t[b]),
                event_type=direction,
                event_level=event_level_from_score(score),
                phase_type="candidate",
                road_type_anchor="curve",
                curvature_anchor=signed_peak,
                trigger_type="raw_curvature_threshold",
                anchor_signal="zx1|lanecurvatureXY",
                anchor_value=signed_peak,
                source_detail="raw vehicle curvature crossing |curvature|>=8e-4 for >=0.8s",
                leakage_risk_anchor="low_if_road_context_available_before_response",
            )
        )
    return rows


def add_raw_dynamic_candidates(cache: VehicleCache) -> list[dict[str, Any]]:
    t = cache.t_grid_rel_s
    if cache.read_status != "ok" or t.size < 3:
        return []
    steer = cache.signals.get("zx|SteeringWheel")
    yaw_rate = cache.signals.get("zx|vyaw")
    ay = cache.signals.get("zx|ay")
    roll = cache.signals.get("zx|roll")
    if steer is None:
        return []

    steer_f = moving_average(fill_nan_linear(steer), 5)
    steer_rate = np.gradient(steer_f, 1.0 / FS)
    yaw_f = fill_nan_linear(yaw_rate) if yaw_rate is not None else np.zeros_like(steer_f)
    ay_f = fill_nan_linear(ay) if ay is not None else np.zeros_like(steer_f)
    roll_f = fill_nan_linear(roll) if roll is not None else np.zeros_like(steer_f)
    roll_rate = np.gradient(moving_average(roll_f, 5), 1.0 / FS)

    score = np.nanmax(
        np.vstack(
            [
                np.abs(steer_rate) / 0.8,
                np.abs(yaw_f) / 0.3,
                np.abs(ay_f) / 1.3,
                np.abs(roll_rate) / 0.05,
            ]
        ),
        axis=0,
    )
    mask = score >= 1.0
    segments = find_segments(mask, t, min_dur_s=0.15, merge_gap_s=0.35)
    rows: list[dict[str, Any]] = []
    for idx, (a, b) in enumerate(segments, start=1):
        seg_score = score[a : b + 1]
        peak_local = int(np.nanargmax(seg_score))
        peak_i = a + peak_local
        anchor_i = a
        peak_score = float(seg_score[peak_local])
        trigger_components = {
            "steer_rate": float(np.nanmax(np.abs(steer_rate[a : b + 1]))),
            "yaw_rate": float(np.nanmax(np.abs(yaw_f[a : b + 1]))),
            "ay": float(np.nanmax(np.abs(ay_f[a : b + 1]))),
            "roll_rate": float(np.nanmax(np.abs(roll_rate[a : b + 1]))),
        }
        main_component = max(trigger_components, key=trigger_components.get)
        rows.append(
            candidate_row_base(
                cache,
                source="raw_vehicle_dynamic_onset",
                source_priority=3,
                event_index=idx,
                anchor_time_rel_s=float(t[anchor_i]),
                event_start_rel_s=float(t[a]),
                event_end_rel_s=float(t[b]),
                event_type=main_component,
                event_level=event_level_from_score(peak_score),
                phase_type="candidate",
                road_type_anchor="unknown_from_raw_dynamic",
                curvature_anchor=float("nan"),
                trigger_type="raw_dynamic_threshold",
                anchor_signal=main_component,
                anchor_value=trigger_components[main_component],
                source_detail=json.dumps(trigger_components, ensure_ascii=False),
                leakage_risk_anchor="high_response_derived_anchor_not_valid_for_event_trigger_claim",
                extra={
                    "raw_dynamic_peak_time_rel_s": float(t[peak_i]),
                    "raw_dynamic_peak_score": peak_score,
                },
            )
        )
    return rows


def candidate_row_base(
    cache: VehicleCache,
    source: str,
    source_priority: int,
    event_index: int,
    anchor_time_rel_s: float,
    event_start_rel_s: float,
    event_end_rel_s: float,
    event_type: str,
    event_level: str,
    phase_type: str,
    road_type_anchor: str,
    curvature_anchor: float,
    trigger_type: str,
    anchor_signal: str,
    anchor_value: float,
    source_detail: str,
    leakage_risk_anchor: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_uid = f"{source}__{cache.subject}__{cache.session_stamp}__{event_index:05d}"
    row = {
        "event_uid": event_uid,
        "anchor_source": source,
        "anchor_source_priority": source_priority,
        "subject": cache.subject,
        "session_stamp": cache.session_stamp,
        "event_index_in_source": event_index,
        "vehicle_raw_relative_path": cache.raw_relative_path,
        "vehicle_raw_absolute_path": cache.raw_absolute_path,
        "vehicle_raw_sha256": cache.sha256,
        "source_reference_path": "",
        "anchor_time_rel_s": anchor_time_rel_s,
        "anchor_time_abs_s": cache.t0_abs_s + anchor_time_rel_s,
        "event_start_rel_s": event_start_rel_s,
        "event_end_rel_s": event_end_rel_s,
        "event_duration_s": event_end_rel_s - event_start_rel_s,
        "event_type": event_type,
        "event_level": event_level,
        "phase_type": phase_type,
        "road_type_anchor": road_type_anchor,
        "curvature_anchor": curvature_anchor,
        "trigger_type": trigger_type,
        "anchor_signal": anchor_signal,
        "anchor_value": anchor_value,
        "source_detail": source_detail,
        "leakage_risk_anchor": leakage_risk_anchor,
        "causal_anchor_status": "candidate_unverified",
    }
    if extra:
        row.update(extra)
    return row


def load_old_event_candidates(vehicle_rows: pd.DataFrame) -> list[dict[str, Any]]:
    raw_lookup = {
        (str(r.subject), str(r.session_stamp)): r
        for r in vehicle_rows.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    used_v400: set[tuple[str, str]] = set()
    for path in sorted(OLD_PROCESSED_ROOT.rglob("*events_v400_context.csv")):
        subject = path.parts[-3]
        stamp = parse_session_stamp(path.name)
        if stamp is None or (subject, stamp) not in raw_lookup:
            continue
        raw = raw_lookup[(subject, stamp)]
        t0 = float(raw.time_min)
        df = pd.read_csv(path)
        used_v400.add((subject, stamp))
        for i, ev in df.reset_index(drop=True).iterrows():
            trigger_idx = ev.get("trigger_idx", np.nan)
            if pd.notna(trigger_idx):
                anchor_rel = float(trigger_idx) / FS
                anchor_signal = "trigger_idx/200Hz"
            else:
                anchor_rel = float(ev.get("start_s", np.nan))
                anchor_signal = "start_s_fallback"
            if not np.isfinite(anchor_rel):
                continue
            start_s = float(ev.get("start_s", anchor_rel))
            end_s = float(ev.get("end_s", anchor_rel))
            source_rel = safe_rel(path, PROJECT_ROOT)
            event_uid = f"old_v400_context__{subject}__{stamp}__{i+1:05d}"
            rows.append(
                {
                    "event_uid": event_uid,
                    "anchor_source": "old_v400_context_trigger_idx",
                    "anchor_source_priority": 2,
                    "subject": subject,
                    "session_stamp": stamp,
                    "event_index_in_source": i + 1,
                    "vehicle_raw_relative_path": raw.relative_path,
                    "vehicle_raw_absolute_path": raw.absolute_path,
                    "vehicle_raw_sha256": raw.sha256,
                    "source_reference_path": source_rel,
                    "anchor_time_rel_s": anchor_rel,
                    "anchor_time_abs_s": t0 + anchor_rel,
                    "event_start_rel_s": start_s,
                    "event_end_rel_s": end_s,
                    "event_duration_s": end_s - start_s,
                    "event_type": str(ev.get("trigger_type", "")),
                    "event_level": str(ev.get("event_level", "")),
                    "phase_type": str(ev.get("phase_type", "")),
                    "road_type_anchor": str(ev.get("road_type_anchor", "")),
                    "curvature_anchor": float(ev.get("curvature_anchor", np.nan))
                    if pd.notna(ev.get("curvature_anchor", np.nan))
                    else np.nan,
                    "trigger_type": str(ev.get("trigger_type", "")),
                    "anchor_signal": anchor_signal,
                    "anchor_value": float(trigger_idx) if pd.notna(trigger_idx) else np.nan,
                    "source_detail": "old processed v400 context event table; reference only",
                    "leakage_risk_anchor": "medium_old_processed_anchor_unverified",
                    "causal_anchor_status": "old_reference_needs_raw_validation",
                    "old_keep_for_training": bool(ev.get("keep_for_training", False)),
                    "old_primary_score": float(ev.get("primary_score", np.nan))
                    if pd.notna(ev.get("primary_score", np.nan))
                    else np.nan,
                    "old_trigger_score": float(ev.get("trigger_score", np.nan))
                    if pd.notna(ev.get("trigger_score", np.nan))
                    else np.nan,
                    "old_episode_id": ev.get("episode_id", np.nan),
                }
            )

    for path in sorted(OLD_PROCESSED_ROOT.rglob("*events_v312.csv")):
        subject = path.parts[-3]
        stamp = parse_session_stamp(path.name)
        if stamp is None or (subject, stamp) not in raw_lookup or (subject, stamp) in used_v400:
            continue
        raw = raw_lookup[(subject, stamp)]
        t0 = float(raw.time_min)
        df = pd.read_csv(path)
        for i, ev in df.reset_index(drop=True).iterrows():
            anchor_rel = float(ev.get("start_s", np.nan))
            if not np.isfinite(anchor_rel):
                continue
            start_s = float(ev.get("start_s", anchor_rel))
            end_s = float(ev.get("end_s", anchor_rel))
            source_rel = safe_rel(path, PROJECT_ROOT)
            event_uid = f"old_v312_start__{subject}__{stamp}__{i+1:05d}"
            rows.append(
                {
                    "event_uid": event_uid,
                    "anchor_source": "old_v312_start_s_fallback",
                    "anchor_source_priority": 4,
                    "subject": subject,
                    "session_stamp": stamp,
                    "event_index_in_source": i + 1,
                    "vehicle_raw_relative_path": raw.relative_path,
                    "vehicle_raw_absolute_path": raw.absolute_path,
                    "vehicle_raw_sha256": raw.sha256,
                    "source_reference_path": source_rel,
                    "anchor_time_rel_s": anchor_rel,
                    "anchor_time_abs_s": t0 + anchor_rel,
                    "event_start_rel_s": start_s,
                    "event_end_rel_s": end_s,
                    "event_duration_s": end_s - start_s,
                    "event_type": str(ev.get("trigger_type", "")),
                    "event_level": str(ev.get("event_level", "")),
                    "phase_type": str(ev.get("phase_type", "")),
                    "road_type_anchor": "",
                    "curvature_anchor": np.nan,
                    "trigger_type": str(ev.get("trigger_type", "")),
                    "anchor_signal": "start_s",
                    "anchor_value": anchor_rel,
                    "source_detail": "old processed v312 event table fallback; reference only",
                    "leakage_risk_anchor": "high_old_processed_start_fallback_unverified",
                    "causal_anchor_status": "old_reference_needs_raw_validation",
                    "old_keep_for_training": bool(ev.get("keep_for_training", False)),
                    "old_primary_score": np.nan,
                    "old_trigger_score": np.nan,
                    "old_episode_id": ev.get("episode_id", np.nan),
                }
            )
    return rows


def coverage_ratio(start: float, end: float, min_t: float | None, max_t: float | None) -> float:
    if min_t is None or max_t is None or not np.isfinite(min_t) or not np.isfinite(max_t):
        return 0.0
    if end <= start:
        return 0.0
    overlap = max(0.0, min(end, max_t) - max(start, min_t))
    return float(overlap / (end - start))


def value_at(t_grid: np.ndarray, arr: np.ndarray, rel_t: float) -> float:
    if t_grid.size == 0 or arr.size == 0 or not np.isfinite(rel_t):
        return np.nan
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return np.nan
    return float(np.interp(rel_t, t_grid[finite], arr[finite], left=np.nan, right=np.nan))


def label_stats(cache: VehicleCache, anchor_rel: float, start_rel: float, end_rel: float) -> dict[str, Any]:
    t = cache.t_grid_rel_s
    steer = cache.signals.get("zx|SteeringWheel")
    if cache.read_status != "ok" or steer is None or t.size == 0:
        return {
            "label_peak_delta": np.nan,
            "label_peak_abs_delta": np.nan,
            "label_peak_time_rel_s": np.nan,
            "label_peak_direction": "unknown",
            "label_tail_delta": np.nan,
            "label_reversal_count_proxy": np.nan,
        }
    t0 = anchor_rel + start_rel
    t1 = anchor_rel + end_rel
    mask = (t >= t0) & (t <= t1) & np.isfinite(steer)
    if mask.sum() < 3:
        return {
            "label_peak_delta": np.nan,
            "label_peak_abs_delta": np.nan,
            "label_peak_time_rel_s": np.nan,
            "label_peak_direction": "unknown",
            "label_tail_delta": np.nan,
            "label_reversal_count_proxy": np.nan,
        }
    base = value_at(t, steer, anchor_rel)
    vals = steer[mask] - base
    times = t[mask] - anchor_rel
    peak_i = int(np.nanargmax(np.abs(vals)))
    deriv = np.diff(fill_nan_linear(vals))
    cutoff = max(0.002, float(np.nanpercentile(np.abs(deriv), 70)) * 0.3) if deriv.size else 0.002
    sign = np.sign(deriv)
    sign[np.abs(deriv) < cutoff] = 0
    nonzero = sign[sign != 0]
    reversal_count = int(np.sum(nonzero[1:] * nonzero[:-1] < 0)) if len(nonzero) > 1 else 0
    peak_delta = float(vals[peak_i])
    return {
        "label_peak_delta": peak_delta,
        "label_peak_abs_delta": abs(peak_delta),
        "label_peak_time_rel_s": float(times[peak_i]),
        "label_peak_direction": "positive" if peak_delta >= 0 else "negative",
        "label_tail_delta": float(vals[-1]),
        "label_reversal_count_proxy": reversal_count,
    }


def build_quality_maps(
    inventory: pd.DataFrame,
    timestamp: pd.DataFrame,
    signal_quality: pd.DataFrame,
    eeg_quality: pd.DataFrame,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    modality_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    inv_by_rel = {str(r.relative_path): r for r in inventory.itertuples(index=False)}
    ts_by_rel = {str(r.relative_path): r for r in timestamp.itertuples(index=False)}
    for key, inv in inv_by_rel.items():
        ts = ts_by_rel.get(key)
        modality_map[(str(inv.subject), str(inv.session_stamp), str(inv.modality))] = {
            "relative_path": str(inv.relative_path),
            "absolute_path": str(inv.absolute_path),
            "sha256": str(inv.sha256),
            "time_min": float(getattr(ts, "time_min", np.nan)) if ts is not None else np.nan,
            "time_max": float(getattr(ts, "time_max", np.nan)) if ts is not None else np.nan,
            "zero_dt_count": int(getattr(ts, "zero_dt_count", 0)) if ts is not None else 0,
            "large_gap_count": int(getattr(ts, "large_gap_count", 0)) if ts is not None else 0,
        }

    file_quality: dict[str, dict[str, Any]] = {}
    for rel, grp in signal_quality.groupby("relative_path"):
        file_quality[str(rel)] = {
            "min_valid_rate": float(grp["valid_rate"].min()),
            "mean_valid_rate": float(grp["valid_rate"].mean()),
            "near_constant_count": int(grp["near_constant"].fillna(False).sum()),
            "signal_count": int(len(grp)),
        }
    for rel, grp in eeg_quality.groupby("relative_path"):
        file_quality[str(rel)] = {
            "min_valid_rate": float(grp["valid_rate"].min()),
            "mean_valid_rate": float(grp["valid_rate"].mean()),
            "near_constant_count": int(grp["near_constant"].fillna(False).sum()),
            "signal_count": int(len(grp)),
        }
    return modality_map, file_quality


def build_samples(
    candidates: pd.DataFrame,
    caches: dict[tuple[str, str], VehicleCache],
    modality_map: dict[tuple[str, str, str], dict[str, Any]],
    file_quality: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ev in candidates.itertuples(index=False):
        key = (str(ev.subject), str(ev.session_stamp))
        cache = caches.get(key)
        if cache is None:
            continue
        for cfg in WINDOW_CONFIGS:
            sample_id = f"{ev.event_uid}__{cfg['window_config_id']}"
            input_abs_start = float(ev.anchor_time_abs_s + cfg["input_start_rel_s"])
            input_abs_end = float(ev.anchor_time_abs_s + cfg["input_end_rel_s"])
            label_abs_start = float(ev.anchor_time_abs_s + cfg["label_start_rel_s"])
            label_abs_end = float(ev.anchor_time_abs_s + cfg["label_end_rel_s"])

            modality_bits = []
            cov: dict[str, float] = {}
            quality_bits = []
            path_bits: dict[str, str] = {}
            sha_bits: dict[str, str] = {}
            for modality in ["vehicle", "physio", "eeg"]:
                meta = modality_map.get((str(ev.subject), str(ev.session_stamp), modality), {})
                rel = str(meta.get("relative_path", ""))
                path_bits[f"{modality}_relative_path"] = rel
                sha_bits[f"{modality}_sha256"] = str(meta.get("sha256", ""))
                cov[f"{modality}_input_coverage"] = coverage_ratio(
                    input_abs_start,
                    input_abs_end,
                    float(meta.get("time_min", np.nan)),
                    float(meta.get("time_max", np.nan)),
                )
                if modality == "vehicle":
                    cov[f"{modality}_label_coverage"] = coverage_ratio(
                        label_abs_start,
                        label_abs_end,
                        float(meta.get("time_min", np.nan)),
                        float(meta.get("time_max", np.nan)),
                    )
                if cov[f"{modality}_input_coverage"] >= 0.95:
                    modality_bits.append(modality)
                if int(meta.get("zero_dt_count", 0)) > 0:
                    quality_bits.append(f"{modality}_zero_dt")
                if int(meta.get("large_gap_count", 0)) > 0:
                    quality_bits.append(f"{modality}_large_gap")
                q = file_quality.get(rel)
                if q:
                    if q["min_valid_rate"] < 0.95:
                        quality_bits.append(f"{modality}_low_valid")
                    if q["near_constant_count"] > 0:
                        quality_bits.append(f"{modality}_near_constant")

            leakage_bits = [str(ev.leakage_risk_anchor)]
            if cfg["input_end_rel_s"] > 0:
                leakage_bits.append("input_contains_post_anchor_observation")
                if "physio" in modality_bits:
                    leakage_bits.append("physio_emg_window_may_include_post_anchor_action")
            if str(ev.anchor_source).startswith("old_"):
                leakage_bits.append("old_processed_event_reference_not_final_truth")
            if str(ev.anchor_source) == "raw_vehicle_dynamic_onset":
                leakage_bits.append("anchor_derived_from_vehicle_response")

            minimal_ok = (
                cov.get("vehicle_input_coverage", 0.0) >= 0.99
                and cov.get("vehicle_label_coverage", 0.0) >= 0.99
            )
            deployable_event_trigger = (
                minimal_ok
                and cfg["input_end_rel_s"] <= 0.0
                and str(ev.anchor_source) == "raw_road_curvature_onset"
            )
            stats = label_stats(cache, float(ev.anchor_time_rel_s), cfg["label_start_rel_s"], cfg["label_end_rel_s"])
            rows.append(
                {
                    "sample_id": sample_id,
                    "event_uid": ev.event_uid,
                    "subject": ev.subject,
                    "session_stamp": ev.session_stamp,
                    "anchor_source": ev.anchor_source,
                    "anchor_time_rel_s": ev.anchor_time_rel_s,
                    "anchor_time_abs_s": ev.anchor_time_abs_s,
                    "event_start_rel_s": ev.event_start_rel_s,
                    "event_end_rel_s": ev.event_end_rel_s,
                    "event_type": ev.event_type,
                    "event_level": ev.event_level,
                    "phase_type": ev.phase_type,
                    "road_type_anchor": ev.road_type_anchor,
                    "curvature_anchor": ev.curvature_anchor,
                    "trigger_type": ev.trigger_type,
                    "window_config_id": cfg["window_config_id"],
                    "causal_setting": cfg["causal_setting"],
                    "input_start_rel_s": cfg["input_start_rel_s"],
                    "input_end_rel_s": cfg["input_end_rel_s"],
                    "label_start_rel_s": cfg["label_start_rel_s"],
                    "label_end_rel_s": cfg["label_end_rel_s"],
                    "input_abs_start_s": input_abs_start,
                    "input_abs_end_s": input_abs_end,
                    "label_abs_start_s": label_abs_start,
                    "label_abs_end_s": label_abs_end,
                    "available_modalities": ",".join(modality_bits),
                    "sample_usable_vehicle_minimal": minimal_ok,
                    "recommended_for_stage3_vehicle_baseline": deployable_event_trigger,
                    "quality_flags": ";".join(sorted(set(quality_bits))) if quality_bits else "ok",
                    "leakage_flags": ";".join(sorted(set(leakage_bits))),
                    "window_note": cfg["window_note"],
                    **cov,
                    **path_bits,
                    **sha_bits,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def nearest_delta_report(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    old = candidates[candidates["anchor_source"].str.startswith("old_")].copy()
    raw = candidates[~candidates["anchor_source"].str.startswith("old_")].copy()
    for ev in old.itertuples(index=False):
        same = raw[(raw["subject"] == ev.subject) & (raw["session_stamp"] == ev.session_stamp)]
        row = {
            "old_event_uid": ev.event_uid,
            "subject": ev.subject,
            "session_stamp": ev.session_stamp,
            "old_anchor_source": ev.anchor_source,
            "old_anchor_time_rel_s": ev.anchor_time_rel_s,
            "phase_type": ev.phase_type,
            "event_level": ev.event_level,
        }
        for source in ["raw_road_curvature_onset", "raw_vehicle_dynamic_onset"]:
            sub = same[same["anchor_source"] == source]
            if sub.empty:
                row[f"nearest_{source}_event_uid"] = ""
                row[f"nearest_{source}_delta_s"] = np.nan
                row[f"nearest_{source}_within_0p5s"] = False
                row[f"nearest_{source}_within_1s"] = False
            else:
                deltas = sub["anchor_time_rel_s"].to_numpy(dtype=float) - float(ev.anchor_time_rel_s)
                j = int(np.nanargmin(np.abs(deltas)))
                nearest = sub.iloc[j]
                delta = float(deltas[j])
                row[f"nearest_{source}_event_uid"] = nearest["event_uid"]
                row[f"nearest_{source}_delta_s"] = delta
                row[f"nearest_{source}_within_0p5s"] = abs(delta) <= 0.5
                row[f"nearest_{source}_within_1s"] = abs(delta) <= 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_split_table(candidates: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    event_rows = []
    sample_counts = samples.groupby("event_uid").size().to_dict()
    for ev in candidates.itertuples(index=False):
        session_key = f"{ev.subject}__{ev.session_stamp}"
        event_rows.append(
            {
                "event_uid": ev.event_uid,
                "subject": ev.subject,
                "session_stamp": ev.session_stamp,
                "anchor_source": ev.anchor_source,
                "phase_type": ev.phase_type,
                "event_level": ev.event_level,
                "sample_row_count": int(sample_counts.get(ev.event_uid, 0)),
                "random_event_split": split_from_key(str(ev.event_uid)),
                "session_level_split": split_from_key(session_key),
                "subject_level_split": split_from_key(str(ev.subject)),
                "normalization_protocol": "fit_scalers_and_feature_learning_on_train_only_for_each_split",
            }
        )
    return pd.DataFrame(event_rows)


def split_feasibility(split_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy in ["random_event_split", "session_level_split", "subject_level_split"]:
        for split_name, grp in split_table.groupby(strategy):
            rows.append(
                {
                    "split_strategy": strategy,
                    "split": split_name,
                    "event_count": int(len(grp)),
                    "sample_row_count": int(grp["sample_row_count"].sum()),
                    "subject_count": int(grp["subject"].nunique()),
                    "session_count": int(grp[["subject", "session_stamp"]].drop_duplicates().shape[0]),
                    "anchor_sources": ",".join(sorted(grp["anchor_source"].dropna().unique())),
                }
            )
    return pd.DataFrame(rows)


def draw_bar_chart(path: Path, labels: list[str], values: list[int], title: str) -> None:
    width, height = 1100, 650
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((40, 25), title, fill=(20, 20, 20), font=font)
    max_v = max(values) if values else 1
    left, top, chart_w, chart_h = 80, 90, 940, 460
    bar_h = max(18, chart_h // max(len(values), 1) - 12)
    for i, (label, val) in enumerate(zip(labels, values)):
        y = top + i * (bar_h + 12)
        w = int(chart_w * val / max_v)
        draw.rectangle((left, y, left + w, y + bar_h), fill=(57, 106, 177))
        draw.text((left + w + 8, y + 2), str(val), fill=(20, 20, 20), font=font)
        draw.text((left, y + bar_h + 1), label[:90], fill=(20, 20, 20), font=font)
    img.save(path)


def draw_hist(path: Path, values: list[float], title: str) -> None:
    clean = [float(v) for v in values if np.isfinite(v)]
    width, height = 1000, 600
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((40, 25), title, fill=(20, 20, 20), font=font)
    if not clean:
        draw.text((80, 120), "No finite values", fill=(160, 0, 0), font=font)
        img.save(path)
        return
    bins = np.linspace(-3.0, 3.0, 31)
    hist, edges = np.histogram(np.clip(clean, -3, 3), bins=bins)
    left, top, chart_w, chart_h = 80, 90, 850, 400
    max_h = max(int(hist.max()), 1)
    bar_w = chart_w / len(hist)
    for i, count in enumerate(hist):
        x0 = left + int(i * bar_w)
        x1 = left + int((i + 1) * bar_w) - 2
        h = int(chart_h * int(count) / max_h)
        draw.rectangle((x0, top + chart_h - h, x1, top + chart_h), fill=(204, 95, 39))
    draw.line((left, top + chart_h, left + chart_w, top + chart_h), fill=(0, 0, 0))
    draw.text((left, top + chart_h + 12), "-3s", fill=(0, 0, 0), font=font)
    draw.text((left + chart_w - 35, top + chart_h + 12), "+3s", fill=(0, 0, 0), font=font)
    draw.text((left, top + chart_h + 34), f"n={len(clean)}, median={np.median(clean):.3f}s", fill=(0, 0, 0), font=font)
    img.save(path)


def draw_overlay(path: Path, cache: VehicleCache, candidates: pd.DataFrame) -> None:
    width, height = 1200, 650
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    title = f"Anchor overlay: {cache.subject} {cache.session_stamp}"
    draw.text((40, 25), title, fill=(20, 20, 20), font=font)
    t = cache.t_grid_rel_s
    steer = cache.signals.get("zx|SteeringWheel")
    if steer is None or t.size == 0:
        draw.text((80, 120), "No steering signal", fill=(160, 0, 0), font=font)
        img.save(path)
        return
    finite = np.isfinite(steer)
    if finite.sum() < 2:
        img.save(path)
        return
    t_min, t_max = float(np.nanmin(t)), min(float(np.nanmax(t)), 180.0)
    y_vals = steer.copy()
    y_min, y_max = float(np.nanpercentile(y_vals[finite], 1)), float(np.nanpercentile(y_vals[finite], 99))
    if abs(y_max - y_min) < 1e-6:
        y_min -= 1.0
        y_max += 1.0
    left, top, chart_w, chart_h = 70, 80, 1060, 470
    idx = np.where((t >= t_min) & (t <= t_max) & finite)[0]
    if len(idx) > 2500:
        idx = idx[:: max(1, len(idx) // 2500)]
    pts = []
    for i in idx:
        x = left + int((t[i] - t_min) / (t_max - t_min) * chart_w)
        y = top + chart_h - int((y_vals[i] - y_min) / (y_max - y_min) * chart_h)
        pts.append((x, y))
    if len(pts) > 1:
        draw.line(pts, fill=(45, 45, 45), width=2)
    colors = {
        "old_v400_context_trigger_idx": (204, 95, 39),
        "raw_road_curvature_onset": (57, 106, 177),
        "raw_vehicle_dynamic_onset": (62, 150, 81),
        "old_v312_start_s_fallback": (160, 80, 160),
    }
    sample = candidates[(candidates["subject"] == cache.subject) & (candidates["session_stamp"] == cache.session_stamp)].copy()
    for _, row in sample.head(220).iterrows():
        x = left + int((float(row["anchor_time_rel_s"]) - t_min) / (t_max - t_min) * chart_w)
        if left <= x <= left + chart_w:
            color = colors.get(row["anchor_source"], (100, 100, 100))
            draw.line((x, top, x, top + chart_h), fill=color)
    draw.rectangle((left, top, left + chart_w, top + chart_h), outline=(0, 0, 0))
    legend_y = top + chart_h + 25
    for i, (src, color) in enumerate(colors.items()):
        x0 = left + i * 260
        draw.rectangle((x0, legend_y, x0 + 20, legend_y + 12), fill=color)
        draw.text((x0 + 28, legend_y), src[:32], fill=(0, 0, 0), font=font)
    img.save(path)


def write_reports(
    candidates: pd.DataFrame,
    samples: pd.DataFrame,
    split_table: pd.DataFrame,
    feasibility: pd.DataFrame,
    comparison: pd.DataFrame,
    road_design_inventory: pd.DataFrame,
) -> None:
    source_counts = candidates.groupby("anchor_source").size().sort_values(ascending=False)
    usable_by_window = samples.groupby("window_config_id")["sample_usable_vehicle_minimal"].sum().astype(int)
    recommended_count = int(samples["recommended_for_stage3_vehicle_baseline"].fillna(False).sum())
    old_count = int(candidates["anchor_source"].str.startswith("old_").sum())
    raw_count = int((~candidates["anchor_source"].str.startswith("old_")).sum())
    v400_primary = int(((candidates["anchor_source"] == "old_v400_context_trigger_idx") & (candidates["phase_type"] == "primary")).sum())
    dyn_within_1s = int(comparison.get("nearest_raw_vehicle_dynamic_onset_within_1s", pd.Series(dtype=bool)).fillna(False).sum()) if not comparison.empty else 0
    road_within_1s = int(comparison.get("nearest_raw_road_curvature_onset_within_1s", pd.Series(dtype=bool)).fillna(False).sum()) if not comparison.empty else 0
    road_design_files = int(len(road_design_inventory)) if road_design_inventory is not None else 0
    road_design_csv = int((road_design_inventory["suffix"] == ".csv").sum()) if road_design_files else 0
    road_design_curv_csv = int((road_design_inventory["curvature_col"].fillna("") != "").sum()) if road_design_files else 0
    window_config_text = pd.DataFrame(WINDOW_CONFIGS).to_string(index=False)

    card = f"""# 阶段 2 数据版本卡：R2E raw candidate manifest v0.2

生成时间：2026-05-12

## 版本定位

本版本只用于事件锚点和样本清单重建，不用于直接训练模型。它把旧事件表、原始道路上下文候选和原始车辆动态候选放在同一张清单里，并用泄漏风险字段区分它们。

## 输入来源

- 原始车辆/生理/脑电清单：`01_audit/tables/raw_file_inventory.csv`
- 时间连续性和模态重叠：`01_audit/tables/timestamp_continuity_report.csv`、`01_audit/tables/modality_overlap_report.csv`
- 旧流程事件参考：`01_datasets/多模态数据/被试数据集合/<subject>/event/*events_v400_context.csv`
- 原始车辆信号：`01_datasets/数据预处理/原始车辆数据/<subject>/*.csv`
- 道路设计记录：`01_datasets/多模态数据/被试数据集合/道路信息`

## 道路设计记录审计

- 道路信息目录文件数：{road_design_files}
- 其中 CSV 文件数：{road_design_csv}
- 含 `curvature/kappa/curvature_1pm` 的道路设计 CSV：{road_design_curv_csv}
- 道路设计清单：`02_samples/tables/road_design_inventory.csv`
- 当前只把道路设计作为锚点来源证据和后续精确对齐依据；本版低泄漏道路候选仍来自原始车辆 `lanecurvatureXY` 的时间序列，未把道路设计文件强行投影到每个原始时间戳。

## 候选锚点来源

{source_counts.to_string()}

## 窗口配置

{window_config_text}

## 样本数量

- 候选事件总数：{len(candidates)}
- 旧处理事件参考候选：{old_count}
- 原始信号重建候选：{raw_count}
- old v400 primary 候选：{v400_primary}
- `samples_master.csv` 行数：{len(samples)}
- 车辆输入和标签窗口均可覆盖的样本行：{int(samples['sample_usable_vehicle_minimal'].sum())}
- 当前可作为较低泄漏道路上下文候选的 stage3 车辆基线行：{recommended_count}

## 窗口可用性

{usable_by_window.to_string()}

## 旧锚点与原始候选的近邻关系

- 旧参考锚点 1 秒内可找到 raw dynamic onset 的数量：{dyn_within_1s}
- 旧参考锚点 1 秒内可找到 raw road curvature onset 的数量：{road_within_1s}

## 切分协议

- `random_event_split`：按 `event_uid` 哈希切分，避免同一事件的不同窗口落入不同 split。
- `session_level_split`：按 `subject + session_stamp` 哈希切分，同一记录内所有事件同 split。
- `subject_level_split`：按 `subject` 哈希切分，评估跨被试泛化可行性。
- 任何标准化、特征学习、风格聚类、质量阈值学习都必须只在 train split 上拟合，再应用到 val/test。

## 当前结论

本版本已经能把候选样本追溯到原始文件、原始时间戳、锚点来源、窗口和模态可用性。但 old v400 和 raw dynamic 锚点都不能直接证明无泄漏；进入正式阶段 3 前，只能优先使用 `raw_road_curvature_onset` 的低泄漏候选做保守车辆基线预研，或者先人工/GPTPro 审查锚点规则。
"""
    (REPORT_DIR / "dataset_version_card_v0_2_cn.md").write_text(card, encoding="utf-8")

    summary = f"""# 阶段 2 事件锚点与样本清单重建总结

更新时间：2026-05-12

## 做了什么

1. 读取阶段 1 的原始文件清单、时间连续性、模态重叠和质量报告。
2. 匹配旧流程 `events_v400_context.csv`，只作为历史参考，不直接继承为最终真相。
3. 检索道路设计记录，生成 `road_design_inventory.csv`，确认存在道路中心线、曲率和模块信息。
4. 从原始车辆信号重新生成两类候选：道路曲率进入候选和车辆动态响应 onset 候选。
5. 为每个候选锚点生成 4 套窗口配置，写入 `samples_master.csv/jsonl`。
6. 生成随机、session-level、subject-level 三类 split 表，并明确 train-only 标准化规则。

## 主要数量

- 候选事件总数：{len(candidates)}
- 样本窗口行数：{len(samples)}
- 道路设计目录文件数：{road_design_files}
- 含曲率信息的道路设计 CSV：{road_design_curv_csv}
- source 计数：

{source_counts.to_string()}

## 风险判断

- `old_v400_context_trigger_idx`：来自旧处理事件表，能对照历史结果，但不能直接当作新流程真相。
- `raw_vehicle_dynamic_onset`：从方向盘、横摆、横向加速度等响应导出，可能已经接近或进入标签响应，不能用于证明事件触发预测无泄漏。
- `raw_road_curvature_onset`：来自原始道路曲率变化，泄漏风险较低；道路设计文件证明项目中有道路几何记录，但本轮还没有完成逐时间戳投影，所以它仍是候选锚点，不是最终道路真值。
- 任何 `input_end_rel_s > 0` 的窗口都属于早期观察后预测剩余轨迹，不能和事件发生时预测完整未来混淆。

## 是否可以进入阶段 3

可以进入阶段 3 的前置准备，但只能先做无学习/强车辆基线的保守版本：优先使用 `raw_road_curvature_onset` 且 `input_end_rel_s<=0` 的样本。旧 v400 和 raw dynamic 样本必须作为对照或上限分析，不能作为最终无泄漏主线。
"""
    (REPORT_DIR / "event_anchor_rebuild_summary_cn.md").write_text(summary, encoding="utf-8")

    user = f"""# 阶段 2 用户查看版总结：事件锚点与样本清单重建

更新时间：2026-05-12

## 这个阶段为什么做

阶段 1 已经证明原始车辆、生理、脑电文件能互相对应，但旧流程的事件锚点不能默认相信。阶段 2 的目的就是重新把“一个样本从哪里来、什么时候开始预测、用哪段输入、预测哪段未来”写清楚。

## 这个阶段检查了什么

- 检查旧事件表能否和原始车辆文件按被试、记录时间对应。
- 找到旧项目中的道路设计/道路信息记录，并生成道路设计清单。
- 从原始车辆曲率信号找道路事件候选。
- 从原始车辆动态信号找响应 onset 候选。
- 给每个候选样本生成 1 秒、2 秒、3 秒和早期观察 0.5 秒四种窗口。
- 给每个样本写入车辆、生理、脑电是否覆盖输入窗口，以及车辆是否覆盖标签窗口。
- 生成随机切分、按记录切分、按被试切分三种 split 方案。

## 目前发现了什么

- 候选事件总数：{len(candidates)}。
- 样本窗口行数：{len(samples)}。
- 道路设计目录文件数：{road_design_files}，其中含曲率信息的 CSV 为 {road_design_curv_csv} 个。
- 旧 v400 primary 事件候选：{v400_primary}。
- 低泄漏道路曲率候选窗口行：{recommended_count}。
- 旧锚点和原始动态响应锚点在 1 秒内匹配的数量为 {dyn_within_1s}，说明旧事件大多能在原始车辆响应中找到对应迹象。
- 旧锚点和道路曲率锚点在 1 秒内匹配的数量为 {road_within_1s}，说明道路曲率只能解释一部分事件。

## 哪些结果可信

- 每个样本行都能追溯到原始车辆文件、SHA256、被试、记录时间和窗口绝对时间。
- 每个样本都有明确的 `anchor_source` 和 `leakage_flags`。
- split 表已经保证同一事件的不同窗口不会分到不同训练/测试集合。

## 哪些结果还不能下结论

- 不能把旧 v400 事件锚点当成最终真相。
- 不能把 raw dynamic onset 当作无泄漏事件触发锚点，因为它来自车辆响应本身。
- 不能说生理数据有效；这里只是记录生理/脑电窗口是否覆盖。
- 不能直接训练最终模型；阶段 3 只能从强车辆基线和保守样本子集开始。

## 下一阶段是否可以继续

可以继续到阶段 3 的准备工作，但要分两条线：

1. 用 `raw_road_curvature_onset` 做低泄漏保守车辆基线。
2. 用 old v400 和 raw dynamic 做历史对照/上限分析，不能混作主结论。

## 推荐优先查看

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/samples_master.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/anchor_source_inventory.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/split_table.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_v0_2_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/figures/stage02_candidate_counts_by_source.png`
"""
    (REPORT_DIR / "stage02_user_summary_cn.md").write_text(user, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    inventory = pd.read_csv(AUDIT_TABLE_DIR / "raw_file_inventory.csv")
    timestamp = pd.read_csv(AUDIT_TABLE_DIR / "timestamp_continuity_report.csv")
    signal_quality = pd.read_csv(AUDIT_TABLE_DIR / "signal_quality_report.csv")
    eeg_quality = pd.read_csv(AUDIT_TABLE_DIR / "eeg_artifact_report.csv")
    road_design_inventory = scan_road_design_inventory()

    modality_map, file_quality = build_quality_maps(inventory, timestamp, signal_quality, eeg_quality)
    vehicle_rows = inventory[inventory["modality"] == "vehicle"].merge(
        timestamp[["relative_path", "time_min", "time_max", "zero_dt_count", "large_gap_count"]],
        on="relative_path",
        how="left",
    )

    old_rows = load_old_event_candidates(vehicle_rows)
    caches: dict[tuple[str, str], VehicleCache] = {}
    raw_rows: list[dict[str, Any]] = []
    cache_status_rows: list[dict[str, Any]] = []
    for row in vehicle_rows.itertuples(index=False):
        cache = read_vehicle_cache(pd.Series(row._asdict()))
        key = (cache.subject, cache.session_stamp)
        caches[key] = cache
        cache_status_rows.append(
            {
                "subject": cache.subject,
                "session_stamp": cache.session_stamp,
                "vehicle_raw_relative_path": cache.raw_relative_path,
                "read_status": cache.read_status,
                "read_error": cache.read_error,
                "duration_s": float(cache.t_grid_rel_s[-1]) if cache.t_grid_rel_s.size else np.nan,
                "grid_rows": int(cache.t_grid_rel_s.size),
            }
        )
        if cache.read_status != "ok":
            continue
        raw_rows.extend(add_raw_road_candidates(cache))
        raw_rows.extend(add_raw_dynamic_candidates(cache))

    candidates = pd.DataFrame(old_rows + raw_rows)
    if candidates.empty:
        raise RuntimeError("no candidate events generated")
    candidates = candidates.sort_values(
        ["subject", "session_stamp", "anchor_source_priority", "anchor_time_rel_s", "event_uid"]
    ).reset_index(drop=True)

    comparison = nearest_delta_report(candidates)
    samples = build_samples(candidates, caches, modality_map, file_quality)
    split_table = build_split_table(candidates, samples)
    feasibility = split_feasibility(split_table)

    source_inventory = (
        candidates.groupby("anchor_source")
        .agg(
            event_count=("event_uid", "count"),
            subject_count=("subject", "nunique"),
            session_count=("session_stamp", "nunique"),
            median_anchor_time_rel_s=("anchor_time_rel_s", "median"),
        )
        .reset_index()
    )
    window_inventory = (
        samples.groupby(["window_config_id", "anchor_source"])
        .agg(
            sample_rows=("sample_id", "count"),
            vehicle_minimal_ok=("sample_usable_vehicle_minimal", "sum"),
            recommended_stage3=("recommended_for_stage3_vehicle_baseline", "sum"),
            median_label_peak_abs_delta=("label_peak_abs_delta", "median"),
        )
        .reset_index()
    )

    candidates.to_csv(TABLE_DIR / "candidate_events_master.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(old_rows).to_csv(TABLE_DIR / "candidate_events_from_old_reference.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(raw_rows).to_csv(TABLE_DIR / "candidate_events_from_raw_vehicle.csv", index=False, encoding="utf-8-sig")
    samples.to_csv(TABLE_DIR / "samples_master.csv", index=False, encoding="utf-8-sig")
    samples.to_json(TABLE_DIR / "samples_master.jsonl", orient="records", lines=True, force_ascii=False)
    split_table.to_csv(TABLE_DIR / "split_table.csv", index=False, encoding="utf-8-sig")
    feasibility.to_csv(TABLE_DIR / "split_feasibility_report.csv", index=False, encoding="utf-8-sig")
    source_inventory.to_csv(TABLE_DIR / "anchor_source_inventory.csv", index=False, encoding="utf-8-sig")
    window_inventory.to_csv(TABLE_DIR / "window_config_comparison.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(TABLE_DIR / "anchor_source_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cache_status_rows).to_csv(TABLE_DIR / "vehicle_resample_status.csv", index=False, encoding="utf-8-sig")
    road_design_inventory.to_csv(TABLE_DIR / "road_design_inventory.csv", index=False, encoding="utf-8-sig")

    counts = source_inventory.sort_values("event_count", ascending=True)
    draw_bar_chart(
        FIG_DIR / "stage02_candidate_counts_by_source.png",
        counts["anchor_source"].tolist(),
        counts["event_count"].astype(int).tolist(),
        "Stage 2 candidate event counts by anchor source",
    )
    draw_bar_chart(
        FIG_DIR / "stage02_sample_window_viability.png",
        window_inventory.groupby("window_config_id")["vehicle_minimal_ok"].sum().sort_values().index.tolist(),
        window_inventory.groupby("window_config_id")["vehicle_minimal_ok"].sum().sort_values().astype(int).tolist(),
        "Vehicle input+label coverage by window config",
    )
    if not comparison.empty and "nearest_raw_vehicle_dynamic_onset_delta_s" in comparison.columns:
        draw_hist(
            FIG_DIR / "stage02_old_to_raw_dynamic_delta_hist.png",
            comparison["nearest_raw_vehicle_dynamic_onset_delta_s"].dropna().tolist(),
            "Old anchor to nearest raw dynamic onset delta",
        )
    if caches:
        overlay_key = next((k for k, v in caches.items() if v.read_status == "ok"), None)
        if overlay_key is not None:
            draw_overlay(FIG_DIR / "stage02_anchor_overlay_example.png", caches[overlay_key], candidates)

    write_reports(candidates, samples, split_table, feasibility, comparison, road_design_inventory)

    run_summary = {
        "candidate_events": int(len(candidates)),
        "samples_master_rows": int(len(samples)),
        "anchor_source_counts": {str(k): int(v) for k, v in candidates.groupby("anchor_source").size().to_dict().items()},
        "window_rows": {str(k): int(v) for k, v in samples.groupby("window_config_id").size().to_dict().items()},
        "recommended_stage3_vehicle_baseline_rows": int(samples["recommended_for_stage3_vehicle_baseline"].sum()),
        "cache_status": {str(k): int(v) for k, v in pd.DataFrame(cache_status_rows).groupby("read_status").size().to_dict().items()},
        "road_design_files": int(len(road_design_inventory)),
    }
    (LOG_DIR / "build_stage2_samples_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
