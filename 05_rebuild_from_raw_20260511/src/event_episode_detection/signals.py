# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FIELD_ALIASES = {
    "time": ["StorageTime", "time", "timestamp", "Time", "Timestamp"],
    "steer": ["zx|SteeringWheel", "SteeringWheel", "steering", "steer", "steering_wheel"],
    "speed": ["zx1|v_km/h", "v_km/h", "speed", "Speed", "vehicle_speed"],
    "ay": ["zx|ay", "ay", "lateral_acceleration", "lat_acc"],
    "yaw_rate": ["zx|vyaw", "vyaw", "yaw_rate", "YawRate"],
    "roll_rate": ["zx|vroll", "vroll", "roll_rate", "RollRate"],
    "roll": ["zx|roll", "roll", "Roll"],
    "lat_offset": ["zx1|lateraldistance", "lateraldistance", "lane_offset", "lat_offset"],
    "brake": ["zx|BrakePedal", "BrakePedal", "brake", "brake_pedal"],
    "ax": ["zx|ax", "ax", "longitudinal_acceleration", "long_acc"],
    "mu": ["zx1|mu", "mu", "friction"],
    "curvature": ["zx1|lanecurvatureXY", "lanecurvatureXY", "curvature"],
}


def robust_mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(arr - med)) * 1.4826)


def robust_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmedian(arr))


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def parse_time_to_relative_seconds(series: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() >= max(10, len(series) // 2):
        base = parsed.dropna().iloc[0]
        return (parsed - base).dt.total_seconds().to_numpy(dtype=float)
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return np.full(len(series), np.nan)
    return numeric - finite[0]


def gradient(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    good = np.isfinite(values) & np.isfinite(times)
    if good.sum() < 3:
        return out
    idx = np.arange(len(values))
    filled = values.copy()
    filled[~good] = np.interp(idx[~good], idx[good], values[good])
    dt = np.gradient(times)
    dt[~np.isfinite(dt) | (np.abs(dt) < 1e-6)] = np.nan
    return np.gradient(filled) / dt


def rolling_smooth(values: np.ndarray, times: np.ndarray, window_sec: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 3 or window_sec <= 0:
        return values.copy()
    dt = np.diff(times)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    sample_dt = float(np.nanmedian(dt)) if dt.size else 0.005
    win = max(3, int(round(window_sec / max(sample_dt, 1e-3))))
    if win % 2 == 0:
        win += 1
    series = pd.Series(values)
    return series.rolling(win, center=True, min_periods=1).median().to_numpy(dtype=float)


def first_existing(columns: list[str], aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    lower_map = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def identify_fields(columns: list[str]) -> dict[str, str | None]:
    return {name: first_existing(columns, aliases) for name, aliases in FIELD_ALIASES.items()}


def load_vehicle_csv(path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {
        "vehicle_path": str(path),
        "read_status": "ok",
        "warnings": [],
    }
    try:
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    except Exception as exc:
        meta["read_status"] = "read_header_failed"
        meta["warnings"].append(str(exc))
        return pd.DataFrame(), meta
    fields = identify_fields(header.columns.tolist())
    meta["identified_fields"] = {k: v for k, v in fields.items() if v}
    if not fields.get("time") or not fields.get("steer"):
        meta["read_status"] = "missing_required_fields"
        meta["warnings"].append("缺少 time 或 steering wheel angle")
        return pd.DataFrame(), meta
    usecols = sorted({v for v in fields.values() if v})
    try:
        df = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    except Exception as exc:
        meta["read_status"] = "read_csv_failed"
        meta["warnings"].append(str(exc))
        return pd.DataFrame(), meta

    df["time_rel_s"] = parse_time_to_relative_seconds(df[str(fields["time"])])
    df = df[np.isfinite(df["time_rel_s"])].copy()
    df = df.drop_duplicates("time_rel_s").sort_values("time_rel_s").reset_index(drop=True)
    if len(df) < 20:
        meta["read_status"] = "too_few_rows"
        meta["warnings"].append("有效时间点过少")
        return pd.DataFrame(), meta

    canonical: dict[str, str] = {}
    for name, col in fields.items():
        if not col or col not in df.columns:
            continue
        new_col = name
        if name == "time":
            continue
        df[new_col] = pd.to_numeric(df[col], errors="coerce")
        df[new_col] = df[new_col].interpolate(limit_direction="both")
        canonical[name] = new_col
    meta["canonical_fields"] = list(canonical)

    time = df["time_rel_s"].to_numpy(dtype=float)
    steer = df["steer"].to_numpy(dtype=float)
    df["steer_smooth"] = rolling_smooth(steer, time, float(config.get("smoothing_window_sec", 0.05)))
    df["steer_rate"] = gradient(df["steer_smooth"].to_numpy(dtype=float), time)
    if "lat_offset" in df.columns:
        df["lat_offset_rate"] = gradient(df["lat_offset"].to_numpy(dtype=float), time)
    else:
        df["lat_offset_rate"] = np.nan
    meta["duration_s"] = float(df["time_rel_s"].iloc[-1] - df["time_rel_s"].iloc[0])
    meta["row_count"] = int(len(df))
    dt = np.diff(time)
    good_dt = dt[np.isfinite(dt) & (dt > 0)]
    meta["median_dt_s"] = float(np.nanmedian(good_dt)) if good_dt.size else float("nan")
    meta["timestamp_gap_count_gt_0_1s"] = int(np.sum(good_dt > 0.1)) if good_dt.size else 0
    if meta["timestamp_gap_count_gt_0_1s"]:
        meta["warnings"].append("存在大于0.1秒的时间间隔")
    return df, meta


def window(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df[(df["time_rel_s"] >= start) & (df["time_rel_s"] <= end)].copy()


def max_abs(series: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmax(np.abs(arr))) if arr.size else float("nan")


def value_range(series: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmax(arr) - np.nanmin(arr)) if arr.size else float("nan")

