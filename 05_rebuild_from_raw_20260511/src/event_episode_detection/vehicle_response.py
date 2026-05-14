# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .signals import max_abs, robust_mad, robust_median, value_range, window


PRIMARY_DYNAMIC = ["ay", "yaw_rate", "roll_rate"]


def _signal_peak_z(vehicle: pd.DataFrame, col: str, base_window: pd.DataFrame, post_window: pd.DataFrame, floor: float) -> tuple[float, float, float]:
    if col not in vehicle or base_window.empty or post_window.empty:
        return float("nan"), float("nan"), float("nan")
    base = robust_median(base_window[col].to_numpy(dtype=float))
    scale = robust_mad(base_window[col].to_numpy(dtype=float) - base)
    scale = max(scale if math.isfinite(scale) else 0.0, floor)
    peak = max_abs(post_window[col] - base)
    z = peak / max(scale, 1e-9) if math.isfinite(peak) else float("nan")
    return peak, z, scale


def score_vehicle_response(vehicle: pd.DataFrame, episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    t0 = float(episode["t_steer_onset"])
    pre = window(vehicle, t0 - float(config.get("pre_window_sec", 2.0)), t0)
    post = window(vehicle, t0, t0 + float(config.get("vehicle_response_window_sec", 2.5)))
    if pre.empty or post.empty:
        return {
            "vehicle_response_status": "window_missing",
            "vehicle_dynamic_score": 0.0,
            "has_vehicle_response": False,
            "coordinate_continuity_ok": False,
        }

    floors = {
        "ay": 0.20,
        "yaw_rate": 0.025,
        "roll_rate": 0.020,
        "lat_offset_rate": 0.08,
        "speed": 1.0,
        "brake": 0.05,
        "mu": 0.02,
    }
    peaks: dict[str, float] = {}
    zscores: dict[str, float] = {}
    support_signals = 0
    primary_support = 0
    for col in ["ay", "yaw_rate", "roll_rate", "lat_offset_rate", "speed", "brake", "mu"]:
        if col not in vehicle:
            peaks[col] = float("nan")
            zscores[col] = float("nan")
            continue
        peak, z, _ = _signal_peak_z(vehicle, col, pre, post, floors[col])
        if col == "speed":
            if "speed" in post and "speed" in pre:
                post_speed = post["speed"].to_numpy(dtype=float)
                post_speed = post_speed[np.isfinite(post_speed)]
                pre_speed = robust_median(pre["speed"].to_numpy(dtype=float))
                if post_speed.size and math.isfinite(pre_speed):
                    peak = max(0.0, pre_speed - float(np.nanmin(post_speed)))
                    z = peak / max(floors[col], 1e-9)
                else:
                    peak = float("nan")
                    z = float("nan")
        if col == "mu":
            peak = value_range(post["mu"]) if "mu" in post else float("nan")
            z = peak / max(floors[col], 1e-9) if math.isfinite(peak) else float("nan")
        peaks[col] = peak
        zscores[col] = z
        if math.isfinite(z) and z >= float(config.get("vehicle_dynamic_z_threshold", 2.5)):
            support_signals += 1
            if col in PRIMARY_DYNAMIC:
                primary_support += 1

    t_vehicle_peak = float("nan")
    dyn_cols = [c for c in ["ay", "yaw_rate", "roll_rate"] if c in vehicle]
    if dyn_cols:
        composite = np.zeros(len(post), dtype=float)
        used = 0
        for col in dyn_cols:
            base = robust_median(pre[col].to_numpy(dtype=float))
            scale = robust_mad(pre[col].to_numpy(dtype=float) - base)
            scale = max(scale if math.isfinite(scale) else 0.0, floors[col])
            composite += np.abs(post[col].to_numpy(dtype=float) - base) / max(scale, 1e-9)
            used += 1
        if used and np.isfinite(composite).any():
            t_vehicle_peak = float(post["time_rel_s"].iloc[int(np.nanargmax(composite))])

    score_parts = [min(zscores.get(col, 0.0), 5.0) for col in ["ay", "yaw_rate", "roll_rate", "lat_offset_rate"] if math.isfinite(zscores.get(col, float("nan")))]
    vehicle_dynamic_score = float(np.mean(score_parts)) if score_parts else 0.0
    max_lat_step = float("nan")
    coordinate_ok = True
    coordinate_status = "not_available"
    if "lat_offset" in vehicle:
        local = window(vehicle, t0 - 0.5, t0 + float(config.get("vehicle_response_window_sec", 2.5)))
        lat = local["lat_offset"].to_numpy(dtype=float)
        lat = lat[np.isfinite(lat)]
        if lat.size > 2:
            max_lat_step = float(np.nanmax(np.abs(np.diff(lat))))
            coordinate_ok = bool(max_lat_step <= float(config.get("coordinate_jump_threshold_m", 1.5)))
            coordinate_status = "ok" if coordinate_ok else "suspicious_jump"
    has_vehicle_response = bool(
        vehicle_dynamic_score >= float(config.get("vehicle_dynamic_score_threshold", 2.0))
        and primary_support >= 1
        and support_signals >= int(config.get("vehicle_dynamic_min_support_signals", 2))
        and coordinate_ok
    )
    return {
        "vehicle_response_status": "ok",
        "vehicle_dynamic_score": vehicle_dynamic_score,
        "vehicle_dynamic_support_signals": support_signals,
        "vehicle_dynamic_primary_support_signals": primary_support,
        "t_vehicle_peak": t_vehicle_peak,
        "ay_peak_post": peaks.get("ay", np.nan),
        "yaw_rate_peak_post": peaks.get("yaw_rate", np.nan),
        "roll_rate_peak_post": peaks.get("roll_rate", np.nan),
        "lat_offset_change_post": value_range(post["lat_offset"]) if "lat_offset" in post else np.nan,
        "lat_offset_rate_peak_post": peaks.get("lat_offset_rate", np.nan),
        "mu_change_near_event": peaks.get("mu", np.nan),
        "brake_response_near_event": peaks.get("brake", np.nan),
        "speed_drop_near_event": peaks.get("speed", np.nan),
        "has_vehicle_response": has_vehicle_response,
        "coordinate_continuity_ok": coordinate_ok,
        "coordinate_continuity_status": coordinate_status,
        "max_lateral_step_near_event": max_lat_step,
    }
