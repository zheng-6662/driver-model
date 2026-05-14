# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .signals import robust_median, window


def score_correction(vehicle: pd.DataFrame, episode: dict[str, Any], vehicle_features: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    t0 = float(episode["t_steer_onset"])
    sign = int(episode.get("steer_onset_sign", 1)) or 1
    baseline = float(episode.get("steer_baseline", np.nan))
    win = window(vehicle, t0, t0 + float(config.get("correction_window_sec", 5.0)))
    if win.empty or "steer_smooth" not in win or not math.isfinite(baseline):
        return {"correction_status": "window_missing", "correction_score": 0.0, "has_correction": False}
    rel = sign * (win["steer_smooth"].to_numpy(dtype=float) - baseline)
    times = win["time_rel_s"].to_numpy(dtype=float)
    if rel.size == 0 or not np.isfinite(rel).any():
        return {"correction_status": "bad_signal", "correction_score": 0.0, "has_correction": False}
    peak_idx = int(np.nanargmax(rel))
    peak_value = float(win["steer_smooth"].iloc[peak_idx])
    peak_delta = float(rel[peak_idx])
    t_peak = float(times[peak_idx])
    after = rel[peak_idx:]
    after_times = times[peak_idx:]
    if after.size < 5 or peak_delta <= 0:
        return {
            "correction_status": "weak_peak",
            "t_steer_peak": t_peak,
            "steer_peak_value": peak_value,
            "steer_peak_delta_from_baseline": peak_delta,
            "correction_score": 0.0,
            "has_correction": False,
        }

    return_ratio = float(config.get("correction_return_ratio", 0.45))
    has_return = bool(np.nanmin(after) <= peak_delta * return_ratio)
    counter_ratio = float(config.get("countersteer_ratio", 0.20))
    has_counter = bool(np.nanmin(after) <= -abs(peak_delta) * counter_ratio)
    correction_onset = float("nan")
    below = np.where(after <= peak_delta * return_ratio)[0]
    if below.size:
        correction_onset = float(after_times[int(below[0])])

    dynamic_decay = False
    if "ay" in vehicle or "yaw_rate" in vehicle or "roll_rate" in vehicle:
        pre_dyn = window(vehicle, t0, min(t_peak + 0.2, t0 + 2.0))
        post_dyn = window(vehicle, t_peak + 0.3, t0 + float(config.get("correction_window_sec", 5.0)))
        dyn_cols = [c for c in ["ay", "yaw_rate", "roll_rate"] if c in vehicle]
        if dyn_cols and not pre_dyn.empty and not post_dyn.empty:
            pre_level = np.nanmean([np.nanmax(np.abs(pre_dyn[c].to_numpy(dtype=float))) for c in dyn_cols])
            post_level = np.nanmean([np.nanmedian(np.abs(post_dyn[c].to_numpy(dtype=float))) for c in dyn_cols])
            dynamic_decay = bool(math.isfinite(pre_level) and math.isfinite(post_level) and post_level <= pre_level * 0.75)

    score = 0.0
    if has_return:
        score += 1.0
    if has_counter:
        score += 1.0
    if dynamic_decay:
        score += 0.8
    if peak_delta >= max(float(episode.get("steer_delta_threshold", 0.08)) * 2.0, 0.16):
        score += 0.4
    has_correction = bool(score >= float(config.get("correction_score_threshold", 1.5)))
    return {
        "correction_status": "ok",
        "t_steer_peak": t_peak,
        "steer_peak_value": peak_value,
        "steer_peak_delta_from_baseline": peak_delta,
        "t_correction_onset": correction_onset,
        "has_return_to_baseline": has_return,
        "has_countersteer": has_counter,
        "vehicle_dynamics_decay_after_correction": dynamic_decay,
        "correction_score": float(score),
        "has_correction": has_correction,
    }

