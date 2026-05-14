# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .signals import max_abs, robust_mad, robust_median, value_range, window


def infer_subject_session(path: Path) -> tuple[str, str, str]:
    subject = path.parent.name
    name = path.stem
    session = name
    if name.startswith("Entity_Recording_") and name.endswith("_vehicle"):
        session = name[len("Entity_Recording_") : -len("_vehicle")]
    rel_hint = f"原始车辆数据/{subject}/{path.name}"
    return subject, session, rel_hint


def compute_steer_thresholds(vehicle: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    rate_abs = np.abs(vehicle["steer_rate"].to_numpy(dtype=float))
    rate_abs = rate_abs[np.isfinite(rate_abs)]
    if rate_abs.size == 0:
        return {
            "steer_rate_threshold": float("nan"),
            "steer_rate_percentile_value": float("nan"),
            "steer_rate_mad_value": float("nan"),
            "steer_delta_threshold": float(config.get("steer_delta_floor", 0.08)),
        }
    percentile_value = float(np.nanpercentile(rate_abs, float(config.get("steer_rate_percentile", 95))))
    med = float(np.nanmedian(rate_abs))
    mad = robust_mad(rate_abs)
    mad_value = med + float(config.get("steer_rate_mad_k", 4.0)) * (mad if math.isfinite(mad) else 0.0)
    threshold = max(percentile_value, mad_value, float(config.get("steer_rate_floor", 0.18)))

    steer = vehicle["steer_smooth"].to_numpy(dtype=float)
    dt = float(config.get("steer_delta_window_sec", 0.5))
    time = vehicle["time_rel_s"].to_numpy(dtype=float)
    sample_dt = np.nanmedian(np.diff(time)) if len(time) > 2 else 0.005
    step = max(2, int(round(dt / max(sample_dt, 1e-3))))
    if len(steer) > step:
        deltas = np.abs(steer[step:] - steer[:-step])
        delta_mad = robust_mad(deltas)
        delta_med = robust_median(deltas)
        delta_thr = max(
            float(config.get("steer_delta_floor", 0.08)),
            (delta_med if math.isfinite(delta_med) else 0.0)
            + float(config.get("steer_delta_mad_k", 4.0)) * (delta_mad if math.isfinite(delta_mad) else 0.0),
        )
    else:
        delta_thr = float(config.get("steer_delta_floor", 0.08))
    return {
        "steer_rate_threshold": threshold,
        "steer_rate_percentile_value": percentile_value,
        "steer_rate_mad_value": mad_value,
        "steer_delta_threshold": delta_thr,
    }


def detect_steering_episodes(
    vehicle: pd.DataFrame,
    record_meta: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if vehicle.empty or "steer_rate" not in vehicle or "steer_smooth" not in vehicle:
        return []
    thresholds = compute_steer_thresholds(vehicle, config)
    rate_thr = thresholds["steer_rate_threshold"]
    delta_thr = thresholds["steer_delta_threshold"]
    if not math.isfinite(rate_thr):
        return []

    time = vehicle["time_rel_s"].to_numpy(dtype=float)
    rate = vehicle["steer_rate"].to_numpy(dtype=float)
    steer = vehicle["steer_smooth"].to_numpy(dtype=float)
    abs_rate = np.abs(rate)
    candidate_idx = np.where(abs_rate >= rate_thr)[0]
    if candidate_idx.size == 0:
        return []
    split_points = np.where(np.diff(candidate_idx) > 1)[0] + 1
    segments = np.split(candidate_idx, split_points)
    segment_indices: list[int] = []
    for segment in segments:
        if segment.size == 0:
            continue
        segment_indices.append(int(segment[0]))

    pre_window_sec = float(config.get("pre_window_sec", 2.0))
    early_sec = float(config.get("early_observation_sec", 0.5))
    correction_sec = float(config.get("correction_window_sec", 5.0))
    stable_sec = float(config.get("pre_stable_window_sec", 0.8))
    stable_rate_thr = rate_thr * float(config.get("pre_stable_rate_ratio", 0.55))
    min_gap = float(config.get("min_episode_gap_sec", 1.5))

    episodes: list[dict[str, Any]] = []
    last_kept_time = -1e9
    last_checked_time = -1e9
    episode_index = 0
    for idx in segment_indices:
        t0 = float(time[idx])
        if t0 - last_checked_time < min_gap:
            continue
        last_checked_time = t0
        if t0 < pre_window_sec or t0 + correction_sec > float(time[-1]):
            continue
        if t0 - last_kept_time < min_gap:
            continue
        pre = vehicle[(vehicle["time_rel_s"] >= t0 - stable_sec) & (vehicle["time_rel_s"] < t0)].copy()
        early = window(vehicle, t0, t0 + early_sec)
        if pre.empty or early.empty:
            continue
        pre_max_rate = max_abs(pre["steer_rate"])
        if math.isfinite(pre_max_rate) and pre_max_rate > stable_rate_thr:
            continue
        steer_baseline = robust_median(pre["steer_smooth"].to_numpy(dtype=float))
        early_delta_signed = float(early["steer_smooth"].iloc[-1] - steer_baseline)
        early_delta_abs = max_abs(early["steer_smooth"] - steer_baseline)
        if not math.isfinite(early_delta_abs) or early_delta_abs < delta_thr:
            continue
        rate_peak_early = max_abs(early["steer_rate"])
        onset_rate = float(np.nanmedian(rate[max(0, idx - 2) : min(len(rate), idx + 3)]))
        event_sign = 1 if onset_rate >= 0 else -1
        rate_z = rate_peak_early / max(rate_thr, 1e-9)
        delta_z = early_delta_abs / max(delta_thr, 1e-9)
        steering_impulse_score = float(0.6 * rate_z + 0.4 * delta_z)
        subject = str(record_meta.get("subject", ""))
        session = str(record_meta.get("session_stamp", ""))
        episode_id = f"steer_episode_v0_6__{subject}__{session}__{episode_index:05d}"
        episodes.append(
            {
                "episode_id": episode_id,
                "row_source": "steering_onset_scan",
                "record_id": session,
                "subject_id": subject,
                "session_stamp": session,
                "vehicle_raw_absolute_path": record_meta.get("vehicle_path", ""),
                "vehicle_raw_relative_path": record_meta.get("vehicle_raw_relative_path", ""),
                "t_steer_onset": t0,
                "t_obs_start": t0 - pre_window_sec,
                "t_obs_end": t0 + early_sec,
                "t_label_start": t0 + early_sec,
                "t_label_end": t0 + correction_sec,
                "steer_baseline": steer_baseline,
                "steer_onset_sign": event_sign,
                "steer_rate_at_onset": float(rate[idx]),
                "steer_rate_peak_early": rate_peak_early,
                "steer_delta_early": early_delta_signed,
                "steer_delta_abs_early": early_delta_abs,
                "steering_impulse_score": steering_impulse_score,
                "steer_rate_threshold": rate_thr,
                "steer_delta_threshold": delta_thr,
                "pre_steady_max_abs_rate": pre_max_rate,
                "pre_window_complete": True,
                "label_window_complete": True,
                "recommended_target_type": "relative_steer_delta",
                "direction_normalization_available": True,
            }
        )
        last_kept_time = t0
        episode_index += 1
    return episodes


def build_trigger_no_effect_rows(
    triggers: pd.DataFrame,
    steering_rows: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if triggers.empty:
        return pd.DataFrame()
    match_window = float(config.get("context_match_window_sec", 2.0))
    out: list[dict[str, Any]] = []
    for _, trig in triggers.iterrows():
        t = pd.to_numeric(pd.Series([trig.get("estimated_trigger_time_rel_s")]), errors="coerce").iloc[0]
        if not math.isfinite(float(t)):
            continue
        subject = str(trig.get("subject", ""))
        session = str(trig.get("session_stamp", ""))
        same = steering_rows[
            (steering_rows["subject_id"].astype(str) == subject)
            & (steering_rows["session_stamp"].astype(str) == session)
        ]
        nearest_delta = float("nan")
        if not same.empty:
            deltas = pd.to_numeric(same["t_steer_onset"], errors="coerce") - float(t)
            if deltas.notna().any():
                nearest_delta = float(deltas.iloc[np.argmin(np.abs(deltas.to_numpy(dtype=float)))])
        if math.isfinite(nearest_delta) and abs(nearest_delta) <= match_window:
            continue
        out.append(
            {
                "episode_id": f"trigger_no_effect_v0_6__{subject}__{session}__{len(out):05d}",
                "row_source": "trigger_context_without_steering_episode",
                "record_id": session,
                "subject_id": subject,
                "session_stamp": session,
                "vehicle_raw_relative_path": trig.get("vehicle_raw_relative_path", ""),
                "t_steer_onset": np.nan,
                "t_obs_start": np.nan,
                "t_obs_end": np.nan,
                "t_label_start": np.nan,
                "t_label_end": np.nan,
                "nearest_aed_trigger_time": float(t),
                "nearest_aed_trigger_type": trig.get("trigger_name", ""),
                "delta_to_nearest_aed_trigger": 0.0,
                "road_context": trig.get("module_name", ""),
                "episode_class": "N_trigger_no_effect_or_no_response",
                "class_reason_cn": "存在场景触发，但附近没有检测到方向盘快速动作 episode",
                "recommended_target_type": "",
                "direction_normalization_available": False,
            }
        )
    return pd.DataFrame(out)
