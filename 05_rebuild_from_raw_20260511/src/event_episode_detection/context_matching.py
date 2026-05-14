# -*- coding: utf-8 -*-
from __future__ import annotations

import bisect
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Paths


def read_optional_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        return pd.DataFrame()


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.get("subject", row.get("subject_id", ""))), str(row.get("session_stamp", ""))


def _time_value(row: pd.Series, candidates: list[str]) -> float:
    for col in candidates:
        if col in row:
            value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            if math.isfinite(float(value)):
                return float(value)
    return float("nan")


def build_time_lookup(df: pd.DataFrame, time_cols: list[str]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if df.empty:
        return lookup
    for _, row in df.iterrows():
        key = _key(row)
        t = _time_value(row, time_cols)
        if not math.isfinite(t):
            continue
        payload = row.to_dict()
        payload["_match_time"] = t
        lookup.setdefault(key, []).append(payload)
    for key in list(lookup):
        lookup[key] = sorted(lookup[key], key=lambda x: float(x["_match_time"]))
    return lookup


def nearest_payload(lookup: dict[tuple[str, str], list[dict[str, Any]]], key: tuple[str, str], t: float) -> dict[str, Any]:
    items = lookup.get(key, [])
    if not items or not math.isfinite(t):
        return {}
    times = [float(item["_match_time"]) for item in items]
    pos = bisect.bisect_left(times, t)
    idxs = []
    if pos < len(times):
        idxs.append(pos)
    if pos > 0:
        idxs.append(pos - 1)
    if not idxs:
        return {}
    idx = min(idxs, key=lambda i: abs(times[i] - t))
    out = dict(items[idx])
    out["_delta_s"] = times[idx] - t
    out["_abs_delta_s"] = abs(times[idx] - t)
    return out


def active_module(module_df: pd.DataFrame, subject: str, session: str, t: float) -> dict[str, Any]:
    if module_df.empty or not math.isfinite(t):
        return {}
    sub = module_df[(module_df["subject"].astype(str) == subject) & (module_df["session_stamp"].astype(str) == session)].copy()
    if sub.empty:
        return {}
    start = pd.to_numeric(sub.get("entry_time_rel_s"), errors="coerce")
    end = pd.to_numeric(sub.get("exit_time_rel_s"), errors="coerce")
    active = sub[(start <= t) & (end >= t)]
    if not active.empty:
        row = active.iloc[0].to_dict()
        row["_active_delta_s"] = 0.0
        return row
    mid = (start + end) / 2.0
    idx = int(np.nanargmin(np.abs(mid.to_numpy(dtype=float) - t)))
    row = sub.iloc[idx].to_dict()
    row["_active_delta_s"] = float(mid.iloc[idx] - t)
    return row


class ContextMatcher:
    def __init__(self, paths: Paths):
        self.scene_triggers = read_optional_csv(paths.scene_triggers)
        self.old_anchors = read_optional_csv(paths.old_anchors)
        self.v05_candidates = read_optional_csv(paths.v05_candidates)
        self.module_segments = read_optional_csv(paths.module_segments)
        self.scene_lookup = build_time_lookup(self.scene_triggers, ["estimated_trigger_time_rel_s"])
        self.old_lookup = build_time_lookup(self.old_anchors, ["old_anchor_time_rel_s"])
        self.v05_lookup = build_time_lookup(self.v05_candidates, ["candidate_time_rel_s"])

    def append_context(self, episode: dict[str, Any]) -> dict[str, Any]:
        out = dict(episode)
        subject = str(out.get("subject_id", ""))
        session = str(out.get("session_stamp", ""))
        t = pd.to_numeric(pd.Series([out.get("t_steer_onset")]), errors="coerce").iloc[0]
        if not math.isfinite(float(t)):
            t = pd.to_numeric(pd.Series([out.get("nearest_aed_trigger_time")]), errors="coerce").iloc[0]
        t_float = float(t) if math.isfinite(float(t)) else float("nan")
        key = (subject, session)

        trig = nearest_payload(self.scene_lookup, key, t_float)
        old = nearest_payload(self.old_lookup, key, t_float)
        v05 = nearest_payload(self.v05_lookup, key, t_float)
        module = active_module(self.module_segments, subject, session, t_float)

        out["nearest_aed_trigger_time"] = trig.get("_match_time", out.get("nearest_aed_trigger_time", np.nan))
        out["nearest_aed_trigger_type"] = trig.get("trigger_name", out.get("nearest_aed_trigger_type", ""))
        out["delta_to_nearest_aed_trigger"] = trig.get("_delta_s", out.get("delta_to_nearest_aed_trigger", np.nan))
        out["nearest_aed_trigger_module"] = trig.get("module_name", "")
        out["nearest_aed_trigger_target"] = trig.get("target_title", "")

        out["nearest_old_anchor_time"] = old.get("_match_time", np.nan)
        out["delta_to_nearest_old_anchor"] = old.get("_delta_s", np.nan)
        out["nearest_old_anchor_level"] = old.get("old_event_level", "")
        out["nearest_old_anchor_phase"] = old.get("old_phase_type", "")

        out["nearest_v05_candidate_time"] = v05.get("_match_time", np.nan)
        out["delta_to_nearest_v05_candidate"] = v05.get("_delta_s", np.nan)
        out["nearest_v05_candidate_type"] = v05.get("candidate_anchor_type_cn", "")
        out["nearest_v05_candidate_decision"] = v05.get("screening_decision_cn", "")

        road_context = module.get("module_name", out.get("road_context", ""))
        out["road_context"] = road_context
        out["road_instance_name"] = module.get("instance_name", "")
        out["road_context_reliability"] = module.get("segment_mapping_reliability", "")
        out["road_context_delta_s"] = module.get("_active_delta_s", np.nan)
        road = str(road_context)
        out["is_curve_context"] = road in {"curve1", "curve2", "curve3"}
        out["is_low_mu_context"] = road == "differentmu_road"
        out["is_fix_road_context"] = road == "fix_road"
        out["is_middle_section_context"] = road == "middle_section"
        out["is_longstraight_context"] = road == "longstraight"
        out["is_stop_context"] = road == "stop"
        return out

    def trigger_count(self) -> int:
        return int(len(self.scene_triggers))

