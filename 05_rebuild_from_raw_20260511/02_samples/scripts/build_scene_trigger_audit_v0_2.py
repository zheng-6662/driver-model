# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
from collections import Counter
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

PREV_AUDIT_DIR = ROOT / "02_samples" / "road_event_anchor_audit_v0_1"
PREV_TABLE_DIR = PREV_AUDIT_DIR / "tables"

AUDIT_DIR = ROOT / "02_samples" / "scene_trigger_audit_v0_2"
TABLE_DIR = AUDIT_DIR / "tables"
FIGURE_DIR = AUDIT_DIR / "figures"
REPORT_DIR = ROOT / "09_reports"

ROAD_POSITION_MAP = PREV_TABLE_DIR / "road_event_position_map_v0_1.csv"
SESSION_MODULE_TIMES = PREV_TABLE_DIR / "session_module_entry_exit_v0_1.csv"
OLD_ANCHORS = PREV_TABLE_DIR / "old_new_anchor_alignment_v0_1.csv"
AED_DIR = (
    PROJECT_ROOT
    / "01_datasets"
    / "多模态数据"
    / "被试数据集合"
    / "道路信息"
    / "道路"
)
DATA_PREP_DIR = PROJECT_ROOT / "01_datasets" / "数据预处理"


AED_MODULE_MAP = {
    "longsrtaight.autosave.1.aed": "longstraight",
    "curve1.autosave.2.aed": "curve1",
    "curve2.autosave.1.aed": "curve2",
    "curve3.autosave.2.aed": "curve3",
    "differentmu_road.aed": "differentmu_road",
    "fix_road.autosave.1.aed": "fix_road",
    "middle_section.autosave.1.autosave.1.aed": "middle_section",
    "stop.autosave.1.aed": "stop",
    "zd.autosave.1.autosave.2.aed": "zd",
}
CFG_MODULE_MAP = {
    "curve1": "curve1_Area2.cfg",
    "curve2": "curve2_Area2.cfg",
    "curve3": "curve3_Area2.cfg",
    "differentmu_road": "differentmu_road_Area2.cfg",
    "fix_road": "fix_road_Area2.cfg",
    "longstraight": "longsrtaight_Area2.cfg",
    "middle_section": "middle_section_Area2.cfg",
    "stop": "stop_Area2.cfg",
    "zd": "zd_Area2.cfg",
}

TRAFFIC_OBJECT_KIND = "AR2FTrafficRoadUser"
TRAFFIC_TRIGGER_KIND = "AR2FTrafficFlowPoint"
LONGSTRAIGHT_LANE_CENTERS_M = {
    "21": 8.5,
    "22": 6.0,
    "23": 2.5,
    "24": 0.0,
    "25": -2.5,
    "26": -6.0,
    "27": -8.5,
}
LONGSTRAIGHT_LANE_DIRECTIONS = {
    "21": "forward_side",
    "22": "forward_side",
    "23": "forward_side",
    "24": "center_buffer",
    "25": "opposite_side",
    "26": "opposite_side",
    "27": "opposite_side",
}


def configure_matplotlib_fonts() -> None:
    font_candidates = [
        Path(r"C:/Windows/Fonts/msyh.ttc"),
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
    for path in [TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def clean_value(value: str | None) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    return value


def float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def parse_lane_geometry() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for module_name, cfg_name in CFG_MODULE_MAP.items():
        path = AED_DIR / cfg_name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"(?:Straight|Circle)\s+l(\d+)\s*\{(?P<body>.*?)\n\s*\};", text, re.S):
            lane_id = match.group(1)
            body = match.group("body")
            dist_match = re.search(r"DistToMid0\s*=\s*([-\d.eE]+)", body)
            if not dist_match:
                dist_match = re.search(r"DistToRef0\s*=\s*([-\d.eE]+)", body)
            width_match = re.search(r"Width0\s*=\s*([-\d.eE]+)", body)
            direction_match = re.search(r"Direction\s*=\s*([-\d.eE]+)", body)
            dist = float_or_nan(dist_match.group(1) if dist_match else np.nan)
            width = float_or_nan(width_match.group(1) if width_match else np.nan)
            direction_raw = clean_value(direction_match.group(1)) if direction_match else ""
            direction_group = "direction_1" if direction_raw == "1" else "direction_0"
            signed_center = dist if direction_raw == "1" else -dist
            if abs(dist) < 1e-9:
                signed_center = 0.0
                direction_group = "center_or_separator"
            rows.append(
                {
                    "module_name": module_name,
                    "lane_id": lane_id,
                    "lane_width_m": width,
                    "dist_to_mid_m": dist,
                    "signed_center_offset_m": signed_center,
                    "direction_raw": direction_raw,
                    "direction_group": direction_group,
                    "source_cfg": str(path),
                }
            )
    return pd.DataFrame(rows)


def load_projection_layout(road_map: pd.DataFrame) -> pd.DataFrame:
    source_paths = [p for p in road_map.get("source_layout_path", pd.Series(dtype=str)).dropna().astype(str).unique() if p]
    for raw in source_paths:
        path = Path(raw)
        if path.exists():
            layout = pd.read_csv(path, low_memory=False)
            break
    else:
        candidates = sorted((PROJECT_ROOT / "01_datasets").rglob("full_centerline_layout.csv"))
        if not candidates:
            return pd.DataFrame()
        layout = pd.read_csv(candidates[0], low_memory=False)
    required = {"s", "x", "y", "module_name", "instance_name"}
    if not required.issubset(layout.columns):
        return pd.DataFrame()
    layout = layout.copy()
    for col in ["s", "x", "y"]:
        layout[col] = pd.to_numeric(layout[col], errors="coerce")
    layout = layout.dropna(subset=["s", "x", "y"]).sort_values("s").reset_index(drop=True)
    dx = np.gradient(layout["x"].to_numpy(dtype=float))
    dy = np.gradient(layout["y"].to_numpy(dtype=float))
    norm = np.hypot(dx, dy)
    norm[norm < 1e-9] = 1.0
    layout["tangent_x"] = dx / norm
    layout["tangent_y"] = dy / norm
    return layout


def extract_braced_block(text: str, start_pos: int) -> str:
    brace_start = text.find("{", start_pos)
    if brace_start < 0:
        return ""
    depth = 0
    for i in range(brace_start, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : i]
    return text[brace_start + 1 :]


def find_layer_body(text: str, layer_name: str = "l5") -> str:
    match = re.search(rf"\bLayer\s+{re.escape(layer_name)}\s*\{{", text)
    if not match:
        return ""
    return extract_braced_block(text, match.start())


def iter_figure_blocks(layer_body: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pos = 0
    while True:
        match = re.search(r"\bFigure\s+(f\d+)\s*\{", layer_body[pos:])
        if not match:
            break
        fig_name = match.group(1)
        abs_start = pos + match.start()
        block = extract_braced_block(layer_body, abs_start)
        out.append((fig_name, block))
        # Move past this block. The +1 is enough because extract_braced_block
        # returns only content; a later regex search will skip nested text.
        brace_start = layer_body.find("{", abs_start)
        depth = 0
        end_pos = brace_start
        for i in range(brace_start, len(layer_body)):
            if layer_body[i] == "{":
                depth += 1
            elif layer_body[i] == "}":
                depth -= 1
                if depth == 0:
                    end_pos = i + 1
                    break
        pos = max(end_pos, abs_start + 1)
    return out


def kv_from_block(block: str) -> dict[str, str]:
    kv: dict[str, str] = {}
    for line in block.splitlines():
        if "=" not in line:
            continue
        key, raw = line.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if raw.endswith(";"):
            raw = raw[:-1]
        if key:
            kv[key] = clean_value(raw)
    return kv


def args_from_kv(kv: dict[str, str], opt: bool = False) -> dict[str, str]:
    prefix_a = "OptArgument" if opt else "Argument"
    prefix_v = "OptValue" if opt else "Value"
    out: dict[str, str] = {}
    for key, arg_name in kv.items():
        if not key.startswith(prefix_a):
            continue
        idx = key[len(prefix_a) :]
        value = kv.get(f"{prefix_v}{idx}", "")
        out[arg_name] = value
    return out


def parse_aed_file(path: Path, road_map: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    module_name = AED_MODULE_MAP.get(path.name, path.stem.split(".")[0])
    text = path.read_text(encoding="utf-8", errors="ignore")
    layer = find_layer_body(text, "l5")
    if not layer:
        return [], []

    module_instances = road_map[road_map["module_name"].astype(str).eq(module_name)].copy()
    if module_instances.empty:
        module_instances = pd.DataFrame(
            [
                {
                    "module_name": module_name,
                    "instance_name": module_name,
                    "s_start_m": np.nan,
                    "s_end_m": np.nan,
                    "length_m": np.nan,
                }
            ]
        )

    raw_objects: list[dict[str, Any]] = []
    raw_triggers: list[dict[str, Any]] = []
    object_by_id: dict[str, dict[str, Any]] = {}

    for fig_name, block in iter_figure_blocks(layer):
        kind_match = re.search(r"#\s*(AR2F\w+)", block)
        kind = kind_match.group(1) if kind_match else ""
        if kind not in {TRAFFIC_OBJECT_KIND, TRAFFIC_TRIGGER_KIND}:
            continue
        kv = kv_from_block(block)
        fig_id = kv.get("ID", "")
        args = args_from_kv(kv, opt=False)
        opt_args = args_from_kv(kv, opt=True)
        common = {
            "aed_file": str(path),
            "aed_file_name": path.name,
            "module_name": module_name,
            "figure_name": fig_name,
            "figure_id": fig_id,
            "kind": kind,
            "raw_name": kv.get("Name", ""),
            "title": kv.get("Title", ""),
            "category": kv.get("Category", ""),
            "group": kv.get("Group", ""),
            "behaviour_scheme": kv.get("BehaviourScheme", ""),
            "behaviour_category": kv.get("BehaviourSchemeCategory", ""),
            "arguments_json": json.dumps(args, ensure_ascii=False),
            "opt_arguments_json": json.dumps(opt_args, ensure_ascii=False),
        }
        if kind == TRAFFIC_OBJECT_KIND:
            tau = float_or_nan(kv.get("tau", ""))
            row = {
                **common,
                "lane_id": kv.get("laneid", ""),
                "tau": tau,
                "length": float_or_nan(kv.get("Length", "")),
                "width": float_or_nan(kv.get("Width", "")),
                "number_of_vehicles": args.get("NumberOfVehicles", ""),
                "vehicle_type": args.get("Vehicle", ""),
                "distance_between_vehicles": args.get("Distance", ""),
                "ego_distance": args.get("EGODistance", ""),
                "start_speed": opt_args.get("StartSpeed", ""),
                "target_speed": opt_args.get("TargetSpeed", ""),
                "user_id": opt_args.get("UserID", ""),
            }
            raw_objects.append(row)
            if fig_id:
                object_by_id[fig_id] = row
        elif kind == TRAFFIC_TRIGGER_KIND:
            tau = float_or_nan(kv.get("tau1", kv.get("tau", "")))
            road_user_id = kv.get("roaduserid", "")
            target = object_by_id.get(road_user_id, {})
            row = {
                **common,
                "lane_id": kv.get("laneid1", kv.get("laneid", "")),
                "tau": tau,
                "road_user_id": road_user_id,
                "mode": kv.get("mode", ""),
                "param": kv.get("param", ""),
                "param2": kv.get("param2", ""),
                "description": kv.get("Description", ""),
                "target_figure_name": target.get("figure_name", ""),
                "target_name": target.get("raw_name", ""),
                "target_title": target.get("title", ""),
                "target_lane_id": target.get("lane_id", ""),
                "target_tau": target.get("tau", np.nan),
                "target_behaviour_scheme": target.get("behaviour_scheme", ""),
                "target_start_speed": target.get("start_speed", ""),
                "target_target_speed": target.get("target_speed", ""),
                "change_target_lane": args.get("Lanenumber", ""),
                "change_time_or_distance": args.get("Time[s]/Distance[m]", ""),
                "change_mode_path_or_time": args.get("Mode:0=Path/1=Time", ""),
                "change_direction": args.get("Direction:-1=Left/1=Right", ""),
            }
            raw_triggers.append(row)

    objects = expand_to_module_instances(raw_objects, module_instances)
    triggers = expand_to_module_instances(raw_triggers, module_instances)
    return objects, triggers


def expand_to_module_instances(rows: list[dict[str, Any]], instances: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        tau = float_or_nan(row.get("tau", np.nan))
        for _, inst in instances.iterrows():
            s_start = float_or_nan(inst.get("s_start_m", np.nan))
            length = float_or_nan(inst.get("length_m", np.nan))
            s_abs = s_start + tau * length if math.isfinite(s_start) and math.isfinite(length) and math.isfinite(tau) else np.nan
            expanded = {
                **row,
                "instance_name": str(inst.get("instance_name", row.get("module_name", ""))),
                "road_s_module_start_m": s_start,
                "road_s_module_end_m": float_or_nan(inst.get("s_end_m", np.nan)),
                "road_module_length_m": length,
                "relative_s_in_module_m": tau * length if math.isfinite(tau) and math.isfinite(length) else np.nan,
                "road_s_global_m": s_abs,
            }
            out.append(expanded)
    return out


def parse_all_aed(road_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_objects: list[dict[str, Any]] = []
    all_triggers: list[dict[str, Any]] = []
    for file_name in AED_MODULE_MAP:
        path = AED_DIR / file_name
        if not path.exists():
            continue
        objects, triggers = parse_aed_file(path, road_map)
        all_objects.extend(objects)
        all_triggers.extend(triggers)
    objects_df = pd.DataFrame(all_objects)
    triggers_df = pd.DataFrame(all_triggers)
    return objects_df, triggers_df


def estimate_trigger_times(triggers: pd.DataFrame, session_segments: pd.DataFrame) -> pd.DataFrame:
    if triggers.empty or session_segments.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    flow = triggers[triggers["kind"].eq(TRAFFIC_TRIGGER_KIND)].copy()
    for _, trig in flow.iterrows():
        inst = str(trig.get("instance_name", ""))
        s_abs = float_or_nan(trig.get("road_s_global_m", np.nan))
        if not inst or not math.isfinite(s_abs):
            continue
        segs = session_segments[session_segments["instance_name"].astype(str).eq(inst)].copy()
        if segs.empty:
            continue
        for _, seg in segs.iterrows():
            s0 = float_or_nan(seg.get("road_s_start_m", np.nan))
            s1 = float_or_nan(seg.get("road_s_end_m", np.nan))
            t0 = float_or_nan(seg.get("entry_time_rel_s", np.nan))
            t1 = float_or_nan(seg.get("exit_time_rel_s", np.nan))
            if not all(math.isfinite(v) for v in [s0, s1, t0, t1]) or abs(s1 - s0) < 1e-6:
                continue
            s_min = min(s0, s1) - 2.0
            s_max = max(s0, s1) + 2.0
            if not (s_min <= s_abs <= s_max):
                continue
            alpha = (s_abs - s0) / (s1 - s0)
            if not (-0.05 <= alpha <= 1.05):
                continue
            trigger_time = t0 + alpha * (t1 - t0)
            rows.append(
                {
                    "scene_trigger_uid": f"scene_trigger__{seg.get('subject')}__{seg.get('session_stamp')}__{trig.get('instance_name')}__{trig.get('figure_name')}__{trig.get('raw_name')}",
                    "subject": seg.get("subject", ""),
                    "session_stamp": seg.get("session_stamp", ""),
                    "vehicle_raw_relative_path": seg.get("vehicle_raw_relative_path", ""),
                    "module_name": trig.get("module_name", ""),
                    "instance_name": trig.get("instance_name", ""),
                    "trigger_figure_name": trig.get("figure_name", ""),
                    "trigger_figure_id": trig.get("figure_id", ""),
                    "trigger_name": trig.get("raw_name", ""),
                    "trigger_description": trig.get("description", ""),
                    "trigger_lane_id": trig.get("lane_id", ""),
                    "trigger_tau": trig.get("tau", np.nan),
                    "trigger_road_s_global_m": s_abs,
                    "trigger_relative_s_in_module_m": trig.get("relative_s_in_module_m", np.nan),
                    "estimated_trigger_time_rel_s": round(float(trigger_time), 6),
                    "segment_entry_time_rel_s": t0,
                    "segment_exit_time_rel_s": t1,
                    "segment_mapping_reliability": seg.get("segment_mapping_reliability", ""),
                    "segment_median_nearest_dist_m": seg.get("median_nearest_dist_m", np.nan),
                    "target_name": trig.get("target_name", ""),
                    "target_title": trig.get("target_title", ""),
                    "target_lane_id": trig.get("target_lane_id", ""),
                    "target_figure_name": trig.get("target_figure_name", ""),
                    "change_target_lane": trig.get("change_target_lane", ""),
                    "change_time_or_distance": trig.get("change_time_or_distance", ""),
                    "change_mode_path_or_time": trig.get("change_mode_path_or_time", ""),
                    "change_direction": trig.get("change_direction", ""),
                }
            )
    return pd.DataFrame(rows)


def compare_old_anchors_to_scene(old: pd.DataFrame, scene_times: pd.DataFrame) -> pd.DataFrame:
    if old.empty or scene_times.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped = {
        (str(k[0]), str(k[1])): g.sort_values("estimated_trigger_time_rel_s").reset_index(drop=True)
        for k, g in scene_times.groupby(["subject", "session_stamp"], dropna=False)
    }
    for _, anchor in old.iterrows():
        key = (str(anchor.get("subject", "")), str(anchor.get("session_stamp", "")))
        candidates = grouped.get(key)
        t = float_or_nan(anchor.get("old_anchor_time_rel_s", np.nan))
        if candidates is None or candidates.empty or not math.isfinite(t):
            nearest = {}
            delta = np.nan
        else:
            times = pd.to_numeric(candidates["estimated_trigger_time_rel_s"], errors="coerce").to_numpy()
            idx = int(np.nanargmin(np.abs(times - t)))
            nearest = candidates.iloc[idx].to_dict()
            delta = t - float(times[idx])
        bucket = delta_bucket(delta)
        rows.append(
            {
                "old_event_uid": anchor.get("old_event_uid", ""),
                "subject": anchor.get("subject", ""),
                "session_stamp": anchor.get("session_stamp", ""),
                "old_anchor_time_rel_s": t,
                "old_phase_type": anchor.get("old_phase_type", ""),
                "old_event_level": anchor.get("old_event_level", ""),
                "old_road_type_anchor": anchor.get("old_road_type_anchor", ""),
                "old_trigger_type": anchor.get("old_trigger_type", ""),
                "old_anchor_audit_bucket": anchor.get("old_anchor_audit_bucket", ""),
                "active_module_name_at_old_anchor": anchor.get("active_module_name_at_old_anchor", ""),
                "active_instance_name_at_old_anchor": anchor.get("active_instance_name_at_old_anchor", ""),
                "nearest_scene_trigger_uid": nearest.get("scene_trigger_uid", ""),
                "nearest_scene_trigger_time_rel_s": nearest.get("estimated_trigger_time_rel_s", np.nan),
                "nearest_scene_trigger_delta_s": delta,
                "nearest_scene_trigger_bucket": bucket,
                "nearest_scene_instance_name": nearest.get("instance_name", ""),
                "nearest_scene_trigger_name": nearest.get("trigger_name", ""),
                "nearest_scene_trigger_lane_id": nearest.get("trigger_lane_id", ""),
                "nearest_scene_trigger_s_global_m": nearest.get("trigger_road_s_global_m", np.nan),
                "nearest_scene_target_name": nearest.get("target_name", ""),
                "nearest_scene_target_title": nearest.get("target_title", ""),
                "nearest_scene_target_lane_id": nearest.get("target_lane_id", ""),
                "nearest_scene_change_target_lane": nearest.get("change_target_lane", ""),
                "nearest_scene_segment_reliability": nearest.get("segment_mapping_reliability", ""),
                "within_0p5s_scene_trigger": bool(math.isfinite(delta) and abs(delta) <= 0.5),
                "within_1s_scene_trigger": bool(math.isfinite(delta) and abs(delta) <= 1.0),
                "within_2s_scene_trigger": bool(math.isfinite(delta) and abs(delta) <= 2.0),
            }
        )
    return pd.DataFrame(rows)


def load_vehicle_track(relative_path: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if relative_path in cache:
        return cache[relative_path]
    path = DATA_PREP_DIR / str(relative_path)
    if not path.exists():
        cache[relative_path] = pd.DataFrame()
        return cache[relative_path]
    usecols = ["StorageTime", "zx|x", "zx|y", "zx1|lateraldistance"]
    try:
        df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    except Exception:  # noqa: BLE001
        cache[relative_path] = pd.DataFrame()
        return cache[relative_path]
    parsed = pd.to_datetime(df["StorageTime"], errors="coerce")
    if parsed.notna().sum() == 0:
        cache[relative_path] = pd.DataFrame()
        return cache[relative_path]
    t0 = parsed.dropna().iloc[0]
    df["time_rel_s"] = (parsed - t0).dt.total_seconds()
    for col in ["zx|x", "zx|y", "zx1|lateraldistance"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_rel_s", "zx|x", "zx|y"]).sort_values("time_rel_s").reset_index(drop=True)
    cache[relative_path] = df
    return df


def nearest_longstraight_lane(signed_lateral_m: float) -> tuple[str, float, str]:
    if not math.isfinite(signed_lateral_m):
        return "", float("nan"), ""
    lane_id, lane_center = min(
        LONGSTRAIGHT_LANE_CENTERS_M.items(),
        key=lambda item: abs(float(item[1]) - signed_lateral_m),
    )
    return lane_id, abs(float(lane_center) - signed_lateral_m), LONGSTRAIGHT_LANE_DIRECTIONS.get(lane_id, "")


def nearest_lane_from_geometry(
    lane_geometry: pd.DataFrame,
    module_name: str,
    signed_lateral_m: float,
) -> tuple[str, float, str]:
    if not math.isfinite(signed_lateral_m):
        return "", float("nan"), ""
    lanes = lane_geometry[lane_geometry["module_name"].astype(str).eq(str(module_name))].copy()
    if lanes.empty:
        return "", float("nan"), ""
    centers = pd.to_numeric(lanes["signed_center_offset_m"], errors="coerce")
    valid = centers.notna()
    if not valid.any():
        return "", float("nan"), ""
    lanes = lanes.loc[valid].copy()
    centers = centers.loc[valid]
    idx = (centers - signed_lateral_m).abs().idxmin()
    row = lanes.loc[idx]
    return (
        str(row.get("lane_id", "")),
        float(abs(float(row.get("signed_center_offset_m", np.nan)) - signed_lateral_m)),
        str(row.get("direction_group", "")),
    )


def trigger_lane_direction(lane_geometry: pd.DataFrame, module_name: str, lane_id: Any) -> tuple[float, str]:
    lanes = lane_geometry[
        lane_geometry["module_name"].astype(str).eq(str(module_name))
        & lane_geometry["lane_id"].astype(str).eq(str(lane_id))
    ]
    if lanes.empty:
        return float("nan"), ""
    row = lanes.iloc[0]
    return float_or_nan(row.get("signed_center_offset_m", np.nan)), str(row.get("direction_group", ""))


def build_layout_index(layout: pd.DataFrame) -> Any:
    if layout.empty:
        return None
    pts = layout[["x", "y"]].to_numpy(dtype=float)
    if cKDTree is not None:
        return cKDTree(pts)
    return pts


def nearest_layout_row(layout: pd.DataFrame, index: Any, x: float, y: float) -> tuple[pd.Series | None, float]:
    if layout.empty or index is None or not math.isfinite(x) or not math.isfinite(y):
        return None, float("nan")
    if cKDTree is not None and hasattr(index, "query"):
        dist, idx = index.query([x, y], k=1)
        return layout.iloc[int(idx)], float(dist)
    pts = index
    d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
    idx = int(np.nanargmin(d))
    return layout.iloc[idx], float(d[idx])


def estimate_ego_lane_at_longstraight_triggers(scene_times: pd.DataFrame) -> pd.DataFrame:
    if scene_times.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    cache: dict[str, pd.DataFrame] = {}
    long_rows = scene_times[scene_times["instance_name"].astype(str).eq("longstraight")].copy()
    for _, trig in long_rows.iterrows():
        rel_path = str(trig.get("vehicle_raw_relative_path", ""))
        track = load_vehicle_track(rel_path, cache)
        t = float_or_nan(trig.get("estimated_trigger_time_rel_s", np.nan))
        if track.empty or not math.isfinite(t):
            rows.append({**trig.to_dict(), "ego_lane_status": "no_vehicle_track"})
            continue
        idx = int((track["time_rel_s"] - t).abs().idxmin())
        sample = track.loc[idx]
        time_gap = float(sample["time_rel_s"] - t)
        x = float_or_nan(sample.get("zx|x", np.nan))
        y = float_or_nan(sample.get("zx|y", np.nan))
        lane_id, lane_dist, lane_direction = nearest_longstraight_lane(y)
        trigger_lane = str(trig.get("trigger_lane_id", ""))
        trigger_direction = LONGSTRAIGHT_LANE_DIRECTIONS.get(trigger_lane, "")
        same_lane = bool(lane_id and trigger_lane and lane_id == trigger_lane)
        same_side = bool(lane_direction and trigger_direction and lane_direction == trigger_direction)
        rows.append(
            {
                **trig.to_dict(),
                "ego_sample_time_rel_s": float(sample["time_rel_s"]),
                "ego_time_gap_s": round(time_gap, 6),
                "ego_x_m": x,
                "ego_y_signed_lateral_m": y,
                "ego_zx1_lateraldistance": sample.get("zx1|lateraldistance", np.nan),
                "ego_est_lane_id": lane_id,
                "ego_lane_center_distance_m": lane_dist,
                "ego_lane_direction_group": lane_direction,
                "trigger_lane_direction_group": trigger_direction,
                "ego_same_lane_as_trigger": same_lane,
                "ego_same_direction_side_as_trigger": same_side,
                "ego_lane_status": "ok" if abs(time_gap) <= 0.25 else "large_time_gap",
            }
        )
    return pd.DataFrame(rows)


def estimate_ego_lane_at_all_scene_triggers(
    scene_times: pd.DataFrame,
    road_map: pd.DataFrame,
    lane_geometry: pd.DataFrame,
) -> pd.DataFrame:
    if scene_times.empty:
        return pd.DataFrame()
    layout = load_projection_layout(road_map)
    layout_index = build_layout_index(layout)
    rows: list[dict[str, Any]] = []
    cache: dict[str, pd.DataFrame] = {}
    for _, trig in scene_times.iterrows():
        rel_path = str(trig.get("vehicle_raw_relative_path", ""))
        track = load_vehicle_track(rel_path, cache)
        t = float_or_nan(trig.get("estimated_trigger_time_rel_s", np.nan))
        if track.empty or not math.isfinite(t):
            rows.append({**trig.to_dict(), "ego_lane_status": "no_vehicle_track"})
            continue
        idx = int((track["time_rel_s"] - t).abs().idxmin())
        sample = track.loc[idx]
        time_gap = float(sample["time_rel_s"] - t)
        x = float_or_nan(sample.get("zx|x", np.nan))
        y = float_or_nan(sample.get("zx|y", np.nan))
        center, nearest_dist = nearest_layout_row(layout, layout_index, x, y)
        if center is None:
            signed_lat = float("nan")
            layout_module = ""
            layout_instance = ""
            layout_s = float("nan")
        else:
            vx = x - float(center["x"])
            vy = y - float(center["y"])
            tx = float(center["tangent_x"])
            ty = float(center["tangent_y"])
            signed_lat = tx * vy - ty * vx
            layout_module = str(center.get("module_name", ""))
            layout_instance = str(center.get("instance_name", ""))
            layout_s = float_or_nan(center.get("s", np.nan))
        module = str(trig.get("module_name", ""))
        ego_lane, ego_lane_dist, ego_dir = nearest_lane_from_geometry(lane_geometry, module, signed_lat)
        trigger_center, trigger_dir = trigger_lane_direction(lane_geometry, module, trig.get("trigger_lane_id", ""))
        same_lane = bool(ego_lane and str(trig.get("trigger_lane_id", "")) and ego_lane == str(trig.get("trigger_lane_id", "")))
        same_side = bool(ego_dir and trigger_dir and ego_dir == trigger_dir)
        relevance = "ego_direction_related" if same_side else "background_or_opposite"
        if not ego_lane:
            relevance = "unknown_lane_geometry"
        rows.append(
            {
                **trig.to_dict(),
                "ego_sample_time_rel_s": float(sample["time_rel_s"]),
                "ego_time_gap_s": round(time_gap, 6),
                "ego_x_m": x,
                "ego_y_m": y,
                "road_layout_s_m": layout_s,
                "road_layout_module_name": layout_module,
                "road_layout_instance_name": layout_instance,
                "road_layout_nearest_dist_m": nearest_dist,
                "ego_signed_lateral_offset_m": signed_lat,
                "ego_est_lane_id": ego_lane,
                "ego_lane_center_distance_m": ego_lane_dist,
                "ego_lane_direction_group": ego_dir,
                "trigger_lane_center_offset_m": trigger_center,
                "trigger_lane_direction_group": trigger_dir,
                "ego_same_lane_as_trigger": same_lane,
                "ego_same_direction_side_as_trigger": same_side,
                "scene_trigger_relevance_to_ego": relevance,
                "ego_lane_status": "ok" if abs(time_gap) <= 0.25 else "large_time_gap",
            }
        )
    return pd.DataFrame(rows)


def delta_bucket(delta: float) -> str:
    if not math.isfinite(delta):
        return "no_scene_trigger"
    if abs(delta) <= 0.5:
        return "old_close_to_scene_0p5s"
    if abs(delta) <= 1.0:
        return "old_close_to_scene_1s"
    if abs(delta) <= 2.0:
        return "old_close_to_scene_2s"
    if delta < -2.0:
        return "old_before_scene_gt2s"
    return "old_after_scene_gt2s"


def make_summary(
    objects: pd.DataFrame,
    triggers: pd.DataFrame,
    scene_times: pd.DataFrame,
    compare: pd.DataFrame,
    ego_lanes: pd.DataFrame,
    all_ego_lanes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(name: str, value: Any, note: str = "") -> None:
        rows.append({"item": name, "value": value, "note": note})

    add("traffic_object_rows", int(len(objects)), "展开到道路实例后的交通参与对象行数")
    add("traffic_trigger_rows", int(len(triggers)), "展开到道路实例后的触发点行数")
    add("session_trigger_time_rows", int(len(scene_times)), "触发点换算到被试记录时间轴后的行数")
    add("old_anchor_rows_compared", int(len(compare)), "参与对齐的旧 v400 锚点数")
    add("longstraight_ego_lane_rows", int(len(ego_lanes)), "longstraight 触发点处的被试车道估计行数")
    add("all_scene_ego_lane_rows", int(len(all_ego_lanes)), "全部场景触发点处的被试车道估计行数")
    if not triggers.empty and "raw_name" in triggers:
        add("trigger_name_counts", json.dumps(Counter(triggers["raw_name"].astype(str)), ensure_ascii=False))
    if not compare.empty:
        add("old_scene_delta_bucket_counts", json.dumps(Counter(compare["nearest_scene_trigger_bucket"].astype(str)), ensure_ascii=False))
        add("old_within_0p5s_scene_trigger", int(compare["within_0p5s_scene_trigger"].sum()))
        add("old_within_1s_scene_trigger", int(compare["within_1s_scene_trigger"].sum()))
        add("old_within_2s_scene_trigger", int(compare["within_2s_scene_trigger"].sum()))
    if not ego_lanes.empty:
        add("longstraight_ego_lane_counts", json.dumps(Counter(ego_lanes["ego_est_lane_id"].astype(str)), ensure_ascii=False))
        add("longstraight_ego_same_lane_as_trigger", int(ego_lanes["ego_same_lane_as_trigger"].sum()))
        add("longstraight_ego_same_direction_side_as_trigger", int(ego_lanes["ego_same_direction_side_as_trigger"].sum()))
    if not all_ego_lanes.empty:
        add("all_scene_relevance_counts", json.dumps(Counter(all_ego_lanes["scene_trigger_relevance_to_ego"].astype(str)), ensure_ascii=False))
        add("all_scene_same_lane_as_trigger", int(all_ego_lanes["ego_same_lane_as_trigger"].sum()))
        add("all_scene_same_direction_side_as_trigger", int(all_ego_lanes["ego_same_direction_side_as_trigger"].sum()))
    return pd.DataFrame(rows)


def build_scene_design_by_module_summary(
    objects: pd.DataFrame,
    triggers: pd.DataFrame,
    all_ego_lanes: pd.DataFrame,
) -> pd.DataFrame:
    modules = sorted(
        set(objects.get("module_name", pd.Series(dtype=str)).dropna().astype(str))
        | set(triggers.get("module_name", pd.Series(dtype=str)).dropna().astype(str))
    )
    rows: list[dict[str, Any]] = []
    for module in modules:
        obj = objects[objects["module_name"].astype(str).eq(module)] if not objects.empty else pd.DataFrame()
        tri = triggers[triggers["module_name"].astype(str).eq(module)] if not triggers.empty else pd.DataFrame()
        lane = all_ego_lanes[all_ego_lanes["module_name"].astype(str).eq(module)] if not all_ego_lanes.empty else pd.DataFrame()
        rows.append(
            {
                "module_name": module,
                "traffic_object_count": int(len(obj)),
                "traffic_object_names": "；".join(sorted(set(obj.get("raw_name", pd.Series(dtype=str)).dropna().astype(str)))) if not obj.empty else "",
                "traffic_trigger_count": int(len(tri)),
                "traffic_trigger_names": "；".join(sorted(set(tri.get("raw_name", pd.Series(dtype=str)).dropna().astype(str)))) if not tri.empty else "",
                "ego_direction_related_rows": int((lane.get("scene_trigger_relevance_to_ego", pd.Series(dtype=str)).astype(str) == "ego_direction_related").sum()) if not lane.empty else 0,
                "background_or_opposite_rows": int((lane.get("scene_trigger_relevance_to_ego", pd.Series(dtype=str)).astype(str) == "background_or_opposite").sum()) if not lane.empty else 0,
                "unknown_lane_rows": int((lane.get("scene_trigger_relevance_to_ego", pd.Series(dtype=str)).astype(str) == "unknown_lane_geometry").sum()) if not lane.empty else 0,
                "ego_lane_counts": json.dumps(Counter(lane.get("ego_est_lane_id", pd.Series(dtype=str)).astype(str)), ensure_ascii=False) if not lane.empty else "{}",
                "trigger_lane_counts": json.dumps(Counter(tri.get("lane_id", pd.Series(dtype=str)).astype(str)), ensure_ascii=False) if not tri.empty else "{}",
                "interpretation_cn": interpret_module_design(module, obj, tri, lane),
            }
        )
    return pd.DataFrame(rows)


def interpret_module_design(module: str, objects: pd.DataFrame, triggers: pd.DataFrame, ego_lanes: pd.DataFrame) -> str:
    if triggers.empty and objects.empty:
        return "未解析到显式交通对象或触发点。"
    if triggers.empty:
        return "解析到交通对象，但没有显式交通触发点；更可能是背景交通或静态场景布置，需要结合实验设计文本确认。"
    if ego_lanes.empty:
        return "解析到显式触发点，但暂未映射到被试记录时间轴。"
    related = int((ego_lanes["scene_trigger_relevance_to_ego"].astype(str) == "ego_direction_related").sum())
    background = int((ego_lanes["scene_trigger_relevance_to_ego"].astype(str) == "background_or_opposite").sum())
    if related > 0 and background > 0:
        return "同时存在被试方向相关触发和背景/对向侧触发；后续样本锚点只应采用被试方向相关部分。"
    if related > 0:
        return "存在被试方向相关触发，可作为后续场景锚点候选，但仍需车辆姿态确认是否真正形成响应。"
    return "当前显式触发主要不在被试同方向侧，应优先标记为背景/对向交通，不能直接作为被试方向样本锚点。"


def plot_longstraight(triggers: pd.DataFrame, objects: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=180)
    ax.set_title("longstraight 场景交通对象与触发点")
    ax.set_xlabel("道路纵向位置 s / m")
    ax.set_ylabel("车道 ID")

    lo = objects[objects["module_name"].eq("longstraight")].copy() if not objects.empty else pd.DataFrame()
    lt = triggers[triggers["module_name"].eq("longstraight")].copy() if not triggers.empty else pd.DataFrame()
    lane_values = []
    for df in [lo, lt]:
        if not df.empty and "lane_id" in df:
            lane_values.extend(pd.to_numeric(df["lane_id"], errors="coerce").dropna().tolist())
    lanes = sorted(set(lane_values))
    for lane in lanes:
        ax.axhline(float(lane), color="#e5e7eb", linewidth=0.8, zorder=0)

    if not lo.empty:
        ax.scatter(
            pd.to_numeric(lo["road_s_global_m"], errors="coerce"),
            pd.to_numeric(lo["lane_id"], errors="coerce"),
            marker="s",
            s=70,
            color="#2563eb",
            label="交通对象/车流源",
            alpha=0.85,
        )
        for _, row in lo.iterrows():
            label = row.get("raw_name", "")
            if str(label) == "Source":
                num = row.get("number_of_vehicles", "")
                label = f"Source {num}车"
            ax.text(
                float_or_nan(row.get("road_s_global_m")),
                float_or_nan(row.get("lane_id")) + 0.08,
                str(label),
                fontsize=8,
                ha="center",
                va="bottom",
            )
    if not lt.empty:
        colors = {"Activate": "#16a34a", "Stop": "#dc2626", "ChangeLane": "#f59e0b"}
        for name, group in lt.groupby("raw_name", dropna=False):
            ax.scatter(
                pd.to_numeric(group["road_s_global_m"], errors="coerce"),
                pd.to_numeric(group["lane_id"], errors="coerce"),
                marker="o",
                s=50,
                color=colors.get(str(name), "#7c3aed"),
                label=f"触发点 {name}",
                alpha=0.9,
            )
            for _, row in group.iterrows():
                ax.text(
                    float_or_nan(row.get("road_s_global_m")),
                    float_or_nan(row.get("lane_id")) - 0.10,
                    str(row.get("raw_name", "")),
                    fontsize=8,
                    ha="center",
                    va="top",
                )
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    if lanes:
        ax.set_ylim(min(lanes) - 0.45, max(lanes) + 0.75)
        ax.set_yticks(lanes)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_delta_hist(compare: pd.DataFrame, path: Path) -> None:
    if compare.empty:
        return
    delta = pd.to_numeric(compare["nearest_scene_trigger_delta_s"], errors="coerce").dropna()
    if delta.empty:
        return
    delta_clip = delta.clip(-20, 20)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
    ax.hist(delta_clip, bins=80, color="#4f46e5", alpha=0.82)
    ax.axvline(0, color="#111827", linewidth=1.2)
    ax.axvspan(-1, 1, color="#16a34a", alpha=0.15, label="±1s")
    ax.set_title("旧锚点相对最近场景触发点的时间差")
    ax.set_xlabel("旧锚点时间 - 场景触发点时间 / s，已裁剪到 ±20s")
    ax.set_ylabel("旧锚点数量")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def lane_center_for_plot(lane_id: Any) -> float:
    return LONGSTRAIGHT_LANE_CENTERS_M.get(str(lane_id), float("nan"))


def plot_longstraight_ego_lane_projection(
    ego_lanes: pd.DataFrame,
    triggers: pd.DataFrame,
    objects: pd.DataFrame,
    path: Path,
) -> None:
    if ego_lanes.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5.4), dpi=180)
    ax.set_title("longstraight 触发点处被试车横向位置与交通车道")
    ax.set_xlabel("道路纵向位置 s / m")
    ax.set_ylabel("相对中心线横向位置 / m")

    for lane_id, center in LONGSTRAIGHT_LANE_CENTERS_M.items():
        ax.axhline(center, color="#e5e7eb", linewidth=1.0, zorder=0)
        ax.text(1180, center + 0.12, f"车道{lane_id}", fontsize=8, color="#6b7280")

    colors = {"Activate": "#16a34a", "Stop": "#dc2626", "ChangeLane": "#f59e0b"}
    for name, group in ego_lanes.groupby("trigger_name", dropna=False):
        ax.scatter(
            pd.to_numeric(group["trigger_road_s_global_m"], errors="coerce"),
            pd.to_numeric(group["ego_y_signed_lateral_m"], errors="coerce"),
            s=12,
            alpha=0.38,
            color=colors.get(str(name), "#4f46e5"),
            label=f"被试车@{name}",
        )

    long_objects = objects[objects["module_name"].eq("longstraight")].copy() if not objects.empty else pd.DataFrame()
    if not long_objects.empty:
        y = [lane_center_for_plot(v) for v in long_objects["lane_id"]]
        ax.scatter(
            pd.to_numeric(long_objects["road_s_global_m"], errors="coerce"),
            y,
            marker="s",
            s=80,
            color="#2563eb",
            edgecolor="#1d4ed8",
            label="交通对象/车流源",
            zorder=4,
        )
    long_triggers = triggers[triggers["module_name"].eq("longstraight")].copy() if not triggers.empty else pd.DataFrame()
    if not long_triggers.empty:
        y = [lane_center_for_plot(v) for v in long_triggers["lane_id"]]
        ax.scatter(
            pd.to_numeric(long_triggers["road_s_global_m"], errors="coerce"),
            y,
            marker="x",
            s=90,
            color="#111827",
            label="交通触发点车道",
            zorder=5,
        )
    ax.axhline(0, color="#9ca3af", linewidth=1.2)
    ax.set_ylim(-9.5, 9.5)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_reports(
    objects: pd.DataFrame,
    triggers: pd.DataFrame,
    scene_times: pd.DataFrame,
    compare: pd.DataFrame,
    summary: pd.DataFrame,
    ego_lanes: pd.DataFrame,
    all_ego_lanes: pd.DataFrame,
    module_summary: pd.DataFrame,
) -> None:
    long_objects = objects[objects["module_name"].eq("longstraight")].copy() if not objects.empty else pd.DataFrame()
    long_triggers = triggers[triggers["module_name"].eq("longstraight")].copy() if not triggers.empty else pd.DataFrame()

    def md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 30) -> str:
        if df.empty:
            return "无。"
        part = df.loc[:, [c for c in cols if c in df.columns]].head(max_rows).copy()
        headers = list(part.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in part.iterrows():
            values = []
            for col in headers:
                value = row.get(col, "")
                if isinstance(value, float):
                    value = f"{value:.4f}" if math.isfinite(value) else ""
                values.append(str(value).replace("\n", " ").replace("|", "/"))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    bucket_counts = compare["nearest_scene_trigger_bucket"].value_counts(dropna=False).to_dict() if not compare.empty else {}
    trigger_counts = triggers["raw_name"].value_counts(dropna=False).to_dict() if not triggers.empty else {}
    within_1s = int(compare["within_1s_scene_trigger"].sum()) if not compare.empty else 0
    within_2s = int(compare["within_2s_scene_trigger"].sum()) if not compare.empty else 0
    ego_lane_counts = ego_lanes["ego_est_lane_id"].value_counts(dropna=False).to_dict() if not ego_lanes.empty else {}
    ego_same_lane = int(ego_lanes["ego_same_lane_as_trigger"].sum()) if not ego_lanes.empty else 0
    ego_same_side = int(ego_lanes["ego_same_direction_side_as_trigger"].sum()) if not ego_lanes.empty else 0
    ego_relevant_rows = (
        ego_lanes[ego_lanes["ego_same_direction_side_as_trigger"].astype(bool)].copy()
        if not ego_lanes.empty
        else pd.DataFrame()
    )
    background_rows = (
        ego_lanes[~ego_lanes["ego_same_direction_side_as_trigger"].astype(bool)].copy()
        if not ego_lanes.empty
        else pd.DataFrame()
    )
    all_relevance_counts = (
        all_ego_lanes["scene_trigger_relevance_to_ego"].value_counts(dropna=False).to_dict()
        if not all_ego_lanes.empty
        else {}
    )

    report = f"""# 场景触发点与旧锚点审计 v0.2

生成时间：{now_str()}

## 这一步为什么做

上一版道路事件审计只知道车辆经过了哪个道路模块，以及旧锚点是否接近道路边界、道路曲率或车身动态。但用户进一步关心的是：例如 `longstraight` 场景里，被试所在道路附近到底布置了哪些交通车，具体设置了哪些激活、停车、换道触发点。

因此本次审计直接读取 SILAB `.aed` 场景布局文件，提取交通参与对象和触发点，再把这些触发点换算到道路纵向位置和每条被试记录的相对时间轴上，最后与旧 v400 锚点对齐。

## 当前关键结论

- 解析到交通对象行数：{len(objects)}
- 解析到场景触发点行数：{len(triggers)}
- 触发点换算到被试记录时间轴后的行数：{len(scene_times)}
- 全部场景触发点处被试车道估计行数：{len(all_ego_lanes)}
- 与旧 v400 锚点完成最近邻对齐的行数：{len(compare)}
- 旧锚点 1 秒内接近场景触发点的数量：{within_1s}
- 旧锚点 2 秒内接近场景触发点的数量：{within_2s}

触发点类型计数：

```text
{json.dumps(trigger_counts, ensure_ascii=False, indent=2)}
```

全部场景按“是否在被试同方向侧”的初步分类：

```text
{json.dumps(all_relevance_counts, ensure_ascii=False, indent=2)}
```

各道路模块解释摘要：

{md_table(module_summary, ["module_name", "traffic_object_count", "traffic_trigger_count", "traffic_trigger_names", "ego_direction_related_rows", "background_or_opposite_rows", "unknown_lane_rows", "interpretation_cn"], max_rows=40)}

旧锚点相对最近场景触发点的时间差分组：

```text
{json.dumps(bucket_counts, ensure_ascii=False, indent=2)}
```

## longstraight 被试车道初步投影

本次还对 `longstraight` 做了第一版被试车道估计。方法是：在每条被试记录中找到车辆经过场景触发点的时刻，读取此时车辆的横向位置，并按 `longstraight_Area2.cfg` 中 21-27 号车道中心偏移做最近车道匹配。

被试车道计数：

```text
{json.dumps(ego_lane_counts, ensure_ascii=False, indent=2)}
```

- 被试车与触发点在同一车道的行数：{ego_same_lane}
- 被试车与触发点在同一方向侧的行数：{ego_same_side}

根据用户补充说明，25/26 车道那一侧的车辆是用于模拟高速公路连续交通流的背景车辆，不应作为被试行驶方向上的主要事件触发原因。结合当前投影结果，被试车主要处在 21/22/23 侧，而解析到的 `longstraight` 交通触发点主要在 25/26 侧。因此本次解析到的 25/26 侧 Activate / Stop / ChangeLane 应标记为“背景交通触发”，不能直接作为被试方向上的事件锚点。

- 当前可视为被试同方向相关的 `longstraight` 触发点行数：{len(ego_relevant_rows)}
- 当前应标记为背景交通的 `longstraight` 触发点时间映射行数：{len(background_rows)}

## longstraight 场景可以确认的信息

`longstraight` 的 `.aed` 文件不是只有道路几何，它的第 5 层里确实有交通对象和交通触发点。当前解析到的交通对象如下：

{md_table(long_objects, ["figure_name", "figure_id", "raw_name", "title", "lane_id", "tau", "relative_s_in_module_m", "road_s_global_m", "number_of_vehicles", "vehicle_type", "start_speed", "target_speed"])}

当前解析到的触发点如下：

{md_table(long_triggers, ["figure_name", "figure_id", "raw_name", "lane_id", "tau", "relative_s_in_module_m", "road_s_global_m", "target_name", "target_title", "target_lane_id", "change_target_lane", "change_time_or_distance", "description"])}

用白话说，`longstraight` 至少包含这些背景交通设置：

- 26 车道上的两个车流源：一个生成 4 辆小汽车，一个生成 5 辆小汽车；
- 25 车道上的一辆 `MAN TGL truck`；
- 25 车道上的一辆 `Chrysler300` 小轿车；
- 25 车道附近的车辆激活点；
- `Chrysler300` 的立即停车触发点；
- `MAN TGL truck` 从 25 车道向 26 车道换道的触发点。

但根据用户补充说明，这些 25/26 车道车辆主要是为了模拟高速公路上的连续交通流。后续建模和锚点重建时，不能把这些背景交通触发点直接当成被试方向的真实事件。

## 仍然不能直接下的结论

1. 还不能仅凭 `.aed` 说被试车一定在 25 车道或 26 车道。要确认被试实际车道，需要把每条车辆轨迹坐标投影到车道线，而不是只看交通车所在车道。
2. `.aed` 触发点是场景设定触发点；旧 v400 锚点多来自方向盘速率或后处理上下文。两者不是同一个定义。
3. 如果旧锚点明显晚于场景触发点，旧模型可能是在事件已经发生、甚至驾驶员已经开始响应之后才对齐样本。
4. 如果旧锚点明显早于场景触发点，可能是旧锚点和真实场景事件错配，也可能旧锚点抓到了其它车辆/道路动态。

## 建议下一步

1. 对 `longstraight`，优先只看被试方向 21/22/23 侧的事件来源，不再把 25/26 侧连续交通流作为主锚点。
2. 继续查实验设计文本或其它场景文件，确认被试方向上是否另有触发点、道路扰动、任务指令或车辆姿态触发规则。
3. 对旧 v400 锚点按“是否处在被试方向、是否接近同方向场景触发、是否更像车身响应后验”重新分组。
4. 抽查旧模型坏样本是否集中在“背景交通误配为事件”或“无同方向场景触发”的样本上。
5. 如果被试方向场景触发点无法在 `.aed` 中找到，则后续锚点应更多依赖道路任务设计文本 + 被试车辆姿态，而不是对向侧背景交通。

## 主要产物

- 交通对象表：`{TABLE_DIR / "aed_traffic_objects_v0_2.csv"}`
- 场景触发点表：`{TABLE_DIR / "aed_traffic_triggers_v0_2.csv"}`
- 触发点到每条被试记录的时间映射：`{TABLE_DIR / "scene_trigger_session_times_v0_2.csv"}`
- longstraight 被试车道估计：`{TABLE_DIR / "longstraight_ego_lane_at_scene_triggers_v0_2.csv"}`
- longstraight 被试同方向触发候选：`{TABLE_DIR / "longstraight_ego_direction_relevant_triggers_v0_2.csv"}`
- longstraight 背景交通触发映射：`{TABLE_DIR / "longstraight_background_traffic_triggers_v0_2.csv"}`
- 全部场景被试车道估计：`{TABLE_DIR / "all_scene_ego_lane_at_scene_triggers_v0_2.csv"}`
- 全部场景被试同方向触发候选：`{TABLE_DIR / "all_scene_ego_direction_relevant_triggers_v0_2.csv"}`
- 全部场景背景/对向触发映射：`{TABLE_DIR / "all_scene_background_or_opposite_triggers_v0_2.csv"}`
- 各场景设计摘要：`{TABLE_DIR / "scene_design_by_module_summary_v0_2.csv"}`
- 旧锚点与最近场景触发点对齐表：`{TABLE_DIR / "old_anchor_vs_scene_trigger_v0_2.csv"}`
- 审计汇总表：`{TABLE_DIR / "scene_trigger_audit_summary_v0_2.csv"}`
- longstraight 场景触发点图：`{FIGURE_DIR / "longstraight_scene_trigger_map_v0_2.png"}`
- 旧锚点相对场景触发点时间差图：`{FIGURE_DIR / "old_anchor_scene_trigger_delta_hist_v0_2.png"}`
- longstraight 被试车道投影图：`{FIGURE_DIR / "longstraight_ego_lane_projection_v0_2.png"}`
"""
    (REPORT_DIR / "scene_trigger_audit_v0_2_cn.md").write_text(report, encoding="utf-8")

    user_summary = f"""# 阶段 2 补充：场景触发点审计用户版说明

生成时间：{now_str()}

## 这个阶段为什么做

我们之前怀疑旧模型卡住，不只是模型结构问题，也可能是事件锚点没有对准真实场景事件。你问到 `longstraight` 场景中被试开的车道附近有哪些车、设置了什么触发点，这正是需要补的一层信息。

## 目前发现了什么

我已经从 `longstraight.autosave.1.aed` 里解析到交通车辆和触发点。这个场景里，25/26 车道附近确实有连续交通流背景设置：26 车道有车流源，25 车道有货车和小轿车，并且有激活、停车、换道触发点。根据你的补充说明，这些主要用于模拟高速公路上的连续车辆背景，后续不应直接作为被试行驶方向上的主要事件锚点。

其中比较关键的是：

- `Chrysler300` 小轿车在 25 车道附近有立即停车触发；
- `MAN TGL truck` 货车在接近位置有换道触发，目标车道写为 26；
- 26 车道还设置了车流源。

根据你的补充说明，25/26 车道那边的车辆是连续出现的背景交通，主要用于模拟高速公路交通流，不是我们要重点建模的被试方向事件。因此我把刚才的解释修正为：`longstraight` 里解析到的 25/26 侧 Activate / Stop / ChangeLane 不应直接作为被试方向的主要事件锚点。

我还做了第一版被试车道投影。结果显示，被试车在场景触发点附近的估计车道分布为：

```text
{json.dumps(ego_lane_counts, ensure_ascii=False, indent=2)}
```

这一步只是直道上的几何近似，但它已经说明：后续要重点查 21/22/23 这一侧，也就是被试实际行驶方向。25/26 侧连续交通流可以作为背景上下文保留，但不能直接拿来定义模型样本锚点。

我也把同样的判断推广到了其它场景：不是所有 `.aed` 中的车辆/触发点都能直接作为模型样本锚点，必须先判断它是否处在被试同方向侧。全部场景初步分类如下：

```text
{json.dumps(all_relevance_counts, ensure_ascii=False, indent=2)}
```

各场景的摘要在这里：

`{TABLE_DIR / "scene_design_by_module_summary_v0_2.csv"}`

## 目前还不能确定什么

还不能只凭这个文件说被试车一定在哪条车道。下一步必须把每条被试车辆轨迹投影到车道线上，确认被试通过触发点时与这些交通车的相对位置。

## 对旧流程有什么影响

这一步说明旧流程用方向盘速率或车辆响应后验选锚点，确实可能和真实场景触发点不是一个时刻。后续更合理的做法是：

1. 先用场景触发点定义事件发生位置；
2. 再用车身姿态确认被试是否真的受到影响；
3. 最后才截取方向盘未来响应作为预测标签。

这样比单纯用方向盘变化找锚点更符合因果顺序。

## 推荐你优先看

1. `{REPORT_DIR / "scene_trigger_audit_v0_2_cn.md"}`
2. `{TABLE_DIR / "aed_traffic_triggers_v0_2.csv"}`
3. `{TABLE_DIR / "longstraight_ego_lane_at_scene_triggers_v0_2.csv"}`
4. `{TABLE_DIR / "old_anchor_vs_scene_trigger_v0_2.csv"}`
5. `{FIGURE_DIR / "longstraight_scene_trigger_map_v0_2.png"}`
6. `{FIGURE_DIR / "longstraight_ego_lane_projection_v0_2.png"}`
7. `{TABLE_DIR / "scene_design_by_module_summary_v0_2.csv"}`
"""
    (REPORT_DIR / "stage02_scene_trigger_user_summary_cn.md").write_text(user_summary, encoding="utf-8")


def main() -> None:
    configure_matplotlib_fonts()
    ensure_dirs()

    road_map = read_csv(ROAD_POSITION_MAP)
    session_segments = read_csv(SESSION_MODULE_TIMES)
    old_anchors = read_csv(OLD_ANCHORS)
    lane_geometry = parse_lane_geometry()

    objects, triggers = parse_all_aed(road_map)
    scene_times = estimate_trigger_times(triggers, session_segments)
    ego_lanes = estimate_ego_lane_at_longstraight_triggers(scene_times)
    all_ego_lanes = estimate_ego_lane_at_all_scene_triggers(scene_times, road_map, lane_geometry)
    compare = compare_old_anchors_to_scene(old_anchors, scene_times)
    module_summary = build_scene_design_by_module_summary(objects, triggers, all_ego_lanes)
    summary = make_summary(objects, triggers, scene_times, compare, ego_lanes, all_ego_lanes)

    write_csv(lane_geometry, TABLE_DIR / "lane_geometry_from_cfg_v0_2.csv")
    write_csv(objects, TABLE_DIR / "aed_traffic_objects_v0_2.csv")
    write_csv(triggers, TABLE_DIR / "aed_traffic_triggers_v0_2.csv")
    write_csv(scene_times, TABLE_DIR / "scene_trigger_session_times_v0_2.csv")
    write_csv(ego_lanes, TABLE_DIR / "longstraight_ego_lane_at_scene_triggers_v0_2.csv")
    write_csv(all_ego_lanes, TABLE_DIR / "all_scene_ego_lane_at_scene_triggers_v0_2.csv")
    if not all_ego_lanes.empty:
        write_csv(
            all_ego_lanes[all_ego_lanes["scene_trigger_relevance_to_ego"].astype(str).eq("ego_direction_related")].copy(),
            TABLE_DIR / "all_scene_ego_direction_relevant_triggers_v0_2.csv",
        )
        write_csv(
            all_ego_lanes[all_ego_lanes["scene_trigger_relevance_to_ego"].astype(str).eq("background_or_opposite")].copy(),
            TABLE_DIR / "all_scene_background_or_opposite_triggers_v0_2.csv",
        )
    else:
        write_csv(pd.DataFrame(), TABLE_DIR / "all_scene_ego_direction_relevant_triggers_v0_2.csv")
        write_csv(pd.DataFrame(), TABLE_DIR / "all_scene_background_or_opposite_triggers_v0_2.csv")
    if not ego_lanes.empty:
        write_csv(
            ego_lanes[ego_lanes["ego_same_direction_side_as_trigger"].astype(bool)].copy(),
            TABLE_DIR / "longstraight_ego_direction_relevant_triggers_v0_2.csv",
        )
        write_csv(
            ego_lanes[~ego_lanes["ego_same_direction_side_as_trigger"].astype(bool)].copy(),
            TABLE_DIR / "longstraight_background_traffic_triggers_v0_2.csv",
        )
    else:
        write_csv(pd.DataFrame(), TABLE_DIR / "longstraight_ego_direction_relevant_triggers_v0_2.csv")
        write_csv(pd.DataFrame(), TABLE_DIR / "longstraight_background_traffic_triggers_v0_2.csv")
    write_csv(compare, TABLE_DIR / "old_anchor_vs_scene_trigger_v0_2.csv")
    write_csv(summary, TABLE_DIR / "scene_trigger_audit_summary_v0_2.csv")
    write_csv(module_summary, TABLE_DIR / "scene_design_by_module_summary_v0_2.csv")

    plot_longstraight(triggers, objects, FIGURE_DIR / "longstraight_scene_trigger_map_v0_2.png")
    plot_longstraight_ego_lane_projection(
        ego_lanes,
        triggers,
        objects,
        FIGURE_DIR / "longstraight_ego_lane_projection_v0_2.png",
    )
    plot_delta_hist(compare, FIGURE_DIR / "old_anchor_scene_trigger_delta_hist_v0_2.png")

    write_reports(objects, triggers, scene_times, compare, summary, ego_lanes, all_ego_lanes, module_summary)
    print(
        json.dumps(
            {
                "objects": len(objects),
                "triggers": len(triggers),
                "scene_times": len(scene_times),
                "longstraight_ego_lanes": len(ego_lanes),
                "all_scene_ego_lanes": len(all_ego_lanes),
                "compare": len(compare),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
