# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
INSTABILITY_CANDIDATE_PATH = (
    ROOT
    / "02_samples"
    / "instability_event_review_v0_1"
    / "tables"
    / "instability_reviewed_events_v0_1.csv"
)
FALLBACK_CANDIDATE_PATH = ROOT / "02_samples" / "tables" / "candidate_events_master.csv"
OUT_DIR = ROOT / "02_samples" / "manual_event_keyboard_player_v0_1"
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"
LABEL_PATH = TABLE_DIR / "keyboard_instability_event_labels_v0_1.csv"

VEHICLE_COLS = [
    "StorageTime",
    "zx1|lanecurvatureXY",
    "zx|lanecurvatureXY",
    "zx|SteeringWheel",
    "zx1|v_km/h",
    "zx1|lateraldistance",
    "zx|vyaw",
    "zx|ay",
    "zx|roll",
    "zx|vroll",
    "zx|yaw",
    "zx|AcceleratorPedal",
    "zx|BrakePedal",
]

SIGNALS = [
    ("curvature", "road curvature", ["zx1|lanecurvatureXY", "zx|lanecurvatureXY"]),
    ("steer", "steering", ["zx|SteeringWheel"]),
    ("speed", "speed km/h", ["zx1|v_km/h"]),
    ("lateral", "lateral distance", ["zx1|lateraldistance"]),
    ("yaw_rate", "yaw rate", ["zx|vyaw"]),
    ("ay", "lateral accel", ["zx|ay"]),
    ("roll", "roll", ["zx|roll"]),
    ("roll_rate", "roll rate", ["zx|vroll"]),
]

LABEL_FIELDS = [
    "label_id",
    "created_at",
    "subject",
    "session_stamp",
    "decision",
    "review_segment_id",
    "event_start_rel_s",
    "event_end_rel_s",
    "anchor_rel_s",
    "event_type",
    "direction",
    "confidence_1_5",
    "note",
    "selected_candidate_event_uid",
    "selected_candidate_source",
    "selected_candidate_reason",
    "nearest_candidate_event_uid",
    "nearest_candidate_anchor_source",
    "nearest_candidate_anchor_rel_s",
    "nearest_candidate_gap_s",
]


def ensure_dirs() -> None:
    for path in [OUT_DIR, TABLE_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    if not LABEL_PATH.exists():
        with LABEL_PATH.open("w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=LABEL_FIELDS).writeheader()
        return
    with LABEL_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        old_fields = reader.fieldnames or []
    if old_fields != LABEL_FIELDS:
        with LABEL_PATH.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in LABEL_FIELDS})


def to_seconds(storage_time: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(storage_time, errors="coerce")
    out = np.full(len(storage_time), np.nan, dtype=np.float64)
    valid = parsed.notna().to_numpy()
    if valid.any():
        parsed_ns = parsed[valid].astype("datetime64[ns]")
        ns = parsed_ns.astype("int64").to_numpy(dtype=np.float64)
        out[valid] = ns / 1e9
    return out


@lru_cache(maxsize=1)
def load_candidates() -> pd.DataFrame:
    if INSTABILITY_CANDIDATE_PATH.exists():
        df = pd.read_csv(INSTABILITY_CANDIDATE_PATH)
        df["event_uid"] = df["instability_event_uid"].astype(str)
        df["anchor_source"] = "raw_vehicle_instability_onset"
        df["event_type"] = df["instability_role"].astype(str)
        df["event_level"] = df["codex_recommended_decision"].astype(str)
        df["source_detail"] = df["leakage_note"].astype(str)
    else:
        df = pd.read_csv(FALLBACK_CANDIDATE_PATH)
    for col in ["anchor_time_rel_s", "event_start_rel_s", "event_end_rel_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def session_key(subject: str, session_stamp: str) -> str:
    return f"{subject}||{session_stamp}"


def get_sessions() -> list[dict[str, Any]]:
    df = load_candidates()
    grouped = (
        df.groupby(["subject", "session_stamp", "anchor_source"], dropna=False)["event_uid"]
        .count()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in [
        "raw_vehicle_instability_onset",
        "raw_road_curvature_onset",
        "old_v400_context_trigger_idx",
        "raw_vehicle_dynamic_onset",
    ]:
        if col not in grouped.columns:
            grouped[col] = 0
    grouped["sort_instability"] = grouped["raw_vehicle_instability_onset"].astype(int)
    grouped["sort_road"] = grouped["raw_road_curvature_onset"].astype(int)
    grouped = grouped.sort_values(["sort_instability", "sort_road", "subject", "session_stamp"], ascending=[False, False, True, True])
    sessions: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        subject = str(row["subject"])
        stamp = str(row["session_stamp"])
        sessions.append(
            {
                "key": session_key(subject, stamp),
                "subject": subject,
                "session_stamp": stamp,
                "raw_vehicle_instability_onset": int(row.get("raw_vehicle_instability_onset", 0)),
                "raw_road_curvature_onset": int(row.get("raw_road_curvature_onset", 0)),
                "old_v400_context_trigger_idx": int(row.get("old_v400_context_trigger_idx", 0)),
                "raw_vehicle_dynamic_onset": int(row.get("raw_vehicle_dynamic_onset", 0)),
            }
        )
    return sessions


def choose_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(x):
        return x
    return None


def downsample_indices(n: int, max_points: int = 7000) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).astype(int)


def build_review_segments(events: pd.DataFrame, duration_s: float) -> list[dict[str, Any]]:
    def clipped(value: float, lo: float = 0.0, hi: float | None = None) -> float:
        if hi is None:
            hi = duration_s
        return max(lo, min(hi, value))

    def count_sources(start_s: float, end_s: float) -> dict[str, int]:
        nearby = events[
            (events["anchor_time_rel_s"] >= start_s)
            & (events["anchor_time_rel_s"] <= end_s)
        ]
        counts = nearby.groupby("anchor_source").size().to_dict()
        return {
            "raw_road_curvature_onset": int(counts.get("raw_road_curvature_onset", 0)),
            "old_v400_context_trigger_idx": int(counts.get("old_v400_context_trigger_idx", 0)),
            "raw_vehicle_dynamic_onset": int(counts.get("raw_vehicle_dynamic_onset", 0)),
        }

    segments: list[dict[str, Any]] = []
    instability = events[events["anchor_source"] == "raw_vehicle_instability_onset"].copy()
    if not instability.empty:
        seg_idx = 1
        instability = instability.sort_values(["anchor_time_rel_s", "event_start_rel_s"])
        for _, ev in instability.iterrows():
            anchor = finite_float(ev.get("anchor_time_rel_s"))
            if anchor is None:
                continue
            start = finite_float(ev.get("event_start_rel_s"))
            end = finite_float(ev.get("event_end_rel_s"))
            if start is None:
                start = anchor
            if end is None or end <= start:
                end = anchor + 1.0
            review_start = clipped(anchor - 8.0)
            review_end = clipped(anchor + 10.0)
            if review_end - review_start < 12.0:
                review_start = clipped(anchor - 6.0)
                review_end = clipped(anchor + 6.0)
            decision = str(ev.get("event_level", ""))
            score = finite_float(ev.get("instability_review_score"))
            role = str(ev.get("event_type", "vehicle_instability"))
            signed_ay = finite_float(ev.get("signed_peak_ay_window"))
            direction = "left_or_positive" if signed_ay is not None and signed_ay > 0 else "right_or_negative" if signed_ay is not None and signed_ay < 0 else "unclear"
            counts = count_sources(review_start, review_end)
            reason = (
                f"vehicle instability candidate; score={score if score is not None else 'NA'}; "
                f"decision={decision}; role={role}; use ay/roll_rate as onset, steering only as future response evidence"
            )
            segments.append(
                {
                    "review_segment_id": f"instability_{seg_idx:04d}",
                    "priority": 1,
                    "candidate_source": "raw_vehicle_instability_onset",
                    "candidate_event_uid": str(ev.get("event_uid", "")),
                    "anchor_time_rel_s": round(float(anchor), 3),
                    "event_start_rel_s": round(float(start), 3),
                    "event_end_rel_s": round(float(end), 3),
                    "review_start_rel_s": round(float(review_start), 3),
                    "review_end_rel_s": round(float(review_end), 3),
                    "event_type": role,
                    "direction": direction,
                    "event_level": decision,
                    "reason": reason,
                    "nearby_counts": counts,
                }
            )
            seg_idx += 1
        return segments

    road = events[events["anchor_source"] == "raw_road_curvature_onset"].copy()
    seg_idx = 1
    for _, ev in road.sort_values("anchor_time_rel_s").iterrows():
        anchor = finite_float(ev.get("anchor_time_rel_s"))
        if anchor is None:
            continue
        start = finite_float(ev.get("event_start_rel_s"))
        end = finite_float(ev.get("event_end_rel_s"))
        if start is None:
            start = anchor
        if end is None or end <= start:
            end = anchor + 2.0
        event_type = str(ev.get("event_type", "curve"))
        direction = "left" if "left" in event_type or "positive" in event_type else "right" if "right" in event_type or "negative" in event_type else "unclear"
        duration = max(0.0, end - start)
        if duration > 30.0:
            sub_specs = [
                ("entry", start, min(end, start + 12.0), start, clipped(start - 8.0), clipped(start + 22.0), "入弯/道路曲率开始"),
                ("exit", max(start, end - 12.0), end, max(start, end - 12.0), clipped(end - 22.0), clipped(end + 8.0), "出弯/尾段回正"),
            ]
        else:
            mid = (start + end) / 2.0
            sub_specs = [
                ("full", start, end, start, clipped(mid - 16.0), clipped(mid + 16.0), "完整短弯道候选"),
            ]
        for suffix, seg_start, seg_end, seg_anchor, review_start, review_end, phase_name in sub_specs:
            if review_end - review_start < 18.0:
                mid = (seg_start + seg_end) / 2.0
                review_start = clipped(mid - 9.0)
                review_end = clipped(mid + 9.0)
            counts = count_sources(review_start, review_end)
            reason = (
                f"{phase_name}；建议标 {seg_start:.1f}-{seg_end:.1f}s；"
                f"原道路曲率段 {start:.1f}-{end:.1f}s，附近 old={counts['old_v400_context_trigger_idx']}，"
                f"dynamic={counts['raw_vehicle_dynamic_onset']}"
            )
            segments.append(
                {
                    "review_segment_id": f"road_{seg_idx:03d}_{suffix}",
                    "priority": 1,
                    "candidate_source": "raw_road_curvature_onset",
                    "candidate_event_uid": str(ev.get("event_uid", "")),
                    "anchor_time_rel_s": round(float(seg_anchor), 3),
                    "event_start_rel_s": round(float(seg_start), 3),
                    "event_end_rel_s": round(float(seg_end), 3),
                    "review_start_rel_s": round(float(review_start), 3),
                    "review_end_rel_s": round(float(review_end), 3),
                    "event_type": event_type,
                    "direction": direction,
                    "event_level": str(ev.get("event_level", "")),
                    "reason": reason,
                    "nearby_counts": counts,
                }
            )
            seg_idx += 1

    if segments:
        return segments

    anchors = events[np.isfinite(events["anchor_time_rel_s"])].copy().sort_values("anchor_time_rel_s")
    cluster: list[pd.Series] = []
    last_anchor: float | None = None
    cluster_id = 1

    def flush_cluster(rows: list[pd.Series], cid: int) -> None:
        if not rows:
            return
        anchors_local = [float(r["anchor_time_rel_s"]) for r in rows if math.isfinite(float(r["anchor_time_rel_s"]))]
        if not anchors_local:
            return
        start = min(anchors_local)
        end = max(anchors_local)
        anchor = start
        review_start = clipped(start - 8.0)
        review_end = clipped(end + 8.0)
        if review_end - review_start < 18.0:
            mid = (start + end) / 2.0
            review_start = clipped(mid - 9.0)
            review_end = clipped(mid + 9.0)
        sources = [str(r.get("anchor_source", "")) for r in rows]
        counts = {k: sources.count(k) for k in set(sources)}
        source_counts = {
            "raw_road_curvature_onset": int(counts.get("raw_road_curvature_onset", 0)),
            "old_v400_context_trigger_idx": int(counts.get("old_v400_context_trigger_idx", 0)),
            "raw_vehicle_dynamic_onset": int(counts.get("raw_vehicle_dynamic_onset", 0)),
        }
        first = rows[0]
        reason = (
            f"候选聚类；{len(rows)} 个 old/dynamic 候选集中在 {start:.1f}-{end:.1f}s；"
            f"old={source_counts['old_v400_context_trigger_idx']}，dynamic={source_counts['raw_vehicle_dynamic_onset']}"
        )
        segments.append(
            {
                "review_segment_id": f"cluster_{cid:03d}",
                "priority": 3,
                "candidate_source": "old_dynamic_cluster",
                "candidate_event_uid": str(first.get("event_uid", "")),
                "anchor_time_rel_s": round(float(anchor), 3),
                "event_start_rel_s": round(float(start), 3),
                "event_end_rel_s": round(float(max(end, start + 1.0)), 3),
                "review_start_rel_s": round(float(review_start), 3),
                "review_end_rel_s": round(float(review_end), 3),
                "event_type": str(first.get("event_type", "unclear")),
                "direction": "unclear",
                "event_level": str(first.get("event_level", "")),
                "reason": reason,
                "nearby_counts": source_counts,
            }
        )

    for _, row in anchors.iterrows():
        anchor = finite_float(row.get("anchor_time_rel_s"))
        if anchor is None:
            continue
        if last_anchor is not None and anchor - last_anchor > 4.0:
            flush_cluster(cluster, cluster_id)
            cluster = []
            cluster_id += 1
        cluster.append(row)
        last_anchor = anchor
    flush_cluster(cluster, cluster_id)
    return segments[:40]


@lru_cache(maxsize=12)
def load_session_payload(subject: str, session_stamp: str) -> dict[str, Any]:
    candidates = load_candidates()
    events = candidates[
        (candidates["subject"].astype(str) == str(subject))
        & (candidates["session_stamp"].astype(str) == str(session_stamp))
    ].copy()
    if events.empty:
        raise ValueError(f"session not found: {subject} {session_stamp}")
    raw_path = Path(str(events.iloc[0]["vehicle_raw_absolute_path"]))
    df = pd.read_csv(raw_path, usecols=lambda c: c in VEHICLE_COLS)
    if "StorageTime" not in df.columns:
        raise ValueError("vehicle CSV missing StorageTime")
    abs_s = to_seconds(df["StorageTime"])
    valid = np.isfinite(abs_s)
    df = df.loc[valid].copy()
    abs_s = abs_s[valid]
    if len(df) == 0:
        raise ValueError("vehicle CSV has no valid StorageTime")
    df["time_rel_s"] = abs_s - float(np.nanmin(abs_s))
    df = df.sort_values("time_rel_s").reset_index(drop=True)
    idx = downsample_indices(len(df))
    time_rel = df["time_rel_s"].to_numpy(dtype=np.float64)[idx]
    signals: list[dict[str, Any]] = []
    for name, label, cols in SIGNALS:
        col = choose_col(df, cols)
        if col is None:
            values = np.full(len(time_rel), np.nan, dtype=np.float64)
            src = ""
        else:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)[idx]
            src = col
        clean_values = [None if not np.isfinite(x) else round(float(x), 7) for x in values]
        signals.append({"name": name, "label": label, "source_column": src, "values": clean_values})
    event_rows: list[dict[str, Any]] = []
    for _, ev in events.sort_values(["anchor_time_rel_s", "anchor_source"]).iterrows():
        event_rows.append(
            {
                "event_uid": str(ev.get("event_uid", "")),
                "anchor_source": str(ev.get("anchor_source", "")),
                "anchor_time_rel_s": finite_float(ev.get("anchor_time_rel_s")),
                "event_start_rel_s": finite_float(ev.get("event_start_rel_s")),
                "event_end_rel_s": finite_float(ev.get("event_end_rel_s")),
                "event_type": str(ev.get("event_type", "")),
                "event_level": str(ev.get("event_level", "")),
            }
        )
    return {
        "subject": subject,
        "session_stamp": session_stamp,
        "vehicle_raw_absolute_path": str(raw_path),
        "duration_s": round(float(np.nanmax(df["time_rel_s"].to_numpy(dtype=np.float64))), 6),
        "time_rel_s": [round(float(x), 6) for x in time_rel],
        "signals": signals,
        "events": event_rows,
        "review_segments": build_review_segments(events, float(np.nanmax(df["time_rel_s"].to_numpy(dtype=np.float64)))),
    }


def nearest_candidate(subject: str, session_stamp: str, anchor_s: float) -> dict[str, str]:
    events = load_candidates()
    events = events[
        (events["subject"].astype(str) == str(subject))
        & (events["session_stamp"].astype(str) == str(session_stamp))
        & np.isfinite(events["anchor_time_rel_s"])
    ].copy()
    if events.empty or not math.isfinite(anchor_s):
        return {
            "nearest_candidate_event_uid": "",
            "nearest_candidate_anchor_source": "",
            "nearest_candidate_anchor_rel_s": "",
            "nearest_candidate_gap_s": "",
        }
    events["gap"] = (events["anchor_time_rel_s"] - anchor_s).abs()
    row = events.sort_values("gap").iloc[0]
    return {
        "nearest_candidate_event_uid": str(row.get("event_uid", "")),
        "nearest_candidate_anchor_source": str(row.get("anchor_source", "")),
        "nearest_candidate_anchor_rel_s": f"{float(row['anchor_time_rel_s']):.3f}",
        "nearest_candidate_gap_s": f"{float(row['gap']):.3f}",
    }


def append_label(payload: dict[str, Any]) -> dict[str, Any]:
    subject = str(payload.get("subject", "")).strip()
    stamp = str(payload.get("session_stamp", "")).strip()
    start = finite_float(payload.get("event_start_rel_s"))
    end = finite_float(payload.get("event_end_rel_s"))
    anchor = finite_float(payload.get("anchor_rel_s"))
    if not subject or not stamp:
        raise ValueError("missing subject/session_stamp")
    if start is None or end is None:
        raise ValueError("missing start/end")
    if end < start:
        start, end = end, start
    if anchor is None:
        anchor = start
    label_id = f"kb_{subject}_{stamp}_{int(time.time() * 1000)}"
    row = {
        "label_id": label_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "subject": subject,
        "session_stamp": stamp,
        "decision": str(payload.get("decision", "accept")).strip() or "accept",
        "review_segment_id": str(payload.get("review_segment_id", "")).strip(),
        "event_start_rel_s": f"{start:.3f}",
        "event_end_rel_s": f"{end:.3f}",
        "anchor_rel_s": f"{anchor:.3f}",
        "event_type": str(payload.get("event_type", "")).strip(),
        "direction": str(payload.get("direction", "")).strip(),
        "confidence_1_5": str(payload.get("confidence_1_5", "")).strip(),
        "note": str(payload.get("note", "")).replace("\r", " ").replace("\n", " ").strip(),
        "selected_candidate_event_uid": str(payload.get("selected_candidate_event_uid", "")).strip(),
        "selected_candidate_source": str(payload.get("selected_candidate_source", "")).strip(),
        "selected_candidate_reason": str(payload.get("selected_candidate_reason", "")).replace("\r", " ").replace("\n", " ").strip(),
    }
    row.update(nearest_candidate(subject, stamp, anchor))
    with LABEL_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=LABEL_FIELDS).writerow(row)
    return row


def read_labels() -> list[dict[str, str]]:
    ensure_dirs()
    with LABEL_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def rewrite_labels(rows: list[dict[str, str]]) -> None:
    with LABEL_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LABEL_FIELDS})


def undo_last_label() -> dict[str, Any]:
    rows = read_labels()
    if not rows:
        return {"removed": None, "count": 0}
    removed = rows.pop()
    rewrite_labels(rows)
    return {"removed": removed, "count": len(rows)}


def write_report(host: str, port: int) -> None:
    report = f"""# 阶段 2 补充：车辆失稳候选键盘审查播放器 v0.1

生成时间：2026-05-12

## 为什么做

用户指出逐个从整段行驶过程中找事件太复杂，并进一步指出之前 404 个自动审阅样本都是弯道样本，不是车辆失稳样本。因此本地播放器已从“弯道候选审查”切换为“车辆失稳候选审查”。

## 使用入口

- 本地页面：`http://{host}:{port}/`
- 当前读取候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_reviewed_events_v0_1.csv`
- 标签输出：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_instability_event_labels_v0_1.csv`
- 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/run_manual_event_keyboard_player.py`

## 当前事件定义

- 主事件是车辆失稳候选。
- 失稳锚点来自 `ay` 和 `roll_rate` 等非方向盘车辆动态异常。
- `steer_rate` 不作为主锚点，因为它已经是驾驶员方向盘动作。
- 方向盘只作为事件后响应证据。

## 默认按键

- 空格：播放/暂停。
- `A`：把当前时间标记为事件开始。
- `S`：把当前时间标记为预测锚点；不按则默认锚点等于开始时间。
- `D`：把当前时间标记为事件结束并保存一行标签。
- `Y`：直接采用当前候选段的建议起止和锚点。
- `Q` / `E`：切换上一段/下一段候选事件。
- 左/右方向键：小步后退/前进；按住 Shift 为大步。
- `N` / `P`：切换下一条/上一条记录。
- `U`：撤销最后一条保存的标签。

## 页面竖线说明

- 红线：车辆失稳候选锚点，来自横向加速度或横滚速率。
- 浅蓝背景：当前正在审查的候选事件段。
- 黑线：当前播放时间。
- 橙线：用户手动调整的事件开始点。
- 紫线：用户手动标记的预测锚点。
- 绿色背景/绿线：已经保存的人工标签。

## 边界

本工具只读取原始车辆 CSV 和候选事件表，不修改原始 CSV，不训练模型，不读取服务器密码。输出的人工标签需要再经过一致性检查，才能升级为 `manual_verified` 样本清单。
"""
    (REPORT_DIR / "manual_event_keyboard_player_v0_1_cn.md").write_text(report, encoding="utf-8")


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>R2E Steering Keyboard Event Labeler</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #6b7280;
      --line: #d5d9e1;
      --accent: #1463d8;
      --ok: #147a3f;
      --warn: #b45309;
      --bad: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, "Microsoft YaHei", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 16px;
      background: #fff;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 {
      font-size: 18px;
      margin: 0;
      white-space: nowrap;
    }
    select, input, button, textarea {
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
    }
    select, input, button { height: 34px; padding: 0 9px; }
    button { cursor: pointer; }
    button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    button.ok { background: var(--ok); color: #fff; border-color: var(--ok); }
    button.warn { background: var(--warn); color: #fff; border-color: var(--warn); }
    main {
      display: grid;
      grid-template-columns: minmax(760px, 1fr) 320px;
      gap: 12px;
      padding: 12px;
    }
    .plot-wrap, .side {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .plot-top {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 14px;
    }
    #plot {
      display: block;
      width: 100%;
      height: calc(100vh - 160px);
      min-height: 620px;
      background: #fff;
    }
    .side {
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      max-height: calc(100vh - 90px);
      overflow: auto;
    }
    .group {
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
    }
    .group:last-child { border-bottom: 0; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .row > * { flex: 1 1 auto; }
    label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .field { margin-bottom: 8px; }
    .field input, .field select, .field textarea { width: 100%; }
    textarea { min-height: 58px; padding: 8px; resize: vertical; }
    .stat {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      font-size: 13px;
    }
    .stat div {
      background: #f3f4f6;
      border-radius: 6px;
      padding: 7px;
    }
    .shortcut {
      display: grid;
      grid-template-columns: 44px 1fr;
      gap: 4px 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .key {
      text-align: center;
      background: #eef2f7;
      border: 1px solid #d7dce4;
      border-radius: 5px;
      color: var(--ink);
      padding: 2px 4px;
      font-weight: 700;
    }
    #labels {
      font-size: 12px;
      border-collapse: collapse;
      width: 100%;
    }
    #labels th, #labels td {
      border-bottom: 1px solid var(--line);
      padding: 5px 4px;
      text-align: left;
      white-space: nowrap;
    }
    .pill {
      display: inline-block;
      border-radius: 999px;
      padding: 3px 8px;
      background: #eef2f7;
      color: var(--muted);
    }
    .status { font-weight: 700; color: var(--accent); }
  </style>
</head>
<body>
  <header>
    <h1>R2E 键盘事件标注</h1>
    <select id="sessionSelect" title="选择记录"></select>
    <button id="prevBtn" title="上一条记录">P</button>
    <button id="nextBtn" title="下一条记录">N</button>
    <button id="playBtn" class="primary" title="空格播放或暂停">Play</button>
    <select id="speedSelect" title="播放速度">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="5">5x</option>
      <option value="10">10x</option>
      <option value="20">20x</option>
    </select>
    <select id="windowSelect" title="时间窗口">
      <option value="15">15s</option>
      <option value="30" selected>30s</option>
      <option value="60">60s</option>
      <option value="120">120s</option>
    </select>
    <span id="status" class="status">loading</span>
  </header>
  <main>
    <section class="plot-wrap">
      <div class="plot-top">
        <span id="sessionTitle"></span>
        <span class="pill" id="timeReadout">0.000s</span>
        <span class="pill" id="rangeReadout"></span>
        <span class="pill">blue road</span>
        <span class="pill">orange old</span>
        <span class="pill">red dynamic</span>
        <span class="pill">green manual</span>
      </div>
      <canvas id="plot"></canvas>
    </section>
    <aside class="side">
      <div class="group">
        <div class="stat">
          <div>Start<br><strong id="startReadout">-</strong></div>
          <div>End<br><strong id="endReadout">-</strong></div>
          <div>Anchor<br><strong id="anchorReadout">-</strong></div>
          <div>Saved<br><strong id="savedReadout">0</strong></div>
        </div>
      </div>
      <div class="group">
        <div class="row">
          <button id="markStartBtn" class="warn" title="A">Start</button>
          <button id="markAnchorBtn" title="S">Anchor</button>
          <button id="markEndBtn" class="ok" title="D">End + Save</button>
        </div>
        <div class="row" style="margin-top:8px;">
          <button id="backBtn" title="左方向键">-0.2s</button>
          <button id="forwardBtn" title="右方向键">+0.2s</button>
          <button id="undoBtn" title="U">Undo</button>
        </div>
      </div>
      <div class="group">
        <div class="field">
          <label>事件类型</label>
          <select id="eventType">
            <option value="vehicle_instability">vehicle_instability</option>
            <option value="instability_ay_only">instability_ay_only</option>
            <option value="instability_roll_only">instability_roll_only</option>
            <option value="instability_ay_roll">instability_ay_roll</option>
            <option value="curve">curve</option>
            <option value="lane_change">lane_change</option>
            <option value="avoidance">avoidance</option>
            <option value="return">return</option>
            <option value="multi_stage">multi_stage</option>
            <option value="unclear">unclear</option>
          </select>
        </div>
        <div class="field">
          <label>方向</label>
          <select id="direction">
            <option value="left">left</option>
            <option value="right">right</option>
            <option value="straight">straight</option>
            <option value="unclear" selected>unclear</option>
          </select>
        </div>
        <div class="field">
          <label>置信度</label>
          <select id="confidence">
            <option value="5">5</option>
            <option value="4" selected>4</option>
            <option value="3">3</option>
            <option value="2">2</option>
            <option value="1">1</option>
          </select>
        </div>
        <div class="field">
          <label>备注</label>
          <textarea id="note"></textarea>
        </div>
      </div>
      <div class="group">
        <div class="shortcut">
          <span class="key">Space</span><span>播放/暂停</span>
          <span class="key">A</span><span>标记开始</span>
          <span class="key">S</span><span>标记锚点</span>
          <span class="key">D</span><span>结束并保存</span>
          <span class="key">←/→</span><span>前后移动，Shift 为大步</span>
          <span class="key">N/P</span><span>下一条/上一条记录</span>
          <span class="key">U</span><span>撤销最后一条</span>
        </div>
      </div>
      <div class="group">
        <a href="/api/labels.csv" target="_blank">打开当前标签 CSV</a>
      </div>
      <div class="group">
        <table id="labels">
          <thead><tr><th>session</th><th>start</th><th>end</th><th>type</th><th>dir</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </aside>
  </main>
<script>
const state = {
  sessions: [],
  sessionIndex: 0,
  data: null,
  labels: [],
  t: 0,
  playing: false,
  speed: 1,
  viewWindow: 30,
  start: null,
  end: null,
  anchor: null,
  lastFrame: performance.now()
};

const $ = (id) => document.getElementById(id);
const canvas = $("plot");
const ctx = canvas.getContext("2d");

function setStatus(text) { $("status").textContent = text; }
function fmt(x) { return x == null || Number.isNaN(x) ? "-" : `${Number(x).toFixed(3)}s`; }

async function api(path, options = {}) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${resp.status} ${text}`);
  }
  return resp.json();
}

async function loadSessions() {
  state.sessions = await api("/api/sessions");
  const select = $("sessionSelect");
  select.innerHTML = "";
  state.sessions.forEach((s, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${s.subject} ${s.session_stamp} | instability ${s.raw_vehicle_instability_onset || 0}`;
    select.appendChild(opt);
  });
  await loadSession(0);
}

async function loadLabels() {
  state.labels = await api("/api/labels");
  renderLabels();
}

async function loadSession(index) {
  if (index < 0 || index >= state.sessions.length) return;
  state.sessionIndex = index;
  $("sessionSelect").value = index;
  const s = state.sessions[index];
  setStatus("loading session");
  state.data = await api(`/api/session?subject=${encodeURIComponent(s.subject)}&session_stamp=${encodeURIComponent(s.session_stamp)}`);
  state.t = 0;
  state.start = null;
  state.end = null;
  state.anchor = null;
  state.playing = false;
  $("playBtn").textContent = "Play";
  $("sessionTitle").textContent = `${s.subject} / ${s.session_stamp}`;
  $("rangeReadout").textContent = `${Number(state.data.duration_s).toFixed(1)}s`;
  updateReadouts();
  setStatus("ready");
  draw();
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function updateReadouts() {
  $("timeReadout").textContent = fmt(state.t);
  $("startReadout").textContent = fmt(state.start);
  $("endReadout").textContent = fmt(state.end);
  $("anchorReadout").textContent = fmt(state.anchor);
  $("savedReadout").textContent = String(state.labels.length);
}

function panelLimits(values, loT, hiT, times) {
  let vals = [];
  for (let i = 0; i < times.length; i++) {
    const t = times[i], v = values[i];
    if (t >= loT && t <= hiT && v != null && Number.isFinite(v)) vals.push(v);
  }
  if (!vals.length) return [-1, 1];
  vals.sort((a, b) => a - b);
  const p = (q) => vals[Math.min(vals.length - 1, Math.max(0, Math.floor(q * (vals.length - 1))))];
  let lo = Math.min(p(0.01), 0), hi = Math.max(p(0.99), 0);
  if (Math.abs(hi - lo) < 1e-9) { lo -= 1; hi += 1; }
  const pad = 0.08 * (hi - lo);
  return [lo - pad, hi + pad];
}

function colorForSource(source) {
  if (source === "raw_vehicle_instability_onset") return "#cf3333";
  if (source === "raw_road_curvature_onset") return "#286ed6";
  if (source === "old_v400_context_trigger_idx") return "#d98520";
  if (source === "raw_vehicle_dynamic_onset") return "#cf3333";
  return "#777";
}

function drawVLine(x, top, bottom, color, width = 1) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x, top);
  ctx.lineTo(x, bottom);
  ctx.stroke();
}

function draw() {
  if (!state.data) return;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);

  const marginL = 120, marginR = 28, top = 22, bottom = 32;
  const plotW = w - marginL - marginR;
  const panelGap = 8;
  const n = state.data.signals.length;
  const panelH = (h - top - bottom - panelGap * (n - 1)) / n;
  const duration = Number(state.data.duration_s);
  const half = state.viewWindow / 2;
  let loT = Math.max(0, state.t - half);
  let hiT = Math.min(duration, loT + state.viewWindow);
  loT = Math.max(0, hiT - state.viewWindow);
  const xOf = (t) => marginL + (t - loT) / Math.max(hiT - loT, 1e-9) * plotW;
  const times = state.data.time_rel_s;

  for (let p = 0; p < n; p++) {
    const sig = state.data.signals[p];
    const y0 = top + p * (panelH + panelGap);
    const y1 = y0 + panelH;
    ctx.strokeStyle = "#d5d9e1";
    ctx.lineWidth = 1;
    ctx.strokeRect(marginL, y0, plotW, panelH);
    ctx.fillStyle = "#111827";
    ctx.font = "13px Arial";
    ctx.fillText(sig.label, 12, y0 + 18);
    const [vLo, vHi] = panelLimits(sig.values, loT, hiT, times);
    const yOf = (v) => y1 - (v - vLo) / Math.max(vHi - vLo, 1e-9) * panelH;
    if (vLo <= 0 && vHi >= 0) {
      const zy = yOf(0);
      ctx.strokeStyle = "#e5e7eb";
      ctx.beginPath(); ctx.moveTo(marginL, zy); ctx.lineTo(marginL + plotW, zy); ctx.stroke();
    }
    ctx.strokeStyle = "#111827";
    if (sig.name === "curvature") ctx.strokeStyle = "#286ed6";
    if (sig.name === "speed") ctx.strokeStyle = "#147a3f";
    if (sig.name === "lateral") ctx.strokeStyle = "#7c3aed";
    if (sig.name === "yaw_rate") ctx.strokeStyle = "#b91c1c";
    if (sig.name === "ay") ctx.strokeStyle = "#d97706";
    if (sig.name === "roll") ctx.strokeStyle = "#8b5e34";
    ctx.lineWidth = 1.7;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < times.length; i++) {
      const t = times[i], v = sig.values[i];
      if (t < loT || t > hiT || v == null || !Number.isFinite(v)) continue;
      const x = xOf(t), y = yOf(v);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    if (started) ctx.stroke();
    ctx.fillStyle = "#6b7280";
    ctx.font = "11px Arial";
    ctx.fillText(`${vLo.toPrecision(3)} to ${vHi.toPrecision(3)}`, w - 116, y0 + 16);
  }

  for (const ev of state.data.events) {
    const t = Number(ev.anchor_time_rel_s);
    if (!Number.isFinite(t) || t < loT || t > hiT) continue;
    drawVLine(xOf(t), top - 6, h - bottom + 4, colorForSource(ev.anchor_source), ev.anchor_source === "raw_road_curvature_onset" ? 2 : 1);
  }
  const sessionLabels = state.labels.filter(l => l.subject === state.data.subject && l.session_stamp === state.data.session_stamp);
  for (const lab of sessionLabels) {
    const s = Number(lab.event_start_rel_s), e = Number(lab.event_end_rel_s);
    if (!Number.isFinite(s) || !Number.isFinite(e) || e < loT || s > hiT) continue;
    const xs = xOf(Math.max(s, loT)), xe = xOf(Math.min(e, hiT));
    ctx.fillStyle = "rgba(20, 122, 63, 0.16)";
    ctx.fillRect(xs, top - 4, Math.max(1, xe - xs), h - top - bottom + 7);
    drawVLine(xOf(s), top - 6, h - bottom + 4, "#147a3f", 2);
    drawVLine(xOf(e), top - 6, h - bottom + 4, "#147a3f", 2);
  }
  if (state.start != null) drawVLine(xOf(state.start), top - 12, h - bottom + 10, "#b45309", 3);
  if (state.anchor != null) drawVLine(xOf(state.anchor), top - 12, h - bottom + 10, "#6d28d9", 2);
  drawVLine(xOf(state.t), top - 14, h - bottom + 12, "#000", 2);

  ctx.fillStyle = "#111827";
  ctx.font = "12px Arial";
  for (let frac = 0; frac <= 1.001; frac += 0.1) {
    const t = loT + frac * (hiT - loT);
    const x = xOf(t);
    ctx.fillRect(x, h - bottom + 8, 1, 6);
    if (Math.abs((frac * 10) % 2) < 0.01) ctx.fillText(`${t.toFixed(0)}s`, x - 14, h - 8);
  }
}

function clampTime(t) {
  const duration = state.data ? Number(state.data.duration_s) : 0;
  return Math.max(0, Math.min(duration, t));
}

function seek(delta) {
  state.t = clampTime(state.t + delta);
  updateReadouts();
  draw();
}

function markStart() {
  state.start = state.t;
  state.end = null;
  if (state.anchor == null) state.anchor = state.t;
  updateReadouts();
  draw();
}

function markAnchor() {
  state.anchor = state.t;
  updateReadouts();
  draw();
}

async function markEndAndSave() {
  if (state.start == null || !state.data) {
    setStatus("mark start first");
    return;
  }
  state.end = state.t;
  const payload = {
    subject: state.data.subject,
    session_stamp: state.data.session_stamp,
    event_start_rel_s: state.start,
    event_end_rel_s: state.end,
    anchor_rel_s: state.anchor == null ? state.start : state.anchor,
    event_type: $("eventType").value,
    direction: $("direction").value,
    confidence_1_5: $("confidence").value,
    note: $("note").value
  };
  setStatus("saving");
  const saved = await api("/api/label", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  state.labels.push(saved);
  state.start = null;
  state.end = null;
  state.anchor = null;
  updateReadouts();
  renderLabels();
  draw();
  setStatus("saved");
}

async function undoLast() {
  setStatus("undo");
  await api("/api/undo", {method: "POST"});
  await loadLabels();
  draw();
  setStatus("ready");
}

function renderLabels() {
  const tbody = $("labels").querySelector("tbody");
  tbody.innerHTML = "";
  const rows = state.labels.slice(-18).reverse();
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${row.subject}/${row.session_stamp}</td><td>${row.event_start_rel_s}</td><td>${row.event_end_rel_s}</td><td>${row.event_type}</td><td>${row.direction}</td>`;
    tbody.appendChild(tr);
  }
  updateReadouts();
}

function togglePlay() {
  state.playing = !state.playing;
  $("playBtn").textContent = state.playing ? "Pause" : "Play";
  state.lastFrame = performance.now();
}

function tick(now) {
  if (state.playing && state.data) {
    const dt = (now - state.lastFrame) / 1000;
    state.t = clampTime(state.t + dt * state.speed);
    if (state.t >= Number(state.data.duration_s)) {
      state.playing = false;
      $("playBtn").textContent = "Play";
    }
    updateReadouts();
    draw();
  }
  state.lastFrame = now;
  requestAnimationFrame(tick);
}

function isEditingTarget(target) {
  const tag = target.tagName ? target.tagName.toLowerCase() : "";
  return tag === "input" || tag === "textarea" || tag === "select";
}

document.addEventListener("keydown", async (ev) => {
  if (isEditingTarget(ev.target)) return;
  const k = ev.key.toLowerCase();
  if (ev.code === "Space") { ev.preventDefault(); togglePlay(); return; }
  if (k === "a") { markStart(); return; }
  if (k === "s") { markAnchor(); return; }
  if (k === "d") { await markEndAndSave(); return; }
  if (k === "u") { await undoLast(); return; }
  if (k === "n") { await loadSession(Math.min(state.sessions.length - 1, state.sessionIndex + 1)); return; }
  if (k === "p") { await loadSession(Math.max(0, state.sessionIndex - 1)); return; }
  if (ev.key === "ArrowLeft") { ev.preventDefault(); seek(ev.shiftKey ? -2 : -0.2); return; }
  if (ev.key === "ArrowRight") { ev.preventDefault(); seek(ev.shiftKey ? 2 : 0.2); return; }
});

$("sessionSelect").addEventListener("change", e => loadSession(Number(e.target.value)));
$("prevBtn").addEventListener("click", () => loadSession(Math.max(0, state.sessionIndex - 1)));
$("nextBtn").addEventListener("click", () => loadSession(Math.min(state.sessions.length - 1, state.sessionIndex + 1)));
$("playBtn").addEventListener("click", togglePlay);
$("speedSelect").addEventListener("change", e => state.speed = Number(e.target.value));
$("windowSelect").addEventListener("change", e => { state.viewWindow = Number(e.target.value); draw(); });
$("markStartBtn").addEventListener("click", markStart);
$("markAnchorBtn").addEventListener("click", markAnchor);
$("markEndBtn").addEventListener("click", markEndAndSave);
$("backBtn").addEventListener("click", () => seek(-0.2));
$("forwardBtn").addEventListener("click", () => seek(0.2));
$("undoBtn").addEventListener("click", undoLast);
window.addEventListener("resize", resizeCanvas);

(async function init() {
  try {
    await loadSessions();
    await loadLabels();
    resizeCanvas();
    requestAnimationFrame(tick);
  } catch (err) {
    setStatus(`error: ${err.message}`);
    console.error(err);
  }
})();
</script>
</body>
</html>
"""


FOCUSED_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>R2E Vehicle Instability Event Reviewer</title>
  <style>
    :root {
      --bg: #f5f6f8;
      --panel: #ffffff;
      --ink: #121826;
      --muted: #667085;
      --line: #d7dce3;
      --blue: #1d5fd1;
      --green: #167443;
      --orange: #ba6b00;
      --red: #c3342b;
      --purple: #6d35c5;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, "Microsoft YaHei", sans-serif;
    }
    header {
      height: 58px;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      background: #fff;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { font-size: 17px; margin: 0; white-space: nowrap; }
    button, select, input, textarea {
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
    }
    button, select, input { height: 34px; padding: 0 9px; }
    button { cursor: pointer; }
    button.primary { background: var(--blue); border-color: var(--blue); color: #fff; }
    button.ok { background: var(--green); border-color: var(--green); color: #fff; }
    button.warn { background: var(--orange); border-color: var(--orange); color: #fff; }
    button.ghost { background: #f1f3f6; }
    main {
      display: grid;
      grid-template-columns: minmax(780px, 1fr) 360px;
      gap: 12px;
      padding: 12px;
    }
    .plot, .side {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .plot-head {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      flex-wrap: wrap;
    }
    #plot {
      width: 100%;
      height: calc(100vh - 152px);
      min-height: 620px;
      display: block;
      background: #fff;
    }
    .side {
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      max-height: calc(100vh - 84px);
      overflow: auto;
    }
    .section {
      padding-bottom: 12px;
      border-bottom: 1px solid var(--line);
    }
    .section:last-child { border-bottom: 0; }
    .candidate {
      background: #eef5ff;
      border: 1px solid #b7d0fb;
      border-radius: 8px;
      padding: 10px;
    }
    .candidate h2 {
      font-size: 15px;
      margin: 0 0 8px;
    }
    .reason {
      color: #344054;
      line-height: 1.35;
      font-size: 13px;
    }
    .grid2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .metric {
      border-radius: 6px;
      background: #f4f6f8;
      padding: 7px;
      font-size: 12px;
      color: var(--muted);
    }
    .metric strong {
      display: block;
      margin-top: 3px;
      font-size: 14px;
      color: var(--ink);
    }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .row > * { flex: 1 1 auto; }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .field { margin-bottom: 8px; }
    .field select, .field textarea { width: 100%; }
    textarea { min-height: 54px; padding: 8px; resize: vertical; }
    .pill {
      display: inline-block;
      border-radius: 999px;
      padding: 3px 8px;
      background: #eef1f5;
      color: var(--muted);
      white-space: nowrap;
    }
    .pill.blue { background: #e7f0ff; color: var(--blue); }
    .pill.green { background: #e7f6ee; color: var(--green); }
    .pill.red { background: #fdecea; color: var(--red); }
    .status { font-weight: 700; color: var(--blue); }
    .keys {
      display: grid;
      grid-template-columns: 44px 1fr;
      gap: 5px 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .legend {
      display: grid;
      grid-template-columns: 18px 1fr;
      gap: 6px 8px;
      color: #344054;
      font-size: 12px;
      line-height: 1.35;
    }
    .swatch {
      width: 4px;
      height: 20px;
      border-radius: 2px;
      justify-self: center;
    }
    .swatch.blue { background: var(--blue); width: 7px; }
    .swatch.thin-blue { background: var(--blue); }
    .swatch.orange { background: var(--orange); }
    .swatch.red { background: var(--red); }
    .swatch.black { background: #000; }
    .swatch.green { background: var(--green); }
    .swatch.purple { background: var(--purple); }
    .key {
      text-align: center;
      background: #eef1f5;
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 2px 4px;
      color: var(--ink);
      font-weight: 700;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      font-size: 12px;
    }
    td, th {
      border-bottom: 1px solid var(--line);
      padding: 5px 4px;
      text-align: left;
      white-space: nowrap;
    }
  </style>
</head>
<body>
  <header>
    <h1>R2E 车辆失稳候选审查</h1>
    <select id="sessionSelect"></select>
    <button id="prevSessionBtn">上一记录</button>
    <button id="nextSessionBtn">下一记录</button>
    <button id="prevSegBtn" class="ghost">上一候选 Q</button>
    <button id="nextSegBtn" class="ghost">下一候选 E</button>
    <button id="playBtn" class="primary">播放</button>
    <select id="speedSelect">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="5">5x</option>
      <option value="10">10x</option>
    </select>
    <span id="status" class="status">loading</span>
  </header>
  <main>
    <section class="plot">
      <div class="plot-head">
        <span id="sessionTitle"></span>
        <span class="pill" id="timeReadout">0.000s</span>
        <span class="pill blue" id="candidateReadout">候选 0/0</span>
        <span class="pill">蓝色背景=当前失稳片段</span>
        <span class="pill red">红线=失稳锚点</span>
        <span class="pill">方向盘只看事件后响应</span>
        <span class="pill green">绿=已保存</span>
      </div>
      <canvas id="plot"></canvas>
    </section>
    <aside class="side">
      <section class="section candidate">
        <h2 id="segmentTitle">当前候选段</h2>
        <div class="reason" id="segmentReason">等待加载</div>
      </section>
      <section class="section">
        <div class="grid2">
          <div class="metric">建议开始<strong id="segStart">-</strong></div>
          <div class="metric">建议结束<strong id="segEnd">-</strong></div>
          <div class="metric">建议锚点<strong id="segAnchor">-</strong></div>
          <div class="metric">已保存<strong id="savedCount">0</strong></div>
        </div>
      </section>
      <section class="section">
        <div class="row">
          <button id="acceptBtn" class="ok">采用候选 Y</button>
          <button id="markStartBtn" class="warn">开始 A</button>
          <button id="markAnchorBtn">锚点 S</button>
        </div>
        <div class="row" style="margin-top:8px;">
          <button id="saveManualBtn" class="ok">结束保存 D</button>
          <button id="undoBtn">撤销 U</button>
        </div>
        <div class="row" style="margin-top:8px;">
          <button id="backBtn">-0.2s</button>
          <button id="forwardBtn">+0.2s</button>
          <button id="jumpStartBtn">回到候选开头</button>
        </div>
      </section>
      <section class="section">
        <div class="grid2">
          <div class="metric">手动开始<strong id="manualStart">-</strong></div>
          <div class="metric">手动锚点<strong id="manualAnchor">-</strong></div>
        </div>
      </section>
      <section class="section">
        <div class="legend">
          <span class="swatch blue"></span><span>浅蓝背景：当前正在审查的车辆失稳候选片段。</span>
          <span class="swatch red"></span><span>红线：失稳锚点，来自横向加速度或横滚速率，不来自方向盘。</span>
          <span class="swatch thin-blue"></span><span>蓝线：当前候选片段起止边界。</span>
          <span class="swatch orange"></span><span>橙线：手动调整的候选开始点。</span>
          <span class="swatch black"></span><span>黑线：当前播放时间。</span>
          <span class="swatch purple"></span><span>紫线：你手动标的预测锚点。</span>
          <span class="swatch green"></span><span>绿线/绿背景：已经保存的人工标签。</span>
        </div>
      </section>
      <section class="section">
        <div class="field">
          <label>事件类型</label>
          <select id="eventType">
            <option value="vehicle_instability">vehicle_instability</option>
            <option value="instability_ay_only">instability_ay_only</option>
            <option value="instability_roll_only">instability_roll_only</option>
            <option value="instability_ay_roll">instability_ay_roll</option>
            <option value="curve">curve</option>
            <option value="lane_change">lane_change</option>
            <option value="avoidance">avoidance</option>
            <option value="return">return</option>
            <option value="multi_stage">multi_stage</option>
            <option value="unclear">unclear</option>
          </select>
        </div>
        <div class="field">
          <label>方向</label>
          <select id="direction">
            <option value="left">left</option>
            <option value="right">right</option>
            <option value="straight">straight</option>
            <option value="unclear">unclear</option>
          </select>
        </div>
        <div class="field">
          <label>置信度</label>
          <select id="confidence">
            <option value="5">5</option>
            <option value="4" selected>4</option>
            <option value="3">3</option>
            <option value="2">2</option>
            <option value="1">1</option>
          </select>
        </div>
        <div class="field">
          <label>备注</label>
          <textarea id="note"></textarea>
        </div>
      </section>
      <section class="section">
        <div class="keys">
          <span class="key">Y</span><span>采用当前候选段</span>
          <span class="key">Q/E</span><span>上一/下一候选段</span>
          <span class="key">Space</span><span>播放/暂停当前窗口</span>
          <span class="key">A/S/D</span><span>手动开始/锚点/结束保存</span>
          <span class="key">←/→</span><span>微调当前时间，Shift 为大步</span>
        </div>
      </section>
      <section class="section">
        <a href="/api/labels.csv" target="_blank">打开标签 CSV</a>
      </section>
      <section class="section">
        <table>
          <thead><tr><th>记录</th><th>开始</th><th>结束</th><th>来源</th></tr></thead>
          <tbody id="labelsBody"></tbody>
        </table>
      </section>
    </aside>
  </main>
<script>
const state = {
  sessions: [],
  sessionIndex: 0,
  data: null,
  labels: [],
  segmentIndex: 0,
  t: 0,
  playing: false,
  speed: 1,
  start: null,
  anchor: null,
  lastFrame: performance.now()
};

const $ = (id) => document.getElementById(id);
const canvas = $("plot");
const ctx = canvas.getContext("2d");

function setStatus(text) { $("status").textContent = text; }
function fmt(x) { return x == null || Number.isNaN(Number(x)) ? "-" : `${Number(x).toFixed(3)}s`; }
function currentSegment() {
  if (!state.data || !state.data.review_segments || !state.data.review_segments.length) return null;
  return state.data.review_segments[Math.max(0, Math.min(state.segmentIndex, state.data.review_segments.length - 1))];
}
async function api(path, options = {}) {
  const resp = await fetch(path, options);
  if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
  return resp.json();
}

async function loadSessions() {
  state.sessions = await api("/api/sessions");
  const select = $("sessionSelect");
  select.innerHTML = "";
  state.sessions.forEach((s, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${s.subject} ${s.session_stamp} | road ${s.raw_road_curvature_onset}`;
    select.appendChild(opt);
  });
  await loadSession(0);
}

async function loadLabels() {
  state.labels = await api("/api/labels");
  renderLabels();
}

async function loadSession(index) {
  if (index < 0 || index >= state.sessions.length) return;
  state.sessionIndex = index;
  $("sessionSelect").value = index;
  const s = state.sessions[index];
  setStatus("loading");
  state.data = await api(`/api/session?subject=${encodeURIComponent(s.subject)}&session_stamp=${encodeURIComponent(s.session_stamp)}`);
  state.segmentIndex = 0;
  state.playing = false;
  $("playBtn").textContent = "播放";
  $("sessionTitle").textContent = `${s.subject} / ${s.session_stamp}`;
  gotoSegment(0, false);
  setStatus("ready");
}

function gotoSegment(index, autoplay = false) {
  if (!state.data) return;
  const n = state.data.review_segments.length;
  state.segmentIndex = Math.max(0, Math.min(n - 1, index));
  const seg = currentSegment();
  state.start = null;
  state.anchor = null;
  if (seg) {
    state.t = Number(seg.review_start_rel_s);
    $("eventType").value = normalizeEventType(seg.event_type);
    $("direction").value = seg.direction || "unclear";
  } else {
    state.t = 0;
  }
  state.playing = autoplay;
  $("playBtn").textContent = state.playing ? "暂停" : "播放";
  updatePanel();
  draw();
}

function normalizeEventType(value) {
  value = String(value || "unclear");
  if (value.includes("instability_ay_roll")) return "instability_ay_roll";
  if (value.includes("instability_roll_only")) return "instability_roll_only";
  if (value.includes("instability_ay_only")) return "instability_ay_only";
  if (value.includes("instability") || value.includes("vehicle")) return "vehicle_instability";
  if (value.includes("curve")) return "curve";
  if (value.includes("lane")) return "lane_change";
  if (value.includes("return")) return "return";
  if (value.includes("multi")) return "multi_stage";
  return "unclear";
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function viewBounds() {
  const seg = currentSegment();
  if (!seg) return [0, state.data ? Number(state.data.duration_s) : 1];
  return [Number(seg.review_start_rel_s), Number(seg.review_end_rel_s)];
}

function updatePanel() {
  const seg = currentSegment();
  const total = state.data ? state.data.review_segments.length : 0;
  $("candidateReadout").textContent = total ? `候选 ${state.segmentIndex + 1}/${total}` : "候选 0/0";
  $("timeReadout").textContent = fmt(state.t);
  $("savedCount").textContent = String(state.labels.length);
  $("manualStart").textContent = fmt(state.start);
  $("manualAnchor").textContent = fmt(state.anchor);
  if (!seg) {
    $("segmentTitle").textContent = "没有候选段";
    $("segmentReason").textContent = "当前记录没有可审查候选。";
    $("segStart").textContent = "-";
    $("segEnd").textContent = "-";
    $("segAnchor").textContent = "-";
    return;
  }
  $("segmentTitle").textContent = `${seg.review_segment_id} | ${seg.candidate_source}`;
  $("segmentReason").textContent = seg.reason;
  $("segStart").textContent = fmt(seg.event_start_rel_s);
  $("segEnd").textContent = fmt(seg.event_end_rel_s);
  $("segAnchor").textContent = fmt(seg.anchor_time_rel_s);
}

function colorForSource(source) {
  if (source === "raw_vehicle_instability_onset") return "#c3342b";
  if (source === "raw_road_curvature_onset") return "#1d5fd1";
  if (source === "old_v400_context_trigger_idx") return "#ba6b00";
  if (source === "raw_vehicle_dynamic_onset") return "#c3342b";
  return "#777";
}

function panelLimits(values, loT, hiT, times) {
  const vals = [];
  for (let i = 0; i < times.length; i++) {
    const v = values[i], t = times[i];
    if (t >= loT && t <= hiT && v != null && Number.isFinite(v)) vals.push(v);
  }
  if (!vals.length) return [-1, 1];
  vals.sort((a, b) => a - b);
  const p = q => vals[Math.min(vals.length - 1, Math.max(0, Math.floor(q * (vals.length - 1))))];
  let lo = Math.min(p(0.01), 0), hi = Math.max(p(0.99), 0);
  if (Math.abs(hi - lo) < 1e-9) { lo -= 1; hi += 1; }
  const pad = 0.08 * (hi - lo);
  return [lo - pad, hi + pad];
}

function drawLine(x, y0, y1, color, width = 1) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x, y0);
  ctx.lineTo(x, y1);
  ctx.stroke();
}

function draw() {
  if (!state.data) return;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  const [loT, hiT] = viewBounds();
  const marginL = 118, marginR = 28, top = 28, bottom = 34;
  const plotW = w - marginL - marginR;
  const n = state.data.signals.length;
  const gap = 8;
  const panelH = (h - top - bottom - gap * (n - 1)) / n;
  const xOf = t => marginL + (Number(t) - loT) / Math.max(hiT - loT, 1e-9) * plotW;
  const times = state.data.time_rel_s;
  const seg = currentSegment();

  if (seg) {
    const xs = xOf(seg.event_start_rel_s);
    const xe = xOf(seg.event_end_rel_s);
    ctx.fillStyle = "rgba(29, 95, 209, 0.08)";
    ctx.fillRect(Math.max(marginL, xs), top - 12, Math.min(marginL + plotW, xe) - Math.max(marginL, xs), h - top - bottom + 22);
  }

  for (let p = 0; p < n; p++) {
    const sig = state.data.signals[p];
    const y0 = top + p * (panelH + gap);
    const y1 = y0 + panelH;
    ctx.strokeStyle = "#d7dce3";
    ctx.lineWidth = 1;
    ctx.strokeRect(marginL, y0, plotW, panelH);
    ctx.fillStyle = "#121826";
    ctx.font = "13px Arial";
    ctx.fillText(sig.label, 10, y0 + 18);
    const [vLo, vHi] = panelLimits(sig.values, loT, hiT, times);
    const yOf = v => y1 - (v - vLo) / Math.max(vHi - vLo, 1e-9) * panelH;
    if (vLo <= 0 && vHi >= 0) {
      const zy = yOf(0);
      ctx.strokeStyle = "#e7eaf0";
      ctx.beginPath();
      ctx.moveTo(marginL, zy);
      ctx.lineTo(marginL + plotW, zy);
      ctx.stroke();
    }
    const colors = {curvature:"#1d5fd1", steer:"#121826", speed:"#167443", lateral:"#6d35c5", yaw_rate:"#c3342b", ay:"#ba6b00", roll:"#8b5e34", roll_rate:"#7f56d9"};
    ctx.strokeStyle = colors[sig.name] || "#121826";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < times.length; i++) {
      const t = times[i], v = sig.values[i];
      if (t < loT || t > hiT || v == null || !Number.isFinite(v)) continue;
      const x = xOf(t), y = yOf(v);
      if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
    }
    if (started) ctx.stroke();
    ctx.fillStyle = "#667085";
    ctx.font = "11px Arial";
    ctx.fillText(`${vLo.toPrecision(3)} to ${vHi.toPrecision(3)}`, w - 116, y0 + 15);
  }

  for (const ev of state.data.events) {
    const t = Number(ev.anchor_time_rel_s);
    if (!Number.isFinite(t) || t < loT || t > hiT) continue;
    const selected = seg && ev.event_uid === seg.candidate_event_uid;
    drawLine(xOf(t), top - 13, h - bottom + 10, colorForSource(ev.anchor_source), selected ? 4 : ev.anchor_source === "raw_road_curvature_onset" ? 2 : 1);
  }
  const sessionLabels = state.labels.filter(l => l.subject === state.data.subject && l.session_stamp === state.data.session_stamp);
  for (const lab of sessionLabels) {
    const s = Number(lab.event_start_rel_s), e = Number(lab.event_end_rel_s);
    if (!Number.isFinite(s) || !Number.isFinite(e) || e < loT || s > hiT) continue;
    const xs = xOf(Math.max(s, loT)), xe = xOf(Math.min(e, hiT));
    ctx.fillStyle = "rgba(22, 116, 67, 0.16)";
    ctx.fillRect(xs, top - 9, Math.max(2, xe - xs), h - top - bottom + 17);
    drawLine(xOf(s), top - 13, h - bottom + 10, "#167443", 2);
    drawLine(xOf(e), top - 13, h - bottom + 10, "#167443", 2);
  }
  if (seg) {
    drawLine(xOf(seg.event_start_rel_s), top - 15, h - bottom + 12, "#1d5fd1", 3);
    drawLine(xOf(seg.event_end_rel_s), top - 15, h - bottom + 12, "#1d5fd1", 3);
  }
  if (state.start != null) drawLine(xOf(state.start), top - 17, h - bottom + 14, "#ba6b00", 3);
  if (state.anchor != null) drawLine(xOf(state.anchor), top - 17, h - bottom + 14, "#6d35c5", 2);
  drawLine(xOf(state.t), top - 18, h - bottom + 16, "#000", 2);

  ctx.fillStyle = "#121826";
  ctx.font = "12px Arial";
  for (let frac = 0; frac <= 1.001; frac += 0.1) {
    const t = loT + frac * (hiT - loT);
    const x = xOf(t);
    ctx.fillRect(x, h - bottom + 8, 1, 6);
    if (Math.abs((frac * 10) % 2) < 0.01) ctx.fillText(`${t.toFixed(0)}s`, x - 14, h - 8);
  }
  updatePanel();
}

function clampTime(t) {
  const [loT, hiT] = viewBounds();
  return Math.max(loT, Math.min(hiT, t));
}
function seek(delta) {
  state.t = clampTime(state.t + delta);
  updatePanel();
  draw();
}
function togglePlay() {
  state.playing = !state.playing;
  $("playBtn").textContent = state.playing ? "暂停" : "播放";
  state.lastFrame = performance.now();
}
function markStart() {
  state.start = state.t;
  if (state.anchor == null) state.anchor = state.t;
  updatePanel();
  draw();
}
function markAnchor() {
  state.anchor = state.t;
  updatePanel();
  draw();
}

async function savePayload(payload, autoNext = true) {
  setStatus("saving");
  const saved = await api("/api/label", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  state.labels.push(saved);
  state.start = null;
  state.anchor = null;
  renderLabels();
  setStatus("saved");
  if (autoNext) gotoSegment(state.segmentIndex + 1, false);
  else draw();
}

async function acceptCandidate() {
  const seg = currentSegment();
  if (!state.data || !seg) return;
  await savePayload({
    subject: state.data.subject,
    session_stamp: state.data.session_stamp,
    decision: "accept_candidate",
    review_segment_id: seg.review_segment_id,
    event_start_rel_s: seg.event_start_rel_s,
    event_end_rel_s: seg.event_end_rel_s,
    anchor_rel_s: seg.anchor_time_rel_s,
    event_type: $("eventType").value || seg.event_type,
    direction: $("direction").value || seg.direction,
    confidence_1_5: $("confidence").value,
    note: $("note").value,
    selected_candidate_event_uid: seg.candidate_event_uid,
    selected_candidate_source: seg.candidate_source,
    selected_candidate_reason: seg.reason
  }, true);
}

async function saveManualEnd() {
  if (!state.data || state.start == null) {
    setStatus("mark start first");
    return;
  }
  const seg = currentSegment();
  await savePayload({
    subject: state.data.subject,
    session_stamp: state.data.session_stamp,
    decision: "manual_adjusted",
    review_segment_id: seg ? seg.review_segment_id : "",
    event_start_rel_s: state.start,
    event_end_rel_s: state.t,
    anchor_rel_s: state.anchor == null ? state.start : state.anchor,
    event_type: $("eventType").value,
    direction: $("direction").value,
    confidence_1_5: $("confidence").value,
    note: $("note").value,
    selected_candidate_event_uid: seg ? seg.candidate_event_uid : "",
    selected_candidate_source: seg ? seg.candidate_source : "",
    selected_candidate_reason: seg ? seg.reason : ""
  }, true);
}

async function undoLast() {
  setStatus("undo");
  await api("/api/undo", {method:"POST"});
  await loadLabels();
  draw();
  setStatus("ready");
}

function renderLabels() {
  const body = $("labelsBody");
  body.innerHTML = "";
  for (const row of state.labels.slice(-14).reverse()) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${row.subject}/${row.session_stamp}</td><td>${row.event_start_rel_s}</td><td>${row.event_end_rel_s}</td><td>${row.selected_candidate_source || row.decision}</td>`;
    body.appendChild(tr);
  }
  updatePanel();
}

function isEditingTarget(target) {
  const tag = target.tagName ? target.tagName.toLowerCase() : "";
  return tag === "input" || tag === "textarea" || tag === "select";
}

function tick(now) {
  if (state.playing && state.data) {
    const dt = (now - state.lastFrame) / 1000;
    const [, hiT] = viewBounds();
    state.t = clampTime(state.t + dt * state.speed);
    if (state.t >= hiT) {
      state.playing = false;
      $("playBtn").textContent = "播放";
    }
    updatePanel();
    draw();
  }
  state.lastFrame = now;
  requestAnimationFrame(tick);
}

document.addEventListener("keydown", async (ev) => {
  if (isEditingTarget(ev.target)) return;
  const k = ev.key.toLowerCase();
  if (ev.code === "Space") { ev.preventDefault(); togglePlay(); return; }
  if (k === "q") { gotoSegment(state.segmentIndex - 1, false); return; }
  if (k === "e") { gotoSegment(state.segmentIndex + 1, false); return; }
  if (k === "y") { await acceptCandidate(); return; }
  if (k === "a") { markStart(); return; }
  if (k === "s") { markAnchor(); return; }
  if (k === "d") { await saveManualEnd(); return; }
  if (k === "u") { await undoLast(); return; }
  if (k === "n") { await loadSession(Math.min(state.sessions.length - 1, state.sessionIndex + 1)); return; }
  if (k === "p") { await loadSession(Math.max(0, state.sessionIndex - 1)); return; }
  if (ev.key === "ArrowLeft") { ev.preventDefault(); seek(ev.shiftKey ? -2 : -0.2); return; }
  if (ev.key === "ArrowRight") { ev.preventDefault(); seek(ev.shiftKey ? 2 : 0.2); return; }
});

$("sessionSelect").addEventListener("change", e => loadSession(Number(e.target.value)));
$("prevSessionBtn").addEventListener("click", () => loadSession(Math.max(0, state.sessionIndex - 1)));
$("nextSessionBtn").addEventListener("click", () => loadSession(Math.min(state.sessions.length - 1, state.sessionIndex + 1)));
$("prevSegBtn").addEventListener("click", () => gotoSegment(state.segmentIndex - 1, false));
$("nextSegBtn").addEventListener("click", () => gotoSegment(state.segmentIndex + 1, false));
$("playBtn").addEventListener("click", togglePlay);
$("speedSelect").addEventListener("change", e => state.speed = Number(e.target.value));
$("acceptBtn").addEventListener("click", acceptCandidate);
$("markStartBtn").addEventListener("click", markStart);
$("markAnchorBtn").addEventListener("click", markAnchor);
$("saveManualBtn").addEventListener("click", saveManualEnd);
$("undoBtn").addEventListener("click", undoLast);
$("backBtn").addEventListener("click", () => seek(-0.2));
$("forwardBtn").addEventListener("click", () => seek(0.2));
$("jumpStartBtn").addEventListener("click", () => { const seg = currentSegment(); if (seg) { state.t = Number(seg.review_start_rel_s); draw(); } });
window.addEventListener("resize", resizeCanvas);

(async function init() {
  try {
    await loadSessions();
    await loadLabels();
    resizeCanvas();
    requestAnimationFrame(tick);
  } catch (err) {
    console.error(err);
    setStatus(`error: ${err.message}`);
  }
})();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "R2EKeyboardLabeler/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        msg = "%s - %s\n" % (datetime.now().isoformat(timespec="seconds"), fmt % args)
        with (LOG_DIR / "player_server.access.log").open("a", encoding="utf-8") as f:
            f.write(msg)

    def send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, obj: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"error": message}, status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_bytes(FOCUSED_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/legacy":
                self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/sessions":
                self.send_json(get_sessions())
            elif parsed.path == "/api/session":
                qs = parse_qs(parsed.query)
                subject = qs.get("subject", [""])[0]
                stamp = qs.get("session_stamp", [""])[0]
                self.send_json(load_session_payload(subject, stamp))
            elif parsed.path == "/api/labels":
                self.send_json(read_labels())
            elif parsed.path == "/api/labels.csv":
                data = LABEL_PATH.read_bytes()
                self.send_bytes(data, "text/csv; charset=utf-8")
            else:
                self.send_error_json(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/label":
                self.send_json(append_label(self.read_json_body()))
            elif parsed.path == "/api/undo":
                self.send_json(undo_last_label())
            else:
                self.send_error_json(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, f"{type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    ensure_dirs()
    write_report(args.host, args.port)
    (LOG_DIR / "player_server.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"url": f"http://{args.host}:{args.port}/", "label_path": str(LABEL_PATH)}, ensure_ascii=False))
    server.serve_forever()


if __name__ == "__main__":
    main()
