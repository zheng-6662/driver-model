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
CANDIDATE_PATH = ROOT / "02_samples" / "tables" / "candidate_events_master.csv"
OUT_DIR = ROOT / "02_samples" / "manual_event_keyboard_player_v0_1"
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"
LABEL_PATH = TABLE_DIR / "keyboard_event_labels_v0_1.csv"

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
]

LABEL_FIELDS = [
    "label_id",
    "created_at",
    "subject",
    "session_stamp",
    "event_start_rel_s",
    "event_end_rel_s",
    "anchor_rel_s",
    "event_type",
    "direction",
    "confidence_1_5",
    "note",
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
    df = pd.read_csv(CANDIDATE_PATH)
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
    for col in ["raw_road_curvature_onset", "old_v400_context_trigger_idx", "raw_vehicle_dynamic_onset"]:
        if col not in grouped.columns:
            grouped[col] = 0
    grouped["sort_road"] = grouped["raw_road_curvature_onset"].astype(int)
    grouped = grouped.sort_values(["sort_road", "subject", "session_stamp"], ascending=[False, True, True])
    sessions: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        subject = str(row["subject"])
        stamp = str(row["session_stamp"])
        sessions.append(
            {
                "key": session_key(subject, stamp),
                "subject": subject,
                "session_stamp": stamp,
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
        "event_start_rel_s": f"{start:.3f}",
        "event_end_rel_s": f"{end:.3f}",
        "anchor_rel_s": f"{anchor:.3f}",
        "event_type": str(payload.get("event_type", "")).strip(),
        "direction": str(payload.get("direction", "")).strip(),
        "confidence_1_5": str(payload.get("confidence_1_5", "")).strip(),
        "note": str(payload.get("note", "")).replace("\r", " ").replace("\n", " ").strip(),
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
    report = f"""# 阶段 2 补充：键盘式人工事件标注播放器 v0.1

生成时间：2026-05-12

## 为什么做

人工填写整张事件表太复杂，因此新增本地键盘标注播放器。用户可以播放原始车辆时间线，用键盘标记事件开始和结束，标签由本地 Python 服务写入 CSV。

## 使用入口

- 本地页面：`http://{host}:{port}/`
- 标签输出：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_event_labels_v0_1.csv`
- 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/run_manual_event_keyboard_player.py`

## 默认按键

- 空格：播放/暂停。
- `A`：把当前时间标记为事件开始。
- `S`：把当前时间标记为预测锚点；不按则默认锚点等于开始时间。
- `D`：把当前时间标记为事件结束并保存一行标签。
- 左/右方向键：小步后退/前进；按住 Shift 为大步。
- `N` / `P`：切换下一条/上一条记录。
- `U`：撤销最后一条保存的标签。

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
