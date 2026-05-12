# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
TABLE_DIR = ROOT / "02_samples" / "tables"
OUT_DIR = ROOT / "02_samples" / "manual_event_label_review_v0_1"
FIG_DIR = OUT_DIR / "figures"
OUT_TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"

CANDIDATE_PATH = TABLE_DIR / "candidate_events_master.csv"

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

PANELS = [
    ("zx1|lanecurvatureXY", "lane curvature", "#1f77b4"),
    ("zx|SteeringWheel", "steering wheel", "#111111"),
    ("zx1|v_km/h", "speed km/h", "#2ca02c"),
    ("zx1|lateraldistance", "lateral distance", "#9467bd"),
    ("zx|vyaw", "yaw rate", "#d62728"),
    ("zx|ay", "lateral accel ay", "#ff7f0e"),
    ("zx|roll", "roll", "#8c564b"),
]

SOURCE_COLORS = {
    "raw_road_curvature_onset": (40, 110, 210),
    "old_v400_context_trigger_idx": (230, 140, 40),
    "raw_vehicle_dynamic_onset": (210, 50, 50),
}


def ensure_dirs() -> None:
    for path in [FIG_DIR, OUT_TABLE_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def get_fonts() -> tuple[Any, Any, Any]:
    try:
        return (
            ImageFont.truetype("arial.ttf", 26),
            ImageFont.truetype("arial.ttf", 18),
            ImageFont.truetype("arial.ttf", 14),
        )
    except OSError:
        font = ImageFont.load_default()
        return font, font, font


def to_seconds(storage_time: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(storage_time, errors="coerce")
    out = np.full(len(storage_time), np.nan, dtype=np.float64)
    valid = parsed.notna().to_numpy()
    if valid.any():
        parsed_ns = parsed[valid].astype("datetime64[ns]")
        ns = parsed_ns.astype("int64").to_numpy(dtype=np.float64)
        out[valid] = ns / 1e9
    return out


def color_from_hex(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def pick_review_sessions(candidates: pd.DataFrame, max_sessions: int) -> pd.DataFrame:
    road = candidates[candidates["anchor_source"] == "raw_road_curvature_onset"].copy()
    if road.empty:
        road = candidates.copy()
    count = (
        road.groupby(["subject", "session_stamp"], dropna=False)
        .agg(
            raw_road_count=("event_uid", "count"),
            first_anchor_s=("anchor_time_rel_s", "min"),
        )
        .reset_index()
    )
    all_counts = (
        candidates.groupby(["subject", "session_stamp", "anchor_source"], dropna=False)["event_uid"]
        .count()
        .unstack(fill_value=0)
        .reset_index()
    )
    selected = count.merge(all_counts, on=["subject", "session_stamp"], how="left")
    selected = selected.sort_values(["raw_road_count", "subject", "session_stamp"], ascending=[False, True, True])
    return selected.head(max_sessions).reset_index(drop=True)


def read_vehicle(path: Path) -> tuple[pd.DataFrame, str]:
    try:
        df = pd.read_csv(path, usecols=lambda c: c in VEHICLE_COLS)
    except Exception as exc:
        return pd.DataFrame(), f"read_error:{type(exc).__name__}:{exc}"
    if "StorageTime" not in df.columns:
        return pd.DataFrame(), "missing_storage_time"
    abs_s = to_seconds(df["StorageTime"])
    if not np.isfinite(abs_s).any():
        return pd.DataFrame(), "unparseable_storage_time"
    df = df.copy()
    df["time_abs_s"] = abs_s
    df["time_rel_s"] = abs_s - float(np.nanmin(abs_s))
    return df.sort_values("time_rel_s").reset_index(drop=True), "ok"


def thin_points(t: np.ndarray, y: np.ndarray, max_points: int = 4500) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if len(t) <= max_points:
        return t, y
    idx = np.linspace(0, len(t) - 1, max_points).astype(int)
    return t[idx], y[idx]


def panel_limits(y: np.ndarray) -> tuple[float, float]:
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.nanpercentile(finite, 1))
    hi = float(np.nanpercentile(finite, 99))
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    if abs(hi - lo) < 1e-9:
        lo -= 1.0
        hi += 1.0
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def x_of_time(t: float, t_min: float, t_max: float, x0: int, x1: int) -> int:
    if t_max <= t_min:
        return x0
    return int(x0 + (float(t) - t_min) / (t_max - t_min) * (x1 - x0))


def draw_timeline(
    df: pd.DataFrame,
    events: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    title_font, font, small = get_fonts()
    width = 1900
    panel_h = 170
    top = 105
    left = 160
    right = 55
    bottom = 60
    height = top + len(PANELS) * panel_h + bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    t_min = float(np.nanmin(df["time_rel_s"].to_numpy()))
    t_max = float(np.nanmax(df["time_rel_s"].to_numpy()))
    plot_x0, plot_x1 = left, width - right

    draw.text((30, 22), title, fill=(0, 0, 0), font=title_font)
    legend = "blue=raw road curvature candidate | orange=old v400 reference | red=raw vehicle dynamic response candidate"
    draw.text((30, 62), legend, fill=(80, 80, 80), font=font)

    for pidx, (col, label, color_hex) in enumerate(PANELS):
        y0 = top + pidx * panel_h
        y1 = y0 + panel_h - 28
        draw.rectangle((plot_x0, y0, plot_x1, y1), outline=(175, 175, 175), width=1)
        draw.text((22, y0 + 10), label, fill=(0, 0, 0), font=font)
        if col not in df.columns and col == "zx1|lanecurvatureXY" and "zx|lanecurvatureXY" in df.columns:
            col = "zx|lanecurvatureXY"
        if col in df.columns:
            t, vals = thin_points(df["time_rel_s"].to_numpy(dtype=np.float64), pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64))
            lo, hi = panel_limits(vals)
            if lo <= 0 <= hi:
                zy = int(y1 - (0.0 - lo) / (hi - lo) * (y1 - y0))
                draw.line((plot_x0, zy, plot_x1, zy), fill=(220, 220, 220), width=1)
            if len(t) > 1:
                xs = plot_x0 + (t - t_min) / max(t_max - t_min, 1e-9) * (plot_x1 - plot_x0)
                ys = y1 - (vals - lo) / (hi - lo) * (y1 - y0)
                points = [(int(x), int(y)) for x, y in zip(xs, ys)]
                draw.line(points, fill=color_from_hex(color_hex), width=2)
            draw.text((plot_x1 - 170, y0 + 6), f"{lo:.3g} to {hi:.3g}", fill=(90, 90, 90), font=small)
        else:
            draw.text((plot_x0 + 20, y0 + 55), f"missing {col}", fill=(150, 0, 0), font=font)

    for _, ev in events.iterrows():
        source = str(ev.get("anchor_source", ""))
        color = SOURCE_COLORS.get(source, (80, 80, 80))
        anchor = pd.to_numeric(ev.get("anchor_time_rel_s"), errors="coerce")
        if not np.isfinite(anchor):
            continue
        x = x_of_time(float(anchor), t_min, t_max, plot_x0, plot_x1)
        width_line = 3 if source == "raw_road_curvature_onset" else 1
        draw.line((x, top - 8, x, height - bottom + 5), fill=color, width=width_line)
        start = pd.to_numeric(ev.get("event_start_rel_s"), errors="coerce")
        end = pd.to_numeric(ev.get("event_end_rel_s"), errors="coerce")
        if np.isfinite(start) and np.isfinite(end) and end > start:
            xs = x_of_time(float(start), t_min, t_max, plot_x0, plot_x1)
            xe = x_of_time(float(end), t_min, t_max, plot_x0, plot_x1)
            ybar = top - 20
            draw.rectangle((xs, ybar, xe, ybar + 6), fill=color)
        if source == "raw_road_curvature_onset":
            idx = str(ev.get("event_index_in_source", ""))
            draw.text((x + 3, top - 42), idx, fill=color, font=small)

    for frac in np.linspace(0, 1, 7):
        t = t_min + frac * (t_max - t_min)
        x = x_of_time(t, t_min, t_max, plot_x0, plot_x1)
        draw.line((x, height - bottom + 8, x, height - bottom + 16), fill=(0, 0, 0), width=1)
        draw.text((x - 28, height - bottom + 20), f"{t:.0f}s", fill=(0, 0, 0), font=small)
    img.save(out_path)


def build_template_rows(events: pd.DataFrame, fig_rel: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    session_events = events.sort_values(["anchor_time_rel_s", "anchor_source"]).reset_index(drop=True)
    for idx, ev in session_events.iterrows():
        rows.append(
            {
                "annotation_id": f"ann_{ev.subject}_{ev.session_stamp}_{idx+1:04d}",
                "subject": ev.subject,
                "session_stamp": ev.session_stamp,
                "review_png": fig_rel,
                "source_candidate_event_uid": ev.event_uid,
                "candidate_anchor_source": ev.anchor_source,
                "candidate_anchor_time_rel_s": ev.anchor_time_rel_s,
                "candidate_event_start_rel_s": ev.event_start_rel_s,
                "candidate_event_end_rel_s": ev.event_end_rel_s,
                "candidate_event_type": ev.event_type,
                "candidate_event_level": ev.event_level,
                "manual_include_for_dataset": "",
                "manual_event_start_rel_s": "",
                "manual_event_end_rel_s": "",
                "manual_anchor_rel_s": "",
                "manual_event_type": "",
                "manual_direction": "",
                "manual_confidence_1_5": "",
                "manual_reason_or_notes": "",
            }
        )
    for j in range(1, 6):
        rows.append(
            {
                "annotation_id": f"custom_{session_events.iloc[0].subject}_{session_events.iloc[0].session_stamp}_{j:02d}",
                "subject": session_events.iloc[0].subject,
                "session_stamp": session_events.iloc[0].session_stamp,
                "review_png": fig_rel,
                "source_candidate_event_uid": "",
                "candidate_anchor_source": "manual_new",
                "candidate_anchor_time_rel_s": "",
                "candidate_event_start_rel_s": "",
                "candidate_event_end_rel_s": "",
                "candidate_event_type": "",
                "candidate_event_level": "",
                "manual_include_for_dataset": "",
                "manual_event_start_rel_s": "",
                "manual_event_end_rel_s": "",
                "manual_anchor_rel_s": "",
                "manual_event_type": "",
                "manual_direction": "",
                "manual_confidence_1_5": "",
                "manual_reason_or_notes": "如果图上还有候选之外的事件，可以填这一行",
            }
        )
    return rows


def write_html(manifest: pd.DataFrame) -> None:
    rows_html = []
    for _, row in manifest.iterrows():
        fig = html.escape(str(row["figure_path"]))
        rows_html.append(
            f"""
            <section>
              <h2>{html.escape(str(row['subject']))} / {html.escape(str(row['session_stamp']))}</h2>
              <p>raw road: {row['raw_road_curvature_onset']} | old v400: {row['old_v400_context_trigger_idx']} | raw dynamic: {row['raw_vehicle_dynamic_onset']} | status: {html.escape(str(row['read_status']))}</p>
              <img src="{fig}" style="max-width: 100%; border: 1px solid #ccc;" />
            </section>
            """
        )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>R2E Steering Manual Event Label Review Pack v0.1</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; }}
    section {{ margin-bottom: 36px; }}
    code {{ background: #f4f4f4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>R2E Steering 人工事件标注审查包 v0.1</h1>
  <p>本页面只用于人工确认事件范围。蓝线是道路曲率候选，橙线是旧 v400 参考，红线是车辆动态响应候选。最终以人工填写的 <code>tables/manual_event_labels_template_v0_1.csv</code> 为准。</p>
  {''.join(rows_html)}
</body>
</html>
"""
    (OUT_DIR / "review_index.html").write_text(html_text, encoding="utf-8")


def write_report(manifest: pd.DataFrame, template: pd.DataFrame, max_sessions: int) -> None:
    report = f"""# 阶段 2 补充：人工事件标注审查包 v0.1

生成时间：2026-05-12

## 为什么做

当前 `raw_road_curvature_onset` 仍只是低泄漏候选锚点，不是最终事件真值。用户提出可以人工打标签，因此本包把原始车辆行驶过程重现成多通道时间线，让人工决定哪里到哪里算事件。

## 本包内容

- 审查页面：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/review_index.html`
- 时间线图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/figures`
- 人工标签模板：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/tables/manual_event_labels_template_v0_1.csv`
- 会话清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/tables/session_review_manifest_v0_1.csv`

## 当前原型范围

- 本次最多选择 {max_sessions} 个记录作为原型。
- 实际生成记录数：{len(manifest)}
- 人工标签模板行数：{len(template)}
- 选择逻辑：优先选择 `raw_road_curvature_onset` 数量较多的记录。

## 图中信号

每张图按原始车辆时间重现以下参数：

1. 道路曲率 `lanecurvatureXY`
2. 方向盘转角 `SteeringWheel`
3. 车速 `v_km/h`
4. 横向位置 `lateraldistance`
5. 横摆角速度 `vyaw`
6. 横向加速度 `ay`
7. 横滚角 `roll`

图中颜色：

- 蓝色：`raw_road_curvature_onset`，道路曲率候选。
- 橙色：`old_v400_context_trigger_idx`，旧流程参考。
- 红色：`raw_vehicle_dynamic_onset`，车辆动态响应候选，不能作为无泄漏事件触发真值。

## 人工标注建议

在模板中填写：

- `manual_include_for_dataset`：是否纳入后续数据集，建议填 `yes/no/unsure`。
- `manual_event_start_rel_s` / `manual_event_end_rel_s`：你人工确认的事件起止时间，单位为图中相对秒。
- `manual_anchor_rel_s`：如果要定义事件触发预测锚点，填你认为模型在此刻应该开始预测未来。
- `manual_event_type` / `manual_direction`：事件类型和方向。
- `manual_confidence_1_5`：置信度，1 表示很不确定，5 表示很确定。
- `manual_reason_or_notes`：为什么这么标，或有什么疑问。

## 重要边界

本包不会修改原始 CSV，不会训练模型，也不会把事件锚点定稿。它的目的就是把候选事件可视化给人工确认。只有人工标签回填并通过一致性审查后，才能生成下一版 `manual_verified` 样本清单。
"""
    (REPORT_DIR / "manual_event_label_review_pack_v0_1_cn.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=12)
    args = parser.parse_args()
    ensure_dirs()
    candidates = pd.read_csv(CANDIDATE_PATH)
    selected = pick_review_sessions(candidates, args.max_sessions)
    manifest_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []

    for _, sess in selected.iterrows():
        subject = str(sess["subject"])
        stamp = str(sess["session_stamp"])
        events = candidates[(candidates["subject"].astype(str) == subject) & (candidates["session_stamp"].astype(str) == stamp)].copy()
        if events.empty:
            continue
        raw_path = Path(str(events.iloc[0]["vehicle_raw_absolute_path"]))
        df, status = read_vehicle(raw_path)
        fig_name = f"{subject}__{stamp}__vehicle_timeline.png"
        fig_path = FIG_DIR / fig_name
        if status == "ok":
            title = f"{subject} / {stamp} / raw vehicle timeline for manual event labeling"
            draw_timeline(df, events, fig_path, title)
        else:
            img = Image.new("RGB", (1200, 400), "white")
            draw = ImageDraw.Draw(img)
            draw.text((30, 40), f"{subject} {stamp} read failed: {status}", fill=(150, 0, 0), font=get_fonts()[1])
            img.save(fig_path)
        rel_fig = f"figures/{fig_name}"
        source_counts = events.groupby("anchor_source").size().to_dict()
        manifest_rows.append(
            {
                "subject": subject,
                "session_stamp": stamp,
                "vehicle_raw_absolute_path": str(raw_path),
                "figure_path": rel_fig,
                "read_status": status,
                "raw_road_curvature_onset": int(source_counts.get("raw_road_curvature_onset", 0)),
                "old_v400_context_trigger_idx": int(source_counts.get("old_v400_context_trigger_idx", 0)),
                "raw_vehicle_dynamic_onset": int(source_counts.get("raw_vehicle_dynamic_onset", 0)),
                "time_min_rel_s": float(df["time_rel_s"].min()) if status == "ok" and not df.empty else np.nan,
                "time_max_rel_s": float(df["time_rel_s"].max()) if status == "ok" and not df.empty else np.nan,
            }
        )
        template_rows.extend(build_template_rows(events, rel_fig))

    manifest = pd.DataFrame(manifest_rows)
    template = pd.DataFrame(template_rows)
    manifest.to_csv(OUT_TABLE_DIR / "session_review_manifest_v0_1.csv", index=False, encoding="utf-8-sig")
    template.to_csv(OUT_TABLE_DIR / "manual_event_labels_template_v0_1.csv", index=False, encoding="utf-8-sig")
    write_html(manifest)
    write_report(manifest, template, args.max_sessions)
    summary = {
        "max_sessions": args.max_sessions,
        "generated_sessions": int(len(manifest)),
        "template_rows": int(len(template)),
        "server_used": False,
        "raw_files_modified": False,
    }
    (LOG_DIR / "manual_event_label_review_pack_v0_1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
