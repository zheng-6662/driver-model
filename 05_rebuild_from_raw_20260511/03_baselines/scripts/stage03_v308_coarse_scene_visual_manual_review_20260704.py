#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v308 coarse scene visual manual-review pack.

目的：
- 把 v306 里需要人工确认的 coarse scene seed 从“看表”改成“看图”；
- 每个事件生成一张曲线图，展示锚点前 3 秒到锚点后 2 秒的驾驶/车辆响应；
- 生成一个本地 HTML 图册，人工可在浏览器里筛选、选择确认标签、填写备注并导出 CSV。

边界：
- 这些图使用了事件锚点之后的真实响应，因此只能用于人工复核标签；
- 复核图册不是模型训练输入，也不能作为“预测前可获得特征”；
- 下坡/平路过弯的划分主要来自 scene_type，当前曲线图不直接显示坡度，只能辅助检查转向/车辆响应形态。
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 20260704
ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "03_baselines"
SCRIPTS = BASELINES / "scripts"
V304_SCRIPT = SCRIPTS / "stage03_v304_fixed_event_label_conditioned_curve_model_20260703.py"
V306_REVIEW = (
    BASELINES
    / "v306_coarse_predefined_scene_label_table_20260704"
    / "tables"
    / "v306_coarse_scene_manual_review_seed_pack.csv"
)

OUT = BASELINES / "v308_coarse_scene_visual_manual_review_20260704"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"

HIST_TIME = np.round(np.linspace(-3.0, 0.0, 31), 6)

LABEL_ORDER = [
    "curve_downhill",
    "curve_flat",
    "continuous_lane_change",
    "emergency_lane_change_instability",
    "other_or_uncertain",
    "exclude_or_unclear",
]

LABEL_CN = {
    "curve_downhill": "下坡过弯",
    "curve_flat": "平路过弯",
    "continuous_lane_change": "连续变道/连续左右修正",
    "emergency_lane_change_instability": "紧急变道/猛打方向失稳",
    "other_or_uncertain": "其他/不确定",
    "exclude_or_unclear": "排除/看不清",
}

DECISION_ORDER = ["", "confirmed", "change_label", "uncertain", "exclude"]
DECISION_CN = {
    "": "未复核",
    "confirmed": "确认候选标签",
    "change_label": "改标签",
    "uncertain": "不确定",
    "exclude": "排除",
}


def import_module_from_path(module_name: str, path: Path):
    """按路径导入已有脚本，复用已经跑通的数据加载逻辑。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少依赖脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V304 = import_module_from_path("stage03_v304_for_v308_visual_review", V304_SCRIPT)
FUTURE_TIME = V304.FUTURE_GRID.astype(np.float32)


def ensure_dirs() -> None:
    """创建 v308 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 使用 utf-8-sig，方便 Windows/Excel 直接打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, object], path: Path) -> None:
    """保存 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def as_bool(value: object) -> bool:
    """把 CSV 中常见的 bool/int/string 标志位统一读成 bool。"""

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


def safe_name(value: object, max_len: int = 120) -> str:
    """生成适合文件名的短字符串。"""

    text = str(value)
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    compact = "".join(out).strip("_")
    return compact[:max_len] if compact else "event"


def infer_hist_feature_names(flat_feature_names: Iterable[str]) -> List[str]:
    """从 v236 的扁平 feature_names 里恢复 x_hist 第三维顺序。"""

    names: List[str] = []
    prefix = "hist_-3.0s_"
    for name in flat_feature_names:
        text = str(name)
        if text.startswith(prefix):
            names.append(text.replace(prefix, "", 1))
    if not names:
        raise AssertionError("无法从 feature_names 推断历史信号顺序")
    return names


def feature_index(names: List[str]) -> Dict[str, int]:
    """构造 feature -> index 映射。"""

    return {name: i for i, name in enumerate(names)}


def target_index(names: Iterable[str]) -> Dict[str, int]:
    """构造 target -> index 映射。"""

    return {str(name): i for i, name in enumerate(names)}


def line(ax, t: np.ndarray, y: np.ndarray, *, color: str, label: str, lw: float = 1.4, alpha: float = 1.0) -> None:
    """只画有限值，避免缺失值把曲线拉断到异常位置。"""

    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(yy)
    if mask.any():
        ax.plot(t[mask], yy[mask], color=color, lw=lw, alpha=alpha, label=label)


def signed_peak(values: np.ndarray) -> float:
    """返回绝对值最大的有符号峰值。"""

    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan")
    valid = arr[finite]
    return float(valid[np.argmax(np.abs(valid))])


def sign_switch_count(values: np.ndarray) -> int:
    """粗略统计非零符号切换次数，用于人工复核排序参考。"""

    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return 0
    valid = arr[finite]
    eps = max(float(np.nanstd(valid)) * 0.05, 1e-4)
    sign = np.sign(np.where(np.abs(valid) < eps, 0.0, valid))
    sign = sign[sign != 0]
    if len(sign) < 2:
        return 0
    return int(np.sum(sign[1:] != sign[:-1]))


def plot_event(row: pd.Series, data, hist_features: List[str], target_names: Dict[str, int], out_path: Path) -> Dict[str, float]:
    """为单个事件生成 7 行曲线图，并返回少量图上统计量。"""

    hidx = feature_index(hist_features)
    array_row = int(row["array_row"])
    x_hist = data.x_hist[array_row].astype(float)
    y_future = data.y_future[array_row].astype(float)
    x_road = data.x_road[array_row].astype(float)

    steer_hist = x_hist[:, hidx["steering"]]
    steer_anchor = steer_hist[-1] if np.isfinite(steer_hist[-1]) else 0.0
    steer_future = steer_anchor + y_future[:, target_names["steering_delta"]]

    hist_dt = float(np.nanmedian(np.diff(HIST_TIME)))
    steer_rate_hist = np.gradient(steer_hist, hist_dt) if np.isfinite(steer_hist).sum() >= 3 else np.full_like(steer_hist, np.nan)
    steer_rate_future = y_future[:, target_names["steering_rate"]]

    ay_hist = x_hist[:, hidx["ay"]]
    ay_future = y_future[:, target_names["ay"]]
    yaw_rate_hist = x_hist[:, hidx["yaw_rate"]]
    yaw_rate_future = y_future[:, target_names["yaw_rate"]]

    roll_hist = x_hist[:, hidx["roll"]]
    roll_anchor = roll_hist[-1] if np.isfinite(roll_hist[-1]) else 0.0
    roll_future = roll_anchor + y_future[:, target_names["roll_delta"]]

    lateral_hist = x_hist[:, hidx["lateral_distance"]]
    lateral_future = x_road[:, 1] if x_road.shape[1] > 1 else np.full_like(FUTURE_TIME, np.nan)
    curv_hist = x_hist[:, hidx["lane_curvature"]]
    curv_future = x_road[:, 0] if x_road.shape[1] > 0 else np.full_like(FUTURE_TIME, np.nan)

    speed_hist = x_hist[:, hidx["speed_kmh"]]
    brake_hist = x_hist[:, hidx["brake"]]

    fig, axes = plt.subplots(7, 1, figsize=(12.8, 10.6), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.axvspan(-3.0, 0.0, color="#ECFDF5", alpha=0.65)
        ax.axvspan(0.0, float(FUTURE_TIME[-1]), color="#EFF6FF", alpha=0.75)
        ax.axvline(0.0, color="#DC2626", lw=1.0, ls="--")
        ax.grid(True, color="#E5E7EB", lw=0.6, alpha=0.9)

    line(axes_arr[0], HIST_TIME, steer_hist, color="#6B7280", label="历史 steering", lw=1.1)
    line(axes_arr[0], FUTURE_TIME, steer_future, color="#111827", label="未来真实 steering", lw=1.7)
    axes_arr[0].set_ylabel("方向盘角")
    axes_arr[0].legend(fontsize=8, loc="best")

    line(axes_arr[1], HIST_TIME, steer_rate_hist, color="#6B7280", label="历史估计", lw=1.1)
    line(axes_arr[1], FUTURE_TIME, steer_rate_future, color="#111827", label="未来真实", lw=1.7)
    axes_arr[1].set_ylabel("方向盘速度")

    line(axes_arr[2], HIST_TIME, ay_hist, color="#6B7280", label="历史", lw=1.1)
    line(axes_arr[2], FUTURE_TIME, ay_future, color="#111827", label="未来真实", lw=1.7)
    axes_arr[2].set_ylabel("ay")

    line(axes_arr[3], HIST_TIME, yaw_rate_hist, color="#6B7280", label="历史", lw=1.1)
    line(axes_arr[3], FUTURE_TIME, yaw_rate_future, color="#111827", label="未来真实", lw=1.7)
    axes_arr[3].set_ylabel("yaw rate")

    line(axes_arr[4], HIST_TIME, roll_hist, color="#6B7280", label="历史 roll", lw=1.1)
    line(axes_arr[4], FUTURE_TIME, roll_future, color="#111827", label="未来 roll", lw=1.7)
    axes_arr[4].set_ylabel("roll")

    line(axes_arr[5], HIST_TIME, curv_hist, color="#7C3AED", label="历史 lane curvature", lw=1.0)
    line(axes_arr[5], FUTURE_TIME, curv_future, color="#A855F7", label="未来 road curvature", lw=1.3)
    axes_arr[5].set_ylabel("曲率")
    ax_lat = axes_arr[5].twinx()
    line(ax_lat, HIST_TIME, lateral_hist, color="#059669", label="历史 lateral distance", lw=0.9, alpha=0.85)
    line(ax_lat, FUTURE_TIME, lateral_future, color="#10B981", label="未来 road lateral", lw=1.0, alpha=0.9)
    ax_lat.set_ylabel("横向距离")

    line(axes_arr[6], HIST_TIME, speed_hist, color="#0369A1", label="speed km/h", lw=1.2)
    axes_arr[6].set_ylabel("车速 km/h")
    ax_brake = axes_arr[6].twinx()
    line(ax_brake, HIST_TIME, brake_hist, color="#EA580C", label="brake", lw=1.0, alpha=0.9)
    ax_brake.set_ylabel("brake")

    candidate = str(row.get("coarse_scene_label", ""))
    candidate_cn = str(row.get("coarse_scene_label_cn", LABEL_CN.get(candidate, candidate)))
    priority = str(row.get("coarse_scene_review_priority", ""))
    flags = []
    for col, text in [
        ("strong_steer", "强转向"),
        ("vehicle_strong", "车辆强响应"),
        ("reverse", "反向"),
        ("multi_correction", "多段修正"),
        ("flag_fast_steer", "快速转向"),
        ("flag_high_yaw_or_ay", "高 yaw/ay"),
    ]:
        if as_bool(row.get(col, False)):
            flags.append(text)
    flag_text = " / ".join(flags) if flags else "无明显标志位"
    title = (
        f"#{int(row['review_rank']):04d}  {candidate_cn}  |  {priority}  |  split={row.get('split', '')}  "
        f"|  subj={row.get('subject', '')}  obs={row.get('observation_s', '')}s\n"
        f"{row.get('event_uid', '')}  |  scene={row.get('scene_type', '')}  route={row.get('route_event', '')}  "
        f"|  v300_rmse={float(row.get('v300_rmse', np.nan)):.3f}  |  {flag_text}"
    )
    fig.suptitle(title, fontsize=10.5, x=0.01, y=0.995, ha="left")
    axes_arr[-1].set_xlabel("相对锚点时间 / s（绿色=锚点前历史，蓝色=锚点后真实响应）")
    axes_arr[-1].set_xlim(-3.0, float(FUTURE_TIME[-1]))
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=135)
    plt.close(fig)

    steer_all = np.concatenate([steer_hist, steer_future])
    steer_rate_all = np.concatenate([steer_rate_hist, steer_rate_future])
    return {
        "plot_steer_peak": signed_peak(steer_all),
        "plot_steer_rate_peak": signed_peak(steer_rate_all),
        "plot_steer_sign_switch_count": sign_switch_count(steer_all),
        "plot_abs_ay_peak": float(np.nanmax(np.abs(np.concatenate([ay_hist, ay_future])))),
        "plot_abs_yaw_rate_peak": float(np.nanmax(np.abs(np.concatenate([yaw_rate_hist, yaw_rate_future])))),
        "plot_abs_roll_peak": float(np.nanmax(np.abs(np.concatenate([roll_hist, roll_future])))),
    }


def build_review_queue(data) -> pd.DataFrame:
    """读取 v306 复核 seed，并映射到 delay0 数组行。"""

    if not V306_REVIEW.exists():
        raise FileNotFoundError(f"缺少 v306 复核表：{V306_REVIEW}")
    review = pd.read_csv(V306_REVIEW, encoding="utf-8-sig")
    priority = review["coarse_scene_review_priority"].astype(str).isin(["high", "medium"])
    review = review[priority].copy()
    delay0 = data.manifest[data.manifest["delay_ms"].astype(int).eq(0)].copy()
    delay0["array_row"] = delay0.index.astype(int)
    map_cols = ["event_uid", "array_row", "delay_ms", "raw_vehicle_csv", "history_finite_ratio", "target_finite_ratio"]
    queue = review.merge(delay0[map_cols], on="event_uid", how="left", validate="one_to_one")
    if queue["array_row"].isna().any():
        missing = queue.loc[queue["array_row"].isna(), "event_uid"].head(10).tolist()
        raise AssertionError(f"v306 复核事件无法映射到 delay0 数组：{missing}")
    queue["priority_sort"] = queue["coarse_scene_review_priority"].map({"high": 0, "medium": 1}).fillna(9)
    queue["v300_rmse_num"] = pd.to_numeric(queue["v300_rmse"], errors="coerce")
    queue = queue.sort_values(
        ["priority_sort", "v300_rmse_num", "coarse_scene_label", "event_uid"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    queue.insert(0, "review_rank", np.arange(1, len(queue) + 1, dtype=int))
    return queue


def html_options(values: Iterable[str], labels: Dict[str, str], selected: str = "") -> str:
    """生成 select options。"""

    parts = []
    for value in values:
        text = labels.get(value, value)
        attr = " selected" if value == selected else ""
        parts.append(f'<option value="{html.escape(value)}"{attr}>{html.escape(text)}</option>')
    return "\n".join(parts)


def write_html(queue: pd.DataFrame, index_path: Path) -> None:
    """生成可筛选、可导出人工决策 CSV 的静态 HTML 图册。"""

    label_options = html_options(LABEL_ORDER, LABEL_CN)
    decision_options = html_options(DECISION_ORDER, DECISION_CN)
    cards: List[str] = []
    for row in queue.to_dict(orient="records"):
        event_uid = str(row["event_uid"])
        label = str(row.get("coarse_scene_label", ""))
        priority = str(row.get("coarse_scene_review_priority", ""))
        image = html.escape(str(row["image_relpath"]).replace("\\", "/"))
        current_cn = str(row.get("coarse_scene_label_cn", LABEL_CN.get(label, label)))
        card = f"""
        <article class="card" data-event="{html.escape(event_uid)}" data-label="{html.escape(label)}" data-priority="{html.escape(priority)}">
          <div class="card-head">
            <div>
              <div class="rank">#{int(row['review_rank']):04d} · {html.escape(current_cn)}</div>
              <div class="meta">{html.escape(event_uid)}</div>
            </div>
            <div class="badge {html.escape(priority)}">{html.escape(priority)}</div>
          </div>
          <a href="{image}" target="_blank" rel="noreferrer">
            <img loading="lazy" src="{image}" alt="{html.escape(event_uid)}">
          </a>
          <div class="info">
            <span>split={html.escape(str(row.get('split', '')))}</span>
            <span>scene={html.escape(str(row.get('scene_type', '')))}</span>
            <span>route={html.escape(str(row.get('route_event', '')))}</span>
            <span>rmse={float(row.get('v300_rmse_num', np.nan)):.3f}</span>
          </div>
          <div class="review-grid">
            <label>复核结论
              <select data-field="review_decision">{decision_options}</select>
            </label>
            <label>人工标签
              <select data-field="manual_confirmed_label">
                <option value="">未选择</option>
                {label_options}
              </select>
            </label>
            <label class="note">备注
              <input data-field="review_note" placeholder="例如：单峰猛打 / 连续左右 / 看不清 / 应排除">
            </label>
          </div>
        </article>
        """
        cards.append(card)

    counts = (
        queue.groupby(["coarse_scene_review_priority", "coarse_scene_label"], dropna=False)
        .size()
        .reset_index(name="n")
        .to_dict(orient="records")
    )
    count_rows = "\n".join(
        f"<tr><td>{html.escape(str(r['coarse_scene_review_priority']))}</td><td>{html.escape(str(r['coarse_scene_label']))}</td><td>{int(r['n'])}</td></tr>"
        for r in counts
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v308 coarse scene 视觉人工复核</title>
  <style>
    :root {{
      --ink: #111827;
      --muted: #6b7280;
      --line: #d1d5db;
      --panel: #ffffff;
      --soft: #f9fafb;
      --blue: #2563eb;
      --orange: #ea580c;
    }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      background: #f3f4f6;
      color: var(--ink);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(255,255,255,0.96);
      border-bottom: 1px solid var(--line);
      padding: 14px 20px;
      box-shadow: 0 2px 8px rgba(15,23,42,0.06);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 20px;
      line-height: 1.25;
    }}
    .sub {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 10px;
    }}
    button, select, input {{
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
    }}
    button {{
      padding: 7px 11px;
      cursor: pointer;
    }}
    button.primary {{
      background: var(--blue);
      color: #fff;
      border-color: var(--blue);
    }}
    #search {{
      min-width: 300px;
      padding: 7px 9px;
    }}
    main {{
      padding: 18px 20px 40px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: minmax(280px, 1fr) minmax(320px, 1.4fr);
      gap: 14px;
      margin-bottom: 18px;
    }}
    .summary section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
    }}
    .summary h2 {{
      margin: 0 0 8px;
      font-size: 15px;
    }}
    .summary p, .summary li {{
      color: #374151;
      font-size: 13px;
      line-height: 1.55;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    td, th {{
      border-bottom: 1px solid #e5e7eb;
      padding: 5px 6px;
      text-align: left;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 4px rgba(15,23,42,0.05);
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px 7px;
      border-bottom: 1px solid #e5e7eb;
    }}
    .rank {{
      font-weight: 700;
      font-size: 14px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
      margin-top: 2px;
    }}
    .badge {{
      align-self: flex-start;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: #f9fafb;
    }}
    .badge.high {{
      background: #fff7ed;
      border-color: #fdba74;
      color: #9a3412;
    }}
    .badge.medium {{
      background: #eff6ff;
      border-color: #93c5fd;
      color: #1d4ed8;
    }}
    img {{
      display: block;
      width: 100%;
      background: #fff;
      border-bottom: 1px solid #e5e7eb;
    }}
    .info {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px 12px;
      padding: 8px 12px;
      color: #374151;
      font-size: 12px;
      border-bottom: 1px solid #e5e7eb;
    }}
    .review-grid {{
      display: grid;
      grid-template-columns: 150px 190px minmax(220px, 1fr);
      gap: 10px;
      padding: 10px 12px 12px;
    }}
    label {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      color: #374151;
      font-size: 12px;
    }}
    label select, label input {{
      padding: 7px 8px;
      min-width: 0;
    }}
    .hidden {{
      display: none !important;
    }}
    @media (max-width: 720px) {{
      .grid, .summary {{
        grid-template-columns: 1fr;
      }}
      .review-grid {{
        grid-template-columns: 1fr;
      }}
      #search {{
        min-width: 0;
        width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>v308 coarse scene 视觉人工复核</h1>
    <div class="sub">
      当前图册包含 high + medium 复核队列共 <strong>{len(queue)}</strong> 个事件。图中绿色为锚点前历史，蓝色为锚点后真实响应；这些未来响应只用于人工复核标签，不作为模型可预测输入。
    </div>
    <div class="toolbar">
      <input id="search" placeholder="搜索 event_uid / subject / label / route">
      <select id="priorityFilter">
        <option value="">全部优先级</option>
        <option value="high">high</option>
        <option value="medium">medium</option>
      </select>
      <select id="labelFilter">
        <option value="">全部候选标签</option>
        {label_options}
      </select>
      <button id="clearFilters">清除筛选</button>
      <button class="primary" id="downloadCsv">导出复核 CSV</button>
      <span id="visibleCount" class="sub"></span>
    </div>
  </header>
  <main>
    <div class="summary">
      <section>
        <h2>判读提示</h2>
        <ul>
          <li><strong>连续变道/连续左右修正：</strong>方向盘角或方向盘速度呈多次正负切换，ay/yaw rate 往往跟着交替。</li>
          <li><strong>紧急变道/猛打方向失稳：</strong>0 秒后出现单次或短时大幅猛打方向，随后 ay/yaw rate/roll 明显放大。</li>
          <li><strong>过弯：</strong>通常是持续曲率/持续转向。下坡和平路主要由 scene_type 给出，图上不一定直接显示坡度。</li>
          <li><strong>其他/不确定：</strong>如果图上看不到清楚模式，先标不确定，不要硬分。</li>
        </ul>
      </section>
      <section>
        <h2>候选标签计数</h2>
        <table>
          <thead><tr><th>priority</th><th>candidate label</th><th>n</th></tr></thead>
          <tbody>{count_rows}</tbody>
        </table>
      </section>
    </div>
    <section class="grid" id="cards">
      {''.join(cards)}
    </section>
  </main>
  <script>
    const STORAGE_KEY = "v308_coarse_scene_visual_review";
    const cards = Array.from(document.querySelectorAll(".card"));
    const search = document.getElementById("search");
    const priorityFilter = document.getElementById("priorityFilter");
    const labelFilter = document.getElementById("labelFilter");
    const visibleCount = document.getElementById("visibleCount");

    function getStore() {{
      try {{
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");
      }} catch {{
        return {{}};
      }}
    }}

    function setStore(store) {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
    }}

    function saveCard(card) {{
      const store = getStore();
      const event = card.dataset.event;
      store[event] = {{}};
      card.querySelectorAll("[data-field]").forEach(el => {{
        store[event][el.dataset.field] = el.value || "";
      }});
      setStore(store);
    }}

    function loadCards() {{
      const store = getStore();
      cards.forEach(card => {{
        const data = store[card.dataset.event] || {{}};
        card.querySelectorAll("[data-field]").forEach(el => {{
          if (data[el.dataset.field] !== undefined) el.value = data[el.dataset.field];
          el.addEventListener("change", () => saveCard(card));
          el.addEventListener("input", () => saveCard(card));
        }});
      }});
    }}

    function applyFilters() {{
      const q = (search.value || "").trim().toLowerCase();
      const pf = priorityFilter.value;
      const lf = labelFilter.value;
      let shown = 0;
      cards.forEach(card => {{
        const text = card.innerText.toLowerCase();
        const okQ = !q || text.includes(q);
        const okP = !pf || card.dataset.priority === pf;
        const okL = !lf || card.dataset.label === lf;
        const show = okQ && okP && okL;
        card.classList.toggle("hidden", !show);
        if (show) shown += 1;
      }});
      visibleCount.textContent = `当前显示 ${{shown}} / ${{cards.length}}`;
    }}

    function csvEscape(value) {{
      const s = String(value ?? "");
      return `"${{s.replaceAll('"', '""')}}"`;
    }}

    function downloadCsv() {{
      const header = [
        "event_uid",
        "suggested_label",
        "review_priority",
        "review_decision",
        "manual_confirmed_label",
        "review_note",
      ];
      const rows = [header];
      cards.forEach(card => {{
        const fields = {{}};
        card.querySelectorAll("[data-field]").forEach(el => {{
          fields[el.dataset.field] = el.value || "";
        }});
        rows.push([
          card.dataset.event,
          card.dataset.label,
          card.dataset.priority,
          fields.review_decision || "",
          fields.manual_confirmed_label || "",
          fields.review_note || "",
        ]);
      }});
      const csv = rows.map(row => row.map(csvEscape).join(",")).join("\\r\\n");
      const blob = new Blob(["\\ufeff" + csv], {{type: "text/csv;charset=utf-8"}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "v308_manual_review_decisions.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}

    document.getElementById("downloadCsv").addEventListener("click", downloadCsv);
    document.getElementById("clearFilters").addEventListener("click", () => {{
      search.value = "";
      priorityFilter.value = "";
      labelFilter.value = "";
      applyFilters();
    }});
    [search, priorityFilter, labelFilter].forEach(el => {{
      el.addEventListener("input", applyFilters);
      el.addEventListener("change", applyFilters);
    }});
    loadCards();
    applyFilters();
  </script>
</body>
</html>
"""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(html_text, encoding="utf-8")


def write_report(queue: pd.DataFrame, elapsed_s: float, zip_path: Path | None) -> Path:
    """写中文说明报告。"""

    label_counts = (
        queue.groupby(["coarse_scene_review_priority", "coarse_scene_label"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["coarse_scene_review_priority", "coarse_scene_label"])
    )
    lines = [
        "# v308 coarse scene 视觉人工复核包",
        "",
        "## 目的",
        "",
        "用户反馈看表不容易区分，因此本版本把 v306 的 high + medium 复核队列改成逐事件曲线图册。",
        "",
        "## 如何使用",
        "",
        f"- 打开 `index.html`：`{OUT / 'index.html'}`",
        "- 点击任意图可以打开大图。",
        "- 在每张图下方选择“复核结论”和“人工标签”，可写备注。",
        "- 页面会把选择暂存在浏览器 localStorage；完成后点击“导出复核 CSV”。",
        "",
        "## 图中信号",
        "",
        "- 方向盘角：锚点前历史 steering + 锚点后真实 steering_delta 还原后的方向盘角。",
        "- 方向盘速度：锚点前由方向盘角估计，锚点后使用真实 steering_rate。",
        "- ay / yaw rate / roll：辅助判断车辆是否在猛打方向后开始失稳。",
        "- 曲率/横向距离/车速/制动：辅助判断过弯、横向偏移、制动参与。",
        "",
        "## 数量",
        "",
        f"- 复核图数量：{len(queue)}",
        f"- high：{int((queue['coarse_scene_review_priority'].astype(str) == 'high').sum())}",
        f"- medium：{int((queue['coarse_scene_review_priority'].astype(str) == 'medium').sum())}",
        "",
        label_counts.to_markdown(index=False),
        "",
        "## 重要边界",
        "",
        "- 图册使用了锚点后真实响应，只用于人工复核标签，不是可部署模型输入。",
        "- 下坡过弯/平路过弯主要由 `scene_type` 给出；当前曲线图不直接显示道路坡度。",
        "- 如果只凭图看不清，应标成 `uncertain` 或 `exclude_or_unclear`，不要为了凑类别强行确认。",
        "",
        "## 输出",
        "",
        f"- HTML 图册：`{OUT / 'index.html'}`",
        f"- 复核队列清单：`{TABLES / 'v308_visual_review_manifest.csv'}`",
        f"- 人工填写模板：`{TABLES / 'v308_manual_review_decision_template.csv'}`",
    ]
    if zip_path is not None:
        lines.append(f"- ZIP 包：`{zip_path}`")
    lines.extend(["", f"生成耗时：{elapsed_s:.1f}s"])
    path = REPORTS / "v308_coarse_scene_visual_manual_review_cn.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_zip() -> Path:
    """打包 HTML、表格、报告和图像，方便迁移或备份。"""

    zip_path = OUT / "v308_coarse_scene_visual_manual_review_20260704.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in OUT.rglob("*"):
            if path == zip_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(OUT))
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v308 visual manual-review gallery.")
    parser.add_argument("--limit", type=int, default=0, help="调试用：只生成前 N 个事件；0 表示全量")
    parser.add_argument("--no-zip", action="store_true", help="调试或快速生成时跳过 zip 打包")
    args = parser.parse_args()

    started = time.time()
    ensure_dirs()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    print("[v308] 读取 v304/v236 数据和 v306 复核队列", flush=True)
    prepared = V304.prepare_v304_data(hard_event_extra=0.0)
    data = prepared.data
    hist_features = infer_hist_feature_names(data.feature_names)
    targets = target_index(data.target_names)
    queue = build_review_queue(data)
    if args.limit and args.limit > 0:
        queue = queue.head(args.limit).copy()
        queue["review_rank"] = np.arange(1, len(queue) + 1, dtype=int)

    rows: List[Dict[str, object]] = []
    image_dir = FIGURES / "priority_review"
    for pos, row in queue.iterrows():
        event_uid = str(row["event_uid"])
        label = str(row.get("coarse_scene_label", ""))
        file_name = f"rank{int(row['review_rank']):04d}_{safe_name(label, 45)}_{safe_name(event_uid, 95)}.png"
        out_path = image_dir / file_name
        stats = plot_event(row, data, hist_features, targets, out_path)
        record = row.to_dict()
        record.update(stats)
        record["image_path"] = str(out_path)
        record["image_relpath"] = out_path.relative_to(OUT).as_posix()
        rows.append(record)
        if (pos + 1) % 50 == 0 or (pos + 1) == len(queue):
            print(f"[v308] plotted {pos + 1}/{len(queue)}", flush=True)

    out_queue = pd.DataFrame(rows)
    write_csv(out_queue, TABLES / "v308_visual_review_manifest.csv")
    template_cols = [
        "review_rank",
        "event_uid",
        "coarse_scene_review_priority",
        "coarse_scene_label",
        "coarse_scene_label_cn",
        "review_decision",
        "manual_confirmed_label",
        "review_note",
        "image_relpath",
    ]
    template = out_queue.copy()
    template["review_decision"] = ""
    template["manual_confirmed_label"] = ""
    template["review_note"] = ""
    write_csv(template[template_cols], TABLES / "v308_manual_review_decision_template.csv")

    write_html(out_queue, OUT / "index.html")
    zip_path = None if args.no_zip else make_zip()
    elapsed = time.time() - started
    report_path = write_report(out_queue, elapsed, zip_path)
    guardrail = {
        "version": "v308_coarse_scene_visual_manual_review_20260704",
        "source_v306_review_pack": str(V306_REVIEW),
        "source_v304_script_reused": str(V304_SCRIPT),
        "review_event_n": int(len(out_queue)),
        "priority_counts": out_queue["coarse_scene_review_priority"].astype(str).value_counts().to_dict(),
        "label_counts": out_queue["coarse_scene_label"].astype(str).value_counts().to_dict(),
        "uses_future_response_for_manual_review": True,
        "future_response_used_as_model_input": False,
        "image_n": int(len(out_queue)),
        "html_index": str(OUT / "index.html"),
        "manifest_csv": str(TABLES / "v308_visual_review_manifest.csv"),
        "decision_template_csv": str(TABLES / "v308_manual_review_decision_template.csv"),
        "zip_path": str(zip_path) if zip_path is not None else None,
        "elapsed_s": elapsed,
    }
    write_json(guardrail, LOGS / "guardrail_check.json")
    print(f"[v308] report={report_path}", flush=True)
    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
