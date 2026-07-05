#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v309 recent best prediction-effect gallery.

目的：
- 用户想先看近期最好一版模型的预测效果，而不是继续看人工复核标签；
- 以 v307 selected 模型为近期最好版本，生成 test delay0 的指标摘要和代表性样本图册；
- 图上同时显示 v307 预测、v300 参照预测、真实 0-2s steering_delta，并尽量补充 2s 之后的真实车辆走势。

边界：
- v307 的模型输出只有 0-2s 的 21 点 steering_delta；
- 2s 之后只展示真实车辆后续走势，用于理解事件发展，不代表模型预测范围；
- v307 使用的粗场景标签仍有自动 seed 成分，因此图册用于观察效果，不等于最终可部署结论。
"""

from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "03_baselines"
SCRIPTS = BASELINES / "scripts"
V304_SCRIPT = SCRIPTS / "stage03_v304_fixed_event_label_conditioned_curve_model_20260703.py"
V307_DIR = BASELINES / "v307_coarse_scene_label_conditioned_curve_model_20260704"
V307_PRED = V307_DIR / "v307_coarse_scene_label_conditioned_predictions.npz"
V307_GROUP_SUMMARY = V307_DIR / "tables" / "v307_delay0_group_summary.csv"

OUT = BASELINES / "v309_recent_best_prediction_effect_gallery_20260704"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"

HIST_TIME = np.round(np.linspace(-3.0, 0.0, 31), 6)
EXTEND_TO_S = 6.0


LABEL_CN = {
    "curve_downhill": "下坡过弯",
    "curve_flat": "平路过弯",
    "continuous_lane_change": "连续变道/连续左右修正",
    "emergency_lane_change_instability": "紧急变道/猛打方向失稳",
    "other_or_uncertain": "其他/不确定",
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


V304 = import_module_from_path("stage03_v304_for_v309_effect_gallery", V304_SCRIPT)


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 使用 utf-8-sig，方便 Windows/Excel 查看中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, object], path: Path) -> None:
    """保存 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    """从 v236 的扁平 feature_names 恢复 x_hist 第三维顺序。"""

    prefix = "hist_-3.0s_"
    names = [str(name).replace(prefix, "", 1) for name in flat_feature_names if str(name).startswith(prefix)]
    if not names:
        raise AssertionError("无法从 feature_names 推断历史信号顺序")
    return names


def finite_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """按 mask 计算单样本 RMSE。"""

    valid = mask & np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(y_true[valid] - y_pred[valid]))))


def signed_peak(values: np.ndarray) -> float:
    """返回绝对值最大的有符号峰值。"""

    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan")
    valid = arr[finite]
    return float(valid[np.argmax(np.abs(valid))])


def load_predictions() -> Dict[str, np.ndarray | str]:
    """读取 v307 预测包。"""

    if not V307_PRED.exists():
        raise FileNotFoundError(f"缺少 v307 预测文件：{V307_PRED}")
    with np.load(V307_PRED, allow_pickle=True) as z:
        return {
            "y_true": z["y_true_steering_delta"].astype(np.float32),
            "pred_v300": z["pred_v300_reference"].astype(np.float32),
            "pred_v307": z["pred_v307_selected"].astype(np.float32),
            "mask": z["original_remaining_valid"].astype(bool),
            "future_grid": z["future_grid_s"].astype(np.float32),
            "event_uid": z["event_uid"].astype(str),
            "split": z["split"].astype(str),
            "delay_ms": z["delay_ms"].astype(int),
            "coarse_scene_label": z["coarse_scene_label"].astype(str),
            "best_v307_model": str(z["best_v307_model"][0]),
            "v300_reference_model": str(z["v300_reference_model"][0]),
        }


def build_delay0_test_table(pred: Dict[str, np.ndarray | str], data) -> pd.DataFrame:
    """构造 test delay0 逐事件指标表。"""

    manifest = data.manifest.copy().reset_index(drop=True)
    rows: List[Dict[str, object]] = []
    split = pred["split"]
    delay_ms = pred["delay_ms"]
    y_true = pred["y_true"]
    pred_v300 = pred["pred_v300"]
    pred_v307 = pred["pred_v307"]
    mask = pred["mask"]
    labels = pred["coarse_scene_label"]
    for i in np.where((split == "test") & (delay_ms == 0))[0]:
        true = y_true[i]
        p300 = pred_v300[i]
        p307 = pred_v307[i]
        valid = mask[i]
        rmse300 = finite_rmse(true, p300, valid)
        rmse307 = finite_rmse(true, p307, valid)
        mrow = manifest.iloc[i].to_dict()
        rows.append(
            {
                "array_row": int(i),
                "event_uid": str(pred["event_uid"][i]),
                "subject": mrow.get("subject", ""),
                "recording": mrow.get("recording", ""),
                "split": str(split[i]),
                "delay_ms": int(delay_ms[i]),
                "scene_type": mrow.get("scene_type", ""),
                "route_event": mrow.get("route_event", ""),
                "observation_s": float(mrow.get("observation_s", float("nan"))),
                "raw_vehicle_csv": str(mrow.get("raw_vehicle_csv", "")),
                "coarse_scene_label": str(labels[i]),
                "coarse_scene_label_cn": LABEL_CN.get(str(labels[i]), str(labels[i])),
                "v300_rmse": rmse300,
                "v307_rmse": rmse307,
                "delta_v307_minus_v300": rmse307 - rmse300,
                "true_peak": signed_peak(true[valid]),
                "v307_peak": signed_peak(p307[valid]),
                "v300_peak": signed_peak(p300[valid]),
                "within_bad_top10_by_v249": int(mrow.get("within_bad_top10_by_v249", 0)),
                "within_bad_top20_by_v249": int(mrow.get("within_bad_top20_by_v249", 0)),
                "strong_steer": bool(mrow.get("strong_steer", False)),
                "vehicle_strong": bool(mrow.get("vehicle_strong", False)),
                "v299_vehicle_ambiguous": int(mrow.get("v299_vehicle_ambiguous", 0)),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_delta"] = out["delta_v307_minus_v300"].abs()
    return out


def choose_gallery_samples(table: pd.DataFrame) -> pd.DataFrame:
    """选择代表性样本，避免只看最差或只看最好。"""

    picks: List[pd.DataFrame] = []

    def add(category: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        tmp = df.copy()
        tmp["gallery_category"] = category
        picks.append(tmp)

    add("best_v307_lowest_rmse", table.sort_values("v307_rmse", ascending=True).head(8))
    median_center = table.assign(median_gap=(table["v307_rmse"] - table["v307_rmse"].median()).abs()).sort_values("median_gap")
    add("median_v307_typical", median_center.head(8))
    add("worst_v307_highest_rmse", table.sort_values("v307_rmse", ascending=False).head(12))
    add("largest_improvement_vs_v300", table.sort_values("delta_v307_minus_v300", ascending=True).head(12))
    add("largest_regression_vs_v300", table.sort_values("delta_v307_minus_v300", ascending=False).head(8))
    add("bad_top10_by_v249", table[table["within_bad_top10_by_v249"].eq(1)].sort_values("v307_rmse", ascending=False).head(12))
    for label, group in table.groupby("coarse_scene_label"):
        add(f"scene_{label}", group.sort_values("v307_rmse", ascending=False).head(5))

    merged = pd.concat(picks, ignore_index=True)
    merged = merged.drop_duplicates("event_uid", keep="first").reset_index(drop=True)
    merged.insert(0, "gallery_rank", np.arange(1, len(merged) + 1, dtype=int))
    return merged


def load_raw_window(raw_path: str, observation_s: float, end_s: float = EXTEND_TO_S) -> pd.DataFrame:
    """读取原始车辆 CSV 的局部窗口，用于显示 2 秒之后真实走势。"""

    path = Path(raw_path)
    if not path.exists():
        return pd.DataFrame()
    needed = ["StorageTime", "zx|SteeringWheel", "zx|ay", "zx|vyaw", "zx|roll", "zx|vx", "zx1|lanecurvatureXY"]
    try:
        df = pd.read_csv(path, usecols=lambda c: c in needed)
    except Exception:
        return pd.DataFrame()
    if "StorageTime" not in df.columns or df.empty:
        return pd.DataFrame()
    t = pd.to_datetime(df["StorageTime"], errors="coerce")
    if t.isna().all():
        return pd.DataFrame()
    rel_record = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    rel_anchor = rel_record - float(observation_s)
    keep = (rel_anchor >= -3.2) & (rel_anchor <= end_s)
    out = df.loc[keep].copy()
    out["rel_anchor_s"] = rel_anchor[keep]
    return out


def plot_line(ax, t: np.ndarray, y: np.ndarray, *, color: str, label: str, lw: float = 1.5, alpha: float = 1.0, ls: str = "-") -> None:
    """只画有限值。"""

    yy = np.asarray(y, dtype=float)
    tt = np.asarray(t, dtype=float)
    valid = np.isfinite(tt) & np.isfinite(yy)
    if valid.any():
        ax.plot(tt[valid], yy[valid], color=color, lw=lw, alpha=alpha, ls=ls, label=label)


def plot_prediction_case(row: pd.Series, pred: Dict[str, np.ndarray | str], data, hist_features: List[str], out_path: Path) -> None:
    """绘制单个样本的预测效果图。"""

    i = int(row["array_row"])
    hidx = {name: j for j, name in enumerate(hist_features)}
    future = pred["future_grid"]
    true_delta = pred["y_true"][i].astype(float)
    p300_delta = pred["pred_v300"][i].astype(float)
    p307_delta = pred["pred_v307"][i].astype(float)
    valid = pred["mask"][i].astype(bool)

    x_hist = data.x_hist[i].astype(float)
    steer_hist = x_hist[:, hidx["steering"]]
    steer_anchor = steer_hist[-1] if np.isfinite(steer_hist[-1]) else 0.0
    true_abs = steer_anchor + true_delta
    p300_abs = steer_anchor + p300_delta
    p307_abs = steer_anchor + p307_delta

    raw = load_raw_window(str(row.get("raw_vehicle_csv", "")), float(row.get("observation_s", np.nan)))

    fig, axes = plt.subplots(5, 1, figsize=(13.2, 9.6), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.axvspan(-3.0, 0.0, color="#ECFDF5", alpha=0.65)
        ax.axvspan(0.0, 2.0, color="#EFF6FF", alpha=0.75)
        ax.axvspan(2.0, EXTEND_TO_S, color="#F9FAFB", alpha=0.9)
        ax.axvline(0.0, color="#DC2626", lw=1.0, ls="--")
        ax.axvline(2.0, color="#9CA3AF", lw=0.9, ls=":")
        ax.grid(True, color="#E5E7EB", lw=0.6, alpha=0.9)

    # 绝对方向盘角：历史 + 真实未来 + 两版预测。
    if not raw.empty and "zx|SteeringWheel" in raw.columns:
        plot_line(axes_arr[0], raw["rel_anchor_s"].to_numpy(), raw["zx|SteeringWheel"].to_numpy(), color="#9CA3AF", label="真实后续(raw, 到+6s)", lw=1.0, alpha=0.75)
    plot_line(axes_arr[0], HIST_TIME, steer_hist, color="#374151", label="历史 steering", lw=1.2)
    plot_line(axes_arr[0], future[valid], true_abs[valid], color="#111827", label="真实 0-2s", lw=2.0)
    plot_line(axes_arr[0], future[valid], p307_abs[valid], color="#2563EB", label="v307 预测", lw=1.8)
    plot_line(axes_arr[0], future[valid], p300_abs[valid], color="#F97316", label="v300 参照", lw=1.4, ls="--")
    axes_arr[0].set_ylabel("方向盘角")
    axes_arr[0].legend(fontsize=8, loc="best", ncol=3)

    # 目标输出 steering_delta，直观看模型输出误差。
    plot_line(axes_arr[1], future[valid], true_delta[valid], color="#111827", label="真实 delta", lw=2.0)
    plot_line(axes_arr[1], future[valid], p307_delta[valid], color="#2563EB", label="v307 delta", lw=1.8)
    plot_line(axes_arr[1], future[valid], p300_delta[valid], color="#F97316", label="v300 delta", lw=1.4, ls="--")
    axes_arr[1].set_ylabel("steering_delta")

    if not raw.empty and "zx|ay" in raw.columns:
        plot_line(axes_arr[2], raw["rel_anchor_s"].to_numpy(), raw["zx|ay"].to_numpy(), color="#111827", label="raw ay", lw=1.2)
    else:
        plot_line(axes_arr[2], HIST_TIME, x_hist[:, hidx["ay"]], color="#374151", label="history ay", lw=1.1)
    axes_arr[2].set_ylabel("ay")

    if not raw.empty and "zx|vyaw" in raw.columns:
        plot_line(axes_arr[3], raw["rel_anchor_s"].to_numpy(), raw["zx|vyaw"].to_numpy(), color="#111827", label="raw yaw_rate", lw=1.2)
    else:
        plot_line(axes_arr[3], HIST_TIME, x_hist[:, hidx["yaw_rate"]], color="#374151", label="history yaw_rate", lw=1.1)
    axes_arr[3].set_ylabel("yaw rate")

    if not raw.empty and "zx|roll" in raw.columns:
        plot_line(axes_arr[4], raw["rel_anchor_s"].to_numpy(), raw["zx|roll"].to_numpy(), color="#111827", label="raw roll", lw=1.2)
    else:
        plot_line(axes_arr[4], HIST_TIME, x_hist[:, hidx["roll"]], color="#374151", label="history roll", lw=1.1)
    axes_arr[4].set_ylabel("roll")

    title = (
        f"#{int(row['gallery_rank']):03d} {row['gallery_category']} | {row['coarse_scene_label_cn']} | "
        f"v307_rmse={float(row['v307_rmse']):.3f}, v300_rmse={float(row['v300_rmse']):.3f}, "
        f"delta={float(row['delta_v307_minus_v300']):+.3f}\n"
        f"{row['event_uid']} | scene={row['scene_type']} route={row['route_event']} "
        f"| strong={row['strong_steer']} vehicle_strong={row['vehicle_strong']} bad10={row['within_bad_top10_by_v249']}"
    )
    fig.suptitle(title, fontsize=10.5, x=0.01, y=0.995, ha="left")
    axes_arr[-1].set_xlabel("相对锚点时间 / s（蓝底 0-2s 是模型预测范围；2s 后只是真实后续）")
    axes_arr[-1].set_xlim(-3.0, EXTEND_TO_S)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_summary_figures(table: pd.DataFrame) -> None:
    """生成整体效果图。"""

    groups = []
    groups.append(("all_test_delay0", table))
    groups.append(("within_bad_top10", table[table["within_bad_top10_by_v249"].eq(1)]))
    groups.append(("within_bad_top20", table[table["within_bad_top20_by_v249"].eq(1)]))
    groups.append(("strong_steer", table[table["strong_steer"].astype(bool)]))
    groups.append(("vehicle_ambiguous", table[table["v299_vehicle_ambiguous"].eq(1)]))
    rows = []
    for name, df in groups:
        if df.empty:
            continue
        rows.append({"group": name, "model": "v300", "rmse": float(df["v300_rmse"].mean()), "n": int(len(df))})
        rows.append({"group": name, "model": "v307", "rmse": float(df["v307_rmse"].mean()), "n": int(len(df))})
    summary = pd.DataFrame(rows)
    write_csv(summary, TABLES / "v309_group_rmse_from_npz.csv")

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    xlabels = summary["group"].drop_duplicates().tolist()
    x = np.arange(len(xlabels))
    width = 0.36
    for offset, model, color in [(-width / 2, "v300", "#F97316"), (width / 2, "v307", "#2563EB")]:
        vals = [summary[(summary["group"].eq(g)) & (summary["model"].eq(model))]["rmse"].iloc[0] for g in xlabels]
        ax.bar(x + offset, vals, width=width, label=model, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=20, ha="right")
    ax.set_ylabel("mean RMSE")
    ax.set_title("v307 selected model vs v300 reference (test delay0)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "v309_group_rmse_v307_vs_v300.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    colors = table["delta_v307_minus_v300"].to_numpy()
    sc = ax.scatter(table["v300_rmse"], table["v307_rmse"], c=colors, cmap="coolwarm", s=28, alpha=0.82)
    lim_max = float(max(table["v300_rmse"].max(), table["v307_rmse"].max()) * 1.05)
    ax.plot([0, lim_max], [0, lim_max], color="#111827", lw=1.0, ls="--")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("v300 RMSE")
    ax.set_ylabel("v307 RMSE")
    ax.set_title("每个 test delay0 事件：点在对角线下方表示 v307 更好")
    fig.colorbar(sc, ax=ax, label="v307 - v300 RMSE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "v309_event_scatter_v307_vs_v300.png", dpi=160)
    plt.close(fig)


def write_html(gallery: pd.DataFrame, table: pd.DataFrame) -> None:
    """生成静态 HTML 图册。"""

    cards = []
    for row in gallery.to_dict(orient="records"):
        img = html.escape(str(row["image_relpath"]).replace("\\", "/"))
        event = html.escape(str(row["event_uid"]))
        category = html.escape(str(row["gallery_category"]))
        label = html.escape(str(row["coarse_scene_label"]))
        label_cn = html.escape(str(row["coarse_scene_label_cn"]))
        delta = float(row["delta_v307_minus_v300"])
        delta_cls = "better" if delta < 0 else "worse"
        cards.append(
            f"""
            <article class="card" data-category="{category}" data-label="{label}">
              <div class="head">
                <div>
                  <div class="title">#{int(row['gallery_rank']):03d} {category} · {label_cn}</div>
                  <div class="meta">{event}</div>
                </div>
                <div class="delta {delta_cls}">delta {delta:+.3f}</div>
              </div>
              <a href="{img}" target="_blank" rel="noreferrer"><img loading="lazy" src="{img}" alt="{event}"></a>
              <div class="info">
                <span>v307={float(row['v307_rmse']):.3f}</span>
                <span>v300={float(row['v300_rmse']):.3f}</span>
                <span>scene={html.escape(str(row['scene_type']))}</span>
                <span>route={html.escape(str(row['route_event']))}</span>
                <span>bad10={int(row['within_bad_top10_by_v249'])}</span>
              </div>
            </article>
            """
        )
    cats = sorted(gallery["gallery_category"].astype(str).unique().tolist())
    labels = sorted(gallery["coarse_scene_label"].astype(str).unique().tolist())
    cat_options = "\n".join(f'<option value="{html.escape(c)}">{html.escape(c)}</option>' for c in cats)
    label_options = "\n".join(f'<option value="{html.escape(l)}">{html.escape(LABEL_CN.get(l, l))}</option>' for l in labels)

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v309 近期最好模型预测效果图册</title>
  <style>
    body {{ margin:0; font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; color:#111827; background:#f3f4f6; }}
    header {{ position:sticky; top:0; z-index:10; background:rgba(255,255,255,.96); border-bottom:1px solid #d1d5db; padding:14px 18px; }}
    h1 {{ margin:0 0 6px; font-size:20px; }}
    .sub {{ color:#4b5563; font-size:13px; line-height:1.5; }}
    .toolbar {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:10px; }}
    select,input {{ font:inherit; border:1px solid #d1d5db; border-radius:6px; background:#fff; padding:7px 9px; }}
    #search {{ min-width:320px; }}
    main {{ padding:18px; }}
    .summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; margin-bottom:18px; }}
    .panel {{ background:#fff; border:1px solid #d1d5db; border-radius:8px; padding:12px; }}
    .panel img {{ width:100%; display:block; background:#fff; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(620px,1fr)); gap:16px; }}
    .card {{ background:#fff; border:1px solid #d1d5db; border-radius:8px; overflow:hidden; box-shadow:0 1px 4px rgba(15,23,42,.05); }}
    .head {{ display:flex; justify-content:space-between; gap:10px; padding:10px 12px; border-bottom:1px solid #e5e7eb; }}
    .title {{ font-weight:700; font-size:14px; }}
    .meta {{ color:#6b7280; font-size:12px; word-break:break-all; margin-top:2px; }}
    .delta {{ align-self:flex-start; border-radius:999px; padding:4px 9px; font-size:12px; border:1px solid #d1d5db; }}
    .delta.better {{ color:#166534; background:#f0fdf4; border-color:#86efac; }}
    .delta.worse {{ color:#9a3412; background:#fff7ed; border-color:#fdba74; }}
    .card img {{ width:100%; display:block; background:#fff; }}
    .info {{ display:flex; flex-wrap:wrap; gap:6px 12px; padding:8px 12px; color:#374151; font-size:12px; border-top:1px solid #e5e7eb; }}
    .hidden {{ display:none!important; }}
  </style>
</head>
<body>
  <header>
    <h1>v309 近期最好模型预测效果图册</h1>
    <div class="sub">
      近期最好版本按当前记录为 <strong>v307_coarse_scene_init_aux003_film005_h64</strong>。
      图中蓝色是 v307，橙色是 v300 参照，黑色是真实 0-2s；2s 后灰/黑线只是真实后续，不是模型预测范围。
      test delay0 共 {len(table)} 个事件，本图册选取 {len(gallery)} 个代表性样本。
    </div>
    <div class="toolbar">
      <input id="search" placeholder="搜索 event_uid / label / category / route">
      <select id="cat"><option value="">全部类别</option>{cat_options}</select>
      <select id="label"><option value="">全部场景标签</option>{label_options}</select>
      <span id="count" class="sub"></span>
    </div>
  </header>
  <main>
    <section class="summary">
      <div class="panel"><img src="figures/v309_group_rmse_v307_vs_v300.png" alt="group rmse"></div>
      <div class="panel"><img src="figures/v309_event_scatter_v307_vs_v300.png" alt="event scatter"></div>
    </section>
    <section class="grid" id="cards">{''.join(cards)}</section>
  </main>
  <script>
    const cards = Array.from(document.querySelectorAll(".card"));
    const search = document.getElementById("search");
    const cat = document.getElementById("cat");
    const label = document.getElementById("label");
    const count = document.getElementById("count");
    function apply() {{
      const q = (search.value || "").toLowerCase().trim();
      let shown = 0;
      cards.forEach(card => {{
        const okQ = !q || card.innerText.toLowerCase().includes(q);
        const okC = !cat.value || card.dataset.category === cat.value;
        const okL = !label.value || card.dataset.label === label.value;
        const show = okQ && okC && okL;
        card.classList.toggle("hidden", !show);
        if (show) shown += 1;
      }});
      count.textContent = `当前显示 ${{shown}} / ${{cards.length}}`;
    }}
    [search, cat, label].forEach(el => {{
      el.addEventListener("input", apply);
      el.addEventListener("change", apply);
    }});
    apply();
  </script>
</body>
</html>
"""
    (OUT / "index.html").write_text(html_text, encoding="utf-8")


def write_report(table: pd.DataFrame, gallery: pd.DataFrame, selected_model: str, v300_model: str, elapsed_s: float, zip_path: Path) -> Path:
    """写中文报告。"""

    def mean_rmse(df: pd.DataFrame, col: str) -> float:
        return float(df[col].mean()) if len(df) else float("nan")

    lines = [
        "# v309 近期最好模型预测效果图册",
        "",
        "## 结论",
        "",
        f"- 近期最好版本：`{selected_model}`。",
        f"- 参照版本：`{v300_model}`。",
        f"- test delay0/all：v300 `{mean_rmse(table, 'v300_rmse'):.6f}` -> v307 `{mean_rmse(table, 'v307_rmse'):.6f}`。",
        f"- test delay0/within_bad_top10：v300 `{mean_rmse(table[table['within_bad_top10_by_v249'].eq(1)], 'v300_rmse'):.6f}` -> v307 `{mean_rmse(table[table['within_bad_top10_by_v249'].eq(1)], 'v307_rmse'):.6f}`。",
        "",
        "## 图册",
        "",
        f"- HTML：`{OUT / 'index.html'}`",
        f"- 代表性样本图数：`{len(gallery)}`",
        f"- 全 test delay0 指标表：`{TABLES / 'v309_test_delay0_prediction_effect_table.csv'}`",
        f"- 图册样本表：`{TABLES / 'v309_gallery_sample_manifest.csv'}`",
        "",
        "## 如何读图",
        "",
        "- 方向盘角第一行：灰色/黑色为真实车辆轨迹，蓝色为 v307，橙色为 v300。",
        "- 第二行是模型真正预测的目标 `steering_delta`，只覆盖 `0-2s`。",
        "- `2s` 后只展示真实后续，帮助判断车辆是否继续失稳、回正或出现二次修正；这不是模型预测范围。",
        "",
        "## 边界",
        "",
        "- v307 仍使用 v306 粗场景 seed，其中直道连续/紧急子类还需要人工确认。",
        "- 本图册用于观察近期最好模型效果，不代表最终部署模型。",
        "",
        "## 验证",
        "",
        "- 读取 v307 NPZ 预测包成功。",
        "- test delay0 事件数 `232`。",
        f"- ZIP：`{zip_path}`。",
        f"- 生成耗时：`{elapsed_s:.1f}s`。",
    ]
    path = REPORTS / "v309_recent_best_prediction_effect_gallery_cn.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_zip() -> Path:
    """打包主要产物。"""

    zip_path = OUT / "v309_recent_best_prediction_effect_gallery_20260704.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in OUT.rglob("*"):
            if path == zip_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(OUT))
    return zip_path


def main() -> None:
    started = time.time()
    ensure_dirs()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    print("[v309] 读取 v307 预测和 v304/v236 数据", flush=True)
    pred = load_predictions()
    prepared = V304.prepare_v304_data(hard_event_extra=0.0)
    data = prepared.data
    hist_features = infer_hist_feature_names(data.feature_names)
    table = build_delay0_test_table(pred, data)
    write_csv(table, TABLES / "v309_test_delay0_prediction_effect_table.csv")
    write_summary_figures(table)

    gallery = choose_gallery_samples(table)
    rows: List[Dict[str, object]] = []
    for pos, row in gallery.iterrows():
        file_name = (
            f"rank{int(row['gallery_rank']):03d}_"
            f"{safe_name(row['gallery_category'], 32)}_"
            f"{safe_name(row['coarse_scene_label'], 28)}.png"
        )
        out_path = FIGURES / "prediction_cases" / file_name
        plot_prediction_case(row, pred, data, hist_features, out_path)
        record = row.to_dict()
        record["image_path"] = str(out_path)
        record["image_relpath"] = out_path.relative_to(OUT).as_posix()
        rows.append(record)
        if (pos + 1) % 20 == 0 or (pos + 1) == len(gallery):
            print(f"[v309] plotted {pos + 1}/{len(gallery)}", flush=True)

    gallery_out = pd.DataFrame(rows)
    write_csv(gallery_out, TABLES / "v309_gallery_sample_manifest.csv")
    write_html(gallery_out, table)
    zip_path = make_zip()
    elapsed = time.time() - started
    report_path = write_report(
        table,
        gallery_out,
        str(pred["best_v307_model"]),
        str(pred["v300_reference_model"]),
        elapsed,
        zip_path,
    )
    guardrail = {
        "version": "v309_recent_best_prediction_effect_gallery_20260704",
        "recent_best_model": str(pred["best_v307_model"]),
        "v300_reference_model": str(pred["v300_reference_model"]),
        "source_prediction_npz": str(V307_PRED),
        "test_delay0_event_n": int(len(table)),
        "gallery_case_n": int(len(gallery_out)),
        "model_prediction_horizon_s": [float(pred["future_grid"][0]), float(pred["future_grid"][-1])],
        "extended_true_future_display_to_s": EXTEND_TO_S,
        "extended_true_future_is_model_prediction": False,
        "test_all_v300_rmse": float(table["v300_rmse"].mean()),
        "test_all_v307_rmse": float(table["v307_rmse"].mean()),
        "zip_path": str(zip_path),
        "report_path": str(report_path),
        "html_index": str(OUT / "index.html"),
        "elapsed_s": elapsed,
    }
    write_json(guardrail, LOGS / "guardrail_check.json")
    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
