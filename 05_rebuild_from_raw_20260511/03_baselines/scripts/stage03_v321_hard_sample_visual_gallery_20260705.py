#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第321版：第320版困难样本可视化图册。

这个脚本不训练新模型，也不修改第320版结果。
它读取第320版逐样本指标、候选原型和第316版预测包，重建测试集上的：
- 第316版原预测；
- 第320版门控修正后预测；
- 候选最优上限；
- 真实0到2秒方向盘变化；
- 锚点前3秒到锚点后6秒的真实车辆信号。

用途是让人工快速看清困难样本到底长什么样，尤其看错误是否来自
“方向盘快速转动引起的后续车辆失稳或大幅横向响应”。
"""

from __future__ import annotations

import html
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "03_baselines"
SCRIPTS = BASELINES / "scripts"
V320_SCRIPT = SCRIPTS / "stage03_v320_rank_budget_repair_gate_20260705.py"
V320_DIR = BASELINES / "v320_rank_budget_repair_gate_20260705"
V320_TABLES = V320_DIR / "tables"

OUT = BASELINES / "v321_hard_sample_visual_gallery_20260705"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"

SELECTED_METHOD = "第320版-排序配额修复门控"
BASE_METHOD = "v316_selected_base"
ORACLE_METHOD = "候选最优上限"
EXTEND_TO_S = 6.0


def import_module_from_path(module_name: str, path: Path):
    """按路径导入第320版脚本，只复用函数，不触发主流程。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少依赖脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V320 = import_module_from_path("stage03_v320_for_v321_gallery", V320_SCRIPT)


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in [TABLES, FIGURES, REPORTS, LOGS]:
        folder.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """保存中文友好的表格。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, object], path: Path) -> None:
    """保存日志。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_filename(value: object, max_len: int = 150) -> str:
    """生成适合文件名的短文本。"""

    text = str(value)
    out: List[str] = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    compact = "".join(out).strip("_")
    return compact[:max_len] if compact else "event"


def bool_series(series: pd.Series) -> pd.Series:
    """把表里的布尔字段稳妥转成布尔值。"""

    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def cn_bool(value: object) -> str:
    """把布尔值显示成中文。"""

    return "是" if bool(value) else "否"


def load_raw_window(raw_path: str, observation_s: float, end_s: float = EXTEND_TO_S) -> pd.DataFrame:
    """读取锚点前后真实车辆信号窗口。"""

    path = Path(str(raw_path))
    if not path.exists() or not np.isfinite(float(observation_s)):
        return pd.DataFrame()
    needed = ["StorageTime", "zx|SteeringWheel", "zx|ay", "zx|vyaw", "zx|roll"]
    try:
        df = pd.read_csv(path, usecols=lambda c: c in needed)
    except Exception:
        return pd.DataFrame()
    if df.empty or "StorageTime" not in df.columns:
        return pd.DataFrame()
    t = pd.to_datetime(df["StorageTime"], errors="coerce")
    if t.isna().all():
        return pd.DataFrame()
    rel_record = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    rel_anchor = rel_record - float(observation_s)
    keep = (rel_anchor >= -3.2) & (rel_anchor <= end_s)
    out = df.loc[keep].copy()
    out["rel_anchor_s"] = rel_anchor[keep]
    return out.sort_values("rel_anchor_s").reset_index(drop=True)


def interp_anchor_steering(raw: pd.DataFrame) -> float:
    """从真实方向盘信号中取锚点处方向盘角，缺失时回退到0。"""

    if raw.empty or "zx|SteeringWheel" not in raw.columns or "rel_anchor_s" not in raw.columns:
        return 0.0
    t = raw["rel_anchor_s"].to_numpy(dtype=float)
    y = raw["zx|SteeringWheel"].to_numpy(dtype=float)
    keep = np.isfinite(t) & np.isfinite(y)
    if not keep.any():
        return 0.0
    order = np.argsort(t[keep])
    tt = t[keep][order]
    yy = y[keep][order]
    if tt.size == 1:
        return float(yy[0])
    return float(np.interp(0.0, tt, yy))


def plot_line(ax, t: np.ndarray, y: np.ndarray, *, color: str, label: str, lw: float, alpha: float = 1.0, ls: str = "-") -> None:
    """只画有效数值，避免断线和异常值导致整张图失败。"""

    tt = np.asarray(t, dtype=float)
    yy = np.asarray(y, dtype=float)
    valid = np.isfinite(tt) & np.isfinite(yy)
    if valid.any():
        ax.plot(tt[valid], yy[valid], color=color, label=label, lw=lw, alpha=alpha, ls=ls)


def category_for_row(row: pd.Series) -> str:
    """按第320版对困难样本的处理结果做可读分组。"""

    selected = bool(row["第320是否修正"])
    gain_320 = float(row["第320相对第316收益"])
    oracle_gain = float(row["候选上限相对第316收益"])
    if selected and gain_320 >= 0.02:
        return "修正后变好"
    if selected and gain_320 <= -0.02:
        return "修正后变坏"
    if oracle_gain >= 0.10 and gain_320 < oracle_gain * 0.35:
        return "候选有空间但门控未抓住"
    if not selected:
        return "未修正仍困难"
    return "修正幅度很小"


def build_reconstructed_curves() -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
    """重建测试集第320版曲线和候选上限曲线。"""

    meta, arrays = V320.load_delay0_dataset()
    grid = arrays["grid"]
    split = meta["split"].astype(str).to_numpy()
    test_mask = split == "test"
    test_meta = meta.loc[test_mask].reset_index(drop=True).copy()
    test_event = test_meta["event_uid"].astype(str).to_numpy()

    prototype_path = V320_TABLES / "v320_train_residual_prototypes.csv"
    per_sample_path = V320_TABLES / "v320_test_per_sample_metrics.csv"
    if not prototype_path.exists():
        raise FileNotFoundError(f"缺少第320版候选原型表：{prototype_path}")
    if not per_sample_path.exists():
        raise FileNotFoundError(f"缺少第320版测试逐样本表：{per_sample_path}")

    prototypes = pd.read_csv(prototype_path, encoding="utf-8-sig").to_numpy(dtype=float)
    candidates_all = V320.build_candidates(arrays["y316"], grid, prototypes)
    test_candidates = candidates_all.curves[test_mask]

    metrics = pd.read_csv(per_sample_path, encoding="utf-8-sig")
    v320 = metrics[metrics["method_name"].eq(SELECTED_METHOD)].copy()
    base = metrics[metrics["method_name"].eq(BASE_METHOD)].copy()
    oracle = metrics[metrics["method_name"].eq(ORACLE_METHOD)].copy()
    if len(v320) != len(test_meta) or len(base) != len(test_meta) or len(oracle) != len(test_meta):
        raise AssertionError("第320版逐样本表与测试集数量不一致")

    v320 = v320.set_index("event_uid").reindex(test_event).reset_index()
    base = base.set_index("event_uid").reindex(test_event).reset_index()
    oracle = oracle.set_index("event_uid").reindex(test_event).reset_index()
    if v320["sample_rmse"].isna().any() or oracle["sample_rmse"].isna().any():
        raise AssertionError("第320版逐样本表与测试事件编号未完全对齐")

    y_true = arrays["y_true"][test_mask]
    y316 = arrays["y316"][test_mask]
    y300 = arrays["y300"][test_mask]
    y307 = arrays["y307"][test_mask]

    chosen_320 = v320["chosen_candidate_idx"].fillna(0).to_numpy(dtype=int)
    alpha_320 = v320["fusion_alpha"].fillna(0.0).to_numpy(dtype=float)
    selected_curve_320 = test_candidates[np.arange(len(test_meta)), chosen_320, :]
    pred_320 = y316 + alpha_320[:, None] * (selected_curve_320 - y316)

    chosen_oracle = oracle["chosen_candidate_idx"].fillna(0).to_numpy(dtype=int)
    pred_oracle = test_candidates[np.arange(len(test_meta)), chosen_oracle, :]

    check_rmse = V320.curve_rmse(pred_320, y_true)
    check_max_abs = float(np.nanmax(np.abs(check_rmse - v320["sample_rmse"].to_numpy(dtype=float))))
    if check_max_abs > 1e-5:
        raise AssertionError(f"重建第320版曲线与逐样本表不一致，最大误差={check_max_abs}")

    out = test_meta.copy()
    out["第316误差"] = base["sample_rmse"].to_numpy(dtype=float)
    out["第320误差"] = v320["sample_rmse"].to_numpy(dtype=float)
    out["候选上限误差"] = oracle["sample_rmse"].to_numpy(dtype=float)
    out["第320相对第316收益"] = out["第316误差"] - out["第320误差"]
    out["候选上限相对第316收益"] = out["第316误差"] - out["候选上限误差"]
    out["第320是否修正"] = bool_series(v320["selected_for_correction"]).to_numpy()
    out["第320修正通道"] = v320["selected_channel"].fillna("未校正").astype(str).to_numpy()
    out["第320候选名"] = v320["chosen_candidate_name"].fillna("原预测不改").astype(str).to_numpy()
    out["候选上限名"] = oracle["chosen_candidate_name"].fillna("原预测不改").astype(str).to_numpy()
    out["困难前10"] = bool_series(v320["within_bad_top10_by_v249"]).to_numpy()
    out["困难前20"] = bool_series(v320["within_bad_top20_by_v249"]).to_numpy()
    out["强方向盘"] = bool_series(v320["strong_steer"]).to_numpy()
    out["真实峰值"] = v320["true_peak_signed"].to_numpy(dtype=float)
    out["第316峰值"] = base["pred_peak_signed"].to_numpy(dtype=float)
    out["第320峰值"] = v320["pred_peak_signed"].to_numpy(dtype=float)
    out["第320峰值比例"] = v320["peak_ratio"].to_numpy(dtype=float)
    out["困难分数"] = v320["hard_proxy_score"].to_numpy(dtype=float)
    out["候选坏风险"] = v320["candidate_bad_prob"].to_numpy(dtype=float)
    out["候选正收益概率"] = v320["candidate_pos_prob"].to_numpy(dtype=float)
    out["困难样本分组"] = out.apply(category_for_row, axis=1)

    curves = {
        "grid": grid,
        "真实": y_true,
        "第316": y316,
        "第320": pred_320,
        "候选上限": pred_oracle,
        "第300": y300,
        "第307": y307,
    }
    meta_info = {
        "candidate_names": candidates_all.names,
        "重建误差最大偏差": check_max_abs,
        "测试样本数": int(len(test_meta)),
    }
    return out, curves, meta_info


def plot_case(row: pd.Series, curves: Dict[str, np.ndarray], row_pos: int, out_path: Path) -> None:
    """绘制单个困难样本的完整观察图。"""

    grid = curves["grid"]
    true_delta = curves["真实"][row_pos]
    base_delta = curves["第316"][row_pos]
    fixed_delta = curves["第320"][row_pos]
    oracle_delta = curves["候选上限"][row_pos]

    raw = load_raw_window(str(row.get("raw_vehicle_csv", "")), float(row.get("observation_s", math.nan)))
    anchor = interp_anchor_steering(raw)
    true_abs = anchor + true_delta
    base_abs = anchor + base_delta
    fixed_abs = anchor + fixed_delta
    oracle_abs = anchor + oracle_delta

    fig, axes = plt.subplots(5, 1, figsize=(13.6, 9.8), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.axvspan(-3.0, 0.0, color="#ECFDF5", alpha=0.65)
        ax.axvspan(0.0, 2.0, color="#EFF6FF", alpha=0.78)
        ax.axvspan(2.0, EXTEND_TO_S, color="#F9FAFB", alpha=0.95)
        ax.axvline(0.0, color="#DC2626", lw=1.0, ls="--")
        ax.axvline(2.0, color="#9CA3AF", lw=0.9, ls=":")
        ax.grid(True, color="#E5E7EB", lw=0.6, alpha=0.92)

    if not raw.empty and "zx|SteeringWheel" in raw.columns:
        plot_line(
            axes_arr[0],
            raw["rel_anchor_s"].to_numpy(),
            raw["zx|SteeringWheel"].to_numpy(),
            color="#9CA3AF",
            label="真实后续到6秒",
            lw=1.0,
            alpha=0.75,
        )
    plot_line(axes_arr[0], grid, true_abs, color="#111827", label="真实0到2秒", lw=2.2)
    plot_line(axes_arr[0], grid, base_abs, color="#F97316", label="第316原预测", lw=1.6, ls="--")
    plot_line(axes_arr[0], grid, fixed_abs, color="#2563EB", label="第320修正", lw=2.0)
    plot_line(axes_arr[0], grid, oracle_abs, color="#16A34A", label="候选上限", lw=1.5, ls=":")
    axes_arr[0].set_ylabel("方向盘角")
    axes_arr[0].legend(fontsize=8, loc="best", ncol=4)

    plot_line(axes_arr[1], grid, true_delta, color="#111827", label="真实变化量", lw=2.2)
    plot_line(axes_arr[1], grid, base_delta, color="#F97316", label="第316变化量", lw=1.6, ls="--")
    plot_line(axes_arr[1], grid, fixed_delta, color="#2563EB", label="第320变化量", lw=2.0)
    plot_line(axes_arr[1], grid, oracle_delta, color="#16A34A", label="候选上限变化量", lw=1.5, ls=":")
    axes_arr[1].set_ylabel("方向盘变化")

    signal_specs = [
        ("zx|ay", "横向加速度", axes_arr[2]),
        ("zx|vyaw", "横摆角速度", axes_arr[3]),
        ("zx|roll", "侧倾", axes_arr[4]),
    ]
    for col, label, ax in signal_specs:
        if not raw.empty and col in raw.columns:
            plot_line(ax, raw["rel_anchor_s"].to_numpy(), raw[col].to_numpy(), color="#111827", label=label, lw=1.2)
        ax.set_ylabel(label)

    title = (
        f"#{int(row['图册序号']):03d} {row['困难样本分组']}｜"
        f"第316误差 {float(row['第316误差']):.3f}，第320误差 {float(row['第320误差']):.3f}，"
        f"收益 {float(row['第320相对第316收益']):+.3f}，候选上限收益 {float(row['候选上限相对第316收益']):+.3f}\n"
        f"事件编号：{row['event_uid']}｜通道：{row['第320修正通道']}｜"
        f"第320候选：{row['第320候选名']}｜候选上限：{row['候选上限名']}｜"
        f"困难前10：{cn_bool(row['困难前10'])}｜强方向盘：{cn_bool(row['强方向盘'])}"
    )
    fig.suptitle(title, fontsize=10.0, x=0.01, y=0.995, ha="left")
    axes_arr[-1].set_xlabel("相对锚点时间/秒（蓝底0到2秒是预测范围；2秒后只是真实后续）")
    axes_arr[-1].set_xlim(-3.0, EXTEND_TO_S)
    fig.tight_layout(rect=(0, 0, 1, 0.940))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=145)
    plt.close(fig)


def write_overview_figures(gallery: pd.DataFrame) -> None:
    """生成一张总览图，帮助先看哪些组最严重。"""

    group_order = ["修正后变坏", "候选有空间但门控未抓住", "未修正仍困难", "修正后变好", "修正幅度很小"]
    counts = gallery["困难样本分组"].value_counts().reindex(group_order).dropna()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(np.arange(len(counts)), counts.to_numpy(dtype=float), color="#2563EB")
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts.index.tolist(), rotation=15, ha="right")
    ax.set_ylabel("样本数")
    ax.set_title("第320版测试困难样本分组")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "困难样本分组总览.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    colors = gallery["第320相对第316收益"].to_numpy(dtype=float)
    sc = ax.scatter(gallery["第316误差"], gallery["第320误差"], c=colors, cmap="coolwarm_r", s=48, alpha=0.88)
    max_lim = float(max(gallery["第316误差"].max(), gallery["第320误差"].max()) * 1.08)
    ax.plot([0, max_lim], [0, max_lim], color="#111827", lw=1.0, ls="--")
    ax.set_xlim(0, max_lim)
    ax.set_ylim(0, max_lim)
    ax.set_xlabel("第316误差")
    ax.set_ylabel("第320误差")
    ax.set_title("困难样本：点在线下方才是第320更好")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label="第320相对第316收益")
    fig.tight_layout()
    fig.savefig(FIGURES / "困难样本第316对第320散点.png", dpi=160)
    plt.close(fig)


def write_html(gallery: pd.DataFrame, summary: pd.DataFrame) -> None:
    """写一个本地可浏览图册页面。"""

    cards: List[str] = []
    for row in gallery.to_dict(orient="records"):
        img = html.escape(str(row["图片相对路径"]).replace("\\", "/"))
        event_uid = html.escape(str(row["event_uid"]))
        group = html.escape(str(row["困难样本分组"]))
        channel = html.escape(str(row["第320修正通道"]))
        cards.append(
            f"""
            <article class="card" data-group="{group}" data-channel="{channel}">
              <div class="head">
                <div>
                  <div class="title">#{int(row['图册序号']):03d} {group}</div>
                  <div class="meta">事件编号：{event_uid}</div>
                </div>
                <div class="pill">收益 {float(row['第320相对第316收益']):+.3f}</div>
              </div>
              <a href="{img}" target="_blank"><img loading="lazy" src="{img}" alt="{event_uid}"></a>
              <div class="info">
                <span>第316误差 {float(row['第316误差']):.3f}</span>
                <span>第320误差 {float(row['第320误差']):.3f}</span>
                <span>候选上限误差 {float(row['候选上限误差']):.3f}</span>
                <span>通道：{channel}</span>
                <span>困难前10：{cn_bool(row['困难前10'])}</span>
                <span>强方向盘：{cn_bool(row['强方向盘'])}</span>
              </div>
            </article>
            """
        )

    group_options = "\n".join(
        f'<option value="{html.escape(str(x))}">{html.escape(str(x))}</option>'
        for x in gallery["困难样本分组"].drop_duplicates().tolist()
    )
    summary_rows = "\n".join(
        f"<tr><td>{html.escape(str(r['困难样本分组']))}</td><td>{int(r['样本数'])}</td>"
        f"<td>{float(r['第320平均收益']):+.4f}</td><td>{float(r['候选上限平均收益']):+.4f}</td></tr>"
        for r in summary.to_dict(orient="records")
    )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>第320版困难样本图册</title>
  <style>
    body {{ margin:0; font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; color:#111827; background:#f3f4f6; }}
    header {{ position:sticky; top:0; z-index:10; background:rgba(255,255,255,.97); border-bottom:1px solid #d1d5db; padding:14px 18px; }}
    h1 {{ margin:0 0 6px; font-size:20px; }}
    .sub {{ color:#4b5563; font-size:13px; line-height:1.55; }}
    .toolbar {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:10px; }}
    select,input {{ font:inherit; border:1px solid #d1d5db; border-radius:6px; background:#fff; padding:7px 9px; }}
    #search {{ min-width:320px; }}
    main {{ padding:18px; }}
    table {{ border-collapse:collapse; width:100%; background:#fff; margin-bottom:16px; }}
    th,td {{ border:1px solid #d1d5db; padding:8px 9px; text-align:left; font-size:13px; }}
    th {{ background:#f9fafb; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(680px,1fr)); gap:16px; }}
    .card {{ background:#fff; border:1px solid #d1d5db; border-radius:8px; overflow:hidden; box-shadow:0 1px 4px rgba(15,23,42,.05); }}
    .head {{ display:flex; justify-content:space-between; gap:10px; padding:10px 12px; border-bottom:1px solid #e5e7eb; }}
    .title {{ font-weight:700; font-size:14px; }}
    .meta {{ color:#6b7280; font-size:12px; word-break:break-all; margin-top:2px; }}
    .pill {{ white-space:nowrap; align-self:flex-start; border-radius:999px; padding:4px 9px; font-size:12px; border:1px solid #d1d5db; background:#f9fafb; }}
    .card img {{ width:100%; display:block; background:#fff; }}
    .info {{ display:flex; flex-wrap:wrap; gap:6px 12px; padding:8px 12px; color:#374151; font-size:12px; border-top:1px solid #e5e7eb; }}
    .hidden {{ display:none!important; }}
  </style>
</head>
<body>
  <header>
    <h1>第320版困难样本图册</h1>
    <div class="sub">
      共 {len(gallery)} 个测试困难前20样本。图中黑线是真实0到2秒，橙线是第316原预测，蓝线是第320修正后预测，绿线是候选最优上限；2秒之后只展示真实后续，用来看车辆是否继续失稳。
    </div>
    <div class="toolbar">
      <input id="search" placeholder="搜索事件编号、分组、通道">
      <select id="group"><option value="">全部分组</option>{group_options}</select>
    </div>
  </header>
  <main>
    <table>
      <thead><tr><th>分组</th><th>样本数</th><th>第320平均收益</th><th>候选上限平均收益</th></tr></thead>
      <tbody>{summary_rows}</tbody>
    </table>
    <section class="grid" id="grid">{''.join(cards)}</section>
  </main>
  <script>
    const search = document.getElementById('search');
    const group = document.getElementById('group');
    const cards = Array.from(document.querySelectorAll('.card'));
    function applyFilter() {{
      const q = search.value.trim().toLowerCase();
      const g = group.value;
      for (const card of cards) {{
        const text = card.innerText.toLowerCase();
        const okQ = !q || text.includes(q);
        const okG = !g || card.dataset.group === g;
        card.classList.toggle('hidden', !(okQ && okG));
      }}
    }}
    search.addEventListener('input', applyFilter);
    group.addEventListener('change', applyFilter);
  </script>
</body>
</html>
"""
    (OUT / "index.html").write_text(html_text, encoding="utf-8")


def write_report(gallery: pd.DataFrame, summary: pd.DataFrame, meta_info: Dict[str, object]) -> Path:
    """写简短中文报告，给后续日志或讨论复用。"""

    lines = [
        "# 第320版困难样本图册结论",
        "",
        "## 关键结论",
        "",
        f"- 本图册覆盖测试困难前20样本，共 {len(gallery)} 个，其中困难前10为 {int(gallery['困难前10'].sum())} 个。",
        f"- 第320版真正修正的困难样本为 {int(gallery['第320是否修正'].sum())} 个；未修正为 {int((~gallery['第320是否修正']).sum())} 个。",
        f"- 第320版在困难前20上的平均收益为 {float(gallery['第320相对第316收益'].mean()):+.6f}；候选上限平均收益为 {float(gallery['候选上限相对第316收益'].mean()):+.6f}。",
        "- 这说明困难样本不是完全没有候选空间，主要问题仍在门控选择：大量样本保守不改，少数被改坏。",
        "",
        "## 分组摘要",
        "",
        summary.to_markdown(index=False),
        "",
        "## 输出",
        "",
        f"- 图册首页：{OUT / 'index.html'}",
        f"- 图册清单：{TABLES / 'v321_hard_sample_gallery_manifest.csv'}",
        f"- 重建误差最大偏差：{float(meta_info['重建误差最大偏差']):.8f}",
    ]
    path = REPORTS / "v321_hard_sample_visual_gallery_report_cn.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    started = time.time()
    ensure_dirs()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    sample_table, curves, meta_info = build_reconstructed_curves()
    hard = sample_table[sample_table["困难前20"]].copy().reset_index(drop=True)
    if hard.empty:
        raise AssertionError("未找到测试困难前20样本")

    order_map = {
        "修正后变坏": 0,
        "候选有空间但门控未抓住": 1,
        "未修正仍困难": 2,
        "修正后变好": 3,
        "修正幅度很小": 4,
    }
    hard["排序组"] = hard["困难样本分组"].map(order_map).fillna(99).astype(int)
    hard = hard.sort_values(
        ["排序组", "第316误差", "候选上限相对第316收益"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    hard.insert(0, "图册序号", np.arange(1, len(hard) + 1, dtype=int))

    event_to_pos = {uid: i for i, uid in enumerate(sample_table["event_uid"].astype(str).tolist())}
    image_paths: List[str] = []
    for _, row in hard.iterrows():
        event_uid = str(row["event_uid"])
        row_pos = int(event_to_pos[event_uid])
        filename = f"{int(row['图册序号']):03d}_{safe_filename(row['困难样本分组'])}_{safe_filename(event_uid)}.png"
        out_path = FIGURES / filename
        plot_case(row, curves, row_pos, out_path)
        image_paths.append(str(out_path.relative_to(OUT)).replace("\\", "/"))
    hard["图片相对路径"] = image_paths
    hard = hard.drop(columns=["排序组"])

    summary = (
        hard.groupby("困难样本分组", dropna=False)
        .agg(
            样本数=("event_uid", "count"),
            第320平均收益=("第320相对第316收益", "mean"),
            候选上限平均收益=("候选上限相对第316收益", "mean"),
            第316平均误差=("第316误差", "mean"),
            第320平均误差=("第320误差", "mean"),
            候选上限平均误差=("候选上限误差", "mean"),
            已修正数=("第320是否修正", "sum"),
        )
        .reset_index()
    )
    summary["排序组"] = summary["困难样本分组"].map(order_map).fillna(99).astype(int)
    summary = summary.sort_values("排序组").drop(columns=["排序组"]).reset_index(drop=True)

    write_csv(hard, TABLES / "v321_hard_sample_gallery_manifest.csv")
    write_csv(summary, TABLES / "v321_hard_sample_group_summary.csv")
    write_overview_figures(hard)
    write_html(hard, summary)
    report_path = write_report(hard, summary, meta_info)

    guardrail = {
        "pass": True,
        "only_visualization": True,
        "uses_existing_v320_outputs": True,
        "does_not_retrain_model": True,
        "hard_sample_count": int(len(hard)),
        "hard_top10_count": int(hard["困难前10"].sum()),
        "corrected_hard_count": int(hard["第320是否修正"].sum()),
        "uncorrected_hard_count": int((~hard["第320是否修正"]).sum()),
        "v320_mean_gain_hard20": float(hard["第320相对第316收益"].mean()),
        "oracle_mean_gain_hard20": float(hard["候选上限相对第316收益"].mean()),
        "reconstruction_max_abs_rmse_diff": float(meta_info["重建误差最大偏差"]),
        "index_html": str(OUT / "index.html"),
        "manifest": str(TABLES / "v321_hard_sample_gallery_manifest.csv"),
        "summary": str(TABLES / "v321_hard_sample_group_summary.csv"),
        "report": str(report_path),
        "runtime_seconds": float(time.time() - started),
    }
    write_json(guardrail, LOGS / "guardrail_check.json")
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
