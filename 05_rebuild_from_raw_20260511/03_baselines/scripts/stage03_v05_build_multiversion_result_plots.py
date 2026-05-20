# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REBUILD_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REBUILD_ROOT.parent
RUN_DIR = REBUILD_ROOT / "03_baselines" / "stage03_v05_physio_mechanism_comparison"
TABLE_DIR = RUN_DIR / "tables"
FIG_DIR = RUN_DIR / "figures"
COMPARISON_PATH = TABLE_DIR / "v05_physio_comparison_table.csv"
FS = 200.0


def resolve_run_root(path_text: str) -> Path:
    text = str(path_text or "").strip()
    if not text:
        return Path("")
    if text.startswith("/root/autodl-tmp/data_process/"):
        return PROJECT_ROOT / text.replace("/root/autodl-tmp/data_process/", "").replace("/", "\\")
    return Path(text)


def load_sequences(row: pd.Series) -> dict[str, Any] | None:
    run_root = resolve_run_root(str(row.get("run_root", "")))
    seq_path = run_root / "prediction_figures" / "test" / "prediction_sequences.npz"
    if not seq_path.exists():
        return None
    data = np.load(seq_path, allow_pickle=True)
    keys = [str(x) for x in data["sample_key"]]
    return {
        "exp_id": str(row["exp_id"]),
        "label_cn": str(row.get("label_cn", row["exp_id"])),
        "path": seq_path,
        "sample_key": keys,
        "pred": data["pred"],
        "true": data["true"],
        "mask": data["mask"],
        "index": {k: i for i, k in enumerate(keys)},
        "channel_names": [str(x) for x in data.get("channel_names", np.asarray(["steer"]))],
    }


def first_channel(seq: np.ndarray) -> np.ndarray:
    if seq.ndim == 1:
        return seq
    return seq[:, 0]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def select_sample_keys(seq_map: dict[str, dict[str, Any]], ref_id: str = "B0", count: int = 8) -> list[str]:
    if ref_id not in seq_map:
        ref_id = next(iter(seq_map))
    common = set(seq_map[ref_id]["sample_key"])
    for item in seq_map.values():
        common &= set(item["sample_key"])
    rows: list[tuple[float, float, str]] = []
    ref = seq_map[ref_id]
    for key in common:
        i = ref["index"][key]
        valid = ref["mask"][i].astype(bool)
        if not valid.any():
            continue
        true = first_channel(ref["true"][i])[valid]
        pred = first_channel(ref["pred"][i])[valid]
        peak = float(np.nanmax(np.abs(true)))
        rows.append((rmse(pred, true), peak, key))
    rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [x[2] for x in rows[:count]]


def draw_metric_overview(comp: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = [
        "B0",
        "B1",
        "S1",
        "S2",
        "S3",
        "S4",
        "SF1",
        "SF2",
        "SF3",
        "SF4",
        "C1",
        "C2",
        "C3",
        "C4",
        "A1",
        "A2",
        "A3",
        "T1",
        "T2",
        "T3",
        "T4",
    ]
    df = comp[comp["exp_id"].isin(order)].copy()
    df["order"] = df["exp_id"].map({k: i for i, k in enumerate(order)})
    df = df.sort_values("order")
    labels = [f"{r.exp_id}\n{str(r.label_cn).replace(' + ', '+')}" for r in df.itertuples()]

    fig, axes = plt.subplots(2, 1, figsize=(max(15, len(df) * 0.72), 8), sharex=True)
    for ax, col, title in [
        (axes[0], "test_steer_rmse", "整体 RMSE（越低越好）"),
        (axes[1], "tail_rmse", "尾段 RMSE（越低越好）"),
    ]:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        colors = ["#2e7d32" if v == np.nanmin(values) else "#90caf9" for v in values]
        ax.bar(np.arange(len(df)), values, color=colors, edgecolor="#335", linewidth=0.5)
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for i, v in enumerate(values):
            if math.isfinite(v):
                ax.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axes[-1].set_xticks(np.arange(len(df)))
    axes[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.suptitle("v0.5 新样本集：连续风格、生理信号、脑电和教师版本指标对照", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "v05_physio_eeg_metric_overview.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def draw_table(comp: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["exp_id", "label_cn", "test_steer_rmse", "primary_rmse", "tail_rmse", "selection", "large_wrong_side_rate", "large_severe_under_rate"]
    df = comp[[c for c in cols if c in comp.columns]].copy()
    df = df.sort_values("test_steer_rmse")
    rename = {
        "exp_id": "版本",
        "label_cn": "模型",
        "test_steer_rmse": "test RMSE",
        "primary_rmse": "主阶段",
        "tail_rmse": "尾段",
        "selection": "selection",
        "large_wrong_side_rate": "大响应错侧率",
        "large_severe_under_rate": "严重幅值不足率",
    }
    df = df.rename(columns=rename)
    if "模型" in df.columns:
        df["模型"] = df["模型"].map(lambda x: "\n".join(textwrap.wrap(str(x), width=18, break_long_words=False)))
    for col in df.columns:
        if col not in ["版本", "模型"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").map(lambda x: "" if not math.isfinite(float(x)) else f"{float(x):.4f}")
    fig_h = max(7, 0.42 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.65)
    try:
        table.auto_set_column_width(col=list(range(len(df.columns))))
    except Exception:
        pass
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dbe3")
        if r == 0:
            cell.set_facecolor("#eaf0f6")
            cell.set_text_props(fontweight="bold")
        elif r == 1:
            cell.set_facecolor("#e8f5e9")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f8fafc")
    ax.set_title("v0.5 新样本集多版本结果表（按 test RMSE 排序）", fontsize=15, fontweight="bold", pad=18)
    out = FIG_DIR / "v05_physio_eeg_result_table_white.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def draw_overlay_group(comp: pd.DataFrame, exp_ids: list[str], filename: str, title: str) -> Path | None:
    rows = comp[comp["exp_id"].isin(exp_ids)].copy()
    rows["order"] = rows["exp_id"].map({k: i for i, k in enumerate(exp_ids)})
    rows = rows.sort_values("order")
    seq_map = {}
    for _, row in rows.iterrows():
        item = load_sequences(row)
        if item is not None:
            seq_map[str(row["exp_id"])] = item
    if len(seq_map) < 2:
        return None
    sample_keys = select_sample_keys(seq_map, "B0" if "B0" in seq_map else next(iter(seq_map)), count=8)
    if not sample_keys:
        return None
    colors = {
        "B0": "#111111",
        "B1": "#607d8b",
        "S4": "#1976d2",
        "SF4": "#0d47a1",
        "C3": "#9c27b0",
        "C4": "#6a1b9a",
        "A3": "#ef6c00",
        "T1": "#2e7d32",
        "T2": "#00897b",
        "T3": "#7b1fa2",
        "T4": "#c62828",
    }
    fig, axes = plt.subplots(len(sample_keys), 1, figsize=(14, max(11, 2.0 * len(sample_keys))), sharex=False)
    if len(sample_keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, sample_keys):
        ref_item = next(iter(seq_map.values()))
        ref_idx = ref_item["index"][key]
        valid = ref_item["mask"][ref_idx].astype(bool)
        t = np.arange(valid.sum()) / FS
        true = first_channel(ref_item["true"][ref_idx])[valid]
        ax.plot(t, true, color="#000000", lw=2.0, label="真实")
        for exp_id, item in seq_map.items():
            idx = item["index"][key]
            valid_i = item["mask"][idx].astype(bool)
            n = min(valid.sum(), valid_i.sum())
            if n <= 0:
                continue
            pred = first_channel(item["pred"][idx])[valid_i][:n]
            ax.plot(t[:n], pred, lw=1.15, alpha=0.9, color=colors.get(exp_id), label=exp_id)
        ax.set_ylabel("方向盘")
        ax.set_title(key, fontsize=9, loc="left")
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=min(8, len(seq_map) + 1), fontsize=8, loc="upper right")
    axes[-1].set_xlabel("预测窗口时间 / s")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG_DIR / filename
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    comp = pd.read_csv(COMPARISON_PATH, encoding="utf-8-sig")
    comp = comp.drop_duplicates(["exp_id", "seed"], keep="last")
    outputs: list[Path] = []
    outputs.append(draw_metric_overview(comp))
    outputs.append(draw_table(comp))
    for result in [
        draw_overlay_group(comp, ["B0", "B1", "S4", "SF4", "C3", "C4", "A3"], "v05_multiversion_overlay_eeg_direct.png", "v0.5 多版本预测曲线对比：脑电直接输入与全生理融合"),
        draw_overlay_group(comp, ["B0", "B1", "T2", "T1", "T3", "T4"], "v05_multiversion_overlay_teacher.png", "v0.5 多版本预测曲线对比：不同教师蒸馏路线"),
    ]:
        if result is not None:
            outputs.append(result)
    index = FIG_DIR / "v05_physio_eeg_figure_index_cn.md"
    index.write_text(
        "# v0.5 生理/脑电多版本结果图索引\n\n"
        + "\n".join(f"- `{p}`" for p in outputs)
        + "\n",
        encoding="utf-8",
    )
    print("\n".join(str(p) for p in outputs))


if __name__ == "__main__":
    main()
