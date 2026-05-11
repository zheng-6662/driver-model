# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_fixed_input_comparison_report import (
    CASE_FILE,
    FIXED_RERUN_START,
    OUT_DIR as BASE_REPORT_DIR,
    RUN_ROOT,
    _timestamp_from_name,
    setup_fonts,
)
from future_steer_speed_subjectsplit_masked import FS


OUT_DIR = BASE_REPORT_DIR / "teacher_state_effect"

PAIRS = [
    {
        "pair": "直接预测",
        "without_prefix": "FAIR01_",
        "without_label": "车辆",
        "with_prefix": "FAIR07_",
        "with_label": "车辆+教师",
    },
    {
        "pair": "直接预测+连续风格",
        "without_prefix": "FAIR08_",
        "without_label": "车辆+连续风格",
        "with_prefix": "FAIR06_",
        "with_label": "车辆+教师+连续风格",
    },
    {
        "pair": "粗细双头",
        "without_prefix": "FAIR09_",
        "without_label": "车辆+粗细双头",
        "with_prefix": "FAIR10_",
        "with_label": "车辆+粗细双头+教师",
    },
    {
        "pair": "粗细双头+连续风格",
        "without_prefix": "FAIR11_",
        "without_label": "车辆+粗细双头+连续风格",
        "with_prefix": "FAIR12_",
        "with_label": "车辆+粗细双头+连续风格+教师",
    },
]

METRICS = [
    ("test steer RMSE", "test_steer_rmse"),
    ("primary RMSE", "primary_rmse"),
    ("tail RMSE", "tail_rmse"),
    ("selection", "selection"),
]


def latest_run(prefix: str) -> Path:
    candidates = [
        path
        for path in RUN_ROOT.glob(prefix + "*")
        if path.is_dir()
        and _timestamp_from_name(path.name) >= FIXED_RERUN_START
        and (path / "run_summary.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"找不到 {prefix} 在 {FIXED_RERUN_START} 之后的完整运行结果")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_summary(run_root: Path) -> dict[str, Any]:
    return json.loads((run_root / "run_summary.json").read_text(encoding="utf-8"))


def extract_metrics(run_root: Path) -> dict[str, float]:
    summary = load_summary(run_root)
    test_metrics = summary.get("final_test_metrics") or summary.get("test_metrics") or {}
    selection_summary = test_metrics.get("selection_summary") or {}
    return {
        "test_steer_rmse": float(test_metrics.get("steer_rmse", np.nan)),
        "primary_rmse": float(selection_summary.get("overall_primary_steer_rmse", np.nan)),
        "tail_rmse": float(selection_summary.get("rmse_tail_abs_steer", np.nan)),
        "selection": float(selection_summary.get("selection_score", np.nan)),
    }


def build_effect_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_index, spec in enumerate(PAIRS, start=1):
        without_run = latest_run(spec["without_prefix"])
        with_run = latest_run(spec["with_prefix"])
        without_metrics = extract_metrics(without_run)
        with_metrics = extract_metrics(with_run)
        row: dict[str, Any] = {
            "对照组": spec["pair"],
            "无教师模型": spec["without_label"],
            "加教师模型": spec["with_label"],
            "无教师run": str(without_run),
            "加教师run": str(with_run),
        }
        worse_count = 0
        for _, key in METRICS:
            delta = with_metrics[key] - without_metrics[key]
            row[f"无教师 {key}"] = without_metrics[key]
            row[f"加教师 {key}"] = with_metrics[key]
            row[f"变化 {key}"] = delta
            if delta > 0:
                worse_count += 1
        row["加教师变差指标数"] = worse_count
        rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(lines) + "\n"


def save_effect_tables(df: pd.DataFrame) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "teacher_state_effect_full.csv", index=False, encoding="utf-8-sig")
    compact_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        compact = {
            "对照组": row["对照组"],
            "无教师": row["无教师模型"],
            "加教师": row["加教师模型"],
            "test steer 变化": row["变化 test_steer_rmse"],
            "primary 变化": row["变化 primary_rmse"],
            "tail 变化": row["变化 tail_rmse"],
            "selection 变化": row["变化 selection"],
            "变差指标数": int(row["加教师变差指标数"]),
        }
        compact_rows.append(compact)
    compact_df = pd.DataFrame(compact_rows)
    compact_df.to_csv(OUT_DIR / "teacher_state_effect_compact.csv", index=False, encoding="utf-8-sig")
    md_df = compact_df.copy()
    for col in ["test steer 变化", "primary 变化", "tail 变化", "selection 变化"]:
        md_df[col] = md_df[col].map(lambda value: f"{float(value):+.4f}")
    (OUT_DIR / "teacher_state_effect_compact.md").write_text(dataframe_to_markdown(md_df), encoding="utf-8")
    save_dark_effect_table(md_df, OUT_DIR / "teacher_state_effect_compact.png")
    return compact_df


def save_dark_effect_table(df: pd.DataFrame, out_path: Path) -> None:
    setup_fonts()
    fig, ax = plt.subplots(figsize=(13.5, 3.2), dpi=180)
    fig.patch.set_facecolor("#141414")
    ax.set_facecolor("#141414")
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        colLoc="left",
        cellLoc="left",
        colWidths=[0.17, 0.19, 0.24, 0.10, 0.10, 0.10, 0.10, 0.08],
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2b2b2b")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#262626")
            cell.get_text().set_color("#e0e0e0")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#181818")
            text = cell.get_text().get_text()
            if 3 <= col <= 6 and text.startswith("+"):
                cell.get_text().set_color("#ff7777")
            elif 3 <= col <= 6 and text.startswith("-"):
                cell.get_text().set_color("#79d279")
            else:
                cell.get_text().set_color("#d7d7d7")
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_delta_bar_chart(compact_df: pd.DataFrame) -> None:
    setup_fonts()
    metric_cols = ["test steer 变化", "primary 变化", "tail 变化", "selection 变化"]
    metric_labels = ["test steer", "primary", "tail", "selection"]
    x = np.arange(len(compact_df))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12.5, 5.2), dpi=180)
    for i, (col, label) in enumerate(zip(metric_cols, metric_labels)):
        values = compact_df[col].to_numpy(dtype=float)
        colors = ["#d65f5f" if value > 0 else "#5da65d" for value in values]
        ax.bar(x + (i - 1.5) * width, values, width, label=label, color=colors, alpha=0.88)
    ax.axhline(0.0, color="#222222", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(compact_df["对照组"].tolist(), rotation=10, ha="right")
    ax.set_ylabel("加教师 - 无教师（正值=变差）")
    ax.set_title("教师状态注入的指标变化：多数指标为正，说明加教师后变差")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "teacher_state_metric_delta_bars.png", bbox_inches="tight")
    plt.close(fig)


def save_metric_pair_bars(df: pd.DataFrame) -> None:
    setup_fonts()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), dpi=180, squeeze=False)
    axes_flat = axes.flatten()
    for ax, (title, key) in zip(axes_flat, METRICS):
        labels = df["对照组"].tolist()
        without_values = df[f"无教师 {key}"].to_numpy(dtype=float)
        with_values = df[f"加教师 {key}"].to_numpy(dtype=float)
        x = np.arange(len(labels))
        width = 0.34
        ax.bar(x - width / 2, without_values, width, label="无教师", color="#4c78a8")
        ax.bar(x + width / 2, with_values, width, label="加教师", color="#e45756")
        for i, (without_value, with_value) in enumerate(zip(without_values, with_values)):
            ax.text(i, max(without_value, with_value) * 1.01, f"{with_value - without_value:+.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=8)
    fig.suptitle("无教师 vs 加教师：成对指标对比", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "teacher_state_metric_pair_bars.png", bbox_inches="tight")
    plt.close(fig)


def load_sequences(run_root: Path) -> dict[str, np.ndarray]:
    seq_path = run_root / "prediction_figures" / "test" / "prediction_sequences.npz"
    if not seq_path.exists():
        raise FileNotFoundError(f"缺少预测序列文件: {seq_path}")
    with np.load(seq_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def event_times(case_row: pd.Series) -> dict[str, float]:
    fields = {
        "turn": "true_first_major_turn_onset_idx",
        "rev": "true_first_reversal_idx",
        "peak": "true_main_peak_idx",
    }
    out: dict[str, float] = {}
    for label, col in fields.items():
        value = case_row.get(col, np.nan)
        if pd.notna(value) and int(value) >= 0:
            out[label] = float(value) / float(FS)
    return out


def save_prediction_pair_overviews() -> None:
    setup_fonts()
    cases = pd.read_csv(CASE_FILE)
    cases["sample_key"] = cases["sample_key"].astype(str)
    pred_dir = OUT_DIR / "prediction_pair_overviews"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for pair_index, spec in enumerate(PAIRS, start=1):
        without_run = latest_run(spec["without_prefix"])
        with_run = latest_run(spec["with_prefix"])
        without_seq = load_sequences(without_run)
        with_seq = load_sequences(with_run)
        without_map = {str(key): idx for idx, key in enumerate(without_seq["sample_key"].astype(str))}
        with_map = {str(key): idx for idx, key in enumerate(with_seq["sample_key"].astype(str))}

        items: list[dict[str, Any]] = []
        for case_idx, case_row in cases.reset_index(drop=True).iterrows():
            sample_key = str(case_row["sample_key"])
            if sample_key not in without_map or sample_key not in with_map:
                continue
            idx_without = without_map[sample_key]
            idx_with = with_map[sample_key]
            valid_len = int(np.sum(without_seq["mask"][idx_without] > 0))
            if valid_len <= 1:
                continue
            anchor = float(without_seq["ctx_raw"][idx_without, 0])
            with_anchor = float(with_seq["ctx_raw"][idx_with, 0])
            truth = np.degrees(without_seq["true"][idx_without, :valid_len, 0] + anchor)
            pred_without = np.degrees(without_seq["pred"][idx_without, :valid_len, 0] + anchor)
            pred_with = np.degrees(with_seq["pred"][idx_with, :valid_len, 0] + with_anchor)
            rmse_without = float(np.sqrt(np.mean((pred_without - truth) ** 2)))
            rmse_with = float(np.sqrt(np.mean((pred_with - truth) ** 2)))
            items.append(
                {
                    "case_idx": case_idx + 1,
                    "tag": str(case_row.get("selection_tag", "case")),
                    "times": np.arange(valid_len, dtype=np.float32) / float(FS),
                    "truth": truth,
                    "pred_without": pred_without,
                    "pred_with": pred_with,
                    "rmse_without": rmse_without,
                    "rmse_with": rmse_with,
                    "events": event_times(case_row),
                }
            )
        save_pair_overview(
            pred_dir / f"{pair_index:02d}_{spec['without_prefix'][:6]}_vs_{spec['with_prefix'][:6]}_no_teacher_vs_teacher.png",
            spec,
            items,
        )
        if items:
            save_pair_single_case(
                pred_dir / f"{pair_index:02d}_{spec['without_prefix'][:6]}_vs_{spec['with_prefix'][:6]}_case01_no_teacher_vs_teacher.png",
                spec,
                items[0],
            )


def save_pair_overview(out_path: Path, spec: dict[str, str], items: list[dict[str, Any]]) -> None:
    if not items:
        return
    cols = 2
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8.8, rows * 3.7), dpi=170, squeeze=False)
    axes_flat = axes.flatten()
    for ax, item in zip(axes_flat, items):
        ax.plot(item["times"], item["truth"], color="#111111", linewidth=2.25, label="真实值")
        ax.plot(item["times"], item["pred_without"], color="#4c78a8", linewidth=1.45, label=spec["without_label"])
        ax.plot(item["times"], item["pred_with"], color="#e45756", linewidth=1.45, label=spec["with_label"])
        for _, event_time in item["events"].items():
            ax.axvline(event_time, color="#555555", linestyle="--", linewidth=0.75, alpha=0.42)
        delta = item["rmse_with"] - item["rmse_without"]
        ax.set_title(f"{item['case_idx']:02d} {item['tag']} | RMSE变化={delta:+.2f}°", fontsize=9)
        ax.set_xlabel("Time after anchor (s)")
        ax.set_ylabel("Steering wheel angle (deg)")
        ax.grid(alpha=0.2)
    for ax in axes_flat[len(items) :]:
        fig.delaxes(ax)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"{spec['pair']}：无教师 vs 加教师（每个子图只有三条线）", fontsize=13)
    fig.tight_layout(rect=[0, 0.055, 1, 0.965])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_pair_single_case(out_path: Path, spec: dict[str, str], item: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=180)
    ax.plot(item["times"], item["truth"], color="#111111", linewidth=2.45, label="真实值")
    ax.plot(item["times"], item["pred_without"], color="#4c78a8", linewidth=1.7, label=spec["without_label"])
    ax.plot(item["times"], item["pred_with"], color="#e45756", linewidth=1.7, label=spec["with_label"])
    for label, event_time in item["events"].items():
        ax.axvline(event_time, color="#555555", linestyle="--", linewidth=0.9, alpha=0.5)
        ax.text(event_time, ax.get_ylim()[1] * 0.96, label, rotation=90, ha="right", va="top", fontsize=7)
    delta = item["rmse_with"] - item["rmse_without"]
    ax.set_title(
        f"{spec['pair']} | {item['tag']} | 无教师RMSE={item['rmse_without']:.2f}° | 加教师RMSE={item['rmse_with']:.2f}° | 变化={delta:+.2f}°"
    )
    ax.set_xlabel("Time after anchor (s)")
    ax.set_ylabel("Steering wheel angle (deg)")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    effect_df = build_effect_table()
    compact_df = save_effect_tables(effect_df)
    save_delta_bar_chart(compact_df)
    save_metric_pair_bars(effect_df)
    save_prediction_pair_overviews()
    print(f"输出目录: {OUT_DIR}")
    print(f"教师状态变化表: {OUT_DIR / 'teacher_state_effect_compact.png'}")
    print(f"教师状态指标变化柱状图: {OUT_DIR / 'teacher_state_metric_delta_bars.png'}")
    print(f"教师状态成对指标图: {OUT_DIR / 'teacher_state_metric_pair_bars.png'}")
    print(f"教师状态成对预测图目录: {OUT_DIR / 'prediction_pair_overviews'}")


if __name__ == "__main__":
    main()
