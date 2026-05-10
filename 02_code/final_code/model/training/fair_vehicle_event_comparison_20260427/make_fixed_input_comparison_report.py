# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from future_steer_speed_subjectsplit_masked import FS


PROJECT_ROOT = TRAINING_DIR.parents[3]
RUN_ROOT = PROJECT_ROOT / "tmp" / "event_conditioned_runs"
OUT_DIR = PROJECT_ROOT / "tmp" / "event_conditioned_runs" / "fair_vehicle_event_comparison_20260428_fixed_input_report"
CASE_FILE = THIS_DIR / "shared_prediction_cases_test.csv"
FIXED_RERUN_START = "20260428_013000"

ALL_MODEL_SPECS = [
    ("FAIR01_", "01 只有车辆"),
    ("FAIR02_", "02 显式事件+教师强制"),
    ("FAIR03_", "03 显式事件无教师强制"),
    ("FAIR04_", "04 显式事件+粗细双头"),
    ("FAIR05_", "05 显式事件+粗细双头+教师状态+风格"),
    ("FAIR06_", "06 直接预测+教师状态+风格"),
    ("FAIR07_", "07 直接预测+教师状态"),
    ("FAIR08_", "08 直接预测+连续风格"),
    ("FAIR09_", "09 粗细双头"),
    ("FAIR10_", "10 粗细双头+教师状态"),
    ("FAIR11_", "11 粗细双头+连续风格"),
    ("FAIR12_", "12 粗细双头+教师状态+连续风格"),
]

SELECTED_MODEL_SPECS = [
    ("FAIR01_", "车辆"),
    ("FAIR07_", "车辆+教师"),
    ("FAIR02_", "车辆+显式注入"),
    ("FAIR08_", "车辆+连续风格"),
    ("FAIR09_", "车辆+粗细双头"),
    ("FAIR12_", "车辆+粗细+连续风格+教师"),
    ("FAIR11_", "车辆+粗细+连续风格"),
    ("FAIR10_", "车辆+粗细+教师"),
]

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


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


def _timestamp_from_name(name: str) -> str:
    match = re.search(r"(20\d{6}_\d{6})$", name)
    return match.group(1) if match else ""


def load_summary(run_root: Path) -> dict[str, Any]:
    return json.loads((run_root / "run_summary.json").read_text(encoding="utf-8"))


def metrics_row(prefix: str, label: str) -> dict[str, Any]:
    run_root = latest_run(prefix)
    summary = load_summary(run_root)
    test_metrics = summary.get("final_test_metrics") or summary.get("test_metrics") or {}
    selection_summary = test_metrics.get("selection_summary") or {}
    context_meta = summary.get("context_augmentation") or {}
    context_tags: list[str] = []
    for aug in context_meta.get("augmentations", []):
        kind = str(aug.get("kind", ""))
        if kind == "teacher_state_context":
            context_tags.append(
                f"teacher(physio={aug.get('physio_available_count')},eeg={aug.get('eeg_available_count')})"
            )
        elif kind == "driver_style_context":
            context_tags.append(f"style(missing={aug.get('missing_sample_count')})")
        elif kind:
            context_tags.append(kind)
    return {
        "模型": label,
        "test steer RMSE": float(test_metrics.get("steer_rmse", np.nan)),
        "primary RMSE": float(selection_summary.get("overall_primary_steer_rmse", np.nan)),
        "tail RMSE": float(selection_summary.get("rmse_tail_abs_steer", np.nan)),
        "selection": float(selection_summary.get("selection_score", np.nan)),
        "best_epoch": summary.get("best_epoch"),
        "run_root": str(run_root),
        "context": "; ".join(context_tags),
    }


def write_metrics_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_df = pd.DataFrame([metrics_row(prefix, label) for prefix, label in ALL_MODEL_SPECS])
    selected_df = pd.DataFrame([metrics_row(prefix, label) for prefix, label in SELECTED_MODEL_SPECS])
    for name, df in [("metrics_table_12_models", all_df), ("metrics_table_selected_8_models", selected_df)]:
        df.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
        markdown_cols = ["模型", "test steer RMSE", "primary RMSE", "tail RMSE", "selection"]
        md_df = df[markdown_cols].copy()
        for col in markdown_cols[1:]:
            md_df[col] = md_df[col].map(lambda value: f"{float(value):.4f}")
        (OUT_DIR / f"{name}.md").write_text(dataframe_to_markdown(md_df), encoding="utf-8")
        save_dark_table_png(md_df, OUT_DIR / f"{name}.png")
    return all_df, selected_df


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(lines) + "\n"


def save_dark_table_png(df: pd.DataFrame, out_path: Path) -> None:
    setup_fonts()
    nrows, ncols = df.shape
    fig_width = 13.2 if nrows >= 12 else 12.6
    fig_height = 0.48 * (nrows + 1) + 0.35
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    fig.patch.set_facecolor("#141414")
    ax.set_facecolor("#141414")
    ax.axis("off")

    col_widths = [0.40, 0.17, 0.16, 0.14, 0.13]
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        colLoc="left",
        cellLoc="left",
        colWidths=col_widths,
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2b2b2b")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#262626")
            cell.get_text().set_color("#d9d9d9")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#181818")
            cell.get_text().set_color("#d7d7d7")
        if col > 0:
            cell._loc = "right"
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def load_sequences(run_root: Path) -> dict[str, Any]:
    seq_path = run_root / "prediction_figures" / "test" / "prediction_sequences.npz"
    if not seq_path.exists():
        raise FileNotFoundError(f"缺少预测序列文件，请先生成预测图: {seq_path}")
    with np.load(seq_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def event_times(case_row: pd.Series) -> dict[str, tuple[str, float]]:
    fields = {
        "turn": "true_first_major_turn_onset_idx",
        "rev": "true_first_reversal_idx",
        "peak": "true_main_peak_idx",
    }
    out: dict[str, tuple[str, float]] = {}
    for label, col in fields.items():
        value = case_row.get(col, np.nan)
        if pd.notna(value) and int(value) >= 0:
            out[label] = (label, float(value) / float(FS))
    return out


def build_overlay_payload() -> tuple[list[dict[str, Any]], pd.DataFrame]:
    cases = pd.read_csv(CASE_FILE)
    cases["sample_key"] = cases["sample_key"].astype(str)
    payload: list[dict[str, Any]] = []
    for prefix, label in SELECTED_MODEL_SPECS:
        run_root = latest_run(prefix)
        seq = load_sequences(run_root)
        sample_keys = [str(item) for item in seq["sample_key"].astype(str)]
        key_to_idx = {key: idx for idx, key in enumerate(sample_keys)}
        payload.append({"prefix": prefix, "label": label, "run_root": run_root, "seq": seq, "key_to_idx": key_to_idx})
    return payload, cases


def plot_overlay_figures() -> None:
    setup_fonts()
    payload, cases = build_overlay_payload()
    fig_dir = OUT_DIR / "combined_prediction_figures"
    case_dir = fig_dir / "per_case"
    case_dir.mkdir(parents=True, exist_ok=True)

    overview_items: list[dict[str, Any]] = []
    for case_index, case_row in cases.reset_index(drop=True).iterrows():
        sample_key = str(case_row["sample_key"])
        first = payload[0]
        if sample_key not in first["key_to_idx"]:
            continue
        first_idx = first["key_to_idx"][sample_key]
        valid_len = int(np.sum(first["seq"]["mask"][first_idx] > 0))
        if valid_len <= 1:
            continue
        anchor = float(first["seq"]["ctx_raw"][first_idx, 0])
        times = np.arange(valid_len, dtype=np.float32) / float(FS)
        truth_abs_deg = np.degrees(first["seq"]["true"][first_idx, :valid_len, 0] + anchor)
        model_lines: list[tuple[str, np.ndarray]] = []
        for item in payload:
            idx = item["key_to_idx"].get(sample_key)
            if idx is None:
                continue
            pred_anchor = float(item["seq"]["ctx_raw"][idx, 0])
            pred_abs_deg = np.degrees(item["seq"]["pred"][idx, :valid_len, 0] + pred_anchor)
            model_lines.append((str(item["label"]), pred_abs_deg))
        events = event_times(case_row)
        overview_items.append(
            {
                "case_index": case_index + 1,
                "sample_key": sample_key,
                "selection_tag": str(case_row.get("selection_tag", "case")),
                "times": times,
                "truth": truth_abs_deg,
                "model_lines": model_lines,
                "events": events,
                "meta": case_row,
            }
        )
        save_case_overlay(case_dir / f"{case_index + 1:02d}_{safe_name(str(case_row.get('selection_tag', 'case')))}.png", overview_items[-1])

    save_overview_overlay(fig_dir / "overview_8_models_same_events.png", overview_items)
    save_legend(fig_dir / "legend_8_models.png", [label for _, label in SELECTED_MODEL_SPECS])
    pd.DataFrame(
        [
            {
                "plot_index": item["case_index"],
                "sample_key": item["sample_key"],
                "selection_tag": item["selection_tag"],
                "plot_file": str(case_dir / f"{item['case_index']:02d}_{safe_name(item['selection_tag'])}.png"),
            }
            for item in overview_items
        ]
    ).to_csv(fig_dir / "combined_plot_index.csv", index=False, encoding="utf-8-sig")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._-")
    return cleaned or "case"


def save_case_overlay(out_path: Path, item: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.5), dpi=180)
    ax.plot(item["times"], item["truth"], color="#111111", linewidth=2.6, label="真实值")
    for line_index, (label, values) in enumerate(item["model_lines"]):
        ax.plot(item["times"], values, color=COLORS[line_index % len(COLORS)], linewidth=1.45, alpha=0.92, label=label)
    for label, event_time in item["events"].values():
        ax.axvline(event_time, color="#444444", linestyle="--", linewidth=0.9, alpha=0.55)
        ax.text(event_time, ax.get_ylim()[1] * 0.95, label, rotation=90, va="top", ha="right", fontsize=7, color="#333333")
    meta = item["meta"]
    ax.set_title(
        f"{item['case_index']:02d} {item['selection_tag']} | {str(item['sample_key'])[:95]}\n"
        f"road={meta.get('road_type_anchor', 'unknown')} | morph={meta.get('eval_morphology_label', 'unknown')} | interaction={meta.get('interaction_slice', 'unknown')}",
        fontsize=10,
    )
    ax.set_xlabel("Time after anchor (s)")
    ax.set_ylabel("Steering wheel angle (deg)")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_overview_overlay(out_path: Path, items: list[dict[str, Any]]) -> None:
    if not items:
        return
    cols = 2
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 10.6, rows * 4.2), dpi=170, squeeze=False)
    axes_flat = axes.flatten()
    for ax, item in zip(axes_flat, items):
        ax.plot(item["times"], item["truth"], color="#111111", linewidth=2.2, label="真实值")
        for line_index, (label, values) in enumerate(item["model_lines"]):
            ax.plot(item["times"], values, color=COLORS[line_index % len(COLORS)], linewidth=1.15, alpha=0.9, label=label)
        for _, event_time in item["events"].values():
            ax.axvline(event_time, color="#555555", linestyle="--", linewidth=0.75, alpha=0.45)
        ax.set_title(f"{item['case_index']:02d} {item['selection_tag']}", fontsize=9)
        ax.set_xlabel("Time after anchor (s)")
        ax.set_ylabel("Steering wheel angle (deg)")
        ax.grid(alpha=0.18)
    for ax in axes_flat[len(items) :]:
        fig.delaxes(ax)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_legend(out_path: Path, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 1.4), dpi=180)
    ax.axis("off")
    handles = [plt.Line2D([0], [0], color="#111111", linewidth=2.5, label="真实值")]
    handles += [plt.Line2D([0], [0], color=COLORS[i % len(COLORS)], linewidth=2.0, label=label) for i, label in enumerate(labels)]
    ax.legend(handles=handles, labels=[h.get_label() for h in handles], ncol=3, loc="center", frameon=False)
    fig.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    setup_fonts()
    all_df, selected_df = write_metrics_tables()
    plot_overlay_figures()
    print(f"输出目录: {OUT_DIR}")
    print(f"12模型指标表: {OUT_DIR / 'metrics_table_12_models.png'}")
    print(f"8模型指标表: {OUT_DIR / 'metrics_table_selected_8_models.png'}")
    print(f"8模型同事件总览图: {OUT_DIR / 'combined_prediction_figures' / 'overview_8_models_same_events.png'}")
    print(f"12模型数量: {len(all_df)} | 8模型数量: {len(selected_df)}")


if __name__ == "__main__":
    main()
