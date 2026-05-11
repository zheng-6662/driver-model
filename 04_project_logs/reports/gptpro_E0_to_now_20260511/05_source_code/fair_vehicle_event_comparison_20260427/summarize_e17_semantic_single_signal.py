# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REMOTE_ROOT = Path("/root/autodl-tmp/data_process")
REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"

DEFAULT_E17_RUNS = REPORTS_DIR / "style_physio_eeg_e17_semantic_single_signal_runs_20260511.csv"
DEFAULT_OUT_DIR = REPORTS_DIR / "style_physio_eeg_e17_semantic_single_signal_seed2026_summary_20260511"

REFERENCE_RUN_FILES = [
    REPORTS_DIR / "style_physio_eeg_e0_e2_fresh_3seed_runs_20260507.csv",
    REPORTS_DIR / "style_physio_eeg_e10_non_eeg_signal_runs_20260509.csv",
    REPORTS_DIR / "style_physio_eeg_e16_eeg_style_runs_20260511.csv",
]

REFERENCE_INFO = {
    "E2": {
        "model": "粗细双头 + 连续风格，不加生理/脑电",
        "signal": "无",
        "form": "无",
        "source": "已有 seed2026",
    },
    "E10A": {
        "model": "粗细双头 + 连续风格 + 心率原始输入",
        "signal": "心率",
        "form": "原始数值",
        "source": "已有 seed2026",
    },
    "E10B": {
        "model": "粗细双头 + 连续风格 + 皮电原始输入",
        "signal": "皮电",
        "form": "原始数值",
        "source": "已有 seed2026",
    },
    "E10C": {
        "model": "粗细双头 + 连续风格 + 肌电原始输入",
        "signal": "肌电",
        "form": "原始数值",
        "source": "已有 seed2026",
    },
    "E16B": {
        "model": "粗细双头 + 连续风格 + 脑电原始输入",
        "signal": "脑电",
        "form": "原始数值",
        "source": "已有 seed2026",
    },
}

E17_INFO = {
    "E17A": {
        "model": "粗细双头 + 连续风格 + 心率语义状态",
        "signal": "心率",
        "form": "语义状态",
        "source": "本轮 seed2026",
        "raw_pair": "E10A",
    },
    "E17B": {
        "model": "粗细双头 + 连续风格 + 皮电唤醒状态",
        "signal": "皮电",
        "form": "语义状态",
        "source": "本轮 seed2026",
        "raw_pair": "E10B",
    },
    "E17C": {
        "model": "粗细双头 + 连续风格 + 肌电控制紧张状态",
        "signal": "肌电",
        "form": "语义状态",
        "source": "本轮 seed2026",
        "raw_pair": "E10C",
    },
    "E17D": {
        "model": "粗细双头 + 连续风格 + 脑电语义状态",
        "signal": "脑电",
        "form": "语义状态",
        "source": "本轮 seed2026",
        "raw_pair": "E16B",
    },
}

ORDER = ["E2", "E10A", "E17A", "E10B", "E17B", "E10C", "E17C", "E16B", "E17D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总四类生理/脑电单信号语义状态 seed2026 初筛。")
    parser.add_argument("--e17-runs", default=str(DEFAULT_E17_RUNS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def configure_font() -> None:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    ]
    font_path = next((p for p in candidates if p.exists()), None)
    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def _localize_path(path_text: str) -> Path:
    raw = str(path_text).strip()
    if not raw:
        return Path(raw)
    raw_norm = raw.replace("\\", "/")
    remote_norm = str(REMOTE_ROOT).replace("\\", "/")
    if raw_norm.startswith(remote_norm):
        rel = raw_norm[len(remote_norm) :].lstrip("/")
        return PROJECT_ROOT / rel
    return Path(raw)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metrics_from_run_root(run_root: Path) -> dict[str, Any] | None:
    metrics_path = run_root / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = _load_json(metrics_path)
    test = metrics["test"]
    selection = test["selection_summary"]
    return {
        "test_rmse": float(test["steer_rmse"]),
        "primary_rmse": float(selection["overall_primary_steer_rmse"]),
        "tail_rmse": float(selection["rmse_tail_abs_steer"]),
        "peak_err_s": float(selection["peak_time_abs_err_s"]),
        "selection": float(selection["selection_score"]),
        "prediction_overview": str(run_root / "prediction_figures" / "test" / "overview.png"),
        "sample_metrics_csv": str(run_root / "prediction_figures" / "test" / "prediction_sample_metrics.csv"),
    }


def _read_records(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "smoke_test" in df.columns:
        smoke = df["smoke_test"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
        df = df[~smoke].copy()
    return df


def _rows_from_run_records(path: Path, info_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    df = _read_records(path)
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows
    df = df[df["seed"].astype(int).eq(2026)].copy()
    for version, info in info_map.items():
        match = df[df["experiment_id"].astype(str).eq(version)]
        if match.empty:
            continue
        row = match.iloc[-1]
        run_root = _localize_path(str(row["run_root"]))
        metrics = _metrics_from_run_root(run_root)
        if metrics is None:
            continue
        rows.append(
            {
                "version": version,
                "seed": 2026,
                "model": info["model"],
                "signal": info["signal"],
                "form": info["form"],
                "source": info["source"],
                "run_root": str(run_root),
                **metrics,
            }
        )
    return rows


def _reference_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in REFERENCE_RUN_FILES:
        rows.extend(_rows_from_run_records(path, REFERENCE_INFO))
    deduped = {row["version"]: row for row in rows}
    return list(deduped.values())


def _e17_rows(path: Path) -> list[dict[str, Any]]:
    rows = _rows_from_run_records(path, E17_INFO)
    found = {row["version"] for row in rows}
    for version, info in E17_INFO.items():
        if version in found:
            continue
        candidates = sorted((PROJECT_ROOT / "tmp" / "event_conditioned_runs").glob(f"{version}_*seed2026_*"))
        for run_root in reversed(candidates):
            metrics = _metrics_from_run_root(run_root)
            if metrics is None:
                continue
            rows.append(
                {
                    "version": version,
                    "seed": 2026,
                    "model": info["model"],
                    "signal": info["signal"],
                    "form": info["form"],
                    "source": info["source"],
                    "run_root": str(run_root),
                    **metrics,
                }
            )
            break
    return rows


def _format(value: float) -> str:
    return f"{float(value):.4f}"


def _build_summary(e17_runs: Path) -> pd.DataFrame:
    rows = _reference_rows() + _e17_rows(e17_runs)
    if not rows:
        raise RuntimeError("没有找到可汇总的 E17 结果。")
    df = pd.DataFrame(rows).drop_duplicates(subset=["version"], keep="last")
    e2_rmse = float(df.loc[df["version"].eq("E2"), "test_rmse"].iloc[0]) if "E2" in set(df["version"]) else math.nan
    raw_by_signal = {
        str(row["signal"]): float(row["test_rmse"])
        for _, row in df[df["form"].eq("原始数值")].iterrows()
    }
    deltas: list[float] = []
    raw_deltas: list[float] = []
    for _, row in df.iterrows():
        deltas.append(float(row["test_rmse"]) - e2_rmse if math.isfinite(e2_rmse) else math.nan)
        if str(row["form"]) == "语义状态":
            raw_ref = raw_by_signal.get(str(row["signal"]), math.nan)
            raw_deltas.append(float(row["test_rmse"]) - raw_ref if math.isfinite(raw_ref) else math.nan)
        else:
            raw_deltas.append(math.nan)
    df["delta_vs_E2"] = deltas
    df["delta_vs_raw_same_signal"] = raw_deltas
    df["order"] = df["version"].map({version: idx for idx, version in enumerate(ORDER)})
    return df.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def _render_table(summary: pd.DataFrame, out_dir: Path) -> None:
    plot_df = summary.copy()
    for col in ["test_rmse", "primary_rmse", "tail_rmse", "selection", "delta_vs_E2", "delta_vs_raw_same_signal"]:
        plot_df[col] = plot_df[col].map(lambda x: "" if pd.isna(x) else _format(float(x)))
    plot_df = plot_df[
        [
            "version",
            "model",
            "signal",
            "form",
            "test_rmse",
            "primary_rmse",
            "tail_rmse",
            "selection",
            "delta_vs_E2",
            "delta_vs_raw_same_signal",
        ]
    ]
    plot_df.columns = ["版本", "模型", "信号", "形式", "test RMSE", "primary", "tail", "selection", "相对E2", "相对同信号原始"]

    fig, ax = plt.subplots(figsize=(19.5, 1.6 + len(plot_df) * 0.62), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.01),
            0.98,
            0.98,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            linewidth=1.0,
            edgecolor="#d6dbe1",
            facecolor="#ffffff",
            transform=ax.transAxes,
            zorder=-1,
        )
    )
    table = ax.table(
        cellText=plot_df.values.tolist(),
        colLabels=list(plot_df.columns),
        colWidths=[0.07, 0.30, 0.07, 0.09, 0.10, 0.10, 0.10, 0.10, 0.09, 0.12],
        loc="center",
        cellLoc="left",
        bbox=[0.02, 0.13, 0.96, 0.74],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    values = plot_df.values.tolist()
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dee5")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#eef1f5")
            cell.get_text().set_color("#20242a")
            cell.get_text().set_weight("bold")
        else:
            version = str(values[row - 1][0])
            if version in {"E10C"}:
                face = "#e8f5ec"
            elif version.startswith("E17"):
                face = "#fff7e6"
            elif version in {"E10B", "E16B"}:
                face = "#fdecec"
            else:
                face = "#ffffff" if row % 2 else "#f8fafc"
            cell.set_facecolor(face)
            cell.get_text().set_color("#222831")
        cell._loc = "center" if col >= 2 else "left"
    ax.text(
        0.025,
        0.95,
        "四类生理/脑电单信号语义状态 seed2026 初筛",
        color="#111827",
        fontsize=15,
        fontweight="bold",
        transform=ax.transAxes,
        va="center",
    )
    ax.text(
        0.025,
        0.052,
        "说明：E17 是本轮语义状态；E10/E16B 是同信号原始数值输入；相对E2为负数表示优于连续风格基准。",
        color="#4b5563",
        fontsize=9.0,
        transform=ax.transAxes,
        va="center",
    )
    fig.savefig(out_dir / "e17_semantic_single_signal_seed2026_table.png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> str:
    display = df.fillna("").astype(str)
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.values.tolist():
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def _write_report(summary: pd.DataFrame, out_dir: Path) -> None:
    display = summary[
        [
            "version",
            "model",
            "signal",
            "form",
            "test_rmse",
            "primary_rmse",
            "tail_rmse",
            "selection",
            "delta_vs_E2",
            "delta_vs_raw_same_signal",
        ]
    ].copy()
    for col in ["test_rmse", "primary_rmse", "tail_rmse", "selection", "delta_vs_E2", "delta_vs_raw_same_signal"]:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else _format(float(x)))
    lines = [
        "# 四类生理/脑电单信号语义状态 seed2026 初筛报告",
        "",
        "## 目的",
        "",
        "本轮只跑 seed2026，用来快速判断心率、皮电、肌电、脑电从原始数值输入改成语义状态输入后，是否有值得继续三种子复验的迹象。",
        "",
        "## 结果表",
        "",
        _markdown_table(display),
        "",
        "## 注意",
        "",
        "- 这只是单种子初筛，不作为最终稳定结论。",
        "- E17D 与旧 E7A 属于同一类“脑电语义状态 + 连续风格”，但本轮重新在同一批对照里跑 seed2026。",
        "- 判断是否继续补种子时，要同时看整体误差、尾段误差、selection 和预测图。",
        "",
        "## 产物",
        "",
        "- `e17_semantic_single_signal_seed2026_summary.csv`",
        "- `e17_semantic_single_signal_seed2026_table.png`",
        "- `e17_prediction_figure_index.csv`",
    ]
    (out_dir / "e17_semantic_single_signal_seed2026_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_font()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _build_summary(Path(args.e17_runs))
    summary.to_csv(out_dir / "e17_semantic_single_signal_seed2026_summary.csv", index=False, encoding="utf-8-sig")
    pred_cols = ["version", "seed", "run_root", "prediction_overview", "sample_metrics_csv"]
    summary[pred_cols].to_csv(out_dir / "e17_prediction_figure_index.csv", index=False, encoding="utf-8-sig")
    _render_table(summary, out_dir)
    _write_report(summary, out_dir)
    print(f"summary_dir={out_dir}")
    print(f"summary_csv={out_dir / 'e17_semantic_single_signal_seed2026_summary.csv'}")
    print(f"table={out_dir / 'e17_semantic_single_signal_seed2026_table.png'}")


if __name__ == "__main__":
    main()
