# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REMOTE_ROOT = Path("/root/autodl-tmp/data_process")
REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"

DEFAULT_RUN_CSVS = [
    REPORTS_DIR / "style_physio_eeg_e15_hr_no_style_runs_20260511.csv",
    REPORTS_DIR / "style_physio_eeg_e15_eda_no_style_runs_20260511.csv",
    REPORTS_DIR / "style_physio_eeg_e15_emg_no_style_runs_20260511.csv",
    REPORTS_DIR / "style_physio_eeg_e15_single_signal_no_style_runs_20260511.csv",
    REPORTS_DIR / "style_physio_eeg_e16_eeg_no_style_runs_20260511.csv",
    REPORTS_DIR / "style_physio_eeg_e16_eeg_style_runs_20260511.csv",
    REPORTS_DIR / "style_physio_eeg_e16_eeg_single_signal_runs_20260511.csv",
]
DEFAULT_OUT_DIR = REPORTS_DIR / "style_physio_eeg_e15_e16_single_signal_summary_20260511"
VERSION_LOG = REPORTS_DIR / "current_model_version_result_log_20260509.csv"

METRIC_COLUMNS = ["test_rmse", "primary_rmse", "tail_rmse", "peak_err_s", "selection"]

VERSION_INFO: dict[str, dict[str, Any]] = {
    "E1": {
        "model": "粗细双头，不加连续风格，不加生理/脑电",
        "style": False,
        "signal": "none",
        "source": "已有三种子结果",
    },
    "E2": {
        "model": "粗细双头 + 连续风格，不加生理/脑电",
        "style": True,
        "signal": "none",
        "source": "已有三种子结果",
    },
    "E10A": {
        "model": "粗细双头 + 连续风格 + 心率单信号",
        "style": True,
        "signal": "hr",
        "source": "已有三种子结果",
    },
    "E10B": {
        "model": "粗细双头 + 连续风格 + 皮电单信号",
        "style": True,
        "signal": "eda",
        "source": "已有三种子结果",
    },
    "E10C": {
        "model": "粗细双头 + 连续风格 + 肌电单信号",
        "style": True,
        "signal": "emg",
        "source": "已有三种子结果",
    },
    "E15A": {
        "model": "粗细双头 + 心率单信号，不加连续风格",
        "style": False,
        "signal": "hr",
        "source": "本轮新跑三种子",
    },
    "E15B": {
        "model": "粗细双头 + 皮电单信号，不加连续风格",
        "style": False,
        "signal": "eda",
        "source": "本轮新跑三种子",
    },
    "E15C": {
        "model": "粗细双头 + 肌电单信号，不加连续风格",
        "style": False,
        "signal": "emg",
        "source": "本轮新跑三种子",
    },
    "E16A": {
        "model": "粗细双头 + 脑电单信号，不加连续风格",
        "style": False,
        "signal": "eeg",
        "source": "本轮新跑三种子",
    },
    "E16B": {
        "model": "粗细双头 + 脑电单信号 + 连续风格",
        "style": True,
        "signal": "eeg",
        "source": "本轮新跑三种子",
    },
}

SIGNAL_NAMES = {
    "none": "无生理/脑电",
    "hr": "心率",
    "eda": "皮电",
    "emg": "肌电",
    "eeg": "脑电",
}

ORDER = ["E1", "E15A", "E15B", "E15C", "E16A", "E2", "E10A", "E10B", "E10C", "E16B"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总生理/脑电单信号在有无连续风格下的三种子结果。")
    parser.add_argument("--run-csvs", nargs="+", default=[str(p) for p in DEFAULT_RUN_CSVS])
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


def _parse_mean_std(text: str) -> tuple[float, float | None]:
    raw = str(text).strip()
    if "±" in raw:
        mean, std = raw.split("±", 1)
        return float(mean), float(std)
    return float(raw), None


def _format_mean_std(mean: float, std: float | None) -> str:
    if std is None or math.isnan(float(std)):
        return f"{mean:.4f}"
    return f"{mean:.4f}±{std:.4f}"


def _read_version_log() -> pd.DataFrame:
    return pd.read_csv(VERSION_LOG, dtype=str)


def _baseline_summary(version: str) -> dict[str, Any]:
    log = _read_version_log()
    row = log[log["version"].eq(version)].iloc[0].to_dict()
    info = VERSION_INFO[version]
    parsed = {col: _parse_mean_std(row[col]) for col in METRIC_COLUMNS}
    return {
        "version": version,
        "model": info["model"],
        "continuous_style": "有" if info["style"] else "无",
        "signal": SIGNAL_NAMES[info["signal"]],
        "source": info["source"],
        "n_seeds": int(row["n_seeds"]),
        **{col: _format_mean_std(*parsed[col]) for col in METRIC_COLUMNS},
        **{f"{col}_mean": parsed[col][0] for col in METRIC_COLUMNS},
        **{f"{col}_std": parsed[col][1] for col in METRIC_COLUMNS},
    }


def _metric_row(row: pd.Series) -> dict[str, Any] | None:
    version = str(row["experiment_id"])
    if version not in VERSION_INFO:
        return None
    run_root = _localize_path(str(row["run_root"]))
    metrics_path = run_root / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = _load_json(metrics_path)
    test = metrics["test"]
    selection = test["selection_summary"]
    info = VERSION_INFO[version]
    return {
        "version": version,
        "model": info["model"],
        "seed": int(row["seed"]),
        "continuous_style": "有" if info["style"] else "无",
        "signal": SIGNAL_NAMES[info["signal"]],
        "run_root": str(run_root),
        "test_rmse": float(test["steer_rmse"]),
        "primary_rmse": float(selection["overall_primary_steer_rmse"]),
        "tail_rmse": float(selection["rmse_tail_abs_steer"]),
        "peak_err_s": float(selection["peak_time_abs_err_s"]),
        "selection": float(selection["selection_score"]),
        "prediction_overview": str(run_root / "prediction_figures" / "test" / "overview.png"),
        "sample_metrics_csv": str(run_root / "prediction_figures" / "test" / "prediction_sample_metrics.csv"),
    }


def _read_new_runs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    runs = pd.concat(frames, ignore_index=True)
    if "smoke_test" in runs.columns:
        smoke = runs["smoke_test"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
        runs = runs[~smoke]
    runs = runs[runs["run_root"].fillna("").astype(str).str.len() > 0].copy()
    rows = []
    for _, row in runs.iterrows():
        item = _metric_row(row)
        if item is not None:
            rows.append(item)
    if not rows:
        return pd.DataFrame()
    seed_df = pd.DataFrame(rows)
    seed_df = seed_df.drop_duplicates(subset=["version", "seed"], keep="last")
    return seed_df


def _summarize_new_runs(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for version, group in seed_df.groupby("version"):
        info = VERSION_INFO[version]
        group = group.sort_values("seed")
        item: dict[str, Any] = {
            "version": version,
            "model": info["model"],
            "continuous_style": "有" if info["style"] else "无",
            "signal": SIGNAL_NAMES[info["signal"]],
            "source": info["source"],
            "n_seeds": int(group["seed"].nunique()),
        }
        for col in METRIC_COLUMNS:
            values = pd.to_numeric(group[col], errors="coerce").dropna().to_numpy(dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            item[f"{col}_mean"] = mean
            item[f"{col}_std"] = std
            item[col] = _format_mean_std(mean, std)
        rows.append(item)
    return pd.DataFrame(rows)


def _build_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    base_versions = ["E1", "E2", "E10A", "E10B", "E10C"]
    base_rows = [_baseline_summary(version) for version in base_versions]
    new_summary = _summarize_new_runs(seed_df)
    summary = pd.concat([pd.DataFrame(base_rows), new_summary], ignore_index=True)
    summary = summary.drop_duplicates(subset=["version"], keep="last")
    summary["order"] = summary["version"].map({version: idx for idx, version in enumerate(ORDER)})
    return summary.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def _build_two_by_five(summary: pd.DataFrame) -> pd.DataFrame:
    row_specs = [
        ("粗细双头，不加连续风格", {"none": "E1", "hr": "E15A", "eda": "E15B", "emg": "E15C", "eeg": "E16A"}),
        ("粗细双头，加连续风格", {"none": "E2", "hr": "E10A", "eda": "E10B", "emg": "E10C", "eeg": "E16B"}),
    ]
    rows = []
    for structure, mapping in row_specs:
        row: dict[str, Any] = {"结构": structure}
        for key in ["none", "hr", "eda", "emg", "eeg"]:
            version = mapping[key]
            match = summary[summary["version"].eq(version)]
            row[SIGNAL_NAMES[key]] = match["test_rmse"].iloc[0] if not match.empty else ""
        rows.append(row)
    return pd.DataFrame(rows)


def _cell_face(version: str, row_index: int) -> str:
    if version in {"E15A", "E15B", "E15C", "E16A"}:
        return "#fff7e6"
    if version == "E10C":
        return "#e8f5ec"
    if version in {"E10B", "E16B"}:
        return "#fdecec"
    return "#ffffff" if row_index % 2 else "#f8fafc"


def _render_main_table(summary: pd.DataFrame, out_dir: Path) -> None:
    plot_df = summary[
        [
            "version",
            "model",
            "continuous_style",
            "signal",
            "source",
            "test_rmse",
            "primary_rmse",
            "tail_rmse",
            "selection",
        ]
    ].copy()
    plot_df.columns = [
        "版本",
        "模型",
        "连续风格",
        "生理/脑电",
        "结果来源",
        "test steer RMSE",
        "primary RMSE",
        "tail RMSE",
        "selection",
    ]

    fig_w = 20.5
    fig_h = 1.55 + len(plot_df) * 0.72
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.axis("off")
    bg = FancyBboxPatch(
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
    ax.add_patch(bg)
    table = ax.table(
        cellText=plot_df.values.tolist(),
        colLabels=list(plot_df.columns),
        colWidths=[0.065, 0.315, 0.075, 0.095, 0.12, 0.12, 0.12, 0.11, 0.10],
        loc="center",
        cellLoc="left",
        bbox=[0.02, 0.12, 0.96, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    values = plot_df.values.tolist()
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dee5")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#eef1f5")
            cell.get_text().set_color("#20242a")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(9.3)
        else:
            version = str(values[row - 1][0])
            cell.set_facecolor(_cell_face(version, row))
            cell.get_text().set_color("#222831")
            cell.get_text().set_fontsize(8.6)
        cell._loc = "center" if col >= 2 else "left"

    ax.text(
        0.025,
        0.955,
        "生理/脑电单信号独立贡献验证：无连续风格 vs 有连续风格",
        color="#111827",
        fontsize=15,
        fontweight="bold",
        transform=ax.transAxes,
        va="center",
    )
    ax.text(
        0.025,
        0.052,
        "说明：上半部分回答单信号本身是否独立有效；下半部分回答在连续风格基础上是否还有额外增益。数值越小越好。",
        color="#4b5563",
        fontsize=9.2,
        transform=ax.transAxes,
        va="center",
    )
    fig.savefig(out_dir / "single_signal_independent_ablation_table.png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _render_two_by_five(two_by_five: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.6, 3.2), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.axis("off")
    table = ax.table(
        cellText=two_by_five.values.tolist(),
        colLabels=list(two_by_five.columns),
        colWidths=[0.28, 0.15, 0.15, 0.15, 0.15, 0.15],
        cellLoc="center",
        bbox=[0.02, 0.18, 0.96, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dee5")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#eef1f5")
            cell.get_text().set_color("#20242a")
            cell.get_text().set_weight("bold")
        elif row == 1:
            cell.set_facecolor("#fff7e6" if col > 0 else "#ffffff")
        else:
            if col == 4:
                cell.set_facecolor("#e8f5ec")
            elif col in {3, 5}:
                cell.set_facecolor("#fdecec")
            else:
                cell.set_facecolor("#f8fafc")
        if row > 0:
            cell.get_text().set_color("#222831")

    ax.text(
        0.02,
        0.91,
        "2×5 对照：单信号是否独立有效",
        fontsize=15,
        fontweight="bold",
        color="#111827",
        transform=ax.transAxes,
    )
    ax.text(
        0.02,
        0.06,
        "第一行是不加连续风格，第二行是加连续风格；横向比较不同生理/脑电信号，纵向比较连续风格是否改变该信号的效果。",
        fontsize=9.5,
        color="#4b5563",
        transform=ax.transAxes,
    )
    fig.savefig(out_dir / "two_by_five_single_signal_table.png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> str:
    display = df.fillna("").astype(str)
    headers = list(display.columns)
    rows = display.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def _write_report(summary: pd.DataFrame, two_by_five: pd.DataFrame, seed_df: pd.DataFrame, out_dir: Path) -> None:
    e1 = float(summary.loc[summary["version"].eq("E1"), "test_rmse_mean"].iloc[0])
    e2 = float(summary.loc[summary["version"].eq("E2"), "test_rmse_mean"].iloc[0])
    lines = [
        "# 生理/脑电单信号独立贡献验证报告",
        "",
        "## 实验目的",
        "",
        "本轮补充了不加连续风格的心率、皮电、肌电三组，以及原始脑电单信号在有无连续风格下的两组。目的不是马上提出新主线，而是把“信号本身有没有贡献”和“信号是否必须依赖连续驾驶风格才能体现价值”分开验证。",
        "",
        "## 对照逻辑",
        "",
        "- E15A/E15B/E15C/E16A 与 E1 比：判断单信号在不加连续风格时是否独立有效。",
        "- E10A/E10B/E10C/E16B 与 E2 比：判断单信号在连续风格基础上是否还有额外增益。",
        "- E16A/E16B 使用的是原始脑电单信号；旧 E7A 是脑电语义状态，不直接放进这张单信号表。",
        "",
        "## 2×5 汇总表",
        "",
        _markdown_table(two_by_five),
        "",
        "## 详细指标",
        "",
        _markdown_table(
            summary[
                [
                    "version",
                    "model",
                    "continuous_style",
                    "signal",
                    "source",
                    "n_seeds",
                    "test_rmse",
                    "primary_rmse",
                    "tail_rmse",
                    "selection",
                ]
            ]
        ),
        "",
        "## 与对应基准相比",
        "",
    ]
    for version in ["E15A", "E15B", "E15C", "E16A"]:
        if version not in set(summary["version"]):
            continue
        row = summary[summary["version"].eq(version)].iloc[0]
        mean = float(row["test_rmse_mean"])
        delta = mean - e1
        direction = "低于" if delta < 0 else "高于"
        lines.append(f"- {version} 相对 E1 的 test RMSE {direction} {abs(delta):.4f}。")
    for version in ["E10A", "E10B", "E10C", "E16B"]:
        if version not in set(summary["version"]):
            continue
        row = summary[summary["version"].eq(version)].iloc[0]
        mean = float(row["test_rmse_mean"])
        delta = mean - e2
        direction = "低于" if delta < 0 else "高于"
        lines.append(f"- {version} 相对 E2 的 test RMSE {direction} {abs(delta):.4f}。")
    lines.extend(
        [
            "",
            "## 当前可以怎么解释",
            "",
            "第一，不加连续风格时，心率、皮电、肌电、原始脑电单独接入都没有超过 E1，说明这些单信号本身不能直接替代连续驾驶风格，也不能证明“只靠生理/脑电就能稳定预测”。",
            "",
            "第二，加上连续风格后，肌电单信号 E10C 是目前最清楚的正向结果，整体误差、主响应误差、尾段误差和综合选择指标都优于 E2；心率 E10A 只有很小的整体改善，皮电 E10B 和原始脑电 E16B 整体变差。",
            "",
            "第三，这个结果不等于“脑电没用”。它说明的是：原始脑电直接作为推理输入并不好；但前面 E5A 的脑电教师蒸馏仍然是有效证据。因此更合理的表述是：脑电更适合作为训练阶段教师或经过更合理表征后使用，而不是当前这种原始脑电单信号直接拼接。",
            "",
            "第四，当前最稳妥的汇报说法是：连续驾驶风格是固定基础输入；肌电是当前非脑电生理信号里最有价值的推理期候选；心率和皮电在当前特征构造下证据弱；原始脑电直接输入不是主线，但脑电教师路线仍保留价值。",
            "",
            "## 产物",
            "",
            "- `seed_wise_metrics.csv`：本轮新跑版本每个种子的指标。",
            "- `single_signal_independent_summary.csv`：E1/E2/E10/E15/E16 汇总表。",
            "- `two_by_five_single_signal_table.csv`：2×5 简表。",
            "- `single_signal_independent_ablation_table.png`：白底详细图表。",
            "- `two_by_five_single_signal_table.png`：白底 2×5 简表。",
            "- `prediction_figure_index.csv`：每个新跑版本的预测图路径索引。",
        ]
    )
    (out_dir / "single_signal_independent_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_font()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_df = _read_new_runs([Path(p) for p in args.run_csvs])
    if seed_df.empty:
        raise RuntimeError("没有找到 E15/E16 的有效运行结果。")
    seed_df = seed_df.sort_values(["version", "seed"]).reset_index(drop=True)
    seed_df.to_csv(out_dir / "seed_wise_metrics.csv", index=False, encoding="utf-8-sig")

    summary = _build_summary(seed_df)
    summary.to_csv(out_dir / "single_signal_independent_summary.csv", index=False, encoding="utf-8-sig")

    two_by_five = _build_two_by_five(summary)
    two_by_five.to_csv(out_dir / "two_by_five_single_signal_table.csv", index=False, encoding="utf-8-sig")

    pred_cols = ["version", "seed", "run_root", "prediction_overview", "sample_metrics_csv"]
    seed_df[pred_cols].to_csv(out_dir / "prediction_figure_index.csv", index=False, encoding="utf-8-sig")

    _render_main_table(summary, out_dir)
    _render_two_by_five(two_by_five, out_dir)
    _write_report(summary, two_by_five, seed_df, out_dir)

    print(f"summary_dir={out_dir}")
    print(f"summary_csv={out_dir / 'single_signal_independent_summary.csv'}")
    print(f"main_table={out_dir / 'single_signal_independent_ablation_table.png'}")
    print(f"two_by_five={out_dir / 'two_by_five_single_signal_table.png'}")


if __name__ == "__main__":
    main()
