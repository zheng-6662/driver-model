# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
DEFAULT_E10_RUNS = REPORTS_DIR / "style_physio_eeg_e10_non_eeg_signal_runs_20260509.csv"
DEFAULT_E0_E2_RUNS = REPORTS_DIR / "style_physio_eeg_e0_e2_fresh_3seed_runs_20260507.csv"
DEFAULT_E3_E4_RUNS = REPORTS_DIR / "style_physio_eeg_e3_e4_runs_20260507.csv"
DEFAULT_E7_RUNS = REPORTS_DIR / "style_physio_eeg_e7_signal_group_runs_20260508.csv"
DEFAULT_OUT_DIR = REPORTS_DIR / "style_physio_eeg_e10_non_eeg_signal_summary_20260509"

EXPERIMENT_NAMES = {
    "E2": "粗细双头 + 连续驾驶风格",
    "E3": "无 EEG 语义生理状态 + 连续驾驶风格",
    "E7C": "raw HR+EDA+EMG 融合 + 连续驾驶风格",
    "E10A": "HR-only + 连续驾驶风格",
    "E10B": "EDA-only + 连续驾驶风格",
    "E10C": "EMG-only + 连续驾驶风格",
}
EXPERIMENT_ORDER = ["E2", "E3", "E7C", "E10A", "E10B", "E10C"]
METRIC_COLUMNS = ["test_steer_rmse", "primary_rmse", "tail_rmse", "peak_err_s", "selection"]
SAMPLE_DELTA_COLUMNS = ["rmse_2s_abs_steer", "rmse_pre_tail_abs_steer", "rmse_tail_abs_steer", "peak_time_abs_err_s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize E10 non-EEG physiological signal attribution runs.")
    parser.add_argument("--e10-runs-csv", default=str(DEFAULT_E10_RUNS))
    parser.add_argument("--e0-e2-runs-csv", default=str(DEFAULT_E0_E2_RUNS))
    parser.add_argument("--e3-e4-runs-csv", default=str(DEFAULT_E3_E4_RUNS))
    parser.add_argument("--e7-runs-csv", default=str(DEFAULT_E7_RUNS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_runs(path: Path, keep_ids: set[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "experiment_id" in df.columns:
        df = df[df["experiment_id"].astype(str).isin(keep_ids)]
    if "smoke_test" in df.columns:
        is_smoke = df["smoke_test"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
        df = df[~is_smoke]
    if "run_root" in df.columns:
        df = df[df["run_root"].fillna("").astype(str).str.len() > 0]
    return df.copy()


def _sort_key(experiment_id: str) -> tuple[int, str]:
    if experiment_id in EXPERIMENT_ORDER:
        return EXPERIMENT_ORDER.index(experiment_id), experiment_id
    return len(EXPERIMENT_ORDER), experiment_id


def _metric_row(row: pd.Series) -> dict[str, Any] | None:
    run_root = Path(str(row["run_root"]))
    metrics_path = run_root / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = _load_json(metrics_path)
    test = metrics["test"]
    selection = test["selection_summary"]
    experiment_id = str(row["experiment_id"])
    return {
        "experiment_id": experiment_id,
        "experiment_name": EXPERIMENT_NAMES.get(experiment_id, str(row.get("label", experiment_id))),
        "seed": int(row["seed"]),
        "run_root": str(run_root),
        "test_steer_rmse": float(test["steer_rmse"]),
        "primary_rmse": float(selection["overall_primary_steer_rmse"]),
        "tail_rmse": float(selection["rmse_tail_abs_steer"]),
        "peak_err_s": float(selection["peak_time_abs_err_s"]),
        "selection": float(selection["selection_score"]),
        "metrics_json": str(metrics_path),
        "sample_metrics_csv": str(run_root / "prediction_figures" / "test" / "prediction_sample_metrics.csv"),
        "prediction_overview": str(run_root / "prediction_figures" / "test" / "overview.png"),
    }


def _load_sample_metrics(run_root: Path) -> pd.DataFrame:
    path = run_root / "prediction_figures" / "test" / "prediction_sample_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing sample metrics: {path}")
    return pd.read_csv(path)


def _paired_deltas(metrics_df: pd.DataFrame, pairs: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        left_id, right_id = pair.split(":", 1)
        left_runs = metrics_df[metrics_df["experiment_id"].astype(str).eq(left_id)]
        right_runs = metrics_df[metrics_df["experiment_id"].astype(str).eq(right_id)]
        shared_seeds = sorted(set(left_runs["seed"].astype(int)).intersection(set(right_runs["seed"].astype(int))))
        for seed in shared_seeds:
            left_root = Path(str(left_runs[left_runs["seed"].astype(int).eq(seed)].iloc[0]["run_root"]))
            right_root = Path(str(right_runs[right_runs["seed"].astype(int).eq(seed)].iloc[0]["run_root"]))
            left_df = _load_sample_metrics(left_root)
            right_df = _load_sample_metrics(right_root)
            merged = left_df.merge(right_df, on="sample_key", suffixes=(f"_{left_id}", f"_{right_id}"))
            item: dict[str, Any] = {"pair": pair, "left": left_id, "right": right_id, "seed": int(seed), "n_samples": int(len(merged))}
            for col in SAMPLE_DELTA_COLUMNS:
                left_col = f"{col}_{left_id}"
                right_col = f"{col}_{right_id}"
                if left_col not in merged.columns or right_col not in merged.columns:
                    continue
                delta = merged[left_col] - merged[right_col]
                item[f"delta_{col}_mean"] = float(delta.mean())
                item[f"delta_{col}_median"] = float(delta.median())
                item[f"delta_{col}_improved_rate"] = float((delta < 0).mean())
            rows.append(item)
    return pd.DataFrame(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "暂无结果。"
    view = df[columns].copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(_fmt(value))
            else:
                vals.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_report(out_dir: Path, metrics_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    physical_path = out_dir / "physical_direction_amplitude_audit" / "physical_direction_amplitude_summary.csv"
    physical_df = pd.read_csv(physical_path) if physical_path.exists() else pd.DataFrame()
    physical_mean = physical_df[physical_df["seed"].astype(str).eq("mean")].copy() if not physical_df.empty else pd.DataFrame()
    if physical_mean.empty and not physical_df.empty:
        physical_mean = physical_df.copy()

    lines: list[str] = [
        "# E10 非 EEG 生理单信号归因报告",
        "",
        "## 目的",
        "",
        "本轮只跑 seed-2026 门槛实验，用来判断 HR、EDA、EMG 单独输入是否有继续价值。它不是最终论文结论，也不用于直接替代 E5A/E6。",
        "",
        "## 版本含义",
        "",
        "- E2：不加生理信号的粗细双头 + 连续驾驶风格基准。",
        "- E3：无 EEG 语义生理状态 + 连续驾驶风格。",
        "- E7C：HR+EDA+EMG raw 融合 + 连续驾驶风格，已有融合对照。",
        "- E10A：HR-only + 连续驾驶风格。",
        "- E10B：EDA-only + 连续驾驶风格。",
        "- E10C：EMG-only + 连续驾驶风格。",
        "",
        "## seed-2026 指标",
        "",
        _markdown_table(metrics_df, ["experiment_id", "experiment_name", "seed", *METRIC_COLUMNS]),
        "",
        "## 配对样本差值",
        "",
        "负数表示左侧版本误差更低。重点看 E10A/B/C 相对 E2 和 E7C。",
        "",
        _markdown_table(delta_df, ["pair", "seed", "n_samples", "delta_rmse_2s_abs_steer_mean", "delta_rmse_2s_abs_steer_improved_rate", "delta_rmse_tail_abs_steer_mean"]),
    ]
    if not physical_mean.empty:
        cols = [
            "experiment_id",
            "seed",
            "median_amp_ratio_major",
            "under_amp_rate_major",
            "severe_under_amp_rate_large",
            "peak_side_wrong_at_true_peak_rate_major",
            "opposite_side_heavy_rate_major",
        ]
        cols = [col for col in cols if col in physical_mean.columns]
        lines.extend([
            "",
            "## 物理幅值/方向审计",
            "",
            _markdown_table(physical_mean, cols),
        ])
    lines.extend([
        "",
        "## 初步判断规则",
        "",
        "- 如果某个单信号明显弱于 E2，并且没有优于 E7C，则不补 2027/2028。",
        "- 如果某个单信号接近或优于 E2，并且比 E7C 更稳，则值得补 2027/2028。",
        "- 如果单信号优于 E7C 但仍弱于 E2，说明融合可能有干扰，但该信号是否进入主模型仍需谨慎。",
        "- 如果三个单信号都弱于 E2，则当前 raw 非 EEG 生理路线应先转为可靠性/事件相关性分析。",
    ])
    (out_dir / "teacher_report_e10_seed2026_20260509.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = {int(seed) for seed in args.seeds}

    runs = pd.concat(
        [
            _read_runs(Path(args.e0_e2_runs_csv), {"E2"}),
            _read_runs(Path(args.e3_e4_runs_csv), {"E3"}),
            _read_runs(Path(args.e7_runs_csv), {"E7C"}),
            _read_runs(Path(args.e10_runs_csv), {"E10A", "E10B", "E10C"}),
        ],
        ignore_index=True,
    )
    if runs.empty:
        raise RuntimeError("No runs found for E10 summary.")
    runs = runs[runs["seed"].astype(int).isin(seeds)].copy()
    runs = runs.sort_values(by=["experiment_id", "seed"], key=lambda s: s.map(lambda x: _sort_key(str(x)) if s.name == "experiment_id" else x))

    metric_rows = []
    for _, row in runs.iterrows():
        item = _metric_row(row)
        if item is not None:
            metric_rows.append(item)
    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        raise RuntimeError("No metric rows could be built.")
    metrics_df = metrics_df.sort_values(["experiment_id", "seed"], key=lambda s: s.map(lambda x: _sort_key(str(x)) if s.name == "experiment_id" else x))
    metrics_df.to_csv(out_dir / "seed_wise_metrics.csv", index=False, encoding="utf-8-sig")

    prediction_index = metrics_df[["experiment_id", "experiment_name", "seed", "run_root", "prediction_overview", "sample_metrics_csv"]].copy()
    prediction_index.to_csv(out_dir / "prediction_figure_index.csv", index=False, encoding="utf-8-sig")

    pairs = []
    for exp_id in ["E10A", "E10B", "E10C"]:
        pairs.append(f"{exp_id}:E2")
        pairs.append(f"{exp_id}:E7C")
        pairs.append(f"{exp_id}:E3")
    delta_df = _paired_deltas(metrics_df, pairs)
    delta_df.to_csv(out_dir / "paired_sample_deltas.csv", index=False, encoding="utf-8-sig")

    _write_report(out_dir, metrics_df, delta_df)

    artifact_lines = [
        "# E10 产物索引",
        "",
        "- `teacher_report_e10_seed2026_20260509.md`",
        "- `seed_wise_metrics.csv`",
        "- `paired_sample_deltas.csv`",
        "- `prediction_figure_index.csv`",
        "- `physical_direction_amplitude_audit/physical_direction_amplitude_report_cn.md`",
        "- `comparison_plots/comparison_overview_seed2026.png`",
    ]
    (out_dir / "artifact_index_e10_20260509.md").write_text("\n".join(artifact_lines) + "\n", encoding="utf-8")

    print(f"summary_dir={out_dir}")
    print(f"metrics={out_dir / 'seed_wise_metrics.csv'}")
    print(f"prediction_index={out_dir / 'prediction_figure_index.csv'}")
    print(f"report={out_dir / 'teacher_report_e10_seed2026_20260509.md'}")


if __name__ == "__main__":
    main()
