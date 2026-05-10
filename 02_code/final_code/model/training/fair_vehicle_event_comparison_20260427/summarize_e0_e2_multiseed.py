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
DEFAULT_OUT_DIR = PROJECT_ROOT / "04_project_logs" / "reports" / "style_physio_eeg_e0_e2_summary"

METRIC_COLUMNS = [
    "test_steer_rmse",
    "primary_rmse",
    "tail_rmse",
    "peak_err_s",
    "tail_direction",
    "selection",
]

SAMPLE_DELTA_COLUMNS = [
    "rmse_2s_abs_steer",
    "rmse_pre_tail_abs_steer",
    "rmse_tail_abs_steer",
    "peak_time_abs_err_s",
    "turning_count_abs_err",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize E0/E1/E2 multi-seed runs without launching training.")
    parser.add_argument("--runs-csv", required=True, help="CSV with experiment_id, seed, run_root columns.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=["E2:E1", "E0:E2"],
        help="Paired comparisons as LEFT:RIGHT. Negative deltas mean LEFT has lower error.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260507)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_row(row: pd.Series) -> dict[str, Any]:
    run_root = Path(str(row["run_root"]))
    metrics = _load_json(run_root / "metrics.json")
    test = metrics["test"]
    selection = test["selection_summary"]
    return {
        "experiment_id": str(row["experiment_id"]),
        "seed": int(row["seed"]),
        "run_root": str(run_root),
        "test_steer_rmse": float(test["steer_rmse"]),
        "primary_rmse": float(selection["overall_primary_steer_rmse"]),
        "tail_rmse": float(selection["rmse_tail_abs_steer"]),
        "peak_err_s": float(selection["peak_time_abs_err_s"]),
        "tail_direction": float(selection["tail_direction_match"]),
        "selection": float(selection["selection_score"]),
        "metrics_json": str(run_root / "metrics.json"),
        "sample_metrics_csv": str(run_root / "prediction_figures" / "test" / "prediction_sample_metrics.csv"),
    }


def _summarize_mean_std(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for experiment_id, group in metrics_df.groupby("experiment_id", sort=True):
        item: dict[str, Any] = {"experiment_id": experiment_id, "n_seeds": int(len(group))}
        for col in METRIC_COLUMNS:
            item[f"{col}_mean"] = float(group[col].mean())
            item[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else float("nan")
        rows.append(item)
    return pd.DataFrame(rows)


def _load_sample_metrics(run_root: Path) -> pd.DataFrame:
    path = run_root / "prediction_figures" / "test" / "prediction_sample_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing sample metrics: {path}")
    return pd.read_csv(path)


def _stable_offset(text: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text))


def _bootstrap_mean_ci(values: pd.Series, n_bootstrap: int, seed: int) -> tuple[float, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) == 0 or n_bootstrap <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    means = arr[sample_idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _paired_delta_rows(
    runs_df: pd.DataFrame,
    pair_spec: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    left_id, right_id = pair_spec.split(":", 1)
    out: list[dict[str, Any]] = []
    left_runs = runs_df[runs_df["experiment_id"].astype(str).eq(left_id)]
    right_runs = runs_df[runs_df["experiment_id"].astype(str).eq(right_id)]
    for seed in sorted(set(left_runs["seed"].astype(int)).intersection(set(right_runs["seed"].astype(int)))):
        left_root = Path(str(left_runs[left_runs["seed"].astype(int).eq(seed)].iloc[0]["run_root"]))
        right_root = Path(str(right_runs[right_runs["seed"].astype(int).eq(seed)].iloc[0]["run_root"]))
        left_df = _load_sample_metrics(left_root)
        right_df = _load_sample_metrics(right_root)
        merged = left_df.merge(right_df, on="sample_key", suffixes=(f"_{left_id}", f"_{right_id}"))
        item: dict[str, Any] = {
            "pair": pair_spec,
            "left": left_id,
            "right": right_id,
            "seed": int(seed),
            "n_samples": int(len(merged)),
        }
        for col in SAMPLE_DELTA_COLUMNS:
            left_col = f"{col}_{left_id}"
            right_col = f"{col}_{right_id}"
            if left_col in merged.columns and right_col in merged.columns:
                delta = merged[left_col] - merged[right_col]
                item[f"delta_{col}_mean"] = float(delta.mean())
                item[f"delta_{col}_median"] = float(delta.median())
                item[f"delta_{col}_improved_rate"] = float((delta < 0).mean())
                ci_seed = bootstrap_seed + int(seed) + _stable_offset(pair_spec) + _stable_offset(col)
                ci_low, ci_high = _bootstrap_mean_ci(delta, bootstrap_samples, ci_seed)
                item[f"delta_{col}_mean_ci95_low"] = ci_low
                item[f"delta_{col}_mean_ci95_high"] = ci_high
        out.append(item)
    return out


def _df_to_markdown(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    view = df if columns is None else df[columns]
    if view.empty:
        return "No rows."
    rows: list[list[str]] = []
    for _, row in view.iterrows():
        rows.append(["" if pd.isna(value) else str(value) for value in row.tolist()])
    headers = [str(col) for col in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _write_markdown(out_dir: Path, metrics_df: pd.DataFrame, summary_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    lines: list[str] = [
        "# E0-E2 Multi-Seed Summary",
        "",
        "说明：这个汇总脚本只读取已有 run 结果，不启动训练。",
        "",
        "## Seed-wise Metrics",
        "",
        _df_to_markdown(metrics_df, ["experiment_id", "seed", *METRIC_COLUMNS]),
        "",
        "## Mean / Std",
        "",
        _df_to_markdown(summary_df),
        "",
        "## Paired Per-Sample Deltas",
        "",
        "负数表示左侧模型误差更低；`improved_rate` 表示左侧模型在多少测试样本上更好。",
        "`mean_ci95_low/high` 是对测试事件成对差值均值做 bootstrap 得到的 95% 置信区间。",
        "",
    ]
    if delta_df.empty:
        lines.append("No paired comparisons were available.")
    else:
        lines.append(_df_to_markdown(delta_df))
    lines.extend(
        [
            "",
            "## Interpretation Template",
            "",
            "- E2 vs E1: 判断连续驾驶风格在 coarse-fine 结构上是否稳定有效。",
            "- E0 vs E2: 判断结构化主线是否在 raw RMSE guardrail 下仍然可接受。",
            "- 只有一个 seed 时只能作为 smoke-check，不能作为最终稳定性结论。",
        ]
    )
    (out_dir / "summary_cn.md").write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _metric_sentence(metrics_df: pd.DataFrame, experiment_id: str) -> str:
    rows = metrics_df[metrics_df["experiment_id"].astype(str).eq(experiment_id)]
    if rows.empty:
        return f"- {experiment_id}: 暂无结果。"
    if len(rows) == 1:
        row = rows.iloc[0]
        return (
            f"- {experiment_id}: test RMSE={_fmt(row['test_steer_rmse'])}, "
            f"primary RMSE={_fmt(row['primary_rmse'])}, tail RMSE={_fmt(row['tail_rmse'])}, "
            f"peak error={_fmt(row['peak_err_s'])}s, selection={_fmt(row['selection'])}。"
        )
    mean = rows[METRIC_COLUMNS].mean(numeric_only=True)
    std = rows[METRIC_COLUMNS].std(numeric_only=True)
    return (
        f"- {experiment_id}: {len(rows)} seeds 平均 test RMSE={_fmt(mean['test_steer_rmse'])}±{_fmt(std['test_steer_rmse'])}, "
        f"primary RMSE={_fmt(mean['primary_rmse'])}±{_fmt(std['primary_rmse'])}, "
        f"tail RMSE={_fmt(mean['tail_rmse'])}±{_fmt(std['tail_rmse'])}, "
        f"peak error={_fmt(mean['peak_err_s'])}±{_fmt(std['peak_err_s'])}s, "
        f"selection={_fmt(mean['selection'])}±{_fmt(std['selection'])}。"
    )


def _delta_sentence(delta_df: pd.DataFrame, pair: str, metric: str, label: str) -> str:
    rows = delta_df[delta_df["pair"].astype(str).eq(pair)]
    if rows.empty:
        return f"- {pair} {label}: 暂无成对样本差异。"
    mean_col = f"delta_{metric}_mean"
    low_col = f"delta_{metric}_mean_ci95_low"
    high_col = f"delta_{metric}_mean_ci95_high"
    rate_col = f"delta_{metric}_improved_rate"
    if len(rows) > 1:
        mean_values = rows[mean_col].astype(float)
        low_values = rows[low_col].astype(float)
        high_values = rows[high_col].astype(float)
        rate_values = rows[rate_col].astype(float)
        left_better = int((mean_values < 0).sum())
        ci_left_better = int((high_values < 0).sum())
        ci_cross_zero = int(((low_values <= 0) & (high_values >= 0)).sum())
        return (
            f"- {pair} {label}: seed 平均={_fmt(mean_values.mean())}±{_fmt(mean_values.std())}, "
            f"左侧更好 seeds={left_better}/{len(rows)}, "
            f"CI支持左侧={ci_left_better}/{len(rows)}, CI跨0={ci_cross_zero}/{len(rows)}, "
            f"improved_rate均值={_fmt(rate_values.mean())}。"
        )
    row = rows.iloc[0]
    return (
        f"- {pair} {label}: mean={_fmt(row.get(mean_col))}, "
        f"CI95=[{_fmt(row.get(low_col))}, {_fmt(row.get(high_col))}], "
        f"improved_rate={_fmt(row.get(rate_col))}。"
    )


def _write_experiment_notes(out_dir: Path, metrics_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    seed_count = int(metrics_df["seed"].nunique()) if "seed" in metrics_df.columns else 0
    lines: list[str] = [
        "# E0-E2 中文实验说明草稿",
        "",
        "说明：这份说明由汇总脚本自动生成，用于快速给老师解释当前 E0/E1/E2 结果。正式汇报前仍应人工检查图和表。",
        "",
        "## 实验目的",
        "",
        "- E0：直接预测 + 连续驾驶风格，用作 raw RMSE 简单强基线。",
        "- E1：粗细双头，无连续驾驶风格，用作验证 style 的干净结构对照。",
        "- E2：粗细双头 + 连续驾驶风格，用于判断连续驾驶风格是否在结构化模型里稳定有效。",
        "",
        "## 当前数据协议",
        "",
        "- 使用固定 FAIR manifest 和固定 train / val / test split。",
        "- 训练配置沿用 FAIR full-run 条件：40 epochs、batch_size=64、lr=1e-3。",
        f"- 当前汇总包含 `{seed_count}` 个 seed。",
        "",
        "## 指标概览",
        "",
        _metric_sentence(metrics_df, "E0"),
        _metric_sentence(metrics_df, "E1"),
        _metric_sentence(metrics_df, "E2"),
        "",
        "## 成对样本差异",
        "",
        "负数表示左侧模型误差更低。",
        "",
        _delta_sentence(delta_df, "E2:E1", "rmse_2s_abs_steer", "2秒整体转向误差"),
        _delta_sentence(delta_df, "E2:E1", "rmse_pre_tail_abs_steer", "前段/主响应误差"),
        _delta_sentence(delta_df, "E2:E1", "rmse_tail_abs_steer", "尾部误差"),
        "",
        _delta_sentence(delta_df, "E0:E2", "rmse_2s_abs_steer", "2秒整体转向误差"),
        _delta_sentence(delta_df, "E0:E2", "peak_time_abs_err_s", "峰值时间误差"),
        "",
        "## 当前可说的结论",
        "",
    ]
    if seed_count < 3:
        lines.extend(
            [
                "- 当前 seed 数不足 3 个，只能作为方向性结果或 smoke-check，不能作为最终稳定结论。",
                "- 如果 E2 在后续 seeds 中继续优于 E1，才能正式说连续驾驶风格有稳定贡献。",
                "- E0 仍应保留为 raw RMSE guardrail，防止结构化模型在整体 RMSE 上输给简单直接预测路线。",
            ]
        )
    else:
        lines.extend(
            [
                "- 当前已有 3 个 seed，可以开始根据 2/3 seeds 是否改善、均值/标准差、成对差异和 bootstrap CI 判断稳定性。",
                "- 若 E2 相对 E1 在至少 2/3 seeds 改善，且 tail/peak 不系统退化，可作为连续驾驶风格有效的正式证据。",
                "- E0 仍应作为 raw RMSE guardrail 与 E2 同时报告。",
            ]
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "- 跑完或确认 E0/E1/E2 的 3-seed 证据后，再进入 E3 非 EEG 生理状态实验。",
            "- 在 E3 稳定前，不应直接进入 EEG 推理输入或 EEG teacher-student 结论。",
        ]
    )
    (out_dir / "experiment_notes_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_df = pd.read_csv(args.runs_csv)
    required = {"experiment_id", "seed", "run_root"}
    missing = required.difference(runs_df.columns)
    if missing:
        raise ValueError(f"runs CSV missing required columns: {sorted(missing)}")

    metric_rows = [_metric_row(row) for _, row in runs_df.iterrows()]
    metrics_df = pd.DataFrame(metric_rows)
    summary_df = _summarize_mean_std(metrics_df)

    delta_rows: list[dict[str, Any]] = []
    for pair_spec in args.pairs:
        delta_rows.extend(
            _paired_delta_rows(
                runs_df,
                pair_spec,
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_seed=int(args.bootstrap_seed),
            )
        )
    delta_df = pd.DataFrame(delta_rows)

    metrics_df.to_csv(out_dir / "seed_wise_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "mean_std_metrics.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(out_dir / "paired_sample_deltas.csv", index=False, encoding="utf-8-sig")
    _write_markdown(out_dir, metrics_df, summary_df, delta_df)
    _write_experiment_notes(out_dir, metrics_df, delta_df)

    print(f"seed_wise_metrics: {out_dir / 'seed_wise_metrics.csv'}")
    print(f"mean_std_metrics: {out_dir / 'mean_std_metrics.csv'}")
    print(f"paired_sample_deltas: {out_dir / 'paired_sample_deltas.csv'}")
    print(f"summary: {out_dir / 'summary_cn.md'}")
    print(f"experiment_notes: {out_dir / 'experiment_notes_cn.md'}")


if __name__ == "__main__":
    main()
