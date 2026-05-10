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
DEFAULT_OUT_DIR = REPORTS_DIR / "style_physio_eeg_e3_e4_summary"
DEFAULT_REFERENCE_RUNS = REPORTS_DIR / "style_physio_eeg_e0_e2_fresh_3seed_runs_20260507.csv"

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

EXPERIMENT_NAMES = {
    "E0": "直接预测 + 连续驾驶风格",
    "E2": "粗细双头 + 连续驾驶风格",
    "E3": "粗细双头 + 无 EEG 生理状态 + 连续驾驶风格",
    "E4": "粗细双头 + 含 EEG 生理状态 + 连续驾驶风格",
}

EXPERIMENT_ORDER = ["E0", "E2", "E3", "E4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize E3/E4 EEG control runs without launching training.")
    parser.add_argument("--runs-csv", required=True, help="CSV with E3/E4 run roots.")
    parser.add_argument("--reference-runs-csv", default=str(DEFAULT_REFERENCE_RUNS), help="Optional E0/E2 reference CSV.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=["E4:E3", "E3:E2", "E4:E2", "E0:E4"],
        help="Paired comparisons as LEFT:RIGHT. Negative deltas mean LEFT has lower error.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260507)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _experiment_sort_key(experiment_id: str) -> tuple[int, str]:
    if experiment_id in EXPERIMENT_ORDER:
        return (EXPERIMENT_ORDER.index(experiment_id), experiment_id)
    return (len(EXPERIMENT_ORDER), experiment_id)


def _read_runs_csv(path: Path, keep_ids: set[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if keep_ids is not None and "experiment_id" in df.columns:
        df = df[df["experiment_id"].astype(str).isin(keep_ids)]
    if "run_root" in df.columns:
        df = df[df["run_root"].fillna("").astype(str).str.len() > 0]
    return df.copy()


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
        "tail_direction": float(selection["tail_direction_match"]),
        "selection": float(selection["selection_score"]),
        "metrics_json": str(metrics_path),
        "sample_metrics_csv": str(run_root / "prediction_figures" / "test" / "prediction_sample_metrics.csv"),
        "prediction_overview": str(run_root / "prediction_figures" / "test" / "overview.png"),
    }


def _summarize_mean_std(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for experiment_id, group in metrics_df.groupby("experiment_id", sort=False):
        item: dict[str, Any] = {
            "experiment_id": experiment_id,
            "experiment_name": EXPERIMENT_NAMES.get(str(experiment_id), str(experiment_id)),
            "n_seeds": int(len(group)),
        }
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
    shared_seeds = sorted(set(left_runs["seed"].astype(int)).intersection(set(right_runs["seed"].astype(int))))
    for seed in shared_seeds:
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
            if left_col not in merged.columns or right_col not in merged.columns:
                continue
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


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _df_to_markdown(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    view = df if columns is None else df[columns]
    if view.empty:
        return "暂无结果。"
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


def _metric_sentence(metrics_df: pd.DataFrame, experiment_id: str) -> str:
    rows = metrics_df[metrics_df["experiment_id"].astype(str).eq(experiment_id)]
    name = EXPERIMENT_NAMES.get(experiment_id, experiment_id)
    if rows.empty:
        return f"- {experiment_id}（{name}）：暂无结果。"
    if len(rows) == 1:
        row = rows.iloc[0]
        return (
            f"- {experiment_id}（{name}）：test RMSE={_fmt(row['test_steer_rmse'])}，"
            f"primary={_fmt(row['primary_rmse'])}，tail={_fmt(row['tail_rmse'])}，"
            f"peak error={_fmt(row['peak_err_s'])}s，selection={_fmt(row['selection'])}。"
        )
    mean = rows[METRIC_COLUMNS].mean(numeric_only=True)
    std = rows[METRIC_COLUMNS].std(numeric_only=True)
    return (
        f"- {experiment_id}（{name}）：{len(rows)} seeds 平均 test RMSE="
        f"{_fmt(mean['test_steer_rmse'])}±{_fmt(std['test_steer_rmse'])}，"
        f"primary={_fmt(mean['primary_rmse'])}±{_fmt(std['primary_rmse'])}，"
        f"tail={_fmt(mean['tail_rmse'])}±{_fmt(std['tail_rmse'])}，"
        f"peak error={_fmt(mean['peak_err_s'])}±{_fmt(std['peak_err_s'])}s，"
        f"selection={_fmt(mean['selection'])}±{_fmt(std['selection'])}。"
    )


def _pair_status(delta_df: pd.DataFrame, pair: str, metric: str = "rmse_2s_abs_steer") -> dict[str, Any]:
    if delta_df.empty or "pair" not in delta_df.columns:
        return {"n": 0, "left_better": 0, "ci_left": 0, "ci_cross": 0, "mean": float("nan")}
    rows = delta_df[delta_df["pair"].astype(str).eq(pair)]
    mean_col = f"delta_{metric}_mean"
    high_col = f"delta_{metric}_mean_ci95_high"
    low_col = f"delta_{metric}_mean_ci95_low"
    if rows.empty or mean_col not in rows:
        return {"n": 0, "left_better": 0, "ci_left": 0, "ci_cross": 0, "mean": float("nan")}
    mean_values = rows[mean_col].astype(float)
    high_values = rows[high_col].astype(float) if high_col in rows else pd.Series(dtype=float)
    low_values = rows[low_col].astype(float) if low_col in rows else pd.Series(dtype=float)
    return {
        "n": int(len(rows)),
        "left_better": int((mean_values < 0).sum()),
        "ci_left": int((high_values < 0).sum()) if len(high_values) else 0,
        "ci_cross": int(((low_values <= 0) & (high_values >= 0)).sum()) if len(high_values) else 0,
        "mean": float(mean_values.mean()),
    }


def _delta_sentence(delta_df: pd.DataFrame, pair: str, metric: str, label: str) -> str:
    if delta_df.empty or "pair" not in delta_df.columns:
        return f"- {pair} {label}：暂无成对结果。"
    rows = delta_df[delta_df["pair"].astype(str).eq(pair)]
    if rows.empty:
        return f"- {pair} {label}：暂无成对结果。"
    status = _pair_status(delta_df, pair, metric)
    return (
        f"- {pair} {label}：seed 平均差={_fmt(status['mean'])}，"
        f"左侧更好 seeds={status['left_better']}/{status['n']}，"
        f"CI 支持左侧={status['ci_left']}/{status['n']}，"
        f"CI 跨 0={status['ci_cross']}/{status['n']}。"
    )


def _write_summary(out_dir: Path, metrics_df: pd.DataFrame, summary_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    metric_cols = ["experiment_id", "experiment_name", "seed", *METRIC_COLUMNS]
    summary_cols = ["experiment_id", "experiment_name", "n_seeds"]
    for col in METRIC_COLUMNS:
        summary_cols.extend([f"{col}_mean", f"{col}_std"])
    lines = [
        "# E3/E4 EEG 对照汇总",
        "",
        "说明：这个文件只读取已经完成的 run，不启动训练。负数表示成对比较左侧模型误差更低。",
        "",
        "## 每个 seed 的指标",
        "",
        _df_to_markdown(metrics_df, metric_cols),
        "",
        "## 均值和标准差",
        "",
        _df_to_markdown(summary_df, summary_cols),
        "",
        "## 成对样本差异",
        "",
        _df_to_markdown(delta_df),
    ]
    (out_dir / "summary_cn.md").write_text("\n".join(lines), encoding="utf-8")


def _write_notes(out_dir: Path, metrics_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    e3_e4_seeds = sorted(
        set(metrics_df[metrics_df["experiment_id"].astype(str).isin(["E3", "E4"])]["seed"].astype(int).tolist())
    )
    eeg_status = _pair_status(delta_df, "E4:E3")
    e4_vs_e2 = _pair_status(delta_df, "E4:E2")
    e3_vs_e2 = _pair_status(delta_df, "E3:E2")
    seed_count = len(e3_e4_seeds)

    lines = [
        "# E3/E4 中文实验说明",
        "",
        "## 这组实验回答什么问题",
        "",
        "- E3：在“粗细双头 + 连续驾驶风格”的基础上，只加入不含 EEG 的生理状态量。",
        "- E4：在同样基础上加入含 EEG 的生理状态量。",
        "- E2：不加入生理状态量，只保留“粗细双头 + 连续驾驶风格”，用作当前结构基准。",
        "- E0：直接预测 + 连续驾驶风格，用作 raw RMSE 保护性对照。",
        "",
        "## 当前固定条件",
        "",
        "- 使用同一 FAIR manifest 和固定 train / val / test split。",
        "- 训练条件保持一致：40 epochs、batch size 64、lr=1e-3、CUDA、seed 为 2026/2027/2028。",
        f"- 当前 E3/E4 已汇总 seed：{', '.join(map(str, e3_e4_seeds)) if e3_e4_seeds else '暂无'}。",
        "",
        "## 指标概览",
        "",
        _metric_sentence(metrics_df, "E0"),
        _metric_sentence(metrics_df, "E2"),
        _metric_sentence(metrics_df, "E3"),
        _metric_sentence(metrics_df, "E4"),
        "",
        "## 关键成对判断",
        "",
        _delta_sentence(delta_df, "E4:E3", "rmse_2s_abs_steer", "整体 2 秒误差"),
        _delta_sentence(delta_df, "E4:E3", "rmse_tail_abs_steer", "尾段误差"),
        _delta_sentence(delta_df, "E3:E2", "rmse_2s_abs_steer", "无 EEG 生理状态相对 E2"),
        _delta_sentence(delta_df, "E4:E2", "rmse_2s_abs_steer", "含 EEG 生理状态相对 E2"),
        "",
        "## 当前可说的结论",
        "",
    ]

    if seed_count < 3:
        lines.extend(
            [
                "- 现在 E3/E4 还没有完成 3 个 seed，所以只能作为阶段观察，不能作为论文级最终结论。",
                "- 如果后续 E4 相对 E3 在至少 2/3 seeds 上稳定降低误差，才可以说 EEG 有继续保留或深入建模的价值。",
                "- 如果 E3/E4 都没有超过 E2，则当前生理状态表示还不能证明对预测有稳定增益，需要考虑更换表示方式或融合方式。",
            ]
        )
    else:
        if eeg_status["left_better"] >= 2 and eeg_status["mean"] < 0:
            lines.append("- EEG 当前有正向证据：E4 相对 E3 在多数 seed 上降低整体误差。下一步可以考虑保留 EEG 分支，或进一步验证 EEG 是否适合作为训练期教师信号。")
        elif eeg_status["left_better"] <= 1 and eeg_status["mean"] >= 0:
            lines.append("- EEG 当前没有稳定正向证据：E4 相对 E3 没有形成多数 seed 的整体误差优势。不能把 EEG 作为默认主输入。")
        else:
            lines.append("- EEG 当前证据不稳定：E4 和 E3 有差异，但方向还不足以支撑明确取舍，需要结合尾段、峰值时间和预测图人工复核。")

        if e3_vs_e2["mean"] < 0 or e4_vs_e2["mean"] < 0:
            lines.append("- 生理状态相对 E2 至少出现了平均误差改善信号，可以继续检查是哪类生理信息贡献了增益。")
        else:
            lines.append("- 生理状态相对 E2 暂未形成平均误差优势，下一步不应简单堆叠生理输入，而应优先考虑特征筛选、可靠性门控或训练期教师方案。")

        lines.append("- E0 仍应保留为 raw RMSE 保护性对照，避免只看结构化模型内部比较。")

    lines.extend(
        [
            "",
            "## 预测图材料",
            "",
            "- 具体预测图路径见 `prediction_figure_index.csv`。",
            "- 汇报时优先展示同一批固定 case 下 E2、E3、E4 的曲线对比。",
        ]
    )
    (out_dir / "experiment_notes_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_runs = _read_runs_csv(Path(args.runs_csv))
    reference_runs = _read_runs_csv(Path(args.reference_runs_csv), keep_ids={"E0", "E2"})
    runs_df = pd.concat([reference_runs, main_runs], ignore_index=True)
    if runs_df.empty:
        raise ValueError("No completed runs found.")

    required = {"experiment_id", "seed", "run_root"}
    missing = required.difference(runs_df.columns)
    if missing:
        raise ValueError(f"runs CSV missing required columns: {sorted(missing)}")

    metric_rows = []
    for _, row in runs_df.iterrows():
        item = _metric_row(row)
        if item is not None:
            metric_rows.append(item)
    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        raise ValueError("No metrics.json files found for completed runs.")

    metrics_df["_sort"] = metrics_df["experiment_id"].map(lambda value: _experiment_sort_key(str(value)))
    metrics_df = metrics_df.sort_values(["_sort", "seed"]).drop(columns=["_sort"]).reset_index(drop=True)
    summary_df = _summarize_mean_std(metrics_df)

    valid_runs = runs_df[runs_df["run_root"].astype(str).isin(metrics_df["run_root"].astype(str))]
    delta_rows: list[dict[str, Any]] = []
    for pair_spec in args.pairs:
        delta_rows.extend(
            _paired_delta_rows(
                valid_runs,
                pair_spec,
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_seed=int(args.bootstrap_seed),
            )
        )
    delta_df = pd.DataFrame(delta_rows)

    figure_cols = ["experiment_id", "experiment_name", "seed", "run_root", "prediction_overview", "sample_metrics_csv"]
    metrics_df.to_csv(out_dir / "seed_wise_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "mean_std_metrics.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(out_dir / "paired_sample_deltas.csv", index=False, encoding="utf-8-sig")
    metrics_df[figure_cols].to_csv(out_dir / "prediction_figure_index.csv", index=False, encoding="utf-8-sig")
    _write_summary(out_dir, metrics_df, summary_df, delta_df)
    _write_notes(out_dir, metrics_df, delta_df)

    print(f"seed_wise_metrics: {out_dir / 'seed_wise_metrics.csv'}")
    print(f"mean_std_metrics: {out_dir / 'mean_std_metrics.csv'}")
    print(f"paired_sample_deltas: {out_dir / 'paired_sample_deltas.csv'}")
    print(f"prediction_figure_index: {out_dir / 'prediction_figure_index.csv'}")
    print(f"summary: {out_dir / 'summary_cn.md'}")
    print(f"experiment_notes: {out_dir / 'experiment_notes_cn.md'}")


if __name__ == "__main__":
    main()
