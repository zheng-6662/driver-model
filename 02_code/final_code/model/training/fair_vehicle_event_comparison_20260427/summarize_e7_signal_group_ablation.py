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
DEFAULT_E7_RUNS = REPORTS_DIR / "style_physio_eeg_e7_signal_group_runs_20260508.csv"
DEFAULT_E0_E2_RUNS = REPORTS_DIR / "style_physio_eeg_e0_e2_fresh_3seed_runs_20260507.csv"
DEFAULT_E3_E4_RUNS = REPORTS_DIR / "style_physio_eeg_e3_e4_runs_20260507.csv"
DEFAULT_OUT_DIR = REPORTS_DIR / "style_physio_eeg_e7_signal_group_summary_20260508"

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
    "E3": "无 EEG 生理状态 + 连续驾驶风格",
    "E4": "含 EEG 生理状态 + 连续驾驶风格",
    "E7A": "EEG 单独语义状态 + 连续驾驶风格",
    "E7B": "raw EEG 单独 + 连续驾驶风格",
    "E7C": "raw 无 EEG 生理信号 + 连续驾驶风格",
}

EXPERIMENT_ORDER = ["E0", "E2", "E3", "E4", "E7A", "E7B", "E7C"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize E7 signal-group ablation runs.")
    parser.add_argument("--runs-csv", default=str(DEFAULT_E7_RUNS))
    parser.add_argument("--e0-e2-runs-csv", default=str(DEFAULT_E0_E2_RUNS))
    parser.add_argument("--e3-e4-runs-csv", default=str(DEFAULT_E3_E4_RUNS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=["E7A:E4", "E7A:E3", "E7A:E2", "E7B:E7A", "E7B:E2", "E7C:E3", "E7C:E2"],
        help="Paired comparisons as LEFT:RIGHT. Negative deltas mean LEFT has lower error.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260508)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_runs_csv(path: Path, keep_ids: set[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if keep_ids is not None and "experiment_id" in df.columns:
        df = df[df["experiment_id"].astype(str).isin(keep_ids)]
    if "run_root" in df.columns:
        df = df[df["run_root"].fillna("").astype(str).str.len() > 0]
    return df.copy()


def _experiment_sort_key(experiment_id: str) -> tuple[int, str]:
    if experiment_id in EXPERIMENT_ORDER:
        return (EXPERIMENT_ORDER.index(experiment_id), experiment_id)
    return (len(EXPERIMENT_ORDER), experiment_id)


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
            f"- {experiment_id}（{name}，seed={int(row['seed'])}）："
            f"RMSE={_fmt(row['test_steer_rmse'])}，tail={_fmt(row['tail_rmse'])}，"
            f"selection={_fmt(row['selection'])}。"
        )
    mean = rows[METRIC_COLUMNS].mean()
    std = rows[METRIC_COLUMNS].std(ddof=1)
    return (
        f"- {experiment_id}（{name}，{len(rows)} seeds）："
        f"RMSE={_fmt(mean['test_steer_rmse'])}±{_fmt(std['test_steer_rmse'])}，"
        f"tail={_fmt(mean['tail_rmse'])}，selection={_fmt(mean['selection'])}。"
    )


def _delta_sentence(delta_df: pd.DataFrame, pair: str, metric_col: str) -> str:
    rows = delta_df[delta_df["pair"].astype(str).eq(pair)]
    if rows.empty:
        return f"- {pair}：暂无配对样本结果。"
    col = f"delta_{metric_col}_mean"
    improved_col = f"delta_{metric_col}_improved_rate"
    if col not in rows.columns:
        return f"- {pair}：缺少 {metric_col} 配对列。"
    mean_delta = float(rows[col].mean())
    better_seeds = int((rows[col] < 0).sum())
    improved_rate = float(rows[improved_col].mean()) if improved_col in rows.columns else float("nan")
    return (
        f"- {pair}：配对样本平均差={_fmt(mean_delta)}，"
        f"更好的 seed 数={better_seeds}/{len(rows)}，"
        f"样本改善率均值={_fmt(improved_rate, 3)}。"
    )


def _write_report(metrics_df: pd.DataFrame, mean_std_df: pd.DataFrame, delta_df: pd.DataFrame, out_dir: Path) -> None:
    report_rows = mean_std_df.copy()
    report_rows["order"] = report_rows["experiment_id"].map(
        {name: idx for idx, name in enumerate(EXPERIMENT_ORDER)}
    ).fillna(99)
    report_rows = report_rows.sort_values(["order", "experiment_id"])
    display = report_rows[
        [
            "experiment_id",
            "experiment_name",
            "n_seeds",
            "test_steer_rmse_mean",
            "test_steer_rmse_std",
            "tail_rmse_mean",
            "peak_err_s_mean",
            "selection_mean",
        ]
    ].copy()
    for col in [
        "test_steer_rmse_mean",
        "test_steer_rmse_std",
        "tail_rmse_mean",
        "peak_err_s_mean",
        "selection_mean",
    ]:
        display[col] = display[col].map(lambda x: _fmt(x))

    lines: list[str] = [
        "# E7 生理信号分组消融汇总",
        "",
        "## 目的",
        "",
        "这一步只回答一个问题：EEG 有用时，其他生理信号是否在当前融合方式下反而带来噪声。固定 FAIR 协议、训练样本、事件锚点和训练参数，只改变输入到模型的生理/EEG 分组。",
        "",
        "## 版本含义",
        "",
        "- E7A：只保留 EEG，按当前语义状态公式构造状态量，再和连续驾驶风格一起输入。",
        "- E7B：只保留 raw EEG 特征直接输入，用来判断 raw EEG 是否能直接用。",
        "- E7C：只保留 raw 无 EEG 生理信号（HR/EDA/EMG）直接输入，用来判断 EEG 之外信号是否可能加噪声。",
        "- E3/E4/E2 是已有对照：E3 是无 EEG 语义生理状态，E4 是含 EEG 语义生理状态，E2 是不加生理的连续风格强基准。",
        "",
        "## 指标汇总",
        "",
        _df_to_markdown(display),
        "",
        "## 单版本结论",
        "",
    ]
    for experiment_id in EXPERIMENT_ORDER:
        lines.append(_metric_sentence(metrics_df, experiment_id))

    lines.extend(
        [
            "",
            "## 配对样本对比",
            "",
            "负数表示左边版本误差更低。",
        ]
    )
    for pair in ["E7A:E4", "E7A:E3", "E7A:E2", "E7B:E7A", "E7B:E2", "E7C:E3", "E7C:E2"]:
        lines.append(_delta_sentence(delta_df, pair, "rmse_2s_abs_steer"))

    lines.extend(
        [
            "",
            "## 保守解释规则",
            "",
            "- 如果 E7A 接近或优于 E4，并且明显优于 E3，说明 E4 的主要收益更可能来自 EEG，其他生理信号至少没有提供稳定额外收益。",
            "- 如果 E7A 明显弱于 E4，说明 EEG 可能需要和其他信号共同表达，或者当前 EEG-only 语义公式过弱，不能直接说其他生理信号都是噪声。",
            "- 如果 E7B 弱于 E7A，说明 raw EEG 直接拼接不如语义/压缩后的 EEG 表达，后续不应继续简单 raw 拼接。",
            "- 如果 E7C 弱于 E2/E3，说明当前 HR/EDA/EMG raw 直接输入不适合作为主线；后续只能考虑筛选、可靠性门控或单信号验证。",
            "- seed 数不足 3 时，只能作为继续或停止的门槛判断，不能写成最终论文结论。",
            "",
            "## 后续动作",
            "",
            "1. 先看 seed-2026 是否出现明确 no-go：如果 E7A/E7B/E7C 都明显弱于 E2，停止 raw/直接输入堆叠路线。",
            "2. 如果 E7A 相比 E4/E2 有希望，再补 2027/2028，并加入方向/幅值物理复核和预测图人工复核。",
            "3. 如果 EEG-only 有用但 raw EEG 不行，下一步优先考虑 EEG 教师、EEG 可靠性门控或更合理的 EEG 状态构造，而不是继续堆 HR/EDA/EMG。",
        ]
    )
    (out_dir / "teacher_report_e7_20260508.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prediction_index_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in metrics_df.iterrows():
        rows.append(
            {
                "experiment_id": row["experiment_id"],
                "experiment_name": row["experiment_name"],
                "seed": int(row["seed"]),
                "run_root": row["run_root"],
                "prediction_overview": row["prediction_overview"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_parts = [
        _read_runs_csv(Path(args.e0_e2_runs_csv), keep_ids={"E0", "E2"}),
        _read_runs_csv(Path(args.e3_e4_runs_csv), keep_ids={"E3", "E4"}),
        _read_runs_csv(Path(args.runs_csv), keep_ids={"E7A", "E7B", "E7C"}),
    ]
    runs_df = pd.concat([df for df in runs_parts if not df.empty], ignore_index=True)
    if runs_df.empty:
        raise RuntimeError("No completed runs found for E7 summary.")

    metric_rows: list[dict[str, Any]] = []
    for _, row in runs_df.iterrows():
        item = _metric_row(row)
        if item is not None:
            metric_rows.append(item)
    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        raise RuntimeError("No metrics.json files found for completed runs.")
    metrics_df["order"] = metrics_df["experiment_id"].map(
        {name: idx for idx, name in enumerate(EXPERIMENT_ORDER)}
    ).fillna(99)
    metrics_df = metrics_df.sort_values(["order", "seed", "experiment_id"]).drop(columns=["order"])
    mean_std_df = _summarize_mean_std(metrics_df)

    delta_rows: list[dict[str, Any]] = []
    for pair in args.pairs:
        delta_rows.extend(
            _paired_delta_rows(
                metrics_df,
                pair,
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_seed=int(args.bootstrap_seed),
            )
        )
    delta_df = pd.DataFrame(delta_rows)

    metrics_df.to_csv(out_dir / "seed_wise_metrics.csv", index=False, encoding="utf-8-sig")
    mean_std_df.to_csv(out_dir / "mean_std_metrics.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(out_dir / "paired_sample_deltas.csv", index=False, encoding="utf-8-sig")
    _prediction_index_rows(metrics_df).to_csv(out_dir / "prediction_index.csv", index=False, encoding="utf-8-sig")
    _write_report(metrics_df, mean_std_df, delta_df, out_dir)

    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
