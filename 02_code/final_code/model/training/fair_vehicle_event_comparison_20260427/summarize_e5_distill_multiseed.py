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

DEFAULT_E5_RUNS = REPORTS_DIR / "style_physio_eeg_e5_distill_runs_20260508.csv"
DEFAULT_REFERENCE_METRICS = (
    REPORTS_DIR / "style_physio_eeg_e3_e4_summary_final_20260507" / "seed_wise_metrics.csv"
)
DEFAULT_OUT_DIR = REPORTS_DIR / "style_physio_eeg_e5_distill_summary_20260508"

EXPERIMENT_NAMES = {
    "E0": "直接预测 + 连续驾驶风格",
    "E2": "粗细双头 + 连续驾驶风格",
    "E3": "无 EEG 生理状态 + 连续驾驶风格",
    "E4": "含 EEG 生理状态 + 连续驾驶风格",
    "E5A": "EEG 教师 / 无 EEG 学生：粗细双头 + 连续驾驶风格",
}

EXPERIMENT_ORDER = ["E0", "E2", "E3", "E4", "E5A", "E5B"]
METRIC_COLUMNS = [
    "test_steer_rmse",
    "primary_rmse",
    "tail_rmse",
    "peak_err_s",
    "tail_direction",
    "selection",
]
DELTA_COLUMNS = [
    "rmse_2s_abs_steer",
    "rmse_pre_tail_abs_steer",
    "rmse_tail_abs_steer",
    "peak_time_abs_err_s",
    "turning_count_abs_err",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize E5 EEG-teacher/no-EEG-student runs.")
    parser.add_argument("--e5-runs-csv", default=str(DEFAULT_E5_RUNS))
    parser.add_argument("--reference-seed-metrics", default=str(DEFAULT_REFERENCE_METRICS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=["E5A:E2", "E5A:E4", "E5A:E0", "E4:E2", "E5B:E2", "E5B:E5A", "E5B:E3"],
        help="Paired comparisons. Negative deltas mean the left experiment has lower error.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260508)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sort_key(experiment_id: str) -> tuple[int, str]:
    if experiment_id in EXPERIMENT_ORDER:
        return (EXPERIMENT_ORDER.index(experiment_id), experiment_id)
    return (len(EXPERIMENT_ORDER), experiment_id)


def _read_reference_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"reference seed metrics not found: {path}")
    df = pd.read_csv(path)
    keep = df["experiment_id"].astype(str).isin(["E0", "E2", "E3", "E4"])
    df = df[keep].copy()
    df["experiment_name"] = df["experiment_id"].map(EXPERIMENT_NAMES).fillna(df["experiment_id"])
    if "prediction_overview" not in df.columns:
        df["prediction_overview"] = df["run_root"].map(
            lambda value: str(Path(str(value)) / "prediction_figures" / "test" / "overview.png")
        )
    return df


def _e5_metric_row(row: pd.Series) -> dict[str, Any] | None:
    run_root = Path(str(row["run_root"]))
    metrics_path = run_root / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = _load_json(metrics_path)
    test = metrics["test"]
    selection = test["selection_summary"]
    experiment_id = str(row["experiment_id"])
    sample_metrics = run_root / "prediction_figures" / "test" / "prediction_sample_metrics.csv"
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
        "sample_metrics_csv": str(sample_metrics),
        "prediction_overview": str(run_root / "prediction_figures" / "test" / "overview.png"),
        "teacher_checkpoint": str(row.get("teacher_checkpoint", "")),
        "distill_weight": float(row.get("distill_weight", 0.0)),
        "distill_tail_weight": float(row.get("distill_tail_weight", 0.0)),
    }


def _read_e5_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"E5 runs CSV not found: {path}")
    runs = pd.read_csv(path)
    runs = runs[runs["run_root"].fillna("").astype(str).str.len() > 0].copy()
    rows = [_e5_metric_row(row) for _, row in runs.iterrows()]
    rows = [row for row in rows if row is not None]
    if not rows:
        raise ValueError("No completed E5 metrics were found.")
    return pd.DataFrame(rows)


def _mean_std(metrics_df: pd.DataFrame) -> pd.DataFrame:
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


def _load_sample_metrics(row: pd.Series) -> pd.DataFrame:
    path = Path(str(row["sample_metrics_csv"]))
    if not path.exists():
        raise FileNotFoundError(f"sample metrics not found: {path}")
    return pd.read_csv(path)


def _stable_offset(text: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text))


def _bootstrap_ci(values: pd.Series, n_bootstrap: int, seed: int) -> tuple[float, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) == 0 or n_bootstrap <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    means = arr[sample_idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _paired_deltas(
    metrics_df: pd.DataFrame,
    pairs: list[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        left_id, right_id = pair.split(":", 1)
        left_runs = metrics_df[metrics_df["experiment_id"].astype(str).eq(left_id)]
        right_runs = metrics_df[metrics_df["experiment_id"].astype(str).eq(right_id)]
        shared_seeds = sorted(set(left_runs["seed"].astype(int)).intersection(right_runs["seed"].astype(int)))
        for seed in shared_seeds:
            left_row = left_runs[left_runs["seed"].astype(int).eq(seed)].iloc[0]
            right_row = right_runs[right_runs["seed"].astype(int).eq(seed)].iloc[0]
            left_df = _load_sample_metrics(left_row)
            right_df = _load_sample_metrics(right_row)
            merged = left_df.merge(right_df, on="sample_key", suffixes=(f"_{left_id}", f"_{right_id}"))
            item: dict[str, Any] = {
                "pair": pair,
                "left": left_id,
                "right": right_id,
                "seed": int(seed),
                "n_samples": int(len(merged)),
            }
            for col in DELTA_COLUMNS:
                left_col = f"{col}_{left_id}"
                right_col = f"{col}_{right_id}"
                if left_col not in merged.columns or right_col not in merged.columns:
                    continue
                delta = merged[left_col] - merged[right_col]
                ci_seed = bootstrap_seed + int(seed) + _stable_offset(pair) + _stable_offset(col)
                ci_low, ci_high = _bootstrap_ci(delta, bootstrap_samples, ci_seed)
                item[f"delta_{col}_mean"] = float(delta.mean())
                item[f"delta_{col}_median"] = float(delta.median())
                item[f"delta_{col}_improved_rate"] = float((delta < 0).mean())
                item[f"delta_{col}_mean_ci95_low"] = ci_low
                item[f"delta_{col}_mean_ci95_high"] = ci_high
            rows.append(item)
    return pd.DataFrame(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    view = df if columns is None else df[columns]
    if view.empty:
        return "暂无结果。"
    lines = [
        "| " + " | ".join(map(str, view.columns)) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for value in row.tolist():
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(_fmt(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _mean_row(summary_df: pd.DataFrame, experiment_id: str) -> pd.Series | None:
    rows = summary_df[summary_df["experiment_id"].astype(str).eq(experiment_id)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _seed_metric_line(metrics_df: pd.DataFrame, experiment_id: str) -> str:
    rows = metrics_df[metrics_df["experiment_id"].astype(str).eq(experiment_id)].sort_values("seed")
    if rows.empty:
        return f"- {experiment_id}：暂无结果。"
    parts = [
        f"{int(row.seed)} RMSE={_fmt(row.test_steer_rmse)}, tail={_fmt(row.tail_rmse)}, selection={_fmt(row.selection)}"
        for row in rows.itertuples(index=False)
    ]
    return f"- {experiment_id}：{'；'.join(parts)}。"


def _pair_summary(delta_df: pd.DataFrame, pair: str, col: str) -> dict[str, Any]:
    rows = delta_df[delta_df["pair"].astype(str).eq(pair)] if not delta_df.empty else pd.DataFrame()
    mean_col = f"delta_{col}_mean"
    low_col = f"delta_{col}_mean_ci95_low"
    high_col = f"delta_{col}_mean_ci95_high"
    if rows.empty or mean_col not in rows:
        return {"n": 0, "better": 0, "ci_negative": 0, "mean": float("nan")}
    means = rows[mean_col].astype(float)
    lows = rows[low_col].astype(float)
    highs = rows[high_col].astype(float)
    return {
        "n": int(len(rows)),
        "better": int((means < 0).sum()),
        "ci_negative": int((highs < 0).sum()),
        "ci_cross_zero": int(((lows <= 0) & (highs >= 0)).sum()),
        "mean": float(means.mean()),
    }


def _write_reports(out_dir: Path, metrics_df: pd.DataFrame, summary_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    metric_cols = ["experiment_id", "experiment_name", "seed", *METRIC_COLUMNS]
    summary_cols = ["experiment_id", "experiment_name", "n_seeds"]
    for col in METRIC_COLUMNS:
        summary_cols.extend([f"{col}_mean", f"{col}_std"])

    e5 = _mean_row(summary_df, "E5A")
    e2 = _mean_row(summary_df, "E2")
    e4 = _mean_row(summary_df, "E4")
    e5_vs_e2 = _pair_summary(delta_df, "E5A:E2", "rmse_2s_abs_steer")
    e5_tail_vs_e2 = _pair_summary(delta_df, "E5A:E2", "rmse_tail_abs_steer")
    e5_vs_e4 = _pair_summary(delta_df, "E5A:E4", "rmse_2s_abs_steer")

    short_conclusion = "E5A 目前值得作为候选继续检查。"
    if e5 is not None and e2 is not None:
        rmse_gain = e5["test_steer_rmse_mean"] - e2["test_steer_rmse_mean"]
        tail_gain = e5["tail_rmse_mean"] - e2["tail_rmse_mean"]
        if rmse_gain < 0 and e5_vs_e2["better"] >= 2:
            short_conclusion = "E5A 相对 E2 出现三种子正向信号，但还要看峰值时间、预测图和是否需要 E5B/E6 对照。"
        elif rmse_gain >= 0:
            short_conclusion = "E5A 没有形成均值优势，不应提升为主线。"
        elif tail_gain > 0:
            short_conclusion = "E5A 虽有整体误差信号，但尾段退化，需要谨慎。"

    summary_lines = [
        "# E5 EEG 教师 / 无 EEG 学生三种子汇总",
        "",
        "说明：负的配对差值表示左侧实验误差更低。E5A 的学生模型推理时只使用车辆数据和连续驾驶风格，不使用生理/EEG；训练时使用同 seed 的 E4 EEG 模型作为教师。",
        "",
        "## Seed 级指标",
        "",
        _markdown_table(metrics_df, metric_cols),
        "",
        "## 均值和标准差",
        "",
        _markdown_table(summary_df, summary_cols),
        "",
        "## 配对样本差异",
        "",
        _markdown_table(delta_df),
    ]
    (out_dir / "summary_cn.md").write_text("\n".join(summary_lines), encoding="utf-8")

    notes = [
        "# E5 中文实验说明",
        "",
        "## 这组实验回答什么问题",
        "",
        "- E4 已经说明 EEG 相比无 EEG 生理状态有信号，但 E4 推理时需要 EEG，部署不一定现实。",
        "- E5A 检查的是：能不能让含 EEG 的 E4 模型只在训练阶段当教师，把一部分 EEG 信息转移给推理时不需要 EEG 的学生模型。",
        "- 学生模型保留当前最稳定的连续驾驶风格输入，避免把“无 EEG”误解成“不要利用驾驶风格”。",
        "",
        "## 固定训练条件",
        "",
        "- manifest：`protocol_allphase_control_v2_context_full2s/sample_manifest.csv`",
        "- train/val/test：`4797 / 692 / 749`",
        "- seeds：`2026 / 2027 / 2028`",
        "- 训练：CUDA，`40` epochs，batch size `64`，lr `1e-3`",
        "- 蒸馏权重：全轨迹 `0.20`，尾段 `0.05`",
        "",
        "## 和谁比较",
        "",
        "- E2：粗细双头 + 连续驾驶风格，是当前可部署强基准。",
        "- E4：含 EEG 生理状态 + 连续驾驶风格，是 EEG 有信号的教师/上限参考，不直接当最终可部署模型。",
        "- E0：直接预测 + 连续驾驶风格，是 raw RMSE 保护性对照。",
        "",
        "## 主要结果",
        "",
        _seed_metric_line(metrics_df, "E2"),
        _seed_metric_line(metrics_df, "E4"),
        _seed_metric_line(metrics_df, "E5A"),
        "",
    ]
    if e5 is not None:
        notes.append(
            f"- E5A 三种子均值：test RMSE={_fmt(e5['test_steer_rmse_mean'])}±{_fmt(e5['test_steer_rmse_std'])}，"
            f"primary={_fmt(e5['primary_rmse_mean'])}，tail={_fmt(e5['tail_rmse_mean'])}，"
            f"peak={_fmt(e5['peak_err_s_mean'])}s，selection={_fmt(e5['selection_mean'])}。"
        )
    if e2 is not None:
        notes.append(
            f"- E2 三种子均值：test RMSE={_fmt(e2['test_steer_rmse_mean'])}±{_fmt(e2['test_steer_rmse_std'])}，"
            f"primary={_fmt(e2['primary_rmse_mean'])}，tail={_fmt(e2['tail_rmse_mean'])}，"
            f"peak={_fmt(e2['peak_err_s_mean'])}s，selection={_fmt(e2['selection_mean'])}。"
        )
    if e4 is not None:
        notes.append(
            f"- E4 三种子均值：test RMSE={_fmt(e4['test_steer_rmse_mean'])}±{_fmt(e4['test_steer_rmse_std'])}，"
            f"tail={_fmt(e4['tail_rmse_mean'])}，selection={_fmt(e4['selection_mean'])}。"
        )

    notes.extend(
        [
            "",
            "## 配对判断",
            "",
            f"- E5A vs E2：整体 2 秒误差均值差={_fmt(e5_vs_e2['mean'])}，E5A 更好的 seed={e5_vs_e2['better']}/{e5_vs_e2['n']}，CI 完全小于 0 的 seed={e5_vs_e2['ci_negative']}/{e5_vs_e2['n']}。",
            f"- E5A vs E2：尾段误差均值差={_fmt(e5_tail_vs_e2['mean'])}，E5A 更好的 seed={e5_tail_vs_e2['better']}/{e5_tail_vs_e2['n']}，CI 完全小于 0 的 seed={e5_tail_vs_e2['ci_negative']}/{e5_tail_vs_e2['n']}。",
            f"- E5A vs E4：整体 2 秒误差均值差={_fmt(e5_vs_e4['mean'])}，E5A 更好的 seed={e5_vs_e4['better']}/{e5_vs_e4['n']}。",
            "",
            "## 当前可向老师汇报的保守结论",
            "",
            f"- {short_conclusion}",
            "- 更保守的说法是：EEG 作为训练期教师是有价值的候选路线；它目前比“直接把无 EEG 生理状态输入模型”的 E3 更有希望。",
            "- 还不能只凭这一组就说问题完全解决，因为需要检查预测图是否有局部形状问题，并决定是否补 E5B（无 EEG 生理状态学生）或 E6（可靠性门控无 EEG 生理融合）作为解释性对照。",
            "",
            "## 预测图材料",
            "",
            "- 每个 seed 的图路径见 `prediction_figure_index.csv`。",
            "- 汇报时建议固定展示同一批 case，对比 E2、E4、E5A 的预测曲线；重点看尾段是否塌陷、峰值时间是否明显偏移。",
        ]
    )
    (out_dir / "experiment_notes_cn.md").write_text("\n".join(notes), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_df = _read_reference_metrics(Path(args.reference_seed_metrics))
    e5_df = _read_e5_metrics(Path(args.e5_runs_csv))
    metrics_df = pd.concat([reference_df, e5_df], ignore_index=True, sort=False)
    metrics_df["_sort"] = metrics_df["experiment_id"].map(lambda value: _sort_key(str(value)))
    metrics_df = metrics_df.sort_values(["_sort", "seed"]).drop(columns=["_sort"]).reset_index(drop=True)
    summary_df = _mean_std(metrics_df)
    delta_df = _paired_deltas(metrics_df, list(args.pairs), int(args.bootstrap_samples), int(args.bootstrap_seed))

    figure_cols = [
        "experiment_id",
        "experiment_name",
        "seed",
        "run_root",
        "prediction_overview",
        "sample_metrics_csv",
    ]
    metrics_df.to_csv(out_dir / "seed_wise_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "mean_std_metrics.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(out_dir / "paired_sample_deltas.csv", index=False, encoding="utf-8-sig")
    metrics_df[figure_cols].to_csv(out_dir / "prediction_figure_index.csv", index=False, encoding="utf-8-sig")
    _write_reports(out_dir, metrics_df, summary_df, delta_df)

    print(f"seed_wise_metrics: {out_dir / 'seed_wise_metrics.csv'}")
    print(f"mean_std_metrics: {out_dir / 'mean_std_metrics.csv'}")
    print(f"paired_sample_deltas: {out_dir / 'paired_sample_deltas.csv'}")
    print(f"prediction_figure_index: {out_dir / 'prediction_figure_index.csv'}")
    print(f"summary_cn: {out_dir / 'summary_cn.md'}")
    print(f"experiment_notes_cn: {out_dir / 'experiment_notes_cn.md'}")


if __name__ == "__main__":
    main()
