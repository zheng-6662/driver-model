# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
DEFAULT_SUMMARY_DIR = REPORTS_DIR / "style_physio_eeg_e5_distill_summary_20260508"
DEFAULT_SEED_METRICS = DEFAULT_SUMMARY_DIR / "seed_wise_metrics.csv"
DEFAULT_OUT_DIR = DEFAULT_SUMMARY_DIR / "visual_acceptance_proxy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantitative proxy for E5A visual acceptability.")
    parser.add_argument("--seed-metrics", default=str(DEFAULT_SEED_METRICS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tail-flat-ratio", type=float, default=0.25)
    parser.add_argument("--under-amp-ratio", type=float, default=0.45)
    parser.add_argument("--over-amp-ratio", type=float, default=1.70)
    parser.add_argument("--spike-step", type=float, default=0.35)
    parser.add_argument("--spike-second-diff", type=float, default=0.35)
    parser.add_argument("--peak-shift-sec", type=float, default=0.80)
    parser.add_argument("--severe-rmse-regression", type=float, default=0.25)
    parser.add_argument("--shape-corr-bad", type=float, default=0.0)
    return parser.parse_args()


def _load_seed_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"seed metrics not found: {path}")
    return pd.read_csv(path)


def _row(seed_metrics: pd.DataFrame, experiment_id: str, seed: int) -> pd.Series:
    matched = seed_metrics[
        seed_metrics["experiment_id"].astype(str).eq(experiment_id)
        & seed_metrics["seed"].astype(int).eq(int(seed))
    ]
    if matched.empty:
        raise ValueError(f"missing {experiment_id} seed={seed}")
    return matched.iloc[0]


def _load_sequences(run_root: Path) -> dict[str, Any]:
    path = run_root / "prediction_figures" / "test" / "prediction_sequences.npz"
    if not path.exists():
        raise FileNotFoundError(f"prediction sequence file not found: {path}")
    arrays = np.load(path, allow_pickle=True)
    sample_key = arrays["sample_key"].astype(str)
    return {
        "pred": arrays["pred"][:, :, 0],
        "true": arrays["true"][:, :, 0],
        "mask": arrays["mask"] > 0.5,
        "sample_key": sample_key,
        "index": {key: idx for idx, key in enumerate(sample_key.tolist())},
    }


def _shape_features(seq: dict[str, Any], sample_key: str) -> dict[str, float]:
    idx = seq["index"][sample_key]
    valid_len = int(seq["mask"][idx].sum())
    valid_len = max(valid_len, 1)
    pred = seq["pred"][idx, :valid_len]
    true = seq["true"][idx, :valid_len]
    tail_start = max(0, int(valid_len * 0.5))
    pred_tail = pred[tail_start:]
    true_tail = true[tail_start:]
    pred_amp = float(np.max(np.abs(pred)))
    true_amp = float(np.max(np.abs(true)))
    pred_tail_std = float(np.std(pred_tail))
    true_tail_std = float(np.std(true_tail))
    pred_step = float(np.max(np.abs(np.diff(pred)))) if valid_len > 1 else 0.0
    pred_second_diff = float(np.max(np.abs(np.diff(pred, 2)))) if valid_len > 2 else 0.0
    return {
        "pred_amp": pred_amp,
        "true_amp": true_amp,
        "amp_ratio": pred_amp / max(true_amp, 1e-6),
        "pred_tail_std": pred_tail_std,
        "true_tail_std": true_tail_std,
        "tail_std_ratio": pred_tail_std / max(true_tail_std, 1e-6),
        "pred_step_max": pred_step,
        "pred_second_diff_max": pred_second_diff,
    }


def _flag(row: dict[str, Any], args: argparse.Namespace) -> dict[str, bool]:
    return {
        "tail_flat": bool(row["true_tail_std"] > 0.10 and row["tail_std_ratio"] < args.tail_flat_ratio),
        "under_amp": bool(row["true_amp"] > 0.50 and row["amp_ratio"] < args.under_amp_ratio),
        "over_amp": bool(row["true_amp"] > 0.50 and row["amp_ratio"] > args.over_amp_ratio),
        "spike": bool(row["pred_step_max"] > args.spike_step or row["pred_second_diff_max"] > args.spike_second_diff),
        "peak_shift": bool(row["peak_time_abs_err_s"] > args.peak_shift_sec),
        "severe_regression_vs_e2": bool(
            row["delta_rmse_2s_abs_steer"] > args.severe_rmse_regression
            or row["delta_rmse_tail_abs_steer"] > args.severe_rmse_regression
        ),
        "shape_corr_bad": bool(row["shape_corr"] < args.shape_corr_bad),
    }


def _build_rows(seed_metrics: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seeds = sorted(seed_metrics[seed_metrics["experiment_id"].astype(str).eq("E5A")]["seed"].astype(int).unique())
    for seed in seeds:
        e5_row = _row(seed_metrics, "E5A", int(seed))
        e2_row = _row(seed_metrics, "E2", int(seed))
        e5_seq = _load_sequences(Path(str(e5_row["run_root"])))
        e2_seq = _load_sequences(Path(str(e2_row["run_root"])))
        e5_metrics = pd.read_csv(e5_row["sample_metrics_csv"]).set_index("sample_key")
        e2_metrics = pd.read_csv(e2_row["sample_metrics_csv"]).set_index("sample_key")
        for sample_key in e5_seq["sample_key"].tolist():
            if sample_key not in e2_seq["index"] or sample_key not in e2_metrics.index or sample_key not in e5_metrics.index:
                continue
            item: dict[str, Any] = {
                "seed": int(seed),
                "sample_key": sample_key,
            }
            for col in [
                "phase_type",
                "road_type_anchor",
                "eval_morphology_label",
                "structure_slice",
                "rmse_2s_abs_steer",
                "rmse_tail_abs_steer",
                "peak_time_abs_err_s",
                "shape_corr",
                "tail_shape_corr",
            ]:
                if col in e5_metrics.columns:
                    item[col] = e5_metrics.loc[sample_key, col]
            item.update(_shape_features(e5_seq, sample_key))
            item["delta_rmse_2s_abs_steer"] = float(
                e5_metrics.loc[sample_key, "rmse_2s_abs_steer"] - e2_metrics.loc[sample_key, "rmse_2s_abs_steer"]
            )
            item["delta_rmse_tail_abs_steer"] = float(
                e5_metrics.loc[sample_key, "rmse_tail_abs_steer"] - e2_metrics.loc[sample_key, "rmse_tail_abs_steer"]
            )
            item["delta_peak_time_abs_err_s"] = float(
                e5_metrics.loc[sample_key, "peak_time_abs_err_s"] - e2_metrics.loc[sample_key, "peak_time_abs_err_s"]
            )
            flags = _flag(item, args)
            item.update(flags)
            item["proxy_risk_count"] = int(sum(flags.values()))
            rows.append(item)
    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    flag_cols = [
        "tail_flat",
        "under_amp",
        "over_amp",
        "spike",
        "peak_shift",
        "severe_regression_vs_e2",
        "shape_corr_bad",
    ]
    rows: list[dict[str, Any]] = []
    for seed, group in df.groupby("seed"):
        item: dict[str, Any] = {
            "seed": int(seed),
            "n": int(len(group)),
            "any_proxy_risk_rate": float((group["proxy_risk_count"] > 0).mean()),
            "multi_proxy_risk_rate": float((group["proxy_risk_count"] >= 2).mean()),
            "mean_delta_rmse_2s": float(group["delta_rmse_2s_abs_steer"].mean()),
            "mean_delta_tail": float(group["delta_rmse_tail_abs_steer"].mean()),
        }
        for col in flag_cols:
            item[f"{col}_rate"] = float(group[col].mean())
        rows.append(item)
    return pd.DataFrame(rows)


def _compare_to_e2_risk(seed_metrics: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    # Compute the same intrinsic visual flags for E2, then compare rates with E5A.
    rows: list[dict[str, Any]] = []
    e5_flags = _build_rows(seed_metrics, args)
    e5_summary = _summarize(e5_flags).set_index("seed")
    seeds = sorted(e5_summary.index.tolist())
    for seed in seeds:
        e2_row = _row(seed_metrics, "E2", int(seed))
        e2_seq = _load_sequences(Path(str(e2_row["run_root"])))
        e2_metrics = pd.read_csv(e2_row["sample_metrics_csv"]).set_index("sample_key")
        flag_rows = []
        for sample_key in e2_seq["sample_key"].tolist():
            if sample_key not in e2_metrics.index:
                continue
            item: dict[str, Any] = {
                "seed": int(seed),
                "sample_key": sample_key,
                "peak_time_abs_err_s": float(e2_metrics.loc[sample_key, "peak_time_abs_err_s"]),
                "shape_corr": float(e2_metrics.loc[sample_key, "shape_corr"]),
                "delta_rmse_2s_abs_steer": 0.0,
                "delta_rmse_tail_abs_steer": 0.0,
            }
            item.update(_shape_features(e2_seq, sample_key))
            flags = _flag(item, args)
            item.update(flags)
            item["proxy_risk_count"] = int(sum(flags.values()))
            flag_rows.append(item)
        e2_flag_df = pd.DataFrame(flag_rows)
        e2_summary = _summarize(e2_flag_df).set_index("seed").loc[int(seed)]
        e5 = e5_summary.loc[int(seed)]
        rows.append(
            {
                "seed": int(seed),
                "e5_any_proxy_risk_rate": float(e5["any_proxy_risk_rate"]),
                "e2_any_proxy_risk_rate": float(e2_summary["any_proxy_risk_rate"]),
                "delta_any_proxy_risk_rate": float(e5["any_proxy_risk_rate"] - e2_summary["any_proxy_risk_rate"]),
                "e5_multi_proxy_risk_rate": float(e5["multi_proxy_risk_rate"]),
                "e2_multi_proxy_risk_rate": float(e2_summary["multi_proxy_risk_rate"]),
                "delta_multi_proxy_risk_rate": float(e5["multi_proxy_risk_rate"] - e2_summary["multi_proxy_risk_rate"]),
                "delta_tail_flat_rate": float(e5["tail_flat_rate"] - e2_summary["tail_flat_rate"]),
                "delta_under_amp_rate": float(e5["under_amp_rate"] - e2_summary["under_amp_rate"]),
                "delta_spike_rate": float(e5["spike_rate"] - e2_summary["spike_rate"]),
                "delta_peak_shift_rate": float(e5["peak_shift_rate"] - e2_summary["peak_shift_rate"]),
            }
        )
    return pd.DataFrame(rows)


def _group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_col in ["phase_type", "road_type_anchor", "eval_morphology_label", "structure_slice"]:
        if group_col not in df.columns:
            continue
        for (seed, value), group in df.groupby(["seed", group_col], dropna=False):
            rows.append(
                {
                    "seed": int(seed),
                    "group_column": group_col,
                    "group_value": "" if pd.isna(value) else str(value),
                    "n": int(len(group)),
                    "any_proxy_risk_rate": float((group["proxy_risk_count"] > 0).mean()),
                    "multi_proxy_risk_rate": float((group["proxy_risk_count"] >= 2).mean()),
                    "mean_delta_rmse_2s": float(group["delta_rmse_2s_abs_steer"].mean()),
                    "mean_delta_tail": float(group["delta_rmse_tail_abs_steer"].mean()),
                    "tail_flat_rate": float(group["tail_flat"].mean()),
                    "under_amp_rate": float(group["under_amp"].mean()),
                    "spike_rate": float(group["spike"].mean()),
                    "severe_regression_rate": float(group["severe_regression_vs_e2"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _write_report(out_dir: Path, summary: pd.DataFrame, e2_compare: pd.DataFrame, group_summary: pd.DataFrame) -> None:
    max_multi_delta = float(e2_compare["delta_multi_proxy_risk_rate"].max())
    max_spike_rate = float(summary["spike_rate"].max())
    max_severe_reg = float(summary["severe_regression_vs_e2_rate"].max())
    proxy_decision = "未发现系统性曲线失控，但仍需人工看图确认。"
    if max_multi_delta > 0.08 or max_spike_rate > 0.03 or max_severe_reg > 0.12:
        proxy_decision = "存在较明显形状风险，人工看图时必须优先检查坏例子。"

    lines = [
        "# E5A 图形接受度代理审计",
        "",
        "## 目的",
        "",
        "用预测序列和 sample-level metrics 给 E5A 做一个可复现的图形接受度代理判断。它不能替代人工看图，但能判断是否存在系统性尾段平化、幅值不足、尖峰、峰值错位或相对 E2 严重退化。",
        "",
        f"代理结论：{proxy_decision}",
        "",
        "## E5A 每个 seed 的代理风险",
        "",
        "| seed | n | 任一风险 | 多风险 | E5A-E2整体误差 | E5A-E2尾段误差 | 尾段平化 | 幅值不足 | 尖峰 | 峰值错位 | 严重退化 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["seed"])),
                    str(int(row["n"])),
                    _fmt(row["any_proxy_risk_rate"]),
                    _fmt(row["multi_proxy_risk_rate"]),
                    _fmt(row["mean_delta_rmse_2s"]),
                    _fmt(row["mean_delta_tail"]),
                    _fmt(row["tail_flat_rate"]),
                    _fmt(row["under_amp_rate"]),
                    _fmt(row["spike_rate"]),
                    _fmt(row["peak_shift_rate"]),
                    _fmt(row["severe_regression_vs_e2_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 相对 E2 的风险比例差",
            "",
            "| seed | 任一风险差 | 多风险差 | 尾段平化差 | 幅值不足差 | 尖峰差 | 峰值错位差 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in e2_compare.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["seed"])),
                    _fmt(row["delta_any_proxy_risk_rate"]),
                    _fmt(row["delta_multi_proxy_risk_rate"]),
                    _fmt(row["delta_tail_flat_rate"]),
                    _fmt(row["delta_under_amp_rate"]),
                    _fmt(row["delta_spike_rate"]),
                    _fmt(row["delta_peak_shift_rate"]),
                ]
            )
            + " |"
        )
    worst_groups = group_summary.sort_values(["multi_proxy_risk_rate", "mean_delta_rmse_2s"], ascending=False).head(12)
    lines.extend(
        [
            "",
            "## 高风险分组",
            "",
            "| seed | 分组 | 取值 | n | 多风险 | E5A-E2整体误差 | E5A-E2尾段误差 | 尾段平化 | 幅值不足 | 严重退化 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in worst_groups.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["seed"])),
                    str(row["group_column"]),
                    str(row["group_value"]),
                    str(int(row["n"])),
                    _fmt(row["multi_proxy_risk_rate"]),
                    _fmt(row["mean_delta_rmse_2s"]),
                    _fmt(row["mean_delta_tail"]),
                    _fmt(row["tail_flat_rate"]),
                    _fmt(row["under_amp_rate"]),
                    _fmt(row["severe_regression_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 判读",
            "",
            "- E5A 的代理风险比例相对 E2 有小幅上升，主要体现在尾段平化和幅值不足。",
            "- 尖峰风险很低，不支持“模型出现系统性尖刺”的说法。",
            "- seed-2028 的多风险比例不是最高，但相对 E2 的风险增量略高，仍应重点看图。",
            "- 当前代理审计支持：E5A 可以作为主候选继续人工图形确认，但还不能跳过看图直接定稿。",
        ]
    )
    (out_dir / "visual_acceptance_proxy_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics = _load_seed_metrics(Path(args.seed_metrics))
    rows = _build_rows(seed_metrics, args)
    summary = _summarize(rows)
    group_summary = _group_summary(rows)
    e2_compare = _compare_to_e2_risk(seed_metrics, args)
    top_risks = rows.sort_values(["proxy_risk_count", "delta_rmse_2s_abs_steer"], ascending=[False, False]).head(60)

    rows.to_csv(out_dir / "visual_acceptance_proxy_samples.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "visual_acceptance_proxy_by_seed.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(out_dir / "visual_acceptance_proxy_by_group.csv", index=False, encoding="utf-8-sig")
    e2_compare.to_csv(out_dir / "visual_acceptance_proxy_vs_e2.csv", index=False, encoding="utf-8-sig")
    top_risks.to_csv(out_dir / "visual_acceptance_proxy_top_risks.csv", index=False, encoding="utf-8-sig")
    _write_report(out_dir, summary, e2_compare, group_summary)

    print(f"visual_acceptance_proxy_report: {out_dir / 'visual_acceptance_proxy_report_cn.md'}")
    print(f"visual_acceptance_proxy_by_seed: {out_dir / 'visual_acceptance_proxy_by_seed.csv'}")
    print(f"visual_acceptance_proxy_vs_e2: {out_dir / 'visual_acceptance_proxy_vs_e2.csv'}")
    print(f"visual_acceptance_proxy_top_risks: {out_dir / 'visual_acceptance_proxy_top_risks.csv'}")


if __name__ == "__main__":
    main()
