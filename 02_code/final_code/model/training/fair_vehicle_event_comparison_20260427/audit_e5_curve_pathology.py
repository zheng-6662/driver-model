# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
DEFAULT_SUMMARY_DIR = REPORTS_DIR / "style_physio_eeg_e5_distill_summary_20260508"
DEFAULT_SEED_METRICS = DEFAULT_SUMMARY_DIR / "seed_wise_metrics.csv"
DEFAULT_OUT_DIR = DEFAULT_SUMMARY_DIR / "curve_pathology_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit curve-shape pathology risks for E5A.")
    parser.add_argument("--seed-metrics", default=str(DEFAULT_SEED_METRICS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--peak-risk-sec", type=float, default=0.6)
    parser.add_argument("--tail-corr-risk", type=float, default=0.0)
    parser.add_argument("--shape-corr-risk", type=float, default=0.2)
    parser.add_argument("--high-rmse-risk", type=float, default=1.0)
    return parser.parse_args()


def _load_seed_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"seed metrics not found: {path}")
    return pd.read_csv(path)


def _load_samples(seed_metrics: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    rows = []
    for _, row in seed_metrics[seed_metrics["experiment_id"].astype(str).eq(experiment_id)].iterrows():
        path = Path(str(row["sample_metrics_csv"]))
        if not path.exists():
            raise FileNotFoundError(f"sample metrics not found: {path}")
        df = pd.read_csv(path)
        df["experiment_id"] = experiment_id
        df["seed"] = int(row["seed"])
        rows.append(df)
    if not rows:
        raise ValueError(f"no sample metrics for {experiment_id}")
    return pd.concat(rows, ignore_index=True)


def _flag_risks(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["risk_high_rmse"] = out["rmse_2s_abs_steer"].astype(float) > float(args.high_rmse_risk)
    out["risk_peak_time"] = out["peak_time_abs_err_s"].astype(float) > float(args.peak_risk_sec)
    out["risk_tail_corr"] = out["tail_trend_corr"].astype(float) < float(args.tail_corr_risk)
    out["risk_shape_corr"] = out["shape_corr"].astype(float) < float(args.shape_corr_risk)
    out["risk_tail_shape_corr"] = out["tail_shape_corr"].astype(float) < float(args.shape_corr_risk)
    risk_cols = [col for col in out.columns if col.startswith("risk_")]
    out["risk_count"] = out[risk_cols].sum(axis=1).astype(int)
    return out


def _risk_summary(flagged: pd.DataFrame) -> pd.DataFrame:
    risk_cols = [col for col in flagged.columns if col.startswith("risk_") and col != "risk_count"]
    rows: list[dict[str, Any]] = []
    for seed, group in flagged.groupby("seed"):
        item: dict[str, Any] = {"seed": int(seed), "n": int(len(group))}
        for col in risk_cols:
            item[f"{col}_rate"] = float(group[col].mean())
            item[f"{col}_count"] = int(group[col].sum())
        item["any_risk_rate"] = float((group["risk_count"] > 0).mean())
        item["multi_risk_rate"] = float((group["risk_count"] >= 2).mean())
        rows.append(item)
    return pd.DataFrame(rows)


def _group_risk_summary(flagged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_col in ["phase_type", "road_type_anchor", "eval_morphology_label", "structure_slice"]:
        if group_col not in flagged.columns:
            continue
        for (seed, value), group in flagged.groupby(["seed", group_col], dropna=False):
            rows.append(
                {
                    "seed": int(seed),
                    "group_column": group_col,
                    "group_value": "" if pd.isna(value) else str(value),
                    "n": int(len(group)),
                    "high_rmse_rate": float(group["risk_high_rmse"].mean()),
                    "peak_time_risk_rate": float(group["risk_peak_time"].mean()),
                    "tail_corr_risk_rate": float(group["risk_tail_corr"].mean()),
                    "shape_corr_risk_rate": float(group["risk_shape_corr"].mean()),
                    "tail_shape_corr_risk_rate": float(group["risk_tail_shape_corr"].mean()),
                    "multi_risk_rate": float((group["risk_count"] >= 2).mean()),
                    "rmse_mean": float(group["rmse_2s_abs_steer"].mean()),
                    "tail_rmse_mean": float(group["rmse_tail_abs_steer"].mean()),
                    "peak_time_mean": float(group["peak_time_abs_err_s"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _top_risk_samples(flagged: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
    cols = [
        "seed",
        "sample_key",
        "phase_type",
        "road_type_anchor",
        "eval_morphology_label",
        "rmse_2s_abs_steer",
        "rmse_tail_abs_steer",
        "peak_time_abs_err_s",
        "tail_trend_corr",
        "shape_corr",
        "tail_shape_corr",
        "risk_count",
        "risk_high_rmse",
        "risk_peak_time",
        "risk_tail_corr",
        "risk_shape_corr",
        "risk_tail_shape_corr",
    ]
    existing = [col for col in cols if col in flagged.columns]
    return flagged.sort_values(["risk_count", "rmse_2s_abs_steer"], ascending=[False, False]).head(top_n)[existing]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _write_report(out_dir: Path, summary: pd.DataFrame, group_summary: pd.DataFrame) -> None:
    lines = [
        "# E5A 曲线病灶自动审计",
        "",
        "## 审计说明",
        "",
        "这一步不重新训练，只用 E5A 的 sample-level metrics 自动筛查潜在曲线问题。它不能替代人工看图，但可以告诉我们哪些风险更值得看。",
        "",
        "风险规则：",
        "",
        "- `risk_high_rmse`：单样本 2 秒方向盘误差较高；",
        "- `risk_peak_time`：峰值时间误差较大；",
        "- `risk_tail_corr`：尾段趋势相关性低；",
        "- `risk_shape_corr`：整体形状相关性低；",
        "- `risk_tail_shape_corr`：尾段形状相关性低。",
        "",
        "## 每个 seed 的风险比例",
        "",
        "| seed | n | 任一风险比例 | 多风险比例 | 高 RMSE | 峰值时间 | 尾段趋势 | 整体形状 | 尾段形状 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["seed"])),
                    str(int(row["n"])),
                    _fmt(row["any_risk_rate"]),
                    _fmt(row["multi_risk_rate"]),
                    _fmt(row["risk_high_rmse_rate"]),
                    _fmt(row["risk_peak_time_rate"]),
                    _fmt(row["risk_tail_corr_rate"]),
                    _fmt(row["risk_shape_corr_rate"]),
                    _fmt(row["risk_tail_shape_corr_rate"]),
                ]
            )
            + " |"
        )
    worst_groups = group_summary.sort_values("multi_risk_rate", ascending=False).head(12)
    lines.extend(
        [
            "",
            "## 多风险比例最高的分组",
            "",
            "| seed | 分组 | 取值 | n | 多风险比例 | RMSE均值 | 尾段RMSE | 峰值时间 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
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
                    _fmt(row["multi_risk_rate"]),
                    _fmt(row["rmse_mean"]),
                    _fmt(row["tail_rmse_mean"]),
                    _fmt(row["peak_time_mean"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `risk_summary_by_seed.csv`",
            "- `risk_summary_by_group.csv`",
            "- `top_curve_risk_samples.csv`",
            "",
            "## 当前用法",
            "",
            "人工看图时，优先看 `top_curve_risk_samples.csv` 里的样本，再结合 `regression_comparison_plots/` 里的退化样本图。",
        ]
    )
    (out_dir / "curve_pathology_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics = _load_seed_metrics(Path(args.seed_metrics))
    e5_samples = _load_samples(seed_metrics, "E5A")
    flagged = _flag_risks(e5_samples, args)
    summary = _risk_summary(flagged)
    group_summary = _group_risk_summary(flagged)
    top_samples = _top_risk_samples(flagged)

    flagged.to_csv(out_dir / "e5a_sample_curve_risk_flags.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "risk_summary_by_seed.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(out_dir / "risk_summary_by_group.csv", index=False, encoding="utf-8-sig")
    top_samples.to_csv(out_dir / "top_curve_risk_samples.csv", index=False, encoding="utf-8-sig")
    _write_report(out_dir, summary, group_summary)

    print(f"curve_pathology_report: {out_dir / 'curve_pathology_report_cn.md'}")
    print(f"risk_summary_by_seed: {out_dir / 'risk_summary_by_seed.csv'}")
    print(f"risk_summary_by_group: {out_dir / 'risk_summary_by_group.csv'}")
    print(f"top_curve_risk_samples: {out_dir / 'top_curve_risk_samples.csv'}")


if __name__ == "__main__":
    main()
