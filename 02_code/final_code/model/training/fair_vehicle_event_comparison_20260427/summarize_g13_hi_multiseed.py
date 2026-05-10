# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import pandas as pd

from summarize_g13_seed2026_diagnostics import (
    G11_DIR,
    G13_DIR,
    PROJECT_ROOT,
    _fmt_num,
    _fmt_pct,
    _load_model_sample_detail,
    _read_csv,
    _safe_bool_mean,
    _summarize_g11,
    _summarize_group,
    _summarize_subject,
    parse_args as _base_parse_args,
)


PULLED_DIR = G13_DIR / "g13_hi_2027_2028_parallel_20260510"
DEFAULT_OUT_DIR = G13_DIR / "g13_hi_multiseed_summary_20260510"
CURRENT_RESULT_LOG = PROJECT_ROOT / "04_project_logs" / "reports" / "current_model_version_result_log_20260509.csv"


def _parse_metric_mean(value: Any) -> float:
    text = str(value)
    if "±" in text:
        text = text.split("±", 1)[0]
    try:
        return float(text)
    except Exception:
        return math.nan


def _parse_metric_std(value: Any) -> float:
    text = str(value)
    if "±" not in text:
        return math.nan
    try:
        return float(text.split("±", 1)[1])
    except Exception:
        return math.nan


def _read_g13_seed2026() -> pd.DataFrame:
    path = G13_DIR / "g13_seed2026_full_index.csv"
    df = _read_csv(path)
    df = df[df["experiment_id"].isin(["G13H", "G13I"])].copy()
    return pd.DataFrame(
        {
            "experiment_id": df["experiment_id"],
            "experiment_name": df["label"],
            "seed": df["seed"].astype(int),
            "local_run_root": df["local_run_root"],
            "test_rmse": pd.to_numeric(df["test_steer_rmse"], errors="coerce"),
            "primary_rmse": pd.to_numeric(df["primary_rmse"], errors="coerce"),
            "tail_rmse": pd.to_numeric(df["tail_rmse"], errors="coerce"),
            "peak_err_s": pd.to_numeric(df["peak_err_s"], errors="coerce"),
            "selection_score": pd.to_numeric(df["selection"], errors="coerce"),
            "source": "seed2026_first_screen",
        }
    )


def _read_pulled_2027_2028() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(PULLED_DIR.glob("G13*_seed*/g13_run_log.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("experiment_id") not in {"G13H", "G13I"}:
                    continue
                run_root = str(row["run_root"])
                local_run_root = PROJECT_ROOT / "tmp" / "event_conditioned_runs" / Path(run_root).name
                rows.append(
                    {
                        "experiment_id": row["experiment_id"],
                        "experiment_name": row["label"],
                        "seed": int(row["seed"]),
                        "local_run_root": str(local_run_root),
                        "test_rmse": float(row["test_steer_rmse"]),
                        "primary_rmse": float(row["primary_rmse"]),
                        "tail_rmse": float(row["tail_rmse"]),
                        "peak_err_s": float(row["peak_err_s"]),
                        "selection_score": float(row["selection"]),
                        "source": "server_parallel_2027_2028",
                    }
                )
    return pd.DataFrame(rows)


def _summarize_metrics(seed_wise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exp_id, group in seed_wise.groupby("experiment_id", sort=True):
        row = {
            "experiment_id": exp_id,
            "experiment_name": str(group["experiment_name"].iloc[0]),
            "n_seeds": int(group["seed"].nunique()),
        }
        for col in ["test_rmse", "primary_rmse", "tail_rmse", "peak_err_s", "selection_score"]:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _accepted_baselines() -> pd.DataFrame:
    df = _read_csv(CURRENT_RESULT_LOG)
    keep = df[df["version"].isin(["E5A", "E6", "E10C"])].copy()
    rows: list[dict[str, Any]] = []
    for _, row in keep.iterrows():
        rows.append(
            {
                "experiment_id": row["version"],
                "experiment_name": row["name"],
                "n_seeds": int(row["n_seeds"]),
                "test_rmse_mean": _parse_metric_mean(row["test_rmse"]),
                "test_rmse_std": _parse_metric_std(row["test_rmse"]),
                "primary_rmse_mean": _parse_metric_mean(row["primary_rmse"]),
                "primary_rmse_std": _parse_metric_std(row["primary_rmse"]),
                "tail_rmse_mean": _parse_metric_mean(row["tail_rmse"]),
                "tail_rmse_std": _parse_metric_std(row["tail_rmse"]),
                "peak_err_s_mean": _parse_metric_mean(row["peak_err_s"]),
                "peak_err_s_std": _parse_metric_std(row["peak_err_s"]),
                "selection_score_mean": _parse_metric_mean(row["selection"]),
                "selection_score_std": _parse_metric_std(row["selection"]),
                "role": row["role"],
                "decision": row["decision"],
            }
        )
    return pd.DataFrame(rows)


def _mean_by_experiment(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exp_id, group in df.groupby("experiment_id", sort=True):
        item = {"experiment_id": exp_id, "experiment_name": str(group["experiment_name"].iloc[0])}
        for col in value_cols:
            item[col] = float(pd.to_numeric(group[col], errors="coerce").mean())
        rows.append(item)
    return pd.DataFrame(rows)


def _summarize_physical_by_seed(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (exp_id, seed), group in detail.groupby(["experiment_id", "seed"], sort=True):
        major = group[group["is_major_response"].fillna(False)]
        large = group[group["is_large_response"].fillna(False)]
        rows.append(
            {
                "experiment_id": exp_id,
                "experiment_name": str(group["experiment_name"].iloc[0]),
                "seed": int(seed),
                "n_samples": int(len(group)),
                "median_amp_ratio_major": float(pd.to_numeric(major["amp_ratio"], errors="coerce").median())
                if len(major)
                else math.nan,
                "mean_amp_ratio_major": float(pd.to_numeric(major["amp_ratio"], errors="coerce").mean())
                if len(major)
                else math.nan,
                "under_amp_rate_major": _safe_bool_mean(major["under_amp"]) if len(major) else math.nan,
                "severe_under_amp_rate_large": _safe_bool_mean(large["severe_under_amp"]) if len(large) else math.nan,
                "opposite_at_true_peak_rate_major": _safe_bool_mean(major["opposite_at_true_peak"])
                if len(major)
                else math.nan,
                "opposite_at_pred_peak_rate_major": _safe_bool_mean(major["opposite_at_pred_peak"])
                if len(major)
                else math.nan,
                "opposite_side_heavy_rate_major": _safe_bool_mean(major["opposite_side_heavy"])
                if len(major)
                else math.nan,
                "tail_drift_risk_rate": _safe_bool_mean(group["tail_drift_risk"]),
                "tail_drift_abs_mean": float(pd.to_numeric(group["tail_drift_abs"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_g11_by_seed(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    g11 = detail[detail["is_g11_case"].fillna(False)].copy()
    for (exp_id, seed), group in g11.groupby(["experiment_id", "seed"], sort=True):
        major = group[group["is_major_response"].fillna(False)]
        large = group[group["is_large_response"].fillna(False)]
        rows.append(
            {
                "experiment_id": exp_id,
                "experiment_name": str(group["experiment_name"].iloc[0]),
                "seed": int(seed),
                "n_g11": int(len(group)),
                "g11_rmse_mean": float(pd.to_numeric(group["rmse_2s_abs_steer"], errors="coerce").mean()),
                "g11_tail_rmse_mean": float(pd.to_numeric(group["rmse_tail_abs_steer"], errors="coerce").mean()),
                "g11_peak_err_s_mean": float(pd.to_numeric(group["peak_time_abs_err_s"], errors="coerce").mean()),
                "g11_under_amp_rate_major": _safe_bool_mean(major["under_amp"]) if len(major) else math.nan,
                "g11_severe_under_amp_rate_large": _safe_bool_mean(large["severe_under_amp"]) if len(large) else math.nan,
                "g11_opposite_at_true_peak_rate_major": _safe_bool_mean(major["opposite_at_true_peak"])
                if len(major)
                else math.nan,
                "g11_tail_drift_risk_rate": _safe_bool_mean(group["tail_drift_risk"]),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    out_path: Path,
    accepted: pd.DataFrame,
    g13_summary: pd.DataFrame,
    seed_wise: pd.DataFrame,
    physical_mean: pd.DataFrame,
    g11_mean: pd.DataFrame,
) -> None:
    accepted_order = ["E5A", "E6", "E10C"]
    g13_order = ["G13H", "G13I"]
    accepted = accepted.set_index("experiment_id").loc[accepted_order].reset_index()
    g13_summary = g13_summary.set_index("experiment_id").loc[g13_order].reset_index()
    seed_wise = seed_wise.sort_values(["experiment_id", "seed"])

    e10c_mean = float(accepted.loc[accepted["experiment_id"].eq("E10C"), "test_rmse_mean"].iloc[0])
    e5a_mean = float(accepted.loc[accepted["experiment_id"].eq("E5A"), "test_rmse_mean"].iloc[0])
    e6_mean = float(accepted.loc[accepted["experiment_id"].eq("E6"), "test_rmse_mean"].iloc[0])
    g13h_mean = float(g13_summary.loc[g13_summary["experiment_id"].eq("G13H"), "test_rmse_mean"].iloc[0])
    g13i_mean = float(g13_summary.loc[g13_summary["experiment_id"].eq("G13I"), "test_rmse_mean"].iloc[0])

    lines: list[str] = []
    lines.append("# G13H/G13I 三种子复验总结")
    lines.append("")
    lines.append("## 直接结论")
    lines.append("")
    lines.append("- G13H 在 seed2026 上看起来有突破，但补完 seed2027/2028 后，三种子平均没有超过当前已接受的 E5A、E6、E10C。")
    lines.append("- G13I 加入困难响应加权和物理约束后，也没有形成稳定增益，三种子整体弱于 G13H。")
    lines.append("- 因此本轮不能把 G13H 或 G13I 升级为新的最终主线；它们更适合作为“脑电教师 + 肌电学生 + 响应类型监督”路线的诊断证据。")
    lines.append("- 当前论文级主线仍应保留 E5A/E6/E10C：E5A 和 E6 代表脑电教师路线，E10C 代表允许肌电推理路线。")
    lines.append("")

    lines.append("## 与当前已接受强基准对比")
    lines.append("")
    lines.append("这里的 E5A/E6/E10C 使用 `current_model_version_result_log_20260509.csv` 中已经整理好的三种子结果；G13H/G13I 使用本次新补齐的三种子结果。")
    lines.append("")
    lines.append("说明：旧主表与 G13 运行器里的“主响应误差”字段口径不完全一致，所以跨版本主结论只比较 test RMSE、尾段误差、峰值时间误差和综合选择指标。")
    lines.append("")
    lines.append("| 版本 | 含义 | seeds | test RMSE | 尾段误差 | 峰值时间误差 | 综合选择指标 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in accepted.iterrows():
        lines.append(
            f"| {row['experiment_id']} | {row['experiment_name']} | {int(row['n_seeds'])} | "
            f"{_fmt_num(row['test_rmse_mean'])}±{_fmt_num(row['test_rmse_std'])} | "
            f"{_fmt_num(row['tail_rmse_mean'])} | "
            f"{_fmt_num(row['peak_err_s_mean'])} | {_fmt_num(row['selection_score_mean'])} |"
        )
    for _, row in g13_summary.iterrows():
        lines.append(
            f"| {row['experiment_id']} | {row['experiment_name']} | {int(row['n_seeds'])} | "
            f"{_fmt_num(row['test_rmse_mean'])}±{_fmt_num(row['test_rmse_std'])} | "
            f"{_fmt_num(row['tail_rmse_mean'])} | "
            f"{_fmt_num(row['peak_err_s_mean'])} | {_fmt_num(row['selection_score_mean'])} |"
        )
    lines.append("")

    lines.append("## G13 单种子明细")
    lines.append("")
    lines.append("| 版本 | seed | test RMSE | 主响应误差（G13 内部口径） | 尾段误差 | 峰值时间误差 | 综合选择指标 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in seed_wise.iterrows():
        lines.append(
            f"| {row['experiment_id']} | {int(row['seed'])} | {_fmt_num(row['test_rmse'])} | "
            f"{_fmt_num(row['primary_rmse'])} | {_fmt_num(row['tail_rmse'])} | "
            f"{_fmt_num(row['peak_err_s'])} | {_fmt_num(row['selection_score'])} |"
        )
    lines.append("")

    lines.append("## 物理风险均值")
    lines.append("")
    lines.append("| 版本 | 幅值比中位数 | 幅值不足率 | 严重幅值不足率 | 真实主峰错侧率 | 后段漂移风险 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in physical_mean.set_index("experiment_id").loc[g13_order].reset_index().iterrows():
        lines.append(
            f"| {row['experiment_id']} | {_fmt_num(row['median_amp_ratio_major'], 3)} | "
            f"{_fmt_pct(row['under_amp_rate_major'])} | {_fmt_pct(row['severe_under_amp_rate_large'])} | "
            f"{_fmt_pct(row['opposite_at_true_peak_rate_major'])} | {_fmt_pct(row['tail_drift_risk_rate'])} |"
        )
    lines.append("")

    lines.append("## G11 困难样本均值")
    lines.append("")
    lines.append("| 版本 | G11 RMSE | G11 尾段误差 | G11 峰值时间误差 | G11 幅值不足率 | G11 错侧率 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in g11_mean.set_index("experiment_id").loc[g13_order].reset_index().iterrows():
        lines.append(
            f"| {row['experiment_id']} | {_fmt_num(row['g11_rmse_mean'])} | {_fmt_num(row['g11_tail_rmse_mean'])} | "
            f"{_fmt_num(row['g11_peak_err_s_mean'])} | {_fmt_pct(row['g11_under_amp_rate_major'])} | "
            f"{_fmt_pct(row['g11_opposite_at_true_peak_rate_major'])} |"
        )
    lines.append("")

    lines.append("## 解释")
    lines.append("")
    lines.append(
        f"- G13H 三种子 test RMSE 均值为 {_fmt_num(g13h_mean)}，比当前 E10C 的 {_fmt_num(e10c_mean)} 高 "
        f"{_fmt_num(g13h_mean - e10c_mean)}，比 E5A 的 {_fmt_num(e5a_mean)} 高 {_fmt_num(g13h_mean - e5a_mean)}，"
        f"比 E6 的 {_fmt_num(e6_mean)} 高 {_fmt_num(g13h_mean - e6_mean)}。"
    )
    lines.append(
        f"- G13I 三种子 test RMSE 均值为 {_fmt_num(g13i_mean)}，也没有超过 E5A/E6/E10C。"
    )
    lines.append("- seed2026 的正向结果主要说明这条组合有潜力，但 seed2027 的回落说明它还不稳定，不能用单种子结果包装成突破。")
    lines.append("- G13H 的问题仍然是幅值偏保守；G13I 试图缓解这个问题，但没有换来整体和困难样本上的稳定收益。")
    lines.append("")

    lines.append("## 当前建议")
    lines.append("")
    lines.append("1. 暂停继续扩展 G13H/G13I 的同类变体，不继续简单加权或继续堆响应类型模块。")
    lines.append("2. 保留 G13H/G13I 的结果作为负面边界：脑电教师 + 肌电学生 + 响应类型监督不能自动带来稳定提升。")
    lines.append("3. 如果继续突破，下一步应回到失败机制本身：为什么 seed2027 掉、为什么幅值不足仍高、为什么 G11 不能超过 E6。")
    lines.append("4. 对老师汇报时应说：G13 主动突破路线已经验证过，单种子有亮点，但三种子没有超过现有主线，因此暂不替换 E5A/E6/E10C。")
    lines.append("")

    lines.append("## 产物")
    lines.append("")
    lines.append("- `g13_hi_seed_wise_metrics.csv`：G13H/G13I 三种子逐 seed 指标。")
    lines.append("- `g13_hi_three_seed_summary.csv`：G13H/G13I 三种子均值和标准差。")
    lines.append("- `g13_hi_physical_mean.csv`：三种子物理风险均值。")
    lines.append("- `g13_hi_g11_mean.csv`：三种子 G11 困难样本均值。")
    lines.append("- `g13_hi_sample_detail.csv`：逐样本明细。")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _base_parse_args()
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_wise = pd.concat([_read_g13_seed2026(), _read_pulled_2027_2028()], ignore_index=True)
    seed_wise = seed_wise.sort_values(["experiment_id", "seed"])
    summary = _summarize_metrics(seed_wise)
    accepted = _accepted_baselines()

    g11_catalog = _read_csv(G11_DIR / "bad_case_catalog.csv")
    g11_keys = set(g11_catalog["sample_key"].astype(str))
    detail_frames = [_load_model_sample_detail(row, args, g11_keys) for _, row in seed_wise.iterrows()]
    detail = pd.concat(detail_frames, ignore_index=True)

    physical_by_seed = _summarize_physical_by_seed(detail)
    physical_mean = _mean_by_experiment(
        physical_by_seed,
        [
            "median_amp_ratio_major",
            "mean_amp_ratio_major",
            "under_amp_rate_major",
            "severe_under_amp_rate_large",
            "opposite_at_true_peak_rate_major",
            "opposite_at_pred_peak_rate_major",
            "opposite_side_heavy_rate_major",
            "tail_drift_risk_rate",
            "tail_drift_abs_mean",
        ],
    )
    subject = _summarize_subject(detail)
    morphology = _summarize_group(detail, "eval_morphology_label")
    g11_by_seed = _summarize_g11_by_seed(detail)
    g11_mean = _mean_by_experiment(
        g11_by_seed,
        [
            "g11_rmse_mean",
            "g11_tail_rmse_mean",
            "g11_peak_err_s_mean",
            "g11_under_amp_rate_major",
            "g11_severe_under_amp_rate_large",
            "g11_opposite_at_true_peak_rate_major",
            "g11_tail_drift_risk_rate",
        ],
    )

    seed_wise.to_csv(out_dir / "g13_hi_seed_wise_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "g13_hi_three_seed_summary.csv", index=False, encoding="utf-8-sig")
    accepted.to_csv(out_dir / "accepted_baseline_summary_from_20260509.csv", index=False, encoding="utf-8-sig")
    physical_by_seed.to_csv(out_dir / "g13_hi_physical_by_seed.csv", index=False, encoding="utf-8-sig")
    physical_mean.to_csv(out_dir / "g13_hi_physical_mean.csv", index=False, encoding="utf-8-sig")
    subject.to_csv(out_dir / "g13_hi_subject_summary.csv", index=False, encoding="utf-8-sig")
    morphology.to_csv(out_dir / "g13_hi_morphology_summary.csv", index=False, encoding="utf-8-sig")
    g11_by_seed.to_csv(out_dir / "g13_hi_g11_by_seed.csv", index=False, encoding="utf-8-sig")
    g11_mean.to_csv(out_dir / "g13_hi_g11_mean.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(out_dir / "g13_hi_sample_detail.csv", index=False, encoding="utf-8-sig")

    _write_report(
        out_path=out_dir / "g13_hi_multiseed_report_cn.md",
        accepted=accepted,
        g13_summary=summary,
        seed_wise=seed_wise,
        physical_mean=physical_mean,
        g11_mean=g11_mean,
    )
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
