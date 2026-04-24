#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(r"F:\data_set_process\data_process")
REPORTS_DIR = REPO_ROOT / "reports"

MASTER_CSV = REPORTS_DIR / "attribution_master_table.csv"
EVENT_CSV = REPORTS_DIR / "attribution_event_table.csv"

RUN_DATE = "20260408"
REPORT_MD = REPORTS_DIR / f"conditioned_v2_fast_boundary_attribution_{RUN_DATE}.md"
Q1_SUMMARY_CSV = REPORTS_DIR / f"conditioned_v2_q1fast_summary_{RUN_DATE}.csv"
BOUNDARY_EVENT_CSV = REPORTS_DIR / f"conditioned_v2_boundary_event_summary_{RUN_DATE}.csv"

LATENCY_BUCKET_COL = "latency_proxy_bucket"
Q1_BUCKET = "Q1_fast"

Q1_COMPARE_METRICS = [
    "delta_rmse_tail_abs_steer",
    "delta_boundary_shift_abs_err",
    "delta_peak_time_abs_err_s",
    "delta_turning_count_abs_err",
]

DELTA_STRUCTURE_METRICS = [
    "delta_peak_time_abs_err_s",
    "delta_turning_count_abs_err",
    "delta_boundary_shift_abs_err",
    "delta_tail_trend_corr",
]

EXTENDED_STRUCTURE_METRICS = [
    "shape_corr_conditioned",
    "peak_abs_amp_err_conditioned",
    "turning_count_abs_err_conditioned",
    "extrema_count_abs_err_conditioned",
    "range_abs_err_conditioned",
    "peak_time_abs_err_s_conditioned",
    "boundary_shift_abs_err_conditioned",
    "tail_shape_corr_conditioned",
    "tail_trend_corr_conditioned",
    "trend_corr_conditioned",
    "boundary_slope_abs_err_conditioned",
    "tail_slope_abs_err_conditioned",
]

EVENTS_OF_INTEREST = ["first_major_turn_onset", "main_peak"]
MODEL_NAME_MAP = {
    "unconditional_baseline": "baseline",
    "event_conditioned_baseline": "conditioned",
}


def require_columns(df: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")


def fmt_num(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def make_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df.loc[:, columns].copy()
    headers = [str(col) for col in view.columns]
    rows = [[str(value) for value in row] for row in view.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def format_row(values: list[str]) -> str:
        cells = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    table_lines = [format_row(headers), separator]
    table_lines.extend(format_row(row) for row in rows)
    return "\n".join(table_lines)


def summarize_metric(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "q75": np.nan,
            "q90": np.nan,
            "worsen_rate": np.nan,
        }
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=1)),
        "q75": float(clean.quantile(0.75)),
        "q90": float(clean.quantile(0.90)),
        "worsen_rate": float((clean > 0).mean()),
    }


def build_q1_compare(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name, group_df in [
        ("Q1_fast", master[master["is_q1_fast"]]),
        ("non_Q1_fast", master[~master["is_q1_fast"]]),
    ]:
        for metric in Q1_COMPARE_METRICS:
            row = {"section": "q1_vs_non_q1", "group": group_name, "metric": metric}
            row.update(summarize_metric(group_df[metric]))
            rows.append(row)
    return pd.DataFrame(rows)


def build_q1_tail_structure_rankings(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q1 = master[master["is_q1_fast"]].copy()
    q1["tail_worsened_flag"] = q1["delta_rmse_tail_abs_steer"] > 0
    q1_worsened = q1[q1["tail_worsened_flag"]].copy()

    corr_rows: list[dict[str, object]] = []
    target = pd.to_numeric(q1_worsened["delta_rmse_tail_abs_steer"], errors="coerce")
    for metric in DELTA_STRUCTURE_METRICS + EXTENDED_STRUCTURE_METRICS:
        series = pd.to_numeric(q1_worsened[metric], errors="coerce")
        mask = target.notna() & series.notna()
        if mask.sum() < 10:
            continue
        corr = float(target[mask].corr(series[mask]))
        corr_rows.append(
            {
                "section": "q1_tail_worsened_corr",
                "metric": metric,
                "n": int(mask.sum()),
                "pearson_corr": corr,
                "abs_pearson_corr": abs(corr),
                "mean_metric": float(series[mask].mean()),
                "median_metric": float(series[mask].median()),
            }
        )

    corr_df = (
        pd.DataFrame(corr_rows)
        .sort_values(["abs_pearson_corr", "pearson_corr"], ascending=[False, False])
        .reset_index(drop=True)
    )

    diff_rows: list[dict[str, object]] = []
    for metric in DELTA_STRUCTURE_METRICS + EXTENDED_STRUCTURE_METRICS:
        worsened = pd.to_numeric(q1.loc[q1["tail_worsened_flag"], metric], errors="coerce").dropna()
        improved = pd.to_numeric(q1.loc[~q1["tail_worsened_flag"], metric], errors="coerce").dropna()
        if len(worsened) < 10 or len(improved) < 10:
            continue
        pooled_var_num = (len(worsened) - 1) * worsened.var(ddof=1) + (len(improved) - 1) * improved.var(ddof=1)
        pooled_var_den = len(worsened) + len(improved) - 2
        pooled_std = np.sqrt(pooled_var_num / pooled_var_den) if pooled_var_den > 0 else np.nan
        smd = (worsened.mean() - improved.mean()) / pooled_std if pooled_std and not np.isnan(pooled_std) else np.nan
        diff_rows.append(
            {
                "section": "q1_tail_worsened_vs_improved",
                "metric": metric,
                "worsened_mean": float(worsened.mean()),
                "improved_mean": float(improved.mean()),
                "mean_diff": float(worsened.mean() - improved.mean()),
                "worsened_median": float(worsened.median()),
                "improved_median": float(improved.median()),
                "median_diff": float(worsened.median() - improved.median()),
                "effect_size_smd": float(smd) if not pd.isna(smd) else np.nan,
            }
        )

    diff_df = (
        pd.DataFrame(diff_rows)
        .sort_values("effect_size_smd", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )
    return corr_df, diff_df


def build_boundary_cross(master: pd.DataFrame) -> pd.DataFrame:
    return (
        master.groupby(["eval_morphology_label", "subj"], dropna=False)
        .agg(
            sample_count=("sample_key", "size"),
            mean_delta_boundary_shift_abs_err=("delta_boundary_shift_abs_err", "mean"),
            median_delta_boundary_shift_abs_err=("delta_boundary_shift_abs_err", "median"),
            worsen_rate=("delta_boundary_shift_abs_err", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
        )
        .reset_index()
        .sort_values(
            ["mean_delta_boundary_shift_abs_err", "worsen_rate"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
        .assign(section="boundary_subj_morph_cross")
    )


def build_subject_boundary_distribution(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subj in ["gf", "cwh", "tyy"]:
        subj_df = master[master["subj"] == subj]
        for variant, metric in [
            ("baseline", "boundary_shift_abs_err_baseline"),
            ("conditioned", "boundary_shift_abs_err_conditioned"),
            ("delta", "delta_boundary_shift_abs_err"),
        ]:
            stats = summarize_metric(subj_df[metric])
            rows.append(
                {
                    "section": "subject_boundary_distribution",
                    "subj": subj,
                    "variant": variant,
                    "metric": metric,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def build_event_summaries(master: pd.DataFrame, event: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    event = event.merge(master[["sample_key", LATENCY_BUCKET_COL]], on="sample_key", how="left")
    event["bucket_group"] = np.where(event[LATENCY_BUCKET_COL].eq(Q1_BUCKET), "Q1_fast", "non_Q1_fast")
    event["model_short"] = event["model_name"].map(MODEL_NAME_MAP)

    filtered = event[event["event_name"].isin(EVENTS_OF_INTEREST)].copy()

    summary_rows: list[dict[str, object]] = []
    for (event_name, model_short, bucket_group), part in filtered.groupby(
        ["event_name", "model_short", "bucket_group"], dropna=False
    ):
        stats = summarize_metric(part["time_abs_err_s"])
        summary_rows.append(
            {
                "section": "event_time_bucket_model_summary",
                "event_name": event_name,
                "model_name": model_short,
                "bucket_group": bucket_group,
                **stats,
            }
        )

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["event_name", "model_name", "bucket_group"])
        .reset_index(drop=True)
    )

    paired = (
        filtered.pivot_table(
            index=["sample_key", "event_name", "bucket_group"],
            columns="model_short",
            values="time_abs_err_s",
            aggfunc="first",
        )
        .reset_index()
    )
    paired["delta_conditioned_minus_baseline"] = paired["conditioned"] - paired["baseline"]

    paired_rows: list[dict[str, object]] = []
    for (event_name, bucket_group), part in paired.groupby(["event_name", "bucket_group"], dropna=False):
        stats = summarize_metric(part["delta_conditioned_minus_baseline"])
        paired_rows.append(
            {
                "section": "event_time_conditioned_minus_baseline",
                "event_name": event_name,
                "bucket_group": bucket_group,
                **stats,
            }
        )

    paired_df = (
        pd.DataFrame(paired_rows)
        .sort_values(["event_name", "bucket_group"])
        .reset_index(drop=True)
    )
    return summary_df, paired_df


def render_report(
    master: pd.DataFrame,
    q1_compare: pd.DataFrame,
    corr_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    boundary_cross: pd.DataFrame,
    subject_boundary: pd.DataFrame,
    event_summary: pd.DataFrame,
    event_paired: pd.DataFrame,
) -> str:
    q1_count = int(master["is_q1_fast"].sum())
    non_q1_count = int((~master["is_q1_fast"]).sum())
    q1_tail_mean = float(
        q1_compare.query("group == 'Q1_fast' and metric == 'delta_rmse_tail_abs_steer'")["mean"].iloc[0]
    )
    non_q1_tail_mean = float(
        q1_compare.query("group == 'non_Q1_fast' and metric == 'delta_rmse_tail_abs_steer'")["mean"].iloc[0]
    )
    q1_boundary_mean = float(
        q1_compare.query("group == 'Q1_fast' and metric == 'delta_boundary_shift_abs_err'")["mean"].iloc[0]
    )
    non_q1_boundary_mean = float(
        q1_compare.query("group == 'non_Q1_fast' and metric == 'delta_boundary_shift_abs_err'")["mean"].iloc[0]
    )
    q1_peak_delta_mean = float(
        q1_compare.query("group == 'Q1_fast' and metric == 'delta_peak_time_abs_err_s'")["mean"].iloc[0]
    )
    non_q1_peak_delta_mean = float(
        q1_compare.query("group == 'non_Q1_fast' and metric == 'delta_peak_time_abs_err_s'")["mean"].iloc[0]
    )

    top_delta_corr = corr_df[corr_df["metric"].isin(DELTA_STRUCTURE_METRICS)].head(3).copy()
    top_extended_corr = corr_df[~corr_df["metric"].isin(DELTA_STRUCTURE_METRICS)].head(5).copy()

    boundary_mean_pivot = (
        boundary_cross.pivot(index="eval_morphology_label", columns="subj", values="mean_delta_boundary_shift_abs_err")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    boundary_rate_pivot = (
        boundary_cross.pivot(index="eval_morphology_label", columns="subj", values="worsen_rate")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    top_boundary = boundary_cross.iloc[0]
    second_boundary = boundary_cross.iloc[1]

    subject_delta = subject_boundary[subject_boundary["variant"] == "delta"].copy()
    subject_base = subject_boundary[subject_boundary["variant"] == "baseline"].copy()
    subject_cond = subject_boundary[subject_boundary["variant"] == "conditioned"].copy()
    subject_compare = (
        subject_base[["subj", "mean", "median", "q90"]]
        .rename(columns={"mean": "baseline_mean", "median": "baseline_median", "q90": "baseline_q90"})
        .merge(
            subject_cond[["subj", "mean", "median", "q90"]].rename(
                columns={"mean": "conditioned_mean", "median": "conditioned_median", "q90": "conditioned_q90"}
            ),
            on="subj",
            how="left",
        )
        .merge(
            subject_delta[["subj", "mean", "median", "q90", "worsen_rate"]].rename(
                columns={
                    "mean": "delta_mean",
                    "median": "delta_median",
                    "q90": "delta_q90",
                    "worsen_rate": "delta_worsen_rate",
                }
            ),
            on="subj",
            how="left",
        )
        .sort_values("delta_mean", ascending=False)
        .reset_index(drop=True)
    )

    event_display = event_summary.copy()
    paired_display = event_paired.copy()

    lines: list[str] = []
    lines.append(f"# conditioned v2 快反应退化与 boundary 恶化专题归因报告（{RUN_DATE}）")
    lines.append("")
    lines.append("## 数据与口径")
    lines.append(f"- 主表：`{MASTER_CSV}`，共 {len(master)} 条 sample-level 记录。")
    lines.append(f"- 事件表：`{EVENT_CSV}`，共 {len(event_summary) * 0 + pd.read_csv(EVENT_CSV).shape[0]} 条 event-level 记录。")
    lines.append(f"- `Q1_fast` 直接使用字段 `latency_proxy_bucket`；本次样本数为 {q1_count}，非 `Q1_fast` 为 {non_q1_count}。")
    lines.append("- event-level 中将 `unconditional_baseline` 记为 baseline，将 `event_conditioned_baseline` 记为 conditioned。")
    lines.append("")
    lines.append("## 结论摘要")
    lines.append(
        f"1. `Q1_fast` 上的 `delta_rmse_tail_abs_steer` 均值为 {fmt_num(q1_tail_mean)}，明显差于非 `Q1_fast` 的 {fmt_num(non_q1_tail_mean)}，确认快反应桶确实存在 tail 退化。"
    )
    lines.append(
        f"2. 但 `Q1_fast` 的 `delta_boundary_shift_abs_err` 均值只有 {fmt_num(q1_boundary_mean)}，低于非 `Q1_fast` 的 {fmt_num(non_q1_boundary_mean)}；`delta_peak_time_abs_err_s` 也没有转成明显更差（{fmt_num(q1_peak_delta_mean)} vs {fmt_num(non_q1_peak_delta_mean)}）。因此，`Q1_fast` 的主要问题并不是更强的 boundary 或 peak timing 恶化。"
    )
    if not top_delta_corr.empty:
        first_delta = top_delta_corr.iloc[0]
        lines.append(
            f"3. 在 `Q1_fast` 且 `delta_rmse_tail_abs_steer > 0` 的样本里，若只看本任务要求的结构性 delta 指标，相关性最高的是 `{first_delta['metric']}`（Pearson r={fmt_num(first_delta['pearson_corr'], 3)}）；`delta_boundary_shift_abs_err` 的相关性接近 0，说明快反应退化并非 boundary 驱动。"
        )
    if not top_extended_corr.empty:
        first_extended = top_extended_corr.iloc[0]
        second_extended = top_extended_corr.iloc[1]
        lines.append(
            f"4. 若扩展到 conditioned 结构指标，最强信号来自 `{first_extended['metric']}`（|r|={fmt_num(first_extended['abs_pearson_corr'], 3)}）和 `{second_extended['metric']}`（|r|={fmt_num(second_extended['abs_pearson_corr'], 3)}），对应的是 shape / amplitude 失配，而不是 boundary 漂移。"
        )
    lines.append(
        f"5. `boundary_shift` 恶化主要集中在 morphology，而不是单一 subject：`{top_boundary['eval_morphology_label']} × {top_boundary['subj']}` 的均值最高（{fmt_num(top_boundary['mean_delta_boundary_shift_abs_err'])}），其次是 `{second_boundary['eval_morphology_label']} × {second_boundary['subj']}`（{fmt_num(second_boundary['mean_delta_boundary_shift_abs_err'])}）；同时三位被试在 `single_lobe` 与 `reverse_correction` 上都为正。"
    )
    main_peak_q1 = paired_display.query("event_name == 'main_peak' and bucket_group == 'Q1_fast'")["mean"].iloc[0]
    main_peak_non = paired_display.query("event_name == 'main_peak' and bucket_group == 'non_Q1_fast'")["mean"].iloc[0]
    onset_q1 = paired_display.query("event_name == 'first_major_turn_onset' and bucket_group == 'Q1_fast'")["mean"].iloc[0]
    onset_non = paired_display.query("event_name == 'first_major_turn_onset' and bucket_group == 'non_Q1_fast'")["mean"].iloc[0]
    lines.append(
        f"6. event-level 上，`Q1_fast` 没有出现 conditioned 相对 baseline 的额外时间对齐惩罚：`first_major_turn_onset` 的 conditioned-baseline 均值差为 {fmt_num(onset_q1)}，`main_peak` 为 {fmt_num(main_peak_q1)}；其中 `main_peak` 在 `Q1_fast` 仍是轻微改善而非恶化。因此，快反应退化更像尾段 shape / amplitude 的问题，而不是 conditioned 带来的事件时间对齐系统性变差。"
    )
    lines.append("")
    lines.append("## 1. Q1_fast vs 非 Q1_fast 关键指标对比")
    q1_display = q1_compare.copy()
    for column in ["mean", "median", "std", "q75", "q90", "worsen_rate"]:
        q1_display[column] = q1_display[column].map(lambda x: round(float(x), 6) if pd.notna(x) else np.nan)
    lines.append(make_markdown_table(q1_display, ["group", "metric", "count", "mean", "median", "std", "q75", "q90", "worsen_rate"]))
    lines.append("")
    lines.append(
        "解读：`Q1_fast` 的 tail 指标是唯一明显转正的恶化项；而 `delta_boundary_shift_abs_err` 在非 `Q1_fast` 更大，`delta_turning_count_abs_err` 在 `Q1_fast` 反而更负，说明 turning count 不是主要问题。"
    )
    lines.append("")
    lines.append("## 2. Q1_fast 中 tail 恶化样本的结构指标排序")
    corr_display = corr_df.copy()
    for column in ["pearson_corr", "abs_pearson_corr", "mean_metric", "median_metric"]:
        corr_display[column] = corr_display[column].map(lambda x: round(float(x), 6) if pd.notna(x) else np.nan)
    lines.append("### 2.1 只看结构性 delta 指标")
    lines.append(make_markdown_table(corr_display[corr_display["metric"].isin(DELTA_STRUCTURE_METRICS)].head(4), ["metric", "n", "pearson_corr", "abs_pearson_corr", "mean_metric", "median_metric"]))
    lines.append("")
    lines.append("### 2.2 扩展到 conditioned 结构指标")
    lines.append(make_markdown_table(corr_display[~corr_display["metric"].isin(DELTA_STRUCTURE_METRICS)].head(6), ["metric", "n", "pearson_corr", "abs_pearson_corr", "mean_metric", "median_metric"]))
    lines.append("")
    diff_display = diff_df.copy()
    for column in [
        "worsened_mean",
        "improved_mean",
        "mean_diff",
        "worsened_median",
        "improved_median",
        "median_diff",
        "effect_size_smd",
    ]:
        diff_display[column] = diff_display[column].map(lambda x: round(float(x), 6) if pd.notna(x) else np.nan)
    lines.append("### 2.3 Q1_fast 内部：tail 恶化 vs 未恶化 的条件均值差")
    lines.append(make_markdown_table(diff_display.head(8), ["metric", "worsened_mean", "improved_mean", "mean_diff", "effect_size_smd"]))
    lines.append("")
    lines.append(
        "解读：若限定在任务要求的 delta 指标里，`delta_peak_time_abs_err_s` 与 `delta_turning_count_abs_err` 的相关性高于 `delta_boundary_shift_abs_err`，但绝对值都不大；真正更强的伴随信号是 `shape_corr_conditioned` 下降和 `peak_abs_amp_err_conditioned` 升高。"
    )
    lines.append("")
    lines.append("## 3. `eval_morphology_label × subj` 的 boundary_shift 恶化交叉表")
    boundary_mean_display = boundary_mean_pivot.copy()
    for col in [c for c in boundary_mean_display.columns if c != "eval_morphology_label"]:
        boundary_mean_display[col] = boundary_mean_display[col].map(lambda x: round(float(x), 6) if pd.notna(x) else np.nan)
    lines.append("### 3.1 conditioned minus baseline 的均值")
    lines.append(make_markdown_table(boundary_mean_display, ["eval_morphology_label", "cwh", "gf", "tyy"]))
    lines.append("")
    boundary_rate_display = boundary_rate_pivot.copy()
    for col in [c for c in boundary_rate_display.columns if c != "eval_morphology_label"]:
        boundary_rate_display[col] = boundary_rate_display[col].map(lambda x: round(float(x), 6) if pd.notna(x) else np.nan)
    lines.append("### 3.2 恶化率 `P(delta_boundary_shift_abs_err > 0)`")
    lines.append(make_markdown_table(boundary_rate_display, ["eval_morphology_label", "cwh", "gf", "tyy"]))
    lines.append("")
    lines.append(
        "解读：恶化不是只锁定在单一 subject。更强的共性是 morphology：`single_lobe` 在三位被试上都最差，`reverse_correction` 次之，`multi_correction` 最轻。被试差异体现在幅度上，`cwh` 在 `reverse_correction` / `single_lobe` 上最重，`gf` 在 `single_lobe` 上也很突出。"
    )
    lines.append("")
    lines.append("## 4. `gf / cwh / tyy` 的 conditioned vs baseline `boundary_shift` 分布对比")
    subject_compare_display = subject_compare.copy()
    for column in [c for c in subject_compare_display.columns if c != "subj"]:
        subject_compare_display[column] = subject_compare_display[column].map(
            lambda x: round(float(x), 6) if pd.notna(x) else np.nan
        )
    lines.append(make_markdown_table(subject_compare_display, ["subj", "baseline_mean", "conditioned_mean", "delta_mean", "baseline_median", "conditioned_median", "delta_median", "baseline_q90", "conditioned_q90", "delta_q90", "delta_worsen_rate"]))
    lines.append("")
    lines.append(
        "解读：三位被试的 `boundary_shift_abs_err_conditioned` 分布都整体右移。按 `delta_boundary_shift_abs_err` 均值看，`cwh` 最差，其次 `gf`，再到 `tyy`；这与背景里观察到的 subject heterogeneity 一致，但仍不是单一被试独占，因为三人都呈正向恶化。"
    )
    lines.append("")
    lines.append("## 5. event-level 时间对齐：`Q1_fast` vs 非 `Q1_fast`")
    event_display = event_display.copy()
    for column in ["mean", "median", "q75", "q90", "worsen_rate"]:
        event_display[column] = event_display[column].map(lambda x: round(float(x), 6) if pd.notna(x) else np.nan)
    lines.append("### 5.1 `time_abs_err_s` 分布")
    lines.append(make_markdown_table(event_display, ["event_name", "model_name", "bucket_group", "count", "mean", "median", "q75", "q90"]))
    lines.append("")
    paired_display = paired_display.copy()
    for column in ["mean", "median", "q75", "q90", "worsen_rate"]:
        paired_display[column] = paired_display[column].map(lambda x: round(float(x), 6) if pd.notna(x) else np.nan)
    lines.append("### 5.2 同一样本上 `conditioned - baseline` 的时间误差差值")
    lines.append(make_markdown_table(paired_display, ["event_name", "bucket_group", "count", "mean", "median", "q75", "q90", "worsen_rate"]))
    lines.append("")
    lines.append(
        "解读：`first_major_turn_onset` 上 conditioned 对两类桶都是改善，且非 `Q1_fast` 的改善更强；`main_peak` 上 `Q1_fast` 仍是轻微改善，非 `Q1_fast` 则接近持平略差。因此 event-level 的时间对齐结果不支持“Q1_fast 因时间对齐更差而退化”的解释。"
    )
    lines.append("")
    lines.append("## 6. 归因收口与下一步建议")
    lines.append("- 归因事实 1：`Q1_fast` 的 tail 退化存在，但并不伴随更重的 `boundary_shift` 恶化，也不伴随 event-level 时间对齐系统性变差。")
    lines.append("- 归因事实 2：在 `Q1_fast` 内部，更强的伴随信号是 `shape_corr_conditioned` 下降和 `peak_abs_amp_err_conditioned` 上升，说明 tail 的 shape / amplitude 失真比 boundary 漂移更像主因。")
    lines.append("- 归因事实 3：`boundary_shift` 恶化更像 morphology 主导的共性问题，重点是 `single_lobe` 与 `reverse_correction`；subject 主要影响恶化幅度，而不是决定是否发生。")
    lines.append("- 推荐下一步：继续做只读切片，优先检查 `Q1_fast` 中高 `peak_abs_amp_err_conditioned` / 低 `shape_corr_conditioned` 的具体样本，确认它们是否集中在某些尾段幅值模式或反向修正强度区间。")
    lines.append("- 推荐下一步：对 `single_lobe` 与 `reverse_correction` 分别追加 trajectory 可视化抽样，验证 `boundary_shift` 恶化是由边界提前/滞后，还是由边界附近幅值不足引起。")
    return "\n".join(lines) + "\n"


def main() -> None:
    master = pd.read_csv(MASTER_CSV)
    event = pd.read_csv(EVENT_CSV)

    require_columns(
        master,
        [
            "sample_key",
            "subj",
            "eval_morphology_label",
            LATENCY_BUCKET_COL,
            "boundary_shift_abs_err_baseline",
            "boundary_shift_abs_err_conditioned",
            *Q1_COMPARE_METRICS,
            *DELTA_STRUCTURE_METRICS,
            *EXTENDED_STRUCTURE_METRICS,
        ],
        MASTER_CSV,
    )
    require_columns(
        event,
        ["sample_key", "model_name", "event_name", "time_abs_err_s"],
        EVENT_CSV,
    )

    master = master.copy()
    master["is_q1_fast"] = master[LATENCY_BUCKET_COL].eq(Q1_BUCKET)

    q1_compare = build_q1_compare(master)
    corr_df, diff_df = build_q1_tail_structure_rankings(master)
    boundary_cross = build_boundary_cross(master)
    subject_boundary = build_subject_boundary_distribution(master)
    event_summary, event_paired = build_event_summaries(master, event)

    q1_summary_csv = pd.concat([q1_compare, corr_df, diff_df], ignore_index=True, sort=False)
    boundary_event_csv = pd.concat([boundary_cross, subject_boundary, event_summary, event_paired], ignore_index=True, sort=False)

    report_text = render_report(
        master=master,
        q1_compare=q1_compare,
        corr_df=corr_df,
        diff_df=diff_df,
        boundary_cross=boundary_cross,
        subject_boundary=subject_boundary,
        event_summary=event_summary,
        event_paired=event_paired,
    )

    Q1_SUMMARY_CSV.write_text(q1_summary_csv.to_csv(index=False), encoding="utf-8")
    BOUNDARY_EVENT_CSV.write_text(boundary_event_csv.to_csv(index=False), encoding="utf-8")
    REPORT_MD.write_text(report_text, encoding="utf-8")

    print(f"[OK] wrote {REPORT_MD}")
    print(f"[OK] wrote {Q1_SUMMARY_CSV}")
    print(f"[OK] wrote {BOUNDARY_EVENT_CSV}")


if __name__ == "__main__":
    main()
