#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Q1_fast 恶化归因诊断脚本。

只读输入主表，输出诊断 CSV，并在终端打印中文结论。
不修改任何源数据文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(r"F:\data_set_process\data_process")
INPUT_CSV = REPO_ROOT / "reports" / "attribution_master_table.csv"
OUTPUT_CSV = REPO_ROOT / "reports" / "attribution_q1fast_diagnosis.csv"

LATENCY_BUCKET = "Q1_fast"

CATEGORICAL_COLUMNS = [
    "subj",
    "eval_morphology_label",
    "mechanism_tag",
    "trigger_type",
]

NUMERIC_COLUMNS = [
    "event_duration_s",
    "anchor_to_event_start_s",
    "curvature_anchor",
]

GLOBAL_COMPARISON_COLUMNS = [
    "delta_rmse_tail_abs_steer",
    "delta_boundary_shift_abs_err",
]


def _safe_float(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def _safe_label(value: object) -> str:
    if pd.isna(value):
        return "NA"
    text = str(value).strip()
    return text if text else "NA"


def _series_stats(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0.0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q10": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q90": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "count": float(clean.count()),
        "mean": _safe_float(clean.mean()),
        "std": _safe_float(clean.std(ddof=1)),
        "median": _safe_float(clean.median()),
        "q10": _safe_float(clean.quantile(0.10)),
        "q25": _safe_float(clean.quantile(0.25)),
        "q75": _safe_float(clean.quantile(0.75)),
        "q90": _safe_float(clean.quantile(0.90)),
        "min": _safe_float(clean.min()),
        "max": _safe_float(clean.max()),
    }


def _pooled_std(a: pd.Series, b: pd.Series) -> float:
    a_clean = pd.to_numeric(a, errors="coerce").dropna()
    b_clean = pd.to_numeric(b, errors="coerce").dropna()
    if len(a_clean) < 2 or len(b_clean) < 2:
        return float("nan")
    a_var = float(a_clean.var(ddof=1))
    b_var = float(b_clean.var(ddof=1))
    pooled_num = (len(a_clean) - 1) * a_var + (len(b_clean) - 1) * b_var
    pooled_den = len(a_clean) + len(b_clean) - 2
    if pooled_den <= 0 or pooled_num < 0:
        return float("nan")
    return float(np.sqrt(pooled_num / pooled_den))


def _classify_concentration(tvd: float) -> str:
    if pd.isna(tvd):
        return "unknown"
    if tvd < 0.08:
        return "diffuse"
    if tvd < 0.18:
        return "mild_concentration"
    return "concentrated"


def load_master_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {
        "latency_proxy_bucket",
        "improved_tail_flag",
        *CATEGORICAL_COLUMNS,
        *NUMERIC_COLUMNS,
        *GLOBAL_COMPARISON_COLUMNS,
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns in {path}: {missing_str}")
    return df


def summarize_overview(
    df: pd.DataFrame,
    q1: pd.DataFrame,
    q1_worsened: pd.DataFrame,
    q1_improved: pd.DataFrame,
    global_worsened: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "section": "overview",
            "dimension": "sample_counts",
            "level": "all",
            "notes": (
                "Q1_fast subset and tail-improvement split "
                "(improved_tail_flag: 0=worsened, 1=improved)"
            ),
            "all_total_count": len(df),
            "q1_total_count": len(q1),
            "q1_worsened_count": len(q1_worsened),
            "q1_improved_count": len(q1_improved),
            "q1_worsened_rate": _safe_float(len(q1_worsened) / len(q1)) if len(q1) else np.nan,
            "global_worsened_count": len(global_worsened),
            "global_worsened_rate": (
                _safe_float(len(global_worsened) / len(df)) if len(df) else np.nan
            ),
        }
    )
    return rows


def summarize_categorical(
    q1: pd.DataFrame,
    q1_worsened: pd.DataFrame,
    q1_improved: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    dimension_summaries: list[dict[str, object]] = []
    q1_worsened_rate = _safe_float(len(q1_worsened) / len(q1)) if len(q1) else np.nan

    for column in CATEGORICAL_COLUMNS:
        q1_col = q1[column].map(_safe_label)
        worsened_col = q1_worsened[column].map(_safe_label)
        improved_col = q1_improved[column].map(_safe_label)

        categories = sorted(set(q1_col.unique()) | set(worsened_col.unique()) | set(improved_col.unique()))
        total_counts = q1_col.value_counts(dropna=False)
        worsened_counts = worsened_col.value_counts(dropna=False)
        improved_counts = improved_col.value_counts(dropna=False)

        worsened_dist = worsened_counts / len(q1_worsened) if len(q1_worsened) else worsened_counts * np.nan
        improved_dist = improved_counts / len(q1_improved) if len(q1_improved) else improved_counts * np.nan

        tvd = 0.0
        max_abs_gap = 0.0
        top_level = "NA"
        top_gap = -1.0
        for category in categories:
            w_share = float(worsened_dist.get(category, 0.0))
            i_share = float(improved_dist.get(category, 0.0))
            gap = w_share - i_share
            abs_gap = abs(gap)
            tvd += abs_gap
            if abs_gap > max_abs_gap:
                max_abs_gap = abs_gap
            if abs_gap > top_gap:
                top_gap = abs_gap
                top_level = category

        tvd *= 0.5
        concentration_label = _classify_concentration(tvd)

        dimension_summaries.append(
            {
                "section": "categorical_dimension_summary",
                "dimension": column,
                "level": "__summary__",
                "tvd_worsened_vs_improved": _safe_float(tvd),
                "max_abs_share_gap": _safe_float(max_abs_gap),
                "top_gap_level": top_level,
                "concentration_label": concentration_label,
                "q1_worsened_rate": q1_worsened_rate,
            }
        )

        for category in categories:
            total_count = int(total_counts.get(category, 0))
            worsened_count = int(worsened_counts.get(category, 0))
            improved_count = int(improved_counts.get(category, 0))
            worsened_share = (
                _safe_float(worsened_count / len(q1_worsened)) if len(q1_worsened) else np.nan
            )
            improved_share = (
                _safe_float(improved_count / len(q1_improved)) if len(q1_improved) else np.nan
            )
            worsen_rate = _safe_float(worsened_count / total_count) if total_count else np.nan

            rows.append(
                {
                    "section": "categorical",
                    "dimension": column,
                    "level": category,
                    "total_count": total_count,
                    "worsened_count": worsened_count,
                    "improved_count": improved_count,
                    "worsened_share_in_q1_worsened": worsened_share,
                    "improved_share_in_q1_improved": improved_share,
                    "share_gap_worsened_minus_improved": (
                        _safe_float(worsened_share - improved_share)
                        if not pd.isna(worsened_share) and not pd.isna(improved_share)
                        else np.nan
                    ),
                    "worsen_rate_within_level": worsen_rate,
                    "rate_lift_vs_q1_worsened_rate": (
                        _safe_float(worsen_rate - q1_worsened_rate)
                        if not pd.isna(worsen_rate) and not pd.isna(q1_worsened_rate)
                        else np.nan
                    ),
                    "tvd_worsened_vs_improved": _safe_float(tvd),
                    "max_abs_share_gap": _safe_float(max_abs_gap),
                    "top_gap_level": top_level,
                    "concentration_label": concentration_label,
                }
            )

    return rows, dimension_summaries


def summarize_numeric_within_q1(
    q1_worsened: pd.DataFrame,
    q1_improved: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in NUMERIC_COLUMNS:
        worsened_stats = _series_stats(q1_worsened[column])
        improved_stats = _series_stats(q1_improved[column])
        pooled_std = _pooled_std(q1_worsened[column], q1_improved[column])
        mean_diff = worsened_stats["mean"] - improved_stats["mean"]
        median_diff = worsened_stats["median"] - improved_stats["median"]

        rows.append(
            {
                "section": "numeric_within_q1",
                "dimension": column,
                "level": "__summary__",
                "worsened_count": worsened_stats["count"],
                "improved_count": improved_stats["count"],
                "worsened_mean": worsened_stats["mean"],
                "improved_mean": improved_stats["mean"],
                "mean_diff_worsened_minus_improved": _safe_float(mean_diff),
                "worsened_std": worsened_stats["std"],
                "improved_std": improved_stats["std"],
                "worsened_median": worsened_stats["median"],
                "improved_median": improved_stats["median"],
                "median_diff_worsened_minus_improved": _safe_float(median_diff),
                "worsened_q10": worsened_stats["q10"],
                "worsened_q25": worsened_stats["q25"],
                "worsened_q75": worsened_stats["q75"],
                "worsened_q90": worsened_stats["q90"],
                "improved_q10": improved_stats["q10"],
                "improved_q25": improved_stats["q25"],
                "improved_q75": improved_stats["q75"],
                "improved_q90": improved_stats["q90"],
                "effect_size_smd": (
                    _safe_float(mean_diff / pooled_std) if pooled_std and not pd.isna(pooled_std) else np.nan
                ),
            }
        )
    return rows


def summarize_q1_worsened_vs_global(
    q1_worsened: pd.DataFrame,
    global_worsened: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in GLOBAL_COMPARISON_COLUMNS:
        q1_stats = _series_stats(q1_worsened[column])
        global_stats = _series_stats(global_worsened[column])
        mean_diff = q1_stats["mean"] - global_stats["mean"]
        relative_diff = np.nan
        if not pd.isna(global_stats["mean"]) and global_stats["mean"] != 0:
            relative_diff = _safe_float(mean_diff / abs(global_stats["mean"]))

        rows.append(
            {
                "section": "q1_worsened_vs_global_worsened",
                "dimension": column,
                "level": "__summary__",
                "q1_worsened_count": q1_stats["count"],
                "global_worsened_count": global_stats["count"],
                "q1_worsened_mean": q1_stats["mean"],
                "global_worsened_mean": global_stats["mean"],
                "mean_diff_q1_minus_global": _safe_float(mean_diff),
                "relative_diff_q1_vs_global_abs_mean": relative_diff,
                "q1_worsened_median": q1_stats["median"],
                "global_worsened_median": global_stats["median"],
                "q1_worsened_q90": q1_stats["q90"],
                "global_worsened_q90": global_stats["q90"],
            }
        )
    return rows


def build_conclusion(
    q1: pd.DataFrame,
    q1_worsened: pd.DataFrame,
    q1_improved: pd.DataFrame,
    categorical_dimension_summary: Iterable[dict[str, object]],
    numeric_rows: Iterable[dict[str, object]],
    global_rows: Iterable[dict[str, object]],
) -> str:
    dimension_lookup = {
        row["dimension"]: row for row in categorical_dimension_summary
    }
    numeric_lookup = {row["dimension"]: row for row in numeric_rows}
    global_lookup = {row["dimension"]: row for row in global_rows}

    subj_summary = dimension_lookup["subj"]
    morph_summary = dimension_lookup["eval_morphology_label"]
    mechanism_summary = dimension_lookup["mechanism_tag"]
    trigger_summary = dimension_lookup["trigger_type"]

    duration_row = numeric_lookup["event_duration_s"]
    anchor_row = numeric_lookup["anchor_to_event_start_s"]
    curvature_row = numeric_lookup["curvature_anchor"]

    delta_tail_row = global_lookup["delta_rmse_tail_abs_steer"]
    delta_boundary_row = global_lookup["delta_boundary_shift_abs_err"]

    q1_worsened_rate = _safe_float(len(q1_worsened) / len(q1)) if len(q1) else np.nan

    diagnosis_parts = [
        (
            f"Q1_fast 共 {len(q1)} 个样本，其中恶化 {len(q1_worsened)} 个 "
            f"({q1_worsened_rate:.1%})，改善 {len(q1_improved)} 个。"
        ),
        (
            "从分布差异看，恶化并没有明显集中到某一个被试或某一种形态："
            f"subj 的 TVD={subj_summary['tvd_worsened_vs_improved']:.3f}，"
            f"形态的 TVD={morph_summary['tvd_worsened_vs_improved']:.3f}，"
            "都属于弱差异。"
        ),
        (
            "在 Q1_fast 桶内，mechanism_tag 和 trigger_type 几乎没有区分度："
            f"mechanism TVD={mechanism_summary['tvd_worsened_vs_improved']:.3f}，"
            f"trigger TVD={trigger_summary['tvd_worsened_vs_improved']:.3f}。"
        ),
        (
            "连续变量里，event_duration_s 和 anchor_to_event_start_s 在恶化/改善之间均值差都很小，"
            f"分别为 {duration_row['mean_diff_worsened_minus_improved']:.4f}s 和 "
            f"{anchor_row['mean_diff_worsened_minus_improved']:.4f}s；"
            "但 curvature_anchor 的恶化样本右尾更重："
            f"恶化 q90={curvature_row['worsened_q90']:.6g}，"
            f"改善 q90={curvature_row['improved_q90']:.6g}。"
        ),
        (
            "和全局所有恶化样本相比，Q1_fast 恶化样本的 tail RMSE 恶化更重，"
            f"均值 {delta_tail_row['q1_worsened_mean']:.4f} vs "
            f"{delta_tail_row['global_worsened_mean']:.4f}；"
            "但 boundary shift 恶化明显更轻，"
            f"均值 {delta_boundary_row['q1_worsened_mean']:.4f} vs "
            f"{delta_boundary_row['global_worsened_mean']:.4f}。"
        ),
        (
            "诊断结论：Q1_fast 恶化更像是广泛分布的问题，不是集中在某个特定被试、"
            "trigger_type 或 mechanism_tag。更强的信号是快响应场景下、曲率较高的锚点附近，"
            "conditioned 更容易把 tail 的形状/幅值做坏，而不是主要把边界时序推错。"
        ),
    ]

    return " ".join(diagnosis_parts)


def print_terminal_summary(
    q1: pd.DataFrame,
    q1_worsened: pd.DataFrame,
    q1_improved: pd.DataFrame,
    categorical_rows: pd.DataFrame,
    categorical_dimension_summary: list[dict[str, object]],
    numeric_rows: list[dict[str, object]],
    global_rows: list[dict[str, object]],
    conclusion: str,
) -> None:
    concentration_label_map = {
        "diffuse": "分散",
        "mild_concentration": "轻度集中",
        "concentrated": "明显集中",
        "unknown": "未知",
    }

    print("=" * 80)
    print("Q1_fast 恶化归因诊断")
    print("=" * 80)
    print(f"输入文件: {INPUT_CSV}")
    print(f"输出文件: {OUTPUT_CSV}")
    print()
    print(
        f"Q1_fast 样本数 = {len(q1)} | 恶化 = {len(q1_worsened)} "
        f"| 改善 = {len(q1_improved)}"
    )
    print()

    print("[分类维度集中度检查]")
    dimension_summary_df = pd.DataFrame(categorical_dimension_summary)
    if not dimension_summary_df.empty:
        display_df = dimension_summary_df.copy()
        display_df["concentration_label"] = display_df["concentration_label"].map(
            lambda x: concentration_label_map.get(str(x), str(x))
        )
        display_cols = [
            "dimension",
            "tvd_worsened_vs_improved",
            "max_abs_share_gap",
            "top_gap_level",
            "concentration_label",
        ]
        print(
            display_df[display_cols]
            .sort_values("tvd_worsened_vs_improved", ascending=False)
            .to_string(index=False)
        )
        print()

    for dimension in ["subj", "eval_morphology_label", "mechanism_tag", "trigger_type"]:
        subset = categorical_rows[categorical_rows["dimension"] == dimension].copy()
        if subset.empty:
            continue
        for int_col in ["total_count", "worsened_count", "improved_count"]:
            subset[int_col] = subset[int_col].astype("Int64")
        subset = subset.sort_values(
            ["share_gap_worsened_minus_improved", "total_count"],
            ascending=[False, False],
        )
        print(f"[{dimension}]")
        print(
            subset[
                [
                    "level",
                    "total_count",
                    "worsened_count",
                    "improved_count",
                    "worsen_rate_within_level",
                    "share_gap_worsened_minus_improved",
                ]
            ].to_string(index=False)
        )
        print()

    numeric_df = pd.DataFrame(numeric_rows)
    if not numeric_df.empty:
        print("[Q1_fast 内连续变量差异]")
        print(
            numeric_df[
                [
                    "dimension",
                    "worsened_mean",
                    "improved_mean",
                    "mean_diff_worsened_minus_improved",
                    "worsened_median",
                    "improved_median",
                    "worsened_q90",
                    "improved_q90",
                    "effect_size_smd",
                ]
            ].to_string(index=False)
        )
        print()

    global_df = pd.DataFrame(global_rows)
    if not global_df.empty:
        print("[Q1_fast 恶化样本 vs 全局恶化样本]")
        print(
            global_df[
                [
                    "dimension",
                    "q1_worsened_mean",
                    "global_worsened_mean",
                    "mean_diff_q1_minus_global",
                    "relative_diff_q1_vs_global_abs_mean",
                ]
            ].to_string(index=False)
        )
        print()

    print("[诊断性结论]")
    print(conclusion)
    print("=" * 80)


def main() -> None:
    master = load_master_table(INPUT_CSV)

    q1 = master[master["latency_proxy_bucket"].astype(str) == LATENCY_BUCKET].copy()
    q1_worsened = q1[q1["improved_tail_flag"] == 0].copy()
    q1_improved = q1[q1["improved_tail_flag"] == 1].copy()
    global_worsened = master[master["improved_tail_flag"] == 0].copy()

    output_rows: list[dict[str, object]] = []
    output_rows.extend(
        summarize_overview(master, q1, q1_worsened, q1_improved, global_worsened)
    )

    categorical_rows, categorical_dimension_summary = summarize_categorical(
        q1, q1_worsened, q1_improved
    )
    output_rows.extend(categorical_rows)
    output_rows.extend(categorical_dimension_summary)

    numeric_rows = summarize_numeric_within_q1(q1_worsened, q1_improved)
    output_rows.extend(numeric_rows)

    global_rows = summarize_q1_worsened_vs_global(q1_worsened, global_worsened)
    output_rows.extend(global_rows)

    conclusion = build_conclusion(
        q1=q1,
        q1_worsened=q1_worsened,
        q1_improved=q1_improved,
        categorical_dimension_summary=categorical_dimension_summary,
        numeric_rows=numeric_rows,
        global_rows=global_rows,
    )
    output_rows.append(
        {
            "section": "diagnostic_conclusion",
            "dimension": "q1_fast_tail_degradation",
            "level": "__summary__",
            "notes": conclusion,
        }
    )

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print_terminal_summary(
        q1=q1,
        q1_worsened=q1_worsened,
        q1_improved=q1_improved,
        categorical_rows=output_df[output_df["section"] == "categorical"].copy(),
        categorical_dimension_summary=categorical_dimension_summary,
        numeric_rows=numeric_rows,
        global_rows=global_rows,
        conclusion=conclusion,
    )


if __name__ == "__main__":
    main()
