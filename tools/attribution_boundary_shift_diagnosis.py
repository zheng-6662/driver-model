#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
attribution_boundary_shift_diagnosis.py
======================================
只读诊断脚本：专题分析 boundary_shift 误差恶化样本。

输入：
  - reports/attribution_master_table.csv

输出：
  - reports/attribution_boundary_shift_diagnosis.csv
  - 终端中文结论打印

分析内容：
  1. 筛选 delta_boundary_shift_abs_err > 0 的恶化样本
  2. 按 subj × eval_morphology_label 交叉统计恶化数量与恶化幅度
  3. 按 trigger_type × mechanism_tag 交叉统计同上
  4. 检查恶化样本中 delta_rmse_tail_abs_steer 是否也恶化
  5. 输出“是否由某个被试/形态/触发类型主导”的诊断结论
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(r"F:\data_set_process\data_process")
INPUT_CSV = REPO_ROOT / "reports" / "attribution_master_table.csv"
OUTPUT_CSV = REPO_ROOT / "reports" / "attribution_boundary_shift_diagnosis.csv"

REQUIRED_COLUMNS = [
    "sample_key",
    "subj",
    "eval_morphology_label",
    "trigger_type",
    "mechanism_tag",
    "delta_boundary_shift_abs_err",
    "delta_rmse_tail_abs_steer",
]


def validate_input(df: pd.DataFrame) -> None:
    """检查输入表是否包含诊断所需字段。"""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print("[ERROR] 输入表缺少以下必需列：")
        for col in missing:
            print(f"  - {col}")
        sys.exit(1)


def group_diagnosis(
    df_all: pd.DataFrame,
    worsening_df: pd.DataFrame,
    group_cols: list[str],
    section_name: str,
) -> pd.DataFrame:
    """统计指定分组下的恶化数量、恶化比例、恶化幅度，以及 tail 同步恶化情况。"""
    total = (
        df_all.groupby(group_cols, dropna=False)
        .size()
        .rename("total_samples")
        .reset_index()
    )

    if worsening_df.empty:
        result = total.copy()
        result["worsening_count"] = 0
        result["worsening_ratio_within_group"] = 0.0
        result["worsening_share_among_all_worsening"] = 0.0
        result["mean_delta_boundary_shift_abs_err"] = pd.NA
        result["median_delta_boundary_shift_abs_err"] = pd.NA
        result["mean_delta_rmse_tail_abs_steer"] = pd.NA
        result["tail_worsening_count"] = 0
        result["tail_worsening_ratio_within_worsening"] = pd.NA
        result["section"] = section_name
        return result

    worsening = (
        worsening_df.groupby(group_cols, dropna=False)
        .agg(
            worsening_count=("sample_key", "size"),
            mean_delta_boundary_shift_abs_err=(
                "delta_boundary_shift_abs_err",
                "mean",
            ),
            median_delta_boundary_shift_abs_err=(
                "delta_boundary_shift_abs_err",
                "median",
            ),
            mean_delta_rmse_tail_abs_steer=(
                "delta_rmse_tail_abs_steer",
                "mean",
            ),
            tail_worsening_count=("tail_worsened_flag", "sum"),
            tail_worsening_ratio_within_worsening=(
                "tail_worsened_flag",
                "mean",
            ),
        )
        .reset_index()
    )

    result = total.merge(worsening, on=group_cols, how="left")
    result["worsening_count"] = result["worsening_count"].fillna(0).astype(int)
    result["tail_worsening_count"] = (
        result["tail_worsening_count"].fillna(0).astype(int)
    )
    result["worsening_ratio_within_group"] = (
        result["worsening_count"] / result["total_samples"]
    )
    result["worsening_share_among_all_worsening"] = (
        result["worsening_count"] / len(worsening_df)
    )
    result["section"] = section_name

    result = result.sort_values(
        by=[
            "worsening_count",
            "mean_delta_boundary_shift_abs_err",
            "worsening_ratio_within_group",
        ],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return result


def build_overview(df_all: pd.DataFrame, worsening_df: pd.DataFrame) -> pd.DataFrame:
    """生成总体概览。"""
    total_samples = int(len(df_all))
    worsening_count = int(len(worsening_df))
    worsening_ratio = worsening_count / total_samples if total_samples else 0.0

    if worsening_count == 0:
        overview_row = {
            "section": "overview",
            "metric": "overall",
            "total_samples": total_samples,
            "worsening_count": worsening_count,
            "worsening_ratio": worsening_ratio,
            "mean_delta_boundary_shift_abs_err": pd.NA,
            "median_delta_boundary_shift_abs_err": pd.NA,
            "mean_delta_rmse_tail_abs_steer": pd.NA,
            "tail_worsening_count": 0,
            "tail_worsening_ratio_within_worsening": pd.NA,
        }
        return pd.DataFrame([overview_row])

    overview_row = {
        "section": "overview",
        "metric": "overall",
        "total_samples": total_samples,
        "worsening_count": worsening_count,
        "worsening_ratio": worsening_ratio,
        "mean_delta_boundary_shift_abs_err": worsening_df[
            "delta_boundary_shift_abs_err"
        ].mean(),
        "median_delta_boundary_shift_abs_err": worsening_df[
            "delta_boundary_shift_abs_err"
        ].median(),
        "mean_delta_rmse_tail_abs_steer": worsening_df[
            "delta_rmse_tail_abs_steer"
        ].mean(),
        "tail_worsening_count": int(worsening_df["tail_worsened_flag"].sum()),
        "tail_worsening_ratio_within_worsening": worsening_df[
            "tail_worsened_flag"
        ].mean(),
    }
    return pd.DataFrame([overview_row])


def build_tail_relation(df_all: pd.DataFrame) -> pd.DataFrame:
    """生成 boundary_shift 与 tail 恶化关系的四象限统计。"""
    temp = df_all.copy()
    temp["boundary_worsened_flag"] = temp["delta_boundary_shift_abs_err"] > 0
    temp["tail_worsened_flag"] = temp["delta_rmse_tail_abs_steer"] > 0
    temp["boundary_status"] = temp["boundary_worsened_flag"].map(
        {True: "boundary_shift恶化", False: "boundary_shift未恶化"}
    )
    temp["tail_status"] = temp["tail_worsened_flag"].map(
        {True: "tail也恶化", False: "tail未恶化"}
    )

    relation = (
        temp.groupby(["boundary_status", "tail_status"], dropna=False)
        .size()
        .rename("sample_count")
        .reset_index()
    )
    relation["sample_ratio"] = relation["sample_count"] / len(temp)
    relation["section"] = "tail_relation_matrix"
    relation = relation.sort_values(
        by=["boundary_status", "tail_status"], ascending=[True, True]
    ).reset_index(drop=True)
    return relation


def build_tail_summary(df_all: pd.DataFrame, worsening_df: pd.DataFrame) -> pd.DataFrame:
    """生成 tail 联动的摘要统计。"""
    boundary_worse_tail_worse_ratio = (
        worsening_df["tail_worsened_flag"].mean() if not worsening_df.empty else pd.NA
    )
    boundary_not_worse = df_all[df_all["delta_boundary_shift_abs_err"] <= 0].copy()
    boundary_not_worse_tail_worse_ratio = (
        boundary_not_worse["tail_worsened_flag"].mean()
        if not boundary_not_worse.empty
        else pd.NA
    )

    rows = [
        {
            "section": "tail_relation_summary",
            "metric": "boundary恶化样本中_tail也恶化比例",
            "value": boundary_worse_tail_worse_ratio,
        },
        {
            "section": "tail_relation_summary",
            "metric": "boundary恶化样本中_tail未恶化比例",
            "value": (
                1.0 - boundary_worse_tail_worse_ratio
                if pd.notna(boundary_worse_tail_worse_ratio)
                else pd.NA
            ),
        },
        {
            "section": "tail_relation_summary",
            "metric": "boundary未恶化样本中_tail也恶化比例",
            "value": boundary_not_worse_tail_worse_ratio,
        },
        {
            "section": "tail_relation_summary",
            "metric": "boundary恶化样本中_tail均值delta",
            "value": (
                worsening_df["delta_rmse_tail_abs_steer"].mean()
                if not worsening_df.empty
                else pd.NA
            ),
        },
        {
            "section": "tail_relation_summary",
            "metric": "boundary恶化样本中_tail中位数delta",
            "value": (
                worsening_df["delta_rmse_tail_abs_steer"].median()
                if not worsening_df.empty
                else pd.NA
            ),
        },
    ]
    return pd.DataFrame(rows)


def assess_dominance(
    group_df: pd.DataFrame,
    group_cols: list[str],
    dimension_name: str,
) -> pd.DataFrame:
    """
    对“是否由某组主导”给出结构化判断。

    判定口径：
      - 若只有 1 个非空组，则记为“无法比较（单组）”
      - 若最大组占全部恶化样本 >= 0.50，则记为“存在明显集中”
      - 否则记为“未见单一主导”
    """
    valid = group_df[group_df["worsening_count"] > 0].copy()
    if valid.empty:
        return pd.DataFrame(
            [
                {
                    "section": "dominance_assessment",
                    "dimension": dimension_name,
                    "top_group": "无恶化样本",
                    "top_worsening_share": 0.0,
                    "top_worsening_ratio_within_group": pd.NA,
                    "judgement": "无恶化样本",
                    "comment": "delta_boundary_shift_abs_err > 0 的样本数为 0。",
                }
            ]
        )

    top1 = valid.iloc[0]
    top_group = " | ".join([f"{col}={top1[col]}" for col in group_cols])
    unique_group_count = len(valid)
    top_share = float(top1["worsening_share_among_all_worsening"])
    top_ratio = float(top1["worsening_ratio_within_group"])

    if unique_group_count == 1:
        judgement = "无法比较（单组）"
        comment = (
            f"{dimension_name} 只有 1 个有恶化样本的分组，缺乏横向比较基础。"
        )
    elif top_share >= 0.50:
        judgement = "存在明显集中"
        comment = (
            f"头部分组占全部恶化样本 {top_share:.1%}，达到明显集中阈值。"
        )
    else:
        judgement = "未见单一主导"
        comment = (
            f"头部分组仅占全部恶化样本 {top_share:.1%}，未达到单一主导的集中度。"
        )

    return pd.DataFrame(
        [
            {
                "section": "dominance_assessment",
                "dimension": dimension_name,
                "top_group": top_group,
                "top_worsening_share": top_share,
                "top_worsening_ratio_within_group": top_ratio,
                "judgement": judgement,
                "comment": comment,
            }
        ]
    )


def format_group_label(row: pd.Series, group_cols: list[str]) -> str:
    """把分组列拼成便于终端展示的标签。"""
    return " | ".join([f"{col}={row[col]}" for col in group_cols])


def print_conclusion(
    df_all: pd.DataFrame,
    worsening_df: pd.DataFrame,
    subj_df: pd.DataFrame,
    morph_df: pd.DataFrame,
    subj_morph_df: pd.DataFrame,
    trigger_mech_df: pd.DataFrame,
) -> None:
    """打印中文诊断摘要和结论段。"""
    total_samples = len(df_all)
    worsening_count = len(worsening_df)
    worsening_ratio = worsening_count / total_samples if total_samples else 0.0

    print("=" * 88)
    print("boundary_shift 恶化专题诊断")
    print("=" * 88)
    print(f"输入文件: {INPUT_CSV}")
    print(f"输出文件: {OUTPUT_CSV}")
    print("-" * 88)
    print("一、总体概览")
    print(
        f"  - 总样本数: {total_samples}"
        f" | boundary_shift 恶化样本数: {worsening_count}"
        f" | 恶化占比: {worsening_ratio:.2%}"
    )

    if worsening_df.empty:
        print("  - 未发现 delta_boundary_shift_abs_err > 0 的样本。")
        print("-" * 88)
        print("结论：本批样本中未观察到 boundary_shift 恶化，无需进一步归因。")
        print("=" * 88)
        return

    tail_worse_count = int(worsening_df["tail_worsened_flag"].sum())
    tail_not_worse_count = worsening_count - tail_worse_count
    tail_worse_ratio = tail_worse_count / worsening_count
    tail_not_worse_ratio = tail_not_worse_count / worsening_count
    tail_mean_delta = worsening_df["delta_rmse_tail_abs_steer"].mean()
    tail_median_delta = worsening_df["delta_rmse_tail_abs_steer"].median()
    top_subj_by_count = subj_df.sort_values(
        by=["worsening_count", "worsening_ratio_within_group"],
        ascending=[False, False],
    ).iloc[0]
    top_subj_by_ratio = subj_df.sort_values(
        by=["worsening_ratio_within_group", "worsening_count"],
        ascending=[False, False],
    ).iloc[0]
    top_morph_by_count = morph_df.sort_values(
        by=["worsening_count", "worsening_ratio_within_group"],
        ascending=[False, False],
    ).iloc[0]
    top_morph_by_ratio = morph_df.sort_values(
        by=["worsening_ratio_within_group", "worsening_count"],
        ascending=[False, False],
    ).iloc[0]
    top_subj_morph = subj_morph_df.sort_values(
        by=["worsening_count", "worsening_ratio_within_group"],
        ascending=[False, False],
    ).iloc[0]
    top_subj_morph_by_ratio = subj_morph_df.sort_values(
        by=[
            "worsening_ratio_within_group",
            "mean_delta_boundary_shift_abs_err",
            "worsening_count",
        ],
        ascending=[False, False, False],
    ).iloc[0]

    print(
        f"  - 恶化样本中的 mean(delta_boundary_shift_abs_err): "
        f"{worsening_df['delta_boundary_shift_abs_err'].mean():.6f}"
    )
    print(
        f"  - 恶化样本中的 median(delta_boundary_shift_abs_err): "
        f"{worsening_df['delta_boundary_shift_abs_err'].median():.6f}"
    )
    print("-" * 88)
    print("二、subj × eval_morphology_label 交叉统计（按恶化数量排序，前 5 组）")
    for _, row in subj_morph_df.head(5).iterrows():
        print(
            "  - "
            f"{format_group_label(row, ['subj', 'eval_morphology_label'])}"
            f" | 恶化数={int(row['worsening_count'])}"
            f" | 组内恶化率={row['worsening_ratio_within_group']:.2%}"
            f" | 恶化均值={row['mean_delta_boundary_shift_abs_err']:.6f}"
        )

    print("-" * 88)
    print("三、trigger_type × mechanism_tag 交叉统计")
    for _, row in trigger_mech_df.head(5).iterrows():
        if int(row["worsening_count"]) == 0:
            continue
        print(
            "  - "
            f"{format_group_label(row, ['trigger_type', 'mechanism_tag'])}"
            f" | 恶化数={int(row['worsening_count'])}"
            f" | 组内恶化率={row['worsening_ratio_within_group']:.2%}"
            f" | 恶化均值={row['mean_delta_boundary_shift_abs_err']:.6f}"
        )

    print("-" * 88)
    print("四、boundary_shift 恶化与 tail 整体恶化的关系")
    print(
        f"  - 在 boundary_shift 恶化样本中，tail 也恶化: {tail_worse_count} "
        f"({tail_worse_ratio:.2%})"
    )
    print(
        f"  - 在 boundary_shift 恶化样本中，tail 未恶化: {tail_not_worse_count} "
        f"({tail_not_worse_ratio:.2%})"
    )
    print(
        f"  - mean(delta_rmse_tail_abs_steer): {tail_mean_delta:.6f}"
        f" | median(delta_rmse_tail_abs_steer): {tail_median_delta:.6f}"
    )

    trigger_unique = int((trigger_mech_df["worsening_count"] > 0).sum())

    print("-" * 88)
    print("五、诊断结论")
    print(
        f"  1. 被试维度未见单一主导：按恶化数量看，{top_subj_by_count['subj']} 最多，"
        f"占全部恶化样本 {top_subj_by_count['worsening_share_among_all_worsening']:.2%}；"
        f"按组内恶化率看，{top_subj_by_ratio['subj']} 最高"
        f"（{top_subj_by_ratio['worsening_ratio_within_group']:.2%}）。"
    )
    print(
        f"  2. 形态维度上，按恶化数量看是 {top_morph_by_count['eval_morphology_label']} 最多"
        f"（{int(top_morph_by_count['worsening_count'])} 个）；"
        f"按组内恶化率看，{top_morph_by_ratio['eval_morphology_label']} 风险最高"
        f"（{top_morph_by_ratio['worsening_ratio_within_group']:.2%}）。"
    )
    print(
        f"  3. 细分到 subj × morphology，当前恶化数量最多的组合是 "
        f"{format_group_label(top_subj_morph, ['subj', 'eval_morphology_label'])}，"
        f"恶化数 {int(top_subj_morph['worsening_count'])}，"
        f"组内恶化率 {top_subj_morph['worsening_ratio_within_group']:.2%}；"
        f"而组内恶化率最高的组合是 "
        f"{format_group_label(top_subj_morph_by_ratio, ['subj', 'eval_morphology_label'])}"
        f"（{top_subj_morph_by_ratio['worsening_ratio_within_group']:.2%}）。"
    )
    if trigger_unique <= 1:
        top_trigger = trigger_mech_df.iloc[0]
        print(
            "  4. trigger_type × mechanism_tag 维度当前没有可比较的结构差异："
            f"恶化样本均落在 {format_group_label(top_trigger, ['trigger_type', 'mechanism_tag'])}。"
        )
    else:
        top_trigger = trigger_mech_df.iloc[0]
        print(
            f"  4. trigger_type × mechanism_tag 头部分组为 "
            f"{format_group_label(top_trigger, ['trigger_type', 'mechanism_tag'])}，"
            f"但是否主导需结合其占比判断，详见 CSV。"
        )
    if tail_not_worse_ratio > 0.50:
        print(
            "  5. boundary_shift 恶化与 tail 整体恶化存在明显分离："
            f"{tail_not_worse_ratio:.2%} 的 boundary_shift 恶化样本并未伴随 tail 恶化。"
        )
    else:
        print(
            "  5. boundary_shift 恶化多数伴随 tail 恶化，分离现象不明显。"
        )
    print("=" * 88)


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"[ERROR] 输入文件不存在: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    validate_input(df)

    df = df.copy()
    df["tail_worsened_flag"] = df["delta_rmse_tail_abs_steer"] > 0

    worsening_df = df[df["delta_boundary_shift_abs_err"] > 0].copy()

    overview_df = build_overview(df, worsening_df)
    subj_df = group_diagnosis(df, worsening_df, ["subj"], "subj_summary")
    morph_df = group_diagnosis(
        df,
        worsening_df,
        ["eval_morphology_label"],
        "morphology_summary",
    )
    subj_morph_df = group_diagnosis(
        df,
        worsening_df,
        ["subj", "eval_morphology_label"],
        "subj_morphology_cross",
    )
    trigger_mech_df = group_diagnosis(
        df,
        worsening_df,
        ["trigger_type", "mechanism_tag"],
        "trigger_mechanism_cross",
    )
    tail_relation_df = build_tail_relation(df)
    tail_summary_df = build_tail_summary(df, worsening_df)
    dominance_df = pd.concat(
        [
            assess_dominance(subj_df, ["subj"], "subj"),
            assess_dominance(morph_df, ["eval_morphology_label"], "eval_morphology_label"),
            assess_dominance(
                trigger_mech_df,
                ["trigger_type", "mechanism_tag"],
                "trigger_type × mechanism_tag",
            ),
        ],
        ignore_index=True,
    )

    output_df = pd.concat(
        [
            overview_df,
            subj_df,
            morph_df,
            subj_morph_df,
            trigger_mech_df,
            tail_relation_df,
            tail_summary_df,
            dominance_df,
        ],
        ignore_index=True,
        sort=False,
    )
    output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print_conclusion(
        df_all=df,
        worsening_df=worsening_df,
        subj_df=subj_df,
        morph_df=morph_df,
        subj_morph_df=subj_morph_df,
        trigger_mech_df=trigger_mech_df,
    )


if __name__ == "__main__":
    main()
