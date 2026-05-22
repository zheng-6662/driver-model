# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
ROOT = Path(os.environ.get("REBUILD_ROOT", str(PROJECT_ROOT / "05_rebuild_from_raw_20260511")))
SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage03_v20_no_history_vehicle_only_gpu_baseline as v20  # noqa: E402


OUT_ROOT = ROOT / "03_baselines" / "stage03_v20_review_inclusion_vehicle_only_gpu"
DATASET_ROOT = ROOT / "03_processed_datasets" / "record_episode_v2_0_review_inclusion_vehicle_only_gpu"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-22.md"
ARTIFACT_INDEX = NOTES_DIR / "00_project_notes" / "ARTIFACT_INDEX_CN.md"

SUMMARY_PATH = OUT_ROOT / "tables" / "v20_review_inclusion_vehicle_only_gpu_summary.csv"
RANKING_PATH = OUT_ROOT / "tables" / "v20_review_inclusion_vehicle_only_gpu_ranking.csv"
REPORT_PATH = REPORT_DIR / "stage03_v20_review_inclusion_vehicle_only_gpu_user_summary_cn.md"

TRAIN_DECISIONS_ALL = list(v20.TRAIN_DECISIONS_ALL)
TRAIN_DECISIONS_NONCURVE = list(v20.TRAIN_DECISIONS_NONCURVE)
TRAIN_DECISIONS_CURVE = list(v20.TRAIN_DECISIONS_CURVE)

REVIEW_DECISIONS_ALL = [
    "review_curve_height_pose_abnormal",
    "review_speed_brake_only",
    "review_mapping_uncertain",
    "review_fast_steer_weak_vehicle",
    "review_noncurve_height_abnormal_weak_dynamic",
]
REVIEW_DECISIONS_NONCURVE = [
    "review_speed_brake_only",
    "review_fast_steer_weak_vehicle",
    "review_mapping_uncertain",
    "review_noncurve_height_abnormal_weak_dynamic",
]
REVIEW_DECISIONS_CURVE = [
    "review_curve_height_pose_abnormal",
    "review_mapping_uncertain",
]


def ensure_dirs() -> None:
    for path in [OUT_ROOT / "tables", OUT_ROOT / "figures", OUT_ROOT / "logs", REPORT_DIR, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def configure_modules() -> None:
    v20.OUT_ROOT = OUT_ROOT
    v20.DATASET_ROOT = DATASET_ROOT
    v20.REPORT_DIR = REPORT_DIR
    v20.NOTES_DIR = NOTES_DIR
    v20.DAILY_LOG = DAILY_LOG
    v20.ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"
    v20.SUMMARY_PATH = SUMMARY_PATH
    v20.RANKING_PATH = RANKING_PATH
    v20.REPORT_PATH = REPORT_PATH
    v20.configure_runner_modules()


def make_variant(
    variant_id: str,
    name_cn: str,
    categories: list[str],
    with_lateral: bool,
    description_cn: str,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "variant_id": variant_id,
        "name_cn": name_cn,
        "description_cn": description_cn,
        "categories": categories,
        "anchor_label": "model_anchor_s_v1_8",
    }
    if not with_lateral:
        item["drop_features"] = list(v20.DROP_LATERAL_FEATURES)
    return item


def score_rows(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    numeric_cols = [
        "test_rmse_steer",
        "test_primary_rmse_0_2s",
        "test_tail_rmse_2_5s",
        "test_wrong_side_rate_large",
        "test_severe_amp_under_rate_large",
        "test_large_response_recall",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    base_row = df[df["variant_id"].eq("v20_train_review_anchor_nolat")]
    if len(base_row):
        b = base_row.iloc[0]
        df["delta_rmse_vs_train_review_nolat"] = df["test_rmse_steer"] - float(b["test_rmse_steer"])
        df["delta_wrong_side_vs_train_review_nolat"] = df["test_wrong_side_rate_large"] - float(
            b["test_wrong_side_rate_large"]
        )
        df["delta_severe_under_vs_train_review_nolat"] = df["test_severe_amp_under_rate_large"] - float(
            b["test_severe_amp_under_rate_large"]
        )
        df["delta_large_recall_vs_train_review_nolat"] = df["test_large_response_recall"] - float(
            b["test_large_response_recall"]
        )
    else:
        df["delta_rmse_vs_train_review_nolat"] = np.nan
        df["delta_wrong_side_vs_train_review_nolat"] = np.nan
        df["delta_severe_under_vs_train_review_nolat"] = np.nan
        df["delta_large_recall_vs_train_review_nolat"] = np.nan

    df["screening_score"] = (
        -df["delta_rmse_vs_train_review_nolat"].fillna(0.0)
        - 0.35 * df["delta_wrong_side_vs_train_review_nolat"].fillna(0.0)
        - 0.25 * df["delta_severe_under_vs_train_review_nolat"].fillna(0.0)
        + 0.15 * df["delta_large_recall_vs_train_review_nolat"].fillna(0.0)
    )
    return df.sort_values(["screening_score", "test_rmse_steer"], ascending=[False, True])


def markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "variant_id",
        "name_cn",
        "sample_count",
        "val_selected_model",
        "test_rmse_steer",
        "test_primary_rmse_0_2s",
        "test_tail_rmse_2_5s",
        "test_wrong_side_rate_large",
        "test_severe_amp_under_rate_large",
        "test_large_response_recall",
        "screening_score",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(str(val) if col in {"variant_id", "name_cn", "val_selected_model"} else v20.fmt(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(ranking: pd.DataFrame, device: torch.device) -> None:
    best = ranking.iloc[0].to_dict()
    best_rmse = ranking.sort_values("test_rmse_steer").iloc[0].to_dict()
    lines = [
        "# v2.0 待复核样本纳入训练：车辆-only GPU 基线",
        "",
        "## 这次为什么做",
        "",
        "用户提出：待复核样本不一定都应该排除，其中可能有很多可训练片段。因此本轮在 v2.0 无历史继承重审样本上，把待复核样本也纳入训练，检查它是否能扩大数据覆盖并改善车辆-only 预测。",
        "",
        "本轮仍然只训练车辆-only，不加入连续风格、生理数据、脑电或教师蒸馏。",
        "",
        "## 运行设置",
        "",
        f"- 设备：`{device}`，本地 CUDA。",
        "- 主锚点：`model_anchor_s_v1_8`。",
        "- 输入：锚点前 2 秒车辆历史，20 Hz。",
        "- 标签：锚点后 5 秒方向盘相对变化，20 Hz。",
        "- 划分：test=`cwh/gf/tyy`，val=`byx/gzj/yyl`，其余被试为 train。",
        "- 对照：全量训练候选 + 待复核、非弯道训练候选 + 非弯道待复核、弯道训练候选 + 弯道待复核。",
        "",
        "## 结果表",
        "",
        markdown_table(ranking),
        "",
        "## 当前读法",
        "",
        f"- 综合排序第一：`{best['variant_id']}`，test RMSE={v20.fmt(best['test_rmse_steer'])}，大响应错侧率={v20.fmt(best['test_wrong_side_rate_large'])}，严重幅值不足率={v20.fmt(best['test_severe_amp_under_rate_large'])}。",
        f"- 单看整体 RMSE 最低：`{best_rmse['variant_id']}`，test RMSE={v20.fmt(best_rmse['test_rmse_steer'])}。",
        "- 如果加待复核后整体指标和物理指标同时改善，说明待复核样本里确实有可用训练信息。",
        "- 如果只改善 RMSE 但恶化错侧、幅值或大响应召回，说明待复核样本会让任务平均化，后续需要继续分层纳入。",
        "",
        "## 产物位置",
        "",
        f"- 汇总表：`{SUMMARY_PATH}`",
        f"- 排名表：`{RANKING_PATH}`",
        f"- 输出目录：`{OUT_ROOT}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def append_notes(ranking: pd.DataFrame, device: torch.device) -> None:
    best = ranking.iloc[0].to_dict()
    marker = "## 2026-05-22 v2.0 待复核样本纳入训练车辆-only GPU 基线"
    block = (
        f"{marker}\n\n"
        "- 为什么做：检查待复核样本是否可以作为训练样本，而不是直接排除。\n"
        f"- 运行设备：`{device}`，本地 CUDA。\n"
        "- 模型：无学习基线 + 线性头 + 小型多层感知机；不加入连续风格、生理、脑电或教师蒸馏。\n"
        f"- 当前综合排序第一：`{best['variant_id']}`，test RMSE={v20.fmt(best['test_rmse_steer'])}，大响应错侧率={v20.fmt(best['test_wrong_side_rate_large'])}，严重幅值不足率={v20.fmt(best['test_severe_amp_under_rate_large'])}，大响应召回={v20.fmt(best['test_large_response_recall'])}。\n"
        f"- 用户查看版报告：`{REPORT_PATH}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        if marker not in raw:
            path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")

    artifact_path = NOTES_DIR / "ARTIFACT_INDEX_CN.md"
    raw = artifact_path.read_text(encoding="utf-8") if artifact_path.exists() else ""
    artifact = (
        f"{marker}\n\n"
        f"- 用户查看版报告：`{REPORT_PATH}`\n"
        f"- 汇总表：`{SUMMARY_PATH}`\n"
        f"- 排名表：`{RANKING_PATH}`\n"
        f"- 输出目录：`{OUT_ROOT}`\n"
    )
    if marker not in raw:
        artifact_path.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_modules()
    v20.runner.gpu.set_seed(20260522)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，本轮要求使用 GPU。")
    device = torch.device("cuda")
    print(f"cuda={torch.cuda.get_device_name(0)}", flush=True)

    compat_table, compat_df = v20.build_compat_episode_table("model_anchor_s_v1_8", "model_anchor_review")
    v20.runner.base.EPISODE_TABLE = compat_table
    sample_split, session_split = v20.fixed_subject_split(compat_df)

    variants = [
        make_variant(
            "v20_train_review_anchor_nolat",
            "v2.0 训练候选 + 全部待复核，推荐锚点，去横向偏移",
            TRAIN_DECISIONS_ALL + REVIEW_DECISIONS_ALL,
            False,
            "把所有待复核样本都纳入训练，去掉横向偏移，检查数据扩充是否有效。",
        ),
        make_variant(
            "v20_train_review_anchor_lat",
            "v2.0 训练候选 + 全部待复核，推荐锚点，保留横向偏移",
            TRAIN_DECISIONS_ALL + REVIEW_DECISIONS_ALL,
            True,
            "把所有待复核样本都纳入训练，同时保留横向偏移，检查横向偏移与待复核样本是否互补。",
        ),
        make_variant(
            "v20_noncurve_train_review_anchor_nolat",
            "v2.0 非弯道训练候选 + 非弯道待复核，推荐锚点，去横向偏移",
            TRAIN_DECISIONS_NONCURVE + REVIEW_DECISIONS_NONCURVE,
            False,
            "只看非弯道方向，把非弯道待复核纳入训练。",
        ),
        make_variant(
            "v20_curve_train_review_anchor_nolat",
            "v2.0 弯道训练候选 + 弯道待复核，推荐锚点，去横向偏移",
            TRAIN_DECISIONS_CURVE + REVIEW_DECISIONS_CURVE,
            False,
            "只看弯道方向，把弯道待复核纳入训练。",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for variant in variants:
        print(f"run {variant['variant_id']} categories={variant['categories']}", flush=True)
        rows.append(v20.runner.run_variant_with_plots(variant, sample_split, session_split, device))

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    ranking = score_rows(summary)
    ranking.to_csv(RANKING_PATH, index=False, encoding="utf-8-sig")
    write_report(ranking, device)
    append_notes(ranking, device)

    cols = [
        "variant_id",
        "sample_count",
        "val_selected_model",
        "test_rmse_steer",
        "test_primary_rmse_0_2s",
        "test_tail_rmse_2_5s",
        "test_wrong_side_rate_large",
        "test_severe_amp_under_rate_large",
        "test_large_response_recall",
        "screening_score",
    ]
    print(ranking[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
