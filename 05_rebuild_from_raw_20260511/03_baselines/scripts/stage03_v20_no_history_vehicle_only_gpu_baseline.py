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

import stage03_v11_vehicle_only_gpu_baseline as runner  # noqa: E402


OUT_ROOT = ROOT / "03_baselines" / "stage03_v20_no_history_vehicle_only_gpu_baseline"
DATASET_ROOT = ROOT / "03_processed_datasets" / "record_episode_v2_0_no_history_vehicle_only_gpu"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-22.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"

V20_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v2_0_no_history_reaudit"
V20_TABLE_DIR = V20_ROOT / "tables"
V20_ALL = V20_TABLE_DIR / "record_level_episodes_all_v2_0.csv"

SUMMARY_PATH = OUT_ROOT / "tables" / "v20_no_history_vehicle_only_gpu_summary.csv"
RANKING_PATH = OUT_ROOT / "tables" / "v20_no_history_vehicle_only_gpu_ranking.csv"
REPORT_PATH = REPORT_DIR / "stage03_v20_no_history_vehicle_only_gpu_user_summary_cn.md"

DROP_LATERAL_FEATURES = ["lateral_distance_selected"]
TEST_SUBJECTS = {"cwh", "gf", "tyy"}
VAL_SUBJECTS = {"byx", "gzj", "yyl"}

TRAIN_DECISIONS_ALL = [
    "train_noncurve_vehicle_dynamic",
    "train_noncurve_secondary_dynamic",
    "train_curve_roll_dynamic",
    "train_curve_normal_or_weak",
]
TRAIN_DECISIONS_NONCURVE = [
    "train_noncurve_vehicle_dynamic",
    "train_noncurve_secondary_dynamic",
]
TRAIN_DECISIONS_CURVE = [
    "train_curve_roll_dynamic",
    "train_curve_normal_or_weak",
]


def ensure_dirs() -> None:
    for path in [OUT_ROOT / "tables", OUT_ROOT / "figures", OUT_ROOT / "logs", REPORT_DIR, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def configure_runner_modules() -> None:
    runner.OUT_ROOT = OUT_ROOT
    runner.DATASET_ROOT = DATASET_ROOT
    runner.REPORT_DIR = REPORT_DIR
    runner.NOTES_DIR = NOTES_DIR
    runner.DAILY_LOG = DAILY_LOG
    runner.ARTIFACT_INDEX = ARTIFACT_INDEX
    runner.SUMMARY_PATH = SUMMARY_PATH
    runner.RANKING_PATH = RANKING_PATH
    runner.REPORT_PATH = REPORT_PATH

    runner.gpu.OUT_ROOT = OUT_ROOT
    runner.gpu.DATASET_ROOT = DATASET_ROOT
    runner.gpu.REPORT_DIR = REPORT_DIR
    runner.gpu.NOTES_DIR = NOTES_DIR
    runner.gpu.DAILY_LOG = DAILY_LOG
    runner.gpu.ARTIFACT_INDEX = ARTIFACT_INDEX
    runner.gpu.SUMMARY_PATH = SUMMARY_PATH
    runner.gpu.RANKING_PATH = RANKING_PATH

    runner.incl.OUT_ROOT = OUT_ROOT
    runner.incl.DATASET_ROOT = DATASET_ROOT
    runner.incl.REPORT_DIR = REPORT_DIR
    runner.incl.NOTES_DIR = NOTES_DIR
    runner.incl.DAILY_LOG = DAILY_LOG
    runner.incl.ARTIFACT_INDEX = ARTIFACT_INDEX

    runner.base.TABLE_DIR = OUT_ROOT / "tables"
    runner.base.FIG_DIR = OUT_ROOT / "figures"
    runner.base.LOG_DIR = OUT_ROOT / "logs"


def coerce_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    raw = df[col]
    if raw.dtype == bool:
        return raw.fillna(False)
    return raw.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def choose_context(row: pd.Series) -> str:
    module = str(row.get("road_coord_dominant_module_v1_9", "")).strip()
    if module and module.lower() != "nan":
        return module
    for col in ["road_module_names", "road_design_categories", "v2_0_decision_cn", "episode_group_cn"]:
        value = str(row.get(col, "")).strip()
        if value and value.lower() != "nan":
            return value
    return "未知上下文"


def build_compat_episode_table(anchor_col: str, anchor_label: str) -> tuple[Path, pd.DataFrame]:
    src = pd.read_csv(V20_ALL, encoding="utf-8-sig", low_memory=False)
    if anchor_col not in src.columns:
        raise RuntimeError(f"v2.0 table missing anchor column: {anchor_col}")

    src[anchor_col] = pd.to_numeric(src[anchor_col], errors="coerce")
    src = src[np.isfinite(src[anchor_col])].copy()
    src["vehicle_raw_absolute_path"] = src["vehicle_file"].astype(str)
    src["vehicle_raw_relative_path"] = src["vehicle_file"].astype(str)
    src["t_condition_anchor"] = src[anchor_col].astype(float)
    src["v0_3_category"] = src["v2_0_decision"].astype(str)
    src["v0_3_category_cn"] = src["v2_0_decision_cn"].astype(str)
    src["condition_context_cn"] = src.apply(choose_context, axis=1)
    src["condition_level"] = src["v2_0_decision"].astype(str)
    src["steer_response_strength"] = src.get("driver_response_type", "").astype(str)
    src["response_shape"] = src.get("response_order", "").astype(str)
    src["condition_score_mean"] = coerce_numeric(src, "condition_score_peak")
    src["median_speed_kmh_window"] = coerce_numeric(src, "median_speed_kmh")
    src["peak_abs_ay_window"] = coerce_numeric(src, "peak_abs_ay")
    src["peak_abs_yaw_rate_window"] = coerce_numeric(src, "peak_abs_yaw_rate")
    src["peak_abs_roll_rate_window"] = coerce_numeric(src, "peak_abs_roll_rate")
    src["peak_abs_roll_window"] = coerce_numeric(src, "peak_abs_roll")
    src["peak_abs_curvature_window"] = coerce_numeric(src, "vehicle_lane_curvature_abs_max_sampled_v1_9")
    src["min_mu_window"] = coerce_numeric(src, "min_mu")
    src["anchor_source_for_training"] = anchor_col
    src["is_train_candidate_v2_0_bool"] = bool_series(src, "is_train_candidate_v2_0")
    src["is_review_candidate_v2_0_bool"] = bool_series(src, "is_review_candidate_v2_0")
    src["is_control_candidate_v2_0_bool"] = bool_series(src, "is_control_candidate_v2_0")

    out = OUT_ROOT / "tables" / f"v20_compat_{anchor_label}.csv"
    src.to_csv(out, index=False, encoding="utf-8-sig")
    return out, src


def fixed_subject_split(episodes: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    split_map: dict[str, str] = {}
    for _, row in episodes.iterrows():
        subject = str(row["subject"])
        if subject in TEST_SUBJECTS:
            split = "test"
        elif subject in VAL_SUBJECTS:
            split = "val"
        else:
            split = "train"
        split_map[str(row["episode_uid"])] = split
    return split_map, {}


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
        item["drop_features"] = DROP_LATERAL_FEATURES
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
    base_row = df[df["variant_id"].eq("v20_all_train_anchor_nolat")]
    if len(base_row):
        b = base_row.iloc[0]
        df["delta_rmse_vs_all_nolat"] = df["test_rmse_steer"] - float(b["test_rmse_steer"])
        df["delta_wrong_side_vs_all_nolat"] = df["test_wrong_side_rate_large"] - float(
            b["test_wrong_side_rate_large"]
        )
        df["delta_severe_under_vs_all_nolat"] = df["test_severe_amp_under_rate_large"] - float(
            b["test_severe_amp_under_rate_large"]
        )
        df["delta_large_recall_vs_all_nolat"] = df["test_large_response_recall"] - float(
            b["test_large_response_recall"]
        )
    else:
        df["delta_rmse_vs_all_nolat"] = np.nan
        df["delta_wrong_side_vs_all_nolat"] = np.nan
        df["delta_severe_under_vs_all_nolat"] = np.nan
        df["delta_large_recall_vs_all_nolat"] = np.nan

    df["screening_score"] = (
        -df["delta_rmse_vs_all_nolat"].fillna(0.0)
        - 0.35 * df["delta_wrong_side_vs_all_nolat"].fillna(0.0)
        - 0.25 * df["delta_severe_under_vs_all_nolat"].fillna(0.0)
        + 0.15 * df["delta_large_recall_vs_all_nolat"].fillna(0.0)
    )
    return df.sort_values(["screening_score", "test_rmse_steer"], ascending=[False, True])


def fmt(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.4f}"


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
            vals.append(str(val) if col in {"variant_id", "name_cn", "val_selected_model"} else fmt(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(ranking: pd.DataFrame, device: torch.device) -> None:
    best_score = ranking.iloc[0].to_dict()
    best_rmse = ranking.sort_values("test_rmse_steer").iloc[0].to_dict()
    lines = [
        "# v2.0 全量无历史继承重审样本：车辆-only GPU 基线",
        "",
        "## 这次训练的是什么模型",
        "",
        "这次不是训练连续驾驶风格模型，也不是训练生理/脑电模型。它是一个车辆-only 诊断基线：只使用车辆历史和事件上下文字段，验证 v2.0 新样本定义本身是否更适合建模。",
        "",
        "具体模型集合包括：",
        "",
        "- 无学习基线：零变化、训练集同类均值、训练集全局均值、历史趋势外推；",
        "- 线性头：把锚点前车辆历史压平成特征后直接预测后续方向盘相对轨迹；",
        "- 小型多层感知机：同样使用车辆历史特征，但允许非线性关系。",
        "",
        "最终按验证集 RMSE 选择一个模型，再报告测试集表现。因此这一步的目的不是最终刷分，而是先看 v2.0 样本能否让车辆-only 模型学到更合理的方向盘后续变化。",
        "",
        "## 运行设置",
        "",
        f"- 设备：`{device}`，本地 CUDA。",
        "- 样本入口：`record_level_episodes_all_v2_0.csv`。",
        "- 历史标签使用方式：不参与样本选择，只作为审计字段保留。",
        "- 主训练锚点：`model_anchor_s_v1_8`。这个锚点比原始 episode_start 更接近前面已讨论过的“去除过长平稳前奏后”的模型锚点。",
        "- 输入窗口：锚点前 2 秒车辆历史，20 Hz。",
        "- 标签窗口：锚点后 5 秒方向盘相对变化，20 Hz。",
        "- 划分：test=`cwh/gf/tyy`，val=`byx/gzj/yyl`，其余被试为 train。",
        "- 主比较版本：全量训练候选去横向偏移、全量训练候选保留横向偏移、非弯道候选、弯道候选。",
        "",
        "## 结果表",
        "",
        markdown_table(ranking),
        "",
        "## 当前读法",
        "",
        f"- 综合排序第一：`{best_score['variant_id']}`，test RMSE={fmt(best_score['test_rmse_steer'])}，大响应错侧率={fmt(best_score['test_wrong_side_rate_large'])}，严重幅值不足率={fmt(best_score['test_severe_amp_under_rate_large'])}。",
        f"- 单看整体 RMSE 最低：`{best_rmse['variant_id']}`，test RMSE={fmt(best_rmse['test_rmse_steer'])}。",
        "- 如果全量候选明显优于非弯道/弯道单独候选，说明 v2.0 的合并样本池更适合先做统一模型。",
        "- 如果非弯道或弯道单独候选更好，说明后续应考虑按道路/工况分开建模。",
        "- 如果保留横向偏移只改善部分物理指标但恶化 RMSE，需要继续按道路坐标质量分层使用，不应直接全局加入。",
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
    block = (
        "## 2026-05-22 v2.0 全量无历史继承重审样本车辆-only GPU 基线\n\n"
        "- 为什么做：用户指出不能再按历史候选身份筛样本，因此 v2.0 已对 1766 个 episode 全量重审。本轮只训练车辆-only 基线，检查新样本池是否具备建模价值。\n"
        f"- 运行设备：`{device}`，本地 CUDA。\n"
        "- 模型：无学习基线 + 线性头 + 小型多层感知机；不加入连续风格、生理、脑电或教师蒸馏。\n"
        "- 划分：test=cwh/gf/tyy，val=byx/gzj/yyl，其余 train。\n"
        f"- 当前综合排序第一：`{best['variant_id']}`，test RMSE={fmt(best['test_rmse_steer'])}，大响应错侧率={fmt(best['test_wrong_side_rate_large'])}，严重幅值不足率={fmt(best['test_severe_amp_under_rate_large'])}，大响应召回={fmt(best['test_large_response_recall'])}。\n"
        f"- 用户查看版报告：`{REPORT_PATH}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        marker = "## 2026-05-22 v2.0 全量无历史继承重审样本车辆-only GPU 基线"
        if marker not in raw:
            path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")

    artifact = (
        "## 2026-05-22 v2.0 全量无历史继承重审样本车辆-only GPU 基线\n\n"
        f"- 用户查看版报告：`{REPORT_PATH}`\n"
        f"- 汇总表：`{SUMMARY_PATH}`\n"
        f"- 排名表：`{RANKING_PATH}`\n"
        f"- 输出目录：`{OUT_ROOT}`\n"
    )
    raw = ARTIFACT_INDEX.read_text(encoding="utf-8") if ARTIFACT_INDEX.exists() else ""
    marker = "## 2026-05-22 v2.0 全量无历史继承重审样本车辆-only GPU 基线"
    if marker not in raw:
        ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_runner_modules()
    runner.gpu.set_seed(20260522)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，本轮要求使用 GPU。")
    device = torch.device("cuda")
    print(f"cuda={torch.cuda.get_device_name(0)}", flush=True)

    compat_table, compat_df = build_compat_episode_table("model_anchor_s_v1_8", "model_anchor")
    runner.base.EPISODE_TABLE = compat_table
    sample_split, session_split = fixed_subject_split(compat_df)

    variants = [
        make_variant(
            "v20_all_train_anchor_nolat",
            "v2.0 全量训练候选，推荐锚点，去横向偏移",
            TRAIN_DECISIONS_ALL,
            False,
            "主诊断版本：使用 v2.0 全量训练候选，但去掉横向偏移，避免道路坐标跳变直接污染训练。",
        ),
        make_variant(
            "v20_all_train_anchor_lat",
            "v2.0 全量训练候选，推荐锚点，保留横向偏移",
            TRAIN_DECISIONS_ALL,
            True,
            "横向偏移诊断版本：检查横向偏移是否提供额外姿态/道路信息，或是否引入坐标噪声。",
        ),
        make_variant(
            "v20_noncurve_train_anchor_nolat",
            "v2.0 非弯道训练候选，推荐锚点，去横向偏移",
            TRAIN_DECISIONS_NONCURVE,
            False,
            "非弯道诊断版本：单独看低附着、维修路段、连续超车等非弯道候选是否更容易建模。",
        ),
        make_variant(
            "v20_curve_train_anchor_nolat",
            "v2.0 弯道训练候选，推荐锚点，去横向偏移",
            TRAIN_DECISIONS_CURVE,
            False,
            "弯道诊断版本：单独看正常/弱侧倾弯道与侧倾动态弯道是否需要独立模型。",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for variant in variants:
        print(f"run {variant['variant_id']} categories={variant['categories']}", flush=True)
        rows.append(runner.run_variant_with_plots(variant, sample_split, session_split, device))

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
