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

import stage03_v03_screening_sweep_gpu as gpu  # noqa: E402
import stage03_v03_vehicle_only_baselines as base  # noqa: E402
import stage03_v03_vehicle_only_inclusion_ablation as incl  # noqa: E402


OUT_ROOT = ROOT / "03_baselines" / "stage03_v04_vehicle_only_gpu_baseline"
DATASET_ROOT = ROOT / "03_processed_datasets" / "extreme_condition_v0_4_vehicle_only_gpu"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-20.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"

V04_ROOT = ROOT / "02_samples" / "extreme_condition_episodes_v0_4"
V04_TABLE_DIR = V04_ROOT / "tables"
V04_ALL = V04_TABLE_DIR / "extreme_condition_episodes_refiltered_v0_4.csv"
V04_PRIMARY = V04_TABLE_DIR / "primary_train_episodes_v0_4.csv"
V04_TRAIN_CANDIDATE = V04_TABLE_DIR / "train_candidate_episodes_v0_4.csv"

SUMMARY_PATH = OUT_ROOT / "tables" / "v04_vehicle_only_gpu_summary.csv"
RANKING_PATH = OUT_ROOT / "tables" / "v04_vehicle_only_gpu_ranking.csv"
REPORT_PATH = REPORT_DIR / "stage03_v04_vehicle_only_gpu_user_summary_cn.md"

DROP_LATERAL_FEATURES = ["lateral_distance_selected"]


def ensure_dirs() -> None:
    for path in [OUT_ROOT / "tables", OUT_ROOT / "logs", REPORT_DIR, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def configure_modules() -> None:
    gpu.OUT_ROOT = OUT_ROOT
    gpu.DATASET_ROOT = DATASET_ROOT
    gpu.REPORT_DIR = REPORT_DIR
    gpu.NOTES_DIR = NOTES_DIR
    gpu.DAILY_LOG = DAILY_LOG
    gpu.ARTIFACT_INDEX = ARTIFACT_INDEX
    gpu.SUMMARY_PATH = SUMMARY_PATH
    gpu.RANKING_PATH = RANKING_PATH

    incl.OUT_ROOT = OUT_ROOT
    incl.DATASET_ROOT = DATASET_ROOT
    incl.REPORT_DIR = REPORT_DIR
    incl.NOTES_DIR = NOTES_DIR
    incl.DAILY_LOG = DAILY_LOG
    incl.ARTIFACT_INDEX = ARTIFACT_INDEX

    base.EPISODE_TABLE = V04_ALL
    base.TABLE_DIR = OUT_ROOT / "tables"
    base.FIG_DIR = OUT_ROOT / "figures"
    base.LOG_DIR = OUT_ROOT / "logs"


def read_uids(path: Path) -> list[str]:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "episode_uid" not in df.columns:
        raise RuntimeError(f"missing episode_uid in {path}")
    return sorted(set(df["episode_uid"].dropna().astype(str)))


def make_variants() -> list[dict[str, Any]]:
    primary_uids = read_uids(V04_PRIMARY)
    train_uids = read_uids(V04_TRAIN_CANDIDATE)

    def variant(variant_id: str, name_cn: str, uids: list[str], with_lateral: bool) -> dict[str, Any]:
        item: dict[str, Any] = {
            "variant_id": variant_id,
            "name_cn": name_cn,
            "description_cn": "v0.4 重筛样本的车辆-only GPU 基线，只改变样本范围和是否保留横向偏移，不加入连续风格、生理或脑电。",
            "categories": [],
            "extra_episode_uids": uids,
            "extra_episode_source": "v04_primary_or_train_candidate",
        }
        if not with_lateral:
            item["drop_features"] = DROP_LATERAL_FEATURES
        return item

    return [
        variant("v04_primary_nolat", "v0.4 主训练候选，去横向偏移", primary_uids, with_lateral=False),
        variant("v04_primary_secondary_nolat", "v0.4 主+次级候选，去横向偏移", train_uids, with_lateral=False),
        variant("v04_primary_lat", "v0.4 主训练候选，保留横向偏移", primary_uids, with_lateral=True),
        variant("v04_primary_secondary_lat", "v0.4 主+次级候选，保留横向偏移", train_uids, with_lateral=True),
    ]


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
    base_row = df[df["variant_id"].eq("v04_primary_nolat")]
    if len(base_row):
        b = base_row.iloc[0]
        df["delta_rmse_vs_primary_nolat"] = df["test_rmse_steer"] - float(b["test_rmse_steer"])
        df["delta_wrong_side_vs_primary_nolat"] = df["test_wrong_side_rate_large"] - float(
            b["test_wrong_side_rate_large"]
        )
        df["delta_severe_under_vs_primary_nolat"] = df["test_severe_amp_under_rate_large"] - float(
            b["test_severe_amp_under_rate_large"]
        )
        df["delta_large_recall_vs_primary_nolat"] = df["test_large_response_recall"] - float(
            b["test_large_response_recall"]
        )
    else:
        df["delta_rmse_vs_primary_nolat"] = np.nan
        df["delta_wrong_side_vs_primary_nolat"] = np.nan
        df["delta_severe_under_vs_primary_nolat"] = np.nan
        df["delta_large_recall_vs_primary_nolat"] = np.nan

    df["screening_score"] = (
        -df["delta_rmse_vs_primary_nolat"].fillna(0.0)
        - 0.35 * df["delta_wrong_side_vs_primary_nolat"].fillna(0.0)
        - 0.25 * df["delta_severe_under_vs_primary_nolat"].fillna(0.0)
        + 0.15 * df["delta_large_recall_vs_primary_nolat"].fillna(0.0)
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
        "# v0.4 重筛样本车辆-only GPU 基线",
        "",
        "## 这次为什么做",
        "",
        "这次不是继续比较旧的 809 样本，而是使用 v0.4 从 1574 个初始片段重新筛出的样本。v0.4 的核心规则是：锚点后车辆有变化就保留，即使驾驶员操作弱；锚点后车辆和驾驶员都弱就排除；快打方向但车辆变化弱先谨慎处理。",
        "",
        "本轮仍然只跑车辆-only，不加入连续驾驶风格、生理或脑电。目的只是检查 v0.4 样本定义本身是否更适合建模。",
        "",
        "## 运行设置",
        "",
        f"- 设备：`{device}`，本地 GPU。",
        "- 输入：车辆历史 + 事件/工况上下文字段。",
        "- 标签：锚点后的方向盘相对轨迹。",
        "- 模型：无学习基线 + PyTorch 线性头/小型神经网络；按验证集 RMSE 选模型，再报告测试集。",
        "",
        "## 结果表",
        "",
        markdown_table(ranking),
        "",
        "## 当前读法",
        "",
        f"- 综合排序第一：`{best_score['variant_id']}`，test RMSE={fmt(best_score['test_rmse_steer'])}，综合分数={fmt(best_score['screening_score'])}。",
        f"- 单看整体 RMSE 最低：`{best_rmse['variant_id']}`，test RMSE={fmt(best_rmse['test_rmse_steer'])}。",
        "- 如果“主+次级”比“主训练候选”更好，说明次级样本有助于扩充任务覆盖；如果变差，说明次级样本仍需要继续分层。",
        "- 如果保留横向偏移改善物理指标但恶化 RMSE，说明横向偏移可能更像极限姿态提示，而不是稳定的通用轨迹输入。",
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
        "## 2026-05-20 v0.4 重筛样本车辆-only GPU 基线\n\n"
        "- 为什么做：用户要求在 v0.4 从 1574 个初始 episode 重筛后的样本上继续跑车辆-only，不使用服务器，直接使用本地 GPU。\n"
        f"- 运行设备：`{device}`。\n"
        f"- 当前综合排序第一：`{best['variant_id']}`，test RMSE={fmt(best['test_rmse_steer'])}，大响应错侧率={fmt(best['test_wrong_side_rate_large'])}，严重幅值不足率={fmt(best['test_severe_amp_under_rate_large'])}，大响应召回={fmt(best['test_large_response_recall'])}。\n"
        f"- 用户查看版报告：`{REPORT_PATH}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        if path.exists():
            raw = path.read_text(encoding="utf-8")
        else:
            raw = ""
        if "## 2026-05-20 v0.4 重筛样本车辆-only GPU 基线" not in raw:
            path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")

    artifact = (
        "## 2026-05-20 v0.4 重筛样本车辆-only GPU 基线\n\n"
        f"- 用户查看版报告：`{REPORT_PATH}`\n"
        f"- 汇总表：`{SUMMARY_PATH}`\n"
        f"- 排名表：`{RANKING_PATH}`\n"
        f"- 输出目录：`{OUT_ROOT}`\n"
    )
    if ARTIFACT_INDEX.exists():
        raw = ARTIFACT_INDEX.read_text(encoding="utf-8")
    else:
        raw = ""
    if "## 2026-05-20 v0.4 重筛样本车辆-only GPU 基线" not in raw:
        ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_modules()
    gpu.set_seed(20260520)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，本轮要求使用本地 GPU。")
    device = torch.device("cuda")
    print(f"cuda={torch.cuda.get_device_name(0)}", flush=True)

    sample_split, session_split = incl.load_reference_split()
    rows: list[dict[str, Any]] = []
    for variant in make_variants():
        print(f"run {variant['variant_id']} samples={len(set(variant.get('extra_episode_uids') or []))}", flush=True)
        rows.append(gpu.run_variant_gpu(variant, sample_split, session_split, device))
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    ranking = score_rows(summary)
    ranking.to_csv(RANKING_PATH, index=False, encoding="utf-8-sig")
    write_report(ranking, device)
    append_notes(ranking, device)
    print(
        ranking[
            [
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
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
