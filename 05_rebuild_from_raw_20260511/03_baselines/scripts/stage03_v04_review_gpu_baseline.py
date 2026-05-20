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


OUT_ROOT = ROOT / "03_baselines" / "stage03_v04_review_gpu_baseline"
DATASET_ROOT = ROOT / "03_processed_datasets" / "extreme_condition_v0_4_review_gpu"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-20.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"

V04_TABLE_DIR = ROOT / "02_samples" / "extreme_condition_episodes_v0_4" / "tables"
V04_ALL = V04_TABLE_DIR / "extreme_condition_episodes_refiltered_v0_4.csv"
V04_PRIMARY = V04_TABLE_DIR / "primary_train_episodes_v0_4.csv"
V04_SECONDARY = V04_TABLE_DIR / "secondary_train_episodes_v0_4.csv"
V04_REVIEW = V04_TABLE_DIR / "manual_review_episodes_v0_4.csv"

SUMMARY_PATH = OUT_ROOT / "tables" / "v04_review_gpu_summary.csv"
RANKING_PATH = OUT_ROOT / "tables" / "v04_review_gpu_ranking.csv"
REPORT_PATH = REPORT_DIR / "stage03_v04_review_gpu_user_summary_cn.md"

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
    uids = sorted(set(read_uids(V04_PRIMARY)) | set(read_uids(V04_SECONDARY)) | set(read_uids(V04_REVIEW)))
    return [
        {
            "variant_id": "v04_primary_secondary_review_nolat",
            "name_cn": "v0.4 主训练+次级+待复核，去横向偏移",
            "description_cn": "在 v0.4 主训练候选和次级候选基础上继续加入待复核样本，先去掉横向偏移，避免坐标/道路模块跳变影响。",
            "categories": [],
            "extra_episode_uids": uids,
            "extra_episode_source": "v04_primary+secondary+manual_review",
            "drop_features": DROP_LATERAL_FEATURES,
        }
    ]


def fmt(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.4f}"


def write_report(summary: pd.DataFrame, device: torch.device) -> None:
    row = summary.iloc[0].to_dict()
    lines = [
        "# v0.4 主训练+次级+待复核样本 GPU 结果",
        "",
        "## 这次为什么做",
        "",
        "用户希望不要轻易丢掉待复核样本，因此本轮在 v0.4 主训练候选和次级候选基础上，继续加入待复核样本，检查它们是否能扩充样本覆盖并改善车辆-only 预测。",
        "",
        "本轮只跑一组，沿用上一轮较稳的设置：去掉横向偏移，不加入连续驾驶风格、生理或脑电。",
        "",
        "## 运行设置",
        "",
        f"- 设备：`{device}`。",
        "- 输入：车辆历史 + 事件/工况上下文字段。",
        "- 标签：锚点后的方向盘相对轨迹。",
        "- 模型：无学习基线 + PyTorch 线性头/小型神经网络；按验证集 RMSE 选模型，再报告测试集。",
        "",
        "## 结果",
        "",
        f"- 可用样本数：{int(row['sample_count'])}",
        f"- 验证集选择模型：`{row['val_selected_model']}`",
        f"- test RMSE：{fmt(row['test_rmse_steer'])}",
        f"- 主阶段 RMSE：{fmt(row['test_primary_rmse_0_2s'])}",
        f"- 尾段 RMSE：{fmt(row['test_tail_rmse_2_5s'])}",
        f"- 大响应错侧率：{fmt(row['test_wrong_side_rate_large'])}",
        f"- 严重幅值不足率：{fmt(row['test_severe_amp_under_rate_large'])}",
        f"- 大响应召回：{fmt(row['test_large_response_recall'])}",
        "",
        "## 产物位置",
        "",
        f"- 汇总表：`{SUMMARY_PATH}`",
        f"- 输出目录：`{OUT_ROOT}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def append_notes(summary: pd.DataFrame, device: torch.device) -> None:
    row = summary.iloc[0].to_dict()
    block = (
        "## 2026-05-20 v0.4 主训练+次级+待复核 GPU 基线\n\n"
        "- 为什么做：用户要求在服务器 GPU 上继续跑 v0.4 主训练+次级+待复核样本，检查待复核样本是否能纳入训练。\n"
        f"- 运行设备：`{device}`。\n"
        f"- 当前结果：样本数 {int(row['sample_count'])}，test RMSE={fmt(row['test_rmse_steer'])}，大响应错侧率={fmt(row['test_wrong_side_rate_large'])}，严重幅值不足率={fmt(row['test_severe_amp_under_rate_large'])}，大响应召回={fmt(row['test_large_response_recall'])}。\n"
        f"- 用户查看版报告：`{REPORT_PATH}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        if "## 2026-05-20 v0.4 主训练+次级+待复核 GPU 基线" not in raw:
            path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")
    raw = ARTIFACT_INDEX.read_text(encoding="utf-8") if ARTIFACT_INDEX.exists() else ""
    artifact = (
        "## 2026-05-20 v0.4 主训练+次级+待复核 GPU 基线\n\n"
        f"- 用户查看版报告：`{REPORT_PATH}`\n"
        f"- 汇总表：`{SUMMARY_PATH}`\n"
        f"- 输出目录：`{OUT_ROOT}`\n"
    )
    if "## 2026-05-20 v0.4 主训练+次级+待复核 GPU 基线" not in raw:
        ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_modules()
    gpu.set_seed(20260520)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。")
    device = torch.device("cuda")
    print(f"cuda={torch.cuda.get_device_name(0)}", flush=True)
    sample_split, session_split = incl.load_reference_split()
    rows: list[dict[str, Any]] = []
    for variant in make_variants():
        print(f"run {variant['variant_id']} samples={len(set(variant.get('extra_episode_uids') or []))}", flush=True)
        rows.append(gpu.run_variant_gpu(variant, sample_split, session_split, device))
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(RANKING_PATH, index=False, encoding="utf-8-sig")
    write_report(summary, device)
    append_notes(summary, device)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
