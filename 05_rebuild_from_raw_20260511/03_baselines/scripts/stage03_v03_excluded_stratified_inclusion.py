# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
ROOT = Path(os.environ.get("REBUILD_ROOT", str(PROJECT_ROOT / "05_rebuild_from_raw_20260511")))
SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage03_v03_vehicle_only_inclusion_ablation as incl  # noqa: E402


OUT_ROOT = ROOT / "03_baselines" / "stage03_v03_excluded_stratified_inclusion"
DATASET_ROOT = ROOT / "03_processed_datasets" / "extreme_condition_v0_3_excluded_stratified_inclusion"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-19.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"
SERVER_RUNS = NOTES_DIR / "SERVER_RUNS_CN.md"

BASE_CATEGORIES = sorted(incl.CLEAN_CATEGORIES | {"manual_review"})
ALL_WITH_EXCLUDED = sorted(incl.CLEAN_CATEGORIES | {"manual_review", "excluded"})
DROP_COORDINATE_RISK_FEATURES = ["lateral_distance_selected"]

VARIANTS: list[dict[str, Any]] = [
    {
        "variant_id": "v03_plus_review_ref",
        "name_cn": "干净集 + 待复核（原特征参考）",
        "description_cn": "当前推荐训练范围，不加入 excluded，保留横向偏移特征，用作本轮参考。",
        "categories": BASE_CATEGORIES,
    },
    {
        "variant_id": "v03_plus_review_no_lateral",
        "name_cn": "干净集 + 待复核（去横向偏移）",
        "description_cn": "不加入 excluded，但去掉横向偏移特征，用于和后续 excluded 去横向偏移版本公平比较。",
        "categories": BASE_CATEGORIES,
        "drop_features": DROP_COORDINATE_RISK_FEATURES,
    },
    {
        "variant_id": "v03_plus_review_excluded_all_no_lateral",
        "name_cn": "干净集 + 待复核 + 全部 excluded（去横向偏移）",
        "description_cn": "加入全部可成窗 excluded，但去掉横向偏移特征，检查坐标跳变风险是否主要来自横向偏移输入。",
        "categories": ALL_WITH_EXCLUDED,
        "drop_features": DROP_COORDINATE_RISK_FEATURES,
    },
    {
        "variant_id": "v03_plus_review_excluded_low_mu_no_lateral",
        "name_cn": "干净集 + 待复核 + 低附着 excluded（去横向偏移）",
        "description_cn": "只加入 excluded 中低附着来源样本，检查低附着风险池是否可用。",
        "categories": ALL_WITH_EXCLUDED,
        "excluded_contexts": ["低附着"],
        "drop_features": DROP_COORDINATE_RISK_FEATURES,
    },
    {
        "variant_id": "v03_plus_review_excluded_curve_no_lateral",
        "name_cn": "干净集 + 待复核 + 弯道 excluded（去横向偏移）",
        "description_cn": "只加入 excluded 中弯道/曲率来源样本，检查弯道风险池是否可用。",
        "categories": ALL_WITH_EXCLUDED,
        "excluded_contexts": ["弯道/曲率"],
        "drop_features": DROP_COORDINATE_RISK_FEATURES,
    },
    {
        "variant_id": "v03_plus_review_excluded_roll_no_lateral",
        "name_cn": "干净集 + 待复核 + 横滚姿态 excluded（去横向偏移）",
        "description_cn": "只加入 excluded 中横滚/姿态来源样本，检查姿态风险池是否可用。",
        "categories": ALL_WITH_EXCLUDED,
        "excluded_contexts": ["横滚/姿态"],
        "drop_features": DROP_COORDINATE_RISK_FEATURES,
    },
    {
        "variant_id": "v03_plus_review_excluded_lateral_dyn_no_lateral",
        "name_cn": "干净集 + 待复核 + 横向动态 excluded（去横向偏移）",
        "description_cn": "只加入 excluded 中横向动态来源样本；该类数量很少，只作为风险检查。",
        "categories": ALL_WITH_EXCLUDED,
        "excluded_contexts": ["横向动态"],
        "drop_features": DROP_COORDINATE_RISK_FEATURES,
    },
]


def ensure_dirs() -> None:
    for path in [OUT_ROOT / "tables", REPORT_DIR, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def configure_inclusion_module() -> None:
    incl.OUT_ROOT = OUT_ROOT
    incl.DATASET_ROOT = DATASET_ROOT
    incl.REPORT_DIR = REPORT_DIR
    incl.NOTES_DIR = NOTES_DIR
    incl.DAILY_LOG = DAILY_LOG
    incl.ARTIFACT_INDEX = ARTIFACT_INDEX


def fmt_float(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.6f}"


def markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "variant_id",
        "name_cn",
        "sample_count",
        "test_best_model",
        "test_rmse_steer",
        "test_wrong_side_rate_large",
        "test_severe_amp_under_rate_large",
        "test_large_response_recall",
        "clean_subset_test_sample_rmse_aggregate",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for col in cols:
            vals.append(fmt_float(row[col]) if col.startswith("test_") or col.endswith("aggregate") else str(row[col]))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame) -> None:
    summary_path = OUT_ROOT / "tables" / "v03_excluded_stratified_inclusion_summary.csv"
    best = summary.sort_values("test_rmse_steer").iloc[0].to_dict()
    lines = [
        "# v0.3 excluded 分层加入实验（用户查看版）",
        "",
        "## 为什么做",
        "",
        "之前直接加入全部 excluded 后，结果比“干净集 + 待复核”差，说明 excluded 不是全坏，但里面有一部分样本会拉乱任务。本轮不直接丢弃 excluded，而是把它拆成低附着、弯道、横滚姿态、横向动态几类，并去掉最容易受坐标跳变影响的横向偏移输入，逐类看哪些样本可以重新纳入。",
        "",
        "## 本轮设置",
        "",
        "- 仍然只跑车辆输入，不加入连续驾驶风格、生理或脑电。",
        "- 训练/验证/测试划分沿用 v0.3 当前基线的记录级划分，尽量避免因为重新切分造成误判。",
        "- `去横向偏移` 指输入特征中去掉 `lateral_distance_selected`，因为 excluded 主要来自坐标连续性异常，直接使用横向偏移可能把坐标跳变当成可学习信号。",
        "- 评价仍看整体误差、大响应错侧率、严重幅值不足率、大响应召回，不只看一个 RMSE。",
        "",
        "## 汇总结果",
        "",
        markdown_table(summary),
        "",
        "## 当前自动判断",
        "",
        f"- 本轮整体 RMSE 最低的是 `{best['variant_id']}`，RMSE={fmt_float(best['test_rmse_steer'])}。",
        "- 是否真正采用某类 excluded，不能只看整体 RMSE；还要看错侧率、严重幅值不足率、大响应召回和坏样本图。",
        "- 如果某一类加入后整体误差改善，但错侧率或严重幅值不足率明显变差，应先标为“只可诊断，不直接进入正式训练”。",
        "",
        "## 产物位置",
        "",
        f"- 汇总表：`{summary_path}`",
        f"- 每个版本的指标、逐样本指标、预测图：`{OUT_ROOT}`",
    ]
    report_path = REPORT_DIR / "stage03_v03_excluded_stratified_inclusion_user_summary_cn.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def append_progress_notes(summary: pd.DataFrame) -> None:
    best = summary.sort_values("test_rmse_steer").iloc[0].to_dict()
    note = (
        "## 2026-05-19 v0.3 excluded 分层加入实验\n\n"
        "- 当前阶段：旧流程样本重筛后的车辆-only 基线继续审查。\n"
        "- 本轮动作：把 excluded 从“直接全量加入”改为“去横向偏移后按来源分层加入”。\n"
        f"- 当前自动最优：`{best['variant_id']}`，test RMSE={fmt_float(best['test_rmse_steer'])}。\n"
        f"- 用户查看版报告：`{REPORT_DIR / 'stage03_v03_excluded_stratified_inclusion_user_summary_cn.md'}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
        "- 注意：本轮仍然没有加入连续驾驶风格、生理或脑电，不能据此判断这些信息是否有效。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            if "## 2026-05-19 v0.3 excluded 分层加入实验" not in raw:
                path.write_text(raw.rstrip() + "\n\n" + note, encoding="utf-8")
    if ARTIFACT_INDEX.exists():
        raw = ARTIFACT_INDEX.read_text(encoding="utf-8")
        block = (
            "## v0.3 excluded 分层加入实验\n\n"
            f"- 用户查看版报告：`{REPORT_DIR / 'stage03_v03_excluded_stratified_inclusion_user_summary_cn.md'}`\n"
            f"- 汇总表：`{OUT_ROOT / 'tables' / 'v03_excluded_stratified_inclusion_summary.csv'}`\n"
            f"- 输出目录：`{OUT_ROOT}`\n"
        )
        if "## v0.3 excluded 分层加入实验" not in raw:
            ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")


def append_server_note() -> None:
    if not SERVER_RUNS.exists():
        return
    raw = SERVER_RUNS.read_text(encoding="utf-8")
    block = (
        "## 2026-05-19 v0.3 excluded 分层加入实验服务器记录\n\n"
        "- 服务器连接格式：`ssh -p 55060 root@connect.westc.seetacloud.com`，密码不记录。\n"
        "- 远程项目路径：`/root/autodl-tmp/data_process`。\n"
        "- 任务：运行 `stage03_v03_excluded_stratified_inclusion.py`，输出 excluded 分层加入结果。\n"
        "- 远程日志路径：待运行后补充。\n"
    )
    if "## 2026-05-19 v0.3 excluded 分层加入实验服务器记录" not in raw:
        SERVER_RUNS.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_inclusion_module()
    sample_split, session_split = incl.load_reference_split()
    results = []
    for variant in VARIANTS:
        print(f"run {variant['variant_id']}", flush=True)
        results.append(incl.run_variant(variant, sample_split, session_split))
    summary = pd.DataFrame(results)
    summary_path = OUT_ROOT / "tables" / "v03_excluded_stratified_inclusion_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(summary)
    append_progress_notes(summary)
    append_server_note()
    print(
        summary[
            [
                "variant_id",
                "sample_count",
                "test_best_model",
                "test_rmse_steer",
                "test_wrong_side_rate_large",
                "test_severe_amp_under_rate_large",
                "test_large_response_recall",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
