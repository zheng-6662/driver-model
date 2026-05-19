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


OUT_ROOT = ROOT / "03_baselines" / "stage03_v03_fast_weakpost_temp_train"
DATASET_ROOT = ROOT / "03_processed_datasets" / "extreme_condition_v0_3_fast_weakpost_temp_train"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-19.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"

FAST_ANCHOR_AUDIT = (
    ROOT
    / "02_samples"
    / "extreme_condition_episodes_v0_3"
    / "tables"
    / "fast_steer_anchor_timing_audit_v0_3.csv"
)
SUMMARY_PATH = OUT_ROOT / "tables" / "v03_fast_weakpost_temp_train_summary.csv"
EXTRA_UID_PATH = OUT_ROOT / "tables" / "v03_fast_weakpost_extra_episode_uids.csv"
SOURCE_DIAG_PATH = OUT_ROOT / "tables" / "v03_fast_weakpost_source_test_diagnostics.csv"

BASE_CATEGORIES = sorted(incl.CLEAN_CATEGORIES | {"manual_review"})
DROP_COORDINATE_RISK_FEATURES = ["lateral_distance_selected"]


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


def read_fast_anchor_groups() -> tuple[list[str], list[str], pd.DataFrame]:
    audit = pd.read_csv(FAST_ANCHOR_AUDIT, encoding="utf-8-sig", low_memory=False)
    audit["episode_uid"] = audit["episode_uid"].astype(str)
    audit["anchor_timing_label"] = audit["anchor_timing_label"].astype(str)

    weak_post = audit.loc[
        audit["anchor_timing_label"].eq("FAST_STEER_WEAK_POST_RESPONSE"), "episode_uid"
    ].dropna().astype(str).tolist()
    usable_fast = audit.loc[
        audit["anchor_timing_label"].eq("ANCHOR_USABLE_FAST_RESPONSE"), "episode_uid"
    ].dropna().astype(str).tolist()
    excluded_late = audit.loc[
        audit["anchor_timing_label"].isin(["EXCLUDE_LATE_ANCHOR_STABILIZED", "RISK_LATE_ANCHOR_REVIEW"]),
        "episode_uid",
    ].dropna().astype(str).tolist()

    extra = audit[
        audit["anchor_timing_label"].isin(["FAST_STEER_WEAK_POST_RESPONSE", "ANCHOR_USABLE_FAST_RESPONSE"])
    ].copy()
    extra["temporary_train_decision_cn"] = np.where(
        extra["anchor_timing_label"].eq("FAST_STEER_WEAK_POST_RESPONSE"),
        "本轮临时加入：锚点后响应弱，但用户认为可先看训练效果",
        "本轮临时加入：锚点后仍有快速响应",
    )
    extra["late_anchor_excluded_count_reference"] = len(excluded_late)
    return weak_post, usable_fast, extra


def build_variants(weak_post: list[str], usable_fast: list[str]) -> list[dict[str, Any]]:
    weak_only = sorted(set(weak_post))
    weak_plus_usable = sorted(set(weak_post) | set(usable_fast))
    return [
        {
            "variant_id": "v03_plus_review_ref_no_lateral",
            "name_cn": "干净集 + 待复核（去横向偏移）",
            "description_cn": "当前较稳的车辆-only 对照：干净集加待复核样本，去掉横向偏移输入。",
            "categories": BASE_CATEGORIES,
            "drop_features": DROP_COORDINATE_RISK_FEATURES,
        },
        {
            "variant_id": "v03_plus_review_fast_weakpost_no_lateral",
            "name_cn": "干净集 + 待复核 + 锚点后响应弱（去横向偏移）",
            "description_cn": "临时加入 FAST_STEER_WEAK_POST_RESPONSE，用来检查这些弱后续响应样本是否能扩大训练集并改善车辆-only 基线。",
            "categories": BASE_CATEGORIES,
            "drop_features": DROP_COORDINATE_RISK_FEATURES,
            "extra_episode_uids": weak_only,
            "extra_episode_source": "FAST_STEER_WEAK_POST_RESPONSE",
        },
        {
            "variant_id": "v03_weakpost_usable_nolat",
            "name_cn": "干净集 + 待复核 + 锚点后响应弱 + 可用快速响应（去横向偏移）",
            "description_cn": "在上一版基础上再加入 1 个 ANCHOR_USABLE_FAST_RESPONSE，检查可用快速响应样本是否需要一起保留。",
            "categories": BASE_CATEGORIES,
            "drop_features": DROP_COORDINATE_RISK_FEATURES,
            "extra_episode_uids": weak_plus_usable,
            "extra_episode_source": "FAST_STEER_WEAK_POST_RESPONSE_PLUS_ANCHOR_USABLE_FAST_RESPONSE",
        },
        {
            "variant_id": "v03_weakpost_with_lateral",
            "name_cn": "干净集 + 待复核 + 锚点后响应弱（保留横向偏移）",
            "description_cn": "同样临时加入锚点后响应弱样本，但保留横向偏移输入，用来判断去横向偏移是否影响结论。",
            "categories": BASE_CATEGORIES,
            "extra_episode_uids": weak_only,
            "extra_episode_source": "FAST_STEER_WEAK_POST_RESPONSE",
        },
    ]


def load_dataset_summary(variant_id: str) -> dict[str, Any]:
    path = DATASET_ROOT / variant_id / "logs" / f"{variant_id}_dataset_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing_result(variant_id: str) -> dict[str, Any] | None:
    path = OUT_ROOT / variant_id / "logs" / f"{variant_id}_summary.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    result = payload.get("result")
    return result if isinstance(result, dict) else None


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
        "extra_episode_count",
        "test_best_model",
        "test_rmse_steer",
        "test_primary_rmse_0_2s",
        "test_tail_rmse_2_5s",
        "test_wrong_side_rate_large",
        "test_severe_amp_under_rate_large",
        "test_large_response_recall",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for col in cols:
            value = row[col]
            vals.append(fmt(value) if col.startswith("test_") else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def extra_split_counts(variant_id: str) -> dict[str, int]:
    path = DATASET_ROOT / variant_id / "tables" / f"{variant_id}_manifest.csv"
    if not path.exists():
        return {}
    meta = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "temporary_inclusion_source" not in meta.columns:
        return {}
    extra = meta[meta["temporary_inclusion_source"].astype(str).ne("category_rule")]
    return {str(k): int(v) for k, v in extra["split"].value_counts().to_dict().items()}


def write_source_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        variant_id = str(row["variant_id"])
        per_sample_path = OUT_ROOT / variant_id / "tables" / f"{variant_id}_per_sample_metrics.csv"
        metrics_path = OUT_ROOT / variant_id / "tables" / f"{variant_id}_baseline_metrics.csv"
        manifest_path = DATASET_ROOT / variant_id / "tables" / f"{variant_id}_manifest.csv"
        if not per_sample_path.exists() or not metrics_path.exists() or not manifest_path.exists():
            continue
        per_sample = pd.read_csv(per_sample_path, encoding="utf-8-sig", low_memory=False)
        metrics = pd.read_csv(metrics_path, encoding="utf-8-sig", low_memory=False)
        manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", low_memory=False)[
            ["sample_id", "temporary_inclusion_source"]
        ]
        best_model = str(metrics[metrics["split"].eq("test")].sort_values("rmse_steer").iloc[0]["model_name"])
        test = per_sample[
            per_sample["split"].eq("test") & per_sample["model_name"].eq(best_model)
        ].merge(manifest, on="sample_id", how="left")
        for source, group in test.groupby("temporary_inclusion_source", dropna=False):
            large = group[group["large_response"].astype(bool)]
            rows.append(
                {
                    "variant_id": variant_id,
                    "source": str(source),
                    "test_n": int(len(group)),
                    "test_large_n": int(len(large)),
                    "test_rmse_aggregate": float(np.sqrt(np.nanmean(np.square(group["sample_rmse"])))),
                    "large_wrong_side_rate": float(large["wrong_side_large"].mean()) if len(large) else float("nan"),
                    "large_severe_amp_under_rate": float(large["severe_amp_under_large"].mean())
                    if len(large)
                    else float("nan"),
                }
            )
    diag = pd.DataFrame(rows)
    diag.to_csv(SOURCE_DIAG_PATH, index=False, encoding="utf-8-sig")
    return diag


def write_report(summary: pd.DataFrame, extra: pd.DataFrame, source_diag: pd.DataFrame) -> None:
    best = summary.sort_values("test_rmse_steer").iloc[0].to_dict()
    ref = summary[summary["variant_id"].eq("v03_plus_review_ref_no_lateral")].iloc[0].to_dict()
    weak = summary[summary["variant_id"].eq("v03_plus_review_fast_weakpost_no_lateral")].iloc[0].to_dict()
    lateral = summary[summary["variant_id"].eq("v03_weakpost_with_lateral")].iloc[0].to_dict()
    weak_count = int(extra["anchor_timing_label"].eq("FAST_STEER_WEAK_POST_RESPONSE").sum())
    usable_count = int(extra["anchor_timing_label"].eq("ANCHOR_USABLE_FAST_RESPONSE").sum())
    main_row = summary[summary["variant_id"].eq("v03_plus_review_fast_weakpost_no_lateral")]
    extra_added = int(main_row.iloc[0]["extra_episode_count"]) if len(main_row) else 0
    already_in_base = max(weak_count - extra_added, 0)
    split_counts = extra_split_counts("v03_plus_review_fast_weakpost_no_lateral")
    split_text = " / ".join(f"{k}:{v}" for k, v in sorted(split_counts.items())) if split_counts else "NA"
    diag_text = ""
    if not source_diag.empty:
        main_diag = source_diag[
            source_diag["variant_id"].eq("v03_plus_review_fast_weakpost_no_lateral")
            & source_diag["source"].astype(str).eq("FAST_STEER_WEAK_POST_RESPONSE")
        ]
        if len(main_diag):
            d = main_diag.iloc[0].to_dict()
            diag_text = (
                f"额外新增样本在测试集中有 {int(d['test_n'])} 个，"
                f"这几个样本自身 RMSE 聚合约 {fmt(d['test_rmse_aggregate'])}；"
                "但数量太少，只能作为方向性参考。"
            )
    report = REPORT_DIR / "stage03_v03_fast_weakpost_temp_train_user_summary_cn.md"
    lines = [
        "# v0.3 临时加入“锚点后响应弱”样本训练结果",
        "",
        "## 为什么做",
        "",
        "你复核图片后认为，有些“锚点后响应弱”的样本虽然不像强极限工况，但也可能代表保守驾驶员、小幅维持、弱响应或轻微姿态变化。直接丢掉会让数据集太小，也可能把保守反应这类驾驶行为排除掉。因此本轮不改模型，只把这部分样本临时加入车辆-only 训练，看它到底是补充信息，还是拉乱任务。",
        "",
        "## 本轮怎么合并",
        "",
        "- 基础范围：干净集四类 + 待复核样本，也就是之前相对更稳的训练范围。",
        f"- `FAST_STEER_WEAK_POST_RESPONSE` 候选池共 {weak_count} 个。",
        f"- 其中 {already_in_base} 个本来已经在基础范围内，真正额外新增进入本轮训练的是 {extra_added} 个。",
        f"- 这 {extra_added} 个额外新增样本在当前旧划分中的分布是：{split_text}。",
        f"- 另有 `ANCHOR_USABLE_FAST_RESPONSE` 共 {usable_count} 个，单独做了一个加 1 个样本的对照。",
        "- 继续排除：明显锚点偏晚、锚点后已经稳定的样本，不加入。",
        "- 主要版本去掉 `lateral_distance_selected`，避免横向偏移坐标跳变把任务带偏；另跑一个保留横向偏移的对照。",
        "",
        "## 结果表",
        "",
        markdown_table(summary),
        "",
        "## 本轮结论",
        "",
        f"- 去横向偏移主版本加入 16 个额外弱后续响应样本后，整体 RMSE 从 {fmt(ref['test_rmse_steer'])} 降到 {fmt(weak['test_rmse_steer'])}，属于小幅改善。",
        f"- 但它的大响应严重幅值不足率从 {fmt(ref['test_severe_amp_under_rate_large'])} 升到 {fmt(weak['test_severe_amp_under_rate_large'])}，大响应召回从 {fmt(ref['test_large_response_recall'])} 降到 {fmt(weak['test_large_response_recall'])}，说明它没有解决“大幅动作预测太轻”的核心问题。",
        f"- 保留横向偏移后 RMSE 进一步降到 {fmt(lateral['test_rmse_steer'])}，错侧率也降低，但大响应召回降到 {fmt(lateral['test_large_response_recall'])}，所以它更像是全局拟合改善，不一定更符合极限工况物理目标。",
        "- 当前建议：可以暂时保留“锚点后响应弱”作为扩充/保守响应样本池，但不要把它升级为极限姿态核心正样本；下一步应按图片复核结果，把它拆成“弱但有效车辆响应”和“只是轻微方向盘维持”两类。",
        "",
        "## 自动读法",
        "",
        f"- 本轮整体 RMSE 最低的是 `{best['variant_id']}`，test RMSE={fmt(best['test_rmse_steer'])}。",
        f"- {diag_text}" if diag_text else "",
        "- 如果加入锚点后响应弱后 RMSE 下降，同时大响应错侧率、严重幅值不足率不恶化，说明这部分样本可以继续保留。",
        "- 如果 RMSE 下降但大响应物理指标变差，说明它可能只是在普通样本上增加数量，不适合作为极限姿态主训练集。",
        "- 如果保留横向偏移版本明显变差，说明横向偏移坐标风险仍然需要谨慎。",
        "",
        "## 可以查看的图",
        "",
        f"- 每个版本固定样本预测图和坏样本预测图：`{OUT_ROOT}`",
        f"- 临时加入的 episode 清单：`{EXTRA_UID_PATH}`",
        f"- 按临时新增来源拆开的测试诊断：`{SOURCE_DIAG_PATH}`",
        f"- 汇总表：`{SUMMARY_PATH}`",
    ]
    report.write_text("\n".join(line for line in lines if line != ""), encoding="utf-8")


def append_notes(summary: pd.DataFrame) -> None:
    best = summary.sort_values("test_rmse_steer").iloc[0].to_dict()
    block = (
        "## 2026-05-19 v0.3 临时加入锚点后响应弱样本\n\n"
        "- 当前阶段：车辆-only 样本范围继续审查，不涉及连续风格、生理或脑电。\n"
        "- 本轮动作：在“干净集 + 待复核”基础上，临时加入 `FAST_STEER_WEAK_POST_RESPONSE` 样本，并重跑车辆-only 基线。\n"
        f"- 当前整体 RMSE 最低版本：`{best['variant_id']}`，test RMSE={fmt(best['test_rmse_steer'])}。\n"
        f"- 用户查看版报告：`{REPORT_DIR / 'stage03_v03_fast_weakpost_temp_train_user_summary_cn.md'}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
        "- 注意：这是探索性样本合并实验，不能直接证明最终样本定义正确；需要继续看预测图和大响应物理指标。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            if "## 2026-05-19 v0.3 临时加入锚点后响应弱样本" not in raw:
                path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")
    if ARTIFACT_INDEX.exists():
        raw = ARTIFACT_INDEX.read_text(encoding="utf-8")
        artifact_block = (
            "## v0.3 临时加入锚点后响应弱样本训练\n\n"
            f"- 用户查看版报告：`{REPORT_DIR / 'stage03_v03_fast_weakpost_temp_train_user_summary_cn.md'}`\n"
            f"- 汇总表：`{SUMMARY_PATH}`\n"
            f"- 临时加入 episode 清单：`{EXTRA_UID_PATH}`\n"
            f"- 输出目录：`{OUT_ROOT}`\n"
        )
        if "## v0.3 临时加入锚点后响应弱样本训练" not in raw:
            ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact_block, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_inclusion_module()
    weak_post, usable_fast, extra = read_fast_anchor_groups()
    extra.to_csv(EXTRA_UID_PATH, index=False, encoding="utf-8-sig")

    sample_split, session_split = incl.load_reference_split()
    variants = build_variants(weak_post, usable_fast)
    rows = []
    for variant in variants:
        existing = load_existing_result(str(variant["variant_id"]))
        if existing is not None:
            print(f"reuse {variant['variant_id']}", flush=True)
            result = existing
        else:
            print(f"run {variant['variant_id']}", flush=True)
            result = incl.run_variant(variant, sample_split, session_split)
        dataset_summary = load_dataset_summary(str(variant["variant_id"]))
        result["extra_episode_count"] = int(dataset_summary.get("extra_episode_count", 0))
        result["dropped_count"] = int(dataset_summary.get("dropped_count", 0))
        result["split_counts_json"] = json.dumps(dataset_summary.get("split_counts", {}), ensure_ascii=False)
        result["category_counts_json"] = json.dumps(dataset_summary.get("category_counts", {}), ensure_ascii=False)
        rows.append(result)

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    source_diag = write_source_diagnostics(summary)
    write_report(summary, extra, source_diag)
    append_notes(summary)
    print(
        summary[
            [
                "variant_id",
                "sample_count",
                "extra_episode_count",
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
