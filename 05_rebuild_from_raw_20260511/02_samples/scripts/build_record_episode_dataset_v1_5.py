#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build v1.5 record-level episode dataset with curve z-drop separated.

User feedback after v1.4:

- The large downward z-drop samples retained in v1.4 are actually curve-related.
- Curve cases should be judged separately instead of being mixed into the main
  extreme-condition training pool.

This script keeps v1.4 intact and adds a v1.5 decision layer. It does not train
any model.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

import build_record_episode_dataset_v1_3 as v13


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
V14_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_4_zdrop_reviewed"
V14_ALL = V14_ROOT / "tables" / "record_level_episodes_all_v1_4.csv"
OUT_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_5_curve_separated"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "review_panels_v1_5"
REPORT_PATH = ROOT / "09_reports" / "stage02_record_episode_reconstruction_v1_5_user_summary_cn.md"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-21.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, REPORT_PATH.parent, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def is_curve_context(row: pd.Series) -> bool:
    text = f"{row.get('road_module_names', '')}|{row.get('road_design_categories', '')}".lower()
    return bool(row.get("is_curve_context", False)) or ("curve" in text) or ("弯道" in text)


def classify_v1_5(row: pd.Series) -> tuple[str, str, str, bool, bool, bool, bool]:
    v14_decision = str(row.get("v1_4_decision", ""))
    curve = is_curve_context(row)

    if v14_decision == "train_z_drop_extreme_keep" and curve:
        reason = "用户复核后调整：高度大幅下降样本属于弯道上下文，需单独判断，不进入当前主训练候选"
        return "review_curve_z_drop_separate", reason, reason, False, True, False, False

    if v14_decision == "train_z_drop_extreme_keep":
        reason = "非弯道高度大幅下降极限样本，暂时保留为训练候选"
        return "train_z_drop_extreme_keep_noncurve", reason, reason, True, False, False, False

    if v14_decision == "train_target_extreme":
        reason = "继承 v1.4：目标极限事件，保留为当前主训练候选"
        return "train_target_extreme", reason, reason, True, False, False, False

    if v14_decision == "train_conservative_extreme":
        reason = "继承 v1.4：保守/弱操作极限样本，保留为当前主训练候选"
        return "train_conservative_extreme", reason, reason, True, False, False, False

    if v14_decision == "control_normal_or_curve":
        reason = "继承 v1.4：正常弯道或普通操控，仅保留为对照样本"
        return "control_normal_or_curve", reason, reason, False, False, True, False

    if bool(row.get("is_deferred_v1_4", False)):
        reason = "继承 v1.4：仍需复核或拆分，不进入当前主训练候选"
        return "defer_prior_review", reason, reason, False, True, False, False

    reason = "继承 v1.4：此前已舍弃或不适合作为当前主训练候选"
    return "discard_prior_review", reason, reason, False, False, False, True


def make_review_figures(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_panel_v1_5_path"] = ""
    cache = {}

    curve_z = df[df["v1_5_decision"].eq("review_curve_z_drop_separate")].copy()
    curve_z = curve_z.sort_values("z_drop_from_start_v1_4", ascending=False)
    for idx, row in curve_z.iterrows():
        out_path = FIG_DIR / "01_弯道高度下降_单独判断" / f"{idx:04d}_{row['episode_uid']}.png"
        if not out_path.exists():
            v13.plot_episode_v1_3(row, out_path, cache)
        if out_path.exists():
            df.at[idx, "review_panel_v1_5_path"] = str(out_path)

    train = df[df["v1_5_decision"].isin(["train_target_extreme", "train_conservative_extreme"])].copy()
    train = train.sort_values(["vehicle_score_peak", "condition_score_peak"], ascending=False)
    for idx, row in train.head(48).iterrows():
        out_path = FIG_DIR / "02_主训练候选抽查" / f"{idx:04d}_{row['episode_uid']}.png"
        if not out_path.exists():
            v13.plot_episode_v1_3(row, out_path, cache)
        if out_path.exists():
            df.at[idx, "review_panel_v1_5_path"] = str(out_path)
    return df


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "暂无。"
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for v in row.tolist():
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_tables(df: pd.DataFrame) -> None:
    df.to_csv(TABLE_DIR / "record_level_episodes_all_v1_5.csv", index=False, encoding="utf-8-sig")
    df[df["is_train_candidate_v1_5"]].to_csv(
        TABLE_DIR / "train_candidate_target_episodes_v1_5.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_5_decision"].eq("review_curve_z_drop_separate")].to_csv(
        TABLE_DIR / "review_curve_z_drop_separate_episodes_v1_5.csv", index=False, encoding="utf-8-sig"
    )
    df[df.apply(is_curve_context, axis=1)].to_csv(
        TABLE_DIR / "all_curve_context_episodes_v1_5.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_deferred_v1_5"]].to_csv(
        TABLE_DIR / "deferred_or_review_episodes_v1_5.csv", index=False, encoding="utf-8-sig"
    )
    summary = (
        df.groupby("v1_5_decision", dropna=False)
        .agg(v1_5_decision_cn=("v1_5_decision_cn", "first"), count=("v1_5_decision", "size"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    summary.to_csv(TABLE_DIR / "record_episode_v1_5_decision_summary.csv", index=False, encoding="utf-8-sig")


def write_report(df: pd.DataFrame) -> None:
    summary = pd.read_csv(TABLE_DIR / "record_episode_v1_5_decision_summary.csv")
    curve_z = df[df["v1_5_decision"].eq("review_curve_z_drop_separate")][
        [
            "episode_uid",
            "subject",
            "road_module_names",
            "episode_duration_s",
            "z_drop_from_start_v1_4",
            "v1_4_decision",
            "review_panel_v1_5_path",
        ]
    ].sort_values("z_drop_from_start_v1_4", ascending=False)
    train_n = int(df["is_train_candidate_v1_5"].fillna(False).astype(bool).sum())
    curve_z_n = len(curve_z)
    curve_all_n = int(df.apply(is_curve_context, axis=1).sum())
    text = f"""# 完整记录级 episode 样本集 v1.5：弯道高度下降单独判断

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 这次为什么改

用户继续复核 v1.4 后指出：v1.4 中保留的“高度大幅下降”样本实际上都来自弯道路段。弯道本身可能有道路高程变化和曲率引起的车身动态，因此不能把这些样本当作上下马路极限样本直接加入主训练。

所以 v1.5 把这 22 个高度大幅下降样本从主训练候选中拿出来，单独归入“弯道高度下降，单独判断”。

## v1.5 规则

- v1.4 中 `train_z_drop_extreme_keep` 且属于 `curve1/curve2` 或弯道上下文的样本，改为 `review_curve_z_drop_separate`。
- 这些样本不进入当前主训练候选，但不删除，后续可作为弯道专门任务或弯道复核池。
- v1.4 原本的目标极限事件和保守/弱操作极限事件继续作为主训练候选。

## 数量变化

- v1.5 主训练候选：{train_n}
- 弯道高度下降单独复核：{curve_z_n}
- 全部弯道上下文样本：{curve_all_n}

## v1.5 分类表

{md_table(summary)}

## 弯道高度下降单独复核样本

{md_table(curve_z)}

## 输出位置

- v1.5 全量表：`{TABLE_DIR / "record_level_episodes_all_v1_5.csv"}`
- v1.5 主训练候选：`{TABLE_DIR / "train_candidate_target_episodes_v1_5.csv"}`
- 弯道高度下降单独复核表：`{TABLE_DIR / "review_curve_z_drop_separate_episodes_v1_5.csv"}`
- 全部弯道上下文表：`{TABLE_DIR / "all_curve_context_episodes_v1_5.csv"}`
- 复核图目录：`{FIG_DIR}`

## 当前建议

v1.5 更符合现在的判断：当前主训练集不再混入弯道高程下降片段。下一步如果要训练，可以先用 v1.5 主训练候选跑车辆-only；弯道样本另起一个“弯道专门判断/弯道专门模型”分支，不要和上下马路、低附着、避让事件混在一起。

本轮没有训练模型。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
    (OUT_ROOT / "record_episode_dataset_v1_5_summary_cn.md").write_text(text, encoding="utf-8")


def append_notes(df: pd.DataFrame) -> None:
    train_n = int(df["is_train_candidate_v1_5"].fillna(False).astype(bool).sum())
    curve_z_n = int(df["v1_5_decision"].eq("review_curve_z_drop_separate").sum())
    block = (
        "## 2026-05-21 完整记录级 episode 样本集 v1.5\n\n"
        "- 为什么做：用户复核后指出 v1.4 保留的高度大幅下降样本实际都是弯道路段，应单独判断，不应混入主训练候选。\n"
        "- 本轮动作：把 v1.4 的 `train_z_drop_extreme_keep` 且属于弯道上下文的样本改为 `review_curve_z_drop_separate`；本轮不训练模型。\n"
        f"- v1.5 主训练候选：{train_n}；弯道高度下降单独复核：{curve_z_n}。\n"
        f"- 用户查看版报告：`{REPORT_PATH}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        if "## 2026-05-21 完整记录级 episode 样本集 v1.5" not in raw:
            path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")
    artifact = (
        "## 2026-05-21 完整记录级 episode 样本集 v1.5\n\n"
        f"- 用户查看版报告：`{REPORT_PATH}`\n"
        f"- 全量表：`{TABLE_DIR / 'record_level_episodes_all_v1_5.csv'}`\n"
        f"- 主训练候选：`{TABLE_DIR / 'train_candidate_target_episodes_v1_5.csv'}`\n"
        f"- 弯道高度下降单独复核：`{TABLE_DIR / 'review_curve_z_drop_separate_episodes_v1_5.csv'}`\n"
        f"- 全部弯道上下文表：`{TABLE_DIR / 'all_curve_context_episodes_v1_5.csv'}`\n"
        f"- 复核图目录：`{FIG_DIR}`\n"
    )
    raw = ARTIFACT_INDEX.read_text(encoding="utf-8") if ARTIFACT_INDEX.exists() else ""
    if "## 2026-05-21 完整记录级 episode 样本集 v1.5" not in raw:
        ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not V14_ALL.exists():
        raise FileNotFoundError(V14_ALL)
    df = pd.read_csv(V14_ALL, encoding="utf-8-sig", low_memory=False)
    decisions = df.apply(classify_v1_5, axis=1, result_type="expand")
    decisions.columns = [
        "v1_5_decision",
        "v1_5_decision_cn",
        "v1_5_decision_detail_cn",
        "is_train_candidate_v1_5",
        "is_deferred_v1_5",
        "is_control_candidate_v1_5",
        "is_discarded_v1_5",
    ]
    df = pd.concat([df, decisions], axis=1)
    df = make_review_figures(df)
    write_tables(df)
    write_report(df)
    append_notes(df)
    print(pd.read_csv(TABLE_DIR / "record_episode_v1_5_decision_summary.csv").to_string(index=False))
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
