#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create v1.1 reviewed episode tables from v1.0 record-level reconstruction.

The user reviewed v1.0 figures and judged that most samples are usable, while
the "needs review" buckets are mostly discardable. This script converts that
manual decision into explicit training/control/discard tables without changing
the original v1.0 detection outputs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"F:/data_set_process/data_process")
V10_ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511/02_samples/record_level_episode_reconstruction_v1_0"
OUT_ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511/02_samples/record_level_episode_reconstruction_v1_1_reviewed"
USER_REPORT = PROJECT_ROOT / "05_rebuild_from_raw_20260511/09_reports/stage02_record_episode_reconstruction_v1_1_user_summary_cn.md"

TRAIN_GROUPS = {"core_extreme", "conservative_extreme", "secondary"}
CONTROL_GROUPS = {"normal_or_curve"}
DISCARD_GROUPS = {"review"}


def scan_figures() -> pd.DataFrame:
    rows = []
    review_root = V10_ROOT / "figures/review_panels"
    for path in review_root.rglob("*.png"):
        rows.append(
            {
                "episode_uid": path.stem,
                "review_panel_path": str(path),
            }
        )
    review_df = pd.DataFrame(rows)
    if not review_df.empty:
        review_df = (
            review_df.groupby("episode_uid", as_index=False)["review_panel_path"]
            .agg(lambda values: ";".join(sorted(set(map(str, values)))))
        )

    rows = []
    trajectory_root = V10_ROOT / "figures/trajectory_3d_static"
    for path in trajectory_root.glob("*.png"):
        stem = path.stem
        if stem.endswith("_3d"):
            stem = stem[:-3]
        rows.append(
            {
                "episode_uid": stem,
                "trajectory_3d_path": str(path),
            }
        )
    traj_df = pd.DataFrame(rows)
    if not traj_df.empty:
        traj_df = (
            traj_df.groupby("episode_uid", as_index=False)["trajectory_3d_path"]
            .agg(lambda values: ";".join(sorted(set(map(str, values)))))
        )

    if review_df.empty and traj_df.empty:
        return pd.DataFrame(columns=["episode_uid", "review_panel_path", "trajectory_3d_path"])
    if review_df.empty:
        merged = traj_df
    elif traj_df.empty:
        merged = review_df
    else:
        merged = review_df.merge(traj_df, on="episode_uid", how="outer")
    return merged


def add_decision_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def decision(row: pd.Series) -> tuple[str, str, bool, bool, bool]:
        group_id = str(row.get("episode_group_id", ""))
        group_cn = str(row.get("episode_group_cn", ""))
        if group_id in TRAIN_GROUPS:
            return (
                "保留为主训练候选",
                "用户已初步复核，核心/保守弱操作/次级样本整体可保留",
                True,
                False,
                False,
            )
        if group_id in CONTROL_GROUPS:
            return (
                "保留为对照样本，不进入主训练",
                "正常弯道或普通操控可用于对照，但不作为极限工况主训练样本",
                False,
                True,
                False,
            )
        if group_id in DISCARD_GROUPS or "复核" in group_cn:
            return (
                "暂不进入训练",
                "用户复核后认为需要复核类基本可舍去，先整体降为舍弃/暂缓",
                False,
                False,
                True,
            )
        return (
            "暂不进入训练",
            "未知分组，保守暂缓",
            False,
            False,
            True,
        )

    decisions = df.apply(decision, axis=1, result_type="expand")
    decisions.columns = [
        "manual_decision_v1_1",
        "manual_decision_reason_v1_1",
        "is_train_candidate_v1_1",
        "is_control_candidate_v1_1",
        "is_discarded_v1_1",
    ]
    return pd.concat([df, decisions], axis=1)


def bool_sum(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].fillna(False).astype(bool).sum())


def write_summary(df: pd.DataFrame, out_root: Path) -> None:
    tables = out_root / "tables"
    report_path = out_root / "record_episode_reviewed_summary_v1_1.md"

    train_df = df[df["is_train_candidate_v1_1"]].copy()
    control_df = df[df["is_control_candidate_v1_1"]].copy()
    discard_df = df[df["is_discarded_v1_1"]].copy()

    group_summary = (
        df.groupby(["manual_decision_v1_1", "episode_group_id", "episode_group_cn"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["manual_decision_v1_1", "episode_group_id", "episode_group_cn"])
    )
    context_summary = pd.DataFrame(
        [
            {
                "scope": "主训练候选",
                "count": len(train_df),
                "low_mu": bool_sum(train_df, "is_low_mu_context"),
                "curve": bool_sum(train_df, "is_curve_context"),
                "roll": bool_sum(train_df, "is_roll_context"),
                "lateral_dynamic": bool_sum(train_df, "is_lateral_dynamic_context"),
            },
            {
                "scope": "对照样本",
                "count": len(control_df),
                "low_mu": bool_sum(control_df, "is_low_mu_context"),
                "curve": bool_sum(control_df, "is_curve_context"),
                "roll": bool_sum(control_df, "is_roll_context"),
                "lateral_dynamic": bool_sum(control_df, "is_lateral_dynamic_context"),
            },
            {
                "scope": "舍弃/暂缓",
                "count": len(discard_df),
                "low_mu": bool_sum(discard_df, "is_low_mu_context"),
                "curve": bool_sum(discard_df, "is_curve_context"),
                "roll": bool_sum(discard_df, "is_roll_context"),
                "lateral_dynamic": bool_sum(discard_df, "is_lateral_dynamic_context"),
            },
        ]
    )
    subject_summary = (
        df.groupby(["manual_decision_v1_1", "subject"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["manual_decision_v1_1", "subject"])
    )

    group_summary.to_csv(tables / "record_episode_review_decision_summary_v1_1.csv", index=False, encoding="utf-8-sig")
    context_summary.to_csv(tables / "record_episode_review_context_summary_v1_1.csv", index=False, encoding="utf-8-sig")
    subject_summary.to_csv(tables / "record_episode_review_subject_summary_v1_1.csv", index=False, encoding="utf-8-sig")

    text = f"""# 完整记录级 episode 复核后样本集 v1.1

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 人工复核决策

用户查看 v1.0 复核图后给出的判断是：大部分自动筛出的极限/近极限 episode 可以继续保留；“需要复核”和“边界复核”类基本可以舍去。因此 v1.1 不重新检测 episode，只把 v1.0 候选库整理成更清晰的训练/对照/舍弃三类。

## 数量

- v1.0 episode 总数：{len(df)}
- v1.1 主训练候选：{len(train_df)}
- v1.1 对照样本：{len(control_df)}
- v1.1 舍弃/暂缓：{len(discard_df)}

## 主训练候选保留规则

保留：

- 核心极限样本；
- 保守/弱操作极限样本；
- 次级训练样本。

不进入主训练：

- 需要复核；
- 边界复核样本；
- 正常弯道或普通操控。

其中正常弯道或普通操控不是删除，而是单独作为对照样本保存。

## 上下文覆盖

| 范围 | 数量 | 低附着 | 弯道 | 横滚/姿态 | 横向动态 |
|---|---:|---:|---:|---:|---:|
| 主训练候选 | {len(train_df)} | {bool_sum(train_df, "is_low_mu_context")} | {bool_sum(train_df, "is_curve_context")} | {bool_sum(train_df, "is_roll_context")} | {bool_sum(train_df, "is_lateral_dynamic_context")} |
| 对照样本 | {len(control_df)} | {bool_sum(control_df, "is_low_mu_context")} | {bool_sum(control_df, "is_curve_context")} | {bool_sum(control_df, "is_roll_context")} | {bool_sum(control_df, "is_lateral_dynamic_context")} |
| 舍弃/暂缓 | {len(discard_df)} | {bool_sum(discard_df, "is_low_mu_context")} | {bool_sum(discard_df, "is_curve_context")} | {bool_sum(discard_df, "is_roll_context")} | {bool_sum(discard_df, "is_lateral_dynamic_context")} |

## 输出位置

- 全量带复核决策表：`{tables / "record_level_episodes_all_reviewed_v1_1.csv"}`
- 主训练候选表：`{tables / "train_candidate_extreme_episodes_v1_1.csv"}`
- 对照样本表：`{tables / "control_normal_or_curve_episodes_v1_1.csv"}`
- 舍弃/暂缓表：`{tables / "discarded_review_episodes_v1_1.csv"}`
- 分组统计表：`{tables / "record_episode_review_decision_summary_v1_1.csv"}`
- 复核图索引：`{tables / "record_episode_figure_index_v1_1.csv"}`

## 下一步

v1.1 已经可以作为下一轮车辆-only 数据集构建入口。但正式训练前建议先从主训练候选里再抽查一小批核心极限和保守/弱操作样本，确认“需要复核类整体舍弃”没有误删大量有效样本。
"""
    report_path.write_text(text, encoding="utf-8")
    USER_REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    tables = OUT_ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    all_path = V10_ROOT / "tables/record_level_episodes_all_v1_0.csv"
    if not all_path.exists():
        raise FileNotFoundError(all_path)
    df = pd.read_csv(all_path)

    figures = scan_figures()
    if not figures.empty:
        df = df.merge(figures, on="episode_uid", how="left")
    else:
        df["review_panel_path"] = ""
        df["trajectory_3d_path"] = ""
    df["has_review_panel_v1_1"] = df["review_panel_path"].fillna("").astype(str).ne("")
    df["has_trajectory_3d_v1_1"] = df["trajectory_3d_path"].fillna("").astype(str).ne("")

    df = add_decision_columns(df)

    train_df = df[df["is_train_candidate_v1_1"]].copy()
    control_df = df[df["is_control_candidate_v1_1"]].copy()
    discard_df = df[df["is_discarded_v1_1"]].copy()

    df.to_csv(tables / "record_level_episodes_all_reviewed_v1_1.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(tables / "train_candidate_extreme_episodes_v1_1.csv", index=False, encoding="utf-8-sig")
    train_df[train_df["episode_group_id"] == "core_extreme"].to_csv(
        tables / "train_candidate_core_extreme_v1_1.csv", index=False, encoding="utf-8-sig"
    )
    train_df[train_df["episode_group_id"] == "conservative_extreme"].to_csv(
        tables / "train_candidate_conservative_extreme_v1_1.csv", index=False, encoding="utf-8-sig"
    )
    train_df[train_df["episode_group_id"] == "secondary"].to_csv(
        tables / "train_candidate_secondary_v1_1.csv", index=False, encoding="utf-8-sig"
    )
    control_df.to_csv(tables / "control_normal_or_curve_episodes_v1_1.csv", index=False, encoding="utf-8-sig")
    discard_df.to_csv(tables / "discarded_review_episodes_v1_1.csv", index=False, encoding="utf-8-sig")

    figure_cols = ["episode_uid", "episode_group_id", "episode_group_cn", "manual_decision_v1_1", "review_panel_path", "trajectory_3d_path"]
    df[figure_cols].to_csv(tables / "record_episode_figure_index_v1_1.csv", index=False, encoding="utf-8-sig")

    write_summary(df, OUT_ROOT)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(all_path),
        "total": int(len(df)),
        "train_candidate": int(len(train_df)),
        "control": int(len(control_df)),
        "discarded": int(len(discard_df)),
        "policy": {
            "train_groups": sorted(TRAIN_GROUPS),
            "control_groups": sorted(CONTROL_GROUPS),
            "discard_groups": sorted(DISCARD_GROUPS),
        },
    }
    (OUT_ROOT / "review_policy_v1_1.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
