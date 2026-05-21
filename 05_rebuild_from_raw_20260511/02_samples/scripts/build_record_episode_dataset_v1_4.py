#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build v1.4 record-level episode dataset after user review of z-drop cases.

User feedback after v1.3:

- Most samples in the height-z suspected folder are indeed off-road / road-edge
  processes.
- However, the cases with a clear large downward z drop should be retained for
  now because they represent an extreme road departure / elevation-drop
  condition.
- Other off-road / road-edge cases without such a large downward z drop can be
  discarded from the current training candidate pool.

This script keeps v1.3 intact and adds a v1.4 decision layer. It does not train
any model.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_record_episode_dataset_v1_2 as v12
import build_record_episode_dataset_v1_3 as v13


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
V13_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_3_cleaned"
V13_ALL = V13_ROOT / "tables" / "record_level_episodes_all_v1_3.csv"
OUT_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_4_zdrop_reviewed"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "review_panels_v1_4"
REPORT_PATH = ROOT / "09_reports" / "stage02_record_episode_reconstruction_v1_4_user_summary_cn.md"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-21.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"


Z_DROP_KEEP_FLOOR_M = 2.0
Z_START_WINDOW_S = 0.5
MAX_DISCARD_REVIEW_FIGURES = 48


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, REPORT_PATH.parent, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def finite_numeric(values: Any) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def compute_zdrop_features(row: pd.Series, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rec = v12.load_vehicle_record(str(row.get("vehicle_file", "")), cache)
    out = {
        "z_start_median_v1_4": np.nan,
        "z_min_after_start_v1_4": np.nan,
        "z_max_after_start_v1_4": np.nan,
        "z_drop_from_start_v1_4": np.nan,
        "z_rise_from_start_v1_4": np.nan,
        "z_drop_large_keep_v1_4": False,
    }
    if rec is None:
        return out
    start = float(row.get("episode_start_s", np.nan))
    end = float(row.get("episode_end_s", np.nan))
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return out
    t = rec["t"]
    z = rec["signals"]["z"]
    mask = np.isfinite(t) & np.isfinite(z) & (t >= start) & (t <= end)
    if mask.sum() < 5:
        return out
    x = t[mask] - start
    zz = z[mask]
    base = zz[x <= Z_START_WINDOW_S]
    if base.size < 3:
        base = zz[: min(10, len(zz))]
    z0 = float(np.nanmedian(base))
    z_min = float(np.nanmin(zz))
    z_max = float(np.nanmax(zz))
    z_drop = z0 - z_min
    z_rise = z_max - z0
    out.update(
        {
            "z_start_median_v1_4": z0,
            "z_min_after_start_v1_4": z_min,
            "z_max_after_start_v1_4": z_max,
            "z_drop_from_start_v1_4": z_drop,
            "z_rise_from_start_v1_4": z_rise,
            "z_drop_large_keep_v1_4": bool(math.isfinite(z_drop) and z_drop >= Z_DROP_KEEP_FLOOR_M),
        }
    )
    return out


def classify_v1_4(row: pd.Series) -> tuple[str, str, str, bool, bool, bool, bool]:
    v13_decision = str(row.get("v1_3_decision", ""))
    is_roadedge = "roadedge" in v13_decision or "offroad" in v13_decision
    large_z_drop = bool(row.get("z_drop_large_keep_v1_4", False))

    if is_roadedge and large_z_drop:
        reason = "用户复核后保留：高度 z 相对 episode 开始大幅下降，作为高度大幅下降极限样本"
        detail = (
            f"用户复核后保留：高度 z 相对 episode 开始大幅下降，"
            f"z_drop={float(row.get('z_drop_from_start_v1_4', np.nan)):.2f}m，作为高度大幅下降极限样本"
        )
        return "train_z_drop_extreme_keep", reason, detail, True, False, False, False

    if is_roadedge:
        reason = "用户复核后抛弃：属于疑似上下马路/路边恢复，但没有明显大幅向下 z_drop，不进入当前训练候选"
        return "discard_roadedge_without_large_zdrop", reason, reason, False, False, False, True

    if bool(row.get("is_train_candidate_v1_3", False)):
        if str(row.get("v1_3_decision", "")) == "train_conservative_extreme":
            return (
                "train_conservative_extreme",
                "继承 v1.3：保守/弱操作极限样本，保留为训练候选",
                "继承 v1.3：保守/弱操作极限样本，保留为训练候选",
                True,
                False,
                False,
                False,
            )
        return (
            "train_target_extreme",
            "继承 v1.3：目标极限事件，保留为训练候选",
            "继承 v1.3：目标极限事件，保留为训练候选",
            True,
            False,
            False,
            False,
        )

    if bool(row.get("is_control_candidate_v1_3", False)):
        return (
            "control_normal_or_curve",
            "继承 v1.3：正常弯道或普通操控，仅保留为对照样本",
            "继承 v1.3：正常弯道或普通操控，仅保留为对照样本",
            False,
            False,
            True,
            False,
        )

    if bool(row.get("is_deferred_v1_3", False)):
        return (
            "defer_prior_review",
            "继承 v1.3：仍需要复核或拆分，暂不进入当前训练候选",
            "继承 v1.3：仍需要复核或拆分，暂不进入当前训练候选",
            False,
            True,
            False,
            False,
        )

    return (
        "discard_prior_review",
        "继承 v1.3：此前已舍弃/暂缓，不进入当前训练候选",
        "继承 v1.3：此前已舍弃/暂缓，不进入当前训练候选",
        False,
        False,
        False,
        True,
    )


def make_review_figures(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_panel_v1_4_path"] = ""
    cache: dict[str, dict[str, Any]] = {}

    keep = df[df["v1_4_decision"].eq("train_z_drop_extreme_keep")].copy()
    keep = keep.sort_values("z_drop_from_start_v1_4", ascending=False)
    for idx, row in keep.iterrows():
        out_path = FIG_DIR / "01_保留_高度大幅下降极限样本" / f"{idx:04d}_{row['episode_uid']}.png"
        if not out_path.exists():
            v13.plot_episode_v1_3(row, out_path, cache)
        if out_path.exists():
            df.at[idx, "review_panel_v1_4_path"] = str(out_path)

    discard = df[df["v1_4_decision"].eq("discard_roadedge_without_large_zdrop")].copy()
    discard = discard.sort_values(
        ["brake_peak_v1_3", "speed_drop_from_start_v1_3", "lat_offset_adjacent_jump_peak_v1_3"],
        ascending=False,
    )
    for idx, row in discard.head(MAX_DISCARD_REVIEW_FIGURES).iterrows():
        out_path = FIG_DIR / "02_抛弃_上下马路但无明显大幅下降" / f"{idx:04d}_{row['episode_uid']}.png"
        if not out_path.exists():
            v13.plot_episode_v1_3(row, out_path, cache)
        if out_path.exists():
            df.at[idx, "review_panel_v1_4_path"] = str(out_path)

    train = df[df["v1_4_decision"].isin(["train_target_extreme", "train_conservative_extreme"])].copy()
    train = train.sort_values(["vehicle_score_peak", "condition_score_peak"], ascending=False)
    for idx, row in train.head(40).iterrows():
        out_path = FIG_DIR / "03_普通保留训练样本抽查" / f"{idx:04d}_{row['episode_uid']}.png"
        if not out_path.exists():
            v13.plot_episode_v1_3(row, out_path, cache)
        if out_path.exists():
            df.at[idx, "review_panel_v1_4_path"] = str(out_path)

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
    df.to_csv(TABLE_DIR / "record_level_episodes_all_v1_4.csv", index=False, encoding="utf-8-sig")
    df[df["is_train_candidate_v1_4"]].to_csv(
        TABLE_DIR / "train_candidate_target_episodes_v1_4.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_4_decision"].eq("train_z_drop_extreme_keep")].to_csv(
        TABLE_DIR / "train_z_drop_extreme_keep_episodes_v1_4.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_4_decision"].eq("discard_roadedge_without_large_zdrop")].to_csv(
        TABLE_DIR / "discard_roadedge_without_large_zdrop_episodes_v1_4.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_deferred_v1_4"]].to_csv(
        TABLE_DIR / "deferred_or_review_episodes_v1_4.csv", index=False, encoding="utf-8-sig"
    )
    decision_summary = (
        df.groupby("v1_4_decision", dropna=False)
        .agg(v1_4_decision_cn=("v1_4_decision_cn", "first"), count=("v1_4_decision", "size"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    decision_summary.to_csv(TABLE_DIR / "record_episode_v1_4_decision_summary.csv", index=False, encoding="utf-8-sig")
    zdrop_summary = (
        df[df["v1_3_decision"].astype(str).str.contains("roadedge|offroad", na=False)]
        .assign(
            z_drop_bin=pd.cut(
                pd.to_numeric(df["z_drop_from_start_v1_4"], errors="coerce"),
                bins=[-np.inf, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf],
                labels=["<=0.5", "0.5-1", "1-2", "2-3", "3-5", ">5"],
            )
        )
        .groupby(["z_drop_bin", "v1_4_decision"], observed=False)
        .size()
        .reset_index(name="count")
    )
    zdrop_summary.to_csv(TABLE_DIR / "roadedge_zdrop_distribution_v1_4.csv", index=False, encoding="utf-8-sig")


def write_report(df: pd.DataFrame) -> None:
    decision = pd.read_csv(TABLE_DIR / "record_episode_v1_4_decision_summary.csv")
    zdrop_keep = df[df["v1_4_decision"].eq("train_z_drop_extreme_keep")].copy()
    zdrop_keep_view = zdrop_keep[
        [
            "episode_uid",
            "subject",
            "road_module_names",
            "episode_duration_s",
            "z_drop_from_start_v1_4",
            "z_start_median_v1_4",
            "z_min_after_start_v1_4",
            "v1_3_decision",
            "review_panel_v1_4_path",
        ]
    ].sort_values("z_drop_from_start_v1_4", ascending=False)
    train_n = int(df["is_train_candidate_v1_4"].fillna(False).astype(bool).sum())
    keep_n = len(zdrop_keep)
    discard_road_n = int(df["v1_4_decision"].eq("discard_roadedge_without_large_zdrop").sum())
    text = f"""# 完整记录级 episode 样本集 v1.4：保留高度大幅下降极限样本

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 这次为什么改

用户复核 v1.3 后认为：高度 `z` 文件夹里的多数样本确实是开下马路或路边恢复，整体判断方向是对的；但其中有一类 `z` 明显大幅向下掉的片段，应先保留，因为它们代表明显的高度突变/路外极限工况。其它没有明显大幅下降的上下马路/路边恢复片段，可以先抛弃，不进入当前训练候选。

## v1.4 规则

- 只在 v1.3 已经标为“疑似路边恢复或上下马路”的样本里重新筛。
- 计算 `z_drop_from_start = episode 开始后 0.5 秒内 z 中位数 - episode 内最低 z`。
- 若 `z_drop_from_start >= {Z_DROP_KEEP_FLOOR_M:.1f} m`，保留为 `train_z_drop_extreme_keep`。
- 其它路边恢复/上下马路样本标为 `discard_roadedge_without_large_zdrop`，不进入当前训练候选。
- v1.3 已经保留的目标极限事件和保守/弱操作极限事件继续保留。

## 数量变化

- v1.4 主训练候选总数：{train_n}
- 其中新增保留的高度大幅下降极限样本：{keep_n}
- 被抛弃的上下马路/路边恢复但无明显大幅下降样本：{discard_road_n}

## v1.4 分类表

{md_table(decision)}

## 保留的高度大幅下降样本

{md_table(zdrop_keep_view)}

## 输出位置

- v1.4 全量表：`{TABLE_DIR / "record_level_episodes_all_v1_4.csv"}`
- v1.4 主训练候选：`{TABLE_DIR / "train_candidate_target_episodes_v1_4.csv"}`
- 高度大幅下降保留样本：`{TABLE_DIR / "train_z_drop_extreme_keep_episodes_v1_4.csv"}`
- 上下马路但无明显大幅下降抛弃样本：`{TABLE_DIR / "discard_roadedge_without_large_zdrop_episodes_v1_4.csv"}`
- 复核图目录：`{FIG_DIR}`

## 当前建议

v1.4 比 v1.3 更贴合你的人工复核意见。下一步建议先看：

1. `01_保留_高度大幅下降极限样本`：确认这 22 个是否确实应该保留；
2. `02_抛弃_上下马路但无明显大幅下降`：抽查是否还有漏掉的可用样本；
3. 如果这两个文件夹大体符合直觉，再用 v1.4 主训练候选重跑车辆-only。

本轮没有训练模型。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
    (OUT_ROOT / "record_episode_dataset_v1_4_summary_cn.md").write_text(text, encoding="utf-8")


def append_notes(df: pd.DataFrame) -> None:
    train_n = int(df["is_train_candidate_v1_4"].fillna(False).astype(bool).sum())
    keep_n = int(df["v1_4_decision"].eq("train_z_drop_extreme_keep").sum())
    discard_road_n = int(df["v1_4_decision"].eq("discard_roadedge_without_large_zdrop").sum())
    block = (
        "## 2026-05-21 完整记录级 episode 样本集 v1.4\n\n"
        "- 为什么做：用户复核后认为多数高度 z 风险样本确实像上下马路/路边恢复，但高度明显大幅下降的片段应先保留为极限工况样本。\n"
        f"- 本轮动作：在 v1.3 基础上计算 episode 开始后的 z 下坠幅度，保留 `z_drop >= {Z_DROP_KEEP_FLOOR_M:.1f}m` 的高度大幅下降样本，其它上下马路/路边恢复样本先抛弃；本轮不训练模型。\n"
        f"- v1.4 主训练候选：{train_n}；高度大幅下降保留样本：{keep_n}；上下马路但无明显大幅下降抛弃样本：{discard_road_n}。\n"
        f"- 用户查看版报告：`{REPORT_PATH}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        if "## 2026-05-21 完整记录级 episode 样本集 v1.4" not in raw:
            path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")

    artifact = (
        "## 2026-05-21 完整记录级 episode 样本集 v1.4\n\n"
        f"- 用户查看版报告：`{REPORT_PATH}`\n"
        f"- 全量表：`{TABLE_DIR / 'record_level_episodes_all_v1_4.csv'}`\n"
        f"- 主训练候选：`{TABLE_DIR / 'train_candidate_target_episodes_v1_4.csv'}`\n"
        f"- 高度大幅下降保留样本：`{TABLE_DIR / 'train_z_drop_extreme_keep_episodes_v1_4.csv'}`\n"
        f"- 上下马路但无明显大幅下降抛弃样本：`{TABLE_DIR / 'discard_roadedge_without_large_zdrop_episodes_v1_4.csv'}`\n"
        f"- 复核图目录：`{FIG_DIR}`\n"
    )
    raw = ARTIFACT_INDEX.read_text(encoding="utf-8") if ARTIFACT_INDEX.exists() else ""
    if "## 2026-05-21 完整记录级 episode 样本集 v1.4" not in raw:
        ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not V13_ALL.exists():
        raise FileNotFoundError(V13_ALL)
    df = pd.read_csv(V13_ALL, encoding="utf-8-sig", low_memory=False)
    cache: dict[str, dict[str, Any]] = {}
    rows = []
    for i, row in df.iterrows():
        rows.append(compute_zdrop_features(row, cache))
        if (i + 1) % 250 == 0:
            print(f"v1.4 zdrop {i + 1}/{len(df)}", flush=True)
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    decisions = df.apply(classify_v1_4, axis=1, result_type="expand")
    decisions.columns = [
        "v1_4_decision",
        "v1_4_decision_cn",
        "v1_4_decision_detail_cn",
        "is_train_candidate_v1_4",
        "is_deferred_v1_4",
        "is_control_candidate_v1_4",
        "is_discarded_v1_4",
    ]
    df = pd.concat([df, decisions], axis=1)
    df = make_review_figures(df)
    write_tables(df)
    write_report(df)
    append_notes(df)
    print(pd.read_csv(TABLE_DIR / "record_episode_v1_4_decision_summary.csv").to_string(index=False))
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
