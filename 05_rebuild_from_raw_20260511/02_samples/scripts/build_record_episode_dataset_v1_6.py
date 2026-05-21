#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build v1.6 dataset: split curve events from the main training pool.

User feedback:

- Curve events cannot be judged by steering alone, because normal cornering
  requires steering.
- For curve episodes, focus on vehicle roll / roll-rate.
- If a fast curve entry makes the car climb onto the side slope, producing
  abnormal z rise/drop/residual, that sample should not be treated as the
  target curve-roll event.

This script keeps v1.5 intact and creates a v1.6 split:

1. non-curve main training candidates;
2. clean curve-roll candidates;
3. curve z/slope abnormal samples to discard from curve modeling;
4. remaining curve samples for review/control.

It does not train any model.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

import build_record_episode_dataset_v1_3 as v13


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
V15_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_5_curve_separated"
V15_ALL = V15_ROOT / "tables" / "record_level_episodes_all_v1_5.csv"
OUT_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_6_curve_roll_split"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "review_panels_v1_6"
REPORT_PATH = ROOT / "09_reports" / "stage02_record_episode_reconstruction_v1_6_user_summary_cn.md"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-21.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"


ROLL_ANGLE_CANDIDATE_RAD = 0.10
ROLL_RATE_CANDIDATE_RADPS = 0.80
CURVE_Z_DROP_ABNORMAL_M = 2.00
CURVE_Z_RISE_ABNORMAL_M = 0.80
CURVE_Z_RESIDUAL_ABNORMAL_M = 1.50


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, REPORT_PATH.parent, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def is_curve_context(row: pd.Series) -> bool:
    text = f"{row.get('road_module_names', '')}|{row.get('road_design_categories', '')}".lower()
    return bool(row.get("is_curve_context", False)) or ("curve" in text) or ("弯道" in text)


def curve_has_roll_candidate(row: pd.Series) -> bool:
    return (
        float(row.get("peak_abs_roll", 0.0) or 0.0) >= ROLL_ANGLE_CANDIDATE_RAD
        or float(row.get("peak_abs_roll_rate", 0.0) or 0.0) >= ROLL_RATE_CANDIDATE_RADPS
    )


def curve_has_z_slope_abnormal(row: pd.Series) -> bool:
    return (
        float(row.get("z_drop_from_start_v1_4", 0.0) or 0.0) >= CURVE_Z_DROP_ABNORMAL_M
        or float(row.get("z_rise_from_start_v1_4", 0.0) or 0.0) >= CURVE_Z_RISE_ABNORMAL_M
        or float(row.get("z_residual_range_v1_3", 0.0) or 0.0) >= CURVE_Z_RESIDUAL_ABNORMAL_M
    )


def classify_v1_6(row: pd.Series) -> tuple[str, str, str, bool, bool, bool, bool]:
    curve = is_curve_context(row)
    train_v15 = bool(row.get("is_train_candidate_v1_5", False))
    control_v15 = bool(row.get("is_control_candidate_v1_5", False))
    deferred_v15 = bool(row.get("is_deferred_v1_5", False))
    z_abnormal = curve_has_z_slope_abnormal(row)
    roll_candidate = curve_has_roll_candidate(row)

    if curve and z_abnormal:
        reason = "弯道高度/坡度异常：疑似开上斜坡或道路边缘，不进入主训练和弯道侧倾候选"
        return "discard_curve_slope_or_z_abnormal", reason, reason, False, False, False, True

    if curve and roll_candidate:
        reason = "弯道侧倾候选：侧倾/横滚明显，且未触发高度异常，单独进入弯道候选池"
        return "review_curve_roll_candidate_clean", reason, reason, False, True, False, False

    if curve:
        reason = "弯道普通或弱侧倾样本：不进入主训练，保留为弯道复核/对照"
        return "review_curve_normal_or_weak_roll", reason, reason, False, True, False, False

    if train_v15:
        reason = "非弯道主训练候选：继承 v1.5，作为当前主训练集"
        return "train_noncurve_target_extreme", reason, reason, True, False, False, False

    if control_v15:
        reason = "非弯道对照样本：继承 v1.5"
        return "control_noncurve", reason, reason, False, False, True, False

    if deferred_v15:
        reason = "非弯道仍需复核或拆分：继承 v1.5"
        return "defer_noncurve_prior_review", reason, reason, False, True, False, False

    reason = "非弯道已舍弃或不适合作为当前候选：继承 v1.5"
    return "discard_noncurve_prior_review", reason, reason, False, False, False, True


def make_review_figures(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_panel_v1_6_path"] = ""
    cache = {}
    specs = [
        (
            "discard_curve_slope_or_z_abnormal",
            "01_弯道高度异常_疑似斜坡或路边_排除",
            ["z_drop_from_start_v1_4", "z_rise_from_start_v1_4", "z_residual_range_v1_3"],
            48,
        ),
        (
            "review_curve_roll_candidate_clean",
            "02_弯道侧倾候选_高度正常",
            ["peak_abs_roll", "peak_abs_roll_rate"],
            48,
        ),
        (
            "review_curve_normal_or_weak_roll",
            "03_弯道普通或弱侧倾_复核对照",
            ["peak_abs_roll", "peak_abs_roll_rate"],
            36,
        ),
        (
            "train_noncurve_target_extreme",
            "04_非弯道主训练候选抽查",
            ["vehicle_score_peak", "condition_score_peak"],
            36,
        ),
    ]
    for decision, folder, sort_cols, max_n in specs:
        subset = df[df["v1_6_decision"].eq(decision)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(sort_cols, ascending=False)
        for idx, row in subset.head(max_n).iterrows():
            out_path = FIG_DIR / folder / f"{idx:04d}_{row['episode_uid']}.png"
            if not out_path.exists():
                v13.plot_episode_v1_3(row, out_path, cache)
            if out_path.exists():
                df.at[idx, "review_panel_v1_6_path"] = str(out_path)
    return df


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "暂无。"
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for v in row.tolist():
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_tables(df: pd.DataFrame) -> None:
    df.to_csv(TABLE_DIR / "record_level_episodes_all_v1_6.csv", index=False, encoding="utf-8-sig")
    df[df["is_train_candidate_v1_6"]].to_csv(
        TABLE_DIR / "train_candidate_noncurve_episodes_v1_6.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_6_decision"].eq("review_curve_roll_candidate_clean")].to_csv(
        TABLE_DIR / "curve_roll_candidate_clean_episodes_v1_6.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_6_decision"].eq("discard_curve_slope_or_z_abnormal")].to_csv(
        TABLE_DIR / "discard_curve_slope_or_z_abnormal_episodes_v1_6.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_6_decision"].eq("review_curve_normal_or_weak_roll")].to_csv(
        TABLE_DIR / "curve_normal_or_weak_roll_review_episodes_v1_6.csv", index=False, encoding="utf-8-sig"
    )
    df[df.apply(is_curve_context, axis=1)].to_csv(
        TABLE_DIR / "all_curve_context_episodes_v1_6.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_deferred_v1_6"]].to_csv(
        TABLE_DIR / "deferred_or_review_episodes_v1_6.csv", index=False, encoding="utf-8-sig"
    )
    summary = (
        df.groupby("v1_6_decision", dropna=False)
        .agg(v1_6_decision_cn=("v1_6_decision_cn", "first"), count=("v1_6_decision", "size"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    summary.to_csv(TABLE_DIR / "record_episode_v1_6_decision_summary.csv", index=False, encoding="utf-8-sig")


def write_report(df: pd.DataFrame) -> None:
    summary = pd.read_csv(TABLE_DIR / "record_episode_v1_6_decision_summary.csv")
    train_n = int(df["is_train_candidate_v1_6"].fillna(False).astype(bool).sum())
    curve_all_n = int(df.apply(is_curve_context, axis=1).sum())
    curve_roll_n = int(df["v1_6_decision"].eq("review_curve_roll_candidate_clean").sum())
    curve_bad_n = int(df["v1_6_decision"].eq("discard_curve_slope_or_z_abnormal").sum())
    curve_weak_n = int(df["v1_6_decision"].eq("review_curve_normal_or_weak_roll").sum())
    curve_roll_view = df[df["v1_6_decision"].eq("review_curve_roll_candidate_clean")][
        [
            "episode_uid",
            "subject",
            "road_module_names",
            "episode_duration_s",
            "peak_abs_roll",
            "peak_abs_roll_rate",
            "z_drop_from_start_v1_4",
            "z_rise_from_start_v1_4",
            "z_residual_range_v1_3",
            "review_panel_v1_6_path",
        ]
    ].sort_values(["peak_abs_roll", "peak_abs_roll_rate"], ascending=False)
    text = f"""# 完整记录级 episode 样本集 v1.6：弯道侧倾单独筛选

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 这次为什么改

用户指出：弯道不能只看方向盘，因为正常过弯本来就需要打方向。弯道应该重点看车辆侧倾/横滚；但如果驾驶员入弯过快开到两侧斜坡上，导致高度突然变大、突然变小，或者和正常下坡趋势不一致，这类样本也不是目标弯道侧倾样本，应从弯道候选中排除。

因此 v1.6 将弯道从主训练集中完全拆出来，并在弯道内部按侧倾和高度异常重新分层。

## v1.6 规则

- 主训练集：只保留非弯道的 v1.5 主训练候选。
- 弯道高度异常排除：`z_drop >= {CURVE_Z_DROP_ABNORMAL_M:.1f}m`，或 `z_rise >= {CURVE_Z_RISE_ABNORMAL_M:.1f}m`，或 `z_residual_range >= {CURVE_Z_RESIDUAL_ABNORMAL_M:.1f}m`。
- 弯道侧倾候选：弯道上下文中，`peak_abs_roll >= {ROLL_ANGLE_CANDIDATE_RAD:.2f}rad` 或 `peak_abs_roll_rate >= {ROLL_RATE_CANDIDATE_RADPS:.2f}rad/s`，且没有触发高度异常。
- 其它弯道样本：作为普通弯道/弱侧倾复核或对照，不进入主训练。

## 数量变化

- v1.6 非弯道主训练候选：{train_n}
- 全部弯道上下文样本：{curve_all_n}
- 弯道侧倾候选且高度正常：{curve_roll_n}
- 弯道高度/坡度异常，疑似斜坡或路边，排除：{curve_bad_n}
- 弯道普通或弱侧倾复核/对照：{curve_weak_n}

## v1.6 分类表

{md_table(summary)}

## 弯道侧倾候选样本

{md_table(curve_roll_view)}

## 输出位置

- v1.6 全量表：`{TABLE_DIR / "record_level_episodes_all_v1_6.csv"}`
- 非弯道主训练候选：`{TABLE_DIR / "train_candidate_noncurve_episodes_v1_6.csv"}`
- 弯道侧倾候选：`{TABLE_DIR / "curve_roll_candidate_clean_episodes_v1_6.csv"}`
- 弯道高度异常排除：`{TABLE_DIR / "discard_curve_slope_or_z_abnormal_episodes_v1_6.csv"}`
- 弯道普通或弱侧倾复核：`{TABLE_DIR / "curve_normal_or_weak_roll_review_episodes_v1_6.csv"}`
- 复核图目录：`{FIG_DIR}`

## 当前建议

后续不要再把弯道和其它极限工况混在一个训练池里。可以先用非弯道主训练候选跑车辆-only；弯道路线则单独使用“弯道侧倾候选且高度正常”的样本做专门分析。

本轮没有训练模型。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
    (OUT_ROOT / "record_episode_dataset_v1_6_summary_cn.md").write_text(text, encoding="utf-8")


def append_notes(df: pd.DataFrame) -> None:
    train_n = int(df["is_train_candidate_v1_6"].fillna(False).astype(bool).sum())
    curve_roll_n = int(df["v1_6_decision"].eq("review_curve_roll_candidate_clean").sum())
    curve_bad_n = int(df["v1_6_decision"].eq("discard_curve_slope_or_z_abnormal").sum())
    block = (
        "## 2026-05-21 完整记录级 episode 样本集 v1.6\n\n"
        "- 为什么做：用户指出弯道不能只看方向盘，需重点看侧倾；开上弯道两侧斜坡造成高度异常的样本也不要作为目标弯道样本。\n"
        "- 本轮动作：将弯道从主训练候选中完全拆出，按弯道侧倾候选、弯道高度异常排除、弯道普通/弱侧倾复核分层；本轮不训练模型。\n"
        f"- v1.6 非弯道主训练候选：{train_n}；弯道侧倾候选：{curve_roll_n}；弯道高度异常排除：{curve_bad_n}。\n"
        f"- 用户查看版报告：`{REPORT_PATH}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        if "## 2026-05-21 完整记录级 episode 样本集 v1.6" not in raw:
            path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")
    artifact = (
        "## 2026-05-21 完整记录级 episode 样本集 v1.6\n\n"
        f"- 用户查看版报告：`{REPORT_PATH}`\n"
        f"- 全量表：`{TABLE_DIR / 'record_level_episodes_all_v1_6.csv'}`\n"
        f"- 非弯道主训练候选：`{TABLE_DIR / 'train_candidate_noncurve_episodes_v1_6.csv'}`\n"
        f"- 弯道侧倾候选：`{TABLE_DIR / 'curve_roll_candidate_clean_episodes_v1_6.csv'}`\n"
        f"- 弯道高度异常排除：`{TABLE_DIR / 'discard_curve_slope_or_z_abnormal_episodes_v1_6.csv'}`\n"
        f"- 复核图目录：`{FIG_DIR}`\n"
    )
    raw = ARTIFACT_INDEX.read_text(encoding="utf-8") if ARTIFACT_INDEX.exists() else ""
    if "## 2026-05-21 完整记录级 episode 样本集 v1.6" not in raw:
        ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not V15_ALL.exists():
        raise FileNotFoundError(V15_ALL)
    df = pd.read_csv(V15_ALL, encoding="utf-8-sig", low_memory=False)
    decisions = df.apply(classify_v1_6, axis=1, result_type="expand")
    decisions.columns = [
        "v1_6_decision",
        "v1_6_decision_cn",
        "v1_6_decision_detail_cn",
        "is_train_candidate_v1_6",
        "is_deferred_v1_6",
        "is_control_candidate_v1_6",
        "is_discarded_v1_6",
    ]
    df = pd.concat([df, decisions], axis=1)
    df = make_review_figures(df)
    write_tables(df)
    write_report(df)
    append_notes(df)
    print(pd.read_csv(TABLE_DIR / "record_episode_v1_6_decision_summary.csv").to_string(index=False))
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
