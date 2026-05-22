#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build v1.7 dataset: revise curve z-profile rules.

User feedback:

- Curve events cannot be judged by steering alone, because normal cornering
  requires steering.
- For curve episodes, focus on vehicle roll / roll-rate.
- For downhill curves, smooth continuous height descent is expected and should
  not be treated as off-road or side-slope abnormality.
- Curve z abnormality should focus on non-smooth height pulses, abrupt steps,
  strong non-monotonic residuals, or a high-dynamics segment whose height
  profile does not match the expected downhill curve.

This script keeps v1.5/v1.7 intact and creates a v1.7 split:

1. non-curve main training candidates;
2. smooth downhill curve-roll candidates;
3. smooth downhill weak/normal curve controls;
4. curve z-profile abnormal samples to exclude from the curve task;
5. remaining curve samples for review.

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
OUT_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_7_curve_zprofile_revised"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "review_panels_v1_7"
REPORT_PATH = ROOT / "09_reports" / "stage02_record_episode_reconstruction_v1_7_user_summary_cn.md"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-22.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"


ROLL_ANGLE_CANDIDATE_RAD = 0.10
ROLL_RATE_CANDIDATE_RADPS = 0.80
CURVE_SMOOTH_DOWNHILL_MIN_DROP_M = 1.00
CURVE_SMOOTH_DOWNHILL_MAX_RISE_M = 0.30
CURVE_SMOOTH_DOWNHILL_MIN_MONOTONIC = 0.82
CURVE_SMOOTH_DOWNHILL_MAX_RESIDUAL_RATE = 2.50
CURVE_SMOOTH_DOWNHILL_MAX_RESIDUAL_RANGE = 3.00
CURVE_Z_RISE_ABNORMAL_M = 0.80
CURVE_Z_RESIDUAL_RATE_ABNORMAL = 3.00
CURVE_Z_RESIDUAL_RANGE_ABNORMAL = 3.50
CURVE_Z_MONOTONIC_ABNORMAL = 0.70
CURVE_FLAT_OR_WRONG_PROFILE_MAX_DROP_M = 0.50
CURVE_HIGH_DYNAMIC_AY = 5.00


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


def curve_is_smooth_downhill(row: pd.Series) -> bool:
    """Expected downhill curve: continuous descent with only small residual wiggles."""
    z_drop = float(row.get("z_drop_from_start_v1_4", 0.0) or 0.0)
    z_rise = float(row.get("z_rise_from_start_v1_4", 0.0) or 0.0)
    z_resid_rate = float(row.get("z_residual_rate_peak_v1_3", 0.0) or 0.0)
    z_resid_range = float(row.get("z_residual_range_v1_3", 0.0) or 0.0)
    monotonic = float(row.get("z_monotonic_fraction_v1_3", 0.0) or 0.0)
    return (
        z_drop >= CURVE_SMOOTH_DOWNHILL_MIN_DROP_M
        and z_rise <= CURVE_SMOOTH_DOWNHILL_MAX_RISE_M
        and monotonic >= CURVE_SMOOTH_DOWNHILL_MIN_MONOTONIC
        and z_resid_rate <= CURVE_SMOOTH_DOWNHILL_MAX_RESIDUAL_RATE
        and z_resid_range <= CURVE_SMOOTH_DOWNHILL_MAX_RESIDUAL_RANGE
    )


def curve_has_z_profile_abnormal(row: pd.Series) -> bool:
    """Abnormal curve z profile: side-slope/off-road-like height behavior."""
    z_drop = float(row.get("z_drop_from_start_v1_4", 0.0) or 0.0)
    z_rise = float(row.get("z_rise_from_start_v1_4", 0.0) or 0.0)
    z_resid_rate = float(row.get("z_residual_rate_peak_v1_3", 0.0) or 0.0)
    z_resid_range = float(row.get("z_residual_range_v1_3", 0.0) or 0.0)
    monotonic = float(row.get("z_monotonic_fraction_v1_3", 0.0) or 0.0)
    ay = float(row.get("peak_abs_ay", 0.0) or 0.0)
    roll_candidate = curve_has_roll_candidate(row)
    high_dynamic_wrong_profile = (
        roll_candidate
        and ay >= CURVE_HIGH_DYNAMIC_AY
        and z_drop <= CURVE_FLAT_OR_WRONG_PROFILE_MAX_DROP_M
    )
    return (
        z_rise >= CURVE_Z_RISE_ABNORMAL_M
        or z_resid_rate >= CURVE_Z_RESIDUAL_RATE_ABNORMAL
        or z_resid_range >= CURVE_Z_RESIDUAL_RANGE_ABNORMAL
        or monotonic <= CURVE_Z_MONOTONIC_ABNORMAL
        or high_dynamic_wrong_profile
    )


def classify_v1_7(row: pd.Series) -> tuple[str, str, str, bool, bool, bool, bool]:
    curve = is_curve_context(row)
    train_v15 = bool(row.get("is_train_candidate_v1_5", False))
    control_v15 = bool(row.get("is_control_candidate_v1_5", False))
    deferred_v15 = bool(row.get("is_deferred_v1_5", False))
    smooth_downhill = curve_is_smooth_downhill(row)
    z_abnormal = curve_has_z_profile_abnormal(row)
    roll_candidate = curve_has_roll_candidate(row)

    if curve and smooth_downhill and roll_candidate:
        reason = "弯道有效样本：高度连续平滑下降，允许小波动，且侧倾/横滚明显"
        return "review_curve_smooth_downhill_roll_candidate", reason, reason, False, True, False, False

    if curve and smooth_downhill:
        reason = "弯道有效对照：高度连续平滑下降，但侧倾/横滚不强"
        return "review_curve_smooth_downhill_normal_or_weak", reason, reason, False, True, False, False

    if curve and z_abnormal:
        reason = "弯道高度轨迹异常：高度突变、非平滑、非连续下降或强动态但高度轨迹不像正常下坡"
        return "discard_curve_z_profile_abnormal", reason, reason, False, False, False, True

    if curve and roll_candidate:
        reason = "弯道侧倾候选但高度形态不够明确：需要人工复核"
        return "review_curve_unclear_profile_roll_candidate", reason, reason, False, True, False, False

    if curve:
        reason = "弯道高度形态或侧倾较弱：保留为复核/对照"
        return "review_curve_unclear_or_weak", reason, reason, False, True, False, False

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
    df["review_panel_v1_7_path"] = ""
    cache = {}
    specs = [
        (
            "review_curve_smooth_downhill_roll_candidate",
            "01_平滑下坡弯道_侧倾候选",
            ["peak_abs_roll", "peak_abs_roll_rate"],
            48,
        ),
        (
            "review_curve_smooth_downhill_normal_or_weak",
            "02_平滑下坡弯道_普通或弱侧倾",
            ["z_drop_from_start_v1_4", "z_monotonic_fraction_v1_3"],
            36,
        ),
        (
            "discard_curve_z_profile_abnormal",
            "03_弯道高度轨迹异常_排除",
            ["z_residual_rate_peak_v1_3", "z_residual_range_v1_3", "peak_abs_ay"],
            48,
        ),
        (
            "review_curve_unclear_profile_roll_candidate",
            "04_弯道侧倾明显但高度形态不明_复核",
            ["peak_abs_roll", "peak_abs_roll_rate"],
            36,
        ),
        (
            "review_curve_unclear_or_weak",
            "05_弯道高度形态不明或弱侧倾_复核",
            ["peak_abs_roll", "peak_abs_roll_rate"],
            36,
        ),
        (
            "train_noncurve_target_extreme",
            "06_非弯道主训练候选抽查",
            ["vehicle_score_peak", "condition_score_peak"],
            36,
        ),
    ]
    for decision, folder, sort_cols, max_n in specs:
        subset = df[df["v1_7_decision"].eq(decision)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(sort_cols, ascending=False)
        for idx, row in subset.head(max_n).iterrows():
            out_path = FIG_DIR / folder / f"{idx:04d}_{row['episode_uid']}.png"
            if not out_path.exists():
                v13.plot_episode_v1_3(row, out_path, cache)
            if out_path.exists():
                df.at[idx, "review_panel_v1_7_path"] = str(out_path)
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
    df.to_csv(TABLE_DIR / "record_level_episodes_all_v1_7.csv", index=False, encoding="utf-8-sig")
    df[df["is_train_candidate_v1_7"]].to_csv(
        TABLE_DIR / "train_candidate_noncurve_episodes_v1_7.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_7_decision"].eq("review_curve_smooth_downhill_roll_candidate")].to_csv(
        TABLE_DIR / "curve_smooth_downhill_roll_candidate_episodes_v1_7.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_7_decision"].eq("review_curve_smooth_downhill_normal_or_weak")].to_csv(
        TABLE_DIR / "curve_smooth_downhill_normal_or_weak_episodes_v1_7.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_7_decision"].eq("discard_curve_z_profile_abnormal")].to_csv(
        TABLE_DIR / "discard_curve_z_profile_abnormal_episodes_v1_7.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_7_decision"].isin(["review_curve_unclear_profile_roll_candidate", "review_curve_unclear_or_weak"])].to_csv(
        TABLE_DIR / "curve_unclear_profile_review_episodes_v1_7.csv", index=False, encoding="utf-8-sig"
    )
    df[df.apply(is_curve_context, axis=1)].to_csv(
        TABLE_DIR / "all_curve_context_episodes_v1_7.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_deferred_v1_7"]].to_csv(
        TABLE_DIR / "deferred_or_review_episodes_v1_7.csv", index=False, encoding="utf-8-sig"
    )
    summary = (
        df.groupby("v1_7_decision", dropna=False)
        .agg(v1_7_decision_cn=("v1_7_decision_cn", "first"), count=("v1_7_decision", "size"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    summary.to_csv(TABLE_DIR / "record_episode_v1_7_decision_summary.csv", index=False, encoding="utf-8-sig")


def write_report(df: pd.DataFrame) -> None:
    summary = pd.read_csv(TABLE_DIR / "record_episode_v1_7_decision_summary.csv")
    train_n = int(df["is_train_candidate_v1_7"].fillna(False).astype(bool).sum())
    curve_all_n = int(df.apply(is_curve_context, axis=1).sum())
    curve_roll_n = int(df["v1_7_decision"].eq("review_curve_smooth_downhill_roll_candidate").sum())
    curve_smooth_weak_n = int(df["v1_7_decision"].eq("review_curve_smooth_downhill_normal_or_weak").sum())
    curve_bad_n = int(df["v1_7_decision"].eq("discard_curve_z_profile_abnormal").sum())
    curve_unclear_n = int(df["v1_7_decision"].isin(["review_curve_unclear_profile_roll_candidate", "review_curve_unclear_or_weak"]).sum())
    curve_roll_view = df[df["v1_7_decision"].eq("review_curve_smooth_downhill_roll_candidate")][
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
            "z_residual_rate_peak_v1_3",
            "z_monotonic_fraction_v1_3",
            "review_panel_v1_7_path",
        ]
    ].sort_values(["peak_abs_roll", "peak_abs_roll_rate"], ascending=False)
    text = f"""# 完整记录级 episode 样本集 v1.7：弯道侧倾单独筛选

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 这次为什么改

用户进一步指出：我之前把“高度持续下降/残差较大”过度当成异常了。对于下坡弯道，正常样本应表现为高度连续、平滑下降，允许有少量波动；真正异常的是高度突变、台阶式变化、反常抬升、非平滑大残差，或强动态片段的高度轨迹不像正常下坡。

因此 v1.7 将弯道从主训练集中完全拆出来，并在弯道内部按侧倾和高度异常重新分层。

## v1.7 规则

- 主训练集：只保留非弯道的 v1.5 主训练候选。
- 平滑下坡弯道：`z_drop >= {CURVE_SMOOTH_DOWNHILL_MIN_DROP_M:.1f}m`，`z_rise <= {CURVE_SMOOTH_DOWNHILL_MAX_RISE_M:.1f}m`，`z_monotonic_fraction >= {CURVE_SMOOTH_DOWNHILL_MIN_MONOTONIC:.2f}`，同时残差速度和残差范围不过大。
- 平滑下坡弯道侧倾候选：满足平滑下坡，并且 `peak_abs_roll >= {ROLL_ANGLE_CANDIDATE_RAD:.2f}rad` 或 `peak_abs_roll_rate >= {ROLL_RATE_CANDIDATE_RADPS:.2f}rad/s`。
- 弯道高度轨迹异常：高度明显反常抬升、残差速度过大、残差范围过大、单调性过低，或强动态但高度轨迹不像正常下坡。
- 其它弯道样本：保留为高度形态不明或弱侧倾复核，不进入当前主训练。

## 数量变化

- v1.7 非弯道主训练候选：{train_n}
- 全部弯道上下文样本：{curve_all_n}
- 平滑下坡弯道侧倾候选：{curve_roll_n}
- 平滑下坡弯道普通/弱侧倾：{curve_smooth_weak_n}
- 弯道高度轨迹异常，排除：{curve_bad_n}
- 弯道高度形态不明，复核：{curve_unclear_n}

## v1.7 分类表

{md_table(summary)}

## 平滑下坡弯道侧倾候选样本

{md_table(curve_roll_view)}

## 输出位置

- v1.7 全量表：`{TABLE_DIR / "record_level_episodes_all_v1_7.csv"}`
- 非弯道主训练候选：`{TABLE_DIR / "train_candidate_noncurve_episodes_v1_7.csv"}`
- 平滑下坡弯道侧倾候选：`{TABLE_DIR / "curve_smooth_downhill_roll_candidate_episodes_v1_7.csv"}`
- 平滑下坡弯道普通/弱侧倾：`{TABLE_DIR / "curve_smooth_downhill_normal_or_weak_episodes_v1_7.csv"}`
- 弯道高度轨迹异常排除：`{TABLE_DIR / "discard_curve_z_profile_abnormal_episodes_v1_7.csv"}`
- 弯道高度形态不明复核：`{TABLE_DIR / "curve_unclear_profile_review_episodes_v1_7.csv"}`
- 复核图目录：`{FIG_DIR}`

## 当前建议

后续不要再把弯道和其它极限工况混在一个训练池里。可以先用非弯道主训练候选跑车辆-only；弯道路线则单独使用“弯道侧倾候选且高度正常”的样本做专门分析。

本轮没有训练模型。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
    (OUT_ROOT / "record_episode_dataset_v1_7_summary_cn.md").write_text(text, encoding="utf-8")


def append_notes(df: pd.DataFrame) -> None:
    train_n = int(df["is_train_candidate_v1_7"].fillna(False).astype(bool).sum())
    curve_roll_n = int(df["v1_7_decision"].eq("review_curve_smooth_downhill_roll_candidate").sum())
    curve_bad_n = int(df["v1_7_decision"].eq("discard_curve_z_profile_abnormal").sum())
    block = (
        "## 2026-05-22 完整记录级 episode 样本集 v1.7\n\n"
        "- 为什么做：用户指出此前把平滑下坡弯道误判为高度异常；正常弯道应是高度连续下降且允许小波动，异常应看突变、反常波动或不符合正常下坡的高度轨迹。\n"
        "- 本轮动作：修正弯道高度规则，将平滑下坡弯道从异常类中救回；本轮不训练模型。\n"
        f"- v1.7 非弯道主训练候选：{train_n}；平滑下坡弯道侧倾候选：{curve_roll_n}；弯道高度轨迹异常排除：{curve_bad_n}。\n"
        f"- 用户查看版报告：`{REPORT_PATH}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        if "## 2026-05-22 完整记录级 episode 样本集 v1.7" not in raw:
            path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")
    artifact = (
        "## 2026-05-22 完整记录级 episode 样本集 v1.7\n\n"
        f"- 用户查看版报告：`{REPORT_PATH}`\n"
        f"- 全量表：`{TABLE_DIR / 'record_level_episodes_all_v1_7.csv'}`\n"
        f"- 非弯道主训练候选：`{TABLE_DIR / 'train_candidate_noncurve_episodes_v1_7.csv'}`\n"
        f"- 平滑下坡弯道侧倾候选：`{TABLE_DIR / 'curve_smooth_downhill_roll_candidate_episodes_v1_7.csv'}`\n"
        f"- 弯道高度轨迹异常排除：`{TABLE_DIR / 'discard_curve_z_profile_abnormal_episodes_v1_7.csv'}`\n"
        f"- 复核图目录：`{FIG_DIR}`\n"
    )
    raw = ARTIFACT_INDEX.read_text(encoding="utf-8") if ARTIFACT_INDEX.exists() else ""
    if "## 2026-05-22 完整记录级 episode 样本集 v1.7" not in raw:
        ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not V15_ALL.exists():
        raise FileNotFoundError(V15_ALL)
    df = pd.read_csv(V15_ALL, encoding="utf-8-sig", low_memory=False)
    decisions = df.apply(classify_v1_7, axis=1, result_type="expand")
    decisions.columns = [
        "v1_7_decision",
        "v1_7_decision_cn",
        "v1_7_decision_detail_cn",
        "is_train_candidate_v1_7",
        "is_deferred_v1_7",
        "is_control_candidate_v1_7",
        "is_discarded_v1_7",
    ]
    df = pd.concat([df, decisions], axis=1)
    df = make_review_figures(df)
    write_tables(df)
    write_report(df)
    append_notes(df)
    print(pd.read_csv(TABLE_DIR / "record_episode_v1_7_decision_summary.csv").to_string(index=False))
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()

