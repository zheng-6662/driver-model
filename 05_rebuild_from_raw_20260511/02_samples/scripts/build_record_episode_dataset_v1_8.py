#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build v1.8 dataset: revise curve inclusion and model-anchor timing.

User feedback:

- Curve events cannot be judged by steering alone, because normal cornering
  requires steering.
- For curve episodes, focus on vehicle roll / roll-rate.
- For downhill curves, smooth continuous height descent is expected and should
  not be treated as off-road or side-slope abnormality.
- Curve z abnormality should focus on non-smooth height pulses, abrupt steps,
  strong non-monotonic residuals, or a high-dynamics segment whose height
  profile does not match the expected downhill curve.

Additional user feedback after v1.7:

- Some episodes start far too early: the first several seconds are stable
  driving, while the actual driver/vehicle response happens much later.
- Curve samples in the "unclear/weak" bucket are mostly usable. Small height
  wiggles around 0.0x m are acceptable; only clear height-rise / abnormal z
  profiles should be excluded.

This script keeps earlier outputs intact and creates a v1.8 split:

1. non-curve main training candidates;
2. usable curve candidates, including normal/weak curve samples;
3. curve z-profile abnormal samples to exclude from the curve task;
4. corrected model anchor fields that trim long stable prefixes.

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
OUT_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_8_anchor_curve_revised"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "review_panels_v1_8"
REPORT_PATH = ROOT / "09_reports" / "stage02_record_episode_reconstruction_v1_8_user_summary_cn.md"
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
CURVE_SMALL_RISE_ALLOWED_M = 0.10
STABLE_PREFIX_TRIM_SEC = 1.50
MODEL_PRE_WINDOW_SEC = 2.00
MODEL_LABEL_WINDOW_SEC = 6.00


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


def curve_has_height_rise_problem(row: pd.Series) -> bool:
    """Small 0.0x m fluctuations are acceptable; larger upward z drift is suspicious."""
    z_rise = float(row.get("z_rise_from_start_v1_4", 0.0) or 0.0)
    return z_rise > CURVE_SMALL_RISE_ALLOWED_M


def classify_v1_8(row: pd.Series) -> tuple[str, str, str, bool, bool, bool, bool]:
    curve = is_curve_context(row)
    train_v15 = bool(row.get("is_train_candidate_v1_5", False))
    control_v15 = bool(row.get("is_control_candidate_v1_5", False))
    deferred_v15 = bool(row.get("is_deferred_v1_5", False))
    smooth_downhill = curve_is_smooth_downhill(row)
    z_abnormal = curve_has_z_profile_abnormal(row)
    roll_candidate = curve_has_roll_candidate(row)
    height_rise_problem = curve_has_height_rise_problem(row)

    if curve and (z_abnormal or height_rise_problem):
        reason = "弯道高度异常：高度明显变高、突变、非平滑，或不像正常下坡弯道"
        return "discard_curve_height_or_z_abnormal", reason, reason, False, False, False, True

    if curve and smooth_downhill and roll_candidate:
        reason = "弯道训练候选：高度连续平滑下降，允许小波动，且侧倾/横滚明显"
        return "train_curve_smooth_downhill_roll_candidate", reason, reason, True, False, False, False

    if curve and smooth_downhill:
        reason = "弯道训练候选：高度连续平滑下降，侧倾较弱或更像正常过弯"
        return "train_curve_smooth_downhill_normal_or_weak", reason, reason, True, False, False, False

    if curve and roll_candidate:
        reason = "弯道训练候选：高度没有明显变高，侧倾/横滚明显，先纳入候选"
        return "train_curve_unclear_profile_roll_candidate", reason, reason, True, False, False, False

    if curve:
        reason = "弯道训练候选：高度只存在小波动，侧倾较弱或普通，先纳入候选"
        return "train_curve_unclear_or_weak", reason, reason, True, False, False, False

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


def finite_time(value: object) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def build_model_anchor_fields(row: pd.Series) -> pd.Series:
    """Keep raw episode boundaries, but define a model anchor that trims stable prefixes."""
    start = finite_time(row.get("episode_start_s"))
    end = finite_time(row.get("episode_end_s"))
    if start is None:
        start = 0.0
    if end is None or end <= start:
        end = start + MODEL_LABEL_WINDOW_SEC

    driver = finite_time(row.get("driver_action_onset_s"))
    vehicle = finite_time(row.get("vehicle_response_onset_s"))
    risk = finite_time(row.get("condition_peak_s"))

    # Some earlier rules marked vehicle_response_onset exactly at the raw
    # episode start for a long stable segment. If the driver/risk evidence is
    # much later, do not let this inherited boundary become the model anchor.
    later_evidence = [v for v in [driver, risk] if v is not None and start - 0.2 <= v <= end + 0.5]
    vehicle_start_like = vehicle is not None and abs(vehicle - start) <= 0.25
    ignore_start_like_vehicle = bool(
        vehicle_start_like and later_evidence and (min(later_evidence) - start >= STABLE_PREFIX_TRIM_SEC)
    )

    candidates: list[tuple[str, float]] = []
    for source, value in [
        ("驾驶员动作", driver),
        ("车辆响应", vehicle),
        ("风险峰值", risk),
    ]:
        if value is None:
            continue
        if source == "车辆响应" and ignore_start_like_vehicle:
            continue
        if start - 0.2 <= value <= end + 0.5:
            candidates.append((source, value))

    if candidates:
        source, anchor = sorted(candidates, key=lambda x: x[1])[0]
    else:
        source, anchor = "原始episode开始", start

    stable_prefix = max(0.0, anchor - start)
    trimmed = stable_prefix >= STABLE_PREFIX_TRIM_SEC
    obs_start = max(0.0, anchor - MODEL_PRE_WINDOW_SEC)
    label_end = anchor + MODEL_LABEL_WINDOW_SEC
    return pd.Series(
        {
            "original_episode_start_s_v1_8": start,
            "model_anchor_s_v1_8": anchor,
            "model_anchor_source_v1_8": source,
            "ignored_start_like_vehicle_onset_v1_8": bool(ignore_start_like_vehicle),
            "stable_prefix_removed_s_v1_8": stable_prefix,
            "stable_prefix_trimmed_v1_8": bool(trimmed),
            "model_obs_start_s_v1_8": obs_start,
            "model_obs_end_s_v1_8": anchor,
            "model_label_start_s_v1_8": anchor,
            "model_label_end_s_v1_8": label_end,
        }
    )


def plot_episode_v1_8(row: pd.Series, out_path: Path, cache: dict) -> None:
    """Plot with the corrected model anchor as the red zero-time line."""
    plot_row = row.copy()
    anchor = finite_time(row.get("model_anchor_s_v1_8"))
    end = finite_time(row.get("episode_end_s"))
    if anchor is None:
        anchor = finite_time(row.get("episode_start_s")) or 0.0
    if end is None or end <= anchor:
        end = anchor + MODEL_LABEL_WINDOW_SEC
    plot_row["episode_start_s"] = anchor
    plot_row["episode_end_s"] = max(end, anchor + 2.0)
    plot_row["v1_3_decision"] = row.get("v1_8_decision", "")
    v13.plot_episode_v1_3(plot_row, out_path, cache)


def make_review_figures(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_panel_v1_8_path"] = ""
    cache = {}
    specs = [
        (
            "train_curve_smooth_downhill_roll_candidate",
            "01_弯道_平滑下坡且侧倾候选_纳入",
            ["peak_abs_roll", "peak_abs_roll_rate"],
            48,
        ),
        (
            "train_curve_smooth_downhill_normal_or_weak",
            "02_弯道_平滑下坡普通或弱侧倾_纳入",
            ["z_drop_from_start_v1_4", "z_monotonic_fraction_v1_3"],
            36,
        ),
        (
            "train_curve_unclear_profile_roll_candidate",
            "03_弯道_高度小波动但侧倾候选_纳入",
            ["peak_abs_roll", "peak_abs_roll_rate"],
            36,
        ),
        (
            "train_curve_unclear_or_weak",
            "04_弯道_高度小波动普通或弱侧倾_纳入",
            ["z_rise_from_start_v1_4", "z_residual_range_v1_3"],
            36,
        ),
        (
            "discard_curve_height_or_z_abnormal",
            "05_弯道_高度变高或异常_排除",
            ["z_residual_rate_peak_v1_3", "z_residual_range_v1_3", "peak_abs_ay"],
            48,
        ),
        (
            "train_noncurve_target_extreme",
            "06_非弯道主训练候选_修正锚点后抽查",
            ["stable_prefix_removed_s_v1_8", "vehicle_score_peak", "condition_score_peak"],
            36,
        ),
    ]
    for decision, folder, sort_cols, max_n in specs:
        subset = df[df["v1_8_decision"].eq(decision)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(sort_cols, ascending=False)
        for idx, row in subset.head(max_n).iterrows():
            out_path = FIG_DIR / folder / f"{idx:04d}_{row['episode_uid']}.png"
            if not out_path.exists():
                plot_episode_v1_8(row, out_path, cache)
            if out_path.exists():
                df.at[idx, "review_panel_v1_8_path"] = str(out_path)
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
    df.to_csv(TABLE_DIR / "record_level_episodes_all_v1_8.csv", index=False, encoding="utf-8-sig")
    df[df["is_train_candidate_v1_8"]].to_csv(
        TABLE_DIR / "train_candidate_all_episodes_v1_8.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_8_decision"].eq("train_noncurve_target_extreme")].to_csv(
        TABLE_DIR / "train_candidate_noncurve_episodes_v1_8.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_8_decision"].str.startswith("train_curve_", na=False)].to_csv(
        TABLE_DIR / "train_candidate_curve_episodes_v1_8.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_8_decision"].eq("train_curve_smooth_downhill_roll_candidate")].to_csv(
        TABLE_DIR / "curve_smooth_downhill_roll_candidate_episodes_v1_8.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_8_decision"].eq("train_curve_smooth_downhill_normal_or_weak")].to_csv(
        TABLE_DIR / "curve_smooth_downhill_normal_or_weak_episodes_v1_8.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_8_decision"].eq("discard_curve_height_or_z_abnormal")].to_csv(
        TABLE_DIR / "discard_curve_height_or_z_abnormal_episodes_v1_8.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v1_8_decision"].isin(["train_curve_unclear_profile_roll_candidate", "train_curve_unclear_or_weak"])].to_csv(
        TABLE_DIR / "curve_small_wiggle_candidate_episodes_v1_8.csv", index=False, encoding="utf-8-sig"
    )
    df[df.apply(is_curve_context, axis=1)].to_csv(
        TABLE_DIR / "all_curve_context_episodes_v1_8.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_deferred_v1_8"]].to_csv(
        TABLE_DIR / "deferred_or_review_episodes_v1_8.csv", index=False, encoding="utf-8-sig"
    )
    summary = (
        df.groupby("v1_8_decision", dropna=False)
        .agg(v1_8_decision_cn=("v1_8_decision_cn", "first"), count=("v1_8_decision", "size"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    summary.to_csv(TABLE_DIR / "record_episode_v1_8_decision_summary.csv", index=False, encoding="utf-8-sig")


def write_report(df: pd.DataFrame) -> None:
    summary = pd.read_csv(TABLE_DIR / "record_episode_v1_8_decision_summary.csv")
    train_n = int(df["is_train_candidate_v1_8"].fillna(False).astype(bool).sum())
    trim_mask = df["stable_prefix_trimmed_v1_8"].fillna(False).astype(bool)
    trim_n = int(trim_mask.sum())
    trim_mean = float(df.loc[trim_mask, "stable_prefix_removed_s_v1_8"].mean()) if trim_n else 0.0
    curve_all_n = int(df.apply(is_curve_context, axis=1).sum())
    curve_train_n = int(df["v1_8_decision"].str.startswith("train_curve_", na=False).sum())
    curve_roll_n = int(df["v1_8_decision"].eq("train_curve_smooth_downhill_roll_candidate").sum())
    curve_smooth_weak_n = int(df["v1_8_decision"].eq("train_curve_smooth_downhill_normal_or_weak").sum())
    curve_bad_n = int(df["v1_8_decision"].eq("discard_curve_height_or_z_abnormal").sum())
    curve_small_wiggle_n = int(
        df["v1_8_decision"].isin(["train_curve_unclear_profile_roll_candidate", "train_curve_unclear_or_weak"]).sum()
    )
    curve_roll_view = df[df["v1_8_decision"].eq("train_curve_smooth_downhill_roll_candidate")][
        [
            "episode_uid",
            "subject",
            "road_module_names",
            "episode_duration_s",
            "model_anchor_s_v1_8",
            "model_anchor_source_v1_8",
            "stable_prefix_removed_s_v1_8",
            "peak_abs_roll",
            "peak_abs_roll_rate",
            "z_drop_from_start_v1_4",
            "z_rise_from_start_v1_4",
            "z_residual_range_v1_3",
            "z_residual_rate_peak_v1_3",
            "z_monotonic_fraction_v1_3",
            "review_panel_v1_8_path",
        ]
    ].sort_values(["peak_abs_roll", "peak_abs_roll_rate"], ascending=False)
    text = f"""# 完整记录级 episode 样本集 v1.8：弯道侧倾单独筛选

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 这次为什么改

用户进一步指出两件事：

1. 有些 episode 划得太早，前面很多秒都是平稳驾驶，真正驾驶员动作/车辆响应发生在后面。
2. v1.7 中“弯道形态不够明确或侧倾弱”的样本，大部分可以先纳入训练；高度 z 只有 0.0x m 的小波动可以接受，少数明显高度变高的才排除。

因此 v1.8 做两项修正：一是新增模型用锚点，把过早的平稳前缀裁掉；二是弯道样本更偏向保留，只有高度明显变高或 z 形态异常才排除。

## v1.8 规则

- 训练候选：保留非弯道 v1.5 主训练候选，同时纳入高度正常/小波动的弯道样本。
- 模型用锚点：优先使用驾驶员动作开始、车辆响应开始、风险峰值三者中最早的可用时间；如果它比原始 episode 开始晚超过 {STABLE_PREFIX_TRIM_SEC:.1f}s，就标记为“裁掉平稳前缀”。
- 平滑下坡弯道：`z_drop >= {CURVE_SMOOTH_DOWNHILL_MIN_DROP_M:.1f}m`，`z_rise <= {CURVE_SMOOTH_DOWNHILL_MAX_RISE_M:.1f}m`，`z_monotonic_fraction >= {CURVE_SMOOTH_DOWNHILL_MIN_MONOTONIC:.2f}`，同时残差速度和残差范围不过大。
- 平滑下坡弯道侧倾候选：满足平滑下坡，并且 `peak_abs_roll >= {ROLL_ANGLE_CANDIDATE_RAD:.2f}rad` 或 `peak_abs_roll_rate >= {ROLL_RATE_CANDIDATE_RADPS:.2f}rad/s`。
- 高度小波动：`z_rise <= {CURVE_SMALL_RISE_ALLOWED_M:.2f}m` 先接受；超过该值或 z 形态异常才排除。

## 数量变化

- v1.8 全部训练候选：{train_n}
- 其中弯道训练候选：{curve_train_n}
- 全部弯道上下文样本：{curve_all_n}
- 平滑下坡弯道侧倾候选：{curve_roll_n}
- 平滑下坡弯道普通/弱侧倾：{curve_smooth_weak_n}
- 弯道高度小波动纳入候选：{curve_small_wiggle_n}
- 弯道高度变高或形态异常，排除：{curve_bad_n}
- 模型用锚点相对原始 episode 开始裁掉平稳前缀的样本：{trim_n}，平均裁掉 {trim_mean:.2f}s

## v1.8 分类表

{md_table(summary)}

## 平滑下坡弯道侧倾候选样本

{md_table(curve_roll_view)}

## 输出位置

- v1.8 全量表：`{TABLE_DIR / "record_level_episodes_all_v1_8.csv"}`
- 全部训练候选：`{TABLE_DIR / "train_candidate_all_episodes_v1_8.csv"}`
- 非弯道主训练候选：`{TABLE_DIR / "train_candidate_noncurve_episodes_v1_8.csv"}`
- 弯道训练候选：`{TABLE_DIR / "train_candidate_curve_episodes_v1_8.csv"}`
- 平滑下坡弯道侧倾候选：`{TABLE_DIR / "curve_smooth_downhill_roll_candidate_episodes_v1_8.csv"}`
- 平滑下坡弯道普通/弱侧倾：`{TABLE_DIR / "curve_smooth_downhill_normal_or_weak_episodes_v1_8.csv"}`
- 弯道高度小波动候选：`{TABLE_DIR / "curve_small_wiggle_candidate_episodes_v1_8.csv"}`
- 弯道高度变高或形态异常排除：`{TABLE_DIR / "discard_curve_height_or_z_abnormal_episodes_v1_8.csv"}`
- 复核图目录：`{FIG_DIR}`

## 当前建议

后续训练时不要再用原始 `episode_start_s` 作为唯一锚点，而应优先使用 `model_anchor_s_v1_8`。这样可以避免“前面很多秒平稳驾驶，标签却从后面才开始变化”的错位问题。

本轮没有训练模型。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
    (OUT_ROOT / "record_episode_dataset_v1_8_summary_cn.md").write_text(text, encoding="utf-8")


def append_notes(df: pd.DataFrame) -> None:
    train_n = int(df["is_train_candidate_v1_8"].fillna(False).astype(bool).sum())
    curve_train_n = int(df["v1_8_decision"].str.startswith("train_curve_", na=False).sum())
    curve_bad_n = int(df["v1_8_decision"].eq("discard_curve_height_or_z_abnormal").sum())
    trim_n = int(df["stable_prefix_trimmed_v1_8"].fillna(False).astype(bool).sum())
    block = (
        "## 2026-05-22 完整记录级 episode 样本集 v1.8\n\n"
        "- 为什么做：用户指出部分 episode 起点过早，前面长时间平稳驾驶；弯道小幅高度波动不应排除，大部分弯道待复核样本可以先纳入训练候选。\n"
        "- 本轮动作：新增模型用锚点，裁掉过早平稳前缀；放宽弯道小波动样本，仅排除高度明显变高或 z 形态异常样本；本轮不训练模型。\n"
        f"- v1.8 全部训练候选：{train_n}；弯道训练候选：{curve_train_n}；弯道高度异常排除：{curve_bad_n}；锚点裁掉平稳前缀样本：{trim_n}。\n"
        f"- 用户查看版报告：`{REPORT_PATH}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        if "## 2026-05-22 完整记录级 episode 样本集 v1.8" not in raw:
            path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")
    artifact = (
        "## 2026-05-22 完整记录级 episode 样本集 v1.8\n\n"
        f"- 用户查看版报告：`{REPORT_PATH}`\n"
        f"- 全量表：`{TABLE_DIR / 'record_level_episodes_all_v1_8.csv'}`\n"
        f"- 全部训练候选：`{TABLE_DIR / 'train_candidate_all_episodes_v1_8.csv'}`\n"
        f"- 非弯道主训练候选：`{TABLE_DIR / 'train_candidate_noncurve_episodes_v1_8.csv'}`\n"
        f"- 弯道训练候选：`{TABLE_DIR / 'train_candidate_curve_episodes_v1_8.csv'}`\n"
        f"- 平滑下坡弯道侧倾候选：`{TABLE_DIR / 'curve_smooth_downhill_roll_candidate_episodes_v1_8.csv'}`\n"
        f"- 弯道高度异常排除：`{TABLE_DIR / 'discard_curve_height_or_z_abnormal_episodes_v1_8.csv'}`\n"
        f"- 复核图目录：`{FIG_DIR}`\n"
    )
    raw = ARTIFACT_INDEX.read_text(encoding="utf-8") if ARTIFACT_INDEX.exists() else ""
    if "## 2026-05-22 完整记录级 episode 样本集 v1.8" not in raw:
        ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not V15_ALL.exists():
        raise FileNotFoundError(V15_ALL)
    df = pd.read_csv(V15_ALL, encoding="utf-8-sig", low_memory=False)
    decisions = df.apply(classify_v1_8, axis=1, result_type="expand")
    decisions.columns = [
        "v1_8_decision",
        "v1_8_decision_cn",
        "v1_8_decision_detail_cn",
        "is_train_candidate_v1_8",
        "is_deferred_v1_8",
        "is_control_candidate_v1_8",
        "is_discarded_v1_8",
    ]
    df = pd.concat([df, decisions], axis=1)
    anchor_fields = df.apply(build_model_anchor_fields, axis=1)
    df = pd.concat([df, anchor_fields], axis=1)
    df = make_review_figures(df)
    write_tables(df)
    write_report(df)
    append_notes(df)
    print(pd.read_csv(TABLE_DIR / "record_episode_v1_8_decision_summary.csv").to_string(index=False))
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()


