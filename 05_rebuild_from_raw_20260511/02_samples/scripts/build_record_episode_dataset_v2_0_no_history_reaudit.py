#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build v2.0 record-level episode dataset by re-auditing all episodes.

This version responds to a user correction:

- Do not inherit old "candidate / non-candidate" decisions.
- Re-audit all 1766 episodes with the current road-coordinate and vehicle
  dynamics evidence.
- Keep old decisions only as audit columns, not as classification inputs.

The script reads the v1.9 table because v1.9 already contains coordinate-based
road mapping and computed vehicle-dynamics metrics. It does not train a model.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_record_episode_dataset_v1_9 as v19


PROJECT_ROOT = Path(r"F:/data_set_process/data_process")
ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
V19_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_9_coord_curve_revised"
V19_ALL = V19_ROOT / "tables" / "record_level_episodes_all_v1_9.csv"

OUT_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v2_0_no_history_reaudit"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "review_panels_v2_0"
REPORT_PATH = ROOT / "09_reports" / "stage02_record_episode_reconstruction_v2_0_user_summary_cn.md"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-22.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"


# Dynamic thresholds. These are intentionally visible and conservative.
ROLL_ANGLE_CANDIDATE_RAD = 0.08
ROLL_RATE_CANDIDATE_RADPS = 0.60
AY_DYNAMIC_CANDIDATE = 5.0
YAW_RATE_CANDIDATE = 0.35
STEER_RATE_FAST = 10.0
STEER_RANGE_MEANINGFUL = 1.0
BRAKE_MEANINGFUL = 0.25
SPEED_CHANGE_MEANINGFUL_KMH = 20.0

# Height is auxiliary only. Small z motion must not exclude an episode.
Z_MICRO_OK = 0.02
Z_LIGHT_REVIEW = 0.05
Z_REVIEW = 0.10
Z_STRONG_ABNORMAL = 0.50
Z_RISE_ABNORMAL = 0.15
PITCH_REVIEW = 0.10


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, REPORT_PATH.parent, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def dynamic_flags(row: pd.Series) -> dict[str, bool]:
    return {
        "横滚角明显": finite_float(row.get("peak_abs_roll")) >= ROLL_ANGLE_CANDIDATE_RAD,
        "横滚角速度明显": finite_float(row.get("peak_abs_roll_rate")) >= ROLL_RATE_CANDIDATE_RADPS,
        "横向加速度明显": finite_float(row.get("peak_abs_ay")) >= AY_DYNAMIC_CANDIDATE,
        "横摆角速度明显": finite_float(row.get("peak_abs_yaw_rate")) >= YAW_RATE_CANDIDATE,
    }


def vehicle_dynamic_count(row: pd.Series) -> int:
    return int(sum(dynamic_flags(row).values()))


def has_vehicle_dynamic(row: pd.Series) -> bool:
    flags = dynamic_flags(row)
    count = int(sum(flags.values()))
    if count >= 2:
        return True
    return bool(
        finite_float(row.get("peak_abs_roll")) >= 0.12
        or finite_float(row.get("peak_abs_ay")) >= 7.0
        or finite_float(row.get("peak_abs_roll_rate")) >= 1.0
    )


def has_weak_vehicle_dynamic(row: pd.Series) -> bool:
    flags = dynamic_flags(row)
    return bool(
        sum(flags.values()) >= 1
        or finite_float(row.get("peak_abs_roll")) >= 0.05
        or finite_float(row.get("peak_abs_ay")) >= 3.0
        or finite_float(row.get("peak_abs_yaw_rate")) >= 0.15
    )


def has_fast_steer(row: pd.Series) -> bool:
    return bool(
        finite_float(row.get("steer_rate_peak")) >= STEER_RATE_FAST
        or finite_float(row.get("steer_angle_range")) >= STEER_RANGE_MEANINGFUL
    )


def has_speed_brake_response(row: pd.Series) -> bool:
    return bool(
        finite_float(row.get("brake_range")) >= BRAKE_MEANINGFUL
        or finite_float(row.get("speed_range_kmh")) >= SPEED_CHANGE_MEANINGFUL_KMH
    )


def height_pose_issue(row: pd.Series) -> str:
    z_resid = finite_float(row.get("z_residual_range_v1_3"), 0.0)
    z_rise = finite_float(row.get("z_rise_from_start_v1_4"), 0.0)
    pitch_resid = finite_float(row.get("pitch_residual_range_v1_3"), 0.0)
    if z_resid >= Z_STRONG_ABNORMAL or z_rise >= Z_RISE_ABNORMAL:
        return "明显高度异常"
    if z_resid >= Z_REVIEW or pitch_resid >= PITCH_REVIEW:
        return "轻度高度/姿态复核"
    if z_resid >= Z_LIGHT_REVIEW:
        return "高度小幅复核"
    if z_resid <= Z_MICRO_OK:
        return "高度微动正常"
    return "高度轻微变化"


def classify_v2_0(row: pd.Series) -> tuple[str, str, str, bool, bool, bool, bool]:
    """Return decision, Chinese label, detail, train, review, control, discard.

    Old v1.8/v1.9 decisions are intentionally not used for branching.
    """
    mapping_quality = str(row.get("road_coord_mapping_quality_v1_9", ""))
    coord_curve = bool_value(row.get("road_coord_is_curve_v1_9"))
    dynamic = has_vehicle_dynamic(row)
    weak_dynamic = has_weak_vehicle_dynamic(row)
    fast_steer = has_fast_steer(row)
    speed_brake = has_speed_brake_response(row)
    height_issue = height_pose_issue(row)
    dyn_count = vehicle_dynamic_count(row)

    if mapping_quality == "very_low_review":
        detail = "道路坐标最近邻距离过大，不能可靠判断道路模块；全量重审中先进入道路映射复核。"
        return "review_mapping_uncertain", "道路坐标映射不确定", detail, False, True, False, False

    if coord_curve:
        if height_issue in {"明显高度异常", "轻度高度/姿态复核"}:
            detail = (
                f"道路坐标确认在弯道，但 {height_issue}；当前不直接训练，优先人工确认是否上斜坡、下路边或异常过弯。"
            )
            return "review_curve_height_pose_abnormal", "弯道高度/姿态异常复核", detail, False, True, False, False
        if dynamic:
            detail = f"道路坐标确认在弯道，车辆横滚/横摆/横向动态明显，动态信号数={dyn_count}。"
            return "train_curve_roll_dynamic", "弯道侧倾/动态训练候选", detail, True, False, False, False
        detail = "道路坐标确认在弯道，车辆动态较弱或更像正常过弯；保留为弯道普通/弱侧倾训练候选。"
        return "train_curve_normal_or_weak", "弯道普通/弱侧倾训练候选", detail, True, False, False, False

    if height_issue == "明显高度异常" and not dynamic:
        detail = "道路坐标显示非弯道，但高度变化明显且横向/横滚动态不强；优先复核是否路外恢复、上下马路或坐标异常。"
        return "review_noncurve_height_abnormal_weak_dynamic", "非弯道高度异常但动态弱", detail, False, True, False, False

    if dynamic:
        detail = f"道路坐标显示非弯道，按当前车辆动态重新纳入训练候选；动态信号数={dyn_count}，{height_issue}。"
        return "train_noncurve_vehicle_dynamic", "非弯道车辆动态训练候选", detail, True, False, False, False

    if weak_dynamic and (fast_steer or speed_brake):
        detail = "道路坐标显示非弯道，车辆动态为弱到中等，但伴随方向盘、速度或制动响应；作为次级训练候选。"
        return "train_noncurve_secondary_dynamic", "非弯道次级动态训练候选", detail, True, False, False, False

    if fast_steer and not weak_dynamic:
        detail = "方向盘动作明显，但车辆横滚/横摆/横向动态弱；按用户意见不直接作为目标极限训练样本，进入复核。"
        return "review_fast_steer_weak_vehicle", "快打方向但车辆响应弱", detail, False, True, False, False

    if speed_brake and not weak_dynamic:
        detail = "速度或制动变化明显，但横向车辆动态弱；可能是纵向事件或控制样本，暂复核。"
        return "review_speed_brake_only", "速度/制动为主但横向动态弱", detail, False, True, False, False

    detail = "道路坐标显示非弯道，方向盘和车辆横向动态都较弱；作为正常/弱响应对照，不直接进入极限轨迹训练。"
    return "control_noncurve_weak_or_normal", "非弯道弱响应/正常对照", detail, False, False, True, False


def add_v2_decisions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    decisions = df.apply(classify_v2_0, axis=1, result_type="expand")
    decisions.columns = [
        "v2_0_decision",
        "v2_0_decision_cn",
        "v2_0_decision_detail_cn",
        "is_train_candidate_v2_0",
        "is_review_candidate_v2_0",
        "is_control_candidate_v2_0",
        "is_discarded_v2_0",
    ]
    df = pd.concat([df, decisions], axis=1)
    df["v2_0_vehicle_dynamic_count"] = df.apply(vehicle_dynamic_count, axis=1)
    df["v2_0_height_pose_issue"] = df.apply(height_pose_issue, axis=1)
    df["v2_0_reaudited_without_history"] = True
    df["v2_0_old_v1_9_train"] = df.get("is_train_candidate_v1_9", False)
    df["v2_0_recovered_from_v1_9_nontrain"] = (
        df["is_train_candidate_v2_0"].fillna(False)
        & ~df["v2_0_old_v1_9_train"].fillna(False).astype(bool)
    )
    return df


def write_tables(df: pd.DataFrame) -> None:
    df.to_csv(TABLE_DIR / "record_level_episodes_all_v2_0.csv", index=False, encoding="utf-8-sig")
    df[df["is_train_candidate_v2_0"]].to_csv(
        TABLE_DIR / "train_candidate_all_episodes_v2_0.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_train_candidate_v2_0"] & df["road_coord_is_curve_v1_9"].fillna(False).astype(bool)].to_csv(
        TABLE_DIR / "train_candidate_curve_coord_episodes_v2_0.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_train_candidate_v2_0"] & ~df["road_coord_is_curve_v1_9"].fillna(False).astype(bool)].to_csv(
        TABLE_DIR / "train_candidate_noncurve_episodes_v2_0.csv", index=False, encoding="utf-8-sig"
    )
    df[df["v2_0_recovered_from_v1_9_nontrain"]].to_csv(
        TABLE_DIR / "recovered_from_v1_9_nontrain_episodes_v2_0.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_review_candidate_v2_0"]].to_csv(
        TABLE_DIR / "manual_review_episodes_v2_0.csv", index=False, encoding="utf-8-sig"
    )
    df[df["is_control_candidate_v2_0"]].to_csv(
        TABLE_DIR / "control_or_weak_episodes_v2_0.csv", index=False, encoding="utf-8-sig"
    )

    summary = (
        df.groupby(["v2_0_decision", "v2_0_decision_cn"], dropna=False)
        .agg(
            count=("episode_uid", "size"),
            train_count=("is_train_candidate_v2_0", "sum"),
            review_count=("is_review_candidate_v2_0", "sum"),
            control_count=("is_control_candidate_v2_0", "sum"),
            recovered_from_v19_nontrain=("v2_0_recovered_from_v1_9_nontrain", "sum"),
        )
        .reset_index()
        .sort_values(["train_count", "count"], ascending=False)
    )
    summary.to_csv(TABLE_DIR / "record_episode_v2_0_decision_summary.csv", index=False, encoding="utf-8-sig")

    transition = (
        df.groupby(["v1_9_decision", "v2_0_decision"], dropna=False)
        .agg(
            count=("episode_uid", "size"),
            train_count=("is_train_candidate_v2_0", "sum"),
            recovered_from_v19_nontrain=("v2_0_recovered_from_v1_9_nontrain", "sum"),
        )
        .reset_index()
        .sort_values(["recovered_from_v19_nontrain", "count"], ascending=False)
    )
    transition.to_csv(TABLE_DIR / "v1_9_to_v2_0_transition_audit.csv", index=False, encoding="utf-8-sig")

    module_summary = (
        df.groupby(["road_coord_dominant_module_v1_9", "v2_0_decision"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["road_coord_dominant_module_v1_9", "count"], ascending=[True, False])
    )
    module_summary.to_csv(TABLE_DIR / "road_coord_module_summary_v2_0.csv", index=False, encoding="utf-8-sig")


def plot_episode_v2_0(row: pd.Series, out_path: Path, cache: dict[str, Any]) -> None:
    plot_row = row.copy()
    plot_row["v1_9_decision"] = row.get("v2_0_decision", "")
    plot_row["v1_8_decision"] = row.get("v2_0_decision", "")
    v19.plot_episode_v1_9(plot_row, out_path, cache)


def make_review_figures(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_panel_v2_0_path"] = ""
    cache: dict[str, Any] = {}
    specs = [
        (
            "00_全量重审新增训练候选_重点看",
            df["v2_0_recovered_from_v1_9_nontrain"],
            ["peak_abs_roll", "peak_abs_ay", "steer_angle_range"],
            20,
        ),
        (
            "01_非弯道车辆动态训练候选",
            df["v2_0_decision"].eq("train_noncurve_vehicle_dynamic"),
            ["peak_abs_roll", "peak_abs_ay", "peak_abs_yaw_rate"],
            16,
        ),
        (
            "02_非弯道次级动态训练候选",
            df["v2_0_decision"].eq("train_noncurve_secondary_dynamic"),
            ["peak_abs_roll", "peak_abs_ay", "steer_rate_peak"],
            12,
        ),
        (
            "03_弯道训练候选",
            df["v2_0_decision"].isin(["train_curve_roll_dynamic", "train_curve_normal_or_weak"]),
            ["peak_abs_roll", "peak_abs_ay", "steer_angle_range"],
            16,
        ),
        (
            "04_快打方向但车辆响应弱_复核",
            df["v2_0_decision"].eq("review_fast_steer_weak_vehicle"),
            ["steer_rate_peak", "steer_angle_range", "peak_abs_ay"],
            12,
        ),
        (
            "05_道路坐标映射不确定_复核",
            df["v2_0_decision"].eq("review_mapping_uncertain"),
            ["road_coord_nearest_dist_median_v1_9", "peak_abs_ay", "peak_abs_roll"],
            10,
        ),
        (
            "06_非弯道弱响应或正常对照",
            df["v2_0_decision"].eq("control_noncurve_weak_or_normal"),
            ["peak_abs_ay", "peak_abs_roll", "steer_angle_range"],
            10,
        ),
    ]
    for folder, mask, sort_cols, limit in specs:
        out_dir = FIG_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        sub = df.loc[mask.fillna(False)].copy()
        if sub.empty:
            continue
        for col in sort_cols:
            if col not in sub.columns:
                sub[col] = 0.0
            sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)
        sub["_rank"] = sub[sort_cols].abs().sum(axis=1)
        sub = sub.sort_values("_rank", ascending=False).head(limit)
        for idx, row in sub.iterrows():
            uid = str(row.get("episode_uid", f"idx_{idx}")).replace(":", "_").replace("\\", "_").replace("/", "_")
            out_path = out_dir / f"{int(idx):04d}_{uid}.png"
            try:
                plot_episode_v2_0(row, out_path, cache)
                df.at[idx, "review_panel_v2_0_path"] = str(out_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] failed to plot {uid}: {exc}")
    return df


def write_report(df: pd.DataFrame) -> None:
    summary = pd.read_csv(TABLE_DIR / "record_episode_v2_0_decision_summary.csv")
    transition = pd.read_csv(TABLE_DIR / "v1_9_to_v2_0_transition_audit.csv")
    module_summary = pd.read_csv(TABLE_DIR / "road_coord_module_summary_v2_0.csv")
    total = len(df)
    train_n = int(df["is_train_candidate_v2_0"].sum())
    review_n = int(df["is_review_candidate_v2_0"].sum())
    control_n = int(df["is_control_candidate_v2_0"].sum())
    curve_train = int((df["is_train_candidate_v2_0"] & df["road_coord_is_curve_v1_9"].fillna(False).astype(bool)).sum())
    noncurve_train = train_n - curve_train
    recovered = int(df["v2_0_recovered_from_v1_9_nontrain"].sum())
    prior_discard_recovered = int(
        (
            df["v2_0_recovered_from_v1_9_nontrain"]
            & df["v1_9_decision"].astype(str).eq("discard_noncurve_prior_review")
        ).sum()
    )
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    text = f"""# v2.0 全量无历史继承 episode 重审总结

生成时间：{now}

## 这版为什么要做

用户指出：不能再沿用“历史上不是候选就不进入训练”的逻辑。此前 v1.9 虽然修正了弯道判定，但仍有一类 `discard_noncurve_prior_review`，本质上是历史非候选的继承结果。

因此 v2.0 做了一个更干净的重审：

- 1766 个 episode 全部重新审查；
- 历史 v1.8/v1.9 候选身份不参与新分类；
- 新分类只看当前可解释证据：道路坐标、车辆横滚/横摆/横向加速度、方向盘、速度/制动、高度/姿态异常和道路坐标映射质量；
- 历史标签只保留为审计对照字段。

## 总体数量

- 全部 episode：{total}
- v2.0 训练候选：{train_n}
- 其中非弯道训练候选：{noncurve_train}
- 其中弯道训练候选：{curve_train}
- 待复核：{review_n}
- 正常/弱响应对照：{control_n}
- 从 v1.9 非训练集合中重新纳入训练：{recovered}
- 其中从 `discard_noncurve_prior_review` 中重新纳入训练：{prior_discard_recovered}

## v2.0 决策分布

{summary.to_markdown(index=False)}

## v1.9 到 v2.0 的变化审计

下面这张表用于检查：哪些历史非候选在 v2.0 中被重新纳入或转入复核。

{transition.head(30).to_markdown(index=False)}

## 道路模块分布

{module_summary.head(40).to_markdown(index=False)}

## 当前解释

1. v2.0 不再使用“历史候选/历史非候选”作为分类依据。
2. 原先历史非候选并没有被直接舍弃，而是重新按车辆动态、道路坐标和姿态指标判断。
3. 快打方向但车辆动态弱的样本不直接纳入极限训练，先进入复核。
4. 车辆动态明显但驾驶员操作弱的样本可以进入训练，因为这符合“保守驾驶员/弱操作也可能处于极限工况”的研究目标。
5. 高度 z 仍然只作为异常辅助证据；直路/非弯道的小幅高度微动不作为排除依据。

## 输出文件

- 全量表：`{TABLE_DIR / "record_level_episodes_all_v2_0.csv"}`
- 全部训练候选：`{TABLE_DIR / "train_candidate_all_episodes_v2_0.csv"}`
- 非弯道训练候选：`{TABLE_DIR / "train_candidate_noncurve_episodes_v2_0.csv"}`
- 弯道训练候选：`{TABLE_DIR / "train_candidate_curve_coord_episodes_v2_0.csv"}`
- 从 v1.9 非训练集合中重新纳入的样本：`{TABLE_DIR / "recovered_from_v1_9_nontrain_episodes_v2_0.csv"}`
- 待复核样本：`{TABLE_DIR / "manual_review_episodes_v2_0.csv"}`
- 正常/弱响应对照：`{TABLE_DIR / "control_or_weak_episodes_v2_0.csv"}`
- 复核图目录：`{FIG_DIR}`

## 下一步建议

先看 `00_全量重审新增训练候选_重点看` 这个复核图文件夹。如果这部分大多数确实合理，说明 v2.0 纠正了历史非候选带来的偏置；如果这里仍混入很多无效片段，就需要继续细化车辆动态阈值，而不是回到历史候选继承逻辑。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def append_notes(df: pd.DataFrame) -> None:
    train_n = int(df["is_train_candidate_v2_0"].sum())
    recovered = int(df["v2_0_recovered_from_v1_9_nontrain"].sum())
    review_n = int(df["is_review_candidate_v2_0"].sum())
    curve_train = int((df["is_train_candidate_v2_0"] & df["road_coord_is_curve_v1_9"].fillna(False).astype(bool)).sum())
    noncurve_train = train_n - curve_train
    block = f"""

## 2026-05-22 完整记录级 episode 样本集 v2.0 全量无历史继承重审

- 为什么做：用户指出不能再用“历史上不是候选”作为排除依据；此前未判为候选的 episode 也必须按当前道路坐标和车辆动态重新审查。
- 本轮动作：基于 v1.9 的道路坐标和车辆动态特征，对全部 `1766` 个 episode 重新分类；历史 v1.8/v1.9 标签只作为审计对照，不参与 v2.0 决策。
- v2.0 训练候选：`{train_n}`，其中非弯道 `{noncurve_train}`，弯道 `{curve_train}`；从 v1.9 非训练集合中重新纳入训练：`{recovered}`；待复核：`{review_n}`。
- 用户查看版报告：`{REPORT_PATH}`。
- 输出目录：`{OUT_ROOT}`。
"""
    for path in [DAILY_LOG, NOTES_DIR / "PROJECT_STATUS_CN.md"]:
        with path.open("a", encoding="utf-8") as f:
            f.write(block)
    with ARTIFACT_INDEX.open("a", encoding="utf-8") as f:
        f.write(
            f"""

## 2026-05-22 完整记录级 episode 样本集 v2.0 全量无历史继承重审

- 用户查看版报告：`{REPORT_PATH}`
- 全量表：`{TABLE_DIR / "record_level_episodes_all_v2_0.csv"}`
- 全部训练候选：`{TABLE_DIR / "train_candidate_all_episodes_v2_0.csv"}`
- 非弯道训练候选：`{TABLE_DIR / "train_candidate_noncurve_episodes_v2_0.csv"}`
- 弯道训练候选：`{TABLE_DIR / "train_candidate_curve_coord_episodes_v2_0.csv"}`
- 重新纳入训练样本：`{TABLE_DIR / "recovered_from_v1_9_nontrain_episodes_v2_0.csv"}`
- 待复核样本：`{TABLE_DIR / "manual_review_episodes_v2_0.csv"}`
- 对照样本：`{TABLE_DIR / "control_or_weak_episodes_v2_0.csv"}`
- 复核图目录：`{FIG_DIR}`
"""
        )


def main() -> None:
    ensure_dirs()
    if not V19_ALL.exists():
        raise FileNotFoundError(V19_ALL)
    df = pd.read_csv(V19_ALL, low_memory=False)
    df = add_v2_decisions(df)
    df["review_panel_v2_0_path"] = ""
    write_tables(df)
    write_report(df)
    df = make_review_figures(df)
    write_tables(df)
    write_report(df)
    append_notes(df)
    print(f"[DONE] v2.0 no-history reaudit: {OUT_ROOT}")
    print(f"[DONE] report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
