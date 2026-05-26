from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
GOAL2_DIR = ROOT / "03_baselines" / "stage03_goal2_clean_task_audit"
MANIFEST_DIR = GOAL2_DIR / "manifests"
OUTPUT_DIR = GOAL2_DIR / "outputs" / "exclusion_recovery_audit"
REPORT_PATH = ROOT / "09_reports" / "stage03_goal2_exclusion_recovery_audit_cn.md"

MANIFEST_PATH = MANIFEST_DIR / "manifest_all_v2_task_goal2.csv"

ROAD_WORDS = [
    "offroad",
    "roadedge",
    "slope",
    "路边",
    "下马路",
    "上斜坡",
    "驶出道路",
]
HEIGHT_WORDS = [
    "height abnormal",
    "z abnormal",
    "高度异常",
]

CURRENT_TEXT_COLS = [
    "v2_0_decision",
    "v2_0_decision_cn",
    "v2_0_decision_detail_cn",
]
OLD_TEXT_COLS_GOAL2_USED = [
    "v1_2_decision",
    "v1_2_decision_cn",
    "v1_2_decision_detail_cn",
    "v1_3_decision",
    "v1_3_decision_cn",
    "v1_3_decision_detail_cn",
    "v1_4_decision",
    "v1_4_decision_cn",
    "v1_4_decision_detail_cn",
]
OLDER_CONTEXT_COLS = [
    "v1_5_decision",
    "v1_5_decision_cn",
    "v1_5_decision_detail_cn",
    "v1_8_decision",
    "v1_8_decision_cn",
    "v1_8_decision_detail_cn",
    "v1_9_decision",
    "v1_9_decision_cn",
    "v1_9_decision_detail_cn",
]


def as_text(row: pd.Series, cols: list[str]) -> str:
    parts: list[str] = []
    for col in cols:
        val = row.get(col, "")
        if pd.notna(val):
            parts.append(str(val))
    return " | ".join(parts).lower()


def has_any(text: str, words: list[str]) -> bool:
    return any(word in text for word in words)


def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "是"}


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def collect_actual_reasons(row: pd.Series) -> list[str]:
    current_text = as_text(row, CURRENT_TEXT_COLS)
    old_text = as_text(row, OLD_TEXT_COLS_GOAL2_USED)
    reasons: list[str] = []

    if str(row.get("v2_0_height_pose_issue", "")).strip() == "明显高度异常":
        reasons.append("当前v2.0标记明显高度异常")
    if has_any(current_text, ROAD_WORDS):
        reasons.append("当前v2.0文字包含路边/下马路/斜坡类关键词")
    if has_any(current_text, HEIGHT_WORDS):
        reasons.append("当前v2.0文字包含高度异常类关键词")
    if has_any(old_text, ROAD_WORDS):
        reasons.append("旧版本文字包含路边/下马路/斜坡类关键词")
    if has_any(old_text, HEIGHT_WORDS):
        reasons.append("旧版本文字包含高度异常类关键词")
    if str(row.get("episode_type", "")) == "excluded_slope_or_offroad":
        reasons.append("Goal2 episode_type 已映射为 excluded_slope_or_offroad")
    if bool_value(row.get("height_jump_suspected_v1_2")):
        reasons.append("旧v1.2高度跳变标记")
    if bool_value(row.get("z_transient_suspected_v1_3")):
        reasons.append("旧v1.3高度瞬态标记")
    if bool_value(row.get("roadedge_or_offroad_suspected_v1_3")):
        reasons.append("旧v1.3路边/路外标记")
    if bool_value(row.get("offroad_or_road_recovery_suspected_v1_2")):
        reasons.append("旧v1.2路外/回路标记")
    if finite_float(row.get("z_residual_range_v1_3")) >= 0.50:
        reasons.append("z_residual_range_v1_3 >= 0.50")
    if finite_float(row.get("z_rise_from_start_v1_4")) >= 0.50:
        reasons.append("z_rise_from_start_v1_4 >= 0.50")

    return reasons


def classify_recovery_priority(row: pd.Series) -> tuple[str, str]:
    current_text = as_text(row, CURRENT_TEXT_COLS)
    old_text = as_text(row, OLD_TEXT_COLS_GOAL2_USED)
    height_issue = str(row.get("v2_0_height_pose_issue", "")).strip()
    z_residual = finite_float(row.get("z_residual_range_v1_3"))
    z_rise = finite_float(row.get("z_rise_from_start_v1_4"))
    z_drop = finite_float(row.get("z_drop_from_start_v1_4"))
    current_road = has_any(current_text, ROAD_WORDS)
    current_height = height_issue == "明显高度异常" or has_any(current_text, HEIGHT_WORDS)
    old_only_warning = (has_any(old_text, ROAD_WORDS + HEIGHT_WORDS)) and not current_road and not current_height
    old_road_flag = bool_value(row.get("roadedge_or_offroad_suspected_v1_3")) or bool_value(
        row.get("offroad_or_road_recovery_suspected_v1_2")
    )
    old_height_flag = bool_value(row.get("height_jump_suspected_v1_2")) or bool_value(row.get("z_transient_suspected_v1_3"))
    curve = bool_value(row.get("road_coord_is_curve_v1_9"))

    small_or_normal_height = height_issue in {"高度微动正常", "高度轻微变化", "高度小幅复核"}
    low_z = z_residual < 0.10 and z_rise < 0.10
    mid_z = z_residual < 0.50 and z_rise < 0.50

    if old_only_warning and small_or_normal_height and low_z and not old_road_flag:
        return (
            "A_优先人工恢复复核",
            "当前高度表现接近正常，主要是旧版本文字触发排除；旧结论不应硬继承，建议优先看图恢复。",
        )
    if small_or_normal_height and mid_z and not current_road:
        return (
            "B_较可能可恢复",
            "当前高度不属于明显异常，z 指标未越过 0.50；如图中无真实路边/下马路，可恢复到候选训练集。",
        )
    if curve and (z_drop >= 0.50 or z_residual >= 0.50) and not current_road:
        return (
            "C1_弯道高度变化重点复核",
            "道路坐标显示弯道且高度变化较大；需要区分正常坡道/弯道高程与上斜坡或路边。",
        )
    if current_height or z_residual >= 0.50 or z_rise >= 0.50 or old_height_flag:
        return (
            "C2_高度姿态重点复核",
            "高度或姿态指标明显，需要结合道路源文件、道路坐标和复核图判断，不建议直接恢复。",
        )
    if current_road or old_road_flag:
        return (
            "D_暂不恢复_疑似路边或路外",
            "包含路边/下马路/斜坡类证据；除非人工确认仍在道路内，否则暂不进入主训练。",
        )
    return (
        "U_原因不清_需要复核",
        "未能归入明确恢复或排除原因，需要人工看图和道路坐标。",
    )


def first_existing_path(row: pd.Series) -> str:
    for col in [
        "review_panel_v2_0_path",
        "review_panel_v1_9_path",
        "review_panel_v1_8_path",
        "review_panel_v1_5_path",
        "review_panel_v1_4_path",
        "review_panel_v1_3_path",
        "review_panel_v1_2_path",
        "review_panel_path",
        "trajectory_3d_path",
    ]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            return str(val).split(";")[0]
    return ""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig", low_memory=False)
    excluded = df[df["training_role"].astype(str).eq("excluded_slope_or_offroad")].copy()

    rows: list[dict[str, Any]] = []
    for _, row in excluded.iterrows():
        current_text = as_text(row, CURRENT_TEXT_COLS)
        old_text = as_text(row, OLD_TEXT_COLS_GOAL2_USED)
        old_context_text = as_text(row, OLDER_CONTEXT_COLS)
        reasons = collect_actual_reasons(row)
        priority, suggestion = classify_recovery_priority(row)

        item = {
            "episode_uid": row.get("episode_uid", ""),
            "subject_id": row.get("subject_id", ""),
            "record_id": row.get("record_id", ""),
            "split": row.get("split", ""),
            "v2_0_decision": row.get("v2_0_decision", ""),
            "v2_0_decision_cn": row.get("v2_0_decision_cn", ""),
            "v2_0_decision_detail_cn": row.get("v2_0_decision_detail_cn", ""),
            "v2_0_height_pose_issue": row.get("v2_0_height_pose_issue", ""),
            "road_coord_is_curve_v1_9": bool_value(row.get("road_coord_is_curve_v1_9")),
            "road_coord_mapping_quality_v1_9": row.get("road_coord_mapping_quality_v1_9", ""),
            "z_residual_range_v1_3": finite_float(row.get("z_residual_range_v1_3")),
            "z_rise_from_start_v1_4": finite_float(row.get("z_rise_from_start_v1_4")),
            "z_drop_from_start_v1_4": finite_float(row.get("z_drop_from_start_v1_4")),
            "height_jump_suspected_v1_2": bool_value(row.get("height_jump_suspected_v1_2")),
            "z_transient_suspected_v1_3": bool_value(row.get("z_transient_suspected_v1_3")),
            "roadedge_or_offroad_suspected_v1_3": bool_value(row.get("roadedge_or_offroad_suspected_v1_3")),
            "offroad_or_road_recovery_suspected_v1_2": bool_value(row.get("offroad_or_road_recovery_suspected_v1_2")),
            "current_text_has_road_keyword": has_any(current_text, ROAD_WORDS),
            "current_text_has_height_keyword": has_any(current_text, HEIGHT_WORDS),
            "old_text_has_road_keyword_goal2_used": has_any(old_text, ROAD_WORDS),
            "old_text_has_height_keyword_goal2_used": has_any(old_text, HEIGHT_WORDS),
            "later_old_context_has_road_keyword": has_any(old_context_text, ROAD_WORDS),
            "later_old_context_has_height_keyword": has_any(old_context_text, HEIGHT_WORDS),
            "z_residual_ge_0_50": finite_float(row.get("z_residual_range_v1_3")) >= 0.50,
            "z_rise_ge_0_50": finite_float(row.get("z_rise_from_start_v1_4")) >= 0.50,
            "actual_exclusion_reasons": "；".join(reasons),
            "actual_reason_count": len(reasons),
            "recovery_priority": priority,
            "recovery_suggestion_cn": suggestion,
            "review_image_path": first_existing_path(row),
        }
        rows.append(item)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "goal2_exclusion_reason_breakdown.csv", index=False, encoding="utf-8-sig")

    reason_rows: list[dict[str, Any]] = []
    for reason in sorted({r for reasons in out["actual_exclusion_reasons"].astype(str) for r in reasons.split("；") if r}):
        mask = out["actual_exclusion_reasons"].astype(str).str.contains(reason, regex=False)
        reason_rows.append({"reason": reason, "count": int(mask.sum())})
    reason_summary = pd.DataFrame(reason_rows).sort_values("count", ascending=False)
    reason_summary.to_csv(OUTPUT_DIR / "goal2_exclusion_reason_summary.csv", index=False, encoding="utf-8-sig")

    priority_summary = (
        out.groupby("recovery_priority")
        .size()
        .reset_index(name="count")
        .sort_values(["recovery_priority"])
    )
    priority_summary.to_csv(OUTPUT_DIR / "goal2_recovery_priority_summary.csv", index=False, encoding="utf-8-sig")

    height_summary = (
        out.groupby(["v2_0_height_pose_issue", "recovery_priority"])
        .size()
        .reset_index(name="count")
        .sort_values(["v2_0_height_pose_issue", "recovery_priority"])
    )
    height_summary.to_csv(OUTPUT_DIR / "goal2_height_issue_by_recovery_priority.csv", index=False, encoding="utf-8-sig")

    for priority, filename in [
        ("A_优先人工恢复复核", "goal2_recovery_candidates_A_priority.csv"),
        ("B_较可能可恢复", "goal2_recovery_candidates_B_likely.csv"),
        ("C1_弯道高度变化重点复核", "goal2_recovery_candidates_C1_curve_height_review.csv"),
        ("C2_高度姿态重点复核", "goal2_recovery_candidates_C2_height_pose_review.csv"),
        ("D_暂不恢复_疑似路边或路外", "goal2_recovery_candidates_D_holdout.csv"),
    ]:
        out[out["recovery_priority"].eq(priority)].to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")

    sample = (
        out.sort_values(["recovery_priority", "actual_reason_count", "z_residual_range_v1_3"])
        .groupby("recovery_priority", group_keys=False)
        .head(30)
    )
    sample.to_csv(OUTPUT_DIR / "goal2_manual_review_sample_30_each_priority.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Goal2 被排除样本原因拆解与恢复优先级",
        "",
        "## 目的",
        "",
        "这份审计只解释 Goal2 中 1407 个被排除样本为什么被排除，并按当前理解给出恢复复核优先级。它不重新训练模型，也不把旧版本结论作为最终事实。",
        "",
        "## 总量",
        "",
        f"- 全部 Goal2 样本：`{len(df)}`",
        f"- Goal2 被标为 slope/offroad/height 排除：`{len(excluded)}`",
        "",
        "## 实际触发排除的原因",
        "",
        reason_summary.to_markdown(index=False),
        "",
        "## 建议恢复优先级",
        "",
        priority_summary.to_markdown(index=False),
        "",
        "## 高度字段与恢复优先级交叉表",
        "",
        height_summary.to_markdown(index=False),
        "",
        "## 解释",
        "",
        "- `A_优先人工恢复复核`：当前高度接近正常，主要是旧版本文字触发排除，最可能是误伤。",
        "- `B_较可能可恢复`：当前高度不属于明显异常，z 指标没有越过 0.50，建议看图后恢复。",
        "- `C1_弯道高度变化重点复核`：道路坐标显示弯道且高度变化较大，需要区分正常坡道/弯道高程与上斜坡或路边。",
        "- `C2_高度姿态重点复核`：高度或姿态指标明显，不建议直接恢复，必须结合道路源文件和图像。",
        "- `D_暂不恢复_疑似路边或路外`：有路边/下马路/斜坡类证据，除非人工确认仍在道路内，否则暂不进入主训练。",
        "",
        "## 重要提醒",
        "",
        "旧版本文字和旧标记只能作为复核提示，不能继续作为硬排除规则。下一版样本规则应优先使用当前道路坐标、道路设计源文件、当前车辆轨迹和人工复核结论。",
        "",
        "## 输出文件",
        "",
        f"- 逐样本拆解：`{OUTPUT_DIR / 'goal2_exclusion_reason_breakdown.csv'}`",
        f"- 排除原因汇总：`{OUTPUT_DIR / 'goal2_exclusion_reason_summary.csv'}`",
        f"- 恢复优先级汇总：`{OUTPUT_DIR / 'goal2_recovery_priority_summary.csv'}`",
        f"- 每档抽查样本：`{OUTPUT_DIR / 'goal2_manual_review_sample_30_each_priority.csv'}`",
    ]
    report = "\n".join(lines)
    (OUTPUT_DIR / "goal2_exclusion_recovery_report_cn.md").write_text(report, encoding="utf-8")
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"wrote {OUTPUT_DIR}")
    print(f"report {REPORT_PATH}")


if __name__ == "__main__":
    main()
