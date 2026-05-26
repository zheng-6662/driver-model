"""Build v2.1 candidate tables after fixing SILAB lateral-offset and height rules.

This script does not train models. It re-reads the Goal2 manifest and creates a
new audit layer that treats lateral-offset jumps as a SILAB reference-frame
warning, and treats small height changes as review/keep evidence rather than
hard off-road exclusion.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path("F:/data_set_process/data_process")
REBUILD_ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
GOAL2_ROOT = REBUILD_ROOT / "03_baselines" / "stage03_goal2_clean_task_audit"
MANIFEST_PATH = GOAL2_ROOT / "manifests" / "manifest_all_v2_task_goal2.csv"
EXCLUSION_BREAKDOWN_PATH = (
    GOAL2_ROOT / "outputs" / "exclusion_recovery_audit" / "goal2_exclusion_reason_breakdown.csv"
)
OUT_ROOT = (
    REBUILD_ROOT
    / "02_samples"
    / "record_level_episode_reconstruction_v2_1_reference_height_recovery"
)
TABLE_DIR = OUT_ROOT / "tables"
REPORT_PATH = REBUILD_ROOT / "09_reports" / "stage02_record_episode_reconstruction_v2_1_user_summary_cn.md"


CURRENT_ROAD_KEYWORDS = [
    "offroad",
    "roadedge",
    "路边",
    "下马路",
    "上斜坡",
    "驶出道路",
    "路外",
]
CURRENT_HEIGHT_KEYWORDS = [
    "height abnormal",
    "z abnormal",
    "高度异常",
    "高度/姿态异常",
]
OLD_ONLY_HINTS = [
    "旧版本文字包含",
    "旧v1.3",
    "旧v1.2",
    "Goal2 episode_type 已映射",
]


def as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def as_bool(value: object) -> bool:
    text = as_text(value).strip().lower()
    return text in {"true", "1", "yes", "y", "是"}


def as_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result


def has_any(text: str, keywords: Iterable[str]) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def is_curve_context(row: pd.Series) -> bool:
    if as_bool(row.get("road_coord_is_curve_v1_9")):
        return True
    module_text = " ".join(
        [
            as_text(row.get("road_coord_modules_seen_v1_9")),
            as_text(row.get("road_coord_anchor_module_v1_9")),
            as_text(row.get("road_coord_dominant_module_v1_9")),
        ]
    ).lower()
    if any(name in module_text for name in ["curve1", "curve2", "curve3"]):
        return True
    context_text = " ".join(
        [
            as_text(row.get("curve_type")),
            as_text(row.get("episode_type")),
            as_text(row.get("v2_0_decision")),
            as_text(row.get("v2_0_decision_cn")),
        ]
    ).lower()
    if "noncurve" in context_text or "非弯道" in context_text:
        return False
    return "弯道" in context_text or "curve_" in context_text


def is_design_grade_curve(row: pd.Series) -> bool:
    module_text = " ".join(
        [
            as_text(row.get("road_coord_modules_seen_v1_9")),
            as_text(row.get("road_coord_anchor_module_v1_9")),
            as_text(row.get("road_coord_dominant_module_v1_9")),
        ]
    ).lower()
    return "curve1" in module_text or "curve2" in module_text


def current_decision_text(row: pd.Series) -> str:
    return " ".join(
        [
            as_text(row.get("v2_0_decision")),
            as_text(row.get("v2_0_decision_cn")),
            as_text(row.get("v2_0_decision_detail_cn")),
        ]
    )


def assign_v21(row: pd.Series) -> pd.Series:
    z_residual = as_float(row.get("z_residual_range_v1_3"))
    z_rise = as_float(row.get("z_rise_from_start_v1_4"))
    z_drop = as_float(row.get("z_drop_from_start_v1_4"))
    z_range = as_float(row.get("z_range_v1_2"))
    current_text = current_decision_text(row)
    actual_reasons = as_text(row.get("actual_exclusion_reasons"))
    recovery_priority = as_text(row.get("recovery_priority"))

    curve_context = is_curve_context(row)
    design_grade_curve = is_design_grade_curve(row)
    current_road_warning = has_any(current_text, CURRENT_ROAD_KEYWORDS)
    current_height_warning = has_any(current_text, CURRENT_HEIGHT_KEYWORDS)
    old_only_warning = has_any(actual_reasons, OLD_ONLY_HINTS)

    lat_jump_count = as_float(row.get("lat_offset_large_jump_count_v1_3"))
    lat_jump_peak = as_float(row.get("lat_offset_adjacent_jump_peak_v1_3"))
    lateral_reference_switch_warning = (
        as_bool(row.get("lat_offset_jump_suspected_v1_3"))
        or (not math.isnan(lat_jump_count) and lat_jump_count > 0)
        or (not math.isnan(lat_jump_peak) and lat_jump_peak >= 1.0)
    )

    if math.isnan(z_residual):
        height_level = "未知高度残差"
    elif z_residual < 0.20:
        height_level = "小幅高度变化_不作为排除依据"
    elif z_residual < 0.50:
        height_level = "中等高度变化_需要复核"
    elif z_residual < 1.00:
        height_level = "较大高度变化_重点复核"
    else:
        height_level = "大幅高度变化_强复核"

    design_height_note = ""
    if design_grade_curve and not math.isnan(z_range) and z_range >= 1.0:
        design_height_note = "curve1/curve2设计存在米级坡度_不能用原始z范围直接排除"

    data_bad = as_bool(row.get("goal2_data_bad")) or "data_bad" in as_text(row.get("episode_type"))
    severe_height = (not math.isnan(z_residual) and z_residual >= 1.0) or (
        not math.isnan(z_rise) and z_rise >= 1.0
    )
    clear_current_offroad = current_road_warning and severe_height and not design_grade_curve
    clear_current_height = current_height_warning and severe_height and not design_grade_curve
    hard_exclude = data_bad or clear_current_offroad or clear_current_height

    is_train_v20 = as_bool(row.get("is_train_candidate_v2_0"))
    is_review_v20 = as_bool(row.get("is_review_candidate_v2_0"))
    is_control_v20 = as_bool(row.get("is_control_candidate_v2_0"))
    is_discarded_v20 = as_bool(row.get("is_discarded_v2_0"))

    if hard_exclude:
        role = "hard_excluded_v2_1"
        include_training_pool = False
        action = "暂不进入当前训练"
    elif is_train_v20:
        role = "main_train_candidate_v2_1"
        include_training_pool = True
        action = "恢复或保留为主训练候选"
    elif is_review_v20:
        role = "review_recovered_candidate_v2_1"
        include_training_pool = True
        action = "恢复为可训练复核候选"
    elif is_control_v20:
        role = "control_or_weak_candidate_v2_1"
        include_training_pool = True
        action = "保留为弱响应或对照候选"
    elif recovery_priority.startswith(("A_", "B_")):
        role = "review_recovered_candidate_v2_1"
        include_training_pool = True
        action = "从Goal2排除集中恢复为复核候选"
    elif recovery_priority.startswith(("C1_", "C2_")) and (curve_context or design_grade_curve):
        role = "height_or_curve_review_v2_1"
        include_training_pool = True
        action = "保留为弯道/高度重点复核候选"
    elif is_discarded_v20:
        role = "manual_review_or_discarded_v2_1"
        include_training_pool = False
        action = "暂不训练，留作人工复核"
    else:
        role = "manual_review_v2_1"
        include_training_pool = False
        action = "人工复核后再决定"

    if include_training_pool and height_level in {"较大高度变化_重点复核", "大幅高度变化_强复核"}:
        action += "；训练前建议优先看图"
    if include_training_pool and lateral_reference_switch_warning:
        action += "；横向偏移突变只作为SILAB参考系提示"

    restored_from_goal2_exclusion = (
        as_bool(row.get("goal2_strict_slope_offroad_height_excluded"))
        and include_training_pool
        and not hard_exclude
    )

    return pd.Series(
        {
            "v2_1_role": role,
            "v2_1_include_training_pool": include_training_pool,
            "v2_1_action_cn": action,
            "v2_1_hard_exclude": hard_exclude,
            "v2_1_restored_from_goal2_exclusion": restored_from_goal2_exclusion,
            "v2_1_height_level": height_level,
            "v2_1_design_height_note": design_height_note,
            "v2_1_lateral_reference_switch_warning": lateral_reference_switch_warning,
            "v2_1_current_road_warning": current_road_warning,
            "v2_1_current_height_warning": current_height_warning,
            "v2_1_old_only_warning": old_only_warning,
            "v2_1_curve_context": curve_context,
            "v2_1_design_grade_curve": design_grade_curve,
            "v2_1_z_residual_range": z_residual,
            "v2_1_z_rise_from_start": z_rise,
            "v2_1_z_drop_from_start": z_drop,
        }
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无记录_"
    return df.to_markdown(index=False)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH, low_memory=False)
    if "split" not in manifest.columns:
        subject_col = "subject" if "subject" in manifest.columns else "subject_id"
        test_subjects = {"cwh", "gf", "tyy"}
        val_subjects = {"byx", "gzj", "yyl"}

        def infer_split(subject: object) -> str:
            subject_text = as_text(subject).strip().lower()
            if subject_text in test_subjects:
                return "test"
            if subject_text in val_subjects:
                return "val"
            if subject_text:
                return "train"
            return "unknown"

        manifest["split"] = manifest[subject_col].map(infer_split) if subject_col in manifest.columns else "unknown"
    if EXCLUSION_BREAKDOWN_PATH.exists():
        exclusion = pd.read_csv(EXCLUSION_BREAKDOWN_PATH, low_memory=False)
        merge_cols = [
            c
            for c in [
                "episode_uid",
                "actual_exclusion_reasons",
                "actual_reason_count",
                "recovery_priority",
                "recovery_suggestion_cn",
                "review_image_path",
            ]
            if c in exclusion.columns
        ]
        manifest = manifest.merge(exclusion[merge_cols], on="episode_uid", how="left")
    else:
        manifest["actual_exclusion_reasons"] = ""
        manifest["recovery_priority"] = ""

    v21_cols = manifest.apply(assign_v21, axis=1)
    out = pd.concat([manifest, v21_cols], axis=1)

    write_csv(out, TABLE_DIR / "manifest_all_v2_1_reference_height_recovery.csv")
    write_csv(
        out[out["v2_1_include_training_pool"]],
        TABLE_DIR / "manifest_training_pool_v2_1.csv",
    )
    write_csv(
        out[out["v2_1_role"].eq("main_train_candidate_v2_1")],
        TABLE_DIR / "manifest_main_train_v2_1.csv",
    )
    write_csv(
        out[out["v2_1_role"].eq("review_recovered_candidate_v2_1")],
        TABLE_DIR / "manifest_review_recovered_v2_1.csv",
    )
    write_csv(
        out[out["v2_1_role"].eq("control_or_weak_candidate_v2_1")],
        TABLE_DIR / "manifest_control_or_weak_v2_1.csv",
    )
    write_csv(
        out[out["v2_1_hard_exclude"]],
        TABLE_DIR / "manifest_hard_excluded_v2_1.csv",
    )
    write_csv(
        out[out["v2_1_height_level"].isin(["中等高度变化_需要复核", "较大高度变化_重点复核", "大幅高度变化_强复核"])],
        TABLE_DIR / "manifest_height_review_v2_1.csv",
    )
    write_csv(
        out[out["v2_1_lateral_reference_switch_warning"]],
        TABLE_DIR / "manifest_lateral_reference_switch_review_v2_1.csv",
    )

    role_summary = (
        out.groupby("v2_1_role", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    split_summary = (
        out.groupby(["split", "v2_1_role"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "count"], ascending=[True, False])
    )
    height_summary = (
        out.groupby(["v2_1_height_level", "v2_1_role"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["v2_1_height_level", "count"], ascending=[True, False])
    )
    restored_summary = (
        out.groupby(["v2_1_restored_from_goal2_exclusion", "v2_1_role"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["v2_1_restored_from_goal2_exclusion", "count"], ascending=[False, False])
    )
    curve_summary = (
        out.groupby(["v2_1_curve_context", "v2_1_role"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["v2_1_curve_context", "count"], ascending=[False, False])
    )

    write_csv(role_summary, TABLE_DIR / "v2_1_role_summary.csv")
    write_csv(split_summary, TABLE_DIR / "v2_1_split_role_summary.csv")
    write_csv(height_summary, TABLE_DIR / "v2_1_height_role_summary.csv")
    write_csv(restored_summary, TABLE_DIR / "v2_1_goal2_restored_summary.csv")
    write_csv(curve_summary, TABLE_DIR / "v2_1_curve_role_summary.csv")

    total = len(out)
    training_pool = int(out["v2_1_include_training_pool"].sum())
    restored = int(out["v2_1_restored_from_goal2_exclusion"].sum())
    hard_excluded = int(out["v2_1_hard_exclude"].sum())
    lat_warn = int(out["v2_1_lateral_reference_switch_warning"].sum())
    small_height = int(out["v2_1_height_level"].eq("小幅高度变化_不作为排除依据").sum())
    goal2_excluded = int(out["goal2_strict_slope_offroad_height_excluded"].map(as_bool).sum())
    goal2_excluded_restored = int(
        (
            (out["goal2_strict_slope_offroad_height_excluded"].map(as_bool))
            & (out["v2_1_include_training_pool"])
            & (~out["v2_1_hard_exclude"])
        ).sum()
    )

    report = f"""# v2.1 横向偏移参考系与道路高程修正后样本表

生成时间：2026-05-26

## 这版修正了什么

1. `SILAB` 中横向偏移在换道/跨道路参考线时可能出现跳变，因此横向偏移突变不再作为硬排除条件，只作为“参考系切换风险提示”。
2. 道路设计文件和道路中心线显示：`curve1/curve2` 本身存在米级高程变化，因此不能用原始 `z` 范围直接判断“上斜坡/下马路”。
3. 车辆侧倾、车身姿态和仿真噪声可能带来厘米级或十几厘米级高度变化。v2.1 中：
   - `z_residual < 0.20 m`：不作为排除依据；
   - `0.20-0.50 m`：进入复核；
   - `0.50-1.00 m`：重点复核；
   - `>=1.00 m`：强复核，只有结合当前道路/高度证据才暂不训练。
4. 旧版本文字里的“路边/下马路/上斜坡/高度异常”不再直接继承为硬排除，只作为历史提示。

## 总体数量

| 项目 | 数量 |
|---|---:|
| 全部 episode | {total} |
| Goal2 严格排除样本 | {goal2_excluded} |
| v2.1 可进入训练池/复核训练池 | {training_pool} |
| 从 Goal2 严格排除集中恢复 | {restored} |
| 其中 Goal2 排除但 v2.1 恢复到训练池 | {goal2_excluded_restored} |
| v2.1 硬排除 | {hard_excluded} |
| 横向偏移参考系风险提示 | {lat_warn} |
| 小幅高度变化且不作为排除依据 | {small_height} |

## v2.1 角色分布

{markdown_table(role_summary)}

## 按数据划分统计

{markdown_table(split_summary)}

## 高度规则与角色分布

{markdown_table(height_summary)}

## Goal2 排除样本恢复情况

{markdown_table(restored_summary)}

## 弯道/非弯道上下文分布

{markdown_table(curve_summary)}

## 当前建议

- 这版只是样本规则修正和候选表，不训练模型。
- 下一步应先从 `manifest_training_pool_v2_1.csv` 中按角色抽样看图，尤其看：
  - `height_or_curve_review_v2_1`
  - `review_recovered_candidate_v2_1`
  - `manifest_lateral_reference_switch_review_v2_1.csv`
- 如果人工确认这些恢复样本多数合理，再基于 v2.1 生成新的 vehicle-only 数据集和共同评价集。
- 不建议再把“横向偏移突变”直接当作车辆真实横向突变；它应和方向盘、横摆、横滚、车速、制动、道路坐标一起解释。
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    (OUT_ROOT / "record_episode_dataset_v2_1_summary_cn.md").write_text(report, encoding="utf-8")

    print(f"[OK] wrote {TABLE_DIR}")
    print(f"[OK] wrote {REPORT_PATH}")
    print(f"[SUMMARY] total={total} training_pool={training_pool} restored={restored} hard_excluded={hard_excluded}")


if __name__ == "__main__":
    main()
