#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v305 formal predefined event label table.

目的：
- 响应“事件可以提前定好，相当于给每个事件打标签”的任务设定；
- 从 v301 自动事件标签草稿生成一张更适合作为正式输入的事件标签表；
- 把“预测前可作为条件输入的主事件类型”和“更依赖未来形状的诊断标签”分开；
- 为后续 v304/v305 条件模型替换人工/实验条件标签提供固定表结构。

重要边界：
- 当前 seed 表仍来自 v301 future_behavior_auto_draft，因此只是人工标注初稿；
- 在人工审核或实验条件确认之前，不能把它写成最终可部署标签；
- 但一旦每个事件的主类型可在预测前确定，formal_primary_type 就可以作为模型输入。
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "03_baselines"
V301_DIR = BASELINES / "v301_event_type_multiclass_label_audit_20260703"
V301_LABELS = V301_DIR / "tables" / "v301_event_type_labels.csv"

OUT = BASELINES / "v305_formal_predefined_event_label_table_20260704"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"


FORMAL_PRIMARY_ORDER = [
    "普通/轻微/不确定",
    "急停/强减速",
    "急左转",
    "急右转",
    "连续变道/横向避让",
    "紧急避让/连续变道",
    "复合制动转向",
]

PLOT_LABEL_EN = {
    "普通/轻微/不确定": "normal/uncertain",
    "急停/强减速": "hard brake",
    "急左转": "sharp left",
    "急右转": "sharp right",
    "连续变道/横向避让": "lane/swerve",
    "紧急避让/连续变道": "emergency swerve",
    "复合制动转向": "compound brake-turn",
}


def ensure_dirs() -> None:
    """创建 v305 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """保存 CSV，使用 utf-8-sig 方便 Excel 查看中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, object], path: Path) -> None:
    """保存 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """计算文件 sha256。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def b(row: pd.Series, col: str) -> bool:
    """鲁棒读取布尔列。"""

    value = row.get(col, False)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def choose_formal_primary(row: pd.Series) -> str:
    """
    把 v301 自动标签收敛为更适合预测前输入的主事件类型。

    注意：这里用 v301 seed 生成初稿；真正论文/部署口径需要人工或实验条件确认。
    """

    direction = str(row.get("true_direction", "")).lower()
    if b(row, "flag_compound_brake_turn"):
        return "复合制动转向"
    if b(row, "flag_emergency_lane_change"):
        return "紧急避让/连续变道"
    if b(row, "flag_emergency_speed_drop") or (b(row, "flag_brake_or_decel") and b(row, "flag_speed_drop")):
        return "急停/强减速"
    if b(row, "flag_sharp_turn") and direction == "left":
        return "急左转"
    if b(row, "flag_sharp_turn") and direction == "right":
        return "急右转"
    if b(row, "flag_lane_change_or_swerve") or b(row, "flag_large_lateral_move"):
        return "连续变道/横向避让"
    return "普通/轻微/不确定"


def build_secondary_tags(row: pd.Series) -> List[str]:
    """把更像未来过程形状或诊断属性的信息放到 secondary/diagnostic。"""

    tags: List[str] = []
    if b(row, "flag_late_response") or b(row, "true_late_peak_flag"):
        tags.append("晚响应")
    if b(row, "flag_continuous_correction") or b(row, "true_multi_correction_flag"):
        tags.append("多段修正")
    if b(row, "flag_fast_steer"):
        tags.append("快速转向")
    if b(row, "flag_high_yaw_or_ay"):
        tags.append("高横摆/高横向加速度")
    if b(row, "flag_strong_steer"):
        tags.append("强转向")
    if b(row, "flag_extreme_steer"):
        tags.append("极端转向")
    if b(row, "manual_review_needed"):
        tags.append("需人工复核")
    if not tags:
        tags.append("无")
    return tags


def review_priority(row: pd.Series) -> str:
    """给人工审核排序，优先看最可能影响模型结论的样本。"""

    source_primary = str(row.get("event_primary_type", ""))
    auto_conf = str(row.get("auto_label_confidence", "")).lower()
    high_source_shape = source_primary in {"多段修正", "晚响应/长事件"}
    if b(row, "manual_review_needed") or b(row, "within_bad_top10_by_v249") or high_source_shape or auto_conf != "high":
        return "high"
    if b(row, "within_bad_top20_by_v249") or b(row, "v299_vehicle_ambiguous"):
        return "medium"
    return "low"


def build_formal_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """生成正式标签表初稿。"""

    out = labels.copy()
    out["formal_primary_type"] = out.apply(choose_formal_primary, axis=1)
    out["formal_secondary_tags"] = out.apply(lambda r: "|".join(build_secondary_tags(r)), axis=1)
    out["primary_label_input_eligible_if_predefined"] = True
    out["label_available_before_prediction_assumption"] = True
    out["current_seed_derived_from_future_behavior"] = True
    out["requires_manual_or_experiment_confirmation"] = True
    out["deployable_as_model_input_after_confirmation"] = True
    out["diagnostic_tags_as_direct_input_allowed"] = False
    out["manual_review_status"] = "auto_seed_needs_review"
    out["review_priority"] = out.apply(review_priority, axis=1)
    out["formal_class_index"] = out["formal_primary_type"].map({name: i for i, name in enumerate(FORMAL_PRIMARY_ORDER)}).astype(int)
    keep_cols = [
        "event_uid",
        "subject",
        "recording",
        "split",
        "observation_s",
        "raw_vehicle_csv",
        "event_primary_type",
        "event_secondary_types",
        "formal_primary_type",
        "formal_class_index",
        "formal_secondary_tags",
        "auto_label_confidence",
        "label_source_level",
        "primary_label_input_eligible_if_predefined",
        "label_available_before_prediction_assumption",
        "current_seed_derived_from_future_behavior",
        "requires_manual_or_experiment_confirmation",
        "deployable_as_model_input_after_confirmation",
        "diagnostic_tags_as_direct_input_allowed",
        "manual_review_status",
        "review_priority",
        "manual_review_needed",
        "within_bad_top10_by_v249",
        "within_bad_top20_by_v249",
        "v299_bad_top10",
        "v299_vehicle_ambiguous",
        "v300_rmse",
        "flag_compound_brake_turn",
        "flag_emergency_lane_change",
        "flag_lane_change_or_swerve",
        "flag_brake_or_decel",
        "flag_speed_drop",
        "flag_emergency_speed_drop",
        "flag_sharp_turn",
        "flag_late_response",
        "flag_continuous_correction",
        "flag_high_yaw_or_ay",
        "flag_fast_steer",
        "flag_large_lateral_move",
        "true_direction",
        "true_peak_abs",
        "true_peak_time_s",
        "true_final_delta",
    ]
    return out[[c for c in keep_cols if c in out.columns]].copy()


def build_summary_tables(formal: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """生成计数、交叉表和人工审核包。"""

    count = (
        formal.groupby(["formal_primary_type", "split"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["formal_primary_type", "split"])
    )
    total = formal.groupby("formal_primary_type", dropna=False).size().reset_index(name="total_n")
    total["ratio"] = total["total_n"] / max(len(formal), 1)
    total["formal_primary_type"] = pd.Categorical(total["formal_primary_type"], FORMAL_PRIMARY_ORDER, ordered=True)
    total = total.sort_values("formal_primary_type").reset_index(drop=True)

    cross = pd.crosstab(formal["event_primary_type"], formal["formal_primary_type"]).reset_index()
    review_order = {"high": 0, "medium": 1, "low": 2}
    review = formal.copy()
    review["_priority_rank"] = review["review_priority"].map(review_order).fillna(9)
    sort_cols = ["_priority_rank"]
    ascending = [True]
    if "v300_rmse" in review.columns:
        sort_cols.append("v300_rmse")
        ascending.append(False)
    review = review.sort_values(sort_cols, ascending=ascending).drop(columns=["_priority_rank"]).reset_index(drop=True)
    return {
        "formal_primary_counts_by_split": count,
        "formal_primary_counts_total": total,
        "v301_to_formal_primary_crosstab": cross,
        "manual_review_seed_pack": review,
    }


def plot_counts(total: pd.DataFrame) -> Path:
    """绘制 formal primary 分布。"""

    path = FIGURES / "v305_formal_primary_type_distribution.png"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = [PLOT_LABEL_EN.get(str(x), str(x)) for x in total["formal_primary_type"].astype(str).tolist()]
    vals = total["total_n"].astype(int).tolist()
    ax.bar(names, vals, color="#4c78a8")
    ax.set_title("v305 formal predefined event labels: primary type distribution")
    ax.set_ylabel("event count")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(formal: pd.DataFrame, tables: Dict[str, pd.DataFrame], guardrail: Dict[str, object]) -> Path:
    """写中文报告。"""

    path = REPORTS / "v305_formal_predefined_event_label_table_cn.md"
    high_n = int((formal["review_priority"] == "high").sum())
    medium_n = int((formal["review_priority"] == "medium").sum())
    lines = [
        "# v305 formal predefined event label table",
        "",
        "## 这一步做了什么",
        "",
        "本轮把“事件可以提前定好”正式落成一张事件标签表。它不直接继续训练模型，而是先把主事件类型、辅助诊断标签和人工审核状态分开，避免把未来轨迹形状误当成预测前输入。",
        "",
        "当前表由 v301 自动事件标签草稿生成，因此仍是人工审核 seed，不是最终人工标签。后续如果用户或实验条件确认每个事件的主类型，`formal_primary_type` 就可以作为 v304/v305 条件模型的正式输入。",
        "",
        "## 主标签设计",
        "",
        "- 可作为条件输入的主标签：`普通/轻微/不确定`、`急停/强减速`、`急左转`、`急右转`、`连续变道/横向避让`、`紧急避让/连续变道`、`复合制动转向`。",
        "- 更依赖未来过程形状的内容，如 `晚响应`、`多段修正`、`快速转向`，放入 `formal_secondary_tags`，默认不作为直接输入。",
        "",
        "## 标签分布",
        "",
        tables["formal_primary_counts_total"].to_markdown(index=False),
        "",
        "## 人工审核工作量",
        "",
        f"- high priority：`{high_n}` 个事件。",
        f"- medium priority：`{medium_n}` 个事件。",
        "- 审核优先级主要来自：原 v301 需人工复核、v249/v300 高误差、原标签为多段修正/晚响应这类更像未来形状的标签、或自动置信度不足。",
        "",
        "## 当前判断",
        "",
        "- 如果事件主类型确实能在预测前由人工、实验条件、感知/规划模块确定，那么它可以作为合法输入。",
        "- 当前 v305 表把这个输入边界固定下来：主类型可输入，诊断标签默认不可直接输入。",
        "- 下一步应让人工审核 `manual_review_seed_pack.csv`，确认或修改 `formal_primary_type` 和 `manual_review_status`。",
        "- 人工确认后，再用这张表替换 v304 里的 v301 自动标签，重跑 fixed event-label conditioned 曲线模型。",
        "",
        "## guardrail",
        "",
        "```json",
        json.dumps(guardrail, ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_inventory() -> pd.DataFrame:
    """写产物清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)})
    inv = pd.DataFrame(rows)
    write_csv(inv, LOGS / "file_inventory.csv")
    return inv


def make_zip() -> Path:
    """打包 v305 产物。"""

    zip_path = OUT / "v305_formal_predefined_event_label_table_20260704.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path == zip_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(OUT))
    return zip_path


def main() -> None:
    ensure_dirs()
    if not V301_LABELS.exists():
        raise FileNotFoundError(f"缺少 v301 标签表：{V301_LABELS}")
    labels = pd.read_csv(V301_LABELS)
    formal = build_formal_labels(labels)
    write_csv(formal, TABLES / "v305_formal_event_labels.csv")

    tables = build_summary_tables(formal)
    for name, df in tables.items():
        write_csv(df, TABLES / f"v305_{name}.csv")
    figure_path = plot_counts(tables["formal_primary_counts_total"])

    input_hashes = pd.DataFrame([{"input_name": "v301_event_type_labels", "path": str(V301_LABELS), "sha256": file_sha256(V301_LABELS)}])
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    guardrail = {
        "pass": True,
        "version": "v305_formal_predefined_event_label_table_20260704",
        "event_n": int(len(formal)),
        "formal_primary_class_n": int(formal["formal_primary_type"].nunique()),
        "formal_primary_order": FORMAL_PRIMARY_ORDER,
        "task_allows_predefined_event_label_input": True,
        "formal_primary_type_can_be_model_input_after_confirmation": True,
        "diagnostic_tags_as_direct_input_allowed": False,
        "current_seed_source": "v301_future_behavior_auto_draft",
        "current_seed_derived_from_future_behavior": True,
        "requires_manual_or_experiment_confirmation": True,
        "deployable_without_manual_or_experiment_confirmation": False,
        "label_available_before_prediction_assumption": True,
        "high_priority_review_n": int((formal["review_priority"] == "high").sum()),
        "medium_priority_review_n": int((formal["review_priority"] == "medium").sum()),
        "figure_paths": [str(figure_path)],
    }
    report_path = write_report(formal, tables, guardrail)
    guardrail["report_path"] = str(report_path)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_inventory()
    zip_path = make_zip()
    with zipfile.ZipFile(zip_path, "r") as zf:
        guardrail["zip_testzip"] = zf.testzip() is None
    guardrail["zip_path"] = str(zip_path)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_inventory()
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
