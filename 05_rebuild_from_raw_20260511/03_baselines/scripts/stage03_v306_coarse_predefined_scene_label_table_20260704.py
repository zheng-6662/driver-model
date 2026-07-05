#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v306 coarse predefined scene label table.

目的：
- 按用户重新确认的粗场景体系，把 v305 的细事件标签收敛为更接近实验条件的场景标签；
- 过弯直接使用当前 rolling manifest 中已有的 scene_type，拆成“下坡过弯 / 平路过弯”；
- 直道内的“连续变道 / 紧急变道失稳”先用 v305 seed 与当前 route/shape 标志生成初稿；
- 为下一步 v307 条件模型提供统一的 coarse_scene_label 输入。

重要边界：
- curve_downhill / curve_flat 来自 v236 rolling manifest 的 scene_type，可视为当前样本表中已存在的场景条件；
- continuous_lane_change / emergency_lane_change_instability 仍部分依赖 v305/v301 自动 seed，
  因此是人工审核 seed，不是最终人工标签；
- other_or_uncertain 不作为明确场景结论，只是避免强行给所有直道样本贴错标签。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 20260704
ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "03_baselines"
SCRIPTS = BASELINES / "scripts"
V304_SCRIPT = SCRIPTS / "stage03_v304_fixed_event_label_conditioned_curve_model_20260703.py"
V305_LABELS = BASELINES / "v305_formal_predefined_event_label_table_20260704" / "tables" / "v305_formal_event_labels.csv"

OUT = BASELINES / "v306_coarse_predefined_scene_label_table_20260704"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"


COARSE_SCENE_ORDER = [
    "curve_downhill",
    "curve_flat",
    "continuous_lane_change",
    "emergency_lane_change_instability",
    "other_or_uncertain",
]

COARSE_SCENE_CN = {
    "curve_downhill": "下坡过弯",
    "curve_flat": "平路过弯",
    "continuous_lane_change": "连续变道/连续左右修正",
    "emergency_lane_change_instability": "紧急变道/猛打方向失稳",
    "other_or_uncertain": "其他/不确定",
}

EMERGENCY_FORMAL_TYPES = {"紧急避让/连续变道", "复合制动转向"}
SHARP_TURN_FORMAL_TYPES = {"急左转", "急右转"}
CONTINUOUS_FORMAL_TYPES = {"连续变道/横向避让"}
CONTINUOUS_ROUTE_EVENTS = {"zero_cross", "multi_correction", "reverse"}


def import_module_from_path(module_name: str, path: Path):
    """按路径导入前序脚本，只复用其已审计的数据构造逻辑。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少依赖脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V304 = import_module_from_path("stage03_v304_for_v306_scene_labels", V304_SCRIPT)


def ensure_dirs() -> None:
    """创建 v306 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v306 自己的输出，不触碰前序实验。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 使用 utf-8-sig，方便 Windows/Excel 直接查看中文。"""

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


def as_bool(value: object) -> bool:
    """鲁棒读取 bool/int/string 标志位。"""

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


def choose_coarse_scene(row: pd.Series) -> str:
    """
    生成粗场景标签。

    规则优先级：
    1. 过弯由 scene_type 直接决定；
    2. 直道中的紧急变道/猛打方向失稳优先于连续左右修正；
    3. 直道中仍不明确的样本进入 other_or_uncertain。
    """

    scene_type = str(row.get("scene_type", ""))
    if scene_type == "下坡弯道事件":
        return "curve_downhill"
    if scene_type == "普通弯道事件":
        return "curve_flat"

    formal = str(row.get("formal_primary_type", ""))
    route_event = str(row.get("route_event", ""))
    secondary = str(row.get("formal_secondary_tags", ""))
    strong_steer = as_bool(row.get("strong_steer", False))
    vehicle_strong = as_bool(row.get("vehicle_strong", False))
    reverse = as_bool(row.get("reverse", False))
    multi_correction = as_bool(row.get("multi_correction", False))
    fast_steer = as_bool(row.get("flag_fast_steer", False)) or ("快速转向" in secondary)
    high_yaw_or_ay = as_bool(row.get("flag_high_yaw_or_ay", False)) or ("高横摆/高横向加速度" in secondary)

    if formal in EMERGENCY_FORMAL_TYPES:
        return "emergency_lane_change_instability"
    if formal in SHARP_TURN_FORMAL_TYPES and strong_steer and (vehicle_strong or high_yaw_or_ay or route_event in {"extreme_peak", "vehicle_strong"}):
        return "emergency_lane_change_instability"
    if strong_steer and fast_steer and high_yaw_or_ay and (vehicle_strong or route_event in {"strong_event", "extreme_peak", "vehicle_strong"}):
        return "emergency_lane_change_instability"

    if formal in CONTINUOUS_FORMAL_TYPES:
        return "continuous_lane_change"
    if route_event in CONTINUOUS_ROUTE_EVENTS or reverse or multi_correction:
        return "continuous_lane_change"

    return "other_or_uncertain"


def label_source_level(row: pd.Series) -> str:
    """说明每个粗标签的主要来源，便于后续 guardrail 和人工审核。"""

    label = str(row.get("coarse_scene_label", ""))
    if label in {"curve_downhill", "curve_flat"}:
        return "v236_scene_type_predefined"
    if label in {"continuous_lane_change", "emergency_lane_change_instability"}:
        return "v305_future_behavior_seed_plus_current_manifest_flags"
    return "current_manifest_straight_other_seed"


def review_priority(row: pd.Series) -> str:
    """按风险给人工复核排序。"""

    label = str(row.get("coarse_scene_label", ""))
    if label in {"continuous_lane_change", "emergency_lane_change_instability"}:
        return "high"
    if label == "other_or_uncertain":
        return "medium"
    if as_bool(row.get("within_bad_top10_by_v249", False)) or as_bool(row.get("v299_vehicle_ambiguous", False)):
        return "medium"
    return "low"


def build_event_manifest() -> pd.DataFrame:
    """复用 v304/v300 已审计路径，取当前 1167 个 delay0 事件。"""

    prepared = V304.prepare_v304_data(hard_event_extra=0.0)
    manifest = prepared.data.manifest.copy()
    events = manifest[manifest["delay_ms"].astype(int).eq(0)].copy().reset_index(drop=True)
    if events["event_uid"].duplicated().any():
        dup = events.loc[events["event_uid"].duplicated(), "event_uid"].head(5).tolist()
        raise AssertionError(f"delay0 event_uid 重复：{dup}")
    return events


def build_coarse_labels(events: pd.DataFrame) -> pd.DataFrame:
    """合并 v305 seed，生成粗场景标签表。"""

    if not V305_LABELS.exists():
        raise FileNotFoundError(f"缺少 v305 标签表：{V305_LABELS}")
    v305 = pd.read_csv(V305_LABELS, encoding="utf-8-sig")
    keep = [
        "event_uid",
        "event_primary_type",
        "formal_primary_type",
        "formal_secondary_tags",
        "auto_label_confidence",
        "manual_review_needed",
        "within_bad_top10_by_v249",
        "within_bad_top20_by_v249",
        "v299_vehicle_ambiguous",
        "v300_rmse",
        "flag_fast_steer",
        "flag_high_yaw_or_ay",
        "flag_emergency_lane_change",
        "flag_lane_change_or_swerve",
        "flag_compound_brake_turn",
        "flag_sharp_turn",
    ]
    v305 = v305[[c for c in keep if c in v305.columns]].copy()
    out = events.merge(v305, on="event_uid", how="left", validate="one_to_one")
    if out["formal_primary_type"].isna().any():
        missing = out.loc[out["formal_primary_type"].isna(), "event_uid"].head(10).tolist()
        raise AssertionError(f"v305 未覆盖当前事件：{missing}")

    out["coarse_scene_label"] = out.apply(choose_coarse_scene, axis=1)
    out["coarse_scene_label_cn"] = out["coarse_scene_label"].map(COARSE_SCENE_CN)
    out["coarse_scene_class_index"] = out["coarse_scene_label"].map({name: i for i, name in enumerate(COARSE_SCENE_ORDER)}).astype(int)
    out["coarse_scene_source_level"] = out.apply(label_source_level, axis=1)
    out["coarse_scene_manual_review_status"] = "auto_seed_needs_review"
    out["coarse_scene_review_priority"] = out.apply(review_priority, axis=1)
    out["coarse_scene_can_be_model_input_after_confirmation"] = True
    out["curve_scene_label_from_predefined_scene_type"] = out["coarse_scene_label"].isin(["curve_downhill", "curve_flat"])
    out["noncurve_subtype_seed_requires_manual_confirmation"] = out["coarse_scene_label"].isin(
        ["continuous_lane_change", "emergency_lane_change_instability"]
    )
    out["uses_future_behavior_seed_for_noncurve_subtype"] = out["coarse_scene_source_level"].eq(
        "v305_future_behavior_seed_plus_current_manifest_flags"
    )

    keep_cols = [
        "event_uid",
        "sample_id",
        "subject",
        "recording",
        "split",
        "scene_type",
        "route_event",
        "observation_s",
        "coarse_scene_label",
        "coarse_scene_label_cn",
        "coarse_scene_class_index",
        "coarse_scene_source_level",
        "coarse_scene_manual_review_status",
        "coarse_scene_review_priority",
        "coarse_scene_can_be_model_input_after_confirmation",
        "curve_scene_label_from_predefined_scene_type",
        "noncurve_subtype_seed_requires_manual_confirmation",
        "uses_future_behavior_seed_for_noncurve_subtype",
        "formal_primary_type",
        "event_primary_type",
        "formal_secondary_tags",
        "auto_label_confidence",
        "manual_review_needed",
        "strong_steer",
        "vehicle_strong",
        "normal_curve",
        "reverse",
        "multi_correction",
        "observe_later_like",
        "within_bad_top10_by_v249",
        "within_bad_top20_by_v249",
        "v299_vehicle_ambiguous",
        "v299_oracle_shape_label",
        "v299_oracle_direction_label",
        "v300_rmse",
        "flag_fast_steer",
        "flag_high_yaw_or_ay",
        "flag_emergency_lane_change",
        "flag_lane_change_or_swerve",
        "flag_compound_brake_turn",
        "flag_sharp_turn",
    ]
    return out[[c for c in keep_cols if c in out.columns]].copy()


def build_tables(labels: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """生成计数、交叉表和人工审核包。"""

    total = labels.groupby(["coarse_scene_label", "coarse_scene_label_cn"], dropna=False).size().reset_index(name="total_n")
    total["ratio"] = total["total_n"] / max(len(labels), 1)
    total["coarse_scene_label"] = pd.Categorical(total["coarse_scene_label"], COARSE_SCENE_ORDER, ordered=True)
    total = total.sort_values("coarse_scene_label").reset_index(drop=True)

    by_split = (
        labels.groupby(["coarse_scene_label", "coarse_scene_label_cn", "split"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["coarse_scene_label", "split"])
    )
    scene_cross = pd.crosstab(labels["scene_type"], labels["coarse_scene_label"]).reset_index()
    formal_cross = pd.crosstab(labels["formal_primary_type"], labels["coarse_scene_label"]).reset_index()

    review_order = {"high": 0, "medium": 1, "low": 2}
    review = labels.copy()
    review["_priority_rank"] = review["coarse_scene_review_priority"].map(review_order).fillna(9)
    sort_cols: List[str] = ["_priority_rank"]
    ascending = [True]
    if "v300_rmse" in review.columns:
        sort_cols.append("v300_rmse")
        ascending.append(False)
    review = review.sort_values(sort_cols, ascending=ascending).drop(columns=["_priority_rank"]).reset_index(drop=True)

    return {
        "coarse_scene_counts_total": total,
        "coarse_scene_counts_by_split": by_split,
        "scene_type_to_coarse_scene_crosstab": scene_cross,
        "v305_formal_to_coarse_scene_crosstab": formal_cross,
        "coarse_scene_manual_review_seed_pack": review,
    }


def plot_counts(total: pd.DataFrame) -> Path:
    """绘制粗场景标签分布。"""

    path = FIGURES / "v306_coarse_scene_label_distribution.png"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = total["coarse_scene_label_cn"].astype(str).tolist()
    vals = total["total_n"].astype(int).tolist()
    ax.bar(names, vals, color="#4c78a8")
    ax.set_title("v306 coarse predefined scene labels")
    ax.set_ylabel("event count")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(labels: pd.DataFrame, tables: Dict[str, pd.DataFrame], guardrail: Dict[str, object]) -> Path:
    """写中文报告。"""

    path = REPORTS / "v306_coarse_predefined_scene_label_table_cn.md"
    total = tables["coarse_scene_counts_total"]
    high_n = int(labels["coarse_scene_review_priority"].eq("high").sum())
    medium_n = int(labels["coarse_scene_review_priority"].eq("medium").sum())
    curve_n = int(labels["curve_scene_label_from_predefined_scene_type"].sum())
    noncurve_seed_n = int(labels["uses_future_behavior_seed_for_noncurve_subtype"].sum())
    lines = [
        "# v306 coarse predefined scene label table",
        "",
        "## 这一步做了什么",
        "",
        "本轮按用户重新确认的粗场景体系，把当前 1167 个事件收敛为下坡过弯、平路过弯、连续变道/连续左右修正、紧急变道/猛打方向失稳、其他/不确定五类。",
        "",
        "这张表的目的不是继续细分“急左转/急右转/多段修正”，而是把模型条件输入改成更接近实验条件本身的粗场景标签。",
        "",
        "## 标签分布",
        "",
        total.to_markdown(index=False),
        "",
        "## 输入边界",
        "",
        f"- 过弯标签来自当前 rolling manifest 的 `scene_type`，共 `{curve_n}` 个事件，可作为预测前场景条件 seed。",
        f"- 直道内连续/紧急子类仍部分使用 v305/v301 自动 seed，共 `{noncurve_seed_n}` 个事件，需要人工或实验条件确认后才能写成最终标签。",
        "- `other_or_uncertain` 不强行解释为某种实验事件，只用于避免把普通或不清楚直道样本误贴成连续/紧急变道。",
        "",
        "## 人工审核工作量",
        "",
        f"- high priority：`{high_n}` 个事件，主要是直道内连续/紧急子类。",
        f"- medium priority：`{medium_n}` 个事件，主要是其他/不确定或关键误差样本。",
        "",
        "## 下一步",
        "",
        "- v307 可直接用 `coarse_scene_label` 替换 v304 的 `event_primary_type` 条件输入，先做一轮训练对比。",
        "- 如果 v307 有收益，再优先人工复核 high priority 中的连续/紧急变道样本。",
        "- 如果 v307 不如 v304，则说明粗场景标签可能丢失了有用细节，需要在粗标签内保留少量二级标签，而不是回到过细的未来形状标签。",
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
    """打包 v306 产物。"""

    zip_path = OUT / "v306_coarse_predefined_scene_label_table_20260704.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path == zip_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(OUT))
    return zip_path


def main() -> None:
    clean_out_dir()
    events = build_event_manifest()
    labels = build_coarse_labels(events)
    write_csv(labels, TABLES / "v306_coarse_scene_event_labels.csv")

    tables = build_tables(labels)
    for name, df in tables.items():
        write_csv(df, TABLES / f"v306_{name}.csv")
    figure_path = plot_counts(tables["coarse_scene_counts_total"])

    input_hashes = pd.DataFrame(
        [
            {"input_name": "v304_script_for_current_manifest", "path": str(V304_SCRIPT), "sha256": file_sha256(V304_SCRIPT)},
            {"input_name": "v305_formal_event_labels", "path": str(V305_LABELS), "sha256": file_sha256(V305_LABELS)},
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    guardrail = {
        "pass": True,
        "version": "v306_coarse_predefined_scene_label_table_20260704",
        "event_n": int(len(labels)),
        "coarse_scene_class_n": int(labels["coarse_scene_label"].nunique()),
        "coarse_scene_order": COARSE_SCENE_ORDER,
        "task_allows_predefined_scene_label_input": True,
        "curve_scene_labels_from_current_scene_type": True,
        "noncurve_subtypes_require_manual_or_experiment_confirmation": True,
        "uses_future_behavior_seed_for_some_noncurve_subtypes": bool(labels["uses_future_behavior_seed_for_noncurve_subtype"].any()),
        "deployable_without_noncurve_manual_confirmation": False,
        "label_available_before_prediction_assumption": True,
        "curve_event_n": int(labels["curve_scene_label_from_predefined_scene_type"].sum()),
        "noncurve_future_seed_n": int(labels["uses_future_behavior_seed_for_noncurve_subtype"].sum()),
        "high_priority_review_n": int(labels["coarse_scene_review_priority"].eq("high").sum()),
        "medium_priority_review_n": int(labels["coarse_scene_review_priority"].eq("medium").sum()),
        "figure_paths": [str(figure_path)],
    }
    report_path = write_report(labels, tables, guardrail)
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
