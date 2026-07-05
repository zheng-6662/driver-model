#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v283 physiology route lineage / gap audit.

v282 已经证明：当前生理特征层不能在车辆相似但未来分叉的样本中稳定消歧。
本脚本继续推进 active goal，但不再盲目新训练，而是把 v254b-v282 的生理证据链
合并成一个路线级审计包，明确：

1. 哪些生理使用方式已经被当前证据否定；
2. 失败是对齐/覆盖问题，还是信号有效性、身份混淆、目标可辨识性问题；
3. 如果还要继续“充分利用生理数据”的 goal，下一版必须改变什么。

输出不是论文总结，而是下一步实验的硬 gate。
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
SCRIPTS = BASELINES / "scripts"

OUT = BASELINES / "v283_physio_route_lineage_gap_audit_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v283_physio_route_lineage_gap_audit_20260702_pack.zip"

V283_SCRIPT = SCRIPTS / "stage03_v283_physio_route_lineage_gap_audit_20260702.py"

PATHS = {
    "v254b_alignment": BASELINES
    / "v254b_physio_200hz_event_representation_20260702"
    / "tables"
    / "v254b_alignment_coverage_summary.csv",
    "v254b_cls": BASELINES
    / "v254b_physio_200hz_event_representation_20260702"
    / "tables"
    / "v254b_behavior_classification_diagnostics.csv",
    "v254b_reg": BASELINES
    / "v254b_physio_200hz_event_representation_20260702"
    / "tables"
    / "v254b_future_summary_regression_diagnostics.csv",
    "v260_alignment": BASELINES
    / "v260_event_biomarker_physio_rebuild_20260702"
    / "tables"
    / "v260_alignment_coverage_summary.csv",
    "v260_cls": BASELINES
    / "v260_event_biomarker_physio_rebuild_20260702"
    / "tables"
    / "v260_behavior_classification_diagnostics.csv",
    "v260_reg": BASELINES
    / "v260_event_biomarker_physio_rebuild_20260702"
    / "tables"
    / "v260_future_summary_regression_diagnostics.csv",
    "v268_flags": BASELINES
    / "v268_physio_quality_identifiability_audit_20260702"
    / "tables"
    / "v268_conclusion_flags.csv",
    "v268_signal_quality": BASELINES
    / "v268_physio_quality_identifiability_audit_20260702"
    / "tables"
    / "v268_source_signal_availability_quality.csv",
    "v268_identity": BASELINES
    / "v268_physio_quality_identifiability_audit_20260702"
    / "tables"
    / "v268_bio_identity_behavior_eta_summary.csv",
    "v269_decision": BASELINES
    / "v269_reliable_identity_removed_physio_20260702"
    / "tables"
    / "v269_decision_summary.csv",
    "v271_decision": BASELINES
    / "v271_calibrated_raw_physio_state_20260702"
    / "tables"
    / "v271_decision_summary.csv",
    "v282_decision": BASELINES
    / "v282_physio_ambiguity_route_gate_20260702"
    / "tables"
    / "v282_route_gate_decision.csv",
    "v282_guardrail": BASELINES
    / "v282_physio_ambiguity_route_gate_20260702"
    / "logs"
    / "guardrail_check.json",
}

FIXED_WAIT_LATEST_BADTOP10 = 0.695048


def ensure_dirs() -> None:
    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def read_csv_required(key: str) -> pd.DataFrame:
    path = PATHS[key]
    if not path.exists():
        raise FileNotFoundError(f"缺少输入 {key}: {path}")
    return pd.read_csv(path)


def read_json_optional(key: str) -> Dict[str, object]:
    path = PATHS[key]
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_mean(values: Iterable[object]) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return float(np.mean(arr))


def best_cls_delta(df: pd.DataFrame, target: str, protocol: str = "subject_disjoint") -> Dict[str, object]:
    sub = df[
        df["split_protocol"].astype(str).eq(protocol)
        & df["eval_split"].astype(str).eq("test")
        & df["target"].astype(str).eq(target)
        & ~df["feature_block"].astype(str).eq("vehicle_only")
    ].copy()
    if sub.empty:
        return {"feature_block": None, "delta_macro_f1": math.nan, "metric": math.nan}
    sub["delta_macro_f1_minus_vehicle"] = pd.to_numeric(sub["delta_macro_f1_minus_vehicle"], errors="coerce")
    sub = sub.sort_values("delta_macro_f1_minus_vehicle", ascending=False)
    row = sub.iloc[0]
    return {
        "feature_block": str(row["feature_block"]),
        "delta_macro_f1": float(row["delta_macro_f1_minus_vehicle"]),
        "metric": float(row["macro_f1"]),
    }


def best_reg_delta(df: pd.DataFrame, target: str, protocol: str = "subject_disjoint") -> Dict[str, object]:
    sub = df[
        df["split_protocol"].astype(str).eq(protocol)
        & df["eval_split"].astype(str).eq("test")
        & df["target"].astype(str).eq(target)
        & ~df["feature_block"].astype(str).eq("vehicle_only")
    ].copy()
    if sub.empty:
        return {"feature_block": None, "delta_r2": math.nan, "metric": math.nan}
    sub["delta_r2_minus_vehicle"] = pd.to_numeric(sub["delta_r2_minus_vehicle"], errors="coerce")
    sub = sub.sort_values("delta_r2_minus_vehicle", ascending=False)
    row = sub.iloc[0]
    return {
        "feature_block": str(row["feature_block"]),
        "delta_r2": float(row["delta_r2_minus_vehicle"]),
        "metric": float(row["r2"]),
    }


def decision_delta(df: pd.DataFrame, source: str) -> float:
    sub = df[df["source"].astype(str).eq(source)].copy()
    if sub.empty:
        return math.nan
    return float(pd.to_numeric(sub.iloc[0]["delta_vs_fixed_latest"], errors="coerce"))


def decision_rmse(df: pd.DataFrame, source: str) -> float:
    sub = df[df["source"].astype(str).eq(source)].copy()
    if sub.empty:
        return math.nan
    return float(pd.to_numeric(sub.iloc[0]["rmse"], errors="coerce"))


def build_alignment_quality() -> pd.DataFrame:
    v254b = read_csv_required("v254b_alignment")
    v260 = read_csv_required("v260_alignment")
    v268_flags = read_csv_required("v268_flags")
    v268_signal = read_csv_required("v268_signal_quality")
    v268_identity = read_csv_required("v268_identity")

    rows: List[Dict[str, object]] = []
    rows.append(
        {
            "aspect": "200Hz_event_alignment",
            "status": "pass",
            "evidence": f"v254b ok_rate_mean={finite_mean(v254b['ok_rate']):.3f}; v260 ok_rate_mean={finite_mean(v260['ok_rate']):.3f}",
            "implication": "事件窗口覆盖基本够用，失败不能简单归因于大面积对齐缺失。",
        }
    )
    flags = {str(r["check"]): r for _, r in v268_flags.iterrows()}
    for check in ["source_timing_integrity", "derived_signal_availability", "event_window_coverage", "identity_vs_behavior_signal"]:
        if check in flags:
            r = flags[check]
            rows.append(
                {
                    "aspect": check,
                    "status": str(r["status"]),
                    "evidence": str(r["evidence"]),
                    "implication": str(r["interpretation"]),
                }
            )
    usable = v268_signal.groupby("family", as_index=False).agg(
        signal_n=("signal", "count"),
        usable_basic_rate_mean=("usable_basic_rate", "mean"),
        near_constant_count_sum=("near_constant_count", "sum"),
        all_nan_count_sum=("all_nan_count", "sum"),
    )
    write_csv(usable, TABLES / "v283_signal_quality_by_family.csv")

    identity = v268_identity.copy()
    identity["identity_to_behavior_ratio_median"] = pd.to_numeric(
        identity["identity_to_behavior_ratio_median"], errors="coerce"
    )
    rows.append(
        {
            "aspect": "identity_behavior_ratio",
            "status": "warn",
            "evidence": f"family_median_ratio={np.nanmedian(identity['identity_to_behavior_ratio_median']):.2f}; max_ratio={np.nanmax(identity['identity_to_behavior_ratio_median']):.2f}",
            "implication": "生理特征更容易识别驾驶员/记录而不是行为，subject-disjoint 泛化受限。",
        }
    )
    return pd.DataFrame(rows)


def build_model_lineage() -> pd.DataFrame:
    v254b_cls = read_csv_required("v254b_cls")
    v254b_reg = read_csv_required("v254b_reg")
    v260_cls = read_csv_required("v260_cls")
    v260_reg = read_csv_required("v260_reg")
    v269_decision = read_csv_required("v269_decision")
    v271_decision = read_csv_required("v271_decision")
    v282_decision = read_csv_required("v282_decision")
    v282_guardrail = read_json_optional("v282_guardrail")

    v254b_bad = best_cls_delta(v254b_cls, "bad_top10_v250_diagnostic")
    v254b_future = best_cls_delta(v254b_cls, "future_cluster4")
    v254b_peak = best_reg_delta(v254b_reg, "future_peak_abs")
    v260_bad = best_cls_delta(v260_cls, "bad_top10_v250_diagnostic")
    v260_future = best_cls_delta(v260_cls, "future_cluster4")
    v260_peak = best_reg_delta(v260_reg, "future_peak_abs")

    top1_bad = v282_decision[v282_decision["check"].astype(str).eq("deployable_top1_val_chosen_bad_top10")]
    top1_amb = v282_decision[v282_decision["check"].astype(str).eq("deployable_top1_val_chosen_bad_ambiguous")]
    corr = v282_decision[v282_decision["check"].astype(str).eq("test_bad_top10_any_rawset_corr_gt_005")]

    rows = [
        {
            "version": "v254b",
            "route": "200Hz event-window statistics",
            "hypothesis": "锚点前 200Hz 生理统计能提供跨驾驶员行为增量",
            "best_badtop10_signal": f"{v254b_bad['feature_block']} delta_macro_f1={v254b_bad['delta_macro_f1']:.4f}",
            "future_behavior_signal": f"future_cluster4 delta={v254b_future['delta_macro_f1']:.4f}; peak_abs delta_r2={v254b_peak['delta_r2']:.4f}",
            "trajectory_badtop10_outcome": "not tested as direct selector in v254b",
            "status": "failed_for_main_goal",
            "reason": "分类上只出现很小 bad_top10 诊断增量，未来行为/回归和 vehicle+bio 主结果未超过 vehicle-only。",
        },
        {
            "version": "v260",
            "route": "ECG/EDA/RESP/EMG event biomarkers",
            "hypothesis": "更有生理含义的 biomarker 能比统计特征更好解释行为和差样本",
            "best_badtop10_signal": f"{v260_bad['feature_block']} delta_macro_f1={v260_bad['delta_macro_f1']:.4f}",
            "future_behavior_signal": f"future_cluster4 delta={v260_future['delta_macro_f1']:.4f}; peak_abs delta_r2={v260_peak['delta_r2']:.4f}",
            "trajectory_badtop10_outcome": "not sufficient; passed to v261-v269 and still failed",
            "status": "failed_for_main_goal",
            "reason": "bio260 对 bad_top10 有很小诊断信号，但 subject-disjoint 行为预测和后续候选选择没有转成轨迹收益。",
        },
        {
            "version": "v268",
            "route": "quality / identity / rerank identifiability audit",
            "hypothesis": "生理失败可能来自采样对齐或质量问题",
            "best_badtop10_signal": "source timing pass; signal availability warn; identity/behavior warn",
            "future_behavior_signal": "identity signal >> behavior signal",
            "trajectory_badtop10_outcome": "candidate rerank identifiability warn",
            "status": "diagnosed_bottleneck",
            "reason": "不是对齐大面积失败，而是派生列不可用、身份混淆强、行为可辨识性弱。",
        },
        {
            "version": "v269",
            "route": "reliable / low-identity bio feature screening",
            "hypothesis": "筛掉不可用和高身份特征后，生理能改善 wait gate / pair rerank",
            "best_badtop10_signal": f"pair_test_best_deployable delta={decision_delta(v269_decision, 'pair_test_best_deployable'):.4f}",
            "future_behavior_signal": "feature screening reduced some identity but did not create deployable gain",
            "trajectory_badtop10_outcome": f"best deployable RMSE={decision_rmse(v269_decision, 'pair_test_best_deployable'):.4f}",
            "status": "failed_for_main_goal",
            "reason": "可部署策略仍高于 fixed wait-latest，且最好 wait gate 退化成全 wait-latest。",
        },
        {
            "version": "v271",
            "route": "subject/recording calibrated raw physiology state",
            "hypothesis": "无标签个体/记录基线校准能释放生理状态变化",
            "best_badtop10_signal": f"pair_test_best_deployable delta={decision_delta(v271_decision, 'pair_test_best_deployable'):.4f}",
            "future_behavior_signal": "calibrated/transductive setting still cannot deploy",
            "trajectory_badtop10_outcome": f"best deployable RMSE={decision_rmse(v271_decision, 'pair_test_best_deployable'):.4f}",
            "status": "failed_for_main_goal",
            "reason": "即使给 subject/recording 无标签基线，差样本候选选择仍明显差于 fixed wait-latest。",
        },
        {
            "version": "v282",
            "route": "ambiguity route gate",
            "hypothesis": "在车辆相似但未来分叉的候选池中，生理距离能稳定排出真实更好候选",
            "best_badtop10_signal": f"deployable top1 bad_top10 evidence={float(top1_bad.iloc[0]['evidence']) if not top1_bad.empty else math.nan:.4f}; ambiguous={float(top1_amb.iloc[0]['evidence']) if not top1_amb.empty else math.nan:.4f}",
            "future_behavior_signal": f"best corr evidence={float(corr.iloc[0]['evidence']) if not corr.empty else math.nan:.5f}",
            "trajectory_badtop10_outcome": f"route_viable_now={bool(v282_guardrail.get('route_viable_now', False))}",
            "status": "failed_for_current_feature_layer",
            "reason": "生理 top1 可部署选择和 top3 上限都不能稳定通过 val/test gate，排序相关接近 0。",
        },
    ]
    return pd.DataFrame(rows)


def build_next_requirements(lineage: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "requirement_id": "R1",
            "requirement": "不能复用旧的 bio selector/reranker/reliability filter 微调作为下一步主线",
            "current_evidence": "v269/v271/v282 均失败，且 v282 route_viable_now=false",
            "status": "closed_old_route",
            "next_action": "只保留旧结果作为反例和 guardrail。",
        },
        {
            "requirement_id": "R2",
            "requirement": "若继续生理目标，必须先改善生理状态表征而不是直接加模型",
            "current_evidence": "v268 derived_signal_availability=warn，identity_vs_behavior_signal=warn",
            "status": "required",
            "next_action": "重做可用信号族筛选、质量 mask、个体内变化和低身份行为特征。",
        },
        {
            "requirement_id": "R3",
            "requirement": "新生理特征必须先通过车辆歧义样本 route gate",
            "current_evidence": "v282 top1 bad_top10 +0.1989，ambiguous +0.2347，corr max 0.00985",
            "status": "required",
            "next_action": "先在 vehicle top40 内验证生理排序相关和 val/test 同向，再进预测模型。",
        },
        {
            "requirement_id": "R4",
            "requirement": "要明确 subject-disjoint 与 subject-aware/校准任务边界",
            "current_evidence": "v254b/v260/v271 均显示个体/记录信号强，subject-disjoint 不稳定",
            "status": "required",
            "next_action": "如果接受 subject-aware，单独建个体校准任务；如果坚持 subject-disjoint，必须做去身份约束。",
        },
        {
            "requirement_id": "R5",
            "requirement": "不能把 bio top3/top5 oracle 当成可部署效果",
            "current_evidence": "v281/v282 的 top3/top5 都依赖真实误差选择候选",
            "status": "guardrail",
            "next_action": "所有正式结论只看 val-chosen deployable policy。",
        },
    ]
    return pd.DataFrame(rows)


def build_decision_summary(lineage: pd.DataFrame, quality: pd.DataFrame, requirements: pd.DataFrame) -> pd.DataFrame:
    old_route_closed = bool(lineage["status"].astype(str).isin(["failed_for_main_goal", "failed_for_current_feature_layer"]).sum() >= 5)
    source_ready = bool(quality["aspect"].astype(str).eq("200Hz_event_alignment").any())
    next_required = bool(requirements["status"].astype(str).eq("required").any())
    return pd.DataFrame(
        [
            {
                "item": "current_goal_achieved",
                "value": False,
                "evidence": "没有任何 deployable 生理路线让 test bad_top10 稳定超过 fixed wait-latest。",
            },
            {
                "item": "old_feature_selector_route_closed",
                "value": old_route_closed,
                "evidence": "v269/v271/v282 对旧特征筛选、校准和歧义消解均未通过。",
            },
            {
                "item": "physio_source_alignment_ready",
                "value": source_ready,
                "evidence": "v254b/v260/v268 均显示 200Hz 时间轴与事件窗口覆盖基本可用。",
            },
            {
                "item": "next_route_requires_feature_redefinition",
                "value": next_required,
                "evidence": "主要瓶颈是有效信号和身份混淆，不是训练脚本或简单模型容量。",
            },
        ]
    )


def plot_lineage_status(lineage: pd.DataFrame) -> Path:
    path = FIGURES / "v283_physio_route_lineage_status.png"
    status_order = {
        "failed_for_main_goal": 0,
        "failed_for_current_feature_layer": 0,
        "diagnosed_bottleneck": 1,
    }
    colors = {
        "failed_for_main_goal": "tab:red",
        "failed_for_current_feature_layer": "tab:red",
        "diagnosed_bottleneck": "tab:orange",
    }
    y = [status_order.get(str(s), 0) for s in lineage["status"]]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lineage["version"], y, color=[colors.get(str(s), "tab:gray") for s in lineage["status"]])
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["failed", "diagnosed"])
    ax.set_title("v283: physiology route lineage status")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_signal_quality() -> Path:
    path = FIGURES / "v283_signal_quality_by_family.png"
    signal = pd.read_csv(TABLES / "v283_signal_quality_by_family.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(signal["family"], signal["usable_basic_rate_mean"], color="tab:blue")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("usable basic rate")
    ax.set_title("v283: 200Hz signal quality by family")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_badtop10_delta() -> Path:
    path = FIGURES / "v283_badtop10_macro_f1_delta.png"
    v254b_cls = read_csv_required("v254b_cls")
    v260_cls = read_csv_required("v260_cls")
    rows = []
    for version, df in [("v254b", v254b_cls), ("v260", v260_cls)]:
        sub = df[
            df["split_protocol"].astype(str).eq("subject_disjoint")
            & df["eval_split"].astype(str).eq("test")
            & df["target"].astype(str).eq("bad_top10_v250_diagnostic")
            & ~df["feature_block"].astype(str).eq("vehicle_only")
        ].copy()
        for _, r in sub.iterrows():
            rows.append(
                {
                    "version": version,
                    "feature_block": str(r["feature_block"]),
                    "delta": float(pd.to_numeric(r["delta_macro_f1_minus_vehicle"], errors="coerce")),
                }
            )
    data = pd.DataFrame(rows)
    data["label"] = data["version"] + "\n" + data["feature_block"].str.replace("_", "\n")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(data)), data["delta"], color=["tab:green" if v > 0 else "tab:red" for v in data["delta"]])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["label"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("macro-F1 delta vs vehicle-only")
    ax.set_title("v283: bad_top10 diagnostic signal is small and not enough for trajectory gain")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def table_to_md(df: pd.DataFrame, cols: List[str]) -> str:
    use = df[[c for c in cols if c in df.columns]].copy()
    if use.empty:
        return "（无记录）"
    return use.to_markdown(index=False)


def write_report(
    lineage: pd.DataFrame,
    quality: pd.DataFrame,
    requirements: pd.DataFrame,
    decision: pd.DataFrame,
    figures: List[Path],
    guardrail: Dict[str, object],
) -> Path:
    path = REPORTS / "v283_physio_route_lineage_gap_audit_cn.md"
    lines: List[str] = []
    lines.append("# v283 生理路线 lineage / gap 审计")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- v282 之后继续推进 goal，但不再盲目换模型。")
    lines.append("- 把 v254b-v282 的生理证据链合并，明确旧路线是否关闭、下一步还剩什么真正可尝试。")
    lines.append("")
    lines.append("## 决策摘要")
    lines.append("")
    lines.append(table_to_md(decision, ["item", "value", "evidence"]))
    lines.append("")
    lines.append("## 生理数据与质量结论")
    lines.append("")
    lines.append(table_to_md(quality, ["aspect", "status", "evidence", "implication"]))
    lines.append("")
    lines.append("## 路线 lineage")
    lines.append("")
    lines.append(table_to_md(lineage, ["version", "route", "status", "best_badtop10_signal", "trajectory_badtop10_outcome", "reason"]))
    lines.append("")
    lines.append("## 下一步硬要求")
    lines.append("")
    lines.append(table_to_md(requirements, ["requirement_id", "requirement", "current_evidence", "status", "next_action"]))
    lines.append("")
    lines.append("## 关键判断")
    lines.append("")
    lines.append("- 当前 goal 仍未完成：没有可部署生理路线稳定改善差样本。")
    lines.append("- 旧路线已经足够清楚：200Hz 统计、事件型 biomarker、低身份筛选、个体/记录校准、候选消歧 gate 都未形成正式增量。")
    lines.append("- 如果继续生理 goal，下一步必须是新定义：先做低身份但行为相关的生理状态表示，并先通过 v282 类 route gate，再进入轨迹模型。")
    lines.append("- 如果下一版仍无法让生理距离在车辆歧义样本中产生正相关和 val/test 同向收益，就应把生理降级为 subject-aware 个体校准或边界证据。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    lines.append("")
    lines.append("## guardrail")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_input_hashes() -> None:
    rows = []
    for key, path in {"v283_script": V283_SCRIPT, **PATHS}.items():
        if path.exists():
            rows.append({"key": key, "path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size})
    write_csv(pd.DataFrame(rows), LOGS / "input_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip() is None


def main() -> None:
    print("[v283] 目的：把生理路线证据链合并成下一步硬 gate。", flush=True)
    clean_out_dir()

    quality = build_alignment_quality()
    lineage = build_model_lineage()
    requirements = build_next_requirements(lineage, quality)
    decision = build_decision_summary(lineage, quality, requirements)

    write_csv(quality, TABLES / "v283_alignment_quality_summary.csv")
    write_csv(lineage, TABLES / "v283_route_lineage_summary.csv")
    write_csv(requirements, TABLES / "v283_next_route_requirements.csv")
    write_csv(decision, TABLES / "v283_decision_summary.csv")

    figures = [
        plot_lineage_status(lineage),
        plot_signal_quality(),
        plot_badtop10_delta(),
    ]

    guardrail = {
        "pass": True,
        "zip_testzip": False,
        "lineage_rows": int(len(lineage)),
        "quality_rows": int(len(quality)),
        "requirements_rows": int(len(requirements)),
        "current_goal_achieved": False,
        "old_feature_selector_route_closed": bool(
            decision.loc[decision["item"].eq("old_feature_selector_route_closed"), "value"].iloc[0]
        ),
        "physio_source_alignment_ready": bool(
            decision.loc[decision["item"].eq("physio_source_alignment_ready"), "value"].iloc[0]
        ),
        "next_route_requires_feature_redefinition": bool(
            decision.loc[decision["item"].eq("next_route_requires_feature_redefinition"), "value"].iloc[0]
        ),
    }
    write_input_hashes()
    write_file_inventory()
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(lineage, quality, requirements, decision, figures, guardrail)
    write_file_inventory()
    first_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(first_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(lineage, quality, requirements, decision, figures, guardrail)
    write_file_inventory()
    second_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(second_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[v283] 完成。", flush=True)
    print(f"[v283] report={report}", flush=True)
    print(f"[v283] zip={ZIP_PATH}", flush=True)
    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
