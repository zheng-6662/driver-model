# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
OUT_ROOT = ROOT / "03_baselines" / "stage03_vehicle_instability_vehicle_only_decision_table_v0_2"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_ROOT = ROOT / "09_reports"

TRACK_ID = "B_response3s_strict_core"
WINDOW_ID = "pre3_label3_response_coverage"
SPLIT_STRATEGY = "session_level_split"
MAIN_RBF = "rbf_kernel_ridge_context_no_subject"

INPUT_TABLES = [
    (
        ROOT
        / "03_baselines"
        / "stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1"
        / "tables"
        / "clean_task_vehicle_metrics.csv",
        "clean_task_vehicle_baselines_v0_1",
    ),
    (
        ROOT
        / "03_baselines"
        / "stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1"
        / "tables"
        / "clean_task_vehicle_transformer_metrics.csv",
        "clean_task_vehicle_transformer_v0_1",
    ),
    (
        ROOT
        / "03_baselines"
        / "stage03_vehicle_instability_structured_vehicle_transformer_v0_1"
        / "tables"
        / "structured_vehicle_transformer_metrics.csv",
        "structured_vehicle_transformer_v0_1",
    ),
    (
        ROOT
        / "03_baselines"
        / "stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1"
        / "tables"
        / "keypoint_residual_vehicle_transformer_metrics.csv",
        "keypoint_residual_vehicle_transformer_v0_1",
    ),
    (
        ROOT
        / "03_baselines"
        / "stage03_vehicle_instability_rbf_keypoint_selector_v0_1"
        / "tables"
        / "rbf_keypoint_selector_metrics.csv",
        "rbf_keypoint_selector_v0_1",
    ),
    (
        ROOT
        / "03_baselines"
        / "stage03_vehicle_instability_topk_reliability_selector_v0_1"
        / "tables"
        / "topk_reliability_selector_metrics.csv",
        "topk_reliability_selector_v0_1",
    ),
    (
        ROOT
        / "03_baselines"
        / "stage03_vehicle_instability_unified_vehicle_comparison_v0_1"
        / "tables"
        / "unified_vehicle_comparison_metrics_test.csv",
        "legacy_unified_vehicle_comparison_v0_1",
    ),
]

DECISION_MODELS = [
    MAIN_RBF,
    "ridge_rich_context_no_subject",
    "formal_ridge_vehicle_context_no_subject",
    "knn_template_context_no_subject",
    "direction_gated_knn_template_no_subject",
    "vehicle_transformer_context_no_subject",
    "structured_vehicle_transformer_aux_no_subject",
    "keypoint_residual_vehicle_transformer_no_subject",
    "selector_logreg_rbf_keypoint_no_subject",
    "topk_vehicle_transformer_top1_no_subject",
    "topk_vehicle_transformer_branch0_no_subject",
    "topk_top1_rbf_fallback_logreg_no_subject",
    "topk_rbf_branch_logreg_selector_no_subject",
    "topk_branch_logreg_selector_no_subject",
    "topk_vehicle_transformer_best_of_3_oracle",
    "oracle_best_of_rbf_keypoint_upper_bound",
    "oracle_best_of_rbf_plus_topk_upper_bound",
]

SOURCE_PREFERENCE = {
    MAIN_RBF: "topk_reliability_selector_v0_1",
    "ridge_rich_context_no_subject": "clean_task_vehicle_transformer_v0_1",
    "formal_ridge_vehicle_context_no_subject": "clean_task_vehicle_transformer_v0_1",
    "knn_template_context_no_subject": "clean_task_vehicle_transformer_v0_1",
    "direction_gated_knn_template_no_subject": "clean_task_vehicle_transformer_v0_1",
    "vehicle_transformer_context_no_subject": "clean_task_vehicle_transformer_v0_1",
    "structured_vehicle_transformer_aux_no_subject": "structured_vehicle_transformer_v0_1",
    "keypoint_residual_vehicle_transformer_no_subject": "keypoint_residual_vehicle_transformer_v0_1",
    "selector_logreg_rbf_keypoint_no_subject": "rbf_keypoint_selector_v0_1",
    "topk_vehicle_transformer_top1_no_subject": "topk_reliability_selector_v0_1",
    "topk_vehicle_transformer_branch0_no_subject": "topk_reliability_selector_v0_1",
    "topk_top1_rbf_fallback_logreg_no_subject": "topk_reliability_selector_v0_1",
    "topk_rbf_branch_logreg_selector_no_subject": "topk_reliability_selector_v0_1",
    "topk_branch_logreg_selector_no_subject": "topk_reliability_selector_v0_1",
    "topk_vehicle_transformer_best_of_3_oracle": "topk_reliability_selector_v0_1",
    "oracle_best_of_rbf_keypoint_upper_bound": "rbf_keypoint_selector_v0_1",
    "oracle_best_of_rbf_plus_topk_upper_bound": "topk_reliability_selector_v0_1",
}

DISPLAY_NAMES = {
    MAIN_RBF: "RBF KRR context",
    "ridge_rich_context_no_subject": "rich ridge context",
    "formal_ridge_vehicle_context_no_subject": "formal ridge",
    "knn_template_context_no_subject": "KNN template",
    "direction_gated_knn_template_no_subject": "direction-gated KNN",
    "vehicle_transformer_context_no_subject": "direct Transformer",
    "structured_vehicle_transformer_aux_no_subject": "structured Transformer",
    "keypoint_residual_vehicle_transformer_no_subject": "keypoint + residual Transformer",
    "selector_logreg_rbf_keypoint_no_subject": "RBF/keypoint selector",
    "topk_vehicle_transformer_top1_no_subject": "top-K top1",
    "topk_vehicle_transformer_branch0_no_subject": "top-K branch0",
    "topk_top1_rbf_fallback_logreg_no_subject": "top1-RBF fallback",
    "topk_rbf_branch_logreg_selector_no_subject": "RBF/topK branch selector",
    "topk_branch_logreg_selector_no_subject": "topK branch selector",
    "topk_vehicle_transformer_best_of_3_oracle": "top-K best-of-3 oracle",
    "oracle_best_of_rbf_keypoint_upper_bound": "RBF/keypoint oracle",
    "oracle_best_of_rbf_plus_topk_upper_bound": "RBF+topK oracle",
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def read_metric_inventory() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path, source in INPUT_TABLES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["source_artifact"] = source
        frames.append(df)
    if not frames:
        raise RuntimeError("no metric tables found")
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "track_id" not in out.columns:
        out["track_id"] = ""
    out["track_id"] = out["track_id"].fillna("").astype(str)
    out["window_config_id"] = out["window_config_id"].fillna("").astype(str)
    out["split_strategy"] = out["split_strategy"].fillna("").astype(str)
    out["split"] = out["split"].fillna("").astype(str)
    return out


def select_decision_rows(inventory: pd.DataFrame) -> pd.DataFrame:
    b_test = inventory[
        (inventory["split"] == "test")
        & (inventory["split_strategy"] == SPLIT_STRATEGY)
        & (inventory["window_config_id"] == WINDOW_ID)
        & ((inventory["track_id"] == TRACK_ID) | (inventory["track_id"] == ""))
    ].copy()
    rows: list[pd.Series] = []
    for model_name in DECISION_MODELS:
        source = SOURCE_PREFERENCE.get(model_name)
        part = b_test[b_test["model_name"] == model_name].copy()
        if source is not None:
            preferred = part[part["source_artifact"] == source]
            if not preferred.empty:
                part = preferred
        if part.empty:
            continue
        rows.append(part.iloc[0])
    if not rows:
        raise RuntimeError("no decision rows selected")
    decision = pd.DataFrame(rows).reset_index(drop=True)
    return decision


def status_for_model(row: pd.Series, rbf_rmse: float) -> tuple[str, str, str, str]:
    model = str(row["model_name"])
    rmse = float(row["rmse_steer"])
    wrong = float(row["wrong_side_rate"])
    large = float(row["large_response_recall"])
    reversal = float(row["reversal_count_exact_match_rate"])
    diff_rmse = rmse - rbf_rmse
    if model == MAIN_RBF:
        return (
            "current_main_reference",
            "deployable_vehicle_only",
            "暂定车辆-only 主参照；按 val/test 规则可部署，但仍存在错侧、反向修正和多段修正问题。",
            "not_fully_frozen",
        )
    if model.startswith("oracle_") or "oracle" in model:
        return (
            "oracle_upper_bound_only",
            "not_deployable",
            "只说明候选池上限，不能作为真实模型性能或阶段通过依据。",
            "blocked",
        )
    if model == "topk_vehicle_transformer_best_of_3_oracle":
        return (
            "oracle_upper_bound_only",
            "not_deployable",
            "best-of-3 是事后选择上限，不能作为可部署 top-K 结果。",
            "blocked",
        )
    if model == "knn_template_context_no_subject":
        return (
            "diagnostic_template",
            "memory_risk",
            "模板方法在多处训练集近零误差，存在记忆风险；只作诊断对照。",
            "blocked",
        )
    if model == "direction_gated_knn_template_no_subject":
        return (
            "diagnostic_template",
            "memory_risk",
            "方向门控模板虽有物理指标线索，但同样存在模板记忆风险，不能冻结主线。",
            "blocked",
        )
    if model == "formal_ridge_vehicle_context_no_subject":
        return (
            "formal_baseline",
            "deployable_vehicle_only",
            "线性公平参照；性能低于 RBF，只用于最低可解释基线。",
            "supporting",
        )
    if model == "ridge_rich_context_no_subject":
        return (
            "strong_linear_baseline",
            "deployable_vehicle_only",
            "RMSE 接近 RBF 但大幅召回较差，保留为强线性参照，不替代 RBF。",
            "supporting",
        )
    if model == "keypoint_residual_vehicle_transformer_no_subject":
        return (
            "structured_candidate",
            "deployable_vehicle_only",
            "错侧率和大幅召回优于 RBF，但 RMSE、困难样本和启动延迟更差；适合作为多候选分支，不单独升级。",
            "weak_candidate",
        )
    if model == "selector_logreg_rbf_keypoint_no_subject":
        return (
            "selector_candidate",
            "deployable_vehicle_only",
            "物理指标有改善但整体 RMSE 未超过 RBF；保留为选择机制线索。",
            "weak_candidate",
        )
    if model == "topk_top1_rbf_fallback_logreg_no_subject":
        return (
            "reliability_selector_no_go",
            "deployable_vehicle_only",
            "val 选择的回退策略 test 仍差于 RBF，且 39/40 回退到 RBF；不能升级。",
            "no_go",
        )
    if model.startswith("topk_") or model.startswith("top-K"):
        return (
            "topk_candidate_no_go",
            "deployable_vehicle_only",
            "top-K 候选覆盖有潜力，但当前选择头/分支选择未超过 RBF。",
            "no_go",
        )
    if "transformer" in model:
        if diff_rmse > 0:
            return (
                "deep_vehicle_candidate_no_go",
                "deployable_vehicle_only",
                "真实 Transformer 已补跑，但 test RMSE 或关键物理指标不如 RBF。",
                "no_go",
            )
    if wrong < 0.20 and large >= 0.75 and diff_rmse <= 0.02 and reversal <= 0.05:
        return (
            "physical_metric_candidate",
            "deployable_vehicle_only",
            "部分物理指标有优势，但反向/多段修正仍弱，需要作为辅助参照。",
            "weak_candidate",
        )
    return (
        "supporting_comparator",
        "deployable_vehicle_only",
        "保留为车辆-only 对照，不作为当前主参照。",
        "supporting",
    )


def add_decisions(decision: pd.DataFrame) -> pd.DataFrame:
    rbf = decision[decision["model_name"] == MAIN_RBF]
    if rbf.empty:
        raise RuntimeError("main RBF row missing")
    rbf_rmse = float(rbf.iloc[0]["rmse_steer"])
    for col in [
        "rmse_steer",
        "wrong_side_rate",
        "large_response_recall",
        "difficult_top20_rmse",
        "reversal_count_exact_match_rate",
        "tail_drift_risk_rate",
    ]:
        decision[col] = pd.to_numeric(decision[col], errors="coerce")
    decision["display_name"] = decision["model_name"].map(DISPLAY_NAMES).fillna(decision["model_name"])
    decision["delta_rmse_vs_rbf"] = decision["rmse_steer"] - rbf_rmse
    decision["rmse_pct_vs_rbf"] = 100.0 * decision["delta_rmse_vs_rbf"] / max(rbf_rmse, 1e-9)
    statuses = decision.apply(lambda row: status_for_model(row, rbf_rmse), axis=1)
    decision["decision_role"] = [s[0] for s in statuses]
    decision["deployability"] = [s[1] for s in statuses]
    decision["decision_reason_cn"] = [s[2] for s in statuses]
    decision["gate_status"] = [s[3] for s in statuses]
    decision["rank_rmse"] = decision["rmse_steer"].rank(method="min", ascending=True).astype(int)
    decision["rank_wrong_side"] = decision["wrong_side_rate"].rank(method="min", ascending=True).astype(int)
    decision["rank_large_recall"] = decision["large_response_recall"].rank(method="min", ascending=False).astype(int)
    return decision.sort_values(["deployability", "gate_status", "rmse_steer"], ascending=[True, True, True]).reset_index(drop=True)


def build_gate_table(decision: pd.DataFrame) -> pd.DataFrame:
    rbf = decision[decision["model_name"] == MAIN_RBF].iloc[0]
    fallback = decision[decision["model_name"] == "topk_top1_rbf_fallback_logreg_no_subject"].iloc[0]
    oracle = decision[decision["model_name"] == "oracle_best_of_rbf_plus_topk_upper_bound"].iloc[0]
    rows = [
        {
            "gate_item": "vehicle_main_reference_available",
            "status": "partial",
            "evidence": f"RBF test RMSE={float(rbf['rmse_steer']):.6f}, wrong_side={float(rbf['wrong_side_rate']):.3f}, large_recall={float(rbf['large_response_recall']):.3f}",
            "decision_cn": "可以作为当前主参照，但不能说物理问题已解决。",
        },
        {
            "gate_item": "strong_vehicle_baseline_frozen",
            "status": "no",
            "evidence": "RBF 反向修正完全匹配仍为 0，错侧率 0.225；top-K fallback 未超过 RBF。",
            "decision_cn": "阶段 3 仍未完全冻结，进入风格/生理前需明确接受 RBF 作为保守主参照或继续车辆-only 结构。",
        },
        {
            "gate_item": "topk_reliability_selector_upgrade",
            "status": "no",
            "evidence": f"fallback test RMSE={float(fallback['rmse_steer']):.6f} > RBF {float(rbf['rmse_steer']):.6f}",
            "decision_cn": "本轮选择器 no-go。",
        },
        {
            "gate_item": "oracle_upper_bound_interpretable",
            "status": "yes_but_not_deployable",
            "evidence": f"best-of-RBF+topK oracle RMSE={float(oracle['rmse_steer']):.6f}",
            "decision_cn": "只说明候选池还有上限空间，不能作为模型效果。",
        },
        {
            "gate_item": "style_physio_eeg_allowed_now",
            "status": "no",
            "evidence": "强车辆基线/主参照冻结仍未闭环。",
            "decision_cn": "继续阻塞连续风格、生理和 EEG 增量结论。",
        },
    ]
    return pd.DataFrame(rows)


def summarize_by_role(decision: pd.DataFrame) -> pd.DataFrame:
    return (
        decision.groupby(["decision_role", "gate_status"], dropna=False)
        .agg(
            n_models=("model_name", "count"),
            best_rmse=("rmse_steer", "min"),
            best_wrong_side=("wrong_side_rate", "min"),
            best_large_recall=("large_response_recall", "max"),
        )
        .reset_index()
        .sort_values(["gate_status", "best_rmse"])
    )


def plot_metrics(decision: pd.DataFrame) -> dict[str, str]:
    deploy = decision[decision["deployability"] == "deployable_vehicle_only"].copy()
    deploy = deploy.sort_values("rmse_steer").head(12)
    colors = deploy["gate_status"].map(
        {
            "not_fully_frozen": "#1f77b4",
            "supporting": "#7f7f7f",
            "weak_candidate": "#ff7f0e",
            "no_go": "#d62728",
            "blocked": "#9467bd",
        }
    ).fillna("#999999")
    x = np.arange(len(deploy))
    metric_fig = FIG_DIR / "vehicle_only_decision_key_metrics_test.png"
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.6))
    metrics = [
        ("rmse_steer", "RMSE"),
        ("wrong_side_rate", "Wrong side"),
        ("large_response_recall", "Large recall"),
        ("difficult_top20_rmse", "Difficult RMSE"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(x, deploy[col], color=colors)
        ax.set_title(title)
        ax.set_xticks(x, deploy["display_name"], rotation=40, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(metric_fig, dpi=180)
    plt.close(fig)

    scatter_fig = FIG_DIR / "vehicle_only_decision_rmse_vs_wrong_side_test.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for status, grp in decision.groupby("gate_status"):
        ax.scatter(grp["rmse_steer"], grp["wrong_side_rate"], s=70, label=status, alpha=0.85)
        for _, row in grp.iterrows():
            if row["model_name"] in [MAIN_RBF, "keypoint_residual_vehicle_transformer_no_subject", "topk_top1_rbf_fallback_logreg_no_subject", "oracle_best_of_rbf_plus_topk_upper_bound"]:
                ax.annotate(str(row["display_name"]), (row["rmse_steer"], row["wrong_side_rate"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Test RMSE")
    ax.set_ylabel("Wrong-side rate")
    ax.set_title("Vehicle-only decision map")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(scatter_fig, dpi=180)
    plt.close(fig)

    role_fig = FIG_DIR / "vehicle_only_decision_role_counts.png"
    role_counts = decision["gate_status"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(role_counts.index, role_counts.values, color="#4c78a8")
    ax.set_ylabel("models")
    ax.set_title("Decision status counts")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(role_fig, dpi=180)
    plt.close(fig)

    return {
        "metric_summary": str(metric_fig).replace("\\", "/"),
        "rmse_wrong_side": str(scatter_fig).replace("\\", "/"),
        "role_counts": str(role_fig).replace("\\", "/"),
    }


def fmt(x: Any) -> str:
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无。"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{float(x):.6f}" if pd.notna(x) else "")
        else:
            display[col] = display[col].astype(str)
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def write_reports(decision: pd.DataFrame, gate: pd.DataFrame, figures: dict[str, str]) -> None:
    rbf = decision[decision["model_name"] == MAIN_RBF].iloc[0]
    fallback = decision[decision["model_name"] == "topk_top1_rbf_fallback_logreg_no_subject"].iloc[0]
    keypoint = decision[decision["model_name"] == "keypoint_residual_vehicle_transformer_no_subject"].iloc[0]
    oracle = decision[decision["model_name"] == "oracle_best_of_rbf_plus_topk_upper_bound"].iloc[0]
    user = f"""# 阶段 3 用户查看版：车辆-only 主参照决策表 v0.2

## 这个阶段为什么做

前面已经补跑了 RBF/KNN、direct Transformer、结构化 Transformer、keypoint+residual、RBF/keypoint selector、top-K 和 top-K 可靠性回退。单看某一次结果容易误把 oracle 上限或弱候选当成主线，所以这一步把阶段 3 的车辆-only 结果放到一张决策表里。

## 这个阶段检查了什么

- 哪个结果可以作为当前车辆-only 主参照。
- 哪些模型只是历史/诊断/弱候选。
- 哪些模型是 no-go。
- 哪些只是事后 oracle 上限，不能作为可部署结果。
- 是否已经允许进入连续风格、生理或 EEG 增量验证。

## 目前发现了什么

- 当前主参照仍是 RBF KRR：test RMSE={fmt(rbf['rmse_steer'])}，错侧率={float(rbf['wrong_side_rate']):.3f}，大幅响应召回={float(rbf['large_response_recall']):.3f}。
- keypoint+residual 错侧率更低、大幅召回更高，但 test RMSE={fmt(keypoint['rmse_steer'])}，困难样本 RMSE={fmt(keypoint['difficult_top20_rmse'])}，不能单独替代 RBF。
- top-K fallback test RMSE={fmt(fallback['rmse_steer'])}，没有超过 RBF，所以可靠性选择 v0.1 是 no-go。
- best-of-RBF+topK oracle RMSE={fmt(oracle['rmse_steer'])}，说明候选池有潜力，但这是事后上限，不能作为真实模型表现。

## 哪些结果可信

可信的是：这些对照都限制在车辆-only 输入，不使用生理、脑电、连续风格或 subject ID；选择器和阈值均按 train/val/test 协议处理，test 只报告。

## 哪些结果还不能下结论

不能说 top-K 已经解决问题，不能说 keypoint 结构已经优于 RBF，也不能因为 oracle 上限好就进入生理或风格结论。当前也不能说强车辆基线已经完全冻结，因为 RBF 的错侧、反向修正和多段修正问题仍未闭环。

## 下一阶段是否可以继续

可以继续阶段 3。当前建议有两条：要么把 RBF KRR 接受为保守主参照并写清物理缺陷，进入阶段 4 前再做一次冻结审查；要么继续做更强的车辆-only 分响应类型/关键点条件多假设。连续风格、生理、EEG 增量验证仍然阻塞。

## 推荐优先查看

1. `{figures['metric_summary']}`
2. `{figures['rmse_wrong_side']}`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/tables/vehicle_only_candidate_decision_table_v0_2.csv`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/tables/vehicle_only_stage3_gate_status_v0_2.csv`
"""
    tech = f"""# 阶段 3 技术报告：车辆-only 主参照决策表 v0.2

## 输入产物

本报告只读取既有阶段 3 指标表，不重新训练模型。输入包括 clean-task baselines、direct Transformer、structured Transformer、keypoint+residual、RBF/keypoint selector、top-K reliability selector 和旧 unified comparison。

## 决策结论

- 当前车辆-only 主参照：`{MAIN_RBF}`。
- 主参照状态：`not_fully_frozen`。
- 阻塞项：RBF 错侧率仍为 {float(rbf['wrong_side_rate']):.3f}，反向修正完全匹配率为 {float(rbf['reversal_count_exact_match_rate']):.3f}；top-K fallback 未超过 RBF。
- 风格/生理/EEG 入口：仍阻塞。

## Gate 表

{markdown_table(gate)}

## 关键模型 test 指标

| 模型 | 角色 | RMSE | 错侧率 | 大幅召回 | 困难 top20 RMSE | 决策 |
|---|---|---:|---:|---:|---:|---|
| RBF KRR | main reference | {fmt(rbf['rmse_steer'])} | {float(rbf['wrong_side_rate']):.3f} | {float(rbf['large_response_recall']):.3f} | {fmt(rbf['difficult_top20_rmse'])} | 暂定主参照 |
| keypoint+residual | weak candidate | {fmt(keypoint['rmse_steer'])} | {float(keypoint['wrong_side_rate']):.3f} | {float(keypoint['large_response_recall']):.3f} | {fmt(keypoint['difficult_top20_rmse'])} | 分支候选 |
| topK fallback | no-go | {fmt(fallback['rmse_steer'])} | {float(fallback['wrong_side_rate']):.3f} | {float(fallback['large_response_recall']):.3f} | {fmt(fallback['difficult_top20_rmse'])} | 不升级 |
| RBF+topK oracle | oracle | {fmt(oracle['rmse_steer'])} | {float(oracle['wrong_side_rate']):.3f} | {float(oracle['large_response_recall']):.3f} | {fmt(oracle['difficult_top20_rmse'])} | 上限诊断 |

## 解释

本轮决策不是为了宣布阶段 3 完成，而是为了避免后续误用阶段 3 产物。RBF KRR 是当前最稳的车辆-only 主参照，但仍不是“物理响应已经解决”的强结论。keypoint 和 top-K 的价值主要体现为候选池/结构线索，尚未形成可部署增益。
"""
    (REPORT_ROOT / "stage03_vehicle_instability_vehicle_only_decision_table_user_summary_cn.md").write_text(user, encoding="utf-8")
    (REPORT_ROOT / "stage03_vehicle_instability_vehicle_only_decision_table_v0_2_cn.md").write_text(tech, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    inventory = read_metric_inventory()
    decision = add_decisions(select_decision_rows(inventory))
    gate = build_gate_table(decision)
    role_summary = summarize_by_role(decision)
    figures = plot_metrics(decision)

    inventory.to_csv(TABLE_DIR / "vehicle_only_stage3_metric_inventory_v0_2.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(TABLE_DIR / "vehicle_only_candidate_decision_table_v0_2.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "vehicle_only_stage3_gate_status_v0_2.csv", index=False, encoding="utf-8-sig")
    role_summary.to_csv(TABLE_DIR / "vehicle_only_decision_role_summary_v0_2.csv", index=False, encoding="utf-8-sig")
    write_reports(decision, gate, figures)

    rbf = decision[decision["model_name"] == MAIN_RBF].iloc[0]
    summary = {
        "output_version": "stage03_vehicle_instability_vehicle_only_decision_table_v0_2",
        "track_id": TRACK_ID,
        "main_reference_model": MAIN_RBF,
        "main_reference_rmse": float(rbf["rmse_steer"]),
        "main_reference_status": "not_fully_frozen",
        "style_physio_eeg_allowed_now": False,
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "decision_table": str(TABLE_DIR / "vehicle_only_candidate_decision_table_v0_2.csv").replace("\\", "/"),
        "gate_table": str(TABLE_DIR / "vehicle_only_stage3_gate_status_v0_2.csv").replace("\\", "/"),
        "figures": figures,
    }
    (LOG_DIR / "vehicle_only_decision_table_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
