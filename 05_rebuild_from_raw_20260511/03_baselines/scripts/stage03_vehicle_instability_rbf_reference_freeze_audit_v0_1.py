# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
OUT_ROOT = ROOT / "03_baselines" / "stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_ROOT = ROOT / "09_reports"

TRACK_ID = "B_response3s_strict_core"
MAIN_RBF = "rbf_kernel_ridge_context_no_subject"

DECISION_TABLE = (
    ROOT
    / "03_baselines"
    / "stage03_vehicle_instability_vehicle_only_decision_table_v0_2"
    / "tables"
    / "vehicle_only_candidate_decision_table_v0_2.csv"
)
DECISION_GATE_TABLE = (
    ROOT
    / "03_baselines"
    / "stage03_vehicle_instability_vehicle_only_decision_table_v0_2"
    / "tables"
    / "vehicle_only_stage3_gate_status_v0_2.csv"
)
RBF_PER_SAMPLE = (
    ROOT
    / "03_baselines"
    / "stage03_vehicle_instability_topk_reliability_selector_v0_1"
    / "tables"
    / "topk_reliability_selector_per_sample_metrics.csv"
)
BAD_SAMPLE_TABLE = (
    ROOT
    / "03_baselines"
    / "stage03_vehicle_instability_clean_task_bad_sample_review_v0_1"
    / "tables"
    / "b_track_rbf_bad_sample_table.csv"
)
BAD_FAILURE_SUMMARY = (
    ROOT
    / "03_baselines"
    / "stage03_vehicle_instability_clean_task_bad_sample_review_v0_1"
    / "tables"
    / "b_track_rbf_failure_summary.csv"
)
ROBUSTNESS_DECISION = (
    ROOT
    / "03_baselines"
    / "stage03_vehicle_instability_strong_vehicle_robustness_v0_1"
    / "tables"
    / "strong_vehicle_robustness_decision_table.csv"
)


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decision = pd.read_csv(DECISION_TABLE)
    gates = pd.read_csv(DECISION_GATE_TABLE)
    per_sample = pd.read_csv(RBF_PER_SAMPLE)
    bad = pd.read_csv(BAD_SAMPLE_TABLE)
    failure = pd.read_csv(BAD_FAILURE_SUMMARY)
    return decision, gates, per_sample, bad, failure


def rbf_test_rows(per_sample: pd.DataFrame) -> pd.DataFrame:
    rows = per_sample[
        (per_sample["track_id"].astype(str) == TRACK_ID)
        & (per_sample["split"].astype(str) == "test")
        & (per_sample["model_name"].astype(str) == MAIN_RBF)
    ].copy()
    if rows.empty:
        raise RuntimeError("RBF test rows missing")
    return rows


def aggregate_rbf_profile(
    rbf: pd.DataFrame, bad: pd.DataFrame, failure: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    large_rows = rbf[rbf["is_large_response"].astype(int) == 1]
    large_recall = float(large_rows["large_response_recalled"].astype(float).mean()) if len(large_rows) else float("nan")
    difficult_rows = rbf[rbf["is_difficult_peak_top20"].astype(int) == 1]
    difficult_rmse = (
        float(np.sqrt(np.mean(np.square(difficult_rows["sample_rmse"].astype(float))))) if len(difficult_rows) else float("nan")
    )
    metric_rows = [
        {"metric": "n_test_samples", "value": float(len(rbf)), "interpretation_cn": "B 轨道严格响应覆盖 test 样本数。"},
        {
            "metric": "rmse_steer",
            "value": float(np.sqrt(np.mean(np.square(rbf["sample_rmse"].astype(float))))),
            "interpretation_cn": "主参照整体误差。",
        },
        {
            "metric": "wrong_side_rate",
            "value": float(rbf["wrong_side"].astype(float).mean()),
            "interpretation_cn": "主峰方向错侧率，仍偏高。",
        },
        {
            "metric": "large_response_recall",
            "value": large_recall,
            "interpretation_cn": "大幅响应召回，仍有漏召回。",
        },
        {
            "metric": "severe_amp_under_rate",
            "value": float(rbf["severe_amp_under"].astype(float).mean()),
            "interpretation_cn": "严重幅值不足率。",
        },
        {
            "metric": "tail_drift_risk_rate",
            "value": float(rbf["tail_drift_risk"].astype(float).mean()),
            "interpretation_cn": "尾段漂移风险。",
        },
        {
            "metric": "reversal_count_exact_match_rate",
            "value": float(rbf["reversal_count_exact"].astype(float).mean()),
            "interpretation_cn": "反向修正计数完全匹配率，是当前最大物理缺陷。",
        },
        {
            "metric": "difficult_top20_rmse",
            "value": difficult_rmse,
            "interpretation_cn": "困难峰值样本 RMSE。",
        },
    ]
    profile = pd.DataFrame(metric_rows)
    failure_profile = failure.copy()
    top_bad = bad.sort_values("sample_rmse", ascending=False).head(12).copy()
    return profile, failure_profile, top_bad


def build_freeze_gates(
    decision: pd.DataFrame, prior_gates: pd.DataFrame, profile: pd.DataFrame, failure: pd.DataFrame
) -> pd.DataFrame:
    rbf = decision[decision["model_name"] == MAIN_RBF].iloc[0]
    wrong = float(profile.loc[profile["metric"] == "wrong_side_rate", "value"].iloc[0])
    reversal = float(profile.loc[profile["metric"] == "reversal_count_exact_match_rate", "value"].iloc[0])
    large = float(profile.loc[profile["metric"] == "large_response_recall", "value"].iloc[0])
    rmse = float(profile.loc[profile["metric"] == "rmse_steer", "value"].iloc[0])
    rows = [
        {
            "gate_item": "reference_identity_fixed",
            "status": "pass_limited",
            "evidence": f"{MAIN_RBF}; B test RMSE={rmse:.6f}",
            "decision_cn": "固定为 B 轨道后续增量实验的保守车辆-only 主参照。",
        },
        {
            "gate_item": "reference_is_deployable_vehicle_only",
            "status": "pass",
            "evidence": "输入只包含事件前车辆历史和因果可得道路/事件上下文；不使用 subject ID、生理、脑电或连续风格。",
            "decision_cn": "可作为公平车辆-only 对照。",
        },
        {
            "gate_item": "physical_errors_explained",
            "status": "pass_limited",
            "evidence": (
                f"wrong_side={wrong:.3f}; reversal_exact={reversal:.3f}; large_recall={large:.3f}; "
                "failure summary 已覆盖错侧、幅值、尾段、反向修正、启动延迟等。"
            ),
            "decision_cn": "错误类型已被列出并可追溯，但还没有被车辆-only 模型解决。",
        },
        {
            "gate_item": "vehicle_only_problem_solved",
            "status": "fail",
            "evidence": "反向修正计数完全匹配率为 0；错侧率仍为 0.225；top-K fallback 未超过 RBF。",
            "decision_cn": "不能宣称车辆-only 已解决方向盘物理响应预测。",
        },
        {
            "gate_item": "oracle_used_as_performance",
            "status": "fail_if_used",
            "evidence": "best-of-RBF+topK 仅作事后上限。",
            "decision_cn": "后续所有主表必须区分可部署模型与 oracle 上限。",
        },
        {
            "gate_item": "stage04_style_protocol_allowed",
            "status": "conditional_pass",
            "evidence": "主参照身份已固定，但必须携带 RBF 物理缺陷，并用置乱、分被试和物理指标验证风格增量。",
            "decision_cn": "允许进入阶段 4 的协议设计/探索性实验；不得直接宣称连续风格有效。",
        },
        {
            "gate_item": "stage05_physio_eeg_allowed",
            "status": "blocked",
            "evidence": "连续风格验证尚未完成；生理/EEG 仍需在更强车辆+风格参照后验证。",
            "decision_cn": "生理、脑电仍阻塞。",
        },
    ]
    gates = pd.DataFrame(rows)
    gates["main_reference_rmse"] = float(rbf["rmse_steer"])
    gates["prior_gate_snapshot"] = "; ".join(f"{r.gate_item}={r.status}" for r in prior_gates.itertuples(index=False))
    return gates


def build_robustness_snapshot() -> pd.DataFrame:
    if not ROBUSTNESS_DECISION.exists():
        return pd.DataFrame()
    robust = pd.read_csv(ROBUSTNESS_DECISION)
    keep = [
        "robustness_config_id",
        "window_config_id",
        "split_strategy",
        "val_selected_model",
        "val_selected_test_rmse",
        "rbf_test_rmse",
        "knn_test_rmse",
        "knn_train_rmse",
        "knn_memory_risk",
        "interpretation_cn",
    ]
    return robust[[c for c in keep if c in robust.columns]].copy()


def plot_failure_profile(failure: pd.DataFrame, profile: pd.DataFrame) -> dict[str, str]:
    failure_fig = FIG_DIR / "rbf_reference_failure_profile.png"
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    part = failure.sort_values("overall_rate", ascending=True).copy()
    labels = part["flag"].astype(str).str.replace("_flag", "", regex=False)
    ax.barh(labels, part["overall_rate"], color="#d62728", alpha=0.82)
    ax.set_xlabel("Test rate")
    ax.set_title("RBF Reference Failure Profile")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(failure_fig, dpi=180)
    plt.close(fig)

    key_fig = FIG_DIR / "rbf_reference_key_metrics.png"
    metric_order = [
        "rmse_steer",
        "wrong_side_rate",
        "large_response_recall",
        "severe_amp_under_rate",
        "tail_drift_risk_rate",
        "reversal_count_exact_match_rate",
        "difficult_top20_rmse",
    ]
    key = profile[profile["metric"].isin(metric_order)].copy()
    key["metric"] = pd.Categorical(key["metric"], categories=metric_order, ordered=True)
    key = key.sort_values("metric")
    labels = [
        "RMSE",
        "Wrong side",
        "Large recall",
        "Severe under",
        "Tail drift",
        "Reversal exact",
        "Diff top20 RMSE",
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(labels, key["value"], color="#1f77b4")
    ax.set_title("RBF Reference Key Metrics")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(key_fig, dpi=180)
    plt.close(fig)
    return {
        "failure_profile": str(failure_fig).replace("\\", "/"),
        "key_metrics": str(key_fig).replace("\\", "/"),
    }


def write_reports(
    profile: pd.DataFrame,
    failure: pd.DataFrame,
    gates: pd.DataFrame,
    robustness: pd.DataFrame,
    top_bad: pd.DataFrame,
    figures: dict[str, str],
) -> None:
    rmse = profile.loc[profile["metric"] == "rmse_steer", "value"].iloc[0]
    wrong = profile.loc[profile["metric"] == "wrong_side_rate", "value"].iloc[0]
    large = profile.loc[profile["metric"] == "large_response_recall", "value"].iloc[0]
    reversal = profile.loc[profile["metric"] == "reversal_count_exact_match_rate", "value"].iloc[0]
    user = f"""# 阶段 3 用户查看版：RBF 主参照冻结审计 v0.1

## 这个阶段为什么做

上一轮车辆-only 决策表显示，RBF KRR 仍是当前最稳的车辆-only 主参照，但它不是一个已经解决物理响应问题的模型。这一步正式回答：后面能不能固定 RBF 作为“车辆历史和事件信息本身能做到什么程度”的参照，同时避免把它误说成最终强模型。

## 这个阶段检查了什么

- RBF 是否可以作为 B 轨道后续增量实验的固定车辆-only 主参照。
- RBF 的主要失败类型是什么。
- top-K / oracle 上限是否会被误用为实际可部署模型性能。
- 是否可以进入连续风格阶段，是否可以进入生理/EEG 阶段。

## 目前发现了什么

- RBF test RMSE={float(rmse):.6f}，错侧率={float(wrong):.3f}，大幅响应召回={float(large):.3f}。
- 反向修正计数完全匹配率={float(reversal):.3f}，这是当前最大物理缺陷。
- 失败类型中，反向修正计数不匹配覆盖 40/40，错侧 9/40，严重幅值不足 5/40，大幅响应漏召回 2/40。
- 结论是“有限冻结”：RBF 可以固定为 B 轨道后续增量实验的保守主参照，但不能宣称车辆-only 已经解决物理响应问题。

## 哪些结果可信

可信的是：RBF 是车辆-only、无 subject ID、无生理、无脑电、无连续风格的参照；错误类型来自 test 逐样本物理指标和坏样本复查表，不是只看 RMSE。

## 哪些结果还不能下结论

不能说 RBF 是最终强模型，不能说 top-K oracle 是实际性能，也不能说连续风格或生理已经有效。连续风格最多可以进入阶段 4 的协议设计和探索性验证；生理/EEG 仍阻塞。

## 下一阶段是否可以继续

可以进入阶段 4 的连续风格协议设计/探索性实验，但所有比较必须以固定 RBF 主参照为底线，并带置乱、分被试、物理指标和坏样本分析。生理/EEG 不能跳过阶段 4 直接验证。

## 推荐优先查看

1. `{figures['failure_profile']}`
2. `{figures['key_metrics']}`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_freeze_gate_table.csv`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_failure_profile.csv`
"""
    tech = f"""# 阶段 3 技术报告：RBF 主参照冻结审计 v0.1

## 决策

固定 `{MAIN_RBF}` 为 `{TRACK_ID}` 后续增量实验的保守车辆-only 主参照。冻结类型为 `limited_reference_freeze`，不是 `vehicle_problem_solved`。

## RBF profile

{markdown_table(profile)}

## Failure profile

{markdown_table(failure[['flag', 'flag_cn', 'overall_count', 'overall_rate', 'high_rmse_top20_count', 'high_rmse_top20_rate']])}

## Freeze gates

{markdown_table(gates[['gate_item', 'status', 'evidence', 'decision_cn']])}

## Robustness snapshot

{markdown_table(robustness)}

## Top bad samples

{markdown_table(top_bad[['sample_id', 'subject', 'road_design_module_name', 'sample_rmse', 'primary_failure_type', 'failure_tags']].head(8))}

## 后续约束

阶段 4 可以开始“连续风格协议设计/探索性实验”，但不能直接宣称风格有效。所有风格增量必须对比固定 RBF 主参照，并做置乱、分被试、物理指标、困难样本和坏样本图。生理/EEG 仍阻塞。
"""
    (REPORT_ROOT / "stage03_vehicle_instability_rbf_reference_freeze_audit_user_summary_cn.md").write_text(
        user, encoding="utf-8-sig"
    )
    (REPORT_ROOT / "stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1_cn.md").write_text(
        tech, encoding="utf-8-sig"
    )


def main() -> None:
    ensure_dirs()
    decision, prior_gates, per_sample, bad, failure = load_inputs()
    rbf = rbf_test_rows(per_sample)
    profile, failure_profile, top_bad = aggregate_rbf_profile(rbf, bad, failure)
    gates = build_freeze_gates(decision, prior_gates, profile, failure_profile)
    robustness = build_robustness_snapshot()
    figures = plot_failure_profile(failure_profile, profile)

    profile.to_csv(TABLE_DIR / "rbf_reference_metric_profile.csv", index=False, encoding="utf-8-sig")
    failure_profile.to_csv(TABLE_DIR / "rbf_reference_failure_profile.csv", index=False, encoding="utf-8-sig")
    top_bad.to_csv(TABLE_DIR / "rbf_reference_top_bad_samples.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(TABLE_DIR / "rbf_reference_freeze_gate_table.csv", index=False, encoding="utf-8-sig")
    robustness.to_csv(TABLE_DIR / "rbf_reference_robustness_snapshot.csv", index=False, encoding="utf-8-sig")
    write_reports(profile, failure_profile, gates, robustness, top_bad, figures)

    summary = {
        "output_version": "stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1",
        "track_id": TRACK_ID,
        "main_reference_model": MAIN_RBF,
        "freeze_type": "limited_reference_freeze",
        "vehicle_problem_solved": False,
        "stage04_style_protocol_allowed": "conditional_pass",
        "stage05_physio_eeg_allowed": "blocked",
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "gate_table": str(TABLE_DIR / "rbf_reference_freeze_gate_table.csv").replace("\\", "/"),
        "figures": figures,
    }
    (LOG_DIR / "rbf_reference_freeze_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
