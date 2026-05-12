from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
STAGE6E = ROOT / "06_structured_models" / "stage06e_multicandidate_oracle_gap_v0_1"
OUT_ROOT = ROOT / "07_multihypothesis" / "stage07a_non_oracle_selection_protocol_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = ROOT / "09_reports"


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_stage6e() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gap = pd.read_csv(STAGE6E / "tables" / "multicandidate_oracle_gap_table.csv")
    metrics = pd.read_csv(STAGE6E / "tables" / "multicandidate_all_metrics.csv")
    winners = pd.read_csv(STAGE6E / "tables" / "multicandidate_oracle_winner_summary.csv")
    availability = pd.read_csv(STAGE6E / "tables" / "multicandidate_model_availability.csv")
    return gap, metrics, winners, availability


def build_candidate_pool(metrics: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    use_models = [
        "rbf_kernel_ridge_context_no_subject",
        "ridge_rich_context_no_subject",
        "ridge_rich_history_no_subject",
        "knn_template_context_no_subject",
        "direction_gated_knn_template_no_subject",
        "peak_scaled_template_context_no_subject",
        "keypoint_residual_vehicle_transformer_no_subject",
        "topk_vehicle_transformer_branch0_no_subject",
        "topk_vehicle_transformer_branch1_no_subject",
        "topk_vehicle_transformer_branch2_no_subject",
    ]
    roles = {
        "rbf_kernel_ridge_context_no_subject": "current_primary_reference",
        "ridge_rich_context_no_subject": "low_variance_vehicle_candidate",
        "ridge_rich_history_no_subject": "low_variance_vehicle_candidate",
        "knn_template_context_no_subject": "template_candidate",
        "direction_gated_knn_template_no_subject": "template_candidate",
        "peak_scaled_template_context_no_subject": "template_candidate",
        "keypoint_residual_vehicle_transformer_no_subject": "keypoint_candidate",
        "topk_vehicle_transformer_branch0_no_subject": "topk_branch_candidate",
        "topk_vehicle_transformer_branch1_no_subject": "topk_branch_candidate",
        "topk_vehicle_transformer_branch2_no_subject": "topk_branch_candidate",
    }
    test_gap = metrics[metrics["split"].eq("test") & metrics["model_name"].isin(use_models)].copy()
    rbf = test_gap[test_gap["model_name"].eq("rbf_kernel_ridge_context_no_subject")].iloc[0]
    test_gap["delta_vs_rbf_rmse"] = test_gap["rmse_steer"].astype(float) - float(rbf["rmse_steer"])
    avail_test = availability[availability["split"].eq("test")][["model_name", "n_samples"]].rename(columns={"n_samples": "test_available_samples"})
    pool = test_gap.merge(avail_test, on="model_name", how="left")
    pool["candidate_role"] = pool["model_name"].map(roles)
    pool["stage07_use"] = pool["model_name"].map(
        lambda x: "required_reference" if x == "rbf_kernel_ridge_context_no_subject" else "candidate_for_selection"
    )
    pool["allowed_as_training_target"] = False
    pool["allowed_as_inference_candidate"] = True
    pool["notes_cn"] = pool["model_name"].map(
        {
            "rbf_kernel_ridge_context_no_subject": "当前 RBF/KNN 主参照，所有选择器必须与它公平比较。",
            "keypoint_residual_vehicle_transformer_no_subject": "单模型 RMSE 较差但错侧/大幅响应有信号，作为候选分支而非主线。",
        }
    ).fillna("作为车辆-only候选分支纳入选择池；不单独升级为主线。")
    cols = [
        "model_name",
        "candidate_role",
        "stage07_use",
        "test_available_samples",
        "rmse_steer",
        "delta_vs_rbf_rmse",
        "wrong_side_rate",
        "large_response_recall",
        "difficult_top20_rmse",
        "allowed_as_training_target",
        "allowed_as_inference_candidate",
        "notes_cn",
    ]
    return pool[cols].sort_values(["stage07_use", "candidate_role", "rmse_steer"])


def build_feature_protocol() -> pd.DataFrame:
    rows = [
        ("event_context", "road_module, event_family, window_config_id", "allowed", "事件/道路上下文来自样本 manifest，可用于 train/val/test 推理。"),
        ("vehicle_history_summary", "pre-window speed/yaw/latacc/steer statistics", "allowed", "只能来自事件锚点之前输入窗口；禁止使用标签窗口。"),
        ("candidate_prediction_shape", "candidate peak_abs, peak_time, tail_abs, reversal_count, multi_segment flag", "allowed_with_train_val_fit", "可由候选预测自身计算；选择器训练和阈值只允许看 train/val。"),
        ("candidate_disagreement", "pairwise RMSE-like distance between candidate predictions, peak spread, sign disagreement", "allowed_with_train_val_fit", "不需要真实标签，适合作为不确定性和候选多样性特征。"),
        ("calibration_prior", "train/val historical reliability by road_module/response_family/candidate", "allowed_train_val_only", "只能用 train/val 统计，测试集不能参与可靠性表或标准化。"),
        ("oracle_winner", "best candidate under test label", "forbidden", "只能用于分析上限，禁止作为训练标签以外的测试决策依据。"),
        ("test_sample_rmse", "per-candidate test RMSE or label-based error", "forbidden", "禁止进入选择器输入、阈值选择或可部署决策。"),
        ("physio_eeg_style", "ECG/EDA/EMG/RESP/EEG/style features", "blocked", "车辆-only Stage 7 选择策略未闭环前继续阻塞。"),
        ("subject_id", "driver identity one-hot or direct subject id", "blocked", "当前不允许用驾驶员 ID 解决选择问题。"),
    ]
    return pd.DataFrame(rows, columns=["feature_group", "feature_examples", "status", "rule_cn"])


def build_selection_protocol() -> pd.DataFrame:
    rows = [
        ("S7A-0", "freeze_inputs", "固定 Stage 6e 候选池、B_response3s_strict_core split 和 RBF/KNN 主参照。", "required_before_training"),
        ("S7A-1", "candidate_prediction_export", "为每个候选保存同一 sample_id 的预测轨迹、候选形态特征和候选间差异。", "pending"),
        ("S7A-2", "train_val_selector", "只用 train 拟合选择器，只用 val 选模型/阈值/温度缩放；test 只最终评估一次。", "pending"),
        ("S7A-3", "calibration", "报告选择置信度分桶、ECE/Brier、coverage-risk 曲线和 abstain/fallback 到 RBF 的策略。", "pending"),
        ("S7A-4", "top1_vs_fallback", "同时报告 top-1 selector、RBF fallback、abstain-on-low-confidence 和 oracle upper bound。", "pending"),
        ("S7A-5", "fixed_plots", "固定样本图、bad samples 图、oracle-gap 样本图、selector-regret 样本图必须全部输出。", "pending"),
        ("S7A-6", "promotion_gate", "只有 test RMSE 不劣于 RBF，且至少一个物理指标或困难样本改善，才允许升级。", "pending"),
    ]
    return pd.DataFrame(rows, columns=["step_id", "step_name", "requirement_cn", "status"])


def build_evaluation_plan() -> pd.DataFrame:
    rows = [
        ("primary_fit", "rmse_steer", "must_not_regress_vs_rbf", "不能为了方向/召回牺牲整体轨迹误差。"),
        ("direction", "wrong_side_rate", "prefer_lower_vs_rbf", "错侧率必须单独报告。"),
        ("large_response", "large_response_recall", "prefer_higher_vs_rbf", "大幅响应召回改善不能来自 oracle 标签选择。"),
        ("difficulty", "difficult_top20_rmse", "prefer_lower_vs_rbf", "优先看 RBF 困难样本是否改善。"),
        ("calibration", "ece_or_brier", "required", "若选择器给概率，必须报告可靠性。"),
        ("coverage_risk", "risk_at_coverage", "required", "允许低置信度 fallback/abstain，但要报告覆盖率下误差。"),
        ("candidate_diversity", "oracle_gap_and_winner_entropy", "diagnostic", "检查候选是否真正多样，而不是多个相似平滑预测。"),
        ("leakage", "test_label_used_for_selection", "must_be_false", "任何用 test 标签选候选的结果只能标为 oracle。"),
    ]
    return pd.DataFrame(rows, columns=["metric_group", "metric_name", "gate_rule", "why_cn"])


def build_plot_protocol() -> pd.DataFrame:
    rows = [
        ("fixed_predictions", "同一批固定 test 样本画 RBF、selector top1、fallback、oracle 和 GT。"),
        ("bad_samples", "按 selector sample RMSE 排序画最差样本，不能只挑好看的。"),
        ("oracle_gap_samples", "画 broad oracle gain 最大但 selector 没选中的样本，解释选择失败。"),
        ("confidence_bins", "按 selector 置信度分桶画 RMSE、错侧、大幅召回和覆盖率。"),
        ("winner_counts", "画 oracle winner 分布和 selector chosen 分布，检查候选是否坍缩到单分支。"),
        ("physical_error_buckets", "分错侧、幅值不足、峰时误差、尾段漂移、反向/多段修正画表和图。"),
    ]
    return pd.DataFrame(rows, columns=["plot_name", "requirement_cn"])


def build_gate(gap: pd.DataFrame) -> pd.DataFrame:
    rbf = gap[gap["model_name"].eq("rbf_kernel_ridge_context_no_subject")].iloc[0]
    broad = gap[gap["model_name"].eq("oracle_broad_vehicle_pool")].iloc[0]
    best_selector = gap[gap["role"].eq("deployable_selector_attempt")].sort_values("rmse_steer").iloc[0]
    rows = [
        {
            "gate": "stage07_protocol_ready",
            "status": "ready_for_non_oracle_design",
            "evidence": "候选池、允许特征、禁止信息、评价指标和固定图协议已定义。",
            "decision": "可以进入 Stage 7 非 oracle 选择器设计，但不能直接训练生理/EEG。",
        },
        {
            "gate": "oracle_gap_status",
            "status": "large_oracle_gap",
            "evidence": f"RBF RMSE={rbf['rmse_steer']:.6f}; broad oracle RMSE={broad['rmse_steer']:.6f}; delta={broad['delta_vs_rbf_rmse']:+.6f}.",
            "decision": "只说明多候选上限存在，不能作为可部署结果。",
        },
        {
            "gate": "deployable_selector_status",
            "status": "blocked",
            "evidence": f"best deployable selector RMSE={best_selector['rmse_steer']:.6f}; delta={best_selector['delta_vs_rbf_rmse']:+.6f}.",
            "decision": "必须先解决选择策略和校准。",
        },
        {
            "gate": "stage05_physio_eeg_allowed",
            "status": "blocked",
            "evidence": "车辆-only 多候选选择协议刚建立，尚未完成可部署 selector。",
            "decision": "继续阻塞生理/EEG有效性结论。",
        },
    ]
    return pd.DataFrame(rows)


def plot_protocol_summary(candidate_pool: pd.DataFrame, gate: pd.DataFrame) -> tuple[Path, Path]:
    show = candidate_pool.copy().sort_values("rmse_steer", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(show["model_name"], show["rmse_steer"], color="#64748b")
    rbf = float(show[show["model_name"].eq("rbf_kernel_ridge_context_no_subject")]["rmse_steer"].iloc[0])
    ax.axvline(rbf, color="#111827", linestyle="--", linewidth=1)
    ax.set_xlabel("test RMSE")
    ax.set_title("Stage 7a candidate pool, RBF reference marked")
    fig.tight_layout()
    pool_path = FIG_DIR / "stage07a_candidate_pool_rmse.png"
    fig.savefig(pool_path, dpi=180)
    plt.close(fig)

    status_counts = gate["status"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(status_counts.index, status_counts.values, color="#0f766e")
    ax.set_xlabel("gate count")
    ax.set_title("Stage 7a protocol gates")
    fig.tight_layout()
    gate_path = FIG_DIR / "stage07a_protocol_gate_status.png"
    fig.savefig(gate_path, dpi=180)
    plt.close(fig)
    return pool_path, gate_path


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 12) -> str:
    view = df[cols].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    widths = {col: max(len(str(col)), *(len(str(v)) for v in view[col])) for col in view.columns}
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in view.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in view.columns) + " |"
    rows = ["| " + " | ".join(str(row[col]).ljust(widths[col]) for col in view.columns) + " |" for _, row in view.iterrows()]
    return "\n".join([header, sep, *rows])


def write_reports(candidate_pool: pd.DataFrame, feature_protocol: pd.DataFrame, selection_protocol: pd.DataFrame, gate: pd.DataFrame, pool_fig: Path, gate_fig: Path) -> tuple[Path, Path]:
    user = f"""# Stage 7a 用户查看版：非 oracle 多候选选择协议 v0.1

## 为什么做

Stage 6e 发现 broad oracle 候选池上限很高，但当前可部署 selector 没有超过 RBF/KNN。这个阶段先把 Stage 7 的规则写清楚，防止后面把 best-of-K 或用真实标签选候选的结果误当成模型能力。

## 这个阶段检查了什么

- 固定候选池：RBF/KNN、ridge、template、keypoint 和 top-K 分支。
- 固定禁止信息：test label、test RMSE、oracle winner、测试集统计、生理/EEG/连续风格、驾驶员 ID。
- 固定选择规则：train 拟合选择器、val 选模型/阈值/校准、test 只最终评估。
- 固定评价：RMSE、错侧、大幅响应、困难样本、校准、coverage-risk、固定图和坏样本图。

## 当前发现

当前可以进入 Stage 7 的“协议准备”状态，但还不能说 Stage 7 模型有效。真正需要证明的是：不用真实标签选择候选时，selector 是否能稳定超过 RBF/KNN。

## 下一阶段是否可以继续

可以继续做 Stage 7 非 oracle 选择器设计和轻量实验；但生理/EEG 仍不能进入有效性结论。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_selection_protocol.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_feature_guard_table.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_gate_table.csv`
4. `{str(pool_fig).replace(chr(92), "/")}`
5. `{str(gate_fig).replace(chr(92), "/")}`
"""
    tech = f"""# Stage 7a：非 oracle 多候选选择协议 v0.1

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

本轮不训练模型，不使用生理、EEG、连续风格或被试 ID。目标是把 Stage 7 的可部署选择规则和禁止使用信息固定下来。

## Gate

{md_table(gate, ['gate', 'status', 'evidence', 'decision'], max_rows=10)}

## Candidate Pool

{md_table(candidate_pool, ['model_name', 'candidate_role', 'rmse_steer', 'wrong_side_rate', 'large_response_recall'], max_rows=20)}

## Feature Guard

{md_table(feature_protocol, ['feature_group', 'status', 'rule_cn'], max_rows=20)}

## Selection Steps

{md_table(selection_protocol, ['step_id', 'step_name', 'status', 'requirement_cn'], max_rows=20)}
"""
    user_path = REPORT_DIR / "stage07a_non_oracle_selection_protocol_user_summary_cn.md"
    tech_path = REPORT_DIR / "stage07a_non_oracle_selection_protocol_v0_1_cn.md"
    user_path.write_text(user, encoding="utf-8")
    tech_path.write_text(tech, encoding="utf-8")
    return user_path, tech_path


def main() -> None:
    ensure_dirs()
    gap, metrics, winners, availability = load_stage6e()
    candidate_pool = build_candidate_pool(metrics, availability)
    feature_protocol = build_feature_protocol()
    selection_protocol = build_selection_protocol()
    evaluation_plan = build_evaluation_plan()
    plot_protocol = build_plot_protocol()
    gate = build_gate(gap)
    pool_fig, gate_fig = plot_protocol_summary(candidate_pool, gate)
    user_path, tech_path = write_reports(candidate_pool, feature_protocol, selection_protocol, gate, pool_fig, gate_fig)

    candidate_pool.to_csv(TABLE_DIR / "stage07a_candidate_pool_manifest.csv", index=False, encoding="utf-8-sig")
    feature_protocol.to_csv(TABLE_DIR / "stage07a_feature_guard_table.csv", index=False, encoding="utf-8-sig")
    selection_protocol.to_csv(TABLE_DIR / "stage07a_selection_protocol.csv", index=False, encoding="utf-8-sig")
    evaluation_plan.to_csv(TABLE_DIR / "stage07a_evaluation_plan.csv", index=False, encoding="utf-8-sig")
    plot_protocol.to_csv(TABLE_DIR / "stage07a_fixed_plot_protocol.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "stage07a_gate_table.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_version": "stage07a_non_oracle_selection_protocol_v0_1",
        "run_time_local": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "candidate_count": int(len(candidate_pool)),
        "feature_guard_rows": int(len(feature_protocol)),
        "selection_protocol_rows": int(len(selection_protocol)),
        "stage07_training_started": False,
        "server_used": False,
        "server_credential_file_read": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "uses_subject_id": False,
        "raw_files_modified": False,
        "stage05_physio_eeg_allowed": "blocked",
        "user_summary_path": str(user_path).replace("\\", "/"),
        "technical_report_path": str(tech_path).replace("\\", "/"),
        "gate_path": str(TABLE_DIR / "stage07a_gate_table.csv").replace("\\", "/"),
    }
    (LOG_DIR / "stage07a_protocol_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
