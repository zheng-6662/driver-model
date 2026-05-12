from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = PROJECT_ROOT / "03_baselines"
REPORT_ROOT = PROJECT_ROOT / "09_reports"
OUTPUT_ROOT = BASELINE_ROOT / "stage03_vehicle_instability_unified_vehicle_comparison_v0_1"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIG_DIR = OUTPUT_ROOT / "figures"
LOG_DIR = OUTPUT_ROOT / "logs"

WINDOW_CONFIG = "pre2_label2_old_main"
SPLIT_STRATEGY = "session_level_split"

FORMAL_METRICS = (
    BASELINE_ROOT
    / "stage03_vehicle_instability_formal_baselines_v0_1"
    / "tables"
    / "formal_baseline_metrics.csv"
)
FORMAL_PER_SAMPLE = (
    BASELINE_ROOT
    / "stage03_vehicle_instability_formal_baselines_v0_1"
    / "tables"
    / "formal_baseline_per_sample_metrics.csv"
)
OLD_METRICS = (
    BASELINE_ROOT
    / "oldcode_vehicle_direct_full_clean_on_instability_v0_1"
    / "tables"
    / "oldcode_vehicle_direct_full_metrics.csv"
)
OLD_PER_SAMPLE = (
    BASELINE_ROOT
    / "oldcode_vehicle_direct_full_clean_on_instability_v0_1"
    / "tables"
    / "oldcode_vehicle_direct_full_per_sample_metrics.csv"
)
STRONG_METRICS = (
    BASELINE_ROOT
    / "stage03_vehicle_instability_strong_vehicle_baselines_v0_1"
    / "tables"
    / "strong_vehicle_baseline_metrics.csv"
)
STRONG_PER_SAMPLE = (
    BASELINE_ROOT
    / "stage03_vehicle_instability_strong_vehicle_baselines_v0_1"
    / "tables"
    / "strong_vehicle_baseline_per_sample_metrics.csv"
)
TRANSFORMER_METRICS = (
    BASELINE_ROOT
    / "stage03_vehicle_instability_vehicle_transformer_v0_1"
    / "tables"
    / "vehicle_transformer_metrics.csv"
)
TRANSFORMER_PER_SAMPLE = (
    BASELINE_ROOT
    / "stage03_vehicle_instability_vehicle_transformer_v0_1"
    / "tables"
    / "vehicle_transformer_per_sample_metrics.csv"
)


MODEL_ORDER = [
    "zero_response_hold_current",
    "history_trend_500ms",
    "train_mean_all",
    "train_mean_by_event_type",
    "ridge_vehicle_history_no_subject",
    "formal_ridge_vehicle_context_no_subject",
    "active_legacy_best",
    "structure_best",
    "ridge_rich_history_no_subject",
    "ridge_rich_context_no_subject",
    "rbf_kernel_ridge_context_no_subject",
    "knn_template_context_no_subject",
    "direction_gated_knn_template_no_subject",
    "peak_scaled_template_context_no_subject",
    "vehicle_transformer_context_no_subject",
]

DISPLAY_NAME = {
    "zero_response_hold_current": "zero hold",
    "history_trend_500ms": "history trend",
    "train_mean_all": "train mean",
    "train_mean_by_event_type": "event mean",
    "ridge_vehicle_history_no_subject": "ridge history",
    "formal_ridge_vehicle_context_no_subject": "formal ridge",
    "active_legacy_best": "old vehicle_direct active",
    "structure_best": "old vehicle_direct structure",
    "ridge_rich_history_no_subject": "rich ridge history",
    "ridge_rich_context_no_subject": "rich ridge context",
    "rbf_kernel_ridge_context_no_subject": "RBF KRR",
    "knn_template_context_no_subject": "KNN template",
    "direction_gated_knn_template_no_subject": "direction-gated KNN",
    "peak_scaled_template_context_no_subject": "peak-scaled template",
    "vehicle_transformer_context_no_subject": "vehicle Transformer",
}

SOURCE_GROUP = {
    "zero_response_hold_current": "no_learning",
    "history_trend_500ms": "no_learning",
    "train_mean_all": "no_learning",
    "train_mean_by_event_type": "no_learning",
    "ridge_vehicle_history_no_subject": "formal_linear",
    "formal_ridge_vehicle_context_no_subject": "formal_linear",
    "active_legacy_best": "old_vehicle_direct_clean",
    "structure_best": "old_vehicle_direct_clean",
    "ridge_rich_history_no_subject": "strong_vehicle_diagnostic",
    "ridge_rich_context_no_subject": "strong_vehicle_diagnostic",
    "rbf_kernel_ridge_context_no_subject": "strong_vehicle_diagnostic",
    "knn_template_context_no_subject": "template_memory_risk",
    "direction_gated_knn_template_no_subject": "template_memory_risk",
    "peak_scaled_template_context_no_subject": "template_memory_risk",
    "vehicle_transformer_context_no_subject": "true_vehicle_transformer",
}

DECISION_NOTE = {
    "zero_response_hold_current": "无学习下界，只作参照。",
    "history_trend_500ms": "无学习趋势外推，尾段和错侧风险高，只作参照。",
    "train_mean_all": "训练集平均轨迹，证明平均化响应的下界。",
    "train_mean_by_event_type": "按事件类型平均，仍不能代表个体响应。",
    "ridge_vehicle_history_no_subject": "浅层车辆历史基线，保留为线性参照。",
    "formal_ridge_vehicle_context_no_subject": "正式 shallow vehicle baseline，是当前所有车辆模型的最低公平参照。",
    "active_legacy_best": "旧代码 clean vehicle-only 历史对照，使用旧架构，不作为新流程主线。",
    "structure_best": "旧代码结构 checkpoint 历史对照，可看物理指标但不继承旧流程假设。",
    "ridge_rich_history_no_subject": "丰富车辆历史线性模型，没有超过强候选，不作为主线。",
    "ridge_rich_context_no_subject": "丰富车辆上下文线性模型，方向指标尚可但 RMSE 不优。",
    "rbf_kernel_ridge_context_no_subject": "非参数强候选，RMSE 和大幅召回好，但反向修正匹配很差，需稳健性验证。",
    "knn_template_context_no_subject": "test RMSE 最低，但 train RMSE 近 0，模板记忆风险高，暂作诊断上限。",
    "direction_gated_knn_template_no_subject": "KNN 变体，仍有模板记忆风险，暂作诊断。",
    "peak_scaled_template_context_no_subject": "模板缩放候选，幅值指标好但仍需检查物理错误。",
    "vehicle_transformer_context_no_subject": "真正车辆-only Transformer，强于 formal ridge，但多段修正预测为 0，需继续结构化改进。",
}

LOWER_BETTER = [
    "rmse_steer",
    "wrong_side_rate",
    "peak_amp_mae",
    "severe_amp_under_rate",
    "peak_time_mae_s",
    "onset_delay_mae_s",
    "tail_abs_error_mean",
    "tail_drift_risk_rate",
    "zero_crossing_mismatch_rate",
    "difficult_top20_rmse",
    "multi_segment_rate_abs_gap",
]
HIGHER_BETTER = [
    "peak_direction_accuracy",
    "large_response_recall",
    "reversal_count_exact_match_rate",
]
KEY_METRICS = [
    "rmse_steer",
    "wrong_side_rate",
    "large_response_recall",
    "severe_amp_under_rate",
    "tail_abs_error_mean",
    "reversal_count_exact_match_rate",
    "difficult_top20_rmse",
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalize_old_metrics(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "sample_window_config_id": "window_config_id",
        "sample_split_strategy": "split_strategy",
        "sample_split": "split",
        "sample_model_name": "model_name",
        "sample_n_samples": "n_samples",
        "sample_rmse_steer": "rmse_steer",
        "sample_peak_direction_accuracy": "peak_direction_accuracy",
        "sample_wrong_side_rate": "wrong_side_rate",
        "sample_large_response_recall": "large_response_recall",
        "sample_peak_amp_mae": "peak_amp_mae",
        "sample_peak_amp_ratio_pred_over_gt_mean": "peak_amp_ratio_pred_over_gt_mean",
        "sample_severe_amp_under_rate": "severe_amp_under_rate",
        "sample_peak_time_mae_s": "peak_time_mae_s",
        "sample_onset_delay_mae_s": "onset_delay_mae_s",
        "sample_tail_abs_error_mean": "tail_abs_error_mean",
        "sample_tail_drift_risk_rate": "tail_drift_risk_rate",
        "sample_zero_crossing_mismatch_rate": "zero_crossing_mismatch_rate",
        "sample_reversal_count_exact_match_rate": "reversal_count_exact_match_rate",
        "sample_multi_segment_gt_rate": "multi_segment_gt_rate",
        "sample_multi_segment_pred_rate": "multi_segment_pred_rate",
        "sample_difficult_top20_rmse": "difficult_top20_rmse",
    }
    out = df[list(mapping)].rename(columns=mapping)
    out["source_artifact"] = "oldcode_vehicle_direct_full_clean_on_instability_v0_1"
    return out


def filter_main(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    mask = (
        (df["window_config_id"] == WINDOW_CONFIG)
        & (df["split_strategy"] == SPLIT_STRATEGY)
        & df["model_name"].isin(models)
    )
    return df.loc[mask].copy()


def add_common_columns(df: pd.DataFrame, source_artifact: str) -> pd.DataFrame:
    out = df.copy()
    out["source_artifact"] = source_artifact
    return out


def build_unified_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    formal = add_common_columns(
        read_csv(FORMAL_METRICS),
        "stage03_vehicle_instability_formal_baselines_v0_1",
    )
    old = normalize_old_metrics(read_csv(OLD_METRICS))
    strong = add_common_columns(
        read_csv(STRONG_METRICS),
        "stage03_vehicle_instability_strong_vehicle_baselines_v0_1",
    )
    transformer = add_common_columns(
        read_csv(TRANSFORMER_METRICS),
        "stage03_vehicle_instability_vehicle_transformer_v0_1",
    )

    formal_models = [
        "zero_response_hold_current",
        "history_trend_500ms",
        "train_mean_all",
        "train_mean_by_event_type",
        "ridge_vehicle_history_no_subject",
        "ridge_vehicle_context_no_subject",
    ]
    formal_main = filter_main(formal, formal_models)
    formal_main["model_name"] = formal_main["model_name"].replace(
        {"ridge_vehicle_context_no_subject": "formal_ridge_vehicle_context_no_subject"}
    )

    old_main = filter_main(old, ["active_legacy_best", "structure_best"])
    strong_main = filter_main(
        strong,
        [
            "ridge_rich_history_no_subject",
            "ridge_rich_context_no_subject",
            "rbf_kernel_ridge_context_no_subject",
            "knn_template_context_no_subject",
            "direction_gated_knn_template_no_subject",
            "peak_scaled_template_context_no_subject",
        ],
    )
    transformer_main = filter_main(transformer, ["vehicle_transformer_context_no_subject"])

    all_metrics = pd.concat(
        [formal_main, old_main, strong_main, transformer_main],
        ignore_index=True,
        sort=False,
    )
    all_metrics["model_order"] = all_metrics["model_name"].map(
        {name: idx for idx, name in enumerate(MODEL_ORDER)}
    )
    all_metrics["display_name"] = all_metrics["model_name"].map(DISPLAY_NAME)
    all_metrics["source_group"] = all_metrics["model_name"].map(SOURCE_GROUP)
    all_metrics["uses_subject_id"] = False
    all_metrics["uses_physio"] = False
    all_metrics["uses_eeg"] = False
    all_metrics["uses_continuous_style"] = False
    all_metrics["multi_segment_rate_abs_gap"] = (
        all_metrics["multi_segment_pred_rate"] - all_metrics["multi_segment_gt_rate"]
    ).abs()
    all_metrics = all_metrics.sort_values(["split", "model_order"]).reset_index(drop=True)
    test_metrics = all_metrics[all_metrics["split"] == "test"].copy()
    return all_metrics, test_metrics


def build_delta_table(test_metrics: pd.DataFrame) -> pd.DataFrame:
    base = test_metrics[
        test_metrics["model_name"] == "formal_ridge_vehicle_context_no_subject"
    ].iloc[0]
    rows = []
    for _, row in test_metrics.iterrows():
        out = {
            "model_name": row["model_name"],
            "display_name": row["display_name"],
            "source_group": row["source_group"],
            "rmse_steer": row["rmse_steer"],
            "delta_rmse_vs_formal": row["rmse_steer"] - base["rmse_steer"],
            "rmse_improvement_pct_vs_formal": (base["rmse_steer"] - row["rmse_steer"])
            / base["rmse_steer"]
            * 100.0,
        }
        for metric in [
            "peak_direction_accuracy",
            "wrong_side_rate",
            "large_response_recall",
            "severe_amp_under_rate",
            "tail_abs_error_mean",
            "reversal_count_exact_match_rate",
            "difficult_top20_rmse",
            "multi_segment_rate_abs_gap",
        ]:
            out[f"delta_{metric}_vs_formal"] = row[metric] - base[metric]
        rows.append(out)
    return pd.DataFrame(rows).sort_values("rmse_steer").reset_index(drop=True)


def build_rankings(test_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in LOWER_BETTER + HIGHER_BETTER:
        ascending = metric in LOWER_BETTER
        ranked = test_metrics[["model_name", "display_name", metric]].sort_values(
            metric, ascending=ascending
        )
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rows.append(
                {
                    "metric": metric,
                    "rank": rank,
                    "higher_is_better": metric in HIGHER_BETTER,
                    "model_name": row["model_name"],
                    "display_name": row["display_name"],
                    "value": row[metric],
                }
            )
    return pd.DataFrame(rows)


def build_decision_table(test_metrics: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    delta_map = delta.set_index("model_name")
    rows = []
    for _, row in test_metrics.sort_values("model_order").iterrows():
        name = row["model_name"]
        decision = "reference"
        priority = "low"
        if name == "formal_ridge_vehicle_context_no_subject":
            decision = "keep_as_formal_reference"
            priority = "required"
        elif name == "rbf_kernel_ridge_context_no_subject":
            decision = "strong_diagnostic_candidate_needs_controls"
            priority = "high"
        elif name == "vehicle_transformer_context_no_subject":
            decision = "true_transformer_candidate_needs_structured_fix"
            priority = "high"
        elif "knn" in name or "template" in name:
            decision = "diagnostic_upper_bound_memory_risk"
            priority = "medium"
        elif name in {"active_legacy_best", "structure_best"}:
            decision = "old_flow_reference_only"
            priority = "medium"
        elif name.startswith("ridge_rich"):
            decision = "not_selected_currently"
            priority = "low"
        rows.append(
            {
                "model_name": name,
                "display_name": row["display_name"],
                "source_group": row["source_group"],
                "decision": decision,
                "continue_priority": priority,
                "rmse_steer": row["rmse_steer"],
                "wrong_side_rate": row["wrong_side_rate"],
                "large_response_recall": row["large_response_recall"],
                "severe_amp_under_rate": row["severe_amp_under_rate"],
                "reversal_count_exact_match_rate": row[
                    "reversal_count_exact_match_rate"
                ],
                "multi_segment_rate_abs_gap": row["multi_segment_rate_abs_gap"],
                "rmse_improvement_pct_vs_formal": delta_map.loc[
                    name, "rmse_improvement_pct_vs_formal"
                ],
                "decision_note_cn": DECISION_NOTE.get(name, ""),
            }
        )
    return pd.DataFrame(rows)


def load_per_sample() -> pd.DataFrame:
    formal = read_csv(FORMAL_PER_SAMPLE)
    formal["source_artifact"] = "stage03_vehicle_instability_formal_baselines_v0_1"
    formal["model_name"] = formal["model_name"].replace(
        {"ridge_vehicle_context_no_subject": "formal_ridge_vehicle_context_no_subject"}
    )
    old = read_csv(OLD_PER_SAMPLE)
    old["source_artifact"] = "oldcode_vehicle_direct_full_clean_on_instability_v0_1"
    strong = read_csv(STRONG_PER_SAMPLE)
    strong["source_artifact"] = "stage03_vehicle_instability_strong_vehicle_baselines_v0_1"
    transformer = read_csv(TRANSFORMER_PER_SAMPLE)
    transformer["source_artifact"] = "stage03_vehicle_instability_vehicle_transformer_v0_1"
    per = pd.concat([formal, old, strong, transformer], ignore_index=True, sort=False)
    keep_models = [
        "formal_ridge_vehicle_context_no_subject",
        "active_legacy_best",
        "structure_best",
        "rbf_kernel_ridge_context_no_subject",
        "knn_template_context_no_subject",
        "peak_scaled_template_context_no_subject",
        "vehicle_transformer_context_no_subject",
    ]
    return per[
        (per["window_config_id"] == WINDOW_CONFIG)
        & (per["split_strategy"] == SPLIT_STRATEGY)
        & (per["split"] == "test")
        & per["model_name"].isin(keep_models)
    ].copy()


def build_bad_overlap(per_sample: pd.DataFrame, top_n: int = 28) -> pd.DataFrame:
    model_sets = {}
    for model, group in per_sample.groupby("model_name"):
        bad = group.sort_values("sample_rmse", ascending=False).head(top_n)
        model_sets[model] = set(bad["sample_id"].astype(str))
    rows = []
    for m1 in model_sets:
        for m2 in model_sets:
            s1, s2 = model_sets[m1], model_sets[m2]
            union = s1 | s2
            inter = s1 & s2
            rows.append(
                {
                    "model_a": m1,
                    "model_b": m2,
                    "top_n": top_n,
                    "overlap_count": len(inter),
                    "jaccard": len(inter) / len(union) if union else np.nan,
                }
            )
    return pd.DataFrame(rows)


def table_to_md(df: pd.DataFrame, columns: list[str], float_digits: int = 6) -> str:
    sub = df[columns].copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda x: f"{x:.{float_digits}f}")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in sub.values]
    return "\n".join([header, sep] + rows)


def plot_key_metrics(test_metrics: pd.DataFrame) -> None:
    selected = test_metrics[
        test_metrics["model_name"].isin(
            [
                "formal_ridge_vehicle_context_no_subject",
                "active_legacy_best",
                "structure_best",
                "rbf_kernel_ridge_context_no_subject",
                "knn_template_context_no_subject",
                "peak_scaled_template_context_no_subject",
                "vehicle_transformer_context_no_subject",
            ]
        )
    ].copy()
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, metric in zip(axes, KEY_METRICS):
        values = selected.sort_values(metric, ascending=metric in LOWER_BETTER)
        ax.barh(values["display_name"], values[metric], color="#4169a8")
        ax.set_title(metric)
        ax.grid(axis="x", alpha=0.25)
    axes[-1].axis("off")
    fig.suptitle("Stage03 vehicle-only unified comparison: key test metrics")
    fig.savefig(FIG_DIR / "unified_vehicle_key_metrics_test.png", dpi=180)
    plt.close(fig)


def plot_failure_heatmap(test_metrics: pd.DataFrame) -> None:
    selected = test_metrics[
        test_metrics["model_name"].isin(
            [
                "formal_ridge_vehicle_context_no_subject",
                "active_legacy_best",
                "structure_best",
                "rbf_kernel_ridge_context_no_subject",
                "knn_template_context_no_subject",
                "peak_scaled_template_context_no_subject",
                "vehicle_transformer_context_no_subject",
            ]
        )
    ].copy()
    failure_cols = [
        "rmse_steer",
        "wrong_side_rate",
        "severe_amp_under_rate",
        "tail_drift_risk_rate",
        "zero_crossing_mismatch_rate",
        "difficult_top20_rmse",
        "multi_segment_rate_abs_gap",
    ]
    mat = selected[failure_cols].astype(float)
    norm = (mat - mat.min()) / (mat.max() - mat.min()).replace(0, 1)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    im = ax.imshow(norm.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(failure_cols)), failure_cols, rotation=35, ha="right")
    ax.set_yticks(range(len(selected)), selected["display_name"])
    ax.set_title("Normalized physical failure risks (lower is better)")
    fig.colorbar(im, ax=ax, label="normalized risk")
    fig.savefig(FIG_DIR / "unified_vehicle_physical_failure_heatmap_test.png", dpi=180)
    plt.close(fig)


def plot_tradeoff(test_metrics: pd.DataFrame) -> None:
    selected = test_metrics.copy()
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    size = 80 + 420 * selected["severe_amp_under_rate"].astype(float)
    sc = ax.scatter(
        selected["rmse_steer"],
        selected["wrong_side_rate"],
        s=size,
        c=selected["large_response_recall"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.5,
    )
    for _, row in selected.iterrows():
        ax.text(row["rmse_steer"] + 0.003, row["wrong_side_rate"], row["display_name"], fontsize=8)
    ax.set_xlabel("RMSE steer")
    ax.set_ylabel("wrong side rate")
    ax.set_title("RMSE vs wrong-side tradeoff; color=large recall, size=amp under")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label="large response recall")
    fig.savefig(FIG_DIR / "unified_vehicle_rmse_vs_wrong_side_test.png", dpi=180)
    plt.close(fig)


def plot_bad_overlap(overlap: pd.DataFrame) -> None:
    models = list(dict.fromkeys(overlap["model_a"]))
    pivot = overlap.pivot(index="model_a", columns="model_b", values="overlap_count").loc[
        models, models
    ]
    labels = [DISPLAY_NAME.get(m, m) for m in models]
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    im = ax.imshow(pivot.values, cmap="Blues")
    ax.set_xticks(range(len(models)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(models)), labels)
    for i in range(len(models)):
        for j in range(len(models)):
            ax.text(j, i, int(pivot.iloc[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Top bad sample overlap count (top 28 by sample RMSE)")
    fig.colorbar(im, ax=ax, label="overlap count")
    fig.savefig(FIG_DIR / "unified_vehicle_top_bad_overlap.png", dpi=180)
    plt.close(fig)


def write_reports(
    test_metrics: pd.DataFrame,
    delta: pd.DataFrame,
    decisions: pd.DataFrame,
    overlap: pd.DataFrame,
) -> None:
    test_sorted = test_metrics.sort_values("rmse_steer")
    best_rmse = test_sorted.iloc[0]
    transformer = test_metrics[
        test_metrics["model_name"] == "vehicle_transformer_context_no_subject"
    ].iloc[0]
    rbf = test_metrics[
        test_metrics["model_name"] == "rbf_kernel_ridge_context_no_subject"
    ].iloc[0]
    knn = test_metrics[
        test_metrics["model_name"] == "knn_template_context_no_subject"
    ].iloc[0]
    formal = test_metrics[
        test_metrics["model_name"] == "formal_ridge_vehicle_context_no_subject"
    ].iloc[0]
    old_active = test_metrics[test_metrics["model_name"] == "active_legacy_best"].iloc[0]

    report = f"""# 阶段 3 统一车辆-only 对照 v0.1

生成时间：2026-05-12

## 目的

本报告把正式失稳样本 `vehicle_instability_highconf_v0_1` 的主窗口 `{WINDOW_CONFIG}` + `{SPLIT_STRATEGY}` 上已经完成的车辆-only 对照放到同一张表中。这里仍然只讨论车辆历史和事件/道路上下文，不讨论连续风格、生理或 EEG 的有效性。

## 输入边界

- 使用已经生成的阶段 3 指标文件，不重新训练模型。
- 所有候选均不使用生理、脑电、连续风格或驾驶员 ID。
- `eval_label_*` 只允许用于评价分层和图表，不作为模型输入。
- 本轮未连接服务器，未读取服务器指令与密码文件。

## test 集核心指标

{table_to_md(test_sorted, [
    "display_name",
    "source_group",
    "rmse_steer",
    "peak_direction_accuracy",
    "wrong_side_rate",
    "large_response_recall",
    "severe_amp_under_rate",
    "tail_abs_error_mean",
    "reversal_count_exact_match_rate",
    "multi_segment_rate_abs_gap",
    "difficult_top20_rmse",
])}

## 当前判断

- RMSE 最低的是 `{DISPLAY_NAME[best_rmse['model_name']]}`，test RMSE={best_rmse['rmse_steer']:.6f}。
- `KNN template` 的 test RMSE={knn['rmse_steer']:.6f}，但训练集 RMSE 近 0，属于模板记忆风险候选，不能直接当最终主线。
- `RBF KRR` 的 test RMSE={rbf['rmse_steer']:.6f}，大幅响应召回={rbf['large_response_recall']:.6f}，但反向修正计数匹配率={rbf['reversal_count_exact_match_rate']:.6f}，说明它的复杂响应结构仍弱。
- `vehicle Transformer` 是真正的车辆-only Transformer：test RMSE={transformer['rmse_steer']:.6f}，优于 formal ridge 的 {formal['rmse_steer']:.6f}，也优于旧 `vehicle_direct active` 的 {old_active['rmse_steer']:.6f}，但它的多段修正预测率与 GT 差距={transformer['multi_segment_rate_abs_gap']:.6f}，暂不能作为最终强车辆主线。
- 因此当前阶段 3 结论不是“某个模型胜出”，而是：车辆-only 已经有多个强对照，下一步必须用物理错误和稳健性验证冻结主车辆参照。

## 候选决策表

{table_to_md(decisions, [
    "display_name",
    "decision",
    "continue_priority",
    "rmse_steer",
    "rmse_improvement_pct_vs_formal",
    "decision_note_cn",
])}

## 关键产物

- 指标总表：`{TABLE_DIR / "unified_vehicle_comparison_metrics_test.csv"}`
- 相对 formal ridge 差异：`{TABLE_DIR / "unified_vehicle_comparison_delta_vs_formal_test.csv"}`
- 候选决策表：`{TABLE_DIR / "unified_vehicle_candidate_decision_table.csv"}`
- 坏样本重合表：`{TABLE_DIR / "unified_vehicle_top_bad_overlap.csv"}`
- 关键指标图：`{FIG_DIR / "unified_vehicle_key_metrics_test.png"}`
- 物理错误热图：`{FIG_DIR / "unified_vehicle_physical_failure_heatmap_test.png"}`
- RMSE/错侧权衡图：`{FIG_DIR / "unified_vehicle_rmse_vs_wrong_side_test.png"}`
- 坏样本重合图：`{FIG_DIR / "unified_vehicle_top_bad_overlap.png"}`

## 下一步

1. 先做 subject-level split 或窗口敏感性检查，验证 RBF/KNN/Transformer 的收益是否稳定。
2. 对 top bad overlap 中反复失败的样本绘图复盘，确认是事件锚点问题、车辆历史信息不足、还是模型结构问题。
3. 在冻结强车辆主参照前，继续阻塞连续风格、生理和 EEG 有效性结论。
"""

    user_summary = f"""# 阶段 3 用户查看版：车辆-only 模型统一对照

## 这个阶段为什么做

前面已经分别跑过 formal ridge、旧 `vehicle_direct`、RBF/KNN/template 和真正 Transformer。单独看每次结果容易误判，所以这一步把它们放到同一张表里比较，避免只看 RMSE。

## 这个阶段检查了什么

- 整体 RMSE。
- 方向是否预测反了。
- 大幅响应有没有召回。
- 幅值是否严重不足。
- 尾段误差和漂移风险。
- 反向修正和多段修正是否能识别。
- 坏样本是否集中在同一批事件。

## 目前发现了什么

1. formal ridge 是最低公平参照，test RMSE={formal['rmse_steer']:.6f}。
2. 旧 `vehicle_direct active` test RMSE={old_active['rmse_steer']:.6f}，只能作历史对照。
3. RBF/KNN/template 的 RMSE 更低，KNN template test RMSE={knn['rmse_steer']:.6f}，但 KNN 训练集几乎记住模板，风险很高。
4. 真正 Transformer test RMSE={transformer['rmse_steer']:.6f}，比 formal ridge 好，但还没有解决多段修正问题。

## 哪些结果可信

这些结果都来自同一批正式失稳样本、同一主窗口和同一 session-level test 集。它们都没有使用生理、脑电、连续风格或驾驶员 ID，所以可以作为车辆-only 对照。

## 哪些结果还不能下结论

不能说 KNN/RBF/Transformer 任何一个已经是最终强车辆主线。KNN/RBF 可能受模板记忆或局部分布影响，Transformer 还漏多段修正。也不能根据这些结果说生理或连续风格有效。

## 下一阶段是否可以继续

可以继续，但下一步仍属于阶段 3：先做强车辆基线稳健性验证和坏样本复盘。只有车辆-only 主参照冻结后，才适合进入连续风格和生理增量验证。

## 推荐优先查看

1. `{TABLE_DIR / "unified_vehicle_comparison_metrics_test.csv"}`
2. `{TABLE_DIR / "unified_vehicle_candidate_decision_table.csv"}`
3. `{FIG_DIR / "unified_vehicle_key_metrics_test.png"}`
4. `{FIG_DIR / "unified_vehicle_physical_failure_heatmap_test.png"}`
5. `{FIG_DIR / "unified_vehicle_top_bad_overlap.png"}`
"""

    (REPORT_ROOT / "stage03_vehicle_instability_unified_vehicle_comparison_v0_1_cn.md").write_text(
        report, encoding="utf-8"
    )
    (REPORT_ROOT / "stage03_vehicle_instability_unified_vehicle_comparison_user_summary_cn.md").write_text(
        user_summary, encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    all_metrics, test_metrics = build_unified_metrics()
    delta = build_delta_table(test_metrics)
    rankings = build_rankings(test_metrics)
    decisions = build_decision_table(test_metrics, delta)
    per_sample = load_per_sample()
    overlap = build_bad_overlap(per_sample, top_n=28)

    all_metrics.to_csv(
        TABLE_DIR / "unified_vehicle_comparison_metrics_all_splits.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_metrics.to_csv(
        TABLE_DIR / "unified_vehicle_comparison_metrics_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    delta.to_csv(
        TABLE_DIR / "unified_vehicle_comparison_delta_vs_formal_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rankings.to_csv(
        TABLE_DIR / "unified_vehicle_metric_rankings_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decisions.to_csv(
        TABLE_DIR / "unified_vehicle_candidate_decision_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overlap.to_csv(
        TABLE_DIR / "unified_vehicle_top_bad_overlap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_key_metrics(test_metrics)
    plot_failure_heatmap(test_metrics)
    plot_tradeoff(test_metrics)
    plot_bad_overlap(overlap)
    write_reports(test_metrics, delta, decisions, overlap)

    summary = {
        "window_config_id": WINDOW_CONFIG,
        "split_strategy": SPLIT_STRATEGY,
        "model_count_test": int(test_metrics["model_name"].nunique()),
        "best_rmse_model": str(test_metrics.sort_values("rmse_steer").iloc[0]["model_name"]),
        "best_rmse": float(test_metrics["rmse_steer"].min()),
        "formal_rmse": float(
            test_metrics.loc[
                test_metrics["model_name"] == "formal_ridge_vehicle_context_no_subject",
                "rmse_steer",
            ].iloc[0]
        ),
        "transformer_rmse": float(
            test_metrics.loc[
                test_metrics["model_name"] == "vehicle_transformer_context_no_subject",
                "rmse_steer",
            ].iloc[0]
        ),
        "uses_subject_id": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "outputs": {
            "metrics_test": str(TABLE_DIR / "unified_vehicle_comparison_metrics_test.csv"),
            "decision_table": str(TABLE_DIR / "unified_vehicle_candidate_decision_table.csv"),
            "user_summary": str(
                REPORT_ROOT
                / "stage03_vehicle_instability_unified_vehicle_comparison_user_summary_cn.md"
            ),
        },
    }
    (LOG_DIR / "unified_vehicle_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
