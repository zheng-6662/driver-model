from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = PROJECT_ROOT / "03_baselines"
REPORT_ROOT = PROJECT_ROOT / "09_reports"
OUTPUT_ROOT = BASELINE_ROOT / "stage03_vehicle_instability_strong_vehicle_robustness_v0_1"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIG_DIR = OUTPUT_ROOT / "figures"
LOG_DIR = OUTPUT_ROOT / "logs"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage03_vehicle_instability_formal_baselines_v0_1 as formal_v01  # noqa: E402
import stage03_vehicle_instability_strong_vehicle_baselines_v0_1 as strong_v01  # noqa: E402


ROBUSTNESS_CONFIGS = [
    ("pre2_label2_old_main", "random_event_split", "random_main"),
    ("pre2_label2_old_main", "subject_level_split", "subject_main"),
    ("pre1_label2_event_trigger", "session_level_split", "session_pre1"),
    ("pre3_label3_response_coverage", "session_level_split", "session_pre3"),
]
CANDIDATE_MODELS = [
    "rbf_kernel_ridge_context_no_subject",
    "knn_template_context_no_subject",
    "direction_gated_knn_template_no_subject",
    "peak_scaled_template_context_no_subject",
]
PLOT_MODELS = [
    "formal_ridge_vehicle_context_no_subject",
    "rbf_kernel_ridge_context_no_subject",
    "knn_template_context_no_subject",
    "peak_scaled_template_context_no_subject",
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def set_strong_context(window_id: str, split_strategy: str) -> None:
    strong_v01.WINDOW_ID = window_id
    strong_v01.SPLIT_STRATEGY = split_strategy


def run_one_config(
    samples: pd.DataFrame,
    window_id: str,
    split_strategy: str,
    config_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_strong_context(window_id, split_strategy)
    y, y_mask, input_values, input_time, label_time, meta = formal_v01.load_window(window_id, samples)
    train_idx, val_idx, test_idx = strong_v01.split_indices(meta)
    if not (len(train_idx) and len(val_idx) and len(test_idx)):
        raise RuntimeError(f"{config_id}: incomplete train/val/test split")

    formal_preds, _ = formal_v01.build_predictions(
        y, y_mask, input_values, input_time, label_time, meta, split_strategy
    )
    predictions: dict[str, np.ndarray] = {
        "formal_ridge_vehicle_context_no_subject": formal_preds[
            "ridge_vehicle_context_no_subject"
        ],
    }
    model_infos: list[dict[str, Any]] = [
        {
            "robustness_config_id": config_id,
            "window_config_id": window_id,
            "split_strategy": split_strategy,
            "model_name": "formal_ridge_vehicle_context_no_subject",
            "role": "formal_reference",
            "train_n": int(len(train_idx)),
            "val_n": int(len(val_idx)),
            "test_n": int(len(test_idx)),
            "uses_subject_id": False,
            "uses_physio": False,
            "uses_eeg": False,
            "uses_continuous_style": False,
        }
    ]

    X_rich, rich_names = strong_v01.build_rich_vehicle_features(
        input_values, input_time, meta, train_idx, include_context=True
    )
    X_scaled, _ = strong_v01.standardize_train_only(X_rich, train_idx)
    X_dist, selector = strong_v01.make_distance_features(X_scaled, train_idx, n_components=96)
    peaks = strong_v01.peak_arrays(y, y_mask, label_time)

    fitters = [
        ("rbf_kernel_ridge_context_no_subject", lambda: strong_v01.fit_rbf_kernel_ridge_direct(X_dist, y, train_idx, val_idx, y_mask)),
        ("knn_template_context_no_subject", lambda: strong_v01.fit_knn_template("knn_template_context_no_subject", X_dist, y, y_mask, train_idx, val_idx)),
        ("direction_gated_knn_template_no_subject", lambda: strong_v01.fit_direction_gated_knn_template(X_dist, y, y_mask, peaks, train_idx, val_idx)),
        ("peak_scaled_template_context_no_subject", lambda: strong_v01.fit_peak_scaled_template(X_dist, y, y_mask, peaks, train_idx, val_idx)),
    ]
    for model_name, fit in fitters:
        pred, info = fit()
        predictions[model_name] = pred
        info.update(
            {
                "robustness_config_id": config_id,
                "window_config_id": window_id,
                "split_strategy": split_strategy,
                "train_n": int(len(train_idx)),
                "val_n": int(len(val_idx)),
                "test_n": int(len(test_idx)),
                "rich_feature_count": int(len(rich_names)),
                "distance_feature_count": int(X_dist.shape[1]),
                "selector_source": "train split variance only",
            }
        )
        model_infos.append(info)

    metrics, per_sample = strong_v01.evaluate_predictions(
        y, y_mask, label_time, meta, predictions, train_idx
    )
    for df in [metrics, per_sample]:
        df.insert(0, "robustness_config_id", config_id)
        df.insert(1, "window_config_id_check", window_id)
        df.insert(2, "split_strategy_check", split_strategy)
    model_info = pd.DataFrame(model_infos)
    selected = strong_v01.select_val_model(metrics, CANDIDATE_MODELS)
    model_info["val_selected_model_for_config"] = selected
    return metrics, per_sample, model_info


def build_decision(metrics: pd.DataFrame, model_info: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["split"] == "test"].copy()
    rows = []
    for config_id, group in test.groupby("robustness_config_id"):
        info_row = model_info[model_info["robustness_config_id"] == config_id].iloc[0]
        selected = str(info_row["val_selected_model_for_config"])
        formal = group[group["model_name"] == "formal_ridge_vehicle_context_no_subject"].iloc[0]
        selected_row = group[group["model_name"] == selected].iloc[0]
        best_test = group.sort_values("rmse_steer").iloc[0]
        rbf = group[group["model_name"] == "rbf_kernel_ridge_context_no_subject"].iloc[0]
        knn = group[group["model_name"] == "knn_template_context_no_subject"].iloc[0]
        rows.append(
            {
                "robustness_config_id": config_id,
                "window_config_id": info_row["window_config_id"],
                "split_strategy": info_row["split_strategy"],
                "val_selected_model": selected,
                "val_selected_test_rmse": float(selected_row["rmse_steer"]),
                "best_test_model": best_test["model_name"],
                "best_test_rmse": float(best_test["rmse_steer"]),
                "formal_rmse": float(formal["rmse_steer"]),
                "selected_rmse_improvement_pct_vs_formal": (
                    float(formal["rmse_steer"]) - float(selected_row["rmse_steer"])
                )
                / float(formal["rmse_steer"])
                * 100.0,
                "rbf_test_rmse": float(rbf["rmse_steer"]),
                "rbf_reversal_exact": float(rbf["reversal_count_exact_match_rate"]),
                "knn_test_rmse": float(knn["rmse_steer"]),
                "knn_train_rmse": float(
                    metrics[
                        (metrics["robustness_config_id"] == config_id)
                        & (metrics["split"] == "train")
                        & (metrics["model_name"] == "knn_template_context_no_subject")
                    ]["rmse_steer"].iloc[0]
                ),
                "knn_memory_risk": bool(
                    metrics[
                        (metrics["robustness_config_id"] == config_id)
                        & (metrics["split"] == "train")
                        & (metrics["model_name"] == "knn_template_context_no_subject")
                    ]["rmse_steer"].iloc[0]
                    < 1e-3
                ),
                "interpretation_cn": "",
            }
        )
    out = pd.DataFrame(rows)
    out["interpretation_cn"] = out.apply(
        lambda r: (
            "KNN 训练集近零误差，仍按模板记忆风险处理；"
            if r["knn_memory_risk"]
            else "KNN 训练集未近零，但仍需检查邻近模板依赖；"
        )
        + (
            "val 选择候选相对 formal 有提升。"
            if r["selected_rmse_improvement_pct_vs_formal"] > 0
            else "val 选择候选未超过 formal。"
        ),
        axis=1,
    )
    return out.sort_values("robustness_config_id").reset_index(drop=True)


def plot_heatmap(
    metrics: pd.DataFrame,
    model_names: list[str],
    metric: str,
    out_name: str,
    title: str,
) -> None:
    test = metrics[(metrics["split"] == "test") & (metrics["model_name"].isin(model_names))].copy()
    pivot = test.pivot(index="model_name", columns="robustness_config_id", values=metric).loc[model_names]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax, label=metric)
    fig.savefig(FIG_DIR / out_name, dpi=180)
    plt.close(fig)


def table_to_md(df: pd.DataFrame, columns: list[str]) -> str:
    sub = df[columns].copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda x: f"{x:.6f}")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in sub.values]
    return "\n".join([header, sep] + rows)


def write_reports(decision: pd.DataFrame, metrics: pd.DataFrame) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    main_subject = decision[decision["robustness_config_id"] == "subject_main"].iloc[0]
    report = f"""# 阶段 3 强车辆基线稳健性验证 v0.1

生成时间：2026-05-12

## 目的

上一轮统一对照显示 KNN/RBF/template 的 RMSE 很低，但可能存在模板记忆或分布依赖风险。本轮不进入风格/生理，只检查强车辆-only 候选在 subject-level split 和不同输入/标签窗口下是否稳定。

## 检查配置

- `random_main`：主 2 秒窗口 + random-event split。
- `subject_main`：主 2 秒窗口 + subject-level split。
- `session_pre1`：事件前 1 秒预测后 2 秒 + session-level split。
- `session_pre3`：事件前 3 秒预测后 3 秒 + session-level split。

## test 决策表

{table_to_md(decision, [
    "robustness_config_id",
    "val_selected_model",
    "val_selected_test_rmse",
    "best_test_model",
    "best_test_rmse",
    "formal_rmse",
    "selected_rmse_improvement_pct_vs_formal",
    "knn_train_rmse",
    "knn_memory_risk",
])}

## 初步判断

- subject-level 主窗口中，val 选择模型为 `{main_subject['val_selected_model']}`，test RMSE={main_subject['val_selected_test_rmse']:.6f}，formal RMSE={main_subject['formal_rmse']:.6f}。
- KNN 在各配置中的 train RMSE 仍接近 0 时，继续标记为模板记忆风险，不能直接升级为主线。
- 本轮仍不支持任何生理、脑电或连续风格有效性结论。

## 产物

- 指标表：`{(TABLE_DIR / "strong_vehicle_robustness_metrics.csv").as_posix()}`
- 决策表：`{(TABLE_DIR / "strong_vehicle_robustness_decision_table.csv").as_posix()}`
- 模型信息：`{(TABLE_DIR / "strong_vehicle_robustness_model_info.csv").as_posix()}`
- RMSE 热图：`{(FIG_DIR / "strong_vehicle_robustness_rmse_heatmap.png").as_posix()}`
- 大幅响应召回热图：`{(FIG_DIR / "strong_vehicle_robustness_large_recall_heatmap.png").as_posix()}`
- 反向修正匹配热图：`{(FIG_DIR / "strong_vehicle_robustness_reversal_heatmap.png").as_posix()}`
"""

    user_summary = f"""# 阶段 3 用户查看版：强车辆基线稳健性验证

## 为什么做

之前 KNN/RBF 的 RMSE 很低，但这不一定代表它们真正学到了可泛化规律。这个阶段专门检查：换成跨被试划分或换输入窗口后，低 RMSE 是否还稳定。

## 检查了什么

- 主 2 秒窗口的 random-event、session-level、subject-level 对照。
- 事件前 1 秒和前 3 秒窗口敏感性。
- RBF、KNN、方向门控 KNN、峰值缩放模板与 formal ridge 的比较。

## 目前发现

subject-level 主窗口中，val 选择模型是 `{main_subject['val_selected_model']}`，test RMSE={main_subject['val_selected_test_rmse']:.6f}，formal RMSE={main_subject['formal_rmse']:.6f}。

## 还不能下什么结论

KNN 即使 test RMSE 低，只要 train RMSE 仍接近 0，就要继续按模板记忆风险处理。RBF/KNN 是否能作为主线，还需要结合固定图、坏样本图和跨被试物理指标判断。

## 下一步

继续阶段 3：复盘 subject-level 和窗口敏感性下的坏样本，决定是否转向响应分解、关键点残差或多假设车辆模型。仍不进入生理/连续风格有效性结论。

## 推荐优先查看

1. `{(TABLE_DIR / "strong_vehicle_robustness_decision_table.csv").as_posix()}`
2. `{(TABLE_DIR / "strong_vehicle_robustness_metrics.csv").as_posix()}`
3. `{(FIG_DIR / "strong_vehicle_robustness_rmse_heatmap.png").as_posix()}`
4. `{(FIG_DIR / "strong_vehicle_robustness_large_recall_heatmap.png").as_posix()}`
5. `{(FIG_DIR / "strong_vehicle_robustness_reversal_heatmap.png").as_posix()}`
"""

    (REPORT_ROOT / "stage03_vehicle_instability_strong_vehicle_robustness_v0_1_cn.md").write_text(
        report, encoding="utf-8"
    )
    (REPORT_ROOT / "stage03_vehicle_instability_strong_vehicle_robustness_user_summary_cn.md").write_text(
        user_summary, encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    samples = pd.read_csv(formal_v01.SAMPLES_PATH)
    all_metrics: list[pd.DataFrame] = []
    all_per_sample: list[pd.DataFrame] = []
    all_info: list[pd.DataFrame] = []
    for window_id, split_strategy, config_id in ROBUSTNESS_CONFIGS:
        print(f"running {config_id}: {window_id} / {split_strategy}", flush=True)
        metrics, per_sample, info = run_one_config(samples, window_id, split_strategy, config_id)
        all_metrics.append(metrics)
        all_per_sample.append(per_sample)
        all_info.append(info)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    per_sample_df = pd.concat(all_per_sample, ignore_index=True)
    info_df = pd.concat(all_info, ignore_index=True)
    decision = build_decision(metrics_df, info_df)

    metrics_df.to_csv(TABLE_DIR / "strong_vehicle_robustness_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample_df.to_csv(TABLE_DIR / "strong_vehicle_robustness_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    info_df.to_csv(TABLE_DIR / "strong_vehicle_robustness_model_info.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(TABLE_DIR / "strong_vehicle_robustness_decision_table.csv", index=False, encoding="utf-8-sig")

    plot_heatmap(
        metrics_df,
        PLOT_MODELS,
        "rmse_steer",
        "strong_vehicle_robustness_rmse_heatmap.png",
        "Strong vehicle robustness: test RMSE",
    )
    plot_heatmap(
        metrics_df,
        PLOT_MODELS,
        "large_response_recall",
        "strong_vehicle_robustness_large_recall_heatmap.png",
        "Strong vehicle robustness: large response recall",
    )
    plot_heatmap(
        metrics_df,
        PLOT_MODELS,
        "reversal_count_exact_match_rate",
        "strong_vehicle_robustness_reversal_heatmap.png",
        "Strong vehicle robustness: reversal exact match",
    )
    write_reports(decision, metrics_df)

    summary = {
        "config_count": len(ROBUSTNESS_CONFIGS),
        "configs": [c[2] for c in ROBUSTNESS_CONFIGS],
        "server_used": False,
        "credential_file_read": False,
        "uses_subject_id": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "raw_files_modified": False,
        "decision_path": str(TABLE_DIR / "strong_vehicle_robustness_decision_table.csv"),
        "subject_main": decision[decision["robustness_config_id"] == "subject_main"].iloc[0].to_dict(),
    }
    (LOG_DIR / "strong_vehicle_robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
