# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
MANIFEST_PATH = ROOT / "02_samples" / "vehicle_instability_response_task_decision_v0_1" / "tables" / "sample_response_task_manifest.csv"
IN_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1"
PER_SAMPLE_PATH = IN_DIR / "tables" / "keypoint_residual_vehicle_transformer_per_sample_metrics.csv"
OUT_ROOT = ROOT / "03_baselines" / "stage03_vehicle_instability_rbf_keypoint_selector_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_ROOT = ROOT / "09_reports"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402


TRACK_ID = "B_response3s_strict_core"
SPLIT_STRATEGY = "session_level_split"
RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
KEYPOINT_MODEL = "keypoint_residual_vehicle_transformer_no_subject"
SELECTOR_MODEL = "selector_logreg_rbf_keypoint_no_subject"
ORACLE_MODEL = "oracle_best_of_rbf_keypoint_upper_bound"

NUMERIC_CONTEXT_COLS = [
    "anchor_time_rel_s",
    "curvature_anchor",
    "input_valid_ratio",
]
CATEGORICAL_CONTEXT_COLS = [
    "event_type",
    "event_level",
    "road_type_anchor",
    "old_v400_road_type_mode",
    "old_v400_phase_mode",
    "road_design_module_name",
    "road_design_instance_name",
    "road_design_risk_class",
    "road_design_mapping_reliability",
]
PRED_FEATURES = [
    "pred_peak_abs",
    "pred_reversal_count",
    "pred_multi_segment",
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def load_candidate_rows() -> pd.DataFrame:
    df = pd.read_csv(PER_SAMPLE_PATH)
    df = df[(df["track_id"] == TRACK_ID) & (df["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL]))].copy()
    if df.empty:
        raise RuntimeError("candidate per-sample rows are missing")
    return df


def build_selector_table(rows: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["sample_id"].isin(rows["sample_id"].unique())].copy()
    keep_cols = ["sample_id", *NUMERIC_CONTEXT_COLS, *CATEGORICAL_CONTEXT_COLS]
    manifest = manifest[[c for c in keep_cols if c in manifest.columns]].drop_duplicates("sample_id")
    base_cols = [
        "sample_id",
        "event_uid",
        "subject",
        "session_stamp",
        "window_config_id",
        "split",
        "split_strategy",
        "track_id",
        "gt_peak_abs",
        "is_large_response",
        "is_difficult_peak_top20",
    ]
    base = rows[base_cols].drop_duplicates("sample_id").set_index("sample_id")
    metric_cols = [
        "sample_rmse",
        "wrong_side",
        "large_response_recalled",
        "severe_amp_under",
        "peak_amp_abs_error",
        "peak_time_abs_error_s",
        "onset_delay_abs_error_s",
        "tail_abs_error",
        "tail_drift_risk",
        "zero_crossing_mismatch",
        "reversal_count_exact",
        *PRED_FEATURES,
    ]
    wide = rows.pivot(index="sample_id", columns="model_name", values=metric_cols)
    wide.columns = [f"{metric}__{model}" for metric, model in wide.columns]
    table = base.join(wide).reset_index().merge(manifest, on="sample_id", how="left", validate="one_to_one")
    table["keypoint_better_rmse"] = (
        table[f"sample_rmse__{KEYPOINT_MODEL}"] < table[f"sample_rmse__{RBF_MODEL}"]
    ).astype(int)
    table["rmse_delta_keypoint_minus_rbf"] = table[f"sample_rmse__{KEYPOINT_MODEL}"] - table[f"sample_rmse__{RBF_MODEL}"]
    for feat in PRED_FEATURES:
        table[f"{feat}__delta_keypoint_minus_rbf"] = table[f"{feat}__{KEYPOINT_MODEL}"] - table[f"{feat}__{RBF_MODEL}"]
    return table


def feature_columns(table: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in NUMERIC_CONTEXT_COLS if c in table.columns]
    for model in [RBF_MODEL, KEYPOINT_MODEL]:
        numeric.extend([f"{feat}__{model}" for feat in PRED_FEATURES])
    numeric.extend([f"{feat}__delta_keypoint_minus_rbf" for feat in PRED_FEATURES])
    numeric = [c for c in numeric if c in table.columns]
    categorical = [c for c in CATEGORICAL_CONTEXT_COLS if c in table.columns]
    return numeric, categorical


def fit_selector(table: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    numeric, categorical = feature_columns(table)
    train = table[table["split"] == "train"].copy()
    if train["keypoint_better_rmse"].nunique() < 2:
        raise RuntimeError("selector train target has only one class")
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260513)
    pipe = Pipeline([("preprocess", pre), ("clf", clf)])
    pipe.fit(train[numeric + categorical], train["keypoint_better_rmse"].astype(int))
    return pipe, numeric, categorical


def choose_rows(rows: pd.DataFrame, table: pd.DataFrame, decisions: pd.Series, model_name: str) -> pd.DataFrame:
    pair = rows[rows["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL])].copy()
    selected_model = pd.DataFrame({"sample_id": table["sample_id"].to_numpy(), "selected_model": decisions.to_numpy()})
    out = pair.merge(selected_model, on="sample_id", how="inner")
    out = out[out["model_name"] == out["selected_model"]].copy()
    out["model_name"] = model_name
    out = out.drop(columns=["selected_model"])
    return out


def aggregate_selected(rows: pd.DataFrame, table: pd.DataFrame, prob: np.ndarray, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = pd.Series(np.where(prob >= threshold, KEYPOINT_MODEL, RBF_MODEL), index=table.index)
    selected = choose_rows(rows, table, decisions, SELECTOR_MODEL)
    selected["selector_prob_keypoint"] = np.repeat(np.nan, len(selected))
    selector_metrics = eval_utils.aggregate_metrics(selected.drop(columns=["selector_prob_keypoint"], errors="ignore"))
    selector_metrics["track_id"] = TRACK_ID
    decision_table = table[
        [
            "sample_id",
            "event_uid",
            "subject",
            "session_stamp",
            "split",
            "gt_peak_abs",
            "is_large_response",
            "is_difficult_peak_top20",
            "rmse_delta_keypoint_minus_rbf",
            "keypoint_better_rmse",
        ]
    ].copy()
    decision_table["selector_prob_keypoint"] = prob
    decision_table["selector_threshold"] = threshold
    decision_table["selected_model"] = decisions.to_numpy()
    return selector_metrics, decision_table


def selector_rmse_for_split(rows: pd.DataFrame, table: pd.DataFrame, prob: np.ndarray, threshold: float, split: str) -> float:
    mask = table["split"].to_numpy() == split
    if not mask.any():
        return float("inf")
    decisions = pd.Series(np.where(prob >= threshold, KEYPOINT_MODEL, RBF_MODEL), index=table.index)
    selected = choose_rows(rows, table.loc[mask].copy(), decisions.loc[mask], SELECTOR_MODEL)
    return float(np.sqrt(np.mean(selected["sample_rmse"].to_numpy(dtype=float) ** 2)))


def threshold_sweep(rows: pd.DataFrame, table: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    records: list[dict[str, Any]] = []
    for threshold in thresholds:
        for split in ["train", "val"]:
            mask = table["split"].to_numpy() == split
            selected_rate = float((prob[mask] >= threshold).mean()) if mask.any() else float("nan")
            records.append(
                {
                    "threshold": float(threshold),
                    "split": split,
                    "selector_rmse": selector_rmse_for_split(rows, table, prob, float(threshold), split),
                    "keypoint_selected_rate": selected_rate,
                }
            )
    return pd.DataFrame(records)


def oracle_rows(rows: pd.DataFrame) -> pd.DataFrame:
    pair = rows[rows["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL])].copy()
    idx = pair.sort_values("sample_rmse").groupby("sample_id").head(1).index
    out = pair.loc[idx].copy()
    out["model_name"] = ORACLE_MODEL
    return out


def build_all_metrics(rows: pd.DataFrame, selector_rows: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    compare = pd.concat(
        [
            rows[rows["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL])].copy(),
            selector_rows.copy(),
            oracle.copy(),
        ],
        ignore_index=True,
        sort=False,
    )
    metrics = eval_utils.aggregate_metrics(compare.drop(columns=["selector_prob_keypoint"], errors="ignore"))
    metrics["track_id"] = TRACK_ID
    return metrics


def plot_outputs(metrics: pd.DataFrame, sweep: pd.DataFrame) -> tuple[Path, Path]:
    test = metrics[(metrics["split"] == "test") & (metrics["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL, SELECTOR_MODEL, ORACLE_MODEL]))].copy()
    order = [RBF_MODEL, KEYPOINT_MODEL, SELECTOR_MODEL, ORACLE_MODEL]
    labels = ["RBF", "keypoint", "selector", "oracle"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, title in [
        (axes[0], "rmse_steer", "test RMSE"),
        (axes[1], "wrong_side_rate", "test wrong-side"),
        (axes[2], "large_response_recall", "test large recall"),
    ]:
        part = test.set_index("model_name").reindex(order)
        ax.bar(labels, part[col].to_numpy(dtype=float), color=["#1f77b4", "#2ca02c", "#9467bd", "#7f7f7f"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    metric_fig = FIG_DIR / "rbf_keypoint_selector_test_metrics.png"
    fig.savefig(metric_fig, dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for split, color in [("train", "#4c78a8"), ("val", "#f58518")]:
        part = sweep[sweep["split"] == split].sort_values("threshold")
        ax.plot(part["threshold"], part["selector_rmse"], marker="o", color=color, label=split)
    ax.set_xlabel("probability threshold for selecting keypoint")
    ax.set_ylabel("selector RMSE")
    ax.set_title("RBF/keypoint selector threshold sweep")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    sweep_fig = FIG_DIR / "rbf_keypoint_selector_threshold_sweep.png"
    fig.savefig(sweep_fig, dpi=170)
    plt.close(fig)
    return metric_fig, sweep_fig


def write_reports(metrics: pd.DataFrame, decision_table: pd.DataFrame, sweep: pd.DataFrame, summary: dict[str, Any], figures: dict[str, str]) -> None:
    test = metrics[(metrics["split"] == "test") & (metrics["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL, SELECTOR_MODEL, ORACLE_MODEL]))].copy()
    show_cols = [
        "model_name",
        "n_samples",
        "rmse_steer",
        "wrong_side_rate",
        "large_response_recall",
        "peak_amp_mae",
        "peak_time_mae_s",
        "onset_delay_mae_s",
        "tail_drift_risk_rate",
        "reversal_count_exact_match_rate",
        "difficult_top20_rmse",
    ]
    test_table = test[[c for c in show_cols if c in test.columns]].sort_values("rmse_steer")
    selected_counts = decision_table[decision_table["split"] == "test"]["selected_model"].value_counts().rename_axis("selected_model").reset_index(name="n_test_samples")
    report = f"""# 阶段 3：RBF vs keypoint train/val 选择器 v0.1

生成时间：2026-05-13

## 为什么做

RBF KRR 的整体 RMSE 稳定，但 keypoint+residual 在错侧率和大幅响应召回上更好。本轮测试一个只用 train 训练、只用 val 定阈值的选择器，判断是否能在 test 前决定每个样本选 RBF 还是 keypoint。

## 输入和无泄漏边界

- 逐样本指标：`{PER_SAMPLE_PATH.as_posix()}`
- 样本 manifest：`{MANIFEST_PATH.as_posix()}`
- 训练：selector 只用 train split 拟合。
- 阈值：只用 val split 按 selector RMSE 选择。
- test：只做最终评价，不参与训练或阈值选择。
- 特征：事件前车辆模型已经可输出的候选预测特征 + 事件/道路上下文；不使用 GT peak、sample_rmse、wrong_side、large_response、subject ID、生理、脑电、连续风格。
- 本轮未连接服务器，未读取服务器指令与密码文件。

## 阈值选择

- val 选择阈值：{summary['selected_threshold']:.2f}
- val selector RMSE：{summary['selected_val_rmse']:.6f}
- test keypoint 选择率：{summary['test_keypoint_selected_rate']:.6f}

## test 指标对照

```text
{test_table.to_string(index=False)}
```

## test 选择计数

```text
{selected_counts.to_string(index=False)}
```

## 结论边界

这个选择器是第一版 train/val 可用策略，不是 test oracle。若它不能超过 RBF，说明当前可用特征还不足以稳定判断何时选 keypoint；若它接近 oracle，则可以继续发展为多假设/可靠性模型。无论结果如何，本轮不能说明连续风格、生理或 EEG 有效。

## 图

- test 指标图：`{figures.get('test_metrics', '')}`
- 阈值扫描图：`{figures.get('threshold_sweep', '')}`
"""
    (REPORT_ROOT / "stage03_vehicle_instability_rbf_keypoint_selector_v0_1_cn.md").write_text(report, encoding="utf-8")

    user = f"""# 阶段 3 用户查看版：RBF 和 keypoint 的自动选择器

## 为什么做

RBF 整体更稳，keypoint 更会修方向和大幅响应。我们不能在 test 上事后挑哪个模型好，所以这一步用 train 训练一个选择器，再用 val 定阈值，最后只在 test 上看结果。

## 这次检查了什么

- 每个样本在预测前应该选 RBF 还是 keypoint。
- 选择器不用生理、脑电、连续风格和驾驶员 ID。
- 选择器不用 test 结果调参。

## 目前发现

- val 选出的阈值：{summary['selected_threshold']:.2f}
- test 上 keypoint 被选择比例：{summary['test_keypoint_selected_rate']:.6f}

```text
{test_table.to_string(index=False)}
```

## 哪些结果可信

可信的是：这是一个不看 test 调参的初版选择器，可以判断当前可用特征是否足以在 RBF/keypoint 之间做可部署选择。

## 哪些还不能下结论

如果 selector 没超过 RBF，不能说 keypoint 没价值，只能说当前选择特征不够；也不能由此证明生理或风格有效。

## 推荐优先查看

1. `{figures.get('test_metrics', '')}`
2. `{figures.get('threshold_sweep', '')}`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_metrics.csv`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_decisions.csv`
"""
    (REPORT_ROOT / "stage03_vehicle_instability_rbf_keypoint_selector_user_summary_cn.md").write_text(user, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    rows = load_candidate_rows()
    table = build_selector_table(rows)
    pipe, numeric, categorical = fit_selector(table)
    prob = pipe.predict_proba(table[numeric + categorical])[:, 1]
    sweep = threshold_sweep(rows, table, prob)
    val_sweep = sweep[sweep["split"] == "val"].sort_values(["selector_rmse", "keypoint_selected_rate"], ascending=[True, True])
    selected_threshold = float(val_sweep.iloc[0]["threshold"])
    selected_val_rmse = float(val_sweep.iloc[0]["selector_rmse"])
    selector_metrics, decision_table = aggregate_selected(rows, table, prob, selected_threshold)
    selector_rows = choose_rows(rows, table, pd.Series(np.where(prob >= selected_threshold, KEYPOINT_MODEL, RBF_MODEL), index=table.index), SELECTOR_MODEL)
    oracle = oracle_rows(rows)
    all_metrics = build_all_metrics(rows, selector_rows, oracle)
    metric_fig, sweep_fig = plot_outputs(all_metrics, sweep)
    figures = {
        "test_metrics": str(metric_fig).replace("\\", "/"),
        "threshold_sweep": str(sweep_fig).replace("\\", "/"),
    }
    test_decisions = decision_table[decision_table["split"] == "test"].copy()
    summary = {
        "output_version": "stage03_vehicle_instability_rbf_keypoint_selector_v0_1",
        "track_id": TRACK_ID,
        "selector_model": SELECTOR_MODEL,
        "candidate_models": [RBF_MODEL, KEYPOINT_MODEL],
        "selected_threshold": selected_threshold,
        "selected_val_rmse": selected_val_rmse,
        "test_keypoint_selected_rate": float((test_decisions["selected_model"] == KEYPOINT_MODEL).mean()),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "figures": figures,
    }

    table.to_csv(TABLE_DIR / "rbf_keypoint_selector_training_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": numeric, "type": "numeric"}).to_csv(TABLE_DIR / "rbf_keypoint_selector_numeric_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": categorical, "type": "categorical"}).to_csv(TABLE_DIR / "rbf_keypoint_selector_categorical_features.csv", index=False, encoding="utf-8-sig")
    sweep.to_csv(TABLE_DIR / "rbf_keypoint_selector_threshold_sweep.csv", index=False, encoding="utf-8-sig")
    decision_table.to_csv(TABLE_DIR / "rbf_keypoint_selector_decisions.csv", index=False, encoding="utf-8-sig")
    all_metrics.to_csv(TABLE_DIR / "rbf_keypoint_selector_metrics.csv", index=False, encoding="utf-8-sig")
    selector_rows.to_csv(TABLE_DIR / "rbf_keypoint_selector_selected_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    (LOG_DIR / "rbf_keypoint_selector_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(all_metrics, decision_table, sweep, summary, figures)
    print(all_metrics[(all_metrics["split"] == "test") & (all_metrics["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL, SELECTOR_MODEL, ORACLE_MODEL]))].sort_values("rmse_steer").to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
