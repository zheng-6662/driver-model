from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path("F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402


IN_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_rbf_keypoint_selector_v0_1" / "tables"
SOURCE_METRICS = ROOT / "03_baselines" / "stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1" / "tables" / "keypoint_residual_vehicle_transformer_per_sample_metrics.csv"
OUT_ROOT = ROOT / "06_structured_models" / "stage06c_selector_feature_revision_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = ROOT / "09_reports"

TRACK_ID = "B_response3s_strict_core"
RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
KEYPOINT_MODEL = "keypoint_residual_vehicle_transformer_no_subject"
BASE_SELECTOR = "selector_logreg_rbf_keypoint_no_subject"
ORACLE_MODEL = "oracle_best_of_rbf_keypoint_upper_bound"

ORIGINAL_NUMERIC = [
    "anchor_time_rel_s",
    "curvature_anchor",
    "input_valid_ratio",
    f"pred_peak_abs__{RBF_MODEL}",
    f"pred_reversal_count__{RBF_MODEL}",
    f"pred_multi_segment__{RBF_MODEL}",
    f"pred_peak_abs__{KEYPOINT_MODEL}",
    f"pred_reversal_count__{KEYPOINT_MODEL}",
    f"pred_multi_segment__{KEYPOINT_MODEL}",
    "pred_peak_abs__delta_keypoint_minus_rbf",
    "pred_reversal_count__delta_keypoint_minus_rbf",
    "pred_multi_segment__delta_keypoint_minus_rbf",
]
ORIGINAL_CATEGORICAL = [
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
TARGET = "keypoint_better_rmse"


@dataclass
class Candidate:
    name: str
    feature_set: str
    estimator: object


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    training = pd.read_csv(IN_DIR / "rbf_keypoint_selector_training_table.csv")
    source_rows = pd.read_csv(SOURCE_METRICS)
    source_rows = source_rows[
        (source_rows["track_id"] == TRACK_ID) & (source_rows["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL]))
    ].copy()
    if training.empty or source_rows.empty:
        raise RuntimeError("Missing selector training table or source per-sample metrics.")
    return training, source_rows


def add_engineered_features(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    eps = 1e-6
    rbf_peak = out[f"pred_peak_abs__{RBF_MODEL}"].astype(float)
    key_peak = out[f"pred_peak_abs__{KEYPOINT_MODEL}"].astype(float)
    rbf_rev = out[f"pred_reversal_count__{RBF_MODEL}"].astype(float)
    key_rev = out[f"pred_reversal_count__{KEYPOINT_MODEL}"].astype(float)
    rbf_multi = out[f"pred_multi_segment__{RBF_MODEL}"].astype(float)
    key_multi = out[f"pred_multi_segment__{KEYPOINT_MODEL}"].astype(float)
    out["pred_peak_abs__max_candidate"] = np.maximum(rbf_peak, key_peak)
    out["pred_peak_abs__min_candidate"] = np.minimum(rbf_peak, key_peak)
    out["pred_peak_abs__abs_delta"] = np.abs(key_peak - rbf_peak)
    out["pred_peak_abs__ratio_keypoint_over_rbf"] = key_peak / (rbf_peak.abs() + eps)
    out["pred_reversal_count__abs_delta"] = np.abs(key_rev - rbf_rev)
    out["pred_reversal_count__keypoint_lower_than_rbf"] = (key_rev < rbf_rev).astype(int)
    out["pred_reversal_count__keypoint_much_lower"] = ((rbf_rev - key_rev) >= 50).astype(int)
    out["pred_multi_segment__disagree"] = (key_multi != rbf_multi).astype(int)
    out["pred_multi_segment__keypoint_less_complex"] = (key_multi < rbf_multi).astype(int)
    out["candidate_disagreement_score"] = (
        out["pred_peak_abs__abs_delta"].rank(pct=True)
        + out["pred_reversal_count__abs_delta"].rank(pct=True)
        + out["pred_multi_segment__disagree"].astype(float)
    )
    out["context_curve_or_surface"] = out["road_design_module_name"].isin(["curve1", "curve2", "differentmu_road"]).astype(int)
    out["context_middle_section"] = out["road_design_module_name"].eq("middle_section").astype(int)
    return out


def feature_manifest() -> pd.DataFrame:
    rows = []
    for feature in ORIGINAL_NUMERIC:
        rows.append({"feature": feature, "type": "numeric_original", "causal_status": "allowed_candidate_prediction_or_context"})
    for feature in ORIGINAL_CATEGORICAL:
        rows.append({"feature": feature, "type": "categorical_context", "causal_status": "allowed_context"})
    engineered = [
        "pred_peak_abs__max_candidate",
        "pred_peak_abs__min_candidate",
        "pred_peak_abs__abs_delta",
        "pred_peak_abs__ratio_keypoint_over_rbf",
        "pred_reversal_count__abs_delta",
        "pred_reversal_count__keypoint_lower_than_rbf",
        "pred_reversal_count__keypoint_much_lower",
        "pred_multi_segment__disagree",
        "pred_multi_segment__keypoint_less_complex",
        "candidate_disagreement_score",
        "context_curve_or_surface",
        "context_middle_section",
    ]
    for feature in engineered:
        rows.append({"feature": feature, "type": "numeric_engineered", "causal_status": "allowed_derived_from_context_or_candidate_predictions"})
    excluded = [
        "gt_peak_abs",
        "is_large_response",
        "is_difficult_peak_top20",
        "sample_rmse__rbf",
        "sample_rmse__keypoint",
        "wrong_side__*",
        "large_response_recalled__*",
        "tail_drift_risk__*",
        "keypoint_better_rmse",
        "rmse_delta_keypoint_minus_rbf",
    ]
    for feature in excluded:
        rows.append({"feature": feature, "type": "excluded", "causal_status": "future_label_or_training_target_not_used_as_input"})
    return pd.DataFrame(rows)


def candidate_defs() -> list[Candidate]:
    return [
        Candidate(
            "logreg_original_balanced",
            "original",
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260513),
        ),
        Candidate(
            "logreg_engineered_balanced",
            "engineered",
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260513),
        ),
        Candidate(
            "logreg_engineered_conservative",
            "engineered",
            LogisticRegression(max_iter=1000, class_weight={0: 1.0, 1: 0.75}, random_state=20260513),
        ),
        Candidate(
            "rf_engineered_shallow",
            "engineered",
            RandomForestClassifier(
                n_estimators=160,
                max_depth=3,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=20260513,
            ),
        ),
    ]


def feature_columns(table: pd.DataFrame, feature_set: str) -> tuple[list[str], list[str]]:
    numeric = [c for c in ORIGINAL_NUMERIC if c in table.columns]
    categorical = [c for c in ORIGINAL_CATEGORICAL if c in table.columns]
    if feature_set == "engineered":
        numeric += [
            "pred_peak_abs__max_candidate",
            "pred_peak_abs__min_candidate",
            "pred_peak_abs__abs_delta",
            "pred_peak_abs__ratio_keypoint_over_rbf",
            "pred_reversal_count__abs_delta",
            "pred_reversal_count__keypoint_lower_than_rbf",
            "pred_reversal_count__keypoint_much_lower",
            "pred_multi_segment__disagree",
            "pred_multi_segment__keypoint_less_complex",
            "candidate_disagreement_score",
            "context_curve_or_surface",
            "context_middle_section",
        ]
    numeric = [c for c in numeric if c in table.columns]
    return numeric, categorical


def make_pipeline(candidate: Candidate, numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", pre), ("clf", candidate.estimator)])


def choose_rows(source_rows: pd.DataFrame, table: pd.DataFrame, selected_model: np.ndarray, model_name: str) -> pd.DataFrame:
    selected = pd.DataFrame({"sample_id": table["sample_id"].to_numpy(), "selected_model": selected_model})
    pair = source_rows.merge(selected, on="sample_id", how="inner")
    pair = pair[pair["model_name"] == pair["selected_model"]].copy()
    pair["model_name"] = model_name
    pair = pair.drop(columns=["selected_model"])
    return pair


def selector_rmse(source_rows: pd.DataFrame, table: pd.DataFrame, selected_model: np.ndarray, split: str, model_name: str) -> float:
    mask = table["split"].to_numpy() == split
    if not mask.any():
        return float("inf")
    selected = choose_rows(source_rows, table.loc[mask].copy(), selected_model[mask], model_name)
    return float(np.sqrt(np.mean(selected["sample_rmse"].to_numpy(dtype=float) ** 2)))


def evaluate_candidate(source_rows: pd.DataFrame, table: pd.DataFrame, candidate: Candidate) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    numeric, categorical = feature_columns(table, candidate.feature_set)
    train = table[table["split"] == "train"].copy()
    pipe = make_pipeline(candidate, numeric, categorical)
    pipe.fit(train[numeric + categorical], train[TARGET].astype(int))
    prob = pipe.predict_proba(table[numeric + categorical])[:, 1]
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    records = []
    for threshold in thresholds:
        selected_model = np.where(prob >= threshold, KEYPOINT_MODEL, RBF_MODEL)
        for split in ["train", "val", "test"]:
            mask = table["split"].to_numpy() == split
            selected_rate = float((selected_model[mask] == KEYPOINT_MODEL).mean()) if mask.any() else np.nan
            rmse = selector_rmse(source_rows, table, selected_model, split, candidate.name)
            records.append(
                {
                    "candidate_model": candidate.name,
                    "feature_set": candidate.feature_set,
                    "threshold": float(threshold),
                    "split": split,
                    "selector_rmse": rmse,
                    "keypoint_selected_rate": selected_rate,
                }
            )
    sweep = pd.DataFrame(records)
    val = sweep[sweep["split"] == "val"].copy()
    best = val.sort_values(["selector_rmse", "keypoint_selected_rate"], ascending=[True, True]).iloc[0]
    best_threshold = float(best["threshold"])
    best_selected = np.where(prob >= best_threshold, KEYPOINT_MODEL, RBF_MODEL)
    selected_rows = choose_rows(source_rows, table, best_selected, candidate.name)
    metrics = eval_utils.aggregate_metrics(selected_rows)
    metrics["track_id"] = TRACK_ID
    metrics["candidate_model"] = candidate.name
    metrics["feature_set"] = candidate.feature_set
    metrics["selected_threshold_from_val"] = best_threshold
    metrics["val_selector_rmse"] = float(best["selector_rmse"])
    detail = table[
        [
            "sample_id",
            "event_uid",
            "subject",
            "session_stamp",
            "split",
            "road_design_module_name",
            "event_level",
            "road_design_risk_class",
            "gt_peak_abs",
            "is_large_response",
            "is_difficult_peak_top20",
            TARGET,
            "rmse_delta_keypoint_minus_rbf",
        ]
    ].copy()
    detail["candidate_model"] = candidate.name
    detail["selector_prob_keypoint"] = prob
    detail["selected_threshold_from_val"] = best_threshold
    detail["selected_model"] = best_selected
    return metrics, sweep, detail


def add_reference_metrics(source_rows: pd.DataFrame) -> pd.DataFrame:
    refs = source_rows[source_rows["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL])].copy()
    pair = refs.sort_values("sample_rmse").groupby("sample_id", as_index=False).head(1).copy()
    pair["model_name"] = ORACLE_MODEL
    combined = pd.concat([refs, pair], ignore_index=True, sort=False)
    metrics = eval_utils.aggregate_metrics(combined)
    metrics["track_id"] = TRACK_ID
    metrics["candidate_model"] = metrics["model_name"]
    metrics["feature_set"] = "reference"
    metrics["selected_threshold_from_val"] = np.nan
    metrics["val_selector_rmse"] = np.nan
    return metrics


def build_selection_report(table: pd.DataFrame, best_detail: pd.DataFrame) -> pd.DataFrame:
    test = best_detail[best_detail["split"] == "test"].copy()
    test["selected_keypoint"] = test["selected_model"].eq(KEYPOINT_MODEL).astype(int)
    test["oracle_keypoint"] = test[TARGET].astype(int)
    test["selection_outcome"] = np.select(
        [
            (test["selected_keypoint"] == 1) & (test["oracle_keypoint"] == 1),
            (test["selected_keypoint"] == 1) & (test["oracle_keypoint"] == 0),
            (test["selected_keypoint"] == 0) & (test["oracle_keypoint"] == 1),
            (test["selected_keypoint"] == 0) & (test["oracle_keypoint"] == 0),
        ],
        ["TP_select_keypoint_correct", "FP_select_keypoint_hurts", "FN_missed_keypoint_gain", "TN_keep_rbf_correct"],
        default="unknown",
    )
    rows = []
    for outcome, g in test.groupby("selection_outcome"):
        rows.append(
            {
                "selection_outcome": outcome,
                "n_samples": int(len(g)),
                "mean_selector_prob_keypoint": float(g["selector_prob_keypoint"].mean()),
                "mean_keypoint_delta_vs_rbf": float(g["rmse_delta_keypoint_minus_rbf"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame) -> tuple[Path, Path]:
    test = metrics[metrics["split"] == "test"].copy()
    interesting = [
        RBF_MODEL,
        KEYPOINT_MODEL,
        ORACLE_MODEL,
        "logreg_original_balanced",
        "logreg_engineered_balanced",
        "logreg_engineered_conservative",
        "rf_engineered_shallow",
    ]
    test = test[test["model_name"].isin(interesting)].copy()
    order = test.sort_values("rmse_steer")["model_name"].tolist()
    labels = [short_name(x) for x in order]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    vals = test.set_index("model_name").loc[order, "rmse_steer"].astype(float)
    ax.barh(labels, vals, color=["#9ca3af" if "oracle" in x else "#2563eb" if x == RBF_MODEL else "#16a34a" for x in order])
    ax.axvline(float(test[test["model_name"] == RBF_MODEL]["rmse_steer"].iloc[0]), color="#111827", linestyle="--", linewidth=1)
    ax.set_xlabel("test RMSE lower is better")
    ax.set_title("Stage 6c selector revision: test RMSE")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    rmse_fig = FIG_DIR / "selector_revision_test_rmse.png"
    fig.savefig(rmse_fig, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    metrics_to_plot = [
        ("wrong_side_rate", "wrong-side lower"),
        ("large_response_recall", "large recall higher"),
        ("difficult_top20_rmse", "difficult RMSE lower"),
    ]
    for ax, (metric, title) in zip(axes, metrics_to_plot):
        vals = test.set_index("model_name").loc[order, metric].astype(float)
        ax.barh(labels, vals, color="#4b5563")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    phys_fig = FIG_DIR / "selector_revision_physical_metrics.png"
    fig.savefig(phys_fig, dpi=180)
    plt.close(fig)
    return rmse_fig, phys_fig


def short_name(name: str) -> str:
    mapping = {
        RBF_MODEL: "RBF",
        KEYPOINT_MODEL: "Keypoint",
        ORACLE_MODEL: "Oracle",
        "logreg_original_balanced": "LogReg original",
        "logreg_engineered_balanced": "LogReg engineered",
        "logreg_engineered_conservative": "LogReg conservative",
        "rf_engineered_shallow": "RF shallow",
    }
    return mapping.get(name, name)


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


def build_gate(metrics: pd.DataFrame, best_model: str) -> pd.DataFrame:
    test = metrics[metrics["split"] == "test"].set_index("model_name")
    rbf = test.loc[RBF_MODEL]
    best = test.loc[best_model]
    rmse_delta = float(best["rmse_steer"] - rbf["rmse_steer"])
    wrong_delta = float(best["wrong_side_rate"] - rbf["wrong_side_rate"])
    large_delta = float(best["large_response_recall"] - rbf["large_response_recall"])
    difficult_delta = float(best["difficult_top20_rmse"] - rbf["difficult_top20_rmse"])
    status = "continue_candidate" if (rmse_delta <= 0.0 and wrong_delta <= 0.0 and large_delta >= 0.0) else "no_upgrade_current_revision"
    rows = [
        {
            "gate": "selector_revision_test_gain",
            "status": status,
            "evidence": f"{best_model} test RMSE delta vs RBF={rmse_delta:+.6f}, wrong-side delta={wrong_delta:+.3f}, large recall delta={large_delta:+.3f}, difficult RMSE delta={difficult_delta:+.6f}.",
            "decision": "只有同时稳定改善RMSE和物理指标才可升级；否则只作为下一版诊断。",
        },
        {
            "gate": "test_tuning_leakage_guard",
            "status": "pass",
            "evidence": "候选模型只在train拟合，模型和阈值只按val RMSE选择；test只做最终评估。",
            "decision": "本轮选择规则无test调参。",
        },
        {
            "gate": "feature_leakage_guard",
            "status": "pass_protocol",
            "evidence": "输入仅包括上下文和候选模型预测特征；gt_peak_abs、sample_rmse、wrong_side、keypoint_better等未来标签不作为输入。",
            "decision": "当前特征协议可继续扩展，但不能加入未来真实标签。",
        },
        {
            "gate": "stage05_physio_eeg_allowed",
            "status": "blocked",
            "evidence": "selector revision尚未形成稳定可部署车辆-only提升。",
            "decision": "继续阻塞生理/EEG有效性结论。",
        },
    ]
    return pd.DataFrame(rows)


def write_reports(metrics: pd.DataFrame, gate: pd.DataFrame, selection: pd.DataFrame, rmse_fig: Path, phys_fig: Path, best_model: str) -> tuple[Path, Path]:
    test = metrics[metrics["split"] == "test"].set_index("model_name")
    rbf = test.loc[RBF_MODEL]
    best = test.loc[best_model]
    selector_summary = selection.set_index("selection_outcome") if not selection.empty else pd.DataFrame()
    fn = int(selector_summary.loc["FN_missed_keypoint_gain", "n_samples"]) if "FN_missed_keypoint_gain" in selector_summary.index else 0
    fp = int(selector_summary.loc["FP_select_keypoint_hurts", "n_samples"]) if "FP_select_keypoint_hurts" in selector_summary.index else 0
    user = f"""# Stage 6c 用户查看版：selector feature revision v0.1

## 为什么做

Stage 6b 发现 keypoint selector 主要问题是漏选 keypoint 收益样本，同时也有错选 keypoint 伤害样本。本轮尝试在不使用未来真实标签、不使用生理/脑电/风格的前提下，加入候选模型预测差异特征，看看能不能把 oracle 上限转成可部署选择策略。

## 检查了什么

- 原始 logistic selector。
- 增加候选差异特征后的 logistic selector。
- 一个浅层随机森林 selector。
- 所有候选只在 train 拟合，只用 val 选阈值，test 只最终评估。

## 目前发现

- 当前 val 选择的最佳 selector：`{best_model}`。
- RBF test RMSE={rbf['rmse_steer']:.6f}；最佳 selector test RMSE={best['rmse_steer']:.6f}，delta={best['rmse_steer'] - rbf['rmse_steer']:+.6f}。
- 最佳 selector wrong-side={best['wrong_side_rate']:.3f}，RBF wrong-side={rbf['wrong_side_rate']:.3f}。
- 最佳 selector large recall={best['large_response_recall']:.3f}，RBF large recall={rbf['large_response_recall']:.3f}。
- 最佳 selector 仍有 FN={fn}、FP={fp} 类错误。

## 当前判断

如果 gate 表显示 `no_upgrade_current_revision`，说明这版 feature revision 仍不能升级为主线，只能作为下一版可靠性门控的诊断依据。生理/EEG 仍不能进入有效性结论。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_gate_table.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_metrics.csv`
3. `{str(rmse_fig).replace(chr(92), '/')}`
4. `{str(phys_fig).replace(chr(92), '/')}`
"""
    tech = f"""# Stage 6c：selector feature revision v0.1

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

## Gate

{md_table(gate, ['gate', 'status', 'decision'], max_rows=10)}

## Test 指标

{md_table(metrics[metrics['split'] == 'test'].sort_values('rmse_steer'), ['model_name', 'rmse_steer', 'wrong_side_rate', 'large_response_recall', 'difficult_top20_rmse'], max_rows=20)}

## 选择错误摘要

{md_table(selection, ['selection_outcome', 'n_samples', 'mean_selector_prob_keypoint', 'mean_keypoint_delta_vs_rbf'], max_rows=10)}

## 边界

- 不使用 test 调参。
- 不使用生理、脑电、连续风格或驾驶员 ID。
- 不使用未来真实标签作为 selector 输入。
"""
    user_path = REPORT_DIR / "stage06c_selector_feature_revision_user_summary_cn.md"
    tech_path = REPORT_DIR / "stage06c_selector_feature_revision_v0_1_cn.md"
    user_path.write_text(user, encoding="utf-8")
    tech_path.write_text(tech, encoding="utf-8")
    return user_path, tech_path


def main() -> None:
    ensure_dirs()
    table_raw, source_rows = load_tables()
    table = add_engineered_features(table_raw)
    feature_manifest().to_csv(TABLE_DIR / "selector_revision_feature_manifest.csv", index=False, encoding="utf-8-sig")

    metric_frames = [add_reference_metrics(source_rows)]
    sweep_frames = []
    detail_frames = []
    for candidate in candidate_defs():
        metrics, sweep, detail = evaluate_candidate(source_rows, table, candidate)
        metric_frames.append(metrics)
        sweep_frames.append(sweep)
        detail_frames.append(detail)
    metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    sweep = pd.concat(sweep_frames, ignore_index=True, sort=False)
    details = pd.concat(detail_frames, ignore_index=True, sort=False)

    val = metrics[metrics["split"] == "val"].copy()
    val_candidates = val[~val["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL, ORACLE_MODEL])].copy()
    best_model = val_candidates.sort_values(["rmse_steer", "selected_threshold_from_val"], ascending=[True, False]).iloc[0]["model_name"]
    best_detail = details[details["candidate_model"] == best_model].copy()
    selection = build_selection_report(table, best_detail)
    gate = build_gate(metrics, best_model)
    rmse_fig, phys_fig = plot_metrics(metrics)
    user_path, tech_path = write_reports(metrics, gate, selection, rmse_fig, phys_fig, best_model)

    metrics.to_csv(TABLE_DIR / "selector_revision_metrics.csv", index=False, encoding="utf-8-sig")
    sweep.to_csv(TABLE_DIR / "selector_revision_threshold_sweep.csv", index=False, encoding="utf-8-sig")
    details.to_csv(TABLE_DIR / "selector_revision_candidate_details.csv", index=False, encoding="utf-8-sig")
    best_detail.to_csv(TABLE_DIR / "selector_revision_best_detail.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(TABLE_DIR / "selector_revision_best_confusion.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "selector_revision_gate_table.csv", index=False, encoding="utf-8-sig")

    test = metrics[metrics["split"] == "test"].set_index("model_name")
    summary = {
        "output_version": "stage06c_selector_feature_revision_v0_1",
        "run_time_local": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "best_model_selected_by_val": best_model,
        "rbf_test_rmse": float(test.loc[RBF_MODEL, "rmse_steer"]),
        "best_selector_test_rmse": float(test.loc[best_model, "rmse_steer"]),
        "best_selector_delta_vs_rbf_rmse": float(test.loc[best_model, "rmse_steer"] - test.loc[RBF_MODEL, "rmse_steer"]),
        "best_selector_wrong_side_rate": float(test.loc[best_model, "wrong_side_rate"]),
        "rbf_wrong_side_rate": float(test.loc[RBF_MODEL, "wrong_side_rate"]),
        "best_selector_large_recall": float(test.loc[best_model, "large_response_recall"]),
        "rbf_large_recall": float(test.loc[RBF_MODEL, "large_response_recall"]),
        "stage05_physio_eeg_allowed": "blocked",
        "server_used": False,
        "server_credential_file_read": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "uses_subject_id": False,
        "raw_files_modified": False,
        "metrics_path": str(TABLE_DIR / "selector_revision_metrics.csv").replace("\\", "/"),
        "gate_path": str(TABLE_DIR / "selector_revision_gate_table.csv").replace("\\", "/"),
        "user_summary_path": str(user_path).replace("\\", "/"),
        "technical_report_path": str(tech_path).replace("\\", "/"),
    }
    (LOG_DIR / "selector_feature_revision_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
