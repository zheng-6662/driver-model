# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


ROOT = Path("F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
BASELINE_SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
MULTI_SCRIPT_DIR = ROOT / "07_multihypothesis" / "scripts"
for path in [BASELINE_SCRIPT_DIR, MULTI_SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1 as clean_v01  # noqa: E402
import stage03_vehicle_instability_strong_vehicle_baselines_v0_1 as strong_v01  # noqa: E402
from stage07f_response_factorized_candidates_v0_1 import (  # noqa: E402
    RBF_MODEL,
    SPLIT_STRATEGY,
    TRACK_ID,
    add_reference_deltas,
    evaluate_predictions,
)
from stage07g_keypoint_segment_candidates_v0_1 import (  # noqa: E402
    TARGETS,
    blend_with_rbf,
    clip_keypoints,
    fit_target_models,
    keypoints_from_predictions,
    piecewise_from_keypoints,
    rbf_scaled_by_keypoints,
    select_candidate,
    true_keypoints,
)


OUTPUT_VERSION = "stage07j_session_cv_stability_v0_1"
TRACK = "B_response3s_strict_core"
WINDOW_ID = "pre3_label3_response_coverage"
N_OUTER_FOLDS = 5
RANDOM_STATE = 20260513

STAGE7C_FEATURES = ROOT / "07_multihypothesis" / "stage07c_candidate_trajectory_export_v0_1" / "tables" / "candidate_feature_and_label_diagnosis.csv"
STAGE7E_RESPONSE = ROOT / "07_multihypothesis" / "stage07e_candidate_generation_redesign_v0_1" / "tables" / "stage07e_response_label_table.csv"

OUT_ROOT = ROOT / "07_multihypothesis" / OUTPUT_VERSION
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = ROOT / "09_reports"

CONTEXT_FEATURES = [
    "anchor_time_rel_s",
    "curvature_anchor",
    "event_type",
    "event_level",
    "road_type_anchor",
    "road_design_module_name",
    "road_design_instance_name",
    "road_design_risk_class",
    "road_design_mapping_reliability",
]
RBF_SHAPE_FEATURES = [
    "fold_rbf_peak_signed",
    "fold_rbf_peak_abs",
    "fold_rbf_peak_time_s",
    "fold_rbf_onset_time_s",
    "fold_rbf_tail_signed",
    "fold_rbf_reversal_count",
]
CORE_CANDIDATES = [
    "segment_abs_rf_piecewise",
    "segment_resid_rf_piecewise",
    "segment_abs_rf_blend_25",
    "segment_abs_rf_blend_50",
    "segment_resid_rf_blend_25",
    "segment_resid_rf_blend_50",
    "rbf_abs_keypoint_scaled",
    "rbf_resid_keypoint_scaled",
    "rbf_abs_keypoint_scaled_blend_50",
    "rbf_resid_keypoint_scaled_blend_50",
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def path_str(path: Path) -> str:
    return str(path).replace("\\", "/")


def load_base_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(clean_v01.TASK_MANIFEST_PATH)
    y, y_mask, input_values, input_time, label_time, meta = clean_v01.load_track(TRACK, clean_v01.TRACKS[TRACK], manifest)
    feature_source = pd.read_csv(STAGE7C_FEATURES)
    response = pd.read_csv(STAGE7E_RESPONSE)
    sample_ids = meta["sample_id"].astype(str).tolist()
    feature_source = feature_source.set_index("sample_id").loc[sample_ids].reset_index()
    response = response.set_index("sample_id").loc[sample_ids].reset_index()
    return y, y_mask, input_values, input_time, label_time, meta, feature_source, response


def build_session_folds(meta: pd.DataFrame, n_splits: int = N_OUTER_FOLDS) -> tuple[pd.DataFrame, list[np.ndarray]]:
    groups = meta["session_stamp"].astype(str).to_numpy()
    sample_idx = np.arange(len(meta))
    outer = GroupKFold(n_splits=n_splits)
    split_rows: list[pd.DataFrame] = []
    split_arrays: list[np.ndarray] = []
    for fold_id, (trainval_idx, test_idx) in enumerate(outer.split(sample_idx, groups=groups)):
        trainval_groups = groups[trainval_idx]
        inner_n = min(n_splits, len(np.unique(trainval_groups)))
        inner = GroupKFold(n_splits=inner_n)
        inner_splits = list(inner.split(trainval_idx, groups=trainval_groups))
        train_rel, val_rel = inner_splits[fold_id % len(inner_splits)]
        train_idx = trainval_idx[train_rel]
        val_idx = trainval_idx[val_rel]
        split = np.array(["train"] * len(meta), dtype=object)
        split[val_idx] = "val"
        split[test_idx] = "test"
        split_arrays.append(split)
        split_rows.append(
            pd.DataFrame(
                {
                    "fold_id": fold_id,
                    "sample_id": meta["sample_id"].astype(str),
                    "subject": meta["subject"].astype(str),
                    "session_stamp": meta["session_stamp"].astype(str),
                    "split": split,
                }
            )
        )
    return pd.concat(split_rows, ignore_index=True), split_arrays


def count_reversals(pred: np.ndarray, mask: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    out = np.zeros(pred.shape[0], dtype=int)
    for i in range(pred.shape[0]):
        vals = pred[i, mask[i] & np.isfinite(pred[i])]
        vals = vals[np.abs(vals) >= threshold]
        if len(vals) < 2:
            continue
        signs = np.sign(vals)
        signs = signs[signs != 0]
        if len(signs) < 2:
            continue
        out[i] = int(np.sum(signs[1:] != signs[:-1]))
    return out


def fit_fold_rbf(
    y: np.ndarray,
    y_mask: np.ndarray,
    input_values: np.ndarray,
    input_time: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    strong_v01.WINDOW_ID = WINDOW_ID
    strong_v01.SPLIT_STRATEGY = SPLIT_STRATEGY
    x_ctx, _ = strong_v01.build_rich_vehicle_features(input_values, input_time, meta, train_idx, include_context=True)
    x_ctx_scaled, _ = strong_v01.standardize_train_only(x_ctx, train_idx)
    x_dist, _ = strong_v01.make_distance_features(x_ctx_scaled, train_idx, n_components=min(96, max(8, x_ctx_scaled.shape[1])))
    pred, info = strong_v01.fit_rbf_kernel_ridge_direct(x_dist, y, train_idx, val_idx, y_mask)
    return pred.astype(np.float32), info


def build_fold_feature_table(
    feature_source: pd.DataFrame,
    response: pd.DataFrame,
    split: np.ndarray,
    rbf_kp: pd.DataFrame,
    rbf_reversal: np.ndarray,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    features = feature_source[["sample_id", *[c for c in CONTEXT_FEATURES if c in feature_source.columns]]].copy()
    features["split"] = split
    for col in CONTEXT_FEATURES:
        if col not in features.columns:
            features[col] = "missing"
    features["fold_rbf_peak_signed"] = rbf_kp["peak_signed"].astype(float).to_numpy()
    features["fold_rbf_peak_abs"] = np.abs(features["fold_rbf_peak_signed"].to_numpy(dtype=float))
    features["fold_rbf_peak_time_s"] = rbf_kp["peak_time_s"].astype(float).to_numpy()
    features["fold_rbf_onset_time_s"] = rbf_kp["onset_time_s"].astype(float).to_numpy()
    features["fold_rbf_tail_signed"] = rbf_kp["tail_signed"].astype(float).to_numpy()
    features["fold_rbf_reversal_count"] = rbf_reversal.astype(float)
    allowed = [c for c in [*CONTEXT_FEATURES, *RBF_SHAPE_FEATURES] if c in features.columns]
    audit_rows = []
    for col in feature_source.columns:
        if col in allowed:
            status, reason = "allowed", "pre_event_context_or_fold_retrained_rbf_shape"
        elif col in {"sample_id", "event_uid", "subject", "session_stamp", "split"}:
            status, reason = "excluded", "identifier_or_split"
        elif "pred_" in col or col.startswith("top") or col.startswith("label_diag__") or "oracle" in col.lower():
            status, reason = "excluded", "fixed_split_candidate_prediction_or_label_diagnostic_leakage_risk"
        else:
            status, reason = "excluded", "not_used_in_strict_cv_variant"
        audit_rows.append({"feature": col, "input_status": status, "reason": reason})
    for col in RBF_SHAPE_FEATURES:
        audit_rows.append({"feature": col, "input_status": "allowed", "reason": "fold_retrained_rbf_prediction_shape"})
    return features, allowed, pd.DataFrame(audit_rows)


def build_keypoint_predictions_for_fold(
    rbf: np.ndarray,
    response: pd.DataFrame,
    keypoint_pred: pd.DataFrame,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    train_true = true_keypoints(response).loc[train_mask].reset_index(drop=True)
    abs_kp = clip_keypoints(
        keypoint_pred[[f"abs_{target}" for target in TARGETS]].rename(columns={f"abs_{target}": target for target in TARGETS}),
        train_true,
    )
    resid_kp = clip_keypoints(
        keypoint_pred[[f"resid_{target}" for target in TARGETS]].rename(columns={f"resid_{target}": target for target in TARGETS}),
        train_true,
    )
    oracle_kp = true_keypoints(response)
    predictions: dict[str, np.ndarray] = {RBF_MODEL: rbf.astype(np.float32)}
    abs_piece = piecewise_from_keypoints(abs_kp, label_time)
    resid_piece = piecewise_from_keypoints(resid_kp, label_time)
    abs_scaled = rbf_scaled_by_keypoints(rbf, abs_kp, label_time)
    resid_scaled = rbf_scaled_by_keypoints(rbf, resid_kp, label_time)
    predictions.update(
        {
            "segment_abs_rf_piecewise": abs_piece,
            "segment_resid_rf_piecewise": resid_piece,
            "segment_abs_rf_blend_25": blend_with_rbf(rbf, abs_piece, 0.25),
            "segment_abs_rf_blend_50": blend_with_rbf(rbf, abs_piece, 0.50),
            "segment_resid_rf_blend_25": blend_with_rbf(rbf, resid_piece, 0.25),
            "segment_resid_rf_blend_50": blend_with_rbf(rbf, resid_piece, 0.50),
            "rbf_abs_keypoint_scaled": abs_scaled,
            "rbf_resid_keypoint_scaled": resid_scaled,
            "rbf_abs_keypoint_scaled_blend_50": blend_with_rbf(rbf, abs_scaled, 0.50),
            "rbf_resid_keypoint_scaled_blend_50": blend_with_rbf(rbf, resid_scaled, 0.50),
            "keypoint_segment_oracle_piecewise": piecewise_from_keypoints(oracle_kp, label_time),
        }
    )
    return predictions


def score_candidates(metrics: pd.DataFrame, fold_id: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    indexed = metrics.set_index(["model_name", "split"])
    for model in CORE_CANDIDATES:
        rec: dict[str, Any] = {"fold_id": fold_id, "model_name": model}
        ok = True
        for split_name in ["train", "val", "test"]:
            if (model, split_name) not in indexed.index:
                ok = False
                break
            row = indexed.loc[(model, split_name)]
            for col in [
                "rmse_steer",
                "rmse_delta_vs_rbf",
                "wrong_side_rate",
                "wrong_side_delta_vs_rbf",
                "large_response_recall",
                "large_recall_delta_vs_rbf",
                "difficult_top20_rmse",
                "difficult_rmse_delta_vs_rbf",
            ]:
                rec[f"{col}_{split_name}"] = float(row[col])
        if not ok:
            continue
        rec["abs_train_val_delta_gap"] = abs(rec["rmse_delta_vs_rbf_val"] - rec["rmse_delta_vs_rbf_train"])
        rec["score_stability_l05"] = rec["rmse_delta_vs_rbf_val"] + 0.5 * rec["abs_train_val_delta_gap"]
        rec["score_stability_l10"] = rec["rmse_delta_vs_rbf_val"] + 1.0 * rec["abs_train_val_delta_gap"]
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["fold_id", "score_stability_l05", "rmse_delta_vs_rbf_val"]).reset_index(drop=True)


def policy_rows(metrics: pd.DataFrame, fold_id: int, score_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_selected, selection_table, original_reason = select_candidate(metrics)
    fold_scores = score_table[score_table["fold_id"].eq(fold_id)].copy()
    stability_selected = str(fold_scores.sort_values(["score_stability_l05", "rmse_delta_vs_rbf_val"]).iloc[0]["model_name"])
    policies = [
        ("always_rbf_reference", RBF_MODEL, "reference"),
        ("stage7g_original_val_gate", original_selected, original_reason),
        ("stability_penalty_l05", stability_selected, "score=val_delta+0.5*abs(train_delta-val_delta)"),
    ]
    indexed = metrics.set_index(["model_name", "split"])
    rows: list[dict[str, Any]] = []
    for policy_name, model_name, rule in policies:
        for split_name in ["train", "val", "test"]:
            row = indexed.loc[(model_name, split_name)].to_dict()
            row.update(
                {
                    "fold_id": fold_id,
                    "model_name": model_name,
                    "split": split_name,
                    "policy_name": policy_name,
                    "selected_model": model_name,
                    "selection_rule": rule,
                }
            )
            rows.append(row)
    selection_table = selection_table.copy()
    selection_table["fold_id"] = fold_id
    selection_table["original_selected_model"] = original_selected
    return pd.DataFrame(rows), selection_table


def run_fold(
    fold_id: int,
    split: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    input_values: np.ndarray,
    input_time: np.ndarray,
    label_time: np.ndarray,
    base_meta: pd.DataFrame,
    feature_source: pd.DataFrame,
    response_base: pd.DataFrame,
) -> dict[str, Any]:
    meta = base_meta.copy()
    meta[SPLIT_STRATEGY] = split
    response = response_base.copy()
    response["split"] = split
    features0 = feature_source.copy()
    features0["split"] = split
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    train_mask = split == "train"

    rbf_pred, rbf_info = fit_fold_rbf(y, y_mask, input_values, input_time, meta, train_idx, val_idx)
    rbf_kp = keypoints_from_predictions(rbf_pred, y_mask, label_time)
    rbf_reversal = count_reversals(rbf_pred, y_mask)
    features, allowed, feature_audit = build_fold_feature_table(features0, response, split, rbf_kp, rbf_reversal)
    true_kp = true_keypoints(response)
    residual_targets = true_kp - rbf_kp
    _, abs_pred = fit_target_models(features, allowed, true_kp, train_mask, "abs", "rf")
    _, resid_delta = fit_target_models(features, allowed, residual_targets, train_mask, "delta", "extra")
    resid_pred = pd.DataFrame({"sample_id": response["sample_id"].astype(str), "split": response["split"].astype(str)})
    for target in TARGETS:
        resid_pred[f"resid_{target}"] = rbf_kp[target].astype(float).to_numpy() + resid_delta[f"delta_{target}"].astype(float).to_numpy()
    keypoint_pred = pd.concat([abs_pred, resid_pred.drop(columns=["sample_id", "split"])], axis=1)
    predictions = build_keypoint_predictions_for_fold(rbf_pred, response, keypoint_pred, y_mask, label_time, train_mask)

    eval_meta = meta[["sample_id", "event_uid", "subject", "session_stamp", SPLIT_STRATEGY]].copy()
    metrics, per_sample = evaluate_predictions(y, y_mask, label_time, eval_meta, train_idx, predictions)
    metrics = add_reference_deltas(metrics)
    metrics["fold_id"] = fold_id
    per_sample["fold_id"] = fold_id
    target_metric = pd.DataFrame()
    score_table = score_candidates(metrics, fold_id)
    policy_metric, selection_table = policy_rows(metrics, fold_id, score_table)
    for df in [feature_audit, keypoint_pred, policy_metric, selection_table]:
        df["fold_id"] = fold_id
    rbf_info = dict(rbf_info)
    rbf_info["fold_id"] = fold_id
    rbf_info["train_n"] = int((split == "train").sum())
    rbf_info["val_n"] = int((split == "val").sum())
    rbf_info["test_n"] = int((split == "test").sum())
    rbf_info["feature_protocol"] = "retrained_rbf_context_only"
    return {
        "metrics": metrics,
        "per_sample": per_sample,
        "score_table": score_table,
        "policy_metric": policy_metric,
        "selection_table": selection_table,
        "feature_audit": feature_audit,
        "rbf_info": pd.DataFrame([rbf_info]),
        "allowed_features": pd.DataFrame({"fold_id": fold_id, "feature": allowed}),
        "keypoint_pred": keypoint_pred,
        "target_metric": target_metric,
    }


def aggregate_policy(policy_metric: pd.DataFrame) -> pd.DataFrame:
    test = policy_metric[policy_metric["split"].eq("test")].copy()
    rows: list[dict[str, Any]] = []
    for policy, grp in test.groupby("policy_name"):
        rows.append(
            {
                "policy_name": policy,
                "n_folds": int(grp["fold_id"].nunique()),
                "mean_test_rmse": float(grp["rmse_steer"].mean()),
                "mean_test_delta_vs_rbf": float(grp["rmse_delta_vs_rbf"].mean()),
                "median_test_delta_vs_rbf": float(grp["rmse_delta_vs_rbf"].median()),
                "std_test_delta_vs_rbf": float(grp["rmse_delta_vs_rbf"].std(ddof=0)),
                "improved_fold_count": int((grp["rmse_delta_vs_rbf"] < -1e-6).sum()),
                "improved_fold_rate": float((grp["rmse_delta_vs_rbf"] < -1e-6).mean()),
                "mean_wrong_side_delta": float(grp["wrong_side_delta_vs_rbf"].mean()),
                "mean_large_recall_delta": float(grp["large_recall_delta_vs_rbf"].mean()),
                "mean_difficult_delta": float(grp["difficult_rmse_delta_vs_rbf"].mean()),
                "difficult_improved_fold_rate": float((grp["difficult_rmse_delta_vs_rbf"] < -1e-6).mean()),
                "selected_models": ", ".join(sorted(grp["selected_model"].astype(str).unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_test_delta_vs_rbf", "mean_difficult_delta"])


def build_gate(aggregate: pd.DataFrame) -> pd.DataFrame:
    row = aggregate[aggregate["policy_name"].eq("stability_penalty_l05")].iloc[0]
    mean_delta = float(row["mean_test_delta_vs_rbf"])
    fold_rate = float(row["improved_fold_rate"])
    difficult_rate = float(row["difficult_improved_fold_rate"])
    weak = mean_delta < -1e-6 and fold_rate >= 0.60 and difficult_rate >= 0.60
    return pd.DataFrame(
        [
            {
                "gate_item": "cv_feature_protocol",
                "status": "strict_retrained_rbf_context_only",
                "evidence": "RBF was retrained per fold; fixed-split topK/Transformer candidate-prediction features were excluded to avoid leakage.",
            },
            {
                "gate_item": "stability_policy_cv_result",
                "status": "weak_candidate_continue" if weak else "no_upgrade",
                "evidence": f"mean test delta={mean_delta:+.6f}; improved fold rate={fold_rate:.3f}; difficult improved fold rate={difficult_rate:.3f}",
            },
            {
                "gate_item": "mainline_upgrade",
                "status": "not_final",
                "evidence": "Even a positive CV result would still need full upstream candidate retraining and fixed-plot review before freezing a mainline.",
            },
            {
                "gate_item": "stage08_physio_eeg_allowed",
                "status": "blocked",
                "evidence": "Vehicle-only candidate stability is still under validation; no physio/EEG evidence is evaluated here.",
            },
            {
                "gate_item": "server_used",
                "status": "no",
                "evidence": "Local CPU diagnostic run only; credential file not read.",
            },
        ]
    )


def plot_fold_deltas(policy_metric: pd.DataFrame, path: Path) -> None:
    test = policy_metric[policy_metric["split"].eq("test") & policy_metric["policy_name"].ne("always_rbf_reference")].copy()
    piv = test.pivot(index="fold_id", columns="policy_name", values="rmse_delta_vs_rbf").sort_index()
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    x = np.arange(len(piv))
    width = 0.36
    for i, col in enumerate(piv.columns):
        ax.bar(x + (i - 0.5) * width, piv[col].to_numpy(dtype=float), width=width, label=col)
    ax.axhline(0, color="#999999", linewidth=0.9)
    ax.set_xticks(x, [f"fold {i}" for i in piv.index])
    ax.set_ylabel("test RMSE delta vs fold RBF")
    ax.set_title("Stage 7j session-CV policy deltas")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_selected_counts(policy_metric: pd.DataFrame, path: Path) -> None:
    test = policy_metric[policy_metric["split"].eq("test")].copy()
    stab = test[test["policy_name"].eq("stability_penalty_l05")]
    counts = stab["selected_model"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.barh(np.arange(len(counts)), counts.to_numpy(), color="#4c78a8")
    ax.set_yticks(np.arange(len(counts)), [x.replace("_", " ") for x in counts.index], fontsize=8)
    ax.set_xlabel("fold count")
    ax.set_title("Stage 7j stability-policy selected models")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_score_val_test(score_table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for model, grp in score_table.groupby("model_name"):
        ax.scatter(grp["rmse_delta_vs_rbf_val"], grp["rmse_delta_vs_rbf_test"], s=22, alpha=0.72, label=model if model in {"segment_resid_rf_blend_25", "segment_abs_rf_blend_25", "rbf_resid_keypoint_scaled"} else None)
    ax.axhline(0, color="#999999", linewidth=0.9)
    ax.axvline(0, color="#999999", linewidth=0.9)
    ax.set_xlabel("val RMSE delta vs fold RBF")
    ax.set_ylabel("test RMSE delta vs fold RBF")
    ax.set_title("Stage 7j candidate val/test deltas")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_reports(aggregate: pd.DataFrame, gate: pd.DataFrame, figures: dict[str, str]) -> None:
    agg_text = aggregate.to_string(index=False)
    gate_text = gate.to_string(index=False)
    stab = aggregate[aggregate["policy_name"].eq("stability_penalty_l05")].iloc[0]
    gate_status = str(gate.set_index("gate_item").loc["stability_policy_cv_result", "status"])
    user = f"""# Stage 7j 用户查看版：session 多折稳定性验证 v0.1

## 这个阶段为什么做

Stage 7i 只在一个固定 split 上显示 `segment_resid_rf_blend_25` 有弱收益。这个结果不能直接升级，因为可能只是当前 validation/test 划分偶然有利。Stage 7j 用 session 分组做 5 折复核，检查稳定选择规则是否能跨 session 复现。

## 这个阶段检查了什么

- 每一折重新训练 RBF/KNN 基座，避免把固定 split 的 RBF 预测直接搬到新折里。
- 只允许事件前车辆/道路上下文，以及该折重训 RBF 得到的预测形态特征。
- 明确排除固定 split 训练出来的 top-K/Transformer/keypoint 预测特征，因为它们在新折里会有训练信息泄漏风险。
- 用 `stability_penalty_l05` 和原始 Stage 7g val gate 做对照，test 只做最终报告。

## 目前发现了什么

- gate={gate_status}。
- `stability_penalty_l05` 平均 test delta vs fold RBF={float(stab["mean_test_delta_vs_rbf"]):+.6f}。
- improved fold rate={float(stab["improved_fold_rate"]):.3f}。
- difficult improved fold rate={float(stab["difficult_improved_fold_rate"]):.3f}。
- 选中的模型集合：{stab["selected_models"]}。

## 多折汇总

```text
{agg_text}
```

## gate

```text
{gate_text}
```

## 哪些结果可信

可信的是：这轮没有用生理、脑电、连续风格、驾驶员 ID，也没有用固定 split 的 top-K/Transformer 预测特征；RBF 基座每折重训，标准化和特征选择只用对应 train split。

## 哪些结果还不能下结论

这仍然不是完整最终主线验证。因为 Stage7i 原模型使用过固定 split 的候选预测特征，而本轮为了避免泄漏只保留了重训 RBF 形态特征；因此 Stage7j 更像严格稳定性审计，而不是完整复刻所有上游候选模型。

## 下一阶段是否可以继续

如果 gate 仍是 `no_upgrade`，应回到候选生成或 split 设计，不进入生理/EEG。如果 gate 是 `weak_candidate_continue`，也只能进入更完整的上游重训/固定图复核，不能直接宣称最终升级。

## 推荐优先查看

1. `{figures["fold_deltas"]}`
2. `{figures["selected_counts"]}`
3. `{figures["val_test_scatter"]}`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_policy_aggregate.csv`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_gate_table.csv`
"""
    (REPORT_DIR / "stage07j_session_cv_stability_user_summary_cn.md").write_text(user, encoding="utf-8-sig")
    tech = f"""# Stage 7j 技术报告：session-CV stability audit v0.1

## Scope

- 5-fold grouped by `session_stamp`.
- Per-fold RBF retraining.
- Feature protocol: event/road context + fold-retrained RBF shape features only.
- Fixed-split topK/Transformer/keypoint prediction features excluded due leakage risk under new folds.
- No physio, EEG, continuous style or subject ID.
- No server used. Credential file not read.

## Aggregate

```text
{agg_text}
```

## Gate

```text
{gate_text}
```
"""
    (REPORT_DIR / "stage07j_session_cv_stability_v0_1_cn.md").write_text(tech, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    y, y_mask, input_values, input_time, label_time, meta, feature_source, response = load_base_data()
    split_table, split_arrays = build_session_folds(meta)
    all_parts: dict[str, list[pd.DataFrame]] = {
        "metrics": [],
        "per_sample": [],
        "score_table": [],
        "policy_metric": [],
        "selection_table": [],
        "feature_audit": [],
        "rbf_info": [],
        "allowed_features": [],
    }
    for fold_id, split in enumerate(split_arrays):
        print(f"[stage07j] running fold {fold_id}: train={(split == 'train').sum()} val={(split == 'val').sum()} test={(split == 'test').sum()}", flush=True)
        result = run_fold(fold_id, split, y, y_mask, input_values, input_time, label_time, meta, feature_source, response)
        for key in all_parts:
            all_parts[key].append(result[key])
    outputs = {key: pd.concat(parts, ignore_index=True) for key, parts in all_parts.items()}
    aggregate = aggregate_policy(outputs["policy_metric"])
    gate = build_gate(aggregate)
    figures = {
        "fold_deltas": path_str(FIG_DIR / "stage07j_policy_fold_deltas.png"),
        "selected_counts": path_str(FIG_DIR / "stage07j_selected_model_counts.png"),
        "val_test_scatter": path_str(FIG_DIR / "stage07j_candidate_val_test_delta_scatter.png"),
    }
    plot_fold_deltas(outputs["policy_metric"], Path(figures["fold_deltas"]))
    plot_selected_counts(outputs["policy_metric"], Path(figures["selected_counts"]))
    plot_score_val_test(outputs["score_table"], Path(figures["val_test_scatter"]))

    split_table.to_csv(TABLE_DIR / "stage07j_session_cv_split_table.csv", index=False, encoding="utf-8-sig")
    outputs["metrics"].to_csv(TABLE_DIR / "stage07j_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    outputs["per_sample"].to_csv(TABLE_DIR / "stage07j_candidate_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    outputs["score_table"].to_csv(TABLE_DIR / "stage07j_candidate_score_table.csv", index=False, encoding="utf-8-sig")
    outputs["policy_metric"].to_csv(TABLE_DIR / "stage07j_policy_fold_metrics.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLE_DIR / "stage07j_policy_aggregate.csv", index=False, encoding="utf-8-sig")
    outputs["selection_table"].to_csv(TABLE_DIR / "stage07j_original_val_gate_selection_table.csv", index=False, encoding="utf-8-sig")
    outputs["feature_audit"].drop_duplicates(["feature", "input_status", "reason"]).to_csv(TABLE_DIR / "stage07j_feature_audit.csv", index=False, encoding="utf-8-sig")
    outputs["allowed_features"].drop_duplicates("feature").to_csv(TABLE_DIR / "stage07j_allowed_features.csv", index=False, encoding="utf-8-sig")
    outputs["rbf_info"].to_csv(TABLE_DIR / "stage07j_fold_rbf_fit_info.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "stage07j_gate_table.csv", index=False, encoding="utf-8-sig")
    write_reports(aggregate, gate, figures)

    stab = aggregate[aggregate["policy_name"].eq("stability_penalty_l05")].iloc[0]
    summary = {
        "output_version": OUTPUT_VERSION,
        "track_id": TRACK_ID,
        "cv_track": TRACK,
        "n_outer_folds": N_OUTER_FOLDS,
        "feature_protocol": "strict_retrained_rbf_context_only",
        "selected_policy": "stability_penalty_l05",
        "mean_test_delta_vs_rbf": float(stab["mean_test_delta_vs_rbf"]),
        "improved_fold_rate": float(stab["improved_fold_rate"]),
        "difficult_improved_fold_rate": float(stab["difficult_improved_fold_rate"]),
        "gate_status": str(gate.set_index("gate_item").loc["stability_policy_cv_result", "status"]),
        "mainline_upgrade": "not_final",
        "stage08_physio_eeg_allowed": False,
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "uses_subject_id": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "figures": figures,
    }
    (LOG_DIR / "stage07j_session_cv_stability_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
