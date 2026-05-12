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


ROOT = Path("F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
BASELINE_SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(BASELINE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402


OUTPUT_VERSION = "stage07e_candidate_generation_redesign_v0_1"
TRACK_ID = "B_response3s_strict_core"
SPLIT_STRATEGY = "session_level_split"

STAGE7C_ROOT = ROOT / "07_multihypothesis" / "stage07c_candidate_trajectory_export_v0_1"
STAGE7D_ROOT = ROOT / "07_multihypothesis" / "stage07d_non_oracle_selector_v0_2"
TRAJECTORY_NPZ = STAGE7C_ROOT / "arrays" / "stage07c_candidate_trajectories.npz"
FEATURE_DIAG = STAGE7C_ROOT / "tables" / "candidate_feature_and_label_diagnosis.csv"
STAGE7D_GATE = STAGE7D_ROOT / "tables" / "stage07d_gate_table.csv"

OUT_ROOT = ROOT / "07_multihypothesis" / OUTPUT_VERSION
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = ROOT / "09_reports"

RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
KEYPOINT_MODEL = "keypoint_residual_vehicle_transformer_no_subject"
BRANCH_MODELS = [f"topk_vehicle_transformer_branch{k}_no_subject" for k in range(3)]
DEPLOYABLE_CANDIDATES = [RBF_MODEL, KEYPOINT_MODEL, *BRANCH_MODELS]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def path_str(path: Path) -> str:
    return str(path).replace("\\", "/")


def load_inputs() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if not TRAJECTORY_NPZ.exists():
        raise FileNotFoundError(TRAJECTORY_NPZ)
    if not FEATURE_DIAG.exists():
        raise FileNotFoundError(FEATURE_DIAG)
    if not STAGE7D_GATE.exists():
        raise FileNotFoundError(STAGE7D_GATE)
    z = np.load(TRAJECTORY_NPZ, allow_pickle=True)
    feature_diag = pd.read_csv(FEATURE_DIAG)
    stage7d_gate = pd.read_csv(STAGE7D_GATE)
    return dict(z), feature_diag, stage7d_gate


def sample_rmse_array(y_true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(y_true) & np.isfinite(pred)
    diff = np.where(valid, pred - y_true, np.nan)
    denom = np.maximum(valid.sum(axis=1), 1)
    return np.sqrt(np.nansum(diff * diff, axis=1) / denom).astype(np.float32)


def first_crossing(arr: np.ndarray, thr: float) -> int:
    idx = np.where(np.abs(arr) >= thr)[0]
    return int(idx[0]) if idx.size else -1


def response_rows(y: np.ndarray, y_mask: np.ndarray, label_time: np.ndarray, sample_ids: np.ndarray, split: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    peak_abs_all = np.nanmax(np.abs(np.where(y_mask, y, np.nan)), axis=1)
    train_peak = peak_abs_all[split == "train"]
    small_thr = float(np.nanpercentile(train_peak, 40))
    large_thr = float(np.nanpercentile(train_peak, 75))
    very_large_thr = float(np.nanpercentile(train_peak, 90))

    peak_times: list[float] = []
    for i in range(y.shape[0]):
        valid = y_mask[i] & np.isfinite(y[i])
        arr = np.where(valid, y[i], np.nan)
        idx = int(np.nanargmax(np.abs(np.nan_to_num(arr, nan=0.0))))
        peak_times.append(float(label_time[idx]))
    peak_time_thr = float(np.nanmedian(np.asarray(peak_times)[split == "train"]))

    for i in range(y.shape[0]):
        valid = y_mask[i] & np.isfinite(y[i])
        arr = np.where(valid, y[i], np.nan)
        clean = np.nan_to_num(arr, nan=0.0)
        peak_idx = int(np.nanargmax(np.abs(clean)))
        peak_signed = float(clean[peak_idx])
        peak_abs = abs(peak_signed)
        peak_time = float(label_time[peak_idx])
        direction = "positive" if peak_signed >= 0 else "negative"
        if peak_abs < small_thr:
            amplitude_mode = "small"
        elif peak_abs < large_thr:
            amplitude_mode = "medium"
        elif peak_abs < very_large_thr:
            amplitude_mode = "large"
        else:
            amplitude_mode = "very_large"

        onset_thr = max(0.015, 0.2 * max(peak_abs, 1e-6))
        onset_idx = first_crossing(clean, onset_thr)
        onset_time = float(label_time[onset_idx]) if onset_idx >= 0 else float(label_time[-1])
        tail_signed = float(arr[valid][-1]) if valid.any() else 0.0
        tail_abs_ratio = abs(tail_signed) / max(peak_abs, 1e-6)
        if tail_abs_ratio <= 0.30:
            tail_mode = "return_near_zero"
        elif np.sign(tail_signed) == np.sign(peak_signed):
            tail_mode = "same_side_tail"
        else:
            tail_mode = "opposite_tail"

        rev_count = int(eval_utils.reversal_count(arr))
        zero_crossing = int(eval_utils.zero_crossing_has(arr))
        if rev_count >= 2:
            correction_mode = "multi_segment"
        elif zero_crossing:
            correction_mode = "zero_crossing"
        elif rev_count == 1:
            correction_mode = "single_reversal"
        else:
            correction_mode = "single_sweep"
        peak_timing = "early_peak" if peak_time <= peak_time_thr else "late_peak"
        response_family = f"{amplitude_mode}|{tail_mode}|{correction_mode}"
        rows.append(
            {
                "sample_id": str(sample_ids[i]),
                "split": str(split[i]),
                "gt_peak_abs": peak_abs,
                "gt_peak_signed": peak_signed,
                "gt_peak_time_s": peak_time,
                "gt_onset_time_s": onset_time,
                "gt_tail_signed": tail_signed,
                "gt_tail_abs_ratio": tail_abs_ratio,
                "direction_mode": direction,
                "amplitude_mode": amplitude_mode,
                "peak_timing": peak_timing,
                "tail_mode": tail_mode,
                "reversal_count": rev_count,
                "zero_crossing": zero_crossing,
                "correction_mode": correction_mode,
                "response_family": response_family,
                "threshold_small_peak_train_p40": small_thr,
                "threshold_large_peak_train_p75": large_thr,
                "threshold_very_large_peak_train_p90": very_large_thr,
                "threshold_peak_time_train_median": peak_time_thr,
            }
        )
    return pd.DataFrame(rows)


def candidate_gap_rows(z: dict[str, Any], response: pd.DataFrame, feature_diag: pd.DataFrame) -> pd.DataFrame:
    names = [str(x) for x in z["candidate_model_names"].tolist()]
    pred = z["candidate_predictions"].astype(np.float32)
    y = z["y_true"].astype(np.float32)
    y_mask = z["y_mask"].astype(bool)
    idx = {name: names.index(name) for name in DEPLOYABLE_CANDIDATES}
    candidate_rmse = {}
    for name in DEPLOYABLE_CANDIDATES:
        candidate_rmse[name] = sample_rmse_array(y, pred[:, idx[name], :], y_mask)
    rmse_mat = np.stack([candidate_rmse[name] for name in DEPLOYABLE_CANDIDATES], axis=1)
    best_idx = np.nanargmin(rmse_mat, axis=1)
    best_model = [DEPLOYABLE_CANDIDATES[int(i)] for i in best_idx]
    candidate_stack = np.stack([pred[:, idx[name], :] for name in DEPLOYABLE_CANDIDATES], axis=1)
    candidate_spread = np.nanstd(candidate_stack, axis=1)
    rbf_rmse = candidate_rmse[RBF_MODEL]
    oracle_rmse = rmse_mat[np.arange(len(best_idx)), best_idx]
    gap = response.copy()
    gap["rbf_sample_rmse"] = rbf_rmse
    gap["deployable_oracle_rmse"] = oracle_rmse
    gap["deployable_oracle_gain_over_rbf"] = rbf_rmse - oracle_rmse
    gap["deployable_oracle_best_model"] = best_model
    gap["deployable_oracle_uses_non_rbf"] = (np.asarray(best_model) != RBF_MODEL).astype(int)
    gap["candidate_spread_mean"] = np.nanmean(candidate_spread, axis=1)
    gap["candidate_spread_peak"] = np.nanmax(candidate_spread, axis=1)
    for name in DEPLOYABLE_CANDIDATES:
        gap[f"{name}__sample_rmse"] = candidate_rmse[name]
    meta_cols = [
        "sample_id",
        "event_uid",
        "subject",
        "session_stamp",
        "event_type",
        "event_level",
        "road_type_anchor",
        "road_design_module_name",
        "road_design_instance_name",
        "road_design_risk_class",
        "road_design_mapping_reliability",
    ]
    meta_cols = [c for c in meta_cols if c in feature_diag.columns]
    gap = gap.merge(feature_diag[meta_cols].drop_duplicates("sample_id"), on="sample_id", how="left")
    return gap


def grouped_rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values.astype(float))))) if len(values) else float("nan")


def status_for_group(row: pd.Series) -> str:
    if int(row["n_samples"]) < 5:
        return "low_n_interpret_carefully"
    if float(row["deployable_oracle_rmse"]) > 0.65 and float(row["mean_gain_over_rbf"]) < 0.04:
        return "candidate_generation_gap"
    if float(row["mean_gain_over_rbf"]) >= 0.06 and float(row["non_rbf_oracle_rate"]) >= 0.25:
        return "selector_gap_candidate_pool_has_signal"
    if abs(float(row["deployable_oracle_rmse"]) - float(row["rbf_rmse"])) < 0.02:
        return "rbf_sufficient_or_low_candidate_gain"
    return "mixed_gap"


def coverage_by_bucket(gap: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    bucket_cols = [
        "direction_mode",
        "amplitude_mode",
        "peak_timing",
        "tail_mode",
        "correction_mode",
        "response_family",
        "event_level",
        "road_design_risk_class",
    ]
    for bucket_col in bucket_cols:
        if bucket_col not in gap.columns:
            continue
        part = (
            gap.groupby(["split", bucket_col], dropna=False)
            .agg(
                n_samples=("sample_id", "size"),
                rbf_rmse=("rbf_sample_rmse", grouped_rmse),
                deployable_oracle_rmse=("deployable_oracle_rmse", grouped_rmse),
                mean_gain_over_rbf=("deployable_oracle_gain_over_rbf", "mean"),
                median_gain_over_rbf=("deployable_oracle_gain_over_rbf", "median"),
                positive_gain_rate=("deployable_oracle_gain_over_rbf", lambda x: float((x > 1e-6).mean())),
                non_rbf_oracle_rate=("deployable_oracle_uses_non_rbf", "mean"),
                mean_candidate_spread=("candidate_spread_mean", "mean"),
                mean_gt_peak_abs=("gt_peak_abs", "mean"),
                mean_reversal_count=("reversal_count", "mean"),
            )
            .reset_index()
            .rename(columns={bucket_col: "bucket_value"})
        )
        part.insert(1, "bucket_type", bucket_col)
        part["oracle_delta_vs_rbf"] = part["deployable_oracle_rmse"] - part["rbf_rmse"]
        part["coverage_status"] = part.apply(status_for_group, axis=1)
        rows.append(part)
    return pd.concat(rows, ignore_index=True, sort=False)


def oracle_winner_distribution(gap: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, split_df in gap.groupby("split"):
        for family, family_df in split_df.groupby("response_family"):
            total = max(len(family_df), 1)
            for model, part in family_df.groupby("deployable_oracle_best_model"):
                rows.append(
                    {
                        "split": split,
                        "response_family": family,
                        "oracle_best_model": model,
                        "n_samples": int(len(part)),
                        "sample_rate": float(len(part) / total),
                    }
                )
    return pd.DataFrame(rows)


def next_experiment_plan(coverage: pd.DataFrame) -> pd.DataFrame:
    test = coverage[coverage["split"] == "test"].copy()
    focus = test[test["coverage_status"].isin(["selector_gap_candidate_pool_has_signal", "candidate_generation_gap", "mixed_gap"])].copy()
    focus["priority_score"] = (
        focus["mean_gain_over_rbf"].clip(lower=0).astype(float) * 1.5
        + focus["non_rbf_oracle_rate"].astype(float) * 0.2
        + (focus["deployable_oracle_rmse"].astype(float) > 0.6).astype(float) * 0.1
    )
    focus = focus.sort_values(["priority_score", "n_samples"], ascending=[False, False]).head(20)

    def recommendation(row: pd.Series) -> str:
        btype = str(row["bucket_type"])
        value = str(row["bucket_value"])
        if btype == "amplitude_mode" or "large" in value:
            return "add amplitude-quantile candidates and explicit severe-underprediction penalty"
        if btype == "peak_timing" or "late_peak" in value or "early_peak" in value:
            return "add peak-time/onset conditional candidates"
        if btype == "tail_mode" or "tail" in value:
            return "add tail-mode candidates: return, same-side persistence, opposite-side overshoot"
        if btype == "correction_mode" or "multi_segment" in value or "reversal" in value:
            return "add reversal/multi-segment candidate constructor"
        if btype == "direction_mode":
            return "add direction-conditioned candidates with wrong-side guard"
        return "use response-factorized candidate generation and non-oracle calibration"

    focus["recommended_action"] = focus.apply(recommendation, axis=1)
    return focus[
        [
            "bucket_type",
            "bucket_value",
            "n_samples",
            "rbf_rmse",
            "deployable_oracle_rmse",
            "mean_gain_over_rbf",
            "non_rbf_oracle_rate",
            "coverage_status",
            "priority_score",
            "recommended_action",
        ]
    ]


def candidate_blueprint() -> pd.DataFrame:
    rows = [
        {
            "candidate_family_id": "F0_rbf_anchor",
            "physical_target": "conservative fallback for samples where local vehicle history is sufficient",
            "trajectory_form": "current RBF/KNN prediction retained as candidate 0",
            "causal_inputs": "pre-event vehicle history + road/event context only",
            "training_supervision": "none beyond existing RBF/KNN fitting",
            "selection_signal_allowed": "candidate confidence, pre-event context, candidate shape features",
            "leakage_guard": "never use test labels or oracle winner as inference input",
            "required_evaluation": "must remain the main reference; all new candidates report delta vs RBF/KNN",
        },
        {
            "candidate_family_id": "F1_direction_amp_quantile",
            "physical_target": "wrong-side and severe-amplitude-underprediction cases",
            "trajectory_form": "direction-conditioned amplitude quantile curves: small, medium, large, very-large",
            "causal_inputs": "pre-event vehicle history + causal road/event context",
            "training_supervision": "train labels: peak direction and peak amplitude bins from train only",
            "selection_signal_allowed": "predicted direction probability, amplitude-bin probability, candidate peak amplitude",
            "leakage_guard": "amplitude thresholds fitted on train only; val chooses bins/weights",
            "required_evaluation": "wrong-side rate, large-response recall, severe-underprediction rate",
        },
        {
            "candidate_family_id": "F2_peak_time_onset",
            "physical_target": "early/late response mismatch and onset delay",
            "trajectory_form": "keypoint candidate parameterized by onset time, peak time, peak amplitude",
            "causal_inputs": "pre-event steering, speed, curvature, instability context",
            "training_supervision": "train labels: onset fraction and peak-time fraction",
            "selection_signal_allowed": "predicted onset/peak-time uncertainty and candidate timing disagreement",
            "leakage_guard": "no label-window samples enter input; time thresholds train-only",
            "required_evaluation": "peak-time MAE and onset-delay MAE by response family",
        },
        {
            "candidate_family_id": "F3_tail_mode",
            "physical_target": "tail return, same-side persistence, opposite-side overshoot",
            "trajectory_form": "candidate tail modes connected to peak keypoint: return-zero, persist, overshoot",
            "causal_inputs": "pre-event vehicle history + road/event context",
            "training_supervision": "train labels: tail mode and tail signed ratio",
            "selection_signal_allowed": "tail-mode probability and candidate tail dispersion",
            "leakage_guard": "tail labels only for training; no test statistics in thresholds",
            "required_evaluation": "tail_abs_error, tail_drift_risk, zero-crossing mismatch",
        },
        {
            "candidate_family_id": "F4_reversal_multisegment",
            "physical_target": "reverse correction and multi-segment steering response",
            "trajectory_form": "piecewise candidate with optional second turning point and zero crossing",
            "causal_inputs": "pre-event vehicle history + road/event context",
            "training_supervision": "train labels: reversal count, zero-crossing flag, secondary peak timing",
            "selection_signal_allowed": "predicted correction-mode probability and candidate curvature features",
            "leakage_guard": "correction labels train-only; no oracle selector at deployment",
            "required_evaluation": "reversal_count_exact_match, multi_segment recall, zero-crossing mismatch",
        },
        {
            "candidate_family_id": "F5_uncertainty_gate",
            "physical_target": "identify samples where candidate choice is unreliable",
            "trajectory_form": "not a trajectory; gate between RBF fallback and structured candidates",
            "causal_inputs": "candidate probabilities, candidate disagreement, pre-event context",
            "training_supervision": "train/val only: oracle regret bins and selected-candidate correctness",
            "selection_signal_allowed": "calibrated confidence and uncertainty; no label-derived test features",
            "leakage_guard": "gate chosen on val only and frozen before test",
            "required_evaluation": "coverage-risk curve, selector regret, RBF fallback rate",
        },
    ]
    return pd.DataFrame(rows)


def gate_table(stage7d_gate: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    stage7d_status = stage7d_gate.set_index("gate_item")["status"].to_dict()
    test = coverage[coverage["split"] == "test"].copy()
    selector_gap_n = int((test["coverage_status"] == "selector_gap_candidate_pool_has_signal").sum())
    generation_gap_n = int((test["coverage_status"] == "candidate_generation_gap").sum())
    return pd.DataFrame(
        [
            {
                "gate_item": "stage07d_deployable_upgrade",
                "status": str(stage7d_status.get("deployable_upgrade", "unknown")),
                "evidence": "Stage 7d val gate selected always_rbf_reference; test delta vs RBF is 0.",
            },
            {
                "gate_item": "continue_selector_only_route",
                "status": "blocked",
                "evidence": "Two selector rounds fell back to RBF/KNN; next improvement should redesign candidate generation.",
            },
            {
                "gate_item": "candidate_generation_redesign_needed",
                "status": "pass",
                "evidence": f"test bucket statuses: selector_gap={selector_gap_n}, generation_gap={generation_gap_n}.",
            },
            {
                "gate_item": "next_training_allowed",
                "status": "conditional",
                "evidence": "Allowed only after implementing response-factorized candidates and keeping RBF/KNN as fixed reference.",
            },
            {
                "gate_item": "stage08_physio_eeg_allowed",
                "status": "blocked",
                "evidence": "Do not enter physio/EEG until vehicle-only candidate generation and non-oracle selection are stable.",
            },
            {
                "gate_item": "server_used",
                "status": "no",
                "evidence": "Local audit only; credential file was not read.",
            },
        ]
    )


def plot_oracle_gain_by_family(coverage: pd.DataFrame, path: Path) -> None:
    test = coverage[(coverage["split"] == "test") & (coverage["bucket_type"] == "response_family")].copy()
    test = test.sort_values("mean_gain_over_rbf", ascending=True).tail(14)
    fig, ax = plt.subplots(figsize=(10.5, max(4.2, 0.42 * len(test))))
    colors = np.where(test["coverage_status"].eq("selector_gap_candidate_pool_has_signal"), "#54a24b", "#e45756")
    ax.barh(np.arange(len(test)), test["mean_gain_over_rbf"], color=colors)
    ax.set_yticks(np.arange(len(test)), test["bucket_value"], fontsize=8)
    ax.set_xlabel("Mean deployable-oracle gain over RBF/KNN")
    ax.set_title("Stage 7e: where candidate pool has useful upper bound")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_oracle_winner_distribution(winners: pd.DataFrame, path: Path) -> None:
    test = winners[winners["split"] == "test"].copy()
    if test.empty:
        return
    fam_order = (
        test.groupby("response_family")["n_samples"].sum().sort_values(ascending=False).head(12).index.tolist()
    )
    models = [m for m in DEPLOYABLE_CANDIDATES if m in set(test["oracle_best_model"])]
    labels = {RBF_MODEL: "RBF/KNN", KEYPOINT_MODEL: "keypoint", BRANCH_MODELS[0]: "branch0", BRANCH_MODELS[1]: "branch1", BRANCH_MODELS[2]: "branch2"}
    colors = {RBF_MODEL: "#4c78a8", KEYPOINT_MODEL: "#b279a2", BRANCH_MODELS[0]: "#e45756", BRANCH_MODELS[1]: "#f58518", BRANCH_MODELS[2]: "#54a24b"}
    fig, ax = plt.subplots(figsize=(11.0, max(4.6, 0.38 * len(fam_order))))
    left = np.zeros(len(fam_order))
    for model in models:
        vals = []
        for fam in fam_order:
            part = test[(test["response_family"] == fam) & (test["oracle_best_model"] == model)]
            vals.append(float(part["sample_rate"].sum()) if not part.empty else 0.0)
        ax.barh(np.arange(len(fam_order)), vals, left=left, color=colors.get(model, "#999999"), label=labels.get(model, model))
        left += np.asarray(vals)
    ax.set_yticks(np.arange(len(fam_order)), fam_order, fontsize=8)
    ax.set_xlabel("Oracle winner share within response family")
    ax.set_title("Stage 7e: existing candidate oracle winners on test")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_gap_scatter(gap: pd.DataFrame, path: Path) -> None:
    test = gap[gap["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    colors = test["correction_mode"].astype("category").cat.codes
    sc = ax.scatter(
        test["gt_peak_abs"],
        test["deployable_oracle_gain_over_rbf"],
        c=colors,
        s=48 + 120 * test["deployable_oracle_uses_non_rbf"].astype(float),
        cmap="tab10",
        alpha=0.82,
        edgecolors="none",
    )
    ax.axhline(0.0, color="#666666", linewidth=0.9)
    ax.set_xlabel("GT peak absolute steering delta")
    ax.set_ylabel("Deployable oracle gain over RBF/KNN")
    ax.set_title("Stage 7e: sample-level candidate opportunity")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="correction mode code")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_blueprint_table(blueprint: pd.DataFrame, path: Path) -> None:
    rows = blueprint[["candidate_family_id", "physical_target", "required_evaluation"]].copy()
    fig, ax = plt.subplots(figsize=(14.0, 6.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows.values,
        colLabels=["candidate family", "physical target", "required evaluation"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.16, 0.48, 0.36],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.75)
    ax.set_title("Stage 7e candidate generation redesign blueprint", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_reports(
    gap: pd.DataFrame,
    coverage: pd.DataFrame,
    plan: pd.DataFrame,
    gate: pd.DataFrame,
    figures: dict[str, str],
) -> None:
    test = gap[gap["split"] == "test"].copy()
    rbf_rmse = float(np.sqrt(np.mean(np.square(test["rbf_sample_rmse"]))))
    oracle_rmse = float(np.sqrt(np.mean(np.square(test["deployable_oracle_rmse"]))))
    non_rbf_rate = float(test["deployable_oracle_uses_non_rbf"].mean())
    mean_gain = float(test["deployable_oracle_gain_over_rbf"].mean())
    top_plan = plan.head(8).to_string(index=False) if not plan.empty else "(no focused rows)"
    gate_text = gate.to_string(index=False)
    status_counts = coverage[coverage["split"] == "test"]["coverage_status"].value_counts().rename_axis("coverage_status").reset_index(name="n")
    status_text = status_counts.to_string(index=False)

    user = f"""# Stage 7e 用户查看版：候选生成重设计审计 v0.1

## 这个阶段为什么做

Stage 7c 说明候选池有 oracle 上限，Stage 7d 说明当前非 oracle selector 学不会稳定选择。因此下一步不能继续只堆 selector，而要先检查候选本身应该怎样生成，才能覆盖真实失稳响应。

## 这个阶段检查了什么

- 从真实方向盘标签里提取响应类型：方向、幅值、峰值时间、尾段模式、反向修正/多段修正。
- 用已有候选轨迹计算每个响应类型下的 RBF/KNN 误差、候选 oracle 误差、oracle gain 和候选胜出比例。
- 把缺口分成两类：候选池有信号但 selector 不会选；候选生成本身还不够。
- 输出下一版候选生成蓝图，不训练新模型。

## 目前发现了什么

- test RBF/KNN RMSE={rbf_rmse:.6f}。
- test deployable candidate oracle RMSE={oracle_rmse:.6f}，平均样本 gain={mean_gain:.6f}。
- test 中 oracle 选择非 RBF/KNN 候选的比例={non_rbf_rate:.3f}。
- 这说明当前候选池不是完全无效，但 Stage 7d 已经证明当前 selector 不能可靠使用它。

test 覆盖状态统计：

```text
{status_text}
```

## 下一版候选生成优先级

```text
{top_plan}
```

## 哪些结果可信

可信的是：这一步只使用 Stage 7c 已导出的候选轨迹和真实标签做离线审计，没有训练模型，没有用生理/脑电/连续风格，也没有读取服务器凭据。它给出的不是性能提升，而是下一版车辆-only 多候选模型应该覆盖哪些物理响应类型。

## 哪些结果还不能下结论

不能说多假设已经可部署有效；不能说生理或 EEG 可以进入；不能把候选 oracle 当作模型结果。Stage 7e 只说明“下一步该怎样重新生成候选”。

## 下一阶段是否可以继续

可以继续 Stage 7，但下一步应按候选生成蓝图实现 response-factorized candidates：方向/幅值、峰值时间、尾段模式、反向修正/多段修正、可靠性门控。RBF/KNN 必须继续作为固定主参照。

## 推荐优先查看

1. `{figures["oracle_gain_by_family"]}`
2. `{figures["winner_distribution"]}`
3. `{figures["gap_scatter"]}`
4. `{figures["blueprint"]}`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_candidate_generation_blueprint.csv`
6. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_gate_table.csv`
"""
    (REPORT_DIR / "stage07e_candidate_generation_redesign_user_summary_cn.md").write_text(user, encoding="utf-8")

    tech = f"""# Stage 7e 技术报告：candidate generation redesign audit v0.1

## Scope

- Input trajectories: `{path_str(TRAJECTORY_NPZ)}`
- Feature diagnosis: `{path_str(FEATURE_DIAG)}`
- Stage 7d gate: `{path_str(STAGE7D_GATE)}`
- Candidate pool audited: `{', '.join(DEPLOYABLE_CANDIDATES)}`
- No new model training.
- No server used. Credential file not read.
- Excluded modalities: physio, EEG, continuous style, subject ID.

## Aggregate Test Result

- RBF/KNN RMSE: {rbf_rmse:.6f}
- Deployable-candidate oracle RMSE: {oracle_rmse:.6f}
- Mean sample gain over RBF/KNN: {mean_gain:.6f}
- Non-RBF oracle winner rate: {non_rbf_rate:.6f}

## Gate

```text
{gate_text}
```

## Interpretation

Stage 7d blocked selector-only continuation. Stage 7e therefore upgrades the next action from selector tuning to response-factorized candidate generation. This is still vehicle-only; it does not authorize physio/EEG claims.

## Tables

- `stage07e_response_label_table.csv`
- `stage07e_sample_candidate_gap_table.csv`
- `stage07e_existing_candidate_coverage_by_bucket.csv`
- `stage07e_oracle_winner_distribution.csv`
- `stage07e_candidate_generation_blueprint.csv`
- `stage07e_next_experiment_plan.csv`
- `stage07e_gate_table.csv`

## Figures

- `{figures["oracle_gain_by_family"]}`
- `{figures["winner_distribution"]}`
- `{figures["gap_scatter"]}`
- `{figures["blueprint"]}`
"""
    (REPORT_DIR / "stage07e_candidate_generation_redesign_v0_1_cn.md").write_text(tech, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    z, feature_diag, stage7d_gate = load_inputs()
    y = z["y_true"].astype(np.float32)
    y_mask = z["y_mask"].astype(bool)
    label_time = z["label_time_rel_s"].astype(np.float32)
    sample_ids = z["sample_ids"].astype(str)
    split = z["split"].astype(str)

    response = response_rows(y, y_mask, label_time, sample_ids, split)
    gap = candidate_gap_rows(z, response, feature_diag)
    coverage = coverage_by_bucket(gap)
    winners = oracle_winner_distribution(gap)
    blueprint = candidate_blueprint()
    plan = next_experiment_plan(coverage)
    gate = gate_table(stage7d_gate, coverage)

    response.to_csv(TABLE_DIR / "stage07e_response_label_table.csv", index=False, encoding="utf-8-sig")
    gap.to_csv(TABLE_DIR / "stage07e_sample_candidate_gap_table.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(TABLE_DIR / "stage07e_existing_candidate_coverage_by_bucket.csv", index=False, encoding="utf-8-sig")
    winners.to_csv(TABLE_DIR / "stage07e_oracle_winner_distribution.csv", index=False, encoding="utf-8-sig")
    blueprint.to_csv(TABLE_DIR / "stage07e_candidate_generation_blueprint.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(TABLE_DIR / "stage07e_next_experiment_plan.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "stage07e_gate_table.csv", index=False, encoding="utf-8-sig")

    figures = {
        "oracle_gain_by_family": path_str(FIG_DIR / "stage07e_oracle_gain_by_response_family_test.png"),
        "winner_distribution": path_str(FIG_DIR / "stage07e_oracle_winner_distribution_test.png"),
        "gap_scatter": path_str(FIG_DIR / "stage07e_candidate_gap_scatter_test.png"),
        "blueprint": path_str(FIG_DIR / "stage07e_candidate_generation_blueprint.png"),
    }
    plot_oracle_gain_by_family(coverage, Path(figures["oracle_gain_by_family"]))
    plot_oracle_winner_distribution(winners, Path(figures["winner_distribution"]))
    plot_gap_scatter(gap, Path(figures["gap_scatter"]))
    plot_blueprint_table(blueprint, Path(figures["blueprint"]))
    write_reports(gap, coverage, plan, gate, figures)

    test = gap[gap["split"] == "test"].copy()
    rbf_rmse = float(np.sqrt(np.mean(np.square(test["rbf_sample_rmse"]))))
    oracle_rmse = float(np.sqrt(np.mean(np.square(test["deployable_oracle_rmse"]))))
    summary = {
        "output_version": OUTPUT_VERSION,
        "track_id": TRACK_ID,
        "n_samples": int(len(gap)),
        "test_n": int(len(test)),
        "test_rbf_rmse": rbf_rmse,
        "test_deployable_oracle_rmse": oracle_rmse,
        "test_oracle_delta_vs_rbf": oracle_rmse - rbf_rmse,
        "test_mean_sample_gain_over_rbf": float(test["deployable_oracle_gain_over_rbf"].mean()),
        "test_non_rbf_oracle_rate": float(test["deployable_oracle_uses_non_rbf"].mean()),
        "candidate_family_count": int(len(blueprint)),
        "next_plan_rows": int(len(plan)),
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "uses_subject_id": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "figures": figures,
    }
    (LOG_DIR / "stage07e_candidate_generation_redesign_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
