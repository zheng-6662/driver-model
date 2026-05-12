# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch


ROOT = Path("F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
BASELINE_SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(BASELINE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402
import stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1 as clean_v01  # noqa: E402
import stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1 as keypoint_v01  # noqa: E402
import stage03_vehicle_instability_topk_vehicle_transformer_v0_1 as topk_v01  # noqa: E402


OUTPUT_VERSION = "stage07c_candidate_trajectory_export_v0_1"
TRACK_ID = "B_response3s_strict_core"
SPLIT_STRATEGY = "session_level_split"
RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
KEYPOINT_MODEL = "keypoint_residual_vehicle_transformer_no_subject"
TOP1_MODEL = "topk_vehicle_transformer_top1_no_subject"
BEST3_ORACLE_MODEL = "topk_vehicle_transformer_best_of_3_oracle"
BEST_RBF_TOPK_ORACLE_MODEL = "oracle_best_of_rbf_topk_upper_bound"
BEST_BROAD_ORACLE_MODEL = "oracle_best_of_rbf_keypoint_topk_upper_bound"
BRANCH_MODELS = [f"topk_vehicle_transformer_branch{k}_no_subject" for k in range(3)]

OUT_ROOT = ROOT / "07_multihypothesis" / OUTPUT_VERSION
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
ARRAY_DIR = OUT_ROOT / "arrays"
REPORT_DIR = ROOT / "09_reports"

TOPK_CKPT = (
    ROOT
    / "03_baselines"
    / "stage03_vehicle_instability_topk_vehicle_transformer_v0_1"
    / "checkpoints"
    / "B_response3s_strict_core_topk_vehicle_transformer_top1_no_subject_best.pt"
)
KEYPOINT_CKPT = (
    ROOT
    / "03_baselines"
    / "stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1"
    / "checkpoints"
    / "B_response3s_strict_core_keypoint_residual_vehicle_transformer_no_subject_best.pt"
)


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, ARRAY_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def path_str(path: Path) -> str:
    return str(path).replace("\\", "/")


def split_indices(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = meta[SPLIT_STRATEGY].astype(str).to_numpy()
    return np.where(split == "train")[0], np.where(split == "val")[0], np.where(split == "test")[0]


def load_track_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    manifest = pd.read_csv(keypoint_v01.TASK_MANIFEST_PATH)
    cfg = keypoint_v01.TRACKS[TRACK_ID]
    return keypoint_v01.load_track(TRACK_ID, cfg, manifest)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def rebuild_context_and_inputs(
    input_values: np.ndarray,
    input_mask: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    expected_step: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    x_scaled, scaler_info = keypoint_v01.standardize_vehicle_inputs(input_values, input_mask, train_idx)
    context, context_names = keypoint_v01.build_context_features(meta, train_idx)
    x_model = x_scaled[:, ::expected_step, :].copy()
    return x_model, context, context_names, scaler_info


def validate_context(expected: list[str], actual: list[str], label: str) -> None:
    if list(expected) != list(actual):
        raise RuntimeError(f"{label}: context feature mismatch; checkpoint cannot be safely replayed")


def load_topk_predictions(
    x_model: np.ndarray,
    context: np.ndarray,
    label_time: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    k = int(checkpoint["k"])
    model = topk_v01.TopKVehicleTransformer(
        vehicle_dim=x_model.shape[2],
        context_dim=context.shape[1],
        label_time=label_time,
        k=k,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    trajs, logits = topk_v01.predict_all(model, x_model, context, float(checkpoint["label_scale"]), batch_size=64)
    top1, best3, top1_idx, best3_idx, probs = topk_v01.select_top1_and_bestk(trajs, logits, y, y_mask)
    return {
        "trajs": trajs.astype(np.float32),
        "logits": logits.astype(np.float32),
        "probs": probs.astype(np.float32),
        "top1": top1.astype(np.float32),
        "best3": best3.astype(np.float32),
        "top1_idx": top1_idx.astype(np.int16),
        "best3_idx": best3_idx.astype(np.int16),
    }


def load_keypoint_predictions(
    x_model: np.ndarray,
    context: np.ndarray,
    label_time: np.ndarray,
    checkpoint: dict[str, Any],
) -> np.ndarray:
    model = keypoint_v01.KeypointResidualVehicleTransformer(
        vehicle_dim=x_model.shape[2],
        context_dim=context.shape[1],
        label_time=label_time,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return keypoint_v01.predict_all(model, x_model, context, float(checkpoint["label_scale"]), batch_size=64).astype(np.float32)


def sample_rmse_array(y_true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(y_true) & np.isfinite(pred)
    diff = np.where(valid, pred - y_true, np.nan)
    denom = np.maximum(valid.sum(axis=1), 1)
    return np.sqrt(np.nansum(diff * diff, axis=1) / denom).astype(np.float32)


def best_oracle(
    y: np.ndarray,
    y_mask: np.ndarray,
    candidate_names: list[str],
    candidate_preds: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rmse_cols = [sample_rmse_array(y, pred, y_mask) for pred in candidate_preds]
    rmse_mat = np.stack(rmse_cols, axis=1)
    best_idx = np.nanargmin(rmse_mat, axis=1).astype(np.int16)
    stacked = np.stack(candidate_preds, axis=1)
    best_pred = stacked[np.arange(stacked.shape[0]), best_idx].astype(np.float32)
    rows = pd.DataFrame({"oracle_best_index": best_idx, "oracle_best_model": [candidate_names[i] for i in best_idx]})
    for j, name in enumerate(candidate_names):
        rows[f"{name}__sample_rmse"] = rmse_mat[:, j]
    rows["oracle_sample_rmse"] = rmse_mat[np.arange(len(best_idx)), best_idx]
    return best_pred, best_idx, rmse_mat, rows


def metric_tables(
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, per_sample = topk_v01.evaluate_predictions(y, y_mask, label_time, meta, train_idx, predictions)
    metrics["track_id"] = TRACK_ID
    per_sample["track_id"] = TRACK_ID
    return metrics, per_sample


def pairwise_disagreement(
    meta: pd.DataFrame,
    y_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    model_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    split_values = meta[SPLIT_STRATEGY].astype(str).to_numpy()
    for i, left in enumerate(model_names):
        for right in model_names[i + 1 :]:
            a = predictions[left]
            b = predictions[right]
            valid = y_mask & np.isfinite(a) & np.isfinite(b)
            diff = np.where(valid, a - b, np.nan)
            denom = np.maximum(valid.sum(axis=1), 1)
            dist = np.sqrt(np.nansum(diff * diff, axis=1) / denom).astype(np.float32)
            for row_idx, value in enumerate(dist):
                rows.append(
                    {
                        "sample_id": meta.at[row_idx, "sample_id"],
                        "event_uid": meta.at[row_idx, "event_uid"],
                        "subject": meta.at[row_idx, "subject"],
                        "session_stamp": meta.at[row_idx, "session_stamp"],
                        "split": split_values[row_idx],
                        "left_model": left,
                        "right_model": right,
                        "pair_name": f"{left}__vs__{right}",
                        "trajectory_rmse_distance": float(value),
                    }
                )
    long_df = pd.DataFrame(rows)
    summary = (
        long_df.groupby(["split", "pair_name"], dropna=False)
        .agg(
            n_samples=("trajectory_rmse_distance", "size"),
            mean_distance=("trajectory_rmse_distance", "mean"),
            median_distance=("trajectory_rmse_distance", "median"),
            q75_distance=("trajectory_rmse_distance", lambda x: float(np.nanpercentile(x, 75))),
            q90_distance=("trajectory_rmse_distance", lambda x: float(np.nanpercentile(x, 90))),
        )
        .reset_index()
    )
    return long_df, summary


def peak_abs(pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.where(mask & np.isfinite(pred), pred, np.nan)
    return np.nanmax(np.abs(arr), axis=1).astype(np.float32)


def peak_sign(pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.where(mask & np.isfinite(pred), pred, np.nan)
    idx = np.nanargmax(np.abs(np.nan_to_num(arr, nan=0.0)), axis=1)
    signed = arr[np.arange(arr.shape[0]), idx]
    return np.where(signed >= 0.0, 1, -1).astype(np.int8)


def prediction_reversal_count(pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(pred.shape[0], dtype=np.int16)
    for i in range(pred.shape[0]):
        arr = np.where(mask[i] & np.isfinite(pred[i]), pred[i], np.nan)
        out[i] = int(eval_utils.reversal_count(arr))
    return out


def build_feature_and_diagnosis_table(
    meta: pd.DataFrame,
    y: np.ndarray,
    y_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    topk_data: dict[str, Any],
    rbf_topk_oracle: pd.DataFrame,
    broad_oracle: pd.DataFrame,
) -> pd.DataFrame:
    split_values = meta[SPLIT_STRATEGY].astype(str).to_numpy()
    base_cols = [
        "sample_id",
        "event_uid",
        "subject",
        "session_stamp",
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
    cols = [c for c in base_cols if c in meta.columns]
    out = meta[cols].copy()
    out["split"] = split_values
    out["top1_branch"] = topk_data["top1_idx"]
    out["top1_prob"] = np.max(topk_data["probs"], axis=1)
    out["prob_margin"] = np.sort(topk_data["probs"], axis=1)[:, -1] - np.sort(topk_data["probs"], axis=1)[:, -2]
    out["prob_entropy"] = -np.sum(topk_data["probs"] * np.log(np.clip(topk_data["probs"], 1e-8, 1.0)), axis=1)
    spread = np.nanstd(topk_data["trajs"], axis=1)
    out["topk_branch_spread_mean"] = np.nanmean(spread, axis=1)
    out["topk_branch_spread_peak"] = np.nanmax(spread, axis=1)
    branch_peak_abs = np.stack([peak_abs(predictions[name], y_mask) for name in BRANCH_MODELS], axis=1)
    out["branch_peak_abs_spread"] = np.nanmax(branch_peak_abs, axis=1) - np.nanmin(branch_peak_abs, axis=1)
    for name in [RBF_MODEL, KEYPOINT_MODEL, TOP1_MODEL, *BRANCH_MODELS]:
        out[f"{name}__pred_peak_abs"] = peak_abs(predictions[name], y_mask)
        out[f"{name}__pred_peak_sign"] = peak_sign(predictions[name], y_mask)
        out[f"{name}__pred_reversal_count"] = prediction_reversal_count(predictions[name], y_mask)
    rbf_rmse = sample_rmse_array(y, predictions[RBF_MODEL], y_mask)
    out["label_diag__rbf_sample_rmse"] = rbf_rmse
    out["label_diag__rbf_topk_oracle_model"] = rbf_topk_oracle["oracle_best_model"].to_numpy()
    out["label_diag__rbf_topk_oracle_rmse"] = rbf_topk_oracle["oracle_sample_rmse"].to_numpy()
    out["label_diag__rbf_topk_oracle_gain_over_rbf"] = rbf_rmse - rbf_topk_oracle["oracle_sample_rmse"].to_numpy()
    out["label_diag__rbf_topk_oracle_uses_non_rbf"] = (rbf_topk_oracle["oracle_best_model"].astype(str) != RBF_MODEL).astype(int)
    out["label_diag__broad_oracle_model"] = broad_oracle["oracle_best_model"].to_numpy()
    out["label_diag__broad_oracle_rmse"] = broad_oracle["oracle_sample_rmse"].to_numpy()
    out["label_diag__broad_oracle_gain_over_rbf"] = rbf_rmse - broad_oracle["oracle_sample_rmse"].to_numpy()
    out["label_diag__broad_oracle_uses_non_rbf"] = (broad_oracle["oracle_best_model"].astype(str) != RBF_MODEL).astype(int)
    return out


def summarize_oracle(
    feature_diag: pd.DataFrame,
    oracle_prefix: str,
    model_col: str,
    gain_col: str,
    rmse_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, grp in feature_diag.groupby("split"):
        positive_gain = grp[gain_col] > 1e-6
        rows.append(
            {
                "oracle_family": oracle_prefix,
                "split": split,
                "n_samples": int(len(grp)),
                "mean_oracle_rmse": float(grp[rmse_col].mean()),
                "mean_gain_over_rbf": float(grp[gain_col].mean()),
                "median_gain_over_rbf": float(grp[gain_col].median()),
                "positive_gain_rate": float(positive_gain.mean()),
                "non_rbf_oracle_rate": float((grp[model_col].astype(str) != RBF_MODEL).mean()),
            }
        )
        for model, part in grp.groupby(model_col):
            rows.append(
                {
                    "oracle_family": oracle_prefix,
                    "split": split,
                    "oracle_best_model": model,
                    "n_samples": int(len(part)),
                    "sample_rate": float(len(part) / max(len(grp), 1)),
                    "mean_gain_over_rbf": float(part[gain_col].mean()),
                    "mean_oracle_rmse": float(part[rmse_col].mean()),
                }
            )
    return pd.DataFrame(rows)


def top_rows(
    feature_diag: pd.DataFrame,
    column: str,
    path: Path,
    n: int = 30,
) -> list[str]:
    rows = feature_diag[feature_diag["split"] == "test"].sort_values(column, ascending=False).head(n).copy()
    rows.to_csv(path, index=False, encoding="utf-8-sig")
    return rows["sample_id"].astype(str).head(12).tolist()


def plot_metric_summary(metrics: pd.DataFrame, path: Path) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    order = [
        RBF_MODEL,
        KEYPOINT_MODEL,
        TOP1_MODEL,
        *BRANCH_MODELS,
        BEST3_ORACLE_MODEL,
        BEST_RBF_TOPK_ORACLE_MODEL,
        BEST_BROAD_ORACLE_MODEL,
    ]
    test["order"] = test["model_name"].map({name: i for i, name in enumerate(order)})
    test = test[test["model_name"].isin(order)].sort_values("order")
    labels = [
        "RBF/KNN",
        "keypoint",
        "top1",
        "branch0",
        "branch1",
        "branch2",
        "best3*",
        "rbf+topK*",
        "broad*",
    ][: len(test)]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2))
    for ax, col, title in [
        (axes[0], "rmse_steer", "RMSE"),
        (axes[1], "wrong_side_rate", "Wrong-side"),
        (axes[2], "large_response_recall", "Large recall"),
        (axes[3], "difficult_top20_rmse", "Difficult RMSE"),
    ]:
        ax.bar(np.arange(len(test)), test[col].astype(float), color="#4777b3")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(test)), labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Stage 7c candidate trajectories on test (* = oracle upper bound)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_disagreement_gain(feature_diag: pd.DataFrame, path: Path) -> None:
    test = feature_diag[feature_diag["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    sc = ax.scatter(
        test["topk_branch_spread_mean"],
        test["label_diag__rbf_topk_oracle_gain_over_rbf"],
        c=test["top1_prob"],
        cmap="viridis",
        s=52,
        alpha=0.86,
        edgecolors="none",
    )
    ax.axhline(0.0, color="#888888", linewidth=0.9)
    ax.set_xlabel("Top-K branch spread mean")
    ax.set_ylabel("Oracle gain over RBF/KNN")
    ax.set_title("Disagreement vs oracle opportunity on test")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="top-1 probability")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_prediction_grid(
    path: Path,
    sample_ids: list[str],
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    title: str,
) -> None:
    lookup = {sid: i for i, sid in enumerate(meta["sample_id"].astype(str))}
    ids = [sid for sid in sample_ids if sid in lookup][:12]
    if not ids:
        return
    ncols = 3
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 3.2 * nrows), squeeze=False)
    plot_models = [
        (RBF_MODEL, "#1f77b4", "RBF/KNN", "-"),
        (KEYPOINT_MODEL, "#9467bd", "keypoint", "-"),
        ("topk_vehicle_transformer_branch0_no_subject", "#d62728", "b0", "--"),
        ("topk_vehicle_transformer_branch1_no_subject", "#ff7f0e", "b1", "--"),
        ("topk_vehicle_transformer_branch2_no_subject", "#2ca02c", "b2", "--"),
        (TOP1_MODEL, "#111111", "top1", "-."),
    ]
    for ax, sid in zip(axes.ravel(), ids):
        i = lookup[sid]
        valid = y_mask[i] & np.isfinite(y[i])
        ax.plot(label_time[valid], y[i, valid], color="#000000", linewidth=1.8, label="GT")
        for model_name, color, label, style in plot_models:
            pred = predictions[model_name][i]
            valid_pred = valid & np.isfinite(pred)
            ax.plot(label_time[valid_pred], pred[valid_pred], color=color, linewidth=1.05, linestyle=style, alpha=0.86, label=label)
        short = sid.split("__")[-2] if "__" in sid else sid[-12:]
        ax.set_title(short, fontsize=8)
        ax.grid(True, alpha=0.22)
        ax.axhline(0.0, color="#dddddd", linewidth=0.8)
    for ax in axes.ravel()[len(ids) :]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=8)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_reports(
    metrics: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    feature_diag: pd.DataFrame,
    figures: dict[str, str],
    arrays_path: Path,
) -> None:
    test_metrics = metrics[metrics["split"] == "test"].set_index("model_name")

    def safe(model: str, col: str) -> float:
        if model not in test_metrics.index or col not in test_metrics.columns:
            return float("nan")
        return float(test_metrics.loc[model, col])

    rbf_rmse = safe(RBF_MODEL, "rmse_steer")
    key_rmse = safe(KEYPOINT_MODEL, "rmse_steer")
    top1_rmse = safe(TOP1_MODEL, "rmse_steer")
    oracle_rmse = safe(BEST_RBF_TOPK_ORACLE_MODEL, "rmse_steer")
    broad_rmse = safe(BEST_BROAD_ORACLE_MODEL, "rmse_steer")
    test_diag = feature_diag[feature_diag["split"] == "test"].copy()
    top_oracle = (
        test_diag["label_diag__rbf_topk_oracle_model"].astype(str).value_counts(normalize=True).rename_axis("model").reset_index(name="rate")
    )
    top_oracle_text = "```text\n" + top_oracle.to_string(index=False) + "\n```"
    test_metric_text = (
        metrics[
            (metrics["split"] == "test")
            & metrics["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL, TOP1_MODEL, *BRANCH_MODELS, BEST_RBF_TOPK_ORACLE_MODEL, BEST_BROAD_ORACLE_MODEL])
        ][["model_name", "rmse_steer", "wrong_side_rate", "large_response_recall", "difficult_top20_rmse"]]
        .to_string(index=False)
    )

    user = f"""# Stage 7c 用户查看版：候选轨迹导出与差异审计 v0.1

## 这个阶段为什么做

前面已经看到 top-K / 多候选有 oracle 上限，但非 oracle selector 最后完全退回 RBF/KNN。这里先不训练新模型，而是把已有候选轨迹完整导出来，看清楚问题到底是“候选本身不够不同”，还是“候选有潜力但选择机制不会选”。

## 这个阶段检查了什么

- 样本：`{TRACK_ID}`，3 秒响应覆盖严格核心失稳样本。
- 主参照：`RBF/KNN`，也就是当前最强车辆-only 部署基线。
- 候选：RBF/KNN、keypoint residual、top-K 的 3 个 branch、top-K top1。
- 上限：best-of-3、RBF+topK oracle、RBF+keypoint+topK broad oracle，只作为诊断，不当作可部署结果。
- 运行方式：只加载已有 checkpoint 和已有样本数组，不训练，不使用生理、脑电、连续风格或驾驶员 ID。

## 目前发现了什么

- RBF/KNN test RMSE = {rbf_rmse:.6f}。
- keypoint residual test RMSE = {key_rmse:.6f}。
- top-K top1 test RMSE = {top1_rmse:.6f}。
- RBF+topK oracle test RMSE = {oracle_rmse:.6f}，比 RBF/KNN 好 {rbf_rmse - oracle_rmse:.6f}，但这是事后用真实标签选候选。
- broad oracle test RMSE = {broad_rmse:.6f}，比 RBF/KNN 好 {rbf_rmse - broad_rmse:.6f}，同样只是上限诊断。

RBF+topK oracle 在 test 上选择候选的比例：

{top_oracle_text}

## 哪些结果可信

可信的是：所有候选轨迹已经能从现有数据和 checkpoint 复现，并保存为一个 npz；RBF/KNN 仍然是当前部署主参照；oracle 上限只能说明候选池里存在潜在更好轨迹，不能说明当前 selector 可用。

## 哪些结果还不能下结论

不能把 best-of-K 或 broad oracle 说成模型性能；不能因为 oracle 好就进入生理/EEG；也不能说 top-K 已经超过 RBF/KNN，因为当前可部署 top1 和 Stage 7b selector 都没有超过 RBF/KNN。

## 下一阶段是否可以继续

可以继续 Stage 7，但下一步应该针对“选择机制”或“候选生成方式”改，而不是直接进入生理。优先做两件事：第一，利用这次导出的候选差异特征设计更严格的非 oracle selector；第二，如果候选差异太小，就重新设计候选生成，让不同候选覆盖方向、幅值、峰值时间和尾段模式。

## 推荐优先查看

1. `{figures["metric_summary"]}`
2. `{figures["disagreement_gain"]}`
3. `{figures["fixed_predictions"]}`
4. `{figures["oracle_gain_predictions"]}`
5. `{path_str(arrays_path)}`
6. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_feature_and_label_diagnosis.csv`
"""
    (REPORT_DIR / "stage07c_candidate_trajectory_export_user_summary_cn.md").write_text(user, encoding="utf-8")

    tech = f"""# Stage 7c 技术报告：candidate trajectory export v0.1

## Scope

- Track: `{TRACK_ID}`.
- Dataset split: `{SPLIT_STRATEGY}`.
- Source checkpoints:
  - `{path_str(TOPK_CKPT)}`
  - `{path_str(KEYPOINT_CKPT)}`
- No training was run.
- No server was used.
- Credential file was not read.
- Modalities used: vehicle history and causal road/event context only.
- Excluded: subject ID, continuous style, physio, EEG, test labels as inputs.

## Test Metrics

```text
{test_metric_text}
```

## Oracle Interpretation

RBF+topK oracle RMSE={oracle_rmse:.6f}, delta vs RBF={oracle_rmse - rbf_rmse:+.6f}. Broad oracle RMSE={broad_rmse:.6f}, delta vs RBF={broad_rmse - rbf_rmse:+.6f}. These rows are upper-bound diagnostics only.

## Gate

- `candidate_trajectories_exported=pass`
- `deployable_upgrade=no`
- `reason`: no non-oracle policy in this stage; previous Stage 7b selected RBF for all test samples.
- `stage08_physio_eeg_allowed=blocked`

## Tables

- `candidate_export_metrics.csv`
- `candidate_export_per_sample_metrics.csv`
- `candidate_pairwise_disagreement_long.csv`
- `candidate_pairwise_disagreement_summary.csv`
- `candidate_feature_and_label_diagnosis.csv`
- `candidate_oracle_summary.csv`
- `candidate_export_gate_table.csv`

## Figures

- `{figures["metric_summary"]}`
- `{figures["disagreement_gain"]}`
- `{figures["fixed_predictions"]}`
- `{figures["high_disagreement_predictions"]}`
- `{figures["oracle_gain_predictions"]}`

## Arrays

The replayable trajectory export is stored at `{path_str(arrays_path)}`.
"""
    (REPORT_DIR / "stage07c_candidate_trajectory_export_v0_1_cn.md").write_text(tech, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    y, y_mask, input_values, input_mask, input_time, label_time, meta = load_track_data()
    train_idx, val_idx, test_idx = split_indices(meta)

    topk_ckpt = load_checkpoint(TOPK_CKPT)
    keypoint_ckpt = load_checkpoint(KEYPOINT_CKPT)
    if topk_ckpt["track_id"] != TRACK_ID or keypoint_ckpt["track_id"] != TRACK_ID:
        raise RuntimeError("checkpoint track mismatch")
    if topk_ckpt["split_strategy"] != SPLIT_STRATEGY or keypoint_ckpt["split_strategy"] != SPLIT_STRATEGY:
        raise RuntimeError("checkpoint split strategy mismatch")
    if topk_ckpt["window_config_id"] != keypoint_v01.TRACKS[TRACK_ID]["window_config_id"]:
        raise RuntimeError("top-K checkpoint window mismatch")
    if keypoint_ckpt["window_config_id"] != keypoint_v01.TRACKS[TRACK_ID]["window_config_id"]:
        raise RuntimeError("keypoint checkpoint window mismatch")
    if int(topk_ckpt["input_downsample_step"]) != int(keypoint_ckpt["input_downsample_step"]):
        raise RuntimeError("checkpoint input downsample mismatch")

    x_model, context, context_names, _ = rebuild_context_and_inputs(
        input_values,
        input_mask,
        meta,
        train_idx,
        int(topk_ckpt["input_downsample_step"]),
    )
    validate_context(topk_ckpt["context_names"], context_names, "top-K")
    validate_context(keypoint_ckpt["context_names"], context_names, "keypoint")

    cfg = keypoint_v01.TRACKS[TRACK_ID]
    baseline_predictions, _ = clean_v01.build_strong_predictions(
        TRACK_ID,
        cfg["window_config_id"],
        y,
        y_mask,
        input_values,
        input_time,
        label_time,
        meta,
        train_idx,
        val_idx,
    )
    if RBF_MODEL not in baseline_predictions:
        raise RuntimeError(f"{RBF_MODEL} not rebuilt")

    topk_data = load_topk_predictions(x_model, context, label_time, y, y_mask, topk_ckpt)
    keypoint_pred = load_keypoint_predictions(x_model, context, label_time, keypoint_ckpt)

    predictions: dict[str, np.ndarray] = {
        RBF_MODEL: baseline_predictions[RBF_MODEL].astype(np.float32),
        KEYPOINT_MODEL: keypoint_pred,
        TOP1_MODEL: topk_data["top1"],
        BEST3_ORACLE_MODEL: topk_data["best3"],
    }
    for k, name in enumerate(BRANCH_MODELS):
        predictions[name] = topk_data["trajs"][:, k, :].astype(np.float32)

    rbf_topk_names = [RBF_MODEL, *BRANCH_MODELS]
    rbf_topk_preds = [predictions[name] for name in rbf_topk_names]
    rbf_topk_oracle_pred, rbf_topk_idx, _, rbf_topk_oracle = best_oracle(y, y_mask, rbf_topk_names, rbf_topk_preds)
    predictions[BEST_RBF_TOPK_ORACLE_MODEL] = rbf_topk_oracle_pred

    broad_names = [RBF_MODEL, KEYPOINT_MODEL, *BRANCH_MODELS]
    broad_preds = [predictions[name] for name in broad_names]
    broad_oracle_pred, broad_idx, _, broad_oracle = best_oracle(y, y_mask, broad_names, broad_preds)
    predictions[BEST_BROAD_ORACLE_MODEL] = broad_oracle_pred

    metrics, per_sample = metric_tables(y, y_mask, label_time, meta, train_idx, predictions)
    deployable_for_disagreement = [RBF_MODEL, KEYPOINT_MODEL, TOP1_MODEL, *BRANCH_MODELS]
    pair_long, pair_summary = pairwise_disagreement(meta, y_mask, predictions, deployable_for_disagreement)
    feature_diag = build_feature_and_diagnosis_table(meta, y, y_mask, predictions, topk_data, rbf_topk_oracle, broad_oracle)
    oracle_summary = pd.concat(
        [
            summarize_oracle(
                feature_diag,
                "rbf_topk",
                "label_diag__rbf_topk_oracle_model",
                "label_diag__rbf_topk_oracle_gain_over_rbf",
                "label_diag__rbf_topk_oracle_rmse",
            ),
            summarize_oracle(
                feature_diag,
                "broad_rbf_keypoint_topk",
                "label_diag__broad_oracle_model",
                "label_diag__broad_oracle_gain_over_rbf",
                "label_diag__broad_oracle_rmse",
            ),
        ],
        ignore_index=True,
    )

    candidate_names = np.array([RBF_MODEL, KEYPOINT_MODEL, TOP1_MODEL, *BRANCH_MODELS], dtype=object)
    candidate_stack = np.stack([predictions[name] for name in candidate_names], axis=1).astype(np.float32)
    arrays_path = ARRAY_DIR / "stage07c_candidate_trajectories.npz"
    np.savez_compressed(
        arrays_path,
        candidate_model_names=candidate_names,
        candidate_predictions=candidate_stack,
        y_true=y.astype(np.float32),
        y_mask=y_mask.astype(bool),
        label_time_rel_s=label_time.astype(np.float32),
        input_time_rel_s=input_time.astype(np.float32),
        sample_ids=meta["sample_id"].astype(str).to_numpy(dtype=object),
        event_uids=meta["event_uid"].astype(str).to_numpy(dtype=object),
        split=meta[SPLIT_STRATEGY].astype(str).to_numpy(dtype=object),
        topk_logits=topk_data["logits"],
        topk_probs=topk_data["probs"],
        topk_top1_idx=topk_data["top1_idx"],
        topk_best3_idx=topk_data["best3_idx"],
        rbf_topk_oracle_idx=rbf_topk_idx,
        broad_oracle_idx=broad_idx,
        pred_topk_best3_oracle=predictions[BEST3_ORACLE_MODEL].astype(np.float32),
        pred_rbf_topk_oracle=predictions[BEST_RBF_TOPK_ORACLE_MODEL].astype(np.float32),
        pred_broad_oracle=predictions[BEST_BROAD_ORACLE_MODEL].astype(np.float32),
    )

    metrics.to_csv(TABLE_DIR / "candidate_export_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(TABLE_DIR / "candidate_export_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    pair_long.to_csv(TABLE_DIR / "candidate_pairwise_disagreement_long.csv", index=False, encoding="utf-8-sig")
    pair_summary.to_csv(TABLE_DIR / "candidate_pairwise_disagreement_summary.csv", index=False, encoding="utf-8-sig")
    feature_diag.to_csv(TABLE_DIR / "candidate_feature_and_label_diagnosis.csv", index=False, encoding="utf-8-sig")
    oracle_summary.to_csv(TABLE_DIR / "candidate_oracle_summary.csv", index=False, encoding="utf-8-sig")

    fixed_ids = meta.loc[test_idx, "sample_id"].astype(str).head(12).tolist()
    high_disagreement_ids = top_rows(feature_diag, "topk_branch_spread_mean", TABLE_DIR / "candidate_high_disagreement_test_samples.csv")
    oracle_gain_ids = top_rows(feature_diag, "label_diag__rbf_topk_oracle_gain_over_rbf", TABLE_DIR / "candidate_top_oracle_gain_test_samples.csv")

    figures = {
        "metric_summary": path_str(FIG_DIR / "candidate_metric_summary_test.png"),
        "disagreement_gain": path_str(FIG_DIR / "candidate_disagreement_vs_oracle_gain_test.png"),
        "fixed_predictions": path_str(FIG_DIR / "candidate_fixed_predictions_test.png"),
        "high_disagreement_predictions": path_str(FIG_DIR / "candidate_high_disagreement_predictions_test.png"),
        "oracle_gain_predictions": path_str(FIG_DIR / "candidate_oracle_gain_predictions_test.png"),
    }
    plot_metric_summary(metrics, Path(figures["metric_summary"]))
    plot_disagreement_gain(feature_diag, Path(figures["disagreement_gain"]))
    plot_prediction_grid(Path(figures["fixed_predictions"]), fixed_ids, y, y_mask, label_time, meta, predictions, "Stage 7c fixed test candidates")
    plot_prediction_grid(
        Path(figures["high_disagreement_predictions"]),
        high_disagreement_ids,
        y,
        y_mask,
        label_time,
        meta,
        predictions,
        "Stage 7c high candidate disagreement samples",
    )
    plot_prediction_grid(
        Path(figures["oracle_gain_predictions"]),
        oracle_gain_ids,
        y,
        y_mask,
        label_time,
        meta,
        predictions,
        "Stage 7c largest RBF+topK oracle gain samples",
    )

    test_metrics = metrics[metrics["split"] == "test"].set_index("model_name")
    rbf_rmse = float(test_metrics.loc[RBF_MODEL, "rmse_steer"])
    rbf_topk_oracle_rmse = float(test_metrics.loc[BEST_RBF_TOPK_ORACLE_MODEL, "rmse_steer"])
    broad_oracle_rmse = float(test_metrics.loc[BEST_BROAD_ORACLE_MODEL, "rmse_steer"])
    top1_rmse = float(test_metrics.loc[TOP1_MODEL, "rmse_steer"])

    gate = pd.DataFrame(
        [
            {
                "gate_item": "candidate_trajectories_exported",
                "status": "pass",
                "evidence": path_str(arrays_path),
            },
            {
                "gate_item": "deployable_upgrade",
                "status": "no",
                "evidence": "Stage 7c only exports/replays candidates; no non-oracle selector is trained.",
            },
            {
                "gate_item": "rbf_topk_oracle_available",
                "status": "diagnostic_only",
                "evidence": f"test RMSE {rbf_topk_oracle_rmse:.6f}, delta vs RBF {rbf_topk_oracle_rmse - rbf_rmse:+.6f}",
            },
            {
                "gate_item": "broad_oracle_available",
                "status": "diagnostic_only",
                "evidence": f"test RMSE {broad_oracle_rmse:.6f}, delta vs RBF {broad_oracle_rmse - rbf_rmse:+.6f}",
            },
            {
                "gate_item": "stage08_physio_eeg_allowed",
                "status": "blocked",
                "evidence": "vehicle-only candidate selection is not solved; no physio/EEG increment claim is allowed.",
            },
        ]
    )
    gate.to_csv(TABLE_DIR / "candidate_export_gate_table.csv", index=False, encoding="utf-8-sig")

    run_summary = {
        "output_version": OUTPUT_VERSION,
        "track_id": TRACK_ID,
        "split_strategy": SPLIT_STRATEGY,
        "n_samples": int(len(meta)),
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "test_n": int(len(test_idx)),
        "rbf_test_rmse": rbf_rmse,
        "topk_top1_test_rmse": top1_rmse,
        "rbf_topk_oracle_test_rmse": rbf_topk_oracle_rmse,
        "rbf_topk_oracle_delta_vs_rbf": rbf_topk_oracle_rmse - rbf_rmse,
        "broad_oracle_test_rmse": broad_oracle_rmse,
        "broad_oracle_delta_vs_rbf": broad_oracle_rmse - rbf_rmse,
        "candidate_arrays_path": path_str(arrays_path),
        "metrics_path": path_str(TABLE_DIR / "candidate_export_metrics.csv"),
        "feature_diagnosis_path": path_str(TABLE_DIR / "candidate_feature_and_label_diagnosis.csv"),
        "figures": figures,
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "uses_subject_id": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (LOG_DIR / "stage07c_candidate_trajectory_export_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_reports(metrics, oracle_summary, feature_diag, figures, arrays_path)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
