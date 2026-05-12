# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import sys
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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
TOPK_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_topk_vehicle_transformer_v0_1"
TOPK_TABLE_DIR = TOPK_DIR / "tables"
TOPK_CKPT = TOPK_DIR / "checkpoints" / "B_response3s_strict_core_topk_vehicle_transformer_top1_no_subject_best.pt"
OUT_ROOT = ROOT / "03_baselines" / "stage03_vehicle_instability_topk_reliability_selector_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_ROOT = ROOT / "09_reports"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402
import stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1 as clean_v01  # noqa: E402
import stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1 as keypoint_v01  # noqa: E402
import stage03_vehicle_instability_topk_vehicle_transformer_v0_1 as topk_v01  # noqa: E402


OUTPUT_VERSION = "stage03_vehicle_instability_topk_reliability_selector_v0_1"
TRACK_ID = "B_response3s_strict_core"
SPLIT_STRATEGY = "session_level_split"
RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
TOP1_MODEL = "topk_vehicle_transformer_top1_no_subject"
BESTK_MODEL = "topk_vehicle_transformer_best_of_3_oracle"
BRANCH_MODELS = [f"topk_vehicle_transformer_branch{k}_no_subject" for k in range(3)]
BRANCH_SELECTOR = "topk_branch_logreg_selector_no_subject"
CANDIDATE_SELECTOR = "topk_rbf_branch_logreg_selector_no_subject"
TOP1_RBF_FALLBACK = "topk_top1_rbf_fallback_logreg_no_subject"
ORACLE_RBF_TOPK = "oracle_best_of_rbf_plus_topk_upper_bound"

NUMERIC_CONTEXT_COLS = [
    "anchor_time_rel_s",
    "curvature_anchor",
    "input_valid_ratio",
    "median_speed_kmh_window",
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


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_checkpoint_predictions() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    if not TOPK_CKPT.exists():
        raise FileNotFoundError(f"missing top-K checkpoint: {TOPK_CKPT}")

    y, y_mask, input_values, input_mask, input_time, label_time, meta = topk_v01.load_track()
    train_idx, val_idx, test_idx = topk_v01.split_indices(meta)
    x_scaled, _ = keypoint_v01.standardize_vehicle_inputs(input_values, input_mask, train_idx)
    context, _ = keypoint_v01.build_context_features(meta, train_idx)
    step = max(1, int(round(len(input_time) / topk_v01.TARGET_INPUT_TOKENS)))
    x_model = x_scaled[:, ::step, :].copy()

    ckpt = torch.load(TOPK_CKPT, map_location="cpu")
    label_scale = float(ckpt.get("label_scale", keypoint_v01.label_scale_train(y, y_mask, train_idx)))
    k = int(ckpt.get("k", 3))
    model = topk_v01.TopKVehicleTransformer(
        vehicle_dim=x_model.shape[2],
        context_dim=context.shape[1],
        label_time=label_time,
        k=k,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    trajs, logits = topk_v01.predict_all(model, x_model, context, label_scale, batch_size=32)
    top1, bestk, top1_idx, best_idx, probs = topk_v01.select_top1_and_bestk(trajs, logits, y, y_mask)

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
        raise RuntimeError(f"{RBF_MODEL} was not rebuilt")

    base_predictions: dict[str, np.ndarray] = {
        RBF_MODEL: baseline_predictions[RBF_MODEL],
        TOP1_MODEL: top1,
        BESTK_MODEL: bestk,
    }
    for i, model_name in enumerate(BRANCH_MODELS):
        base_predictions[model_name] = trajs[:, i, :]

    aux = {
        "top1_idx": top1_idx,
        "best_idx": best_idx,
        "probs": probs,
        "trajs": trajs,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    return y, y_mask, label_time, meta, train_idx, val_idx, test_idx, base_predictions, aux, baseline_predictions


def evaluate_predictions(
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_peak = np.nanmax(np.abs(np.where(y_mask, y, np.nan)), axis=1)
    large_thr = float(np.nanpercentile(gt_peak[train_idx], 75))
    difficult_thr = float(np.nanpercentile(gt_peak[train_idx], 80))
    cfg = keypoint_v01.TRACKS[TRACK_ID]
    rows: list[dict[str, Any]] = []
    split_values = meta[SPLIT_STRATEGY].astype(str).to_numpy()
    for split_name in ["train", "val", "test"]:
        split_mask = split_values == split_name
        if not split_mask.any():
            continue
        split_meta = meta.loc[split_mask].reset_index(drop=True)
        for model_name, pred in predictions.items():
            rows.extend(
                eval_utils.sample_metric_rows(
                    y[split_mask],
                    pred[split_mask],
                    y_mask[split_mask],
                    label_time,
                    split_meta,
                    model_name,
                    SPLIT_STRATEGY,
                    split_name,
                    cfg["window_config_id"],
                    large_thr,
                    difficult_thr,
                )
            )
    per_sample = pd.DataFrame(rows)
    per_sample["track_id"] = TRACK_ID
    metrics = eval_utils.aggregate_metrics(per_sample)
    metrics["track_id"] = TRACK_ID
    return metrics, per_sample


def prefixed_metrics(per_sample: pd.DataFrame, model_name: str, prefix: str) -> pd.DataFrame:
    keep = [
        "sample_id",
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
        "pred_peak_abs",
        "pred_reversal_count",
        "pred_multi_segment",
        "gt_peak_abs",
        "is_large_response",
        "is_difficult_peak_top20",
    ]
    out = per_sample[per_sample["model_name"] == model_name][keep].copy()
    return out.rename(columns={c: f"{prefix}_{c}" for c in keep if c != "sample_id"})


def build_feature_table(
    meta: pd.DataFrame,
    per_sample: pd.DataFrame,
    aux: dict[str, np.ndarray],
) -> pd.DataFrame:
    diag = pd.read_csv(TOPK_TABLE_DIR / "topk_vehicle_transformer_branch_diagnostics.csv")
    diag = diag[
        [
            "sample_id",
            "top1_branch",
            "best_branch_oracle",
            "top1_matches_best",
            "top1_prob",
            "prob_margin",
            "branch_spread_mean",
            "branch_spread_peak",
        ]
    ].copy()
    base_cols = [
        "sample_id",
        "event_uid",
        "subject",
        "session_stamp",
        SPLIT_STRATEGY,
        *NUMERIC_CONTEXT_COLS,
        *CATEGORICAL_CONTEXT_COLS,
    ]
    base_cols = [c for c in base_cols if c in meta.columns]
    table = meta[base_cols].drop_duplicates("sample_id").copy()
    table = table.rename(columns={SPLIT_STRATEGY: "split"})
    table = table.merge(diag, on="sample_id", how="left", validate="one_to_one")

    probs = np.asarray(aux["probs"])
    prob_df = pd.DataFrame(
        {
            "sample_id": meta["sample_id"].astype(str).to_numpy(),
            "branch0_prob": probs[:, 0],
            "branch1_prob": probs[:, 1],
            "branch2_prob": probs[:, 2],
            "prob_entropy": -np.sum(np.clip(probs, 1e-8, 1.0) * np.log(np.clip(probs, 1e-8, 1.0)), axis=1),
        }
    )
    table = table.merge(prob_df, on="sample_id", how="left", validate="one_to_one")

    for model_name, prefix in [(RBF_MODEL, "rbf"), (TOP1_MODEL, "top1"), *[(m, f"branch{i}") for i, m in enumerate(BRANCH_MODELS)]]:
        table = table.merge(prefixed_metrics(per_sample, model_name, prefix), on="sample_id", how="left", validate="one_to_one")

    candidate_cols = {
        "rbf": "rbf_sample_rmse",
        "branch0": "branch0_sample_rmse",
        "branch1": "branch1_sample_rmse",
        "branch2": "branch2_sample_rmse",
    }
    rmse_mat = table[list(candidate_cols.values())].to_numpy(dtype=float)
    labels = np.asarray(list(candidate_cols.keys()), dtype=object)
    table["best_candidate_oracle"] = labels[np.nanargmin(rmse_mat, axis=1)]
    table["top1_worse_than_rbf"] = (table["top1_sample_rmse"] > table["rbf_sample_rmse"]).astype(int)
    table["top1_minus_rbf_rmse"] = table["top1_sample_rmse"] - table["rbf_sample_rmse"]
    table["best_of_rbf_topk_rmse"] = np.nanmin(rmse_mat, axis=1)
    table["best_of_rbf_topk_gain_over_rbf"] = table["rbf_sample_rmse"] - table["best_of_rbf_topk_rmse"]
    return table


def feature_columns(table: pd.DataFrame, include_rbf: bool) -> tuple[list[str], list[str]]:
    numeric = [
        c
        for c in [
            *NUMERIC_CONTEXT_COLS,
            "top1_branch",
            "top1_prob",
            "prob_margin",
            "branch_spread_mean",
            "branch_spread_peak",
            "branch0_prob",
            "branch1_prob",
            "branch2_prob",
            "prob_entropy",
        ]
        if c in table.columns
    ]
    prefixes = [f"branch{i}" for i in range(3)]
    if include_rbf:
        prefixes.append("rbf")
    for prefix in prefixes:
        for feat in PRED_FEATURES:
            col = f"{prefix}_{feat}"
            if col in table.columns:
                numeric.append(col)
    numeric = list(dict.fromkeys(numeric))
    categorical = [c for c in CATEGORICAL_CONTEXT_COLS if c in table.columns]
    return numeric, categorical


def make_pipeline(numeric: list[str], categorical: list[str], class_weight: str | None = "balanced") -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("scale", StandardScaler())]), numeric),
            ("cat", one_hot_encoder(), categorical),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(max_iter=1500, class_weight=class_weight, random_state=20260513)
    return Pipeline([("preprocess", pre), ("clf", clf)])


def choose_prediction_arrays(
    selected: pd.Series,
    base_predictions: dict[str, np.ndarray],
    model_name: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    sample_count = len(selected)
    first = next(iter(base_predictions.values()))
    out = np.zeros((sample_count, first.shape[1]), dtype=np.float32)
    decisions = pd.DataFrame({"row_idx": np.arange(sample_count), "selected_source": selected.to_numpy()})
    source_to_model = {
        "rbf": RBF_MODEL,
        "top1": TOP1_MODEL,
        "bestk": BESTK_MODEL,
        "branch0": BRANCH_MODELS[0],
        "branch1": BRANCH_MODELS[1],
        "branch2": BRANCH_MODELS[2],
    }
    for source, pred_name in source_to_model.items():
        mask = selected.to_numpy() == source
        if mask.any():
            out[mask] = base_predictions[pred_name][mask]
    decisions["model_name"] = model_name
    return out, decisions


def train_selectors(table: pd.DataFrame, base_predictions: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = table["split"].astype(str) == "train"
    val = table["split"].astype(str) == "val"
    selector_predictions: dict[str, np.ndarray] = {}
    decision_frames: list[pd.DataFrame] = []
    info: dict[str, Any] = {}

    branch_numeric, branch_categorical = feature_columns(table, include_rbf=False)
    branch_pipe = make_pipeline(branch_numeric, branch_categorical, class_weight="balanced")
    branch_pipe.fit(table.loc[train, branch_numeric + branch_categorical], table.loc[train, "best_branch_oracle"].astype(int))
    branch_pred = branch_pipe.predict(table[branch_numeric + branch_categorical]).astype(int)
    branch_sources = pd.Series([f"branch{i}" for i in branch_pred], index=table.index)
    pred, dec = choose_prediction_arrays(branch_sources, base_predictions, BRANCH_SELECTOR)
    selector_predictions[BRANCH_SELECTOR] = pred
    dec["sample_id"] = table["sample_id"].to_numpy()
    dec["split"] = table["split"].to_numpy()
    decision_frames.append(dec)
    info["branch_selector_features"] = branch_numeric + branch_categorical

    cand_numeric, cand_categorical = feature_columns(table, include_rbf=True)
    cand_pipe = make_pipeline(cand_numeric, cand_categorical, class_weight="balanced")
    cand_pipe.fit(table.loc[train, cand_numeric + cand_categorical], table.loc[train, "best_candidate_oracle"].astype(str))
    cand_sources = pd.Series(cand_pipe.predict(table[cand_numeric + cand_categorical]).astype(str), index=table.index)
    pred, dec = choose_prediction_arrays(cand_sources, base_predictions, CANDIDATE_SELECTOR)
    selector_predictions[CANDIDATE_SELECTOR] = pred
    dec["sample_id"] = table["sample_id"].to_numpy()
    dec["split"] = table["split"].to_numpy()
    decision_frames.append(dec)
    info["candidate_selector_features"] = cand_numeric + cand_categorical

    fallback_pipe = make_pipeline(cand_numeric, cand_categorical, class_weight="balanced")
    fallback_pipe.fit(table.loc[train, cand_numeric + cand_categorical], table.loc[train, "top1_worse_than_rbf"].astype(int))
    fallback_prob = fallback_pipe.predict_proba(table[cand_numeric + cand_categorical])[:, 1]
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    sweep_rows: list[dict[str, Any]] = []
    for thr in thresholds:
        src = pd.Series(np.where(fallback_prob >= thr, "rbf", "top1"), index=table.index)
        for split_name in ["train", "val"]:
            mask = table["split"].astype(str) == split_name
            rmse_values = np.where(src[mask].to_numpy() == "rbf", table.loc[mask, "rbf_sample_rmse"], table.loc[mask, "top1_sample_rmse"])
            sweep_rows.append(
                {
                    "threshold": float(thr),
                    "split": split_name,
                    "selector_rmse": float(np.sqrt(np.mean(np.square(rmse_values.astype(float))))),
                    "rbf_fallback_rate": float((src[mask] == "rbf").mean()),
                }
            )
    sweep = pd.DataFrame(sweep_rows)
    val_sweep = sweep[sweep["split"] == "val"].copy()
    val_sweep = val_sweep.sort_values(["selector_rmse", "rbf_fallback_rate", "threshold"], ascending=[True, True, True])
    selected_threshold = float(val_sweep.iloc[0]["threshold"]) if not val_sweep.empty else 0.50
    fallback_sources = pd.Series(np.where(fallback_prob >= selected_threshold, "rbf", "top1"), index=table.index)
    pred, dec = choose_prediction_arrays(fallback_sources, base_predictions, TOP1_RBF_FALLBACK)
    selector_predictions[TOP1_RBF_FALLBACK] = pred
    dec["sample_id"] = table["sample_id"].to_numpy()
    dec["split"] = table["split"].to_numpy()
    dec["fallback_prob_top1_worse_than_rbf"] = fallback_prob
    dec["fallback_threshold"] = selected_threshold
    decision_frames.append(dec)
    info["fallback_features"] = cand_numeric + cand_categorical
    info["selected_fallback_threshold"] = selected_threshold

    oracle_sources = table["best_candidate_oracle"].astype(str)
    pred, dec = choose_prediction_arrays(oracle_sources, base_predictions, ORACLE_RBF_TOPK)
    selector_predictions[ORACLE_RBF_TOPK] = pred
    dec["sample_id"] = table["sample_id"].to_numpy()
    dec["split"] = table["split"].to_numpy()
    decision_frames.append(dec)

    decisions = pd.concat(decision_frames, ignore_index=True)
    return selector_predictions, decisions, sweep, info


def summarize_selected_models(metrics: pd.DataFrame, decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = metrics[metrics["split"] == "test"].copy()
    val = metrics[metrics["split"] == "val"].copy()
    rows = []
    candidate_models = [BRANCH_SELECTOR, CANDIDATE_SELECTOR, TOP1_RBF_FALLBACK]
    for model in candidate_models:
        val_row = val[val["model_name"] == model]
        test_row = test[test["model_name"] == model]
        if val_row.empty or test_row.empty:
            continue
        rows.append(
            {
                "selector_model": model,
                "val_rmse": float(val_row.iloc[0]["rmse_steer"]),
                "test_rmse": float(test_row.iloc[0]["rmse_steer"]),
                "test_wrong_side_rate": float(test_row.iloc[0]["wrong_side_rate"]),
                "test_large_response_recall": float(test_row.iloc[0]["large_response_recall"]),
                "test_difficult_top20_rmse": float(test_row.iloc[0]["difficult_top20_rmse"]),
            }
        )
    selection = pd.DataFrame(rows).sort_values(["val_rmse", "test_rmse"], ascending=[True, True])
    chosen = selection.iloc[0]["selector_model"] if not selection.empty else TOP1_RBF_FALLBACK
    for model in candidate_models:
        decisions.loc[decisions["model_name"] == model, "chosen_by_val_rmse"] = int(model == chosen)
    return selection, decisions


def group_summary(per_sample: pd.DataFrame, table: pd.DataFrame, chosen_model: str, field: str) -> pd.DataFrame:
    if field not in table.columns and field not in per_sample.columns:
        return pd.DataFrame()
    model_rows = per_sample[(per_sample["split"] == "test") & (per_sample["model_name"] == chosen_model)].copy()
    if field in model_rows.columns:
        merged = model_rows
    else:
        merged = model_rows.merge(table[["sample_id", field]], on="sample_id", how="left")
    return (
        merged.groupby(field, dropna=False)
        .agg(
            n_samples=("sample_id", "count"),
            rmse_steer=("sample_rmse", lambda x: float(np.sqrt(np.mean(np.square(x.astype(float)))))),
            wrong_side_rate=("wrong_side", "mean"),
            large_response_recall=("large_response_recalled", "mean"),
            severe_amp_under_rate=("severe_amp_under", "mean"),
            difficult_top20_rate=("is_difficult_peak_top20", "mean"),
        )
        .reset_index()
        .sort_values(["rmse_steer", "n_samples"], ascending=[False, False])
    )


def sample_indices(meta: pd.DataFrame, sample_ids: list[str]) -> list[int]:
    lookup = {sid: i for i, sid in enumerate(meta["sample_id"].astype(str).tolist())}
    return [lookup[sid] for sid in sample_ids if sid in lookup]


def plot_prediction_grid(
    path: Path,
    sample_ids: list[str],
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    plot_models: list[tuple[str, str, str, str]],
    title: str,
) -> None:
    idxs = sample_indices(meta, sample_ids)[:12]
    if not idxs:
        return
    ncols = 3
    nrows = int(math.ceil(len(idxs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, max(3.0, 2.55 * nrows)), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr[len(idxs) :]:
        ax.axis("off")
    for ax, idx in zip(axes_arr, idxs):
        sid = str(meta.iloc[idx]["sample_id"])
        valid = y_mask[idx] & np.isfinite(y[idx])
        ax.plot(label_time[valid], y[idx, valid], color="#111111", linewidth=1.8, label="gt")
        for model_name, color, label, style in plot_models:
            pred = predictions[model_name][idx]
            ax.plot(label_time[valid], pred[valid], color=color, linestyle=style, linewidth=1.15, alpha=0.9, label=label)
        short_id = sid.split("__")[-2] if "__" in sid else sid[-10:]
        ax.set_title(short_id, fontsize=8)
        ax.grid(True, alpha=0.22)
    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), fontsize=9)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metric_summary(path: Path, metrics: pd.DataFrame, chosen_model: str) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    order = [RBF_MODEL, TOP1_MODEL, BRANCH_SELECTOR, CANDIDATE_SELECTOR, TOP1_RBF_FALLBACK, chosen_model, BESTK_MODEL, ORACLE_RBF_TOPK]
    order = list(dict.fromkeys([m for m in order if m in test["model_name"].values]))
    test = test.set_index("model_name").loc[order].reset_index()
    colors = ["#1f77b4", "#d62728", "#9467bd", "#8c564b", "#ff7f0e", "#17becf", "#2ca02c", "#111111"][: len(test)]
    x = np.arange(len(test))
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2))
    cols = [
        ("rmse_steer", "RMSE"),
        ("wrong_side_rate", "Wrong side"),
        ("large_response_recall", "Large recall"),
        ("difficult_top20_rmse", "Difficult RMSE"),
    ]
    for ax, (col, title) in zip(axes, cols):
        ax.bar(x, test[col], color=colors)
        ax.set_title(title)
        ax.set_xticks(x, [m.replace("_no_subject", "").replace("topk_", "") for m in test["model_name"]], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_decision_counts(path: Path, decisions: pd.DataFrame) -> None:
    test = decisions[decisions["split"] == "test"].copy()
    pivot = test.groupby(["model_name", "selected_source"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("test samples")
    ax.set_title("Selector decisions on test")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_fallback_scatter(path: Path, table: pd.DataFrame, decisions: pd.DataFrame) -> None:
    dec = decisions[decisions["model_name"] == TOP1_RBF_FALLBACK][
        ["sample_id", "fallback_prob_top1_worse_than_rbf", "selected_source", "split"]
    ].copy()
    df = table.merge(dec, on=["sample_id", "split"], how="inner")
    df = df[df["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    colors = np.where(df["selected_source"] == "rbf", "#1f77b4", "#d62728")
    ax.scatter(df["fallback_prob_top1_worse_than_rbf"], df["top1_minus_rbf_rmse"], c=colors, s=50, alpha=0.85)
    ax.axhline(0.0, color="#222222", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Predicted risk: top1 worse than RBF")
    ax.set_ylabel("Top1 RMSE - RBF RMSE")
    ax.set_title("RBF fallback diagnostic on test")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def report_value(metrics: pd.DataFrame, model: str, col: str, split: str = "test") -> float:
    row = metrics[(metrics["split"] == split) & (metrics["model_name"] == model)]
    return float(row.iloc[0][col]) if not row.empty else float("nan")


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


def write_reports(
    metrics: pd.DataFrame,
    selection: pd.DataFrame,
    decisions: pd.DataFrame,
    chosen_model: str,
    info: dict[str, Any],
    figures: dict[str, str],
) -> None:
    chosen_label = chosen_model.replace("_no_subject", "")
    chosen_dec = decisions[(decisions["split"] == "test") & (decisions["model_name"] == chosen_model)]
    decision_counts = chosen_dec["selected_source"].value_counts().to_dict()
    rbf_rmse = report_value(metrics, RBF_MODEL, "rmse_steer")
    chosen_rmse = report_value(metrics, chosen_model, "rmse_steer")
    chosen_minus_rbf = chosen_rmse - rbf_rmse
    if np.isfinite(chosen_minus_rbf) and chosen_minus_rbf < 0:
        deployable_conclusion = f"validation 选中的策略比 RBF 低 {-chosen_minus_rbf:.6f} RMSE，可继续复核。"
    else:
        deployable_conclusion = f"validation 选中的策略没有超过 RBF，test RMSE 比 RBF 高 {chosen_minus_rbf:.6f}；本轮不能升级为强车辆基线。"
    user = f"""# 阶段 3 用户查看版：top-K 可靠性选择/回退 v0.1

## 这个阶段为什么做

上一轮 top-K 的 best-of-3 很好，但 top-1 选不中：test 上 top-1 和 best 分支一致率只有 0.300。这个阶段不重新训练轨迹模型，而是检查“能不能用车辆-only、事件前可得的信息，把 top-K 的候选分支选得更好，或者在不可靠时回退到 RBF/KNN 类强车辆基线”。

## 这个阶段检查了什么

- `branch_logreg`：只在 3 条 top-K 分支里选一条。
- `candidate_logreg`：在 RBF 与 3 条 top-K 分支之间直接选择。
- `top1_rbf_fallback`：先用 top-K 自己的 top-1，若可靠性模型认为 top-1 会比 RBF 差，则回退到 RBF。
- `best-of-3` 和 `best-of-RBF+topK` 只作为事后上限，不作为可部署结果。

## 目前发现了什么

- RBF test RMSE={report_value(metrics, RBF_MODEL, "rmse_steer"):.6f}，错侧率={report_value(metrics, RBF_MODEL, "wrong_side_rate"):.3f}，大幅响应召回={report_value(metrics, RBF_MODEL, "large_response_recall"):.3f}。
- top-K top-1 test RMSE={report_value(metrics, TOP1_MODEL, "rmse_steer"):.6f}，错侧率={report_value(metrics, TOP1_MODEL, "wrong_side_rate"):.3f}，大幅响应召回={report_value(metrics, TOP1_MODEL, "large_response_recall"):.3f}。
- 按 validation RMSE 选中的可靠性策略是 `{chosen_label}`，test RMSE={report_value(metrics, chosen_model, "rmse_steer"):.6f}，错侧率={report_value(metrics, chosen_model, "wrong_side_rate"):.3f}，大幅响应召回={report_value(metrics, chosen_model, "large_response_recall"):.3f}。
- `{chosen_label}` 在 test 上的选择来源计数：{decision_counts}。
- 结论：{deployable_conclusion}
- best-of-RBF+topK 上限 test RMSE={report_value(metrics, ORACLE_RBF_TOPK, "rmse_steer"):.6f}，说明候选池仍有明显潜力，但选择机制还没有完全吃到。

## 哪些结果可信

可信的是：本轮选择器只用 train 训练，`top1_rbf_fallback` 的阈值只用 val 固定；输入特征只来自事件前车辆/道路上下文、候选模型自己的预测形态和 top-K 概率，不使用 subject ID、生理、脑电、连续风格，也不把 test 标签用于训练标准化或阈值选择。

## 哪些结果还不能下结论

不能把 best-of-RBF+topK 当成真实部署性能；它是事后知道哪条轨迹最接近真值的上限。若 validation 选中的策略 test 仍不能稳定超过 RBF，就不能说 top-K 可靠性选择已经解决问题，只能说“候选池有潜力，选择头仍需改进”。

## 下一阶段是否可以继续

可以继续阶段 3，但仍不进入风格、生理或 EEG 有效性结论。下一步应把可靠性选择作为诊断结果，决定是否做关键点条件多假设、分响应类型的选择头，或者回到更稳的 RBF/KNN 类强车辆基线作为暂定主参照。

## 推荐优先查看

1. `{figures["metric_summary"]}`
2. `{figures["fixed"]}`
3. `{figures["bad"]}`
4. `{figures["decision_counts"]}`
5. `{figures["fallback_scatter"]}`
"""
    tech = f"""# 阶段 3 技术报告：top-K 可靠性选择/回退 v0.1

## 范围

- 轨道：`{TRACK_ID}`。
- 输入：事件前车辆历史、因果可得道路/事件上下文、候选轨迹自身的预测形态、top-K 分支概率。
- 不使用：subject ID、生理、脑电、连续风格、服务器、服务器密码文件。
- 训练协议：选择器仅 train 拟合；回退阈值仅 val 固定；test 只报告。

## validation 选择

{markdown_table(selection)}

## test 指标

| 模型 | RMSE | 错侧率 | 大幅召回 | 困难 top20 RMSE |
|---|---:|---:|---:|---:|
| RBF | {report_value(metrics, RBF_MODEL, "rmse_steer"):.6f} | {report_value(metrics, RBF_MODEL, "wrong_side_rate"):.3f} | {report_value(metrics, RBF_MODEL, "large_response_recall"):.3f} | {report_value(metrics, RBF_MODEL, "difficult_top20_rmse"):.6f} |
| top-1 | {report_value(metrics, TOP1_MODEL, "rmse_steer"):.6f} | {report_value(metrics, TOP1_MODEL, "wrong_side_rate"):.3f} | {report_value(metrics, TOP1_MODEL, "large_response_recall"):.3f} | {report_value(metrics, TOP1_MODEL, "difficult_top20_rmse"):.6f} |
| branch selector | {report_value(metrics, BRANCH_SELECTOR, "rmse_steer"):.6f} | {report_value(metrics, BRANCH_SELECTOR, "wrong_side_rate"):.3f} | {report_value(metrics, BRANCH_SELECTOR, "large_response_recall"):.3f} | {report_value(metrics, BRANCH_SELECTOR, "difficult_top20_rmse"):.6f} |
| candidate selector | {report_value(metrics, CANDIDATE_SELECTOR, "rmse_steer"):.6f} | {report_value(metrics, CANDIDATE_SELECTOR, "wrong_side_rate"):.3f} | {report_value(metrics, CANDIDATE_SELECTOR, "large_response_recall"):.3f} | {report_value(metrics, CANDIDATE_SELECTOR, "difficult_top20_rmse"):.6f} |
| top1-RBF fallback | {report_value(metrics, TOP1_RBF_FALLBACK, "rmse_steer"):.6f} | {report_value(metrics, TOP1_RBF_FALLBACK, "wrong_side_rate"):.3f} | {report_value(metrics, TOP1_RBF_FALLBACK, "large_response_recall"):.3f} | {report_value(metrics, TOP1_RBF_FALLBACK, "difficult_top20_rmse"):.6f} |
| best-of-3 oracle | {report_value(metrics, BESTK_MODEL, "rmse_steer"):.6f} | {report_value(metrics, BESTK_MODEL, "wrong_side_rate"):.3f} | {report_value(metrics, BESTK_MODEL, "large_response_recall"):.3f} | {report_value(metrics, BESTK_MODEL, "difficult_top20_rmse"):.6f} |
| best-of-RBF+topK oracle | {report_value(metrics, ORACLE_RBF_TOPK, "rmse_steer"):.6f} | {report_value(metrics, ORACLE_RBF_TOPK, "wrong_side_rate"):.3f} | {report_value(metrics, ORACLE_RBF_TOPK, "large_response_recall"):.3f} | {report_value(metrics, ORACLE_RBF_TOPK, "difficult_top20_rmse"):.6f} |

## 选择器信息

- validation 选中策略：`{chosen_model}`。
- top1-RBF fallback 阈值：{info.get("selected_fallback_threshold", float("nan"))}。
- branch selector 特征数：{len(info.get("branch_selector_features", []))}。
- candidate/fallback 特征数：{len(info.get("candidate_selector_features", []))}。

## 结论

本轮用于判断 top-K 的问题是不是“候选覆盖有潜力但选择机制不足”。本轮可部署选择策略结论：{deployable_conclusion} oracle 只能说明上限空间，不能作为结论性能。
"""
    (REPORT_ROOT / "stage03_vehicle_instability_topk_reliability_selector_user_summary_cn.md").write_text(user, encoding="utf-8")
    (REPORT_ROOT / "stage03_vehicle_instability_topk_reliability_selector_v0_1_cn.md").write_text(tech, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    y, y_mask, label_time, meta, train_idx, val_idx, test_idx, base_predictions, aux, _ = load_checkpoint_predictions()
    base_metrics, base_per_sample = evaluate_predictions(
        y,
        y_mask,
        label_time,
        meta,
        train_idx,
        {k: v for k, v in base_predictions.items() if k in [RBF_MODEL, TOP1_MODEL, BESTK_MODEL, *BRANCH_MODELS]},
    )
    table = build_feature_table(meta, base_per_sample, aux)
    selector_predictions, decisions, sweep, info = train_selectors(table, base_predictions)
    all_predictions = {**base_predictions, **selector_predictions}
    metrics, per_sample = evaluate_predictions(y, y_mask, label_time, meta, train_idx, all_predictions)
    selection, decisions = summarize_selected_models(metrics, decisions)
    chosen_model = str(selection.iloc[0]["selector_model"]) if not selection.empty else TOP1_RBF_FALLBACK

    subject_summary = group_summary(per_sample, table, chosen_model, "subject")
    road_module_summary = group_summary(per_sample, table, chosen_model, "road_design_module_name")
    event_level_summary = group_summary(per_sample, table, chosen_model, "event_level")

    metrics.to_csv(TABLE_DIR / "topk_reliability_selector_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(TABLE_DIR / "topk_reliability_selector_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    table.to_csv(TABLE_DIR / "topk_reliability_selector_feature_table.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(TABLE_DIR / "topk_reliability_selector_decisions.csv", index=False, encoding="utf-8-sig")
    sweep.to_csv(TABLE_DIR / "topk_reliability_selector_threshold_sweep.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(TABLE_DIR / "topk_reliability_selector_validation_selection.csv", index=False, encoding="utf-8-sig")
    subject_summary.to_csv(TABLE_DIR / "topk_reliability_selector_subject_summary.csv", index=False, encoding="utf-8-sig")
    road_module_summary.to_csv(TABLE_DIR / "topk_reliability_selector_road_module_summary.csv", index=False, encoding="utf-8-sig")
    event_level_summary.to_csv(TABLE_DIR / "topk_reliability_selector_event_level_summary.csv", index=False, encoding="utf-8-sig")

    test_chosen = per_sample[(per_sample["split"] == "test") & (per_sample["model_name"] == chosen_model)].copy()
    fixed_ids = meta.loc[test_idx, "sample_id"].astype(str).head(12).tolist()
    bad_ids = test_chosen.sort_values("sample_rmse", ascending=False).head(12)["sample_id"].astype(str).tolist()
    gain_table = table[table["split"] == "test"].copy()
    gain_ids = gain_table.sort_values("best_of_rbf_topk_gain_over_rbf", ascending=False).head(12)["sample_id"].astype(str).tolist()
    pd.DataFrame({"track_id": TRACK_ID, "sample_id": fixed_ids}).to_csv(TABLE_DIR / "topk_reliability_selector_fixed_plot_samples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"track_id": TRACK_ID, "sample_id": bad_ids}).to_csv(TABLE_DIR / "topk_reliability_selector_bad_plot_samples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"track_id": TRACK_ID, "sample_id": gain_ids}).to_csv(TABLE_DIR / "topk_reliability_selector_oracle_gain_plot_samples.csv", index=False, encoding="utf-8-sig")

    plot_predictions = {
        RBF_MODEL: all_predictions[RBF_MODEL],
        TOP1_MODEL: all_predictions[TOP1_MODEL],
        chosen_model: all_predictions[chosen_model],
        BESTK_MODEL: all_predictions[BESTK_MODEL],
        ORACLE_RBF_TOPK: all_predictions[ORACLE_RBF_TOPK],
    }
    plot_models = [
        (RBF_MODEL, "#1f77b4", "RBF", "-"),
        (TOP1_MODEL, "#d62728", "top1", "-"),
        (chosen_model, "#ff7f0e", "chosen", "-"),
        (BESTK_MODEL, "#2ca02c", "bestK", "--"),
        (ORACLE_RBF_TOPK, "#111111", "oracle+", ":"),
    ]
    fixed_fig = FIG_DIR / "topk_reliability_selector_fixed_predictions_test.png"
    bad_fig = FIG_DIR / "topk_reliability_selector_bad_samples_test.png"
    gain_fig = FIG_DIR / "topk_reliability_selector_oracle_gain_samples_test.png"
    metric_fig = FIG_DIR / "topk_reliability_selector_metric_summary_test.png"
    decision_fig = FIG_DIR / "topk_reliability_selector_decision_counts_test.png"
    fallback_fig = FIG_DIR / "topk_reliability_selector_fallback_scatter_test.png"
    plot_prediction_grid(fixed_fig, fixed_ids, y, y_mask, label_time, meta, plot_predictions, plot_models, "Top-K reliability selector fixed test samples")
    plot_prediction_grid(bad_fig, bad_ids, y, y_mask, label_time, meta, plot_predictions, plot_models, "Top-K reliability selector worst chosen test samples")
    plot_prediction_grid(gain_fig, gain_ids, y, y_mask, label_time, meta, plot_predictions, plot_models, "Best-of-RBF+topK largest gains over RBF")
    plot_metric_summary(metric_fig, metrics, chosen_model)
    plot_decision_counts(decision_fig, decisions)
    plot_fallback_scatter(fallback_fig, table, decisions)

    figures = {
        "fixed": str(fixed_fig).replace("\\", "/"),
        "bad": str(bad_fig).replace("\\", "/"),
        "oracle_gain": str(gain_fig).replace("\\", "/"),
        "metric_summary": str(metric_fig).replace("\\", "/"),
        "decision_counts": str(decision_fig).replace("\\", "/"),
        "fallback_scatter": str(fallback_fig).replace("\\", "/"),
    }
    write_reports(metrics, selection, decisions, chosen_model, info, figures)

    summary = {
        "output_version": OUTPUT_VERSION,
        "track_id": TRACK_ID,
        "chosen_model_by_val_rmse": chosen_model,
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "uses_subject_id": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "selector_train_split": "train only",
        "fallback_threshold_split": "val only",
        "test_is_report_only": True,
        "metrics_path": str(TABLE_DIR / "topk_reliability_selector_metrics.csv").replace("\\", "/"),
        "selection_path": str(TABLE_DIR / "topk_reliability_selector_validation_selection.csv").replace("\\", "/"),
        "figures": figures,
        "selected_fallback_threshold": info.get("selected_fallback_threshold"),
    }
    (LOG_DIR / "topk_reliability_selector_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
