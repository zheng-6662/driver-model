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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path("F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
BASELINE_SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(BASELINE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402


OUTPUT_VERSION = "stage07f_response_factorized_candidates_v0_1"
TRACK_ID = "B_response3s_strict_core"
SPLIT_STRATEGY = "session_level_split"

STAGE7C_ROOT = ROOT / "07_multihypothesis" / "stage07c_candidate_trajectory_export_v0_1"
STAGE7E_ROOT = ROOT / "07_multihypothesis" / "stage07e_candidate_generation_redesign_v0_1"
TRAJECTORY_NPZ = STAGE7C_ROOT / "arrays" / "stage07c_candidate_trajectories.npz"
FEATURE_DIAG = STAGE7C_ROOT / "tables" / "candidate_feature_and_label_diagnosis.csv"
RESPONSE_TABLE = STAGE7E_ROOT / "tables" / "stage07e_response_label_table.csv"

OUT_ROOT = ROOT / "07_multihypothesis" / OUTPUT_VERSION
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = ROOT / "09_reports"

RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
KEYPOINT_MODEL = "keypoint_residual_vehicle_transformer_no_subject"
TOP1_MODEL = "topk_vehicle_transformer_top1_no_subject"
BRANCH_MODELS = [f"topk_vehicle_transformer_branch{k}_no_subject" for k in range(3)]
FORBIDDEN_PREFIXES = ("label_diag__",)
IDENTIFIER_COLUMNS = {"sample_id", "event_uid", "subject", "session_stamp", "split"}
FORBIDDEN_SUBSTRINGS = (
    "oracle",
    "sample_rmse",
    "wrong_side",
    "large_response",
    "severe_amp",
    "peak_amp_abs_error",
    "peak_time_abs_error",
    "onset_delay_abs_error",
    "tail_abs_error",
    "tail_drift",
    "zero_crossing_mismatch",
    "reversal_count_exact",
    "gt_peak",
    "is_large_response",
    "is_difficult",
)
RESPONSE_FACTORS = ["direction_mode", "amplitude_mode", "peak_timing", "tail_mode", "correction_mode"]
PROTO_CANDIDATES = {
    "proto_combo_full": ["direction_mode", "amplitude_mode", "peak_timing", "tail_mode", "correction_mode"],
    "proto_direction_amp": ["direction_mode", "amplitude_mode"],
    "proto_peak_tail": ["peak_timing", "tail_mode"],
    "proto_tail_correction": ["tail_mode", "correction_mode"],
    "proto_correction": ["correction_mode"],
    "proto_amplitude": ["amplitude_mode"],
}
FALLBACK_THRESHOLDS = [0.10, 0.18, 0.25, 0.32, 0.40, 0.50]


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
    if not RESPONSE_TABLE.exists():
        raise FileNotFoundError(RESPONSE_TABLE)
    z = np.load(TRAJECTORY_NPZ, allow_pickle=True)
    features = pd.read_csv(FEATURE_DIAG)
    response = pd.read_csv(RESPONSE_TABLE)
    return dict(z), features, response


def select_allowed_features(features: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, str]] = []
    allowed: list[str] = []
    for col in features.columns:
        status = "excluded"
        reason = "not_in_allowlist"
        low = col.lower()
        if col in IDENTIFIER_COLUMNS:
            reason = "identifier_or_split"
        elif col.startswith(FORBIDDEN_PREFIXES):
            reason = "label_derived_diagnostic"
        elif any(token in low for token in FORBIDDEN_SUBSTRINGS):
            reason = "label_or_outcome_derived"
        else:
            status = "allowed"
            reason = "pre_event_context_or_candidate_prediction_only"
            allowed.append(col)
        rows.append({"feature": col, "input_status": status, "reason": reason})
    return allowed, pd.DataFrame(rows)


def make_preprocessor(features: pd.DataFrame, allowed: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    use = features[allowed]
    categorical = [c for c in use.columns if use[c].dtype == object]
    numeric = [c for c in use.columns if c not in categorical]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )
    return pre, numeric, categorical


def make_factor_model(y_train: pd.Series, pre: ColumnTransformer) -> Pipeline:
    if y_train.nunique(dropna=False) <= 1:
        clf: Any = DummyClassifier(strategy="most_frequent")
    else:
        clf = RandomForestClassifier(
            n_estimators=350,
            max_depth=4,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=20260513,
        )
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_factor_models(features: pd.DataFrame, response: pd.DataFrame, allowed: list[str], train_mask: np.ndarray) -> tuple[dict[str, Pipeline], pd.DataFrame, pd.DataFrame]:
    pre, numeric, categorical = make_preprocessor(features, allowed)
    models: dict[str, Pipeline] = {}
    pred_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for factor in RESPONSE_FACTORS:
        model = make_factor_model(response.loc[train_mask, factor].astype(str), pre)
        model.fit(features.loc[train_mask, allowed], response.loc[train_mask, factor].astype(str))
        models[factor] = model
        pred = model.predict(features[allowed])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features[allowed])
            conf = np.max(proba, axis=1)
        else:
            conf = np.ones(len(features), dtype=np.float32)
        pred_rows.append(
            pd.DataFrame(
                {
                    "sample_id": features["sample_id"].astype(str).to_numpy(),
                    "split": features["split"].astype(str).to_numpy(),
                    "factor": factor,
                    "true_label": response[factor].astype(str).to_numpy(),
                    "pred_label": pred.astype(str),
                    "confidence": conf.astype(np.float32),
                }
            )
        )
        for split_name, grp_idx in response.groupby(features["split"].astype(str)).groups.items():
            idx = np.asarray(list(grp_idx), dtype=int)
            true = response.iloc[idx][factor].astype(str).to_numpy()
            part_pred = pred[idx].astype(str)
            metric_rows.append(
                {
                    "factor": factor,
                    "split": split_name,
                    "n_samples": int(len(idx)),
                    "accuracy": float(accuracy_score(true, part_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(true, part_pred)) if len(np.unique(true)) > 1 else float("nan"),
                    "mean_confidence": float(np.mean(conf[idx])),
                    "n_classes_train": int(response.loc[train_mask, factor].astype(str).nunique()),
                    "numeric_feature_count": int(len(numeric)),
                    "categorical_feature_count": int(len(categorical)),
                }
            )
    return models, pd.concat(pred_rows, ignore_index=True), pd.DataFrame(metric_rows)


def masked_mean_trajectory(y: np.ndarray, y_mask: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if len(idx) == 0:
        raise ValueError("empty prototype index")
    vals = np.where(y_mask[idx], y[idx], np.nan)
    mean = np.nanmean(vals, axis=0)
    return np.nan_to_num(mean, nan=0.0).astype(np.float32)


def build_prototype_bank(y: np.ndarray, y_mask: np.ndarray, response: pd.DataFrame, train_idx: np.ndarray) -> dict[str, np.ndarray]:
    bank: dict[str, np.ndarray] = {}
    bank["__global__"] = masked_mean_trajectory(y, y_mask, train_idx)
    train_resp = response.iloc[train_idx].copy()
    for factor in RESPONSE_FACTORS:
        for value in sorted(train_resp[factor].astype(str).unique()):
            idx = train_resp.index[train_resp[factor].astype(str).eq(value)].to_numpy(dtype=int)
            if len(idx):
                bank[f"{factor}={value}"] = masked_mean_trajectory(y, y_mask, idx)
    return bank


def prototype_for_conditions(
    y: np.ndarray,
    y_mask: np.ndarray,
    response: pd.DataFrame,
    train_idx: np.ndarray,
    conditions: dict[str, str],
    min_count: int = 5,
) -> tuple[np.ndarray, str, int]:
    ordered = list(conditions.items())
    fallback_sets: list[list[tuple[str, str]]] = [ordered]
    if len(ordered) > 3:
        fallback_sets.extend([ordered[:3], ordered[:2], ordered[:1]])
    elif len(ordered) > 1:
        fallback_sets.extend([ordered[:1]])
    fallback_sets.append([])
    train_resp = response.iloc[train_idx]
    for cond in fallback_sets:
        mask = np.ones(len(train_resp), dtype=bool)
        for col, value in cond:
            mask &= train_resp[col].astype(str).to_numpy() == str(value)
        idx = train_resp.index[mask].to_numpy(dtype=int)
        if len(idx) >= min_count or not cond:
            if len(idx) == 0:
                idx = train_idx
            rule = "&".join([f"{k}={v}" for k, v in cond]) if cond else "__global__"
            return masked_mean_trajectory(y, y_mask, idx), rule, int(len(idx))
    return masked_mean_trajectory(y, y_mask, train_idx), "__global__", int(len(train_idx))


def generate_proto_predictions(
    y: np.ndarray,
    y_mask: np.ndarray,
    response: pd.DataFrame,
    train_idx: np.ndarray,
    factor_pred_wide: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    n, t_len = y.shape
    predictions: dict[str, np.ndarray] = {}
    trace_rows: list[dict[str, Any]] = []
    for candidate_name, factor_cols in PROTO_CANDIDATES.items():
        arr = np.zeros((n, t_len), dtype=np.float32)
        for i in range(n):
            cond = {factor: str(factor_pred_wide.at[i, f"{factor}__pred"]) for factor in factor_cols}
            proto, rule, count = prototype_for_conditions(y, y_mask, response, train_idx, cond)
            arr[i] = proto
            trace_rows.append(
                {
                    "sample_id": response.at[i, "sample_id"],
                    "split": response.at[i, "split"],
                    "candidate_name": candidate_name,
                    "prototype_rule": rule,
                    "prototype_train_n": count,
                }
            )
        predictions[candidate_name] = arr
    return predictions, pd.DataFrame(trace_rows)


def sample_rmse_array(y_true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(y_true) & np.isfinite(pred)
    diff = np.where(valid, pred - y_true, np.nan)
    denom = np.maximum(valid.sum(axis=1), 1)
    return np.sqrt(np.nansum(diff * diff, axis=1) / denom).astype(np.float32)


def oracle_prediction(y: np.ndarray, y_mask: np.ndarray, predictions: dict[str, np.ndarray], candidate_names: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    rmses = np.stack([sample_rmse_array(y, predictions[name], y_mask) for name in candidate_names], axis=1)
    best_idx = np.nanargmin(rmses, axis=1)
    stack = np.stack([predictions[name] for name in candidate_names], axis=1)
    best = stack[np.arange(len(best_idx)), best_idx]
    diag = pd.DataFrame(
        {
            "oracle_best_index": best_idx,
            "oracle_best_model": [candidate_names[int(i)] for i in best_idx],
            "oracle_sample_rmse": rmses[np.arange(len(best_idx)), best_idx],
        }
    )
    for j, name in enumerate(candidate_names):
        diag[f"{name}__sample_rmse"] = rmses[:, j]
    return best.astype(np.float32), diag


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
    rows: list[pd.DataFrame] = []
    for split_name in ["train", "val", "test"]:
        mask = meta[SPLIT_STRATEGY].astype(str).to_numpy() == split_name
        if not mask.any():
            continue
        split_meta = meta.loc[mask].reset_index(drop=True)
        for model_name, pred in predictions.items():
            sample_rows = eval_utils.sample_metric_rows(
                y[mask],
                pred[mask],
                y_mask[mask],
                label_time,
                split_meta,
                model_name=model_name,
                split_strategy=SPLIT_STRATEGY,
                split_name=split_name,
                window_id="pre3_label3_response_coverage",
                large_thr=large_thr,
                difficult_thr=difficult_thr,
            )
            if sample_rows:
                part = pd.DataFrame(sample_rows)
                part["track_id"] = TRACK_ID
                rows.append(part)
    per_sample = pd.concat(rows, ignore_index=True)
    metrics = eval_utils.aggregate_metrics(per_sample)
    metrics["track_id"] = TRACK_ID
    return metrics, per_sample


def build_factor_pred_wide(factor_pred_long: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for factor, part in factor_pred_long.groupby("factor"):
        pieces.append(
            part[["sample_id", "pred_label", "confidence"]].rename(
                columns={"pred_label": f"{factor}__pred", "confidence": f"{factor}__confidence"}
            )
        )
    wide = pieces[0]
    for part in pieces[1:]:
        wide = wide.merge(part, on="sample_id", how="inner")
    return wide.reset_index(drop=True)


def policy_predictions(
    base_predictions: dict[str, np.ndarray],
    factor_pred_wide: pd.DataFrame,
    metrics: pd.DataFrame,
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    proto_names = list(PROTO_CANDIDATES.keys())
    conf_cols = [f"{factor}__confidence" for factor in RESPONSE_FACTORS]
    combo_conf = factor_pred_wide[conf_cols].astype(float).prod(axis=1).to_numpy()
    policies: dict[str, np.ndarray] = {}
    for proto in proto_names:
        policies[f"always_{proto}"] = base_predictions[proto]
        for thr in FALLBACK_THRESHOLDS:
            name = f"{proto}__fallback_rbf_conf_prod_lt_{thr:.2f}"
            use_proto = combo_conf >= thr
            policies[name] = np.where(use_proto[:, None], base_predictions[proto], base_predictions[RBF_MODEL]).astype(np.float32)
    policy_metrics, policy_per_sample = evaluate_predictions(y, y_mask, label_time, meta, train_idx, {**{RBF_MODEL: base_predictions[RBF_MODEL]}, **policies})
    return policies, policy_metrics, policy_per_sample


def add_reference_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    cleanup_cols = [
        "rbf_ref_rmse",
        "rbf_ref_wrong_side",
        "rbf_ref_large_recall",
        "rbf_ref_difficult_rmse",
        "rmse_delta_vs_rbf",
        "wrong_side_delta_vs_rbf",
        "large_recall_delta_vs_rbf",
        "difficult_rmse_delta_vs_rbf",
    ]
    out = metrics.drop(columns=cleanup_cols, errors="ignore").copy()
    refs = out[out["model_name"] == RBF_MODEL][["split", "rmse_steer", "wrong_side_rate", "large_response_recall", "difficult_top20_rmse"]].rename(
        columns={
            "rmse_steer": "rbf_ref_rmse",
            "wrong_side_rate": "rbf_ref_wrong_side",
            "large_response_recall": "rbf_ref_large_recall",
            "difficult_top20_rmse": "rbf_ref_difficult_rmse",
        }
    )
    out = out.merge(refs, on="split", how="left")
    out["rmse_delta_vs_rbf"] = out["rmse_steer"] - out["rbf_ref_rmse"]
    out["wrong_side_delta_vs_rbf"] = out["wrong_side_rate"] - out["rbf_ref_wrong_side"]
    out["large_recall_delta_vs_rbf"] = out["large_response_recall"] - out["rbf_ref_large_recall"]
    out["difficult_rmse_delta_vs_rbf"] = out["difficult_top20_rmse"] - out["rbf_ref_difficult_rmse"]
    return out


def select_policy(policy_metrics: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    val = policy_metrics[policy_metrics["split"] == "val"].copy()
    rbf = val[val["model_name"].eq(RBF_MODEL)].iloc[0]
    candidates = val[
        (~val["model_name"].eq(RBF_MODEL))
        & (~val["model_name"].astype(str).str.contains("oracle", case=False, na=False))
    ].copy()
    candidates["meets_rmse_improvement"] = candidates["rmse_steer"] < float(rbf["rmse_steer"]) - 1e-6
    candidates["meets_noninferior_physical"] = (
        (candidates["rmse_steer"] <= float(rbf["rmse_steer"]) + 0.002)
        & (
            (candidates["wrong_side_rate"] < float(rbf["wrong_side_rate"]))
            | (candidates["large_response_recall"] > float(rbf["large_response_recall"]))
            | (candidates["difficult_top20_rmse"] < float(rbf["difficult_top20_rmse"]))
        )
    )
    if candidates["meets_rmse_improvement"].any():
        selected = str(candidates[candidates["meets_rmse_improvement"]].sort_values(["rmse_steer", "wrong_side_rate"]).iloc[0]["model_name"])
        reason = "val_rmse_improves_rbf"
    elif candidates["meets_noninferior_physical"].any():
        selected = str(candidates[candidates["meets_noninferior_physical"]].sort_values(["rmse_steer", "wrong_side_rate"]).iloc[0]["model_name"])
        reason = "val_noninferior_with_physical_gain"
    else:
        selected = RBF_MODEL
        reason = "no_policy_passed_val_gate"
    table = candidates.sort_values(["rmse_steer", "wrong_side_rate"]).copy()
    table["selected_by_val_gate"] = table["model_name"].eq(selected).astype(int)
    table["selection_reason"] = reason
    return selected, table


def sample_indices(meta: pd.DataFrame, sample_ids: list[str]) -> list[int]:
    lookup = {sid: i for i, sid in enumerate(meta["sample_id"].astype(str).tolist())}
    return [lookup[sid] for sid in sample_ids if sid in lookup]


def plot_prediction_grid(path: Path, sample_ids: list[str], y: np.ndarray, y_mask: np.ndarray, label_time: np.ndarray, meta: pd.DataFrame, predictions: dict[str, np.ndarray], title: str) -> None:
    ids = sample_indices(meta, sample_ids)[:12]
    if not ids:
        return
    ncols = 3
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 3.2 * nrows), squeeze=False)
    plot_models = [
        (RBF_MODEL, "#1f77b4", "RBF/KNN", "-"),
        ("proto_combo_full", "#d62728", "combo", "--"),
        ("proto_direction_amp", "#ff7f0e", "dir_amp", "--"),
        ("proto_tail_correction", "#2ca02c", "tail_corr", "--"),
        ("response_factorized_oracle", "#111111", "oracle*", "-."),
    ]
    for ax, idx in zip(axes.ravel(), ids):
        valid = y_mask[idx] & np.isfinite(y[idx])
        ax.plot(label_time[valid], y[idx, valid], color="#000000", linewidth=1.8, label="GT")
        for model_name, color, label, style in plot_models:
            if model_name not in predictions:
                continue
            pred = predictions[model_name][idx]
            valid_pred = valid & np.isfinite(pred)
            ax.plot(label_time[valid_pred], pred[valid_pred], color=color, linestyle=style, linewidth=1.05, alpha=0.9, label=label)
        sid = str(meta.at[idx, "sample_id"])
        short = sid.split("__")[-2] if "__" in sid else sid[-12:]
        ax.set_title(short, fontsize=8)
        ax.grid(True, alpha=0.22)
        ax.axhline(0.0, color="#dddddd", linewidth=0.8)
    for ax in axes.ravel()[len(ids) :]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=8)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metric_summary(metrics: pd.DataFrame, selected: str, path: Path) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    keep = [RBF_MODEL, selected, "response_factorized_oracle", "response_factorized_plus_existing_oracle"]
    keep_unique = []
    for name in keep:
        if name not in keep_unique:
            keep_unique.append(name)
    test = test[test["model_name"].isin(keep_unique)].copy()
    test["order"] = test["model_name"].map({name: i for i, name in enumerate(keep_unique)})
    test = test.sort_values("order")
    labels = [x.replace("response_factorized_", "rf_").replace("_", " ") for x in test["model_name"]]
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2))
    for ax, col, title in [
        (axes[0], "rmse_steer", "RMSE"),
        (axes[1], "wrong_side_rate", "Wrong-side"),
        (axes[2], "large_response_recall", "Large recall"),
        (axes[3], "difficult_top20_rmse", "Difficult RMSE"),
    ]:
        ax.bar(np.arange(len(test)), test[col].astype(float), color="#4c78a8")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(test)), labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Stage 7f response-factorized candidates on test", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_reports(
    all_metrics: pd.DataFrame,
    selected: str,
    selection_table: pd.DataFrame,
    factor_metrics: pd.DataFrame,
    gate: pd.DataFrame,
    figures: dict[str, str],
) -> None:
    test = all_metrics[all_metrics["split"] == "test"].set_index("model_name")
    val = all_metrics[all_metrics["split"] == "val"].set_index("model_name")

    def safe(frame: pd.DataFrame, model: str, col: str) -> float:
        if model not in frame.index or col not in frame.columns:
            return float("nan")
        return float(frame.loc[model, col])

    selected_test = safe(test, selected, "rmse_steer")
    rbf_test = safe(test, RBF_MODEL, "rmse_steer")
    oracle_test = safe(test, "response_factorized_oracle", "rmse_steer")
    combo_oracle_test = safe(test, "response_factorized_plus_existing_oracle", "rmse_steer")
    gate_status = str(gate.set_index("gate_item").loc["deployable_upgrade", "status"])
    selection_text = selection_table[["model_name", "rmse_steer", "rmse_delta_vs_rbf", "wrong_side_rate", "large_response_recall", "selected_by_val_gate"]].head(12).to_string(index=False)
    factor_text = factor_metrics[factor_metrics["split"].isin(["val", "test"])][["factor", "split", "accuracy", "balanced_accuracy", "mean_confidence"]].to_string(index=False)
    user = f"""# Stage 7f 用户查看版：response-factorized 车辆-only 原型候选 v0.1

## 这个阶段为什么做

Stage 7e 判断不能继续只调 selector，而要让候选生成本身覆盖方向、幅值、峰值时间、尾段和反向/多段修正。这个阶段先做一个轻量版本：用 train split 的真实响应类型建立原型轨迹，再用事件前特征预测响应类型，生成车辆-only 候选。

## 这个阶段检查了什么

- 输入仍然只用事件前车辆/道路/事件上下文和已有候选预测形态特征。
- 禁止使用 subject ID、session ID、test 标签、生理、脑电、连续风格。
- 原型轨迹只从 train split 估计。
- val 选择策略，test 只报告。

## 目前发现了什么

- val 选择策略：`{selected}`。
- test 上该策略 RMSE={selected_test:.6f}，RBF/KNN RMSE={rbf_test:.6f}，delta={selected_test - rbf_test:+.6f}。
- response-factorized oracle RMSE={oracle_test:.6f}。
- response-factorized + existing candidates oracle RMSE={combo_oracle_test:.6f}。
- gate={gate_status}。

## 响应类型预测质量

```text
{factor_text}
```

## val 策略选择表

```text
{selection_text}
```

## 哪些结果可信

可信的是：这一版严格 train-only 建原型、val 选策略、test 报告；没有用生理/脑电/风格，也没有读取服务器凭据。它能判断“响应类型原型候选”这一方向是否值得继续。

## 哪些结果还不能下结论

不能把 oracle 当作可部署性能；如果 selected 策略没有超过 RBF/KNN，就不能说多假设已经解决。即便 oracle 好，也只能说明下一版需要更强的非 oracle 选择和候选生成。

## 下一阶段是否可以继续

如果 gate 仍是 no_upgrade，下一步不要进入生理/EEG；应把 response-factorized 原型升级成可训练关键点/分段候选模型，重点提升响应类型预测与候选选择。

## 推荐优先查看

1. `{figures["metric_summary"]}`
2. `{figures["fixed_predictions"]}`
3. `{figures["oracle_gain_predictions"]}`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_gate_table.csv`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_factor_prediction_metrics.csv`
"""
    (REPORT_DIR / "stage07f_response_factorized_candidates_user_summary_cn.md").write_text(user, encoding="utf-8")

    tech = f"""# Stage 7f 技术报告：response-factorized vehicle-only prototype candidates v0.1

## Scope

- Track: `{TRACK_ID}`
- Input trajectories: `{path_str(TRAJECTORY_NPZ)}`
- Response labels: `{path_str(RESPONSE_TABLE)}`
- No server used. Credential file not read.
- Excluded: subject ID, session ID, physio, EEG, continuous style, test labels as inputs.

## Selected Policy

- selected_policy=`{selected}`
- gate=`{gate_status}`
- test_delta_vs_rbf={selected_test - rbf_test:+.6f}

## Test Summary

- RBF/KNN RMSE={rbf_test:.6f}
- selected RMSE={selected_test:.6f}
- response-factorized oracle RMSE={oracle_test:.6f}
- response-factorized + existing candidates oracle RMSE={combo_oracle_test:.6f}

## Factor Prediction Metrics

```text
{factor_text}
```

## Gate

```text
{gate.to_string(index=False)}
```
"""
    (REPORT_DIR / "stage07f_response_factorized_candidates_v0_1_cn.md").write_text(tech, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    z, feature_diag, response = load_inputs()
    names = [str(x) for x in z["candidate_model_names"].tolist()]
    y = z["y_true"].astype(np.float32)
    y_mask = z["y_mask"].astype(bool)
    label_time = z["label_time_rel_s"].astype(np.float32)
    sample_ids = z["sample_ids"].astype(str)
    split = z["split"].astype(str)
    response = response.set_index("sample_id").loc[sample_ids].reset_index()
    feature_diag = feature_diag.set_index("sample_id").loc[sample_ids].reset_index()
    feature_diag["split"] = split
    response["split"] = split
    meta_cols = ["sample_id", "event_uid", "subject", "session_stamp"]
    response = response.merge(feature_diag[meta_cols], on="sample_id", how="left", validate="one_to_one")
    train_idx = np.where(split == "train")[0]
    train_mask = split == "train"

    allowed, feature_audit = select_allowed_features(feature_diag)
    factor_models, factor_pred_long, factor_metrics = fit_factor_models(feature_diag, response, allowed, train_mask)
    factor_pred_wide = build_factor_pred_wide(factor_pred_long)

    candidate_idx = {name: names.index(name) for name in names}
    base_predictions: dict[str, np.ndarray] = {
        RBF_MODEL: z["candidate_predictions"][:, candidate_idx[RBF_MODEL], :].astype(np.float32),
        KEYPOINT_MODEL: z["candidate_predictions"][:, candidate_idx[KEYPOINT_MODEL], :].astype(np.float32),
    }
    proto_predictions, proto_trace = generate_proto_predictions(y, y_mask, response, train_idx, factor_pred_wide)
    base_predictions.update(proto_predictions)
    rf_oracle, rf_oracle_diag = oracle_prediction(y, y_mask, base_predictions, [RBF_MODEL, *PROTO_CANDIDATES.keys()])
    base_predictions["response_factorized_oracle"] = rf_oracle

    existing_plus = dict(base_predictions)
    for name in [KEYPOINT_MODEL, TOP1_MODEL, *BRANCH_MODELS]:
        if name in candidate_idx:
            existing_plus[name] = z["candidate_predictions"][:, candidate_idx[name], :].astype(np.float32)
    combo_oracle, combo_oracle_diag = oracle_prediction(y, y_mask, existing_plus, [RBF_MODEL, KEYPOINT_MODEL, *BRANCH_MODELS, *PROTO_CANDIDATES.keys()])
    base_predictions["response_factorized_plus_existing_oracle"] = combo_oracle

    proto_metrics, proto_per_sample = evaluate_predictions(y, y_mask, label_time, response.rename(columns={"split": SPLIT_STRATEGY}), train_idx, base_predictions)
    policies, policy_metrics, policy_per_sample = policy_predictions(base_predictions, factor_pred_wide, proto_metrics, y, y_mask, label_time, response.rename(columns={"split": SPLIT_STRATEGY}), train_idx)
    all_metrics = pd.concat([proto_metrics, policy_metrics[~policy_metrics["model_name"].isin([RBF_MODEL])]], ignore_index=True, sort=False)
    all_metrics = add_reference_deltas(all_metrics)
    selected, selection_table = select_policy(all_metrics)
    selected_predictions = {selected: policies[selected]} if selected in policies else {selected: base_predictions[selected]}
    selected_metrics, selected_per_sample = evaluate_predictions(y, y_mask, label_time, response.rename(columns={"split": SPLIT_STRATEGY}), train_idx, selected_predictions)
    selected_metrics = add_reference_deltas(pd.concat([all_metrics[all_metrics["model_name"] == RBF_MODEL], selected_metrics], ignore_index=True, sort=False))

    test_selected = all_metrics[(all_metrics["split"] == "test") & (all_metrics["model_name"] == selected)]
    if test_selected.empty:
        test_selected = selected_metrics[(selected_metrics["split"] == "test") & (selected_metrics["model_name"] == selected)]
    test_rbf = all_metrics[(all_metrics["split"] == "test") & (all_metrics["model_name"] == RBF_MODEL)].iloc[0]
    selected_test_rmse = float(test_selected.iloc[0]["rmse_steer"])
    rbf_test_rmse = float(test_rbf["rmse_steer"])
    gate_status = "upgrade" if selected != RBF_MODEL and selected_test_rmse < rbf_test_rmse - 1e-6 else "no_upgrade"
    gate = pd.DataFrame(
        [
            {"gate_item": "selected_policy", "status": selected, "evidence": "selected by validation gate only"},
            {"gate_item": "deployable_upgrade", "status": gate_status, "evidence": f"test delta vs RBF {selected_test_rmse - rbf_test_rmse:+.6f}"},
            {"gate_item": "response_factorized_oracle", "status": "diagnostic_only", "evidence": "oracle uses true labels and is not deployable"},
            {"gate_item": "stage08_physio_eeg_allowed", "status": "blocked", "evidence": "vehicle-only response-factorized candidate route is not yet stable"},
            {"gate_item": "server_used", "status": "no", "evidence": "local run only; credential file not read"},
        ]
    )

    test_ids = response.loc[split == "test", "sample_id"].astype(str).head(12).tolist()
    oracle_gain = rf_oracle_diag.copy()
    oracle_gain["sample_id"] = sample_ids
    oracle_gain["split"] = split
    oracle_gain["gain_over_rbf"] = oracle_gain[f"{RBF_MODEL}__sample_rmse"] - oracle_gain["oracle_sample_rmse"]
    oracle_ids = oracle_gain[oracle_gain["split"] == "test"].sort_values("gain_over_rbf", ascending=False)["sample_id"].astype(str).head(12).tolist()
    figure_predictions = {**base_predictions}
    figures = {
        "metric_summary": path_str(FIG_DIR / "stage07f_metric_summary_test.png"),
        "fixed_predictions": path_str(FIG_DIR / "stage07f_fixed_predictions_test.png"),
        "oracle_gain_predictions": path_str(FIG_DIR / "stage07f_oracle_gain_predictions_test.png"),
    }
    plot_metric_summary(all_metrics, selected, Path(figures["metric_summary"]))
    plot_prediction_grid(Path(figures["fixed_predictions"]), test_ids, y, y_mask, label_time, response.rename(columns={"split": SPLIT_STRATEGY}), figure_predictions, "Stage 7f fixed test response-factorized candidates")
    plot_prediction_grid(Path(figures["oracle_gain_predictions"]), oracle_ids, y, y_mask, label_time, response.rename(columns={"split": SPLIT_STRATEGY}), figure_predictions, "Stage 7f largest response-factorized oracle gains")

    feature_audit.to_csv(TABLE_DIR / "stage07f_feature_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": allowed}).to_csv(TABLE_DIR / "stage07f_allowed_features.csv", index=False, encoding="utf-8-sig")
    factor_pred_long.to_csv(TABLE_DIR / "stage07f_factor_predictions_long.csv", index=False, encoding="utf-8-sig")
    factor_metrics.to_csv(TABLE_DIR / "stage07f_factor_prediction_metrics.csv", index=False, encoding="utf-8-sig")
    proto_trace.to_csv(TABLE_DIR / "stage07f_prototype_trace.csv", index=False, encoding="utf-8-sig")
    rf_oracle_diag.assign(sample_id=sample_ids, split=split).to_csv(TABLE_DIR / "stage07f_response_factorized_oracle_diag.csv", index=False, encoding="utf-8-sig")
    combo_oracle_diag.assign(sample_id=sample_ids, split=split).to_csv(TABLE_DIR / "stage07f_combo_oracle_diag.csv", index=False, encoding="utf-8-sig")
    all_metrics.to_csv(TABLE_DIR / "stage07f_policy_and_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    proto_per_sample.to_csv(TABLE_DIR / "stage07f_candidate_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    policy_metrics.to_csv(TABLE_DIR / "stage07f_policy_metrics.csv", index=False, encoding="utf-8-sig")
    policy_per_sample.to_csv(TABLE_DIR / "stage07f_policy_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    selection_table.to_csv(TABLE_DIR / "stage07f_validation_selection_table.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "stage07f_gate_table.csv", index=False, encoding="utf-8-sig")

    write_reports(all_metrics, selected, selection_table, factor_metrics, gate, figures)
    oracle_test_rmse = float(all_metrics[(all_metrics["split"] == "test") & (all_metrics["model_name"] == "response_factorized_oracle")]["rmse_steer"].iloc[0])
    combo_test_rmse = float(all_metrics[(all_metrics["split"] == "test") & (all_metrics["model_name"] == "response_factorized_plus_existing_oracle")]["rmse_steer"].iloc[0])
    summary = {
        "output_version": OUTPUT_VERSION,
        "track_id": TRACK_ID,
        "selected_policy": selected,
        "gate_status": gate_status,
        "rbf_test_rmse": rbf_test_rmse,
        "selected_test_rmse": selected_test_rmse,
        "selected_test_delta_vs_rbf": selected_test_rmse - rbf_test_rmse,
        "response_factorized_oracle_test_rmse": oracle_test_rmse,
        "response_factorized_oracle_delta_vs_rbf": oracle_test_rmse - rbf_test_rmse,
        "combo_oracle_test_rmse": combo_test_rmse,
        "combo_oracle_delta_vs_rbf": combo_test_rmse - rbf_test_rmse,
        "allowed_feature_count": int(len(allowed)),
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
    (LOG_DIR / "stage07f_response_factorized_candidates_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
