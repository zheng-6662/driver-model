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


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
FORMAL_SAMPLE_DIR = ROOT / "02_samples" / "vehicle_instability_highconf_v0_1"
SAMPLES_PATH = FORMAL_SAMPLE_DIR / "tables" / "samples_master.csv"
PROCESSED_DIR = ROOT / "03_processed_datasets" / "vehicle_instability_allraw_highconf_v0_1"
ARRAY_DIR = PROCESSED_DIR / "arrays"
FORMAL_BASELINE_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_formal_baselines_v0_1"
OLD_DIRECT_DIR = ROOT / "03_baselines" / "oldcode_vehicle_direct_full_clean_on_instability_v0_1"
OUT_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_strong_vehicle_baselines_v0_1"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402
import stage03_vehicle_instability_formal_baselines_v0_1 as formal_v01  # noqa: E402


WINDOW_ID = "pre2_label2_old_main"
SPLIT_STRATEGY = "session_level_split"
RANDOM_SEED = 20260512

CONTEXT_COLS = [
    "event_type",
    "event_level",
    "road_type_anchor",
    "old_v400_road_type_mode",
    "old_v400_phase_mode",
    "road_design_risk_class",
    "road_design_mapping_reliability",
]

NUMERIC_CONTEXT_COLS = [
    "anchor_time_rel_s",
    "curvature_anchor",
    "input_valid_ratio",
]

RICH_RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
KNN_K_VALUES = [3, 5, 9, 15, 25, 45, 65]
RBF_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
RBF_GAMMA_SCALE_GRID = [0.25, 0.5, 1.0, 2.0]

DISPLAY_NAMES = {
    "formal_ridge_vehicle_context_no_subject": "formal ridge",
    "ridge_rich_history_no_subject": "rich ridge hist",
    "ridge_rich_context_no_subject": "rich ridge ctx",
    "rbf_kernel_ridge_context_no_subject": "RBF KRR",
    "knn_template_context_no_subject": "KNN template",
    "direction_gated_knn_template_no_subject": "dir-gated KNN",
    "peak_scaled_template_context_no_subject": "peak-scaled template",
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def split_indices(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = meta[SPLIT_STRATEGY].astype(str).to_numpy()
    return (
        np.where(split == "train")[0],
        np.where(split == "val")[0],
        np.where(split == "test")[0],
    )


def safe_nan_stats(values: np.ndarray, axis: int = 1) -> dict[str, np.ndarray]:
    all_nan = np.all(~np.isfinite(values), axis=axis)
    filled = np.where(np.isfinite(values), values, np.nan)
    stats = {
        "mean": np.nanmean(filled, axis=axis),
        "std": np.nanstd(filled, axis=axis),
        "min": np.nanmin(filled, axis=axis),
        "max": np.nanmax(filled, axis=axis),
        "q10": np.nanpercentile(filled, 10, axis=axis),
        "q90": np.nanpercentile(filled, 90, axis=axis),
        "abs_mean": np.nanmean(np.abs(filled), axis=axis),
        "abs_max": np.nanmax(np.abs(filled), axis=axis),
    }
    for key, vals in stats.items():
        vals = np.asarray(vals, dtype=np.float64)
        vals[all_nan] = np.nan
        stats[key] = vals
    return stats


def slope_feature(arr: np.ndarray, time_axis: np.ndarray) -> np.ndarray:
    if arr.shape[1] < 2:
        return np.zeros(arr.shape[0], dtype=np.float64)
    t = time_axis.astype(np.float64)
    tc = t - t.mean()
    denom = float(np.sum(tc * tc)) or 1.0
    centered = arr - np.nanmean(arr, axis=1, keepdims=True)
    return np.nansum(centered * tc[None, :], axis=1) / denom


def build_rich_vehicle_features(
    input_values: np.ndarray,
    input_time: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    include_context: bool,
) -> tuple[np.ndarray, list[str]]:
    x = input_values.astype(np.float64)
    x = np.where(np.isfinite(x), x, np.nan)
    feature_names = [str(v) for v in np.load(ARRAY_DIR / f"{WINDOW_ID}.npz", allow_pickle=True)["feature_names"].tolist()]

    features: list[np.ndarray] = []
    names: list[str] = []
    windows = [
        ("full2s", -2.0, 0.0),
        ("early_2to1s", -2.0, -1.0),
        ("mid_1to05s", -1.0, -0.5),
        ("last1s", -1.0, 0.0),
        ("last500ms", -0.5, 0.0),
        ("last250ms", -0.25, 0.0),
    ]

    for j in range(x.shape[2]):
        signal_name = feature_names[j] if j < len(feature_names) else f"signal{j}"
        arr_all = x[:, :, j]
        for window_name, start_s, end_s in windows:
            mask = (input_time >= start_s - 1e-9) & (input_time <= end_s + 1e-9)
            if mask.sum() == 0:
                continue
            arr = arr_all[:, mask]
            t = input_time[mask]
            stats = safe_nan_stats(arr)
            for stat_name, vals in stats.items():
                features.append(vals)
                names.append(f"{signal_name}:{window_name}:{stat_name}")
            first = arr[:, 0]
            last = arr[:, -1]
            features.append(last)
            names.append(f"{signal_name}:{window_name}:last")
            features.append(last - first)
            names.append(f"{signal_name}:{window_name}:delta")
            features.append(last - stats["mean"])
            names.append(f"{signal_name}:{window_name}:last_minus_mean")
            features.append(stats["max"] - stats["min"])
            names.append(f"{signal_name}:{window_name}:range")
            features.append(slope_feature(arr, t))
            names.append(f"{signal_name}:{window_name}:slope")

        raw_idx = np.unique(np.linspace(0, len(input_time) - 1, 41).round().astype(int))
        for k in raw_idx:
            features.append(arr_all[:, k])
            names.append(f"{signal_name}:raw_t{float(input_time[k]):.3f}s")

    if include_context:
        for col in NUMERIC_CONTEXT_COLS:
            if col in meta.columns:
                features.append(pd.to_numeric(meta[col], errors="coerce").to_numpy(dtype=np.float64))
                names.append(col)
        for col in CONTEXT_COLS:
            if col not in meta.columns:
                continue
            values = meta[col].astype(str).fillna("NA")
            train_values = sorted(values.iloc[train_idx].unique().tolist())
            for val in train_values:
                features.append((values == val).to_numpy(dtype=np.float64))
                names.append(f"{col}={val}")

    X = np.vstack(features).T if features else np.zeros((len(meta), 0), dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


class TrainStandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "TrainStandardScaler":
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.scale_ = np.std(X, axis=0, keepdims=True)
        self.scale_[self.scale_ < 1e-6] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("TrainStandardScaler must be fitted before transform")
        return (X - self.mean_) / self.scale_


def standardize_train_only(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, TrainStandardScaler]:
    scaler = TrainStandardScaler().fit(X[train_idx])
    return scaler.transform(X), scaler


class TrainVarianceSelector:
    def __init__(self, n_components: int) -> None:
        self.n_components = int(n_components)
        self.indices_: np.ndarray | None = None
        self.n_components_: int = 0
        self.explained_variance_ratio_: np.ndarray = np.zeros(0, dtype=np.float64)

    def fit(self, X: np.ndarray) -> "TrainVarianceSelector":
        variances = np.nanvar(X, axis=0)
        variances = np.nan_to_num(variances, nan=0.0, posinf=0.0, neginf=0.0)
        n = min(self.n_components, X.shape[1])
        self.indices_ = np.argsort(-variances)[:n]
        self.n_components_ = int(len(self.indices_))
        total = float(np.sum(variances)) or 1.0
        self.explained_variance_ratio_ = variances[self.indices_] / total
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.indices_ is None:
            raise RuntimeError("TrainVarianceSelector must be fitted before transform")
        return X[:, self.indices_]


def make_distance_features(
    X_scaled: np.ndarray,
    train_idx: np.ndarray,
    n_components: int = 96,
) -> tuple[np.ndarray, TrainVarianceSelector]:
    selector = TrainVarianceSelector(n_components=n_components)
    selector.fit(X_scaled[train_idx])
    return selector.transform(X_scaled), selector


def peak_arrays(y: np.ndarray, y_mask: np.ndarray, label_time: np.ndarray) -> dict[str, np.ndarray]:
    gt = np.where(y_mask, y, np.nan)
    abs_gt = np.abs(gt)
    idx = np.nanargmax(np.nan_to_num(abs_gt, nan=-1.0), axis=1)
    rows = np.arange(y.shape[0])
    signed = gt[rows, idx]
    peak_abs = np.abs(signed)
    direction = np.where(signed >= 0.0, 1, -1).astype(int)
    return {
        "peak_idx": idx.astype(int),
        "peak_abs": peak_abs.astype(np.float64),
        "peak_signed": signed.astype(np.float64),
        "direction": direction,
        "peak_time": label_time[idx].astype(np.float64),
    }


def fit_direct_ridge(
    model_name: str,
    X_scaled: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    y_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    y_fit = np.nan_to_num(np.where(y_mask, y, np.nan), nan=0.0).astype(np.float64)
    Xd = np.c_[np.ones((X_scaled.shape[0], 1), dtype=np.float64), X_scaled.astype(np.float64)]
    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None
    for alpha in RICH_RIDGE_ALPHAS:
        Xt = Xd[train_idx]
        reg = np.eye(Xt.shape[1], dtype=np.float64) * float(alpha)
        reg[0, 0] = 0.0
        coef = np.linalg.solve(Xt.T @ Xt + reg, Xt.T @ y_fit[train_idx])
        pred = (Xd @ coef).astype(np.float32)
        val_rmse = eval_utils.rmse(y[val_idx], pred[val_idx], y_mask[val_idx])
        row = {
            "model_name": model_name,
            "status": "ok",
            "selected_alpha": float(alpha),
            "val_rmse": float(val_rmse),
            "train_rmse": float(eval_utils.rmse(y[train_idx], pred[train_idx], y_mask[train_idx])),
            "selection_metric": "val_rmse",
            "uses_subject_id": False,
            "uses_physio": False,
            "uses_eeg": False,
            "uses_continuous_style": False,
            "standardization_scope": "train split only",
        }
        if best is None or val_rmse < best["val_rmse"]:
            best = row
            best_pred = pred
    assert best is not None and best_pred is not None
    return best_pred, best


def squared_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = np.sum(A * A, axis=1, keepdims=True)
    bb = np.sum(B * B, axis=1, keepdims=True).T
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def median_gamma(X_train: np.ndarray) -> float:
    d2 = squared_distance(X_train, X_train)
    vals = d2[np.triu_indices_from(d2, k=1)]
    vals = vals[np.isfinite(vals) & (vals > 1e-9)]
    med = float(np.median(vals)) if vals.size else 1.0
    return 1.0 / max(med, 1e-6)


def fit_rbf_kernel_ridge_direct(
    X_dist: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    y_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    model_name = "rbf_kernel_ridge_context_no_subject"
    y_fit = np.nan_to_num(np.where(y_mask, y, np.nan), nan=0.0).astype(np.float64)
    X_train = X_dist[train_idx].astype(np.float64)
    d2_train = squared_distance(X_train, X_train)
    d2_all = squared_distance(X_dist.astype(np.float64), X_train)
    base_gamma = median_gamma(X_train)
    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None
    eye = np.eye(len(train_idx), dtype=np.float64)
    for gamma_scale in RBF_GAMMA_SCALE_GRID:
        gamma = base_gamma * float(gamma_scale)
        K_train = np.exp(-gamma * d2_train)
        K_all = np.exp(-gamma * d2_all)
        for alpha in RBF_ALPHA_GRID:
            coef = np.linalg.solve(K_train + float(alpha) * eye, y_fit[train_idx])
            pred = (K_all @ coef).astype(np.float32)
            val_rmse = eval_utils.rmse(y[val_idx], pred[val_idx], y_mask[val_idx])
            row = {
                "model_name": model_name,
                "status": "ok",
                "selected_gamma": float(gamma),
                "selected_gamma_scale": float(gamma_scale),
                "selected_alpha": float(alpha),
                "val_rmse": float(val_rmse),
                "train_rmse": float(eval_utils.rmse(y[train_idx], pred[train_idx], y_mask[train_idx])),
                "selection_metric": "val_rmse",
                "uses_subject_id": False,
                "uses_physio": False,
                "uses_eeg": False,
                "uses_continuous_style": False,
                "standardization_scope": "train split scaler + train-variance distance feature selection only",
            }
            if best is None or val_rmse < best["val_rmse"]:
                best = row
                best_pred = pred
    assert best is not None and best_pred is not None
    return best_pred, best


def fit_knn_template(
    model_name: str,
    X_dist: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    y_fit = np.nan_to_num(np.where(y_mask, y, np.nan), nan=0.0).astype(np.float32)
    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None
    for k in KNN_K_VALUES:
        k_eff = int(min(k, len(train_idx)))
        pred = weighted_template_prediction(X_dist, y_fit, train_idx, k_eff)
        val_rmse = eval_utils.rmse(y[val_idx], pred[val_idx], y_mask[val_idx])
        row = {
            "model_name": model_name,
            "status": "ok",
            "selected_k": k_eff,
            "val_rmse": float(val_rmse),
            "train_rmse": float(eval_utils.rmse(y[train_idx], pred[train_idx], y_mask[train_idx])),
            "selection_metric": "val_rmse",
            "template_source": "train split labels only",
            "uses_subject_id": False,
            "uses_physio": False,
            "uses_eeg": False,
            "uses_continuous_style": False,
            "standardization_scope": "train split scaler + train-variance distance feature selection only",
        }
        if best is None or val_rmse < best["val_rmse"]:
            best = row
            best_pred = pred
    assert best is not None and best_pred is not None
    return best_pred, best


def knn_weighted_values(
    X_dist: np.ndarray,
    train_idx: np.ndarray,
    train_values: np.ndarray,
    k: int,
) -> np.ndarray:
    train_x = X_dist[train_idx]
    out = np.zeros((X_dist.shape[0],) + train_values.shape[1:], dtype=np.float64)
    for i in range(X_dist.shape[0]):
        dx = train_x - X_dist[i : i + 1]
        dist = np.sqrt(np.sum(dx * dx, axis=1))
        order = np.argsort(dist)[: min(k, len(dist))]
        selected_dist = dist[order]
        weights = 1.0 / np.maximum(selected_dist, 1e-6)
        weights = weights / weights.sum()
        out[i] = np.sum(train_values[order] * weights.reshape((-1,) + (1,) * (train_values.ndim - 1)), axis=0)
    return out


class KnnDirectionModel:
    def __init__(self, X_dist: np.ndarray, train_idx: np.ndarray, direction: np.ndarray, k: int) -> None:
        self.X_dist = X_dist
        self.train_idx = train_idx
        self.direction = direction.astype(np.float64)
        self.k = int(k)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X is not self.X_dist:
            raise RuntimeError("KnnDirectionModel is scoped to the fitted feature matrix")
        votes = knn_weighted_values(self.X_dist, self.train_idx, self.direction[self.train_idx], self.k)
        return np.where(votes >= 0.0, 1, -1)


class KnnScalarModel:
    def __init__(self, X_dist: np.ndarray, train_idx: np.ndarray, target: np.ndarray, k: int) -> None:
        self.X_dist = X_dist
        self.train_idx = train_idx
        self.target = target.astype(np.float64)
        self.k = int(k)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X is not self.X_dist:
            raise RuntimeError("KnnScalarModel is scoped to the fitted feature matrix")
        return knn_weighted_values(self.X_dist, self.train_idx, self.target[self.train_idx], self.k)


def fit_direction_classifier(
    X_dist: np.ndarray,
    direction: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[KnnDirectionModel, dict[str, Any]]:
    best: dict[str, Any] | None = None
    best_model: KnnDirectionModel | None = None
    for k in KNN_K_VALUES:
        k_eff = int(min(k, len(train_idx)))
        model = KnnDirectionModel(X_dist, train_idx, direction, k_eff)
        pred_val = model.predict(X_dist)[val_idx]
        acc = float(np.mean(pred_val == direction[val_idx]))
        row = {"direction_clf_k": k_eff, "direction_val_accuracy": acc}
        if best is None or acc > best["direction_val_accuracy"]:
            best = row
            best_model = model
    assert best is not None and best_model is not None
    return best_model, best


def weighted_template_prediction(
    X_dist: np.ndarray,
    y_templates: np.ndarray,
    train_idx: np.ndarray,
    k: int,
    direction_filter: np.ndarray | None = None,
    train_direction: np.ndarray | None = None,
) -> np.ndarray:
    pred = np.zeros((X_dist.shape[0], y_templates.shape[1]), dtype=np.float32)
    train_x = X_dist[train_idx]
    train_y = y_templates[train_idx]
    for i in range(X_dist.shape[0]):
        eligible_local = np.arange(len(train_idx))
        if direction_filter is not None and train_direction is not None:
            same = np.where(train_direction[train_idx] == direction_filter[i])[0]
            if same.size >= 3:
                eligible_local = same
        dx = train_x[eligible_local] - X_dist[i : i + 1]
        dist = np.sqrt(np.sum(dx * dx, axis=1))
        order = np.argsort(dist)[: min(k, len(dist))]
        selected = eligible_local[order]
        selected_dist = dist[order]
        weights = 1.0 / np.maximum(selected_dist, 1e-6)
        weights = weights / weights.sum()
        pred[i] = np.sum(train_y[selected] * weights[:, None], axis=0)
    return pred


def fit_direction_gated_knn_template(
    X_dist: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    peaks: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    model_name = "direction_gated_knn_template_no_subject"
    y_fit = np.nan_to_num(np.where(y_mask, y, np.nan), nan=0.0).astype(np.float32)
    clf, clf_info = fit_direction_classifier(X_dist, peaks["direction"], train_idx, val_idx)
    pred_direction = np.where(clf.predict(X_dist) > 0, 1, -1)
    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None
    for k in KNN_K_VALUES:
        k_eff = int(min(k, len(train_idx)))
        pred = weighted_template_prediction(
            X_dist,
            y_fit,
            train_idx,
            k_eff,
            direction_filter=pred_direction,
            train_direction=peaks["direction"],
        )
        val_rmse = eval_utils.rmse(y[val_idx], pred[val_idx], y_mask[val_idx])
        row = {
            "model_name": model_name,
            "status": "ok",
            "selected_k": k_eff,
            "val_rmse": float(val_rmse),
            "train_rmse": float(eval_utils.rmse(y[train_idx], pred[train_idx], y_mask[train_idx])),
            "selection_metric": "val_rmse_after_direction_classifier_selected_on_val_accuracy",
            "template_source": "train split labels only",
            "uses_subject_id": False,
            "uses_physio": False,
            "uses_eeg": False,
            "uses_continuous_style": False,
            "standardization_scope": "train split scaler + train-variance distance feature selection only",
            **clf_info,
        }
        if best is None or val_rmse < best["val_rmse"]:
            best = row
            best_pred = pred
    assert best is not None and best_pred is not None
    return best_pred, best


def fit_peak_magnitude_regressor(
    X_dist: np.ndarray,
    peaks: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[KnnScalarModel, dict[str, Any]]:
    best: dict[str, Any] | None = None
    best_model: KnnScalarModel | None = None
    target = peaks["peak_abs"].astype(np.float64)
    for k in KNN_K_VALUES:
        k_eff = int(min(k, len(train_idx)))
        model = KnnScalarModel(X_dist, train_idx, target, k_eff)
        pred = model.predict(X_dist)
        val_mae = float(np.mean(np.abs(pred[val_idx] - target[val_idx])))
        row = {"peak_regressor_k": k_eff, "peak_abs_val_mae": val_mae}
        if best is None or val_mae < best["peak_abs_val_mae"]:
            best = row
            best_model = model
    assert best is not None and best_model is not None
    return best_model, best


def fit_peak_scaled_template(
    X_dist: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    peaks: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    model_name = "peak_scaled_template_context_no_subject"
    y_fit = np.nan_to_num(np.where(y_mask, y, np.nan), nan=0.0).astype(np.float32)
    scale = np.maximum(peaks["peak_abs"].astype(np.float32), 1e-4)
    y_norm = y_fit / scale[:, None]
    clf, clf_info = fit_direction_classifier(X_dist, peaks["direction"], train_idx, val_idx)
    peak_reg, peak_info = fit_peak_magnitude_regressor(X_dist, peaks, train_idx, val_idx)
    pred_direction = np.where(clf.predict(X_dist) > 0, 1, -1)
    pred_peak_abs = np.maximum(peak_reg.predict(X_dist).astype(np.float32), 0.0)
    train_p99 = float(np.nanpercentile(peaks["peak_abs"][train_idx], 99))
    pred_peak_abs = np.clip(pred_peak_abs, 0.0, max(train_p99 * 1.3, 1.0))

    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None
    for k in KNN_K_VALUES:
        k_eff = int(min(k, len(train_idx)))
        norm_pred = weighted_template_prediction(
            X_dist,
            y_norm,
            train_idx,
            k_eff,
            direction_filter=pred_direction,
            train_direction=peaks["direction"],
        )
        pred = (norm_pred * pred_peak_abs[:, None]).astype(np.float32)
        val_rmse = eval_utils.rmse(y[val_idx], pred[val_idx], y_mask[val_idx])
        row = {
            "model_name": model_name,
            "status": "ok",
            "selected_k": k_eff,
            "val_rmse": float(val_rmse),
            "train_rmse": float(eval_utils.rmse(y[train_idx], pred[train_idx], y_mask[train_idx])),
            "selection_metric": "val_rmse_after_direction_and_peak_aux_models_selected_on_val",
            "template_source": "train split labels only; normalized by train label peak for templates",
            "uses_subject_id": False,
            "uses_physio": False,
            "uses_eeg": False,
            "uses_continuous_style": False,
            "standardization_scope": "train split scaler + train-variance distance feature selection only",
            **clf_info,
            **peak_info,
        }
        if best is None or val_rmse < best["val_rmse"]:
            best = row
            best_pred = pred
    assert best is not None and best_pred is not None
    return best_pred, best


def load_old_direct_metrics() -> dict[str, Any]:
    path = OLD_DIRECT_DIR / "tables" / "oldcode_vehicle_direct_full_metrics.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    checkpoint_col = "checkpoint" if "checkpoint" in df.columns else "checkpoint_tag"
    rows = df[(df[checkpoint_col] == "active_legacy_best") & (df["split"] == "test")]
    if rows.empty:
        rows = df[df["split"] == "test"].head(1)
    if rows.empty:
        return {}
    row = rows.iloc[0].to_dict()
    return {
        "rmse_steer": row.get("rmse_steer", row.get("sample_rmse_steer", np.nan)),
        "wrong_side_rate": row.get("wrong_side_rate", row.get("sample_wrong_side_rate", np.nan)),
        "severe_amp_under_rate": row.get("severe_amp_under_rate", row.get("sample_severe_amp_under_rate", np.nan)),
    }


def evaluate_predictions(
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    train_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    peaks = np.nanmax(np.abs(np.where(y_mask, y, np.nan)), axis=1)
    large_thr = float(np.nanpercentile(peaks[train_idx], 75))
    difficult_thr = float(np.nanpercentile(peaks[train_idx], 80))
    rows: list[pd.DataFrame] = []
    for split_name in ["train", "val", "test"]:
        split_mask = meta[SPLIT_STRATEGY].astype(str).to_numpy() == split_name
        if not split_mask.any():
            continue
        split_meta = meta.loc[split_mask].reset_index(drop=True)
        for model_name, pred in predictions.items():
            sample_rows = eval_utils.sample_metric_rows(
                y[split_mask],
                pred[split_mask],
                y_mask[split_mask],
                label_time,
                split_meta,
                model_name=model_name,
                split_strategy=SPLIT_STRATEGY,
                split_name=split_name,
                window_id=WINDOW_ID,
                large_thr=large_thr,
                difficult_thr=difficult_thr,
            )
            if sample_rows:
                rows.append(pd.DataFrame(sample_rows))
    per_sample = pd.concat(rows, ignore_index=True)
    metrics = eval_utils.aggregate_metrics(per_sample)
    return metrics, per_sample


def select_val_model(metrics: pd.DataFrame, candidates: list[str]) -> str:
    val = metrics[(metrics["split"] == "val") & (metrics["model_name"].isin(candidates))].copy()
    if val.empty:
        return candidates[0]
    return str(val.sort_values("rmse_steer").iloc[0]["model_name"])


def plot_prediction_grid(
    out_path: Path,
    sample_ids: list[str],
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    plot_models: list[tuple[str, str]],
    title: str,
) -> None:
    n = len(sample_ids)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, max(3.2 * rows, 3.4)), squeeze=False)
    id_to_idx = {sid: i for i, sid in enumerate(meta["sample_id"].astype(str))}
    for ax in axes.ravel():
        ax.axis("off")
    for k, sid in enumerate(sample_ids):
        ax = axes.ravel()[k]
        ax.axis("on")
        idx = id_to_idx[sid]
        valid = y_mask[idx] & np.isfinite(y[idx])
        gt = np.where(valid, y[idx], np.nan)
        ax.plot(label_time, gt, color="black", linewidth=1.8, label="GT")
        for model_name, color in plot_models:
            if model_name in predictions:
                ax.plot(
                    label_time,
                    predictions[model_name][idx],
                    color=color,
                    linewidth=1.25,
                    alpha=0.95,
                    label=DISPLAY_NAMES.get(model_name, model_name),
                )
        ax.axhline(0.0, color="#dddddd", linewidth=0.8)
        ax.set_title(
            f"{meta.at[idx, 'subject']} {meta.at[idx, 'anchor_time_rel_s']:.1f}s\n"
            f"peak={np.nanmax(np.abs(gt)):.2f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=8)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(len(labels), 5), fontsize=8)
    fig.suptitle(title, fontsize=12, y=0.955)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_metric_bar_plot(metrics: pd.DataFrame) -> Path:
    test = metrics[metrics["split"] == "test"].copy()
    keep = [
        "formal_ridge_vehicle_context_no_subject",
        "ridge_rich_context_no_subject",
        "rbf_kernel_ridge_context_no_subject",
        "knn_template_context_no_subject",
        "direction_gated_knn_template_no_subject",
        "peak_scaled_template_context_no_subject",
    ]
    test = test[test["model_name"].isin(keep)]
    labels = test["model_name"].tolist()
    display_labels = [DISPLAY_NAMES.get(label, label) for label in labels]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, metric, title in [
        (axes[0, 0], "rmse_steer", "RMSE lower is better"),
        (axes[0, 1], "wrong_side_rate", "Wrong-side rate lower is better"),
        (axes[1, 0], "severe_amp_under_rate", "Severe amplitude under-rate lower is better"),
        (axes[1, 1], "reversal_count_exact_match_rate", "Reversal exact match higher is better"),
    ]:
        ax.barh(display_labels, test[metric].to_numpy(), color="#4c78a8")
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "strong_vehicle_model_metric_bars_test.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_delta_scatter(per_sample: pd.DataFrame, selected_model: str) -> Path:
    formal = per_sample[
        (per_sample["split"] == "test") & (per_sample["model_name"] == "formal_ridge_vehicle_context_no_subject")
    ][["sample_id", "sample_rmse", "gt_peak_abs"]].rename(columns={"sample_rmse": "formal_rmse"})
    selected = per_sample[
        (per_sample["split"] == "test") & (per_sample["model_name"] == selected_model)
    ][["sample_id", "sample_rmse"]].rename(columns={"sample_rmse": "selected_rmse"})
    df = formal.merge(selected, on="sample_id", how="inner")
    df["delta_selected_minus_formal"] = df["selected_rmse"] - df["formal_rmse"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(df["gt_peak_abs"], df["delta_selected_minus_formal"], s=22, alpha=0.75, color="#d95f02")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("GT peak abs")
    ax.set_ylabel(f"{selected_model} RMSE - formal ridge RMSE")
    ax.set_title("Per-sample RMSE change vs formal ridge")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "strong_vehicle_selected_vs_formal_rmse_delta.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    df.to_csv(TABLE_DIR / "selected_vs_formal_per_sample_delta.csv", index=False, encoding="utf-8-sig")
    return path


def build_error_flag_summary(per_sample: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    rows = per_sample[(per_sample["split"] == "test") & (per_sample["model_name"] == selected_model)].copy()
    rows["high_rmse_top20pct"] = rows["sample_rmse"] >= rows["sample_rmse"].quantile(0.8)
    rows["wrong_side_flag"] = rows["wrong_side"] == 1
    rows["large_response_missed_flag"] = (rows["is_large_response"] == 1) & (rows["large_response_recalled"] == 0)
    rows["severe_amp_under_flag"] = rows["severe_amp_under"] == 1
    rows["peak_time_large_error_flag"] = rows["peak_time_abs_error_s"] > 0.5
    rows["onset_delay_large_error_flag"] = rows["onset_delay_abs_error_s"] > 0.5
    rows["tail_drift_flag"] = rows["tail_drift_risk"] == 1
    rows["zero_crossing_mismatch_flag"] = rows["zero_crossing_mismatch"] == 1
    rows["reversal_mismatch_flag"] = rows["reversal_count_exact"] == 0
    rows["multi_segment_mismatch_flag"] = rows["gt_multi_segment"] != rows["pred_multi_segment"]
    flag_cols = [
        "high_rmse_top20pct",
        "wrong_side_flag",
        "large_response_missed_flag",
        "severe_amp_under_flag",
        "peak_time_large_error_flag",
        "onset_delay_large_error_flag",
        "tail_drift_flag",
        "zero_crossing_mismatch_flag",
        "reversal_mismatch_flag",
        "multi_segment_mismatch_flag",
    ]
    summary = pd.DataFrame(
        [
            {
                "selected_model": selected_model,
                "error_flag": col,
                "n_samples": int(rows[col].sum()),
                "rate": float(rows[col].mean()),
                "mean_rmse": float(rows.loc[rows[col], "sample_rmse"].mean()) if rows[col].any() else float("nan"),
            }
            for col in flag_cols
        ]
    ).sort_values(["n_samples", "error_flag"], ascending=[False, True])
    rows.to_csv(TABLE_DIR / "selected_model_per_sample_error_flags.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "selected_model_error_flag_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def write_reports(
    metrics: pd.DataFrame,
    per_sample: pd.DataFrame,
    model_info: pd.DataFrame,
    selected_model: str,
    figure_paths: dict[str, Path],
    error_summary: pd.DataFrame,
    old_direct: dict[str, Any],
) -> None:
    test = metrics[metrics["split"] == "test"].sort_values("rmse_steer").copy()
    val = metrics[metrics["split"] == "val"].sort_values("rmse_steer").copy()
    show_cols = [
        "model_name",
        "n_samples",
        "rmse_steer",
        "peak_direction_accuracy",
        "wrong_side_rate",
        "large_response_recall",
        "peak_amp_mae",
        "peak_amp_ratio_pred_over_gt_mean",
        "severe_amp_under_rate",
        "peak_time_mae_s",
        "onset_delay_mae_s",
        "tail_abs_error_mean",
        "tail_drift_risk_rate",
        "reversal_count_exact_match_rate",
        "difficult_top20_rmse",
    ]
    test_table = test[[c for c in show_cols if c in test.columns]]
    val_table = val[["model_name", "n_samples", "rmse_steer", "wrong_side_rate", "severe_amp_under_rate", "reversal_count_exact_match_rate"]]
    selected_test = test[test["model_name"] == selected_model].iloc[0].to_dict()
    formal_test = test[test["model_name"] == "formal_ridge_vehicle_context_no_subject"].iloc[0].to_dict()
    old_rmse = old_direct.get("rmse_steer", np.nan)
    old_wrong = old_direct.get("wrong_side_rate", np.nan)
    old_amp = old_direct.get("severe_amp_under_rate", np.nan)

    report = f"""# 阶段 3：更强车辆-only 时序/结构化基线 v0.1

生成时间：2026-05-12

## 目的

上一轮正式车辆-only ridge 基线说明，主要错误集中在幅值不足、启动延迟、反向修正、多段修正和尾段漂移。本轮继续在阶段 3 内部强化车辆-only 基线，仍然不使用生理、脑电、连续风格或驾驶员 ID。

## 输入和边界

- 样本：`{SAMPLES_PATH.as_posix()}`
- 车辆窗口：`{(ARRAY_DIR / (WINDOW_ID + '.npz')).as_posix()}`
- 主窗口：`{WINDOW_ID}`
- 切分：`{SPLIT_STRATEGY}`
- 选择规则：候选模型和超参数只使用 train/val；test 只用于最终报告。
- 特征：事件前 2 秒车辆时序统计、下采样车辆历史、事件/道路上下文；不包含 `eval_label_*`、生理、脑电、连续风格、驾驶员 ID。

## 候选模型

1. `formal_ridge_vehicle_context_no_subject`：上一轮正式 ridge 上下文基线，作为本轮内部参考。
2. `ridge_rich_history_no_subject`：更丰富的事件前车辆历史统计 + 下采样时序，不含上下文。
3. `ridge_rich_context_no_subject`：丰富车辆历史 + 事件/道路上下文。
4. `rbf_kernel_ridge_context_no_subject`：RBF kernel ridge 非线性车辆模型，直接预测整条轨迹。
5. `knn_template_context_no_subject`：车辆特征检索训练集响应模板。
6. `direction_gated_knn_template_no_subject`：先预测主峰方向，再在同方向训练模板中检索。
7. `peak_scaled_template_context_no_subject`：先预测方向和峰值幅值，再检索归一化模板并按幅值缩放。

## val 选择结果

本轮预先规定按 val RMSE 选择候选模型。val 排名如下：

{val_table.to_string(index=False)}

val 选择模型：`{selected_model}`。

## session-level test 指标

{test_table.to_string(index=False)}

旧 `vehicle_direct` clean active checkpoint 只作为历史参照：RMSE={old_rmse:.6f}，错侧率={old_wrong:.6f}，严重幅值不足率={old_amp:.6f}。

## 关键图

- 固定样本预测图：`{figure_paths['fixed'].as_posix()}`
- val 选择模型坏样本图：`{figure_paths['bad'].as_posix()}`
- test 指标柱状图：`{figure_paths['bars'].as_posix()}`
- 与 formal ridge 的逐样本 RMSE 差异：`{figure_paths['delta'].as_posix()}`

## val 选择模型的坏样本错误分型

{error_summary.to_string(index=False)}

## 当前判断

本轮是车辆-only 强化，不支持任何风格、生理或 EEG 有效性结论。和上一轮 formal ridge 相比，val 选择模型 test RMSE 从 {formal_test['rmse_steer']:.6f} 变为 {selected_test['rmse_steer']:.6f}，错侧率从 {formal_test['wrong_side_rate']:.6f} 变为 {selected_test['wrong_side_rate']:.6f}，严重幅值不足率从 {formal_test['severe_amp_under_rate']:.6f} 变为 {selected_test['severe_amp_under_rate']:.6f}，反向修正精确匹配率从 {formal_test['reversal_count_exact_match_rate']:.6f} 变为 {selected_test['reversal_count_exact_match_rate']:.6f}。

是否升级为阶段 3 主车辆基线，不能只看一个 RMSE，需要同时看固定图、坏样本图、错侧、幅值、尾段和反向/多段修正。
"""
    (REPORT_DIR / "stage03_vehicle_instability_strong_vehicle_baselines_v0_1_cn.md").write_text(report, encoding="utf-8")

    user = f"""# 阶段 3 用户查看版：更强车辆-only 时序/结构化基线 v0.1

生成时间：2026-05-12

## 为什么做

上一轮车辆-only ridge 基线已经说明：只靠简单统计特征，模型经常出现幅值不足、启动延迟、尾段漂移、反向修正和多段修正错误。所以这一步继续强化“纯车辆基线”，先把车辆本身能做到什么程度压实，再谈风格和生理。

## 这次检查了什么

这次仍然只用车辆历史和事件/道路上下文，不用生理、脑电、连续风格，也不用驾驶员 ID。模型包括更丰富的 ridge、RBF kernel ridge 非线性模型、KNN 模板检索、方向门控模板、峰值缩放模板。

## 目前发现

模型选择只看验证集。验证集选出的模型是：`{selected_model}`。

session-level test 的主要结果如下：

{test_table.to_string(index=False)}

旧 `vehicle_direct` clean 只作为历史参照，不能当新流程真相。

## 哪些结果可信

- 使用同一批正式高置信失稳样本。
- 使用同一个 session-level split。
- 不含生理、脑电、连续风格或驾驶员 ID。
- 标准化和距离特征筛选都只在 train split 拟合。
- 超参数和候选模型选择只用 val，test 只用于最终评估。

## 哪些还不能下结论

这一步仍然不能说明生理、脑电或连续风格是否有效。即使某个纯车辆模型 RMSE 更低，也还要看方向、幅值、尾段、反向修正、多段修正和坏样本图是否真的变好。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_strong_vehicle_baselines_v0_1_cn.md`
2. `{figure_paths['fixed'].as_posix()}`
3. `{figure_paths['bad'].as_posix()}`
4. `{figure_paths['bars'].as_posix()}`
5. `{(TABLE_DIR / 'strong_vehicle_baseline_metrics.csv').as_posix()}`
"""
    (REPORT_DIR / "stage03_vehicle_instability_strong_vehicle_baselines_user_summary_cn.md").write_text(user, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    samples = pd.read_csv(SAMPLES_PATH)
    y, y_mask, input_values, input_time, label_time, meta = formal_v01.load_window(WINDOW_ID, samples)
    train_idx, val_idx, test_idx = split_indices(meta)
    if not (len(train_idx) and len(val_idx) and len(test_idx)):
        raise RuntimeError("train/val/test split is incomplete for strong vehicle baseline")

    formal_preds, _ = formal_v01.build_predictions(y, y_mask, input_values, input_time, label_time, meta, SPLIT_STRATEGY)
    predictions: dict[str, np.ndarray] = {
        "formal_ridge_vehicle_context_no_subject": formal_preds["ridge_vehicle_context_no_subject"],
    }
    model_info_rows: list[dict[str, Any]] = [
        {
            "model_name": "formal_ridge_vehicle_context_no_subject",
            "status": "reference_from_stage03_formal_baselines_v0_1",
            "selection_metric": "previous formal ridge alpha selected on val",
            "uses_subject_id": False,
            "uses_physio": False,
            "uses_eeg": False,
            "uses_continuous_style": False,
        }
    ]

    X_hist, hist_names = build_rich_vehicle_features(input_values, input_time, meta, train_idx, include_context=False)
    X_ctx, ctx_names = build_rich_vehicle_features(input_values, input_time, meta, train_idx, include_context=True)
    X_hist_scaled, hist_scaler = standardize_train_only(X_hist, train_idx)
    X_ctx_scaled, ctx_scaler = standardize_train_only(X_ctx, train_idx)
    X_ctx_dist, ctx_selector = make_distance_features(X_ctx_scaled, train_idx)
    peaks = peak_arrays(y, y_mask, label_time)

    pred, info = fit_direct_ridge("ridge_rich_history_no_subject", X_hist_scaled, y, train_idx, val_idx, y_mask)
    predictions[info["model_name"]] = pred
    info.update({"feature_count": int(len(hist_names)), "context_included": False})
    model_info_rows.append(info)

    pred, info = fit_direct_ridge("ridge_rich_context_no_subject", X_ctx_scaled, y, train_idx, val_idx, y_mask)
    predictions[info["model_name"]] = pred
    info.update({"feature_count": int(len(ctx_names)), "context_included": True})
    model_info_rows.append(info)

    pred, info = fit_rbf_kernel_ridge_direct(X_ctx_dist, y, train_idx, val_idx, y_mask)
    predictions[info["model_name"]] = pred
    info.update({"feature_count": int(len(ctx_names)), "context_included": True})
    model_info_rows.append(info)

    pred, info = fit_knn_template("knn_template_context_no_subject", X_ctx_dist, y, y_mask, train_idx, val_idx)
    predictions[info["model_name"]] = pred
    info.update(
        {
            "feature_count": int(len(ctx_names)),
            "context_included": True,
            "distance_feature_components": int(ctx_selector.n_components_),
            "distance_feature_variance_ratio_sum": float(np.sum(ctx_selector.explained_variance_ratio_)),
        }
    )
    model_info_rows.append(info)

    pred, info = fit_direction_gated_knn_template(X_ctx_dist, y, y_mask, peaks, train_idx, val_idx)
    predictions[info["model_name"]] = pred
    info.update(
        {
            "feature_count": int(len(ctx_names)),
            "context_included": True,
            "distance_feature_components": int(ctx_selector.n_components_),
            "distance_feature_variance_ratio_sum": float(np.sum(ctx_selector.explained_variance_ratio_)),
        }
    )
    model_info_rows.append(info)

    pred, info = fit_peak_scaled_template(X_ctx_dist, y, y_mask, peaks, train_idx, val_idx)
    predictions[info["model_name"]] = pred
    info.update(
        {
            "feature_count": int(len(ctx_names)),
            "context_included": True,
            "distance_feature_components": int(ctx_selector.n_components_),
            "distance_feature_variance_ratio_sum": float(np.sum(ctx_selector.explained_variance_ratio_)),
        }
    )
    model_info_rows.append(info)

    metrics, per_sample = evaluate_predictions(y, y_mask, label_time, meta, predictions, train_idx)
    candidate_names = [
        "ridge_rich_history_no_subject",
        "ridge_rich_context_no_subject",
        "rbf_kernel_ridge_context_no_subject",
        "knn_template_context_no_subject",
        "direction_gated_knn_template_no_subject",
        "peak_scaled_template_context_no_subject",
    ]
    selected_model = select_val_model(metrics, candidate_names)
    model_info = pd.DataFrame(model_info_rows)
    model_info["selected_by_val_for_v0_1"] = model_info["model_name"] == selected_model

    metrics.to_csv(TABLE_DIR / "strong_vehicle_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(TABLE_DIR / "strong_vehicle_baseline_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    model_info.to_csv(TABLE_DIR / "strong_vehicle_model_info.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature_name": hist_names}).to_csv(TABLE_DIR / "rich_history_feature_names.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature_name": ctx_names}).to_csv(TABLE_DIR / "rich_context_feature_names.csv", index=False, encoding="utf-8-sig")

    fixed_path = FORMAL_BASELINE_DIR / "tables" / "formal_baseline_fixed_plot_samples.csv"
    if fixed_path.exists():
        fixed_ids = pd.read_csv(fixed_path)["sample_id"].astype(str).head(12).tolist()
    else:
        fixed_ids = meta.loc[test_idx, "sample_id"].astype(str).head(12).tolist()
    selected_test = per_sample[(per_sample["split"] == "test") & (per_sample["model_name"] == selected_model)].copy()
    bad_ids = selected_test.sort_values("sample_rmse", ascending=False).head(12)["sample_id"].astype(str).tolist()
    pd.DataFrame({"sample_id": fixed_ids}).to_csv(TABLE_DIR / "strong_vehicle_fixed_plot_samples.csv", index=False, encoding="utf-8-sig")
    selected_test[selected_test["sample_id"].isin(bad_ids)].sort_values("sample_rmse", ascending=False).to_csv(
        TABLE_DIR / "strong_vehicle_bad_plot_samples.csv", index=False, encoding="utf-8-sig"
    )

    plot_models: list[tuple[str, str]] = []
    for model_name, color in [
        ("formal_ridge_vehicle_context_no_subject", "#d62728"),
        (selected_model, "#1f77b4"),
        ("knn_template_context_no_subject", "#2ca02c"),
        ("peak_scaled_template_context_no_subject", "#9467bd"),
    ]:
        if model_name in predictions and model_name not in [name for name, _ in plot_models]:
            plot_models.append((model_name, color))
    fixed_fig = FIG_DIR / "strong_vehicle_fixed_predictions_test.png"
    bad_fig = FIG_DIR / "strong_vehicle_bad_samples_test.png"
    plot_prediction_grid(
        fixed_fig,
        fixed_ids,
        y,
        y_mask,
        label_time,
        meta,
        predictions,
        plot_models,
        f"Fixed test samples (val-selected: {DISPLAY_NAMES.get(selected_model, selected_model)})",
    )
    plot_prediction_grid(
        bad_fig,
        bad_ids,
        y,
        y_mask,
        label_time,
        meta,
        predictions,
        plot_models,
        f"Worst test samples (val-selected: {DISPLAY_NAMES.get(selected_model, selected_model)})",
    )
    bars_fig = write_metric_bar_plot(metrics)
    delta_fig = write_delta_scatter(per_sample, selected_model)
    error_summary = build_error_flag_summary(per_sample, selected_model)

    old_direct = load_old_direct_metrics()
    summary = {
        "window_config_id": WINDOW_ID,
        "split_strategy": SPLIT_STRATEGY,
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "test_n": int(len(test_idx)),
        "selected_model_by_val_rmse": selected_model,
        "models": sorted(predictions.keys()),
        "server_used": False,
        "credential_file_read": False,
        "uses_subject_id": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "raw_files_modified": False,
        "metrics_path": str(TABLE_DIR / "strong_vehicle_baseline_metrics.csv").replace("\\", "/"),
        "per_sample_path": str(TABLE_DIR / "strong_vehicle_baseline_per_sample_metrics.csv").replace("\\", "/"),
        "fixed_plot": str(fixed_fig).replace("\\", "/"),
        "bad_plot": str(bad_fig).replace("\\", "/"),
        "metric_bars": str(bars_fig).replace("\\", "/"),
        "delta_plot": str(delta_fig).replace("\\", "/"),
    }
    (LOG_DIR / "strong_vehicle_baseline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(
        metrics,
        per_sample,
        model_info,
        selected_model,
        {"fixed": fixed_fig, "bad": bad_fig, "bars": bars_fig, "delta": delta_fig},
        error_summary,
        old_direct,
    )

    test = metrics[metrics["split"] == "test"].sort_values("rmse_steer")
    print(test.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
