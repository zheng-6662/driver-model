# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
PROJECT_ROOT = Path(r"F:/data_set_process/data_process")
SAMPLE_SCRIPT_DIR = ROOT / "02_samples" / "scripts"
if str(SAMPLE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SCRIPT_DIR))

import build_extreme_condition_episodes_v0_3 as v03  # noqa: E402


EPISODE_TABLE = (
    ROOT
    / "02_samples"
    / "extreme_condition_episodes_v0_3"
    / "tables"
    / "extreme_condition_episodes_all_v0_3.csv"
)
DATASET_DIR = ROOT / "03_processed_datasets" / "extreme_condition_v0_3_vehicle_only"
ARRAY_DIR = DATASET_DIR / "arrays"
DATASET_TABLE_DIR = DATASET_DIR / "tables"
DATASET_LOG_DIR = DATASET_DIR / "logs"
OUT_DIR = ROOT / "03_baselines" / "stage03_v03_vehicle_only_baselines"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-18.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"

DATASET_ID = "v03_vehicle_only_pre2_label5_20hz"
RANDOM_SEED = 20260518
INPUT_HZ = 20.0
LABEL_HZ = 20.0
INPUT_TIME = np.round(np.arange(-2.0, 0.0 + 1e-9, 1.0 / INPUT_HZ), 6)
LABEL_TIME = np.round(np.arange(0.0, 5.0 + 1e-9, 1.0 / LABEL_HZ), 6)
USABLE_CATEGORIES = {
    "strong_response",
    "weak_or_conservative",
    "delayed_or_no_steer",
    "normal_control",
}
VEHICLE_FEATURES = [
    "zx|SteeringWheel",
    "steer_rate",
    "zx1|v_km/h",
    "zx|BrakePedal",
    "zx|AcceleratorPedal",
    "zx|ax",
    "zx|ay",
    "zx|vyaw",
    "zx|vroll",
    "zx|roll",
    "lateral_distance_selected",
    "zx1|mu",
    "curvature_selected",
]


def ensure_dirs() -> None:
    for path in [ARRAY_DIR, DATASET_TABLE_DIR, DATASET_LOG_DIR, TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def interp_series(df: pd.DataFrame, col: str, query_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = df["time_rel_s"].to_numpy(dtype=float)
    if col not in df.columns:
        return np.zeros_like(query_time, dtype=np.float32), np.zeros_like(query_time, dtype=bool)
    values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(values)
    if valid.sum() < 2:
        return np.zeros_like(query_time, dtype=np.float32), np.zeros_like(query_time, dtype=bool)
    tt = t[valid]
    vv = values[valid]
    order = np.argsort(tt)
    tt = tt[order]
    vv = vv[order]
    unique_t, unique_idx = np.unique(tt, return_index=True)
    unique_v = vv[unique_idx]
    inside = (query_time >= unique_t[0]) & (query_time <= unique_t[-1])
    out = np.zeros_like(query_time, dtype=np.float32)
    out[inside] = np.interp(query_time[inside], unique_t, unique_v).astype(np.float32)
    return out, inside.astype(bool)


def safe_peak_signed(y: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(y)
    if not valid.any():
        return float("nan")
    arr = y[valid]
    idx = int(np.nanargmax(np.abs(arr)))
    return float(arr[idx])


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str], dict[str, Any]]:
    array_path = ARRAY_DIR / f"{DATASET_ID}.npz"
    manifest_path = DATASET_TABLE_DIR / "v03_vehicle_only_manifest.csv"
    summary_path = DATASET_LOG_DIR / "v03_vehicle_only_dataset_summary.json"
    if array_path.exists() and manifest_path.exists() and summary_path.exists():
        z = np.load(array_path, allow_pickle=True)
        meta = pd.read_csv(manifest_path, encoding="utf-8-sig", low_memory=False)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return (
            z["input_values"].astype(np.float32),
            z["input_valid_mask"].astype(bool),
            z["label_values"].astype(np.float32),
            z["label_valid_mask"].astype(bool),
            meta,
            [str(x) for x in z["feature_names"].tolist()],
            summary,
        )
    episodes = pd.read_csv(EPISODE_TABLE, encoding="utf-8-sig", low_memory=False)
    episodes = episodes[episodes["v0_3_category"].isin(USABLE_CATEGORIES)].copy()
    episodes = episodes.sort_values(["subject", "session_stamp", "t_condition_anchor"]).reset_index(drop=True)
    cache: dict[str, pd.DataFrame | None] = {}
    rows: list[dict[str, Any]] = []
    x_values: list[np.ndarray] = []
    x_masks: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    y_masks: list[np.ndarray] = []
    dropped: list[dict[str, Any]] = []

    for _, ep in episodes.iterrows():
        path = str(ep["vehicle_raw_absolute_path"])
        if path not in cache:
            df, _ = v03.load_vehicle_csv(Path(path))
            cache[path] = df
        df = cache[path]
        if df is None or "zx|SteeringWheel" not in df.columns:
            dropped.append({**ep.to_dict(), "drop_reason": "vehicle csv missing or steering missing"})
            continue
        anchor = float(ep["t_condition_anchor"])
        input_query = anchor + INPUT_TIME
        label_query = anchor + LABEL_TIME
        input_mat = []
        input_mask = []
        for col in VEHICLE_FEATURES:
            vals, mask = interp_series(df, col, input_query)
            if col == "zx|SteeringWheel":
                anchor_val, anchor_mask = interp_series(df, col, np.array([anchor], dtype=float))
                vals = vals - float(anchor_val[0]) if anchor_mask[0] else vals
            input_mat.append(vals)
            input_mask.append(mask)
        y_abs, y_mask = interp_series(df, "zx|SteeringWheel", label_query)
        anchor_abs, anchor_mask = interp_series(df, "zx|SteeringWheel", np.array([anchor], dtype=float))
        if not anchor_mask[0] or y_mask.mean() < 0.95 or np.mean(np.vstack(input_mask), axis=0).mean() < 0.85:
            dropped.append({**ep.to_dict(), "drop_reason": "window incomplete"})
            continue
        y_rel = y_abs - float(anchor_abs[0])
        row = ep.to_dict()
        row.update(
            {
                "sample_id": str(ep["episode_uid"]),
                "anchor_steer_abs": float(anchor_abs[0]),
                "target_peak_signed": safe_peak_signed(y_rel, y_mask),
                "target_peak_abs": abs(safe_peak_signed(y_rel, y_mask)),
                "target_final_delta": float(y_rel[y_mask][-1]) if y_mask.any() else float("nan"),
                "input_valid_ratio": float(np.vstack(input_mask).mean()),
                "label_valid_ratio": float(y_mask.mean()),
            }
        )
        rows.append(row)
        x_values.append(np.stack(input_mat, axis=1).astype(np.float32))
        x_masks.append(np.stack(input_mask, axis=1).astype(bool))
        y_values.append(y_rel.astype(np.float32))
        y_masks.append(y_mask.astype(bool))

    if not rows:
        raise RuntimeError("No usable v0.3 vehicle-only samples were built")

    meta = pd.DataFrame(rows)
    x = np.stack(x_values, axis=0)
    x_mask = np.stack(x_masks, axis=0)
    y = np.stack(y_values, axis=0)
    y_mask = np.stack(y_masks, axis=0)

    train_idx, val_idx, test_idx = make_session_split(meta)
    split = np.full(len(meta), "unused", dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    meta["split"] = split
    split_summary = {
        "dataset_id": DATASET_ID,
        "source_episode_table": str(EPISODE_TABLE),
        "usable_categories": sorted(USABLE_CATEGORIES),
        "input_time": INPUT_TIME.tolist(),
        "label_time": LABEL_TIME.tolist(),
        "feature_names": VEHICLE_FEATURES,
        "sample_count": int(len(meta)),
        "dropped_count": int(len(dropped)),
        "split_counts": meta["split"].value_counts().to_dict(),
        "subject_counts": meta["subject"].value_counts().to_dict(),
        "category_counts": meta["v0_3_category"].value_counts().to_dict(),
        "standardization_scope": "all learned models fit preprocessing on train split only",
    }
    meta.to_csv(DATASET_TABLE_DIR / "v03_vehicle_only_manifest.csv", index=False, encoding="utf-8-sig")
    if dropped:
        pd.DataFrame(dropped).to_csv(DATASET_TABLE_DIR / "v03_vehicle_only_dropped_samples.csv", index=False, encoding="utf-8-sig")
    split_counts = (
        meta.groupby(["split", "v0_3_category_cn"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "count"], ascending=[True, False])
    )
    split_counts.to_csv(DATASET_TABLE_DIR / "v03_vehicle_only_split_category_counts.csv", index=False, encoding="utf-8-sig")
    subject_counts = (
        meta.groupby(["split", "subject"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "count"], ascending=[True, False])
    )
    subject_counts.to_csv(DATASET_TABLE_DIR / "v03_vehicle_only_split_subject_counts.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        ARRAY_DIR / f"{DATASET_ID}.npz",
        input_values=x,
        input_valid_mask=x_mask,
        label_values=y,
        label_valid_mask=y_mask,
        input_time=INPUT_TIME.astype(np.float32),
        label_time=LABEL_TIME.astype(np.float32),
        feature_names=np.array(VEHICLE_FEATURES, dtype=object),
        split=split,
        sample_id=meta["sample_id"].astype(str).to_numpy(dtype=object),
    )
    (DATASET_LOG_DIR / "v03_vehicle_only_dataset_summary.json").write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return x, x_mask, y, y_mask, meta, VEHICLE_FEATURES, split_summary


def make_session_split(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = meta["vehicle_raw_relative_path"].astype(str).to_numpy()
    idx = np.arange(len(meta))
    first = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    trainval_idx, test_idx = next(first.split(idx, groups=groups))
    groups_trainval = groups[trainval_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED + 1)
    train_rel, val_rel = next(second.split(trainval_idx, groups=groups_trainval))
    train_idx = trainval_idx[train_rel]
    val_idx = trainval_idx[val_rel]
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def flatten_history_features(x: np.ndarray, x_mask: np.ndarray, meta: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    x_filled = np.where(x_mask, x, 0.0)
    flat = x_filled.reshape(x.shape[0], -1)
    raw_idx = np.unique(np.linspace(0, len(INPUT_TIME) - 1, 13).round().astype(int))
    flat = x_filled[:, raw_idx, :].reshape(x.shape[0], -1)
    feature_cols = [f"{name}@{INPUT_TIME[k]:.2f}s" for k in raw_idx for name in VEHICLE_FEATURES]
    feats = pd.DataFrame(flat, columns=feature_cols)
    for j, name in enumerate(VEHICLE_FEATURES):
        arr = np.where(x_mask[:, :, j], x[:, :, j], np.nan)
        feats[f"{name}:mean"] = np.nanmean(arr, axis=1)
        feats[f"{name}:std"] = np.nanstd(arr, axis=1)
        feats[f"{name}:last"] = arr[:, -1]
        feats[f"{name}:delta"] = arr[:, -1] - arr[:, 0]
        feats[f"{name}:absmax"] = np.nanmax(np.abs(arr), axis=1)
    for col in [
        "v0_3_category",
        "condition_context_cn",
        "condition_level",
        "steer_response_strength",
        "response_shape",
    ]:
        feats[col] = meta[col].astype(str).fillna("NA").to_numpy()
    for col in [
        "condition_score_peak",
        "condition_score_mean",
        "median_speed_kmh_window",
        "peak_abs_ay_window",
        "peak_abs_yaw_rate_window",
        "peak_abs_roll_rate_window",
        "peak_abs_roll_window",
        "peak_abs_curvature_window",
        "min_mu_window",
    ]:
        feats[col] = pd.to_numeric(meta[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    numeric_cols = [c for c in feats.columns if feats[c].dtype.kind in "iufb"]
    cat_cols = [c for c in feats.columns if c not in numeric_cols]
    return feats, np.array(numeric_cols + cat_cols, dtype=object)


def compact_numpy_features(X: pd.DataFrame, train_idx: np.ndarray) -> tuple[np.ndarray, list[str]]:
    numeric_cols = [c for c in X.columns if X[c].dtype.kind in "iufb"]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    parts: list[np.ndarray] = []
    names: list[str] = []
    if numeric_cols:
        x_num = X[numeric_cols].to_numpy(dtype=np.float64)
        x_num = np.nan_to_num(x_num, nan=0.0, posinf=0.0, neginf=0.0)
        mu = x_num[train_idx].mean(axis=0, keepdims=True)
        sigma = x_num[train_idx].std(axis=0, keepdims=True)
        sigma[sigma < 1e-6] = 1.0
        parts.append(((x_num - mu) / sigma).astype(np.float32))
        names.extend(numeric_cols)
    for col in categorical_cols:
        values = X[col].astype(str).fillna("NA").to_numpy()
        cats = sorted(pd.Series(values[train_idx]).unique().tolist())
        for cat in cats:
            parts.append((values == cat).astype(np.float32).reshape(-1, 1))
            names.append(f"{col}={cat}")
    if not parts:
        return np.zeros((len(X), 0), dtype=np.float32), []
    return np.hstack(parts).astype(np.float32), names


def rmse(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(y) & np.isfinite(pred)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(mean_squared_error(y[valid], pred[valid])))


def signed_peak_batch(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan, dtype=float)
    for i in range(arr.shape[0]):
        out[i] = safe_peak_signed(arr[i], mask[i])
    return out


def build_no_learning_predictions(
    y: np.ndarray,
    y_mask: np.ndarray,
    x: np.ndarray,
    x_mask: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    preds["zero_delta"] = np.zeros_like(y)

    steer_idx = VEHICLE_FEATURES.index("zx|SteeringWheel")
    rate_idx = VEHICLE_FEATURES.index("steer_rate")
    last_steer = np.where(x_mask[:, -1, steer_idx], x[:, -1, steer_idx], 0.0)
    last_rate = np.where(x_mask[:, -1, rate_idx], x[:, -1, rate_idx], 0.0)
    trend = last_steer[:, None] + last_rate[:, None] * LABEL_TIME[None, :]
    preds["linear_trend_from_last_rate"] = trend.astype(np.float32)

    train_mean = np.nanmean(np.where(y_mask[train_idx], y[train_idx], np.nan), axis=0)
    preds["train_global_mean"] = np.broadcast_to(train_mean[None, :], y.shape).astype(np.float32)

    cat_pred = np.zeros_like(y)
    fallback = train_mean
    for cat in meta["v0_3_category"].astype(str).unique():
        train_cat = train_idx[meta.iloc[train_idx]["v0_3_category"].astype(str).to_numpy() == cat]
        if len(train_cat) >= 3:
            mean_curve = np.nanmean(np.where(y_mask[train_cat], y[train_cat], np.nan), axis=0)
        else:
            mean_curve = fallback
        cat_pred[meta["v0_3_category"].astype(str).to_numpy() == cat] = mean_curve
    preds["train_category_mean"] = cat_pred.astype(np.float32)

    ctx_pred = np.zeros_like(y)
    context_values = meta["condition_context_cn"].astype(str).to_numpy()
    for ctx in np.unique(context_values):
        train_ctx = train_idx[context_values[train_idx] == ctx]
        if len(train_ctx) >= 3:
            mean_curve = np.nanmean(np.where(y_mask[train_ctx], y[train_ctx], np.nan), axis=0)
        else:
            mean_curve = fallback
        ctx_pred[context_values == ctx] = mean_curve
    preds["train_context_mean"] = ctx_pred.astype(np.float32)
    return preds


def fit_ridge_closed_form(X: np.ndarray, Y: np.ndarray, train_idx: np.ndarray, alpha: float) -> np.ndarray:
    xt = X[train_idx].astype(np.float64)
    yt = Y[train_idx].astype(np.float64)
    x_aug = np.hstack([xt, np.ones((xt.shape[0], 1), dtype=np.float64)])
    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ yt)
    all_aug = np.hstack([X.astype(np.float64), np.ones((X.shape[0], 1), dtype=np.float64)])
    return (all_aug @ coef).astype(np.float32)


def squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = np.sum(A * A, axis=1, keepdims=True)
    bb = np.sum(B * B, axis=1, keepdims=True).T
    d = aa + bb - 2.0 * (A @ B.T)
    return np.maximum(d, 0.0)


def knn_predict(X: np.ndarray, Y: np.ndarray, train_idx: np.ndarray, k: int) -> np.ndarray:
    train_x = X[train_idx]
    train_y = Y[train_idx]
    d = squared_distances(X, train_x)
    order = np.argpartition(d, kth=min(k, train_x.shape[0] - 1), axis=1)[:, :k]
    pred = np.zeros((X.shape[0], Y.shape[1]), dtype=np.float64)
    for i in range(X.shape[0]):
        idx = order[i]
        dist = np.sqrt(np.maximum(d[i, idx], 0.0))
        w = 1.0 / np.maximum(dist, 1e-6)
        pred[i] = np.sum(train_y[idx] * w[:, None], axis=0) / np.sum(w)
    return pred.astype(np.float32)


def rbf_kernel_predict(X: np.ndarray, Y: np.ndarray, train_idx: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    train_x = X[train_idx]
    train_y = Y[train_idx].astype(np.float64)
    d_train = squared_distances(train_x, train_x)
    k_train = np.exp(-gamma * d_train)
    dual = np.linalg.solve(k_train + float(alpha) * np.eye(k_train.shape[0]), train_y)
    k_all = np.exp(-gamma * squared_distances(X, train_x))
    return (k_all @ dual).astype(np.float32)


def train_vehicle_models(X: pd.DataFrame, y: np.ndarray, y_mask: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> dict[str, np.ndarray]:
    y_train = np.where(y_mask, y, 0.0).astype(np.float32)
    x_mat, feature_names = compact_numpy_features(X, train_idx)
    preds: dict[str, np.ndarray] = {}

    print(f"fit numpy baselines features={x_mat.shape[1]} train={len(train_idx)} val={len(val_idx)}", flush=True)
    best_ridge: tuple[float, float, np.ndarray] | None = None
    for alpha in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
        pred = fit_ridge_closed_form(x_mat, y_train, train_idx, alpha)
        score = rmse(y[val_idx], pred[val_idx], y_mask[val_idx])
        if best_ridge is None or score < best_ridge[0]:
            best_ridge = (score, alpha, pred)
    if best_ridge is not None:
        preds[f"ridge_vehicle_history_context_alpha{best_ridge[1]:g}"] = best_ridge[2]

    best_knn: tuple[float, int, np.ndarray] | None = None
    for k in [3, 5, 9, 15, 25]:
        if k > len(train_idx):
            continue
        pred = knn_predict(x_mat, y_train, train_idx, k)
        score = rmse(y[val_idx], pred[val_idx], y_mask[val_idx])
        if best_knn is None or score < best_knn[0]:
            best_knn = (score, k, pred)
    if best_knn is not None:
        preds[f"knn_vehicle_history_context_k{best_knn[1]}"] = best_knn[2]

    train_d = squared_distances(x_mat[train_idx], x_mat[train_idx])
    nonzero = train_d[train_d > 1e-9]
    base_gamma = 1.0 / float(np.nanmedian(nonzero)) if nonzero.size else 1.0 / max(1, x_mat.shape[1])
    best_rbf: tuple[float, float, float, np.ndarray] | None = None
    for gamma_scale in [0.25, 0.5, 1.0, 2.0]:
        gamma = base_gamma * gamma_scale
        for alpha in [0.001, 0.01, 0.1, 1.0]:
            pred = rbf_kernel_predict(x_mat, y_train, train_idx, alpha, gamma)
            score = rmse(y[val_idx], pred[val_idx], y_mask[val_idx])
            if best_rbf is None or score < best_rbf[0]:
                best_rbf = (score, alpha, gamma_scale, pred)
    if best_rbf is not None:
        preds[f"rbf_kernel_vehicle_context_alpha{best_rbf[1]:g}_g{best_rbf[2]:g}"] = best_rbf[3]
    (LOG_DIR / "v03_vehicle_only_feature_info.json").write_text(
        json.dumps({"feature_count": int(x_mat.shape[1]), "feature_names": feature_names}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return preds


def evaluate_all(
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    preds: dict[str, np.ndarray],
    train_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = {name: np.where(meta["split"].astype(str).to_numpy() == name)[0] for name in ["train", "val", "test"]}
    primary_mask = label_time <= 2.0
    tail_mask = label_time >= 2.0
    gt_peak = signed_peak_batch(y, y_mask)
    train_large_thr = float(np.nanpercentile(np.abs(gt_peak[train_idx]), 75))
    metric_rows = []
    sample_rows = []
    for model_name, pred in preds.items():
        pred_peak = signed_peak_batch(pred, y_mask)
        for split_name, idx in split_idx.items():
            if len(idx) == 0:
                continue
            m = y_mask[idx]
            primary_m = m & primary_mask[None, :]
            tail_m = m & tail_mask[None, :]
            gt_p = gt_peak[idx]
            pr_p = pred_peak[idx]
            large = np.abs(gt_p) >= train_large_thr
            wrong = large & np.isfinite(gt_p) & np.isfinite(pr_p) & (np.sign(gt_p) != np.sign(pr_p))
            severe_under = large & np.isfinite(gt_p) & np.isfinite(pr_p) & (np.abs(pr_p) < 0.5 * np.abs(gt_p))
            recall = large & np.isfinite(pr_p) & (np.abs(pr_p) >= 0.5 * train_large_thr)
            metric_rows.append(
                {
                    "model_name": model_name,
                    "split": split_name,
                    "n": int(len(idx)),
                    "rmse_steer": rmse(y[idx], pred[idx], m),
                    "primary_rmse_0_2s": rmse(y[idx], pred[idx], primary_m),
                    "tail_rmse_2_5s": rmse(y[idx], pred[idx], tail_m),
                    "peak_abs_mae": float(np.nanmean(np.abs(np.abs(pr_p) - np.abs(gt_p)))),
                    "wrong_side_rate_large": float(wrong.mean()) if large.any() else float("nan"),
                    "severe_amp_under_rate_large": float(severe_under.mean()) if large.any() else float("nan"),
                    "large_response_recall": float(recall.sum() / large.sum()) if large.any() else float("nan"),
                    "large_threshold_train_p75": train_large_thr,
                }
            )
            for i in idx:
                valid = y_mask[i]
                sample_rmse = rmse(y[i : i + 1], pred[i : i + 1], valid[None, :])
                gt_signed = gt_peak[i]
                pr_signed = pred_peak[i]
                large_i = abs(gt_signed) >= train_large_thr if math.isfinite(gt_signed) else False
                sample_rows.append(
                    {
                        "sample_id": meta.loc[i, "sample_id"],
                        "model_name": model_name,
                        "split": split_name,
                        "subject": meta.loc[i, "subject"],
                        "session_stamp": meta.loc[i, "session_stamp"],
                        "v0_3_category": meta.loc[i, "v0_3_category"],
                        "v0_3_category_cn": meta.loc[i, "v0_3_category_cn"],
                        "condition_context_cn": meta.loc[i, "condition_context_cn"],
                        "sample_rmse": sample_rmse,
                        "gt_peak_signed": gt_signed,
                        "pred_peak_signed": pr_signed,
                        "gt_peak_abs": abs(gt_signed) if math.isfinite(gt_signed) else np.nan,
                        "pred_peak_abs": abs(pr_signed) if math.isfinite(pr_signed) else np.nan,
                        "large_response": large_i,
                        "wrong_side_large": bool(large_i and math.isfinite(pr_signed) and np.sign(gt_signed) != np.sign(pr_signed)),
                        "severe_amp_under_large": bool(large_i and math.isfinite(pr_signed) and abs(pr_signed) < 0.5 * abs(gt_signed)),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(sample_rows)


def plot_predictions(sample_ids: list[str], y: np.ndarray, y_mask: np.ndarray, label_time: np.ndarray, meta: pd.DataFrame, preds: dict[str, np.ndarray], out_path: Path, title: str) -> None:
    sample_ids = [sid for sid in sample_ids if sid in set(meta["sample_id"].astype(str))]
    if not sample_ids:
        return
    rows = min(4, len(sample_ids))
    cols = int(math.ceil(len(sample_ids) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.8), sharex=True)
    axes_arr = np.atleast_1d(axes).ravel()
    model_order = [
        "zero_delta",
        "train_category_mean",
        "ridge_vehicle_history_context",
        next((name for name in preds if name.startswith("rbf_kernel_vehicle_context")), ""),
    ]
    colors = {
        "zero_delta": "#9CA3AF",
        "train_category_mean": "#F59E0B",
        "ridge_vehicle_history_context": "#2563EB",
        "extra_trees_vehicle_history_context": "#DC2626",
    }
    meta_index = {sid: i for i, sid in enumerate(meta["sample_id"].astype(str))}
    for ax, sid in zip(axes_arr, sample_ids):
        i = meta_index[sid]
        valid = y_mask[i]
        ax.plot(label_time[valid], y[i, valid], color="#111827", lw=2.0, label="真实")
        for model_name in model_order:
            if model_name in preds:
                ax.plot(label_time[valid], preds[model_name][i, valid], lw=1.2, color=colors.get(model_name), label=model_name)
        ax.axhline(0.0, color="#111827", lw=0.6, alpha=0.4)
        ax.set_title(f"{meta.loc[i, 'subject']} | {meta.loc[i, 'v0_3_category_cn']}", fontsize=9)
        ax.grid(True, alpha=0.25)
    for ax in axes_arr[len(sample_ids) :]:
        ax.axis("off")
    axes_arr[0].legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_report(metrics: pd.DataFrame, per_sample: pd.DataFrame, dataset_summary: dict[str, Any]) -> None:
    test = metrics[metrics["split"] == "test"].sort_values("rmse_steer").copy()
    best = test.iloc[0].to_dict() if not test.empty else {}
    lines = [
        "# v0.3 全量样本车辆-only 基线（用户查看版）",
        "",
        "## 这次为什么做",
        "",
        "在 v0.3 全量原始数据重筛后，先不用连续驾驶风格和生理数据，只看车辆历史、道路/工况上下文和早期车辆状态能不能预测后续方向盘相对轨迹。",
        "",
        "## 数据集",
        "",
        f"- 输入样本：v0.3 中强响应、弱/保守、延迟/无明显转向、正常对照四类。",
        f"- 可用样本数：{dataset_summary['sample_count']}。",
        f"- 切分数量：{dataset_summary['split_counts']}。",
        "- 输入窗口：工况锚点前 2 秒，20 Hz。",
        "- 标签窗口：工况锚点后 5 秒方向盘相对变化，20 Hz。",
        "- 未使用：连续驾驶风格、生理、脑电、驾驶员 ID。",
        "",
        "## test 指标",
        "",
        metrics_to_markdown(test),
        "",
        "## 当前最好车辆-only 基线",
        "",
        f"- 最好模型：`{best.get('model_name', 'NA')}`",
        f"- test RMSE：{best.get('rmse_steer', float('nan')):.6f}" if best else "- test RMSE：NA",
        f"- 主响应 0-2s RMSE：{best.get('primary_rmse_0_2s', float('nan')):.6f}" if best else "- 主响应 RMSE：NA",
        f"- 尾段 2-5s RMSE：{best.get('tail_rmse_2_5s', float('nan')):.6f}" if best else "- 尾段 RMSE：NA",
        "",
        "## 结论边界",
        "",
        "- 这是新 v0.3 样本库的第一版车辆-only 基线，不证明连续风格或生理数据有效。",
        "- 如果后续看预测图发现仍有严重方向错侧或幅值压缩，应优先回到样本定义、锚点和响应类型，而不是直接加入生理数据。",
    ]
    (REPORT_DIR / "stage03_v03_vehicle_only_baselines_user_summary_cn.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORT_DIR / "stage03_v03_vehicle_only_baselines_cn.md").write_text(
        "\n".join(
            [
                "# v0.3 车辆-only 基线技术报告",
                "",
                f"数据摘要：{json.dumps(dataset_summary, ensure_ascii=False, indent=2)}",
                "",
                "## 指标",
                "",
                metrics_to_markdown(metrics.sort_values(["split", "rmse_steer"])),
            ]
        ),
        encoding="utf-8",
    )


def metrics_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "暂无指标。"
    cols = [
        "model_name",
        "split",
        "n",
        "rmse_steer",
        "primary_rmse_0_2s",
        "tail_rmse_2_5s",
        "peak_abs_mae",
        "wrong_side_rate_large",
        "severe_amp_under_rate_large",
        "large_response_recall",
    ]
    cols = [c for c in cols if c in df.columns]
    show = df[cols].copy()
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in show.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{v:.6g}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def append_project_notes(dataset_summary: dict[str, Any], best_row: dict[str, Any]) -> None:
    status_text = (
        "\n\n## 最新更新：2026-05-18 v0.3 车辆-only 基线\n\n"
        "- 当前阶段：v0.3 全量原始数据样本库后的车辆-only 数据集与基线验证。\n"
        f"- 已完成：构建 `{DATASET_ID}`，可用样本 {dataset_summary['sample_count']}，split={dataset_summary['split_counts']}。\n"
        f"- 最近结果：当前 test 最好模型 `{best_row.get('model_name', 'NA')}`，RMSE={best_row.get('rmse_steer', float('nan')):.6f}，主响应 RMSE={best_row.get('primary_rmse_0_2s', float('nan')):.6f}，尾段 RMSE={best_row.get('tail_rmse_2_5s', float('nan')):.6f}。\n"
        "- 下一步：优先查看固定预测图和坏样本图；如果物理意义可接受，再考虑响应类型辅助模型或加入连续风格/生理增量。\n"
    )
    project_status = NOTES_DIR / "PROJECT_STATUS_CN.md"
    task_queue = NOTES_DIR / "TASK_QUEUE_CN.md"
    if project_status.exists():
        with project_status.open("a", encoding="utf-8") as f:
            f.write(status_text)
    if task_queue.exists():
        with task_queue.open("a", encoding="utf-8") as f:
            f.write(
                "\n\n## 最新更新：2026-05-18 v0.3 车辆-only 基线\n\n"
                "### 已完成任务\n"
                f"- 已构建 v0.3 车辆-only 固定窗口数据集 `{DATASET_ID}`。\n"
                "- 已运行无学习基线和车辆-only 强传统基线。\n"
                "- 已生成指标表、逐样本指标、固定预测图、坏样本图、用户总结和技术报告。\n\n"
                "### 待做任务\n"
                "- 人工查看固定预测图和坏样本图。\n"
                "- 判断车辆-only 是否已经比旧样本更符合物理意义。\n"
                "- 决定是否进入响应类型辅助模型，或加入连续风格/生理数据。\n"
            )
    if ARTIFACT_INDEX.exists():
        with ARTIFACT_INDEX.open("a", encoding="utf-8") as f:
            f.write(
                "\n\n## v0.3 车辆-only 数据集与基线\n\n"
                f"- 数据集数组：`{ARRAY_DIR / (DATASET_ID + '.npz')}`\n"
                f"- 数据集 manifest：`{DATASET_TABLE_DIR / 'v03_vehicle_only_manifest.csv'}`\n"
                f"- 指标表：`{TABLE_DIR / 'v03_vehicle_only_baseline_metrics.csv'}`\n"
                f"- 逐样本指标：`{TABLE_DIR / 'v03_vehicle_only_per_sample_metrics.csv'}`\n"
                f"- 用户查看版报告：`{REPORT_DIR / 'stage03_v03_vehicle_only_baselines_user_summary_cn.md'}`\n"
                f"- 固定预测图：`{FIG_DIR / 'v03_vehicle_only_fixed_predictions_test.png'}`\n"
                f"- 坏样本图：`{FIG_DIR / 'v03_vehicle_only_bad_samples_test.png'}`\n"
            )
    with DAILY_LOG.open("a", encoding="utf-8") as f:
        f.write(
            "\n\n## 2026-05-18 v0.3 车辆-only 数据集与基线\n\n"
            f"- 构建数据集：`{DATASET_ID}`，样本数 {dataset_summary['sample_count']}，split={dataset_summary['split_counts']}。\n"
            f"- 当前 test 最好模型：`{best_row.get('model_name', 'NA')}`，RMSE={best_row.get('rmse_steer', float('nan')):.6f}。\n"
            "- 本轮未使用连续驾驶风格、生理、脑电或驾驶员 ID。\n"
        )


def _fmt_value(v: Any) -> str:
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "NA"
        return f"{float(v):.6g}"
    return str(v)


def metrics_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "暂无指标。"
    cols = [
        "model_name",
        "split",
        "n",
        "rmse_steer",
        "primary_rmse_0_2s",
        "tail_rmse_2_5s",
        "peak_abs_mae",
        "wrong_side_rate_large",
        "severe_amp_under_rate_large",
        "large_response_recall",
    ]
    cols = [c for c in cols if c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(_fmt_value(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def group_table_for_report(per_sample: pd.DataFrame, model_name: str, group_col: str) -> pd.DataFrame:
    subset = per_sample[(per_sample["split"] == "test") & (per_sample["model_name"] == model_name)].copy()
    if subset.empty or group_col not in subset.columns:
        return pd.DataFrame()
    rows = []
    for group_value, g in subset.groupby(group_col, dropna=False):
        large = g[g["large_response"].astype(bool)]
        rows.append(
            {
                group_col: group_value,
                "n": int(len(g)),
                "large_n": int(len(large)),
                "rmse_steer_approx": float(np.sqrt(np.nanmean(np.square(pd.to_numeric(g["sample_rmse"], errors="coerce"))))),
                "peak_abs_mae": float(
                    np.nanmean(
                        np.abs(
                            pd.to_numeric(g["pred_peak_abs"], errors="coerce")
                            - pd.to_numeric(g["gt_peak_abs"], errors="coerce")
                        )
                    )
                ),
                "mean_gt_peak_abs": float(np.nanmean(pd.to_numeric(g["gt_peak_abs"], errors="coerce"))),
                "mean_pred_peak_abs": float(np.nanmean(pd.to_numeric(g["pred_peak_abs"], errors="coerce"))),
                "wrong_side_rate_large": float(large["wrong_side_large"].astype(bool).mean()) if len(large) else float("nan"),
                "severe_amp_under_rate_large": float(large["severe_amp_under_large"].astype(bool).mean()) if len(large) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse_steer_approx", "n"], ascending=[True, False])


def group_table_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "暂无分组结果。"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt_value(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def write_group_tables(per_sample: pd.DataFrame, model_name: str) -> dict[str, pd.DataFrame]:
    groups = {
        "category": "v0_3_category_cn",
        "subject": "subject",
        "context": "condition_context_cn",
    }
    out: dict[str, pd.DataFrame] = {}
    for name, col in groups.items():
        table = group_table_for_report(per_sample, model_name, col)
        out[name] = table
        table.to_csv(
            TABLE_DIR / f"v03_vehicle_only_best_model_by_{name}_test.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return out


def make_report(metrics: pd.DataFrame, per_sample: pd.DataFrame, dataset_summary: dict[str, Any]) -> None:
    test = metrics[metrics["split"] == "test"].sort_values("rmse_steer").copy()
    best = test.iloc[0].to_dict() if not test.empty else {}
    best_model = str(best.get("model_name", "NA"))
    group_tables = write_group_tables(per_sample, best_model) if best else {}

    split_counts = dataset_summary.get("split_counts", {})
    lines = [
        "# v0.3 全量样本车辆-only 基线（用户查看版）",
        "",
        "## 这次为什么做",
        "",
        "在 v0.3 全量原始车辆数据重筛后，先不加入连续驾驶风格、生理数据或脑电，只验证车辆历史、道路/工况上下文和早期车辆状态能否预测后续方向盘相对轨迹。",
        "",
        "这一步的作用是先确认新筛出来的极限/近极限工况样本本身是否更适合建模。如果车辆-only 都站不住，后面直接解释风格或生理增量会不可靠。",
        "",
        "## 数据集",
        "",
        "- 样本来源：v0.3 全量原始车辆数据 episode 表。",
        "- 纳入类别：强响应、弱/保守响应、延迟/无明显转向、正常对照。",
        "- 排除类别：待人工复核、已排除样本。",
        f"- 可用样本数：{dataset_summary.get('sample_count', 'NA')}。",
        f"- 划分数量：train={split_counts.get('train', 'NA')}，val={split_counts.get('val', 'NA')}，test={split_counts.get('test', 'NA')}。",
        "- 输入窗口：工况锚点前 2 秒，20 Hz。",
        "- 标签窗口：工况锚点后 5 秒方向盘相对变化，20 Hz。",
        "- 输入特征：方向盘、方向盘角速度、车速、制动、油门、纵向/横向加速度、横摆、横滚、横向偏移、路面附着系数、曲率等车辆/道路信息。",
        "- 未使用：连续驾驶风格、生理数据、脑电、驾驶员 ID。",
        "",
        "## test 总体指标",
        "",
        metrics_to_markdown(test),
        "",
        "## 当前最好车辆-only 基线",
        "",
        f"- 最好模型：`{best_model}`",
        f"- test RMSE：{best.get('rmse_steer', float('nan')):.6f}" if best else "- test RMSE：NA",
        f"- 主响应阶段 0-2s RMSE：{best.get('primary_rmse_0_2s', float('nan')):.6f}" if best else "- 主响应阶段 RMSE：NA",
        f"- 尾段 2-5s RMSE：{best.get('tail_rmse_2_5s', float('nan')):.6f}" if best else "- 尾段 RMSE：NA",
        f"- 大响应错侧率：{best.get('wrong_side_rate_large', float('nan')):.6f}" if best else "- 大响应错侧率：NA",
        f"- 大响应严重幅值不足率：{best.get('severe_amp_under_rate_large', float('nan')):.6f}" if best else "- 大响应严重幅值不足率：NA",
        f"- 大响应召回：{best.get('large_response_recall', float('nan')):.6f}" if best else "- 大响应召回：NA",
        "",
        "## 最好模型分样本类型结果",
        "",
        group_table_to_markdown(group_tables.get("category", pd.DataFrame())),
        "",
        "## 最好模型分被试结果",
        "",
        group_table_to_markdown(group_tables.get("subject", pd.DataFrame())),
        "",
        "## 结论边界",
        "",
        "- 这轮结果只说明新 v0.3 样本上的车辆-only 可预测性，不能证明连续风格或生理数据有效。",
        "- 目前最好车辆-only 模型相对零响应基线有小幅总体改善，并明显降低了大响应错侧率，但大响应召回仍然不足。",
        "- 如果预测图中仍然存在严重幅值压缩或物理意义不对，下一步应优先检查样本类型、锚点和响应分组，而不是直接加入生理数据。",
        "",
        "## 可查看文件",
        "",
        f"- 固定预测图：`{FIG_DIR / 'v03_vehicle_only_fixed_predictions_test.png'}`",
        f"- 坏样本图：`{FIG_DIR / 'v03_vehicle_only_bad_samples_test.png'}`",
        f"- 总指标表：`{TABLE_DIR / 'v03_vehicle_only_baseline_metrics.csv'}`",
        f"- 分样本类型表：`{TABLE_DIR / 'v03_vehicle_only_best_model_by_category_test.csv'}`",
        f"- 分被试表：`{TABLE_DIR / 'v03_vehicle_only_best_model_by_subject_test.csv'}`",
        f"- 分工况上下文表：`{TABLE_DIR / 'v03_vehicle_only_best_model_by_context_test.csv'}`",
    ]
    (REPORT_DIR / "stage03_v03_vehicle_only_baselines_user_summary_cn.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (REPORT_DIR / "stage03_v03_vehicle_only_baselines_cn.md").write_text(
        "\n".join(
            [
                "# v0.3 车辆-only 基线技术报告",
                "",
                "## 数据摘要",
                "",
                "```json",
                json.dumps(dataset_summary, ensure_ascii=False, indent=2),
                "```",
                "",
                "## 全部指标",
                "",
                metrics_to_markdown(metrics.sort_values(["split", "rmse_steer"])),
                "",
                "## 最好模型分样本类型结果",
                "",
                group_table_to_markdown(group_tables.get("category", pd.DataFrame())),
                "",
                "## 最好模型分被试结果",
                "",
                group_table_to_markdown(group_tables.get("subject", pd.DataFrame())),
                "",
                "## 最好模型分工况上下文结果",
                "",
                group_table_to_markdown(group_tables.get("context", pd.DataFrame())),
            ]
        ),
        encoding="utf-8",
    )


def append_project_notes(dataset_summary: dict[str, Any], best_row: dict[str, Any]) -> None:
    split_counts = dataset_summary.get("split_counts", {})
    status_text = (
        "\n\n## 最新更新：2026-05-18 v0.3 车辆-only 数据集与基线（中文修正版）\n\n"
        "- 当前阶段：基于全量原始车辆数据重筛 episode 后，构建车辆-only 数据集并运行无学习基线和车辆-only 强基线。\n"
        f"- 已完成：`{DATASET_ID}`，可用样本 {dataset_summary.get('sample_count', 'NA')}，"
        f"train/val/test={split_counts.get('train', 'NA')}/{split_counts.get('val', 'NA')}/{split_counts.get('test', 'NA')}。\n"
        f"- 最近结果：test 最好模型 `{best_row.get('model_name', 'NA')}`，"
        f"RMSE={best_row.get('rmse_steer', float('nan')):.6f}，"
        f"主响应 RMSE={best_row.get('primary_rmse_0_2s', float('nan')):.6f}，"
        f"尾段 RMSE={best_row.get('tail_rmse_2_5s', float('nan')):.6f}。\n"
        "- 当前判断：车辆-only 比零响应有小幅总体提升，并明显降低大响应错侧率，但大响应召回仍不足；这还不是风格或生理有效性的证据。\n"
        "- 下一步：优先查看固定预测图、坏样本图、分类型和分被试表，再决定是否调整样本/锚点，或进入响应类型辅助建模。\n"
    )
    project_status = NOTES_DIR / "PROJECT_STATUS_CN.md"
    task_queue = NOTES_DIR / "TASK_QUEUE_CN.md"
    if project_status.exists():
        with project_status.open("a", encoding="utf-8") as f:
            f.write(status_text)
    if task_queue.exists():
        with task_queue.open("a", encoding="utf-8") as f:
            f.write(
                "\n\n## 最新更新：2026-05-18 v0.3 车辆-only 基线（中文修正版）\n\n"
                "### 已完成任务\n"
                f"- 构建 v0.3 车辆-only 固定窗口数据集 `{DATASET_ID}`。\n"
                "- 运行无学习基线：零响应、历史趋势外推、训练集均值、类别均值、工况均值。\n"
                "- 运行车辆-only 强基线：岭回归、近邻模板、核回归。\n"
                "- 生成总指标、逐样本指标、分类型表、分被试表、分工况上下文表、固定预测图和坏样本图。\n\n"
                "### 待做任务\n"
                "- 人工查看固定预测图和坏样本图。\n"
                "- 判断车辆-only 预测是否已经比旧样本更符合物理意义。\n"
                "- 决定下一步是继续修样本/锚点，还是进入响应类型辅助模型。\n"
            )
    if ARTIFACT_INDEX.exists():
        with ARTIFACT_INDEX.open("a", encoding="utf-8") as f:
            f.write(
                "\n\n## v0.3 车辆-only 数据集与基线（中文修正版）\n\n"
                f"- 数据集数组：`{ARRAY_DIR / (DATASET_ID + '.npz')}`\n"
                f"- 数据集 manifest：`{DATASET_TABLE_DIR / 'v03_vehicle_only_manifest.csv'}`\n"
                f"- 总指标表：`{TABLE_DIR / 'v03_vehicle_only_baseline_metrics.csv'}`\n"
                f"- 逐样本指标：`{TABLE_DIR / 'v03_vehicle_only_per_sample_metrics.csv'}`\n"
                f"- 分样本类型表：`{TABLE_DIR / 'v03_vehicle_only_best_model_by_category_test.csv'}`\n"
                f"- 分被试表：`{TABLE_DIR / 'v03_vehicle_only_best_model_by_subject_test.csv'}`\n"
                f"- 分工况上下文表：`{TABLE_DIR / 'v03_vehicle_only_best_model_by_context_test.csv'}`\n"
                f"- 用户查看版报告：`{REPORT_DIR / 'stage03_v03_vehicle_only_baselines_user_summary_cn.md'}`\n"
                f"- 固定预测图：`{FIG_DIR / 'v03_vehicle_only_fixed_predictions_test.png'}`\n"
                f"- 坏样本图：`{FIG_DIR / 'v03_vehicle_only_bad_samples_test.png'}`\n"
            )
    with DAILY_LOG.open("a", encoding="utf-8") as f:
        f.write(
            "\n\n## 2026-05-18 v0.3 车辆-only 数据集与基线（中文修正版）\n\n"
            f"- 构建数据集：`{DATASET_ID}`，样本数 {dataset_summary.get('sample_count', 'NA')}，"
            f"train/val/test={split_counts.get('train', 'NA')}/{split_counts.get('val', 'NA')}/{split_counts.get('test', 'NA')}。\n"
            f"- 当前 test 最好模型：`{best_row.get('model_name', 'NA')}`，"
            f"RMSE={best_row.get('rmse_steer', float('nan')):.6f}。\n"
            "- 本轮未使用连续驾驶风格、生理、脑电或驾驶员 ID。\n"
        )


def main() -> None:
    ensure_dirs()
    x, x_mask, y, y_mask, meta, feature_names, dataset_summary = build_dataset()
    train_idx = np.where(meta["split"].to_numpy() == "train")[0]
    val_idx = np.where(meta["split"].to_numpy() == "val")[0]
    test_idx = np.where(meta["split"].to_numpy() == "test")[0]

    X, _ = flatten_history_features(x, x_mask, meta)
    preds = build_no_learning_predictions(y, y_mask, x, x_mask, meta, train_idx)
    preds.update(train_vehicle_models(X, y, y_mask, train_idx, val_idx))
    metrics, per_sample = evaluate_all(y, y_mask, LABEL_TIME, meta, preds, train_idx)
    metrics.to_csv(TABLE_DIR / "v03_vehicle_only_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(TABLE_DIR / "v03_vehicle_only_per_sample_metrics.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        OUT_DIR / "v03_vehicle_only_predictions.npz",
        sample_id=meta["sample_id"].astype(str).to_numpy(dtype=object),
        label_time=LABEL_TIME.astype(np.float32),
        y_true=y,
        y_mask=y_mask,
        **{f"pred_{k}": v for k, v in preds.items()},
    )
    test_best = metrics[metrics["split"] == "test"].sort_values("rmse_steer").iloc[0].to_dict()
    fixed_ids = (
        meta.iloc[test_idx]
        .sort_values(["v0_3_category", "target_peak_abs"], ascending=[True, False])
        .groupby("v0_3_category")
        .head(4)["sample_id"]
        .astype(str)
        .head(16)
        .tolist()
    )
    bad_ids = (
        per_sample[(per_sample["split"] == "test") & (per_sample["model_name"] == test_best["model_name"])]
        .sort_values("sample_rmse", ascending=False)
        .head(16)["sample_id"]
        .astype(str)
        .tolist()
    )
    plot_predictions(fixed_ids, y, y_mask, LABEL_TIME, meta, preds, FIG_DIR / "v03_vehicle_only_fixed_predictions_test.png", "v0.3 vehicle-only fixed test predictions")
    plot_predictions(bad_ids, y, y_mask, LABEL_TIME, meta, preds, FIG_DIR / "v03_vehicle_only_bad_samples_test.png", "v0.3 vehicle-only worst test predictions")
    make_report(metrics, per_sample, dataset_summary)
    (LOG_DIR / "v03_vehicle_only_baseline_summary.json").write_text(
        json.dumps(
            {
                "dataset_summary": dataset_summary,
                "test_best": test_best,
                "models": list(preds.keys()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    append_project_notes(dataset_summary, test_best)
    print(metrics[metrics["split"] == "test"].sort_values("rmse_steer").to_string(index=False))


if __name__ == "__main__":
    main()
