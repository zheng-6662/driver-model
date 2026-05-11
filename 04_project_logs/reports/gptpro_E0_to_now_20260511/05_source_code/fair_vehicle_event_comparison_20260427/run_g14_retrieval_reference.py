# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
TRAINING_DIR = PROJECT_ROOT / "02_code" / "final_code" / "model" / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common_compare_runner import build_args
from event_conditioned_eval_support import annotate_event_meta, build_primary_selection_bundle
from run_event_conditioned_trajectory_baseline import (
    DEFAULT_MANIFEST,
    build_sample_bundle_from_manifest,
    build_teacher_state_context,
    build_driver_style_context,
)


REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
OUT_DIR = REPORTS_DIR / "g14_non_average_prediction_20260510" / "retrieval_reference_stage1"
G11_CATALOG = REPORTS_DIR / "style_physio_eeg_g11_bad_case_attribution_20260509" / "bad_case_catalog.csv"
BASELINE_LOG = REPORTS_DIR / "current_model_version_result_log_20260509.csv"
E10C_SUMMARY = REPORTS_DIR / "style_physio_eeg_e10c_emg_only_3seed_summary_20260509" / "seed_wise_metrics.csv"


@dataclass
class RetrievalResult:
    feature_set: str
    k: int
    pred: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def split_indices(meta_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = meta_df["split"].astype(str).to_numpy()
    return (
        np.where(split == "train")[0].astype(np.int64),
        np.where(split == "val")[0].astype(np.int64),
        np.where(split == "test")[0].astype(np.int64),
    )


def summarize_history_features(x_pool: np.ndarray) -> tuple[np.ndarray, list[str]]:
    x = np.asarray(x_pool, dtype=np.float32)
    n, t, c = x.shape
    windows: list[tuple[str, slice]] = [
        ("all", slice(0, t)),
        ("last_half", slice(t // 2, t)),
        ("last_quarter", slice(3 * t // 4, t)),
    ]
    pieces: list[np.ndarray] = []
    names: list[str] = []
    for ch in range(c):
        xi = x[:, :, ch]
        pieces.append(xi[:, -1:])
        names.append(f"history_ch{ch}_last")
        pieces.append((xi[:, -1:] - xi[:, :1]))
        names.append(f"history_ch{ch}_delta_full")
        for label, slc in windows:
            block = xi[:, slc]
            pieces.append(np.nanmean(block, axis=1, keepdims=True))
            names.append(f"history_ch{ch}_{label}_mean")
            pieces.append(np.nanstd(block, axis=1, keepdims=True))
            names.append(f"history_ch{ch}_{label}_std")
            pieces.append(np.nanmax(block, axis=1, keepdims=True) - np.nanmin(block, axis=1, keepdims=True))
            names.append(f"history_ch{ch}_{label}_range")
        last_len = max(3, min(t, t // 4))
        y = xi[:, -last_len:]
        time = np.linspace(-1.0, 0.0, last_len, dtype=np.float32)
        time = time - float(time.mean())
        denom = float(np.sum(time**2)) + 1e-8
        slope = np.sum((y - np.nanmean(y, axis=1, keepdims=True)) * time.reshape(1, -1), axis=1, keepdims=True) / denom
        pieces.append(slope.astype(np.float32))
        names.append(f"history_ch{ch}_last_quarter_slope")
    feat = np.concatenate(pieces, axis=1).astype(np.float32)
    return feat, names


def numeric_manifest_features(meta_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [
        "is_curve",
        "curvature_anchor",
        "trigger_score",
        "primary_score",
        "event_duration_s",
        "anchor_s",
        "anchor_idx",
    ]
    pieces: list[np.ndarray] = []
    names: list[str] = []
    for col in cols:
        if col not in meta_df.columns:
            continue
        values = pd.to_numeric(meta_df[col], errors="coerce").to_numpy(dtype=np.float32).reshape(-1, 1)
        pieces.append(values)
        names.append(f"manifest_{col}")
    if not pieces:
        return np.zeros((len(meta_df), 0), dtype=np.float32), []
    return np.concatenate(pieces, axis=1).astype(np.float32), names


def categorical_manifest_features(meta_df: pd.DataFrame, train_idx: np.ndarray) -> tuple[np.ndarray, list[str]]:
    cols = ["phase_type", "event_level", "trigger_type", "road_type_anchor", "mechanism_tag"]
    pieces: list[np.ndarray] = []
    names: list[str] = []
    for col in cols:
        if col not in meta_df.columns:
            continue
        values = meta_df[col].fillna("unknown").astype(str)
        cats = sorted(set(values.iloc[train_idx].tolist()))
        for cat in cats:
            pieces.append((values.to_numpy() == cat).astype(np.float32).reshape(-1, 1))
            names.append(f"manifest_{col}={cat}")
    if not pieces:
        return np.zeros((len(meta_df), 0), dtype=np.float32), []
    return np.concatenate(pieces, axis=1).astype(np.float32), names


def standardize_from_train(features: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    feat = np.asarray(features, dtype=np.float32)
    train = feat[train_idx]
    mu = np.nanmean(train, axis=0).astype(np.float32)
    sd = np.nanstd(train, axis=0).astype(np.float32)
    mu[~np.isfinite(mu)] = 0.0
    sd[~np.isfinite(sd)] = 1.0
    sd[sd < 1e-6] = 1.0
    filled = feat.copy()
    bad = ~np.isfinite(filled)
    if np.any(bad):
        rows, cols = np.where(bad)
        filled[rows, cols] = mu[cols]
    z = ((filled - mu.reshape(1, -1)) / sd.reshape(1, -1)).astype(np.float32)
    return z, {"mean": mu.tolist(), "std": sd.tolist()}


def build_available_feature_sets(
    x_pool: np.ndarray,
    ctx_pool: np.ndarray,
    meta_df: pd.DataFrame,
    train_idx: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], dict[str, Any]]:
    hist_feat, hist_names = summarize_history_features(x_pool)
    ctx_feat = np.asarray(ctx_pool, dtype=np.float32)
    ctx_names = ["anchor_steer", "anchor_speed", "anchor_steer_rate", "anchor_ay", "anchor_yawrate"][: ctx_feat.shape[1]]
    num_feat, num_names = numeric_manifest_features(meta_df)
    cat_feat, cat_names = categorical_manifest_features(meta_df, train_idx)

    base = np.concatenate([hist_feat, ctx_feat, num_feat, cat_feat], axis=1).astype(np.float32)
    base_names = hist_names + ctx_names + num_names + cat_names

    e10_args = build_args("vehicle_direct_coarse_fine_raw_emg_only_continuous_style")
    style_ctx, style_meta = build_driver_style_context(
        meta_df=meta_df,
        train_idx=list(map(int, train_idx.tolist())),
        style_vector_csv=str(e10_args.driver_style_vector_csv),
        embed_dim=int(e10_args.driver_style_embed_dim),
        include_iqr=bool(e10_args.driver_style_include_iqr),
    )
    emg_ctx, emg_meta = build_teacher_state_context(
        meta_df=meta_df,
        train_idx=list(map(int, train_idx.tolist())),
        mode="raw_emg_only",
        state_dim=1,
    )
    style_names = [f"style_{i + 1}" for i in range(style_ctx.shape[1])]
    emg_names = [f"emg_state_{i + 1}" for i in range(emg_ctx.shape[1])]

    feature_sets = {
        "触发前车辆和事件信息": base,
        "触发前车辆事件加连续风格": np.concatenate([base, style_ctx], axis=1).astype(np.float32),
        "触发前车辆事件加连续风格和肌电": np.concatenate([base, style_ctx, emg_ctx], axis=1).astype(np.float32),
    }
    feature_names = {
        "触发前车辆和事件信息": base_names,
        "触发前车辆事件加连续风格": base_names + style_names,
        "触发前车辆事件加连续风格和肌电": base_names + style_names + emg_names,
    }
    meta = {
        "style_context_meta": style_meta,
        "emg_context_meta": emg_meta,
        "base_feature_count": int(base.shape[1]),
        "style_feature_count": int(style_ctx.shape[1]),
        "emg_feature_count": int(emg_ctx.shape[1]),
    }
    return feature_sets, feature_names, meta


def response_descriptor_features(y_pool: np.ndarray, mask_pool: np.ndarray) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    names = [
        "true_peak_abs",
        "true_peak_sign_pos",
        "true_peak_sign_neg",
        "true_peak_time_s",
        "true_tail_mean",
        "true_tail_same_side",
        "true_tail_cross_side",
        "morph_single",
        "morph_reverse",
        "morph_multi",
    ]
    from baseline_eval_primary_aux import classify_eval_morphology

    for i in range(y_pool.shape[0]):
        valid = int(mask_pool[i].sum())
        valid = max(1, min(valid, y_pool.shape[1]))
        y = np.asarray(y_pool[i, :valid, 0], dtype=np.float32)
        peak_idx = int(np.argmax(np.abs(y)))
        peak = float(y[peak_idx])
        peak_abs = abs(peak)
        tail_start = max(0, int(0.70 * valid))
        tail_mean = float(np.mean(y[tail_start:]))
        morph = classify_eval_morphology(y, valid)
        rows.append(
            [
                peak_abs,
                1.0 if peak > 0 else 0.0,
                1.0 if peak < 0 else 0.0,
                peak_idx * 2.0 / max(1, y_pool.shape[1]),
                tail_mean,
                1.0 if abs(tail_mean) >= max(0.06, 0.25 * peak_abs) and np.sign(tail_mean) == np.sign(peak) else 0.0,
                1.0 if abs(tail_mean) >= max(0.06, 0.25 * peak_abs) and np.sign(tail_mean) != np.sign(peak) else 0.0,
                1.0 if morph == "single_lobe" else 0.0,
                1.0 if morph == "reverse_correction" else 0.0,
                1.0 if morph == "multi_correction" else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32), names


def nearest_neighbors(
    z_features: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    max_k: int,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    train_feat = z_features[train_idx].astype(np.float32)
    test_feat = z_features[test_idx].astype(np.float32)
    all_indices: list[np.ndarray] = []
    all_dist: list[np.ndarray] = []
    for start in range(0, len(test_idx), chunk_size):
        end = min(len(test_idx), start + chunk_size)
        q = test_feat[start:end]
        dist = (
            np.sum(q * q, axis=1, keepdims=True)
            + np.sum(train_feat * train_feat, axis=1, keepdims=True).T
            - 2.0 * q @ train_feat.T
        )
        dist = np.maximum(dist, 0.0)
        part = np.argpartition(dist, kth=min(max_k, dist.shape[1] - 1), axis=1)[:, :max_k]
        part_dist = np.take_along_axis(dist, part, axis=1)
        order = np.argsort(part_dist, axis=1)
        part = np.take_along_axis(part, order, axis=1)
        part_dist = np.take_along_axis(part_dist, order, axis=1)
        all_indices.append(train_idx[part])
        all_dist.append(np.sqrt(part_dist).astype(np.float32))
    return np.concatenate(all_indices, axis=0), np.concatenate(all_dist, axis=0)


def predict_from_neighbors(
    y_pool: np.ndarray,
    test_idx: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    k: int,
) -> np.ndarray:
    true_test = y_pool[test_idx].astype(np.float32)
    pred = true_test.copy()
    idx = neighbor_indices[:, :k]
    dist = neighbor_distances[:, :k]
    weights = 1.0 / (dist + 1e-3)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    steer_neighbors = y_pool[idx, :, 0].astype(np.float32)
    pred[:, :, 0] = np.sum(steer_neighbors * weights[:, :, None], axis=1)
    if pred.shape[2] > 1:
        pred[:, :, 1:] = true_test[:, :, 1:]
    return pred.astype(np.float32)


def add_physical_columns(sample_df: pd.DataFrame, pred: np.ndarray, true: np.ndarray, mask: np.ndarray, ctx_raw: np.ndarray) -> pd.DataFrame:
    out = sample_df.reset_index(drop=True).copy()
    anchors = ctx_raw[:, 0].astype(np.float32)
    true_abs = true[:, :, 0] + anchors.reshape(-1, 1)
    pred_abs = pred[:, :, 0] + anchors.reshape(-1, 1)
    rows: list[dict[str, float | int | str]] = []
    for i in range(pred.shape[0]):
        valid = int(mask[i].sum())
        valid = max(1, min(valid, pred.shape[1]))
        t = true_abs[i, :valid]
        p = pred_abs[i, :valid]
        peak_i = int(np.argmax(np.abs(t)))
        true_peak = float(t[peak_i])
        pred_at_true_peak = float(p[peak_i])
        true_peak_abs = abs(true_peak)
        pred_peak_abs = float(np.max(np.abs(p)))
        amp_ratio = pred_peak_abs / (true_peak_abs + 1e-6)
        rows.append(
            {
                "true_peak_abs": true_peak_abs,
                "pred_peak_abs": pred_peak_abs,
                "amp_ratio_pred_over_gt": amp_ratio,
                "under_amp": int(true_peak_abs >= 0.10 and amp_ratio < 0.70),
                "severe_under_amp": int(true_peak_abs >= 0.10 and amp_ratio < 0.45),
                "opposite_at_true_peak": int(true_peak_abs >= 0.10 and abs(pred_at_true_peak) >= 0.03 and np.sign(pred_at_true_peak) != np.sign(true_peak)),
                "tail_drift_risk": int(float(out.loc[i, "tail_pre_ratio_abs_steer"]) > 1.20) if "tail_pre_ratio_abs_steer" in out.columns else 0,
                "true_peak_abs_bin": (
                    "large_>=0.3"
                    if true_peak_abs >= 0.30
                    else ("medium_0.1-0.3" if true_peak_abs >= 0.10 else "tiny_<0.1")
                ),
            }
        )
    return pd.concat([out, pd.DataFrame(rows)], axis=1)


def summarize_variant(
    feature_set: str,
    k: int,
    sample_df: pd.DataFrame,
    selection_summary: dict[str, Any],
    g11_keys: set[str],
) -> dict[str, Any]:
    g11 = sample_df[sample_df["sample_key"].astype(str).isin(g11_keys)].reset_index(drop=True)
    high = sample_df[sample_df.get("true_peak_abs_bin", "").astype(str).eq("large_>=0.3")]
    reverse = sample_df[sample_df.get("eval_morphology_label", "").astype(str).eq("reverse_correction")]
    multi = sample_df[sample_df.get("eval_morphology_label", "").astype(str).eq("multi_correction")]

    def mean(part: pd.DataFrame, col: str) -> float:
        return float(part[col].mean()) if col in part.columns and not part.empty else float("nan")

    return {
        "feature_set": feature_set,
        "k": int(k),
        "test_rmse": mean(sample_df, "rmse_2s_abs_steer"),
        "primary_rmse": float(selection_summary.get("overall_primary_steer_rmse", float("nan"))),
        "tail_rmse": mean(sample_df, "rmse_tail_abs_steer"),
        "peak_err_s": mean(sample_df, "peak_time_abs_err_s"),
        "selection_score": float(selection_summary.get("selection_score", float("nan"))),
        "under_amp_rate": mean(sample_df, "under_amp"),
        "severe_under_amp_rate": mean(sample_df, "severe_under_amp"),
        "opposite_peak_rate": mean(sample_df, "opposite_at_true_peak"),
        "tail_drift_risk_rate": mean(sample_df, "tail_drift_risk"),
        "g11_count": int(len(g11)),
        "g11_rmse": mean(g11, "rmse_2s_abs_steer"),
        "g11_tail_rmse": mean(g11, "rmse_tail_abs_steer"),
        "g11_under_amp_rate": mean(g11, "under_amp"),
        "large_rmse": mean(high, "rmse_2s_abs_steer"),
        "reverse_rmse": mean(reverse, "rmse_2s_abs_steer"),
        "multi_rmse": mean(multi, "rmse_2s_abs_steer"),
    }


def group_rows(feature_set: str, k: int, sample_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    families = ["subj", "eval_morphology_label", "true_peak_abs_bin", "phase_type", "road_type_anchor"]
    for family in families:
        if family not in sample_df.columns:
            continue
        for label, part in sample_df.groupby(family, dropna=False):
            rows.append(
                {
                    "feature_set": feature_set,
                    "k": int(k),
                    "group_family": family,
                    "group_label": str(label),
                    "sample_count": int(len(part)),
                    "rmse": float(part["rmse_2s_abs_steer"].mean()),
                    "tail_rmse": float(part["rmse_tail_abs_steer"].mean()),
                    "under_amp_rate": float(part["under_amp"].mean()) if "under_amp" in part.columns else float("nan"),
                    "severe_under_amp_rate": float(part["severe_under_amp"].mean()) if "severe_under_amp" in part.columns else float("nan"),
                    "opposite_peak_rate": float(part["opposite_at_true_peak"].mean()) if "opposite_at_true_peak" in part.columns else float("nan"),
                }
            )
    return rows


def find_latest_prediction_npz(pattern: str) -> Path | None:
    root = PROJECT_ROOT / "tmp" / "event_conditioned_runs"
    paths = list(root.glob(pattern))
    if not paths:
        return None
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0]


def load_baseline_prediction_by_key() -> dict[str, np.ndarray]:
    path = find_latest_prediction_npz("*E10C*seed2026*/prediction_figures/test/prediction_sequences.npz")
    if path is None:
        return {}
    data = np.load(path, allow_pickle=True)
    keys = [str(x) for x in data["sample_key"]]
    pred = data["pred"].astype(np.float32)
    return {key: pred[i] for i, key in enumerate(keys)}


def plot_case_grid(
    plot_dir: Path,
    sample_df: pd.DataFrame,
    pred_map: dict[str, np.ndarray],
    true: np.ndarray,
    ctx_raw: np.ndarray,
    sample_keys: list[str],
    baseline_map: dict[str, np.ndarray],
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    key_to_idx = {str(row.sample_key): int(i) for i, row in sample_df.reset_index(drop=True).iterrows()}
    selected = [key for key in sample_keys if key in key_to_idx][:12]
    if not selected:
        return
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True)
    axes = axes.reshape(-1)
    t = np.arange(true.shape[1]) * 2.0 / max(1, true.shape[1])
    for ax, key in zip(axes, selected):
        i = key_to_idx[key]
        anchor = float(ctx_raw[i, 0])
        ax.plot(t, true[i, :, 0] + anchor, color="black", linewidth=2.0, label="真实")
        if key in baseline_map:
            ax.plot(t, baseline_map[key][:, 0] + anchor, color="#1f77b4", linewidth=1.3, alpha=0.85, label="E10C")
        for name, pred in pred_map.items():
            ax.plot(t, pred[i, :, 0] + anchor, linewidth=1.2, alpha=0.85, label=name)
        row = sample_df.iloc[i]
        ax.set_title(f"{row.get('subj','?')} | {row.get('eval_morphology_label','?')} | {row.get('true_peak_abs_bin','?')}", fontsize=9)
        ax.axhline(0.0, color="#999999", linewidth=0.6)
        ax.grid(True, alpha=0.2)
    for ax in axes[len(selected) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(plot_dir / "g14_retrieval_selected_g11_overview.png", dpi=160)
    plt.close(fig)


def build_neighbor_examples(
    meta_test: pd.DataFrame,
    meta_all: pd.DataFrame,
    result: RetrievalResult,
    feature_set: str,
    g11_keys: set[str],
    max_cases: int = 20,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    test = meta_test.reset_index(drop=True)
    chosen = test[test["sample_key"].astype(str).isin(g11_keys)].head(max_cases).index.to_numpy(dtype=np.int64)
    for local_i in chosen:
        sample_key = str(test.loc[local_i, "sample_key"])
        for rank in range(min(result.k, result.neighbor_indices.shape[1])):
            ni = int(result.neighbor_indices[local_i, rank])
            nrow = meta_all.iloc[ni]
            rows.append(
                {
                    "feature_set": feature_set,
                    "test_sample_key": sample_key,
                    "rank": int(rank + 1),
                    "distance": float(result.neighbor_distances[local_i, rank]),
                    "neighbor_sample_key": str(nrow.get("sample_key", "")),
                    "neighbor_subj": str(nrow.get("subj", "")),
                    "neighbor_phase_type": str(nrow.get("phase_type", "")),
                    "neighbor_road_type": str(nrow.get("road_type_anchor", "")),
                    "neighbor_morphology": str(nrow.get("eval_morphology_label", "")),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("G14 stage 1: building sample bundle...", flush=True)
    x_pool, y_pool, _curve_pool, ctx_pool, mask_pool, meta_df, dropped = build_sample_bundle_from_manifest(
        DEFAULT_MANIFEST,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        seed=2026,
    )
    train_idx, val_idx, test_idx = split_indices(meta_df)
    meta_annotated = annotate_event_meta(meta_df, y_pool, mask_pool)
    meta_test = meta_annotated.iloc[test_idx].reset_index(drop=True)
    y_test = y_pool[test_idx].astype(np.float32)
    mask_test = mask_pool[test_idx].astype(np.float32)
    if mask_test.ndim == 3:
        mask_test_2d = mask_test[:, :, 0]
    else:
        mask_test_2d = mask_test
    ctx_test = ctx_pool[test_idx].astype(np.float32)

    g11_df = pd.read_csv(G11_CATALOG) if G11_CATALOG.exists() else pd.DataFrame()
    g11_keys = set(g11_df["sample_key"].astype(str).tolist()) if not g11_df.empty else set()

    feature_sets, feature_names, context_meta = build_available_feature_sets(x_pool, ctx_pool, meta_annotated, train_idx)
    oracle_feat, oracle_names = response_descriptor_features(y_pool, mask_pool)
    feature_sets["未来响应标签上限诊断"] = oracle_feat
    feature_names["未来响应标签上限诊断"] = oracle_names

    max_k = 20
    k_values = [1, 3, 5, 10, 20]
    metrics_rows: list[dict[str, Any]] = []
    group_summary_rows: list[dict[str, Any]] = []
    g11_detail_frames: list[pd.DataFrame] = []
    neighbor_frames: list[pd.DataFrame] = []
    plot_pred_map: dict[str, np.ndarray] = {}

    baseline_map = load_baseline_prediction_by_key()

    for feature_set, features in feature_sets.items():
        print(f"retrieval feature set: {feature_set}", flush=True)
        z, stat = standardize_from_train(features, train_idx)
        neighbors, distances = nearest_neighbors(z, train_idx, test_idx, max_k=max_k)
        save_json(
            OUT_DIR / f"feature_stats_{feature_set}.json",
            {
                "feature_set": feature_set,
                "feature_count": int(features.shape[1]),
                "feature_names": feature_names[feature_set],
                "standardization": stat,
            },
        )
        for k in k_values:
            pred = predict_from_neighbors(y_pool, test_idx, neighbors, distances, k=k)
            bundle = build_primary_selection_bundle(
                pred=pred,
                true=y_test,
                mask=mask_test_2d,
                ctx_raw=ctx_test,
                meta_df=meta_test,
                split_name="test",
                seed=2026,
            )
            sample_df = add_physical_columns(bundle["sample_df"], pred, y_test, mask_test_2d, ctx_test)
            sample_df["feature_set"] = feature_set
            sample_df["k"] = int(k)
            metrics_rows.append(summarize_variant(feature_set, k, sample_df, bundle["selection_summary"], g11_keys))
            group_summary_rows.extend(group_rows(feature_set, k, sample_df))
            g11_detail = sample_df[sample_df["sample_key"].astype(str).isin(g11_keys)].copy()
            if not g11_detail.empty:
                g11_detail_frames.append(g11_detail)
            if k == 5 and feature_set in {"触发前车辆事件加连续风格", "触发前车辆事件加连续风格和肌电", "未来响应标签上限诊断"}:
                display_name = {
                    "触发前车辆事件加连续风格": "相似历史",
                    "触发前车辆事件加连续风格和肌电": "相似历史+肌电",
                    "未来响应标签上限诊断": "上限诊断",
                }[feature_set]
                plot_pred_map[display_name] = pred
                neighbor_frames.append(
                    build_neighbor_examples(
                        meta_test=meta_test,
                        meta_all=meta_annotated,
                        result=RetrievalResult(feature_set, k, pred, neighbors, distances),
                        feature_set=feature_set,
                        g11_keys=g11_keys,
                    )
                )

    metrics_df = pd.DataFrame(metrics_rows)
    group_df = pd.DataFrame(group_summary_rows)
    g11_detail_df = pd.concat(g11_detail_frames, ignore_index=True) if g11_detail_frames else pd.DataFrame()
    neighbor_df = pd.concat(neighbor_frames, ignore_index=True) if neighbor_frames else pd.DataFrame()

    metrics_df.to_csv(OUT_DIR / "g14_retrieval_metrics.csv", index=False, encoding="utf-8-sig")
    group_df.to_csv(OUT_DIR / "g14_retrieval_group_summary.csv", index=False, encoding="utf-8-sig")
    g11_detail_df.to_csv(OUT_DIR / "g14_retrieval_g11_detail.csv", index=False, encoding="utf-8-sig")
    neighbor_df.to_csv(OUT_DIR / "g14_retrieval_neighbor_examples.csv", index=False, encoding="utf-8-sig")

    if not g11_df.empty and not g11_detail_df.empty:
        compare_cols = ["sample_key", "E10C_rmse_2s", "E6_rmse_2s", "E5A_rmse_2s", "case_id", "primary_failure_type", "repair_class"]
        merged = g11_detail_df.merge(g11_df[[c for c in compare_cols if c in g11_df.columns]], on="sample_key", how="left")
        merged["delta_vs_E10C_catalog"] = merged["rmse_2s_abs_steer"] - pd.to_numeric(merged.get("E10C_rmse_2s"), errors="coerce")
        merged.to_csv(OUT_DIR / "g14_retrieval_g11_vs_existing_models.csv", index=False, encoding="utf-8-sig")

    selected_keys: list[str] = []
    if not g11_df.empty:
        order_col = "E10C_rmse_2s" if "E10C_rmse_2s" in g11_df.columns else "case_score"
        selected_keys = g11_df.sort_values(order_col, ascending=False)["sample_key"].astype(str).head(12).tolist()
    plot_case_grid(
        OUT_DIR / "figures",
        sample_df=meta_test.merge(
            add_physical_columns(
                build_primary_selection_bundle(
                    pred=plot_pred_map.get("上限诊断", y_test),
                    true=y_test,
                    mask=mask_test_2d,
                    ctx_raw=ctx_test,
                    meta_df=meta_test,
                    split_name="test",
                    seed=2026,
                )["sample_df"],
                plot_pred_map.get("上限诊断", y_test),
                y_test,
                mask_test_2d,
                ctx_test,
            )[["sample_key", "true_peak_abs_bin"]],
            on="sample_key",
            how="left",
        ),
        pred_map=plot_pred_map,
        true=y_test,
        ctx_raw=ctx_test,
        sample_keys=selected_keys,
        baseline_map=baseline_map,
    )

    save_json(
        OUT_DIR / "g14_retrieval_run_meta.json",
        {
            "manifest": str(DEFAULT_MANIFEST),
            "dropped_samples": int(dropped),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "k_values": k_values,
            "feature_sets": {name: int(arr.shape[1]) for name, arr in feature_sets.items()},
            "context_meta": context_meta,
        },
    )

    report = build_report(metrics_df, group_df, g11_detail_df)
    write_text(OUT_DIR / "g14_retrieval_stage1_report_cn.md", report)
    print(f"done: {OUT_DIR}", flush=True)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        f = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(f):
        return "NA"
    return f"{f:.{digits}f}"


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据。"
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: fmt(x) if pd.notna(x) else "NA")
        else:
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(col) for col in work.columns]
    rows = work.astype(str).values.tolist()
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    header = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    body = ["| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def build_report(metrics_df: pd.DataFrame, group_df: pd.DataFrame, g11_detail_df: pd.DataFrame) -> str:
    baseline_text = ""
    if BASELINE_LOG.exists():
        try:
            base = pd.read_csv(BASELINE_LOG)
            keep = base[base["version"].astype(str).isin(["E5A", "E6", "E10C"])][
                ["version", "test_rmse", "tail_rmse", "selection", "decision"]
            ]
            baseline_text = df_to_markdown(keep)
        except Exception:
            baseline_text = "基准表读取失败。"

    k5 = metrics_df[metrics_df["k"].eq(5)].copy()
    k5_table = df_to_markdown(k5[
        [
            "feature_set",
            "test_rmse",
            "tail_rmse",
            "selection_score",
            "g11_rmse",
            "large_rmse",
            "reverse_rmse",
            "multi_rmse",
            "severe_under_amp_rate",
            "opposite_peak_rate",
        ]
    ])

    best_overall = metrics_df.sort_values("test_rmse").head(1).iloc[0].to_dict()
    best_g11 = metrics_df.sort_values("g11_rmse").head(1).iloc[0].to_dict()
    subject_lines = ""
    if not group_df.empty:
        subj = group_df[(group_df["k"].eq(5)) & (group_df["group_family"].eq("subj"))].copy()
        subject_lines = df_to_markdown(subj[["feature_set", "group_label", "sample_count", "rmse", "tail_rmse", "severe_under_amp_rate"]])

    g11_text = ""
    if not g11_detail_df.empty:
        g11_k5 = g11_detail_df[g11_detail_df["k"].eq(5)].groupby("feature_set").agg(
            sample_count=("sample_key", "count"),
            rmse=("rmse_2s_abs_steer", "mean"),
            tail_rmse=("rmse_tail_abs_steer", "mean"),
            severe_under_amp_rate=("severe_under_amp", "mean"),
            opposite_peak_rate=("opposite_at_true_peak", "mean"),
        ).reset_index()
        g11_text = df_to_markdown(g11_k5)

    return f"""# G14 第一阶段：相似历史事件参考预测报告

## 1. 这个实验为什么做

G13 的预测图显示，新模型仍然经常输出平均化轨迹：趋势有时相似，但幅值、方向、零线两侧和反向/多段修正的物理意义没有明显改善。因此本阶段先不训练新大模型，而是检查一个更基础的问题：

如果我们在训练集中寻找触发前最相似的历史事件，用这些历史事件的真实后续轨迹作为参考，能不能得到更合理的预测？

这一步可以帮助判断后续应该改模型，还是先改“响应类型判断”和“多候选输出”。

## 2. 输入和公平边界

本阶段用了两类检索：

- 可部署检索：只使用触发前可见信息，包括车辆历史、事件锚点上下文、道路/事件信息、连续驾驶风格和肌电。
- 上限诊断检索：允许使用真实未来响应标签，只用于判断理论上限，不能作为正式推理模型。

样本划分仍使用：

`{DEFAULT_MANIFEST}`

训练、验证、测试样本数量保持为当前 FAIR 协议对应划分。

## 3. 当前强基准

{baseline_text}

## 4. K=5 主要结果

{k5_table}

## 5. 最优结果摘要

- 全体测试样本最优：`{best_overall.get("feature_set")}`，K={int(best_overall.get("k", 0))}，RMSE={fmt(best_overall.get("test_rmse"))}。
- G11 困难样本最优：`{best_g11.get("feature_set")}`，K={int(best_g11.get("k", 0))}，G11 RMSE={fmt(best_g11.get("g11_rmse"))}。

## 6. G11 困难样本汇总

{g11_text}

## 7. 分被试结果

{subject_lines}

## 8. 初步判断

读这个结果时要分清两件事：

- 如果“触发前车辆事件加连续风格和肌电”明显好，说明训练集中存在可用的相似事件，而且肌电/风格能帮助提前找到更像的响应。
- 如果只有“未来响应标签上限诊断”明显好，说明训练集中有相似响应，但当前触发前输入还不能可靠判断应该找哪一类响应。下一步应优先做“先判别响应强弱、方向和形态，再预测轨迹”或“多候选轨迹”。
- 如果上限诊断也没有明显好，说明简单找相似事件不足以修复这些困难样本，后续更应该考虑多解预测、被试适配或任务定义调整。

## 9. 输出文件

- `g14_retrieval_metrics.csv`：所有检索版本和 K 值的整体指标；
- `g14_retrieval_group_summary.csv`：分被试、分响应类型、分幅值等级结果；
- `g14_retrieval_g11_detail.csv`：G11 困难样本逐样本结果；
- `g14_retrieval_g11_vs_existing_models.csv`：G11 与已有模型的逐样本对照；
- `g14_retrieval_neighbor_examples.csv`：代表性困难样本找到的相似训练事件；
- `figures/g14_retrieval_selected_g11_overview.png`：精选困难样本预测图。
"""


if __name__ == "__main__":
    main()
