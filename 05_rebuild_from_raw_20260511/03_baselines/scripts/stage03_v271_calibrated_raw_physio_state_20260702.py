#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v271 calibrated raw physiology state.

v270 证明：直接使用 raw waveform latent 仍不能超过 fixed wait-latest。
一个重要现象是 raw summary/FFT 更容易携带 subject/recording 身份差异。
本轮不再把生理绝对值直接当预测信号，而是验证：

    相对 subject / recording 无标签基线的生理状态变化是否更有行为价值？

做法：
- 复用 v256 已验证的 raw 生理序列缓存：[sample, 6 channel, 400 step]；
- 每个 event 只取 0ms 锚点对应的 observation 前 20s 生理；
- 先构造 raw summary/FFT，再按 subject 和 recording 做无标签 robust z 校准；
- 对 subject-centered / recording-centered raw waveform 做 train-only PCA；
- 只在 train split 做 identity/behavior 筛选；
- 复用 v267/v269 的 wait gate 与 query-prototype pair reranker 评估口径。

边界：
- 不使用 test 标签做特征选择、阈值选择或策略选择；
- subject/recording baseline 只使用无标签生理输入，不使用未来轨迹标签；
- 但 subject/recording baseline 对 val/test 是 transductive / calibrated setting，
  不是纯 cold-start subject-disjoint；
- prototype 仍只来自 train split；
- 生理缓存来自 v256，已通过 no-post-observation 守卫。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

OUT = BASELINES / "v271_calibrated_raw_physio_state_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v271_calibrated_raw_physio_state_20260702_pack.zip"

V256_SEQ = BASELINES / "v256_raw_physio_cnn_fusion_20260702" / "tensors" / "v256_physio_seq_20s_20hz.npz"
V256_META = BASELINES / "v256_raw_physio_cnn_fusion_20260702" / "tables" / "v256_per_sample_prediction_metrics.csv"
V256_GUARDRAIL = BASELINES / "v256_raw_physio_cnn_fusion_20260702" / "logs" / "guardrail_check.json"
V266_EVENTS = BASELINES / "v266_vehicle_matched_bio_residual_prototype_20260702" / "tables" / "v266_event_context_table.csv"
V269_SCRIPT = BASELINES / "scripts" / "stage03_v269_reliable_identity_removed_physio_20260702.py"

SEED = 27102
K_VALUES = [3, 5, 10, 20, 40]
FIXED_WAIT_LATEST_BADTOP10 = 0.695048


def import_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少前序脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


V269 = import_module_from_path("stage03_v269_for_v271", V269_SCRIPT)
V266 = V269.V266
V267 = V269.V267


def ensure_dirs() -> None:
    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw_delay0_for_events(events: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    从 v256 raw cache 取每个 event 的 0ms 生理序列。

    v256 的 row_index 与缓存第一维对齐；metrics 表按模型/协议重复，
    因此这里先按 row_index 去重，只保留 delay_ms=0。
    """
    cache = np.load(V256_SEQ, allow_pickle=False)
    seq = cache["physio_seq"].astype(np.float32)
    ok = cache["physio_ok"].astype(np.float32)
    signals = [str(x) for x in cache["signals"]]
    meta = pd.read_csv(V256_META, encoding="utf-8-sig", usecols=["row_index", "event_uid", "delay_ms"], low_memory=False)
    meta = meta.drop_duplicates("row_index").copy()
    meta["delay_ms"] = pd.to_numeric(meta["delay_ms"], errors="coerce")
    meta0 = meta[meta["delay_ms"].eq(0)].drop_duplicates("event_uid", keep="first").set_index("event_uid")
    missing = [uid for uid in events["event_uid"].astype(str).tolist() if uid not in meta0.index]
    if missing:
        raise RuntimeError(f"v256 raw cache 缺少 {len(missing)} 个 event 的 0ms 序列，例如：{missing[:3]}")
    row_index = meta0.loc[events["event_uid"].astype(str), "row_index"].to_numpy(dtype=int)
    audit = pd.DataFrame(
        {
            "event_uid": events["event_uid"].astype(str).to_numpy(),
            "row_index_v256": row_index,
            "physio_ok": ok[row_index],
        }
    )
    return seq[row_index].astype(np.float32), ok[row_index].astype(np.float32), signals, audit


def standardize_matrix_by_train(x: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    """只用 train split 拟合均值/方差，应用到全部 split。"""
    x = np.asarray(x, dtype=float)
    train_x = x[train_mask]
    med = np.nanmedian(train_x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isfinite(x), x, med[None, :])
    mean = np.nanmean(filled[train_mask], axis=0)
    std = np.nanstd(filled[train_mask], axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    z = (filled - mean[None, :]) / std[None, :]
    audit = pd.DataFrame({"feature_i": np.arange(x.shape[1]), "train_mean": mean, "train_std": std})
    return z.astype(np.float32), audit


def build_raw_summary_features(seq: np.ndarray, signals: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """从 raw z 序列提取透明的时域/频域摘要特征。"""
    rows: Dict[str, np.ndarray] = {}
    n, c, t = seq.shape
    time = np.linspace(-20.0, 0.0, t, dtype=np.float32)
    time_center = time - time.mean()
    denom = float(np.sum(time_center * time_center))
    diff = np.diff(seq, axis=2)
    for ci, sig in enumerate(signals):
        x = seq[:, ci, :]
        prefix = f"raw_{sig}"
        rows[f"{prefix}_mean"] = x.mean(axis=1)
        rows[f"{prefix}_std"] = x.std(axis=1)
        rows[f"{prefix}_p10"] = np.quantile(x, 0.10, axis=1)
        rows[f"{prefix}_p50"] = np.quantile(x, 0.50, axis=1)
        rows[f"{prefix}_p90"] = np.quantile(x, 0.90, axis=1)
        rows[f"{prefix}_range"] = rows[f"{prefix}_p90"] - rows[f"{prefix}_p10"]
        rows[f"{prefix}_abs_mean"] = np.mean(np.abs(x), axis=1)
        rows[f"{prefix}_energy"] = np.mean(x * x, axis=1)
        rows[f"{prefix}_first2s_mean"] = x[:, :40].mean(axis=1)
        rows[f"{prefix}_last2s_mean"] = x[:, -40:].mean(axis=1)
        rows[f"{prefix}_last_minus_first2s"] = rows[f"{prefix}_last2s_mean"] - rows[f"{prefix}_first2s_mean"]
        rows[f"{prefix}_last5_minus_pre20_10"] = x[:, -100:].mean(axis=1) - x[:, :200].mean(axis=1)
        rows[f"{prefix}_diff_abs_mean"] = np.mean(np.abs(diff[:, ci, :]), axis=1)
        rows[f"{prefix}_diff_std"] = np.std(diff[:, ci, :], axis=1)
        rows[f"{prefix}_slope"] = (x @ time_center) / max(denom, 1e-6)

        mag = np.abs(np.fft.rfft(x, axis=1)) / float(t)
        for fi in range(1, min(16, mag.shape[1])):
            rows[f"{prefix}_fft_mag_{fi:02d}"] = mag[:, fi]
    df = pd.DataFrame(rows)
    return df, df.columns.tolist()


def robust_group_z_features(
    base: pd.DataFrame,
    events: pd.DataFrame,
    cols: List[str],
    group_col: str,
    prefix: str,
    train_mask: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    用 subject / recording 内部无标签分布做 robust z 校准。

    这里只用输入生理特征本身计算 group median / MAD，不看轨迹标签。
    对 val/test 来说，这是 calibrated / transductive setting；报告中会明确标注。
    """
    x = base[cols].to_numpy(dtype=float)
    train_x = x[train_mask]
    global_med = np.nanmedian(train_x, axis=0)
    global_med = np.where(np.isfinite(global_med), global_med, 0.0)
    global_mad = 1.4826 * np.nanmedian(np.abs(train_x - global_med[None, :]), axis=0)
    global_std = np.nanstd(np.where(np.isfinite(train_x), train_x, global_med[None, :]), axis=0)
    global_scale = np.where(np.isfinite(global_mad) & (global_mad > 1e-6), global_mad, global_std)
    global_scale = np.where(np.isfinite(global_scale) & (global_scale > 1e-6), global_scale, 1.0)

    z = np.zeros_like(x, dtype=np.float32)
    audit_rows: List[Dict[str, object]] = []
    groups = events[group_col].astype(str).to_numpy()
    for group in pd.unique(groups):
        idx = np.flatnonzero(groups == group)
        gx = x[idx]
        med = np.nanmedian(gx, axis=0)
        med = np.where(np.isfinite(med), med, global_med)
        mad = 1.4826 * np.nanmedian(np.abs(gx - med[None, :]), axis=0)
        scale = np.where(np.isfinite(mad) & (mad > 1e-6), mad, global_scale)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, global_scale)
        z[idx] = ((np.where(np.isfinite(gx), gx, med[None, :]) - med[None, :]) / scale[None, :]).astype(np.float32)
        audit_rows.append(
            {
                "calibration_group_col": group_col,
                "calibration_group": group,
                "event_n": int(len(idx)),
                "scale_fallback_rate": float(np.mean(~(np.isfinite(mad) & (mad > 1e-6)))),
            }
        )
    out_cols = [f"{prefix}_{col}" for col in cols]
    return pd.DataFrame(z, columns=out_cols), pd.DataFrame(audit_rows)


def center_seq_by_group(seq: np.ndarray, events: pd.DataFrame, group_col: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """把每个事件的 raw waveform 减去同 subject / recording 的典型波形。"""
    centered = np.zeros_like(seq, dtype=np.float32)
    audit_rows: List[Dict[str, object]] = []
    groups = events[group_col].astype(str).to_numpy()
    for group in pd.unique(groups):
        idx = np.flatnonzero(groups == group)
        median_wave = np.nanmedian(seq[idx], axis=0)
        median_wave = np.where(np.isfinite(median_wave), median_wave, 0.0).astype(np.float32)
        centered[idx] = seq[idx] - median_wave[None, :, :]
        audit_rows.append(
            {
                "calibration_group_col": group_col,
                "calibration_group": group,
                "event_n": int(len(idx)),
                "center_abs_mean": float(np.mean(np.abs(median_wave))),
            }
        )
    return centered, pd.DataFrame(audit_rows)


def build_centered_pca_features(
    seq: np.ndarray,
    events: pd.DataFrame,
    group_col: str,
    prefix: str,
    train_mask: np.ndarray,
    n_pca: int = 48,
    n_diff_pca: int = 24,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对 group-centered raw 序列和一阶差分序列分别做 train-only PCA。"""
    centered, center_audit = center_seq_by_group(seq, events, group_col)
    flat = centered.reshape(centered.shape[0], -1)
    flat_z, flat_audit = standardize_matrix_by_train(flat, train_mask)
    pca = PCA(n_components=n_pca, svd_solver="randomized", random_state=SEED)
    pca.fit(flat_z[train_mask])
    comp = pca.transform(flat_z).astype(np.float32)

    dflat = np.diff(centered, axis=2).reshape(centered.shape[0], -1)
    dflat_z, dflat_audit = standardize_matrix_by_train(dflat, train_mask)
    dpca = PCA(n_components=n_diff_pca, svd_solver="randomized", random_state=SEED + 1)
    dpca.fit(dflat_z[train_mask])
    dcomp = dpca.transform(dflat_z).astype(np.float32)

    rows: Dict[str, np.ndarray] = {}
    for i in range(comp.shape[1]):
        rows[f"{prefix}_pca_{i:02d}"] = comp[:, i]
    for i in range(dcomp.shape[1]):
        rows[f"{prefix}_diff_pca_{i:02d}"] = dcomp[:, i]
    audit = pd.concat(
        [
            pd.DataFrame({"block": f"{prefix}_pca", "component": np.arange(n_pca), "explained_variance_ratio": pca.explained_variance_ratio_}),
            pd.DataFrame({"block": f"{prefix}_diff_pca", "component": np.arange(n_diff_pca), "explained_variance_ratio": dpca.explained_variance_ratio_}),
        ],
        ignore_index=True,
    )
    flat_audit.assign(block=f"{prefix}_flat").to_csv(TABLES / f"v271_{prefix}_flat_scaler_audit.csv", index=False, encoding="utf-8-sig")
    dflat_audit.assign(block=f"{prefix}_diff_flat").to_csv(TABLES / f"v271_{prefix}_diff_scaler_audit.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows), audit, center_audit


def screen_raw_features(events: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """只用 train split 对 raw latent 做身份/行为可识别性筛选。"""
    train = events["split"].astype(str).eq("train").to_numpy()
    labels = {
        "subject": events.loc[train, "subject"].astype(str).to_numpy(),
        "recording": events.loc[train, "recording"].astype(str).to_numpy(),
        "bad_top10": events.loc[train, "bad_top10"].astype(str).to_numpy(),
        "early_best_after_400": events.loc[train, "early_best_after_400"].astype(str).to_numpy(),
        "wait_better_latest_vs_keep0": events.loc[train, "wait_better_latest_vs_keep0"].astype(str).to_numpy(),
    }
    rows: List[Dict[str, object]] = []
    for col in cols:
        x_all = pd.to_numeric(events[col], errors="coerce").to_numpy(dtype=float)
        x = x_all[train]
        finite = np.isfinite(x)
        std = float(np.nanstd(x)) if finite.any() else 0.0
        eta_subject = V269.eta_squared(x, labels["subject"])
        eta_recording = V269.eta_squared(x, labels["recording"])
        eta_bad = V269.eta_squared(x, labels["bad_top10"])
        eta_early = V269.eta_squared(x, labels["early_best_after_400"])
        eta_wait = V269.eta_squared(x, labels["wait_better_latest_vs_keep0"])
        identity_eta = max(eta_subject, eta_recording)
        behavior_eta = max(eta_bad, eta_early, eta_wait)
        missing = 1.0 - float(finite.mean())
        score = (behavior_eta + 0.002) / (identity_eta + 0.02) - 0.2 * missing
        rows.append(
            {
                "feature": col,
                "block": raw_block(col),
                "finite_rate_train": float(finite.mean()),
                "std_train": std,
                "eta_subject_train": eta_subject,
                "eta_recording_train": eta_recording,
                "eta_bad_top10_train": eta_bad,
                "eta_early_best_after_400_train": eta_early,
                "eta_wait_better_train": eta_wait,
                "identity_eta_max_train": identity_eta,
                "behavior_eta_max_train": behavior_eta,
                "identity_to_behavior_ratio_train": identity_eta / max(behavior_eta, 1e-6),
                "selection_score": score,
                "reliable": bool(finite.mean() >= 0.85 and std > 1e-9),
            }
        )
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False)


def raw_block(col: str) -> str:
    if col.startswith("subj_z_"):
        return "subject_summary"
    if col.startswith("rec_z_"):
        return "recording_summary"
    if col.startswith("subjseq_pca_"):
        return "subject_seq_pca"
    if col.startswith("subjseq_diff_pca_"):
        return "subject_seq_diff_pca"
    if col.startswith("recseq_pca_"):
        return "recording_seq_pca"
    if col.startswith("recseq_diff_pca_"):
        return "recording_seq_diff_pca"
    if "_fft_mag_" in col:
        return "fft"
    if col == "raw_physio_ok":
        return "quality"
    return "summary"


def choose_raw_feature_sets(
    screen: pd.DataFrame,
    subject_summary_cols: List[str],
    recording_summary_cols: List[str],
    subject_pca_cols: List[str],
    recording_pca_cols: List[str],
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    reliable = screen[screen["reliable"].astype(bool)].copy()
    reliable_features = set(reliable["feature"].astype(str))

    def top_scored(candidates: List[str], n: int) -> List[str]:
        sub = reliable[reliable["feature"].isin(candidates)].sort_values("selection_score", ascending=False)
        return sub["feature"].head(n).tolist()

    def keep_order(candidates: List[str], n: int) -> List[str]:
        return [c for c in candidates if c in reliable_features][:n]

    sets: Dict[str, List[str]] = {
        "subject_summary64": top_scored(subject_summary_cols, 64),
        "recording_summary64": top_scored(recording_summary_cols, 64),
        "subject_seq_pca72": keep_order(subject_pca_cols, 72),
        "recording_seq_pca72": keep_order(recording_pca_cols, 72),
        "calibrated_screened64": reliable.sort_values("selection_score", ascending=False)["feature"].head(64).tolist(),
        "calibrated_low_identity48": reliable.sort_values(["identity_eta_max_train", "selection_score"], ascending=[True, False])["feature"].head(48).tolist(),
    }
    rows: List[Dict[str, object]] = []
    for name, cols in sets.items():
        sub = screen[screen["feature"].isin(cols)].copy()
        rows.append(
            {
                "raw_set": name,
                "feature_n": int(len(cols)),
                "subject_summary_n": int(sub["block"].eq("subject_summary").sum()),
                "recording_summary_n": int(sub["block"].eq("recording_summary").sum()),
                "subject_seq_pca_n": int(sub["block"].eq("subject_seq_pca").sum()),
                "subject_seq_diff_pca_n": int(sub["block"].eq("subject_seq_diff_pca").sum()),
                "recording_seq_pca_n": int(sub["block"].eq("recording_seq_pca").sum()),
                "recording_seq_diff_pca_n": int(sub["block"].eq("recording_seq_diff_pca").sum()),
                "behavior_eta_max_mean": float(sub["behavior_eta_max_train"].mean()) if len(sub) else math.nan,
                "identity_eta_max_mean": float(sub["identity_eta_max_train"].mean()) if len(sub) else math.nan,
                "identity_to_behavior_ratio_median": float(sub["identity_to_behavior_ratio_train"].median()) if len(sub) else math.nan,
                "features": ";".join(cols),
            }
        )
    return sets, pd.DataFrame(rows)


def build_wait_summary(events: pd.DataFrame, veh_cols: List[str], feature_sets: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """训练 raw latent wait gate，并汇总逐事件选择结果。"""
    out = events.copy()
    out["target_gain_latest_vs_keep0"] = (
        pd.to_numeric(out["keep0_tail_rmse_v241"], errors="coerce") - pd.to_numeric(out["latest_tail_rmse_v241"], errors="coerce")
    )
    selected_parts = [
        V269.build_wait_selected(out, "policy_keep_0ms_anchor", "baseline"),
        V269.build_wait_selected(out, "policy_wait_to_latest_anchor", "baseline"),
        V269.build_wait_selected(out, "oracle_best_anchor_upper_bound", "oracle"),
    ]
    audit_rows: List[Dict[str, object]] = []
    for raw_set, cols in feature_sets.items():
        for model_name, family, use_cols, bad_weight in [
            (f"wait_raw_{raw_set}_gain", "raw_bio", cols, False),
            (f"wait_vehicle_raw_{raw_set}_gain", "vehicle_raw", veh_cols + cols, False),
            (f"wait_vehicle_raw_{raw_set}_gain_badweighted", "vehicle_raw", veh_cols + cols, True),
        ]:
            pred, _fill = V269.fit_predict_hgb(out, use_cols, "target_gain_latest_vs_keep0", bad_weight=bad_weight)
            pred_col = f"pred_{model_name}"
            out[pred_col] = pred
            threshold, threshold_audit = V269.tune_threshold(out, pred_col, bad_weight=bad_weight)
            selected_parts.append(V269.build_wait_selected(out, model_name, family, pred_col=pred_col, threshold=threshold))
            best_val = float(threshold_audit["val_tail_rmse_weighted"].iloc[0]) if len(threshold_audit) else math.nan
            audit_rows.append(
                {
                    "model_name": model_name,
                    "raw_set": raw_set,
                    "family": family,
                    "feature_n": int(len(use_cols)),
                    "bad_weight": bool(bad_weight),
                    "threshold": float(threshold),
                    "best_val_tail_rmse_weighted": best_val,
                }
            )
    selected = pd.concat(selected_parts, ignore_index=True)
    return selected, V266.summarize_selected(selected).merge(pd.DataFrame(audit_rows), left_on="strategy", right_on="model_name", how="left")


def run_pair_rerank(
    events: pd.DataFrame,
    cand: pd.DataFrame,
    veh_cols: List[str],
    feature_sets: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对每个 raw latent set 运行 v267 式 pair reranker。"""
    lookup = V266.candidate_rmse_lookup(cand)
    train_mask = events["split"].astype(str).eq("train").to_numpy()
    veh_z, _, _, _ = V266.fit_fill_scale(events[veh_cols].to_numpy(dtype=float), train_mask)
    selected_parts: List[pd.DataFrame] = []
    summary_parts: List[pd.DataFrame] = []
    chosen_parts: List[pd.DataFrame] = []
    pair_parts: List[pd.DataFrame] = []
    for raw_set, cols in feature_sets.items():
        if not cols:
            continue
        print(f"[v271] pair rerank raw_set={raw_set} feature_n={len(cols)}", flush=True)
        bio_z, _, _, _ = V266.fit_fill_scale(events[cols].to_numpy(dtype=float), train_mask)
        neighbors = V266.build_neighbor_table(events, veh_z, bio_z, train_mask, max_k=max(K_VALUES))
        pair_meta, matrices, names = V267.build_pair_dataset(events, neighbors, lookup, veh_z, bio_z, max_k=max(K_VALUES))
        pair_pred, _fill_audit, feature_block = V267.add_pair_predictions(pair_meta, matrices, names)
        selected = V267.build_selected(events, pair_pred, lookup)
        summary = V267.summarize_selected(selected)
        chosen = V267.choose_val_strategies(summary)
        for df in (selected, summary, chosen, feature_block):
            df["raw_set"] = raw_set
            df["raw_feature_n"] = int(len(cols))
        selected_parts.append(selected)
        summary_parts.append(summary)
        chosen_parts.append(chosen)
        pred_cols = [c for c in pair_pred.columns if c.startswith("pred_pair_")]
        compact_cols = [
            "event_uid",
            "split",
            "subject",
            "prototype_event_uid",
            "prototype_subject",
            "neighbor_rank_vehicle",
            "prototype_oracle_delay_ms",
            "mapped_delay_ms",
            "target_tail_rmse_v241",
            "vehicle_distance",
            "bio_distance",
            "bad_top10",
        ] + pred_cols
        compact = pair_pred[[c for c in compact_cols if c in pair_pred.columns]].copy()
        compact.insert(0, "raw_set", raw_set)
        pair_parts.append(compact)
    return (
        pd.concat(selected_parts, ignore_index=True),
        pd.concat(summary_parts, ignore_index=True),
        pd.concat(chosen_parts, ignore_index=True),
        pd.concat(pair_parts, ignore_index=True),
    )


def choose_cross_set_pair(chosen: pd.DataFrame) -> pd.DataFrame:
    """跨 raw_set 按 val bad_top10 选择 vehicle+raw 策略，并映射到 test。"""
    if chosen.empty:
        return chosen
    val = chosen[
        chosen["chosen_label"].eq("val_best_pair_vehicle_bio")
        & chosen["split"].eq("val")
        & chosen["event_group"].eq("bad_top10")
    ].copy()
    if val.empty:
        return pd.DataFrame()
    best = val.sort_values(["selected_tail_rmse_mean", "selected_delay_ms_mean", "raw_set"], ascending=[True, True, True]).iloc[0]
    raw_set = str(best["raw_set"])
    strategy = str(best["chosen_strategy"])
    mapped = chosen[
        chosen["raw_set"].astype(str).eq(raw_set)
        & chosen["chosen_strategy"].astype(str).eq(strategy)
        & chosen["chosen_label"].eq("val_best_pair_vehicle_bio")
        & chosen["event_group"].eq("bad_top10")
        & chosen["split"].isin(["val", "test"])
    ].copy()
    return mapped


def build_decision(wait_summary: pd.DataFrame, pair_summary: pd.DataFrame, pair_chosen: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    wait_bad = wait_summary[wait_summary["split"].eq("test") & wait_summary["event_group"].eq("bad_top10")].copy()
    pair_bad = pair_summary[pair_summary["split"].eq("test") & pair_summary["event_group"].eq("bad_top10")].copy()
    for strategy in ["policy_keep_0ms_anchor", "policy_wait_to_latest_anchor", "oracle_best_anchor_upper_bound"]:
        src = wait_bad[wait_bad["strategy"].eq(strategy)]
        if len(src):
            rows.append({"source": "baseline", "label": strategy, "rmse": float(src["selected_tail_rmse_mean"].iloc[0])})
    learned_wait = wait_bad[
        wait_bad["deployable"].astype(bool)
        & ~wait_bad["strategy_family"].isin(["baseline", "oracle", "candidate_oracle"])
    ].sort_values("selected_tail_rmse_mean")
    if len(learned_wait):
        row = learned_wait.iloc[0]
        rows.append({"source": "wait_test_best", "label": str(row["strategy"]), "rmse": float(row["selected_tail_rmse_mean"])})
    oracle = pair_bad[pair_bad["strategy_family"].eq("candidate_oracle")].sort_values("selected_tail_rmse_mean")
    if len(oracle):
        row = oracle.iloc[0]
        rows.append({"source": "pair_candidate_oracle", "label": f"{row['raw_set']}:{row['strategy']}", "rmse": float(row["selected_tail_rmse_mean"])})
    deploy = pair_bad[
        pair_bad["deployable"].astype(bool)
        & ~pair_bad["strategy_family"].isin(["baseline", "oracle", "candidate_oracle"])
    ].sort_values("selected_tail_rmse_mean")
    if len(deploy):
        row = deploy.iloc[0]
        rows.append({"source": "pair_test_best_deployable", "label": f"{row['raw_set']}:{row['strategy']}", "rmse": float(row["selected_tail_rmse_mean"])})
    cross = choose_cross_set_pair(pair_chosen)
    cross_test = cross[cross["split"].eq("test")]
    if len(cross_test):
        row = cross_test.iloc[0]
        rows.append({"source": "pair_val_best_vehicle_raw", "label": f"{row['raw_set']}:{row['chosen_strategy']}", "rmse": float(row["selected_tail_rmse_mean"])})
    out = pd.DataFrame(rows)
    if len(out):
        out["delta_vs_fixed_latest"] = out["rmse"] - FIXED_WAIT_LATEST_BADTOP10
        out["passes_fixed_latest"] = out["rmse"] < FIXED_WAIT_LATEST_BADTOP10
    return out


def plot_decision(decision: pd.DataFrame) -> Path:
    path = FIGURES / "v271_test_badtop10_decision_summary.png"
    if decision.empty:
        return path
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    x = np.arange(len(decision))
    ax.bar(x, decision["rmse"], color="#4C78A8")
    ax.axhline(FIXED_WAIT_LATEST_BADTOP10, color="#E15759", linestyle="--", linewidth=1.2, label="fixed latest")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in decision["source"]], fontsize=8)
    ax.set_ylabel("test bad_top10 tail RMSE")
    ax.set_title("v271: calibrated raw physiology state decision summary")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(feature_audit: pd.DataFrame, decision: pd.DataFrame, wait_summary: pd.DataFrame, pair_summary: pd.DataFrame, pair_chosen: pd.DataFrame, figs: List[Path]) -> None:
    lines: List[str] = []
    lines.append("# v271 calibrated raw physiology state")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- v270 的 raw waveform latent 仍未突破 fixed wait-latest。")
    lines.append("- v271 把 raw 生理从绝对值改为相对 subject / recording 无标签基线的状态变化。")
    lines.append("- 这是 calibrated / transductive setting，不是纯 cold-start subject-disjoint；只用于判断个体基线是否能释放生理信息。")
    lines.append("")
    lines.append("## 特征集")
    lines.append("")
    display_cols = [
        "raw_set",
        "feature_n",
        "subject_summary_n",
        "recording_summary_n",
        "subject_seq_pca_n",
        "subject_seq_diff_pca_n",
        "recording_seq_pca_n",
        "recording_seq_diff_pca_n",
        "behavior_eta_max_mean",
        "identity_eta_max_mean",
        "identity_to_behavior_ratio_median",
    ]
    lines.append(feature_audit[[c for c in display_cols if c in feature_audit.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## test bad_top10 决策收口")
    lines.append("")
    lines.append(decision.to_markdown(index=False) if len(decision) else "- 无可用结果。")
    lines.append("")
    lines.append("## wait gate test bad_top10 top")
    lines.append("")
    wait_bad = wait_summary[wait_summary["split"].eq("test") & wait_summary["event_group"].eq("bad_top10")].sort_values("selected_tail_rmse_mean").head(12)
    lines.append(wait_bad[["strategy", "strategy_family", "selected_tail_rmse_mean", "delta_selected_minus_latest_mean", "selected_latest_rate"]].to_markdown(index=False))
    lines.append("")
    lines.append("## pair reranker test bad_top10 top")
    lines.append("")
    pair_bad = pair_summary[pair_summary["split"].eq("test") & pair_summary["event_group"].eq("bad_top10")].sort_values("selected_tail_rmse_mean").head(18)
    lines.append(pair_bad[["raw_set", "strategy", "strategy_family", "selected_tail_rmse_mean", "delta_selected_minus_latest_mean", "selected_delay_ms_mean", "selected_latest_rate"]].to_markdown(index=False))
    lines.append("")
    lines.append("## val 选择 vehicle+raw 策略")
    lines.append("")
    cross = choose_cross_set_pair(pair_chosen)
    lines.append(cross.to_markdown(index=False) if len(cross) else "- 无 val-best vehicle+raw 策略。")
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    deploy = decision[decision["source"].isin(["wait_test_best", "pair_test_best_deployable", "pair_val_best_vehicle_raw"])] if len(decision) else pd.DataFrame()
    if len(deploy) and bool(deploy["passes_fixed_latest"].astype(bool).any()):
        lines.append("- 至少一个 calibrated raw 生理可部署策略低于 fixed wait-latest，说明个体/recording 基线校准可能释放了可用状态信息。")
    else:
        lines.append("- 当前 calibrated raw 生理可部署策略仍未低于 fixed wait-latest，不能称为差样本本质改善。")
    if len(deploy):
        best = deploy.sort_values("rmse").iloc[0]
        lines.append(f"- 最好可部署策略 `{best['label']}` 的 test bad_top10 RMSE 为 `{float(best['rmse']):.4f}`。")
    lines.append("- 若 calibrated setting 仍失败，说明当前生理即使有个体基线也难以支撑该预测任务；应转回车辆多未来/不确定性主线。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figs:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v271_calibrated_raw_physio_state_cn.md").write_text("\n".join(lines), encoding="utf-8")


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v256_seq", V256_SEQ),
        ("v256_meta", V256_META),
        ("v256_guardrail", V256_GUARDRAIL),
        ("v266_events", V266_EVENTS),
        ("v269_script", V269_SCRIPT),
    ]:
        rows.append({"label": label, "path": str(path), "exists": path.exists(), "sha256": file_sha256(path) if path.exists() else ""})
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    if OUT.exists():
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                rows.append({"relative_path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip() is None


def main() -> None:
    print("[v271] calibrated raw physiology state", flush=True)
    clean_out_dir()
    cand, events, _merge_audit, veh_cols, _old_bio_cols = V266.load_candidate_and_events()
    events = events.copy()
    events["wait_better_latest_vs_keep0"] = (
        pd.to_numeric(events["latest_tail_rmse_v241"], errors="coerce")
        < pd.to_numeric(events["keep0_tail_rmse_v241"], errors="coerce")
    )
    train_mask = events["split"].astype(str).eq("train").to_numpy()

    seq0, ok0, signals, raw_audit = load_raw_delay0_for_events(events)
    raw_summary, summary_cols = build_raw_summary_features(seq0, signals)
    subject_summary, subject_summary_audit = robust_group_z_features(raw_summary, events, summary_cols, "subject", "subj_z", train_mask)
    recording_summary, recording_summary_audit = robust_group_z_features(raw_summary, events, summary_cols, "recording", "rec_z", train_mask)
    subject_pca, subject_pca_audit, subject_center_audit = build_centered_pca_features(seq0, events, "subject", "subjseq", train_mask)
    recording_pca, recording_pca_audit, recording_center_audit = build_centered_pca_features(seq0, events, "recording", "recseq", train_mask)

    raw_quality = pd.DataFrame({"raw_physio_ok": ok0})
    events = pd.concat(
        [
            events.reset_index(drop=True),
            raw_quality.reset_index(drop=True),
            subject_summary.reset_index(drop=True),
            recording_summary.reset_index(drop=True),
            subject_pca.reset_index(drop=True),
            recording_pca.reset_index(drop=True),
        ],
        axis=1,
    )
    raw_cols = (
        subject_summary.columns.tolist()
        + recording_summary.columns.tolist()
        + subject_pca.columns.tolist()
        + recording_pca.columns.tolist()
        + ["raw_physio_ok"]
    )
    screen = screen_raw_features(events, raw_cols)
    feature_sets, feature_audit = choose_raw_feature_sets(
        screen,
        subject_summary.columns.tolist(),
        recording_summary.columns.tolist(),
        subject_pca.columns.tolist(),
        recording_pca.columns.tolist(),
    )

    write_csv(raw_audit, TABLES / "v271_raw_delay0_alignment_audit.csv")
    write_csv(pd.concat([subject_pca_audit, recording_pca_audit], ignore_index=True), TABLES / "v271_centered_raw_pca_audit.csv")
    write_csv(pd.concat([subject_summary_audit, recording_summary_audit], ignore_index=True), TABLES / "v271_summary_calibration_audit.csv")
    write_csv(pd.concat([subject_center_audit, recording_center_audit], ignore_index=True), TABLES / "v271_sequence_centering_audit.csv")
    write_csv(screen, TABLES / "v271_raw_feature_screening_train_only.csv")
    write_csv(feature_audit, TABLES / "v271_raw_feature_set_audit.csv")
    write_csv(events[["event_uid", "split", "subject", "recording"] + veh_cols + raw_cols], TABLES / "v271_event_context_table.csv")

    wait_selected, wait_summary = build_wait_summary(events, veh_cols, feature_sets)
    pair_selected, pair_summary, pair_chosen, pair_compact = run_pair_rerank(events, cand, veh_cols, feature_sets)
    decision = build_decision(wait_summary, pair_summary, pair_chosen)
    fig = plot_decision(decision)

    write_csv(wait_selected, TABLES / "v271_wait_selected_by_strategy.csv")
    write_csv(wait_summary, TABLES / "v271_wait_summary.csv")
    write_csv(pair_selected, TABLES / "v271_pair_selected_by_strategy.csv")
    write_csv(pair_summary, TABLES / "v271_pair_reranker_summary.csv")
    write_csv(pair_chosen, TABLES / "v271_pair_val_chosen_summary.csv")
    write_csv(pair_compact, TABLES / "v271_pair_predictions_compact.csv")
    write_csv(decision, TABLES / "v271_decision_summary.csv")

    write_input_hashes()
    write_file_inventory()
    write_report(feature_audit, decision, wait_summary, pair_summary, pair_chosen, [fig])
    write_file_inventory()
    zip_ok = make_zip()

    v256_guard = json.loads(V256_GUARDRAIL.read_text(encoding="utf-8")) if V256_GUARDRAIL.exists() else {}
    deploy = decision[decision["source"].isin(["wait_test_best", "pair_test_best_deployable", "pair_val_best_vehicle_raw"])] if len(decision) else pd.DataFrame()
    best_rmse = float(deploy["rmse"].min()) if len(deploy) else math.nan
    guardrail = {
        "pass": bool(zip_ok and bool(v256_guard.get("pass", False))),
        "zip_testzip": bool(zip_ok),
        "v256_guardrail_pass": bool(v256_guard.get("pass", False)),
        "event_n": int(events["event_uid"].nunique()),
        "raw_sequence_shape_delay0": list(seq0.shape),
        "raw_physio_ok_rate_delay0": float(np.mean(ok0)),
        "raw_feature_n": int(len(raw_cols)),
        "raw_set_n": int(len(feature_sets)),
        "calibration_setting": "subject_recording_unsupervised_transductive",
        "calibration_group_cols": ["subject", "recording"],
        "pair_row_n": int(len(pair_compact)),
        "best_deployable_test_badtop10": best_rmse,
        "fixed_wait_latest_badtop10": float(FIXED_WAIT_LATEST_BADTOP10),
        "best_deployable_passes_fixed_latest": bool(np.isfinite(best_rmse) and best_rmse < FIXED_WAIT_LATEST_BADTOP10),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v271 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    print(f"[v271] report={REPORTS / 'v271_calibrated_raw_physio_state_cn.md'}", flush=True)
    print(f"[v271] zip={ZIP_PATH}", flush=True)
    if len(decision):
        print(decision.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
