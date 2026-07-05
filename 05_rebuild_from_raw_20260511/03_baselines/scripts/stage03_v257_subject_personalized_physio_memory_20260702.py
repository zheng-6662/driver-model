#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v257 同驾驶员生理状态记忆检索实验。

目的：
- v254b-v256 说明当前生理数据不适合作为 subject-disjoint 直接泛化特征；
- 但 subject-aware 的 bad_top10 诊断出现过弱信号；
- 本轮验证一个更合理的生理用法：同一驾驶员有历史样本时，用车辆相似 + 生理状态相似检索该驾驶员历史未来原型。

边界：
- 只在 subject-aware 诊断口径下成立；
- query 的未来曲线不作为输入；
- 同一 recording 内只允许使用 observation_s 更早的训练事件，避免在线时序泄漏；
- 用 validation 选择检索策略，test 只报告。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V252_SCRIPT = BASELINES / "scripts" / "stage03_v252_input_similarity_future_divergence_20260701.py"
V254B_SCRIPT = BASELINES / "scripts" / "stage03_v254b_physio_200hz_event_representation_20260702.py"
V254B_FEATURES = (
    BASELINES
    / "v254b_physio_200hz_event_representation_20260702"
    / "tables"
    / "v254b_event_physio200_features.csv"
)
V256_SEQ = (
    BASELINES
    / "v256_raw_physio_cnn_fusion_20260702"
    / "tensors"
    / "v256_physio_seq_20s_20hz.npz"
)

OUT = BASELINES / "v257_subject_personalized_physio_memory_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v257_subject_personalized_physio_memory_20260702_pack.zip"

SEED = 25702
RAW_PCA_DIM = 32

CONFIGS = [
    {"strategy": "same_subject_vehicle_k1", "k": 1, "w_vehicle": 1.0, "w_physio_stats": 0.0, "w_raw_seq": 0.0},
    {"strategy": "same_subject_vehicle_k3", "k": 3, "w_vehicle": 1.0, "w_physio_stats": 0.0, "w_raw_seq": 0.0},
    {"strategy": "same_subject_physio_stats_k1", "k": 1, "w_vehicle": 0.0, "w_physio_stats": 1.0, "w_raw_seq": 0.0},
    {"strategy": "same_subject_raw_seq_k1", "k": 1, "w_vehicle": 0.0, "w_physio_stats": 0.0, "w_raw_seq": 1.0},
    {"strategy": "same_subject_vehicle_physio25_k3", "k": 3, "w_vehicle": 1.0, "w_physio_stats": 0.25, "w_raw_seq": 0.0},
    {"strategy": "same_subject_vehicle_raw25_k3", "k": 3, "w_vehicle": 1.0, "w_physio_stats": 0.0, "w_raw_seq": 0.25},
    {"strategy": "same_subject_vehicle_physio_raw15_k3", "k": 3, "w_vehicle": 1.0, "w_physio_stats": 0.15, "w_raw_seq": 0.15},
    {"strategy": "same_subject_vehicle_physio50_raw25_k5", "k": 5, "w_vehicle": 1.0, "w_physio_stats": 0.50, "w_raw_seq": 0.25},
]

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V252 = import_module_from_path("stage03_v252_for_v257", V252_SCRIPT)
V254B = import_module_from_path("stage03_v254b_for_v257", V254B_SCRIPT)


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


def finite_nanmedian(x: np.ndarray, axis: int = 0) -> np.ndarray:
    with np.errstate(all="ignore"):
        med = np.nanmedian(x, axis=axis)
    med = np.asarray(med, dtype=float)
    med[~np.isfinite(med)] = 0.0
    return med


def standardize_by_train(x: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    x = np.asarray(x, dtype=float)
    train_x = x[train_mask]
    med = finite_nanmedian(train_x, axis=0)
    filled = np.where(np.isfinite(x), x, med[None, :])
    mean = np.nanmean(filled[train_mask], axis=0)
    std = np.nanstd(filled[train_mask], axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    z = (filled - mean[None, :]) / std[None, :]
    audit = pd.DataFrame({"feature_i": np.arange(x.shape[1]), "train_mean": mean, "train_std": std})
    return z.astype(np.float32), audit


def load_physio_stats(manifest: pd.DataFrame, train_mask: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if not V254B_FEATURES.exists():
        raise FileNotFoundError(f"缺少 v254b 生理特征表：{V254B_FEATURES}")
    physio = pd.read_csv(V254B_FEATURES, encoding="utf-8-sig")
    if len(physio) != len(manifest):
        raise AssertionError("v254b 生理特征行数与 manifest 不一致")
    numeric_cols = [
        c
        for c in physio.columns
        if c.startswith("physio200_")
        and pd.api.types.is_numeric_dtype(physio[c])
        and ("_z_" in c or c.endswith("_index") or "burst_rate" in c)
    ]
    curated_cols = [
        c
        for c in numeric_cols
        if any(sig in c for sig in ["HR_bpm", "EMG_RMS", "EMG_filt200", "EDA_Phasic", "EDA_Tonic", "RESP_filt200", "ECG_filt200"])
    ]
    x, audit = standardize_by_train(physio[curated_cols].to_numpy(dtype=float), train_mask)
    audit["feature"] = curated_cols
    return x, physio, audit


def load_raw_seq_pca(train_mask: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    if not V256_SEQ.exists():
        raise FileNotFoundError(f"缺少 v256 raw 生理序列缓存：{V256_SEQ}")
    cache = np.load(V256_SEQ)
    seq = cache["physio_seq"].astype(np.float32)
    n = seq.shape[0]
    flat = seq.reshape(n, -1).astype(np.float32)
    flat_z, scaler_audit = standardize_by_train(flat, train_mask)
    n_components = min(RAW_PCA_DIM, flat_z.shape[1], int(train_mask.sum()) - 1)
    pca = PCA(n_components=n_components, random_state=SEED, svd_solver="randomized")
    pca.fit(flat_z[train_mask])
    emb = pca.transform(flat_z).astype(np.float32)
    audit = pd.DataFrame(
        {
            "component": np.arange(n_components),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance_ratio_cumsum": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    audit["raw_flat_dim"] = int(flat_z.shape[1])
    audit["train_n"] = int(train_mask.sum())
    return emb, audit


def build_bad_top10_by_split(sample_metrics: pd.DataFrame, split: np.ndarray) -> np.ndarray:
    tail = pd.to_numeric(sample_metrics["tail_rmse_v250"], errors="coerce").to_numpy(dtype=float)
    bad = np.zeros(len(sample_metrics), dtype=bool)
    for split_name in ["train", "val", "test"]:
        mask = split == split_name
        vals = tail[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        q90 = float(np.quantile(vals, 0.90))
        bad[mask] = tail[mask] >= q90
    return bad


def l2_mean(x: np.ndarray, qi: int, candidates: np.ndarray) -> np.ndarray:
    if candidates.size == 0 or x.shape[1] == 0:
        return np.full(candidates.size, np.nan, dtype=float)
    diff = x[candidates] - x[qi][None, :]
    return np.sqrt(np.nanmean(np.square(diff), axis=1)).astype(float)


def weighted_curve_average(curves: np.ndarray, masks: np.ndarray, weights: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if float(weights.sum()) <= 0:
        return fallback.astype(np.float32)
    numer = np.nansum(curves * masks * weights[:, None], axis=0)
    denom = np.sum(masks * weights[:, None], axis=0)
    out = np.where(denom > 0, numer / np.maximum(denom, 1e-12), fallback)
    out[~np.isfinite(out)] = fallback[~np.isfinite(out)]
    return out.astype(np.float32)


def predict_memory_for_split(
    split_name: str,
    query_idx: np.ndarray,
    train_idx_by_subject_delay: Dict[Tuple[str, int], np.ndarray],
    manifest: pd.DataFrame,
    vehicle_x: np.ndarray,
    physio_stats_x: np.ndarray,
    raw_seq_x: np.ndarray,
    y_true: np.ndarray,
    valid_mask: np.ndarray,
    fallback_pred: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    subjects = manifest["subject"].astype(str).to_numpy()
    recordings = manifest["recording"].astype(str).to_numpy()
    event_uids = manifest["event_uid"].astype(str).to_numpy()
    delays = manifest["delay_ms"].astype(int).to_numpy()
    obs = manifest["observation_s"].astype(float).to_numpy()
    pred_map = {cfg["strategy"]: np.array(fallback_pred, copy=True).astype(np.float32) for cfg in CONFIGS}
    rows: List[Dict[str, object]] = []

    for done, qi in enumerate(query_idx, start=1):
        subject = str(subjects[qi])
        delay = int(delays[qi])
        raw_candidates = train_idx_by_subject_delay.get((subject, delay), np.array([], dtype=int))
        if raw_candidates.size:
            same_recording = recordings[raw_candidates] == recordings[qi]
            earlier_in_same_recording = obs[raw_candidates] < (obs[qi] - 1e-6)
            different_recording = ~same_recording
            candidates = raw_candidates[(event_uids[raw_candidates] != event_uids[qi]) & (different_recording | earlier_in_same_recording)]
        else:
            candidates = raw_candidates
        if candidates.size == 0:
            for cfg in CONFIGS:
                rows.append(
                    {
                        "query_split": split_name,
                        "strategy": cfg["strategy"],
                        "query_row_index": int(qi),
                        "event_uid": str(event_uids[qi]),
                        "subject": subject,
                        "delay_ms": delay,
                        "candidate_n": 0,
                        "selected_neighbor_rows": "",
                        "selected_neighbor_events": "",
                        "used_fallback_v250": True,
                    }
                )
            continue

        d_vehicle = l2_mean(vehicle_x, qi, candidates)
        d_stats = l2_mean(physio_stats_x, qi, candidates)
        d_raw = l2_mean(raw_seq_x, qi, candidates)
        curves = y_true[candidates].astype(np.float32)
        masks = valid_mask[candidates].astype(float)
        fallback = fallback_pred[qi].astype(np.float32)

        for cfg in CONFIGS:
            dist = (
                float(cfg["w_vehicle"]) * d_vehicle
                + float(cfg["w_physio_stats"]) * d_stats
                + float(cfg["w_raw_seq"]) * d_raw
            )
            finite = np.isfinite(dist)
            if not finite.any():
                chosen = np.arange(min(int(cfg["k"]), len(candidates)))
                chosen_dist = np.zeros(len(chosen), dtype=float)
            else:
                valid_pos = np.where(finite)[0]
                order_valid = valid_pos[np.argsort(dist[valid_pos], kind="mergesort")]
                chosen = order_valid[: min(int(cfg["k"]), len(order_valid))]
                chosen_dist = dist[chosen]
            if len(chosen) == 0:
                pred = fallback
                chosen_rows: List[int] = []
                chosen_events: List[str] = []
            else:
                scale = float(np.median(chosen_dist[np.isfinite(chosen_dist)])) if np.isfinite(chosen_dist).any() else 1.0
                scale = max(scale, 1e-6)
                weights = np.exp(-chosen_dist / scale)
                pred = weighted_curve_average(curves[chosen], masks[chosen], weights, fallback)
                chosen_rows = [int(candidates[j]) for j in chosen]
                chosen_events = [str(event_uids[candidates[j]]) for j in chosen]
            pred_map[cfg["strategy"]][qi] = pred
            rows.append(
                {
                    "query_split": split_name,
                    "strategy": cfg["strategy"],
                    "query_row_index": int(qi),
                    "event_uid": str(event_uids[qi]),
                    "subject": subject,
                    "delay_ms": delay,
                    "candidate_n": int(len(candidates)),
                    "selected_neighbor_rows": "|".join(map(str, chosen_rows)),
                    "selected_neighbor_events": "|".join(chosen_events),
                    "used_fallback_v250": False,
                    "selected_distance_mean": float(np.mean(chosen_dist)) if len(chosen_dist) else math.nan,
                }
            )
        if done % 500 == 0:
            print(f"[v257] {split_name}: processed {done}/{len(query_idx)} queries", flush=True)

    return pred_map, pd.DataFrame(rows)


def sample_rmse(pred: np.ndarray, y: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    diff2 = np.square(pred - y)
    diff2 = np.where(valid_mask, diff2, np.nan)
    with np.errstate(all="ignore"):
        return np.sqrt(np.nanmean(diff2, axis=1))


def sample_tail_rmse(pred: np.ndarray, y: np.ndarray, valid_mask: np.ndarray, delays: np.ndarray) -> np.ndarray:
    out = np.full(len(y), np.nan, dtype=float)
    for i, delay in enumerate(delays):
        tail = V252.future_tail_mask(int(delay))
        mask = valid_mask[i] & tail
        if int(mask.sum()) < 2:
            continue
        out[i] = float(np.sqrt(np.mean(np.square(pred[i, mask] - y[i, mask]))))
    return out


def summarize_predictions(
    pred_map: Dict[str, np.ndarray],
    y_true: np.ndarray,
    valid_mask: np.ndarray,
    manifest: pd.DataFrame,
    sample_metrics: pd.DataFrame,
    split: np.ndarray,
    bad_top10: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    delays = manifest["delay_ms"].astype(int).to_numpy()
    rows = []
    sample_rows = []
    bucket_defs = [
        ("all", np.ones(len(split), dtype=bool)),
        ("bad_top10_v250", bad_top10),
        ("strong_steer", sample_metrics["is_strong_steer"].astype(bool).to_numpy()),
        ("observe_later_like", sample_metrics["is_observe_later_like"].astype(bool).to_numpy()),
    ]
    for strategy, pred in pred_map.items():
        rmse = sample_rmse(pred, y_true, valid_mask)
        tail = sample_tail_rmse(pred, y_true, valid_mask, delays)
        for eval_split in ["val", "test"]:
            split_mask = split == eval_split
            for bucket, bucket_mask in bucket_defs:
                mask = split_mask & bucket_mask
                if int(mask.sum()) == 0:
                    continue
                rows.append(
                    {
                        "protocol": "subject_aware_personalized",
                        "eval_split": eval_split,
                        "bucket": bucket,
                        "strategy": strategy,
                        "n": int(mask.sum()),
                        "sample_rmse_mean": float(np.nanmean(rmse[mask])),
                        "tail_rmse_mean": float(np.nanmean(tail[mask])),
                        "tail_rmse_median": float(np.nanmedian(tail[mask])),
                    }
                )
        for idx in np.where((split == "val") | (split == "test"))[0]:
            sample_rows.append(
                {
                    "strategy": strategy,
                    "row_index": int(idx),
                    "event_uid": str(manifest.iloc[idx]["event_uid"]),
                    "split": str(split[idx]),
                    "subject": str(manifest.iloc[idx]["subject"]),
                    "delay_ms": int(delays[idx]),
                    "sample_rmse": float(rmse[idx]),
                    "tail_rmse": float(tail[idx]),
                    "bad_top10_v250": bool(bad_top10[idx]),
                }
            )
    summary = pd.DataFrame(rows)
    base = summary[summary["strategy"].eq("v250_existing")][["eval_split", "bucket", "sample_rmse_mean", "tail_rmse_mean"]].rename(
        columns={"sample_rmse_mean": "v250_sample_rmse_mean", "tail_rmse_mean": "v250_tail_rmse_mean"}
    )
    summary = summary.merge(base, on=["eval_split", "bucket"], how="left")
    summary["delta_sample_rmse_vs_v250"] = summary["sample_rmse_mean"] - summary["v250_sample_rmse_mean"]
    summary["delta_tail_rmse_vs_v250"] = summary["tail_rmse_mean"] - summary["v250_tail_rmse_mean"]
    return summary, pd.DataFrame(sample_rows)


def select_strategy_from_val(summary: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    val = summary[summary["eval_split"].eq("val")].copy()
    all_base = float(val[(val["bucket"].eq("all")) & (val["strategy"].eq("v250_existing"))]["tail_rmse_mean"].iloc[0])
    rows = []
    for strategy, g in val.groupby("strategy"):
        if strategy == "v250_existing":
            continue
        all_row = g[g["bucket"].eq("all")]
        bad_row = g[g["bucket"].eq("bad_top10_v250")]
        if all_row.empty or bad_row.empty:
            continue
        all_tail = float(all_row["tail_rmse_mean"].iloc[0])
        bad_tail = float(bad_row["tail_rmse_mean"].iloc[0])
        all_harm = max(0.0, all_tail - all_base)
        # 选择逻辑：优先降低差样本，同时强惩罚整体伤害。
        score = bad_tail + 3.0 * all_harm
        rows.append(
            {
                "strategy": strategy,
                "val_all_tail_rmse": all_tail,
                "val_bad_top10_tail_rmse": bad_tail,
                "val_all_harm_vs_v250": all_harm,
                "selection_score": score,
            }
        )
    table = pd.DataFrame(rows).sort_values(["selection_score", "val_bad_top10_tail_rmse"], ascending=[True, True])
    chosen = str(table.iloc[0]["strategy"]) if len(table) else "v250_existing"
    table["chosen_by_validation"] = table["strategy"].eq(chosen)
    return chosen, table


def plot_summary(summary: pd.DataFrame, chosen: str) -> Path:
    path = FIGURES / "v257_subject_memory_test_tail_rmse.png"
    sub = summary[
        summary["eval_split"].eq("test")
        & summary["bucket"].isin(["all", "bad_top10_v250", "strong_steer", "observe_later_like"])
        & summary["strategy"].isin(["v250_existing", chosen, "same_subject_vehicle_k1", "same_subject_vehicle_physio_raw15_k3"])
    ].copy()
    if sub.empty:
        return path
    buckets = ["all", "bad_top10_v250", "strong_steer", "observe_later_like"]
    strategies = list(dict.fromkeys(["v250_existing", chosen, "same_subject_vehicle_k1", "same_subject_vehicle_physio_raw15_k3"]))
    strategies = [s for s in strategies if s in set(sub["strategy"])]
    x = np.arange(len(buckets))
    width = 0.82 / max(1, len(strategies))
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for i, strategy in enumerate(strategies):
        vals = []
        for bucket in buckets:
            r = sub[sub["bucket"].eq(bucket) & sub["strategy"].eq(strategy)]
            vals.append(float(r["tail_rmse_mean"].iloc[0]) if len(r) else np.nan)
        ax.bar(x + (i - (len(strategies) - 1) / 2) * width, vals, width=width, label=strategy)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("test tail RMSE")
    ax.set_title("v257: 同驾驶员生理/车辆记忆检索 vs v250")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v252_script", V252_SCRIPT),
        ("v254b_script", V254B_SCRIPT),
        ("v254b_features", V254B_FEATURES),
        ("v256_physio_seq", V256_SEQ),
    ]:
        rows.append({"label": label, "path": str(path), "exists": bool(path.exists()), "sha256": file_sha256(path) if path.exists() and path.is_file() else ""})
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    rows = []
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


def write_report(summary: pd.DataFrame, selection: pd.DataFrame, chosen: str, coverage: pd.DataFrame, figures: List[Path]) -> None:
    lines: List[str] = []
    lines.append("# v257 同驾驶员生理状态记忆检索实验")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("- v254b-v256 已经说明当前生理数据不能直接提供 subject-disjoint 跨驾驶员增量。")
    lines.append("- 本轮检验一个更合理但更窄的路线：同一驾驶员有历史事件时，生理状态是否能帮助从该驾驶员历史未来原型中检索出更接近的未来。")
    lines.append("- 这是 subject-aware 个体化诊断，不是 subject-disjoint 正式泛化结果。")
    lines.append("")
    lines.append("## 方法")
    lines.append("")
    lines.append("- 候选池：同一 subject、同一 delay、训练 split 的历史事件。")
    lines.append("- 同一 recording 内只允许 observation_s 更早的训练事件，避免在线时序泄漏。")
    lines.append("- 特征距离：车辆输入、v254b 200Hz 生理统计、v256 raw 生理序列 PCA。")
    lines.append("- 预测：用候选训练未来曲线的加权平均作为个体化记忆预测。")
    lines.append("- 策略选择：只看 validation，score = bad_top10 tail + 3 * all-tail harm。")
    lines.append("")
    lines.append("## 候选覆盖")
    lines.append("")
    lines.append(coverage.to_markdown(index=False))
    lines.append("")
    lines.append("## Validation 选型")
    lines.append("")
    lines.append(selection.to_markdown(index=False))
    lines.append("")
    lines.append(f"- validation 选择策略：`{chosen}`")
    lines.append("")
    lines.append("## Test 结果")
    lines.append("")
    focus = summary[
        summary["eval_split"].eq("test")
        & summary["bucket"].isin(["all", "bad_top10_v250", "strong_steer", "observe_later_like"])
        & summary["strategy"].isin(["v250_existing", chosen, "same_subject_vehicle_k1", "same_subject_vehicle_physio_raw15_k3"])
    ].copy()
    lines.append(
        focus[
            [
                "bucket",
                "strategy",
                "n",
                "sample_rmse_mean",
                "tail_rmse_mean",
                "delta_tail_rmse_vs_v250",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    bad_v250 = focus[(focus["bucket"].eq("bad_top10_v250")) & (focus["strategy"].eq("v250_existing"))]
    bad_chosen = focus[(focus["bucket"].eq("bad_top10_v250")) & (focus["strategy"].eq(chosen))]
    all_chosen = focus[(focus["bucket"].eq("all")) & (focus["strategy"].eq(chosen))]
    if len(bad_v250) and len(bad_chosen):
        lines.append(
            f"- bad_top10：v250 tail={float(bad_v250['tail_rmse_mean'].iloc[0]):.4f}，"
            f"{chosen} tail={float(bad_chosen['tail_rmse_mean'].iloc[0]):.4f}，"
            f"delta={float(bad_chosen['delta_tail_rmse_vs_v250'].iloc[0]):+.4f}。"
        )
    if len(all_chosen):
        lines.append(
            f"- all：{chosen} 相对 v250 tail delta={float(all_chosen['delta_tail_rmse_vs_v250'].iloc[0]):+.4f}。"
        )
    lines.append("- 如果 chosen 在 bad_top10 上没有大幅优于 v250，说明即使转成个体化记忆范式，当前生理也没有达到 goal 要求。")
    lines.append("- 如果 chosen 只小幅改善或明显伤害 all，则不能作为主线，只能作为个体化补充诊断。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v257_subject_personalized_physio_memory_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v257] subject-personalized physio memory")
    clean_out_dir()
    np.random.seed(SEED)

    loaded = V252.load_fixed_inputs()
    manifest = loaded["data"].manifest.copy()
    split = V254B.make_subject_aware_split(manifest)
    train_mask = split == "train"
    y_true = loaded["y_true"].astype(np.float32)
    valid_mask = loaded["valid_mask"].astype(bool)
    sample_metrics = loaded["sample_metrics"].copy()
    fallback = loaded["pred_v250"].astype(np.float32)

    vehicle_x, vehicle_audit = standardize_by_train(loaded["x_flat"].astype(np.float32), train_mask)
    physio_stats_x, physio_stats, physio_audit = load_physio_stats(manifest, train_mask)
    raw_seq_x, raw_pca_audit = load_raw_seq_pca(train_mask)
    bad_top10 = build_bad_top10_by_split(sample_metrics, split)

    write_csv(vehicle_audit, TABLES / "v257_vehicle_standardization_audit.csv")
    write_csv(physio_audit, TABLES / "v257_physio_stats_feature_audit.csv")
    write_csv(raw_pca_audit, TABLES / "v257_raw_seq_pca_audit.csv")

    delays = manifest["delay_ms"].astype(int).to_numpy()
    subjects = manifest["subject"].astype(str).to_numpy()
    train_idx_by_subject_delay: Dict[Tuple[str, int], np.ndarray] = {}
    for subject in sorted(pd.unique(manifest["subject"].astype(str))):
        for delay in sorted(pd.unique(manifest["delay_ms"].astype(int))):
            idx = np.where(train_mask & (subjects == subject) & (delays == int(delay)))[0]
            if idx.size:
                train_idx_by_subject_delay[(str(subject), int(delay))] = idx

    all_pred_map = {"v250_existing": np.array(fallback, copy=True)}
    all_details = []
    for split_name in ["val", "test"]:
        query_idx = np.where(split == split_name)[0]
        print(f"[v257] predict {split_name} queries={len(query_idx)}", flush=True)
        pred_map, details = predict_memory_for_split(
            split_name,
            query_idx,
            train_idx_by_subject_delay,
            manifest,
            vehicle_x,
            physio_stats_x,
            raw_seq_x,
            y_true,
            valid_mask,
            fallback,
        )
        for strategy, pred in pred_map.items():
            if strategy not in all_pred_map:
                all_pred_map[strategy] = np.array(fallback, copy=True)
            all_pred_map[strategy][query_idx] = pred[query_idx]
        all_details.append(details)

    details = pd.concat(all_details, ignore_index=True)
    coverage = (
        details.groupby(["query_split", "strategy"], as_index=False)
        .agg(query_rows=("query_row_index", "count"), fallback_rate=("used_fallback_v250", "mean"), candidate_n_mean=("candidate_n", "mean"), candidate_n_p10=("candidate_n", lambda s: float(np.quantile(s, 0.10))))
    )

    summary, per_sample = summarize_predictions(all_pred_map, y_true, valid_mask, manifest, sample_metrics, split, bad_top10)
    chosen, selection = select_strategy_from_val(summary)
    figures = [plot_summary(summary, chosen)]

    write_csv(details, TABLES / "v257_memory_retrieval_details.csv")
    write_csv(coverage, TABLES / "v257_memory_candidate_coverage.csv")
    write_csv(summary, TABLES / "v257_prediction_summary.csv")
    write_csv(per_sample, TABLES / "v257_per_sample_metrics.csv")
    write_csv(selection, TABLES / "v257_validation_strategy_selection.csv")
    write_input_hashes()
    write_file_inventory()
    write_report(summary, selection, chosen, coverage, figures)
    write_file_inventory()
    zip_ok = make_zip()
    guardrail = {
        "pass": bool(zip_ok),
        "zip_testzip": bool(zip_ok),
        "protocol": "subject_aware_personalized_only",
        "uses_query_future_as_input": False,
        "same_recording_future_candidate_guard": True,
        "chosen_strategy": chosen,
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not guardrail["pass"]:
        raise AssertionError("v257 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    focus = summary[
        summary["eval_split"].eq("test")
        & summary["bucket"].eq("bad_top10_v250")
        & summary["strategy"].isin(["v250_existing", chosen])
    ].copy()
    print(f"[v257] report={REPORTS / 'v257_subject_personalized_physio_memory_cn.md'}")
    print(f"[v257] zip={ZIP_PATH}")
    if len(focus):
        print(focus[["strategy", "tail_rmse_mean", "delta_tail_rmse_vs_v250"]].to_string(index=False))


if __name__ == "__main__":
    main()
