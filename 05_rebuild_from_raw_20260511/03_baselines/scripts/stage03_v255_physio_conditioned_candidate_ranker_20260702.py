#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v255 生理状态条件化候选轨迹选择实验。

核心问题：
- 车辆历史在锚点前区分度不足时，训练集中常常存在“车辆很像但未来很不一样”的候选池；
- v253b 已证明 oracle 在候选池内很强，但简单生理最近邻选不出来；
- 本轮不做 residual/gate/删样本，而是训练一个候选重排序器，让生理状态参与“从车辆相似候选未来里选哪一种行为原型”。

输入：
- v252 固定 rolling sample、x_flat、y_true、v250/v241 样本误差；
- v254b 已抽取的 200Hz 事件级生理特征。

输出：
- subject-disjoint 正式口径与 subject-aware 个体化诊断口径下的候选选择效果；
- learned vehicle-only selector、learned physio-state selector、bad-focused physio selector 与 vehicle rank1 / oracle 的对照。
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
from sklearn.ensemble import HistGradientBoostingRegressor


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

OUT = BASELINES / "v255_physio_conditioned_candidate_ranker_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v255_physio_conditioned_candidate_ranker_20260702_pack.zip"

SEED = 25502
POOL_K = 60
THRESHOLD_GRID = [-1.0e9, 0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0e9]

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    """按路径导入已有阶段脚本，复用固定数据读取和 split 逻辑。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V252 = import_module_from_path("stage03_v252_for_v255", V252_SCRIPT)
V254B = import_module_from_path("stage03_v254b_for_v255", V254B_SCRIPT)


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


def stable_numeric_cols(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    out = []
    for col in cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def finite_nanmedian(x: np.ndarray, axis: int = 0) -> np.ndarray:
    with np.errstate(all="ignore"):
        med = np.nanmedian(x, axis=axis)
    med = np.asarray(med, dtype=float)
    med[~np.isfinite(med)] = 0.0
    return med


def standardize_by_train(x: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    用训练行估计 median/mean/std，再对全体样本标准化。
    这里不用 sklearn imputer，是为了保留全空列并避免不同版本行为不一致。
    """

    x = np.asarray(x, dtype=float)
    train_x = x[train_mask]
    med = finite_nanmedian(train_x, axis=0)
    filled = np.where(np.isfinite(x), x, med[None, :])
    mean = np.nanmean(filled[train_mask], axis=0)
    std = np.nanstd(filled[train_mask], axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    z = (filled - mean[None, :]) / std[None, :]
    audit = pd.DataFrame(
        {
            "feature_i": np.arange(x.shape[1]),
            "train_finite_n": np.isfinite(train_x).sum(axis=0),
            "train_mean_after_fill": mean,
            "train_std_after_fill": std,
        }
    )
    return z.astype(np.float32), audit


def load_physio200_blocks(manifest: pd.DataFrame, train_mask: np.ndarray) -> Dict[str, object]:
    """读取 v254b 的 200Hz 事件级生理特征，并构造 norm/curated/index 三组状态向量。"""

    if not V254B_FEATURES.exists():
        raise FileNotFoundError(f"缺少 v254b 生理特征表，请先运行 v254b：{V254B_FEATURES}")
    physio = pd.read_csv(V254B_FEATURES, encoding="utf-8-sig")
    if len(physio) != len(manifest):
        raise AssertionError(f"v254b 生理特征行数 {len(physio)} 与 manifest {len(manifest)} 不一致")
    if "row_index" in physio.columns and not np.array_equal(physio["row_index"].to_numpy(dtype=int), np.arange(len(physio))):
        raise AssertionError("v254b 生理特征 row_index 不是 0..n-1，不能安全对齐")

    physio_cols_all = stable_numeric_cols(physio, [c for c in physio.columns if c.startswith("physio200_")])
    norm_cols = [c for c in physio_cols_all if "_z_" in c or c.endswith("_index") or "burst_rate" in c]
    curated_cols = [
        c
        for c in norm_cols
        if any(sig in c for sig in ["HR_bpm", "EMG_RMS", "EMG_filt200", "EDA_Phasic", "EDA_Tonic", "RESP_filt200"])
    ]
    index_cols = stable_numeric_cols(
        physio,
        [
            "physio200_recent_arousal_index",
            "physio200_recent_motor_tension_index",
            "physio200_recent_resp_activity_index",
        ],
    )

    norm_x, norm_audit = standardize_by_train(physio[norm_cols].to_numpy(dtype=float), train_mask)
    curated_x, curated_audit = standardize_by_train(physio[curated_cols].to_numpy(dtype=float), train_mask)
    index_x, index_audit = standardize_by_train(physio[index_cols].to_numpy(dtype=float), train_mask)
    ok = physio["physio200_status"].astype(str).eq("ok").to_numpy(dtype=float) if "physio200_status" in physio.columns else np.ones(len(physio), dtype=float)

    audit = pd.concat(
        [
            norm_audit.assign(block="physio200_norm", feature=[norm_cols[i] for i in norm_audit["feature_i"]]),
            curated_audit.assign(block="physio200_curated", feature=[curated_cols[i] for i in curated_audit["feature_i"]]),
            index_audit.assign(block="physio200_index", feature=[index_cols[i] for i in index_audit["feature_i"]]),
        ],
        ignore_index=True,
    )
    return {
        "physio": physio,
        "norm_x": norm_x,
        "curated_x": curated_x,
        "index_x": index_x,
        "ok": ok.astype(np.float32),
        "norm_cols": norm_cols,
        "curated_cols": curated_cols,
        "index_cols": index_cols,
        "audit": audit,
    }


def build_future_summary(y_true: np.ndarray, valid_mask: np.ndarray) -> pd.DataFrame:
    """为训练库候选轨迹准备可部署的未来原型摘要。候选属于训练库，未来曲线本来就是 retrieval 输出原型。"""

    y = np.asarray(y_true, dtype=float)
    y_masked = np.where(valid_mask, y, np.nan)
    peak_abs = np.nanmax(np.abs(y_masked), axis=1)
    future_range = np.nanmax(y_masked, axis=1) - np.nanmin(y_masked, axis=1)
    mean_abs = np.nanmean(np.abs(y_masked), axis=1)
    final = y_masked[:, -1]
    slope = (y_masked[:, -1] - y_masked[:, 0]) / 2.0
    return pd.DataFrame(
        {
            "candidate_future_peak_abs": peak_abs,
            "candidate_future_range": future_range,
            "candidate_future_mean_abs": mean_abs,
            "candidate_future_final": final,
            "candidate_future_slope": slope,
        }
    ).replace([np.inf, -np.inf], np.nan)


def future_rmse_to_query(
    y_true: np.ndarray,
    valid_mask: np.ndarray,
    qi: int,
    cand_idx: np.ndarray,
    delay_ms: int,
) -> np.ndarray:
    """计算候选未来曲线与 query 真实未来之间的 RMSE，只用于训练/评价，不作为部署输入。"""

    horizon = V252.future_horizon_mask(int(delay_ms))
    q_valid = valid_mask[qi] & horizon
    if int(q_valid.sum()) < 3:
        return np.full(len(cand_idx), np.nan, dtype=float)
    cand_valid = valid_mask[cand_idx] & q_valid[None, :]
    diff2 = np.square(y_true[cand_idx] - y_true[qi][None, :])
    diff2 = np.where(cand_valid, diff2, np.nan)
    counts = cand_valid.sum(axis=1)
    with np.errstate(all="ignore"):
        rmse = np.sqrt(np.nanmean(diff2, axis=1))
    rmse[counts < 3] = np.nan
    return rmse.astype(float)


def build_bad_top10_by_protocol(sample_metrics: pd.DataFrame, split: np.ndarray) -> np.ndarray:
    """按当前协议的 train/val/test 各自 90% 分位重建 v250 差样本诊断标签。"""

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


def get_vehicle_pool(
    qi: int,
    candidate_idx: np.ndarray,
    vehicle_x: np.ndarray,
    event_uids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """同 delay 的训练候选里取车辆历史最近的 POOL_K 个，并排除同一 event_uid。"""

    cand = candidate_idx[event_uids[candidate_idx] != event_uids[qi]]
    if cand.size == 0:
        return cand, np.array([], dtype=float)
    diff = vehicle_x[cand] - vehicle_x[qi][None, :]
    dist = np.sqrt(np.nanmean(np.square(diff), axis=1))
    finite = np.isfinite(dist)
    cand = cand[finite]
    dist = dist[finite]
    if cand.size == 0:
        return cand, dist
    order = np.argsort(dist, kind="mergesort")[: min(POOL_K, cand.size)]
    return cand[order], dist[order]


def euclidean_dist_block(x: np.ndarray, qi: int, cand_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """返回 query-candidate 状态向量的 L2 和 L1 距离。"""

    if x.shape[1] == 0 or len(cand_idx) == 0:
        return np.full(len(cand_idx), np.nan), np.full(len(cand_idx), np.nan)
    diff = x[cand_idx] - x[qi][None, :]
    l2 = np.sqrt(np.nanmean(np.square(diff), axis=1))
    l1 = np.nanmean(np.abs(diff), axis=1)
    return l2.astype(float), l1.astype(float)


def build_pair_rows(
    protocol: str,
    query_split: str,
    query_indices: np.ndarray,
    candidate_indices_by_delay: Dict[int, np.ndarray],
    vehicle_x: np.ndarray,
    y_true: np.ndarray,
    valid_mask: np.ndarray,
    manifest: pd.DataFrame,
    sample_metrics: pd.DataFrame,
    future_summary: pd.DataFrame,
    physio_blocks: Dict[str, object],
    bad_top10: np.ndarray,
) -> pd.DataFrame:
    """为一组 query 构造候选池 pair 表，一行表示 query-candidate 对。"""

    event_uids = manifest["event_uid"].astype(str).to_numpy()
    subjects = manifest["subject"].astype(str).to_numpy()
    recordings = manifest["recording"].astype(str).to_numpy()
    delays = manifest["delay_ms"].astype(int).to_numpy()
    norm_x = physio_blocks["norm_x"]
    curated_x = physio_blocks["curated_x"]
    index_x = physio_blocks["index_x"]
    physio_ok = physio_blocks["ok"]
    future_np = future_summary.to_numpy(dtype=float)
    future_cols = list(future_summary.columns)

    parts: List[pd.DataFrame] = []
    for n_done, qi in enumerate(query_indices, start=1):
        delay = int(delays[qi])
        candidate_idx = candidate_indices_by_delay.get(delay)
        if candidate_idx is None or candidate_idx.size == 0:
            continue
        pool_idx, vehicle_dist = get_vehicle_pool(qi, candidate_idx, vehicle_x, event_uids)
        if pool_idx.size == 0:
            continue
        future_rmse = future_rmse_to_query(y_true, valid_mask, qi, pool_idx, delay)
        valid_future = np.isfinite(future_rmse)
        if not valid_future.any():
            continue
        pool_idx = pool_idx[valid_future]
        vehicle_dist = vehicle_dist[valid_future]
        future_rmse = future_rmse[valid_future]
        n = len(pool_idx)
        ranks = np.arange(1, n + 1, dtype=float)
        rank1_dist = float(vehicle_dist[0]) if n else math.nan
        norm_l2, norm_l1 = euclidean_dist_block(norm_x, qi, pool_idx)
        curated_l2, curated_l1 = euclidean_dist_block(curated_x, qi, pool_idx)
        index_l2, index_l1 = euclidean_dist_block(index_x, qi, pool_idx)
        q_index = index_x[qi]
        c_index = index_x[pool_idx]
        cand_future = future_np[pool_idx]
        q_bad = bool(bad_top10[qi])

        data: Dict[str, object] = {
            "protocol": protocol,
            "query_split": query_split,
            "query_row_index": int(qi),
            "query_event_uid": str(event_uids[qi]),
            "query_subject": str(subjects[qi]),
            "query_recording": str(recordings[qi]),
            "candidate_row_index": pool_idx.astype(int),
            "candidate_event_uid": event_uids[pool_idx],
            "candidate_subject": subjects[pool_idx],
            "candidate_recording": recordings[pool_idx],
            "delay_ms": int(delay),
            "vehicle_rank": ranks,
            "vehicle_rank_frac": ranks / max(1.0, float(POOL_K)),
            "vehicle_dist": vehicle_dist.astype(float),
            "vehicle_dist_delta_rank1": vehicle_dist.astype(float) - rank1_dist,
            "physio_norm_l2": norm_l2,
            "physio_norm_l1": norm_l1,
            "physio_curated_l2": curated_l2,
            "physio_curated_l1": curated_l1,
            "physio_index_l2": index_l2,
            "physio_index_l1": index_l1,
            "query_physio_ok": float(physio_ok[qi]),
            "candidate_physio_ok": physio_ok[pool_idx].astype(float),
            "both_physio_ok": (physio_ok[qi] * physio_ok[pool_idx]).astype(float),
            "same_subject": subjects[pool_idx] == subjects[qi],
            "same_recording": recordings[pool_idx] == recordings[qi],
            "future_rmse_to_query": future_rmse.astype(float),
            "query_bad_top10_v250": q_bad,
            "query_strong_steer": bool(sample_metrics.iloc[qi]["is_strong_steer"]),
            "query_observe_later_like": bool(sample_metrics.iloc[qi]["is_observe_later_like"]),
        }
        for j, col in enumerate(future_cols):
            data[col] = cand_future[:, j]
        for j, name in enumerate(["arousal", "motor_tension", "resp_activity"][: index_x.shape[1]]):
            data[f"query_physio_{name}"] = float(q_index[j])
            data[f"candidate_physio_{name}"] = c_index[:, j].astype(float)
            data[f"absdiff_physio_{name}"] = np.abs(c_index[:, j] - q_index[j]).astype(float)
        parts.append(pd.DataFrame(data))

        if n_done % 500 == 0:
            print(f"[v255] {protocol}/{query_split}: built pools for {n_done}/{len(query_indices)} queries", flush=True)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


VEHICLE_CONTEXT_FEATURES = [
    "delay_ms",
    "vehicle_rank",
    "vehicle_rank_frac",
    "vehicle_dist",
    "vehicle_dist_delta_rank1",
    "candidate_future_peak_abs",
    "candidate_future_range",
    "candidate_future_mean_abs",
    "candidate_future_final",
    "candidate_future_slope",
]

PHYSIO_FEATURES = [
    "physio_norm_l2",
    "physio_norm_l1",
    "physio_curated_l2",
    "physio_curated_l1",
    "physio_index_l2",
    "physio_index_l1",
    "query_physio_ok",
    "candidate_physio_ok",
    "both_physio_ok",
    "query_physio_arousal",
    "candidate_physio_arousal",
    "absdiff_physio_arousal",
    "query_physio_motor_tension",
    "candidate_physio_motor_tension",
    "absdiff_physio_motor_tension",
    "query_physio_resp_activity",
    "candidate_physio_resp_activity",
    "absdiff_physio_resp_activity",
]


def fill_pair_features(df: pd.DataFrame, feature_cols: List[str], train_ref: pd.DataFrame | None = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """将 pair 特征转为矩阵；缺失值用训练 pair 的 median 填充。"""

    x = df[feature_cols].to_numpy(dtype=float)
    if train_ref is None:
        ref = x
    else:
        ref = train_ref[feature_cols].to_numpy(dtype=float)
    med = finite_nanmedian(ref, axis=0)
    x = np.where(np.isfinite(x), x, med[None, :])
    return x.astype(np.float32), pd.DataFrame({"feature": feature_cols, "fill_median": med})


def fit_ranker(train_pairs: pd.DataFrame, feature_cols: List[str], sample_weight: np.ndarray | None = None) -> Tuple[HistGradientBoostingRegressor, pd.DataFrame]:
    """训练候选 pair 的未来 RMSE 预测器，预测值越低表示越应该选。"""

    x_train, fill = fill_pair_features(train_pairs, feature_cols)
    y = train_pairs["future_rmse_to_query"].to_numpy(dtype=float)
    good = np.isfinite(y)
    if int(good.sum()) < 1000:
        raise AssertionError(f"pair 训练样本太少：{int(good.sum())}")
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=260,
        learning_rate=0.055,
        max_leaf_nodes=31,
        l2_regularization=0.10,
        random_state=SEED,
    )
    if sample_weight is not None:
        model.fit(x_train[good], y[good], sample_weight=sample_weight[good])
    else:
        model.fit(x_train[good], y[good])
    return model, fill


def add_predictions(
    pairs: pd.DataFrame,
    train_pairs: pd.DataFrame,
    model: HistGradientBoostingRegressor,
    feature_cols: List[str],
    pred_col: str,
) -> pd.DataFrame:
    x, _ = fill_pair_features(pairs, feature_cols, train_ref=train_pairs)
    out = pairs.copy()
    out[pred_col] = model.predict(x)
    return out


def select_from_pairs(pairs: pd.DataFrame, pred_col: str | None, strategy: str, threshold: float) -> pd.DataFrame:
    """按预测 RMSE 从候选池选择一个候选；threshold 越大越保守。"""

    rows: List[Dict[str, object]] = []
    for qi, g in pairs.groupby("query_row_index", sort=False):
        g = g.sort_values("vehicle_rank")
        rank1 = g.iloc[0]
        oracle = g.loc[g["future_rmse_to_query"].idxmin()]
        if pred_col is None:
            chosen = rank1
            pred_rank1 = math.nan
            pred_best = math.nan
            predicted_gain = math.nan
        else:
            best = g.loc[g[pred_col].idxmin()]
            pred_rank1 = float(rank1[pred_col])
            pred_best = float(best[pred_col])
            predicted_gain = pred_rank1 - pred_best
            chosen = best if predicted_gain >= threshold else rank1
        rows.append(
            {
                "strategy": strategy,
                "threshold": threshold,
                "query_row_index": int(qi),
                "query_event_uid": str(rank1["query_event_uid"]),
                "query_subject": str(rank1["query_subject"]),
                "query_recording": str(rank1["query_recording"]),
                "delay_ms": int(rank1["delay_ms"]),
                "selected_candidate_row_index": int(chosen["candidate_row_index"]),
                "selected_candidate_event_uid": str(chosen["candidate_event_uid"]),
                "selected_candidate_subject": str(chosen["candidate_subject"]),
                "selected_vehicle_rank": int(chosen["vehicle_rank"]),
                "selected_future_rmse_to_query": float(chosen["future_rmse_to_query"]),
                "vehicle_rank1_future_rmse_to_query": float(rank1["future_rmse_to_query"]),
                "oracle_best_future_rmse_to_query": float(oracle["future_rmse_to_query"]),
                "delta_selected_minus_vehicle_rank1": float(chosen["future_rmse_to_query"] - rank1["future_rmse_to_query"]),
                "delta_selected_minus_oracle": float(chosen["future_rmse_to_query"] - oracle["future_rmse_to_query"]),
                "predicted_rmse_rank1": pred_rank1,
                "predicted_rmse_best": pred_best,
                "predicted_gain_vs_rank1": predicted_gain,
                "selected_same_subject": bool(chosen["same_subject"]),
                "selected_same_recording": bool(chosen["same_recording"]),
                "query_bad_top10_v250": bool(rank1["query_bad_top10_v250"]),
                "query_strong_steer": bool(rank1["query_strong_steer"]),
                "query_observe_later_like": bool(rank1["query_observe_later_like"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_selection(selection: pd.DataFrame, protocol: str, eval_split: str) -> pd.DataFrame:
    """按 bucket 汇总候选选择效果。"""

    rows: List[Dict[str, object]] = []
    bucket_defs = [
        ("all", np.ones(len(selection), dtype=bool)),
        ("bad_top10_v250", selection["query_bad_top10_v250"].astype(bool).to_numpy()),
        ("strong_steer", selection["query_strong_steer"].astype(bool).to_numpy()),
        ("observe_later_like", selection["query_observe_later_like"].astype(bool).to_numpy()),
    ]
    for bucket, mask in bucket_defs:
        sub = selection[mask].copy()
        if sub.empty:
            continue
        for strategy, g in sub.groupby("strategy", sort=False):
            delta = g["delta_selected_minus_vehicle_rank1"].to_numpy(dtype=float)
            selected = g["selected_future_rmse_to_query"].to_numpy(dtype=float)
            vehicle = g["vehicle_rank1_future_rmse_to_query"].to_numpy(dtype=float)
            oracle = g["oracle_best_future_rmse_to_query"].to_numpy(dtype=float)
            rows.append(
                {
                    "protocol": protocol,
                    "eval_split": eval_split,
                    "bucket": bucket,
                    "strategy": strategy,
                    "threshold": float(g["threshold"].iloc[0]),
                    "n": int(len(g)),
                    "selected_future_rmse_mean": float(np.nanmean(selected)),
                    "vehicle_rank1_future_rmse_mean": float(np.nanmean(vehicle)),
                    "oracle_future_rmse_mean": float(np.nanmean(oracle)),
                    "delta_selected_minus_vehicle_mean": float(np.nanmean(delta)),
                    "delta_selected_minus_vehicle_median": float(np.nanmedian(delta)),
                    "improve_rate_vs_vehicle_rank1": float(np.nanmean(delta < 0.0)),
                    "selected_neighbor_vehicle_rank_mean": float(g["selected_vehicle_rank"].mean()),
                    "same_subject_rate": float(g["selected_same_subject"].mean()),
                    "same_recording_rate": float(g["selected_same_recording"].mean()),
                }
            )
    return pd.DataFrame(rows)


def tune_threshold(
    protocol: str,
    model_name: str,
    val_pairs: pd.DataFrame,
    pred_col: str,
) -> Tuple[float, pd.DataFrame]:
    """在 val 上选择 guarded rerank 阈值：优先不伤害 all，再看 bad_top10 改善。"""

    rows = []
    selected_by_t = []
    for t in THRESHOLD_GRID:
        sel = select_from_pairs(val_pairs, pred_col, model_name, t)
        selected_by_t.append(sel)
        summary = summarize_selection(sel, protocol, "val")
        rows.append(summary)
    table = pd.concat(rows, ignore_index=True)
    pivot = table[table["bucket"].isin(["all", "bad_top10_v250"])].copy()
    candidates = []
    for t in THRESHOLD_GRID:
        all_row = pivot[(pivot["threshold"].eq(t)) & pivot["bucket"].eq("all")]
        bad_row = pivot[(pivot["threshold"].eq(t)) & pivot["bucket"].eq("bad_top10_v250")]
        if all_row.empty:
            continue
        all_delta = float(all_row["delta_selected_minus_vehicle_mean"].iloc[0])
        bad_delta = float(bad_row["delta_selected_minus_vehicle_mean"].iloc[0]) if not bad_row.empty else math.inf
        candidates.append({"threshold": t, "all_delta": all_delta, "bad_delta": bad_delta})
    cand = pd.DataFrame(candidates)
    no_harm = cand[cand["all_delta"] <= 0.02].copy()
    if no_harm.empty:
        chosen = 1.0e9
    else:
        no_harm = no_harm.sort_values(["bad_delta", "all_delta", "threshold"], ascending=[True, True, True])
        chosen = float(no_harm.iloc[0]["threshold"])
    table["chosen_for_model"] = table["threshold"].eq(chosen)
    return chosen, table


def plot_bad_bucket(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v255_badtop10_candidate_selection_rmse.png"
    sub = summary[
        summary["eval_split"].eq("test")
        & summary["bucket"].eq("bad_top10_v250")
        & summary["strategy"].isin(
            [
                "vehicle_rank1",
                "learned_vehicle_context_guarded",
                "learned_physio_state_guarded",
                "learned_physio_badweighted_guarded",
                "oracle_best_future",
            ]
        )
    ].copy()
    if sub.empty:
        return path
    order = [
        "vehicle_rank1",
        "learned_vehicle_context_guarded",
        "learned_physio_state_guarded",
        "learned_physio_badweighted_guarded",
        "oracle_best_future",
    ]
    sub["strategy"] = pd.Categorical(sub["strategy"], categories=order, ordered=True)
    protocols = list(sub["protocol"].drop_duplicates())
    fig, axes = plt.subplots(1, len(protocols), figsize=(7 * len(protocols), 5.2), squeeze=False)
    for ax, protocol in zip(axes[0], protocols):
        g = sub[sub["protocol"].eq(protocol)].sort_values("strategy")
        x = np.arange(len(g))
        ax.bar(x, g["selected_future_rmse_mean"], color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s).replace("_", "\n") for s in g["strategy"]], fontsize=8)
        ax.set_title(f"{protocol}: bad_top10_v250")
        ax.set_ylabel("selected future RMSE")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_delta(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v255_test_delta_vs_vehicle_rank1.png"
    sub = summary[
        summary["eval_split"].eq("test")
        & summary["bucket"].isin(["all", "bad_top10_v250", "strong_steer", "observe_later_like"])
        & summary["strategy"].isin(
            [
                "learned_vehicle_context_guarded",
                "learned_physio_state_guarded",
                "learned_physio_badweighted_guarded",
            ]
        )
    ].copy()
    if sub.empty:
        return path
    labels = sorted(sub["bucket"].unique())
    protocols = list(sub["protocol"].drop_duplicates())
    fig, axes = plt.subplots(len(protocols), 1, figsize=(12, 4.2 * len(protocols)), squeeze=False)
    for ax, protocol in zip(axes[:, 0], protocols):
        g = sub[sub["protocol"].eq(protocol)].copy()
        strategies = list(g["strategy"].drop_duplicates())
        x = np.arange(len(labels))
        width = 0.78 / max(1, len(strategies))
        for i, strategy in enumerate(strategies):
            vals = []
            for bucket in labels:
                r = g[g["bucket"].eq(bucket) & g["strategy"].eq(strategy)]
                vals.append(float(r["delta_selected_minus_vehicle_mean"].iloc[0]) if len(r) else np.nan)
            ax.bar(x + (i - (len(strategies) - 1) / 2) * width, vals, width=width, label=strategy.replace("_guarded", ""))
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{protocol}: test delta vs vehicle rank1 (negative is better)")
        ax.set_ylabel("delta RMSE")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_protocol(
    protocol: str,
    split: np.ndarray,
    loaded: Dict[str, object],
) -> Dict[str, pd.DataFrame]:
    """运行一个 split 协议下的训练、val 调阈值、test 评价。"""

    manifest = loaded["data"].manifest.copy()
    vehicle_x = loaded["x_flat"].astype(np.float32)
    y_true = loaded["y_true"].astype(np.float32)
    valid_mask = loaded["valid_mask"].astype(bool)
    sample_metrics = loaded["sample_metrics"].copy()
    delays = manifest["delay_ms"].astype(int).to_numpy()

    train_mask = split == "train"
    physio_blocks = load_physio200_blocks(manifest, train_mask)
    future_summary = build_future_summary(y_true, valid_mask)
    bad_top10 = build_bad_top10_by_protocol(sample_metrics, split)

    candidate_indices_by_delay = {
        int(delay): np.where(train_mask & (delays == int(delay)))[0]
        for delay in sorted(pd.unique(manifest["delay_ms"].astype(int)))
    }
    pair_tables = {}
    for split_name in ["train", "val", "test"]:
        query_idx = np.where((split == split_name))[0]
        print(f"[v255] {protocol}: build {split_name} pairs for {len(query_idx)} queries", flush=True)
        pair_tables[split_name] = build_pair_rows(
            protocol,
            split_name,
            query_idx,
            candidate_indices_by_delay,
            vehicle_x,
            y_true,
            valid_mask,
            manifest,
            sample_metrics,
            future_summary,
            physio_blocks,
            bad_top10,
        )
        print(f"[v255] {protocol}: {split_name} pair rows={len(pair_tables[split_name])}", flush=True)

    train_pairs = pair_tables["train"]
    val_pairs = pair_tables["val"]
    test_pairs = pair_tables["test"]
    if train_pairs.empty or val_pairs.empty or test_pairs.empty:
        raise AssertionError(f"{protocol} pair 表为空，不能训练/评价")

    vehicle_cols = [c for c in VEHICLE_CONTEXT_FEATURES if c in train_pairs.columns]
    physio_cols = [c for c in PHYSIO_FEATURES if c in train_pairs.columns]
    all_physio_cols = vehicle_cols + physio_cols

    print(f"[v255] {protocol}: train learned_vehicle_context", flush=True)
    model_vehicle, fill_vehicle = fit_ranker(train_pairs, vehicle_cols)
    bad_weight = 1.0 + 3.0 * train_pairs["query_bad_top10_v250"].astype(float).to_numpy()
    print(f"[v255] {protocol}: train learned_physio_state", flush=True)
    model_physio, fill_physio = fit_ranker(train_pairs, all_physio_cols)
    print(f"[v255] {protocol}: train learned_physio_badweighted", flush=True)
    model_bad, fill_bad = fit_ranker(train_pairs, all_physio_cols, sample_weight=bad_weight)

    for split_name in ["val", "test"]:
        pair_tables[split_name] = add_predictions(pair_tables[split_name], train_pairs, model_vehicle, vehicle_cols, "pred_vehicle_context_rmse")
        pair_tables[split_name] = add_predictions(pair_tables[split_name], train_pairs, model_physio, all_physio_cols, "pred_physio_state_rmse")
        pair_tables[split_name] = add_predictions(pair_tables[split_name], train_pairs, model_bad, all_physio_cols, "pred_physio_badweighted_rmse")

    chosen_vehicle, tune_vehicle = tune_threshold(protocol, "learned_vehicle_context_guarded", pair_tables["val"], "pred_vehicle_context_rmse")
    chosen_physio, tune_physio = tune_threshold(protocol, "learned_physio_state_guarded", pair_tables["val"], "pred_physio_state_rmse")
    chosen_bad, tune_bad = tune_threshold(protocol, "learned_physio_badweighted_guarded", pair_tables["val"], "pred_physio_badweighted_rmse")
    tune = pd.concat([tune_vehicle, tune_physio, tune_bad], ignore_index=True)

    selections = []
    for split_name in ["val", "test"]:
        p = pair_tables[split_name]
        rank1 = select_from_pairs(p, None, "vehicle_rank1", 1.0e9)
        oracle = select_from_pairs(p.assign(pred_oracle=p["future_rmse_to_query"]), "pred_oracle", "oracle_best_future", -1.0e9)
        vehicle_sel = select_from_pairs(p, "pred_vehicle_context_rmse", "learned_vehicle_context_guarded", chosen_vehicle)
        physio_sel = select_from_pairs(p, "pred_physio_state_rmse", "learned_physio_state_guarded", chosen_physio)
        bad_sel = select_from_pairs(p, "pred_physio_badweighted_rmse", "learned_physio_badweighted_guarded", chosen_bad)
        selections.extend([rank1, vehicle_sel, physio_sel, bad_sel, oracle])
    selection = pd.concat(selections, ignore_index=True)
    summary = pd.concat(
        [
            summarize_selection(selection[selection["strategy"].notna() & selection["query_row_index"].isin(pair_tables[split_name]["query_row_index"].unique())], protocol, split_name)
            for split_name in ["val", "test"]
        ],
        ignore_index=True,
    )

    feature_audit = pd.DataFrame(
        [
            {"protocol": protocol, "model": "learned_vehicle_context_guarded", "n_features": len(vehicle_cols), "features": "|".join(vehicle_cols), "chosen_threshold": chosen_vehicle},
            {"protocol": protocol, "model": "learned_physio_state_guarded", "n_features": len(all_physio_cols), "features": "|".join(all_physio_cols), "chosen_threshold": chosen_physio},
            {"protocol": protocol, "model": "learned_physio_badweighted_guarded", "n_features": len(all_physio_cols), "features": "|".join(all_physio_cols), "chosen_threshold": chosen_bad},
        ]
    )
    fill_audit = pd.concat(
        [
            fill_vehicle.assign(protocol=protocol, model="learned_vehicle_context_guarded"),
            fill_physio.assign(protocol=protocol, model="learned_physio_state_guarded"),
            fill_bad.assign(protocol=protocol, model="learned_physio_badweighted_guarded"),
        ],
        ignore_index=True,
    )
    physio_audit = physio_blocks["audit"].assign(protocol=protocol)

    # pair 表很大，只保存 test 的关键列，避免产物过重。
    keep_pair_cols = [
        "protocol",
        "query_split",
        "query_row_index",
        "candidate_row_index",
        "delay_ms",
        "vehicle_rank",
        "vehicle_dist",
        "physio_norm_l2",
        "physio_curated_l2",
        "physio_index_l2",
        "future_rmse_to_query",
        "pred_vehicle_context_rmse",
        "pred_physio_state_rmse",
        "pred_physio_badweighted_rmse",
    ]
    test_pairs_export = pair_tables["test"][[c for c in keep_pair_cols if c in pair_tables["test"].columns]].copy()
    return {
        "selection": selection,
        "summary": summary,
        "tuning": tune,
        "feature_audit": feature_audit,
        "fill_audit": fill_audit,
        "physio_audit": physio_audit,
        "test_pairs": test_pairs_export,
    }


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v252_script", V252_SCRIPT),
        ("v254b_script", V254B_SCRIPT),
        ("v254b_features", V254B_FEATURES),
    ]:
        rows.append(
            {
                "label": label,
                "path": str(path),
                "exists": bool(path.exists()),
                "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
            }
        )
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


def best_rows(summary: pd.DataFrame, protocol: str, bucket: str) -> pd.DataFrame:
    sub = summary[summary["protocol"].eq(protocol) & summary["eval_split"].eq("test") & summary["bucket"].eq(bucket)].copy()
    return sub.sort_values("selected_future_rmse_mean")


def write_report(summary: pd.DataFrame, tuning: pd.DataFrame, feature_audit: pd.DataFrame, figures: List[Path]) -> None:
    lines: List[str] = []
    lines.append("# v255 生理状态条件化候选轨迹选择实验")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("- v254b 说明：200Hz 生理直接拼接到车辆输入后，正式 subject-disjoint 轨迹行为诊断没有增量。")
    lines.append("- v253b 说明：车辆相似候选池内 oracle 上限很高，但简单生理最近邻没有选中好未来。")
    lines.append("- v255 因此改成学习式候选重排序：车辆先给候选池，生理状态只负责在候选未来原型中参与选择。")
    lines.append("")
    lines.append("## 方法边界")
    lines.append("")
    lines.append("- 不使用 query 的未来作为部署输入；未来 RMSE 只用于训练 pair 监督和离线评价。")
    lines.append("- 候选未来摘要来自训练库候选样本，因为 retrieval 预测本身就是从训练库选未来原型。")
    lines.append("- 不做删样本、不做 residual 修正、不做 v222a 式 gate；这里是候选轨迹选择模型。")
    lines.append("- subject-disjoint 是正式泛化口径；subject-aware 只表示同一驾驶员有历史样本时的个体化潜力。")
    lines.append("")
    lines.append("## 特征与阈值")
    lines.append("")
    lines.append(feature_audit[["protocol", "model", "n_features", "chosen_threshold"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Test 关键结果")
    lines.append("")
    focus = summary[
        summary["eval_split"].eq("test")
        & summary["bucket"].isin(["all", "bad_top10_v250", "strong_steer", "observe_later_like"])
        & summary["strategy"].isin(
            [
                "vehicle_rank1",
                "learned_vehicle_context_guarded",
                "learned_physio_state_guarded",
                "learned_physio_badweighted_guarded",
                "oracle_best_future",
            ]
        )
    ].copy()
    lines.append(
        focus[
            [
                "protocol",
                "bucket",
                "strategy",
                "n",
                "selected_future_rmse_mean",
                "delta_selected_minus_vehicle_mean",
                "improve_rate_vs_vehicle_rank1",
                "selected_neighbor_vehicle_rank_mean",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")
    lines.append("## 关键判读")
    lines.append("")
    for protocol in ["subject_disjoint", "subject_aware"]:
        bad = best_rows(summary, protocol, "bad_top10_v250")
        if bad.empty:
            continue
        rank1 = bad[bad["strategy"].eq("vehicle_rank1")]
        best_non_oracle = bad[~bad["strategy"].isin(["vehicle_rank1", "oracle_best_future"])].head(1)
        oracle = bad[bad["strategy"].eq("oracle_best_future")]
        if not rank1.empty and not best_non_oracle.empty and not oracle.empty:
            r0 = rank1.iloc[0]
            rb = best_non_oracle.iloc[0]
            ro = oracle.iloc[0]
            lines.append(
                f"- {protocol} / bad_top10：vehicle rank1={float(r0['selected_future_rmse_mean']):.4f}，"
                f"最佳非 oracle={rb['strategy']} {float(rb['selected_future_rmse_mean']):.4f} "
                f"(delta={float(rb['delta_selected_minus_vehicle_mean']):+.4f})，"
                f"oracle={float(ro['selected_future_rmse_mean']):.4f}。"
            )
    lines.append("- 如果 learned_physio_state 明显优于 learned_vehicle_context，才说明生理状态真正提供了候选选择增量。")
    lines.append("- 如果只在 subject-aware 改善，说明生理更适合作为个体化校准信号；若 subject-disjoint 仍无改善，就不能把它宣称为跨驾驶员通用行为信息。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v255_physio_conditioned_candidate_ranker_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v255] physio-conditioned candidate ranker")
    clean_out_dir()
    np.random.seed(SEED)

    loaded = V252.load_fixed_inputs()
    manifest = loaded["data"].manifest.copy()
    split_subject_disjoint = manifest["split"].astype(str).to_numpy()
    split_subject_aware = V254B.make_subject_aware_split(manifest)

    outputs = []
    for protocol, split in [
        ("subject_disjoint", split_subject_disjoint),
        ("subject_aware", split_subject_aware),
    ]:
        outputs.append(run_protocol(protocol, split, loaded))

    selection = pd.concat([o["selection"] for o in outputs], ignore_index=True)
    summary = pd.concat([o["summary"] for o in outputs], ignore_index=True)
    tuning = pd.concat([o["tuning"] for o in outputs], ignore_index=True)
    feature_audit = pd.concat([o["feature_audit"] for o in outputs], ignore_index=True)
    fill_audit = pd.concat([o["fill_audit"] for o in outputs], ignore_index=True)
    physio_audit = pd.concat([o["physio_audit"] for o in outputs], ignore_index=True)
    test_pairs = pd.concat([o["test_pairs"] for o in outputs], ignore_index=True)

    write_csv(selection, TABLES / "v255_selected_candidate_per_query.csv")
    write_csv(summary, TABLES / "v255_candidate_selection_summary.csv")
    write_csv(tuning, TABLES / "v255_threshold_tuning_summary.csv")
    write_csv(feature_audit, TABLES / "v255_ranker_feature_audit.csv")
    write_csv(fill_audit, TABLES / "v255_pair_feature_fill_values.csv")
    write_csv(physio_audit, TABLES / "v255_physio_feature_standardization_audit.csv")
    write_csv(test_pairs, TABLES / "v255_test_pair_predictions_compact.csv")

    figures = [plot_bad_bucket(summary), plot_delta(summary)]
    write_input_hashes()
    write_file_inventory()
    write_report(summary, tuning, feature_audit, figures)
    write_file_inventory()
    zip_ok = make_zip()
    guardrail = {
        "pass": bool(zip_ok),
        "zip_testzip": bool(zip_ok),
        "v254b_features_exists": bool(V254B_FEATURES.exists()),
        "n_selection_rows": int(len(selection)),
        "n_summary_rows": int(len(summary)),
        "pool_k": int(POOL_K),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not guardrail["pass"]:
        raise AssertionError("v255 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    focus = summary[
        summary["protocol"].eq("subject_disjoint")
        & summary["eval_split"].eq("test")
        & summary["bucket"].eq("bad_top10_v250")
    ].sort_values("selected_future_rmse_mean")
    print(f"[v255] report={REPORTS / 'v255_physio_conditioned_candidate_ranker_cn.md'}")
    print(f"[v255] zip={ZIP_PATH}")
    if len(focus):
        r = focus.iloc[0]
        print(
            "[v255] subject_disjoint bad_top10 best strategy={} rmse={:.6f} delta={:.6f}".format(
                str(r["strategy"]),
                float(r["selected_future_rmse_mean"]),
                float(r["delta_selected_minus_vehicle_mean"]),
            )
        )


if __name__ == "__main__":
    main()
