#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v252 input-similarity future-divergence audit。

本轮目标：
- 回答用户提出的核心问题：锚点前输入很像的样本，锚点后的真实轨迹是否会明显分叉；
- 不训练新模型，不调通道，不做样本删除，也不做 anchor/gate/selector；
- 固定 v250_minimal_lateral7 的输入通道口径，复用 v251 的逐样本误差；
- 对每个 test rolling sample，在同 delay 的 train 样本里找锚点前输入最相似的近邻；
- 量化这些近邻的未来真实轨迹分歧，并与 v250/v241 的真实误差做相关和重叠分析；
- 输出 casebook 图，直接展示“前面像、后面不一样”的样本。

解释边界：
- 这是可辨识性审计，不是模型提升实验；
- 若高误差样本的近邻未来分歧明显更大，说明当前任务存在同输入多未来问题；
- 后续改法应转向概率/多模态/不确定性预测，而不是继续堆确定性回归模型。
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


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V250_SCRIPT = BASELINES / "scripts" / "stage03_v250_history_channel_ablation_20260630.py"
V250_DIR = BASELINES / "v250_history_channel_ablation_20260630"
V250_PRED = V250_DIR / "v250_channel_ablation_predictions.npz"
V250_SELECTION = V250_DIR / "tables" / "v250_model_selection_validation_channel_ablation.csv"

V251_DIR = BASELINES / "v251_locked_robustness_v250_20260701"
V251_SAMPLE = V251_DIR / "tables" / "v251_sample_locked_delta.csv"

OUT = BASELINES / "v252_input_similarity_future_divergence_20260701"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v252_input_similarity_future_divergence_20260701_pack.zip"

SEED = 252
K_NEIGHBORS = 12
CASEBOOK_K = 8
DELAY_MS = [0, 200, 400, 600, 800, 1000]

MINIMAL_CHANNELS = [
    "steering",
    "speed_kmh",
    "ay",
    "yaw_rate",
    "roll",
    "lane_curvature",
    "lateral_distance",
]

BUCKETS = [
    "all",
    "normal_predictable",
    "observe_later_like",
    "strong_steer",
    "reverse_or_multi_correction",
    "bad_top10_v241",
    "bad_top10_v250",
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
    """按路径导入前序脚本，复用已经验证过的数据构造函数。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V250 = import_module_from_path("stage03_v250_history_channel_ablation_20260630_for_v252", V250_SCRIPT)
V249 = V250.V249
V238 = V250.V238
V239 = V250.V239
FUTURE_GRID = V238.FUTURE_GRID.astype(np.float32)
HIST_GRID = np.linspace(-3.0, 0.0, 31).astype(np.float32)


def ensure_dirs() -> None:
    """创建 v252 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v252 自己的输出，不碰前序版本。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 统一使用 utf-8-sig，方便 Windows/Excel 直接打开中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算输入文件 SHA256，用于复现追溯。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_corr(x: Iterable[float], y: Iterable[float]) -> Tuple[float, float, int]:
    """返回 Pearson、Spearman 和有效样本数。"""

    xs = pd.Series(list(x), dtype="float64")
    ys = pd.Series(list(y), dtype="float64")
    mask = np.isfinite(xs.to_numpy()) & np.isfinite(ys.to_numpy())
    xs = xs[mask]
    ys = ys[mask]
    n = int(len(xs))
    if n < 3 or float(xs.std()) == 0.0 or float(ys.std()) == 0.0:
        return math.nan, math.nan, n
    pearson = float(np.corrcoef(xs.to_numpy(), ys.to_numpy())[0, 1])
    sx = xs.rank(method="average").to_numpy()
    sy = ys.rank(method="average").to_numpy()
    spearman = float(np.corrcoef(sx, sy)[0, 1])
    return pearson, spearman, n


def pairwise_rmse(curves: np.ndarray) -> float:
    """计算一组未来曲线之间的两两 RMSE 均值。"""

    if len(curves) < 2:
        return math.nan
    vals: List[float] = []
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            vals.append(float(np.sqrt(np.mean(np.square(curves[i] - curves[j])))))
    return float(np.mean(vals)) if vals else math.nan


def future_horizon_mask(delay_ms: int) -> np.ndarray:
    """当前 rolling sample 在 original_remaining 下可评价的未来 horizon。"""

    original_rel = delay_ms / 1000.0 + FUTURE_GRID
    return original_rel <= 2.0 + 1e-9


def future_tail_mask(delay_ms: int) -> np.ndarray:
    """当前 rolling sample 在 original_remaining 下的 tail horizon。"""

    original_rel = delay_ms / 1000.0 + FUTURE_GRID
    return (original_rel <= 2.0 + 1e-9) & (original_rel >= 1.0 - 1e-9)


def flatten_sample_input(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    """把标准化后的 hist/road/phase 拼成 sample-level 输入向量。"""

    n = int(arrays["hist"].shape[0])
    return np.concatenate(
        [
            arrays["hist"].reshape(n, -1),
            arrays["road"].reshape(n, -1),
            arrays["phase"].reshape(n, -1),
        ],
        axis=1,
    ).astype(np.float32)


def load_fixed_inputs() -> Dict[str, object]:
    """
    读取并构造 v252 所需输入。

    注意：这里只重新构造 v250_minimal_lateral7 的标准化输入，不训练模型。
    """

    if not V250_PRED.exists():
        raise FileNotFoundError(f"缺少 v250 prediction：{V250_PRED}")
    if not V251_SAMPLE.exists():
        raise FileNotFoundError(f"缺少 v251 sample metrics：{V251_SAMPLE}")
    if not V250_SELECTION.exists():
        raise FileNotFoundError(f"缺少 v250 selection：{V250_SELECTION}")

    selection = pd.read_csv(V250_SELECTION, encoding="utf-8-sig")
    best_name = str(selection.iloc[0]["model_name"])
    if best_name != "v250_minimal_lateral7":
        raise AssertionError(f"v252 预期固定 v250_minimal_lateral7，实际 best={best_name}")

    base_data = V238.load_v236_data()
    group = next(g for g in V250.channel_groups() if g.model_name == "v250_minimal_lateral7")
    if list(group.channels) != MINIMAL_CHANNELS:
        raise AssertionError(f"minimal channel 定义不一致：{group.channels}")

    reduced = V250.make_reduced_data(base_data, group)
    x_base = V238.build_base_design_matrix(reduced)
    point_data = V238.build_point_dataset(reduced, x_base)
    point_masks = V238.split_point_masks(point_data, reduced.manifest)
    scalers = V239.fit_scalers(reduced, point_data, point_masks)
    arrays = V239.standardize_arrays(reduced, point_data, scalers)
    x_flat = flatten_sample_input(arrays)

    with np.load(V250_PRED, allow_pickle=False) as pred:
        y_true = pred["y_true_steering_delta"].astype(np.float32)
        pred_v241 = pred["pred_v241_steering_delta"].astype(np.float32)
        pred_v250 = pred["pred_v250_minimal_lateral7_steering_delta"].astype(np.float32)
        best_from_npz = str(pred["best_channel_model"][0])
    if best_from_npz != best_name:
        raise AssertionError(f"npz best={best_from_npz} 与 selection best={best_name} 不一致")

    sample_metrics = pd.read_csv(V251_SAMPLE, encoding="utf-8-sig")
    if len(sample_metrics) != len(reduced.manifest):
        raise AssertionError("v251 sample metrics 与 manifest 行数不一致")
    if not np.all(sample_metrics["row_index"].to_numpy(dtype=int) == np.arange(len(sample_metrics))):
        raise AssertionError("v251 sample metrics row_index 与 manifest index 不一致")

    valid_mask, _ = V238.build_original_remaining_mask(reduced.manifest)
    split_check = V238.split_integrity_check(reduced.manifest)

    return {
        "selection": selection,
        "data": reduced,
        "x_flat": x_flat,
        "y_true": y_true,
        "pred_v241": pred_v241,
        "pred_v250": pred_v250,
        "sample_metrics": sample_metrics,
        "valid_mask": valid_mask,
        "split_check": split_check,
    }


def sample_bucket_flags(sample_metrics: pd.DataFrame) -> Dict[str, np.ndarray]:
    """生成样本分层标记。"""

    test = sample_metrics[sample_metrics["split"].eq("test")].copy()
    v250_bad_threshold = float(np.nanquantile(test["tail_rmse_v250"].to_numpy(dtype=float), 0.90))
    return {
        "all": np.ones(len(sample_metrics), dtype=bool),
        "normal_predictable": sample_metrics["is_normal_predictable"].astype(bool).to_numpy(),
        "observe_later_like": sample_metrics["is_observe_later_like"].astype(bool).to_numpy(),
        "strong_steer": sample_metrics["is_strong_steer"].astype(bool).to_numpy(),
        "reverse_or_multi_correction": sample_metrics["is_reverse_or_multi_correction"].astype(bool).to_numpy(),
        "bad_top10_v241": sample_metrics["bad_top10_v241"].astype(bool).to_numpy(),
        "bad_top10_v250": pd.to_numeric(sample_metrics["tail_rmse_v250"], errors="coerce").to_numpy(dtype=float)
        >= v250_bad_threshold,
    }


def row_tags(row: pd.Series) -> str:
    """把样本所属类别压缩成易读标签。"""

    tags = []
    for name in [
        "bad_top10_v241",
        "bad_top10_v250",
        "normal_predictable",
        "observe_later_like",
        "strong_steer",
        "reverse_or_multi_correction",
    ]:
        col = f"is_{name}" if name not in {"bad_top10_v241", "bad_top10_v250"} else name
        if col in row.index and bool(row[col]):
            tags.append(name)
    return "|".join(tags) if tags else "unlabeled"


def compute_neighbor_audit(
    manifest: pd.DataFrame,
    x_flat: np.ndarray,
    y_true: np.ndarray,
    pred_v241: np.ndarray,
    pred_v250: np.ndarray,
    sample_metrics: pd.DataFrame,
    valid_mask: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """对每个 test rolling sample 做同 delay 的 train 近邻未来分歧审计。"""

    rng = np.random.default_rng(SEED)
    split = manifest["split"].astype(str).to_numpy()
    delays = manifest["delay_ms"].astype(int).to_numpy()
    flags = sample_bucket_flags(sample_metrics)
    test_tail_error = sample_metrics.loc[sample_metrics["split"].eq("test"), "tail_rmse_v250"].to_numpy(dtype=float)
    tail_error_q90 = float(np.nanquantile(test_tail_error, 0.90))

    rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    for delay in DELAY_MS:
        query_idx = np.where((split == "test") & (delays == delay))[0]
        train_idx = np.where((split == "train") & (delays == delay))[0]
        if len(query_idx) == 0 or len(train_idx) == 0:
            continue

        x_train = x_flat[train_idx]
        for qi in query_idx:
            diff = x_train - x_flat[qi][None, :]
            dist = np.sqrt(np.mean(np.square(diff), axis=1))
            order = np.argsort(dist, kind="mergesort")[: min(K_NEIGHBORS, len(dist))]
            neighbor_idx = train_idx[order]
            neighbor_dist = dist[order]

            horizon = future_horizon_mask(delay)
            q_valid = valid_mask[qi] & horizon
            all_neighbor_valid = np.all(valid_mask[neighbor_idx], axis=0)
            curve_mask = q_valid & all_neighbor_valid
            if int(curve_mask.sum()) < 3:
                curve_mask = q_valid
            if int(curve_mask.sum()) < 3:
                continue

            curves = y_true[neighbor_idx][:, curve_mask]
            q_curve = y_true[qi, curve_mask]
            pred241_curve = pred_v241[qi, curve_mask]
            pred250_curve = pred_v250[qi, curve_mask]

            neighbor_to_query = np.sqrt(np.mean(np.square(curves - q_curve[None, :]), axis=1))
            neighbor_pairwise = pairwise_rmse(curves)
            neighbor_mean_curve = curves.mean(axis=0)
            neighbor_mean_to_query = float(np.sqrt(np.mean(np.square(neighbor_mean_curve - q_curve))))
            query_pred250_rmse = float(np.sqrt(np.mean(np.square(pred250_curve - q_curve))))
            query_pred241_rmse = float(np.sqrt(np.mean(np.square(pred241_curve - q_curve))))

            peak_abs = np.max(np.abs(curves), axis=1)
            final_abs = np.abs(curves[:, -1])
            signed_final = curves[:, -1]
            q_peak_abs = float(np.max(np.abs(q_curve)))
            q_final = float(q_curve[-1])
            sign_disagree_rate = float(np.mean(np.sign(signed_final) != np.sign(q_final))) if abs(q_final) > 1e-6 else math.nan
            slope_abs = (
                np.max(np.abs(np.diff(curves, axis=1)), axis=1) if curves.shape[1] >= 2 else np.full(len(curves), np.nan)
            )

            sm = sample_metrics.iloc[qi]
            row = {
                "row_index": int(qi),
                "rolling_sample_index": int(manifest.iloc[qi]["rolling_sample_index"]),
                "event_uid": str(manifest.iloc[qi]["event_uid"]),
                "subject": str(manifest.iloc[qi]["subject"]),
                "recording": str(manifest.iloc[qi]["recording"]),
                "split": "test",
                "delay_ms": int(delay),
                "neighbor_k": int(len(neighbor_idx)),
                "future_points": int(curve_mask.sum()),
                "neighbor_input_distance_min": float(np.min(neighbor_dist)),
                "neighbor_input_distance_mean": float(np.mean(neighbor_dist)),
                "neighbor_input_distance_p90": float(np.quantile(neighbor_dist, 0.90)),
                "neighbor_future_pairwise_rmse_mean": float(neighbor_pairwise),
                "neighbor_future_to_query_best_rmse": float(np.min(neighbor_to_query)),
                "neighbor_future_to_query_mean_rmse": float(np.mean(neighbor_to_query)),
                "neighbor_future_to_query_median_rmse": float(np.median(neighbor_to_query)),
                "neighbor_mean_curve_to_query_rmse": neighbor_mean_to_query,
                "neighbor_peak_abs_mean": float(np.mean(peak_abs)),
                "neighbor_peak_abs_std": float(np.std(peak_abs)),
                "neighbor_final_std": float(np.std(signed_final)),
                "neighbor_final_range": float(np.max(signed_final) - np.min(signed_final)),
                "neighbor_slope_abs_std": float(np.nanstd(slope_abs)),
                "neighbor_sign_disagree_rate_to_query": sign_disagree_rate,
                "query_true_peak_abs": q_peak_abs,
                "query_true_final": q_final,
                "pred250_curve_rmse_same_window": query_pred250_rmse,
                "pred241_curve_rmse_same_window": query_pred241_rmse,
                "tail_rmse_v241": float(sm["tail_rmse_v241"]),
                "tail_rmse_v250": float(sm["tail_rmse_v250"]),
                "delta_tail_rmse_v250_minus_v241": float(sm["delta_tail_rmse_v250_minus_v241"]),
                "sample_rmse_v241": float(sm["sample_rmse_v241"]),
                "sample_rmse_v250": float(sm["sample_rmse_v250"]),
                "bad_top10_v241": bool(sm["bad_top10_v241"]),
                "bad_top10_v250": bool(float(sm["tail_rmse_v250"]) >= tail_error_q90),
                "is_normal_predictable": bool(sm["is_normal_predictable"]),
                "is_observe_later_like": bool(sm["is_observe_later_like"]),
                "is_strong_steer": bool(sm["is_strong_steer"]),
                "is_reverse_or_multi_correction": bool(sm["is_reverse_or_multi_correction"]),
                "neighbor_event_uids": "|".join(manifest.iloc[neighbor_idx]["event_uid"].astype(str).tolist()),
                "neighbor_row_indices": "|".join(str(int(x)) for x in neighbor_idx),
            }
            row["tags"] = row_tags(pd.Series(row))
            rows.append(row)

            for rank, (ni, nd, fr) in enumerate(zip(neighbor_idx, neighbor_dist, neighbor_to_query), start=1):
                detail_rows.append(
                    {
                        "query_row_index": int(qi),
                        "query_event_uid": str(manifest.iloc[qi]["event_uid"]),
                        "query_delay_ms": int(delay),
                        "neighbor_rank": int(rank),
                        "neighbor_row_index": int(ni),
                        "neighbor_event_uid": str(manifest.iloc[ni]["event_uid"]),
                        "neighbor_subject": str(manifest.iloc[ni]["subject"]),
                        "neighbor_recording": str(manifest.iloc[ni]["recording"]),
                        "neighbor_input_distance": float(nd),
                        "future_rmse_to_query": float(fr),
                        "neighbor_true_peak_abs": float(np.max(np.abs(y_true[ni, curve_mask]))),
                        "neighbor_true_final": float(y_true[ni, curve_mask][-1]),
                    }
                )

        # 防止不同平台排序完全相同时出现不稳定，这里固定触发一次 rng 记录种子使用。
        _ = rng.random()

    audit = pd.DataFrame(rows)
    details = pd.DataFrame(detail_rows)
    if audit.empty:
        return audit, details

    # 分位数标签只用于诊断解释，不参与模型选择。
    pair_q75 = float(np.nanquantile(audit["neighbor_future_pairwise_rmse_mean"], 0.75))
    pair_q90 = float(np.nanquantile(audit["neighbor_future_pairwise_rmse_mean"], 0.90))
    to_query_q75 = float(np.nanquantile(audit["neighbor_future_to_query_mean_rmse"], 0.75))
    audit["ambiguity_q75_threshold"] = pair_q75
    audit["ambiguity_q90_threshold"] = pair_q90
    audit["to_query_q75_threshold"] = to_query_q75
    audit["high_neighbor_divergence_q75"] = audit["neighbor_future_pairwise_rmse_mean"] >= pair_q75
    audit["very_high_neighbor_divergence_q90"] = audit["neighbor_future_pairwise_rmse_mean"] >= pair_q90
    audit["high_query_neighbor_mismatch_q75"] = audit["neighbor_future_to_query_mean_rmse"] >= to_query_q75
    audit["input_ambiguous_abs050"] = audit["neighbor_future_pairwise_rmse_mean"] >= 0.50
    audit["bad_top10_v250_and_high_ambiguity"] = audit["bad_top10_v250"] & audit["high_neighbor_divergence_q75"]
    return audit, details


def summarize_by_bucket_delay(audit: pd.DataFrame) -> pd.DataFrame:
    """按 delay 和 bucket 汇总未来分歧与误差。"""

    if audit.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for bucket in BUCKETS:
        if bucket == "all":
            mask = np.ones(len(audit), dtype=bool)
        elif bucket == "bad_top10_v241":
            mask = audit["bad_top10_v241"].astype(bool).to_numpy()
        elif bucket == "bad_top10_v250":
            mask = audit["bad_top10_v250"].astype(bool).to_numpy()
        else:
            mask = audit[f"is_{bucket}"].astype(bool).to_numpy()
        for delay_label, sub in [("all_delays", audit[mask]), *[(str(d), audit[mask & audit["delay_ms"].eq(d)]) for d in DELAY_MS]]:
            if sub.empty:
                continue
            rows.append(
                {
                    "bucket": bucket,
                    "delay_ms": delay_label,
                    "n": int(len(sub)),
                    "event_n": int(sub["event_uid"].nunique()),
                    "neighbor_input_distance_mean": float(sub["neighbor_input_distance_mean"].mean()),
                    "neighbor_future_pairwise_rmse_mean": float(sub["neighbor_future_pairwise_rmse_mean"].mean()),
                    "neighbor_future_pairwise_rmse_median": float(sub["neighbor_future_pairwise_rmse_mean"].median()),
                    "neighbor_future_to_query_mean_rmse": float(sub["neighbor_future_to_query_mean_rmse"].mean()),
                    "neighbor_peak_abs_std_mean": float(sub["neighbor_peak_abs_std"].mean()),
                    "neighbor_final_range_mean": float(sub["neighbor_final_range"].mean()),
                    "high_neighbor_divergence_q75_rate": float(sub["high_neighbor_divergence_q75"].mean()),
                    "very_high_neighbor_divergence_q90_rate": float(sub["very_high_neighbor_divergence_q90"].mean()),
                    "input_ambiguous_abs050_rate": float(sub["input_ambiguous_abs050"].mean()),
                    "tail_rmse_v250_mean": float(sub["tail_rmse_v250"].mean()),
                    "tail_rmse_v241_mean": float(sub["tail_rmse_v241"].mean()),
                    "bad_top10_v250_rate": float(sub["bad_top10_v250"].mean()),
                    "bad_top10_v241_rate": float(sub["bad_top10_v241"].mean()),
                }
            )
    return pd.DataFrame(rows)


def error_ambiguity_correlations(audit: pd.DataFrame) -> pd.DataFrame:
    """计算未来分歧指标与真实误差的相关。"""

    if audit.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    subsets = [("all_delays", audit), *[(f"{d}ms", audit[audit["delay_ms"].eq(d)]) for d in DELAY_MS]]
    x_cols = [
        "neighbor_future_pairwise_rmse_mean",
        "neighbor_future_to_query_mean_rmse",
        "neighbor_mean_curve_to_query_rmse",
        "neighbor_peak_abs_std",
        "neighbor_final_range",
        "neighbor_input_distance_mean",
    ]
    y_cols = [
        "tail_rmse_v250",
        "tail_rmse_v241",
        "sample_rmse_v250",
        "sample_rmse_v241",
        "delta_tail_rmse_v250_minus_v241",
    ]
    for subset_name, sub in subsets:
        if len(sub) < 5:
            continue
        for x_col in x_cols:
            for y_col in y_cols:
                pearson, spearman, n = finite_corr(sub[x_col], sub[y_col])
                rows.append(
                    {
                        "subset": subset_name,
                        "x_metric": x_col,
                        "y_metric": y_col,
                        "n": n,
                        "pearson": pearson,
                        "spearman": spearman,
                    }
                )
    return pd.DataFrame(rows)


def overlap_table(audit: pd.DataFrame) -> pd.DataFrame:
    """统计高歧义样本和高误差样本是否重叠。"""

    if audit.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    subsets = [("all_delays", audit), ("delay0", audit[audit["delay_ms"].eq(0)])]
    for name, sub in subsets:
        if sub.empty:
            continue
        high_amb = sub["high_neighbor_divergence_q75"].astype(bool)
        very_high_amb = sub["very_high_neighbor_divergence_q90"].astype(bool)
        bad250 = sub["bad_top10_v250"].astype(bool)
        bad241 = sub["bad_top10_v241"].astype(bool)
        for amb_name, amb_mask in [("high_q75", high_amb), ("very_high_q90", very_high_amb)]:
            rows.append(
                {
                    "subset": name,
                    "ambiguity_group": amb_name,
                    "n": int(len(sub)),
                    "ambiguity_n": int(amb_mask.sum()),
                    "bad_top10_v250_n": int(bad250.sum()),
                    "overlap_bad250_and_ambiguity_n": int((bad250 & amb_mask).sum()),
                    "bad250_covered_by_ambiguity_rate": float((bad250 & amb_mask).sum() / max(1, bad250.sum())),
                    "bad250_rate_inside_ambiguity": float((bad250 & amb_mask).sum() / max(1, amb_mask.sum())),
                    "bad250_rate_outside_ambiguity": float((bad250 & ~amb_mask).sum() / max(1, (~amb_mask).sum())),
                    "bad_top10_v241_n": int(bad241.sum()),
                    "overlap_bad241_and_ambiguity_n": int((bad241 & amb_mask).sum()),
                    "bad241_covered_by_ambiguity_rate": float((bad241 & amb_mask).sum() / max(1, bad241.sum())),
                    "bad241_rate_inside_ambiguity": float((bad241 & amb_mask).sum() / max(1, amb_mask.sum())),
                    "bad241_rate_outside_ambiguity": float((bad241 & ~amb_mask).sum() / max(1, (~amb_mask).sum())),
                }
            )
    return pd.DataFrame(rows)


def zscore(values: pd.Series) -> pd.Series:
    """安全 z-score，用于 casebook 排序。"""

    arr = pd.to_numeric(values, errors="coerce")
    std = float(arr.std())
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(arr)), index=arr.index)
    return (arr - float(arr.mean())) / std


def build_casebook_index(audit: pd.DataFrame) -> pd.DataFrame:
    """选择最值得人工看的样本：高误差、高近邻分歧、近邻均值也无法贴近 query。"""

    if audit.empty:
        return pd.DataFrame()
    work = audit.copy()
    work["casebook_score"] = (
        zscore(work["tail_rmse_v250"])
        + zscore(work["neighbor_future_pairwise_rmse_mean"])
        + zscore(work["neighbor_future_to_query_mean_rmse"])
        + 0.5 * zscore(work["neighbor_final_range"])
    )
    work["casebook_reason"] = np.where(
        work["bad_top10_v250_and_high_ambiguity"],
        "bad_top10_v250_and_high_neighbor_divergence",
        np.where(work["bad_top10_v250"], "bad_top10_v250", "high_neighbor_divergence_or_mismatch"),
    )
    cols = [
        "row_index",
        "event_uid",
        "subject",
        "recording",
        "delay_ms",
        "tags",
        "casebook_score",
        "casebook_reason",
        "tail_rmse_v250",
        "tail_rmse_v241",
        "delta_tail_rmse_v250_minus_v241",
        "neighbor_input_distance_mean",
        "neighbor_future_pairwise_rmse_mean",
        "neighbor_future_to_query_mean_rmse",
        "neighbor_mean_curve_to_query_rmse",
        "neighbor_peak_abs_std",
        "neighbor_final_range",
        "neighbor_sign_disagree_rate_to_query",
        "neighbor_event_uids",
        "neighbor_row_indices",
    ]
    return work.sort_values("casebook_score", ascending=False)[cols].reset_index(drop=True)


def plot_error_scatter(audit: pd.DataFrame) -> Path:
    """画未来分歧 vs v250 误差散点。"""

    path = FIGURES / "v252_error_vs_neighbor_future_divergence.png"
    if audit.empty:
        return path
    fig, ax = plt.subplots(figsize=(9, 6))
    delays = sorted(audit["delay_ms"].unique())
    cmap = plt.get_cmap("viridis", len(delays))
    for i, delay in enumerate(delays):
        sub = audit[audit["delay_ms"].eq(delay)]
        ax.scatter(
            sub["neighbor_future_pairwise_rmse_mean"],
            sub["tail_rmse_v250"],
            s=np.where(sub["bad_top10_v250"].astype(bool), 42, 18),
            alpha=0.72,
            color=cmap(i),
            label=f"{int(delay)}ms",
            edgecolors=np.where(sub["bad_top10_v250"].astype(bool), "black", "none"),
            linewidths=0.5,
        )
    x = audit["neighbor_future_pairwise_rmse_mean"].to_numpy(dtype=float)
    y = audit["tail_rmse_v250"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 3:
        coef = np.polyfit(x[mask], y[mask], deg=1)
        xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="black", linewidth=1.2, linestyle="--", label="linear trend")
    ax.set_title("v252: 输入近邻的未来分歧 vs 当前 v250 tail error")
    ax.set_xlabel("同 delay 训练近邻的未来两两 RMSE 均值")
    ax.set_ylabel("v250 tail RMSE")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_error_group_bar(audit: pd.DataFrame) -> Path:
    """按 v250 误差分组画近邻未来分歧均值。"""

    path = FIGURES / "v252_neighbor_divergence_by_error_group.png"
    if audit.empty:
        return path
    work = audit.copy()
    q50 = float(np.nanquantile(work["tail_rmse_v250"], 0.50))
    q75 = float(np.nanquantile(work["tail_rmse_v250"], 0.75))
    q90 = float(np.nanquantile(work["tail_rmse_v250"], 0.90))
    work["error_group"] = pd.cut(
        work["tail_rmse_v250"],
        bins=[-np.inf, q50, q75, q90, np.inf],
        labels=["low<=p50", "mid p50-p75", "high p75-p90", "bad top10"],
    )
    summary = work.groupby("error_group", observed=True).agg(
        n=("row_index", "count"),
        future_pairwise=("neighbor_future_pairwise_rmse_mean", "mean"),
        query_neighbor=("neighbor_future_to_query_mean_rmse", "mean"),
        high_amb_rate=("high_neighbor_divergence_q75", "mean"),
    )
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(summary.index))
    ax1.bar(x - 0.18, summary["future_pairwise"], width=0.36, label="近邻之间未来分歧", color="#4C78A8")
    ax1.bar(x + 0.18, summary["query_neighbor"], width=0.36, label="query vs 近邻未来差距", color="#F58518")
    ax1.set_ylabel("future RMSE")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{idx}\nN={int(summary.loc[idx, 'n'])}" for idx in summary.index])
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, summary["high_amb_rate"], color="#2CA02C", marker="o", label="高歧义率")
    ax2.set_ylabel("高歧义率")
    ax2.set_ylim(0, 1)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax1.set_title("v252: 误差越高，近邻未来是否越分叉")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_delay_summary(summary: pd.DataFrame) -> Path:
    """画 all bucket 下不同 delay 的未来分歧。"""

    path = FIGURES / "v252_delay_future_divergence_summary.png"
    if summary.empty:
        return path
    sub = summary[summary["bucket"].eq("all") & ~summary["delay_ms"].eq("all_delays")].copy()
    if sub.empty:
        return path
    sub["delay_num"] = sub["delay_ms"].astype(int)
    sub = sub.sort_values("delay_num")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sub["delay_num"], sub["neighbor_future_pairwise_rmse_mean"], marker="o", label="近邻之间未来分歧")
    ax.plot(sub["delay_num"], sub["neighbor_future_to_query_mean_rmse"], marker="s", label="query vs 近邻未来差距")
    ax.plot(sub["delay_num"], sub["tail_rmse_v250_mean"], marker="^", label="v250 tail RMSE")
    ax.set_title("v252: 等待更多观察后，近邻未来分叉是否下降")
    ax.set_xlabel("delay / ms")
    ax.set_ylabel("RMSE / divergence")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_neighbor_indices(value: str, k: int = CASEBOOK_K) -> List[int]:
    """从表格字符串解析近邻 row index。"""

    out: List[int] = []
    for part in str(value).split("|"):
        if part.strip():
            out.append(int(part))
        if len(out) >= k:
            break
    return out


def plot_casebook(
    casebook: pd.DataFrame,
    data,
    y_true: np.ndarray,
    pred_v241: np.ndarray,
    pred_v250: np.ndarray,
    valid_mask: np.ndarray,
    title: str,
    path: Path,
    n_cases: int = 6,
) -> Path:
    """
    画“锚点前相似、锚点后分叉”的 casebook。

    左列：query 与近邻的锚点前 steering 历史；
    右列：query、近邻、近邻均值、v241/v250 的锚点后 steering_delta。
    """

    chosen = casebook.head(n_cases).copy()
    if chosen.empty:
        return path
    fig, axes = plt.subplots(len(chosen), 2, figsize=(16, 3.0 * len(chosen)), squeeze=False)
    steering_idx = MINIMAL_CHANNELS.index("steering")

    for row_i, (_, row) in enumerate(chosen.iterrows()):
        qi = int(row["row_index"])
        delay = int(row["delay_ms"])
        neigh = parse_neighbor_indices(str(row["neighbor_row_indices"]), CASEBOOK_K)
        horizon = future_horizon_mask(delay)
        q_valid = valid_mask[qi] & horizon
        if neigh:
            all_neighbor_valid = np.all(valid_mask[neigh], axis=0)
            curve_mask = q_valid & all_neighbor_valid
            if int(curve_mask.sum()) < 3:
                curve_mask = q_valid
        else:
            curve_mask = q_valid

        ax_hist = axes[row_i, 0]
        for ni in neigh:
            ax_hist.plot(HIST_GRID, data.x_hist[ni, :, steering_idx], color="#b8b8b8", linewidth=0.8, alpha=0.65)
        ax_hist.plot(HIST_GRID, data.x_hist[qi, :, steering_idx], color="black", linewidth=2.0, label="query history")
        ax_hist.axvline(0.0, color="#666666", linewidth=0.8)
        ax_hist.set_title(
            f"锚点前 steering 输入近邻 | {row['subject']} {delay}ms | input_dist={row['neighbor_input_distance_mean']:.3f}",
            fontsize=9,
        )
        ax_hist.set_xlabel("anchor 前时间 / s")
        ax_hist.set_ylabel("absolute steering")
        ax_hist.grid(alpha=0.25)
        if row_i == 0:
            ax_hist.legend(fontsize=8, loc="best")

        ax_future = axes[row_i, 1]
        x = FUTURE_GRID[curve_mask]
        if neigh:
            curves = y_true[neigh][:, curve_mask]
            for curve in curves:
                ax_future.plot(x, curve, color="#b8b8b8", linewidth=0.8, alpha=0.65)
            ax_future.plot(x, curves.mean(axis=0), color="#1f77b4", linewidth=1.8, label="neighbor mean true")
        ax_future.plot(x, y_true[qi, curve_mask], color="black", linewidth=2.2, label="query true")
        ax_future.plot(x, pred_v241[qi, curve_mask], color="#00a88f", linestyle="--", linewidth=1.5, label="v241 pred")
        ax_future.plot(x, pred_v250[qi, curve_mask], color="#f27c1e", linestyle="-.", linewidth=1.7, label="v250 pred")
        ax_future.axhline(0.0, color="#666666", linewidth=0.7)
        ax_future.set_title(
            f"{row['event_uid']} | v250_tail={row['tail_rmse_v250']:.3f} | "
            f"near_future_pair={row['neighbor_future_pairwise_rmse_mean']:.3f} | "
            f"q-near={row['neighbor_future_to_query_mean_rmse']:.3f}",
            fontsize=9,
        )
        ax_future.set_xlabel("anchor 后时间 / s")
        ax_future.set_ylabel("steering_delta")
        ax_future.grid(alpha=0.25)
        if row_i == 0:
            ax_future.legend(fontsize=8, loc="best")

    fig.suptitle(title, fontsize=13, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    selection: pd.DataFrame,
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    corr: pd.DataFrame,
    overlap: pd.DataFrame,
    casebook: pd.DataFrame,
    figures: List[Path],
) -> None:
    """写 v252 中文报告。"""

    lines: List[str] = []
    lines.append("# v252 输入相似样本的未来分叉审计")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("本轮只回答一个问题：锚点前输入相似的样本，锚点后的真实 steering_delta 是否会明显分叉。")
    lines.append("如果答案是肯定的，那么当前差样本并不只是模型结构问题，而是存在同输入多未来或可观测信息不足。")
    lines.append("")
    lines.append("## 固定边界")
    lines.append("")
    lines.append("- 固定 v250 validation-only 选出的 `v250_minimal_lateral7` 输入口径。")
    lines.append("- 不重新训练，不调通道，不删样本，不做 anchor selector / gate / router。")
    lines.append("- 每个 test sample 只在同 delay 的 train sample 中找近邻，避免 delay 口径混在一起。")
    lines.append("- 近邻搜索使用标准化后的 `hist + road + phase` sample-level 输入。")
    lines.append("")
    lines.append("## v250 选择来源")
    lines.append("")
    lines.append(
        selection.head(1)[
            ["model_name", "n_hist_channels", "channels", "best_epoch", "best_val_loss", "accepted_as_channel_candidate"]
        ].to_markdown(index=False)
    )
    lines.append("")
    lines.append("## 总体摘要")
    lines.append("")
    all_row = summary[summary["bucket"].eq("all") & summary["delay_ms"].eq("all_delays")].head(1)
    bad_row = summary[summary["bucket"].eq("bad_top10_v250") & summary["delay_ms"].eq("all_delays")].head(1)
    delay0 = summary[summary["bucket"].eq("all") & summary["delay_ms"].eq("0")].head(1)
    if not all_row.empty:
        r = all_row.iloc[0]
        lines.append(
            f"- 全 test rolling sample：N={int(r['n'])}，近邻未来两两 RMSE 均值={float(r['neighbor_future_pairwise_rmse_mean']):.3f}，"
            f"query-vs-neighbor 未来 RMSE 均值={float(r['neighbor_future_to_query_mean_rmse']):.3f}，"
            f"高近邻分歧率={float(r['high_neighbor_divergence_q75_rate']):.3f}。"
        )
    if not bad_row.empty:
        r = bad_row.iloc[0]
        lines.append(
            f"- 当前 v250 bad_top10 样本：N={int(r['n'])}，近邻未来两两 RMSE 均值={float(r['neighbor_future_pairwise_rmse_mean']):.3f}，"
            f"query-vs-neighbor 未来 RMSE 均值={float(r['neighbor_future_to_query_mean_rmse']):.3f}，"
            f"高近邻分歧率={float(r['high_neighbor_divergence_q75_rate']):.3f}。"
        )
    if not delay0.empty:
        r = delay0.iloc[0]
        lines.append(
            f"- 0ms 原始锚点：N={int(r['n'])}，近邻未来两两 RMSE 均值={float(r['neighbor_future_pairwise_rmse_mean']):.3f}，"
            f"query-vs-neighbor 未来 RMSE 均值={float(r['neighbor_future_to_query_mean_rmse']):.3f}。"
        )
    lines.append("")
    lines.append("## Bucket / Delay 摘要")
    lines.append("")
    keep = summary[
        summary["bucket"].isin(["all", "normal_predictable", "observe_later_like", "strong_steer", "bad_top10_v250"])
        & summary["delay_ms"].isin(["all_delays", "0", "600", "1000"])
    ].copy()
    show_cols = [
        "bucket",
        "delay_ms",
        "n",
        "event_n",
        "neighbor_input_distance_mean",
        "neighbor_future_pairwise_rmse_mean",
        "neighbor_future_to_query_mean_rmse",
        "high_neighbor_divergence_q75_rate",
        "tail_rmse_v250_mean",
    ]
    lines.append(keep[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 误差与未来分叉相关")
    lines.append("")
    corr_keep = corr[
        corr["subset"].isin(["all_delays", "0ms"])
        & corr["x_metric"].isin(
            [
                "neighbor_future_pairwise_rmse_mean",
                "neighbor_future_to_query_mean_rmse",
                "neighbor_mean_curve_to_query_rmse",
                "neighbor_input_distance_mean",
            ]
        )
        & corr["y_metric"].isin(["tail_rmse_v250", "tail_rmse_v241"])
    ].copy()
    lines.append(corr_keep.to_markdown(index=False))
    lines.append("")
    lines.append("## 高误差与高分叉重叠")
    lines.append("")
    lines.append(overlap.to_markdown(index=False))
    lines.append("")
    lines.append("## 人工审查优先样本")
    lines.append("")
    case_cols = [
        "event_uid",
        "subject",
        "delay_ms",
        "casebook_reason",
        "tail_rmse_v250",
        "neighbor_future_pairwise_rmse_mean",
        "neighbor_future_to_query_mean_rmse",
        "neighbor_input_distance_mean",
    ]
    lines.append(casebook.head(12)[case_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 解释")
    lines.append("")
    lines.append(
        "若 casebook 图中左侧锚点前 steering 历史高度相似，但右侧近邻真实未来呈扇形分叉，"
        "则说明单条确定性曲线预测会自然学成折中曲线。此时继续增强 MLP/TCN/attention 只能有限改善，"
        "更合理的下一步是概率预测、多模态候选轨迹或显式不确定性建模。"
    )
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    lines.append("")
    lines.append("## 关键表")
    lines.append("")
    lines.append("- `tables/v252_neighbor_divergence_by_sample.csv`")
    lines.append("- `tables/v252_neighbor_detail.csv`")
    lines.append("- `tables/v252_summary_by_delay_bucket.csv`")
    lines.append("- `tables/v252_error_ambiguity_correlation.csv`")
    lines.append("- `tables/v252_high_ambiguity_error_overlap.csv`")
    lines.append("- `tables/v252_casebook_index.csv`")
    lines.append("")
    (REPORTS / "v252_input_similarity_future_divergence_cn.md").write_text("\n".join(lines), encoding="utf-8")


def write_input_hashes() -> None:
    """记录关键输入文件哈希。"""

    paths = [V250_SCRIPT, V250_PRED, V250_SELECTION, V251_SAMPLE, V238.V236_ARRAYS, V238.V236_MANIFEST]
    rows = []
    for path in paths:
        p = Path(path)
        if p.exists():
            rows.append({"path": str(p), "sha256": file_sha256(p), "bytes": int(p.stat().st_size)})
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    """记录输出目录文件清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "bytes": int(path.stat().st_size)})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> str | None:
    """打包 v252 关键产物。"""

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(Path(__file__), arcname=f"scripts/{Path(__file__).name}")
        for folder in [TABLES, FIGURES, REPORTS, LOGS]:
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(OUT)))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip()


def build_guardrail(split_check: pd.DataFrame, zip_test: str | None) -> Dict[str, object]:
    """生成本轮审计约束。"""

    cross = int(split_check["same_event_uid_cross_split"].sum()) if "same_event_uid_cross_split" in split_check.columns else 0
    return {
        "pass": bool(cross == 0 and zip_test is None),
        "same_event_uid_cross_split_count": cross,
        "test_used_for_model_selection": False,
        "fixed_input_model": "v250_minimal_lateral7",
        "retrained_model": False,
        "changed_channels": False,
        "sample_deletion": False,
        "neighbor_pool": "train_same_delay_only",
        "query_pool": "test_all_delays",
        "k_neighbors": K_NEIGHBORS,
        "zip_testzip": zip_test,
    }


def main() -> None:
    clean_out_dir()
    print("[v252] input-similarity future-divergence audit")
    print("[v252] fixed input model=v250_minimal_lateral7; no retraining; train same-delay neighbors only")

    loaded = load_fixed_inputs()
    selection = loaded["selection"]
    data = loaded["data"]
    x_flat = loaded["x_flat"]
    y_true = loaded["y_true"]
    pred_v241 = loaded["pred_v241"]
    pred_v250 = loaded["pred_v250"]
    sample_metrics = loaded["sample_metrics"]
    valid_mask = loaded["valid_mask"]
    split_check = loaded["split_check"]

    print("[v252] compute nearest-neighbor future divergence")
    audit, details = compute_neighbor_audit(
        manifest=data.manifest,
        x_flat=x_flat,
        y_true=y_true,
        pred_v241=pred_v241,
        pred_v250=pred_v250,
        sample_metrics=sample_metrics,
        valid_mask=valid_mask,
    )
    if audit.empty:
        raise AssertionError("v252 未生成任何近邻审计样本")

    summary = summarize_by_bucket_delay(audit)
    corr = error_ambiguity_correlations(audit)
    overlap = overlap_table(audit)
    casebook = build_casebook_index(audit)

    print("[v252] write tables and figures")
    write_csv(audit, TABLES / "v252_neighbor_divergence_by_sample.csv")
    write_csv(details, TABLES / "v252_neighbor_detail.csv")
    write_csv(summary, TABLES / "v252_summary_by_delay_bucket.csv")
    write_csv(corr, TABLES / "v252_error_ambiguity_correlation.csv")
    write_csv(overlap, TABLES / "v252_high_ambiguity_error_overlap.csv")
    write_csv(casebook, TABLES / "v252_casebook_index.csv")
    write_csv(split_check, TABLES / "v252_split_integrity_check.csv")

    high_ambiguity_cases = casebook.copy()
    worst_regression_cases = casebook[casebook["delta_tail_rmse_v250_minus_v241"] > 0].sort_values(
        "delta_tail_rmse_v250_minus_v241", ascending=False
    )
    if worst_regression_cases.empty:
        worst_regression_cases = casebook.head(6)

    figures = [
        plot_error_scatter(audit),
        plot_error_group_bar(audit),
        plot_delay_summary(summary),
        plot_casebook(
            high_ambiguity_cases,
            data,
            y_true,
            pred_v241,
            pred_v250,
            valid_mask,
            "v252 casebook: 锚点前相似，但锚点后真实未来分叉",
            FIGURES / "v252_casebook_high_error_high_ambiguity.png",
            n_cases=6,
        ),
        plot_casebook(
            worst_regression_cases,
            data,
            y_true,
            pred_v241,
            pred_v250,
            valid_mask,
            "v252 casebook: v250 退化样本的输入近邻与未来分叉",
            FIGURES / "v252_casebook_worst_regression_neighbors.png",
            n_cases=6,
        ),
    ]

    write_input_hashes()
    write_file_inventory()
    write_report(selection, audit, summary, corr, overlap, casebook, figures)
    write_file_inventory()
    zip_test = make_zip()
    guardrail = build_guardrail(split_check, zip_test)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v252 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    all_summary = summary[summary["bucket"].eq("all") & summary["delay_ms"].eq("all_delays")].iloc[0]
    bad_summary = summary[summary["bucket"].eq("bad_top10_v250") & summary["delay_ms"].eq("all_delays")].iloc[0]
    corr_main = corr[
        corr["subset"].eq("all_delays")
        & corr["x_metric"].eq("neighbor_future_pairwise_rmse_mean")
        & corr["y_metric"].eq("tail_rmse_v250")
    ].iloc[0]
    print(
        "[v252] all_pairwise={:.6f} bad250_pairwise={:.6f} corr_spearman={:.6f}".format(
            float(all_summary["neighbor_future_pairwise_rmse_mean"]),
            float(bad_summary["neighbor_future_pairwise_rmse_mean"]),
            float(corr_main["spearman"]),
        )
    )
    print(f"[v252] report={REPORTS / 'v252_input_similarity_future_divergence_cn.md'}")
    print(f"[v252] zip={ZIP_PATH}")


if __name__ == "__main__":
    main()
