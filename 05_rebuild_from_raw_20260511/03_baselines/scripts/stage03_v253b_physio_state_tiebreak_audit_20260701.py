#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v253b physio/style state tie-break audit。

本轮目标：
- 回应用户修正后的假设：生理状态/驾驶风格不是简单拼进输入距离，而是在车辆锚点前相似时，
  提供驾驶员状态区别，帮助从多个车辆相似候选里挑出更接近未来的候选。
- 固定 v250/v252/v253a，不训练预测模型。
- 对每个 test sample：先用 vehicle-only 输入在同 delay train 中找 top-K 车辆相似候选；
  再只在这个候选池内，用生理/风格距离做 tie-break；
  检查 tie-break 后选中的候选未来是否比 vehicle 最近邻更接近 query 未来。

关键边界：
- 当前 split 是 subject-disjoint，test 被试不在 train 中；因此本轮不能检验“同一个驾驶员的个体识别”，
  只能检验“跨驾驶员的生理/风格状态相似性是否能解释未来差异”。
- tie-break 评估会使用 query 未来计算结果好坏，但这只作为诊断评价，不作为部署策略输入。
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

V253A_SCRIPT = BASELINES / "scripts" / "stage03_v253_state_signal_disambiguation_audit_20260701.py"
V253A_DIR = BASELINES / "v253_state_signal_disambiguation_audit_20260701"
STYLE_FEATURES = V253A_DIR / "tables" / "v253a_current_style_features_last60_guard3.csv"
PHYSIO_FEATURES = V253A_DIR / "tables" / "v253a_current_physio_features_1hz.csv"

OUT = BASELINES / "v253b_physio_state_tiebreak_audit_20260701"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v253b_physio_state_tiebreak_audit_20260701_pack.zip"

POOL_K = 60
SEED = 2532
DELAY_MS = [0, 200, 400, 600, 800, 1000]
STRATEGY_ORDER = [
    "vehicle_rank1",
    "style_nearest_in_vehicle_pool",
    "physio_recent_nearest_in_vehicle_pool",
    "physio_guarded_nearest_in_vehicle_pool",
    "style_physio_nearest_in_vehicle_pool",
    "oracle_best_future_in_vehicle_pool",
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
    """按路径导入前序脚本，复用 v253a/v252 的数据构造与标准化函数。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V253A = import_module_from_path("stage03_v253a_for_v253b", V253A_SCRIPT)
V252 = V253A.V252


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v253b 自己的输出。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 统一使用 utf-8-sig。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算文件 SHA256。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    """计算单个候选池内距离与未来误差的 Spearman。"""

    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 5:
        return math.nan, int(mask.sum())
    xs = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    ys = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    if float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
        return math.nan, int(mask.sum())
    return float(np.corrcoef(xs, ys)[0, 1]), int(mask.sum())


def dist_to_pool(x: np.ndarray, qi: int, pool_idx: np.ndarray) -> np.ndarray:
    """计算 query 到候选池的 RMSE 距离。"""

    if x.shape[1] == 0:
        return np.full(len(pool_idx), np.nan, dtype=float)
    diff = x[pool_idx] - x[qi][None, :]
    return np.sqrt(np.mean(np.square(diff), axis=1))


def future_rmse_to_query(
    y_true: np.ndarray,
    valid_mask: np.ndarray,
    qi: int,
    cand_idx: np.ndarray,
    delay_ms: int,
) -> np.ndarray:
    """计算候选未来和 query 真实未来的 RMSE。"""

    horizon = V252.future_horizon_mask(delay_ms)
    q_valid = valid_mask[qi] & horizon
    out = np.full(len(cand_idx), np.nan, dtype=float)
    for j, ni in enumerate(cand_idx):
        mask = q_valid & valid_mask[ni]
        if int(mask.sum()) < 3:
            continue
        out[j] = float(np.sqrt(np.mean(np.square(y_true[ni, mask] - y_true[qi, mask]))))
    return out


def bucket_flags(row: pd.Series) -> Dict[str, bool]:
    """取 test 样本分层标签。"""

    return {
        "is_all": True,
        "is_bad_top10_v241": bool(row["bad_top10_v241"]),
        "is_bad_top10_v250": bool(row["bad_top10_v250"]),
        "is_normal_predictable": bool(row["is_normal_predictable"]),
        "is_observe_later_like": bool(row["is_observe_later_like"]),
        "is_strong_steer": bool(row["is_strong_steer"]),
        "is_reverse_or_multi_correction": bool(row["is_reverse_or_multi_correction"]),
    }


def load_aux_blocks(manifest: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """读取 v253a 生成的当前样本 style/physio 特征，并按 train-only 标准化。"""

    if not STYLE_FEATURES.exists():
        raise FileNotFoundError(f"缺少 v253a style features：{STYLE_FEATURES}")
    if not PHYSIO_FEATURES.exists():
        raise FileNotFoundError(f"缺少 v253a physio features：{PHYSIO_FEATURES}")
    style_df = pd.read_csv(STYLE_FEATURES, encoding="utf-8-sig")
    physio_df = pd.read_csv(PHYSIO_FEATURES, encoding="utf-8-sig")
    if len(style_df) != len(manifest) or len(physio_df) != len(manifest):
        raise AssertionError("v253a style/physio 特征行数与当前 manifest 不一致")

    train_mask = manifest["split"].astype(str).eq("train").to_numpy()
    style_x, style_scaler = V253A.standardize_aux_features(style_df, train_mask, include_prefixes=("style_",))
    physio_recent_x, physio_recent_scaler = V253A.standardize_aux_features(
        physio_df,
        train_mask,
        include_prefixes=("physio_recent_pre2_0_", "physio_delta_recent_minus_guard_", "physio_recording_"),
    )
    physio_guard_x, physio_guard_scaler = V253A.standardize_aux_features(
        physio_df,
        train_mask,
        include_prefixes=("physio_guard_pre5_pre2_", "physio_recording_"),
    )
    both_x = np.concatenate([style_x, physio_recent_x], axis=1)
    scaler = pd.concat(
        [
            style_scaler.assign(feature_block="style"),
            physio_recent_scaler.assign(feature_block="physio_recent"),
            physio_guard_scaler.assign(feature_block="physio_guard"),
        ],
        ignore_index=True,
    )
    return style_x, physio_recent_x, physio_guard_x, both_x, scaler


def run_tiebreak_audit() -> Dict[str, pd.DataFrame]:
    """运行车辆相似池内的生理/风格 tie-break 审计。"""

    loaded = V252.load_fixed_inputs()
    data = loaded["data"]
    manifest = data.manifest.copy()
    base_x = loaded["x_flat"].astype(np.float32)
    y_true = loaded["y_true"]
    sample_metrics = loaded["sample_metrics"].copy()
    # bad_top10_v250 是 v253/v252 审计层的派生标签，固定输入表不保证自带；
    # 这里仅用 test 集 tail_rmse_v250 的 90% 分位数重建，保证后续分层口径一致。
    test_tail_v250 = pd.to_numeric(
        sample_metrics.loc[sample_metrics["split"].eq("test"), "tail_rmse_v250"],
        errors="coerce",
    )
    v250_q90 = float(np.nanquantile(test_tail_v250.to_numpy(dtype=float), 0.90))
    sample_metrics["bad_top10_v250"] = (
        pd.to_numeric(sample_metrics["tail_rmse_v250"], errors="coerce") >= v250_q90
    )
    valid_mask = loaded["valid_mask"]
    split_check = loaded["split_check"]

    style_x, physio_recent_x, physio_guard_x, both_x, scaler = load_aux_blocks(manifest)
    split = manifest["split"].astype(str).to_numpy()
    delays = manifest["delay_ms"].astype(int).to_numpy()
    subjects = manifest["subject"].astype(str).to_numpy()
    recordings = manifest["recording"].astype(str).to_numpy()

    strategy_to_aux = {
        "vehicle_rank1": None,
        "style_nearest_in_vehicle_pool": style_x,
        "physio_recent_nearest_in_vehicle_pool": physio_recent_x,
        "physio_guarded_nearest_in_vehicle_pool": physio_guard_x,
        "style_physio_nearest_in_vehicle_pool": both_x,
        "oracle_best_future_in_vehicle_pool": None,
    }

    rows: List[Dict[str, object]] = []
    corr_rows: List[Dict[str, object]] = []
    pool_rows: List[Dict[str, object]] = []

    for delay in DELAY_MS:
        query_idx = np.where((split == "test") & (delays == delay))[0]
        train_idx = np.where((split == "train") & (delays == delay))[0]
        x_train = base_x[train_idx]
        for qi in query_idx:
            vehicle_dist_all = np.sqrt(np.mean(np.square(x_train - base_x[qi][None, :]), axis=1))
            order = np.argsort(vehicle_dist_all, kind="mergesort")[: min(POOL_K, len(train_idx))]
            pool_idx = train_idx[order]
            vehicle_dist = vehicle_dist_all[order]
            future_rmse = future_rmse_to_query(y_true, valid_mask, qi, pool_idx, delay)

            style_dist = dist_to_pool(style_x, qi, pool_idx)
            physio_recent_dist = dist_to_pool(physio_recent_x, qi, pool_idx)
            physio_guard_dist = dist_to_pool(physio_guard_x, qi, pool_idx)
            both_dist = dist_to_pool(both_x, qi, pool_idx)

            dist_map = {
                "style": style_dist,
                "physio_recent": physio_recent_dist,
                "physio_guard": physio_guard_dist,
                "style_physio": both_dist,
                "vehicle": vehicle_dist,
            }
            for name, dist in dist_map.items():
                rho, n = safe_spearman(dist, future_rmse)
                corr_rows.append(
                    {
                        "row_index": int(qi),
                        "event_uid": str(manifest.iloc[qi]["event_uid"]),
                        "delay_ms": int(delay),
                        "distance_block": name,
                        "spearman_distance_vs_future_rmse": rho,
                        "pool_valid_n": n,
                    }
                )

            finite_future = np.isfinite(future_rmse)
            if not finite_future.any():
                continue
            vehicle_choice = int(np.nanargmin(vehicle_dist))
            oracle_choice = int(np.nanargmin(future_rmse))
            choice_by_strategy: Dict[str, int] = {
                "vehicle_rank1": vehicle_choice,
                "oracle_best_future_in_vehicle_pool": oracle_choice,
            }
            for strategy, aux_x in strategy_to_aux.items():
                if aux_x is None or strategy in choice_by_strategy:
                    continue
                if strategy.startswith("style_nearest"):
                    d = style_dist
                elif strategy.startswith("physio_recent"):
                    d = physio_recent_dist
                elif strategy.startswith("physio_guarded"):
                    d = physio_guard_dist
                elif strategy.startswith("style_physio"):
                    d = both_dist
                else:
                    d = vehicle_dist
                valid = np.isfinite(d)
                if valid.any():
                    choice_by_strategy[strategy] = int(np.where(valid)[0][np.argmin(d[valid])])
                else:
                    choice_by_strategy[strategy] = vehicle_choice

            sm = sample_metrics.iloc[qi]
            flags = bucket_flags(sm)
            for strategy in STRATEGY_ORDER:
                choice = choice_by_strategy[strategy]
                ni = int(pool_idx[choice])
                selected_future = float(future_rmse[choice])
                vehicle_future = float(future_rmse[vehicle_choice])
                oracle_future = float(future_rmse[oracle_choice])
                row = {
                    "strategy": strategy,
                    "row_index": int(qi),
                    "event_uid": str(manifest.iloc[qi]["event_uid"]),
                    "subject": str(subjects[qi]),
                    "recording": str(recordings[qi]),
                    "delay_ms": int(delay),
                    "pool_k": int(len(pool_idx)),
                    "selected_neighbor_row_index": ni,
                    "selected_neighbor_event_uid": str(manifest.iloc[ni]["event_uid"]),
                    "selected_neighbor_subject": str(subjects[ni]),
                    "selected_neighbor_recording": str(recordings[ni]),
                    "selected_neighbor_vehicle_rank": int(choice + 1),
                    "selected_future_rmse_to_query": selected_future,
                    "vehicle_rank1_future_rmse_to_query": vehicle_future,
                    "oracle_best_future_rmse_to_query": oracle_future,
                    "delta_selected_minus_vehicle_rank1": selected_future - vehicle_future,
                    "delta_selected_minus_oracle": selected_future - oracle_future,
                    "selected_same_subject": bool(subjects[ni] == subjects[qi]),
                    "selected_same_recording": bool(recordings[ni] == recordings[qi]),
                    "tail_rmse_v250": float(sm["tail_rmse_v250"]),
                    "tail_rmse_v241": float(sm["tail_rmse_v241"]),
                }
                row.update(flags)
                rows.append(row)

            # 只保留每个 query 的 top pool 摘要，避免输出过大。
            pool_rows.append(
                {
                    "row_index": int(qi),
                    "event_uid": str(manifest.iloc[qi]["event_uid"]),
                    "delay_ms": int(delay),
                    "pool_k": int(len(pool_idx)),
                    "vehicle_rank1_future_rmse": float(future_rmse[vehicle_choice]),
                    "pool_future_rmse_mean": float(np.nanmean(future_rmse)),
                    "pool_future_rmse_min_oracle": float(future_rmse[oracle_choice]),
                    "pool_future_rmse_std": float(np.nanstd(future_rmse)),
                    "pool_same_subject_rate": float(np.mean(subjects[pool_idx] == subjects[qi])),
                    "pool_same_recording_rate": float(np.mean(recordings[pool_idx] == recordings[qi])),
                    "pool_neighbor_event_uids": "|".join(manifest.iloc[pool_idx]["event_uid"].astype(str).tolist()),
                }
            )

    per_strategy = pd.DataFrame(rows)
    corr = pd.DataFrame(corr_rows)
    pool = pd.DataFrame(pool_rows)
    return {
        "per_strategy": per_strategy,
        "correlation": corr,
        "pool": pool,
        "split_check": split_check,
        "scaler": scaler,
        "subject_split": pd.crosstab(manifest["subject"], manifest["split"]).reset_index(),
    }


def summarize(per_strategy: pd.DataFrame) -> pd.DataFrame:
    """按策略/bucket/delay 汇总 tie-break 是否改善。"""

    rows: List[Dict[str, object]] = []
    bucket_cols = [
        ("all", "is_all"),
        ("bad_top10_v250", "is_bad_top10_v250"),
        ("bad_top10_v241", "is_bad_top10_v241"),
        ("strong_steer", "is_strong_steer"),
        ("observe_later_like", "is_observe_later_like"),
        ("normal_predictable", "is_normal_predictable"),
    ]
    for bucket, col in bucket_cols:
        base_mask = per_strategy[col].astype(bool)
        for delay_label, delay_mask in [("all_delays", np.ones(len(per_strategy), dtype=bool))] + [
            (str(d), per_strategy["delay_ms"].eq(d).to_numpy()) for d in DELAY_MS
        ]:
            for strategy in STRATEGY_ORDER:
                sub = per_strategy[base_mask & delay_mask & per_strategy["strategy"].eq(strategy)].copy()
                if sub.empty:
                    continue
                delta = sub["delta_selected_minus_vehicle_rank1"].to_numpy(dtype=float)
                selected = sub["selected_future_rmse_to_query"].to_numpy(dtype=float)
                vehicle = sub["vehicle_rank1_future_rmse_to_query"].to_numpy(dtype=float)
                oracle = sub["oracle_best_future_rmse_to_query"].to_numpy(dtype=float)
                rows.append(
                    {
                        "bucket": bucket,
                        "delay_ms": delay_label,
                        "strategy": strategy,
                        "n": int(len(sub)),
                        "event_n": int(sub["event_uid"].nunique()),
                        "selected_future_rmse_mean": float(np.nanmean(selected)),
                        "vehicle_rank1_future_rmse_mean": float(np.nanmean(vehicle)),
                        "oracle_future_rmse_mean": float(np.nanmean(oracle)),
                        "delta_selected_minus_vehicle_mean": float(np.nanmean(delta)),
                        "delta_selected_minus_vehicle_median": float(np.nanmedian(delta)),
                        "improve_rate_vs_vehicle_rank1": float(np.nanmean(delta < 0.0)),
                        "same_subject_rate": float(sub["selected_same_subject"].mean()),
                        "same_recording_rate": float(sub["selected_same_recording"].mean()),
                        "selected_neighbor_vehicle_rank_mean": float(sub["selected_neighbor_vehicle_rank"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def summarize_correlations(corr: pd.DataFrame, per_strategy: pd.DataFrame) -> pd.DataFrame:
    """按 bucket 汇总候选池内距离和未来误差的相关。"""

    flags = per_strategy[per_strategy["strategy"].eq("vehicle_rank1")][
        ["row_index", "is_bad_top10_v250", "is_bad_top10_v241", "is_strong_steer", "is_observe_later_like", "is_normal_predictable"]
    ].drop_duplicates("row_index")
    work = corr.merge(flags, on="row_index", how="left")
    rows: List[Dict[str, object]] = []
    bucket_defs = [
        ("all", np.ones(len(work), dtype=bool)),
        ("bad_top10_v250", work["is_bad_top10_v250"].fillna(False).astype(bool).to_numpy()),
        ("strong_steer", work["is_strong_steer"].fillna(False).astype(bool).to_numpy()),
        ("observe_later_like", work["is_observe_later_like"].fillna(False).astype(bool).to_numpy()),
    ]
    for bucket, mask in bucket_defs:
        for block, sub in work[mask].groupby("distance_block"):
            vals = sub["spearman_distance_vs_future_rmse"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            rows.append(
                {
                    "bucket": bucket,
                    "distance_block": block,
                    "n_query": int(vals.size),
                    "mean_spearman_distance_vs_future_rmse": float(np.mean(vals)),
                    "median_spearman_distance_vs_future_rmse": float(np.median(vals)),
                    "positive_rate": float(np.mean(vals > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def plot_badtop10(summary: pd.DataFrame) -> Path:
    """画 bad_top10_v250 all-delay tie-break 对比。"""

    path = FIGURES / "v253b_badtop10_tiebreak_selected_future_rmse.png"
    sub = summary[summary["bucket"].eq("bad_top10_v250") & summary["delay_ms"].eq("all_delays")].copy()
    if sub.empty:
        return path
    sub["strategy"] = pd.Categorical(sub["strategy"], categories=STRATEGY_ORDER, ordered=True)
    sub = sub.sort_values("strategy")
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x, sub["selected_future_rmse_mean"], color="#4C78A8")
    ax.axhline(float(sub[sub["strategy"].eq("vehicle_rank1")]["selected_future_rmse_mean"].iloc[0]), color="black", linestyle="--", linewidth=1, label="vehicle rank1")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_in_vehicle_pool", "").replace("_", "\n") for s in sub["strategy"]], fontsize=8)
    ax.set_ylabel("selected neighbor future RMSE to query")
    ax.set_title("v253b: 车辆相似候选池内，用生理/风格做 tie-break 是否更接近未来")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_delta(summary: pd.DataFrame) -> Path:
    """画相对 vehicle rank1 的 delta。"""

    path = FIGURES / "v253b_tiebreak_delta_vs_vehicle_rank1.png"
    sub = summary[
        summary["bucket"].isin(["all", "bad_top10_v250", "strong_steer", "observe_later_like"])
        & summary["delay_ms"].eq("all_delays")
        & ~summary["strategy"].isin(["vehicle_rank1", "oracle_best_future_in_vehicle_pool"])
    ].copy()
    if sub.empty:
        return path
    strategies = [s for s in STRATEGY_ORDER if s in set(sub["strategy"]) and s not in {"vehicle_rank1", "oracle_best_future_in_vehicle_pool"}]
    buckets = ["all", "bad_top10_v250", "strong_steer", "observe_later_like"]
    x = np.arange(len(strategies))
    width = 0.18
    fig, ax = plt.subplots(figsize=(14, 5.8))
    for i, bucket in enumerate(buckets):
        vals = (
            sub[sub["bucket"].eq(bucket)]
            .set_index("strategy")
            .reindex(strategies)["delta_selected_minus_vehicle_mean"]
            .to_numpy(dtype=float)
        )
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=bucket)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_nearest_in_vehicle_pool", "").replace("_", "\n") for s in strategies], fontsize=8)
    ax.set_ylabel("selected future RMSE delta vs vehicle rank1（负数=更好）")
    ax.set_title("v253b: 生理/风格只作为车辆相似候选池内 tie-break 的效果")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    summary: pd.DataFrame,
    corr_summary: pd.DataFrame,
    subject_split: pd.DataFrame,
    figures: List[Path],
) -> None:
    """写中文报告。"""

    lines: List[str] = []
    lines.append("# v253b 生理/驾驶风格状态 tie-break 审计")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("本轮不再把生理/风格简单拼进全局输入距离，而是先用 vehicle-only 找同 delay 的车辆相似候选池，再看生理/风格能否在这个池内挑出未来更接近 query 的样本。")
    lines.append("")
    lines.append("## 关键边界")
    lines.append("")
    lines.append("- 不训练预测模型，不修改 v250/v252/v253a。")
    lines.append("- 当前 split 是 subject-disjoint：test 被试不在 train 中，因此不能验证同一驾驶员个体记忆，只能验证跨驾驶员状态相似性。")
    lines.append("- 未来 RMSE 只用于诊断评价 tie-break 是否挑对，不作为部署输入。")
    lines.append("")
    lines.append("## Subject Split")
    lines.append("")
    lines.append(subject_split.to_markdown(index=False))
    lines.append("")
    lines.append("## 关键结果")
    lines.append("")
    keep = summary[
        summary["bucket"].isin(["all", "bad_top10_v250", "strong_steer", "observe_later_like"])
        & summary["delay_ms"].isin(["all_delays", "0"])
    ].copy()
    show_cols = [
        "bucket",
        "delay_ms",
        "strategy",
        "n",
        "selected_future_rmse_mean",
        "delta_selected_minus_vehicle_mean",
        "improve_rate_vs_vehicle_rank1",
        "selected_neighbor_vehicle_rank_mean",
    ]
    lines.append(keep[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 候选池内距离-未来误差相关")
    lines.append("")
    lines.append(corr_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    lines.append("- 如果生理/风格能提供驾驶员状态区别，应该看到 tie-break 策略的 `delta_selected_minus_vehicle_mean < 0`，尤其在 bad_top10_v250 上。")
    lines.append("- 如果距离-未来误差相关为正，说明生理/风格距离越近，未来也越近；若接近 0 或负值，则当前状态表示没有提供有效排序。")
    lines.append("- oracle 只表示车辆相似候选池内还存在更好未来上限，不代表可部署。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    lines.append("")
    (REPORTS / "v253b_physio_state_tiebreak_audit_cn.md").write_text("\n".join(lines), encoding="utf-8")


def write_input_hashes() -> None:
    """记录输入哈希。"""

    paths = [V253A_SCRIPT, STYLE_FEATURES, PHYSIO_FEATURES]
    rows = []
    for path in paths:
        if path.exists():
            rows.append({"path": str(path), "sha256": file_sha256(path), "bytes": int(path.stat().st_size)})
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    """记录输出文件清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "bytes": int(path.stat().st_size)})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> str | None:
    """打包产物。"""

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
    """生成 guardrail。"""

    cross = int(split_check["same_event_uid_cross_split"].sum()) if "same_event_uid_cross_split" in split_check.columns else 0
    return {
        "pass": bool(cross == 0 and zip_test is None),
        "same_event_uid_cross_split_count": cross,
        "retrained_model": False,
        "test_used_for_model_selection": False,
        "candidate_pool": f"same_delay_train_vehicle_top{POOL_K}",
        "future_used_only_for_diagnostic_scoring": True,
        "zip_testzip": zip_test,
    }


def main() -> None:
    clean_out_dir()
    print("[v253b] physio/style state tie-break audit")
    print(f"[v253b] vehicle-only top-{POOL_K} same-delay pool, then aux tie-break")

    outputs = run_tiebreak_audit()
    per_strategy = outputs["per_strategy"]
    corr = outputs["correlation"]
    pool = outputs["pool"]
    split_check = outputs["split_check"]
    scaler = outputs["scaler"]
    subject_split = outputs["subject_split"]
    summary = summarize(per_strategy)
    corr_summary = summarize_correlations(corr, per_strategy)
    figures = [plot_badtop10(summary), plot_delta(summary)]

    write_csv(per_strategy, TABLES / "v253b_tiebreak_per_strategy.csv")
    write_csv(summary, TABLES / "v253b_tiebreak_summary.csv")
    write_csv(corr, TABLES / "v253b_pool_distance_future_correlation_by_sample.csv")
    write_csv(corr_summary, TABLES / "v253b_pool_distance_future_correlation_summary.csv")
    write_csv(pool, TABLES / "v253b_vehicle_candidate_pool_summary.csv")
    write_csv(scaler, TABLES / "v253b_aux_train_scaler.csv")
    write_csv(subject_split, TABLES / "v253b_subject_split_table.csv")
    write_csv(split_check, TABLES / "v253b_split_integrity_check.csv")
    write_input_hashes()
    write_report(summary, corr_summary, subject_split, figures)
    write_file_inventory()
    zip_test = make_zip()
    guardrail = build_guardrail(split_check, zip_test)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v253b guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    bad = summary[summary["bucket"].eq("bad_top10_v250") & summary["delay_ms"].eq("all_delays")].copy()
    base = bad[bad["strategy"].eq("vehicle_rank1")].iloc[0]
    best_non_oracle = bad[~bad["strategy"].isin(["oracle_best_future_in_vehicle_pool"])].sort_values("selected_future_rmse_mean").iloc[0]
    oracle = bad[bad["strategy"].eq("oracle_best_future_in_vehicle_pool")].iloc[0]
    print(
        "[v253b] bad_top10 vehicle={:.6f}; best_non_oracle={} {:.6f} delta={:.6f}; oracle={:.6f}".format(
            float(base["selected_future_rmse_mean"]),
            str(best_non_oracle["strategy"]),
            float(best_non_oracle["selected_future_rmse_mean"]),
            float(best_non_oracle["delta_selected_minus_vehicle_mean"]),
            float(oracle["selected_future_rmse_mean"]),
        )
    )
    print(f"[v253b] report={REPORTS / 'v253b_physio_state_tiebreak_audit_cn.md'}")
    print(f"[v253b] zip={ZIP_PATH}")


if __name__ == "__main__":
    main()
