#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v251 locked robustness audit for v250_minimal_lateral7。

本轮目标：
- 固定 v250 的 best validation channel model：v250_minimal_lateral7；
- 不重新训练、不改通道、不调阈值、不用 test 做模型选择；
- 在 locked test 上做稳健性审计：bucket/delay、subject、recording、event-level bootstrap CI；
- 输出逐样本回退、bad_top10 casebook 和下一步决策。

边界：
- 本轮是审计，不是新模型；
- 不做 anchor selector、gate/router、response-type hard routing，不删除样本；
- v250 是否能成为主线，需要看稳健性是否支持，而不是只看均值改善。
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

OUT = BASELINES / "v251_locked_robustness_v250_20260701"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v251_locked_robustness_v250_20260701_pack.zip"

DELAY_MS = [0, 200, 400, 600, 800, 1000]
BUCKET_ORDER = [
    "all",
    "normal_predictable",
    "observe_later_like",
    "strong_steer",
    "reverse_or_multi_correction",
    "bad_top10_v241",
]
SEED = 251
N_BOOT = 4000

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    """按路径导入前序脚本，复用已验证的数据和 shape 指标函数。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V250 = import_module_from_path("stage03_v250_history_channel_ablation_20260630_for_v251", V250_SCRIPT)
V249 = V250.V249
V238 = V250.V238
FUTURE_GRID = V238.FUTURE_GRID.astype(np.float32)


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v251 自己的输出。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 统一使用 utf-8-sig，方便 Windows Excel 打开中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算文件 SHA256。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_locked_inputs() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, str, pd.DataFrame]:
    """读取 v250 locked prediction、v241 prediction 和 manifest。"""

    if not V250_PRED.exists():
        raise FileNotFoundError(f"缺少 v250 prediction：{V250_PRED}")
    if not V250_SELECTION.exists():
        raise FileNotFoundError(f"缺少 v250 selection：{V250_SELECTION}")
    selection = pd.read_csv(V250_SELECTION, encoding="utf-8-sig")
    accepted = selection[selection["accepted_as_channel_candidate"].astype(bool)].copy()
    if accepted.empty:
        raise AssertionError("v250 selection 中没有 accepted channel candidate，不能进入 v251 locked audit。")
    best_name = str(selection.iloc[0]["model_name"])
    if best_name != "v250_minimal_lateral7":
        raise AssertionError(f"v251 预期审计 v250_minimal_lateral7，实际 best={best_name}")

    data = V238.load_v236_data()
    with np.load(V250_PRED, allow_pickle=False) as pred:
        y_true = pred["y_true_steering_delta"].astype(np.float32)
        pred_v241 = pred["pred_v241_steering_delta"].astype(np.float32)
        pred_v250 = pred[f"pred_{best_name}_steering_delta"].astype(np.float32)
        best_from_npz = str(pred["best_channel_model"][0])
    if best_from_npz != best_name:
        raise AssertionError(f"npz best={best_from_npz} 与 selection best={best_name} 不一致")
    if y_true.shape != pred_v241.shape or y_true.shape != pred_v250.shape:
        raise AssertionError(f"预测 shape 不一致：{y_true.shape}, {pred_v241.shape}, {pred_v250.shape}")
    if len(data.manifest) != y_true.shape[0]:
        raise AssertionError("manifest 行数与 prediction 不一致")
    return data.manifest.copy(), y_true, pred_v241, pred_v250, best_name, selection


def horizon_masks(delay_ms: int) -> Tuple[np.ndarray, np.ndarray]:
    """返回 original_remaining horizon/tail mask。"""

    original_rel = delay_ms / 1000.0 + FUTURE_GRID
    horizon = original_rel <= 2.0 + 1e-9
    tail = horizon & (original_rel >= 1.0 - 1e-9)
    return horizon, tail


def bucket_masks(manifest: pd.DataFrame, bad_top10_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """生成审计 bucket。"""

    reverse_multi = (
        manifest["reverse"].astype(bool).to_numpy()
        | manifest["multi_correction"].astype(bool).to_numpy()
        | manifest["zero_cross"].astype(bool).to_numpy()
    )
    observe = manifest["observe_later_like"].astype(bool).to_numpy()
    normal = manifest["normal_curve"].astype(bool).to_numpy() & ~observe
    return {
        "all": np.ones(len(manifest), dtype=bool),
        "normal_predictable": normal,
        "observe_later_like": observe,
        "strong_steer": manifest["strong_steer"].astype(bool).to_numpy(),
        "reverse_or_multi_correction": reverse_multi,
        "bad_top10_v241": bad_top10_mask,
    }


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE。"""

    if a.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(a - b))))


def compute_sample_metrics(
    manifest: pd.DataFrame,
    y_true: np.ndarray,
    pred_v241: np.ndarray,
    pred_v250: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """生成逐 rolling sample 的 original_remaining RMSE/shape delta。"""

    valid_matrix, _ = V238.build_original_remaining_mask(manifest)
    split = manifest["split"].astype(str).to_numpy()
    delay_values = manifest["delay_ms"].astype(int).to_numpy()

    v241_rmse = np.full(len(manifest), np.nan, dtype=float)
    for i in range(len(manifest)):
        valid = valid_matrix[i]
        if np.any(valid):
            v241_rmse[i] = rmse(y_true[i, valid], pred_v241[i, valid])

    bad_top10 = np.zeros(len(manifest), dtype=bool)
    bad_thresholds: Dict[str, float] = {}
    for split_name in ["train", "val", "test"]:
        split_mask = split == split_name
        base = v241_rmse[split_mask & np.isfinite(v241_rmse)]
        if base.size == 0:
            continue
        threshold = float(np.quantile(base, 0.90))
        bad_thresholds[split_name] = threshold
        bad_top10 |= split_mask & (v241_rmse >= threshold)

    buckets = bucket_masks(manifest, bad_top10)
    rows: List[Dict[str, object]] = []
    for i in range(len(manifest)):
        delay = int(delay_values[i])
        horizon, tail = horizon_masks(delay)
        valid = valid_matrix[i] & horizon
        tail_valid = valid_matrix[i] & tail
        if not np.any(valid):
            continue
        y = y_true[i]
        p241 = pred_v241[i]
        p250 = pred_v250[i]
        sample_v241 = rmse(y[valid], p241[valid])
        sample_v250 = rmse(y[valid], p250[valid])
        tail_v241 = rmse(y[tail_valid], p241[tail_valid]) if np.any(tail_valid) else math.nan
        tail_v250 = rmse(y[tail_valid], p250[tail_valid]) if np.any(tail_valid) else math.nan
        shape_v241 = V249.shape_metrics_np(y[valid], p241[valid])
        shape_v250 = V249.shape_metrics_np(y[valid], p250[valid])
        row = {
            "row_index": i,
            "rolling_sample_index": int(manifest.iloc[i]["rolling_sample_index"]),
            "event_uid": str(manifest.iloc[i]["event_uid"]),
            "subject": str(manifest.iloc[i]["subject"]),
            "recording": str(manifest.iloc[i]["recording"]),
            "split": str(manifest.iloc[i]["split"]),
            "delay_ms": delay,
            "sample_rmse_v241": sample_v241,
            "sample_rmse_v250": sample_v250,
            "delta_sample_rmse_v250_minus_v241": sample_v250 - sample_v241,
            "tail_rmse_v241": tail_v241,
            "tail_rmse_v250": tail_v250,
            "delta_tail_rmse_v250_minus_v241": tail_v250 - tail_v241 if np.isfinite(tail_v241) else math.nan,
            "range_ratio_v241": shape_v241["range_ratio"],
            "range_ratio_v250": shape_v250["range_ratio"],
            "delta_range_ratio_v250_minus_v241": shape_v250["range_ratio"] - shape_v241["range_ratio"],
            "slope_ratio_v241": shape_v241["slope_ratio"],
            "slope_ratio_v250": shape_v250["slope_ratio"],
            "delta_slope_ratio_v250_minus_v241": shape_v250["slope_ratio"] - shape_v241["slope_ratio"],
            "bad_top10_v241": bool(bad_top10[i]),
        }
        for bucket_name, mask in buckets.items():
            row[f"is_{bucket_name}"] = bool(mask[i])
        rows.append(row)
    return pd.DataFrame(rows), bad_thresholds


def summarize_group(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """对任意分组生成 paired delta 摘要。"""

    rows: List[Dict[str, object]] = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        delta_tail = pd.to_numeric(g["delta_tail_rmse_v250_minus_v241"], errors="coerce").to_numpy(dtype=float)
        delta_sample = pd.to_numeric(g["delta_sample_rmse_v250_minus_v241"], errors="coerce").to_numpy(dtype=float)
        delta_tail = delta_tail[np.isfinite(delta_tail)]
        delta_sample = delta_sample[np.isfinite(delta_sample)]
        if delta_tail.size == 0 or delta_sample.size == 0:
            continue
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "n": int(len(g)),
                "event_n": int(g["event_uid"].nunique()),
                "mean_tail_rmse_v241": float(g["tail_rmse_v241"].mean()),
                "mean_tail_rmse_v250": float(g["tail_rmse_v250"].mean()),
                "mean_delta_tail": float(delta_tail.mean()),
                "median_delta_tail": float(np.median(delta_tail)),
                "p90_delta_tail": float(np.quantile(delta_tail, 0.90)),
                "max_delta_tail": float(np.max(delta_tail)),
                "tail_improve_rate": float(np.mean(delta_tail < 0.0)),
                "mean_sample_rmse_v241": float(g["sample_rmse_v241"].mean()),
                "mean_sample_rmse_v250": float(g["sample_rmse_v250"].mean()),
                "mean_delta_sample": float(delta_sample.mean()),
                "sample_improve_rate": float(np.mean(delta_sample < 0.0)),
                "mean_delta_range_ratio": float(g["delta_range_ratio_v250_minus_v241"].mean()),
                "mean_delta_slope_ratio": float(g["delta_slope_ratio_v250_minus_v241"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def bucket_delay_summary(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    """生成 split/bucket/delay 层指标。"""

    rows: List[pd.DataFrame] = []
    for bucket in BUCKET_ORDER:
        flag = f"is_{bucket}"
        sub = sample_metrics[sample_metrics[flag].astype(bool)].copy()
        if sub.empty:
            continue
        out = summarize_group(sub, ["split", "delay_ms"])
        if not out.empty:
            out.insert(1, "bucket", bucket)
            rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def subject_delay_summary(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    """生成 test subject/bucket/delay 层稳健性表。"""

    test = sample_metrics[sample_metrics["split"].eq("test")].copy()
    rows: List[pd.DataFrame] = []
    for bucket in BUCKET_ORDER:
        sub = test[test[f"is_{bucket}"].astype(bool)].copy()
        if sub.empty:
            continue
        out = summarize_group(sub, ["subject", "delay_ms"])
        if not out.empty:
            out.insert(1, "bucket", bucket)
            rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def subject_summary(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    """生成 test subject/bucket 全 delay 汇总。"""

    test = sample_metrics[sample_metrics["split"].eq("test")].copy()
    rows: List[pd.DataFrame] = []
    for bucket in BUCKET_ORDER:
        sub = test[test[f"is_{bucket}"].astype(bool)].copy()
        if sub.empty:
            continue
        out = summarize_group(sub, ["subject"])
        if not out.empty:
            out.insert(1, "bucket", bucket)
            rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def recording_summary(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    """生成 test recording/bucket 全 delay 汇总。"""

    test = sample_metrics[sample_metrics["split"].eq("test")].copy()
    rows: List[pd.DataFrame] = []
    for bucket in BUCKET_ORDER:
        sub = test[test[f"is_{bucket}"].astype(bool)].copy()
        if sub.empty:
            continue
        out = summarize_group(sub, ["recording"])
        if not out.empty:
            out.insert(1, "bucket", bucket)
            rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def event_level_bootstrap(
    df: pd.DataFrame,
    value_col: str,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> Tuple[float, float, float, float]:
    """按 event_uid 做 paired bootstrap，返回均值和 95% CI。"""

    if df.empty:
        return math.nan, math.nan, math.nan, math.nan
    event_values = df.groupby("event_uid")[value_col].mean().dropna().to_numpy(dtype=float)
    event_values = event_values[np.isfinite(event_values)]
    if event_values.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(np.mean(event_values))
    if event_values.size == 1:
        return mean, mean, mean, float(mean < 0)
    samples = rng.choice(event_values, size=(n_boot, event_values.size), replace=True).mean(axis=1)
    lo = float(np.quantile(samples, 0.025))
    hi = float(np.quantile(samples, 0.975))
    prob_negative = float(np.mean(samples < 0.0))
    return mean, lo, hi, prob_negative


def bootstrap_ci_table(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    """生成 locked test 的 event-level bootstrap CI。"""

    rng = np.random.default_rng(SEED)
    test = sample_metrics[sample_metrics["split"].eq("test")].copy()
    rows: List[Dict[str, object]] = []
    for bucket in BUCKET_ORDER:
        bsub = test[test[f"is_{bucket}"].astype(bool)].copy()
        if bsub.empty:
            continue
        for delay in DELAY_MS + [-1]:
            if delay == -1:
                sub = bsub.copy()
                delay_label = "all_delays"
            else:
                sub = bsub[bsub["delay_ms"].eq(delay)].copy()
                delay_label = str(delay)
            if sub.empty:
                continue
            tail_mean, tail_lo, tail_hi, tail_prob = event_level_bootstrap(
                sub, "delta_tail_rmse_v250_minus_v241", rng
            )
            sample_mean, sample_lo, sample_hi, sample_prob = event_level_bootstrap(
                sub, "delta_sample_rmse_v250_minus_v241", rng
            )
            rows.append(
                {
                    "split": "test",
                    "bucket": bucket,
                    "delay_ms": delay_label,
                    "n": int(len(sub)),
                    "event_n": int(sub["event_uid"].nunique()),
                    "tail_delta_mean": tail_mean,
                    "tail_delta_ci95_low": tail_lo,
                    "tail_delta_ci95_high": tail_hi,
                    "tail_prob_delta_lt0": tail_prob,
                    "tail_ci_excludes_zero_negative": bool(np.isfinite(tail_hi) and tail_hi < 0.0),
                    "sample_delta_mean": sample_mean,
                    "sample_delta_ci95_low": sample_lo,
                    "sample_delta_ci95_high": sample_hi,
                    "sample_prob_delta_lt0": sample_prob,
                    "sample_ci_excludes_zero_negative": bool(np.isfinite(sample_hi) and sample_hi < 0.0),
                }
            )
    return pd.DataFrame(rows)


def robustness_decision(
    bucket_summary: pd.DataFrame,
    subj_summary: pd.DataFrame,
    boot: pd.DataFrame,
) -> pd.DataFrame:
    """根据 locked audit 生成下一步决策。"""

    test_summary = bucket_summary[bucket_summary["split"].eq("test")].copy()
    key_buckets = ["all", "normal_predictable", "observe_later_like", "strong_steer"]
    key = test_summary[test_summary["bucket"].isin(key_buckets)].copy()
    all_delay_key = boot[boot["delay_ms"].eq("all_delays") & boot["bucket"].isin(key_buckets)].copy()
    subj_key = subj_summary[subj_summary["bucket"].isin(key_buckets)].copy()

    all_bucket_delay_negative = bool((key["mean_delta_tail"] < 0).all()) if not key.empty else False
    boot_all_delay_pass = bool((all_delay_key["tail_delta_ci95_high"] < 0).all()) if not all_delay_key.empty else False
    subject_win_rate = float((subj_key["mean_delta_tail"] < 0).mean()) if not subj_key.empty else math.nan
    subject_pass = bool(np.isfinite(subject_win_rate) and subject_win_rate >= 0.80)

    pass_locked = bool(all_bucket_delay_negative and boot_all_delay_pass and subject_pass)
    if pass_locked:
        recommended = "v252_mainline_candidate_pack_or_subject_level_retest"
        decision_text = "pass_locked_robustness"
        reason = "All key test bucket/delay tail deltas are negative, all-delay event-level bootstrap CIs exclude zero, and subject win rate is high."
    else:
        recommended = "v251_case_review_before_mainline"
        decision_text = "diagnostic_only_until_review"
        reason = "At least one robustness guardrail did not pass; review subject/bucket regressions before promoting v250."

    return pd.DataFrame(
        [
            {
                "decision_item": "locked_robustness_pass",
                "decision": pass_locked,
                "reason": reason,
            },
            {
                "decision_item": "all_key_bucket_delay_tail_negative",
                "decision": all_bucket_delay_negative,
                "reason": "Requires every test delay in all/normal/observe_later/strong to have mean tail delta < 0.",
            },
            {
                "decision_item": "all_delay_bootstrap_ci_pass",
                "decision": boot_all_delay_pass,
                "reason": "Requires all-delay event-level bootstrap 95% CI upper bound < 0 for key buckets.",
            },
            {
                "decision_item": "subject_win_rate",
                "decision": subject_win_rate,
                "reason": "Fraction of subject/bucket summaries with mean tail delta < 0 across key buckets.",
            },
            {
                "decision_item": "formal_replacement_allowed",
                "decision": False,
                "reason": "v251 is locked robustness evidence; formal replacement still needs mainline packaging and final consistency audit.",
            },
            {
                "decision_item": "current_status",
                "decision": decision_text,
                "reason": "Robustness status for v250_minimal_lateral7.",
            },
            {
                "decision_item": "recommended_next_task",
                "decision": recommended,
                "reason": "Next bounded step after locked robustness audit.",
            },
        ]
    )


def plot_bucket_delay(bucket_summary: pd.DataFrame) -> Path:
    """绘制关键 test bucket/delay 的 tail delta。"""

    path = FIGURES / "v251_test_bucket_delay_tail_delta.png"
    test = bucket_summary[
        bucket_summary["split"].eq("test")
        & bucket_summary["bucket"].isin(["all", "normal_predictable", "observe_later_like", "strong_steer"])
    ].copy()
    if test.empty:
        return path
    labels = [f"{b}\n{d}ms" for b, d in zip(test["bucket"], test["delay_ms"])]
    x = np.arange(len(test))
    fig, ax = plt.subplots(figsize=(16, 5.5))
    colors = np.where(test["mean_delta_tail"].to_numpy() < 0, "#2ca02c", "#d62728")
    ax.bar(x, test["mean_delta_tail"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("v251 locked test: v250_minimal_lateral7 tail RMSE delta vs v241")
    ax.set_ylabel("mean tail RMSE delta（负数=优于 v241）")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_subject_summary(subj_summary: pd.DataFrame) -> Path:
    """绘制 subject/bucket all-delay tail delta。"""

    path = FIGURES / "v251_subject_bucket_tail_delta.png"
    keep = subj_summary[subj_summary["bucket"].isin(["all", "normal_predictable", "observe_later_like", "strong_steer"])].copy()
    if keep.empty:
        return path
    piv = keep.pivot_table(index="subject", columns="bucket", values="mean_delta_tail", aggfunc="first")
    piv = piv.reindex(columns=["all", "normal_predictable", "observe_later_like", "strong_steer"])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(piv.index))
    width = 0.18
    for i, col in enumerate(piv.columns):
        ax.bar(x + (i - 1.5) * width, piv[col], width=width, label=col)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("v251 subject-level robustness: all-delay tail RMSE delta")
    ax.set_ylabel("mean tail RMSE delta（负数=优于 v241）")
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_bootstrap_ci(boot: pd.DataFrame) -> Path:
    """绘制 all-delay bootstrap CI。"""

    path = FIGURES / "v251_bootstrap_ci_all_delay.png"
    keep = boot[
        boot["delay_ms"].eq("all_delays")
        & boot["bucket"].isin(["all", "normal_predictable", "observe_later_like", "strong_steer", "bad_top10_v241"])
    ].copy()
    if keep.empty:
        return path
    keep = keep.reset_index(drop=True)
    x = np.arange(len(keep))
    y = keep["tail_delta_mean"].to_numpy(dtype=float)
    lo = keep["tail_delta_ci95_low"].to_numpy(dtype=float)
    hi = keep["tail_delta_ci95_high"].to_numpy(dtype=float)
    yerr = np.vstack([y - lo, hi - y])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=5, color="#1f77b4")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("v251 event-level bootstrap CI: all-delay tail delta")
    ax.set_ylabel("tail RMSE delta 95% CI（负数=优于 v241）")
    ax.set_xticks(x)
    ax.set_xticklabels(keep["bucket"], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_casebook(
    sample_metrics: pd.DataFrame,
    y_true: np.ndarray,
    pred_v241: np.ndarray,
    pred_v250: np.ndarray,
    title: str,
    path: Path,
    rows: int = 8,
) -> Path:
    """绘制 casebook 曲线。"""

    chosen = sample_metrics.head(rows).copy()
    if chosen.empty:
        return path
    fig, axes = plt.subplots(len(chosen), 1, figsize=(14, 2.25 * len(chosen)), sharex=False)
    if len(chosen) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, chosen.iterrows()):
        idx = int(row["row_index"])
        delay = int(row["delay_ms"])
        valid, _ = horizon_masks(delay)
        x = FUTURE_GRID[valid]
        ax.plot(x, y_true[idx, valid], color="black", linewidth=2.0, label="真实")
        ax.plot(x, pred_v241[idx, valid], color="#00a88f", linestyle="--", linewidth=1.6, label="v241")
        ax.plot(x, pred_v250[idx, valid], color="#f27c1e", linestyle="-.", linewidth=1.6, label="v250_minimal_lateral7")
        ax.set_title(
            f"{row['event_uid']} | {delay}ms | v241={row['tail_rmse_v241']:.3f} -> "
            f"v250={row['tail_rmse_v250']:.3f} | delta={row['delta_tail_rmse_v250_minus_v241']:.3f}",
            fontsize=9,
        )
        ax.set_ylabel("steering_delta")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("observation 后时间 / s")
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    selection: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    subj_summary: pd.DataFrame,
    boot: pd.DataFrame,
    regression_cases: pd.DataFrame,
    decision: pd.DataFrame,
    figures: List[Path],
) -> None:
    """写中文审计报告。"""

    lines: List[str] = []
    lines.append("# v251 locked robustness audit for v250_minimal_lateral7")
    lines.append("")
    lines.append("## 本轮边界")
    lines.append("")
    lines.append("- 固定 v250 best validation model：`v250_minimal_lateral7`。")
    lines.append("- 不重新训练，不调通道，不用 test 做选择。")
    lines.append("- 只在 locked test 上做 bucket/delay、subject、recording、event-level bootstrap CI 和逐样本回退审计。")
    lines.append("- 不做 anchor selector、gate/router、response-type hard routing，不删除样本。")
    lines.append("")
    lines.append("## v250 选择来源")
    lines.append("")
    keep_sel = selection.head(1)[["model_name", "n_hist_channels", "channels", "best_epoch", "best_val_loss", "accepted_as_channel_candidate"]]
    lines.append(keep_sel.to_markdown(index=False))
    lines.append("")
    lines.append("## Locked Test Bucket/Delay 摘要")
    lines.append("")
    keep = bucket_summary[
        bucket_summary["split"].eq("test")
        & bucket_summary["bucket"].isin(["all", "normal_predictable", "observe_later_like", "strong_steer", "bad_top10_v241"])
        & bucket_summary["delay_ms"].isin([0, 600, 1000])
    ].copy()
    cols = [
        "bucket",
        "delay_ms",
        "n",
        "event_n",
        "mean_tail_rmse_v241",
        "mean_tail_rmse_v250",
        "mean_delta_tail",
        "tail_improve_rate",
        "mean_delta_sample",
        "mean_delta_range_ratio",
        "mean_delta_slope_ratio",
    ]
    lines.append(keep[cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Subject-Level 摘要")
    lines.append("")
    subj_keep = subj_summary[subj_summary["bucket"].isin(["all", "normal_predictable", "observe_later_like", "strong_steer"])].copy()
    subj_cols = ["subject", "bucket", "n", "event_n", "mean_delta_tail", "tail_improve_rate", "max_delta_tail"]
    lines.append(subj_keep[subj_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Event-Level Bootstrap CI")
    lines.append("")
    boot_keep = boot[
        boot["delay_ms"].eq("all_delays")
        & boot["bucket"].isin(["all", "normal_predictable", "observe_later_like", "strong_steer", "bad_top10_v241"])
    ].copy()
    boot_cols = [
        "bucket",
        "n",
        "event_n",
        "tail_delta_mean",
        "tail_delta_ci95_low",
        "tail_delta_ci95_high",
        "tail_prob_delta_lt0",
        "tail_ci_excludes_zero_negative",
    ]
    lines.append(boot_keep[boot_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 主要回退样本")
    lines.append("")
    reg_cols = [
        "event_uid",
        "subject",
        "delay_ms",
        "tail_rmse_v241",
        "tail_rmse_v250",
        "delta_tail_rmse_v250_minus_v241",
        "sample_rmse_v241",
        "sample_rmse_v250",
    ]
    lines.append(regression_cases.head(12)[reg_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 下一步决策")
    lines.append("")
    lines.append(decision.to_markdown(index=False))
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    lines.append("")
    lines.append("## 关键产物")
    lines.append("")
    lines.append("- `tables/v251_sample_locked_delta.csv`")
    lines.append("- `tables/v251_bucket_delay_locked_summary.csv`")
    lines.append("- `tables/v251_subject_locked_summary.csv`")
    lines.append("- `tables/v251_recording_locked_summary.csv`")
    lines.append("- `tables/v251_event_bootstrap_ci.csv`")
    lines.append("- `tables/v251_worst_regressions.csv`")
    lines.append("- `tables/v251_bad_top10_casebook_index.csv`")
    lines.append("")
    (REPORTS / "v251_locked_robustness_v250_cn.md").write_text("\n".join(lines), encoding="utf-8")


def write_input_hashes() -> None:
    """记录关键输入文件哈希。"""

    paths = [V250_SCRIPT, V250_PRED, V250_SELECTION, V238.V236_ARRAYS, V238.V236_MANIFEST]
    rows = []
    for path in paths:
        if Path(path).exists():
            p = Path(path)
            rows.append({"path": str(p), "sha256": file_sha256(p), "bytes": int(p.stat().st_size)})
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    """记录输出文件清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "bytes": int(path.stat().st_size)})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> str | None:
    """打包关键产物，并返回 zipfile.testzip 结果。"""

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
    """生成约束检查。"""

    cross = int(split_check["same_event_uid_cross_split"].sum()) if "same_event_uid_cross_split" in split_check.columns else 0
    return {
        "pass": bool(cross == 0 and zip_test is None),
        "same_event_uid_cross_split_count": cross,
        "test_used_for_model_selection": False,
        "model_selection_source": "v250 validation-only selection",
        "fixed_model": "v250_minimal_lateral7",
        "retrained_model": False,
        "changed_channels": False,
        "forbidden_routes": {
            "anchor_selector": False,
            "gate_router_selector": False,
            "response_type_hard_routing": False,
            "sample_deletion": False,
            "oracle_best_anchor_as_policy": False,
        },
        "zip_testzip": zip_test,
    }


def main() -> None:
    clean_out_dir()
    np.random.seed(SEED)
    print("[v251] locked robustness audit for v250_minimal_lateral7")
    print("[v251] no retraining, no channel tuning, no test-based selection")

    manifest, y_true, pred_v241, pred_v250, best_name, selection = load_locked_inputs()
    if best_name != "v250_minimal_lateral7":
        raise AssertionError(f"Unexpected v250 model: {best_name}")

    print("[v251] compute per-sample locked metrics")
    sample_metrics, bad_thresholds = compute_sample_metrics(manifest, y_true, pred_v241, pred_v250)
    bucket_sum = bucket_delay_summary(sample_metrics)
    subj_delay = subject_delay_summary(sample_metrics)
    subj_sum = subject_summary(sample_metrics)
    rec_sum = recording_summary(sample_metrics)
    boot = bootstrap_ci_table(sample_metrics)
    split_check = V238.split_integrity_check(manifest)

    test = sample_metrics[sample_metrics["split"].eq("test")].copy()
    regressions = test[np.isfinite(test["delta_tail_rmse_v250_minus_v241"])].sort_values(
        "delta_tail_rmse_v250_minus_v241", ascending=False
    )
    improvements = test[np.isfinite(test["delta_tail_rmse_v250_minus_v241"])].sort_values(
        "delta_tail_rmse_v250_minus_v241", ascending=True
    )
    bad_casebook = test[test["bad_top10_v241"].astype(bool)].sort_values("tail_rmse_v241", ascending=False)

    decision = robustness_decision(bucket_sum, subj_sum, boot)

    print("[v251] write tables and figures")
    write_csv(sample_metrics, TABLES / "v251_sample_locked_delta.csv")
    write_csv(bucket_sum, TABLES / "v251_bucket_delay_locked_summary.csv")
    write_csv(subj_delay, TABLES / "v251_subject_delay_locked_summary.csv")
    write_csv(subj_sum, TABLES / "v251_subject_locked_summary.csv")
    write_csv(rec_sum, TABLES / "v251_recording_locked_summary.csv")
    write_csv(boot, TABLES / "v251_event_bootstrap_ci.csv")
    write_csv(regressions.head(120), TABLES / "v251_worst_regressions.csv")
    write_csv(improvements.head(120), TABLES / "v251_top_improvements.csv")
    write_csv(bad_casebook.head(120), TABLES / "v251_bad_top10_casebook_index.csv")
    write_csv(
        pd.DataFrame([{"split": k, "bad_top10_v241_threshold": v} for k, v in bad_thresholds.items()]),
        TABLES / "v251_bad_top10_thresholds.csv",
    )
    write_csv(decision, TABLES / "v251_next_decision.csv")
    write_csv(split_check, TABLES / "v251_split_integrity_check.csv")

    figures = [
        plot_bucket_delay(bucket_sum),
        plot_subject_summary(subj_sum),
        plot_bootstrap_ci(boot),
        plot_casebook(
            bad_casebook,
            y_true,
            pred_v241,
            pred_v250,
            "v251 bad_top10_v241 casebook: v241 vs v250_minimal_lateral7",
            FIGURES / "v251_bad_top10_casebook.png",
            rows=8,
        ),
        plot_casebook(
            regressions,
            y_true,
            pred_v241,
            pred_v250,
            "v251 worst regressions: v250_minimal_lateral7 vs v241",
            FIGURES / "v251_worst_regression_casebook.png",
            rows=6,
        ),
    ]

    write_input_hashes()
    write_file_inventory()
    zip_test = make_zip()
    guardrail = build_guardrail(split_check, zip_test)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v251 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_report(selection, bucket_sum, subj_sum, boot, regressions, decision, figures)
    write_file_inventory()
    zip_test = make_zip()
    guardrail["zip_testzip"] = zip_test
    guardrail["pass"] = bool(guardrail["same_event_uid_cross_split_count"] == 0 and zip_test is None)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()

    print(f"[v251] locked_robustness_pass={decision.loc[decision.decision_item.eq('locked_robustness_pass'), 'decision'].iloc[0]}")
    print(f"[v251] report={REPORTS / 'v251_locked_robustness_v250_cn.md'}")
    print(f"[v251] zip={ZIP_PATH}")


if __name__ == "__main__":
    main()
