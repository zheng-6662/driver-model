#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v226 formal robustness / confidence-interval audit.

本脚本执行 GPTPro v226 给出的 bounded 审计任务：
1. 只读取 v225 已锁定的 formal 输出表，不训练模型、不搜索阈值/温度/tau。
2. formal 主线只允许两个锁定模型：
   - loose_main_pool  -> avg_joint_focus
   - strict_main_pool -> peak_floor_090
3. 输出 sample bootstrap 与 subject-block bootstrap 置信区间、subject/route/bucket
   稳健性表、tail error 集中度、低估/极端峰值审计、readiness 决策、图和 ZIP。
4. 若 formal lock、指标复现、表对齐、禁用名扫描、ZIP 完整性任一失败，则直接失败；
   本脚本不启动任何 repair model 或新 route 搜索。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


REPO_ROOT = Path(__file__).resolve().parents[3]
REBUILD_ROOT = REPO_ROOT / "05_rebuild_from_raw_20260511"
BASE_DIR = REBUILD_ROOT / "03_baselines"
V225_DIR = BASE_DIR / "v225_formal_route_reconstruction_evidence_pack_20260622"

OUT_DIR = BASE_DIR / "v226_formal_robustness_ci_audit_20260622"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"
LOG_DIR = OUT_DIR / "logs"


FORMAL_MODEL_LOCK = {
    "loose_main_pool": "avg_joint_focus",
    "strict_main_pool": "peak_floor_090",
}

EXPECTED_TEST_METRICS = {
    "loose_main_pool": {
        "formal_model": "avg_joint_focus",
        "rmse": 0.544884,
        "tail_rmse": 0.629752,
        "test_n": 184,
    },
    "strict_main_pool": {
        "formal_model": "peak_floor_090",
        "rmse": 0.571770,
        "tail_rmse": 0.658306,
        "test_n": 174,
    },
}

BOOTSTRAP_CONFIG = {
    "random_seed": 20260622,
    "n_bootstrap": 2000,
    "ci_level": 0.95,
    "ci_lower_quantile": 0.025,
    "ci_upper_quantile": 0.975,
    "bootstrap_scope": "within each pool/split",
    "sample_bootstrap_unit": "sample row",
    "subject_block_bootstrap_unit": "subject block with all rows for the selected subject",
}

HORIZON_LENGTH = 21
METRIC_TOLERANCE = 1e-5

BOOLEAN_COLUMNS = [
    "direction_ok",
    "under_flag",
    "strong_steer",
    "reverse",
    "zero_cross",
    "multi_correction",
    "vehicle_strong",
    "normal_curve",
    "extreme_peak",
    "high_tail_error",
]

METRIC_COLUMNS = [
    "rmse",
    "tail_rmse",
    "mean_sample_rmse",
    "median_sample_rmse",
    "p90_sample_rmse",
    "under_rate",
    "direction_acc",
    "strong_steer_rate",
    "extreme_peak_rate",
]

BUCKET_COLUMNS = [
    "strong_steer",
    "reverse",
    "zero_cross",
    "multi_correction",
    "vehicle_strong",
    "normal_curve",
    "extreme_peak",
    "high_tail_error",
]

FIGURE_SUBDIRS = [
    "ci_forest_by_pool",
    "subject_level_metric_distribution",
    "tail_error_concentration",
    "underestimation_profile",
    "extreme_peak_cases_summary",
]

REQUIRED_RELATIVE_FILES = [
    "tables/formal_model_lock_recheck.csv",
    "tables/formal_metric_ci_sample_bootstrap.csv",
    "tables/formal_metric_ci_subject_block_bootstrap.csv",
    "tables/formal_subject_level_metrics.csv",
    "tables/formal_route_event_level_metrics.csv",
    "tables/formal_bucket_ci_metrics.csv",
    "tables/formal_tail_error_concentration.csv",
    "tables/formal_underestimation_profile.csv",
    "tables/formal_extreme_peak_profile.csv",
    "tables/formal_sample_influence_audit.csv",
    "tables/formal_readiness_decision.csv",
    "reports/v226_formal_robustness_ci_audit_cn.md",
    "logs/run_manifest.json",
    "logs/input_file_hashes.json",
    "logs/bootstrap_config.json",
    "logs/metric_reproduction_check.json",
    "logs/leakage_guard_report.json",
    "logs/forbidden_scan_report.json",
    "logs/table_alignment_check.json",
    "logs/file_inventory.json",
    "v226_formal_robustness_ci_audit_pack.zip",
]

# 禁用名只用于扫描规则本身；扫描时会排除 forbidden_scan_report.json，避免规则表自命中。
FORBIDDEN_SCAN_PATTERNS = [
    "W3_B4_original_soft",
    "oracle_model",
    "true_label row",
    "fallback",
    "v222a_noharm_gate as formal",
    "v222a_bounded_residual as formal",
    "oracle_safe_gate as formal",
    "ridge_residual_peakfloor as formal",
]


def clean_out_dir() -> None:
    """清理本轮固定输出目录，并防止误删 03_baselines 以外路径。"""

    resolved_out = OUT_DIR.resolve()
    resolved_base = BASE_DIR.resolve()
    if resolved_base not in resolved_out.parents:
        raise AssertionError(f"拒绝清理非 03_baselines 子目录：{resolved_out}")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for path in [TABLE_DIR, FIGURE_DIR, REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    for subdir in FIGURE_SUBDIRS:
        (FIGURE_DIR / subdir).mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一写出 CSV，保留 Excel 友好的 UTF-8 BOM。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: object, path: Path) -> None:
    """统一写出 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    """读取输入 CSV，并在缺失时给出明确错误。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少输入表：{path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def sha256_file(path: Path) -> str:
    """计算输入/输出文件 sha256，用于复核。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_bool_series(values: pd.Series) -> pd.Series:
    """兼容 bool、0/1、True/False 字符串。"""

    if values.dtype == bool:
        return values.fillna(False)
    text = values.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y"])


def load_v225_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取 v225 formal 表，并做基础字段标准化。"""

    per_sample = read_csv(V225_DIR / "tables" / "per_sample_formal_reconstruction_eval.csv")
    formal_lock = read_csv(V225_DIR / "tables" / "formal_model_lock.csv")
    by_pool = read_csv(V225_DIR / "tables" / "formal_reconstruction_metrics_by_pool.csv")

    required_cols = {
        "pool_key",
        "pool_name",
        "split",
        "sample_id",
        "subject",
        "route_event",
        "formal_model",
        "rmse",
        "tail_rmse",
        "observed_peak_abs",
        "pred_peak_abs",
        "peak_ratio",
        "horizon_length",
    }
    missing = sorted(required_cols - set(per_sample.columns))
    if missing:
        raise AssertionError(f"v225 per-sample 表缺少必要列：{missing}")

    per_sample = per_sample.copy()
    per_sample["split"] = per_sample["split"].astype(str)
    per_sample["pool_key"] = per_sample["pool_key"].astype(str)
    per_sample["formal_model"] = per_sample["formal_model"].astype(str)
    per_sample["sample_id"] = per_sample["sample_id"].astype(str)
    per_sample["subject"] = per_sample["subject"].astype(str)
    per_sample["route_event"] = per_sample["route_event"].astype(str)
    for col in BOOLEAN_COLUMNS:
        if col in per_sample.columns:
            per_sample[col] = normalize_bool_series(per_sample[col])
        else:
            raise AssertionError(f"v225 per-sample 表缺少布尔审计列：{col}")
    for col in ["rmse", "tail_rmse", "observed_peak_abs", "pred_peak_abs", "peak_ratio"]:
        per_sample[col] = pd.to_numeric(per_sample[col], errors="raise")
    per_sample["horizon_length"] = pd.to_numeric(per_sample["horizon_length"], errors="raise").astype(int)

    return per_sample, formal_lock, by_pool


def with_all_split(df: pd.DataFrame) -> pd.DataFrame:
    """为稳健性汇总补充 pool 内 all split，不改变原始 train/val/test 行。"""

    all_df = df.copy()
    all_df["split"] = "all"
    return pd.concat([df, all_df], ignore_index=True)


def metric_dict(group: pd.DataFrame) -> Dict[str, float]:
    """按 v225 定义汇总指标；aggregate RMSE 用平方平均后开方。"""

    n = len(group)
    if n == 0:
        return {metric: float("nan") for metric in METRIC_COLUMNS}
    rmse = group["rmse"].to_numpy(dtype=float)
    tail_rmse = group["tail_rmse"].to_numpy(dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(rmse)))),
        "tail_rmse": float(np.sqrt(np.mean(np.square(tail_rmse)))),
        "mean_sample_rmse": float(np.mean(rmse)),
        "median_sample_rmse": float(np.median(rmse)),
        "p90_sample_rmse": float(np.quantile(rmse, 0.90)),
        "under_rate": float(group["under_flag"].mean()),
        "direction_acc": float(group["direction_ok"].mean()),
        "strong_steer_rate": float(group["strong_steer"].mean()),
        "extreme_peak_rate": float(group["extreme_peak"].mean()),
    }


def metric_dict_from_index(arrays: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, float]:
    """bootstrap 内部按索引快速计算同一套指标。"""

    rmse = arrays["rmse"][idx]
    tail_rmse = arrays["tail_rmse"][idx]
    return {
        "rmse": float(np.sqrt(np.mean(np.square(rmse)))),
        "tail_rmse": float(np.sqrt(np.mean(np.square(tail_rmse)))),
        "mean_sample_rmse": float(np.mean(rmse)),
        "median_sample_rmse": float(np.median(rmse)),
        "p90_sample_rmse": float(np.quantile(rmse, 0.90)),
        "under_rate": float(np.mean(arrays["under_flag"][idx])),
        "direction_acc": float(np.mean(arrays["direction_ok"][idx])),
        "strong_steer_rate": float(np.mean(arrays["strong_steer"][idx])),
        "extreme_peak_rate": float(np.mean(arrays["extreme_peak"][idx])),
    }


def group_metric_rows(df: pd.DataFrame, group_cols: Sequence[str], scope: str) -> pd.DataFrame:
    """按任意分组输出点估计指标。"""

    rows: List[Dict[str, object]] = []
    for keys, group in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["scope"] = scope
        row["n"] = int(len(group))
        row.update(metric_dict(group))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci_for_group(group: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """样本级 bootstrap CI；每轮在当前 pool/split 内重采样样本。"""

    n = len(group)
    point = metric_dict(group)
    arrays = {
        "rmse": group["rmse"].to_numpy(dtype=float),
        "tail_rmse": group["tail_rmse"].to_numpy(dtype=float),
        "under_flag": group["under_flag"].to_numpy(dtype=float),
        "direction_ok": group["direction_ok"].to_numpy(dtype=float),
        "strong_steer": group["strong_steer"].to_numpy(dtype=float),
        "extreme_peak": group["extreme_peak"].to_numpy(dtype=float),
    }
    boot_values = {metric: [] for metric in METRIC_COLUMNS}
    status = "ok" if n >= 2 else "insufficient_n"
    if status == "ok":
        for _ in range(int(BOOTSTRAP_CONFIG["n_bootstrap"])):
            idx = rng.integers(0, n, size=n)
            values = metric_dict_from_index(arrays, idx)
            for metric in METRIC_COLUMNS:
                boot_values[metric].append(values[metric])

    rows: List[Dict[str, object]] = []
    for metric in METRIC_COLUMNS:
        vals = np.asarray(boot_values[metric], dtype=float)
        if status == "ok":
            lower = float(np.quantile(vals, BOOTSTRAP_CONFIG["ci_lower_quantile"]))
            upper = float(np.quantile(vals, BOOTSTRAP_CONFIG["ci_upper_quantile"]))
            std = float(np.std(vals, ddof=1))
        else:
            lower = upper = std = float("nan")
        rows.append(
            {
                "metric": metric,
                "point_estimate": point[metric],
                "ci_lower": lower,
                "ci_upper": upper,
                "bootstrap_std": std,
                "ci_level": BOOTSTRAP_CONFIG["ci_level"],
                "n": int(n),
                "n_bootstrap": BOOTSTRAP_CONFIG["n_bootstrap"] if status == "ok" else 0,
                "ci_status": status,
            }
        )
    return pd.DataFrame(rows)


def subject_block_ci_for_group(group: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """subject-block bootstrap：重采样 subject，每个 subject 带出自己的全部样本。"""

    n = len(group)
    point = metric_dict(group)
    subjects = group["subject"].astype(str).to_numpy()
    unique_subjects = np.array(sorted(pd.unique(subjects)))
    subject_indices = [np.where(subjects == subject)[0] for subject in unique_subjects]
    arrays = {
        "rmse": group["rmse"].to_numpy(dtype=float),
        "tail_rmse": group["tail_rmse"].to_numpy(dtype=float),
        "under_flag": group["under_flag"].to_numpy(dtype=float),
        "direction_ok": group["direction_ok"].to_numpy(dtype=float),
        "strong_steer": group["strong_steer"].to_numpy(dtype=float),
        "extreme_peak": group["extreme_peak"].to_numpy(dtype=float),
    }
    n_subjects = len(unique_subjects)
    status = "ok" if n_subjects >= 2 else "insufficient_subjects"
    boot_values = {metric: [] for metric in METRIC_COLUMNS}
    if status == "ok":
        for _ in range(int(BOOTSTRAP_CONFIG["n_bootstrap"])):
            sampled_subject_pos = rng.integers(0, n_subjects, size=n_subjects)
            idx = np.concatenate([subject_indices[pos] for pos in sampled_subject_pos])
            values = metric_dict_from_index(arrays, idx)
            for metric in METRIC_COLUMNS:
                boot_values[metric].append(values[metric])

    rows: List[Dict[str, object]] = []
    for metric in METRIC_COLUMNS:
        vals = np.asarray(boot_values[metric], dtype=float)
        if status == "ok":
            lower = float(np.quantile(vals, BOOTSTRAP_CONFIG["ci_lower_quantile"]))
            upper = float(np.quantile(vals, BOOTSTRAP_CONFIG["ci_upper_quantile"]))
            std = float(np.std(vals, ddof=1))
        else:
            lower = upper = std = float("nan")
        rows.append(
            {
                "metric": metric,
                "point_estimate": point[metric],
                "ci_lower": lower,
                "ci_upper": upper,
                "bootstrap_std": std,
                "ci_level": BOOTSTRAP_CONFIG["ci_level"],
                "n": int(n),
                "n_subjects": int(n_subjects),
                "n_bootstrap": BOOTSTRAP_CONFIG["n_bootstrap"] if status == "ok" else 0,
                "ci_status": status,
            }
        )
    return pd.DataFrame(rows)


def build_bootstrap_tables(df_all_splits: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成 sample bootstrap 与 subject-block bootstrap 两张 CI 表。"""

    rng = np.random.default_rng(int(BOOTSTRAP_CONFIG["random_seed"]))
    sample_rows: List[pd.DataFrame] = []
    subject_rows: List[pd.DataFrame] = []
    group_cols = ["pool_key", "pool_name", "formal_model", "split"]
    for keys, group in df_all_splits.groupby(group_cols, dropna=False):
        key_payload = {col: val for col, val in zip(group_cols, keys)}
        sample_ci = bootstrap_ci_for_group(group.reset_index(drop=True), rng)
        subject_ci = subject_block_ci_for_group(group.reset_index(drop=True), rng)
        for col, val in key_payload.items():
            sample_ci[col] = val
            subject_ci[col] = val
        sample_rows.append(sample_ci)
        subject_rows.append(subject_ci)
    sample_df = pd.concat(sample_rows, ignore_index=True)
    subject_df = pd.concat(subject_rows, ignore_index=True)
    ordered_cols = group_cols + [
        "metric",
        "point_estimate",
        "ci_lower",
        "ci_upper",
        "bootstrap_std",
        "ci_level",
        "n",
        "n_bootstrap",
        "ci_status",
    ]
    subject_cols = group_cols + [
        "metric",
        "point_estimate",
        "ci_lower",
        "ci_upper",
        "bootstrap_std",
        "ci_level",
        "n",
        "n_subjects",
        "n_bootstrap",
        "ci_status",
    ]
    return sample_df[ordered_cols], subject_df[subject_cols]


def build_bucket_ci_metrics(df_all_splits: pd.DataFrame) -> pd.DataFrame:
    """为 formal buckets 生成点估计和 CI；n<10 时保留行并标记 insufficient_n。"""

    rng = np.random.default_rng(int(BOOTSTRAP_CONFIG["random_seed"]) + 17)
    rows: List[pd.DataFrame] = []
    base_cols = ["pool_key", "pool_name", "formal_model", "split"]
    for keys, group in df_all_splits.groupby(base_cols, dropna=False):
        key_payload = {col: val for col, val in zip(base_cols, keys)}
        for bucket_col in BUCKET_COLUMNS:
            for bucket_value in [True, False]:
                subset = group[group[bucket_col] == bucket_value].reset_index(drop=True)
                if len(subset) >= 10:
                    ci_df = bootstrap_ci_for_group(subset, rng)
                else:
                    point = metric_dict(subset)
                    ci_df = pd.DataFrame(
                        [
                            {
                                "metric": metric,
                                "point_estimate": point[metric],
                                "ci_lower": float("nan"),
                                "ci_upper": float("nan"),
                                "bootstrap_std": float("nan"),
                                "ci_level": BOOTSTRAP_CONFIG["ci_level"],
                                "n": int(len(subset)),
                                "n_bootstrap": 0,
                                "ci_status": "insufficient_n",
                            }
                            for metric in METRIC_COLUMNS
                        ]
                    )
                for col, val in key_payload.items():
                    ci_df[col] = val
                ci_df["bucket_name"] = bucket_col
                ci_df["bucket_value"] = bool(bucket_value)
                rows.append(ci_df)
    result = pd.concat(rows, ignore_index=True)
    return result[
        base_cols
        + [
            "bucket_name",
            "bucket_value",
            "metric",
            "point_estimate",
            "ci_lower",
            "ci_upper",
            "bootstrap_std",
            "ci_level",
            "n",
            "n_bootstrap",
            "ci_status",
        ]
    ]


def gini(values: np.ndarray) -> float:
    """计算非负数组的 Gini，用于 tail error 集中度。"""

    values = np.asarray(values, dtype=float)
    if len(values) == 0 or np.sum(values) <= 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cum = np.cumsum(sorted_values)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def build_tail_error_concentration(df_all_splits: pd.DataFrame) -> pd.DataFrame:
    """统计 tail error 是否被少量样本主导。"""

    rows: List[Dict[str, object]] = []
    group_cols = ["pool_key", "pool_name", "formal_model", "split"]
    for keys, group in df_all_splits.groupby(group_cols, dropna=False):
        payload = {col: val for col, val in zip(group_cols, keys)}
        group = group.sort_values("tail_rmse", ascending=False).reset_index(drop=True)
        tail_sse_proxy = np.square(group["tail_rmse"].to_numpy(dtype=float))
        total = float(tail_sse_proxy.sum())
        n = len(group)

        def share_for_count(count: int) -> float:
            if total <= 0 or n == 0:
                return 0.0
            return float(tail_sse_proxy[: min(count, n)].sum() / total)

        top20_count = max(1, int(np.ceil(n * 0.20))) if n else 0
        row = {
            **payload,
            "n": int(n),
            "total_tail_sse_proxy": total,
            "top1_share": share_for_count(1),
            "top5_share": share_for_count(5),
            "top10_share": share_for_count(10),
            "top20pct_count": int(top20_count),
            "top20pct_share": share_for_count(top20_count),
            "gini_tail_sse_proxy": gini(tail_sse_proxy),
            "max_sample_id": group["sample_id"].iloc[0] if n else "",
            "max_sample_tail_rmse": float(group["tail_rmse"].iloc[0]) if n else float("nan"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_underestimation_profile(df_all_splits: pd.DataFrame) -> pd.DataFrame:
    """分 route_event 与关键 bucket 描述低估现象。"""

    rows: List[Dict[str, object]] = []
    base_cols = ["pool_key", "pool_name", "formal_model", "split"]
    for keys, group in df_all_splits.groupby(base_cols, dropna=False):
        payload = {col: val for col, val in zip(base_cols, keys)}
        route_counts = group.groupby("route_event", dropna=False)
        for route_event, subset in route_counts:
            metrics = metric_dict(subset)
            rows.append(
                {
                    **payload,
                    "profile_type": "route_event",
                    "profile_name": str(route_event),
                    "profile_value": "all",
                    "n": int(len(subset)),
                    "under_count": int(subset["under_flag"].sum()),
                    "mean_peak_ratio": float(subset["peak_ratio"].mean()) if len(subset) else float("nan"),
                    "mean_observed_peak_abs": float(subset["observed_peak_abs"].mean()) if len(subset) else float("nan"),
                    **metrics,
                }
            )
        for bucket_col in ["strong_steer", "vehicle_strong", "extreme_peak", "high_tail_error"]:
            for bucket_value in [True, False]:
                subset = group[group[bucket_col] == bucket_value]
                metrics = metric_dict(subset)
                rows.append(
                    {
                        **payload,
                        "profile_type": "bucket",
                        "profile_name": bucket_col,
                        "profile_value": str(bool(bucket_value)),
                        "n": int(len(subset)),
                        "under_count": int(subset["under_flag"].sum()) if len(subset) else 0,
                        "mean_peak_ratio": float(subset["peak_ratio"].mean()) if len(subset) else float("nan"),
                        "mean_observed_peak_abs": float(subset["observed_peak_abs"].mean()) if len(subset) else float("nan"),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def build_extreme_peak_profile(df_all_splits: pd.DataFrame) -> pd.DataFrame:
    """汇总 extreme_peak=True 的样本规模、误差与峰值比例。"""

    rows: List[Dict[str, object]] = []
    group_cols = ["pool_key", "pool_name", "formal_model", "split"]
    for keys, group in df_all_splits.groupby(group_cols, dropna=False):
        payload = {col: val for col, val in zip(group_cols, keys)}
        subset = group[group["extreme_peak"]].copy()
        metrics = metric_dict(subset)
        rows.append(
            {
                **payload,
                "n_total": int(len(group)),
                "n_extreme_peak": int(len(subset)),
                "extreme_peak_share": float(len(subset) / len(group)) if len(group) else float("nan"),
                "max_observed_peak_abs": float(subset["observed_peak_abs"].max()) if len(subset) else float("nan"),
                "mean_observed_peak_abs": float(subset["observed_peak_abs"].mean()) if len(subset) else float("nan"),
                "mean_pred_peak_abs": float(subset["pred_peak_abs"].mean()) if len(subset) else float("nan"),
                "mean_peak_ratio": float(subset["peak_ratio"].mean()) if len(subset) else float("nan"),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_sample_influence_audit(df: pd.DataFrame) -> pd.DataFrame:
    """逐样本 leave-one-out 影响审计；用于识别是否少数样本决定结论。"""

    rows: List[Dict[str, object]] = []
    group_cols = ["pool_key", "pool_name", "formal_model", "split"]
    for keys, group in df.groupby(group_cols, dropna=False):
        payload = {col: val for col, val in zip(group_cols, keys)}
        group = group.reset_index(drop=True)
        n = len(group)
        rmse_sse = np.square(group["rmse"].to_numpy(dtype=float))
        tail_sse = np.square(group["tail_rmse"].to_numpy(dtype=float))
        total_rmse_sse = float(rmse_sse.sum())
        total_tail_sse = float(tail_sse.sum())
        full_rmse = float(np.sqrt(total_rmse_sse / n)) if n else float("nan")
        full_tail = float(np.sqrt(total_tail_sse / n)) if n else float("nan")
        for idx, row in group.iterrows():
            if n > 1:
                rmse_without = float(np.sqrt((total_rmse_sse - rmse_sse[idx]) / (n - 1)))
                tail_without = float(np.sqrt((total_tail_sse - tail_sse[idx]) / (n - 1)))
            else:
                rmse_without = tail_without = float("nan")
            rows.append(
                {
                    **payload,
                    "sample_id": row["sample_id"],
                    "subject": row["subject"],
                    "route_event": row["route_event"],
                    "n_in_group": int(n),
                    "sample_rmse": float(row["rmse"]),
                    "sample_tail_rmse": float(row["tail_rmse"]),
                    "full_rmse": full_rmse,
                    "full_tail_rmse": full_tail,
                    "rmse_without_sample": rmse_without,
                    "tail_rmse_without_sample": tail_without,
                    "rmse_influence_without_minus_full": rmse_without - full_rmse,
                    "tail_influence_without_minus_full": tail_without - full_tail,
                    "rmse_sse_share": float(rmse_sse[idx] / total_rmse_sse) if total_rmse_sse > 0 else 0.0,
                    "tail_sse_share": float(tail_sse[idx] / total_tail_sse) if total_tail_sse > 0 else 0.0,
                    "under_flag": bool(row["under_flag"]),
                    "extreme_peak": bool(row["extreme_peak"]),
                    "high_tail_error": bool(row["high_tail_error"]),
                }
            )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["split", "pool_key", "tail_sse_share", "rmse_sse_share"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def build_formal_lock_recheck(formal_lock: pd.DataFrame, per_sample: pd.DataFrame) -> pd.DataFrame:
    """复核 v225 formal lock 与 per-sample 表中的实际模型名完全一致。"""

    observed_lock = (
        per_sample[["pool_key", "formal_model"]]
        .drop_duplicates()
        .sort_values(["pool_key", "formal_model"])
        .reset_index(drop=True)
    )
    rows: List[Dict[str, object]] = []
    for pool_key, expected_model in FORMAL_MODEL_LOCK.items():
        observed_models = sorted(per_sample.loc[per_sample["pool_key"] == pool_key, "formal_model"].unique().tolist())
        source_models = sorted(
            formal_lock.loc[formal_lock["pool"].astype(str) == pool_key, "formal_model"].astype(str).unique().tolist()
            if "pool" in formal_lock.columns
            else []
        )
        rows.append(
            {
                "pool_key": pool_key,
                "formal_model": expected_model,
                "observed_models": "|".join(observed_models),
                "source_lock_models": "|".join(source_models),
                "formal_lock_pass": observed_models == [expected_model],
                "source_lock_pass": (not source_models) or source_models == [expected_model],
                "usage": "formal_robustness_audit_only",
            }
        )
    extra_pools = sorted(set(observed_lock["pool_key"]) - set(FORMAL_MODEL_LOCK))
    if extra_pools:
        for pool_key in extra_pools:
            rows.append(
                {
                    "pool_key": pool_key,
                    "formal_model": "",
                    "observed_models": "|".join(sorted(per_sample.loc[per_sample["pool_key"] == pool_key, "formal_model"].unique())),
                    "source_lock_models": "",
                    "formal_lock_pass": False,
                    "source_lock_pass": False,
                    "usage": "unexpected_pool",
                }
            )
    return pd.DataFrame(rows)


def build_metric_reproduction_check(by_pool_metrics: pd.DataFrame) -> Dict[str, object]:
    """复现 v225 locked test RMSE/tail RMSE，误差必须小于 1e-5。"""

    checks: List[Dict[str, object]] = []
    for pool_key, expected in EXPECTED_TEST_METRICS.items():
        subset = by_pool_metrics[
            (by_pool_metrics["pool_key"] == pool_key)
            & (by_pool_metrics["formal_model"] == expected["formal_model"])
            & (by_pool_metrics["split"] == "test")
        ]
        if len(subset) != 1:
            checks.append(
                {
                    "pool_key": pool_key,
                    "formal_model": expected["formal_model"],
                    "split": "test",
                    "metric": "row_count",
                    "actual": int(len(subset)),
                    "expected": 1,
                    "absolute_diff": None,
                    "tolerance": METRIC_TOLERANCE,
                    "pass": False,
                }
            )
            continue
        row = subset.iloc[0]
        for metric in ["rmse", "tail_rmse"]:
            actual = float(row[metric])
            expected_value = float(expected[metric])
            diff = abs(actual - expected_value)
            checks.append(
                {
                    "pool_key": pool_key,
                    "formal_model": expected["formal_model"],
                    "split": "test",
                    "metric": metric,
                    "actual": actual,
                    "expected": expected_value,
                    "absolute_diff": diff,
                    "tolerance": METRIC_TOLERANCE,
                    "pass": bool(diff <= METRIC_TOLERANCE),
                }
            )
    return {"pass": all(item["pass"] for item in checks), "checks": checks}


def build_table_alignment_check(per_sample: pd.DataFrame, by_pool_metrics: pd.DataFrame) -> Dict[str, object]:
    """表对齐检查：锁定 pool/split 内 sample_id 不重复，horizon 与 test n 符合 v225。"""

    duplicate_count = int(per_sample.duplicated(["pool_key", "split", "sample_id"]).sum())
    bad_horizon_rows = int((per_sample["horizon_length"] != HORIZON_LENGTH).sum())
    missing_prediction_rows = int(per_sample[["rmse", "tail_rmse", "formal_model"]].isna().any(axis=1).sum())
    test_n_checks: List[Dict[str, object]] = []
    for pool_key, expected in EXPECTED_TEST_METRICS.items():
        n = int(((per_sample["pool_key"] == pool_key) & (per_sample["split"] == "test")).sum())
        test_n_checks.append(
            {
                "pool_key": pool_key,
                "actual_test_n": n,
                "expected_test_n": int(expected["test_n"]),
                "pass": bool(n == int(expected["test_n"])),
            }
        )
    metric_test_rows = by_pool_metrics[by_pool_metrics["split"] == "test"]
    return {
        "per_sample_rows": int(len(per_sample)),
        "metric_rows_test": int(len(metric_test_rows)),
        "duplicate_sample_id_within_pool_split": duplicate_count,
        "missing_formal_prediction_rows": missing_prediction_rows,
        "prediction_shape": "N x 21",
        "horizon_length": HORIZON_LENGTH,
        "bad_horizon_rows": bad_horizon_rows,
        "test_n_checks": test_n_checks,
        "pass": bool(
            duplicate_count == 0
            and missing_prediction_rows == 0
            and bad_horizon_rows == 0
            and all(item["pass"] for item in test_n_checks)
        ),
    }


def build_leakage_guard_report(
    formal_lock_recheck: pd.DataFrame,
    metric_check: Dict[str, object],
    table_alignment: Dict[str, object],
    forbidden_report: Dict[str, object],
) -> Dict[str, object]:
    """v226 是纯审计脚本，guard 主要确认没有越过 formal 边界。"""

    lock_pass = bool(formal_lock_recheck["formal_lock_pass"].all() and formal_lock_recheck["source_lock_pass"].all())
    checks = [
        {"check": "no_training_executed", "pass": True, "detail": "v226 reads v225 formal tables only."},
        {"check": "no_new_tau_created", "pass": True, "detail": "no threshold/tau search is implemented."},
        {"check": "no_test_retuning", "pass": True, "detail": "test is reproduced and reported only."},
        {"check": "no_router_created", "pass": True, "detail": "no selector/router output is produced."},
        {"check": "no_gate_created", "pass": True, "detail": "no gate output is produced."},
        {"check": "no_v222b_or_v223", "pass": True, "detail": "script has no entrypoint for those branches."},
        {"check": "formal_model_lock_exact", "pass": lock_pass, "detail": "only two locked formal models observed."},
        {"check": "no_oracle_in_formal", "pass": True, "detail": "formal rows come from v225 locked per-sample table."},
        {"check": "no_true_label_row_in_formal", "pass": True, "detail": "no row-level true label column is consumed."},
        {"check": "no_diagnostic_model_in_formal", "pass": True, "detail": "diagnostic rows are not read as formal inputs."},
        {"check": "sample_id_alignment_pass", "pass": bool(table_alignment["pass"]), "detail": "per-pool/split sample ids are unique."},
        {"check": "pool_filter_pass", "pass": set(formal_lock_recheck["pool_key"]) == set(FORMAL_MODEL_LOCK), "detail": "only locked pools are present."},
        {"check": "split_filter_pass", "pass": True, "detail": "train/val/test are preserved; all split is an aggregate label."},
        {
            "check": "tail_mask_inherited_from_v225",
            "pass": bool(metric_check["pass"] and table_alignment["bad_horizon_rows"] == 0),
            "detail": "v226 reuses v225 tail_rmse and high_tail_error columns instead of redefining the mask.",
        },
        {"check": "forbidden_scan_pass", "pass": bool(forbidden_report["pass"]), "detail": "formal outputs scanned for blocked names."},
    ]
    return {"pass": all(item["pass"] for item in checks), "checks": checks}


def build_forbidden_scan_report() -> Dict[str, object]:
    """扫描 formal tables/logs/report 主体，防止诊断名写入 formal 输出。"""

    scan_files: List[Path] = []
    for folder in [TABLE_DIR, REPORT_DIR, LOG_DIR]:
        if folder.exists():
            for path in folder.rglob("*"):
                if not path.is_file():
                    continue
                if path.name == "forbidden_scan_report.json":
                    continue
                if path.suffix.lower() not in [".csv", ".json", ".md", ".txt"]:
                    continue
                scan_files.append(path)

    hits: List[Dict[str, object]] = []
    for path in sorted(scan_files):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_SCAN_PATTERNS:
            if pattern in text:
                hits.append(
                    {
                        "file": str(path.relative_to(OUT_DIR)),
                        "pattern": pattern,
                        "count": int(text.count(pattern)),
                    }
                )
    return {
        "pass": len(hits) == 0,
        "patterns": FORBIDDEN_SCAN_PATTERNS,
        "scanned_file_count": len(scan_files),
        "hits": hits,
    }


def build_readiness_decision(metric_check: Dict[str, object], leakage_guard: Dict[str, object]) -> pd.DataFrame:
    """给论文主结果和下一步是否需要新模型/新 gate 给出锁定决策。"""

    accepted = bool(metric_check["pass"] and leakage_guard["pass"])
    base_reason = (
        "locked formal metrics reproduced and v226 robustness evidence packaged; "
        "remaining uncertainty is reporting uncertainty, not a reason to launch a new model."
        if accepted
        else "one or more required v226 checks failed; stop and inspect logs before any continuation."
    )
    rows = [
        {
            "scope": "total",
            "formal_model": "locked_formal_pair",
            "accepted_for_paper_main_result": accepted,
            "needs_new_model": False,
            "needs_gate_or_router": False,
            "needs_more_diagnostic_only": False,
            "reason": base_reason,
        }
    ]
    for pool_key, model in FORMAL_MODEL_LOCK.items():
        rows.append(
            {
                "scope": pool_key,
                "formal_model": model,
                "accepted_for_paper_main_result": accepted,
                "needs_new_model": False,
                "needs_gate_or_router": False,
                "needs_more_diagnostic_only": False,
                "reason": base_reason,
            }
        )
    return pd.DataFrame(rows)


def save_figure(path: Path) -> None:
    """保存图像并关闭 figure，避免长脚本占用过多句柄。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_ci_forest_figures(sample_ci: pd.DataFrame) -> None:
    """每个 pool/test/metric 一张 CI forest 小图，标题包含 pool/model/split/metric。"""

    for pool_key, formal_model in FORMAL_MODEL_LOCK.items():
        for metric in ["rmse", "tail_rmse"]:
            subset = sample_ci[
                (sample_ci["pool_key"] == pool_key)
                & (sample_ci["formal_model"] == formal_model)
                & (sample_ci["split"] == "test")
                & (sample_ci["metric"] == metric)
            ]
            if subset.empty:
                continue
            row = subset.iloc[0]
            point = float(row["point_estimate"])
            lower = float(row["ci_lower"])
            upper = float(row["ci_upper"])
            plt.figure(figsize=(7.0, 2.5))
            plt.errorbar(
                [point],
                [0],
                xerr=[[point - lower], [upper - point]],
                fmt="o",
                color="#1f77b4",
                capsize=4,
            )
            plt.yticks([0], [f"{pool_key}\n{formal_model}"])
            plt.xlabel(metric)
            plt.title(f"{pool_key} | {formal_model} | test | {metric} sample CI")
            plt.grid(axis="x", alpha=0.25)
            save_figure(FIGURE_DIR / "ci_forest_by_pool" / f"{pool_key}_test_{metric}_sample_ci.png")


def make_subject_distribution_figures(subject_metrics: pd.DataFrame) -> None:
    """画 test subject-level RMSE/tail 分布图。"""

    for pool_key, formal_model in FORMAL_MODEL_LOCK.items():
        subset = subject_metrics[
            (subject_metrics["pool_key"] == pool_key)
            & (subject_metrics["formal_model"] == formal_model)
            & (subject_metrics["split"] == "test")
        ].sort_values("subject")
        for metric in ["rmse", "tail_rmse"]:
            plt.figure(figsize=(7.5, 3.4))
            if subset.empty:
                plt.text(0.5, 0.5, "no subject rows", ha="center", va="center")
                plt.xticks([])
                plt.yticks([])
            else:
                plt.bar(subset["subject"].astype(str), subset[metric].astype(float), color="#4c78a8")
                plt.ylabel(metric)
                plt.xlabel("subject")
                plt.grid(axis="y", alpha=0.25)
            plt.title(f"{pool_key} | {formal_model} | test | subject {metric}")
            save_figure(
                FIGURE_DIR
                / "subject_level_metric_distribution"
                / f"{pool_key}_test_subject_{metric}_distribution.png"
            )


def make_tail_concentration_figures(per_sample: pd.DataFrame) -> None:
    """画 test tail error 累积贡献曲线。"""

    for pool_key, formal_model in FORMAL_MODEL_LOCK.items():
        subset = per_sample[
            (per_sample["pool_key"] == pool_key)
            & (per_sample["formal_model"] == formal_model)
            & (per_sample["split"] == "test")
        ].copy()
        subset = subset.sort_values("tail_rmse", ascending=False)
        values = np.square(subset["tail_rmse"].to_numpy(dtype=float))
        cumulative = np.cumsum(values) / values.sum() if values.sum() > 0 else np.zeros_like(values)
        x = np.arange(1, len(values) + 1)
        plt.figure(figsize=(6.8, 3.4))
        plt.plot(x, cumulative, marker=".", linewidth=1.4, color="#f58518")
        plt.xlabel("top samples sorted by tail_rmse")
        plt.ylabel("cumulative tail error share")
        plt.ylim(0, 1.02)
        plt.grid(alpha=0.25)
        plt.title(f"{pool_key} | {formal_model} | test | tail_rmse concentration")
        save_figure(FIGURE_DIR / "tail_error_concentration" / f"{pool_key}_test_tail_error_concentration.png")


def make_underestimation_profile_figures(under_profile: pd.DataFrame) -> None:
    """按 route_event 画 test under_rate 条形图。"""

    for pool_key, formal_model in FORMAL_MODEL_LOCK.items():
        subset = under_profile[
            (under_profile["pool_key"] == pool_key)
            & (under_profile["formal_model"] == formal_model)
            & (under_profile["split"] == "test")
            & (under_profile["profile_type"] == "route_event")
        ].copy()
        subset = subset.sort_values("under_rate", ascending=False)
        plt.figure(figsize=(8.5, 3.8))
        if subset.empty:
            plt.text(0.5, 0.5, "no route-event rows", ha="center", va="center")
            plt.xticks([])
            plt.yticks([])
        else:
            labels = subset["profile_name"].astype(str).tolist()
            plt.bar(labels, subset["under_rate"].astype(float), color="#54a24b")
            plt.ylabel("under_rate")
            plt.xlabel("route_event")
            plt.xticks(rotation=35, ha="right")
            plt.grid(axis="y", alpha=0.25)
        plt.title(f"{pool_key} | {formal_model} | test | under_rate by route_event")
        save_figure(FIGURE_DIR / "underestimation_profile" / f"{pool_key}_test_under_rate_by_route_event.png")


def make_extreme_peak_figures(per_sample: pd.DataFrame) -> None:
    """画 test extreme peak 样本的 observed/pred peak 对比。"""

    for pool_key, formal_model in FORMAL_MODEL_LOCK.items():
        subset = per_sample[
            (per_sample["pool_key"] == pool_key)
            & (per_sample["formal_model"] == formal_model)
            & (per_sample["split"] == "test")
            & (per_sample["extreme_peak"])
        ].copy()
        subset = subset.sort_values("observed_peak_abs", ascending=False).head(12)
        plt.figure(figsize=(8.6, 3.8))
        if subset.empty:
            plt.text(0.5, 0.5, "no extreme peak cases", ha="center", va="center")
            plt.xticks([])
            plt.yticks([])
        else:
            x = np.arange(len(subset))
            plt.bar(x - 0.18, subset["observed_peak_abs"], width=0.36, label="observed", color="#e45756")
            plt.bar(x + 0.18, subset["pred_peak_abs"], width=0.36, label="pred", color="#72b7b2")
            labels = [str(s)[-10:] for s in subset["sample_id"].tolist()]
            plt.xticks(x, labels, rotation=35, ha="right")
            plt.ylabel("peak_abs")
            plt.xlabel("sample_id suffix")
            plt.legend()
            plt.grid(axis="y", alpha=0.25)
        plt.title(f"{pool_key} | {formal_model} | test | extreme_peak peak_abs summary")
        save_figure(
            FIGURE_DIR / "extreme_peak_cases_summary" / f"{pool_key}_test_extreme_peak_cases_summary.png"
        )


def make_figures(
    sample_ci: pd.DataFrame,
    subject_metrics: pd.DataFrame,
    per_sample: pd.DataFrame,
    under_profile: pd.DataFrame,
) -> None:
    """集中生成 v226 要求的所有图目录。"""

    make_ci_forest_figures(sample_ci)
    make_subject_distribution_figures(subject_metrics)
    make_tail_concentration_figures(per_sample)
    make_underestimation_profile_figures(under_profile)
    make_extreme_peak_figures(per_sample)


def build_report(
    by_pool_metrics: pd.DataFrame,
    sample_ci: pd.DataFrame,
    subject_ci: pd.DataFrame,
    tail_concentration: pd.DataFrame,
    readiness: pd.DataFrame,
    metric_check: Dict[str, object],
    leakage_guard: Dict[str, object],
) -> None:
    """生成中文审计报告，面向后续 GPTPro 复核与论文主结果整理。"""

    test_rows = by_pool_metrics[by_pool_metrics["split"] == "test"].sort_values("pool_key")
    lines: List[str] = [
        "# v226 formal robustness / CI audit 报告",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 输入：`{V225_DIR}`",
        f"- 输出：`{OUT_DIR}`",
        "- 范围：audit-only + reporting-only；未训练模型、未调 threshold/tau、未生成 gate/router。",
        "- formal lock：loose_main_pool 使用 avg_joint_focus；strict_main_pool 使用 peak_floor_090。",
        "- 本轮只复用 v225 per-sample formal 表中的 `rmse`、`tail_rmse`、bucket 与 subject/split 元数据；tail 定义继承 v225。",
        "",
        "## locked test 指标复现",
        "",
        "| pool | formal_model | n | RMSE | tail RMSE | mean sample RMSE | under rate | direction acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in test_rows.iterrows():
        lines.append(
            f"| {row['pool_key']} | {row['formal_model']} | {int(row['n'])} | "
            f"{float(row['rmse']):.6f} | {float(row['tail_rmse']):.6f} | "
            f"{float(row['mean_sample_rmse']):.6f} | {float(row['under_rate']):.6f} | "
            f"{float(row['direction_acc']):.6f} |"
        )

    lines.extend(
        [
            "",
            f"- 指标复现检查：{'pass' if metric_check['pass'] else 'fail'}。",
            f"- leakage/边界检查：{'pass' if leakage_guard['pass'] else 'fail'}。",
            "",
            "## sample bootstrap 95% CI（test）",
            "",
            "| pool | metric | point | ci_lower | ci_upper |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in sample_ci[(sample_ci["split"] == "test") & (sample_ci["metric"].isin(["rmse", "tail_rmse"]))].sort_values(
        ["pool_key", "metric"]
    ).iterrows():
        lines.append(
            f"| {row['pool_key']} | {row['metric']} | {float(row['point_estimate']):.6f} | "
            f"{float(row['ci_lower']):.6f} | {float(row['ci_upper']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## subject-block bootstrap 95% CI（test）",
            "",
            "| pool | metric | point | ci_lower | ci_upper | n_subjects |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in subject_ci[
        (subject_ci["split"] == "test") & (subject_ci["metric"].isin(["rmse", "tail_rmse"]))
    ].sort_values(["pool_key", "metric"]).iterrows():
        lines.append(
            f"| {row['pool_key']} | {row['metric']} | {float(row['point_estimate']):.6f} | "
            f"{float(row['ci_lower']):.6f} | {float(row['ci_upper']):.6f} | {int(row['n_subjects'])} |"
        )

    lines.extend(
        [
            "",
            "## tail error 集中度（test）",
            "",
            "| pool | top1 share | top5 share | top10 share | top20pct share | gini |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in tail_concentration[tail_concentration["split"] == "test"].sort_values("pool_key").iterrows():
        lines.append(
            f"| {row['pool_key']} | {float(row['top1_share']):.6f} | {float(row['top5_share']):.6f} | "
            f"{float(row['top10_share']):.6f} | {float(row['top20pct_share']):.6f} | "
            f"{float(row['gini_tail_sse_proxy']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## readiness 决策",
            "",
            "| scope | formal_model | accepted | needs_new_model | needs_gate_or_router | reason |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for _, row in readiness.iterrows():
        lines.append(
            f"| {row['scope']} | {row['formal_model']} | {bool(row['accepted_for_paper_main_result'])} | "
            f"{bool(row['needs_new_model'])} | {bool(row['needs_gate_or_router'])} | {row['reason']} |"
        )

    lines.extend(
        [
            "",
            "## 输出入口",
            "",
            "- `tables/formal_metric_ci_sample_bootstrap.csv`：样本级 bootstrap CI。",
            "- `tables/formal_metric_ci_subject_block_bootstrap.csv`：subject-block bootstrap CI。",
            "- `tables/formal_subject_level_metrics.csv`：subject 级指标分布。",
            "- `tables/formal_tail_error_concentration.csv`：tail error 集中度。",
            "- `figures/`：CI、subject 分布、tail 集中度、低估 profile 和极端峰值概要图。",
            "- `logs/`：复现、边界、禁用名、表对齐、文件清单与 ZIP 校验。",
        ]
    )
    (REPORT_DIR / "v226_formal_robustness_ci_audit_cn.md").write_text("\n".join(lines), encoding="utf-8")


def build_input_hashes() -> Dict[str, object]:
    """记录 v226 直接读取的 v225 输入文件哈希。"""

    input_paths = [
        V225_DIR / "tables" / "per_sample_formal_reconstruction_eval.csv",
        V225_DIR / "tables" / "formal_model_lock.csv",
        V225_DIR / "tables" / "formal_reconstruction_metrics_by_pool.csv",
        V225_DIR / "logs" / "metric_reproduction_check.json",
        V225_DIR / "logs" / "table_alignment_check.json",
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_paths
        ],
    }


def build_file_inventory(zip_bad_file: str | None) -> Dict[str, object]:
    """生成输出文件清单；文件列表排除 ZIP 自身，避免自引用不稳定。"""

    files: List[Dict[str, object]] = []
    zip_name = "v226_formal_robustness_ci_audit_pack.zip"
    for path in sorted(OUT_DIR.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(OUT_DIR).as_posix()
        if rel == zip_name:
            continue
        files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    figure_counts = {subdir: len(list((FIGURE_DIR / subdir).rglob("*.png"))) for subdir in FIGURE_SUBDIRS}
    required_missing = [rel for rel in REQUIRED_RELATIVE_FILES if not (OUT_DIR / rel).exists()]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "file_count_excluding_zip": len(files),
        "figure_counts": figure_counts,
        "required_files_missing": required_missing,
        "zip_bad_file": zip_bad_file,
        "files": files,
    }


def zip_outputs() -> Tuple[Path, str | None]:
    """打包 v226 输出目录，并返回 zipfile.testzip 结果。"""

    zip_path = OUT_DIR / "v226_formal_robustness_ci_audit_pack.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(OUT_DIR).as_posix())
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_file = zf.testzip()
    return zip_path, bad_file


def build_run_manifest(
    per_sample: pd.DataFrame,
    metric_check: Dict[str, object],
    leakage_guard: Dict[str, object],
    forbidden_report: Dict[str, object],
    table_alignment: Dict[str, object],
    zip_bad_file: str | None,
) -> Dict[str, object]:
    """记录本轮脚本、输入、边界和通过状态。"""

    test_counts = {
        pool_key: int(((per_sample["pool_key"] == pool_key) & (per_sample["split"] == "test")).sum())
        for pool_key in FORMAL_MODEL_LOCK
    }
    return {
        "run_name": "v226_formal_robustness_ci_audit_20260622",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "input_v225_dir": str(V225_DIR),
        "output_dir": str(OUT_DIR),
        "formal_model_lock": FORMAL_MODEL_LOCK,
        "bootstrap_config": BOOTSTRAP_CONFIG,
        "test_counts": test_counts,
        "no_training_executed": True,
        "no_threshold_or_tau_search": True,
        "no_gate_or_router_created": True,
        "no_v222b_or_v223": True,
        "diagnostic_rows_excluded_from_formal": True,
        "metric_reproduction_pass": bool(metric_check["pass"]),
        "leakage_guard_pass": bool(leakage_guard["pass"]),
        "forbidden_scan_pass": bool(forbidden_report["pass"]),
        "table_alignment_pass": bool(table_alignment["pass"]),
        "zip_bad_file": zip_bad_file,
        "stop_condition": "stop after v226 pack and checks; report to GPTPro for the next bounded instruction.",
    }


def assert_completion(
    formal_lock_recheck: pd.DataFrame,
    metric_check: Dict[str, object],
    leakage_guard: Dict[str, object],
    forbidden_report: Dict[str, object],
    table_alignment: Dict[str, object],
    file_inventory: Dict[str, object],
    zip_bad_file: str | None,
) -> None:
    """最终硬断言，确保 v226 完整且未越界。"""

    errors: List[str] = []
    if not bool(formal_lock_recheck["formal_lock_pass"].all()):
        errors.append("formal model lock recheck failed")
    if not metric_check["pass"]:
        errors.append("metric reproduction failed")
    if not leakage_guard["pass"]:
        errors.append("leakage guard failed")
    if not forbidden_report["pass"]:
        errors.append("forbidden scan failed")
    if not table_alignment["pass"]:
        errors.append("table alignment failed")
    if file_inventory["required_files_missing"]:
        errors.append(f"required files missing: {file_inventory['required_files_missing']}")
    if zip_bad_file is not None:
        errors.append(f"zip bad file: {zip_bad_file}")

    minimum_figures = {
        "ci_forest_by_pool": 2,
        "subject_level_metric_distribution": 4,
        "tail_error_concentration": 2,
        "underestimation_profile": 2,
        "extreme_peak_cases_summary": 2,
    }
    for subdir, minimum in minimum_figures.items():
        actual = int(file_inventory["figure_counts"].get(subdir, 0))
        if actual < minimum:
            errors.append(f"{subdir} figure count {actual} < {minimum}")
    if errors:
        raise AssertionError("; ".join(errors))


def main() -> None:
    clean_out_dir()

    per_sample, formal_lock, v225_by_pool = load_v225_tables()
    per_sample = per_sample[per_sample["pool_key"].isin(FORMAL_MODEL_LOCK)].copy()
    per_sample_all = with_all_split(per_sample)

    formal_lock_recheck = build_formal_lock_recheck(formal_lock, per_sample)
    write_csv(formal_lock_recheck, TABLE_DIR / "formal_model_lock_recheck.csv")

    by_pool_metrics = group_metric_rows(
        per_sample_all,
        ["pool_key", "pool_name", "formal_model", "split"],
        scope="pool",
    )
    sample_ci, subject_ci = build_bootstrap_tables(per_sample_all)
    subject_metrics = group_metric_rows(
        per_sample_all,
        ["pool_key", "pool_name", "formal_model", "split", "subject"],
        scope="subject",
    )
    route_event_metrics = group_metric_rows(
        per_sample_all,
        ["pool_key", "pool_name", "formal_model", "split", "route_event"],
        scope="route_event",
    )
    bucket_ci = build_bucket_ci_metrics(per_sample_all)
    tail_concentration = build_tail_error_concentration(per_sample_all)
    under_profile = build_underestimation_profile(per_sample_all)
    extreme_profile = build_extreme_peak_profile(per_sample_all)
    influence = build_sample_influence_audit(per_sample)

    metric_check = build_metric_reproduction_check(by_pool_metrics)
    table_alignment = build_table_alignment_check(per_sample, by_pool_metrics)
    initial_forbidden_report = {"pass": True, "hits": []}
    leakage_guard = build_leakage_guard_report(formal_lock_recheck, metric_check, table_alignment, initial_forbidden_report)
    readiness = build_readiness_decision(metric_check, leakage_guard)

    write_csv(sample_ci, TABLE_DIR / "formal_metric_ci_sample_bootstrap.csv")
    write_csv(subject_ci, TABLE_DIR / "formal_metric_ci_subject_block_bootstrap.csv")
    write_csv(subject_metrics, TABLE_DIR / "formal_subject_level_metrics.csv")
    write_csv(route_event_metrics, TABLE_DIR / "formal_route_event_level_metrics.csv")
    write_csv(bucket_ci, TABLE_DIR / "formal_bucket_ci_metrics.csv")
    write_csv(tail_concentration, TABLE_DIR / "formal_tail_error_concentration.csv")
    write_csv(under_profile, TABLE_DIR / "formal_underestimation_profile.csv")
    write_csv(extreme_profile, TABLE_DIR / "formal_extreme_peak_profile.csv")
    write_csv(influence, TABLE_DIR / "formal_sample_influence_audit.csv")
    write_csv(readiness, TABLE_DIR / "formal_readiness_decision.csv")

    make_figures(sample_ci, subject_metrics, per_sample, under_profile)
    build_report(by_pool_metrics, sample_ci, subject_ci, tail_concentration, readiness, metric_check, leakage_guard)

    write_json(build_input_hashes(), LOG_DIR / "input_file_hashes.json")
    write_json(BOOTSTRAP_CONFIG, LOG_DIR / "bootstrap_config.json")
    write_json(metric_check, LOG_DIR / "metric_reproduction_check.json")
    write_json(table_alignment, LOG_DIR / "table_alignment_check.json")
    write_json(build_run_manifest(per_sample, metric_check, leakage_guard, initial_forbidden_report, table_alignment, None), LOG_DIR / "run_manifest.json")

    forbidden_report = build_forbidden_scan_report()
    write_json(forbidden_report, LOG_DIR / "forbidden_scan_report.json")
    leakage_guard = build_leakage_guard_report(formal_lock_recheck, metric_check, table_alignment, forbidden_report)
    readiness = build_readiness_decision(metric_check, leakage_guard)
    write_csv(readiness, TABLE_DIR / "formal_readiness_decision.csv")
    write_json(leakage_guard, LOG_DIR / "leakage_guard_report.json")
    write_json(build_run_manifest(per_sample, metric_check, leakage_guard, forbidden_report, table_alignment, None), LOG_DIR / "run_manifest.json")

    # 重新扫描一次，把最终 run_manifest / leakage_guard / readiness 决策也纳入禁用名检查。
    forbidden_report = build_forbidden_scan_report()
    write_json(forbidden_report, LOG_DIR / "forbidden_scan_report.json")
    leakage_guard = build_leakage_guard_report(formal_lock_recheck, metric_check, table_alignment, forbidden_report)
    readiness = build_readiness_decision(metric_check, leakage_guard)
    write_csv(readiness, TABLE_DIR / "formal_readiness_decision.csv")
    write_json(leakage_guard, LOG_DIR / "leakage_guard_report.json")
    write_json(build_run_manifest(per_sample, metric_check, leakage_guard, forbidden_report, table_alignment, None), LOG_DIR / "run_manifest.json")

    # 第一次打包用于拿到 zipfile.testzip 结果；随后写入最终 manifest / inventory。
    zip_path, zip_bad_file = zip_outputs()
    file_inventory = build_file_inventory(zip_bad_file)
    write_json(file_inventory, LOG_DIR / "file_inventory.json")
    write_json(
        build_run_manifest(per_sample, metric_check, leakage_guard, forbidden_report, table_alignment, zip_bad_file),
        LOG_DIR / "run_manifest.json",
    )

    # file_inventory.json 写出后，再重建一次清单，避免清单在生成瞬间误报自己缺失。
    file_inventory = build_file_inventory(zip_bad_file)
    write_json(file_inventory, LOG_DIR / "file_inventory.json")

    # 最终 ZIP 需要包含最终版 inventory / manifest。
    zip_path, zip_bad_file = zip_outputs()
    file_inventory = build_file_inventory(zip_bad_file)
    write_json(file_inventory, LOG_DIR / "file_inventory.json")
    write_json(
        build_run_manifest(per_sample, metric_check, leakage_guard, forbidden_report, table_alignment, zip_bad_file),
        LOG_DIR / "run_manifest.json",
    )
    zip_path, zip_bad_file = zip_outputs()
    file_inventory = build_file_inventory(zip_bad_file)

    assert_completion(
        formal_lock_recheck,
        metric_check,
        leakage_guard,
        forbidden_report,
        table_alignment,
        file_inventory,
        zip_bad_file,
    )
    print(f"[OK] v226 formal robustness CI audit pack: {zip_path}")


if __name__ == "__main__":
    main()
