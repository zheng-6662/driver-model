#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v225 formal route reconstruction evidence pack.

本脚本执行 GPTPro 2026-06-22 给出的 bounded closeout 后续指令：
1. 不训练任何模型，不调阈值，不新建 tau/gate/router。
2. 正式主线只锁定两个 formal headline：
   - loose_main_pool  -> avg_joint_focus
   - strict_main_pool -> peak_floor_090
3. v222a/no-harm/oracle 相关内容只写入 diagnostic-only 表和报告附录，
   不进入 formal leaderboard、formal selected config 或 formal usage。
4. 输出可复现的正式重建指标、分桶指标、失败案例索引、论文级案例图和 ZIP 包。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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

V221_DIR = BASE_DIR / "v221_formal_model_leaderboard_20260622"
CACHE_DIR = BASE_DIR / "v222a_candidate_curve_cache_20260622"
CLOSEOUT_DIR = BASE_DIR / "v222a_closeout_candidate_gap_audit_20260622"
V222A_DIR = BASE_DIR / "v222a_light_fusion_residual_20260622"
NOHARM_DIR = BASE_DIR / "v222a_noharm_gate_diagnostic_20260622"

OUT_DIR = BASE_DIR / "v225_formal_route_reconstruction_evidence_pack_20260622"
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
    },
    "strict_main_pool": {
        "formal_model": "peak_floor_090",
        "rmse": 0.571770,
        "tail_rmse": 0.658306,
    },
}

POOL_CN = {
    "loose_main_pool": "可用主池",
    "strict_main_pool": "严格主池",
}

TAIL_START_INDEX = 10
HORIZON_LENGTH = 21
STRONG_STEER_THRESHOLD = 1.5
EXTREME_PEAK_THRESHOLD = 3.0
METRIC_REPRO_TOL = 1e-5

CASE_CONFIG = {
    "formal_examples": {"per_pool": 6, "title_cn": "正式样例"},
    "worst_tail_cases": {"per_pool": 6, "title_cn": "后段误差最差案例"},
    "strong_under_cases": {"per_pool": 4, "title_cn": "强反应低估案例"},
    "baseline_sufficient_cases": {"per_pool": 4, "title_cn": "基线已经足够案例"},
}

FORBIDDEN_FORMAL_TOKENS = [
    "W3_B4_original_soft",
    "oracle",
    "oracle_model",
    "true_label",
    "fallback",
    "v222a_noharm_gate",
    "v222a_bounded_residual",
    "oracle_safe_gate",
]

LEAKAGE_GUARD_EXPECTED_KEYS = [
    "formal_model_lock_exact",
    "no_training_executed",
    "no_new_tau_created",
    "no_test_retuning",
    "no_router_created",
    "no_v222b_or_v223",
    "no_oracle_in_formal",
    "no_true_label_in_formal",
    "sample_id_alignment_pass",
    "pool_filter_pass",
    "split_filter_pass",
]

REQUIRED_RELATIVE_FILES = [
    "tables/formal_model_lock.csv",
    "tables/formal_reconstruction_metrics_overall.csv",
    "tables/formal_reconstruction_metrics_by_pool.csv",
    "tables/formal_reconstruction_metrics_by_bucket.csv",
    "tables/formal_reconstruction_metrics_by_route_event.csv",
    "tables/per_sample_formal_reconstruction_eval.csv",
    "tables/formal_failure_case_index.csv",
    "tables/diagnostic_only_v222a_closeout_summary.csv",
    "tables/excluded_diagnostic_models_audit.csv",
    "reports/v225_formal_route_reconstruction_evidence_cn.md",
    "logs/run_manifest.json",
    "logs/leakage_guard_report.json",
    "logs/forbidden_scan_report.json",
    "logs/metric_reproduction_check.json",
    "logs/file_inventory.json",
    "v225_formal_route_reconstruction_evidence_pack.zip",
]


def clean_out_dir() -> None:
    """清理本轮输出目录；只允许删除 03_baselines 下的 v225 固定目录。"""

    resolved_out = OUT_DIR.resolve()
    resolved_base = BASE_DIR.resolve()
    if resolved_base not in resolved_out.parents:
        raise AssertionError(f"拒绝清理非 03_baselines 子目录：{resolved_out}")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for path in [TABLE_DIR, FIGURE_DIR, REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    for case_name in CASE_CONFIG:
        for pool_key in FORMAL_MODEL_LOCK:
            (FIGURE_DIR / case_name / pool_key).mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一用 utf-8-sig 写出 CSV，方便 Excel 直接打开中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: object, path: Path) -> None:
    """统一写出 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    """读取项目 CSV，并在缺失时明确报错。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少输入表：{path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def bool_series(values: pd.Series) -> pd.Series:
    """兼容 bool、0/1、True/False 字符串。"""

    if values.dtype == bool:
        return values
    text = values.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y"])


def peak_values(curves: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回每条曲线的绝对峰值、带符号峰值和峰值位置。"""

    idx = np.nanargmax(np.abs(curves), axis=1)
    signed = curves[np.arange(curves.shape[0]), idx]
    return np.abs(signed), signed, idx


def per_sample_metrics(y_observed: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """计算逐样本正式重建误差。"""

    if y_observed.shape != y_pred.shape:
        raise AssertionError(f"曲线 shape 不一致：observed={y_observed.shape}, pred={y_pred.shape}")
    if y_observed.shape[1] != HORIZON_LENGTH:
        raise AssertionError(f"预测 horizon 应为 {HORIZON_LENGTH}，实际 {y_observed.shape[1]}")

    diff = y_pred - y_observed
    tail = diff[:, TAIL_START_INDEX:]
    obs_peak_abs, obs_peak_signed, obs_peak_index = peak_values(y_observed)
    pred_peak_abs, pred_peak_signed, pred_peak_index = peak_values(y_pred)
    peak_ratio = np.divide(pred_peak_abs, obs_peak_abs, out=np.zeros_like(pred_peak_abs), where=obs_peak_abs > 1e-8)
    direction_ok = np.sign(obs_peak_signed) == np.sign(pred_peak_signed)

    return {
        "rmse": np.sqrt(np.mean(np.square(diff), axis=1)),
        "tail_rmse": np.sqrt(np.mean(np.square(tail), axis=1)),
        "observed_peak_abs": obs_peak_abs,
        "pred_peak_abs": pred_peak_abs,
        "observed_peak_index": obs_peak_index,
        "pred_peak_index": pred_peak_index,
        "peak_ratio": peak_ratio,
        "direction_ok": direction_ok,
        "under_flag": pred_peak_abs < (0.5 * obs_peak_abs),
        "point_diff": diff,
    }


def aggregate_rows(df: pd.DataFrame, group_cols: List[str], scope: str) -> pd.DataFrame:
    """按固定分组汇总指标；RMSE 用逐样本平方均值复原，保证与点位 RMSE 等价。"""

    rows: List[Dict[str, object]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = dict(zip(group_cols, keys))
        rmse = float(np.sqrt(np.mean(np.square(group["rmse"].astype(float)))))
        tail_rmse = float(np.sqrt(np.mean(np.square(group["tail_rmse"].astype(float)))))
        item.update(
            {
                "scope": scope,
                "n": int(len(group)),
                "rmse": rmse,
                "tail_rmse": tail_rmse,
                "mean_sample_rmse": float(group["rmse"].mean()),
                "median_sample_rmse": float(group["rmse"].median()),
                "p90_sample_rmse": float(group["rmse"].quantile(0.90)),
                "under_rate": float(group["under_flag"].mean()),
                "direction_acc": float(group["direction_ok"].mean()),
                "strong_steer_rate": float(group["strong_steer"].mean()),
                "extreme_peak_rate": float(group["extreme_peak"].mean()),
            }
        )
        rows.append(item)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def split_or_all_frames(df: pd.DataFrame) -> pd.DataFrame:
    """复制一个 all split，用于报告全量汇总。"""

    all_df = df.copy()
    all_df["split"] = "all"
    return pd.concat([df, all_df], ignore_index=True)


def load_closeout_flags() -> pd.DataFrame:
    """读取 closeout 逐样本标签，仅取正式评估需要的分桶字段。"""

    path = CLOSEOUT_DIR / "tables" / "per_sample_failure_taxonomy.csv"
    df = read_csv(path)
    cols = [
        "pool",
        "split",
        "sample_id",
        "strong_steer",
        "reverse",
        "zero_cross",
        "multi_correction",
        "vehicle_strong",
        "normal_curve",
        "extreme_peak",
        "high_tail_error",
    ]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise AssertionError(f"closeout taxonomy 缺少字段：{missing}")
    out = df[cols].copy().rename(columns={"pool": "pool_key"})
    for col in cols[3:]:
        out[col] = bool_series(out[col])
    return out


def load_pool_payload(pool_key: str, closeout_flags: pd.DataFrame) -> Dict[str, object]:
    """读取一个 pool 的 locked formal prediction 和样本定位信息。"""

    formal_model = FORMAL_MODEL_LOCK[pool_key]
    cache_path = CACHE_DIR / f"candidate_predictions_{pool_key}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"缺少候选缓存：{cache_path}")

    with np.load(cache_path, allow_pickle=False) as cache:
        y_observed = cache["true_steer"].astype(np.float64)
        pred_key = f"pred_{formal_model}"
        if pred_key not in cache.files:
            raise AssertionError(f"{cache_path.name} 缺少 locked formal prediction: {pred_key}")
        y_pred = cache[pred_key].astype(np.float64)
        split_values = cache["split"].astype(str)
        event_uid = cache["event_uid"].astype(str)
        array_index = cache["array_index"].astype(int)
        candidate_names = cache["candidate_names"].astype(str).tolist()

    if y_pred.shape != y_observed.shape:
        raise AssertionError(f"{pool_key}:{formal_model} prediction shape 不一致")
    if y_pred.shape[1] != HORIZON_LENGTH:
        raise AssertionError(f"{pool_key}:{formal_model} horizon length 不是 {HORIZON_LENGTH}")
    if formal_model not in candidate_names:
        raise AssertionError(f"{pool_key} candidate_names 未列出 locked model {formal_model}")

    sample_manifest = read_csv(CACHE_DIR / "sample_manifest.csv")
    pool_samples = sample_manifest[sample_manifest["pool_key"].eq(pool_key)].copy().reset_index(drop=True)
    if len(pool_samples) != len(event_uid):
        raise AssertionError(f"{pool_key} sample_manifest 行数 {len(pool_samples)} 与 cache {len(event_uid)} 不一致")
    if not np.array_equal(pool_samples["event_uid"].astype(str).to_numpy(), event_uid):
        raise AssertionError(f"{pool_key} sample_manifest event_uid 与 cache event_uid 不一致")
    if not np.array_equal(pool_samples["split"].astype(str).to_numpy(), split_values):
        raise AssertionError(f"{pool_key} sample_manifest split 与 cache split 不一致")

    metrics = per_sample_metrics(y_observed, y_pred)
    eval_df = pool_samples.copy()
    eval_df["pool_key"] = pool_key
    eval_df["pool_name"] = POOL_CN[pool_key]
    eval_df["sample_id"] = event_uid
    eval_df["array_index"] = array_index
    eval_df["formal_model"] = formal_model
    eval_df["rmse"] = metrics["rmse"]
    eval_df["tail_rmse"] = metrics["tail_rmse"]
    eval_df["observed_peak_abs"] = metrics["observed_peak_abs"]
    eval_df["pred_peak_abs"] = metrics["pred_peak_abs"]
    eval_df["observed_peak_index"] = metrics["observed_peak_index"]
    eval_df["pred_peak_index"] = metrics["pred_peak_index"]
    eval_df["peak_ratio"] = metrics["peak_ratio"]
    eval_df["direction_ok"] = metrics["direction_ok"]
    eval_df["under_flag"] = metrics["under_flag"]
    eval_df["prediction_shape"] = f"{y_pred.shape[0]}x{y_pred.shape[1]}"
    eval_df["horizon_length"] = y_pred.shape[1]

    flags = closeout_flags[closeout_flags["pool_key"].eq(pool_key)].copy()
    eval_df = eval_df.merge(flags, on=["pool_key", "split", "sample_id"], how="left", validate="one_to_one")
    for col in [
        "strong_steer",
        "reverse",
        "zero_cross",
        "multi_correction",
        "vehicle_strong",
        "normal_curve",
        "extreme_peak",
        "high_tail_error",
    ]:
        if eval_df[col].isna().any():
            if col == "strong_steer":
                eval_df[col] = eval_df["observed_peak_abs"] >= STRONG_STEER_THRESHOLD
            elif col == "extreme_peak":
                eval_df[col] = eval_df["observed_peak_abs"] >= EXTREME_PEAK_THRESHOLD
            else:
                eval_df[col] = eval_df[col].fillna(False)
        eval_df[col] = bool_series(eval_df[col])

    eval_df["route_event"] = eval_df.apply(assign_route_event, axis=1)

    return {
        "pool_key": pool_key,
        "formal_model": formal_model,
        "sample_manifest": pool_samples,
        "eval": eval_df,
        "y_observed": y_observed,
        "y_pred": y_pred,
        "split": split_values,
        "event_uid": event_uid,
    }


def assign_route_event(row: pd.Series) -> str:
    """为论文表格分配一个可读 route/event 桶；只用于评估分组。"""

    if bool(row.get("extreme_peak", False)):
        return "extreme_peak"
    if bool(row.get("strong_steer", False)):
        return "strong_event"
    if bool(row.get("reverse", False)):
        return "reverse"
    if bool(row.get("zero_cross", False)):
        return "zero_cross"
    if bool(row.get("multi_correction", False)):
        return "multi_correction"
    if bool(row.get("vehicle_strong", False)):
        return "vehicle_strong"
    if bool(row.get("normal_curve", False)):
        return "normal_curve"
    return str(row.get("scene_type", "unknown_scene"))


def build_formal_model_lock() -> pd.DataFrame:
    """生成 GPTPro 指定的 formal model lock 表。"""

    return pd.DataFrame(
        [
            {
                "pool": "loose_main_pool",
                "formal_model": "avg_joint_focus",
                "source": "v221_formal_leaderboard",
                "usage": "formal_headline",
            },
            {
                "pool": "strict_main_pool",
                "formal_model": "peak_floor_090",
                "source": "v221_formal_leaderboard",
                "usage": "formal_headline",
            },
        ]
    )


def build_bucket_metrics(eval_all: pd.DataFrame) -> pd.DataFrame:
    """生成布尔机制桶和场景桶的汇总指标。"""

    bucket_cols = [
        "strong_steer",
        "extreme_peak",
        "reverse",
        "zero_cross",
        "multi_correction",
        "vehicle_strong",
        "normal_curve",
        "high_tail_error",
        "under_flag",
    ]
    rows: List[pd.DataFrame] = []
    source = split_or_all_frames(eval_all)
    for bucket in bucket_cols:
        sub = source[source[bucket].astype(bool)].copy()
        if sub.empty:
            continue
        agg = aggregate_rows(sub, ["pool_key", "pool_name", "formal_model", "split"], f"bucket:{bucket}")
        agg.insert(0, "bucket", bucket)
        agg.insert(1, "bucket_value", True)
        rows.append(agg)

    scene_rows = []
    for scene, group in source.groupby("scene_type", dropna=False):
        agg = aggregate_rows(group, ["pool_key", "pool_name", "formal_model", "split"], f"scene:{scene}")
        agg.insert(0, "bucket", "scene_type")
        agg.insert(1, "bucket_value", scene)
        scene_rows.append(agg)
    rows.extend(scene_rows)

    return pd.concat(rows, ignore_index=True, sort=False)


def build_route_event_table(eval_all: pd.DataFrame) -> pd.DataFrame:
    """生成与 per-sample 表行对齐的 route_event 指标表。"""

    out = eval_all[
        [
            "pool_key",
            "pool_name",
            "split",
            "sample_id",
            "array_index",
            "scene_type",
            "route_event",
            "formal_model",
            "rmse",
            "tail_rmse",
            "under_flag",
            "strong_steer",
            "extreme_peak",
            "reverse",
            "zero_cross",
            "multi_correction",
            "vehicle_strong",
            "normal_curve",
        ]
    ].copy()
    group_cols = ["pool_key", "split", "route_event"]
    out["route_event_row_count"] = out.groupby(group_cols)["sample_id"].transform("count")
    out["route_event_mean_rmse"] = out.groupby(group_cols)["rmse"].transform("mean")
    out["route_event_mean_tail_rmse"] = out.groupby(group_cols)["tail_rmse"].transform("mean")
    out["route_event_under_rate"] = out.groupby(group_cols)["under_flag"].transform("mean")
    return out.sort_values(["pool_key", "split", "route_event", "sample_id"]).reset_index(drop=True)


def build_failure_case_index(eval_all: pd.DataFrame) -> pd.DataFrame:
    """生成与 per-sample 表行对齐的正式失败/案例索引。"""

    out = eval_all[
        [
            "pool_key",
            "pool_name",
            "split",
            "sample_id",
            "array_index",
            "scene_type",
            "route_event",
            "formal_model",
            "rmse",
            "tail_rmse",
            "under_flag",
            "strong_steer",
            "extreme_peak",
            "high_tail_error",
            "observed_peak_abs",
            "pred_peak_abs",
            "peak_ratio",
        ]
    ].copy()
    out["tail_p90_within_pool_split"] = out.groupby(["pool_key", "split"])["tail_rmse"].transform(
        lambda s: s.quantile(0.90)
    )
    out["rmse_median_within_pool_split"] = out.groupby(["pool_key", "split"])["rmse"].transform("median")
    out["tail_median_within_pool_split"] = out.groupby(["pool_key", "split"])["tail_rmse"].transform("median")
    out["worst_tail_case"] = out["tail_rmse"] >= out["tail_p90_within_pool_split"]
    out["strong_under_case"] = out["strong_steer"].astype(bool) & out["under_flag"].astype(bool)
    out["baseline_sufficient_case"] = (
        (~out["under_flag"].astype(bool))
        & (out["rmse"] <= out["rmse_median_within_pool_split"])
        & (out["tail_rmse"] <= out["tail_median_within_pool_split"])
    )
    out["formal_example_case"] = (~out["strong_under_case"]) & (~out["worst_tail_case"])

    def primary(row: pd.Series) -> str:
        if bool(row["strong_under_case"]):
            return "strong_under_cases"
        if bool(row["worst_tail_case"]):
            return "worst_tail_cases"
        if bool(row["baseline_sufficient_case"]):
            return "baseline_sufficient_cases"
        return "formal_examples"

    out["primary_case_group"] = out.apply(primary, axis=1)
    out["selected_for_figure"] = False
    out["figure_path"] = ""
    return out.sort_values(["pool_key", "split", "sample_id"]).reset_index(drop=True)


def select_case_rows(eval_all: pd.DataFrame, pool_key: str, case_name: str, count: int) -> pd.DataFrame:
    """按 case 类型选择 test split 案例；不足时退回全 split，保证图表数量。"""

    pool = eval_all[eval_all["pool_key"].eq(pool_key)].copy()
    test = pool[pool["split"].eq("test")].copy()
    source = test if not test.empty else pool

    if case_name == "formal_examples":
        median_rmse = float(source["rmse"].median())
        selected = source.assign(_score=(source["rmse"] - median_rmse).abs()).sort_values("_score")
    elif case_name == "worst_tail_cases":
        selected = source.sort_values(["tail_rmse", "rmse"], ascending=False)
    elif case_name == "strong_under_cases":
        selected = source[source["strong_steer"].astype(bool) & source["under_flag"].astype(bool)].copy()
        selected = selected.sort_values(["tail_rmse", "observed_peak_abs"], ascending=False)
        if len(selected) < count:
            fallback = source[source["strong_steer"].astype(bool)].sort_values(
                ["under_flag", "tail_rmse", "observed_peak_abs"], ascending=False
            )
            selected = pd.concat([selected, fallback], ignore_index=True).drop_duplicates("sample_id")
    elif case_name == "baseline_sufficient_cases":
        median_rmse = float(source["rmse"].median())
        median_tail = float(source["tail_rmse"].median())
        selected = source[
            (~source["under_flag"].astype(bool))
            & (source["rmse"] <= median_rmse)
            & (source["tail_rmse"] <= median_tail)
        ].copy()
        selected = selected.sort_values(["rmse", "tail_rmse"])
        if len(selected) < count:
            fallback = source[~source["under_flag"].astype(bool)].sort_values(["rmse", "tail_rmse"])
            selected = pd.concat([selected, fallback], ignore_index=True).drop_duplicates("sample_id")
    else:
        raise ValueError(case_name)

    if len(selected) < count:
        fallback = source.sort_values(["rmse", "tail_rmse"])
        selected = pd.concat([selected, fallback], ignore_index=True).drop_duplicates("sample_id")
    return selected.head(count).copy()


def safe_filename(text: str, max_len: int = 90) -> str:
    """生成 Windows 友好的短文件名。"""

    keep = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_", "."]:
            keep.append(ch)
        else:
            keep.append("_")
    name = "".join(keep).strip("_")
    if len(name) > max_len:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
        name = f"{name[: max_len - 11]}_{digest}"
    return name or "case"


def plot_case(pool_payloads: Dict[str, Dict[str, object]], row: pd.Series, case_name: str, out_path: Path) -> None:
    """绘制单个正式重建案例图。"""

    pool_key = str(row["pool_key"])
    idx = int(row["array_index"])
    payload = pool_payloads[pool_key]
    observed = payload["y_observed"][idx]
    pred = payload["y_pred"][idx]
    time_s = np.arange(HORIZON_LENGTH) * 0.1

    plt.figure(figsize=(9.5, 5.2))
    plt.plot(time_s, observed, label="observed steering", color="#222222", linewidth=2.2)
    plt.plot(time_s, pred, label=f"formal {row['formal_model']}", color="#1f77b4", linewidth=2.0)
    plt.axvline(time_s[TAIL_START_INDEX], color="#999999", linestyle="--", linewidth=1.0, label="tail start")
    plt.grid(True, alpha=0.25)
    plt.xlabel("horizon second")
    plt.ylabel("steering")
    title = (
        f"{case_name} | pool={pool_key} | sample_id={row['sample_id']}\n"
        f"formal_model={row['formal_model']} | RMSE={row['rmse']:.6f} | "
        f"tail RMSE={row['tail_rmse']:.6f} | under flag={bool(row['under_flag'])}"
    )
    plt.title(title, fontsize=9)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_figures(pool_payloads: Dict[str, Dict[str, object]], eval_all: pd.DataFrame) -> pd.DataFrame:
    """生成 GPTPro 要求的四类案例图。"""

    rows: List[Dict[str, object]] = []
    for case_name, cfg in CASE_CONFIG.items():
        count = int(cfg["per_pool"])
        for pool_key in FORMAL_MODEL_LOCK:
            selected = select_case_rows(eval_all, pool_key, case_name, count)
            for rank, (_, row) in enumerate(selected.iterrows(), start=1):
                stem = safe_filename(f"{rank:02d}_{row['sample_id']}")
                rel = Path("figures") / case_name / pool_key / f"{stem}.png"
                out_path = OUT_DIR / rel
                plot_case(pool_payloads, row, case_name, out_path)
                rows.append(
                    {
                        "case_group": case_name,
                        "pool_key": pool_key,
                        "rank": rank,
                        "sample_id": row["sample_id"],
                        "split": row["split"],
                        "scene_type": row["scene_type"],
                        "formal_model": row["formal_model"],
                        "rmse": float(row["rmse"]),
                        "tail_rmse": float(row["tail_rmse"]),
                        "under_flag": bool(row["under_flag"]),
                        "figure_path": str(rel).replace("\\", "/"),
                    }
                )
    return pd.DataFrame(rows)


def build_diagnostic_summary() -> pd.DataFrame:
    """生成 v222a/oracle diagnostic-only 摘要；这些行不进入任何 formal 表。"""

    rows: List[Dict[str, object]] = []

    stop = read_csv(CLOSEOUT_DIR / "tables" / "v222a_stop_evidence.csv")
    gap = read_csv(CLOSEOUT_DIR / "tables" / "oracle_vs_learned_gap.csv")
    future = read_csv(CLOSEOUT_DIR / "tables" / "future_route_decision.csv")
    selected_metrics = read_csv(V222A_DIR / "tables" / "v222a_selected_metrics.csv")
    gate_report = read_csv(NOHARM_DIR / "tables" / "test_locked_gate_report.csv")
    oracle_report = read_csv(NOHARM_DIR / "tables" / "oracle_safe_gate_report.csv")

    for _, row in stop.iterrows():
        pool = row.get("pool", row.get("pool_key", "unknown"))
        rows.append(
            {
                "pool": pool,
                "diagnostic_name": "v222a_noharm_gate",
                "usage": "diagnostic_only",
                "allowed_in_formal": False,
                "source_file": "v222a_stop_evidence.csv",
                "split": "test",
                "rmse": row.get("rmse_delta_vs_baseline", np.nan),
                "tail_rmse": row.get("tail_delta_vs_baseline", np.nan),
                "summary": (
                    f"validation_formal_pass={row.get('validation_formal_pass')}; "
                    f"locked_test_formal_pass={row.get('locked_test_formal_pass')}"
                ),
            }
        )

    for _, row in selected_metrics[selected_metrics["split"].eq("test")].iterrows():
        rows.append(
            {
                "pool": row["pool_key"],
                "diagnostic_name": "v222a_bounded_residual",
                "usage": "diagnostic_only",
                "allowed_in_formal": False,
                "source_file": "v222a_selected_metrics.csv",
                "split": "test",
                "rmse": row.get("steer_rmse", np.nan),
                "tail_rmse": row.get("steer_tail_rmse_1to2s", np.nan),
                "summary": str(row.get("output_name", "")),
            }
        )

    for _, row in gate_report[gate_report["split"].eq("test")].iterrows():
        rows.append(
            {
                "pool": row["pool_key"],
                "diagnostic_name": "v222a_noharm_gate",
                "usage": "diagnostic_only",
                "allowed_in_formal": False,
                "source_file": "test_locked_gate_report.csv",
                "split": "test",
                "rmse": row.get("steer_rmse", np.nan),
                "tail_rmse": row.get("steer_tail_rmse_1to2s", np.nan),
                "summary": str(row.get("output_name", "")),
            }
        )

    for _, row in oracle_report[oracle_report["split"].eq("test")].iterrows():
        rows.append(
            {
                "pool": row["pool_key"],
                "diagnostic_name": "oracle_safe_gate",
                "usage": "diagnostic_only",
                "allowed_in_formal": False,
                "source_file": "oracle_safe_gate_report.csv",
                "split": "test",
                "rmse": row.get("steer_rmse", np.nan),
                "tail_rmse": row.get("steer_tail_rmse_1to2s", np.nan),
                "summary": "upper-bound diagnostic-only",
            }
        )

    for _, row in gap[gap["split"].eq("test")].iterrows():
        rows.append(
            {
                "pool": row["pool"],
                "diagnostic_name": "oracle_safe_gate",
                "usage": "diagnostic_only",
                "allowed_in_formal": False,
                "source_file": "oracle_vs_learned_gap.csv",
                "split": "test",
                "rmse": row.get("oracle_rmse", np.nan),
                "tail_rmse": row.get("oracle_tail_rmse", np.nan),
                "summary": (
                    f"selector_failed_rate={row.get('selector_failed_rate')}; "
                    f"candidate_missing_rate={row.get('candidate_missing_rate')}"
                ),
            }
        )

    for _, row in future.iterrows():
        rows.append(
            {
                "pool": row["pool"],
                "diagnostic_name": "future_route_decision",
                "usage": "diagnostic_only",
                "allowed_in_formal": False,
                "source_file": "future_route_decision.csv",
                "split": row.get("basis_split", "test"),
                "rmse": np.nan,
                "tail_rmse": np.nan,
                "summary": f"v222b_allowed={row.get('v222b_allowed')}; v223_allowed={row.get('v223_allowed')}",
            }
        )

    for pool in FORMAL_MODEL_LOCK:
        rows.append(
            {
                "pool": pool,
                "diagnostic_name": "ridge_residual_peakfloor",
                "usage": "diagnostic_only",
                "allowed_in_formal": False,
                "source_file": "candidate_manifest.csv",
                "split": "test",
                "rmse": np.nan,
                "tail_rmse": np.nan,
                "summary": "可在 diagnostic appendix 中引用，但本轮不作为 formal headline。",
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["pool", "diagnostic_name", "source_file"]).reset_index(drop=True)


def build_excluded_diagnostic_audit() -> pd.DataFrame:
    """列出所有非本轮 locked headline 的候选或禁用名称。"""

    candidate_manifest = read_csv(CACHE_DIR / "candidate_manifest.csv")
    rows: List[Dict[str, object]] = []
    locked = set(FORMAL_MODEL_LOCK.items())
    for _, row in candidate_manifest.iterrows():
        pool = str(row["pool_key"])
        name = str(row["candidate_name"])
        is_locked = (pool, name) in locked
        if is_locked:
            continue
        rows.append(
            {
                "pool": pool,
                "name": name,
                "source": row.get("source_stage", "candidate_manifest"),
                "original_scope": row.get("candidate_scope", ""),
                "usage": "excluded_from_v225_formal_headline",
                "allowed_in_formal": False,
                "allowed_location": "diagnostic appendix only if discussed",
                "reason": "v225 formal headline is locked by GPTPro to avg_joint_focus / peak_floor_090.",
            }
        )

    for name in [
        "W3_B4_original_soft",
        "oracle",
        "oracle_model",
        "true_label",
        "fallback",
        "v222a_noharm_gate",
        "v222a_bounded_residual",
        "oracle_safe_gate",
    ]:
        rows.append(
            {
                "pool": "all",
                "name": name,
                "source": "GPTPro v225 guardrail",
                "original_scope": "forbidden_or_diagnostic_only",
                "usage": "excluded_from_formal",
                "allowed_in_formal": False,
                "allowed_location": "diagnostic summary/audit/report appendix only",
                "reason": "Forbidden in formal usage, formal selected config, and formal leaderboard.",
            }
        )
    return pd.DataFrame(rows).sort_values(["pool", "name"]).reset_index(drop=True)


def build_metric_reproduction_check(by_pool: pd.DataFrame) -> Dict[str, object]:
    """验证 locked test formal baseline 指标复现误差不超过 1e-5。"""

    checks: List[Dict[str, object]] = []
    for pool_key, expected in EXPECTED_TEST_METRICS.items():
        model = expected["formal_model"]
        row = by_pool[
            by_pool["pool_key"].eq(pool_key)
            & by_pool["formal_model"].eq(model)
            & by_pool["split"].eq("test")
        ]
        if len(row) != 1:
            raise AssertionError(f"metric reproduction 找不到唯一行：{pool_key}/{model}/test")
        actual = row.iloc[0]
        for metric, actual_col in [("rmse", "rmse"), ("tail_rmse", "tail_rmse")]:
            actual_value = float(actual[actual_col])
            expected_value = float(expected[metric])
            diff = abs(actual_value - expected_value)
            checks.append(
                {
                    "pool_key": pool_key,
                    "formal_model": model,
                    "split": "test",
                    "metric": metric,
                    "actual": actual_value,
                    "expected": expected_value,
                    "absolute_diff": diff,
                    "tolerance": METRIC_REPRO_TOL,
                    "pass": diff <= METRIC_REPRO_TOL,
                }
            )
    passed = all(item["pass"] for item in checks)
    return {"pass": passed, "checks": checks}


def formal_scan_files() -> List[Path]:
    """只扫描 formal usage/leaderboard/config 相关输出，diagnostic appendix 不参与 forbidden fail。"""

    return [
        TABLE_DIR / "formal_model_lock.csv",
        TABLE_DIR / "formal_reconstruction_metrics_overall.csv",
        TABLE_DIR / "formal_reconstruction_metrics_by_pool.csv",
        TABLE_DIR / "formal_reconstruction_metrics_by_bucket.csv",
        TABLE_DIR / "formal_reconstruction_metrics_by_route_event.csv",
        TABLE_DIR / "per_sample_formal_reconstruction_eval.csv",
        TABLE_DIR / "formal_failure_case_index.csv",
    ]


def build_forbidden_scan_report() -> Dict[str, object]:
    """扫描 formal 表，确保禁用名称没有进入 formal usage。"""

    hits: List[Dict[str, object]] = []
    for path in formal_scan_files():
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        for token in FORBIDDEN_FORMAL_TOKENS:
            count = text.lower().count(token.lower())
            if count:
                hits.append(
                    {
                        "file": str(path.relative_to(OUT_DIR)).replace("\\", "/"),
                        "token": token,
                        "count": count,
                    }
                )
    return {
        "pass": len(hits) == 0,
        "scanned_files": [str(p.relative_to(OUT_DIR)).replace("\\", "/") for p in formal_scan_files()],
        "forbidden_tokens": FORBIDDEN_FORMAL_TOKENS,
        "hits": hits,
        "allowed_locations": [
            "tables/diagnostic_only_v222a_closeout_summary.csv",
            "tables/excluded_diagnostic_models_audit.csv",
            "reports/v225_formal_route_reconstruction_evidence_cn.md diagnostic appendix",
        ],
    }


def build_leakage_guard_report(
    formal_lock: pd.DataFrame,
    pool_payloads: Dict[str, Dict[str, object]],
    forbidden_report: Dict[str, object],
) -> Dict[str, object]:
    """生成 GPTPro 指定的只读证据包 guard。"""

    expected_lock = build_formal_model_lock()
    lock_exact = formal_lock.equals(expected_lock)
    all_splits = set()
    alignment_pass = True
    for pool_key, payload in pool_payloads.items():
        sample_manifest = payload["sample_manifest"]
        event_uid = payload["event_uid"]
        split = payload["split"]
        all_splits.update(split.tolist())
        alignment_pass = alignment_pass and np.array_equal(sample_manifest["event_uid"].astype(str).to_numpy(), event_uid)
        alignment_pass = alignment_pass and np.array_equal(sample_manifest["split"].astype(str).to_numpy(), split)

    rows = [
        ("formal_model_lock_exact", bool(lock_exact), "formal_model_lock.csv exactly matches GPTPro lock."),
        ("no_training_executed", True, "Script contains no fit/train/checkpoint branch and loads predictions only."),
        ("no_new_tau_created", True, "No tau or threshold search is created."),
        ("no_test_retuning", True, "Test split is reporting/reproduction only."),
        ("no_router_created", True, "No router/gate model is created."),
        ("no_v222b_or_v223", True, "No v222b/v223 path is read or written."),
        ("no_oracle_in_formal", bool(forbidden_report["pass"]), "Forbidden scan covers oracle token in formal files."),
        ("no_true_label_in_formal", bool(forbidden_report["pass"]), "Forbidden scan covers true_label token in formal files."),
        ("sample_id_alignment_pass", bool(alignment_pass), "cache event_uid/split align with sample_manifest."),
        ("pool_filter_pass", set(pool_payloads.keys()) == set(FORMAL_MODEL_LOCK.keys()), "Only locked pools are loaded."),
        ("split_filter_pass", all_splits.issubset({"train", "val", "test"}), f"Observed splits: {sorted(all_splits)}"),
    ]
    checks = [{"check": name, "pass": passed, "detail": detail} for name, passed, detail in rows]
    return {"pass": all(item["pass"] for item in checks), "checks": checks}


def check_table_alignment(
    per_sample: pd.DataFrame, route_event: pd.DataFrame, failure_index: pd.DataFrame
) -> Dict[str, object]:
    """验证 GPTPro 要求的 row count 和 sample_id 对齐。"""

    key_cols = ["pool_key", "split", "sample_id", "formal_model"]
    base_keys = set(per_sample[key_cols].astype(str).agg("||".join, axis=1).tolist())
    route_keys = set(route_event[key_cols].astype(str).agg("||".join, axis=1).tolist())
    failure_keys = set(failure_index[key_cols].astype(str).agg("||".join, axis=1).tolist())
    duplicate_count = int(per_sample.duplicated(["pool_key", "split", "sample_id"]).sum())
    missing_pred = int(per_sample[["rmse", "tail_rmse"]].isna().any(axis=1).sum())
    bad_shape = int((per_sample["horizon_length"].astype(int) != HORIZON_LENGTH).sum())
    checks = {
        "per_sample_rows": int(len(per_sample)),
        "route_event_rows": int(len(route_event)),
        "failure_case_rows": int(len(failure_index)),
        "route_event_keys_match": base_keys == route_keys,
        "route_event_missing_key_count": int(len(base_keys - route_keys)),
        "route_event_extra_key_count": int(len(route_keys - base_keys)),
        "failure_case_keys_match": base_keys == failure_keys,
        "failure_case_missing_key_count": int(len(base_keys - failure_keys)),
        "failure_case_extra_key_count": int(len(failure_keys - base_keys)),
        "duplicate_sample_id_within_pool_split": duplicate_count,
        "missing_formal_prediction_rows": missing_pred,
        "bad_horizon_rows": bad_shape,
        "prediction_shape": "N x 21",
        "horizon_length": HORIZON_LENGTH,
    }
    checks["pass"] = (
        checks["route_event_keys_match"]
        and checks["failure_case_keys_match"]
        and duplicate_count == 0
        and missing_pred == 0
        and bad_shape == 0
    )
    return checks


def build_report(
    by_pool: pd.DataFrame,
    by_bucket: pd.DataFrame,
    metric_check: Dict[str, object],
    leakage_guard: Dict[str, object],
    forbidden_report: Dict[str, object],
    figure_index: pd.DataFrame,
    diagnostic_summary: pd.DataFrame,
) -> None:
    """生成中文证据报告。"""

    lines: List[str] = []
    lines.append("# v225 formal route reconstruction evidence pack")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append(
        "本包按 GPTPro 指令只固化 formal baseline 证据："
        "`loose_main_pool=avg_joint_focus`，`strict_main_pool=peak_floor_090`。"
    )
    lines.append("未训练新模型，未调 threshold/tau，未创建 router/gate，未运行 v222b/v223。")
    lines.append("")
    lines.append("## Formal model lock")
    lines.append("")
    lines.append("- `loose_main_pool`: `avg_joint_focus`")
    lines.append("- `strict_main_pool`: `peak_floor_090`")
    lines.append("")
    lines.append("## Locked test reproduction")
    lines.append("")
    test_rows = by_pool[by_pool["split"].eq("test")].sort_values("pool_key")
    for row in test_rows.itertuples(index=False):
        lines.append(
            f"- `{row.pool_key}` / `{row.formal_model}`: "
            f"RMSE={row.rmse:.6f}, tail RMSE={row.tail_rmse:.6f}, "
            f"under_rate={row.under_rate:.6f}, n={int(row.n)}"
        )
    lines.append("")
    lines.append(
        f"- metric reproduction pass: `{metric_check['pass']}` "
        f"(tolerance <= {METRIC_REPRO_TOL:g})"
    )
    lines.append(f"- leakage guard pass: `{leakage_guard['pass']}`")
    lines.append(f"- forbidden scan pass: `{forbidden_report['pass']}`")
    lines.append("")
    lines.append("## Bucket summary (test split)")
    lines.append("")
    bucket_test = by_bucket[by_bucket["split"].eq("test")].copy()
    for row in bucket_test.sort_values(["pool_key", "bucket"]).itertuples(index=False):
        lines.append(
            f"- `{row.pool_key}` `{row.bucket}={row.bucket_value}`: "
            f"n={int(row.n)}, RMSE={row.rmse:.6f}, tail={row.tail_rmse:.6f}, "
            f"under_rate={row.under_rate:.6f}"
        )
    lines.append("")
    lines.append("## Figure inventory")
    lines.append("")
    counts = figure_index.groupby(["case_group", "pool_key"]).size().reset_index(name="n")
    for row in counts.itertuples(index=False):
        lines.append(f"- `{row.case_group}/{row.pool_key}`: {int(row.n)} PNG")
    lines.append("")
    lines.append("## Diagnostic appendix boundary")
    lines.append("")
    lines.append("下列内容仅用于 diagnostic appendix，不进入 formal usage / formal selected config / formal leaderboard：")
    diag_names = sorted(set(diagnostic_summary["diagnostic_name"].astype(str)))
    for name in diag_names:
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## Required files")
    lines.append("")
    for rel in REQUIRED_RELATIVE_FILES:
        lines.append(f"- `{rel}`")
    lines.append("")
    (REPORT_DIR / "v225_formal_route_reconstruction_evidence_cn.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    """计算文件 sha256。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_file_inventory(zip_bad_file: str | None = None) -> Dict[str, object]:
    """生成输出文件清单和 required-missing 检查。"""

    files: List[Dict[str, object]] = []
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(OUT_DIR)).replace("\\", "/")
            files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    figure_counts: Dict[str, int] = {}
    for case_name in CASE_CONFIG:
        figure_counts[case_name] = len(list((FIGURE_DIR / case_name).rglob("*.png")))

    required_missing = [rel for rel in REQUIRED_RELATIVE_FILES if not (OUT_DIR / rel).exists()]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "file_count": len(files),
        "figure_counts": figure_counts,
        "required_files_missing": required_missing,
        "zip_bad_file": zip_bad_file,
        "files": files,
    }


def zip_outputs() -> Tuple[Path, str | None]:
    """打包输出目录并返回 zip 自检结果。"""

    zip_path = OUT_DIR / "v225_formal_route_reconstruction_evidence_pack.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(OUT_DIR).as_posix())
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_file = zf.testzip()
    return zip_path, bad_file


def write_run_manifest(
    pool_payloads: Dict[str, Dict[str, object]],
    metric_check: Dict[str, object],
    leakage_guard: Dict[str, object],
    forbidden_report: Dict[str, object],
    table_alignment: Dict[str, object],
    zip_bad_file: str | None,
) -> None:
    """写运行 manifest，明确本轮只读边界。"""

    manifest = {
        "run_name": "v225_formal_route_reconstruction_evidence_pack_20260622",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "formal_model_lock": FORMAL_MODEL_LOCK,
        "input_dirs": {
            "v221": str(V221_DIR),
            "cache": str(CACHE_DIR),
            "closeout": str(CLOSEOUT_DIR),
            "v222a": str(V222A_DIR),
            "noharm": str(NOHARM_DIR),
        },
        "output_dir": str(OUT_DIR),
        "no_training_executed": True,
        "no_threshold_tuning": True,
        "no_router_or_gate_created": True,
        "no_v222b_or_v223": True,
        "pool_counts": {
            pool: {
                "n": int(len(payload["eval"])),
                "test_n": int((payload["eval"]["split"].astype(str) == "test").sum()),
                "formal_model": payload["formal_model"],
            }
            for pool, payload in pool_payloads.items()
        },
        "metric_reproduction_pass": bool(metric_check["pass"]),
        "leakage_guard_pass": bool(leakage_guard["pass"]),
        "forbidden_scan_pass": bool(forbidden_report["pass"]),
        "table_alignment_pass": bool(table_alignment["pass"]),
        "zip_bad_file": zip_bad_file,
        "stop_condition": "v225 evidence pack is one-shot; stop after packaging and reporting to GPTPro.",
    }
    write_json(manifest, LOG_DIR / "run_manifest.json")


def assert_completion(
    metric_check: Dict[str, object],
    leakage_guard: Dict[str, object],
    forbidden_report: Dict[str, object],
    table_alignment: Dict[str, object],
    file_inventory: Dict[str, object],
    zip_bad_file: str | None,
) -> None:
    """脚本内最后一层硬断言。"""

    errors: List[str] = []
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
    for case_name, cfg in CASE_CONFIG.items():
        required_total = int(cfg["per_pool"]) * len(FORMAL_MODEL_LOCK)
        actual = int(file_inventory["figure_counts"].get(case_name, 0))
        if actual < required_total:
            errors.append(f"{case_name} figure count {actual} < {required_total}")
    if errors:
        raise AssertionError("; ".join(errors))


def main() -> None:
    clean_out_dir()

    closeout_flags = load_closeout_flags()
    pool_payloads = {
        pool_key: load_pool_payload(pool_key, closeout_flags) for pool_key in FORMAL_MODEL_LOCK
    }
    eval_all = pd.concat([payload["eval"] for payload in pool_payloads.values()], ignore_index=True)

    formal_lock = build_formal_model_lock()
    write_csv(formal_lock, TABLE_DIR / "formal_model_lock.csv")

    per_sample_cols = [
        "pool_key",
        "pool_name",
        "split",
        "sample_id",
        "array_index",
        "subject",
        "recording",
        "anchor_s",
        "scene_type",
        "route_event",
        "formal_model",
        "rmse",
        "tail_rmse",
        "observed_peak_abs",
        "pred_peak_abs",
        "observed_peak_index",
        "pred_peak_index",
        "peak_ratio",
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
        "prediction_shape",
        "horizon_length",
    ]
    per_sample = eval_all[per_sample_cols].sort_values(["pool_key", "split", "sample_id"]).reset_index(drop=True)

    eval_with_all = split_or_all_frames(eval_all)
    overall = aggregate_rows(eval_with_all, ["split"], "combined_pool_rows")
    by_pool = aggregate_rows(eval_with_all, ["pool_key", "pool_name", "formal_model", "split"], "pool")
    by_bucket = build_bucket_metrics(eval_all)
    by_route_event = build_route_event_table(eval_all)
    failure_index = build_failure_case_index(eval_all)

    figure_index = make_figures(pool_payloads, eval_all)
    figure_path_map = figure_index.set_index(["pool_key", "sample_id", "case_group"])["figure_path"].to_dict()
    for idx, row in failure_index.iterrows():
        key = (row["pool_key"], row["sample_id"], row["primary_case_group"])
        if key in figure_path_map:
            failure_index.at[idx, "selected_for_figure"] = True
            failure_index.at[idx, "figure_path"] = figure_path_map[key]

    write_csv(overall, TABLE_DIR / "formal_reconstruction_metrics_overall.csv")
    write_csv(by_pool, TABLE_DIR / "formal_reconstruction_metrics_by_pool.csv")
    write_csv(by_bucket, TABLE_DIR / "formal_reconstruction_metrics_by_bucket.csv")
    write_csv(by_route_event, TABLE_DIR / "formal_reconstruction_metrics_by_route_event.csv")
    write_csv(per_sample, TABLE_DIR / "per_sample_formal_reconstruction_eval.csv")
    write_csv(failure_index, TABLE_DIR / "formal_failure_case_index.csv")

    diagnostic_summary = build_diagnostic_summary()
    excluded_audit = build_excluded_diagnostic_audit()
    write_csv(diagnostic_summary, TABLE_DIR / "diagnostic_only_v222a_closeout_summary.csv")
    write_csv(excluded_audit, TABLE_DIR / "excluded_diagnostic_models_audit.csv")

    metric_check = build_metric_reproduction_check(by_pool)
    write_json(metric_check, LOG_DIR / "metric_reproduction_check.json")

    forbidden_report = build_forbidden_scan_report()
    write_json(forbidden_report, LOG_DIR / "forbidden_scan_report.json")

    leakage_guard = build_leakage_guard_report(formal_lock, pool_payloads, forbidden_report)
    write_json(leakage_guard, LOG_DIR / "leakage_guard_report.json")

    table_alignment = check_table_alignment(per_sample, by_route_event, failure_index)
    write_json(table_alignment, LOG_DIR / "table_alignment_check.json")

    build_report(by_pool, by_bucket, metric_check, leakage_guard, forbidden_report, figure_index, diagnostic_summary)
    write_json(build_file_inventory(zip_bad_file=None), LOG_DIR / "file_inventory.json")

    # 先写无 zip 信息的清单，确保 file_inventory 自身会进入 ZIP。
    write_json(build_file_inventory(zip_bad_file=None), LOG_DIR / "file_inventory.json")
    zip_path, zip_bad_file = zip_outputs()
    file_inventory = build_file_inventory(zip_bad_file=zip_bad_file)
    # 更新外部清单和 manifest，供本地验收直接读取；ZIP 内容仍包含完整必需文件。
    file_inventory = build_file_inventory(zip_bad_file=zip_bad_file)
    write_json(file_inventory, LOG_DIR / "file_inventory.json")
    write_run_manifest(pool_payloads, metric_check, leakage_guard, forbidden_report, table_alignment, zip_bad_file)

    # 重新打包一次，把最终 file_inventory/run_manifest 放入 ZIP。
    zip_path, zip_bad_file = zip_outputs()
    file_inventory = build_file_inventory(zip_bad_file=zip_bad_file)
    write_json(file_inventory, LOG_DIR / "file_inventory.json")
    write_run_manifest(pool_payloads, metric_check, leakage_guard, forbidden_report, table_alignment, zip_bad_file)
    zip_path, zip_bad_file = zip_outputs()
    if zip_bad_file is not None:
        file_inventory = build_file_inventory(zip_bad_file=zip_bad_file)
        write_json(file_inventory, LOG_DIR / "file_inventory.json")
        write_run_manifest(pool_payloads, metric_check, leakage_guard, forbidden_report, table_alignment, zip_bad_file)

    assert_completion(metric_check, leakage_guard, forbidden_report, table_alignment, file_inventory, zip_bad_file)
    print(f"[OK] v225 formal route reconstruction evidence pack: {zip_path}")


if __name__ == "__main__":
    main()
