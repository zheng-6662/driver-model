#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v222a closeout + candidate gap audit.

本脚本执行 GPTPro 2026-06-22 给出的 closeout-only 指令：

1. 正式停止 v222a bounded residual / no-harm gate formal 主线。
2. 不训练 v222b/v223，不新增 router，不重新选择 tau，不根据 test 反调配置。
3. 只读取已有 v221/v222a/no-harm 产物，回答失败主要来自 learned selector/gate，
   还是现有候选池本身缺少可用曲线。

核心比较：
- B = formal headline baseline：loose 为 avg_joint_focus，strict 为 peak_floor_090。
- M = validation-selected no-harm gate 输出：逐样本在 B 与 v222a selected residual 间切换。
- O = best allowed candidate oracle diagnostic：只在固定 formal candidate pool 内逐样本选最优候选，
      仅用于诊断，不是 deployable model。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
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
CACHE_DIR = BASE_DIR / "v222a_candidate_curve_cache_20260622"
V222A_DIR = BASE_DIR / "v222a_light_fusion_residual_20260622"
NOHARM_DIR = BASE_DIR / "v222a_noharm_gate_diagnostic_20260622"
V221_DIR = BASE_DIR / "v221_formal_model_leaderboard_20260622"
OUT_DIR = BASE_DIR / "v222a_closeout_candidate_gap_audit_20260622"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"
LOG_DIR = OUT_DIR / "logs"

BASELINE_BY_POOL = {
    "loose_main_pool": "avg_joint_focus",
    "strict_main_pool": "peak_floor_090",
}

POOL_CN = {
    "loose_main_pool": "可用主池",
    "strict_main_pool": "严格主池",
}

CASE_DIR_BY_LABEL = {
    "selector_failed": "top_selector_failed_cases",
    "candidate_missing": "top_candidate_missing_cases",
    "safe_under_fix": "top_safe_under_fix_cases",
    "baseline_sufficient": "top_baseline_sufficient_cases",
}

FORBIDDEN_FORMAL_SUBSTRINGS = [
    "W3_B4_original_soft",
    "oracle_model",
    "true_label",
    "fallback",
]

TAIL_START_INDEX = 10
STRONG_STEER_THRESHOLD = 1.5
EXTREME_PEAK_THRESHOLD = 3.0
SAFE_UNDER_RMSE_MARGIN = 0.02
SAFE_UNDER_TAIL_MARGIN = 0.03
ORACLE_RMSE_GAIN_MARGIN = 0.02
ORACLE_TAIL_GAIN_MARGIN = 0.03
TOP_CASES_PER_LABEL_POOL = 8


def ensure_dirs() -> None:
    """创建所有输出目录。"""

    for path in [TABLE_DIR, REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    for dirname in CASE_DIR_BY_LABEL.values():
        (FIGURE_DIR / dirname).mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """清理旧 closeout 输出，避免旧图或旧 CSV 混入本轮判断。"""

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一以 utf-8-sig 写出 CSV，方便 Excel 直接打开中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def assert_no_forbidden_formal_names(names: Iterable[str], context: str) -> None:
    """正式候选、正式 headline 或 deployable 字段不得含禁用身份。"""

    bad: List[str] = []
    for name in names:
        lowered = str(name).lower()
        for token in FORBIDDEN_FORMAL_SUBSTRINGS:
            if token.lower() in lowered:
                bad.append(str(name))
    if bad:
        raise AssertionError(f"{context} 含禁用正式身份: {sorted(set(bad))}")


def peak_values(curves: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """返回每条曲线的绝对峰值和带符号峰值。"""

    idx = np.nanargmax(np.abs(curves), axis=1)
    signed = curves[np.arange(curves.shape[0]), idx]
    return np.abs(signed), signed


def per_sample_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """计算逐样本 steering RMSE、tail RMSE、峰值和 severe-under。"""

    diff = y_pred - y_true
    tail_mask = np.arange(y_true.shape[1]) >= TAIL_START_INDEX
    true_peak_abs, true_peak_signed = peak_values(y_true)
    pred_peak_abs, pred_peak_signed = peak_values(y_pred)
    return {
        "sample_rmse": np.sqrt(np.mean(np.square(diff), axis=1)),
        "tail_rmse": np.sqrt(np.mean(np.square(diff[:, tail_mask]), axis=1)),
        "true_peak_abs": true_peak_abs,
        "true_peak_signed": true_peak_signed,
        "pred_peak_abs": pred_peak_abs,
        "pred_peak_signed": pred_peak_signed,
        "severe_under": pred_peak_abs < (0.5 * true_peak_abs),
        "direction_ok": np.sign(true_peak_signed) == np.sign(pred_peak_signed),
        "strong_steer": true_peak_abs >= STRONG_STEER_THRESHOLD,
        "extreme_peak": true_peak_abs >= EXTREME_PEAK_THRESHOLD,
    }


def aggregate_curve_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """按曲线点位计算整体指标，保持与前序脚本的 RMSE/tail 口径一致。"""

    diff = y_pred - y_true
    tail_mask = np.arange(y_true.shape[1]) >= TAIL_START_INDEX
    sample = per_sample_metrics(y_true, y_pred)
    strong = sample["strong_steer"]
    strong_under = np.nan
    if bool(strong.any()):
        strong_under = float(np.mean(sample["severe_under"][strong]))
    return {
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "tail": float(np.sqrt(np.mean(np.square(diff[:, tail_mask])))),
        "sample_rmse_mean": float(np.mean(sample["sample_rmse"])),
        "sample_rmse_p90": float(np.quantile(sample["sample_rmse"], 0.90)),
        "under": float(np.mean(sample["severe_under"])),
        "strong_under": strong_under,
    }


def matrix_candidate_metrics(y_true: np.ndarray, y_pred_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """对 N x C x T 的候选预测矩阵计算逐样本逐候选指标。"""

    diff = y_pred_matrix - y_true[:, None, :]
    tail_mask = np.arange(y_true.shape[1]) >= TAIL_START_INDEX
    true_peak_abs, _ = peak_values(y_true)
    pred_peak_abs = np.max(np.abs(y_pred_matrix), axis=2)
    return {
        "sample_rmse": np.sqrt(np.mean(np.square(diff), axis=2)),
        "tail_rmse": np.sqrt(np.mean(np.square(diff[:, :, tail_mask]), axis=2)),
        "pred_peak_abs": pred_peak_abs,
        "severe_under": pred_peak_abs < (0.5 * true_peak_abs[:, None]),
    }


def split_mask(split_values: np.ndarray, split_name: str) -> np.ndarray:
    """生成 split 掩码；all 代表全量。"""

    if split_name == "all":
        return np.ones(len(split_values), dtype=bool)
    return split_values.astype(str) == split_name


def nonzero_signs(curve: np.ndarray, eps: float = 0.05) -> np.ndarray:
    """把接近 0 的片段去掉后返回符号序列，用于判断过零和反转。"""

    signs = np.sign(curve[np.abs(curve) >= eps])
    return signs.astype(int)


def has_zero_cross(curve: np.ndarray) -> bool:
    """诊断用过零标记：曲线有效符号序列出现正负切换。"""

    signs = nonzero_signs(curve)
    if len(signs) < 2:
        return False
    return bool(np.any(signs[1:] * signs[:-1] < 0))


def has_reverse(curve: np.ndarray) -> bool:
    """诊断用明显反转标记：正负方向都出现较大幅度。"""

    return bool(np.max(curve) >= 0.50 and np.min(curve) <= -0.50)


def is_multi_correction(curve: np.ndarray) -> bool:
    """诊断用多次修正标记：一阶差分方向多次切换。"""

    delta = np.diff(curve)
    signs = np.sign(delta[np.abs(delta) >= 0.03]).astype(int)
    if len(signs) < 4:
        return False
    return bool(np.sum(signs[1:] * signs[:-1] < 0) >= 3)


def feature_column(feature_names: List[str], wanted: str) -> int | None:
    """按精确名称查找 feature_matrix 中的输入列。"""

    try:
        return feature_names.index(wanted)
    except ValueError:
        return None


def derive_vehicle_strong(feature_matrix: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    诊断用车辆/道路强输入分桶。

    该字段只用于 closeout 分析，不作为新模型推理特征。定义为历史 ay、yaw_rate、
    curvature 或 future curvature 的 absmax/mean 列命中各自 pool 内 75 分位之一。
    """

    wanted = [
        "hist_ay_absmax",
        "hist_yaw_rate_absmax",
        "hist_curvature_absmax",
        "future_curv_absmax",
        "future_curv_mean",
    ]
    masks: List[np.ndarray] = []
    thresholds: Dict[str, float] = {}
    for name in wanted:
        idx = feature_column(feature_names, name)
        if idx is None:
            continue
        values = np.abs(feature_matrix[:, idx].astype(float))
        threshold = float(np.quantile(values, 0.75))
        thresholds[name] = threshold
        masks.append(values >= threshold)
    if not masks:
        return np.zeros(feature_matrix.shape[0], dtype=bool), thresholds
    return np.logical_or.reduce(masks), thresholds


def select_allowed_oracle(
    candidate_names: List[str],
    candidate_manifest: pd.DataFrame,
    metric_matrix: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    在固定 formal candidate pool 内构造 diagnostic oracle。

    选择规则是确定性的、非训练的逐样本排序：
    1. 优先避免 severe-under；
    2. 在同等 under 状态下最小化 tail RMSE；
    3. tail 持平时最小化整条曲线 sample RMSE。
    """

    formal = candidate_manifest[candidate_manifest["candidate_scope"].eq("formal")].copy()
    formal_names = [str(x) for x in formal["candidate_name"].tolist() if str(x) in candidate_names]
    assert_no_forbidden_formal_names(formal_names, "allowed diagnostic oracle candidate pool")
    if not formal_names:
        raise AssertionError("formal candidate pool 为空，无法做 candidate gap audit")

    allowed_indices = np.array([candidate_names.index(name) for name in formal_names], dtype=int)
    under = metric_matrix["severe_under"][:, allowed_indices].astype(int)
    tail = metric_matrix["tail_rmse"][:, allowed_indices]
    rmse = metric_matrix["sample_rmse"][:, allowed_indices]

    chosen_local = np.empty(rmse.shape[0], dtype=int)
    for row_idx in range(rmse.shape[0]):
        order = np.lexsort((rmse[row_idx], tail[row_idx], under[row_idx]))
        chosen_local[row_idx] = int(order[0])
    chosen_global = allowed_indices[chosen_local]
    return chosen_global, allowed_indices, formal_names


def read_required_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取 closeout 需要的四类表，并显式检查存在性。"""

    sample_manifest_path = CACHE_DIR / "sample_manifest.csv"
    candidate_manifest_path = CACHE_DIR / "candidate_manifest.csv"
    gate_decision_path = NOHARM_DIR / "tables" / "per_sample_gate_decisions.csv"
    locked_report_path = NOHARM_DIR / "tables" / "test_locked_gate_report.csv"
    for path in [sample_manifest_path, candidate_manifest_path, gate_decision_path, locked_report_path]:
        if not path.exists():
            raise FileNotFoundError(f"缺少 closeout 输入文件: {path}")
    sample_manifest = pd.read_csv(sample_manifest_path, encoding="utf-8-sig")
    candidate_manifest = pd.read_csv(candidate_manifest_path, encoding="utf-8-sig")
    gate_decision = pd.read_csv(gate_decision_path, encoding="utf-8-sig")
    locked_report = pd.read_csv(locked_report_path, encoding="utf-8-sig")
    return sample_manifest, candidate_manifest, gate_decision, locked_report


def load_pool_payload(
    pool_key: str,
    sample_manifest: pd.DataFrame,
    candidate_manifest_all: pd.DataFrame,
    gate_decision_all: pd.DataFrame,
) -> Dict[str, object]:
    """读取一个 pool 的 cache、selected residual、gate 决策，并生成 B/M/O 曲线。"""

    cache_path = CACHE_DIR / f"candidate_predictions_{pool_key}.npz"
    selected_path = V222A_DIR / f"v222a_selected_predictions_{pool_key}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"缺少候选曲线 cache: {cache_path}")
    if not selected_path.exists():
        raise FileNotFoundError(f"缺少 v222a selected prediction: {selected_path}")

    with np.load(cache_path, allow_pickle=False) as cache:
        candidate_names = cache["candidate_names"].astype(str).tolist()
        y_true = cache["true_steer"].astype(np.float32)
        predictions = cache["predictions"].astype(np.float32)
        feature_matrix = cache["feature_matrix"].astype(np.float32)
        feature_names = cache["feature_names"].astype(str).tolist()
        split_values = cache["split"].astype(str)
        array_index = cache["array_index"].astype(np.int64)
        event_uid = cache["event_uid"].astype(str)

    baseline_name = BASELINE_BY_POOL[pool_key]
    if baseline_name not in candidate_names:
        raise AssertionError(f"{pool_key} 缺少 baseline candidate: {baseline_name}")
    baseline_idx = candidate_names.index(baseline_name)
    baseline_pred = predictions[:, baseline_idx, :]

    with np.load(selected_path, allow_pickle=False) as selected:
        selected_name = str(selected["selected_output_name"][0])
        selected_pred = selected["pred_v222a_val_selected"].astype(np.float32)
        selected_split = selected["split"].astype(str)
        selected_event_uid = selected["event_uid"].astype(str)
    if not np.array_equal(selected_split, split_values):
        raise AssertionError(f"{pool_key} selected split 与 cache split 不一致")
    if not np.array_equal(selected_event_uid, event_uid):
        raise AssertionError(f"{pool_key} selected event_uid 与 cache event_uid 不一致")

    pool_samples = sample_manifest[sample_manifest["pool_key"].eq(pool_key)].copy().reset_index(drop=True)
    pool_gate = gate_decision_all[gate_decision_all["pool_key"].eq(pool_key)].copy().reset_index(drop=True)
    if len(pool_samples) != len(split_values):
        raise AssertionError(f"{pool_key} sample_manifest 行数与 cache 不一致")
    if len(pool_gate) != len(split_values):
        raise AssertionError(f"{pool_key} no-harm gate decision 行数与 cache 不一致")
    if not np.array_equal(pool_samples["event_uid"].astype(str).to_numpy(), event_uid):
        raise AssertionError(f"{pool_key} sample_manifest event_uid 与 cache 不一致")
    if not np.array_equal(pool_gate["event_uid"].astype(str).to_numpy(), event_uid):
        raise AssertionError(f"{pool_key} gate decision event_uid 与 cache 不一致")

    gate_use_m = pool_gate["gate_use_m"].astype(bool).to_numpy()
    gate_pred = np.where(gate_use_m[:, None], selected_pred, baseline_pred).astype(np.float32)

    candidate_manifest = candidate_manifest_all[candidate_manifest_all["pool_key"].eq(pool_key)].copy()
    candidate_metric_matrix = matrix_candidate_metrics(y_true, predictions)
    oracle_idx, allowed_indices, allowed_names = select_allowed_oracle(
        candidate_names,
        candidate_manifest,
        candidate_metric_matrix,
    )
    oracle_pred = predictions[np.arange(predictions.shape[0]), oracle_idx, :]
    oracle_names = np.array(candidate_names, dtype=object)[oracle_idx].astype(str)

    vehicle_strong, vehicle_thresholds = derive_vehicle_strong(feature_matrix, feature_names)

    return {
        "pool_key": pool_key,
        "pool_name": POOL_CN[pool_key],
        "sample_manifest": pool_samples,
        "candidate_names": candidate_names,
        "candidate_manifest": candidate_manifest,
        "allowed_candidate_indices": allowed_indices,
        "allowed_candidate_names": allowed_names,
        "baseline_name": baseline_name,
        "selected_residual_name": selected_name,
        "gate_output_name": "v222a_validation_selected_noharm_gate",
        "split": split_values,
        "array_index": array_index,
        "event_uid": event_uid,
        "true_steer": y_true,
        "predictions": predictions,
        "baseline_pred": baseline_pred,
        "selected_pred": selected_pred,
        "gate_pred": gate_pred,
        "gate_use_m": gate_use_m,
        "oracle_pred": oracle_pred.astype(np.float32),
        "oracle_candidate_index": oracle_idx,
        "oracle_candidate_name": oracle_names,
        "feature_matrix": feature_matrix,
        "feature_names": feature_names,
        "vehicle_strong": vehicle_strong,
        "vehicle_thresholds": vehicle_thresholds,
    }


def build_candidate_gap_audit(pool: Dict[str, object]) -> pd.DataFrame:
    """构造逐样本 B/M/O 诊断表，并按 GPTPro 规则分配 primary taxonomy label。"""

    y_true = pool["true_steer"]
    baseline_pred = pool["baseline_pred"]
    gate_pred = pool["gate_pred"]
    oracle_pred = pool["oracle_pred"]
    sample_manifest = pool["sample_manifest"].copy()
    split_values = pool["split"]
    true_metrics = per_sample_metrics(y_true, y_true)
    baseline = per_sample_metrics(y_true, baseline_pred)
    gate = per_sample_metrics(y_true, gate_pred)
    oracle = per_sample_metrics(y_true, oracle_pred)

    true_curves = y_true.astype(float)
    zero_cross = np.array([has_zero_cross(curve) for curve in true_curves], dtype=bool)
    reverse = np.array([has_reverse(curve) for curve in true_curves], dtype=bool)
    multi_correction = np.array([is_multi_correction(curve) for curve in true_curves], dtype=bool)

    baseline_tail_median = float(np.median(baseline["tail_rmse"]))
    baseline_tail_p75 = float(np.quantile(baseline["tail_rmse"], 0.75))
    high_tail_error = baseline["tail_rmse"] > baseline_tail_p75

    out = pd.DataFrame(
        {
            "pool": str(pool["pool_key"]),
            "pool_name": str(pool["pool_name"]),
            "split": split_values.astype(str),
            "sample_id": sample_manifest["event_uid"].astype(str).to_numpy(),
            "array_index": sample_manifest["array_index"].astype(int).to_numpy(),
            "subject": sample_manifest["subject"].astype(str).to_numpy(),
            "scenario_type": sample_manifest["scene_type"].astype(str).to_numpy(),
            "strong_steer": baseline["true_peak_abs"] >= STRONG_STEER_THRESHOLD,
            "reverse": reverse,
            "zero_cross": zero_cross,
            "multi_correction": multi_correction,
            "vehicle_strong": pool["vehicle_strong"],
            "normal_curve": baseline["true_peak_abs"] < STRONG_STEER_THRESHOLD,
            "extreme_peak": baseline["true_peak_abs"] >= EXTREME_PEAK_THRESHOLD,
            "high_tail_error": high_tail_error,
            "true_peak_abs": baseline["true_peak_abs"],
            "baseline_tail_median": baseline_tail_median,
            "baseline_tail_p75": baseline_tail_p75,
            "baseline_name": str(pool["baseline_name"]),
            "baseline_rmse": baseline["sample_rmse"],
            "baseline_tail_rmse": baseline["tail_rmse"],
            "baseline_under": baseline["severe_under"].astype(int),
            "baseline_strong_under": (baseline["severe_under"] & true_metrics["strong_steer"]).astype(int),
            "v222a_name": str(pool["gate_output_name"]),
            "selected_residual_name": str(pool["selected_residual_name"]),
            "gate_use_m": pool["gate_use_m"].astype(bool),
            "v222a_rmse": gate["sample_rmse"],
            "v222a_tail_rmse": gate["tail_rmse"],
            "v222a_under": gate["severe_under"].astype(int),
            "v222a_strong_under": (gate["severe_under"] & true_metrics["strong_steer"]).astype(int),
            "oracle_best_allowed_candidate": pool["oracle_candidate_name"],
            "oracle_rmse": oracle["sample_rmse"],
            "oracle_tail_rmse": oracle["tail_rmse"],
            "oracle_under": oracle["severe_under"].astype(int),
            "oracle_strong_under": (oracle["severe_under"] & true_metrics["strong_steer"]).astype(int),
        }
    )
    out["gain_v222a_rmse"] = out["baseline_rmse"] - out["v222a_rmse"]
    out["gain_v222a_tail"] = out["baseline_tail_rmse"] - out["v222a_tail_rmse"]
    out["gain_oracle_rmse"] = out["baseline_rmse"] - out["oracle_rmse"]
    out["gain_oracle_tail"] = out["baseline_tail_rmse"] - out["oracle_tail_rmse"]

    out["rule_baseline_sufficient"] = (
        (out["baseline_tail_rmse"] <= out["baseline_tail_median"]) & out["baseline_under"].eq(0)
    )
    out["rule_safe_under_fix"] = (
        out["baseline_under"].eq(1)
        & out["v222a_under"].eq(0)
        & (out["v222a_tail_rmse"] <= out["baseline_tail_rmse"] + SAFE_UNDER_TAIL_MARGIN)
        & (out["v222a_rmse"] <= out["baseline_rmse"] + SAFE_UNDER_RMSE_MARGIN)
    )
    out["rule_under_tradeoff"] = (
        out["baseline_under"].eq(1)
        & out["v222a_under"].eq(0)
        & (
            (out["v222a_tail_rmse"] > out["baseline_tail_rmse"] + SAFE_UNDER_TAIL_MARGIN)
            | (out["v222a_rmse"] > out["baseline_rmse"] + SAFE_UNDER_RMSE_MARGIN)
        )
    )
    out["rule_pure_harm"] = (
        (out["v222a_under"] >= out["baseline_under"])
        & (
            (out["v222a_tail_rmse"] > out["baseline_tail_rmse"] + SAFE_UNDER_TAIL_MARGIN)
            | (out["v222a_rmse"] > out["baseline_rmse"] + SAFE_UNDER_RMSE_MARGIN)
        )
    )
    out["rule_oracle_has_gain"] = (
        (out["oracle_tail_rmse"] <= out["baseline_tail_rmse"] - ORACLE_TAIL_GAIN_MARGIN)
        | (out["oracle_rmse"] <= out["baseline_rmse"] - ORACLE_RMSE_GAIN_MARGIN)
        | (out["oracle_under"] < out["baseline_under"])
    )
    out["rule_m_missed_oracle"] = (
        (
            (out["oracle_tail_rmse"] <= out["baseline_tail_rmse"] - ORACLE_TAIL_GAIN_MARGIN)
            & (out["v222a_tail_rmse"] > out["oracle_tail_rmse"] + ORACLE_TAIL_GAIN_MARGIN)
        )
        | (
            (out["oracle_rmse"] <= out["baseline_rmse"] - ORACLE_RMSE_GAIN_MARGIN)
            & (out["v222a_rmse"] > out["oracle_rmse"] + ORACLE_RMSE_GAIN_MARGIN)
        )
        | ((out["oracle_under"] < out["baseline_under"]) & (out["v222a_under"] > out["oracle_under"]))
    )
    out["rule_selector_failed"] = out["rule_oracle_has_gain"] & out["rule_m_missed_oracle"]
    out["rule_candidate_missing"] = (
        out["high_tail_error"]
        & (out["oracle_tail_rmse"] > out["baseline_tail_rmse"] - ORACLE_TAIL_GAIN_MARGIN)
        & (out["oracle_under"] >= out["baseline_under"])
    )

    def choose_label(row: pd.Series) -> str:
        """把非互斥诊断规则压成一个 primary label，顺序写入 manifest。"""

        if bool(row["rule_baseline_sufficient"]):
            return "baseline_sufficient"
        if bool(row["rule_candidate_missing"]):
            return "candidate_missing"
        if bool(row["rule_safe_under_fix"]):
            return "safe_under_fix"
        if bool(row["rule_under_tradeoff"]):
            return "under_tradeoff"
        if bool(row["rule_selector_failed"]):
            return "selector_failed"
        if bool(row["rule_pure_harm"]):
            return "pure_harm"
        if bool(row["rule_oracle_has_gain"]):
            return "selector_failed"
        if bool(row["rule_pure_harm"]):
            return "pure_harm"
        return "baseline_sufficient"

    out["taxonomy_label"] = out.apply(choose_label, axis=1)
    return out


def summarize_oracle_vs_learned(pool: Dict[str, object], audit: pd.DataFrame) -> pd.DataFrame:
    """生成 B/M/O 在 train/val/test/all 上的整体差距表。"""

    rows: List[Dict[str, object]] = []
    split_values = pool["split"]
    for split_name in ["train", "val", "test", "all"]:
        mask = split_mask(split_values, split_name)
        if not bool(mask.any()):
            continue
        b = aggregate_curve_metrics(pool["true_steer"][mask], pool["baseline_pred"][mask])
        m = aggregate_curve_metrics(pool["true_steer"][mask], pool["gate_pred"][mask])
        o = aggregate_curve_metrics(pool["true_steer"][mask], pool["oracle_pred"][mask])
        sub = audit.loc[mask].copy()
        rows.append(
            {
                "pool": str(pool["pool_key"]),
                "pool_name": str(pool["pool_name"]),
                "split": split_name,
                "n": int(mask.sum()),
                "baseline_name": str(pool["baseline_name"]),
                "v222a_name": str(pool["gate_output_name"]),
                "oracle_role": "diagnostic_only_best_allowed_formal_candidate",
                "baseline_rmse": b["rmse"],
                "baseline_tail": b["tail"],
                "baseline_under": b["under"],
                "baseline_strong_under": b["strong_under"],
                "v222a_rmse": m["rmse"],
                "v222a_tail": m["tail"],
                "v222a_under": m["under"],
                "v222a_strong_under": m["strong_under"],
                "oracle_rmse": o["rmse"],
                "oracle_tail": o["tail"],
                "oracle_under": o["under"],
                "oracle_strong_under": o["strong_under"],
                "learned_gain_rmse": b["rmse"] - m["rmse"],
                "learned_gain_tail": b["tail"] - m["tail"],
                "oracle_gain_rmse": b["rmse"] - o["rmse"],
                "oracle_gain_tail": b["tail"] - o["tail"],
                "oracle_minus_learned_rmse_gap": m["rmse"] - o["rmse"],
                "oracle_minus_learned_tail_gap": m["tail"] - o["tail"],
                "selector_failed_rate": float(sub["taxonomy_label"].eq("selector_failed").mean()),
                "candidate_missing_rate": float(sub["taxonomy_label"].eq("candidate_missing").mean()),
                "under_tradeoff_rate": float(sub["taxonomy_label"].eq("under_tradeoff").mean()),
                "safe_under_fix_rate": float(sub["taxonomy_label"].eq("safe_under_fix").mean()),
                "pure_harm_rate": float(sub["taxonomy_label"].eq("pure_harm").mean()),
                "baseline_sufficient_rate": float(sub["taxonomy_label"].eq("baseline_sufficient").mean()),
            }
        )
    return pd.DataFrame(rows)


def build_bucket_failure_summary(audit_all: pd.DataFrame) -> pd.DataFrame:
    """按 bucket 汇总主导失败类型和各类失败率。"""

    group_cols = [
        "pool",
        "split",
        "scenario_type",
        "strong_steer",
        "reverse",
        "zero_cross",
        "multi_correction",
        "vehicle_strong",
        "normal_curve",
        "extreme_peak",
        "high_tail_error",
    ]
    rows: List[Dict[str, object]] = []
    for keys, group in audit_all.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, keys))
        label_counts = group["taxonomy_label"].value_counts()
        dominant = str(label_counts.index[0])
        row: Dict[str, object] = {
            **key_dict,
            "taxonomy_label": dominant,
            "n": int(len(group)),
            "baseline_rmse": float(group["baseline_rmse"].mean()),
            "baseline_tail": float(group["baseline_tail_rmse"].mean()),
            "v222a_rmse": float(group["v222a_rmse"].mean()),
            "v222a_tail": float(group["v222a_tail_rmse"].mean()),
            "oracle_rmse": float(group["oracle_rmse"].mean()),
            "oracle_tail": float(group["oracle_tail_rmse"].mean()),
            "baseline_under": float(group["baseline_under"].mean()),
            "v222a_under": float(group["v222a_under"].mean()),
            "oracle_under": float(group["oracle_under"].mean()),
            "selector_failed_rate": float(group["taxonomy_label"].eq("selector_failed").mean()),
            "candidate_missing_rate": float(group["taxonomy_label"].eq("candidate_missing").mean()),
            "under_tradeoff_rate": float(group["taxonomy_label"].eq("under_tradeoff").mean()),
            "safe_under_fix_rate": float(group["taxonomy_label"].eq("safe_under_fix").mean()),
            "pure_harm_rate": float(group["taxonomy_label"].eq("pure_harm").mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pool", "split", "n"], ascending=[True, True, False])


def build_formal_headline_decision() -> pd.DataFrame:
    """写出 GPTPro 要求的 formal headline 锁定表。"""

    rows: List[Dict[str, object]] = []
    for pool_key, baseline_name in BASELINE_BY_POOL.items():
        rows.append(
            {
                "pool": pool_key,
                "entry_name": baseline_name,
                "decision_role": "formal_headline_locked",
                "formal_headline_model": baseline_name,
                "deployable_status": "formal_headline",
                "selected_by": "validation_only_prior_line",
                "test_used_for_selection": False,
                "formal_leaderboard_allowed": True,
                "diagnostic_only": False,
                "reason": "v222a no-harm gate locked test failed; headline remains fixed baseline.",
            }
        )
        rows.extend(
            [
                {
                    "pool": pool_key,
                    "entry_name": "v222a_bounded_residual",
                    "decision_role": "stop_formal_promotion",
                    "formal_headline_model": baseline_name,
                    "deployable_status": "diagnostic_only",
                    "selected_by": "validation_only_but_test_failed",
                    "test_used_for_selection": False,
                    "formal_leaderboard_allowed": False,
                    "diagnostic_only": True,
                    "reason": "bounded residual reduces some underestimation but does not pass locked no-harm gate.",
                },
                {
                    "pool": pool_key,
                    "entry_name": "v222a_noharm_gate",
                    "decision_role": "stop_formal_promotion",
                    "formal_headline_model": baseline_name,
                    "deployable_status": "diagnostic_only",
                    "selected_by": "validation_only_but_test_failed",
                    "test_used_for_selection": False,
                    "formal_leaderboard_allowed": False,
                    "diagnostic_only": True,
                    "reason": "validation passes but locked test formal gate fails.",
                },
                {
                    "pool": pool_key,
                    "entry_name": "oracle_safe_gate",
                    "decision_role": "upper_bound_only",
                    "formal_headline_model": baseline_name,
                    "deployable_status": "upper_bound_diagnostic_only",
                    "selected_by": "oracle_diagnostic",
                    "test_used_for_selection": False,
                    "formal_leaderboard_allowed": False,
                    "diagnostic_only": True,
                    "reason": "oracle uses per-sample true outcome metrics and is not deployable.",
                },
                {
                    "pool": pool_key,
                    "entry_name": "ridge_residual_peakfloor",
                    "decision_role": "low_under_diagnostic_reference",
                    "formal_headline_model": baseline_name,
                    "deployable_status": "diagnostic_reference",
                    "selected_by": "validation_only_prior_line",
                    "test_used_for_selection": False,
                    "formal_leaderboard_allowed": False,
                    "diagnostic_only": True,
                    "reason": "kept as low-under reference, not as closeout headline.",
                },
            ]
        )
    out = pd.DataFrame(rows)
    assert_no_forbidden_formal_names(out["entry_name"], "formal_headline_decision.entry_name")
    assert_no_forbidden_formal_names(out["formal_headline_model"], "formal_headline_decision.formal_headline_model")
    return out


def build_stop_evidence(locked_report: pd.DataFrame) -> pd.DataFrame:
    """从 no-harm locked report 生成 v222a 停止证据表。"""

    rows: List[Dict[str, object]] = []
    for pool_key in BASELINE_BY_POOL:
        val = locked_report[(locked_report["pool_key"].eq(pool_key)) & (locked_report["split"].eq("val"))].iloc[0]
        test = locked_report[(locked_report["pool_key"].eq(pool_key)) & (locked_report["split"].eq("test"))].iloc[0]
        rows.append(
            {
                "pool": pool_key,
                "pool_name": POOL_CN[pool_key],
                "validation_formal_pass": bool(val["formal_gate_pass_vs_baseline"]),
                "locked_test_formal_pass": bool(test["formal_gate_pass_vs_baseline"]),
                "locked_test_aggregate_noharm_pass": bool(test["aggregate_noharm_pass_vs_baseline"]),
                "locked_test_under_improved": bool(test["under_improved_vs_baseline"]),
                "test_rmse_delta_vs_baseline": float(test["rmse_delta_vs_baseline"]),
                "test_tail_delta_vs_baseline": float(test["tail_delta_vs_baseline"]),
                "test_under_reduction_vs_baseline": float(test["under_reduction_vs_baseline"]),
                "test_strong_under_reduction_vs_baseline": float(test["strong_under_reduction_vs_baseline"]),
                "test_coverage_m_rate": float(test["coverage_m_rate"]),
                "stop_v222a_formal_model_development": True,
                "stop_v222a_threshold_tuning": True,
                "stop_v222a_noharm_gate_optimization": True,
                "stop_v222a_bounded_residual_as_headline": True,
                "diagnosis": (
                    "validation pass but locked test fail; loose preserves under gain but harms RMSE/tail"
                    if pool_key == "loose_main_pool"
                    else "validation pass but locked test fail; strict protects RMSE/tail but worsens under"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_per_sample_failure_taxonomy(audit: pd.DataFrame) -> pd.DataFrame:
    """输出更紧凑的逐样本 taxonomy 表，保留规则布尔列便于复核。"""

    cols = [
        "pool",
        "split",
        "sample_id",
        "scenario_type",
        "strong_steer",
        "reverse",
        "zero_cross",
        "multi_correction",
        "vehicle_strong",
        "normal_curve",
        "extreme_peak",
        "high_tail_error",
        "baseline_name",
        "v222a_name",
        "oracle_best_allowed_candidate",
        "baseline_rmse",
        "baseline_tail_rmse",
        "baseline_under",
        "v222a_rmse",
        "v222a_tail_rmse",
        "v222a_under",
        "oracle_rmse",
        "oracle_tail_rmse",
        "oracle_under",
        "gain_v222a_rmse",
        "gain_v222a_tail",
        "gain_oracle_rmse",
        "gain_oracle_tail",
        "taxonomy_label",
        "rule_baseline_sufficient",
        "rule_safe_under_fix",
        "rule_under_tradeoff",
        "rule_pure_harm",
        "rule_selector_failed",
        "rule_candidate_missing",
        "rule_oracle_has_gain",
        "rule_m_missed_oracle",
    ]
    return audit[cols].copy()


def build_future_route_decision(audit_all: pd.DataFrame) -> pd.DataFrame:
    """根据 locked test closeout taxonomy 判断 v222b/v223 是否允许进入下一步讨论。"""

    rows: List[Dict[str, object]] = []
    scopes = list(BASELINE_BY_POOL.keys()) + ["combined"]
    for scope in scopes:
        if scope == "combined":
            sub = audit_all[audit_all["split"].eq("test")].copy()
            pool_name = "combined_test"
        else:
            sub = audit_all[audit_all["pool"].eq(scope) & audit_all["split"].eq("test")].copy()
            pool_name = POOL_CN[scope]
        if sub.empty:
            continue
        high_tail = sub[sub["high_tail_error"]].copy()
        high_tail_n = int(len(high_tail))
        label_counts = sub["taxonomy_label"].value_counts()
        main_failure = str(label_counts.index[0])
        selector_failed_rate = float(sub["taxonomy_label"].eq("selector_failed").mean())
        candidate_missing_rate = float(sub["taxonomy_label"].eq("candidate_missing").mean())
        high_tail_candidate_missing_rate = (
            float(high_tail["taxonomy_label"].eq("candidate_missing").mean()) if high_tail_n else np.nan
        )
        high_tail_oracle_clear_gain_rate = (
            float(
                (
                    (high_tail["gain_oracle_tail"] >= ORACLE_TAIL_GAIN_MARGIN)
                    | (high_tail["gain_oracle_rmse"] >= ORACLE_RMSE_GAIN_MARGIN)
                    | (high_tail["oracle_under"] < high_tail["baseline_under"])
                ).mean()
            )
            if high_tail_n
            else np.nan
        )
        v223_condition = bool(
            high_tail_n > 0
            and high_tail_candidate_missing_rate > 0.50
            and main_failure not in {"selector_failed", "under_tradeoff", "pure_harm"}
        )
        rows.append(
            {
                "pool": scope,
                "pool_name": pool_name,
                "basis_split": "test",
                "n": int(len(sub)),
                "high_tail_error_n": high_tail_n,
                "main_failure_source": main_failure,
                "selector_failed_rate": selector_failed_rate,
                "candidate_missing_rate": candidate_missing_rate,
                "high_tail_candidate_missing_rate": high_tail_candidate_missing_rate,
                "high_tail_oracle_clear_gain_rate": high_tail_oracle_clear_gain_rate,
                "v222b_allowed": False,
                "v222b_reason": (
                    "learned gate validation passed but locked test failed; larger neural gate is likely to overfit selector signal"
                ),
                "v223_allowed": v223_condition,
                "v223_reason": (
                    "candidate_missing dominates high-tail locked-test cases; future may discuss a new candidate generator"
                    if v223_condition
                    else "v223 remains prohibited unless high-tail candidate_missing exceeds 50 percent and allowed oracle cannot clearly improve baseline"
                ),
                "v222a_next_action": "STOP v222a formal model development; closeout only",
                "test_used_for_selection": False,
            }
        )
    return pd.DataFrame(rows)


def case_score(label: str, rows: pd.DataFrame) -> pd.Series:
    """给不同 case 目录选择最值得画图的样本。"""

    if label == "selector_failed":
        return (rows["v222a_tail_rmse"] - rows["oracle_tail_rmse"]) + (
            rows["v222a_rmse"] - rows["oracle_rmse"]
        )
    if label == "candidate_missing":
        return rows["baseline_tail_rmse"]
    if label == "safe_under_fix":
        return (rows["baseline_tail_rmse"] - rows["v222a_tail_rmse"]) + (
            rows["baseline_rmse"] - rows["v222a_rmse"]
        )
    return -rows["baseline_tail_rmse"]


def plot_case(pool: Dict[str, object], row: pd.Series, label: str, output_path: Path) -> None:
    """绘制单个样本的 true/B/M/O 曲线对比图。"""

    idx = int(row["array_index"])
    array_index_values = pool["array_index"]
    matches = np.where(array_index_values == idx)[0]
    if len(matches) != 1:
        raise AssertionError(f"{pool['pool_key']} array_index={idx} 无法唯一定位")
    local_idx = int(matches[0])
    t = np.arange(pool["true_steer"].shape[1], dtype=float) * 0.1
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(t, pool["true_steer"][local_idx], color="black", linewidth=2.2, label="true")
    plt.plot(t, pool["baseline_pred"][local_idx], color="#2b6cb0", linewidth=1.8, label=f"B {pool['baseline_name']}")
    plt.plot(t, pool["gate_pred"][local_idx], color="#c05621", linewidth=1.8, label="M v222a gate")
    plt.plot(
        t,
        pool["oracle_pred"][local_idx],
        color="#2f855a",
        linewidth=1.8,
        linestyle="--",
        label=f"O {row['oracle_best_allowed_candidate']}",
    )
    plt.axhline(0.0, color="#999999", linewidth=0.8)
    plt.title(f"{label} | {pool['pool_key']} | {row['split']} | {row['scenario_type']}")
    plt.xlabel("future horizon (s)")
    plt.ylabel("steering relative angle")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.25)
    text = (
        f"B tail={row['baseline_tail_rmse']:.3f}, M tail={row['v222a_tail_rmse']:.3f}, "
        f"O tail={row['oracle_tail_rmse']:.3f}\n"
        f"B under={int(row['baseline_under'])}, M under={int(row['v222a_under'])}, "
        f"O under={int(row['oracle_under'])}"
    )
    plt.figtext(0.12, 0.01, text, fontsize=8)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_case_figures(pool_payloads: Dict[str, Dict[str, object]], audit_all: pd.DataFrame) -> pd.DataFrame:
    """为 GPTPro 要求的四类 case 目录生成曲线图，并输出图索引。"""

    rows: List[Dict[str, object]] = []
    for pool_key, pool in pool_payloads.items():
        pool_rows = audit_all[audit_all["pool"].eq(pool_key)].copy()
        for label, dirname in CASE_DIR_BY_LABEL.items():
            candidates = pool_rows[pool_rows["taxonomy_label"].eq(label)].copy()
            if candidates.empty:
                continue
            candidates["_case_score"] = case_score(label, candidates)
            selected = candidates.sort_values("_case_score", ascending=False).head(TOP_CASES_PER_LABEL_POOL)
            for rank, (_, row) in enumerate(selected.iterrows(), start=1):
                safe_id = str(row["sample_id"]).replace("/", "_").replace("\\", "_").replace(":", "_")
                output_path = FIGURE_DIR / dirname / f"{pool_key}_{rank:02d}_{safe_id}.png"
                plot_case(pool, row, label, output_path)
                rows.append(
                    {
                        "pool": pool_key,
                        "taxonomy_label": label,
                        "rank": rank,
                        "sample_id": row["sample_id"],
                        "split": row["split"],
                        "scenario_type": row["scenario_type"],
                        "score": float(row["_case_score"]),
                        "figure_path": str(output_path.relative_to(OUT_DIR)),
                    }
                )
    return pd.DataFrame(rows)


def build_leakage_guard_result(
    formal_headline: pd.DataFrame,
    candidate_manifest: pd.DataFrame,
    future_decision: pd.DataFrame,
) -> pd.DataFrame:
    """写出 closeout 的 guardrail 结果。"""

    formal_names = formal_headline["entry_name"].astype(str).tolist() + formal_headline[
        "formal_headline_model"
    ].astype(str).tolist()
    formal_has_forbidden = any(
        token.lower() in name.lower() for name in formal_names for token in FORBIDDEN_FORMAL_SUBSTRINGS
    )
    candidate_oracle_bad = candidate_manifest[
        candidate_manifest["candidate_scope"].eq("formal")
        & candidate_manifest["candidate_name"].astype(str).str.contains(
            "|".join(FORBIDDEN_FORMAL_SUBSTRINGS), case=False, regex=True, na=False
        )
    ]
    rows = [
        {
            "check_name": "closeout_no_new_training",
            "status": "pass",
            "detail": "脚本只读取已有 NPZ/CSV，不 fit 新模型。",
        },
        {
            "check_name": "no_threshold_retuning",
            "status": "pass",
            "detail": "复用 no-harm gate 已锁定逐样本 gate_use_m，不重新选择 tau。",
        },
        {
            "check_name": "formal_headline_forbidden_identity",
            "status": "fail" if formal_has_forbidden else "pass",
            "detail": "formal_headline_decision 中不允许禁用模型身份。",
        },
        {
            "check_name": "oracle_diagnostic_only",
            "status": "pass",
            "detail": "O 只写为 diagnostic oracle，不写入 deployable model 或 formal headline。",
        },
        {
            "check_name": "candidate_oracle_allowed_formal_pool_only",
            "status": "fail" if not candidate_oracle_bad.empty else "pass",
            "detail": "逐样本 O 仅从 candidate_manifest 中 candidate_scope=formal 的固定候选池选择。",
        },
        {
            "check_name": "future_route_no_test_based_config",
            "status": "pass" if future_decision["test_used_for_selection"].eq(False).all() else "fail",
            "detail": "future_route_decision 只给是否允许未来讨论的 stop/unlock 条件，不反向调 test 配置。",
        },
    ]
    out = pd.DataFrame(rows)
    if out["status"].eq("fail").any():
        raise AssertionError("closeout guardrail failed:\n" + out.to_string(index=False))
    return out


def sha256_file(path: Path) -> str:
    """计算文件 sha256，用于 evidence pack 自校验。"""

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sha256_manifest() -> pd.DataFrame:
    """生成除 zip 本身以外所有输出文件的 sha256 manifest。"""

    rows: List[Dict[str, object]] = []
    for path in sorted(OUT_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() == ".zip":
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(OUT_DIR)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, LOG_DIR / "sha256_manifest.csv")
    return out


def zip_outputs() -> Path:
    """打包并校验本轮 closeout 产物。"""

    zip_path = OUT_DIR / "v222a_closeout_candidate_gap_audit_pack.zip"
    required_files = [
        "tables/formal_headline_decision.csv",
        "tables/v222a_stop_evidence.csv",
        "tables/oracle_vs_learned_gap.csv",
        "tables/candidate_gap_audit.csv",
        "tables/per_sample_failure_taxonomy.csv",
        "tables/bucket_failure_summary.csv",
        "tables/future_route_decision.csv",
        "reports/v222a_closeout_candidate_gap_audit_cn.md",
        "logs/closeout_manifest.json",
        "logs/sha256_manifest.csv",
        "logs/zip_verify.json",
    ]
    zip_verify: Dict[str, object] = {
        "zip_name": "v222a_closeout_candidate_gap_audit_pack.zip",
        "required_files": required_files,
        "bad_file": "pending",
        "file_count": None,
        "missing_required_files": "pending",
    }
    (LOG_DIR / "zip_verify.json").write_text(json.dumps(zip_verify, ensure_ascii=False, indent=2), encoding="utf-8")
    write_sha256_manifest()

    def build_zip() -> Tuple[object, set[str]]:
        """写 zip 并返回 testzip 结果和包内文件名。"""

        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(OUT_DIR.rglob("*")):
                if path == zip_path or not path.is_file():
                    continue
                zf.write(path, path.relative_to(OUT_DIR))
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad_file = zf.testzip()
            name_set = set(zf.namelist())
        return bad_file, name_set

    bad, names = build_zip()
    missing = [name for name in required_files if name not in names]
    if bad is not None:
        raise AssertionError(f"ZIP 预校验失败: {bad}")
    if missing:
        raise AssertionError(f"ZIP 预校验缺少必需文件: {missing}")

    zip_verify["bad_file"] = bad
    zip_verify["file_count"] = len(names)
    zip_verify["missing_required_files"] = missing
    (LOG_DIR / "zip_verify.json").write_text(json.dumps(zip_verify, ensure_ascii=False, indent=2), encoding="utf-8")
    write_sha256_manifest()

    bad, names = build_zip()
    missing = [name for name in required_files if name not in names]
    if bad is not None:
        raise AssertionError(f"ZIP 终校验失败: {bad}")
    if missing:
        raise AssertionError(f"ZIP 终校验缺少必需文件: {missing}")
    return zip_path


def make_report(
    stop_evidence: pd.DataFrame,
    oracle_gap: pd.DataFrame,
    audit_all: pd.DataFrame,
    future_decision: pd.DataFrame,
    figure_index: pd.DataFrame,
    zip_path: Path,
) -> None:
    """生成中文 closeout 报告。"""

    lines: List[str] = []
    lines.append("# v222a closeout + candidate gap audit 报告")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("- `v222a bounded residual / no-harm gate` formal 主线停止。")
    lines.append("- formal headline 锁定为：`loose_main_pool=avg_joint_focus`，`strict_main_pool=peak_floor_090`。")
    lines.append("- `v222a_bounded_residual`、`v222a_noharm_gate`、`oracle_safe_gate` 均为 diagnostic-only。")
    lines.append("- 本轮没有训练 v222b/v223，没有新增 router，没有重新选择 tau，也没有根据 locked test 反调配置。")
    lines.append("")
    lines.append("## v222a 停止证据")
    lines.append("")
    for _, row in stop_evidence.iterrows():
        lines.append(
            f"- {row['pool']}: validation pass={row['validation_formal_pass']}，"
            f"locked test pass={row['locked_test_formal_pass']}，"
            f"test RMSE delta={row['test_rmse_delta_vs_baseline']:.6f}，"
            f"tail delta={row['test_tail_delta_vs_baseline']:.6f}，"
            f"under reduction={row['test_under_reduction_vs_baseline']:.6f}。"
        )
    lines.append("")
    lines.append("## Oracle vs learned gate")
    lines.append("")
    test_gap = oracle_gap[oracle_gap["split"].eq("test")].copy()
    for _, row in test_gap.iterrows():
        lines.append(
            f"- {row['pool']}: learned tail gain={row['learned_gain_tail']:.6f}，"
            f"oracle tail gain={row['oracle_gain_tail']:.6f}，"
            f"oracle-minus-learned tail gap={row['oracle_minus_learned_tail_gap']:.6f}，"
            f"selector_failed_rate={row['selector_failed_rate']:.3f}，"
            f"candidate_missing_rate={row['candidate_missing_rate']:.3f}。"
        )
    lines.append("")
    lines.append("## Failure taxonomy")
    lines.append("")
    taxonomy_counts = (
        audit_all[audit_all["split"].eq("test")]
        .groupby(["pool", "taxonomy_label"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["pool", "n"], ascending=[True, False])
    )
    for pool_key in sorted(taxonomy_counts["pool"].unique()):
        pool_counts = taxonomy_counts[taxonomy_counts["pool"].eq(pool_key)]
        parts = [f"{r.taxonomy_label}={int(r.n)}" for r in pool_counts.itertuples(index=False)]
        lines.append(f"- {pool_key}: " + ", ".join(parts))
    lines.append("")
    lines.append("## Future route decision")
    lines.append("")
    for _, row in future_decision.iterrows():
        lines.append(
            f"- {row['pool']}: main_failure={row['main_failure_source']}，"
            f"v222b_allowed={row['v222b_allowed']}，v223_allowed={row['v223_allowed']}，"
            f"high_tail_candidate_missing_rate={row['high_tail_candidate_missing_rate']}。"
        )
    lines.append("")
    lines.append("## 诊断口径")
    lines.append("")
    lines.append("- O 是 best allowed formal candidate oracle diagnostic，逐样本选择规则为：先避免 severe-under，再最小 tail RMSE，再最小 sample RMSE。")
    lines.append("- `candidate_missing` 的含义是 baseline high-tail 且固定 formal candidate oracle 也不能清晰改善 baseline。")
    lines.append("- `selector_failed` 的含义是候选池存在可改善样本的候选，但 learned gate 输出没有抓住对应收益。")
    lines.append("- `vehicle_strong` 只作为 closeout bucket：由历史 ay/yaw/curvature 或 future curvature 输入强度的 pool 内 75 分位派生，不作为新模型特征。")
    lines.append("")
    lines.append("## 关键产物")
    lines.append("")
    for name in [
        "tables/formal_headline_decision.csv",
        "tables/v222a_stop_evidence.csv",
        "tables/oracle_vs_learned_gap.csv",
        "tables/candidate_gap_audit.csv",
        "tables/per_sample_failure_taxonomy.csv",
        "tables/bucket_failure_summary.csv",
        "tables/future_route_decision.csv",
        "logs/closeout_manifest.json",
        "logs/sha256_manifest.csv",
        zip_path.name,
    ]:
        lines.append(f"- `{name}`")
    if not figure_index.empty:
        lines.append(f"- case figures: {len(figure_index)} 张，索引见 `tables/case_figure_index.csv`")
    lines.append("")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "v222a_closeout_candidate_gap_audit_cn.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_manifest(
    pool_payloads: Dict[str, Dict[str, object]],
    zip_path: Path | None,
    future_decision: pd.DataFrame | None,
) -> None:
    """写出 closeout manifest，记录输入、定义和 guardrails。"""

    payload = {
        "stage": "v222a_closeout_candidate_gap_audit",
        "created_by": Path(__file__).name,
        "output_dir": str(OUT_DIR),
        "zip_path": str(zip_path) if zip_path is not None else None,
        "input_dirs": {
            "v221_formal_model_leaderboard": str(V221_DIR),
            "v222a_candidate_curve_cache": str(CACHE_DIR),
            "v222a_light_fusion_residual": str(V222A_DIR),
            "v222a_noharm_gate_diagnostic": str(NOHARM_DIR),
        },
        "formal_headline_locked": BASELINE_BY_POOL,
        "v222a_status": {
            "bounded_residual": "diagnostic_only",
            "noharm_gate": "diagnostic_only",
            "oracle_safe_gate": "upper_bound_diagnostic_only",
            "v222b_allowed": False,
            "v223_training_started": False,
        },
        "taxonomy_priority": [
            "baseline_sufficient",
            "candidate_missing",
            "safe_under_fix",
            "under_tradeoff",
            "selector_failed",
            "pure_harm",
            "selector_failed_default_if_oracle_has_gain",
            "baseline_sufficient_default",
        ],
        "thresholds": {
            "tail_start_index": TAIL_START_INDEX,
            "strong_steer_threshold": STRONG_STEER_THRESHOLD,
            "extreme_peak_threshold": EXTREME_PEAK_THRESHOLD,
            "safe_under_rmse_margin": SAFE_UNDER_RMSE_MARGIN,
            "safe_under_tail_margin": SAFE_UNDER_TAIL_MARGIN,
            "oracle_rmse_gain_margin": ORACLE_RMSE_GAIN_MARGIN,
            "oracle_tail_gain_margin": ORACLE_TAIL_GAIN_MARGIN,
        },
        "oracle_rule": (
            "diagnostic-only best allowed formal candidate; lexicographic order: "
            "no severe-under, lower tail RMSE, lower sample RMSE"
        ),
        "allowed_oracle_candidates_by_pool": {
            pool_key: pool["allowed_candidate_names"] for pool_key, pool in pool_payloads.items()
        },
        "vehicle_strong_thresholds_by_pool": {
            pool_key: pool["vehicle_thresholds"] for pool_key, pool in pool_payloads.items()
        },
        "guardrails": [
            "no new training",
            "no threshold retuning",
            "no v222b",
            "no v223",
            "no multi-candidate router",
            "oracle is diagnostic only",
            "test is reporting/audit only and not used for selection",
        ],
        "future_route_decision": future_decision.to_dict(orient="records") if future_decision is not None else None,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "closeout_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """主流程：读已有产物、生成 closeout 表/图/报告、打包并校验。"""

    clean_out_dir()
    sample_manifest, candidate_manifest, gate_decision, locked_report = read_required_inputs()

    pool_payloads: Dict[str, Dict[str, object]] = {}
    audit_frames: List[pd.DataFrame] = []
    gap_frames: List[pd.DataFrame] = []
    for pool_key in BASELINE_BY_POOL:
        pool = load_pool_payload(pool_key, sample_manifest, candidate_manifest, gate_decision)
        pool_payloads[pool_key] = pool
        audit = build_candidate_gap_audit(pool)
        audit_frames.append(audit)
        gap_frames.append(summarize_oracle_vs_learned(pool, audit))

    audit_all = pd.concat(audit_frames, ignore_index=True)
    oracle_gap = pd.concat(gap_frames, ignore_index=True)
    formal_headline = build_formal_headline_decision()
    stop_evidence = build_stop_evidence(locked_report)
    failure_taxonomy = build_per_sample_failure_taxonomy(audit_all)
    bucket_summary = build_bucket_failure_summary(audit_all)
    future_decision = build_future_route_decision(audit_all)
    guard = build_leakage_guard_result(formal_headline, candidate_manifest, future_decision)
    figure_index = make_case_figures(pool_payloads, audit_all)

    write_csv(formal_headline, TABLE_DIR / "formal_headline_decision.csv")
    write_csv(stop_evidence, TABLE_DIR / "v222a_stop_evidence.csv")
    write_csv(oracle_gap, TABLE_DIR / "oracle_vs_learned_gap.csv")
    write_csv(audit_all, TABLE_DIR / "candidate_gap_audit.csv")
    write_csv(failure_taxonomy, TABLE_DIR / "per_sample_failure_taxonomy.csv")
    write_csv(bucket_summary, TABLE_DIR / "bucket_failure_summary.csv")
    write_csv(future_decision, TABLE_DIR / "future_route_decision.csv")
    write_csv(guard, TABLE_DIR / "leakage_guard_result.csv")
    write_csv(figure_index, TABLE_DIR / "case_figure_index.csv")

    write_manifest(pool_payloads, None, future_decision)
    placeholder_zip = OUT_DIR / "v222a_closeout_candidate_gap_audit_pack.zip"
    make_report(stop_evidence, oracle_gap, audit_all, future_decision, figure_index, placeholder_zip)
    write_sha256_manifest()
    zip_path = zip_outputs()
    write_manifest(pool_payloads, zip_path, future_decision)
    write_sha256_manifest()
    zip_path = zip_outputs()

    print("v222a closeout candidate gap audit finished.")
    print(f"output_dir={OUT_DIR}")
    print(f"report={REPORT_DIR / 'v222a_closeout_candidate_gap_audit_cn.md'}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
