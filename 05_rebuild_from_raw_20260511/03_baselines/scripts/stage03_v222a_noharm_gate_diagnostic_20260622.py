#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v222a no-harm gate 诊断。

本脚本执行 GPTPro 新一轮指令：

1. v222a_gain_harm_decomposition
2. oracle safe gate upper bound
3. binary validation-only no-harm gate
4. 判断 v222a 是否应该停止，或是否值得进入下一轮

边界：
- baseline B 和 selected residual M 均来自已经完成的 v222a 产物；
- safe/useful/oracle_use_M 只作为训练标签或 diagnostic label，不进入推理特征；
- gate 的推理输入只使用 v222a cache 中已经审计过的 `feature_matrix`；
- gate 模型只在 train split 拟合；
- tau_safe/tau_useful/tau_harm 只在 validation split 选择；
- test split 只在 validation-selected gate 固定后报告一次；
- 不做多候选 router，不进入 v222b/v223。
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


REPO_ROOT = Path(__file__).resolve().parents[3]
REBUILD_ROOT = REPO_ROOT / "05_rebuild_from_raw_20260511"
BASE_DIR = REBUILD_ROOT / "03_baselines"
CACHE_DIR = BASE_DIR / "v222a_candidate_curve_cache_20260622"
V222A_DIR = BASE_DIR / "v222a_light_fusion_residual_20260622"
OUT_DIR = BASE_DIR / "v222a_noharm_gate_diagnostic_20260622"
TABLE_DIR = OUT_DIR / "tables"
REPORT_DIR = OUT_DIR / "reports"
MODEL_DIR = OUT_DIR / "models"
LOG_DIR = OUT_DIR / "logs"


BASELINE_BY_POOL = {
    "loose_main_pool": "avg_joint_focus",
    "strict_main_pool": "peak_floor_090",
}

FORBIDDEN_FEATURE_TOKENS = [
    "sample_id",
    "event_uid",
    "split",
    "subject",
    "subject_id",
    "panel",
    "panel_label",
    "manifest",
    "row_index",
    "roll_phase",
    "oracle",
    "true",
    "label",
    "target",
    "metric",
    "cost",
    "rmse",
    "false_large",
    "severe_under",
    "wrong_side",
    "usable_large",
    "physical_utility",
    "large_true",
    "quiet_true",
    "late_true",
    "reversal_true",
]

FORBIDDEN_OUTPUT_SUBSTRINGS = [
    "W3_B4_original_soft",
    "oracle_model",
    "true_label",
    "fallback",
]

SAMPLE_SAFE_RMSE_MARGIN = 0.0
SAMPLE_SAFE_TAIL_MARGIN = 0.0
USEFUL_TAIL_IMPROVE = 0.02
AGG_NOHARM_TOL = 0.002

TAU_SAFE_GRID = [0.40, 0.50, 0.60, 0.70, 0.80]
TAU_USEFUL_GRID = [0.20, 0.30, 0.40, 0.50, 0.60]
TAU_TAIL_HARM_GRID = [0.00, 0.02, 0.05]


@dataclass
class GateModels:
    """保存 no-harm gate 的三个轻量预测器。"""

    safe_model: object
    useful_model: object
    tail_delta_model: object
    safe_constant: float | None
    useful_constant: float | None


def ensure_dirs() -> None:
    """创建输出目录。"""

    for path in [TABLE_DIR, REPORT_DIR, MODEL_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """清理旧输出，避免旧 gate 表混入本轮结果。"""

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一用 utf-8-sig 写 CSV，方便 Excel 直接查看中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def assert_finite(name: str, arr: np.ndarray) -> None:
    """确保数组没有 NaN 或 inf。"""

    if not np.isfinite(arr).all():
        bad = int(np.size(arr) - np.isfinite(arr).sum())
        raise AssertionError(f"{name} 包含非有限值：bad={bad}")


def forbidden_feature_token(name: str) -> str:
    """返回特征名命中的禁用 token。"""

    lowered = str(name).lower()
    for token in FORBIDDEN_FEATURE_TOKENS:
        if token in lowered:
            return token
    return ""


def audit_feature_schema(pool_key: str, feature_names: List[str]) -> pd.DataFrame:
    """审计 gate 的推理特征名，不允许身份字段或目标派生字段。"""

    rows: List[Dict[str, object]] = []
    for idx, name in enumerate(feature_names):
        bad = forbidden_feature_token(name)
        rows.append(
            {
                "pool_key": pool_key,
                "feature_order": idx,
                "feature_name": name,
                "forbidden_token": bad,
                "guard_status": "fail" if bad else "pass",
            }
        )
    out = pd.DataFrame(rows)
    bad_rows = out[out["guard_status"].eq("fail")]
    if not bad_rows.empty:
        raise AssertionError("gate feature schema 命中禁用字段：\n" + bad_rows.to_string(index=False))
    return out


def assert_no_forbidden_outputs(names: Iterable[str]) -> None:
    """输出名不能混入禁用模型身份。"""

    bad: List[str] = []
    for name in names:
        lowered = str(name).lower()
        for token in FORBIDDEN_OUTPUT_SUBSTRINGS:
            if token.lower() in lowered:
                bad.append(str(name))
    if bad:
        raise AssertionError("输出名含禁用身份：" + ", ".join(sorted(set(bad))))


def peak_values(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """返回每条曲线的绝对峰值和峰值符号。"""

    idx = np.nanargmax(np.abs(arr), axis=1)
    signed = arr[np.arange(arr.shape[0]), idx]
    return np.abs(signed), signed


def per_sample_curve_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """计算逐样本 steering 指标。"""

    diff = y_pred - y_true
    tail_mask = np.arange(y_true.shape[1]) >= 10
    sample_rmse = np.sqrt(np.mean(np.square(diff), axis=1))
    tail_rmse = np.sqrt(np.mean(np.square(diff[:, tail_mask]), axis=1))
    true_peak_abs, true_peak_signed = peak_values(y_true)
    pred_peak_abs, pred_peak_signed = peak_values(y_pred)
    return {
        "sample_rmse": sample_rmse,
        "tail_rmse": tail_rmse,
        "true_peak_abs": true_peak_abs,
        "pred_peak_abs": pred_peak_abs,
        "direction_ok": np.sign(true_peak_signed) == np.sign(pred_peak_signed),
        "severe_under": pred_peak_abs < (0.5 * true_peak_abs),
        "strong_response": true_peak_abs >= 1.0,
    }


def aggregate_metrics(
    pool_key: str,
    pool_name: str,
    output_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variant_type: str,
) -> Dict[str, object]:
    """计算某个 split 的整体指标。"""

    diff = y_pred - y_true
    tail_mask = np.arange(y_true.shape[1]) >= 10
    sample = per_sample_curve_metrics(y_true, y_pred)
    strong = sample["strong_response"]
    strong_rmse = np.nan
    strong_under = np.nan
    if strong.any():
        strong_rmse = float(np.sqrt(np.mean(np.square(diff[strong]))))
        strong_under = float(np.mean(sample["severe_under"][strong]))
    return {
        "pool_key": pool_key,
        "pool_name": pool_name,
        "output_name": output_name,
        "variant_type": variant_type,
        "split": split_name,
        "n": int(y_true.shape[0]),
        "steer_rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "steer_tail_rmse_1to2s": float(np.sqrt(np.mean(np.square(diff[:, tail_mask])))),
        "steer_sample_rmse_mean": float(np.mean(sample["sample_rmse"])),
        "steer_sample_rmse_p90": float(np.quantile(sample["sample_rmse"], 0.90)),
        "steer_direction_acc": float(np.mean(sample["direction_ok"])),
        "steer_severe_under_rate": float(np.mean(sample["severe_under"])),
        "strong_response_n": int(strong.sum()),
        "strong_response_rmse": strong_rmse,
        "strong_response_severe_under_rate": strong_under,
        "true_peak_abs_mean": float(np.mean(sample["true_peak_abs"])),
        "pred_peak_abs_mean": float(np.mean(sample["pred_peak_abs"])),
    }


def split_mask(split_values: np.ndarray, split_name: str) -> np.ndarray:
    """生成 split 掩码。"""

    if split_name == "all":
        return np.ones(len(split_values), dtype=bool)
    return split_values.astype(str) == split_name


def metrics_for_splits(
    pool_key: str,
    pool_name: str,
    output_name: str,
    split_values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variant_type: str,
    splits: Iterable[str],
) -> pd.DataFrame:
    """按多个 split 聚合指标。"""

    rows: List[Dict[str, object]] = []
    for split_name in splits:
        mask = split_mask(split_values, split_name)
        if not mask.any():
            continue
        rows.append(
            aggregate_metrics(
                pool_key,
                pool_name,
                output_name,
                split_name,
                y_true[mask],
                y_pred[mask],
                variant_type,
            )
        )
    return pd.DataFrame(rows)


def load_pool_payload(pool_key: str) -> Dict[str, object]:
    """读取一个 pool 的 cache、baseline 预测和 selected residual 预测。"""

    cache_path = CACHE_DIR / f"candidate_predictions_{pool_key}.npz"
    selected_path = V222A_DIR / f"v222a_selected_predictions_{pool_key}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"缺少 cache：{cache_path}")
    if not selected_path.exists():
        raise FileNotFoundError(f"缺少 selected residual：{selected_path}")

    with np.load(cache_path, allow_pickle=False) as cache:
        candidate_names = cache["candidate_names"].astype(str).tolist()
        baseline_name = BASELINE_BY_POOL[pool_key]
        if baseline_name not in candidate_names:
            raise AssertionError(f"{pool_key} cache 缺少 baseline：{baseline_name}")
        baseline_idx = candidate_names.index(baseline_name)
        payload: Dict[str, object] = {
            "pool_key": pool_key,
            "candidate_names": candidate_names,
            "baseline_name": baseline_name,
            "baseline_pred": cache["predictions"][:, baseline_idx, :].astype(np.float32),
            "true_steer": cache["true_steer"].astype(np.float32),
            "feature_matrix": cache["feature_matrix"].astype(np.float32),
            "feature_names": cache["feature_names"].astype(str).tolist(),
            "split": cache["split"].astype(str),
            "array_index": cache["array_index"].astype(np.int64),
            "event_uid": cache["event_uid"].astype(str),
        }
    with np.load(selected_path, allow_pickle=False) as selected:
        payload["selected_name"] = str(selected["selected_output_name"][0])
        payload["selected_pred"] = selected["pred_v222a_val_selected"].astype(np.float32)

    assert_finite(f"{pool_key}:baseline_pred", payload["baseline_pred"])
    assert_finite(f"{pool_key}:selected_pred", payload["selected_pred"])
    assert_finite(f"{pool_key}:true_steer", payload["true_steer"])
    assert_finite(f"{pool_key}:feature_matrix", payload["feature_matrix"])
    audit_feature_schema(pool_key, payload["feature_names"])
    assert_no_forbidden_outputs([payload["baseline_name"], payload["selected_name"]])
    return payload


def build_gain_harm_table(pool: Dict[str, object], sample_manifest: pd.DataFrame) -> pd.DataFrame:
    """构建 baseline B 与 residual M 的逐样本 gain/harm 分解。"""

    pool_key = str(pool["pool_key"])
    pool_samples = sample_manifest[sample_manifest["pool_key"].eq(pool_key)].copy().reset_index(drop=True)
    y_true = pool["true_steer"]
    b_pred = pool["baseline_pred"]
    m_pred = pool["selected_pred"]
    split_values = pool["split"]
    assert isinstance(y_true, np.ndarray) and isinstance(b_pred, np.ndarray) and isinstance(m_pred, np.ndarray)
    if len(pool_samples) != y_true.shape[0]:
        raise AssertionError(f"{pool_key} sample_manifest 行数与 cache 不一致")

    b = per_sample_curve_metrics(y_true, b_pred)
    m = per_sample_curve_metrics(y_true, m_pred)
    out = pool_samples.copy()
    out["baseline_name"] = str(pool["baseline_name"])
    out["model_name"] = str(pool["selected_name"])
    out["split"] = split_values.astype(str)
    out["b_sample_rmse"] = b["sample_rmse"]
    out["m_sample_rmse"] = m["sample_rmse"]
    out["rmse_delta_m_minus_b"] = out["m_sample_rmse"] - out["b_sample_rmse"]
    out["b_tail_rmse"] = b["tail_rmse"]
    out["m_tail_rmse"] = m["tail_rmse"]
    out["tail_delta_m_minus_b"] = out["m_tail_rmse"] - out["b_tail_rmse"]
    out["true_steer_peak_abs"] = b["true_peak_abs"]
    out["b_pred_peak_abs"] = b["pred_peak_abs"]
    out["m_pred_peak_abs"] = m["pred_peak_abs"]
    out["b_severe_under"] = b["severe_under"]
    out["m_severe_under"] = m["severe_under"]
    out["strong_response"] = b["strong_response"]
    out["under_fixed_by_m"] = out["b_severe_under"] & ~out["m_severe_under"]
    out["under_regressed_by_m"] = ~out["b_severe_under"] & out["m_severe_under"]
    out["rmse_harmed_by_m"] = out["rmse_delta_m_minus_b"] > SAMPLE_SAFE_RMSE_MARGIN
    out["tail_harmed_by_m"] = out["tail_delta_m_minus_b"] > SAMPLE_SAFE_TAIL_MARGIN
    out["safe_label"] = ~out["rmse_harmed_by_m"] & ~out["tail_harmed_by_m"]
    out["useful_label"] = (
        out["under_fixed_by_m"]
        | (out["strong_response"] & (out["tail_delta_m_minus_b"] <= -USEFUL_TAIL_IMPROVE))
        | (out["rmse_delta_m_minus_b"] <= -USEFUL_TAIL_IMPROVE)
    )
    out["oracle_use_m"] = out["safe_label"] & out["useful_label"]
    out["gain_harm_bucket"] = np.select(
        [
            out["oracle_use_m"],
            ~out["safe_label"],
            out["safe_label"] & ~out["useful_label"],
        ],
        ["safe_and_useful", "harmful", "safe_not_useful"],
        default="other",
    )
    return out


def fit_classifier_or_constant(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Tuple[object | None, float | None]:
    """训练轻量二分类器；如果标签只有一类，则退化为常数概率。"""

    unique = np.unique(y_train.astype(int))
    if len(unique) < 2:
        return None, float(unique[0])
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=4,
        min_samples_leaf=12,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X_train, y_train.astype(int))
    return model, None


def predict_probability(model: object | None, constant: float | None, X: np.ndarray) -> np.ndarray:
    """输出正类概率。"""

    if model is None:
        if constant is None:
            raise AssertionError("model 和 constant 不能同时为空")
        return np.full(X.shape[0], constant, dtype=np.float64)
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        return np.full(X.shape[0], float(model.classes_[0]), dtype=np.float64)
    class_list = list(model.classes_)
    pos_idx = class_list.index(1) if 1 in class_list else int(np.argmax(class_list))
    return proba[:, pos_idx].astype(np.float64)


def fit_gate_models(pool: Dict[str, object], gain_harm: pd.DataFrame) -> GateModels:
    """在 train split 上拟合 safe/useful/tail-delta 三个轻量预测器。"""

    X = pool["feature_matrix"]
    split_values = pool["split"]
    assert isinstance(X, np.ndarray) and isinstance(split_values, np.ndarray)
    train_mask = split_values.astype(str) == "train"
    if not train_mask.any():
        raise AssertionError(f"{pool['pool_key']} 缺少 train split")

    safe_model, safe_const = fit_classifier_or_constant(
        X[train_mask],
        gain_harm.loc[train_mask, "safe_label"].to_numpy(bool),
        seed=20260622,
    )
    useful_model, useful_const = fit_classifier_or_constant(
        X[train_mask],
        gain_harm.loc[train_mask, "useful_label"].to_numpy(bool),
        seed=20260623,
    )
    tail_model = RandomForestRegressor(
        n_estimators=180,
        max_depth=4,
        min_samples_leaf=12,
        random_state=20260624,
        n_jobs=1,
    )
    tail_model.fit(X[train_mask], gain_harm.loc[train_mask, "tail_delta_m_minus_b"].to_numpy(float))
    return GateModels(safe_model, useful_model, tail_model, safe_const, useful_const)


def apply_gate_predictions(pool: Dict[str, object], models: GateModels, gain_harm: pd.DataFrame) -> pd.DataFrame:
    """为所有样本生成 p_safe/p_useful/predicted_tail_harm。"""

    X = pool["feature_matrix"]
    assert isinstance(X, np.ndarray)
    out = gain_harm.copy()
    out["p_safe"] = predict_probability(models.safe_model, models.safe_constant, X)
    out["p_useful"] = predict_probability(models.useful_model, models.useful_constant, X)
    out["predicted_tail_harm"] = np.asarray(models.tail_delta_model.predict(X), dtype=np.float64)
    return out


def choose_prediction(pool: Dict[str, object], use_m: np.ndarray) -> np.ndarray:
    """根据 gate 决策拼接 baseline B 和 residual M。"""

    b_pred = pool["baseline_pred"]
    m_pred = pool["selected_pred"]
    assert isinstance(b_pred, np.ndarray) and isinstance(m_pred, np.ndarray)
    return np.where(use_m[:, None], m_pred, b_pred).astype(np.float32)


def aggregate_for_decision(
    pool: Dict[str, object],
    use_m: np.ndarray,
    output_name: str,
    split_name: str,
    variant_type: str,
) -> Dict[str, object]:
    """对某个 gate 决策计算 split 聚合指标。"""

    pool_key = str(pool["pool_key"])
    pool_name = "可用主池" if pool_key == "loose_main_pool" else "严格主池"
    split_values = pool["split"]
    y_true = pool["true_steer"]
    assert isinstance(split_values, np.ndarray) and isinstance(y_true, np.ndarray)
    mask = split_mask(split_values, split_name)
    pred = choose_prediction(pool, use_m)
    row = aggregate_metrics(pool_key, pool_name, output_name, split_name, y_true[mask], pred[mask], variant_type)
    row["coverage_m_rate"] = float(np.mean(use_m[mask]))
    row["coverage_m_n"] = int(np.sum(use_m[mask]))
    return row


def baseline_rows(pool: Dict[str, object]) -> pd.DataFrame:
    """输出固定 baseline B 的 train/val/test/all 指标。"""

    pool_key = str(pool["pool_key"])
    pool_name = "可用主池" if pool_key == "loose_main_pool" else "严格主池"
    return metrics_for_splits(
        pool_key,
        pool_name,
        str(pool["baseline_name"]),
        pool["split"],
        pool["true_steer"],
        pool["baseline_pred"],
        "fixed_baseline_B",
        ["train", "val", "test", "all"],
    )


def selected_rows(pool: Dict[str, object]) -> pd.DataFrame:
    """输出 selected residual M 的 train/val/test/all 指标。"""

    pool_key = str(pool["pool_key"])
    pool_name = "可用主池" if pool_key == "loose_main_pool" else "严格主池"
    return metrics_for_splits(
        pool_key,
        pool_name,
        str(pool["selected_name"]),
        pool["split"],
        pool["true_steer"],
        pool["selected_pred"],
        "selected_residual_M",
        ["train", "val", "test", "all"],
    )


def oracle_safe_gate_rows(pool: Dict[str, object], gain_harm: pd.DataFrame) -> pd.DataFrame:
    """计算 diagnostic-only oracle safe gate 上限。"""

    use_m = gain_harm["oracle_use_m"].to_numpy(bool)
    rows = [
        aggregate_for_decision(pool, use_m, "oracle_safe_gate_upper_bound", split_name, "diagnostic_oracle_gate")
        for split_name in ["train", "val", "test", "all"]
    ]
    out = pd.DataFrame(rows)
    for split_name in ["train", "val", "test", "all"]:
        mask = split_mask(pool["split"], split_name)
        gh = gain_harm.loc[mask]
        total_under = int(gh["b_severe_under"].sum())
        safe_fixed = int((gh["under_fixed_by_m"] & gh["oracle_use_m"]).sum())
        out.loc[out["split"].eq(split_name), "baseline_under_n"] = total_under
        out.loc[out["split"].eq(split_name), "safe_under_fix_n"] = safe_fixed
        out.loc[out["split"].eq(split_name), "safe_under_fix_coverage"] = (
            safe_fixed / total_under if total_under else np.nan
        )
    return out


def build_gate_tradeoff(pool: Dict[str, object], gate_features: pd.DataFrame, baseline_metric: pd.DataFrame) -> pd.DataFrame:
    """在 validation 上枚举 gate 阈值并计算 no-harm-first 选择表。"""

    rows: List[Dict[str, object]] = []
    split_values = pool["split"]
    assert isinstance(split_values, np.ndarray)
    base_val = baseline_metric[baseline_metric["split"].eq("val")].iloc[0]
    for tau_safe in TAU_SAFE_GRID:
        for tau_useful in TAU_USEFUL_GRID:
            for tau_tail_harm in TAU_TAIL_HARM_GRID:
                use_m = (
                    (gate_features["p_safe"].to_numpy(float) >= tau_safe)
                    & (gate_features["p_useful"].to_numpy(float) >= tau_useful)
                    & (gate_features["predicted_tail_harm"].to_numpy(float) <= tau_tail_harm)
                )
                val_row = aggregate_for_decision(
                    pool,
                    use_m,
                    f"noharm_gate_s{tau_safe:.2f}_u{tau_useful:.2f}_h{tau_tail_harm:.2f}",
                    "val",
                    "validation_candidate_gate",
                )
                val_row["tau_safe"] = tau_safe
                val_row["tau_useful"] = tau_useful
                val_row["tau_tail_harm"] = tau_tail_harm
                val_row["baseline_val_rmse"] = float(base_val["steer_rmse"])
                val_row["baseline_val_tail_rmse"] = float(base_val["steer_tail_rmse_1to2s"])
                val_row["baseline_val_under_rate"] = float(base_val["steer_severe_under_rate"])
                val_row["baseline_val_strong_under_rate"] = float(
                    base_val["strong_response_severe_under_rate"]
                )
                val_row["rmse_delta_vs_baseline"] = val_row["steer_rmse"] - val_row["baseline_val_rmse"]
                val_row["tail_delta_vs_baseline"] = (
                    val_row["steer_tail_rmse_1to2s"] - val_row["baseline_val_tail_rmse"]
                )
                val_row["under_reduction_vs_baseline"] = (
                    val_row["baseline_val_under_rate"] - val_row["steer_severe_under_rate"]
                )
                val_row["strong_under_reduction_vs_baseline"] = (
                    val_row["baseline_val_strong_under_rate"]
                    - val_row["strong_response_severe_under_rate"]
                )
                val_row["aggregate_noharm_pass"] = (
                    val_row["rmse_delta_vs_baseline"] <= AGG_NOHARM_TOL
                    and val_row["tail_delta_vs_baseline"] <= AGG_NOHARM_TOL
                )
                val_row["under_improved"] = (
                    val_row["under_reduction_vs_baseline"] > 0
                    or val_row["strong_under_reduction_vs_baseline"] > 0
                )
                val_row["formal_gate_pass"] = (
                    val_row["aggregate_noharm_pass"]
                    and val_row["under_improved"]
                    and val_row["coverage_m_n"] > 0
                )
                val_row["selection_score"] = (
                    (1.0 if val_row["formal_gate_pass"] else 0.0) * 1000.0
                    + 20.0 * max(float(val_row["under_reduction_vs_baseline"]), 0.0)
                    + 20.0 * max(float(val_row["strong_under_reduction_vs_baseline"]), 0.0)
                    + 0.10 * float(val_row["coverage_m_rate"])
                    - 50.0 * max(float(val_row["rmse_delta_vs_baseline"]), 0.0)
                    - 50.0 * max(float(val_row["tail_delta_vs_baseline"]), 0.0)
                )
                rows.append(val_row)
    out = pd.DataFrame(rows)
    out = out.sort_values(
        [
            "pool_key",
            "formal_gate_pass",
            "selection_score",
            "rmse_delta_vs_baseline",
            "tail_delta_vs_baseline",
            "coverage_m_rate",
        ],
        ascending=[True, False, False, True, True, False],
    )
    out["validation_rank"] = out.groupby("pool_key").cumcount() + 1
    return out


def selected_gate_report(
    pool: Dict[str, object],
    gate_features: pd.DataFrame,
    selected: pd.Series,
    baseline_metric: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """用 validation-selected 阈值锁定 gate，并只报告一次 test。"""

    use_m = (
        (gate_features["p_safe"].to_numpy(float) >= float(selected["tau_safe"]))
        & (gate_features["p_useful"].to_numpy(float) >= float(selected["tau_useful"]))
        & (gate_features["predicted_tail_harm"].to_numpy(float) <= float(selected["tau_tail_harm"]))
    )
    rows: List[Dict[str, object]] = []
    for split_name in ["train", "val", "test", "all"]:
        row = aggregate_for_decision(
            pool,
            use_m,
            "v222a_validation_selected_noharm_gate",
            split_name,
            "validation_locked_binary_gate",
        )
        base = baseline_metric[baseline_metric["split"].eq(split_name)].iloc[0]
        row["baseline_rmse"] = float(base["steer_rmse"])
        row["baseline_tail_rmse"] = float(base["steer_tail_rmse_1to2s"])
        row["baseline_under_rate"] = float(base["steer_severe_under_rate"])
        row["baseline_strong_under_rate"] = float(base["strong_response_severe_under_rate"])
        row["rmse_delta_vs_baseline"] = row["steer_rmse"] - row["baseline_rmse"]
        row["tail_delta_vs_baseline"] = row["steer_tail_rmse_1to2s"] - row["baseline_tail_rmse"]
        row["under_reduction_vs_baseline"] = row["baseline_under_rate"] - row["steer_severe_under_rate"]
        row["strong_under_reduction_vs_baseline"] = (
            row["baseline_strong_under_rate"] - row["strong_response_severe_under_rate"]
        )
        row["aggregate_noharm_pass_vs_baseline"] = (
            row["rmse_delta_vs_baseline"] <= AGG_NOHARM_TOL
            and row["tail_delta_vs_baseline"] <= AGG_NOHARM_TOL
        )
        row["under_improved_vs_baseline"] = (
            row["under_reduction_vs_baseline"] > 0
            or row["strong_under_reduction_vs_baseline"] > 0
        )
        row["formal_gate_pass_vs_baseline"] = (
            row["aggregate_noharm_pass_vs_baseline"]
            and row["under_improved_vs_baseline"]
            and row["coverage_m_n"] > 0
        )
        row["tau_safe"] = float(selected["tau_safe"])
        row["tau_useful"] = float(selected["tau_useful"])
        row["tau_tail_harm"] = float(selected["tau_tail_harm"])
        row["selected_by"] = "validation_only"
        row["test_used_for_selection"] = False
        rows.append(row)

    decision = gate_features.copy()
    decision["tau_safe"] = float(selected["tau_safe"])
    decision["tau_useful"] = float(selected["tau_useful"])
    decision["tau_tail_harm"] = float(selected["tau_tail_harm"])
    decision["gate_use_m"] = use_m
    decision["gate_output_name"] = "v222a_validation_selected_noharm_gate"
    return pd.DataFrame(rows), decision


def make_report(
    baseline_all: pd.DataFrame,
    selected_all: pd.DataFrame,
    oracle_all: pd.DataFrame,
    tradeoff_all: pd.DataFrame,
    locked_all: pd.DataFrame,
    zip_path: Path,
) -> None:
    """生成中文总结报告。"""

    lines: List[str] = []
    lines.append("# v222a no-harm gate 诊断报告")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    for pool_key in sorted(locked_all["pool_key"].unique()):
        locked_test = locked_all[(locked_all["pool_key"].eq(pool_key)) & (locked_all["split"].eq("test"))].iloc[0]
        selected_val = tradeoff_all[(tradeoff_all["pool_key"].eq(pool_key)) & (tradeoff_all["validation_rank"].eq(1))].iloc[0]
        val_pass_text = "通过" if bool(selected_val["formal_gate_pass"]) else "未通过"
        test_pass_text = "通过" if bool(locked_test.formal_gate_pass_vs_baseline) else "未通过"
        lines.append(
            f"- {pool_key}: validation no-harm gate {val_pass_text}，locked test {test_pass_text}。"
            f"test RMSE delta={locked_test.rmse_delta_vs_baseline:.6f}, "
            f"tail delta={locked_test.tail_delta_vs_baseline:.6f}, "
            f"under reduction={locked_test.under_reduction_vs_baseline:.6f}, "
            f"coverage={locked_test.coverage_m_rate:.6f}。"
        )
    lines.append("")
    lines.append("## GPTPro 指令执行情况")
    lines.append("")
    lines.append("- 已完成 gain/harm decomposition。")
    lines.append("- 已完成 diagnostic-only oracle safe gate upper bound。")
    lines.append("- 已完成 binary validation-only no-harm gate。")
    lines.append("- test 只在 validation-selected gate 固定后报告一次。")
    lines.append("- 本轮未训练 v222b/v223，也未做多候选 router。")
    lines.append("")
    lines.append("## 关键文件")
    lines.append("")
    lines.append("- `tables/gain_harm_decomposition.csv`")
    lines.append("- `tables/oracle_safe_gate_report.csv`")
    lines.append("- `tables/val_gate_tradeoff_table.csv`")
    lines.append("- `tables/test_locked_gate_report.csv`")
    lines.append("- `tables/per_sample_gate_decisions.csv`")
    lines.append("- `logs/selected_gate_manifest.json`")
    lines.append(f"- `{zip_path.name}`")
    lines.append("")

    (REPORT_DIR / "v222a_noharm_gate_diagnostic_report_cn.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def zip_outputs() -> Path:
    """打包并校验本轮输出。"""

    zip_path = OUT_DIR / "v222a_noharm_gate_diagnostic_pack.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT_DIR.rglob("*")):
            if path == zip_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(OUT_DIR))
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
    if bad is not None:
        raise AssertionError(f"ZIP 校验失败：{bad}")
    return zip_path


def main() -> None:
    """主流程。"""

    clean_out_dir()
    sample_manifest_path = CACHE_DIR / "sample_manifest.csv"
    if not sample_manifest_path.exists():
        raise FileNotFoundError(f"缺少 sample_manifest：{sample_manifest_path}")
    sample_manifest = pd.read_csv(sample_manifest_path, encoding="utf-8-sig")

    baseline_frames: List[pd.DataFrame] = []
    selected_frames: List[pd.DataFrame] = []
    gain_harm_frames: List[pd.DataFrame] = []
    oracle_frames: List[pd.DataFrame] = []
    tradeoff_frames: List[pd.DataFrame] = []
    locked_frames: List[pd.DataFrame] = []
    decision_frames: List[pd.DataFrame] = []
    feature_audits: List[pd.DataFrame] = []
    selected_manifest: List[Dict[str, object]] = []

    for pool_key in BASELINE_BY_POOL:
        pool = load_pool_payload(pool_key)
        feature_audits.append(audit_feature_schema(pool_key, pool["feature_names"]))
        gain_harm = build_gain_harm_table(pool, sample_manifest)
        models = fit_gate_models(pool, gain_harm)
        gate_features = apply_gate_predictions(pool, models, gain_harm)

        baseline_metric = baseline_rows(pool)
        selected_metric = selected_rows(pool)
        oracle_metric = oracle_safe_gate_rows(pool, gain_harm)
        tradeoff = build_gate_tradeoff(pool, gate_features, baseline_metric)
        chosen = tradeoff[tradeoff["validation_rank"].eq(1)].iloc[0]
        locked_metric, decisions = selected_gate_report(pool, gate_features, chosen, baseline_metric)

        baseline_frames.append(baseline_metric)
        selected_frames.append(selected_metric)
        gain_harm_frames.append(gain_harm)
        oracle_frames.append(oracle_metric)
        tradeoff_frames.append(tradeoff)
        locked_frames.append(locked_metric)
        decision_frames.append(decisions)

        selected_manifest.append(
            {
                "pool_key": pool_key,
                "baseline_name": pool["baseline_name"],
                "selected_residual_name": pool["selected_name"],
                "gate_output_name": "v222a_validation_selected_noharm_gate",
                "tau_safe": float(chosen["tau_safe"]),
                "tau_useful": float(chosen["tau_useful"]),
                "tau_tail_harm": float(chosen["tau_tail_harm"]),
                "formal_gate_pass_on_validation": bool(chosen["formal_gate_pass"]),
                "validation_rank": int(chosen["validation_rank"]),
                "selected_by": "validation_only",
                "test_used_for_selection": False,
                "fit_split": "train",
                "gate_type": "binary_noharm_gate",
            }
        )

    baseline_all = pd.concat(baseline_frames, ignore_index=True)
    selected_all = pd.concat(selected_frames, ignore_index=True)
    gain_harm_all = pd.concat(gain_harm_frames, ignore_index=True)
    oracle_all = pd.concat(oracle_frames, ignore_index=True)
    tradeoff_all = pd.concat(tradeoff_frames, ignore_index=True)
    locked_all = pd.concat(locked_frames, ignore_index=True)
    decisions_all = pd.concat(decision_frames, ignore_index=True)
    feature_audit_all = pd.concat(feature_audits, ignore_index=True)

    assert_no_forbidden_outputs(
        list(baseline_all["output_name"].astype(str))
        + list(selected_all["output_name"].astype(str))
        + list(locked_all["output_name"].astype(str))
    )

    write_csv(baseline_all, TABLE_DIR / "baseline_B_metrics.csv")
    write_csv(selected_all, TABLE_DIR / "selected_residual_M_metrics.csv")
    write_csv(gain_harm_all, TABLE_DIR / "gain_harm_decomposition.csv")
    write_csv(oracle_all, TABLE_DIR / "oracle_safe_gate_report.csv")
    write_csv(tradeoff_all, TABLE_DIR / "val_gate_tradeoff_table.csv")
    write_csv(locked_all, TABLE_DIR / "test_locked_gate_report.csv")
    write_csv(decisions_all, TABLE_DIR / "per_sample_gate_decisions.csv")
    write_csv(feature_audit_all, TABLE_DIR / "feature_schema_audit.csv")

    guard = pd.DataFrame(
        [
            {
                "check_name": "feature_schema_forbidden_tokens",
                "status": "pass" if feature_audit_all["guard_status"].eq("pass").all() else "fail",
                "detail": "gate feature_matrix 不含 split/subject/true/oracle/RMSE 等禁用字段",
            },
            {
                "check_name": "selection_uses_validation_only",
                "status": "pass",
                "detail": "tau_safe/tau_useful/tau_tail_harm 只按 validation tradeoff 表选择",
            },
            {
                "check_name": "train_only_fit",
                "status": "pass",
                "detail": "safe/useful/tail-delta 预测器只在 train split 拟合",
            },
            {
                "check_name": "test_locked_once",
                "status": "pass",
                "detail": "test 只在 selected gate 固定后写入 test_locked_gate_report",
            },
            {
                "check_name": "no_v222b_or_v223",
                "status": "pass",
                "detail": "本轮只做二元 no-harm gate 诊断，没有训练 v222b/v223",
            },
        ]
    )
    write_csv(guard, TABLE_DIR / "leakage_guard_result.csv")

    manifest = {
        "stage": "v222a_noharm_gate_diagnostic",
        "created_by": Path(__file__).name,
        "cache_dir": str(CACHE_DIR),
        "v222a_dir": str(V222A_DIR),
        "output_dir": str(OUT_DIR),
        "baseline_by_pool": BASELINE_BY_POOL,
        "sample_safe_rmse_margin": SAMPLE_SAFE_RMSE_MARGIN,
        "sample_safe_tail_margin": SAMPLE_SAFE_TAIL_MARGIN,
        "useful_tail_improve": USEFUL_TAIL_IMPROVE,
        "agg_noharm_tol": AGG_NOHARM_TOL,
        "tau_safe_grid": TAU_SAFE_GRID,
        "tau_useful_grid": TAU_USEFUL_GRID,
        "tau_tail_harm_grid": TAU_TAIL_HARM_GRID,
        "selected_gates": selected_manifest,
        "test_used_for_selection": False,
        "notes": [
            "oracle safe gate 是 diagnostic-only，使用 true per-sample metrics 决策。",
            "validation-selected no-harm gate 的推理输入只来自 feature_matrix。",
        ],
    }
    (LOG_DIR / "selected_gate_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = zip_outputs()
    make_report(baseline_all, selected_all, oracle_all, tradeoff_all, locked_all, zip_path)
    zip_path = zip_outputs()

    print("v222a no-harm gate diagnostic finished.")
    print(f"output_dir={OUT_DIR}")
    print(f"selected_gate_manifest={LOG_DIR / 'selected_gate_manifest.json'}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
