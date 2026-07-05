#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v222a 轻量软融合与受限残差校准。

输入来自 `v222a_candidate_curve_cache_20260622`，候选曲线池保持固定。
本脚本只做两类 deployable calibration：

1. formal 候选曲线的全局非负凸融合；
2. 以 formal 候选为 base 的 Ridge 残差校准，并对每个时间点的残差做硬边界裁剪。

训练纪律：
- 模型只在 train split 拟合；
- alpha、残差边界、base 候选和最终输出只按 validation 指标选择；
- test split 只在最终 validation-selected 配置固定后报告；
- split、subject、event_uid、true/oracle/RMSE/severe-under 等字段不进入推理特征；
- `W3_B4_original_soft` 不进入 formal 候选、选择表、gate 或 usage 表。
"""

from __future__ import annotations

import json
import pickle
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[3]
REBUILD_ROOT = REPO_ROOT / "05_rebuild_from_raw_20260511"
BASE_DIR = REBUILD_ROOT / "03_baselines"
CACHE_DIR = BASE_DIR / "v222a_candidate_curve_cache_20260622"
OUT_DIR = BASE_DIR / "v222a_light_fusion_residual_20260622"
TABLE_DIR = OUT_DIR / "tables"
REPORT_DIR = OUT_DIR / "reports"
MODEL_DIR = OUT_DIR / "models"
LOG_DIR = OUT_DIR / "logs"


FORMAL_CANDIDATES = [
    "steering_only",
    "joint_equal",
    "joint_steer_focus",
    "avg_joint_focus",
    "global_blend",
    "peak_floor_090",
    "ridge_residual_joint",
    "ridge_residual_peakfloor",
]

RESIDUAL_BASES = [
    "global_blend",
    "avg_joint_focus",
    "peak_floor_090",
    "ridge_residual_joint",
    "ridge_residual_peakfloor",
]

RIDGE_ALPHAS = [1.0, 10.0, 100.0]
RESIDUAL_BOUNDS = [0.05, 0.10, 0.20]

FORBIDDEN_FORMAL_SUBSTRINGS = [
    "W3_B4_original_soft",
    "oracle",
    "true_label",
    "fallback",
]

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

# selection_score 只在 validation split 上计算。权重提前固定，不根据 test 调整。
SELECTION_TAIL_WEIGHT = 0.05
SELECTION_UNDER_WEIGHT = 0.10


def ensure_dirs() -> None:
    """创建输出目录。"""

    for path in [TABLE_DIR, REPORT_DIR, MODEL_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """清理旧输出，避免旧模型和旧表混入本轮结果。"""

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一用 utf-8-sig 导出 CSV，方便中文环境打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_float_token(value: float) -> str:
    """把浮点超参转换为稳定文件名片段。"""

    return str(value).replace(".", "p")


def assert_no_forbidden_name(names: Iterable[str], context: str) -> None:
    """候选名和输出名不能混入禁用身份。"""

    bad: List[str] = []
    for name in names:
        lowered = str(name).lower()
        for token in FORBIDDEN_FORMAL_SUBSTRINGS:
            if token.lower() in lowered:
                bad.append(str(name))
    if bad:
        raise AssertionError(f"{context} 包含禁用名称：" + ", ".join(sorted(set(bad))))


def forbidden_feature_token(name: str) -> str:
    """检查特征名是否命中禁用 token。"""

    lowered = name.lower()
    for token in FORBIDDEN_FEATURE_TOKENS:
        if token in lowered:
            return token
    return ""


def assert_feature_schema(feature_names: List[str], pool_key: str) -> pd.DataFrame:
    """确保 feature_matrix 不含身份字段、split 字段或目标派生字段。"""

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
        raise AssertionError("v222a feature schema 命中禁用字段：\n" + bad_rows.to_string(index=False))
    return out


def assert_finite(name: str, arr: np.ndarray) -> None:
    """所有训练输入和预测输出必须为有限值。"""

    if not np.isfinite(arr).all():
        bad = int(np.size(arr) - np.isfinite(arr).sum())
        raise AssertionError(f"{name} 包含非有限值：bad={bad}")


def load_pool_cache(path: Path) -> Dict[str, object]:
    """读取单个 pool 的候选曲线缓存。"""

    pool_key = path.stem.replace("candidate_predictions_", "", 1)
    with np.load(path, allow_pickle=False) as data:
        candidate_names = data["candidate_names"].astype(str).tolist()
        feature_names = data["feature_names"].astype(str).tolist()
        payload: Dict[str, object] = {
            "pool_key": pool_key,
            "X_hist": data["X_hist"].astype(np.float32),
            "Y_future": data["Y_future"].astype(np.float32),
            "true_steer": data["true_steer"].astype(np.float32),
            "road_future": data["road_future"].astype(np.float32),
            "candidate_names": candidate_names,
            "predictions": data["predictions"].astype(np.float32),
            "feature_matrix": data["feature_matrix"].astype(np.float32),
            "feature_names": feature_names,
            "split": data["split"].astype(str),
            "array_index": data["array_index"].astype(np.int64),
            "event_uid": data["event_uid"].astype(str),
        }

    preds = payload["predictions"]
    true_steer = payload["true_steer"]
    feature_matrix = payload["feature_matrix"]
    if not isinstance(preds, np.ndarray) or not isinstance(true_steer, np.ndarray):
        raise AssertionError(f"{pool_key} 缓存结构异常")
    if preds.shape[0] != true_steer.shape[0] or preds.shape[2] != true_steer.shape[1]:
        raise AssertionError(f"{pool_key} predictions 与 true_steer shape 不一致")
    if not isinstance(feature_matrix, np.ndarray) or feature_matrix.shape[0] != true_steer.shape[0]:
        raise AssertionError(f"{pool_key} feature_matrix 行数不一致")

    missing = [name for name in FORMAL_CANDIDATES if name not in candidate_names]
    if missing:
        raise AssertionError(f"{pool_key} 缺少 formal 候选：" + ", ".join(missing))
    assert_no_forbidden_name(FORMAL_CANDIDATES, f"{pool_key} formal candidate list")
    assert_feature_schema(feature_names, pool_key)
    assert_finite(f"{pool_key}:feature_matrix", feature_matrix)
    assert_finite(f"{pool_key}:predictions", preds)
    assert_finite(f"{pool_key}:true_steer", true_steer)
    return payload


def candidate_prediction_map(cache: Dict[str, object]) -> Dict[str, np.ndarray]:
    """把缓存中的三维 prediction stack 拆成 name -> (N,T) 字典。"""

    names = list(cache["candidate_names"])
    predictions = cache["predictions"]
    assert isinstance(predictions, np.ndarray)
    return {name: predictions[:, idx, :] for idx, name in enumerate(names)}


def peak_values(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """返回每条曲线的绝对峰值和峰值符号值。"""

    idx = np.nanargmax(np.abs(arr), axis=1)
    signed = arr[np.arange(arr.shape[0]), idx]
    return np.abs(signed), signed


def split_mask(split_values: np.ndarray, split_name: str) -> np.ndarray:
    """生成 train/val/test/all 掩码。"""

    if split_name == "all":
        return np.ones(len(split_values), dtype=bool)
    return split_values.astype(str) == split_name


def metric_for_prediction(
    pool_key: str,
    pool_name: str,
    output_name: str,
    variant_type: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, object]:
    """计算 steering 曲线指标。"""

    diff = y_pred - y_true
    sample_rmse = np.sqrt(np.mean(np.square(diff), axis=1))
    true_peak_abs, true_peak_signed = peak_values(y_true)
    pred_peak_abs, pred_peak_signed = peak_values(y_pred)
    direction_ok = np.sign(true_peak_signed) == np.sign(pred_peak_signed)
    severe_under = pred_peak_abs < 0.5 * true_peak_abs
    strong = true_peak_abs >= 1.0
    tail = np.arange(y_true.shape[1]) >= 10

    strong_rmse = np.nan
    strong_under = np.nan
    if strong.any():
        strong_rmse = float(np.sqrt(np.mean(np.square(diff[strong]))))
        strong_under = float(np.mean(severe_under[strong]))

    return {
        "pool_key": pool_key,
        "pool_name": pool_name,
        "output_name": output_name,
        "variant_type": variant_type,
        "split": split_name,
        "n": int(y_true.shape[0]),
        "steer_rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "steer_tail_rmse_1to2s": float(np.sqrt(np.mean(np.square(diff[:, tail])))),
        "steer_sample_rmse_mean": float(np.mean(sample_rmse)),
        "steer_sample_rmse_p90": float(np.quantile(sample_rmse, 0.90)),
        "steer_direction_acc": float(np.mean(direction_ok)),
        "steer_severe_under_rate": float(np.mean(severe_under)),
        "strong_response_n": int(strong.sum()),
        "strong_response_rmse": strong_rmse,
        "strong_response_severe_under_rate": strong_under,
        "true_peak_abs_mean": float(np.mean(true_peak_abs)),
        "pred_peak_abs_mean": float(np.mean(pred_peak_abs)),
    }


def metrics_for_splits(
    pool_key: str,
    pool_name: str,
    split_values: np.ndarray,
    true_steer: np.ndarray,
    predictions: Dict[str, np.ndarray],
    variant_types: Dict[str, str],
    splits: Iterable[str],
) -> pd.DataFrame:
    """只对指定 split 计算指标，避免在选择前批量评估 test。"""

    rows: List[Dict[str, object]] = []
    for name, pred in predictions.items():
        for split_name in splits:
            mask = split_mask(split_values, split_name)
            if not mask.any():
                continue
            rows.append(
                metric_for_prediction(
                    pool_key,
                    pool_name,
                    name,
                    variant_types.get(name, "unknown"),
                    split_name,
                    true_steer[mask],
                    pred[mask],
                )
            )
    return pd.DataFrame(rows)


def fit_convex_formal_blend(
    cache: Dict[str, object],
    pred_map: Dict[str, np.ndarray],
    train_mask: np.ndarray,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, object]]:
    """在 train 上拟合 formal 候选的非负全局凸融合权重。"""

    y_train = cache["true_steer"][train_mask]
    formal_stack = np.stack([pred_map[name] for name in FORMAL_CANDIDATES], axis=1)
    train_features = formal_stack[train_mask].transpose(0, 2, 1).reshape(-1, len(FORMAL_CANDIDATES))
    train_target = y_train.reshape(-1)

    # 这里不用 sklearn 的 positive LinearRegression，因为该路径在本机真实矩阵上
    # 触发过底层无 traceback 退出。8 维凸权重用投影梯度即可稳定求解。
    weights = solve_simplex_least_squares(train_features, train_target)

    pred = np.einsum("nct,c->nt", formal_stack, weights).astype(np.float32)
    weight_rows = [
        {
            "component_candidate": name,
            "weight": float(weight),
        }
        for name, weight in zip(FORMAL_CANDIDATES, weights)
    ]
    payload = {
        "model_kind": "convex_formal_blend",
        "formal_candidates": FORMAL_CANDIDATES,
        "weights": weights,
        "fit_split": "train",
        "selected_by": "validation_only",
        "test_used_for_selection": False,
    }
    return pred, pd.DataFrame(weight_rows), payload


def project_to_simplex(vec: np.ndarray) -> np.ndarray:
    """把向量投影到 sum(w)=1, w>=0 的概率单纯形。"""

    values = np.asarray(vec, dtype=np.float64)
    if values.ndim != 1:
        raise AssertionError("simplex projection 只接受一维向量")
    order = np.sort(values)[::-1]
    cssv = np.cumsum(order) - 1.0
    idx = np.arange(1, len(values) + 1)
    cond = order - cssv / idx > 0
    if not cond.any():
        return np.ones_like(values) / len(values)
    rho = idx[cond][-1]
    theta = cssv[cond][-1] / rho
    projected = np.maximum(values - theta, 0.0)
    total = projected.sum()
    if total <= 0:
        return np.ones_like(values) / len(values)
    return projected / total


def solve_simplex_least_squares(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """用投影梯度求解 min ||Xw-y||^2, s.t. w 在概率单纯形上。"""

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    n_rows, n_cols = X.shape
    # 避免调用本机 BLAS 的大矩阵-向量路径；候选维度很小，逐列求和更稳。
    gram = np.zeros((n_cols, n_cols), dtype=np.float64)
    rhs = np.zeros(n_cols, dtype=np.float64)
    for col in range(n_cols):
        x_col = X[:, col]
        rhs[col] = float(np.mean(x_col * y))
        for other in range(col, n_cols):
            value = float(np.mean(x_col * X[:, other]))
            gram[col, other] = value
            gram[other, col] = value
    eig_max = float(np.linalg.eigvalsh(gram).max())
    step = 1.0 / max(eig_max, 1e-9)
    weights = np.ones(n_cols, dtype=np.float64) / n_cols
    for _ in range(2000):
        grad = gram @ weights - rhs
        next_weights = project_to_simplex(weights - step * grad)
        if float(np.max(np.abs(next_weights - weights))) < 1e-12:
            weights = next_weights
            break
        weights = next_weights
    return weights


def fit_bounded_residual_variants(
    cache: Dict[str, object],
    pred_map: Dict[str, np.ndarray],
    train_mask: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    """训练受限 residual 校准变体；所有模型只看 train。"""

    X = cache["feature_matrix"]
    y = cache["true_steer"]
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_all = scaler.transform(X)

    variant_preds: Dict[str, np.ndarray] = {}
    manifest_rows: List[Dict[str, object]] = []
    payloads: Dict[str, Dict[str, object]] = {}

    for base_name in RESIDUAL_BASES:
        if base_name not in pred_map:
            raise AssertionError(f"缺少 residual base：{base_name}")
        base_pred = pred_map[base_name]
        residual_target = y[train_mask] - base_pred[train_mask]
        for alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=alpha)
            model.fit(X_train, residual_target)
            raw_delta = np.asarray(model.predict(X_all), dtype=np.float32)
            for bound in RESIDUAL_BOUNDS:
                output_name = (
                    f"v222a_bounded_residual_{base_name}_"
                    f"a{safe_float_token(alpha)}_b{safe_float_token(bound)}"
                )
                clipped_delta = np.clip(raw_delta, -bound, bound)
                pred = (base_pred + clipped_delta).astype(np.float32)
                assert_finite(output_name, pred)
                variant_preds[output_name] = pred
                manifest_rows.append(
                    {
                        "output_name": output_name,
                        "variant_type": "bounded_residual",
                        "base_candidate": base_name,
                        "ridge_alpha": alpha,
                        "residual_bound_rad": bound,
                        "fit_split": "train",
                        "selected_by": "validation_only",
                        "test_used_for_selection": False,
                        "feature_count": int(X.shape[1]),
                    }
                )
                payloads[output_name] = {
                    "model_kind": "bounded_residual",
                    "base_candidate": base_name,
                    "ridge_alpha": alpha,
                    "residual_bound_rad": bound,
                    "scaler": scaler,
                    "model": model,
                    "feature_names": list(cache["feature_names"]),
                    "fit_split": "train",
                    "selected_by": "validation_only",
                    "test_used_for_selection": False,
                }
    return variant_preds, manifest_rows, payloads


def add_selection_score(validation_metrics: pd.DataFrame) -> pd.DataFrame:
    """给 validation 指标加固定选择分数。"""

    out = validation_metrics.copy()
    strong_under = out["strong_response_severe_under_rate"].fillna(out["steer_severe_under_rate"])
    out["selection_score"] = (
        out["steer_rmse"]
        + SELECTION_TAIL_WEIGHT * out["steer_tail_rmse_1to2s"]
        + SELECTION_UNDER_WEIGHT * strong_under
    )
    return out


def select_by_validation(validation_metrics: pd.DataFrame) -> pd.DataFrame:
    """每个 pool 只按 validation split 选择最终输出。"""

    val = validation_metrics[validation_metrics["split"].eq("val")].copy()
    val = add_selection_score(val)
    val = val.sort_values(
        ["pool_key", "selection_score", "steer_rmse", "strong_response_severe_under_rate", "output_name"],
        ascending=[True, True, True, True, True],
    )
    val["validation_rank"] = val.groupby("pool_key").cumcount() + 1
    selected = val[val["validation_rank"].eq(1)].copy()
    if selected["pool_key"].nunique() != val["pool_key"].nunique():
        raise AssertionError("validation selection 没有覆盖所有 pool")
    return val


def selected_per_sample_metrics(
    pool_key: str,
    sample_manifest: pd.DataFrame,
    split_values: np.ndarray,
    output_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """仅为 validation-selected 输出导出逐样本评估，便于后续查错。"""

    diff = y_pred - y_true
    sample_rmse = np.sqrt(np.mean(np.square(diff), axis=1))
    tail = np.arange(y_true.shape[1]) >= 10
    sample_tail_rmse = np.sqrt(np.mean(np.square(diff[:, tail]), axis=1))
    true_peak_abs, true_peak_signed = peak_values(y_true)
    pred_peak_abs, pred_peak_signed = peak_values(y_pred)
    out = sample_manifest[sample_manifest["pool_key"].eq(pool_key)].copy().reset_index(drop=True)
    out["output_name"] = output_name
    out["split"] = split_values.astype(str)
    out["steer_sample_rmse"] = sample_rmse
    out["steer_tail_rmse_1to2s"] = sample_tail_rmse
    out["true_steer_peak_abs"] = true_peak_abs
    out["pred_steer_peak_abs"] = pred_peak_abs
    out["steer_direction_ok"] = np.sign(true_peak_signed) == np.sign(pred_peak_signed)
    out["steer_severe_under"] = pred_peak_abs < 0.5 * true_peak_abs
    return out


def save_selected_model(pool_key: str, output_name: str, payload: Dict[str, object]) -> Path:
    """保存 validation-selected 配置，供后续复现实验读取。"""

    path = MODEL_DIR / f"v222a_{pool_key}_selected.pkl"
    save_payload = dict(payload)
    save_payload["pool_key"] = pool_key
    save_payload["output_name"] = output_name
    save_payload["selected_by"] = "validation_only"
    save_payload["test_used_for_selection"] = False
    with path.open("wb") as f:
        pickle.dump(save_payload, f)
    return path


def zip_outputs() -> Path:
    """打包输出并校验 ZIP 完整性。"""

    zip_path = OUT_DIR / "v222a_light_fusion_residual_pack.zip"
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


def make_report(
    selected_metrics: pd.DataFrame,
    validation_selection: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    model_manifest: pd.DataFrame,
    zip_path: Path,
) -> None:
    """生成中文报告。"""

    selected_val = validation_selection[validation_selection["validation_rank"].eq(1)].copy()
    selected_test = selected_metrics[selected_metrics["split"].eq("test")].copy()
    baseline_test = baseline_metrics[baseline_metrics["split"].eq("test")].copy()
    best_baseline = (
        baseline_test.sort_values(["pool_key", "steer_rmse"], ascending=[True, True])
        .groupby("pool_key")
        .head(1)
    )

    lines: List[str] = []
    lines.append("# v222a 轻量软融合与受限残差报告")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("- 已在固定 formal 候选池上完成非负凸融合和 bounded residual 校准。")
    lines.append("- 所有模型只在 train 拟合，最终输出只按 validation selection score 选择。")
    lines.append("- test 指标仅对 validation-selected 输出和固定 baseline 参考项报告。")
    lines.append("- 未使用 `W3_B4_original_soft`、oracle、true-label fallback 或 diagnostic-only row。")
    lines.append("")
    lines.append("## Validation-selected 输出")
    lines.append("")
    for row in selected_val.itertuples(index=False):
        test_row = selected_test[selected_test["pool_key"].eq(row.pool_key)].iloc[0]
        lines.append(
            f"- {row.pool_key}: `{row.output_name}`，variant={row.variant_type}，"
            f"val_score={row.selection_score:.6f}，val_RMSE={row.steer_rmse:.6f}，"
            f"test_RMSE={test_row.steer_rmse:.6f}，test_tail={test_row.steer_tail_rmse_1to2s:.6f}，"
            f"test_under={test_row.steer_severe_under_rate:.6f}"
        )
    lines.append("")
    lines.append("## 固定 baseline 对照")
    lines.append("")
    for row in best_baseline.itertuples(index=False):
        lines.append(
            f"- {row.pool_key}: best fixed baseline on test = `{row.output_name}`，"
            f"RMSE={row.steer_rmse:.6f}，tail={row.steer_tail_rmse_1to2s:.6f}，"
            f"under={row.steer_severe_under_rate:.6f}"
        )
    lines.append("")
    lines.append("## 选择纪律")
    lines.append("")
    lines.append(f"- validation 参与排序的输出数：{len(validation_selection)}")
    lines.append(f"- 模型 manifest 行数：{len(model_manifest)}")
    lines.append(f"- selection score = RMSE + {SELECTION_TAIL_WEIGHT} * tail_RMSE + {SELECTION_UNDER_WEIGHT} * strong_under_rate")
    lines.append(f"- ZIP：`{zip_path.name}`")
    lines.append("")

    (REPORT_DIR / "v222a_light_fusion_residual_report_cn.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """主流程：读取 cache、训练轻量校准、validation 选择、最终 test 报告。"""

    clean_out_dir()
    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"缺少 v222a cache 目录：{CACHE_DIR}")

    sample_manifest_path = CACHE_DIR / "sample_manifest.csv"
    if not sample_manifest_path.exists():
        raise FileNotFoundError(f"缺少 sample_manifest：{sample_manifest_path}")
    sample_manifest = pd.read_csv(sample_manifest_path, encoding="utf-8-sig")

    all_validation_metrics: List[pd.DataFrame] = []
    all_baseline_metrics: List[pd.DataFrame] = []
    all_selected_metrics: List[pd.DataFrame] = []
    all_per_sample: List[pd.DataFrame] = []
    all_weight_rows: List[pd.DataFrame] = []
    all_model_rows: List[Dict[str, object]] = []
    all_feature_audits: List[pd.DataFrame] = []
    selected_model_paths: List[Dict[str, object]] = []

    for cache_path in sorted(CACHE_DIR.glob("candidate_predictions_*.npz")):
        cache = load_pool_cache(cache_path)
        pool_key = str(cache["pool_key"])
        pool_rows = sample_manifest[sample_manifest["pool_key"].eq(pool_key)]
        if pool_rows.empty:
            raise AssertionError(f"sample_manifest 缺少 pool：{pool_key}")
        pool_name = str(pool_rows["pool_name"].iloc[0]) if "pool_name" in pool_rows.columns else pool_key
        split_values = cache["split"]
        true_steer = cache["true_steer"]
        assert isinstance(split_values, np.ndarray) and isinstance(true_steer, np.ndarray)
        train_mask = split_values.astype(str) == "train"
        val_mask = split_values.astype(str) == "val"
        if not train_mask.any() or not val_mask.any():
            raise AssertionError(f"{pool_key} 缺少 train 或 val split")

        feature_audit = assert_feature_schema(list(cache["feature_names"]), pool_key)
        all_feature_audits.append(feature_audit)

        pred_map = candidate_prediction_map(cache)
        baseline_preds = {name: pred_map[name] for name in FORMAL_CANDIDATES}
        baseline_variant_types = {name: "fixed_formal_baseline" for name in baseline_preds}

        convex_pred, convex_weights, convex_payload = fit_convex_formal_blend(cache, pred_map, train_mask)
        convex_name = "v222a_convex_formal_blend"
        convex_weights.insert(0, "pool_key", pool_key)
        convex_weights.insert(1, "output_name", convex_name)
        all_weight_rows.append(convex_weights)

        residual_preds, residual_rows, residual_payloads = fit_bounded_residual_variants(cache, pred_map, train_mask)

        learned_preds: Dict[str, np.ndarray] = {convex_name: convex_pred}
        learned_preds.update(residual_preds)
        learned_variant_types: Dict[str, str] = {convex_name: "convex_formal_blend"}
        learned_variant_types.update({name: "bounded_residual" for name in residual_preds})

        selection_preds: Dict[str, np.ndarray] = {}
        selection_preds.update(baseline_preds)
        selection_preds.update(learned_preds)
        selection_variant_types: Dict[str, str] = {}
        selection_variant_types.update(baseline_variant_types)
        selection_variant_types.update(learned_variant_types)

        # 选择前只计算 train/val，避免对所有超参变体评估 test。
        validation_metrics = metrics_for_splits(
            pool_key,
            pool_name,
            split_values,
            true_steer,
            selection_preds,
            selection_variant_types,
            splits=["train", "val"],
        )
        all_validation_metrics.append(validation_metrics)

        val_ranked = select_by_validation(validation_metrics)
        selected = val_ranked[val_ranked["validation_rank"].eq(1)].iloc[0]
        selected_name = str(selected["output_name"])
        selected_pred = selection_preds[selected_name]
        selected_variant_type = selection_variant_types[selected_name]

        # 固定 baseline 参考项可以报告 test；learned 超参变体只报告最终 selected。
        baseline_metrics = metrics_for_splits(
            pool_key,
            pool_name,
            split_values,
            true_steer,
            baseline_preds,
            baseline_variant_types,
            splits=["train", "val", "test"],
        )
        all_baseline_metrics.append(baseline_metrics)

        selected_metrics = metrics_for_splits(
            pool_key,
            pool_name,
            split_values,
            true_steer,
            {selected_name: selected_pred},
            {selected_name: selected_variant_type},
            splits=["train", "val", "test"],
        )
        selected_metrics["selected_by"] = "validation_only"
        selected_metrics["test_used_for_selection"] = False
        all_selected_metrics.append(selected_metrics)

        per_sample = selected_per_sample_metrics(
            pool_key,
            sample_manifest,
            split_values,
            selected_name,
            true_steer,
            selected_pred,
        )
        all_per_sample.append(per_sample)

        selected_cache_path = OUT_DIR / f"v222a_selected_predictions_{pool_key}.npz"
        np.savez_compressed(
            selected_cache_path,
            pred_v222a_val_selected=selected_pred.astype(np.float32),
            selected_output_name=np.array([selected_name], dtype="U160"),
            selected_variant_type=np.array([selected_variant_type], dtype="U80"),
            true_steer=true_steer.astype(np.float32),
            split=split_values.astype("U16"),
            array_index=cache["array_index"],
            event_uid=cache["event_uid"],
        )

        payload: Dict[str, object]
        if selected_name == convex_name:
            payload = convex_payload
        elif selected_name in residual_payloads:
            payload = residual_payloads[selected_name]
        else:
            payload = {
                "model_kind": "fixed_formal_baseline",
                "base_candidate": selected_name,
                "fit_split": "none",
                "selected_by": "validation_only",
                "test_used_for_selection": False,
            }
        model_path = save_selected_model(pool_key, selected_name, payload)
        selected_model_paths.append(
            {
                "pool_key": pool_key,
                "output_name": selected_name,
                "model_path": str(model_path.relative_to(REPO_ROOT)),
            }
        )

        all_model_rows.append(
            {
                "pool_key": pool_key,
                "pool_name": pool_name,
                "output_name": convex_name,
                "variant_type": "convex_formal_blend",
                "base_candidate": "",
                "ridge_alpha": np.nan,
                "residual_bound_rad": np.nan,
                "fit_split": "train",
                "selected_by": "validation_only",
                "test_used_for_selection": False,
                "feature_count": 0,
                "component_count": len(FORMAL_CANDIDATES),
            }
        )
        for row in residual_rows:
            row = dict(row)
            row["pool_key"] = pool_key
            row["pool_name"] = pool_name
            row["component_count"] = 0
            all_model_rows.append(row)

    validation_metrics_all = pd.concat(all_validation_metrics, ignore_index=True)
    validation_ranked_all = select_by_validation(validation_metrics_all)
    baseline_metrics_all = pd.concat(all_baseline_metrics, ignore_index=True)
    selected_metrics_all = pd.concat(all_selected_metrics, ignore_index=True)
    per_sample_all = pd.concat(all_per_sample, ignore_index=True)
    feature_audit_all = pd.concat(all_feature_audits, ignore_index=True)
    blend_weights_all = pd.concat(all_weight_rows, ignore_index=True)
    model_manifest = pd.DataFrame(all_model_rows)
    selected_models = pd.DataFrame(selected_model_paths)

    assert_no_forbidden_name(validation_ranked_all["output_name"].astype(str), "validation outputs")
    assert_no_forbidden_name(selected_metrics_all["output_name"].astype(str), "selected outputs")

    write_csv(validation_ranked_all, TABLE_DIR / "v222a_validation_selection.csv")
    write_csv(baseline_metrics_all, TABLE_DIR / "v222a_reference_baseline_metrics.csv")
    write_csv(selected_metrics_all, TABLE_DIR / "v222a_selected_metrics.csv")
    write_csv(per_sample_all, TABLE_DIR / "v222a_selected_per_sample_metrics.csv")
    write_csv(blend_weights_all, TABLE_DIR / "v222a_convex_blend_weights.csv")
    write_csv(model_manifest, TABLE_DIR / "v222a_model_manifest.csv")
    write_csv(selected_models, TABLE_DIR / "v222a_selected_model_paths.csv")
    write_csv(feature_audit_all, TABLE_DIR / "v222a_feature_schema_audit.csv")

    leakage_guard = pd.DataFrame(
        [
            {
                "check_name": "feature_schema_forbidden_tokens",
                "status": "pass" if feature_audit_all["guard_status"].eq("pass").all() else "fail",
                "detail": "feature_matrix 不含 split/subject/true/oracle/RMSE 等禁用字段",
            },
            {
                "check_name": "formal_candidate_forbidden_names",
                "status": "pass",
                "detail": "输出名不含 W3_B4_original_soft/oracle/fallback/true_label",
            },
            {
                "check_name": "selection_uses_validation_only",
                "status": "pass",
                "detail": "validation_selection 表只含 val 排序行；test 仅在 selected_metrics 与固定 baseline 参考中报告",
            },
            {
                "check_name": "train_only_fit",
                "status": "pass",
                "detail": "凸融合和 residual 校准均只在 train split 拟合",
            },
        ]
    )
    write_csv(leakage_guard, TABLE_DIR / "v222a_leakage_guard_result.csv")

    manifest = {
        "stage": "v222a_light_fusion_residual",
        "created_by": Path(__file__).name,
        "cache_dir": str(CACHE_DIR),
        "output_dir": str(OUT_DIR),
        "formal_candidates": FORMAL_CANDIDATES,
        "residual_bases": RESIDUAL_BASES,
        "ridge_alphas": RIDGE_ALPHAS,
        "residual_bounds": RESIDUAL_BOUNDS,
        "selection_score": {
            "formula": "steer_rmse + tail_weight * steer_tail_rmse_1to2s + under_weight * strong_response_severe_under_rate",
            "tail_weight": SELECTION_TAIL_WEIGHT,
            "under_weight": SELECTION_UNDER_WEIGHT,
            "split": "val",
        },
        "test_used_for_selection": False,
        "selected_models": selected_model_paths,
    }
    (LOG_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = zip_outputs()
    make_report(selected_metrics_all, validation_ranked_all, baseline_metrics_all, model_manifest, zip_path)
    zip_path = zip_outputs()

    print("v222a light fusion residual finished.")
    print(f"output_dir={OUT_DIR}")
    print(f"selected_metrics={TABLE_DIR / 'v222a_selected_metrics.csv'}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
