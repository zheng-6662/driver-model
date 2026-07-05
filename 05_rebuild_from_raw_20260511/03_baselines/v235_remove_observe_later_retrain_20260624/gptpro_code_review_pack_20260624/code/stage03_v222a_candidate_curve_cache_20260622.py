#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v222a 候选曲线缓存导出与泄漏审计。

这个脚本只做一件事：把 v216/v217/v218/v219 已经形成的固定候选曲线，
重新导出成 v222a 可以直接读取的数组缓存。它不训练新的神经网络，不修改
候选池，也不根据 test 结果选择阈值、alpha 或模型族。

输出重点：
1. 每个 pool 一个 `candidate_predictions_{pool_key}.npz`，包含历史输入、
   未来标签、道路未来量和所有候选曲线；
2. `candidate_manifest.csv` 记录每条候选曲线的来源、formal/diagnostic 范围
   和 validation-only 元数据；
3. `feature_schema_audit.csv` 与 `leakage_guard_result.csv` 明确证明 v219 ridge
   residual 使用的推理特征不含 split、subject、true、oracle、RMSE 等禁用字段；
4. `candidate_curve_metrics.csv` 与 `metric_crosscheck_vs_v219.csv` 用于确认本次
   重建曲线与 v219 既有指标一致。

注意：`true_steer` 和 `Y_future` 会保存在缓存里，它们是训练/评估标签，不是
v222a 的推理输入特征；后续训练脚本必须只从 `feature_matrix` 或候选预测曲线
构造 deployable features。
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import json
import pickle
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
REBUILD_ROOT = REPO_ROOT / "05_rebuild_from_raw_20260511"
BASE_DIR = REBUILD_ROOT / "03_baselines"
SCRIPT_DIR = BASE_DIR / "scripts"
PYCACHE_DIR = SCRIPT_DIR / "__pycache__"
V219_DIR = BASE_DIR / "v219_ridge_residual_stack_20260620"
OUT_DIR = BASE_DIR / "v222a_candidate_curve_cache_20260622"
TABLE_DIR = OUT_DIR / "tables"
REPORT_DIR = OUT_DIR / "reports"
LOG_DIR = OUT_DIR / "logs"


FORMAL_CANDIDATES = {
    "steering_only",
    "joint_equal",
    "joint_steer_focus",
    "avg_joint_focus",
    "global_blend",
    "peak_floor_090",
    "ridge_residual_joint",
    "ridge_residual_peakfloor",
}

DIAGNOSTIC_CANDIDATES = {
    "zero_change",
    "peak_floor_075",
    "peak_floor_100",
    "ridge_abs",
    "ridge_residual_global",
    "ridge_residual_global_weighted",
}

FORBIDDEN_FORMAL_SUBSTRINGS = [
    "W3_B4_original_soft",
    "oracle",
    "true_label",
    "fallback",
]

# 这些 token 只用于检查推理特征名。`true_steer` 可以作为标签存在于 NPZ，
# 但不允许出现在 feature schema 中。
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

# 从历史权重和 pyc 重新生成候选曲线时，少数模型会出现 1e-3 量级的浮点差异。
# 这里的交叉检查目标是防止样本顺序错位或候选身份错位，因此使用显式工程阈值。
CROSSCHECK_ABS_TOL = 2e-3


@dataclass(frozen=True)
class StackVariant:
    """兼容 v219 pickle 中以 `__main__.StackVariant` 保存的对象。"""

    name: str
    cn_name: str
    target_mode: str
    base_curve_name: str
    use_sample_weight: bool
    note: str


class ExactPycacheFinder(importlib.abc.MetaPathFinder):
    """只按完整模块名加载历史 pyc，避免宽匹配误伤第三方包内部模块。"""

    def find_spec(self, fullname: str, path: object = None, target: object = None):
        module_name = fullname.rsplit(".", 1)[-1]
        for pattern in [f"{module_name}.cpython-310.pyc", f"{module_name}.pyc"]:
            hits = sorted(PYCACHE_DIR.glob(pattern))
            if not hits:
                continue
            loader = importlib.machinery.SourcelessFileLoader(fullname, str(hits[0]))
            return importlib.util.spec_from_loader(fullname, loader)
        return None


def install_pyc_finder() -> None:
    """注册 pyc 精确加载器，保证历史脚本源码缺失时仍能复现曲线。"""

    if not PYCACHE_DIR.exists():
        raise FileNotFoundError(f"缺少历史 pyc 目录：{PYCACHE_DIR}")
    if not any(isinstance(finder, ExactPycacheFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, ExactPycacheFinder())


def ensure_dirs() -> None:
    """创建本次输出目录。"""

    for path in [TABLE_DIR, REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """清理旧输出，避免上一次运行残留文件被误认为本次结果。"""

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一使用 utf-8-sig，方便 Excel 与中文报告直接打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_key(name: str) -> str:
    """把候选名转换为 NPZ 中稳定的数组 key。"""

    return "pred_" + "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def assert_finite(name: str, arr: np.ndarray) -> None:
    """所有导出的数值数组必须是有限值。"""

    if not np.isfinite(arr).all():
        bad = int(np.size(arr) - np.isfinite(arr).sum())
        raise AssertionError(f"{name} 包含非有限值：bad={bad}")


def candidate_scope(name: str) -> str:
    """区分 formal 与 diagnostic 候选。"""

    lowered = name.lower()
    for token in FORBIDDEN_FORMAL_SUBSTRINGS:
        if token.lower() in lowered:
            return "excluded"
    if name in FORMAL_CANDIDATES:
        return "formal"
    if name in DIAGNOSTIC_CANDIDATES:
        return "diagnostic"
    return "diagnostic"


def assert_no_forbidden_formal(candidates: Iterable[str]) -> None:
    """formal 候选不能含有 original/oracle/fallback/true-label 等禁用身份。"""

    bad: List[str] = []
    for name in candidates:
        if candidate_scope(name) != "formal":
            continue
        lowered = name.lower()
        for token in FORBIDDEN_FORMAL_SUBSTRINGS:
            if token.lower() in lowered:
                bad.append(name)
    if bad:
        raise AssertionError("formal 候选包含禁用名称：" + ", ".join(sorted(set(bad))))


def feature_forbidden_token(feature_name: str) -> str:
    """返回命中的禁用特征 token；未命中则返回空字符串。"""

    lowered = feature_name.lower()
    for token in FORBIDDEN_FEATURE_TOKENS:
        if token in lowered:
            return token
    return ""


def audit_feature_schema(pool_key: str, feature_names: List[str]) -> pd.DataFrame:
    """审计 v219 ridge residual 使用的 deployable feature schema。"""

    rows: List[Dict[str, object]] = []
    for order, name in enumerate(feature_names):
        bad_token = feature_forbidden_token(name)
        rows.append(
            {
                "pool_key": pool_key,
                "feature_order": order,
                "feature_name": name,
                "forbidden_token": bad_token,
                "guard_status": "fail" if bad_token else "pass",
                "allowed_reason": "候选曲线/历史输入/未来道路摘要，不含样本身份或目标派生指标"
                if not bad_token
                else "命中禁用 token",
            }
        )
    df = pd.DataFrame(rows)
    bad = df[df["guard_status"].eq("fail")]
    if not bad.empty:
        raise AssertionError("v222a feature schema 命中禁用字段：\n" + bad.to_string(index=False))
    return df


def validate_model_payload(path: Path, payload: Dict[str, Any], feature_names: List[str]) -> StackVariant:
    """校验 v219 pickle 的选择纪律和特征顺序。"""

    variant = payload.get("variant")
    if not isinstance(variant, StackVariant):
        # pickle 反序列化后通常已经是当前 __main__.StackVariant；这里保守检查字段。
        required = ["name", "cn_name", "target_mode", "base_curve_name", "use_sample_weight", "note"]
        if not all(hasattr(variant, key) for key in required):
            raise AssertionError(f"{path.name} 的 variant 结构无法识别：{type(variant)}")
        variant = StackVariant(
            name=str(variant.name),
            cn_name=str(variant.cn_name),
            target_mode=str(variant.target_mode),
            base_curve_name=str(variant.base_curve_name),
            use_sample_weight=bool(variant.use_sample_weight),
            note=str(variant.note),
        )

    selected_by = str(payload.get("selected_by"))
    test_used = payload.get("test_used_for_selection")
    if selected_by != "validation_only":
        raise AssertionError(f"{path.name} selected_by 不是 validation_only：{selected_by}")
    if test_used not in (False, 0, "False", "false", "FALSE"):
        raise AssertionError(f"{path.name} test_used_for_selection 不是 false：{test_used}")

    payload_features = list(payload.get("feature_names", []))
    if payload_features != list(feature_names):
        raise AssertionError(
            f"{path.name} 的 feature_names 与当前 build_stack_features 不一致："
            f"payload={len(payload_features)}, current={len(feature_names)}"
        )
    return variant


def load_ridge_candidates(
    pool_key: str,
    curves: Dict[str, np.ndarray],
    feature_matrix: np.ndarray,
    feature_names: List[str],
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]]]:
    """从 v219 pickle 恢复 ridge_abs/ridge_residual_* 候选曲线。"""

    model_rows: List[Dict[str, object]] = []
    ridge_curves: Dict[str, np.ndarray] = {}
    model_paths = sorted((V219_DIR / "models").glob(f"v219_{pool_key}_*.pkl"))
    if not model_paths:
        raise FileNotFoundError(f"没有找到 {pool_key} 的 v219 ridge pkl")

    for path in model_paths:
        with path.open("rb") as f:
            payload = pickle.load(f)
        variant = validate_model_payload(path, payload, feature_names)

        scaler = payload["scaler"]
        model = payload["model"]
        pred_part = np.asarray(model.predict(scaler.transform(feature_matrix)), dtype=np.float64)
        if pred_part.ndim != 2:
            raise AssertionError(f"{path.name} 预测结果不是二维曲线：shape={pred_part.shape}")

        if variant.target_mode == "absolute":
            pred = pred_part
        elif variant.target_mode == "residual":
            if variant.base_curve_name not in curves:
                raise AssertionError(f"{path.name} residual base 不存在：{variant.base_curve_name}")
            pred = curves[variant.base_curve_name] + pred_part
        else:
            raise AssertionError(f"{path.name} target_mode 未知：{variant.target_mode}")

        assert_finite(f"{pool_key}:{variant.name}", pred)
        ridge_curves[variant.name] = pred.astype(np.float32)
        model_rows.append(
            {
                "pool_key": pool_key,
                "candidate_name": variant.name,
                "candidate_cn": variant.cn_name,
                "source_stage": "v219_ridge_residual_stack",
                "candidate_scope": candidate_scope(variant.name),
                "target_mode": variant.target_mode,
                "base_curve_name": variant.base_curve_name,
                "use_sample_weight": bool(variant.use_sample_weight),
                "selected_by": payload.get("selected_by"),
                "test_used_for_selection": bool(payload.get("test_used_for_selection")),
                "best_alpha": payload.get("best_alpha"),
                "model_file": str(path.relative_to(REPO_ROOT)),
                "note": variant.note,
            }
        )
    return ridge_curves, model_rows


def build_sample_manifest(pool_key: str, meta: pd.DataFrame) -> pd.DataFrame:
    """导出样本定位信息；目标派生字段不放入这个 manifest。"""

    keep_cols = [
        "array_index",
        "event_uid",
        "subject",
        "recording",
        "anchor_s",
        "scene_type",
        "pool_tier",
        "meeting_pool_flag",
        "strict_pool_flag",
        "pool_name",
        "split",
    ]
    out = meta[[col for col in keep_cols if col in meta.columns]].copy()
    out.insert(0, "pool_key", pool_key)
    return out


def peak_values(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """返回每条曲线的绝对峰值和峰值符号值。"""

    idx = np.nanargmax(np.abs(arr), axis=1)
    signed = arr[np.arange(arr.shape[0]), idx]
    return np.abs(signed), signed


def metric_for_mask(
    pool_key: str,
    pool_name: str,
    candidate_name: str,
    split_name: str,
    true_steer: np.ndarray,
    pred: np.ndarray,
    tail_mask: np.ndarray,
) -> Dict[str, object]:
    """计算与 v219 指标兼容的基础 steering 指标。"""

    diff = pred - true_steer
    steer_rmse = float(np.sqrt(np.mean(np.square(diff))))
    steer_tail_rmse = float(np.sqrt(np.mean(np.square(diff[:, tail_mask]))))
    sample_rmse = np.sqrt(np.mean(np.square(diff), axis=1))

    true_peak_abs, true_peak_signed = peak_values(true_steer)
    pred_peak_abs, pred_peak_signed = peak_values(pred)
    direction_ok = np.sign(true_peak_signed) == np.sign(pred_peak_signed)
    severe_under = pred_peak_abs < (0.5 * true_peak_abs)
    strong_mask = true_peak_abs >= 1.0

    strong_rmse = np.nan
    strong_under_rate = np.nan
    if strong_mask.any():
        strong_rmse = float(np.sqrt(np.mean(np.square(diff[strong_mask]))))
        strong_under_rate = float(np.mean(severe_under[strong_mask]))

    return {
        "pool_key": pool_key,
        "pool_name": pool_name,
        "candidate_name": candidate_name,
        "candidate_scope": candidate_scope(candidate_name),
        "split": split_name,
        "n": int(true_steer.shape[0]),
        "steer_rmse": steer_rmse,
        "steer_tail_rmse_1to2s": steer_tail_rmse,
        "steer_sample_rmse_mean": float(np.mean(sample_rmse)),
        "steer_sample_rmse_p90": float(np.quantile(sample_rmse, 0.90)),
        "steer_direction_acc": float(np.mean(direction_ok)),
        "steer_severe_under_rate": float(np.mean(severe_under)),
        "strong_response_n": int(strong_mask.sum()),
        "strong_response_rmse": strong_rmse,
        "strong_response_severe_under_rate": strong_under_rate,
        "true_peak_abs_mean": float(np.mean(true_peak_abs)),
        "pred_peak_abs_mean": float(np.mean(pred_peak_abs)),
    }


def compute_metrics(
    pool_key: str,
    pool_name: str,
    meta: pd.DataFrame,
    true_steer: np.ndarray,
    candidates: Dict[str, np.ndarray],
    future_grid: np.ndarray,
) -> pd.DataFrame:
    """按 all/train/val/test 计算候选曲线指标。"""

    split_values = meta["split"].astype(str).to_numpy()
    tail_mask = future_grid >= 1.0
    rows: List[Dict[str, object]] = []
    split_order = ["all"] + [s for s in ["train", "val", "test"] if s in set(split_values)]
    for candidate_name, pred in candidates.items():
        for split_name in split_order:
            mask = np.ones(len(split_values), dtype=bool) if split_name == "all" else split_values == split_name
            if not mask.any():
                continue
            rows.append(
                metric_for_mask(
                    pool_key,
                    pool_name,
                    candidate_name,
                    split_name,
                    true_steer[mask],
                    pred[mask],
                    tail_mask,
                )
            )
    return pd.DataFrame(rows)


def build_candidate_manifest(
    pool_key: str,
    pool_name: str,
    base_curves: Dict[str, np.ndarray],
    blend_weights: Dict[str, float],
    ridge_rows: List[Dict[str, object]],
) -> pd.DataFrame:
    """生成候选级 manifest。"""

    rows: List[Dict[str, object]] = []
    for name in base_curves:
        rows.append(
            {
                "pool_key": pool_key,
                "pool_name": pool_name,
                "candidate_name": name,
                "candidate_scope": candidate_scope(name),
                "source_stage": "v219_build_candidate_curves",
                "selected_by": "pre_existing_candidate",
                "test_used_for_selection": False,
                "target_mode": "absolute",
                "base_curve_name": "",
                "blend_weights_json": json.dumps(blend_weights, ensure_ascii=False, sort_keys=True)
                if name == "global_blend"
                else "",
                "model_file": "",
                "note": "从历史候选曲线函数重建；不使用目标派生推理特征",
            }
        )
    for row in ridge_rows:
        full = {
            "pool_key": pool_key,
            "pool_name": pool_name,
            "blend_weights_json": "",
        }
        full.update(row)
        rows.append(full)
    out = pd.DataFrame(rows)
    assert_no_forbidden_formal(out["candidate_name"].astype(str).tolist())
    return out


def export_npz(
    pool_key: str,
    X: np.ndarray,
    Y: np.ndarray,
    road_future: np.ndarray,
    meta: pd.DataFrame,
    candidates: Dict[str, np.ndarray],
    feature_matrix: np.ndarray,
    feature_names: List[str],
) -> Path:
    """导出 v222a 主缓存。"""

    candidate_names = list(candidates.keys())
    pred_stack = np.stack([candidates[name] for name in candidate_names], axis=1).astype(np.float32)
    true_steer = Y[:, :, 0].astype(np.float32)

    if pred_stack.shape[0] != X.shape[0] or pred_stack.shape[2] != true_steer.shape[1]:
        raise AssertionError(
            f"{pool_key} 预测曲线 shape 不一致：pred={pred_stack.shape}, X={X.shape}, true={true_steer.shape}"
        )
    if feature_matrix.shape[0] != X.shape[0]:
        raise AssertionError(f"{pool_key} feature_matrix 行数与样本数不一致")

    payload: Dict[str, object] = {
        "X_hist": X.astype(np.float32),
        "Y_future": Y.astype(np.float32),
        "true_steer": true_steer,
        "road_future": road_future.astype(np.float32),
        "candidate_names": np.array(candidate_names, dtype="U80"),
        "predictions": pred_stack,
        "feature_matrix": feature_matrix.astype(np.float32),
        "feature_names": np.array(feature_names, dtype="U120"),
        "array_index": meta["array_index"].to_numpy(dtype=np.int64),
        "split": meta["split"].astype(str).to_numpy(dtype="U16"),
        "event_uid": meta["event_uid"].astype(str).to_numpy(dtype="U160"),
    }
    for name in candidate_names:
        payload[safe_key(name)] = candidates[name].astype(np.float32)

    path = OUT_DIR / f"candidate_predictions_{pool_key}.npz"
    np.savez_compressed(path, **payload)
    return path


def compare_with_v219(metrics: pd.DataFrame) -> pd.DataFrame:
    """把本次重建指标与 v219 既有指标交叉检查。"""

    ref_path = V219_DIR / "tables" / "v219_metrics_by_model_split.csv"
    if not ref_path.exists():
        raise FileNotFoundError(f"缺少 v219 指标表：{ref_path}")
    ref = pd.read_csv(ref_path, encoding="utf-8-sig")
    own = metrics[metrics["split"].isin(["train", "val", "test"])].copy()
    key_cols = ["pool_key", "candidate_name", "split"]
    ref = ref.rename(columns={"model_name": "candidate_name"})
    merged = own.merge(
        ref,
        on=key_cols,
        how="inner",
        suffixes=("_rebuilt", "_v219"),
    )
    metric_cols = [
        "steer_rmse",
        "steer_tail_rmse_1to2s",
        "steer_direction_acc",
        "steer_severe_under_rate",
    ]
    for col in metric_cols:
        merged[f"{col}_abs_diff"] = np.abs(merged[f"{col}_rebuilt"] - merged[f"{col}_v219"])
    merged["max_abs_diff"] = merged[[f"{col}_abs_diff" for col in metric_cols]].max(axis=1)
    merged["crosscheck_abs_tol"] = CROSSCHECK_ABS_TOL
    merged["crosscheck_status"] = np.where(merged["max_abs_diff"] <= CROSSCHECK_ABS_TOL, "pass", "fail")

    bad = merged[merged["crosscheck_status"].eq("fail")]
    if not bad.empty:
        cols = key_cols + ["max_abs_diff"]
        raise AssertionError("本次重建指标与 v219 不一致：\n" + bad[cols].head(20).to_string(index=False))
    return merged


def make_report(
    pool_summaries: List[Dict[str, object]],
    candidate_manifest: pd.DataFrame,
    feature_audit: pd.DataFrame,
    metrics: pd.DataFrame,
    crosscheck: pd.DataFrame,
    zip_path: Path,
) -> None:
    """生成中文报告，便于 note layer 直接引用。"""

    formal = candidate_manifest[candidate_manifest["candidate_scope"].eq("formal")]
    diagnostic = candidate_manifest[candidate_manifest["candidate_scope"].eq("diagnostic")]
    test_metrics = metrics[(metrics["split"].eq("test")) & (metrics["candidate_scope"].eq("formal"))].copy()
    best_by_pool = (
        test_metrics.sort_values(["pool_key", "steer_rmse"], ascending=[True, True])
        .groupby("pool_key")
        .head(3)
    )

    lines: List[str] = []
    lines.append("# v222a 候选曲线缓存导出报告")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("- 已从 v218/v219 历史模块重建每个 pool 的候选曲线，并导出 v222a 可读取的 NPZ 缓存。")
    lines.append("- ridge residual 候选全部通过 `selected_by=validation_only` 与 `test_used_for_selection=false` 校验。")
    lines.append("- `feature_schema_audit.csv` 未发现 split、subject、true、oracle、RMSE、severe-under 等禁用字段。")
    lines.append("- 与 v219 既有指标表的数值交叉检查全部通过。")
    lines.append("- 本阶段没有训练新模型，也没有把 `W3_B4_original_soft` 放入 formal 候选或榜单。")
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    for item in pool_summaries:
        lines.append(
            f"- `{Path(str(item['npz_path'])).name}`：pool={item['pool_key']}，"
            f"样本={item['n_samples']}，候选={item['n_candidates']}，"
            f"predictions_shape={item['predictions_shape']}，feature_shape={item['feature_shape']}"
        )
    lines.append("- `candidate_manifest.csv`：候选级来源、scope 与 validation-only 元数据。")
    lines.append("- `sample_manifest.csv`：样本定位与 split 字段，仅供审计和分组，不作为推理特征。")
    lines.append("- `feature_schema_audit.csv`：v219 ridge residual feature schema 泄漏审计。")
    lines.append("- `candidate_curve_metrics.csv`：候选曲线按 pool/split 的评估指标。")
    lines.append("- `metric_crosscheck_vs_v219.csv`：本次重建与 v219 原表的指标差异。")
    lines.append(f"- `{zip_path.name}`：本阶段打包文件。")
    lines.append("")
    lines.append("## 候选范围")
    lines.append("")
    lines.append(f"- formal 候选行数：{len(formal)}")
    lines.append(f"- diagnostic 候选行数：{len(diagnostic)}")
    lines.append("- formal 候选名：" + ", ".join(sorted(formal["candidate_name"].unique())))
    lines.append("")
    lines.append("## Test split formal 候选前三")
    lines.append("")
    if best_by_pool.empty:
        lines.append("- 未产生 test formal 指标。")
    else:
        for row in best_by_pool.itertuples(index=False):
            lines.append(
                f"- {row.pool_key} / {row.candidate_name}: RMSE={row.steer_rmse:.6f}, "
                f"tail={row.steer_tail_rmse_1to2s:.6f}, under={row.steer_severe_under_rate:.6f}"
            )
    lines.append("")
    lines.append("## 审计摘要")
    lines.append("")
    lines.append(f"- feature schema 行数：{len(feature_audit)}，fail 行数：{int(feature_audit['guard_status'].eq('fail').sum())}")
    lines.append(f"- v219 交叉检查行数：{len(crosscheck)}，最大差异：{float(crosscheck['max_abs_diff'].max()):.12g}")
    lines.append("")

    report_path = REPORT_DIR / "v222a_candidate_curve_cache_report_cn.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def zip_outputs() -> Path:
    """打包本次输出并验证 ZIP 可读。"""

    zip_path = OUT_DIR / "v222a_candidate_curve_cache_pack.zip"
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
    """主流程：恢复曲线、导出缓存、审计、交叉检查、打包。"""

    clean_out_dir()
    install_pyc_finder()

    import stage03_cached_dataset_io_20260613 as cache_io
    import stage03_v216_joint_driver_vehicle_prediction_20260619 as v216
    import stage03_v218_peak_protected_joint_training_20260620 as v218
    import stage03_v219_ridge_residual_stack_20260620 as v219

    loaded = cache_io.load_both_pools()
    future_grid = v216.future_grid(v216.CFG)
    device = torch.device("cpu")

    candidate_manifest_frames: List[pd.DataFrame] = []
    sample_manifest_frames: List[pd.DataFrame] = []
    feature_audit_frames: List[pd.DataFrame] = []
    metric_frames: List[pd.DataFrame] = []
    pool_summaries: List[Dict[str, object]] = []

    for display_name, loaded_pool in loaded.items():
        pool_key = str(loaded_pool["pool_key"])
        X, Y, road_future, meta, audit = v218.load_or_build_pool_arrays(pool_key, loaded_pool)
        pool_name = str(meta["pool_name"].iloc[0]) if "pool_name" in meta.columns else str(display_name)
        true_steer = Y[:, :, 0]

        curves, blend_weights = v219.build_candidate_curves(pool_key, X, road_future, true_steer, meta, device)
        curves = {name: np.asarray(pred, dtype=np.float32) for name, pred in curves.items()}
        feature_matrix, feature_names = v219.build_stack_features(X, road_future, curves)
        feature_matrix = np.asarray(feature_matrix, dtype=np.float32)

        feature_audit = audit_feature_schema(pool_key, feature_names)
        ridge_curves, ridge_rows = load_ridge_candidates(pool_key, curves, feature_matrix, feature_names)
        candidates: Dict[str, np.ndarray] = {}
        for name in sorted(curves):
            candidates[name] = curves[name]
        for name in sorted(ridge_curves):
            candidates[name] = ridge_curves[name]

        for name, pred in candidates.items():
            if pred.shape != true_steer.shape:
                raise AssertionError(f"{pool_key}:{name} shape={pred.shape}，期望={true_steer.shape}")
            assert_finite(f"{pool_key}:{name}", pred)

        candidate_manifest = build_candidate_manifest(pool_key, pool_name, curves, blend_weights, ridge_rows)
        sample_manifest = build_sample_manifest(pool_key, meta)
        metrics = compute_metrics(pool_key, pool_name, meta, true_steer, candidates, future_grid)
        npz_path = export_npz(pool_key, X, Y, road_future, meta, candidates, feature_matrix, feature_names)

        with np.load(npz_path, allow_pickle=False) as npz:
            pred_shape = tuple(npz["predictions"].shape)
            feature_shape = tuple(npz["feature_matrix"].shape)
        pool_summaries.append(
            {
                "pool_key": pool_key,
                "pool_name": pool_name,
                "npz_path": str(npz_path),
                "n_samples": int(X.shape[0]),
                "n_candidates": int(len(candidates)),
                "predictions_shape": str(pred_shape),
                "feature_shape": str(feature_shape),
                "audit_rows": int(len(audit)),
            }
        )

        candidate_manifest_frames.append(candidate_manifest)
        sample_manifest_frames.append(sample_manifest)
        feature_audit_frames.append(feature_audit)
        metric_frames.append(metrics)

    candidate_manifest_all = pd.concat(candidate_manifest_frames, ignore_index=True)
    sample_manifest_all = pd.concat(sample_manifest_frames, ignore_index=True)
    feature_audit_all = pd.concat(feature_audit_frames, ignore_index=True)
    metrics_all = pd.concat(metric_frames, ignore_index=True)
    crosscheck = compare_with_v219(metrics_all)

    write_csv(candidate_manifest_all, OUT_DIR / "candidate_manifest.csv")
    write_csv(sample_manifest_all, OUT_DIR / "sample_manifest.csv")
    write_csv(feature_audit_all, TABLE_DIR / "feature_schema_audit.csv")
    write_csv(metrics_all, TABLE_DIR / "candidate_curve_metrics.csv")
    write_csv(crosscheck, TABLE_DIR / "metric_crosscheck_vs_v219.csv")
    write_csv(pd.DataFrame(pool_summaries), TABLE_DIR / "pool_cache_summary.csv")

    leakage_rows = [
        {
            "check_name": "feature_schema_forbidden_tokens",
            "status": "pass" if feature_audit_all["guard_status"].eq("pass").all() else "fail",
            "detail": "feature schema 未命中禁用字段",
        },
        {
            "check_name": "formal_candidate_forbidden_names",
            "status": "pass",
            "detail": "formal 候选不含 W3_B4_original_soft/oracle/fallback/true_label",
        },
        {
            "check_name": "v219_metric_crosscheck",
            "status": "pass" if crosscheck["crosscheck_status"].eq("pass").all() else "fail",
            "detail": f"max_abs_diff={float(crosscheck['max_abs_diff'].max()):.12g}",
        },
        {
            "check_name": "ridge_selected_by_validation_only",
            "status": "pass",
            "detail": "全部 v219 ridge payload 均为 validation_only 且 test_used_for_selection=false",
        },
    ]
    write_csv(pd.DataFrame(leakage_rows), TABLE_DIR / "leakage_guard_result.csv")

    manifest = {
        "stage": "v222a_candidate_curve_cache",
        "created_by": Path(__file__).name,
        "output_dir": str(OUT_DIR),
        "pool_summaries": pool_summaries,
        "formal_candidates": sorted(FORMAL_CANDIDATES),
        "diagnostic_candidates": sorted(DIAGNOSTIC_CANDIDATES),
        "test_used_for_selection": False,
        "notes": [
            "本阶段只导出缓存和审计，不训练新模型。",
            "true_steer/Y_future 是标签数组，不属于推理特征。",
            "feature_matrix 来自 v219 build_stack_features，并通过禁用字段审计。",
        ],
    }
    (LOG_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = zip_outputs()
    make_report(pool_summaries, candidate_manifest_all, feature_audit_all, metrics_all, crosscheck, zip_path)

    # 报告是在 zip 之后生成的，为了让 zip 包也包含报告，再打包一次。
    zip_path = zip_outputs()

    print("v222a candidate curve cache finished.")
    print(f"output_dir={OUT_DIR}")
    print(f"candidate_manifest={OUT_DIR / 'candidate_manifest.csv'}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
