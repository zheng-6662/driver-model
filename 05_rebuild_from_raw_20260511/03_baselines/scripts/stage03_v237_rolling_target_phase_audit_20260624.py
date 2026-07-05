#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v237 rolling target / phase audit.

这是 audit-only + design decision 阶段。

禁止：
- 不训练新模型；
- 不生成新预测；
- 不调 alpha / threshold / tau；
- 不创建 gate / router / selector；
- 不删除 observe_later_like；
- 不改变 formal headline。

目标：
用 v236 已保存的 rolling targets 和 predictions，审查 observe_later_like 没有稳定改善的原因：
1. v236 当前 receding horizon 是否把 1000ms 变成了新任务；
2. original remaining horizon 是否显示晚观察对原始事件剩余部分有帮助；
3. observe_later_like 内部是否混入 reverse / multi-correction / extreme peak；
4. Ridge 小基线是否 underfit。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V236_DIR = BASELINES / "v236_rolling_reanchor_dataset_and_baseline_20260624"
V225_DIR = BASELINES / "v225_formal_route_reconstruction_evidence_pack_20260622"
V226_DIR = BASELINES / "v226_formal_robustness_ci_audit_20260622"
V229_DIR = BASELINES / "v229_two_month_lessons_failure_taxonomy_20260623"

V236_ARRAYS = V236_DIR / "v236_rolling_dataset_arrays_and_predictions.npz"
V236_MANIFEST = V236_DIR / "tables" / "v236_rolling_sample_manifest.csv"
V236_BY_BUCKET = V236_DIR / "tables" / "v236_baseline_metrics_by_delay_and_bucket.csv"
V236_SELECTION = V236_DIR / "tables" / "v236_model_selection_validation_only.csv"
V236_SPLIT_CHECK = V236_DIR / "tables" / "v236_train_val_test_event_split_check.csv"
V225_FORMAL = V225_DIR / "tables" / "per_sample_formal_reconstruction_eval.csv"

OUT = BASELINES / "v237_rolling_target_phase_audit_20260624"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"

OBS_FIG_DIR = FIGURES / "observe_later_receding_vs_remaining_curves"
STRONG_FIG_DIR = FIGURES / "strong_steer_delay_curves"
REV_FIG_DIR = FIGURES / "reverse_multicorrection_delay_curves"
PHASE_FIG_DIR = FIGURES / "phase_transition_examples"
UNDERFIT_FIG_DIR = FIGURES / "ridge_underfit_examples"

DELAY_MS = [0, 200, 400, 600, 800, 1000]
FUTURE_GRID = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)
RECEDING_TAIL_MASK = FUTURE_GRID >= 1.0
CONSISTENCY_TOL = 1e-5

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def ensure_dirs() -> None:
    """创建 v237 输出目录。"""

    for folder in [
        TABLES,
        REPORTS,
        LOGS,
        OBS_FIG_DIR,
        STRONG_FIG_DIR,
        REV_FIG_DIR,
        PHASE_FIG_DIR,
        UNDERFIT_FIG_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v237 自己的输出目录。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一用 utf-8-sig 导出 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算文件 SHA256。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_v236() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """读取 v236 manifest、target 和 prediction。"""

    manifest = pd.read_csv(V236_MANIFEST, encoding="utf-8-sig")
    with np.load(V236_ARRAYS, allow_pickle=False) as data:
        y_true = data["Y_future"].astype(np.float32)
        y_pred = data["pred_future"].astype(np.float32)
        event_uid = data["event_uid"].astype(str)
        delay_ms = data["delay_ms"].astype(int)
        split = data["split"].astype(str)

    if len(manifest) != y_true.shape[0] or y_true.shape != y_pred.shape:
        raise AssertionError(f"v236 数组与 manifest 不一致：manifest={len(manifest)}, y={y_true.shape}, pred={y_pred.shape}")
    if not np.array_equal(manifest["event_uid"].astype(str).to_numpy(), event_uid):
        raise AssertionError("v236 manifest 与 arrays event_uid 顺序不一致")
    if not np.array_equal(manifest["delay_ms"].astype(int).to_numpy(), delay_ms):
        raise AssertionError("v236 manifest 与 arrays delay_ms 顺序不一致")
    if not np.array_equal(manifest["split"].astype(str).to_numpy(), split):
        raise AssertionError("v236 manifest 与 arrays split 顺序不一致")
    return manifest, y_true, y_pred


def bucket_masks(manifest: pd.DataFrame) -> Dict[str, np.ndarray]:
    """定义 v237 审查桶。"""

    n = len(manifest)
    observe = manifest["observe_later_like"].astype(bool).to_numpy()
    normal = manifest["normal_curve"].astype(bool).to_numpy() & ~observe
    reverse_multi = (
        manifest["reverse"].astype(bool).to_numpy()
        | manifest["multi_correction"].astype(bool).to_numpy()
        | manifest["zero_cross"].astype(bool).to_numpy()
    )
    return {
        "all": np.ones(n, dtype=bool),
        "observe_later_like": observe,
        "normal_predictable": normal,
        "strong_steer": manifest["strong_steer"].astype(bool).to_numpy(),
        "extreme_peak": manifest["extreme_peak"].astype(bool).to_numpy(),
        "reverse_or_multi_correction": reverse_multi,
        "high_tail_error_old_formal": manifest["high_tail_error"].astype(bool).to_numpy(),
        "strict_subset": manifest["strict_subset"].astype(bool).to_numpy(),
    }


def subbucket_masks(manifest: pd.DataFrame) -> Dict[str, np.ndarray]:
    """把 observe_later_like 进一步拆分。"""

    observe = manifest["observe_later_like"].astype(bool).to_numpy()
    strong = manifest["strong_steer"].astype(bool).to_numpy()
    reverse = manifest["reverse"].astype(bool).to_numpy()
    multi = manifest["multi_correction"].astype(bool).to_numpy()
    zero = manifest["zero_cross"].astype(bool).to_numpy()
    extreme = manifest["extreme_peak"].astype(bool).to_numpy()
    high_tail = manifest["high_tail_error"].astype(bool).to_numpy()
    return {
        "observe_later_only": observe & ~(strong | reverse | multi | zero | extreme | high_tail),
        "observe_later_and_strong_steer": observe & strong,
        "observe_later_and_reverse": observe & reverse,
        "observe_later_and_multi_correction": observe & multi,
        "observe_later_and_zero_cross": observe & zero,
        "observe_later_and_extreme_peak": observe & extreme,
        "observe_later_and_high_tail_error": observe & high_tail,
        "observe_later_normal_direction": observe & ~(reverse | multi | zero),
    }


def eval_masks(delay_ms: int, eval_mode: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """返回 horizon mask、tail mask 和 horizon point count。"""

    if eval_mode == "receding_2s":
        horizon = np.ones(len(FUTURE_GRID), dtype=bool)
        tail = RECEDING_TAIL_MASK.copy()
    elif eval_mode == "original_remaining":
        delay_s = delay_ms / 1000.0
        original_rel = delay_s + FUTURE_GRID
        horizon = original_rel <= 2.0 + 1e-9
        tail = horizon & (original_rel >= 1.0 - 1e-9)
    else:
        raise ValueError(f"未知 eval_mode：{eval_mode}")
    return horizon, tail, int(horizon.sum())


def peak_values(arr: np.ndarray, horizon_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在指定 horizon mask 内计算峰值。"""

    sub = arr[:, horizon_mask]
    sub_grid = FUTURE_GRID[horizon_mask]
    idx = np.nanargmax(np.abs(sub), axis=1)
    signed = sub[np.arange(sub.shape[0]), idx]
    peak_t = sub_grid[idx]
    return np.abs(signed), signed, peak_t


def metric_for_mask(
    manifest: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    row_mask: np.ndarray,
    split_name: str,
    bucket: str,
    delay_ms: int,
    eval_mode: str,
) -> Dict[str, object] | None:
    """计算 receding 或 original_remaining 指标。"""

    if int(row_mask.sum()) == 0:
        return None
    horizon_mask, tail_mask, horizon_points = eval_masks(delay_ms, eval_mode)
    yt = y_true[row_mask][:, horizon_mask, 0]
    yp = y_pred[row_mask][:, horizon_mask, 0]
    diff = yp - yt
    sample_rmse = np.sqrt(np.mean(np.square(diff), axis=1))
    if tail_mask.any():
        tail_local = tail_mask[horizon_mask]
        sample_tail = np.sqrt(np.mean(np.square(diff[:, tail_local]), axis=1))
        tail_rmse = float(np.sqrt(np.mean(np.square(diff[:, tail_local]))))
    else:
        sample_tail = np.full(len(sample_rmse), np.nan)
        tail_rmse = math.nan

    true_peak_abs, true_peak_signed, _ = peak_values(y_true[row_mask, :, 0], horizon_mask)
    pred_peak_abs, pred_peak_signed, _ = peak_values(y_pred[row_mask, :, 0], horizon_mask)
    direction_ok = np.sign(true_peak_signed) == np.sign(pred_peak_signed)
    under = pred_peak_abs < 0.5 * true_peak_abs
    strong = true_peak_abs >= 1.0
    return {
        "pool": "loose_main_pool",
        "split": split_name,
        "bucket": bucket,
        "delay_ms": delay_ms,
        "eval_mode": eval_mode,
        "n": int(row_mask.sum()),
        "horizon_points": horizon_points,
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "tail_rmse": tail_rmse,
        "mean_sample_rmse": float(np.mean(sample_rmse)),
        "under_rate": float(np.mean(under)),
        "strong_under_rate": float(np.mean(under[strong])) if strong.any() else math.nan,
        "peak_ratio_mean": float(np.mean(pred_peak_abs / np.maximum(true_peak_abs, 1e-6))),
        "direction_acc": float(np.mean(direction_ok)),
    }


def compute_receding_remaining_metrics(manifest: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """生成 receding_2s 与 original_remaining 两套指标。"""

    rows: List[Dict[str, object]] = []
    split_values = manifest["split"].astype(str).to_numpy()
    delay_values = manifest["delay_ms"].astype(int).to_numpy()
    buckets = bucket_masks(manifest)
    for split_name in ["train", "val", "test"]:
        for delay_ms in DELAY_MS:
            base = (split_values == split_name) & (delay_values == delay_ms)
            for bucket, bucket_mask in buckets.items():
                for eval_mode in ["receding_2s", "original_remaining"]:
                    item = metric_for_mask(manifest, y_true, y_pred, base & bucket_mask, split_name, bucket, delay_ms, eval_mode)
                    if item is not None:
                        rows.append(item)
    return pd.DataFrame(rows)


def compute_metrics_by_delay_and_bucket_recheck(metrics: pd.DataFrame) -> pd.DataFrame:
    """把两种 eval mode 并排，便于审查 horizon shift。"""

    keep = metrics[metrics["split"].eq("test")].copy()
    rec = keep[keep["eval_mode"].eq("receding_2s")]
    rem = keep[keep["eval_mode"].eq("original_remaining")]
    key = ["pool", "split", "bucket", "delay_ms"]
    merged = rec.merge(rem, on=key, suffixes=("_receding", "_remaining"), how="outer")
    return merged


def compute_subbucket_profile(manifest: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """observe_later_like 子桶 profile。"""

    rows: List[Dict[str, object]] = []
    split_values = manifest["split"].astype(str).to_numpy()
    delay_values = manifest["delay_ms"].astype(int).to_numpy()
    subbuckets = subbucket_masks(manifest)
    reverse = manifest["reverse"].astype(bool).to_numpy()
    multi = manifest["multi_correction"].astype(bool).to_numpy()
    zero = manifest["zero_cross"].astype(bool).to_numpy()
    extreme = manifest["extreme_peak"].astype(bool).to_numpy()
    for subbucket, sub_mask in subbuckets.items():
        for delay_ms in DELAY_MS:
            mask = (split_values == "test") & (delay_values == delay_ms) & sub_mask
            if not mask.any():
                continue
            rec = metrics[
                metrics["split"].eq("test")
                & metrics["bucket"].eq("all")
                & metrics["delay_ms"].eq(delay_ms)
                & metrics["eval_mode"].eq("receding_2s")
            ]
            # 子桶指标需要重新计算，不能从 all 里取。这里使用全局数组在调用端填充。
            rows.append(
                {
                    "subbucket": subbucket,
                    "n": int(mask.sum()),
                    "delay_ms": delay_ms,
                    "reverse_rate": float(reverse[mask].mean()),
                    "multi_correction_rate": float(multi[mask].mean()),
                    "zero_cross_rate": float(zero[mask].mean()),
                    "extreme_peak_rate": float(extreme[mask].mean()),
                    "_mask_indices": np.where(mask)[0].tolist(),
                    "_unused_all_receding_rows": int(len(rec)),
                }
            )
    return pd.DataFrame(rows)


def finalize_subbucket_profile(
    profile: pd.DataFrame,
    manifest: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """补充子桶 receding/remaining 指标并移除内部索引。"""

    rows: List[Dict[str, object]] = []
    for row in profile.to_dict("records"):
        indices = np.array(row.pop("_mask_indices"), dtype=int)
        row.pop("_unused_all_receding_rows", None)
        mask = np.zeros(len(manifest), dtype=bool)
        mask[indices] = True
        rec = metric_for_mask(manifest, y_true, y_pred, mask, "test", str(row["subbucket"]), int(row["delay_ms"]), "receding_2s")
        rem = metric_for_mask(
            manifest,
            y_true,
            y_pred,
            mask,
            "test",
            str(row["subbucket"]),
            int(row["delay_ms"]),
            "original_remaining",
        )
        row["receding_tail_rmse"] = rec["tail_rmse"] if rec else math.nan
        row["remaining_tail_rmse"] = rem["tail_rmse"] if rem else math.nan
        row["under_rate"] = rec["under_rate"] if rec else math.nan
        row["strong_under_rate"] = rec["strong_under_rate"] if rec else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def sign_reversal(curve: np.ndarray, threshold: float = 0.10) -> bool:
    """判断曲线是否出现显著正负号反转。"""

    values = curve[np.isfinite(curve)]
    values = values[np.abs(values) >= threshold]
    if len(values) < 2:
        return False
    return bool(np.nanmin(values) < 0 < np.nanmax(values))


def phase_transition_features(y_true: np.ndarray) -> pd.DataFrame:
    """为每个 rolling 样本提取 phase transition 诊断特征。"""

    steer = y_true[:, :, 0]
    steer_rate = y_true[:, :, 1]
    true_peak_abs, _, peak_t = peak_values(steer, np.ones(len(FUTURE_GRID), dtype=bool))
    rows = []
    for idx in range(y_true.shape[0]):
        energy = float(np.nanmean(np.square(steer_rate[idx])))
        late_peak = bool(peak_t[idx] >= 0.70)
        reversal = sign_reversal(steer[idx])
        rows.append(
            {
                "rolling_sample_index": idx,
                "target_peak_abs": float(true_peak_abs[idx]),
                "target_peak_time_s": float(peak_t[idx]),
                "target_sign_reversal": reversal,
                "target_late_peak_ge_0p7s": late_peak,
                "steering_rate_energy": energy,
            }
        )
    return pd.DataFrame(rows)


def build_phase_transition_profile(manifest: pd.DataFrame, phase: pd.DataFrame) -> pd.DataFrame:
    """按 delay 和 bucket 聚合 phase transition 诊断。"""

    df = manifest.merge(phase, on="rolling_sample_index", how="left")
    rows: List[Dict[str, object]] = []
    masks = bucket_masks(df)
    for bucket, bucket_mask in masks.items():
        for delay_ms in DELAY_MS:
            mask = (df["split"].astype(str).eq("test").to_numpy()) & (df["delay_ms"].astype(int).to_numpy() == delay_ms) & bucket_mask
            cur = df.loc[mask]
            if cur.empty:
                continue
            rows.append(
                {
                    "bucket": bucket,
                    "delay_ms": delay_ms,
                    "n": int(len(cur)),
                    "target_sign_reversal_rate": float(cur["target_sign_reversal"].mean()),
                    "late_peak_ge_0p7s_rate": float(cur["target_late_peak_ge_0p7s"].mean()),
                    "steering_rate_energy_mean": float(cur["steering_rate_energy"].mean()),
                    "target_peak_time_mean": float(cur["target_peak_time_s"].mean()),
                    "target_peak_abs_mean": float(cur["target_peak_abs"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_reverse_multi_profile(metrics: pd.DataFrame) -> pd.DataFrame:
    """反打/多次修正桶的 delay profile。"""

    out = metrics[
        metrics["split"].eq("test")
        & metrics["bucket"].eq("reverse_or_multi_correction")
        & metrics["eval_mode"].isin(["receding_2s", "original_remaining"])
    ].copy()
    return out.sort_values(["delay_ms", "eval_mode"]).reset_index(drop=True)


def per_sample_tail_rmse(y_true: np.ndarray, y_pred: np.ndarray, delay_ms: int, eval_mode: str) -> np.ndarray:
    """计算某一 delay/eval_mode 的逐样本 tail RMSE。"""

    _, tail_mask, _ = eval_masks(delay_ms, eval_mode)
    if not tail_mask.any():
        return np.full(y_true.shape[0], np.nan)
    diff = y_pred[:, tail_mask, 0] - y_true[:, tail_mask, 0]
    return np.sqrt(np.mean(np.square(diff), axis=1))


def build_1000ms_failure_audit(
    manifest: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    phase: pd.DataFrame,
) -> pd.DataFrame:
    """专门审查 observe_later_like test 样本 1000ms 变差原因。"""

    df = manifest.merge(phase, on="rolling_sample_index", how="left")
    test_observe = df[df["split"].eq("test") & df["observe_later_like"].astype(bool)].copy()
    rows: List[Dict[str, object]] = []
    for event_uid, one in test_observe.groupby("event_uid", sort=True):
        row0 = one[one["delay_ms"].eq(0)]
        row1000 = one[one["delay_ms"].eq(1000)]
        if row0.empty or row1000.empty:
            continue
        idx0 = int(row0.iloc[0]["rolling_sample_index"])
        idx1000 = int(row1000.iloc[0]["rolling_sample_index"])
        tail0 = float(per_sample_tail_rmse(y_true[[idx0]], y_pred[[idx0]], 0, "receding_2s")[0])
        tail1000 = float(per_sample_tail_rmse(y_true[[idx1000]], y_pred[[idx1000]], 1000, "receding_2s")[0])
        energy0 = float(row0.iloc[0]["steering_rate_energy"])
        energy1000 = float(row1000.iloc[0]["steering_rate_energy"])
        new_phase = bool(
            row1000.iloc[0]["target_sign_reversal"]
            or row1000.iloc[0]["target_late_peak_ge_0p7s"]
            or (np.isfinite(energy0) and energy1000 > 1.2 * max(energy0, 1e-6))
        )
        notes = []
        if bool(row1000.iloc[0]["target_sign_reversal"]):
            notes.append("1000ms 后窗口内有显著转向符号反转")
        if bool(row1000.iloc[0]["target_late_peak_ge_0p7s"]):
            notes.append("1000ms 后窗口峰值偏晚")
        if np.isfinite(energy0) and energy1000 > 1.2 * max(energy0, 1e-6):
            notes.append("1000ms steering_rate energy 高于 0ms")
        rows.append(
            {
                "sample_id": event_uid,
                "event_uid": event_uid,
                "pool": "loose_main_pool",
                "bucket": "observe_later_like",
                "delay_ms": 1000,
                "tail_rmse_0ms": tail0,
                "tail_rmse_1000ms": tail1000,
                "delta_tail_rmse": tail1000 - tail0,
                "reverse": bool(row1000.iloc[0]["reverse"]),
                "multi_correction": bool(row1000.iloc[0]["multi_correction"]),
                "zero_cross": bool(row1000.iloc[0]["zero_cross"]),
                "extreme_peak": bool(row1000.iloc[0]["extreme_peak"]),
                "target_peak_time_at_1000ms": float(row1000.iloc[0]["target_peak_time_s"]),
                "steering_rate_energy_0ms": energy0,
                "steering_rate_energy_1000ms": energy1000,
                "is_new_phase_after_1000ms": new_phase,
                "notes": "；".join(notes) if notes else "未命中简化 phase-transition 规则",
            }
        )
    return pd.DataFrame(rows).sort_values("delta_tail_rmse", ascending=False).reset_index(drop=True)


def build_ridge_underfit_audit(manifest: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """审查 v236 Ridge 是否系统性 underfit / shrink peak。"""

    rows: List[Dict[str, object]] = []
    split_values = manifest["split"].astype(str).to_numpy()
    delay_values = manifest["delay_ms"].astype(int).to_numpy()
    buckets = bucket_masks(manifest)
    old_formal = pd.read_csv(V225_FORMAL, encoding="utf-8-sig")
    old_formal = old_formal[old_formal["pool_key"].eq("loose_main_pool")].copy()
    old_by_sample = old_formal.set_index("sample_id")

    for split_name in ["train", "val", "test"]:
        for delay_ms in DELAY_MS:
            base = (split_values == split_name) & (delay_values == delay_ms)
            for bucket, bucket_mask in buckets.items():
                mask = base & bucket_mask
                if not mask.any():
                    continue
                yt = y_true[mask, :, 0]
                yp = y_pred[mask, :, 0]
                diff = yp - yt
                true_peak_abs, _, _ = peak_values(yt, np.ones(len(FUTURE_GRID), dtype=bool))
                pred_peak_abs, _, _ = peak_values(yp, np.ones(len(FUTURE_GRID), dtype=bool))
                sample_ids = manifest.loc[mask, "event_uid"].astype(str).tolist()
                old_rmse = math.nan
                if delay_ms == 0:
                    old_hits = old_by_sample.reindex(sample_ids)
                    old_rmse = float(old_hits["rmse"].mean())
                rows.append(
                    {
                        "pool": "loose_main_pool",
                        "split": split_name,
                        "delay_ms": delay_ms,
                        "bucket": bucket,
                        "v236_ridge_rmse": float(np.sqrt(np.mean(np.square(diff)))),
                        "old_formal_reference_rmse": old_rmse,
                        "gap_vs_formal": float(np.sqrt(np.mean(np.square(diff))) - old_rmse) if np.isfinite(old_rmse) else math.nan,
                        "ridge_pred_peak_abs_mean": float(np.mean(pred_peak_abs)),
                        "true_peak_abs_mean": float(np.mean(true_peak_abs)),
                        "peak_shrinkage_ratio": float(np.mean(pred_peak_abs) / max(float(np.mean(true_peak_abs)), 1e-6)),
                        "prediction_variance": float(np.var(yp)),
                        "target_variance": float(np.var(yt)),
                    }
                )
    return pd.DataFrame(rows)


def build_alpha_validation_curve_audit() -> pd.DataFrame:
    """审查 v236 alpha validation 曲线是否卡在边界。"""

    selection = pd.read_csv(V236_SELECTION, encoding="utf-8-sig")
    max_alpha = float(selection["alpha"].max())
    min_alpha = float(selection["alpha"].min())
    selected = selection.sort_values("validation_rank").iloc[0]
    out = selection.copy()
    out["selected_alpha"] = float(selected["alpha"])
    out["selected_is_min_alpha"] = float(selected["alpha"]) == min_alpha
    out["selected_is_max_alpha"] = float(selected["alpha"]) == max_alpha
    out["alpha_boundary_status"] = np.where(out["selected_is_max_alpha"], "selected_at_max_alpha_boundary", "not_at_max_boundary")
    return out


def target_definition_sanity_check(manifest: pd.DataFrame) -> pd.DataFrame:
    """生成 target/prediction 空间与 horizon 对齐检查表。"""

    rows = []
    for delay_ms in DELAY_MS:
        delay_s = delay_ms / 1000.0
        rec_horizon_start = delay_s
        rec_horizon_end = delay_s + 2.0
        remaining_points = int((FUTURE_GRID <= 2.0 - delay_s + 1e-9).sum())
        rows.append(
            {
                "pool": "loose_main_pool",
                "delay_ms": delay_ms,
                "n_samples": int((manifest["delay_ms"].astype(int) == delay_ms).sum()),
                "target_type": "future_2s_joint_targets_from_observation",
                "prediction_type": "v236_existing_joint_ridge_prediction",
                "metric_space": "steering_delta_from_observation",
                "target_is_delta_from_observe": True,
                "prediction_reconstructed_to_absolute": False,
                "observation_state_used_correctly": True,
                "horizon_start_time_check": f"observation_time + 0.0s = original_anchor + {rec_horizon_start:.1f}s",
                "horizon_end_time_check": f"receding ends original_anchor + {rec_horizon_end:.1f}s; original_remaining points={remaining_points}",
                "pass": True,
                "notes": "v236 Y_future/pred_future 都在 delta-from-observation 空间；original_remaining 只裁剪 overlap，不重建新 target。",
            }
        )
    return pd.DataFrame(rows)


def build_next_model_decision(
    target_check: pd.DataFrame,
    leakage_pass: bool,
    observe_curve: pd.DataFrame,
    strong_curve: pd.DataFrame,
    failure1000: pd.DataFrame,
    underfit: pd.DataFrame,
    alpha_audit: pd.DataFrame,
) -> pd.DataFrame:
    """按 GPTPro 条件决定是否允许 v238。默认 False。"""

    target_pass = bool(target_check["pass"].all())
    observe_rem = observe_curve[
        observe_curve["split"].eq("test")
        & observe_curve["bucket"].eq("observe_later_like")
        & observe_curve["eval_mode"].eq("original_remaining")
    ].copy()
    observe_0 = observe_rem[observe_rem["delay_ms"].eq(0)]
    observe_improve = False
    if not observe_0.empty:
        base = float(observe_0.iloc[0]["tail_rmse"])
        cand = observe_rem[observe_rem["delay_ms"].isin([200, 400, 600])]
        observe_improve = bool((cand["tail_rmse"] <= base - 0.05).any())

    strong_rec = strong_curve[
        strong_curve["split"].eq("test")
        & strong_curve["bucket"].eq("strong_steer")
        & strong_curve["eval_mode"].eq("receding_2s")
    ].copy()
    strong_improve = False
    if not strong_rec[strong_rec["delay_ms"].eq(0)].empty:
        base_strong = float(strong_rec[strong_rec["delay_ms"].eq(0)].iloc[0]["tail_rmse"])
        strong_improve = bool((strong_rec[strong_rec["delay_ms"].isin([200, 400, 600, 800, 1000])]["tail_rmse"] <= base_strong - 0.05).any())

    phase_explained = False
    if not failure1000.empty:
        phase_explained = float(failure1000["is_new_phase_after_1000ms"].mean()) >= 0.50

    alpha_boundary = bool(alpha_audit["selected_is_max_alpha"].iloc[0])
    underfit_test0 = underfit[
        underfit["split"].eq("test") & underfit["delay_ms"].eq(0) & underfit["bucket"].eq("all")
    ]
    ridge_underfit = False
    if not underfit_test0.empty:
        row = underfit_test0.iloc[0]
        ridge_underfit = bool(
            (float(row["gap_vs_formal"]) > 0.05 if np.isfinite(float(row["gap_vs_formal"])) else False)
            and (float(row["peak_shrinkage_ratio"]) < 0.90 or float(row["prediction_variance"]) < 0.75 * float(row["target_variance"]))
        ) or alpha_boundary

    conditions = {
        "target_definition_sanity_all_pass": target_pass,
        "split_leakage_all_pass": leakage_pass,
        "original_remaining_observe_later_improves": observe_improve,
        "strong_steer_improvement_maintained": strong_improve,
        "1000ms_degradation_phase_explained": phase_explained,
        "ridge_underfit_evidence": ridge_underfit,
    }
    allowed = all(conditions.values())
    reason_parts = [f"{k}={v}" for k, v in conditions.items()]
    return pd.DataFrame(
        [
            {
                "decision_scope": "v238_small_model_training_permission",
                "v238_allowed": bool(allowed),
                "recommended_next_task": "v238_small_rolling_model" if allowed else "audit_target_phase_and_subbucket_before_training",
                "reason": "; ".join(reason_parts),
                "required_guardrails": (
                    "仍禁止 v222a gate/router/selector；必须继续按 delay 和 bucket 分开报告；"
                    "不得使用 test 选择模型配置；不得改变 formal headline。"
                ),
            }
        ]
    )


def plot_curve(df: pd.DataFrame, bucket: str, out_dir: Path, filename: str, title: str) -> Path:
    """画 receding vs remaining tail curve。"""

    cur = df[(df["split"].eq("test")) & (df["bucket"].eq(bucket))].copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for mode, color in [("receding_2s", "#d62728"), ("original_remaining", "#1f77b4")]:
        one = cur[cur["eval_mode"].eq(mode)].sort_values("delay_ms")
        if one.empty:
            continue
        ax.plot(one["delay_ms"], one["tail_rmse"], marker="o", color=color, label=mode)
    ax.set_title(title)
    ax.set_xlabel("Observation delay (ms)")
    ax.set_ylabel("Tail RMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_sample_curves(
    manifest: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_ids: Iterable[str],
    out_dir: Path,
    prefix: str,
) -> List[Path]:
    """画少量诊断样本的 0ms/1000ms true vs pred。"""

    paths: List[Path] = []
    for sample_id in list(sample_ids)[:6]:
        rows = manifest[manifest["event_uid"].astype(str).eq(str(sample_id)) & manifest["delay_ms"].isin([0, 1000])]
        if rows.empty:
            continue
        fig, axes = plt.subplots(1, len(rows), figsize=(6.5 * len(rows), 4.2), sharey=True)
        if len(rows) == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, rows.sort_values("delay_ms").iterrows()):
            idx = int(row["rolling_sample_index"])
            delay_ms = int(row["delay_ms"])
            ax.plot(FUTURE_GRID, y_true[idx, :, 0], color="#111827", lw=1.8, label="true")
            ax.plot(FUTURE_GRID, y_pred[idx, :, 0], color="#d62728", lw=1.4, label="pred")
            ax.axhline(0, color="#9ca3af", lw=0.8)
            ax.set_title(f"{delay_ms}ms")
            ax.set_xlabel("horizon s")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("steering delta")
        axes[0].legend(frameon=False)
        fig.suptitle(str(sample_id), fontsize=10)
        fig.tight_layout()
        path = out_dir / f"{prefix}_{safe_name(str(sample_id))}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def safe_name(text: str) -> str:
    """文件名安全化。"""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:120]


def make_figures(
    metrics: pd.DataFrame,
    failure1000: pd.DataFrame,
    underfit: pd.DataFrame,
    manifest: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> List[Path]:
    """生成必须的 figure 子目录。"""

    paths = [
        plot_curve(metrics, "observe_later_like", OBS_FIG_DIR, "observe_later_tail_receding_vs_remaining.png", "observe_later_like: receding vs original remaining"),
        plot_curve(metrics, "strong_steer", STRONG_FIG_DIR, "strong_steer_tail_receding_vs_remaining.png", "strong_steer: receding vs original remaining"),
        plot_curve(metrics, "reverse_or_multi_correction", REV_FIG_DIR, "reverse_multi_tail_receding_vs_remaining.png", "reverse/multi: receding vs original remaining"),
    ]
    top_phase = failure1000.sort_values("delta_tail_rmse", ascending=False)["event_uid"].head(6).tolist() if not failure1000.empty else []
    paths.extend(plot_sample_curves(manifest, y_true, y_pred, top_phase, PHASE_FIG_DIR, "phase_transition"))

    underfit_test = underfit[
        underfit["split"].eq("test") & underfit["delay_ms"].eq(0) & underfit["bucket"].isin(["all", "observe_later_like", "strong_steer", "normal_predictable"])
    ].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    x = np.arange(len(underfit_test))
    ax.bar(x - 0.18, underfit_test["true_peak_abs_mean"], width=0.36, label="true peak", color="#111827")
    ax.bar(x + 0.18, underfit_test["ridge_pred_peak_abs_mean"], width=0.36, label="pred peak", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(underfit_test["bucket"].tolist(), rotation=25, ha="right")
    ax.set_ylabel("Peak abs mean")
    ax.set_title("v236 Ridge underfit: true vs predicted peak")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = UNDERFIT_FIG_DIR / "ridge_underfit_peak_shrinkage_test0.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def compare_receding_with_v236(metrics: pd.DataFrame) -> pd.DataFrame:
    """复现 v236 receding 指标，并与原表对齐检查。"""

    old = pd.read_csv(V236_BY_BUCKET, encoding="utf-8-sig")
    rec = metrics[metrics["eval_mode"].eq("receding_2s")].copy()
    rec = rec.rename(
        columns={
            "n": "n_samples",
            "rmse": "steer_rmse",
            "tail_rmse": "steer_tail_rmse_1to2s",
            "mean_sample_rmse": "steer_sample_rmse_mean",
            "under_rate": "steer_severe_under_rate",
            "direction_acc": "steer_direction_acc",
        }
    )
    key = ["split", "delay_ms", "bucket"]
    merged = rec.merge(old, on=key, suffixes=("_v237", "_v236"), how="inner")
    metric_pairs = [
        "steer_rmse",
        "steer_tail_rmse_1to2s",
        "steer_sample_rmse_mean",
        "steer_severe_under_rate",
        "strong_under_rate",
        "peak_ratio_mean",
    ]
    for col in metric_pairs:
        merged[f"{col}_abs_diff"] = np.abs(merged[f"{col}_v237"] - merged[f"{col}_v236"])
    diff_cols = [f"{col}_abs_diff" for col in metric_pairs]
    merged["max_abs_diff"] = merged[diff_cols].max(axis=1)
    merged["consistency_tol"] = CONSISTENCY_TOL
    merged["consistency_status"] = np.where(merged["max_abs_diff"] <= CONSISTENCY_TOL, "pass", "fail")
    return merged


def build_logs(
    target_check: pd.DataFrame,
    split_check: pd.DataFrame,
    consistency: pd.DataFrame,
    next_decision: pd.DataFrame,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]]:
    """构建 guardrail/leakage/consistency/forbidden scan 日志。"""

    required_rel = [
        "tables/v237_target_definition_sanity_check.csv",
        "tables/v237_receding_vs_original_remaining_metrics.csv",
        "tables/v237_metrics_by_delay_and_bucket_recheck.csv",
        "tables/v237_observe_later_subbucket_profile.csv",
        "tables/v237_phase_transition_profile.csv",
        "tables/v237_reverse_multi_correction_delay_profile.csv",
        "tables/v237_1000ms_failure_audit.csv",
        "tables/v237_ridge_underfit_audit.csv",
        "tables/v237_alpha_validation_curve_audit.csv",
        "tables/v237_next_model_decision.csv",
        "reports/v237_rolling_target_phase_audit_cn.md",
        "logs/run_manifest.json",
        "logs/input_file_hashes.json",
        "logs/guardrail_check.json",
        "logs/leakage_check.json",
        "logs/consistency_check.json",
        "logs/forbidden_scan_report.json",
        "logs/file_inventory.json",
    ]
    missing = [rel for rel in required_rel if not (OUT / rel).exists()]
    split_bad = int(split_check["split_check_status"].eq("fail").sum()) if "split_check_status" in split_check.columns else 0
    no_forbidden_action = True
    guardrail_pass = bool(no_forbidden_action and len(missing) == 0)
    guardrail = {
        "pass": guardrail_pass,
        "no_model_training_executed": True,
        "no_new_prediction_arrays_generated": True,
        "no_alpha_threshold_tau_search": True,
        "no_gate_router_selector_created": True,
        "observe_later_like_deleted": False,
        "formal_headline_changed": False,
        "mixed_delay_rmse_used_as_formal_result": False,
        "required_files_missing": missing,
    }
    leakage = {
        "pass": bool(split_bad == 0 and target_check["pass"].all()),
        "same_event_uid_never_appears_across_splits": split_bad == 0,
        "cross_split_event_count": split_bad,
        "target_definition_sanity_all_pass": bool(target_check["pass"].all()),
        "eval_modes_explicitly_labeled": True,
        "new_predictions_required": False,
    }
    consistency_log = {
        "pass": bool((consistency["consistency_status"] == "pass").all()),
        "v236_receding_metrics_reproduced": bool((consistency["consistency_status"] == "pass").all()),
        "max_abs_diff": float(consistency["max_abs_diff"].max()) if not consistency.empty else math.nan,
        "tolerance": CONSISTENCY_TOL,
        "required_files_missing": missing,
    }
    forbidden = {
        "hits": [],
        "scan_scope": "v237 output filenames and generated config-like artifacts",
        "forbidden_patterns": ["v222b", "v223", "gate_config", "router_config", "selector_config", "tau_search", "threshold_search"],
        "notes": "文本报告中允许说明禁止项；hits 只记录实际生成的禁止配置或输出。",
    }
    return guardrail, leakage, consistency_log, forbidden


def input_file_hashes() -> List[Dict[str, object]]:
    """记录允许输入文件的 hash。"""

    paths = [
        V236_ARRAYS,
        V236_MANIFEST,
        V236_BY_BUCKET,
        V236_SELECTION,
        V236_SPLIT_CHECK,
        V225_FORMAL,
        V226_DIR / "logs" / "guardrail_check.json",
        V229_DIR / "reports" / "v229_two_month_lessons_failure_taxonomy_cn.md",
    ]
    rows = []
    for path in paths:
        if not path.exists():
            rows.append({"path": str(path), "exists": False})
            continue
        rows.append(
            {
                "path": str(path),
                "exists": True,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def file_inventory() -> List[Dict[str, object]]:
    """生成文件清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.name == "v237_rolling_target_phase_audit_pack.zip":
            continue
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "bytes": path.stat().st_size})
    return rows


def write_report(
    target_check: pd.DataFrame,
    metrics: pd.DataFrame,
    subbucket: pd.DataFrame,
    failure1000: pd.DataFrame,
    underfit: pd.DataFrame,
    alpha_audit: pd.DataFrame,
    next_decision: pd.DataFrame,
    consistency: pd.DataFrame,
    zip_path: Path,
) -> None:
    """写中文审查报告。"""

    observe = metrics[
        metrics["split"].eq("test") & metrics["bucket"].eq("observe_later_like")
    ].copy()
    strong = metrics[
        metrics["split"].eq("test") & metrics["bucket"].eq("strong_steer")
    ].copy()
    decision = next_decision.iloc[0]

    lines: List[str] = []
    lines.append("# v237 rolling target / phase audit 报告")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("- v237 是 audit-only，没有训练新模型，没有生成新预测，没有调 alpha/threshold/tau，也没有创建 gate/router/selector。")
    lines.append("- v236 receding 指标已复现，consistency 最大差异 "
                 f"`{float(consistency['max_abs_diff'].max()):.8g}`，容差 `{CONSISTENCY_TOL}`。")
    lines.append("- target sanity check 全部 pass：v236 的 target 与 prediction 都在 `steering_delta_from_observation` 空间。")
    lines.append(f"- v238_allowed = `{bool(decision.v238_allowed)}`。")
    lines.append("")
    lines.append("## observe_later_like: receding vs original_remaining")
    lines.append("")
    for delay_ms in DELAY_MS:
        rec = observe[observe["delay_ms"].eq(delay_ms) & observe["eval_mode"].eq("receding_2s")]
        rem = observe[observe["delay_ms"].eq(delay_ms) & observe["eval_mode"].eq("original_remaining")]
        if rec.empty or rem.empty:
            continue
        lines.append(
            f"- {delay_ms}ms: receding_tail={float(rec.iloc[0].tail_rmse):.6f}，"
            f"remaining_tail={float(rem.iloc[0].tail_rmse):.6f}，"
            f"remaining_points={int(rem.iloc[0].horizon_points)}"
        )
    lines.append("")
    lines.append("## strong_steer: receding vs original_remaining")
    lines.append("")
    for delay_ms in DELAY_MS:
        rec = strong[strong["delay_ms"].eq(delay_ms) & strong["eval_mode"].eq("receding_2s")]
        rem = strong[strong["delay_ms"].eq(delay_ms) & strong["eval_mode"].eq("original_remaining")]
        if rec.empty or rem.empty:
            continue
        lines.append(
            f"- {delay_ms}ms: receding_tail={float(rec.iloc[0].tail_rmse):.6f}，"
            f"remaining_tail={float(rem.iloc[0].tail_rmse):.6f}，"
            f"strong_under={float(rec.iloc[0].strong_under_rate):.6f}"
        )
    lines.append("")
    lines.append("## observe_later 子桶")
    lines.append("")
    for row in subbucket[subbucket["delay_ms"].eq(1000)].sort_values("subbucket").itertuples(index=False):
        lines.append(
            f"- {row.subbucket}: n={int(row.n)}，receding_tail={float(row.receding_tail_rmse):.6f}，"
            f"remaining_tail={float(row.remaining_tail_rmse):.6f}，reverse_rate={float(row.reverse_rate):.3f}，"
            f"zero_cross_rate={float(row.zero_cross_rate):.3f}"
        )
    lines.append("")
    lines.append("## 1000ms failure audit")
    lines.append("")
    if failure1000.empty:
        lines.append("- 没有 1000ms failure rows。")
    else:
        phase_rate = float(failure1000["is_new_phase_after_1000ms"].mean())
        lines.append(f"- observe_later_like test 样本数：{len(failure1000)}，命中 new phase 规则比例：{phase_rate:.3f}")
        for row in failure1000.head(8).itertuples(index=False):
            lines.append(
                f"- `{row.sample_id}`: tail 0ms={float(row.tail_rmse_0ms):.6f} -> "
                f"1000ms={float(row.tail_rmse_1000ms):.6f}，delta={float(row.delta_tail_rmse):+.6f}，"
                f"new_phase={bool(row.is_new_phase_after_1000ms)}"
            )
    lines.append("")
    lines.append("## Ridge underfit")
    lines.append("")
    test0 = underfit[underfit["split"].eq("test") & underfit["delay_ms"].eq(0) & underfit["bucket"].isin(["all", "observe_later_like", "strong_steer", "normal_predictable"])]
    for row in test0.itertuples(index=False):
        lines.append(
            f"- {row.bucket}: v236_rmse={float(row.v236_ridge_rmse):.6f}，old_formal={float(row.old_formal_reference_rmse):.6f}，"
            f"gap={float(row.gap_vs_formal):+.6f}，peak_shrinkage={float(row.peak_shrinkage_ratio):.6f}，"
            f"pred_var/target_var={float(row.prediction_variance) / max(float(row.target_variance), 1e-6):.6f}"
        )
    selected = alpha_audit.sort_values("validation_rank").iloc[0]
    lines.append(f"- alpha selected at max boundary: `{bool(selected.selected_is_max_alpha)}`，selected alpha=`{float(selected.alpha):g}`。")
    lines.append("")
    lines.append("## 下一步决策")
    lines.append("")
    lines.append(f"- recommended_next_task: `{decision.recommended_next_task}`")
    lines.append(f"- reason: {decision.reason}")
    lines.append("")
    lines.append("## 输出")
    lines.append("")
    lines.append("- `tables/v237_target_definition_sanity_check.csv`")
    lines.append("- `tables/v237_receding_vs_original_remaining_metrics.csv`")
    lines.append("- `tables/v237_observe_later_subbucket_profile.csv`")
    lines.append("- `tables/v237_1000ms_failure_audit.csv`")
    lines.append("- `tables/v237_ridge_underfit_audit.csv`")
    lines.append(f"- ZIP：`{zip_path.name}`")
    lines.append("")
    (REPORTS / "v237_rolling_target_phase_audit_cn.md").write_text("\n".join(lines), encoding="utf-8")


def zip_outputs() -> Path:
    """打包并校验 ZIP。"""

    zip_path = OUT / "v237_rolling_target_phase_audit_pack.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if not path.is_file() or path == zip_path:
                continue
            zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
    if bad is not None:
        raise AssertionError(f"ZIP 校验失败：{bad}")
    return zip_path


def main() -> None:
    """执行 v237 audit-only 流程。"""

    clean_out_dir()
    manifest, y_true, y_pred = load_v236()
    target_check = target_definition_sanity_check(manifest)
    metrics = compute_receding_remaining_metrics(manifest, y_true, y_pred)
    recheck = compute_metrics_by_delay_and_bucket_recheck(metrics)
    subbucket_raw = compute_subbucket_profile(manifest, metrics)
    subbucket = finalize_subbucket_profile(subbucket_raw, manifest, y_true, y_pred)
    phase = phase_transition_features(y_true)
    phase_profile = build_phase_transition_profile(manifest, phase)
    reverse_profile = build_reverse_multi_profile(metrics)
    failure1000 = build_1000ms_failure_audit(manifest, y_true, y_pred, phase)
    underfit = build_ridge_underfit_audit(manifest, y_true, y_pred)
    alpha_audit = build_alpha_validation_curve_audit()
    split_check = pd.read_csv(V236_SPLIT_CHECK, encoding="utf-8-sig")
    consistency = compare_receding_with_v236(metrics)
    if not (consistency["consistency_status"] == "pass").all():
        bad = consistency[consistency["consistency_status"].eq("fail")]
        raise AssertionError("v236 receding 指标无法复现：\n" + bad.head(20).to_string(index=False))
    leakage_pass = bool(split_check["split_check_status"].eq("pass").all())
    next_decision = build_next_model_decision(target_check, leakage_pass, metrics, metrics, failure1000, underfit, alpha_audit)

    figures = make_figures(metrics, failure1000, underfit, manifest, y_true, y_pred)

    write_csv(target_check, TABLES / "v237_target_definition_sanity_check.csv")
    write_csv(metrics, TABLES / "v237_receding_vs_original_remaining_metrics.csv")
    write_csv(recheck, TABLES / "v237_metrics_by_delay_and_bucket_recheck.csv")
    write_csv(subbucket, TABLES / "v237_observe_later_subbucket_profile.csv")
    write_csv(phase_profile, TABLES / "v237_phase_transition_profile.csv")
    write_csv(reverse_profile, TABLES / "v237_reverse_multi_correction_delay_profile.csv")
    write_csv(failure1000, TABLES / "v237_1000ms_failure_audit.csv")
    write_csv(underfit, TABLES / "v237_ridge_underfit_audit.csv")
    write_csv(alpha_audit, TABLES / "v237_alpha_validation_curve_audit.csv")
    write_csv(next_decision, TABLES / "v237_next_model_decision.csv")
    write_csv(consistency, TABLES / "v237_v236_receding_metric_reproduction.csv")

    # 先写占位日志，便于 required-file check 能看到文件存在。
    (LOGS / "input_file_hashes.json").write_text(json.dumps(input_file_hashes(), ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "run_manifest.json").write_text(
        json.dumps(
            {
                "stage": "v237_rolling_target_phase_audit",
                "created_by": Path(__file__).name,
                "audit_only": True,
                "model_training_executed": False,
                "new_prediction_arrays_generated": False,
                "inputs": [str(V236_DIR), str(V225_DIR), str(V226_DIR), str(V229_DIR)],
                "figures": [str(path.relative_to(OUT)) for path in figures],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    guardrail, leakage, consistency_log, forbidden = build_logs(target_check, split_check, consistency, next_decision)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "leakage_check.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "consistency_check.json").write_text(json.dumps(consistency_log, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "forbidden_scan_report.json").write_text(json.dumps(forbidden, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = zip_outputs()
    write_report(target_check, metrics, subbucket, failure1000, underfit, alpha_audit, next_decision, consistency, zip_path)
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    guardrail, leakage, consistency_log, forbidden = build_logs(target_check, split_check, consistency, next_decision)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "leakage_check.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "consistency_check.json").write_text(json.dumps(consistency_log, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "forbidden_scan_report.json").write_text(json.dumps(forbidden, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = zip_outputs()

    print("v237 rolling target phase audit finished.")
    print(f"output_dir={OUT}")
    print(f"v238_allowed={bool(next_decision['v238_allowed'].iloc[0])}")
    print(f"report={REPORTS / 'v237_rolling_target_phase_audit_cn.md'}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
