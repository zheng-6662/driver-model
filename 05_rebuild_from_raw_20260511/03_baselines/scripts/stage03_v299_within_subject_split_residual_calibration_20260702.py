#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v299 within-subject split residual calibration.

用户提出的新任务边界：
- 不再按被试完全隔离划分。
- 每个被试自己的样本都拆到 train / val / test。
- 同一个 event_uid 只能出现在一个 split，绝不允许同一样本重复进入训练、验证或测试。
- 样本顺序暂不作为约束。

本轮先做快速可验证实验，而不是直接重训 v241/v249 大模型：
1. 固定 v249 delay0 预测作为 base prediction。
2. 在新的 within-subject split 上，只用 train residual 学习轻量校准器。
3. 用 val 选择校准策略，最后只在 test 报告。

解释边界：
- 这不是完整重训 v249；它测试“同一驾驶员已有样本进入训练集后，是否足以校正现有模型误差”。
- 如果轻量 subject-aware residual 已经有明显收益，再做完整同被试 split 重训才值得。
- 如果这一步仍无本质改善，说明同被试划分本身也不一定解决样本分叉问题。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
OUT = BASELINES / "v299_within_subject_split_residual_calibration_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v299_within_subject_split_residual_calibration_20260702_pack.zip"

V249_NPZ = BASELINES / "v249_shape_aware_curve_model_20260630" / "v249_shape_aware_predictions.npz"
V297_DESC = BASELINES / "v297_subject_style_stability_audit_20260702" / "tables" / "v297_event_response_descriptors.csv"
V298_GUARDRAIL = BASELINES / "v298_event_label_explanatory_audit_20260702" / "logs" / "guardrail_check.json"
THIS_SCRIPT = Path(__file__).resolve()

SEED = 20260702
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
SUBJECT_SHRINKAGE = 10.0
RECORDING_SHRINKAGE = 8.0


def ensure_dirs() -> None:
    """清理并创建本轮输出目录。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    for folder in [TABLES, FIGURES, REPORTS, LOGS]:
        folder.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(obj: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_random_score(text: str, seed: int) -> int:
    """用 hash 做可复现随机排序，避免依赖原始行顺序。"""

    payload = f"{seed}|{text}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def curve_rmse(y_true: np.ndarray, y_pred: np.ndarray, valid: np.ndarray) -> np.ndarray:
    diff = np.where(valid, y_true - y_pred, np.nan)
    mse = np.nanmean(diff * diff, axis=1)
    return np.sqrt(mse)


def sanitize_col(value: object) -> str:
    s = str(value)
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out).strip("_") or "empty"


def curve_features(prefix: str, arr: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    """把 base curve 压成模型特征，同时保留每个时间点。"""

    rows: Dict[str, np.ndarray] = {}
    for j in range(arr.shape[1]):
        rows[f"{prefix}_t{j:02d}"] = arr[:, j]
    rows[f"{prefix}_mean"] = np.nanmean(arr, axis=1)
    rows[f"{prefix}_std"] = np.nanstd(arr, axis=1)
    rows[f"{prefix}_min"] = np.nanmin(arr, axis=1)
    rows[f"{prefix}_max"] = np.nanmax(arr, axis=1)
    rows[f"{prefix}_range"] = rows[f"{prefix}_max"] - rows[f"{prefix}_min"]
    rows[f"{prefix}_final"] = arr[:, -1]
    rows[f"{prefix}_peak_abs"] = np.nanmax(np.abs(arr), axis=1)
    rows[f"{prefix}_line_length"] = np.nansum(np.abs(np.diff(arr, axis=1)), axis=1)
    rows[f"{prefix}_slope"] = (arr[:, -1] - arr[:, 0]) / max(float(grid[-1] - grid[0]), 1e-6)
    return pd.DataFrame(rows)


def load_current_delay0() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取 1167 个当前事件，并与 v249 delay0 曲线对齐。"""

    desc = pd.read_csv(V297_DESC)
    with np.load(V249_NPZ, allow_pickle=False) as z:
        event_uid = z["event_uid"].astype(str)
        delay_ms = z["delay_ms"].astype(int)
        y_true_all = z["y_true_steering_delta"].astype(float)
        pred_all = z["pred_v249_best_shape_steering_delta"].astype(float)
        valid_all = z["original_remaining_valid"].astype(bool)
        grid = z["future_grid_s"].astype(float)
    idx = np.where(delay_ms == 0)[0]
    row_map = {event_uid[i]: i for i in idx}
    missing = [e for e in desc["event_uid"].astype(str) if e not in row_map]
    if missing:
        raise RuntimeError(f"v249 delay0 missing events: {len(missing)}")
    aligned = np.array([row_map[e] for e in desc["event_uid"].astype(str)], dtype=int)
    y_true = y_true_all[aligned]
    pred = pred_all[aligned]
    valid = valid_all[aligned]
    desc = desc.reset_index(drop=True)
    for col in ["bad_top10", "vehicle_ambiguous", "bad_top10_vehicle_ambiguous", "candidate_pool_gain_gt_005"]:
        if col in desc.columns:
            desc[col] = pd.to_numeric(desc[col], errors="coerce").fillna(0).astype(int)
    desc["v249_rmse_recalc"] = curve_rmse(y_true, pred, valid)
    return desc, y_true, pred, valid, grid


def make_within_subject_split(data: pd.DataFrame) -> pd.DataFrame:
    """每个 subject 内独立随机切分，保证同一 event_uid 只出现一次。"""

    rows: List[Dict[str, object]] = []
    for subject, sub in data.groupby("subject", sort=True):
        work = sub[["event_uid", "subject", "recording", "observation_s"]].copy()
        work["_score"] = work["event_uid"].astype(str).map(lambda x: stable_random_score(x, SEED))
        work = work.sort_values(["_score", "event_uid"]).reset_index(drop=True)
        n = len(work)
        n_train = max(1, int(round(n * TRAIN_RATIO)))
        n_val = max(1, int(round(n * VAL_RATIO)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            n_train = max(1, n - n_val - n_test)
        split = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test, dtype=object)
        for i, (_, row) in enumerate(work.iterrows()):
            rows.append(
                {
                    "event_uid": row["event_uid"],
                    "subject": row["subject"],
                    "recording": row["recording"],
                    "observation_s": row["observation_s"],
                    "within_subject_split": split[i],
                    "within_subject_order": int(i),
                    "subject_event_n": int(n),
                }
            )
    split_df = pd.DataFrame(rows)
    if split_df["event_uid"].duplicated().any():
        dup = split_df.loc[split_df["event_uid"].duplicated(), "event_uid"].head(5).tolist()
        raise RuntimeError(f"duplicated event in split: {dup}")
    return split_df


def build_feature_blocks(data: pd.DataFrame, pred: np.ndarray, grid: np.ndarray) -> Dict[str, pd.DataFrame]:
    train = data["within_subject_split"].eq("train")
    base = curve_features("base", pred, grid)
    meta = pd.DataFrame(
        {
            "observation_s": pd.to_numeric(data["observation_s"], errors="coerce"),
            "event_index_in_uid": pd.to_numeric(data["event_index_in_uid"], errors="coerce"),
            "order_observation_s": pd.to_numeric(data["order_observation_s"], errors="coerce"),
            "base_rmse_proxy_peak_abs": np.nanmax(np.abs(pred), axis=1),
            "base_final": pred[:, -1],
        }
    )
    subjects = data["subject"].fillna("unknown").astype(str)
    train_subjects = sorted(subjects[train].unique().tolist())
    subject_frame = pd.DataFrame({f"subject_{sanitize_col(s)}": subjects.eq(s).astype(float) for s in train_subjects})
    subject_frame["subject_unseen"] = (~subjects.isin(train_subjects)).astype(float)

    recordings = data["recording"].fillna("unknown").astype(str)
    train_recordings = sorted(recordings[train].unique().tolist())
    recording_frame = pd.DataFrame({f"recording_{sanitize_col(s)}": recordings.eq(s).astype(float) for s in train_recordings})
    recording_frame["recording_unseen"] = (~recordings.isin(train_recordings)).astype(float)

    blocks = {
        "base_curve_only": base,
        "base_curve_plus_meta": pd.concat([base, meta], axis=1),
        "subject_onehot_only": subject_frame,
        "base_curve_meta_subject": pd.concat([base, meta, subject_frame], axis=1),
        # recording 是 session 级强诊断，可能包含场次泄漏；只作为上限/风险提示，不作为首选部署策略。
        "base_curve_meta_subject_recording_diagnostic": pd.concat([base, meta, subject_frame, recording_frame], axis=1),
    }
    return {k: v.loc[:, ~v.columns.duplicated()].copy() for k, v in blocks.items()}


def mean_residual_prediction(data: pd.DataFrame, residual: np.ndarray, group_col: str | None, shrinkage: float) -> np.ndarray:
    """基于 train 的全局/分组 residual 均值，为所有样本生成校正量。"""

    train = data["within_subject_split"].eq("train").to_numpy()
    global_mean = np.nanmean(residual[train], axis=0)
    out = np.tile(global_mean[None, :], (len(data), 1))
    if group_col is None:
        return out
    group_values = data[group_col].fillna("unknown").astype(str)
    stats: Dict[str, Tuple[int, np.ndarray]] = {}
    for group, idx in data.loc[train].groupby(group_col).groups.items():
        mask = np.zeros(len(data), dtype=bool)
        mask[list(idx)] = True
        stats[str(group)] = (int(mask.sum()), np.nanmean(residual[mask], axis=0))
    for i, group in enumerate(group_values):
        if group not in stats:
            continue
        n, mean = stats[group]
        w = float(n / (n + shrinkage))
        out[i] = w * mean + (1.0 - w) * global_mean
    return out


def fit_residual_models(data: pd.DataFrame, y_true: np.ndarray, pred: np.ndarray, valid: np.ndarray, grid: np.ndarray) -> Dict[str, np.ndarray]:
    """训练所有轻量 residual 校准器，并返回每个方法的校正后预测曲线。"""

    residual = np.where(valid, y_true - pred, np.nan)
    y_res = np.nan_to_num(residual, nan=0.0)
    train = data["within_subject_split"].eq("train").to_numpy()
    blocks = build_feature_blocks(data, pred, grid)
    outputs: Dict[str, np.ndarray] = {}
    outputs["v249_no_correction"] = pred.copy()

    for name, corr in [
        ("global_train_mean_residual", mean_residual_prediction(data, residual, None, SUBJECT_SHRINKAGE)),
        ("subject_train_mean_residual", mean_residual_prediction(data, residual, "subject", SUBJECT_SHRINKAGE)),
        (
            "recording_train_mean_residual_diagnostic",
            mean_residual_prediction(data, residual, "recording", RECORDING_SHRINKAGE),
        ),
    ]:
        outputs[name] = pred + corr

    ridge_models = {
        "ridge_a1": Ridge(alpha=1.0),
        "ridge_a10": Ridge(alpha=10.0),
        "ridge_a100": Ridge(alpha=100.0),
    }
    for block_name, x in blocks.items():
        for model_name, model in ridge_models.items():
            pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
            pipe.fit(x.loc[train], y_res[train])
            outputs[f"{block_name}__{model_name}"] = pred + np.asarray(pipe.predict(x), dtype=float)

    for block_name in ["base_curve_plus_meta", "base_curve_meta_subject"]:
        x = blocks[block_name]
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=8,
                random_state=SEED,
                n_jobs=-1,
            ),
        )
        model.fit(x.loc[train], y_res[train])
        outputs[f"{block_name}__extra_trees_d5"] = pred + np.asarray(model.predict(x), dtype=float)
    return outputs


def add_within_bad_flags(data: pd.DataFrame) -> pd.DataFrame:
    """在 within split 内重新定义 test/val bad_top10，避免只依赖旧 split 的坏样本标记。"""

    out = data.copy()
    out["within_bad_top10_by_v249"] = 0
    out["within_bad_top20_by_v249"] = 0
    for split_name, sub in out.groupby("within_subject_split"):
        if sub.empty:
            continue
        q90 = float(sub["v249_rmse_recalc"].quantile(0.90))
        q80 = float(sub["v249_rmse_recalc"].quantile(0.80))
        idx = sub.index
        out.loc[idx, "within_bad_top10_by_v249"] = (sub["v249_rmse_recalc"] >= q90).astype(int).to_numpy()
        out.loc[idx, "within_bad_top20_by_v249"] = (sub["v249_rmse_recalc"] >= q80).astype(int).to_numpy()
    return out


def evaluate_predictions(data: pd.DataFrame, y_true: np.ndarray, preds: Dict[str, np.ndarray], valid: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """对所有方法输出 split/group summary 和 event-level delta。"""

    base_rmse = curve_rmse(y_true, preds["v249_no_correction"], valid)
    event_rows = []
    summary_rows = []
    for method, curve in preds.items():
        rmse = curve_rmse(y_true, curve, valid)
        delta = rmse - base_rmse
        for i, row in data.iterrows():
            event_rows.append(
                {
                    "event_uid": row["event_uid"],
                    "subject": row["subject"],
                    "recording": row["recording"],
                    "within_subject_split": row["within_subject_split"],
                    "original_v249_split": row["split"],
                    "method": method,
                    "baseline_rmse": float(base_rmse[i]),
                    "method_rmse": float(rmse[i]),
                    "delta_vs_v249": float(delta[i]),
                    "bad_top10_original": int(row.get("bad_top10", 0)),
                    "bad_top10_vehicle_ambiguous": int(row.get("bad_top10_vehicle_ambiguous", 0)),
                    "within_bad_top10_by_v249": int(row["within_bad_top10_by_v249"]),
                    "within_bad_top20_by_v249": int(row["within_bad_top20_by_v249"]),
                }
            )
        for split_name in ["train", "val", "test"]:
            split_mask = data["within_subject_split"].eq(split_name).to_numpy()
            groups = [
                ("all", np.ones(len(data), dtype=bool)),
                ("within_bad_top10", data["within_bad_top10_by_v249"].to_numpy(dtype=bool)),
                ("within_bad_top20", data["within_bad_top20_by_v249"].to_numpy(dtype=bool)),
                ("original_bad_top10", data["bad_top10"].to_numpy(dtype=bool)),
                ("bad_top10_vehicle_ambiguous", data["bad_top10_vehicle_ambiguous"].to_numpy(dtype=bool)),
            ]
            for group_name, group_mask in groups:
                mask = split_mask & group_mask
                if int(mask.sum()) == 0:
                    continue
                summary_rows.append(
                    {
                        "method": method,
                        "split": split_name,
                        "group": group_name,
                        "n": int(mask.sum()),
                        "baseline_rmse_mean": float(np.nanmean(base_rmse[mask])),
                        "method_rmse_mean": float(np.nanmean(rmse[mask])),
                        "delta_vs_v249_mean": float(np.nanmean(delta[mask])),
                        "delta_vs_v249_median": float(np.nanmedian(delta[mask])),
                        "improved_rate": float(np.mean(delta[mask] < 0)),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(event_rows)


def chosen_original_split_summary(event_delta: pd.DataFrame, chosen_method: str) -> pd.DataFrame:
    """查看 chosen 方法在新 test 内不同旧 v249 split 子集上的表现。"""

    sub = event_delta[
        event_delta["method"].eq(chosen_method)
        & event_delta["within_subject_split"].eq("test")
    ].copy()
    rows = []
    for old_split, g in sub.groupby("original_v249_split", dropna=False):
        rows.append(
            {
                "original_v249_split": old_split,
                "group": "all",
                "n": int(len(g)),
                "delta_vs_v249_mean": float(g["delta_vs_v249"].mean()),
                "delta_vs_v249_median": float(g["delta_vs_v249"].median()),
                "improved_rate": float(g["delta_vs_v249"].lt(0).mean()),
            }
        )
        bad = g[g["within_bad_top10_by_v249"].eq(1)]
        if not bad.empty:
            rows.append(
                {
                    "original_v249_split": old_split,
                    "group": "within_bad_top10",
                    "n": int(len(bad)),
                    "delta_vs_v249_mean": float(bad["delta_vs_v249"].mean()),
                    "delta_vs_v249_median": float(bad["delta_vs_v249"].median()),
                    "improved_rate": float(bad["delta_vs_v249"].lt(0).mean()),
                }
            )
    nontrain = sub[~sub["original_v249_split"].eq("train")]
    if not nontrain.empty:
        rows.append(
            {
                "original_v249_split": "nontrain_combined",
                "group": "all",
                "n": int(len(nontrain)),
                "delta_vs_v249_mean": float(nontrain["delta_vs_v249"].mean()),
                "delta_vs_v249_median": float(nontrain["delta_vs_v249"].median()),
                "improved_rate": float(nontrain["delta_vs_v249"].lt(0).mean()),
            }
        )
        bad = nontrain[nontrain["within_bad_top10_by_v249"].eq(1)]
        if not bad.empty:
            rows.append(
                {
                    "original_v249_split": "nontrain_combined",
                    "group": "within_bad_top10",
                    "n": int(len(bad)),
                    "delta_vs_v249_mean": float(bad["delta_vs_v249"].mean()),
                    "delta_vs_v249_median": float(bad["delta_vs_v249"].median()),
                    "improved_rate": float(bad["delta_vs_v249"].lt(0).mean()),
                }
            )
    return pd.DataFrame(rows)


def choose_by_val(summary: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """只用 val 选择策略，然后报告 test。"""

    val_all = summary[(summary["split"].eq("val")) & (summary["group"].eq("all"))][
        ["method", "delta_vs_v249_mean"]
    ].rename(columns={"delta_vs_v249_mean": "val_all_delta"})
    val_bad = summary[(summary["split"].eq("val")) & (summary["group"].eq("within_bad_top10"))][
        ["method", "delta_vs_v249_mean"]
    ].rename(columns={"delta_vs_v249_mean": "val_within_bad_top10_delta"})
    test_all = summary[(summary["split"].eq("test")) & (summary["group"].eq("all"))][
        ["method", "delta_vs_v249_mean", "method_rmse_mean", "baseline_rmse_mean"]
    ].rename(
        columns={
            "delta_vs_v249_mean": "test_all_delta",
            "method_rmse_mean": "test_all_rmse",
            "baseline_rmse_mean": "test_all_baseline_rmse",
        }
    )
    test_bad = summary[(summary["split"].eq("test")) & (summary["group"].eq("within_bad_top10"))][
        ["method", "delta_vs_v249_mean", "method_rmse_mean", "baseline_rmse_mean"]
    ].rename(
        columns={
            "delta_vs_v249_mean": "test_within_bad_top10_delta",
            "method_rmse_mean": "test_within_bad_top10_rmse",
            "baseline_rmse_mean": "test_within_bad_top10_baseline_rmse",
        }
    )
    table = val_all.merge(val_bad, on="method", how="outer").merge(test_all, on="method", how="outer").merge(test_bad, on="method", how="outer")
    table["is_recording_diagnostic"] = table["method"].astype(str).str.contains("recording", regex=False)
    table["uses_subject"] = table["method"].astype(str).str.contains("subject", regex=False)
    table["val_score"] = table["val_within_bad_top10_delta"].fillna(0) + 4.0 * table["val_all_delta"].clip(lower=0).fillna(0)
    deployable = table[
        (~table["method"].eq("v249_no_correction"))
        & (~table["is_recording_diagnostic"])
        & table["val_all_delta"].le(0.003)
        & table["val_within_bad_top10_delta"].lt(0)
    ].copy()
    if deployable.empty:
        chosen = table[~table["method"].eq("v249_no_correction")].sort_values("val_score", ascending=True).head(1).copy()
        choice_rule = "best validation score, no strict no-harm candidate"
    else:
        chosen = deployable.sort_values(["val_within_bad_top10_delta", "val_all_delta"], ascending=True).head(1).copy()
        choice_rule = "val no-harm among non-recording methods"

    test_best = table[~table["method"].eq("v249_no_correction")].sort_values("test_within_bad_top10_delta", ascending=True).head(1).copy()
    recording_best = table[table["is_recording_diagnostic"]].sort_values("test_within_bad_top10_delta", ascending=True).head(1).copy()
    rows = []
    if not chosen.empty:
        row = chosen.iloc[0].to_dict()
        row["choice_name"] = "chosen_by_val"
        row["choice_rule"] = choice_rule
        rows.append(row)
    if not test_best.empty:
        row = test_best.iloc[0].to_dict()
        row["choice_name"] = "test_best_diagnostic"
        row["choice_rule"] = "test diagnostic only, not selectable"
        rows.append(row)
    if not recording_best.empty:
        row = recording_best.iloc[0].to_dict()
        row["choice_name"] = "recording_best_diagnostic"
        row["choice_rule"] = "session/recording diagnostic only"
        rows.append(row)
    chosen_table = pd.DataFrame(rows)

    chosen_row = chosen_table[chosen_table["choice_name"].eq("chosen_by_val")].iloc[0]
    test_best_row = chosen_table[chosen_table["choice_name"].eq("test_best_diagnostic")].iloc[0]
    guard = {
        "chosen_method": str(chosen_row["method"]),
        "chosen_test_all_delta": float(chosen_row["test_all_delta"]),
        "chosen_test_within_bad_top10_delta": float(chosen_row["test_within_bad_top10_delta"]),
        "chosen_test_within_bad_top10_rmse": float(chosen_row["test_within_bad_top10_rmse"]),
        "chosen_test_within_bad_top10_baseline_rmse": float(chosen_row["test_within_bad_top10_baseline_rmse"]),
        "test_best_method": str(test_best_row["method"]),
        "test_best_within_bad_top10_delta": float(test_best_row["test_within_bad_top10_delta"]),
        "test_best_all_delta": float(test_best_row["test_all_delta"]),
    }
    return chosen_table, guard


def split_audit(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows = []
    for subject, sub in data.groupby("subject"):
        counts = sub["within_subject_split"].value_counts().to_dict()
        rows.append(
            {
                "subject": subject,
                "event_n": int(len(sub)),
                "train_n": int(counts.get("train", 0)),
                "val_n": int(counts.get("val", 0)),
                "test_n": int(counts.get("test", 0)),
                "recording_n": int(sub["recording"].nunique()),
            }
        )
    table = pd.DataFrame(rows).sort_values("event_n", ascending=False)
    event_split_n = data.groupby("event_uid")["within_subject_split"].nunique()
    subject_split = data.groupby("subject")["within_subject_split"].nunique()
    split_cross = pd.crosstab(data["within_subject_split"], data["split"])
    test_rows = data["within_subject_split"].eq("test")
    within_test_original_train_rate = float(data.loc[test_rows, "split"].eq("train").mean()) if int(test_rows.sum()) else math.nan
    guard = {
        "event_n": int(len(data)),
        "unique_event_n": int(data["event_uid"].nunique()),
        "duplicate_event_uid_n": int(data["event_uid"].duplicated().sum()),
        "event_in_multiple_splits_n": int((event_split_n > 1).sum()),
        "subject_n": int(data["subject"].nunique()),
        "subject_with_all_three_splits_n": int((subject_split == 3).sum()),
        "train_n": int(data["within_subject_split"].eq("train").sum()),
        "val_n": int(data["within_subject_split"].eq("val").sum()),
        "test_n": int(data["within_subject_split"].eq("test").sum()),
        "within_test_original_v249_train_n": int((test_rows & data["split"].eq("train")).sum()),
        "within_test_original_v249_train_rate": within_test_original_train_rate,
        "fixed_v249_predictions_have_original_split_exposure": bool(within_test_original_train_rate > 0),
    }
    return table, guard


def original_split_crosstab(data: pd.DataFrame) -> pd.DataFrame:
    """记录新的 within split 和旧 v249 split 的交叉关系。"""

    rows = []
    for within_split, sub in data.groupby("within_subject_split"):
        denom = max(len(sub), 1)
        for old_split, n in sub["split"].value_counts().items():
            rows.append(
                {
                    "within_subject_split": within_split,
                    "original_v249_split": old_split,
                    "n": int(n),
                    "rate_within_split": float(n / denom),
                }
            )
    return pd.DataFrame(rows).sort_values(["within_subject_split", "original_v249_split"])


def plot_split_counts(split_table: pd.DataFrame) -> Path:
    path = FIGURES / "v299_within_subject_split_counts.png"
    data = split_table.sort_values("event_n", ascending=True)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9, 6))
    left = np.zeros(len(data))
    for col, color in [("train_n", "#4E79A7"), ("val_n", "#F28E2B"), ("test_n", "#E15759")]:
        ax.barh(y, data[col], left=left, label=col.replace("_n", ""), color=color)
        left += data[col].to_numpy()
    ax.set_yticks(y)
    ax.set_yticklabels(data["subject"])
    ax.set_xlabel("event count")
    ax.set_title("v299 within-subject random split counts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_test_delta(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v299_test_delta_by_method.png"
    data = summary[
        summary["split"].eq("test")
        & summary["group"].eq("within_bad_top10")
        & ~summary["method"].eq("v249_no_correction")
    ].copy()
    data = data.sort_values("delta_vs_v249_mean", ascending=True).head(18)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#E15759" if x < 0 else "#BAB0AC" for x in data["delta_vs_v249_mean"]]
    ax.barh(data["method"], data["delta_vs_v249_mean"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(-0.05, color="tab:blue", linestyle="--", linewidth=1, label="-0.05 useful threshold")
    ax.set_xlabel("test within_bad_top10 RMSE delta vs v249")
    ax.set_title("v299 within-subject split: test bad sample residual correction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_bad_curves(data: pd.DataFrame, y_true: np.ndarray, pred_base: np.ndarray, preds: Dict[str, np.ndarray], grid: np.ndarray, chosen_method: str) -> Path:
    path = FIGURES / "v299_test_bad_top6_curves.png"
    test = data[data["within_subject_split"].eq("test") & data["within_bad_top10_by_v249"].eq(1)].copy()
    test = test.sort_values("v249_rmse_recalc", ascending=False).head(6)
    if test.empty or chosen_method not in preds:
        return path
    fig, axes = plt.subplots(len(test), 1, figsize=(10, 2.2 * len(test)), sharex=True)
    if len(test) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, test.iterrows()):
        i = int(row["_row_id"])
        ax.plot(grid, y_true[i], color="black", linewidth=1.8, label="true")
        ax.plot(grid, pred_base[i], color="#00A087", linestyle="--", linewidth=1.4, label="v249")
        ax.plot(grid, preds[chosen_method][i], color="#E64B35", linestyle="-.", linewidth=1.4, label=chosen_method)
        ax.set_ylabel("steer delta")
        ax.set_title(f"{row['event_uid']} | subject={row['subject']} | base RMSE={row['v249_rmse_recalc']:.3f}", fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("future time / s")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def markdown_table(df: pd.DataFrame, cols: Sequence[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_empty_"
    cols = [c for c in cols if c in df.columns]
    view = df.loc[:, cols].head(max_rows).copy()
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    return view.to_markdown(index=False)


def write_report(
    split_table: pd.DataFrame,
    summary: pd.DataFrame,
    chosen: pd.DataFrame,
    chosen_old_split_summary: pd.DataFrame,
    guardrail: Dict[str, object],
) -> Path:
    lines: List[str] = []
    lines.append("# v299 within-subject split residual calibration")
    lines.append("")
    lines.append("## 结论")
    if guardrail["within_subject_residual_route_promising"]:
        lines.append("- 在同被试内切分后，轻量 residual 校准已经显示明显收益，值得进入完整模型重训。")
    else:
        lines.append("- 在同被试内切分后，轻量 residual 校准仍未达到本质改善标准；仅改变 split 不能自动解决轨迹分叉问题。")
    lines.append(
        f"- 重要边界：本轮固定使用旧 v249 预测，新的 within-test 中有 "
        f"{guardrail['within_test_original_v249_train_rate']:.3f} 原本属于旧 v249 train split，"
        "所以本轮是快速潜力审计，不是正式重训结论。"
    )
    lines.append(
        f"- val 选择方法 `{guardrail['chosen_method']}`：test all delta={guardrail['chosen_test_all_delta']:.6f}, "
        f"test within_bad_top10 delta={guardrail['chosen_test_within_bad_top10_delta']:.6f}。"
    )
    lines.append(
        f"- test-best diagnostic `{guardrail['test_best_method']}`：test within_bad_top10 delta="
        f"{guardrail['test_best_within_bad_top10_delta']:.6f}，该值只作诊断，不作为可部署选择。"
    )
    lines.append("")
    lines.append("## split guardrail")
    lines.append("```json")
    split_keys = [
        "event_n",
        "unique_event_n",
        "duplicate_event_uid_n",
        "event_in_multiple_splits_n",
        "subject_n",
        "subject_with_all_three_splits_n",
        "train_n",
        "val_n",
        "test_n",
        "within_test_original_v249_train_n",
        "within_test_original_v249_train_rate",
        "fixed_v249_predictions_have_original_split_exposure",
    ]
    lines.append(json.dumps({k: guardrail[k] for k in split_keys}, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## subject split counts")
    lines.append(markdown_table(split_table, ["subject", "event_n", "train_n", "val_n", "test_n", "recording_n"], 30))
    lines.append("")
    lines.append("## chosen by val")
    lines.append(
        markdown_table(
            chosen,
            [
                "choice_name",
                "method",
                "choice_rule",
                "val_all_delta",
                "val_within_bad_top10_delta",
                "test_all_delta",
                "test_within_bad_top10_delta",
                "test_within_bad_top10_rmse",
            ],
            10,
        )
    )
    lines.append("")
    lines.append("## test summary top methods")
    test = summary[summary["split"].eq("test") & summary["group"].eq("within_bad_top10")].sort_values("delta_vs_v249_mean")
    lines.append(
        markdown_table(
            test,
            [
                "method",
                "n",
                "baseline_rmse_mean",
                "method_rmse_mean",
                "delta_vs_v249_mean",
                "delta_vs_v249_median",
                "improved_rate",
            ],
            30,
        )
    )
    lines.append("")
    lines.append("## chosen method by original v249 split")
    lines.append(
        markdown_table(
            chosen_old_split_summary,
            [
                "original_v249_split",
                "group",
                "n",
                "delta_vs_v249_mean",
                "delta_vs_v249_median",
                "improved_rate",
            ],
            20,
        )
    )
    lines.append("")
    lines.append("## guardrail")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    path = REPORTS / "v299_within_subject_split_residual_calibration_cn.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(OUT.parent))
        zf.write(THIS_SCRIPT, Path("scripts") / THIS_SCRIPT.name)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"zip test failed at {bad}")


def main() -> None:
    ensure_dirs()
    np.random.seed(SEED)
    print("[v299] 计划：同一被试内随机切分 train/val/test，并验证轻量 subject-aware residual 校准。", flush=True)
    input_hashes = pd.DataFrame(
        [
            {"path": str(V249_NPZ), "sha256": file_sha256(V249_NPZ), "role": "v249 baseline prediction curves"},
            {"path": str(V297_DESC), "sha256": file_sha256(V297_DESC), "role": "event descriptors and subject ids"},
            {"path": str(V298_GUARDRAIL), "sha256": file_sha256(V298_GUARDRAIL), "role": "previous label audit guardrail"},
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    data, y_true, pred, valid, grid = load_current_delay0()
    data["_row_id"] = np.arange(len(data), dtype=int)
    split_df = make_within_subject_split(data)
    data = data.merge(split_df[["event_uid", "within_subject_split", "within_subject_order", "subject_event_n"]], on="event_uid", how="left", validate="one_to_one")
    if data["within_subject_split"].isna().any():
        raise RuntimeError("within-subject split missing rows")
    data = add_within_bad_flags(data)
    split_table, split_guard = split_audit(data)

    print("[v299] fit residual calibration models", flush=True)
    preds = fit_residual_models(data, y_true, pred, valid, grid)
    summary, event_delta = evaluate_predictions(data, y_true, preds, valid)
    chosen, choice_guard = choose_by_val(summary)
    chosen_method = str(choice_guard["chosen_method"])
    chosen_old_split_summary = chosen_original_split_summary(event_delta, chosen_method)

    guardrail: Dict[str, object] = {
        "pass": True,
        "split_method": "within_subject_random_event_split_60_20_20",
        "seed": SEED,
        "same_event_never_repeated_across_splits": bool(split_guard["event_in_multiple_splits_n"] == 0 and split_guard["duplicate_event_uid_n"] == 0),
        "uses_original_subject_disjoint_split_for_training": False,
        "full_v249_retrained": False,
        "experiment_scope": "fast residual calibration on fixed v249 predictions",
        "formal_claim_requires_full_retrain_on_within_subject_split": True,
        **split_guard,
        **choice_guard,
    }
    nontrain_bad = chosen_old_split_summary[
        chosen_old_split_summary["original_v249_split"].eq("nontrain_combined")
        & chosen_old_split_summary["group"].eq("within_bad_top10")
    ]
    if not nontrain_bad.empty:
        guardrail["chosen_test_original_nontrain_within_bad_top10_delta"] = float(nontrain_bad["delta_vs_v249_mean"].iloc[0])
        guardrail["chosen_test_original_nontrain_within_bad_top10_n"] = int(nontrain_bad["n"].iloc[0])
    guardrail["within_subject_residual_route_promising"] = bool(
        guardrail["chosen_test_all_delta"] <= 0.005 and guardrail["chosen_test_within_bad_top10_delta"] <= -0.05
    )
    guardrail["complete_model_retrain_recommended_next"] = bool(guardrail["within_subject_residual_route_promising"])
    guardrail["goal_achieved_now"] = False

    write_csv(data, TABLES / "v299_within_subject_split_event_table.csv")
    write_csv(split_table, TABLES / "v299_within_subject_split_subject_counts.csv")
    write_csv(original_split_crosstab(data), TABLES / "v299_within_vs_original_split_crosstab.csv")
    write_csv(summary, TABLES / "v299_within_subject_residual_summary.csv")
    write_csv(event_delta, TABLES / "v299_within_subject_residual_event_deltas.csv")
    write_csv(chosen, TABLES / "v299_chosen_by_val.csv")
    write_csv(chosen_old_split_summary, TABLES / "v299_chosen_test_delta_by_original_split.csv")
    write_json(guardrail, LOGS / "guardrail_check.json")

    plot_split_counts(split_table)
    plot_test_delta(summary)
    plot_bad_curves(data, y_true, pred, preds, grid, chosen_method)
    write_report(split_table, summary, chosen, chosen_old_split_summary, guardrail)

    inventory = [{"path": str(p), "bytes": int(p.stat().st_size)} for p in sorted(OUT.rglob("*")) if p.is_file()]
    write_csv(pd.DataFrame(inventory), LOGS / "file_inventory.csv")
    make_zip()
    guardrail["zip_testzip"] = True
    write_json(guardrail, LOGS / "guardrail_check.json")
    print("[v299] done", flush=True)
    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
