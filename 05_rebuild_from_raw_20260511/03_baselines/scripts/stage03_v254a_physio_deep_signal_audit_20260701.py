#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v254a 生理信号深层挖掘审计。

本轮目标不是把生理特征简单拼接进轨迹回归模型，而是先回答三件事：
1. 当前生理数据在事件锚点前是否覆盖充分、对齐稳定；
2. 生理特征本身是否含有驾驶员/记录/状态结构；
3. 在固定车辆输入之外，生理特征是否能预测未来行为模式或未来强度。

边界：
- 不训练新轨迹预测模型；
- 不用 test 结果调参或选择特征；
- 未来轨迹只用于构造诊断标签/评价指标，不进入特征；
- 生理窗口全部截止在 observation_s 之前或等于 observation_s。
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
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V252_SCRIPT = BASELINES / "scripts" / "stage03_v252_input_similarity_future_divergence_20260701.py"
PHYSIO_DIR = REBUILD / "06_physio_processing" / "physio_subject_collection_v1_20260603" / "tables"
PHYSIO_1HZ = PHYSIO_DIR / "physio_features_1hz.csv"
PHYSIO_10HZ = PHYSIO_DIR / "physio_features_10hz.csv"
PHYSIO_INVENTORY = PHYSIO_DIR / "physio_recording_inventory.csv"
V253A_PHYSIO_FEATURES = (
    BASELINES
    / "v253_state_signal_disambiguation_audit_20260701"
    / "tables"
    / "v253a_current_physio_features_1hz.csv"
)

OUT = BASELINES / "v254a_physio_deep_signal_audit_20260701"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v254a_physio_deep_signal_audit_20260701_pack.zip"

SEED = 254

# 10Hz 表来自清洗后 200Hz 连续层聚合，比 v253a 的 1Hz 窗口更细。
PHYSIO_SIGNALS = [
    "ECG_raw200",
    "EMG_raw200",
    "EDA_raw200",
    "RESP_raw200",
    "ECG_filt200",
    "EMG_filt200",
    "EDA_filt200",
    "RESP_filt200",
    "HR_bpm",
    "HRV_RMSSD",
    "EDA_Tonic",
    "EDA_Phasic",
    "EMG_RMS",
    "RESP_BPM",
    "RESP_Amplitude",
]

WINDOWS = {
    "pre20_pre10": (-20.0, -10.0),
    "pre10_pre5": (-10.0, -5.0),
    "pre5_pre2": (-5.0, -2.0),
    "pre2_0": (-2.0, 0.0),
}

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    """按路径导入前序脚本，复用 v252 的固定样本和输入。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V252 = import_module_from_path("stage03_v252_for_v254a", V252_SCRIPT)


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v254a 自己的输出。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 统一使用 utf-8-sig，便于中文 Windows 环境打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算文件 SHA256，便于之后回溯输入。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def session_stamp_from_recording(recording: str) -> str:
    """从 Entity_Recording_xxx 中取 session stamp。"""

    return str(recording).replace("Entity_Recording_", "")


def finite_array(values: Iterable[object]) -> np.ndarray:
    """转成有限浮点数组。"""

    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def slope_feature(times: np.ndarray, values: np.ndarray) -> float:
    """用首末有限点估计窗口斜率，避免短窗口线性拟合不稳。"""

    mask = np.isfinite(times) & np.isfinite(values)
    if int(mask.sum()) < 2:
        return math.nan
    t = times[mask]
    v = values[mask]
    dt = float(t[-1] - t[0])
    if abs(dt) < 1e-9:
        return math.nan
    return float((v[-1] - v[0]) / dt)


def window_stats(times: np.ndarray, values: np.ndarray, prefix: str) -> Dict[str, float]:
    """计算一个锚点前窗口的稳健统计。"""

    vals = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    out: Dict[str, float] = {
        f"{prefix}_valid_ratio": float(len(finite) / max(1, len(vals))),
        f"{prefix}_mean": math.nan,
        f"{prefix}_std": math.nan,
        f"{prefix}_p10": math.nan,
        f"{prefix}_p50": math.nan,
        f"{prefix}_p90": math.nan,
        f"{prefix}_range": math.nan,
        f"{prefix}_abs_mean": math.nan,
        f"{prefix}_rms": math.nan,
        f"{prefix}_first": math.nan,
        f"{prefix}_last": math.nan,
        f"{prefix}_last_minus_first": math.nan,
        f"{prefix}_slope": math.nan,
    }
    if len(finite) == 0:
        return out
    out.update(
        {
            f"{prefix}_mean": float(np.mean(finite)),
            f"{prefix}_std": float(np.std(finite)),
            f"{prefix}_p10": float(np.quantile(finite, 0.10)),
            f"{prefix}_p50": float(np.quantile(finite, 0.50)),
            f"{prefix}_p90": float(np.quantile(finite, 0.90)),
            f"{prefix}_range": float(np.max(finite) - np.min(finite)),
            f"{prefix}_abs_mean": float(np.mean(np.abs(finite))),
            f"{prefix}_rms": float(np.sqrt(np.mean(np.square(finite)))),
        }
    )
    mask = np.isfinite(times) & np.isfinite(vals)
    if int(mask.sum()) >= 1:
        finite_vals_in_time = vals[mask]
        out[f"{prefix}_first"] = float(finite_vals_in_time[0])
        out[f"{prefix}_last"] = float(finite_vals_in_time[-1])
        out[f"{prefix}_last_minus_first"] = float(finite_vals_in_time[-1] - finite_vals_in_time[0])
    out[f"{prefix}_slope"] = slope_feature(times, vals)
    return out


def load_physio_10hz_groups() -> Dict[Tuple[str, str], pd.DataFrame]:
    """读取 10Hz 生理特征并按 subject/session_stamp 分组。"""

    usecols = ["subject", "session_stamp", "time_bin_s"] + PHYSIO_SIGNALS
    df = pd.read_csv(PHYSIO_10HZ, usecols=lambda c: c in set(usecols), encoding="utf-8-sig")
    df["time_bin_s"] = pd.to_numeric(df["time_bin_s"], errors="coerce")
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for key, g in df.groupby(["subject", "session_stamp"], dropna=False):
        out[(str(key[0]), str(key[1]))] = g.sort_values("time_bin_s").reset_index(drop=True)
    return out


def build_event_physio_10hz_features(manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    为当前 rolling sample 提取 10Hz 锚点前生理特征。

    这里故意保留多个时间窗口和窗口间差值，因为用户关心的是驾驶员状态；
    状态可能不体现在单点均值，而体现在基线、趋势、短时波动和最近变化。
    """

    groups = load_physio_10hz_groups()
    inv = pd.read_csv(PHYSIO_INVENTORY, encoding="utf-8-sig")
    inv_keys = set(zip(inv["subject"].astype(str), inv["session_stamp"].astype(str)))
    rows: List[Dict[str, object]] = []

    for i, row in manifest.iterrows():
        subject = str(row["subject"])
        session = session_stamp_from_recording(str(row["recording"]))
        obs = float(row["observation_s"])
        key = (subject, session)
        g = groups.get(key)
        out: Dict[str, object] = {
            "row_index": int(i),
            "event_uid": str(row["event_uid"]),
            "subject": subject,
            "recording": str(row["recording"]),
            "split": str(row["split"]),
            "delay_ms": int(row["delay_ms"]),
            "observation_s": obs,
            "physio_recording_in_inventory": bool(key in inv_keys),
            "physio_recording_has_10hz": bool(g is not None and not g.empty),
            "physio_uses_post_observation": False,
        }
        if g is None or g.empty:
            out["physio_status"] = "missing_recording"
            rows.append(out)
            continue

        out["physio_status"] = "ok"
        times = pd.to_numeric(g["time_bin_s"], errors="coerce").to_numpy(dtype=float)
        per_window_means: Dict[Tuple[str, str], float] = {}
        per_window_stds: Dict[Tuple[str, str], float] = {}
        for win_name, (offset_start, offset_end) in WINDOWS.items():
            start_s = max(0.0, obs + offset_start)
            end_s = max(0.0, obs + offset_end)
            if end_s > obs + 1e-9:
                out["physio_uses_post_observation"] = True
            left = int(np.searchsorted(times, start_s, side="left"))
            right = int(np.searchsorted(times, end_s, side="right"))
            win_t = times[left:right]
            out[f"{win_name}_start_s"] = start_s
            out[f"{win_name}_end_s"] = end_s
            out[f"{win_name}_rows"] = int(len(win_t))
            out[f"{win_name}_duration_s"] = float(win_t[-1] - win_t[0]) if len(win_t) >= 2 else 0.0
            for sig in PHYSIO_SIGNALS:
                vals = pd.to_numeric(g[sig].iloc[left:right], errors="coerce").to_numpy(dtype=float) if sig in g.columns else np.array([], dtype=float)
                prefix = f"physio10_{win_name}_{sig}"
                stats = window_stats(win_t, vals, prefix)
                out.update(stats)
                per_window_means[(win_name, sig)] = stats[f"{prefix}_mean"]
                per_window_stds[(win_name, sig)] = stats[f"{prefix}_std"]

        for sig in PHYSIO_SIGNALS:
            recent = per_window_means.get(("pre2_0", sig), math.nan)
            pre5 = per_window_means.get(("pre5_pre2", sig), math.nan)
            pre10 = per_window_means.get(("pre10_pre5", sig), math.nan)
            base = per_window_means.get(("pre20_pre10", sig), math.nan)
            recent_std = per_window_stds.get(("pre2_0", sig), math.nan)
            pre5_std = per_window_stds.get(("pre5_pre2", sig), math.nan)
            out[f"physio10_delta_pre2_0_minus_pre5_pre2_{sig}_mean"] = float(recent - pre5) if np.isfinite(recent) and np.isfinite(pre5) else math.nan
            out[f"physio10_delta_pre2_0_minus_pre10_pre5_{sig}_mean"] = float(recent - pre10) if np.isfinite(recent) and np.isfinite(pre10) else math.nan
            out[f"physio10_delta_pre2_0_minus_pre20_pre10_{sig}_mean"] = float(recent - base) if np.isfinite(recent) and np.isfinite(base) else math.nan
            out[f"physio10_delta_pre2_0_minus_pre5_pre2_{sig}_std"] = float(recent_std - pre5_std) if np.isfinite(recent_std) and np.isfinite(pre5_std) else math.nan
        rows.append(out)

    return pd.DataFrame(rows), inv


def load_v253a_physio_1hz_features(n_rows: int) -> pd.DataFrame:
    """加载 v253a 已生成的 1Hz 锚点前生理窗口特征，用作粗粒度对照。"""

    if not V253A_PHYSIO_FEATURES.exists():
        return pd.DataFrame({"row_index": np.arange(n_rows)})
    df = pd.read_csv(V253A_PHYSIO_FEATURES, encoding="utf-8-sig")
    keep = ["row_index"] + [c for c in df.columns if c.startswith("physio_") and pd.api.types.is_numeric_dtype(df[c])]
    return df[keep].copy()


def build_future_targets(y_true: np.ndarray, sample_metrics: pd.DataFrame, split: np.ndarray) -> pd.DataFrame:
    """从真实未来轨迹构造行为诊断标签，特征不会使用这些未来信息。"""

    y = np.asarray(y_true, dtype=float)
    peak_idx = np.nanargmax(np.abs(np.where(np.isfinite(y), y, np.nan)), axis=1)
    signed_peak = np.array([y[i, peak_idx[i]] for i in range(len(y))], dtype=float)
    future_peak_abs = np.nanmax(np.abs(y), axis=1)
    future_range = np.nanmax(y, axis=1) - np.nanmin(y, axis=1)
    future_mean_abs = np.nanmean(np.abs(y), axis=1)
    future_final = y[:, -1]
    future_slope = (y[:, -1] - y[:, 0]) / 2.0

    out = pd.DataFrame(
        {
            "row_index": sample_metrics["row_index"].to_numpy(dtype=int),
            "future_peak_abs": future_peak_abs,
            "future_signed_peak": signed_peak,
            "future_peak_sign": (signed_peak >= 0).astype(int),
            "future_range": future_range,
            "future_mean_abs": future_mean_abs,
            "future_final": future_final,
            "future_slope": future_slope,
        }
    )
    train_mask = split == "train"
    out["high_future_abs_q75"] = (out["future_peak_abs"] >= np.nanquantile(out.loc[train_mask, "future_peak_abs"], 0.75)).astype(int)
    out["high_future_range_q75"] = (out["future_range"] >= np.nanquantile(out.loc[train_mask, "future_range"], 0.75)).astype(int)
    out["strong_steer_existing"] = sample_metrics["is_strong_steer"].astype(int).to_numpy()

    test_tail_v250 = pd.to_numeric(sample_metrics.loc[sample_metrics["split"].eq("test"), "tail_rmse_v250"], errors="coerce")
    v250_q90 = float(np.nanquantile(test_tail_v250.to_numpy(dtype=float), 0.90))
    out["bad_top10_v250_diagnostic"] = (
        pd.to_numeric(sample_metrics["tail_rmse_v250"], errors="coerce").to_numpy(dtype=float) >= v250_q90
    ).astype(int)
    return out


def add_future_clusters(y_true: np.ndarray, split: np.ndarray, targets: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """只用 train 未来轨迹拟合聚类，再给 val/test 分配诊断用行为模式标签。"""

    train_mask = split == "train"
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_true[train_mask])
    y_all = scaler.transform(y_true)
    cluster_rows: List[Dict[str, object]] = []
    for k in [4, 6]:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        labels_train = km.fit_predict(y_train)
        labels_all = km.predict(y_all)
        targets[f"future_cluster{k}"] = labels_all.astype(int)
        for cluster_id in range(k):
            mask = labels_all == cluster_id
            train_cluster = train_mask & mask
            cluster_rows.append(
                {
                    "cluster_target": f"future_cluster{k}",
                    "cluster_id": int(cluster_id),
                    "n_all": int(mask.sum()),
                    "n_train": int(train_cluster.sum()),
                    "n_val": int(((split == "val") & mask).sum()),
                    "n_test": int(((split == "test") & mask).sum()),
                    "train_future_peak_abs_mean": float(np.nanmean(targets.loc[train_cluster, "future_peak_abs"])) if int(train_cluster.sum()) else math.nan,
                    "train_future_range_mean": float(np.nanmean(targets.loc[train_cluster, "future_range"])) if int(train_cluster.sum()) else math.nan,
                    "train_future_final_mean": float(np.nanmean(targets.loc[train_cluster, "future_final"])) if int(train_cluster.sum()) else math.nan,
                }
            )
    return targets, pd.DataFrame(cluster_rows)


def numeric_feature_columns(df: pd.DataFrame, prefixes: Tuple[str, ...]) -> List[str]:
    """按前缀选择数值特征列。"""

    skip = {
        "row_index",
        "delay_ms",
        "observation_s",
    }
    cols: List[str] = []
    for col in df.columns:
        if col in skip:
            continue
        if not col.startswith(prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def clean_train_feature_block(x: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """删除 train 内几乎全空或无方差的列，避免诊断模型被坏列干扰。"""

    x = np.asarray(x, dtype=float)
    train_x = x[train_mask]
    finite_count = np.isfinite(train_x).sum(axis=0)
    train_std = np.nanstd(train_x, axis=0)
    keep = (finite_count >= 20) & np.isfinite(train_std) & (train_std > 1e-12)
    if not bool(np.any(keep)):
        keep = np.zeros(x.shape[1], dtype=bool)
    return x[:, keep], keep


def make_feature_blocks(
    vehicle_x: np.ndarray,
    physio10: pd.DataFrame,
    physio1: pd.DataFrame,
    split: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """构造诊断模型使用的特征块，并记录每个块保留了多少列。"""

    train_mask = split == "train"
    feature_blocks: Dict[str, np.ndarray] = {}
    audit_rows: List[Dict[str, object]] = []

    blocks: Dict[str, Tuple[np.ndarray, List[str]]] = {
        "vehicle_only": (vehicle_x.astype(float), [f"vehicle_{i}" for i in range(vehicle_x.shape[1])]),
    }
    physio10_cols = numeric_feature_columns(physio10, ("physio10_",))
    physio1_cols = numeric_feature_columns(physio1, ("physio_",))
    blocks["physio10hz_deep"] = (physio10[physio10_cols].to_numpy(dtype=float), physio10_cols)
    blocks["physio1hz_v253a"] = (physio1[physio1_cols].to_numpy(dtype=float), physio1_cols)
    blocks["vehicle_plus_physio10hz"] = (
        np.concatenate([vehicle_x.astype(float), physio10[physio10_cols].to_numpy(dtype=float)], axis=1),
        [f"vehicle_{i}" for i in range(vehicle_x.shape[1])] + physio10_cols,
    )

    for block, (x, cols) in blocks.items():
        x_keep, keep = clean_train_feature_block(x, train_mask)
        feature_blocks[block] = x_keep
        kept_cols = [c for c, k in zip(cols, keep) if bool(k)]
        audit_rows.append(
            {
                "feature_block": block,
                "raw_dim": int(x.shape[1]),
                "kept_dim": int(x_keep.shape[1]),
                "kept_physio_columns": int(sum(c.startswith("physio") for c in kept_cols)),
            }
        )
    return feature_blocks, pd.DataFrame(audit_rows)


def safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """二分类 AUC，类别不足时返回 NaN。"""

    if len(np.unique(y_true)) < 2:
        return math.nan
    try:
        return float(roc_auc_score(y_true, proba))
    except Exception:
        return math.nan


def evaluate_classification_blocks(
    feature_blocks: Dict[str, np.ndarray],
    targets: pd.DataFrame,
    split: np.ndarray,
) -> pd.DataFrame:
    """用 train 拟合轻量分类器，在 val/test 上看生理是否有行为模式信号。"""

    target_cols = [
        "future_cluster4",
        "future_cluster6",
        "high_future_abs_q75",
        "high_future_range_q75",
        "future_peak_sign",
        "strong_steer_existing",
        "bad_top10_v250_diagnostic",
    ]
    rows: List[Dict[str, object]] = []
    train_mask = split == "train"
    eval_masks = {"val": split == "val", "test": split == "test"}

    for target_col in target_cols:
        y_raw = targets[target_col].to_numpy()
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        train_classes = np.unique(y[train_mask])
        if len(train_classes) < 2:
            continue
        for block, x in feature_blocks.items():
            if x.shape[1] == 0:
                continue
            clf = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1500,
                            class_weight="balanced",
                            random_state=SEED,
                            solver="lbfgs",
                        ),
                    ),
                ]
            )
            clf.fit(x[train_mask], y[train_mask])
            for eval_name, eval_mask in eval_masks.items():
                if int(eval_mask.sum()) == 0:
                    continue
                pred = clf.predict(x[eval_mask])
                row = {
                    "task_type": "classification",
                    "target": target_col,
                    "feature_block": block,
                    "eval_split": eval_name,
                    "n_eval": int(eval_mask.sum()),
                    "accuracy": float(accuracy_score(y[eval_mask], pred)),
                    "macro_f1": float(f1_score(y[eval_mask], pred, average="macro", zero_division=0)),
                    "auc": math.nan,
                }
                if len(le.classes_) == 2 and hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(x[eval_mask])[:, 1]
                    row["auc"] = safe_auc(y[eval_mask], proba)
                rows.append(row)
    return pd.DataFrame(rows)


def evaluate_regression_blocks(
    feature_blocks: Dict[str, np.ndarray],
    targets: pd.DataFrame,
    split: np.ndarray,
) -> pd.DataFrame:
    """用 train 拟合轻量回归器，检查生理是否能预测未来强度/形态摘要。"""

    target_cols = [
        "future_peak_abs",
        "future_range",
        "future_mean_abs",
        "future_final",
        "future_slope",
    ]
    rows: List[Dict[str, object]] = []
    train_mask = split == "train"
    eval_masks = {"val": split == "val", "test": split == "test"}
    for target_col in target_cols:
        y = pd.to_numeric(targets[target_col], errors="coerce").to_numpy(dtype=float)
        good_train = train_mask & np.isfinite(y)
        if int(good_train.sum()) < 50:
            continue
        for block, x in feature_blocks.items():
            if x.shape[1] == 0:
                continue
            reg = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                    ("reg", Ridge(alpha=10.0, random_state=SEED)),
                ]
            )
            reg.fit(x[good_train], y[good_train])
            for eval_name, eval_mask in eval_masks.items():
                good_eval = eval_mask & np.isfinite(y)
                if int(good_eval.sum()) < 10:
                    continue
                pred = reg.predict(x[good_eval])
                rows.append(
                    {
                        "task_type": "regression",
                        "target": target_col,
                        "feature_block": block,
                        "eval_split": eval_name,
                        "n_eval": int(good_eval.sum()),
                        "r2": float(r2_score(y[good_eval], pred)),
                        "mae": float(mean_absolute_error(y[good_eval], pred)),
                    }
                )
    return pd.DataFrame(rows)


def eta_squared(feature: np.ndarray, labels: np.ndarray) -> float:
    """单特征对类别标签的 eta^2，可看作描述性可分性。"""

    x = np.asarray(feature, dtype=float)
    labels = np.asarray(labels)
    mask = np.isfinite(x) & pd.notna(labels)
    if int(mask.sum()) < 20:
        return math.nan
    x = x[mask]
    labels = labels[mask]
    grand = float(np.mean(x))
    ss_total = float(np.sum((x - grand) ** 2))
    if ss_total <= 1e-12:
        return math.nan
    ss_between = 0.0
    for label in np.unique(labels):
        vals = x[labels == label]
        if len(vals) == 0:
            continue
        ss_between += float(len(vals) * (np.mean(vals) - grand) ** 2)
    return float(ss_between / ss_total)


def build_feature_quality_and_eta(
    physio10: pd.DataFrame,
    targets: pd.DataFrame,
    split: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成生理特征质量表和可分性 eta^2 表。"""

    cols = numeric_feature_columns(physio10, ("physio10_",))
    train_mask = split == "train"
    quality_rows: List[Dict[str, object]] = []
    for col in cols:
        arr = pd.to_numeric(physio10[col], errors="coerce").to_numpy(dtype=float)
        train_arr = arr[train_mask]
        finite = np.isfinite(arr)
        train_finite = np.isfinite(train_arr)
        signal = "unknown"
        for sig in PHYSIO_SIGNALS:
            if sig in col:
                signal = sig
                break
        quality_rows.append(
            {
                "feature": col,
                "signal": signal,
                "finite_rate_all": float(finite.mean()),
                "finite_rate_train": float(train_finite.mean()),
                "train_std": float(np.nanstd(train_arr)) if int(train_finite.sum()) else math.nan,
                "train_unique_rounded": int(pd.Series(train_arr[np.isfinite(train_arr)]).round(6).nunique()) if int(train_finite.sum()) else 0,
            }
        )
    quality = pd.DataFrame(quality_rows)

    eta_targets = {
        "subject": physio10["subject"].astype(str).to_numpy(),
        "recording": physio10["recording"].astype(str).to_numpy(),
        "future_cluster4": targets["future_cluster4"].astype(str).to_numpy(),
        "high_future_abs_q75": targets["high_future_abs_q75"].astype(str).to_numpy(),
        "high_future_range_q75": targets["high_future_range_q75"].astype(str).to_numpy(),
        "strong_steer_existing": targets["strong_steer_existing"].astype(str).to_numpy(),
    }
    eta_rows: List[Dict[str, object]] = []
    for target_name, labels in eta_targets.items():
        for col in cols:
            arr = pd.to_numeric(physio10[col], errors="coerce").to_numpy(dtype=float)
            eta = eta_squared(arr, labels)
            if np.isfinite(eta):
                signal = next((sig for sig in PHYSIO_SIGNALS if sig in col), "unknown")
                eta_rows.append({"target": target_name, "feature": col, "signal": signal, "eta2": float(eta)})
    eta_df = pd.DataFrame(eta_rows).sort_values(["target", "eta2"], ascending=[True, False])
    return quality, eta_df


def summarize_alignment(physio10: pd.DataFrame) -> pd.DataFrame:
    """汇总 10Hz 生理窗口覆盖和对齐状态。"""

    rows: List[Dict[str, object]] = []
    for split_name, g in physio10.groupby("split"):
        rows.append(
            {
                "split": split_name,
                "n": int(len(g)),
                "recording_inventory_rate": float(g["physio_recording_in_inventory"].mean()),
                "recording_has_10hz_rate": float(g["physio_recording_has_10hz"].mean()),
                "uses_post_observation_rate": float(g["physio_uses_post_observation"].mean()),
            }
        )
        for win_name in WINDOWS:
            rows[-1][f"{win_name}_rows_mean"] = float(pd.to_numeric(g[f"{win_name}_rows"], errors="coerce").mean())
            rows[-1][f"{win_name}_rows_p10"] = float(pd.to_numeric(g[f"{win_name}_rows"], errors="coerce").quantile(0.10))
            rows[-1][f"{win_name}_rows_p50"] = float(pd.to_numeric(g[f"{win_name}_rows"], errors="coerce").quantile(0.50))
            rows[-1][f"{win_name}_rows_p90"] = float(pd.to_numeric(g[f"{win_name}_rows"], errors="coerce").quantile(0.90))
    return pd.DataFrame(rows)


def signal_quality_summary(feature_quality: pd.DataFrame) -> pd.DataFrame:
    """按生理信号汇总特征可用性。"""

    rows = []
    for signal, g in feature_quality.groupby("signal"):
        rows.append(
            {
                "signal": signal,
                "n_features": int(len(g)),
                "finite_rate_train_mean": float(g["finite_rate_train"].mean()),
                "train_std_median": float(g["train_std"].median()),
                "near_constant_feature_rate": float((g["train_std"].fillna(0.0) <= 1e-12).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("finite_rate_train_mean", ascending=False)


def add_delta_vs_vehicle(df: pd.DataFrame, metric_col: str, higher_is_better: bool) -> pd.DataFrame:
    """给评估表增加相对 vehicle_only 的变化量。"""

    if df.empty or metric_col not in df.columns:
        return df
    base = df[df["feature_block"].eq("vehicle_only")][["target", "eval_split", metric_col]].rename(columns={metric_col: "vehicle_metric"})
    out = df.merge(base, on=["target", "eval_split"], how="left")
    out[f"delta_{metric_col}_minus_vehicle"] = out[metric_col] - out["vehicle_metric"]
    if not higher_is_better:
        out[f"delta_{metric_col}_minus_vehicle"] = out["vehicle_metric"] - out[metric_col]
    return out


def plot_classification_summary(cls: pd.DataFrame) -> Path:
    """画分类任务 macro-F1 对比。"""

    path = FIGURES / "v254a_behavior_classification_macro_f1.png"
    sub = cls[
        cls["target"].isin(["future_cluster4", "high_future_abs_q75", "high_future_range_q75"])
        & cls["eval_split"].eq("test")
    ].copy()
    if sub.empty:
        return path
    order = ["vehicle_only", "physio1hz_v253a", "physio10hz_deep", "vehicle_plus_physio10hz"]
    sub["feature_block"] = pd.Categorical(sub["feature_block"], categories=order, ordered=True)
    targets = list(sub["target"].drop_duplicates())
    x = np.arange(len(targets))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, block in enumerate(order):
        vals = []
        for target in targets:
            m = sub[sub["target"].eq(target) & sub["feature_block"].eq(block)]
            vals.append(float(m["macro_f1"].iloc[0]) if len(m) else np.nan)
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=block)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=15, ha="right")
    ax.set_ylabel("test macro-F1")
    ax.set_title("v254a: 生理深层特征对未来行为标签的诊断分类能力")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_regression_summary(reg: pd.DataFrame) -> Path:
    """画回归任务 R2 对比。"""

    path = FIGURES / "v254a_future_summary_regression_r2.png"
    sub = reg[
        reg["target"].isin(["future_peak_abs", "future_range", "future_mean_abs"])
        & reg["eval_split"].eq("test")
    ].copy()
    if sub.empty:
        return path
    order = ["vehicle_only", "physio1hz_v253a", "physio10hz_deep", "vehicle_plus_physio10hz"]
    sub["feature_block"] = pd.Categorical(sub["feature_block"], categories=order, ordered=True)
    targets = list(sub["target"].drop_duplicates())
    x = np.arange(len(targets))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, block in enumerate(order):
        vals = []
        for target in targets:
            m = sub[sub["target"].eq(target) & sub["feature_block"].eq(block)]
            vals.append(float(m["r2"].iloc[0]) if len(m) else np.nan)
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=block)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=15, ha="right")
    ax.set_ylabel("test R2")
    ax.set_title("v254a: 生理深层特征对未来强度摘要的诊断回归能力")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_eta_summary(eta_df: pd.DataFrame) -> Path:
    """画 subject 和未来行为标签的 top eta2。"""

    path = FIGURES / "v254a_top_physio_eta2.png"
    targets = ["subject", "future_cluster4", "high_future_abs_q75", "strong_steer_existing"]
    sub_rows = []
    for target in targets:
        top = eta_df[eta_df["target"].eq(target)].head(6)
        for _, row in top.iterrows():
            sub_rows.append({"target": target, "label": f"{row['signal']}\n{str(row['feature']).split('_')[-1]}", "eta2": float(row["eta2"])})
    sub = pd.DataFrame(sub_rows)
    if sub.empty:
        return path
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    for ax, target in zip(axes, targets):
        g = sub[sub["target"].eq(target)].sort_values("eta2")
        ax.barh(g["label"], g["eta2"])
        ax.set_title(target)
        ax.set_xlabel("eta^2")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("v254a: 生理特征对身份/行为标签的描述性可分性")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_alignment(alignment: pd.DataFrame) -> Path:
    """画不同 split 的窗口行数覆盖。"""

    path = FIGURES / "v254a_physio10hz_window_rows.png"
    rows = []
    for _, row in alignment.iterrows():
        for win in WINDOWS:
            rows.append({"split": row["split"], "window": win, "rows_mean": float(row[f"{win}_rows_mean"])})
    sub = pd.DataFrame(rows)
    if sub.empty:
        return path
    splits = list(sub["split"].drop_duplicates())
    wins = list(WINDOWS.keys())
    x = np.arange(len(wins))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, split_name in enumerate(splits):
        vals = []
        for win in wins:
            m = sub[sub["split"].eq(split_name) & sub["window"].eq(win)]
            vals.append(float(m["rows_mean"].iloc[0]) if len(m) else np.nan)
        ax.bar(x + (i - (len(splits) - 1) / 2) * width, vals, width=width, label=split_name)
    ax.set_xticks(x)
    ax.set_xticklabels(wins, rotation=15, ha="right")
    ax.set_ylabel("mean 10Hz rows")
    ax.set_title("v254a: 锚点前 10Hz 生理窗口覆盖")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    """记录关键输入哈希。"""

    paths = [V252_SCRIPT, PHYSIO_1HZ, PHYSIO_10HZ, PHYSIO_INVENTORY, V253A_PHYSIO_FEATURES]
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
            rows.append(
                {
                    "relative_path": str(path.relative_to(OUT)),
                    "bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
            )
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> str | None:
    """打包 v254a 产物，并返回 zipfile.testzip 结果。"""

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(__file__, arcname=f"scripts/{Path(__file__).name}")
        for folder in [TABLES, FIGURES, REPORTS, LOGS]:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    z.write(path, arcname=str(path.relative_to(OUT)))
    with zipfile.ZipFile(ZIP_PATH) as z:
        return z.testzip()


def write_report(
    alignment: pd.DataFrame,
    block_audit: pd.DataFrame,
    signal_summary: pd.DataFrame,
    cls: pd.DataFrame,
    reg: pd.DataFrame,
    eta_df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    figures: List[Path],
) -> None:
    """写中文审计报告。"""

    report = REPORTS / "v254a_physio_deep_signal_audit_cn.md"
    lines: List[str] = []
    lines.append("# v254a 生理信号深层挖掘审计")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("本轮不把生理数据简单拼接进轨迹模型，而是先检查生理数据是否真的含有可学习的驾驶员状态结构，以及这种结构是否与未来驾驶行为有关。")
    lines.append("")
    lines.append("## 数据与边界")
    lines.append("")
    lines.append("- 生理主输入：`physio_features_10hz.csv`，来自清洗后 200Hz 连续生理层的 10Hz 聚合特征。")
    lines.append("- 对照输入：v253a 已提取的 1Hz 锚点前窗口生理特征。")
    lines.append("- 事件样本：复用 v252/v253 固定 rolling sample 和 split。")
    lines.append("- 诊断模型：只训练轻量 logistic/ridge 分类或回归头，不训练轨迹预测模型。")
    lines.append("- 训练口径：所有诊断模型只在 train 拟合，val/test 只报告，不用于调参。")
    lines.append("")
    lines.append("## 对齐覆盖")
    lines.append("")
    lines.append(alignment.to_markdown(index=False))
    lines.append("")
    lines.append("## 特征块维度")
    lines.append("")
    lines.append(block_audit.to_markdown(index=False))
    lines.append("")
    lines.append("## 生理信号质量摘要")
    lines.append("")
    lines.append(signal_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## 未来轨迹聚类摘要")
    lines.append("")
    lines.append(cluster_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## 行为分类诊断")
    lines.append("")
    show_cls = cls[
        cls["target"].isin(["future_cluster4", "high_future_abs_q75", "high_future_range_q75", "strong_steer_existing"])
        & cls["eval_split"].isin(["val", "test"])
    ].copy()
    lines.append(show_cls.to_markdown(index=False))
    lines.append("")
    lines.append("## 未来摘要回归诊断")
    lines.append("")
    show_reg = reg[
        reg["target"].isin(["future_peak_abs", "future_range", "future_mean_abs"])
        & reg["eval_split"].isin(["val", "test"])
    ].copy()
    lines.append(show_reg.to_markdown(index=False))
    lines.append("")
    lines.append("## 可分性 eta^2 Top 特征")
    lines.append("")
    top_eta = eta_df.groupby("target", group_keys=False).head(12)
    lines.append(top_eta.to_markdown(index=False))
    lines.append("")
    lines.append("## 判读规则")
    lines.append("")
    lines.append("- 如果 `physio10hz_deep` 自己能预测未来行为标签，说明生理中有行为相关状态信号。")
    lines.append("- 如果 `vehicle_plus_physio10hz` 明显优于 `vehicle_only`，说明生理对车辆输入有增量。")
    lines.append("- 如果 subject/recording 的 eta^2 很高但未来行为 eta^2 很低，说明生理更像身份/记录状态，而不是当前任务可用的行为状态。")
    lines.append("- 如果 10Hz 明显优于 1Hz，说明 v253a/v253b 的失败可能来自 1Hz 特征太粗。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")


def build_guardrail(
    split_check: pd.DataFrame,
    physio10: pd.DataFrame,
    zip_test: str | None,
) -> Dict[str, object]:
    """生成守门检查。"""

    cross_split = int(split_check["split_nunique"].gt(1).sum()) if "split_nunique" in split_check.columns else 0
    return {
        "pass": bool(cross_split == 0 and not bool(physio10["physio_uses_post_observation"].any()) and zip_test is None),
        "same_event_uid_cross_split_count": cross_split,
        "retrained_trajectory_model": False,
        "diagnostic_heads_train_split_only": True,
        "test_used_for_feature_or_model_selection": False,
        "physio_uses_post_observation_any": bool(physio10["physio_uses_post_observation"].any()),
        "physio10_recording_has_rate": float(physio10["physio_recording_has_10hz"].mean()),
        "zip_testzip": zip_test,
    }


def main() -> None:
    print("[v254a] physio deep signal audit")
    print("[v254a] 10Hz pre-anchor windows + train-only diagnostic heads")
    clean_out_dir()
    np.random.seed(SEED)

    loaded = V252.load_fixed_inputs()
    data = loaded["data"]
    manifest = data.manifest.copy()
    sample_metrics = loaded["sample_metrics"].copy()
    split_check = loaded["split_check"].copy()
    split = manifest["split"].astype(str).to_numpy()

    physio10, inventory = build_event_physio_10hz_features(manifest)
    physio1 = load_v253a_physio_1hz_features(len(manifest))
    targets = build_future_targets(loaded["y_true"], sample_metrics, split)
    targets, cluster_summary = add_future_clusters(loaded["y_true"], split, targets)

    # row_index 保证所有表按 v252 fixed sample 对齐。
    physio10 = manifest[["event_uid", "subject", "recording", "split", "delay_ms"]].reset_index(names="row_index").merge(
        physio10.drop(columns=["event_uid", "subject", "recording", "split", "delay_ms"], errors="ignore"),
        on="row_index",
        how="left",
    )
    physio1 = pd.DataFrame({"row_index": np.arange(len(manifest))}).merge(physio1, on="row_index", how="left")

    alignment = summarize_alignment(physio10)
    feature_blocks, block_audit = make_feature_blocks(loaded["x_flat"], physio10, physio1, split)
    quality, eta_df = build_feature_quality_and_eta(physio10, targets, split)
    signal_summary = signal_quality_summary(quality)

    cls = evaluate_classification_blocks(feature_blocks, targets, split)
    reg = evaluate_regression_blocks(feature_blocks, targets, split)
    cls = add_delta_vs_vehicle(cls, "macro_f1", higher_is_better=True)
    reg = add_delta_vs_vehicle(reg, "r2", higher_is_better=True)

    write_csv(physio10, TABLES / "v254a_event_physio10hz_deep_features.csv")
    write_csv(physio1, TABLES / "v254a_event_physio1hz_v253a_features.csv")
    write_csv(targets, TABLES / "v254a_future_behavior_targets.csv")
    write_csv(alignment, TABLES / "v254a_alignment_coverage_summary.csv")
    write_csv(block_audit, TABLES / "v254a_feature_block_audit.csv")
    write_csv(quality, TABLES / "v254a_physio10hz_feature_quality.csv")
    write_csv(signal_summary, TABLES / "v254a_physio_signal_quality_summary.csv")
    write_csv(eta_df, TABLES / "v254a_physio_eta2_by_target_feature.csv")
    write_csv(cluster_summary, TABLES / "v254a_future_cluster_summary.csv")
    write_csv(cls, TABLES / "v254a_behavior_classification_diagnostics.csv")
    write_csv(reg, TABLES / "v254a_future_summary_regression_diagnostics.csv")
    write_csv(inventory, TABLES / "v254a_physio_recording_inventory_copy.csv")

    figures = [
        plot_alignment(alignment),
        plot_eta_summary(eta_df),
        plot_classification_summary(cls),
        plot_regression_summary(reg),
    ]
    write_input_hashes()
    write_file_inventory()
    write_report(alignment, block_audit, signal_summary, cls, reg, eta_df, cluster_summary, figures)
    write_file_inventory()
    zip_test = make_zip()
    guardrail = build_guardrail(split_check, physio10, zip_test)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v254a guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    cls_test = cls[cls["eval_split"].eq("test")].copy()
    reg_test = reg[reg["eval_split"].eq("test")].copy()
    best_cls = cls_test.sort_values("macro_f1", ascending=False).head(1)
    best_reg = reg_test.sort_values("r2", ascending=False).head(1)
    print(f"[v254a] report={REPORTS / 'v254a_physio_deep_signal_audit_cn.md'}")
    print(f"[v254a] zip={ZIP_PATH}")
    if len(best_cls):
        r = best_cls.iloc[0]
        print(f"[v254a] best test classification: target={r['target']} block={r['feature_block']} macro_f1={float(r['macro_f1']):.4f}")
    if len(best_reg):
        r = best_reg.iloc[0]
        print(f"[v254a] best test regression: target={r['target']} block={r['feature_block']} r2={float(r['r2']):.4f}")


if __name__ == "__main__":
    main()
