#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v254b 200Hz 连续生理事件相关表征重设计。

目标：
- 不再使用简单 1Hz/10Hz 表格窗口拼接；
- 直接从清洗后的 200Hz 连续生理层按事件抽取锚点前状态变化；
- 对每个事件使用自身锚点前 baseline 做因果归一化；
- 同时报告 subject-disjoint 与 subject-aware 两种诊断口径。

边界：
- 不训练轨迹预测模型；
- 不使用 observation_s 之后的生理；
- subject-aware 只是诊断个体化潜力，不替代正式 subject-disjoint 测试。
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V252_SCRIPT = BASELINES / "scripts" / "stage03_v252_input_similarity_future_divergence_20260701.py"
V254A_SCRIPT = BASELINES / "scripts" / "stage03_v254a_physio_deep_signal_audit_20260701.py"
PHYSIO_INVENTORY = (
    REBUILD
    / "06_physio_processing"
    / "physio_subject_collection_v1_20260603"
    / "tables"
    / "physio_recording_inventory.csv"
)

OUT = BASELINES / "v254b_physio_200hz_event_representation_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v254b_physio_200hz_event_representation_20260702_pack.zip"

SEED = 25402

PHYSIO_COLS = [
    "ECG_filt200",
    "EMG_filt200",
    "EMG_RMS",
    "EDA_filt200",
    "EDA_Tonic",
    "EDA_Phasic",
    "RESP_filt200",
    "HR_bpm",
    "RESP_BPM",
    "RESP_Amplitude",
    "t_s",
]

SIGNALS = [
    "HR_bpm",
    "EMG_RMS",
    "EMG_filt200",
    "EDA_filt200",
    "EDA_Tonic",
    "EDA_Phasic",
    "RESP_filt200",
    "ECG_filt200",
    "RESP_BPM",
    "RESP_Amplitude",
]

BASELINE_WINDOW = (-60.0, -20.0)
EVENT_WINDOWS = {
    "pre20_pre10": (-20.0, -10.0),
    "pre10_pre5": (-10.0, -5.0),
    "pre5_pre2": (-5.0, -2.0),
    "pre2_0": (-2.0, 0.0),
}

TARGETS_FOR_REPORT = [
    "future_cluster4",
    "high_future_abs_q75",
    "high_future_range_q75",
    "strong_steer_existing",
    "bad_top10_v250_diagnostic",
]

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    """按路径导入前序脚本。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V252 = import_module_from_path("stage03_v252_for_v254b", V252_SCRIPT)
V254A = import_module_from_path("stage03_v254a_for_v254b", V254A_SCRIPT)


def ensure_dirs() -> None:
    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def session_stamp_from_recording(recording: str) -> str:
    return str(recording).replace("Entity_Recording_", "")


def finite(values: Iterable[object]) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def robust_scale(values: np.ndarray) -> float:
    vals = finite(values)
    if vals.size < 5:
        return math.nan
    q25, q75 = np.quantile(vals, [0.25, 0.75])
    iqr = float(q75 - q25)
    mad = float(np.median(np.abs(vals - np.median(vals))))
    std = float(np.std(vals))
    for scale in [iqr / 1.349 if iqr > 0 else math.nan, mad * 1.4826 if mad > 0 else math.nan, std]:
        if np.isfinite(scale) and scale > 1e-9:
            return float(scale)
    return math.nan


def slope(times: np.ndarray, vals: np.ndarray) -> float:
    mask = np.isfinite(times) & np.isfinite(vals)
    if int(mask.sum()) < 2:
        return math.nan
    t = times[mask]
    v = vals[mask]
    dt = float(t[-1] - t[0])
    if abs(dt) < 1e-9:
        return math.nan
    return float((v[-1] - v[0]) / dt)


def safe_div(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-9:
        return math.nan
    return float(num / den)


def window_basic_stats(times: np.ndarray, vals: np.ndarray, prefix: str, baseline_median: float, baseline_scale: float) -> Dict[str, float]:
    """一个窗口的 raw 与 baseline-normalized 特征。"""

    out: Dict[str, float] = {}
    arr = np.asarray(vals, dtype=float)
    good = arr[np.isfinite(arr)]
    out[f"{prefix}_n"] = int(len(arr))
    out[f"{prefix}_valid_ratio"] = float(len(good) / max(1, len(arr)))
    if len(good) == 0:
        for name in ["mean", "std", "p10", "p50", "p90", "range", "abs_mean", "rms", "slope", "last_minus_first"]:
            out[f"{prefix}_{name}"] = math.nan
        for name in ["z_mean", "z_p90", "z_range", "z_slope", "z_last_minus_first", "burst_rate_z2", "burst_rate_z3"]:
            out[f"{prefix}_{name}"] = math.nan
        return out

    mean = float(np.mean(good))
    std = float(np.std(good))
    p10, p50, p90 = [float(v) for v in np.quantile(good, [0.10, 0.50, 0.90])]
    rng = float(np.max(good) - np.min(good))
    abs_mean = float(np.mean(np.abs(good)))
    rms = float(np.sqrt(np.mean(good**2)))
    win_slope = slope(times, arr)
    mask = np.isfinite(times) & np.isfinite(arr)
    last_minus_first = float(arr[mask][-1] - arr[mask][0]) if int(mask.sum()) >= 2 else math.nan

    out.update(
        {
            f"{prefix}_mean": mean,
            f"{prefix}_std": std,
            f"{prefix}_p10": p10,
            f"{prefix}_p50": p50,
            f"{prefix}_p90": p90,
            f"{prefix}_range": rng,
            f"{prefix}_abs_mean": abs_mean,
            f"{prefix}_rms": rms,
            f"{prefix}_slope": win_slope,
            f"{prefix}_last_minus_first": last_minus_first,
            f"{prefix}_z_mean": safe_div(mean - baseline_median, baseline_scale),
            f"{prefix}_z_p90": safe_div(p90 - baseline_median, baseline_scale),
            f"{prefix}_z_range": safe_div(rng, baseline_scale),
            f"{prefix}_z_slope": safe_div(win_slope, baseline_scale),
            f"{prefix}_z_last_minus_first": safe_div(last_minus_first, baseline_scale),
        }
    )
    if np.isfinite(baseline_median) and np.isfinite(baseline_scale) and baseline_scale > 0:
        z = (arr - baseline_median) / baseline_scale
        out[f"{prefix}_burst_rate_z2"] = float(np.nanmean(np.abs(z) >= 2.0))
        out[f"{prefix}_burst_rate_z3"] = float(np.nanmean(np.abs(z) >= 3.0))
    else:
        out[f"{prefix}_burst_rate_z2"] = math.nan
        out[f"{prefix}_burst_rate_z3"] = math.nan
    return out


def load_physio_inventory() -> Dict[Tuple[str, str], Path]:
    inv = pd.read_csv(PHYSIO_INVENTORY, encoding="utf-8-sig")
    out: Dict[Tuple[str, str], Path] = {}
    for _, row in inv.iterrows():
        out[(str(row["subject"]), str(row["session_stamp"]))] = Path(str(row["physio_file"]))
    return out


def read_physio_recording(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in PHYSIO_COLS if c in header]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["t_s"] = pd.to_numeric(df["t_s"], errors="coerce")
    df = df.sort_values("t_s").reset_index(drop=True)
    return df


def extract_recording_features(recording_df: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """对同一 recording 的所有 rolling sample 抽取事件相关生理特征。"""

    times = pd.to_numeric(recording_df["t_s"], errors="coerce").to_numpy(dtype=float)
    signal_arrays: Dict[str, np.ndarray] = {}
    for sig in SIGNALS:
        if sig in recording_df.columns:
            signal_arrays[sig] = pd.to_numeric(recording_df[sig], errors="coerce").to_numpy(dtype=float)
        else:
            signal_arrays[sig] = np.full(len(times), np.nan, dtype=float)
    rows: List[Dict[str, object]] = []

    for _, sample in samples.iterrows():
        obs = float(sample["observation_s"])
        out: Dict[str, object] = {
            "row_index": int(sample["row_index"]),
            "event_uid": str(sample["event_uid"]),
            "subject": str(sample["subject"]),
            "recording": str(sample["recording"]),
            "split": str(sample["split"]),
            "delay_ms": int(sample["delay_ms"]),
            "observation_s": obs,
            "physio200_status": "ok",
            "physio200_uses_post_observation": False,
        }

        b_start = max(0.0, obs + BASELINE_WINDOW[0])
        b_end = max(0.0, obs + BASELINE_WINDOW[1])
        b_left = int(np.searchsorted(times, b_start, side="left"))
        b_right = int(np.searchsorted(times, b_end, side="right"))
        b_t = times[b_left:b_right]
        out["baseline_pre60_pre20_rows"] = int(len(b_t))
        out["baseline_pre60_pre20_duration_s"] = float(b_t[-1] - b_t[0]) if len(b_t) >= 2 else 0.0

        recent_feature_means: Dict[str, float] = {}
        for sig in SIGNALS:
            vals_all = signal_arrays[sig]
            baseline_vals = vals_all[b_left:b_right]
            base_good = finite(baseline_vals)
            base_median = float(np.median(base_good)) if len(base_good) else math.nan
            base_mean = float(np.mean(base_good)) if len(base_good) else math.nan
            base_scale = robust_scale(baseline_vals)
            out[f"physio200_base_{sig}_median"] = base_median
            out[f"physio200_base_{sig}_mean"] = base_mean
            out[f"physio200_base_{sig}_scale"] = base_scale
            out[f"physio200_base_{sig}_valid_ratio"] = float(len(base_good) / max(1, len(baseline_vals)))

            for win_name, (offset_start, offset_end) in EVENT_WINDOWS.items():
                start = max(0.0, obs + offset_start)
                end = max(0.0, obs + offset_end)
                if end > obs + 1e-9:
                    out["physio200_uses_post_observation"] = True
                left = int(np.searchsorted(times, start, side="left"))
                right = int(np.searchsorted(times, end, side="right"))
                win_t = times[left:right]
                win_vals = vals_all[left:right]
                out[f"{win_name}_rows"] = int(len(win_t))
                prefix = f"physio200_{win_name}_{sig}"
                stats = window_basic_stats(win_t, win_vals, prefix, base_median, base_scale)
                out.update(stats)
                if win_name == "pre2_0":
                    recent_feature_means[sig] = stats.get(f"{prefix}_z_mean", math.nan)

        # 生理学上更紧凑的组合指数，只使用 baseline-normalized recent/pre5 窗口。
        out["physio200_recent_arousal_index"] = float(
            np.nanmean(
                [
                    out.get("physio200_pre2_0_HR_bpm_z_mean", math.nan),
                    out.get("physio200_pre2_0_EMG_RMS_z_p90", math.nan),
                    out.get("physio200_pre2_0_EDA_Phasic_z_p90", math.nan),
                ]
            )
        )
        out["physio200_recent_motor_tension_index"] = float(
            np.nanmean(
                [
                    out.get("physio200_pre2_0_EMG_RMS_z_mean", math.nan),
                    out.get("physio200_pre2_0_EMG_filt200_z_range", math.nan),
                ]
            )
        )
        out["physio200_recent_resp_activity_index"] = float(
            np.nanmean(
                [
                    out.get("physio200_pre2_0_RESP_filt200_z_range", math.nan),
                    out.get("physio200_pre2_0_RESP_filt200_z_slope", math.nan),
                ]
            )
        )
        rows.append(out)

    return pd.DataFrame(rows)


def build_physio200_features(manifest: pd.DataFrame) -> pd.DataFrame:
    """从 200Hz 连续层构造事件相关生理表征。"""

    inventory = load_physio_inventory()
    samples = manifest[["event_uid", "subject", "recording", "split", "delay_ms", "observation_s"]].reset_index(names="row_index").copy()
    samples["session_stamp"] = samples["recording"].map(session_stamp_from_recording)
    parts: List[pd.DataFrame] = []
    missing: List[Dict[str, object]] = []

    grouped = samples.groupby(["subject", "session_stamp"], sort=False)
    n_groups = len(grouped)
    for group_i, ((subject, session), g) in enumerate(grouped, start=1):
        path = inventory.get((str(subject), str(session)))
        if path is None or not path.exists():
            print(f"[v254b] missing physio recording {group_i}/{n_groups}: subject={subject} session={session} samples={len(g)}", flush=True)
            for _, row in g.iterrows():
                missing.append(
                    {
                        "row_index": int(row["row_index"]),
                        "event_uid": str(row["event_uid"]),
                        "subject": str(row["subject"]),
                        "recording": str(row["recording"]),
                        "split": str(row["split"]),
                        "delay_ms": int(row["delay_ms"]),
                        "observation_s": float(row["observation_s"]),
                        "physio200_status": "missing_recording",
                        "physio200_uses_post_observation": False,
                    }
                )
            continue
        print(f"[v254b] extracting recording {group_i}/{n_groups}: subject={subject} session={session} samples={len(g)}", flush=True)
        rec = read_physio_recording(path)
        parts.append(extract_recording_features(rec, g))

    if missing:
        parts.append(pd.DataFrame(missing))
    out = pd.concat(parts, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    return out


def numeric_feature_columns(df: pd.DataFrame, prefixes: Tuple[str, ...]) -> List[str]:
    skip = {"row_index", "delay_ms", "observation_s"}
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
    x = np.asarray(x, dtype=float)
    train_x = x[train_mask]
    finite_count = np.isfinite(train_x).sum(axis=0)
    train_std = np.nanstd(train_x, axis=0)
    keep = (finite_count >= 20) & np.isfinite(train_std) & (train_std > 1e-12)
    if not bool(np.any(keep)):
        keep = np.zeros(x.shape[1], dtype=bool)
    return x[:, keep], keep


def make_feature_blocks(vehicle_x: np.ndarray, physio200: pd.DataFrame, split: np.ndarray) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """构造诊断特征块。"""

    train_mask = split == "train"
    physio_cols_all = numeric_feature_columns(physio200, ("physio200_",))
    norm_cols = [c for c in physio_cols_all if "_z_" in c or c.endswith("_index") or "burst_rate" in c]
    curated_cols = [
        c
        for c in norm_cols
        if any(sig in c for sig in ["HR_bpm", "EMG_RMS", "EMG_filt200", "EDA_Phasic", "EDA_Tonic", "RESP_filt200"])
    ]
    blocks: Dict[str, Tuple[np.ndarray, List[str]]] = {
        "vehicle_only": (vehicle_x.astype(float), [f"vehicle_{i}" for i in range(vehicle_x.shape[1])]),
        "physio200_all": (physio200[physio_cols_all].to_numpy(dtype=float), physio_cols_all),
        "physio200_norm": (physio200[norm_cols].to_numpy(dtype=float), norm_cols),
        "physio200_curated": (physio200[curated_cols].to_numpy(dtype=float), curated_cols),
        "vehicle_plus_physio200_norm": (
            np.concatenate([vehicle_x.astype(float), physio200[norm_cols].to_numpy(dtype=float)], axis=1),
            [f"vehicle_{i}" for i in range(vehicle_x.shape[1])] + norm_cols,
        ),
        "vehicle_plus_physio200_curated": (
            np.concatenate([vehicle_x.astype(float), physio200[curated_cols].to_numpy(dtype=float)], axis=1),
            [f"vehicle_{i}" for i in range(vehicle_x.shape[1])] + curated_cols,
        ),
    }

    feature_blocks: Dict[str, np.ndarray] = {}
    audit_rows: List[Dict[str, object]] = []
    for name, (x, cols) in blocks.items():
        x_keep, keep = clean_train_feature_block(x, train_mask)
        feature_blocks[name] = x_keep
        kept_cols = [c for c, k in zip(cols, keep) if bool(k)]
        audit_rows.append(
            {
                "feature_block": name,
                "raw_dim": int(x.shape[1]),
                "kept_dim": int(x_keep.shape[1]),
                "kept_physio_columns": int(sum(c.startswith("physio200") for c in kept_cols)),
            }
        )
    return feature_blocks, pd.DataFrame(audit_rows)


def make_subject_aware_split(manifest: pd.DataFrame) -> np.ndarray:
    """
    诊断用 subject-aware event split。

    同一个 event_uid 的所有 delay 必须在同一 split，避免事件泄漏。
    该 split 不是正式测试协议，只用于回答“同一驾驶员有历史样本时生理是否更有价值”。
    """

    rng = np.random.default_rng(SEED)
    event_subject = manifest[["event_uid", "subject"]].drop_duplicates("event_uid").copy()
    split_map: Dict[str, str] = {}
    for subject, g in event_subject.groupby("subject"):
        events = g["event_uid"].astype(str).to_numpy()
        rng.shuffle(events)
        n = len(events)
        if n < 5:
            train_cut = max(1, int(n * 0.7))
            val_cut = train_cut
        else:
            train_cut = int(round(n * 0.60))
            val_cut = int(round(n * 0.80))
        for ev in events[:train_cut]:
            split_map[str(ev)] = "train"
        for ev in events[train_cut:val_cut]:
            split_map[str(ev)] = "val"
        for ev in events[val_cut:]:
            split_map[str(ev)] = "test"
    return manifest["event_uid"].astype(str).map(split_map).to_numpy()


def safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    try:
        return float(roc_auc_score(y_true, proba))
    except Exception:
        return math.nan


def evaluate_classification(feature_blocks: Dict[str, np.ndarray], targets: pd.DataFrame, split: np.ndarray, split_protocol: str) -> pd.DataFrame:
    target_cols = TARGETS_FOR_REPORT
    rows: List[Dict[str, object]] = []
    train_mask = split == "train"
    eval_masks = {"val": split == "val", "test": split == "test"}
    for target_col in target_cols:
        y_raw = targets[target_col].to_numpy()
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        if len(np.unique(y[train_mask])) < 2:
            continue
        for block, x in feature_blocks.items():
            if x.shape[1] == 0:
                continue
            print(f"[v254b] {split_protocol} classification target={target_col} block={block} dim={x.shape[1]}", flush=True)
            clf = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        SGDClassifier(
                            loss="log_loss",
                            max_iter=2000,
                            tol=1e-3,
                            alpha=5e-4,
                            class_weight="balanced",
                            random_state=SEED,
                            n_jobs=1,
                        ),
                    ),
                ]
            )
            clf.fit(x[train_mask], y[train_mask])
            for eval_name, mask in eval_masks.items():
                if int(mask.sum()) == 0:
                    continue
                pred = clf.predict(x[mask])
                row = {
                    "split_protocol": split_protocol,
                    "task_type": "classification",
                    "target": target_col,
                    "feature_block": block,
                    "eval_split": eval_name,
                    "n_eval": int(mask.sum()),
                    "accuracy": float(accuracy_score(y[mask], pred)),
                    "macro_f1": float(f1_score(y[mask], pred, average="macro", zero_division=0)),
                    "auc": math.nan,
                }
                if len(le.classes_) == 2 and hasattr(clf, "predict_proba"):
                    row["auc"] = safe_auc(y[mask], clf.predict_proba(x[mask])[:, 1])
                rows.append(row)
    out = pd.DataFrame(rows)
    return add_delta_vs_vehicle(out, "macro_f1")


def evaluate_regression(feature_blocks: Dict[str, np.ndarray], targets: pd.DataFrame, split: np.ndarray, split_protocol: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    train_mask = split == "train"
    eval_masks = {"val": split == "val", "test": split == "test"}
    for target_col in ["future_peak_abs", "future_range", "future_mean_abs", "future_final", "future_slope"]:
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
                    ("reg", Ridge(alpha=50.0)),
                ]
            )
            reg.fit(x[good_train], y[good_train])
            for eval_name, mask in eval_masks.items():
                good_eval = mask & np.isfinite(y)
                if int(good_eval.sum()) < 10:
                    continue
                pred = reg.predict(x[good_eval])
                rows.append(
                    {
                        "split_protocol": split_protocol,
                        "task_type": "regression",
                        "target": target_col,
                        "feature_block": block,
                        "eval_split": eval_name,
                        "n_eval": int(good_eval.sum()),
                        "r2": float(r2_score(y[good_eval], pred)),
                        "mae": float(mean_absolute_error(y[good_eval], pred)),
                    }
                )
    out = pd.DataFrame(rows)
    return add_delta_vs_vehicle(out, "r2")


def add_delta_vs_vehicle(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    if df.empty or metric_col not in df.columns:
        return df
    base = df[df["feature_block"].eq("vehicle_only")][["split_protocol", "target", "eval_split", metric_col]].rename(columns={metric_col: "vehicle_metric"})
    out = df.merge(base, on=["split_protocol", "target", "eval_split"], how="left")
    out[f"delta_{metric_col}_minus_vehicle"] = out[metric_col] - out["vehicle_metric"]
    return out


def eta_squared(feature: np.ndarray, labels: np.ndarray) -> float:
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
        ss_between += float(len(vals) * (np.mean(vals) - grand) ** 2)
    return float(ss_between / ss_total)


def build_eta(physio200: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    cols = numeric_feature_columns(physio200, ("physio200_",))
    eta_targets = {
        "subject": physio200["subject"].astype(str).to_numpy(),
        "recording": physio200["recording"].astype(str).to_numpy(),
        "future_cluster4": targets["future_cluster4"].astype(str).to_numpy(),
        "high_future_abs_q75": targets["high_future_abs_q75"].astype(str).to_numpy(),
        "strong_steer_existing": targets["strong_steer_existing"].astype(str).to_numpy(),
    }
    rows: List[Dict[str, object]] = []
    for target, labels in eta_targets.items():
        for col in cols:
            e = eta_squared(pd.to_numeric(physio200[col], errors="coerce").to_numpy(dtype=float), labels)
            if np.isfinite(e):
                signal = next((s for s in SIGNALS if s in col), "unknown")
                rows.append({"target": target, "feature": col, "signal": signal, "eta2": float(e)})
    return pd.DataFrame(rows).sort_values(["target", "eta2"], ascending=[True, False])


def summarize_alignment(physio200: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, g in physio200.groupby("split"):
        row = {
            "split": split_name,
            "n": int(len(g)),
            "ok_rate": float(g["physio200_status"].eq("ok").mean()),
            "uses_post_observation_rate": float(g["physio200_uses_post_observation"].fillna(False).astype(bool).mean()),
            "baseline_rows_mean": float(pd.to_numeric(g["baseline_pre60_pre20_rows"], errors="coerce").mean()),
            "baseline_rows_p10": float(pd.to_numeric(g["baseline_pre60_pre20_rows"], errors="coerce").quantile(0.10)),
        }
        for win in EVENT_WINDOWS:
            row[f"{win}_rows_mean"] = float(pd.to_numeric(g[f"{win}_rows"], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_macro_f1(cls: pd.DataFrame) -> Path:
    path = FIGURES / "v254b_macro_f1_subject_disjoint_vs_subject_aware.png"
    sub = cls[
        cls["target"].isin(["future_cluster4", "high_future_abs_q75", "strong_steer_existing"])
        & cls["eval_split"].eq("test")
        & cls["feature_block"].isin(["vehicle_only", "physio200_curated", "vehicle_plus_physio200_curated"])
    ].copy()
    if sub.empty:
        return path
    labels = [f"{r.split_protocol}\n{r.target}" for r in sub[["split_protocol", "target"]].drop_duplicates().itertuples(index=False)]
    blocks = ["vehicle_only", "physio200_curated", "vehicle_plus_physio200_curated"]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(14, 6))
    keys = sub[["split_protocol", "target"]].drop_duplicates().to_dict("records")
    for i, block in enumerate(blocks):
        vals = []
        for key in keys:
            m = sub[
                sub["split_protocol"].eq(key["split_protocol"])
                & sub["target"].eq(key["target"])
                & sub["feature_block"].eq(block)
            ]
            vals.append(float(m["macro_f1"].iloc[0]) if len(m) else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=block)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("test macro-F1")
    ax.set_title("v254b: 200Hz事件相关生理表征的行为分类诊断")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_eta(eta: pd.DataFrame) -> Path:
    path = FIGURES / "v254b_top_eta2_physio200.png"
    targets = ["subject", "future_cluster4", "high_future_abs_q75", "strong_steer_existing"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()
    for ax, target in zip(axes, targets):
        top = eta[eta["target"].eq(target)].head(8).sort_values("eta2")
        labels = [f"{r.signal}\n{str(r.feature).split('_')[-1]}" for r in top.itertuples(index=False)]
        ax.barh(labels, top["eta2"].astype(float))
        ax.set_title(target)
        ax.set_xlabel("eta^2")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("v254b: 200Hz事件相关生理特征的描述性可分性")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for path in [V252_SCRIPT, V254A_SCRIPT, PHYSIO_INVENTORY]:
        if path.exists():
            rows.append({"path": str(path), "sha256": file_sha256(path), "bytes": int(path.stat().st_size)})
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "bytes": int(path.stat().st_size), "sha256": file_sha256(path)})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> str | None:
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
    cls: pd.DataFrame,
    reg: pd.DataFrame,
    eta: pd.DataFrame,
    figures: List[Path],
) -> None:
    report = REPORTS / "v254b_physio_200hz_event_representation_cn.md"
    lines: List[str] = []
    lines.append("# v254b 200Hz 连续生理事件相关表征审计")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("v254a 证明 1Hz/10Hz 表格窗口统计没有跨驾驶员行为增量。本轮改为从清洗后 200Hz 连续层直接抽取事件相关变化，并做每个事件自身锚点前 baseline 归一化。")
    lines.append("")
    lines.append("## 对齐覆盖")
    lines.append("")
    lines.append(alignment.to_markdown(index=False))
    lines.append("")
    lines.append("## 特征块")
    lines.append("")
    lines.append(block_audit.to_markdown(index=False))
    lines.append("")
    lines.append("## 行为分类诊断")
    lines.append("")
    show_cls = cls[
        cls["target"].isin(TARGETS_FOR_REPORT)
        & cls["eval_split"].eq("test")
        & cls["feature_block"].isin(["vehicle_only", "physio200_curated", "physio200_norm", "vehicle_plus_physio200_curated"])
    ].copy()
    lines.append(show_cls.to_markdown(index=False))
    lines.append("")
    lines.append("## 未来摘要回归诊断")
    lines.append("")
    show_reg = reg[
        reg["target"].isin(["future_peak_abs", "future_range", "future_mean_abs"])
        & reg["eval_split"].eq("test")
        & reg["feature_block"].isin(["vehicle_only", "physio200_curated", "physio200_norm", "vehicle_plus_physio200_curated"])
    ].copy()
    lines.append(show_reg.to_markdown(index=False))
    lines.append("")
    lines.append("## eta² top")
    lines.append("")
    lines.append(eta.groupby("target", group_keys=False).head(12).to_markdown(index=False))
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    lines.append("- subject_disjoint 是当前正式泛化口径。")
    lines.append("- subject_aware 只是诊断同一驾驶员历史样本可用时的个体化潜力。")
    lines.append("- 如果 subject_aware 明显好而 subject_disjoint 不好，说明生理更适合个体化/校准，不适合作为跨驾驶员直接泛化特征。")
    lines.append("- 如果 vehicle_plus_physio200_curated 仍不超过 vehicle_only，下一步应进入表示学习/时序编码，而不是继续手工统计。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    report.write_text("\n".join(lines), encoding="utf-8")


def build_guardrail(split_check: pd.DataFrame, physio200: pd.DataFrame, zip_test: str | None) -> Dict[str, object]:
    cross_split = int(split_check["split_nunique"].gt(1).sum()) if "split_nunique" in split_check.columns else 0
    return {
        "pass": bool(cross_split == 0 and not bool(physio200["physio200_uses_post_observation"].fillna(False).astype(bool).any()) and zip_test is None),
        "same_event_uid_cross_split_count": cross_split,
        "retrained_trajectory_model": False,
        "subject_aware_split_is_diagnostic_only": True,
        "test_used_for_feature_selection": False,
        "physio200_uses_post_observation_any": bool(physio200["physio200_uses_post_observation"].fillna(False).astype(bool).any()),
        "physio200_ok_rate": float(physio200["physio200_status"].eq("ok").mean()),
        "zip_testzip": zip_test,
    }


def main() -> None:
    print("[v254b] 200Hz physio event representation audit")
    clean_out_dir()
    np.random.seed(SEED)

    loaded = V252.load_fixed_inputs()
    manifest = loaded["data"].manifest.copy()
    sample_metrics = loaded["sample_metrics"].copy()
    split_check = loaded["split_check"].copy()
    split_main = manifest["split"].astype(str).to_numpy()
    split_subject_aware = make_subject_aware_split(manifest)

    physio200 = build_physio200_features(manifest)
    write_csv(physio200, TABLES / "v254b_event_physio200_features.csv")
    print(f"[v254b] saved physio200 feature table: {TABLES / 'v254b_event_physio200_features.csv'}", flush=True)
    targets = V254A.build_future_targets(loaded["y_true"], sample_metrics, split_main)
    targets, cluster_summary = V254A.add_future_clusters(loaded["y_true"], split_main, targets)
    feature_blocks, block_audit = make_feature_blocks(loaded["x_flat"], physio200, split_main)
    eta = build_eta(physio200, targets)
    alignment = summarize_alignment(physio200)

    cls_main = evaluate_classification(feature_blocks, targets, split_main, "subject_disjoint")
    reg_main = evaluate_regression(feature_blocks, targets, split_main, "subject_disjoint")
    cls_sa = evaluate_classification(feature_blocks, targets, split_subject_aware, "subject_aware")
    reg_sa = evaluate_regression(feature_blocks, targets, split_subject_aware, "subject_aware")
    cls = pd.concat([cls_main, cls_sa], ignore_index=True)
    reg = pd.concat([reg_main, reg_sa], ignore_index=True)

    split_table = pd.DataFrame({"row_index": np.arange(len(manifest)), "subject_disjoint_split": split_main, "subject_aware_split": split_subject_aware, "event_uid": manifest["event_uid"], "subject": manifest["subject"]})

    write_csv(physio200, TABLES / "v254b_event_physio200_features.csv")
    write_csv(targets, TABLES / "v254b_future_behavior_targets.csv")
    write_csv(cluster_summary, TABLES / "v254b_future_cluster_summary.csv")
    write_csv(split_table, TABLES / "v254b_split_protocol_table.csv")
    write_csv(alignment, TABLES / "v254b_alignment_coverage_summary.csv")
    write_csv(block_audit, TABLES / "v254b_feature_block_audit.csv")
    write_csv(eta, TABLES / "v254b_physio200_eta2_by_target_feature.csv")
    write_csv(cls, TABLES / "v254b_behavior_classification_diagnostics.csv")
    write_csv(reg, TABLES / "v254b_future_summary_regression_diagnostics.csv")

    figures = [plot_macro_f1(cls), plot_eta(eta)]
    write_input_hashes()
    write_file_inventory()
    write_report(alignment, block_audit, cls, reg, eta, figures)
    write_file_inventory()
    zip_test = make_zip()
    guardrail = build_guardrail(split_check, physio200, zip_test)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v254b guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    focus = cls[
        cls["split_protocol"].eq("subject_disjoint")
        & cls["eval_split"].eq("test")
        & cls["target"].eq("high_future_abs_q75")
    ].sort_values("macro_f1", ascending=False)
    print(f"[v254b] report={REPORTS / 'v254b_physio_200hz_event_representation_cn.md'}")
    print(f"[v254b] zip={ZIP_PATH}")
    if len(focus):
        r = focus.iloc[0]
        print(f"[v254b] subject_disjoint high_future_abs best block={r['feature_block']} macro_f1={float(r['macro_f1']):.4f}")


if __name__ == "__main__":
    main()
