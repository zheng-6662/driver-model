#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v293 physiology response visibility / latency audit.

本轮目标：
- v288-v292 已经证明：只用 observation 前 ECG/RESP/EDA 源信号做 selector/reranker/pairwise
  matching，都无法稳定改善差样本；
- v293 不再换 selector，而是检查一个更基础的问题：
  生理差异是否在 observation 前本来就不可见，而是在 observation 后 1-5 秒才显现；
- 这一步是诊断/路线判断，不把 post-observation 特征当作当前锚点可部署输入。

边界：
- pre windows 仍是可部署前信息；
- post windows 明确标为 diagnostic / waiting-policy evidence；
- feature screening 只用 train split；
- 分类器只用 train split 训练，val/test 只报告；
- test 不参与窗口、特征、阈值选择。
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
SCRIPTS = BASELINES / "scripts"

V285_SCRIPT = SCRIPTS / "stage03_v285_raw200_shape_state_route_gate_20260702.py"
V291_EVENT_TABLE = (
    BASELINES
    / "v291_multisignal_physio_supervised_probe_20260702"
    / "tables"
    / "v291_multisignal_event_table.csv"
)
V292_PAIR_TABLE = (
    BASELINES
    / "v292_source_physio_pairwise_candidate_ranker_20260702"
    / "tables"
    / "v292_pairwise_candidate_table.csv"
)
V292_GUARDRAIL = (
    BASELINES
    / "v292_source_physio_pairwise_candidate_ranker_20260702"
    / "logs"
    / "guardrail_check.json"
)

OUT = BASELINES / "v293_physio_response_visibility_latency_audit_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v293_physio_response_visibility_latency_audit_20260702_pack.zip"

SEED = 29302
TOP_N = 80

BASELINE_WINDOW = (-60.0, -20.0)
WINDOW_SPECS: Dict[str, Tuple[float, float]] = {
    "pre10_pre5": (-10.0, -5.0),
    "pre5_pre2": (-5.0, -2.0),
    "pre2_0": (-2.0, 0.0),
    "post0_1": (0.0, 1.0),
    "post0_2": (0.0, 2.0),
    "post0_3": (0.0, 3.0),
    "post0_5": (0.0, 5.0),
    "post1_3": (1.0, 3.0),
    "post2_5": (2.0, 5.0),
    "post5_10": (5.0, 10.0),
}

SIGNAL_SPECS: Dict[str, List[str]] = {
    "ecg": ["ECG_filt200", "ECG_raw200"],
    "resp": ["RESP_filt200", "RESP_raw200"],
    "eda_phasic": ["EDA_Phasic", "EDA_filt200", "EDA_raw200"],
    "eda_tonic": ["EDA_Tonic", "EDA_filt200", "EDA_raw200"],
    "emg": ["EMG_RMS", "EMG_filt200", "EMG_raw200"],
    "hr": ["HR_bpm"],
}

TARGETS = [
    "bad_top10",
    "bad_top10_vehicle_ambiguous",
    "vehicle_ambiguous",
    "candidate_pool_gain_gt_005",
    "candidate_pool_gain_gt_02",
]

# 路线判断只看与当前预测修正直接相关的核心目标；
# 辅助目标仍完整输出，但不让高基线比例的辅助标签单独翻转 guardrail。
CORE_DECISION_TARGETS = [
    "bad_top10",
    "bad_top10_vehicle_ambiguous",
    "candidate_pool_gain_gt_005",
]

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V285 = import_module_from_path("stage03_v285_for_v293", V285_SCRIPT)


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


def safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    return s.astype(str).str.lower().isin(["1", "true", "yes", "y"]).astype(int)


def finite(values: Iterable[object]) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def robust_center_scale(values: Iterable[object]) -> Tuple[float, float]:
    vals = finite(values)
    if vals.size == 0:
        return math.nan, math.nan
    center = float(np.median(vals))
    q25, q75 = np.quantile(vals, [0.25, 0.75])
    iqr = float(q75 - q25)
    mad = float(np.median(np.abs(vals - center)))
    std = float(np.std(vals))
    for scale in [iqr / 1.349 if iqr > 0 else math.nan, mad * 1.4826 if mad > 0 else math.nan, std]:
        if np.isfinite(scale) and scale > 1e-9:
            return center, float(scale)
    return center, math.nan


def robust_z(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    center, scale = robust_center_scale(baseline)
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 1e-9:
        return np.full(arr.shape, np.nan, dtype=float)
    out = (arr - center) / scale
    out[~np.isfinite(out)] = np.nan
    return out


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


def choose_signal(arrays: Dict[str, np.ndarray], candidates: List[str], baseline_idx: np.ndarray) -> Tuple[str, np.ndarray]:
    """在同一信号族中选 baseline 覆盖最好且非近常数的列。"""

    best_col = candidates[0]
    best_score = -1.0
    best_arr = np.full(0, np.nan)
    for col in candidates:
        arr = arrays.get(col)
        if arr is None:
            continue
        base = finite(arr[baseline_idx])
        score = float(base.size)
        if base.size >= 10:
            score += min(float(np.nanstd(base)), 10.0)
        if score > best_score:
            best_score = score
            best_col = col
            best_arr = arr
    return best_col, np.asarray(best_arr, dtype=float)


def window_phase(name: str) -> str:
    if name.startswith("pre"):
        return "pre"
    if name.startswith("post0") or name in {"post1_3"}:
        return "early_post"
    return "late_post"


def extract_window_features(times: np.ndarray, raw: np.ndarray, baseline: np.ndarray, start: float, end: float, prefix: str) -> Dict[str, float]:
    left = int(np.searchsorted(times, start, side="left"))
    right = int(np.searchsorted(times, end, side="right"))
    t = times[left:right]
    vals = raw[left:right]
    z = robust_z(vals, baseline)
    valid = finite(z)
    duration = max(0.0, end - start)
    out: Dict[str, float] = {
        f"{prefix}_rows": int(max(0, right - left)),
        f"{prefix}_valid_ratio": float(np.isfinite(vals).mean()) if len(vals) else 0.0,
    }
    if valid.size < 3:
        for metric in ["z_mean", "z_abs_mean", "z_std", "z_range", "z_p05", "z_p95", "z_last_minus_first", "z_slope", "line_length_per_s"]:
            out[f"{prefix}_{metric}"] = math.nan
        return out
    out[f"{prefix}_z_mean"] = float(np.nanmean(valid))
    out[f"{prefix}_z_abs_mean"] = float(np.nanmean(np.abs(valid)))
    out[f"{prefix}_z_std"] = float(np.nanstd(valid))
    out[f"{prefix}_z_range"] = float(np.nanmax(valid) - np.nanmin(valid))
    out[f"{prefix}_z_p05"] = float(np.nanpercentile(valid, 5))
    out[f"{prefix}_z_p95"] = float(np.nanpercentile(valid, 95))
    good = np.isfinite(z) & np.isfinite(t)
    if int(good.sum()) >= 2:
        zg = z[good]
        tg = t[good]
        out[f"{prefix}_z_last_minus_first"] = float(zg[-1] - zg[0])
        out[f"{prefix}_z_slope"] = slope(tg, zg)
        out[f"{prefix}_line_length_per_s"] = float(np.nansum(np.abs(np.diff(zg))) / max(duration, 1e-9))
    else:
        out[f"{prefix}_z_last_minus_first"] = math.nan
        out[f"{prefix}_z_slope"] = math.nan
        out[f"{prefix}_line_length_per_s"] = math.nan
    return out


def load_event_targets() -> pd.DataFrame:
    events = pd.read_csv(V291_EVENT_TABLE, encoding="utf-8-sig", low_memory=False)
    cols = [
        "event_uid",
        "subject",
        "recording",
        "split",
        "observation_s",
        "bad_top10",
        "vehicle_ambiguous",
        "bad_top10_vehicle_ambiguous",
        "method_oracle_gain_vs_latest",
        "method_oracle_gain_gt_002",
    ]
    events = events[[c for c in cols if c in events.columns]].drop_duplicates("event_uid").copy()
    for col in ["bad_top10", "vehicle_ambiguous", "bad_top10_vehicle_ambiguous", "method_oracle_gain_gt_002"]:
        if col in events.columns:
            events[col] = safe_bool_series(events[col])

    pair = pd.read_csv(
        V292_PAIR_TABLE,
        encoding="utf-8-sig",
        low_memory=False,
        usecols=lambda c: c in ["event_uid", "target_tail_rmse_v241", "latest_tail_rmse_v241"],
    )
    pair["target_tail_rmse_v241"] = pd.to_numeric(pair["target_tail_rmse_v241"], errors="coerce")
    pair["latest_tail_rmse_v241"] = pd.to_numeric(pair["latest_tail_rmse_v241"], errors="coerce")
    pool = (
        pair.groupby("event_uid", as_index=False)
        .agg(latest_rmse=("latest_tail_rmse_v241", "first"), candidate_pool_oracle_rmse=("target_tail_rmse_v241", "min"))
    )
    pool["candidate_pool_gain_vs_latest"] = pool["latest_rmse"] - pool["candidate_pool_oracle_rmse"]
    pool["candidate_pool_gain_gt_005"] = pool["candidate_pool_gain_vs_latest"].gt(0.05).astype(int)
    pool["candidate_pool_gain_gt_02"] = pool["candidate_pool_gain_vs_latest"].gt(0.02).astype(int)
    return events.merge(pool, on="event_uid", how="left", validate="one_to_one")


def build_visibility_features(events: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    inventory = V285.load_physio_inventory()
    samples = events.copy()
    samples["session_stamp"] = samples["recording"].map(V285.session_stamp_from_recording)
    rows: List[Dict[str, object]] = []
    missing_records = 0
    for (subject, session), group in samples.groupby(["subject", "session_stamp"], sort=False):
        path = inventory.get((str(subject), str(session)))
        if path is None or not path.exists():
            missing_records += 1
            for _, sample in group.iterrows():
                rows.append(
                    {
                        "event_uid": sample["event_uid"],
                        "subject": sample["subject"],
                        "recording": sample["recording"],
                        "split": sample["split"],
                        "observation_s": sample["observation_s"],
                        "v293_status": "missing_physio",
                    }
                )
            continue
        rec = V285.read_physio_recording(path)
        times = pd.to_numeric(rec["t_s"], errors="coerce").to_numpy(dtype=float)
        arrays = {
            col: pd.to_numeric(rec[col], errors="coerce").to_numpy(dtype=float)
            for col in rec.columns
            if col != "t_s"
        }
        for _, sample in group.iterrows():
            obs = float(sample["observation_s"])
            row: Dict[str, object] = {
                "event_uid": sample["event_uid"],
                "subject": sample["subject"],
                "recording": sample["recording"],
                "split": sample["split"],
                "observation_s": obs,
                "v293_status": "ok",
            }
            b_start = max(0.0, obs + BASELINE_WINDOW[0])
            b_end = max(0.0, obs + BASELINE_WINDOW[1])
            baseline_idx = (times >= b_start) & (times <= b_end)
            row["v293_baseline_rows"] = int(baseline_idx.sum())
            row["v293_recording_end_s"] = float(np.nanmax(times)) if len(times) else math.nan
            for signal, candidates in SIGNAL_SPECS.items():
                chosen_col, raw = choose_signal(arrays, candidates, baseline_idx)
                baseline = raw[baseline_idx] if len(raw) else np.array([], dtype=float)
                row[f"v293_{signal}_chosen_col"] = chosen_col
                row[f"v293_{signal}_baseline_valid_n"] = int(finite(baseline).size)
                for win_name, (offset_start, offset_end) in WINDOW_SPECS.items():
                    start = obs + offset_start
                    end = obs + offset_end
                    prefix = f"v293_{win_name}_{signal}"
                    row.update(extract_window_features(times, raw, baseline, start, end, prefix))
            rows.append(row)
    features = pd.DataFrame(rows)
    audit = {
        "event_n": int(events["event_uid"].nunique()),
        "feature_event_n": int(features["event_uid"].nunique()),
        "missing_recording_groups": int(missing_records),
        "ok_rate": float(features["v293_status"].eq("ok").mean()) if len(features) else 0.0,
        "uses_post_observation": True,
        "post_features_are_diagnostic_only": True,
    }
    return features, audit


def parse_feature_meta(feature: str) -> Tuple[str, str, str]:
    # v293_post0_2_resp_z_mean -> post0_2, resp, z_mean
    parts = feature.split("_")
    if len(parts) < 5 or parts[0] != "v293":
        return "unknown", "unknown", "unknown"
    # window names can be pre10_pre5 or post0_2.
    win = "_".join(parts[1:3])
    signal = parts[3]
    metric = "_".join(parts[4:])
    if parts[1].startswith("post") and parts[2] in {"1", "2", "3", "5", "10"}:
        win = "_".join(parts[1:3])
        signal = parts[3]
        metric = "_".join(parts[4:])
    return win, signal, metric


def abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 8:
        return 0.0
    xx = x[mask]
    yy = y[mask]
    if float(np.nanstd(xx)) <= 1e-12 or float(np.nanstd(yy)) <= 1e-12:
        return 0.0
    return float(abs(np.corrcoef(xx, yy)[0, 1]))


def screen_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    train = df["split"].astype(str).eq("train")
    rows: List[Dict[str, object]] = []
    for feature in feature_cols:
        x = pd.to_numeric(df.loc[train, feature], errors="coerce").to_numpy(dtype=float)
        finite_rate = float(np.isfinite(x).mean()) if len(x) else 0.0
        if finite_rate < 0.55 or float(np.nanstd(x)) <= 1e-10:
            continue
        target_scores = {}
        for target in TARGETS:
            if target in df.columns:
                y = pd.to_numeric(df.loc[train, target], errors="coerce").to_numpy(dtype=float)
                target_scores[f"corr_{target}"] = abs_corr(x, y)
        max_corr = max(target_scores.values()) if target_scores else 0.0
        win, signal, metric = parse_feature_meta(feature)
        rows.append(
            {
                "feature": feature,
                "window": win,
                "phase": window_phase(win),
                "signal": signal,
                "metric": metric,
                "finite_rate_train": finite_rate,
                "std_train": float(np.nanstd(x)),
                "max_abs_corr_train": max_corr,
                **target_scores,
            }
        )
    return pd.DataFrame(rows).sort_values(["max_abs_corr_train", "finite_rate_train"], ascending=[False, False])


def feature_sets_from_screen(screen: pd.DataFrame) -> Dict[str, List[str]]:
    sets: Dict[str, List[str]] = {}
    for phase in ["pre", "early_post", "late_post"]:
        sub = screen[screen["phase"].eq(phase)]
        if len(sub):
            sets[f"{phase}_top{TOP_N}"] = sub.head(TOP_N)["feature"].tolist()
    for win in WINDOW_SPECS:
        sub = screen[screen["window"].eq(win)]
        if len(sub):
            sets[f"window_{win}_top{min(TOP_N, 48)}"] = sub.head(min(TOP_N, 48))["feature"].tolist()
    if len(screen):
        sets[f"all_prepost_top120"] = screen.head(120)["feature"].tolist()
    return {k: v for k, v in sets.items() if len(v)}


def classifier_models() -> Dict[str, Pipeline]:
    return {
        "logreg_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)),
            ]
        ),
        "extra_trees_d5": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=240, max_depth=5, min_samples_leaf=6, class_weight="balanced", random_state=SEED, n_jobs=1)),
            ]
        ),
    }


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    yy = np.asarray(y)[mask]
    ss = np.asarray(score, dtype=float)[mask]
    if len(np.unique(yy)) < 2:
        return math.nan
    return float(roc_auc_score(yy, ss))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    yy = np.asarray(y)[mask]
    ss = np.asarray(score, dtype=float)[mask]
    if len(np.unique(yy)) < 2:
        return math.nan
    return float(average_precision_score(yy, ss))


def run_visibility_classifiers(df: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    train = df["split"].astype(str).eq("train")
    for set_name, cols in feature_sets.items():
        phase = "mixed"
        if set_name.startswith("window_"):
            win_name = set_name[len("window_") :].rsplit("_top", 1)[0]
            phase = window_phase(win_name)
        elif set_name.startswith("pre"):
            phase = "pre"
        elif set_name.startswith("early_post"):
            phase = "early_post"
        elif set_name.startswith("late_post"):
            phase = "late_post"
        X = df[cols].replace([np.inf, -np.inf], np.nan)
        for target in TARGETS:
            if target not in df.columns:
                continue
            y = safe_bool_series(df[target]).to_numpy(dtype=int)
            if len(np.unique(y[train.to_numpy()])) < 2:
                continue
            for model_name, model in classifier_models().items():
                print(f"[v293] classifier set={set_name} target={target} model={model_name} feature_n={len(cols)}")
                model.fit(X.loc[train], y[train.to_numpy()])
                if hasattr(model, "predict_proba"):
                    score = model.predict_proba(X)[:, 1]
                else:
                    score = model.decision_function(X)
                for split in ["val", "test"]:
                    m = df["split"].astype(str).eq(split).to_numpy()
                    rows.append(
                        {
                            "feature_set": set_name,
                            "phase": phase,
                            "target": target,
                            "model_name": model_name,
                            "split": split,
                            "n": int(m.sum()),
                            "positive_rate": float(np.mean(y[m])) if int(m.sum()) else math.nan,
                            "auc": safe_auc(y[m], score[m]),
                            "average_precision": safe_ap(y[m], score[m]),
                            "feature_n": int(len(cols)),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_window_screen(screen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (phase, window, signal), sub in screen.groupby(["phase", "window", "signal"], dropna=False):
        rows.append(
            {
                "phase": phase,
                "window": window,
                "signal": signal,
                "feature_n": int(len(sub)),
                "max_corr": float(sub["max_abs_corr_train"].max()),
                "mean_top10_corr": float(sub.sort_values("max_abs_corr_train", ascending=False).head(10)["max_abs_corr_train"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["phase", "max_corr"], ascending=[True, False])


def decision_table(cls: pd.DataFrame) -> pd.DataFrame:
    def best(split: str, target: str, phase: str | None = None) -> float:
        sub = cls[cls["split"].eq(split) & cls["target"].eq(target)].copy()
        if phase is not None:
            sub = sub[sub["phase"].eq(phase)]
        if sub.empty:
            return math.nan
        return float(pd.to_numeric(sub["auc"], errors="coerce").max())

    rows: List[Dict[str, object]] = []
    for target in TARGETS:
        pre_test = best("test", target, "pre")
        early_test = best("test", target, "early_post")
        late_test = best("test", target, "late_post")
        mixed_test = best("test", target, None)
        rows.append(
            {
                "target": target,
                "pre_test_best_auc": pre_test,
                "early_post_test_best_auc": early_test,
                "late_post_test_best_auc": late_test,
                "any_test_best_auc": mixed_test,
                "early_minus_pre": early_test - pre_test if np.isfinite(early_test) and np.isfinite(pre_test) else math.nan,
                "pre_visible": bool(np.isfinite(pre_test) and pre_test >= 0.60),
                "post_visibility_gain": bool(np.isfinite(early_test) and np.isfinite(pre_test) and early_test - pre_test >= 0.08 and early_test >= 0.60),
            }
        )
    out = pd.DataFrame(rows)
    out["is_core_decision_target"] = out["target"].isin(CORE_DECISION_TARGETS)
    main_bad = out[out["target"].eq("bad_top10")]
    candidate_gain = out[out["target"].eq("candidate_pool_gain_gt_005")]
    weak_subgroup = out[out["target"].eq("bad_top10_vehicle_ambiguous")]
    out["pre_route_supported_now"] = bool(
        (not main_bad.empty and bool(main_bad["pre_visible"].iloc[0]))
        or (not candidate_gain.empty and bool(candidate_gain["pre_visible"].iloc[0]))
    )
    out["pre_weak_subgroup_signal_exists"] = bool(
        not weak_subgroup.empty and bool(weak_subgroup["pre_visible"].iloc[0])
    )
    out["post_wait_route_supported_diagnostic"] = bool(
        not main_bad.empty and bool(main_bad["post_visibility_gain"].iloc[0])
    )
    return out


def plot_phase_auc(cls: pd.DataFrame) -> Path:
    data = (
        cls[cls["split"].eq("test")]
        .groupby(["target", "phase"], as_index=False)["auc"]
        .max()
        .pivot(index="target", columns="phase", values="auc")
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    phases = [p for p in ["pre", "early_post", "late_post", "mixed"] if p in data.columns]
    x = np.arange(len(data.index))
    width = 0.18
    for i, phase in enumerate(phases):
        ax.bar(x + (i - (len(phases) - 1) / 2) * width, data[phase].values, width=width, label=phase)
    ax.axhline(0.60, color="#E15759", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(data.index, rotation=20, ha="right")
    ax.set_ylabel("test AUC")
    ax.set_title("v293 physiology visibility by phase")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGURES / "v293_phase_test_auc.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_window_corr(window_summary: pd.DataFrame) -> Path:
    top = window_summary.sort_values("max_corr", ascending=False).head(30).copy()
    top["label"] = top["window"] + " | " + top["signal"]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["label"], top["max_corr"], color="#4E79A7")
    ax.invert_yaxis()
    ax.set_xlabel("train max abs corr with targets")
    ax.set_title("v293 top window/signal screen signals")
    fig.tight_layout()
    path = FIGURES / "v293_window_signal_screen_corr.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_quality(features: pd.DataFrame) -> Path:
    quality_cols = [c for c in features.columns if c.endswith("_valid_ratio") and "v293_" in c]
    rows = []
    for col in quality_cols:
        parts = col.split("_")
        if len(parts) >= 5:
            win = "_".join(parts[1:3])
            signal = parts[3]
            rows.append({"window": win, "signal": signal, "valid_ratio_mean": float(pd.to_numeric(features[col], errors="coerce").mean())})
    data = pd.DataFrame(rows)
    if data.empty:
        data = pd.DataFrame({"window": ["none"], "signal": ["none"], "valid_ratio_mean": [0.0]})
    pivot = data.groupby("window", as_index=False)["valid_ratio_mean"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(pivot["window"], pivot["valid_ratio_mean"], color="#59A14F")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean valid ratio")
    ax.set_title("v293 window physiological coverage")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = FIGURES / "v293_window_valid_ratio.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def markdown_table(df: pd.DataFrame, cols: List[str], n: int | None = None) -> str:
    if df is None or df.empty:
        return "- 无记录。"
    view = df[[c for c in cols if c in df.columns]].copy()
    if n is not None:
        view = view.head(n)
    return view.to_markdown(index=False)


def write_report(features: pd.DataFrame, screen: pd.DataFrame, window_summary: pd.DataFrame, cls: pd.DataFrame, decision: pd.DataFrame, audit: Dict[str, object], guardrail: Dict[str, object]) -> Path:
    path = REPORTS / "v293_physio_response_visibility_latency_audit_cn.md"
    lines: List[str] = []
    lines.append("# v293 physiology response visibility / latency audit")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- v292 已经说明好候选存在，但 observation 前源生理 pairwise matching 不能稳定选对。")
    lines.append("- v293 检查生理差异是否可能主要出现在 observation 后短时间窗。")
    lines.append("- post 特征只作为 diagnostic / waiting-policy evidence，不作为当前锚点部署输入。")
    lines.append("")
    lines.append("## decision")
    lines.append("")
    lines.append(markdown_table(decision, ["target", "is_core_decision_target", "pre_test_best_auc", "early_post_test_best_auc", "late_post_test_best_auc", "early_minus_pre", "pre_visible", "post_visibility_gain", "pre_weak_subgroup_signal_exists", "pre_route_supported_now", "post_wait_route_supported_diagnostic"]))
    lines.append("")
    lines.append("## classifier top results")
    lines.append("")
    top = cls.sort_values(["target", "split", "auc"], ascending=[True, True, False]).groupby(["target", "split"]).head(8)
    lines.append(markdown_table(top, ["target", "split", "feature_set", "phase", "model_name", "n", "positive_rate", "auc", "average_precision", "feature_n"], 120))
    lines.append("")
    lines.append("## window/signal train screen")
    lines.append("")
    lines.append(markdown_table(window_summary, ["phase", "window", "signal", "feature_n", "max_corr", "mean_top10_corr"], 80))
    lines.append("")
    lines.append("## top screened features")
    lines.append("")
    lines.append(markdown_table(screen, ["feature", "phase", "window", "signal", "metric", "finite_rate_train", "max_abs_corr_train"], 50))
    lines.append("")
    lines.append("## audit")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(audit, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## guardrail")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 判断")
    lines.append("")
    if bool(guardrail.get("pre_route_supported_now", False)):
        lines.append("- pre-observation 生理出现可用可见性信号，需要回到部署模型复核。")
    elif bool(guardrail.get("post_wait_route_supported_diagnostic", False)):
        lines.append("- observation 后短窗出现比 pre 明显更强的生理可见性，后续应考虑 wait/late-observation 策略，而不是当前锚点前预测。")
    else:
        lines.append("- pre 和 early-post 都没有形成足够强的泛化可见性；当前失败不只是某个 pre 窗口没取对。")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_input_hashes() -> None:
    rows = []
    for name, path in {
        "v291_event_table": V291_EVENT_TABLE,
        "v292_pair_table": V292_PAIR_TABLE,
        "v292_guardrail": V292_GUARDRAIL,
        "v285_script": V285_SCRIPT,
    }.items():
        rows.append({"name": name, "path": str(path), "exists": path.exists(), "sha256": file_sha256(path) if path.exists() else None})
    write_csv(pd.DataFrame(rows), LOGS / "input_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    for p in OUT.rglob("*"):
        if p.is_file():
            rows.append({"relative_path": str(p.relative_to(OUT)), "size_bytes": p.stat().st_size})
    write_csv(pd.DataFrame(rows).sort_values("relative_path"), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in OUT.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(OUT.parent))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip() is None


def main() -> None:
    print("[v293] 目的：比较 observation 前后生理可见性，判断生理信息是否延迟出现。")
    clean_out_dir()
    events = load_event_targets()
    features, audit = build_visibility_features(events)
    data = events.merge(features, on=["event_uid", "subject", "recording", "split", "observation_s"], how="left", validate="one_to_one")
    feature_cols = [
        c
        for c in data.columns
        if c.startswith("v293_")
        and any(c.endswith(metric) for metric in ["z_mean", "z_abs_mean", "z_std", "z_range", "z_p05", "z_p95", "z_last_minus_first", "z_slope", "line_length_per_s"])
    ]
    screen = screen_features(data, feature_cols)
    sets = feature_sets_from_screen(screen)
    cls = run_visibility_classifiers(data, sets)
    window_summary = summarize_window_screen(screen)
    decision = decision_table(cls)
    bad_decision = decision[decision["target"].eq("bad_top10")].copy()
    subgroup_decision = decision[decision["target"].eq("bad_top10_vehicle_ambiguous")].copy()
    candidate_decision = decision[decision["target"].eq("candidate_pool_gain_gt_005")].copy()

    guardrail = {
        "pass": True,
        "event_n": int(data["event_uid"].nunique()),
        "feature_n": int(len(feature_cols)),
        "screen_feature_n": int(len(screen)),
        "feature_set_n": int(len(sets)),
        "ok_rate": float(data["v293_status"].eq("ok").mean()) if "v293_status" in data.columns else 0.0,
        "uses_post_observation": True,
        "post_features_are_diagnostic_only": True,
        "guardrail_core_targets": CORE_DECISION_TARGETS,
        "pre_route_supported_now": bool(decision["pre_route_supported_now"].iloc[0]),
        "pre_weak_subgroup_signal_exists": bool(decision["pre_weak_subgroup_signal_exists"].iloc[0]),
        "post_wait_route_supported_diagnostic": bool(decision["post_wait_route_supported_diagnostic"].iloc[0]),
        "best_pre_badtop10_test_auc": float(decision.loc[decision["target"].eq("bad_top10"), "pre_test_best_auc"].iloc[0]),
        "best_early_post_badtop10_test_auc": float(decision.loc[decision["target"].eq("bad_top10"), "early_post_test_best_auc"].iloc[0]),
        "best_pre_badtop10_vehicle_ambiguous_test_auc": float(subgroup_decision["pre_test_best_auc"].iloc[0]) if not subgroup_decision.empty else math.nan,
        "best_early_post_badtop10_vehicle_ambiguous_test_auc": float(subgroup_decision["early_post_test_best_auc"].iloc[0]) if not subgroup_decision.empty else math.nan,
        "best_pre_candidate_gain_test_auc": float(decision.loc[decision["target"].eq("candidate_pool_gain_gt_005"), "pre_test_best_auc"].iloc[0]),
        "best_early_post_candidate_gain_test_auc": float(decision.loc[decision["target"].eq("candidate_pool_gain_gt_005"), "early_post_test_best_auc"].iloc[0]),
        "test_used_for_feature_screen_or_threshold": False,
        "v292_route_viable_now": json.loads(V292_GUARDRAIL.read_text(encoding="utf-8")).get("route_viable_now", False) if V292_GUARDRAIL.exists() else False,
    }

    write_csv(data, TABLES / "v293_prepost_physio_visibility_features.csv")
    write_csv(screen, TABLES / "v293_train_only_feature_screen.csv")
    write_csv(pd.DataFrame([{"feature_set": k, "feature_n": len(v), "features": json.dumps(v, ensure_ascii=False)} for k, v in sets.items()]), TABLES / "v293_feature_sets.csv")
    write_csv(window_summary, TABLES / "v293_window_signal_screen_summary.csv")
    write_csv(cls, TABLES / "v293_visibility_classifier_summary.csv")
    write_csv(decision, TABLES / "v293_visibility_decision.csv")
    plot_phase_auc(cls)
    plot_window_corr(window_summary)
    plot_quality(features)
    write_report(features, screen, window_summary, cls, decision, audit, guardrail)
    write_input_hashes()
    guardrail["zip_testzip"] = False
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    guardrail["zip_testzip"] = bool(make_zip())
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()
    print(f"[v293] report={REPORTS / 'v293_physio_response_visibility_latency_audit_cn.md'}")
    print(f"[v293] zip={ZIP_PATH}")
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
