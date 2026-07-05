from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
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
from sklearn.metrics import roc_auc_score


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
OUT = BASELINES / "v297_subject_style_stability_audit_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v297_subject_style_stability_audit_20260702_pack.zip"

V249_NPZ = BASELINES / "v249_shape_aware_curve_model_20260630" / "v249_shape_aware_predictions.npz"
V293_FEATURES = (
    BASELINES
    / "v293_physio_response_visibility_latency_audit_20260702"
    / "tables"
    / "v293_prepost_physio_visibility_features.csv"
)
V295_GUARDRAIL = BASELINES / "v295_wait1_direct_residual_physio_20260702" / "logs" / "guardrail_check.json"
THIS_SCRIPT = Path(__file__).resolve()

SEED = 20260702
PAIR_SAMPLE_N = 250_000

NUMERIC_STYLE_TARGETS = [
    "true_peak_abs",
    "true_peak_time_s",
    "true_final_delta",
    "true_range",
    "true_line_length",
    "true_tail_mean_abs",
    "true_early_peak_abs",
    "true_late_peak_abs",
    "v249_rmse",
    "v249_tail_rmse",
    "v249_residual_mean",
    "v249_residual_final",
    "v249_peak_abs_error",
]

KEY_STYLE_TARGETS = [
    "true_peak_abs",
    "true_peak_time_s",
    "true_final_delta",
    "true_line_length",
    "v249_rmse",
    "v249_tail_rmse",
    "v249_residual_mean",
]

BINARY_TARGETS = [
    "true_reverse_flag",
    "true_multi_correction_flag",
    "true_late_peak_flag",
    "bad_top10",
    "bad_top10_vehicle_ambiguous",
]


def ensure_dirs() -> None:
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


def finite(values: Iterable[object]) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def safe_mean(values: Iterable[object]) -> float:
    arr = finite(values)
    if arr.size == 0:
        return math.nan
    return float(np.mean(arr))


def safe_rmse(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(pred)
    if int(mask.sum()) == 0:
        return math.nan
    return float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2)))


def eta_squared(values: Iterable[object], groups: Iterable[object]) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    g = pd.Series(groups).astype(str)
    mask = x.notna() & g.notna()
    if int(mask.sum()) < 8:
        return math.nan
    x = x[mask].astype(float)
    g = g[mask].astype(str)
    total = float(((x - x.mean()) ** 2).sum())
    if total <= 1e-12:
        return 0.0
    means = x.groupby(g).mean()
    counts = g.value_counts()
    between = 0.0
    for key, mean in means.items():
        between += float(counts[key]) * float((mean - x.mean()) ** 2)
    return float(max(0.0, min(1.0, between / total)))


def sign_with_deadzone(x: np.ndarray, threshold: float) -> np.ndarray:
    out = np.zeros(len(x), dtype=int)
    out[x > threshold] = 1
    out[x < -threshold] = -1
    return out


def count_sign_changes(signs: np.ndarray) -> int:
    nz = signs[signs != 0]
    if len(nz) < 2:
        return 0
    return int(np.sum(nz[1:] != nz[:-1]))


def count_local_extrema(y: np.ndarray, min_delta: float) -> int:
    if len(y) < 5:
        return 0
    dy = np.diff(y)
    dy[np.abs(dy) < min_delta] = 0.0
    signs = sign_with_deadzone(dy, 0.0)
    return count_sign_changes(signs)


def parse_event_order(event_uid: str, recording: str, observation_s: float) -> Tuple[str, int, float]:
    rec = str(recording)
    match = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", rec)
    rec_stamp = match.group(1) if match else rec
    suffix = re.search(r"_(\d{3,})$", str(event_uid))
    event_index = int(suffix.group(1)) if suffix else 0
    try:
        obs = float(observation_s)
    except (TypeError, ValueError):
        obs = math.nan
    return rec_stamp, event_index, obs


def load_event_descriptors() -> pd.DataFrame:
    meta = pd.read_csv(V293_FEATURES, usecols=lambda c: c in [
        "event_uid",
        "subject",
        "recording",
        "split",
        "observation_s",
        "bad_top10",
        "vehicle_ambiguous",
        "bad_top10_vehicle_ambiguous",
        "candidate_pool_gain_gt_005",
    ])
    meta = meta.drop_duplicates("event_uid").copy()

    with np.load(V249_NPZ, allow_pickle=False) as z:
        event_uid = z["event_uid"].astype(str)
        split = z["split"].astype(str)
        delay_ms = z["delay_ms"].astype(int)
        y_true = z["y_true_steering_delta"].astype(float)
        pred = z["pred_v249_best_shape_steering_delta"].astype(float)
        grid = z["future_grid_s"].astype(float)
        valid = z["original_remaining_valid"].astype(bool)

    idx = np.where(delay_ms == 0)[0]
    rows: List[Dict[str, object]] = []
    for i in idx:
        y = y_true[i].copy()
        p = pred[i].copy()
        m = valid[i].copy()
        y[~m] = np.nan
        p[~m] = np.nan
        abs_y = np.abs(y)
        if np.all(~np.isfinite(abs_y)):
            peak_i = 0
            peak_abs = math.nan
            peak_signed = math.nan
            peak_t = math.nan
        else:
            peak_i = int(np.nanargmax(abs_y))
            peak_abs = float(abs_y[peak_i])
            peak_signed = float(y[peak_i])
            peak_t = float(grid[peak_i])
        threshold = max(0.15, 0.20 * peak_abs) if np.isfinite(peak_abs) else 0.15
        signs = sign_with_deadzone(np.nan_to_num(y, nan=0.0), threshold)
        reverse_flag = int(count_sign_changes(signs) > 0)
        extrema_n = count_local_extrema(np.nan_to_num(y, nan=0.0), min_delta=max(0.03, 0.04 * peak_abs if np.isfinite(peak_abs) else 0.03))
        tail = grid >= 1.0
        early = grid <= 1.0
        late = grid > 1.0
        residual = y - p
        row: Dict[str, object] = {
            "event_uid": str(event_uid[i]),
            "split_npz": str(split[i]),
            "delay_ms": int(delay_ms[i]),
            "true_peak_abs": peak_abs,
            "true_peak_signed": peak_signed,
            "true_peak_time_s": peak_t,
            "true_final_delta": float(y[np.where(np.isfinite(y))[0][-1]]) if np.isfinite(y).any() else math.nan,
            "true_range": float(np.nanmax(y) - np.nanmin(y)) if np.isfinite(y).any() else math.nan,
            "true_line_length": float(np.nansum(np.abs(np.diff(y)))) if np.isfinite(y).any() else math.nan,
            "true_tail_mean_abs": safe_mean(np.abs(y[tail])),
            "true_early_peak_abs": float(np.nanmax(np.abs(y[early]))) if np.isfinite(y[early]).any() else math.nan,
            "true_late_peak_abs": float(np.nanmax(np.abs(y[late]))) if np.isfinite(y[late]).any() else math.nan,
            "true_reverse_flag": reverse_flag,
            "true_multi_correction_flag": int(extrema_n >= 2),
            "true_extrema_n": int(extrema_n),
            "true_late_peak_flag": int(np.isfinite(peak_t) and peak_t > 1.0),
            "true_direction": "right" if np.isfinite(peak_signed) and peak_signed > 0 else ("left" if np.isfinite(peak_signed) and peak_signed < 0 else "flat"),
            "v249_rmse": safe_rmse(y, p),
            "v249_tail_rmse": safe_rmse(y[tail], p[tail]),
            "v249_residual_mean": safe_mean(residual),
            "v249_residual_final": float(residual[np.where(np.isfinite(residual))[0][-1]]) if np.isfinite(residual).any() else math.nan,
            "v249_peak_abs_error": float(np.nanmax(np.abs(p)) - peak_abs) if np.isfinite(peak_abs) and np.isfinite(p).any() else math.nan,
        }
        rows.append(row)

    desc = pd.DataFrame(rows)
    data = desc.merge(meta, on="event_uid", how="left", validate="one_to_one")
    if "split" in data.columns:
        data["split_consistent"] = data["split"].astype(str).eq(data["split_npz"].astype(str))
    else:
        data["split"] = data["split_npz"]
        data["split_consistent"] = True
    for col in ["bad_top10", "vehicle_ambiguous", "bad_top10_vehicle_ambiguous", "candidate_pool_gain_gt_005"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)
    order = data.apply(lambda r: parse_event_order(r["event_uid"], r["recording"], r["observation_s"]), axis=1)
    data["recording_stamp"] = [x[0] for x in order]
    data["event_index_in_uid"] = [x[1] for x in order]
    data["order_observation_s"] = [x[2] for x in order]
    data = data.sort_values(["subject", "recording_stamp", "event_index_in_uid", "order_observation_s", "event_uid"]).reset_index(drop=True)
    return data


def assign_oracle_labels(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    train = out["split"].astype(str).eq("train")
    q_peak = out.loc[train, "true_peak_abs"].quantile([0.33, 0.66]).to_numpy()
    q_line = out.loc[train, "true_line_length"].quantile([0.66]).iloc[0]
    q_rmse = out.loc[train, "v249_rmse"].quantile([0.80, 0.90]).to_numpy()
    out["oracle_strength_label"] = pd.cut(
        out["true_peak_abs"],
        bins=[-np.inf, q_peak[0], q_peak[1], np.inf],
        labels=["weak", "medium", "strong"],
    ).astype(str)
    out["oracle_timing_label"] = np.where(out["true_peak_time_s"] > 1.0, "late_peak", "early_peak")
    out["oracle_error_label"] = pd.cut(
        out["v249_rmse"],
        bins=[-np.inf, q_rmse[0], q_rmse[1], np.inf],
        labels=["normal_error", "high_error", "very_high_error"],
    ).astype(str)
    shape = np.full(len(out), "single_or_smooth", dtype=object)
    shape[out["true_reverse_flag"].astype(bool).to_numpy()] = "reverse"
    shape[out["true_multi_correction_flag"].astype(bool).to_numpy()] = "multi_correction"
    shape[(out["true_line_length"] > q_line) & (out["true_reverse_flag"].eq(0)) & (out["true_multi_correction_flag"].eq(0))] = "large_smooth"
    out["oracle_shape_label"] = shape
    out["oracle_direction_label"] = out["true_direction"].astype(str)
    return out


def compute_eta_table(data: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    groups = [("subject", "subject"), ("recording", "recording"), ("subject_recording", "subject_recording")]
    data = data.copy()
    data["subject_recording"] = data["subject"].astype(str) + "|" + data["recording"].astype(str)
    for split_name in ["train", "val", "test", "all"]:
        subset = data if split_name == "all" else data[data["split"].astype(str).eq(split_name)]
        for target in NUMERIC_STYLE_TARGETS:
            for group_name, group_col in groups:
                rows.append(
                    {
                        "split": split_name,
                        "target": target,
                        "group": group_name,
                        "eta_squared": eta_squared(subset[target], subset[group_col]) if len(subset) else math.nan,
                        "n": int(pd.to_numeric(subset[target], errors="coerce").notna().sum()),
                        "group_n": int(subset[group_col].nunique()) if len(subset) else 0,
                    }
                )
    return pd.DataFrame(rows)


def subject_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, sub in data.groupby("subject", dropna=False):
        row: Dict[str, object] = {
            "subject": subject,
            "event_n": int(len(sub)),
            "recording_n": int(sub["recording"].nunique()),
            "train_n": int(sub["split"].astype(str).eq("train").sum()),
            "val_n": int(sub["split"].astype(str).eq("val").sum()),
            "test_n": int(sub["split"].astype(str).eq("test").sum()),
        }
        for target in KEY_STYLE_TARGETS:
            row[f"{target}_mean"] = safe_mean(sub[target])
            row[f"{target}_std"] = float(np.nanstd(pd.to_numeric(sub[target], errors="coerce"))) if len(sub) else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("event_n", ascending=False).reset_index(drop=True)


def pair_distance_summary(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = data["split"].astype(str).eq("train")
    means = data.loc[train, KEY_STYLE_TARGETS].apply(pd.to_numeric, errors="coerce").mean(axis=0)
    stds = data.loc[train, KEY_STYLE_TARGETS].apply(pd.to_numeric, errors="coerce").std(axis=0).replace(0, np.nan)
    z = (data[KEY_STYLE_TARGETS].apply(pd.to_numeric, errors="coerce") - means) / stds
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    n = len(data)
    rng = np.random.default_rng(SEED)
    total_pairs = n * (n - 1) // 2
    sample_n = min(PAIR_SAMPLE_N, total_pairs)
    if sample_n == total_pairs:
        pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=int)
    else:
        left = rng.integers(0, n, size=sample_n * 2)
        right = rng.integers(0, n, size=sample_n * 2)
        mask = left != right
        left = left[mask][:sample_n]
        right = right[mask][:sample_n]
        lo = np.minimum(left, right)
        hi = np.maximum(left, right)
        pairs = np.column_stack([lo, hi])
    dist = np.sqrt(np.mean((z[pairs[:, 0]] - z[pairs[:, 1]]) ** 2, axis=1))
    pair_df = pd.DataFrame(
        {
            "i": pairs[:, 0],
            "j": pairs[:, 1],
            "distance": dist,
            "same_subject": data.iloc[pairs[:, 0]]["subject"].to_numpy() == data.iloc[pairs[:, 1]]["subject"].to_numpy(),
            "same_recording": data.iloc[pairs[:, 0]]["recording"].to_numpy() == data.iloc[pairs[:, 1]]["recording"].to_numpy(),
            "same_split": data.iloc[pairs[:, 0]]["split"].to_numpy() == data.iloc[pairs[:, 1]]["split"].to_numpy(),
        }
    )
    rows: List[Dict[str, object]] = []
    for name, mask in [
        ("same_subject", pair_df["same_subject"].to_numpy()),
        ("different_subject", ~pair_df["same_subject"].to_numpy()),
        ("same_recording", pair_df["same_recording"].to_numpy()),
        ("different_recording", ~pair_df["same_recording"].to_numpy()),
    ]:
        d = pair_df.loc[mask, "distance"]
        rows.append(
            {
                "pair_group": name,
                "pair_n": int(mask.sum()),
                "distance_mean": float(d.mean()) if len(d) else math.nan,
                "distance_median": float(d.median()) if len(d) else math.nan,
                "distance_p25": float(d.quantile(0.25)) if len(d) else math.nan,
                "distance_p75": float(d.quantile(0.75)) if len(d) else math.nan,
            }
        )
    summary = pd.DataFrame(rows)
    same = summary.loc[summary["pair_group"].eq("same_subject"), "distance_mean"]
    diff = summary.loc[summary["pair_group"].eq("different_subject"), "distance_mean"]
    if len(same) and len(diff) and float(diff.iloc[0]) > 1e-12:
        ratio = float(same.iloc[0] / diff.iloc[0])
    else:
        ratio = math.nan
    summary["same_subject_mean_distance_ratio"] = ratio
    return summary, pair_df


def rolling_history_predictability(data: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    ordered = data.sort_values(["subject", "recording_stamp", "event_index_in_uid", "order_observation_s", "event_uid"]).copy()
    for target in NUMERIC_STYLE_TARGETS:
        global_train_mean = safe_mean(ordered.loc[ordered["split"].astype(str).eq("train"), target])
        preds = []
        actual = []
        splits = []
        hist_ns = []
        subjects = []
        for subject, sub in ordered.groupby("subject", sort=False):
            values: List[float] = []
            for _, row in sub.iterrows():
                y = pd.to_numeric(pd.Series([row[target]]), errors="coerce").iloc[0]
                pred = float(np.mean(values)) if len(values) else global_train_mean
                preds.append(pred)
                actual.append(float(y) if np.isfinite(y) else math.nan)
                splits.append(str(row["split"]))
                hist_ns.append(len(values))
                subjects.append(subject)
                if np.isfinite(y):
                    values.append(float(y))
        tmp = pd.DataFrame({"actual": actual, "pred": preds, "split": splits, "history_n": hist_ns, "subject": subjects})
        tmp["global_pred"] = global_train_mean
        for split_name in ["train", "val", "test", "all"]:
            for min_hist in [0, 1, 3, 5]:
                sub = tmp if split_name == "all" else tmp[tmp["split"].eq(split_name)]
                sub = sub[sub["history_n"] >= min_hist].copy()
                m = np.isfinite(sub["actual"]) & np.isfinite(sub["pred"])
                if int(m.sum()) < 5:
                    rows.append(
                        {
                            "target": target,
                            "split": split_name,
                            "min_history_n": min_hist,
                            "n": int(m.sum()),
                            "rmse_history": math.nan,
                            "rmse_global": math.nan,
                            "rmse_improvement_vs_global": math.nan,
                            "r2_history_vs_global": math.nan,
                        }
                    )
                    continue
                a = sub.loc[m, "actual"].to_numpy(dtype=float)
                p = sub.loc[m, "pred"].to_numpy(dtype=float)
                g = sub.loc[m, "global_pred"].to_numpy(dtype=float)
                rmse_h = float(np.sqrt(np.mean((a - p) ** 2)))
                rmse_g = float(np.sqrt(np.mean((a - g) ** 2)))
                rows.append(
                    {
                        "target": target,
                        "split": split_name,
                        "min_history_n": min_hist,
                        "n": int(m.sum()),
                        "rmse_history": rmse_h,
                        "rmse_global": rmse_g,
                        "rmse_improvement_vs_global": float(rmse_g - rmse_h),
                        "relative_rmse_improvement": float((rmse_g - rmse_h) / rmse_g) if rmse_g > 1e-12 else math.nan,
                        "r2_history_vs_global": float(1.0 - np.mean((a - p) ** 2) / max(np.mean((a - g) ** 2), 1e-12)),
                    }
                )
    return pd.DataFrame(rows)


def binary_history_auc(data: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    ordered = data.sort_values(["subject", "recording_stamp", "event_index_in_uid", "order_observation_s", "event_uid"]).copy()
    for target in BINARY_TARGETS:
        if target not in ordered.columns:
            continue
        global_rate = float(pd.to_numeric(ordered.loc[ordered["split"].astype(str).eq("train"), target], errors="coerce").mean())
        preds = []
        actual = []
        splits = []
        hist_ns = []
        for _, sub in ordered.groupby("subject", sort=False):
            values: List[float] = []
            for _, row in sub.iterrows():
                y = pd.to_numeric(pd.Series([row[target]]), errors="coerce").iloc[0]
                pred = float(np.mean(values)) if len(values) else global_rate
                preds.append(pred)
                actual.append(float(y) if np.isfinite(y) else math.nan)
                splits.append(str(row["split"]))
                hist_ns.append(len(values))
                if np.isfinite(y):
                    values.append(float(y))
        tmp = pd.DataFrame({"actual": actual, "pred": preds, "split": splits, "history_n": hist_ns})
        for split_name in ["train", "val", "test", "all"]:
            for min_hist in [0, 1, 3, 5]:
                sub = tmp if split_name == "all" else tmp[tmp["split"].eq(split_name)]
                sub = sub[sub["history_n"] >= min_hist].copy()
                m = np.isfinite(sub["actual"]) & np.isfinite(sub["pred"])
                if int(m.sum()) < 8 or len(np.unique(sub.loc[m, "actual"])) < 2:
                    auc = math.nan
                else:
                    auc = float(roc_auc_score(sub.loc[m, "actual"], sub.loc[m, "pred"]))
                rows.append(
                    {
                        "target": target,
                        "split": split_name,
                        "min_history_n": min_hist,
                        "n": int(m.sum()),
                        "positive_rate": float(sub.loc[m, "actual"].mean()) if int(m.sum()) else math.nan,
                        "history_auc": auc,
                    }
                )
    return pd.DataFrame(rows)


def label_candidate_counts(data: pd.DataFrame) -> pd.DataFrame:
    label_cols = [
        "oracle_strength_label",
        "oracle_timing_label",
        "oracle_shape_label",
        "oracle_direction_label",
        "oracle_error_label",
    ]
    rows: List[Dict[str, object]] = []
    for col in label_cols:
        for split_name in ["train", "val", "test", "all"]:
            sub = data if split_name == "all" else data[data["split"].astype(str).eq(split_name)]
            counts = sub[col].astype(str).value_counts(dropna=False)
            for label, n in counts.items():
                rows.append({"label_family": col, "split": split_name, "label": label, "n": int(n), "rate": float(n / max(len(sub), 1))})
    return pd.DataFrame(rows)


def make_decision(eta: pd.DataFrame, pair_summary: pd.DataFrame, rolling: pd.DataFrame, binary_auc: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    key_eta = eta[
        eta["split"].eq("train") & eta["group"].eq("subject") & eta["target"].isin(KEY_STYLE_TARGETS)
    ].copy()
    key_eta_mean = float(key_eta["eta_squared"].mean()) if len(key_eta) else math.nan
    key_eta_median = float(key_eta["eta_squared"].median()) if len(key_eta) else math.nan
    same_ratio = float(pair_summary["same_subject_mean_distance_ratio"].dropna().iloc[0]) if pair_summary["same_subject_mean_distance_ratio"].notna().any() else math.nan
    roll_key = rolling[
        rolling["split"].eq("test") & rolling["min_history_n"].eq(3) & rolling["target"].isin(KEY_STYLE_TARGETS)
    ].copy()
    roll_mean_rel = float(roll_key["relative_rmse_improvement"].mean()) if len(roll_key) else math.nan
    roll_positive_rate = float((roll_key["relative_rmse_improvement"] > 0).mean()) if len(roll_key) else math.nan
    bin_key = binary_auc[binary_auc["split"].eq("test") & binary_auc["min_history_n"].eq(3)].copy()
    bin_auc_mean = float(bin_key["history_auc"].mean()) if len(bin_key) and bin_key["history_auc"].notna().any() else math.nan

    checks = [
        {
            "check": "subject_eta_mean_train",
            "requirement": "key train subject eta mean >= 0.05",
            "value": key_eta_mean,
            "pass": bool(np.isfinite(key_eta_mean) and key_eta_mean >= 0.05),
        },
        {
            "check": "same_subject_distance_ratio",
            "requirement": "same-subject response distance <= 0.95 * different-subject distance",
            "value": same_ratio,
            "pass": bool(np.isfinite(same_ratio) and same_ratio <= 0.95),
        },
        {
            "check": "rolling_history_test_improvement",
            "requirement": "test rolling style mean relative RMSE improvement >= 0.02 for history_n>=3",
            "value": roll_mean_rel,
            "pass": bool(np.isfinite(roll_mean_rel) and roll_mean_rel >= 0.02),
        },
        {
            "check": "rolling_history_positive_targets",
            "requirement": "more than half key targets improve on test with history_n>=3",
            "value": roll_positive_rate,
            "pass": bool(np.isfinite(roll_positive_rate) and roll_positive_rate > 0.5),
        },
    ]
    decision = pd.DataFrame(checks)
    style_supported = bool(decision["pass"].all())
    weak_style = bool(decision["pass"].sum() >= 2)
    guardrail = {
        "pass": True,
        "event_n": int(eta[eta["split"].eq("all")]["n"].max()) if len(eta) else 0,
        "key_subject_eta_train_mean": key_eta_mean,
        "key_subject_eta_train_median": key_eta_median,
        "same_subject_mean_distance_ratio": same_ratio,
        "rolling_history_test_relative_rmse_improvement_mean_history3": roll_mean_rel,
        "rolling_history_test_positive_target_rate_history3": roll_positive_rate,
        "binary_history_test_auc_mean_history3": bin_auc_mean,
        "style_route_supported_now": style_supported,
        "weak_style_signal_exists": weak_style,
        "event_label_route_priority": not style_supported,
        "test_used_for_model_training_or_threshold": False,
        "future_derived_oracle_labels_are_not_deployable_inputs": True,
    }
    return decision, guardrail


def markdown_table(df: pd.DataFrame, cols: List[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_empty_"
    cols = [c for c in cols if c in df.columns]
    view = df.loc[:, cols].head(max_rows).copy()
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    return view.to_markdown(index=False)


def plot_eta(eta: pd.DataFrame) -> Path:
    path = FIGURES / "v297_subject_eta_by_descriptor.png"
    data = eta[eta["split"].eq("train") & eta["group"].eq("subject") & eta["target"].isin(KEY_STYLE_TARGETS)].copy()
    data = data.sort_values("eta_squared", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(data["target"], data["eta_squared"], color="#4E79A7")
    ax.axvline(0.05, color="tab:red", linestyle="--", linewidth=1, label="weak threshold 0.05")
    ax.set_xlabel("train subject eta squared")
    ax.set_title("v297 subject-level stability by response descriptor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_distance(pair_summary: pd.DataFrame) -> Path:
    path = FIGURES / "v297_same_vs_different_subject_distance.png"
    data = pair_summary[pair_summary["pair_group"].isin(["same_subject", "different_subject", "same_recording", "different_recording"])].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(data["pair_group"], data["distance_mean"], color=["#59A14F", "#E15759", "#76B7B2", "#F28E2B"][: len(data)])
    ax.set_ylabel("mean standardized response distance")
    ax.set_title("v297 response similarity: same vs different subject")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_rolling(rolling: pd.DataFrame) -> Path:
    path = FIGURES / "v297_rolling_history_improvement.png"
    data = rolling[rolling["split"].eq("test") & rolling["min_history_n"].eq(3) & rolling["target"].isin(KEY_STYLE_TARGETS)].copy()
    data = data.sort_values("relative_rmse_improvement", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(data["target"], data["relative_rmse_improvement"], color="#B07AA1")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.02, color="tab:red", linestyle="--", linewidth=1, label="useful threshold 0.02")
    ax.set_xlabel("test relative RMSE improvement vs global mean")
    ax.set_title("v297 causal rolling-history style predictor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    data: pd.DataFrame,
    eta: pd.DataFrame,
    subject_sum: pd.DataFrame,
    pair_summary: pd.DataFrame,
    rolling: pd.DataFrame,
    binary_auc: pd.DataFrame,
    label_counts: pd.DataFrame,
    decision: pd.DataFrame,
    guardrail: Dict[str, object],
) -> Path:
    lines: List[str] = []
    lines.append("# v297 subject style stability audit")
    lines.append("")
    lines.append("## 结论")
    if guardrail["style_route_supported_now"]:
        lines.append("- 当前审计支持继续做 subject-adaptive 驾驶风格建模。")
    elif guardrail["weak_style_signal_exists"]:
        lines.append("- 当前审计只显示弱 subject 风格信号；驾驶风格可以作为辅助，但不应单独作为主线。")
    else:
        lines.append("- 当前审计不支持把驾驶风格作为主线；优先级应转向事件级标签/实验条件标签。")
    lines.append(
        f"- train key subject eta mean={guardrail['key_subject_eta_train_mean']:.6f}, "
        f"same-subject distance ratio={guardrail['same_subject_mean_distance_ratio']:.6f}, "
        f"test rolling-history relative RMSE improvement={guardrail['rolling_history_test_relative_rmse_improvement_mean_history3']:.6f}."
    )
    lines.append("")
    lines.append("## 解释边界")
    lines.append("- 这里不假设一场实验内前后 trial 有直接因果关系。")
    lines.append("- rolling history 只用来检验同一被试是否存在稳定总体倾向，而不是事件序列记忆。")
    lines.append("- oracle labels 来自未来轨迹，只能用于辅助监督/分层/上限分析，不能作为测试时直接输入。")
    lines.append("")
    lines.append("## split / subject 概况")
    split_counts = data.groupby("split").size().reset_index(name="n")
    lines.append(markdown_table(split_counts, ["split", "n"], 10))
    lines.append("")
    lines.append(markdown_table(subject_sum, ["subject", "event_n", "recording_n", "train_n", "val_n", "test_n"], 30))
    lines.append("")
    lines.append("## decision")
    lines.append(markdown_table(decision, ["check", "requirement", "value", "pass"], 20))
    lines.append("")
    lines.append("## subject eta top")
    top_eta = eta[eta["split"].eq("train") & eta["group"].eq("subject")].sort_values("eta_squared", ascending=False)
    lines.append(markdown_table(top_eta, ["target", "eta_squared", "n", "group_n"], 30))
    lines.append("")
    lines.append("## pair distance")
    lines.append(markdown_table(pair_summary, ["pair_group", "pair_n", "distance_mean", "distance_median", "same_subject_mean_distance_ratio"], 10))
    lines.append("")
    lines.append("## rolling history predictability")
    roll = rolling[rolling["split"].eq("test") & rolling["min_history_n"].eq(3)].sort_values("relative_rmse_improvement", ascending=False)
    lines.append(markdown_table(roll, ["target", "n", "rmse_history", "rmse_global", "relative_rmse_improvement", "r2_history_vs_global"], 30))
    lines.append("")
    lines.append("## binary rolling history")
    bin_view = binary_auc[binary_auc["split"].eq("test") & binary_auc["min_history_n"].eq(3)]
    lines.append(markdown_table(bin_view, ["target", "n", "positive_rate", "history_auc"], 20))
    lines.append("")
    lines.append("## oracle label candidates")
    all_counts = label_counts[label_counts["split"].eq("all")]
    lines.append(markdown_table(all_counts, ["label_family", "label", "n", "rate"], 80))
    lines.append("")
    lines.append("## guardrail")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    path = REPORTS / "v297_subject_style_stability_audit_cn.md"
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
        raise RuntimeError(f"zip failed at {bad}")


def main() -> None:
    np.random.seed(SEED)
    ensure_dirs()
    print("[v297] build response descriptors", flush=True)
    input_hashes = pd.DataFrame(
        [
            {"path": str(V249_NPZ), "sha256": file_sha256(V249_NPZ), "role": "future truth and v249 prediction"},
            {"path": str(V293_FEATURES), "sha256": file_sha256(V293_FEATURES), "role": "event metadata and bad flags"},
            {"path": str(V295_GUARDRAIL), "sha256": file_sha256(V295_GUARDRAIL), "role": "previous physiology residual result"},
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")
    data = assign_oracle_labels(load_event_descriptors())
    write_csv(data, TABLES / "v297_event_response_descriptors.csv")

    print("[v297] compute subject eta and distance audit", flush=True)
    eta = compute_eta_table(data)
    subject_sum = subject_summary(data)
    pair_summary, pair_sample = pair_distance_summary(data)
    rolling = rolling_history_predictability(data)
    binary_auc = binary_history_auc(data)
    labels = label_candidate_counts(data)
    decision, guardrail = make_decision(eta, pair_summary, rolling, binary_auc)

    write_csv(eta, TABLES / "v297_subject_recording_eta.csv")
    write_csv(subject_sum, TABLES / "v297_subject_descriptor_summary.csv")
    write_csv(pair_summary, TABLES / "v297_pair_distance_summary.csv")
    write_csv(pair_sample.head(50_000), TABLES / "v297_pair_distance_sample.csv")
    write_csv(rolling, TABLES / "v297_rolling_history_predictability.csv")
    write_csv(binary_auc, TABLES / "v297_binary_history_auc.csv")
    write_csv(labels, TABLES / "v297_oracle_label_candidate_counts.csv")
    write_csv(decision, TABLES / "v297_style_route_decision.csv")
    write_json(guardrail, LOGS / "guardrail_check.json")

    plot_eta(eta)
    plot_distance(pair_summary)
    plot_rolling(rolling)
    write_report(data, eta, subject_sum, pair_summary, rolling, binary_auc, labels, decision, guardrail)

    inventory = [{"path": str(p), "bytes": int(p.stat().st_size)} for p in sorted(OUT.rglob("*")) if p.is_file()]
    write_csv(pd.DataFrame(inventory), LOGS / "file_inventory.csv")
    make_zip()
    guardrail["zip_testzip"] = True
    write_json(guardrail, LOGS / "guardrail_check.json")
    print("[v297] done")
    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
