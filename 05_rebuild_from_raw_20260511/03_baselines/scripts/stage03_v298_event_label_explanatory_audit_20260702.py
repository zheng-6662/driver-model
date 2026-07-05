#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v298 event label explanatory audit.

本轮不是继续训练更复杂的融合模型，而是回答一个更基础的问题：
1. 当前 1167 个事件里，未来响应标签是否真的解释了 bad_top10 和 v249 误差？
2. 如果响应标签“已知”，按标签做残差均值修正，理论上能给 v249 带来多大上限收益？
3. 旧阶段的事件任务/道路风险标签能否通过 subject + session + anchor time 匹配迁移到当前事件？

严格边界：
- oracle_* 标签来自未来真实轨迹，只能用于 auxiliary target / stratification / oracle upper bound。
- oracle_error_label 直接来自 v249 误差，是最强泄漏诊断，只能用于反证/上限，不允许进入模型输入。
- 历史规则标签只在严格时间匹配覆盖内审计，覆盖不足时不能当成当前全量标签。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
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
from sklearn.metrics import roc_auc_score


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
OUT = BASELINES / "v298_event_label_explanatory_audit_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v298_event_label_explanatory_audit_20260702_pack.zip"

V249_NPZ = BASELINES / "v249_shape_aware_curve_model_20260630" / "v249_shape_aware_predictions.npz"
V297_DESC = (
    BASELINES
    / "v297_subject_style_stability_audit_20260702"
    / "tables"
    / "v297_event_response_descriptors.csv"
)
V297_GUARDRAIL = BASELINES / "v297_subject_style_stability_audit_20260702" / "logs" / "guardrail_check.json"
OLD_EVENT_DECISION = (
    REBUILD
    / "02_samples"
    / "vehicle_instability_response_task_decision_v0_1"
    / "tables"
    / "event_response_task_decision_table.csv"
)
THIS_SCRIPT = Path(__file__).resolve()

SEED = 20260702
SHRINKAGE = 12.0
HISTORY_MATCH_TOL_S = 1.0
HISTORY_STRICT_TOL_S = 0.5

ORACLE_RESPONSE_LABELS = [
    "oracle_strength_label",
    "oracle_timing_label",
    "oracle_shape_label",
    "oracle_direction_label",
]
ORACLE_ALL_LABELS = ORACLE_RESPONSE_LABELS + ["oracle_error_label"]
HISTORY_LABELS = [
    "hist_response_task_track",
    "hist_response_task_class",
    "hist_event_level",
    "hist_road_design_risk_class",
    "hist_road_type_anchor",
]
NUMERIC_TARGETS = [
    "v249_rmse",
    "v249_tail_rmse",
    "true_peak_abs",
    "true_peak_time_s",
    "true_final_delta",
    "true_line_length",
    "true_tail_mean_abs",
]


def ensure_dirs() -> None:
    """清理并创建 v298 输出目录，避免旧产物混入本轮审计。"""

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


def safe_auc(y: Iterable[object], score: Iterable[object]) -> float:
    yy = pd.to_numeric(pd.Series(y), errors="coerce")
    ss = pd.to_numeric(pd.Series(score), errors="coerce")
    mask = yy.notna() & ss.notna()
    if int(mask.sum()) < 8 or yy[mask].nunique() < 2:
        return math.nan
    try:
        return float(roc_auc_score(yy[mask].astype(int), ss[mask].astype(float)))
    except ValueError:
        return math.nan


def eta_squared(values: Iterable[object], groups: Iterable[object]) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    g = pd.Series(groups).astype(str)
    mask = x.notna() & g.notna() & ~g.eq("") & ~g.eq("nan")
    if int(mask.sum()) < 8 or g[mask].nunique() < 2:
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


def curve_rmse(y_true: np.ndarray, y_pred: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """逐事件 masked RMSE。valid=False 的未来点不参与误差。"""

    diff = np.where(valid, y_true - y_pred, np.nan)
    mse = np.nanmean(diff * diff, axis=1)
    return np.sqrt(mse)


def parse_session_stamp(recording: object) -> str:
    match = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", str(recording))
    return match.group(1) if match else str(recording)


def parse_old_anchor_time(event_uid: object) -> float:
    match = re.search(r"__(\d+)$", str(event_uid))
    if not match:
        return math.nan
    return float(int(match.group(1)) / 1000.0)


def load_curve_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取当前事件描述符和 v249 delay0 曲线，并保证事件顺序完全对齐。"""

    desc = pd.read_csv(V297_DESC)
    for col in ["bad_top10", "vehicle_ambiguous", "bad_top10_vehicle_ambiguous", "candidate_pool_gain_gt_005"]:
        if col in desc.columns:
            desc[col] = pd.to_numeric(desc[col], errors="coerce").fillna(0).astype(int)
    desc["session_stamp"] = desc["recording"].map(parse_session_stamp)

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
        raise RuntimeError(f"v249 delay0 missing events: {len(missing)} example={missing[:3]}")
    aligned = np.array([row_map[e] for e in desc["event_uid"].astype(str)], dtype=int)
    y_true = y_true_all[aligned]
    pred = pred_all[aligned]
    valid = valid_all[aligned]
    desc["v249_curve_rmse_recalc"] = curve_rmse(y_true, pred, valid)
    return desc, y_true, pred, valid, grid


def build_history_label_match(desc: pd.DataFrame) -> pd.DataFrame:
    """把旧阶段事件任务标签按 subject + session + anchor time 最近邻匹配到当前事件。

    这一步只用于覆盖审计。即使时间接近，也不能把它直接当作人工真值标签；
    因为旧表来自不同样本构造版本，且当前 1167 事件覆盖范围更大。
    """

    base_cols = [
        "event_uid",
        "subject",
        "session_stamp",
        "response_task_class",
        "response_task_track",
        "event_type",
        "event_level",
        "road_type_anchor",
        "road_design_risk_class",
        "recommended_training_action",
    ]
    if not OLD_EVENT_DECISION.exists():
        out = desc[["event_uid", "split"]].copy()
        out["history_match_exists"] = False
        out["history_match_dt_s"] = math.nan
        return out

    old = pd.read_csv(OLD_EVENT_DECISION, usecols=lambda c: c in base_cols)
    old["old_event_uid"] = old["event_uid"].astype(str)
    old["old_anchor_time_s"] = old["old_event_uid"].map(parse_old_anchor_time)
    old = old.dropna(subset=["old_anchor_time_s"]).drop_duplicates("old_event_uid").reset_index(drop=True)

    rows: List[Dict[str, object]] = []
    for _, row in desc.iterrows():
        cand = old[
            old["subject"].astype(str).eq(str(row["subject"]))
            & old["session_stamp"].astype(str).eq(str(row["session_stamp"]))
        ]
        rec: Dict[str, object] = {
            "event_uid": row["event_uid"],
            "split": row["split"],
            "subject": row["subject"],
            "session_stamp": row["session_stamp"],
            "observation_s": row["observation_s"],
        }
        if cand.empty:
            rec.update({"history_match_exists": False, "history_match_dt_s": math.nan, "old_event_uid": ""})
        else:
            dt = (pd.to_numeric(cand["old_anchor_time_s"], errors="coerce") - float(row["observation_s"])).abs()
            best_idx = dt.idxmin()
            best = cand.loc[best_idx]
            rec.update(
                {
                    "history_match_exists": True,
                    "history_match_dt_s": float(dt.loc[best_idx]),
                    "old_event_uid": best["old_event_uid"],
                    "hist_response_task_class": best.get("response_task_class", ""),
                    "hist_response_task_track": best.get("response_task_track", ""),
                    "hist_event_type": best.get("event_type", ""),
                    "hist_event_level": best.get("event_level", ""),
                    "hist_road_type_anchor": best.get("road_type_anchor", ""),
                    "hist_road_design_risk_class": best.get("road_design_risk_class", ""),
                    "hist_recommended_training_action": best.get("recommended_training_action", ""),
                }
            )
        rows.append(rec)
    match = pd.DataFrame(rows)
    for col in HISTORY_LABELS + ["hist_event_type", "hist_recommended_training_action"]:
        if col not in match.columns:
            match[col] = ""
        match[col] = match[col].fillna("").astype(str)
        match.loc[~match["history_match_dt_s"].le(HISTORY_MATCH_TOL_S), col] = ""
    match["history_match_tol1s"] = match["history_match_dt_s"].le(HISTORY_MATCH_TOL_S)
    match["history_match_tol0p5s"] = match["history_match_dt_s"].le(HISTORY_STRICT_TOL_S)
    return match


def add_metadata_bins(data: pd.DataFrame) -> pd.DataFrame:
    """构造少量锚点前可知但很弱的 metadata proxy，用于反证覆盖。"""

    out = data.copy()
    train = out["split"].astype(str).eq("train")
    obs_q = out.loc[train, "observation_s"].quantile([0.25, 0.5, 0.75]).to_numpy()
    idx_q = out.loc[train, "event_index_in_uid"].quantile([0.25, 0.5, 0.75]).to_numpy()
    out["meta_observation_bin"] = pd.cut(
        out["observation_s"],
        bins=[-np.inf, obs_q[0], obs_q[1], obs_q[2], np.inf],
        labels=["obs_q1", "obs_q2", "obs_q3", "obs_q4"],
    ).astype(str)
    out["meta_event_order_bin"] = pd.cut(
        out["event_index_in_uid"],
        bins=[-np.inf, idx_q[0], idx_q[1], idx_q[2], np.inf],
        labels=["order_q1", "order_q2", "order_q3", "order_q4"],
    ).astype(str)
    out["meta_subject"] = out["subject"].astype(str)
    out["meta_recording"] = out["recording"].astype(str)
    return out


def label_key(frame: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    parts = []
    for col in cols:
        parts.append(frame[col].fillna("MISSING").replace("", "MISSING").astype(str))
    if len(parts) == 1:
        return parts[0]
    out = parts[0].copy()
    for p in parts[1:]:
        out = out + "|" + p
    return out


def label_family_catalog(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    families = ORACLE_ALL_LABELS + HISTORY_LABELS + ["meta_observation_bin", "meta_event_order_bin", "meta_subject", "meta_recording"]
    for fam in families:
        if fam not in data.columns:
            continue
        nonempty = data[fam].fillna("").astype(str).ne("")
        train_keys = set(data.loc[data["split"].eq("train") & nonempty, fam].astype(str))
        test_keys = set(data.loc[data["split"].eq("test") & nonempty, fam].astype(str))
        rows.append(
            {
                "label_family": fam,
                "label_type": label_type(fam),
                "coverage_all": float(nonempty.mean()),
                "coverage_train": float((data["split"].eq("train") & nonempty).sum() / max(data["split"].eq("train").sum(), 1)),
                "coverage_val": float((data["split"].eq("val") & nonempty).sum() / max(data["split"].eq("val").sum(), 1)),
                "coverage_test": float((data["split"].eq("test") & nonempty).sum() / max(data["split"].eq("test").sum(), 1)),
                "label_n_all": int(data.loc[nonempty, fam].nunique()),
                "test_seen_in_train_key_rate": float(
                    data.loc[data["split"].eq("test") & nonempty, fam].astype(str).isin(train_keys).mean()
                )
                if int((data["split"].eq("test") & nonempty).sum())
                else math.nan,
                "train_key_n": int(len(train_keys)),
                "test_key_n": int(len(test_keys)),
            }
        )
    return pd.DataFrame(rows)


def label_type(label_family: str) -> str:
    if label_family == "oracle_error_label":
        return "future_error_leakage_diagnostic_only"
    if label_family.startswith("oracle_"):
        return "future_response_auxiliary_oracle"
    if label_family.startswith("hist_"):
        return "historical_rule_label_time_matched_subset"
    if label_family.startswith("meta_"):
        return "pre_anchor_metadata_proxy"
    return "unknown"


def label_level_stats(data: pd.DataFrame, label_cols: Sequence[str], family_name: str, availability: pd.Series | None = None) -> pd.DataFrame:
    if availability is None:
        availability = pd.Series(True, index=data.index)
    tmp = data.loc[availability].copy()
    if tmp.empty:
        return pd.DataFrame()
    tmp["_label_key"] = label_key(tmp, label_cols)
    rows: List[Dict[str, object]] = []
    for split_name in ["all", "train", "val", "test"]:
        sub = tmp if split_name == "all" else tmp[tmp["split"].eq(split_name)]
        if sub.empty:
            continue
        split_bad = float(sub["bad_top10"].mean())
        split_rmse = float(sub["v249_rmse"].mean())
        for label, g in sub.groupby("_label_key", dropna=False):
            rows.append(
                {
                    "label_family": family_name,
                    "label_cols": "+".join(label_cols),
                    "label": label,
                    "label_type": label_type(family_name),
                    "split": split_name,
                    "n": int(len(g)),
                    "rate_in_split": float(len(g) / max(len(sub), 1)),
                    "bad_top10_rate": float(g["bad_top10"].mean()),
                    "bad_top10_enrichment_vs_split": float(g["bad_top10"].mean() / split_bad) if split_bad > 0 else math.nan,
                    "v249_rmse_mean": float(g["v249_rmse"].mean()),
                    "v249_rmse_delta_vs_split": float(g["v249_rmse"].mean() - split_rmse),
                    "true_peak_abs_mean": float(g["true_peak_abs"].mean()),
                    "true_line_length_mean": float(g["true_line_length"].mean()),
                    "right_direction_rate": float(g["oracle_direction_label"].eq("right").mean()) if "oracle_direction_label" in g.columns else math.nan,
                }
            )
    return pd.DataFrame(rows)


def train_label_rate_scores(data: pd.DataFrame, label_cols: Sequence[str], family_name: str, availability: pd.Series | None = None) -> pd.DataFrame:
    if availability is None:
        availability = pd.Series(True, index=data.index)
    tmp = data.loc[availability].copy()
    if tmp.empty or not tmp["split"].eq("train").any():
        return pd.DataFrame()
    tmp["_label_key"] = label_key(tmp, label_cols)
    train = tmp[tmp["split"].eq("train")].copy()
    global_bad = float(train["bad_top10"].mean())
    global_bad_amb = float(train["bad_top10_vehicle_ambiguous"].mean()) if "bad_top10_vehicle_ambiguous" in train.columns else global_bad
    counts = train.groupby("_label_key").size()
    bad_sum = train.groupby("_label_key")["bad_top10"].sum()
    amb_sum = train.groupby("_label_key")["bad_top10_vehicle_ambiguous"].sum()
    smooth = 3.0
    bad_rate = (bad_sum + smooth * global_bad) / (counts + smooth)
    amb_rate = (amb_sum + smooth * global_bad_amb) / (counts + smooth)
    rows: List[Dict[str, object]] = []
    for split_name in ["train", "val", "test"]:
        sub = tmp[tmp["split"].eq(split_name)].copy()
        if sub.empty:
            continue
        score_bad = sub["_label_key"].map(bad_rate).fillna(global_bad)
        score_amb = sub["_label_key"].map(amb_rate).fillna(global_bad_amb)
        rows.append(
            {
                "label_family": family_name,
                "label_cols": "+".join(label_cols),
                "label_type": label_type(family_name),
                "split": split_name,
                "n": int(len(sub)),
                "coverage_in_split": float(len(sub) / max(data["split"].eq(split_name).sum(), 1)),
                "bad_top10_positive_rate": float(sub["bad_top10"].mean()),
                "bad_top10_auc_from_train_label_rate": safe_auc(sub["bad_top10"], score_bad),
                "bad_top10_vehicle_ambiguous_auc_from_train_label_rate": safe_auc(
                    sub["bad_top10_vehicle_ambiguous"], score_amb
                )
                if "bad_top10_vehicle_ambiguous" in sub.columns
                else math.nan,
                "train_label_key_n": int(counts.size),
                "test_or_split_seen_key_rate": float(sub["_label_key"].isin(set(counts.index)).mean()),
            }
        )
    return pd.DataFrame(rows)


def family_numeric_explanation(data: pd.DataFrame, label_cols: Sequence[str], family_name: str, availability: pd.Series | None = None) -> pd.DataFrame:
    if availability is None:
        availability = pd.Series(True, index=data.index)
    tmp = data.loc[availability].copy()
    if tmp.empty:
        return pd.DataFrame()
    tmp["_label_key"] = label_key(tmp, label_cols)
    rows = []
    for split_name in ["train", "val", "test", "all"]:
        sub = tmp if split_name == "all" else tmp[tmp["split"].eq(split_name)]
        if sub.empty:
            continue
        for target in NUMERIC_TARGETS:
            rows.append(
                {
                    "label_family": family_name,
                    "label_cols": "+".join(label_cols),
                    "label_type": label_type(family_name),
                    "split": split_name,
                    "target": target,
                    "eta_squared": eta_squared(sub[target], sub["_label_key"]),
                    "n": int(pd.to_numeric(sub[target], errors="coerce").notna().sum()),
                    "label_key_n": int(sub["_label_key"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def fit_label_residual_curves(
    data: pd.DataFrame,
    y_true: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    label_cols: Sequence[str],
    config_name: str,
    label_source: str,
    availability: pd.Series | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """用 train 中每个标签 key 的平均残差曲线修正 v249，并在 val/test 上审计。

    这不是可部署模型；它只回答“如果这个标签未来能被可靠预测出来，残差修正是否有上限价值”。
    """

    if availability is None:
        availability = pd.Series(True, index=data.index)
    keys = label_key(data, label_cols)
    train_mask = data["split"].eq("train").to_numpy() & availability.to_numpy()
    if int(train_mask.sum()) < 20:
        return pd.DataFrame(), pd.DataFrame()

    residual = np.where(valid, y_true - pred, np.nan)
    global_resid = np.nanmean(residual[train_mask], axis=0)
    label_stats: Dict[str, Dict[str, object]] = {}
    for key in sorted(set(keys[train_mask])):
        idx = train_mask & keys.eq(key).to_numpy()
        if int(idx.sum()) == 0:
            continue
        mean_resid = np.nanmean(residual[idx], axis=0)
        weight = float(int(idx.sum()) / (int(idx.sum()) + SHRINKAGE))
        correction = weight * mean_resid + (1.0 - weight) * global_resid
        label_stats[str(key)] = {
            "n_train": int(idx.sum()),
            "weight": weight,
            "correction": correction,
        }

    pred_corr = pred.copy()
    correction_rows: List[Dict[str, object]] = []
    for i, key in enumerate(keys.astype(str)):
        if not bool(availability.iloc[i]):
            continue
        stat = label_stats.get(key)
        if stat is None:
            corr = global_resid
            n_train = 0
            weight = 0.0
            seen = False
        else:
            corr = stat["correction"]
            n_train = int(stat["n_train"])
            weight = float(stat["weight"])
            seen = True
        pred_corr[i] = pred[i] + corr
        correction_rows.append(
            {
                "event_uid": data.iloc[i]["event_uid"],
                "split": data.iloc[i]["split"],
                "config_name": config_name,
                "label_key": key,
                "label_seen_in_train": bool(seen),
                "label_train_n": n_train,
                "label_shrinkage_weight": weight,
                "available_for_config": True,
            }
        )

    base_rmse = curve_rmse(y_true, pred, valid)
    corr_rmse = curve_rmse(y_true, pred_corr, valid)
    event_delta = corr_rmse - base_rmse
    event_table = pd.DataFrame(correction_rows)
    if not event_table.empty:
        idx_map = {e: j for j, e in enumerate(data["event_uid"].astype(str))}
        event_table["_row"] = event_table["event_uid"].astype(str).map(idx_map).astype(int)
        event_table["baseline_rmse"] = base_rmse[event_table["_row"].to_numpy()]
        event_table["corrected_rmse"] = corr_rmse[event_table["_row"].to_numpy()]
        event_table["delta_vs_v249"] = event_delta[event_table["_row"].to_numpy()]
        event_table["bad_top10"] = data.iloc[event_table["_row"].to_numpy()]["bad_top10"].to_numpy()
        event_table["bad_top10_vehicle_ambiguous"] = data.iloc[event_table["_row"].to_numpy()][
            "bad_top10_vehicle_ambiguous"
        ].to_numpy()
        event_table = event_table.drop(columns=["_row"])

    rows: List[Dict[str, object]] = []
    for split_name in ["train", "val", "test"]:
        split_mask = data["split"].eq(split_name).to_numpy() & availability.to_numpy()
        if int(split_mask.sum()) == 0:
            continue
        for group_name, group_mask in [
            ("all_available", np.ones(len(data), dtype=bool)),
            ("bad_top10", data["bad_top10"].to_numpy(dtype=bool)),
            ("bad_top10_vehicle_ambiguous", data["bad_top10_vehicle_ambiguous"].to_numpy(dtype=bool)),
        ]:
            mask = split_mask & group_mask
            if int(mask.sum()) == 0:
                continue
            rows.append(
                {
                    "config_name": config_name,
                    "label_cols": "+".join(label_cols),
                    "label_source": label_source,
                    "split": split_name,
                    "group": group_name,
                    "n": int(mask.sum()),
                    "coverage_in_split": float(mask.sum() / max(data["split"].eq(split_name).sum(), 1)),
                    "label_key_n": int(keys[mask].nunique()),
                    "seen_key_rate": float(keys[mask].isin(set(label_stats.keys())).mean()),
                    "baseline_rmse_mean": float(np.nanmean(base_rmse[mask])),
                    "corrected_rmse_mean": float(np.nanmean(corr_rmse[mask])),
                    "delta_vs_v249_mean": float(np.nanmean(event_delta[mask])),
                    "delta_vs_v249_median": float(np.nanmedian(event_delta[mask])),
                }
            )
    return pd.DataFrame(rows), event_table


def run_all_label_audits(
    data: pd.DataFrame,
    y_true: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    level_tables = []
    risk_tables = []
    eta_tables = []
    correction_summaries = []
    correction_events = []

    # 单标签解释力。
    for fam in ORACLE_ALL_LABELS + HISTORY_LABELS + ["meta_observation_bin", "meta_event_order_bin", "meta_subject", "meta_recording"]:
        if fam not in data.columns:
            continue
        available = data[fam].fillna("").astype(str).ne("")
        if available.sum() == 0:
            continue
        level_tables.append(label_level_stats(data, [fam], fam, available))
        risk_tables.append(train_label_rate_scores(data, [fam], fam, available))
        eta_tables.append(family_numeric_explanation(data, [fam], fam, available))

    # 标签已知时的残差修正上限。error label 保留为泄漏诊断，不参与推荐。
    configs = [
        ("oracle_strength", ["oracle_strength_label"], "future_response_auxiliary_oracle"),
        ("oracle_timing", ["oracle_timing_label"], "future_response_auxiliary_oracle"),
        ("oracle_shape", ["oracle_shape_label"], "future_response_auxiliary_oracle"),
        ("oracle_direction", ["oracle_direction_label"], "future_response_auxiliary_oracle"),
        ("oracle_shape_direction", ["oracle_shape_label", "oracle_direction_label"], "future_response_auxiliary_oracle"),
        ("oracle_strength_shape", ["oracle_strength_label", "oracle_shape_label"], "future_response_auxiliary_oracle"),
        (
            "oracle_strength_shape_direction",
            ["oracle_strength_label", "oracle_shape_label", "oracle_direction_label"],
            "future_response_auxiliary_oracle",
        ),
        (
            "oracle_strength_timing_shape_direction",
            ["oracle_strength_label", "oracle_timing_label", "oracle_shape_label", "oracle_direction_label"],
            "future_response_auxiliary_oracle",
        ),
        ("oracle_error_label_leaky", ["oracle_error_label"], "future_error_leakage_diagnostic_only"),
        ("meta_observation_order", ["meta_observation_bin", "meta_event_order_bin"], "pre_anchor_metadata_proxy"),
        ("meta_subject", ["meta_subject"], "pre_anchor_metadata_proxy_subject_disjoint_unseen"),
        ("history_task_track_tol1s", ["hist_response_task_track"], "historical_rule_label_time_matched_subset"),
        (
            "history_task_track_risk_tol1s",
            ["hist_response_task_track", "hist_road_design_risk_class"],
            "historical_rule_label_time_matched_subset",
        ),
    ]
    for name, cols, source in configs:
        if not all(c in data.columns for c in cols):
            continue
        if name.startswith("history_"):
            available = data["history_match_tol1s"].astype(bool)
        else:
            available = pd.Series(True, index=data.index)
        summary, events = fit_label_residual_curves(data, y_true, pred, valid, cols, name, source, available)
        if not summary.empty:
            correction_summaries.append(summary)
        if not events.empty:
            correction_events.append(events)

    return (
        pd.concat(level_tables, ignore_index=True) if level_tables else pd.DataFrame(),
        pd.concat(risk_tables, ignore_index=True) if risk_tables else pd.DataFrame(),
        pd.concat(eta_tables, ignore_index=True) if eta_tables else pd.DataFrame(),
        pd.concat(correction_summaries, ignore_index=True) if correction_summaries else pd.DataFrame(),
        pd.concat(correction_events, ignore_index=True) if correction_events else pd.DataFrame(),
    )


def choose_decision(
    data: pd.DataFrame,
    catalog: pd.DataFrame,
    risk: pd.DataFrame,
    eta: pd.DataFrame,
    correction: pd.DataFrame,
    match: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """根据审计结果给出下一步路线判断。"""

    oracle_resp = correction[
        correction["label_source"].eq("future_response_auxiliary_oracle")
        & correction["split"].eq("test")
        & correction["group"].eq("bad_top10")
    ].copy()
    best_oracle_bad = math.nan
    best_oracle_name = ""
    if not oracle_resp.empty:
        best_row = oracle_resp.sort_values("delta_vs_v249_mean").iloc[0]
        best_oracle_bad = float(best_row["delta_vs_v249_mean"])
        best_oracle_name = str(best_row["config_name"])

    oracle_resp_all = correction[
        correction["config_name"].eq(best_oracle_name)
        & correction["split"].eq("test")
        & correction["group"].eq("all_available")
    ]
    best_oracle_all = float(oracle_resp_all["delta_vs_v249_mean"].iloc[0]) if len(oracle_resp_all) else math.nan

    error_diag = correction[
        correction["config_name"].eq("oracle_error_label_leaky")
        & correction["split"].eq("test")
        & correction["group"].eq("bad_top10")
    ]
    error_diag_bad = float(error_diag["delta_vs_v249_mean"].iloc[0]) if len(error_diag) else math.nan

    test_shape_eta = eta[
        eta["label_family"].eq("oracle_shape_label") & eta["split"].eq("test") & eta["target"].eq("v249_rmse")
    ]
    shape_eta = float(test_shape_eta["eta_squared"].iloc[0]) if len(test_shape_eta) else math.nan

    test_shape_auc = risk[
        risk["label_family"].eq("oracle_shape_label") & risk["split"].eq("test")
    ]
    shape_auc = float(test_shape_auc["bad_top10_auc_from_train_label_rate"].iloc[0]) if len(test_shape_auc) else math.nan

    hist_cov1 = float(match["history_match_tol1s"].mean()) if len(match) else 0.0
    hist_cov05 = float(match["history_match_tol0p5s"].mean()) if len(match) else 0.0
    hist_test_cov1 = float(match.loc[match["split"].eq("test"), "history_match_tol1s"].mean()) if len(match) else 0.0

    risk_test = risk[risk["split"].eq("test")].copy()
    eta_test = eta[(eta["split"].eq("test")) & (eta["target"].eq("v249_rmse"))].copy()
    risk_lookup = risk_test.set_index("label_family")["bad_top10_auc_from_train_label_rate"].to_dict()
    eta_lookup = eta_test.set_index("label_family")["eta_squared"].to_dict()

    deployable_catalog = catalog[
        catalog["label_type"].isin(["pre_anchor_metadata_proxy", "historical_rule_label_time_matched_subset"])
    ].copy()
    deployable_catalog["test_bad_auc"] = deployable_catalog["label_family"].map(risk_lookup)
    deployable_catalog["test_v249_rmse_eta"] = deployable_catalog["label_family"].map(eta_lookup)
    deployable_catalog["passes_coverage_seen_and_signal"] = (
        deployable_catalog["coverage_test"].fillna(0).ge(0.80)
        & deployable_catalog["test_seen_in_train_key_rate"].fillna(0).ge(0.80)
        & (
            deployable_catalog["test_bad_auc"].fillna(0).ge(0.60)
            | deployable_catalog["test_v249_rmse_eta"].fillna(0).ge(0.05)
        )
    )
    deployable_full = bool(deployable_catalog["passes_coverage_seen_and_signal"].any()) if not deployable_catalog.empty else False
    if deployable_catalog.empty:
        best_deployable_label = ""
        best_deployable_auc = math.nan
        best_deployable_eta = math.nan
    else:
        deployable_rank = deployable_catalog.copy()
        deployable_rank["rank_score"] = deployable_rank["test_bad_auc"].fillna(0) + deployable_rank[
            "test_v249_rmse_eta"
        ].fillna(0)
        best_dep = deployable_rank.sort_values("rank_score", ascending=False).iloc[0]
        best_deployable_label = str(best_dep["label_family"])
        best_deployable_auc = float(best_dep["test_bad_auc"]) if pd.notna(best_dep["test_bad_auc"]) else math.nan
        best_deployable_eta = float(best_dep["test_v249_rmse_eta"]) if pd.notna(best_dep["test_v249_rmse_eta"]) else math.nan

    oracle_risk = risk_test[risk_test["label_family"].isin(ORACLE_RESPONSE_LABELS)].copy()
    if oracle_risk.empty:
        best_oracle_risk_label = ""
        best_oracle_risk_auc = math.nan
    else:
        row = oracle_risk.sort_values("bad_top10_auc_from_train_label_rate", ascending=False).iloc[0]
        best_oracle_risk_label = str(row["label_family"])
        best_oracle_risk_auc = float(row["bad_top10_auc_from_train_label_rate"])
    oracle_upper_bound_useful = bool(np.isfinite(best_oracle_bad) and best_oracle_bad <= -0.05)
    oracle_all_no_big_harm = bool(np.isfinite(best_oracle_all) and best_oracle_all <= 0.01)

    rows = [
        {
            "check": "oracle_response_label_upper_bound_badtop10",
            "requirement": "best future response label-known correction improves test bad_top10 by at least 0.05 RMSE",
            "value": best_oracle_bad,
            "pass": oracle_upper_bound_useful,
        },
        {
            "check": "oracle_response_label_all_no_big_harm",
            "requirement": "same label-known correction has test all delta <= 0.01",
            "value": best_oracle_all,
            "pass": oracle_all_no_big_harm,
        },
        {
            "check": "history_rule_label_current_coverage",
            "requirement": "historical rule labels match >= 80% current test events within 1s",
            "value": hist_test_cov1,
            "pass": bool(hist_test_cov1 >= 0.80),
        },
        {
            "check": "deployable_current_label_available",
            "requirement": "pre-anchor/current labels have coverage>=0.8, train-seen>=0.8, and useful test signal",
            "value": best_deployable_auc,
            "pass": deployable_full,
        },
        {
            "check": "coarse_response_label_risk_only",
            "requirement": "future response labels may identify risk, but risk AUC alone is not trajectory correction",
            "value": best_oracle_risk_auc,
            "pass": bool(np.isfinite(best_oracle_risk_auc) and best_oracle_risk_auc >= 0.70 and not oracle_upper_bound_useful),
        },
    ]
    decision = pd.DataFrame(rows)
    guardrail = {
        "pass": True,
        "event_n": int(len(data)),
        "history_match_tol0p5_rate_all": hist_cov05,
        "history_match_tol1_rate_all": hist_cov1,
        "history_match_tol1_rate_test": hist_test_cov1,
        "best_oracle_response_config": best_oracle_name,
        "best_oracle_response_test_badtop10_delta": best_oracle_bad,
        "best_oracle_response_test_all_delta": best_oracle_all,
        "oracle_error_label_leaky_test_badtop10_delta": error_diag_bad,
        "best_oracle_response_risk_label": best_oracle_risk_label,
        "best_oracle_response_risk_test_auc": best_oracle_risk_auc,
        "oracle_shape_label_test_v249_rmse_eta": shape_eta,
        "oracle_shape_label_test_badtop10_auc_from_train_rate": shape_auc,
        "best_pre_anchor_or_history_label": best_deployable_label,
        "best_pre_anchor_or_history_label_test_bad_auc": best_deployable_auc,
        "best_pre_anchor_or_history_label_test_v249_rmse_eta": best_deployable_eta,
        "oracle_response_label_upper_bound_useful": oracle_upper_bound_useful,
        "oracle_response_label_all_no_big_harm": oracle_all_no_big_harm,
        "deployable_event_label_available_now": deployable_full,
        "history_rule_label_coverage_sufficient_now": bool(hist_test_cov1 >= 0.80),
        "manual_or_experimental_condition_label_priority": True,
        "coarse_response_labels_are_risk_markers_not_correction_solution": bool(
            np.isfinite(best_oracle_risk_auc) and best_oracle_risk_auc >= 0.70 and not oracle_upper_bound_useful
        ),
        "future_derived_labels_used_as_inputs": False,
        "test_used_for_threshold_model_selection": False,
        "recommended_next_step": "build/current-event manual or experimental-condition labels, then train auxiliary response heads; do not use oracle labels as inputs",
        "goal_achieved_now": False,
    }
    return decision, guardrail


def markdown_table(df: pd.DataFrame, cols: Sequence[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_empty_"
    cols = [c for c in cols if c in df.columns]
    view = df.loc[:, cols].head(max_rows).copy()
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    return view.to_markdown(index=False)


def plot_label_bad_rate(levels: pd.DataFrame) -> Path:
    path = FIGURES / "v298_oracle_label_bad_rate.png"
    data = levels[
        levels["split"].eq("test")
        & levels["label_family"].isin(["oracle_shape_label", "oracle_strength_label", "oracle_direction_label"])
    ].copy()
    data = data.sort_values(["label_family", "bad_top10_rate"])
    if data.empty:
        return path
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = data["label_family"].str.replace("oracle_", "", regex=False) + ":" + data["label"].astype(str)
    ax.bar(labels, data["bad_top10_rate"], color="#4E79A7")
    ax.axhline(0.10, color="tab:red", linestyle="--", linewidth=1, label="approx bad_top10 base rate")
    ax.set_ylabel("test bad_top10 rate")
    ax.set_title("v298 future-derived response labels: bad sample concentration")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_correction_delta(correction: pd.DataFrame) -> Path:
    path = FIGURES / "v298_label_known_residual_delta.png"
    data = correction[
        correction["split"].eq("test")
        & correction["group"].isin(["all_available", "bad_top10"])
        & correction["label_source"].isin(["future_response_auxiliary_oracle", "future_error_leakage_diagnostic_only"])
    ].copy()
    if data.empty:
        return path
    pivot = data.pivot_table(index="config_name", columns="group", values="delta_vs_v249_mean", aggfunc="first").reset_index()
    pivot = pivot.sort_values("bad_top10", ascending=True)
    x = np.arange(len(pivot))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, pivot.get("all_available", pd.Series(np.zeros(len(pivot)))), width, label="test all", color="#59A14F")
    ax.bar(x + width / 2, pivot.get("bad_top10", pd.Series(np.zeros(len(pivot)))), width, label="test bad_top10", color="#E15759")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(-0.05, color="tab:blue", linestyle="--", linewidth=1, label="-0.05 target")
    ax.set_ylabel("mean RMSE delta vs v249")
    ax.set_title("v298 label-known residual correction upper bound")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot["config_name"], rotation=40, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_history_coverage(match: pd.DataFrame) -> Path:
    path = FIGURES / "v298_history_label_match_coverage.png"
    rows = []
    for split_name in ["train", "val", "test", "all"]:
        sub = match if split_name == "all" else match[match["split"].eq(split_name)]
        if sub.empty:
            continue
        rows.append({"split": split_name, "tol": "0.5s", "coverage": float(sub["history_match_tol0p5s"].mean())})
        rows.append({"split": split_name, "tol": "1.0s", "coverage": float(sub["history_match_tol1s"].mean())})
    data = pd.DataFrame(rows)
    if data.empty:
        return path
    fig, ax = plt.subplots(figsize=(8, 5))
    splits = ["train", "val", "test", "all"]
    x = np.arange(len(splits))
    width = 0.35
    for j, tol in enumerate(["0.5s", "1.0s"]):
        vals = [float(data[(data["split"].eq(s)) & (data["tol"].eq(tol))]["coverage"].iloc[0]) for s in splits]
        ax.bar(x + (j - 0.5) * width, vals, width, label=tol)
    ax.axhline(0.8, color="tab:red", linestyle="--", linewidth=1, label="coverage target 0.8")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("matched coverage")
    ax.set_title("v298 historical rule label time-match coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build_bad_casebook(data: pd.DataFrame, correction_events: pd.DataFrame) -> pd.DataFrame:
    best_cfg = "oracle_strength_shape_direction"
    corr = correction_events[correction_events["config_name"].eq(best_cfg)][
        ["event_uid", "corrected_rmse", "delta_vs_v249", "label_key", "label_seen_in_train"]
    ].copy()
    corr = corr.rename(
        columns={
            "corrected_rmse": f"{best_cfg}_corrected_rmse",
            "delta_vs_v249": f"{best_cfg}_delta_vs_v249",
            "label_key": f"{best_cfg}_label_key",
            "label_seen_in_train": f"{best_cfg}_label_seen_in_train",
        }
    )
    cols = [
        "event_uid",
        "split",
        "subject",
        "recording",
        "observation_s",
        "bad_top10",
        "bad_top10_vehicle_ambiguous",
        "v249_rmse",
        "v249_tail_rmse",
        "true_peak_abs",
        "true_peak_time_s",
        "true_final_delta",
        "true_line_length",
        "oracle_strength_label",
        "oracle_timing_label",
        "oracle_shape_label",
        "oracle_direction_label",
        "oracle_error_label",
        "history_match_tol1s",
        "history_match_dt_s",
        "hist_response_task_track",
        "hist_road_design_risk_class",
    ]
    base = data[cols].copy()
    out = base.merge(corr, on="event_uid", how="left")
    out = out.sort_values(["split", "v249_rmse"], ascending=[True, False])
    return out


def write_report(
    catalog: pd.DataFrame,
    levels: pd.DataFrame,
    risk: pd.DataFrame,
    eta: pd.DataFrame,
    correction: pd.DataFrame,
    decision: pd.DataFrame,
    guardrail: Dict[str, object],
    match: pd.DataFrame,
) -> Path:
    lines: List[str] = []
    lines.append("# v298 event label explanatory audit")
    lines.append("")
    lines.append("## 结论")
    lines.append("- v298 的核心问题不是继续堆模型，而是检查“事件/响应标签”这条线有没有上限价值。")
    if guardrail["oracle_response_label_upper_bound_useful"]:
        lines.append(
            f"- 如果未来响应标签已知，最佳 response-label 残差修正 `{guardrail['best_oracle_response_config']}` "
            f"在 test bad_top10 上 delta={guardrail['best_oracle_response_test_badtop10_delta']:.6f}，说明标签路线有上限价值。"
        )
    else:
        lines.append("- 当前 future response label-known 修正对 test bad_top10 的上限收益仍不足，标签路线需要谨慎。")
    lines.append(
        f"- 历史规则标签 1s 时间匹配覆盖率：all={guardrail['history_match_tol1_rate_all']:.3f}, "
        f"test={guardrail['history_match_tol1_rate_test']:.3f}，覆盖不足，不能直接当成当前全量标签。"
    )
    lines.append("- 当前没有足够覆盖、可部署、锚点前可知的事件标签；下一步应做当前事件级人工/实验条件标签，而不是直接把 oracle 标签输入模型。")
    lines.append("")
    lines.append("## 边界")
    lines.append("- `oracle_strength/timing/shape/direction` 来自未来真实轨迹，只能作为 auxiliary target、分层审计和上限分析。")
    lines.append("- `oracle_error_label` 直接来自 v249 误差，是泄漏诊断，只能看理论极限，不能参与模型输入或部署决策。")
    lines.append("- 历史规则标签来自旧事件版本，本轮只按 subject + session + anchor time 做最近邻匹配；覆盖不足时只作为线索。")
    lines.append("")
    lines.append("## decision")
    lines.append(markdown_table(decision, ["check", "requirement", "value", "pass"], 20))
    lines.append("")
    lines.append("## label catalog")
    cat = catalog.sort_values(["label_type", "coverage_test"], ascending=[True, False])
    lines.append(
        markdown_table(
            cat,
            [
                "label_family",
                "label_type",
                "coverage_all",
                "coverage_test",
                "label_n_all",
                "test_seen_in_train_key_rate",
            ],
            40,
        )
    )
    lines.append("")
    lines.append("## test label risk AUC")
    risk_test = risk[risk["split"].eq("test")].sort_values("bad_top10_auc_from_train_label_rate", ascending=False)
    lines.append(
        markdown_table(
            risk_test,
            [
                "label_family",
                "label_type",
                "n",
                "coverage_in_split",
                "bad_top10_auc_from_train_label_rate",
                "bad_top10_vehicle_ambiguous_auc_from_train_label_rate",
                "test_or_split_seen_key_rate",
            ],
            40,
        )
    )
    lines.append("")
    lines.append("## test v249_rmse eta by label")
    eta_test = eta[eta["split"].eq("test") & eta["target"].eq("v249_rmse")].sort_values("eta_squared", ascending=False)
    lines.append(markdown_table(eta_test, ["label_family", "label_type", "eta_squared", "n", "label_key_n"], 40))
    lines.append("")
    lines.append("## label-known residual correction")
    corr_test = correction[correction["split"].eq("test")].sort_values(["group", "delta_vs_v249_mean"])
    lines.append(
        markdown_table(
            corr_test,
            [
                "config_name",
                "label_source",
                "group",
                "n",
                "coverage_in_split",
                "seen_key_rate",
                "baseline_rmse_mean",
                "corrected_rmse_mean",
                "delta_vs_v249_mean",
            ],
            80,
        )
    )
    lines.append("")
    lines.append("## oracle label level examples")
    lev = levels[
        levels["split"].eq("test")
        & levels["label_family"].isin(["oracle_shape_label", "oracle_strength_label", "oracle_direction_label"])
    ].sort_values("bad_top10_rate", ascending=False)
    lines.append(
        markdown_table(
            lev,
            [
                "label_family",
                "label",
                "n",
                "bad_top10_rate",
                "bad_top10_enrichment_vs_split",
                "v249_rmse_mean",
                "true_peak_abs_mean",
                "true_line_length_mean",
            ],
            40,
        )
    )
    lines.append("")
    lines.append("## history label match coverage")
    cov_rows = []
    for split_name in ["train", "val", "test", "all"]:
        sub = match if split_name == "all" else match[match["split"].eq(split_name)]
        if sub.empty:
            continue
        cov_rows.append(
            {
                "split": split_name,
                "n": int(len(sub)),
                "tol0p5_rate": float(sub["history_match_tol0p5s"].mean()),
                "tol1_rate": float(sub["history_match_tol1s"].mean()),
            }
        )
    lines.append(markdown_table(pd.DataFrame(cov_rows), ["split", "n", "tol0p5_rate", "tol1_rate"], 10))
    lines.append("")
    lines.append("## guardrail")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    path = REPORTS / "v298_event_label_explanatory_audit_cn.md"
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
    np.random.seed(SEED)
    ensure_dirs()
    print("[v298] 计划：先审计标签解释力，再做 label-known residual 上限，不把 oracle 标签当输入。", flush=True)
    input_hashes = pd.DataFrame(
        [
            {"path": str(V249_NPZ), "sha256": file_sha256(V249_NPZ), "role": "v249 baseline curves"},
            {"path": str(V297_DESC), "sha256": file_sha256(V297_DESC), "role": "current response descriptors and oracle labels"},
            {"path": str(V297_GUARDRAIL), "sha256": file_sha256(V297_GUARDRAIL), "role": "upstream style audit guardrail"},
            {
                "path": str(OLD_EVENT_DECISION),
                "sha256": file_sha256(OLD_EVENT_DECISION) if OLD_EVENT_DECISION.exists() else "",
                "role": "historical rule label candidate",
            },
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    data, y_true, pred, valid, grid = load_curve_data()
    match = build_history_label_match(data)
    data = data.merge(
        match.drop(columns=["split", "subject", "session_stamp", "observation_s"], errors="ignore"),
        on="event_uid",
        how="left",
    )
    for col in ["history_match_tol1s", "history_match_tol0p5s"]:
        data[col] = data[col].fillna(False).astype(bool)
    for col in HISTORY_LABELS + ["hist_event_type", "hist_recommended_training_action"]:
        if col not in data.columns:
            data[col] = ""
        data[col] = data[col].fillna("").astype(str)
    data = add_metadata_bins(data)
    data["future_grid_s"] = ",".join([f"{x:.3f}" for x in grid])
    write_csv(data, TABLES / "v298_event_label_audit_table.csv")
    write_csv(match, TABLES / "v298_historical_rule_label_time_match.csv")

    print("[v298] compute label explanatory tables", flush=True)
    catalog = label_family_catalog(data)
    levels, risk, eta, correction, correction_events = run_all_label_audits(data, y_true, pred, valid)
    decision, guardrail = choose_decision(data, catalog, risk, eta, correction, match)

    write_csv(catalog, TABLES / "v298_label_family_catalog.csv")
    write_csv(levels, TABLES / "v298_label_level_bad_rmse_summary.csv")
    write_csv(risk, TABLES / "v298_label_risk_auc_from_train_rates.csv")
    write_csv(eta, TABLES / "v298_label_numeric_eta_summary.csv")
    write_csv(correction, TABLES / "v298_label_known_residual_correction_summary.csv")
    write_csv(correction_events, TABLES / "v298_label_known_residual_event_deltas.csv")
    write_csv(decision, TABLES / "v298_event_label_route_decision.csv")
    casebook = build_bad_casebook(data, correction_events)
    write_csv(casebook, TABLES / "v298_bad_sample_label_casebook.csv")

    plot_label_bad_rate(levels)
    plot_correction_delta(correction)
    plot_history_coverage(match)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_report(catalog, levels, risk, eta, correction, decision, guardrail, match)

    inventory = [{"path": str(p), "bytes": int(p.stat().st_size)} for p in sorted(OUT.rglob("*")) if p.is_file()]
    write_csv(pd.DataFrame(inventory), LOGS / "file_inventory.csv")
    make_zip()
    guardrail["zip_testzip"] = True
    write_json(guardrail, LOGS / "guardrail_check.json")
    print("[v298] done", flush=True)
    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
