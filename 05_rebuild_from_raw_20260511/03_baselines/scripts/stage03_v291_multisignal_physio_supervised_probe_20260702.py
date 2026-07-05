#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v291 multi-signal physiology supervised probe.

本轮目标：
- v288/v289/v290 已分别证明 ECG、RESP、EDA 源信号作为 distance/rerank/gate
  不能形成可部署 top1 改善；
- v291 不再沿同一类无监督距离重排继续微调，而是把三路源信号合并后做严格监督探针：
  1) 生理能否识别差样本、车辆歧义样本、候选方法存在收益的样本；
  2) 生理能否帮助选择 latest / vehicle listrank / vehicle+bio listrank /
     vehicle+style+bio listrank 中实际更可靠的方法；
  3) 这些能力是否能在 validation 选择后泛化到 test bad_top10。

边界：
- 只读取已有的 causal source features 与 v278 候选预测，不重新抽原始波形；
- 特征筛选只用 train split；
- 监督模型只在 train split 拟合；
- 覆盖 latest 的阈值只由 val split 选择；
- test split 只报告，不参与特征筛选、模型选择或阈值选择。
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

OUT = BASELINES / "v291_multisignal_physio_supervised_probe_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v291_multisignal_physio_supervised_probe_20260702_pack.zip"

V278_CANDIDATES = (
    BASELINES
    / "v278_listwise_candidate_rank_loss_20260702"
    / "tables"
    / "v278_candidate_listrank_predictions_compact.csv"
)
V288_FEATURES = (
    BASELINES
    / "v288_ecg_source_signal_route_audit_20260702"
    / "tables"
    / "v288_ecg_source_features_with_targets.csv"
)
V289_FEATURES = (
    BASELINES
    / "v289_resp_source_phase_route_audit_20260702"
    / "tables"
    / "v289_resp_source_features_with_targets.csv"
)
V290_FEATURES = (
    BASELINES
    / "v290_eda_scr_usable_subset_route_audit_20260702"
    / "tables"
    / "v290_eda_source_features_with_targets.csv"
)

FIXED_WAIT_LATEST_BADTOP10 = 0.6950484153471495
SEED = 29102
BIO_TOP_N = 180
LOW_ID_TOP_N = 160

METHODS = [
    "latest",
    "listrank_vehicle",
    "listrank_vehicle_bio",
    "listrank_vehicle_style_bio",
]

GROUP_FLAGS = {
    "all": None,
    "bad_top10": "bad_top10",
    "vehicle_ambiguous": "vehicle_ambiguous",
    "bad_top10_vehicle_ambiguous": "bad_top10_vehicle_ambiguous",
    "eda_usable": "bio290_eda_event_usable",
}

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


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
    """把 bool/0/1/字符串标签统一成 0/1，便于分组和监督目标使用。"""

    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    lowered = s.astype(str).str.lower()
    return lowered.isin(["1", "true", "yes", "y"]).astype(int)


def numeric_feature_cols(df: pd.DataFrame, prefixes: Iterable[str]) -> List[str]:
    """只保留源信号数值列，避免把 event_uid、split、recording 等标识列当特征。"""

    cols: List[str] = []
    for col in df.columns:
        if not any(col.startswith(prefix) for prefix in prefixes):
            continue
        if col.endswith("_uses_post_observation"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == bool:
            cols.append(col)
    return cols


def load_source_features() -> pd.DataFrame:
    """合并 v288 ECG、v289 RESP、v290 EDA 的 event-level causal source features。"""

    missing = [p for p in [V288_FEATURES, V289_FEATURES, V290_FEATURES] if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少源信号特征表：" + "; ".join(str(p) for p in missing))

    ecg = pd.read_csv(V288_FEATURES, encoding="utf-8-sig", low_memory=False)
    resp = pd.read_csv(V289_FEATURES, encoding="utf-8-sig", low_memory=False)
    eda = pd.read_csv(V290_FEATURES, encoding="utf-8-sig", low_memory=False)

    meta_cols = [
        "event_uid",
        "subject",
        "recording",
        "split",
        "delay_ms",
        "observation_s",
        "future_peak_abs",
        "future_range",
        "future_mean_abs",
        "future_final",
        "future_slope",
        "high_future_abs_q75",
        "high_future_range_q75",
        "bad_top10_v250_diagnostic",
        "future_cluster4",
        "future_cluster6",
        "bad_top10",
        "vehicle_ambiguous",
        "bad_top10_vehicle_ambiguous",
        "candidate_rmse_std",
    ]
    base_cols = [c for c in meta_cols if c in eda.columns]
    base = eda[base_cols].drop_duplicates("event_uid").copy()

    # EDA 可用性是后续分层需要的质量标记，也允许作为生理质量特征使用。
    if "bio290_eda_event_usable" in eda.columns:
        base["bio290_eda_event_usable"] = safe_bool_series(eda["bio290_eda_event_usable"])
    else:
        base["bio290_eda_event_usable"] = 0

    for flag in ["bad_top10", "vehicle_ambiguous", "bad_top10_vehicle_ambiguous"]:
        if flag in base.columns:
            base[flag] = safe_bool_series(base[flag])

    for src, prefixes in [(ecg, ["bio288_"]), (resp, ["bio289_"]), (eda, ["bio290_"])]:
        cols = numeric_feature_cols(src, prefixes)
        cols = [c for c in cols if c not in base.columns]
        keep = ["event_uid"] + cols
        part = src[keep].drop_duplicates("event_uid")
        base = base.merge(part, on="event_uid", how="left", validate="one_to_one")

    return base


def softmax_weights(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    out = np.zeros_like(scores, dtype=float)
    if not finite.any():
        out[:] = 1.0 / max(len(scores), 1)
        return out
    x = scores.copy()
    x[~finite] = np.nanmin(x[finite])
    x = x - np.nanmax(x)
    e = np.exp(np.clip(x, -50, 50))
    denom = float(np.nansum(e))
    if denom <= 0:
        out[:] = 1.0 / max(len(scores), 1)
    else:
        out = e / denom
    return out


def summarize_candidate_scores(cand: pd.DataFrame) -> pd.DataFrame:
    """把每个 listrank 方法的候选分数压成 event-level 可部署特征。"""

    rows_by_event: Dict[object, Dict[str, object]] = {}
    for (event_uid, split, feature_set), g in cand.groupby(["event_uid", "split", "feature_set"], sort=False):
        scores = pd.to_numeric(g["pred_rank_score"], errors="coerce").to_numpy(dtype=float)
        delays = pd.to_numeric(g["mapped_delay_ms"], errors="coerce").to_numpy(dtype=float)
        valid_scores = scores[np.isfinite(scores)]
        if len(valid_scores) == 0:
            top_idx = 0
            top_score = math.nan
            second_score = math.nan
        else:
            order = np.argsort(scores)[::-1]
            top_idx = int(order[0])
            top_score = float(scores[top_idx])
            second_score = float(scores[order[1]]) if len(order) > 1 else float(scores[top_idx])
        weights = softmax_weights(scores)
        delay_mean = float(np.nansum(weights * delays)) if len(delays) else math.nan
        delay_std = float(np.sqrt(np.nansum(weights * (delays - delay_mean) ** 2))) if len(delays) else math.nan
        prefix = str(feature_set)
        row = rows_by_event.setdefault(event_uid, {"event_uid": event_uid, "split_from_score_summary": split})
        # 这里只保留推理时可见的候选分数和延迟摘要，不放入真实 RMSE，避免目标泄漏。
        row.update(
            {
            f"{prefix}_score_top": top_score,
            f"{prefix}_score_second": second_score,
            f"{prefix}_score_margin": top_score - second_score if np.isfinite(top_score) and np.isfinite(second_score) else math.nan,
            f"{prefix}_score_mean": float(np.nanmean(scores)),
            f"{prefix}_score_std": float(np.nanstd(scores)),
            f"{prefix}_score_range": float(np.nanmax(scores) - np.nanmin(scores)),
            f"{prefix}_score_p90": float(np.nanpercentile(scores, 90)),
            f"{prefix}_delay_top_ms": float(delays[top_idx]) if len(delays) else math.nan,
            f"{prefix}_delay_weighted_mean_ms": delay_mean,
            f"{prefix}_delay_weighted_std_ms": delay_std,
            f"{prefix}_top_is_latest": float(delays[top_idx] == 1000.0) if len(delays) else math.nan,
            }
        )

    if not rows_by_event:
        return pd.DataFrame(columns=["event_uid"])
    return pd.DataFrame(rows_by_event.values())


def load_candidate_methods() -> pd.DataFrame:
    """从 v278 候选表中得到每个事件的 latest 与三种 listrank top1 方法误差。"""

    if not V278_CANDIDATES.exists():
        raise FileNotFoundError(f"缺少 v278 候选预测表：{V278_CANDIDATES}")
    cand = pd.read_csv(V278_CANDIDATES, encoding="utf-8-sig", low_memory=False)
    for col in ["target_tail_rmse_v241", "latest_tail_rmse_v241", "mapped_delay_ms", "pred_rank_score"]:
        cand[col] = pd.to_numeric(cand[col], errors="coerce")

    method_rows: List[Dict[str, object]] = []
    for (event_uid, split), g0 in cand.groupby(["event_uid", "split"], sort=False):
        row: Dict[str, object] = {
            "event_uid": event_uid,
            "split_from_candidates": split,
            "latest_rmse": float(g0["latest_tail_rmse_v241"].iloc[0]),
        }
        for feature_set, g in g0.groupby("feature_set"):
            top = g.sort_values(["pred_rank_score", "target_tail_rmse_v241"], ascending=[False, True]).iloc[0]
            row[f"{feature_set}_rmse"] = float(top["target_tail_rmse_v241"])
            row[f"{feature_set}_delay_ms"] = float(top["mapped_delay_ms"])
            row[f"{feature_set}_score"] = float(top["pred_rank_score"])
        method_rows.append(row)
    methods = pd.DataFrame(method_rows)
    score_summary = summarize_candidate_scores(cand)
    methods = methods.merge(score_summary, on="event_uid", how="left", validate="one_to_one")
    rmse_cols = ["latest_rmse"] + [f"{m}_rmse" for m in METHODS if m != "latest"]
    methods["method_oracle_rmse"] = methods[rmse_cols].min(axis=1)
    methods["method_oracle_gain_vs_latest"] = methods["latest_rmse"] - methods["method_oracle_rmse"]
    methods["method_oracle_gain_gt_002"] = (methods["method_oracle_gain_vs_latest"] > 0.02).astype(int)
    return methods


def eta_squared(values: np.ndarray, labels: Iterable[object]) -> float:
    """粗略估计某个特征被 subject 身份解释的比例，用于标记高身份风险特征。"""

    x = np.asarray(values, dtype=float)
    labs = np.asarray(list(labels), dtype=object)
    mask = np.isfinite(x)
    if mask.sum() < 3:
        return 0.0
    x = x[mask]
    labs = labs[mask]
    grand = float(np.mean(x))
    ss_total = float(np.sum((x - grand) ** 2))
    if ss_total <= 1e-12:
        return 0.0
    ss_between = 0.0
    for lab in pd.unique(labs):
        vals = x[labs == lab]
        if len(vals):
            ss_between += float(len(vals) * (np.mean(vals) - grand) ** 2)
    return max(0.0, min(1.0, ss_between / ss_total))


def abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return 0.0
    if float(np.nanstd(x[mask])) <= 1e-12 or float(np.nanstd(y[mask])) <= 1e-12:
        return 0.0
    return float(abs(np.corrcoef(x[mask], y[mask])[0, 1]))


def screen_bio_features(df: pd.DataFrame, bio_cols: List[str]) -> pd.DataFrame:
    """只用 train split 对三路源生理特征做筛选与身份风险标记。"""

    train = df["split"].astype(str).eq("train")
    targets = [
        "bad_top10",
        "vehicle_ambiguous",
        "bad_top10_vehicle_ambiguous",
        "latest_rmse",
        "method_oracle_gain_vs_latest",
        "method_oracle_gain_gt_002",
        "candidate_rmse_std",
    ]
    rows: List[Dict[str, object]] = []
    for col in bio_cols:
        values = pd.to_numeric(df.loc[train, col], errors="coerce").to_numpy(dtype=float)
        finite_rate = float(np.isfinite(values).mean()) if len(values) else 0.0
        std = float(np.nanstd(values)) if np.isfinite(values).any() else 0.0
        if finite_rate < 0.60 or std <= 1e-10:
            continue
        scores = {
            f"corr_{target}": abs_corr(
                values,
                pd.to_numeric(df.loc[train, target], errors="coerce").to_numpy(dtype=float),
            )
            for target in targets
            if target in df.columns
        }
        behavior_score = max(scores.values()) if scores else 0.0
        identity_eta = eta_squared(values, df.loc[train, "subject"].astype(str))
        source = "ecg" if col.startswith("bio288_") else "resp" if col.startswith("bio289_") else "eda"
        rows.append(
            {
                "feature": col,
                "source": source,
                "finite_rate_train": finite_rate,
                "std_train": std,
                "behavior_corr_max": behavior_score,
                "identity_eta": identity_eta,
                "low_identity_candidate": bool(identity_eta <= 0.25 or identity_eta <= max(0.08, behavior_score * 4.0)),
                **scores,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "source", "behavior_corr_max", "identity_eta", "low_identity_candidate"])
    screen = pd.DataFrame(rows)
    screen["screen_score_all"] = screen["behavior_corr_max"]
    screen["screen_score_lowid"] = screen["behavior_corr_max"] - 0.20 * np.sqrt(screen["identity_eta"].clip(lower=0))
    return screen.sort_values(["screen_score_all", "finite_rate_train"], ascending=[False, False])


def build_feature_blocks(df: pd.DataFrame, screen: pd.DataFrame) -> Dict[str, List[str]]:
    """构造对照特征块：候选分数、源生理、低身份源生理、两者组合。"""

    score_cols = [
        c
        for c in df.columns
        if c.startswith("listrank_vehicle_")
        or c.startswith("listrank_vehicle_bio_")
        or c.startswith("listrank_vehicle_style_bio_")
    ]
    score_cols = [c for c in score_cols if "candidate_rmse" not in c and not c.endswith("_rmse")]
    vehicle_score_cols = [
        c
        for c in score_cols
        if c.startswith("listrank_vehicle_")
        and not c.startswith("listrank_vehicle_bio_")
        and not c.startswith("listrank_vehicle_style_bio_")
    ]
    bio_all = screen.sort_values(["screen_score_all", "finite_rate_train"], ascending=[False, False])["feature"].head(BIO_TOP_N).tolist()
    lowid_pool = screen[screen["low_identity_candidate"].astype(bool)].copy()
    if len(lowid_pool) < 40:
        lowid_pool = screen.copy()
    bio_lowid = lowid_pool.sort_values(["screen_score_lowid", "finite_rate_train"], ascending=[False, False])["feature"].head(LOW_ID_TOP_N).tolist()
    blocks = {
        "vehicle_scores_only": vehicle_score_cols,
        "all_listrank_scores": score_cols,
        "bio_source_all_top": bio_all,
        "bio_source_lowid_top": bio_lowid,
        "vehicle_scores_plus_bio_all": vehicle_score_cols + bio_all,
        "all_listrank_scores_plus_bio_all": score_cols + bio_all,
        "vehicle_scores_plus_bio_lowid": vehicle_score_cols + bio_lowid,
    }
    return {k: [c for c in v if c in df.columns] for k, v in blocks.items() if len(v)}


def make_regressors() -> Dict[str, Pipeline]:
    """多输出误差回归器。Ridge 看线性可解释性，ExtraTrees 看非线性但限制深度。"""

    return {
        "ridge_a1": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "ridge_a10": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "extra_trees_d4": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=240,
                        max_depth=4,
                        min_samples_leaf=8,
                        random_state=SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "extra_trees_d6": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=240,
                        max_depth=6,
                        min_samples_leaf=6,
                        random_state=SEED + 1,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def method_rmse_columns() -> List[str]:
    return ["latest_rmse"] + [f"{m}_rmse" for m in METHODS if m != "latest"]


def predict_method_errors(df: pd.DataFrame, cols: List[str], model: Pipeline) -> np.ndarray:
    train = df["split"].astype(str).eq("train").to_numpy()
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    y = df[method_rmse_columns()].astype(float)
    model.fit(X.loc[train], y.loc[train])
    return np.asarray(model.predict(X), dtype=float)


def build_policy_frame(df: pd.DataFrame, pred_errors: np.ndarray, tag: str) -> pd.DataFrame:
    """把每个事件的预测方法误差转成：是否覆盖 latest、覆盖哪个方法、margin 多大。"""

    out = df[
        [
            "event_uid",
            "split",
            "subject",
            "recording",
            "bad_top10",
            "vehicle_ambiguous",
            "bad_top10_vehicle_ambiguous",
            "bio290_eda_event_usable",
        ]
        + method_rmse_columns()
    ].copy()
    pred_cols = [f"pred_error_{m}" for m in METHODS]
    for i, col in enumerate(pred_cols):
        out[col] = pred_errors[:, i]
    nonlatest = pred_errors[:, 1:]
    best_nonlatest_idx = np.nanargmin(nonlatest, axis=1) + 1
    latest_pred = pred_errors[:, 0]
    best_nonlatest_pred = pred_errors[np.arange(len(out)), best_nonlatest_idx]
    out["selector_tag"] = tag
    out["pred_best_nonlatest_method"] = [METHODS[i] for i in best_nonlatest_idx]
    out["pred_best_nonlatest_error"] = best_nonlatest_pred
    out["selector_margin_vs_latest"] = latest_pred - best_nonlatest_pred
    for method in METHODS:
        out[f"actual_rmse_{method}"] = out["latest_rmse"] if method == "latest" else out[f"{method}_rmse"]
    return out


def summarize_selected(selected: pd.DataFrame, split: str, group: str, flag: str | None) -> Dict[str, object]:
    sub = selected[selected["split"].astype(str).eq(split)].copy()
    if flag is not None:
        if flag not in sub.columns:
            sub = sub.iloc[0:0].copy()
        else:
            sub = sub[safe_bool_series(sub[flag]).astype(bool)].copy()
    if sub.empty:
        return {
            "split": split,
            "event_group": group,
            "n": 0,
            "latest_rmse_mean": math.nan,
            "selected_rmse_mean": math.nan,
            "delta_vs_latest_mean": math.nan,
            "override_rate": math.nan,
            "override_n": 0,
        }
    return {
        "split": split,
        "event_group": group,
        "n": int(len(sub)),
        "latest_rmse_mean": float(sub["latest_rmse"].mean()),
        "selected_rmse_mean": float(sub["selected_rmse"].mean()),
        "delta_vs_latest_mean": float((sub["selected_rmse"] - sub["latest_rmse"]).mean()),
        "override_rate": float(sub["override_latest"].astype(bool).mean()),
        "override_n": int(sub["override_latest"].astype(bool).sum()),
    }


def evaluate_thresholds(policy: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """只用 val margin 构造阈值网格；test 不参与阈值构造。"""

    val_margin = pd.to_numeric(
        policy.loc[policy["split"].astype(str).eq("val"), "selector_margin_vs_latest"],
        errors="coerce",
    )
    positive = val_margin[np.isfinite(val_margin) & (val_margin > 0)]
    thresholds = [float("inf")]
    if len(positive):
        thresholds.extend(float(x) for x in np.unique(np.nanquantile(positive, np.linspace(0.05, 0.95, 19))))
        thresholds.append(0.0)
    thresholds = sorted(set(thresholds), key=lambda x: (math.isinf(x), x))

    per_event_frames: List[pd.DataFrame] = []
    rows: List[Dict[str, object]] = []
    for threshold in thresholds:
        selected = policy.copy()
        selected["threshold"] = threshold
        selected["override_latest"] = pd.to_numeric(selected["selector_margin_vs_latest"], errors="coerce") >= threshold
        selected.loc[np.isinf(threshold), "override_latest"] = False
        selected["selected_method"] = np.where(selected["override_latest"], selected["pred_best_nonlatest_method"], "latest")
        selected["selected_rmse"] = selected["latest_rmse"].astype(float)
        for method in METHODS[1:]:
            mask = selected["selected_method"].astype(str).eq(method)
            selected.loc[mask, "selected_rmse"] = selected.loc[mask, f"{method}_rmse"].astype(float)
        per_event_frames.append(selected)

        row: Dict[str, object] = {
            "selector_tag": str(policy["selector_tag"].iloc[0]),
            "threshold": threshold,
        }
        for split in ["train", "val", "test"]:
            for group, flag in GROUP_FLAGS.items():
                m = summarize_selected(selected, split, group, flag)
                prefix = f"{split}_{group}"
                for key, value in m.items():
                    if key in {"split", "event_group"}:
                        continue
                    row[f"{prefix}_{key}"] = value
        row["active_val_bad_top10"] = int(row.get("val_bad_top10_override_n", 0) or 0) > 0
        row["noharm_val"] = (
            float(row.get("val_bad_top10_delta_vs_latest_mean", math.inf)) <= 0.0
            and float(row.get("val_all_delta_vs_latest_mean", math.inf)) <= 0.003
            and float(row.get("val_vehicle_ambiguous_delta_vs_latest_mean", math.inf)) <= 0.005
        )
        row["selection_score"] = float(row.get("val_bad_top10_delta_vs_latest_mean", 0.0)) + 5.0 * max(
            0.0, float(row.get("val_all_delta_vs_latest_mean", 0.0))
        )
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(per_event_frames, ignore_index=True)


def run_selector_probe(df: pd.DataFrame, feature_blocks: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    events: List[pd.DataFrame] = []
    regressors = make_regressors()
    for block_name, cols in feature_blocks.items():
        for model_name, model in regressors.items():
            tag = f"{block_name}__{model_name}"
            print(f"[v291] selector {tag} feature_n={len(cols)}")
            pred = predict_method_errors(df, cols, model)
            policy = build_policy_frame(df, pred, tag)
            summary, per_event = evaluate_thresholds(policy)
            summary["feature_block"] = block_name
            summary["model_name"] = model_name
            summary["feature_n"] = len(cols)
            per_event["feature_block"] = block_name
            per_event["model_name"] = model_name
            rows.append(summary)
            events.append(per_event)
    return pd.concat(rows, ignore_index=True), pd.concat(events, ignore_index=True)


def classifier_models() -> Dict[str, Pipeline]:
    return {
        "logreg_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "extra_trees_cls": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=240,
                        max_depth=5,
                        min_samples_leaf=6,
                        class_weight="balanced",
                        random_state=SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    y = np.asarray(y_true)[mask]
    s = np.asarray(score, dtype=float)[mask]
    if len(np.unique(y)) < 2:
        return math.nan
    return float(roc_auc_score(y, s))


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    y = np.asarray(y_true)[mask]
    s = np.asarray(score, dtype=float)[mask]
    if len(np.unique(y)) < 2:
        return math.nan
    return float(average_precision_score(y, s))


def run_classification_probe(df: pd.DataFrame, feature_blocks: Dict[str, List[str]]) -> pd.DataFrame:
    """辅助诊断：生理是否至少能识别坏样本/歧义样本/方法池有收益样本。"""

    target_cols = ["bad_top10", "vehicle_ambiguous", "bad_top10_vehicle_ambiguous", "method_oracle_gain_gt_002"]
    rows: List[Dict[str, object]] = []
    train = df["split"].astype(str).eq("train")
    for block_name, cols in feature_blocks.items():
        for target in target_cols:
            y = safe_bool_series(df[target]).to_numpy(dtype=int)
            if len(np.unique(y[train.to_numpy()])) < 2:
                continue
            for model_name, model in classifier_models().items():
                tag = f"{block_name}__{model_name}__{target}"
                print(f"[v291] classifier {tag} feature_n={len(cols)}")
                X = df[cols].replace([np.inf, -np.inf], np.nan)
                model.fit(X.loc[train], y[train.to_numpy()])
                if hasattr(model, "predict_proba"):
                    score = model.predict_proba(X)[:, 1]
                else:
                    score = model.decision_function(X)
                for split in ["val", "test"]:
                    m = df["split"].astype(str).eq(split).to_numpy()
                    rows.append(
                        {
                            "feature_block": block_name,
                            "model_name": model_name,
                            "target": target,
                            "split": split,
                            "n": int(m.sum()),
                            "positive_rate": float(np.mean(y[m])) if m.sum() else math.nan,
                            "auc": safe_auc(y[m], score[m]),
                            "average_precision": safe_ap(y[m], score[m]),
                            "feature_n": int(len(cols)),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_method_pool(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split in ["train", "val", "test"]:
        for group, flag in GROUP_FLAGS.items():
            sub = df[df["split"].astype(str).eq(split)].copy()
            if flag is not None:
                if flag not in sub.columns:
                    sub = sub.iloc[0:0]
                else:
                    sub = sub[safe_bool_series(sub[flag]).astype(bool)]
            if sub.empty:
                continue
            for method in METHODS:
                col = "latest_rmse" if method == "latest" else f"{method}_rmse"
                rows.append(
                    {
                        "split": split,
                        "event_group": group,
                        "method": method,
                        "n": int(len(sub)),
                        "rmse_mean": float(sub[col].mean()),
                        "delta_vs_latest_mean": float((sub[col] - sub["latest_rmse"]).mean()),
                        "beats_latest_rate": float((sub[col] < sub["latest_rmse"]).mean()) if method != "latest" else math.nan,
                    }
                )
            rows.append(
                {
                    "split": split,
                    "event_group": group,
                    "method": "oracle_best_of_methods",
                    "n": int(len(sub)),
                    "rmse_mean": float(sub["method_oracle_rmse"].mean()),
                    "delta_vs_latest_mean": float((sub["method_oracle_rmse"] - sub["latest_rmse"]).mean()),
                    "beats_latest_rate": float((sub["method_oracle_rmse"] < sub["latest_rmse"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def choose_deployable(selector_summary: pd.DataFrame) -> pd.DataFrame:
    active = selector_summary[
        selector_summary["active_val_bad_top10"].astype(bool) & selector_summary["noharm_val"].astype(bool)
    ].copy()
    rows: List[Dict[str, object]] = []
    if len(active):
        chosen = active.sort_values(
            ["selection_score", "val_bad_top10_delta_vs_latest_mean", "val_all_delta_vs_latest_mean"]
        ).iloc[0]
        rows.append({"chosen_type": "best_val_noharm_active", **chosen.to_dict()})
    else:
        inactive = selector_summary[selector_summary["threshold"].map(math.isinf)].copy()
        if len(inactive):
            chosen = inactive.sort_values("val_bad_top10_delta_vs_latest_mean").iloc[0]
            rows.append({"chosen_type": "fallback_no_override", **chosen.to_dict()})
    active_test = selector_summary[pd.to_numeric(selector_summary["test_bad_top10_override_n"], errors="coerce").fillna(0) > 0].copy()
    if len(active_test):
        diag = active_test.sort_values(
            ["test_bad_top10_delta_vs_latest_mean", "val_bad_top10_delta_vs_latest_mean"]
        ).iloc[0]
        rows.append({"chosen_type": "test_best_diagnostic_not_deployable", **diag.to_dict()})
    return pd.DataFrame(rows)


def route_decision(chosen: pd.DataFrame, method_pool: pd.DataFrame, cls: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    dep = chosen[chosen["chosen_type"].astype(str).eq("best_val_noharm_active")].copy()
    if len(dep):
        row = dep.iloc[0]
        dep_bad_delta = float(row.get("test_bad_top10_delta_vs_latest_mean", math.inf))
        dep_amb_delta = float(row.get("test_bad_top10_vehicle_ambiguous_delta_vs_latest_mean", math.inf))
        dep_override = float(row.get("test_bad_top10_override_rate", 0.0))
    else:
        dep_bad_delta = math.inf
        dep_amb_delta = math.inf
        dep_override = 0.0
    oracle_bad = method_pool[
        method_pool["split"].eq("test")
        & method_pool["event_group"].eq("bad_top10")
        & method_pool["method"].eq("oracle_best_of_methods")
    ]
    oracle_delta = float(oracle_bad["delta_vs_latest_mean"].iloc[0]) if len(oracle_bad) else math.nan
    best_cls = cls[
        cls["split"].eq("test")
        & cls["target"].eq("bad_top10")
        & cls["feature_block"].str.contains("bio", regex=False)
    ].copy()
    best_auc = float(best_cls["auc"].max()) if len(best_cls) else math.nan
    rows.append(
        {
            "check": "deployable_val_noharm_selector_beats_latest_bad_top10",
            "requirement": "val no-harm active selector 在 test bad_top10 上低于 latest",
            "pass": bool(dep_bad_delta < -1e-9 and dep_override > 0),
            "evidence": dep_bad_delta if np.isfinite(dep_bad_delta) else None,
            "deployable": True,
        }
    )
    rows.append(
        {
            "check": "deployable_val_noharm_selector_beats_latest_bad_ambiguous",
            "requirement": "同一 selector 在 test bad_top10_vehicle_ambiguous 上低于 latest",
            "pass": bool(dep_amb_delta < -1e-9 and dep_override > 0),
            "evidence": dep_amb_delta if np.isfinite(dep_amb_delta) else None,
            "deployable": True,
        }
    )
    rows.append(
        {
            "check": "method_pool_oracle_has_enough_headroom",
            "requirement": "现成方法池的事后 oracle 在 test bad_top10 至少改善 0.03 RMSE",
            "pass": bool(np.isfinite(oracle_delta) and oracle_delta <= -0.03),
            "evidence": oracle_delta,
            "deployable": False,
        }
    )
    rows.append(
        {
            "check": "bio_classifier_badtop10_auc_gt_060",
            "requirement": "源生理特征至少能在 test 上识别 bad_top10，AUC > 0.60",
            "pass": bool(np.isfinite(best_auc) and best_auc > 0.60),
            "evidence": best_auc,
            "deployable": False,
        }
    )
    decision = pd.DataFrame(rows)
    decision["route_viable_now"] = bool(
        decision.loc[
            decision["check"].isin(
                [
                    "deployable_val_noharm_selector_beats_latest_bad_top10",
                    "deployable_val_noharm_selector_beats_latest_bad_ambiguous",
                ]
            ),
            "pass",
        ].all()
    )
    return decision


def plot_method_pool(method_pool: pd.DataFrame) -> Path:
    data = method_pool[method_pool["split"].eq("test") & method_pool["event_group"].eq("bad_top10")].copy()
    data = data.sort_values("delta_vs_latest_mean")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#2F4B7C" if m == "oracle_best_of_methods" else "#A05195" if "bio" in m else "#665191" for m in data["method"]]
    ax.barh(data["method"], data["delta_vs_latest_mean"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("test bad_top10 delta vs latest RMSE")
    ax.set_title("v291 method-pool headroom on bad_top10")
    fig.tight_layout()
    path = FIGURES / "v291_method_pool_badtop10_delta.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_selector_choices(chosen: pd.DataFrame) -> Path:
    if chosen.empty:
        data = pd.DataFrame({"chosen_type": ["none"], "test_bad_top10_delta_vs_latest_mean": [math.nan]})
    else:
        data = chosen.copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(data["chosen_type"].astype(str), pd.to_numeric(data["test_bad_top10_delta_vs_latest_mean"], errors="coerce"), color="#1F77B4")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("test bad_top10 delta vs latest RMSE")
    ax.set_title("v291 validation-chosen selector outcome")
    fig.tight_layout()
    path = FIGURES / "v291_selector_chosen_badtop10_delta.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_feature_screen(screen: pd.DataFrame) -> Path:
    top = screen.head(80).copy()
    counts = top["source"].value_counts().reindex(["ecg", "resp", "eda"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(counts.index, counts.values, color=["#2F4B7C", "#F28E2B", "#59A14F"])
    ax.set_ylabel("top-80 train-screen feature count")
    ax.set_title("v291 source feature screen composition")
    fig.tight_layout()
    path = FIGURES / "v291_feature_screen_source_counts.png"
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


def write_report(
    method_pool: pd.DataFrame,
    selector_summary: pd.DataFrame,
    chosen: pd.DataFrame,
    cls: pd.DataFrame,
    screen: pd.DataFrame,
    decision: pd.DataFrame,
    guardrail: Dict[str, object],
) -> Path:
    path = REPORTS / "v291_multisignal_physio_supervised_probe_cn.md"
    lines: List[str] = []
    lines.append("# v291 multi-signal physiology supervised probe")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- v288/v289/v290 已经排除 ECG、RESP、EDA 单路源信号 distance/rerank/gate 的可部署改善。")
    lines.append("- v291 把三路源信号合并，改做严格监督探针：看生理是否能识别差样本、识别方法池有收益样本，并帮助选择现成预测方法。")
    lines.append("- 仍然执行 train 训练、val 选阈值、test 只报告。")
    lines.append("")
    lines.append("## route decision")
    lines.append("")
    lines.append(markdown_table(decision, ["check", "requirement", "pass", "evidence", "deployable", "route_viable_now"]))
    lines.append("")
    lines.append("## method pool 上限")
    lines.append("")
    mp = method_pool[method_pool["split"].eq("test") & method_pool["event_group"].isin(["all", "bad_top10", "bad_top10_vehicle_ambiguous"])].copy()
    lines.append(markdown_table(mp, ["event_group", "method", "n", "rmse_mean", "delta_vs_latest_mean", "beats_latest_rate"], 60))
    lines.append("")
    lines.append("## validation 选择出的 selector")
    lines.append("")
    lines.append(
        markdown_table(
            chosen,
            [
                "chosen_type",
                "selector_tag",
                "threshold",
                "feature_block",
                "model_name",
                "feature_n",
                "val_bad_top10_delta_vs_latest_mean",
                "val_all_delta_vs_latest_mean",
                "test_bad_top10_delta_vs_latest_mean",
                "test_bad_top10_vehicle_ambiguous_delta_vs_latest_mean",
                "test_bad_top10_override_rate",
            ],
        )
    )
    lines.append("")
    lines.append("## test diagnostic top selector")
    lines.append("")
    diag = selector_summary[pd.to_numeric(selector_summary["test_bad_top10_override_n"], errors="coerce").fillna(0) > 0].copy()
    diag = diag.sort_values(["test_bad_top10_delta_vs_latest_mean", "val_bad_top10_delta_vs_latest_mean"]).head(20)
    lines.append(
        markdown_table(
            diag,
            [
                "selector_tag",
                "threshold",
                "feature_block",
                "model_name",
                "feature_n",
                "val_bad_top10_delta_vs_latest_mean",
                "val_all_delta_vs_latest_mean",
                "test_bad_top10_delta_vs_latest_mean",
                "test_bad_top10_vehicle_ambiguous_delta_vs_latest_mean",
                "test_bad_top10_override_rate",
                "noharm_val",
            ],
        )
    )
    lines.append("")
    lines.append("## 分类探针")
    lines.append("")
    cls_top = cls.sort_values(["target", "split", "auc"], ascending=[True, True, False]).groupby(["target", "split"]).head(5)
    lines.append(markdown_table(cls_top, ["target", "split", "feature_block", "model_name", "n", "positive_rate", "auc", "average_precision", "feature_n"], 80))
    lines.append("")
    lines.append("## 源生理特征筛选概况")
    lines.append("")
    lines.append(markdown_table(screen, ["feature", "source", "finite_rate_train", "behavior_corr_max", "identity_eta", "low_identity_candidate"], 40))
    lines.append("")
    lines.append("## guardrail")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    if bool(decision["route_viable_now"].iloc[0]):
        lines.append("## 判断")
        lines.append("")
        lines.append("- v291 找到了 validation no-harm 后仍能在 test 差样本改善的 selector，需要继续复核具体样本与泄漏边界。")
    else:
        lines.append("## 判断")
        lines.append("")
        lines.append("- v291 没有找到可部署的多信号生理监督 selector。")
        lines.append("- 如果 method-pool oracle 有上限但 selector 学不到，说明现有生理源信号不足以稳定判断何时覆盖 latest。")
        lines.append("- 如果分类 AUC 有弱信号但 selector 不改善，说明生理更适合做可观测性/不确定性分层，而不是直接选择预测方法。")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_input_hashes() -> None:
    rows = []
    for name, path in {
        "v278_candidates": V278_CANDIDATES,
        "v288_features": V288_FEATURES,
        "v289_features": V289_FEATURES,
        "v290_features": V290_FEATURES,
    }.items():
        rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "sha256": file_sha256(path) if path.exists() else None,
            }
        )
    write_csv(pd.DataFrame(rows), LOGS / "input_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    for path in OUT.rglob("*"):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size})
    write_csv(pd.DataFrame(rows).sort_values("relative_path"), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in OUT.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(OUT.parent))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip() is None


def main() -> None:
    print("[v291] 目的：合并 ECG/RESP/EDA 源信号，做监督 selector/分类探针。")
    clean_out_dir()

    source = load_source_features()
    methods = load_candidate_methods()
    df = source.merge(methods, on="event_uid", how="inner", validate="one_to_one")
    if "split" in df.columns and "split_from_candidates" in df.columns:
        mismatch = df["split"].astype(str).ne(df["split_from_candidates"].astype(str)).sum()
        if mismatch:
            raise RuntimeError(f"source split 与 v278 candidate split 不一致：{mismatch}")
    for col in ["high_future_abs_q75", "high_future_range_q75", "bad_top10_v250_diagnostic", "bio290_eda_event_usable"]:
        if col in df.columns:
            df[col] = safe_bool_series(df[col])

    bio_cols = [c for c in df.columns if c.startswith(("bio288_", "bio289_", "bio290_")) and pd.api.types.is_numeric_dtype(df[c])]
    screen = screen_bio_features(df, bio_cols)
    feature_blocks = build_feature_blocks(df, screen)
    method_pool = summarize_method_pool(df)
    selector_summary, selector_events = run_selector_probe(df, feature_blocks)
    cls = run_classification_probe(df, feature_blocks)
    chosen = choose_deployable(selector_summary)
    decision = route_decision(chosen, method_pool, cls)

    write_csv(df, TABLES / "v291_multisignal_event_table.csv")
    write_csv(screen, TABLES / "v291_train_only_bio_feature_screen.csv")
    write_csv(pd.DataFrame([{"feature_block": k, "feature_n": len(v), "features": json.dumps(v, ensure_ascii=False)} for k, v in feature_blocks.items()]), TABLES / "v291_feature_blocks.csv")
    write_csv(method_pool, TABLES / "v291_method_pool_summary.csv")
    write_csv(selector_summary, TABLES / "v291_selector_threshold_summary.csv")
    write_csv(selector_events, TABLES / "v291_selector_per_event_thresholds.csv")
    write_csv(chosen, TABLES / "v291_selector_chosen_by_val.csv")
    write_csv(cls, TABLES / "v291_classification_probe_summary.csv")
    write_csv(decision, TABLES / "v291_route_decision.csv")

    plot_method_pool(method_pool)
    plot_selector_choices(chosen)
    plot_feature_screen(screen)

    guardrail = {
        "pass": True,
        "event_n": int(len(df)),
        "train_n": int(df["split"].astype(str).eq("train").sum()),
        "val_n": int(df["split"].astype(str).eq("val").sum()),
        "test_n": int(df["split"].astype(str).eq("test").sum()),
        "bio_source_feature_n": int(len(bio_cols)),
        "screen_feature_n": int(len(screen)),
        "feature_block_n": int(len(feature_blocks)),
        "selector_config_n": int(selector_summary["selector_tag"].nunique()),
        "route_viable_now": bool(decision["route_viable_now"].iloc[0]),
        "method_pool_test_badtop10_oracle_delta": float(
            method_pool[
                method_pool["split"].eq("test")
                & method_pool["event_group"].eq("bad_top10")
                & method_pool["method"].eq("oracle_best_of_methods")
            ]["delta_vs_latest_mean"].iloc[0]
        ),
        "best_val_noharm_active_exists": bool(chosen["chosen_type"].astype(str).eq("best_val_noharm_active").any()) if len(chosen) else False,
        "best_deployable_test_badtop10_delta": None,
        "best_test_diagnostic_badtop10_delta": None,
        "test_used_for_feature_screen_or_threshold": False,
    }
    if len(chosen):
        dep = chosen[chosen["chosen_type"].astype(str).eq("best_val_noharm_active")]
        if len(dep):
            guardrail["best_deployable_test_badtop10_delta"] = float(dep["test_bad_top10_delta_vs_latest_mean"].iloc[0])
        diag = chosen[chosen["chosen_type"].astype(str).eq("test_best_diagnostic_not_deployable")]
        if len(diag):
            guardrail["best_test_diagnostic_badtop10_delta"] = float(diag["test_bad_top10_delta_vs_latest_mean"].iloc[0])

    write_report(method_pool, selector_summary, chosen, cls, screen, decision, guardrail)
    write_input_hashes()
    guardrail["zip_testzip"] = False
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()
    guardrail["zip_testzip"] = bool(make_zip())
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()
    guardrail["zip_testzip"] = bool(make_zip())
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()

    print(f"[v291] report={REPORTS / 'v291_multisignal_physio_supervised_probe_cn.md'}")
    print(f"[v291] zip={ZIP_PATH}")
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
