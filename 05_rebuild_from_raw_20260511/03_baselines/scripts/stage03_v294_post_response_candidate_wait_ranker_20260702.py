#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v294 post-response candidate wait ranker.

本轮目标：
- v293 证明：主差样本 bad_top10 在 observation 前生理不可见，但 observation 后 0-3s
  生理响应对 bad_top10 有明显可见性；
- v294 把这个发现转成真正的预测任务：等待 1/2/3/5 秒后，使用 query 与 train prototype
  的 post-response 生理匹配来重新选择 v292 的 vehicle-similar 候选；
- 评价必须回到 RMSE，并且 threshold 只允许 val 选择，test 只报告。

边界：
- post 特征只代表短等待/延迟观测策略，不是原锚点即时输入；
- prototype 仍来自 train split；
- feature screening 只用 train query；
- pairwise ranker 只用 train query 训练；
- test 不参与窗口、特征、模型、阈值选择。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

OUT = BASELINES / "v294_post_response_candidate_wait_ranker_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v294_post_response_candidate_wait_ranker_20260702_pack.zip"

V292_PAIR_TABLE = (
    BASELINES
    / "v292_source_physio_pairwise_candidate_ranker_20260702"
    / "tables"
    / "v292_pairwise_candidate_table.csv"
)
V292_ORACLE = (
    BASELINES
    / "v292_source_physio_pairwise_candidate_ranker_20260702"
    / "tables"
    / "v292_candidate_pool_oracle_summary.csv"
)
V293_FEATURE_TABLE = (
    BASELINES
    / "v293_physio_response_visibility_latency_audit_20260702"
    / "tables"
    / "v293_prepost_physio_visibility_features.csv"
)
V293_GUARDRAIL = (
    BASELINES
    / "v293_physio_response_visibility_latency_audit_20260702"
    / "logs"
    / "guardrail_check.json"
)

SEED = 29402

WAIT_SPECS = {
    "wait1_post0_1": {"windows": ["post0_1"], "wait_s": 1.0},
    "wait2_post0_2": {"windows": ["post0_2"], "wait_s": 2.0},
    "wait3_post0_3": {"windows": ["post0_3"], "wait_s": 3.0},
    "wait5_post0_5": {"windows": ["post0_5"], "wait_s": 5.0},
}

WINDOW_METRIC_SUFFIXES = [
    "valid_ratio",
    "z_mean",
    "z_abs_mean",
    "z_std",
    "z_range",
    "z_p05",
    "z_p95",
    "z_last_minus_first",
    "z_slope",
    "line_length_per_s",
]

VEHICLE_COLS = [
    "pred_rank_score",
    "vehicle_score_rank_desc",
    "vehicle_score_rank_pct",
    "vehicle_score_best",
    "vehicle_score_margin_to_best",
    "vehicle_score_mean",
    "vehicle_score_std",
    "vehicle_score_z_within_event",
    "candidate_delay_s",
    "candidate_is_latest_delay",
]

GROUP_FLAGS = {
    "all": None,
    "bad_top10": "bad_top10",
    "vehicle_ambiguous": "vehicle_ambiguous",
    "bad_top10_vehicle_ambiguous": "bad_top10_vehicle_ambiguous",
    "candidate_pool_gain_gt_005": "candidate_pool_gain_gt_005",
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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).ne(0)
    return s.astype(str).str.lower().isin(["1", "true", "yes", "y"])


def finite(v: Iterable[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    return arr[np.isfinite(arr)]


def abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return 0.0
    xx = x[mask]
    yy = y[mask]
    if float(np.nanstd(xx)) <= 1e-12 or float(np.nanstd(yy)) <= 1e-12:
        return 0.0
    return float(abs(np.corrcoef(xx, yy)[0, 1]))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_input_hashes() -> None:
    rows = []
    for p in [V292_PAIR_TABLE, V292_ORACLE, V293_FEATURE_TABLE, V293_GUARDRAIL]:
        rows.append(
            {
                "path": str(p),
                "exists": p.exists(),
                "sha256": file_sha256(p) if p.exists() else "",
                "size": p.stat().st_size if p.exists() else 0,
            }
        )
    write_csv(pd.DataFrame(rows), LOGS / "input_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    for p in OUT.rglob("*"):
        if p.is_file():
            rows.append({"path": str(p.relative_to(OUT)), "size": p.stat().st_size})
    write_csv(pd.DataFrame(rows).sort_values("path"), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in OUT.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(OUT)))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip() is None


def load_base_pairs() -> pd.DataFrame:
    need = [
        "event_uid",
        "split",
        "subject",
        "recording",
        "prototype_event_uid",
        "prototype_split",
        "prototype_subject",
        "target_tail_rmse_v241",
        "latest_tail_rmse_v241",
        "target_gain_vs_latest",
        "rank_target_z",
        "pred_rank_score",
        "vehicle_score_rank_desc",
        "vehicle_score_rank_pct",
        "vehicle_score_best",
        "vehicle_score_margin_to_best",
        "vehicle_score_mean",
        "vehicle_score_std",
        "vehicle_score_z_within_event",
        "candidate_delay_s",
        "candidate_is_latest_delay",
        "bad_top10",
        "vehicle_ambiguous",
        "bad_top10_vehicle_ambiguous",
    ]
    pairs = pd.read_csv(V292_PAIR_TABLE, usecols=lambda c: c in set(need), low_memory=False)
    for col in [
        "target_tail_rmse_v241",
        "latest_tail_rmse_v241",
        "target_gain_vs_latest",
        "rank_target_z",
        *VEHICLE_COLS,
    ]:
        if col in pairs.columns:
            pairs[col] = pd.to_numeric(pairs[col], errors="coerce")
    for col in ["bad_top10", "vehicle_ambiguous", "bad_top10_vehicle_ambiguous"]:
        if col in pairs.columns:
            pairs[col] = safe_bool_series(pairs[col]).astype(int)
    return pairs


def load_features() -> pd.DataFrame:
    features = pd.read_csv(V293_FEATURE_TABLE, low_memory=False)
    for col in [
        "bad_top10",
        "vehicle_ambiguous",
        "bad_top10_vehicle_ambiguous",
        "candidate_pool_gain_gt_005",
        "candidate_pool_gain_gt_02",
    ]:
        if col in features.columns:
            features[col] = safe_bool_series(features[col]).astype(int)
    return features


def attach_event_flags(base_pairs: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """把 v293 事件级标签挂到 base pair，保证 oracle 与 wait-ranker 使用同一组分层。"""

    flag_cols = [
        "event_uid",
        "candidate_pool_gain_gt_005",
        "candidate_pool_gain_gt_02",
        "v293_status",
    ]
    flag_cols = [c for c in flag_cols if c in features.columns]
    flags = features[flag_cols].drop_duplicates("event_uid").copy()
    out = base_pairs.merge(flags, on="event_uid", how="left", validate="many_to_one")
    for col in ["candidate_pool_gain_gt_005", "candidate_pool_gain_gt_02"]:
        if col in out.columns:
            out[col] = safe_bool_series(out[col]).astype(int)
    return out


def columns_for_windows(features: pd.DataFrame, windows: List[str]) -> List[str]:
    cols: List[str] = []
    for col in features.columns:
        if not col.startswith("v293_"):
            continue
        if not any(col.startswith(f"v293_{win}_") for win in windows):
            continue
        if not any(col.endswith(suf) for suf in WINDOW_METRIC_SUFFIXES):
            continue
        cols.append(col)
    return cols


def build_wait_pair_table(base_pairs: pd.DataFrame, features: pd.DataFrame, wait_name: str, windows: List[str]) -> Tuple[pd.DataFrame, List[str], Dict[str, object]]:
    """把 query/prototype 的 post-response 生理窗口拼成候选 pair 表。"""

    feature_cols = columns_for_windows(features, windows)
    meta_cols = ["event_uid"]
    meta_cols = [c for c in meta_cols if c in features.columns]
    small = features[meta_cols + feature_cols].drop_duplicates("event_uid").copy()

    q = small.rename(columns={c: f"query__{c}" for c in feature_cols})
    p = small[["event_uid"] + feature_cols].rename(
        columns={"event_uid": "prototype_event_uid", **{c: f"proto__{c}" for c in feature_cols}}
    )
    pairs = base_pairs.merge(q, on="event_uid", how="left", validate="many_to_one")
    pairs = pairs.merge(p, on="prototype_event_uid", how="left", validate="many_to_one")
    pair_cols: Dict[str, pd.Series] = {}
    for col in feature_cols:
        qcol = f"query__{col}"
        pcol = f"proto__{col}"
        qv = pd.to_numeric(pairs[qcol], errors="coerce")
        pv = pd.to_numeric(pairs[pcol], errors="coerce")
        pair_cols[f"pair_absdiff__{col}"] = (qv - pv).abs()
        pair_cols[f"pair_signeddiff__{col}"] = qv - pv
    pairs = pd.concat([pairs, pd.DataFrame(pair_cols, index=pairs.index)], axis=1)

    generated = [f"query__{c}" for c in feature_cols] + [f"proto__{c}" for c in feature_cols] + list(pair_cols.keys())
    audit = {
        "wait_name": wait_name,
        "windows": ",".join(windows),
        "source_feature_n": len(feature_cols),
        "generated_feature_n": len(generated),
        "event_n": int(pairs["event_uid"].nunique()),
        "candidate_rows": int(len(pairs)),
        "prototype_train_only": bool(pairs["prototype_split"].dropna().astype(str).eq("train").all()),
        "query_v293_ok_rate": float(pairs.drop_duplicates("event_uid")["v293_status"].astype(str).eq("ok").mean())
        if "v293_status" in pairs.columns
        else math.nan,
    }
    return pairs, generated, audit


def screen_features(pairs: pd.DataFrame, candidate_cols: List[str]) -> pd.DataFrame:
    train = pairs["split"].astype(str).eq("train")
    y_gain = pd.to_numeric(pairs.loc[train, "target_gain_vs_latest"], errors="coerce").to_numpy(dtype=float)
    y_rank = pd.to_numeric(pairs.loc[train, "rank_target_z"], errors="coerce").to_numpy(dtype=float)
    rows: List[Dict[str, object]] = []
    for col in candidate_cols:
        if col not in pairs.columns:
            continue
        x = pd.to_numeric(pairs.loc[train, col], errors="coerce").to_numpy(dtype=float)
        finite_rate = float(np.isfinite(x).mean()) if len(x) else 0.0
        std = float(np.nanstd(x)) if len(x) else 0.0
        if finite_rate < 0.55 or std <= 1e-12:
            continue
        corr_gain = abs_corr(x, y_gain)
        corr_rank = abs_corr(x, y_rank)
        rows.append(
            {
                "feature": col,
                "finite_rate_train": finite_rate,
                "std_train": std,
                "corr_target_gain_train": corr_gain,
                "corr_rank_target_train": corr_rank,
                "max_abs_corr_train": max(corr_gain, corr_rank),
                "source_kind": "vehicle"
                if col in VEHICLE_COLS
                else ("query" if col.startswith("query__") else ("proto" if col.startswith("proto__") else "pair")),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("max_abs_corr_train", ascending=False).reset_index(drop=True)


def feature_blocks(screen: pd.DataFrame) -> Dict[str, List[str]]:
    vehicle = [c for c in VEHICLE_COLS if c in set(screen["feature"])]
    pair_rank = screen[~screen["feature"].isin(vehicle)]["feature"].tolist()
    return {
        "vehicle_meta_only": vehicle,
        "post_response_pair_top64": pair_rank[:64],
        "vehicle_post_response_pair_top96": vehicle + pair_rank[:96],
    }


def model_factory() -> Dict[str, Pipeline]:
    return {
        "ridge_a10": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0, random_state=SEED)),
            ]
        ),
        "hgb_d3": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=120,
                        max_leaf_nodes=15,
                        learning_rate=0.06,
                        l2_regularization=0.10,
                        random_state=SEED,
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
                        n_estimators=180,
                        max_depth=6,
                        min_samples_leaf=10,
                        random_state=SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def fit_predict_gain(pairs: pd.DataFrame, cols: List[str], model: Pipeline) -> np.ndarray:
    train = pairs["split"].astype(str).eq("train")
    X = pairs[cols].replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(pairs["target_gain_vs_latest"], errors="coerce").to_numpy(dtype=float)
    ok = train.to_numpy() & np.isfinite(y)
    model.fit(X.loc[ok], y[ok])
    return np.asarray(model.predict(X), dtype=float)


def top_candidate_per_event(pairs: pd.DataFrame, pred_gain: np.ndarray, selector_tag: str, wait_s: float) -> pd.DataFrame:
    tmp = pairs[
        [
            "event_uid",
            "split",
            "subject",
            "prototype_event_uid",
            "target_tail_rmse_v241",
            "latest_tail_rmse_v241",
            "target_gain_vs_latest",
            "pred_rank_score",
            "candidate_delay_s",
            "bad_top10",
            "vehicle_ambiguous",
            "bad_top10_vehicle_ambiguous",
            "candidate_pool_gain_gt_005",
        ]
    ].copy()
    tmp["pred_gain_vs_latest"] = pred_gain
    tmp["selector_tag"] = selector_tag
    tmp["wait_s"] = float(wait_s)
    top = (
        tmp.sort_values(["event_uid", "pred_gain_vs_latest", "target_tail_rmse_v241"], ascending=[True, False, True])
        .drop_duplicates("event_uid")
        .rename(
            columns={
                "target_tail_rmse_v241": "candidate_rmse",
                "latest_tail_rmse_v241": "latest_rmse",
                "candidate_delay_s": "prototype_delay_s",
            }
        )
    )
    return top


def summarize_selected(selected: pd.DataFrame, split: str, group: str, flag: str | None) -> Dict[str, object]:
    sub = selected[selected["split"].astype(str).eq(split)].copy()
    if flag is not None:
        if flag not in sub.columns:
            sub = sub.iloc[0:0]
        else:
            sub = sub[safe_bool_series(sub[flag]).astype(bool)]
    if sub.empty:
        return {
            "n": 0,
            "latest_rmse_mean": math.nan,
            "selected_rmse_mean": math.nan,
            "delta_vs_latest_mean": math.nan,
            "override_n": 0,
            "override_rate": math.nan,
            "wait_s_mean": math.nan,
        }
    latest = pd.to_numeric(sub["latest_rmse"], errors="coerce")
    selected_rmse = pd.to_numeric(sub["selected_rmse"], errors="coerce")
    override = safe_bool_series(sub["override_latest"])
    return {
        "n": int(len(sub)),
        "latest_rmse_mean": float(latest.mean()),
        "selected_rmse_mean": float(selected_rmse.mean()),
        "delta_vs_latest_mean": float(selected_rmse.mean() - latest.mean()),
        "override_n": int(override.sum()),
        "override_rate": float(override.mean()),
        "wait_s_mean": float(pd.to_numeric(sub["wait_s"], errors="coerce").mean()),
    }


def threshold_search(top: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val = pd.to_numeric(top.loc[top["split"].astype(str).eq("val"), "pred_gain_vs_latest"], errors="coerce")
    positive = val[np.isfinite(val) & (val > 0)]
    thresholds = [float("inf")]
    if len(positive):
        thresholds += [float(x) for x in np.unique(np.nanquantile(positive, np.linspace(0.05, 0.95, 19)))]
        thresholds.append(0.0)
    thresholds = sorted(set(thresholds), key=lambda x: (math.isinf(x), x))

    selected_all: List[pd.DataFrame] = []
    rows: List[Dict[str, object]] = []
    for threshold in thresholds:
        selected = top.copy()
        selected["threshold"] = threshold
        selected["override_latest"] = pd.to_numeric(selected["pred_gain_vs_latest"], errors="coerce") >= threshold
        selected.loc[np.isinf(threshold), "override_latest"] = False
        selected["selected_rmse"] = np.where(selected["override_latest"], selected["candidate_rmse"], selected["latest_rmse"])
        selected_all.append(selected)

        row: Dict[str, object] = {"selector_tag": str(top["selector_tag"].iloc[0]), "threshold": threshold}
        for split in ["train", "val", "test"]:
            for group, flag in GROUP_FLAGS.items():
                m = summarize_selected(selected, split, group, flag)
                prefix = f"{split}_{group}"
                for key, value in m.items():
                    row[f"{prefix}_{key}"] = value
        row["active_val_bad_top10"] = int(row.get("val_bad_top10_override_n", 0) or 0) > 0
        row["noharm_val"] = (
            float(row.get("val_bad_top10_delta_vs_latest_mean", math.inf)) <= 0.0
            and float(row.get("val_all_delta_vs_latest_mean", math.inf)) <= 0.003
            and float(row.get("val_vehicle_ambiguous_delta_vs_latest_mean", math.inf)) <= 0.005
        )
        row["noharm_active_val"] = bool(row["noharm_val"] and row["active_val_bad_top10"])
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(selected_all, ignore_index=True)


def run_wait_rankers(base_pairs: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: List[pd.DataFrame] = []
    selected_events: List[pd.DataFrame] = []
    feature_rows: List[Dict[str, object]] = []
    screen_rows: List[pd.DataFrame] = []

    for wait_name, spec in WAIT_SPECS.items():
        print(f"[v294] build wait table {wait_name}")
        pairs, generated, audit = build_wait_pair_table(base_pairs, features, wait_name, spec["windows"])
        screen = screen_features(pairs, VEHICLE_COLS + generated)
        screen["wait_name"] = wait_name
        screen_rows.append(screen)
        blocks = feature_blocks(screen)
        for block_name, cols in blocks.items():
            cols = [c for c in cols if c in pairs.columns]
            if not cols:
                continue
            for model_name, model in model_factory().items():
                tag = f"{wait_name}__{block_name}__{model_name}"
                print(f"[v294] ranker {tag} feature_n={len(cols)}")
                pred = fit_predict_gain(pairs, cols, model)
                top = top_candidate_per_event(pairs, pred, tag, float(spec["wait_s"]))
                summary, selected = threshold_search(top)
                summary["wait_name"] = wait_name
                summary["wait_s"] = float(spec["wait_s"])
                summary["feature_block"] = block_name
                summary["model_name"] = model_name
                summary["feature_n"] = len(cols)
                selected["wait_name"] = wait_name
                selected["feature_block"] = block_name
                selected["model_name"] = model_name
                summaries.append(summary)
                selected_events.append(selected)
                feature_rows.append(
                    {
                        **audit,
                        "feature_block": block_name,
                        "model_name": model_name,
                        "feature_n": len(cols),
                        "features": json.dumps(cols, ensure_ascii=False),
                    }
                )
    return (
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
        pd.concat(selected_events, ignore_index=True) if selected_events else pd.DataFrame(),
        pd.DataFrame(feature_rows),
        pd.concat(screen_rows, ignore_index=True) if screen_rows else pd.DataFrame(),
    )


def choose_deployable(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.Series] = []
    if summary.empty:
        return pd.DataFrame()
    cand = summary[summary["noharm_active_val"].astype(bool)].copy()
    if len(cand):
        cand = cand.sort_values(
            ["val_bad_top10_delta_vs_latest_mean", "val_all_delta_vs_latest_mean", "threshold"],
            ascending=[True, True, False],
        )
        row = cand.iloc[0].copy()
        row["chosen_type"] = "best_val_noharm_active"
        rows.append(row)
    fallback = summary[np.isinf(pd.to_numeric(summary["threshold"], errors="coerce"))].copy()
    if len(fallback):
        row = fallback.iloc[0].copy()
        row["chosen_type"] = "fallback_no_override"
        rows.append(row)
    diag = summary.sort_values("test_bad_top10_delta_vs_latest_mean", ascending=True).iloc[0].copy()
    diag["chosen_type"] = "test_best_diagnostic_not_deployable"
    rows.append(diag)
    return pd.DataFrame(rows).reset_index(drop=True)


def oracle_pair_summary(base_pairs: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    oracle = base_pairs.sort_values(["event_uid", "target_tail_rmse_v241"]).drop_duplicates("event_uid").copy()
    vehicle = (
        base_pairs.sort_values(["event_uid", "pred_rank_score", "target_tail_rmse_v241"], ascending=[True, False, True])
        .drop_duplicates("event_uid")
        .copy()
    )
    for policy, top in [("oracle_best_candidate", oracle), ("vehicle_score_top1_no_threshold", vehicle)]:
        selected = top.rename(columns={"target_tail_rmse_v241": "selected_rmse", "latest_tail_rmse_v241": "latest_rmse"}).copy()
        selected["override_latest"] = True
        selected["wait_s"] = 0.0
        for split in ["train", "val", "test"]:
            for group, flag in GROUP_FLAGS.items():
                m = summarize_selected(selected, split, group, flag)
                rows.append({"policy": policy, "split": split, "event_group": group, **m})
    return pd.DataFrame(rows)


def route_decision(chosen: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    deploy = chosen[chosen["chosen_type"].astype(str).eq("best_val_noharm_active")].copy() if len(chosen) else pd.DataFrame()
    if deploy.empty:
        route_ok = False
        evidence = "validation 没有 no-harm active 等待后候选选择器"
    else:
        bad_delta = float(deploy["test_bad_top10_delta_vs_latest_mean"].iloc[0])
        all_delta = float(deploy["test_all_delta_vs_latest_mean"].iloc[0])
        route_ok = bool(bad_delta <= -0.05 and all_delta <= 0.01)
        evidence = f"deployable test bad_top10 delta={bad_delta:.4f}, all delta={all_delta:.4f}"
    oracle_bad = oracle[
        oracle["policy"].eq("oracle_best_candidate")
        & oracle["split"].eq("test")
        & oracle["event_group"].eq("bad_top10")
    ]
    oracle_delta = float(oracle_bad["delta_vs_latest_mean"].iloc[0]) if len(oracle_bad) else math.nan
    return pd.DataFrame(
        [
            {
                "check": "post_response_wait_ranker",
                "requirement": "val no-harm active, and test bad_top10 improves by at least 0.05 RMSE without all-sample harm",
                "pass": route_ok,
                "evidence": evidence,
                "candidate_pool_oracle_test_badtop10_delta": oracle_delta,
                "deployable": route_ok,
                "route_viable_now": route_ok,
            }
        ]
    )


def markdown_table(df: pd.DataFrame, cols: List[str], max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_empty_"
    show_cols = [c for c in cols if c in df.columns]
    return df[show_cols].head(max_rows).to_markdown(index=False)


def plot_chosen(chosen: pd.DataFrame) -> Path:
    data = chosen.copy()
    if data.empty:
        data = pd.DataFrame(
            {
                "chosen_type": ["empty"],
                "test_bad_top10_delta_vs_latest_mean": [0.0],
                "test_all_delta_vs_latest_mean": [0.0],
            }
        )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(data))
    ax.bar(x - 0.18, data["test_bad_top10_delta_vs_latest_mean"], width=0.36, label="test bad_top10")
    ax.bar(x + 0.18, data["test_all_delta_vs_latest_mean"], width=0.36, label="test all")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(data["chosen_type"].astype(str), rotation=20, ha="right")
    ax.set_ylabel("delta vs latest RMSE")
    ax.set_title("v294 chosen wait-ranker policies")
    ax.legend()
    fig.tight_layout()
    path = FIGURES / "v294_chosen_wait_ranker_delta.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_wait_diagnostic(summary: pd.DataFrame) -> Path:
    data = (
        summary[summary["threshold"].astype(float).replace(np.inf, np.nan).notna()]
        .sort_values("test_bad_top10_delta_vs_latest_mean")
        .head(20)
        .copy()
    )
    if data.empty:
        data = summary.sort_values("test_bad_top10_delta_vs_latest_mean").head(20).copy()
    if data.empty:
        data = pd.DataFrame({"selector_tag": ["empty"], "test_bad_top10_delta_vs_latest_mean": [0.0]})
    data["label"] = data["selector_tag"].astype(str).str.replace("__", "\n", regex=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(data["label"], data["test_bad_top10_delta_vs_latest_mean"], color="#4E79A7")
    ax.axvline(0, color="black", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("test bad_top10 delta vs latest RMSE")
    ax.set_title("v294 top diagnostic wait-ranker selectors")
    fig.tight_layout()
    path = FIGURES / "v294_top_diagnostic_wait_rankers.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(decision: pd.DataFrame, chosen: pd.DataFrame, summary: pd.DataFrame, oracle: pd.DataFrame, feature_audit: pd.DataFrame, guardrail: Dict[str, object]) -> Path:
    lines: List[str] = []
    lines.append("# v294 post-response candidate wait ranker")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- v293 说明 observation 后短时间生理响应对 bad_top10 有可见性。")
    lines.append("- v294 检查这种可见性能否转成真实 RMSE 改善，而不是只停留在分类 AUC。")
    lines.append("- post-response 特征只代表等待 1/2/3/5 秒后的延迟观测策略，不是原锚点即时输入。")
    lines.append("")
    lines.append("## route decision")
    lines.append("")
    lines.append(markdown_table(decision, ["check", "requirement", "pass", "evidence", "candidate_pool_oracle_test_badtop10_delta", "deployable", "route_viable_now"]))
    lines.append("")
    lines.append("## validation 选择出的策略")
    lines.append("")
    lines.append(
        markdown_table(
            chosen,
            [
                "chosen_type",
                "selector_tag",
                "threshold",
                "wait_s",
                "feature_block",
                "model_name",
                "feature_n",
                "val_bad_top10_delta_vs_latest_mean",
                "val_all_delta_vs_latest_mean",
                "test_bad_top10_delta_vs_latest_mean",
                "test_bad_top10_vehicle_ambiguous_delta_vs_latest_mean",
                "test_bad_top10_override_rate",
                "test_all_delta_vs_latest_mean",
            ],
        )
    )
    lines.append("")
    lines.append("## 候选池边界")
    lines.append("")
    lines.append(markdown_table(oracle[oracle["split"].eq("test")], ["policy", "event_group", "n", "latest_rmse_mean", "selected_rmse_mean", "delta_vs_latest_mean", "override_rate"], 20))
    lines.append("")
    lines.append("## test-best 诊断 top")
    lines.append("")
    lines.append(
        markdown_table(
            summary.sort_values("test_bad_top10_delta_vs_latest_mean"),
            [
                "selector_tag",
                "threshold",
                "wait_s",
                "feature_block",
                "model_name",
                "feature_n",
                "noharm_active_val",
                "val_bad_top10_delta_vs_latest_mean",
                "val_all_delta_vs_latest_mean",
                "test_bad_top10_delta_vs_latest_mean",
                "test_bad_top10_vehicle_ambiguous_delta_vs_latest_mean",
                "test_bad_top10_override_rate",
                "test_all_delta_vs_latest_mean",
            ],
            30,
        )
    )
    lines.append("")
    lines.append("## feature audit")
    lines.append("")
    lines.append(markdown_table(feature_audit, ["wait_name", "windows", "source_feature_n", "generated_feature_n", "feature_block", "model_name", "feature_n", "prototype_train_only", "query_v293_ok_rate"], 30))
    lines.append("")
    lines.append("## 判断")
    lines.append("")
    if bool(guardrail.get("route_viable_now", False)):
        lines.append("- v294 找到 validation no-harm 后仍能改善 test bad_top10 的等待后生理候选排序路线，需要继续做样本级图像复核。")
    else:
        lines.append("- v294 没有找到可部署的 post-response 候选选择策略。")
        lines.append("- 若 test-best 诊断很好但 val 不支持，说明 post 生理能识别差样本风险，但仍不一定知道该选哪个未来候选。")
        lines.append("- 下一步必须区分“识别会失败”和“生成/选择更好轨迹”这两个任务。")
    path = REPORTS / "v294_post_response_candidate_wait_ranker_cn.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    print("[v294] 目的：把 post-response 生理可见性转成等待后候选选择 RMSE 验证。")
    clean_out_dir()
    base_pairs = load_base_pairs()
    features = load_features()
    base_pairs = attach_event_flags(base_pairs, features)
    oracle = oracle_pair_summary(base_pairs)
    summary, selected, feature_audit, screen = run_wait_rankers(base_pairs, features)
    chosen = choose_deployable(summary)
    decision = route_decision(chosen, oracle)

    write_csv(oracle, TABLES / "v294_candidate_pool_oracle_summary.csv")
    write_csv(screen, TABLES / "v294_train_only_feature_screen.csv")
    write_csv(feature_audit, TABLES / "v294_feature_block_audit.csv")
    write_csv(summary, TABLES / "v294_wait_ranker_threshold_summary.csv")
    # per-event-threshold 表较大，但后续人工抽样要用；保留精简后的 event 级结果。
    keep_cols = [
        "event_uid",
        "split",
        "subject",
        "prototype_event_uid",
        "candidate_rmse",
        "latest_rmse",
        "pred_gain_vs_latest",
        "threshold",
        "override_latest",
        "selected_rmse",
        "wait_s",
        "selector_tag",
        "wait_name",
        "feature_block",
        "model_name",
        "bad_top10",
        "vehicle_ambiguous",
        "bad_top10_vehicle_ambiguous",
        "candidate_pool_gain_gt_005",
    ]
    write_csv(selected[[c for c in keep_cols if c in selected.columns]], TABLES / "v294_wait_ranker_selected_per_event_thresholds.csv")
    write_csv(chosen, TABLES / "v294_wait_ranker_chosen_by_val.csv")
    write_csv(decision, TABLES / "v294_route_decision.csv")
    plot_chosen(chosen)
    plot_wait_diagnostic(summary)

    v293_guard = json.loads(V293_GUARDRAIL.read_text(encoding="utf-8")) if V293_GUARDRAIL.exists() else {}
    guardrail = {
        "pass": True,
        "event_n": int(base_pairs["event_uid"].nunique()),
        "candidate_rows": int(len(base_pairs)),
        "wait_policy_n": int(len(WAIT_SPECS)),
        "selector_config_n": int(summary["selector_tag"].nunique()) if len(summary) else 0,
        "uses_post_observation": True,
        "post_features_are_wait_policy_only": True,
        "route_viable_now": bool(decision["route_viable_now"].iloc[0]),
        "best_val_noharm_active_exists": bool(chosen["chosen_type"].astype(str).eq("best_val_noharm_active").any()) if len(chosen) else False,
        "candidate_pool_test_badtop10_oracle_delta": float(
            oracle[
                oracle["policy"].eq("oracle_best_candidate")
                & oracle["split"].eq("test")
                & oracle["event_group"].eq("bad_top10")
            ]["delta_vs_latest_mean"].iloc[0]
        ),
        "vehicle_score_top1_test_badtop10_delta": float(
            oracle[
                oracle["policy"].eq("vehicle_score_top1_no_threshold")
                & oracle["split"].eq("test")
                & oracle["event_group"].eq("bad_top10")
            ]["delta_vs_latest_mean"].iloc[0]
        ),
        "best_deployable_test_badtop10_delta": None,
        "best_test_diagnostic_badtop10_delta": None,
        "test_used_for_feature_screen_model_or_threshold": False,
        "v293_pre_route_supported_now": bool(v293_guard.get("pre_route_supported_now", False)),
        "v293_post_wait_route_supported_diagnostic": bool(v293_guard.get("post_wait_route_supported_diagnostic", False)),
    }
    if len(chosen):
        dep = chosen[chosen["chosen_type"].astype(str).eq("best_val_noharm_active")]
        if len(dep):
            guardrail["best_deployable_test_badtop10_delta"] = float(dep["test_bad_top10_delta_vs_latest_mean"].iloc[0])
            guardrail["best_deployable_test_all_delta"] = float(dep["test_all_delta_vs_latest_mean"].iloc[0])
            guardrail["best_deployable_wait_s"] = float(dep["wait_s"].iloc[0])
        diag = chosen[chosen["chosen_type"].astype(str).eq("test_best_diagnostic_not_deployable")]
        if len(diag):
            guardrail["best_test_diagnostic_badtop10_delta"] = float(diag["test_bad_top10_delta_vs_latest_mean"].iloc[0])
            guardrail["best_test_diagnostic_wait_s"] = float(diag["wait_s"].iloc[0])

    write_report(decision, chosen, summary, oracle, feature_audit, guardrail)
    write_input_hashes()
    guardrail["zip_testzip"] = False
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()
    guardrail["zip_testzip"] = bool(make_zip())
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()
    print(f"[v294] report={REPORTS / 'v294_post_response_candidate_wait_ranker_cn.md'}")
    print(f"[v294] zip={ZIP_PATH}")
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
