#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v279 physiology reliability filter for v278 listwise candidate.

这一步不再让生理信号直接选择轨迹候选，而是回答一个更具体的问题：

1. 先沿用 v278 的 vehicle-only listwise ranker，为每个事件找一个车辆信息上最像
   “值得替换 latest”的候选轨迹；
2. 再训练一个二级可靠性模型，预测这个替换候选相对 latest 是否真的会降低误差；
3. 分别比较 vehicle、vehicle+bio、vehicle+bio+state、vehicle+style+bio+state；
4. 最终是否替换 latest 仍然只用 validation 阈值决定，test 只做报告和诊断。

这样做的目的不是把生理特征简单拼进最终轨迹模型，而是把生理状态用于“车辆候选是否可信”
这个中间判断，看它能否弥补锚点前车辆序列本身区分度不足的问题。
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
SCRIPTS = BASELINES / "scripts"

OUT = BASELINES / "v279_physio_reliability_filter_for_listrank_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v279_physio_reliability_filter_for_listrank_20260702_pack.zip"

V279_SCRIPT = SCRIPTS / "stage03_v279_physio_reliability_filter_for_listrank_20260702.py"
V278_SCRIPT = SCRIPTS / "stage03_v278_listwise_candidate_rank_loss_20260702.py"
SEED = 27902
FIXED_WAIT_LATEST_BADTOP10 = 0.695048

SEARCH_BASE_COLS = [
    "event_uid",
    "split",
    "subject",
    "recording",
    "mapped_delay_ms",
    "target_tail_rmse_v241",
    "latest_tail_rmse_v241",
    "keep0_tail_rmse_v241",
    "oracle_tail_rmse_v241",
    "bad_top10",
    "very_bad_top5",
    "normal",
    "observe_later_like",
    "strong_steer",
    "early_best_after_400",
    "neighbor_rank_vehicle",
    "vehicle_distance",
    "bio_distance",
]


def import_module_from_path(module_name: str, path: Path):
    """按文件路径导入前序脚本，避免依赖包安装或 PYTHONPATH 设置。"""

    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


V278 = import_module_from_path("stage03_v278_for_v279", V278_SCRIPT)
V277 = V278.V277
V276 = V278.V276


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


def finite_feature_frame(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """取数值特征并删除全空列，保留 HGB 可直接处理的 NaN。"""

    present = [col for col in cols if col in df.columns]
    X = df[present].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().any(axis=0)].copy()
    return X, list(X.columns)


def unique_keep_order(cols: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def build_vehicle_listrank_top(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    复现 v278 的 vehicle-only listwise 第一候选。

    注意这里的第一候选只是“车辆相似和候选排序分数上最像可替换”的候选；
    v279 后续会重新预测它是否可信，而不是无条件使用它。
    """

    ranked = V278.add_listwise_rank_target(df)
    vehicle_cols = V278.feature_sets(ranked)["listrank_vehicle"]
    pred_rank, feature_n = V278.fit_predict_listrank(ranked, vehicle_cols)

    rich = ranked.copy()
    rich["v278_vehicle_rank_score"] = pred_rank
    rich = rich.sort_values(
        ["event_uid", "v278_vehicle_rank_score", "target_tail_rmse_v241", "neighbor_rank_vehicle"],
        ascending=[True, False, True, True],
    ).drop_duplicates("event_uid")
    rich["actual_gain_vs_latest"] = (
        pd.to_numeric(rich["latest_tail_rmse_v241"], errors="coerce")
        - pd.to_numeric(rich["target_tail_rmse_v241"], errors="coerce")
    )
    rich["actual_override_good"] = pd.to_numeric(rich["actual_gain_vs_latest"], errors="coerce") > 0.0
    rich["actual_override_hurts"] = pd.to_numeric(rich["actual_gain_vs_latest"], errors="coerce") < 0.0

    audit = pd.DataFrame(
        [
            {
                "stage": "v278_vehicle_listrank_reproduced",
                "feature_n": int(feature_n),
                "features": "|".join(vehicle_cols),
                "event_n": int(rich["event_uid"].nunique()),
                "candidate_n": int(len(ranked)),
            }
        ]
    )
    return rich.reset_index(drop=True), audit


def reliability_feature_sets(top: pd.DataFrame) -> Dict[str, List[str]]:
    """定义二级可靠性模型特征组，用于检验生理状态是否带来增量。"""

    vehicle_base = [
        "v278_vehicle_rank_score",
        "mapped_delay_ms",
        "neighbor_rank_vehicle",
        "vehicle_distance",
        "pred_pair_base_hgb",
        "pred_pair_vehicle_hgb",
        "pred_gain_vehicle",
        "score_vehicle_gain",
        "score_vehicle_keep0_risk",
        "score_vehicle_badprob",
        "score_vehicle_oracle_gap",
    ]
    bio_pair = [
        "bio_distance",
        "bio271_distance_calibrated",
        "pred_pair_bio_hgb",
        "pred_pair_vehicle_bio_hgb",
        "pred_pair_vehicle_bio_badweighted_hgb",
        "pred_gain_vehicle_bio260_sp64",
        "score_vehicle_bio_gain",
        "score_bio_only_gain",
        "score_vehicle_bio_keep0_risk",
        "score_bio_only_keep0_risk",
        "score_vehicle_bio_badprob",
        "score_bio_only_badprob",
        "score_vehicle_bio_oracle_gap",
    ]
    # v271 已经做过筛选、标准化和 subject/recording 两套状态构造，这里保留为状态表征。
    bio_state = [col for col in top.columns if col.startswith("bio271z__")]
    style_state = ["style_distance_v253_current"] + [col for col in top.columns if col.startswith("stylez__")]

    out = {
        "reliability_vehicle": vehicle_base,
        "reliability_vehicle_bio_pair": vehicle_base + bio_pair,
        "reliability_vehicle_bio_state": vehicle_base + bio_pair + bio_state,
        "reliability_vehicle_style_bio_state": vehicle_base + bio_pair + bio_state + style_state,
    }
    return {name: [col for col in unique_keep_order(cols) if col in top.columns] for name, cols in out.items()}


def weighted_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return math.nan
    aa = a[mask]
    bb = b[mask]
    if float(np.nanstd(aa)) < 1e-12 or float(np.nanstd(bb)) < 1e-12:
        return math.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def fit_predict_reliability(top: pd.DataFrame, cols: List[str], seed_offset: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    训练二级可靠性模型。

    回归器预测“替换候选相对 latest 的真实收益”；分类器预测“替换是否不伤害”。
    两者组合成不同 score，交给 validation 阈值搜索判断是否覆盖 latest。
    """

    X, used_cols = finite_feature_frame(top, cols)
    y_gain = pd.to_numeric(top["actual_gain_vs_latest"], errors="coerce")
    train_mask = top["split"].astype(str).eq("train").to_numpy() & np.isfinite(y_gain.to_numpy(dtype=float))
    if int(train_mask.sum()) < 20:
        raise RuntimeError("训练行数过少，无法训练可靠性模型")

    y_train = y_gain.loc[train_mask].to_numpy(dtype=float)
    bad_weight = top.loc[train_mask, "bad_top10"].astype(bool).to_numpy(dtype=float)
    very_bad_weight = top.loc[train_mask, "very_bad_top5"].astype(bool).to_numpy(dtype=float)
    magnitude_weight = np.minimum(3.0, np.abs(np.nan_to_num(y_train, nan=0.0)) / 0.20)
    sample_weight = 1.0 + 2.0 * bad_weight + 1.5 * very_bad_weight + magnitude_weight

    reg = HistGradientBoostingRegressor(
        max_iter=260,
        learning_rate=0.035,
        max_leaf_nodes=15,
        min_samples_leaf=12,
        l2_regularization=0.20,
        random_state=SEED + seed_offset,
    )
    reg.fit(X.loc[train_mask], y_gain.loc[train_mask], sample_weight=sample_weight)
    pred_gain = reg.predict(X)

    y_good = (y_gain > 0.0).astype(int)
    train_classes = sorted(set(y_good.loc[train_mask].astype(int).tolist()))
    if len(train_classes) >= 2:
        clf = HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=0.20,
            random_state=SEED + 100 + seed_offset,
        )
        clf.fit(X.loc[train_mask], y_good.loc[train_mask], sample_weight=sample_weight)
        prob_good = clf.predict_proba(X)[:, 1]
    else:
        prob_good = np.full(len(top), float(train_classes[0]), dtype=float)

    pred = top[["event_uid", "split", "subject", "recording", "actual_gain_vs_latest", "actual_override_good"]].copy()
    pred["pred_reliability_gain"] = pred_gain
    pred["pred_good_prob"] = prob_good
    pred["score_gain"] = pred_gain
    pred["score_risk_adjusted"] = pred_gain + 0.25 * (prob_good - 0.5)
    pred["score_prob_good"] = prob_good - 0.5

    val_mask = top["split"].astype(str).eq("val").to_numpy()
    audit = {
        "feature_n": int(len(used_cols)),
        "features": "|".join(used_cols),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "val_gain_mae": float(np.nanmean(np.abs(pred_gain[val_mask] - y_gain.to_numpy(dtype=float)[val_mask]))),
        "val_gain_corr": weighted_corr(pred_gain[val_mask], y_gain.to_numpy(dtype=float)[val_mask]),
        "val_good_rate_actual": float(np.nanmean(y_good.to_numpy(dtype=float)[val_mask])),
        "val_good_prob_mean": float(np.nanmean(prob_good[val_mask])),
    }
    return pred, audit


def make_score_top(top: pd.DataFrame, feature_set: str, score: np.ndarray) -> pd.DataFrame:
    out = top[SEARCH_BASE_COLS].copy()
    out["feature_set"] = feature_set
    out["pred_gain_vs_latest"] = score
    return out


def build_reliability_outputs(top: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_rows: List[Dict[str, object]] = []
    pred_rows: List[pd.DataFrame] = []
    score_tops: List[pd.DataFrame] = []

    # v278 原始 rank score 作为对照，验证 v279 评估口径能复现 v278 vehicle-only 筛选。
    score_tops.append(make_score_top(top, "v278_vehicle_rank_score_only", top["v278_vehicle_rank_score"].to_numpy(dtype=float)))
    feature_rows.append(
        {
            "feature_set": "v278_vehicle_rank_score_only",
            "model_kind": "baseline_score",
            "feature_n": 1,
            "features": "v278_vehicle_rank_score",
            "train_rows": int(top["split"].astype(str).eq("train").sum()),
            "val_rows": int(top["split"].astype(str).eq("val").sum()),
            "val_gain_mae": math.nan,
            "val_gain_corr": math.nan,
            "val_good_rate_actual": math.nan,
            "val_good_prob_mean": math.nan,
        }
    )

    for i, (name, cols) in enumerate(reliability_feature_sets(top).items(), start=1):
        pred, audit = fit_predict_reliability(top, cols, seed_offset=i)
        pred.insert(0, "reliability_feature_set", name)
        pred_rows.append(pred)
        feature_rows.append({"feature_set": name, "model_kind": "gain_regressor_plus_good_classifier", **audit})

        score_defs = {
            f"{name}_gain": pred["score_gain"].to_numpy(dtype=float),
            f"{name}_risk_adjusted": pred["score_risk_adjusted"].to_numpy(dtype=float),
            f"{name}_prob_good": pred["score_prob_good"].to_numpy(dtype=float),
        }
        for score_name, score in score_defs.items():
            score_tops.append(make_score_top(top, score_name, score))

    all_top = pd.concat(score_tops, ignore_index=True)
    search, selected = V276.threshold_search(all_top)
    chosen = V276.choose_configs(search)
    return pd.DataFrame(feature_rows), pd.concat(pred_rows, ignore_index=True), all_top, search, selected, chosen


def decision_summary(event_table: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    rows = V276.baseline_decision(event_table[event_table["split"].astype(str).eq("test")])
    for chosen_type in ["best_any", "best_active", "best_stable_active", "best_noharm_all", "test_best_diagnostic"]:
        sub = chosen[chosen["chosen_type"].astype(str).eq(chosen_type)].copy()
        if sub.empty:
            continue
        row = sub.iloc[0]
        rows.append(
            {
                "source": chosen_type,
                "label": f"{row['feature_set']} threshold={row['threshold']}",
                "rmse": float(row["test_bad_top10_selected_rmse"]),
                "deployable": bool(row["deployable"]),
                "override_rate": float(row["test_bad_top10_override_rate"]),
                "val_bad_delta": float(row["val_bad_top10_delta_vs_latest"]),
                "val_all_delta": float(row["val_all_delta_vs_latest"]),
                "stable_pass": bool(row.get("stable_pass", False)),
            }
        )
    out = pd.DataFrame(rows)
    out["delta_vs_fixed_latest"] = pd.to_numeric(out["rmse"], errors="coerce") - FIXED_WAIT_LATEST_BADTOP10
    out["passes_fixed_latest"] = pd.to_numeric(out["rmse"], errors="coerce") < FIXED_WAIT_LATEST_BADTOP10
    return out


def best_test_bad(search: pd.DataFrame, contains: str, require_override: bool = True) -> float:
    sub = search[search["feature_set"].astype(str).str.contains(contains, regex=False)].copy()
    if require_override:
        sub = sub[pd.to_numeric(sub["test_bad_top10_override_n"], errors="coerce") > 0]
    if sub.empty:
        return math.nan
    return float(pd.to_numeric(sub["test_bad_top10_selected_rmse"], errors="coerce").min())


def build_guardrail(
    enriched: pd.DataFrame,
    top: pd.DataFrame,
    feature_audit: pd.DataFrame,
    search: pd.DataFrame,
    chosen: pd.DataFrame,
    decision: pd.DataFrame,
    zip_ok: bool,
) -> Dict[str, object]:
    deployable = chosen[chosen["deployable"].astype(bool)].copy()
    diagnostic = chosen[~chosen["deployable"].astype(bool)].copy()
    best_deploy = (
        float(pd.to_numeric(deployable["test_bad_top10_selected_rmse"], errors="coerce").min())
        if not deployable.empty
        else math.nan
    )
    best_diag = (
        float(pd.to_numeric(diagnostic["test_bad_top10_selected_rmse"], errors="coerce").min())
        if not diagnostic.empty
        else math.nan
    )
    best_vehicle_rel = best_test_bad(search, "reliability_vehicle_", require_override=True)
    best_bio_rel = min(
        [
            x
            for x in [
                best_test_bad(search, "reliability_vehicle_bio_pair", require_override=True),
                best_test_bad(search, "reliability_vehicle_bio_state", require_override=True),
                best_test_bad(search, "reliability_vehicle_style_bio_state", require_override=True),
            ]
            if np.isfinite(x)
        ],
        default=math.nan,
    )
    return {
        "pass": bool(zip_ok and V276.V265_GUARDRAIL.exists() and V276.V267_GUARDRAIL.exists()),
        "zip_testzip": bool(zip_ok),
        "v267_guardrail_pass": bool(V276.V267_GUARDRAIL.exists()),
        "v265_guardrail_pass": bool(V276.V265_GUARDRAIL.exists()),
        "event_n": int(top["event_uid"].nunique()),
        "candidate_rows": int(len(enriched)),
        "feature_set_n": int(len(feature_audit)),
        "search_rows": int(len(search)),
        "chosen_rows": int(len(chosen)),
        "fixed_wait_latest_badtop10": FIXED_WAIT_LATEST_BADTOP10,
        "best_val_chosen_deployable_test_badtop10": best_deploy,
        "best_test_diagnostic_badtop10": best_diag,
        "best_v278_rank_score_only_badtop10": best_test_bad(search, "v278_vehicle_rank_score_only", require_override=False),
        "best_vehicle_reliability_badtop10": best_vehicle_rel,
        "best_bio_reliability_badtop10": best_bio_rel,
        "best_deployable_passes_fixed_latest": bool(np.isfinite(best_deploy) and best_deploy < FIXED_WAIT_LATEST_BADTOP10),
        "best_diagnostic_passes_fixed_latest": bool(np.isfinite(best_diag) and best_diag < FIXED_WAIT_LATEST_BADTOP10),
        "bio_beats_vehicle_reliability": bool(
            np.isfinite(best_bio_rel) and np.isfinite(best_vehicle_rel) and best_bio_rel < best_vehicle_rel
        ),
        "decision_sources": "|".join(decision["source"].astype(str).tolist()),
    }


def markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def plot_decision(decision: pd.DataFrame) -> Path:
    path = FIGURES / "v279_test_badtop10_physio_reliability_filter.png"
    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    x = np.arange(len(decision))
    colors = ["#4C78A8" if bool(v) else "#9C755F" for v in decision["deployable"]]
    ax.bar(x, pd.to_numeric(decision["rmse"], errors="coerce"), color=colors)
    ax.axhline(FIXED_WAIT_LATEST_BADTOP10, color="#E15759", linestyle="--", linewidth=1.2, label="fixed wait-latest")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in decision["source"]], fontsize=8)
    ax.set_ylabel("test bad_top10 tail RMSE")
    ax.set_title("v279: physiology reliability filter for v278 listwise candidate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    decision: pd.DataFrame,
    chosen: pd.DataFrame,
    search: pd.DataFrame,
    feature_audit: pd.DataFrame,
    top: pd.DataFrame,
    guardrail: Dict[str, object],
    figs: Iterable[Path],
) -> None:
    top_val = search.sort_values(["selection_score", "val_bad_top10_selected_rmse"]).head(18)
    top_test = search[search["test_bad_top10_override_n"] > 0].sort_values("test_bad_top10_selected_rmse").head(18)
    top_gain_stats = (
        top.groupby("split", as_index=False)
        .agg(
            event_n=("event_uid", "nunique"),
            actual_gain_mean=("actual_gain_vs_latest", "mean"),
            actual_gain_median=("actual_gain_vs_latest", "median"),
            actual_good_rate=("actual_override_good", "mean"),
            bad_top10_n=("bad_top10", "sum"),
        )
        .copy()
    )

    report = f"""# v279 生理可靠性过滤实验

## 目的

v278 已经说明：如果只看 test diagnostic，vehicle-only listwise 候选在 bad_top10 上有一点空间，但验证集阈值选不出可部署提升；加入生理直接排序也没有赢过 vehicle-only。

v279 换一个任务构造：生理不直接选轨迹，而是判断 v278 车辆候选是否可信。也就是先让车辆模型给出“一个看起来值得替换 latest 的候选”，再让可靠性模型决定是否真的覆盖 latest。

## 方法

- 候选来源：复现 v278 的 `listrank_vehicle`，每个事件只保留第一候选。
- 监督目标：`actual_gain_vs_latest = latest_tail_rmse_v241 - candidate_tail_rmse_v241`。
- 二级模型：HGB 回归器预测收益，HGB 分类器预测替换是否为正收益。
- 得分形式：纯收益分数、风险校正收益分数、正收益概率分数。
- 阈值选择：仍只用 validation，test 只报告。
- 对照组：保留 `v278_vehicle_rank_score_only`，确认这一版口径能复现 v278 的车辆排序筛选。

## 核心结论

- fixed wait-latest test bad_top10: `{FIXED_WAIT_LATEST_BADTOP10:.6f}`
- val 选择的最好可部署 test bad_top10: `{guardrail["best_val_chosen_deployable_test_badtop10"]:.6f}`
- test diagnostic 最好 bad_top10: `{guardrail["best_test_diagnostic_badtop10"]:.6f}`
- 车辆可靠性最好 diagnostic: `{guardrail["best_vehicle_reliability_badtop10"]:.6f}`
- 生理可靠性最好 diagnostic: `{guardrail["best_bio_reliability_badtop10"]:.6f}`
- 生理是否赢过车辆可靠性: `{guardrail["bio_beats_vehicle_reliability"]}`
- 可部署规则是否超过 fixed latest: `{guardrail["best_deployable_passes_fixed_latest"]}`
- diagnostic 是否超过 fixed latest: `{guardrail["best_diagnostic_passes_fixed_latest"]}`

## v278 第一候选真实收益分布

{markdown_table(top_gain_stats)}

## 决策汇总

{markdown_table(decision)}

## 可靠性特征组

{markdown_table(feature_audit[["feature_set", "model_kind", "feature_n", "train_rows", "val_rows", "val_gain_mae", "val_gain_corr", "val_good_rate_actual", "val_good_prob_mean"]])}

## val 口径排名前 18

{markdown_table(top_val[["feature_set", "threshold", "val_bad_top10_selected_rmse", "val_bad_top10_delta_vs_latest", "val_all_delta_vs_latest", "val_bad_top10_override_rate", "test_bad_top10_selected_rmse", "test_bad_top10_override_rate", "selection_score"]])}

## test diagnostic 排名前 18

{markdown_table(top_test[["feature_set", "threshold", "val_bad_top10_selected_rmse", "val_bad_top10_delta_vs_latest", "val_all_delta_vs_latest", "test_bad_top10_selected_rmse", "test_bad_top10_delta_vs_latest", "test_bad_top10_override_rate"]])}

## 产物

{chr(10).join(f"- `{p.relative_to(OUT)}`" for p in figs)}
- `tables/v279_vehicle_listrank_top_candidate_rich.csv`
- `tables/v279_reliability_feature_set_audit.csv`
- `tables/v279_reliability_predictions.csv`
- `tables/v279_score_top_candidates.csv`
- `tables/v279_threshold_search.csv`
- `tables/v279_selected_by_strategy.csv`
- `tables/v279_chosen_configs.csv`
- `tables/v279_decision_summary.csv`
- `logs/guardrail_check.json`
"""
    (REPORTS / "v279_physio_reliability_filter_cn.md").write_text(report, encoding="utf-8")


def write_input_hashes() -> None:
    rows = []
    for path in [
        V279_SCRIPT,
        V278_SCRIPT,
        V277.V277_SCRIPT,
        V276.V267_PAIR,
        V276.V267_SELECTED,
        V276.V265_EVENT_SCORES,
    ]:
        if path.exists():
            rows.append({"path": str(path), "sha256": file_sha256(path), "size": int(path.stat().st_size)})
    write_csv(pd.DataFrame(rows), LOGS / "input_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(OUT)), "size": int(path.stat().st_size), "sha256": file_sha256(path)})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT.parent))
        zf.write(V279_SCRIPT, Path("scripts") / V279_SCRIPT.name)
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def main() -> None:
    print("[v279] physiology reliability filter for v278 listwise candidate", flush=True)
    print("[v279] 目的：检验生理状态能否判断车辆候选是否可信，而不是直接拼接预测轨迹。", flush=True)
    clean_out_dir()

    enriched, event_table, upstream_feature_audit, _ = V277.load_enriched_inputs()
    top, listrank_audit = build_vehicle_listrank_top(enriched)
    feature_audit, reliability_predictions, score_top, search, selected, chosen = build_reliability_outputs(top)
    decision = decision_summary(event_table, chosen)

    fig = plot_decision(decision)
    guardrail = build_guardrail(enriched, top, feature_audit, search, chosen, decision, zip_ok=False)

    write_csv(upstream_feature_audit, TABLES / "v279_upstream_event_feature_audit_from_v277.csv")
    write_csv(listrank_audit, TABLES / "v279_reproduced_v278_listrank_audit.csv")
    write_csv(top, TABLES / "v279_vehicle_listrank_top_candidate_rich.csv")
    write_csv(feature_audit, TABLES / "v279_reliability_feature_set_audit.csv")
    write_csv(reliability_predictions, TABLES / "v279_reliability_predictions.csv")
    write_csv(score_top, TABLES / "v279_score_top_candidates.csv")
    write_csv(search, TABLES / "v279_threshold_search.csv")
    write_csv(selected, TABLES / "v279_selected_by_strategy.csv")
    write_csv(chosen, TABLES / "v279_chosen_configs.csv")
    write_csv(decision, TABLES / "v279_decision_summary.csv")

    write_report(decision, chosen, search, feature_audit, top, guardrail, [fig])
    write_input_hashes()
    write_file_inventory()
    zip_ok = make_zip()
    guardrail = build_guardrail(enriched, top, feature_audit, search, chosen, decision, zip_ok=zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()

    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)
    print(f"[v279] report={REPORTS / 'v279_physio_reliability_filter_cn.md'}", flush=True)
    print(f"[v279] zip={ZIP_PATH}", flush=True)


if __name__ == "__main__":
    main()
