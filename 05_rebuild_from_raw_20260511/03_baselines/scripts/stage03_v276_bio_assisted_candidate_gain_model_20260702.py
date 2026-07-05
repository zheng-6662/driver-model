#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v276 bio-assisted candidate gain model.

v274/v275 说明：用生理信号直接做 sparse override 或 consensus override，test 上
有少量 diagnostic headroom，但 val 无法选出可部署规则。v276 换一个任务构造：

1. 不再让生理直接决定锚点；
2. 回到 v267 的 full vehicle top40 prototype 候选池；
3. 对每个 query-candidate pair 训练“相对 latest 的候选收益”预测器；
4. 比较 candidate_vehicle、candidate_vehicle_bio、candidate_bio_only 三组特征；
5. 每个事件只允许选择预测收益最高的候选，是否覆盖 latest 的阈值只在 val 上选；
6. test 只报告，不能用 test 选阈值。

这一步检验的是：生理是否能作为车辆多未来候选选择中的辅助校准信号，而不是
作为单独 selector / reranker / override 规则。
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
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

OUT = BASELINES / "v276_bio_assisted_candidate_gain_model_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v276_bio_assisted_candidate_gain_model_20260702_pack.zip"

V267_PAIR = BASELINES / "v267_supervised_bio_prototype_reranker_20260702" / "tables" / "v267_pair_predictions_compact.csv"
V267_SELECTED = BASELINES / "v267_supervised_bio_prototype_reranker_20260702" / "tables" / "v267_selected_pair_reranker_by_strategy.csv"
V267_GUARDRAIL = BASELINES / "v267_supervised_bio_prototype_reranker_20260702" / "logs" / "guardrail_check.json"
V265_EVENT_SCORES = BASELINES / "v265_physio_uncertainty_wait_frontier_20260702" / "tables" / "v265_event_risk_scores.csv"
V265_GUARDRAIL = BASELINES / "v265_physio_uncertainty_wait_frontier_20260702" / "logs" / "guardrail_check.json"

FIXED_WAIT_LATEST_BADTOP10 = 0.695048
SEED = 27602

EVENT_GROUPS = [
    ("all", None),
    ("bad_top10", "bad_top10"),
    ("very_bad_top5", "very_bad_top5"),
    ("normal", "normal"),
    ("observe_later_like", "observe_later_like"),
    ("strong_steer", "strong_steer"),
    ("early_best_after_400", "early_best_after_400"),
]

V265_SCORE_COLS = [
    "score_vehicle_gain",
    "score_vehicle_bio_gain",
    "score_bio_only_gain",
    "score_vehicle_keep0_risk",
    "score_vehicle_bio_keep0_risk",
    "score_bio_only_keep0_risk",
    "score_vehicle_badprob",
    "score_vehicle_bio_badprob",
    "score_bio_only_badprob",
    "score_vehicle_oracle_gap",
    "score_vehicle_bio_oracle_gap",
    "pred_gain_vehicle",
    "pred_gain_vehicle_bio260_sp64",
]


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


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if not V267_PAIR.exists():
        raise FileNotFoundError(f"缺少 v267 pair 表：{V267_PAIR}")
    if not V267_SELECTED.exists():
        raise FileNotFoundError(f"缺少 v267 selected 表：{V267_SELECTED}")
    if not V265_EVENT_SCORES.exists():
        raise FileNotFoundError(f"缺少 v265 event score 表：{V265_EVENT_SCORES}")

    pair = pd.read_csv(V267_PAIR, encoding="utf-8-sig", low_memory=False)
    selected = pd.read_csv(V267_SELECTED, encoding="utf-8-sig", low_memory=False)
    events = selected[selected["strategy"].astype(str).eq("policy_wait_to_latest_anchor")].copy()
    event_cols = [
        "event_uid",
        "split",
        "subject",
        "recording",
        "keep0_tail_rmse_v241",
        "latest_tail_rmse_v241",
        "oracle_tail_rmse_v241",
        "bad_top10",
        "very_bad_top5",
        "normal",
        "observe_later_like",
        "strong_steer",
        "reverse",
        "early_best_after_400",
    ]
    events = events[event_cols].drop_duplicates("event_uid")

    v265 = pd.read_csv(
        V265_EVENT_SCORES,
        encoding="utf-8-sig",
        low_memory=False,
        usecols=lambda c: c in ["event_uid"] + V265_SCORE_COLS,
    ).drop_duplicates("event_uid")
    # v267 候选表本身已经有 split/subject；事件表只负责补充 recording、标签和基线误差，
    # 避免 pandas 自动生成 split_x/split_y 后让后续严格 train/val/test 筛选失效。
    events_for_pair = events.drop(columns=["split", "subject"], errors="ignore")
    df = pair.drop(columns=["bad_top10"], errors="ignore").merge(
        events_for_pair,
        on="event_uid",
        how="left",
        validate="many_to_one",
    )
    df = df.merge(v265, on="event_uid", how="left", validate="many_to_one")

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    numeric = [
        "neighbor_rank_vehicle",
        "prototype_oracle_delay_ms",
        "mapped_delay_ms",
        "target_tail_rmse_v241",
        "vehicle_distance",
        "bio_distance",
        "keep0_tail_rmse_v241",
        "latest_tail_rmse_v241",
        "oracle_tail_rmse_v241",
    ] + [c for c in df.columns if c.startswith("pred_pair_")] + V265_SCORE_COLS
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target_gain_vs_latest"] = df["latest_tail_rmse_v241"] - df["target_tail_rmse_v241"]
    df["candidate_better_than_latest"] = (df["target_gain_vs_latest"] > 0.005).astype(int)

    event_table = events.merge(v265, on="event_uid", how="left")
    guard = {
        "v267_guardrail_pass": json.loads(V267_GUARDRAIL.read_text(encoding="utf-8")).get("pass", False)
        if V267_GUARDRAIL.exists()
        else False,
        "v265_guardrail_pass": json.loads(V265_GUARDRAIL.read_text(encoding="utf-8")).get("pass", False)
        if V265_GUARDRAIL.exists()
        else False,
    }
    return df, event_table, guard


def feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    vehicle_pair_cols = [
        "mapped_delay_ms",
        "neighbor_rank_vehicle",
        "vehicle_distance",
        "pred_pair_base_hgb",
        "pred_pair_vehicle_hgb",
    ]
    vehicle_score_cols = [c for c in V265_SCORE_COLS if c.startswith("score_vehicle_") and "_bio_" not in c] + ["pred_gain_vehicle"]
    bio_pair_cols = [
        "bio_distance",
        "pred_pair_bio_hgb",
        "pred_pair_vehicle_bio_hgb",
        "pred_pair_vehicle_bio_badweighted_hgb",
    ]
    bio_score_cols = [c for c in V265_SCORE_COLS if "bio" in c] + ["pred_gain_vehicle_bio260_sp64"]
    out = {
        "candidate_vehicle": vehicle_pair_cols + vehicle_score_cols,
        "candidate_vehicle_bio": vehicle_pair_cols + vehicle_score_cols + bio_pair_cols + bio_score_cols,
        "candidate_bio_only": ["mapped_delay_ms", "bio_distance", "pred_pair_bio_hgb"] + bio_score_cols,
    }
    return {name: [c for c in cols if c in df.columns] for name, cols in out.items()}


def fit_predict_gain(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    train_mask = df["split"].astype(str).eq("train").to_numpy()
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any(axis=0)].copy()
    model = HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=0.5,
        random_state=SEED,
    )
    model.fit(X.loc[train_mask], df.loc[train_mask, "target_gain_vs_latest"])
    return model.predict(X)


def top_candidate_per_event(df: pd.DataFrame, feature_set_name: str, pred_gain: np.ndarray) -> pd.DataFrame:
    cols = [
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
    top = df[cols].copy()
    top["feature_set"] = feature_set_name
    top["pred_gain_vs_latest"] = pred_gain
    top = top.sort_values(
        ["event_uid", "pred_gain_vs_latest", "target_tail_rmse_v241", "neighbor_rank_vehicle"],
        ascending=[True, False, True, True],
    ).drop_duplicates("event_uid")
    return top


def subset_metrics(selected: pd.DataFrame, split: str, flag: str | None) -> Dict[str, float]:
    mask = selected["split"].astype(str).eq(split)
    if flag is not None:
        mask &= selected[flag].astype(bool)
    sub = selected[mask].copy()
    if sub.empty:
        return {
            "n": 0,
            "selected_rmse": math.nan,
            "latest_rmse": math.nan,
            "oracle_rmse": math.nan,
            "delta_vs_latest": math.nan,
            "override_n": 0,
            "override_rate": math.nan,
        }
    selected_rmse = float(pd.to_numeric(sub["selected_tail_rmse_v241"], errors="coerce").mean())
    latest_rmse = float(pd.to_numeric(sub["latest_tail_rmse_v241"], errors="coerce").mean())
    return {
        "n": int(len(sub)),
        "selected_rmse": selected_rmse,
        "latest_rmse": latest_rmse,
        "oracle_rmse": float(pd.to_numeric(sub["oracle_tail_rmse_v241"], errors="coerce").mean()),
        "delta_vs_latest": selected_rmse - latest_rmse,
        "override_n": int(sub["override_latest"].astype(bool).sum()),
        "override_rate": float(sub["override_latest"].astype(bool).mean()),
    }


def threshold_search(top: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    selected_rows: List[pd.DataFrame] = []
    for feature_set_name, sub in top.groupby("feature_set", sort=False):
        val_top = sub[sub["split"].astype(str).eq("val")].copy()
        thresholds = sorted(set([float("inf")] + [float(x) for x in np.nanquantile(val_top["pred_gain_vs_latest"], np.linspace(0.50, 1.00, 31)) if np.isfinite(x)]))
        for threshold in thresholds:
            selected = sub.copy()
            selected["strategy"] = f"{feature_set_name}_gain_thr{threshold:.6g}"
            selected["threshold"] = threshold
            selected["override_latest"] = pd.to_numeric(selected["pred_gain_vs_latest"], errors="coerce") >= threshold
            selected["selected_delay_ms"] = np.where(selected["override_latest"], selected["mapped_delay_ms"], 1000)
            selected["selected_tail_rmse_v241"] = np.where(
                selected["override_latest"], selected["target_tail_rmse_v241"], selected["latest_tail_rmse_v241"]
            )
            selected_rows.append(selected)

            row: Dict[str, object] = {"feature_set": feature_set_name, "threshold": threshold}
            for split in ["val", "test"]:
                for group, flag in EVENT_GROUPS:
                    metrics = subset_metrics(selected, split, flag)
                    prefix = f"{split}_{group}"
                    for key, value in metrics.items():
                        row[f"{prefix}_{key}"] = value
            row["active"] = int(row.get("val_bad_top10_override_n", 0)) > 0
            row["stable_pass"] = bool(
                row["active"]
                and float(row.get("val_bad_top10_delta_vs_latest", math.inf)) <= 0.0
                and float(row.get("val_all_delta_vs_latest", math.inf)) <= 0.003
                and float(row.get("val_normal_delta_vs_latest", math.inf)) <= 0.005
                and float(row.get("val_strong_steer_delta_vs_latest", math.inf)) <= 0.005
                and float(row.get("val_observe_later_like_delta_vs_latest", math.inf)) <= 0.005
            )
            row["stability_penalty"] = float(
                sum(
                    max(0.0, float(row.get(name, 0.0)))
                    for name in [
                        "val_all_delta_vs_latest",
                        "val_normal_delta_vs_latest",
                        "val_strong_steer_delta_vs_latest",
                        "val_observe_later_like_delta_vs_latest",
                    ]
                )
            )
            row["selection_score"] = float(row.get("val_bad_top10_delta_vs_latest", 0.0)) + 3.0 * float(row["stability_penalty"])
            rows.append(row)
    return pd.DataFrame(rows), pd.concat(selected_rows, ignore_index=True)


def choose_configs(search: pd.DataFrame) -> pd.DataFrame:
    chosen_rows: List[Dict[str, object]] = []

    def add_choice(label: str, sub: pd.DataFrame, deployable: bool) -> None:
        if sub.empty:
            return
        best = sub.sort_values(
            [
                "selection_score",
                "val_bad_top10_selected_rmse",
                "val_bad_top10_override_rate",
                "feature_set",
            ],
            ascending=[True, True, True, True],
        ).iloc[0].to_dict()
        best["chosen_type"] = label
        best["deployable"] = bool(deployable)
        chosen_rows.append(best)

    add_choice("best_any", search.copy(), True)
    add_choice("best_active", search[search["active"].astype(bool)].copy(), True)
    add_choice("best_stable_active", search[search["stable_pass"].astype(bool)].copy(), True)
    noharm = search[
        search["active"].astype(bool)
        & (pd.to_numeric(search["val_bad_top10_delta_vs_latest"], errors="coerce") <= 0.0)
        & (pd.to_numeric(search["val_all_delta_vs_latest"], errors="coerce") <= 0.0)
    ].copy()
    add_choice("best_noharm_all", noharm, True)

    diag = search[search["test_bad_top10_override_n"] > 0].copy()
    if not diag.empty:
        best = diag.sort_values(
            ["test_bad_top10_selected_rmse", "val_bad_top10_delta_vs_latest", "val_all_delta_vs_latest"],
            ascending=[True, True, True],
        ).iloc[0].to_dict()
        best["chosen_type"] = "test_best_diagnostic"
        best["deployable"] = False
        chosen_rows.append(best)
    return pd.DataFrame(chosen_rows)


def baseline_decision(event_table: pd.DataFrame) -> List[Dict[str, object]]:
    test_bad = event_table[event_table["bad_top10"].astype(bool)].copy()
    rows = []
    rows.append(
        {
            "source": "baseline",
            "label": "policy_wait_to_latest_anchor",
            "rmse": float(pd.to_numeric(test_bad["latest_tail_rmse_v241"], errors="coerce").mean()),
            "deployable": True,
            "override_rate": math.nan,
        }
    )
    rows.append(
        {
            "source": "oracle",
            "label": "oracle_best_anchor_upper_bound",
            "rmse": float(pd.to_numeric(test_bad["oracle_tail_rmse_v241"], errors="coerce").mean()),
            "deployable": False,
            "override_rate": math.nan,
        }
    )
    return rows


def build_decision(chosen: pd.DataFrame, event_table: pd.DataFrame) -> pd.DataFrame:
    rows = baseline_decision(event_table[event_table["split"].astype(str).eq("test")])
    for _, row in chosen.iterrows():
        rows.append(
            {
                "source": f"val_{row['chosen_type']}" if bool(row["deployable"]) else "test_best_gain_diagnostic",
                "label": f"{row['feature_set']} threshold={row['threshold']}",
                "rmse": float(row["test_bad_top10_selected_rmse"]),
                "deployable": bool(row["deployable"]),
                "override_rate": float(row["test_bad_top10_override_rate"]),
                "val_bad_delta": float(row["val_bad_top10_delta_vs_latest"]),
                "val_all_delta": float(row["val_all_delta_vs_latest"]),
                "stable_pass": bool(row.get("stable_pass", False)),
            }
        )
    decision = pd.DataFrame(rows)
    decision["delta_vs_fixed_latest"] = pd.to_numeric(decision["rmse"], errors="coerce") - FIXED_WAIT_LATEST_BADTOP10
    decision["passes_fixed_latest"] = pd.to_numeric(decision["rmse"], errors="coerce") < FIXED_WAIT_LATEST_BADTOP10
    return decision


def plot_decision(decision: pd.DataFrame) -> Path:
    path = FIGURES / "v276_test_badtop10_candidate_gain_model.png"
    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    x = np.arange(len(decision))
    colors = ["#4C78A8" if bool(v) else "#9C755F" for v in decision["deployable"]]
    ax.bar(x, pd.to_numeric(decision["rmse"], errors="coerce"), color=colors)
    ax.axhline(FIXED_WAIT_LATEST_BADTOP10, color="#E15759", linestyle="--", linewidth=1.2, label="fixed wait-latest")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in decision["source"]], fontsize=8)
    ax.set_ylabel("test bad_top10 tail RMSE")
    ax.set_title("v276: bio-assisted candidate gain model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(decision: pd.DataFrame, chosen: pd.DataFrame, search: pd.DataFrame, figs: Iterable[Path]) -> None:
    lines: List[str] = []
    lines.append("# v276 bio-assisted candidate gain model")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- 回到 v267 full vehicle top40 候选池，而不是只用 bio top5。")
    lines.append("- 对 query-candidate pair 训练相对 latest 的候选收益预测器。")
    lines.append("- 比较 candidate_vehicle、candidate_vehicle_bio、candidate_bio_only 三组特征，判断生理是否能辅助车辆多未来候选选择。")
    lines.append("- threshold 只由 val 选择，test 只报告。")
    lines.append("")
    lines.append("## test bad_top10 决策收口")
    lines.append("")
    lines.append(decision.to_markdown(index=False))
    lines.append("")
    lines.append("## val 选择出的配置")
    lines.append("")
    cols = [
        "chosen_type",
        "deployable",
        "feature_set",
        "threshold",
        "val_bad_top10_delta_vs_latest",
        "val_all_delta_vs_latest",
        "val_normal_delta_vs_latest",
        "test_bad_top10_selected_rmse",
        "test_bad_top10_delta_vs_latest",
        "test_bad_top10_override_rate",
        "stable_pass",
    ]
    show_cols = [c for c in cols if c in chosen.columns]
    lines.append(chosen[show_cols].to_markdown(index=False) if len(chosen) else "- 没有选出配置。")
    lines.append("")
    lines.append("## search top by val bad_top10")
    lines.append("")
    top = search[search["active"].astype(bool)].sort_values(["val_bad_top10_selected_rmse", "stability_penalty"]).head(24)
    top_cols = [
        "feature_set",
        "threshold",
        "val_bad_top10_selected_rmse",
        "val_bad_top10_delta_vs_latest",
        "val_all_delta_vs_latest",
        "val_normal_delta_vs_latest",
        "test_bad_top10_selected_rmse",
        "test_bad_top10_delta_vs_latest",
        "test_bad_top10_override_rate",
        "stable_pass",
    ]
    lines.append(top[[c for c in top_cols if c in top.columns]].to_markdown(index=False) if len(top) else "- 没有 active override。")
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    deploy = decision[decision["deployable"].astype(bool) & decision["source"].astype(str).str.startswith("val_")]
    if len(deploy) and bool(deploy["passes_fixed_latest"].any()):
        lines.append("- 至少一个 val 选择出的候选收益模型低于 fixed wait-latest，可继续复核。")
    else:
        lines.append("- val 选择出的候选收益模型仍未低于 fixed wait-latest。")
    lines.append("- 如果 test diagnostic 低于 fixed wait-latest 但 val 上伤害，说明模型事后能碰到少数样本，但没有稳定可部署规则。")
    lines.append("- 若 candidate_vehicle_bio 没有稳定优于 candidate_vehicle，则当前生理仍不能作为多未来候选选择的主增量。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figs:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v276_bio_assisted_candidate_gain_model_cn.md").write_text("\n".join(lines), encoding="utf-8")


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v267_pair", V267_PAIR),
        ("v267_selected", V267_SELECTED),
        ("v267_guardrail", V267_GUARDRAIL),
        ("v265_event_scores", V265_EVENT_SCORES),
        ("v265_guardrail", V265_GUARDRAIL),
    ]:
        rows.append({"label": label, "path": str(path), "exists": path.exists(), "sha256": file_sha256(path) if path.exists() else ""})
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip() is None


def main() -> None:
    print("[v276] bio-assisted candidate gain model", flush=True)
    clean_out_dir()
    df, event_table, guard = load_inputs()
    features = feature_sets(df)
    feature_audit = pd.DataFrame(
        [
            {
                "feature_set": name,
                "feature_n": len(cols),
                "vehicle_feature_n": sum(("bio" not in c and "sp64" not in c) for c in cols),
                "bio_feature_n": sum(("bio" in c or "sp64" in c) for c in cols),
                "features": "|".join(cols),
            }
            for name, cols in features.items()
        ]
    )

    top_parts = []
    prediction_parts = []
    for name, cols in features.items():
        pred_gain = fit_predict_gain(df, cols)
        part = df[["event_uid", "split", "mapped_delay_ms", "target_tail_rmse_v241", "target_gain_vs_latest"]].copy()
        part["feature_set"] = name
        part["pred_gain_vs_latest"] = pred_gain
        prediction_parts.append(part)
        top_parts.append(top_candidate_per_event(df, name, pred_gain))
    top = pd.concat(top_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    search, selected = threshold_search(top)
    chosen = choose_configs(search)
    decision = build_decision(chosen, event_table)
    fig = plot_decision(decision)

    write_csv(feature_audit, TABLES / "v276_feature_set_audit.csv")
    write_csv(predictions, TABLES / "v276_candidate_gain_predictions.csv")
    write_csv(top, TABLES / "v276_top_candidate_by_event.csv")
    write_csv(search, TABLES / "v276_threshold_search.csv")
    write_csv(chosen, TABLES / "v276_chosen_configs.csv")
    write_csv(selected, TABLES / "v276_selected_by_strategy.csv")
    write_csv(decision, TABLES / "v276_decision_summary.csv")
    write_input_hashes()
    write_file_inventory()
    write_report(decision, chosen, search, [fig])
    write_file_inventory()
    zip_ok = make_zip()

    deploy = decision[decision["deployable"].astype(bool) & decision["source"].astype(str).str.startswith("val_")]
    best_deploy = float(pd.to_numeric(deploy["rmse"], errors="coerce").min()) if len(deploy) else math.nan
    diag = decision[decision["source"].astype(str).eq("test_best_gain_diagnostic")]
    best_diag = float(pd.to_numeric(diag["rmse"], errors="coerce").min()) if len(diag) else math.nan
    guardrail = {
        "pass": bool(zip_ok and guard["v267_guardrail_pass"] and guard["v265_guardrail_pass"] and len(search) > 0 and len(decision) > 0),
        "zip_testzip": bool(zip_ok),
        "v267_guardrail_pass": bool(guard["v267_guardrail_pass"]),
        "v265_guardrail_pass": bool(guard["v265_guardrail_pass"]),
        "candidate_rows": int(len(df)),
        "event_n": int(event_table["event_uid"].nunique()),
        "feature_set_n": int(len(features)),
        "search_rows": int(len(search)),
        "chosen_rows": int(len(chosen)),
        "best_val_chosen_deployable_test_badtop10": best_deploy,
        "best_test_diagnostic_badtop10": best_diag,
        "fixed_wait_latest_badtop10": float(FIXED_WAIT_LATEST_BADTOP10),
        "best_deployable_passes_fixed_latest": bool(np.isfinite(best_deploy) and best_deploy < FIXED_WAIT_LATEST_BADTOP10),
        "best_diagnostic_passes_fixed_latest": bool(np.isfinite(best_diag) and best_diag < FIXED_WAIT_LATEST_BADTOP10),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v276 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    print(f"[v276] report={REPORTS / 'v276_bio_assisted_candidate_gain_model_cn.md'}", flush=True)
    print(f"[v276] zip={ZIP_PATH}", flush=True)
    print(decision.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
