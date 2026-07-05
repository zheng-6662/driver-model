#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v273 bio-prefiltered pair reranker.

v272 发现：生理 top1 最近邻不可靠，但 bio top3/top5 oracle 偶尔低于 fixed wait-latest。
因此本轮把生理改成“候选预筛”而不是最终排序：

    车辆 top40 prototype -> 按 v271 calibrated physiology 距离取 bio top5 -> 监督式 pair reranker 只在 top5 内选择

目标是检验：v272 暴露的 bio top5 上界，能否被一个更窄的小候选 selector 转成可部署收益。

边界：
- prototype 只来自 train split；
- 生理只使用 v271 observation 前 calibrated raw physiology；
- raw_set / strategy 由 val bad_top10 选择，test 只报告；
- candidate oracle 仍只是上界，不是可部署结果。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

OUT = BASELINES / "v273_bio_prefiltered_pair_reranker_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v273_bio_prefiltered_pair_reranker_20260702_pack.zip"

V267_SCRIPT = BASELINES / "scripts" / "stage03_v267_supervised_bio_prototype_reranker_20260702.py"
V272_SCRIPT = BASELINES / "scripts" / "stage03_v272_physio_ambiguity_disambiguation_20260702.py"
V272_GUARDRAIL = BASELINES / "v272_physio_ambiguity_disambiguation_20260702" / "logs" / "guardrail_check.json"

SEED = 27302
VEHICLE_POOL_K = 40
BIO_PREFILTER_K = 5
FIXED_WAIT_LATEST_BADTOP10 = 0.695048


def import_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少前序脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


V267 = import_module_from_path("stage03_v267_for_v273", V267_SCRIPT)
V272 = import_module_from_path("stage03_v272_for_v273", V272_SCRIPT)
V266 = V272.V266


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


def build_bio_prefilter_neighbors(
    events: pd.DataFrame,
    feature_cols: List[str],
    neighbor_idx_by_query: Dict[int, np.ndarray],
    train_mask: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """先取车辆 top40，再在其中按生理距离取 top5，并把 rank 重新定义成 bio rank。"""
    bio_z, _med, _mean, _std = V266.fit_fill_scale(events[feature_cols].to_numpy(dtype=float), train_mask)
    event_uid = events["event_uid"].astype(str).to_numpy()
    split = events["split"].astype(str).to_numpy()
    subject = events["subject"].astype(str).to_numpy()
    oracle_delay = pd.to_numeric(events["oracle_delay_ms"], errors="coerce").to_numpy(dtype=int)
    rows: List[Dict[str, object]] = []
    for qi, veh_neighbors in neighbor_idx_by_query.items():
        pool = veh_neighbors[:VEHICLE_POOL_K]
        if len(pool) == 0:
            continue
        bio_distance = V272.pairwise_mean_sq_distance(bio_z[qi], bio_z[pool])
        bio_order = np.argsort(bio_distance, kind="mergesort")[: min(BIO_PREFILTER_K, len(pool))]
        for rank, local_pos in enumerate(bio_order, start=1):
            ti = int(pool[int(local_pos)])
            rows.append(
                {
                    "event_uid": event_uid[qi],
                    "split": split[qi],
                    "subject": subject[qi],
                    "neighbor_rank_vehicle": int(rank),
                    "prototype_event_uid": event_uid[ti],
                    "prototype_subject": subject[ti],
                    "prototype_oracle_delay_ms": int(oracle_delay[ti]),
                    "vehicle_distance": float(int(local_pos) + 1) / float(VEHICLE_POOL_K),
                    "bio_distance": float(bio_distance[int(local_pos)]),
                    "same_subject_as_prototype": bool(subject[qi] == subject[ti]),
                    "vehicle_pool_rank_original": int(local_pos) + 1,
                }
            )
    return pd.DataFrame(rows), bio_z


def run_one_raw_set(
    raw_set: str,
    feature_cols: List[str],
    events: pd.DataFrame,
    lookup: Dict[str, Dict[int, float]],
    veh_z: np.ndarray,
    neighbor_idx_by_query: Dict[int, np.ndarray],
    train_mask: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    neighbors, bio_z = build_bio_prefilter_neighbors(events, feature_cols, neighbor_idx_by_query, train_mask)
    pair_meta, matrices, names = V267.build_pair_dataset(events, neighbors, lookup, veh_z, bio_z, max_k=BIO_PREFILTER_K)
    pair_pred, fill_audit, feature_block = V267.add_pair_predictions(pair_meta, matrices, names)
    selected = V267.build_selected(events, pair_pred, lookup)
    summary = V267.summarize_selected(selected)
    chosen = V267.choose_val_strategies(summary)
    for df in (neighbors, pair_pred, selected, summary, chosen, fill_audit, feature_block):
        df["raw_set"] = raw_set
        df["raw_feature_n"] = int(len(feature_cols))
        df["vehicle_pool_k"] = int(VEHICLE_POOL_K)
        df["bio_prefilter_k"] = int(BIO_PREFILTER_K)
    return neighbors, pair_pred, selected, summary, chosen, feature_block


def choose_cross_set(chosen: pd.DataFrame, chosen_label: str) -> pd.DataFrame:
    val = chosen[
        chosen["chosen_label"].eq(chosen_label)
        & chosen["split"].eq("val")
        & chosen["event_group"].eq("bad_top10")
    ].copy()
    if val.empty:
        return pd.DataFrame()
    val = val.sort_values(["selected_tail_rmse_mean", "selected_delay_ms_mean", "raw_set"], ascending=[True, True, True])
    best = val.iloc[0]
    raw_set = str(best["raw_set"])
    strategy = str(best["chosen_strategy"])
    mapped = chosen[
        chosen["chosen_label"].eq(chosen_label)
        & chosen["raw_set"].astype(str).eq(raw_set)
        & chosen["chosen_strategy"].astype(str).eq(strategy)
        & chosen["event_group"].eq("bad_top10")
        & chosen["split"].isin(["val", "test"])
    ].copy()
    return mapped


def build_decision(summary: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    test_all = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
    ].copy()
    test_bad = test_all[test_all["k"].isin([5, 10, 20, 40])].copy()
    rows: List[Dict[str, object]] = []
    if test_all.empty:
        return pd.DataFrame()

    for strategy, source, deployable in [
        ("policy_keep_0ms_anchor", "baseline", True),
        ("policy_wait_to_latest_anchor", "baseline", True),
        ("oracle_best_anchor_upper_bound", "oracle", False),
    ]:
        sub = test_all[test_all["strategy"].eq(strategy)]
        if len(sub):
            rows.append({"source": source, "label": strategy, "rmse": float(sub.iloc[0]["selected_tail_rmse_mean"]), "deployable": deployable})
    cand = test_bad[test_bad["strategy_family"].eq("candidate_oracle")].sort_values("selected_tail_rmse_mean")
    if len(cand):
        row = cand.iloc[0]
        rows.append({"source": "bio_prefilter_candidate_oracle", "label": f"{row['raw_set']}:{row['strategy']}", "rmse": float(row["selected_tail_rmse_mean"]), "deployable": False})
    deploy = test_bad[
        test_bad["deployable"].astype(bool)
        & ~test_bad["strategy_family"].isin(["baseline", "oracle", "candidate_oracle"])
    ].sort_values("selected_tail_rmse_mean")
    if len(deploy):
        row = deploy.iloc[0]
        rows.append({"source": "test_best_deployable_diagnostic", "label": f"{row['raw_set']}:{row['strategy']}", "rmse": float(row["selected_tail_rmse_mean"]), "deployable": False})

    for label, source in [
        ("val_best_pair_vehicle_bio", "val_best_vehicle_bio"),
        ("val_best_pair_any", "val_best_any"),
    ]:
        mapped = choose_cross_set(chosen, label)
        test = mapped[mapped["split"].eq("test")]
        if len(test):
            row = test.iloc[0]
            rows.append({"source": source, "label": f"{row['raw_set']}:{row['chosen_strategy']}", "rmse": float(row["selected_tail_rmse_mean"]), "deployable": True})

    out = pd.DataFrame(rows)
    out["delta_vs_fixed_latest"] = out["rmse"] - FIXED_WAIT_LATEST_BADTOP10
    out["passes_fixed_latest"] = out["rmse"] < FIXED_WAIT_LATEST_BADTOP10
    return out


def plot_decision(decision: pd.DataFrame) -> Path:
    path = FIGURES / "v273_test_badtop10_bio_prefiltered_pair.png"
    if decision.empty:
        return path
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    x = np.arange(len(decision))
    colors = ["#4C78A8" if bool(v) else "#9C755F" for v in decision["deployable"]]
    ax.bar(x, decision["rmse"], color=colors)
    ax.axhline(FIXED_WAIT_LATEST_BADTOP10, color="#E15759", linestyle="--", linewidth=1.2, label="fixed wait-latest")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in decision["source"]], fontsize=8)
    ax.set_ylabel("test bad_top10 tail RMSE")
    ax.set_title("v273: supervised selector after bio top5 prefilter")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v267_script", V267_SCRIPT),
        ("v272_script", V272_SCRIPT),
        ("v272_guardrail", V272_GUARDRAIL),
        ("v271_events", V272.V271_EVENTS),
        ("v271_feature_audit", V272.V271_FEATURE_AUDIT),
    ]:
        rows.append({"label": label, "path": str(path), "exists": path.exists(), "sha256": file_sha256(path) if path.exists() else ""})
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    if OUT.exists():
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


def write_report(summary: pd.DataFrame, chosen: pd.DataFrame, decision: pd.DataFrame, figs: List[Path]) -> None:
    lines: List[str] = []
    lines.append("# v273 bio-prefiltered pair reranker")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- v272 显示 bio top1 不可靠，但 bio top3/top5 oracle 有少量上界。")
    lines.append("- v273 先用车辆 top40 建候选池，再按生理距离取 top5，最后只在 top5 内训练监督式 pair reranker。")
    lines.append("- 这检验“生理作为候选预筛”是否能被模型转成可部署收益。")
    lines.append("")
    lines.append("## test bad_top10 决策收口")
    lines.append("")
    lines.append(decision.to_markdown(index=False) if len(decision) else "- 无可用结果。")
    lines.append("")
    lines.append("## test bad_top10 top")
    lines.append("")
    test_bad = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")].sort_values("selected_tail_rmse_mean").head(24)
    cols = ["raw_set", "strategy", "strategy_family", "deployable", "k", "selected_tail_rmse_mean", "delta_selected_minus_latest_mean", "selected_delay_ms_mean", "selected_latest_rate"]
    lines.append(test_bad[[c for c in cols if c in test_bad.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## val 选择映射")
    lines.append("")
    show_chosen = chosen[chosen["event_group"].eq("bad_top10")]
    lines.append(show_chosen.head(36).to_markdown(index=False) if len(show_chosen) else "- 无 val 选择结果。")
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    deploy = decision[decision["deployable"].astype(bool)] if len(decision) else pd.DataFrame()
    if len(deploy) and bool(deploy["passes_fixed_latest"].any()):
        lines.append("- 至少一个 val 选择的可部署策略低于 fixed wait-latest，说明 bio 预筛 + selector 可能接近 goal。")
    else:
        lines.append("- val 选择的可部署策略仍未低于 fixed wait-latest，bio top5 上界没有稳定转成模型收益。")
    if len(decision):
        best = decision.sort_values("rmse").iloc[0]
        lines.append(f"- 当前最小 RMSE 条目为 `{best['label']}`，test bad_top10 `{float(best['rmse']):.4f}`，deployable={bool(best['deployable'])}。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figs:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v273_bio_prefiltered_pair_reranker_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v273] bio-prefiltered pair reranker", flush=True)
    clean_out_dir()
    cand, events, veh_cols, feature_sets, _feature_audit = V272.load_data()
    lookup = V266.candidate_rmse_lookup(cand)
    neighbor_idx_by_query, veh_z, train_mask = V272.build_query_neighbor_base(events, veh_cols, VEHICLE_POOL_K)

    all_neighbors: List[pd.DataFrame] = []
    all_pairs: List[pd.DataFrame] = []
    all_selected: List[pd.DataFrame] = []
    all_summary: List[pd.DataFrame] = []
    all_chosen: List[pd.DataFrame] = []
    all_blocks: List[pd.DataFrame] = []

    for raw_set, cols in feature_sets.items():
        print(f"[v273] raw_set={raw_set} feature_n={len(cols)}", flush=True)
        neighbors, pair_pred, selected, summary, chosen, blocks = run_one_raw_set(
            raw_set, cols, events, lookup, veh_z, neighbor_idx_by_query, train_mask
        )
        all_neighbors.append(neighbors)
        all_pairs.append(pair_pred)
        all_selected.append(selected)
        all_summary.append(summary)
        all_chosen.append(chosen)
        all_blocks.append(blocks)

    neighbors_all = pd.concat(all_neighbors, ignore_index=True)
    pair_all = pd.concat(all_pairs, ignore_index=True)
    selected_all = pd.concat(all_selected, ignore_index=True)
    summary_all = pd.concat(all_summary, ignore_index=True)
    chosen_all = pd.concat(all_chosen, ignore_index=True)
    blocks_all = pd.concat(all_blocks, ignore_index=True)
    decision = build_decision(summary_all, chosen_all)
    fig = plot_decision(decision)

    pred_cols = [c for c in pair_all.columns if c.startswith("pred_pair_")]
    compact_cols = [
        "raw_set",
        "event_uid",
        "split",
        "subject",
        "prototype_event_uid",
        "prototype_subject",
        "neighbor_rank_vehicle",
        "prototype_oracle_delay_ms",
        "mapped_delay_ms",
        "target_tail_rmse_v241",
        "vehicle_distance",
        "bio_distance",
        "bad_top10",
    ] + pred_cols
    write_csv(neighbors_all, TABLES / "v273_bio_prefilter_neighbors.csv")
    write_csv(pair_all[[c for c in compact_cols if c in pair_all.columns]], TABLES / "v273_pair_predictions_compact.csv")
    write_csv(selected_all, TABLES / "v273_selected_by_strategy.csv")
    write_csv(summary_all, TABLES / "v273_pair_reranker_summary.csv")
    write_csv(chosen_all, TABLES / "v273_val_chosen_summary.csv")
    write_csv(blocks_all, TABLES / "v273_feature_block_audit.csv")
    write_csv(decision, TABLES / "v273_decision_summary.csv")
    write_input_hashes()
    write_file_inventory()
    write_report(summary_all, chosen_all, decision, [fig])
    write_file_inventory()
    zip_ok = make_zip()

    v272_guard = json.loads(V272_GUARDRAIL.read_text(encoding="utf-8")) if V272_GUARDRAIL.exists() else {}
    learned_deploy = decision[
        decision["deployable"].astype(bool)
        & ~decision["source"].isin(["baseline", "oracle"])
    ] if len(decision) else pd.DataFrame()
    best_deploy = float(learned_deploy["rmse"].min()) if len(learned_deploy) else math.nan
    best_any = float(decision["rmse"].min()) if len(decision) else math.nan
    guardrail = {
        "pass": bool(zip_ok and bool(v272_guard.get("pass", False)) and len(pair_all) > 0 and len(summary_all) > 0),
        "zip_testzip": bool(zip_ok),
        "v272_guardrail_pass": bool(v272_guard.get("pass", False)),
        "event_n": int(events["event_uid"].nunique()),
        "raw_set_n": int(len(feature_sets)),
        "vehicle_pool_k": int(VEHICLE_POOL_K),
        "bio_prefilter_k": int(BIO_PREFILTER_K),
        "pair_row_n": int(len(pair_all)),
        "best_deployable_test_badtop10": best_deploy,
        "best_any_test_badtop10": best_any,
        "fixed_wait_latest_badtop10": float(FIXED_WAIT_LATEST_BADTOP10),
        "best_deployable_passes_fixed_latest": bool(np.isfinite(best_deploy) and best_deploy < FIXED_WAIT_LATEST_BADTOP10),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v273 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    print(f"[v273] report={REPORTS / 'v273_bio_prefiltered_pair_reranker_cn.md'}", flush=True)
    print(f"[v273] zip={ZIP_PATH}", flush=True)
    if len(decision):
        print(decision.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
