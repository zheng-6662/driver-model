#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v281 bio-top3 constrained selector.

v272 显示：直接用生理最近邻 top1 不能解决问题，但 vehicle top40 内的 bio top3 oracle
在 test bad_top10 上有少量上限。v281 把这个上限变成一个可训练问题：

1. 先用 vehicle-only 找每个事件最相似的 40 个训练事件原型；
2. 在这 40 个原型中，用 v271 的生理状态距离取 bio 最近的 3 个候选；
3. 只在这 3 个候选里训练 selector，预测候选相对 latest 的收益；
4. 是否覆盖 latest 仍只用 validation 阈值决定，test 只报告。

如果这个窄化后的候选选择器仍然不能稳定超过 fixed wait-latest，就说明“生理做候选消歧”
这条线基本没有可部署增量，后续不应再在同类 selector 上消耗。
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


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
SCRIPTS = BASELINES / "scripts"

OUT = BASELINES / "v281_bio_top3_constrained_selector_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v281_bio_top3_constrained_selector_20260702_pack.zip"

V281_SCRIPT = SCRIPTS / "stage03_v281_bio_top3_constrained_selector_20260702.py"
V272_SCRIPT = SCRIPTS / "stage03_v272_physio_ambiguity_disambiguation_20260702.py"
V276_SCRIPT = SCRIPTS / "stage03_v276_bio_assisted_candidate_gain_model_20260702.py"

SEED = 28102
FIXED_WAIT_LATEST_BADTOP10 = 0.695048
VEHICLE_K = 40
BIO_TOP_M = 3
SELECTED_RAW_SETS = ["subject_seq_pca72", "subject_summary64"]


def import_module_from_path(module_name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


V272 = import_module_from_path("stage03_v272_for_v281", V272_SCRIPT)
V276 = import_module_from_path("stage03_v276_for_v281", V276_SCRIPT)
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


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """v272 合并后存在少量重复列名，这里保留第一次出现的列。"""

    return df.loc[:, ~df.columns.duplicated()].copy()


def finite_numeric_cols(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    out = []
    for col in cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def build_bio_top3_candidates(
    events: pd.DataFrame,
    lookup: Dict[str, Dict[int, float]],
    veh_cols: List[str],
    raw_feature_sets: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """重建候选级 bio top3 表，每行是一个 query-event / raw_set / candidate。"""

    neighbor_idx_by_query, veh_z, train_mask = V272.build_query_neighbor_base(events, veh_cols, VEHICLE_K)
    event_uid = events["event_uid"].astype(str).to_numpy()
    oracle_delay = pd.to_numeric(events["oracle_delay_ms"], errors="coerce").to_numpy(dtype=int)
    latest_rmse = pd.to_numeric(events["latest_tail_rmse_v241"], errors="coerce").to_numpy(dtype=float)
    keep0_rmse = pd.to_numeric(events["keep0_tail_rmse_v241"], errors="coerce").to_numpy(dtype=float)
    oracle_rmse = pd.to_numeric(events["oracle_tail_rmse_v241"], errors="coerce").to_numpy(dtype=float)

    rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    for raw_set, raw_cols in raw_feature_sets.items():
        print(f"[v281] build candidates raw_set={raw_set} feature_n={len(raw_cols)}", flush=True)
        bio_z, _med, _mean, _std = V266.fit_fill_scale(events[raw_cols].to_numpy(dtype=float), train_mask)
        raw_cols_present = finite_numeric_cols(events, raw_cols)
        for qi, train_neighbors in neighbor_idx_by_query.items():
            if len(train_neighbors) == 0:
                continue
            use_neighbors = train_neighbors[:VEHICLE_K]
            q_uid = event_uid[qi]
            vehicle_dist = V272.pairwise_mean_sq_distance(veh_z[qi], veh_z[use_neighbors])
            bio_dist = V272.pairwise_mean_sq_distance(bio_z[qi], bio_z[use_neighbors])
            proto_delay = oracle_delay[use_neighbors]

            mapped_delay_all: List[int] = []
            mapped_rmse_all: List[float] = []
            for delay in proto_delay:
                mapped_delay, rmse = V272.rmse_at_delay(lookup, q_uid, int(delay))
                mapped_delay_all.append(int(mapped_delay))
                mapped_rmse_all.append(float(rmse))
            mapped_delay_arr = np.asarray(mapped_delay_all, dtype=int)
            mapped_rmse_arr = np.asarray(mapped_rmse_all, dtype=float)
            bio_order = np.argsort(bio_dist, kind="mergesort")
            top_pos = bio_order[: min(BIO_TOP_M, len(bio_order))]

            vehicle_best_pos = int(np.nanargmin(mapped_rmse_arr))
            vehicle_best_in_bio_top3 = bool(vehicle_best_pos in set(top_pos.tolist()))
            bio_top3_oracle_rmse = float(np.nanmin(mapped_rmse_arr[top_pos]))

            for bio_rank, pos in enumerate(top_pos, start=1):
                proto_idx = int(use_neighbors[pos])
                row: Dict[str, object] = {
                    "raw_set": raw_set,
                    "event_uid": q_uid,
                    "split": str(events.iloc[qi]["split"]),
                    "subject": str(events.iloc[qi]["subject"]),
                    "recording": str(events.iloc[qi]["recording"]),
                    "prototype_event_uid": str(event_uid[proto_idx]),
                    "mapped_delay_ms": int(mapped_delay_arr[pos]),
                    "proto_oracle_delay_ms": int(proto_delay[pos]),
                    "target_tail_rmse_v241": float(mapped_rmse_arr[pos]),
                    "latest_tail_rmse_v241": float(latest_rmse[qi]),
                    "keep0_tail_rmse_v241": float(keep0_rmse[qi]),
                    "oracle_tail_rmse_v241": float(oracle_rmse[qi]),
                    "target_gain_vs_latest": float(latest_rmse[qi] - mapped_rmse_arr[pos]),
                    "bad_top10": bool(events.iloc[qi].get("bad_top10", False)),
                    "very_bad_top5": bool(events.iloc[qi].get("very_bad_top5", False)),
                    "normal": bool(events.iloc[qi].get("normal", False)),
                    "observe_later_like": bool(events.iloc[qi].get("observe_later_like", False)),
                    "strong_steer": bool(events.iloc[qi].get("strong_steer", False)),
                    "early_best_after_400": bool(events.iloc[qi].get("early_best_after_400", False)),
                    "neighbor_rank_vehicle": int(pos + 1),
                    "bio_rank": int(bio_rank),
                    "vehicle_distance": float(vehicle_dist[pos]),
                    "bio_distance": float(bio_dist[pos]),
                    "vehicle_distance_gap_to_rank1": float(vehicle_dist[pos] - np.nanmin(vehicle_dist)),
                    "bio_distance_gap_to_rank1": float(bio_dist[pos] - np.nanmin(bio_dist)),
                    "vehicle_unique_delay_n": int(pd.Series(mapped_delay_arr).nunique()),
                    "vehicle_delay_std": float(np.nanstd(mapped_delay_arr)),
                    "vehicle_candidate_oracle_rmse": float(np.nanmin(mapped_rmse_arr)),
                    "bio_top3_oracle_rmse": bio_top3_oracle_rmse,
                    "vehicle_best_in_bio_top3": vehicle_best_in_bio_top3,
                }
                for col in veh_cols:
                    if col in events.columns:
                        row[f"veh__{col}"] = events.iloc[qi][col]
                for col in raw_cols_present:
                    row[f"bio__{col}"] = events.iloc[qi][col]
                rows.append(row)

        audit_rows.append(
            {
                "raw_set": raw_set,
                "raw_feature_n": int(len(raw_cols_present)),
                "candidate_rows": int(sum(1 for r in rows if r["raw_set"] == raw_set)),
                "vehicle_k": VEHICLE_K,
                "bio_top_m": BIO_TOP_M,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def feature_sets_for_raw(candidates: pd.DataFrame, veh_cols: List[str], raw_set: str) -> Dict[str, List[str]]:
    base = [
        "mapped_delay_ms",
        "proto_oracle_delay_ms",
        "neighbor_rank_vehicle",
        "bio_rank",
        "vehicle_distance",
        "bio_distance",
        "vehicle_distance_gap_to_rank1",
        "bio_distance_gap_to_rank1",
        "vehicle_unique_delay_n",
        "vehicle_delay_std",
    ]
    vehicle_query = [f"veh__{c}" for c in veh_cols if f"veh__{c}" in candidates.columns]
    bio_query = [c for c in candidates.columns if c.startswith("bio__")]
    return {
        f"bio_top3_{raw_set}_rankdist": base,
        f"bio_top3_{raw_set}_rankdist_vehicle": base + vehicle_query,
        f"bio_top3_{raw_set}_rankdist_vehicle_bio": base + vehicle_query + bio_query,
    }


def build_model_outputs(candidates: pd.DataFrame, veh_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_rows: List[Dict[str, object]] = []
    pred_rows: List[pd.DataFrame] = []
    top_rows: List[pd.DataFrame] = []

    for raw_set, sub in candidates.groupby("raw_set", sort=False):
        for name, cols in feature_sets_for_raw(sub, veh_cols, raw_set).items():
            print(f"[v281] train selector {name}", flush=True)
            cols = [c for c in cols if c in sub.columns]
            if not cols:
                continue
            pred_gain = V276.fit_predict_gain(sub, cols)
            compact = sub[
                [
                    "raw_set",
                    "event_uid",
                    "split",
                    "subject",
                    "prototype_event_uid",
                    "mapped_delay_ms",
                    "target_tail_rmse_v241",
                    "latest_tail_rmse_v241",
                    "target_gain_vs_latest",
                    "neighbor_rank_vehicle",
                    "bio_rank",
                    "vehicle_distance",
                    "bio_distance",
                ]
            ].copy()
            compact["feature_set"] = name
            compact["pred_gain_vs_latest"] = pred_gain
            pred_rows.append(compact)
            top_rows.append(V276.top_candidate_per_event(sub, name, pred_gain))
            feature_rows.append(
                {
                    "feature_set": name,
                    "raw_set": raw_set,
                    "feature_n": int(len(cols)),
                    "features": "|".join(cols),
                    "train_candidate_rows": int(sub["split"].astype(str).eq("train").sum()),
                    "val_candidate_rows": int(sub["split"].astype(str).eq("val").sum()),
                }
            )

    top = pd.concat(top_rows, ignore_index=True)
    search, selected = V276.threshold_search(top)
    chosen = V276.choose_configs(search)
    return pd.DataFrame(feature_rows), pd.concat(pred_rows, ignore_index=True), selected, search, chosen


def bio_top3_oracle_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    event_oracle = (
        candidates.sort_values(["raw_set", "event_uid", "target_tail_rmse_v241"])
        .drop_duplicates(["raw_set", "event_uid"])
        .copy()
    )
    rows: List[Dict[str, object]] = []
    for raw_set, sub in event_oracle.groupby("raw_set", sort=False):
        for split in ["val", "test"]:
            s = sub[sub["split"].astype(str).eq(split)].copy()
            bad = s[s["bad_top10"].astype(bool)]
            rows.append(
                {
                    "raw_set": raw_set,
                    "split": split,
                    "all_n": int(len(s)),
                    "all_oracle_rmse": float(pd.to_numeric(s["target_tail_rmse_v241"], errors="coerce").mean()),
                    "bad_top10_n": int(len(bad)),
                    "bad_top10_oracle_rmse": float(pd.to_numeric(bad["target_tail_rmse_v241"], errors="coerce").mean()),
                    "bad_top10_latest_rmse": float(pd.to_numeric(bad["latest_tail_rmse_v241"], errors="coerce").mean()),
                    "bad_top10_delta_vs_latest": float(
                        pd.to_numeric(bad["target_tail_rmse_v241"], errors="coerce").mean()
                        - pd.to_numeric(bad["latest_tail_rmse_v241"], errors="coerce").mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def decision_summary(events: pd.DataFrame, chosen: pd.DataFrame, oracle_summary: pd.DataFrame) -> pd.DataFrame:
    rows = V276.baseline_decision(events[events["split"].astype(str).eq("test")])
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

    val_bad = oracle_summary[oracle_summary["split"].astype(str).eq("val")].copy()
    test_bad = oracle_summary[oracle_summary["split"].astype(str).eq("test")].copy()
    if not val_bad.empty and not test_bad.empty:
        raw_val = str(val_bad.sort_values(["bad_top10_oracle_rmse", "raw_set"]).iloc[0]["raw_set"])
        test_val_raw = test_bad[test_bad["raw_set"].astype(str).eq(raw_val)].iloc[0]
        rows.append(
            {
                "source": "bio_top3_oracle_val_chosen",
                "label": f"{raw_val}: oracle inside bio_top3",
                "rmse": float(test_val_raw["bad_top10_oracle_rmse"]),
                "deployable": False,
                "override_rate": math.nan,
            }
        )
        best_test = test_bad.sort_values(["bad_top10_oracle_rmse", "raw_set"]).iloc[0]
        rows.append(
            {
                "source": "bio_top3_oracle_test_best",
                "label": f"{best_test['raw_set']}: oracle inside bio_top3",
                "rmse": float(best_test["bad_top10_oracle_rmse"]),
                "deployable": False,
                "override_rate": math.nan,
            }
        )

    out = pd.DataFrame(rows)
    out["delta_vs_fixed_latest"] = pd.to_numeric(out["rmse"], errors="coerce") - FIXED_WAIT_LATEST_BADTOP10
    out["passes_fixed_latest"] = pd.to_numeric(out["rmse"], errors="coerce") < FIXED_WAIT_LATEST_BADTOP10
    return out


def build_guardrail(candidates: pd.DataFrame, search: pd.DataFrame, chosen: pd.DataFrame, decision: pd.DataFrame, zip_ok: bool) -> Dict[str, object]:
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
    oracle_rows = decision[decision["source"].astype(str).str.contains("bio_top3_oracle", regex=False)]
    best_oracle = float(pd.to_numeric(oracle_rows["rmse"], errors="coerce").min()) if not oracle_rows.empty else math.nan
    return {
        "pass": bool(zip_ok),
        "zip_testzip": bool(zip_ok),
        "event_n": int(candidates["event_uid"].nunique()),
        "candidate_rows": int(len(candidates)),
        "raw_set_n": int(candidates["raw_set"].nunique()),
        "search_rows": int(len(search)),
        "chosen_rows": int(len(chosen)),
        "fixed_wait_latest_badtop10": FIXED_WAIT_LATEST_BADTOP10,
        "best_val_chosen_deployable_test_badtop10": best_deploy,
        "best_test_diagnostic_badtop10": best_diag,
        "best_bio_top3_oracle_badtop10": best_oracle,
        "best_deployable_passes_fixed_latest": bool(np.isfinite(best_deploy) and best_deploy < FIXED_WAIT_LATEST_BADTOP10),
        "best_diagnostic_passes_fixed_latest": bool(np.isfinite(best_diag) and best_diag < FIXED_WAIT_LATEST_BADTOP10),
        "bio_top3_oracle_passes_fixed_latest": bool(np.isfinite(best_oracle) and best_oracle < FIXED_WAIT_LATEST_BADTOP10),
    }


def markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def plot_decision(decision: pd.DataFrame) -> Path:
    path = FIGURES / "v281_test_badtop10_bio_top3_selector.png"
    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    x = np.arange(len(decision))
    colors = ["#4C78A8" if bool(v) else "#9C755F" for v in decision["deployable"]]
    ax.bar(x, pd.to_numeric(decision["rmse"], errors="coerce"), color=colors)
    ax.axhline(FIXED_WAIT_LATEST_BADTOP10, color="#E15759", linestyle="--", linewidth=1.2, label="fixed wait-latest")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in decision["source"]], fontsize=8)
    ax.set_ylabel("test bad_top10 tail RMSE")
    ax.set_title("v281: bio-top3 constrained selector")
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
    oracle_summary: pd.DataFrame,
    guardrail: Dict[str, object],
    figs: Iterable[Path],
) -> None:
    top_val = search.sort_values(["selection_score", "val_bad_top10_selected_rmse"]).head(18)
    top_test = search[search["test_bad_top10_override_n"] > 0].sort_values("test_bad_top10_selected_rmse").head(18)
    report = f"""# v281 bio-top3 constrained selector

## 目的

v272 的 bio top3 oracle 有少量上限，但 bio top1 和直接生理最近邻失败。v281 将问题缩窄为：在 vehicle top40 内只看 bio 最近 3 个候选，训练选择器判断哪个候选以及何时覆盖 latest。

## 核心结果

- fixed wait-latest test bad_top10: `{FIXED_WAIT_LATEST_BADTOP10:.6f}`
- val 选择的最好可部署 test bad_top10: `{guardrail["best_val_chosen_deployable_test_badtop10"]:.6f}`
- test diagnostic 最好 bad_top10: `{guardrail["best_test_diagnostic_badtop10"]:.6f}`
- bio top3 oracle bad_top10: `{guardrail["best_bio_top3_oracle_badtop10"]:.6f}`
- 可部署规则是否超过 fixed latest: `{guardrail["best_deployable_passes_fixed_latest"]}`
- diagnostic 是否超过 fixed latest: `{guardrail["best_diagnostic_passes_fixed_latest"]}`
- bio top3 oracle 是否超过 fixed latest: `{guardrail["bio_top3_oracle_passes_fixed_latest"]}`

## 决策汇总

{markdown_table(decision)}

## raw_set / top3 oracle 汇总

{markdown_table(oracle_summary)}

## 特征组

{markdown_table(feature_audit[["feature_set", "raw_set", "feature_n", "train_candidate_rows", "val_candidate_rows"]])}

## val 口径排名前 18

{markdown_table(top_val[["feature_set", "threshold", "val_bad_top10_selected_rmse", "val_bad_top10_delta_vs_latest", "val_all_delta_vs_latest", "test_bad_top10_selected_rmse", "test_bad_top10_override_rate", "selection_score"]])}

## test diagnostic 排名前 18

{markdown_table(top_test[["feature_set", "threshold", "val_bad_top10_selected_rmse", "val_bad_top10_delta_vs_latest", "val_all_delta_vs_latest", "test_bad_top10_selected_rmse", "test_bad_top10_delta_vs_latest", "test_bad_top10_override_rate"]])}

## 产物

{chr(10).join(f"- `{p.relative_to(OUT)}`" for p in figs)}
- `tables/v281_bio_top3_candidates.csv`
- `tables/v281_feature_set_audit.csv`
- `tables/v281_predictions.csv`
- `tables/v281_threshold_search.csv`
- `tables/v281_selected_by_strategy.csv`
- `tables/v281_chosen_configs.csv`
- `tables/v281_decision_summary.csv`
- `logs/guardrail_check.json`
"""
    (REPORTS / "v281_bio_top3_constrained_selector_cn.md").write_text(report, encoding="utf-8")


def write_input_hashes() -> None:
    rows = []
    for path in [V281_SCRIPT, V272_SCRIPT, V276_SCRIPT]:
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
        zf.write(V281_SCRIPT, Path("scripts") / V281_SCRIPT.name)
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def main() -> None:
    print("[v281] bio-top3 constrained selector", flush=True)
    print("[v281] 目的：把 v272 的 bio top3 oracle 上限转成可训练 selector。", flush=True)
    clean_out_dir()

    cand, events, veh_cols, raw_feature_sets, raw_feature_audit = V272.load_data()
    events = dedupe_columns(events)
    raw_feature_sets = {k: v for k, v in raw_feature_sets.items() if k in SELECTED_RAW_SETS}
    raw_feature_audit = raw_feature_audit[raw_feature_audit["raw_set"].astype(str).isin(SELECTED_RAW_SETS)].copy()
    lookup = V266.candidate_rmse_lookup(cand)
    candidates, candidate_audit = build_bio_top3_candidates(events, lookup, veh_cols, raw_feature_sets)
    feature_audit, predictions, selected, search, chosen = build_model_outputs(candidates, veh_cols)
    oracle_summary = bio_top3_oracle_summary(candidates)
    decision = decision_summary(events, chosen, oracle_summary)
    fig = plot_decision(decision)

    guardrail = build_guardrail(candidates, search, chosen, decision, zip_ok=False)
    write_csv(raw_feature_audit, TABLES / "v281_raw_feature_audit_from_v271.csv")
    write_csv(candidate_audit, TABLES / "v281_candidate_build_audit.csv")
    write_csv(candidates, TABLES / "v281_bio_top3_candidates.csv")
    write_csv(feature_audit, TABLES / "v281_feature_set_audit.csv")
    write_csv(predictions, TABLES / "v281_predictions.csv")
    write_csv(oracle_summary, TABLES / "v281_bio_top3_oracle_summary.csv")
    write_csv(search, TABLES / "v281_threshold_search.csv")
    write_csv(selected, TABLES / "v281_selected_by_strategy.csv")
    write_csv(chosen, TABLES / "v281_chosen_configs.csv")
    write_csv(decision, TABLES / "v281_decision_summary.csv")

    write_report(decision, chosen, search, feature_audit, oracle_summary, guardrail, [fig])
    write_input_hashes()
    write_file_inventory()
    zip_ok = make_zip()
    guardrail = build_guardrail(candidates, search, chosen, decision, zip_ok=zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()

    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)
    print(f"[v281] report={REPORTS / 'v281_bio_top3_constrained_selector_cn.md'}", flush=True)
    print(f"[v281] zip={ZIP_PATH}", flush=True)


if __name__ == "__main__":
    main()
