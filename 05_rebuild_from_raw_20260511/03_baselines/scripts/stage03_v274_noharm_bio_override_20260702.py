#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v274 no-harm bio override.

v273 证明：bio top5 小候选集合有 oracle headroom，但监督 selector 仍选不准。
本轮不再强迫每个样本都使用生理候选，而是做一个保守策略：

    默认使用 fixed wait-latest；
    只有 v273 pair model 对 bio-prefilter 候选足够有把握时，才用候选覆盖 latest。

阈值只在 val bad_top10 上选择；test 只报告。这个实验检验：
生理是否可以作为“稀疏高置信 override”带来 no-harm 改善。
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

OUT = BASELINES / "v274_noharm_bio_override_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v274_noharm_bio_override_20260702_pack.zip"

V272_SCRIPT = BASELINES / "scripts" / "stage03_v272_physio_ambiguity_disambiguation_20260702.py"
V273_PAIR = BASELINES / "v273_bio_prefiltered_pair_reranker_20260702" / "tables" / "v273_pair_predictions_compact.csv"
V273_DECISION = BASELINES / "v273_bio_prefiltered_pair_reranker_20260702" / "tables" / "v273_decision_summary.csv"
V273_GUARDRAIL = BASELINES / "v273_bio_prefiltered_pair_reranker_20260702" / "logs" / "guardrail_check.json"

FIXED_WAIT_LATEST_BADTOP10 = 0.695048
PRED_COLS = [
    "pred_pair_base_hgb",
    "pred_pair_vehicle_hgb",
    "pred_pair_bio_hgb",
    "pred_pair_vehicle_bio_hgb",
    "pred_pair_vehicle_bio_badweighted_hgb",
]


def import_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少前序脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


V272 = import_module_from_path("stage03_v272_for_v274", V272_SCRIPT)
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


def selected_row(
    event: pd.Series,
    strategy: str,
    family: str,
    deployable: bool,
    delay: int,
    rmse: float,
    override: bool,
    pred_score: float = math.nan,
    pred_margin: float = math.nan,
) -> Dict[str, object]:
    keep0 = float(event["keep0_tail_rmse_v241"])
    latest = float(event["latest_tail_rmse_v241"])
    oracle = float(event["oracle_tail_rmse_v241"])
    return {
        "strategy": strategy,
        "strategy_family": family,
        "deployable": bool(deployable),
        "k": math.nan,
        "bio_lambda": math.nan,
        "event_uid": str(event["event_uid"]),
        "split": str(event["split"]),
        "subject": str(event["subject"]),
        "recording": str(event["recording"]),
        "selected_delay_ms": int(delay),
        "selected_tail_rmse_v241": float(rmse),
        "keep0_tail_rmse_v241": keep0,
        "latest_tail_rmse_v241": latest,
        "oracle_tail_rmse_v241": oracle,
        "delta_selected_minus_keep0": float(rmse - keep0),
        "delta_selected_minus_latest": float(rmse - latest),
        "bad_top10": bool(event["bad_top10"]),
        "very_bad_top5": bool(event["very_bad_top5"]),
        "normal": bool(event["normal"]),
        "observe_later_like": bool(event["observe_later_like"]),
        "strong_steer": bool(event["strong_steer"]),
        "reverse": bool(event["reverse"]),
        "early_best_after_400": bool(event["early_best_after_400"]),
        "prototype_unique_delay_n": 1,
        "override_latest": bool(override),
        "override_pred_score": float(pred_score) if np.isfinite(pred_score) else math.nan,
        "override_pred_margin": float(pred_margin) if np.isfinite(pred_margin) else math.nan,
    }


def baseline_rows(events: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for _, event in events.iterrows():
        rows.append(selected_row(event, "policy_keep_0ms_anchor", "baseline", True, 0, float(event["keep0_tail_rmse_v241"]), False))
        rows.append(
            selected_row(
                event,
                "policy_wait_to_latest_anchor",
                "baseline",
                True,
                int(event["latest_delay_ms"]),
                float(event["latest_tail_rmse_v241"]),
                False,
            )
        )
        rows.append(
            selected_row(
                event,
                "oracle_best_anchor_upper_bound",
                "oracle",
                False,
                int(event["oracle_delay_ms"]),
                float(event["oracle_tail_rmse_v241"]),
                False,
            )
        )
    return rows


def build_event_candidates(pair_df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    event_cols = [
        "event_uid",
        "split",
        "subject",
        "recording",
        "keep0_delay_ms",
        "latest_delay_ms",
        "oracle_delay_ms",
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
    event_base = events[event_cols].copy()
    rows: List[Dict[str, object]] = []
    for (raw_set, event_uid), g in pair_df.groupby(["raw_set", "event_uid"], sort=False):
        for pred_col in PRED_COLS:
            if pred_col not in g.columns:
                continue
            sub = g.copy()
            score = pd.to_numeric(sub[pred_col], errors="coerce")
            if not np.isfinite(score).any():
                continue
            order = np.argsort(score.to_numpy(dtype=float), kind="mergesort")
            best_idx = int(order[0])
            second_idx = int(order[1]) if len(order) > 1 else int(order[0])
            best = sub.iloc[best_idx]
            best_score = float(score.iloc[best_idx])
            second_score = float(score.iloc[second_idx])
            rows.append(
                {
                    "raw_set": raw_set,
                    "event_uid": event_uid,
                    "pred_col": pred_col,
                    "candidate_delay_ms": int(best["mapped_delay_ms"]),
                    "candidate_rmse": float(best["target_tail_rmse_v241"]),
                    "candidate_pred_score": best_score,
                    "candidate_pred_margin": float(second_score - best_score),
                    "candidate_neighbor_rank": int(best["neighbor_rank_vehicle"]),
                    "candidate_bio_distance": float(best["bio_distance"]),
                    "candidate_vehicle_distance": float(best["vehicle_distance"]),
                }
            )
    out = pd.DataFrame(rows)
    out = out.merge(event_base, on="event_uid", how="left", validate="many_to_one")
    out["candidate_gain_vs_latest"] = pd.to_numeric(out["latest_tail_rmse_v241"], errors="coerce") - pd.to_numeric(out["candidate_rmse"], errors="coerce")
    return out


def evaluate_override_table(candidates: pd.DataFrame, score_threshold: float, margin_threshold: float) -> pd.DataFrame:
    out = candidates.copy()
    score = pd.to_numeric(out["candidate_pred_score"], errors="coerce")
    margin = pd.to_numeric(out["candidate_pred_margin"], errors="coerce")
    override = (score <= score_threshold) & (margin >= margin_threshold)
    out["override_latest"] = override
    out["selected_tail_rmse_v241"] = np.where(override, out["candidate_rmse"], out["latest_tail_rmse_v241"])
    out["selected_delay_ms"] = np.where(override, out["candidate_delay_ms"], out["latest_delay_ms"])
    return out


def tune_thresholds(candidates: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    chosen_rows: List[Dict[str, object]] = []
    for (raw_set, pred_col), sub in candidates.groupby(["raw_set", "pred_col"], sort=False):
        val = sub[sub["split"].astype(str).eq("val") & sub["bad_top10"].astype(bool)].copy()
        if val.empty:
            continue
        scores = pd.to_numeric(val["candidate_pred_score"], errors="coerce")
        margins = pd.to_numeric(val["candidate_pred_margin"], errors="coerce")
        score_grid = sorted(set([float("-inf")] + [float(x) for x in np.nanquantile(scores, np.linspace(0.05, 1.0, 20)) if np.isfinite(x)]))
        margin_grid = sorted(set([float("-inf"), 0.0] + [float(x) for x in np.nanquantile(margins, [0.25, 0.50, 0.75, 0.90]) if np.isfinite(x)]))
        for score_thr in score_grid:
            for margin_thr in margin_grid:
                ev = evaluate_override_table(sub, score_thr, margin_thr)
                val_ev = ev[ev["split"].astype(str).eq("val") & ev["bad_top10"].astype(bool)]
                if val_ev.empty:
                    continue
                rmse = float(pd.to_numeric(val_ev["selected_tail_rmse_v241"], errors="coerce").mean())
                latest = float(pd.to_numeric(val_ev["latest_tail_rmse_v241"], errors="coerce").mean())
                override_rate = float(val_ev["override_latest"].astype(bool).mean())
                active_n = int(val_ev["override_latest"].astype(bool).sum())
                rows.append(
                    {
                        "raw_set": raw_set,
                        "pred_col": pred_col,
                        "score_threshold": score_thr,
                        "margin_threshold": margin_thr,
                        "val_bad_top10_rmse": rmse,
                        "val_bad_top10_latest_rmse": latest,
                        "val_bad_top10_delta_vs_latest": rmse - latest,
                        "val_bad_top10_override_rate": override_rate,
                        "val_bad_top10_override_n": active_n,
                    }
                )
    search = pd.DataFrame(rows)
    if search.empty:
        return search, search

    for (raw_set, pred_col), sub in search.groupby(["raw_set", "pred_col"], sort=False):
        # any: 允许不覆盖；active: 至少覆盖一个 val bad_top10 样本。
        for label, filt in [
            ("best_any", np.ones(len(sub), dtype=bool)),
            ("best_active", sub["val_bad_top10_override_n"].to_numpy(dtype=int) > 0),
            ("best_noharm_active", (sub["val_bad_top10_override_n"].to_numpy(dtype=int) > 0) & (sub["val_bad_top10_delta_vs_latest"].to_numpy(dtype=float) <= 0.0)),
        ]:
            cand = sub[filt].copy()
            if cand.empty:
                continue
            cand = cand.sort_values(
                ["val_bad_top10_rmse", "val_bad_top10_override_rate", "margin_threshold", "score_threshold"],
                ascending=[True, True, False, True],
            )
            best = cand.iloc[0].to_dict()
            best["chosen_type"] = label
            chosen_rows.append(best)
    chosen = pd.DataFrame(chosen_rows)
    return search, chosen


def build_selected(events: pd.DataFrame, candidates: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    event_map = {str(row["event_uid"]): row for _, row in events.iterrows()}
    rows = baseline_rows(events)
    for _, cfg in chosen.iterrows():
        raw_set = str(cfg["raw_set"])
        pred_col = str(cfg["pred_col"])
        chosen_type = str(cfg["chosen_type"])
        score_thr = float(cfg["score_threshold"])
        margin_thr = float(cfg["margin_threshold"])
        sub = candidates[candidates["raw_set"].astype(str).eq(raw_set) & candidates["pred_col"].astype(str).eq(pred_col)].copy()
        ev = evaluate_override_table(sub, score_thr, margin_thr)
        strategy = f"override_{chosen_type}_{raw_set}_{pred_col.replace('pred_', '')}"
        for _, row in ev.iterrows():
            event = event_map[str(row["event_uid"])]
            rows.append(
                selected_row(
                    event,
                    strategy,
                    "bio_override",
                    True,
                    int(row["selected_delay_ms"]),
                    float(row["selected_tail_rmse_v241"]),
                    bool(row["override_latest"]),
                    float(row["candidate_pred_score"]),
                    float(row["candidate_pred_margin"]),
                )
            )
    return pd.DataFrame(rows)


def summarize_selected(selected: pd.DataFrame) -> pd.DataFrame:
    summary = V266.summarize_selected(selected)
    override = (
        selected.groupby(["split", "strategy"], as_index=False)
        .agg(override_rate=("override_latest", "mean"), override_n=("override_latest", "sum"))
    )
    return summary.merge(override, on=["split", "strategy"], how="left")


def choose_cross_strategy(summary: pd.DataFrame, chosen_type: str) -> pd.DataFrame:
    val = summary[
        summary["split"].eq("val")
        & summary["event_group"].eq("bad_top10")
        & summary["strategy"].astype(str).str.startswith(f"override_{chosen_type}_")
    ].copy()
    if val.empty:
        return pd.DataFrame()
    val = val.sort_values(["selected_tail_rmse_mean", "override_rate", "strategy"], ascending=[True, True, True])
    strategy = str(val.iloc[0]["strategy"])
    return summary[
        summary["strategy"].astype(str).eq(strategy)
        & summary["event_group"].eq("bad_top10")
        & summary["split"].isin(["val", "test"])
    ].copy()


def build_decision(summary: pd.DataFrame, v273_decision: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    test_bad = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")].copy()
    for strategy, source in [
        ("policy_keep_0ms_anchor", "baseline"),
        ("policy_wait_to_latest_anchor", "baseline"),
        ("oracle_best_anchor_upper_bound", "oracle"),
    ]:
        sub = test_bad[test_bad["strategy"].eq(strategy)]
        if len(sub):
            rows.append({"source": source, "label": strategy, "rmse": float(sub.iloc[0]["selected_tail_rmse_mean"]), "deployable": source == "baseline"})
    cand = v273_decision[v273_decision["source"].astype(str).eq("bio_prefilter_candidate_oracle")]
    if len(cand):
        rows.append({"source": "bio_prefilter_candidate_oracle", "label": str(cand.iloc[0]["label"]), "rmse": float(cand.iloc[0]["rmse"]), "deployable": False})
    learned = test_bad[test_bad["strategy"].astype(str).str.startswith("override_")].sort_values("selected_tail_rmse_mean")
    if len(learned):
        row = learned.iloc[0]
        rows.append(
            {
                "source": "test_best_override_diagnostic",
                "label": str(row["strategy"]),
                "rmse": float(row["selected_tail_rmse_mean"]),
                "deployable": False,
                "override_rate": float(row.get("override_rate", math.nan)),
            }
        )
    for chosen_type in ["best_any", "best_active", "best_noharm_active"]:
        mapped = choose_cross_strategy(summary, chosen_type)
        test = mapped[mapped["split"].eq("test")]
        if len(test):
            row = test.iloc[0]
            rows.append(
                {
                    "source": f"val_{chosen_type}",
                    "label": str(row["strategy"]),
                    "rmse": float(row["selected_tail_rmse_mean"]),
                    "deployable": True,
                    "override_rate": float(row.get("override_rate", math.nan)),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["delta_vs_fixed_latest"] = out["rmse"] - FIXED_WAIT_LATEST_BADTOP10
        out["passes_fixed_latest"] = out["rmse"] < FIXED_WAIT_LATEST_BADTOP10
    return out


def plot_decision(decision: pd.DataFrame) -> Path:
    path = FIGURES / "v274_test_badtop10_noharm_override.png"
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
    ax.set_title("v274: sparse no-harm override from bio-prefiltered candidates")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v272_script", V272_SCRIPT),
        ("v273_pair_predictions", V273_PAIR),
        ("v273_decision", V273_DECISION),
        ("v273_guardrail", V273_GUARDRAIL),
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


def write_report(decision: pd.DataFrame, summary: pd.DataFrame, chosen: pd.DataFrame, threshold_search: pd.DataFrame, figs: List[Path]) -> None:
    lines: List[str] = []
    lines.append("# v274 no-harm bio override")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- v273 的 bio top5 candidate oracle 有上界，但强制选择候选会失败。")
    lines.append("- v274 默认使用 fixed wait-latest，只在 pair model 对生理候选高置信时才覆盖。")
    lines.append("- 阈值只在 val bad_top10 上选，test 只报告。")
    lines.append("")
    lines.append("## test bad_top10 决策收口")
    lines.append("")
    lines.append(decision.to_markdown(index=False) if len(decision) else "- 无可用结果。")
    lines.append("")
    lines.append("## val 选择的 override 策略")
    lines.append("")
    show = chosen.sort_values(["chosen_type", "val_bad_top10_rmse"]).head(30)
    lines.append(show.to_markdown(index=False) if len(show) else "- 没有 active override 策略。")
    lines.append("")
    lines.append("## test bad_top10 override top")
    lines.append("")
    test = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["strategy"].astype(str).str.startswith("override_")
    ].sort_values("selected_tail_rmse_mean").head(20)
    cols = ["strategy", "selected_tail_rmse_mean", "delta_selected_minus_latest_mean", "override_rate", "selected_delay_ms_mean", "selected_latest_rate"]
    lines.append(test[[c for c in cols if c in test.columns]].to_markdown(index=False) if len(test) else "- 无 override 结果。")
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    deploy = decision[decision["deployable"].astype(bool) & decision["source"].astype(str).str.startswith("val_")] if len(decision) else pd.DataFrame()
    if len(deploy) and bool(deploy["passes_fixed_latest"].any()):
        lines.append("- 至少一个 val 选择的 no-harm override 低于 fixed wait-latest，说明生理可作为稀疏覆盖信号。")
    else:
        lines.append("- val 选择的 no-harm override 仍未低于 fixed wait-latest，稀疏覆盖也没有兑现生理上界。")
    lines.append("- 若本轮失败，现有生理在可部署层面的主增量基本被证伪，应回到车辆多未来/不确定性主线。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figs:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v274_noharm_bio_override_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v274] no-harm bio override", flush=True)
    clean_out_dir()
    cand, events, _veh_cols, _feature_sets, _feature_audit = V272.load_data()
    pair_df = pd.read_csv(V273_PAIR, encoding="utf-8-sig", low_memory=False)
    v273_decision = pd.read_csv(V273_DECISION, encoding="utf-8-sig", low_memory=False)
    event_candidates = build_event_candidates(pair_df, events)
    threshold_search, chosen = tune_thresholds(event_candidates)
    selected = build_selected(events, event_candidates, chosen)
    summary = summarize_selected(selected)
    decision = build_decision(summary, v273_decision)
    fig = plot_decision(decision)

    write_csv(event_candidates, TABLES / "v274_event_candidate_predictions.csv")
    write_csv(threshold_search, TABLES / "v274_threshold_search.csv")
    write_csv(chosen, TABLES / "v274_chosen_thresholds.csv")
    write_csv(selected, TABLES / "v274_selected_by_strategy.csv")
    write_csv(summary, TABLES / "v274_override_summary.csv")
    write_csv(decision, TABLES / "v274_decision_summary.csv")
    write_input_hashes()
    write_file_inventory()
    write_report(decision, summary, chosen, threshold_search, [fig])
    write_file_inventory()
    zip_ok = make_zip()

    v273_guard = json.loads(V273_GUARDRAIL.read_text(encoding="utf-8")) if V273_GUARDRAIL.exists() else {}
    deploy = decision[decision["deployable"].astype(bool) & decision["source"].astype(str).str.startswith("val_")] if len(decision) else pd.DataFrame()
    best_deploy = float(deploy["rmse"].min()) if len(deploy) else math.nan
    guardrail = {
        "pass": bool(zip_ok and bool(v273_guard.get("pass", False)) and len(threshold_search) > 0 and len(summary) > 0),
        "zip_testzip": bool(zip_ok),
        "v273_guardrail_pass": bool(v273_guard.get("pass", False)),
        "event_n": int(events["event_uid"].nunique()),
        "candidate_event_model_rows": int(len(event_candidates)),
        "threshold_search_rows": int(len(threshold_search)),
        "chosen_strategy_rows": int(len(chosen)),
        "best_val_chosen_deployable_test_badtop10": best_deploy,
        "fixed_wait_latest_badtop10": float(FIXED_WAIT_LATEST_BADTOP10),
        "best_deployable_passes_fixed_latest": bool(np.isfinite(best_deploy) and best_deploy < FIXED_WAIT_LATEST_BADTOP10),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v274 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    print(f"[v274] report={REPORTS / 'v274_noharm_bio_override_cn.md'}", flush=True)
    print(f"[v274] zip={ZIP_PATH}", flush=True)
    if len(decision):
        print(decision.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
