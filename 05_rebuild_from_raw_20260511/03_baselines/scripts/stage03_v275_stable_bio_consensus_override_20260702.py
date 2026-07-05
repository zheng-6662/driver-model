#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v275 stable bio consensus override.

v274 只用单个 raw_set + pred_col 的阈值做 sparse override，test 上有很小的
diagnostic headroom，但 val 选不出能泛化的策略。v275 不继续堆 selector 深度，
而是把生理信号改成“稳定一致性证据”：

1. 仍然默认使用 fixed wait-latest，不让生理直接主导每个样本。
2. 对同一事件，收集 v274 中所有 calibrated physiology raw_set / pred_col 的
   候选锚点。
3. 只有多个生理视角同时支持同一个非 latest 锚点，并且支持票数超过 latest
   票数时，才允许 override。
4. 阈值和投票规则只在 val 上选择；test 只报告。
5. 选择时不仅看 val bad_top10，也检查 val all / normal / strong_steer /
   observe_later_like 等子集，避免“只救少数 bad 样本但整体伤害”的假阳性。

这个实验检验：生理是否能作为车辆不确定性下的稳定辅助证据，而不是单点 selector。
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


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

OUT = BASELINES / "v275_stable_bio_consensus_override_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v275_stable_bio_consensus_override_20260702_pack.zip"

V274_CANDIDATES = BASELINES / "v274_noharm_bio_override_20260702" / "tables" / "v274_event_candidate_predictions.csv"
V274_DECISION = BASELINES / "v274_noharm_bio_override_20260702" / "tables" / "v274_decision_summary.csv"
V274_GUARDRAIL = BASELINES / "v274_noharm_bio_override_20260702" / "logs" / "guardrail_check.json"

FIXED_WAIT_LATEST_BADTOP10 = 0.695048

EVENT_COLS = [
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

GROUP_FLAGS = [
    ("all", None),
    ("bad_top10", "bad_top10"),
    ("very_bad_top5", "very_bad_top5"),
    ("normal", "normal"),
    ("observe_later_like", "observe_later_like"),
    ("strong_steer", "strong_steer"),
    ("early_best_after_400", "early_best_after_400"),
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


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if not V274_CANDIDATES.exists():
        raise FileNotFoundError(f"缺少 v274 候选表：{V274_CANDIDATES}")
    candidates = pd.read_csv(V274_CANDIDATES, encoding="utf-8-sig", low_memory=False)
    decision = pd.read_csv(V274_DECISION, encoding="utf-8-sig", low_memory=False) if V274_DECISION.exists() else pd.DataFrame()
    guardrail = json.loads(V274_GUARDRAIL.read_text(encoding="utf-8")) if V274_GUARDRAIL.exists() else {}

    numeric_cols = [
        "candidate_delay_ms",
        "candidate_rmse",
        "candidate_pred_score",
        "candidate_pred_margin",
        "candidate_neighbor_rank",
        "candidate_bio_distance",
        "candidate_vehicle_distance",
        "keep0_delay_ms",
        "latest_delay_ms",
        "oracle_delay_ms",
        "keep0_tail_rmse_v241",
        "latest_tail_rmse_v241",
        "oracle_tail_rmse_v241",
        "candidate_gain_vs_latest",
    ]
    for col in numeric_cols:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    for col in ["bad_top10", "very_bad_top5", "normal", "observe_later_like", "strong_steer", "reverse", "early_best_after_400"]:
        if col in candidates.columns:
            candidates[col] = candidates[col].astype(bool)

    events = candidates[EVENT_COLS].drop_duplicates("event_uid").copy()
    return candidates, events, decision, guardrail


def make_grid(candidates: pd.DataFrame) -> pd.DataFrame:
    """只用 val bad_top10 的候选分布生成候选阈值，避免 test 参与调参。"""
    val = candidates[candidates["split"].astype(str).eq("val") & candidates["bad_top10"].astype(bool)].copy()
    if val.empty:
        raise ValueError("val bad_top10 为空，无法调 v275 consensus 阈值")
    score_values = [
        float(x)
        for x in np.nanquantile(pd.to_numeric(val["candidate_pred_score"], errors="coerce"), [0.10, 0.20, 0.35, 0.50])
        if np.isfinite(x)
    ]
    margin_values = [float("-inf"), 0.0] + [
        float(x)
        for x in np.nanquantile(pd.to_numeric(val["candidate_pred_margin"], errors="coerce"), [0.25, 0.50, 0.75])
        if np.isfinite(x)
    ]
    rows: List[Dict[str, object]] = []
    for score_threshold in sorted(set(score_values + [float("inf")])):
        for margin_threshold in sorted(set(margin_values)):
            for min_votes in [2, 4, 6, 8, 10]:
                for vote_margin in [0, 1, 2]:
                    for min_raw_sets in [1, 2]:
                        rows.append(
                            {
                                "score_threshold": score_threshold,
                                "margin_threshold": margin_threshold,
                                "min_votes": min_votes,
                                "vote_margin": vote_margin,
                                "min_raw_sets": min_raw_sets,
                            }
                        )
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def select_by_consensus(candidates: pd.DataFrame, events: pd.DataFrame, cfg: pd.Series) -> pd.DataFrame:
    """
    对一个投票配置生成每个事件的最终选择。

    只使用候选预测分数、margin、raw_set/pred_col 一致性和候选 delay；
    selected_rmse 只在离线评估时用于打分，不参与 test 配置选择。
    """
    score_threshold = float(cfg["score_threshold"])
    margin_threshold = float(cfg["margin_threshold"])
    min_votes = int(cfg["min_votes"])
    vote_margin = int(cfg["vote_margin"])
    min_raw_sets = int(cfg["min_raw_sets"])

    filt = candidates[
        (pd.to_numeric(candidates["candidate_pred_score"], errors="coerce") <= score_threshold)
        & (pd.to_numeric(candidates["candidate_pred_margin"], errors="coerce") >= margin_threshold)
    ].copy()
    if filt.empty:
        out = events.copy()
        out["selected_delay_ms"] = out["latest_delay_ms"]
        out["selected_tail_rmse_v241"] = out["latest_tail_rmse_v241"]
        out["override_latest"] = False
        out["consensus_vote_n"] = 0
        out["consensus_latest_vote_n"] = 0
        out["consensus_raw_set_n"] = 0
        out["consensus_pred_col_n"] = 0
        out["consensus_score"] = math.nan
        out["consensus_margin"] = math.nan
        return out

    # 每个 event-delay 用最高置信候选作为代表轨迹；投票数只统计独立视角数量。
    ordered = filt.sort_values(
        ["event_uid", "candidate_delay_ms", "candidate_pred_score", "candidate_pred_margin"],
        ascending=[True, True, True, False],
    )
    representative = ordered.drop_duplicates(["event_uid", "candidate_delay_ms"]).copy()
    aggregate = (
        filt.groupby(["event_uid", "candidate_delay_ms"], as_index=False)
        .agg(
            vote_n=("candidate_delay_ms", "size"),
            raw_set_n=("raw_set", "nunique"),
            pred_col_n=("pred_col", "nunique"),
            mean_margin=("candidate_pred_margin", "mean"),
            best_score=("candidate_pred_score", "min"),
        )
        .merge(
            representative[
                [
                    "event_uid",
                    "candidate_delay_ms",
                    "candidate_rmse",
                    "candidate_pred_score",
                    "candidate_pred_margin",
                    "latest_delay_ms",
                ]
            ],
            on=["event_uid", "candidate_delay_ms"],
            how="left",
            validate="one_to_one",
        )
    )

    latest_votes = aggregate[aggregate["candidate_delay_ms"].eq(aggregate["latest_delay_ms"])][["event_uid", "vote_n"]]
    latest_votes = latest_votes.rename(columns={"vote_n": "latest_vote_n"})
    aggregate = aggregate.merge(latest_votes, on="event_uid", how="left")
    aggregate["latest_vote_n"] = pd.to_numeric(aggregate["latest_vote_n"], errors="coerce").fillna(0)

    # latest 是默认动作，不作为 override 目标；只允许多个生理视角一致支持非 latest。
    non_latest = aggregate[~aggregate["candidate_delay_ms"].eq(aggregate["latest_delay_ms"])].copy()
    non_latest = non_latest[
        (non_latest["vote_n"] >= min_votes)
        & ((non_latest["vote_n"] - non_latest["latest_vote_n"]) >= vote_margin)
        & (non_latest["raw_set_n"] >= min_raw_sets)
    ].copy()
    if non_latest.empty:
        out = events.copy()
        out["selected_delay_ms"] = out["latest_delay_ms"]
        out["selected_tail_rmse_v241"] = out["latest_tail_rmse_v241"]
        out["override_latest"] = False
        out["consensus_vote_n"] = 0
        out["consensus_latest_vote_n"] = 0
        out["consensus_raw_set_n"] = 0
        out["consensus_pred_col_n"] = 0
        out["consensus_score"] = math.nan
        out["consensus_margin"] = math.nan
        return out

    chosen = non_latest.sort_values(
        ["event_uid", "vote_n", "raw_set_n", "pred_col_n", "mean_margin", "best_score"],
        ascending=[True, False, False, False, False, True],
    ).drop_duplicates("event_uid")
    chosen = chosen.rename(
        columns={
            "candidate_delay_ms": "consensus_delay_ms",
            "candidate_rmse": "consensus_tail_rmse_v241",
            "vote_n": "consensus_vote_n",
            "latest_vote_n": "consensus_latest_vote_n",
            "raw_set_n": "consensus_raw_set_n",
            "pred_col_n": "consensus_pred_col_n",
            "best_score": "consensus_score",
            "mean_margin": "consensus_margin",
        }
    )
    out = events.merge(
        chosen[
            [
                "event_uid",
                "consensus_delay_ms",
                "consensus_tail_rmse_v241",
                "consensus_vote_n",
                "consensus_latest_vote_n",
                "consensus_raw_set_n",
                "consensus_pred_col_n",
                "consensus_score",
                "consensus_margin",
            ]
        ],
        on="event_uid",
        how="left",
    )
    out["override_latest"] = out["consensus_tail_rmse_v241"].notna()
    out["selected_delay_ms"] = np.where(out["override_latest"], out["consensus_delay_ms"], out["latest_delay_ms"])
    out["selected_tail_rmse_v241"] = np.where(
        out["override_latest"], out["consensus_tail_rmse_v241"], out["latest_tail_rmse_v241"]
    )
    for col in ["consensus_vote_n", "consensus_latest_vote_n", "consensus_raw_set_n", "consensus_pred_col_n"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    return out


def group_metrics(selected: pd.DataFrame, split: str, flag: str | None) -> Dict[str, float]:
    mask = selected["split"].astype(str).eq(split)
    if flag is not None:
        mask &= selected[flag].astype(bool)
    sub = selected[mask].copy()
    if sub.empty:
        return {
            "n": 0,
            "selected_rmse": math.nan,
            "latest_rmse": math.nan,
            "delta_vs_latest": math.nan,
            "override_n": 0,
            "override_rate": math.nan,
        }
    selected_rmse = float(pd.to_numeric(sub["selected_tail_rmse_v241"], errors="coerce").mean())
    latest_rmse = float(pd.to_numeric(sub["latest_tail_rmse_v241"], errors="coerce").mean())
    override_n = int(sub["override_latest"].astype(bool).sum())
    return {
        "n": int(len(sub)),
        "selected_rmse": selected_rmse,
        "latest_rmse": latest_rmse,
        "delta_vs_latest": selected_rmse - latest_rmse,
        "override_n": override_n,
        "override_rate": float(override_n / max(len(sub), 1)),
    }


def evaluate_grid(candidates: pd.DataFrame, events: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for idx, cfg in grid.iterrows():
        selected = select_by_consensus(candidates, events, cfg)
        row: Dict[str, object] = {"config_id": int(idx)}
        row.update(cfg.to_dict())
        for split in ["val", "test"]:
            for group_name, flag in GROUP_FLAGS:
                m = group_metrics(selected, split, flag)
                prefix = f"{split}_{group_name}"
                for key, value in m.items():
                    row[f"{prefix}_{key}"] = value

        # 稳定性约束：目标子集要改善，整体和常规样本不能明显受伤。
        row["active"] = int(row.get("val_bad_top10_override_n", 0)) > 0
        row["stable_pass"] = bool(
            row["active"]
            and float(row.get("val_bad_top10_delta_vs_latest", math.inf)) <= 0.0
            and float(row.get("val_all_delta_vs_latest", math.inf)) <= 0.003
            and float(row.get("val_normal_delta_vs_latest", math.inf)) <= 0.005
            and float(row.get("val_strong_steer_delta_vs_latest", math.inf)) <= 0.005
            and float(row.get("val_observe_later_like_delta_vs_latest", math.inf)) <= 0.005
        )
        harm_terms = [
            max(0.0, float(row.get("val_all_delta_vs_latest", 0.0))),
            max(0.0, float(row.get("val_normal_delta_vs_latest", 0.0))),
            max(0.0, float(row.get("val_strong_steer_delta_vs_latest", 0.0))),
            max(0.0, float(row.get("val_observe_later_like_delta_vs_latest", 0.0))),
        ]
        row["stability_penalty"] = float(sum(harm_terms))
        row["selection_score"] = float(row.get("val_bad_top10_delta_vs_latest", 0.0)) + 3.0 * float(row["stability_penalty"])
        rows.append(row)
    return pd.DataFrame(rows)


def choose_configs(search: pd.DataFrame) -> pd.DataFrame:
    chosen_rows: List[Dict[str, object]] = []

    def add_choice(label: str, sub: pd.DataFrame, deployable: bool = True) -> None:
        if sub.empty:
            return
        best = sub.sort_values(
            [
                "selection_score",
                "val_bad_top10_selected_rmse",
                "val_bad_top10_override_rate",
                "min_votes",
                "min_raw_sets",
            ],
            ascending=[True, True, True, False, False],
        ).iloc[0].to_dict()
        best["chosen_type"] = label
        best["deployable"] = bool(deployable)
        chosen_rows.append(best)

    add_choice("best_any", search.copy())
    add_choice("best_active", search[search["active"].astype(bool)].copy())
    add_choice("best_stable_active", search[search["stable_pass"].astype(bool)].copy())
    noharm = search[
        search["active"].astype(bool)
        & (pd.to_numeric(search["val_bad_top10_delta_vs_latest"], errors="coerce") <= 0.0)
        & (pd.to_numeric(search["val_all_delta_vs_latest"], errors="coerce") <= 0.0)
    ].copy()
    add_choice("best_noharm_all", noharm)

    test_diag = search[search["test_bad_top10_override_n"] > 0].copy()
    if not test_diag.empty:
        best = test_diag.sort_values(
            ["test_bad_top10_selected_rmse", "test_bad_top10_override_rate", "val_bad_top10_delta_vs_latest"],
            ascending=[True, True, True],
        ).iloc[0].to_dict()
        best["chosen_type"] = "test_best_diagnostic"
        best["deployable"] = False
        chosen_rows.append(best)

    return pd.DataFrame(chosen_rows)


def baseline_selected(events: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, event in events.iterrows():
        for strategy, delay_col, rmse_col, family, deployable in [
            ("policy_keep_0ms_anchor", "keep0_delay_ms", "keep0_tail_rmse_v241", "baseline", True),
            ("policy_wait_to_latest_anchor", "latest_delay_ms", "latest_tail_rmse_v241", "baseline", True),
            ("oracle_best_anchor_upper_bound", "oracle_delay_ms", "oracle_tail_rmse_v241", "oracle", False),
        ]:
            rows.append(
                {
                    "strategy": strategy,
                    "strategy_family": family,
                    "deployable": deployable,
                    "event_uid": event["event_uid"],
                    "split": event["split"],
                    "subject": event["subject"],
                    "recording": event["recording"],
                    "selected_delay_ms": int(event[delay_col]),
                    "selected_tail_rmse_v241": float(event[rmse_col]),
                    "keep0_tail_rmse_v241": float(event["keep0_tail_rmse_v241"]),
                    "latest_tail_rmse_v241": float(event["latest_tail_rmse_v241"]),
                    "oracle_tail_rmse_v241": float(event["oracle_tail_rmse_v241"]),
                    "bad_top10": bool(event["bad_top10"]),
                    "very_bad_top5": bool(event["very_bad_top5"]),
                    "normal": bool(event["normal"]),
                    "observe_later_like": bool(event["observe_later_like"]),
                    "strong_steer": bool(event["strong_steer"]),
                    "reverse": bool(event["reverse"]),
                    "early_best_after_400": bool(event["early_best_after_400"]),
                    "override_latest": False,
                    "consensus_vote_n": 0,
                    "consensus_latest_vote_n": 0,
                    "consensus_raw_set_n": 0,
                    "consensus_pred_col_n": 0,
                }
            )
    return pd.DataFrame(rows)


def build_selected_for_choices(candidates: pd.DataFrame, events: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    selected_parts = [baseline_selected(events)]
    if chosen.empty:
        return pd.concat(selected_parts, ignore_index=True)
    for _, cfg in chosen.iterrows():
        if str(cfg["chosen_type"]) == "test_best_diagnostic":
            deployable = False
        else:
            deployable = True
        sel = select_by_consensus(candidates, events, cfg)
        sel = sel.copy()
        sel["strategy"] = (
            "consensus_"
            + str(cfg["chosen_type"])
            + f"_score{float(cfg['score_threshold']):.6g}"
            + f"_margin{float(cfg['margin_threshold']):.6g}"
            + f"_votes{int(cfg['min_votes'])}"
            + f"_vm{int(cfg['vote_margin'])}"
            + f"_raw{int(cfg['min_raw_sets'])}"
        )
        sel["strategy_family"] = "stable_bio_consensus"
        sel["deployable"] = bool(deployable)
        selected_parts.append(sel)
    out = pd.concat(selected_parts, ignore_index=True)
    out["delta_selected_minus_latest"] = (
        pd.to_numeric(out["selected_tail_rmse_v241"], errors="coerce")
        - pd.to_numeric(out["latest_tail_rmse_v241"], errors="coerce")
    )
    out["delta_selected_minus_keep0"] = (
        pd.to_numeric(out["selected_tail_rmse_v241"], errors="coerce")
        - pd.to_numeric(out["keep0_tail_rmse_v241"], errors="coerce")
    )
    return out


def summarize_selected(selected: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (split, strategy), sub in selected.groupby(["split", "strategy"], sort=False):
        for group_name, flag in GROUP_FLAGS:
            mask = pd.Series(True, index=sub.index)
            if flag is not None:
                mask &= sub[flag].astype(bool)
            g = sub[mask]
            if g.empty:
                continue
            selected_rmse = float(pd.to_numeric(g["selected_tail_rmse_v241"], errors="coerce").mean())
            latest_rmse = float(pd.to_numeric(g["latest_tail_rmse_v241"], errors="coerce").mean())
            keep0_rmse = float(pd.to_numeric(g["keep0_tail_rmse_v241"], errors="coerce").mean())
            oracle_rmse = float(pd.to_numeric(g["oracle_tail_rmse_v241"], errors="coerce").mean())
            rows.append(
                {
                    "split": split,
                    "event_group": group_name,
                    "strategy": strategy,
                    "strategy_family": str(g["strategy_family"].iloc[0]),
                    "deployable": bool(g["deployable"].iloc[0]),
                    "n": int(len(g)),
                    "selected_tail_rmse_mean": selected_rmse,
                    "keep0_tail_rmse_mean": keep0_rmse,
                    "latest_tail_rmse_mean": latest_rmse,
                    "oracle_tail_rmse_mean": oracle_rmse,
                    "delta_selected_minus_latest_mean": selected_rmse - latest_rmse,
                    "selected_delay_ms_mean": float(pd.to_numeric(g["selected_delay_ms"], errors="coerce").mean()),
                    "selected_latest_rate": float((pd.to_numeric(g["selected_delay_ms"], errors="coerce") == pd.to_numeric(g["latest_delay_ms"], errors="coerce")).mean()),
                    "override_rate": float(g["override_latest"].astype(bool).mean()),
                    "override_n": int(g["override_latest"].astype(bool).sum()),
                    "consensus_vote_n_mean": float(pd.to_numeric(g["consensus_vote_n"], errors="coerce").mean()),
                    "consensus_raw_set_n_mean": float(pd.to_numeric(g["consensus_raw_set_n"], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)


def build_decision(summary: pd.DataFrame, search: pd.DataFrame, chosen: pd.DataFrame, v274_decision: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    test_bad = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")].copy()
    for strategy, source in [
        ("policy_keep_0ms_anchor", "baseline"),
        ("policy_wait_to_latest_anchor", "baseline"),
        ("oracle_best_anchor_upper_bound", "oracle"),
    ]:
        sub = test_bad[test_bad["strategy"].eq(strategy)]
        if len(sub):
            rows.append(
                {
                    "source": source,
                    "label": strategy,
                    "rmse": float(sub.iloc[0]["selected_tail_rmse_mean"]),
                    "deployable": source == "baseline",
                    "override_rate": math.nan,
                }
            )
    cand = v274_decision[v274_decision["source"].astype(str).eq("bio_prefilter_candidate_oracle")]
    if len(cand):
        rows.append(
            {
                "source": "bio_prefilter_candidate_oracle",
                "label": str(cand.iloc[0]["label"]),
                "rmse": float(cand.iloc[0]["rmse"]),
                "deployable": False,
                "override_rate": math.nan,
            }
        )

    for _, row in chosen.iterrows():
        label_prefix = "test_best_consensus_diagnostic" if str(row["chosen_type"]) == "test_best_diagnostic" else f"val_{row['chosen_type']}"
        rows.append(
            {
                "source": label_prefix,
                "label": (
                    f"score={row['score_threshold']};margin={row['margin_threshold']};"
                    f"votes={int(row['min_votes'])};vote_margin={int(row['vote_margin'])};raw={int(row['min_raw_sets'])}"
                ),
                "rmse": float(row["test_bad_top10_selected_rmse"]),
                "deployable": bool(row["deployable"]),
                "override_rate": float(row["test_bad_top10_override_rate"]),
                "val_bad_delta": float(row["val_bad_top10_delta_vs_latest"]),
                "val_all_delta": float(row["val_all_delta_vs_latest"]),
                "stable_pass": bool(row.get("stable_pass", False)),
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        out["delta_vs_fixed_latest"] = pd.to_numeric(out["rmse"], errors="coerce") - FIXED_WAIT_LATEST_BADTOP10
        out["passes_fixed_latest"] = pd.to_numeric(out["rmse"], errors="coerce") < FIXED_WAIT_LATEST_BADTOP10
    return out


def plot_decision(decision: pd.DataFrame) -> Path:
    path = FIGURES / "v275_test_badtop10_stable_consensus.png"
    if decision.empty:
        return path
    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    x = np.arange(len(decision))
    colors = ["#4C78A8" if bool(v) else "#9C755F" for v in decision["deployable"]]
    ax.bar(x, pd.to_numeric(decision["rmse"], errors="coerce"), color=colors)
    ax.axhline(FIXED_WAIT_LATEST_BADTOP10, color="#E15759", linestyle="--", linewidth=1.2, label="fixed wait-latest")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in decision["source"]], fontsize=8)
    ax.set_ylabel("test bad_top10 tail RMSE")
    ax.set_title("v275: stable physiology consensus override")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(decision: pd.DataFrame, chosen: pd.DataFrame, search: pd.DataFrame, figs: Iterable[Path]) -> None:
    lines: List[str] = []
    lines.append("# v275 stable bio consensus override")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- v274 的单模型 sparse override 在 test-best diagnostic 上有极小改善，但 val 选择无法泛化。")
    lines.append("- v275 改为多生理视角一致投票：多个 raw_set / pred_col 支持同一个非 latest 锚点时才覆盖。")
    lines.append("- 选择规则同时约束 val bad_top10、val all、val normal、val strong_steer 与 val observe_later_like，避免只对少数样本过拟合。")
    lines.append("")
    lines.append("## test bad_top10 决策收口")
    lines.append("")
    lines.append(decision.to_markdown(index=False) if len(decision) else "- 无决策结果。")
    lines.append("")
    lines.append("## val 选择出的 consensus 配置")
    lines.append("")
    cols = [
        "chosen_type",
        "deployable",
        "score_threshold",
        "margin_threshold",
        "min_votes",
        "vote_margin",
        "min_raw_sets",
        "val_bad_top10_delta_vs_latest",
        "val_all_delta_vs_latest",
        "val_normal_delta_vs_latest",
        "test_bad_top10_selected_rmse",
        "test_bad_top10_delta_vs_latest",
        "test_bad_top10_override_rate",
        "stable_pass",
    ]
    show_cols = [c for c in cols if c in chosen.columns]
    lines.append(chosen[show_cols].to_markdown(index=False) if len(chosen) else "- 没有选出 active 配置。")
    lines.append("")
    lines.append("## search top by val bad_top10")
    lines.append("")
    top_val = search[search["active"].astype(bool)].sort_values(["val_bad_top10_selected_rmse", "stability_penalty"]).head(20)
    top_cols = [
        "score_threshold",
        "margin_threshold",
        "min_votes",
        "vote_margin",
        "min_raw_sets",
        "val_bad_top10_selected_rmse",
        "val_bad_top10_delta_vs_latest",
        "val_all_delta_vs_latest",
        "val_normal_delta_vs_latest",
        "test_bad_top10_selected_rmse",
        "test_bad_top10_delta_vs_latest",
        "stable_pass",
    ]
    lines.append(top_val[[c for c in top_cols if c in top_val.columns]].to_markdown(index=False) if len(top_val) else "- 没有 active override。")
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    deploy = decision[decision["deployable"].astype(bool) & decision["source"].astype(str).str.startswith("val_")] if len(decision) else pd.DataFrame()
    if len(deploy) and bool(deploy["passes_fixed_latest"].any()):
        lines.append("- 至少一个 val 选择出的 stable consensus 策略低于 fixed wait-latest，可进入下一步复核。")
    else:
        lines.append("- val 选择出的 stable consensus 策略仍未低于 fixed wait-latest。")
    lines.append("- 如果 test-best diagnostic 低于 fixed wait-latest 但 val 策略不低于，说明生理一致性仍主要是事后可见信号，不是稳定可部署规则。")
    lines.append("- 这轮不是放弃生理，而是检验生理能否作为车辆不确定性下的稳定辅助证据。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figs:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v275_stable_bio_consensus_override_cn.md").write_text("\n".join(lines), encoding="utf-8")


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v274_candidates", V274_CANDIDATES),
        ("v274_decision", V274_DECISION),
        ("v274_guardrail", V274_GUARDRAIL),
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


def main() -> None:
    print("[v275] stable bio consensus override", flush=True)
    clean_out_dir()
    candidates, events, v274_decision, v274_guardrail = load_inputs()
    grid = make_grid(candidates)
    search = evaluate_grid(candidates, events, grid)
    chosen = choose_configs(search)
    selected = build_selected_for_choices(candidates, events, chosen)
    summary = summarize_selected(selected)
    decision = build_decision(summary, search, chosen, v274_decision)
    fig = plot_decision(decision)

    write_csv(grid, TABLES / "v275_consensus_grid.csv")
    write_csv(search, TABLES / "v275_consensus_search.csv")
    write_csv(chosen, TABLES / "v275_chosen_consensus_configs.csv")
    write_csv(selected, TABLES / "v275_selected_by_strategy.csv")
    write_csv(summary, TABLES / "v275_consensus_summary.csv")
    write_csv(decision, TABLES / "v275_decision_summary.csv")
    write_input_hashes()
    write_file_inventory()
    write_report(decision, chosen, search, [fig])
    write_file_inventory()
    zip_ok = make_zip()

    deploy = decision[decision["deployable"].astype(bool) & decision["source"].astype(str).str.startswith("val_")] if len(decision) else pd.DataFrame()
    best_deploy = float(pd.to_numeric(deploy["rmse"], errors="coerce").min()) if len(deploy) else math.nan
    test_diag = decision[decision["source"].astype(str).eq("test_best_consensus_diagnostic")] if len(decision) else pd.DataFrame()
    best_diag = float(pd.to_numeric(test_diag["rmse"], errors="coerce").min()) if len(test_diag) else math.nan
    guardrail = {
        "pass": bool(zip_ok and bool(v274_guardrail.get("pass", False)) and len(search) > 0 and len(decision) > 0),
        "zip_testzip": bool(zip_ok),
        "v274_guardrail_pass": bool(v274_guardrail.get("pass", False)),
        "event_n": int(events["event_uid"].nunique()),
        "candidate_rows": int(len(candidates)),
        "grid_rows": int(len(grid)),
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
        raise AssertionError("v275 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    print(f"[v275] report={REPORTS / 'v275_stable_bio_consensus_override_cn.md'}", flush=True)
    print(f"[v275] zip={ZIP_PATH}", flush=True)
    if len(decision):
        print(decision.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
