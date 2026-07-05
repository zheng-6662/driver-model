#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v282 physiology ambiguity route gate.

本脚本不再继续做 bio selector / reranker / reliability filter 的阈值微调，
而是把前面 v272-v281 暴露出来的关键问题单独审清楚：

1. 在车辆 topK 相似候选内部，生理距离是否真的能把真实 tail RMSE 更低的候选排前；
2. 这种信号在 train / val / test 是否方向一致；
3. 若只用 validation 选择 raw_set，test bad_top10 是否有真正可部署收益；
4. 如果连这个 route gate 都过不了，就说明当前生理特征层不适合作为 subject-disjoint
   差样本消歧主线，下一步应转向重新定义/重提取生理状态，而不是继续堆模型。

注意：
- bio_top1 是可部署近似：在 vehicle topK 中取生理距离最近候选；
- bio_top3 / bio_top5 是非部署上限：仍然用真实误差在 topM 内挑最好候选，只能说明上限；
- 所有 val-chosen 结果都只用 validation 选择 raw_set，再报告 test。
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
SCRIPTS = BASELINES / "scripts"

OUT = BASELINES / "v282_physio_ambiguity_route_gate_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v282_physio_ambiguity_route_gate_20260702_pack.zip"

V282_SCRIPT = SCRIPTS / "stage03_v282_physio_ambiguity_route_gate_20260702.py"
V272_DIAG = (
    BASELINES
    / "v272_physio_ambiguity_disambiguation_20260702"
    / "tables"
    / "v272_neighbor_rank_diagnostics_by_event.csv"
)
V272_SUMMARY = (
    BASELINES
    / "v272_physio_ambiguity_disambiguation_20260702"
    / "tables"
    / "v272_ambiguity_reduction_summary.csv"
)
V281_GUARDRAIL = (
    BASELINES
    / "v281_bio_top3_constrained_selector_20260702"
    / "logs"
    / "guardrail_check.json"
)

K_FOCUS = 40
FIXED_WAIT_LATEST_BADTOP10 = 0.695048
MIN_GROUP_N = 8


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
    """兼容 bool / 0-1 / 字符串三种 CSV 读入形态。"""

    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float).ne(0)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if not V272_DIAG.exists():
        raise FileNotFoundError(f"缺少 v272 事件级诊断表: {V272_DIAG}")
    if not V272_SUMMARY.exists():
        raise FileNotFoundError(f"缺少 v272 汇总表: {V272_SUMMARY}")

    diag = pd.read_csv(V272_DIAG)
    prev_summary = pd.read_csv(V272_SUMMARY)
    for col in ["bad_top10", "very_bad_top5", "early_best_after_400", "vehicle_ambiguous"]:
        if col in diag.columns:
            diag[col] = safe_bool_series(diag[col])

    numeric_cols = [
        "k",
        "keep0_rmse",
        "latest_rmse",
        "oracle_rmse",
        "vehicle_nearest_rmse",
        "vehicle_candidate_oracle_rmse",
        "vehicle_unique_delay_n",
        "vehicle_delay_std",
        "bio_best_candidate_rank",
        "bio_distance_rmse_rank_corr",
        "bio_top1_oracle_rmse",
        "bio_top3_oracle_rmse",
        "bio_top5_oracle_rmse",
    ]
    for col in numeric_cols:
        if col in diag.columns:
            diag[col] = pd.to_numeric(diag[col], errors="coerce")

    v281_guardrail: Dict[str, object] = {}
    if V281_GUARDRAIL.exists():
        v281_guardrail = json.loads(V281_GUARDRAIL.read_text(encoding="utf-8"))
    return diag, prev_summary, v281_guardrail


def event_groups(row: pd.Series) -> List[str]:
    """给每个事件打多重分析标签，重点看车辆歧义差样本。"""

    groups = ["all"]
    bad = bool(row.get("bad_top10", False))
    very_bad = bool(row.get("very_bad_top5", False))
    early = bool(row.get("early_best_after_400", False))
    ambiguous = bool(row.get("vehicle_ambiguous", False))
    if bad:
        groups.append("bad_top10")
    if very_bad:
        groups.append("very_bad_top5")
    if early:
        groups.append("early_best_after_400")
    if ambiguous:
        groups.append("vehicle_ambiguous")
    if bad and ambiguous:
        groups.append("bad_top10_vehicle_ambiguous")
    if bad and (not ambiguous):
        groups.append("bad_top10_vehicle_nonambiguous")
    if early and ambiguous:
        groups.append("early_ambiguous")
    return groups


def expand_groups(diag: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    focus = diag[diag["k"].eq(K_FOCUS)].copy()
    for _, row in focus.iterrows():
        base = row.to_dict()
        for group in event_groups(row):
            item = dict(base)
            item["event_group"] = group
            rows.append(item)
    return pd.DataFrame(rows)


def rmse_mean(s: pd.Series) -> float:
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    if len(arr) == 0:
        return math.nan
    return float(np.nanmean(arr))


def rate_mean(s: pd.Series) -> float:
    if len(s) == 0:
        return math.nan
    return float(pd.Series(s).fillna(False).astype(bool).mean())


def summarize_route_groups(expanded: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = ["raw_set", "split", "event_group"]
    for (raw_set, split, event_group), sub in expanded.groupby(group_cols, dropna=False):
        if len(sub) == 0:
            continue
        latest = pd.to_numeric(sub["latest_rmse"], errors="coerce").to_numpy(dtype=float)
        vehicle_oracle = pd.to_numeric(sub["vehicle_candidate_oracle_rmse"], errors="coerce").to_numpy(dtype=float)
        bio_top1 = pd.to_numeric(sub["bio_top1_oracle_rmse"], errors="coerce").to_numpy(dtype=float)
        bio_top3 = pd.to_numeric(sub["bio_top3_oracle_rmse"], errors="coerce").to_numpy(dtype=float)
        bio_top5 = pd.to_numeric(sub["bio_top5_oracle_rmse"], errors="coerce").to_numpy(dtype=float)
        gap = latest - vehicle_oracle
        valid_gap = np.isfinite(gap) & (gap > 1e-9)
        top3_closure = np.full_like(gap, np.nan, dtype=float)
        top1_closure = np.full_like(gap, np.nan, dtype=float)
        top3_closure[valid_gap] = (latest[valid_gap] - bio_top3[valid_gap]) / gap[valid_gap]
        top1_closure[valid_gap] = (latest[valid_gap] - bio_top1[valid_gap]) / gap[valid_gap]

        corr = pd.to_numeric(sub["bio_distance_rmse_rank_corr"], errors="coerce")
        rows.append(
            {
                "raw_set": raw_set,
                "split": split,
                "event_group": event_group,
                "n": int(sub["event_uid"].astype(str).nunique()),
                "latest_rmse_mean": rmse_mean(sub["latest_rmse"]),
                "keep0_rmse_mean": rmse_mean(sub["keep0_rmse"]),
                "vehicle_nearest_rmse_mean": rmse_mean(sub["vehicle_nearest_rmse"]),
                "vehicle_candidate_oracle_rmse_mean": rmse_mean(sub["vehicle_candidate_oracle_rmse"]),
                "bio_top1_rmse_mean": rmse_mean(sub["bio_top1_oracle_rmse"]),
                "bio_top3_oracle_rmse_mean": rmse_mean(sub["bio_top3_oracle_rmse"]),
                "bio_top5_oracle_rmse_mean": rmse_mean(sub["bio_top5_oracle_rmse"]),
                "vehicle_oracle_minus_latest_mean": rmse_mean(
                    sub["vehicle_candidate_oracle_rmse"] - sub["latest_rmse"]
                ),
                "vehicle_nearest_minus_latest_mean": rmse_mean(sub["vehicle_nearest_rmse"] - sub["latest_rmse"]),
                "bio_top1_minus_latest_mean": rmse_mean(sub["bio_top1_oracle_rmse"] - sub["latest_rmse"]),
                "bio_top3_minus_latest_mean": rmse_mean(sub["bio_top3_oracle_rmse"] - sub["latest_rmse"]),
                "bio_top5_minus_latest_mean": rmse_mean(sub["bio_top5_oracle_rmse"] - sub["latest_rmse"]),
                "bio_top1_beats_latest_rate": float(np.nanmean(bio_top1 < latest)),
                "bio_top3_beats_latest_rate": float(np.nanmean(bio_top3 < latest)),
                "bio_top5_beats_latest_rate": float(np.nanmean(bio_top5 < latest)),
                "bio_top1_gap_closure_mean": float(np.nanmean(top1_closure)),
                "bio_top3_gap_closure_mean": float(np.nanmean(top3_closure)),
                "bio_best_rank_mean": rmse_mean(sub["bio_best_candidate_rank"]),
                "bio_best_rank_median": float(np.nanmedian(pd.to_numeric(sub["bio_best_candidate_rank"], errors="coerce"))),
                "bio_best_in_top3_rate": rate_mean(sub["bio_best_in_top3"]),
                "bio_best_in_top5_rate": rate_mean(sub["bio_best_in_top5"]),
                "bio_corr_mean": rmse_mean(corr),
                "bio_corr_median": float(np.nanmedian(corr.to_numpy(dtype=float))),
                "bio_corr_positive_rate": float(np.nanmean(corr.to_numpy(dtype=float) > 0)),
                "bio_corr_gt_010_rate": float(np.nanmean(corr.to_numpy(dtype=float) > 0.10)),
                "vehicle_unique_delay_n_mean": rmse_mean(sub["vehicle_unique_delay_n"]),
                "vehicle_delay_std_mean": rmse_mean(sub["vehicle_delay_std"]),
            }
        )
    out = pd.DataFrame(rows)
    sort_cols = ["event_group", "split", "bio_top1_minus_latest_mean", "bio_top3_minus_latest_mean"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def val_chosen_generalization(summary: pd.DataFrame) -> pd.DataFrame:
    """只用 validation 选 raw_set，再报告 test。top1 是可部署，top3/top5 是上限。"""

    rows: List[Dict[str, object]] = []
    event_groups_to_check = [
        "all",
        "vehicle_ambiguous",
        "bad_top10",
        "bad_top10_vehicle_ambiguous",
        "early_best_after_400",
    ]
    methods = [
        ("bio_top1", "bio_top1_rmse_mean", "bio_top1_minus_latest_mean", True),
        ("bio_top3_oracle", "bio_top3_oracle_rmse_mean", "bio_top3_minus_latest_mean", False),
        ("bio_top5_oracle", "bio_top5_oracle_rmse_mean", "bio_top5_minus_latest_mean", False),
    ]
    for event_group in event_groups_to_check:
        for method, rmse_col, delta_col, deployable in methods:
            val = summary[
                summary["split"].eq("val")
                & summary["event_group"].eq(event_group)
                & summary["n"].ge(MIN_GROUP_N)
            ].copy()
            if val.empty:
                continue
            # val 只按该方法的 delta 选 raw_set；不看 test。
            val = val.sort_values([delta_col, "bio_corr_mean"], ascending=[True, False]).reset_index(drop=True)
            chosen = val.iloc[0]
            raw_set = str(chosen["raw_set"])
            test = summary[
                summary["split"].eq("test")
                & summary["event_group"].eq(event_group)
                & summary["raw_set"].astype(str).eq(raw_set)
            ]
            if test.empty:
                continue
            t = test.iloc[0]
            rows.append(
                {
                    "event_group": event_group,
                    "method": method,
                    "deployable": bool(deployable),
                    "val_chosen_raw_set": raw_set,
                    "val_n": int(chosen["n"]),
                    "test_n": int(t["n"]),
                    "val_latest_rmse_mean": float(chosen["latest_rmse_mean"]),
                    "val_method_rmse_mean": float(chosen[rmse_col]),
                    "val_delta_vs_latest_mean": float(chosen[delta_col]),
                    "val_corr_mean": float(chosen["bio_corr_mean"]),
                    "val_corr_positive_rate": float(chosen["bio_corr_positive_rate"]),
                    "test_latest_rmse_mean": float(t["latest_rmse_mean"]),
                    "test_method_rmse_mean": float(t[rmse_col]),
                    "test_delta_vs_latest_mean": float(t[delta_col]),
                    "test_corr_mean": float(t["bio_corr_mean"]),
                    "test_corr_positive_rate": float(t["bio_corr_positive_rate"]),
                    "test_gap_closure_mean": float(
                        t["bio_top1_gap_closure_mean"] if method == "bio_top1" else t["bio_top3_gap_closure_mean"]
                    ),
                    "test_passes_latest": bool(float(t[delta_col]) < -1e-9),
                    "val_and_test_same_direction_gain": bool(
                        float(chosen[delta_col]) < -1e-9 and float(t[delta_col]) < -1e-9
                    ),
                }
            )
    return pd.DataFrame(rows)


def split_consistency(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    methods = [
        ("bio_top1", "bio_top1_minus_latest_mean"),
        ("bio_top3_oracle", "bio_top3_minus_latest_mean"),
        ("bio_top5_oracle", "bio_top5_minus_latest_mean"),
    ]
    for (raw_set, event_group), sub in summary.groupby(["raw_set", "event_group"], dropna=False):
        by_split = {str(r["split"]): r for _, r in sub.iterrows()}
        if not {"train", "val", "test"}.issubset(by_split.keys()):
            continue
        for method, delta_col in methods:
            train_delta = float(by_split["train"][delta_col])
            val_delta = float(by_split["val"][delta_col])
            test_delta = float(by_split["test"][delta_col])
            rows.append(
                {
                    "raw_set": raw_set,
                    "event_group": event_group,
                    "method": method,
                    "train_n": int(by_split["train"]["n"]),
                    "val_n": int(by_split["val"]["n"]),
                    "test_n": int(by_split["test"]["n"]),
                    "train_delta_vs_latest_mean": train_delta,
                    "val_delta_vs_latest_mean": val_delta,
                    "test_delta_vs_latest_mean": test_delta,
                    "train_corr_mean": float(by_split["train"]["bio_corr_mean"]),
                    "val_corr_mean": float(by_split["val"]["bio_corr_mean"]),
                    "test_corr_mean": float(by_split["test"]["bio_corr_mean"]),
                    "all_splits_improve_latest": bool(
                        train_delta < -1e-9 and val_delta < -1e-9 and test_delta < -1e-9
                    ),
                    "val_test_improve_latest": bool(val_delta < -1e-9 and test_delta < -1e-9),
                    "corr_positive_all_splits": bool(
                        float(by_split["train"]["bio_corr_mean"]) > 0
                        and float(by_split["val"]["bio_corr_mean"]) > 0
                        and float(by_split["test"]["bio_corr_mean"]) > 0
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["event_group", "method", "val_delta_vs_latest_mean", "test_delta_vs_latest_mean"]
    )


def bad_ambiguous_event_audit(expanded: pd.DataFrame) -> pd.DataFrame:
    focus = expanded[
        expanded["split"].eq("test")
        & expanded["event_group"].eq("bad_top10_vehicle_ambiguous")
    ].copy()
    if focus.empty:
        return pd.DataFrame()
    cols = [
        "raw_set",
        "event_uid",
        "subject",
        "recording",
        "latest_rmse",
        "vehicle_nearest_rmse",
        "vehicle_candidate_oracle_rmse",
        "bio_top1_oracle_rmse",
        "bio_top3_oracle_rmse",
        "bio_top5_oracle_rmse",
        "bio_best_candidate_rank",
        "bio_best_in_top3",
        "bio_distance_rmse_rank_corr",
        "vehicle_unique_delay_n",
        "vehicle_delay_std",
        "early_best_after_400",
    ]
    out = focus[[c for c in cols if c in focus.columns]].copy()
    out["bio_top1_delta_vs_latest"] = out["bio_top1_oracle_rmse"] - out["latest_rmse"]
    out["bio_top3_delta_vs_latest"] = out["bio_top3_oracle_rmse"] - out["latest_rmse"]
    out["vehicle_oracle_delta_vs_latest"] = out["vehicle_candidate_oracle_rmse"] - out["latest_rmse"]
    return out.sort_values(["bio_top3_delta_vs_latest", "raw_set", "event_uid"]).reset_index(drop=True)


def build_decision(summary: pd.DataFrame, val_test: pd.DataFrame, v281_guardrail: Dict[str, object]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def get_val_test(event_group: str, method: str) -> pd.Series | None:
        sub = val_test[val_test["event_group"].eq(event_group) & val_test["method"].eq(method)]
        if sub.empty:
            return None
        return sub.iloc[0]

    top1_bad = get_val_test("bad_top10", "bio_top1")
    top1_bad_amb = get_val_test("bad_top10_vehicle_ambiguous", "bio_top1")
    top3_bad_amb = get_val_test("bad_top10_vehicle_ambiguous", "bio_top3_oracle")

    rows.append(
        {
            "check": "deployable_top1_val_chosen_bad_top10",
            "requirement": "validation 选出的生理 top1 在 test bad_top10 上低于 latest",
            "pass": bool(top1_bad is not None and top1_bad["test_delta_vs_latest_mean"] < -1e-9),
            "evidence": None if top1_bad is None else float(top1_bad["test_delta_vs_latest_mean"]),
            "deployable": True,
        }
    )
    rows.append(
        {
            "check": "deployable_top1_val_chosen_bad_ambiguous",
            "requirement": "validation 选出的生理 top1 在 test bad_top10_vehicle_ambiguous 上低于 latest",
            "pass": bool(top1_bad_amb is not None and top1_bad_amb["test_delta_vs_latest_mean"] < -1e-9),
            "evidence": None if top1_bad_amb is None else float(top1_bad_amb["test_delta_vs_latest_mean"]),
            "deployable": True,
        }
    )
    rows.append(
        {
            "check": "oracle_top3_val_test_same_direction_bad_ambiguous",
            "requirement": "非部署 top3 上限在 val/test 歧义差样本上同向改善",
            "pass": bool(top3_bad_amb is not None and top3_bad_amb["val_and_test_same_direction_gain"]),
            "evidence": None
            if top3_bad_amb is None
            else f"val={top3_bad_amb['val_delta_vs_latest_mean']:.6f}, test={top3_bad_amb['test_delta_vs_latest_mean']:.6f}",
            "deployable": False,
        }
    )
    bad_summary = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")]
    corr_ok = bool((bad_summary["bio_corr_mean"] > 0.05).any()) if not bad_summary.empty else False
    rows.append(
        {
            "check": "test_bad_top10_any_rawset_corr_gt_005",
            "requirement": "test bad_top10 至少一个 raw_set 的生理距离-真实误差排序相关均值 > 0.05",
            "pass": corr_ok,
            "evidence": None if bad_summary.empty else float(bad_summary["bio_corr_mean"].max()),
            "deployable": False,
        }
    )
    rows.append(
        {
            "check": "v281_selector_deployable_passes_fixed_latest",
            "requirement": "前序 v281 已证明可训练 selector 能超过 fixed latest",
            "pass": bool(v281_guardrail.get("best_deployable_passes_fixed_latest", False)),
            "evidence": v281_guardrail.get("best_val_chosen_deployable_test_badtop10"),
            "deployable": True,
        }
    )

    decision = pd.DataFrame(rows)
    decision["route_viable_now"] = bool(decision["pass"].all())
    return decision


def plot_badtop10_deltas(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v282_badtop10_val_test_bio_delta.png"
    data = summary[
        summary["event_group"].eq("bad_top10")
        & summary["split"].isin(["val", "test"])
    ].copy()
    data = data.sort_values(["raw_set", "split"])
    raw_sets = list(data["raw_set"].drop_duplicates())
    x = np.arange(len(raw_sets))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (split, col, label) in enumerate(
        [
            ("val", "bio_top1_minus_latest_mean", "val top1 deployable"),
            ("test", "bio_top1_minus_latest_mean", "test top1 deployable"),
            ("val", "bio_top3_minus_latest_mean", "val top3 oracle"),
            ("test", "bio_top3_minus_latest_mean", "test top3 oracle"),
        ]
    ):
        vals = []
        for raw in raw_sets:
            sub = data[data["raw_set"].eq(raw) & data["split"].eq(split)]
            vals.append(float(sub[col].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=label)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r).replace("_", "\n") for r in raw_sets], rotation=0, fontsize=8)
    ax.set_ylabel("RMSE delta vs latest, lower is better")
    ax.set_title("v282: physiology gain consistency on bad_top10")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_corr(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v282_bad_ambiguous_bio_rank_corr.png"
    data = summary[
        summary["event_group"].eq("bad_top10_vehicle_ambiguous")
        & summary["split"].isin(["train", "val", "test"])
    ].copy()
    raw_sets = list(data["raw_set"].drop_duplicates())
    x = np.arange(len(raw_sets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, split in enumerate(["train", "val", "test"]):
        vals = []
        for raw in raw_sets:
            sub = data[data["raw_set"].eq(raw) & data["split"].eq(split)]
            vals.append(float(sub["bio_corr_mean"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=split)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.05, color="tab:green", linestyle="--", linewidth=1, label="weak useful corr=0.05")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r).replace("_", "\n") for r in raw_sets], rotation=0, fontsize=8)
    ax.set_ylabel("mean rank corr: bio distance vs candidate RMSE")
    ax.set_title("v282: does physiology order good candidates in bad ambiguous cases?")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_val_chosen(val_test: pd.DataFrame) -> Path:
    path = FIGURES / "v282_val_chosen_test_delta.png"
    data = val_test[val_test["event_group"].isin(["bad_top10", "bad_top10_vehicle_ambiguous"])].copy()
    data["label"] = data["event_group"] + "\n" + data["method"]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:blue" if bool(v) else "tab:orange" for v in data["deployable"]]
    ax.bar(np.arange(len(data)), data["test_delta_vs_latest_mean"], color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["label"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("test RMSE delta vs latest, lower is better")
    ax.set_title("v282: validation-chosen physiology route generalization")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def table_to_md(df: pd.DataFrame, cols: List[str], n: int | None = None) -> str:
    use = df[[c for c in cols if c in df.columns]].copy()
    if n is not None:
        use = use.head(n)
    if use.empty:
        return "（无记录）"
    return use.to_markdown(index=False)


def write_report(
    summary: pd.DataFrame,
    val_test: pd.DataFrame,
    consistency: pd.DataFrame,
    decision: pd.DataFrame,
    fig_paths: List[Path],
    guardrail: Dict[str, object],
) -> Path:
    path = REPORTS / "v282_physio_ambiguity_route_gate_cn.md"

    bad_cols = [
        "raw_set",
        "split",
        "n",
        "latest_rmse_mean",
        "bio_top1_rmse_mean",
        "bio_top1_minus_latest_mean",
        "bio_top3_oracle_rmse_mean",
        "bio_top3_minus_latest_mean",
        "bio_corr_mean",
        "bio_corr_positive_rate",
        "bio_best_in_top3_rate",
    ]
    bad = summary[summary["event_group"].eq("bad_top10") & summary["split"].isin(["val", "test"])].sort_values(
        ["split", "bio_top1_minus_latest_mean"]
    )
    bad_amb = summary[
        summary["event_group"].eq("bad_top10_vehicle_ambiguous")
        & summary["split"].isin(["val", "test"])
    ].sort_values(["split", "bio_top1_minus_latest_mean"])

    lines: List[str] = []
    lines.append("# v282 生理歧义消解路线门控审计")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- 不继续同类 bio selector / reranker / reliability filter 微调。")
    lines.append("- 只审计一个基础问题：车辆锚点前相似、候选未来分叉时，生理距离是否稳定指向真实更好的候选。")
    lines.append("- `bio_top1` 是可部署近似；`bio_top3/top5 oracle` 只是上限，不可当部署结论。")
    lines.append("")
    lines.append("## route gate 判定")
    lines.append("")
    lines.append(table_to_md(decision, ["check", "requirement", "pass", "evidence", "deployable", "route_viable_now"]))
    lines.append("")
    lines.append("## validation 选择 raw_set 后的 test 泛化")
    lines.append("")
    lines.append(
        table_to_md(
            val_test,
            [
                "event_group",
                "method",
                "deployable",
                "val_chosen_raw_set",
                "val_delta_vs_latest_mean",
                "test_delta_vs_latest_mean",
                "test_corr_mean",
                "test_passes_latest",
                "val_and_test_same_direction_gain",
            ],
        )
    )
    lines.append("")
    lines.append("## bad_top10 分层结果")
    lines.append("")
    lines.append(table_to_md(bad, bad_cols))
    lines.append("")
    lines.append("## bad_top10 + vehicle_ambiguous 分层结果")
    lines.append("")
    lines.append(table_to_md(bad_amb, bad_cols))
    lines.append("")
    lines.append("## split 一致性摘要")
    lines.append("")
    show_cons = consistency[
        consistency["event_group"].isin(["bad_top10", "bad_top10_vehicle_ambiguous"])
        & consistency["method"].isin(["bio_top1", "bio_top3_oracle"])
    ].sort_values(["event_group", "method", "val_delta_vs_latest_mean"])
    lines.append(
        table_to_md(
            show_cons,
            [
                "raw_set",
                "event_group",
                "method",
                "train_delta_vs_latest_mean",
                "val_delta_vs_latest_mean",
                "test_delta_vs_latest_mean",
                "train_corr_mean",
                "val_corr_mean",
                "test_corr_mean",
                "val_test_improve_latest",
                "corr_positive_all_splits",
            ],
            n=30,
        )
    )
    lines.append("")
    lines.append("## 关键判读")
    lines.append("")
    route_viable = bool(decision["route_viable_now"].iloc[0]) if len(decision) else False
    if route_viable:
        lines.append("- route gate 通过：当前生理特征在车辆歧义差样本中同时具备可部署 top1 收益、稳定上限和排序相关性。")
        lines.append("- 下一步应把该 raw_set/分层条件升级为正式建模路线。")
    else:
        lines.append("- route gate 未通过：当前生理特征不能稳定解决 subject-disjoint 的车辆相似/未来分叉问题。")
        lines.append("- 如果继续坚持生理主线，下一步不应继续换 selector，而应重新定义生理状态特征，例如从 200Hz 连续层重提取事件前自主神经状态、响应相位、短时变化率和质量控制后的个体内变化。")
        lines.append("- 对预测效果主线而言，当前更可靠的路线仍是车辆多未来分布、不确定性建模或 anchor-aware 联合任务；生理只能作为新的特征重构分支重新进入。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in fig_paths:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    lines.append("")
    lines.append("## guardrail")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_input_hashes() -> None:
    rows = []
    for path in [V282_SCRIPT, V272_DIAG, V272_SUMMARY, V281_GUARDRAIL]:
        if path.exists():
            rows.append({"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size})
    write_csv(pd.DataFrame(rows), LOGS / "input_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size})
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
    print("[v282] 目的：审计当前生理特征是否足以成为车辆歧义差样本消歧主线。", flush=True)
    clean_out_dir()
    diag, prev_summary, v281_guardrail = load_inputs()

    expanded = expand_groups(diag)
    summary = summarize_route_groups(expanded)
    val_test = val_chosen_generalization(summary)
    consistency = split_consistency(summary)
    event_audit = bad_ambiguous_event_audit(expanded)
    decision = build_decision(summary, val_test, v281_guardrail)

    write_csv(expanded, TABLES / "v282_k40_event_group_long.csv")
    write_csv(summary, TABLES / "v282_route_group_summary.csv")
    write_csv(val_test, TABLES / "v282_val_chosen_generalization.csv")
    write_csv(consistency, TABLES / "v282_split_consistency.csv")
    write_csv(event_audit, TABLES / "v282_test_bad_ambiguous_event_audit.csv")
    write_csv(prev_summary, TABLES / "v282_v272_summary_snapshot.csv")
    write_csv(decision, TABLES / "v282_route_gate_decision.csv")

    fig_paths = [
        plot_badtop10_deltas(summary),
        plot_corr(summary),
        plot_val_chosen(val_test),
    ]

    route_viable = bool(decision["route_viable_now"].iloc[0]) if len(decision) else False
    guardrail = {
        "pass": True,
        "zip_testzip": None,
        "k_focus": K_FOCUS,
        "input_diag_rows": int(len(diag)),
        "expanded_rows": int(len(expanded)),
        "raw_set_n": int(diag["raw_set"].nunique()),
        "summary_rows": int(len(summary)),
        "val_chosen_rows": int(len(val_test)),
        "fixed_wait_latest_badtop10": FIXED_WAIT_LATEST_BADTOP10,
        "route_viable_now": route_viable,
        "deployable_top1_badtop10_pass": bool(
            decision.loc[decision["check"].eq("deployable_top1_val_chosen_bad_top10"), "pass"].iloc[0]
        ),
        "deployable_top1_bad_ambiguous_pass": bool(
            decision.loc[decision["check"].eq("deployable_top1_val_chosen_bad_ambiguous"), "pass"].iloc[0]
        ),
        "oracle_top3_bad_ambiguous_stable_pass": bool(
            decision.loc[decision["check"].eq("oracle_top3_val_test_same_direction_bad_ambiguous"), "pass"].iloc[0]
        ),
        "v281_selector_deployable_passes_fixed_latest": bool(
            v281_guardrail.get("best_deployable_passes_fixed_latest", False)
        ),
    }

    write_input_hashes()
    write_file_inventory()

    # 先写一版完整产物并打包，再把 ZIP 自检结果回写到报告和 guardrail，
    # 最后重新打包，保证包内报告与日志里的 zip_testzip 状态一致。
    guardrail["zip_testzip"] = False
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(summary, val_test, consistency, decision, fig_paths, guardrail)
    write_file_inventory()
    first_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(first_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(summary, val_test, consistency, decision, fig_paths, guardrail)
    write_file_inventory()
    second_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(second_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[v282] 完成。", flush=True)
    print(f"[v282] report={report}", flush=True)
    print(f"[v282] zip={ZIP_PATH}", flush=True)
    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
