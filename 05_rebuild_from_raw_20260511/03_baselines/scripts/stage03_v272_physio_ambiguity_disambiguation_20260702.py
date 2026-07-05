#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v272 physiology ambiguity disambiguation diagnostic.

v271 之后，生理表征/校准/重排仍没有形成可部署收益。本轮不再训练新预测器，
而是做一个更基础的诊断：

    在车辆历史 topK 相似候选里，生理距离是否能把真正更好的候选排到前面？

如果车辆相似候选库本身有 oracle headroom，但最佳候选在生理距离排序中经常排不到前几，
那么问题就不是“模型还不够强”，而是生理状态没有稳定消解车辆歧义。

边界：
- prototype 仍只来自 train split；
- query 只使用 0ms 已有车辆上下文和 v271 observation 前 calibrated raw physiology；
- 不训练新 trajectory model；
- test 只报告；val 只用于选择一个 raw_set/K 的 bio-top1 检索策略；
- candidate oracle、bio-top3/top5 oracle 只作为上界诊断，不作为可部署结果。
"""

from __future__ import annotations

import hashlib
import importlib.util
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

OUT = BASELINES / "v272_physio_ambiguity_disambiguation_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v272_physio_ambiguity_disambiguation_20260702_pack.zip"

V266_SCRIPT = BASELINES / "scripts" / "stage03_v266_vehicle_matched_bio_residual_prototype_20260702.py"
V271_EVENTS = BASELINES / "v271_calibrated_raw_physio_state_20260702" / "tables" / "v271_event_context_table.csv"
V271_FEATURE_AUDIT = BASELINES / "v271_calibrated_raw_physio_state_20260702" / "tables" / "v271_raw_feature_set_audit.csv"
V271_GUARDRAIL = BASELINES / "v271_calibrated_raw_physio_state_20260702" / "logs" / "guardrail_check.json"

SEED = 27202
K_VALUES = [5, 10, 20, 40]
BIO_TOP_M = [1, 3, 5]
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


V266 = import_module_from_path("stage03_v266_for_v272", V266_SCRIPT)


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


def pairwise_mean_sq_distance(query: np.ndarray, mat: np.ndarray) -> np.ndarray:
    diff = mat - query[None, :]
    return np.mean(diff * diff, axis=1)


def parse_feature_sets() -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    audit = pd.read_csv(V271_FEATURE_AUDIT, encoding="utf-8-sig", low_memory=False)
    feature_sets: Dict[str, List[str]] = {}
    for _, row in audit.iterrows():
        name = str(row["raw_set"])
        features = [x for x in str(row["features"]).split(";") if x]
        feature_sets[name] = features
    return feature_sets, audit


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, List[str]], pd.DataFrame]:
    cand, events, _merge_audit, veh_cols, _bio_cols = V266.load_candidate_and_events()
    feature_sets, feature_audit = parse_feature_sets()
    ctx = pd.read_csv(V271_EVENTS, encoding="utf-8-sig", low_memory=False)
    keep_cols = ["event_uid"]
    for cols in feature_sets.values():
        keep_cols.extend(cols)
    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in ctx.columns]))
    ctx = ctx[keep_cols].drop_duplicates("event_uid")
    events = events.merge(ctx, on="event_uid", how="left", validate="one_to_one")

    missing_sets = []
    for name, cols in feature_sets.items():
        present = [c for c in cols if c in events.columns]
        if not present:
            missing_sets.append(name)
        feature_sets[name] = present
    if missing_sets:
        raise RuntimeError(f"v271 event_context 缺少这些特征集：{missing_sets}")
    return cand, events, veh_cols, feature_sets, feature_audit


def rmse_at_delay(lookup: Dict[str, Dict[int, float]], event_uid: str, delay: int) -> Tuple[int, float]:
    return V266.rmse_at_delay(lookup, event_uid, int(delay))


def rank_corr_distance_rmse(distance: np.ndarray, rmse: np.ndarray) -> float:
    """Spearman-like rank correlation；正值表示生理越远 RMSE 越大，越有消歧价值。"""
    if len(distance) < 3:
        return math.nan
    s1 = pd.Series(distance).rank(method="average").to_numpy(dtype=float)
    s2 = pd.Series(rmse).rank(method="average").to_numpy(dtype=float)
    if np.nanstd(s1) < 1e-9 or np.nanstd(s2) < 1e-9:
        return math.nan
    return float(np.corrcoef(s1, s2)[0, 1])


def event_group_rows(row: pd.Series, ambiguous: bool) -> List[str]:
    groups = ["all"]
    if bool(row.get("bad_top10", False)):
        groups.append("bad_top10")
    if bool(row.get("very_bad_top5", False)):
        groups.append("very_bad_top5")
    if bool(row.get("early_best_after_400", False)):
        groups.append("early_best_after_400")
    if ambiguous:
        groups.append("vehicle_ambiguous")
    if bool(row.get("bad_top10", False)) and ambiguous:
        groups.append("bad_top10_vehicle_ambiguous")
    return groups


def build_query_neighbor_base(events: pd.DataFrame, veh_cols: List[str], max_k: int) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    train_mask = events["split"].astype(str).eq("train").to_numpy()
    veh_z, _med, _mean, _std = V266.fit_fill_scale(events[veh_cols].to_numpy(dtype=float), train_mask)
    train_idx = np.flatnonzero(train_mask)
    train_veh = veh_z[train_idx]
    event_uid = events["event_uid"].astype(str).to_numpy()
    neighbor_idx_by_query: Dict[int, np.ndarray] = {}
    for qi in range(len(events)):
        d_vehicle = pairwise_mean_sq_distance(veh_z[qi], train_veh)
        if train_mask[qi]:
            same = event_uid[train_idx] == event_uid[qi]
            d_vehicle = d_vehicle.copy()
            d_vehicle[same] = np.inf
        order_pos = np.argsort(d_vehicle, kind="mergesort")
        order_pos = order_pos[np.isfinite(d_vehicle[order_pos])][:max_k]
        neighbor_idx_by_query[qi] = train_idx[order_pos]
    return neighbor_idx_by_query, veh_z, train_mask


def analyze_feature_set(
    events: pd.DataFrame,
    lookup: Dict[str, Dict[int, float]],
    feature_name: str,
    feature_cols: List[str],
    neighbor_idx_by_query: Dict[int, np.ndarray],
    train_mask: np.ndarray,
) -> pd.DataFrame:
    bio_z, _med, _mean, _std = V266.fit_fill_scale(events[feature_cols].to_numpy(dtype=float), train_mask)
    rows: List[Dict[str, object]] = []
    event_uid = events["event_uid"].astype(str).to_numpy()
    oracle_delay = pd.to_numeric(events["oracle_delay_ms"], errors="coerce").to_numpy(dtype=int)
    latest_rmse = pd.to_numeric(events["latest_tail_rmse_v241"], errors="coerce").to_numpy(dtype=float)
    keep0_rmse = pd.to_numeric(events["keep0_tail_rmse_v241"], errors="coerce").to_numpy(dtype=float)
    oracle_rmse = pd.to_numeric(events["oracle_tail_rmse_v241"], errors="coerce").to_numpy(dtype=float)

    for qi, train_neighbors in neighbor_idx_by_query.items():
        q_uid = event_uid[qi]
        if len(train_neighbors) == 0:
            continue
        bio_distance_all = pairwise_mean_sq_distance(bio_z[qi], bio_z[train_neighbors])
        proto_delay_all = oracle_delay[train_neighbors]
        mapped_delay_all: List[int] = []
        mapped_rmse_all: List[float] = []
        for delay in proto_delay_all:
            mapped_delay, rmse = rmse_at_delay(lookup, q_uid, int(delay))
            mapped_delay_all.append(int(mapped_delay))
            mapped_rmse_all.append(float(rmse))
        mapped_delay_arr = np.asarray(mapped_delay_all, dtype=int)
        mapped_rmse_arr = np.asarray(mapped_rmse_all, dtype=float)

        for k in K_VALUES:
            use_n = min(k, len(train_neighbors))
            if use_n <= 0:
                continue
            rmse_k = mapped_rmse_arr[:use_n]
            delay_k = mapped_delay_arr[:use_n]
            bio_d_k = bio_distance_all[:use_n]
            bio_order = np.argsort(bio_d_k, kind="mergesort")
            best_vehicle_pos = int(np.nanargmin(rmse_k))
            best_rank_by_vehicle = best_vehicle_pos + 1
            best_pos_in_bio_order = int(np.flatnonzero(bio_order == best_vehicle_pos)[0]) + 1

            row = {
                "raw_set": feature_name,
                "feature_n": int(len(feature_cols)),
                "k": int(k),
                "event_uid": q_uid,
                "split": str(events.iloc[qi]["split"]),
                "subject": str(events.iloc[qi]["subject"]),
                "recording": str(events.iloc[qi]["recording"]),
                "bad_top10": bool(events.iloc[qi].get("bad_top10", False)),
                "very_bad_top5": bool(events.iloc[qi].get("very_bad_top5", False)),
                "early_best_after_400": bool(events.iloc[qi].get("early_best_after_400", False)),
                "keep0_rmse": float(keep0_rmse[qi]),
                "latest_rmse": float(latest_rmse[qi]),
                "oracle_rmse": float(oracle_rmse[qi]),
                "vehicle_nearest_delay_ms": int(delay_k[0]),
                "vehicle_nearest_rmse": float(rmse_k[0]),
                "vehicle_candidate_oracle_delay_ms": int(delay_k[best_vehicle_pos]),
                "vehicle_candidate_oracle_rmse": float(rmse_k[best_vehicle_pos]),
                "vehicle_candidate_oracle_rank": int(best_rank_by_vehicle),
                "vehicle_unique_delay_n": int(pd.Series(delay_k).nunique()),
                "vehicle_delay_std": float(np.nanstd(delay_k)),
                "vehicle_ambiguous": bool((pd.Series(delay_k).nunique() >= 3) and ((rmse_k[0] - np.nanmin(rmse_k)) >= 0.05)),
                "bio_best_candidate_rank": int(best_pos_in_bio_order),
                "bio_best_in_top3": bool(best_pos_in_bio_order <= 3),
                "bio_best_in_top5": bool(best_pos_in_bio_order <= 5),
                "bio_distance_rmse_rank_corr": rank_corr_distance_rmse(bio_d_k, rmse_k),
            }
            for m in BIO_TOP_M:
                m_use = min(m, use_n)
                top_pos = bio_order[:m_use]
                best_m_pos = int(top_pos[np.nanargmin(rmse_k[top_pos])])
                row[f"bio_top{m}_nearest_delay_ms"] = int(delay_k[bio_order[0]]) if m == 1 else int(delay_k[best_m_pos])
                row[f"bio_top{m}_oracle_rmse"] = float(np.nanmin(rmse_k[top_pos]))
                row[f"bio_top{m}_contains_vehicle_best"] = bool(best_vehicle_pos in set(top_pos.tolist()))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_groups(diag: pd.DataFrame) -> pd.DataFrame:
    expanded_rows: List[Dict[str, object]] = []
    metric_cols = [
        "keep0_rmse",
        "latest_rmse",
        "oracle_rmse",
        "vehicle_nearest_rmse",
        "vehicle_candidate_oracle_rmse",
        "bio_top1_oracle_rmse",
        "bio_top3_oracle_rmse",
        "bio_top5_oracle_rmse",
        "bio_best_candidate_rank",
        "bio_distance_rmse_rank_corr",
        "vehicle_unique_delay_n",
        "vehicle_delay_std",
    ]
    for _, row in diag.iterrows():
        for group in event_group_rows(row, bool(row["vehicle_ambiguous"])):
            out = {"event_group": group}
            out.update(row.to_dict())
            expanded_rows.append(out)
    expanded = pd.DataFrame(expanded_rows)
    rows: List[Dict[str, object]] = []
    for (raw_set, k, split, event_group), g in expanded.groupby(["raw_set", "k", "split", "event_group"], sort=False):
        row: Dict[str, object] = {
            "raw_set": raw_set,
            "k": int(k),
            "split": split,
            "event_group": event_group,
            "n": int(len(g)),
            "vehicle_ambiguous_rate": float(g["vehicle_ambiguous"].astype(bool).mean()),
            "bio_best_in_top3_rate": float(g["bio_best_in_top3"].astype(bool).mean()),
            "bio_best_in_top5_rate": float(g["bio_best_in_top5"].astype(bool).mean()),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(pd.to_numeric(g[col], errors="coerce").mean())
            row[f"{col}_median"] = float(pd.to_numeric(g[col], errors="coerce").median())
        row["bio_top1_minus_vehicle_nearest_mean"] = row["bio_top1_oracle_rmse_mean"] - row["vehicle_nearest_rmse_mean"]
        row["bio_top3_minus_vehicle_oracle_mean"] = row["bio_top3_oracle_rmse_mean"] - row["vehicle_candidate_oracle_rmse_mean"]
        row["vehicle_oracle_minus_latest_mean"] = row["vehicle_candidate_oracle_rmse_mean"] - row["latest_rmse_mean"]
        row["bio_top1_minus_latest_mean"] = row["bio_top1_oracle_rmse_mean"] - row["latest_rmse_mean"]
        row["bio_top3_minus_latest_mean"] = row["bio_top3_oracle_rmse_mean"] - row["latest_rmse_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_decision(summary: pd.DataFrame) -> pd.DataFrame:
    test_bad = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["k"].eq(40)
    ].copy()
    val_bad = summary[
        summary["split"].eq("val")
        & summary["event_group"].eq("bad_top10")
        & summary["k"].eq(40)
    ].copy()
    rows: List[Dict[str, object]] = []
    if test_bad.empty:
        return pd.DataFrame()

    first = test_bad.iloc[0]
    rows.extend(
        [
            {"source": "baseline", "label": "policy_keep_0ms_anchor", "rmse": float(first["keep0_rmse_mean"]), "deployable": True},
            {"source": "baseline", "label": "policy_wait_to_latest_anchor", "rmse": float(first["latest_rmse_mean"]), "deployable": True},
            {"source": "oracle", "label": "oracle_best_anchor_upper_bound", "rmse": float(first["oracle_rmse_mean"]), "deployable": False},
            {"source": "vehicle", "label": "vehicle_nearest_train_prototype_k40", "rmse": float(first["vehicle_nearest_rmse_mean"]), "deployable": True},
            {"source": "vehicle_oracle", "label": "vehicle_candidate_oracle_k40", "rmse": float(first["vehicle_candidate_oracle_rmse_mean"]), "deployable": False},
        ]
    )

    if len(val_bad):
        val_top1 = val_bad.sort_values(["bio_top1_oracle_rmse_mean", "raw_set"]).iloc[0]
        raw_top1 = str(val_top1["raw_set"])
        test_top1 = test_bad[test_bad["raw_set"].astype(str).eq(raw_top1)].iloc[0]
        rows.append(
            {
                "source": "bio_top1_val_chosen",
                "label": f"{raw_top1}:bio_nearest_within_vehicle_k40",
                "rmse": float(test_top1["bio_top1_oracle_rmse_mean"]),
                "deployable": True,
            }
        )
        val_top3 = val_bad.sort_values(["bio_top3_oracle_rmse_mean", "raw_set"]).iloc[0]
        raw_top3 = str(val_top3["raw_set"])
        test_top3 = test_bad[test_bad["raw_set"].astype(str).eq(raw_top3)].iloc[0]
        rows.append(
            {
                "source": "bio_top3_oracle_val_chosen",
                "label": f"{raw_top3}:bio_top3_oracle_within_vehicle_k40",
                "rmse": float(test_top3["bio_top3_oracle_rmse_mean"]),
                "deployable": False,
            }
        )

    best_top1 = test_bad.sort_values(["bio_top1_oracle_rmse_mean", "raw_set"]).iloc[0]
    rows.append(
        {
            "source": "bio_top1_test_best_diagnostic",
            "label": f"{best_top1['raw_set']}:bio_nearest_within_vehicle_k40",
            "rmse": float(best_top1["bio_top1_oracle_rmse_mean"]),
            "deployable": False,
        }
    )
    best_top3 = test_bad.sort_values(["bio_top3_oracle_rmse_mean", "raw_set"]).iloc[0]
    rows.append(
        {
            "source": "bio_top3_oracle_test_best",
            "label": f"{best_top3['raw_set']}:bio_top3_oracle_within_vehicle_k40",
            "rmse": float(best_top3["bio_top3_oracle_rmse_mean"]),
            "deployable": False,
        }
    )

    out = pd.DataFrame(rows)
    out["delta_vs_fixed_latest"] = out["rmse"] - FIXED_WAIT_LATEST_BADTOP10
    out["passes_fixed_latest"] = out["rmse"] < FIXED_WAIT_LATEST_BADTOP10
    return out


def plot_decision(decision: pd.DataFrame) -> Path:
    path = FIGURES / "v272_test_badtop10_ambiguity_decision.png"
    if decision.empty:
        return path
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    x = np.arange(len(decision))
    colors = ["#4C78A8" if bool(v) else "#9C755F" for v in decision["deployable"]]
    ax.bar(x, decision["rmse"], color=colors)
    ax.axhline(FIXED_WAIT_LATEST_BADTOP10, color="#E15759", linestyle="--", linewidth=1.2, label="fixed wait-latest")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in decision["source"]], fontsize=8)
    ax.set_ylabel("test bad_top10 tail RMSE")
    ax.set_title("v272: can physiology rank good candidates inside vehicle-similar neighborhood?")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_rank_capture(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v272_test_badtop10_bio_rank_capture.png"
    test_bad = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["k"].eq(40)
    ].copy()
    if test_bad.empty:
        return path
    test_bad = test_bad.sort_values("bio_best_in_top3_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    x = np.arange(len(test_bad))
    ax.bar(x - 0.18, test_bad["bio_best_in_top3_rate"], width=0.36, label="best in bio top3")
    ax.bar(x + 0.18, test_bad["bio_best_in_top5_rate"], width=0.36, label="best in bio top5")
    ax.set_xticks(x)
    ax.set_xticklabels(test_bad["raw_set"], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("capture rate")
    ax.set_title("v272: rank of vehicle-candidate oracle by physiology distance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v266_script", V266_SCRIPT),
        ("v271_events", V271_EVENTS),
        ("v271_feature_audit", V271_FEATURE_AUDIT),
        ("v271_guardrail", V271_GUARDRAIL),
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


def write_report(summary: pd.DataFrame, decision: pd.DataFrame, feature_audit: pd.DataFrame, figs: List[Path]) -> None:
    lines: List[str] = []
    lines.append("# v272 physiology ambiguity disambiguation")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- v266/v267/v270/v271 都显示候选库有 headroom，但可部署选择器选不准。")
    lines.append("- v272 不再训练新模型，而是检查：在车辆 topK 相似候选内部，生理距离能不能把真正更好的候选排到前面。")
    lines.append("- 如果生理 top1/top3 排序不能接近 vehicle candidate oracle，说明生理不是稳定消歧信号。")
    lines.append("")
    lines.append("## 使用的 v271 特征集")
    lines.append("")
    cols = ["raw_set", "feature_n", "behavior_eta_max_mean", "identity_eta_max_mean", "identity_to_behavior_ratio_median"]
    lines.append(feature_audit[[c for c in cols if c in feature_audit.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## test bad_top10 决策收口")
    lines.append("")
    lines.append(decision.to_markdown(index=False) if len(decision) else "- 无可用结果。")
    lines.append("")
    lines.append("## test bad_top10 K=40 生理排序诊断")
    lines.append("")
    test_bad = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["k"].eq(40)
    ].sort_values("bio_top1_oracle_rmse_mean")
    show = [
        "raw_set",
        "n",
        "vehicle_nearest_rmse_mean",
        "vehicle_candidate_oracle_rmse_mean",
        "bio_top1_oracle_rmse_mean",
        "bio_top3_oracle_rmse_mean",
        "bio_best_candidate_rank_mean",
        "bio_best_in_top3_rate",
        "bio_best_in_top5_rate",
        "bio_distance_rmse_rank_corr_mean",
    ]
    lines.append(test_bad[[c for c in show if c in test_bad.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    if len(test_bad):
        best_top1 = test_bad.iloc[0]
        best_top3 = test_bad.sort_values("bio_top3_oracle_rmse_mean").iloc[0]
        lines.append(f"- 最好的 bio top1 检索为 `{best_top1['raw_set']}`，test bad_top10 RMSE `{float(best_top1['bio_top1_oracle_rmse_mean']):.4f}`。")
        lines.append(f"- 最好的 bio top3 oracle 为 `{best_top3['raw_set']}`，test bad_top10 RMSE `{float(best_top3['bio_top3_oracle_rmse_mean']):.4f}`。")
        lines.append(f"- vehicle candidate oracle 为 `{float(best_top1['vehicle_candidate_oracle_rmse_mean']):.4f}`，fixed wait-latest 为 `{FIXED_WAIT_LATEST_BADTOP10:.4f}`。")
        if float(best_top1["bio_top1_oracle_rmse_mean"]) < FIXED_WAIT_LATEST_BADTOP10:
            lines.append("- 生理最近邻 top1 已低于 fixed wait-latest，说明生理距离可能有直接可部署价值。")
        else:
            lines.append("- 生理最近邻 top1 未低于 fixed wait-latest，说明直接用生理距离不能完成 goal。")
        if float(best_top3["bio_top3_oracle_rmse_mean"]) < FIXED_WAIT_LATEST_BADTOP10:
            lines.append("- bio top3 oracle 低于 fixed wait-latest，说明若另有强选择器，生理邻域内仍有少量上界。")
        else:
            lines.append("- bio top3 oracle 也未低于 fixed wait-latest，说明生理邻域本身不足以承载本质改善。")
    lines.append("- 本实验的核心不是提交新模型，而是判断生理是否真的能在车辆相似样本之间消歧。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figs:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v272_physio_ambiguity_disambiguation_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v272] physiology ambiguity disambiguation diagnostic", flush=True)
    clean_out_dir()
    cand, events, veh_cols, feature_sets, feature_audit = load_data()
    lookup = V266.candidate_rmse_lookup(cand)
    max_k = max(K_VALUES)
    neighbor_idx_by_query, _veh_z, train_mask = build_query_neighbor_base(events, veh_cols, max_k)

    diag_parts: List[pd.DataFrame] = []
    for raw_set, cols in feature_sets.items():
        print(f"[v272] analyze raw_set={raw_set} feature_n={len(cols)}", flush=True)
        diag = analyze_feature_set(events, lookup, raw_set, cols, neighbor_idx_by_query, train_mask)
        diag_parts.append(diag)
    diag_all = pd.concat(diag_parts, ignore_index=True)
    summary = summarize_by_groups(diag_all)
    decision = build_decision(summary)
    fig1 = plot_decision(decision)
    fig2 = plot_rank_capture(summary)

    feature_rows = []
    for raw_set, cols in feature_sets.items():
        sub = feature_audit[feature_audit["raw_set"].astype(str).eq(raw_set)]
        row = {"raw_set": raw_set, "feature_n_used": int(len(cols)), "missing_feature_n": 0}
        if len(sub):
            row.update(sub.iloc[0].to_dict())
        feature_rows.append(row)
    feature_input_audit = pd.DataFrame(feature_rows)

    write_csv(diag_all, TABLES / "v272_neighbor_rank_diagnostics_by_event.csv")
    write_csv(summary, TABLES / "v272_ambiguity_reduction_summary.csv")
    write_csv(decision, TABLES / "v272_decision_summary.csv")
    write_csv(feature_input_audit, TABLES / "v272_feature_set_input_audit.csv")
    write_input_hashes()
    write_file_inventory()
    write_report(summary, decision, feature_input_audit, [fig1, fig2])
    write_file_inventory()
    zip_ok = make_zip()

    v271_guard = json.loads(V271_GUARDRAIL.read_text(encoding="utf-8")) if V271_GUARDRAIL.exists() else {}
    test_bad = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["k"].eq(40)
    ]
    best_top1 = float(test_bad["bio_top1_oracle_rmse_mean"].min()) if len(test_bad) else math.nan
    best_top3 = float(test_bad["bio_top3_oracle_rmse_mean"].min()) if len(test_bad) else math.nan
    guardrail = {
        "pass": bool(zip_ok and bool(v271_guard.get("pass", False)) and len(diag_all) > 0 and len(summary) > 0),
        "zip_testzip": bool(zip_ok),
        "v271_guardrail_pass": bool(v271_guard.get("pass", False)),
        "event_n": int(events["event_uid"].nunique()),
        "raw_set_n": int(len(feature_sets)),
        "diagnostic_row_n": int(len(diag_all)),
        "summary_row_n": int(len(summary)),
        "best_bio_top1_test_badtop10_k40": best_top1,
        "best_bio_top3_oracle_test_badtop10_k40": best_top3,
        "fixed_wait_latest_badtop10": float(FIXED_WAIT_LATEST_BADTOP10),
        "bio_top1_passes_fixed_latest": bool(np.isfinite(best_top1) and best_top1 < FIXED_WAIT_LATEST_BADTOP10),
        "bio_top3_oracle_passes_fixed_latest": bool(np.isfinite(best_top3) and best_top3 < FIXED_WAIT_LATEST_BADTOP10),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v272 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    print(f"[v272] report={REPORTS / 'v272_physio_ambiguity_disambiguation_cn.md'}", flush=True)
    print(f"[v272] zip={ZIP_PATH}", flush=True)
    if len(decision):
        print(decision.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
