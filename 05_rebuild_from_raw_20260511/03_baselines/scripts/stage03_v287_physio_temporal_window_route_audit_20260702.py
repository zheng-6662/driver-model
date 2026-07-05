#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v287 physiology temporal-window route audit.

本轮目的：
- v285 把 cleaned 200Hz 连续信号重构成 1146 个 raw shape-state 特征，但整体
  route gate 失败；
- v287 不再重新抽波形，而是把 v285 特征按时间窗口、信号族、特征类型拆开，
  检查是否存在“某个事件前窗口 / 某个生理信号族”被整体特征混合稀释；
- 如果没有任何窗口或信号族通过 deployable route gate，就进一步说明问题不是
  简单的窗口选择，而是当前生理层本身缺少稳定可部署的差样本消歧信号。

边界：
- 只使用 v285 已经 causal 抽取的 observation_s 前特征；
- 特征分组和排序只使用 v285 train-only screen；
- route gate 复用 v284/v285 的 vehicle top40 候选池和 v272 差样本标签；
- test 只报告，不用于选择窗口、信号族、阈值或策略。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
SCRIPTS = BASELINES / "scripts"

V284_SCRIPT = SCRIPTS / "stage03_v284_dynamic_low_identity_physio_route_gate_20260702.py"
V285_FEATURES = BASELINES / "v285_raw200_shape_state_route_gate_20260702" / "tables" / "v285_raw200_shape_state_features.csv"
V285_SCREEN = BASELINES / "v285_raw200_shape_state_route_gate_20260702" / "tables" / "v285_train_only_feature_screen.csv"
V285_GUARDRAIL = BASELINES / "v285_raw200_shape_state_route_gate_20260702" / "logs" / "guardrail_check.json"

OUT = BASELINES / "v287_physio_temporal_window_route_audit_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v287_physio_temporal_window_route_audit_20260702_pack.zip"

SEED = 28702
MIN_FEATURES = 8
MAX_FEATURES_PER_SET = 32

WINDOW_ORDER = [
    "pre30_pre20",
    "pre20_pre10",
    "pre10_pre5",
    "pre5_pre2",
    "pre2_0",
    "pre1_0",
    "pre5_0",
    "pre10_0",
    "delta_pre2_0_minus_pre30_pre20",
    "delta_pre2_0_minus_pre20_pre10",
    "delta_pre2_0_minus_pre10_pre5",
    "delta_pre2_0_minus_pre5_pre2",
]

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    """按路径导入 v284 的 route gate 工具。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V284 = import_module_from_path("stage03_v284_for_v287", V284_SCRIPT)


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


def load_v285_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取 v285 causal raw-shape 特征和 train-only screen。"""

    if not V285_FEATURES.exists():
        raise FileNotFoundError(f"缺少 v285 特征表：{V285_FEATURES}")
    if not V285_SCREEN.exists():
        raise FileNotFoundError(f"缺少 v285 train-only screen：{V285_SCREEN}")
    features = pd.read_csv(V285_FEATURES, encoding="utf-8-sig", low_memory=False)
    screen = pd.read_csv(V285_SCREEN, encoding="utf-8-sig", low_memory=False)
    if features["event_uid"].duplicated().any():
        raise RuntimeError("v285 feature 表存在重复 event_uid")
    for col in ["behavior_eta_max", "bad_eta_max", "identity_eta_max", "behavior_identity_score", "bad_identity_score", "finite_rate_train"]:
        if col in screen.columns:
            screen[col] = pd.to_numeric(screen[col], errors="coerce")
    return features, screen


def infer_window(feature: str) -> str:
    """从 v285 特征名中解析时间窗口或 delta 窗口。"""

    name = str(feature)
    for win in WINDOW_ORDER:
        if f"bio285_{win}_" in name or f"bio285_{win}" in name:
            return win
    match = re.search(r"bio285_(pre\d+_pre\d+|pre\d+_0|pre\d+_\d+)_", name)
    if match:
        return match.group(1)
    return "unknown"


def enrich_screen(screen: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """补充窗口、可用性和安全过滤字段。"""

    out = screen.copy()
    out["window_group"] = out["feature"].map(infer_window)
    if "feature_category" not in out.columns:
        out["feature_category"] = "unknown"
    if "signal_family" not in out.columns:
        out["signal_family"] = "unknown"
    numeric_cols = [c for c in features.columns if c.startswith("bio285_") and pd.api.types.is_numeric_dtype(features[c])]
    available = set(numeric_cols)
    out = out[out["feature"].astype(str).isin(available)].copy()
    out = out[out["finite_rate_train"].ge(0.70)].copy()
    out["rank_score"] = (
        out["behavior_identity_score"].fillna(0.0)
        + 0.5 * out["bad_identity_score"].fillna(0.0)
        + 0.1 * out["behavior_eta_max"].fillna(0.0)
    )
    return out.reset_index(drop=True)


def top_features(df: pd.DataFrame, n: int = MAX_FEATURES_PER_SET) -> List[str]:
    """用 train-only screen 选择 feature set 内的特征。"""

    if df.empty:
        return []
    return (
        df.sort_values(["rank_score", "behavior_identity_score", "bad_identity_score"], ascending=False)["feature"]
        .drop_duplicates()
        .head(n)
        .astype(str)
        .tolist()
    )


def build_feature_sets(screen: pd.DataFrame) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """按时间窗口、信号族、特征类型和窗口×信号组合构造候选特征集。"""

    sets: Dict[str, List[str]] = {}
    audit_rows: List[Dict[str, object]] = []

    def add_set(name: str, group_type: str, group_value: str, df: pd.DataFrame, n: int = MAX_FEATURES_PER_SET) -> None:
        cols = top_features(df, n=n)
        if len(cols) < MIN_FEATURES:
            return
        sets[name] = cols
        audit_rows.append(
            {
                "feature_set": name,
                "group_type": group_type,
                "group_value": group_value,
                "candidate_feature_n": int(len(df)),
                "feature_n": int(len(cols)),
                "rank_score_max": float(df["rank_score"].max()),
                "behavior_eta_max": float(df["behavior_eta_max"].max()),
                "bad_eta_max": float(df["bad_eta_max"].max()),
                "identity_eta_median": float(df["identity_eta_max"].median()),
            }
        )

    for win in WINDOW_ORDER:
        sub = screen[screen["window_group"].eq(win)]
        add_set(f"win_{win}_top{MAX_FEATURES_PER_SET}", "window", win, sub)

    for sig in sorted(screen["signal_family"].dropna().astype(str).unique()):
        if sig in {"unknown", "other"}:
            continue
        sub = screen[screen["signal_family"].astype(str).eq(sig)]
        add_set(f"signal_{sig}_top{MAX_FEATURES_PER_SET}", "signal", sig, sub)

    for cat in sorted(screen["feature_category"].dropna().astype(str).unique()):
        sub = screen[screen["feature_category"].astype(str).eq(cat)]
        add_set(f"category_{cat}_top{MAX_FEATURES_PER_SET}", "category", cat, sub)

    # 组合特征集只保留 train-only rank_score 最靠前的窗口×信号组合，避免测试后验选择。
    combo_scores = (
        screen.groupby(["window_group", "signal_family"], as_index=False)
        .agg(
            feature_n=("feature", "count"),
            rank_score_max=("rank_score", "max"),
            behavior_eta_max=("behavior_eta_max", "max"),
            bad_eta_max=("bad_eta_max", "max"),
        )
        .sort_values(["rank_score_max", "behavior_eta_max"], ascending=False)
    )
    combo_scores = combo_scores[
        combo_scores["feature_n"].ge(MIN_FEATURES)
        & ~combo_scores["window_group"].eq("unknown")
        & ~combo_scores["signal_family"].isin(["unknown", "other"])
    ].head(24)
    for row in combo_scores.itertuples(index=False):
        win = str(row.window_group)
        sig = str(row.signal_family)
        sub = screen[screen["window_group"].eq(win) & screen["signal_family"].astype(str).eq(sig)]
        add_set(f"combo_{win}_{sig}_top16", "window_signal", f"{win}|{sig}", sub, n=16)

    return sets, pd.DataFrame(audit_rows)


def summarize_group_winners(summary: pd.DataFrame, feature_audit: pd.DataFrame) -> pd.DataFrame:
    """整理 test bad_top10 和歧义差样本上每个 group_type 的最好结果。"""

    focus = summary[
        summary["split"].eq("test")
        & summary["event_group"].isin(["bad_top10", "bad_top10_vehicle_ambiguous"])
    ].merge(feature_audit[["feature_set", "group_type", "group_value"]], on="feature_set", how="left")
    rows: List[Dict[str, object]] = []
    for (event_group, group_type), sub in focus.groupby(["event_group", "group_type"], dropna=False):
        if sub.empty:
            continue
        top1 = sub.sort_values("bio_top1_minus_latest_mean").iloc[0]
        corr = sub.sort_values("bio_corr_mean", ascending=False).iloc[0]
        rows.append(
            {
                "event_group": event_group,
                "group_type": group_type,
                "best_top1_feature_set": str(top1["feature_set"]),
                "best_top1_group_value": str(top1["group_value"]),
                "best_top1_delta": float(top1["bio_top1_minus_latest_mean"]),
                "best_top3_delta": float(top1["bio_top3_minus_latest_mean"]),
                "best_corr_feature_set": str(corr["feature_set"]),
                "best_corr_group_value": str(corr["group_value"]),
                "best_corr_mean": float(corr["bio_corr_mean"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["event_group", "best_top1_delta"])


def table_to_md(df: pd.DataFrame, cols: List[str] | None = None, max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "_空表_"
    show = df.copy()
    if cols is not None:
        show = show[[c for c in cols if c in show.columns]]
    return show.head(max_rows).to_markdown(index=False)


def plot_window_delta(summary: pd.DataFrame, feature_audit: pd.DataFrame) -> Path:
    """画窗口级 feature set 在 test bad_top10 上的 top1 delta。"""

    path = FIGURES / "v287_window_badtop10_top1_delta.png"
    data = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
    ].merge(feature_audit, on="feature_set", how="left")
    data = data[data["group_type"].eq("window")].copy()
    if data.empty:
        return path
    order = [w for w in WINDOW_ORDER if w in set(data["group_value"])]
    data["group_value"] = pd.Categorical(data["group_value"], categories=order, ordered=True)
    data = data.sort_values("group_value")
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(data))
    ax.bar(x, data["bio_top1_minus_latest_mean"].astype(float), color="#4C78A8")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v).replace("_", "\n") for v in data["group_value"]], fontsize=8)
    ax.set_ylabel("test bad_top10 top1 RMSE delta vs latest")
    ax.set_title("v287: temporal windows, lower than 0 would pass latest")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_signal_corr(summary: pd.DataFrame, feature_audit: pd.DataFrame) -> Path:
    """画信号族在歧义差样本上的 rank correlation。"""

    path = FIGURES / "v287_signal_bad_ambiguous_corr.png"
    data = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10_vehicle_ambiguous")
    ].merge(feature_audit, on="feature_set", how="left")
    data = data[data["group_type"].eq("signal")].copy()
    if data.empty:
        return path
    data = data.sort_values("bio_corr_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(data))
    ax.bar(x, data["bio_corr_mean"].astype(float), color="#59A14F")
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.05, color="tab:red", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(data["group_value"].astype(str), rotation=20, ha="right")
    ax.set_ylabel("test bad ambiguous rank corr")
    ax.set_title("v287: signal-family rank correlation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_group_winners(winners: pd.DataFrame) -> Path:
    """画各 group_type 的 best top1 delta。"""

    path = FIGURES / "v287_group_type_winners.png"
    data = winners[winners["event_group"].eq("bad_top10")].copy()
    if data.empty:
        return path
    data = data.sort_values("best_top1_delta")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(data))
    ax.bar(x, data["best_top1_delta"].astype(float), color="#F28E2B")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(data["group_type"].astype(str), rotation=20, ha="right")
    ax.set_ylabel("best test bad_top10 top1 delta")
    ax.set_title("v287: best deployable top1 by grouping strategy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v287_script", Path(__file__)),
        ("v284_script_for_gate_logic", V284_SCRIPT),
        ("v285_raw200_shape_features", V285_FEATURES),
        ("v285_train_only_feature_screen", V285_SCREEN),
        ("v285_guardrail", V285_GUARDRAIL),
        ("v278_candidates", V284.V278_CANDIDATES),
        ("v272_diag", V284.V272_DIAG),
    ]:
        rows.append(
            {
                "label": label,
                "path": str(path),
                "exists": bool(path.exists()),
                "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
            }
        )
    write_csv(pd.DataFrame(rows), LOGS / "input_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "size_bytes": int(path.stat().st_size)})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(__file__, arcname=f"scripts/{Path(__file__).name}")
        for folder in [TABLES, FIGURES, REPORTS, LOGS]:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(OUT)))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip() is None


def write_report(
    feature_audit: pd.DataFrame,
    winners: pd.DataFrame,
    summary: pd.DataFrame,
    val_test: pd.DataFrame,
    decision: pd.DataFrame,
    guardrail: Dict[str, object],
    figures: List[Path],
) -> Path:
    """写中文报告。"""

    report = REPORTS / "v287_physio_temporal_window_route_audit_cn.md"
    test_bad = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
    ].merge(feature_audit[["feature_set", "group_type", "group_value"]], on="feature_set", how="left")
    test_bad = test_bad.sort_values("bio_top1_minus_latest_mean")
    test_amb = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10_vehicle_ambiguous")
    ].merge(feature_audit[["feature_set", "group_type", "group_value"]], on="feature_set", how="left")
    test_amb = test_amb.sort_values("bio_top1_minus_latest_mean")

    lines: List[str] = []
    lines.append("# v287 physiology temporal-window route audit")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- v285 整体 raw 200Hz shape-state route gate 失败后，检查是否存在被混合稀释的有效时间窗口或信号族。")
    lines.append("- 本轮不重新抽波形、不训练轨迹融合模型，只复用 v285 causal 特征和 train-only screen。")
    lines.append("- 正式部署判断仍只看 validation 选择后的 deployable top1。")
    lines.append("")
    lines.append("## route gate 判定")
    lines.append("")
    lines.append(table_to_md(decision, ["check", "requirement", "pass", "evidence", "deployable", "route_viable_now"]))
    lines.append("")
    lines.append("## 各分组类型最好结果")
    lines.append("")
    lines.append(table_to_md(winners, max_rows=40))
    lines.append("")
    lines.append("## validation 选择后的 test 泛化")
    lines.append("")
    lines.append(
        table_to_md(
            val_test,
            [
                "event_group",
                "method",
                "deployable",
                "val_chosen_feature_set",
                "val_delta_vs_latest_mean",
                "test_delta_vs_latest_mean",
                "test_corr_mean",
                "test_passes_latest",
                "val_and_test_same_direction_gain",
            ],
            max_rows=40,
        )
    )
    lines.append("")
    lines.append("## test bad_top10 top feature sets")
    lines.append("")
    lines.append(
        table_to_md(
            test_bad,
            [
                "feature_set",
                "group_type",
                "group_value",
                "n",
                "latest_rmse_mean",
                "bio_top1_rmse_mean",
                "bio_top1_minus_latest_mean",
                "bio_top3_minus_latest_mean",
                "bio_corr_mean",
                "bio_best_in_top3_rate",
            ],
            max_rows=60,
        )
    )
    lines.append("")
    lines.append("## test bad_top10 + vehicle_ambiguous top feature sets")
    lines.append("")
    lines.append(
        table_to_md(
            test_amb,
            [
                "feature_set",
                "group_type",
                "group_value",
                "n",
                "latest_rmse_mean",
                "bio_top1_rmse_mean",
                "bio_top1_minus_latest_mean",
                "bio_top3_minus_latest_mean",
                "bio_corr_mean",
                "bio_best_in_top3_rate",
            ],
            max_rows=60,
        )
    )
    lines.append("")
    lines.append("## feature set 审计")
    lines.append("")
    lines.append(table_to_md(feature_audit, max_rows=80))
    lines.append("")
    lines.append("## 关键判读")
    lines.append("")
    route_viable = bool(decision["route_viable_now"].iloc[0]) if len(decision) else False
    if route_viable:
        lines.append("- route gate 通过：至少一个时间窗口/信号族分组已经具备进入轨迹模型的最低证据。")
    else:
        lines.append("- route gate 未通过：没有发现单独时间窗口、信号族或特征类型能够把生理信号转成可部署 top1 收益。")
    lines.append("- 如果连窗口/信号族拆分都没有通过，下一步不应继续在同一 v285 特征层上做复杂融合。")
    lines.append("- 若仍坚持生理路线，应先回到源信号清洗/事件同步证据，而不是继续换 selector。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    lines.append("")
    lines.append("## guardrail")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    print("[v287] 目的：拆分 v285 raw 200Hz 特征的时间窗口/信号族并重新验证 route gate。", flush=True)
    clean_out_dir()
    np.random.seed(SEED)

    features, screen = load_v285_inputs()
    enriched = enrich_screen(screen, features)
    feature_sets, feature_audit = build_feature_sets(enriched)
    if len(feature_sets) < 5:
        raise RuntimeError(f"v287 可用 feature set 太少：{len(feature_sets)}")

    cand = V284.load_candidate_table()
    context = V284.build_event_context(cand)

    per_event_parts = []
    scaler_parts = []
    audit_parts = []
    for name, cols in feature_sets.items():
        print(f"[v287] evaluate feature_set={name} feature_n={len(cols)}", flush=True)
        per_event, scaler, audit = V284.evaluate_feature_set(name, cols, features, cand, context)
        per_event_parts.append(per_event)
        scaler_parts.append(scaler)
        audit_parts.append(audit)

    per_event_all = pd.concat(per_event_parts, ignore_index=True)
    scaler_all = pd.concat(scaler_parts, ignore_index=True)
    eval_audit = pd.concat(audit_parts, ignore_index=True)
    feature_audit = feature_audit.merge(eval_audit, on="feature_set", how="left", suffixes=("", "_eval"))
    expanded = V284.expand_groups(per_event_all)
    summary = V284.summarize_groups(expanded)
    val_test = V284.val_chosen_generalization(summary)
    decision = V284.route_gate_decision(summary, val_test)
    winners = summarize_group_winners(summary, feature_audit)

    write_csv(enriched, TABLES / "v287_enriched_train_only_feature_screen.csv")
    write_csv(feature_audit, TABLES / "v287_feature_set_audit.csv")
    write_csv(scaler_all, TABLES / "v287_train_scaler_audit.csv")
    write_csv(per_event_all, TABLES / "v287_route_gate_per_event.csv")
    write_csv(summary, TABLES / "v287_route_group_summary.csv")
    write_csv(val_test, TABLES / "v287_val_chosen_generalization.csv")
    write_csv(decision, TABLES / "v287_route_gate_decision.csv")
    write_csv(winners, TABLES / "v287_group_winner_summary.csv")
    write_input_hashes()

    figures = [plot_window_delta(summary, feature_audit), plot_signal_corr(summary, feature_audit), plot_group_winners(winners)]
    v285_guard = json.loads(V285_GUARDRAIL.read_text(encoding="utf-8")) if V285_GUARDRAIL.exists() else {}
    test_bad = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")]
    best_top1_delta = float(test_bad["bio_top1_minus_latest_mean"].min()) if not test_bad.empty else math.nan
    best_corr = float(test_bad["bio_corr_mean"].max()) if not test_bad.empty else math.nan
    guardrail: Dict[str, object] = {
        "pass": True,
        "zip_testzip": False,
        "event_n": int(features["event_uid"].nunique()),
        "candidate_rows": int(len(cand)),
        "screen_feature_n": int(len(enriched)),
        "feature_set_n": int(len(feature_sets)),
        "v285_source_guardrail_pass": bool(v285_guard.get("pass", False)),
        "v285_source_uses_post_observation_any": bool(v285_guard.get("uses_post_observation_any", False)),
        "route_viable_now": bool(decision["route_viable_now"].iloc[0]),
        "deployable_top1_badtop10_pass": bool(
            decision.loc[decision["check"].eq("deployable_top1_val_chosen_bad_top10"), "pass"].iloc[0]
        ),
        "deployable_top1_bad_ambiguous_pass": bool(
            decision.loc[decision["check"].eq("deployable_top1_val_chosen_bad_ambiguous"), "pass"].iloc[0]
        ),
        "test_best_top1_diagnostic_pass": bool(
            decision.loc[decision["check"].eq("test_best_top1_diagnostic_beats_latest"), "pass"].iloc[0]
        ),
        "best_test_badtop10_top1_delta": best_top1_delta,
        "best_test_badtop10_corr": best_corr,
        "test_used_for_feature_selection": False,
    }
    guardrail["pass"] = bool(
        guardrail["event_n"] > 0
        and guardrail["candidate_rows"] > 0
        and guardrail["screen_feature_n"] >= 100
        and guardrail["feature_set_n"] >= 10
        and guardrail["v285_source_guardrail_pass"]
        and not guardrail["v285_source_uses_post_observation_any"]
        and not guardrail["test_used_for_feature_selection"]
    )
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(feature_audit, winners, summary, val_test, decision, guardrail, figures)
    write_file_inventory()
    first_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(first_zip_ok)
    guardrail["pass"] = bool(guardrail["pass"] and first_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(feature_audit, winners, summary, val_test, decision, guardrail, figures)
    write_file_inventory()
    second_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(second_zip_ok)
    guardrail["pass"] = bool(guardrail["pass"] and second_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()
    if not bool(guardrail["pass"]):
        raise AssertionError("v287 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))

    print(f"[v287] report={report}", flush=True)
    print(f"[v287] zip={ZIP_PATH}", flush=True)
    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
