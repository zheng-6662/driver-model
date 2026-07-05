#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v263 bio260 wait gate。

v261/v262 都是在 v247 的 fine-grid 候选锚点中做连续 selector。这个目标可能过难：
模型要同时判断是否等待、等待多久、以及候选锚点 replay 误差。

本轮把问题简化成一个更直接的门控任务：
- 只在 0ms 时刻做决定；
- 预测“直接等到 1000ms 是否比 0ms 原锚点更好”；
- 比较 vehicle gate 与 vehicle+subject-invariant bio260 gate；
- 如果简单 wait gate 仍不能接近固定 wait-latest，说明当前生理状态还不能可靠解决差样本的信息不足。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
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

V262_SCRIPT = BASELINES / "scripts" / "stage03_v262_subject_invariant_bio260_selector_20260702.py"

OUT = BASELINES / "v263_bio260_wait_gate_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v263_bio260_wait_gate_20260702_pack.zip"

SEED = 26302

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def load_v262_module():
    if not V262_SCRIPT.exists():
        raise FileNotFoundError(f"缺少 v262 脚本：{V262_SCRIPT}")
    spec = importlib.util.spec_from_file_location("v262_subject_invariant_selector", V262_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 v262 脚本：{V262_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.TABLES = TABLES
    module.FIGURES = FIGURES
    module.REPORTS = REPORTS
    module.LOGS = LOGS
    module.ZIP_PATH = ZIP_PATH
    module.SEED = SEED
    module.V261.OUT = OUT
    module.V261.TABLES = TABLES
    module.V261.FIGURES = FIGURES
    module.V261.REPORTS = REPORTS
    module.V261.LOGS = LOGS
    module.V261.ZIP_PATH = ZIP_PATH
    module.V261.SEED = SEED
    module.V258.OUT = OUT
    module.V258.TABLES = TABLES
    module.V258.FIGURES = FIGURES
    module.V258.REPORTS = REPORTS
    module.V258.LOGS = LOGS
    module.V258.ZIP_PATH = ZIP_PATH
    module.V258.SEED = SEED
    return module


V262 = load_v262_module()
V261 = V262.V261
V258 = V262.V258


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


def make_event_table(candidate_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for event_uid, group in candidate_df.groupby("event_uid", sort=False):
        group = group.copy()
        delay = pd.to_numeric(group["candidate_delay_ms"], errors="coerce")
        keep0 = group.loc[delay.idxmin()]
        latest = group.loc[delay.idxmax()]
        oracle = group.loc[pd.to_numeric(group["candidate_tail_rmse_v241"], errors="coerce").idxmin()]
        row = keep0.to_dict()
        row.update(
            {
                "event_uid": str(event_uid),
                "keep0_tail_rmse_v241": float(keep0["candidate_tail_rmse_v241"]),
                "latest_tail_rmse_v241": float(latest["candidate_tail_rmse_v241"]),
                "oracle_tail_rmse_v241": float(oracle["candidate_tail_rmse_v241"]),
                "oracle_delay_ms": int(oracle["candidate_delay_ms"]),
                "gain_latest_vs_keep0": float(keep0["candidate_tail_rmse_v241"] - latest["candidate_tail_rmse_v241"]),
                "wait_better_label": float(latest["candidate_tail_rmse_v241"] < keep0["candidate_tail_rmse_v241"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def tune_threshold(
    events: pd.DataFrame,
    pred_col: str,
    split_name: str = "val",
    bad_weight: float = 0.0,
) -> Tuple[float, pd.DataFrame]:
    val = events[events["split"].astype(str).eq(split_name)].copy()
    if val.empty:
        return 0.0, pd.DataFrame()
    pred = pd.to_numeric(val[pred_col], errors="coerce").to_numpy(dtype=float)
    finite = pred[np.isfinite(pred)]
    if len(finite) == 0:
        return 0.0, pd.DataFrame()
    grid = np.unique(np.concatenate([np.quantile(finite, np.linspace(0.0, 1.0, 81)), np.array([0.0])]))
    weights = np.ones(len(val), dtype=float)
    if bad_weight > 0:
        weights += bad_weight * val["bad_top10_split_v241"].fillna(False).astype(bool).to_numpy(dtype=float)
    rows = []
    keep0 = val["keep0_tail_rmse_v241"].to_numpy(dtype=float)
    latest = val["latest_tail_rmse_v241"].to_numpy(dtype=float)
    for threshold in grid:
        choose_latest = pred > threshold
        selected = np.where(choose_latest, latest, keep0)
        rows.append(
            {
                "pred_col": pred_col,
                "threshold": float(threshold),
                "bad_weight": float(bad_weight),
                "val_weighted_tail_rmse": float(np.average(selected, weights=weights)),
                "val_tail_rmse": float(selected.mean()),
                "val_latest_rate": float(choose_latest.mean()),
            }
        )
    audit = pd.DataFrame(rows).sort_values("val_weighted_tail_rmse", ascending=True)
    return float(audit["threshold"].iloc[0]), audit


def build_selected(
    events: pd.DataFrame,
    strategy: str,
    pred_col: str | None = None,
    threshold: float = 0.0,
    force: str | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in events.iterrows():
        if force == "keep0":
            choose_latest = False
        elif force == "latest":
            choose_latest = True
        elif force == "oracle":
            choose_latest = False
        else:
            if pred_col is None:
                raise ValueError("pred_col is required for learned gate")
            choose_latest = float(row[pred_col]) > float(threshold)
        if force == "oracle":
            selected_delay = int(row["oracle_delay_ms"])
            selected_rmse = float(row["oracle_tail_rmse_v241"])
        elif choose_latest:
            selected_delay = 1000
            selected_rmse = float(row["latest_tail_rmse_v241"])
        else:
            selected_delay = 0
            selected_rmse = float(row["keep0_tail_rmse_v241"])
        rows.append(
            {
                "strategy": strategy,
                "event_uid": str(row["event_uid"]),
                "split": str(row["split"]),
                "subject": str(row.get("subject", "")),
                "recording": str(row.get("recording", "")),
                "selected_delay_ms": selected_delay,
                "selected_tail_rmse_v241": selected_rmse,
                "keep0_tail_rmse_v241": float(row["keep0_tail_rmse_v241"]),
                "latest_tail_rmse_v241": float(row["latest_tail_rmse_v241"]),
                "oracle_tail_rmse_v241": float(row["oracle_tail_rmse_v241"]),
                "delta_selected_minus_keep0": float(selected_rmse - row["keep0_tail_rmse_v241"]),
                "delta_selected_minus_latest": float(selected_rmse - row["latest_tail_rmse_v241"]),
                "bad_top10": bool(row["bad_top10_split_v241"]),
                "very_bad_top5": bool(row["very_bad_top5_split_v241"]),
                "normal": bool(row["normal_curve_current0"]),
                "observe_later_like": bool(row["observe_later_like_current0"]),
                "strong_steer": bool(row["strong_steer_current0"]),
                "reverse": bool(row["reverse_current0"]),
                "early_best_after_400": bool(int(row["oracle_delay_ms"]) >= 400),
            }
        )
    return pd.DataFrame(rows)


def plot_test_summary(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v263_bio260_wait_gate_test_badtop10.png"
    order = [
        "policy_keep_0ms_anchor",
        "gate_vehicle_gain_t0",
        "gate_vehicle_bio260_sp64_gain_t0",
        "gate_vehicle_bio260_sp64_gain_val_all",
        "gate_vehicle_bio260_sp64_gain_val_badweighted",
        "policy_wait_to_latest_anchor",
        "oracle_best_anchor_upper_bound",
    ]
    sub = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["strategy"].isin(order)
    ].copy()
    if sub.empty:
        return path
    sub["strategy"] = pd.Categorical(sub["strategy"], categories=order, ordered=True)
    sub = sub.sort_values("strategy")
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    x = np.arange(len(sub))
    colors = ["#9CA3AF", "#4C78A8", "#59A14F", "#76B7B2", "#F28E2B", "#E15759", "#B07AA1"]
    ax.bar(x, sub["selected_tail_rmse_mean"], color=colors[: len(sub)])
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in sub["strategy"]], fontsize=8)
    ax.set_ylabel("test bad_top10 selected tail RMSE")
    ax.set_title("v263: 0ms 生理状态是否能门控 wait-latest")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v263_script", Path(__file__)),
        ("v262_script", V262_SCRIPT),
        ("v260_event_biomarker_features", V261.BIO260_FEATURES),
        ("v260_eta2_table", V262.ETA_TABLE),
        ("v247_selector_training_table", V261.V247_TABLE),
        ("v247_fine_anchor_candidate_table", V261.V247_FINE_TABLE),
    ]:
        rows.append(
            {
                "label": label,
                "path": str(path),
                "exists": bool(path.exists()),
                "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
            }
        )
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


def write_report(
    summary: pd.DataFrame,
    threshold_audit: pd.DataFrame,
    feature_audit: pd.DataFrame,
    figures: List[Path],
) -> None:
    lines: List[str] = []
    lines.append("# v263 bio260 wait gate")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("- v261/v262 的 fine-grid selector 只得到小幅生理增益，可能是任务目标太细。")
    lines.append("- v263 把任务简化成 0ms 决策：直接保留原锚点，还是等到 1000ms。")
    lines.append("- 如果 vehicle+bio260 wait gate 仍不能明显接近 fixed latest，说明当前生理状态对差样本决策仍不够强。")
    lines.append("")
    lines.append("## 特征与阈值")
    lines.append("")
    lines.append(feature_audit.to_markdown(index=False))
    lines.append("")
    best_thresholds = threshold_audit.groupby(["pred_col", "bad_weight"], as_index=False).head(1)
    lines.append(best_thresholds.to_markdown(index=False))
    lines.append("")
    lines.append("## Test 关键结果")
    lines.append("")
    focus = summary[
        summary["split"].eq("test")
        & summary["event_group"].isin(
            ["all", "bad_top10", "early_best_after_400", "normal", "strong_steer", "observe_later_like"]
        )
        & summary["strategy"].isin(
            [
                "policy_keep_0ms_anchor",
                "gate_vehicle_gain_t0",
                "gate_vehicle_bio260_sp64_gain_t0",
                "gate_vehicle_bio260_sp64_gain_val_all",
                "gate_vehicle_bio260_sp64_gain_val_badweighted",
                "policy_wait_to_latest_anchor",
                "oracle_best_anchor_upper_bound",
            ]
        )
    ].copy()
    lines.append(
        focus[
            [
                "event_group",
                "strategy",
                "n",
                "selected_tail_rmse_mean",
                "delta_selected_minus_keep0_mean",
                "delta_selected_minus_latest_mean",
                "improve_rate_vs_keep0",
                "selected_delay_ms_mean",
                "selected_latest_rate",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    bad = focus[focus["event_group"].eq("bad_top10")].copy()
    scores: Dict[str, float] = {}
    for strategy in [
        "policy_keep_0ms_anchor",
        "gate_vehicle_gain_t0",
        "gate_vehicle_bio260_sp64_gain_t0",
        "gate_vehicle_bio260_sp64_gain_val_all",
        "gate_vehicle_bio260_sp64_gain_val_badweighted",
        "policy_wait_to_latest_anchor",
        "oracle_best_anchor_upper_bound",
    ]:
        row = bad[bad["strategy"].eq(strategy)]
        if len(row):
            score = float(row["selected_tail_rmse_mean"].iloc[0])
            latest_rate = float(row["selected_latest_rate"].iloc[0])
            scores[strategy] = score
            lines.append(f"- bad_top10 / {strategy}: tail={score:.4f}, latest_rate={latest_rate:.3f}.")
    vehicle = scores.get("gate_vehicle_gain_t0", np.nan)
    bio = scores.get("gate_vehicle_bio260_sp64_gain_t0", np.nan)
    latest = scores.get("policy_wait_to_latest_anchor", np.nan)
    if np.isfinite(vehicle) and np.isfinite(bio):
        if bio < vehicle:
            lines.append(f"- 结论：bio260 wait gate 比 vehicle gate 低 {vehicle - bio:.4f}。")
        else:
            lines.append(f"- 结论：bio260 wait gate 比 vehicle gate 高 {bio - vehicle:.4f}。")
    if np.isfinite(latest):
        lines.append(f"- fixed latest 是不需要生理判断的强基线，tail={latest:.4f}；任何生理 gate 若接近它才有实际意义。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v263_bio260_wait_gate_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v263] bio260 wait gate")
    clean_out_dir()
    np.random.seed(SEED)

    candidate_df, merge_audit = V261.load_augmented_table()
    vehicle_cols, all_bio_cols = V261.feature_columns(candidate_df)
    eta_table = V262.load_eta_feature_table()
    feature_sets, feature_selection = V262.build_feature_sets(all_bio_cols, eta_table)
    events = make_event_table(candidate_df)

    # 0ms 决策只能用原锚点时刻可见的信息。candidate_delay 常量保留也不会提供未来信息。
    bio_sp64 = feature_sets["bio260_sp64"]
    train_mask = events["split"].astype(str).eq("train").to_numpy()
    events["target_gain_latest_vs_keep0"] = pd.to_numeric(events["gain_latest_vs_keep0"], errors="coerce")

    pred_vehicle, audit_vehicle = V258.train_model(events, vehicle_cols, "target_gain_latest_vs_keep0", train_mask)
    pred_vehicle_bio, audit_vehicle_bio = V258.train_model(
        events, vehicle_cols + bio_sp64, "target_gain_latest_vs_keep0", train_mask
    )
    events["pred_gain_vehicle"] = pred_vehicle
    events["pred_gain_vehicle_bio260_sp64"] = pred_vehicle_bio

    threshold_vehicle_all, audit_vehicle_all = tune_threshold(events, "pred_gain_vehicle", bad_weight=0.0)
    threshold_bio_all, audit_bio_all = tune_threshold(events, "pred_gain_vehicle_bio260_sp64", bad_weight=0.0)
    threshold_bio_bad, audit_bio_bad = tune_threshold(events, "pred_gain_vehicle_bio260_sp64", bad_weight=4.0)
    threshold_audit = pd.concat([audit_vehicle_all, audit_bio_all, audit_bio_bad], ignore_index=True)
    threshold_audit["selected_by_script"] = False
    for pred_col, bad_weight, threshold in [
        ("pred_gain_vehicle", 0.0, threshold_vehicle_all),
        ("pred_gain_vehicle_bio260_sp64", 0.0, threshold_bio_all),
        ("pred_gain_vehicle_bio260_sp64", 4.0, threshold_bio_bad),
    ]:
        mask = (
            threshold_audit["pred_col"].eq(pred_col)
            & np.isclose(threshold_audit["bad_weight"].astype(float), bad_weight)
            & np.isclose(threshold_audit["threshold"].astype(float), threshold)
        )
        threshold_audit.loc[mask, "selected_by_script"] = True

    selected = pd.concat(
        [
            build_selected(events, "policy_keep_0ms_anchor", force="keep0"),
            build_selected(events, "policy_wait_to_latest_anchor", force="latest"),
            build_selected(events, "oracle_best_anchor_upper_bound", force="oracle"),
            build_selected(events, "gate_vehicle_gain_t0", "pred_gain_vehicle", threshold=0.0),
            build_selected(events, "gate_vehicle_bio260_sp64_gain_t0", "pred_gain_vehicle_bio260_sp64", threshold=0.0),
            build_selected(
                events,
                "gate_vehicle_bio260_sp64_gain_val_all",
                "pred_gain_vehicle_bio260_sp64",
                threshold=threshold_bio_all,
            ),
            build_selected(
                events,
                "gate_vehicle_bio260_sp64_gain_val_badweighted",
                "pred_gain_vehicle_bio260_sp64",
                threshold=threshold_bio_bad,
            ),
        ],
        ignore_index=True,
    )
    summary = V258.summarize_selected(selected)
    figures = [plot_test_summary(summary)]

    feature_audit = pd.DataFrame(
        [
            {"model": "gate_vehicle_gain", "feature_n": len(vehicle_cols), "bio260_feature_n": 0},
            {
                "model": "gate_vehicle_bio260_sp64_gain",
                "feature_n": len(vehicle_cols) + len(bio_sp64),
                "bio260_feature_n": len(bio_sp64),
            },
        ]
    )
    fill_audit = pd.concat(
        [
            audit_vehicle.assign(model="gate_vehicle_gain"),
            audit_vehicle_bio.assign(model="gate_vehicle_bio260_sp64_gain"),
        ],
        ignore_index=True,
    )

    write_csv(events, TABLES / "v263_event_wait_gate_predictions.csv")
    write_csv(selected, TABLES / "v263_selected_wait_gate_by_strategy.csv")
    write_csv(summary, TABLES / "v263_wait_gate_summary.csv")
    write_csv(merge_audit, TABLES / "v263_bio260_merge_audit.csv")
    write_csv(feature_audit, TABLES / "v263_feature_block_audit.csv")
    write_csv(fill_audit, TABLES / "v263_feature_fill_audit.csv")
    write_csv(feature_selection, TABLES / "v263_feature_selection_audit.csv")
    write_csv(threshold_audit, TABLES / "v263_threshold_tuning_audit.csv")
    write_input_hashes()
    write_file_inventory()
    write_report(summary, threshold_audit, feature_audit, figures)
    write_file_inventory()

    zip_ok = make_zip()
    guardrail = {
        "pass": bool(zip_ok and float(merge_audit["bio260_uses_post_observation_max"].iloc[0]) == 0.0),
        "zip_testzip": bool(zip_ok),
        "train_only_fit": True,
        "decision_time": "0ms",
        "candidate_policy": "keep0_or_latest_only",
        "bio260_uses_post_observation_max": float(merge_audit["bio260_uses_post_observation_max"].iloc[0]),
        "event_n": int(events["event_uid"].nunique()),
        "bio260_sp64_feature_n": int(len(bio_sp64)),
        "threshold_vehicle_all": float(threshold_vehicle_all),
        "threshold_bio_all": float(threshold_bio_all),
        "threshold_bio_badweighted": float(threshold_bio_bad),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not guardrail["pass"]:
        raise AssertionError("v263 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    focus = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["strategy"].isin(
            [
                "policy_keep_0ms_anchor",
                "gate_vehicle_gain_t0",
                "gate_vehicle_bio260_sp64_gain_t0",
                "gate_vehicle_bio260_sp64_gain_val_all",
                "gate_vehicle_bio260_sp64_gain_val_badweighted",
                "policy_wait_to_latest_anchor",
                "oracle_best_anchor_upper_bound",
            ]
        )
    ]
    print(f"[v263] report={REPORTS / 'v263_bio260_wait_gate_cn.md'}")
    print(f"[v263] zip={ZIP_PATH}")
    print(
        focus[
            [
                "strategy",
                "selected_tail_rmse_mean",
                "delta_selected_minus_keep0_mean",
                "selected_delay_ms_mean",
                "selected_latest_rate",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
