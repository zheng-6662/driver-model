#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v262 subject-invariant bio260 anchor selector。

v261 的全量 bio260 没有改善 bad_top10 selector。一个关键嫌疑是：
生理特征里有大量个体/记录差异，模型学到的是“人是谁/设备状态”，而不是事件前的生理状态。

本轮只改变 bio260 特征筛选：
- 使用 v260 eta2 表，惩罚 subject / recording eta2 过高的特征；
- 单独构造“状态变化类”特征集合，如 delta、slope、range、last_minus_first、phase；
- 其余训练、候选锚点、split、评价口径都复用 v261/v258，保证对照干净。
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

V261_SCRIPT = BASELINES / "scripts" / "stage03_v261_bio260_anchor_selector_20260702.py"
ETA_TABLE = (
    BASELINES
    / "v260_event_biomarker_physio_rebuild_20260702"
    / "tables"
    / "v260_biomarker_eta2_by_target_feature.csv"
)
V261_SUMMARY = (
    BASELINES
    / "v261_bio260_anchor_selector_20260702"
    / "tables"
    / "v261_anchor_selector_summary.csv"
)

OUT = BASELINES / "v262_subject_invariant_bio260_selector_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v262_subject_invariant_bio260_selector_20260702_pack.zip"

SEED = 26202

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def load_v261_module():
    if not V261_SCRIPT.exists():
        raise FileNotFoundError(f"缺少 v261 脚本：{V261_SCRIPT}")
    spec = importlib.util.spec_from_file_location("v261_bio260_selector", V261_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 v261 脚本：{V261_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.TABLES = TABLES
    module.FIGURES = FIGURES
    module.REPORTS = REPORTS
    module.LOGS = LOGS
    module.ZIP_PATH = ZIP_PATH
    module.SEED = SEED
    module.V258.OUT = OUT
    module.V258.TABLES = TABLES
    module.V258.FIGURES = FIGURES
    module.V258.REPORTS = REPORTS
    module.V258.LOGS = LOGS
    module.V258.ZIP_PATH = ZIP_PATH
    module.V258.SEED = SEED
    return module


V261 = load_v261_module()
V258 = V261.V258


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


def load_eta_feature_table() -> pd.DataFrame:
    if not ETA_TABLE.exists():
        raise FileNotFoundError(f"缺少 v260 eta2 表：{ETA_TABLE}")
    eta = pd.read_csv(ETA_TABLE, encoding="utf-8-sig", low_memory=False)
    pivot = eta.pivot_table(index="feature", columns="target", values="eta2", aggfunc="max").fillna(0.0)
    for col in ["bad_top10_v250_diagnostic", "future_cluster4", "high_future_abs_q75", "subject", "recording"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    out = pivot.reset_index().rename_axis(None, axis=1)
    out["future_eta_max"] = out[["future_cluster4", "high_future_abs_q75"]].max(axis=1)
    out["subject_recording_eta_max"] = out[["subject", "recording"]].max(axis=1)
    # 分数只用于排序：鼓励 bad_top10 / future 相关，惩罚 subject 和 recording 相关。
    out["invariant_score"] = (
        out["bad_top10_v250_diagnostic"]
        + 0.35 * out["future_eta_max"]
        - 0.18 * out["subject"]
        - 0.08 * out["recording"]
    )
    return out


def is_state_change_feature(feature: str) -> bool:
    tokens = ["delta_", "_slope", "_range", "last_minus_first", "phase_sin", "phase_cos", "burst_rate"]
    return any(token in feature for token in tokens)


def build_feature_sets(
    all_bio_cols: List[str],
    eta_table: pd.DataFrame,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    eta_by_feature = eta_table.set_index("feature")
    candidates = []
    status_cols = [col for col in all_bio_cols if col == "bio260_floor_status_ok"]
    for col in all_bio_cols:
        if not col.startswith("floor_bio260_"):
            continue
        original = col.replace("floor_", "", 1)
        if original not in eta_by_feature.index:
            continue
        record = eta_by_feature.loc[original].to_dict()
        record["column"] = col
        record["feature"] = original
        record["is_state_change"] = is_state_change_feature(original)
        candidates.append(record)
    info = pd.DataFrame(candidates)
    if info.empty:
        raise ValueError("没有可用于 v262 的 bio260 eta 特征。")

    low_subject = info[
        (info["subject_recording_eta_max"] <= 0.40)
        & ((info["bad_top10_v250_diagnostic"] > 0.002) | (info["future_eta_max"] > 0.002))
    ].copy()
    if len(low_subject) < 16:
        low_subject = info.sort_values(
            ["subject_recording_eta_max", "bad_top10_v250_diagnostic", "future_eta_max"],
            ascending=[True, False, False],
        ).head(64)
    else:
        low_subject = low_subject.sort_values("invariant_score", ascending=False)

    sp32 = low_subject.head(32)["column"].tolist()
    sp64 = low_subject.head(64)["column"].tolist()

    delta = info[
        info["is_state_change"].astype(bool) & (info["subject_recording_eta_max"] <= 0.60)
    ].sort_values("invariant_score", ascending=False)
    if len(delta) == 0:
        delta = info[info["is_state_change"].astype(bool)].sort_values("invariant_score", ascending=False)
    delta_cols = delta.head(96)["column"].tolist()

    sets = {
        "bio260_sp32": list(dict.fromkeys(sp32 + status_cols)),
        "bio260_sp64": list(dict.fromkeys(sp64 + status_cols)),
        "bio260_state_change": list(dict.fromkeys(delta_cols + status_cols)),
    }
    for set_name, cols in sets.items():
        selected_originals = [col.replace("floor_", "", 1) for col in cols if col.startswith("floor_bio260_")]
        sub = info[info["feature"].isin(selected_originals)].copy()
        rows.append(
            {
                "feature_set": set_name,
                "feature_n": len(cols),
                "eta_known_feature_n": len(sub),
                "bad_eta_mean": float(sub["bad_top10_v250_diagnostic"].mean()) if len(sub) else 0.0,
                "future_eta_mean": float(sub["future_eta_max"].mean()) if len(sub) else 0.0,
                "subject_recording_eta_max_mean": float(sub["subject_recording_eta_max"].mean()) if len(sub) else 0.0,
                "state_change_rate": float(sub["is_state_change"].mean()) if len(sub) else 0.0,
            }
        )
    selection_audit = pd.DataFrame(rows)
    detail = pd.concat(
        [
            info.assign(in_sp32=info["column"].isin(sets["bio260_sp32"])),
            info.assign(in_sp64=info["column"].isin(sets["bio260_sp64"])),
            info.assign(in_state_change=info["column"].isin(sets["bio260_state_change"])),
        ],
        ignore_index=True,
    )
    # detail 上面是便于人工查阅的长表，下面再补一张去重宽表。
    wide = info.copy()
    wide["in_sp32"] = wide["column"].isin(sets["bio260_sp32"])
    wide["in_sp64"] = wide["column"].isin(sets["bio260_sp64"])
    wide["in_state_change"] = wide["column"].isin(sets["bio260_state_change"])
    return sets, pd.concat([selection_audit.assign(row_type="summary"), wide.assign(row_type="feature")], ignore_index=True)


def plot_test_summary(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v262_subject_invariant_bio260_test_badtop10.png"
    order = [
        "policy_keep_0ms_anchor",
        "selector_vehicle_hgb",
        "selector_vehicle_bio260_sp32_hgb",
        "selector_vehicle_bio260_sp64_hgb",
        "selector_vehicle_bio260_sp64_badweighted_hgb",
        "selector_vehicle_bio260_state_change_hgb",
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
    fig, ax = plt.subplots(figsize=(14, 5.2))
    x = np.arange(len(sub))
    colors = ["#9CA3AF", "#4C78A8", "#59A14F", "#76B7B2", "#F28E2B", "#EDC948", "#E15759", "#B07AA1"]
    ax.bar(x, sub["selected_tail_rmse_mean"], color=colors[: len(sub)])
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in sub["strategy"]], fontsize=8)
    ax.set_ylabel("test bad_top10 selected tail RMSE")
    ax.set_title("v262: subject-invariant bio260 特征是否改善 bad_top10 anchor selector")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def load_v261_reference() -> pd.DataFrame:
    if not V261_SUMMARY.exists():
        return pd.DataFrame()
    ref = pd.read_csv(V261_SUMMARY, encoding="utf-8-sig", low_memory=False)
    keep = ref[
        ref["split"].eq("test")
        & ref["event_group"].eq("bad_top10")
        & ref["strategy"].isin(
            [
                "policy_keep_0ms_anchor",
                "selector_vehicle_hgb",
                "selector_bio260_hgb",
                "selector_vehicle_bio260_hgb",
                "selector_vehicle_bio260_badweighted_hgb",
                "policy_wait_to_latest_anchor",
                "oracle_best_anchor_upper_bound",
            ]
        )
    ].copy()
    if keep.empty:
        return keep
    keep = keep[
        [
            "strategy",
            "n",
            "selected_tail_rmse_mean",
            "delta_selected_minus_keep0_mean",
            "selected_delay_ms_mean",
        ]
    ].copy()
    keep.insert(0, "source", "v261_full_bio260_ref")
    return keep


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v261_script", V261_SCRIPT),
        ("v260_eta2_table", ETA_TABLE),
        ("v260_event_biomarker_features", V261.BIO260_FEATURES),
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
    merge_audit: pd.DataFrame,
    feature_audit: pd.DataFrame,
    feature_selection: pd.DataFrame,
    v261_ref: pd.DataFrame,
    figures: List[Path],
) -> None:
    lines: List[str] = []
    lines.append("# v262 subject-invariant bio260 anchor selector")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("- v261 全量 bio260 selector 在 bad_top10 上弱于 vehicle selector。")
    lines.append("- v260 eta2 显示部分生理特征有强 subject / recording 成分，这会干扰 subject-disjoint 泛化。")
    lines.append("- v262 检查：剔除高个体差异特征、保留状态变化特征后，生理是否能重新产生锚点选择增益。")
    lines.append("")
    lines.append("## 合并审计")
    lines.append("")
    lines.append(merge_audit.to_markdown(index=False))
    lines.append("")
    lines.append("## 特征集合")
    lines.append("")
    summary_rows = feature_selection[feature_selection["row_type"].eq("summary")].copy()
    lines.append(summary_rows.to_markdown(index=False))
    lines.append("")
    lines.append(feature_audit.to_markdown(index=False))
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
                "selector_vehicle_hgb",
                "selector_vehicle_bio260_sp32_hgb",
                "selector_vehicle_bio260_sp64_hgb",
                "selector_vehicle_bio260_sp64_badweighted_hgb",
                "selector_vehicle_bio260_state_change_hgb",
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
            ]
        ].to_markdown(index=False)
    )
    lines.append("")
    if not v261_ref.empty:
        lines.append("## v261 全量 bio260 参考")
        lines.append("")
        lines.append(v261_ref.to_markdown(index=False))
        lines.append("")
    lines.append("## 判读")
    lines.append("")
    bad = focus[focus["event_group"].eq("bad_top10")].copy()
    score_by_strategy: Dict[str, float] = {}
    for strategy in [
        "policy_keep_0ms_anchor",
        "selector_vehicle_hgb",
        "selector_vehicle_bio260_sp32_hgb",
        "selector_vehicle_bio260_sp64_hgb",
        "selector_vehicle_bio260_sp64_badweighted_hgb",
        "selector_vehicle_bio260_state_change_hgb",
        "policy_wait_to_latest_anchor",
        "oracle_best_anchor_upper_bound",
    ]:
        row = bad[bad["strategy"].eq(strategy)]
        if len(row):
            score = float(row["selected_tail_rmse_mean"].iloc[0])
            delta = float(row["delta_selected_minus_keep0_mean"].iloc[0])
            delay = float(row["selected_delay_ms_mean"].iloc[0])
            score_by_strategy[strategy] = score
            lines.append(f"- bad_top10 / {strategy}: tail={score:.4f}, delta_keep0={delta:+.4f}, delay={delay:.1f}ms.")
    vehicle = score_by_strategy.get("selector_vehicle_hgb", np.nan)
    best_bio_strategy = None
    best_bio_score = np.inf
    for strategy, score in score_by_strategy.items():
        if "bio260" in strategy and score < best_bio_score:
            best_bio_strategy = strategy
            best_bio_score = score
    if best_bio_strategy is not None and np.isfinite(vehicle):
        lines.append("")
        if best_bio_score < vehicle:
            lines.append(
                f"- 结论：最佳 subject-invariant bio260 策略 `{best_bio_strategy}` 比 vehicle selector 低 {vehicle - best_bio_score:.4f}，说明去个体差异后生理存在可用增益。"
            )
        else:
            lines.append(
                f"- 结论：最佳 subject-invariant bio260 策略 `{best_bio_strategy}` 仍比 vehicle selector 高 {best_bio_score - vehicle:.4f}，说明问题不只是 subject 混淆。"
            )
    latest = score_by_strategy.get("policy_wait_to_latest_anchor", np.nan)
    if np.isfinite(latest) and np.isfinite(best_bio_score):
        lines.append(
            f"- 与固定 latest 比：最佳 bio260 tail={best_bio_score:.4f}，latest tail={latest:.4f}；若仍高很多，则当前生理还不能替代简单多观察。"
        )
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v262_subject_invariant_bio260_selector_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v262] subject-invariant bio260 selector")
    clean_out_dir()
    np.random.seed(SEED)

    df, merge_audit = V261.load_augmented_table()
    vehicle_cols, all_bio_cols = V261.feature_columns(df)
    eta_table = load_eta_feature_table()
    feature_sets, feature_selection = build_feature_sets(all_bio_cols, eta_table)

    train_mask = df["split"].astype(str).eq("train").to_numpy()
    bad_weight = 1.0 + 4.0 * df["bad_top10_split_v241"].fillna(False).astype(bool).to_numpy(dtype=float)

    pred_vehicle, audit_vehicle = V258.train_model(df, vehicle_cols, "target_score_primary", train_mask)
    pred_sp32, audit_sp32 = V258.train_model(
        df, vehicle_cols + feature_sets["bio260_sp32"], "target_score_primary", train_mask
    )
    pred_sp64, audit_sp64 = V258.train_model(
        df, vehicle_cols + feature_sets["bio260_sp64"], "target_score_primary", train_mask
    )
    pred_sp64_bad, audit_sp64_bad = V258.train_model(
        df,
        vehicle_cols + feature_sets["bio260_sp64"],
        "target_score_primary",
        train_mask,
        sample_weight=bad_weight,
    )
    pred_state, audit_state = V258.train_model(
        df, vehicle_cols + feature_sets["bio260_state_change"], "target_score_primary", train_mask
    )

    df["pred_selector_vehicle_hgb"] = pred_vehicle
    df["pred_selector_vehicle_bio260_sp32_hgb"] = pred_sp32
    df["pred_selector_vehicle_bio260_sp64_hgb"] = pred_sp64
    df["pred_selector_vehicle_bio260_sp64_badweighted_hgb"] = pred_sp64_bad
    df["pred_selector_vehicle_bio260_state_change_hgb"] = pred_state

    feature_audit = pd.DataFrame(
        [
            {"model": "selector_vehicle_hgb", "feature_n": len(vehicle_cols), "bio260_feature_n": 0},
            {
                "model": "selector_vehicle_bio260_sp32_hgb",
                "feature_n": len(vehicle_cols) + len(feature_sets["bio260_sp32"]),
                "bio260_feature_n": len(feature_sets["bio260_sp32"]),
            },
            {
                "model": "selector_vehicle_bio260_sp64_hgb",
                "feature_n": len(vehicle_cols) + len(feature_sets["bio260_sp64"]),
                "bio260_feature_n": len(feature_sets["bio260_sp64"]),
            },
            {
                "model": "selector_vehicle_bio260_sp64_badweighted_hgb",
                "feature_n": len(vehicle_cols) + len(feature_sets["bio260_sp64"]),
                "bio260_feature_n": len(feature_sets["bio260_sp64"]),
            },
            {
                "model": "selector_vehicle_bio260_state_change_hgb",
                "feature_n": len(vehicle_cols) + len(feature_sets["bio260_state_change"]),
                "bio260_feature_n": len(feature_sets["bio260_state_change"]),
            },
        ]
    )
    fill_audit = pd.concat(
        [
            audit_vehicle.assign(model="selector_vehicle_hgb"),
            audit_sp32.assign(model="selector_vehicle_bio260_sp32_hgb"),
            audit_sp64.assign(model="selector_vehicle_bio260_sp64_hgb"),
            audit_sp64_bad.assign(model="selector_vehicle_bio260_sp64_badweighted_hgb"),
            audit_state.assign(model="selector_vehicle_bio260_state_change_hgb"),
        ],
        ignore_index=True,
    )

    selections = [
        V258.select_by_strategy(df, None, "policy_keep_0ms_anchor"),
        V258.select_by_strategy(df, None, "policy_wait_to_latest_anchor"),
        V258.select_by_strategy(df, None, "oracle_best_anchor_upper_bound"),
        V258.select_by_strategy(df, "pred_selector_vehicle_hgb", "selector_vehicle_hgb"),
        V258.select_by_strategy(df, "pred_selector_vehicle_bio260_sp32_hgb", "selector_vehicle_bio260_sp32_hgb"),
        V258.select_by_strategy(df, "pred_selector_vehicle_bio260_sp64_hgb", "selector_vehicle_bio260_sp64_hgb"),
        V258.select_by_strategy(
            df,
            "pred_selector_vehicle_bio260_sp64_badweighted_hgb",
            "selector_vehicle_bio260_sp64_badweighted_hgb",
        ),
        V258.select_by_strategy(
            df,
            "pred_selector_vehicle_bio260_state_change_hgb",
            "selector_vehicle_bio260_state_change_hgb",
        ),
    ]
    selected = pd.concat(selections, ignore_index=True)
    summary = V258.summarize_selected(selected)
    figures = [plot_test_summary(summary)]
    v261_ref = load_v261_reference()

    export_cols = [
        "event_uid",
        "split",
        "candidate_delay_ms",
        "target_score_primary",
        "candidate_tail_rmse_v241",
        "pred_selector_vehicle_hgb",
        "pred_selector_vehicle_bio260_sp32_hgb",
        "pred_selector_vehicle_bio260_sp64_hgb",
        "pred_selector_vehicle_bio260_sp64_badweighted_hgb",
        "pred_selector_vehicle_bio260_state_change_hgb",
        "bio260_floor_delay_ms",
        "bio260_floor_status_ok",
    ]
    write_csv(df[[col for col in export_cols if col in df.columns]], TABLES / "v262_candidate_predictions_compact.csv")
    write_csv(selected, TABLES / "v262_selected_anchor_by_strategy.csv")
    write_csv(summary, TABLES / "v262_anchor_selector_summary.csv")
    write_csv(merge_audit, TABLES / "v262_bio260_merge_audit.csv")
    write_csv(feature_audit, TABLES / "v262_feature_block_audit.csv")
    write_csv(fill_audit, TABLES / "v262_feature_fill_audit.csv")
    write_csv(feature_selection, TABLES / "v262_feature_selection_audit.csv")
    if not v261_ref.empty:
        write_csv(v261_ref, TABLES / "v262_v261_badtop10_reference.csv")
    write_input_hashes()
    write_file_inventory()
    write_report(summary, merge_audit, feature_audit, feature_selection, v261_ref, figures)
    write_file_inventory()

    zip_ok = make_zip()
    guardrail = {
        "pass": bool(zip_ok and float(merge_audit["bio260_uses_post_observation_max"].iloc[0]) == 0.0),
        "zip_testzip": bool(zip_ok),
        "train_only_fit": True,
        "oracle_deployable": False,
        "bio260_delay_merge": "floor_coarse_delay_no_after_candidate",
        "bio260_uses_post_observation_max": float(merge_audit["bio260_uses_post_observation_max"].iloc[0]),
        "candidate_rows": int(len(df)),
        "event_n": int(df["event_uid"].nunique()),
        "feature_sets": {name: int(len(cols)) for name, cols in feature_sets.items()},
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not guardrail["pass"]:
        raise AssertionError("v262 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    focus = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["strategy"].isin(
            [
                "policy_keep_0ms_anchor",
                "selector_vehicle_hgb",
                "selector_vehicle_bio260_sp32_hgb",
                "selector_vehicle_bio260_sp64_hgb",
                "selector_vehicle_bio260_sp64_badweighted_hgb",
                "selector_vehicle_bio260_state_change_hgb",
                "policy_wait_to_latest_anchor",
                "oracle_best_anchor_upper_bound",
            ]
        )
    ]
    print(f"[v262] report={REPORTS / 'v262_subject_invariant_bio260_selector_cn.md'}")
    print(f"[v262] zip={ZIP_PATH}")
    print(
        focus[
            [
                "strategy",
                "selected_tail_rmse_mean",
                "delta_selected_minus_keep0_mean",
                "selected_delay_ms_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
