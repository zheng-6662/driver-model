#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v261 bio260 事件级生理 biomarker anchor selector。

本轮目的：
- v260 已经说明：事件级 ECG/EDA/RESP/EMG biomarker 对 bad_top10 有一点风险识别信号，
  但直接拼进未来行为分类/轨迹预测并不稳定；
- v261 因此只检查一个更窄的问题：这些生理状态能否帮助判断“当前样本要不要多看一点、
  以及在 fine-grid 候选锚点中选哪个锚点”。

边界：
- 复用 v247 fine-grid 候选锚点与 v241 replay 误差，不重新生成轨迹；
- 训练只使用 train split，val/test 只报告；
- 对每个 50ms 候选锚点，只合并不晚于该候选锚点的 floor coarse delay 生理特征，
  避免使用候选锚点之后的生理窗口；
- oracle 只作为上限，不作为可部署策略。
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

V258_SCRIPT = BASELINES / "scripts" / "stage03_v258_physio_augmented_anchor_selector_20260702.py"
V247_TABLE = (
    BASELINES
    / "v247_multi_resolution_best_anchor_discovery_20260630"
    / "tables"
    / "v247_selector_training_table.csv"
)
V247_FINE_TABLE = (
    BASELINES
    / "v247_multi_resolution_best_anchor_discovery_20260630"
    / "tables"
    / "v247_fine_anchor_candidate_table.csv"
)
BIO260_FEATURES = (
    BASELINES
    / "v260_event_biomarker_physio_rebuild_20260702"
    / "tables"
    / "v260_event_biomarker_features.csv"
)
V258_SUMMARY = (
    BASELINES
    / "v258_physio_augmented_anchor_selector_20260702"
    / "tables"
    / "v258_anchor_selector_summary.csv"
)

OUT = BASELINES / "v261_bio260_anchor_selector_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v261_bio260_anchor_selector_20260702_pack.zip"

SEED = 26102
COARSE_DELAYS = np.array([0, 200, 400, 600, 800, 1000], dtype=int)

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def load_v258_module():
    """复用 v258 中已经验证过的训练、选点、汇总函数，保证两轮可比。"""
    if not V258_SCRIPT.exists():
        raise FileNotFoundError(f"缺少 v258 脚本：{V258_SCRIPT}")
    spec = importlib.util.spec_from_file_location("v258_anchor_selector", V258_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 v258 脚本：{V258_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.TABLES = TABLES
    module.FIGURES = FIGURES
    module.REPORTS = REPORTS
    module.LOGS = LOGS
    module.ZIP_PATH = ZIP_PATH
    module.SEED = SEED
    return module


V258 = load_v258_module()


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


def floor_coarse_delay(delay_ms: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(delay_ms, errors="coerce").fillna(0).to_numpy(dtype=int)
    out = np.zeros(len(vals), dtype=int)
    for i, value in enumerate(vals):
        out[i] = int(COARSE_DELAYS[COARSE_DELAYS <= value].max())
    return out


def bio260_feature_columns(bio: pd.DataFrame) -> List[str]:
    """只保留事件生理信号特征，排除状态、采样率、baseline 覆盖行数等元数据。"""
    cols: List[str] = []
    for col in bio.columns:
        if not col.startswith("bio260_"):
            continue
        if not pd.api.types.is_numeric_dtype(bio[col]):
            continue
        if col in {
            "bio260_sample_hz",
            "bio260_recording_duration_s",
            "bio260_uses_post_observation",
            "bio260_floor_delay_ms",
        }:
            continue
        if "baseline" in col:
            continue
        if col.endswith("_rows") or col.endswith("_duration_s"):
            continue
        cols.append(col)
    return cols


def load_augmented_table() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not V247_TABLE.exists():
        raise FileNotFoundError(f"缺少 v247 selector training table：{V247_TABLE}")
    if not V247_FINE_TABLE.exists():
        raise FileNotFoundError(f"缺少 v247 fine candidate table：{V247_FINE_TABLE}")
    if not BIO260_FEATURES.exists():
        raise FileNotFoundError(f"缺少 v260 bio260 features：{BIO260_FEATURES}")

    cand = pd.read_csv(V247_TABLE, encoding="utf-8-sig", low_memory=False)
    flag_cols = {
        "candidate_row_idx",
        "subject",
        "recording",
        "bad_top10_split_v241",
        "very_bad_top5_split_v241",
        "normal_curve_current0",
        "observe_later_like_current0",
        "strong_steer_current0",
        "reverse_current0",
        "current_0ms_tail_rmse_v241",
    }
    flags = pd.read_csv(
        V247_FINE_TABLE,
        encoding="utf-8-sig",
        low_memory=False,
        usecols=lambda c: c in flag_cols,
    ).drop_duplicates("candidate_row_idx")
    cand = cand.merge(flags, on="candidate_row_idx", how="left")

    bio = pd.read_csv(BIO260_FEATURES, encoding="utf-8-sig", low_memory=False)
    bio["bio260_floor_delay_ms"] = pd.to_numeric(bio["delay_ms"], errors="coerce").astype("Int64")
    bio_cols = bio260_feature_columns(bio)
    bio_keep = [
        "event_uid",
        "bio260_floor_delay_ms",
        "bio260_status",
        "bio260_uses_post_observation",
    ] + bio_cols
    bio_small = bio[bio_keep].copy()
    bio_small = bio_small.drop_duplicates(["event_uid", "bio260_floor_delay_ms"], keep="first")
    bio_small = bio_small.rename(columns={col: f"floor_{col}" for col in bio_cols})

    cand["bio260_floor_delay_ms"] = floor_coarse_delay(cand["candidate_delay_ms"])
    out = cand.merge(bio_small, on=["event_uid", "bio260_floor_delay_ms"], how="left")
    out["bio260_floor_status_ok"] = out["bio260_status"].astype(str).eq("ok").astype(float)
    out["bio260_floor_uses_post_observation"] = pd.to_numeric(
        out["bio260_uses_post_observation"], errors="coerce"
    )

    feature_cols = [f"floor_{col}" for col in bio_cols]
    feature_matrix = out[feature_cols].to_numpy(dtype=float) if feature_cols else np.empty((len(out), 0))
    missing_rate = float(np.isnan(feature_matrix).mean()) if feature_matrix.size else 0.0
    audit = pd.DataFrame(
        [
            {
                "candidate_rows": int(len(out)),
                "event_n": int(out["event_uid"].nunique()),
                "bio260_source_rows": int(len(bio)),
                "bio260_source_event_n": int(bio["event_uid"].nunique()),
                "bio260_feature_n": int(len(bio_cols)),
                "bio260_merge_ok_rate": float(out["bio260_floor_status_ok"].mean()),
                "bio260_feature_missing_rate_after_merge": missing_rate,
                "bio260_uses_post_observation_max": float(
                    np.nanmax(out["bio260_floor_uses_post_observation"].to_numpy(dtype=float))
                )
                if out["bio260_floor_uses_post_observation"].notna().any()
                else 0.0,
                "candidate_delay_min": int(pd.to_numeric(out["candidate_delay_ms"], errors="coerce").min()),
                "candidate_delay_max": int(pd.to_numeric(out["candidate_delay_ms"], errors="coerce").max()),
            }
        ]
    )
    return out, audit


def feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    vehicle_prefixes = ("candidate_delay", "hist_", "road_", "phase_", "instability_penalty")
    vehicle_cols: List[str] = []
    for col in df.columns:
        if col in {"candidate_delay_ms", "candidate_delay_s"} or col.startswith(vehicle_prefixes):
            if pd.api.types.is_numeric_dtype(df[col]):
                vehicle_cols.append(col)
    bio_cols = [
        col
        for col in df.columns
        if col.startswith("floor_bio260_") and pd.api.types.is_numeric_dtype(df[col])
    ] + ["bio260_floor_status_ok"]
    vehicle_cols = list(dict.fromkeys(vehicle_cols))
    bio_cols = list(dict.fromkeys(bio_cols))
    return vehicle_cols, bio_cols


def plot_test_summary(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v261_anchor_selector_test_badtop10.png"
    order = [
        "policy_keep_0ms_anchor",
        "selector_vehicle_hgb",
        "selector_bio260_hgb",
        "selector_vehicle_bio260_hgb",
        "selector_vehicle_bio260_badweighted_hgb",
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
    fig, ax = plt.subplots(figsize=(13, 5.2))
    x = np.arange(len(sub))
    colors = ["#9CA3AF", "#4C78A8", "#72B7B2", "#59A14F", "#F28E2B", "#E15759", "#B07AA1"]
    ax.bar(x, sub["selected_tail_rmse_mean"], color=colors[: len(sub)])
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in sub["strategy"]], fontsize=8)
    ax.set_ylabel("test bad_top10 selected tail RMSE")
    ax.set_title("v261: bio260 事件级生理是否能帮助 bad_top10 锚点/等待选择")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def load_v258_reference() -> pd.DataFrame:
    if not V258_SUMMARY.exists():
        return pd.DataFrame()
    ref = pd.read_csv(V258_SUMMARY, encoding="utf-8-sig", low_memory=False)
    keep = ref[
        ref["split"].eq("test")
        & ref["event_group"].eq("bad_top10")
        & ref["strategy"].isin(
            [
                "policy_keep_0ms_anchor",
                "selector_vehicle_hgb",
                "selector_vehicle_physio_hgb",
                "selector_vehicle_physio_badweighted_hgb",
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
    keep.insert(0, "source", "v258_physio200_ref")
    return keep


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v247_selector_training_table", V247_TABLE),
        ("v247_fine_anchor_candidate_table", V247_FINE_TABLE),
        ("v260_event_biomarker_features", BIO260_FEATURES),
        ("v258_reused_script", V258_SCRIPT),
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
    v258_ref: pd.DataFrame,
    figures: List[Path],
) -> None:
    lines: List[str] = []
    lines.append("# v261 bio260 事件级生理 anchor selector")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("- v260 说明事件级生理 biomarker 对 bad_top10 有少量风险识别能力，但直接预测未来行为不稳定。")
    lines.append("- v261 不再把生理当成轨迹预测输入，而是只让它参与锚点/等待选择。")
    lines.append("- 如果 bio260 selector 不能超过 vehicle selector，说明当前生理信号还不足以弥补锚点前车辆信息不足。")
    lines.append("")
    lines.append("## 方法")
    lines.append("")
    lines.append("- 候选锚点：复用 v247 的 50ms fine-grid 候选。")
    lines.append("- 监督目标：复用 v247 的 `target_score_primary`，也就是候选锚点 replay 后的综合误差目标。")
    lines.append("- 生理输入：使用 v260 从 200Hz 波形重构的 ECG/EDA/RESP/EMG 事件级 biomarker。")
    lines.append("- 防泄漏：候选锚点为 50ms 粒度，生理按 floor coarse delay 合并，只使用不晚于候选锚点的生理窗口。")
    lines.append("- 训练边界：只在 train split 拟合，val/test 只报告；oracle 只作上限。")
    lines.append("")
    lines.append("## 合并与特征审计")
    lines.append("")
    lines.append(merge_audit.to_markdown(index=False))
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
                "selector_bio260_hgb",
                "selector_vehicle_bio260_hgb",
                "selector_vehicle_bio260_badweighted_hgb",
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
    if not v258_ref.empty:
        lines.append("## v258 physio200 参考")
        lines.append("")
        lines.append(v258_ref.to_markdown(index=False))
        lines.append("")
    lines.append("## 判读")
    lines.append("")
    bad = focus[focus["event_group"].eq("bad_top10")].copy()
    score_by_strategy: Dict[str, float] = {}
    delay_by_strategy: Dict[str, float] = {}
    for strategy in [
        "policy_keep_0ms_anchor",
        "selector_vehicle_hgb",
        "selector_bio260_hgb",
        "selector_vehicle_bio260_hgb",
        "selector_vehicle_bio260_badweighted_hgb",
        "policy_wait_to_latest_anchor",
        "oracle_best_anchor_upper_bound",
    ]:
        row = bad[bad["strategy"].eq(strategy)]
        if len(row):
            score = float(row["selected_tail_rmse_mean"].iloc[0])
            delta = float(row["delta_selected_minus_keep0_mean"].iloc[0])
            delay = float(row["selected_delay_ms_mean"].iloc[0])
            score_by_strategy[strategy] = score
            delay_by_strategy[strategy] = delay
            lines.append(f"- bad_top10 / {strategy}: tail={score:.4f}, delta_keep0={delta:+.4f}, delay={delay:.1f}ms.")

    vehicle = score_by_strategy.get("selector_vehicle_hgb", np.nan)
    bio = score_by_strategy.get("selector_vehicle_bio260_hgb", np.nan)
    bio_bad = score_by_strategy.get("selector_vehicle_bio260_badweighted_hgb", np.nan)
    latest = score_by_strategy.get("policy_wait_to_latest_anchor", np.nan)
    keep0 = score_by_strategy.get("policy_keep_0ms_anchor", np.nan)
    if np.isfinite(vehicle) and np.isfinite(bio_bad):
        lines.append("")
        if bio_bad < vehicle:
            lines.append(
                f"- 结论：bad-weighted vehicle+bio260 比 vehicle selector 低 {vehicle - bio_bad:.4f}，说明 bio260 在差样本等待/锚点选择上有增益。"
            )
        else:
            lines.append(
                f"- 结论：bad-weighted vehicle+bio260 比 vehicle selector 高 {bio_bad - vehicle:.4f}，说明 bio260 尚不能稳定改善差样本锚点选择。"
            )
    if np.isfinite(latest) and np.isfinite(bio_bad):
        lines.append(
            f"- 与固定 latest 比：bad-weighted vehicle+bio260 tail={bio_bad:.4f}，latest tail={latest:.4f}；如果仍高于 latest，说明最简单的“多看一点”仍比生理驱动选择更稳。"
        )
    if np.isfinite(keep0) and np.isfinite(bio_bad):
        lines.append(
            f"- 与 0ms 原锚点比：bad-weighted vehicle+bio260 改变量为 {bio_bad - keep0:+.4f}；这是判断是否弥补锚点前信息不足的核心数字。"
        )
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v261_bio260_anchor_selector_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v261] bio260 anchor selector")
    clean_out_dir()
    np.random.seed(SEED)

    df, merge_audit = load_augmented_table()
    vehicle_cols, bio_cols = feature_columns(df)
    train_mask = df["split"].astype(str).eq("train").to_numpy()
    bad_weight = 1.0 + 4.0 * df["bad_top10_split_v241"].fillna(False).astype(bool).to_numpy(dtype=float)

    pred_vehicle, audit_vehicle = V258.train_model(df, vehicle_cols, "target_score_primary", train_mask)
    pred_bio, audit_bio = V258.train_model(df, bio_cols, "target_score_primary", train_mask)
    pred_vehicle_bio, audit_vb = V258.train_model(
        df, vehicle_cols + bio_cols, "target_score_primary", train_mask
    )
    pred_vehicle_bio_bad, audit_vbb = V258.train_model(
        df,
        vehicle_cols + bio_cols,
        "target_score_primary",
        train_mask,
        sample_weight=bad_weight,
    )

    df["pred_selector_vehicle_hgb"] = pred_vehicle
    df["pred_selector_bio260_hgb"] = pred_bio
    df["pred_selector_vehicle_bio260_hgb"] = pred_vehicle_bio
    df["pred_selector_vehicle_bio260_badweighted_hgb"] = pred_vehicle_bio_bad

    feature_audit = pd.DataFrame(
        [
            {"model": "selector_vehicle_hgb", "feature_n": len(vehicle_cols), "bio260_feature_n": 0},
            {"model": "selector_bio260_hgb", "feature_n": len(bio_cols), "bio260_feature_n": len(bio_cols)},
            {
                "model": "selector_vehicle_bio260_hgb",
                "feature_n": len(vehicle_cols) + len(bio_cols),
                "bio260_feature_n": len(bio_cols),
            },
            {
                "model": "selector_vehicle_bio260_badweighted_hgb",
                "feature_n": len(vehicle_cols) + len(bio_cols),
                "bio260_feature_n": len(bio_cols),
            },
        ]
    )
    fill_audit = pd.concat(
        [
            audit_vehicle.assign(model="selector_vehicle_hgb"),
            audit_bio.assign(model="selector_bio260_hgb"),
            audit_vb.assign(model="selector_vehicle_bio260_hgb"),
            audit_vbb.assign(model="selector_vehicle_bio260_badweighted_hgb"),
        ],
        ignore_index=True,
    )

    selections = [
        V258.select_by_strategy(df, None, "policy_keep_0ms_anchor"),
        V258.select_by_strategy(df, None, "policy_wait_to_latest_anchor"),
        V258.select_by_strategy(df, None, "oracle_best_anchor_upper_bound"),
        V258.select_by_strategy(df, "pred_selector_vehicle_hgb", "selector_vehicle_hgb"),
        V258.select_by_strategy(df, "pred_selector_bio260_hgb", "selector_bio260_hgb"),
        V258.select_by_strategy(df, "pred_selector_vehicle_bio260_hgb", "selector_vehicle_bio260_hgb"),
        V258.select_by_strategy(
            df,
            "pred_selector_vehicle_bio260_badweighted_hgb",
            "selector_vehicle_bio260_badweighted_hgb",
        ),
    ]
    selected = pd.concat(selections, ignore_index=True)
    summary = V258.summarize_selected(selected)
    figures = [plot_test_summary(summary)]
    v258_ref = load_v258_reference()

    export_cols = [
        "event_uid",
        "split",
        "candidate_delay_ms",
        "target_score_primary",
        "candidate_tail_rmse_v241",
        "pred_selector_vehicle_hgb",
        "pred_selector_bio260_hgb",
        "pred_selector_vehicle_bio260_hgb",
        "pred_selector_vehicle_bio260_badweighted_hgb",
        "bio260_floor_delay_ms",
        "bio260_floor_status_ok",
    ]
    write_csv(df[[col for col in export_cols if col in df.columns]], TABLES / "v261_candidate_predictions_compact.csv")
    write_csv(selected, TABLES / "v261_selected_anchor_by_strategy.csv")
    write_csv(summary, TABLES / "v261_anchor_selector_summary.csv")
    write_csv(merge_audit, TABLES / "v261_bio260_merge_audit.csv")
    write_csv(feature_audit, TABLES / "v261_feature_block_audit.csv")
    write_csv(fill_audit, TABLES / "v261_feature_fill_audit.csv")
    if not v258_ref.empty:
        write_csv(v258_ref, TABLES / "v261_v258_badtop10_reference.csv")
    write_input_hashes()
    write_file_inventory()
    write_report(summary, merge_audit, feature_audit, v258_ref, figures)
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
        "bio260_feature_n": int(len(bio_cols)),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not guardrail["pass"]:
        raise AssertionError("v261 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    focus = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["strategy"].isin(
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
    ]
    print(f"[v261] report={REPORTS / 'v261_bio260_anchor_selector_cn.md'}")
    print(f"[v261] zip={ZIP_PATH}")
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
