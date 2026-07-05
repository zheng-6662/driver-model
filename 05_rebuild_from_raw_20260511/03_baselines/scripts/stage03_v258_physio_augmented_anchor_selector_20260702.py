#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v258 生理增强 anchor selector。

目的：
- v245-v247 证明等待/重锚定对差样本有大上限；
- v254b-v257 证明生理直接预测轨迹或候选未来都不行；
- 本轮检查生理是否能参与“什么时候多看一点”的 anchor 选择。

边界：
- 复用 v247 fine-grid 候选锚点，不重新推理 v241；
- 训练只用 train split，val/test 只报告；
- 生理特征按 candidate_delay_ms 的 floor coarse delay 合并，避免使用候选锚点之后的生理窗口；
- oracle 只作为上限，不作为可部署策略。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

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
V254B_FEATURES = (
    BASELINES
    / "v254b_physio_200hz_event_representation_20260702"
    / "tables"
    / "v254b_event_physio200_features.csv"
)

OUT = BASELINES / "v258_physio_augmented_anchor_selector_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v258_physio_augmented_anchor_selector_20260702_pack.zip"

SEED = 25802
COARSE_DELAYS = np.array([0, 200, 400, 600, 800, 1000], dtype=int)

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


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
    for i, v in enumerate(vals):
        out[i] = int(COARSE_DELAYS[COARSE_DELAYS <= v].max())
    return out


def finite_nanmedian(x: np.ndarray, axis: int = 0) -> np.ndarray:
    with np.errstate(all="ignore"):
        med = np.nanmedian(x, axis=axis)
    med = np.asarray(med, dtype=float)
    med[~np.isfinite(med)] = 0.0
    return med


def fit_fill_scale(x: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_x = x[train_mask]
    med = finite_nanmedian(train_x, axis=0)
    filled = np.where(np.isfinite(x), x, med[None, :])
    mean = np.nanmean(filled[train_mask], axis=0)
    std = np.nanstd(filled[train_mask], axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    z = (filled - mean[None, :]) / std[None, :]
    return z.astype(np.float32), med, std


def load_augmented_table() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not V247_TABLE.exists():
        raise FileNotFoundError(f"缺少 v247 selector training table：{V247_TABLE}")
    if not V247_FINE_TABLE.exists():
        raise FileNotFoundError(f"缺少 v247 fine candidate table：{V247_FINE_TABLE}")
    if not V254B_FEATURES.exists():
        raise FileNotFoundError(f"缺少 v254b physio features：{V254B_FEATURES}")
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
    phys = pd.read_csv(V254B_FEATURES, encoding="utf-8-sig", low_memory=False)
    phys["physio_floor_delay_ms"] = pd.to_numeric(phys["delay_ms"], errors="coerce").astype("Int64")
    phys_cols = [
        c
        for c in phys.columns
        if c.startswith("physio200_")
        and pd.api.types.is_numeric_dtype(phys[c])
        and ("_z_" in c or c.endswith("_index") or "burst_rate" in c)
        and any(sig in c for sig in ["HR_bpm", "EMG_RMS", "EMG_filt200", "EDA_Phasic", "EDA_Tonic", "RESP_filt200", "ECG_filt200"])
    ]
    keep = ["event_uid", "physio_floor_delay_ms", "physio200_status"] + phys_cols
    phys_small = phys[keep].copy()
    phys_small = phys_small.rename(columns={c: f"floor_{c}" for c in phys_cols})
    cand["physio_floor_delay_ms"] = floor_coarse_delay(cand["candidate_delay_ms"])
    out = cand.merge(phys_small, on=["event_uid", "physio_floor_delay_ms"], how="left")
    out["physio_floor_status_ok"] = out["physio200_status"].astype(str).eq("ok").astype(float)
    audit = pd.DataFrame(
        [
            {
                "candidate_rows": int(len(out)),
                "event_n": int(out["event_uid"].nunique()),
                "physio_feature_n": int(len(phys_cols)),
                "physio_merge_ok_rate": float(out["physio_floor_status_ok"].mean()),
                "candidate_delay_min": int(pd.to_numeric(out["candidate_delay_ms"], errors="coerce").min()),
                "candidate_delay_max": int(pd.to_numeric(out["candidate_delay_ms"], errors="coerce").max()),
            }
        ]
    )
    return out, audit


def feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    vehicle_prefixes = ("candidate_delay", "hist_", "road_", "phase_", "instability_penalty")
    vehicle_cols = []
    for c in df.columns:
        if c in {"candidate_delay_ms", "candidate_delay_s"} or c.startswith(vehicle_prefixes):
            if pd.api.types.is_numeric_dtype(df[c]):
                vehicle_cols.append(c)
    physio_cols = [
        c
        for c in df.columns
        if c.startswith("floor_physio200_") and pd.api.types.is_numeric_dtype(df[c])
    ] + ["physio_floor_status_ok"]
    # 去重并保持顺序
    vehicle_cols = list(dict.fromkeys(vehicle_cols))
    physio_cols = list(dict.fromkeys(physio_cols))
    return vehicle_cols, physio_cols


def train_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_mask: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    x = df[feature_cols].to_numpy(dtype=float)
    xz, med, std = fit_fill_scale(x, train_mask)
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    good = train_mask & np.isfinite(y)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=360,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.08,
        random_state=SEED,
    )
    if sample_weight is not None:
        model.fit(xz[good], y[good], sample_weight=sample_weight[good])
    else:
        model.fit(xz[good], y[good])
    pred = model.predict(xz)
    audit = pd.DataFrame({"feature": feature_cols, "fill_median": med, "scale_std": std})
    return pred.astype(float), audit


def select_by_strategy(df: pd.DataFrame, pred_col: str | None, strategy: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for event_uid, g in df.groupby("event_uid", sort=False):
        g = g.copy()
        keep0 = g.loc[pd.to_numeric(g["candidate_delay_ms"], errors="coerce").idxmin()]
        latest = g.loc[pd.to_numeric(g["candidate_delay_ms"], errors="coerce").idxmax()]
        oracle = g.loc[pd.to_numeric(g["candidate_tail_rmse_v241"], errors="coerce").idxmin()]
        if pred_col is None:
            if strategy == "policy_keep_0ms_anchor":
                chosen = keep0
            elif strategy == "policy_wait_to_latest_anchor":
                chosen = latest
            elif strategy == "oracle_best_anchor_upper_bound":
                chosen = oracle
            else:
                raise ValueError(strategy)
        else:
            chosen = g.loc[pd.to_numeric(g[pred_col], errors="coerce").idxmin()]
        row = {
            "strategy": strategy,
            "event_uid": str(event_uid),
            "split": str(chosen["split"]),
            "subject": str(chosen.get("subject", "")),
            "recording": str(chosen.get("recording", "")),
            "selected_delay_ms": int(chosen["candidate_delay_ms"]),
            "selected_tail_rmse_v241": float(chosen["candidate_tail_rmse_v241"]),
            "keep0_tail_rmse_v241": float(keep0["candidate_tail_rmse_v241"]),
            "latest_tail_rmse_v241": float(latest["candidate_tail_rmse_v241"]),
            "oracle_tail_rmse_v241": float(oracle["candidate_tail_rmse_v241"]),
            "delta_selected_minus_keep0": float(chosen["candidate_tail_rmse_v241"] - keep0["candidate_tail_rmse_v241"]),
            "delta_selected_minus_latest": float(chosen["candidate_tail_rmse_v241"] - latest["candidate_tail_rmse_v241"]),
            "bad_top10": bool(keep0["bad_top10_split_v241"]),
            "very_bad_top5": bool(keep0["very_bad_top5_split_v241"]),
            "normal": bool(keep0["normal_curve_current0"]),
            "observe_later_like": bool(keep0["observe_later_like_current0"]),
            "strong_steer": bool(keep0["strong_steer_current0"]),
            "reverse": bool(keep0["reverse_current0"]),
            "early_best_after_400": bool(int(oracle["candidate_delay_ms"]) >= 400),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_selected(selected: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    buckets = [
        ("all", np.ones(len(selected), dtype=bool)),
        ("bad_top10", selected["bad_top10"].astype(bool).to_numpy()),
        ("very_bad_top5", selected["very_bad_top5"].astype(bool).to_numpy()),
        ("normal", selected["normal"].astype(bool).to_numpy()),
        ("observe_later_like", selected["observe_later_like"].astype(bool).to_numpy()),
        ("strong_steer", selected["strong_steer"].astype(bool).to_numpy()),
        ("early_best_after_400", selected["early_best_after_400"].astype(bool).to_numpy()),
    ]
    for split_name, split_mask in [("train", selected["split"].eq("train").to_numpy()), ("val", selected["split"].eq("val").to_numpy()), ("test", selected["split"].eq("test").to_numpy())]:
        for bucket, bucket_mask in buckets:
            mask = split_mask & bucket_mask
            if int(mask.sum()) == 0:
                continue
            sub = selected[mask]
            for strategy, g in sub.groupby("strategy", sort=False):
                rows.append(
                    {
                        "split": split_name,
                        "event_group": bucket,
                        "strategy": strategy,
                        "n": int(len(g)),
                        "selected_tail_rmse_mean": float(g["selected_tail_rmse_v241"].mean()),
                        "keep0_tail_rmse_mean": float(g["keep0_tail_rmse_v241"].mean()),
                        "latest_tail_rmse_mean": float(g["latest_tail_rmse_v241"].mean()),
                        "oracle_tail_rmse_mean": float(g["oracle_tail_rmse_v241"].mean()),
                        "delta_selected_minus_keep0_mean": float(g["delta_selected_minus_keep0"].mean()),
                        "delta_selected_minus_latest_mean": float(g["delta_selected_minus_latest"].mean()),
                        "improve_rate_vs_keep0": float((g["delta_selected_minus_keep0"] < 0).mean()),
                        "selected_delay_ms_mean": float(g["selected_delay_ms"].mean()),
                        "selected_latest_rate": float((g["selected_delay_ms"] >= 1000).mean()),
                    }
                )
    return pd.DataFrame(rows)


def plot_test_summary(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v258_anchor_selector_test_badtop10.png"
    sub = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["strategy"].isin(
            [
                "policy_keep_0ms_anchor",
                "policy_wait_to_latest_anchor",
                "selector_vehicle_hgb",
                "selector_vehicle_physio_hgb",
                "selector_vehicle_physio_badweighted_hgb",
                "oracle_best_anchor_upper_bound",
            ]
        )
    ].copy()
    if sub.empty:
        return path
    order = [
        "policy_keep_0ms_anchor",
        "selector_vehicle_hgb",
        "selector_vehicle_physio_hgb",
        "selector_vehicle_physio_badweighted_hgb",
        "policy_wait_to_latest_anchor",
        "oracle_best_anchor_upper_bound",
    ]
    sub["strategy"] = pd.Categorical(sub["strategy"], categories=order, ordered=True)
    sub = sub.sort_values("strategy")
    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(sub))
    ax.bar(x, sub["selected_tail_rmse_mean"], color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in sub["strategy"]], fontsize=8)
    ax.set_ylabel("test bad_top10 selected tail RMSE")
    ax.set_title("v258: 生理增强 anchor selector 是否优于车辆 selector / 固定等待")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for label, path in [("v247_selector_training_table", V247_TABLE), ("v254b_features", V254B_FEATURES)]:
        rows.append({"label": label, "path": str(path), "exists": bool(path.exists()), "sha256": file_sha256(path) if path.exists() and path.is_file() else ""})
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


def write_report(summary: pd.DataFrame, merge_audit: pd.DataFrame, feature_audit: pd.DataFrame, figures: List[Path]) -> None:
    lines: List[str] = []
    lines.append("# v258 生理增强 anchor selector")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("- v245-v247 证明等待/重锚定对差样本有明确上限。")
    lines.append("- v254b-v257 证明生理直接预测轨迹、候选未来或个体化记忆都没有本质改善。")
    lines.append("- v258 检查最后一个较合理的生理用途：让生理参与判断什么时候等待/重锚定。")
    lines.append("")
    lines.append("## 方法")
    lines.append("")
    lines.append("- 复用 v247 的 50ms fine-grid 候选锚点和 v241 replay 误差。")
    lines.append("- 候选特征 = v247 车辆/road/phase 特征 + v254b 生理特征。")
    lines.append("- 生理特征用 floor coarse delay 合并，确保不使用候选锚点之后的生理。")
    lines.append("- 训练目标是候选 `target_score_primary`，训练 split 训练，val/test 只报告。")
    lines.append("")
    lines.append("## 合并与特征")
    lines.append("")
    lines.append(merge_audit.to_markdown(index=False))
    lines.append("")
    lines.append(feature_audit.to_markdown(index=False))
    lines.append("")
    lines.append("## Test 关键结果")
    lines.append("")
    focus = summary[
        summary["split"].eq("test")
        & summary["event_group"].isin(["all", "bad_top10", "early_best_after_400", "normal", "strong_steer", "observe_later_like"])
        & summary["strategy"].isin(
            [
                "policy_keep_0ms_anchor",
                "policy_wait_to_latest_anchor",
                "selector_vehicle_hgb",
                "selector_vehicle_physio_hgb",
                "selector_vehicle_physio_badweighted_hgb",
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
    lines.append("## 判读")
    lines.append("")
    bad = focus[focus["event_group"].eq("bad_top10")].copy()
    for strategy in ["policy_keep_0ms_anchor", "selector_vehicle_hgb", "selector_vehicle_physio_hgb", "selector_vehicle_physio_badweighted_hgb", "policy_wait_to_latest_anchor", "oracle_best_anchor_upper_bound"]:
        r = bad[bad["strategy"].eq(strategy)]
        if len(r):
            lines.append(f"- bad_top10 / {strategy}: tail={float(r['selected_tail_rmse_mean'].iloc[0]):.4f}, delta_keep0={float(r['delta_selected_minus_keep0_mean'].iloc[0]):+.4f}.")
    lines.append("- 若 vehicle+physio selector 不明显优于 vehicle selector 和固定 wait-latest，则当前生理不能承担等待决策。")
    lines.append("- 若 selector 仍弱于 wait-latest，说明等待上限主要来自多观察车辆状态，而不是生理状态判断。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v258_physio_augmented_anchor_selector_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v258] physio augmented anchor selector")
    clean_out_dir()
    np.random.seed(SEED)

    df, merge_audit = load_augmented_table()
    vehicle_cols, physio_cols = feature_columns(df)
    train_mask = df["split"].astype(str).eq("train").to_numpy()
    y = pd.to_numeric(df["target_score_primary"], errors="coerce").to_numpy(dtype=float)
    bad_weight = 1.0 + 3.0 * df["bad_top10_split_v241"].fillna(False).astype(bool).to_numpy(dtype=float)

    pred_vehicle, audit_vehicle = train_model(df, vehicle_cols, "target_score_primary", train_mask)
    pred_vehicle_physio, audit_vp = train_model(df, vehicle_cols + physio_cols, "target_score_primary", train_mask)
    pred_vehicle_physio_bad, audit_vpb = train_model(df, vehicle_cols + physio_cols, "target_score_primary", train_mask, sample_weight=bad_weight)

    df["pred_selector_vehicle_hgb"] = pred_vehicle
    df["pred_selector_vehicle_physio_hgb"] = pred_vehicle_physio
    df["pred_selector_vehicle_physio_badweighted_hgb"] = pred_vehicle_physio_bad

    feature_audit = pd.DataFrame(
        [
            {"model": "selector_vehicle_hgb", "feature_n": len(vehicle_cols), "physio_feature_n": 0},
            {"model": "selector_vehicle_physio_hgb", "feature_n": len(vehicle_cols) + len(physio_cols), "physio_feature_n": len(physio_cols)},
            {"model": "selector_vehicle_physio_badweighted_hgb", "feature_n": len(vehicle_cols) + len(physio_cols), "physio_feature_n": len(physio_cols)},
        ]
    )
    fill_audit = pd.concat(
        [
            audit_vehicle.assign(model="selector_vehicle_hgb"),
            audit_vp.assign(model="selector_vehicle_physio_hgb"),
            audit_vpb.assign(model="selector_vehicle_physio_badweighted_hgb"),
        ],
        ignore_index=True,
    )

    selections = [
        select_by_strategy(df, None, "policy_keep_0ms_anchor"),
        select_by_strategy(df, None, "policy_wait_to_latest_anchor"),
        select_by_strategy(df, None, "oracle_best_anchor_upper_bound"),
        select_by_strategy(df, "pred_selector_vehicle_hgb", "selector_vehicle_hgb"),
        select_by_strategy(df, "pred_selector_vehicle_physio_hgb", "selector_vehicle_physio_hgb"),
        select_by_strategy(df, "pred_selector_vehicle_physio_badweighted_hgb", "selector_vehicle_physio_badweighted_hgb"),
    ]
    selected = pd.concat(selections, ignore_index=True)
    summary = summarize_selected(selected)
    figures = [plot_test_summary(summary)]

    export_cols = [
        "event_uid",
        "split",
        "candidate_delay_ms",
        "target_score_primary",
        "candidate_tail_rmse_v241",
        "pred_selector_vehicle_hgb",
        "pred_selector_vehicle_physio_hgb",
        "pred_selector_vehicle_physio_badweighted_hgb",
        "physio_floor_delay_ms",
        "physio_floor_status_ok",
    ]
    write_csv(df[[c for c in export_cols if c in df.columns]], TABLES / "v258_candidate_predictions_compact.csv")
    write_csv(selected, TABLES / "v258_selected_anchor_by_strategy.csv")
    write_csv(summary, TABLES / "v258_anchor_selector_summary.csv")
    write_csv(merge_audit, TABLES / "v258_physio_merge_audit.csv")
    write_csv(feature_audit, TABLES / "v258_feature_block_audit.csv")
    write_csv(fill_audit, TABLES / "v258_feature_fill_audit.csv")
    write_input_hashes()
    write_file_inventory()
    write_report(summary, merge_audit, feature_audit, figures)
    write_file_inventory()
    zip_ok = make_zip()
    guardrail = {
        "pass": bool(zip_ok),
        "zip_testzip": bool(zip_ok),
        "train_only_fit": True,
        "oracle_deployable": False,
        "physio_delay_merge": "floor_coarse_delay_no_after_candidate",
        "candidate_rows": int(len(df)),
        "event_n": int(df["event_uid"].nunique()),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not guardrail["pass"]:
        raise AssertionError("v258 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    focus = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["strategy"].isin(["policy_keep_0ms_anchor", "selector_vehicle_hgb", "selector_vehicle_physio_hgb", "selector_vehicle_physio_badweighted_hgb", "policy_wait_to_latest_anchor", "oracle_best_anchor_upper_bound"])
    ]
    print(f"[v258] report={REPORTS / 'v258_physio_augmented_anchor_selector_cn.md'}")
    print(f"[v258] zip={ZIP_PATH}")
    print(focus[["strategy", "selected_tail_rmse_mean", "delta_selected_minus_keep0_mean", "selected_delay_ms_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
