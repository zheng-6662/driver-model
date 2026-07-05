#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v286 raw-200Hz subject-aware online calibration.

本轮目的：
- v285 证明 raw 200Hz shape-state 在 subject-disjoint route gate 中仍失败；
- v286 单独测试另一个任务边界：如果允许同一驾驶员更早事件的反馈，
  v285 底层生理形态特征是否能在 online subject-aware calibration 中提供额外价值。

重要边界：
- 这不是正式 subject-disjoint 结果；
- global 模型只用 train split 训练；
- val/test 的在线校准只允许用同 split、同 subject、当前事件之前的历史事件；
- 当前事件未来、当前事件之后历史、test 后验阈值都不进入输入；
- 核心比较是 raw285 physiology KNN 是否优于纯 subject mean / recent 校准。
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

V258_SCRIPT = SCRIPTS / "stage03_v258_physio_augmented_anchor_selector_20260702.py"
V263_EVENTS = BASELINES / "v263_bio260_wait_gate_20260702" / "tables" / "v263_event_wait_gate_predictions.csv"
V285_FEATURES = BASELINES / "v285_raw200_shape_state_route_gate_20260702" / "tables" / "v285_raw200_shape_state_features.csv"
V285_SCALER = BASELINES / "v285_raw200_shape_state_route_gate_20260702" / "tables" / "v285_train_scaler_audit.csv"
V285_GUARDRAIL = BASELINES / "v285_raw200_shape_state_route_gate_20260702" / "logs" / "guardrail_check.json"

OUT = BASELINES / "v286_raw200_online_subject_calibration_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v286_raw200_online_subject_calibration_20260702_pack.zip"

SEED = 28602
MIN_HISTORY = 3
KNN_K = 10

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    """按路径导入 v258 的训练和汇总工具。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V258 = import_module_from_path("stage03_v258_for_v286", V258_SCRIPT)


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


def event_order_from_uid(uid: str) -> int:
    """从 event_uid 尾部编号提取同 recording 内顺序。"""

    match = re.search(r"_(\d+)$", str(uid))
    if match:
        return int(match.group(1))
    return 0


def vehicle_feature_columns(df: pd.DataFrame) -> List[str]:
    """v263 中用于 wait gate 的车辆/道路/相位特征。"""

    prefixes = ("candidate_delay", "hist_", "road_", "phase_", "instability_penalty")
    cols: List[str] = []
    for col in df.columns:
        if col in {"candidate_delay_ms", "candidate_delay_s"} or col.startswith(prefixes):
            if pd.api.types.is_numeric_dtype(df[col]):
                cols.append(col)
    return list(dict.fromkeys(cols))


def load_v285_feature_columns() -> Tuple[List[str], pd.DataFrame]:
    """复用 v285 train-only 选择出的特征集合，避免在 v286 用 test 重选特征。"""

    if not V285_SCALER.exists():
        raise FileNotFoundError(f"缺少 v285 scaler audit：{V285_SCALER}")
    scaler = pd.read_csv(V285_SCALER, encoding="utf-8-sig", low_memory=False)
    preferred_sets = ["raw_shape_bad_top64", "raw_coupling_top48", "raw_low_identity_top64"]
    sub = scaler[scaler["feature_set"].astype(str).isin(preferred_sets)].copy()
    cols = sub["feature"].dropna().astype(str).drop_duplicates().tolist()
    return cols, sub


def load_events_with_raw285() -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """合并 v263 wait gate 事件表与 v285 raw shape-state 特征。"""

    if not V263_EVENTS.exists():
        raise FileNotFoundError(f"缺少 v263 event 表：{V263_EVENTS}")
    if not V285_FEATURES.exists():
        raise FileNotFoundError(f"缺少 v285 feature 表：{V285_FEATURES}")
    events = pd.read_csv(V263_EVENTS, encoding="utf-8-sig", low_memory=False)
    raw_cols, raw_audit = load_v285_feature_columns()
    raw = pd.read_csv(
        V285_FEATURES,
        encoding="utf-8-sig",
        low_memory=False,
        usecols=lambda c: c
        in {"event_uid", "observation_s", "bio285_status", "bio285_uses_post_observation"} | set(raw_cols),
    )
    if raw["event_uid"].duplicated().any():
        raise RuntimeError("v285 feature 表存在重复 event_uid")
    events = events.merge(raw, on="event_uid", how="left", validate="one_to_one")
    events["observation_s"] = pd.to_numeric(events.get("observation_s", np.nan), errors="coerce")
    events["event_order"] = events["event_uid"].map(event_order_from_uid)
    events["online_sort_key"] = (
        events["recording"].astype(str)
        + "|"
        + events["observation_s"].fillna(events["event_order"].astype(float)).map(lambda x: f"{float(x):012.3f}")
        + "|"
        + events["event_order"].map(lambda x: f"{int(x):05d}")
    )
    for col in raw_cols:
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors="coerce")
    return events, [c for c in raw_cols if c in events.columns], raw_audit


def online_calibrate(events: pd.DataFrame, pred_col: str, raw_z: np.ndarray) -> pd.DataFrame:
    """同 split、同 subject 的在线残差校准。"""

    out = events.copy()
    for col in [
        f"{pred_col}_subject_mean_resid",
        f"{pred_col}_subject_recent_resid",
        f"{pred_col}_raw285_knn_resid",
    ]:
        out[col] = pd.to_numeric(out[pred_col], errors="coerce")
    out["online_history_n"] = 0
    out["online_raw285_knn_used"] = False

    for (_split, _subject), group in out.groupby(["split", "subject"], sort=False):
        ordered = group.sort_values(["online_sort_key", "event_uid"]).index.tolist()
        history: List[int] = []
        for idx in ordered:
            pred = float(out.at[idx, pred_col])
            if len(history) >= MIN_HISTORY:
                hist = history[-40:]
                actual = out.loc[hist, "gain_latest_vs_keep0"].to_numpy(dtype=float)
                hist_pred = out.loc[hist, pred_col].to_numpy(dtype=float)
                resid = actual - hist_pred
                if np.isfinite(resid).any():
                    out.at[idx, f"{pred_col}_subject_mean_resid"] = pred + float(np.nanmean(resid))
                recent = history[-min(8, len(history)) :]
                recent_resid = (
                    out.loc[recent, "gain_latest_vs_keep0"].to_numpy(dtype=float)
                    - out.loc[recent, pred_col].to_numpy(dtype=float)
                )
                if np.isfinite(recent_resid).any():
                    out.at[idx, f"{pred_col}_subject_recent_resid"] = pred + float(np.nanmean(recent_resid))

                hist_arr = np.asarray(hist, dtype=int)
                hist_z = raw_z[hist_arr]
                cur_z = raw_z[int(idx)]
                d = np.sqrt(np.nanmean((hist_z - cur_z[None, :]) ** 2, axis=1))
                finite = np.isfinite(d)
                if finite.any():
                    valid_hist = hist_arr[finite]
                    valid_d = d[finite]
                    order = np.argsort(valid_d)[: min(KNN_K, len(valid_d))]
                    nn_idx = valid_hist[order]
                    nn_d = valid_d[order]
                    sigma = float(np.nanmedian(valid_d)) if np.isfinite(np.nanmedian(valid_d)) else 1.0
                    sigma = max(sigma, 1e-6)
                    weights = np.exp(-(nn_d ** 2) / (2.0 * sigma ** 2))
                    nn_resid = (
                        out.loc[nn_idx, "gain_latest_vs_keep0"].to_numpy(dtype=float)
                        - out.loc[nn_idx, pred_col].to_numpy(dtype=float)
                    )
                    if np.isfinite(nn_resid).any() and float(weights.sum()) > 0:
                        out.at[idx, f"{pred_col}_raw285_knn_resid"] = pred + float(np.average(nn_resid, weights=weights))
                        out.at[idx, "online_raw285_knn_used"] = True
            out.at[idx, "online_history_n"] = len(history)
            history.append(idx)
    return out


def selected_rows(events: pd.DataFrame, strategy: str, pred_col: str | None = None, force: str | None = None) -> pd.DataFrame:
    """把 gain 预测转换成 keep0/latest 选择并计算 tail RMSE。"""

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
                raise ValueError("pred_col required")
            choose_latest = float(row[pred_col]) > 0.0

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
                "online_history_n": int(row.get("online_history_n", 0)),
            }
        )
    return pd.DataFrame(rows)


def plot_test_badtop10(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v286_raw285_online_badtop10.png"
    order = [
        "policy_keep_0ms_anchor",
        "policy_wait_to_latest_anchor",
        "gate_vehicle_gain_t0",
        "gate_vehicle_raw285_gain_t0",
        "online_subject_mean_vehicle",
        "online_raw285_knn_vehicle",
        "online_subject_mean_vehicle_raw285",
        "online_raw285_knn_vehicle_raw285",
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
    fig, ax = plt.subplots(figsize=(15, 5.5))
    x = np.arange(len(sub))
    ax.bar(x, sub["selected_tail_rmse_mean"], color="#4C78A8")
    ax.axhline(float(sub[sub["strategy"].eq("policy_wait_to_latest_anchor")]["selected_tail_rmse_mean"].iloc[0]), color="black", linewidth=1, linestyle="--", label="wait-latest")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in sub["strategy"]], fontsize=8)
    ax.set_ylabel("test bad_top10 selected tail RMSE")
    ax.set_title("v286: raw285 online subject-aware calibration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v286_script", Path(__file__)),
        ("v258_training_utils", V258_SCRIPT),
        ("v263_event_wait_gate_predictions", V263_EVENTS),
        ("v285_raw200_shape_features", V285_FEATURES),
        ("v285_train_scaler_audit", V285_SCALER),
        ("v285_guardrail", V285_GUARDRAIL),
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


def table_to_md(df: pd.DataFrame, cols: List[str] | None = None, max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "_空表_"
    show = df.copy()
    if cols is not None:
        show = show[[c for c in cols if c in show.columns]]
    return show.head(max_rows).to_markdown(index=False)


def write_report(summary: pd.DataFrame, feature_audit: pd.DataFrame, online_audit: pd.DataFrame, guardrail: Dict[str, object], figures: List[Path]) -> Path:
    path = REPORTS / "v286_raw200_online_subject_calibration_cn.md"
    keep = [
        "policy_keep_0ms_anchor",
        "policy_wait_to_latest_anchor",
        "gate_vehicle_gain_t0",
        "gate_vehicle_raw285_gain_t0",
        "online_subject_mean_vehicle",
        "online_subject_recent_vehicle",
        "online_raw285_knn_vehicle",
        "online_subject_mean_vehicle_raw285",
        "online_raw285_knn_vehicle_raw285",
        "oracle_best_anchor_upper_bound",
    ]
    focus = summary[
        summary["split"].eq("test")
        & summary["event_group"].isin(["all", "bad_top10", "normal", "observe_later_like", "strong_steer"])
        & summary["strategy"].isin(keep)
    ].copy()
    bad = focus[focus["event_group"].eq("bad_top10")]
    scores = {str(r.strategy): float(r.selected_tail_rmse_mean) for r in bad.itertuples(index=False)}
    vehicle = scores.get("gate_vehicle_gain_t0", math.nan)
    subject_mean = scores.get("online_subject_mean_vehicle", math.nan)
    raw_knn = scores.get("online_raw285_knn_vehicle", math.nan)
    raw_knn_bio = scores.get("online_raw285_knn_vehicle_raw285", math.nan)
    latest = scores.get("policy_wait_to_latest_anchor", math.nan)

    lines: List[str] = []
    lines.append("# v286 raw-200Hz online subject-aware calibration")
    lines.append("")
    lines.append("## 本轮边界")
    lines.append("")
    lines.append("- 这不是 subject-disjoint 正式结果，而是 subject-aware / online adaptation 边界实验。")
    lines.append("- global gate 只用 train split 训练；val/test 校准只用同 split 同 subject 的更早事件。")
    lines.append("- 生理表示来自 v285 raw 200Hz shape-state train-only feature set。")
    lines.append("")
    lines.append("## 特征与在线历史")
    lines.append("")
    lines.append(table_to_md(feature_audit))
    lines.append("")
    lines.append(table_to_md(online_audit))
    lines.append("")
    lines.append("## Test 关键结果")
    lines.append("")
    lines.append(
        table_to_md(
            focus,
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
            ],
            max_rows=120,
        )
    )
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    for strategy in keep:
        if strategy in scores:
            lines.append(f"- bad_top10 / {strategy}: tail={scores[strategy]:.4f}.")
    if np.isfinite(raw_knn) and np.isfinite(subject_mean):
        lines.append(f"- raw285 KNN online 相对纯 subject mean online 改变量为 {raw_knn - subject_mean:+.4f}。")
    if np.isfinite(raw_knn_bio) and np.isfinite(subject_mean):
        lines.append(f"- vehicle+raw285 global 后再 raw285 KNN online，相对纯 subject mean online 改变量为 {raw_knn_bio - subject_mean:+.4f}。")
    if np.isfinite(latest):
        lines.append(f"- fixed wait-latest bad_top10 为 {latest:.4f}，这是当前线上策略必须击败的强基线。")
    lines.append("- 若 subject-aware online 仍无法稳定低于 wait-latest，说明当前生理数据即使在个体化边界下也没有形成差样本本质改善。")
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
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    print("[v286] raw285 online subject-aware calibration", flush=True)
    clean_out_dir()
    np.random.seed(SEED)

    events, raw_cols, raw_audit = load_events_with_raw285()
    vehicle_cols = vehicle_feature_columns(events)
    train_mask = events["split"].astype(str).eq("train").to_numpy()
    events["target_gain_latest_vs_keep0"] = pd.to_numeric(events["gain_latest_vs_keep0"], errors="coerce")

    pred_vehicle, audit_vehicle = V258.train_model(events, vehicle_cols, "target_gain_latest_vs_keep0", train_mask)
    pred_vehicle_raw, audit_vehicle_raw = V258.train_model(events, vehicle_cols + raw_cols, "target_gain_latest_vs_keep0", train_mask)
    events["pred_gain_vehicle"] = pred_vehicle
    events["pred_gain_vehicle_raw285"] = pred_vehicle_raw

    raw_z, raw_med, raw_std = V258.fit_fill_scale(events[raw_cols].to_numpy(dtype=float), train_mask)
    events = online_calibrate(events, "pred_gain_vehicle", raw_z)
    events = online_calibrate(events, "pred_gain_vehicle_raw285", raw_z)

    events["pred_online_subject_mean_vehicle"] = events["pred_gain_vehicle_subject_mean_resid"]
    events["pred_online_subject_recent_vehicle"] = events["pred_gain_vehicle_subject_recent_resid"]
    events["pred_online_raw285_knn_vehicle"] = events["pred_gain_vehicle_raw285_knn_resid"]
    events["pred_online_subject_mean_vehicle_raw285"] = events["pred_gain_vehicle_raw285_subject_mean_resid"]
    events["pred_online_raw285_knn_vehicle_raw285"] = events["pred_gain_vehicle_raw285_raw285_knn_resid"]

    selected = pd.concat(
        [
            selected_rows(events, "policy_keep_0ms_anchor", force="keep0"),
            selected_rows(events, "policy_wait_to_latest_anchor", force="latest"),
            selected_rows(events, "oracle_best_anchor_upper_bound", force="oracle"),
            selected_rows(events, "gate_vehicle_gain_t0", "pred_gain_vehicle"),
            selected_rows(events, "gate_vehicle_raw285_gain_t0", "pred_gain_vehicle_raw285"),
            selected_rows(events, "online_subject_mean_vehicle", "pred_online_subject_mean_vehicle"),
            selected_rows(events, "online_subject_recent_vehicle", "pred_online_subject_recent_vehicle"),
            selected_rows(events, "online_raw285_knn_vehicle", "pred_online_raw285_knn_vehicle"),
            selected_rows(events, "online_subject_mean_vehicle_raw285", "pred_online_subject_mean_vehicle_raw285"),
            selected_rows(events, "online_raw285_knn_vehicle_raw285", "pred_online_raw285_knn_vehicle_raw285"),
        ],
        ignore_index=True,
    )
    summary = V258.summarize_selected(selected)
    figures = [plot_test_badtop10(summary)]

    feature_audit = pd.DataFrame(
        [
            {"model": "global_vehicle_gain", "feature_n": len(vehicle_cols), "raw285_feature_n": 0},
            {"model": "global_vehicle_raw285_gain", "feature_n": len(vehicle_cols) + len(raw_cols), "raw285_feature_n": len(raw_cols)},
            {"model": "online_raw285_knn", "feature_n": len(raw_cols), "raw285_feature_n": len(raw_cols)},
        ]
    )
    online_audit = events.groupby(["split", "subject"], as_index=False).agg(
        event_n=("event_uid", "count"),
        history_ge_min_rate=("online_history_n", lambda s: float((s >= MIN_HISTORY).mean())),
        raw285_knn_used_rate=("online_raw285_knn_used", "mean"),
        bad_top10_n=("bad_top10_split_v241", "sum"),
    )
    fill_audit = pd.DataFrame({"feature": raw_cols, "raw_fill_median": raw_med, "raw_scale_std": raw_std})
    audit_vehicle["model"] = "global_vehicle_gain"
    audit_vehicle_raw["model"] = "global_vehicle_raw285_gain"

    write_csv(events, TABLES / "v286_event_online_predictions.csv")
    write_csv(selected, TABLES / "v286_selected_wait_gate_by_strategy.csv")
    write_csv(summary, TABLES / "v286_online_strategy_summary.csv")
    write_csv(feature_audit, TABLES / "v286_feature_block_audit.csv")
    write_csv(raw_audit, TABLES / "v286_raw285_feature_source_audit.csv")
    write_csv(fill_audit, TABLES / "v286_raw285_fill_scale_audit.csv")
    write_csv(pd.concat([audit_vehicle, audit_vehicle_raw], ignore_index=True), TABLES / "v286_global_model_feature_audit.csv")
    write_csv(online_audit, TABLES / "v286_online_history_audit.csv")
    write_input_hashes()

    guardrail = {
        "pass": True,
        "zip_testzip": False,
        "task_boundary": "subject_aware_online_adaptation_diagnostic_not_formal_subject_disjoint",
        "global_model_train_only": True,
        "online_history_only_previous_same_split_subject_events": True,
        "min_history_for_calibration": int(MIN_HISTORY),
        "raw285_knn_k": int(KNN_K),
        "event_n": int(events["event_uid"].nunique()),
        "vehicle_feature_n": int(len(vehicle_cols)),
        "raw285_feature_n": int(len(raw_cols)),
        "v285_source_guardrail_pass": False,
    }
    if V285_GUARDRAIL.exists():
        v285_guard = json.loads(V285_GUARDRAIL.read_text(encoding="utf-8"))
        guardrail["v285_source_guardrail_pass"] = bool(v285_guard.get("pass", False))
    write_file_inventory()
    zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(zip_ok)
    guardrail["pass"] = bool(
        zip_ok
        and guardrail["event_n"] > 0
        and guardrail["vehicle_feature_n"] > 0
        and guardrail["raw285_feature_n"] >= 64
        and guardrail["v285_source_guardrail_pass"]
    )
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(summary, feature_audit, online_audit, guardrail, figures)
    write_file_inventory()
    zip_ok2 = make_zip()
    guardrail["zip_testzip"] = bool(zip_ok2)
    guardrail["pass"] = bool(guardrail["pass"] and zip_ok2)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(summary, feature_audit, online_audit, guardrail, figures)
    write_file_inventory()
    if not guardrail["pass"]:
        raise AssertionError("v286 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))

    focus = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["strategy"].isin(
            [
                "policy_keep_0ms_anchor",
                "policy_wait_to_latest_anchor",
                "gate_vehicle_gain_t0",
                "gate_vehicle_raw285_gain_t0",
                "online_subject_mean_vehicle",
                "online_raw285_knn_vehicle",
                "online_subject_mean_vehicle_raw285",
                "online_raw285_knn_vehicle_raw285",
                "oracle_best_anchor_upper_bound",
            ]
        )
    ]
    print(f"[v286] report={report}", flush=True)
    print(f"[v286] zip={ZIP_PATH}", flush=True)
    print(focus[["strategy", "selected_tail_rmse_mean", "delta_selected_minus_latest_mean", "selected_latest_rate"]].to_string(index=False), flush=True)
    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
