#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v264 online subject-aware physiology calibration。

重要边界：
- 这不是正式 subject-disjoint 替代结果，因为它允许使用同一驾驶员更早事件的已知结果做在线校准；
- 目的只是验证一个关键问题：如果把任务边界改成 subject-aware / online adaptation，
  生理状态是否终于能帮助差样本，而不是继续在 subject-disjoint 中强行拼接。

本轮策略：
- 以 v263 的 0ms wait gate 为基础，只决策 keep0 或 wait-latest；
- 全局模型仍只在 train subjects 上训练；
- 对 val/test subject，按 recording + observation_s 顺序，只用当前事件之前的同 subject 历史事件残差做校准；
- 比较纯 subject 残差均值校准与 physiology KNN 残差校准，判断生理是否带来额外价值。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
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
V263_EVENTS = (
    BASELINES
    / "v263_bio260_wait_gate_20260702"
    / "tables"
    / "v263_event_wait_gate_predictions.csv"
)
V262_FEATURE_SELECTION = (
    BASELINES
    / "v262_subject_invariant_bio260_selector_20260702"
    / "tables"
    / "v262_feature_selection_audit.csv"
)
V260_BIO = (
    BASELINES
    / "v260_event_biomarker_physio_rebuild_20260702"
    / "tables"
    / "v260_event_biomarker_features.csv"
)

OUT = BASELINES / "v264_online_subject_physio_calibration_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v264_online_subject_physio_calibration_20260702_pack.zip"

SEED = 26402
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


def load_v258_module():
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


def feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    vehicle_prefixes = ("candidate_delay", "hist_", "road_", "phase_", "instability_penalty")
    vehicle_cols: List[str] = []
    for col in df.columns:
        if col in {"candidate_delay_ms", "candidate_delay_s"} or col.startswith(vehicle_prefixes):
            if pd.api.types.is_numeric_dtype(df[col]):
                vehicle_cols.append(col)
    if not V262_FEATURE_SELECTION.exists():
        raise FileNotFoundError(f"缺少 v262 特征选择表：{V262_FEATURE_SELECTION}")
    fs = pd.read_csv(V262_FEATURE_SELECTION, encoding="utf-8-sig", low_memory=False)
    sp64 = fs[
        fs["row_type"].astype(str).eq("feature")
        & fs["in_sp64"].astype(str).str.lower().eq("true")
    ]["column"].dropna().astype(str).tolist()
    bio_cols = [col for col in sp64 if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if "bio260_floor_status_ok" in df.columns:
        bio_cols.append("bio260_floor_status_ok")
    return list(dict.fromkeys(vehicle_cols)), list(dict.fromkeys(bio_cols))


def event_order_from_uid(uid: str) -> int:
    match = re.search(r"_(\d+)$", str(uid))
    if match:
        return int(match.group(1))
    return 0


def load_events() -> pd.DataFrame:
    if not V263_EVENTS.exists():
        raise FileNotFoundError(f"缺少 v263 event 表：{V263_EVENTS}")
    events = pd.read_csv(V263_EVENTS, encoding="utf-8-sig", low_memory=False)
    if V260_BIO.exists():
        meta = pd.read_csv(
            V260_BIO,
            encoding="utf-8-sig",
            low_memory=False,
            usecols=lambda c: c in {"event_uid", "delay_ms", "observation_s", "bio260_uses_post_observation"},
        )
        meta = meta[pd.to_numeric(meta["delay_ms"], errors="coerce").eq(0)].drop_duplicates("event_uid")
        events = events.merge(
            meta[["event_uid", "observation_s", "bio260_uses_post_observation"]],
            on="event_uid",
            how="left",
        )
    events["event_order"] = events["event_uid"].map(event_order_from_uid)
    events["observation_s"] = pd.to_numeric(events.get("observation_s", np.nan), errors="coerce")
    events["online_sort_key"] = (
        events["recording"].astype(str)
        + "|"
        + events["observation_s"].fillna(events["event_order"].astype(float)).map(lambda x: f"{float(x):012.3f}")
        + "|"
        + events["event_order"].map(lambda x: f"{int(x):05d}")
    )
    return events


def selected_rows(events: pd.DataFrame, strategy: str, pred_col: str | None = None, force: str | None = None) -> pd.DataFrame:
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


def online_calibrate(
    events: pd.DataFrame,
    pred_col: str,
    bio_z: np.ndarray,
) -> pd.DataFrame:
    out = events.copy()
    for col in [
        f"{pred_col}_subject_mean_resid",
        f"{pred_col}_subject_recent_resid",
        f"{pred_col}_physio_knn_resid",
    ]:
        out[col] = pd.to_numeric(out[pred_col], errors="coerce")
    out["online_history_n"] = 0
    out["online_physio_knn_used"] = False

    # 每个 split 内单独在线回放，避免 train/val/test 之间共享同一驾驶员历史。
    for (split, subject), group in out.groupby(["split", "subject"], sort=False):
        ordered = group.sort_values(["online_sort_key", "event_uid"]).index.tolist()
        history: List[int] = []
        for idx in ordered:
            pred = float(out.at[idx, pred_col])
            if len(history) >= MIN_HISTORY:
                hist = history[-30:]
                actual = out.loc[hist, "gain_latest_vs_keep0"].to_numpy(dtype=float)
                hist_pred = out.loc[hist, pred_col].to_numpy(dtype=float)
                resid = actual - hist_pred
                out.at[idx, f"{pred_col}_subject_mean_resid"] = pred + float(np.nanmean(resid))
                recent = history[-min(8, len(history)) :]
                recent_resid = (
                    out.loc[recent, "gain_latest_vs_keep0"].to_numpy(dtype=float)
                    - out.loc[recent, pred_col].to_numpy(dtype=float)
                )
                out.at[idx, f"{pred_col}_subject_recent_resid"] = pred + float(np.nanmean(recent_resid))

                hist_z = bio_z[hist]
                cur_z = bio_z[idx]
                d = np.sqrt(np.nanmean((hist_z - cur_z[None, :]) ** 2, axis=1))
                finite = np.isfinite(d)
                if finite.any():
                    hist_arr = np.asarray(hist)
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
                    if np.isfinite(nn_resid).any() and weights.sum() > 0:
                        out.at[idx, f"{pred_col}_physio_knn_resid"] = pred + float(np.average(nn_resid, weights=weights))
                        out.at[idx, "online_physio_knn_used"] = True
            out.at[idx, "online_history_n"] = len(history)
            history.append(idx)
    return out


def plot_test_summary(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v264_online_subject_physio_badtop10.png"
    order = [
        "policy_keep_0ms_anchor",
        "policy_wait_to_latest_anchor",
        "gate_vehicle_gain_t0",
        "gate_vehicle_bio260_sp64_gain_t0",
        "online_subject_mean_vehicle",
        "online_subject_recent_vehicle",
        "online_physio_knn_vehicle",
        "online_subject_mean_vehicle_bio",
        "online_physio_knn_vehicle_bio",
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
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in sub["strategy"]], fontsize=8)
    ax.set_ylabel("test bad_top10 selected tail RMSE")
    ax.set_title("v264: online subject-aware physiology calibration")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v264_script", Path(__file__)),
        ("v263_event_wait_gate_predictions", V263_EVENTS),
        ("v262_feature_selection", V262_FEATURE_SELECTION),
        ("v260_event_biomarker_features", V260_BIO),
        ("v258_reused_training_utils", V258_SCRIPT),
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


def write_report(summary: pd.DataFrame, feature_audit: pd.DataFrame, online_audit: pd.DataFrame, figures: List[Path]) -> None:
    lines: List[str] = []
    lines.append("# v264 online subject-aware physiology calibration")
    lines.append("")
    lines.append("## 本轮边界")
    lines.append("")
    lines.append("- 这不是正式 subject-disjoint 替代结果。")
    lines.append("- 本轮允许同一 subject 的更早事件结果作为在线历史反馈，只用于判断生理是否在 subject-aware / online adaptation 设定下更有价值。")
    lines.append("- 当前事件本身、当前事件之后的结果、observation_s 之后的生理都不进入输入。")
    lines.append("")
    lines.append("## 特征与在线历史")
    lines.append("")
    lines.append(feature_audit.to_markdown(index=False))
    lines.append("")
    lines.append(online_audit.to_markdown(index=False))
    lines.append("")
    lines.append("## Test 关键结果")
    lines.append("")
    focus = summary[
        summary["split"].eq("test")
        & summary["event_group"].isin(["all", "bad_top10", "normal", "observe_later_like", "strong_steer"])
    ].copy()
    keep = [
        "policy_keep_0ms_anchor",
        "policy_wait_to_latest_anchor",
        "gate_vehicle_gain_t0",
        "gate_vehicle_bio260_sp64_gain_t0",
        "online_subject_mean_vehicle",
        "online_subject_recent_vehicle",
        "online_physio_knn_vehicle",
        "online_subject_mean_vehicle_bio",
        "online_physio_knn_vehicle_bio",
        "oracle_best_anchor_upper_bound",
    ]
    focus = focus[focus["strategy"].isin(keep)]
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
    bad = focus[focus["event_group"].eq("bad_top10")]
    scores: Dict[str, float] = {}
    for strategy in keep:
        row = bad[bad["strategy"].eq(strategy)]
        if len(row):
            score = float(row["selected_tail_rmse_mean"].iloc[0])
            latest_rate = float(row["selected_latest_rate"].iloc[0])
            scores[strategy] = score
            lines.append(f"- bad_top10 / {strategy}: tail={score:.4f}, latest_rate={latest_rate:.3f}.")
    vehicle = scores.get("gate_vehicle_gain_t0", np.nan)
    best_online = min((v, k) for k, v in scores.items() if k.startswith("online_")) if any(k.startswith("online_") for k in scores) else (np.nan, "")
    physio = scores.get("online_physio_knn_vehicle_bio", np.nan)
    subject_mean = scores.get("online_subject_mean_vehicle", np.nan)
    if np.isfinite(best_online[0]) and np.isfinite(vehicle):
        lines.append("")
        lines.append(f"- 最佳 online 策略 `{best_online[1]}` 相对 global vehicle gate 改变量为 {best_online[0] - vehicle:+.4f}。")
    if np.isfinite(physio) and np.isfinite(subject_mean):
        lines.append(f"- physiology KNN online 相对纯 subject mean online 改变量为 {physio - subject_mean:+.4f}；这是判断生理额外价值的核心数。")
    lines.append("- 如果 online subject calibration 有效但 physiology KNN 无额外收益，说明需要的是同驾驶员反馈，而不是当前生理特征本身。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v264_online_subject_physio_calibration_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v264] online subject-aware physiology calibration")
    clean_out_dir()
    np.random.seed(SEED)

    events = load_events()
    vehicle_cols, bio_cols = feature_columns(events)
    train_mask = events["split"].astype(str).eq("train").to_numpy()
    events["target_gain_latest_vs_keep0"] = pd.to_numeric(events["gain_latest_vs_keep0"], errors="coerce")

    pred_vehicle, audit_vehicle = V258.train_model(events, vehicle_cols, "target_gain_latest_vs_keep0", train_mask)
    pred_vehicle_bio, audit_vehicle_bio = V258.train_model(
        events, vehicle_cols + bio_cols, "target_gain_latest_vs_keep0", train_mask
    )
    events["pred_gain_vehicle"] = pred_vehicle
    events["pred_gain_vehicle_bio260_sp64"] = pred_vehicle_bio

    bio_z, _, _ = V258.fit_fill_scale(events[bio_cols].to_numpy(dtype=float), train_mask)
    events = online_calibrate(events, "pred_gain_vehicle", bio_z)
    events = online_calibrate(events, "pred_gain_vehicle_bio260_sp64", bio_z)

    # 统一命名，便于策略表阅读。
    events["pred_online_subject_mean_vehicle"] = events["pred_gain_vehicle_subject_mean_resid"]
    events["pred_online_subject_recent_vehicle"] = events["pred_gain_vehicle_subject_recent_resid"]
    events["pred_online_physio_knn_vehicle"] = events["pred_gain_vehicle_physio_knn_resid"]
    events["pred_online_subject_mean_vehicle_bio"] = events["pred_gain_vehicle_bio260_sp64_subject_mean_resid"]
    events["pred_online_physio_knn_vehicle_bio"] = events["pred_gain_vehicle_bio260_sp64_physio_knn_resid"]

    selected = pd.concat(
        [
            selected_rows(events, "policy_keep_0ms_anchor", force="keep0"),
            selected_rows(events, "policy_wait_to_latest_anchor", force="latest"),
            selected_rows(events, "oracle_best_anchor_upper_bound", force="oracle"),
            selected_rows(events, "gate_vehicle_gain_t0", "pred_gain_vehicle"),
            selected_rows(events, "gate_vehicle_bio260_sp64_gain_t0", "pred_gain_vehicle_bio260_sp64"),
            selected_rows(events, "online_subject_mean_vehicle", "pred_online_subject_mean_vehicle"),
            selected_rows(events, "online_subject_recent_vehicle", "pred_online_subject_recent_vehicle"),
            selected_rows(events, "online_physio_knn_vehicle", "pred_online_physio_knn_vehicle"),
            selected_rows(events, "online_subject_mean_vehicle_bio", "pred_online_subject_mean_vehicle_bio"),
            selected_rows(events, "online_physio_knn_vehicle_bio", "pred_online_physio_knn_vehicle_bio"),
        ],
        ignore_index=True,
    )
    summary = V258.summarize_selected(selected)
    figures = [plot_test_summary(summary)]

    feature_audit = pd.DataFrame(
        [
            {"model": "global_vehicle_gain", "feature_n": len(vehicle_cols), "bio260_feature_n": 0},
            {
                "model": "global_vehicle_bio260_sp64_gain",
                "feature_n": len(vehicle_cols) + len(bio_cols),
                "bio260_feature_n": len(bio_cols),
            },
        ]
    )
    online_audit = events.groupby(["split", "subject"], as_index=False).agg(
        event_n=("event_uid", "count"),
        history_ge_min_rate=("online_history_n", lambda s: float((s >= MIN_HISTORY).mean())),
        physio_knn_used_rate=("online_physio_knn_used", "mean"),
        bad_top10_n=("bad_top10_split_v241", "sum"),
    )
    fill_audit = pd.concat(
        [
            audit_vehicle.assign(model="global_vehicle_gain"),
            audit_vehicle_bio.assign(model="global_vehicle_bio260_sp64_gain"),
        ],
        ignore_index=True,
    )

    write_csv(events, TABLES / "v264_event_online_predictions.csv")
    write_csv(selected, TABLES / "v264_selected_wait_gate_by_strategy.csv")
    write_csv(summary, TABLES / "v264_online_strategy_summary.csv")
    write_csv(feature_audit, TABLES / "v264_feature_block_audit.csv")
    write_csv(fill_audit, TABLES / "v264_feature_fill_audit.csv")
    write_csv(online_audit, TABLES / "v264_online_history_audit.csv")
    write_input_hashes()
    write_file_inventory()
    write_report(summary, feature_audit, online_audit, figures)
    write_file_inventory()

    zip_ok = make_zip()
    guardrail = {
        "pass": bool(zip_ok),
        "zip_testzip": bool(zip_ok),
        "task_boundary": "subject_aware_online_adaptation_diagnostic_not_formal_subject_disjoint",
        "global_model_train_only": True,
        "online_history_only_previous_same_split_subject_events": True,
        "min_history_for_calibration": int(MIN_HISTORY),
        "physio_knn_k": int(KNN_K),
        "event_n": int(events["event_uid"].nunique()),
        "vehicle_feature_n": int(len(vehicle_cols)),
        "bio260_sp64_feature_n": int(len(bio_cols)),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not guardrail["pass"]:
        raise AssertionError("v264 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    focus = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["strategy"].isin(
            [
                "policy_keep_0ms_anchor",
                "policy_wait_to_latest_anchor",
                "gate_vehicle_gain_t0",
                "gate_vehicle_bio260_sp64_gain_t0",
                "online_subject_mean_vehicle",
                "online_subject_recent_vehicle",
                "online_physio_knn_vehicle",
                "online_subject_mean_vehicle_bio",
                "online_physio_knn_vehicle_bio",
                "oracle_best_anchor_upper_bound",
            ]
        )
    ]
    print(f"[v264] report={REPORTS / 'v264_online_subject_physio_calibration_cn.md'}")
    print(f"[v264] zip={ZIP_PATH}")
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
