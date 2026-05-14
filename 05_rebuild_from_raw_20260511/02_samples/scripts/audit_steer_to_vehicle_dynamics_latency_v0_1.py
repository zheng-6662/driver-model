# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(r"F:/data_set_process/data_process")
ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
EVENTS_CSV = (
    ROOT
    / "02_samples"
    / "episode_first_event_v0_6"
    / "tables"
    / "episode_candidates_v0_6.csv"
)
OUT_DIR = ROOT / "02_samples" / "steer_to_vehicle_dynamics_latency_v0_1"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
PANEL_DIR = FIG_DIR / "latency_review_panels"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-14.md"

VEHICLE_COLS = [
    "StorageTime",
    "zx|SteeringWheel",
    "zx|ay",
    "zx|vyaw",
    "zx|vroll",
    "zx|roll",
    "zx|BrakePedal",
    "zx|ax",
    "zx1|v_km/h",
    "zx1|mu",
    "zx1|lateraldistance",
    "zx1|lanecurvatureXY",
]

SIGNAL_SPECS = {
    "ay": {"col": "zx|ay", "floor": 0.25, "label": "横向加速度"},
    "yaw_rate": {"col": "zx|vyaw", "floor": 0.035, "label": "横摆角速度"},
    "roll_rate": {"col": "zx|vroll", "floor": 0.025, "label": "侧倾角速度"},
    "roll_angle": {"col": "zx|roll", "floor": 0.015, "label": "侧倾角"},
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, PANEL_DIR, REPORT_DIR, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)
    for old in PANEL_DIR.glob("*.png"):
        old.unlink()


def robust_mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(arr - med)) * 1.4826)


def robust_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmedian(arr))


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def parse_storage_time_to_rel_seconds(series: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() >= max(10, len(series) // 2):
        base = parsed.dropna().iloc[0]
        return (parsed - base).dt.total_seconds().to_numpy(dtype=float)
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return np.full(len(series), np.nan)
    return numeric - finite[0]


def load_vehicle(path_text: str) -> pd.DataFrame | None:
    path = Path(str(path_text))
    if not path.exists():
        return None
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    usecols = [col for col in VEHICLE_COLS if col in header.columns]
    if "StorageTime" not in usecols:
        return None
    df = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig")
    df["time_rel_s"] = parse_storage_time_to_rel_seconds(df["StorageTime"])
    df = df[np.isfinite(df["time_rel_s"])].copy()
    df = df.drop_duplicates("time_rel_s").sort_values("time_rel_s")
    for col in usecols:
        if col == "StorageTime":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def gradient(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.size < 3:
        return np.full_like(values, np.nan, dtype=float)
    good = np.isfinite(values) & np.isfinite(times)
    if good.sum() < 3:
        return np.full_like(values, np.nan, dtype=float)
    filled = values.copy()
    idx = np.arange(values.size)
    filled[~good] = np.interp(idx[~good], idx[good], values[good])
    dt = np.gradient(times)
    dt[~np.isfinite(dt) | (np.abs(dt) < 1e-6)] = np.nan
    rate = np.gradient(filled) / dt
    return rate


def sustained_onset(times: np.ndarray, cond: np.ndarray, min_duration_s: float = 0.06) -> tuple[float, bool]:
    times = np.asarray(times, dtype=float)
    cond = np.asarray(cond, dtype=bool)
    if times.size == 0 or cond.size == 0:
        return float("nan"), False
    finite_dt = np.diff(times)
    finite_dt = finite_dt[np.isfinite(finite_dt) & (finite_dt > 0)]
    dt = float(np.nanmedian(finite_dt)) if finite_dt.size else 0.005
    run_need = max(3, int(round(min_duration_s / max(dt, 1e-3))))
    run = 0
    start_idx = 0
    for idx, flag in enumerate(cond):
        if flag:
            if run == 0:
                start_idx = idx
            run += 1
            if run >= run_need:
                left_censored = start_idx == 0
                return float(times[start_idx]), bool(left_censored)
        else:
            run = 0
    return float("nan"), False


def future_window_max_abs(values: np.ndarray, times: np.ndarray, horizon_s: float) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    abs_values = np.abs(values)
    for idx, t in enumerate(times):
        if not np.isfinite(t):
            continue
        end = np.searchsorted(times, t + horizon_s, side="right")
        window = abs_values[idx:end]
        window = window[np.isfinite(window)]
        if window.size:
            out[idx] = float(np.nanmax(window))
    return out


def detect_event_latency(row: pd.Series, vehicle: pd.DataFrame | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "latency_audit_status": "ok",
        "latency_audit_note_cn": "",
    }
    if vehicle is None or vehicle.empty:
        result["latency_audit_status"] = "vehicle_missing"
        result["latency_audit_note_cn"] = "原始车辆文件缺失或无法读取"
        return result

    anchor = safe_float(row.get("t_train_anchor", np.nan))
    if not math.isfinite(anchor):
        anchor = safe_float(row.get("anchor_time_rel_s", np.nan))
    if not math.isfinite(anchor):
        result["latency_audit_status"] = "anchor_missing"
        result["latency_audit_note_cn"] = "样本锚点缺失，无法审计"
        return result

    time = vehicle["time_rel_s"].to_numpy(dtype=float)
    search_mask = (time >= anchor - 2.0) & (time <= anchor + 4.0)
    pre_mask = (time >= anchor - 4.0) & (time < anchor - 2.0)
    if pre_mask.sum() < 50:
        pre_mask = (time >= anchor - 5.0) & (time < anchor - 1.5)
    if search_mask.sum() < 30 or pre_mask.sum() < 30:
        result["latency_audit_status"] = "window_insufficient"
        result["latency_audit_note_cn"] = "事件前基线或事件附近搜索窗口不足"
        return result

    tw = time[search_mask]
    result["audit_anchor_time_s"] = anchor
    result["audit_search_start_s"] = float(tw[0])
    result["audit_search_end_s"] = float(tw[-1])
    result["audit_pre_sample_count"] = int(pre_mask.sum())
    result["audit_search_sample_count"] = int(search_mask.sum())

    steer_col = "zx|SteeringWheel"
    if steer_col not in vehicle.columns:
        result["latency_audit_status"] = "steer_missing"
        result["latency_audit_note_cn"] = "方向盘列缺失"
        return result
    steer = vehicle[steer_col].to_numpy(dtype=float)
    steer_pre = steer[pre_mask]
    steer_base = robust_median(steer_pre)
    steer_dev = steer - steer_base
    steer_rate = gradient(steer, time)
    steer_rate_pre = steer_rate[pre_mask]
    steer_amp_thr = max(0.08, 4.0 * (robust_mad(steer_pre - steer_base) or 0.0))
    steer_rate_thr = max(0.18, 4.0 * (robust_mad(steer_rate_pre) or 0.0))
    steer_dev_w = steer_dev[search_mask]
    steer_rate_w = steer_rate[search_mask]
    future_steer_delta = future_window_max_abs(steer_dev_w, tw, 0.50)
    steer_cond = (np.abs(steer_dev_w) >= steer_amp_thr) | (
        (np.abs(steer_rate_w) >= steer_rate_thr) & (future_steer_delta >= steer_amp_thr)
    )
    t_steer, steer_left_censored = sustained_onset(tw, steer_cond, min_duration_s=0.05)
    result.update(
        {
            "audit_t_steer_onset": t_steer,
            "audit_steer_left_censored": int(steer_left_censored),
            "audit_steer_baseline": steer_base,
            "audit_steer_amp_threshold": steer_amp_thr,
            "audit_steer_rate_threshold": steer_rate_thr,
        }
    )

    dyn_onsets: dict[str, float] = {}
    dyn_left: dict[str, int] = {}
    dyn_thresholds: dict[str, float] = {}
    for key, spec in SIGNAL_SPECS.items():
        col = str(spec["col"])
        if col not in vehicle.columns:
            dyn_onsets[key] = float("nan")
            dyn_left[key] = 0
            dyn_thresholds[key] = float("nan")
            continue
        values = vehicle[col].to_numpy(dtype=float)
        base = robust_median(values[pre_mask])
        thr = max(float(spec["floor"]), 4.0 * (robust_mad(values[pre_mask] - base) or 0.0))
        dev_w = values[search_mask] - base
        cond = np.abs(dev_w) >= thr
        onset, left = sustained_onset(tw, cond, min_duration_s=0.06)
        dyn_onsets[key] = onset
        dyn_left[key] = int(left)
        dyn_thresholds[key] = thr
        result[f"audit_t_{key}_onset"] = onset
        result[f"audit_{key}_left_censored"] = int(left)
        result[f"audit_{key}_threshold"] = thr

    valid_dyn = {key: value for key, value in dyn_onsets.items() if math.isfinite(value)}
    if valid_dyn:
        first_dyn_key = min(valid_dyn, key=valid_dyn.get)
        first_dyn_time = float(valid_dyn[first_dyn_key])
    else:
        first_dyn_key = ""
        first_dyn_time = float("nan")

    result["audit_first_dynamic_signal"] = first_dyn_key
    result["audit_t_first_dynamic_onset"] = first_dyn_time

    if math.isfinite(t_steer) and math.isfinite(first_dyn_time):
        delta = first_dyn_time - t_steer
        result["audit_delta_first_dynamic_minus_steer_s"] = delta
        for key, onset in dyn_onsets.items():
            result[f"audit_delta_{key}_minus_steer_s"] = onset - t_steer if math.isfinite(onset) else np.nan
        if steer_left_censored:
            category = "steer_onset_left_censored"
            note = "搜索窗口开始时方向盘已经在变化，无法精确判断提前量"
        elif delta >= 0.50:
            category = "steer_first_gap_ge_0_5s"
            note = "方向盘明显先动，且到第一类车辆动态响应至少有0.5秒"
        elif delta >= 0.20:
            category = "steer_first_gap_0_2_to_0_5s"
            note = "方向盘先动，存在0.2秒以上但不足0.5秒的早期窗口"
        elif delta >= 0.0:
            category = "near_sync_steer_first_lt_0_2s"
            note = "方向盘略早于车辆动态，但提前量不足0.2秒"
        elif delta > -0.20:
            category = "near_sync_vehicle_first_lt_0_2s"
            note = "车辆动态略早于方向盘，二者几乎同步"
        else:
            category = "vehicle_dynamic_first_ge_0_2s"
            note = "车辆动态明显早于方向盘，更像扰动后纠偏或锚点偏在动态开始处"
        result["audit_latency_category"] = category
        result["audit_gap_ge_0_2s"] = int(delta >= 0.20 and not steer_left_censored)
        result["audit_gap_ge_0_5s"] = int(delta >= 0.50 and not steer_left_censored)
        result["audit_satisfy_early_input_requirement"] = int(delta >= 0.20 and not steer_left_censored)
        result["latency_audit_note_cn"] = note
    elif not math.isfinite(t_steer) and math.isfinite(first_dyn_time):
        result["audit_latency_category"] = "dynamic_detected_no_steer"
        result["audit_satisfy_early_input_requirement"] = 0
        result["latency_audit_note_cn"] = "检测到车辆动态响应，但未检测到明确方向盘动作开始"
    elif math.isfinite(t_steer) and not math.isfinite(first_dyn_time):
        result["audit_latency_category"] = "steer_detected_no_dynamic"
        result["audit_satisfy_early_input_requirement"] = 0
        result["latency_audit_note_cn"] = "检测到方向盘动作，但未检测到横向/横摆/侧倾响应"
    else:
        result["audit_latency_category"] = "no_clear_onset"
        result["audit_satisfy_early_input_requirement"] = 0
        result["latency_audit_note_cn"] = "方向盘和车辆动态起点都不清楚"

    existing_steer = safe_float(row.get("t_steer_onset", np.nan))
    existing_dyn = safe_float(row.get("t_dyn_onset", np.nan))
    result["existing_delta_dyn_minus_steer_s"] = (
        existing_dyn - existing_steer if math.isfinite(existing_steer) and math.isfinite(existing_dyn) else np.nan
    )
    return result


def category_cn(category: str) -> str:
    mapping = {
        "steer_first_gap_ge_0_5s": "方向盘先动，提前量>=0.5秒",
        "steer_first_gap_0_2_to_0_5s": "方向盘先动，提前量0.2-0.5秒",
        "near_sync_steer_first_lt_0_2s": "方向盘略早但不足0.2秒",
        "near_sync_vehicle_first_lt_0_2s": "车辆动态略早，几乎同步",
        "vehicle_dynamic_first_ge_0_2s": "车辆动态明显早于方向盘",
        "steer_onset_left_censored": "方向盘起点左截断",
        "dynamic_detected_no_steer": "有车辆动态但无明确方向盘起点",
        "steer_detected_no_dynamic": "有方向盘但无明确车辆动态",
        "no_clear_onset": "起点都不清楚",
    }
    return mapping.get(str(category), str(category))


def write_tables(events: pd.DataFrame, audited: pd.DataFrame) -> dict[str, Path]:
    merged = pd.concat([events.reset_index(drop=True), audited.reset_index(drop=True)], axis=1)
    merged["audit_latency_category_cn"] = merged["audit_latency_category"].map(category_cn)
    out_events = TABLE_DIR / "steer_to_dynamics_latency_events_v0_1.csv"
    merged.to_csv(out_events, index=False, encoding="utf-8-sig")

    status_counts = (
        merged.groupby(["latency_audit_status", "audit_latency_category", "audit_latency_category_cn"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    status_counts["ratio"] = status_counts["n"] / max(len(merged), 1)
    out_summary = TABLE_DIR / "steer_to_dynamics_latency_summary_v0_1.csv"
    status_counts.to_csv(out_summary, index=False, encoding="utf-8-sig")

    group_cols = ["v0_6_final_bucket_cn", "road_design_module_name", "audit_latency_category_cn"]
    by_group = merged.groupby(group_cols, dropna=False).size().reset_index(name="n")
    totals = merged.groupby(["v0_6_final_bucket_cn", "road_design_module_name"], dropna=False).size().reset_index(name="total")
    by_group = by_group.merge(totals, on=["v0_6_final_bucket_cn", "road_design_module_name"], how="left")
    by_group["ratio_in_group"] = by_group["n"] / by_group["total"].replace(0, np.nan)
    out_group = TABLE_DIR / "steer_to_dynamics_latency_by_bucket_module_v0_1.csv"
    by_group.to_csv(out_group, index=False, encoding="utf-8-sig")

    finite = merged[np.isfinite(merged["audit_delta_first_dynamic_minus_steer_s"])].copy()
    quantile_rows: list[dict[str, Any]] = []
    for name, sub in [("all", finite)] + list(finite.groupby("v0_6_final_bucket_cn", dropna=False)):
        values = sub["audit_delta_first_dynamic_minus_steer_s"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        quantile_rows.append(
            {
                "group": name,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p10": float(np.quantile(values, 0.10)),
                "p25": float(np.quantile(values, 0.25)),
                "p75": float(np.quantile(values, 0.75)),
                "p90": float(np.quantile(values, 0.90)),
                "gap_ge_0_2_ratio": float(np.mean(values >= 0.20)),
                "gap_ge_0_5_ratio": float(np.mean(values >= 0.50)),
                "vehicle_first_ge_0_2_ratio": float(np.mean(values <= -0.20)),
            }
        )
    out_quantile = TABLE_DIR / "steer_to_dynamics_latency_quantiles_v0_1.csv"
    pd.DataFrame(quantile_rows).to_csv(out_quantile, index=False, encoding="utf-8-sig")

    return {
        "events": out_events,
        "summary": out_summary,
        "group": out_group,
        "quantile": out_quantile,
    }


def plot_figures(merged: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    finite = merged[np.isfinite(merged["audit_delta_first_dynamic_minus_steer_s"])].copy()
    if finite.empty:
        return paths

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    vals = finite["audit_delta_first_dynamic_minus_steer_s"].clip(-2.0, 3.0)
    ax.hist(vals, bins=50, color="#4C78A8", alpha=0.85)
    for x, label, color in [(0.0, "0s", "#111111"), (0.2, "0.2s", "#F58518"), (0.5, "0.5s", "#E45756")]:
        ax.axvline(x, color=color, linewidth=1.5, linestyle="--")
        ax.text(x, ax.get_ylim()[1] * 0.92, label, color=color, ha="left", va="top", fontsize=9)
    ax.set_title("Steering onset to first vehicle-dynamics onset")
    ax.set_xlabel("first dynamics onset - steering onset (s)")
    ax.set_ylabel("event count")
    ax.grid(alpha=0.2)
    out = FIG_DIR / "steer_to_dynamics_latency_histogram_v0_1.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    paths.append(out)

    pivot = (
        merged.groupby(["v0_6_final_bucket_cn", "audit_latency_category_cn"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    if not pivot.empty:
        pivot_ratio = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0)
        fig, ax = plt.subplots(figsize=(11, 5.8), dpi=160)
        pivot_ratio.plot(kind="barh", stacked=True, ax=ax, colormap="tab20")
        ax.set_title("Latency category ratio by v0.6 bucket")
        ax.set_xlabel("ratio")
        ax.set_ylabel("")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
        ax.grid(axis="x", alpha=0.2)
        out = FIG_DIR / "steer_to_dynamics_latency_by_bucket_v0_1.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        paths.append(out)

    modules = finite["road_design_module_name"].fillna("unknown").value_counts().index.tolist()
    box_data = []
    labels = []
    for module in modules:
        sub = finite[finite["road_design_module_name"].fillna("unknown") == module]
        if len(sub) >= 5:
            box_data.append(sub["audit_delta_first_dynamic_minus_steer_s"].clip(-2, 3).to_numpy())
            labels.append(module)
    if box_data:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
        ax.boxplot(box_data, tick_labels=labels, vert=True, showfliers=False)
        ax.axhline(0, color="#111111", linestyle="--", linewidth=1)
        ax.axhline(0.2, color="#F58518", linestyle="--", linewidth=1)
        ax.axhline(0.5, color="#E45756", linestyle="--", linewidth=1)
        ax.set_title("Latency distribution by road module")
        ax.set_ylabel("first dynamics onset - steering onset (s)")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
        out = FIG_DIR / "steer_to_dynamics_latency_by_module_box_v0_1.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        paths.append(out)
    return paths


def plot_review_panel(row: pd.Series, vehicle: pd.DataFrame, out: Path) -> None:
    anchor = safe_float(row.get("audit_anchor_time_s", row.get("t_train_anchor", np.nan)))
    if not math.isfinite(anchor):
        anchor = safe_float(row.get("anchor_time_rel_s", np.nan))
    if not math.isfinite(anchor):
        return
    time = vehicle["time_rel_s"].to_numpy(dtype=float)
    mask = (time >= anchor - 2.5) & (time <= anchor + 4.0)
    if mask.sum() < 20:
        return
    t = time[mask] - anchor

    fig, axes = plt.subplots(5, 1, figsize=(10, 8), dpi=140, sharex=True)
    plot_specs = [
        ("zx|SteeringWheel", "steering"),
        ("zx|ay", "ay"),
        ("zx|vyaw", "yaw_rate"),
        ("zx|vroll", "roll_rate"),
        ("zx|roll", "roll_angle"),
    ]
    for ax, (col, label) in zip(axes, plot_specs):
        if col in vehicle.columns:
            y = vehicle[col].to_numpy(dtype=float)[mask]
            ax.plot(t, y, linewidth=1.0)
        ax.set_ylabel(label)
        ax.grid(alpha=0.18)
        ax.axvline(0, color="#111111", linewidth=1.0, linestyle="--")
        ts = safe_float(row.get("audit_t_steer_onset", np.nan))
        td = safe_float(row.get("audit_t_first_dynamic_onset", np.nan))
        if math.isfinite(ts):
            ax.axvline(ts - anchor, color="#4C78A8", linewidth=1.0, linestyle="--")
        if math.isfinite(td):
            ax.axvline(td - anchor, color="#E45756", linewidth=1.0, linestyle="--")
    axes[-1].set_xlabel("time around audit anchor (s)")
    title = (
        f"{row.get('episode_id_v0_6','')} | {row.get('road_design_module_name','')} | "
        f"{row.get('audit_latency_category_cn','')}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out)
    plt.close(fig)


def make_review_panels(merged: pd.DataFrame, vehicle_cache: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    categories = [
        "steer_first_gap_ge_0_5s",
        "steer_first_gap_0_2_to_0_5s",
        "near_sync_steer_first_lt_0_2s",
        "near_sync_vehicle_first_lt_0_2s",
        "vehicle_dynamic_first_ge_0_2s",
        "dynamic_detected_no_steer",
        "steer_detected_no_dynamic",
    ]
    for cat in categories:
        sub = merged[merged["audit_latency_category"] == cat].copy()
        if sub.empty:
            continue
        if "instability_review_score" in sub.columns:
            sub["_sort_score"] = pd.to_numeric(sub["instability_review_score"], errors="coerce").fillna(0)
            sub = sub.sort_values("_sort_score", ascending=False)
        for _, row in sub.head(4).iterrows():
            path_text = str(row.get("vehicle_raw_absolute_path", ""))
            vehicle = vehicle_cache.get(path_text)
            if vehicle is None:
                continue
            out = PANEL_DIR / f"{cat}__{str(row.get('episode_id_v0_6', 'event')).replace(':', '_')}.png"
            plot_review_panel(row, vehicle, out)
            if out.exists():
                rows.append(
                    {
                        "episode_id_v0_6": row.get("episode_id_v0_6", ""),
                        "latency_category": cat,
                        "latency_category_cn": category_cn(cat),
                        "road_design_module_name": row.get("road_design_module_name", ""),
                        "v0_6_final_bucket_cn": row.get("v0_6_final_bucket_cn", ""),
                        "delta_first_dynamic_minus_steer_s": row.get("audit_delta_first_dynamic_minus_steer_s", np.nan),
                        "figure_path": str(out),
                    }
                )
    out_index = TABLE_DIR / "latency_review_panel_index_v0_1.csv"
    panel_df = pd.DataFrame(rows)
    panel_df.to_csv(out_index, index=False, encoding="utf-8-sig")
    return panel_df


def pct(num: float) -> str:
    if not math.isfinite(num):
        return "NA"
    return f"{num * 100:.1f}%"


def write_reports(merged: pd.DataFrame, table_paths: dict[str, Path], figure_paths: list[Path], panel_df: pd.DataFrame) -> None:
    total = len(merged)
    valid = merged[np.isfinite(merged["audit_delta_first_dynamic_minus_steer_s"])]
    valid_n = len(valid)
    gap02 = float(valid["audit_gap_ge_0_2s"].mean()) if valid_n else float("nan")
    gap05 = float(valid["audit_gap_ge_0_5s"].mean()) if valid_n else float("nan")
    vehicle_first = float((valid["audit_delta_first_dynamic_minus_steer_s"] <= -0.20).mean()) if valid_n else float("nan")
    near_sync = float((valid["audit_delta_first_dynamic_minus_steer_s"].abs() < 0.20).mean()) if valid_n else float("nan")
    median_delta = float(valid["audit_delta_first_dynamic_minus_steer_s"].median()) if valid_n else float("nan")
    mean_delta = float(valid["audit_delta_first_dynamic_minus_steer_s"].mean()) if valid_n else float("nan")
    existing = pd.to_numeric(merged["existing_delta_dyn_minus_steer_s"], errors="coerce").dropna()
    existing_n = len(existing)
    existing_median = float(existing.median()) if existing_n else float("nan")
    existing_gap02 = float((existing >= 0.20).mean()) if existing_n else float("nan")
    existing_vehicle_first = float((existing <= -0.20).mean()) if existing_n else float("nan")
    core = merged[merged["v0_6_final_bucket_cn"].eq("第一版最干净核心训练候选")].copy()
    core_valid = pd.to_numeric(core["audit_delta_first_dynamic_minus_steer_s"], errors="coerce").dropna()
    core_valid_n = len(core_valid)
    core_gap02 = float((core_valid >= 0.20).mean()) if core_valid_n else float("nan")
    core_median = float(core_valid.median()) if core_valid_n else float("nan")

    cat_counts = Counter(merged["audit_latency_category_cn"].fillna("NA").tolist())
    cat_lines = "\n".join([f"- {k}：{v} 个，占 {v / max(total, 1) * 100:.1f}%" for k, v in cat_counts.most_common()])

    if gap02 >= 0.60:
        conclusion = "多数样本存在 0.2 秒以上的方向盘提前量，可以把“方向盘动作早期片段”作为一类输入设定继续验证。"
    elif gap02 >= 0.35:
        conclusion = "只有一部分样本存在 0.2 秒以上提前量，适合分类型使用，不能把所有事件都改成这种任务。"
    else:
        conclusion = "大多数样本没有稳定的 0.2 秒以上提前量，不能直接假设“方向盘先动很久后车辆才侧倾”。"

    user_report = REPORT_DIR / "stage02_steer_to_vehicle_dynamics_latency_user_summary_cn.md"
    user_report.write_text(
        f"""# 方向盘动作到车辆动态响应时间差审计（用户查看版）

## 这次为什么做

你提出了一个关键判断：如果所有事件本质上都是驾驶员主动打方向盘引起的，那么也许可以把“方向盘刚开始变化、车辆还没有明显侧倾/横摆/横向加速度响应”的这段时间作为输入，再预测后续方向盘轨迹或车辆动态变化。

所以这次没有训练模型，而是先审计时间关系：

- 方向盘什么时候开始明显变化；
- 横向加速度、横摆角速度、侧倾角速度、侧倾角什么时候开始明显变化；
- 二者之间是否有足够时间差；
- 0.2 秒和 0.5 秒早期输入窗口是否现实。

## 审计对象

- 使用 v0.6 事件候选表：`{EVENTS_CSV}`
- 共审计样本：{total} 个
- 成功得到方向盘与车辆动态时间差的样本：{valid_n} 个

## 核心结果

- 方向盘到第一类车辆动态响应的时间差中位数：{median_delta:.3f} 秒
- 平均时间差：{mean_delta:.3f} 秒
- 方向盘至少提前 0.2 秒的比例：{pct(gap02)}
- 方向盘至少提前 0.5 秒的比例：{pct(gap05)}
- 方向盘和车辆动态几乎同步（绝对差 < 0.2 秒）的比例：{pct(near_sync)}
- 车辆动态明显早于方向盘（车辆至少早 0.2 秒）的比例：{pct(vehicle_first)}

## 和 v0.6 旧检测字段交叉验证

v0.6 表里原本也有 `t_steer_onset` 和 `t_dyn_onset` 字段。用旧字段直接计算时：

- 可计算样本：{existing_n} 个
- 中位时间差：{existing_median:.3f} 秒
- 方向盘至少提前 0.2 秒的比例：{pct(existing_gap02)}
- 车辆动态至少早于方向盘 0.2 秒的比例：{pct(existing_vehicle_first)}

也就是说，旧字段和这次重新检测虽然具体起点不完全一致，但方向上是一致的：目前并不支持“多数样本中方向盘先明显动作，然后隔较长时间车辆才侧倾/横摆”的假设。

## 第一版核心干净样本的情况

v0.6 最干净核心训练候选共 {len(core)} 个，其中这次能计算明确时间差的有 {core_valid_n} 个：

- 核心样本中位时间差：{core_median:.3f} 秒
- 核心样本方向盘至少提前 0.2 秒的比例：{pct(core_gap02)}

这说明如果只看第一版最干净样本，也不能直接把任务改成“方向盘先动较长时间后预测车辆侧倾”。

## 分类统计

{cat_lines}

## 目前判断

{conclusion}

更具体地说，如果目标是“预测车辆什么时候侧倾/横摆/横向动态增强”，那么必须只挑方向盘确实提前的那一类样本；如果目标仍然是“预测后续方向盘轨迹”，那么方向盘已经开始变化后的 0.2 秒输入是可行的，但任务定义就变成了“早期动作后预测剩余轨迹”，不再是“事件发生前预测完整方向盘响应”。

## 对后续事件筛选的影响

建议把样本分成三类，而不是混在一起：

1. 方向盘明显先动：可以尝试“方向盘早期动作 → 后续车辆动态/剩余方向盘轨迹”。
2. 几乎同步：更适合做“动作发生后的短时延续预测”，不适合作为侧倾前预警。
3. 车辆动态先出现：更像“车辆扰动后驾驶员纠偏”，不应和主动打方向样本混训。

## 推荐你优先看

- 总表：`{table_paths['events']}`
- 汇总表：`{table_paths['summary']}`
- 分组表：`{table_paths['group']}`
- 分位数表：`{table_paths['quantile']}`
- 直方图：`{figure_paths[0] if figure_paths else '未生成'}`
- 代表性复核图索引：`{TABLE_DIR / 'latency_review_panel_index_v0_1.csv'}`

## 结论边界

这次是信号时间差审计，不是模型结果。它只能回答“这种任务设定有没有时间基础”，不能直接证明模型一定能预测得更好。下一步如果要训练，也应该按上述三类样本分别训练或分别评估。
""",
        encoding="utf-8",
    )

    tech_report = REPORT_DIR / "steer_to_vehicle_dynamics_latency_v0_1_cn.md"
    tech_report.write_text(
        f"""# 方向盘到车辆动态时间差审计 v0.1

## 方法

对每个 v0.6 episode，在最终训练锚点附近取 `t0-2s` 到 `t0+4s` 搜索窗口，并用 `t0-4s` 到 `t0-2s` 作为局部基线。重新检测：

- 方向盘起点：方向盘角偏离局部基线，或方向盘角速度显著升高且随后 0.5 秒内幅值确实变大；
- 横向加速度起点；
- 横摆角速度起点；
- 侧倾角速度起点；
- 侧倾角起点。

时间差定义为：

`第一类车辆动态起点 - 方向盘起点`

正值表示方向盘先动；负值表示车辆动态先出现。

## 输出文件

- 明细表：`{table_paths['events']}`
- 汇总表：`{table_paths['summary']}`
- 分组表：`{table_paths['group']}`
- 分位数表：`{table_paths['quantile']}`
- 复核图索引：`{TABLE_DIR / 'latency_review_panel_index_v0_1.csv'}`

## 关键数字

- 总样本：{total}
- 有有效时间差：{valid_n}
- 中位时间差：{median_delta:.4f} 秒
- 平均时间差：{mean_delta:.4f} 秒
- `gap >= 0.2s`：{pct(gap02)}
- `gap >= 0.5s`：{pct(gap05)}
- `abs(gap) < 0.2s`：{pct(near_sync)}
- `vehicle first <= -0.2s`：{pct(vehicle_first)}

## 旧字段交叉验证

- 旧字段可计算样本：{existing_n}
- 旧字段中位时间差：{existing_median:.4f} 秒
- 旧字段 `gap >= 0.2s`：{pct(existing_gap02)}
- 旧字段 `vehicle first <= -0.2s`：{pct(existing_vehicle_first)}

## 核心干净样本

- 核心样本总数：{len(core)}
- 核心样本有效时间差数：{core_valid_n}
- 核心样本中位时间差：{core_median:.4f} 秒
- 核心样本 `gap >= 0.2s`：{pct(core_gap02)}

## 分类统计

{cat_lines}

## 技术注意

1. 该审计没有使用未来方向盘峰值来定义方向盘起点，只用局部偏离和角速度变化检测起点。
2. 车辆动态响应同时看横向加速度、横摆角速度、侧倾角速度和侧倾角。
3. 如果搜索窗口一开始方向盘已经变化，会标记为“方向盘起点左截断”，这种样本不能精确判断提前量。
4. 该结果用于判断任务设定，不直接作为最终训练样本纳入标准。
""",
        encoding="utf-8",
    )

    append_project_notes(user_report, tech_report, table_paths, figure_paths, panel_df, conclusion)


def append_project_notes(
    user_report: Path,
    tech_report: Path,
    table_paths: dict[str, Path],
    figure_paths: list[Path],
    panel_df: pd.DataFrame,
    conclusion: str,
) -> None:
    DAILY_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not DAILY_LOG.exists():
        DAILY_LOG.write_text("# 2026-05-14 执行日志\n\n", encoding="utf-8")
    with DAILY_LOG.open("a", encoding="utf-8") as f:
        f.write(
            f"""## 方向盘动作到车辆动态响应时间差审计

- 为什么做：验证“方向盘先开始变化，到车辆侧倾/横摆/横向动态有时间差，能否把这段作为输入”的设想。
- 做了什么：基于 v0.6 episode 候选，从原始车辆 CSV 重新检测方向盘起点、横向加速度起点、横摆角速度起点、侧倾角速度起点和侧倾角起点。
- 主要输出：
  - 用户查看版：`{user_report}`
  - 技术报告：`{tech_report}`
  - 明细表：`{table_paths['events']}`
  - 复核图索引：`{TABLE_DIR / 'latency_review_panel_index_v0_1.csv'}`
- 当前判断：{conclusion}

"""
        )

    status_path = NOTES_DIR / "PROJECT_STATUS_CN.md"
    if status_path.exists():
        old = status_path.read_text(encoding="utf-8", errors="ignore")
    else:
        old = "# 项目状态\n\n"
    entry = f"""# 项目状态更新：方向盘到车辆动态时间差审计

更新时间：2026-05-14

当前阶段：旧流程事件样本与锚点继续审计。

当前刚完成：方向盘动作开始到横向/横摆/侧倾响应开始的时间差审计。

最近一次结果：{conclusion}

用户优先查看：

- `{user_report}`
- `{tech_report}`
- `{table_paths['events']}`
- `{figure_paths[0] if figure_paths else '图未生成'}`
- `{TABLE_DIR / 'latency_review_panel_index_v0_1.csv'}`

下一步建议：根据本次比例决定是否把任务拆成“方向盘早期动作预测剩余轨迹”“车辆扰动后纠偏”“几乎同步动作延续”三类，而不是继续混合训练。

---

"""
    status_path.write_text(entry + old, encoding="utf-8")

    artifact_path = NOTES_DIR / "ARTIFACT_INDEX_CN.md"
    if artifact_path.exists():
        old = artifact_path.read_text(encoding="utf-8", errors="ignore")
    else:
        old = "# 产物索引\n\n"
    figs = "\n".join([f"- `{p}`" for p in figure_paths])
    artifact_entry = f"""## 2026-05-14 方向盘到车辆动态时间差审计

- 用户查看版：`{user_report}`
- 技术报告：`{tech_report}`
- 明细表：`{table_paths['events']}`
- 汇总表：`{table_paths['summary']}`
- 分组表：`{table_paths['group']}`
- 分位数表：`{table_paths['quantile']}`
- 代表性复核图索引：`{TABLE_DIR / 'latency_review_panel_index_v0_1.csv'}`
- 代表性复核图数量：{len(panel_df)}

图表：
{figs}

"""
    artifact_path.write_text(artifact_entry + old, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    events = pd.read_csv(EVENTS_CSV, encoding="utf-8-sig")
    vehicle_cache: dict[str, pd.DataFrame | None] = {}
    audit_rows: list[dict[str, Any]] = []

    for idx, row in events.iterrows():
        path_text = str(row.get("vehicle_raw_absolute_path", ""))
        if path_text not in vehicle_cache:
            vehicle_cache[path_text] = load_vehicle(path_text)
        audit = detect_event_latency(row, vehicle_cache[path_text])
        audit["row_index"] = int(idx)
        audit_rows.append(audit)
        if (idx + 1) % 100 == 0:
            print(f"audited {idx + 1}/{len(events)}")

    audited = pd.DataFrame(audit_rows)
    table_paths = write_tables(events, audited)
    merged = pd.read_csv(table_paths["events"], encoding="utf-8-sig")
    figure_paths = plot_figures(merged)
    panel_df = make_review_panels(merged, vehicle_cache)
    write_reports(merged, table_paths, figure_paths, panel_df)

    valid = merged[np.isfinite(pd.to_numeric(merged["audit_delta_first_dynamic_minus_steer_s"], errors="coerce"))]
    print(f"done: total={len(merged)}, valid_delta={len(valid)}")
    if len(valid):
        print(
            "median_delta=",
            float(pd.to_numeric(valid["audit_delta_first_dynamic_minus_steer_s"], errors="coerce").median()),
        )
        print("gap_ge_0_2=", float(pd.to_numeric(valid["audit_gap_ge_0_2s"], errors="coerce").mean()))
        print("gap_ge_0_5=", float(pd.to_numeric(valid["audit_gap_ge_0_5s"], errors="coerce").mean()))
    print(f"report={REPORT_DIR / 'stage02_steer_to_vehicle_dynamics_latency_user_summary_cn.md'}")


if __name__ == "__main__":
    main()
