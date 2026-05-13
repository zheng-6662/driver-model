# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
PROJECT_ROOT = Path(r"F:/data_set_process/data_process")
RAW_ROOT = PROJECT_ROOT / "01_datasets" / "数据预处理"

CANDIDATE_PATH = (
    ROOT
    / "02_samples"
    / "ego_direction_design_anchor_v0_4"
    / "tables"
    / "ego_direction_design_anchor_candidates_v0_4.csv"
)
OLD_ANCHOR_PATH = (
    ROOT
    / "02_samples"
    / "road_event_anchor_audit_v0_1"
    / "tables"
    / "old_new_anchor_alignment_v0_1.csv"
)
HIGHCONF_EVENT_PATH = (
    ROOT
    / "02_samples"
    / "vehicle_instability_highconf_v0_1"
    / "tables"
    / "event_anchor_table.csv"
)

OUT_DIR = ROOT / "02_samples" / "event_candidate_filter_v0_5"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
PANEL_DIR = FIG_DIR / "review_panels"
REPORT_DIR = ROOT / "09_reports"

VEHICLE_COLS = [
    "StorageTime",
    "zx|SteeringWheel",
    "zx|ay",
    "zx|vyaw",
    "zx|vroll",
    "zx1|lateraldistance",
    "zx|BrakePedal",
    "zx|ax",
    "zx1|v_km/h",
    "zx1|mu",
]

HIGH_PRIORITY_MODULES = {
    "middle_section",
    "longstraight",
    "fix_road",
    "curve1",
    "curve2",
    "differentmu_road",
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def safe_max_abs(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.max(np.abs(arr)))


def safe_range(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.max(arr) - np.min(arr))


def safe_median(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def abs_change_from_baseline(series: pd.Series, baseline: float) -> float:
    if not math.isfinite(baseline):
        return float("nan")
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.max(np.abs(arr - baseline)))


def first_value_near(window: pd.DataFrame, column: str) -> float:
    if window.empty or column not in window:
        return float("nan")
    values = pd.to_numeric(window[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[0])


def peak_rate(window: pd.DataFrame, column: str) -> float:
    if window.empty or column not in window:
        return float("nan")
    tmp = window[["time_rel_s", column]].copy()
    tmp[column] = pd.to_numeric(tmp[column], errors="coerce")
    tmp = tmp.dropna()
    if len(tmp) < 3:
        return float("nan")
    t = tmp["time_rel_s"].to_numpy(dtype=float)
    y = tmp[column].to_numpy(dtype=float)
    dt = np.diff(t)
    dy = np.diff(y)
    good = np.isfinite(dt) & np.isfinite(dy) & (np.abs(dt) > 1e-6)
    if not np.any(good):
        return float("nan")
    return float(np.max(np.abs(dy[good] / dt[good])))


def load_vehicle(relative_path: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not relative_path:
        return pd.DataFrame()
    if relative_path in cache:
        return cache[relative_path]
    path = RAW_ROOT / relative_path
    if not path.exists():
        cache[relative_path] = pd.DataFrame()
        return cache[relative_path]
    try:
        header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
        usecols = [col for col in VEHICLE_COLS if col in header.columns]
        df = pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    except Exception:
        cache[relative_path] = pd.DataFrame()
        return cache[relative_path]
    if "StorageTime" not in df.columns:
        cache[relative_path] = pd.DataFrame()
        return cache[relative_path]
    raw_time = df["StorageTime"].copy()
    numeric_time = pd.to_numeric(raw_time, errors="coerce")
    if float(numeric_time.notna().mean()) >= 0.8:
        time_s = numeric_time.astype(float)
    else:
        dt = pd.to_datetime(raw_time, errors="coerce")
        time_s = pd.Series(np.where(dt.notna(), dt.astype("int64") / 1e9, np.nan), index=df.index)
    for col in df.columns:
        if col != "StorageTime":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["storage_time_s"] = time_s
    df = df.dropna(subset=["storage_time_s"]).sort_values("storage_time_s").drop_duplicates("storage_time_s")
    if df.empty:
        cache[relative_path] = pd.DataFrame()
        return cache[relative_path]
    for col in df.columns:
        if col != "StorageTime":
            df[col] = df[col].interpolate(limit_direction="both")
    df["time_rel_s"] = df["storage_time_s"] - float(df["storage_time_s"].iloc[0])
    cache[relative_path] = df
    return df


def nearest_delta(times: list[float], t: float) -> tuple[float, float]:
    if not times or not math.isfinite(t):
        return float("nan"), float("nan")
    arr = np.asarray(times, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    idx = int(np.argmin(np.abs(arr - t)))
    nearest = float(arr[idx])
    return nearest, float(nearest - t)


def load_time_lookup(path: Path, time_col: str) -> dict[tuple[str, str], list[float]]:
    if not path.exists():
        return {}
    df = read_csv(path)
    if df.empty or time_col not in df.columns:
        return {}
    lookup: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _, row in df.iterrows():
        subject = str(row.get("subject", ""))
        session = str(row.get("session_stamp", ""))
        t = finite_float(row.get(time_col))
        if subject and session and math.isfinite(t):
            lookup[(subject, session)].append(t)
    return {key: sorted(values) for key, values in lookup.items()}


def classify_candidate_role(anchor_type: str, source: str, module: str) -> tuple[str, float]:
    text = f"{anchor_type} {source}"
    if "显式" in text or "SILAB .aed" in text:
        return "场景显式触发候选", 1.00
    if "低附着" in text or "mu" in text or "cfg" in text:
        return "道路/任务设计候选", 0.90
    if "连续超车段入口" in text:
        return "连续任务段候选", 0.78
    if "道路模块入口" in text:
        return "道路/任务设计候选", 0.65
    if "道路模块中点" in text or "连续超车段中点" in text:
        return "道路段中间参考点", 0.42
    if "峰值" in text or "首次明显制动" in text:
        return "车身响应确认点", 0.58
    if "上下文" in text:
        return "场景上下文点", 0.25
    if module in HIGH_PRIORITY_MODULES:
        return "未分类高优先级候选", 0.45
    return "未分类普通候选", 0.30


def score_candidate_response(df: pd.DataFrame, t: float) -> dict[str, float | str]:
    if df.empty or not math.isfinite(t):
        return {
            "vehicle_window_status": "vehicle_missing_or_bad_time",
            "has_pre2s": 0,
            "has_post2s": 0,
        }
    time_min = float(df["time_rel_s"].min())
    time_max = float(df["time_rel_s"].max())
    pre2_ok = time_min <= t - 2.0
    post2_ok = time_max >= t + 2.0
    post3_ok = time_max >= t + 3.0
    pre = df[(df["time_rel_s"] >= t - 0.5) & (df["time_rel_s"] <= t)]
    pre2 = df[(df["time_rel_s"] >= t - 2.0) & (df["time_rel_s"] <= t)]
    post2 = df[(df["time_rel_s"] >= t) & (df["time_rel_s"] <= t + 2.0)]
    post3 = df[(df["time_rel_s"] >= t) & (df["time_rel_s"] <= t + 3.0)]
    post4 = df[(df["time_rel_s"] >= t) & (df["time_rel_s"] <= t + 4.0)]

    steer_base = safe_median(pre["zx|SteeringWheel"]) if "zx|SteeringWheel" in pre else float("nan")
    if not math.isfinite(steer_base) and "zx|SteeringWheel" in pre2:
        steer_base = safe_median(pre2["zx|SteeringWheel"])
    lat0 = first_value_near(post3, "zx1|lateraldistance")
    speed0 = first_value_near(post3, "zx1|v_km/h")
    mu0 = first_value_near(post4, "zx1|mu")

    lateral_delta = (
        abs_change_from_baseline(post3["zx1|lateraldistance"], lat0)
        if "zx1|lateraldistance" in post3
        else float("nan")
    )
    speed_min = safe_median(post3["zx1|v_km/h"]) if "zx1|v_km/h" in post3 else float("nan")
    if "zx1|v_km/h" in post3 and not post3.empty:
        values = pd.to_numeric(post3["zx1|v_km/h"], errors="coerce").dropna()
        speed_min = float(values.min()) if not values.empty else float("nan")
    mu_min = float("nan")
    if "zx1|mu" in post4 and not post4.empty:
        values = pd.to_numeric(post4["zx1|mu"], errors="coerce").dropna()
        mu_min = float(values.min()) if not values.empty else float("nan")

    return {
        "vehicle_window_status": "ok",
        "has_pre2s": int(pre2_ok),
        "has_post2s": int(post2_ok),
        "has_post3s": int(post3_ok),
        "pre2_row_count": int(len(pre2)),
        "post2_row_count": int(len(post2)),
        "steer_abs_change_post2": abs_change_from_baseline(post2["zx|SteeringWheel"], steer_base)
        if "zx|SteeringWheel" in post2
        else float("nan"),
        "steer_abs_change_post3": abs_change_from_baseline(post3["zx|SteeringWheel"], steer_base)
        if "zx|SteeringWheel" in post3
        else float("nan"),
        "steer_rate_peak_post2": peak_rate(post2, "zx|SteeringWheel"),
        "ay_abs_peak_post2": safe_max_abs(post2["zx|ay"]) if "zx|ay" in post2 else float("nan"),
        "yaw_abs_peak_post2": safe_max_abs(post2["zx|vyaw"]) if "zx|vyaw" in post2 else float("nan"),
        "roll_abs_peak_post2": safe_max_abs(post2["zx|vroll"]) if "zx|vroll" in post2 else float("nan"),
        "lateral_distance_delta_post3": lateral_delta,
        "brake_peak_post2": safe_max_abs(post2["zx|BrakePedal"]) if "zx|BrakePedal" in post2 else float("nan"),
        "ax_decel_peak_post2": max(0.0, -safe_median(post2["zx|ax"])) if "zx|ax" in post2 else float("nan"),
        "speed_drop_post3": max(0.0, speed0 - speed_min)
        if math.isfinite(speed0) and math.isfinite(speed_min)
        else float("nan"),
        "mu_drop_post4": max(0.0, mu0 - mu_min) if math.isfinite(mu0) and math.isfinite(mu_min) else float("nan"),
        "steer_pre_baseline": steer_base,
        "speed_at_anchor": speed0,
        "mu_at_anchor": mu0,
    }


def percentile_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if values.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(values)), index=series.index, dtype=float)
    return values.rank(pct=True, method="average")


def add_scores(scored: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "steer_abs_change_post2",
        "steer_rate_peak_post2",
        "ay_abs_peak_post2",
        "yaw_abs_peak_post2",
        "roll_abs_peak_post2",
        "lateral_distance_delta_post3",
        "brake_peak_post2",
        "ax_decel_peak_post2",
        "speed_drop_post3",
        "mu_drop_post4",
    ]
    for col in metric_cols:
        scored[f"rank_{col}"] = percentile_rank(scored[col]) if col in scored else 0.0

    lateral_score = (
        0.28 * scored["rank_steer_abs_change_post2"]
        + 0.18 * scored["rank_steer_rate_peak_post2"]
        + 0.18 * scored["rank_ay_abs_peak_post2"]
        + 0.16 * scored["rank_yaw_abs_peak_post2"]
        + 0.12 * scored["rank_lateral_distance_delta_post3"]
        + 0.08 * scored["rank_roll_abs_peak_post2"]
    )
    stop_score = (
        0.30 * scored["rank_brake_peak_post2"]
        + 0.25 * scored["rank_speed_drop_post3"]
        + 0.20 * scored["rank_ax_decel_peak_post2"]
        + 0.15 * scored["rank_steer_abs_change_post2"]
        + 0.10 * scored["rank_ay_abs_peak_post2"]
    )
    mu_score = (
        0.40 * scored["rank_mu_drop_post4"]
        + 0.20 * scored["rank_yaw_abs_peak_post2"]
        + 0.20 * scored["rank_ay_abs_peak_post2"]
        + 0.20 * scored["rank_steer_abs_change_post2"]
    )

    scored["module_response_score_0_1"] = lateral_score
    stop_mask = scored["module_name"].astype(str).eq("stop") | scored["candidate_anchor_type_cn"].astype(str).str.contains("停车|制动", na=False)
    mu_mask = scored["module_name"].astype(str).eq("differentmu_road") | scored["candidate_anchor_type_cn"].astype(str).str.contains("低附着|mu", na=False)
    scored.loc[stop_mask, "module_response_score_0_1"] = stop_score[stop_mask]
    scored.loc[mu_mask, "module_response_score_0_1"] = mu_score[mu_mask]

    close_old = pd.to_numeric(scored["nearest_old_anchor_abs_delta_s"], errors="coerce")
    old_support = np.where(close_old <= 1.0, 1.0, np.where(close_old <= 2.0, 0.65, np.where(close_old <= 4.0, 0.35, 0.0)))
    scored["old_anchor_support_0_1"] = old_support

    window_ok = (
        pd.to_numeric(scored["has_pre2s"], errors="coerce").fillna(0).astype(int).eq(1)
        & pd.to_numeric(scored["has_post2s"], errors="coerce").fillna(0).astype(int).eq(1)
    )
    scored["model_window_ok"] = window_ok.astype(int)
    scored["high_priority_module"] = scored["module_name"].astype(str).isin(HIGH_PRIORITY_MODULES).astype(int)

    scored["overall_screen_score_0_100"] = (
        100.0
        * (
            0.52 * scored["module_response_score_0_1"]
            + 0.35 * scored["causal_design_score_0_1"]
            + 0.08 * scored["old_anchor_support_0_1"]
            + 0.05 * scored["high_priority_module"]
        )
    )
    return scored


def decide(row: pd.Series) -> str:
    role = str(row.get("candidate_role_cn", ""))
    module = str(row.get("module_name", ""))
    response = finite_float(row.get("module_response_score_0_1"), 0.0)
    design = finite_float(row.get("causal_design_score_0_1"), 0.0)
    window_ok = int(finite_float(row.get("model_window_ok"), 0.0)) == 1
    anchor_type = str(row.get("candidate_anchor_type_cn", ""))

    if "场景上下文" in role or "上下文" in anchor_type:
        return "暂不进入样本-仅作场景上下文"
    if "车身响应确认点" in role:
        return "只作响应确认-不直接定因果锚点"
    if not window_ok:
        return "人工复核-窗口不足暂不能训练"
    if design >= 0.90 and response >= 0.42:
        return "优先复核-显式/道路设计触发且有响应"
    if module == "middle_section" and design >= 0.70 and response >= 0.55:
        return "优先复核-连续超车候选且有响应"
    if module in {"curve1", "curve2", "differentmu_road"} and design >= 0.60 and response >= 0.50:
        return "优先复核-道路设计候选且有响应"
    if design >= 0.65:
        return "人工复核-设计点明确但响应弱"
    return "暂不进入样本-证据不足"


def deduplicate_for_review(scored: pd.DataFrame, max_per_module: int = 80) -> pd.DataFrame:
    wanted = scored[
        scored["screening_decision_cn"].astype(str).str.startswith("优先复核")
        | scored["screening_decision_cn"].astype(str).str.startswith("人工复核")
    ].copy()
    wanted = wanted.sort_values("overall_screen_score_0_100", ascending=False)
    selected: list[pd.Series] = []
    seen: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    per_module: dict[str, int] = defaultdict(int)
    for _, row in wanted.iterrows():
        module = str(row["module_name"])
        if per_module[module] >= max_per_module:
            continue
        key = (str(row["subject"]), str(row["session_stamp"]), module)
        t = finite_float(row["candidate_time_rel_s"])
        if any(abs(t - old_t) < 1.0 for old_t in seen[key]):
            continue
        seen[key].append(t)
        per_module[module] += 1
        selected.append(row)
    return pd.DataFrame(selected)


def build_scores() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = read_csv(CANDIDATE_PATH)
    old_lookup = load_time_lookup(OLD_ANCHOR_PATH, "old_anchor_time_rel_s")
    highconf_lookup = load_time_lookup(HIGHCONF_EVENT_PATH, "anchor_time_rel_s")
    vehicle_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []

    for idx, candidate in candidates.iterrows():
        relative_path = str(candidate.get("vehicle_raw_relative_path", ""))
        t = finite_float(candidate.get("candidate_time_rel_s"))
        vehicle = load_vehicle(relative_path, vehicle_cache)
        metrics = score_candidate_response(vehicle, t)
        subject = str(candidate.get("subject", ""))
        session = str(candidate.get("session_stamp", ""))
        old_t, old_delta = nearest_delta(old_lookup.get((subject, session), []), t)
        hc_t, hc_delta = nearest_delta(highconf_lookup.get((subject, session), []), t)
        role, design_score = classify_candidate_role(
            str(candidate.get("candidate_anchor_type_cn", "")),
            str(candidate.get("candidate_source_cn", "")),
            str(candidate.get("module_name", "")),
        )
        row = candidate.to_dict()
        row["candidate_uid"] = (
            f"event_filter_v0_5__{subject}__{session}__{idx:05d}"
        )
        row["candidate_role_cn"] = role
        row["causal_design_score_0_1"] = design_score
        row["nearest_old_anchor_time_rel_s"] = old_t
        row["nearest_old_anchor_delta_s"] = old_delta
        row["nearest_old_anchor_abs_delta_s"] = abs(old_delta) if math.isfinite(old_delta) else float("nan")
        row["nearest_highconf_event_time_rel_s"] = hc_t
        row["nearest_highconf_event_delta_s"] = hc_delta
        row["nearest_highconf_event_abs_delta_s"] = abs(hc_delta) if math.isfinite(hc_delta) else float("nan")
        row.update(metrics)
        rows.append(row)

    scored = pd.DataFrame(rows)
    scored = add_scores(scored)
    scored["screening_decision_cn"] = scored.apply(decide, axis=1)
    review = deduplicate_for_review(scored)
    high_conf = review[
        review["screening_decision_cn"].astype(str).str.startswith("优先复核")
        & (pd.to_numeric(review["module_response_score_0_1"], errors="coerce") >= 0.55)
        & (pd.to_numeric(review["model_window_ok"], errors="coerce") == 1)
    ].copy()
    return scored, review, high_conf


def make_summary_tables(scored: pd.DataFrame, review: pd.DataFrame, high_conf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        scored.groupby(["module_name", "candidate_anchor_type_cn", "screening_decision_cn"])
        .agg(
            count=("candidate_uid", "size"),
            mean_score=("overall_screen_score_0_100", "mean"),
            mean_response=("module_response_score_0_1", "mean"),
            model_window_ok_rate=("model_window_ok", "mean"),
            median_old_anchor_abs_delta_s=("nearest_old_anchor_abs_delta_s", "median"),
        )
        .reset_index()
    )
    module_summary = (
        scored.groupby("module_name")
        .agg(
            candidate_count=("candidate_uid", "size"),
            review_count=("screening_decision_cn", lambda s: int(s.astype(str).str.contains("复核").sum())),
            priority_count=("screening_decision_cn", lambda s: int(s.astype(str).str.startswith("优先复核").sum())),
            response_confirm_only_count=("screening_decision_cn", lambda s: int(s.astype(str).str.startswith("只作响应确认").sum())),
            context_or_hold_count=("screening_decision_cn", lambda s: int(s.astype(str).str.startswith("暂不进入").sum())),
            mean_score=("overall_screen_score_0_100", "mean"),
            median_old_anchor_abs_delta_s=("nearest_old_anchor_abs_delta_s", "median"),
        )
        .reset_index()
    )
    module_summary["selected_review_count_after_dedup"] = module_summary["module_name"].map(
        review.groupby("module_name").size().to_dict()
    ).fillna(0).astype(int)
    module_summary["high_conf_count_after_dedup"] = module_summary["module_name"].map(
        high_conf.groupby("module_name").size().to_dict()
    ).fillna(0).astype(int)
    return summary, module_summary


def configure_plot_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_overview(scored: pd.DataFrame, module_summary: pd.DataFrame) -> None:
    configure_plot_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    decision_counts = scored["screening_decision_cn"].value_counts().sort_values()
    axes[0].barh(decision_counts.index, decision_counts.values, color="#4f7cac")
    axes[0].set_title("候选筛选决策数量")
    axes[0].set_xlabel("数量")

    ms = module_summary.sort_values("high_conf_count_after_dedup", ascending=True)
    x = np.arange(len(ms))
    axes[1].bar(x, ms["candidate_count"], label="候选总数", color="#d0d7de")
    axes[1].bar(x, ms["selected_review_count_after_dedup"], label="去重后复核", color="#80b918")
    axes[1].bar(x, ms["high_conf_count_after_dedup"], label="高置信复核", color="#e76f51")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ms["module_name"], rotation=35, ha="right")
    axes[1].set_title("分场景候选数量")
    axes[1].set_ylabel("数量")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "event_candidate_filter_overview_v0_5.png", dpi=180)
    plt.close(fig)


def relative_window(df: pd.DataFrame, t: float, before: float = 3.0, after: float = 4.0) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    win = df[(df["time_rel_s"] >= t - before) & (df["time_rel_s"] <= t + after)].copy()
    win["rel_to_candidate_s"] = win["time_rel_s"] - t
    return win


def plot_candidate_panel(row: pd.Series, vehicle_cache: dict[str, pd.DataFrame], out_path: Path) -> None:
    configure_plot_font()
    df = load_vehicle(str(row.get("vehicle_raw_relative_path", "")), vehicle_cache)
    t = finite_float(row.get("candidate_time_rel_s"))
    win = relative_window(df, t)
    if win.empty:
        return
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    rel = win["rel_to_candidate_s"]
    old_delta = finite_float(row.get("nearest_old_anchor_delta_s"))
    high_delta = finite_float(row.get("nearest_highconf_event_delta_s"))

    steer_base = finite_float(row.get("steer_pre_baseline"))
    if "zx|SteeringWheel" in win:
        y = win["zx|SteeringWheel"] - steer_base if math.isfinite(steer_base) else win["zx|SteeringWheel"]
        axes[0].plot(rel, y, color="#1f77b4", linewidth=1.2)
    axes[0].set_ylabel("方向盘")

    if "zx|ay" in win:
        axes[1].plot(rel, win["zx|ay"], label="横向加速度", color="#d62728", linewidth=1.0)
    if "zx|vyaw" in win:
        axes[1].plot(rel, win["zx|vyaw"], label="横摆角速度", color="#2ca02c", linewidth=1.0)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("横向动态")

    if "zx1|lateraldistance" in win:
        lat0 = first_value_near(win, "zx1|lateraldistance")
        axes[2].plot(rel, win["zx1|lateraldistance"] - lat0, color="#9467bd", linewidth=1.0)
    axes[2].set_ylabel("横向偏移")

    if "zx1|v_km/h" in win:
        axes[3].plot(rel, win["zx1|v_km/h"], label="车速", color="#ff7f0e", linewidth=1.0)
    if "zx|BrakePedal" in win:
        brake = pd.to_numeric(win["zx|BrakePedal"], errors="coerce")
        if brake.notna().any():
            bmax = float(brake.max()) if float(brake.max()) != 0 else 1.0
            axes[3].plot(rel, brake / bmax * 20.0, label="制动(缩放)", color="#111111", linewidth=0.9)
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].set_ylabel("速度/制动")

    if "zx1|mu" in win:
        axes[4].plot(rel, win["zx1|mu"], color="#8c564b", linewidth=1.0)
    axes[4].set_ylabel("附着系数")
    axes[4].set_xlabel("相对候选锚点时间/s")

    for ax in axes:
        ax.axvline(0, color="#e63946", linestyle="-", linewidth=1.2, label="候选锚点")
        if math.isfinite(old_delta) and -3.0 <= old_delta <= 4.0:
            ax.axvline(old_delta, color="#6c757d", linestyle="--", linewidth=1.0)
        if math.isfinite(high_delta) and -3.0 <= high_delta <= 4.0:
            ax.axvline(high_delta, color="#f4a261", linestyle=":", linewidth=1.0)
        ax.grid(True, alpha=0.25)

    title = (
        f"{row.get('module_name')} | {row.get('candidate_anchor_type_cn')} | "
        f"分数 {finite_float(row.get('overall_screen_score_0_100'), 0.0):.1f} | "
        f"{row.get('screening_decision_cn')}"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_review_panels(review: pd.DataFrame) -> pd.DataFrame:
    rows = []
    vehicle_cache: dict[str, pd.DataFrame] = {}
    review_sorted = review.sort_values(["module_name", "overall_screen_score_0_100"], ascending=[True, False])
    panel_candidates = []
    for module, group in review_sorted.groupby("module_name"):
        panel_candidates.append(group.head(8))
    if panel_candidates:
        panel_df = pd.concat(panel_candidates, ignore_index=True)
    else:
        panel_df = pd.DataFrame()
    for idx, row in panel_df.iterrows():
        module = str(row.get("module_name", "unknown"))
        out_path = PANEL_DIR / f"{idx + 1:03d}_{module}_{str(row.get('candidate_anchor_type_cn', 'candidate'))[:20]}.png"
        safe_path = Path(str(out_path).replace("/", "_slash_"))
        # Windows allows Chinese file names but not path separators from accidental text.
        out_path = PANEL_DIR / safe_path.name
        plot_candidate_panel(row, vehicle_cache, out_path)
        rows.append(
            {
                "candidate_uid": row.get("candidate_uid", ""),
                "module_name": module,
                "candidate_anchor_type_cn": row.get("candidate_anchor_type_cn", ""),
                "screening_decision_cn": row.get("screening_decision_cn", ""),
                "overall_screen_score_0_100": row.get("overall_screen_score_0_100", ""),
                "figure_path": str(out_path),
            }
        )
    return pd.DataFrame(rows)


def write_report(scored: pd.DataFrame, review: pd.DataFrame, high_conf: pd.DataFrame, module_summary: pd.DataFrame, panel_index: pd.DataFrame) -> None:
    lines = [
        "# 事件候选筛选 v0.5",
        "",
        f"生成时间：{now_str()}",
        "",
        "## 这次做了什么",
        "",
        "本轮没有训练模型，而是对 v0.4 生成的候选锚点做第一轮自动筛选。筛选目标不是直接产出最终训练样本，而是把 4519 个候选点分成：优先复核、人工复核、只作响应确认、暂不进入。",
        "",
        "筛选原则是：显式触发点和道路/任务设计点优先，但必须看触发点附近是否真的有被试车辆响应；车身姿态峰值只能作为确认点，不能直接当因果锚点。",
        "",
        "## 输入",
        "",
        f"- 候选锚点表：`{CANDIDATE_PATH}`",
        f"- 旧锚点对齐表：`{OLD_ANCHOR_PATH}`",
        f"- 高置信车辆失稳事件表：`{HIGHCONF_EVENT_PATH}`",
        "",
        "## 输出",
        "",
        f"- 全部候选评分表：`{TABLE_DIR / 'event_candidate_scores_v0_5.csv'}`",
        f"- 去重后复核清单：`{TABLE_DIR / 'event_candidates_for_review_v0_5.csv'}`",
        f"- 高置信复核清单：`{TABLE_DIR / 'event_candidates_high_confidence_v0_5.csv'}`",
        f"- 分场景汇总：`{TABLE_DIR / 'event_candidate_module_summary_v0_5.csv'}`",
        f"- 分类型汇总：`{TABLE_DIR / 'event_candidate_decision_summary_v0_5.csv'}`",
        f"- 图像索引：`{TABLE_DIR / 'event_candidate_review_panel_index_v0_5.csv'}`",
        f"- 概览图：`{FIG_DIR / 'event_candidate_filter_overview_v0_5.png'}`",
        f"- 代表性复核图目录：`{PANEL_DIR}`",
        "",
        "## 数量概览",
        "",
        f"- 输入候选：{len(scored)} 行",
        f"- 去重后建议复核：{len(review)} 行",
        f"- 去重后高置信复核：{len(high_conf)} 行",
        f"- 已生成代表性复核图：{len(panel_index)} 张",
        "",
        "分场景汇总：",
        "",
        "| 场景 | 候选总数 | 去重后复核 | 高置信复核 | 只作响应确认 | 暂不进入 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in module_summary.sort_values("module_name").iterrows():
        lines.append(
            f"| `{row['module_name']}` | {int(row['candidate_count'])} | "
            f"{int(row['selected_review_count_after_dedup'])} | {int(row['high_conf_count_after_dedup'])} | "
            f"{int(row['response_confirm_only_count'])} | {int(row['context_or_hold_count'])} |"
        )
    lines.extend(
        [
            "",
            "## 当前可以怎么理解",
            "",
            "1. `longstraight` 和 `fix_road` 的显式变道/停车触发已经进入候选审查，但仍不能直接当最终训练锚点。",
            "2. `middle_section` 的连接段入口属于连续任务段候选；如果触发后没有明显横向动态，只能算弱响应样本。",
            "3. 横向加速度峰值、横摆角速度峰值、横向偏移峰值等更适合确认响应是否发生，不适合单独作为因果触发点。",
            "4. 事件筛选下一步应看复核图，把候选锚点分成“可进入样本清单、偏早、偏晚、无明显响应、语义不清”。",
            "",
            "## 下一步建议",
            "",
            "1. 先人工看本轮代表性复核图，重点看 `longstraight`、`fix_road`、`middle_section`、`differentmu_road` 和 `curve1/curve2`。",
            "2. 对通过视觉复核的事件，再生成 v0.6 样本清单。",
            "3. 在 v0.6 样本清单固定前，不建议继续训练风格/生理模型。",
        ]
    )
    report = "\n".join(lines)
    (REPORT_DIR / "event_candidate_filter_v0_5_cn.md").write_text(report, encoding="utf-8")

    user_lines = [
        "# 阶段 2：事件候选筛选用户版说明 v0.5",
        "",
        f"生成时间：{now_str()}",
        "",
        "## 这一步为什么做",
        "",
        "现在我们已经知道每个场景大概有哪些设计点，但还不能把所有候选点直接拿去训练。因为有些点只是道路入口，有些是背景，有些是车辆已经响应后的峰值。为了避免样本继续错位，这一步先把事件候选筛一遍。",
        "",
        "## 目前做到什么程度",
        "",
        f"我把 {len(scored)} 个候选点都和原始车辆数据对齐，计算了触发点前后方向盘、横向加速度、横摆角速度、横向偏移、制动、车速、路面附着变化等指标。",
        "",
        f"自动筛完后，去重得到 {len(review)} 个建议复核的事件，其中 {len(high_conf)} 个属于高置信复核候选。这里的“高置信”仍然只是进入复核，不等于最终训练样本。",
        "",
        "## 你应该怎么看这个结果",
        "",
        "优先看代表性复核图。每张图里红线是候选锚点，灰线如果出现则是最近旧锚点，橙线如果出现则是之前高置信车辆失稳点。我们要判断红线是不是比旧锚点更接近真实场景触发。",
        "",
        "## 重点文件",
        "",
        f"1. 中文报告：`{REPORT_DIR / 'event_candidate_filter_v0_5_cn.md'}`",
        f"2. 复核清单：`{TABLE_DIR / 'event_candidates_for_review_v0_5.csv'}`",
        f"3. 高置信复核清单：`{TABLE_DIR / 'event_candidates_high_confidence_v0_5.csv'}`",
        f"4. 分场景汇总：`{TABLE_DIR / 'event_candidate_module_summary_v0_5.csv'}`",
        f"5. 概览图：`{FIG_DIR / 'event_candidate_filter_overview_v0_5.png'}`",
        f"6. 代表性复核图目录：`{PANEL_DIR}`",
        "",
        "## 当前还不能下的结论",
        "",
        "不能说这些事件已经是最终样本。下一步必须看图，把候选分成可保留、偏早、偏晚、无明显响应和语义不清。只有视觉和物理意义都合理的事件，才进入新的样本清单。",
    ]
    (REPORT_DIR / "stage02_event_filter_user_summary_cn.md").write_text("\n".join(user_lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    scored, review, high_conf = build_scores()
    decision_summary, module_summary = make_summary_tables(scored, review, high_conf)
    write_csv(scored, TABLE_DIR / "event_candidate_scores_v0_5.csv")
    write_csv(review, TABLE_DIR / "event_candidates_for_review_v0_5.csv")
    write_csv(high_conf, TABLE_DIR / "event_candidates_high_confidence_v0_5.csv")
    write_csv(decision_summary, TABLE_DIR / "event_candidate_decision_summary_v0_5.csv")
    write_csv(module_summary, TABLE_DIR / "event_candidate_module_summary_v0_5.csv")
    plot_overview(scored, module_summary)
    panel_index = plot_review_panels(review)
    write_csv(panel_index, TABLE_DIR / "event_candidate_review_panel_index_v0_5.csv")
    write_report(scored, review, high_conf, module_summary, panel_index)
    print(
        {
            "scored": len(scored),
            "review": len(review),
            "high_conf": len(high_conf),
            "panels": len(panel_index),
        }
    )


if __name__ == "__main__":
    main()
