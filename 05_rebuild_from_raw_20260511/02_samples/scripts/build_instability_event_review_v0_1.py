# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
INPUT_EVENTS = ROOT / "02_samples" / "tables" / "candidate_events_master.csv"
OUT_DIR = ROOT / "02_samples" / "instability_event_review_v0_1"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"

MERGE_GAP_S = 2.5
REVIEW_PRE_S = 5.0
REVIEW_POST_S = 8.0

VEHICLE_COLS = [
    "StorageTime",
    "zx|SteeringWheel",
    "zx|roll",
    "zx|vroll",
    "zx|vyaw",
    "zx|ay",
    "zx|vx",
    "zx|vy",
    "zx1|v_km/h",
    "zx1|lateraldistance",
    "zx|lateraldistance",
    "zx1|lanecurvatureXY",
    "zx|lanecurvatureXY",
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def to_seconds(storage_time: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(storage_time, errors="coerce")
    out = np.full(len(storage_time), np.nan, dtype=np.float64)
    valid = parsed.notna().to_numpy()
    if valid.any():
        parsed_ns = parsed[valid].astype("datetime64[ns]")
        ns = parsed_ns.astype("int64").to_numpy(dtype=np.float64)
        out[valid] = ns / 1e9
    return out


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def choose_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def max_abs(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.nanmax(np.abs(values)))


def signed_at_abs_max(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    v = values[finite]
    return float(v[int(np.nanargmax(np.abs(v)))])


def robust_range(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    return float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5))


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def load_vehicle(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in VEHICLE_COLS)
    if "StorageTime" not in df.columns:
        raise ValueError("missing StorageTime")
    t_abs = to_seconds(df["StorageTime"])
    valid = np.isfinite(t_abs)
    df = df.loc[valid].copy()
    t_abs = t_abs[valid]
    if len(df) == 0:
        raise ValueError("no valid timestamp rows")
    df["t_abs_s"] = t_abs
    df["t_rel_s"] = t_abs - float(t_abs[0])
    df = df.sort_values("t_rel_s").drop_duplicates("t_rel_s", keep="first").reset_index(drop=True)
    for col in df.columns:
        if col not in {"StorageTime", "t_abs_s", "t_rel_s"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def relative_window(df: pd.DataFrame, start_s: float, end_s: float) -> pd.DataFrame:
    return df[(df["t_rel_s"] >= start_s) & (df["t_rel_s"] <= end_s)]


def merge_dynamic_seeds(seeds: pd.DataFrame) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seeds = seeds.sort_values(["subject", "session_stamp", "event_start_rel_s", "anchor_time_rel_s"])
    for (subject, session), group in seeds.groupby(["subject", "session_stamp"], sort=False):
        current: dict[str, Any] | None = None
        for _, row in group.iterrows():
            start = finite_float(row.get("event_start_rel_s")) or finite_float(row.get("anchor_time_rel_s"))
            end = finite_float(row.get("event_end_rel_s")) or finite_float(row.get("anchor_time_rel_s"))
            anchor = finite_float(row.get("anchor_time_rel_s"))
            if start is None or end is None or anchor is None:
                continue
            if current is None or start > float(current["event_end_rel_s"]) + MERGE_GAP_S:
                if current is not None:
                    merged.append(current)
                current = {
                    "subject": subject,
                    "session_stamp": session,
                    "vehicle_raw_relative_path": row.get("vehicle_raw_relative_path", ""),
                    "vehicle_raw_absolute_path": row.get("vehicle_raw_absolute_path", ""),
                    "vehicle_raw_sha256": row.get("vehicle_raw_sha256", ""),
                    "event_start_rel_s": float(start),
                    "event_end_rel_s": float(end),
                    "anchor_time_rel_s": float(anchor),
                    "source_event_uids": [str(row.get("event_uid", ""))],
                    "source_event_types": [str(row.get("event_type", ""))],
                    "source_event_levels": [str(row.get("event_level", ""))],
                    "source_anchor_values": [finite_float(row.get("anchor_value"))],
                    "source_peak_scores": [finite_float(row.get("raw_dynamic_peak_score"))],
                }
                continue
            current["event_end_rel_s"] = max(float(current["event_end_rel_s"]), float(end))
            current["source_event_uids"].append(str(row.get("event_uid", "")))
            current["source_event_types"].append(str(row.get("event_type", "")))
            current["source_event_levels"].append(str(row.get("event_level", "")))
            current["source_anchor_values"].append(finite_float(row.get("anchor_value")))
            current["source_peak_scores"].append(finite_float(row.get("raw_dynamic_peak_score")))
        if current is not None:
            merged.append(current)
    return merged


def count_nearby(events: pd.DataFrame, subject: str, session: str, center_s: float, lo_s: float, hi_s: float) -> pd.DataFrame:
    if events.empty:
        return events
    mask = (
        (events["subject"] == subject)
        & (events["session_stamp"] == session)
        & (events["anchor_time_rel_s"] >= center_s + lo_s)
        & (events["anchor_time_rel_s"] <= center_s + hi_s)
    )
    return events.loc[mask]


def compute_roll_rate(df: pd.DataFrame) -> pd.Series:
    if "zx|vroll" in df.columns:
        return pd.to_numeric(df["zx|vroll"], errors="coerce")
    if "zx|roll" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    roll = pd.to_numeric(df["zx|roll"], errors="coerce").to_numpy(dtype=np.float64)
    t = df["t_rel_s"].to_numpy(dtype=np.float64)
    out = np.full(len(df), np.nan, dtype=np.float64)
    if len(df) >= 3:
        dt = np.gradient(t)
        dr = np.gradient(roll)
        valid = np.isfinite(dt) & (np.abs(dt) > 1e-6)
        out[valid] = dr[valid] / dt[valid]
    return pd.Series(out, index=df.index)


def classify_instability_role(type_counts: dict[str, int]) -> str:
    has_ay = type_counts.get("ay", 0) > 0
    has_roll = type_counts.get("roll_rate", 0) > 0
    if has_ay and has_roll:
        return "instability_ay_roll"
    if has_roll:
        return "instability_roll_only"
    return "instability_ay_only"


def score_episode(row: dict[str, Any]) -> tuple[float, str]:
    score = 0.0
    ay = float(row.get("peak_abs_ay_window", 0.0) or 0.0)
    roll_rate = float(row.get("peak_abs_roll_rate_window", 0.0) or 0.0)
    yaw_rate = float(row.get("peak_abs_yaw_rate_window", 0.0) or 0.0)
    lateral = float(row.get("lateral_distance_range_window", 0.0) or 0.0)
    steering_after = float(row.get("steering_delta_peak_post3s", 0.0) or 0.0)
    speed = float(row.get("median_speed_kmh_window", 0.0) or 0.0)
    duration = float(row.get("event_duration_s", 0.0) or 0.0)
    count_ay = int(row.get("ay_seed_count", 0) or 0)
    count_roll = int(row.get("roll_rate_seed_count", 0) or 0)
    point_count = int(row.get("review_point_count", 0) or 0)

    score += clip((ay - 1.0) / 5.0, 0.0, 1.0) * 30.0
    score += clip(roll_rate / 0.8, 0.0, 1.0) * 22.0
    score += clip(yaw_rate / 0.35, 0.0, 1.0) * 10.0
    score += clip(lateral / 0.8, 0.0, 1.0) * 8.0
    score += clip(duration / 2.0, 0.0, 1.0) * 8.0
    score += clip(steering_after / 8.0, 0.0, 1.0) * 7.0
    if count_ay > 0 and count_roll > 0:
        score += 10.0
    elif count_ay >= 2:
        score += 5.0
    elif count_roll >= 1:
        score += 4.0
    if speed >= 10.0:
        score += 5.0
    elif speed < 2.0:
        score -= 15.0
    if point_count < 100:
        score -= 10.0

    score = round(clip(score, 0.0, 100.0), 2)
    if score >= 75.0:
        decision = "auto_accept_instability_high"
    elif score >= 58.0:
        decision = "auto_accept_instability_medium"
    elif score >= 38.0:
        decision = "needs_human_review"
    else:
        decision = "reject_low_instability_evidence"
    return score, decision


def review_episode(
    episode: dict[str, Any],
    all_events: pd.DataFrame,
    vehicle_cache: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    subject = str(episode["subject"])
    session = str(episode["session_stamp"])
    anchor = float(episode["anchor_time_rel_s"])
    start = float(episode["event_start_rel_s"])
    end = float(episode["event_end_rel_s"])
    path = str(episode["vehicle_raw_absolute_path"])

    type_counts = {k: int(v) for k, v in pd.Series(episode["source_event_types"]).value_counts().to_dict().items()}
    values = [v for v in episode["source_anchor_values"] if v is not None and math.isfinite(v)]
    peak_scores = [v for v in episode["source_peak_scores"] if v is not None and math.isfinite(v)]
    result: dict[str, Any] = {
        "instability_event_uid": f"vehicle_instability_onset__{subject}__{session}__{int(round(anchor * 1000)):09d}",
        "dataset_candidate_version": "vehicle_instability_onset_codex_v0_1",
        "subject": subject,
        "session_stamp": session,
        "vehicle_raw_relative_path": episode["vehicle_raw_relative_path"],
        "vehicle_raw_absolute_path": path,
        "vehicle_raw_sha256": episode["vehicle_raw_sha256"],
        "anchor_time_rel_s": round(anchor, 6),
        "event_start_rel_s": round(start, 6),
        "event_end_rel_s": round(end, 6),
        "event_duration_s": round(max(0.0, end - start), 6),
        "instability_anchor_source": "raw_vehicle_dynamic_onset_non_steering",
        "instability_role": classify_instability_role(type_counts),
        "ay_seed_count": type_counts.get("ay", 0),
        "roll_rate_seed_count": type_counts.get("roll_rate", 0),
        "merged_seed_count": len(episode["source_event_types"]),
        "source_event_uids": ";".join(episode["source_event_uids"]),
        "source_event_types": ";".join(episode["source_event_types"]),
        "source_event_levels": ";".join(episode["source_event_levels"]),
        "max_source_anchor_value": round(max(values), 6) if values else np.nan,
        "max_source_peak_score": round(max(peak_scores), 6) if peak_scores else np.nan,
    }

    nearby_old = count_nearby(all_events[all_events["anchor_source"] == "old_v400_context_trigger_idx"], subject, session, anchor, -2.0, 5.0)
    nearby_road = count_nearby(all_events[all_events["anchor_source"] == "raw_road_curvature_onset"], subject, session, anchor, -5.0, 5.0)
    nearby_steer = count_nearby(
        all_events[(all_events["anchor_source"] == "raw_vehicle_dynamic_onset") & (all_events["event_type"] == "steer_rate")],
        subject,
        session,
        anchor,
        0.0,
        4.0,
    )
    result["nearby_old_context_count_m2_p5s"] = int(len(nearby_old))
    result["nearby_road_curvature_count_m5_p5s"] = int(len(nearby_road))
    result["nearby_steer_rate_count_p0_p4s"] = int(len(nearby_steer))

    try:
        if path not in vehicle_cache:
            vehicle_cache[path] = load_vehicle(path)
        df = vehicle_cache[path]
        review = relative_window(df, anchor - REVIEW_PRE_S, anchor + REVIEW_POST_S).copy()
        pre = relative_window(df, anchor - 1.0, anchor)
        post3 = relative_window(df, anchor, anchor + 3.0)
        event_window = relative_window(df, start, end)

        result["vehicle_read_status"] = "ok"
        result["vehicle_read_error"] = ""
        result["review_point_count"] = int(len(review))
        result["event_point_count"] = int(len(event_window))

        ay_col = choose_col(review, ["zx|ay"])
        yaw_col = choose_col(review, ["zx|vyaw"])
        steering_col = choose_col(review, ["zx|SteeringWheel"])
        speed_col = choose_col(review, ["zx1|v_km/h"])
        lateral_col = choose_col(review, ["zx1|lateraldistance", "zx|lateraldistance"])
        curvature_col = choose_col(review, ["zx1|lanecurvatureXY", "zx|lanecurvatureXY"])

        if len(review) > 0:
            review["roll_rate_for_review"] = compute_roll_rate(review)
        else:
            review["roll_rate_for_review"] = pd.Series(dtype=float)

        result["peak_abs_ay_window"] = round(max_abs(review[ay_col].to_numpy()) if ay_col else np.nan, 6)
        result["signed_peak_ay_window"] = round(signed_at_abs_max(review[ay_col].to_numpy()) if ay_col else np.nan, 6)
        result["peak_abs_ay_event"] = round(max_abs(event_window[ay_col].to_numpy()) if ay_col and len(event_window) else np.nan, 6)
        result["peak_abs_roll_rate_window"] = round(max_abs(review["roll_rate_for_review"].to_numpy()), 6)
        result["peak_abs_yaw_rate_window"] = round(max_abs(review[yaw_col].to_numpy()) if yaw_col else np.nan, 6)
        result["lateral_distance_range_window"] = round(robust_range(review[lateral_col].to_numpy()) if lateral_col else np.nan, 6)
        result["peak_abs_curvature_window"] = round(max_abs(review[curvature_col].to_numpy()) if curvature_col else np.nan, 8)
        result["median_speed_kmh_window"] = round(float(np.nanmedian(review[speed_col].to_numpy())) if speed_col and len(review) else np.nan, 6)

        if steering_col and len(pre) > 0 and len(post3) > 0:
            baseline = float(np.nanmedian(pre[steering_col].to_numpy(dtype=np.float64)))
            post_values = post3[steering_col].to_numpy(dtype=np.float64) - baseline
            result["steering_baseline_pre1s"] = round(baseline, 6)
            result["steering_delta_peak_post3s"] = round(max_abs(post_values), 6)
            result["steering_signed_delta_peak_post3s"] = round(signed_at_abs_max(post_values), 6)
        else:
            result["steering_baseline_pre1s"] = np.nan
            result["steering_delta_peak_post3s"] = np.nan
            result["steering_signed_delta_peak_post3s"] = np.nan
    except Exception as exc:  # noqa: BLE001
        result["vehicle_read_status"] = "error"
        result["vehicle_read_error"] = str(exc)
        result["review_point_count"] = 0
        result["event_point_count"] = 0
        for col in [
            "peak_abs_ay_window",
            "signed_peak_ay_window",
            "peak_abs_ay_event",
            "peak_abs_roll_rate_window",
            "peak_abs_yaw_rate_window",
            "lateral_distance_range_window",
            "peak_abs_curvature_window",
            "median_speed_kmh_window",
            "steering_baseline_pre1s",
            "steering_delta_peak_post3s",
            "steering_signed_delta_peak_post3s",
        ]:
            result[col] = np.nan

    score, decision = score_episode(result)
    result["instability_review_score"] = score
    result["codex_recommended_decision"] = decision
    result["causal_setting"] = "detected_vehicle_instability_onset_predict_future_steering_response"
    result["leakage_note"] = (
        "Anchor is derived from non-steering vehicle dynamics (ay/roll_rate). "
        "Steering metrics are only response evidence and must not be used to define onset."
    )
    result["human_review_priority"] = (
        "high" if decision == "needs_human_review" and result.get("nearby_road_curvature_count_m5_p5s", 0) > 0 else "normal"
    )
    return result


def plot_overview(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=140)
    axes = axes.ravel()

    df["instability_review_score"].hist(ax=axes[0], bins=30, color="#4c78a8")
    axes[0].set_title("Instability review score")
    axes[0].set_xlabel("score")
    axes[0].set_ylabel("count")

    decision_counts = df["codex_recommended_decision"].value_counts()
    axes[1].bar(range(len(decision_counts)), decision_counts.values, color="#59a14f")
    axes[1].set_xticks(range(len(decision_counts)))
    axes[1].set_xticklabels(decision_counts.index, rotation=35, ha="right", fontsize=8)
    axes[1].set_title("Recommended decision")
    axes[1].set_ylabel("count")

    axes[2].scatter(df["peak_abs_ay_window"], df["steering_delta_peak_post3s"], s=10, alpha=0.55, color="#f28e2b")
    axes[2].set_xlabel("peak |ay| in review window")
    axes[2].set_ylabel("future steering delta peak")
    axes[2].set_title("Dynamic severity vs response evidence")

    role_counts = df["instability_role"].value_counts()
    axes[3].bar(range(len(role_counts)), role_counts.values, color="#e15759")
    axes[3].set_xticks(range(len(role_counts)))
    axes[3].set_xticklabels(role_counts.index, rotation=25, ha="right", fontsize=8)
    axes[3].set_title("Instability role")
    axes[3].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "instability_event_score_overview_v0_1.png")
    plt.close(fig)


def plot_examples(df: pd.DataFrame, vehicle_cache: dict[str, pd.DataFrame]) -> list[str]:
    selected = (
        df[df["codex_recommended_decision"].isin(["auto_accept_instability_high", "needs_human_review"])]
        .sort_values(["codex_recommended_decision", "instability_review_score"], ascending=[True, False])
        .head(24)
    )
    out_paths: list[str] = []
    for _, row in selected.iterrows():
        path = str(row["vehicle_raw_absolute_path"])
        try:
            if path not in vehicle_cache:
                vehicle_cache[path] = load_vehicle(path)
            vehicle = vehicle_cache[path]
            anchor = float(row["anchor_time_rel_s"])
            start = float(row["event_start_rel_s"])
            end = float(row["event_end_rel_s"])
            review = relative_window(vehicle, anchor - REVIEW_PRE_S, anchor + REVIEW_POST_S).copy()
            if review.empty:
                continue
            t = review["t_rel_s"].to_numpy(dtype=np.float64) - anchor
            fig, axes = plt.subplots(4, 1, figsize=(9, 7), dpi=130, sharex=True)
            signal_specs = [
                ("zx|ay", "ay", "#4c78a8"),
                ("zx|roll", "roll", "#59a14f"),
                ("zx|vyaw", "yaw rate", "#f28e2b"),
                ("zx|SteeringWheel", "steering", "#e15759"),
            ]
            for ax, (col, label, color) in zip(axes, signal_specs):
                if col in review.columns:
                    y = review[col].to_numpy(dtype=np.float64)
                    if label == "steering":
                        pre = review[(review["t_rel_s"] >= anchor - 1.0) & (review["t_rel_s"] <= anchor)]
                        if len(pre) > 0:
                            y = y - float(np.nanmedian(pre[col].to_numpy(dtype=np.float64)))
                    ax.plot(t, y, color=color, linewidth=1.2)
                ax.axvline(0.0, color="#111111", linewidth=1.2)
                ax.axvspan(start - anchor, end - anchor, color="#4c78a8", alpha=0.12)
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.2)
            axes[-1].set_xlabel("seconds around instability onset")
            fig.suptitle(
                f"{row['instability_event_uid']} | {row['codex_recommended_decision']} | score={row['instability_review_score']}",
                fontsize=9,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            out = FIG_DIR / f"{row['instability_event_uid']}.png"
            fig.savefig(out)
            plt.close(fig)
            out_paths.append(str(out).replace("\\", "/"))
        except Exception:
            continue
    return out_paths


def write_report(df: pd.DataFrame, accepted: pd.DataFrame, review: pd.DataFrame, example_paths: list[str]) -> None:
    decision_counts = df["codex_recommended_decision"].value_counts().to_string()
    role_counts = df["instability_role"].value_counts().to_string()
    subject_counts = df.groupby("subject").size().sort_index().to_string()
    seed_counts = df[["ay_seed_count", "roll_rate_seed_count", "merged_seed_count"]].sum().to_string()
    report = f"""# 阶段 2 修正：车辆失稳事件自动审阅 v0.1

生成时间：2026-05-12

## 为什么重做

用户指出，上一版 `codex_event_review_v0_1` 的 404 个样本本质上是弯道/道路曲率候选，不是本项目真正要找的车辆失稳样本。这个判断是正确的。

因此本版把道路曲率样本降级为道路上下文参考，重新用车辆动态异常来建立候选事件。主锚点不再来自弯道开始/结束，而来自原始车辆信号中的非方向盘动态异常。

## 本版事件定义

- 主事件：车辆失稳候选。
- 主锚点来源：`raw_vehicle_dynamic_onset` 中的 `ay` 和 `roll_rate`。
- 不作为主锚点：`raw_road_curvature_onset`，因为它只说明道路是弯的，不等于车辆失稳。
- 不作为主锚点：`steer_rate`，因为它已经是驾驶员方向盘动作，直接用它找事件会把响应结果混入事件定义。
- 方向盘信号只用于事件后响应证据：例如失稳后 3 秒内方向盘是否出现明显修正。

## 因果设定

本版对应的是：

`检测到车辆失稳动态开始后，预测未来方向盘响应轨迹`

这和“进入弯道前预测未来方向盘”不是同一个任务。后续建模时必须把这个数据版本单独命名，不能和弯道事件混在一起。

## 输入

- 候选事件总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/candidate_events_master.csv`
- 原始车辆 CSV：`原始车辆数据/<被试名>/Entity_Recording_*_vehicle.csv`
- 动态种子：`ay`、`roll_rate`
- 辅助上下文：附近弯道候选、旧流程候选、事件后方向盘响应

## 输出

- 全量失稳审阅表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_reviewed_events_v0_1.csv`
- 自动采用失稳候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_auto_accepted_events_v0_1.csv`
- 需要人工复核候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_needs_human_review_v0_1.csv`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_event_review_summary_v0_1.csv`
- 概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/instability_event_score_overview_v0_1.png`

## 当前数量

合并后的车辆失稳候选总数：{len(df)}

自动采用候选数：{len(accepted)}

需要人工复核候选数：{len(review)}

按决策统计：

```text
{decision_counts}
```

按失稳证据类型统计：

```text
{role_counts}
```

按被试统计：

```text
{subject_counts}
```

动态种子使用量：

```text
{seed_counts}
```

## 当前解释

这版比 404 个弯道样本更接近用户目标，因为它把“车辆有没有出现动态异常”放在主位，而不是把“道路是不是弯道”当成事件。

但它仍然不是最终真值标签。它是 Codex 根据规则做的第一轮失稳候选筛选，后面还要做三件事：

1. 检查高分样本是否真的表现为车辆失稳，而不是正常高速过弯。
2. 检查失稳锚点之后的方向盘响应窗口是否完整覆盖了修正过程。
3. 明确后续模型任务是“失稳检测后预测响应”，还是“失稳发生前预警并预测响应”。两者输入窗口不能混用。

## 和上一版 404 个样本的关系

上一版 `codex_event_review_v0_1` 不再作为主事件样本。它只能作为道路曲率上下文、弯道背景或对照材料保存。

## 推荐优先查看

- 先看全量表：`instability_reviewed_events_v0_1.csv`
- 再看自动采用表：`instability_auto_accepted_events_v0_1.csv`
- 再看概览图：`instability_event_score_overview_v0_1.png`
- 抽查示例图目录：`instability_event_review_v0_1/figures/`

## 示例图

```text
{chr(10).join(example_paths[:12])}
```
"""
    (REPORT_DIR / "instability_event_review_v0_1_cn.md").write_text(report, encoding="utf-8")


def write_summary_table(df: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    rows.append({"metric": "total_reviewed_instability_events", "value": len(df)})
    for key, value in df["codex_recommended_decision"].value_counts().items():
        rows.append({"metric": f"decision__{key}", "value": int(value)})
    for key, value in df["instability_role"].value_counts().items():
        rows.append({"metric": f"role__{key}", "value": int(value)})
    rows.append({"metric": "subject_count", "value": int(df["subject"].nunique())})
    rows.append({"metric": "session_count", "value": int(df["session_stamp"].nunique())})
    rows.append({"metric": "accepted_count", "value": int(df["codex_recommended_decision"].str.startswith("auto_accept").sum())})
    rows.append({"metric": "needs_human_review_count", "value": int((df["codex_recommended_decision"] == "needs_human_review").sum())})
    pd.DataFrame(rows).to_csv(TABLE_DIR / "instability_event_review_summary_v0_1.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    all_events = pd.read_csv(INPUT_EVENTS)
    for col in ["anchor_time_rel_s", "event_start_rel_s", "event_end_rel_s"]:
        all_events[col] = pd.to_numeric(all_events[col], errors="coerce")

    seeds = all_events[
        (all_events["anchor_source"] == "raw_vehicle_dynamic_onset")
        & (all_events["event_type"].isin(["ay", "roll_rate"]))
    ].copy()
    episodes = merge_dynamic_seeds(seeds)

    vehicle_cache: dict[str, pd.DataFrame] = {}
    reviewed = [review_episode(ep, all_events, vehicle_cache) for ep in episodes]
    df = pd.DataFrame(reviewed)
    if not df.empty:
        df = df.sort_values(["subject", "session_stamp", "anchor_time_rel_s"]).reset_index(drop=True)

    accepted = df[df["codex_recommended_decision"].str.startswith("auto_accept")].copy()
    needs_review = df[df["codex_recommended_decision"] == "needs_human_review"].copy()

    df.to_csv(TABLE_DIR / "instability_reviewed_events_v0_1.csv", index=False, encoding="utf-8-sig")
    accepted.to_csv(TABLE_DIR / "instability_auto_accepted_events_v0_1.csv", index=False, encoding="utf-8-sig")
    needs_review.to_csv(TABLE_DIR / "instability_needs_human_review_v0_1.csv", index=False, encoding="utf-8-sig")
    write_summary_table(df)
    plot_overview(df)
    example_paths = plot_examples(df, vehicle_cache)
    write_report(df, accepted, needs_review, example_paths)

    run_log = {
        "input_events": str(INPUT_EVENTS).replace("\\", "/"),
        "output_dir": str(OUT_DIR).replace("\\", "/"),
        "seed_count": int(len(seeds)),
        "episode_count": int(len(df)),
        "accepted_count": int(len(accepted)),
        "needs_human_review_count": int(len(needs_review)),
        "merge_gap_s": MERGE_GAP_S,
        "review_pre_s": REVIEW_PRE_S,
        "review_post_s": REVIEW_POST_S,
    }
    (LOG_DIR / "build_instability_event_review_v0_1.json").write_text(
        json.dumps(run_log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
