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
OUT_DIR = ROOT / "02_samples" / "codex_event_review_v0_1"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"

VEHICLE_COLS = [
    "StorageTime",
    "zx1|lanecurvatureXY",
    "zx|lanecurvatureXY",
    "zx|SteeringWheel",
    "zx1|v_km/h",
    "zx1|lateraldistance",
    "zx|vyaw",
    "zx|ay",
    "zx|roll",
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
        raise ValueError("no valid StorageTime")
    df["time_rel_s"] = t_abs - float(np.nanmin(t_abs))
    df = df.sort_values("time_rel_s").reset_index(drop=True)
    return df


def make_road_sublabels(ev: pd.Series, duration_s: float) -> list[dict[str, Any]]:
    anchor = finite_float(ev.get("anchor_time_rel_s"))
    start = finite_float(ev.get("event_start_rel_s"))
    end = finite_float(ev.get("event_end_rel_s"))
    if anchor is None:
        return []
    if start is None:
        start = anchor
    if end is None or end <= start:
        end = anchor + 2.0
    start = clip(start, 0.0, duration_s)
    end = clip(end, 0.0, duration_s)
    if end <= start:
        end = clip(start + 2.0, 0.0, duration_s)
    event_type = str(ev.get("event_type", "curve"))
    direction = (
        "left"
        if "left" in event_type or "positive" in event_type
        else "right"
        if "right" in event_type or "negative" in event_type
        else "unclear"
    )
    raw_duration = end - start
    if raw_duration > 30.0:
        specs = [
            ("entry", "curve_entry", start, min(end, start + 12.0), start),
            ("exit", "curve_exit_or_return", max(start, end - 12.0), end, max(start, end - 12.0)),
        ]
    elif raw_duration > 0.8:
        specs = [("full", "curve_short", start, end, start)]
    else:
        specs = [("brief", "curve_brief", start, end, start)]
    rows: list[dict[str, Any]] = []
    for suffix, role, seg_start, seg_end, seg_anchor in specs:
        mid = (seg_start + seg_end) / 2.0
        review_start = clip(mid - 16.0, 0.0, duration_s)
        review_end = clip(mid + 16.0, 0.0, duration_s)
        if role == "curve_entry":
            review_start = clip(seg_start - 8.0, 0.0, duration_s)
            review_end = clip(seg_start + 22.0, 0.0, duration_s)
        elif role == "curve_exit_or_return":
            review_start = clip(seg_end - 22.0, 0.0, duration_s)
            review_end = clip(seg_end + 8.0, 0.0, duration_s)
        rows.append(
            {
                "review_label_suffix": suffix,
                "auto_event_role": role,
                "event_start_rel_s": round(float(seg_start), 3),
                "event_end_rel_s": round(float(seg_end), 3),
                "anchor_rel_s": round(float(seg_anchor), 3),
                "review_start_rel_s": round(float(review_start), 3),
                "review_end_rel_s": round(float(review_end), 3),
                "original_road_start_rel_s": round(float(start), 3),
                "original_road_end_rel_s": round(float(end), 3),
                "original_road_duration_s": round(float(raw_duration), 3),
                "direction_hint": direction,
            }
        )
    return rows


def window_values(df: pd.DataFrame, col: str | None, start: float, end: float) -> np.ndarray:
    if col is None:
        return np.array([], dtype=np.float64)
    mask = (df["time_rel_s"] >= start) & (df["time_rel_s"] <= end)
    vals = pd.to_numeric(df.loc[mask, col], errors="coerce").to_numpy(dtype=np.float64)
    return vals[np.isfinite(vals)]


def signed_peak_delta(values: np.ndarray, baseline: float) -> tuple[float, float]:
    if values.size == 0 or not math.isfinite(baseline):
        return np.nan, np.nan
    delta = values - baseline
    idx = int(np.nanargmax(np.abs(delta)))
    return float(delta[idx]), float(abs(delta[idx]))


def count_candidates(events: pd.DataFrame, source: str, start: float, end: float) -> int:
    sub = events[
        (events["anchor_source"] == source)
        & (events["anchor_time_rel_s"] >= start)
        & (events["anchor_time_rel_s"] <= end)
    ]
    return int(len(sub))


def compute_row_metrics(df: pd.DataFrame, events: pd.DataFrame, label: dict[str, Any]) -> dict[str, Any]:
    t0 = float(label["anchor_rel_s"])
    start = float(label["event_start_rel_s"])
    end = float(label["event_end_rel_s"])
    review_start = float(label["review_start_rel_s"])
    review_end = float(label["review_end_rel_s"])
    steer_col = choose_col(df, ["zx|SteeringWheel"])
    curv_col = choose_col(df, ["zx1|lanecurvatureXY", "zx|lanecurvatureXY"])
    speed_col = choose_col(df, ["zx1|v_km/h"])
    yaw_col = choose_col(df, ["zx|vyaw"])
    ay_col = choose_col(df, ["zx|ay"])
    roll_col = choose_col(df, ["zx|roll"])

    pre_steer = window_values(df, steer_col, t0 - 2.0, t0)
    response_steer = window_values(df, steer_col, t0, review_end)
    curve_vals = window_values(df, curv_col, start, end)
    speed_vals = window_values(df, speed_col, review_start, review_end)
    yaw_vals = window_values(df, yaw_col, review_start, review_end)
    ay_vals = window_values(df, ay_col, review_start, review_end)
    roll_vals = window_values(df, roll_col, review_start, review_end)
    steer_baseline = float(np.nanmedian(pre_steer)) if pre_steer.size else np.nan
    steer_delta_signed, steer_delta_abs = signed_peak_delta(response_steer, steer_baseline)
    curve_peak_abs = float(np.nanmax(np.abs(curve_vals))) if curve_vals.size else np.nan
    curve_mean = float(np.nanmean(curve_vals)) if curve_vals.size else np.nan
    return {
        "pre_points": int(pre_steer.size),
        "response_points": int(response_steer.size),
        "curve_points": int(curve_vals.size),
        "steer_baseline": steer_baseline,
        "steer_delta_peak_signed": steer_delta_signed,
        "steer_delta_peak_abs": steer_delta_abs,
        "curve_peak_abs": curve_peak_abs,
        "curve_mean": curve_mean,
        "speed_median": float(np.nanmedian(speed_vals)) if speed_vals.size else np.nan,
        "yaw_peak_abs": float(np.nanmax(np.abs(yaw_vals))) if yaw_vals.size else np.nan,
        "ay_peak_abs": float(np.nanmax(np.abs(ay_vals))) if ay_vals.size else np.nan,
        "roll_peak_abs": float(np.nanmax(np.abs(roll_vals))) if roll_vals.size else np.nan,
        "nearby_old_v400_count": count_candidates(events, "old_v400_context_trigger_idx", review_start, review_end),
        "nearby_dynamic_count": count_candidates(events, "raw_vehicle_dynamic_onset", review_start, review_end),
        "nearby_road_count": count_candidates(events, "raw_road_curvature_onset", review_start, review_end),
    }


def scaled(value: Any, q_low: float, q_high: float) -> float:
    x = finite_float(value)
    if x is None or not math.isfinite(q_low) or not math.isfinite(q_high) or abs(q_high - q_low) < 1e-12:
        return 0.0
    return float(np.clip((x - q_low) / (q_high - q_low), 0.0, 1.0))


def assign_scores(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    curve_q25, curve_q75 = labels["curve_peak_abs"].quantile([0.25, 0.75]).tolist()
    steer_q25, steer_q75 = labels["steer_delta_peak_abs"].quantile([0.25, 0.75]).tolist()
    ay_q25, ay_q75 = labels["ay_peak_abs"].quantile([0.25, 0.75]).tolist()
    scores: list[float] = []
    decisions: list[str] = []
    reasons: list[str] = []
    for _, row in labels.iterrows():
        score = 35.0
        score += 18.0 * scaled(row["curve_peak_abs"], curve_q25, curve_q75)
        score += 18.0 * scaled(row["steer_delta_peak_abs"], steer_q25, steer_q75)
        score += 6.0 * scaled(row["ay_peak_abs"], ay_q25, ay_q75)
        score += min(8.0, float(row["nearby_dynamic_count"]) * 2.0)
        score += min(5.0, float(row["nearby_old_v400_count"]) * 1.0)
        score += 5.0 if row["pre_points"] >= 100 and row["response_points"] >= 300 else -8.0
        speed = finite_float(row["speed_median"])
        if speed is not None and speed > 5.0:
            score += 5.0
        elif speed is not None and speed < 1.0:
            score -= 15.0
        if row["steer_delta_peak_abs"] < steer_q25 and row["nearby_dynamic_count"] == 0:
            score -= 10.0
        if row["curve_points"] < 100:
            score -= 8.0
        score = float(np.clip(score, 0.0, 100.0))
        if score >= 72.0:
            decision = "auto_accept_high"
        elif score >= 58.0:
            decision = "auto_accept_medium"
        elif score >= 43.0:
            decision = "needs_human_review"
        else:
            decision = "reject_low_evidence"
        reason = (
            f"road={row['auto_event_role']}; score={score:.1f}; "
            f"curve_peak={row['curve_peak_abs']:.4g}; steer_delta_peak={row['steer_delta_peak_abs']:.4g}; "
            f"old={int(row['nearby_old_v400_count'])}; dynamic={int(row['nearby_dynamic_count'])}; "
            f"speed={row['speed_median']:.3g}"
        )
        scores.append(score)
        decisions.append(decision)
        reasons.append(reason)
    labels["codex_review_score"] = scores
    labels["codex_recommended_decision"] = decisions
    labels["codex_evidence_reason"] = reasons
    labels["codex_review_status"] = "codex_auto_reviewed_not_human_verified"
    return labels


def make_overview_figure(labels: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    axes[0].hist(labels["codex_review_score"], bins=20, color="#2f6fbb", edgecolor="white")
    axes[0].set_title("Codex review score distribution")
    axes[0].set_xlabel("score")
    axes[0].set_ylabel("count")
    decision_counts = labels["codex_recommended_decision"].value_counts().sort_index()
    axes[1].barh(decision_counts.index, decision_counts.values, color="#4b8f63")
    axes[1].set_title("Recommended decision counts")
    axes[1].set_xlabel("count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "codex_event_review_score_overview_v0_1.png")
    plt.close(fig)


def make_example_figures(labels: pd.DataFrame, events: pd.DataFrame, max_examples: int = 12) -> None:
    candidates = []
    for decision in ["auto_accept_high", "auto_accept_medium", "needs_human_review", "reject_low_evidence"]:
        sub = labels[labels["codex_recommended_decision"] == decision].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("codex_review_score", ascending=(decision != "reject_low_evidence")).head(3)
        candidates.append(sub)
    if not candidates:
        return
    examples = pd.concat(candidates, ignore_index=True).head(max_examples)
    for _, row in examples.iterrows():
        try:
            df = load_vehicle(str(row["vehicle_raw_absolute_path"]))
        except Exception:
            continue
        curv_col = choose_col(df, ["zx1|lanecurvatureXY", "zx|lanecurvatureXY"])
        steer_col = choose_col(df, ["zx|SteeringWheel"])
        speed_col = choose_col(df, ["zx1|v_km/h"])
        review_start = float(row["review_start_rel_s"])
        review_end = float(row["review_end_rel_s"])
        mask = (df["time_rel_s"] >= review_start) & (df["time_rel_s"] <= review_end)
        t = df.loc[mask, "time_rel_s"].to_numpy(dtype=np.float64)
        fig, axes = plt.subplots(3, 1, figsize=(10, 5), dpi=150, sharex=True)
        for ax, col, title in [
            (axes[0], curv_col, "road curvature"),
            (axes[1], steer_col, "steering"),
            (axes[2], speed_col, "speed km/h"),
        ]:
            if col is not None:
                ax.plot(t, pd.to_numeric(df.loc[mask, col], errors="coerce").to_numpy(dtype=np.float64), lw=1.2)
            ax.axvspan(float(row["event_start_rel_s"]), float(row["event_end_rel_s"]), color="#79aee8", alpha=0.18)
            ax.axvline(float(row["anchor_rel_s"]), color="#1d5fd1", lw=2)
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.25)
        sess_events = events[
            (events["subject"].astype(str) == str(row["subject"]))
            & (events["session_stamp"].astype(str) == str(row["session_stamp"]))
            & (events["anchor_time_rel_s"] >= review_start)
            & (events["anchor_time_rel_s"] <= review_end)
        ]
        for ev in sess_events.itertuples(index=False):
            source = getattr(ev, "anchor_source")
            x = getattr(ev, "anchor_time_rel_s")
            color = "#d08020" if source == "old_v400_context_trigger_idx" else "#c3342b" if source == "raw_vehicle_dynamic_onset" else "#1d5fd1"
            for ax in axes:
                ax.axvline(float(x), color=color, lw=0.8, alpha=0.6)
        axes[-1].set_xlabel("relative time (s)")
        fig.suptitle(f"{row['review_label_uid']} | {row['codex_recommended_decision']} | score={row['codex_review_score']:.1f}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{row['review_label_uid']}.png")
        plt.close(fig)


def write_summary_report(labels: pd.DataFrame, summary: pd.DataFrame, review_queue: pd.DataFrame) -> None:
    decision_text = labels["codex_recommended_decision"].value_counts().to_string()
    role_text = labels["auto_event_role"].value_counts().to_string()
    report = f"""# 阶段 2 补充：Codex 自动事件审阅 v0.1

生成时间：2026-05-12

## 为什么做

用户认为逐个播放和人工标注事件仍然太耗时，因此本阶段先由 Codex 对低泄漏道路曲率候选进行规则化自动审阅。输出不是最终真值，而是带证据、分数和置信度的候选标签，用来减少人工复核范围。

## 输入

- 候选事件表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/candidate_events_master.csv`
- 原始车辆 CSV：只读取 `原始车辆数据/<被试名>/*.csv`
- 主锚点来源：`raw_road_curvature_onset`
- 辅助证据：`old_v400_context_trigger_idx` 和 `raw_vehicle_dynamic_onset` 只作附近支持计数，不作为无泄漏真值。

## 输出

- 自动审阅标签：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_reviewed_event_labels_v0_1.csv`
- 自动采用标签：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_auto_accepted_event_labels_v0_1.csv`
- 需要人工复核队列：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_needs_human_review_v0_1.csv`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_event_review_summary_v0_1.csv`
- 分数图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/figures/codex_event_review_score_overview_v0_1.png`

## 审阅规则

1. 只把道路曲率候选作为主事件来源。
2. 长道路曲率段拆成 `curve_entry` 和 `curve_exit_or_return`，避免把一整段弯道当成一个事件。
3. 短道路曲率段保留为 `curve_short` 或 `curve_brief`。
4. 每个候选计算道路曲率强度、方向盘响应幅值、横向加速度、旧流程邻近点、车辆动态邻近点、车速和采样点数量。
5. 得到 0-100 分的 `codex_review_score`，并分为 `auto_accept_high`、`auto_accept_medium`、`needs_human_review`、`reject_low_evidence`。

## 当前数量

总自动审阅标签数：{len(labels)}

按决策统计：

```text
{decision_text}
```

按事件角色统计：

```text
{role_text}
```

需要人工复核或剔除的数量：{len(review_queue)}

## 重要边界

这不是人工真值，也不能直接证明事件锚点最终正确。它的用途是先由 Codex 做第一轮筛选：高/中置信标签可以进入下一步候选 `codex_auto_accepted` 数据版本，低置信和冲突样本再由用户少量复核。
"""
    (REPORT_DIR / "codex_event_review_v0_1_cn.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    events = pd.read_csv(INPUT_EVENTS)
    for col in ["anchor_time_rel_s", "event_start_rel_s", "event_end_rel_s"]:
        events[col] = pd.to_numeric(events[col], errors="coerce")
    road = events[events["anchor_source"] == "raw_road_curvature_onset"].copy()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for (subject, session_stamp), group in road.groupby(["subject", "session_stamp"], dropna=False):
        raw_path = str(group.iloc[0]["vehicle_raw_absolute_path"])
        try:
            df = load_vehicle(raw_path)
        except Exception as exc:
            failures.append({"subject": str(subject), "session_stamp": str(session_stamp), "error": f"{type(exc).__name__}: {exc}"})
            continue
        duration_s = float(df["time_rel_s"].max())
        sess_events = events[(events["subject"].astype(str) == str(subject)) & (events["session_stamp"].astype(str) == str(session_stamp))].copy()
        for _, ev in group.sort_values("anchor_time_rel_s").iterrows():
            sublabels = make_road_sublabels(ev, duration_s)
            for sub_idx, label in enumerate(sublabels, start=1):
                base = {
                    "review_label_uid": f"codex_{subject}_{session_stamp}_{int(ev.get('event_index_in_source', sub_idx)):04d}_{label['review_label_suffix']}",
                    "subject": str(subject),
                    "session_stamp": str(session_stamp),
                    "vehicle_raw_absolute_path": raw_path,
                    "source_event_uid": str(ev.get("event_uid", "")),
                    "source_anchor": "raw_road_curvature_onset",
                    "source_event_type": str(ev.get("event_type", "")),
                    "source_event_level": str(ev.get("event_level", "")),
                    "source_curvature_anchor": finite_float(ev.get("curvature_anchor")),
                    "leakage_risk_anchor": str(ev.get("leakage_risk_anchor", "")),
                    "causal_anchor_status": "codex_auto_reviewed_candidate",
                }
                base.update(label)
                base.update(compute_row_metrics(df, sess_events, label))
                rows.append(base)
    labels = pd.DataFrame(rows)
    if labels.empty:
        raise RuntimeError("no labels generated")
    labels = assign_scores(labels)
    labels = labels.sort_values(["subject", "session_stamp", "anchor_rel_s", "auto_event_role"]).reset_index(drop=True)
    accepted = labels[labels["codex_recommended_decision"].isin(["auto_accept_high", "auto_accept_medium"])].copy()
    review_queue = labels[labels["codex_recommended_decision"].isin(["needs_human_review", "reject_low_evidence"])].copy()
    summary = (
        labels.groupby(["codex_recommended_decision", "auto_event_role"], dropna=False)
        .agg(
            n=("review_label_uid", "count"),
            score_mean=("codex_review_score", "mean"),
            steer_delta_peak_abs_median=("steer_delta_peak_abs", "median"),
            curve_peak_abs_median=("curve_peak_abs", "median"),
            nearby_dynamic_median=("nearby_dynamic_count", "median"),
        )
        .reset_index()
    )
    labels.to_csv(TABLE_DIR / "codex_reviewed_event_labels_v0_1.csv", index=False, encoding="utf-8-sig")
    accepted.to_csv(TABLE_DIR / "codex_auto_accepted_event_labels_v0_1.csv", index=False, encoding="utf-8-sig")
    review_queue.to_csv(TABLE_DIR / "codex_needs_human_review_v0_1.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "codex_event_review_summary_v0_1.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(TABLE_DIR / "codex_event_review_failures_v0_1.csv", index=False, encoding="utf-8-sig")
    make_overview_figure(labels, summary)
    make_example_figures(labels, events)
    write_summary_report(labels, summary, review_queue)
    run_summary = {
        "total_labels": int(len(labels)),
        "accepted_labels": int(len(accepted)),
        "needs_review_or_reject": int(len(review_queue)),
        "failures": int(len(failures)),
        "server_used": False,
        "raw_files_modified": False,
    }
    (LOG_DIR / "codex_event_review_v0_1_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
