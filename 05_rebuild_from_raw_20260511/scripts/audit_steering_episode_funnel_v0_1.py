# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
REBUILD_ROOT = SCRIPT_PATH.parents[1]
if str(REBUILD_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REBUILD_ROOT / "src"))

from event_episode_detection.config import load_config, resolve_paths
from event_episode_detection.context_matching import ContextMatcher
from event_episode_detection.signals import load_vehicle_csv, max_abs, robust_median, window
from event_episode_detection.steering_onset import compute_steer_thresholds, infer_subject_session


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def find_vehicle_files(project_root: Path, raw_glob: str) -> list[Path]:
    files = sorted(project_root.glob(raw_glob))
    files = [p for p in files if p.name.startswith("Entity_Recording_") and p.name.endswith("_vehicle.csv")]
    return files


def segment_starts(candidate_idx: np.ndarray) -> list[int]:
    if candidate_idx.size == 0:
        return []
    split_points = np.where(np.diff(candidate_idx) > 1)[0] + 1
    return [int(seg[0]) for seg in np.split(candidate_idx, split_points) if seg.size]


def audit_record(vehicle_path: Path, config: dict[str, Any], matcher: ContextMatcher) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    vehicle, meta = load_vehicle_csv(vehicle_path, config)
    subject, session, rel_hint = infer_subject_session(vehicle_path)
    record_row: dict[str, Any] = {
        "subject_id": subject,
        "session_stamp": session,
        "vehicle_raw_absolute_path": str(vehicle_path),
        "vehicle_raw_relative_path": rel_hint,
        "read_status": meta.get("read_status", ""),
        "row_count": meta.get("row_count", 0),
        "duration_s": meta.get("duration_s", np.nan),
        "high_rate_point_count": 0,
        "high_rate_segment_count": 0,
        "after_gap_merge_count": 0,
        "reject_window_incomplete": 0,
        "reject_pre_or_early_empty": 0,
        "reject_pre_not_stable": 0,
        "reject_early_delta_too_small": 0,
        "accepted_strict_episode_count": 0,
        "loose_candidate_count": 0,
    }
    rows: list[dict[str, Any]] = []
    if vehicle.empty or "steer_rate" not in vehicle:
        return record_row, rows

    thresholds = compute_steer_thresholds(vehicle, config)
    rate_thr = thresholds["steer_rate_threshold"]
    delta_thr = thresholds["steer_delta_threshold"]
    if not math.isfinite(rate_thr):
        return record_row, rows

    time = vehicle["time_rel_s"].to_numpy(dtype=float)
    rate = vehicle["steer_rate"].to_numpy(dtype=float)
    candidate_idx = np.where(np.abs(rate) >= rate_thr)[0]
    starts = segment_starts(candidate_idx)
    record_row["high_rate_point_count"] = int(candidate_idx.size)
    record_row["high_rate_segment_count"] = int(len(starts))
    pre_window_sec = float(config.get("pre_window_sec", 2.0))
    early_sec = float(config.get("early_observation_sec", 0.5))
    correction_sec = float(config.get("correction_window_sec", 5.0))
    stable_sec = float(config.get("pre_stable_window_sec", 0.8))
    stable_rate_thr = rate_thr * float(config.get("pre_stable_rate_ratio", 0.70))
    min_gap = float(config.get("min_episode_gap_sec", 1.5))
    last_checked_time = -1e9
    kept_idx = 0

    for idx in starts:
        t0 = float(time[idx])
        if t0 - last_checked_time < min_gap:
            record_row["after_gap_merge_count"] += 1
            continue
        last_checked_time = t0
        if t0 < pre_window_sec or t0 + correction_sec > float(time[-1]):
            record_row["reject_window_incomplete"] += 1
            continue
        pre = vehicle[(vehicle["time_rel_s"] >= t0 - stable_sec) & (vehicle["time_rel_s"] < t0)].copy()
        early = window(vehicle, t0, t0 + early_sec)
        if pre.empty or early.empty:
            record_row["reject_pre_or_early_empty"] += 1
            continue
        pre_max_rate = max_abs(pre["steer_rate"])
        pre_stable_pass = bool(math.isfinite(pre_max_rate) and pre_max_rate <= stable_rate_thr)
        baseline = robust_median(pre["steer_smooth"].to_numpy(dtype=float))
        early_delta_abs = max_abs(early["steer_smooth"] - baseline)
        early_delta_pass = bool(math.isfinite(early_delta_abs) and early_delta_abs >= delta_thr)
        loose_pass = bool(early_delta_pass)
        strict_pass = bool(pre_stable_pass and early_delta_pass)
        if not pre_stable_pass:
            record_row["reject_pre_not_stable"] += 1
        if pre_stable_pass and not early_delta_pass:
            record_row["reject_early_delta_too_small"] += 1
        if loose_pass:
            record_row["loose_candidate_count"] += 1
        if strict_pass:
            record_row["accepted_strict_episode_count"] += 1
        road_context = matcher.append_context(
            {
                "subject_id": subject,
                "session_stamp": session,
                "t_steer_onset": t0,
            }
        ).get("road_context", "")
        rows.append(
            {
                "candidate_id": f"steer_funnel_v0_1__{subject}__{session}__{kept_idx:05d}",
                "subject_id": subject,
                "session_stamp": session,
                "vehicle_raw_absolute_path": str(vehicle_path),
                "vehicle_raw_relative_path": rel_hint,
                "t_steer_onset_candidate": t0,
                "road_context": road_context,
                "abs_rate_at_candidate": abs(float(rate[idx])),
                "steer_rate_threshold": rate_thr,
                "steer_delta_threshold": delta_thr,
                "pre_max_abs_rate": pre_max_rate,
                "pre_stable_rate_threshold": stable_rate_thr,
                "pre_stable_pass": pre_stable_pass,
                "early_delta_abs": early_delta_abs,
                "early_delta_pass": early_delta_pass,
                "loose_candidate_pass": loose_pass,
                "strict_episode_pass": strict_pass,
                "reject_reason_cn": reject_reason(pre_stable_pass, early_delta_pass, strict_pass),
            }
        )
        kept_idx += 1
    return record_row, rows


def reject_reason(pre_stable: bool, early_delta: bool, strict: bool) -> str:
    if strict:
        return "通过严格 episode 规则"
    if not pre_stable and early_delta:
        return "启动前不够平稳，但启动后幅值足够；可能是连续转向中段"
    if pre_stable and not early_delta:
        return "启动前平稳，但启动后0.5秒幅值不足；可能是噪声或轻微修正"
    return "启动前不平稳且短时幅值不足"


def write_report(out_dir: Path, record_df: pd.DataFrame, cand_df: pd.DataFrame, strict_df: pd.DataFrame) -> None:
    total_records = len(record_df)
    total_segments = int(record_df["high_rate_segment_count"].sum())
    total_points = int(record_df["high_rate_point_count"].sum())
    merged = int(record_df["after_gap_merge_count"].sum())
    window_rej = int(record_df["reject_window_incomplete"].sum())
    empty_rej = int(record_df["reject_pre_or_early_empty"].sum())
    pre_rej = int(record_df["reject_pre_not_stable"].sum())
    delta_rej = int(record_df["reject_early_delta_too_small"].sum())
    loose = int(record_df["loose_candidate_count"].sum())
    strict = int(record_df["accepted_strict_episode_count"].sum())
    road_loose = cand_df[cand_df["loose_candidate_pass"]].groupby("road_context").size().sort_values(ascending=False)
    road_strict = cand_df[cand_df["strict_episode_pass"]].groupby("road_context").size().sort_values(ascending=False)
    report = out_dir / "steering_episode_funnel_audit_v0_1.md"
    report.write_text(
        f"""# 方向盘动作 episode 候选漏斗审计 v0.1

生成时间：{now()}

## 这次回答什么问题

上一版方向盘主锚点样本重建只得到 159 个严格 episode。这个审计专门拆开筛选过程，回答：到底是方向盘动作真的少，还是规则过严。

## 漏斗总览

- 处理车辆记录：{total_records}
- 高方向盘角速度原始点数：{total_points}
- 连续高角速度片段数：{total_segments}
- 因 1.5 秒内过近而合并/跳过的片段：{merged}
- 因前后窗口不完整被删：{window_rej}
- 因前段或早期窗口为空被删：{empty_rej}
- 因启动前不够平稳被删：{pre_rej}
- 因启动后 0.5 秒方向盘幅值不足被删：{delta_rej}
- 宽松候选池数量：{loose}
- 严格 episode 数量：{strict}

## 如何理解

宽松候选池只要求“方向盘角速度显著升高后，0.5 秒内方向盘角确实有明显变化”，不强制要求启动前完全平稳。严格 episode 则额外要求启动前 0.8 秒内相对平稳。

如果宽松候选明显多于严格 episode，说明主要被“启动前平稳”规则筛掉，这通常发生在连续超车、弯道、施工路段持续调整中。这类样本不一定无效，但它们不是“从静止状态突然猛打方向”的事件，更像连续转向 episode 的子段。

## 宽松候选分场景

```text
{road_loose.to_string() if not road_loose.empty else '无'}
```

## 严格 episode 分场景

```text
{road_strict.to_string() if not road_strict.empty else '无'}
```

## 结论建议

1. 如果研究目标是“猛打方向的启动事件”，应继续使用严格 episode。
2. 如果研究目标是“正常变道、弯道、连续超车中的方向盘轨迹预测”，应使用宽松候选池，并另外标注它是否处于连续转向中段。
3. 下一步人工复核时，不只看严格 P1，也要抽看宽松候选里“启动前不平稳但幅值足够”的样本，判断它们是否是你想要的事件。

## 输出文件

- 逐记录漏斗表：`{out_dir / 'steering_funnel_by_record_v0_1.csv'}`
- 候选明细表：`{out_dir / 'steering_funnel_candidates_v0_1.csv'}`
- 宽松候选池：`{out_dir / 'loose_steering_candidates_v0_1.csv'}`
- 严格通过表：`{out_dir / 'strict_steering_episode_candidates_v0_1.csv'}`
""",
        encoding="utf-8",
    )


def update_notes(rebuild_root: Path, out_dir: Path, record_df: pd.DataFrame) -> None:
    notes = rebuild_root / "00_project_notes"
    daily = notes / "daily_logs" / "2026-05-14.md"
    if not daily.exists():
        daily.write_text("# 2026-05-14 执行日志\n\n", encoding="utf-8")
    with daily.open("a", encoding="utf-8") as f:
        f.write(
            f"""## 方向盘动作候选漏斗审计 v0.1

- 为什么做：解释为什么严格方向盘 episode 只有 159 个。
- 做了什么：统计高方向盘角速度点、连续片段、间隔合并、窗口过滤、启动前平稳过滤、启动后幅值过滤，并输出宽松候选池。
- 输出目录：`{out_dir}`
- 严格 episode 数：{int(record_df['accepted_strict_episode_count'].sum())}
- 宽松候选数：{int(record_df['loose_candidate_count'].sum())}

"""
        )
    artifact = notes / "ARTIFACT_INDEX_CN.md"
    old = artifact.read_text(encoding="utf-8", errors="ignore") if artifact.exists() else "# 产物索引\n\n"
    entry = f"""## 2026-05-14 方向盘动作候选漏斗审计 v0.1

- 报告：`{out_dir / 'steering_episode_funnel_audit_v0_1.md'}`
- 逐记录漏斗表：`{out_dir / 'steering_funnel_by_record_v0_1.csv'}`
- 候选明细表：`{out_dir / 'steering_funnel_candidates_v0_1.csv'}`
- 宽松候选池：`{out_dir / 'loose_steering_candidates_v0_1.csv'}`
- 严格通过表：`{out_dir / 'strict_steering_episode_candidates_v0_1.csv'}`

"""
    artifact.write_text(entry + old, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = resolve_paths(config)
    out_dir = paths.output_dir / "funnel_audit_v0_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    matcher = ContextMatcher(paths)
    files = find_vehicle_files(paths.project_root, paths.raw_vehicle_glob)
    record_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for i, path in enumerate(files):
        rec, rows = audit_record(path, config, matcher)
        record_rows.append(rec)
        candidate_rows.extend(rows)
        if (i + 1) % 10 == 0:
            print(f"audited {i+1}/{len(files)}")
    record_df = pd.DataFrame(record_rows)
    cand_df = pd.DataFrame(candidate_rows)
    loose_df = cand_df[cand_df["loose_candidate_pass"]].copy() if not cand_df.empty else pd.DataFrame()
    strict_df = cand_df[cand_df["strict_episode_pass"]].copy() if not cand_df.empty else pd.DataFrame()
    write_csv(record_df, out_dir / "steering_funnel_by_record_v0_1.csv")
    write_csv(cand_df, out_dir / "steering_funnel_candidates_v0_1.csv")
    write_csv(loose_df, out_dir / "loose_steering_candidates_v0_1.csv")
    write_csv(strict_df, out_dir / "strict_steering_episode_candidates_v0_1.csv")
    if not cand_df.empty:
        by_scene = pd.crosstab(cand_df["road_context"], cand_df["reject_reason_cn"])
        write_csv(by_scene.reset_index(), out_dir / "steering_funnel_by_scene_reason_v0_1.csv")
    write_report(out_dir, record_df, cand_df, strict_df)
    update_notes(paths.rebuild_root, out_dir, record_df)
    print("done")
    print("records", len(record_df))
    print("segments", int(record_df["high_rate_segment_count"].sum()))
    print("loose", int(record_df["loose_candidate_count"].sum()))
    print("strict", int(record_df["accepted_strict_episode_count"].sum()))
    print(out_dir / "steering_episode_funnel_audit_v0_1.md")


if __name__ == "__main__":
    main()

