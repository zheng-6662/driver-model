# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_stage2_samples import FS, fill_nan_linear, read_vehicle_cache, value_at  # noqa: E402


PROJECT_ROOT = Path(r"F:/data_set_process/data_process")
REBUILD_ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
INPUT_CSV = (
    REBUILD_ROOT
    / "02_samples"
    / "vehicle_instability_all_raw_rescreen_v0_1"
    / "tables"
    / "all_raw_vehicle_instability_primary_high_confidence_v0_1.csv"
)
OUT_ROOT = REBUILD_ROOT / "03_processed_datasets" / "vehicle_instability_allraw_highconf_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
ARRAY_DIR = OUT_ROOT / "arrays"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = REBUILD_ROOT / "09_reports"

FEATURES = [
    "zx|SteeringWheel",
    "zx1|v_km/h",
    "zx1|lanecurvatureXY",
    "zx1|lateraldistance",
    "zx|roll",
    "zx|vyaw",
    "zx|ay",
    "zx|vx",
    "zx|vy",
]

WINDOWS = [
    {
        "window_config_id": "pre1_label2_event_trigger",
        "input_start_rel_s": -1.0,
        "input_end_rel_s": 0.0,
        "label_start_rel_s": 0.0,
        "label_end_rel_s": 2.0,
        "note": "Old stage-3 compatible: 1 s pre-event vehicle input, 2 s steering-delta label.",
    },
    {
        "window_config_id": "pre2_label2_old_main",
        "input_start_rel_s": -2.0,
        "input_end_rel_s": 0.0,
        "label_start_rel_s": 0.0,
        "label_end_rel_s": 2.0,
        "note": "Old main comparison window: 2 s pre-event vehicle input, 2 s steering-delta label.",
    },
    {
        "window_config_id": "pre3_label3_response_coverage",
        "input_start_rel_s": -3.0,
        "input_end_rel_s": 0.0,
        "label_start_rel_s": 0.0,
        "label_end_rel_s": 3.0,
        "note": "Response-coverage diagnostic: 3 s pre-event vehicle input, 3 s steering-delta label.",
    },
]

SPLIT_STRATEGIES = ["random_event_split", "session_level_split", "subject_level_split"]
OLD_MANIFEST_SPLIT = "session_level_split"
OLD_HISTORY_LEN = 600
OLD_FUTURE_LEN = 400


def ensure_dirs() -> None:
    for path in [TABLE_DIR, ARRAY_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def assign_by_group(values: pd.Series, group_values: pd.Series) -> pd.Series:
    groups = sorted(pd.Series(group_values.astype(str).unique()).dropna().tolist(), key=stable_int)
    n = len(groups)
    train_end = max(1, int(round(n * 0.70)))
    val_end = max(train_end + 1, int(round(n * 0.85))) if n >= 3 else train_end
    val_end = min(val_end, n)
    mapping: dict[str, str] = {}
    for i, group in enumerate(groups):
        if i < train_end:
            mapping[group] = "train"
        elif i < val_end:
            mapping[group] = "val"
        else:
            mapping[group] = "test"
    if n == 1:
        mapping[groups[0]] = "train"
    elif n == 2:
        mapping[groups[0]] = "train"
        mapping[groups[1]] = "test"
    return values.astype(str).map(mapping).fillna("train")


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def bool_str(value: bool) -> str:
    return "True" if value else "False"


def normalize_event_level(row: pd.Series) -> str:
    old_level = str(row.get("old_v400_max_level", "")).strip()
    if old_level and old_level.lower() not in {"nan", "none", "unknown"}:
        return old_level
    score = to_float(row.get("instability_review_score"))
    if score >= 85:
        return "instability_extreme"
    if score >= 70:
        return "instability_strong"
    return "instability_medium"


def road_type(row: pd.Series) -> str:
    old_road = str(row.get("old_v400_road_type_mode", "")).strip()
    if old_road and old_road.lower() not in {"nan", "none", "unknown"}:
        return old_road
    risk = str(row.get("road_design_risk_class", "")).strip()
    return risk if risk and risk.lower() != "nan" else "unknown"


def event_type(row: pd.Series) -> str:
    role = str(row.get("instability_role", "")).strip()
    if role and role.lower() != "nan":
        return role
    sources = str(row.get("source_event_types", "")).strip()
    return sources if sources and sources.lower() != "nan" else "vehicle_instability"


def interp_signal(t_grid: np.ndarray, arr: np.ndarray | None, target_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if arr is None or t_grid.size < 2:
        return np.full(len(target_t), np.nan, dtype=np.float32), np.zeros(len(target_t), dtype=bool)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return np.full(len(target_t), np.nan, dtype=np.float32), np.zeros(len(target_t), dtype=bool)
    values = np.interp(target_t, t_grid[finite], arr[finite], left=np.nan, right=np.nan)
    valid = np.isfinite(values)
    return values.astype(np.float32), valid


def make_time_axis(start_rel: float, end_rel: float) -> np.ndarray:
    n = int(round((end_rel - start_rel) * FS)) + 1
    return np.linspace(start_rel, end_rel, n, dtype=np.float32)


def load_candidates() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df = df.copy()
    df["event_uid"] = df["instability_event_uid"].astype(str)
    df["event_type"] = df.apply(event_type, axis=1)
    df["event_level"] = df.apply(normalize_event_level, axis=1)
    df["road_type_anchor"] = df.apply(road_type, axis=1)
    df["curvature_anchor"] = pd.to_numeric(df.get("road_design_curvature", np.nan), errors="coerce").fillna(0.0)
    df["is_curve"] = df["road_design_risk_class"].astype(str).str.contains("curve", case=False, na=False).astype(int)
    df["anchor_idx_oldcode"] = np.rint(pd.to_numeric(df["anchor_time_rel_s"], errors="coerce") * FS).astype("Int64")
    df["future_rows_available_oldcode"] = np.nan
    df["history_full_3s_oldcode"] = False
    df["future_full_2s_oldcode"] = False
    df["oldcode_usable"] = False
    df["oldcode_drop_reason"] = ""

    df["random_event_split"] = assign_by_group(df["event_uid"], df["event_uid"])
    df["session_key"] = df["subject"].astype(str) + "__" + df["session_stamp"].astype(str)
    df["session_level_split"] = assign_by_group(df["session_key"], df["session_key"])
    df["subject_level_split"] = assign_by_group(df["subject"].astype(str), df["subject"].astype(str))
    return df


def build_cache_rows(candidates: pd.DataFrame) -> tuple[dict[tuple[str, str], Any], pd.DataFrame]:
    caches: dict[tuple[str, str], Any] = {}
    status_rows: list[dict[str, Any]] = []
    session_rows = (
        candidates[
            [
                "subject",
                "session_stamp",
                "vehicle_raw_relative_path",
                "vehicle_raw_absolute_path",
                "vehicle_raw_sha256",
            ]
        ]
        .drop_duplicates()
        .sort_values(["subject", "session_stamp"])
    )
    for row in session_rows.itertuples(index=False):
        series = pd.Series(
            {
                "subject": str(row.subject),
                "session_stamp": str(row.session_stamp),
                "relative_path": str(row.vehicle_raw_relative_path),
                "absolute_path": str(row.vehicle_raw_absolute_path),
                "sha256": str(row.vehicle_raw_sha256),
            }
        )
        cache = read_vehicle_cache(series)
        key = (str(row.subject), str(row.session_stamp))
        caches[key] = cache
        status_rows.append(
            {
                "subject": cache.subject,
                "session_stamp": cache.session_stamp,
                "read_status": cache.read_status,
                "read_error": cache.read_error,
                "duration_s": float(cache.t_grid_rel_s[-1]) if cache.t_grid_rel_s.size else np.nan,
                "grid_rows": int(cache.t_grid_rel_s.size),
                "raw_relative_path": cache.raw_relative_path,
                "raw_sha256": cache.sha256,
            }
        )
    return caches, pd.DataFrame(status_rows)


def mark_oldcode_usable(candidates: pd.DataFrame, caches: dict[tuple[str, str], Any]) -> pd.DataFrame:
    out = candidates.copy()
    for idx, row in out.iterrows():
        key = (str(row["subject"]), str(row["session_stamp"]))
        cache = caches.get(key)
        if cache is None or cache.read_status != "ok":
            out.at[idx, "oldcode_drop_reason"] = "vehicle_cache_read_failed"
            continue
        anchor_idx = int(row["anchor_idx_oldcode"]) if pd.notna(row["anchor_idx_oldcode"]) else -1
        if anchor_idx < 0:
            out.at[idx, "oldcode_drop_reason"] = "invalid_anchor_idx"
            continue
        future_rows = max(int(cache.t_grid_rel_s.size) - (anchor_idx + 1), 0)
        out.at[idx, "future_rows_available_oldcode"] = future_rows
        history_ok = anchor_idx - OLD_HISTORY_LEN + 1 >= 0
        future_ok = future_rows >= OLD_FUTURE_LEN
        out.at[idx, "history_full_3s_oldcode"] = bool(history_ok)
        out.at[idx, "future_full_2s_oldcode"] = bool(future_ok)
        if not history_ok:
            out.at[idx, "oldcode_drop_reason"] = "history_underflow_for_3s_oldcode"
            continue
        if not future_ok:
            out.at[idx, "oldcode_drop_reason"] = "future_underflow_for_2s_oldcode"
            continue
        out.at[idx, "oldcode_usable"] = True
    return out


def build_selected_windows(candidates: pd.DataFrame) -> pd.DataFrame:
    usable = candidates[candidates["oldcode_usable"]].copy()
    rows: list[dict[str, Any]] = []
    for _, row in usable.iterrows():
        for cfg in WINDOWS:
            sample_id = f"{row['event_uid']}__{cfg['window_config_id']}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "event_uid": row["event_uid"],
                    "subject": row["subject"],
                    "session_stamp": row["session_stamp"],
                    "anchor_time_rel_s": float(row["anchor_time_rel_s"]),
                    "anchor_time_abs_s": np.nan,
                    "window_config_id": cfg["window_config_id"],
                    "input_start_rel_s": cfg["input_start_rel_s"],
                    "input_end_rel_s": cfg["input_end_rel_s"],
                    "label_start_rel_s": cfg["label_start_rel_s"],
                    "label_end_rel_s": cfg["label_end_rel_s"],
                    "anchor_source": row["instability_anchor_source"],
                    "causal_setting": row["causal_setting"],
                    "event_type": row["event_type"],
                    "event_level": row["event_level"],
                    "road_type_anchor": row["road_type_anchor"],
                    "is_curve": int(row["is_curve"]),
                    "curvature_anchor": float(row["curvature_anchor"]),
                    "instability_review_score": float(row["instability_review_score"]),
                    "road_guided_instability_score": float(row["road_guided_instability_score"]),
                    "codex_recommended_decision": row["codex_recommended_decision"],
                    "road_guided_recommended_decision": row["road_guided_recommended_decision"],
                    "vehicle_relative_path": row["vehicle_raw_relative_path"],
                    "vehicle_absolute_path": row["vehicle_raw_absolute_path"],
                    "vehicle_sha256": row["vehicle_raw_sha256"],
                    "anchor_idx_oldcode": int(row["anchor_idx_oldcode"]),
                    "event_start_rel_s": float(row["event_start_rel_s"]),
                    "event_end_rel_s": float(row["event_end_rel_s"]),
                    "event_duration_s": float(row["event_duration_s"]),
                    "split": row[OLD_MANIFEST_SPLIT],
                    **{split_col: row[split_col] for split_col in SPLIT_STRATEGIES},
                }
            )
    return pd.DataFrame(rows).sort_values(["window_config_id", "subject", "session_stamp", "anchor_time_rel_s"])


def process_window(window_df: pd.DataFrame, caches: dict[tuple[str, str], Any], window_id: str) -> dict[str, Any]:
    first = window_df.iloc[0]
    input_time = make_time_axis(float(first["input_start_rel_s"]), float(first["input_end_rel_s"]))
    label_time = make_time_axis(float(first["label_start_rel_s"]), float(first["label_end_rel_s"]))
    n = len(window_df)
    input_values = np.full((n, len(input_time), len(FEATURES)), np.nan, dtype=np.float32)
    input_mask = np.zeros((n, len(input_time), len(FEATURES)), dtype=bool)
    label_delta = np.full((n, len(label_time)), np.nan, dtype=np.float32)
    label_mask = np.zeros((n, len(label_time)), dtype=bool)
    anchor_steer = np.full(n, np.nan, dtype=np.float32)

    index_rows: list[dict[str, Any]] = []
    for i, row in enumerate(window_df.itertuples(index=False)):
        key = (str(row.subject), str(row.session_stamp))
        cache = caches.get(key)
        if cache is None or cache.read_status != "ok":
            continue
        anchor_rel = float(row.anchor_time_rel_s)
        t_grid = cache.t_grid_rel_s
        steer = cache.signals.get("zx|SteeringWheel")
        base = value_at(t_grid, steer, anchor_rel) if steer is not None else np.nan
        anchor_steer[i] = np.float32(base) if np.isfinite(base) else np.nan
        target_input = anchor_rel + input_time.astype(np.float64)
        target_label = anchor_rel + label_time.astype(np.float64)
        for j, feat in enumerate(FEATURES):
            vals, valid = interp_signal(t_grid, cache.signals.get(feat), target_input)
            input_values[i, :, j] = vals
            input_mask[i, :, j] = valid
        label_vals, label_valid = interp_signal(t_grid, steer, target_label)
        if np.isfinite(base):
            label_delta[i, :] = (label_vals - np.float32(base)).astype(np.float32)
            label_mask[i, :] = label_valid
        meta = row._asdict()
        meta.update(
            {
                "array_row": i,
                "vehicle_raw_relative_path": row.vehicle_relative_path,
                "vehicle_raw_sha256": row.vehicle_sha256,
                "input_valid_ratio": float(input_mask[i].mean()),
                "label_valid_ratio": float(label_mask[i].mean()),
                "anchor_steer": float(anchor_steer[i]) if np.isfinite(anchor_steer[i]) else np.nan,
            }
        )
        index_rows.append(meta)

    out_npz = ARRAY_DIR / f"{window_id}.npz"
    np.savez_compressed(
        out_npz,
        input_values=input_values,
        input_valid_mask=input_mask,
        label_steer_delta=label_delta,
        label_valid_mask=label_mask,
        input_time_rel_s=input_time,
        label_time_rel_s=label_time,
        feature_names=np.array(FEATURES, dtype=object),
        sample_ids=window_df["sample_id"].astype(str).to_numpy(dtype=object),
        anchor_steer=anchor_steer,
        fs=np.array([FS], dtype=np.float32),
    )
    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(TABLE_DIR / f"sample_index_{window_id}.csv", index=False, encoding="utf-8-sig")

    gt_peak = np.nanmax(np.abs(np.where(label_mask, label_delta, np.nan)), axis=1)
    return {
        "window_config_id": window_id,
        "npz_path": str(out_npz).replace("\\", "/"),
        "index_path": str((TABLE_DIR / f"sample_index_{window_id}.csv")).replace("\\", "/"),
        "sample_count": int(n),
        "input_shape": list(input_values.shape),
        "label_shape": list(label_delta.shape),
        "mean_input_valid_ratio": float(input_mask.mean()),
        "mean_label_valid_ratio": float(label_mask.mean()),
        "label_peak_abs_mean": float(np.nanmean(gt_peak)),
        "label_peak_abs_p75": float(np.nanpercentile(gt_peak, 75)),
        "label_peak_abs_p90": float(np.nanpercentile(gt_peak, 90)),
    }


def make_old_manifest(candidates: pd.DataFrame, split_col: str) -> pd.DataFrame:
    usable = candidates[candidates["oldcode_usable"]].copy()
    rows: list[dict[str, Any]] = []
    for event_idx, row in enumerate(usable.itertuples(index=False), start=1):
        anchor_s = float(row.anchor_time_rel_s)
        anchor_idx = int(row.anchor_idx_oldcode)
        sample_key = f"{row.subject}::{row.session_stamp}::{event_idx:05d}::instability_dynamic_onset"
        rows.append(
            {
                "protocol_version": "instability_allraw_highconf_oldcode_v0_1",
                "sample_key": sample_key,
                "pool": "allraw_vehicle_instability_highconf",
                "subj": row.subject,
                "split": getattr(row, split_col),
                "file": Path(str(row.vehicle_raw_absolute_path)).name,
                "recording_id": Path(str(row.vehicle_raw_absolute_path)).name,
                "vehicle_file": row.vehicle_raw_absolute_path,
                "event_idx": event_idx,
                "episode_id": event_idx,
                "source_event_version": "allraw_instability_rescreen_v0_1",
                "phase_type": getattr(row, "old_v400_phase_mode", "unknown"),
                "event_level": row.event_level,
                "trigger_type": row.instability_anchor_source,
                "event_type": row.event_type,
                "road_type_anchor": row.road_type_anchor,
                "is_curve": int(row.is_curve),
                "curvature_anchor": float(row.curvature_anchor),
                "anchor_source": "all_raw_nonsteering_dynamic_onset",
                "anchor_idx": anchor_idx,
                "anchor_s": anchor_s,
                "history_len": OLD_HISTORY_LEN,
                "future_len": OLD_FUTURE_LEN,
                "valid_future_len": OLD_FUTURE_LEN,
                "valid_future_s": 2.0,
                "full_future_2s": "True",
                "history_full_3s": "True",
                "time_left_after_anchor_s": float(row.future_rows_available_oldcode) / FS,
                "keep_for_training": "True",
                "usable_sample": "True",
                "drop_reason": "",
                "event_start_s": float(row.event_start_rel_s),
                "event_end_s": float(row.event_end_rel_s),
                "event_duration_s": float(row.event_duration_s),
                "start_idx": int(round(float(row.event_start_rel_s) * FS)),
                "end_idx": int(round(float(row.event_end_rel_s) * FS)),
                "trigger_score": float(row.road_guided_instability_score),
                "primary_score": float(row.instability_review_score),
                "mechanism_tag": row.instability_role,
                "d3_included": "True",
                "instability_event_uid": row.event_uid,
                "leakage_note": row.leakage_note,
            }
        )
    return pd.DataFrame(rows)


def write_reports(summary: dict[str, Any], selected: pd.DataFrame, manifest: pd.DataFrame) -> None:
    split_table = manifest["split"].value_counts().rename_axis("split").reset_index(name="n_events")
    window_rows = pd.DataFrame(summary["window_outputs"])
    report = f"""# 旧代码兼容数据包：全原始车辆失稳高置信样本 v0.1

生成时间：2026-05-12

## 目的

这一步不是训练新模型，而是把前一步重新筛出的高置信车辆失稳事件转成旧代码可以读取的格式，用来快速测试旧车辆代码在这些样本上的表现。

## 输入

- 高置信失稳事件：`{INPUT_CSV.as_posix()}`
- 原始车辆 CSV：`F:/data_set_process/data_process/01_datasets/数据预处理/原始车辆数据/<被试名>/*.csv`
- 锚点规则：非转向车辆动力学 onset，主要来自 `ay` 和 `roll_rate`，转向响应只作为后验证据，不用于定义锚点。

## 输出

- 旧阶段 3 `.npz` 窗口：`{ARRAY_DIR.as_posix()}`
- 样本索引与 split：`{TABLE_DIR.as_posix()}`
- 旧深度模型 manifest：`{(TABLE_DIR / 'oldcode_manifest_session_level_split.csv').as_posix()}`

## 数量

- 输入高置信事件数：{summary['input_high_confidence_events']}
- 旧代码可用事件数：{summary['oldcode_usable_events']}
- 旧代码不可用事件数：{summary['oldcode_dropped_events']}
- 窗口样本行数：{len(selected)}

## 旧代码 split 分布

{split_table.to_string(index=False)}

## 窗口输出

{window_rows.to_string(index=False)}

## 边界说明

1. 原始 CSV 未修改。
2. 窗口和 manifest 只使用车辆数据，不包含生理和脑电。
3. 标准化没有在这里做；后续旧代码测试时只能在训练集内拟合统计量。
4. 这一步仍然是诊断，不能由此宣称连续风格或生理有效。
5. 旧代码 manifest 使用 `session_level_split` 作为默认 split，另外也输出 random/session/subject 三种 split 字段供对照。
"""
    (REPORT_DIR / "instability_oldcode_dataset_v0_1_cn.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    candidates = load_candidates()
    caches, cache_status = build_cache_rows(candidates)
    candidates = mark_oldcode_usable(candidates, caches)
    selected = build_selected_windows(candidates)

    candidates.to_csv(TABLE_DIR / "instability_highconf_events_oldcode_eligibility_v0_1.csv", index=False, encoding="utf-8-sig")
    cache_status.to_csv(TABLE_DIR / "vehicle_cache_status_v0_1.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(TABLE_DIR / "selected_samples_vehicle_instability_highconf_v0_1.csv", index=False, encoding="utf-8-sig")

    split_base = candidates[candidates["oldcode_usable"]][["event_uid", *SPLIT_STRATEGIES]].copy()
    split_base.to_csv(TABLE_DIR / "split_table_vehicle_instability_highconf_v0_1.csv", index=False, encoding="utf-8-sig")

    manifest_by_split = {}
    for split_col in SPLIT_STRATEGIES:
        manifest = make_old_manifest(candidates, split_col)
        manifest_path = TABLE_DIR / f"oldcode_manifest_{split_col}.csv"
        manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        manifest_by_split[split_col] = {
            "path": str(manifest_path).replace("\\", "/"),
            "rows": int(len(manifest)),
            "split_counts": manifest["split"].value_counts().to_dict() if len(manifest) else {},
        }
        if split_col == OLD_MANIFEST_SPLIT:
            primary_manifest = manifest

    summary_rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        window_id = window["window_config_id"]
        window_df = selected[selected["window_config_id"] == window_id].reset_index(drop=True)
        if not window_df.empty:
            summary_rows.append(process_window(window_df, caches, window_id))

    summary = {
        "input_high_confidence_events": int(len(candidates)),
        "oldcode_usable_events": int(candidates["oldcode_usable"].sum()),
        "oldcode_dropped_events": int((~candidates["oldcode_usable"]).sum()),
        "drop_reasons": candidates.loc[~candidates["oldcode_usable"], "oldcode_drop_reason"].value_counts().to_dict(),
        "selected_window_rows": int(len(selected)),
        "window_outputs": summary_rows,
        "features": FEATURES,
        "fs": FS,
        "old_manifest_split": OLD_MANIFEST_SPLIT,
        "old_manifests": manifest_by_split,
        "raw_files_modified": False,
    }
    (LOG_DIR / "instability_oldcode_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(summary_rows).to_csv(TABLE_DIR / "processed_vehicle_window_outputs.csv", index=False, encoding="utf-8-sig")
    write_reports(summary, selected, primary_manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
