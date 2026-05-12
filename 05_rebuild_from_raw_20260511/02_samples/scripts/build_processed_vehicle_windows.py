# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_stage2_samples import (
    AUDIT_TABLE_DIR,
    FS,
    REBUILD_ROOT,
    fill_nan_linear,
    read_vehicle_cache,
    value_at,
)


SAMPLES_TABLE = REBUILD_ROOT / "02_samples" / "tables" / "samples_master.csv"
OUT_ROOT = REBUILD_ROOT / "03_processed_datasets" / "vehicle_road_curvature_v0_2"
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

WINDOWS_TO_PROCESS = [
    "pre1_label2_event_trigger",
    "pre2_label2_old_main",
    "pre3_label3_response_coverage",
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, ARRAY_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


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


def load_vehicle_rows() -> pd.DataFrame:
    inventory = pd.read_csv(AUDIT_TABLE_DIR / "raw_file_inventory.csv")
    timestamp = pd.read_csv(AUDIT_TABLE_DIR / "timestamp_continuity_report.csv")
    return inventory[inventory["modality"] == "vehicle"].merge(
        timestamp[["relative_path", "time_min", "time_max", "zero_dt_count", "large_gap_count"]],
        on="relative_path",
        how="left",
    )


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


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
        index_rows.append(
            {
                "array_row": i,
                "sample_id": row.sample_id,
                "event_uid": row.event_uid,
                "subject": row.subject,
                "session_stamp": row.session_stamp,
                "anchor_time_rel_s": row.anchor_time_rel_s,
                "anchor_time_abs_s": row.anchor_time_abs_s,
                "window_config_id": row.window_config_id,
                "vehicle_raw_relative_path": row.vehicle_relative_path,
                "vehicle_raw_sha256": row.vehicle_sha256,
                "input_valid_ratio": float(input_mask[i].mean()),
                "label_valid_ratio": float(label_mask[i].mean()),
                "anchor_steer": float(anchor_steer[i]) if np.isfinite(anchor_steer[i]) else np.nan,
            }
        )

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
    index_path = TABLE_DIR / f"sample_index_{window_id}.csv"
    index_df.to_csv(index_path, index=False, encoding="utf-8-sig")
    return {
        "window_config_id": window_id,
        "npz_path": str(out_npz).replace("\\", "/"),
        "index_path": str(index_path).replace("\\", "/"),
        "sample_count": int(n),
        "input_shape": list(input_values.shape),
        "label_shape": list(label_delta.shape),
        "mean_input_valid_ratio": float(input_mask.mean()),
        "mean_label_valid_ratio": float(label_mask.mean()),
    }


def write_processing_report(summary_rows: list[dict[str, Any]]) -> None:
    table = pd.DataFrame(summary_rows)
    report = f"""# 处理后车辆窗口数据 v0.2 说明

生成时间：2026-05-12

## 处理目标

把阶段 2 中低泄漏的 `raw_road_curvature_onset` 候选样本，处理成阶段 3 可用的车辆输入窗口和方向盘未来标签窗口。本版本只处理车辆数据，不处理生理或脑电，不训练模型。

## 输入

- 样本清单：`02_samples/tables/samples_master.csv`
- 原始车辆 CSV：`01_datasets/数据预处理/原始车辆数据/<被试名>/*.csv`
- 选择规则：`recommended_for_stage3_vehicle_baseline=True`
- 窗口：`pre1_label2_event_trigger`、`pre2_label2_old_main`、`pre3_label3_response_coverage`

## 处理规则

1. 原始 CSV 只读，不覆盖。
2. `StorageTime` 强制转为 `datetime64[ns]` 后换算为秒，避免微秒/纳秒单位错误。
3. 同一时间戳的同一信号先按均值折叠，再插值到 200 Hz 车辆时间网格。
4. 输入特征保持原始物理量，不做标准化、不做 train/test 统计拟合、不做基线校正。
5. 标签为 `zx|SteeringWheel` 相对锚点时刻方向盘值的未来增量。
6. 每个数组同时保存 valid mask，后续模型或基线必须显式使用 mask。

## 输出概要

{table.to_string(index=False)}

## 无泄漏边界

本处理版本只包含 `raw_road_curvature_onset` 且 `input_end_rel_s<=0` 的样本，适合作为阶段 3 低泄漏车辆基线的起点。早期观察窗口、旧 v400 参考锚点和 raw dynamic 响应锚点没有被处理进本版本，避免把响应结果混入事件触发预测主线。
"""
    (REPORT_DIR / "processed_vehicle_windows_v0_2_cn.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    samples = pd.read_csv(SAMPLES_TABLE)
    samples["recommended_for_stage3_vehicle_baseline"] = bool_series(samples["recommended_for_stage3_vehicle_baseline"])
    selected = samples[
        samples["recommended_for_stage3_vehicle_baseline"]
        & samples["window_config_id"].isin(WINDOWS_TO_PROCESS)
        & (samples["anchor_source"] == "raw_road_curvature_onset")
    ].copy()
    selected = selected.sort_values(["window_config_id", "subject", "session_stamp", "anchor_time_rel_s"]).reset_index(drop=True)
    selected.to_csv(TABLE_DIR / "selected_samples_vehicle_road_v0_2.csv", index=False, encoding="utf-8-sig")

    vehicle_rows = load_vehicle_rows()
    needed_keys = set(zip(selected["subject"].astype(str), selected["session_stamp"].astype(str)))
    caches = {}
    cache_rows = []
    for row in vehicle_rows.itertuples(index=False):
        key = (str(row.subject), str(row.session_stamp))
        if key not in needed_keys:
            continue
        cache = read_vehicle_cache(pd.Series(row._asdict()))
        caches[key] = cache
        cache_rows.append(
            {
                "subject": cache.subject,
                "session_stamp": cache.session_stamp,
                "read_status": cache.read_status,
                "read_error": cache.read_error,
                "duration_s": float(cache.t_grid_rel_s[-1]) if cache.t_grid_rel_s.size else np.nan,
                "grid_rows": int(cache.t_grid_rel_s.size),
            }
        )
    pd.DataFrame(cache_rows).to_csv(TABLE_DIR / "vehicle_cache_status_v0_2.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for window_id in WINDOWS_TO_PROCESS:
        window_df = selected[selected["window_config_id"] == window_id].reset_index(drop=True)
        if window_df.empty:
            continue
        summary_rows.append(process_window(window_df, caches, window_id))

    summary = {
        "selected_sample_rows": int(len(selected)),
        "window_outputs": summary_rows,
        "feature_names": FEATURES,
        "fs": FS,
        "raw_files_modified": False,
    }
    (LOG_DIR / "processed_vehicle_windows_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(summary_rows).to_csv(TABLE_DIR / "processed_vehicle_window_outputs.csv", index=False, encoding="utf-8-sig")
    write_processing_report(summary_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
