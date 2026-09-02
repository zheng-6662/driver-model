from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


RATE_HZ = 20
DT_S = 1.0 / RATE_HZ
QUERY_STRIDE_S = 0.2
QUERY_STRIDE_STEPS = int(round(QUERY_STRIDE_S * RATE_HZ))
HISTORY_STEPS = 40
FUTURE_STEPS = 20
MAX_RAW_SUPPORT_DISTANCE_S = 0.10
TIME_BLOCK_S = 10.0

SOURCE_MODULE_RELATIVE = Path("02_code/tools/build_multiaction_reframe_audit.py")
STIMULUS_EVENTS_RELATIVE = Path(
    "review_packages/MULTIACTION_REFRAME_20260901/tables_private/stimulus_events_private.csv"
)
RUN57_LEDGER_RELATIVE = Path(
    "05_rebuild_from_raw_20260511/03_baselines/"
    "run57_p0_event_population_ledger_20260827/run_1/tables/event_quality_ledger.csv"
)

INPUT_CHANNELS = (
    "steer_deg",
    "brake",
    "accelerator",
    "speed_kmh",
    "ax",
    "ay",
    "yaw_rate",
    "roll",
    "roll_rate",
    "curvature",
)
SOURCE_COLUMNS = (
    "_steer",
    "_brake",
    "_accelerator",
    "_speed_kmh",
    "_ax",
    "_ay",
    "_yaw_rate",
    "_roll",
    "_roll_rate",
    "_curvature",
)
TARGET_CHANNELS = ("steer_deg", "brake", "accelerator", "speed_kmh")
TARGET_INPUT_INDICES = np.array([0, 1, 2, 3], dtype=np.int64)
CORE_INPUT_INDICES = TARGET_INPUT_INDICES
ACTION_CHANGE_THRESHOLDS = np.array([5.0, 0.05, 0.05], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 Run84 20 Hz 连续多动作数据集")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--august-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_source_module(path: Path):
    spec = importlib.util.spec_from_file_location("run84_vehicle_reader", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"无法加载车辆读取模块: {path}")
    spec.loader.exec_module(module)
    return module


def interpolate_offline_label(time_s: np.ndarray, values: np.ndarray, grid_s: np.ndarray) -> np.ndarray:
    """未来监督标签可使用离线双侧插值，但仍禁止跨长缺失段补值。"""
    finite = np.isfinite(time_s) & np.isfinite(values)
    valid_time = time_s[finite]
    valid_values = values[finite]
    if len(valid_time) < 2:
        return np.full(len(grid_s), np.nan, dtype=np.float32)
    result = np.interp(grid_s, valid_time, valid_values).astype(np.float32)
    positions = np.searchsorted(valid_time, grid_s)
    left = np.clip(positions - 1, 0, len(valid_time) - 1)
    right = np.clip(positions, 0, len(valid_time) - 1)
    nearest = np.minimum(np.abs(grid_s - valid_time[left]), np.abs(grid_s - valid_time[right]))
    result[nearest > MAX_RAW_SUPPORT_DISTANCE_S + 1e-9] = np.nan
    return result


def causal_previous_value(time_s: np.ndarray, values: np.ndarray, grid_s: np.ndarray) -> np.ndarray:
    """历史输入只使用该网格时点及以前最近的原始样本。"""
    finite = np.isfinite(time_s) & np.isfinite(values)
    valid_time = time_s[finite]
    valid_values = values[finite]
    if len(valid_time) < 1:
        return np.full(len(grid_s), np.nan, dtype=np.float32)
    positions = np.searchsorted(valid_time, grid_s, side="right") - 1
    result = np.full(len(grid_s), np.nan, dtype=np.float32)
    supported = positions >= 0
    past_age = np.full(len(grid_s), np.inf, dtype=np.float64)
    past_age[supported] = grid_s[supported] - valid_time[positions[supported]]
    supported &= past_age <= MAX_RAW_SUPPORT_DISTANCE_S + 1e-9
    result[supported] = valid_values[positions[supported]].astype(np.float32)
    return result


def make_stat_features(histories: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """生成 ExtraTrees 所需的过去统计特征；每个特征都严格位于查询点及以前。"""
    feature_arrays: list[np.ndarray] = []
    feature_names: list[str] = []
    windows = (("hist2s", histories), ("recent0p5s", histories[:, -11:, :]))
    for window_name, window in windows:
        sample_times = np.arange(window.shape[1], dtype=np.float64) * DT_S
        for channel_index, channel_name in enumerate(INPUT_CHANNELS):
            values = window[:, :, channel_index].astype(np.float64, copy=False)
            finite = np.isfinite(values)
            count = finite.sum(axis=1)
            safe_count = np.maximum(count, 1)
            filled = np.where(finite, values, 0.0)
            mean = filled.sum(axis=1) / safe_count
            variance = (np.where(finite, (values - mean[:, None]) ** 2, 0.0).sum(axis=1) / safe_count)
            minimum = np.where(finite, values, np.inf).min(axis=1)
            maximum = np.where(finite, values, -np.inf).max(axis=1)
            minimum[count == 0] = np.nan
            maximum[count == 0] = np.nan
            mean[count == 0] = np.nan
            variance[count == 0] = np.nan

            time_sum = np.where(finite, sample_times[None, :], 0.0).sum(axis=1)
            time_mean = time_sum / safe_count
            centered_time = sample_times[None, :] - time_mean[:, None]
            denominator = np.where(finite, centered_time**2, 0.0).sum(axis=1)
            numerator = np.where(finite, centered_time * (values - mean[:, None]), 0.0).sum(axis=1)
            slope = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan),
                where=denominator > 0,
            )

            statistics = {
                "last": values[:, -1],
                "mean": mean,
                "std": np.sqrt(variance),
                "min": minimum,
                "max": maximum,
                "delta": values[:, -1] - values[:, 0],
                "slope": slope,
                "missing_fraction": 1.0 - count / values.shape[1],
            }
            for statistic_name, statistic_values in statistics.items():
                feature_arrays.append(statistic_values.astype(np.float32))
                feature_names.append(f"{channel_name}__{window_name}__{statistic_name}")
    return np.column_stack(feature_arrays).astype(np.float32), feature_names


def resample_recording(data: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """把 recording 切成真实连续段，并把每段统一到 20 Hz。"""
    time_s = data["_t_s"].to_numpy(float)
    if not np.all(np.diff(time_s) > 0):
        raise AssertionError("车辆时间轴并非严格递增")
    values = np.column_stack([data[column].to_numpy(float) for column in SOURCE_COLUMNS])
    values[:, 0] = np.degrees(values[:, 0])
    split_positions = np.flatnonzero(np.diff(time_s) > MAX_RAW_SUPPORT_DISTANCE_S + 1e-9) + 1
    boundaries = np.r_[0, split_positions, len(time_s)]
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment_time = time_s[start:end]
        if len(segment_time) < HISTORY_STEPS + FUTURE_STEPS:
            continue
        first_tick = int(math.ceil(segment_time[0] / DT_S - 1e-9))
        last_tick = int(math.floor(segment_time[-1] / DT_S + 1e-9))
        if last_tick - first_tick + 1 < HISTORY_STEPS + FUTURE_STEPS:
            continue
        grid = np.arange(first_tick, last_tick + 1, dtype=np.int64) * DT_S
        causal_values = np.column_stack(
            [causal_previous_value(segment_time, values[start:end, index], grid) for index in range(len(INPUT_CHANNELS))]
        ).astype(np.float32)
        offline_values = np.column_stack(
            [
                interpolate_offline_label(segment_time, values[start:end, index], grid)
                for index in range(len(INPUT_CHANNELS))
            ]
        ).astype(np.float32)
        segments.append((grid, causal_values, offline_values))
    return segments


def windows_from_segment(
    grid: np.ndarray,
    causal_values: np.ndarray,
    offline_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate = np.arange(HISTORY_STEPS - 1, len(grid) - FUTURE_STEPS, dtype=np.int64)
    global_ticks = np.rint(grid[candidate] / DT_S).astype(np.int64)
    candidate = candidate[global_ticks % QUERY_STRIDE_STEPS == 0]
    if not len(candidate):
        return (
            np.empty((0, HISTORY_STEPS, len(INPUT_CHANNELS)), dtype=np.float32),
            np.empty((0, FUTURE_STEPS, len(TARGET_CHANNELS)), dtype=np.float32),
            np.empty(0, dtype=np.float64),
        )

    history_offsets = np.arange(-HISTORY_STEPS + 1, 1, dtype=np.int64)
    future_offsets = np.arange(1, FUTURE_STEPS + 1, dtype=np.int64)
    histories = causal_values[candidate[:, None] + history_offsets[None, :]]
    future_absolute = offline_values[candidate[:, None] + future_offsets[None, :]][:, :, TARGET_INPUT_INDICES]
    current = causal_values[candidate][:, TARGET_INPUT_INDICES]
    targets = future_absolute - current[:, None, :]

    core_history_ok = np.isfinite(histories[:, :, CORE_INPUT_INDICES]).all(axis=(1, 2))
    core_future_ok = np.isfinite(future_absolute).all(axis=(1, 2))
    legal = core_history_ok & core_future_ok
    return histories[legal], targets[legal].astype(np.float32), grid[candidate][legal]


def classify_windows(histories: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    recent_action = histories[:, -9:, :3]
    action_delta = np.max(np.abs(recent_action - recent_action[:, :1, :]), axis=1)
    action_started = np.any(action_delta >= ACTION_CHANGE_THRESHOLDS[None, :], axis=1)

    recent_dynamic = histories[:, -11:, :]
    def finite_abs_max(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        maxima = np.where(finite, np.abs(values), -np.inf).max(axis=1)
        return maxima

    dynamic_components = np.column_stack(
        [
            finite_abs_max(recent_dynamic[:, :, 4]) >= 3.0,
            finite_abs_max(recent_dynamic[:, :, 5]) >= 3.0,
            finite_abs_max(recent_dynamic[:, :, 6]) >= 0.15,
            finite_abs_max(recent_dynamic[:, :, 8]) >= 0.15,
        ]
    )
    high_dynamic = np.any(dynamic_components, axis=1)
    action_change = np.max(np.abs(targets[:, :, :3]), axis=1) >= ACTION_CHANGE_THRESHOLDS[None, :]
    return {
        "high_dynamic": high_dynamic,
        "action_started": action_started,
        "high_dynamic_not_started": high_dynamic & ~action_started,
        "ordinary": ~high_dynamic & ~action_started,
        "steer_change": action_change[:, 0],
        "brake_change": action_change[:, 1],
        "accelerator_change": action_change[:, 2],
        "any_action_change": np.any(action_change, axis=1),
        "no_action_change": ~np.any(action_change, axis=1),
    }


def map_events_to_windows(
    events: pd.DataFrame,
    metadata: pd.DataFrame,
    subset_name: str,
    event_id_column: str,
    recording_column: str,
    anchor_column: str,
) -> pd.DataFrame:
    by_recording = {
        recording: group[["window_index", "query_time_s"]].sort_values("query_time_s")
        for recording, group in metadata.groupby("recording_alias", sort=False)
    }
    rows = []
    for event in events.itertuples(index=False):
        recording = str(getattr(event, recording_column))
        anchor_s = float(getattr(event, anchor_column))
        candidates = by_recording[recording]
        differences = np.abs(candidates["query_time_s"].to_numpy(float) - anchor_s)
        position = int(np.argmin(differences))
        mapping_error = float(differences[position])
        if mapping_error > QUERY_STRIDE_S / 2 + 1e-6:
            raise AssertionError(f"{subset_name} 锚点无法映射到0.2秒查询网格: {getattr(event, event_id_column)}")
        selected = candidates.iloc[position]
        rows.append(
            {
                "subset": subset_name,
                "event_id": str(getattr(event, event_id_column)),
                "recording_alias": recording,
                "anchor_time_s": anchor_s,
                "window_index": int(selected.window_index),
                "query_time_s": float(selected.query_time_s),
                "mapping_error_s": mapping_error,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    dataset_root = output_root / "dataset"
    tables_root = output_root / "tables"
    dataset_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    source_module = load_source_module(project_root / SOURCE_MODULE_RELATIVE)
    recordings, _, _ = source_module.load_cohort_sources(project_root, args.august_root.resolve())
    if len(recordings) != 221:
        raise AssertionError(f"连续车辆来源应为221条，实际为{len(recordings)}")

    history_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []
    inventory_rows = []
    feature_names: list[str] | None = None
    window_offset = 0

    for record in recordings.itertuples(index=False):
        data, _, source_rate = source_module.read_vehicle(Path(record.source_path))
        recording_histories = []
        recording_targets = []
        recording_times = []
        segments = resample_recording(data)
        for grid, causal_values, offline_values in segments:
            histories, targets, query_times = windows_from_segment(grid, causal_values, offline_values)
            if len(histories):
                recording_histories.append(histories)
                recording_targets.append(targets)
                recording_times.append(query_times)
        if recording_histories:
            histories = np.concatenate(recording_histories, axis=0)
            targets = np.concatenate(recording_targets, axis=0)
            query_times = np.concatenate(recording_times, axis=0)
            classifications = classify_windows(histories, targets)
            features, current_feature_names = make_stat_features(histories)
            if feature_names is None:
                feature_names = current_feature_names
            elif feature_names != current_feature_names:
                raise AssertionError("ExtraTrees特征列顺序发生变化")

            indices = np.arange(window_offset, window_offset + len(histories), dtype=np.int64)
            metadata = pd.DataFrame(
                {
                    "window_index": indices,
                    "window_id": [f"W{index:09d}" for index in indices],
                    "subject_alias": record.subject_alias,
                    "recording_alias": record.recording_alias,
                    "cohort": record.cohort,
                    "chronological_recording_index": int(record.chronological_recording_index),
                    "query_time_s": query_times,
                    "time_block_id": [
                        f"{record.recording_alias}-B{int(time // TIME_BLOCK_S):04d}" for time in query_times
                    ],
                    **classifications,
                }
            )
            history_parts.append(histories)
            target_parts.append(targets)
            feature_parts.append(features)
            metadata_parts.append(metadata)
            window_offset += len(histories)
            legal_windows = len(histories)
        else:
            legal_windows = 0
        inventory_rows.append(
            {
                "subject_alias": record.subject_alias,
                "recording_alias": record.recording_alias,
                "cohort": record.cohort,
                "chronological_recording_index": int(record.chronological_recording_index),
                "source_rate_hz": source_rate,
                "source_rows": len(data),
                "duration_s": float(data["_t_s"].iloc[-1] - data["_t_s"].iloc[0]),
                "continuous_segments": len(segments),
                "legal_windows": legal_windows,
                "source_path": str(record.source_path),
            }
        )
        print(f"DATA {record.recording_alias} {record.subject_alias} windows={legal_windows}", flush=True)

    histories = np.concatenate(history_parts, axis=0).astype(np.float32)
    targets = np.concatenate(target_parts, axis=0).astype(np.float32)
    features = np.concatenate(feature_parts, axis=0).astype(np.float32)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    inventory = pd.DataFrame(inventory_rows)
    if len(metadata) != len(histories) or not np.array_equal(metadata["window_index"], np.arange(len(metadata))):
        raise AssertionError("窗口数组与元数据索引不一致")
    if inventory["legal_windows"].sum() != len(metadata):
        raise AssertionError("recording人口汇总与窗口数组不一致")

    np.save(dataset_root / "history_20hz.npy", histories)
    np.save(dataset_root / "targets_relative_20hz.npy", targets)
    np.save(dataset_root / "extratrees_features.npy", features)
    metadata.to_csv(dataset_root / "window_metadata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature_name": feature_names}).to_csv(
        dataset_root / "extratrees_feature_names.csv", index=False, encoding="utf-8-sig"
    )
    inventory.to_csv(tables_root / "recording_inventory.csv", index=False, encoding="utf-8-sig")

    stimulus_events = pd.read_csv(project_root / STIMULUS_EVENTS_RELATIVE, low_memory=False)
    included = stimulus_events["included_candidate"].astype(str).str.lower().eq("true")
    distance_events = stimulus_events.loc[
        included & stimulus_events["stimulus_type"].str.startswith("distance_threshold_ExternTrigger")
    ].copy()
    low_mu_events = stimulus_events.loc[included & stimulus_events["stimulus_type"].eq("enter_low_mu_scene")].copy()
    if len(distance_events) != 305 or len(low_mu_events) != 70:
        raise AssertionError({"distance_v2": len(distance_events), "low_mu_v2": len(low_mu_events)})

    distance_mapping = map_events_to_windows(
        distance_events, metadata, "distance_v2_305", "event_id", "recording_alias", "stimulus_onset_s"
    )
    low_mu_mapping = map_events_to_windows(
        low_mu_events, metadata, "low_mu_v2_70", "event_id", "recording_alias", "stimulus_onset_s"
    )

    ledger = pd.read_csv(project_root / RUN57_LEDGER_RELATIVE, low_memory=False)
    def boolean_column(frame: pd.DataFrame, column: str) -> pd.Series:
        return frame[column].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

    history_ok = boolean_column(ledger, "history_complete")
    target_ok = boolean_column(ledger, "target_1s_complete") & boolean_column(ledger, "target_curve_complete")
    speed_ok = pd.to_numeric(ledger["pre_speed_p10_kmh"], errors="coerce") >= 60.0
    not_reverse = ~boolean_column(ledger, "pre_reverse")
    input_ok = pd.to_numeric(ledger["causal_input_coverage_min"], errors="coerce") >= 0.90
    direction_ok = pd.to_numeric(ledger["causal_direction_consistency_at_release"], errors="coerce") >= 0.70
    p_full = history_ok & target_ok & speed_ok & not_reverse & input_ok & direction_ok
    release_events = ledger.loc[
        p_full,
        ["event_uid", "subject", "session_stamp", "primary_release_s"],
    ].drop_duplicates("event_uid")
    if len(release_events) != 2323:
        raise AssertionError(f"Run57 V3 release历史对照应为2323，实际为{len(release_events)}")
    original_lookup = recordings.loc[recordings["cohort"].eq("original")].set_index(
        ["subject_key", "session_stamp"]
    )["recording_alias"]
    release_events["recording_alias"] = [
        original_lookup.loc[(subject, stamp)]
        for subject, stamp in zip(release_events["subject"], release_events["session_stamp"])
    ]
    release_mapping = map_events_to_windows(
        release_events,
        metadata,
        "release_v3_historical_2323",
        "event_uid",
        "recording_alias",
        "primary_release_s",
    )
    evaluation_mapping = pd.concat([distance_mapping, low_mu_mapping, release_mapping], ignore_index=True)
    evaluation_mapping.to_csv(tables_root / "fixed_evaluation_mapping.csv", index=False, encoding="utf-8-sig")

    subject_population = (
        metadata.groupby(["subject_alias", "cohort"], as_index=False)
        .agg(
            windows=("window_id", "size"),
            recordings=("recording_alias", "nunique"),
            high_dynamic=("high_dynamic", "sum"),
            action_started=("action_started", "sum"),
            ordinary=("ordinary", "sum"),
        )
    )
    subject_population.to_csv(tables_root / "population_by_subject_and_cohort.csv", index=False, encoding="utf-8-sig")

    action_rows = []
    for action in ["steer", "brake", "accelerator", "any_action"]:
        column = f"{action}_change"
        changed = int(metadata[column].sum())
        action_rows.append(
            {
                "action": action,
                "changed_windows": changed,
                "unchanged_windows": len(metadata) - changed,
                "changed_fraction": changed / len(metadata),
                "unchanged_fraction": 1.0 - changed / len(metadata),
            }
        )
    pd.DataFrame(action_rows).to_csv(tables_root / "action_change_summary.csv", index=False, encoding="utf-8-sig")

    dynamic_rows = []
    for population in ["high_dynamic", "high_dynamic_not_started", "action_started", "ordinary"]:
        count = int(metadata[population].sum())
        dynamic_rows.append({"population": population, "windows": count, "fraction": count / len(metadata)})
    pd.DataFrame(dynamic_rows).to_csv(tables_root / "dynamic_window_summary.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "run": "run84_continuous_multiaction_forecast_20260902",
        "status": "DATASET_COMPLETE",
        "recordings_total": int(len(inventory)),
        "recordings_with_legal_windows": int((inventory["legal_windows"] > 0).sum()),
        "subjects": int(metadata["subject_alias"].nunique()),
        "windows": int(len(metadata)),
        "history_shape": list(histories.shape),
        "target_shape": list(targets.shape),
        "extratrees_feature_shape": list(features.shape),
        "fixed_evaluation_counts": evaluation_mapping.groupby("subset").size().to_dict(),
        "source_inventory": "tables/recording_inventory.csv",
        "window_metadata": "dataset/window_metadata.csv",
        "history_array": "dataset/history_20hz.npy",
        "target_array": "dataset/targets_relative_20hz.npy",
        "extratrees_features": "dataset/extratrees_features.npy",
        "fixed_evaluation_mapping": "tables/fixed_evaluation_mapping.csv",
        "source_reader": str(SOURCE_MODULE_RELATIVE).replace("\\", "/"),
        "builder": "build_dataset.py",
        "causal_input_rule": "previous raw observation at or before each 20 Hz input time",
        "future_label_rule": "offline bilateral linear interpolation at future label times",
        "causal_input_future_support_used": False,
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
