# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"F:/data_set_process/data_process")
REBUILD_ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"

DATASET_VERSION = "vehicle_instability_highconf_v0_1"
SOURCE_EVENT_VERSION = "vehicle_instability_all_raw_rescreen_v0_1"
PROCESSED_DATASET_VERSION = "vehicle_instability_allraw_highconf_v0_1"

PRIMARY_EVENTS = (
    REBUILD_ROOT
    / "02_samples"
    / "vehicle_instability_all_raw_rescreen_v0_1"
    / "tables"
    / "all_raw_vehicle_instability_primary_high_confidence_v0_1.csv"
)
MODALITY_MATRIX = REBUILD_ROOT / "01_audit" / "tables" / "subject_session_modality_matrix.csv"
PROCESSED_ROOT = REBUILD_ROOT / "03_processed_datasets" / PROCESSED_DATASET_VERSION
PROCESSED_TABLE_DIR = PROCESSED_ROOT / "tables"
PROCESSED_ARRAY_DIR = PROCESSED_ROOT / "arrays"
ELIGIBILITY_TABLE = PROCESSED_TABLE_DIR / "instability_highconf_events_oldcode_eligibility_v0_1.csv"

OUT_ROOT = REBUILD_ROOT / "02_samples" / DATASET_VERSION
TABLE_DIR = OUT_ROOT / "tables"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = REBUILD_ROOT / "09_reports"

WINDOW_CONFIGS = [
    {
        "window_config_id": "pre1_label2_event_trigger",
        "input_start_rel_s": -1.0,
        "input_end_rel_s": 0.0,
        "label_start_rel_s": 0.0,
        "label_end_rel_s": 2.0,
        "role": "early_vehicle_history_control",
        "is_primary_window": False,
        "description_cn": "事件前 1 秒车辆历史预测事件后 2 秒方向盘响应。",
    },
    {
        "window_config_id": "pre2_label2_old_main",
        "input_start_rel_s": -2.0,
        "input_end_rel_s": 0.0,
        "label_start_rel_s": 0.0,
        "label_end_rel_s": 2.0,
        "role": "main_old_comparison_and_initial_new_baseline",
        "is_primary_window": True,
        "description_cn": "事件前 2 秒车辆历史预测事件后 2 秒方向盘响应，作为旧代码对照和新流程初始主窗口。",
    },
    {
        "window_config_id": "pre3_label3_response_coverage",
        "input_start_rel_s": -3.0,
        "input_end_rel_s": 0.0,
        "label_start_rel_s": 0.0,
        "label_end_rel_s": 3.0,
        "role": "response_coverage_diagnostic",
        "is_primary_window": False,
        "description_cn": "事件前 3 秒车辆历史预测事件后 3 秒方向盘响应，用于检查尾段和完整响应覆盖。",
    },
]

SPLIT_STRATEGIES = ["random_event_split", "session_level_split", "subject_level_split"]
DEFAULT_SPLIT_STRATEGY = "session_level_split"
FS = 200.0


def ensure_dirs() -> None:
    for path in [TABLE_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return default
    return text


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_str(value).lower()
    return text in {"true", "1", "yes", "y"}


def first_crossing(arr: np.ndarray, thr: float) -> int:
    idx = np.where(np.abs(arr) >= thr)[0]
    return int(idx[0]) if idx.size else -1


def zero_crossing_has(arr: np.ndarray) -> bool:
    valid = arr[np.isfinite(arr)]
    if valid.size < 2:
        return False
    return bool(np.nanmin(valid) < 0.0 and np.nanmax(valid) > 0.0)


def reversal_count(arr: np.ndarray) -> int:
    valid = arr[np.isfinite(arr)]
    if valid.size < 4:
        return 0
    deriv = np.diff(valid)
    if deriv.size == 0:
        return 0
    cutoff = max(0.002, float(np.nanpercentile(np.abs(deriv), 70)) * 0.3)
    sign = np.sign(deriv)
    sign[np.abs(deriv) < cutoff] = 0
    nonzero = sign[sign != 0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0))


def peak_stats(arr: np.ndarray, time_axis: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return {
            "eval_label_peak_abs": float("nan"),
            "eval_label_peak_signed": float("nan"),
            "eval_label_peak_idx": -1,
            "eval_label_peak_time_rel_s": float("nan"),
            "eval_label_peak_direction": "unknown",
        }
    vals = arr.copy()
    vals[~valid] = 0.0
    idx = int(np.nanargmax(np.abs(vals)))
    signed = float(vals[idx])
    if signed > 0:
        direction = "positive"
    elif signed < 0:
        direction = "negative"
    else:
        direction = "zero"
    return {
        "eval_label_peak_abs": abs(signed),
        "eval_label_peak_signed": signed,
        "eval_label_peak_idx": idx,
        "eval_label_peak_time_rel_s": float(time_axis[idx]),
        "eval_label_peak_direction": direction,
    }


def morphology_from_reversal_count(count: int) -> str:
    if count <= 0:
        return "single_lobe"
    if count == 1:
        return "reverse_correction"
    return "multi_correction"


def load_index_tables() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cfg in WINDOW_CONFIGS:
        window_id = cfg["window_config_id"]
        path = PROCESSED_TABLE_DIR / f"sample_index_{window_id}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["processed_npz_path"] = str((PROCESSED_ARRAY_DIR / f"{window_id}.npz")).replace("\\", "/")
        df["processed_index_path"] = str(path).replace("\\", "/")
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_response_eval_metadata(index_df: pd.DataFrame) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    thresholds: dict[str, dict[str, float]] = {}
    for cfg in WINDOW_CONFIGS:
        window_id = cfg["window_config_id"]
        npz_path = PROCESSED_ARRAY_DIR / f"{window_id}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        z = np.load(npz_path, allow_pickle=True)
        y = z["label_steer_delta"].astype(np.float32)
        mask = z["label_valid_mask"].astype(bool)
        time_axis = z["label_time_rel_s"].astype(np.float32)
        meta = index_df[index_df["window_config_id"] == window_id].sort_values("array_row").reset_index(drop=True)
        if len(meta) != y.shape[0]:
            raise ValueError(f"{window_id}: index rows {len(meta)} != label rows {y.shape[0]}")

        peak_abs = np.full(y.shape[0], np.nan, dtype=np.float32)
        for i in range(y.shape[0]):
            valid = mask[i] & np.isfinite(y[i])
            if valid.any():
                peak_abs[i] = float(np.nanmax(np.abs(np.where(valid, y[i], np.nan))))
        train_mask = meta[DEFAULT_SPLIT_STRATEGY].astype(str).to_numpy() == "train"
        train_peak = peak_abs[train_mask & np.isfinite(peak_abs)]
        all_peak = peak_abs[np.isfinite(peak_abs)]
        large_thr = float(np.nanpercentile(train_peak if train_peak.size else all_peak, 75)) if all_peak.size else float("nan")
        difficult_thr = float(np.nanpercentile(train_peak if train_peak.size else all_peak, 80)) if all_peak.size else float("nan")
        thresholds[window_id] = {
            "large_response_threshold_train_session_p75": large_thr,
            "difficult_threshold_train_session_p80": difficult_thr,
        }

        for i in range(y.shape[0]):
            valid = mask[i] & np.isfinite(y[i])
            arr = np.where(valid, y[i], np.nan)
            peak = peak_stats(arr, time_axis)
            onset_thr = max(0.015, 0.2 * max(float(peak["eval_label_peak_abs"]), 1e-6))
            onset_idx = first_crossing(np.nan_to_num(arr, nan=0.0), onset_thr)
            rev_count = reversal_count(arr)
            tail_valid = arr[valid]
            row = {
                "sample_id": meta.at[i, "sample_id"],
                "window_config_id": window_id,
                **peak,
                "eval_label_onset_time_rel_s": float(time_axis[onset_idx]) if onset_idx >= 0 else float("nan"),
                "eval_label_reversal_count": rev_count,
                "eval_label_zero_crossing_has": zero_crossing_has(arr),
                "eval_label_morphology": morphology_from_reversal_count(rev_count),
                "eval_label_tail_value": float(tail_valid[-1]) if tail_valid.size else float("nan"),
                "eval_label_tail_abs": abs(float(tail_valid[-1])) if tail_valid.size else float("nan"),
                "eval_is_large_response_train_session_p75": int(
                    np.isfinite(peak["eval_label_peak_abs"]) and np.isfinite(large_thr) and peak["eval_label_peak_abs"] >= large_thr
                ),
                "eval_is_difficult_train_session_p80": int(
                    np.isfinite(peak["eval_label_peak_abs"]) and np.isfinite(difficult_thr) and peak["eval_label_peak_abs"] >= difficult_thr
                ),
                "eval_threshold_note": "label-derived eval-only; thresholds fitted on session-level train samples for each window",
                **thresholds[window_id],
            }
            all_rows.append(row)
    return pd.DataFrame(all_rows)


def build_event_anchor_table(events: pd.DataFrame, eligibility: pd.DataFrame, modality: pd.DataFrame) -> pd.DataFrame:
    event_df = events.copy()
    event_df["event_uid"] = event_df["instability_event_uid"].astype(str)
    elig_cols = [
        "event_uid",
        "oldcode_usable",
        "oldcode_drop_reason",
        "history_full_3s_oldcode",
        "future_full_2s_oldcode",
        "future_rows_available_oldcode",
        "random_event_split",
        "session_level_split",
        "subject_level_split",
    ]
    if not eligibility.empty:
        event_df = event_df.merge(eligibility[elig_cols], on="event_uid", how="left", suffixes=("", "_elig"))
    for col in elig_cols[1:]:
        if col not in event_df.columns:
            event_df[col] = np.nan

    modality_cols = [
        "subject",
        "session_stamp",
        "vehicle_file_count",
        "physio_file_count",
        "eeg_file_count",
        "vehicle_paths",
        "physio_paths",
        "eeg_paths",
        "vehicle_time_min",
        "vehicle_time_max",
        "physio_time_min",
        "physio_time_max",
        "eeg_time_min",
        "eeg_time_max",
        "vehicle_available",
        "physio_available",
        "eeg_available",
        "all_three_modalities_available",
    ]
    event_df = event_df.merge(modality[modality_cols], on=["subject", "session_stamp"], how="left")
    event_df["anchor_time_abs_storage_s"] = pd.to_numeric(event_df["vehicle_time_min"], errors="coerce") + pd.to_numeric(
        event_df["anchor_time_rel_s"], errors="coerce"
    )
    event_df["event_start_abs_storage_s"] = pd.to_numeric(event_df["vehicle_time_min"], errors="coerce") + pd.to_numeric(
        event_df["event_start_rel_s"], errors="coerce"
    )
    event_df["event_end_abs_storage_s"] = pd.to_numeric(event_df["vehicle_time_min"], errors="coerce") + pd.to_numeric(
        event_df["event_end_rel_s"], errors="coerce"
    )
    event_df["anchor_uses_steering"] = False
    event_df["anchor_source_nonsteering_vehicle_dynamics"] = True
    event_df["steering_used_only_as_response_evidence"] = True
    event_df["raw_vehicle_file_modified"] = False

    keep_cols = [
        "event_uid",
        "dataset_candidate_version",
        "subject",
        "session_stamp",
        "vehicle_raw_relative_path",
        "vehicle_raw_absolute_path",
        "vehicle_raw_sha256",
        "anchor_time_rel_s",
        "anchor_time_abs_storage_s",
        "event_start_rel_s",
        "event_end_rel_s",
        "event_start_abs_storage_s",
        "event_end_abs_storage_s",
        "event_duration_s",
        "instability_anchor_source",
        "instability_role",
        "ay_seed_count",
        "roll_rate_seed_count",
        "merged_seed_count",
        "source_event_uids",
        "source_event_types",
        "source_event_levels",
        "vehicle_read_status",
        "vehicle_read_error",
        "peak_abs_ay_window",
        "peak_abs_roll_rate_window",
        "peak_abs_yaw_rate_window",
        "lateral_distance_range_window",
        "median_speed_kmh_window",
        "steering_delta_peak_post3s",
        "instability_review_score",
        "codex_recommended_decision",
        "road_guided_instability_score",
        "road_guided_recommended_decision",
        "old_v400_near_count",
        "old_v400_overlap_count",
        "old_v400_max_level",
        "old_v400_road_type_mode",
        "old_v400_phase_mode",
        "old_v400_min_abs_anchor_gap_s",
        "road_design_map_status",
        "road_design_module_name",
        "road_design_instance_name",
        "road_design_mapping_reliability",
        "road_design_risk_class",
        "manual_label_count",
        "manual_label_decision_mode",
        "oldcode_usable",
        "oldcode_drop_reason",
        "history_full_3s_oldcode",
        "future_full_2s_oldcode",
        "future_rows_available_oldcode",
        "random_event_split",
        "session_level_split",
        "subject_level_split",
        "vehicle_file_count",
        "physio_file_count",
        "eeg_file_count",
        "vehicle_paths",
        "physio_paths",
        "eeg_paths",
        "vehicle_time_min",
        "vehicle_time_max",
        "physio_time_min",
        "physio_time_max",
        "eeg_time_min",
        "eeg_time_max",
        "vehicle_available",
        "physio_available",
        "eeg_available",
        "all_three_modalities_available",
        "anchor_uses_steering",
        "anchor_source_nonsteering_vehicle_dynamics",
        "steering_used_only_as_response_evidence",
        "raw_vehicle_file_modified",
    ]
    return event_df[[c for c in keep_cols if c in event_df.columns]].sort_values(["subject", "session_stamp", "anchor_time_rel_s"])


def build_samples(index_df: pd.DataFrame, response_meta: pd.DataFrame, event_anchor: pd.DataFrame) -> pd.DataFrame:
    cfg_df = pd.DataFrame(WINDOW_CONFIGS)
    samples = index_df.copy()
    samples = samples.merge(cfg_df, on=["window_config_id", "input_start_rel_s", "input_end_rel_s", "label_start_rel_s", "label_end_rel_s"], how="left")
    samples = samples.merge(response_meta, on=["sample_id", "window_config_id"], how="left")
    event_cols = [
        "event_uid",
        "dataset_candidate_version",
        "vehicle_raw_absolute_path",
        "vehicle_raw_sha256",
        "anchor_time_abs_storage_s",
        "event_start_abs_storage_s",
        "event_end_abs_storage_s",
        "instability_anchor_source",
        "instability_role",
        "ay_seed_count",
        "roll_rate_seed_count",
        "peak_abs_ay_window",
        "peak_abs_roll_rate_window",
        "peak_abs_yaw_rate_window",
        "lateral_distance_range_window",
        "median_speed_kmh_window",
        "steering_delta_peak_post3s",
        "old_v400_near_count",
        "old_v400_overlap_count",
        "old_v400_max_level",
        "old_v400_road_type_mode",
        "old_v400_phase_mode",
        "road_design_map_status",
        "road_design_module_name",
        "road_design_instance_name",
        "road_design_mapping_reliability",
        "road_design_risk_class",
        "manual_label_count",
        "manual_label_decision_mode",
        "vehicle_file_count",
        "physio_file_count",
        "eeg_file_count",
        "vehicle_paths",
        "physio_paths",
        "eeg_paths",
        "vehicle_time_min",
        "vehicle_time_max",
        "physio_time_min",
        "physio_time_max",
        "eeg_time_min",
        "eeg_time_max",
        "vehicle_available",
        "physio_available",
        "eeg_available",
        "all_three_modalities_available",
        "anchor_uses_steering",
        "anchor_source_nonsteering_vehicle_dynamics",
        "steering_used_only_as_response_evidence",
    ]
    samples = samples.merge(event_anchor[[c for c in event_cols if c in event_anchor.columns]], on="event_uid", how="left")

    samples["dataset_version"] = DATASET_VERSION
    samples["source_event_version"] = SOURCE_EVENT_VERSION
    samples["processed_dataset_version"] = PROCESSED_DATASET_VERSION
    samples["sample_trace_status"] = "traceable_to_raw_vehicle_and_processed_window"
    samples["sample_quality_status"] = np.where(
        (pd.to_numeric(samples["input_valid_ratio"], errors="coerce") >= 0.999)
        & (pd.to_numeric(samples["label_valid_ratio"], errors="coerce") >= 0.999),
        "vehicle_window_ok",
        "vehicle_window_partial",
    )
    samples["default_split_strategy"] = DEFAULT_SPLIT_STRATEGY
    samples["default_split"] = samples[DEFAULT_SPLIT_STRATEGY]
    samples["input_start_time_rel_s"] = pd.to_numeric(samples["anchor_time_rel_s"], errors="coerce") + pd.to_numeric(
        samples["input_start_rel_s"], errors="coerce"
    )
    samples["input_end_time_rel_s"] = pd.to_numeric(samples["anchor_time_rel_s"], errors="coerce") + pd.to_numeric(
        samples["input_end_rel_s"], errors="coerce"
    )
    samples["label_start_time_rel_s"] = pd.to_numeric(samples["anchor_time_rel_s"], errors="coerce") + pd.to_numeric(
        samples["label_start_rel_s"], errors="coerce"
    )
    samples["label_end_time_rel_s"] = pd.to_numeric(samples["anchor_time_rel_s"], errors="coerce") + pd.to_numeric(
        samples["label_end_rel_s"], errors="coerce"
    )
    samples["input_start_abs_storage_s"] = pd.to_numeric(samples["vehicle_time_min"], errors="coerce") + samples["input_start_time_rel_s"]
    samples["input_end_abs_storage_s"] = pd.to_numeric(samples["vehicle_time_min"], errors="coerce") + samples["input_end_time_rel_s"]
    samples["label_start_abs_storage_s"] = pd.to_numeric(samples["vehicle_time_min"], errors="coerce") + samples["label_start_time_rel_s"]
    samples["label_end_abs_storage_s"] = pd.to_numeric(samples["vehicle_time_min"], errors="coerce") + samples["label_end_time_rel_s"]
    samples["input_start_grid_idx_200hz"] = np.rint(samples["input_start_time_rel_s"] * FS).astype("Int64")
    samples["input_end_grid_idx_200hz"] = np.rint(samples["input_end_time_rel_s"] * FS).astype("Int64")
    samples["label_start_grid_idx_200hz"] = np.rint(samples["label_start_time_rel_s"] * FS).astype("Int64")
    samples["label_end_grid_idx_200hz"] = np.rint(samples["label_end_time_rel_s"] * FS).astype("Int64")
    samples["leakage_risk_level"] = "controlled_vehicle_only_stage2"
    samples["leakage_risk_notes"] = (
        "Anchor comes from non-steering vehicle dynamics ay/roll_rate; input window ends at anchor; "
        "label window starts at anchor; no normalization fitted here; label-derived eval fields are not model inputs or split criteria."
    )
    samples["standardization_scope"] = "not_applied_in_manifest; future training must fit scalers on train split only"
    samples["physio_window_status"] = np.where(samples["physio_available"].map(bool_value), "available_not_extracted_in_v0_1", "not_available")
    samples["eeg_window_status"] = np.where(samples["eeg_available"].map(bool_value), "available_not_extracted_in_v0_1", "not_available")
    samples["style_feature_status"] = "not_extracted_in_v0_1"

    preferred = [
        "sample_id",
        "dataset_version",
        "source_event_version",
        "processed_dataset_version",
        "event_uid",
        "subject",
        "session_stamp",
        "window_config_id",
        "role",
        "is_primary_window",
        "default_split_strategy",
        "default_split",
        *SPLIT_STRATEGIES,
        "causal_setting",
        "anchor_source",
        "instability_anchor_source",
        "event_type",
        "instability_role",
        "event_level",
        "road_type_anchor",
        "anchor_time_rel_s",
        "anchor_time_abs_storage_s",
        "event_start_rel_s",
        "event_end_rel_s",
        "event_duration_s",
        "input_start_rel_s",
        "input_end_rel_s",
        "label_start_rel_s",
        "label_end_rel_s",
        "input_start_time_rel_s",
        "input_end_time_rel_s",
        "label_start_time_rel_s",
        "label_end_time_rel_s",
        "input_start_abs_storage_s",
        "input_end_abs_storage_s",
        "label_start_abs_storage_s",
        "label_end_abs_storage_s",
        "input_start_grid_idx_200hz",
        "input_end_grid_idx_200hz",
        "label_start_grid_idx_200hz",
        "label_end_grid_idx_200hz",
        "array_row",
        "processed_npz_path",
        "processed_index_path",
        "vehicle_relative_path",
        "vehicle_absolute_path",
        "vehicle_sha256",
        "vehicle_paths",
        "physio_paths",
        "eeg_paths",
        "vehicle_available",
        "physio_available",
        "eeg_available",
        "all_three_modalities_available",
        "physio_window_status",
        "eeg_window_status",
        "style_feature_status",
        "input_valid_ratio",
        "label_valid_ratio",
        "sample_quality_status",
        "anchor_uses_steering",
        "anchor_source_nonsteering_vehicle_dynamics",
        "steering_used_only_as_response_evidence",
        "leakage_risk_level",
        "leakage_risk_notes",
        "standardization_scope",
        "instability_review_score",
        "road_guided_instability_score",
        "codex_recommended_decision",
        "road_guided_recommended_decision",
        "ay_seed_count",
        "roll_rate_seed_count",
        "peak_abs_ay_window",
        "peak_abs_roll_rate_window",
        "peak_abs_yaw_rate_window",
        "lateral_distance_range_window",
        "median_speed_kmh_window",
        "steering_delta_peak_post3s",
        "old_v400_near_count",
        "old_v400_overlap_count",
        "old_v400_max_level",
        "old_v400_road_type_mode",
        "old_v400_phase_mode",
        "road_design_module_name",
        "road_design_instance_name",
        "road_design_mapping_reliability",
        "road_design_risk_class",
        "manual_label_count",
        "manual_label_decision_mode",
        "eval_label_peak_abs",
        "eval_label_peak_signed",
        "eval_label_peak_time_rel_s",
        "eval_label_peak_direction",
        "eval_label_onset_time_rel_s",
        "eval_label_reversal_count",
        "eval_label_zero_crossing_has",
        "eval_label_morphology",
        "eval_label_tail_abs",
        "eval_is_large_response_train_session_p75",
        "eval_is_difficult_train_session_p80",
        "large_response_threshold_train_session_p75",
        "difficult_threshold_train_session_p80",
        "eval_threshold_note",
        "sample_trace_status",
        "description_cn",
    ]
    tail = [c for c in samples.columns if c not in preferred]
    return samples[[c for c in preferred if c in samples.columns] + tail].sort_values(
        ["window_config_id", "subject", "session_stamp", "anchor_time_rel_s"]
    )


def build_split_table(event_anchor: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, event in event_anchor[event_anchor["oldcode_usable"].map(bool_value)].iterrows():
        row = {
            "event_uid": event["event_uid"],
            "dataset_version": DATASET_VERSION,
            "subject": event["subject"],
            "session_stamp": event["session_stamp"],
            "anchor_time_rel_s": event["anchor_time_rel_s"],
            "event_type": event.get("instability_role", ""),
            "event_level": event.get("old_v400_max_level", ""),
            "road_type_anchor": event.get("old_v400_road_type_mode", ""),
            "n_window_samples": int((samples["event_uid"] == event["event_uid"]).sum()),
        }
        for split in SPLIT_STRATEGIES:
            row[split] = event.get(split, "")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["subject", "session_stamp", "anchor_time_rel_s"])


def build_split_feasibility(samples: pd.DataFrame) -> pd.DataFrame:
    primary = samples[samples["is_primary_window"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for strategy in SPLIT_STRATEGIES:
        for split_name, grp in primary.groupby(strategy, dropna=False):
            rows.append(
                {
                    "dataset_version": DATASET_VERSION,
                    "split_strategy": strategy,
                    "split": split_name,
                    "n_primary_samples": int(len(grp)),
                    "n_events": int(grp["event_uid"].nunique()),
                    "n_subjects": int(grp["subject"].nunique()),
                    "n_sessions": int((grp["subject"].astype(str) + "__" + grp["session_stamp"].astype(str)).nunique()),
                    "n_all_three_modalities_available": int(grp["all_three_modalities_available"].map(bool_value).sum()),
                    "n_physio_available": int(grp["physio_available"].map(bool_value).sum()),
                    "n_eeg_available": int(grp["eeg_available"].map(bool_value).sum()),
                    "note": "counts only; no label distribution used for assigning splits",
                }
            )
    return pd.DataFrame(rows).sort_values(["split_strategy", "split"])


def build_exclusion_table(events: pd.DataFrame, event_anchor: pd.DataFrame) -> pd.DataFrame:
    usable_events = set(event_anchor.loc[event_anchor["oldcode_usable"].map(bool_value), "event_uid"].astype(str))
    out = event_anchor[~event_anchor["event_uid"].astype(str).isin(usable_events)].copy()
    if out.empty:
        return pd.DataFrame(
            columns=[
                "event_uid",
                "dataset_version",
                "subject",
                "session_stamp",
                "anchor_time_rel_s",
                "exclude_reason",
                "impact",
            ]
        )
    out["dataset_version"] = DATASET_VERSION
    out["exclude_reason"] = out["oldcode_drop_reason"].map(lambda x: clean_str(x, "not_selected_for_complete_window_set"))
    out["impact"] = "excluded from v0.1 formal samples because complete 3 s history and 2 s future coverage is required"
    keep = ["event_uid", "dataset_version", "subject", "session_stamp", "anchor_time_rel_s", "exclude_reason", "impact"]
    return out[keep].sort_values(["subject", "session_stamp", "anchor_time_rel_s"])


def write_copy(df: pd.DataFrame, generic_name: str, versioned_name: str) -> None:
    generic = TABLE_DIR / generic_name
    versioned = TABLE_DIR / versioned_name
    df.to_csv(generic, index=False, encoding="utf-8-sig")
    if versioned != generic:
        shutil.copyfile(generic, versioned)


def write_reports(
    samples: pd.DataFrame,
    event_anchor: pd.DataFrame,
    split_table: pd.DataFrame,
    split_feasibility: pd.DataFrame,
    exclusions: pd.DataFrame,
    response_summary: pd.DataFrame,
) -> dict[str, Any]:
    primary = samples[samples["is_primary_window"].astype(bool)].copy()
    split_counts = (
        primary[DEFAULT_SPLIT_STRATEGY].value_counts().rename_axis("split").reset_index(name="n_primary_samples")
    )
    modality_counts = primary[
        ["vehicle_available", "physio_available", "eeg_available", "all_three_modalities_available"]
    ].apply(lambda col: int(col.map(bool_value).sum()))
    morphology_counts = (
        primary["eval_label_morphology"].value_counts().rename_axis("eval_label_morphology").reset_index(name="n_primary_samples")
    )
    window_counts = samples["window_config_id"].value_counts().rename_axis("window_config_id").reset_index(name="n_samples")

    summary = {
        "dataset_version": DATASET_VERSION,
        "source_event_version": SOURCE_EVENT_VERSION,
        "processed_dataset_version": PROCESSED_DATASET_VERSION,
        "input_high_confidence_events": int(len(event_anchor)),
        "usable_events": int(split_table["event_uid"].nunique()),
        "excluded_events": int(len(exclusions)),
        "sample_rows": int(len(samples)),
        "primary_window_samples": int(len(primary)),
        "window_counts": dict(zip(window_counts["window_config_id"], window_counts["n_samples"])),
        "default_split_strategy": DEFAULT_SPLIT_STRATEGY,
        "default_split_counts": dict(zip(split_counts["split"], split_counts["n_primary_samples"])),
        "modality_available_primary_counts": modality_counts.to_dict(),
        "response_morphology_primary_counts": dict(
            zip(morphology_counts["eval_label_morphology"], morphology_counts["n_primary_samples"])
        ),
        "server_used": False,
        "credential_file_read": False,
        "raw_csv_modified": False,
    }
    (LOG_DIR / "vehicle_instability_highconf_samples_summary_v0_1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    card = f"""# 数据版本卡：车辆失稳高置信正式样本清单 v0.1

生成时间：2026-05-12

## 数据版本

- 数据版本：`{DATASET_VERSION}`
- 事件来源版本：`{SOURCE_EVENT_VERSION}`
- 处理后车辆窗口版本：`{PROCESSED_DATASET_VERSION}`
- 主输入事件表：`{PRIMARY_EVENTS.as_posix()}`
- 正式样本目录：`{OUT_ROOT.as_posix()}`

## 构建规则

1. 只使用全原始车辆重筛得到的高置信车辆失稳事件。
2. 事件锚点来自非方向盘车辆动力学 `ay/roll_rate`，方向盘只作为事件后响应标签和评估元数据。
3. v0.1 要求每个事件具备完整 3 秒历史和 2 秒未来覆盖，因此从 908 个高置信事件中保留 906 个，排除 2 个。
4. 同一事件生成 3 个窗口：`pre1_label2_event_trigger`、`pre2_label2_old_main`、`pre3_label3_response_coverage`。
5. 默认正式切分为 `session_level_split`，同时保留 `random_event_split` 和 `subject_level_split`，但不使用任何标签统计分配 split。
6. 本版本不提取生理、脑电或连续风格窗口，只记录对应模态在原始文件层面的可用性和路径。

## 数量

- 输入高置信事件：{summary['input_high_confidence_events']}
- 可用事件：{summary['usable_events']}
- 排除事件：{summary['excluded_events']}
- 样本行数：{summary['sample_rows']}
- 主窗口样本数：{summary['primary_window_samples']}

## 窗口分布

{window_counts.to_string(index=False)}

## 默认 session-level split

{split_counts.to_string(index=False)}

## 模态可用性，按主窗口样本计数

{modality_counts.rename('n_primary_samples').reset_index().rename(columns={'index': 'modality_flag'}).to_string(index=False)}

## 响应类型，按主窗口 eval-only 标签计数

{morphology_counts.to_string(index=False)}

## 无泄漏说明

- split 由事件、session 或 subject 标识的稳定哈希决定，不用方向盘未来标签和测试集统计。
- manifest 未做标准化；后续训练必须只在 train split 拟合 scaler。
- `eval_label_*` 字段来自未来方向盘标签，只允许用于评估分层、固定图和困难样本分析，不允许作为训练输入、split 决策或特征学习依据。
- 生理和脑电在本版本只记录原始文件是否可用，未抽取窗口，因此不会引入生理窗口泄漏。

## 关键输出

- `samples_master.csv/jsonl`
- `event_anchor_table.csv`
- `split_table.csv`
- `split_feasibility_report.csv`
- `sample_exclusion_reasons.csv`
- `label_eval_only_response_summary.csv`
"""
    card_path = REPORT_DIR / "dataset_version_card_vehicle_instability_highconf_v0_1_cn.md"
    card_path.write_text(card, encoding="utf-8")

    detail = f"""# 车辆失稳高置信正式样本清单 v0.1

生成时间：2026-05-12

## 这一步做了什么

这一步把全原始车辆 CSV 重筛得到的高置信失稳事件，整理成新流程正式 `samples_master`。它不是新模型训练，也不是生理/风格有效性验证。

## 输入

- 高置信事件：`{PRIMARY_EVENTS.as_posix()}`
- 处理后车辆窗口：`{PROCESSED_ROOT.as_posix()}`
- 模态完整性矩阵：`{MODALITY_MATRIX.as_posix()}`

## 输出

- 样本清单：`{(TABLE_DIR / 'samples_master.csv').as_posix()}`
- 样本 JSONL：`{(TABLE_DIR / 'samples_master.jsonl').as_posix()}`
- 事件锚点表：`{(TABLE_DIR / 'event_anchor_table.csv').as_posix()}`
- split 表：`{(TABLE_DIR / 'split_table.csv').as_posix()}`
- split 可行性：`{(TABLE_DIR / 'split_feasibility_report.csv').as_posix()}`
- 排除原因：`{(TABLE_DIR / 'sample_exclusion_reasons.csv').as_posix()}`
- 数据版本卡：`{card_path.as_posix()}`

## 当前判断

906 个高置信失稳事件已经具备可追溯样本记录，可以进入新流程车辆基线准备。需要注意，本版本只是车辆失稳样本清单和窗口索引，不证明旧模型有效，也不证明连续风格、生理或脑电有效。

## 下一步

使用 `pre2_label2_old_main` + `session_level_split` 建立新流程无学习基线和强车辆基线；训练和标准化必须只用 train split。
"""
    detail_path = REPORT_DIR / "vehicle_instability_highconf_samples_v0_1_cn.md"
    detail_path.write_text(detail, encoding="utf-8")

    user_summary = f"""# 阶段 2 用户查看版：车辆失稳高置信样本清单 v0.1

生成时间：2026-05-12

## 为什么做

之前旧代码已经能在 906 个高置信失稳样本上跑通，但那只是旧代码对照。要进入新流程强车辆基线，必须先有正式、可追溯、无泄漏的 `samples_master`。

## 检查了什么

- 每个样本是否能追溯到原始车辆 CSV、sha256、被试、记录和事件锚点。
- 每个样本是否有明确输入窗口和标签窗口。
- split 是否不依赖未来方向盘标签。
- 生理/脑电是否只是记录可用性，没有提前抽窗口或使用。
- 两个未进入正式样本的事件是否有排除原因。

## 目前发现

908 个高置信车辆失稳事件中，906 个满足完整历史和未来窗口要求；2 个因为窗口覆盖不足被排除。906 个事件各生成 3 个窗口，总样本行数 2718。主窗口 `pre2_label2_old_main` 的 session-level split 为 train 611、val 156、test 139。

## 哪些结果可信

样本锚点来自 `ay/roll_rate` 等非方向盘车辆动态，方向盘没有参与锚点定义。manifest 没有做标准化，后续训练必须只在训练集拟合 scaler。`eval_label_*` 字段只用于评估分层，不能作为模型输入。

## 哪些结果还不能下结论

这还不是强车辆基线结果，也不能证明连续风格、生理或脑电有效。生理和脑电目前只是记录了原始文件是否可用，还没有进入窗口构建和增量验证。

## 下一阶段是否可以继续

可以继续进入新流程车辆基线阶段。下一步应先做无学习基线和强车辆基线，再决定是否进入连续风格和生理验证。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_vehicle_instability_highconf_v0_1_cn.md`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/split_feasibility_report.csv`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/sample_exclusion_reasons.csv`
"""
    user_path = REPORT_DIR / "stage02_vehicle_instability_highconf_user_summary_cn.md"
    user_path.write_text(user_summary, encoding="utf-8")
    return summary


def main() -> None:
    ensure_dirs()
    events = pd.read_csv(PRIMARY_EVENTS)
    modality = pd.read_csv(MODALITY_MATRIX)
    eligibility = pd.read_csv(ELIGIBILITY_TABLE) if ELIGIBILITY_TABLE.exists() else pd.DataFrame()
    event_anchor = build_event_anchor_table(events, eligibility, modality)
    index_df = load_index_tables()
    response_meta = load_response_eval_metadata(index_df)
    samples = build_samples(index_df, response_meta, event_anchor)
    split_table = build_split_table(event_anchor, samples)
    split_feasibility = build_split_feasibility(samples)
    exclusions = build_exclusion_table(events, event_anchor)
    response_summary = (
        samples.groupby(["window_config_id", "eval_label_morphology"], dropna=False)
        .agg(
            n_samples=("sample_id", "count"),
            peak_abs_mean=("eval_label_peak_abs", "mean"),
            peak_abs_p75=("eval_label_peak_abs", lambda s: float(np.nanpercentile(s, 75)) if len(s.dropna()) else float("nan")),
        )
        .reset_index()
    )

    write_copy(samples, "samples_master.csv", "samples_master_vehicle_instability_highconf_v0_1.csv")
    samples.to_json(TABLE_DIR / "samples_master.jsonl", orient="records", lines=True, force_ascii=False)
    shutil.copyfile(TABLE_DIR / "samples_master.jsonl", TABLE_DIR / "samples_master_vehicle_instability_highconf_v0_1.jsonl")
    write_copy(event_anchor, "event_anchor_table.csv", "event_anchor_table_vehicle_instability_highconf_v0_1.csv")
    write_copy(split_table, "split_table.csv", "split_table_vehicle_instability_highconf_v0_1.csv")
    write_copy(split_feasibility, "split_feasibility_report.csv", "split_feasibility_report_vehicle_instability_highconf_v0_1.csv")
    write_copy(exclusions, "sample_exclusion_reasons.csv", "sample_exclusion_reasons_vehicle_instability_highconf_v0_1.csv")
    write_copy(pd.DataFrame(WINDOW_CONFIGS), "window_config_table.csv", "window_config_table_vehicle_instability_highconf_v0_1.csv")
    write_copy(response_summary, "label_eval_only_response_summary.csv", "label_eval_only_response_summary_vehicle_instability_highconf_v0_1.csv")
    summary = write_reports(samples, event_anchor, split_table, split_feasibility, exclusions, response_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
