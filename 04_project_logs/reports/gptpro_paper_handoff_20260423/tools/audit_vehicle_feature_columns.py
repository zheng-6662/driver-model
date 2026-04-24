#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audit key feature inputs for vehicle-aligned CSV files.

The script follows the training script's data-root semantics in a lightweight way:
- `DRIVER_MODEL_ROOT` overrides the input root when set.
- otherwise the default root is inferred from `PROJECT_ROOT / "01_datasets"` by
  selecting the two-level directory that matches the training layout
  `<data_root>/<subject>/vehicle/*_vehicle_aligned_cleaned.csv`.

Outputs are written to:
  04_project_logs/reports/feature_input_audit_20260421/
    - feature_presence_summary.csv
    - speed_unit_check.csv
    - feature_presence_report.md
"""

import argparse
import csv
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


VEHICLE_SUFFIX = "_vehicle_aligned_cleaned.csv"
EVENT_V312_SUFFIX = "_vehicle_aligned_cleaned_events_v312.csv"
EVENT_V400_SUFFIX = "_vehicle_aligned_cleaned_events_v400_context.csv"
REPORT_DIR_NAME = "feature_input_audit_20260421"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_SCRIPT_PATH = (
    PROJECT_ROOT
    / "02_code"
    / "final_code"
    / "model"
    / "training"
    / "future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py"
)
DEFAULT_REPORT_DIR = (
    PROJECT_ROOT / "04_project_logs" / "reports" / REPORT_DIR_NAME
)


FEATURE_SPECS = [
    {
        "key": "time_s",
        "label": "Time axis",
        "observed_aliases": ["t_s", "time", "timestamp"],
        "training_exact_aliases": ["t_s", "time", "timestamp"],
        "numeric": True,
    },
    {
        "key": "storage_time",
        "label": "Storage time",
        "observed_aliases": ["StorageTime", "storage_time", "storagetime"],
        "training_exact_aliases": ["StorageTime", "storage_time", "storagetime"],
        "numeric": True,
    },
    {
        "key": "roll",
        "label": "Roll",
        "observed_aliases": ["zx|roll", "roll", "Roll"],
        "training_exact_aliases": ["zx|roll", "roll", "Roll"],
        "numeric": True,
    },
    {
        "key": "steering_wheel",
        "label": "Steering wheel",
        "observed_aliases": ["zx|SteeringWheel", "SteeringWheel", "steer"],
        "training_exact_aliases": ["zx|SteeringWheel", "SteeringWheel", "steer"],
        "numeric": True,
    },
    {
        "key": "yaw_rate",
        "label": "Yaw rate",
        "observed_aliases": ["zx|vyaw", "vyaw", "YawRate", "zx|YawRate", "yaw_rate"],
        "training_exact_aliases": ["vyaw", "zx|vyaw", "YawRate", "zx|YawRate", "yaw_rate"],
        "numeric": True,
    },
    {
        "key": "speed_vx",
        "label": "Longitudinal speed",
        "observed_aliases": ["zx|vx", "Vx", "vx", "Speed", "speed"],
        "training_exact_aliases": ["zx|vx", "Vx", "vx", "Speed", "speed"],
        "numeric": True,
    },
    {
        "key": "speed_kmh",
        "label": "Speed km/h companion",
        "observed_aliases": ["zx1|v_km/h", "v_km/h"],
        "training_exact_aliases": ["zx1|v_km/h", "v_km/h"],
        "numeric": True,
    },
    {
        "key": "z_position",
        "label": "Z position",
        "observed_aliases": ["zx|z", "z", "Z"],
        "training_exact_aliases": ["zx|z", "z", "Z"],
        "numeric": True,
    },
    {
        "key": "lateral_accel",
        "label": "Lateral accel",
        "observed_aliases": ["zx|ay", "ay", "Ay", "lat_acc"],
        "training_exact_aliases": ["zx|ay", "ay", "Ay", "lat_acc"],
        "numeric": True,
    },
    {
        "key": "longitudinal_accel",
        "label": "Longitudinal accel",
        "observed_aliases": ["zx|ax", "ax", "Ax", "Long_acc"],
        "training_exact_aliases": ["zx|ax", "ax", "Ax", "Long_acc"],
        "numeric": True,
    },
    {
        "key": "lane_distance",
        "label": "Lane distance",
        "observed_aliases": [
            "zx1|lateraldistance",
            "lateraldistance",
            "lateralDistance",
            "lateraldistance_start",
        ],
        "training_exact_aliases": [
            "lateraldistance",
            "lateralDistance",
            "lateraldistance_start",
        ],
        "numeric": True,
    },
    {
        "key": "lane_curvature",
        "label": "Lane curvature",
        "observed_aliases": [
            "zx1|lanecurvatureXY",
            "laneCurvature",
            "lanecurvature_start",
        ],
        "training_exact_aliases": [
            "zx1|lanecurvatureXY",
            "laneCurvature",
            "lanecurvature_start",
        ],
        "numeric": True,
    },
    {
        "key": "road_type_fixed",
        "label": "Road type fixed",
        "observed_aliases": ["road_type_fixed", "road_type", "roadType_fixed"],
        "training_exact_aliases": ["road_type_fixed", "road_type", "roadType_fixed"],
        "numeric": True,
    },
    {
        "key": "ref_nn_ok",
        "label": "Reference NN ok",
        "observed_aliases": ["ref_nn_ok", "ref_ok", "refnn_ok"],
        "training_exact_aliases": ["ref_nn_ok", "ref_ok", "refnn_ok"],
        "numeric": True,
    },
    {
        "key": "yaw",
        "label": "Yaw",
        "observed_aliases": ["zx|yaw", "yaw", "Yaw"],
        "training_exact_aliases": ["zx|yaw", "yaw", "Yaw"],
        "numeric": True,
    },
]


def infer_default_data_root(project_root):
    dataset_root = project_root / "01_datasets"
    candidates = []
    if not dataset_root.exists():
        raise FileNotFoundError("01_datasets directory not found under project root: {}".format(project_root))

    for first_level in sorted(dataset_root.iterdir(), key=lambda p: str(p)):
        if not first_level.is_dir():
            continue
        for second_level in sorted(first_level.iterdir(), key=lambda p: str(p)):
            if not second_level.is_dir():
                continue
            count = sum(1 for _ in second_level.glob("*/vehicle/*{}".format(VEHICLE_SUFFIX)))
            if count:
                candidates.append((count, second_level))

    if not candidates:
        raise FileNotFoundError(
            "Could not infer a training-style data root under {}".format(dataset_root)
        )

    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    return candidates[0][1]


DEFAULT_DATA_ROOT = infer_default_data_root(PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit vehicle feature column presence and speed-unit consistency."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DRIVER_MODEL_ROOT", str(DEFAULT_DATA_ROOT)),
        help="Root directory containing <subject>/vehicle/*.csv and <subject>/event/*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Directory for audit outputs",
    )
    return parser.parse_args()


def find_first_exact(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def find_vehicle_files(data_root):
    exact_pattern_files = sorted(data_root.glob("*/vehicle/*{}".format(VEHICLE_SUFFIX)))
    if exact_pattern_files:
        return exact_pattern_files, "exact_training_pattern"

    fallback_files = sorted(
        path
        for path in data_root.rglob("*{}".format(VEHICLE_SUFFIX))
        if path.parent.name.lower() == "vehicle"
    )
    return fallback_files, "recursive_fallback"


def make_feature_state():
    return {
        "files_present_observed": 0,
        "files_present_training_exact": 0,
        "observed_columns": Counter(),
        "training_columns": Counter(),
        "numeric_file_count": 0,
        "finite_value_count": 0,
        "global_min": None,
        "global_max": None,
        "file_medians": [],
    }


def update_numeric_summary(state, series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return

    state["numeric_file_count"] += 1
    state["finite_value_count"] += int(finite.size)
    state["global_min"] = (
        float(finite.min())
        if state["global_min"] is None
        else min(state["global_min"], float(finite.min()))
    )
    state["global_max"] = (
        float(finite.max())
        if state["global_max"] is None
        else max(state["global_max"], float(finite.max()))
    )
    state["file_medians"].append(float(np.median(finite)))


def vehicle_basename(vehicle_file):
    return vehicle_file.name[: -len(VEHICLE_SUFFIX)]


def expected_event_file(vehicle_file, suffix):
    return vehicle_file.parent.parent / "event" / "{}{}".format(vehicle_basename(vehicle_file), suffix)


def format_counter(counter_obj):
    if not counter_obj:
        return ""
    parts = []
    for name, count in sorted(counter_obj.items(), key=lambda item: (-item[1], item[0])):
        parts.append("{} ({})".format(name, count))
    return "; ".join(parts)


def format_float(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    text = ("{0:." + str(digits) + "f}").format(float(value))
    return text.rstrip("0").rstrip(".") if "." in text else text


def markdown_table(headers, rows):
    if not rows:
        return ""

    def stringify(value):
        if value is None:
            return ""
        return str(value)

    widths = []
    for idx, header in enumerate(headers):
        width = len(header)
        for row in rows:
            width = max(width, len(stringify(row[idx])))
        widths.append(width)

    def build_row(row_values):
        padded = []
        for idx, value in enumerate(row_values):
            padded.append(stringify(value).ljust(widths[idx]))
        return "| " + " | ".join(padded) + " |"

    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [build_row(headers), divider]
    for row in rows:
        lines.append(build_row(row))
    return "\n".join(lines)


def build_feature_summary_rows(feature_states, total_files):
    rows = []
    for spec in FEATURE_SPECS:
        state = feature_states[spec["key"]]
        row = {
            "feature_key": spec["key"],
            "label": spec["label"],
            "observed_aliases": " | ".join(spec["observed_aliases"]),
            "training_exact_aliases": " | ".join(spec["training_exact_aliases"]),
            "files_present_observed": state["files_present_observed"],
            "files_missing_observed": total_files - state["files_present_observed"],
            "presence_pct_observed": round(
                100.0 * state["files_present_observed"] / total_files, 2
            )
            if total_files
            else 0.0,
            "files_present_training_exact": state["files_present_training_exact"],
            "files_missing_training_exact": total_files - state["files_present_training_exact"],
            "observed_exact_columns": format_counter(state["observed_columns"]),
            "training_exact_columns_found": format_counter(state["training_columns"]),
            "numeric_file_count": state["numeric_file_count"],
            "finite_value_count": state["finite_value_count"],
            "global_min": state["global_min"],
            "median_of_file_medians": (
                float(np.median(np.asarray(state["file_medians"], dtype=np.float64)))
                if state["file_medians"]
                else None
            ),
            "global_max": state["global_max"],
        }
        rows.append(row)
    return rows


def build_report(
    data_root,
    scan_mode,
    vehicle_files,
    feature_summary_df,
    speed_df,
    missing_v312,
    missing_v400,
    lane_naming_counter,
):
    total_vehicle_files = len(vehicle_files)
    dominant_lane_name = ""
    dominant_lane_count = 0
    if lane_naming_counter:
        dominant_lane_name, dominant_lane_count = lane_naming_counter.most_common(1)[0]
    lane_is_dominant = (
        dominant_lane_name == "zx1|lateraldistance" and dominant_lane_count == total_vehicle_files
    )

    ratio_valid = speed_df[speed_df["ratio_positive_finite_rows"] > 0].copy()
    ratio_zero = speed_df[speed_df["ratio_positive_finite_rows"] == 0].copy()
    strong_ratio = pd.DataFrame(columns=speed_df.columns)
    ratio_outliers = pd.DataFrame(columns=speed_df.columns)
    if not ratio_valid.empty:
        strong_ratio = ratio_valid[
            ratio_valid["abs_ratio_median_minus_3_6"] <= 0.01
        ].copy()
        ratio_outliers = ratio_valid[
            ratio_valid["abs_ratio_median_minus_3_6"] > 0.05
        ].copy()
        ratio_outliers = ratio_outliers.sort_values(
            by=["abs_ratio_median_minus_3_6", "ratio_positive_finite_rows"],
            ascending=[False, True],
        )

    summary_rows = []
    for row in feature_summary_df.to_dict(orient="records"):
        summary_rows.append(
            [
                row["feature_key"],
                row["files_present_observed"],
                row["files_present_training_exact"],
                row["observed_exact_columns"],
                format_float(row["global_min"]),
                format_float(row["median_of_file_medians"]),
                format_float(row["global_max"]),
            ]
        )

    outlier_rows = []
    for row in ratio_outliers.head(5).to_dict(orient="records"):
        outlier_rows.append(
            [
                row["basename"],
                row["ratio_positive_finite_rows"],
                format_float(row["ratio_min"]),
                format_float(row["ratio_median"]),
                format_float(row["ratio_max"]),
            ]
        )

    lines = []
    lines.append("# Feature Input Audit")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Training script: `{}`".format(TRAINING_SCRIPT_PATH))
    lines.append("- Resolved data root: `{}`".format(data_root))
    lines.append("- Vehicle file scan mode: `{}`".format(scan_mode))
    lines.append("- Total vehicle files audited: `{}`".format(total_vehicle_files))
    lines.append("")
    lines.append("## Event Pair Coverage")
    lines.append("")
    lines.append("- Missing paired `v312` event files: `{}`".format(len(missing_v312)))
    lines.append("- Missing paired `v400` event files: `{}`".format(len(missing_v400)))
    if missing_v312:
        lines.append("- Missing `v312` basenames: `{}`".format(", ".join(missing_v312)))
    if missing_v400:
        lines.append("- Missing `v400` basenames: `{}`".format(", ".join(missing_v400)))
    if not missing_v312 and not missing_v400:
        lines.append("- All vehicle files have both paired event files.")
    lines.append("")
    lines.append("## Lane Column Naming")
    lines.append("")
    lines.append("- Exact lane-related column counts: `{}`".format(format_counter(lane_naming_counter)))
    lines.append(
        "- `zx1|lateraldistance` dominant naming: `{}`".format("yes" if lane_is_dominant else "no")
    )
    lane_row = feature_summary_df[feature_summary_df["feature_key"] == "lane_distance"].iloc[0]
    lines.append(
        "- Lane distance observed in `{}` files, but exact training lookup aliases match `{}` files.".format(
            int(lane_row["files_present_observed"]),
            int(lane_row["files_present_training_exact"]),
        )
    )
    lines.append("")
    lines.append("## Feature Presence Summary")
    lines.append("")
    lines.append(
        "Numeric columns report `global_min`, `median_of_file_medians`, and `global_max` across files."
    )
    lines.append("")
    lines.append(
        markdown_table(
            [
                "feature_key",
                "files_present",
                "training_exact",
                "observed_exact_columns",
                "global_min",
                "median_of_file_medians",
                "global_max",
            ],
            summary_rows,
        )
    )
    lines.append("")
    lines.append("## Speed Unit Check")
    lines.append("")
    lines.append(
        "- Files with both speed columns and positive finite ratio rows: `{}`".format(
            int((speed_df["ratio_positive_finite_rows"] > 0).sum())
        )
    )
    lines.append(
        "- Files with zero positive finite ratio rows: `{}`".format(len(ratio_zero))
    )
    lines.append(
        "- Files whose ratio median is within `+-0.01` of `3.6`: `{}`".format(len(strong_ratio))
    )
    if not ratio_valid.empty:
        lines.append(
            "- Median of per-file ratio medians: `{}`".format(
                format_float(float(ratio_valid["ratio_median"].median()))
            )
        )
    if not ratio_zero.empty:
        zero_names = ", ".join(ratio_zero["basename"].astype(str).tolist())
        lines.append("- Zero-ratio basenames: `{}`".format(zero_names))
    if not ratio_outliers.empty:
        lines.append(
            "- Outlier files with ratio median more than `0.05` away from `3.6` are shown below."
        )
        lines.append("")
        lines.append(
            markdown_table(
                ["basename", "ratio_rows", "ratio_min", "ratio_median", "ratio_max"],
                outlier_rows,
            )
        )
        lines.append("")
    lines.append(
        "- Interpretation: when the ratio clusters near `3.6`, `zx|vx` behaves like m/s and `zx1|v_km/h` behaves like km/h."
    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vehicle_files, scan_mode = find_vehicle_files(data_root)
    if not vehicle_files:
        raise FileNotFoundError(
            "No vehicle files matching *{} found under {}".format(VEHICLE_SUFFIX, data_root)
        )

    feature_states = dict((spec["key"], make_feature_state()) for spec in FEATURE_SPECS)
    lane_naming_counter = Counter()
    missing_v312 = []
    missing_v400 = []
    speed_rows = []

    for vehicle_file in vehicle_files:
        with vehicle_file.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))

        matched_observed = {}
        numeric_columns = []
        for spec in FEATURE_SPECS:
            state = feature_states[spec["key"]]
            observed_match = find_first_exact(header, spec["observed_aliases"])
            training_match = find_first_exact(header, spec["training_exact_aliases"])
            if observed_match is not None:
                state["files_present_observed"] += 1
                state["observed_columns"][observed_match] += 1
                matched_observed[spec["key"]] = observed_match
                if spec["numeric"]:
                    numeric_columns.append(observed_match)
            if training_match is not None:
                state["files_present_training_exact"] += 1
                state["training_columns"][training_match] += 1

        for exact_lane_name in [
            "zx1|lateraldistance",
            "lateraldistance",
            "lateralDistance",
            "lateraldistance_start",
        ]:
            if exact_lane_name in header:
                lane_naming_counter[exact_lane_name] += 1

        event_v312 = expected_event_file(vehicle_file, EVENT_V312_SUFFIX)
        event_v400 = expected_event_file(vehicle_file, EVENT_V400_SUFFIX)
        if not event_v312.exists():
            missing_v312.append(vehicle_basename(vehicle_file))
        if not event_v400.exists():
            missing_v400.append(vehicle_basename(vehicle_file))

        if numeric_columns:
            df = pd.read_csv(
                vehicle_file,
                usecols=sorted(set(numeric_columns)),
                low_memory=False,
            )
        else:
            df = pd.DataFrame()

        for spec in FEATURE_SPECS:
            observed_column = matched_observed.get(spec["key"])
            if observed_column is None or not spec["numeric"]:
                continue
            update_numeric_summary(feature_states[spec["key"]], df[observed_column])

        speed_vx_col = matched_observed.get("speed_vx")
        speed_kmh_col = matched_observed.get("speed_kmh")
        ratio_min = None
        ratio_median = None
        ratio_max = None
        ratio_p01 = None
        ratio_p99 = None
        abs_ratio_median_minus_3_6 = None
        ratio_rows = 0
        unit_inference = "speed_columns_missing"
        if speed_vx_col is not None and speed_kmh_col is not None:
            vx_values = pd.to_numeric(df[speed_vx_col], errors="coerce").to_numpy(
                dtype=np.float64, copy=False
            )
            kmh_values = pd.to_numeric(df[speed_kmh_col], errors="coerce").to_numpy(
                dtype=np.float64, copy=False
            )
            mask = (
                np.isfinite(vx_values)
                & np.isfinite(kmh_values)
                & (vx_values > 0.0)
                & (kmh_values > 0.0)
            )
            ratio = kmh_values[mask] / vx_values[mask]
            ratio_rows = int(ratio.size)
            if ratio_rows:
                ratio_min = float(np.min(ratio))
                ratio_median = float(np.median(ratio))
                ratio_max = float(np.max(ratio))
                ratio_p01 = float(np.quantile(ratio, 0.01))
                ratio_p99 = float(np.quantile(ratio, 0.99))
                abs_ratio_median_minus_3_6 = abs(ratio_median - 3.6)
                if ratio_rows < 100:
                    unit_inference = "review_sparse_positive_rows"
                elif abs_ratio_median_minus_3_6 <= 0.05:
                    unit_inference = "consistent_with_vx_m_per_s_and_v_km_h"
                else:
                    unit_inference = "review_ratio_median"
            else:
                unit_inference = "no_positive_finite_rows"

        speed_rows.append(
            {
                "subject": vehicle_file.parent.parent.name,
                "basename": vehicle_basename(vehicle_file),
                "vehicle_file": str(vehicle_file),
                "speed_vx_column": speed_vx_col or "",
                "speed_kmh_column": speed_kmh_col or "",
                "ratio_positive_finite_rows": ratio_rows,
                "ratio_min": ratio_min,
                "ratio_median": ratio_median,
                "ratio_max": ratio_max,
                "ratio_p01": ratio_p01,
                "ratio_p99": ratio_p99,
                "abs_ratio_median_minus_3_6": abs_ratio_median_minus_3_6,
                "unit_inference": unit_inference,
            }
        )

    feature_summary_rows = build_feature_summary_rows(feature_states, len(vehicle_files))
    feature_summary_df = pd.DataFrame(feature_summary_rows)
    speed_df = pd.DataFrame(speed_rows).sort_values(by=["subject", "basename"]).reset_index(drop=True)

    feature_summary_path = output_dir / "feature_presence_summary.csv"
    speed_check_path = output_dir / "speed_unit_check.csv"
    report_path = output_dir / "feature_presence_report.md"

    feature_summary_df.to_csv(feature_summary_path, index=False, encoding="utf-8-sig")
    speed_df.to_csv(speed_check_path, index=False, encoding="utf-8-sig")
    report_text = build_report(
        data_root=data_root,
        scan_mode=scan_mode,
        vehicle_files=vehicle_files,
        feature_summary_df=feature_summary_df,
        speed_df=speed_df,
        missing_v312=missing_v312,
        missing_v400=missing_v400,
        lane_naming_counter=lane_naming_counter,
    )
    report_path.write_text(report_text, encoding="utf-8-sig")

    print("Audited {} vehicle files under {}".format(len(vehicle_files), data_root))
    print("Wrote {}".format(feature_summary_path))
    print("Wrote {}".format(speed_check_path))
    print("Wrote {}".format(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
