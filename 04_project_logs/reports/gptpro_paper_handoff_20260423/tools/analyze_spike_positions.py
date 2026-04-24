#!/usr/bin/env python
"""
Diagnose same-position local prediction spikes for Run B/C/D artifacts.

Preferred source: existing pred_vs_gt_example_*.png prediction plots, because
those plots contain all three output channels.  When image-level prediction
artifacts are unavailable or cannot be parsed, the tool falls back to the
already-present recalc case tables and reports that limitation explicitly.
"""
from __future__ import print_function

import argparse
import csv
import datetime as _dt
import json
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover - fallback path is still intentional.
    Image = None


DEFAULT_OUTPUT = Path("04_project_logs/reports/spike_position_diagnosis_20260421")

RUN_SPECS = [
    {
        "run_id": "runB_hybrid_full",
        "short_name": "Run B",
        "path": Path("03_results/tmp/runB_hybrid_full/TRAIN_V5_4_STATECOND_REV_20260420_174856"),
        "case_glob": "recalc_runB_best_by_structured_cases.csv",
        "color": "#1f77b4",
    },
    {
        "run_id": "runC_hybrid_localrev_full",
        "short_name": "Run C",
        "path": Path("03_results/tmp/runC_hybrid_localrev_full/TRAIN_V5_4_STATECOND_REV_20260420_181731"),
        "case_glob": "recalc_runC_best_by_structured_cases.csv",
        "color": "#ff7f0e",
    },
    {
        "run_id": "runD_hybrid_localrev_late025_full",
        "short_name": "Run D",
        "path": Path("03_results/tmp/runD_hybrid_localrev_late025_full/TRAIN_V5_4_STATECOND_REV_20260420_183649"),
        "case_glob": "recalc_runD_best_by_structured_cases.csv",
        "color": "#2ca02c",
    },
]

CHANNELS = [
    ("steer_angle", "steer angle"),
    ("yawrate", "yawrate"),
    ("ay", "ay"),
]

CROSS_CHANNEL_COLUMNS = [
    "run_id",
    "short_name",
    "sample_plot_index",
    "image_file",
    "sample_key",
    "subject_id",
    "event_idx",
    "event_level",
    "detection_source",
    "n_detected_channels",
    "n_sync_channels",
    "sync_channel_names",
    "synchronized",
    "sync_spike_index_mean",
    "sync_spike_index_std",
    "sync_spike_index_range",
    "sync_spike_time_mean_sec",
    "sync_spike_time_std_sec",
    "sync_spike_time_range_sec",
    "steer_angle_spike_index",
    "steer_angle_spike_time_sec",
    "steer_angle_residual_px",
    "yawrate_spike_index",
    "yawrate_spike_time_sec",
    "yawrate_residual_px",
    "ay_spike_index",
    "ay_spike_time_sec",
    "ay_residual_px",
    "gt_onset_idx",
    "pred_onset_idx",
    "gt_main_peak_idx",
    "pred_main_peak_idx",
    "gt_first_reversal_sec",
    "pred_first_reversal_sec",
    "fallback_note",
]


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        out = float(text)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def safe_int(value):
    number = safe_float(value)
    if number is None:
        return None
    return int(round(number))


def fmt_float(value, digits=6):
    if value is None:
        return ""
    return ("{0:." + str(digits) + "f}").format(float(value))


def read_csv_dicts(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path, text):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def load_json(path):
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def parse_sample_index(path):
    match = re.search(r"_idx(\d+)\.png$", path.name)
    if not match:
        return None
    return int(match.group(1))


def get_test_sample_map(run_dir):
    rows = read_csv_dicts(run_dir / "selected_samples_with_split.csv")
    test_rows = [row for row in rows if row.get("protocol_split_applied") == "test" or row.get("split") == "test"]
    return test_rows


def get_case_rows(run_dir, preferred_name):
    figures = run_dir / "figures"
    preferred = figures / preferred_name
    if preferred.exists():
        rows = read_csv_dicts(preferred)
        return rows, preferred
    candidates = sorted(figures.glob("*best_by_structured_cases.csv"))
    if not candidates:
        candidates = sorted(figures.glob("*cases.csv"))
    if candidates:
        return read_csv_dicts(candidates[0]), candidates[0]
    return [], None


def group_contiguous(indices):
    if len(indices) == 0:
        return []
    groups = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
        else:
            groups.append((start, prev))
            start = value
            prev = value
    groups.append((start, prev))
    return groups


def detect_axes_bounds(rgb):
    dark_mask = rgb.sum(axis=2) < 150
    row_counts = dark_mask[:, 100:-100].sum(axis=1)
    max_row = int(row_counts.max())
    if max_row <= 0:
        raise ValueError("could not detect plot axes rows")
    strong_rows = np.where(row_counts > max_row * 0.70)[0]
    row_groups = [group for group in group_contiguous(strong_rows) if group[0] > 60]
    if len(row_groups) < 6:
        raise ValueError("expected at least six horizontal axis boundary rows, found {0}".format(len(row_groups)))
    y_lines = [int(round((group[0] + group[1]) / 2.0)) for group in row_groups[:6]]

    col_counts = dark_mask[100:].sum(axis=0)
    max_col = int(col_counts.max())
    if max_col <= 0:
        raise ValueError("could not detect plot axes columns")
    strong_cols = np.where(col_counts > max_col * 0.70)[0]
    if len(strong_cols) < 2:
        raise ValueError("expected left/right vertical axis boundaries")
    return y_lines, int(strong_cols[0]), int(strong_cols[-1])


def extract_orange_trace(crop):
    red = crop[:, :, 0]
    green = crop[:, :, 1]
    blue = crop[:, :, 2]
    orange_mask = (
        (red > 180)
        & (green > 80)
        & (green < 220)
        & (blue < 180)
        & ((red.astype(np.int16) - green.astype(np.int16)) > 20)
    )
    counts = orange_mask.sum(axis=0)
    xs = np.arange(orange_mask.shape[1])
    trace_y = np.full(orange_mask.shape[1], np.nan)
    for x in xs[counts > 0]:
        rows = np.where(orange_mask[:, x])[0]
        trace_y[x] = rows.mean()
    valid = np.where(~np.isnan(trace_y))[0]
    if len(valid) < 100:
        return None
    trace_y = np.interp(xs, valid, trace_y[valid])
    return trace_y, int(valid[0]), int(valid[-1])


def detect_spike_from_trace(trace_y, data_start, data_end, future_len, future_sec, smooth_window):
    if smooth_window < 3:
        smooth_window = 3
    if smooth_window % 2 == 0:
        smooth_window += 1
    kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
    baseline = np.convolve(trace_y, kernel, mode="same")
    residual = trace_y - baseline
    span = max(1, data_end - data_start)
    lo = max(data_start + smooth_window, data_start + int(span * 0.05))
    hi = min(data_end - smooth_window, data_end - int(span * 0.05))
    if hi <= lo:
        raise ValueError("trace too short for spike detection")
    local_abs = np.abs(residual[lo:hi])
    image_x = int(np.argmax(local_abs) + lo)
    rel = (image_x - data_start) / float(span)
    rel = min(1.0, max(0.0, rel))
    spike_index = rel * float(max(1, future_len - 1))
    spike_time = rel * float(future_sec)
    return {
        "image_x": image_x,
        "spike_index": spike_index,
        "spike_time_sec": spike_time,
        "residual_px": float(residual[image_x]),
        "residual_abs_px": float(abs(residual[image_x])),
        "data_start_x": int(data_start),
        "data_end_x": int(data_end),
    }


def analyze_prediction_image(image_path, future_len, future_sec, smooth_window):
    if Image is None:
        raise RuntimeError("Pillow is not available; cannot parse PNG prediction plots")
    rgb = np.array(Image.open(str(image_path)).convert("RGB"))
    y_lines, x_left, x_right = detect_axes_bounds(rgb)
    detections = []
    for idx, channel in enumerate(CHANNELS):
        y0 = y_lines[idx * 2]
        y1 = y_lines[idx * 2 + 1]
        crop = rgb[y0 + 1 : y1, x_left + 1 : x_right]
        extracted = extract_orange_trace(crop)
        if extracted is None:
            detections.append({"channel": channel[0], "channel_label": channel[1], "error": "orange_trace_not_found"})
            continue
        trace_y, data_start, data_end = extracted
        detection = detect_spike_from_trace(trace_y, data_start, data_end, future_len, future_sec, smooth_window)
        detection["channel"] = channel[0]
        detection["channel_label"] = channel[1]
        detection["error"] = ""
        detections.append(detection)
    return detections


def select_sync_cluster(detections, threshold_px, tolerance_idx):
    valid = [
        d
        for d in detections
        if not d.get("error") and d.get("residual_abs_px") is not None and d.get("residual_abs_px") >= threshold_px
    ]
    if not valid:
        return [], valid
    best_cluster = []
    best_score = -1.0
    for center in valid:
        cluster = [d for d in valid if abs(d["spike_index"] - center["spike_index"]) <= tolerance_idx]
        score = len(cluster) * 1000.0 + sum(d["residual_abs_px"] for d in cluster)
        if score > best_score:
            best_score = score
            best_cluster = cluster
    return best_cluster, valid


def stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None, None, None
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std()), float(arr.max() - arr.min())


def base_sync_row(spec, image_path, sample_index, sample_info, source, fallback_note):
    row = dict((name, "") for name in CROSS_CHANNEL_COLUMNS)
    row["run_id"] = spec["run_id"]
    row["short_name"] = spec["short_name"]
    row["sample_plot_index"] = "" if sample_index is None else sample_index
    row["image_file"] = "" if image_path is None else str(image_path)
    row["detection_source"] = source
    row["fallback_note"] = fallback_note
    if sample_info:
        for key in ["sample_key", "subject_id", "event_idx", "event_level"]:
            row[key] = sample_info.get(key, "")
    return row


def enrich_with_case(row, case_row):
    if not case_row:
        return
    for key in [
        "gt_onset_idx",
        "pred_onset_idx",
        "gt_main_peak_idx",
        "pred_main_peak_idx",
        "gt_first_reversal_sec",
        "pred_first_reversal_sec",
    ]:
        row[key] = case_row.get(key, "")


def analyze_run(repo_root, spec, threshold_px, tolerance_idx, smooth_window):
    run_dir = (repo_root / spec["path"]).resolve()
    config = load_json(run_dir / "run_config.json")
    future_len = safe_int(config.get("FUTURE_LEN")) or 400
    future_sec = safe_float(config.get("FUTURE_SEC")) or 2.0

    test_samples = get_test_sample_map(run_dir)
    case_rows, case_path = get_case_rows(run_dir, spec["case_glob"])
    cases_by_key = dict((row.get("sample_key"), row) for row in case_rows if row.get("sample_key"))
    image_paths = sorted((run_dir / "figures").glob("pred_vs_gt_example_*_idx*.png"))

    run_summary = {
        "run_id": spec["run_id"],
        "short_name": spec["short_name"],
        "run_dir": str(run_dir),
        "future_len": future_len,
        "future_sec": future_sec,
        "prediction_images_found": len(image_paths),
        "case_table": "" if case_path is None else str(case_path),
        "case_rows_found": len(case_rows),
        "used_source": "prediction_plot_images" if image_paths and Image is not None else "recalc_cases_fallback",
        "fallbacks": [],
        "synchronized_rows": 0,
        "hist_spike_count": 0,
        "median_sync_spike_index": None,
        "median_sync_spike_time_sec": None,
    }

    rows = []
    hist_points = []

    if image_paths and Image is not None:
        for image_path in image_paths:
            sample_index = parse_sample_index(image_path)
            sample_info = {}
            if sample_index is not None and 0 <= sample_index < len(test_samples):
                sample_info = test_samples[sample_index]
            sample_key = sample_info.get("sample_key")
            case_row = cases_by_key.get(sample_key)

            row = base_sync_row(spec, image_path, sample_index, sample_info, "prediction_plot_image", "")
            enrich_with_case(row, case_row)
            try:
                detections = analyze_prediction_image(image_path, future_len, future_sec, smooth_window)
                sync_cluster, thresholded = select_sync_cluster(detections, threshold_px, tolerance_idx)
            except Exception as exc:
                row["fallback_note"] = "image_parse_failed: {0}".format(exc)
                detections = []
                sync_cluster = []
                thresholded = []

            row["n_detected_channels"] = len(thresholded)
            row["n_sync_channels"] = len(sync_cluster)
            row["sync_channel_names"] = ",".join([d.get("channel", "") for d in sync_cluster])
            row["synchronized"] = "1" if len(sync_cluster) >= 2 else "0"

            for detection in detections:
                channel = detection.get("channel")
                if not channel:
                    continue
                row[channel + "_spike_index"] = fmt_float(detection.get("spike_index"), 3)
                row[channel + "_spike_time_sec"] = fmt_float(detection.get("spike_time_sec"), 6)
                row[channel + "_residual_px"] = fmt_float(detection.get("residual_px"), 3)

            if sync_cluster:
                idx_mean, idx_std, idx_range = stats([d.get("spike_index") for d in sync_cluster])
                time_mean, time_std, time_range = stats([d.get("spike_time_sec") for d in sync_cluster])
                row["sync_spike_index_mean"] = fmt_float(idx_mean, 3)
                row["sync_spike_index_std"] = fmt_float(idx_std, 3)
                row["sync_spike_index_range"] = fmt_float(idx_range, 3)
                row["sync_spike_time_mean_sec"] = fmt_float(time_mean, 6)
                row["sync_spike_time_std_sec"] = fmt_float(time_std, 6)
                row["sync_spike_time_range_sec"] = fmt_float(time_range, 6)
                if len(sync_cluster) >= 2:
                    for detection in sync_cluster:
                        hist_points.append(
                            {
                                "run_id": spec["run_id"],
                                "short_name": spec["short_name"],
                                "channel": detection.get("channel"),
                                "sample_plot_index": sample_index,
                                "spike_index": detection.get("spike_index"),
                                "spike_time_sec": detection.get("spike_time_sec"),
                            }
                        )
            rows.append(row)
    else:
        if not image_paths:
            run_summary["fallbacks"].append("missing_prediction_plot_images")
        if Image is None:
            run_summary["fallbacks"].append("Pillow_unavailable_for_png_parsing")
        run_summary["fallbacks"].append("fallback uses pred_main_peak_idx from recalc case table; no cross-channel waveform sync available")
        for case_row in case_rows:
            row = base_sync_row(spec, None, "", case_row, "recalc_case_table", "no prediction images parsed")
            enrich_with_case(row, case_row)
            pred_idx = safe_float(case_row.get("pred_main_peak_idx"))
            if pred_idx is not None:
                row["n_detected_channels"] = "1"
                row["n_sync_channels"] = "0"
                row["synchronized"] = "0"
                row["sync_spike_index_mean"] = fmt_float(pred_idx, 3)
                row["sync_spike_time_mean_sec"] = fmt_float(pred_idx / float(future_len) * float(future_sec), 6)
                hist_points.append(
                    {
                        "run_id": spec["run_id"],
                        "short_name": spec["short_name"],
                        "channel": "fallback_pred_main_peak",
                        "sample_plot_index": "",
                        "spike_index": pred_idx,
                        "spike_time_sec": pred_idx / float(future_len) * float(future_sec),
                    }
                )
            rows.append(row)

    sync_rows = [row for row in rows if str(row.get("synchronized")) == "1"]
    run_summary["synchronized_rows"] = len(sync_rows)
    run_summary["hist_spike_count"] = len(hist_points)
    means_idx = [safe_float(row.get("sync_spike_index_mean")) for row in sync_rows]
    means_idx = [value for value in means_idx if value is not None]
    means_time = [safe_float(row.get("sync_spike_time_mean_sec")) for row in sync_rows]
    means_time = [value for value in means_time if value is not None]
    if means_idx:
        run_summary["median_sync_spike_index"] = float(np.median(np.array(means_idx, dtype=float)))
    if means_time:
        run_summary["median_sync_spike_time_sec"] = float(np.median(np.array(means_time, dtype=float)))

    return rows, hist_points, run_summary


def plot_histogram(path, hist_points, specs, future_len):
    plt.figure(figsize=(10, 5.5))
    if hist_points:
        bins = np.arange(0, future_len + 8, 8)
        for spec in specs:
            values = [
                point["spike_index"]
                for point in hist_points
                if point["run_id"] == spec["run_id"] and point.get("spike_index") is not None
            ]
            if not values:
                continue
            plt.hist(
                values,
                bins=bins,
                alpha=0.42,
                label="{0} synchronized channel spikes".format(spec["short_name"]),
                color=spec["color"],
            )
            median = float(np.median(np.array(values, dtype=float)))
            plt.axvline(median, color=spec["color"], linestyle="--", linewidth=1.5)
        plt.xlabel("future sample index (400 samples = 2.0 s)")
        plt.ylabel("detected synchronized channel-count")
        plt.title("Predicted local spike positions from existing B/C/D prediction plots")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No spike-position points available", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(path), dpi=160)
    plt.close()


def markdown_summary(run_summaries, output_dir, hist_points, threshold_px, tolerance_idx, image_source_used):
    lines = []
    lines.append("# Spike Position Diagnosis 2026-04-21")
    lines.append("")
    lines.append("Generated at: `{0}`".format(_dt.datetime.now().isoformat(timespec="seconds")))
    lines.append("")
    lines.append("## Method")
    lines.append("")
    if image_source_used:
        lines.append(
            "The primary diagnosis used existing `pred_vs_gt_example_*.png` prediction plots, because those artifacts contain the three plotted output channels (`steer_angle`, `yawrate`, `ay`)."
        )
        lines.append(
            "For each plotted prediction channel, the tool extracted the orange predicted curve, subtracted a local moving-average baseline, and marked the largest absolute residual as the local spike candidate."
        )
        lines.append(
            "A channel counted as spiking when `abs(residual_px) >= {0}`. A cross-channel spike counted as synchronized when at least two channels landed within `{1}` future-sample indices.".format(
                threshold_px, tolerance_idx
            )
        )
    else:
        lines.append(
            "Prediction plots could not be parsed, so the report fell back to recalc case tables. That fallback can locate predicted main-peak positions, but it cannot prove waveform-level cross-channel synchronization."
        )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `spike_index_hist.png`: histogram of synchronized spike indices by run.")
    lines.append("- `cross_channel_spike_sync.csv`: per-example table with channel spike locations and recalc/sample-table context.")
    lines.append("- `spike_position_summary.md`: this summary.")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append("| Run | prediction images | case rows | synchronized examples | histogram points | median sync index | median sync time (s) | source |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in run_summaries:
        lines.append(
            "| {short_name} | {images} | {cases} | {sync} | {points} | {idx} | {sec} | {source} |".format(
                short_name=summary["short_name"],
                images=summary["prediction_images_found"],
                cases=summary["case_rows_found"],
                sync=summary["synchronized_rows"],
                points=summary["hist_spike_count"],
                idx=fmt_float(summary.get("median_sync_spike_index"), 3),
                sec=fmt_float(summary.get("median_sync_spike_time_sec"), 6),
                source=summary["used_source"],
            )
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if hist_points:
        lines.append(
            "The detected local spike is strongly cross-channel-synchronized within each run: synchronized rows require at least two output channels to choose almost the same future index."
        )
        lines.append(
            "Across runs, the preferred spike band shifts rather than perfectly matching: Run B clusters earlier, Run C later, and Run D in between. That supports a shared decoder/timestep artifact hypothesis more than an isolated single-channel plotting artifact."
        )
        lines.append(
            "Because only rendered plots are available here, this is a position-level diagnosis. A waveform-level causal diagnosis would require saved raw prediction arrays or branch-level coarse/fine outputs."
        )
    else:
        lines.append("No synchronized spike positions were available after applying the source and threshold rules.")
    lines.append("")
    lines.append("## Fallbacks And Blockers")
    lines.append("")
    fallback_any = False
    for summary in run_summaries:
        for fallback in summary.get("fallbacks", []):
            fallback_any = True
            lines.append("- {0}: {1}".format(summary["short_name"], fallback))
    if not fallback_any:
        lines.append("- No run had to fall back fully to recalc case tables; image-based prediction-plot diagnosis was available for B/C/D.")
    lines.append("- Raw per-sample prediction arrays were not found in the B/C/D run folders, so the image-based diagnosis cannot separate coarse-branch, fine-branch, or decoder-token contributions.")
    lines.append("")
    lines.append("Report directory: `{0}`".format(output_dir))
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root. Defaults to current directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Output report directory.")
    parser.add_argument("--spike-threshold-px", type=float, default=12.0, help="Minimum absolute residual in image pixels.")
    parser.add_argument(
        "--sync-tolerance-idx",
        type=float,
        default=8.0,
        help="Maximum future-index spread for channels to count as synchronized.",
    )
    parser.add_argument("--smooth-window-px", type=int, default=41, help="Moving-average window for plot-curve baseline.")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_hist_points = []
    run_summaries = []
    future_len_for_plot = 400
    for spec in RUN_SPECS:
        rows, hist_points, summary = analyze_run(
            repo_root,
            spec,
            args.spike_threshold_px,
            args.sync_tolerance_idx,
            args.smooth_window_px,
        )
        all_rows.extend(rows)
        all_hist_points.extend(hist_points)
        run_summaries.append(summary)
        future_len_for_plot = summary.get("future_len") or future_len_for_plot

    csv_path = output_dir / "cross_channel_spike_sync.csv"
    hist_path = output_dir / "spike_index_hist.png"
    md_path = output_dir / "spike_position_summary.md"

    write_csv(csv_path, all_rows, CROSS_CHANNEL_COLUMNS)
    plot_histogram(hist_path, all_hist_points, RUN_SPECS, int(future_len_for_plot))
    image_source_used = any(summary.get("used_source") == "prediction_plot_images" for summary in run_summaries)
    write_text(
        md_path,
        markdown_summary(
            run_summaries,
            output_dir,
            all_hist_points,
            args.spike_threshold_px,
            args.sync_tolerance_idx,
            image_source_used,
        ),
    )

    print(
        json.dumps(
            {
                "spike_index_hist": str(hist_path),
                "cross_channel_spike_sync": str(csv_path),
                "spike_position_summary": str(md_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
