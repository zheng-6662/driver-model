from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_SCRIPT_PATH = REPO_ROOT / "02_code" / "final_code" / "model" / "training" / "future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "04_project_logs" / "reports" / "trigger_response_lag_20260421"


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def find_events_index_v2(data_root: Path) -> Path | None:
    candidate = data_root.parent / "Event_Dataset_v2" / "events_index.csv"
    if candidate.exists():
        return candidate
    return None


def resolve_event_files(vehicle_file: Path) -> dict[str, Path | None]:
    subject_dir = vehicle_file.parent.parent
    event_dir = subject_dir / "event"
    stem = vehicle_file.name.replace("_vehicle_aligned_cleaned.csv", "")
    v400 = event_dir / f"{stem}_vehicle_aligned_cleaned_events_v400_context.csv"
    v312 = event_dir / f"{stem}_vehicle_aligned_cleaned_events_v312.csv"
    return {
        "v400": v400 if v400.exists() else None,
        "v312": v312 if v312.exists() else None,
    }


def resolve_data_root(training_module) -> Path:
    root = Path(str(training_module.ROOT))
    if root.exists():
        return root
    matches = list((REPO_ROOT / "01_datasets").rglob("被试数据集合"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Unable to resolve vehicle data root from {root}")


def resolve_vehicle_columns(training_module, df_v: pd.DataFrame) -> dict[str, str | None]:
    cols = df_v.columns.tolist()
    return {
        "roll": training_module.find_col(cols, ["zx|roll", "roll", "Roll"]),
        "steer": training_module.find_col(cols, ["zx|SteeringWheel", "SteeringWheel", "steer"]),
        "yawrate": training_module.find_col(cols, ["vyaw", "zx|vyaw", "YawRate", "zx|YawRate", "yaw_rate"]),
        "ay": training_module.find_col(cols, ["zx|ay", "ay", "Ay", "lat_acc"]),
        "curve": training_module.find_col(cols, ["zx1|lanecurvatureXY", "laneCurvature", "lanecurvature_start"]),
        "road_type_fixed": training_module.find_col(cols, ["road_type_fixed", "road_type", "roadType_fixed"]),
        "ref_nn_ok": training_module.find_col(cols, ["ref_nn_ok", "ref_ok", "refnn_ok"]),
    }


def choose_is_curve(training_module, df_v: pd.DataFrame, cols: dict[str, str | None], i0: int, i1: int, curve_seg_mean: float) -> bool:
    is_curve = None
    col_roadtype = cols["road_type_fixed"]
    col_refok = cols["ref_nn_ok"]
    if col_roadtype is not None and col_refok is not None:
        ok_seg = pd.to_numeric(df_v[col_refok], errors="coerce").to_numpy(dtype=np.float32, copy=False)[i0:i1]
        ok_ratio = float(np.nanmean(ok_seg > 0.5)) if ok_seg.size else 0.0
        if ok_ratio >= float(training_module.ROAD_OK_RATIO_THR):
            rt_seg = df_v[col_roadtype].to_numpy(copy=False)[i0:i1]
            if rt_seg.dtype.kind in ("i", "u", "f"):
                is_curve = float(np.nanmean(rt_seg)) >= 0.5
            else:
                rt_low = np.char.lower(rt_seg.astype(str))
                is_curve = float(np.mean(rt_low == "curve")) >= 0.5
    if is_curve is None:
        is_curve = curve_seg_mean > float(training_module.CURVE_THR_FOR_ANCHOR)
    return bool(is_curve)


def compute_anchor_idx(training_module, steer_rate: np.ndarray, roll: np.ndarray, curve: np.ndarray, df_v: pd.DataFrame, cols: dict[str, str | None], i0: int, i1: int) -> tuple[int | None, bool, float, str]:
    curve_seg = curve[i0:i1]
    curve_seg_mean = float(np.nanmean(np.abs(curve_seg))) if curve_seg.size else 0.0
    is_curve = choose_is_curve(training_module, df_v, cols, i0, i1, curve_seg_mean)
    if is_curve:
        roll_seg = roll[i0:i1]
        if roll_seg.size == 0:
            return None, is_curve, curve_seg_mean, "roll_peak"
        peak_rel = int(np.argmax(np.abs(roll_seg)))
        return i0 + peak_rel, is_curve, curve_seg_mean, "roll_peak"

    sr_seg = steer_rate[i0:i1]
    if sr_seg.size == 0:
        return None, is_curve, curve_seg_mean, "steer_rate_peak80_first"
    abs_sr = np.abs(sr_seg)
    max_abs = float(np.nanmax(abs_sr))
    if (not np.isfinite(max_abs)) or max_abs < 1e-6:
        roll_seg = roll[i0:i1]
        if roll_seg.size == 0:
            return None, is_curve, curve_seg_mean, "roll_peak_fallback"
        peak_rel = int(np.argmax(np.abs(roll_seg)))
        return i0 + peak_rel, is_curve, curve_seg_mean, "roll_peak_fallback"

    thr = float(training_module.STEER_RATE_PEAK_FRAC) * max_abs
    cand = np.where(abs_sr >= thr)[0]
    peak_rel = int(cand[0]) if cand.size else int(np.argmax(abs_sr))
    return i0 + peak_rel, is_curve, curve_seg_mean, "steer_rate_peak80_first"


def compute_onset_idx(training_module, steer: np.ndarray, trigger_idx: int, end_idx: int) -> tuple[int | None, float]:
    search_end = min(len(steer), max(end_idx, trigger_idx + int(training_module.FUTURE_LEN)))
    if search_end <= trigger_idx + 1:
        return None, float(training_module.STEER_ONSET_THR_ABS)
    seq = np.asarray(steer[trigger_idx:search_end], dtype=np.float64)
    base = float(seq[0])
    true_peak_delta = float(np.max(np.abs(seq - base))) if seq.size else 0.0
    onset_thr = max(float(training_module.STEER_ONSET_THR_ABS), 0.15 * true_peak_delta)
    onset_rel = training_module._first_threshold_crossing_idx_np(seq, threshold=onset_thr, ref_value=base)
    if onset_rel is None:
        return None, float(onset_thr)
    return int(trigger_idx + onset_rel), float(onset_thr)


def summarize_lags(df: pd.DataFrame) -> dict[str, Any]:
    def make_bucket(bucket_df: pd.DataFrame) -> dict[str, Any]:
        values = bucket_df["trigger_to_onset_lag_sec"].dropna().to_numpy(dtype=np.float64)
        payload: dict[str, Any] = {
            "event_count": int(len(bucket_df)),
            "with_onset_count": int(values.size),
            "missing_onset_count": int(len(bucket_df) - values.size),
        }
        if values.size:
            payload.update(
                {
                    "lag_mean_sec": float(np.mean(values)),
                    "lag_median_sec": float(np.median(values)),
                    "lag_p10_sec": float(np.percentile(values, 10)),
                    "lag_p90_sec": float(np.percentile(values, 90)),
                    "anchor_mean_sec": float(np.mean(bucket_df["trigger_to_anchor_lag_sec"].dropna().to_numpy(dtype=np.float64))),
                }
            )
        else:
            payload.update(
                {
                    "lag_mean_sec": None,
                    "lag_median_sec": None,
                    "lag_p10_sec": None,
                    "lag_p90_sec": None,
                    "anchor_mean_sec": None,
                }
            )
        return payload

    summary = {
        "overall": make_bucket(df),
        "by_is_curve_applied": {},
    }
    for key, bucket_df in df.groupby("is_curve_applied", dropna=False):
        summary["by_is_curve_applied"][str(int(key)) if pd.notna(key) else "nan"] = make_bucket(bucket_df)
    return summary


def build_task_definition_markdown(protocol_split_enabled: bool, protocol_reason: str, counts: dict[str, Any]) -> str:
    protocol_line = (
        "Enabled via unambiguous Event_Dataset_v2 join."
        if protocol_split_enabled
        else f"Not enabled: {protocol_reason}."
    )
    return "\n".join(
        [
            "# Task Definition And Event Logic",
            "",
            "## Current Task Framing",
            "- The task stays pooled post-trigger steering response prediction.",
            "- This diagnostic measures how far the actual steering response onset lags behind the event trigger marker.",
            "- Anchor logic matches the active training script and is reported as supporting context, not as a replacement target.",
            "",
            "## Event Source Priority",
            "- Prefer `*_events_v400_context.csv` for `road_type_anchor`, `curvature_anchor`, `trigger_type`, `phase_type`, and `trigger_idx`.",
            "- Fall back to `*_events_v312.csv` when the v400 context file is missing.",
            "",
            "## Anchor Logic Reused From Active Script",
            "- Curve: `roll_peak` over the event segment.",
            "- Straight: first `|steer_rate| >= 0.8 * max_abs(steer_rate)` within the event segment.",
            "",
            "## Onset Logic Reused From Active Script",
            "- Helper: `_first_threshold_crossing_idx_np`.",
            "- Absolute threshold floor: `STEER_ONSET_THR_ABS`.",
            "- Final threshold: `max(STEER_ONSET_THR_ABS, 0.15 * true_peak_delta)` measured on the trigger-to-response search window.",
            "",
            "## Protocol Split",
            f"- {protocol_line}",
            "",
            "## Coverage",
            f"- Vehicle files scanned: `{counts['vehicle_file_count']}`",
            f"- Vehicle files with usable paired events: `{counts['paired_event_vehicle_count']}`",
            f"- Missing v312 basenames: `{counts['missing_v312_count']}`",
            f"- Missing v400 basenames: `{counts['missing_v400_count']}`",
            f"- Strong events analyzed: `{counts['strong_event_count']}`",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze trigger-to-onset response lag with the active-script event logic.")
    parser.add_argument("--script-path", default=str(ACTIVE_SCRIPT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_module = load_module(Path(args.script_path).resolve(), "trigger_lag_training_module")
    data_root = resolve_data_root(training_module)

    events_index_v2 = find_events_index_v2(data_root)
    protocol_split_enabled = False
    protocol_reason = "Event_Dataset_v2/events_index.csv not found"
    if events_index_v2 is not None:
        protocol_reason = "join logic intentionally disabled because no unambiguous mapping rule was approved"

    rows: list[dict[str, Any]] = []
    missing_v312: list[str] = []
    missing_v400: list[str] = []

    vehicle_files = sorted(data_root.glob("*/vehicle/*_vehicle_aligned_cleaned.csv"))
    for vehicle_file in vehicle_files:
        event_paths = resolve_event_files(vehicle_file)
        stem = vehicle_file.name.replace("_vehicle_aligned_cleaned.csv", "")
        if event_paths["v312"] is None:
            missing_v312.append(stem)
        if event_paths["v400"] is None:
            missing_v400.append(stem)

        chosen_event_path = event_paths["v400"] or event_paths["v312"]
        if chosen_event_path is None:
            continue

        df_v = pd.read_csv(vehicle_file)
        df_e = pd.read_csv(chosen_event_path)
        df_e = df_e[df_e["event_level"].isin(training_module.STRONG_LABELS)].copy()
        if df_e.empty:
            continue

        cols = resolve_vehicle_columns(training_module, df_v)
        if any(cols[key] is None for key in ("roll", "steer", "yawrate", "ay", "curve")):
            continue

        steer = training_module.steer_array_from_rad(pd.to_numeric(df_v[cols["steer"]], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
        roll = pd.to_numeric(df_v[cols["roll"]], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        curve = pd.to_numeric(df_v[cols["curve"]], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        steer_rate = np.gradient(steer, 1.0 / float(training_module.FS)).astype(np.float32)

        for event_idx, event_row in df_e.iterrows():
            start_s = float(event_row["start_s"])
            end_s = float(event_row["end_s"])
            start_idx = int(event_row["start_idx"]) if "start_idx" in event_row and pd.notna(event_row["start_idx"]) else int(start_s * training_module.FS)
            end_idx = int(event_row["end_idx"]) if "end_idx" in event_row and pd.notna(event_row["end_idx"]) else int(end_s * training_module.FS)
            start_idx = max(0, start_idx)
            end_idx = min(len(df_v), end_idx)
            if end_idx - start_idx < 10:
                continue

            anchor_idx, is_curve, curve_seg_mean, anchor_source = compute_anchor_idx(
                training_module,
                steer_rate=steer_rate,
                roll=roll,
                curve=curve,
                df_v=df_v,
                cols=cols,
                i0=start_idx,
                i1=end_idx,
            )
            if anchor_idx is None:
                continue

            if "trigger_idx" in event_row and pd.notna(event_row["trigger_idx"]):
                trigger_idx = int(event_row["trigger_idx"])
                trigger_idx_source = "trigger_idx"
            else:
                trigger_idx = start_idx
                trigger_idx_source = "event_start"
            trigger_idx = max(0, min(trigger_idx, len(df_v) - 1))

            onset_idx, onset_thr = compute_onset_idx(training_module, steer=steer, trigger_idx=trigger_idx, end_idx=end_idx)
            rows.append(
                {
                    "subject_id": vehicle_file.parent.parent.name,
                    "vehicle_file": str(vehicle_file),
                    "event_file": str(chosen_event_path),
                    "event_version": "v400_context" if chosen_event_path == event_paths["v400"] else "v312",
                    "event_idx": int(event_idx),
                    "event_level": str(event_row.get("event_level", "")),
                    "phase_type": None if pd.isna(event_row.get("phase_type")) else str(event_row.get("phase_type")),
                    "trigger_type": None if pd.isna(event_row.get("trigger_type")) else str(event_row.get("trigger_type")),
                    "road_type_anchor": None if pd.isna(event_row.get("road_type_anchor")) else str(event_row.get("road_type_anchor")),
                    "curvature_anchor": None if pd.isna(event_row.get("curvature_anchor")) else float(event_row.get("curvature_anchor")),
                    "start_s": start_s,
                    "end_s": end_s,
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "trigger_idx": int(trigger_idx),
                    "trigger_time_s": float(trigger_idx / training_module.FS),
                    "trigger_idx_source": trigger_idx_source,
                    "anchor_idx": int(anchor_idx),
                    "anchor_time_s": float(anchor_idx / training_module.FS),
                    "anchor_source_applied": anchor_source,
                    "maintained_anchor_policy": "curve->roll_peak; straight->steer_rate_peak80_first",
                    "is_curve_applied": int(bool(is_curve)),
                    "curve_score_event_mean_abs": float(curve_seg_mean),
                    "steer_onset_threshold": float(onset_thr),
                    "onset_idx": None if onset_idx is None else int(onset_idx),
                    "onset_time_s": None if onset_idx is None else float(onset_idx / training_module.FS),
                    "trigger_to_onset_lag_sec": None if onset_idx is None else float((onset_idx - trigger_idx) / training_module.FS),
                    "trigger_to_anchor_lag_sec": float((anchor_idx - trigger_idx) / training_module.FS),
                    "protocol_label": None,
                }
            )

    lag_df = pd.DataFrame(rows)
    lag_csv_path = output_dir / "trigger_to_onset_lag.csv"
    lag_df.to_csv(lag_csv_path, index=False, encoding="utf-8-sig")

    hist_path = output_dir / "trigger_to_onset_hist.png"
    plt.figure(figsize=(9, 5))
    for label, bucket_df in lag_df.groupby("is_curve_applied", dropna=False):
        values = bucket_df["trigger_to_onset_lag_sec"].dropna().to_numpy(dtype=np.float64)
        if values.size == 0:
            continue
        curve_label = "curve" if int(label) == 1 else "straight"
        plt.hist(values, bins=30, alpha=0.55, label=curve_label)
    plt.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    plt.xlabel("trigger_to_onset_lag_sec")
    plt.ylabel("event_count")
    plt.title("Trigger to onset lag (current active-script onset logic)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path, dpi=180)
    plt.close()

    counts = {
        "vehicle_file_count": int(len(vehicle_files)),
        "paired_event_vehicle_count": int(lag_df["vehicle_file"].nunique()),
        "missing_v312_count": int(len(missing_v312)),
        "missing_v400_count": int(len(missing_v400)),
        "missing_v312_basenames": missing_v312,
        "missing_v400_basenames": missing_v400,
        "strong_event_count": int(len(lag_df)),
    }
    summary_payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "script_path": str(Path(args.script_path).resolve()),
        "data_root": str(data_root),
        "protocol_split_enabled": protocol_split_enabled,
        "protocol_split_reason": protocol_reason,
        "counts": counts,
        "lag_summary": summarize_lags(lag_df),
        "outputs": {
            "trigger_to_onset_lag_csv": str(lag_csv_path),
            "trigger_to_onset_hist_png": str(hist_path),
            "task_definition_md": str(output_dir / "TASK_DEFINITION_AND_EVENT_LOGIC.md"),
        },
    }
    summary_json_path = output_dir / "trigger_to_onset_summary.json"
    save_json(summary_json_path, summary_payload)

    task_definition_path = output_dir / "TASK_DEFINITION_AND_EVENT_LOGIC.md"
    task_definition_path.write_text(
        build_task_definition_markdown(protocol_split_enabled, protocol_reason, counts),
        encoding="utf-8",
    )

    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
