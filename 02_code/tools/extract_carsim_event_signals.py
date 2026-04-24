#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract event-level vehicle signals for selected CarSim samples.

Outputs:
1. One metadata CSV describing each extracted sample and its event timing.
2. One signal CSV per sample containing:
   - absolute time
   - relative time to event start / anchor
   - steering wheel
   - roll / pitch / yaw
   - optional extra motion signals when available
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SELECTION = Path(
    r"F:\data_set_process\data_process\04_project_logs\reports\carsim_sample_selection\round4_stage2_refine_rmse_2s_abs_steer_summary_top10.csv"
)
DEFAULT_MANIFEST = Path(
    r"F:\data_set_process\data_process\04_project_logs\reports\trajectory_foundation_audit_20260325\task_01_manifest_audit\unified_manifest_v1_draft.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"F:\data_set_process\data_process\04_project_logs\reports\carsim_event_signal_exports"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract event signal segments for selected CarSim samples.")
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pre-seconds",
        type=float,
        default=1.0,
        help="Extra context to include before event start.",
    )
    parser.add_argument(
        "--post-seconds",
        type=float,
        default=1.0,
        help="Extra context to include after event end.",
    )
    parser.add_argument(
        "--dedupe-sample-keys",
        action="store_true",
        help="Keep one export per unique sample_key. Recommended for summary files that contain both good and bad buckets.",
    )
    return parser.parse_args()


def normalize_sample_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "selection_bucket" not in out.columns:
        out["selection_bucket"] = "selected"
    return out


def load_manifest(manifest_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    required = {"sample_key_unified", "episode_id", "phase_type", "anchor_idx"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Manifest missing required columns: {sorted(missing)}")

    cols = [
        c for c in [
            "sample_key_unified",
            "event_idx",
            "episode_id",
            "phase_type",
            "anchor_idx",
            "protocol_version",
            "morphology_label",
            "mechanism_tag",
        ] if c in df.columns
    ]
    out = df[cols].drop_duplicates().copy()
    out = out.rename(columns={"sample_key_unified": "sample_key"})
    out["episode_id"] = pd.to_numeric(out["episode_id"], errors="coerce")
    out["anchor_idx"] = pd.to_numeric(out["anchor_idx"], errors="coerce")
    out = out.drop_duplicates(subset=["sample_key"], keep="first")
    return out


def find_first_col(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for cand in candidates:
        cand_l = cand.lower()
        for col in columns:
            if cand_l == col.lower():
                return col
    return None


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def select_event_row(events_df: pd.DataFrame, episode_id: float, phase_type: str, anchor_time_s: float | None) -> pd.Series:
    all_events = events_df.copy()

    if anchor_time_s is not None and np.isfinite(anchor_time_s) and {"start_s", "end_s"} <= set(all_events.columns):
        contained = all_events[(all_events["start_s"] <= anchor_time_s) & (all_events["end_s"] >= anchor_time_s)].copy()
        if not contained.empty:
            if "phase_type" in contained.columns and isinstance(phase_type, str) and phase_type:
                phase_contained = contained[contained["phase_type"].astype(str) == phase_type]
                if not phase_contained.empty:
                    return phase_contained.iloc[0]
            return contained.iloc[0]

    cand = all_events.copy()
    if "episode_id" in cand.columns and not pd.isna(episode_id):
        cand = cand[pd.to_numeric(cand["episode_id"], errors="coerce") == float(episode_id)]
    if "phase_type" in cand.columns and isinstance(phase_type, str) and phase_type:
        cand = cand[cand["phase_type"].astype(str) == phase_type]
    if not cand.empty:
        if anchor_time_s is not None and np.isfinite(anchor_time_s) and {"start_s", "end_s"} <= set(cand.columns):
            center_dist = ((cand["start_s"] + cand["end_s"]) / 2.0 - anchor_time_s).abs()
            return cand.iloc[int(center_dist.argmin())]
        return cand.iloc[0]

    if anchor_time_s is not None and np.isfinite(anchor_time_s) and {"start_s", "end_s"} <= set(all_events.columns):
        center_dist = ((all_events["start_s"] + all_events["end_s"]) / 2.0 - anchor_time_s).abs()
        return all_events.iloc[int(center_dist.argmin())]

    raise ValueError("No matching event row after anchor/episode/phase matching.")


def build_signal_export(vehicle_df: pd.DataFrame, event_row: pd.Series, anchor_idx: float | None, pre_seconds: float, post_seconds: float) -> tuple[pd.DataFrame, dict]:
    cols = vehicle_df.columns.tolist()
    t_col = find_first_col(cols, ["t_s", "StorageTime", "time", "Time"])
    steer_col = find_first_col(cols, ["zx|SteeringWheel", "SteeringWheel", "steer", "steeringwheel"])
    roll_col = find_first_col(cols, ["zx|roll", "roll"])
    pitch_col = find_first_col(cols, ["zx|pitch", "pitch"])
    yaw_col = find_first_col(cols, ["zx|yaw", "yaw"])
    vx_col = find_first_col(cols, ["zx|vx", "vx", "speed"])
    ay_col = find_first_col(cols, ["zx|ay", "ay"])
    yaw_rate_col = find_first_col(cols, ["zx|vyaw", "vyaw", "yaw_rate"])

    required = {"time": t_col, "steering_wheel": steer_col, "roll": roll_col, "pitch": pitch_col, "yaw": yaw_col}
    missing_required = [name for name, col in required.items() if col is None]
    if missing_required:
        raise KeyError(f"Vehicle file missing required signal columns: {missing_required}")

    start_s = float(event_row["start_s"])
    end_s = float(event_row["end_s"])
    clip_start = start_s - pre_seconds
    clip_end = end_s + post_seconds

    seg = vehicle_df[(vehicle_df[t_col] >= clip_start) & (vehicle_df[t_col] <= clip_end)].copy()
    if seg.empty:
        raise ValueError("No vehicle rows found inside requested event clip.")

    anchor_time_s = np.nan
    if anchor_idx is not None and np.isfinite(anchor_idx):
        idx = int(anchor_idx)
        if 0 <= idx < len(vehicle_df):
            anchor_time_s = float(vehicle_df.iloc[idx][t_col])

    export = pd.DataFrame(
        {
            "time_s": seg[t_col].astype(float),
            "time_rel_event_start_s": seg[t_col].astype(float) - start_s,
            "time_rel_event_end_s": seg[t_col].astype(float) - end_s,
            "steering_wheel": seg[steer_col].astype(float),
            "roll": seg[roll_col].astype(float),
            "pitch": seg[pitch_col].astype(float),
            "yaw": seg[yaw_col].astype(float),
        }
    )

    if np.isfinite(anchor_time_s):
        export["time_rel_anchor_s"] = export["time_s"] - anchor_time_s
    else:
        export["time_rel_anchor_s"] = np.nan

    optional_map = {
        "vx": vx_col,
        "ay": ay_col,
        "yaw_rate": yaw_rate_col,
    }
    for out_col, src_col in optional_map.items():
        if src_col is not None:
            export[out_col] = seg[src_col].astype(float)

    export["is_event_window"] = ((export["time_s"] >= start_s) & (export["time_s"] <= end_s)).astype(int)

    meta = {
        "event_start_s": start_s,
        "event_end_s": end_s,
        "event_duration_s": end_s - start_s,
        "clip_start_s": float(export["time_s"].min()),
        "clip_end_s": float(export["time_s"].max()),
        "anchor_time_s": None if not np.isfinite(anchor_time_s) else anchor_time_s,
        "row_count": int(len(export)),
    }
    return export, meta


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selection_df = normalize_sample_table(pd.read_csv(args.selection_csv))
    if args.dedupe_sample_keys:
        selection_df = selection_df.sort_values(["sample_key", "selection_bucket"]).drop_duplicates("sample_key", keep="first")

    manifest_df = load_manifest(args.manifest_csv)
    merged = selection_df.merge(manifest_df, how="left", on="sample_key", suffixes=("", "_manifest"))

    metadata_rows: list[dict] = []
    missing_rows: list[dict] = []

    for _, row in merged.iterrows():
        sample_key = str(row["sample_key"])
        try:
            vehicle_file = str(row["vehicle_file"])
            event_file = str(row["event_file"])
            vehicle_df = pd.read_csv(vehicle_file)
            events_df = pd.read_csv(event_file)

            time_col = find_first_col(vehicle_df.columns.tolist(), ["t_s", "StorageTime", "time", "Time"])
            if time_col is None:
                raise KeyError("Vehicle file missing time column.")

            anchor_idx = row.get("anchor_idx", np.nan)
            anchor_time_s = np.nan
            if pd.notna(anchor_idx):
                anchor_idx_int = int(anchor_idx)
                if 0 <= anchor_idx_int < len(vehicle_df):
                    anchor_time_s = float(vehicle_df.iloc[anchor_idx_int][time_col])

            event_row = select_event_row(
                events_df=events_df,
                episode_id=row.get("episode_id", np.nan),
                phase_type=str(row.get("phase_type", "")),
                anchor_time_s=anchor_time_s if np.isfinite(anchor_time_s) else None,
            )
            export_df, event_meta = build_signal_export(
                vehicle_df=vehicle_df,
                event_row=event_row,
                anchor_idx=anchor_idx,
                pre_seconds=args.pre_seconds,
                post_seconds=args.post_seconds,
            )

            sample_dir = args.output_dir / sanitize_name(sample_key)
            sample_dir.mkdir(parents=True, exist_ok=True)
            signal_csv = sample_dir / "signals.csv"
            meta_json = sample_dir / "meta.json"

            export_df.to_csv(signal_csv, index=False, encoding="utf-8-sig")

            meta_payload = {
                "sample_key": sample_key,
                "selection_bucket": row.get("selection_bucket"),
                "vehicle_file": vehicle_file,
                "event_file": event_file,
                "episode_id": None if pd.isna(row.get("episode_id", np.nan)) else int(row["episode_id"]),
                "phase_type": row.get("phase_type"),
                "event_idx": None if pd.isna(row.get("event_idx", np.nan)) else int(row["event_idx"]),
                "anchor_idx": None if pd.isna(anchor_idx) else int(anchor_idx),
                "protocol_version": row.get("protocol_version"),
                "metric_mean": None if pd.isna(row.get("metric_mean", np.nan)) else float(row["metric_mean"]),
                "anchor_roll_from_selection": None if pd.isna(row.get("anchor_roll", np.nan)) else float(row["anchor_roll"]),
                **event_meta,
            }
            meta_json.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            metadata_rows.append(
                {
                    "sample_key": sample_key,
                    "selection_bucket": row.get("selection_bucket"),
                    "metric_mean": row.get("metric_mean"),
                    "episode_id": row.get("episode_id"),
                    "phase_type": row.get("phase_type"),
                    "event_idx": row.get("event_idx"),
                    "anchor_idx": row.get("anchor_idx"),
                    "event_start_s": event_meta["event_start_s"],
                    "event_end_s": event_meta["event_end_s"],
                    "event_duration_s": event_meta["event_duration_s"],
                    "anchor_time_s": event_meta["anchor_time_s"],
                    "vehicle_file": vehicle_file,
                    "event_file": event_file,
                    "signal_csv": str(signal_csv),
                    "meta_json": str(meta_json),
                }
            )
        except Exception as exc:
            missing_rows.append(
                {
                    "sample_key": sample_key,
                    "selection_bucket": row.get("selection_bucket"),
                    "vehicle_file": row.get("vehicle_file"),
                    "event_file": row.get("event_file"),
                    "error": repr(exc),
                }
            )

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_csv = args.output_dir / "event_signal_export_manifest.csv"
    metadata_df.to_csv(metadata_csv, index=False, encoding="utf-8-sig")

    if missing_rows:
        failures_csv = args.output_dir / "event_signal_export_failures.csv"
        pd.DataFrame(missing_rows).to_csv(failures_csv, index=False, encoding="utf-8-sig")
        print(f"[WARN] failures exported to: {failures_csv}")

    print(f"[OK] exported samples: {len(metadata_rows)}")
    print(f"[OK] manifest: {metadata_csv}")


if __name__ == "__main__":
    main()

