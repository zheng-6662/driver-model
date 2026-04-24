#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Select good/bad model samples for CarSim replay, mapped back to original files.

Workflow:
1. Read sample-level metrics exported by the current model evaluation.
2. Aggregate the chosen error metric per sample across seeds.
3. Join with the probe manifest to recover original vehicle/event files.
4. Prefer samples with larger absolute anchor_roll (used here as a body-roll proxy).
5. Export "good" and "bad" sample tables for later CarSim clipping.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_SAMPLE_METRICS = Path(
    r"F:\data_set_process\data_process\04_project_logs\reports\trajectory_aware_round4_20260326\round4_sample_level_long.csv"
)
DEFAULT_MANIFEST = Path(
    r"F:\data_set_process\data_process\04_project_logs\reports\trajectory_foundation_audit_20260325\task_01_manifest_audit\unified_manifest_v1_draft.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"F:\data_set_process\data_process\04_project_logs\reports\carsim_sample_selection"
)
DEFAULT_DATA_ROOT = Path(
    r"F:\\data_set_process\\data_process\\01_datasets\\多模态数据\\被试数据集合"
)


@dataclass
class ParsedSampleKey:
    subj: str
    file_name: str
    episode_id: int
    anchor_or_suffix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select good/bad samples with large body-roll proxy for CarSim inspection."
    )
    parser.add_argument(
        "--sample-metrics",
        type=Path,
        default=DEFAULT_SAMPLE_METRICS,
        help="Sample-level metric CSV. Default uses the latest round4 table.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest CSV that contains original vehicle/event paths and anchor_roll.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for exported CSV summaries.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="round4_stage2_refine",
        help="Filter a specific model/run label from the sample metric table.",
    )
    parser.add_argument(
        "--metric-column",
        type=str,
        default="rmse_2s_abs_steer",
        help="Metric used to judge prediction quality. Smaller is better.",
    )
    parser.add_argument(
        "--roll-column",
        type=str,
        default="anchor_roll",
        help="Column used as the large-body-attitude proxy. If missing, it will be computed from vehicle CSV using anchor_idx.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to filter in the sample metric table.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="How many good and bad samples to export.",
    )
    parser.add_argument(
        "--roll-top-percentile",
        type=float,
        default=0.7,
        help="Keep only samples with abs(roll) above this percentile. 0.7 means top 30%% roll magnitude.",
    )
    parser.add_argument(
        "--min-roll-abs",
        type=float,
        default=None,
        help="Optional absolute roll threshold. If set, samples below this threshold are removed.",
    )
    parser.add_argument(
        "--phase-type",
        type=str,
        default=None,
        help="Optional phase_type filter, e.g. primary.",
    )
    parser.add_argument(
        "--road-type",
        type=str,
        default=None,
        help="Optional road_type_anchor filter, e.g. curve.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory of the original subject folders.",
    )
    return parser.parse_args()


def parse_sample_key(sample_key: str) -> ParsedSampleKey:
    if "::" in sample_key:
        parts = sample_key.split("::")
        if len(parts) < 4:
            raise ValueError(f"Unexpected sample_key format: {sample_key}")
        return ParsedSampleKey(
            subj=parts[0],
            file_name=parts[1],
            episode_id=int(parts[2]),
            anchor_or_suffix=parts[3],
        )

    if "|" in sample_key:
        parts = sample_key.split("|")
        if len(parts) < 4:
            raise ValueError(f"Unexpected sample_key format: {sample_key}")
        return ParsedSampleKey(
            subj=parts[0],
            file_name=parts[1],
            episode_id=int(parts[2]),
            anchor_or_suffix=parts[3],
        )

    raise ValueError(f"Unsupported sample_key format: {sample_key}")


def load_sample_metrics(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.sample_metrics)
    required_cols = {"sample_key", args.metric_column}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Sample metric table missing columns: {sorted(missing)}")

    if args.run_label and "run_label" in df.columns:
        df = df[df["run_label"] == args.run_label].copy()
    if args.split and "split" in df.columns:
        df = df[df["split"] == args.split].copy()
    if args.phase_type and "phase_type" in df.columns:
        df = df[df["phase_type"] == args.phase_type].copy()
    if args.road_type and "road_type_anchor" in df.columns:
        df = df[df["road_type_anchor"] == args.road_type].copy()

    if df.empty:
        raise ValueError("No rows left after filtering sample metrics. Check run_label/split filters.")

    parsed = df["sample_key"].map(parse_sample_key)
    df["subj_from_key"] = [x.subj for x in parsed]
    df["file_from_key"] = [x.file_name for x in parsed]
    df["episode_id_from_key"] = [x.episode_id for x in parsed]

    agg_spec = {
        args.metric_column: ["mean", "std", "min", "max", "count"],
    }
    passthrough_cols = [
        c
        for c in [
            "subj",
            "phase_type",
            "road_type_anchor",
            "mechanism_tag",
            "eval_morphology_label",
            "run_label",
        ]
        if c in df.columns
    ]
    for col in passthrough_cols:
        agg_spec[col] = "first"

    grouped = df.groupby("sample_key", as_index=False).agg(agg_spec)
    grouped.columns = [
        "sample_key"
        if col == ("sample_key", "")
        else (
            col[0]
            if col[1] in ("", "first")
            else f"{col[0]}_{col[1]}"
        )
        for col in grouped.columns.to_flat_index()
    ]

    grouped = grouped.rename(
        columns={
            f"{args.metric_column}_mean": "metric_mean",
            f"{args.metric_column}_std": "metric_std",
            f"{args.metric_column}_min": "metric_min",
            f"{args.metric_column}_max": "metric_max",
            f"{args.metric_column}_count": "seed_count",
        }
    )

    parsed_group = grouped["sample_key"].map(parse_sample_key)
    grouped["subj"] = [x.subj for x in parsed_group]
    grouped["file"] = [x.file_name for x in parsed_group]
    grouped["episode_id"] = [x.episode_id for x in parsed_group]
    return grouped


def build_raw_paths(df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    out = df.copy()
    out["vehicle_file"] = out.apply(
        lambda r: str(data_root / str(r["subj"]) / "vehicle" / str(r["file"])),
        axis=1,
    )
    out["event_file"] = out.apply(
        lambda r: str(data_root / str(r["subj"]) / "event" / str(r["file"]).replace(
            "_vehicle_aligned_cleaned.csv",
            "_vehicle_aligned_cleaned_events_v312.csv",
        )),
        axis=1,
    )
    return out


def compute_anchor_roll(df: pd.DataFrame, roll_column: str) -> pd.DataFrame:
    if roll_column in df.columns and df[roll_column].notna().any():
        return df

    roll_cache: dict[str, Optional[float]] = {}

    def read_roll(row: pd.Series) -> float:
        vehicle_file = str(row["vehicle_file"])
        anchor_idx = row.get("anchor_idx", np.nan)
        if pd.isna(anchor_idx):
            return np.nan
        if vehicle_file not in roll_cache:
            try:
                vf = pd.read_csv(vehicle_file)
                roll_col = next((c for c in vf.columns if c.lower() in {"zx|roll", "roll"}), None)
                if roll_col is None:
                    roll_cache[vehicle_file] = np.nan
                else:
                    roll_cache[vehicle_file] = roll_col
                    roll_cache[f"{vehicle_file}::__df__"] = vf
            except Exception:
                roll_cache[vehicle_file] = np.nan
        roll_meta = roll_cache.get(vehicle_file)
        if isinstance(roll_meta, float) and np.isnan(roll_meta):
            return np.nan
        try:
            vf = roll_cache[f"{vehicle_file}::__df__"]
            idx = int(anchor_idx)
            if idx < 0 or idx >= len(vf):
                return np.nan
            return float(vf.iloc[idx][roll_meta])
        except Exception:
            return np.nan

    out = df.copy()
    out[roll_column] = out.apply(read_roll, axis=1)
    return out


def load_manifest(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.manifest)
    required_cols = {"subj", "file"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Manifest missing columns: {sorted(missing)}")

    df = df.copy()
    if "split" in df.columns and args.split:
        df = df[df["split"] == args.split].copy()
    if "phase_type" in df.columns and args.phase_type:
        df = df[df["phase_type"] == args.phase_type].copy()

    if "sample_key_unified" in df.columns:
        df["sample_key"] = df["sample_key_unified"].astype(str)
    elif "sample_key" not in df.columns:
        raise KeyError("Manifest must contain either sample_key_unified or sample_key.")

    if "episode_id" in df.columns:
        df["episode_id"] = pd.to_numeric(df["episode_id"], errors="coerce")

    if "vehicle_file" not in df.columns or "event_file" not in df.columns:
        df = build_raw_paths(df, args.data_root)

    if args.roll_column not in df.columns:
        df[args.roll_column] = np.nan
    df = compute_anchor_roll(df, args.roll_column)

    sort_keys = [args.roll_column] if args.roll_column in df.columns else None
    if sort_keys:
        df = df.sort_values(by=sort_keys, key=lambda s: s.abs(), ascending=False)
    df = df.drop_duplicates(subset=["sample_key"], keep="first")
    return df


def filter_large_roll(df: pd.DataFrame, roll_column: str, percentile: float, min_roll_abs: Optional[float]) -> pd.DataFrame:
    out = df.copy()
    out["roll_abs"] = out[roll_column].abs()

    if percentile is not None:
        percentile = float(np.clip(percentile, 0.0, 1.0))
        threshold = float(out["roll_abs"].quantile(percentile))
        out = out[out["roll_abs"] >= threshold].copy()
        out["roll_percentile_threshold"] = threshold
    else:
        out["roll_percentile_threshold"] = np.nan

    if min_roll_abs is not None:
        out = out[out["roll_abs"] >= float(min_roll_abs)].copy()

    return out


def enrich_paths(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["vehicle_file"] = out["vehicle_file"].map(str)
    out["event_file"] = out["event_file"].map(str)
    out["subject_dir"] = out["vehicle_file"].map(lambda x: str(Path(x).parents[1]))
    out["vehicle_dir"] = out["vehicle_file"].map(lambda x: str(Path(x).parent))
    out["event_dir"] = out["event_file"].map(lambda x: str(Path(x).parent))
    out["recording_stem"] = out["file"].map(lambda x: Path(x).stem)
    return out


def export_tables(df: pd.DataFrame, args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common_cols = [
        "sample_key",
        "subj",
        "file",
        "episode_id",
        "metric_mean",
        "metric_std",
        "metric_min",
        "metric_max",
        "seed_count",
        "roll_abs",
        args.roll_column,
        "anchor_ay",
        "anchor_yawrate",
        "road_type_anchor",
        "mechanism_tag",
        "eval_morphology_label",
        "subject_dir",
        "vehicle_dir",
        "event_dir",
        "vehicle_file",
        "event_file",
    ]
    available_cols = [c for c in common_cols if c in df.columns]

    good = (
        df.sort_values(by=["metric_mean", "roll_abs"], ascending=[True, False])
        .head(args.top_n)
        .copy()
    )
    good["selection_bucket"] = "good"

    bad = (
        df.sort_values(by=["metric_mean", "roll_abs"], ascending=[False, False])
        .head(args.top_n)
        .copy()
    )
    bad["selection_bucket"] = "bad"

    summary = pd.concat([good, bad], ignore_index=True)
    summary_cols = ["selection_bucket"] + available_cols

    percentile_tag = f"p{int(round(float(args.roll_top_percentile) * 100))}" if args.roll_top_percentile is not None else "pNA"
    prefix = f"{args.run_label}_{args.metric_column}"
    variant = f"top{args.top_n}_{percentile_tag}"
    good_path = args.output_dir / f"{prefix}_good_top{args.top_n}.csv"
    bad_path = args.output_dir / f"{prefix}_bad_top{args.top_n}.csv"
    summary_path = args.output_dir / f"{prefix}_summary_top{args.top_n}.csv"

    good[summary_cols].to_csv(good_path, index=False, encoding="utf-8-sig")
    bad[summary_cols].to_csv(bad_path, index=False, encoding="utf-8-sig")
    summary[summary_cols].to_csv(summary_path, index=False, encoding="utf-8-sig")

    info = pd.DataFrame(
        [
            {
                "sample_metrics": str(args.sample_metrics),
                "manifest": str(args.manifest),
                "run_label": args.run_label,
                "metric_column": args.metric_column,
                "split": args.split,
                "phase_type": args.phase_type,
                "road_type": args.road_type,
                "top_n": args.top_n,
                "roll_top_percentile": args.roll_top_percentile,
                "min_roll_abs": args.min_roll_abs,
                "selected_pool_size": len(df),
                "metric_mean_min": float(df["metric_mean"].min()),
                "metric_mean_max": float(df["metric_mean"].max()),
                "roll_abs_min": float(df["roll_abs"].min()),
                "roll_abs_max": float(df["roll_abs"].max()),
                "good_csv": str(good_path),
                "bad_csv": str(bad_path),
                "summary_csv": str(summary_path),
            }
        ]
    )
    info.to_csv(args.output_dir / f"{prefix}_run_info_{variant}.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] selected pool size: {len(df)}")
    print(f"[OK] good samples: {good_path}")
    print(f"[OK] bad samples:  {bad_path}")
    print(f"[OK] summary:      {summary_path}")


def main() -> None:
    args = parse_args()
    sample_df = load_sample_metrics(args)
    manifest_df = load_manifest(args)

    if "sample_key" not in manifest_df.columns:
        raise KeyError("Manifest is expected to expose a sample_key column after preprocessing.")

    merged = sample_df.merge(manifest_df, how="left", on=["sample_key"], suffixes=("", "_manifest"))

    for col in ["subj", "file", "episode_id"]:
        manifest_col = f"{col}_manifest"
        if col not in merged.columns and manifest_col in merged.columns:
            merged[col] = merged[manifest_col]
        elif manifest_col in merged.columns:
            merged[col] = merged[col].fillna(merged[manifest_col])

    missing_match = merged["vehicle_file"].isna().sum()
    if missing_match > 0:
        print(f"[WARN] samples without manifest match: {missing_match}")

    merged = merged[merged["vehicle_file"].notna()].copy()
    merged = filter_large_roll(
        merged,
        roll_column=args.roll_column,
        percentile=args.roll_top_percentile,
        min_roll_abs=args.min_roll_abs,
    )

    if merged.empty:
        raise ValueError("No samples left after manifest join and roll filtering.")

    merged = enrich_paths(merged)
    export_tables(merged, args)


if __name__ == "__main__":
    main()


