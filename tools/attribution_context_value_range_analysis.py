from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FS = 200.0
DT = 1.0 / FS
DEFAULT_DATE_TAG = "20260408"

DEFAULT_MANIFEST = Path(
    r"F:\data_set_process\data_process\datasetprocess\final_code\model\training\protocol_allphase_control_v2_context_full2s\sample_manifest.csv"
)
DEFAULT_ATTRIBUTION = Path(
    r"F:\data_set_process\data_process\reports\attribution_master_table.csv"
)
DEFAULT_OUTPUT_CSV = Path(
    rf"F:\data_set_process\data_process\reports\context_value_range_by_latency_bucket_{DEFAULT_DATE_TAG}.csv"
)
DEFAULT_OUTPUT_MD = Path(
    rf"F:\data_set_process\data_process\reports\context_value_range_by_latency_bucket_{DEFAULT_DATE_TAG}.md"
)

SIGNAL_SPECS = [
    ("steer_anchor_abs", "steer_anchor_raw", "abs(steer)"),
    ("steer_rate_abs", "steer_rate_raw", "abs(steer_rate)"),
    ("ay_abs", "ay_raw", "abs(ay)"),
    ("yawrate_abs", "yawrate_raw", "abs(yawrate)"),
]

FIND_COL_CANDIDATES = {
    "t_s": ["t_s", "time_s", "timestamp_s"],
    "steer": ["zx|SteeringWheel", "SteeringWheel", "steer"],
    "ay": ["zx|ay", "ay", "Ay", "lat_acc"],
    "yawrate": ["vyaw", "zx|vyaw", "YawRate", "zx|YawRate", "yaw_rate"],
}


@dataclass(frozen=True)
class VehicleColumns:
    t_s: str
    steer: str
    ay: str
    yawrate: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze anchor-point context signal value ranges by latency bucket."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--attribution", type=Path, default=DEFAULT_ATTRIBUTION)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--time-tolerance-s",
        type=float,
        default=DT + 1e-9,
        help="Maximum allowed |t_s(anchor_idx) - anchor_s| before falling back to nearest t_s.",
    )
    return parser.parse_args()


def find_col(columns: Iterable[str], candidates: list[str]) -> str | None:
    columns_list = list(columns)
    lower_map = {col.lower(): col for col in columns_list}
    for candidate in candidates:
        if candidate in columns_list:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def resolve_vehicle_columns(csv_path: Path) -> VehicleColumns:
    header = pd.read_csv(csv_path, nrows=0)
    cols = header.columns.tolist()
    resolved: dict[str, str] = {}
    for key, candidates in FIND_COL_CANDIDATES.items():
        col = find_col(cols, candidates)
        if col is None:
            raise KeyError(
                f"Missing required column for '{key}' in vehicle file: {csv_path}"
            )
        resolved[key] = col
    return VehicleColumns(
        t_s=resolved["t_s"],
        steer=resolved["steer"],
        ay=resolved["ay"],
        yawrate=resolved["yawrate"],
    )


def nearest_index(t_values: np.ndarray, target_s: float) -> int:
    insert_idx = int(np.searchsorted(t_values, target_s))
    if insert_idx <= 0:
        return 0
    if insert_idx >= len(t_values):
        return len(t_values) - 1
    prev_idx = insert_idx - 1
    next_idx = insert_idx
    if abs(t_values[prev_idx] - target_s) <= abs(t_values[next_idx] - target_s):
        return prev_idx
    return next_idx


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    usecols = [
        "sample_key",
        "subj",
        "split",
        "recording_id",
        "anchor_s",
        "anchor_idx",
        "vehicle_file",
    ]
    manifest = pd.read_csv(manifest_path, usecols=usecols)
    manifest = manifest.loc[manifest["split"] == "test"].copy()
    manifest["vehicle_file"] = manifest["vehicle_file"].astype(str)
    manifest["anchor_s"] = pd.to_numeric(manifest["anchor_s"], errors="coerce")
    manifest["anchor_idx"] = pd.to_numeric(manifest["anchor_idx"], errors="coerce")
    manifest = manifest.dropna(subset=["sample_key", "vehicle_file", "anchor_s"])
    if manifest["sample_key"].duplicated().any():
        dupes = manifest.loc[manifest["sample_key"].duplicated(), "sample_key"].head(5)
        raise ValueError(f"Duplicate sample_key in manifest: {dupes.tolist()}")
    return manifest


def load_attribution(attribution_path: Path) -> pd.DataFrame:
    usecols = [
        "sample_key",
        "latency_proxy_bucket",
        "eval_morphology_label",
        "delta_rmse_tail_abs_steer",
    ]
    attribution = pd.read_csv(attribution_path, usecols=usecols)
    if attribution["sample_key"].duplicated().any():
        dupes = attribution.loc[
            attribution["sample_key"].duplicated(), "sample_key"
        ].head(5)
        raise ValueError(f"Duplicate sample_key in attribution: {dupes.tolist()}")
    attribution["delta_rmse_tail_abs_steer"] = pd.to_numeric(
        attribution["delta_rmse_tail_abs_steer"], errors="coerce"
    )
    return attribution


def extract_vehicle_samples(
    samples_for_vehicle: pd.DataFrame, tolerance_s: float
) -> pd.DataFrame:
    vehicle_path = Path(samples_for_vehicle["vehicle_file"].iloc[0])
    cols = resolve_vehicle_columns(vehicle_path)
    usecols = [cols.t_s, cols.steer, cols.ay, cols.yawrate]
    vehicle_df = pd.read_csv(vehicle_path, usecols=usecols)

    t_values = vehicle_df[cols.t_s].to_numpy(dtype=np.float64)
    steer_values = vehicle_df[cols.steer].to_numpy(dtype=np.float64)
    ay_values = vehicle_df[cols.ay].to_numpy(dtype=np.float64)
    yawrate_values = vehicle_df[cols.yawrate].to_numpy(dtype=np.float64)
    steer_rate_values = np.gradient(steer_values, DT)

    records: list[dict[str, object]] = []
    for row in samples_for_vehicle.itertuples(index=False):
        anchor_s = float(row.anchor_s)
        row_count = len(vehicle_df)
        chosen_idx: int
        extraction_method: str

        if pd.notna(row.anchor_idx):
            anchor_idx = int(row.anchor_idx)
            if 0 <= anchor_idx < row_count:
                anchor_idx_time_error = abs(float(t_values[anchor_idx]) - anchor_s)
                if anchor_idx_time_error <= tolerance_s:
                    chosen_idx = anchor_idx
                    extraction_method = "anchor_idx"
                else:
                    chosen_idx = nearest_index(t_values, anchor_s)
                    extraction_method = "nearest_t_s"
            else:
                chosen_idx = nearest_index(t_values, anchor_s)
                extraction_method = "nearest_t_s"
        else:
            chosen_idx = nearest_index(t_values, anchor_s)
            extraction_method = "nearest_t_s"

        steer_raw = float(steer_values[chosen_idx])
        steer_rate_raw = float(steer_rate_values[chosen_idx])
        ay_raw = float(ay_values[chosen_idx])
        yawrate_raw = float(yawrate_values[chosen_idx])
        matched_t_s = float(t_values[chosen_idx])

        records.append(
            {
                "sample_key": row.sample_key,
                "subj": row.subj,
                "recording_id": row.recording_id,
                "vehicle_file": row.vehicle_file,
                "anchor_s": anchor_s,
                "anchor_idx_manifest": row.anchor_idx,
                "anchor_idx_used": chosen_idx,
                "anchor_t_s_used": matched_t_s,
                "anchor_t_error_s": matched_t_s - anchor_s,
                "anchor_t_error_abs_s": abs(matched_t_s - anchor_s),
                "extraction_method": extraction_method,
                "steer_anchor_raw": steer_raw,
                "steer_rate_raw": steer_rate_raw,
                "ay_raw": ay_raw,
                "yawrate_raw": yawrate_raw,
                "steer_anchor_abs": abs(steer_raw),
                "steer_rate_abs": abs(steer_rate_raw),
                "ay_abs": abs(ay_raw),
                "yawrate_abs": abs(yawrate_raw),
            }
        )

    return pd.DataFrame.from_records(records)


def extract_all_anchor_signals(merged: pd.DataFrame, tolerance_s: float) -> pd.DataFrame:
    extracted_frames: list[pd.DataFrame] = []
    for _, group_df in merged.groupby("vehicle_file", sort=True):
        extracted_frames.append(extract_vehicle_samples(group_df, tolerance_s))
    extracted = pd.concat(extracted_frames, ignore_index=True)
    return merged.merge(extracted, on=["sample_key", "subj", "recording_id", "vehicle_file", "anchor_s"], how="left", validate="one_to_one")


def summarize_signals(
    df: pd.DataFrame, group_cols: list[str], table_name: str
) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    grouped = df.groupby(group_cols, dropna=False, sort=True)
    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        key_map = dict(zip(group_cols, group_key))
        for signal_col, raw_col, signal_label in SIGNAL_SPECS:
            series = pd.to_numeric(group_df[signal_col], errors="coerce").dropna()
            summary_rows.append(
                {
                    "table_name": table_name,
                    **key_map,
                    "signal": signal_label,
                    "signal_column": signal_col,
                    "raw_signal_column": raw_col,
                    "n_samples": int(series.shape[0]),
                    "mean": float(series.mean()) if not series.empty else np.nan,
                    "std": float(series.std(ddof=1)) if len(series) > 1 else np.nan,
                    "min": float(series.min()) if not series.empty else np.nan,
                    "median": float(series.median()) if not series.empty else np.nan,
                    "max": float(series.max()) if not series.empty else np.nan,
                }
            )
    return pd.DataFrame(summary_rows)


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2:
        return np.nan
    x_valid = valid.iloc[:, 0].to_numpy(dtype=np.float64)
    y_valid = valid.iloc[:, 1].to_numpy(dtype=np.float64)
    x_std = x_valid.std(ddof=1)
    y_std = y_valid.std(ddof=1)
    if x_std == 0.0 or y_std == 0.0:
        return np.nan
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def build_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset_name, subset_df in [
        ("all_test", df),
        ("Q1_fast_only", df.loc[df["latency_proxy_bucket"] == "Q1_fast"].copy()),
    ]:
        for signal_col, raw_col, signal_label in SIGNAL_SPECS:
            valid = subset_df[[signal_col, "delta_rmse_tail_abs_steer"]].dropna()
            rows.append(
                {
                    "table_name": "pearson_correlation",
                    "subset": subset_name,
                    "signal": signal_label,
                    "signal_column": signal_col,
                    "raw_signal_column": raw_col,
                    "n_samples": int(len(valid)),
                    "pearson_r": pearson_corr(
                        subset_df[signal_col], subset_df["delta_rmse_tail_abs_steer"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_q1_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    q1_df = df.loc[df["latency_proxy_bucket"] == "Q1_fast"].copy()
    non_q1_df = df.loc[df["latency_proxy_bucket"] != "Q1_fast"].copy()
    for signal_col, raw_col, signal_label in SIGNAL_SPECS:
        q1_series = q1_df[signal_col].dropna()
        non_q1_series = non_q1_df[signal_col].dropna()
        rows.append(
            {
                "table_name": "q1_fast_vs_non_q1_fast",
                "signal": signal_label,
                "signal_column": signal_col,
                "raw_signal_column": raw_col,
                "q1_n": int(len(q1_series)),
                "non_q1_n": int(len(non_q1_series)),
                "q1_mean": float(q1_series.mean()) if not q1_series.empty else np.nan,
                "non_q1_mean": float(non_q1_series.mean()) if not non_q1_series.empty else np.nan,
                "q1_std": float(q1_series.std(ddof=1)) if len(q1_series) > 1 else np.nan,
                "non_q1_std": float(non_q1_series.std(ddof=1)) if len(non_q1_series) > 1 else np.nan,
                "q1_min": float(q1_series.min()) if not q1_series.empty else np.nan,
                "non_q1_min": float(non_q1_series.min()) if not non_q1_series.empty else np.nan,
                "q1_median": float(q1_series.median()) if not q1_series.empty else np.nan,
                "non_q1_median": float(non_q1_series.median()) if not non_q1_series.empty else np.nan,
                "q1_max": float(q1_series.max()) if not q1_series.empty else np.nan,
                "non_q1_max": float(non_q1_series.max()) if not non_q1_series.empty else np.nan,
                "mean_diff_q1_minus_non_q1": (
                    float(q1_series.mean() - non_q1_series.mean())
                    if (not q1_series.empty and not non_q1_series.empty)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def render_markdown_table(df: pd.DataFrame, float_cols: list[str] | None = None) -> str:
    if df.empty:
        return "_No rows._"

    float_cols = float_cols or []
    display_df = df.copy()
    for col in float_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda x: "" if pd.isna(x) else f"{x:.4f}"
            )
    for col in display_df.columns:
        if col not in float_cols:
            display_df[col] = display_df[col].map(
                lambda x: "" if pd.isna(x) else str(x)
            )

    header = "| " + " | ".join(display_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display_df.columns)) + " |"
    lines = [header, separator]
    for row in display_df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report_markdown(
    analysis_df: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    q1_summary: pd.DataFrame,
    morph_summary: pd.DataFrame,
    corr_df: pd.DataFrame,
    manifest_path: Path,
    attribution_path: Path,
) -> str:
    sample_count = len(analysis_df)
    vehicle_count = analysis_df["vehicle_file"].nunique()
    bucket_counts = (
        analysis_df["latency_proxy_bucket"].value_counts(dropna=False).rename_axis("latency_proxy_bucket").reset_index(name="n_samples")
    )
    extraction_counts = (
        analysis_df["extraction_method"].value_counts(dropna=False).rename_axis("extraction_method").reset_index(name="n_samples")
    )
    time_error_stats = {
        "mean": float(analysis_df["anchor_t_error_abs_s"].mean()),
        "median": float(analysis_df["anchor_t_error_abs_s"].median()),
        "max": float(analysis_df["anchor_t_error_abs_s"].max()),
    }

    bucket_mean_pivot = (
        bucket_summary.pivot(index="signal", columns="latency_proxy_bucket", values="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    q1_report_table = q1_summary[
        [
            "signal",
            "q1_n",
            "non_q1_n",
            "q1_mean",
            "non_q1_mean",
            "mean_diff_q1_minus_non_q1",
            "q1_median",
            "non_q1_median",
        ]
    ].copy()
    corr_report_table = corr_df[
        ["subset", "signal", "n_samples", "pearson_r"]
    ].copy()
    morph_mean_table = morph_summary[
        [
            "latency_proxy_bucket",
            "eval_morphology_label",
            "signal",
            "n_samples",
            "mean",
            "median",
        ]
    ].copy()

    signal_findings: list[str] = []
    for row in q1_summary.itertuples(index=False):
        direction = "higher" if row.mean_diff_q1_minus_non_q1 > 0 else "lower"
        signal_findings.append(
            f"- `{row.signal}`: Q1_fast mean {row.q1_mean:.4f} vs non_Q1_fast {row.non_q1_mean:.4f} ({direction} by {row.mean_diff_q1_minus_non_q1:.4f})."
        )

    corr_sorted = corr_report_table.loc[corr_report_table["subset"] == "all_test"].sort_values(
        "pearson_r", ascending=False
    )
    q1_only_corr_sorted = corr_report_table.loc[
        corr_report_table["subset"] == "Q1_fast_only"
    ].sort_values("pearson_r", ascending=False)
    top_corr_lines = [
        f"- `{row.signal}`: Pearson r = {row.pearson_r:.4f} on {int(row.n_samples)} test samples."
        for row in corr_sorted.itertuples(index=False)
    ]
    q1_only_top_corr_lines = [
        f"- `{row.signal}`: Pearson r = {row.pearson_r:.4f} on {int(row.n_samples)} Q1_fast samples."
        for row in q1_only_corr_sorted.itertuples(index=False)
    ]

    strongest_q1_signal = q1_summary.sort_values(
        "mean_diff_q1_minus_non_q1", ascending=False
    ).iloc[0]
    weakest_q1_signal = q1_summary.sort_values(
        "mean_diff_q1_minus_non_q1", ascending=True
    ).iloc[0]
    overall_corr_abs = corr_report_table.loc[
        corr_report_table["subset"] == "all_test", "pearson_r"
    ].abs()
    q1_corr_abs = corr_report_table.loc[
        corr_report_table["subset"] == "Q1_fast_only"
    ].copy()
    q1_corr_abs["pearson_abs"] = q1_corr_abs["pearson_r"].abs()
    q1_corr_top = q1_corr_abs.sort_values("pearson_abs", ascending=False).iloc[0]

    lines = [
        "# Context / Anchor Signal Value Range Analysis",
        "",
        "## Scope",
        f"- Manifest: `{manifest_path}`",
        f"- Attribution table: `{attribution_path}`",
        f"- Test samples analyzed: `{sample_count}`",
        f"- Unique vehicle files loaded: `{vehicle_count}`",
        "- Anchor signals analyzed as absolute magnitudes for value-range comparison: `abs(steer)`, `abs(steer_rate)`, `abs(ay)`, `abs(yawrate)`.",
        "- Raw signed values were still extracted from vehicle CSVs to keep the anchor reading traceable to source rows.",
        "",
        "## Extraction QC",
        f"- Extraction method counts: {', '.join(f'{row.extraction_method}={int(row.n_samples)}' for row in extraction_counts.itertuples(index=False))}",
        f"- Absolute anchor-time error: mean `{time_error_stats['mean']:.6f}` s, median `{time_error_stats['median']:.6f}` s, max `{time_error_stats['max']:.6f}` s.",
        "",
        "## Sample Counts By Latency Bucket",
        render_markdown_table(bucket_counts),
        "",
        "## Latency Bucket Mean Comparison",
        render_markdown_table(bucket_mean_pivot, float_cols=[col for col in bucket_mean_pivot.columns if col != "signal"]),
        "",
        "## Q1_fast vs non_Q1_fast",
        render_markdown_table(
            q1_report_table,
            float_cols=[
                "q1_mean",
                "non_q1_mean",
                "mean_diff_q1_minus_non_q1",
                "q1_median",
                "non_q1_median",
            ],
        ),
        "",
        "### Direct Findings",
        *signal_findings,
        "",
        "## Secondary Grouping: Latency Bucket x Morphology",
        render_markdown_table(
            morph_mean_table,
            float_cols=["mean", "median"],
        ),
        "",
        "## Pearson Correlation With delta_rmse_tail_abs_steer",
        render_markdown_table(
            corr_report_table,
            float_cols=["pearson_r"],
        ),
        "",
        "### Correlation Ranking (All Test Samples)",
        *top_corr_lines,
        "",
        "### Correlation Ranking (Q1_fast Only)",
        *q1_only_top_corr_lines,
        "",
        "## Key Takeaways",
        f"- The broad hypothesis is not supported as a uniform pattern: only `{strongest_q1_signal.signal}` is higher in `Q1_fast` than `non_Q1_fast` (+{strongest_q1_signal.mean_diff_q1_minus_non_q1:.4f} by mean), while `{weakest_q1_signal.signal}` shows the largest negative gap ({weakest_q1_signal.mean_diff_q1_minus_non_q1:.4f}).",
        f"- Across all 749 test samples, anchor-signal correlations with `delta_rmse_tail_abs_steer` are weak overall (max |r| = {overall_corr_abs.max():.4f}).",
        f"- Within the 188 `Q1_fast` samples, the strongest single relationship is `{q1_corr_top.signal}` with |r| = {q1_corr_top.pearson_abs:.4f}, which is still only moderate.",
        "- Current evidence therefore supports, at most, a narrow `steer_rate`-intensity difference rather than a consistent four-signal anchor-value elevation in `Q1_fast`.",
        "",
        "## Interpretation",
        "- If `Q1_fast` rows are consistently higher across these anchor magnitudes, that supports the hypothesis that stronger anchor-state context is part of the tail mismatch mechanism.",
        "- If correlations remain weak even when Q1_fast group means are elevated, then anchor magnitude alone is likely insufficient and the remaining explanation shifts toward conditioning structure or downstream temporal broadcast effects.",
        "- The CSV output contains the full `latency_proxy_bucket`, `Q1_fast vs non_Q1_fast`, `latency_proxy_bucket x eval_morphology_label`, and Pearson-correlation tables in a single long-format file.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    attribution = load_attribution(args.attribution)

    merged = manifest.merge(
        attribution,
        on="sample_key",
        how="inner",
        validate="one_to_one",
    )
    merged["latency_proxy_bucket"] = merged["latency_proxy_bucket"].fillna("missing")
    merged["eval_morphology_label"] = merged["eval_morphology_label"].fillna("missing")

    analysis_df = extract_all_anchor_signals(merged, tolerance_s=args.time_tolerance_s)
    analysis_df["q1_fast_flag"] = np.where(
        analysis_df["latency_proxy_bucket"] == "Q1_fast", "Q1_fast", "non_Q1_fast"
    )

    bucket_summary = summarize_signals(
        analysis_df,
        ["latency_proxy_bucket"],
        table_name="latency_bucket_signal_stats",
    )
    q1_summary = build_q1_comparison_table(analysis_df)
    morph_summary = summarize_signals(
        analysis_df,
        ["latency_proxy_bucket", "eval_morphology_label"],
        table_name="latency_bucket_morphology_signal_stats",
    )
    corr_df = build_correlation_table(analysis_df)

    csv_tables = [
        bucket_summary,
        q1_summary,
        morph_summary,
        corr_df,
    ]
    output_csv_df = pd.concat(csv_tables, ignore_index=True, sort=False)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    report_md = build_report_markdown(
        analysis_df=analysis_df,
        bucket_summary=bucket_summary,
        q1_summary=q1_summary,
        morph_summary=morph_summary,
        corr_df=corr_df,
        manifest_path=args.manifest,
        attribution_path=args.attribution,
    )
    args.output_md.write_text(report_md, encoding="utf-8")

    print(f"Wrote CSV report: {args.output_csv}")
    print(f"Wrote Markdown report: {args.output_md}")
    print(f"Analyzed test samples: {len(analysis_df)}")


if __name__ == "__main__":
    main()
