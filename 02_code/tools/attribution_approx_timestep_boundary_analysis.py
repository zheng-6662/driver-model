from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATE_TAG = "20260408"

DEFAULT_ATTRIBUTION = Path(
    r"F:\data_set_process\data_process\04_project_logs\reports\attribution_master_table.csv"
)
DEFAULT_BASELINE = Path(
    r"F:\data_set_process\data_process\04_project_logs\reports\v3_selection_conditioned_interaction_pilot_20260327\task_2_conditioned_v2\formal_eval\baseline_trajectory_sample_metrics.csv"
)
DEFAULT_CONDITIONED = Path(
    r"F:\data_set_process\data_process\04_project_logs\reports\v3_selection_conditioned_interaction_pilot_20260327\task_2_conditioned_v2\formal_eval\conditioned_trajectory_sample_metrics.csv"
)
DEFAULT_COMPARISON = Path(
    r"F:\data_set_process\data_process\04_project_logs\reports\v3_selection_conditioned_interaction_pilot_20260327\task_2_conditioned_v2\formal_eval\sample_level_comparison.csv"
)
DEFAULT_OUTPUT_CSV = Path(
    rf"F:\data_set_process\data_process\04_project_logs\reports\approx_timestep_boundary_analysis_{DEFAULT_DATE_TAG}.csv"
)
DEFAULT_OUTPUT_MD = Path(
    rf"F:\data_set_process\data_process\04_project_logs\reports\approx_timestep_boundary_analysis_{DEFAULT_DATE_TAG}.md"
)
DEFAULT_BOUNDARY_FIG = Path(
    rf"F:\data_set_process\data_process\04_project_logs\reports\approx_boundary_slope_shift_scatter_{DEFAULT_DATE_TAG}.png"
)
DEFAULT_Q1_SINGLE_LOBE_FIG = Path(
    rf"F:\data_set_process\data_process\04_project_logs\reports\approx_q1fast_single_lobe_amp_boundary_scatter_{DEFAULT_DATE_TAG}.png"
)

ALIGNMENT_METRICS = [
    "rmse_pre_tail_abs_steer",
    "rmse_tail_abs_steer",
    "tail_slope_abs_err",
    "boundary_slope_abs_err",
    "boundary_shift_abs_err",
    "peak_abs_amp_err",
    "shape_corr",
]

REPORT_FLOAT_COLUMNS = [
    "baseline_front_rmse_mean",
    "baseline_tail_rmse_mean",
    "conditioned_front_rmse_mean",
    "conditioned_tail_rmse_mean",
    "baseline_ratio_mean",
    "conditioned_ratio_mean",
    "delta_ratio_mean",
    "delta_front_rmse_mean",
    "delta_tail_rmse_mean",
    "tail_minus_front_delta_mean",
    "tail_positive_share",
    "front_positive_share",
    "tail_driven_share",
    "front_driven_share",
    "boundary_slope_baseline_mean",
    "boundary_slope_conditioned_mean",
    "boundary_shift_baseline_mean",
    "boundary_shift_conditioned_mean",
    "tail_slope_baseline_mean",
    "tail_slope_conditioned_mean",
    "peak_amp_baseline_mean",
    "peak_amp_conditioned_mean",
    "delta_boundary_slope_mean",
    "delta_boundary_shift_mean",
    "delta_tail_slope_mean",
    "delta_peak_amp_mean",
    "slope_worsening_share",
    "shift_worsening_share",
    "amplitude_worsening_share",
    "pearson_r",
    "mean_value",
    "median_value",
    "value",
    "peak_abs_amp_err_conditioned",
    "boundary_shift_abs_err_conditioned",
    "boundary_slope_abs_err_conditioned",
    "shape_corr_conditioned",
    "delta_rmse_tail_abs_steer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Approximate timestep concentration and boundary-mode analysis "
            "using existing formal-eval CSVs."
        )
    )
    parser.add_argument("--attribution", type=Path, default=DEFAULT_ATTRIBUTION)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--conditioned", type=Path, default=DEFAULT_CONDITIONED)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--boundary-fig", type=Path, default=DEFAULT_BOUNDARY_FIG)
    parser.add_argument(
        "--q1-single-lobe-fig", type=Path, default=DEFAULT_Q1_SINGLE_LOBE_FIG
    )
    parser.add_argument(
        "--shift-threshold",
        type=float,
        default=0.02,
        help="Minimum delta_boundary_shift_abs_err treated as a meaningful worsening.",
    )
    parser.add_argument(
        "--slope-threshold",
        type=float,
        default=0.02,
        help="Minimum delta_boundary_slope_abs_err treated as a meaningful smoothing increase.",
    )
    return parser.parse_args()


def require_unique(df: pd.DataFrame, key: str, name: str) -> None:
    if df[key].duplicated().any():
        dupes = df.loc[df[key].duplicated(), key].head(5).tolist()
        raise ValueError(f"Duplicate {key} detected in {name}: {dupes}")


def load_inputs(
    attribution_path: Path,
    baseline_path: Path,
    conditioned_path: Path,
    comparison_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attribution = pd.read_csv(attribution_path)
    baseline = pd.read_csv(baseline_path)
    conditioned = pd.read_csv(conditioned_path)
    comparison = pd.read_csv(comparison_path)

    for name, df in [
        ("attribution", attribution),
        ("baseline", baseline),
        ("conditioned", conditioned),
        ("comparison", comparison),
    ]:
        require_unique(df, "sample_key", name)

    return attribution, baseline, conditioned, comparison


def validate_formal_eval_alignment(
    baseline: pd.DataFrame, conditioned: pd.DataFrame, comparison: pd.DataFrame
) -> list[str]:
    notes: list[str] = []
    base_renamed = baseline[["sample_key", *ALIGNMENT_METRICS]].rename(
        columns={metric: f"{metric}_baseline_src" for metric in ALIGNMENT_METRICS}
    )
    cond_renamed = conditioned[["sample_key", *ALIGNMENT_METRICS]].rename(
        columns={metric: f"{metric}_conditioned_src" for metric in ALIGNMENT_METRICS}
    )
    merged = comparison.merge(
        base_renamed, on="sample_key", how="inner", validate="one_to_one"
    )
    merged = merged.merge(
        cond_renamed, on="sample_key", how="inner", validate="one_to_one"
    )

    if len(merged) != len(comparison):
        raise ValueError(
            "Failed to align all comparison rows with baseline/conditioned sample metrics."
        )

    for metric in ALIGNMENT_METRICS:
        baseline_diff = (
            merged[f"{metric}_baseline"] - merged[f"{metric}_baseline_src"]
        ).abs().max()
        conditioned_diff = (
            merged[f"{metric}_conditioned"] - merged[f"{metric}_conditioned_src"]
        ).abs().max()
        notes.append(
            f"{metric}: baseline max abs diff={baseline_diff:.6g}, "
            f"conditioned max abs diff={conditioned_diff:.6g}"
        )
        if baseline_diff > 1e-12 or conditioned_diff > 1e-12:
            raise ValueError(
                f"Detected mismatch between comparison CSV and source metric CSV for {metric}."
            )
    return notes


def build_analysis_frame(attribution: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    attr_cols = [
        "sample_key",
        "subj",
        "latency_proxy_bucket",
        "eval_morphology_label",
        "delta_rmse_tail_abs_steer",
    ]
    merged = comparison.merge(
        attribution[attr_cols],
        on="sample_key",
        how="left",
        suffixes=("", "_attr"),
        validate="one_to_one",
    )
    if merged["latency_proxy_bucket"].isna().any():
        missing = merged.loc[
            merged["latency_proxy_bucket"].isna(), "sample_key"
        ].head(5)
        raise ValueError(
            f"Missing latency_proxy_bucket after attribution merge: {missing.tolist()}"
        )

    merged["latency_proxy_bucket"] = merged["latency_proxy_bucket"].fillna("missing")
    merged["eval_morphology_label"] = merged["eval_morphology_label"].fillna("missing")

    merged["tail_to_front_ratio_baseline"] = (
        merged["rmse_tail_abs_steer_baseline"]
        / merged["rmse_pre_tail_abs_steer_baseline"].replace(0, np.nan)
    )
    merged["tail_to_front_ratio_conditioned"] = (
        merged["rmse_tail_abs_steer_conditioned"]
        / merged["rmse_pre_tail_abs_steer_conditioned"].replace(0, np.nan)
    )
    merged["delta_tail_to_front_ratio"] = (
        merged["tail_to_front_ratio_conditioned"] - merged["tail_to_front_ratio_baseline"]
    )
    merged["delta_front_rmse"] = (
        merged["rmse_pre_tail_abs_steer_conditioned"]
        - merged["rmse_pre_tail_abs_steer_baseline"]
    )
    merged["delta_tail_rmse"] = (
        merged["rmse_tail_abs_steer_conditioned"] - merged["rmse_tail_abs_steer_baseline"]
    )
    merged["tail_minus_front_delta"] = (
        merged["delta_tail_rmse"] - merged["delta_front_rmse"]
    )
    merged["error_focus"] = np.where(
        merged["tail_minus_front_delta"] > 1e-12,
        "tail-driven",
        np.where(merged["tail_minus_front_delta"] < -1e-12, "front-driven", "balanced"),
    )

    for metric in [
        "boundary_slope_abs_err",
        "boundary_shift_abs_err",
        "tail_slope_abs_err",
        "peak_abs_amp_err",
    ]:
        merged[f"delta_{metric}"] = (
            merged[f"{metric}_conditioned"] - merged[f"{metric}_baseline"]
        )

    return merged


def infer_focus_label(delta_tail: float, delta_front: float, delta_ratio: float) -> str:
    if pd.isna(delta_tail) or pd.isna(delta_front):
        return "unknown"
    if delta_tail <= 0 and delta_front <= 0:
        if delta_tail > delta_front and delta_ratio > 0:
            return "net-improved-but-tail-relatively-heavier"
        if delta_front > delta_tail:
            return "net-improved-front-heavier"
        return "net-improved-balanced"
    if delta_tail > delta_front and delta_ratio > 0:
        return "tail-concentrated"
    if delta_tail > delta_front:
        return "tail-heavier-but-not-ratio-driven"
    if delta_front > delta_tail and delta_tail > 0:
        return "mixed-with-front-heavier"
    if delta_front > delta_tail:
        return "front-heavier"
    return "balanced"


def summarize_tail_focus(
    df: pd.DataFrame, group_cols: list[str], table_name: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(group_cols, dropna=False, sort=True)
    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_cols, group_key))
        tail_positive_share = float((group_df["delta_tail_rmse"] > 0).mean())
        front_positive_share = float((group_df["delta_front_rmse"] > 0).mean())
        tail_driven_share = float((group_df["error_focus"] == "tail-driven").mean())
        front_driven_share = float((group_df["error_focus"] == "front-driven").mean())
        delta_ratio_mean = float(group_df["delta_tail_to_front_ratio"].mean())
        delta_front_mean = float(group_df["delta_front_rmse"].mean())
        delta_tail_mean = float(group_df["delta_tail_rmse"].mean())
        rows.append(
            {
                "table_name": table_name,
                **row,
                "n_samples": int(len(group_df)),
                "baseline_front_rmse_mean": float(
                    group_df["rmse_pre_tail_abs_steer_baseline"].mean()
                ),
                "baseline_tail_rmse_mean": float(
                    group_df["rmse_tail_abs_steer_baseline"].mean()
                ),
                "conditioned_front_rmse_mean": float(
                    group_df["rmse_pre_tail_abs_steer_conditioned"].mean()
                ),
                "conditioned_tail_rmse_mean": float(
                    group_df["rmse_tail_abs_steer_conditioned"].mean()
                ),
                "baseline_ratio_mean": float(group_df["tail_to_front_ratio_baseline"].mean()),
                "conditioned_ratio_mean": float(
                    group_df["tail_to_front_ratio_conditioned"].mean()
                ),
                "delta_ratio_mean": delta_ratio_mean,
                "delta_front_rmse_mean": delta_front_mean,
                "delta_tail_rmse_mean": delta_tail_mean,
                "tail_minus_front_delta_mean": float(
                    group_df["tail_minus_front_delta"].mean()
                ),
                "tail_positive_share": tail_positive_share,
                "front_positive_share": front_positive_share,
                "tail_driven_share": tail_driven_share,
                "front_driven_share": front_driven_share,
                "mean_focus_label": infer_focus_label(
                    delta_tail=delta_tail_mean,
                    delta_front=delta_front_mean,
                    delta_ratio=delta_ratio_mean,
                ),
            }
        )
    return pd.DataFrame(rows)


def infer_boundary_mechanism(
    delta_shift_mean: float,
    delta_slope_mean: float,
    shift_threshold: float,
    slope_threshold: float,
) -> str:
    if pd.isna(delta_shift_mean) or pd.isna(delta_slope_mean):
        return "unknown"
    if (
        delta_shift_mean > shift_threshold
        and delta_shift_mean >= 2.0 * max(delta_slope_mean, slope_threshold)
    ):
        return "time-shift dominant"
    if (
        delta_slope_mean > slope_threshold
        and delta_slope_mean >= 2.0 * max(delta_shift_mean, shift_threshold)
    ):
        return "smoothing dominant"
    if delta_shift_mean > shift_threshold and delta_slope_mean > slope_threshold:
        return "mixed shift + smoothing"
    return "no clear worsening"


def classify_boundary_mode(
    delta_shift: pd.Series,
    delta_slope: pd.Series,
    shift_threshold: float,
    slope_threshold: float,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                (delta_shift > shift_threshold) & (delta_slope <= slope_threshold),
                (delta_shift > shift_threshold) & (delta_slope > slope_threshold),
                (delta_shift <= shift_threshold) & (delta_slope > slope_threshold),
            ],
            ["time-shift", "shift-plus-smoothing", "smoothing-only"],
            default="no-clear-worsening",
        ),
        index=delta_shift.index,
    )


def summarize_boundary_modes(
    df: pd.DataFrame, shift_threshold: float, slope_threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    working["boundary_mode"] = classify_boundary_mode(
        working["delta_boundary_shift_abs_err"],
        working["delta_boundary_slope_abs_err"],
        shift_threshold=shift_threshold,
        slope_threshold=slope_threshold,
    )

    rows: list[dict[str, object]] = []
    for morph, morph_df in working.groupby("eval_morphology_label", sort=True):
        delta_slope_mean = float(morph_df["delta_boundary_slope_abs_err"].mean())
        delta_shift_mean = float(morph_df["delta_boundary_shift_abs_err"].mean())
        rows.append(
            {
                "table_name": "morphology_boundary_summary",
                "eval_morphology_label": morph,
                "n_samples": int(len(morph_df)),
                "boundary_slope_baseline_mean": float(
                    morph_df["boundary_slope_abs_err_baseline"].mean()
                ),
                "boundary_slope_conditioned_mean": float(
                    morph_df["boundary_slope_abs_err_conditioned"].mean()
                ),
                "boundary_shift_baseline_mean": float(
                    morph_df["boundary_shift_abs_err_baseline"].mean()
                ),
                "boundary_shift_conditioned_mean": float(
                    morph_df["boundary_shift_abs_err_conditioned"].mean()
                ),
                "tail_slope_baseline_mean": float(
                    morph_df["tail_slope_abs_err_baseline"].mean()
                ),
                "tail_slope_conditioned_mean": float(
                    morph_df["tail_slope_abs_err_conditioned"].mean()
                ),
                "peak_amp_baseline_mean": float(
                    morph_df["peak_abs_amp_err_baseline"].mean()
                ),
                "peak_amp_conditioned_mean": float(
                    morph_df["peak_abs_amp_err_conditioned"].mean()
                ),
                "delta_boundary_slope_mean": delta_slope_mean,
                "delta_boundary_shift_mean": delta_shift_mean,
                "delta_tail_slope_mean": float(morph_df["delta_tail_slope_abs_err"].mean()),
                "delta_peak_amp_mean": float(morph_df["delta_peak_abs_amp_err"].mean()),
                "slope_worsening_share": float(
                    (morph_df["delta_boundary_slope_abs_err"] > 0).mean()
                ),
                "shift_worsening_share": float(
                    (morph_df["delta_boundary_shift_abs_err"] > 0).mean()
                ),
                "amplitude_worsening_share": float(
                    (morph_df["delta_peak_abs_amp_err"] > 0).mean()
                ),
                "mechanism_guess": infer_boundary_mechanism(
                    delta_shift_mean=delta_shift_mean,
                    delta_slope_mean=delta_slope_mean,
                    shift_threshold=shift_threshold,
                    slope_threshold=slope_threshold,
                ),
            }
        )

    counts = (
        working.groupby(["eval_morphology_label", "boundary_mode"], dropna=False)
        .size()
        .rename("n_samples")
        .reset_index()
    )
    counts["share_within_morphology"] = counts.groupby("eval_morphology_label")[
        "n_samples"
    ].transform(lambda s: s / s.sum())
    counts["table_name"] = "morphology_boundary_mode_counts"
    return pd.DataFrame(rows), counts


def safe_corr(series_a: pd.Series, series_b: pd.Series) -> float:
    valid = pd.concat([series_a, series_b], axis=1).dropna()
    if len(valid) < 2:
        return np.nan
    if float(valid.iloc[:, 0].std(ddof=1)) == 0.0 or float(
        valid.iloc[:, 1].std(ddof=1)
    ) == 0.0:
        return np.nan
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))


def build_boundary_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    metric_pairs = [
        ("boundary_shift_abs_err_conditioned", "boundary_slope_abs_err_conditioned"),
        ("boundary_shift_abs_err_conditioned", "peak_abs_amp_err_conditioned"),
        ("delta_boundary_shift_abs_err", "delta_boundary_slope_abs_err"),
        ("delta_boundary_shift_abs_err", "delta_peak_abs_amp_err"),
        ("delta_rmse_tail_abs_steer", "peak_abs_amp_err_conditioned"),
        ("delta_rmse_tail_abs_steer", "boundary_shift_abs_err_conditioned"),
    ]
    rows: list[dict[str, object]] = []
    for morph in ["single_lobe", "reverse_correction"]:
        morph_df = df.loc[df["eval_morphology_label"] == morph].copy()
        for metric_x, metric_y in metric_pairs:
            rows.append(
                {
                    "table_name": "morphology_boundary_correlations",
                    "eval_morphology_label": morph,
                    "metric_x": metric_x,
                    "metric_y": metric_y,
                    "n_samples": int(len(morph_df[[metric_x, metric_y]].dropna())),
                    "pearson_r": safe_corr(morph_df[metric_x], morph_df[metric_y]),
                }
            )
    return pd.DataFrame(rows)


def build_q1_single_lobe_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = df.loc[
        (df["latency_proxy_bucket"] == "Q1_fast")
        & (df["eval_morphology_label"] == "single_lobe")
    ].copy()

    if subset.empty:
        summary = pd.DataFrame(
            [
                {
                    "table_name": "q1_fast_single_lobe_summary",
                    "n_samples": 0,
                    "dominant_error_signal": "no samples",
                }
            ]
        )
        return summary, pd.DataFrame(columns=["table_name"])

    for col in [
        "peak_abs_amp_err_conditioned",
        "boundary_shift_abs_err_conditioned",
        "boundary_slope_abs_err_conditioned",
    ]:
        std = float(subset[col].std(ddof=1))
        if std == 0.0 or np.isnan(std):
            subset[f"{col}_z"] = 0.0
        else:
            subset[f"{col}_z"] = (subset[col] - subset[col].mean()) / std

    subset["dominant_dimension"] = np.select(
        [
            (
                subset["peak_abs_amp_err_conditioned_z"]
                >= subset["boundary_shift_abs_err_conditioned_z"]
            )
            & (
                subset["peak_abs_amp_err_conditioned_z"]
                >= subset["boundary_slope_abs_err_conditioned_z"]
            ),
            (
                subset["boundary_shift_abs_err_conditioned_z"]
                > subset["peak_abs_amp_err_conditioned_z"]
            )
            & (
                subset["boundary_shift_abs_err_conditioned_z"]
                >= subset["boundary_slope_abs_err_conditioned_z"]
            ),
        ],
        ["amplitude", "boundary_shift"],
        default="boundary_slope",
    )
    subset["sample_label"] = subset["sample_key"].map(short_sample_label)

    summary = pd.DataFrame(
        [
            {
                "table_name": "q1_fast_single_lobe_summary",
                "n_samples": int(len(subset)),
                "mean_peak_abs_amp_err_conditioned": float(
                    subset["peak_abs_amp_err_conditioned"].mean()
                ),
                "mean_boundary_shift_abs_err_conditioned": float(
                    subset["boundary_shift_abs_err_conditioned"].mean()
                ),
                "mean_boundary_slope_abs_err_conditioned": float(
                    subset["boundary_slope_abs_err_conditioned"].mean()
                ),
                "mean_delta_boundary_shift_abs_err": float(
                    subset["delta_boundary_shift_abs_err"].mean()
                ),
                "mean_delta_boundary_slope_abs_err": float(
                    subset["delta_boundary_slope_abs_err"].mean()
                ),
                "mean_delta_peak_abs_amp_err": float(
                    subset["delta_peak_abs_amp_err"].mean()
                ),
                "mean_delta_rmse_tail_abs_steer": float(
                    subset["delta_rmse_tail_abs_steer"].mean()
                ),
                "corr_tail_rmse_vs_amp": safe_corr(
                    subset["delta_rmse_tail_abs_steer"],
                    subset["peak_abs_amp_err_conditioned"],
                ),
                "corr_tail_rmse_vs_boundary_shift": safe_corr(
                    subset["delta_rmse_tail_abs_steer"],
                    subset["boundary_shift_abs_err_conditioned"],
                ),
                "corr_tail_rmse_vs_boundary_slope": safe_corr(
                    subset["delta_rmse_tail_abs_steer"],
                    subset["boundary_slope_abs_err_conditioned"],
                ),
                "dominant_error_signal": subset["dominant_dimension"]
                .value_counts()
                .idxmax(),
            }
        ]
    )

    case_columns = [
        "sample_key",
        "sample_label",
        "subj",
        "peak_abs_amp_err_conditioned",
        "boundary_shift_abs_err_conditioned",
        "boundary_slope_abs_err_conditioned",
        "shape_corr_conditioned",
        "delta_rmse_tail_abs_steer",
        "dominant_dimension",
    ]
    cases = subset[case_columns].sort_values(
        "delta_rmse_tail_abs_steer", ascending=False
    ).copy()
    cases.insert(0, "table_name", "q1_fast_single_lobe_cases")
    return summary, cases


def short_sample_label(sample_key: str) -> str:
    parts = str(sample_key).split("::")
    if len(parts) < 3:
        return str(sample_key)
    subj = parts[0]
    event_id = parts[2]
    return f"{subj}#{event_id}"


def render_markdown_table(df: pd.DataFrame, float_cols: list[str] | None = None) -> str:
    if df.empty:
        return "_No rows._"

    display = df.copy()
    float_cols = float_cols or []
    for col in display.columns:
        if col in float_cols:
            display[col] = display[col].map(
                lambda x: "" if pd.isna(x) else f"{float(x):.4f}"
            )
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))

    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    lines = [header, separator]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def format_subject_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="subj", fontsize=8, title_fontsize=9)


def make_boundary_scatter(df: pd.DataFrame, output_path: Path) -> None:
    subset_order = ["single_lobe", "reverse_correction"]
    colors = {"cwh": "#1f77b4", "gf": "#ff7f0e", "tyy": "#2ca02c"}
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), dpi=180, constrained_layout=True
    )

    for ax, morph in zip(axes, subset_order):
        morph_df = df.loc[df["eval_morphology_label"] == morph].copy()
        for subj, subj_df in morph_df.groupby("subj", sort=True):
            ax.scatter(
                subj_df["boundary_slope_abs_err_conditioned"],
                subj_df["boundary_shift_abs_err_conditioned"],
                s=38,
                alpha=0.72,
                label=subj,
                color=colors.get(subj, "#666666"),
                edgecolors="white",
                linewidths=0.4,
            )

        ax.set_title(f"{morph} conditioned boundary metrics")
        ax.set_xlabel("boundary_slope_abs_err_conditioned")
        ax.set_ylabel("boundary_shift_abs_err_conditioned")
        ax.grid(alpha=0.25, linewidth=0.6)
        format_subject_legend(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_q1_single_lobe_scatter(df: pd.DataFrame, output_path: Path) -> None:
    subset = df.loc[
        (df["latency_proxy_bucket"] == "Q1_fast")
        & (df["eval_morphology_label"] == "single_lobe")
    ].copy()
    if subset.empty:
        return

    colors = {"cwh": "#1f77b4", "gf": "#ff7f0e", "tyy": "#2ca02c"}
    subset["sample_label"] = subset["sample_key"].map(short_sample_label)
    shape_min = float(subset["shape_corr_conditioned"].min())
    shape_max = float(subset["shape_corr_conditioned"].max())
    if shape_max == shape_min:
        subset["point_size"] = 140.0
    else:
        subset["point_size"] = 80.0 + 240.0 * (
            (subset["shape_corr_conditioned"] - shape_min) / (shape_max - shape_min)
        )

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=180, constrained_layout=True)
    for subj, subj_df in subset.groupby("subj", sort=True):
        ax.scatter(
            subj_df["peak_abs_amp_err_conditioned"],
            subj_df["boundary_shift_abs_err_conditioned"],
            s=subj_df["point_size"],
            alpha=0.75,
            label=subj,
            color=colors.get(subj, "#666666"),
            edgecolors="white",
            linewidths=0.5,
        )

    annotate_df = subset.nlargest(6, "delta_rmse_tail_abs_steer")
    for row in annotate_df.itertuples(index=False):
        ax.annotate(
            row.sample_label,
            (
                float(row.peak_abs_amp_err_conditioned),
                float(row.boundary_shift_abs_err_conditioned),
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
        )

    ax.set_title("Q1_fast x single_lobe: amplitude vs boundary shift")
    ax.set_xlabel("peak_abs_amp_err_conditioned")
    ax.set_ylabel("boundary_shift_abs_err_conditioned")
    ax.grid(alpha=0.25, linewidth=0.6)
    format_subject_legend(ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_report_markdown(
    analysis_df: pd.DataFrame,
    alignment_notes: list[str],
    tail_bucket_summary: pd.DataFrame,
    tail_morph_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    boundary_modes: pd.DataFrame,
    boundary_corr: pd.DataFrame,
    q1_single_lobe_summary: pd.DataFrame,
    q1_single_lobe_cases: pd.DataFrame,
    attribution_path: Path,
    baseline_path: Path,
    conditioned_path: Path,
    comparison_path: Path,
    boundary_fig_path: Path,
    q1_single_lobe_fig_path: Path,
) -> str:
    q1_row = tail_bucket_summary.loc[
        tail_bucket_summary["latency_proxy_bucket"] == "Q1_fast"
    ].iloc[0]
    q1_single_lobe_row = tail_morph_summary.loc[
        (tail_morph_summary["latency_proxy_bucket"] == "Q1_fast")
        & (tail_morph_summary["eval_morphology_label"] == "single_lobe")
    ].iloc[0]
    q1_reverse_row = tail_morph_summary.loc[
        (tail_morph_summary["latency_proxy_bucket"] == "Q1_fast")
        & (tail_morph_summary["eval_morphology_label"] == "reverse_correction")
    ].iloc[0]
    single_lobe_boundary = boundary_summary.loc[
        boundary_summary["eval_morphology_label"] == "single_lobe"
    ].iloc[0]
    reverse_boundary = boundary_summary.loc[
        boundary_summary["eval_morphology_label"] == "reverse_correction"
    ].iloc[0]
    q1_boundary = q1_single_lobe_summary.iloc[0]

    q1_top_cases = q1_single_lobe_cases.head(6).copy()
    q1_top_cases = q1_top_cases[
        [
            "sample_label",
            "subj",
            "peak_abs_amp_err_conditioned",
            "boundary_shift_abs_err_conditioned",
            "boundary_slope_abs_err_conditioned",
            "shape_corr_conditioned",
            "delta_rmse_tail_abs_steer",
            "dominant_dimension",
        ]
    ]

    boundary_mode_report = boundary_modes.loc[
        boundary_modes["eval_morphology_label"].isin(
            ["single_lobe", "reverse_correction"]
        )
    ].copy()
    boundary_mode_report = boundary_mode_report[
        ["eval_morphology_label", "boundary_mode", "n_samples", "share_within_morphology"]
    ].sort_values(["eval_morphology_label", "n_samples"], ascending=[True, False])

    boundary_corr_report = boundary_corr[
        ["eval_morphology_label", "metric_x", "metric_y", "pearson_r"]
    ].copy()

    lines = [
        "# Approximate Timestep & Boundary Analysis Using Existing CSVs",
        "",
        "## Scope",
        f"- Attribution master table: `{attribution_path}`",
        f"- Baseline sample metrics: `{baseline_path}`",
        f"- Conditioned sample metrics: `{conditioned_path}`",
        f"- Sample-level comparison: `{comparison_path}`",
        f"- Samples analyzed: `{len(analysis_df)}`",
        f"- Latency buckets: {', '.join(sorted(analysis_df['latency_proxy_bucket'].unique()))}",
        "",
        "## Input Consistency Check",
        "- The baseline / conditioned sample-metric CSVs were aligned against `sample_level_comparison.csv` on `sample_key` before analysis.",
        *[f"- {note}" for note in alignment_notes],
        "",
        "## Key Answers",
        (
            f"- `Q1_fast` degradation is **not purely tail-concentrated overall**. "
            f"Its mean front RMSE delta is `{q1_row['delta_front_rmse_mean']:.4f}` while "
            f"its mean tail RMSE delta is `{q1_row['delta_tail_rmse_mean']:.4f}`, and the mean "
            f"tail/front ratio shifts from `{q1_row['baseline_ratio_mean']:.4f}` to "
            f"`{q1_row['conditioned_ratio_mean']:.4f}` (`delta={q1_row['delta_ratio_mean']:.4f}`)."
        ),
        (
            f"- The strongest tail-focused worsening sits in `Q1_fast x single_lobe`: front delta "
            f"`{q1_single_lobe_row['delta_front_rmse_mean']:.4f}`, tail delta "
            f"`{q1_single_lobe_row['delta_tail_rmse_mean']:.4f}`, ratio delta "
            f"`{q1_single_lobe_row['delta_ratio_mean']:.4f}`, tail-driven share "
            f"`{q1_single_lobe_row['tail_driven_share']:.4f}`."
        ),
        (
            f"- `single_lobe` boundary worsening looks **time-shift dominant, not slope-flattening dominant**: "
            f"mean `delta_boundary_shift_abs_err={single_lobe_boundary['delta_boundary_shift_mean']:.4f}` "
            f"versus mean `delta_boundary_slope_abs_err={single_lobe_boundary['delta_boundary_slope_mean']:.4f}`."
        ),
        (
            f"- `reverse_correction` shows the same direction, only weaker: "
            f"mean `delta_boundary_shift_abs_err={reverse_boundary['delta_boundary_shift_mean']:.4f}` "
            f"versus mean `delta_boundary_slope_abs_err={reverse_boundary['delta_boundary_slope_mean']:.4f}`."
        ),
        (
            f"- In `Q1_fast x single_lobe`, the worst cases are more **amplitude-driven** than boundary-driven: "
            f"`corr(delta_tail_rmse, peak_abs_amp_err_conditioned)={q1_boundary['corr_tail_rmse_vs_amp']:.4f}` "
            f"versus `corr(delta_tail_rmse, boundary_shift_abs_err_conditioned)="
            f"{q1_boundary['corr_tail_rmse_vs_boundary_shift']:.4f}`."
        ),
        "",
        "## Part A: Front vs Tail Error Concentration",
        render_markdown_table(
            tail_bucket_summary[
                [
                    "latency_proxy_bucket",
                    "n_samples",
                    "baseline_front_rmse_mean",
                    "baseline_tail_rmse_mean",
                    "conditioned_front_rmse_mean",
                    "conditioned_tail_rmse_mean",
                    "baseline_ratio_mean",
                    "conditioned_ratio_mean",
                    "delta_ratio_mean",
                    "delta_front_rmse_mean",
                    "delta_tail_rmse_mean",
                    "mean_focus_label",
                ]
            ],
            float_cols=REPORT_FLOAT_COLUMNS,
        ),
        "",
        "### Morphology x Latency Interaction",
        render_markdown_table(
            tail_morph_summary[
                [
                    "eval_morphology_label",
                    "latency_proxy_bucket",
                    "n_samples",
                    "delta_front_rmse_mean",
                    "delta_tail_rmse_mean",
                    "delta_ratio_mean",
                    "tail_driven_share",
                    "front_driven_share",
                    "mean_focus_label",
                ]
            ],
            float_cols=REPORT_FLOAT_COLUMNS,
        ),
        "",
        "### Interpretation",
        (
            f"- `Q1_fast` overall is best described as **mixed with front heavier**, not as a tail-only failure: "
            f"`delta_front_rmse_mean={q1_row['delta_front_rmse_mean']:.4f}` is larger than "
            f"`delta_tail_rmse_mean={q1_row['delta_tail_rmse_mean']:.4f}`, and the average tail/front ratio declines."
        ),
        (
            f"- `Q1_fast x reverse_correction` does not support a worsening story at all on mean RMSE: "
            f"front delta `{q1_reverse_row['delta_front_rmse_mean']:.4f}`, tail delta "
            f"`{q1_reverse_row['delta_tail_rmse_mean']:.4f}`."
        ),
        (
            f"- The localized tail-worsening story is concentrated in `Q1_fast x single_lobe`, where the tail delta "
            f"exceeds the front delta by `{q1_single_lobe_row['tail_minus_front_delta_mean']:.4f}` on average."
        ),
        "",
        "## Part B: Boundary Smoothing vs Shifting",
        render_markdown_table(
            boundary_summary[
                [
                    "eval_morphology_label",
                    "n_samples",
                    "boundary_slope_baseline_mean",
                    "boundary_slope_conditioned_mean",
                    "boundary_shift_baseline_mean",
                    "boundary_shift_conditioned_mean",
                    "delta_boundary_slope_mean",
                    "delta_boundary_shift_mean",
                    "delta_peak_amp_mean",
                    "mechanism_guess",
                ]
            ],
            float_cols=REPORT_FLOAT_COLUMNS,
        ),
        "",
        "### Boundary-Mode Counts",
        render_markdown_table(
            boundary_mode_report,
            float_cols=REPORT_FLOAT_COLUMNS,
        ),
        "",
        "### Boundary Correlations",
        render_markdown_table(
            boundary_corr_report,
            float_cols=REPORT_FLOAT_COLUMNS,
        ),
        "",
        "### Interpretation",
        (
            f"- `single_lobe` shows a large mean shift increase (`{single_lobe_boundary['delta_boundary_shift_mean']:.4f}`) "
            f"with only a small mean slope increase (`{single_lobe_boundary['delta_boundary_slope_mean']:.4f}`), "
            "so the average picture is time-shift dominant."
        ),
        (
            f"- `reverse_correction` shows the same sign pattern: shift increase "
            f"`{reverse_boundary['delta_boundary_shift_mean']:.4f}` exceeds slope increase "
            f"`{reverse_boundary['delta_boundary_slope_mean']:.4f}`."
        ),
        (
            "- Both morphologies still contain a substantial `shift-plus-smoothing` subset, so the result is better read as "
            "`time-shift dominant with a mixed secondary smoothing component`, not as a perfectly pure mode."
        ),
        "",
        "## Part C: Q1_fast x single_lobe Scatter",
        f"- Boundary scatter figure: `{boundary_fig_path}`",
        f"- Q1_fast x single_lobe amplitude-vs-boundary scatter: `{q1_single_lobe_fig_path}`",
        "",
        render_markdown_table(
            q1_top_cases,
            float_cols=REPORT_FLOAT_COLUMNS,
        ),
        "",
        "### Interpretation",
        (
            f"- The subset mean `delta_boundary_shift_abs_err` is positive (`{q1_boundary['mean_delta_boundary_shift_abs_err']:.4f}`), "
            f"but mean `delta_boundary_slope_abs_err` is slightly negative (`{q1_boundary['mean_delta_boundary_slope_abs_err']:.4f}`), "
            "which argues against slope flattening as the main driver in this exact intersection."
        ),
        (
            f"- The strongest relationship to tail degradation is amplitude error, not boundary error: "
            f"`corr_tail_rmse_vs_amp={q1_boundary['corr_tail_rmse_vs_amp']:.4f}`, "
            f"`corr_tail_rmse_vs_boundary_shift={q1_boundary['corr_tail_rmse_vs_boundary_shift']:.4f}`, "
            f"`corr_tail_rmse_vs_boundary_slope={q1_boundary['corr_tail_rmse_vs_boundary_slope']:.4f}`."
        ),
        (
            "- The worst individual rows are mixed, but the largest tail-RMSE failures are led by high amplitude error "
            "more often than by exceptionally large boundary shift."
        ),
        "",
        "## Bottom Line",
        "- Existing CSV metrics are sufficient to approximate the missing raw-sequence analysis well enough to answer the immediate diagnostic questions.",
        "- The global `Q1_fast` issue is broader than a tail-only phenomenon, but the `Q1_fast x single_lobe` slice is genuinely tail-heavier.",
        "- Boundary worsening is better described as time-shift dominant than slope-flattening dominant, especially in `Q1_fast x single_lobe`.",
        "- For the worst `Q1_fast x single_lobe` rows, amplitude mismatch is the strongest companion of tail degradation, so boundary-only fixes would likely miss the most severe failures.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    attribution, baseline, conditioned, comparison = load_inputs(
        attribution_path=args.attribution,
        baseline_path=args.baseline,
        conditioned_path=args.conditioned,
        comparison_path=args.comparison,
    )
    alignment_notes = validate_formal_eval_alignment(
        baseline=baseline,
        conditioned=conditioned,
        comparison=comparison,
    )
    analysis_df = build_analysis_frame(attribution=attribution, comparison=comparison)

    tail_bucket_summary = summarize_tail_focus(
        analysis_df, ["latency_proxy_bucket"], "latency_bucket_tail_focus"
    )
    tail_morph_summary = summarize_tail_focus(
        analysis_df,
        ["eval_morphology_label", "latency_proxy_bucket"],
        "latency_bucket_morphology_tail_focus",
    )
    boundary_summary, boundary_modes = summarize_boundary_modes(
        analysis_df,
        shift_threshold=args.shift_threshold,
        slope_threshold=args.slope_threshold,
    )
    boundary_corr = build_boundary_correlation_table(analysis_df)
    q1_single_lobe_summary, q1_single_lobe_cases = build_q1_single_lobe_tables(
        analysis_df
    )

    make_boundary_scatter(analysis_df, args.boundary_fig)
    make_q1_single_lobe_scatter(analysis_df, args.q1_single_lobe_fig)

    output_tables = [
        tail_bucket_summary,
        tail_morph_summary,
        boundary_summary,
        boundary_modes,
        boundary_corr,
        q1_single_lobe_summary,
        q1_single_lobe_cases,
    ]
    output_df = pd.concat(output_tables, ignore_index=True, sort=False)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    report_md = build_report_markdown(
        analysis_df=analysis_df,
        alignment_notes=alignment_notes,
        tail_bucket_summary=tail_bucket_summary,
        tail_morph_summary=tail_morph_summary,
        boundary_summary=boundary_summary,
        boundary_modes=boundary_modes,
        boundary_corr=boundary_corr,
        q1_single_lobe_summary=q1_single_lobe_summary,
        q1_single_lobe_cases=q1_single_lobe_cases,
        attribution_path=args.attribution,
        baseline_path=args.baseline,
        conditioned_path=args.conditioned,
        comparison_path=args.comparison,
        boundary_fig_path=args.boundary_fig,
        q1_single_lobe_fig_path=args.q1_single_lobe_fig,
    )
    args.output_md.write_text(report_md, encoding="utf-8")

    print(f"Wrote CSV report: {args.output_csv}")
    print(f"Wrote Markdown report: {args.output_md}")
    print(f"Wrote figure: {args.boundary_fig}")
    print(f"Wrote figure: {args.q1_single_lobe_fig}")
    print(f"Analyzed samples: {len(analysis_df)}")


if __name__ == "__main__":
    main()

