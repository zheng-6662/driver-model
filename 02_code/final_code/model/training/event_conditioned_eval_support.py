from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baseline_eval_primary_aux import (
    build_trajectory_selection_summary,
    compute_morphology_metric_rows,
    compute_trajectory_sample_metrics,
    compute_weighted_metrics,
)


D3_MANIFEST = Path(__file__).resolve().parent / "protocol_d3_response_aligned_extended_v1" / "sample_manifest.csv"
STRUCTURE_HEAVY_MORPH = {"reverse_correction", "multi_correction"}
PRIMARY_RMSE_SCALE = 0.45


def annotate_event_meta(meta_df: pd.DataFrame, y_pool: np.ndarray, mask_pool: np.ndarray) -> pd.DataFrame:
    work = meta_df.reset_index(drop=True).copy()
    morph_df, _ = compute_morphology_metric_rows(
        pred=y_pool,
        true=y_pool,
        mask=mask_pool,
        meta_df=work,
        split_name="meta",
        seed=0,
    )
    work["eval_morphology_label"] = morph_df["eval_morphology_label"].astype(str)
    road_type = work.get("road_type_anchor", pd.Series(["unknown"] * len(work))).astype(str)
    is_curve = road_type.eq("curve")
    heavy = is_curve | work["eval_morphology_label"].isin(STRUCTURE_HEAVY_MORPH)
    work["structure_heavy"] = heavy.astype(int)
    work["structure_slice"] = np.where(heavy, "structure_heavy", "non_structure_heavy")
    work["reversal_slice"] = np.where(
        work["eval_morphology_label"].isin(STRUCTURE_HEAVY_MORPH),
        "reversal",
        "non_reversal",
    )

    if D3_MANIFEST.exists():
        d3 = pd.read_csv(
            D3_MANIFEST,
            usecols=["vehicle_file", "anchor_idx", "episode_id", "mechanism_tag"],
        )
        anchor_lookup = d3.drop_duplicates(subset=["vehicle_file", "anchor_idx"]).rename(
            columns={"mechanism_tag": "d3_mechanism_tag_anchor"}
        )
        episode_lookup = d3.drop_duplicates(subset=["vehicle_file", "episode_id"]).rename(
            columns={"mechanism_tag": "d3_mechanism_tag_episode"}
        )
        work = work.merge(
            anchor_lookup[["vehicle_file", "anchor_idx", "d3_mechanism_tag_anchor"]],
            on=["vehicle_file", "anchor_idx"],
            how="left",
        )
        if "episode_id" in work.columns:
            work = work.merge(
                episode_lookup[["vehicle_file", "episode_id", "d3_mechanism_tag_episode"]],
                on=["vehicle_file", "episode_id"],
                how="left",
            )
        else:
            work["d3_mechanism_tag_episode"] = np.nan
        effective = work.get("mechanism_tag", pd.Series(["unknown"] * len(work))).astype(str)
        effective = (
            effective.mask(effective.eq("unknown"), other=np.nan)
            .fillna(work["d3_mechanism_tag_anchor"])
            .fillna(work["d3_mechanism_tag_episode"])
            .fillna("unknown")
        )
        work["effective_mechanism_tag"] = effective.astype(str)
    else:
        work["effective_mechanism_tag"] = work.get("mechanism_tag", pd.Series(["unknown"] * len(work))).astype(str)

    work["interaction_slice"] = np.where(
        work["effective_mechanism_tag"].eq("traffic_interaction"),
        "interaction",
        np.where(work["effective_mechanism_tag"].eq("unknown"), "unknown", "non_interaction"),
    )
    return work


def build_structure_aware_selection_summary(
    weighted_metrics: dict[str, Any],
    sample_df: pd.DataFrame,
    subset_name: str,
) -> dict[str, float]:
    summary = build_trajectory_selection_summary(weighted_metrics, sample_df, subset_name=subset_name)
    primary_rmse_score = float(summary["overall_primary_steer_rmse"]) / PRIMARY_RMSE_SCALE
    selection_score = (
        0.25 * primary_rmse_score
        + 0.35 * float(summary["tail_score"])
        + 0.20 * float(summary["trend_score"])
        + 0.10 * float(summary["turning_score"])
        + 0.10 * float(summary["continuity_score"])
    )
    summary = dict(summary)
    summary["primary_rmse_score"] = float(primary_rmse_score)
    summary["selection_score"] = float(selection_score)
    return summary


def structure_aware_selection_key(summary: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(summary["selection_score"]),
        float(summary["trajectory_score"]),
        float(summary["turning_score"]),
        float(summary["overall_primary_steer_rmse"]),
    )


def build_primary_selection_bundle(
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    ctx_raw: np.ndarray,
    meta_df: pd.DataFrame,
    split_name: str,
    seed: int,
) -> dict[str, Any]:
    work = meta_df.reset_index(drop=True).copy()
    weighted = compute_weighted_metrics(pred, true, mask)
    sample_df = compute_trajectory_sample_metrics(
        meta_df=work,
        pred=pred,
        true=true,
        mask=mask,
        ctx_raw=ctx_raw,
        split_name=split_name,
        seed=seed,
    )
    merge_cols = [
        col
        for col in (
            "sample_key",
            "interaction_slice",
            "reversal_slice",
            "effective_mechanism_tag",
            "structure_slice",
            "structure_heavy",
        )
        if col in work.columns
    ]
    if merge_cols:
        sample_df = sample_df.merge(
            work[merge_cols].drop_duplicates("sample_key"),
            on="sample_key",
            how="left",
            suffixes=("", "_meta"),
        )
        for col in ("structure_slice", "structure_heavy"):
            meta_col = f"{col}_meta"
            if meta_col in sample_df.columns:
                sample_df[col] = sample_df[meta_col]
                sample_df = sample_df.drop(columns=[meta_col])

    primary_idx = work["phase_type"].astype(str).eq("primary").to_numpy() if "phase_type" in work.columns else np.ones(len(work), dtype=bool)
    if primary_idx.any():
        primary_weighted = compute_weighted_metrics(pred[primary_idx], true[primary_idx], mask[primary_idx])
        primary_sample_df = sample_df[sample_df["phase_type"].astype(str).eq("primary")].reset_index(drop=True)
    else:
        primary_weighted = weighted
        primary_sample_df = sample_df

    selection_summary = build_structure_aware_selection_summary(primary_weighted, primary_sample_df, subset_name="primary")
    return {
        "weighted": weighted,
        "sample_df": sample_df,
        "primary_weighted": primary_weighted,
        "primary_sample_df": primary_sample_df,
        "selection_summary": selection_summary,
    }
