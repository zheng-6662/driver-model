from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from baseline_eval_primary_aux import (
    build_trajectory_selection_summary,
    build_trajectory_subset_rows,
    compute_group_metric_rows,
    compute_morphology_metric_rows,
    compute_trajectory_sample_metrics,
    compute_weighted_metrics,
    flatten_weighted_metrics,
)
from event_conditioned_baseline_model import (
    EventConditionedDataset,
    EventConditionedTrajectoryModel,
    build_event_schema_targets,
)
from event_targets import EventTargetConfig, sequence_to_event_targets
from future_steer_speed_subjectsplit_masked import (
    DEFAULT_BATCH_SIZE,
    FS,
    save_json,
)
from run_event_conditioned_trajectory_baseline import build_sample_bundle_from_manifest


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_RUN = (
    Path("F:/data_set_process/data_process/tmp/single_output_d3_runs")
    / "EXP2_ALLPHASE_V2_CONTEXT_FULL2S_TRUE2S_SUP_20260324_224343"
)
D3_MANIFEST = THIS_DIR / "protocol_d3_response_aligned_extended_v1" / "sample_manifest.csv"
REPORT_ROOT = (
    Path("F:/data_set_process/data_process/reports/event_plus_conditioned_trajectory_baseline_20260326")
    / "task_C_eval_visualization"
)
BASELINE_SEQUENCE_OUTPUT = Path("F:/data_set_process/data_process/reports/baseline_prediction_sequences.npz")
CONDITIONED_SEQUENCE_OUTPUT = Path("F:/data_set_process/data_process/reports/conditioned_v2_prediction_sequences.npz")
SEQUENCE_CHANNEL_NAMES = np.asarray(["steer_rel", "speed_delta"], dtype="<U32")
SEQUENCE_CHANNEL_NOTE = np.asarray(
    "Restored conditioned trajectory checkpoints emit 2 channels (steer_rel, speed_delta); "
    "the handoff requested 3 channels, but yawrate/ay are not predicted by this model family."
)
STRUCTURE_HEAVY_MORPH = {"reverse_correction", "multi_correction"}
TRAJ_COMPARE_METRICS = [
    "rmse_2s_abs_steer",
    "rmse_tail_abs_steer",
    "tail_trend_corr",
    "turning_count_abs_err",
    "peak_time_abs_err_s",
    "boundary_shift_abs_err",
]
EVENT_NAMES = ("first_major_turn_onset", "first_reversal", "main_peak")
EVENT_FIELD_MAP = {
    "first_major_turn_onset": {
        "has": "first_major_turn_onset_has",
        "idx": "first_major_turn_onset_idx",
        "direction": "first_major_turn_direction",
    },
    "first_reversal": {
        "has": "first_reversal_has",
        "idx": "first_reversal_idx",
        "direction": None,
    },
    "main_peak": {
        "has": None,
        "idx": "main_peak_idx",
        "direction": "main_peak_direction",
    },
}


def _choose_manifest(run_root: Path, preferred_name: str) -> Path:
    candidate = run_root / preferred_name
    if candidate.exists():
        return candidate
    fallback = run_root / "sample_manifest_with_split.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"cannot find manifest under {run_root}")


def _annotate_meta(meta_df: pd.DataFrame, y_pool: np.ndarray, mask_pool: np.ndarray) -> pd.DataFrame:
    work = meta_df.reset_index(drop=True).copy()
    for col in (
        "eval_morphology_label",
        "structure_heavy",
        "structure_slice",
        "reversal_slice",
        "d3_mechanism_tag_anchor",
        "d3_mechanism_tag_episode",
        "effective_mechanism_tag",
        "interaction_slice",
    ):
        if col in work.columns:
            work = work.drop(columns=[col])
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


def _event_rows_from_sequences(
    meta_df: pd.DataFrame,
    steer_rel: np.ndarray,
    mask: np.ndarray,
    config: EventTargetConfig,
    prefix: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, meta_row in meta_df.reset_index(drop=True).iterrows():
        valid_len = int(mask[idx].sum())
        targets = sequence_to_event_targets(steer_rel[idx, :, 0], valid_len, config=config)
        row = {
            "sample_key": str(meta_row.get("sample_key", idx)),
            "split": str(meta_row.get("split", "unknown")),
            "phase_type": str(meta_row.get("phase_type", "unknown")),
            "road_type_anchor": str(meta_row.get("road_type_anchor", "unknown")),
            "eval_morphology_label": str(meta_row.get("eval_morphology_label", "unknown")),
            "structure_slice": str(meta_row.get("structure_slice", "unknown")),
            "interaction_slice": str(meta_row.get("interaction_slice", "unknown")),
            "effective_mechanism_tag": str(meta_row.get("effective_mechanism_tag", "unknown")),
            "valid_future_len": valid_len,
        }
        for key, value in targets.items():
            row[f"{prefix}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _event_comparison_summary(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = true_df.merge(pred_df, on="sample_key", suffixes=("_true", "_pred"))
    rows: list[dict[str, Any]] = []
    for event_name in EVENT_NAMES:
        field_map = EVENT_FIELD_MAP[event_name]
        true_has_col = f"true_{field_map['has']}" if field_map["has"] else None
        pred_has_col = f"pred_{field_map['has']}" if field_map["has"] else None
        true_idx_col = f"true_{field_map['idx']}"
        pred_idx_col = f"pred_{field_map['idx']}"
        true_dir_col = f"true_{field_map['direction']}" if field_map["direction"] else None
        pred_dir_col = f"pred_{field_map['direction']}" if field_map["direction"] else None

        if true_has_col and true_has_col in merged.columns:
            has_match = (merged[true_has_col] == merged[pred_has_col]).astype(float)
            support_true = merged[true_has_col].astype(float)
            matched = (merged[true_has_col].astype(float) > 0) & (merged[pred_has_col].astype(float) > 0)
        else:
            has_match = pd.Series(np.ones(len(merged), dtype=float))
            support_true = pd.Series(np.ones(len(merged), dtype=float))
            matched = pd.Series(np.ones(len(merged), dtype=bool))

        time_err = pd.Series(np.nan, index=merged.index, dtype=float)
        if true_idx_col in merged.columns:
            time_err.loc[matched] = (
                (merged.loc[matched, pred_idx_col] - merged.loc[matched, true_idx_col]).abs() / FS
            )

        direction_match = pd.Series(np.nan, index=merged.index, dtype=float)
        if true_dir_col and true_dir_col in merged.columns:
            direction_match.loc[matched] = (
                merged.loc[matched, pred_dir_col] == merged.loc[matched, true_dir_col]
            ).astype(float)

        per_sample = pd.DataFrame(
            {
                "sample_key": merged["sample_key"],
                "model_name": model_name,
                "event_name": event_name,
                "presence_acc": has_match,
                "time_abs_err_s": time_err,
                "direction_acc": direction_match,
                "support_true": support_true,
                "support_matched": matched.astype(int),
                "interaction_slice": merged["interaction_slice_true"],
                "structure_slice": merged["structure_slice_true"],
            }
        )
        rows.append(per_sample)

    sample_level = pd.concat(rows, ignore_index=True)
    summary = (
        sample_level.groupby("event_name", dropna=False)
        .agg(
            model_name=("model_name", "first"),
            presence_acc=("presence_acc", "mean"),
            time_abs_err_s=("time_abs_err_s", "mean"),
            direction_acc=("direction_acc", "mean"),
            support_true=("support_true", "sum"),
            support_matched=("support_matched", "sum"),
        )
        .reset_index()
    )
    return sample_level, summary


def _trajectory_subset_summary(
    trajectory_subset_rows: list[dict[str, Any]],
    model_name: str,
) -> pd.DataFrame:
    df = pd.DataFrame(trajectory_subset_rows)
    if df.empty:
        return df
    df.insert(0, "model_name", model_name)
    return df


def _subset_rows_for_family(sample_df: pd.DataFrame, split: str, seed: int, family: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        col
        for col in sample_df.columns
        if col
        not in {
            "split",
            "seed",
            "local_idx",
            "subj",
            "sample_key",
            "phase_type",
            "road_type_anchor",
            "mechanism_tag",
            "is_curve",
            "structure_slice",
            "structure_heavy",
            "valid_future_len",
            "eval_morphology_label",
        }
        and np.issubdtype(sample_df[col].dtype, np.number)
    ]
    for label, part in sample_df.groupby(family, dropna=False):
        row: dict[str, Any] = {
            "split": split,
            "seed": int(seed),
            "subset_family": family,
            "subset_name": str(label),
            "sample_count": int(len(part)),
        }
        for col in metric_cols:
            row[col] = float(part[col].mean()) if len(part) > 0 else float("nan")
        rows.append(row)
    return rows


def _pairwise_summary(
    baseline_df: pd.DataFrame,
    conditioned_df: pd.DataFrame,
    id_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    merged = baseline_df.merge(conditioned_df, on=id_cols, suffixes=("_baseline", "_conditioned"))
    for col in value_cols:
        base_col = f"{col}_baseline"
        cond_col = f"{col}_conditioned"
        if base_col in merged.columns and cond_col in merged.columns:
            merged[f"delta_{col}"] = merged[cond_col] - merged[base_col]
    return merged


def _compare_score(row: pd.Series) -> float:
    return float(
        -0.35 * row.get("delta_rmse_2s_abs_steer", 0.0)
        - 0.25 * row.get("delta_rmse_tail_abs_steer", 0.0)
        - 0.15 * row.get("delta_turning_count_abs_err", 0.0)
        - 0.15 * row.get("delta_peak_time_abs_err_s", 0.0)
        - 0.10 * row.get("delta_boundary_shift_abs_err", 0.0)
        + 0.15 * row.get("delta_tail_trend_corr", 0.0)
    )


def _hard_score(row: pd.Series) -> float:
    return float(
        row.get("rmse_2s_abs_steer_conditioned", 0.0)
        + 0.8 * row.get("rmse_tail_abs_steer_conditioned", 0.0)
        + 0.4 * row.get("turning_count_abs_err_conditioned", 0.0)
        + 0.2 * row.get("peak_time_abs_err_s_conditioned", 0.0)
    )


def _event_time_map(event_row: pd.Series) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for event_name in EVENT_NAMES:
        idx_col = f"true_{event_name}_idx"
        if idx_col not in event_row or int(event_row[idx_col]) < 0:
            mapping[event_name] = np.nan
        else:
            mapping[event_name] = float(event_row[idx_col]) / FS
    return mapping


def _plot_compare_case(
    out_path: Path,
    title: str,
    times: np.ndarray,
    truth_abs: np.ndarray,
    baseline_abs: np.ndarray,
    conditioned_abs: np.ndarray,
    event_time_map: dict[str, float],
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.plot(times, np.degrees(truth_abs), color="#1f77b4", linewidth=2.0, label="True")
    ax.plot(times, np.degrees(baseline_abs), color="#d62728", linewidth=1.3, label="Unconditional")
    ax.plot(times, np.degrees(conditioned_abs), color="#2ca02c", linewidth=1.5, label="Event-conditioned")
    for event_name, event_time in event_time_map.items():
        if np.isnan(event_time):
            continue
        ax.axvline(event_time, color="#444444", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.text(
            event_time,
            ax.get_ylim()[1] * 0.92,
            event_name,
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color="#444444",
        )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Steering (deg)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_overview_mosaic(
    out_path: Path,
    selected: list[dict[str, Any]],
) -> None:
    cols = 2
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8.2, rows * 3.8), squeeze=False)
    axes_flat = axes.flatten()
    for ax, item in zip(axes_flat, selected):
        ax.plot(item["times"], np.degrees(item["truth_abs"]), color="#1f77b4", linewidth=2.0, label="True")
        ax.plot(item["times"], np.degrees(item["baseline_abs"]), color="#d62728", linewidth=1.2, label="Unconditional")
        ax.plot(item["times"], np.degrees(item["conditioned_abs"]), color="#2ca02c", linewidth=1.4, label="Event-conditioned")
        for event_name, event_time in item["event_time_map"].items():
            if np.isnan(event_time):
                continue
            ax.axvline(event_time, color="#444444", linestyle=":", linewidth=0.9, alpha=0.7)
        ax.set_title(item["title"], fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Steering (deg)")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=7, loc="best")
    for ax in axes_flat[len(selected) :]:
        fig.delaxes(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _select_cases(compare_df: pd.DataFrame) -> pd.DataFrame:
    work = compare_df.copy()
    work["compare_gain_score"] = work.apply(_compare_score, axis=1)
    work["hard_case_score"] = work.apply(_hard_score, axis=1)

    frames: list[pd.DataFrame] = []
    if not work.empty:
        frames.append(work.sort_values("compare_gain_score", ascending=False).head(2).assign(selection_tag="good_case"))
        frames.append(work.sort_values("hard_case_score", ascending=False).head(2).assign(selection_tag="hard_case"))
    interaction = work[work["interaction_slice"].astype(str) == "interaction"]
    if not interaction.empty:
        frames.append(
            interaction.sort_values("compare_gain_score", ascending=False).head(2).assign(selection_tag="interaction_case")
        )
    reversal = work[work["reversal_slice"].astype(str) == "reversal"]
    if not reversal.empty:
        frames.append(
            reversal.sort_values("compare_gain_score", ascending=False).head(2).assign(selection_tag="reversal_case")
        )
    if not frames:
        return work.head(0).copy()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["sample_key", "selection_tag"]).reset_index(drop=True)


def _baseline_arrays(
    run_root: Path,
    seed: int,
    split: str,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    run_summary = pd.read_json(run_root / "run_summary.json", typ="series")
    config = dict(run_summary["experiment_config"])
    manifest_path = _choose_manifest(run_root, "sample_manifest_with_split.csv")
    X_pool, y_pool, curve_pool, ctx_pool, mask_pool, meta_df = build_sample_bundle(manifest_path)
    meta_df = _annotate_meta(meta_df, y_pool, mask_pool)

    split_mask = meta_df["split"].astype(str).eq(split).to_numpy()
    train_idx = meta_df.index[meta_df["split"].astype(str) == "train"].tolist()
    X_norm, norm_stats = normalize_inputs(X_pool, y_pool, curve_pool, ctx_pool, train_idx)
    split_ds = ControlDataset(
        X_norm[split_mask],
        y_pool[split_mask],
        curve_pool[split_mask],
        ctx_pool[split_mask],
        mask_pool[split_mask],
        norm_stats=norm_stats,
        meta_df=meta_df.loc[split_mask].reset_index(drop=True),
        mechanism_ids=None,
    )
    loader = DataLoader(split_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(
        backbone=str(config.get("backbone", "transformer")),
        input_dim=int(split_ds.src.shape[-1]),
        context_dim=int(split_ds.ctx.shape[-1]),
        config=config,
    ).to(device)
    ckpt = torch.load(run_root / f"seed_{seed}" / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_mean_t = torch.tensor(norm_stats["y_mean"], dtype=torch.float32, device=device)
    y_std_t = torch.tensor(norm_stats["y_std"], dtype=torch.float32, device=device)
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ctxs_raw: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            src = batch["src"].to(device=device, dtype=torch.float32)
            y_true = batch["y_true"].to(device=device, dtype=torch.float32)
            curve_norm = batch["curve_norm"].to(device=device, dtype=torch.float32)
            ctx = batch["ctx"].to(device=device, dtype=torch.float32)
            event_mask = batch["event_mask"].to(device=device, dtype=torch.float32)
            y_hat, _ = model(src, ctx, curve_norm, mechanism_id=None)
            preds.append((y_hat * y_std_t.view(1, 1, -1) + y_mean_t.view(1, 1, -1)).cpu().numpy())
            trues.append((y_true * y_std_t.view(1, 1, -1) + y_mean_t.view(1, 1, -1)).cpu().numpy())
            masks.append(event_mask.cpu().numpy())
            ctxs_raw.append(batch["ctx_raw"].cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    mask = np.concatenate(masks, axis=0)
    ctx_raw = np.concatenate(ctxs_raw, axis=0)
    return {
        "model_name": "unconditional_baseline",
        "seed": seed,
        "pred": pred,
        "true": true,
        "mask": mask,
        "ctx_raw": ctx_raw,
        "meta_df": split_ds.meta_df.reset_index(drop=True),
    }


def _conditioned_arrays(
    run_root: Path,
    split: str,
    batch_size: int,
    device: str,
    model_name: str,
) -> dict[str, Any]:
    run_summary = pd.read_json(run_root / "run_summary.json", typ="series")
    config_section = run_summary.get("config")
    if not isinstance(config_section, dict):
        config_section = run_summary.get("experiment_config", {})
    config = dict(config_section or {})
    manifest_path = _choose_manifest(run_root, "sample_manifest_used.csv")
    X_pool, y_pool, curve_pool, ctx_pool, mask_pool, meta_df, _ = build_sample_bundle_from_manifest(
        manifest_path=manifest_path,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        seed=int(config.get("seed", 2026)),
    )
    meta_df = _annotate_meta(meta_df, y_pool, mask_pool)
    split_mask = meta_df["split"].astype(str).eq(split).to_numpy()

    ckpt = torch.load(run_root / "best_model.pt", map_location=device, weights_only=False)
    norm_stats = ckpt["norm_stats"]
    event_targets = build_event_schema_targets(
        y_pool=y_pool,
        mask_pool=mask_pool,
        future_len=int(config.get("event_bin_size", 20)) * ((y_pool.shape[1] + int(config.get("event_bin_size", 20)) - 1) // int(config.get("event_bin_size", 20))),
        event_bin_size=int(config.get("event_bin_size", 20)),
    )
    split_ds = EventConditionedDataset(
        X_norm=((X_pool - norm_stats["feat_mean"].reshape(1, 1, -1)) / norm_stats["feat_std"].reshape(1, 1, -1)).astype(np.float32)[split_mask],
        y_pool=y_pool[split_mask],
        curve_pool=curve_pool[split_mask],
        ctx_pool=ctx_pool[split_mask],
        mask_pool=mask_pool[split_mask],
        norm_stats=norm_stats,
        event_targets={key: value[split_mask] for key, value in event_targets.items()},
        meta_df=meta_df.loc[split_mask].reset_index(drop=True),
    )
    loader = DataLoader(split_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = EventConditionedTrajectoryModel(
        input_dim=int(split_ds.src.shape[-1]),
        context_dim=int(split_ds.ctx.shape[-1]),
        future_len=int(y_pool.shape[1]),
        event_bin_size=int(config.get("event_bin_size", 20)),
        d_model=int(config.get("d_model", 128)),
        nhead=int(config.get("nhead", 2)),
        enc_layers=int(config.get("enc_layers", 2)),
        dec_layers=int(config.get("dec_layers", 2)),
        ffn_dim=int(config.get("ffn_dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
        event_embed_dim=int(config.get("event_embed_dim", 96)),
        out_dim=2,
        conditioning_mode=str(config.get("conditioning_mode", "baseline")),
        structure_width=float(config.get("structure_width", 0.065)),
        gate_temperature=float(config.get("gate_temperature", 0.040)),
        event_residual_scale=float(config.get("event_residual_scale", 1.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_mean_t = torch.tensor(norm_stats["y_mean"], dtype=torch.float32, device=device)
    y_std_t = torch.tensor(norm_stats["y_std"], dtype=torch.float32, device=device)
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ctxs_raw: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            src = batch["src"].to(device=device, dtype=torch.float32)
            ctx = batch["ctx"].to(device=device, dtype=torch.float32)
            curve_norm = batch["curve_norm"].to(device=device, dtype=torch.float32)
            y_true = batch["y_true"].to(device=device, dtype=torch.float32)
            event_mask = batch["event_mask"].to(device=device, dtype=torch.float32)
            y_hat, _ = model(src=src, ctx=ctx, curve_norm=curve_norm, event_teacher=None, privileged_event_teacher=None)
            preds.append((y_hat * y_std_t.view(1, 1, -1) + y_mean_t.view(1, 1, -1)).cpu().numpy())
            trues.append((y_true * y_std_t.view(1, 1, -1) + y_mean_t.view(1, 1, -1)).cpu().numpy())
            masks.append(event_mask.cpu().numpy())
            ctxs_raw.append(batch["ctx_raw"].cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    mask = np.concatenate(masks, axis=0)
    ctx_raw = np.concatenate(ctxs_raw, axis=0)
    return {
        "model_name": model_name,
        "seed": int(config.get("seed", 2026)),
        "pred": pred,
        "true": true,
        "mask": mask,
        "ctx_raw": ctx_raw,
        "meta_df": split_ds.meta_df.reset_index(drop=True),
        "run_root": run_root,
        "split": split,
    }


def _build_eval_pack(arrays: dict[str, Any], split: str) -> dict[str, Any]:
    pred = arrays["pred"]
    true = arrays["true"]
    mask = arrays["mask"]
    ctx_raw = arrays["ctx_raw"]
    meta_df = arrays["meta_df"].reset_index(drop=True).copy()

    weighted = compute_weighted_metrics(pred, true, mask)
    trajectory_sample_df = compute_trajectory_sample_metrics(
        meta_df=meta_df,
        pred=pred,
        true=true,
        mask=mask,
        ctx_raw=ctx_raw,
        split_name=split,
        seed=int(arrays["seed"]),
    )
    trajectory_sample_df = trajectory_sample_df.merge(
        meta_df[
            [
                "sample_key",
                "interaction_slice",
                "reversal_slice",
                "effective_mechanism_tag",
            ]
        ].drop_duplicates("sample_key"),
        on="sample_key",
        how="left",
    )
    trajectory_subset_rows = build_trajectory_subset_rows(trajectory_sample_df, split, int(arrays["seed"]))
    trajectory_subset_rows.extend(_subset_rows_for_family(trajectory_sample_df, split, int(arrays["seed"]), "interaction_slice"))
    trajectory_subset_rows.extend(_subset_rows_for_family(trajectory_sample_df, split, int(arrays["seed"]), "reversal_slice"))
    selection_summary = build_trajectory_selection_summary(weighted, trajectory_sample_df, subset_name=split)

    event_cfg = EventTargetConfig(future_len=int(pred.shape[1]))
    true_events = _event_rows_from_sequences(meta_df, true, mask, event_cfg, prefix="true")
    pred_events = _event_rows_from_sequences(meta_df, pred, mask, event_cfg, prefix="pred")
    event_sample_df, event_summary_df = _event_comparison_summary(true_events, pred_events, arrays["model_name"])

    return {
        "weighted": weighted,
        "weighted_long": pd.DataFrame(flatten_weighted_metrics(weighted, split_name=split, seed=int(arrays["seed"]))),
        "trajectory_sample_df": trajectory_sample_df,
        "trajectory_subset_df": _trajectory_subset_summary(trajectory_subset_rows, arrays["model_name"]),
        "selection_summary": pd.DataFrame([{"model_name": arrays["model_name"], **selection_summary}]),
        "event_true_df": true_events,
        "event_pred_df": pred_events,
        "event_sample_df": event_sample_df,
        "event_summary_df": event_summary_df,
        "mechanism_group_df": pd.DataFrame(
            compute_group_metric_rows(pred, true, mask, meta_df, "interaction_slice", split, int(arrays["seed"]))
        ),
        "reversal_group_df": pd.DataFrame(
            compute_group_metric_rows(pred, true, mask, meta_df, "reversal_slice", split, int(arrays["seed"]))
        ),
    }


def _save_eval_pack(output_dir: Path, prefix: str, pack: dict[str, Any]) -> None:
    for name, df in (
        ("weighted_metrics_long.csv", pack["weighted_long"]),
        ("trajectory_sample_metrics.csv", pack["trajectory_sample_df"]),
        ("trajectory_subset_metrics.csv", pack["trajectory_subset_df"]),
        ("selection_summary.csv", pack["selection_summary"]),
        ("event_sample_metrics.csv", pack["event_sample_df"]),
        ("event_summary.csv", pack["event_summary_df"]),
        ("interaction_group_metrics.csv", pack["mechanism_group_df"]),
        ("reversal_group_metrics.csv", pack["reversal_group_df"]),
    ):
        df.to_csv(output_dir / f"{prefix}_{name}", index=False)


def _save_prediction_sequences(arrays: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_keys = arrays["meta_df"]["sample_key"].astype(str).to_numpy(dtype=str, copy=True)
    np.savez_compressed(
        output_path,
        pred=arrays["pred"].astype(np.float32, copy=False),
        true=arrays["true"].astype(np.float32, copy=False),
        sample_keys=sample_keys,
        mask=arrays["mask"].astype(np.float32, copy=False),
        channel_names=SEQUENCE_CHANNEL_NAMES,
        channel_note=SEQUENCE_CHANNEL_NOTE,
        run_root=np.asarray(str(arrays["run_root"])),
        split=np.asarray(str(arrays["split"])),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditioned-run-root", required=True)
    parser.add_argument("--baseline-run-root", default=str(DEFAULT_BASELINE_RUN))
    parser.add_argument("--baseline-seed", type=int, default=2026)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default=str(REPORT_ROOT))
    parser.add_argument("--baseline-sequence-output", default=str(BASELINE_SEQUENCE_OUTPUT))
    parser.add_argument("--conditioned-sequence-output", default=str(CONDITIONED_SEQUENCE_OUTPUT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    baseline_arrays = _conditioned_arrays(
        Path(args.baseline_run_root),
        args.split,
        args.batch_size,
        device,
        model_name="baseline_conditioned",
    )
    conditioned_arrays = _conditioned_arrays(
        Path(args.conditioned_run_root),
        args.split,
        args.batch_size,
        device,
        model_name="conditioned_v2",
    )
    _save_prediction_sequences(baseline_arrays, Path(args.baseline_sequence_output).resolve())
    _save_prediction_sequences(conditioned_arrays, Path(args.conditioned_sequence_output).resolve())

    baseline_pack = _build_eval_pack(baseline_arrays, args.split)
    conditioned_pack = _build_eval_pack(conditioned_arrays, args.split)
    _save_eval_pack(output_dir, "baseline", baseline_pack)
    _save_eval_pack(output_dir, "conditioned", conditioned_pack)

    subset_compare = _pairwise_summary(
        baseline_pack["trajectory_subset_df"],
        conditioned_pack["trajectory_subset_df"],
        id_cols=["split", "subset_family", "subset_name"],
        value_cols=TRAJ_COMPARE_METRICS,
    )
    subset_compare.to_csv(output_dir / "trajectory_subset_comparison.csv", index=False)

    event_compare = _pairwise_summary(
        baseline_pack["event_summary_df"],
        conditioned_pack["event_summary_df"],
        id_cols=["event_name"],
        value_cols=["presence_acc", "time_abs_err_s", "direction_acc", "support_true", "support_matched"],
    )
    event_compare.to_csv(output_dir / "event_summary_comparison.csv", index=False)

    sample_compare = _pairwise_summary(
        baseline_pack["trajectory_sample_df"],
        conditioned_pack["trajectory_sample_df"],
        id_cols=[
            "sample_key",
            "split",
            "subj",
            "phase_type",
            "road_type_anchor",
            "mechanism_tag",
            "is_curve",
            "structure_slice",
            "structure_heavy",
            "valid_future_len",
            "eval_morphology_label",
        ],
        value_cols=TRAJ_COMPARE_METRICS,
    )
    sample_compare = sample_compare.merge(
        conditioned_arrays["meta_df"][["sample_key", "interaction_slice", "reversal_slice"]].drop_duplicates("sample_key"),
        on="sample_key",
        how="left",
    )
    sample_compare.to_csv(output_dir / "sample_level_comparison.csv", index=False)

    true_event_map = baseline_pack["event_true_df"].set_index("sample_key")
    baseline_lookup = {key: idx for idx, key in enumerate(baseline_arrays["meta_df"]["sample_key"].astype(str))}
    conditioned_lookup = {key: idx for idx, key in enumerate(conditioned_arrays["meta_df"]["sample_key"].astype(str))}

    selected_df = _select_cases(sample_compare)
    selected_df.to_csv(output_dir / "representative_samples_index.csv", index=False)
    overview_items: list[dict[str, Any]] = []
    for row_idx, row in selected_df.iterrows():
        sample_key = str(row["sample_key"])
        base_idx = baseline_lookup[sample_key]
        cond_idx = conditioned_lookup[sample_key]
        anchor = float(conditioned_arrays["ctx_raw"][cond_idx, 0])
        valid_len = int(conditioned_arrays["mask"][cond_idx].sum())
        times = np.arange(valid_len, dtype=np.float32) / FS
        truth_abs = conditioned_arrays["true"][cond_idx, :valid_len, 0] + anchor
        baseline_abs = baseline_arrays["pred"][base_idx, :valid_len, 0] + anchor
        conditioned_abs = conditioned_arrays["pred"][cond_idx, :valid_len, 0] + anchor
        event_map = _event_time_map(true_event_map.loc[sample_key])
        title = (
            f"{row['selection_tag']} | {sample_key}\n"
            f"{row['road_type_anchor']} | {row['eval_morphology_label']} | {row['interaction_slice']}"
        )
        out_path = figures_dir / f"{row_idx+1:02d}_{row['selection_tag']}_{sample_key.replace(':', '_')}.png"
        _plot_compare_case(out_path, title, times, truth_abs, baseline_abs, conditioned_abs, event_map)
        overview_items.append(
            {
                "title": title,
                "times": times,
                "truth_abs": truth_abs,
                "baseline_abs": baseline_abs,
                "conditioned_abs": conditioned_abs,
                "event_time_map": event_map,
            }
        )

    if overview_items:
        _plot_overview_mosaic(figures_dir / "representative_samples_overview.png", overview_items)

    save_json(
        output_dir / "eval_summary.json",
        {
            "device": device,
            "split": args.split,
            "baseline_run_root": str(Path(args.baseline_run_root).resolve()),
            "conditioned_run_root": str(Path(args.conditioned_run_root).resolve()),
            "baseline_seed": int(args.baseline_seed),
            "selected_case_count": int(len(selected_df)),
            "baseline_sequence_output": str(Path(args.baseline_sequence_output).resolve()),
            "conditioned_sequence_output": str(Path(args.conditioned_sequence_output).resolve()),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
