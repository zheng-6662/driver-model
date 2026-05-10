# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from baseline_eval_primary_aux import compute_trajectory_sample_metrics  # noqa: E402
from event_conditioned_baseline_model import (  # noqa: E402
    EventConditionedDataset,
    EventConditionedTrajectoryModel,
    build_event_schema_targets,
    build_response_type_targets,
)
from event_targets import EventTargetConfig, sequence_to_event_targets  # noqa: E402
from event_conditioned_eval_support import annotate_event_meta  # noqa: E402
from future_steer_speed_subjectsplit_masked import DEFAULT_BATCH_SIZE, FS, FUTURE_LEN  # noqa: E402
from run_event_conditioned_trajectory_baseline import (  # noqa: E402
    RUN_ROOT,
    apply_optional_context_augmentation,
    build_sample_bundle_from_manifest,
)

DEFAULT_SHARED_CASE_FILE = THIS_DIR / "shared_prediction_cases_test.csv"
PLOT_CHANNEL_NAMES = np.asarray(["steer_rel", "speed_delta"], dtype="<U32")
EVENT_NAMES = ("first_major_turn_onset", "first_reversal", "main_peak")
EVENT_LABELS = {
    "first_major_turn_onset": "major turn",
    "first_reversal": "reversal",
    "main_peak": "main peak",
}
EVENT_COLORS = {
    "first_major_turn_onset": "#ff7f0e",
    "first_reversal": "#2ca02c",
    "main_peak": "#d62728",
}
EVENT_STYLES = {
    "first_major_turn_onset": "--",
    "first_reversal": "-.",
    "main_peak": ":",
}
CASE_COLUMNS = [
    "selection_tag",
    "sample_key",
    "split",
    "subj",
    "phase_type",
    "road_type_anchor",
    "eval_morphology_label",
    "interaction_slice",
    "reversal_slice",
    "valid_future_len",
    "true_first_major_turn_onset_has",
    "true_first_major_turn_onset_idx",
    "true_first_reversal_has",
    "true_first_reversal_idx",
    "true_main_peak_idx",
    "true_peak_abs_steer_rel",
]


def _fresh_annotate_event_meta(meta_df: pd.DataFrame, y_pool: np.ndarray, mask_pool: np.ndarray) -> pd.DataFrame:
    stale_cols = [
        "eval_morphology_label",
        "structure_heavy",
        "structure_slice",
        "reversal_slice",
        "d3_mechanism_tag_anchor",
        "d3_mechanism_tag_episode",
        "effective_mechanism_tag",
        "interaction_slice",
    ]
    clean = meta_df.drop(columns=[col for col in stale_cols if col in meta_df.columns], errors="ignore")
    return annotate_event_meta(clean, y_pool, mask_pool)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _choose_manifest(run_root: Path, config: dict[str, Any]) -> Path:
    used_manifest = run_root / "sample_manifest_used.csv"
    if used_manifest.exists():
        return used_manifest
    configured = config.get("manifest")
    if configured:
        return Path(str(configured))
    raise FileNotFoundError(f"cannot find sample_manifest_used.csv or config.manifest under {run_root}")


def _safe_filename(value: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._-")
    return cleaned[:max_len] or "case"


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:10]


def _model_label(run_root: Path, config: dict[str, Any]) -> str:
    mode = str(config.get("conditioning_mode") or "unknown_mode")
    match = re.match(r"(SMOKE_[A-Za-z0-9]+|FAIR\d+)", run_root.name)
    run_code = match.group(1) if match else f"run_{_short_hash(run_root.name)}"
    return f"{run_code} | {mode}"


def _event_rows_from_sequences(
    meta_df: pd.DataFrame,
    y_pool: np.ndarray,
    mask_pool: np.ndarray,
    event_bin_size: int,
    prefix: str = "true",
) -> pd.DataFrame:
    cfg = EventTargetConfig(future_len=int(y_pool.shape[1]), bin_size=int(event_bin_size))
    rows: list[dict[str, Any]] = []
    for idx, meta_row in meta_df.reset_index(drop=True).iterrows():
        valid_len = int(np.sum(mask_pool[idx] > 0))
        targets = sequence_to_event_targets(y_pool[idx, :, 0], valid_len, config=cfg)
        row: dict[str, Any] = {
            "sample_key": str(meta_row.get("sample_key", idx)),
            "split": str(meta_row.get("split", "unknown")),
            "subj": str(meta_row.get("subj", "unknown")),
            "phase_type": str(meta_row.get("phase_type", "unknown")),
            "road_type_anchor": str(meta_row.get("road_type_anchor", "unknown")),
            "eval_morphology_label": str(meta_row.get("eval_morphology_label", "unknown")),
            "interaction_slice": str(meta_row.get("interaction_slice", "unknown")),
            "reversal_slice": str(meta_row.get("reversal_slice", "unknown")),
            "valid_future_len": valid_len,
            "true_peak_abs_steer_rel": float(np.max(np.abs(y_pool[idx, :valid_len, 0]))) if valid_len > 0 else 0.0,
        }
        for key, value in targets.items():
            row[f"{prefix}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _event_time_map(event_row: pd.Series) -> dict[str, float]:
    field_names = {
        "first_major_turn_onset": "true_first_major_turn_onset_idx",
        "first_reversal": "true_first_reversal_idx",
        "main_peak": "true_main_peak_idx",
    }
    out: dict[str, float] = {}
    for event_name, col_name in field_names.items():
        if col_name not in event_row or pd.isna(event_row[col_name]) or int(event_row[col_name]) < 0:
            out[event_name] = float("nan")
        else:
            out[event_name] = float(event_row[col_name]) / float(FS)
    return out


def _append_unique_cases(frames: list[pd.DataFrame], candidates: pd.DataFrame, tag: str, n_keep: int) -> None:
    if candidates.empty or n_keep <= 0:
        return
    sort_cols = ["true_peak_abs_steer_rel", "valid_future_len", "sample_key"]
    ascending = [False, False, True]
    existing_cols = [col for col in sort_cols if col in candidates.columns]
    existing_ascending = [ascending[sort_cols.index(col)] for col in existing_cols]
    picked = candidates.sort_values(existing_cols, ascending=existing_ascending).head(n_keep).copy()
    picked["selection_tag"] = tag
    frames.append(picked)


def _select_shared_cases(case_source_df: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    source = case_source_df.copy()
    source["sample_key"] = source["sample_key"].astype(str)
    source["road_type_anchor"] = source.get("road_type_anchor", "unknown").astype(str)
    source["eval_morphology_label"] = source.get("eval_morphology_label", "unknown").astype(str)
    source["interaction_slice"] = source.get("interaction_slice", "unknown").astype(str)
    source["reversal_slice"] = source.get("reversal_slice", "unknown").astype(str)

    frames: list[pd.DataFrame] = []
    _append_unique_cases(frames, source[source["interaction_slice"].eq("interaction")], "interaction_case", 2)
    _append_unique_cases(
        frames,
        source[source["eval_morphology_label"].isin(["reverse_correction", "multi_correction"])],
        "reversal_case",
        2,
    )
    _append_unique_cases(frames, source[source["road_type_anchor"].eq("curve")], "curve_case", 2)
    _append_unique_cases(frames, source, "large_peak_case", 2)
    _append_unique_cases(
        frames,
        source[(~source["road_type_anchor"].eq("curve")) & (~source["interaction_slice"].eq("interaction"))],
        "non_interaction_case",
        2,
    )
    if not frames:
        return source.head(0).copy()

    selected = pd.concat(frames, ignore_index=True)
    selected = selected.drop_duplicates(subset=["sample_key"], keep="first")
    if len(selected) < max_cases:
        remaining = source[~source["sample_key"].isin(set(selected["sample_key"]))].copy()
        if not remaining.empty:
            remaining = remaining.sort_values(
                ["true_peak_abs_steer_rel", "valid_future_len", "sample_key"],
                ascending=[False, False, True],
            ).head(max_cases - len(selected))
            remaining["selection_tag"] = "fill_case"
            selected = pd.concat([selected, remaining], ignore_index=True)
    selected = selected.head(max_cases).copy()
    for col in CASE_COLUMNS:
        if col not in selected.columns:
            selected[col] = np.nan
    return selected[CASE_COLUMNS]


def _load_or_build_cases(
    arrays: dict[str, Any],
    split: str,
    case_file: Path,
    max_cases: int,
    force_rebuild_cases: bool,
) -> pd.DataFrame:
    meta_df = arrays["meta_df"].reset_index(drop=True)
    true = arrays["true"]
    mask = arrays["mask"]
    event_df = _event_rows_from_sequences(
        meta_df=meta_df,
        y_pool=true,
        mask_pool=mask,
        event_bin_size=int(arrays["event_bin_size"]),
        prefix="true",
    )
    available_keys = set(meta_df["sample_key"].astype(str))

    if case_file.exists() and not force_rebuild_cases:
        cases = pd.read_csv(case_file)
        cases["sample_key"] = cases["sample_key"].astype(str)
        cases = cases[cases["sample_key"].isin(available_keys)].copy()
        if not cases.empty:
            merged = cases.drop(columns=[col for col in event_df.columns if col in cases.columns and col != "sample_key"], errors="ignore")
            merged = merged.merge(event_df, on="sample_key", how="left")
            for col in CASE_COLUMNS:
                if col not in merged.columns:
                    merged[col] = np.nan
            return merged[CASE_COLUMNS].head(max_cases).reset_index(drop=True)

    cases = _select_shared_cases(event_df[event_df["split"].astype(str).eq(split)].copy(), max_cases=max_cases)
    case_file.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(case_file, index=False, encoding="utf-8-sig")
    return cases.reset_index(drop=True)


def build_prediction_arrays(
    run_root: str | Path,
    split: str = "test",
    batch_size: int | None = None,
    device: str = "auto",
    checkpoint_name: str = "best_model.pt",
) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    summary_path = run_root / "run_summary.json"
    summary = _load_json(summary_path) if summary_path.exists() else {}
    config = dict(summary.get("config") or {})

    checkpoint_path = run_root / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    resolved_device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    config.update(checkpoint.get("args") or {})
    norm_stats = checkpoint["norm_stats"]

    manifest_path = _choose_manifest(run_root, config)
    max_train_samples = config.get("max_train_samples")
    max_val_samples = config.get("max_val_samples")
    max_test_samples = config.get("max_test_samples")
    if bool(config.get("smoke_test", False)):
        max_train_samples = config.get("smoke_train_samples")
        max_val_samples = config.get("smoke_val_samples")
        max_test_samples = config.get("smoke_test_samples")
    X_pool, y_pool, curve_pool, ctx_pool, mask_pool, meta_df, _ = build_sample_bundle_from_manifest(
        manifest_path=manifest_path,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        seed=int(config.get("seed", 2026)),
    )
    meta_df = _fresh_annotate_event_meta(meta_df, y_pool, mask_pool)
    train_idx = meta_df.index[meta_df["split"].astype(str).eq("train")].tolist()
    ctx_pool, _ = apply_optional_context_augmentation(
        ctx_pool=ctx_pool,
        meta_df=meta_df,
        train_idx=train_idx,
        args=argparse.Namespace(**config),
        run_root=run_root,
    )
    split_mask = meta_df["split"].astype(str).eq(split).to_numpy()
    if not bool(split_mask.any()):
        raise RuntimeError(f"no samples found for split={split!r} under {run_root}")

    event_bin_size = int(config.get("event_bin_size", 20))
    event_targets = build_event_schema_targets(
        y_pool=y_pool,
        mask_pool=mask_pool,
        future_len=FUTURE_LEN,
        event_bin_size=event_bin_size,
    )
    response_targets = None
    if bool(config.get("enable_response_type_head", False)) or bool(config.get("enable_response_type_condition", False)):
        response_targets = build_response_type_targets(
            y_pool=y_pool,
            mask_pool=mask_pool,
            amp_threshold=float(config.get("response_type_amp_threshold", 0.30)),
            late_peak_threshold_s=float(config.get("response_type_late_peak_threshold_s", 1.20)),
        )
    X_norm = ((X_pool - norm_stats["feat_mean"].reshape(1, 1, -1)) / norm_stats["feat_std"].reshape(1, 1, -1)).astype(np.float32)
    split_ds = EventConditionedDataset(
        X_norm=X_norm[split_mask],
        y_pool=y_pool[split_mask],
        curve_pool=curve_pool[split_mask],
        ctx_pool=ctx_pool[split_mask],
        mask_pool=mask_pool[split_mask],
        norm_stats=norm_stats,
        event_targets={key: value[split_mask] for key, value in event_targets.items()},
        meta_df=meta_df.loc[split_mask].reset_index(drop=True),
        response_targets=None if response_targets is None else {key: value[split_mask] for key, value in response_targets.items()},
    )
    loader = DataLoader(
        split_ds,
        batch_size=int(batch_size or config.get("batch_size") or DEFAULT_BATCH_SIZE),
        shuffle=False,
        num_workers=0,
    )

    candidate_prototypes = None
    if str(config.get("candidate_base_mode", "learned_delta")) == "response_prototype":
        proto_path = run_root / "candidate_prototypes_norm.npy"
        if not proto_path.exists() and str(config.get("candidate_prototype_path", "")).strip():
            candidate = Path(str(config.get("candidate_prototype_path")))
            if candidate.exists():
                proto_path = candidate
        if proto_path.exists():
            candidate_prototypes = np.load(proto_path).astype(np.float32)

    model = EventConditionedTrajectoryModel(
        input_dim=int(split_ds.src.shape[-1]),
        context_dim=int(split_ds.ctx.shape[-1]),
        future_len=FUTURE_LEN,
        event_bin_size=event_bin_size,
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
        enable_response_type_head=bool(config.get("enable_response_type_head", False)),
        enable_response_type_condition=bool(config.get("enable_response_type_condition", False)),
        response_type_hidden_dim=int(config.get("response_type_hidden_dim", 96)),
        num_trajectory_candidates=int(config.get("num_trajectory_candidates", 1)),
        candidate_delta_scale=float(config.get("candidate_delta_scale", 1.0)),
        candidate_base_mode=str(config.get("candidate_base_mode", "learned_delta")),
        candidate_prototypes=candidate_prototypes,
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    y_mean_t = torch.tensor(norm_stats["y_mean"], dtype=torch.float32, device=resolved_device)
    y_std_t = torch.tensor(norm_stats["y_std"], dtype=torch.float32, device=resolved_device)
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ctxs_raw: list[np.ndarray] = []
    with torch.no_grad():
        for raw_batch in loader:
            src = raw_batch["src"].to(device=resolved_device, dtype=torch.float32)
            ctx = raw_batch["ctx"].to(device=resolved_device, dtype=torch.float32)
            curve_norm = raw_batch["curve_norm"].to(device=resolved_device, dtype=torch.float32)
            y_true = raw_batch["y_true"].to(device=resolved_device, dtype=torch.float32)
            event_mask = raw_batch["event_mask"].to(device=resolved_device, dtype=torch.float32)
            y_hat, _ = model(src=src, ctx=ctx, curve_norm=curve_norm, event_teacher=None, privileged_event_teacher=None)
            preds.append((y_hat * y_std_t.view(1, 1, -1) + y_mean_t.view(1, 1, -1)).cpu().numpy())
            trues.append((y_true * y_std_t.view(1, 1, -1) + y_mean_t.view(1, 1, -1)).cpu().numpy())
            masks.append(event_mask.cpu().numpy())
            ctxs_raw.append(raw_batch["ctx_raw"].cpu().numpy())

    return {
        "run_root": run_root,
        "summary": summary,
        "config": config,
        "split": split,
        "event_bin_size": event_bin_size,
        "pred": np.concatenate(preds, axis=0),
        "true": np.concatenate(trues, axis=0),
        "mask": np.concatenate(masks, axis=0),
        "ctx_raw": np.concatenate(ctxs_raw, axis=0),
        "meta_df": split_ds.meta_df.reset_index(drop=True),
        "device": resolved_device,
        "checkpoint_name": checkpoint_name,
    }


def _plot_case(
    out_path: Path,
    times: np.ndarray,
    truth_abs: np.ndarray,
    pred_abs: np.ndarray,
    event_times: dict[str, float],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.plot(times, np.degrees(truth_abs), color="#1f77b4", linewidth=2.0, label="True")
    ax.plot(times, np.degrees(pred_abs), color="#ff7f0e", linewidth=1.45, label="Pred")
    for event_name, event_time in event_times.items():
        if not np.isfinite(event_time):
            continue
        ax.axvline(
            event_time,
            color=EVENT_COLORS.get(event_name, "#444444"),
            linestyle=EVENT_STYLES.get(event_name, ":"),
            linewidth=1.1,
            alpha=0.85,
        )
        ax.text(
            event_time,
            ax.get_ylim()[1] * 0.94,
            EVENT_LABELS.get(event_name, event_name),
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color=EVENT_COLORS.get(event_name, "#444444"),
            bbox={"boxstyle": "square,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.72},
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time after anchor (s)")
    ax.set_ylabel("Steering wheel angle (deg)")
    ax.set_xlim(float(times[0]), float(times[-1]))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_overview(out_path: Path, items: list[dict[str, Any]]) -> None:
    if not items:
        return
    cols = 2
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8.4, rows * 3.8), squeeze=False)
    axes_flat = axes.flatten()
    for ax, item in zip(axes_flat, items):
        ax.plot(item["times"], np.degrees(item["truth_abs"]), color="#1f77b4", linewidth=1.9, label="True")
        ax.plot(item["times"], np.degrees(item["pred_abs"]), color="#ff7f0e", linewidth=1.25, label="Pred")
        for event_name, event_time in item["event_times"].items():
            if not np.isfinite(event_time):
                continue
            ax.axvline(
                event_time,
                color=EVENT_COLORS.get(event_name, "#444444"),
                linestyle=EVENT_STYLES.get(event_name, ":"),
                linewidth=0.95,
                alpha=0.8,
            )
        ax.set_title(item["short_title"], fontsize=9)
        ax.set_xlabel("Time after anchor (s)")
        ax.set_ylabel("Steering wheel angle (deg)")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=7, loc="best")
    for ax in axes_flat[len(items) :]:
        fig.delaxes(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_prediction_plots_for_run(
    run_root: str | Path,
    split: str = "test",
    case_file: str | Path | None = None,
    max_cases: int = 8,
    batch_size: int | None = None,
    device: str = "auto",
    checkpoint_name: str = "best_model.pt",
    force_rebuild_cases: bool = False,
    save_sequences: bool = True,
) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    case_file_path = Path(case_file) if case_file is not None else THIS_DIR / f"shared_prediction_cases_{split}.csv"
    arrays = build_prediction_arrays(
        run_root=run_root,
        split=split,
        batch_size=batch_size,
        device=device,
        checkpoint_name=checkpoint_name,
    )
    figures_dir = run_root / "prediction_figures" / split
    figures_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_or_build_cases(
        arrays=arrays,
        split=split,
        case_file=case_file_path,
        max_cases=max_cases,
        force_rebuild_cases=force_rebuild_cases,
    )
    cases.to_csv(figures_dir / "selected_cases_used.csv", index=False, encoding="utf-8-sig")

    sample_metrics_df = compute_trajectory_sample_metrics(
        meta_df=arrays["meta_df"],
        pred=arrays["pred"],
        true=arrays["true"],
        mask=arrays["mask"],
        ctx_raw=arrays["ctx_raw"],
        split_name=split,
        seed=int(arrays["config"].get("seed", 2026)),
    )
    sample_metrics_df.to_csv(figures_dir / "prediction_sample_metrics.csv", index=False, encoding="utf-8-sig")

    if save_sequences:
        np.savez_compressed(
            figures_dir / "prediction_sequences.npz",
            pred=arrays["pred"].astype(np.float32, copy=False),
            true=arrays["true"].astype(np.float32, copy=False),
            mask=arrays["mask"].astype(np.float32, copy=False),
            ctx_raw=arrays["ctx_raw"].astype(np.float32, copy=False),
            sample_key=arrays["meta_df"]["sample_key"].astype(str).to_numpy(dtype="<U512"),
            channel_names=PLOT_CHANNEL_NAMES,
        )

    meta_lookup = {key: idx for idx, key in enumerate(arrays["meta_df"]["sample_key"].astype(str))}
    model_label = _model_label(run_root, arrays["config"])
    overview_items: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for plot_idx, case_row in cases.reset_index(drop=True).iterrows():
        sample_key = str(case_row["sample_key"])
        if sample_key not in meta_lookup:
            continue
        local_idx = meta_lookup[sample_key]
        valid_len = int(np.sum(arrays["mask"][local_idx] > 0))
        if valid_len <= 1:
            continue
        anchor = float(arrays["ctx_raw"][local_idx, 0])
        times = np.arange(valid_len, dtype=np.float32) / float(FS)
        truth_abs = arrays["true"][local_idx, :valid_len, 0] + anchor
        pred_abs = arrays["pred"][local_idx, :valid_len, 0] + anchor
        event_times = _event_time_map(case_row)
        rmse = float(np.sqrt(np.mean((pred_abs - truth_abs) ** 2)))
        short_key = textwrap.shorten(sample_key, width=92, placeholder="...")
        title = (
            f"{textwrap.shorten(model_label, width=96, placeholder='...')}\n"
            f"{case_row.get('selection_tag', 'case')} | RMSE={rmse:.4f} rad | {short_key}\n"
            f"road={case_row.get('road_type_anchor', 'unknown')} | morph={case_row.get('eval_morphology_label', 'unknown')} | interaction={case_row.get('interaction_slice', 'unknown')}"
        )
        filename = f"{plot_idx + 1:02d}_{_safe_filename(str(case_row.get('selection_tag', 'case')), 36)}_{_short_hash(sample_key)}.png"
        out_path = figures_dir / filename
        _plot_case(out_path, times, truth_abs, pred_abs, event_times, title)
        overview_items.append(
            {
                "times": times,
                "truth_abs": truth_abs,
                "pred_abs": pred_abs,
                "event_times": event_times,
                "short_title": f"{plot_idx + 1:02d} {case_row.get('selection_tag', 'case')} | RMSE={rmse:.4f}",
            }
        )
        case_rows.append(
            {
                "plot_index": int(plot_idx + 1),
                "plot_file": str(out_path),
                "sample_key": sample_key,
                "selection_tag": str(case_row.get("selection_tag", "case")),
                "rmse_abs_steer_rad": rmse,
            }
        )

    _plot_overview(figures_dir / "overview.png", overview_items)
    plot_index_df = pd.DataFrame(case_rows)
    plot_index_df.to_csv(figures_dir / "plot_index.csv", index=False, encoding="utf-8-sig")
    return {
        "run_root": str(run_root),
        "figures_dir": str(figures_dir),
        "overview_path": str(figures_dir / "overview.png"),
        "case_file": str(case_file_path),
        "plot_count": int(len(case_rows)),
        "plot_index_path": str(figures_dir / "plot_index.csv"),
        "sequence_path": str(figures_dir / "prediction_sequences.npz"),
    }


def latest_run_root(include_smoke: bool = False) -> Path:
    if not RUN_ROOT.exists():
        raise FileNotFoundError(f"run root does not exist: {RUN_ROOT}")
    candidates = [path for path in RUN_ROOT.iterdir() if path.is_dir() and (path / "best_model.pt").exists()]
    if not include_smoke:
        candidates = [path for path in candidates if not path.name.upper().startswith("SMOKE")]
    if not candidates:
        raise FileNotFoundError(f"no trained run with best_model.pt found under {RUN_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparable prediction plots for one trained comparison run.")
    parser.add_argument("run_root", nargs="?", default=None, help="Trained run directory. If omitted, use latest non-SMOKE run.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--case-file", default=None, help="Shared case CSV. Defaults to shared_prediction_cases_<split>.csv here.")
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument("--force-rebuild-cases", action="store_true")
    parser.add_argument("--include-smoke", action="store_true", help="Allow latest-run auto detection to choose SMOKE runs.")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve() if args.run_root else latest_run_root(include_smoke=bool(args.include_smoke))
    result = save_prediction_plots_for_run(
        run_root=run_root,
        split=args.split,
        case_file=args.case_file,
        max_cases=int(args.max_cases),
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_name=args.checkpoint_name,
        force_rebuild_cases=bool(args.force_rebuild_cases),
    )
    print(f"run_root: {result['run_root']}")
    print(f"figures_dir: {result['figures_dir']}")
    print(f"overview: {result['overview_path']}")
    print(f"shared_cases: {result['case_file']}")
    print(f"plot_count: {result['plot_count']}")


if __name__ == "__main__":
    main()
