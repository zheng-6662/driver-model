# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
PROJECT_ROOT = THIS_DIR.parents[4]
REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
DEFAULT_INDEX = REPORTS_DIR / "style_physio_eeg_e8_reliable_physical_summary_20260508" / "e8_gate_prediction_index.csv"
DEFAULT_OUT_DIR = REPORTS_DIR / "style_physio_eeg_g9_candidate_convergence_20260508"
DEFAULT_CASE_FILE = THIS_DIR / "shared_prediction_cases_test.csv"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from baseline_eval_primary_aux import compute_trajectory_sample_metrics  # noqa: E402
from event_conditioned_eval_support import build_primary_selection_bundle  # noqa: E402
from prediction_plotting import PLOT_CHANNEL_NAMES, build_prediction_arrays  # noqa: E402


FS = 200
TAIL_START = 300
MAJOR_AMP = 0.20
LARGE_AMP = 0.30
POINT_EPS = 0.03
AREA_MEAN_THRESHOLD = 0.05
UNDER_RATIO = 0.70
SEVERE_UNDER_RATIO = 0.45
OVER_RATIO = 1.50
TAIL_DRIFT_ABS_ERR = 0.20

LABELS = {
    "E2": "E2 基准",
    "E5A": "E5A 数值候选",
    "E6": "E6 物理平衡候选",
    "E8": "E8 no-go 对照",
    "E5A-Cal": "E5A-Cal validation 幅值校准",
    "E6-Cal": "E6-Cal validation 幅值校准",
}

PLOT_LABELS = {
    "E2": "E2 baseline",
    "E5A": "E5A numeric",
    "E6": "E6 physical-balance",
    "E8": "E8 no-go",
    "E5A-Cal": "E5A-Cal",
    "E6-Cal": "E6-Cal",
}

COLORS = {
    "true": "#111111",
    "E2": "#1f77b4",
    "E5A": "#2ca02c",
    "E6": "#d62728",
    "E8": "#e377c2",
    "E5A-Cal": "#ff7f0e",
    "E6-Cal": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G9 candidate convergence and validation-only amplitude calibration.")
    parser.add_argument("--prediction-index", default=str(DEFAULT_INDEX))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--case-file", default=str(DEFAULT_CASE_FILE))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2028])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-gain", type=float, default=1.18)
    parser.add_argument("--gain-step", type=float, default=0.01)
    parser.add_argument("--val-rmse-tolerance", type=float, default=0.006)
    parser.add_argument("--val-tail-tolerance", type=float, default=0.010)
    parser.add_argument("--val-selection-tolerance", type=float, default=0.015)
    return parser.parse_args()


def _safe_sign(value: float, eps: float = POINT_EPS) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _apply_gain(pred: np.ndarray, gain: float) -> np.ndarray:
    out = np.asarray(pred, dtype=np.float32).copy()
    out[:, :, 0] *= float(gain)
    return out


def _amp_bin(true_amp: float) -> str:
    if true_amp < 0.20:
        return "<0.20"
    if true_amp < 0.50:
        return "0.20-0.50"
    if true_amp < 1.00:
        return "0.50-1.00"
    if true_amp < 2.00:
        return "1.00-2.00"
    return ">=2.00"


def _load_index(path: Path, seeds: list[int]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"prediction index not found: {path}")
    idx = pd.read_csv(path)
    idx = idx[idx["seed"].astype(int).isin([int(s) for s in seeds])].copy()
    idx = idx[idx["experiment_id"].astype(str).isin(["E2", "E5A", "E6", "E8"])].copy()
    if idx.empty:
        raise RuntimeError(f"no E2/E5A/E6/E8 rows found in {path}")
    return idx.sort_values(["seed", "experiment_id"]).reset_index(drop=True)


def _array_key(exp_id: str, seed: int, split: str) -> tuple[str, int, str]:
    return str(exp_id), int(seed), str(split)


def _build_arrays(index_df: pd.DataFrame, args: argparse.Namespace) -> dict[tuple[str, int, str], dict[str, Any]]:
    arrays: dict[tuple[str, int, str], dict[str, Any]] = {}
    needed_splits: dict[str, list[str]] = {
        "E2": ["test"],
        "E5A": ["val", "test"],
        "E6": ["val", "test"],
        "E8": ["test"],
    }
    for _, row in index_df.iterrows():
        exp_id = str(row["experiment_id"])
        seed = int(row["seed"])
        for split in needed_splits.get(exp_id, ["test"]):
            key = _array_key(exp_id, seed, split)
            if key in arrays:
                continue
            print(f"Building arrays: {exp_id} seed={seed} split={split}", flush=True)
            arrays[key] = build_prediction_arrays(
                run_root=str(row["run_root"]),
                split=split,
                batch_size=int(args.batch_size),
                device=str(args.device),
            )
    return arrays


def _compute_metrics(
    exp_id: str,
    seed: int,
    split: str,
    arrays: dict[str, Any],
    pred_override: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pred = np.asarray(arrays["pred"] if pred_override is None else pred_override, dtype=np.float32)
    true = np.asarray(arrays["true"], dtype=np.float32)
    mask = np.asarray(arrays["mask"], dtype=np.float32)
    steer_mask = mask > 0.5
    steer_sq = ((pred[:, :, 0] - true[:, :, 0]) ** 2) * steer_mask
    rmse = float(np.sqrt(steer_sq.sum() / max(float(steer_mask.sum()), 1.0)))

    tail_mask = steer_mask[:, TAIL_START:]
    tail_sq = ((pred[:, TAIL_START:, 0] - true[:, TAIL_START:, 0]) ** 2) * tail_mask
    tail_point_rmse = float(np.sqrt(tail_sq.sum() / max(float(tail_mask.sum()), 1.0)))

    bundle = build_primary_selection_bundle(
        pred=pred,
        true=true,
        mask=mask,
        ctx_raw=np.asarray(arrays["ctx_raw"], dtype=np.float32),
        meta_df=arrays["meta_df"],
        split_name=split,
        seed=int(seed),
    )
    selection = bundle["selection_summary"]
    physical_df = _physical_detail_rows(
        exp_id=exp_id,
        seed=seed,
        split=split,
        arrays=arrays,
        pred_override=pred,
    )
    phys = _physical_summary_from_detail(physical_df)
    metrics = {
        "experiment_id": exp_id,
        "experiment_name": LABELS.get(exp_id, exp_id),
        "seed": int(seed),
        "split": split,
        "test_steer_rmse": rmse,
        "tail_point_rmse": tail_point_rmse,
        "tail_rmse": float(selection["rmse_tail_abs_steer"]),
        "selection": float(selection["selection_score"]),
        "primary_rmse": float(selection["overall_primary_steer_rmse"]),
        "under_amp_rate_major": phys["under_amp_rate_major"],
        "severe_under_amp_rate_large": phys["severe_under_amp_rate_large"],
        "peak_side_wrong_at_true_peak_rate_major": phys["peak_side_wrong_at_true_peak_rate_major"],
        "peak_side_wrong_at_pred_peak_rate_major": phys["peak_side_wrong_at_pred_peak_rate_major"],
        "opposite_side_heavy_rate_major": phys["opposite_side_heavy_rate_major"],
        "tail_drift_risk_rate_major": phys["tail_drift_risk_rate_major"],
        "median_amp_ratio_major": phys["median_amp_ratio_major"],
    }
    return metrics, physical_df


def _physical_detail_rows(
    exp_id: str,
    seed: int,
    split: str,
    arrays: dict[str, Any],
    pred_override: np.ndarray | None = None,
) -> pd.DataFrame:
    pred = np.asarray(arrays["pred"] if pred_override is None else pred_override, dtype=np.float32)
    true = np.asarray(arrays["true"], dtype=np.float32)
    mask = np.asarray(arrays["mask"], dtype=np.float32) > 0.5
    meta = arrays["meta_df"].reset_index(drop=True).copy()
    rows: list[dict[str, Any]] = []
    steps = np.arange(pred.shape[1])
    for idx, meta_row in meta.iterrows():
        valid = mask[idx]
        if not bool(valid.any()):
            continue
        p = pred[idx, valid, 0].astype(float)
        t = true[idx, valid, 0].astype(float)
        true_peak_idx = int(np.argmax(np.abs(t)))
        pred_peak_idx = int(np.argmax(np.abs(p)))
        true_peak_signed = float(t[true_peak_idx])
        pred_at_true_peak = float(p[true_peak_idx])
        pred_peak_signed = float(p[pred_peak_idx])
        true_amp = abs(true_peak_signed)
        pred_amp = abs(pred_peak_signed)
        amp_ratio = pred_amp / max(true_amp, 1e-6)
        true_peak_sign = _safe_sign(true_peak_signed)
        pred_at_true_peak_sign = _safe_sign(pred_at_true_peak)
        pred_peak_sign = _safe_sign(pred_peak_signed)
        significant = np.abs(t) >= MAJOR_AMP
        opposite_side = significant & (t * p < -(POINT_EPS**2))
        significant_n = int(significant.sum())
        opposite_side_rate = float(opposite_side.sum() / significant_n) if significant_n else np.nan

        valid_steps = steps[valid]
        tail_local = valid_steps >= TAIL_START
        if bool(tail_local.any()):
            true_tail_mean = float(np.mean(t[tail_local]))
            pred_tail_mean = float(np.mean(p[tail_local]))
            tail_mean_abs_err = abs(pred_tail_mean - true_tail_mean)
            true_tail_sign = _safe_sign(true_tail_mean, AREA_MEAN_THRESHOLD)
            pred_tail_sign = _safe_sign(pred_tail_mean, AREA_MEAN_THRESHOLD)
            tail_side_wrong = bool(true_tail_sign != 0 and pred_tail_sign == -true_tail_sign)
        else:
            true_tail_mean = np.nan
            pred_tail_mean = np.nan
            tail_mean_abs_err = np.nan
            tail_side_wrong = False
        true_area_sign = _safe_sign(float(np.mean(t)), AREA_MEAN_THRESHOLD)
        pred_area_sign = _safe_sign(float(np.mean(p)), AREA_MEAN_THRESHOLD)
        is_major = bool(true_amp >= MAJOR_AMP)
        is_large = bool(true_amp >= LARGE_AMP)
        sample_rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        rows.append(
            {
                "experiment_id": exp_id,
                "experiment_name": LABELS.get(exp_id, exp_id),
                "seed": int(seed),
                "split": split,
                "sample_key": str(meta_row.get("sample_key", idx)),
                "phase_type": str(meta_row.get("phase_type", "unknown")),
                "road_type_anchor": str(meta_row.get("road_type_anchor", "unknown")),
                "eval_morphology_label": str(meta_row.get("eval_morphology_label", "unknown")),
                "interaction_slice": str(meta_row.get("interaction_slice", "unknown")),
                "effective_mechanism_tag": str(meta_row.get("effective_mechanism_tag", "unknown")),
                "true_amp": true_amp,
                "pred_amp": pred_amp,
                "amp_ratio": amp_ratio,
                "amp_bin": _amp_bin(true_amp),
                "sample_rmse": sample_rmse,
                "true_peak_signed": true_peak_signed,
                "pred_at_true_peak": pred_at_true_peak,
                "pred_peak_signed": pred_peak_signed,
                "true_peak_idx": true_peak_idx,
                "pred_peak_idx": pred_peak_idx,
                "peak_time_abs_err_s": abs(pred_peak_idx - true_peak_idx) / float(FS),
                "true_peak_sign": true_peak_sign,
                "pred_at_true_peak_sign": pred_at_true_peak_sign,
                "pred_peak_sign": pred_peak_sign,
                "true_area_sign": true_area_sign,
                "pred_area_sign": pred_area_sign,
                "opposite_side_rate": opposite_side_rate,
                "true_tail_mean": true_tail_mean,
                "pred_tail_mean": pred_tail_mean,
                "tail_mean_abs_err": tail_mean_abs_err,
                "tail_side_wrong": tail_side_wrong,
                "is_major_response": is_major,
                "is_large_response": is_large,
                "under_amp": bool(is_major and amp_ratio < UNDER_RATIO),
                "severe_under_amp": bool(is_large and amp_ratio < SEVERE_UNDER_RATIO),
                "over_amp": bool(is_major and amp_ratio > OVER_RATIO),
                "peak_side_wrong_at_true_peak": bool(is_major and true_peak_sign != 0 and pred_at_true_peak_sign == -true_peak_sign),
                "peak_side_wrong_at_pred_peak": bool(is_major and true_peak_sign != 0 and pred_peak_sign == -true_peak_sign),
                "area_side_wrong": bool(true_area_sign != 0 and pred_area_sign == -true_area_sign),
                "opposite_side_heavy": bool(is_major and not np.isnan(opposite_side_rate) and opposite_side_rate >= 0.20),
                "tail_drift_risk": bool(is_major and ((not np.isnan(tail_mean_abs_err) and tail_mean_abs_err >= TAIL_DRIFT_ABS_ERR) or tail_side_wrong)),
            }
        )
    return pd.DataFrame(rows)


def _physical_summary_from_detail(df: pd.DataFrame) -> dict[str, float]:
    major = df[df["is_major_response"]]
    large = df[df["is_large_response"]]
    return {
        "median_amp_ratio_major": float(major["amp_ratio"].median()) if len(major) else np.nan,
        "under_amp_rate_major": float(major["under_amp"].mean()) if len(major) else np.nan,
        "severe_under_amp_rate_large": float(large["severe_under_amp"].mean()) if len(large) else np.nan,
        "peak_side_wrong_at_true_peak_rate_major": float(major["peak_side_wrong_at_true_peak"].mean()) if len(major) else np.nan,
        "peak_side_wrong_at_pred_peak_rate_major": float(major["peak_side_wrong_at_pred_peak"].mean()) if len(major) else np.nan,
        "opposite_side_heavy_rate_major": float(major["opposite_side_heavy"].mean()) if len(major) else np.nan,
        "tail_drift_risk_rate_major": float(major["tail_drift_risk"].mean()) if len(major) else np.nan,
    }


def _group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = {col: key for col, key in zip(group_cols, keys)}
        major = group[group["is_major_response"]]
        large = group[group["is_large_response"]]
        item.update(
            {
                "n": int(len(group)),
                "n_major": int(len(major)),
                "sample_rmse_mean": float(group["sample_rmse"].mean()) if len(group) else np.nan,
                "true_amp_mean": float(group["true_amp"].mean()) if len(group) else np.nan,
                "amp_ratio_median_major": float(major["amp_ratio"].median()) if len(major) else np.nan,
                "under_amp_rate_major": float(major["under_amp"].mean()) if len(major) else np.nan,
                "severe_under_amp_rate_large": float(large["severe_under_amp"].mean()) if len(large) else np.nan,
                "peak_side_wrong_at_true_peak_rate_major": float(major["peak_side_wrong_at_true_peak"].mean()) if len(major) else np.nan,
                "peak_side_wrong_at_pred_peak_rate_major": float(major["peak_side_wrong_at_pred_peak"].mean()) if len(major) else np.nan,
                "opposite_side_heavy_rate_major": float(major["opposite_side_heavy"].mean()) if len(major) else np.nan,
                "tail_drift_risk_rate_major": float(major["tail_drift_risk"].mean()) if len(major) else np.nan,
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def _fit_gain_for_model_seed(exp_id: str, seed: int, val_arrays: dict[str, Any], args: argparse.Namespace) -> tuple[float, pd.DataFrame]:
    raw_metrics, _ = _compute_metrics(exp_id=exp_id, seed=seed, split="val", arrays=val_arrays)
    gains = np.round(np.arange(1.0, float(args.max_gain) + 0.0001, float(args.gain_step)), 4)
    rows: list[dict[str, Any]] = []
    for gain in gains:
        pred_gain = _apply_gain(val_arrays["pred"], float(gain))
        metrics, _ = _compute_metrics(exp_id=f"{exp_id}-CalGrid", seed=seed, split="val", arrays=val_arrays, pred_override=pred_gain)
        item = {
            "source_experiment_id": exp_id,
            "seed": int(seed),
            "gain": float(gain),
            "val_rmse": float(metrics["test_steer_rmse"]),
            "val_tail": float(metrics["tail_rmse"]),
            "val_selection": float(metrics["selection"]),
            "val_under_amp_rate_major": float(metrics["under_amp_rate_major"]),
            "val_peak_wrong_true_rate_major": float(metrics["peak_side_wrong_at_true_peak_rate_major"]),
            "raw_val_rmse": float(raw_metrics["test_steer_rmse"]),
            "raw_val_tail": float(raw_metrics["tail_rmse"]),
            "raw_val_selection": float(raw_metrics["selection"]),
            "raw_val_under_amp_rate_major": float(raw_metrics["under_amp_rate_major"]),
        }
        item["within_tolerance"] = bool(
            item["val_rmse"] <= item["raw_val_rmse"] + float(args.val_rmse_tolerance)
            and item["val_tail"] <= item["raw_val_tail"] + float(args.val_tail_tolerance)
            and item["val_selection"] <= item["raw_val_selection"] + float(args.val_selection_tolerance)
            and item["val_peak_wrong_true_rate_major"] <= raw_metrics["peak_side_wrong_at_true_peak_rate_major"] + 0.005
        )
        rows.append(item)
    grid = pd.DataFrame(rows)
    feasible = grid[grid["within_tolerance"]].copy()
    if feasible.empty:
        chosen = grid.sort_values(["val_selection", "val_rmse", "gain"], ascending=[True, True, True]).iloc[0]
    else:
        feasible["gain_penalty"] = (feasible["gain"] - 1.0).abs()
        chosen = feasible.sort_values(
            ["val_under_amp_rate_major", "val_selection", "val_rmse", "gain_penalty"],
            ascending=[True, True, True, True],
        ).iloc[0]
    return float(chosen["gain"]), grid


def _save_calibrated_sequences(
    out_dir: Path,
    exp_id: str,
    seed: int,
    arrays: dict[str, Any],
    pred_cal: np.ndarray,
) -> dict[str, str]:
    cal_root = out_dir / "calibrated_predictions" / f"{exp_id}_seed{seed}"
    pred_dir = cal_root / "prediction_figures" / "test"
    pred_dir.mkdir(parents=True, exist_ok=True)
    sample_df = compute_trajectory_sample_metrics(
        meta_df=arrays["meta_df"],
        pred=pred_cal,
        true=arrays["true"],
        mask=arrays["mask"],
        ctx_raw=arrays["ctx_raw"],
        split_name="test",
        seed=int(seed),
    )
    sample_df.to_csv(pred_dir / "prediction_sample_metrics.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        pred_dir / "prediction_sequences.npz",
        pred=pred_cal.astype(np.float32, copy=False),
        true=np.asarray(arrays["true"], dtype=np.float32),
        mask=np.asarray(arrays["mask"], dtype=np.float32),
        ctx_raw=np.asarray(arrays["ctx_raw"], dtype=np.float32),
        sample_key=arrays["meta_df"]["sample_key"].astype(str).to_numpy(dtype="<U512"),
        channel_names=PLOT_CHANNEL_NAMES,
    )
    return {
        "virtual_run_root": str(cal_root),
        "prediction_sequences": str(pred_dir / "prediction_sequences.npz"),
        "sample_metrics_csv": str(pred_dir / "prediction_sample_metrics.csv"),
    }


def _plot_fixed_cases(
    out_dir: Path,
    seed: int,
    arrays_by_exp: dict[str, dict[str, Any]],
    case_file: Path,
) -> Path:
    cases = pd.read_csv(case_file).head(8)
    fig, axes = plt.subplots(4, 2, figsize=(16, 13.5), squeeze=False)
    axes_flat = axes.flatten()
    anchor_exp = "E2" if "E2" in arrays_by_exp else next(iter(arrays_by_exp))
    anchor_meta = arrays_by_exp[anchor_exp]["meta_df"].reset_index(drop=True)
    anchor_lookup = {str(key): idx for idx, key in enumerate(anchor_meta["sample_key"].astype(str))}
    for ax, (_, case) in zip(axes_flat, cases.iterrows()):
        sample_key = str(case["sample_key"])
        if sample_key not in anchor_lookup:
            ax.axis("off")
            continue
        idx = anchor_lookup[sample_key]
        valid_len = int((arrays_by_exp[anchor_exp]["mask"][idx] > 0.5).sum())
        times = np.arange(valid_len, dtype=float) / float(FS)
        truth = arrays_by_exp[anchor_exp]["true"][idx, :valid_len, 0]
        ax.plot(times, np.degrees(truth), color=COLORS["true"], linewidth=2.0, label="True")
        for exp_id, arr in arrays_by_exp.items():
            meta = arr["meta_df"].reset_index(drop=True)
            lookup = {str(key): j for j, key in enumerate(meta["sample_key"].astype(str))}
            if sample_key not in lookup:
                continue
            j = lookup[sample_key]
            pred = arr["pred"][j, :valid_len, 0]
            ax.plot(
                times,
                np.degrees(pred),
                color=COLORS.get(exp_id, None),
                linewidth=1.35,
                label=PLOT_LABELS.get(exp_id, exp_id),
            )
        ax.set_title(f"{int(case.get('plot_index', len(cases))):02d} {case.get('selection_tag', 'case')} | {sample_key[:36]}", fontsize=9)
        ax.set_xlabel("Time after anchor (s)")
        ax.set_ylabel("Steering wheel angle (deg)")
        ax.grid(alpha=0.24)
        ax.legend(fontsize=6.8, loc="best")
    for ax in axes_flat[len(cases) :]:
        ax.axis("off")
    fig.suptitle(f"G9 fixed-case comparison, seed {seed}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plot_dir = out_dir / "comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"g9_fixed_case_seed{seed}.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def _fmt(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def _write_reports(out_dir: Path, gate: dict[str, Any], visual_notes: dict[str, str]) -> None:
    metrics = pd.read_csv(out_dir / "g9_metric_summary_seed.csv")
    mean = pd.read_csv(out_dir / "g9_metric_summary_mean.csv")
    phys_mean = mean[mean["split"].eq("test")].copy()
    conv_lines = [
        "# G9 候选收敛报告",
        "",
        "## 目的",
        "",
        "本轮不新增 EEG/生理推理输入，不改变粗细双头结构，也不继续 E8 式强物理训练。G9 只基于已有 E2、E5A、E6、E8 的预测结果，做候选收敛、分箱诊断，并验证 validation-only 的轻量幅值校准是否值得进入最终候选。",
        "",
        "## seed 2026/2028 指标汇总",
        "",
        "| 版本 | test RMSE | tail | selection | 幅值不足率 | 严重幅值不足率 | 真实主峰错号率 | 预测主峰错号率 | 零线两侧明显相反率 | 固定 case 视觉结论 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    order = ["E2", "E5A", "E6", "E8", "E5A-Cal", "E6-Cal"]
    for exp_id in order:
        row = phys_mean[phys_mean["experiment_id"].eq(exp_id)]
        if row.empty:
            continue
        r = row.iloc[0]
        conv_lines.append(
            "| "
            + " | ".join(
                [
                    exp_id,
                    _fmt(r["test_steer_rmse_mean"]),
                    _fmt(r["tail_rmse_mean"]),
                    _fmt(r["selection_mean"]),
                    _pct(r["under_amp_rate_major_mean"]),
                    _pct(r["severe_under_amp_rate_large_mean"]),
                    _pct(r["peak_side_wrong_at_true_peak_rate_major_mean"]),
                    _pct(r["peak_side_wrong_at_pred_peak_rate_major_mean"]),
                    _pct(r["opposite_side_heavy_rate_major_mean"]),
                    visual_notes.get(exp_id, "-"),
                ]
            )
            + " |"
        )
    conv_lines += [
        "",
        "## 收敛判断",
        "",
        "- E5A 仍是数值准确性最强的候选，但幅值不足最明显。",
        "- E6 是当前更稳妥的物理平衡候选，牺牲少量 RMSE 换来更低的幅值不足。",
        "- E8 可以证明强主峰约束会降低幅值不足，但整体 RMSE/tail/selection 退化，且固定 case 有过强修正和后段漂移风险，因此继续作为 no-go 对照。",
        "- 校准版本是否进入最终候选见 `g9_calibration_report_cn.md`。",
    ]
    (out_dir / "g9_candidate_convergence_report_cn.md").write_text("\n".join(conv_lines) + "\n", encoding="utf-8")

    amp = pd.read_csv(out_dir / "g9_binned_by_amp.csv")
    scenario = pd.read_csv(out_dir / "g9_binned_by_scenario.csv")
    diag_lines = [
        "# E5A/E6/E8 分箱诊断报告",
        "",
        "## 读法",
        "",
        "分箱只看 test set，不用于拟合校准。重点看真实主峰幅值区间、场景/事件类型下的 RMSE、幅值比、方向错号、零线两侧明显相反率和尾段漂移风险。",
        "",
        "## 真实主峰幅值区间诊断",
        "",
        "| 版本 | 幅值区间(rad) | n | RMSE | 幅值比中位数 | 幅值不足率 | 主峰错号率 | 后段漂移风险 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in amp[amp["experiment_id"].isin(["E5A", "E6", "E8"])].iterrows():
        diag_lines.append(
            f"| {r['experiment_id']} | {r['amp_bin']} | {int(r['n'])} | {_fmt(r['sample_rmse_mean'])} | "
            f"{_fmt(r['amp_ratio_median_major'], 3)} | {_pct(r['under_amp_rate_major'])} | "
            f"{_pct(r['peak_side_wrong_at_true_peak_rate_major'])} | {_pct(r['tail_drift_risk_rate_major'])} |"
        )
    diag_lines += [
        "",
        "## 场景/事件诊断摘要",
        "",
        "完整表见 `g9_binned_by_scenario.csv`。下面列出每个版本中样本数较多且风险较高的场景。",
        "",
    ]
    risky = scenario[scenario["experiment_id"].isin(["E5A", "E6", "E8"]) & (scenario["n"] >= 20)].copy()
    risky["risk_score"] = (
        pd.to_numeric(risky["under_amp_rate_major"], errors="coerce").fillna(0)
        + pd.to_numeric(risky["tail_drift_risk_rate_major"], errors="coerce").fillna(0)
        + pd.to_numeric(risky["opposite_side_heavy_rate_major"], errors="coerce").fillna(0)
    )
    for exp_id, group in risky.groupby("experiment_id", sort=True):
        diag_lines.append(f"### {exp_id}")
        top = group.sort_values("risk_score", ascending=False).head(5)
        for _, r in top.iterrows():
            diag_lines.append(
                f"- `{r['road_type_anchor']}` / `{r['interaction_slice']}` / `{r['eval_morphology_label']}`："
                f"n={int(r['n'])}，RMSE={_fmt(r['sample_rmse_mean'])}，幅值不足={_pct(r['under_amp_rate_major'])}，"
                f"后段漂移={_pct(r['tail_drift_risk_rate_major'])}。"
            )
        diag_lines.append("")
    diag_lines += [
        "## 解释",
        "",
        "- E5A 的主要问题集中在中高幅值响应中，表现为幅值比偏低和后段漂移风险。",
        "- E6 能缓解 E5A 的幅值不足，但不是所有场景都变好，因此它更适合作为物理平衡候选。",
        "- E8 的退化主要不是方向错号大幅上升，而是强修正让部分 seed/场景的整体误差和后段漂移变差。",
    ]
    (out_dir / "g9_binned_diagnostics_report_cn.md").write_text("\n".join(diag_lines) + "\n", encoding="utf-8")

    gain_df = pd.read_csv(out_dir / "g9_calibration_chosen_gains.csv")
    cal_lines = [
        "# 校准实验报告",
        "",
        "## 方法",
        "",
        "校准只在 validation set 上拟合，不使用 test set。形式是最轻量的正号保持幅值后处理：",
        "",
        "`pred_steer_calibrated = gain * pred_steer`",
        "",
        "速度通道不变，不重新训练模型，不引入 EEG 或其他生理推理输入。每个版本/seed 的 gain 只从 validation 网格搜索选择。",
        "",
        "## validation 选择的 gain",
        "",
        "| 版本 | seed | gain | validation 说明 |",
        "|---|---:|---:|---|",
    ]
    for _, r in gain_df.iterrows():
        cal_lines.append(
            f"| {r['calibrated_experiment_id']} | {int(r['seed'])} | {float(r['gain']):.2f} | "
            f"val RMSE={_fmt(r['chosen_val_rmse'])}，val 幅值不足={_pct(r['chosen_val_under_amp_rate_major'])} |"
        )
    cal_lines += [
        "",
        "## test 门槛判断",
        "",
        f"- E5A-Cal：{gate['E5A-Cal']['decision']}。{gate['E5A-Cal']['reason']}",
        f"- E6-Cal：{gate['E6-Cal']['decision']}。{gate['E6-Cal']['reason']}",
        "",
        "固定 case 图：",
        "",
        "- `comparison_plots/g9_fixed_case_seed2026.png`",
        "- `comparison_plots/g9_fixed_case_seed2028.png`",
    ]
    (out_dir / "g9_calibration_report_cn.md").write_text("\n".join(cal_lines) + "\n", encoding="utf-8")

    final_lines = [
        "# 最终候选建议",
        "",
        "## 结论",
        "",
    ]
    if any(gate[x]["passes"] for x in ["E5A-Cal", "E6-Cal"]):
        passed = [x for x in ["E5A-Cal", "E6-Cal"] if gate[x]["passes"]]
        final_lines.append(f"校准版本 `{', '.join(passed)}` 通过 2026/2028 门槛，可以补 seed-2027 后再形成三种子最终表。")
    else:
        final_lines += [
            "E5A-Cal / E6-Cal 没有通过 2026/2028 门槛，不补 seed-2027。",
            "",
            "最终主候选保持为：",
            "",
            "- **E6**：物理幅值更平衡的主候选。",
            "- **E5A**：数值准确性候选，用来说明 EEG 教师蒸馏带来的平均指标收益。",
            "- **E8**：解释性 no-go 对照，用来说明过强主峰物理修正会牺牲整体轨迹拟合。",
        ]
    final_lines += [
        "",
        "## 汇报口径",
        "",
        "可以保守表述为：连续驾驶风格稳定有效；EEG 相关信息通过教师蒸馏可以提升无 EEG 推理学生；但直接堆生理输入或过强物理约束并不会自动带来更好的预测。当前最稳妥的模型路线是 E5A/E6 双候选，其中 E6 更适合承接用户提出的物理幅值可信问题。",
    ]
    (out_dir / "g9_final_candidate_recommendation_cn.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")


def _make_mean_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        "test_steer_rmse",
        "tail_rmse",
        "selection",
        "primary_rmse",
        "under_amp_rate_major",
        "severe_under_amp_rate_large",
        "peak_side_wrong_at_true_peak_rate_major",
        "peak_side_wrong_at_pred_peak_rate_major",
        "opposite_side_heavy_rate_major",
        "tail_drift_risk_rate_major",
        "median_amp_ratio_major",
    ]
    for (exp_id, split), group in seed_metrics.groupby(["experiment_id", "split"], sort=True):
        item: dict[str, Any] = {
            "experiment_id": exp_id,
            "experiment_name": LABELS.get(exp_id, exp_id),
            "split": split,
            "n_seeds": int(len(group)),
        }
        for col in metric_cols:
            vals = pd.to_numeric(group[col], errors="coerce")
            item[f"{col}_mean"] = float(vals.mean())
            item[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else 0.0
        rows.append(item)
    return pd.DataFrame(rows)


def _gate_decision(mean_df: pd.DataFrame) -> dict[str, Any]:
    test = mean_df[mean_df["split"].eq("test")].set_index("experiment_id")
    e5a = test.loc["E5A"]
    e6 = test.loc["E6"]
    out: dict[str, Any] = {}
    for cal_id in ["E5A-Cal", "E6-Cal"]:
        if cal_id not in test.index:
            out[cal_id] = {"passes": False, "decision": "NO-GO", "reason": "没有生成 test 评估。"}
            continue
        row = test.loc[cal_id]
        rmse_ok = row["test_steer_rmse_mean"] <= e6["test_steer_rmse_mean"] + 0.005
        tail_ok = row["tail_rmse_mean"] <= e6["tail_rmse_mean"] + 0.010
        sel_ok = row["selection_mean"] <= e6["selection_mean"] + 0.015
        under_ok = row["under_amp_rate_major_mean"] <= e5a["under_amp_rate_major_mean"] - 0.05
        dir_ok = row["peak_side_wrong_at_true_peak_rate_major_mean"] <= e6["peak_side_wrong_at_true_peak_rate_major_mean"] + 0.005
        drift_ok = row["tail_drift_risk_rate_major_mean"] <= e6["tail_drift_risk_rate_major_mean"] + 0.030
        passes = bool(rmse_ok and tail_ok and sel_ok and under_ok and dir_ok and drift_ok)
        failed = []
        if not rmse_ok:
            failed.append("RMSE 弱于 E6 门槛")
        if not tail_ok:
            failed.append("tail 退化")
        if not sel_ok:
            failed.append("selection 退化")
        if not under_ok:
            failed.append("幅值不足没有明显低于 E5A")
        if not dir_ok:
            failed.append("真实主峰错号高于 E6 门槛")
        if not drift_ok:
            failed.append("后段漂移风险高于 E6 门槛")
        out[cal_id] = {
            "passes": passes,
            "decision": "PASS" if passes else "NO-GO",
            "reason": "通过全部数值门槛。" if passes else "；".join(failed),
        }
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_df = _load_index(Path(args.prediction_index), seeds=[int(s) for s in args.seeds])
    index_df.to_csv(out_dir / "g9_source_prediction_index.csv", index=False, encoding="utf-8-sig")
    arrays = _build_arrays(index_df, args)

    seed_metric_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for _, row in index_df.iterrows():
        exp_id = str(row["experiment_id"])
        seed = int(row["seed"])
        arr = arrays[_array_key(exp_id, seed, "test")]
        metrics, detail = _compute_metrics(exp_id=exp_id, seed=seed, split="test", arrays=arr)
        seed_metric_rows.append(metrics)
        detail_frames.append(detail)

    gain_rows: list[dict[str, Any]] = []
    grid_frames: list[pd.DataFrame] = []
    calibrated_test_arrays: dict[tuple[str, int], dict[str, Any]] = {}
    for exp_id in ["E5A", "E6"]:
        for seed in [int(s) for s in args.seeds]:
            val_arr = arrays[_array_key(exp_id, seed, "val")]
            gain, grid = _fit_gain_for_model_seed(exp_id, seed, val_arr, args)
            grid_frames.append(grid)
            chosen = grid[np.isclose(grid["gain"].astype(float), gain)].iloc[0]
            cal_id = f"{exp_id}-Cal"
            gain_rows.append(
                {
                    "source_experiment_id": exp_id,
                    "calibrated_experiment_id": cal_id,
                    "seed": int(seed),
                    "gain": float(gain),
                    "chosen_val_rmse": float(chosen["val_rmse"]),
                    "chosen_val_tail": float(chosen["val_tail"]),
                    "chosen_val_selection": float(chosen["val_selection"]),
                    "chosen_val_under_amp_rate_major": float(chosen["val_under_amp_rate_major"]),
                    "raw_val_rmse": float(chosen["raw_val_rmse"]),
                    "raw_val_tail": float(chosen["raw_val_tail"]),
                    "raw_val_selection": float(chosen["raw_val_selection"]),
                    "raw_val_under_amp_rate_major": float(chosen["raw_val_under_amp_rate_major"]),
                    "within_tolerance": bool(chosen["within_tolerance"]),
                }
            )
            test_arr = arrays[_array_key(exp_id, seed, "test")]
            pred_cal = _apply_gain(test_arr["pred"], gain)
            cal_arr = dict(test_arr)
            cal_arr["pred"] = pred_cal
            calibrated_test_arrays[(cal_id, seed)] = cal_arr
            metrics, detail = _compute_metrics(exp_id=cal_id, seed=seed, split="test", arrays=test_arr, pred_override=pred_cal)
            seed_metric_rows.append(metrics)
            detail_frames.append(detail)
            artifact_paths = _save_calibrated_sequences(out_dir, cal_id, seed, test_arr, pred_cal)
            gain_rows[-1].update(artifact_paths)

    seed_metrics = pd.DataFrame(seed_metric_rows)
    seed_metrics.to_csv(out_dir / "g9_metric_summary_seed.csv", index=False, encoding="utf-8-sig")
    mean_metrics = _make_mean_summary(seed_metrics)
    mean_metrics.to_csv(out_dir / "g9_metric_summary_mean.csv", index=False, encoding="utf-8-sig")
    pd.concat(grid_frames, ignore_index=True).to_csv(out_dir / "g9_calibration_validation_grid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gain_rows).to_csv(out_dir / "g9_calibration_chosen_gains.csv", index=False, encoding="utf-8-sig")

    detail = pd.concat(detail_frames, ignore_index=True)
    detail.to_csv(out_dir / "g9_physical_detail.csv", index=False, encoding="utf-8-sig")
    diag_base = detail[detail["experiment_id"].isin(["E5A", "E6", "E8"]) & detail["split"].eq("test")].copy()
    _group_summary(diag_base, ["experiment_id", "amp_bin"]).to_csv(out_dir / "g9_binned_by_amp.csv", index=False, encoding="utf-8-sig")
    _group_summary(diag_base, ["experiment_id", "seed", "amp_bin"]).to_csv(out_dir / "g9_binned_by_seed_amp.csv", index=False, encoding="utf-8-sig")
    _group_summary(diag_base, ["experiment_id", "road_type_anchor", "interaction_slice", "eval_morphology_label"]).to_csv(
        out_dir / "g9_binned_by_scenario.csv", index=False, encoding="utf-8-sig"
    )
    _group_summary(diag_base, ["experiment_id", "effective_mechanism_tag"]).to_csv(out_dir / "g9_binned_by_event.csv", index=False, encoding="utf-8-sig")

    plot_rows: list[dict[str, Any]] = []
    for seed in [int(s) for s in args.seeds]:
        plot_arrays: dict[str, dict[str, Any]] = {
            exp_id: arrays[_array_key(exp_id, seed, "test")]
            for exp_id in ["E2", "E5A", "E6", "E8"]
            if _array_key(exp_id, seed, "test") in arrays
        }
        plot_arrays["E5A-Cal"] = calibrated_test_arrays[("E5A-Cal", seed)]
        plot_arrays["E6-Cal"] = calibrated_test_arrays[("E6-Cal", seed)]
        plot_path = _plot_fixed_cases(out_dir, seed, plot_arrays, Path(args.case_file))
        plot_rows.append({"seed": seed, "plot_file": str(plot_path)})
    pd.DataFrame(plot_rows).to_csv(out_dir / "g9_fixed_case_plot_index.csv", index=False, encoding="utf-8-sig")

    gate = _gate_decision(mean_metrics)
    visual_notes = {
        "E2": "强基准，幅值较稳但整体不是最优",
        "E5A": "数值最好，但固定图和审计均显示幅值压小",
        "E6": "幅值更平衡，未见系统性过强修正",
        "E8": "seed-2028 有过强修正/后段漂移，no-go",
        "E5A-Cal": "需看校准报告和固定图",
        "E6-Cal": "需看校准报告和固定图",
    }
    _write_reports(out_dir, gate=gate, visual_notes=visual_notes)

    artifact_lines = [
        "# G9 产物索引",
        "",
        "## 先看这些",
        "",
        "- `g9_candidate_convergence_report_cn.md`",
        "- `g9_binned_diagnostics_report_cn.md`",
        "- `g9_calibration_report_cn.md`",
        "- `g9_final_candidate_recommendation_cn.md`",
        "- `g9_metric_summary_mean.csv`",
        "- `g9_binned_by_amp.csv`",
        "- `comparison_plots/g9_fixed_case_seed2026.png`",
        "- `comparison_plots/g9_fixed_case_seed2028.png`",
        "",
        "## 校准数据",
        "",
        "- `g9_calibration_validation_grid.csv`",
        "- `g9_calibration_chosen_gains.csv`",
        "- `calibrated_predictions/`",
        "",
        "## 门槛结论",
        "",
        f"- E5A-Cal：{gate['E5A-Cal']['decision']}，{gate['E5A-Cal']['reason']}",
        f"- E6-Cal：{gate['E6-Cal']['decision']}，{gate['E6-Cal']['reason']}",
    ]
    (out_dir / "artifact_index_g9_20260508.md").write_text("\n".join(artifact_lines) + "\n", encoding="utf-8")
    (out_dir / "g9_gate_decision.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"G9 out_dir: {out_dir}")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
