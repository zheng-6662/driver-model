# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
DEFAULT_SUMMARY_DIR = REPORTS_DIR / "style_physio_eeg_e5_distill_summary_20260508"
DEFAULT_SEED_METRICS = DEFAULT_SUMMARY_DIR / "seed_wise_metrics.csv"
DEFAULT_REGRESSIONS = DEFAULT_SUMMARY_DIR / "shape_audit" / "top_regression_samples_e5a_vs_e2.csv"
DEFAULT_OUT_DIR = DEFAULT_SUMMARY_DIR / "regression_comparison_plots"

MODEL_LABELS = {
    "E2": "E2 baseline",
    "E4": "E4 EEG input",
    "E5A": "E5A EEG teacher / no-EEG student",
}
MODEL_COLORS = {
    "true": "#111111",
    "E2": "#1f77b4",
    "E4": "#ff7f0e",
    "E5A": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot top E5A-vs-E2 regression cases.")
    parser.add_argument("--seed-metrics", default=str(DEFAULT_SEED_METRICS))
    parser.add_argument("--regression-csv", default=str(DEFAULT_REGRESSIONS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--top-per-seed", type=int, default=6)
    return parser.parse_args()


def _load_sequences(run_root: Path) -> dict[str, Any]:
    path = run_root / "prediction_figures" / "test" / "prediction_sequences.npz"
    if not path.exists():
        raise FileNotFoundError(f"prediction sequence file not found: {path}")
    arrays = np.load(path, allow_pickle=True)
    sample_keys = arrays["sample_key"].astype(str)
    return {
        "pred": arrays["pred"],
        "true": arrays["true"],
        "mask": arrays["mask"],
        "index": {key: idx for idx, key in enumerate(sample_keys.tolist())},
    }


def _short_key(sample_key: str) -> str:
    parts = sample_key.split("::")
    if len(parts) >= 3:
        return f"{parts[0]} #{parts[2]}"
    return sample_key[:42]


def _seed_sequences(seed_metrics: pd.DataFrame, seed: int) -> dict[str, dict[str, Any]]:
    sequences: dict[str, dict[str, Any]] = {}
    for exp_id in ["E2", "E4", "E5A"]:
        matched = seed_metrics[
            seed_metrics["experiment_id"].astype(str).eq(exp_id)
            & seed_metrics["seed"].astype(int).eq(int(seed))
        ]
        if matched.empty:
            raise ValueError(f"missing {exp_id} seed={seed} in seed metrics")
        sequences[exp_id] = _load_sequences(Path(str(matched.iloc[0]["run_root"])))
    return sequences


def _plot_seed(out_path: Path, seed: int, cases: pd.DataFrame, sequences: dict[str, dict[str, Any]]) -> pd.DataFrame:
    cols = 2
    rows_n = int(np.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(16, max(4.0, rows_n * 3.7)), squeeze=False)
    axes_flat = axes.flatten()
    rows: list[dict[str, Any]] = []
    for ax, (_, case) in zip(axes_flat, cases.iterrows()):
        sample_key = str(case["sample_key"])
        idx = sequences["E5A"]["index"].get(sample_key)
        if idx is None:
            ax.axis("off")
            continue
        valid_len = int(np.sum(sequences["E5A"]["mask"][idx] > 0.5))
        valid_len = max(valid_len, 1)
        times = np.arange(valid_len, dtype=float) / 200.0
        truth = sequences["E5A"]["true"][idx, :valid_len, 0]
        ax.plot(times, np.degrees(truth), color=MODEL_COLORS["true"], linewidth=2.0, label="True")
        for exp_id, seq in sequences.items():
            model_idx = seq["index"].get(sample_key)
            if model_idx is None:
                continue
            pred = seq["pred"][model_idx, :valid_len, 0]
            ax.plot(
                times,
                np.degrees(pred),
                color=MODEL_COLORS[exp_id],
                linewidth=1.35,
                label=MODEL_LABELS[exp_id],
            )
        delta = float(case["delta_rmse_2s_abs_steer"])
        title = (
            f"{_short_key(sample_key)} | delta RMSE={delta:.4f}\n"
            f"{case.get('phase_type_E5A', '')}, {case.get('road_type_anchor_E5A', '')}, "
            f"{case.get('eval_morphology_label_E5A', '')}"
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time after anchor (s)")
        ax.set_ylabel("Steering wheel angle (deg)")
        ax.grid(alpha=0.24)
        ax.legend(fontsize=7, loc="best")
        rows.append(
            {
                "seed": int(seed),
                "plot_file": str(out_path),
                "sample_key": sample_key,
                "delta_rmse_2s_abs_steer": delta,
                "delta_rmse_tail_abs_steer": float(case.get("delta_rmse_tail_abs_steer", np.nan)),
                "delta_peak_time_abs_err_s": float(case.get("delta_peak_time_abs_err_s", np.nan)),
            }
        )
    for ax in axes_flat[len(cases) :]:
        ax.axis("off")
    fig.suptitle(f"Top E5A-vs-E2 regression cases, seed {seed}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    seed_metrics = pd.read_csv(args.seed_metrics)
    regressions = pd.read_csv(args.regression_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[pd.DataFrame] = []
    for seed, group in regressions.groupby("seed"):
        cases = group.sort_values("delta_rmse_2s_abs_steer", ascending=False).head(int(args.top_per_seed))
        sequences = _seed_sequences(seed_metrics, int(seed))
        out_path = out_dir / f"top_regression_comparison_seed{int(seed)}.png"
        index_rows.append(_plot_seed(out_path, int(seed), cases, sequences))

    index_df = pd.concat(index_rows, ignore_index=True)
    index_df.to_csv(out_dir / "top_regression_plot_index.csv", index=False, encoding="utf-8-sig")
    print(f"top_regression_plot_index: {out_dir / 'top_regression_plot_index.csv'}")
    for path in sorted(out_dir.glob("top_regression_comparison_seed*.png")):
        print(f"top_regression_plot: {path}")


if __name__ == "__main__":
    main()
