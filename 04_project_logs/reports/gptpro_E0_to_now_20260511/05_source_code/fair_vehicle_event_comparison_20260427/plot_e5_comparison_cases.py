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
DEFAULT_CASE_FILE = THIS_DIR / "shared_prediction_cases_test.csv"
DEFAULT_OUT_DIR = DEFAULT_SUMMARY_DIR / "comparison_plots"

MODEL_LABELS = {
    "E2": "E2 baseline",
    "E4": "E4 EEG input",
    "E5A": "E5A EEG teacher / no-EEG student",
    "E5B": "E5B EEG teacher / no-EEG physio student",
    "E6": "E6 physical-loss repair",
    "E7A": "E7A EEG-only semantic state",
    "E7B": "E7B raw EEG-only",
    "E7C": "E7C raw non-EEG physio",
    "E8": "E8 reliable teacher + peak physical",
    "E10A": "E10A HR-only",
    "E10B": "E10B EDA-only",
    "E10C": "E10C EMG-only",
}
MODEL_COLORS = {
    "true": "#111111",
    "E2": "#1f77b4",
    "E4": "#ff7f0e",
    "E5A": "#2ca02c",
    "E5B": "#9467bd",
    "E6": "#d62728",
    "E7A": "#17becf",
    "E7B": "#8c564b",
    "E7C": "#bcbd22",
    "E8": "#e377c2",
    "E10A": "#ff7f0e",
    "E10B": "#2ca02c",
    "E10C": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fixed-case E2/E4/E5A prediction comparisons.")
    parser.add_argument("--seed-metrics", default=str(DEFAULT_SEED_METRICS))
    parser.add_argument("--case-file", default=str(DEFAULT_CASE_FILE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--experiments", nargs="+", default=["E2", "E4", "E5A"])
    return parser.parse_args()


def _load_sequences(run_root: Path) -> dict[str, Any]:
    path = run_root / "prediction_figures" / "test" / "prediction_sequences.npz"
    if not path.exists():
        raise FileNotFoundError(f"prediction sequence file not found: {path}")
    arrays = np.load(path, allow_pickle=True)
    sample_keys = arrays["sample_key"].astype(str)
    index = {key: idx for idx, key in enumerate(sample_keys.tolist())}
    return {
        "pred": arrays["pred"],
        "true": arrays["true"],
        "mask": arrays["mask"],
        "sample_key": sample_keys,
        "index": index,
    }


def _short_key(sample_key: str) -> str:
    parts = sample_key.split("::")
    if len(parts) >= 3:
        return f"{parts[0]} #{parts[2]}"
    return sample_key[:42]


def _plot_seed(
    out_path: Path,
    seed: int,
    cases: pd.DataFrame,
    sequences: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(4, 2, figsize=(16, 13.5), squeeze=False)
    axes_flat = axes.flatten()
    rows: list[dict[str, Any]] = []
    anchor_exp = next(iter(sequences))
    for ax, (_, case) in zip(axes_flat, cases.iterrows()):
        sample_key = str(case["sample_key"])
        if sample_key not in sequences[anchor_exp]["index"]:
            ax.axis("off")
            continue
        idx = sequences[anchor_exp]["index"][sample_key]
        valid_len = int(np.sum(sequences[anchor_exp]["mask"][idx] > 0.5))
        valid_len = max(valid_len, 1)
        times = np.arange(valid_len, dtype=float) / 200.0
        truth = sequences[anchor_exp]["true"][idx, :valid_len, 0]
        ax.plot(times, np.degrees(truth), color=MODEL_COLORS["true"], linewidth=2.0, label="True")

        for exp_id, seq in sequences.items():
            other_idx = seq["index"].get(sample_key)
            if other_idx is None:
                continue
            pred = seq["pred"][other_idx, :valid_len, 0]
            ax.plot(
                times,
                np.degrees(pred),
                color=MODEL_COLORS.get(exp_id, None),
                linewidth=1.35,
                label=MODEL_LABELS.get(exp_id, exp_id),
            )
        title = (
            f"{int(case.get('plot_index', len(rows) + 1)):02d} "
            f"{case.get('selection_tag', 'case')} | {_short_key(sample_key)}"
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
                "selection_tag": str(case.get("selection_tag", "case")),
            }
        )
    for ax in axes_flat[len(cases) :]:
        ax.axis("off")
    title = " / ".join(sequences.keys())
    fig.suptitle(f"{title} fixed-case comparison, seed {seed}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return rows


def main() -> None:
    args = parse_args()
    seed_metrics = pd.read_csv(args.seed_metrics)
    case_path = Path(args.case_file)
    cases = pd.read_csv(case_path).head(8)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    seed_sets = []
    for exp_id in args.experiments:
        exp_seeds = set(
            seed_metrics[seed_metrics["experiment_id"].astype(str).eq(exp_id)]["seed"].astype(int).unique()
        )
        seed_sets.append(exp_seeds)
    if not seed_sets:
        raise ValueError("No experiments requested.")
    seeds = sorted(set.intersection(*seed_sets))
    for seed in seeds:
        sequences: dict[str, dict[str, Any]] = {}
        for exp_id in args.experiments:
            matched = seed_metrics[
                seed_metrics["experiment_id"].astype(str).eq(exp_id)
                & seed_metrics["seed"].astype(int).eq(int(seed))
            ]
            if matched.empty:
                continue
            run_root = Path(str(matched.iloc[0]["run_root"]))
            sequences[exp_id] = _load_sequences(run_root)
        missing = set(args.experiments).difference(sequences)
        if missing:
            raise ValueError(f"missing sequence files for seed={seed}: {sorted(missing)}")
        out_path = out_dir / f"comparison_overview_seed{seed}.png"
        rows.extend(_plot_seed(out_path, int(seed), cases, sequences))

    index_df = pd.DataFrame(rows)
    index_df.to_csv(out_dir / "comparison_plot_index.csv", index=False, encoding="utf-8-sig")
    print(f"comparison_plot_index: {out_dir / 'comparison_plot_index.csv'}")
    for path in sorted(out_dir.glob("comparison_overview_seed*.png")):
        print(f"comparison_plot: {path}")


if __name__ == "__main__":
    main()
