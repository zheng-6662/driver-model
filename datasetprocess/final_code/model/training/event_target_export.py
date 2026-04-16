from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_DIR))

from datasetprocess.final_code.model.training.event_targets import (
    EventTargetConfig,
    load_steer_signal,
    sequence_to_event_targets,
)
DEFAULT_MANIFEST = Path(
    "datasetprocess/final_code/model/training/protocol_allphase_control_v2_context_full2s/sample_manifest.csv"
)
OUTPUT_DIR = Path(
    "reports/event_plus_conditioned_trajectory_baseline_20260326/task_A_event_targets"
)
FS = 200
DEFAULT_FUTURE_LEN = EventTargetConfig().future_len


def _load_manifest(manifest_path: Path, max_rows: int) -> pd.DataFrame:
    cols = [
        "sample_key",
        "pool",
        "subj",
        "split",
        "vehicle_file",
        "anchor_idx",
        "valid_future_len",
        "future_len",
        "anchor_s",
        "trigger_type",
        "primary_score",
        "phase_type",
        "event_level",
    ]
    df = pd.read_csv(
        manifest_path,
        usecols=cols,
        nrows=max_rows,
        dtype={"anchor_idx": np.int64, "valid_future_len": np.float32},
        na_filter=False,
    )
    return df.reset_index(drop=True)


def _plot_case(
    key: str,
    sequence: np.ndarray,
    metadata: pd.Series,
    targets: dict[str, float | int],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    time_axis = np.arange(sequence.size) / FS
    ax.plot(time_axis, sequence, label="future steer Δ", color="#0c7c9f")
    ax.set_xlabel("s after anchor")
    ax.set_ylabel("steer Δ (rad)")
    ax.set_title(f"{key} | {metadata.pool} | {metadata.trigger_type}")
    ax.grid(True, linestyle=":", color="#aaaaaa")

    def _add_line(idx: int, label: str, color: str, linestyle: str = "--"):
        if 0 <= idx < sequence.size:
            ax.axvline(idx / FS, color=color, linestyle=linestyle, linewidth=1.2)
            ax.text(
                idx / FS,
                ax.get_ylim()[1] * 0.85,
                label,
                rotation=90,
                color=color,
                fontsize=8,
                ha="center",
                va="top",
            )

    _add_line(int(targets["first_major_turn_onset_idx"]), "turn", "#f59323")
    _add_line(int(targets["first_reversal_idx"]), "reversal", "#d32f2f")
    _add_line(int(targets["main_peak_idx"]), "peak", "#388e3c", linestyle=":")

    ax.legend(loc="upper right", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _collect_sequences(
    manifest: pd.DataFrame,
    max_samples: int,
    config: EventTargetConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, dict[str, float | int]]]:
    rows = []
    sequences: dict[str, np.ndarray] = {}
    targets_map: dict[str, dict[str, float | int]] = {}
    vehicle_cache: dict[str, np.ndarray] = {}

    for _, row in manifest.iterrows():
        vehicle_path = Path(row.vehicle_file)
        try:
            steer = vehicle_cache.setdefault(
                row.vehicle_file,
                load_steer_signal(vehicle_path),
            )
        except Exception as exc:
            print(f"skip {row.sample_key}: failed to load steer ({exc})")
            continue

        anchor_idx = int(row.anchor_idx)
        if anchor_idx >= steer.shape[0]:
            print(f"anchor idx {anchor_idx} outside {steer.shape[0]} for {row.sample_key}")
            continue

        valid_future = max(0, min(int(row.valid_future_len), config.future_len))
        future_start = anchor_idx + 1
        future_end = min(future_start + valid_future, steer.shape[0])
        actual_len = max(future_end - future_start, 0)
        steer_anchor = float(steer[anchor_idx])
        seq = steer[future_start:future_end] - steer_anchor
        seq = seq.astype(np.float32, copy=False)

        targets = sequence_to_event_targets(seq, actual_len, config=config)
        row_data = row.to_dict()
        row_data.update(targets)
        rows.append(row_data)
        sequences[row.sample_key] = seq
        targets_map[row.sample_key] = targets

    return pd.DataFrame(rows), sequences, targets_map


def _write_stats(df: pd.DataFrame, output_dir: Path) -> Path:
    metrics = [
        {"metric": "total_samples", "value": df.shape[0]},
        {
            "metric": "turn_coverage",
            "value": df["first_major_turn_onset_has"].mean() if "first_major_turn_onset_has" in df else 0.0,
        },
        {
            "metric": "reversal_coverage",
            "value": df["first_reversal_has"].mean() if "first_reversal_has" in df else 0.0,
        },
        {
            "metric": "main_peak_positive_ratio",
            "value": df["main_peak_direction"].mean() if "main_peak_direction" in df else 0.0,
        },
        {"metric": "valid_future_len_mean", "value": df["valid_future_len"].mean()},
        {"metric": "turn_amplitude_mean", "value": df["first_major_turn_amplitude"].mean()},
        {"metric": "reversal_rate_mean", "value": df["first_reversal_rate"].mean()},
    ]
    stats_path = output_dir / "event_target_stats.csv"
    pd.DataFrame(metrics).to_csv(stats_path, index=False)
    return stats_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build steering event targets for the latest baseline.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = EventTargetConfig()

    print(f"loading manifest from {args.manifest}")
    manifest = _load_manifest(args.manifest, args.max_samples)
    print(f"processing {manifest.shape[0]} rows (<= max {args.max_samples})")

    df, sequences, targets_map = _collect_sequences(manifest, args.max_samples, config)
    output_csv = args.output_dir / "event_targets.csv"
    df.to_csv(output_csv, index=False)
    print(f"wrote event targets to {output_csv}")

    stats_path = _write_stats(df, args.output_dir)
    print(f"stats saved to {stats_path}")

    cases = {
        "positive_turn": df[
            (df["first_major_turn_onset_has"] == 1.0) & (df["first_major_turn_direction"] == 1)
        ],
        "negative_turn": df[
            (df["first_major_turn_onset_has"] == 1.0) & (df["first_major_turn_direction"] == 0)
        ],
        "no_turn": df[df["first_major_turn_onset_has"] == 0.0],
    }

    for label, subset in cases.items():
        if subset.empty:
            continue
        row = subset.iloc[0]
        seq = sequences.get(row.sample_key)
        if seq is None or seq.size == 0:
            continue
        target = targets_map[row.sample_key]
        plot_path = args.output_dir / f"case_{label}.png"
        _plot_case(row.sample_key, seq, row, target, plot_path)
        print(f"plotted {label} case to {plot_path}")


if __name__ == "__main__":
    main()
