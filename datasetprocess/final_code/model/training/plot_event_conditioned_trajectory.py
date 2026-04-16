from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

EVENT_DISPLAY: Mapping[str, str] = {
    "first_major_turn_onset": "First Major Turn",
    "first_reversal": "First Reversal",
    "main_peak": "Main Peak",
}

EVENT_COLOR: Mapping[str, str] = {
    "first_major_turn_onset": "tab:orange",
    "first_reversal": "tab:green",
    "main_peak": "tab:red",
}

EVENT_LINESTYLE: Mapping[str, str] = {
    "first_major_turn_onset": "--",
    "first_reversal": "-.",
    "main_peak": ":",
}

def _to_degrees(values: np.ndarray) -> np.ndarray:
    return np.degrees(values)


def plot_steering_trace(
    ax: plt.Axes,
    times: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    events: Mapping[str, float],
    title: str,
) -> None:
    ax.plot(times, _to_degrees(truth), label="True", color="tab:blue", linewidth=1.6)
    ax.plot(times, _to_degrees(prediction), label="Pred", color="tab:orange", linewidth=1.2)
    ymin, ymax = ax.get_ylim()
    for event_name, event_time in events.items():
        if event_time is None or np.isnan(event_time):
            continue
        ax.axvline(event_time, color=EVENT_COLOR.get(event_name, "#333333"), linestyle=EVENT_LINESTYLE.get(event_name, "--"), linewidth=1.2)
        label = EVENT_DISPLAY.get(event_name, event_name)
        ax.text(
            event_time,
            0.95 * ymax,
            label,
            rotation=90,
            va="top",
            ha="right",
            color=EVENT_COLOR.get(event_name, "#333333"),
            fontsize=8,
            bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.7},
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Steering (deg)")
    ax.set_xlim(times[0], times[-1])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def save_case_plot(
    path: Path,
    times: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    events: Mapping[str, float],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plot_steering_trace(ax, times, truth, prediction, events, title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_overview_mosaic(samples: Sequence[Mapping[str, object]], out_path: Path) -> None:
    cols = 2
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 3.5), squeeze=False)
    axes_flat = axes.flatten()
    for ax, sample in zip(axes_flat, samples):
        plot_steering_trace(
            ax,
            sample["times"],
            sample["truth"],
            sample["prediction"],
            sample["events"],
            sample["title"],
        )
    for ax in axes_flat[len(samples) :]:
        fig.delaxes(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
