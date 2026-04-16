from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

FS = 200
EVENT_BIN_SIZE = 20
DEFAULT_FUTURE_LEN = 400

STEER_CANDIDATES = (
    "zx|SteeringWheel",
    "SteeringWheel",
    "steer",
    "steering_wheel",
    "steering",
)


def _find_column(columns: Sequence[str], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
        alt = candidate.lower()
        for col in columns:
            if col.lower() == alt:
                return col
    return None


def load_steer_signal(vehicle_file: str | Path) -> np.ndarray:
    df = pd.read_csv(vehicle_file)
    steer_col = _find_column(df.columns.tolist(), STEER_CANDIDATES)
    if steer_col is None:
        raise ValueError(f"cannot find steer column in {vehicle_file}")
    steer = df[steer_col].to_numpy(dtype=np.float32, copy=False)
    return np.nan_to_num(steer, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(frozen=True)
class EventTargetConfig:
    future_len: int = DEFAULT_FUTURE_LEN
    bin_size: int = EVENT_BIN_SIZE
    turn_frac: float = 0.20
    turn_min_amp: float = 0.015
    reversal_frac: float = 0.30
    reversal_min_rate: float = 0.002
    reversal_min_gap: int = 3
    reversal_search_start: int = 0
    reversal_percentile: float = 70.0


def _event_time_to_bin(time_idx: int, valid_len: int, config: EventTargetConfig) -> int:
    if valid_len <= 0:
        return 0
    capped = min(max(int(time_idx), 0), max(int(valid_len) - 1, 0))
    max_bin = max((int(config.future_len) - 1) // int(config.bin_size), 0)
    return int(min(capped // int(config.bin_size), max_bin))


def sequence_to_event_targets(
    steer_deltas: Sequence[float],
    valid_len: int,
    config: EventTargetConfig | None = None,
) -> dict[str, float | int]:
    cfg = config or EventTargetConfig()
    valid_len = max(0, min(valid_len, cfg.future_len))
    arr = np.asarray(steer_deltas, dtype=np.float32)[:valid_len]
    if valid_len != arr.shape[0]:
        arr = arr[:valid_len]

    peak_idx = -1
    peak_bin = 0
    peak_direction = 1
    peak_amplitude = 0.0
    turn_idx = -1
    turn_bin = 0
    turn_dir = 1
    turn_amp = 0.0
    turn_confidence = 0.0
    reversal_idx = -1
    reversal_bin = 0
    reversal_rate = 0.0

    if valid_len > 0:
        abs_steer = np.abs(arr)
        peak_idx = int(np.argmax(abs_steer))
        peak_bin = _event_time_to_bin(peak_idx, valid_len, cfg)
        peak_direction = 1 if arr[peak_idx] >= 0.0 else 0
        peak_amplitude = float(abs_steer[peak_idx])

        max_abs = float(abs_steer.max())
        threshold = max(cfg.turn_min_amp, cfg.turn_frac * max(max_abs, 1e-6))
        candidates = np.where(abs_steer >= threshold)[0]
        if candidates.size:
            turn_idx = int(candidates[0])
            turn_bin = _event_time_to_bin(turn_idx, valid_len, cfg)
            turn_dir = 1 if arr[turn_idx] >= 0.0 else 0
            turn_amp = float(abs_steer[turn_idx])
            turn_confidence = float(turn_amp / max(max_abs, 1e-6))

        if valid_len > 1:
            deriv = np.diff(arr)
            abs_deriv = np.abs(deriv)
            if abs_deriv.size:
                cutoff = float(max(cfg.reversal_min_rate, cfg.reversal_frac * max(np.percentile(abs_deriv, cfg.reversal_percentile), 1e-6)))
                sign = np.sign(deriv).astype(np.int8)
                sign[abs_deriv < cutoff] = 0
                nonzero_idx = np.where(sign != 0)[0]
                search_floor = max(cfg.reversal_min_gap, cfg.reversal_search_start)
                if turn_idx >= 0:
                    search_floor = max(search_floor, turn_idx + 1)
                for j in range(len(nonzero_idx) - 1):
                    left = nonzero_idx[j]
                    right = nonzero_idx[j + 1]
                    if sign[left] * sign[right] < 0:
                        candidate_idx = int(right + 1)
                        if candidate_idx >= search_floor and candidate_idx < valid_len:
                            reversal_idx = candidate_idx
                            reversal_rate = float(abs_deriv[right])
                            reversal_bin = _event_time_to_bin(reversal_idx, valid_len, cfg)
                            break

    return {
        "first_major_turn_onset_has": 1.0 if turn_idx >= 0 else 0.0,
        "first_major_turn_onset_idx": turn_idx if turn_idx >= 0 else -1,
        "first_major_turn_onset_bin": turn_bin,
        "first_major_turn_direction": turn_dir,
        "first_major_turn_amplitude": turn_amp,
        "first_major_turn_confidence": turn_confidence,
        "first_reversal_has": 1.0 if reversal_idx >= 0 else 0.0,
        "first_reversal_idx": reversal_idx if reversal_idx >= 0 else -1,
        "first_reversal_bin": reversal_bin,
        "first_reversal_rate": reversal_rate,
        "main_peak_idx": peak_idx if peak_idx >= 0 else -1,
        "main_peak_bin": peak_bin,
        "main_peak_direction": peak_direction,
        "main_peak_amplitude": peak_amplitude,
        "valid_future_len": valid_len,
        "future_len": cfg.future_len,
    }


def describe_config(config: EventTargetConfig | None = None) -> str:
    cfg = config or EventTargetConfig()
    template = textwrap.dedent(
        """
        EventTargetConfig(future_len={future_len}, bin_size={bin_size}, turn_frac={turn_frac:.2f},
          turn_min_amp={turn_min_amp:.3f}, reversal_frac={reversal_frac:.2f},
          reversal_min_rate={reversal_min_rate:.3f}, reversal_min_gap={reversal_min_gap},
          reversal_search_start={reversal_search_start}, reversal_percentile={reversal_percentile})
        """
    )
    return template.format_map(vars(cfg))


__all__ = [
    "EventTargetConfig",
    "sequence_to_event_targets",
    "load_steer_signal",
    "describe_config",
]
