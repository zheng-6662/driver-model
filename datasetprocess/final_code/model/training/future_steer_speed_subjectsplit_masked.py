from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from protocol_d3_response_aligned_extended_v1.dataset_builder import (
    build_protocol_manifest,
    load_protocol_config,
)


FS = 200
WIN_LEN = 600
FUTURE_LEN = 400
PRIMARY_HORIZON_LEN = 300
PRIMARY_BIN = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3
DEFAULT_SEEDS = (2025, 2026, 2027)
LTR_COEFF = 0.11243
EPS = 1e-6
EVENT_BIN_SIZE = 20

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULT_ROOT = PROJECT_ROOT / "tmp" / "single_output_d3_runs"

ROLL_CANDIDATES = ("zx|roll", "roll", "Roll")
STEER_CANDIDATES = ("zx|SteeringWheel", "SteeringWheel", "steer")
YAWRATE_CANDIDATES = ("vyaw", "zx|vyaw", "YawRate", "zx|YawRate", "yaw_rate")
SPEED_CANDIDATES = ("zx|vx", "Vx", "vx", "Speed", "speed")
Z_CANDIDATES = ("zx|z", "z", "Z")
AY_CANDIDATES = ("zx|ay", "ay", "Ay", "lat_acc")
AX_CANDIDATES = ("zx|ax", "ax", "Ax", "Long_acc")
LANE_CANDIDATES = ("lateraldistance", "lateralDistance", "lateraldistance_start")
CURVE_CANDIDATES = ("zx1|lanecurvatureXY", "laneCurvature", "lanecurvature_start")
ROADTYPE_CANDIDATES = ("road_type_fixed", "road_type", "roadType_fixed")
REFOK_CANDIDATES = ("ref_nn_ok", "ref_ok", "refnn_ok")
YAW_CANDIDATES = ("zx|yaw", "yaw", "Yaw")

_BANK_CACHE: dict[str, "VehicleFeatureBank"] = {}


def save_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_col(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def unwrap_lane_center_signal(
    x: np.ndarray,
    lane_width: float = 3.5,
    jump_thr: float = 1.8,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    y = x.copy()
    offset = 0.0
    for i in range(1, y.size):
        if not np.isfinite(y[i]) or not np.isfinite(y[i - 1]):
            continue
        diff = y[i] - y[i - 1]
        if diff > jump_thr:
            k = int(np.round(diff / lane_width)) or 1
            offset -= k * lane_width
        elif diff < -jump_thr:
            k = int(np.round((-diff) / lane_width)) or 1
            offset += k * lane_width
        y[i] = y[i] + offset
    return y.astype(np.float32)


def speed_series_to_mps(values: np.ndarray, col_name: str | None) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32)
    if not col_name:
        return out
    lower = col_name.lower()
    if "km/h" in lower or "kmh" in lower or lower.endswith("|v_km/h"):
        out = out / 3.6
    return out.astype(np.float32)


@dataclass
class VehicleFeatureBank:
    X_all: np.ndarray
    steer: np.ndarray
    speed: np.ndarray
    feature_cols: list[str]
    steer_rate_idx: int
    ay_idx: int
    yawrate_idx: int
    curve_idx: int


def _safe_array(series: pd.Series) -> np.ndarray:
    arr = series.to_numpy(dtype=np.float32, copy=True)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def load_vehicle_feature_bank(vehicle_file: str) -> VehicleFeatureBank:
    if vehicle_file in _BANK_CACHE:
        return _BANK_CACHE[vehicle_file]

    df_v = pd.read_csv(vehicle_file)
    cols = df_v.columns.tolist()

    col_roll = find_col(cols, ROLL_CANDIDATES)
    col_steer = find_col(cols, STEER_CANDIDATES)
    col_yawrate = find_col(cols, YAWRATE_CANDIDATES)
    col_speed = find_col(cols, SPEED_CANDIDATES)
    col_z = find_col(cols, Z_CANDIDATES)
    col_ay = find_col(cols, AY_CANDIDATES)
    col_ax = find_col(cols, AX_CANDIDATES)
    col_lane = find_col(cols, LANE_CANDIDATES)
    col_curve = find_col(cols, CURVE_CANDIDATES)
    col_roadtype = find_col(cols, ROADTYPE_CANDIDATES)
    col_refok = find_col(cols, REFOK_CANDIDATES)
    col_yaw = find_col(cols, YAW_CANDIDATES)

    essential = [col_roll, col_steer, col_yawrate, col_speed, col_z, col_ay, col_ax, col_curve, col_yaw]
    if any(c is None for c in essential):
        raise ValueError(f"missing essential D3 feature columns for {vehicle_file}")

    steer = _safe_array(df_v[col_steer])
    speed = speed_series_to_mps(_safe_array(df_v[col_speed]), col_speed)
    roll = _safe_array(df_v[col_roll])
    yawrate = _safe_array(df_v[col_yawrate])
    ay = _safe_array(df_v[col_ay])
    ax = _safe_array(df_v[col_ax])
    z = _safe_array(df_v[col_z])
    curve = _safe_array(df_v[col_curve])
    yaw = _safe_array(df_v[col_yaw])

    blocks: list[tuple[str, np.ndarray]] = [
        (col_roll, roll),
        (col_yawrate, yawrate),
        (col_ay, ay),
        (col_ax, ax),
        (col_speed, speed),
        (col_z, z),
    ]

    if col_lane is not None:
        lane_err = _safe_array(df_v[col_lane])
        lane_rate = np.gradient(lane_err, 1.0 / FS).astype(np.float32)
        lane_unwrap = unwrap_lane_center_signal(lane_err)
        lane_unwrap_rate = np.gradient(lane_unwrap, 1.0 / FS).astype(np.float32)
        blocks.extend(
            [
                (col_lane, lane_err),
                ("lane_rate", lane_rate),
                ("lane_unwrap", lane_unwrap),
                ("lane_unwrap_rate", lane_unwrap_rate),
            ]
        )

    blocks.extend(
        [
            (col_curve, curve),
            (col_yaw, yaw),
            (col_steer, steer),
            ("LTR_est", (ay * LTR_COEFF).astype(np.float32)),
            ("steer_rate", np.gradient(steer, 1.0 / FS).astype(np.float32)),
            ("speed_rate", np.gradient(speed, 1.0 / FS).astype(np.float32)),
        ]
    )

    feature_cols = [name for name, _ in blocks]
    X_all = np.stack([arr for _, arr in blocks], axis=1).astype(np.float32)

    bank = VehicleFeatureBank(
        X_all=X_all,
        steer=steer,
        speed=speed,
        feature_cols=feature_cols,
        steer_rate_idx=feature_cols.index("steer_rate"),
        ay_idx=feature_cols.index(col_ay),
        yawrate_idx=feature_cols.index(col_yawrate),
        curve_idx=feature_cols.index(col_curve),
    )
    _BANK_CACHE[vehicle_file] = bank
    return bank


def _make_sample(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bank = load_vehicle_feature_bank(str(row["vehicle_file"]))
    anchor_idx = int(row["anchor_idx"])
    valid_future_len = int(min(int(row["valid_future_len"]), FUTURE_LEN))
    hist_start = anchor_idx - WIN_LEN + 1
    future_start = anchor_idx + 1
    future_end = future_start + valid_future_len

    if hist_start < 0:
        raise ValueError(f"history window underflow for {row['vehicle_file']} @ {anchor_idx}")
    if future_end > bank.X_all.shape[0]:
        valid_future_len = max(bank.X_all.shape[0] - future_start, 0)
        future_end = future_start + valid_future_len

    x_win = bank.X_all[hist_start : anchor_idx + 1]
    if x_win.shape[0] != WIN_LEN:
        raise ValueError(f"unexpected history length {x_win.shape[0]} for {row['vehicle_file']}")

    y = np.zeros((FUTURE_LEN, 2), dtype=np.float32)
    curve_future = np.zeros((FUTURE_LEN,), dtype=np.float32)
    mask = np.zeros((FUTURE_LEN,), dtype=np.float32)

    steer_anchor = float(bank.steer[anchor_idx])
    speed_anchor = float(bank.speed[anchor_idx])
    if valid_future_len > 0:
        y[:valid_future_len, 0] = bank.steer[future_start:future_end] - steer_anchor
        y[:valid_future_len, 1] = bank.speed[future_start:future_end] - speed_anchor
        curve_future[:valid_future_len] = bank.X_all[future_start:future_end, bank.curve_idx]
        mask[:valid_future_len] = 1.0

    ctx = np.array(
        [
            steer_anchor,
            speed_anchor,
            float(bank.X_all[anchor_idx, bank.steer_rate_idx]),
            float(bank.X_all[anchor_idx, bank.ay_idx]),
            float(bank.X_all[anchor_idx, bank.yawrate_idx]),
        ],
        dtype=np.float32,
    )
    return x_win.astype(np.float32), y, curve_future, ctx, mask


def build_all_samples(
    protocol_config_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    meta_df = build_protocol_manifest(protocol_config_path)
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    curve_list: list[np.ndarray] = []
    ctx_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []

    for _, row in meta_df.iterrows():
        x_win, y_seq, curve_future, ctx, future_mask = _make_sample(row)
        X_list.append(x_win)
        y_list.append(y_seq)
        curve_list.append(curve_future)
        ctx_list.append(ctx)
        mask_list.append(future_mask)

    return (
        np.stack(X_list).astype(np.float32),
        np.stack(y_list).astype(np.float32),
        np.stack(curve_list).astype(np.float32),
        np.stack(ctx_list).astype(np.float32),
        np.stack(mask_list).astype(np.float32),
        meta_df.reset_index(drop=True).copy(),
    )


def normalize_inputs(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    curve_pool: np.ndarray,
    ctx_pool: np.ndarray,
    train_idx: list[int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    X_train = X_pool[train_idx].reshape(-1, X_pool.shape[-1])
    y_train = y_pool[train_idx].reshape(-1, y_pool.shape[-1])
    curve_train = curve_pool[train_idx].reshape(-1, 1)
    ctx_train = ctx_pool[train_idx]

    feat_mean = X_train.mean(axis=0).astype(np.float32)
    feat_std = np.clip(X_train.std(axis=0).astype(np.float32), EPS, None)
    y_mean = y_train.mean(axis=0).astype(np.float32)
    y_std = np.clip(y_train.std(axis=0).astype(np.float32), EPS, None)
    curve_mean = curve_train.mean(axis=0).astype(np.float32)
    curve_std = np.clip(curve_train.std(axis=0).astype(np.float32), EPS, None)
    ctx_mean = ctx_train.mean(axis=0).astype(np.float32)
    ctx_std = np.clip(ctx_train.std(axis=0).astype(np.float32), EPS, None)

    X_norm = ((X_pool - feat_mean.reshape(1, 1, -1)) / feat_std.reshape(1, 1, -1)).astype(np.float32)
    stats = {
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "curve_mean": curve_mean,
        "curve_std": curve_std,
        "ctx_mean": ctx_mean,
        "ctx_std": ctx_std,
    }
    return X_norm, stats


def _event_time_to_bin(time_idx: int, valid_len: int, bin_size: int, future_len: int) -> int:
    if valid_len <= 0:
        return 0
    capped = min(max(int(time_idx), 0), max(int(valid_len) - 1, 0))
    max_bin = max((int(future_len) - 1) // int(bin_size), 0)
    return int(min(capped // int(bin_size), max_bin))


def build_structure_event_targets(
    y_pool: np.ndarray,
    mask_pool: np.ndarray,
    future_len: int = FUTURE_LEN,
    bin_size: int = EVENT_BIN_SIZE,
    turn_frac: float = 0.20,
    turn_min_amp: float = 0.015,
    reversal_frac: float = 0.30,
    reversal_min_rate: float = 0.002,
) -> dict[str, np.ndarray]:
    n = int(y_pool.shape[0])
    turn_has = np.zeros((n,), dtype=np.float32)
    turn_bin = np.zeros((n,), dtype=np.int64)
    turn_dir = np.zeros((n,), dtype=np.int64)
    reversal_has = np.zeros((n,), dtype=np.float32)
    reversal_bin = np.zeros((n,), dtype=np.int64)
    peak_bin = np.zeros((n,), dtype=np.int64)

    for idx in range(n):
        valid_len = int(mask_pool[idx].sum())
        if valid_len <= 0:
            continue
        steer = np.asarray(y_pool[idx, :valid_len, 0], dtype=np.float32)
        abs_steer = np.abs(steer)
        peak_idx = int(np.argmax(abs_steer))
        peak_bin[idx] = _event_time_to_bin(peak_idx, valid_len, bin_size, future_len)

        max_abs = float(abs_steer.max()) if abs_steer.size else 0.0
        turn_thr = max(float(turn_min_amp), float(turn_frac) * max(max_abs, 1e-6))
        turn_candidates = np.where(abs_steer >= turn_thr)[0]
        if turn_candidates.size > 0:
            first_turn_idx = int(turn_candidates[0])
            turn_has[idx] = 1.0
            turn_bin[idx] = _event_time_to_bin(first_turn_idx, valid_len, bin_size, future_len)
            turn_dir[idx] = 1 if float(steer[first_turn_idx]) >= 0.0 else 0

        if valid_len <= 2:
            continue
        d1 = np.diff(steer)
        abs_d1 = np.abs(d1)
        if abs_d1.size == 0:
            continue
        reversal_thr = max(float(reversal_min_rate), float(reversal_frac) * max(float(np.percentile(abs_d1, 70)), 1e-6))
        sign = np.sign(d1).astype(np.int32)
        sign[abs_d1 < reversal_thr] = 0
        nz = np.where(sign != 0)[0]
        if nz.size < 2:
            continue
        sign_nz = sign[nz]
        changes = np.where(sign_nz[1:] * sign_nz[:-1] < 0)[0]
        if changes.size == 0:
            continue
        first_rev_idx = int(nz[changes[0] + 1] + 1)
        reversal_has[idx] = 1.0
        reversal_bin[idx] = _event_time_to_bin(first_rev_idx, valid_len, bin_size, future_len)

    return {
        "first_turn_has": turn_has,
        "first_turn_bin": turn_bin,
        "first_turn_dir": turn_dir,
        "first_reversal_has": reversal_has,
        "first_reversal_bin": reversal_bin,
        "major_peak_bin": peak_bin,
    }


class ControlDataset(Dataset):
    def __init__(
        self,
        X_norm: np.ndarray,
        y_pool: np.ndarray,
        curve_pool: np.ndarray,
        ctx_pool: np.ndarray,
        mask_pool: np.ndarray,
        norm_stats: dict[str, np.ndarray],
        meta_df: pd.DataFrame | None = None,
        mechanism_ids: np.ndarray | None = None,
    ) -> None:
        self.src = X_norm.astype(np.float32)
        self.y = y_pool.astype(np.float32)
        self.curve = curve_pool.astype(np.float32)
        self.ctx = ctx_pool.astype(np.float32)
        self.mask = mask_pool.astype(np.float32)
        self.meta_df = None if meta_df is None else meta_df.reset_index(drop=True).copy()
        self.mechanism_ids = None if mechanism_ids is None else mechanism_ids.astype(np.int64)
        self.event_targets = build_structure_event_targets(self.y, self.mask, future_len=int(self.y.shape[1]))
        self.y_norm = ((self.y - norm_stats["y_mean"].reshape(1, 1, -1)) / norm_stats["y_std"].reshape(1, 1, -1)).astype(np.float32)
        self.curve_norm = (
            (self.curve - norm_stats["curve_mean"].reshape(1,)) / norm_stats["curve_std"].reshape(1,)
        ).astype(np.float32)
        self.ctx_norm = (
            (self.ctx - norm_stats["ctx_mean"].reshape(1, -1)) / norm_stats["ctx_std"].reshape(1, -1)
        ).astype(np.float32)

    def __len__(self) -> int:
        return int(self.src.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {
            "src": torch.from_numpy(self.src[idx]),
            "y_true": torch.from_numpy(self.y_norm[idx]),
            "curve_norm": torch.from_numpy(self.curve_norm[idx]),
            "ctx": torch.from_numpy(self.ctx_norm[idx]),
            "ctx_raw": torch.from_numpy(self.ctx[idx]),
            "event_mask": torch.from_numpy(self.mask[idx]),
            "first_turn_has": torch.tensor(self.event_targets["first_turn_has"][idx], dtype=torch.float32),
            "first_turn_bin": torch.tensor(self.event_targets["first_turn_bin"][idx], dtype=torch.long),
            "first_turn_dir": torch.tensor(self.event_targets["first_turn_dir"][idx], dtype=torch.long),
            "first_reversal_has": torch.tensor(self.event_targets["first_reversal_has"][idx], dtype=torch.float32),
            "first_reversal_bin": torch.tensor(self.event_targets["first_reversal_bin"][idx], dtype=torch.long),
            "major_peak_bin": torch.tensor(self.event_targets["major_peak_bin"][idx], dtype=torch.long),
        }
        if self.mechanism_ids is not None:
            item["mechanism_id"] = torch.tensor(self.mechanism_ids[idx], dtype=torch.long)
        return item


def _diff1(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x[:, 1:] - x[:, :-1]
    return x[:, 1:, :] - x[:, :-1, :]


def _diff2(x: torch.Tensor) -> torch.Tensor:
    return _diff1(_diff1(x))


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(mask.sum(), min=1.0)
    return (values * mask).sum() / denom


def masked_regression(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
    kind: str = "mse",
    beta: float = 0.25,
) -> torch.Tensor:
    if kind == "smoothl1":
        loss = torch.nn.functional.smooth_l1_loss(pred, true, beta=beta, reduction="none")
    else:
        loss = (pred - true) ** 2
    return masked_mean(loss, mask)


def masked_peak_range_loss(
    y_hat_denorm: torch.Tensor,
    y_true_denorm: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    steer_pred = y_hat_denorm[:, :, 0]
    steer_true = y_true_denorm[:, :, 0]
    valid = mask > 0

    def _masked_peak(x: torch.Tensor) -> torch.Tensor:
        masked = x.abs().masked_fill(~valid, 0.0)
        return masked.amax(dim=1)

    def _masked_range(x: torch.Tensor) -> torch.Tensor:
        pos = x.masked_fill(~valid, -1e9).amax(dim=1)
        neg = x.masked_fill(~valid, 1e9).amin(dim=1)
        empty = (~valid).all(dim=1)
        rng = pos - neg
        rng[empty] = 0.0
        return rng

    peak_loss = torch.nn.functional.l1_loss(_masked_peak(steer_pred), _masked_peak(steer_true))
    range_loss = torch.nn.functional.l1_loss(_masked_range(steer_pred), _masked_range(steer_true))
    return 0.7 * peak_loss + 0.3 * range_loss


def build_primary_mask(mask: torch.Tensor, horizon_len: int = PRIMARY_HORIZON_LEN) -> torch.Tensor:
    out = mask.clone()
    if out.shape[1] > horizon_len:
        out[:, horizon_len:] = 0.0
    return out


def build_point_weights(
    mask: torch.Tensor,
    scheme: str = "baseline",
    horizon_len: int = PRIMARY_HORIZON_LEN,
) -> torch.Tensor:
    weights = build_primary_mask(mask, horizon_len=horizon_len)
    if scheme == "baseline":
        return weights

    device = mask.device
    time_idx = torch.arange(mask.shape[1], device=device, dtype=torch.float32)

    if scheme == "horizon_weighted":
        seg = torch.ones_like(time_idx)
        seg[(time_idx >= 100) & (time_idx < 200)] = 1.15
        seg[(time_idx >= 200) & (time_idx < 300)] = 1.35
        return weights * seg.unsqueeze(0)

    if scheme == "critical_step_weighted":
        boost = torch.ones_like(time_idx)
        for center in (99.0, 199.0, 299.0):
            boost = boost + 0.45 * torch.exp(-0.5 * ((time_idx - center) / 10.0) ** 2)
        return weights * boost.unsqueeze(0)

    if scheme == "full2s_tail_weighted":
        seg = torch.ones_like(time_idx)
        seg[(time_idx >= 100) & (time_idx < 200)] = 1.10
        seg[(time_idx >= 200) & (time_idx < 300)] = 1.25
        seg[(time_idx >= 300) & (time_idx < 400)] = 1.45
        return weights * seg.unsqueeze(0)

    raise ValueError(f"unknown weighting scheme: {scheme}")


class UnifiedOutputHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class DecoupledOutputHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.steer_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.speed_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        steer = self.steer_head(hidden)
        speed = self.speed_head(hidden)
        return torch.cat([steer, speed], dim=-1)


class HorizonAwareOutputHead(nn.Module):
    def __init__(self, d_model: int, future_len: int, out_dim: int = 2) -> None:
        super().__init__()
        self.future_len = future_len
        self.segment_ids = torch.zeros(future_len, dtype=torch.long)
        self.segment_ids[100:200] = 1
        self.segment_ids[200:300] = 2
        self.segment_ids[300:] = 3
        self.heads = nn.ModuleList([nn.Linear(d_model, out_dim) for _ in range(4)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        seg_ids = self.segment_ids.to(hidden.device)
        out = torch.zeros(hidden.shape[0], hidden.shape[1], 2, device=hidden.device, dtype=hidden.dtype)
        for seg_idx, head in enumerate(self.heads):
            seg_mask = seg_ids == seg_idx
            if seg_mask.any():
                out[:, seg_mask, :] = head(hidden[:, seg_mask, :])
        return out


def make_output_head(head_type: str, d_model: int, future_len: int) -> nn.Module:
    if head_type == "unified":
        return UnifiedOutputHead(d_model=d_model)
    if head_type == "decoupled":
        return DecoupledOutputHead(d_model=d_model)
    if head_type == "horizon_aware":
        return HorizonAwareOutputHead(d_model=d_model, future_len=future_len)
    raise ValueError(f"unknown head type: {head_type}")


class TrajectoryEventAuxHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.turn_time_head = nn.Linear(d_model, 1)
        self.reversal_time_head = nn.Linear(d_model, 1)
        self.peak_time_head = nn.Linear(d_model, 1)
        self.turn_has_head = nn.Linear(d_model, 1)
        self.turn_dir_head = nn.Linear(d_model, 2)
        self.reversal_has_head = nn.Linear(d_model, 1)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = hidden.mean(dim=1)
        return {
            "first_turn_step_logits": self.turn_time_head(hidden).squeeze(-1),
            "first_reversal_step_logits": self.reversal_time_head(hidden).squeeze(-1),
            "major_peak_step_logits": self.peak_time_head(hidden).squeeze(-1),
            "first_turn_has_logit": self.turn_has_head(pooled).squeeze(-1),
            "first_turn_dir_logits": self.turn_dir_head(pooled),
            "first_reversal_has_logit": self.reversal_has_head(pooled).squeeze(-1),
        }


class FutureControlTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        future_len: int = FUTURE_LEN,
        out_dim: int = 2,
        d_model: int = 128,
        nhead: int = 2,
        enc_layers: int = 2,
        dec_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        head_type: str = "unified",
        critical_aux: bool = False,
        event_aux: bool = False,
    ) -> None:
        super().__init__()
        self.future_len = future_len
        self.critical_aux = critical_aux
        self.enc_input_proj = nn.Linear(input_dim, d_model)
        self.enc_pos = nn.Parameter(torch.zeros(1, WIN_LEN, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.pool_score = nn.Linear(d_model, 1)
        self.lat_proj = nn.Linear(d_model, d_model)
        self.ctx_proj = nn.Linear(context_dim, d_model)
        self.curve_proj = nn.Linear(1, d_model)
        self.dec_pos = nn.Parameter(torch.zeros(1, FUTURE_LEN, d_model))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
        self.dropout = nn.Dropout(dropout)
        self.output_head = make_output_head(head_type=head_type, d_model=d_model, future_len=future_len)
        self.aux_head = None
        self.event_aux_head = TrajectoryEventAuxHead(d_model=d_model) if event_aux else None
        if critical_aux:
            self.aux_head = nn.Linear(d_model, out_dim)

    def forward(
        self,
        src: torch.Tensor,
        ctx: torch.Tensor,
        curve_norm: torch.Tensor,
        mechanism_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del mechanism_id
        bsz, t_in, _ = src.shape
        h_src = self.enc_input_proj(src) + self.enc_pos[:, :t_in, :]
        memory = self.encoder(self.dropout(h_src))
        scores = self.pool_score(memory)
        alpha = torch.softmax(scores, dim=1)
        pooled = torch.sum(alpha * memory, dim=1)
        lat_emb = self.lat_proj(pooled).unsqueeze(1).expand(bsz, self.future_len, -1)
        ctx_emb = self.ctx_proj(ctx).unsqueeze(1).expand(bsz, self.future_len, -1)
        curve_emb = self.curve_proj(curve_norm.unsqueeze(-1))
        tgt = self.dec_pos[:, : self.future_len, :] + lat_emb + ctx_emb + curve_emb
        hidden = self.decoder(tgt, memory)
        out = self.output_head(hidden)
        extras: dict[str, torch.Tensor] = {"hidden": hidden}
        if self.event_aux_head is not None:
            extras["event_aux"] = self.event_aux_head(hidden)
        if self.aux_head is not None:
            critical_idx = torch.tensor([99, 199, 299], device=hidden.device)
            critical_idx = critical_idx[critical_idx < hidden.shape[1]]
            extras["critical_aux"] = self.aux_head(hidden.index_select(dim=1, index=critical_idx))
            extras["critical_idx"] = critical_idx
        return out, extras


class FutureControlGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        future_len: int = FUTURE_LEN,
        d_model: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        head_type: str = "unified",
        critical_aux: bool = False,
        event_aux: bool = False,
    ) -> None:
        super().__init__()
        self.future_len = future_len
        self.critical_aux = critical_aux
        self.input_proj = nn.Linear(input_dim, d_model)
        self.encoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.ctx_proj = nn.Linear(context_dim, d_model)
        self.curve_proj = nn.Linear(1, d_model)
        self.dec_pos = nn.Parameter(torch.zeros(1, FUTURE_LEN, d_model))
        self.decoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.lat_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_head = make_output_head(head_type=head_type, d_model=d_model, future_len=future_len)
        self.aux_head = nn.Linear(d_model, 2) if critical_aux else None
        self.event_aux_head = TrajectoryEventAuxHead(d_model=d_model) if event_aux else None

    def forward(
        self,
        src: torch.Tensor,
        ctx: torch.Tensor,
        curve_norm: torch.Tensor,
        mechanism_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del mechanism_id
        enc_in = self.input_proj(src)
        memory, h_n = self.encoder(enc_in)
        pooled = memory.mean(dim=1)
        lat = self.lat_proj(pooled).unsqueeze(1).expand(src.shape[0], self.future_len, -1)
        ctx_emb = self.ctx_proj(ctx).unsqueeze(1).expand(src.shape[0], self.future_len, -1)
        curve_emb = self.curve_proj(curve_norm.unsqueeze(-1))
        dec_in = self.dec_pos[:, : self.future_len, :] + lat + ctx_emb + curve_emb
        hidden, _ = self.decoder(self.dropout(dec_in), h_n)
        out = self.output_head(hidden)
        extras: dict[str, torch.Tensor] = {"hidden": hidden}
        if self.event_aux_head is not None:
            extras["event_aux"] = self.event_aux_head(hidden)
        if self.aux_head is not None:
            critical_idx = torch.tensor([99, 199, 299], device=hidden.device)
            critical_idx = critical_idx[critical_idx < hidden.shape[1]]
            extras["critical_aux"] = self.aux_head(hidden.index_select(dim=1, index=critical_idx))
            extras["critical_idx"] = critical_idx
        return out, extras


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        # Two same-padded causal-style convolutions expand the time axis twice.
        self.chomp = 2 * padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.chomp > 0:
            out = out[:, :, :-self.chomp]
        residual = self.down(x[:, :, : out.shape[2]])
        return out + residual


class FutureControlTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        future_len: int = FUTURE_LEN,
        d_model: int = 128,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        head_type: str = "unified",
        critical_aux: bool = False,
        event_aux: bool = False,
    ) -> None:
        super().__init__()
        self.future_len = future_len
        self.critical_aux = critical_aux
        self.input_proj = nn.Linear(input_dim, d_model)
        enc_blocks = []
        for level in range(levels):
            in_ch = d_model if level > 0 else d_model
            enc_blocks.append(TemporalBlock(in_ch, d_model, kernel_size=kernel_size, dilation=2**level, dropout=dropout))
        self.encoder = nn.ModuleList(enc_blocks)
        self.ctx_proj = nn.Linear(context_dim, d_model)
        self.curve_proj = nn.Linear(1, d_model)
        self.lat_proj = nn.Linear(d_model, d_model)
        self.dec_pos = nn.Parameter(torch.zeros(1, FUTURE_LEN, d_model))
        dec_blocks = []
        for level in range(levels):
            dec_blocks.append(TemporalBlock(d_model, d_model, kernel_size=kernel_size, dilation=2**level, dropout=dropout))
        self.decoder = nn.ModuleList(dec_blocks)
        self.output_head = make_output_head(head_type=head_type, d_model=d_model, future_len=future_len)
        self.aux_head = nn.Linear(d_model, 2) if critical_aux else None
        self.event_aux_head = TrajectoryEventAuxHead(d_model=d_model) if event_aux else None

    def forward(
        self,
        src: torch.Tensor,
        ctx: torch.Tensor,
        curve_norm: torch.Tensor,
        mechanism_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del mechanism_id
        x = self.input_proj(src).transpose(1, 2)
        for block in self.encoder:
            x = block(x)
        memory = x.transpose(1, 2)
        pooled = memory.mean(dim=1)
        lat = self.lat_proj(pooled).unsqueeze(1).expand(src.shape[0], self.future_len, -1)
        ctx_emb = self.ctx_proj(ctx).unsqueeze(1).expand(src.shape[0], self.future_len, -1)
        curve_emb = self.curve_proj(curve_norm.unsqueeze(-1))
        hidden = (self.dec_pos[:, : self.future_len, :] + lat + ctx_emb + curve_emb).transpose(1, 2)
        for block in self.decoder:
            hidden = block(hidden)
        hidden = hidden.transpose(1, 2)
        out = self.output_head(hidden)
        extras: dict[str, torch.Tensor] = {"hidden": hidden}
        if self.event_aux_head is not None:
            extras["event_aux"] = self.event_aux_head(hidden)
        if self.aux_head is not None:
            critical_idx = torch.tensor([99, 199, 299], device=hidden.device)
            critical_idx = critical_idx[critical_idx < hidden.shape[1]]
            extras["critical_aux"] = self.aux_head(hidden.index_select(dim=1, index=critical_idx))
            extras["critical_idx"] = critical_idx
        return out, extras


def build_model(
    backbone: str,
    input_dim: int,
    context_dim: int,
    config: dict[str, Any],
) -> nn.Module:
    hidden_size = int(config.get("d_model", config.get("hidden_size", 128)))
    common = {
        "input_dim": input_dim,
        "context_dim": context_dim,
        "future_len": FUTURE_LEN,
        "d_model": hidden_size,
        "dropout": float(config.get("dropout", 0.1)),
        "head_type": str(config.get("head_type", "unified")),
        "critical_aux": bool(config.get("critical_aux", False)),
        "event_aux": bool(config.get("event_aux", False)),
    }
    if backbone == "transformer":
        return FutureControlTransformer(
            **common,
            nhead=int(config.get("nhead", 2)),
            enc_layers=int(config.get("enc_layers", 2)),
            dec_layers=int(config.get("dec_layers", 2)),
            ffn_dim=int(config.get("ffn_dim", hidden_size * 2)),
        )
    if backbone == "gru":
        return FutureControlGRU(
            **common,
            num_layers=int(config.get("num_layers", 2)),
        )
    if backbone == "tcn":
        return FutureControlTCN(
            **common,
            levels=int(config.get("levels", 4)),
            kernel_size=int(config.get("kernel_size", 3)),
        )
    raise ValueError(f"unknown backbone: {backbone}")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
