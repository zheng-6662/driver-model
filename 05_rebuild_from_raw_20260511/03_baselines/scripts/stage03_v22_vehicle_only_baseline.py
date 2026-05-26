# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
REBUILD_ROOT = Path(os.environ.get("REBUILD_ROOT", str(PROJECT_ROOT / "05_rebuild_from_raw_20260511")))
V22_DIR = REBUILD_ROOT / "02_samples" / "record_level_episode_reconstruction_v2_2_epoch_refined"
V22_TRAINING_POOL = V22_DIR / "tables" / "training_pool_epoch_refined_v2_2.csv"

OUT_ROOT = REBUILD_ROOT / "03_baselines" / "stage03_v22_vehicle_only_baseline"
DATASET_DIR = REBUILD_ROOT / "03_processed_datasets" / "record_episode_v22_vehicle_only_baseline"
ARRAY_PATH = DATASET_DIR / "arrays" / "v22_vehicle_only_pre3_post2_20hz.npz"
META_PATH = DATASET_DIR / "tables" / "v22_vehicle_only_pre3_post2_20hz_meta.csv"
DATASET_SUMMARY_PATH = DATASET_DIR / "logs" / "v22_vehicle_only_pre3_post2_20hz_summary.json"

TEST_SUBJECTS = {"cwh", "gf", "tyy"}
VAL_SUBJECTS = {"byx", "gzj", "yyl"}

HZ = 20.0
INPUT_TIME = np.round(np.arange(-3.0, 0.0 + 1e-9, 1.0 / HZ), 6)
TARGET_TIME = np.round(np.arange(0.0, 2.0 + 1e-9, 1.0 / HZ), 6)

INPUT_FEATURES = [
    "zx|SteeringWheel",
    "steer_rate",
    "zx1|v_km/h",
    "zx|BrakePedal",
    "zx|AcceleratorPedal",
    "zx|ax",
    "zx|ay",
    "zx|vyaw",
    "zx|vroll",
    "zx|roll",
    "lateral_distance_selected",
    "zx1|mu",
    "curvature_selected",
]
ESSENTIAL_INPUT_FEATURES = ["zx|SteeringWheel", "zx1|v_km/h"]

OUTPUT_SPECS = [
    ("steering_delta", "zx|SteeringWheel", True),
    ("yaw_rate", "zx|vyaw", False),
    ("ay", "zx|ay", False),
]
OUTPUT_NAMES = [x[0] for x in OUTPUT_SPECS]
OUTPUT_LOSS_WEIGHTS = np.array([1.5, 1.0, 0.7], dtype=np.float32)

RAW_USECOLS = sorted(
    {
        "StorageTime",
        "zx|SteeringWheel",
        "zx1|v_km/h",
        "zx|BrakePedal",
        "zx|AcceleratorPedal",
        "zx|ax",
        "zx|ay",
        "zx|vyaw",
        "zx|vroll",
        "zx|roll",
        "zx1|mu",
        "zx1|lateraldistance",
        "zx|lateraldistance",
        "zx1|lanecurvatureXY",
        "zx|lanecurvatureXY",
    }
)

CORE_ROLE = "main_train_candidate_v2_1"
REVIEW_ROLE = "review_recovered_candidate_v2_1"
CONTROL_ROLE = "control_or_weak_candidate_v2_1"


def ensure_dirs() -> None:
    for path in [
        OUT_ROOT,
        DATASET_DIR / "arrays",
        DATASET_DIR / "tables",
        DATASET_DIR / "logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def parse_time_seconds(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.8:
        arr = numeric.to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.full(len(series), np.nan)
        return arr - finite[0]
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() == 0:
        return np.full(len(series), np.nan)
    base = parsed.dropna().iloc[0]
    return (parsed - base).dt.total_seconds().to_numpy(dtype=float)


def gradient(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & np.isfinite(t)
    if valid.sum() < 3:
        return out
    order = np.argsort(t[valid])
    tv = t[valid][order]
    vv = values[valid][order]
    keep = np.r_[True, np.diff(tv) > 1e-6]
    tv = tv[keep]
    vv = vv[keep]
    if tv.size < 3:
        return np.zeros_like(values, dtype=float)
    deriv = np.gradient(vv, tv)
    out = np.interp(t, tv, deriv, left=deriv[0], right=deriv[-1])
    out[~np.isfinite(out)] = 0.0
    return out


def resolve_project_root_candidates() -> list[Path]:
    candidates = [
        PROJECT_ROOT,
        Path.cwd(),
        Path("/root/autodl-tmp/data_process"),
        Path("/root/data_process"),
        Path("/workspace/data_process"),
    ]
    seen: set[str] = set()
    out: list[Path] = []
    for p in candidates:
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def remap_vehicle_path(path_text: str) -> Path:
    raw = str(path_text).strip()
    if not raw or raw.lower() == "nan":
        return Path(raw)
    direct = Path(raw)
    if direct.exists():
        return direct
    norm = raw.replace("\\", "/")
    marker = "/01_datasets/"
    if marker in norm:
        suffix = norm.split(marker, 1)[1]
        for root in resolve_project_root_candidates():
            candidate = root / "01_datasets" / Path(*suffix.split("/"))
            if candidate.exists():
                return candidate
    parts = [p for p in norm.split("/") if p]
    if "原始车辆数据" in parts:
        i = parts.index("原始车辆数据")
        subject = parts[i + 1] if i + 1 < len(parts) else ""
        file_name = parts[-1] if parts else ""
        stem = Path(file_name).stem
        cleaned_name = f"{stem}_aligned_cleaned.csv"
        for root in resolve_project_root_candidates():
            candidate = root / "01_datasets" / "多模态数据" / "被试数据集合" / subject / "vehicle" / cleaned_name
            if candidate.exists():
                return candidate
    file_name = parts[-1] if parts else ""
    subject_hint = ""
    if "原始车辆数据" in parts:
        j = parts.index("原始车辆数据")
        subject_hint = parts[j + 1] if j + 1 < len(parts) else ""
    stem = Path(file_name).stem
    cleaned_name = f"{stem}_aligned_cleaned.csv"
    for root in resolve_project_root_candidates():
        base = root / "01_datasets" / "多模态数据" / "被试数据集合"
        if subject_hint:
            candidate = base / subject_hint / "vehicle" / cleaned_name
            if candidate.exists():
                return candidate
        if base.exists() and file_name:
            matches = list(base.glob(f"*/vehicle/{cleaned_name}"))
            if matches:
                return matches[0]
    return direct


def load_vehicle_csv_light(path: Path) -> pd.DataFrame | None:
    try:
        header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
        usecols = [c for c in RAW_USECOLS if c in header.columns]
        if "StorageTime" not in usecols:
            return None
        df = pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    except Exception:
        return None
    df["time_rel_s"] = parse_time_seconds(df["StorageTime"])
    df = df[np.isfinite(df["time_rel_s"])].copy()
    df = df.drop_duplicates("time_rel_s").sort_values("time_rel_s")
    if len(df) < 20:
        return None
    for col in df.columns:
        if col not in {"StorageTime", "time_rel_s"}:
            df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(limit_direction="both")
    t = df["time_rel_s"].to_numpy(dtype=float)
    if "zx|SteeringWheel" in df.columns:
        df["steer_rate"] = gradient(df["zx|SteeringWheel"].to_numpy(dtype=float), t)
    else:
        df["steer_rate"] = np.nan
    lat_col = "zx1|lateraldistance" if "zx1|lateraldistance" in df.columns else "zx|lateraldistance"
    df["lateral_distance_selected"] = df[lat_col].to_numpy(dtype=float) if lat_col in df.columns else np.nan
    curv_col = "zx1|lanecurvatureXY" if "zx1|lanecurvatureXY" in df.columns else "zx|lanecurvatureXY"
    df["curvature_selected"] = df[curv_col].to_numpy(dtype=float) if curv_col in df.columns else np.nan
    return df.reset_index(drop=True)


def interp_series(df: pd.DataFrame, col: str, query_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "time_rel_s" not in df.columns or col not in df.columns:
        return np.zeros_like(query_time, dtype=np.float32), np.zeros_like(query_time, dtype=bool)
    t = df["time_rel_s"].to_numpy(dtype=float)
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(v)
    if valid.sum() < 2:
        return np.zeros_like(query_time, dtype=np.float32), np.zeros_like(query_time, dtype=bool)
    tt = t[valid]
    vv = v[valid]
    order = np.argsort(tt)
    tt = tt[order]
    vv = vv[order]
    unique_t, unique_idx = np.unique(tt, return_index=True)
    unique_v = vv[unique_idx]
    inside = (query_time >= unique_t[0]) & (query_time <= unique_t[-1])
    out = np.zeros_like(query_time, dtype=np.float32)
    out[inside] = np.interp(query_time[inside], unique_t, unique_v).astype(np.float32)
    return out, inside.astype(bool)


def valid_seconds(mask: np.ndarray, time_axis: np.ndarray) -> float:
    if len(mask) == 0 or not np.any(mask):
        return 0.0
    return float(mask.sum() / max(1.0, len(mask)) * (time_axis[-1] - time_axis[0]))


def build_arrays(force_rebuild: bool = False) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    ensure_dirs()
    required = {"X", "X_mask", "Y", "Y_mask", "input_time", "target_time", "input_feature_names", "output_names"}
    if not force_rebuild and ARRAY_PATH.exists() and META_PATH.exists() and DATASET_SUMMARY_PATH.exists():
        with np.load(ARRAY_PATH, allow_pickle=True) as z:
            if required.issubset(set(z.files)):
                arrays = {name: z[name] for name in z.files}
                meta = pd.read_csv(META_PATH, encoding="utf-8-sig", low_memory=False)
                summary = json.loads(DATASET_SUMMARY_PATH.read_text(encoding="utf-8"))
                return arrays, meta, summary

    manifest = pd.read_csv(V22_TRAINING_POOL, encoding="utf-8-sig", low_memory=False)
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    xs: list[np.ndarray] = []
    xmasks: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ymasks: list[np.ndarray] = []
    cache: dict[str, pd.DataFrame | None] = {}

    sorted_manifest = manifest.sort_values(["subject", "vehicle_file", "v2_2_model_anchor_s"]).reset_index(drop=True)
    for row_no, ep in sorted_manifest.iterrows():
        if row_no % 100 == 0:
            print(f"build arrays {row_no}/{len(sorted_manifest)}", flush=True)
        path = remap_vehicle_path(str(ep.get("vehicle_file", "")))
        key = str(path)
        if key not in cache:
            cache[key] = load_vehicle_csv_light(path)
        df = cache[key]
        sample_id = str(ep.get("episode_uid", f"row_{row_no}"))
        if df is None:
            dropped.append({"sample_id": sample_id, "drop_reason": "vehicle_csv_unreadable", "vehicle_file": str(path)})
            continue
        anchor = finite_float(ep.get("v2_2_model_anchor_s"))
        if not math.isfinite(anchor):
            dropped.append({"sample_id": sample_id, "drop_reason": "anchor_missing", "vehicle_file": str(path)})
            continue

        input_query = anchor + INPUT_TIME
        target_query = anchor + TARGET_TIME

        input_values: list[np.ndarray] = []
        input_masks: list[np.ndarray] = []
        for col in INPUT_FEATURES:
            vals, m = interp_series(df, col, input_query)
            if col == "zx|SteeringWheel":
                anchor_vals, anchor_m = interp_series(df, col, np.array([anchor], dtype=float))
                if anchor_m[0]:
                    vals = vals - float(anchor_vals[0])
            input_values.append(vals)
            input_masks.append(m)
        x = np.stack(input_values, axis=1).astype(np.float32)
        x_mask = np.stack(input_masks, axis=1).astype(bool)

        target_values: list[np.ndarray] = []
        target_masks: list[np.ndarray] = []
        baseline: dict[str, float] = {}
        for out_name, col, relative in OUTPUT_SPECS:
            anchor_vals, anchor_m = interp_series(df, col, np.array([anchor], dtype=float))
            baseline[out_name] = float(anchor_vals[0]) if anchor_m[0] else float("nan")
            vals, m = interp_series(df, col, target_query)
            if relative and anchor_m[0]:
                vals = vals - float(anchor_vals[0])
            target_values.append(vals)
            target_masks.append(m)
        y = np.stack(target_values, axis=1).astype(np.float32)
        y_mask = np.stack(target_masks, axis=1).astype(bool)

        essential_idx = [INPUT_FEATURES.index(c) for c in ESSENTIAL_INPUT_FEATURES if c in INPUT_FEATURES]
        input_time_mask = x_mask[:, essential_idx].mean(axis=1) >= 0.5 if essential_idx else x_mask.any(axis=1)
        target_time_mask = y_mask[:, 0]
        input_valid_sec = valid_seconds(input_time_mask, INPUT_TIME)
        target_valid_sec = valid_seconds(target_time_mask, TARGET_TIME)
        if input_valid_sec < 2.8 or target_valid_sec < 1.8:
            dropped.append(
                {
                    "sample_id": sample_id,
                    "drop_reason": "window_incomplete",
                    "input_valid_sec": input_valid_sec,
                    "target_valid_sec": target_valid_sec,
                    "vehicle_file": str(path),
                }
            )
            continue

        split = str(ep.get("split", "")).strip()
        if split not in {"train", "val", "test"}:
            subject = str(ep.get("subject", "")).strip()
            split = "test" if subject in TEST_SUBJECTS else ("val" if subject in VAL_SUBJECTS else "train")

        row = ep.to_dict()
        row.update(
            {
                "sample_id": sample_id,
                "anchor_time_s": anchor,
                "split": split,
                "input_valid_sec": input_valid_sec,
                "target_valid_sec": target_valid_sec,
                "vehicle_file_resolved": str(path),
                "steering_anchor_value": baseline["steering_delta"],
                "yaw_anchor_value": baseline["yaw_rate"],
                "ay_anchor_value": baseline["ay"],
            }
        )
        rows.append(row)
        xs.append(x)
        xmasks.append(x_mask)
        ys.append(y)
        ymasks.append(y_mask)

    if not rows:
        raise RuntimeError("No usable v2.2 samples were built.")

    meta = pd.DataFrame(rows)
    arrays = {
        "X": np.stack(xs, axis=0).astype(np.float32),
        "X_mask": np.stack(xmasks, axis=0).astype(bool),
        "Y": np.stack(ys, axis=0).astype(np.float32),
        "Y_mask": np.stack(ymasks, axis=0).astype(bool),
        "input_time": INPUT_TIME.astype(np.float32),
        "target_time": TARGET_TIME.astype(np.float32),
        "input_feature_names": np.array(INPUT_FEATURES, dtype=object),
        "output_names": np.array(OUTPUT_NAMES, dtype=object),
    }
    np.savez_compressed(ARRAY_PATH, **arrays)
    meta.to_csv(META_PATH, index=False, encoding="utf-8-sig")
    if dropped:
        pd.DataFrame(dropped).to_csv(DATASET_DIR / "tables" / "v22_vehicle_only_dropped.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source_manifest": str(V22_TRAINING_POOL),
        "sample_count": int(len(meta)),
        "dropped_count": int(len(dropped)),
        "split_counts": meta["split"].value_counts().to_dict(),
        "role_counts": meta["v2_1_role"].value_counts().to_dict(),
        "input_window_sec": [-3.0, 0.0],
        "target_window_sec": [0.0, 2.0],
        "sampling_hz": HZ,
        "input_features": INPUT_FEATURES,
        "output_names": OUTPUT_NAMES,
        "no_future_input_rule": "only anchor-pre/current time-series values are used as model input; no episode peak fields are included",
    }
    DATASET_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return arrays, meta, summary


def include_mask_for_dataset(meta: pd.DataFrame, dataset: str) -> np.ndarray:
    role = meta["v2_1_role"].astype(str)
    if dataset == "core":
        return role.eq(CORE_ROLE).to_numpy()
    if dataset == "core_review":
        return role.isin([CORE_ROLE, REVIEW_ROLE]).to_numpy()
    if dataset == "core_review_control":
        return role.isin([CORE_ROLE, REVIEW_ROLE, CONTROL_ROLE]).to_numpy()
    raise ValueError(f"Unknown dataset: {dataset}")


def split_indices(meta: pd.DataFrame, include: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = meta["split"].astype(str).to_numpy()
    idx = np.arange(len(meta))
    train = idx[include & (split == "train")]
    val = idx[include & (split == "val")]
    test = idx[include & (split == "test")]
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError(f"Bad split sizes: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def scale_inputs(x: np.ndarray, x_mask: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_filled = np.where(x_mask, x, np.nan).astype(np.float32)
    mean = np.nanmean(x_filled[train_idx], axis=(0, 1))
    std = np.nanstd(x_filled[train_idx], axis=(0, 1))
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where((std >= 1e-6) & np.isfinite(std), std, 1.0).astype(np.float32)
    scaled = (np.where(x_mask, x, mean[None, None, :]) - mean[None, None, :]) / std[None, None, :]
    scaled = np.where(x_mask, scaled, 0.0).astype(np.float32)
    return scaled, mean, std


def scale_targets(y: np.ndarray, y_mask: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.zeros(y.shape[-1], dtype=np.float32)
    std = np.ones(y.shape[-1], dtype=np.float32)
    scaled = np.zeros_like(y, dtype=np.float32)
    for j in range(y.shape[-1]):
        vals = y[train_idx, :, j][y_mask[train_idx, :, j]]
        if len(vals):
            mean[j] = float(np.nanmean(vals))
            s = float(np.nanstd(vals))
            std[j] = s if s >= 1e-6 and math.isfinite(s) else 1.0
        scaled[:, :, j] = (y[:, :, j] - mean[j]) / std[j]
    return np.where(y_mask, scaled, 0.0).astype(np.float32), mean, std


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, x_mask: np.ndarray, y: np.ndarray, y_mask: np.ndarray, weights: np.ndarray, indices: np.ndarray) -> None:
        self.x = x[indices]
        self.x_mask = x_mask[indices].astype(np.float32)
        self.y = y[indices]
        self.y_mask = y_mask[indices].astype(np.float32)
        self.weights = weights[indices].astype(np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = np.concatenate([self.x[idx], self.x_mask[idx]], axis=1).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(self.y[idx]),
            torch.from_numpy(self.y_mask[idx]),
            torch.tensor(self.weights[idx], dtype=torch.float32),
        )


class GRURegressor(nn.Module):
    def __init__(self, input_dim: int, horizon: int, out_dim: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, horizon * out_dim))
        self.horizon = horizon
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        y = self.head(out[:, -1])
        return y.view(-1, self.horizon, self.out_dim)


class TCNRegressor(nn.Module):
    def __init__(self, input_dim: int, horizon: int, out_dim: int, channels: tuple[int, ...] = (64, 128, 128), dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_dim
        for i, ch in enumerate(channels):
            dilation = 2**i
            padding = dilation
            layers.extend(
                [
                    nn.Conv1d(in_ch, ch, kernel_size=3, padding=padding, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(ch, ch, kernel_size=3, padding=padding, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = ch
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(in_ch)
        self.head = nn.Linear(in_ch, horizon * out_dim)
        self.horizon = horizon
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.transpose(1, 2))
        z = z[:, :, -x.shape[1] :]
        pooled = z.mean(dim=2)
        y = self.head(self.norm(pooled))
        return y.view(-1, self.horizon, self.out_dim)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        out_dim: int,
        d_model: int = 128,
        layers: int = 2,
        heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, len(INPUT_TIME), d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, horizon * out_dim))
        self.horizon = horizon
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x) + self.pos[:, : x.shape[1]]
        z = self.encoder(z)
        y = self.head(z[:, -1])
        return y.view(-1, self.horizon, self.out_dim)


def make_model(model_name: str, input_dim: int, horizon: int, out_dim: int) -> nn.Module:
    if model_name == "gru":
        return GRURegressor(input_dim, horizon, out_dim)
    if model_name == "tcn":
        return TCNRegressor(input_dim, horizon, out_dim)
    if model_name == "transformer":
        return TransformerRegressor(input_dim, horizon, out_dim)
    raise ValueError(f"Unknown model: {model_name}")


def masked_huber_loss(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    diff = torch.nn.functional.smooth_l1_loss(pred, y, reduction="none", beta=1.0)
    output_weight = torch.tensor(OUTPUT_LOSS_WEIGHTS, dtype=pred.dtype, device=pred.device).view(1, 1, -1)
    valid = mask > 0.5
    w = output_weight * sample_weight.view(-1, 1, 1)
    denom = torch.clamp((valid.float() * w).sum(), min=1.0)
    return (diff * valid.float() * w).sum() / denom


@torch.no_grad()
def predict_model(model: nn.Module, x: np.ndarray, x_mask: np.ndarray, indices: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        xb = np.concatenate([x[idx], x_mask[idx].astype(np.float32)], axis=2).astype(np.float32)
        yb = model(torch.from_numpy(xb).to(device)).cpu().numpy()
        preds.append(yb)
    return np.concatenate(preds, axis=0) if preds else np.empty((0, len(TARGET_TIME), len(OUTPUT_NAMES)), dtype=np.float32)


def train_model(args: argparse.Namespace, arrays: dict[str, np.ndarray], meta: pd.DataFrame, run_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    include = include_mask_for_dataset(meta, args.dataset)
    train_idx, val_idx, _ = split_indices(meta, include)
    x_scaled, x_mean, x_std = scale_inputs(arrays["X"], arrays["X_mask"], train_idx)
    y_scaled, y_mean, y_std = scale_targets(arrays["Y"], arrays["Y_mask"], train_idx)

    sample_weight = np.ones(len(meta), dtype=np.float32)
    if args.dataset != "core":
        sample_weight[meta["v2_1_role"].astype(str).eq(REVIEW_ROLE).to_numpy()] = args.review_weight
        sample_weight[meta["v2_1_role"].astype(str).eq(CONTROL_ROLE).to_numpy()] = args.control_weight

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    input_dim = arrays["X"].shape[-1] * 2
    model = make_model(args.model, input_dim, len(TARGET_TIME), len(OUTPUT_NAMES)).to(device)
    lr = args.lr
    if lr <= 0:
        lr = 5e-4 if args.model == "transformer" else 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    train_ds = SequenceDataset(x_scaled, arrays["X_mask"], y_scaled, arrays["Y_mask"], sample_weight, train_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)

    xv = np.concatenate([x_scaled[val_idx], arrays["X_mask"][val_idx].astype(np.float32)], axis=2).astype(np.float32)
    yv = torch.from_numpy(y_scaled[val_idx]).float().to(device)
    mv = torch.from_numpy(arrays["Y_mask"][val_idx].astype(np.float32)).float().to(device)
    wv = torch.from_numpy(sample_weight[val_idx]).float().to(device)
    xv_t = torch.from_numpy(xv).float().to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb, mb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            wb = wb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = masked_huber_loss(pred, yb, mb, wb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model(xv_t)
            val_loss = float(masked_huber_loss(val_pred, yv, mv, wv).detach().cpu())
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            torch.save(
                {
                    "model_state": best_state,
                    "model": args.model,
                    "dataset": args.dataset,
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "input_features": INPUT_FEATURES,
                    "output_names": OUTPUT_NAMES,
                    "input_time": INPUT_TIME,
                    "target_time": TARGET_TIME,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                },
                run_dir / "best_model.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    pd.DataFrame(history).to_csv(run_dir / "loss_history.csv", index=False, encoding="utf-8-sig")

    all_idx = np.arange(len(meta))
    pred_scaled = predict_model(model, x_scaled, arrays["X_mask"], all_idx, device)
    pred = pred_scaled * y_std[None, None, :] + y_mean[None, None, :]
    train_info = {"best_val_loss": best_val, "best_epoch": best_epoch, "epochs_ran": len(history), "lr": lr}
    return pred.astype(np.float32), train_info


def persistence_prediction(arrays: dict[str, np.ndarray]) -> np.ndarray:
    x = arrays["X"]
    y = np.zeros((len(x), len(TARGET_TIME), len(OUTPUT_NAMES)), dtype=np.float32)
    feature_map = {name: i for i, name in enumerate(INPUT_FEATURES)}
    y[:, :, 0] = 0.0
    if "zx|vyaw" in feature_map:
        y[:, :, 1] = x[:, -1, feature_map["zx|vyaw"]][:, None]
    if "zx|ay" in feature_map:
        y[:, :, 2] = x[:, -1, feature_map["zx|ay"]][:, None]
    return y


def masked_rmse(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool) & np.isfinite(y) & np.isfinite(pred)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean((y[valid] - pred[valid]) ** 2)))


def signed_peak(curve: np.ndarray, mask: np.ndarray, times: np.ndarray) -> tuple[float, float]:
    valid = mask.astype(bool) & np.isfinite(curve)
    if not valid.any():
        return float("nan"), float("nan")
    vals = curve[valid]
    tt = times[valid]
    i = int(np.nanargmax(np.abs(vals)))
    return float(vals[i]), float(tt[i])


def per_sample_metrics(y: np.ndarray, pred: np.ndarray, mask: np.ndarray, meta: pd.DataFrame, include: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tail_mask = TARGET_TIME >= 1.0
    for i in range(len(meta)):
        steering_mask = mask[i, :, 0]
        true_peak, true_peak_t = signed_peak(y[i, :, 0], steering_mask, TARGET_TIME)
        pred_peak, pred_peak_t = signed_peak(pred[i, :, 0], steering_mask, TARGET_TIME)
        true_abs = abs(true_peak) if math.isfinite(true_peak) else float("nan")
        pred_abs = abs(pred_peak) if math.isfinite(pred_peak) else float("nan")
        wrong_side = bool(math.isfinite(true_peak) and math.isfinite(pred_peak) and true_abs >= 0.2 and np.sign(true_peak) != np.sign(pred_peak))
        severe_under = bool(math.isfinite(true_abs) and true_abs >= 0.5 and math.isfinite(pred_abs) and pred_abs < 0.5 * true_abs)
        large_true = bool(math.isfinite(true_abs) and true_abs >= 0.5)
        large_pred = bool(math.isfinite(pred_abs) and pred_abs >= 0.5)
        rows.append(
            {
                "sample_id": meta.iloc[i].get("sample_id", meta.iloc[i].get("episode_uid", i)),
                "split": meta.iloc[i].get("split"),
                "subject": meta.iloc[i].get("subject"),
                "v2_1_role": meta.iloc[i].get("v2_1_role"),
                "response_type": meta.iloc[i].get("response_type"),
                "episode_type": meta.iloc[i].get("episode_type"),
                "curve_type": meta.iloc[i].get("curve_type"),
                "included_for_training_protocol": bool(include[i]),
                "steering_rmse": masked_rmse(y[i, :, 0], pred[i, :, 0], steering_mask),
                "tail_steering_rmse": masked_rmse(y[i, tail_mask, 0], pred[i, tail_mask, 0], steering_mask[tail_mask]),
                "yaw_rmse": masked_rmse(y[i, :, 1], pred[i, :, 1], mask[i, :, 1]),
                "ay_rmse": masked_rmse(y[i, :, 2], pred[i, :, 2], mask[i, :, 2]),
                "true_steering_peak": true_peak,
                "pred_steering_peak": pred_peak,
                "true_steering_peak_t": true_peak_t,
                "pred_steering_peak_t": pred_peak_t,
                "peak_time_abs_error": abs(pred_peak_t - true_peak_t) if math.isfinite(pred_peak_t) and math.isfinite(true_peak_t) else float("nan"),
                "amplitude_ratio_pred_over_true": pred_abs / true_abs if math.isfinite(true_abs) and true_abs > 1e-6 and math.isfinite(pred_abs) else float("nan"),
                "wrong_side": wrong_side,
                "severe_under_amplitude": severe_under,
                "large_response_true": large_true,
                "large_response_pred": large_pred,
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(per_sample: pd.DataFrame, subset_name: str, subset_mask: np.ndarray) -> dict[str, Any]:
    df = per_sample[subset_mask].copy()
    out: dict[str, Any] = {"subset": subset_name, "n": int(len(df))}
    if len(df) == 0:
        return out
    for col in ["steering_rmse", "tail_steering_rmse", "yaw_rmse", "ay_rmse", "peak_time_abs_error"]:
        out[col] = float(pd.to_numeric(df[col], errors="coerce").mean())
    out["wrong_side_rate"] = float(df["wrong_side"].mean())
    out["severe_under_amplitude_rate"] = float(df["severe_under_amplitude"].mean())
    large = df[df["large_response_true"].astype(bool)]
    out["large_response_count"] = int(len(large))
    out["large_response_recall"] = float(large["large_response_pred"].mean()) if len(large) else float("nan")
    return out


def compute_all_metrics(y: np.ndarray, pred: np.ndarray, mask: np.ndarray, meta: pd.DataFrame, include: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    per = per_sample_metrics(y, pred, mask, meta, include)
    split = meta["split"].astype(str).to_numpy()
    rows = [
        aggregate_metrics(per, "train_included", include & (split == "train")),
        aggregate_metrics(per, "val_included", include & (split == "val")),
        aggregate_metrics(per, "test_included", include & (split == "test")),
    ]
    for role in [CORE_ROLE, REVIEW_ROLE, CONTROL_ROLE]:
        rows.append(aggregate_metrics(per, f"test_role_{role}", include & (split == "test") & meta["v2_1_role"].astype(str).eq(role).to_numpy()))
    return per, pd.DataFrame(rows)


def select_figures(per: pd.DataFrame, max_each: int = 30) -> dict[str, list[str]]:
    test = per[per["split"].astype(str).eq("test")].copy()
    groups: dict[str, list[str]] = {}
    if len(test) == 0:
        return groups
    groups["random_30"] = test.sample(min(max_each, len(test)), random_state=2026)["sample_id"].astype(str).tolist()
    groups["worst_steering_rmse_30"] = (
        test.sort_values("steering_rmse", ascending=False).head(max_each)["sample_id"].astype(str).tolist()
    )
    groups["large_response_top_30"] = (
        test.sort_values("true_steering_peak", key=lambda s: s.abs(), ascending=False).head(max_each)["sample_id"].astype(str).tolist()
    )
    groups["wrong_side_cases"] = test[test["wrong_side"].astype(bool)].head(max_each)["sample_id"].astype(str).tolist()
    groups["severe_under_amplitude_cases"] = (
        test[test["severe_under_amplitude"].astype(bool)].head(max_each)["sample_id"].astype(str).tolist()
    )
    return groups


def plot_sample(
    i: int,
    arrays: dict[str, np.ndarray],
    pred: np.ndarray,
    meta: pd.DataFrame,
    out_path: Path,
    title_prefix: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = arrays["X"][i]
    y = arrays["Y"][i]
    ymask = arrays["Y_mask"][i]
    fmap = {name: j for j, name in enumerate(INPUT_FEATURES)}
    panels = [
        ("steering_delta", "zx|SteeringWheel", 0),
        ("yaw_rate", "zx|vyaw", 1),
        ("ay", "zx|ay", 2),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 7), sharex=True)
    sid = str(meta.iloc[i].get("sample_id", meta.iloc[i].get("episode_uid", i)))
    title = (
        f"{title_prefix} | {sid} | {meta.iloc[i].get('split')} | {meta.iloc[i].get('subject')} | "
        f"{meta.iloc[i].get('v2_1_role')} | {meta.iloc[i].get('response_type')}"
    )
    fig.suptitle(title, fontsize=10)
    for ax, (name, input_col, out_idx) in zip(axes, panels):
        if input_col in fmap:
            ax.plot(INPUT_TIME, x[:, fmap[input_col]], color="#6b7280", lw=1.5, label="input true")
        valid = ymask[:, out_idx]
        ax.plot(TARGET_TIME[valid], y[valid, out_idx], color="#111827", lw=2.0, label="target true")
        ax.plot(TARGET_TIME[valid], pred[i, valid, out_idx], color="#dc2626", lw=1.8, ls="--", label="pred")
        ax.axvline(0.0, color="#2563eb", ls=":", lw=1.2)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("relative time / s")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_figures(run_dir: Path, arrays: dict[str, np.ndarray], pred: np.ndarray, meta: pd.DataFrame, per: pd.DataFrame, title_prefix: str) -> None:
    id_to_index = {str(row.get("sample_id", row.get("episode_uid", idx))): idx for idx, row in meta.iterrows()}
    fig_rows: list[dict[str, str]] = []
    for group, ids in select_figures(per).items():
        for sid in ids:
            idx = id_to_index.get(str(sid))
            if idx is None:
                continue
            file_name = f"{group}/{sid}.png".replace("\\", "_").replace("/", "_")
            out_path = run_dir / "figures" / file_name
            plot_sample(idx, arrays, pred, meta, out_path, title_prefix)
            fig_rows.append({"group": group, "sample_id": sid, "path": str(out_path)})
    pd.DataFrame(fig_rows).to_csv(run_dir / "figure_index.csv", index=False, encoding="utf-8-sig")
    html_lines = ["<html><meta charset='utf-8'><body>", f"<h1>{html.escape(title_prefix)}</h1>"]
    for row in fig_rows:
        rel = os.path.relpath(row["path"], run_dir).replace("\\", "/")
        html_lines.append(f"<h3>{html.escape(row['group'])} - {html.escape(row['sample_id'])}</h3><img src='{html.escape(rel)}' width='980'>")
    html_lines.append("</body></html>")
    (run_dir / "figure_index.html").write_text("\n".join(html_lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    cols = list(text.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_summary(run_dir: Path, args: argparse.Namespace, dataset_summary: dict[str, Any], train_info: dict[str, Any], metrics: pd.DataFrame) -> None:
    lines = [
        "# v2.2 vehicle-only first-round baseline",
        "",
        f"- experiment: `{args.experiment_name}`",
        f"- model: `{args.model}`",
        f"- dataset: `{args.dataset}`",
        f"- input: anchor -3s to 0s vehicle-only sequence, 20 Hz",
        f"- output: anchor 0s to +2s `steering_delta`, `yaw_rate`, `ay`",
        f"- no future peak features in input: yes",
        f"- samples: `{dataset_summary.get('sample_count')}` usable from v2.2 training pool",
        f"- best epoch: `{train_info.get('best_epoch')}`",
        f"- best val loss: `{train_info.get('best_val_loss')}`",
        "",
        "## Metrics",
        "",
        markdown_table(metrics),
        "",
        "## Files",
        "",
        f"- per-sample metrics: `{run_dir / 'per_sample_metrics.csv'}`",
        f"- aggregate metrics: `{run_dir / 'metrics_summary.csv'}`",
        f"- figures: `{run_dir / 'figure_index.html'}`",
    ]
    (run_dir / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def append_result_index(run_dir: Path, args: argparse.Namespace, metrics: pd.DataFrame, train_info: dict[str, Any]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    index_path = OUT_ROOT / "v22_first_round_result_index.csv"
    test_row = metrics[metrics["subset"].eq("test_included")]
    row: dict[str, Any] = {
        "experiment": args.experiment_name,
        "model": args.model,
        "dataset": args.dataset,
        "run_dir": str(run_dir),
        "best_epoch": train_info.get("best_epoch"),
        "best_val_loss": train_info.get("best_val_loss"),
    }
    if len(test_row):
        row.update(test_row.iloc[0].to_dict())
    exists = index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    arrays, meta, dataset_summary = build_arrays(force_rebuild=args.force_rebuild)
    include = include_mask_for_dataset(meta, args.dataset)
    train_idx, val_idx, test_idx = split_indices(meta, include)
    run_dir = OUT_ROOT / "runs" / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "dataset_counts.json").write_text(
        json.dumps(
            {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
                "dataset": args.dataset,
                "role_counts_included": meta[include]["v2_1_role"].value_counts().to_dict(),
                "split_counts_included": meta[include]["split"].value_counts().to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.model == "persistence":
        pred = persistence_prediction(arrays)
        train_info = {"best_epoch": 0, "best_val_loss": float("nan"), "epochs_ran": 0, "lr": 0.0}
    else:
        pred, train_info = train_model(args, arrays, meta, run_dir)

    np.savez_compressed(run_dir / "predictions.npz", pred=pred, y=arrays["Y"], y_mask=arrays["Y_mask"])
    per, metrics = compute_all_metrics(arrays["Y"], pred, arrays["Y_mask"], meta, include)
    per.to_csv(run_dir / "per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(run_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    write_figures(run_dir, arrays, pred, meta, per, args.experiment_name)
    write_summary(run_dir, args, dataset_summary, train_info, metrics)
    append_result_index(run_dir, args, metrics, train_info)
    print(metrics.to_string(index=False), flush=True)
    print(f"RUN_DIR={run_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2.2 vehicle-only first-round sequence baselines")
    parser.add_argument("--model", choices=["persistence", "gru", "tcn", "transformer"], required=True)
    parser.add_argument("--dataset", choices=["core", "core_review", "core_review_control"], default="core")
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=-1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--review-weight", type=float, default=0.6)
    parser.add_argument("--control-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 3)
    if not args.experiment_name:
        suffix = "smoke" if args.smoke else "full"
        args.experiment_name = f"V22_{args.dataset}_{args.model}_{suffix}_seed{args.seed}"
    return args


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.build_only:
        arrays, meta, summary = build_arrays(force_rebuild=args.force_rebuild)
        print(json.dumps({"array_shape": arrays["X"].shape, "meta_rows": len(meta), **summary}, ensure_ascii=False, indent=2), flush=True)
        return
    run_experiment(args)


if __name__ == "__main__":
    main()
