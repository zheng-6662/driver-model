# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
TASK_MANIFEST_PATH = (
    ROOT
    / "02_samples"
    / "vehicle_instability_response_task_decision_v0_1"
    / "tables"
    / "sample_response_task_manifest.csv"
)
ARRAY_DIR = ROOT / "03_processed_datasets" / "vehicle_instability_allraw_highconf_v0_1" / "arrays"
CLEAN_BASELINE_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1"
DIRECT_TRANSFORMER_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1"
DIRECT_TRANSFORMER_METRICS_PATH = DIRECT_TRANSFORMER_DIR / "tables" / "clean_task_vehicle_transformer_metrics.csv"
RESPONSE_LABEL_PATH = (
    ROOT
    / "03_baselines"
    / "stage03_vehicle_instability_response_decomposition_labels_v0_1"
    / "tables"
    / "response_decomposition_sample_labels.csv"
)
OUT_ROOT = ROOT / "03_baselines" / "stage03_vehicle_instability_structured_vehicle_transformer_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
CHECKPOINT_DIR = OUT_ROOT / "checkpoints"
REPORT_ROOT = ROOT / "09_reports"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402
import stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1 as clean_v01  # noqa: E402


SPLIT_STRATEGY = "session_level_split"
MODEL_NAME = "structured_vehicle_transformer_aux_no_subject"
DIRECT_TRANSFORMER_MODEL = "vehicle_transformer_context_no_subject"
SEED = 2026051302
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 16
LR = 1.5e-3
WEIGHT_DECAY = 1e-4
TARGET_INPUT_TOKENS = 100

TRACKS = {
    "B_response3s_strict_core": {
        "window_config_id": "pre3_label3_response_coverage",
        "task_sample_role": "response3s_strict_core_candidate",
        "description_cn": "3秒响应覆盖严格核心候选；当前主线车辆-only 对照。",
    },
}

NUMERIC_CONTEXT_COLS = [
    "anchor_time_rel_s",
    "curvature_anchor",
    "input_valid_ratio",
]
CATEGORICAL_CONTEXT_COLS = [
    "event_type",
    "event_level",
    "road_type_anchor",
    "old_v400_road_type_mode",
    "old_v400_phase_mode",
    "road_design_module_name",
    "road_design_instance_name",
    "road_design_risk_class",
    "road_design_mapping_reliability",
]

PLOT_MODELS = [
    ("rbf_kernel_ridge_context_no_subject", "#1f77b4", "rbf krr"),
    ("knn_template_context_no_subject", "#ff7f0e", "knn"),
    (DIRECT_TRANSFORMER_MODEL, "#9467bd", "direct tx"),
    (MODEL_NAME, "#2ca02c", "structured"),
]

LABEL_MAPS = {
    "peak_direction": {"negative": 0, "positive": 1},
    "amplitude_bucket": {"small_response": 0, "medium_response": 1, "large_response": 2},
    "peak_time_bucket": {"early_peak": 0, "middle_peak": 1, "late_peak": 2},
    "onset_bucket": {"fast_onset": 0, "middle_onset": 1, "delayed_onset": 2, "no_onset": 3},
    "computed_morphology": {"single_lobe": 0, "reverse_correction": 1, "multi_correction": 2},
    "tail_state": {"returned_tail": 0, "residual_tail": 1, "unsettled_tail": 2},
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, CHECKPOINT_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def load_track(track_id: str, cfg: dict[str, str], manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    window_id = cfg["window_config_id"]
    role = cfg["task_sample_role"]
    rows = manifest[
        (manifest["window_config_id"].astype(str) == window_id)
        & (manifest["task_sample_role"].astype(str) == role)
    ].copy()
    if rows.empty:
        raise RuntimeError(f"{track_id}: no clean task samples for {window_id}/{role}")
    rows["array_row"] = pd.to_numeric(rows["array_row"], errors="raise").astype(int)
    rows = rows.sort_values("array_row").reset_index(drop=True)
    idx = rows["array_row"].to_numpy(dtype=int)
    z = np.load(ARRAY_DIR / f"{window_id}.npz", allow_pickle=True)
    y = z["label_steer_delta"].astype(np.float32)[idx]
    y_mask = z["label_valid_mask"].astype(bool)[idx]
    input_values = z["input_values"].astype(np.float32)[idx]
    input_mask = z["input_valid_mask"].astype(bool)[idx]
    input_time = z["input_time_rel_s"].astype(np.float32)
    label_time = z["label_time_rel_s"].astype(np.float32)
    rows["track_id"] = track_id
    rows["track_description_cn"] = cfg["description_cn"]
    return y, y_mask, input_values, input_mask, input_time, label_time, rows


def split_indices(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = meta[SPLIT_STRATEGY].astype(str).to_numpy()
    return np.where(split == "train")[0], np.where(split == "val")[0], np.where(split == "test")[0]


def build_context_features(meta: pd.DataFrame, train_idx: np.ndarray) -> tuple[np.ndarray, list[str]]:
    parts: list[np.ndarray] = []
    names: list[str] = []
    for col in NUMERIC_CONTEXT_COLS:
        if col not in meta.columns:
            continue
        values = pd.to_numeric(meta[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        mu = float(values[train_idx].mean()) if len(train_idx) else 0.0
        sigma = float(values[train_idx].std()) if len(train_idx) else 1.0
        if not np.isfinite(sigma) or sigma < 1e-6:
            sigma = 1.0
        parts.append(((values - mu) / sigma).reshape(-1, 1))
        names.append(col)
    for col in CATEGORICAL_CONTEXT_COLS:
        if col not in meta.columns:
            continue
        values = meta[col].astype(str).fillna("NA")
        train_values = sorted(values.iloc[train_idx].unique().tolist())
        for val in train_values:
            parts.append((values == val).to_numpy(dtype=np.float32).reshape(-1, 1))
            names.append(f"{col}={val}")
    if not parts:
        return np.zeros((len(meta), 0), dtype=np.float32), []
    return np.concatenate(parts, axis=1).astype(np.float32), names


def standardize_vehicle_inputs(
    input_values: np.ndarray,
    input_mask: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = input_values.astype(np.float32, copy=True)
    valid = input_mask & np.isfinite(x)
    train_valid = valid[train_idx]
    train_values = np.where(train_valid, x[train_idx], np.nan)
    mu = np.nanmean(train_values, axis=(0, 1)).astype(np.float32)
    sigma = np.nanstd(train_values, axis=(0, 1)).astype(np.float32)
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sigma = np.where(np.isfinite(sigma) & (sigma >= 1e-6), sigma, 1.0).astype(np.float32)
    scaled = (x - mu.reshape(1, 1, -1)) / sigma.reshape(1, 1, -1)
    scaled = np.where(valid, scaled, 0.0).astype(np.float32)
    return scaled, {"mean": mu.tolist(), "std": sigma.tolist(), "scope": "train split valid input points only"}


def label_scale_train(y: np.ndarray, y_mask: np.ndarray, train_idx: np.ndarray) -> float:
    values = np.where(y_mask[train_idx], y[train_idx], np.nan)
    scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale < 1e-6:
        return 1.0
    return scale


class VehicleWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        context: np.ndarray,
        y: np.ndarray,
        y_mask: np.ndarray,
        aux_targets: dict[str, np.ndarray],
        indices: np.ndarray,
        label_scale: float,
    ) -> None:
        self.x = torch.from_numpy(x[indices].astype(np.float32))
        self.context = torch.from_numpy(context[indices].astype(np.float32))
        self.y = torch.from_numpy((y[indices] / label_scale).astype(np.float32))
        self.y_mask = torch.from_numpy(y_mask[indices].astype(bool))
        self.aux = {k: torch.from_numpy(v[indices].astype(np.int64)) for k, v in aux_targets.items()}

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return self.x[idx], self.context[idx], self.y[idx], self.y_mask[idx], {k: v[idx] for k, v in self.aux.items()}


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class StructuredVehicleTransformer(nn.Module):
    def __init__(
        self,
        vehicle_dim: int,
        context_dim: int,
        label_time: np.ndarray,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.vehicle_proj = nn.Linear(vehicle_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=160,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        if context_dim > 0:
            self.context_net = nn.Sequential(nn.Linear(context_dim, 48), nn.GELU(), nn.Dropout(dropout))
            context_out = 48
        else:
            self.context_net = None
            context_out = 0
        self.time_net = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        global_dim = d_model + context_out
        self.direction_head = nn.Linear(global_dim, 2)
        self.large_head = nn.Linear(global_dim, 2)
        self.amplitude_head = nn.Linear(global_dim, 3)
        self.peak_time_head = nn.Linear(global_dim, 3)
        self.onset_head = nn.Linear(global_dim, 4)
        self.morphology_head = nn.Linear(global_dim, 3)
        self.tail_head = nn.Linear(global_dim, 3)
        structure_dim = 2 + 2 + 3 + 3 + 4 + 3 + 3
        self.structure_net = nn.Sequential(nn.Linear(structure_dim, 32), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(global_dim + 32 + 32, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        t = torch.as_tensor(label_time.astype(np.float32)).reshape(1, -1, 1)
        if float(torch.max(torch.abs(t))) > 0:
            t = t / float(torch.max(torch.abs(t)))
        self.register_buffer("label_time_norm", t, persistent=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.vehicle_proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        pieces = [pooled]
        if self.context_net is not None:
            pieces.append(self.context_net(context))
        global_repr = torch.cat(pieces, dim=1)
        aux = {
            "peak_direction": self.direction_head(global_repr),
            "is_large_response_target": self.large_head(global_repr),
            "amplitude_bucket": self.amplitude_head(global_repr),
            "peak_time_bucket": self.peak_time_head(global_repr),
            "onset_bucket": self.onset_head(global_repr),
            "computed_morphology": self.morphology_head(global_repr),
            "tail_state": self.tail_head(global_repr),
        }
        structure_vec = torch.cat(
            [
                torch.softmax(aux["peak_direction"], dim=1),
                torch.softmax(aux["is_large_response_target"], dim=1),
                torch.softmax(aux["amplitude_bucket"], dim=1),
                torch.softmax(aux["peak_time_bucket"], dim=1),
                torch.softmax(aux["onset_bucket"], dim=1),
                torch.softmax(aux["computed_morphology"], dim=1),
                torch.softmax(aux["tail_state"], dim=1),
            ],
            dim=1,
        )
        structure_feat = self.structure_net(structure_vec)
        batch = x.shape[0]
        t = self.label_time_norm.to(x.device).expand(batch, -1, -1)
        t_feat = self.time_net(t)
        global_aug = torch.cat([global_repr, structure_feat], dim=1)
        g = global_aug.unsqueeze(1).expand(-1, t_feat.shape[1], -1)
        out = self.head(torch.cat([g, t_feat], dim=2)).squeeze(-1)
        aux["traj"] = out - out[:, :1]
        return aux


def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    mse = (((pred - target) ** 2) * mask_f).sum() / denom
    if pred.shape[1] < 2:
        return mse
    dmask = (mask[:, 1:] & mask[:, :-1]).float()
    ddenom = dmask.sum().clamp_min(1.0)
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    dmse = (((dp - dt) ** 2) * dmask).sum() / ddenom
    return mse + 0.08 * dmse


def multitask_loss(out: dict[str, torch.Tensor], target: torch.Tensor, mask: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
    loss = masked_loss(out["traj"], target, mask)
    ce = nn.functional.cross_entropy
    weights = {
        "peak_direction": 0.18,
        "is_large_response_target": 0.12,
        "amplitude_bucket": 0.14,
        "peak_time_bucket": 0.10,
        "onset_bucket": 0.08,
        "computed_morphology": 0.10,
        "tail_state": 0.08,
    }
    for name, weight in weights.items():
        loss = loss + weight * ce(out[name], aux[name])
    return loss


@torch.no_grad()
def predict_all(model: nn.Module, x: np.ndarray, context: np.ndarray, label_scale: float, batch_size: int = 32) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    preds: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
        cb = torch.from_numpy(context[start : start + batch_size].astype(np.float32)).to(device)
        pred = model(xb, cb)["traj"].cpu().numpy() * label_scale
        preds.append(pred.astype(np.float32))
    return np.concatenate(preds, axis=0).astype(np.float32)


def attach_response_labels(meta: pd.DataFrame, label_table: pd.DataFrame, track_id: str) -> pd.DataFrame:
    label_cols = [
        "track_id",
        "sample_id",
        "peak_direction",
        "amplitude_bucket",
        "peak_time_bucket",
        "onset_bucket",
        "computed_morphology",
        "tail_state",
        "is_large_response_target",
    ]
    labels = label_table[label_table["track_id"].astype(str) == track_id][label_cols].copy()
    out = meta.merge(labels, on=["track_id", "sample_id"], how="left", validate="one_to_one")
    missing = out["peak_direction"].isna().sum()
    if missing:
        raise RuntimeError(f"{track_id}: missing response decomposition labels for {missing} samples")
    return out


def build_aux_targets(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    for col, mapping in LABEL_MAPS.items():
        unknown = sorted(set(meta[col].astype(str)) - set(mapping))
        if unknown:
            raise RuntimeError(f"unknown {col} labels: {unknown}")
        targets[col] = meta[col].astype(str).map(mapping).to_numpy(dtype=np.int64)
    targets["is_large_response_target"] = meta["is_large_response_target"].astype(bool).astype(np.int64).to_numpy()
    return targets


@torch.no_grad()
def predict_aux_all(model: nn.Module, x: np.ndarray, context: np.ndarray, batch_size: int = 32) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    buckets: dict[str, list[np.ndarray]] = {name: [] for name in LABEL_MAPS}
    buckets["is_large_response_target"] = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
        cb = torch.from_numpy(context[start : start + batch_size].astype(np.float32)).to(device)
        out = model(xb, cb)
        for name in buckets:
            buckets[name].append(out[name].argmax(dim=1).cpu().numpy().astype(np.int64))
    return {k: np.concatenate(v, axis=0) for k, v in buckets.items()}


def aux_accuracy_table(aux_true: dict[str, np.ndarray], aux_pred: dict[str, np.ndarray], meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    splits = meta[SPLIT_STRATEGY].astype(str).to_numpy()
    for split in ["train", "val", "test"]:
        mask = splits == split
        if not mask.any():
            continue
        row: dict[str, Any] = {"split": split, "n_samples": int(mask.sum())}
        for name, true_values in aux_true.items():
            row[f"{name}_accuracy"] = float((aux_pred[name][mask] == true_values[mask]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def train_one_track(
    track_id: str,
    x: np.ndarray,
    context: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    aux_targets: dict[str, np.ndarray],
    label_time: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    label_scale: float,
) -> tuple[StructuredVehicleTransformer, pd.DataFrame, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StructuredVehicleTransformer(
        vehicle_dim=x.shape[2],
        context_dim=context.shape[1],
        label_time=label_time,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_ds = VehicleWindowDataset(x, context, y, y_mask, aux_targets, train_idx, label_scale)
    val_ds = VehicleWindowDataset(x, context, y, y_mask, aux_targets, val_idx, label_scale)
    train_loader = DataLoader(train_ds, batch_size=min(BATCH_SIZE, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(BATCH_SIZE, max(1, len(val_ds))), shuffle=False)
    best_state = None
    best_val_rmse = float("inf")
    best_epoch = 0
    bad_epochs = 0
    rows: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for xb, cb, yb, mb, ab in train_loader:
            xb, cb, yb, mb = xb.to(device), cb.to(device), yb.to(device), mb.to(device)
            ab = {k: v.to(device) for k, v in ab.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = multitask_loss(model(xb, cb), yb, mb, ab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_loss_num = 0.0
        val_count = 0.0
        with torch.no_grad():
            for xb, cb, yb, mb, _ab in val_loader:
                xb, cb, yb, mb = xb.to(device), cb.to(device), yb.to(device), mb.to(device)
                pred = model(xb, cb)["traj"]
                mask_f = mb.float()
                val_loss_num += float((((pred - yb) ** 2) * mask_f).sum().cpu())
                val_count += float(mask_f.sum().cpu())
        val_rmse = math.sqrt(max(val_loss_num / max(val_count, 1.0), 0.0)) * label_scale
        rows.append(
            {
                "track_id": track_id,
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else np.nan,
                "val_rmse": val_rmse,
            }
        )
        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {
        "track_id": track_id,
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse),
        "epochs_ran": int(len(rows)),
        "early_stopping_patience": PATIENCE,
        "max_epochs": MAX_EPOCHS,
        "optimizer": f"AdamW(lr={LR}, weight_decay={WEIGHT_DECAY})",
        "loss": "masked trajectory MSE + 0.08 first-difference MSE + response-decomposition auxiliary CE heads",
        "device": str(device),
    }
    return model, pd.DataFrame(rows), info


def evaluate_predictions(
    track_id: str,
    window_id: str,
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    train_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_peak = np.nanmax(np.abs(np.where(y_mask, y, np.nan)), axis=1)
    large_thr = float(np.nanpercentile(gt_peak[train_idx], 75))
    difficult_thr = float(np.nanpercentile(gt_peak[train_idx], 80))
    rows: list[pd.DataFrame] = []
    for split_name in ["train", "val", "test"]:
        mask = meta[SPLIT_STRATEGY].astype(str).to_numpy() == split_name
        if not mask.any():
            continue
        split_meta = meta.loc[mask].reset_index(drop=True)
        for model_name, pred in predictions.items():
            sample_rows = eval_utils.sample_metric_rows(
                y[mask],
                pred[mask],
                y_mask[mask],
                label_time,
                split_meta,
                model_name=model_name,
                split_strategy=SPLIT_STRATEGY,
                split_name=split_name,
                window_id=window_id,
                large_thr=large_thr,
                difficult_thr=difficult_thr,
            )
            if sample_rows:
                part = pd.DataFrame(sample_rows)
                part["track_id"] = track_id
                rows.append(part)
    per_sample = pd.concat(rows, ignore_index=True)
    metrics = eval_utils.aggregate_metrics(per_sample)
    metrics["track_id"] = track_id
    return metrics, per_sample


def select_best_val(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    val = metrics[metrics["split"] == "val"].copy()
    for track_id, part in val.groupby("track_id"):
        rows.append(part.sort_values("rmse_steer").iloc[0].to_dict())
    return pd.DataFrame(rows)


def append_direct_transformer_reference(metrics: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    metrics = metrics.copy()
    metrics["comparison_source"] = "current_structured_vehicle_transformer_run"
    if not DIRECT_TRANSFORMER_METRICS_PATH.exists():
        return metrics, False
    ref = pd.read_csv(DIRECT_TRANSFORMER_METRICS_PATH)
    ref = ref[
        ref["track_id"].isin(TRACKS.keys())
        & (ref["model_name"] == DIRECT_TRANSFORMER_MODEL)
    ].copy()
    if ref.empty:
        return metrics, False
    ref["comparison_source"] = "stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1"
    return pd.concat([metrics, ref], ignore_index=True, sort=False), True


def plot_metric_summary(metrics: pd.DataFrame) -> Path:
    test = metrics[(metrics["split"] == "test") & (metrics["model_name"].isin([m[0] for m in PLOT_MODELS]))].copy()
    tracks = list(TRACKS.keys())
    fig, axes = plt.subplots(len(tracks), 2, figsize=(15, 5.2 * len(tracks)), squeeze=False)
    for i, track_id in enumerate(tracks):
        part = test[test["track_id"] == track_id].set_index("model_name").reindex([m[0] for m in PLOT_MODELS]).dropna(subset=["rmse_steer"])
        labels = [dict((m[0], m[2]) for m in PLOT_MODELS).get(v, v) for v in part.index]
        axes[i, 0].barh(labels, part["rmse_steer"].to_numpy(), color="#4c78a8")
        axes[i, 0].set_title(f"{track_id}: test RMSE")
        axes[i, 0].grid(axis="x", alpha=0.25)
        axes[i, 1].barh(labels, part["wrong_side_rate"].to_numpy(), color="#e45756")
        axes[i, 1].set_title(f"{track_id}: test wrong-side rate")
        axes[i, 1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "structured_vehicle_transformer_metric_summary_test.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_samples(
    track_id: str,
    sample_ids: list[str],
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    n = len(sample_ids)
    cols = 4
    rows = int(np.ceil(max(n, 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, max(3.2 * rows, 3.4)), squeeze=False)
    id_to_idx = {sid: i for i, sid in enumerate(meta["sample_id"].astype(str))}
    colors = {name: color for name, color, _ in PLOT_MODELS}
    display = {name: label for name, _, label in PLOT_MODELS}
    for ax in axes.ravel():
        ax.axis("off")
    for k, sid in enumerate(sample_ids):
        ax = axes.ravel()[k]
        ax.axis("on")
        idx = id_to_idx[sid]
        gt = np.where(y_mask[idx] & np.isfinite(y[idx]), y[idx], np.nan)
        ax.plot(label_time, gt, color="black", linewidth=1.8, label="GT")
        for model_name, color in colors.items():
            if model_name in predictions:
                ax.plot(label_time, predictions[model_name][idx], color=color, linewidth=1.1, alpha=0.95, label=display.get(model_name, model_name))
        ax.axhline(0, color="#dddddd", linewidth=0.8)
        ax.set_title(f"{meta.at[idx, 'subject']} {meta.at[idx, 'anchor_time_rel_s']:.1f}s\npeak={np.nanmax(np.abs(gt)):.2f}", fontsize=9)
        ax.tick_params(labelsize=8)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(5, len(labels)), fontsize=8)
    fig.suptitle(title, fontsize=12, y=0.955)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_reports(metrics: pd.DataFrame, best_val: pd.DataFrame, model_info: pd.DataFrame, figures: dict[str, str], aux_metrics: pd.DataFrame) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    show_cols = [
        "track_id",
        "model_name",
        "n_samples",
        "rmse_steer",
        "peak_direction_accuracy",
        "wrong_side_rate",
        "large_response_recall",
        "peak_amp_mae",
        "severe_amp_under_rate",
        "peak_time_mae_s",
        "onset_delay_mae_s",
        "tail_abs_error_mean",
        "tail_drift_risk_rate",
        "reversal_count_exact_match_rate",
        "difficult_top20_rmse",
    ]
    test_comp = test[test["model_name"].isin([m[0] for m in PLOT_MODELS])].sort_values(["track_id", "rmse_steer"])
    test_table = test_comp[[c for c in show_cols if c in test_comp.columns]]
    best_lines = []
    for _, row in best_val.iterrows():
        track_id = str(row["track_id"])
        model = str(row["model_name"])
        test_row = test[(test["track_id"] == track_id) & (test["model_name"] == model)]
        if not test_row.empty:
            r = test_row.iloc[0]
            best_lines.append(
                f"- {track_id}: val 选择 `{model}`；test RMSE={r['rmse_steer']:.6f}，错侧率={r['wrong_side_rate']:.6f}，大幅响应召回={r['large_response_recall']:.6f}。"
            )
    transformer_lines = []
    for track_id, part in test[test["model_name"] == MODEL_NAME].groupby("track_id"):
        r = part.iloc[0]
        transformer_lines.append(
            f"- {track_id}: 结构化 Transformer test RMSE={r['rmse_steer']:.6f}，错侧率={r['wrong_side_rate']:.6f}，大幅响应召回={r['large_response_recall']:.6f}，反向修正完全匹配率={r['reversal_count_exact_match_rate']:.6f}。"
        )
    b_note = "未找到 B 轨道结构化 Transformer test 结果。"
    b_row = test[(test["track_id"] == "B_response3s_strict_core") & (test["model_name"] == MODEL_NAME)]
    b_rbf = test[(test["track_id"] == "B_response3s_strict_core") & (test["model_name"] == "rbf_kernel_ridge_context_no_subject")]
    if not b_row.empty:
        r = b_row.iloc[0]
        if not b_rbf.empty:
            rb = b_rbf.iloc[0]
            b_note = (
                f"B 轨道结构化 Transformer 的 test RMSE={r['rmse_steer']:.6f}，RBF KRR 为 {rb['rmse_steer']:.6f}；"
                f"wrong-side 二者同为 {r['wrong_side_rate']:.6f}/{rb['wrong_side_rate']:.6f}，"
                f"large recall 为 {r['large_response_recall']:.6f} vs {rb['large_response_recall']:.6f}。"
                "结构化辅助头是否值得继续，要同时看 RMSE、方向、幅值、峰时、尾段和坏样本图。"
            )
        else:
            b_note = (
                f"B 轨道结构化 Transformer 的 test RMSE={r['rmse_steer']:.6f}，wrong-side={r['wrong_side_rate']:.6f}，"
                f"large recall={r['large_response_recall']:.6f}。"
            )
    info_cols = [
        "track_id",
        "best_epoch",
        "best_val_rmse",
        "epochs_ran",
        "model_name",
        "window_config_id",
        "task_sample_role",
        "train_n",
        "val_n",
        "test_n",
        "label_scale_train_std",
        "context_feature_count",
        "vehicle_input_tokens",
        "vehicle_input_downsample_step",
        "uses_response_decomposition_labels_as_targets",
        "trajectory_uses_predicted_structure_features",
        "prediction_zero_origin_constraint",
        "device",
        "checkpoint_path",
    ]
    model_text = model_info[[c for c in info_cols if c in model_info.columns]].to_string(index=False)
    aux_text = aux_metrics.to_string(index=False)
    best_text = "\n".join(best_lines)
    transformer_text = "\n".join(transformer_lines)
    report = f"""# 阶段 3：B 轨道车辆-only 响应分解 Transformer v0.1

生成时间：2026-05-13

## 为什么做

直接车辆-only Transformer 没有超过 B 轨道 RBF KRR，且坏样本仍集中在反向修正、多段修正、峰值时间、启动延迟和尾段问题。因此本轮在 B 轨道上做响应分解辅助头：先让模型预测方向、幅值桶、峰值时间桶、启动时间桶、响应形态和尾段状态，再用模型自己的结构表征辅助轨迹头。

## 输入和无泄漏边界

- 样本 manifest：`{TASK_MANIFEST_PATH.as_posix()}`
- B 轨道：`pre3_label3_response_coverage` + `response3s_strict_core_candidate`，270 个样本。
- 输入：事件前车辆时序 9 个车辆特征 + 可因果获得的事件/道路上下文。
- 响应分解标签：只作为训练目标和评估目标，不作为推理输入。
- 不使用：生理、脑电、连续风格、驾驶员 ID、真实响应分解标签输入、`eval_label_*` 未来标签输入。
- 标准化：车辆时序和数值上下文只在各轨道 train split 拟合。
- 模型选择：结构化 Transformer 早停只看 val 轨迹 RMSE；test 只用于最终评估。
- 物理边界：模型内部扣除自己的 t=0 输出，使方向盘增量轨迹从 0 开始。
- 本轮未连接服务器，未读取服务器指令与密码文件。

## 模型信息

```text
{model_text}
```

## test 指标对照

说明：`direct tx` 是上一轮真正 vehicle-only Transformer 的同一 B 轨道结果；`knn` 只是模板参照，不是本轮主模型。

```text
{test_table.to_string(index=False)}
```

## 响应分解辅助头准确率

```text
{aux_text}
```

## 按 val 选择的当前结果

{best_text}

## 结构化 Transformer 单独结果

{transformer_text}

## B 轨道判断

{b_note}

## 图

- 指标概览：`{figures.get('metric_summary', '')}`
- B 轨道固定图：`{figures.get('B_response3s_strict_core_fixed', '')}`
- B 轨道坏样本图：`{figures.get('B_response3s_strict_core_bad', '')}`

## 当前结论边界

这一步只回答车辆历史和事件/道路上下文下的结构化车辆模型表现，不能说明连续风格、生理或 EEG 有效。如果响应分解辅助头带来物理指标改善但 RMSE 没有明显改善，只能把它作为结构路线候选，不能升级为最终主线。
"""
    (REPORT_ROOT / "stage03_vehicle_instability_structured_vehicle_transformer_v0_1_cn.md").write_text(report, encoding="utf-8")

    user = f"""# 阶段 3 用户查看版：B 轨道车辆-only 响应分解 Transformer v0.1

## 为什么做

上一轮已经补跑了真正的车辆-only Transformer，但它没有超过 B 轨道 RBF KRR。这个阶段进一步测试“结构化预测”是否更合适：模型先预测方向、幅值、峰值时间、启动时间、响应形态和尾段状态，再预测轨迹。

## 这次检查了什么

- B 轨道：3 秒响应覆盖严格核心样本，270 条，是当前更重要的主线。
- 输入只用事件前车辆历史和道路/事件上下文。
- 不用生理、脑电、连续风格、驾驶员 ID，也不把未来响应分解标签当输入；响应分解标签只作为训练目标。
- 模型内部加了一个物理约束：方向盘增量在 t=0 从 0 开始。

## 目前发现

一句话判断：这是车辆-only 结构化路线的第一版，不是生理/风格实验；是否继续要看它相对 RBF KRR 是否改善物理错误和坏样本。
表里的 `direct tx` 是上一轮真正 vehicle-only Transformer；`knn` 只是模板参照，不是本轮要验证的 Transformer。

```text
{test_table.to_string(index=False)}
```

## 哪些结果可信

可信的是：这次只在 B 轨道干净响应样本上跑了车辆-only 结构化 Transformer，训练标准化和早停都只看 train/val，没有把 test 信息用于训练。

## 哪些还不能下结论

还不能说生理、脑电或连续风格有效；也不能只因为加了响应分解辅助头就默认更强，必须看 B 轨道 test 指标、辅助头准确率和坏样本图。

## 下一步是否可以继续

可以继续。若这版结构化模型仍然解决不了反向修正、多段修正、峰值时间和尾段错误，下一步应尝试关键点+残差或多假设车辆模型，而不是跳到生理结论。

## 推荐优先查看

1. `{figures.get('B_response3s_strict_core_fixed', '')}`
2. `{figures.get('B_response3s_strict_core_bad', '')}`
3. `{figures.get('metric_summary', '')}`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/tables/structured_vehicle_transformer_metrics.csv`
"""
    (REPORT_ROOT / "stage03_vehicle_instability_structured_vehicle_transformer_user_summary_cn.md").write_text(user, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    set_seed(SEED)
    manifest = pd.read_csv(TASK_MANIFEST_PATH)
    label_table = pd.read_csv(RESPONSE_LABEL_PATH)
    all_metrics: list[pd.DataFrame] = []
    all_per_sample: list[pd.DataFrame] = []
    all_info: list[dict[str, Any]] = []
    all_history: list[pd.DataFrame] = []
    all_aux_metrics: list[pd.DataFrame] = []
    figures: dict[str, str] = {}
    summary_tracks: list[dict[str, Any]] = []

    for track_id, cfg in TRACKS.items():
        y, y_mask, input_values, input_mask, input_time, label_time, meta = load_track(track_id, cfg, manifest)
        meta = attach_response_labels(meta, label_table, track_id)
        aux_targets = build_aux_targets(meta)
        train_idx, val_idx, test_idx = split_indices(meta)
        x_scaled, scaler_info = standardize_vehicle_inputs(input_values, input_mask, train_idx)
        context, context_names = build_context_features(meta, train_idx)
        step = max(1, int(round(len(input_time) / TARGET_INPUT_TOKENS)))
        x_model = x_scaled[:, ::step, :].copy()
        label_scale = label_scale_train(y, y_mask, train_idx)

        baseline_predictions, baseline_info = clean_v01.build_strong_predictions(
            track_id,
            cfg["window_config_id"],
            y,
            y_mask,
            input_values,
            input_time,
            label_time,
            meta,
            train_idx,
            val_idx,
        )
        model, history, train_info = train_one_track(track_id, x_model, context, y, y_mask, aux_targets, label_time, train_idx, val_idx, label_scale)
        pred_transformer = predict_all(model, x_model, context, label_scale)
        aux_pred = predict_aux_all(model, x_model, context)
        aux_metrics = aux_accuracy_table(aux_targets, aux_pred, meta)
        aux_metrics["track_id"] = track_id
        aux_metrics["model_name"] = MODEL_NAME
        all_aux_metrics.append(aux_metrics)
        predictions = dict(baseline_predictions)
        predictions[MODEL_NAME] = pred_transformer

        metrics, per_sample = evaluate_predictions(track_id, cfg["window_config_id"], y, y_mask, label_time, meta, predictions, train_idx)
        all_metrics.append(metrics)
        all_per_sample.append(per_sample)
        history["window_config_id"] = cfg["window_config_id"]
        all_history.append(history)

        ckpt_path = CHECKPOINT_DIR / f"{track_id}_{MODEL_NAME}_best.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "track_id": track_id,
                "model_name": MODEL_NAME,
                "window_config_id": cfg["window_config_id"],
                "split_strategy": SPLIT_STRATEGY,
                "label_scale": label_scale,
                "context_names": context_names,
                "input_downsample_step": step,
                "scaler_info": scaler_info,
                "train_info": train_info,
                "label_maps": LABEL_MAPS,
            },
            ckpt_path,
        )
        info = {
            **train_info,
            "model_name": MODEL_NAME,
            "window_config_id": cfg["window_config_id"],
            "task_sample_role": cfg["task_sample_role"],
            "split_strategy": SPLIT_STRATEGY,
            "train_n": int(len(train_idx)),
            "val_n": int(len(val_idx)),
            "test_n": int(len(test_idx)),
            "label_scale_train_std": label_scale,
            "context_feature_count": int(context.shape[1]),
            "vehicle_feature_count": int(input_values.shape[2]),
            "vehicle_input_tokens": int(x_model.shape[1]),
            "vehicle_input_downsample_step": int(step),
            "uses_subject_id": False,
            "uses_physio": False,
            "uses_eeg": False,
            "uses_continuous_style": False,
            "uses_response_decomposition_labels_as_input": False,
            "uses_response_decomposition_labels_as_targets": True,
            "trajectory_uses_predicted_structure_features": True,
            "prediction_zero_origin_constraint": True,
            "server_used": False,
            "credential_file_read": False,
            "raw_files_modified": False,
            "standardization_scope": "train split only",
            "checkpoint_path": str(ckpt_path).replace("\\", "/"),
        }
        all_info.append(info)
        for base in baseline_info:
            base = dict(base)
            base["is_refit_for_plot_and_comparison"] = True
            all_info.append(base)

        pd.DataFrame({"track_id": track_id, "context_feature": context_names}).to_csv(
            TABLE_DIR / f"{track_id}_context_features.csv", index=False, encoding="utf-8-sig"
        )

        test_transformer = per_sample[(per_sample["split"] == "test") & (per_sample["model_name"] == MODEL_NAME)].copy()
        fixed_ids = meta.loc[test_idx, "sample_id"].astype(str).head(12).tolist()
        bad_ids = test_transformer.sort_values("sample_rmse", ascending=False).head(12)["sample_id"].astype(str).tolist()
        fixed_fig = FIG_DIR / f"{track_id}_fixed_predictions_test.png"
        bad_fig = FIG_DIR / f"{track_id}_structured_bad_samples_test.png"
        plot_samples(track_id, fixed_ids, y, y_mask, label_time, meta, predictions, fixed_fig, f"{track_id}: fixed test samples")
        plot_samples(track_id, bad_ids, y, y_mask, label_time, meta, predictions, bad_fig, f"{track_id}: worst structured Transformer test samples")
        pd.DataFrame({"track_id": track_id, "sample_id": fixed_ids}).to_csv(
            TABLE_DIR / f"{track_id}_fixed_plot_samples.csv", index=False, encoding="utf-8-sig"
        )
        test_transformer[test_transformer["sample_id"].isin(bad_ids)].sort_values("sample_rmse", ascending=False).to_csv(
            TABLE_DIR / f"{track_id}_structured_bad_plot_samples.csv", index=False, encoding="utf-8-sig"
        )
        figures[f"{track_id}_fixed"] = str(fixed_fig).replace("\\", "/")
        figures[f"{track_id}_bad"] = str(bad_fig).replace("\\", "/")
        summary_tracks.append(
            {
                "track_id": track_id,
                "window_config_id": cfg["window_config_id"],
                "task_sample_role": cfg["task_sample_role"],
                "n_samples": int(len(meta)),
                "train_n": int(len(train_idx)),
                "val_n": int(len(val_idx)),
                "test_n": int(len(test_idx)),
                "label_horizon_s": float(label_time[-1] - label_time[0]),
                "vehicle_input_tokens": int(x_model.shape[1]),
                "context_feature_count": int(context.shape[1]),
            }
        )

    metrics_all_current = pd.concat(all_metrics, ignore_index=True)
    metrics_all, direct_reference_loaded = append_direct_transformer_reference(metrics_all_current)
    per_sample_all = pd.concat(all_per_sample, ignore_index=True)
    info_all = pd.DataFrame(all_info)
    history_all = pd.concat(all_history, ignore_index=True)
    aux_metrics_all = pd.concat(all_aux_metrics, ignore_index=True)
    best_val = select_best_val(metrics_all)
    track_summary = pd.DataFrame(summary_tracks)
    metric_fig = plot_metric_summary(metrics_all)
    figures["metric_summary"] = str(metric_fig).replace("\\", "/")

    metrics_all.to_csv(TABLE_DIR / "structured_vehicle_transformer_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample_all.to_csv(TABLE_DIR / "structured_vehicle_transformer_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    info_all.to_csv(TABLE_DIR / "structured_vehicle_transformer_model_info.csv", index=False, encoding="utf-8-sig")
    history_all.to_csv(TABLE_DIR / "structured_vehicle_transformer_training_history.csv", index=False, encoding="utf-8-sig")
    best_val.to_csv(TABLE_DIR / "structured_vehicle_transformer_val_selected_models.csv", index=False, encoding="utf-8-sig")
    track_summary.to_csv(TABLE_DIR / "structured_vehicle_transformer_track_summary.csv", index=False, encoding="utf-8-sig")
    aux_metrics_all.to_csv(TABLE_DIR / "structured_vehicle_transformer_aux_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_version": "stage03_vehicle_instability_structured_vehicle_transformer_v0_1",
        "tracks": summary_tracks,
        "model_name": MODEL_NAME,
        "seed": SEED,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "uses_subject_id": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "uses_response_decomposition_labels_as_input": False,
        "uses_response_decomposition_labels_as_targets": True,
        "trajectory_uses_predicted_structure_features": True,
        "prediction_zero_origin_constraint": True,
        "direct_transformer_reference_metrics_loaded": direct_reference_loaded,
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "metrics_path": str(TABLE_DIR / "structured_vehicle_transformer_metrics.csv").replace("\\", "/"),
        "per_sample_path": str(TABLE_DIR / "structured_vehicle_transformer_per_sample_metrics.csv").replace("\\", "/"),
        "aux_metrics_path": str(TABLE_DIR / "structured_vehicle_transformer_aux_metrics.csv").replace("\\", "/"),
        "figures": figures,
    }
    (LOG_DIR / "structured_vehicle_transformer_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(metrics_all, best_val, info_all[info_all["model_name"] == MODEL_NAME].copy(), figures, aux_metrics_all)
    print(metrics_all[(metrics_all["split"] == "test") & (metrics_all["model_name"].isin([m[0] for m in PLOT_MODELS]))].sort_values(["track_id", "rmse_steer"]).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
