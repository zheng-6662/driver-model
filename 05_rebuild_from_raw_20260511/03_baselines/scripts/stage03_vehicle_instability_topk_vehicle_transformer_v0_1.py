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
OUT_ROOT = ROOT / "03_baselines" / "stage03_vehicle_instability_topk_vehicle_transformer_v0_1"
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
import stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1 as keypoint_v01  # noqa: E402


OUTPUT_VERSION = "stage03_vehicle_instability_topk_vehicle_transformer_v0_1"
TRACK_ID = "B_response3s_strict_core"
SPLIT_STRATEGY = "session_level_split"
RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
TOP1_MODEL = "topk_vehicle_transformer_top1_no_subject"
BESTK_MODEL = "topk_vehicle_transformer_best_of_3_oracle"
BRANCH_PREFIX = "topk_vehicle_transformer_branch"
SEED = 2026051304
K = 3
MAX_EPOCHS = 140
PATIENCE = 20
BATCH_SIZE = 16
LR = 1.2e-3
WEIGHT_DECAY = 1e-4
TARGET_INPUT_TOKENS = 100

REFERENCE_METRICS = (
    ROOT
    / "03_baselines"
    / "stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1"
    / "tables"
    / "multihypothesis_metrics.csv"
)

PLOT_MODELS = [
    (RBF_MODEL, "#1f77b4", "rbf"),
    (TOP1_MODEL, "#d62728", "top1"),
    (BESTK_MODEL, "#2ca02c", "best-k"),
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, CHECKPOINT_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


class VehicleDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        context: np.ndarray,
        y: np.ndarray,
        y_mask: np.ndarray,
        indices: np.ndarray,
        label_scale: float,
    ) -> None:
        self.x = torch.from_numpy(x[indices].astype(np.float32))
        self.context = torch.from_numpy(context[indices].astype(np.float32))
        self.y = torch.from_numpy((y[indices] / label_scale).astype(np.float32))
        self.y_mask = torch.from_numpy(y_mask[indices].astype(bool))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.context[idx], self.y[idx], self.y_mask[idx]


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


class TopKVehicleTransformer(nn.Module):
    def __init__(
        self,
        vehicle_dim: int,
        context_dim: int,
        label_time: np.ndarray,
        k: int = K,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.k = int(k)
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
        global_dim = d_model + context_out
        self.logit_head = nn.Sequential(nn.Linear(global_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, self.k))
        self.time_net = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.branch_embed = nn.Parameter(torch.randn(self.k, 24) * 0.02)
        self.traj_head = nn.Sequential(
            nn.Linear(global_dim + 32 + 24, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        t = label_time.astype(np.float32)
        t_min = float(np.nanmin(t))
        t_span = max(float(np.nanmax(t) - t_min), 1e-6)
        t_frac = ((t - t_min) / t_span).astype(np.float32)
        self.register_buffer("label_time_feat", torch.as_tensor(t_frac).reshape(1, 1, -1, 1), persistent=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.vehicle_proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        pieces = [pooled]
        if self.context_net is not None:
            pieces.append(self.context_net(context))
        global_repr = torch.cat(pieces, dim=1)
        logits = self.logit_head(global_repr)
        batch = x.shape[0]
        t_feat = self.time_net(self.label_time_feat.to(x.device).expand(batch, self.k, -1, -1))
        g = global_repr[:, None, None, :].expand(-1, self.k, t_feat.shape[2], -1)
        b = self.branch_embed[None, :, None, :].to(x.device).expand(batch, -1, t_feat.shape[2], -1)
        traj = self.traj_head(torch.cat([g, t_feat, b], dim=3)).squeeze(-1)
        traj = traj - traj[:, :, :1]
        return {"traj": traj, "logits": logits}


def branch_losses(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float().unsqueeze(1)
    target_k = target.unsqueeze(1)
    denom = mask_f.sum(dim=2).clamp_min(1.0)
    mse = (((pred - target_k) ** 2) * mask_f).sum(dim=2) / denom
    return mse


def topk_loss(out: dict[str, torch.Tensor], target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    traj = out["traj"]
    logits = out["logits"]
    losses = branch_losses(traj, target, mask)
    best_idx = torch.argmin(losses.detach(), dim=1)
    min_loss = losses.gather(1, best_idx[:, None]).mean()
    ce = nn.functional.cross_entropy(logits, best_idx)
    probs = torch.softmax(logits, dim=1)
    expected = (losses * probs).sum(dim=1).mean()
    if traj.shape[2] > 1:
        dmask = (mask[:, 1:] & mask[:, :-1]).float().unsqueeze(1)
        ddenom = dmask.sum(dim=2).clamp_min(1.0)
        dp = traj[:, :, 1:] - traj[:, :, :-1]
        dt = (target[:, 1:] - target[:, :-1]).unsqueeze(1)
        dloss = (((dp - dt) ** 2) * dmask).sum(dim=2) / ddenom
        smooth = dloss.gather(1, best_idx[:, None]).mean()
    else:
        smooth = torch.tensor(0.0, device=traj.device)
    diversity_penalty = torch.tensor(0.0, device=traj.device)
    if traj.shape[1] > 1:
        pair_terms = []
        for i in range(traj.shape[1]):
            for j in range(i + 1, traj.shape[1]):
                dist = torch.sqrt(torch.mean((traj[:, i] - traj[:, j]) ** 2, dim=1) + 1e-6)
                pair_terms.append(torch.exp(-dist).mean())
        diversity_penalty = torch.stack(pair_terms).mean()
    return min_loss + 0.12 * ce + 0.05 * expected + 0.06 * smooth + 0.008 * diversity_penalty


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return float("nan")
    diff = y_pred[valid] - y_true[valid]
    return float(np.sqrt(np.mean(diff * diff)))


def split_indices(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = meta[SPLIT_STRATEGY].astype(str).to_numpy()
    return np.where(split == "train")[0], np.where(split == "val")[0], np.where(split == "test")[0]


def load_track() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    manifest = pd.read_csv(keypoint_v01.TASK_MANIFEST_PATH)
    cfg = keypoint_v01.TRACKS[TRACK_ID]
    return keypoint_v01.load_track(TRACK_ID, cfg, manifest)


def predict_all(model: nn.Module, x: np.ndarray, context: np.ndarray, label_scale: float, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    trajs: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
            cb = torch.from_numpy(context[start : start + batch_size].astype(np.float32)).to(device)
            out = model(xb, cb)
            trajs.append((out["traj"].cpu().numpy() * float(label_scale)).astype(np.float32))
            logits.append(out["logits"].cpu().numpy().astype(np.float32))
    return np.concatenate(trajs, axis=0), np.concatenate(logits, axis=0)


def select_top1_and_bestk(
    trajs: np.ndarray,
    logits: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    top1_idx = np.argmax(logits, axis=1).astype(int)
    top1 = trajs[np.arange(len(trajs)), top1_idx]
    branch_rmse = np.zeros((len(trajs), trajs.shape[1]), dtype=np.float32)
    for k in range(trajs.shape[1]):
        valid = y_mask & np.isfinite(y) & np.isfinite(trajs[:, k, :])
        diff = np.where(valid, trajs[:, k, :] - y, np.nan)
        denom = np.maximum(valid.sum(axis=1), 1)
        branch_rmse[:, k] = np.sqrt(np.nansum(diff * diff, axis=1) / denom).astype(np.float32)
    best_idx = np.argmin(branch_rmse, axis=1).astype(int)
    best = trajs[np.arange(len(trajs)), best_idx]
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return top1, best, top1_idx, best_idx, probs


def train_model(
    x: np.ndarray,
    context: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    label_scale: float,
) -> tuple[TopKVehicleTransformer, pd.DataFrame, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TopKVehicleTransformer(vehicle_dim=x.shape[2], context_dim=context.shape[1], label_time=label_time, k=K).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_ds = VehicleDataset(x, context, y, y_mask, train_idx, label_scale)
    val_ds = VehicleDataset(x, context, y, y_mask, val_idx, label_scale)
    train_loader = DataLoader(train_ds, batch_size=min(BATCH_SIZE, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(BATCH_SIZE, max(1, len(val_ds))), shuffle=False)
    best_state = None
    best_val_top1 = float("inf")
    best_epoch = 0
    bad_epochs = 0
    rows: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, cb, yb, mb in train_loader:
            xb, cb, yb, mb = xb.to(device), cb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb, cb)
            loss = topk_loss(out, yb, mb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_loss_num = 0.0
        val_denom = 0.0
        all_top1 = []
        all_best = []
        all_y = []
        all_m = []
        with torch.no_grad():
            for xb, cb, yb, mb in val_loader:
                xb, cb, yb, mb = xb.to(device), cb.to(device), yb.to(device), mb.to(device)
                out = model(xb, cb)
                losses = branch_losses(out["traj"], yb, mb)
                top_idx = torch.argmax(out["logits"], dim=1)
                best_idx = torch.argmin(losses, dim=1)
                all_top1.append(out["traj"][torch.arange(out["traj"].shape[0], device=device), top_idx].cpu().numpy())
                all_best.append(out["traj"][torch.arange(out["traj"].shape[0], device=device), best_idx].cpu().numpy())
                all_y.append(yb.cpu().numpy())
                all_m.append(mb.cpu().numpy())
                val_loss_num += float(losses.gather(1, top_idx[:, None]).sum().cpu())
                val_denom += float(out["traj"].shape[0])
        val_top1 = rmse_np(np.concatenate(all_y), np.concatenate(all_top1), np.concatenate(all_m))
        val_best = rmse_np(np.concatenate(all_y), np.concatenate(all_best), np.concatenate(all_m))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "val_top1_rmse_norm": val_top1,
            "val_bestk_rmse_norm": val_best,
            "val_top1_loss_mean": val_loss_num / max(val_denom, 1.0),
        }
        rows.append(row)
        if val_top1 + 1e-7 < best_val_top1:
            best_val_top1 = val_top1
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
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_val_top1_rmse_norm": float(best_val_top1),
        "epochs_ran": int(len(rows)),
        "early_stop_patience": PATIENCE,
    }
    return model, pd.DataFrame(rows), info


def evaluate_predictions(
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_peak = np.nanmax(np.abs(np.where(y_mask, y, np.nan)), axis=1)
    large_thr = float(np.nanpercentile(gt_peak[train_idx], 75))
    difficult_thr = float(np.nanpercentile(gt_peak[train_idx], 80))
    rows: list[dict[str, Any]] = []
    split_values = meta[SPLIT_STRATEGY].astype(str).to_numpy()
    window_id = keypoint_v01.TRACKS[TRACK_ID]["window_config_id"]
    for split_name in ["train", "val", "test"]:
        mask = split_values == split_name
        if not mask.any():
            continue
        split_meta = meta.loc[mask].reset_index(drop=True)
        for model_name, pred in predictions.items():
            rows.extend(
                eval_utils.sample_metric_rows(
                    y[mask],
                    pred[mask],
                    y_mask[mask],
                    label_time,
                    split_meta,
                    model_name,
                    SPLIT_STRATEGY,
                    split_name,
                    window_id,
                    large_thr,
                    difficult_thr,
                )
            )
    per_sample = pd.DataFrame(rows)
    per_sample["track_id"] = TRACK_ID
    metrics = eval_utils.aggregate_metrics(per_sample)
    metrics["track_id"] = TRACK_ID
    return metrics, per_sample


def branch_diagnostics(
    meta: pd.DataFrame,
    logits: np.ndarray,
    top1_idx: np.ndarray,
    best_idx: np.ndarray,
    probs: np.ndarray,
    trajs: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
) -> pd.DataFrame:
    branch_rmse = []
    for k in range(K):
        valid = y_mask & np.isfinite(y) & np.isfinite(trajs[:, k, :])
        diff = np.where(valid, trajs[:, k, :] - y, np.nan)
        denom = np.maximum(valid.sum(axis=1), 1)
        branch_rmse.append(np.sqrt(np.nansum(diff * diff, axis=1) / denom).astype(np.float32))
    rmse_arr = np.stack(branch_rmse, axis=1)
    spread = np.nanstd(trajs, axis=1)
    return pd.DataFrame(
        {
            "sample_id": meta["sample_id"].astype(str).to_numpy(),
            "event_uid": meta["event_uid"].astype(str).to_numpy(),
            "subject": meta["subject"].astype(str).to_numpy(),
            "session_stamp": meta["session_stamp"].astype(str).to_numpy(),
            "split": meta[SPLIT_STRATEGY].astype(str).to_numpy(),
            "top1_branch": top1_idx,
            "best_branch_oracle": best_idx,
            "top1_matches_best": (top1_idx == best_idx).astype(int),
            "top1_prob": np.max(probs, axis=1),
            "prob_margin": np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2],
            "branch_spread_mean": np.nanmean(spread, axis=1),
            "branch_spread_peak": np.nanmax(spread, axis=1),
            "rmse_branch0": rmse_arr[:, 0],
            "rmse_branch1": rmse_arr[:, 1],
            "rmse_branch2": rmse_arr[:, 2],
            "rmse_bestk": np.min(rmse_arr, axis=1),
            "rmse_top1_branch": rmse_arr[np.arange(len(rmse_arr)), top1_idx],
        }
    )


def calibration_table(diag: pd.DataFrame, per_sample: pd.DataFrame) -> pd.DataFrame:
    top1_rmse = per_sample[per_sample["model_name"] == TOP1_MODEL][["sample_id", "sample_rmse"]].rename(columns={"sample_rmse": "top1_sample_rmse"})
    best_rmse = per_sample[per_sample["model_name"] == BESTK_MODEL][["sample_id", "sample_rmse"]].rename(columns={"sample_rmse": "bestk_sample_rmse"})
    df = diag.merge(top1_rmse, on="sample_id", how="left").merge(best_rmse, on="sample_id", how="left")
    rows: list[dict[str, Any]] = []
    for split, grp in df.groupby("split"):
        qs = np.nanquantile(grp["top1_prob"], [0.0, 0.33, 0.66, 1.0])
        qs[0] -= 1e-6
        qs[-1] += 1e-6
        for i in range(3):
            part = grp[(grp["top1_prob"] > qs[i]) & (grp["top1_prob"] <= qs[i + 1])]
            if part.empty:
                continue
            rows.append(
                {
                    "split": split,
                    "confidence_bin": f"{qs[i]:.3f}-{qs[i + 1]:.3f}",
                    "n_samples": int(len(part)),
                    "mean_top1_prob": float(part["top1_prob"].mean()),
                    "top1_matches_best_rate": float(part["top1_matches_best"].mean()),
                    "mean_top1_rmse": float(part["top1_sample_rmse"].mean()),
                    "mean_bestk_rmse": float(part["bestk_sample_rmse"].mean()),
                    "mean_oracle_gap": float((part["top1_sample_rmse"] - part["bestk_sample_rmse"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def sample_indices(meta: pd.DataFrame, sample_ids: list[str]) -> list[int]:
    lookup = {sid: i for i, sid in enumerate(meta["sample_id"].astype(str).tolist())}
    return [lookup[sid] for sid in sample_ids if sid in lookup]


def plot_prediction_grid(
    path: Path,
    sample_ids: list[str],
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    title: str,
) -> None:
    idxs = sample_indices(meta, sample_ids)[:12]
    if not idxs:
        return
    ncols = 3
    nrows = int(math.ceil(len(idxs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, max(3.0, 2.55 * nrows)), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr[len(idxs) :]:
        ax.axis("off")
    for ax, idx in zip(axes_arr, idxs):
        sid = str(meta.iloc[idx]["sample_id"])
        valid = y_mask[idx] & np.isfinite(y[idx])
        ax.plot(label_time[valid], y[idx, valid], color="#111111", linewidth=1.8, label="gt")
        for model_name, color, label in PLOT_MODELS:
            pred = predictions[model_name][idx]
            style = "--" if model_name == BESTK_MODEL else "-"
            width = 1.3 if model_name in [TOP1_MODEL, BESTK_MODEL] else 1.0
            ax.plot(label_time[valid], pred[valid], color=color, linestyle=style, linewidth=width, alpha=0.86, label=label)
        short_id = sid.split("__")[-2] if "__" in sid else sid[-10:]
        ax.set_title(short_id, fontsize=8)
        ax.grid(True, alpha=0.22)
    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metric_summary(path: Path, metrics: pd.DataFrame) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    order = [RBF_MODEL, TOP1_MODEL, BESTK_MODEL]
    test = test[test["model_name"].isin(order)]
    x = np.arange(len(test))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    axes[0].bar(x, test["rmse_steer"], color=["#1f77b4", "#d62728", "#2ca02c"])
    axes[0].set_title("RMSE")
    axes[1].bar(x, test["wrong_side_rate"], color=["#1f77b4", "#d62728", "#2ca02c"])
    axes[1].set_title("Wrong side")
    axes[2].bar(x, test["large_response_recall"], color=["#1f77b4", "#d62728", "#2ca02c"])
    axes[2].set_title("Large recall")
    for ax in axes:
        ax.set_xticks(x, ["rbf", "top1", "bestK"], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_reliability(path: Path, diag: pd.DataFrame, per_sample: pd.DataFrame) -> None:
    top1 = per_sample[per_sample["model_name"] == TOP1_MODEL][["sample_id", "sample_rmse", "split"]]
    df = diag.merge(top1[["sample_id", "sample_rmse"]], on="sample_id", how="left")
    df = df[df["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    sc = ax.scatter(df["top1_prob"], df["sample_rmse"], c=df["branch_spread_mean"], cmap="viridis", s=48, alpha=0.85)
    ax.set_xlabel("Top-1 branch probability")
    ax.set_ylabel("Top-1 sample RMSE")
    ax.set_title("Reliability diagnostic on test")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="branch spread mean")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_reports(metrics: pd.DataFrame, branch_diag: pd.DataFrame, calibration: pd.DataFrame, figures: dict[str, str]) -> None:
    test = metrics[metrics["split"] == "test"].set_index("model_name")

    def val(model: str, col: str) -> float:
        return float(test.loc[model, col])

    test_diag = branch_diag[branch_diag["split"] == "test"]
    user = f"""# 阶段 3 用户查看版：top-K 车辆-only Transformer v0.1

## 这个阶段为什么做

上一轮 RBF/keypoint 复盘显示两个候选有 oracle 互补空间，但 selector 还不能稳定选择。这个阶段开始做真正的车辆-only 多假设模型：模型一次输出 3 条可能轨迹，并给每条轨迹一个选择概率。

## 这个阶段检查了什么

- top-1：模型自己按概率选出的轨迹。
- best-of-3：事后从 3 条轨迹里选最接近真实的一条，只作为上限诊断。
- RBF 强车辆基线：仍作为主参照。
- 可靠性：top-1 概率、分支分散度和误差是否有关系。

## 目前发现了什么

- RBF：RMSE={val(RBF_MODEL, 'rmse_steer'):.6f}，错侧率={val(RBF_MODEL, 'wrong_side_rate'):.3f}，大幅响应召回={val(RBF_MODEL, 'large_response_recall'):.3f}。
- top-1：RMSE={val(TOP1_MODEL, 'rmse_steer'):.6f}，错侧率={val(TOP1_MODEL, 'wrong_side_rate'):.3f}，大幅响应召回={val(TOP1_MODEL, 'large_response_recall'):.3f}。
- best-of-3：RMSE={val(BESTK_MODEL, 'rmse_steer'):.6f}，错侧率={val(BESTK_MODEL, 'wrong_side_rate'):.3f}，大幅响应召回={val(BESTK_MODEL, 'large_response_recall'):.3f}。
- test 上 top-1 分支与 best-of-3 分支一致率={float(test_diag['top1_matches_best'].mean()):.3f}，平均 top-1 概率={float(test_diag['top1_prob'].mean()):.3f}。

## 哪些结果可信

可信的是：这是一个真正车辆-only 的 top-K 模型，输入只含事件前车辆历史和道路/事件上下文；top-1 是可部署策略，best-of-3 只是上限。所有标准化和训练选择都只用 train/val。

## 哪些结果还不能下结论

best-of-3 不能当成可部署结果。若 top-1 没有超过 RBF，就不能说 top-K 车辆-only 已经成为主线；若 best-of-3 明显好但 top-1 不好，只能说明“候选覆盖有潜力，但选择机制还不够”。

## 下一阶段是否可以继续

可以继续阶段 3，但仍不能进入风格、生理或 EEG 增量结论。下一步应根据本轮 top-1/best-of-3 差距，决定是改 selector/可靠性头，还是换成关键点条件的多假设结构。

## 推荐优先查看

1. `{figures['fixed']}`
2. `{figures['bad']}`
3. `{figures['metric_summary']}`
4. `{figures['reliability']}`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_metrics.csv`
"""

    tech = f"""# 阶段 3 技术报告：top-K 车辆-only Transformer v0.1

## 范围

- 轨道：`{TRACK_ID}`。
- 输入：事件前车辆时序 + 因果可得道路/事件上下文。
- 输出：K={K} 条轨迹 + 分支 logits。
- 训练：min-of-K 轨迹损失 + 分支选择交叉熵 + 平滑项 + 轻量多样性项。
- checkpoint 选择：validation top-1 RMSE。
- 未使用：subject ID、生理、脑电、连续风格、服务器、服务器密码文件。

## test 指标

| 模型 | RMSE | 错侧率 | 大幅召回 | 困难 top20 RMSE |
|---|---:|---:|---:|---:|
| RBF | {val(RBF_MODEL, 'rmse_steer'):.6f} | {val(RBF_MODEL, 'wrong_side_rate'):.3f} | {val(RBF_MODEL, 'large_response_recall'):.3f} | {val(RBF_MODEL, 'difficult_top20_rmse'):.6f} |
| top-1 | {val(TOP1_MODEL, 'rmse_steer'):.6f} | {val(TOP1_MODEL, 'wrong_side_rate'):.3f} | {val(TOP1_MODEL, 'large_response_recall'):.3f} | {val(TOP1_MODEL, 'difficult_top20_rmse'):.6f} |
| best-of-3 | {val(BESTK_MODEL, 'rmse_steer'):.6f} | {val(BESTK_MODEL, 'wrong_side_rate'):.3f} | {val(BESTK_MODEL, 'large_response_recall'):.3f} | {val(BESTK_MODEL, 'difficult_top20_rmse'):.6f} |

## 可靠性诊断

- test top1_matches_best_rate={float(test_diag['top1_matches_best'].mean()):.6f}
- test mean_top1_prob={float(test_diag['top1_prob'].mean()):.6f}
- test mean_prob_margin={float(test_diag['prob_margin'].mean()):.6f}
- test mean_branch_spread={float(test_diag['branch_spread_mean'].mean()):.6f}

## 结论

本轮用于判断真正 top-K 车辆-only 是否比 RBF/keypoint 事后二选一更适合继续。是否升级主线必须以 top-1 指标、固定图和坏样本图为准；best-of-3 只能作为候选覆盖上限。
"""
    (REPORT_ROOT / "stage03_vehicle_instability_topk_vehicle_transformer_user_summary_cn.md").write_text(user, encoding="utf-8")
    (REPORT_ROOT / "stage03_vehicle_instability_topk_vehicle_transformer_v0_1_cn.md").write_text(tech, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    set_seed(SEED)
    y, y_mask, input_values, input_mask, input_time, label_time, meta = load_track()
    train_idx, val_idx, test_idx = split_indices(meta)
    x_scaled, scaler_info = keypoint_v01.standardize_vehicle_inputs(input_values, input_mask, train_idx)
    context, context_names = keypoint_v01.build_context_features(meta, train_idx)
    step = max(1, int(round(len(input_time) / TARGET_INPUT_TOKENS)))
    x_model = x_scaled[:, ::step, :].copy()
    label_scale = keypoint_v01.label_scale_train(y, y_mask, train_idx)

    cfg = keypoint_v01.TRACKS[TRACK_ID]
    baseline_predictions, _ = clean_v01.build_strong_predictions(
        TRACK_ID,
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
    if RBF_MODEL not in baseline_predictions:
        raise RuntimeError(f"{RBF_MODEL} not rebuilt")

    model, history, train_info = train_model(x_model, context, y, y_mask, label_time, train_idx, val_idx, label_scale)
    trajs, logits = predict_all(model, x_model, context, label_scale)
    top1, bestk, top1_idx, best_idx, probs = select_top1_and_bestk(trajs, logits, y, y_mask)

    predictions = {RBF_MODEL: baseline_predictions[RBF_MODEL], TOP1_MODEL: top1, BESTK_MODEL: bestk}
    for k in range(K):
        predictions[f"{BRANCH_PREFIX}{k}_no_subject"] = trajs[:, k, :]
    metrics, per_sample = evaluate_predictions(y, y_mask, label_time, meta, train_idx, predictions)
    branch_diag = branch_diagnostics(meta, logits, top1_idx, best_idx, probs, trajs, y, y_mask)
    calib = calibration_table(branch_diag, per_sample)

    reference = pd.read_csv(REFERENCE_METRICS) if REFERENCE_METRICS.exists() else pd.DataFrame()
    if not reference.empty:
        keep_models = [
            "keypoint_residual_vehicle_transformer_no_subject",
            "selector_logreg_rbf_keypoint_no_subject",
            "oracle_best_of_rbf_keypoint_upper_bound",
        ]
        reference = reference[(reference["split"] == "test") & (reference["model_name"].isin(keep_models))].copy()
        reference["source"] = "rbf_keypoint_multihypothesis_review_v0_1"
    comparison = metrics[metrics["split"] == "test"].copy()
    comparison["source"] = OUTPUT_VERSION
    comparison = pd.concat([comparison, reference], ignore_index=True, sort=False) if not reference.empty else comparison

    metrics.to_csv(TABLE_DIR / "topk_vehicle_transformer_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(TABLE_DIR / "topk_vehicle_transformer_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    branch_diag.to_csv(TABLE_DIR / "topk_vehicle_transformer_branch_diagnostics.csv", index=False, encoding="utf-8-sig")
    calib.to_csv(TABLE_DIR / "topk_vehicle_transformer_reliability_bins.csv", index=False, encoding="utf-8-sig")
    history.to_csv(TABLE_DIR / "topk_vehicle_transformer_training_history.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(TABLE_DIR / "topk_vehicle_transformer_comparison_with_references.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"track_id": TRACK_ID, "context_feature": context_names}).to_csv(TABLE_DIR / "topk_vehicle_transformer_context_features.csv", index=False, encoding="utf-8-sig")

    test_top1 = per_sample[(per_sample["split"] == "test") & (per_sample["model_name"] == TOP1_MODEL)].copy()
    fixed_ids = meta.loc[test_idx, "sample_id"].astype(str).head(12).tolist()
    bad_ids = test_top1.sort_values("sample_rmse", ascending=False).head(12)["sample_id"].astype(str).tolist()
    gap_diag = branch_diag[branch_diag["split"] == "test"].copy()
    gap_diag["top1_minus_bestk"] = gap_diag["rmse_top1_branch"] - gap_diag["rmse_bestk"]
    gap_ids = gap_diag.sort_values("top1_minus_bestk", ascending=False).head(12)["sample_id"].astype(str).tolist()
    pd.DataFrame({"track_id": TRACK_ID, "sample_id": fixed_ids}).to_csv(TABLE_DIR / "topk_fixed_plot_samples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"track_id": TRACK_ID, "sample_id": bad_ids}).to_csv(TABLE_DIR / "topk_bad_plot_samples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"track_id": TRACK_ID, "sample_id": gap_ids}).to_csv(TABLE_DIR / "topk_gap_plot_samples.csv", index=False, encoding="utf-8-sig")

    fixed_fig = FIG_DIR / "topk_fixed_predictions_test.png"
    bad_fig = FIG_DIR / "topk_bad_samples_test.png"
    gap_fig = FIG_DIR / "topk_top1_bestk_gap_samples_test.png"
    metric_fig = FIG_DIR / "topk_metric_summary_test.png"
    reliability_fig = FIG_DIR / "topk_reliability_scatter_test.png"
    plot_prediction_grid(fixed_fig, fixed_ids, y, y_mask, label_time, meta, predictions, "Top-K fixed test samples")
    plot_prediction_grid(bad_fig, bad_ids, y, y_mask, label_time, meta, predictions, "Top-K worst top-1 test samples")
    plot_prediction_grid(gap_fig, gap_ids, y, y_mask, label_time, meta, predictions, "Top-K largest top1/bestK gaps")
    plot_metric_summary(metric_fig, metrics)
    plot_reliability(reliability_fig, branch_diag, per_sample)
    figures = {
        "fixed": str(fixed_fig).replace("\\", "/"),
        "bad": str(bad_fig).replace("\\", "/"),
        "gap": str(gap_fig).replace("\\", "/"),
        "metric_summary": str(metric_fig).replace("\\", "/"),
        "reliability": str(reliability_fig).replace("\\", "/"),
    }

    ckpt_path = CHECKPOINT_DIR / f"{TRACK_ID}_{TOP1_MODEL}_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "track_id": TRACK_ID,
            "model_name": TOP1_MODEL,
            "window_config_id": cfg["window_config_id"],
            "split_strategy": SPLIT_STRATEGY,
            "k": K,
            "label_scale": label_scale,
            "context_names": context_names,
            "input_downsample_step": step,
            "scaler_info": scaler_info,
            "train_info": train_info,
        },
        ckpt_path,
    )
    info = {
        **train_info,
        "output_version": OUTPUT_VERSION,
        "track_id": TRACK_ID,
        "model_name": TOP1_MODEL,
        "bestk_model_name": BESTK_MODEL,
        "window_config_id": cfg["window_config_id"],
        "split_strategy": SPLIT_STRATEGY,
        "k": K,
        "seed": SEED,
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "test_n": int(len(test_idx)),
        "label_scale_train_std": float(label_scale),
        "context_feature_count": int(context.shape[1]),
        "vehicle_feature_count": int(input_values.shape[2]),
        "vehicle_input_tokens": int(x_model.shape[1]),
        "vehicle_input_downsample_step": int(step),
        "uses_subject_id": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "uses_future_labels_as_input": False,
        "checkpoint_selection": "validation top1 RMSE",
        "best_of_k_is_oracle_only": True,
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "checkpoint_path": str(ckpt_path).replace("\\", "/"),
    }
    pd.DataFrame([info]).to_csv(TABLE_DIR / "topk_vehicle_transformer_model_info.csv", index=False, encoding="utf-8-sig")
    write_reports(metrics, branch_diag, calib, figures)
    summary = {
        "output_version": OUTPUT_VERSION,
        "track_id": TRACK_ID,
        "model_name": TOP1_MODEL,
        "bestk_model_name": BESTK_MODEL,
        "k": K,
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "metrics_path": str(TABLE_DIR / "topk_vehicle_transformer_metrics.csv").replace("\\", "/"),
        "branch_diagnostics_path": str(TABLE_DIR / "topk_vehicle_transformer_branch_diagnostics.csv").replace("\\", "/"),
        "figures": figures,
    }
    (LOG_DIR / "topk_vehicle_transformer_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
