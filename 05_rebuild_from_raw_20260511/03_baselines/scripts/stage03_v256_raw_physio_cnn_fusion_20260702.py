#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v256 raw 200Hz 生理时序编码与车辆融合预测基线。

本轮目的：
- 不再把生理压成 v254b 那类手工窗口统计；
- 直接把锚点前 20s 生理片段下采样为 20Hz 序列，用 1D CNN 学状态表示；
- 与同一套车辆 MLP baseline 对比，判断 raw 生理时序是否带来轨迹预测增量。

边界：
- 每个样本只用 observation_s 之前的生理；
- baseline 归一化窗口为 observation_s-60s 到 observation_s-20s；
- subject-disjoint 是正式口径，subject-aware 只作为个体化潜力诊断。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V252_SCRIPT = BASELINES / "scripts" / "stage03_v252_input_similarity_future_divergence_20260701.py"
V254B_SCRIPT = BASELINES / "scripts" / "stage03_v254b_physio_200hz_event_representation_20260702.py"

OUT = BASELINES / "v256_raw_physio_cnn_fusion_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
TENSORS = OUT / "tensors"
ZIP_PATH = BASELINES / "v256_raw_physio_cnn_fusion_20260702_pack.zip"

SEED = 25602
SEQ_HZ = 20
SEQ_SECONDS = 20
SEQ_LEN = SEQ_HZ * SEQ_SECONDS
BASELINE_WINDOW = (-60.0, -20.0)
EPOCHS = 70
PATIENCE = 10
BATCH_SIZE = 256

RAW_PHYSIO_SIGNALS = [
    "HR_bpm",
    "EMG_RMS",
    "EDA_Tonic",
    "EDA_Phasic",
    "RESP_filt200",
    "ECG_filt200",
]

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V252 = import_module_from_path("stage03_v252_for_v256", V252_SCRIPT)
V254B = import_module_from_path("stage03_v254b_for_v256", V254B_SCRIPT)


def ensure_dirs() -> None:
    for folder in (TABLES, FIGURES, REPORTS, LOGS, TENSORS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finite(values: Iterable[object]) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def robust_scale(values: np.ndarray) -> float:
    vals = finite(values)
    if vals.size < 5:
        return 1.0
    q25, q75 = np.quantile(vals, [0.25, 0.75])
    iqr = float(q75 - q25)
    mad = float(np.median(np.abs(vals - np.median(vals))))
    std = float(np.std(vals))
    for scale in [iqr / 1.349 if iqr > 0 else math.nan, mad * 1.4826 if mad > 0 else math.nan, std]:
        if np.isfinite(scale) and scale > 1e-6:
            return float(scale)
    return 1.0


def standardize_by_train(x: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    x = np.asarray(x, dtype=float)
    train_x = x[train_mask]
    med = np.nanmedian(train_x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isfinite(x), x, med[None, :])
    mean = np.nanmean(filled[train_mask], axis=0)
    std = np.nanstd(filled[train_mask], axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    z = (filled - mean[None, :]) / std[None, :]
    audit = pd.DataFrame({"feature_i": np.arange(x.shape[1]), "train_mean": mean, "train_std": std})
    return z.astype(np.float32), audit


def interp_normalized_signal(times: np.ndarray, vals: np.ndarray, grid: np.ndarray, baseline_vals: np.ndarray) -> np.ndarray:
    base_good = finite(baseline_vals)
    if base_good.size:
        base_median = float(np.median(base_good))
        base_scale = robust_scale(base_good)
    else:
        good_all = finite(vals)
        base_median = float(np.median(good_all)) if good_all.size else 0.0
        base_scale = robust_scale(good_all) if good_all.size else 1.0
    mask = np.isfinite(times) & np.isfinite(vals)
    if int(mask.sum()) < 2:
        return np.zeros(len(grid), dtype=np.float32)
    seg = np.interp(grid, times[mask], vals[mask], left=base_median, right=base_median)
    z = (seg - base_median) / max(base_scale, 1e-6)
    z = np.clip(z, -8.0, 8.0)
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def build_or_load_physio_sequence(manifest: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    构造每个 sample 的 raw 生理序列缓存。
    输出形状为 [n_sample, n_channel, 400]，每个通道为样本内 baseline z-score。
    """

    cache_path = TENSORS / "v256_physio_seq_20s_20hz.npz"
    if cache_path.exists():
        cache = np.load(cache_path)
        seq = cache["physio_seq"].astype(np.float32)
        ok = cache["physio_ok"].astype(np.float32)
        audit = pd.read_csv(TABLES / "v256_physio_sequence_alignment_audit.csv", encoding="utf-8-sig")
        return seq, ok, audit

    inventory = V254B.load_physio_inventory()
    samples = manifest[["event_uid", "subject", "recording", "split", "delay_ms", "observation_s"]].reset_index(names="row_index").copy()
    samples["session_stamp"] = samples["recording"].map(V254B.session_stamp_from_recording)
    n = len(samples)
    seq = np.zeros((n, len(RAW_PHYSIO_SIGNALS), SEQ_LEN), dtype=np.float32)
    ok = np.zeros(n, dtype=np.float32)
    audit_rows: List[Dict[str, object]] = []

    grouped = samples.groupby(["subject", "session_stamp"], sort=False)
    n_groups = len(grouped)
    for group_i, ((subject, session), g) in enumerate(grouped, start=1):
        path = inventory.get((str(subject), str(session)))
        if path is None or not path.exists():
            print(f"[v256] missing physio recording {group_i}/{n_groups}: subject={subject} session={session} samples={len(g)}", flush=True)
            for _, row in g.iterrows():
                audit_rows.append(
                    {
                        "row_index": int(row["row_index"]),
                        "subject": str(subject),
                        "session_stamp": str(session),
                        "status": "missing_recording",
                        "baseline_rows": 0,
                        "segment_points": 0,
                    }
                )
            continue

        print(f"[v256] sequence extracting {group_i}/{n_groups}: subject={subject} session={session} samples={len(g)}", flush=True)
        rec = V254B.read_physio_recording(path)
        times = pd.to_numeric(rec["t_s"], errors="coerce").to_numpy(dtype=float)
        signal_arrays: Dict[str, np.ndarray] = {}
        for sig in RAW_PHYSIO_SIGNALS:
            if sig in rec.columns:
                signal_arrays[sig] = pd.to_numeric(rec[sig], errors="coerce").to_numpy(dtype=float)
            else:
                signal_arrays[sig] = np.full(len(times), np.nan, dtype=float)

        for _, sample in g.iterrows():
            row_i = int(sample["row_index"])
            obs = float(sample["observation_s"])
            grid = obs - SEQ_SECONDS + (np.arange(SEQ_LEN, dtype=float) + 0.5) / SEQ_HZ
            grid = np.maximum(grid, 0.0)
            b_start = max(0.0, obs + BASELINE_WINDOW[0])
            b_end = max(0.0, obs + BASELINE_WINDOW[1])
            b_left = int(np.searchsorted(times, b_start, side="left"))
            b_right = int(np.searchsorted(times, b_end, side="right"))
            seg_left = int(np.searchsorted(times, max(0.0, obs - SEQ_SECONDS), side="left"))
            seg_right = int(np.searchsorted(times, obs, side="right"))
            for c_i, sig in enumerate(RAW_PHYSIO_SIGNALS):
                vals = signal_arrays[sig]
                seq[row_i, c_i, :] = interp_normalized_signal(times, vals, grid, vals[b_left:b_right])
            ok[row_i] = 1.0
            audit_rows.append(
                {
                    "row_index": row_i,
                    "subject": str(subject),
                    "session_stamp": str(session),
                    "status": "ok",
                    "baseline_rows": int(max(0, b_right - b_left)),
                    "segment_points": int(max(0, seg_right - seg_left)),
                }
            )

    audit = pd.DataFrame(audit_rows).sort_values("row_index").reset_index(drop=True)
    write_csv(audit, TABLES / "v256_physio_sequence_alignment_audit.csv")
    np.savez_compressed(cache_path, physio_seq=seq, physio_ok=ok, signals=np.array(RAW_PHYSIO_SIGNALS))
    return seq, ok, audit


class VehicleEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, emb_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysioEncoder(nn.Module):
    def __init__(self, in_channels: int, emb_dim: int = 96):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(96, emb_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class TrajectoryModel(nn.Module):
    def __init__(self, kind: str, vehicle_dim: int, physio_channels: int, out_dim: int):
        super().__init__()
        self.kind = kind
        self.vehicle = VehicleEncoder(vehicle_dim, 128)
        self.physio = PhysioEncoder(physio_channels, 96)
        if kind == "vehicle_only":
            head_in = 128
        elif kind == "physio_cnn":
            head_in = 96 + 1
        elif kind == "vehicle_physio_cnn":
            head_in = 128 + 96 + 1
        else:
            raise ValueError(f"unknown model kind: {kind}")
        self.head = nn.Sequential(
            nn.Linear(head_in, 160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, out_dim),
        )

    def forward(self, vehicle_x: torch.Tensor, physio_x: torch.Tensor, physio_ok: torch.Tensor) -> torch.Tensor:
        if self.kind == "vehicle_only":
            emb = self.vehicle(vehicle_x)
        elif self.kind == "physio_cnn":
            emb = torch.cat([self.physio(physio_x), physio_ok[:, None]], dim=1)
        else:
            emb = torch.cat([self.vehicle(vehicle_x), self.physio(physio_x), physio_ok[:, None]], dim=1)
        return self.head(emb)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = torch.square(pred - target) * mask
    den = torch.clamp(mask.sum(), min=1.0)
    return diff2.sum() / den


def make_loader(
    indices: np.ndarray,
    vehicle_x: np.ndarray,
    physio_seq: np.ndarray,
    physio_ok: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(vehicle_x[indices].astype(np.float32)),
        torch.from_numpy(physio_seq[indices].astype(np.float32)),
        torch.from_numpy(physio_ok[indices].astype(np.float32)),
        torch.from_numpy(y[indices].astype(np.float32)),
        torch.from_numpy(valid_mask[indices].astype(np.float32)),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray]:
    model.eval()
    losses = []
    preds = []
    for vehicle_x, physio_x, physio_ok, y, mask in loader:
        vehicle_x = vehicle_x.to(device, non_blocking=True)
        physio_x = physio_x.to(device, non_blocking=True)
        physio_ok = physio_ok.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pred = model(vehicle_x, physio_x, physio_ok)
        losses.append(float(masked_mse(pred, y, mask).detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(preds, axis=0)


def train_one_model(
    protocol: str,
    kind: str,
    split: np.ndarray,
    vehicle_x: np.ndarray,
    physio_seq: np.ndarray,
    physio_ok: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, pd.DataFrame]:
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    all_idx = np.arange(len(split))
    train_loader = make_loader(train_idx, vehicle_x, physio_seq, physio_ok, y, valid_mask, shuffle=True)
    val_loader = make_loader(val_idx, vehicle_x, physio_seq, physio_ok, y, valid_mask, shuffle=False)
    all_loader = make_loader(all_idx, vehicle_x, physio_seq, physio_ok, y, valid_mask, shuffle=False)

    set_seed(SEED + hash((protocol, kind)) % 10000)
    model = TrajectoryModel(kind, vehicle_x.shape[1], physio_seq.shape[1], y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best_state = None
    best_val = math.inf
    bad_epochs = 0
    rows = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for vehicle_b, physio_b, ok_b, y_b, mask_b in train_loader:
            vehicle_b = vehicle_b.to(device, non_blocking=True)
            physio_b = physio_b.to(device, non_blocking=True)
            ok_b = ok_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            mask_b = mask_b.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(vehicle_b, physio_b, ok_b)
            loss = masked_mse(pred, y_b, mask_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss, _ = evaluate_model(model, val_loader, device)
        rows.append(
            {
                "protocol": protocol,
                "model_name": f"v256_{kind}",
                "epoch": epoch,
                "train_mse": float(np.mean(train_losses)),
                "val_mse": val_loss,
                "val_rmse": float(math.sqrt(max(val_loss, 0.0))),
            }
        )
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"[v256] {protocol}/{kind} epoch={epoch} val_rmse={math.sqrt(max(val_loss, 0.0)):.4f}", flush=True)
        if bad_epochs >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, pred_all = evaluate_model(model, all_loader, device)
    return pred_all.astype(np.float32), pd.DataFrame(rows)


def sample_rmse(pred: np.ndarray, y: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    diff2 = np.square(pred - y)
    diff2 = np.where(valid_mask, diff2, np.nan)
    with np.errstate(all="ignore"):
        return np.sqrt(np.nanmean(diff2, axis=1))


def sample_tail_rmse(pred: np.ndarray, y: np.ndarray, valid_mask: np.ndarray, delays: np.ndarray) -> np.ndarray:
    out = np.full(len(y), np.nan, dtype=float)
    for i, delay in enumerate(delays):
        tail = V252.future_tail_mask(int(delay))
        mask = valid_mask[i] & tail
        if int(mask.sum()) < 2:
            continue
        out[i] = float(np.sqrt(np.mean(np.square(pred[i, mask] - y[i, mask]))))
    return out


def build_bad_top10_by_protocol(reference_tail: np.ndarray, split: np.ndarray) -> np.ndarray:
    bad = np.zeros(len(reference_tail), dtype=bool)
    for split_name in ["train", "val", "test"]:
        mask = split == split_name
        vals = reference_tail[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        q90 = float(np.quantile(vals, 0.90))
        bad[mask] = reference_tail[mask] >= q90
    return bad


def summarize_predictions(
    protocol: str,
    split: np.ndarray,
    manifest: pd.DataFrame,
    sample_metrics: pd.DataFrame,
    y: np.ndarray,
    valid_mask: np.ndarray,
    pred_map: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    delays = manifest["delay_ms"].astype(int).to_numpy()
    v250_tail = pd.to_numeric(sample_metrics["tail_rmse_v250"], errors="coerce").to_numpy(dtype=float)
    bad_top10 = build_bad_top10_by_protocol(v250_tail, split)
    metric_rows = []
    per_sample_rows = []

    for model_name, pred in pred_map.items():
        rmse = sample_rmse(pred, y, valid_mask)
        tail = sample_tail_rmse(pred, y, valid_mask, delays)
        for i in range(len(manifest)):
            per_sample_rows.append(
                {
                    "protocol": protocol,
                    "model_name": model_name,
                    "row_index": i,
                    "event_uid": str(manifest.iloc[i]["event_uid"]),
                    "split": str(split[i]),
                    "delay_ms": int(delays[i]),
                    "sample_rmse": float(rmse[i]),
                    "tail_rmse": float(tail[i]),
                    "bad_top10_v250_bucket": bool(bad_top10[i]),
                    "is_strong_steer": bool(sample_metrics.iloc[i]["is_strong_steer"]),
                    "is_observe_later_like": bool(sample_metrics.iloc[i]["is_observe_later_like"]),
                }
            )
        bucket_defs = [
            ("all", np.ones(len(split), dtype=bool)),
            ("bad_top10_v250", bad_top10),
            ("strong_steer", sample_metrics["is_strong_steer"].astype(bool).to_numpy()),
            ("observe_later_like", sample_metrics["is_observe_later_like"].astype(bool).to_numpy()),
        ]
        for eval_split in ["val", "test"]:
            split_mask = split == eval_split
            for bucket, bucket_mask in bucket_defs:
                mask = split_mask & bucket_mask
                if int(mask.sum()) == 0:
                    continue
                metric_rows.append(
                    {
                        "protocol": protocol,
                        "eval_split": eval_split,
                        "bucket": bucket,
                        "model_name": model_name,
                        "n": int(mask.sum()),
                        "sample_rmse_mean": float(np.nanmean(rmse[mask])),
                        "sample_rmse_median": float(np.nanmedian(rmse[mask])),
                        "tail_rmse_mean": float(np.nanmean(tail[mask])),
                        "tail_rmse_median": float(np.nanmedian(tail[mask])),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    base = metrics[metrics["model_name"].eq("v256_vehicle_only")][["protocol", "eval_split", "bucket", "sample_rmse_mean", "tail_rmse_mean"]].rename(
        columns={"sample_rmse_mean": "vehicle_sample_rmse_mean", "tail_rmse_mean": "vehicle_tail_rmse_mean"}
    )
    metrics = metrics.merge(base, on=["protocol", "eval_split", "bucket"], how="left")
    metrics["delta_sample_rmse_vs_v256_vehicle"] = metrics["sample_rmse_mean"] - metrics["vehicle_sample_rmse_mean"]
    metrics["delta_tail_rmse_vs_v256_vehicle"] = metrics["tail_rmse_mean"] - metrics["vehicle_tail_rmse_mean"]
    return metrics, pd.DataFrame(per_sample_rows)


def plot_test_buckets(metrics: pd.DataFrame) -> Path:
    path = FIGURES / "v256_test_bucket_tail_rmse.png"
    sub = metrics[
        metrics["eval_split"].eq("test")
        & metrics["bucket"].isin(["all", "bad_top10_v250", "strong_steer", "observe_later_like"])
        & metrics["model_name"].isin(["v250_existing", "v256_vehicle_only", "v256_physio_cnn", "v256_vehicle_physio_cnn"])
    ].copy()
    if sub.empty:
        return path
    protocols = list(sub["protocol"].drop_duplicates())
    buckets = ["all", "bad_top10_v250", "strong_steer", "observe_later_like"]
    models = ["v250_existing", "v256_vehicle_only", "v256_physio_cnn", "v256_vehicle_physio_cnn"]
    fig, axes = plt.subplots(len(protocols), 1, figsize=(13, 4.5 * len(protocols)), squeeze=False)
    for ax, protocol in zip(axes[:, 0], protocols):
        g = sub[sub["protocol"].eq(protocol)]
        x = np.arange(len(buckets))
        width = 0.82 / len(models)
        for j, model in enumerate(models):
            vals = []
            for bucket in buckets:
                r = g[g["bucket"].eq(bucket) & g["model_name"].eq(model)]
                vals.append(float(r["tail_rmse_mean"].iloc[0]) if len(r) else np.nan)
            ax.bar(x + (j - (len(models) - 1) / 2) * width, vals, width=width, label=model)
        ax.set_title(f"{protocol}: test tail RMSE by bucket")
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)
        ax.set_ylabel("tail RMSE")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for label, path in [("v252_script", V252_SCRIPT), ("v254b_script", V254B_SCRIPT)]:
        rows.append({"label": label, "path": str(path), "exists": path.exists(), "sha256": file_sha256(path) if path.exists() else ""})
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                # 生理序列缓存较大，报告包保留审计和预测结果，缓存可由脚本复现。
                if path.name == "v256_physio_seq_20s_20hz.npz":
                    continue
                zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip() is None


def write_report(metrics: pd.DataFrame, train_log: pd.DataFrame, align: pd.DataFrame, figures: List[Path]) -> None:
    lines: List[str] = []
    lines.append("# v256 raw 200Hz 生理 CNN 融合预测基线")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("- v254b 的 200Hz 手工统计没有带来跨驾驶员轨迹行为增量。")
    lines.append("- v256 改为直接输入锚点前 20s raw 生理序列，用 1D CNN 学时序状态，再与车辆 MLP 融合。")
    lines.append("")
    lines.append("## 输入")
    lines.append("")
    lines.append(f"- 生理通道：{', '.join(RAW_PHYSIO_SIGNALS)}。")
    lines.append(f"- 生理窗口：observation_s 前 {SEQ_SECONDS}s，下采样到 {SEQ_HZ}Hz，共 {SEQ_LEN} 步。")
    lines.append("- 每个样本用自身 observation_s-60s 到 observation_s-20s 做 baseline z-score，不使用锚点后数据。")
    lines.append("")
    lines.append("## 对齐覆盖")
    lines.append("")
    coverage = align.groupby("status").size().reset_index(name="n")
    coverage["rate"] = coverage["n"] / max(1, len(align))
    lines.append(coverage.to_markdown(index=False))
    lines.append("")
    lines.append("## Test 指标")
    lines.append("")
    focus = metrics[
        metrics["eval_split"].eq("test")
        & metrics["bucket"].isin(["all", "bad_top10_v250", "strong_steer", "observe_later_like"])
        & metrics["model_name"].isin(["v250_existing", "v256_vehicle_only", "v256_physio_cnn", "v256_vehicle_physio_cnn"])
    ].copy()
    lines.append(
        focus[
            [
                "protocol",
                "bucket",
                "model_name",
                "n",
                "sample_rmse_mean",
                "tail_rmse_mean",
                "delta_tail_rmse_vs_v256_vehicle",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    for protocol in ["subject_disjoint", "subject_aware"]:
        bad = focus[focus["protocol"].eq(protocol) & focus["bucket"].eq("bad_top10_v250")].copy()
        veh = bad[bad["model_name"].eq("v256_vehicle_only")]
        fusion = bad[bad["model_name"].eq("v256_vehicle_physio_cnn")]
        if len(veh) and len(fusion):
            lines.append(
                f"- {protocol} bad_top10：vehicle tail={float(veh['tail_rmse_mean'].iloc[0]):.4f}，"
                f"vehicle+physio tail={float(fusion['tail_rmse_mean'].iloc[0]):.4f}，"
                f"delta={float(fusion['delta_tail_rmse_vs_v256_vehicle'].iloc[0]):+.4f}。"
            )
    lines.append("- 如果 fusion 仍不优于同架构 vehicle-only，说明问题不只是 v254b 手工统计太浅；当前生理在这个任务构造下没有稳定可用增量。")
    lines.append("- 如果只在 subject-aware 改善，后续应转向个体化校准范式，而不是宣称跨驾驶员通用生理行为预测。")
    lines.append("")
    lines.append("## 训练日志摘要")
    lines.append("")
    last = train_log.sort_values("epoch").groupby(["protocol", "model_name"], as_index=False).tail(1)
    lines.append(last[["protocol", "model_name", "epoch", "val_rmse"]].to_markdown(index=False))
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v256_raw_physio_cnn_fusion_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v256] raw physio CNN fusion baseline")
    clean_out_dir()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v256] device={device}", flush=True)

    loaded = V252.load_fixed_inputs()
    manifest = loaded["data"].manifest.copy()
    y = loaded["y_true"].astype(np.float32)
    valid_mask = loaded["valid_mask"].astype(bool)
    sample_metrics = loaded["sample_metrics"].copy()
    split_disjoint = manifest["split"].astype(str).to_numpy()
    split_aware = V254B.make_subject_aware_split(manifest)

    physio_seq, physio_ok, align = build_or_load_physio_sequence(manifest)
    write_csv(align, TABLES / "v256_physio_sequence_alignment_audit.csv")

    all_metrics = []
    all_sample_rows = []
    all_train_logs = []
    all_predictions: Dict[str, np.ndarray] = {}
    for protocol, split in [("subject_disjoint", split_disjoint), ("subject_aware", split_aware)]:
        train_mask = split == "train"
        vehicle_x, vehicle_audit = standardize_by_train(loaded["x_flat"].astype(np.float32), train_mask)
        vehicle_audit["protocol"] = protocol
        write_csv(vehicle_audit, TABLES / f"v256_{protocol}_vehicle_standardization_audit.csv")

        pred_map: Dict[str, np.ndarray] = {
            "v241_existing": loaded["pred_v241"].astype(np.float32),
            "v250_existing": loaded["pred_v250"].astype(np.float32),
        }
        for kind in ["vehicle_only", "physio_cnn", "vehicle_physio_cnn"]:
            print(f"[v256] train {protocol}/{kind}", flush=True)
            pred, log = train_one_model(protocol, kind, split, vehicle_x, physio_seq, physio_ok, y, valid_mask, device)
            model_name = f"v256_{kind}"
            pred_map[model_name] = pred
            all_train_logs.append(log)
            all_predictions[f"{protocol}__{model_name}"] = pred

        metrics, sample_rows = summarize_predictions(protocol, split, manifest, sample_metrics, y, valid_mask, pred_map)
        all_metrics.append(metrics)
        all_sample_rows.append(sample_rows)

    metrics = pd.concat(all_metrics, ignore_index=True)
    sample_rows = pd.concat(all_sample_rows, ignore_index=True)
    train_log = pd.concat(all_train_logs, ignore_index=True)
    write_csv(metrics, TABLES / "v256_prediction_metrics_by_bucket.csv")
    write_csv(sample_rows, TABLES / "v256_per_sample_prediction_metrics.csv")
    write_csv(train_log, TABLES / "v256_training_log.csv")
    np.savez_compressed(TENSORS / "v256_predictions.npz", **all_predictions)

    figures = [plot_test_buckets(metrics)]
    write_input_hashes()
    write_file_inventory()
    write_report(metrics, train_log, align, figures)
    write_file_inventory()
    zip_ok = make_zip()
    guardrail = {
        "pass": bool(zip_ok),
        "zip_testzip": bool(zip_ok),
        "device": str(device),
        "physio_seq_shape": list(physio_seq.shape),
        "physio_ok_rate": float(np.mean(physio_ok)),
        "n_metric_rows": int(len(metrics)),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not guardrail["pass"]:
        raise AssertionError("v256 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    focus = metrics[
        metrics["protocol"].eq("subject_disjoint")
        & metrics["eval_split"].eq("test")
        & metrics["bucket"].eq("bad_top10_v250")
        & metrics["model_name"].isin(["v256_vehicle_only", "v256_vehicle_physio_cnn"])
    ].copy()
    print(f"[v256] report={REPORTS / 'v256_raw_physio_cnn_fusion_cn.md'}")
    print(f"[v256] zip={ZIP_PATH}")
    if len(focus):
        print(focus[["model_name", "tail_rmse_mean", "delta_tail_rmse_vs_v256_vehicle"]].to_string(index=False))


if __name__ == "__main__":
    main()
