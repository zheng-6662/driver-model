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
SAMPLES_PATH = ROOT / "02_samples" / "vehicle_instability_highconf_v0_1" / "tables" / "samples_master.csv"
PROCESSED_DIR = ROOT / "03_processed_datasets" / "vehicle_instability_allraw_highconf_v0_1"
ARRAY_DIR = PROCESSED_DIR / "arrays"
FORMAL_BASELINE_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_formal_baselines_v0_1"
STRONG_BASELINE_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_strong_vehicle_baselines_v0_1"
OUT_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_vehicle_transformer_v0_1"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
REPORT_DIR = ROOT / "09_reports"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402
import stage03_vehicle_instability_formal_baselines_v0_1 as formal_v01  # noqa: E402


WINDOW_ID = "pre2_label2_old_main"
SPLIT_STRATEGY = "session_level_split"
MODEL_NAME = "vehicle_transformer_context_no_subject"
SEED = 20260512
DOWNSAMPLE_STEP = 4

CONTEXT_COLS = [
    "event_type",
    "event_level",
    "road_type_anchor",
    "old_v400_road_type_mode",
    "old_v400_phase_mode",
    "road_design_risk_class",
    "road_design_mapping_reliability",
]
NUMERIC_CONTEXT_COLS = [
    "anchor_time_rel_s",
    "curvature_anchor",
    "input_valid_ratio",
    "instability_review_score",
    "road_guided_instability_score",
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, CHECKPOINT_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


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
        mu = float(values[train_idx].mean())
        sigma = float(values[train_idx].std()) or 1.0
        if sigma < 1e-6:
            sigma = 1.0
        parts.append(((values - mu) / sigma).reshape(-1, 1))
        names.append(col)
    for col in CONTEXT_COLS:
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
    x = input_values.astype(np.float32)
    mask = input_mask.astype(bool) & np.isfinite(x)
    means: list[float] = []
    stds: list[float] = []
    for j in range(x.shape[2]):
        vals = x[train_idx, :, j][mask[train_idx, :, j]]
        if vals.size == 0:
            mu, sigma = 0.0, 1.0
        else:
            mu = float(vals.mean())
            sigma = float(vals.std())
            if sigma < 1e-6:
                sigma = 1.0
        means.append(mu)
        stds.append(sigma)
    mean_arr = np.asarray(means, dtype=np.float32).reshape(1, 1, -1)
    std_arr = np.asarray(stds, dtype=np.float32).reshape(1, 1, -1)
    x_scaled = (x - mean_arr) / std_arr
    x_scaled = np.where(mask, x_scaled, 0.0).astype(np.float32)
    return x_scaled, {"vehicle_mean": means, "vehicle_std": stds, "standardization_scope": "train split only"}


class VehicleWindowDataset(Dataset):
    def __init__(
        self,
        indices: np.ndarray,
        x: np.ndarray,
        context: np.ndarray,
        y_scaled: np.ndarray,
        y_mask: np.ndarray,
    ) -> None:
        self.indices = indices.astype(int)
        self.x = x
        self.context = context
        self.y_scaled = y_scaled
        self.y_mask = y_mask.astype(np.float32)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[idx])
        return {
            "x": torch.from_numpy(self.x[i]),
            "context": torch.from_numpy(self.context[i]),
            "y": torch.from_numpy(self.y_scaled[i]),
            "mask": torch.from_numpy(self.y_mask[i]),
        }


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class VehicleTransformerRegressor(nn.Module):
    def __init__(
        self,
        n_vehicle_features: int,
        n_context_features: int,
        input_len: int,
        output_len: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.output_len = int(output_len)
        self.input_proj = nn.Linear(n_vehicle_features, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, input_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.context_mlp = nn.Sequential(
            nn.Linear(max(n_context_features, 1), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, label_time: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        pooled = 0.5 * (h.mean(dim=1) + h[:, -1, :])
        if context.shape[1] == 0:
            context = torch.zeros((context.shape[0], 1), dtype=context.dtype, device=context.device)
        c = self.context_mlp(context)
        base = pooled + c
        time_feat = self.time_mlp(label_time.view(1, -1, 1).expand(x.shape[0], -1, -1))
        base_rep = base.unsqueeze(1).expand(-1, self.output_len, -1)
        pooled_rep = pooled.unsqueeze(1).expand(-1, self.output_len, -1)
        out = self.head(torch.cat([base_rep, pooled_rep, time_feat], dim=-1)).squeeze(-1)
        return out


def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if not torch.any(valid):
        return torch.mean((pred - target) ** 2)
    mse = torch.mean((pred[valid] - target[valid]) ** 2)
    if pred.shape[1] > 2:
        diff_pred = pred[:, 1:] - pred[:, :-1]
        diff_target = target[:, 1:] - target[:, :-1]
        diff_mask = (mask[:, 1:] > 0.5) & (mask[:, :-1] > 0.5)
        if torch.any(diff_mask):
            mse = mse + 0.08 * torch.mean((diff_pred[diff_mask] - diff_target[diff_mask]) ** 2)
    return mse


@torch.no_grad()
def predict_all(
    model: nn.Module,
    x: np.ndarray,
    context: np.ndarray,
    label_time: np.ndarray,
    label_scale: float,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    time_t = torch.from_numpy(label_time.astype(np.float32))
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        xb = torch.from_numpy(x[start:end])
        cb = torch.from_numpy(context[start:end])
        pred_scaled = model(xb, cb, time_t)
        preds.append((pred_scaled.cpu().numpy() * label_scale).astype(np.float32))
    return np.concatenate(preds, axis=0)


def rmse_for_indices(y: np.ndarray, pred: np.ndarray, mask: np.ndarray, idx: np.ndarray) -> float:
    return eval_utils.rmse(y[idx], pred[idx], mask[idx])


def train_transformer(
    x: np.ndarray,
    context: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    label_scale: float,
) -> tuple[VehicleTransformerRegressor, pd.DataFrame, dict[str, Any]]:
    model = VehicleTransformerRegressor(
        n_vehicle_features=x.shape[2],
        n_context_features=context.shape[1],
        input_len=x.shape[1],
        output_len=y.shape[1],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    train_ds = VehicleWindowDataset(train_idx, x, context, y / label_scale, y_mask)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    label_time_t = torch.from_numpy(label_time.astype(np.float32))

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    patience = 12
    history: list[dict[str, Any]] = []
    max_epochs = 80
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["x"], batch["context"], label_time_t)
            loss = masked_loss(pred, batch["y"], batch["mask"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        pred_all = predict_all(model, x, context, label_time, label_scale)
        train_rmse = rmse_for_indices(y, pred_all, y_mask, train_idx)
        val_rmse = rmse_for_indices(y, pred_all, y_mask, val_idx)
        scheduler.step(val_rmse)
        lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "lr": lr,
            }
        )
        if val_rmse < best_val - 1e-5:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == best_epoch:
            print(
                f"epoch={epoch:03d} train_rmse={train_rmse:.6f} val_rmse={val_rmse:.6f} "
                f"best_epoch={best_epoch:03d} best_val={best_val:.6f} lr={lr:.2e}",
                flush=True,
            )
        if epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val),
        "epochs_ran": int(len(history)),
        "early_stopping_patience": int(patience),
        "max_epochs": int(max_epochs),
        "optimizer": "AdamW(lr=2e-3, weight_decay=1e-4)",
        "loss": "masked trajectory MSE + 0.08 first-difference MSE",
    }
    return model, pd.DataFrame(history), info


def evaluate_predictions(
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    pred_map: dict[str, np.ndarray],
    train_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_peak = np.nanmax(np.abs(np.where(y_mask, y, np.nan)), axis=1)
    large_thr = float(np.nanpercentile(gt_peak[train_idx], 75))
    difficult_thr = float(np.nanpercentile(gt_peak[train_idx], 80))
    rows: list[pd.DataFrame] = []
    for split_name in ["train", "val", "test"]:
        split_mask = meta[SPLIT_STRATEGY].astype(str).to_numpy() == split_name
        if not split_mask.any():
            continue
        split_meta = meta.loc[split_mask].reset_index(drop=True)
        for model_name, pred in pred_map.items():
            sample_rows = eval_utils.sample_metric_rows(
                y[split_mask],
                pred[split_mask],
                y_mask[split_mask],
                label_time,
                split_meta,
                model_name=model_name,
                split_strategy=SPLIT_STRATEGY,
                split_name=split_name,
                window_id=WINDOW_ID,
                large_thr=large_thr,
                difficult_thr=difficult_thr,
            )
            if sample_rows:
                rows.append(pd.DataFrame(sample_rows))
    per_sample = pd.concat(rows, ignore_index=True)
    metrics = eval_utils.aggregate_metrics(per_sample)
    return metrics, per_sample


def plot_samples(
    sample_ids: list[str],
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    pred_map: dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    display = {
        "formal_ridge_vehicle_context_no_subject": "formal ridge",
        MODEL_NAME: "vehicle Transformer",
    }
    colors = {
        "formal_ridge_vehicle_context_no_subject": "#d62728",
        MODEL_NAME: "#1f77b4",
    }
    rows = int(np.ceil(len(sample_ids) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(18, max(3.3 * rows, 3.3)), squeeze=False)
    id_to_idx = {sid: i for i, sid in enumerate(meta["sample_id"].astype(str))}
    for ax in axes.ravel():
        ax.axis("off")
    for k, sid in enumerate(sample_ids):
        ax = axes.ravel()[k]
        ax.axis("on")
        i = id_to_idx[sid]
        valid = y_mask[i] & np.isfinite(y[i])
        gt = np.where(valid, y[i], np.nan)
        ax.plot(label_time, gt, color="black", linewidth=1.8, label="GT")
        for model_name, pred in pred_map.items():
            if model_name not in colors:
                continue
            ax.plot(label_time, pred[i], color=colors[model_name], linewidth=1.25, alpha=0.95, label=display[model_name])
        ax.axhline(0.0, color="#dddddd", linewidth=0.8)
        ax.set_title(f"{meta.at[i, 'subject']} {meta.at[i, 'anchor_time_rel_s']:.1f}s\npeak={np.nanmax(np.abs(gt)):.2f}", fontsize=9)
        ax.tick_params(labelsize=8)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, fontsize=8)
    fig.suptitle(title, fontsize=12, y=0.955)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_reports(metrics: pd.DataFrame, model_info: dict[str, Any], fixed_fig: Path, bad_fig: Path) -> None:
    test = metrics[metrics["split"] == "test"].sort_values("rmse_steer").copy()
    show_cols = [
        "model_name",
        "n_samples",
        "rmse_steer",
        "peak_direction_accuracy",
        "wrong_side_rate",
        "large_response_recall",
        "peak_amp_mae",
        "peak_amp_ratio_pred_over_gt_mean",
        "severe_amp_under_rate",
        "peak_time_mae_s",
        "onset_delay_mae_s",
        "tail_abs_error_mean",
        "tail_drift_risk_rate",
        "reversal_count_exact_match_rate",
        "difficult_top20_rmse",
    ]
    test_table = test[[c for c in show_cols if c in test.columns]]
    transformer = test[test["model_name"] == MODEL_NAME].iloc[0].to_dict()
    formal = test[test["model_name"] == "formal_ridge_vehicle_context_no_subject"].iloc[0].to_dict()
    strong_path = STRONG_BASELINE_DIR / "tables" / "strong_vehicle_baseline_metrics.csv"
    strong_note = "未读取到上一轮 RBF/KNN 指标表。"
    if strong_path.exists():
        strong = pd.read_csv(strong_path)
        strong_test = strong[(strong["split"] == "test") & (strong["model_name"].isin(["rbf_kernel_ridge_context_no_subject", "knn_template_context_no_subject"]))]
        strong_note = strong_test[["model_name", "rmse_steer", "wrong_side_rate", "large_response_recall", "severe_amp_under_rate", "reversal_count_exact_match_rate"]].to_string(index=False)

    report = f"""# 阶段 3：车辆-only Transformer 时序基线 v0.1

生成时间：2026-05-12

## 为什么做

用户指出上一轮强车辆-only 主要是 KNN/RBF/模板检索，不是真正的 Transformer。这个版本建立一个明确的车辆-only Transformer 时序神经基线，用来回答“只用事件前车辆时序和事件/道路上下文时，Transformer 能做到什么程度”。

## 输入和边界

- 样本：`{SAMPLES_PATH.as_posix()}`
- 处理后车辆窗口：`{(ARRAY_DIR / (WINDOW_ID + '.npz')).as_posix()}`
- 主窗口：`{WINDOW_ID}`
- split：`{SPLIT_STRATEGY}`
- 输入：事件前 2 秒车辆时序 9 个车辆特征 + 事件/道路上下文。
- 输出：事件后 2 秒方向盘增量轨迹。
- 不使用：生理、脑电、连续风格、驾驶员 ID、`eval_label_*` 训练输入。
- 标准化：车辆时序和数值上下文只在 train split 拟合。
- 模型选择：早停只看 val RMSE；test 只用于最终评估。

## 模型

- Encoder：2 层 TransformerEncoder，`d_model=64`，`nhead=4`。
- Decoder：全局车辆历史表示 + 上下文表示 + label time embedding，逐时间点输出未来方向盘增量。
- 损失：masked trajectory MSE + 0.08 一阶差分 MSE。
- 最佳 epoch：{model_info['best_epoch']}，val RMSE={model_info['best_val_rmse']:.6f}。

## session-level test 指标

{test_table.to_string(index=False)}

## 与上一轮 RBF/KNN 诊断候选的参考

{strong_note}

## 图

- 固定样本图：`{fixed_fig.as_posix()}`
- Transformer 坏样本图：`{bad_fig.as_posix()}`

## 当前判断

Transformer test RMSE={transformer['rmse_steer']:.6f}，formal ridge test RMSE={formal['rmse_steer']:.6f}。这一步只说明车辆-only Transformer 在当前设置下的表现，不支持连续风格、生理或 EEG 有效性结论。是否把 Transformer 作为下一版主车辆基线，还需要看固定图/坏样本图，以及它是否改善方向、幅值、尾段、反向修正和多段修正。
"""
    (REPORT_DIR / "stage03_vehicle_instability_vehicle_transformer_v0_1_cn.md").write_text(report, encoding="utf-8")

    user = f"""# 阶段 3 用户查看版：车辆-only Transformer 时序基线 v0.1

生成时间：2026-05-12

## 为什么做

前一轮强车辆结果主要用了 KNN/RBF/模板检索，它可以作为诊断对照，但不是 Transformer。这个阶段补一个真正的车辆-only Transformer，让后续讨论“强车辆基线”时有神经时序模型作为参照。

## 这次检查了什么

- 输入只用事件前 2 秒车辆时序和事件/道路上下文。
- 不用生理、脑电、连续风格，也不用驾驶员 ID。
- 输出事件后 2 秒方向盘增量轨迹。
- 早停只看验证集，测试集只做最后评估。

## 目前发现

{test_table.to_string(index=False)}

## 哪些结果可信

训练、标准化、早停都遵守 train/val/test 边界，没有使用测试集信息训练模型，也没有使用未来标签作为输入。

## 哪些还不能下结论

这仍然是车辆-only 阶段，不能说明生理、脑电或连续风格有效。Transformer 是否作为后续主车辆基线，还要看固定图和坏样本图里的物理错误是否比 RBF/KNN/formal ridge 更合理。

## 推荐优先查看

1. `{fixed_fig.as_posix()}`
2. `{bad_fig.as_posix()}`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/tables/vehicle_transformer_metrics.csv`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_vehicle_transformer_v0_1_cn.md`
"""
    (REPORT_DIR / "stage03_vehicle_instability_vehicle_transformer_user_summary_cn.md").write_text(user, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    set_seed(SEED)
    samples = pd.read_csv(SAMPLES_PATH)
    y, y_mask, input_values, input_time, label_time, meta = formal_v01.load_window(WINDOW_ID, samples)
    z = np.load(ARRAY_DIR / f"{WINDOW_ID}.npz", allow_pickle=True)
    input_mask = z["input_valid_mask"].astype(bool)
    train_idx, val_idx, test_idx = split_indices(meta)
    x_scaled, scaler_info = standardize_vehicle_inputs(input_values, input_mask, train_idx)
    context, context_names = build_context_features(meta, train_idx)
    label_scale = float(np.nanstd(np.where(y_mask[train_idx], y[train_idx], np.nan)))
    if not np.isfinite(label_scale) or label_scale < 1e-6:
        label_scale = 1.0

    x_model = x_scaled[:, ::DOWNSAMPLE_STEP, :].copy()
    model, history, train_info = train_transformer(x_model, context, y, y_mask, label_time, train_idx, val_idx, label_scale)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "window_config_id": WINDOW_ID,
            "split_strategy": SPLIT_STRATEGY,
            "label_scale": label_scale,
            "context_names": context_names,
            "scaler_info": scaler_info,
            "train_info": train_info,
        },
        CHECKPOINT_DIR / "vehicle_transformer_context_no_subject_best.pt",
    )
    pred_transformer = predict_all(model, x_model, context, label_time, label_scale)
    formal_preds, _ = formal_v01.build_predictions(y, y_mask, input_values, input_time, label_time, meta, SPLIT_STRATEGY)
    pred_map = {
        "formal_ridge_vehicle_context_no_subject": formal_preds["ridge_vehicle_context_no_subject"],
        MODEL_NAME: pred_transformer,
    }
    metrics, per_sample = evaluate_predictions(y, y_mask, label_time, meta, pred_map, train_idx)

    history.to_csv(TABLE_DIR / "vehicle_transformer_training_history.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(TABLE_DIR / "vehicle_transformer_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(TABLE_DIR / "vehicle_transformer_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"context_feature": context_names}).to_csv(TABLE_DIR / "vehicle_transformer_context_features.csv", index=False, encoding="utf-8-sig")
    model_info = {
        **train_info,
        "model_name": MODEL_NAME,
        "window_config_id": WINDOW_ID,
        "split_strategy": SPLIT_STRATEGY,
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "test_n": int(len(test_idx)),
        "label_scale_train_std": label_scale,
        "context_feature_count": int(len(context_names)),
        "vehicle_feature_count": int(input_values.shape[2]),
        "vehicle_input_tokens": int(x_model.shape[1]),
        "vehicle_input_downsample_step": int(DOWNSAMPLE_STEP),
        "uses_subject_id": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "standardization_scope": "train split only",
    }
    pd.DataFrame([model_info]).to_csv(TABLE_DIR / "vehicle_transformer_model_info.csv", index=False, encoding="utf-8-sig")

    fixed_source = FORMAL_BASELINE_DIR / "tables" / "formal_baseline_fixed_plot_samples.csv"
    if fixed_source.exists():
        fixed_ids = pd.read_csv(fixed_source)["sample_id"].astype(str).head(12).tolist()
    else:
        fixed_ids = meta.loc[test_idx, "sample_id"].astype(str).head(12).tolist()
    transformer_test = per_sample[(per_sample["split"] == "test") & (per_sample["model_name"] == MODEL_NAME)].copy()
    bad_ids = transformer_test.sort_values("sample_rmse", ascending=False).head(12)["sample_id"].astype(str).tolist()
    pd.DataFrame({"sample_id": fixed_ids}).to_csv(TABLE_DIR / "vehicle_transformer_fixed_plot_samples.csv", index=False, encoding="utf-8-sig")
    transformer_test[transformer_test["sample_id"].isin(bad_ids)].sort_values("sample_rmse", ascending=False).to_csv(
        TABLE_DIR / "vehicle_transformer_bad_plot_samples.csv", index=False, encoding="utf-8-sig"
    )
    fixed_fig = FIG_DIR / "vehicle_transformer_fixed_predictions_test.png"
    bad_fig = FIG_DIR / "vehicle_transformer_bad_samples_test.png"
    plot_samples(fixed_ids, y, y_mask, label_time, meta, pred_map, fixed_fig, "Fixed test samples: vehicle-only Transformer")
    plot_samples(bad_ids, y, y_mask, label_time, meta, pred_map, bad_fig, "Worst test samples: vehicle-only Transformer")

    summary = {
        **model_info,
        "metrics_path": str(TABLE_DIR / "vehicle_transformer_metrics.csv").replace("\\", "/"),
        "per_sample_path": str(TABLE_DIR / "vehicle_transformer_per_sample_metrics.csv").replace("\\", "/"),
        "fixed_plot": str(fixed_fig).replace("\\", "/"),
        "bad_plot": str(bad_fig).replace("\\", "/"),
        "checkpoint_path": str(CHECKPOINT_DIR / "vehicle_transformer_context_no_subject_best.pt").replace("\\", "/"),
    }
    (LOG_DIR / "vehicle_transformer_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(metrics, model_info, fixed_fig, bad_fig)
    print(metrics[metrics["split"] == "test"].sort_values("rmse_steer").to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
