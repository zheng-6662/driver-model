from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from joblib import parallel_backend
from sklearn.ensemble import ExtraTreesRegressor
from torch import nn


matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


SEEDS = (20260902, 20260903, 20260904)
N_FOLDS = 3
HISTORY_STEPS = 40
FUTURE_STEPS = 20
INPUT_CHANNELS = (
    "steer_deg",
    "brake",
    "accelerator",
    "speed_kmh",
    "ax",
    "ay",
    "yaw_rate",
    "roll",
    "roll_rate",
    "curvature",
)
TARGET_CHANNELS = ("steer_deg", "brake", "accelerator", "speed_kmh")
MODEL_NAMES = ("hold", "linear", "extra_trees", "transformer", "et_transformer_residual")
MODEL_CN = {
    "hold": "当前值保持",
    "linear": "线性趋势外推",
    "extra_trees": "ExtraTrees",
    "transformer": "小型Transformer",
    "et_transformer_residual": "ExtraTrees+Transformer残差",
}
ACTION_NORMALIZERS = np.array([5.0, 0.05, 0.05], dtype=np.float32)
CHANNEL_WEIGHTS = np.array([1.0, 1.0, 1.0, 0.2], dtype=np.float32)
RESIDUAL_BOUNDS = np.array([20.0, 0.15, 0.15, 3.0], dtype=np.float32)
FUTURE_TIME = np.arange(1, FUTURE_STEPS + 1, dtype=np.float32) / 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run84 连续多动作五模型统一比较")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


class SmallTransformer(nn.Module):
    """两层、64维、4头的固定小型编码器；残差版额外读取 ExtraTrees 基础曲线。"""

    def __init__(self, residual: bool, residual_bound_scaled: np.ndarray | None = None):
        super().__init__()
        self.residual = residual
        self.input_projection = nn.Linear(len(INPUT_CHANNELS) * 2, 64)
        self.position = nn.Parameter(torch.zeros(1, HISTORY_STEPS, 64))
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.summary_norm = nn.LayerNorm(64)
        self.base_projection = nn.Linear(FUTURE_STEPS * len(TARGET_CHANNELS), 64) if residual else None
        self.output = nn.Linear(64, FUTURE_STEPS * len(TARGET_CHANNELS))
        nn.init.normal_(self.position, std=0.02)
        if residual:
            if residual_bound_scaled is None:
                raise ValueError("残差模型必须提供边界")
            bound = np.tile(residual_bound_scaled[None, :], (FUTURE_STEPS, 1)).astype(np.float32)
            self.register_buffer("residual_bound", torch.from_numpy(bound))

    def forward(self, sequence: torch.Tensor, base_prediction: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(self.input_projection(sequence) + self.position)
        summary = self.summary_norm(0.5 * (encoded[:, -1] + encoded.mean(dim=1)))
        if self.residual:
            if base_prediction is None:
                raise ValueError("残差模型缺少 ExtraTrees 基础预测")
            summary = summary + self.base_projection(base_prediction.flatten(1))
        output = self.output(summary).view(-1, FUTURE_STEPS, len(TARGET_CHANNELS))
        return torch.tanh(output) * self.residual_bound if self.residual else output


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assign_subject_folds(metadata: pd.DataFrame, seed: int) -> pd.DataFrame:
    """把同一驾驶员的所有批次和 recording 绑定在同一外折，并平衡窗口数量。"""
    counts = metadata.groupby("subject_alias").size().rename("windows").reset_index()
    rng = np.random.default_rng(seed)
    counts["tie_break"] = rng.random(len(counts))
    counts = counts.sort_values(["windows", "tie_break"], ascending=[False, True]).reset_index(drop=True)
    totals = np.zeros(N_FOLDS, dtype=np.int64)
    subject_counts = np.zeros(N_FOLDS, dtype=np.int64)
    folds = []
    for row in counts.itertuples(index=False):
        candidates = np.flatnonzero(totals == totals.min())
        fold = int(candidates[np.argmin(subject_counts[candidates])])
        folds.append(fold + 1)
        totals[fold] += int(row.windows)
        subject_counts[fold] += 1
    counts["fold"] = folds
    return counts[["subject_alias", "windows", "fold"]].sort_values("subject_alias").reset_index(drop=True)


def fit_input_scaler(history: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.zeros(len(INPUT_CHANNELS), dtype=np.float32)
    stds = np.ones(len(INPUT_CHANNELS), dtype=np.float32)
    finite_counts = np.zeros(len(INPUT_CHANNELS), dtype=np.int64)
    for channel in range(len(INPUT_CHANNELS)):
        values = np.asarray(history[train_indices, :, channel], dtype=np.float64)
        finite = np.isfinite(values)
        finite_counts[channel] = int(finite.sum())
        if finite_counts[channel] == 0:
            continue
        means[channel] = float(values[finite].mean())
        std = float(values[finite].std())
        stds[channel] = std if std > 1e-8 else 1.0
    return means, stds, finite_counts


def fit_target_scale(targets: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    values = np.asarray(targets[train_indices], dtype=np.float64)
    scales = values.reshape(-1, len(TARGET_CHANNELS)).std(axis=0).astype(np.float32)
    if np.any(scales <= 1e-8):
        raise AssertionError(f"训练折目标尺度异常: {scales}")
    return scales


def fit_feature_imputer(features: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    train = np.asarray(features[train_indices], dtype=np.float32)
    finite_counts = np.isfinite(train).sum(axis=0)
    medians = np.zeros(train.shape[1], dtype=np.float32)
    present = finite_counts > 0
    medians[present] = np.nanmedian(train[:, present], axis=0)
    return medians


def impute_features(values: np.ndarray, medians: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    missing = ~np.isfinite(result)
    result[missing] = medians[np.nonzero(missing)[1]]
    return result


def make_training_weights(metadata: pd.DataFrame, train_indices: np.ndarray) -> np.ndarray:
    train = metadata.iloc[train_indices]
    weights = np.ones(len(train), dtype=np.float64)
    weights *= np.where(train["high_dynamic"].to_numpy(bool), 1.5, 1.0)
    weights *= np.where(train["any_action_change"].to_numpy(bool), 2.0, 1.0)
    subject = train["subject_alias"].to_numpy()
    target_total = len(train) / train["subject_alias"].nunique()
    for name in np.unique(subject):
        mask = subject == name
        weights[mask] *= target_total / weights[mask].sum()
    weights /= weights.mean()
    return weights.astype(np.float32)


def fit_extra_trees_predict(
    features: np.ndarray,
    targets: np.ndarray,
    metadata: pd.DataFrame,
    fit_indices: np.ndarray,
    predict_indices: np.ndarray,
    target_scale: np.ndarray,
    seed: int,
    trees: int,
) -> np.ndarray:
    """按实际拟合驾驶员重新平衡权重，并恢复车速0.2辅助权重后的普通目标尺度。"""
    feature_median = fit_feature_imputer(features, fit_indices)
    fit_features = impute_features(features[fit_indices], feature_median)
    prediction_features = impute_features(features[predict_indices], feature_median)
    fit_weights = make_training_weights(metadata, fit_indices)
    square_root_channel_weight = np.sqrt(CHANNEL_WEIGHTS).astype(np.float32)
    fit_target = (
        targets[fit_indices] / target_scale[None, None, :] * square_root_channel_weight[None, None, :]
    )
    model = ExtraTreesRegressor(
        n_estimators=trees,
        max_depth=18,
        min_samples_leaf=8,
        max_features=0.65,
        random_state=seed,
        n_jobs=6,
    )
    with parallel_backend("threading", n_jobs=6):
        model.fit(fit_features, fit_target.reshape(len(fit_indices), -1), sample_weight=fit_weights)
        weighted_prediction = model.predict(prediction_features).reshape(
            -1, FUTURE_STEPS, len(TARGET_CHANNELS)
        )
    prediction = weighted_prediction / square_root_channel_weight[None, None, :]
    del model, fit_features, prediction_features, fit_target, fit_weights
    gc.collect()
    return prediction.astype(np.float32)


def crossfit_extra_trees_for_residual(
    features: np.ndarray,
    targets: np.ndarray,
    metadata: pd.DataFrame,
    outer_train_indices: np.ndarray,
    target_scale: np.ndarray,
    seed: int,
    trees: int,
) -> tuple[np.ndarray, list[dict]]:
    """在外层训练人口内部按驾驶员做3折交叉拟合，生成诚实的残差基础预测。"""
    outer_metadata = metadata.iloc[outer_train_indices].reset_index(drop=True)
    assignments = assign_subject_folds(outer_metadata, seed)
    inner_fold_by_subject = assignments.set_index("subject_alias")["fold"].to_dict()
    local_folds = outer_metadata["subject_alias"].map(inner_fold_by_subject).to_numpy(int)
    oof_prediction = np.empty(
        (len(outer_train_indices), FUTURE_STEPS, len(TARGET_CHANNELS)), dtype=np.float32
    )
    audit_rows = []
    for inner_fold in range(1, N_FOLDS + 1):
        inner_fit_positions = np.flatnonzero(local_folds != inner_fold)
        inner_valid_positions = np.flatnonzero(local_folds == inner_fold)
        inner_fit_indices = outer_train_indices[inner_fit_positions]
        inner_valid_indices = outer_train_indices[inner_valid_positions]
        fit_subjects = set(metadata.iloc[inner_fit_indices]["subject_alias"])
        valid_subjects = set(metadata.iloc[inner_valid_indices]["subject_alias"])
        fit_recordings = set(metadata.iloc[inner_fit_indices]["recording_alias"])
        valid_recordings = set(metadata.iloc[inner_valid_indices]["recording_alias"])
        if fit_subjects & valid_subjects or fit_recordings & valid_recordings:
            raise AssertionError(f"残差内层交叉拟合泄漏: inner_fold={inner_fold}")
        inner_target_scale = fit_target_scale(targets, inner_fit_indices)
        inner_prediction_scaled = fit_extra_trees_predict(
            features,
            targets,
            metadata,
            inner_fit_indices,
            inner_valid_indices,
            inner_target_scale,
            seed + inner_fold,
            trees,
        )
        # 内折预测先恢复物理单位，再转到外层训练人口的尺度供残差 Transformer 使用。
        oof_prediction[inner_valid_positions] = (
            inner_prediction_scaled * inner_target_scale[None, None, :] / target_scale[None, None, :]
        )
        audit_rows.append(
            {
                "inner_fold": inner_fold,
                "fit_windows": len(inner_fit_indices),
                "valid_windows": len(inner_valid_indices),
                "fit_subjects": len(fit_subjects),
                "valid_subjects": len(valid_subjects),
                "subject_overlap": 0,
                "recording_overlap": 0,
            }
        )
    if not np.isfinite(oof_prediction).all():
        raise AssertionError("残差内层 OOF ExtraTrees 预测不完整")
    return oof_prediction, audit_rows


def linear_prediction(histories: np.ndarray) -> np.ndarray:
    recent = np.asarray(histories[:, -11:, :4], dtype=np.float32)
    time = np.arange(11, dtype=np.float32) / 20.0
    centered = time - time.mean()
    denominator = float(np.dot(centered, centered))
    slopes = np.einsum("t,ntc->nc", centered, recent - recent.mean(axis=1, keepdims=True)) / denominator
    prediction = slopes[:, None, :] * FUTURE_TIME[None, :, None]
    current = histories[:, -1, :4]
    for channel in (1, 2):
        prediction[:, :, channel] = np.clip(
            current[:, None, channel] + prediction[:, :, channel], 0.0, 1.0
        ) - current[:, None, channel]
    return prediction.astype(np.float32)


def normalized_sequence_batch(
    history: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    values = np.asarray(history[indices], dtype=np.float32)
    missing = ~np.isfinite(values)
    normalized = (np.where(missing, mean[None, None, :], values) - mean[None, None, :]) / std[None, None, :]
    return np.concatenate([normalized, missing.astype(np.float32)], axis=2).astype(np.float32)


def weighted_curve_loss(prediction: torch.Tensor, truth: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    element = torch.nn.functional.smooth_l1_loss(prediction, truth, reduction="none")
    channel_weight = torch.as_tensor(CHANNEL_WEIGHTS, dtype=element.dtype, device=element.device)
    per_sample = (element * channel_weight[None, None, :]).sum(dim=(1, 2)) / (
        FUTURE_STEPS * float(CHANNEL_WEIGHTS.sum())
    )
    return (per_sample * sample_weight).sum() / sample_weight.sum()


def train_transformer(
    history: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    sample_weights: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    target_scale: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    base_prediction_scaled: np.ndarray | None,
    batch_size: int,
) -> tuple[SmallTransformer, list[dict]]:
    residual = base_prediction_scaled is not None
    bound_scaled = RESIDUAL_BOUNDS / target_scale if residual else None
    model = SmallTransformer(residual=residual, residual_bound_scaled=bound_scaled).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    log_rows = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_indices))
        model.train()
        total_weighted_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            global_indices = train_indices[positions]
            sequence = torch.from_numpy(
                normalized_sequence_batch(history, global_indices, input_mean, input_std)
            ).to(device)
            truth = torch.from_numpy(
                np.asarray(targets[global_indices], dtype=np.float32) / target_scale[None, None, :]
            ).to(device)
            weight = torch.from_numpy(sample_weights[positions]).to(device)
            optimizer.zero_grad(set_to_none=True)
            if residual:
                base = torch.from_numpy(base_prediction_scaled[positions]).to(device)
                prediction = base + model(sequence, base)
            else:
                prediction = model(sequence)
            loss = weighted_curve_loss(prediction, truth, weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_weight = float(weight.sum().detach().cpu())
            total_weighted_loss += float(loss.detach().cpu()) * batch_weight
            total_weight += batch_weight
        epoch_loss = total_weighted_loss / total_weight
        log_rows.append({"epoch": epoch, "loss": epoch_loss, "residual": residual})
        print(f"TRANSFORMER residual={residual} epoch={epoch}/{epochs} loss={epoch_loss:.6f}", flush=True)
    return model, log_rows


def predict_transformer(
    model: SmallTransformer,
    history: np.ndarray,
    indices: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    target_scale: np.ndarray,
    device: torch.device,
    base_prediction_scaled: np.ndarray | None,
    batch_size: int,
) -> np.ndarray:
    predictions = np.empty((len(indices), FUTURE_STEPS, len(TARGET_CHANNELS)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            sequence = torch.from_numpy(
                normalized_sequence_batch(history, indices[start:end], input_mean, input_std)
            ).to(device)
            if base_prediction_scaled is None:
                prediction = model(sequence)
            else:
                base = torch.from_numpy(base_prediction_scaled[start:end]).to(device)
                prediction = base + model(sequence, base)
            predictions[start:end] = prediction.cpu().numpy() * target_scale[None, None, :]
    return predictions


def population_positions(
    population: str,
    test_metadata: pd.DataFrame,
    fixed_mapping: pd.DataFrame,
) -> np.ndarray:
    if population == "all_continuous":
        return np.arange(len(test_metadata), dtype=np.int64)
    metadata_column = {
        "high_dynamic_not_started": "high_dynamic_not_started",
        "action_started": "action_started",
        "ordinary": "ordinary",
    }.get(population)
    if metadata_column is not None:
        return np.flatnonzero(test_metadata[metadata_column].to_numpy(bool))
    subset = {
        "distance_v2_305": "distance_v2_305",
        "low_mu_v2_70": "low_mu_v2_70",
        "release_v3_historical_2323": "release_v3_historical_2323",
    }[population]
    mapping = fixed_mapping.loc[fixed_mapping["subset"].eq(subset)]
    local_lookup = {int(window): position for position, window in enumerate(test_metadata["window_index"])}
    return np.array(
        [local_lookup[int(window)] for window in mapping["window_index"] if int(window) in local_lookup],
        dtype=np.int64,
    )


def metric_rows(
    seed: int,
    fold: int,
    population: str,
    model_name: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    positions: np.ndarray,
) -> list[dict]:
    if not len(positions):
        return []
    error = prediction[positions] - truth[positions]
    rows = []
    channels = [0] if population == "release_v3_historical_2323" else list(range(len(TARGET_CHANNELS)))
    for channel in channels:
        channel_error = error[:, :, channel]
        rows.append(
            {
                "seed": seed,
                "fold": fold,
                "population": population,
                "model": model_name,
                "channel": TARGET_CHANNELS[channel],
                "windows": len(positions),
                "mae": float(np.mean(np.abs(channel_error))),
                "rmse": float(np.sqrt(np.mean(channel_error**2))),
                "endpoint_mae": float(np.mean(np.abs(channel_error[:, -1]))),
            }
        )
    if population != "release_v3_historical_2323":
        action_error = error[:, :, :3] / ACTION_NORMALIZERS[None, None, :]
        rows.append(
            {
                "seed": seed,
                "fold": fold,
                "population": population,
                "model": model_name,
                "channel": "action_macro",
                "windows": len(positions),
                "mae": float(np.mean(np.abs(action_error))),
                "rmse": float(np.sqrt(np.mean(action_error**2))),
                "endpoint_mae": float(np.mean(np.abs(action_error[:, -1]))),
            }
        )
    return rows


def subject_metric_rows(
    seed: int,
    fold: int,
    population: str,
    model_name: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    positions: np.ndarray,
    test_metadata: pd.DataFrame,
) -> list[dict]:
    rows = []
    subjects = test_metadata.iloc[positions]["subject_alias"].to_numpy()
    for subject in np.unique(subjects):
        selected = positions[subjects == subject]
        error = prediction[selected] - truth[selected]
        channels = [0] if population == "release_v3_historical_2323" else list(range(len(TARGET_CHANNELS)))
        for channel in channels:
            channel_error = error[:, :, channel]
            rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "population": population,
                    "model": model_name,
                    "subject_alias": subject,
                    "channel": TARGET_CHANNELS[channel],
                    "windows": len(selected),
                    "mae": float(np.mean(np.abs(channel_error))),
                    "rmse": float(np.sqrt(np.mean(channel_error**2))),
                    "endpoint_mae": float(np.mean(np.abs(channel_error[:, -1]))),
                }
            )
        if population != "release_v3_historical_2323":
            action_error = error[:, :, :3] / ACTION_NORMALIZERS[None, None, :]
            rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "population": population,
                    "model": model_name,
                    "subject_alias": subject,
                    "channel": "action_macro",
                    "windows": len(selected),
                    "mae": float(np.mean(np.abs(action_error))),
                    "rmse": float(np.sqrt(np.mean(action_error**2))),
                    "endpoint_mae": float(np.mean(np.abs(action_error[:, -1]))),
                }
            )
    return rows


def block_metric_rows(
    seed: int,
    fold: int,
    model_name: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    test_metadata: pd.DataFrame,
) -> list[dict]:
    per_window = np.mean(
        np.abs(prediction[:, :, :3] - truth[:, :, :3]) / ACTION_NORMALIZERS[None, None, :], axis=(1, 2)
    )
    frame = test_metadata[["subject_alias", "recording_alias", "time_block_id"]].copy()
    frame["action_nmae"] = per_window
    grouped = frame.groupby(["subject_alias", "recording_alias", "time_block_id"], as_index=False).agg(
        windows=("action_nmae", "size"), action_nmae=("action_nmae", "mean")
    )
    grouped.insert(0, "model", model_name)
    grouped.insert(0, "fold", fold)
    grouped.insert(0, "seed", seed)
    return grouped.to_dict("records")


def choose_examples(metadata: pd.DataFrame, targets: np.ndarray, fixed_mapping: pd.DataFrame) -> pd.DataFrame:
    magnitude = np.mean(np.abs(targets[:, :, :3]) / ACTION_NORMALIZERS[None, None, :], axis=(1, 2))
    rows = []
    populations = {
        "distance_v2_305": fixed_mapping.loc[fixed_mapping["subset"].eq("distance_v2_305"), "window_index"].unique(),
        "low_mu_v2_70": fixed_mapping.loc[fixed_mapping["subset"].eq("low_mu_v2_70"), "window_index"].unique(),
        "high_dynamic_not_started": metadata.loc[metadata["high_dynamic_not_started"], "window_index"].to_numpy(),
        "ordinary": metadata.loc[metadata["ordinary"], "window_index"].to_numpy(),
    }
    for population, candidates in populations.items():
        candidates = np.asarray(candidates, dtype=np.int64)
        order = candidates[np.argsort(magnitude[candidates], kind="stable")]
        selected = int(order[len(order) // 2])
        row = metadata.iloc[selected]
        rows.append(
            {
                "example_id": f"EX{len(rows) + 1:02d}",
                "population": population,
                "window_index": selected,
                "subject_alias": row.subject_alias,
                "recording_alias": row.recording_alias,
                "query_time_s": float(row.query_time_s),
                "target_action_nmae_magnitude": float(magnitude[selected]),
                "selection_rule": "目标动作归一化幅值中位窗口；不按模型误差选择",
            }
        )
    return pd.DataFrame(rows)


def run_seed(
    run_root: Path,
    seed: int,
    history: np.ndarray,
    targets: np.ndarray,
    features: np.ndarray,
    metadata: pd.DataFrame,
    fixed_mapping: pd.DataFrame,
    examples: pd.DataFrame,
    device: torch.device,
    sanity: bool,
) -> None:
    output_root = run_root / ("sanity" if sanity else "raw_results")
    output_root.mkdir(parents=True, exist_ok=True)
    assignments = assign_subject_folds(metadata, seed)
    fold_by_subject = assignments.set_index("subject_alias")["fold"].to_dict()
    window_folds = metadata["subject_alias"].map(fold_by_subject).to_numpy(int)
    assignments.to_csv(output_root / f"fold_assignments_seed{seed}.csv", index=False, encoding="utf-8-sig")

    fold_rows: list[dict] = []
    subject_rows: list[dict] = []
    block_rows: list[dict] = []
    example_rows: list[dict] = []
    split_rows: list[dict] = []
    scaler_rows: list[dict] = []
    training_rows: list[dict] = []
    inner_split_rows: list[dict] = []

    folds_to_run = [1] if sanity else list(range(1, N_FOLDS + 1))
    for fold in folds_to_run:
        set_seed(seed + fold)
        train_indices = np.flatnonzero(window_folds != fold)
        test_indices = np.flatnonzero(window_folds == fold)
        if sanity:
            rng = np.random.default_rng(seed)
            train_indices = np.sort(rng.choice(train_indices, size=min(8000, len(train_indices)), replace=False))
            test_indices = np.sort(rng.choice(test_indices, size=min(3000, len(test_indices)), replace=False))
        train_subjects = set(metadata.iloc[train_indices]["subject_alias"])
        test_subjects = set(metadata.iloc[test_indices]["subject_alias"])
        train_recordings = set(metadata.iloc[train_indices]["recording_alias"])
        test_recordings = set(metadata.iloc[test_indices]["recording_alias"])
        if train_subjects & test_subjects or train_recordings & test_recordings:
            raise AssertionError(f"seed={seed} fold={fold} 外层泄漏")
        split_rows.append(
            {
                "seed": seed,
                "fold": fold,
                "train_windows": len(train_indices),
                "test_windows": len(test_indices),
                "train_subjects": len(train_subjects),
                "test_subjects": len(test_subjects),
                "train_recordings": len(train_recordings),
                "test_recordings": len(test_recordings),
                "subject_overlap": 0,
                "recording_overlap": 0,
            }
        )

        input_mean, input_std, finite_counts = fit_input_scaler(history, train_indices)
        target_scale = fit_target_scale(targets, train_indices)
        train_weights = make_training_weights(metadata, train_indices)
        for channel, mean, std, count in zip(INPUT_CHANNELS, input_mean, input_std, finite_counts):
            scaler_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "kind": "input",
                    "channel": channel,
                    "center": float(mean),
                    "scale": float(std),
                    "train_finite_values": int(count),
                }
            )
        for channel, scale in zip(TARGET_CHANNELS, target_scale):
            scaler_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "kind": "target",
                    "channel": channel,
                    "center": 0.0,
                    "scale": float(scale),
                    "train_finite_values": int(len(train_indices) * FUTURE_STEPS),
                }
            )

        trees = 8 if sanity else 64
        print(f"ET FULL seed={seed} fold={fold} train={len(train_indices)} trees={trees}", flush=True)
        et_test_scaled = fit_extra_trees_predict(
            features,
            targets,
            metadata,
            train_indices,
            test_indices,
            target_scale,
            seed + fold,
            trees,
        )
        print(f"ET INNER-OOF seed={seed} fold={fold}", flush=True)
        et_train_scaled, inner_audit = crossfit_extra_trees_for_residual(
            features,
            targets,
            metadata,
            train_indices,
            target_scale,
            seed + fold * 100,
            trees,
        )
        for row in inner_audit:
            inner_split_rows.append({"seed": seed, "outer_fold": fold, **row})

        batch_size = 512 if sanity else 1024
        direct_epochs = 1 if sanity else 5
        residual_epochs = 1 if sanity else 4
        direct_model, direct_log = train_transformer(
            history,
            targets,
            train_indices,
            train_weights,
            input_mean,
            input_std,
            target_scale,
            seed + fold * 10 + 1,
            direct_epochs,
            device,
            None,
            batch_size,
        )
        transformer_prediction = predict_transformer(
            direct_model,
            history,
            test_indices,
            input_mean,
            input_std,
            target_scale,
            device,
            None,
            batch_size,
        )
        residual_model, residual_log = train_transformer(
            history,
            targets,
            train_indices,
            train_weights,
            input_mean,
            input_std,
            target_scale,
            seed + fold * 10 + 2,
            residual_epochs,
            device,
            et_train_scaled,
            batch_size,
        )
        fusion_prediction = predict_transformer(
            residual_model,
            history,
            test_indices,
            input_mean,
            input_std,
            target_scale,
            device,
            et_test_scaled,
            batch_size,
        )
        for row in direct_log:
            training_rows.append({"seed": seed, "fold": fold, "model": "transformer", **row})
        for row in residual_log:
            training_rows.append({"seed": seed, "fold": fold, "model": "et_transformer_residual", **row})

        test_history = history[test_indices]
        test_truth = targets[test_indices]
        predictions = {
            "hold": np.zeros_like(test_truth),
            "linear": linear_prediction(test_history),
            "extra_trees": et_test_scaled * target_scale[None, None, :],
            "transformer": transformer_prediction,
            "et_transformer_residual": fusion_prediction,
        }
        test_metadata = metadata.iloc[test_indices].reset_index(drop=True)
        populations = (
            "all_continuous",
            "distance_v2_305",
            "low_mu_v2_70",
            "high_dynamic_not_started",
            "action_started",
            "ordinary",
            "release_v3_historical_2323",
        )
        for population in populations:
            positions = population_positions(population, test_metadata, fixed_mapping)
            for model_name, prediction in predictions.items():
                fold_rows.extend(metric_rows(seed, fold, population, model_name, test_truth, prediction, positions))
                subject_rows.extend(
                    subject_metric_rows(
                        seed, fold, population, model_name, test_truth, prediction, positions, test_metadata
                    )
                )
        for model_name, prediction in predictions.items():
            block_rows.extend(block_metric_rows(seed, fold, model_name, test_truth, prediction, test_metadata))

        example_lookup = {int(window): example for window, example in zip(examples.window_index, examples.example_id)}
        local_lookup = {int(window): position for position, window in enumerate(test_metadata["window_index"])}
        for window_index, example_id in example_lookup.items():
            if window_index not in local_lookup:
                continue
            position = local_lookup[window_index]
            for model_name, prediction in predictions.items():
                for channel_index, channel in enumerate(TARGET_CHANNELS):
                    for time_index, time_s in enumerate(FUTURE_TIME):
                        example_rows.append(
                            {
                                "seed": seed,
                                "fold": fold,
                                "example_id": example_id,
                                "window_index": window_index,
                                "model": model_name,
                                "channel": channel,
                                "future_time_s": float(time_s),
                                "truth": float(test_truth[position, time_index, channel_index]),
                                "prediction": float(prediction[position, time_index, channel_index]),
                            }
                        )

        del et_train_scaled, et_test_scaled, direct_model, residual_model
        del transformer_prediction, fusion_prediction, predictions, test_history, test_truth
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"DONE seed={seed} fold={fold}", flush=True)

    pd.DataFrame(fold_rows).to_csv(output_root / f"fold_metrics_seed{seed}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(subject_rows).to_csv(output_root / f"subject_metrics_seed{seed}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(block_rows).to_csv(output_root / f"block_metrics_seed{seed}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(example_rows).to_csv(output_root / f"example_predictions_seed{seed}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(split_rows).to_csv(output_root / f"split_audit_seed{seed}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(scaler_rows).to_csv(output_root / f"scalers_seed{seed}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(training_rows).to_csv(output_root / f"training_log_seed{seed}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inner_split_rows).to_csv(
        output_root / f"inner_split_audit_seed{seed}.csv", index=False, encoding="utf-8-sig"
    )


def weighted_group_mean(frame: pd.DataFrame) -> float:
    return float(np.average(frame["mae"], weights=frame["windows"]))


def format_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda value: f"{value:.{decimals}f}" if pd.notna(value) else "")
    return display.to_markdown(index=False)


def summarize(run_root: Path) -> None:
    raw_root = run_root / "raw_results"
    result_root = run_root / "results"
    review_root = run_root / "review_light"
    figure_root = review_root / "figures"
    result_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        required = [
            raw_root / f"fold_metrics_seed{seed}.csv",
            raw_root / f"subject_metrics_seed{seed}.csv",
            raw_root / f"block_metrics_seed{seed}.csv",
            raw_root / f"example_predictions_seed{seed}.csv",
            raw_root / f"split_audit_seed{seed}.csv",
            raw_root / f"inner_split_audit_seed{seed}.csv",
            raw_root / f"scalers_seed{seed}.csv",
            raw_root / f"training_log_seed{seed}.csv",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"三种子结果尚未齐全: {missing}")

    fold_metrics = pd.concat(
        [pd.read_csv(raw_root / f"fold_metrics_seed{seed}.csv") for seed in SEEDS], ignore_index=True
    )
    subject_metrics = pd.concat(
        [pd.read_csv(raw_root / f"subject_metrics_seed{seed}.csv") for seed in SEEDS], ignore_index=True
    )
    block_metrics = pd.concat(
        [pd.read_csv(raw_root / f"block_metrics_seed{seed}.csv") for seed in SEEDS], ignore_index=True
    )
    example_predictions = pd.concat(
        [pd.read_csv(raw_root / f"example_predictions_seed{seed}.csv") for seed in SEEDS], ignore_index=True
    )
    split_audit = pd.concat(
        [pd.read_csv(raw_root / f"split_audit_seed{seed}.csv") for seed in SEEDS], ignore_index=True
    )
    fold_metrics.to_csv(result_root / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    subject_metrics.to_csv(result_root / "subject_metrics.csv", index=False, encoding="utf-8-sig")
    block_metrics.to_csv(result_root / "block_metrics.csv", index=False, encoding="utf-8-sig")
    split_audit.to_csv(result_root / "split_audit.csv", index=False, encoding="utf-8-sig")

    driver_macro = (
        subject_metrics.groupby(["seed", "population", "model", "channel"], as_index=False)
        .agg(
            driver_macro_mae=("mae", "mean"),
            driver_macro_rmse=("rmse", "mean"),
            driver_macro_endpoint_mae=("endpoint_mae", "mean"),
            drivers=("subject_alias", "nunique"),
        )
    )
    pooled_rows = []
    for keys, group in fold_metrics.groupby(["seed", "population", "model", "channel"], sort=True):
        pooled_rows.append(
            {
                "seed": keys[0],
                "population": keys[1],
                "model": keys[2],
                "channel": keys[3],
                "pooled_mae": weighted_group_mean(group),
                "pooled_rmse": float(
                    np.sqrt(np.average(group["rmse"] ** 2, weights=group["windows"]))
                ),
                "pooled_endpoint_mae": float(
                    np.average(group["endpoint_mae"], weights=group["windows"])
                ),
                "windows": int(group["windows"].sum()),
            }
        )
    seed_summary = driver_macro.merge(pd.DataFrame(pooled_rows), on=["seed", "population", "model", "channel"])
    seed_summary.to_csv(result_root / "seed_summary.csv", index=False, encoding="utf-8-sig")

    all_summary = seed_summary.loc[seed_summary["population"].eq("all_continuous")]
    main_rows = []
    for model in MODEL_NAMES:
        model_rows = all_summary.loc[all_summary["model"].eq(model)]
        row = {"model": model, "模型": MODEL_CN[model]}
        for channel in ["action_macro", *TARGET_CHANNELS]:
            channel_rows = model_rows.loc[model_rows["channel"].eq(channel)]
            for metric in ["mae", "rmse", "endpoint_mae"]:
                driver_values = channel_rows[f"driver_macro_{metric}"]
                row[f"{channel}_driver_macro_{metric}_mean"] = float(driver_values.mean())
                row[f"{channel}_driver_macro_{metric}_seed_sd"] = float(driver_values.std(ddof=1))
                row[f"{channel}_pooled_{metric}_mean"] = float(channel_rows[f"pooled_{metric}"].mean())
        main_rows.append(row)
    main_comparison = pd.DataFrame(main_rows).sort_values("action_macro_driver_macro_mae_mean").reset_index(drop=True)
    main_comparison.to_csv(result_root / "main_model_comparison.csv", index=False, encoding="utf-8-sig")

    population_summary = (
        seed_summary.loc[seed_summary["channel"].eq("action_macro")]
        .groupby(["population", "model"], as_index=False)
        .agg(
            driver_macro_mean=("driver_macro_mae", "mean"),
            driver_macro_seed_sd=("driver_macro_mae", "std"),
            driver_macro_rmse_mean=("driver_macro_rmse", "mean"),
            driver_macro_endpoint_mae_mean=("driver_macro_endpoint_mae", "mean"),
            pooled_mean=("pooled_mae", "mean"),
            pooled_rmse_mean=("pooled_rmse", "mean"),
            pooled_endpoint_mae_mean=("pooled_endpoint_mae", "mean"),
            represented_drivers=("drivers", "max"),
            evaluated_windows=("windows", "max"),
        )
    )
    release_summary = (
        seed_summary.loc[
            seed_summary["population"].eq("release_v3_historical_2323")
            & seed_summary["channel"].eq("steer_deg")
        ]
        .groupby(["population", "model"], as_index=False)
        .agg(
            driver_macro_mean=("driver_macro_mae", "mean"),
            driver_macro_seed_sd=("driver_macro_mae", "std"),
            driver_macro_rmse_mean=("driver_macro_rmse", "mean"),
            driver_macro_endpoint_mae_mean=("driver_macro_endpoint_mae", "mean"),
            pooled_mean=("pooled_mae", "mean"),
            pooled_rmse_mean=("pooled_rmse", "mean"),
            pooled_endpoint_mae_mean=("pooled_endpoint_mae", "mean"),
            represented_drivers=("drivers", "max"),
            evaluated_windows=("windows", "max"),
        )
    )
    population_summary = pd.concat([population_summary, release_summary], ignore_index=True)
    population_summary["模型"] = population_summary["model"].map(MODEL_CN)
    population_summary.to_csv(result_root / "evaluation_population_comparison.csv", index=False, encoding="utf-8-sig")

    all_driver = subject_metrics.loc[
        subject_metrics["population"].eq("all_continuous") & subject_metrics["channel"].eq("action_macro")
    ]
    driver_average = (
        all_driver.groupby(["subject_alias", "model"], as_index=False)
        .agg(action_nmae=("mae", "mean"), seeds=("seed", "nunique"))
    )
    driver_wide = driver_average.pivot(index="subject_alias", columns="model", values="action_nmae").reset_index()
    simple_models = ["hold", "linear"]
    learned_models = ["extra_trees", "transformer", "et_transformer_residual"]
    best_simple = main_comparison.loc[
        main_comparison["model"].isin(simple_models)
    ].sort_values("action_macro_driver_macro_mae_mean").iloc[0]["model"]
    best_learned = main_comparison.loc[
        main_comparison["model"].isin(learned_models)
    ].sort_values("action_macro_driver_macro_mae_mean").iloc[0]["model"]
    best_single = main_comparison.loc[
        main_comparison["model"].isin(["extra_trees", "transformer"])
    ].sort_values("action_macro_driver_macro_mae_mean").iloc[0]["model"]
    driver_wide["reference_simple_model"] = best_simple
    for model in learned_models:
        driver_wide[f"{model}_improvement_pct_vs_simple"] = (
            (driver_wide[best_simple] - driver_wide[model]) / driver_wide[best_simple] * 100.0
        )
        driver_wide[f"{model}_benefited_vs_simple"] = driver_wide[model] < driver_wide[best_simple]
    driver_wide["fusion_reference_single_model"] = best_single
    driver_wide["fusion_improvement_pct_vs_best_single"] = (
        (driver_wide[best_single] - driver_wide["et_transformer_residual"]) / driver_wide[best_single] * 100.0
    )
    driver_wide["fusion_benefited_vs_best_single"] = (
        driver_wide["et_transformer_residual"] < driver_wide[best_single]
    )
    driver_wide.to_csv(result_root / "driver_paired_results.csv", index=False, encoding="utf-8-sig")

    driver_benefit_rows = []
    for model in learned_models:
        driver_benefit_rows.append(
            {
                "model": model,
                "模型": MODEL_CN[model],
                "reference": best_simple,
                "drivers": len(driver_wide),
                "benefited_drivers": int(driver_wide[f"{model}_benefited_vs_simple"].sum()),
                "benefit_fraction": float(driver_wide[f"{model}_benefited_vs_simple"].mean()),
                "median_improvement_pct": float(
                    driver_wide[f"{model}_improvement_pct_vs_simple"].median()
                ),
            }
        )
    driver_benefit_rows.append(
        {
            "model": "et_transformer_residual_vs_best_single",
            "模型": "残差融合相对较强单模型",
            "reference": best_single,
            "drivers": len(driver_wide),
            "benefited_drivers": int(driver_wide["fusion_benefited_vs_best_single"].sum()),
            "benefit_fraction": float(driver_wide["fusion_benefited_vs_best_single"].mean()),
            "median_improvement_pct": float(driver_wide["fusion_improvement_pct_vs_best_single"].median()),
        }
    )
    driver_benefit = pd.DataFrame(driver_benefit_rows)
    driver_benefit.to_csv(result_root / "driver_benefit_summary.csv", index=False, encoding="utf-8-sig")

    examples = pd.read_csv(run_root / "tables" / "curve_example_selection.csv")
    averaged_examples = (
        example_predictions.groupby(
            ["example_id", "window_index", "model", "channel", "future_time_s"], as_index=False
        )
        .agg(truth=("truth", "first"), prediction=("prediction", "mean"), prediction_seed_sd=("prediction", "std"))
    )
    averaged_examples.to_csv(result_root / "curve_example_predictions.csv", index=False, encoding="utf-8-sig")
    colors = {
        "hold": "#777777",
        "linear": "#d95f02",
        "extra_trees": "#1b9e77",
        "transformer": "#7570b3",
        "et_transformer_residual": "#e7298a",
    }
    for example in examples.itertuples(index=False):
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for axis, channel in zip(axes.flat, TARGET_CHANNELS):
            subset = averaged_examples.loc[
                averaged_examples["example_id"].eq(example.example_id)
                & averaged_examples["channel"].eq(channel)
            ]
            truth = subset.loc[subset["model"].eq("hold")].sort_values("future_time_s")
            axis.plot(truth["future_time_s"], truth["truth"], color="black", linewidth=2.4, label="真实变化")
            for model in MODEL_NAMES:
                curve = subset.loc[subset["model"].eq(model)].sort_values("future_time_s")
                axis.plot(
                    curve["future_time_s"],
                    curve["prediction"],
                    color=colors[model],
                    linewidth=1.4,
                    label=MODEL_CN[model],
                )
            unit = "度" if channel == "steer_deg" else ("km/h" if channel == "speed_kmh" else "踏板比例")
            axis.set_title(f"{channel} 相对当前值（{unit}）")
            axis.grid(alpha=0.25)
        axes[1, 0].set_xlabel("未来时间（秒）")
        axes[1, 1].set_xlabel("未来时间（秒）")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        figure.suptitle(f"{example.example_id} · {example.population} · 三种子OOF均值")
        figure.tight_layout(rect=(0, 0.08, 1, 0.95))
        figure.savefig(figure_root / f"{example.example_id}_curves.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    main_index = main_comparison.set_index("model")
    best_simple_score = float(main_index.loc[best_simple, "action_macro_driver_macro_mae_mean"])
    best_learned_score = float(main_index.loc[best_learned, "action_macro_driver_macro_mae_mean"])
    improvement_pct = (best_simple_score - best_learned_score) / best_simple_score * 100.0
    benefit_fraction = float(
        driver_wide[f"{best_learned}_benefited_vs_simple"].mean()
    )

    pop_index = population_summary.set_index(["population", "model"])
    ordinary_simple = float(pop_index.loc[("ordinary", best_simple), "driver_macro_mean"])
    ordinary_learned = float(pop_index.loc[("ordinary", best_learned), "driver_macro_mean"])
    ordinary_worsening_pct = (ordinary_learned - ordinary_simple) / ordinary_simple * 100.0
    fixed_worsening = {}
    for population in ["distance_v2_305", "low_mu_v2_70"]:
        simple_value = float(pop_index.loc[(population, best_simple), "driver_macro_mean"])
        learned_value = float(pop_index.loc[(population, best_learned), "driver_macro_mean"])
        fixed_worsening[population] = (learned_value - simple_value) / simple_value * 100.0
    gates = {
        "all_improvement_ge_5pct": improvement_pct >= 5.0,
        "driver_benefit_ge_60pct": benefit_fraction >= 0.60,
        "ordinary_worsening_le_2pct": ordinary_worsening_pct <= 2.0,
        "distance_worsening_le_5pct": fixed_worsening["distance_v2_305"] <= 5.0,
        "low_mu_worsening_le_5pct": fixed_worsening["low_mu_v2_70"] <= 5.0,
    }
    personalize = all(gates.values())
    decision = {
        "best_simple_model": best_simple,
        "best_learned_model": best_learned,
        "best_single_learned_model": best_single,
        "all_continuous_improvement_pct": improvement_pct,
        "driver_benefit_fraction": benefit_fraction,
        "ordinary_worsening_pct": ordinary_worsening_pct,
        "distance_v2_worsening_pct": fixed_worsening["distance_v2_305"],
        "low_mu_v2_worsening_pct": fixed_worsening["low_mu_v2_70"],
        "gates": gates,
        "worth_entering_personalization": personalize,
        "verdict": "值得进入个体化阶段" if personalize else "暂缓进入个体化阶段",
    }
    (result_root / "personalization_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    compact_main = main_comparison[
        [
            "模型",
            "action_macro_driver_macro_mae_mean",
            "action_macro_driver_macro_mae_seed_sd",
            "action_macro_driver_macro_rmse_mean",
            "action_macro_driver_macro_endpoint_mae_mean",
            "action_macro_pooled_mae_mean",
            "action_macro_pooled_rmse_mean",
            "action_macro_pooled_endpoint_mae_mean",
            "steer_deg_driver_macro_mae_mean",
            "brake_driver_macro_mae_mean",
            "accelerator_driver_macro_mae_mean",
            "speed_kmh_driver_macro_mae_mean",
        ]
    ].rename(
        columns={
            "action_macro_driver_macro_mae_mean": "动作宏平均归一化MAE",
            "action_macro_driver_macro_mae_seed_sd": "MAE种子间SD",
            "action_macro_driver_macro_rmse_mean": "动作宏平均归一化RMSE",
            "action_macro_driver_macro_endpoint_mae_mean": "动作末点归一化MAE",
            "action_macro_pooled_mae_mean": "窗口pooled归一化MAE",
            "action_macro_pooled_rmse_mean": "窗口pooled归一化RMSE",
            "action_macro_pooled_endpoint_mae_mean": "窗口pooled末点归一化MAE",
            "steer_deg_driver_macro_mae_mean": "方向盘MAE_度",
            "brake_driver_macro_mae_mean": "制动MAE",
            "accelerator_driver_macro_mae_mean": "加速MAE",
            "speed_kmh_driver_macro_mae_mean": "车速辅助MAE_kmh",
        }
    )
    compact_population = population_summary[
        [
            "population",
            "模型",
            "driver_macro_mean",
            "driver_macro_rmse_mean",
            "driver_macro_endpoint_mae_mean",
            "driver_macro_seed_sd",
            "pooled_mean",
            "pooled_rmse_mean",
            "pooled_endpoint_mae_mean",
            "represented_drivers",
            "evaluated_windows",
        ]
    ].rename(
        columns={
            "population": "评价人口",
            "driver_macro_mean": "驾驶员宏平均MAE",
            "driver_macro_rmse_mean": "驾驶员宏平均RMSE",
            "driver_macro_endpoint_mae_mean": "驾驶员宏平均末点MAE",
            "driver_macro_seed_sd": "种子间SD",
            "pooled_mean": "窗口pooled MAE",
            "pooled_rmse_mean": "窗口pooled RMSE",
            "pooled_endpoint_mae_mean": "窗口pooled末点MAE",
            "represented_drivers": "驾驶员数",
            "evaluated_windows": "窗口或事件数",
        }
    )

    manifest = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    action_summary = pd.read_csv(run_root / "tables" / "action_change_summary.csv")
    dynamic_summary = pd.read_csv(run_root / "tables" / "dynamic_window_summary.csv")
    subject_population = pd.read_csv(run_root / "tables" / "population_by_subject_and_cohort.csv")
    recording_inventory = pd.read_csv(run_root / "tables" / "recording_inventory.csv")

    conclusion_text = f"""# 最终结论

本轮五模型统一比较已经完成。按预先冻结的三个种子、三折驾驶员隔离 OOF 和驾驶员宏平均主指标，较强简单基线是 **{MODEL_CN[best_simple]}**，最佳学习模型是 **{MODEL_CN[best_learned]}**。

- 全部连续窗口相对较强简单基线改善：`{improvement_pct:.2f}%`；要求至少 5%。
- 逐驾驶员获益比例：`{benefit_fraction:.2%}`；要求至少 60%。
- 普通窗口相对恶化：`{ordinary_worsening_pct:.2f}%`；上限 2%。
- 305 个距离触发相对恶化：`{fixed_worsening['distance_v2_305']:.2f}%`；上限 5%。
- 70 个低附着入口相对恶化：`{fixed_worsening['low_mu_v2_70']:.2f}%`；上限 5%。

最终判断：**{decision['verdict']}**。这一判断只来自 Run84 新连续人口，不继承旧事件结论，也不改写 Run57—Run83 历史结果。

## 停止规则逐项结果

{pd.DataFrame([{'检查项': key, '通过': value} for key, value in gates.items()]).to_markdown(index=False)}
"""
    (review_root / "00_FINAL_CONCLUSION_CN.md").write_text(conclusion_text, encoding="utf-8")

    data_population = f"""# 数据人口表

- 连续来源：`{manifest['recordings_total']}` 条 recording，其中 `{manifest['recordings_with_legal_windows']}` 条产生合法窗口。
- 驾驶员：`{manifest['subjects']}` 名。
- 20 Hz、0.2 秒查询步长的合法窗口：`{manifest['windows']}` 个。
- 固定评价：距离触发 305、低附着入口 70、Run57 V3 release 历史方向盘对照 2323。

## 动作变化与无变化

{format_table(action_summary, 4)}

## 高动态、动作已开始和普通窗口

{format_table(dynamic_summary, 4)}

## 驾驶员和批次分布

{format_table(subject_population, 0)}

## recording 纳入审计

所有 221 条 recording 均在清单中；`legal_windows=0` 的数量为 `{int((recording_inventory['legal_windows'] == 0).sum())}`。
"""
    (review_root / "01_DATA_POPULATION_CN.md").write_text(data_population, encoding="utf-8")
    (review_root / "02_MAIN_MODEL_COMPARISON_CN.md").write_text(
        "# 主要模型对比表\n\n主排序使用全部连续窗口的驾驶员宏平均三动作归一化曲线 MAE，数值越低越好。车速只作辅助输出。\n\n"
        + format_table(compact_main, 4)
        + "\n",
        encoding="utf-8",
    )
    (review_root / "03_EVALUATION_POPULATIONS_CN.md").write_text(
        "# 各评价人口对照表\n\n除 release 历史对照使用方向盘角 MAE（度）外，其余人口使用三动作阈值归一化 MAE。所有人口先固定，再读取未来真值评分。\n\n"
        + format_table(compact_population, 4)
        + "\n",
        encoding="utf-8",
    )
    (review_root / "04_DRIVER_PAIRED_SUMMARY_CN.md").write_text(
        "# 逐驾驶员配对汇总\n\n每名驾驶员先在各自全部 OOF 窗口求误差，再跨三个种子平均；获益表示同一驾驶员的误差低于全局选定的较强简单基线。\n\n"
        + format_table(driver_benefit, 4)
        + "\n\n逐驾驶员明细见 `results/driver_paired_results.csv`，块级配对明细见 `results/block_metrics.csv`。\n",
        encoding="utf-8",
    )
    script_inventory = pd.DataFrame(
        [
            {"脚本": "build_dataset.py", "作用": "恢复221条来源，20 Hz重采样，构建连续窗口、固定评价映射和人口表"},
            {"脚本": "experiment.py", "作用": "三种子三折驾驶员隔离OOF，训练五模型，汇总指标并生成曲线图"},
            {"脚本": "validate.py", "作用": "独立检查人口、模型、种子、泄漏、表格和Review-light完整性"},
            {"上游只读脚本": "02_code/tools/build_multiaction_reframe_audit.py", "作用": "恢复原始与8月车辆来源及统一通道"},
            {"上游只读脚本": "02_code/scripts/verify_run57_contract_invariants.py", "作用": "解释Run57 V3 P_full=2323历史对照人口"},
        ]
    )
    (review_root / "05_SCRIPT_INVENTORY_CN.md").write_text(
        "# 数据和评估脚本清单\n\n" + script_inventory.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    curve_links = "\n".join(
        [f"- `{row.example_id}`：`figures/{row.example_id}_curves.png`（{row.population}）" for row in examples.itertuples()]
    )
    (review_root / "06_CURVE_EXAMPLES_CN.md").write_text(
        "# 预测曲线示例\n\n示例在每个预设人口内按真实动作归一化幅值取中位窗口，不按任何模型误差挑选。曲线为三个种子的 OOF 预测均值。\n\n"
        + curve_links
        + "\n",
        encoding="utf-8",
    )

    output_manifest = {
        "status": "EXPERIMENT_COMPLETE_PENDING_INDEPENDENT_VALIDATION",
        "seeds": list(SEEDS),
        "folds_per_seed": N_FOLDS,
        "models": list(MODEL_NAMES),
        "windows": manifest["windows"],
        "best_simple_model": best_simple,
        "best_learned_model": best_learned,
        "personalization_verdict": decision["verdict"],
        "results": [str(path.relative_to(run_root)).replace("\\", "/") for path in sorted(result_root.glob("*"))],
        "review_light": [str(path.relative_to(run_root)).replace("\\", "/") for path in sorted(review_root.glob("*.md"))],
        "figures": [str(path.relative_to(run_root)).replace("\\", "/") for path in sorted(figure_root.glob("*.png"))],
        "commands": {
            "build": "py -3.11 build_dataset.py --project-root <PROJECT_ROOT> --august-root <AUGUST_ROOT> --output-root <RUN_ROOT>",
            "experiment": "py -3.11 experiment.py --run-root <RUN_ROOT> --device cuda",
            "validation": "py -3.11 validate.py --run-root <RUN_ROOT>",
        },
    }
    (run_root / "MANIFEST.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output_manifest, ensure_ascii=False, indent=2), flush=True)


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    if args.summarize_only:
        summarize(run_root)
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求CUDA训练，但当前PyTorch未检测到CUDA")
    device = torch.device(args.device)
    dataset_root = run_root / "dataset"
    dataset_manifest = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if dataset_manifest.get("causal_input_future_support_used") is not False:
        raise AssertionError("数据manifest未确认历史输入严格因果")
    history = np.load(dataset_root / "history_20hz.npy", mmap_mode="r")
    targets = np.load(dataset_root / "targets_relative_20hz.npy", mmap_mode="r")
    features = np.load(dataset_root / "extratrees_features.npy", mmap_mode="r")
    metadata = pd.read_csv(dataset_root / "window_metadata.csv", low_memory=False)
    fixed_mapping = pd.read_csv(run_root / "tables" / "fixed_evaluation_mapping.csv", low_memory=False)
    examples = choose_examples(metadata, targets, fixed_mapping)
    examples.to_csv(run_root / "tables" / "curve_example_selection.csv", index=False, encoding="utf-8-sig")
    if len(metadata) != len(history) or len(metadata) != len(targets) or len(metadata) != len(features):
        raise AssertionError("数据数组长度不一致")
    for seed in args.seeds:
        if seed not in SEEDS:
            raise ValueError(f"种子不在冻结合同中: {seed}")
        run_seed(
            run_root,
            seed,
            history,
            targets,
            features,
            metadata,
            fixed_mapping,
            examples,
            device,
            args.sanity,
        )
    if not args.sanity and all((run_root / "raw_results" / f"fold_metrics_seed{seed}.csv").exists() for seed in SEEDS):
        summarize(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
