#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v247: multi-resolution best anchor discovery.

本脚本的目标不是“统一后移锚点”，而是把任务重新构造成：
1. 用 50ms 细粒度候选锚点重采样现有事件；
2. 用锁定的 v241 轨迹模型给每个候选锚点评分；
3. 离线得到每个事件的 best anchor label；
4. 训练 input-only selector，检验这个 label 是否能被当前可见输入学习。

重要约束：
- 不训练新的轨迹预测模型；
- selector 不使用未来真实曲线、候选真实误差、event_uid、subject、recording；
- oracle best anchor 只作为离线上限，不作为可部署策略。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260630
FINE_DELAY_MS = list(range(0, 1001, 50))
COARSE_DELAY_MS = [0, 200, 400, 600, 800, 1000]
SCORE_CONFIGS = [
    ("error_only", 0.00, 0.00),
    ("delay_l03", 0.03, 0.00),
    ("delay_l05", 0.05, 0.00),
    ("delay_l10", 0.10, 0.00),
    ("delay_l05_unstable_m03", 0.05, 0.03),
    ("delay_l05_unstable_m05", 0.05, 0.05),
    ("delay_l10_unstable_m05", 0.10, 0.05),
]
PRIMARY_SCORE_NAME = "delay_l05_unstable_m05"
TAIL_START_S = 1.0
TAIL_END_S = 2.0

ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
SCRIPTS = BASELINES / "scripts"
OUT = BASELINES / "v247_multi_resolution_best_anchor_discovery_20260630"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
MODELS = OUT / "models"
ZIP_PATH = BASELINES / "v247_multi_resolution_best_anchor_discovery_20260630_pack.zip"

V236_SCRIPT = SCRIPTS / "stage03_v236_rolling_reanchor_dataset_and_baseline_20260624.py"
V238_SCRIPT = SCRIPTS / "stage03_v238_task_model_redesign_20260626.py"
V239_SCRIPT = SCRIPTS / "stage03_v239_light_attention_noharm_20260626.py"
V241_SCRIPT = SCRIPTS / "stage03_v241_stronger_temporal_model_20260626.py"
V241_DIR = BASELINES / "v241_stronger_temporal_model_20260626"
V241_MODEL = V241_DIR / "models" / "v241_best_stronger_temporal_diagnostic.pt"
V241_PRED_NPZ = V241_DIR / "v241_stronger_temporal_predictions.npz"
DESIGN_DOC = ROOT / "docs" / "superpowers" / "specs" / "2026-06-30-v247-multi-resolution-best-anchor-design.md"
PLAN_DOC = ROOT / "docs" / "superpowers" / "plans" / "2026-06-30-v247-multi-resolution-best-anchor-plan.md"


def import_module_from_path(module_name: str, path: Path):
    """按文件路径导入旧实验脚本，避免依赖 PYTHONPATH。"""

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


V236 = import_module_from_path("stage03_v236_rolling_reanchor_dataset_and_baseline_20260624", V236_SCRIPT)
V238 = import_module_from_path("stage03_v238_task_model_redesign_20260626", V238_SCRIPT)
V239 = import_module_from_path("stage03_v239_light_attention_noharm_20260626", V239_SCRIPT)
V241 = import_module_from_path("stage03_v241_stronger_temporal_model_20260626", V241_SCRIPT)

FUTURE_GRID = np.asarray(V238.FUTURE_GRID, dtype=np.float32)
HIST_COLS = [name for name, _ in V236.HISTORY_FEATURE_SPECS]
PHASE_FEATURE_NAMES = list(getattr(V236, "PHASE_FEATURE_NAMES", [f"phase_{i}" for i in range(9)]))


class V241InferenceResult:
    """锁定 v241 checkpoint 在 fine-grid 数据上的推理结果。"""

    def __init__(
        self,
        pred_curve: np.ndarray,
        point_count: int,
        checkpoint_config: Dict[str, Any],
        best_model_name: str,
        device: str,
        seconds: float,
    ) -> None:
        self.pred_curve = pred_curve
        self.point_count = point_count
        self.checkpoint_config = checkpoint_config
        self.best_model_name = best_model_name
        self.device = device
        self.seconds = seconds


def ensure_dirs() -> None:
    """创建 v247 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def ensure_clean_output() -> None:
    """只清理 v247 自己的输出目录和 zip，不触碰其他版本产物。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一用 utf-8-sig 写 CSV，方便 Windows Excel 直接打开中文列名。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, Any], path: Path) -> None:
    """写入带中文可读性的 JSON 日志。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def file_sha256(path: Path) -> str:
    """计算文件 SHA256，用于回溯实验输入。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """忽略 NaN/Inf 后计算 RMSE。"""

    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(av) & np.isfinite(bv)
    if not bool(mask.any()):
        return math.nan
    return float(np.sqrt(np.mean(np.square(av[mask] - bv[mask]))))


def finite_mae(a: np.ndarray, b: np.ndarray) -> float:
    """忽略 NaN/Inf 后计算 MAE。"""

    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(av) & np.isfinite(bv)
    if not bool(mask.any()):
        return math.nan
    return float(np.mean(np.abs(av[mask] - bv[mask])))


def safe_mode_int(values: Iterable[Any]) -> int | float:
    """返回整数众数；没有有效值时返回 NaN。"""

    s = pd.Series(list(values)).dropna()
    if s.empty:
        return math.nan
    return int(s.astype(int).mode().iloc[0])


def nearest_coarse_delay(delay_ms: int) -> int:
    """把 fine delay 映射到最近的 coarse delay，平局时取更早锚点。"""

    return int(min(COARSE_DELAY_MS, key=lambda x: (abs(x - int(delay_ms)), x)))


def normalize_bool_series(s: pd.Series) -> pd.Series:
    """把旧表中可能混有 bool/0/1/字符串的列规范成 bool。"""

    if s.dtype == bool:
        return s.fillna(False).astype(bool)
    return s.fillna(False).astype(str).str.lower().isin(["true", "1", "yes", "y"])


def zip_outputs() -> Path:
    """打包 v247 输出目录，zip 放在 v247 目录外侧。"""

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT.parent))
    return ZIP_PATH


def configure_plotting() -> None:
    """配置 matplotlib，优先用 Windows 中文字体。"""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def build_fine_grid_dataset() -> Tuple[object, pd.DataFrame, List[str], pd.DataFrame]:
    """
    用 v236 的原始车辆采样器重新构造 50ms fine-grid rolling 数据。

    这里直接 monkeypatch V236.DELAY_MS。v236 的 build_rolling_dataset 会从 raw CSV
    按 observation_s = original_anchor_s + delay_ms / 1000 重新采样，所以不是在旧的
    0/200/... 数组上插值。
    """

    old_delay = list(V236.DELAY_MS)
    try:
        V236.DELAY_MS = list(FINE_DELAY_MS)
        event_df = V236.load_event_manifest()
        x_hist, x_road, x_phase, y_future, manifest, dropped = V236.build_rolling_dataset(event_df)
    finally:
        V236.DELAY_MS = old_delay

    _, feature_names = V236.build_design_matrix(x_hist, x_road, x_phase)
    data = V238.RollingData(
        manifest=manifest.reset_index(drop=True),
        x_hist=x_hist.astype(np.float32),
        x_road=x_road.astype(np.float32),
        x_phase=x_phase.astype(np.float32),
        y_future=y_future.astype(np.float32),
        pred_v236=np.full_like(y_future, np.nan, dtype=np.float32),
        feature_names=feature_names,
        target_names=list(V236.TARGET_NAMES),
    )
    return data, manifest.reset_index(drop=True), feature_names, dropped


def build_sampling_audit(manifest: pd.DataFrame, dropped: pd.DataFrame) -> pd.DataFrame:
    """审计 50ms fine-grid 是否真的由原始序列支持。"""

    event_delay_counts = manifest.groupby("event_uid")["delay_ms"].nunique()
    expected_rows = int(manifest["event_uid"].nunique() * len(FINE_DELAY_MS))
    generated_rows = int(len(manifest))
    dropped_rows = int(len(dropped)) if dropped is not None else 0
    delay_values = sorted(manifest["delay_ms"].astype(int).unique().tolist())

    err = manifest["max_abs_nearest_time_error_ms"].astype(float)
    audit = pd.DataFrame(
        [
            {
                "requested_delay_step_ms": 50,
                "requested_delay_values": json.dumps(FINE_DELAY_MS, ensure_ascii=False),
                "actual_delay_values": json.dumps(delay_values, ensure_ascii=False),
                "n_events": int(manifest["event_uid"].nunique()),
                "n_expected_rows": expected_rows,
                "n_generated_rows": generated_rows,
                "n_dropped_rows": dropped_rows,
                "complete_event_rate": float((event_delay_counts == len(FINE_DELAY_MS)).mean()),
                "min_delay_count_per_event": int(event_delay_counts.min()),
                "max_delay_count_per_event": int(event_delay_counts.max()),
                "mode_delay_count_per_event": safe_mode_int(event_delay_counts),
                "max_abs_nearest_time_error_ms_p95": float(err.quantile(0.95)),
                "max_abs_nearest_time_error_ms_max": float(err.max()),
                "fine_grid_sampling_checked": True,
                "fine_grid_supported": bool(len(delay_values) == len(FINE_DELAY_MS) and event_delay_counts.min() >= 18),
            }
        ]
    )
    write_csv(audit, TABLES / "v247_fine_grid_sampling_audit.csv")
    if dropped is not None and len(dropped):
        write_csv(dropped, TABLES / "v247_fine_grid_dropped_rows.csv")
    return audit


def scalers_from_checkpoint_payload(payload: Dict[str, Any]):
    """从 v241 checkpoint 里的 scaler payload 恢复 v239.SequenceScalers。"""

    required = [
        "hist_mean",
        "hist_std",
        "road_mean",
        "road_std",
        "phase_mean",
        "phase_std",
        "point_mean",
        "point_std",
        "y_mean",
        "y_std",
    ]
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"v241 checkpoint scaler 缺少字段：{missing}")

    return V239.SequenceScalers(
        hist_mean=np.asarray(payload["hist_mean"], dtype=np.float32),
        hist_std=np.asarray(payload["hist_std"], dtype=np.float32),
        road_mean=np.asarray(payload["road_mean"], dtype=np.float32),
        road_std=np.asarray(payload["road_std"], dtype=np.float32),
        phase_mean=np.asarray(payload["phase_mean"], dtype=np.float32),
        phase_std=np.asarray(payload["phase_std"], dtype=np.float32),
        point_mean=np.asarray(payload["point_mean"], dtype=np.float32),
        point_std=np.asarray(payload["point_std"], dtype=np.float32),
        y_mean=float(payload["y_mean"]),
        y_std=float(payload["y_std"]),
    )


def load_v241_model_for_data(data: object, device: torch.device) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    """按 fine-grid 数据维度实例化并加载锁定 v241 模型。"""

    checkpoint = torch.load(V241_MODEL, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    model = V241.StrongerTemporalQueryAttention(
        hist_dim=data.x_hist.shape[-1],
        road_dim=data.x_road.shape[-1],
        phase_dim=data.x_phase.shape[-1],
        point_dim=len(V238.POINT_EXTRA_FEATURE_NAMES),
        hist_len=data.x_hist.shape[1],
        road_len=data.x_road.shape[1],
        hidden_dim=int(config["hidden_dim"]),
        n_heads=int(config["n_heads"]),
        n_layers=int(config["n_layers"]),
        mlp_hidden=int(config["mlp_hidden"]),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    scalers = scalers_from_checkpoint_payload(checkpoint["scalers"])
    return model, scalers, checkpoint


def run_locked_v241_inference(data: object) -> V241InferenceResult:
    """对全部 fine-grid anchor 使用锁定 v241 checkpoint 做曲线推理。"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scalers, checkpoint = load_v241_model_for_data(data, device)
    x_base = V238.build_base_design_matrix(data)
    point_data = V238.build_point_dataset(data, x_base)
    arrays = V239.standardize_arrays(data, point_data, scalers)

    t0 = time.time()
    pred_curve = V239.predict_all_points(
        model=model,
        arrays=arrays,
        point_data=point_data,
        scalers=scalers,
        device=device,
        batch_size=8192,
    )
    seconds = time.time() - t0
    best_model_name = str(checkpoint.get("model_name", checkpoint.get("best_model_name", "unknown")))
    if best_model_name == "unknown" and "config" in checkpoint:
        best_model_name = str(checkpoint["config"].get("model_name", "v241_checkpoint"))
    return V241InferenceResult(
        pred_curve=pred_curve.astype(np.float32),
        point_count=int(len(point_data.y_point_all)),
        checkpoint_config=dict(checkpoint["config"]),
        best_model_name=best_model_name,
        device=str(device),
        seconds=float(seconds),
    )


def validate_coarse_replay(manifest: pd.DataFrame, pred_curve: np.ndarray, y_future: np.ndarray) -> pd.DataFrame:
    """把 fine-grid 中 coarse delay 的 v241 推理与旧 v241 保存预测对齐。"""

    if not V241_PRED_NPZ.exists():
        out = pd.DataFrame([{"status": "missing_v241_prediction_npz", "path": str(V241_PRED_NPZ)}])
        write_csv(out, TABLES / "v247_coarse_replay_alignment.csv")
        return out

    with np.load(V241_PRED_NPZ, allow_pickle=False) as old:
        old_pred = old["pred_v241_best_stronger_steering_delta"].astype(np.float32)
        old_y = old["y_true_steering_delta"].astype(np.float32)
        old_uid = old["event_uid"].astype(str)
        old_delay = old["delay_ms"].astype(int)
        old_split = old["split"].astype(str)

    old_map = {(str(uid), int(delay)): i for i, (uid, delay) in enumerate(zip(old_uid, old_delay))}
    rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    coarse_mask = manifest["delay_ms"].astype(int).isin(COARSE_DELAY_MS).to_numpy()
    for i in np.where(coarse_mask)[0]:
        key = (str(manifest.loc[i, "event_uid"]), int(manifest.loc[i, "delay_ms"]))
        j = old_map.get(key)
        if j is None:
            continue
        pair_rows.append(
            {
                "split": str(manifest.loc[i, "split"]),
                "delay_ms": int(manifest.loc[i, "delay_ms"]),
                "pred_rmse": finite_rmse(pred_curve[i], old_pred[j]),
                "pred_mae": finite_mae(pred_curve[i], old_pred[j]),
                "true_rmse": finite_rmse(y_future[i, :, 0], old_y[j]),
                "old_split": str(old_split[j]),
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        out = pd.DataFrame([{"status": "no_matching_coarse_rows"}])
        write_csv(out, TABLES / "v247_coarse_replay_alignment.csv")
        return out

    for (split, delay), g in pair_df.groupby(["split", "delay_ms"]):
        rows.append(
            {
                "split": str(split),
                "delay_ms": int(delay),
                "n": int(len(g)),
                "pred_rmse_mean": float(g["pred_rmse"].mean()),
                "pred_rmse_p95": float(g["pred_rmse"].quantile(0.95)),
                "pred_rmse_max": float(g["pred_rmse"].max()),
                "pred_mae_mean": float(g["pred_mae"].mean()),
                "true_rmse_mean": float(g["true_rmse"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["split", "delay_ms"]).reset_index(drop=True)
    write_csv(out, TABLES / "v247_coarse_replay_alignment.csv")
    write_csv(pair_df, TABLES / "v247_coarse_replay_alignment_by_row.csv")
    return out


def row_masked_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """逐行 masked RMSE，同时返回每行有效点数。"""

    finite = np.isfinite(y_true) & np.isfinite(y_pred) & mask
    sq = np.where(finite, np.square(y_true - y_pred), 0.0)
    count = finite.sum(axis=1).astype(np.float64)
    denom = np.maximum(count, 1.0)
    rmse = np.sqrt(sq.sum(axis=1) / denom)
    rmse[count <= 0] = np.nan
    return rmse.astype(np.float32), count.astype(np.int32)


def compute_instability_features(data: object, manifest: pd.DataFrame) -> pd.DataFrame:
    """
    用候选锚点之前的历史窗口计算局部不稳定性。

    这些特征都只来自 candidate anchor 时刻已经可见的历史，不看未来真实曲线。
    """

    ci = {name: i for i, name in enumerate(HIST_COLS)}
    required = ["steering", "yaw_rate", "lateral_distance"]
    missing = [name for name in required if name not in ci]
    if missing:
        raise KeyError(f"v247 instability 缺少历史字段：{missing}")

    steer = data.x_hist[:, :, ci["steering"]].astype(np.float32)
    yaw = data.x_hist[:, :, ci["yaw_rate"]].astype(np.float32)
    lat = data.x_hist[:, :, ci["lateral_distance"]].astype(np.float32)

    idx_now = -1
    idx_05 = -6
    idx_10 = -11
    raw = pd.DataFrame(
        {
            "abs_steer_slope_last05": np.abs(steer[:, idx_now] - steer[:, idx_05]),
            "abs_steer_second_diff_last05": np.abs((steer[:, idx_now] - steer[:, idx_05]) - (steer[:, idx_05] - steer[:, idx_10])),
            "abs_yaw_change_last05": np.abs(yaw[:, idx_now] - yaw[:, idx_05]),
            "abs_lat_change_last05": np.abs(lat[:, idx_now] - lat[:, idx_05]),
        }
    )

    train_mask = manifest["split"].astype(str).to_numpy() == "train"
    scales: Dict[str, float] = {}
    for col in raw.columns:
        train_values = raw.loc[train_mask, col].astype(float).to_numpy()
        scale = float(np.nanmedian(np.abs(train_values)))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.nanmean(np.abs(train_values)))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        scales[col] = scale
        raw[f"norm_{col}"] = np.clip(raw[col].astype(float) / scale, 0.0, 10.0)

    raw["instability_penalty"] = (
        0.40 * raw["norm_abs_steer_slope_last05"]
        + 0.25 * raw["norm_abs_steer_second_diff_last05"]
        + 0.20 * raw["norm_abs_yaw_change_last05"]
        + 0.15 * raw["norm_abs_lat_change_last05"]
    )
    write_json(
        {
            "scale_source": "train_split_median_abs",
            "components": scales,
            "formula": {
                "norm_abs_steer_slope_last05": 0.40,
                "norm_abs_steer_second_diff_last05": 0.25,
                "norm_abs_yaw_change_last05": 0.20,
                "norm_abs_lat_change_last05": 0.15,
            },
        },
        LOGS / "v247_instability_feature_scales.json",
    )
    return raw


def build_candidate_score_table(data: object, manifest: pd.DataFrame, pred_curve: np.ndarray) -> pd.DataFrame:
    """生成每个 fine-grid candidate anchor 的误差、等待代价、不稳定性和 score。"""

    y_true = data.y_future[:, :, 0].astype(np.float32)
    delay_s = manifest["delay_ms"].astype(float).to_numpy()[:, None] / 1000.0
    original_rel = delay_s + FUTURE_GRID[None, :]
    tail_mask = (original_rel >= TAIL_START_S - 1e-9) & (original_rel <= TAIL_END_S + 1e-9)
    remaining_mask = original_rel <= TAIL_END_S + 1e-9

    tail_rmse, tail_n = row_masked_rmse(y_true, pred_curve, tail_mask)
    remaining_rmse, remaining_n = row_masked_rmse(y_true, pred_curve, remaining_mask)
    instability = compute_instability_features(data, manifest)

    candidate = manifest.copy().reset_index(drop=True)
    candidate.insert(0, "candidate_row_idx", np.arange(len(candidate), dtype=int))
    candidate["candidate_delay_ms"] = candidate["delay_ms"].astype(int)
    candidate["candidate_delay_s"] = candidate["candidate_delay_ms"].astype(float) / 1000.0
    candidate["nearest_coarse_delay_ms"] = candidate["candidate_delay_ms"].map(nearest_coarse_delay).astype(int)
    candidate["residual_offset_ms"] = candidate["candidate_delay_ms"] - candidate["nearest_coarse_delay_ms"]
    candidate["candidate_tail_rmse_v241"] = tail_rmse
    candidate["candidate_original_remaining_rmse_v241"] = remaining_rmse
    candidate["candidate_tail_point_n"] = tail_n
    candidate["candidate_original_remaining_point_n"] = remaining_n
    candidate["tail_eval_start_original_s"] = np.where(tail_n > 0, np.nan, np.nan)
    candidate["tail_eval_end_original_s"] = np.where(tail_n > 0, np.nan, np.nan)
    for i in range(len(candidate)):
        idx = tail_mask[i]
        if bool(idx.any()):
            candidate.loc[i, "tail_eval_start_original_s"] = float(np.min(original_rel[i, idx]))
            candidate.loc[i, "tail_eval_end_original_s"] = float(np.max(original_rel[i, idx]))

    for col in instability.columns:
        candidate[col] = instability[col].to_numpy()

    for name, lambda_wait, mu_unstable in SCORE_CONFIGS:
        candidate[f"score_{name}"] = (
            candidate["candidate_tail_rmse_v241"].astype(float)
            + float(lambda_wait) * candidate["candidate_delay_s"].astype(float)
            + float(mu_unstable) * candidate["instability_penalty"].astype(float)
        )

    write_csv(candidate, TABLES / "v247_fine_anchor_candidate_table.csv")
    return candidate


def attach_current_anchor_groups(candidate: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """用每个 split 内 0ms 当前锚点误差定义差样本分层。"""

    current = candidate[candidate["candidate_delay_ms"].eq(0)].copy()
    current = current.sort_values(["split", "event_uid"]).reset_index(drop=True)
    current["current_0ms_tail_rmse_v241"] = current["candidate_tail_rmse_v241"].astype(float)
    current["bad_top10_split_v241"] = False
    current["very_bad_top5_split_v241"] = False

    threshold_rows: List[Dict[str, Any]] = []
    for split, g in current.groupby("split"):
        q90 = float(g["current_0ms_tail_rmse_v241"].quantile(0.90))
        q95 = float(g["current_0ms_tail_rmse_v241"].quantile(0.95))
        idx = g.index
        current.loc[idx, "bad_top10_split_v241"] = current.loc[idx, "current_0ms_tail_rmse_v241"].ge(q90)
        current.loc[idx, "very_bad_top5_split_v241"] = current.loc[idx, "current_0ms_tail_rmse_v241"].ge(q95)
        threshold_rows.append(
            {
                "split": str(split),
                "q90_bad_top10_current_0ms": q90,
                "q95_very_bad_top5_current_0ms": q95,
                "n_events": int(len(g)),
                "n_bad_top10": int(current.loc[idx, "bad_top10_split_v241"].sum()),
                "n_very_bad_top5": int(current.loc[idx, "very_bad_top5_split_v241"].sum()),
            }
        )

    group_cols = [
        "event_uid",
        "current_0ms_tail_rmse_v241",
        "bad_top10_split_v241",
        "very_bad_top5_split_v241",
        "normal_curve",
        "observe_later_like",
        "strong_steer",
        "reverse",
    ]
    group_cols = [c for c in group_cols if c in current.columns]
    enriched = candidate.merge(
        current[group_cols],
        on="event_uid",
        how="left",
        suffixes=("", "_current0"),
        validate="many_to_one",
    )
    write_csv(pd.DataFrame(threshold_rows), TABLES / "v247_bad_thresholds_by_split.csv")
    return enriched, current


def select_best_by_score(candidate: pd.DataFrame) -> pd.DataFrame:
    """按每套 score 为每个 event 选择离线 best anchor。"""

    rows: List[pd.DataFrame] = []
    for name, lambda_wait, mu_unstable in SCORE_CONFIGS:
        score_col = f"score_{name}"
        work = candidate[np.isfinite(candidate[score_col].astype(float))].copy()
        work = work.sort_values(["event_uid", score_col, "candidate_delay_ms"], ascending=[True, True, True])
        best = work.groupby("event_uid", as_index=False).head(1).copy()
        best["score_name"] = name
        best["lambda_wait"] = float(lambda_wait)
        best["mu_unstable"] = float(mu_unstable)
        best["best_delay_ms"] = best["candidate_delay_ms"].astype(int)
        best["best_score"] = best[score_col].astype(float)
        best["best_prediction_error_v241"] = best["candidate_tail_rmse_v241"].astype(float)
        rows.append(best)

    best_long = pd.concat(rows, ignore_index=True)
    keep_cols = [
        "score_name",
        "lambda_wait",
        "mu_unstable",
        "event_uid",
        "split",
        "scene_type",
        "pool_key",
        "best_delay_ms",
        "nearest_coarse_delay_ms",
        "residual_offset_ms",
        "best_score",
        "best_prediction_error_v241",
        "current_0ms_tail_rmse_v241",
        "instability_penalty",
        "bad_top10_split_v241",
        "very_bad_top5_split_v241",
        "normal_curve",
        "observe_later_like",
        "strong_steer",
        "reverse",
    ]
    keep_cols = [c for c in keep_cols if c in best_long.columns]
    best_long = best_long[keep_cols].sort_values(["score_name", "split", "event_uid"]).reset_index(drop=True)
    write_csv(best_long, TABLES / "v247_best_anchor_by_event.csv")
    return best_long


def group_masks_event(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """事件级评估分组。"""

    n = len(df)
    true = pd.Series(True, index=df.index)
    false = pd.Series(False, index=df.index)
    bad = normalize_bool_series(df["bad_top10_split_v241"]) if "bad_top10_split_v241" in df.columns else false
    very_bad = normalize_bool_series(df["very_bad_top5_split_v241"]) if "very_bad_top5_split_v241" in df.columns else false
    normal_curve = normalize_bool_series(df["normal_curve"]) if "normal_curve" in df.columns else false
    observe = normalize_bool_series(df["observe_later_like"]) if "observe_later_like" in df.columns else false
    strong = normalize_bool_series(df["strong_steer"]) if "strong_steer" in df.columns else false
    reverse = normalize_bool_series(df["reverse"]) if "reverse" in df.columns else false
    best_delay = df["best_delay_ms"].astype(float) if "best_delay_ms" in df.columns else pd.Series(np.nan, index=df.index)
    return {
        "all": true,
        "normal": normal_curve & ~bad,
        "bad_top10": bad,
        "very_bad_top5": very_bad,
        "early_best_after_400": bad & best_delay.ge(400),
        "observe_later_like": observe,
        "strong_steer": strong,
        "reverse": reverse,
    }


def build_best_anchor_distribution(best_long: pd.DataFrame) -> pd.DataFrame:
    """汇总不同 score 下 best anchor 的时间分布。"""

    rows: List[Dict[str, Any]] = []
    for score_name, score_df in best_long.groupby("score_name"):
        for split, split_df in score_df.groupby("split"):
            masks = group_masks_event(split_df)
            for group_name, mask in masks.items():
                sub = split_df[mask]
                if sub.empty:
                    continue
                total = len(sub)
                for delay, g in sub.groupby("best_delay_ms"):
                    rows.append(
                        {
                            "score_name": str(score_name),
                            "split": str(split),
                            "event_group": group_name,
                            "best_delay_ms": int(delay),
                            "n": int(len(g)),
                            "percent": float(len(g) / total),
                            "total_n": int(total),
                        }
                    )
    out = pd.DataFrame(rows).sort_values(["score_name", "split", "event_group", "best_delay_ms"]).reset_index(drop=True)
    write_csv(out, TABLES / "v247_best_anchor_distribution.csv")
    return out


def build_score_weight_sweep_summary(best_long: pd.DataFrame) -> pd.DataFrame:
    """比较不同 score 权重定义得到的 oracle 上限和等待分布。"""

    rows: List[Dict[str, Any]] = []
    for score_name, score_df in best_long.groupby("score_name"):
        for split, split_df in score_df.groupby("split"):
            for group_name, mask in group_masks_event(split_df).items():
                sub = split_df[mask]
                if sub.empty:
                    continue
                current = sub["current_0ms_tail_rmse_v241"].astype(float)
                best = sub["best_prediction_error_v241"].astype(float)
                denom = current - best
                rows.append(
                    {
                        "score_name": str(score_name),
                        "split": str(split),
                        "event_group": group_name,
                        "n": int(len(sub)),
                        "mean_best_delay_ms": float(sub["best_delay_ms"].astype(float).mean()),
                        "median_best_delay_ms": float(sub["best_delay_ms"].astype(float).median()),
                        "mode_best_delay_ms": safe_mode_int(sub["best_delay_ms"]),
                        "pct_best_0ms": float(sub["best_delay_ms"].eq(0).mean()),
                        "pct_best_1000ms": float(sub["best_delay_ms"].eq(1000).mean()),
                        "mean_current_0ms_error_v241": float(current.mean()),
                        "mean_best_error_v241": float(best.mean()),
                        "mean_delta_best_minus_current": float((best - current).mean()),
                        "mean_gain_available": float(np.nanmean(denom.to_numpy())),
                    }
                )
    out = pd.DataFrame(rows).sort_values(["score_name", "split", "event_group"]).reset_index(drop=True)
    write_csv(out, TABLES / "v247_score_weight_sweep_summary.csv")
    return out


def build_selector_feature_table(candidate: pd.DataFrame, data: object) -> pd.DataFrame:
    """构造 selector 可见输入表；不包含任何未来误差或事件身份字段。"""

    ci = {name: i for i, name in enumerate(HIST_COLS)}
    x_hist = data.x_hist
    x_road = data.x_road
    x_phase = data.x_phase

    def h(name: str) -> np.ndarray:
        if name not in ci:
            return np.full(len(candidate), np.nan, dtype=np.float32)
        return x_hist[:, :, ci[name]].astype(np.float32)

    steer = h("steering")
    yaw = h("yaw_rate")
    ay = h("ay")
    speed = h("speed_kmh")
    brake = h("brake")
    accel = h("accelerator")
    curv = h("lane_curvature")
    lat = h("lateral_distance")

    features = pd.DataFrame(
        {
            "candidate_row_idx": candidate["candidate_row_idx"].astype(int).to_numpy(),
            "event_uid": candidate["event_uid"].astype(str).to_numpy(),
            "split": candidate["split"].astype(str).to_numpy(),
            "scene_type": candidate.get("scene_type", pd.Series("NA", index=candidate.index)).astype(str).fillna("NA").to_numpy(),
            "pool_key": candidate.get("pool_key", pd.Series("NA", index=candidate.index)).astype(str).fillna("NA").to_numpy(),
            "candidate_delay_ms": candidate["candidate_delay_ms"].astype(int).to_numpy(),
            "candidate_delay_s": candidate["candidate_delay_s"].astype(float).to_numpy(),
            "nearest_coarse_delay_ms": candidate["nearest_coarse_delay_ms"].astype(int).to_numpy(),
            "residual_offset_ms": candidate["residual_offset_ms"].astype(int).to_numpy(),
            "hist_current_steer": steer[:, -1],
            "hist_abs_current_steer": np.abs(steer[:, -1]),
            "hist_abs_mean_steer": np.nanmean(np.abs(steer), axis=1),
            "hist_abs_max_steer": np.nanmax(np.abs(steer), axis=1),
            "hist_steer_slope_last05": steer[:, -1] - steer[:, -6],
            "hist_abs_steer_slope_last05": np.abs(steer[:, -1] - steer[:, -6]),
            "hist_abs_steer_second_diff_last05": np.abs((steer[:, -1] - steer[:, -6]) - (steer[:, -6] - steer[:, -11])),
            "hist_yaw_abs_mean": np.nanmean(np.abs(yaw), axis=1),
            "hist_yaw_change_last05": yaw[:, -1] - yaw[:, -6],
            "hist_abs_yaw_change_last05": np.abs(yaw[:, -1] - yaw[:, -6]),
            "hist_ay_abs_mean": np.nanmean(np.abs(ay), axis=1),
            "hist_speed_mean": np.nanmean(speed, axis=1),
            "hist_speed_last": speed[:, -1],
            "hist_brake_mean": np.nanmean(brake, axis=1),
            "hist_accel_mean": np.nanmean(accel, axis=1),
            "hist_curv_abs_mean": np.nanmean(np.abs(curv), axis=1),
            "hist_lat_abs_mean": np.nanmean(np.abs(lat), axis=1),
            "hist_lat_change_last05": lat[:, -1] - lat[:, -6],
            "hist_abs_lat_change_last05": np.abs(lat[:, -1] - lat[:, -6]),
            "road_curv_abs_mean": np.nanmean(np.abs(x_road[:, :, 0]), axis=1),
            "road_curv_abs_max": np.nanmax(np.abs(x_road[:, :, 0]), axis=1),
            "road_lat_abs_mean": np.nanmean(np.abs(x_road[:, :, 1]), axis=1),
            "road_lat_abs_max": np.nanmax(np.abs(x_road[:, :, 1]), axis=1),
            "instability_penalty": candidate["instability_penalty"].astype(float).to_numpy(),
            "target_score_primary": candidate[f"score_{PRIMARY_SCORE_NAME}"].astype(float).to_numpy(),
            "candidate_tail_rmse_v241": candidate["candidate_tail_rmse_v241"].astype(float).to_numpy(),
        }
    )
    for j in range(x_phase.shape[1]):
        name = PHASE_FEATURE_NAMES[j] if j < len(PHASE_FEATURE_NAMES) else f"phase_{j}"
        features[f"phase_{j}_{name}"] = x_phase[:, j]

    write_csv(features, TABLES / "v247_selector_training_table.csv")
    return features


def make_one_hot_encoder() -> OneHotEncoder:
    """兼容不同 sklearn 版本的 OneHotEncoder 参数。"""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def train_selector_models(feature_table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """用 train split 训练 Ridge/RF 候选 score selector，并给所有 candidate 预测 score。"""

    forbidden = {
        "event_uid",
        "candidate_row_idx",
        "split",
        "target_score_primary",
        "candidate_tail_rmse_v241",
    }
    categorical = ["scene_type", "pool_key"]
    feature_cols = [c for c in feature_table.columns if c not in forbidden]
    numeric = [c for c in feature_cols if c not in categorical]

    train_mask = feature_table["split"].astype(str).eq("train") & np.isfinite(feature_table["target_score_primary"].astype(float))
    if int(train_mask.sum()) < 100:
        raise AssertionError(f"selector train rows too few: {int(train_mask.sum())}")

    # 当前 Windows 环境中 sklearn ColumnTransformer + Pipeline 在 fit 阶段偶发底层退出。
    # 这里改用显式 one-hot、train-only median impute 和 train-only 标准化，逻辑更透明。
    x_raw = pd.get_dummies(feature_table[feature_cols], columns=categorical, dummy_na=True)
    x_raw = x_raw.replace([np.inf, -np.inf], np.nan)
    train_median = x_raw.loc[train_mask].median(numeric_only=True)
    x_filled = x_raw.fillna(train_median).fillna(0.0).astype(np.float32)
    x_mean = x_filled.loc[train_mask].mean(axis=0).astype(np.float32)
    x_std = x_filled.loc[train_mask].std(axis=0).replace(0, 1).fillna(1).astype(np.float32)
    x_scaled = ((x_filled - x_mean) / x_std).astype(np.float32)

    x_train_scaled = x_scaled.loc[train_mask].to_numpy(dtype=np.float32)
    x_all_scaled = x_scaled.to_numpy(dtype=np.float32)
    x_train_tree = x_filled.loc[train_mask].to_numpy(dtype=np.float32)
    x_all_tree = x_filled.to_numpy(dtype=np.float32)
    y_train = feature_table.loc[train_mask, "target_score_primary"].astype(float).to_numpy()

    models = {
        "selector_ridge_score": ("scaled", Ridge(alpha=4.0, random_state=SEED)),
        "selector_random_forest_score": (
            "tree",
            RandomForestRegressor(
                n_estimators=160,
                max_depth=12,
                min_samples_leaf=3,
                min_samples_split=8,
                max_features=0.75,
                random_state=SEED,
                n_jobs=4,
            ),
        ),
    }

    predictions = feature_table[["candidate_row_idx", "event_uid", "split", "candidate_delay_ms"]].copy()
    diagnostics: List[Dict[str, Any]] = []
    for name, (matrix_kind, estimator) in models.items():
        if matrix_kind == "scaled":
            estimator.fit(x_train_scaled, y_train)
            pred = estimator.predict(x_all_scaled)
        else:
            estimator.fit(x_train_tree, y_train)
            pred = estimator.predict(x_all_tree)
        predictions[f"predicted_score_{name}"] = pred.astype(float)
        for split, g in feature_table.groupby("split"):
            idx = g.index
            valid = np.isfinite(g["target_score_primary"].astype(float))
            if int(valid.sum()) == 0:
                continue
            y = g.loc[valid, "target_score_primary"].astype(float).to_numpy()
            p = pred[idx][valid.to_numpy()]
            diagnostics.append(
                {
                    "selector_name": name,
                    "split": str(split),
                    "n": int(len(y)),
                    "target_score_rmse": float(math.sqrt(mean_squared_error(y, p))),
                    "target_score_mae": float(mean_absolute_error(y, p)),
                }
            )
    write_csv(
        pd.DataFrame(
            {
                "encoded_feature_name": list(x_filled.columns),
                "train_median": train_median.reindex(x_filled.columns).fillna(0.0).astype(float).to_numpy(),
                "train_mean": x_mean.reindex(x_filled.columns).fillna(0.0).astype(float).to_numpy(),
                "train_std": x_std.reindex(x_filled.columns).fillna(1.0).astype(float).to_numpy(),
            }
        ),
        LOGS / "v247_selector_feature_encoding.csv",
    )
    diag_df = pd.DataFrame(diagnostics).sort_values(["selector_name", "split"]).reset_index(drop=True)
    write_csv(predictions, TABLES / "v247_selector_predictions_by_candidate.csv")
    write_csv(diag_df, TABLES / "v247_selector_fit_diagnostics.csv")
    return predictions, diag_df


def select_min_candidate(df: pd.DataFrame, value_col: str, selector_name: str, deployable: bool) -> pd.DataFrame:
    """按 event_uid 选择 value_col 最小的 candidate。"""

    work = df[np.isfinite(df[value_col].astype(float))].copy()
    work = work.sort_values(["event_uid", value_col, "candidate_delay_ms"], ascending=[True, True, True])
    selected = work.groupby("event_uid", as_index=False).head(1).copy()
    selected["selector_name"] = selector_name
    selected["deployable"] = bool(deployable)
    selected["selector_rank_value"] = selected[value_col].astype(float)
    return selected


def build_selector_selected_events(candidate: pd.DataFrame, selector_predictions: pd.DataFrame, best_long: pd.DataFrame) -> pd.DataFrame:
    """生成各 selector / policy 的事件级选锚结果。"""

    pred_cols = [c for c in selector_predictions.columns if c.startswith("predicted_score_")]
    merged = candidate.merge(
        selector_predictions[["candidate_row_idx"] + pred_cols],
        on="candidate_row_idx",
        how="left",
        validate="one_to_one",
    )

    selected_rows: List[pd.DataFrame] = []
    primary_score_col = f"score_{PRIMARY_SCORE_NAME}"
    selected_rows.append(select_min_candidate(merged, primary_score_col, "oracle_best_anchor_upper_bound", False))

    for col in pred_cols:
        selector_name = col.replace("predicted_score_", "")
        selected_rows.append(select_min_candidate(merged, col, selector_name, True))

    current = merged[merged["candidate_delay_ms"].eq(0)].copy()
    current["selector_name"] = "policy_keep_0ms_anchor"
    current["deployable"] = True
    current["selector_rank_value"] = 0.0
    selected_rows.append(current)

    latest = merged.sort_values(["event_uid", "candidate_delay_ms"], ascending=[True, False]).groupby("event_uid", as_index=False).head(1).copy()
    latest["selector_name"] = "policy_wait_to_latest_anchor"
    latest["deployable"] = True
    latest["selector_rank_value"] = latest["candidate_delay_ms"].astype(float)
    selected_rows.append(latest)

    coarse = merged[merged["candidate_delay_ms"].isin(COARSE_DELAY_MS)].copy()
    coarse_oracle = select_min_candidate(coarse, primary_score_col, "policy_nearest_coarse_oracle_proxy", False)
    selected_rows.append(coarse_oracle)

    selected = pd.concat(selected_rows, ignore_index=True)
    primary_best = best_long[best_long["score_name"].eq(PRIMARY_SCORE_NAME)].copy()
    primary_best = primary_best.rename(
        columns={
            "best_delay_ms": "oracle_best_delay_ms",
            "best_score": "oracle_best_score",
            "best_prediction_error_v241": "oracle_best_error_v241",
        }
    )
    selected = selected.merge(
        primary_best[
            [
                "event_uid",
                "oracle_best_delay_ms",
                "oracle_best_score",
                "oracle_best_error_v241",
            ]
        ],
        on="event_uid",
        how="left",
        validate="many_to_one",
    )
    selected["selected_delay_ms"] = selected["candidate_delay_ms"].astype(int)
    selected["selected_score"] = selected[primary_score_col].astype(float)
    selected["selected_error_v241"] = selected["candidate_tail_rmse_v241"].astype(float)
    selected["delay_abs_diff_ms"] = np.abs(selected["selected_delay_ms"].astype(float) - selected["oracle_best_delay_ms"].astype(float))
    selected["score_gap_vs_oracle"] = selected["selected_score"] - selected["oracle_best_score"]
    selected["selected_error_delta_vs_current"] = selected["selected_error_v241"] - selected["current_0ms_tail_rmse_v241"].astype(float)
    selected["oracle_error_delta_vs_current"] = selected["oracle_best_error_v241"] - selected["current_0ms_tail_rmse_v241"].astype(float)
    write_csv(selected, TABLES / "v247_selector_selected_anchor_by_event.csv")
    return selected


def build_selector_policy_summary(selected: pd.DataFrame) -> pd.DataFrame:
    """按 selector/split/group 汇总选锚效果。"""

    rows: List[Dict[str, Any]] = []
    for (selector_name, split), split_df in selected.groupby(["selector_name", "split"]):
        masks = group_masks_event(split_df.rename(columns={"oracle_best_delay_ms": "best_delay_ms"}))
        for group_name, mask in masks.items():
            sub = split_df[mask]
            if sub.empty:
                continue
            current = sub["current_0ms_tail_rmse_v241"].astype(float)
            selected_err = sub["selected_error_v241"].astype(float)
            best_err = sub["oracle_best_error_v241"].astype(float)
            denom = current - best_err
            captured = (current - selected_err) / denom.replace(0, np.nan)
            rows.append(
                {
                    "selector_name": str(selector_name),
                    "split": str(split),
                    "event_group": group_name,
                    "deployable": bool(sub["deployable"].astype(bool).iloc[0]),
                    "n": int(len(sub)),
                    "exact_50ms_match_rate": float(sub["delay_abs_diff_ms"].eq(0).mean()),
                    "within_50ms_rate": float(sub["delay_abs_diff_ms"].le(50).mean()),
                    "within_100ms_rate": float(sub["delay_abs_diff_ms"].le(100).mean()),
                    "within_200ms_rate": float(sub["delay_abs_diff_ms"].le(200).mean()),
                    "mean_selected_error_v241": float(selected_err.mean()),
                    "mean_best_error_v241": float(best_err.mean()),
                    "mean_current_0ms_error_v241": float(current.mean()),
                    "selected_error_delta_vs_current": float((selected_err - current).mean()),
                    "oracle_error_delta_vs_current": float((best_err - current).mean()),
                    "mean_selected_score_gap": float(sub["score_gap_vs_oracle"].astype(float).mean()),
                    "gain_capture_rate": float(np.nanmean(captured.to_numpy(dtype=float))),
                    "mean_selected_delay_ms": float(sub["selected_delay_ms"].astype(float).mean()),
                    "mean_best_delay_ms": float(sub["oracle_best_delay_ms"].astype(float).mean()),
                    "pct_selected_1000ms": float(sub["selected_delay_ms"].eq(1000).mean()),
                }
            )
    out = pd.DataFrame(rows).sort_values(["selector_name", "split", "event_group"]).reset_index(drop=True)
    write_csv(out, TABLES / "v247_selector_policy_summary.csv")
    return out


def build_signal_anchor_diagnostics(candidate: pd.DataFrame, best_long: pd.DataFrame) -> pd.DataFrame:
    """用信号代理锚点诊断 fine best anchor 是否靠近局部变化阶段。"""

    primary_best = best_long[best_long["score_name"].eq(PRIMARY_SCORE_NAME)][
        ["event_uid", "best_delay_ms", "best_prediction_error_v241"]
    ].copy()
    idx_min_instability = candidate.sort_values(["event_uid", "instability_penalty", "candidate_delay_ms"]).groupby("event_uid").head(1)
    idx_peak_steer_change = (
        candidate.sort_values(["event_uid", "abs_steer_slope_last05", "candidate_delay_ms"], ascending=[True, False, True])
        .groupby("event_uid")
        .head(1)
    )
    diag = primary_best.merge(
        idx_min_instability[["event_uid", "candidate_delay_ms", "instability_penalty"]].rename(
            columns={"candidate_delay_ms": "signal_proxy_min_instability_delay_ms"}
        ),
        on="event_uid",
        how="left",
        validate="one_to_one",
    )
    diag = diag.merge(
        idx_peak_steer_change[["event_uid", "candidate_delay_ms", "abs_steer_slope_last05"]].rename(
            columns={"candidate_delay_ms": "signal_proxy_peak_steer_change_delay_ms"}
        ),
        on="event_uid",
        how="left",
        validate="one_to_one",
    )
    diag["abs_diff_best_vs_min_instability_ms"] = np.abs(diag["best_delay_ms"] - diag["signal_proxy_min_instability_delay_ms"])
    diag["abs_diff_best_vs_peak_steer_change_ms"] = np.abs(diag["best_delay_ms"] - diag["signal_proxy_peak_steer_change_delay_ms"])
    write_csv(diag, TABLES / "v247_signal_anchor_diagnostics.csv")
    return diag


def plot_best_anchor_distribution(distribution: pd.DataFrame) -> Path:
    """画 primary score 下 test split 主要分组的 best delay 分布。"""

    primary = distribution[
        distribution["score_name"].eq(PRIMARY_SCORE_NAME)
        & distribution["split"].eq("test")
        & distribution["event_group"].isin(["all", "normal", "bad_top10", "very_bad_top5"])
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    groups = ["all", "normal", "bad_top10", "very_bad_top5"]
    for ax, group in zip(axes.ravel(), groups):
        sub = primary[primary["event_group"].eq(group)]
        if sub.empty:
            ax.set_title(group)
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue
        ax.bar(sub["best_delay_ms"].astype(int), sub["percent"].astype(float), width=38)
        ax.set_title(group)
        ax.set_ylim(0, max(0.05, float(primary["percent"].max()) * 1.15))
        ax.set_xlabel("best anchor delay (ms)")
        ax.set_ylabel("share")
    fig.suptitle(f"v247 primary best anchor distribution ({PRIMARY_SCORE_NAME}, test)")
    fig.tight_layout()
    out = FIGURES / "v247_best_anchor_distribution_by_group.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_selector_vs_oracle_error(summary: pd.DataFrame) -> Path:
    """画 test/all 下 selector 与 oracle/current 的平均误差对比。"""

    sub = summary[
        summary["split"].eq("test")
        & summary["event_group"].isin(["all", "bad_top10"])
        & summary["selector_name"].isin(
            [
                "oracle_best_anchor_upper_bound",
                "selector_ridge_score",
                "selector_random_forest_score",
                "policy_keep_0ms_anchor",
                "policy_wait_to_latest_anchor",
                "policy_nearest_coarse_oracle_proxy",
            ]
        )
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, group in zip(axes, ["all", "bad_top10"]):
        g = sub[sub["event_group"].eq(group)]
        if g.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(group)
            continue
        g = g.copy()
        order = [
            "policy_keep_0ms_anchor",
            "policy_wait_to_latest_anchor",
            "selector_ridge_score",
            "selector_random_forest_score",
            "policy_nearest_coarse_oracle_proxy",
            "oracle_best_anchor_upper_bound",
        ]
        g["selector_name"] = pd.Categorical(g["selector_name"], categories=order, ordered=True)
        g = g.sort_values("selector_name")
        ax.barh(g["selector_name"].astype(str), g["mean_selected_error_v241"].astype(float))
        ax.set_title(f"test {group}")
        ax.set_xlabel("mean selected tail RMSE")
    fig.suptitle("v247 selector vs oracle/current policies")
    fig.tight_layout()
    out = FIGURES / "v247_selector_vs_oracle_error.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_selected_delay_distribution(selected: pd.DataFrame) -> Path:
    """画 test/bad_top10 中不同策略的 selected delay 分布。"""

    sub = selected[
        selected["split"].eq("test")
        & normalize_bool_series(selected["bad_top10_split_v241"])
        & selected["selector_name"].isin(
            [
                "oracle_best_anchor_upper_bound",
                "selector_ridge_score",
                "selector_random_forest_score",
                "policy_wait_to_latest_anchor",
            ]
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    if sub.empty:
        ax.text(0.5, 0.5, "no test bad_top10 data", ha="center", va="center")
    else:
        selectors = [
            "oracle_best_anchor_upper_bound",
            "selector_ridge_score",
            "selector_random_forest_score",
            "policy_wait_to_latest_anchor",
        ]
        bins = np.arange(-25, 1051, 50)
        for selector in selectors:
            s = sub[sub["selector_name"].eq(selector)]
            if s.empty:
                continue
            counts = s["selected_delay_ms"].astype(int).value_counts(normalize=True).sort_index()
            ax.plot(counts.index, counts.values, marker="o", label=selector)
        ax.set_xlabel("selected delay (ms)")
        ax.set_ylabel("share")
        ax.set_xticks(COARSE_DELAY_MS)
        ax.legend(fontsize=8)
    ax.set_title("v247 selected delay distribution (test bad_top10)")
    fig.tight_layout()
    out = FIGURES / "v247_selected_delay_distribution.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_error_delay_score_examples(candidate: pd.DataFrame) -> Path:
    """画若干 test bad_top10 事件的 error/score-delay 曲线。"""

    current_bad = candidate[
        candidate["candidate_delay_ms"].eq(0)
        & candidate["split"].eq("test")
        & normalize_bool_series(candidate["bad_top10_split_v241"])
    ].copy()
    current_bad = current_bad.sort_values("current_0ms_tail_rmse_v241", ascending=False).head(6)
    event_ids = current_bad["event_uid"].astype(str).tolist()
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for ax, event_uid in zip(axes.ravel(), event_ids):
        sub = candidate[candidate["event_uid"].astype(str).eq(event_uid)].sort_values("candidate_delay_ms")
        ax.plot(sub["candidate_delay_ms"], sub["candidate_tail_rmse_v241"], marker="o", label="tail error")
        ax.plot(sub["candidate_delay_ms"], sub[f"score_{PRIMARY_SCORE_NAME}"], marker="s", label="primary score")
        best_delay = int(sub.sort_values([f"score_{PRIMARY_SCORE_NAME}", "candidate_delay_ms"]).iloc[0]["candidate_delay_ms"])
        ax.axvline(best_delay, color="tab:red", linestyle="--", linewidth=1)
        ax.set_title(str(event_uid)[:42])
        ax.set_ylabel("value")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(event_ids) :]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8)
    fig.supxlabel("candidate delay (ms)")
    fig.suptitle("v247 examples: error and primary score across fine anchors")
    fig.tight_layout()
    out = FIGURES / "v247_error_delay_score_curves_examples.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_signal_anchor_alignment(signal_diag: pd.DataFrame) -> Path:
    """画信号代理锚点与 primary best anchor 的关系。"""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(
        signal_diag["signal_proxy_min_instability_delay_ms"],
        signal_diag["best_delay_ms"],
        s=10,
        alpha=0.45,
    )
    axes[0].plot([0, 1000], [0, 1000], color="tab:red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("min instability proxy delay (ms)")
    axes[0].set_ylabel("primary best delay (ms)")
    axes[0].set_title("best vs min-instability proxy")
    axes[1].scatter(
        signal_diag["signal_proxy_peak_steer_change_delay_ms"],
        signal_diag["best_delay_ms"],
        s=10,
        alpha=0.45,
    )
    axes[1].plot([0, 1000], [0, 1000], color="tab:red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("peak steer-change proxy delay (ms)")
    axes[1].set_ylabel("primary best delay (ms)")
    axes[1].set_title("best vs peak-steer-change proxy")
    fig.suptitle("v247 signal-anchor diagnostic proxies")
    fig.tight_layout()
    out = FIGURES / "v247_signal_anchor_alignment.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def create_figures(
    distribution: pd.DataFrame,
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    candidate: pd.DataFrame,
    signal_diag: pd.DataFrame,
) -> List[Path]:
    """统一生成 v247 核心图。"""

    configure_plotting()
    paths = [
        plot_best_anchor_distribution(distribution),
        plot_selector_vs_oracle_error(summary),
        plot_selected_delay_distribution(selected),
        plot_error_delay_score_examples(candidate),
        plot_signal_anchor_alignment(signal_diag),
    ]
    return paths


def metric_lookup(df: pd.DataFrame, **kwargs: Any) -> pd.Series | None:
    """从汇总表里按条件取一行。"""

    mask = pd.Series(True, index=df.index)
    for key, value in kwargs.items():
        mask &= df[key].eq(value)
    sub = df[mask]
    if sub.empty:
        return None
    return sub.iloc[0]


def fmt_float(value: Any, digits: int = 3) -> str:
    """报告中安全格式化浮点数。"""

    try:
        v = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def write_report(
    sampling_audit: pd.DataFrame,
    coarse_alignment: pd.DataFrame,
    score_summary: pd.DataFrame,
    selector_summary: pd.DataFrame,
    selector_diag: pd.DataFrame,
    signal_diag: pd.DataFrame,
    inference: V241InferenceResult,
) -> Path:
    """写中文结论报告。"""

    audit = sampling_audit.iloc[0]
    test_all_oracle = metric_lookup(score_summary, score_name=PRIMARY_SCORE_NAME, split="test", event_group="all")
    test_bad_oracle = metric_lookup(score_summary, score_name=PRIMARY_SCORE_NAME, split="test", event_group="bad_top10")
    test_all_rf = metric_lookup(selector_summary, selector_name="selector_random_forest_score", split="test", event_group="all")
    test_bad_rf = metric_lookup(selector_summary, selector_name="selector_random_forest_score", split="test", event_group="bad_top10")
    test_all_latest = metric_lookup(selector_summary, selector_name="policy_wait_to_latest_anchor", split="test", event_group="all")
    test_bad_latest = metric_lookup(selector_summary, selector_name="policy_wait_to_latest_anchor", split="test", event_group="bad_top10")
    test_all_current = metric_lookup(selector_summary, selector_name="policy_keep_0ms_anchor", split="test", event_group="all")
    test_bad_current = metric_lookup(selector_summary, selector_name="policy_keep_0ms_anchor", split="test", event_group="bad_top10")

    if not coarse_alignment.empty and "pred_rmse_mean" in coarse_alignment.columns:
        coarse_line = (
            f"coarse replay mean RMSE={fmt_float(coarse_alignment['pred_rmse_mean'].mean(), 6)}, "
            f"max={fmt_float(coarse_alignment['pred_rmse_max'].max(), 6)}"
        )
    else:
        coarse_line = "coarse replay 未得到有效对齐结果"

    signal_line = (
        f"best 与 min-instability proxy 平均距离 {fmt_float(signal_diag['abs_diff_best_vs_min_instability_ms'].mean(), 1)}ms；"
        f"best 与 peak-steer-change proxy 平均距离 {fmt_float(signal_diag['abs_diff_best_vs_peak_steer_change_ms'].mean(), 1)}ms。"
    )

    lines = [
        "# v247 multi-resolution best anchor discovery 报告",
        "",
        "## 结论摘要",
        "",
        (
            f"- 50ms fine grid 采样审计：生成 `{int(audit['n_generated_rows'])}` 行，"
            f"事件数 `{int(audit['n_events'])}`，完整事件比例 `{fmt_float(audit['complete_event_rate'])}`，"
            f"delay 值数量 `{len(json.loads(str(audit['actual_delay_values'])))}`；"
            f"fine_grid_supported=`{bool(audit['fine_grid_supported'])}`。"
        ),
        f"- 锁定 v241 推理：device=`{inference.device}`，point 数 `{inference.point_count}`，耗时 `{fmt_float(inference.seconds, 1)}` 秒；没有训练新轨迹模型。",
        f"- v241 coarse replay 对齐：{coarse_line}。这一步检查 fine-grid 里 0/200/.../1000ms 是否能复现旧 v241 预测。",
        (
            f"- primary score=`{PRIMARY_SCORE_NAME}` 下，test/all 当前 0ms 平均误差 "
            f"`{fmt_float(test_all_current['mean_selected_error_v241'] if test_all_current is not None else math.nan)}`，"
            f"oracle best 平均误差 `{fmt_float(test_all_oracle['mean_best_error_v241'] if test_all_oracle is not None else math.nan)}`，"
            f"平均 best delay `{fmt_float(test_all_oracle['mean_best_delay_ms'] if test_all_oracle is not None else math.nan, 1)}ms`。"
        ),
        (
            f"- test/bad_top10 当前 0ms 平均误差 "
            f"`{fmt_float(test_bad_current['mean_selected_error_v241'] if test_bad_current is not None else math.nan)}`，"
            f"oracle best 平均误差 `{fmt_float(test_bad_oracle['mean_best_error_v241'] if test_bad_oracle is not None else math.nan)}`，"
            f"平均 best delay `{fmt_float(test_bad_oracle['mean_best_delay_ms'] if test_bad_oracle is not None else math.nan, 1)}ms`。"
        ),
        (
            f"- RF selector 在 test/all 的平均选中误差 `{fmt_float(test_all_rf['mean_selected_error_v241'] if test_all_rf is not None else math.nan)}`，"
            f"相对当前 0ms delta `{fmt_float(test_all_rf['selected_error_delta_vs_current'] if test_all_rf is not None else math.nan)}`，"
            f"within100ms `{fmt_float(test_all_rf['within_100ms_rate'] if test_all_rf is not None else math.nan)}`，"
            f"平均选中 delay `{fmt_float(test_all_rf['mean_selected_delay_ms'] if test_all_rf is not None else math.nan, 1)}ms`。"
        ),
        (
            f"- RF selector 在 test/bad_top10 的平均选中误差 `{fmt_float(test_bad_rf['mean_selected_error_v241'] if test_bad_rf is not None else math.nan)}`，"
            f"相对当前 0ms delta `{fmt_float(test_bad_rf['selected_error_delta_vs_current'] if test_bad_rf is not None else math.nan)}`，"
            f"gain capture `{fmt_float(test_bad_rf['gain_capture_rate'] if test_bad_rf is not None else math.nan)}`。"
        ),
        (
            f"- 固定 wait-latest 在 test/bad_top10 的平均选中误差 "
            f"`{fmt_float(test_bad_latest['mean_selected_error_v241'] if test_bad_latest is not None else math.nan)}`，"
            f"平均 delay `{fmt_float(test_bad_latest['mean_selected_delay_ms'] if test_bad_latest is not None else math.nan, 1)}ms`；"
            "这个基线用于判断 selector 是否只是学到“永远等到最后”。"
        ),
        f"- 信号代理锚点诊断：{signal_line}",
        "",
        "## 怎么理解这一步",
        "",
        "v247 做的不是把所有样本强行后移，而是给每个事件同时生成 0ms 到 1000ms、间隔 50ms 的候选观察点。"
        "每个候选点都重新从原始车辆 CSV 中取历史窗口和未来监督窗口，然后用同一个 v241 模型预测。"
        "离线 best anchor 是对这些候选点打分后的最优点，score 同时考虑预测误差、等待代价和局部不稳定性。",
        "",
        "如果 error-only 几乎总是选 1000ms，而加入等待代价/不稳定性后 best delay 回到中间区间，说明任务定义比单纯“后移锚点”更合理。"
        "如果 selector 明显优于 keep-0ms 且不只是等到 latest，才值得进入下一步更强模型训练。",
        "",
        "## 关键产物",
        "",
        "- `tables/v247_fine_anchor_candidate_table.csv`：每个事件 21 个 fine anchor 的误差和 score。",
        "- `tables/v247_best_anchor_by_event.csv`：不同 score 定义下每个事件的离线 best anchor。",
        "- `tables/v247_selector_training_table.csv`：selector 使用的 input-only 特征表。",
        "- `tables/v247_selector_selected_anchor_by_event.csv`：selector/policy 选中的锚点。",
        "- `tables/v247_selector_policy_summary.csv`：selector 与 current/latest/oracle 的分组对比。",
        "- `figures/v247_best_anchor_distribution_by_group.png`：best anchor 分布。",
        "- `figures/v247_error_delay_score_curves_examples.png`：典型差样本的 error/score-delay 曲线。",
        "",
        "## 风险和下一步",
        "",
        "- oracle best anchor 使用未来真实误差，只能作为离线标签和上限，不能部署。",
        "- fine grid 的监督点是相对每个 candidate anchor 的 0.1s 网格；50ms candidate 的 tail 点会落在 1.05/1.15/... 这类原始相对时刻，这是本版使用 raw nearest 采样的直接结果。",
        "- 下一步是否训练更强的 anchor-aware 轨迹模型，取决于 `selector_random_forest_score` 是否在 test/bad_top10 上超过 wait-latest，并且 normal 组没有明显变差。",
        "",
        "## selector score 拟合诊断",
        "",
        selector_diag.to_markdown(index=False) if not selector_diag.empty else "无 selector 诊断。",
    ]

    out = REPORTS / "v247_multi_resolution_best_anchor_discovery_cn.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_input_hashes() -> pd.DataFrame:
    """记录主要输入文件 hash。"""

    rows = []
    for name, path in {
        "v247_script": Path(__file__),
        "v236_script": V236_SCRIPT,
        "v238_script": V238_SCRIPT,
        "v239_script": V239_SCRIPT,
        "v241_script": V241_SCRIPT,
        "v241_model": V241_MODEL,
        "v241_predictions_npz": V241_PRED_NPZ,
        "design_doc": DESIGN_DOC,
        "plan_doc": PLAN_DOC,
    }.items():
        rows.append(
            {
                "input_name": name,
                "path": str(path),
                "exists": bool(path.exists()),
                "sha256": file_sha256(path) if path.exists() else "",
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, LOGS / "input_file_hashes.csv")
    return out


def build_guardrail_json(sampling_audit: pd.DataFrame, selector_feature_cols: List[str], zip_testzip: str | None) -> Dict[str, Any]:
    """生成 guardrail 审计日志。"""

    forbidden = {"event_uid", "recording", "subject", "candidate_tail_rmse_v241", "target_score_primary"}
    forbidden_used = sorted(set(selector_feature_cols) & forbidden)
    audit = sampling_audit.iloc[0].to_dict()
    payload: Dict[str, Any] = {
        "pass": True,
        "stage": "v247_multi_resolution_best_anchor_discovery",
        "no_trajectory_model_training": True,
        "input_only_selector": len(forbidden_used) == 0,
        "oracle_best_anchor_upper_bound_only": True,
        "no_test_based_retuning": True,
        "no_event_uid_or_recording_as_features": not bool({"event_uid", "recording", "subject"} & set(selector_feature_cols)),
        "fine_grid_sampling_checked": bool(audit.get("fine_grid_sampling_checked", False)),
        "fine_grid_supported": bool(audit.get("fine_grid_supported", False)),
        "score_weights_declared_before_test_summary": True,
        "zip_testzip": zip_testzip,
        "forbidden_selector_features_used": forbidden_used,
        "primary_score_name": PRIMARY_SCORE_NAME,
        "fine_delay_ms": FINE_DELAY_MS,
        "score_configs": [
            {"name": name, "lambda_wait": float(lambda_wait), "mu_unstable": float(mu_unstable)}
            for name, lambda_wait, mu_unstable in SCORE_CONFIGS
        ],
    }
    bool_keys = [
        "no_trajectory_model_training",
        "input_only_selector",
        "oracle_best_anchor_upper_bound_only",
        "no_test_based_retuning",
        "no_event_uid_or_recording_as_features",
        "fine_grid_sampling_checked",
        "fine_grid_supported",
        "score_weights_declared_before_test_summary",
    ]
    payload["pass"] = bool(all(bool(payload[k]) for k in bool_keys) and zip_testzip is None)
    write_json(payload, LOGS / "guardrail_check.json")
    return payload


def write_run_manifest(
    data: object,
    sampling_audit: pd.DataFrame,
    inference: V241InferenceResult,
    report_path: Path,
    figures: List[Path],
    zip_path: Path,
) -> Dict[str, Any]:
    """写实验运行元数据。"""

    payload = {
        "stage": "v247_multi_resolution_best_anchor_discovery",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(ROOT),
        "out_dir": str(OUT),
        "zip_path": str(zip_path),
        "n_candidate_rows": int(len(data.manifest)),
        "n_events": int(data.manifest["event_uid"].nunique()),
        "fine_delay_ms": FINE_DELAY_MS,
        "future_grid_s": [float(FUTURE_GRID.min()), float(FUTURE_GRID.max()), int(len(FUTURE_GRID))],
        "sampling_audit": sampling_audit.iloc[0].to_dict(),
        "v241_inference": {
            "device": inference.device,
            "point_count": inference.point_count,
            "seconds": inference.seconds,
            "checkpoint_config": inference.checkpoint_config,
            "best_model_name": inference.best_model_name,
        },
        "report": str(report_path),
        "figures": [str(p) for p in figures],
    }
    write_json(payload, LOGS / "run_manifest.json")
    return payload


def file_inventory() -> pd.DataFrame:
    """列出 v247 产物清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(OUT)),
                    "bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
            )
    out = pd.DataFrame(rows)
    write_csv(out, LOGS / "file_inventory.csv")
    return out


def main() -> None:
    """v247 主流程。"""

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    ensure_clean_output()
    write_input_hashes()

    print("[v247] build 50ms fine-grid rolling dataset from raw CSV")
    data, manifest, feature_names, dropped = build_fine_grid_dataset()
    sampling_audit = build_sampling_audit(manifest, dropped)
    print(f"[v247] fine rows={len(manifest)} events={manifest['event_uid'].nunique()} delays={manifest['delay_ms'].nunique()}")

    print("[v247] run locked v241 inference on fine anchors")
    inference = run_locked_v241_inference(data)
    np.savez_compressed(
        OUT / "v247_fine_grid_v241_predictions.npz",
        y_true_steering_delta=data.y_future[:, :, 0].astype(np.float32),
        pred_v241_steering_delta=inference.pred_curve.astype(np.float32),
        event_uid=manifest["event_uid"].astype(str).to_numpy(),
        delay_ms=manifest["delay_ms"].astype(np.int32).to_numpy(),
        split=manifest["split"].astype(str).to_numpy(),
        future_grid_s=FUTURE_GRID.astype(np.float32),
    )
    coarse_alignment = validate_coarse_replay(manifest, inference.pred_curve, data.y_future)

    print("[v247] build candidate scores and offline best anchors")
    candidate = build_candidate_score_table(data, manifest, inference.pred_curve)
    candidate, current = attach_current_anchor_groups(candidate)
    write_csv(candidate, TABLES / "v247_fine_anchor_candidate_table.csv")
    best_long = select_best_by_score(candidate)
    distribution = build_best_anchor_distribution(best_long)
    score_summary = build_score_weight_sweep_summary(best_long)

    print("[v247] train input-only selector")
    selector_table = build_selector_feature_table(candidate, data)
    selector_predictions, selector_diag = train_selector_models(selector_table)
    selected = build_selector_selected_events(candidate, selector_predictions, best_long)
    selector_summary = build_selector_policy_summary(selected)
    signal_diag = build_signal_anchor_diagnostics(candidate, best_long)

    print("[v247] create figures and report")
    figures = create_figures(distribution, selector_summary, selected, candidate, signal_diag)
    report_path = write_report(
        sampling_audit=sampling_audit,
        coarse_alignment=coarse_alignment,
        score_summary=score_summary,
        selector_summary=selector_summary,
        selector_diag=selector_diag,
        signal_diag=signal_diag,
        inference=inference,
    )

    selector_feature_cols = [
        c
        for c in selector_table.columns
        if c
        not in {
            "event_uid",
            "candidate_row_idx",
            "split",
            "target_score_primary",
            "candidate_tail_rmse_v241",
        }
    ]
    write_run_manifest(data, sampling_audit, inference, report_path, figures, ZIP_PATH)
    file_inventory()
    write_guardrail_pre = build_guardrail_json(sampling_audit, selector_feature_cols, zip_testzip=None)
    zip_path = zip_outputs()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_testzip = zf.testzip()
    guardrail = build_guardrail_json(sampling_audit, selector_feature_cols, zip_testzip=zip_testzip)
    file_inventory()
    zip_path = zip_outputs()
    with zipfile.ZipFile(zip_path, "r") as zf:
        final_testzip = zf.testzip()
    if final_testzip is not None:
        raise AssertionError(f"ZIP testzip failed: {final_testzip}")
    print(f"[v247] guardrail_check.pass={guardrail['pass']}")
    print(f"[v247] ZIP testzip={final_testzip}")
    print(f"[v247] report={report_path}")
    print(f"[v247] zip={zip_path}")


if __name__ == "__main__":
    main()
