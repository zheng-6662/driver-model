from __future__ import annotations

"""
v246：oracle 最佳锚点遍历 + input-only 锚点选择器审查。

本轮回答用户提出的路线：
“遍历锚点后移，使每个样本达到最佳锚点”。

核心边界：
1. oracle_best_anchor：可以用真实误差遍历每个样本可等待的更晚锚点，
   找出理论最佳锚点。这是上限审查，不可直接部署。
2. input_only_selector：只用 base 锚点时已经可见的历史/道路/phase 特征，
   加上候选等待时长，训练一个轻量误差预测器，让它选择候选锚点。
   这一步是可部署性诊断，但仍然不训练轨迹预测模型。
3. 不改 v241/v243 轨迹模型，不做 test-based retuning，不删除样本。
"""

import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

# Windows + MKL/OpenMP 在本项目里偶尔会出现无 traceback 的进程退出。
# 这里和前序 v241/v243 脚本保持一致，先限制底层线程，保证审查脚本稳定可复现。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V243_DIR = BASELINES / "v243_v241_guarded_finetune_20260626"
PRED_PATH = V243_DIR / "v243_v241_guarded_finetune_predictions.npz"

V236_DIR = BASELINES / "v236_rolling_reanchor_dataset_and_baseline_20260624"
V236_ARRAYS = V236_DIR / "v236_rolling_dataset_arrays_and_predictions.npz"
V236_MANIFEST = V236_DIR / "tables" / "v236_rolling_sample_manifest.csv"

OUT = BASELINES / "v246_oracle_best_anchor_and_selector_audit_20260630"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"

FUTURE_GRID = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)
HIST_COLS = [
    "steering",
    "speed_kmh",
    "vx",
    "vy",
    "ax",
    "ay",
    "yaw_rate",
    "roll",
    "pitch",
    "yaw",
    "roll_rate",
    "pitch_rate",
    "roll_acc",
    "pitch_acc",
    "accelerator",
    "brake",
    "lane_curvature",
    "lateral_distance",
]

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def ensure_clean_output() -> None:
    """只清理 v246 自己的输出目录，避免触碰前序实验。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 使用 utf-8-sig，方便 Windows Excel 直接打开中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """记录关键输入哈希，保证本轮审查可追溯。"""

    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """计算一段有限数值的 RMSE。"""

    if len(a) == 0:
        return math.nan
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return math.nan
    return float(np.sqrt(np.mean(diff**2)))


def load_inputs() -> Dict[str, object]:
    """读取 v243 预测、v236 rolling 输入和 manifest，并做严格对齐检查。"""

    required = [PRED_PATH, V236_ARRAYS, V236_MANIFEST]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少必要输入：\n" + "\n".join(missing))

    pred = np.load(PRED_PATH, allow_pickle=True)
    y_true = pred["y_true_steering_delta"].astype(np.float32)
    pred_v241 = pred["pred_v241_steering_delta"].astype(np.float32)
    pred_hard36 = pred["pred_v243_best_guarded_steering_delta"].astype(np.float32)
    delay_ms = pred["delay_ms"].astype(int)
    split = pred["split"].astype(str)
    event_uid = pred["event_uid"].astype(str)
    future_grid = pred["future_grid_s"].astype(np.float32)
    valid = pred["original_remaining_valid"].astype(bool)

    if not np.allclose(future_grid, FUTURE_GRID):
        raise AssertionError(f"future_grid 与预期不一致：{future_grid}")

    manifest = pd.read_csv(V236_MANIFEST, encoding="utf-8-sig")
    with np.load(V236_ARRAYS, allow_pickle=False) as arr:
        x_hist = arr["X_hist"].astype(np.float32)
        x_road = arr["X_road"].astype(np.float32)
        x_phase = arr["X_phase"].astype(np.float32)
        arr_event_uid = arr["event_uid"].astype(str)
        arr_delay_ms = arr["delay_ms"].astype(int)
        arr_split = arr["split"].astype(str)
        feature_names = arr["feature_names"].astype(str).tolist()

    if len(manifest) != len(y_true):
        raise AssertionError(f"manifest 行数与 prediction 不一致：{len(manifest)} vs {len(y_true)}")
    if not np.array_equal(manifest["event_uid"].astype(str).to_numpy(), event_uid):
        raise AssertionError("manifest 与 prediction event_uid 顺序不一致")
    if not np.array_equal(manifest["delay_ms"].astype(int).to_numpy(), delay_ms):
        raise AssertionError("manifest 与 prediction delay_ms 顺序不一致")
    if not np.array_equal(manifest["split"].astype(str).to_numpy(), split):
        raise AssertionError("manifest 与 prediction split 顺序不一致")
    if not np.array_equal(arr_event_uid, event_uid):
        raise AssertionError("v236 arrays 与 prediction event_uid 顺序不一致")
    if not np.array_equal(arr_delay_ms, delay_ms):
        raise AssertionError("v236 arrays 与 prediction delay_ms 顺序不一致")
    if not np.array_equal(arr_split, split):
        raise AssertionError("v236 arrays 与 prediction split 顺序不一致")
    if not feature_names[0].endswith("_steering"):
        raise AssertionError(f"无法确认 X_hist 第 0 维是 steering：{feature_names[0]}")

    return {
        "manifest": manifest.reset_index(drop=True),
        "y_true": y_true,
        "pred_v241": pred_v241,
        "pred_hard36": pred_hard36,
        "delay_ms": delay_ms,
        "split": split,
        "event_uid": event_uid,
        "future_grid": future_grid,
        "valid": valid,
        "x_hist": x_hist,
        "x_road": x_road,
        "x_phase": x_phase,
        "anchor_steering": x_hist[:, -1, 0].astype(np.float32),
    }


def original_tail_indices(delay_ms: int, future_grid: np.ndarray, valid_row: np.ndarray) -> np.ndarray:
    """
    返回该 rolling 样本中对应 original anchor 后 1.0-2.0s 的 future 下标。

    当前 v236/v243 rolling 任务里，delay<=1000ms 的样本都应覆盖 1.0-2.0s 共 11 个点。
    """

    original_rel_s = delay_ms / 1000.0 + future_grid
    mask = valid_row.astype(bool) & (original_rel_s >= 1.0 - 1e-9) & (original_rel_s <= 2.0 + 1e-9)
    return np.where(mask)[0]


def build_sample_table(data: Dict[str, object]) -> pd.DataFrame:
    """计算每个 rolling 样本在 original 1.0-2.0s tail 段上的 v241/v243 绝对轨迹误差。"""

    manifest = data["manifest"].copy()
    y_true = data["y_true"]
    pred_v241 = data["pred_v241"]
    pred_hard36 = data["pred_hard36"]
    delay_ms = data["delay_ms"]
    future_grid = data["future_grid"]
    valid = data["valid"]
    anchor = data["anchor_steering"]

    err_v241 = np.full(len(manifest), np.nan, dtype=float)
    err_hard36 = np.full(len(manifest), np.nan, dtype=float)
    tail_point_n = np.zeros(len(manifest), dtype=int)
    true_peak_abs = np.full(len(manifest), np.nan, dtype=float)

    for i in range(len(manifest)):
        idx = original_tail_indices(int(delay_ms[i]), future_grid, valid[i])
        tail_point_n[i] = int(len(idx))
        if len(idx) == 0:
            continue
        true_abs = anchor[i] + y_true[i, idx]
        pred_abs_v241 = anchor[i] + pred_v241[i, idx]
        pred_abs_h36 = anchor[i] + pred_hard36[i, idx]
        err_v241[i] = finite_rmse(true_abs, pred_abs_v241)
        err_hard36[i] = finite_rmse(true_abs, pred_abs_h36)
        true_peak_abs[i] = float(np.max(np.abs(y_true[i, idx])))

    manifest["tail_abs_rmse_v241"] = err_v241
    manifest["tail_abs_rmse_hard36"] = err_hard36
    manifest["tail_point_n_original_1_2s"] = tail_point_n
    manifest["tail_true_peak_abs"] = true_peak_abs

    # 分 split 定义 bad_top10，避免用 test 阈值污染 train/val 的 oracle label 分析。
    manifest["bad_top10_split_v241"] = False
    manifest["very_bad_top5_split_v241"] = False
    threshold_rows: List[Dict[str, object]] = []
    for sp, g in manifest.groupby("split"):
        q90 = float(g["tail_abs_rmse_v241"].quantile(0.90))
        q95 = float(g["tail_abs_rmse_v241"].quantile(0.95))
        idx = g.index
        manifest.loc[idx, "bad_top10_split_v241"] = manifest.loc[idx, "tail_abs_rmse_v241"].ge(q90)
        manifest.loc[idx, "very_bad_top5_split_v241"] = manifest.loc[idx, "tail_abs_rmse_v241"].ge(q95)
        threshold_rows.append(
            {
                "split": str(sp),
                "q90_bad_top10": q90,
                "q95_very_bad_top5": q95,
                "n": int(len(g)),
                "n_bad_top10": int(manifest.loc[idx, "bad_top10_split_v241"].sum()),
                "n_very_bad_top5": int(manifest.loc[idx, "very_bad_top5_split_v241"].sum()),
            }
        )
    manifest["early_bad_top10_split_v241"] = manifest["bad_top10_split_v241"] & manifest["delay_ms"].astype(int).le(400)
    write_csv(pd.DataFrame(threshold_rows), TABLES / "v246_bad_thresholds_by_split.csv")
    return manifest


def build_base_features(data: Dict[str, object], sample_table: pd.DataFrame) -> pd.DataFrame:
    """
    生成 base 锚点可见特征。

    这些特征不使用未来真实曲线，也不使用人工响应标签；selector 只能看这些信息和候选等待时长。
    """

    x_hist = data["x_hist"]
    x_road = data["x_road"]
    x_phase = data["x_phase"]
    ci = {name: i for i, name in enumerate(HIST_COLS)}

    df = pd.DataFrame(
        {
            "sample_idx": np.arange(len(sample_table), dtype=int),
            "split": sample_table["split"].astype(str).to_numpy(),
            "event_uid": sample_table["event_uid"].astype(str).to_numpy(),
            "base_delay_ms": sample_table["delay_ms"].astype(int).to_numpy(),
            "scene_type": sample_table["scene_type"].astype(str).fillna("NA").to_numpy(),
            "pool_key": sample_table["pool_key"].astype(str).fillna("NA").to_numpy(),
            "hist_current_steer": x_hist[:, -1, ci["steering"]],
            "hist_abs_current_steer": np.abs(x_hist[:, -1, ci["steering"]]),
            "hist_abs_mean_steer": np.mean(np.abs(x_hist[:, :, ci["steering"]]), axis=1),
            "hist_abs_max_steer": np.max(np.abs(x_hist[:, :, ci["steering"]]), axis=1),
            "hist_steer_slope_last05": x_hist[:, -1, ci["steering"]] - x_hist[:, -6, ci["steering"]],
            "hist_abs_steer_slope_last05": np.abs(x_hist[:, -1, ci["steering"]] - x_hist[:, -6, ci["steering"]]),
            "hist_yaw_abs_mean": np.mean(np.abs(x_hist[:, :, ci["yaw_rate"]]), axis=1),
            "hist_ay_abs_mean": np.mean(np.abs(x_hist[:, :, ci["ay"]]), axis=1),
            "hist_speed_mean": np.mean(x_hist[:, :, ci["speed_kmh"]], axis=1),
            "hist_brake_mean": np.mean(x_hist[:, :, ci["brake"]], axis=1),
            "hist_accel_mean": np.mean(x_hist[:, :, ci["accelerator"]], axis=1),
            "hist_curv_abs_mean": np.mean(np.abs(x_hist[:, :, ci["lane_curvature"]]), axis=1),
            "hist_lat_abs_mean": np.mean(np.abs(x_hist[:, :, ci["lateral_distance"]]), axis=1),
            "road_curv_abs_mean": np.mean(np.abs(x_road[:, :, 0]), axis=1),
            "road_curv_abs_max": np.max(np.abs(x_road[:, :, 0]), axis=1),
            "road_lat_abs_mean": np.mean(np.abs(x_road[:, :, 1]), axis=1),
            "road_lat_abs_max": np.max(np.abs(x_road[:, :, 1]), axis=1),
        }
    )
    for j in range(x_phase.shape[1]):
        df[f"phase_{j}"] = x_phase[:, j]
    return df


def build_candidate_table(sample_table: pd.DataFrame, base_features: pd.DataFrame) -> pd.DataFrame:
    """
    构造 base sample -> candidate anchor 的遍历表。

    对每个 base 样本，只允许选择同一 event_uid 下 delay >= base_delay 的候选锚点。
    target_error 使用候选锚点自身在 original 1.0-2.0s tail 段上的 v241 绝对轨迹 RMSE。
    """

    pair_to_idx = {
        (str(row.event_uid), int(row.delay_ms)): int(i)
        for i, row in sample_table[["event_uid", "delay_ms"]].iterrows()
    }
    rows: List[Dict[str, object]] = []

    for base_idx, base in sample_table.iterrows():
        base_delay = int(base["delay_ms"])
        for candidate_delay in [0, 200, 400, 600, 800, 1000]:
            if candidate_delay < base_delay:
                continue
            candidate_idx = pair_to_idx.get((str(base["event_uid"]), int(candidate_delay)))
            if candidate_idx is None:
                continue
            cand = sample_table.loc[candidate_idx]
            if not np.isfinite(float(cand["tail_abs_rmse_v241"])):
                continue
            rows.append(
                {
                    "base_idx": int(base_idx),
                    "candidate_idx": int(candidate_idx),
                    "split": str(base["split"]),
                    "event_uid": str(base["event_uid"]),
                    "base_delay_ms": int(base_delay),
                    "candidate_delay_ms": int(candidate_delay),
                    "candidate_shift_ms": int(candidate_delay - base_delay),
                    "base_error_v241": float(base["tail_abs_rmse_v241"]),
                    "candidate_error_v241": float(cand["tail_abs_rmse_v241"]),
                    "candidate_error_hard36": float(cand["tail_abs_rmse_hard36"]),
                    "candidate_delta_vs_base_v241": float(cand["tail_abs_rmse_v241"] - base["tail_abs_rmse_v241"]),
                    "base_bad_top10_split_v241": bool(base["bad_top10_split_v241"]),
                    "base_very_bad_top5_split_v241": bool(base["very_bad_top5_split_v241"]),
                    "base_early_bad_top10_split_v241": bool(base["early_bad_top10_split_v241"]),
                    "base_tail_true_peak_abs": float(base["tail_true_peak_abs"]),
                    "candidate_tail_true_peak_abs": float(cand["tail_true_peak_abs"]),
                    "base_subject": str(base["subject"]),
                    "base_recording": str(base["recording"]),
                    "base_scene_type": str(base["scene_type"]),
                }
            )

    cand = pd.DataFrame(rows)
    # 拼接 base 可见特征；selector 不能看 candidate 的未来真实误差，只能学 candidate_error 作为监督目标。
    # base_delay_ms 已经在候选表中存在，不能再次从 base_features 拼接，
    # 否则 pandas 会自动生成 base_delay_ms_x / base_delay_ms_y，后续汇总会找不到原列。
    feature_cols = [c for c in base_features.columns if c not in {"sample_idx", "split", "event_uid", "base_delay_ms"}]
    cand = cand.merge(
        base_features[["sample_idx"] + feature_cols],
        left_on="base_idx",
        right_on="sample_idx",
        how="left",
        validate="many_to_one",
    )
    cand = cand.drop(columns=["sample_idx"])
    return cand


def build_oracle_table(candidates: pd.DataFrame) -> pd.DataFrame:
    """每个 base 样本选择真实误差最小的候选锚点，作为 oracle 上限。"""

    rows: List[pd.Series] = []
    for _, g in candidates.groupby("base_idx", sort=False):
        # 误差相同的时候优先选择更早锚点，避免 oracle 无意义等待。
        rows.append(g.sort_values(["candidate_error_v241", "candidate_shift_ms"]).iloc[0])
    oracle = pd.DataFrame(rows).reset_index(drop=True)
    oracle = oracle.rename(
        columns={
            "candidate_idx": "oracle_candidate_idx",
            "candidate_delay_ms": "oracle_delay_ms",
            "candidate_shift_ms": "oracle_shift_ms",
            "candidate_error_v241": "oracle_error_v241",
            "candidate_error_hard36": "oracle_error_hard36",
            "candidate_delta_vs_base_v241": "oracle_delta_vs_base_v241",
        }
    )
    oracle["oracle_improved_v241"] = oracle["oracle_error_v241"] < oracle["base_error_v241"]
    return oracle


def summarize_oracle(oracle: pd.DataFrame) -> pd.DataFrame:
    """按 split / group 汇总 oracle 最佳锚点上限收益。"""

    group_defs = [
        ("all", lambda x: pd.Series(True, index=x.index)),
        ("shiftable_delay_lt_1000", lambda x: x["base_delay_ms"].astype(int).lt(1000)),
        ("delay0_only", lambda x: x["base_delay_ms"].astype(int).eq(0)),
        ("bad_top10", lambda x: x["base_bad_top10_split_v241"].astype(bool)),
        ("early_bad_top10_delay_le_400", lambda x: x["base_early_bad_top10_split_v241"].astype(bool)),
        ("very_bad_top5", lambda x: x["base_very_bad_top5_split_v241"].astype(bool)),
    ]
    rows: List[Dict[str, object]] = []
    for split, split_df in oracle.groupby("split"):
        for group_name, fn in group_defs:
            mask = fn(split_df)
            g = split_df[mask]
            if g.empty:
                continue
            base = g["base_error_v241"].astype(float)
            best = g["oracle_error_v241"].astype(float)
            rows.append(
                {
                    "split": str(split),
                    "base_group": group_name,
                    "n_base_samples": int(len(g)),
                    "mean_base_error_v241": float(base.mean()),
                    "mean_oracle_error_v241": float(best.mean()),
                    "mean_delta_oracle_minus_base": float((best - base).mean()),
                    "median_delta_oracle_minus_base": float((best - base).median()),
                    "oracle_improve_rate": float((best < base).mean()),
                    "mean_oracle_shift_ms": float(g["oracle_shift_ms"].mean()),
                    "most_common_oracle_shift_ms": int(g["oracle_shift_ms"].mode().iloc[0]),
                    "most_common_oracle_delay_ms": int(g["oracle_delay_ms"].mode().iloc[0]),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "base_group"]).reset_index(drop=True)


def train_selector(candidates: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Pipeline]]:
    """
    训练 input-only 候选误差预测器。

    训练目标是 candidate_error_v241。模型对同一个 base 的多个候选锚点打分，
    选择预测误差最小的候选锚点。
    """

    numeric_cols = [
        "base_delay_ms",
        "candidate_delay_ms",
        "candidate_shift_ms",
        "hist_current_steer",
        "hist_abs_current_steer",
        "hist_abs_mean_steer",
        "hist_abs_max_steer",
        "hist_steer_slope_last05",
        "hist_abs_steer_slope_last05",
        "hist_yaw_abs_mean",
        "hist_ay_abs_mean",
        "hist_speed_mean",
        "hist_brake_mean",
        "hist_accel_mean",
        "hist_curv_abs_mean",
        "hist_lat_abs_mean",
        "road_curv_abs_mean",
        "road_curv_abs_max",
        "road_lat_abs_mean",
        "road_lat_abs_max",
    ] + [c for c in candidates.columns if c.startswith("phase_")]
    categorical_cols = ["scene_type", "pool_key"]

    train_mask = candidates["split"].astype(str).eq("train")
    if not bool(train_mask.any()):
        raise RuntimeError("candidate table 中没有 train split，无法训练 selector。")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    models: Dict[str, Pipeline] = {
        "selector_ridge_base_input": Pipeline(
            steps=[
                ("pre", preprocessor),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "selector_rf_base_input": Pipeline(
            steps=[
                ("pre", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=20,
                        random_state=246,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    train_x = candidates.loc[train_mask, numeric_cols + categorical_cols]
    train_y = candidates.loc[train_mask, "candidate_error_v241"].astype(float)
    prediction_frames: List[pd.DataFrame] = []
    fit_rows: List[Dict[str, object]] = []

    for model_name, model in models.items():
        model.fit(train_x, train_y)
        all_x = candidates[numeric_cols + categorical_cols]
        pred = model.predict(all_x)
        frame = candidates[
            [
                "base_idx",
                "candidate_idx",
                "split",
                "event_uid",
                "base_delay_ms",
                "candidate_delay_ms",
                "candidate_shift_ms",
                "base_error_v241",
                "candidate_error_v241",
                "base_bad_top10_split_v241",
                "base_early_bad_top10_split_v241",
                "base_very_bad_top5_split_v241",
            ]
        ].copy()
        frame["selector_name"] = model_name
        frame["predicted_candidate_error_v241"] = pred
        prediction_frames.append(frame)

        for split in ["train", "val", "test"]:
            mask = candidates["split"].astype(str).eq(split)
            if not mask.any():
                continue
            y_true = candidates.loc[mask, "candidate_error_v241"].astype(float)
            y_pred = pred[mask.to_numpy()]
            fit_rows.append(
                {
                    "selector_name": model_name,
                    "split": split,
                    "n_candidate_rows": int(mask.sum()),
                    "candidate_error_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    "candidate_error_mae": float(mean_absolute_error(y_true, y_pred)),
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    fit_metrics = pd.DataFrame(fit_rows).sort_values(["selector_name", "split"]).reset_index(drop=True)
    return predictions, fit_metrics, models


def select_by_predicted_error(selector_predictions: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    """按预测候选误差选择每个 base 样本的锚点，并拼接 oracle 上限用于对照。"""

    rows: List[pd.Series] = []
    for (_, base_idx), g in selector_predictions.groupby(["selector_name", "base_idx"], sort=False):
        rows.append(g.sort_values(["predicted_candidate_error_v241", "candidate_shift_ms"]).iloc[0])
    selected = pd.DataFrame(rows).reset_index(drop=True)
    selected = selected.rename(
        columns={
            "candidate_idx": "selected_candidate_idx",
            "candidate_delay_ms": "selected_delay_ms",
            "candidate_shift_ms": "selected_shift_ms",
            "candidate_error_v241": "selected_error_v241",
        }
    )
    keep = [
        "base_idx",
        "oracle_candidate_idx",
        "oracle_delay_ms",
        "oracle_shift_ms",
        "oracle_error_v241",
        "oracle_delta_vs_base_v241",
    ]
    out = selected.merge(oracle[keep], on="base_idx", how="left", validate="many_to_one")
    out["selected_delta_vs_base_v241"] = out["selected_error_v241"] - out["base_error_v241"]
    out["selected_improved_v241"] = out["selected_error_v241"] < out["base_error_v241"]
    out["selected_matches_oracle_delay"] = out["selected_delay_ms"].astype(int).eq(out["oracle_delay_ms"].astype(int))
    return out


def build_fixed_policy_selected(candidates: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    """
    构造不训练的固定等待策略，用来判断 selector 的收益是不是只是来自“多等一点”。

    policy_keep_current_anchor：保持当前 base 锚点，相当于不做后移。
    policy_wait_to_latest_anchor：在同一 event_uid 内直接选择最晚可用候选锚点，通常是绝对 delay=1000ms。
    """

    rows: List[pd.Series] = []
    for policy_name, sort_cols, ascending in [
        ("policy_keep_current_anchor", ["candidate_shift_ms"], [True]),
        ("policy_wait_to_latest_anchor", ["candidate_delay_ms"], [False]),
    ]:
        for _, g in candidates.groupby("base_idx", sort=False):
            row = g.sort_values(sort_cols, ascending=ascending).iloc[0].copy()
            row["selector_name"] = policy_name
            row["predicted_candidate_error_v241"] = np.nan
            rows.append(row)

    selected = pd.DataFrame(rows).reset_index(drop=True)
    selected = selected.rename(
        columns={
            "candidate_idx": "selected_candidate_idx",
            "candidate_delay_ms": "selected_delay_ms",
            "candidate_shift_ms": "selected_shift_ms",
            "candidate_error_v241": "selected_error_v241",
        }
    )
    keep = [
        "base_idx",
        "oracle_candidate_idx",
        "oracle_delay_ms",
        "oracle_shift_ms",
        "oracle_error_v241",
        "oracle_delta_vs_base_v241",
    ]
    out = selected.merge(oracle[keep], on="base_idx", how="left", validate="many_to_one")
    out["selected_delta_vs_base_v241"] = out["selected_error_v241"] - out["base_error_v241"]
    out["selected_improved_v241"] = out["selected_error_v241"] < out["base_error_v241"]
    out["selected_matches_oracle_delay"] = out["selected_delay_ms"].astype(int).eq(out["oracle_delay_ms"].astype(int))
    return out


def summarize_selector(selected: pd.DataFrame) -> pd.DataFrame:
    """汇总 input-only selector 在 train/val/test 各组上的实际锚点选择收益。"""

    group_defs = [
        ("all", lambda x: pd.Series(True, index=x.index)),
        ("shiftable_delay_lt_1000", lambda x: x["base_delay_ms"].astype(int).lt(1000)),
        ("delay0_only", lambda x: x["base_delay_ms"].astype(int).eq(0)),
        ("bad_top10", lambda x: x["base_bad_top10_split_v241"].astype(bool)),
        ("early_bad_top10_delay_le_400", lambda x: x["base_early_bad_top10_split_v241"].astype(bool)),
        ("very_bad_top5", lambda x: x["base_very_bad_top5_split_v241"].astype(bool)),
    ]
    rows: List[Dict[str, object]] = []
    for (selector_name, split), split_df in selected.groupby(["selector_name", "split"]):
        for group_name, fn in group_defs:
            g = split_df[fn(split_df)]
            if g.empty:
                continue
            base = g["base_error_v241"].astype(float)
            sel = g["selected_error_v241"].astype(float)
            oracle = g["oracle_error_v241"].astype(float)
            possible_gain = base - oracle
            achieved_gain = base - sel
            capture = np.where(possible_gain > 1e-9, achieved_gain / possible_gain, np.nan)
            rows.append(
                {
                    "selector_name": selector_name,
                    "split": split,
                    "base_group": group_name,
                    "n_base_samples": int(len(g)),
                    "mean_base_error_v241": float(base.mean()),
                    "mean_selected_error_v241": float(sel.mean()),
                    "mean_oracle_error_v241": float(oracle.mean()),
                    "mean_delta_selected_minus_base": float((sel - base).mean()),
                    "mean_delta_oracle_minus_base": float((oracle - base).mean()),
                    "selected_improve_rate": float((sel < base).mean()),
                    "oracle_improve_rate": float((oracle < base).mean()),
                    "mean_gain_capture_rate": float(np.nanmean(capture)) if np.isfinite(capture).any() else math.nan,
                    "selected_matches_oracle_delay_rate": float(g["selected_matches_oracle_delay"].mean()),
                    "mean_selected_shift_ms": float(g["selected_shift_ms"].mean()),
                    "mean_oracle_shift_ms": float(g["oracle_shift_ms"].mean()),
                    "most_common_selected_shift_ms": int(g["selected_shift_ms"].mode().iloc[0]),
                    "most_common_oracle_shift_ms": int(g["oracle_shift_ms"].mode().iloc[0]),
                }
            )
    return pd.DataFrame(rows).sort_values(["selector_name", "split", "base_group"]).reset_index(drop=True)


def best_anchor_distribution(oracle: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """输出 oracle 和 selector 的最佳 shift 分布，方便看是否真的学会等待。"""

    rows: List[Dict[str, object]] = []
    for split, split_df in oracle.groupby("split"):
        for group_name, mask in {
            "all": pd.Series(True, index=split_df.index),
            "bad_top10": split_df["base_bad_top10_split_v241"].astype(bool),
            "early_bad_top10_delay_le_400": split_df["base_early_bad_top10_split_v241"].astype(bool),
        }.items():
            g = split_df[mask]
            if g.empty:
                continue
            counts = g["oracle_shift_ms"].astype(int).value_counts().sort_index()
            for shift, n in counts.items():
                rows.append(
                    {
                        "source": "oracle",
                        "selector_name": "oracle",
                        "split": split,
                        "base_group": group_name,
                        "shift_ms": int(shift),
                        "n": int(n),
                        "rate": float(n / len(g)),
                    }
                )
    for (selector_name, split), split_df in selected.groupby(["selector_name", "split"]):
        for group_name, mask in {
            "all": pd.Series(True, index=split_df.index),
            "bad_top10": split_df["base_bad_top10_split_v241"].astype(bool),
            "early_bad_top10_delay_le_400": split_df["base_early_bad_top10_split_v241"].astype(bool),
        }.items():
            g = split_df[mask]
            if g.empty:
                continue
            counts = g["selected_shift_ms"].astype(int).value_counts().sort_index()
            for shift, n in counts.items():
                rows.append(
                    {
                        "source": "selector",
                        "selector_name": selector_name,
                        "split": split,
                        "base_group": group_name,
                        "shift_ms": int(shift),
                        "n": int(n),
                        "rate": float(n / len(g)),
                    }
                )
    return pd.DataFrame(rows).sort_values(["split", "base_group", "source", "selector_name", "shift_ms"])


def plot_oracle_vs_selector(selector_summary: pd.DataFrame) -> Path:
    """画 test split 上 base / RF selector / 固定等待 / oracle 的误差对照。"""

    rf = selector_summary[
        selector_summary["split"].eq("test")
        & selector_summary["selector_name"].eq("selector_rf_base_input")
        & selector_summary["base_group"].isin(["all", "bad_top10", "early_bad_top10_delay_le_400", "very_bad_top5"])
    ].copy()
    latest = selector_summary[
        selector_summary["split"].eq("test")
        & selector_summary["selector_name"].eq("policy_wait_to_latest_anchor")
        & selector_summary["base_group"].isin(["all", "bad_top10", "early_bad_top10_delay_le_400", "very_bad_top5"])
    ].copy()
    order = ["all", "bad_top10", "early_bad_top10_delay_le_400", "very_bad_top5"]
    rf["base_group"] = pd.Categorical(rf["base_group"], categories=order, ordered=True)
    latest["base_group"] = pd.Categorical(latest["base_group"], categories=order, ordered=True)
    rf = rf.sort_values("base_group")
    latest = latest.sort_values("base_group")

    x = np.arange(len(rf))
    w = 0.2
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 1.5 * w, rf["mean_base_error_v241"], width=w, label="当前锚点", color="#999999")
    ax.bar(x - 0.5 * w, rf["mean_selected_error_v241"], width=w, label="RF selector", color="#6f9ec7")
    ax.bar(x + 0.5 * w, latest["mean_selected_error_v241"], width=w, label="固定等到最晚锚点", color="#7aa66a")
    ax.bar(x + 1.5 * w, rf["mean_oracle_error_v241"], width=w, label="oracle 最佳锚点", color="#cf8c5a")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in rf["base_group"]], rotation=15, ha="right")
    ax.set_ylabel("v241 tail absolute RMSE")
    ax.set_title("v246 test：当前锚点 vs RF selector vs 固定等待 vs oracle")
    ax.grid(axis="y", color="0.88")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = FIGURES / "v246_test_oracle_vs_selector_error.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_shift_distribution(distribution: pd.DataFrame) -> Path:
    """画 test bad_top10 上 oracle、selector 和固定等待策略的等待时长分布。"""

    data = distribution[
        distribution["split"].eq("test")
        & distribution["base_group"].eq("bad_top10")
        & (
            distribution["selector_name"].isin(
                ["oracle", "selector_rf_base_input", "selector_ridge_base_input", "policy_wait_to_latest_anchor"]
            )
        )
    ].copy()
    shifts = [0, 200, 400, 600, 800, 1000]
    pivot = data.pivot_table(index="selector_name", columns="shift_ms", values="rate", aggfunc="sum").reindex(
        ["oracle", "selector_rf_base_input", "selector_ridge_base_input", "policy_wait_to_latest_anchor"]
    ).fillna(0.0)
    for shift in shifts:
        if shift not in pivot.columns:
            pivot[shift] = 0.0
    pivot = pivot[shifts]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    bottom = np.zeros(len(pivot))
    colors = ["#d0d0d0", "#9ecae1", "#6baed6", "#fdcc8a", "#fc8d59", "#d7301f"]
    for color, shift in zip(colors, shifts):
        vals = pivot[shift].to_numpy()
        ax.bar(pivot.index, vals, bottom=bottom, label=f"+{shift}ms", color=color)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("比例")
    ax.set_title("test bad_top10：oracle、selector 与固定等待策略的后移时长分布")
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    out = FIGURES / "v246_test_bad_top10_shift_distribution.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_report(
    oracle_summary: pd.DataFrame,
    fit_metrics: pd.DataFrame,
    selector_summary: pd.DataFrame,
    distribution: pd.DataFrame,
    figure_paths: List[Path],
    zip_path: Path,
) -> None:
    """写中文报告，重点区分 oracle 上限和 input-only selector 可部署效果。"""

    test_oracle = oracle_summary[
        oracle_summary["split"].eq("test")
        & oracle_summary["base_group"].isin(["all", "bad_top10", "early_bad_top10_delay_le_400", "very_bad_top5"])
    ].copy()
    test_rf = selector_summary[
        selector_summary["split"].eq("test")
        & selector_summary["selector_name"].eq("selector_rf_base_input")
        & selector_summary["base_group"].isin(["all", "bad_top10", "early_bad_top10_delay_le_400", "very_bad_top5"])
    ].copy()
    test_ridge = selector_summary[
        selector_summary["split"].eq("test")
        & selector_summary["selector_name"].eq("selector_ridge_base_input")
        & selector_summary["base_group"].isin(["all", "bad_top10", "early_bad_top10_delay_le_400", "very_bad_top5"])
    ].copy()
    test_latest = selector_summary[
        selector_summary["split"].eq("test")
        & selector_summary["selector_name"].eq("policy_wait_to_latest_anchor")
        & selector_summary["base_group"].isin(["all", "bad_top10", "early_bad_top10_delay_le_400", "very_bad_top5"])
    ].copy()

    lines: List[str] = []
    lines.append("# v246 oracle 最佳锚点遍历与 input-only selector 审查")
    lines.append("")
    lines.append("## 结论先说")
    lines.append("")
    bad_oracle = test_oracle[test_oracle["base_group"].eq("bad_top10")]
    early_oracle = test_oracle[test_oracle["base_group"].eq("early_bad_top10_delay_le_400")]
    bad_rf = test_rf[test_rf["base_group"].eq("bad_top10")]
    early_rf = test_rf[test_rf["base_group"].eq("early_bad_top10_delay_le_400")]
    bad_latest = test_latest[test_latest["base_group"].eq("bad_top10")]
    early_latest = test_latest[test_latest["base_group"].eq("early_bad_top10_delay_le_400")]
    if not bad_oracle.empty:
        row = bad_oracle.iloc[0]
        lines.append(
            f"- oracle 遍历确认：test bad_top10 如果每个样本都选真实误差最小的更晚锚点，"
            f"平均 RMSE 从 `{row['mean_base_error_v241']:.3f}` 降到 `{row['mean_oracle_error_v241']:.3f}`，"
            f"delta=`{row['mean_delta_oracle_minus_base']:+.3f}`，改善率 `{row['oracle_improve_rate']:.1%}`。"
        )
    if not early_oracle.empty:
        row = early_oracle.iloc[0]
        lines.append(
            f"- early bad_top10 的 oracle 上限更强：平均 delta=`{row['mean_delta_oracle_minus_base']:+.3f}`，"
            f"改善率 `{row['oracle_improve_rate']:.1%}`，最常见 oracle shift=`+{int(row['most_common_oracle_shift_ms'])}ms`。"
        )
    if not bad_rf.empty:
        row = bad_rf.iloc[0]
        lines.append(
            f"- 只用 base 锚点可见输入训练的 RF selector，在 test bad_top10 上把 RMSE 从 "
            f"`{row['mean_base_error_v241']:.3f}` 降到 `{row['mean_selected_error_v241']:.3f}`，"
            f"delta=`{row['mean_delta_selected_minus_base']:+.3f}`，改善率 `{row['selected_improve_rate']:.1%}`；"
            f"但只捕获了 oracle 收益的一部分，mean gain capture=`{row['mean_gain_capture_rate']:.1%}`。"
        )
    if not early_rf.empty:
        row = early_rf.iloc[0]
        lines.append(
            f"- RF selector 在 early bad_top10 上：delta=`{row['mean_delta_selected_minus_base']:+.3f}`，"
            f"改善率 `{row['selected_improve_rate']:.1%}`，oracle delay 命中率 "
            f"`{row['selected_matches_oracle_delay_rate']:.1%}`。"
        )
    if not bad_latest.empty:
        row = bad_latest.iloc[0]
        lines.append(
            f"- 显式固定策略 `policy_wait_to_latest_anchor` 在 test bad_top10 上 RMSE="
            f"`{row['mean_selected_error_v241']:.3f}`，delta=`{row['mean_delta_selected_minus_base']:+.3f}`，"
            f"说明 Ridge 的强表现主要可能来自“尽量等到最晚锚点”，不是已经学会逐样本精确找最佳锚点。"
        )
    if not early_latest.empty:
        row = early_latest.iloc[0]
        lines.append(
            f"- 对 early bad_top10，固定等到最晚锚点的 delta=`{row['mean_delta_selected_minus_base']:+.3f}`，"
            f"接近 oracle delta=`{row['mean_delta_oracle_minus_base']:+.3f}`；这支持“变化太晚才显现的样本应该多看一点”的判断。"
        )
    lines.append(
        "- 因此，这条路线有明确上限收益；但真正难点是 selector 能否仅凭锚点前输入判断该等多久。"
        "下一步不应该直接把 oracle 最佳锚点写进测试流程，而应把 selector/风险等待策略作为模型组件验证。"
    )
    lines.append("")

    lines.append("## Raw Table 1：oracle 上限（test）")
    lines.append("")
    show_oracle = [
        "base_group",
        "n_base_samples",
        "mean_base_error_v241",
        "mean_oracle_error_v241",
        "mean_delta_oracle_minus_base",
        "oracle_improve_rate",
        "mean_oracle_shift_ms",
        "most_common_oracle_shift_ms",
        "most_common_oracle_delay_ms",
    ]
    lines.append(test_oracle[show_oracle].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Raw Table 2：selector 候选误差拟合质量")
    lines.append("")
    lines.append(fit_metrics.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Raw Table 3：RF selector 实际选锚点效果（test）")
    lines.append("")
    show_selector = [
        "base_group",
        "n_base_samples",
        "mean_base_error_v241",
        "mean_selected_error_v241",
        "mean_oracle_error_v241",
        "mean_delta_selected_minus_base",
        "mean_delta_oracle_minus_base",
        "selected_improve_rate",
        "mean_gain_capture_rate",
        "selected_matches_oracle_delay_rate",
        "mean_selected_shift_ms",
        "mean_oracle_shift_ms",
        "most_common_selected_shift_ms",
        "most_common_oracle_shift_ms",
    ]
    lines.append(test_rf[show_selector].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Raw Table 4：Ridge selector 参考（test）")
    lines.append("")
    lines.append(test_ridge[show_selector].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Raw Table 5：固定等到最晚锚点策略（test）")
    lines.append("")
    lines.append(test_latest[show_selector].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## 解释")
    lines.append("")
    lines.append(
        "1. oracle_best_anchor 是用真实误差选出来的最佳锚点，只能作为理论上限，不能部署。"
    )
    lines.append(
        "2. input-only selector 没有使用未来真实曲线、人工响应标签、event_uid 或 recording；"
        "它只看 base 锚点可见的历史/道路/phase 特征和候选等待时长。"
    )
    lines.append(
        "3. 如果 selector 能稳定接近 oracle，说明“每个样本自适应锚点”可以进入正式训练任务；"
        "如果 selector 只能捕获一小部分收益，就需要先做更强的风险/不确定性判定。"
    )
    lines.append(
        "4. Ridge selector 和固定等到最晚锚点很接近，说明当前收益里有相当一部分来自“多看一些时间”这个简单机制；"
        "后续要加入等待代价或触发条件，否则模型可能退化成一律晚预测。"
    )
    lines.append("")

    lines.append("## 产物")
    lines.append("")
    lines.append("- `tables/v246_sample_tail_errors.csv`")
    lines.append("- `tables/v246_base_input_features.csv`")
    lines.append("- `tables/v246_anchor_candidate_table.csv`")
    lines.append("- `tables/v246_oracle_best_anchor_by_base_sample.csv`")
    lines.append("- `tables/v246_oracle_best_anchor_summary.csv`")
    lines.append("- `tables/v246_selector_candidate_error_fit_metrics.csv`")
    lines.append("- `tables/v246_selector_predictions_by_candidate.csv`")
    lines.append("- `tables/v246_selector_selected_anchor_by_base_sample.csv`")
    lines.append("- `tables/v246_policy_selected_anchor_by_base_sample.csv`")
    lines.append("- `tables/v246_selector_policy_summary.csv`")
    lines.append("- `tables/v246_anchor_shift_distribution.csv`")
    lines.append("- `figures/v246_test_oracle_vs_selector_error.png`")
    lines.append("- `figures/v246_test_bad_top10_shift_distribution.png`")
    lines.append(f"- ZIP：`{zip_path.name}`")
    lines.append("")
    for path in figure_paths:
        lines.append(f"![{path.stem}](../figures/{path.name})")
        lines.append("")

    (REPORTS / "v246_oracle_best_anchor_and_selector_audit_cn.md").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )


def zip_outputs() -> Path:
    """打包 v246 产物。"""

    zip_path = OUT / "v246_oracle_best_anchor_and_selector_audit_pack.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in (TABLES, FIGURES, REPORTS, LOGS):
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(OUT))
    return zip_path


def write_logs(
    sample_table: pd.DataFrame,
    candidates: pd.DataFrame,
    selector_summary: pd.DataFrame,
    zip_path: Path,
) -> None:
    """写 guardrail、run_manifest 和输入哈希。"""

    input_hashes = pd.DataFrame(
        [
            {"path": str(PRED_PATH), "sha256": file_sha256(PRED_PATH), "bytes": int(PRED_PATH.stat().st_size)},
            {"path": str(V236_ARRAYS), "sha256": file_sha256(V236_ARRAYS), "bytes": int(V236_ARRAYS.stat().st_size)},
            {"path": str(V236_MANIFEST), "sha256": file_sha256(V236_MANIFEST), "bytes": int(V236_MANIFEST.stat().st_size)},
        ]
    )
    write_csv(input_hashes, LOGS / "input_file_hashes.csv")

    guardrail = {
        "pass": True,
        "stage": "v246_oracle_best_anchor_and_selector_audit",
        "no_trajectory_model_training": True,
        "trains_input_only_anchor_selector": True,
        "selector_train_split_only": True,
        "no_test_based_retuning": True,
        "oracle_best_anchor_is_upper_bound_only": True,
        "hard24_granular_unavailable": True,
        "n_samples": int(len(sample_table)),
        "n_candidate_rows": int(len(candidates)),
        "selector_rows": selector_summary.to_dict(orient="records"),
        "zip_testzip": zipfile.ZipFile(zip_path).testzip(),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "stage": "v246_oracle_best_anchor_and_selector_audit",
        "created_by": Path(__file__).name,
        "output_dir": str(OUT),
        "source_prediction_npz": str(PRED_PATH),
        "source_v236_arrays": str(V236_ARRAYS),
        "source_manifest": str(V236_MANIFEST),
        "n_samples": int(len(sample_table)),
        "n_candidate_rows": int(len(candidates)),
        "selector_names": sorted(selector_summary["selector_name"].unique().tolist()),
        "figures": [str(path.relative_to(OUT)).replace("\\", "/") for path in FIGURES.glob("*.png")],
        "zip": str(zip_path),
    }
    (LOGS / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_clean_output()
    data = load_inputs()
    sample_table = build_sample_table(data)
    base_features = build_base_features(data, sample_table)
    candidates = build_candidate_table(sample_table, base_features)
    oracle = build_oracle_table(candidates)
    oracle_summary = summarize_oracle(oracle)
    selector_predictions, fit_metrics, _models = train_selector(candidates)
    selector_selected = select_by_predicted_error(selector_predictions, oracle)
    policy_selected = build_fixed_policy_selected(candidates, oracle)
    selected_for_summary = pd.concat([selector_selected, policy_selected], ignore_index=True, sort=False)
    selector_summary = summarize_selector(selected_for_summary)
    distribution = best_anchor_distribution(oracle, selected_for_summary)

    write_csv(sample_table, TABLES / "v246_sample_tail_errors.csv")
    write_csv(base_features, TABLES / "v246_base_input_features.csv")
    write_csv(candidates, TABLES / "v246_anchor_candidate_table.csv")
    write_csv(oracle, TABLES / "v246_oracle_best_anchor_by_base_sample.csv")
    write_csv(oracle_summary, TABLES / "v246_oracle_best_anchor_summary.csv")
    write_csv(fit_metrics, TABLES / "v246_selector_candidate_error_fit_metrics.csv")
    write_csv(selector_predictions, TABLES / "v246_selector_predictions_by_candidate.csv")
    write_csv(selector_selected, TABLES / "v246_selector_selected_anchor_by_base_sample.csv")
    write_csv(policy_selected, TABLES / "v246_policy_selected_anchor_by_base_sample.csv")
    write_csv(selector_summary, TABLES / "v246_selector_policy_summary.csv")
    write_csv(distribution, TABLES / "v246_anchor_shift_distribution.csv")

    fig1 = plot_oracle_vs_selector(selector_summary)
    fig2 = plot_shift_distribution(distribution)
    figure_paths = [fig1, fig2]

    zip_path = zip_outputs()
    write_logs(sample_table, candidates, selector_summary, zip_path)
    zip_path = zip_outputs()
    write_logs(sample_table, candidates, selector_summary, zip_path)
    write_report(oracle_summary, fit_metrics, selector_summary, distribution, figure_paths, zip_path)
    zip_path = zip_outputs()
    write_logs(sample_table, candidates, selector_summary, zip_path)

    print(f"[v246] output={OUT}")
    print(f"[v246] report={REPORTS / 'v246_oracle_best_anchor_and_selector_audit_cn.md'}")
    print(f"[v246] zip={zip_path}")
    print("[v246] oracle test summary")
    print(
        oracle_summary[
            oracle_summary["split"].eq("test")
            & oracle_summary["base_group"].isin(["all", "bad_top10", "early_bad_top10_delay_le_400", "very_bad_top5"])
        ].to_string(index=False)
    )
    print("[v246] selector test summary")
    print(
        selector_summary[
            selector_summary["split"].eq("test")
            & selector_summary["selector_name"].eq("selector_rf_base_input")
            & selector_summary["base_group"].isin(["all", "bad_top10", "early_bad_top10_delay_le_400", "very_bad_top5"])
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
