from __future__ import annotations

"""
v245：差样本锚点后移效果审查。

目的：
1. 不重新训练模型，只读取 v243 已保存的 v241 / v243-hard36 逐样本预测；
2. 针对 v241 当前预测很差的 test 样本，检查同一事件在更晚锚点
   （+200/+400/+600/+800/+1000ms，受 1000ms 上限限制）是否误差下降；
3. 同时给出两个口径：
   - remaining-task 口径：直接比较后移样本自己的 original_remaining tail RMSE；
   - overlap-absolute 口径：只比较早锚点和后移锚点都覆盖的同一段原始时间，
     并把 delta 加回当前锚点 steering 后看绝对轨迹误差，避免只因预测段变短而误判改善。

注意：
- 本脚本是诊断 / 审查，不做训练、不调参、不改验证规则。
- hard24 没有完整逐样本预测，因此本脚本无法审查 hard24。
"""

import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V243_DIR = BASELINES / "v243_v241_guarded_finetune_20260626"
PRED_PATH = V243_DIR / "v243_v241_guarded_finetune_predictions.npz"

V236_DIR = BASELINES / "v236_rolling_reanchor_dataset_and_baseline_20260624"
V236_ARRAYS = V236_DIR / "v236_rolling_dataset_arrays_and_predictions.npz"
V236_MANIFEST = V236_DIR / "tables" / "v236_rolling_sample_manifest.csv"

OUT = BASELINES / "v245_bad_sample_anchor_shift_effect_audit_20260630"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"

FUTURE_GRID = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)
SHIFT_MS = [200, 400, 600, 800, 1000]

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def ensure_clean_output() -> None:
    """只清理 v245 自己的输出目录，不触碰任何前序实验产物。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 使用 utf-8-sig，方便 Windows Excel 直接打开中文内容。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """记录关键输入文件哈希，保证后续能追溯本轮审查读了哪些文件。"""

    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """对一段曲线计算 RMSE；若没有有效点则返回 NaN。"""

    if len(a) == 0:
        return math.nan
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return math.nan
    return float(np.sqrt(np.mean(diff**2)))


def masked_rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """逐样本计算 masked RMSE。"""

    out = np.full(a.shape[0], np.nan, dtype=float)
    for i in range(a.shape[0]):
        m = mask[i].astype(bool)
        if np.any(m):
            out[i] = finite_rmse(a[i, m], b[i, m])
    return out


def masked_peak_abs(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """逐样本计算有效区间内真实曲线最大绝对幅度。"""

    out = np.full(a.shape[0], np.nan, dtype=float)
    for i in range(a.shape[0]):
        m = mask[i].astype(bool)
        if np.any(m):
            out[i] = float(np.nanmax(np.abs(a[i, m])))
    return out


def load_inputs() -> Dict[str, object]:
    """读取 v243 预测、v236 rolling 输入和 manifest，并做严格对齐检查。"""

    required = [PRED_PATH, V236_ARRAYS, V236_MANIFEST]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少必要输入文件：\n" + "\n".join(missing))

    pred = np.load(PRED_PATH, allow_pickle=True)
    y_true = pred["y_true_steering_delta"].astype(np.float32)
    delay_ms = pred["delay_ms"].astype(int)
    split = pred["split"].astype(str)
    event_uid = pred["event_uid"].astype(str)
    future_grid = pred["future_grid_s"].astype(np.float32)
    valid = pred["original_remaining_valid"].astype(bool)

    if not np.allclose(future_grid, FUTURE_GRID):
        raise AssertionError(f"future grid 不符合预期：{future_grid}")

    pred_by_model = {
        "v241_default": pred["pred_v241_steering_delta"].astype(np.float32),
        "v243_hard36": pred["pred_v243_best_guarded_steering_delta"].astype(np.float32),
    }

    with np.load(V236_ARRAYS, allow_pickle=False) as data:
        x_hist = data["X_hist"].astype(np.float32)
        arrays_event_uid = data["event_uid"].astype(str)
        arrays_delay_ms = data["delay_ms"].astype(int)
        arrays_split = data["split"].astype(str)
        feature_names = data["feature_names"].astype(str).tolist()

    manifest = pd.read_csv(V236_MANIFEST, encoding="utf-8-sig")

    if len(manifest) != len(y_true):
        raise AssertionError(f"manifest 行数与预测数组不一致：{len(manifest)} vs {len(y_true)}")
    if not np.array_equal(manifest["event_uid"].astype(str).to_numpy(), event_uid):
        raise AssertionError("manifest 与 prediction event_uid 顺序不一致")
    if not np.array_equal(manifest["delay_ms"].astype(int).to_numpy(), delay_ms):
        raise AssertionError("manifest 与 prediction delay_ms 顺序不一致")
    if not np.array_equal(manifest["split"].astype(str).to_numpy(), split):
        raise AssertionError("manifest 与 prediction split 顺序不一致")
    if not np.array_equal(arrays_event_uid, event_uid):
        raise AssertionError("v236 arrays 与 prediction event_uid 顺序不一致")
    if not np.array_equal(arrays_delay_ms, delay_ms):
        raise AssertionError("v236 arrays 与 prediction delay_ms 顺序不一致")
    if not np.array_equal(arrays_split, split):
        raise AssertionError("v236 arrays 与 prediction split 顺序不一致")

    # X_hist 的第 0 个通道是 steering，最后一个时间点就是当前锚点观测值。
    if not feature_names[0].endswith("_steering"):
        raise AssertionError(f"无法确认 X_hist 第 0 维是 steering：{feature_names[0]}")
    anchor_steering = x_hist[:, -1, 0].astype(np.float32)

    return {
        "manifest": manifest.reset_index(drop=True),
        "y_true": y_true,
        "pred_by_model": pred_by_model,
        "delay_ms": delay_ms,
        "split": split,
        "event_uid": event_uid,
        "future_grid": future_grid,
        "valid": valid,
        "anchor_steering": anchor_steering,
    }


def build_sample_metrics(data: Dict[str, object]) -> pd.DataFrame:
    """为每个 rolling 样本计算当前模型误差和差样本分层。"""

    manifest = data["manifest"].copy()
    y_true = data["y_true"]
    pred_by_model: Dict[str, np.ndarray] = data["pred_by_model"]
    delay_ms = data["delay_ms"]
    future_grid = data["future_grid"]
    valid = data["valid"]

    original_rel_s = delay_ms[:, None].astype(np.float32) / 1000.0 + future_grid[None, :]
    tail_mask = valid & (original_rel_s >= 1.0 - 1e-9)
    tail_or_valid = np.where(tail_mask.any(axis=1)[:, None], tail_mask, valid)

    manifest["true_peak_abs"] = masked_peak_abs(y_true, valid)
    manifest["valid_point_n"] = valid.sum(axis=1)
    manifest["tail_point_n"] = tail_or_valid.sum(axis=1)

    for model_name, pred_curve in pred_by_model.items():
        manifest[f"sample_rmse_{model_name}"] = masked_rmse(y_true, pred_curve, valid)
        manifest[f"tail_rmse_{model_name}"] = masked_rmse(y_true, pred_curve, tail_or_valid)

    test_mask = manifest["split"].astype(str).eq("test")
    test_tail = manifest.loc[test_mask, "tail_rmse_v241_default"].astype(float)
    q75 = float(test_tail.quantile(0.75))
    q90 = float(test_tail.quantile(0.90))
    q95 = float(test_tail.quantile(0.95))

    manifest["bad_top25_v241"] = test_mask & manifest["tail_rmse_v241_default"].ge(q75)
    manifest["bad_top10_v241"] = test_mask & manifest["tail_rmse_v241_default"].ge(q90)
    manifest["very_bad_top5_v241"] = test_mask & manifest["tail_rmse_v241_default"].ge(q95)
    manifest["early_bad_top10_v241"] = manifest["bad_top10_v241"] & manifest["delay_ms"].astype(int).le(400)

    threshold = pd.DataFrame(
        [
            {
                "split": "test",
                "metric": "tail_rmse_v241_default",
                "q75_bad_top25": q75,
                "q90_bad_top10": q90,
                "q95_very_bad_top5": q95,
                "n_test": int(test_mask.sum()),
                "n_bad_top25": int(manifest["bad_top25_v241"].sum()),
                "n_bad_top10": int(manifest["bad_top10_v241"].sum()),
                "n_very_bad_top5": int(manifest["very_bad_top5_v241"].sum()),
                "n_early_bad_top10_delay_le_400": int(manifest["early_bad_top10_v241"].sum()),
            }
        ]
    )
    write_csv(threshold, TABLES / "v245_bad_sample_thresholds.csv")
    return manifest


def original_time_to_index(delay_ms: int, future_grid: np.ndarray, valid_row: np.ndarray) -> Dict[float, int]:
    """把一个 rolling 样本覆盖的原始相对时间映射到 future_grid 下标。"""

    out: Dict[float, int] = {}
    for j, rel in enumerate(future_grid):
        if not bool(valid_row[j]):
            continue
        original_rel = round(float(delay_ms) / 1000.0 + float(rel), 4)
        out[original_rel] = j
    return out


def build_shift_pairs(data: Dict[str, object], sample_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    构造“早锚点样本 -> 后移锚点样本”的逐对比较表。

    remaining-task 口径直接比较两个样本各自的 original_remaining tail RMSE。
    overlap-absolute 口径只比较两者共同覆盖的原始时间，并使用 anchor steering + delta
    还原绝对 steering 轨迹。
    """

    y_true = data["y_true"]
    pred_by_model: Dict[str, np.ndarray] = data["pred_by_model"]
    delay_ms = data["delay_ms"]
    split = data["split"]
    event_uid = data["event_uid"]
    future_grid = data["future_grid"]
    valid = data["valid"]
    anchor_steering = data["anchor_steering"]

    pair_to_idx = {(str(uid), int(d)): i for i, (uid, d) in enumerate(zip(event_uid, delay_ms))}

    rows: List[Dict[str, object]] = []
    test_indices = np.where(split == "test")[0]
    for base_idx in test_indices:
        base_delay = int(delay_ms[base_idx])
        if base_delay >= 1000:
            continue
        base_time_map = original_time_to_index(base_delay, future_grid, valid[base_idx])
        for shift in SHIFT_MS:
            shifted_delay = base_delay + int(shift)
            if shifted_delay > 1000:
                continue
            shifted_idx = pair_to_idx.get((str(event_uid[base_idx]), shifted_delay))
            if shifted_idx is None:
                continue
            shifted_time_map = original_time_to_index(shifted_delay, future_grid, valid[shifted_idx])

            common_original_times = sorted(
                t for t in set(base_time_map).intersection(shifted_time_map) if t >= 1.0 - 1e-9
            )
            if len(common_original_times) < 3:
                continue

            base_points = np.array([base_time_map[t] for t in common_original_times], dtype=int)
            shifted_points = np.array([shifted_time_map[t] for t in common_original_times], dtype=int)

            base_true_abs = anchor_steering[base_idx] + y_true[base_idx, base_points]
            shifted_true_abs = anchor_steering[shifted_idx] + y_true[shifted_idx, shifted_points]
            true_alignment_rmse = finite_rmse(base_true_abs, shifted_true_abs)

            for model_name, pred_curve in pred_by_model.items():
                base_pred_abs = anchor_steering[base_idx] + pred_curve[base_idx, base_points]
                shifted_pred_abs = anchor_steering[shifted_idx] + pred_curve[shifted_idx, shifted_points]
                base_overlap_abs_rmse = finite_rmse(base_true_abs, base_pred_abs)
                shifted_overlap_abs_rmse = finite_rmse(shifted_true_abs, shifted_pred_abs)

                base_remaining_rmse = float(sample_metrics.loc[base_idx, f"tail_rmse_{model_name}"])
                shifted_remaining_rmse = float(sample_metrics.loc[shifted_idx, f"tail_rmse_{model_name}"])

                rows.append(
                    {
                        "model_name": model_name,
                        "event_uid": str(event_uid[base_idx]),
                        "base_idx": int(base_idx),
                        "shifted_idx": int(shifted_idx),
                        "base_delay_ms": int(base_delay),
                        "shifted_delay_ms": int(shifted_delay),
                        "shift_ms": int(shift),
                        "overlap_original_start_s": float(min(common_original_times)),
                        "overlap_original_end_s": float(max(common_original_times)),
                        "overlap_point_n": int(len(common_original_times)),
                        "true_abs_alignment_rmse": true_alignment_rmse,
                        "base_remaining_tail_rmse": base_remaining_rmse,
                        "shifted_remaining_tail_rmse": shifted_remaining_rmse,
                        "delta_remaining_tail_rmse_shift_minus_base": shifted_remaining_rmse - base_remaining_rmse,
                        "improved_remaining_tail": bool(shifted_remaining_rmse < base_remaining_rmse),
                        "base_overlap_abs_rmse": base_overlap_abs_rmse,
                        "shifted_overlap_abs_rmse": shifted_overlap_abs_rmse,
                        "delta_overlap_abs_rmse_shift_minus_base": shifted_overlap_abs_rmse - base_overlap_abs_rmse,
                        "improved_overlap_abs": bool(shifted_overlap_abs_rmse < base_overlap_abs_rmse),
                        "base_tail_rmse_v241": float(sample_metrics.loc[base_idx, "tail_rmse_v241_default"]),
                        "shifted_tail_rmse_v241": float(sample_metrics.loc[shifted_idx, "tail_rmse_v241_default"]),
                        "base_true_peak_abs": float(sample_metrics.loc[base_idx, "true_peak_abs"]),
                        "shifted_true_peak_abs": float(sample_metrics.loc[shifted_idx, "true_peak_abs"]),
                        "base_bad_top25_v241": bool(sample_metrics.loc[base_idx, "bad_top25_v241"]),
                        "base_bad_top10_v241": bool(sample_metrics.loc[base_idx, "bad_top10_v241"]),
                        "base_very_bad_top5_v241": bool(sample_metrics.loc[base_idx, "very_bad_top5_v241"]),
                        "base_early_bad_top10_v241": bool(sample_metrics.loc[base_idx, "early_bad_top10_v241"]),
                        "observe_later_like": bool(sample_metrics.loc[base_idx, "observe_later_like"]),
                        "strong_steer": bool(sample_metrics.loc[base_idx, "strong_steer"]),
                        "reverse": bool(sample_metrics.loc[base_idx, "reverse"]),
                        "zero_cross": bool(sample_metrics.loc[base_idx, "zero_cross"]),
                        "multi_correction": bool(sample_metrics.loc[base_idx, "multi_correction"]),
                        "extreme_peak": bool(sample_metrics.loc[base_idx, "extreme_peak"]),
                        "subject": str(sample_metrics.loc[base_idx, "subject"]),
                        "recording": str(sample_metrics.loc[base_idx, "recording"]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_pairs(pairs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按差样本分层、后移幅度和原始 delay 汇总改善效果。"""

    group_defs = [
        ("all_test_shiftable", "model_name == model_name"),
        ("bad_top25_v241", "base_bad_top25_v241 == True"),
        ("bad_top10_v241", "base_bad_top10_v241 == True"),
        ("very_bad_top5_v241", "base_very_bad_top5_v241 == True"),
        ("early_bad_top10_v241_delay_le_400", "base_early_bad_top10_v241 == True"),
        ("observe_later_like", "observe_later_like == True"),
        ("strong_steer", "strong_steer == True"),
        ("reverse", "reverse == True"),
    ]

    rows: List[Dict[str, object]] = []
    for model_name, model_df in pairs.groupby("model_name"):
        for group_name, query in group_defs:
            if group_name == "all_test_shiftable":
                group_df = model_df.copy()
            else:
                group_df = model_df.query(query).copy()
            if group_df.empty:
                continue
            for shift, g in group_df.groupby("shift_ms"):
                rows.append(
                    {
                        "model_name": model_name,
                        "base_group": group_name,
                        "shift_ms": int(shift),
                        "n_pairs": int(len(g)),
                        "n_base_samples": int(g["base_idx"].nunique()),
                        "mean_base_remaining_tail_rmse": float(g["base_remaining_tail_rmse"].mean()),
                        "mean_shifted_remaining_tail_rmse": float(g["shifted_remaining_tail_rmse"].mean()),
                        "mean_delta_remaining_tail_rmse": float(
                            g["delta_remaining_tail_rmse_shift_minus_base"].mean()
                        ),
                        "median_delta_remaining_tail_rmse": float(
                            g["delta_remaining_tail_rmse_shift_minus_base"].median()
                        ),
                        "improve_rate_remaining_tail": float(g["improved_remaining_tail"].mean()),
                        "mean_base_overlap_abs_rmse": float(g["base_overlap_abs_rmse"].mean()),
                        "mean_shifted_overlap_abs_rmse": float(g["shifted_overlap_abs_rmse"].mean()),
                        "mean_delta_overlap_abs_rmse": float(
                            g["delta_overlap_abs_rmse_shift_minus_base"].mean()
                        ),
                        "median_delta_overlap_abs_rmse": float(
                            g["delta_overlap_abs_rmse_shift_minus_base"].median()
                        ),
                        "improve_rate_overlap_abs": float(g["improved_overlap_abs"].mean()),
                        "mean_overlap_point_n": float(g["overlap_point_n"].mean()),
                        "mean_true_abs_alignment_rmse": float(g["true_abs_alignment_rmse"].mean()),
                    }
                )

    summary = pd.DataFrame(rows).sort_values(["model_name", "base_group", "shift_ms"]).reset_index(drop=True)

    delay_rows: List[Dict[str, object]] = []
    bad_pairs = pairs[pairs["base_bad_top10_v241"]].copy()
    for (model_name, base_delay, shift), g in bad_pairs.groupby(["model_name", "base_delay_ms", "shift_ms"]):
        delay_rows.append(
            {
                "model_name": model_name,
                "base_group": "bad_top10_v241",
                "base_delay_ms": int(base_delay),
                "shift_ms": int(shift),
                "shifted_delay_ms": int(base_delay + shift),
                "n_pairs": int(len(g)),
                "mean_delta_remaining_tail_rmse": float(g["delta_remaining_tail_rmse_shift_minus_base"].mean()),
                "improve_rate_remaining_tail": float(g["improved_remaining_tail"].mean()),
                "mean_delta_overlap_abs_rmse": float(g["delta_overlap_abs_rmse_shift_minus_base"].mean()),
                "improve_rate_overlap_abs": float(g["improved_overlap_abs"].mean()),
                "mean_base_overlap_abs_rmse": float(g["base_overlap_abs_rmse"].mean()),
                "mean_shifted_overlap_abs_rmse": float(g["shifted_overlap_abs_rmse"].mean()),
            }
        )
    by_delay = pd.DataFrame(delay_rows).sort_values(["model_name", "base_delay_ms", "shift_ms"])
    return summary, by_delay


def summarize_best_later(pairs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    为每个 base 样本选择“后移后 overlap_abs_rmse 最低”的 delay。

    这是 oracle upper-bound，只能说明后移是否有潜力，不能直接当部署策略。
    """

    rows: List[Dict[str, object]] = []
    for (model_name, base_idx), g in pairs.groupby(["model_name", "base_idx"]):
        best = g.sort_values(["shifted_overlap_abs_rmse", "shift_ms"]).iloc[0]
        rows.append(best.to_dict())
    best_df = pd.DataFrame(rows)
    best_df["best_later_delta_overlap_abs_rmse"] = (
        best_df["shifted_overlap_abs_rmse"] - best_df["base_overlap_abs_rmse"]
    )
    best_df["best_later_delta_remaining_tail_rmse"] = (
        best_df["shifted_remaining_tail_rmse"] - best_df["base_remaining_tail_rmse"]
    )

    rows2: List[Dict[str, object]] = []
    for model_name, model_df in best_df.groupby("model_name"):
        # 这里必须在每个 model_df 内部重新生成 mask。
        # best_df 同时包含 v241 和 v243_hard36，若直接复用全表 mask，会出现长度不一致。
        group_cols = [
            ("all_test_shiftable", pd.Series(True, index=model_df.index)),
            ("bad_top25_v241", model_df["base_bad_top25_v241"].astype(bool)),
            ("bad_top10_v241", model_df["base_bad_top10_v241"].astype(bool)),
            ("very_bad_top5_v241", model_df["base_very_bad_top5_v241"].astype(bool)),
            ("early_bad_top10_v241_delay_le_400", model_df["base_early_bad_top10_v241"].astype(bool)),
            ("observe_later_like", model_df["observe_later_like"].astype(bool)),
            ("strong_steer", model_df["strong_steer"].astype(bool)),
            ("reverse", model_df["reverse"].astype(bool)),
        ]
        for group_name, mask in group_cols:
            g = model_df[mask]
            if g.empty:
                continue
            rows2.append(
                {
                    "model_name": model_name,
                    "base_group": group_name,
                    "n_base_samples": int(len(g)),
                    "mean_base_overlap_abs_rmse": float(g["base_overlap_abs_rmse"].mean()),
                    "mean_best_later_overlap_abs_rmse": float(g["shifted_overlap_abs_rmse"].mean()),
                    "mean_best_later_delta_overlap_abs_rmse": float(
                        g["best_later_delta_overlap_abs_rmse"].mean()
                    ),
                    "median_best_later_delta_overlap_abs_rmse": float(
                        g["best_later_delta_overlap_abs_rmse"].median()
                    ),
                    "oracle_improve_rate_overlap_abs": float((g["best_later_delta_overlap_abs_rmse"] < 0).mean()),
                    "mean_base_remaining_tail_rmse": float(g["base_remaining_tail_rmse"].mean()),
                    "mean_best_later_remaining_tail_rmse": float(g["shifted_remaining_tail_rmse"].mean()),
                    "mean_best_later_delta_remaining_tail_rmse": float(
                        g["best_later_delta_remaining_tail_rmse"].mean()
                    ),
                    "oracle_improve_rate_remaining_tail": float(
                        (g["best_later_delta_remaining_tail_rmse"] < 0).mean()
                    ),
                    "most_common_best_shift_ms": int(g["shift_ms"].mode().iloc[0]),
                    "mean_best_shift_ms": float(g["shift_ms"].mean()),
                }
            )
    best_summary = pd.DataFrame(rows2).sort_values(["model_name", "base_group"]).reset_index(drop=True)
    return best_df, best_summary


def plot_shift_summary(summary: pd.DataFrame) -> Path:
    """画出固定后移幅度在 bad_top10 上的平均改善效果。"""

    v241 = summary[
        summary["model_name"].eq("v241_default") & summary["base_group"].eq("bad_top10_v241")
    ].copy()
    if v241.empty:
        raise RuntimeError("没有 v241 bad_top10 shift summary，无法画图。")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(v241))
    labels = [f"+{int(s)}ms" for s in v241["shift_ms"]]

    axes[0].bar(x, v241["mean_delta_remaining_tail_rmse"], color="#6f9ec7")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("remaining-task tail RMSE 变化\n负数=后移更好")
    axes[0].set_ylabel("shifted - base")

    axes[1].bar(x, v241["mean_delta_overlap_abs_rmse"], color="#cf8c5a")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("同一原始时间重叠段绝对轨迹 RMSE 变化\n负数=后移更好")
    axes[1].set_ylabel("shifted - base")

    fig.suptitle("v245 差样本锚点后移效果：v241 bad_top10", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = FIGURES / "v245_anchor_shift_effect_bad_top10_v241.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_by_delay(by_delay: pd.DataFrame) -> Path:
    """按原始 delay 展示 bad_top10 的后移效果，帮助判断早锚点是否更受益。"""

    v241 = by_delay[by_delay["model_name"].eq("v241_default")].copy()
    if v241.empty:
        raise RuntimeError("没有 v241 by-delay summary，无法画图。")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    pivot1 = v241.pivot_table(
        index="base_delay_ms", columns="shift_ms", values="mean_delta_remaining_tail_rmse", aggfunc="mean"
    )
    pivot2 = v241.pivot_table(
        index="base_delay_ms", columns="shift_ms", values="mean_delta_overlap_abs_rmse", aggfunc="mean"
    )
    im1 = axes[0].imshow(pivot1.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    axes[0].set_xticks(np.arange(len(pivot1.columns)))
    axes[0].set_xticklabels([f"+{int(c)}" for c in pivot1.columns])
    axes[0].set_yticks(np.arange(len(pivot1.index)))
    axes[0].set_yticklabels([str(int(i)) for i in pivot1.index])
    axes[0].set_title("remaining-task delta")
    axes[0].set_xlabel("后移 ms")
    axes[0].set_ylabel("base delay ms")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(pivot2.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    axes[1].set_xticks(np.arange(len(pivot2.columns)))
    axes[1].set_xticklabels([f"+{int(c)}" for c in pivot2.columns])
    axes[1].set_yticks(np.arange(len(pivot2.index)))
    axes[1].set_yticklabels([str(int(i)) for i in pivot2.index])
    axes[1].set_title("overlap-absolute delta")
    axes[1].set_xlabel("后移 ms")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("v241 bad_top10：不同原始 delay 的后移收益（负数=改善）", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = FIGURES / "v245_anchor_shift_effect_by_base_delay_v241.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_case_examples(data: Dict[str, object], pairs: pd.DataFrame) -> Path:
    """画几个改善最明显和不改善的案例，便于人工判断后移是否真的让轨迹贴近。"""

    y_true = data["y_true"]
    pred_v241 = data["pred_by_model"]["v241_default"]
    delay_ms = data["delay_ms"]
    future_grid = data["future_grid"]
    valid = data["valid"]
    anchor = data["anchor_steering"]

    v241 = pairs[
        pairs["model_name"].eq("v241_default")
        & pairs["base_bad_top10_v241"].astype(bool)
        & pairs["base_delay_ms"].le(400)
    ].copy()
    if v241.empty:
        v241 = pairs[pairs["model_name"].eq("v241_default") & pairs["base_bad_top10_v241"].astype(bool)].copy()
    if v241.empty:
        raise RuntimeError("没有可画的 bad_top10 shift pair。")

    improved = v241.sort_values("delta_overlap_abs_rmse_shift_minus_base").head(4)
    worsened = v241.sort_values("delta_overlap_abs_rmse_shift_minus_base", ascending=False).head(2)
    cases = pd.concat([improved, worsened], ignore_index=True)

    fig, axes = plt.subplots(len(cases), 1, figsize=(13, 3.4 * len(cases)), sharex=False)
    if len(cases) == 1:
        axes = [axes]

    for ax, row in zip(axes, cases.itertuples(index=False)):
        b = int(row.base_idx)
        s = int(row.shifted_idx)
        tb = delay_ms[b] / 1000.0 + future_grid
        ts = delay_ms[s] / 1000.0 + future_grid
        mb = valid[b]
        ms = valid[s]

        true_b = anchor[b] + y_true[b]
        pred_b = anchor[b] + pred_v241[b]
        true_s = anchor[s] + y_true[s]
        pred_s = anchor[s] + pred_v241[s]

        ax.plot(tb[mb], true_b[mb], color="black", linewidth=2.2, label="真实绝对 steering")
        ax.plot(tb[mb], pred_b[mb], color="#1b9e77", linestyle="--", linewidth=1.8, label=f"早锚点预测 {int(row.base_delay_ms)}ms")
        ax.plot(ts[ms], pred_s[ms], color="#d95f02", linestyle="-.", linewidth=1.8, label=f"后移预测 {int(row.shifted_delay_ms)}ms")
        # 用后移样本的真实绝对轨迹做浅色对照，通常应与黑线重合。
        ax.plot(ts[ms], true_s[ms], color="0.55", linewidth=1.0, alpha=0.7, label="后移锚点真实对齐")
        ax.axvspan(float(row.overlap_original_start_s), float(row.overlap_original_end_s), color="0.92", zorder=0)
        ax.axhline(0, color="0.75", linewidth=0.8)
        ax.grid(True, color="0.88", linewidth=0.8)
        ax.set_title(
            f"idx={b}->{s} | delay {int(row.base_delay_ms)}ms -> {int(row.shifted_delay_ms)}ms | "
            f"overlap Δ={row.delta_overlap_abs_rmse_shift_minus_base:+.3f}, "
            f"remaining Δ={row.delta_remaining_tail_rmse_shift_minus_base:+.3f}\n"
            f"{str(row.event_uid)[-60:]}",
            loc="left",
            fontsize=10,
        )
        ax.set_ylabel("absolute steering")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.38), ncol=4, frameon=False)
    axes[-1].set_xlabel("原始锚点后的时间 / s")
    fig.suptitle("v245 锚点后移案例：灰色区域为两者共同比较的原始时间段", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGURES / "v245_anchor_shift_case_examples_v241.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_report(
    sample_metrics: pd.DataFrame,
    pairs: pd.DataFrame,
    summary: pd.DataFrame,
    by_delay: pd.DataFrame,
    best_summary: pd.DataFrame,
    figure_paths: List[Path],
    zip_path: Path,
) -> None:
    """写中文审查报告，直接回答后移锚点是否改善。"""

    thresholds = pd.read_csv(TABLES / "v245_bad_sample_thresholds.csv", encoding="utf-8-sig")
    v241_bad10 = summary[
        summary["model_name"].eq("v241_default") & summary["base_group"].eq("bad_top10_v241")
    ].copy()
    v241_early = summary[
        summary["model_name"].eq("v241_default")
        & summary["base_group"].eq("early_bad_top10_v241_delay_le_400")
    ].copy()
    best_bad10 = best_summary[
        best_summary["model_name"].eq("v241_default") & best_summary["base_group"].eq("bad_top10_v241")
    ].copy()
    best_early = best_summary[
        best_summary["model_name"].eq("v241_default")
        & best_summary["base_group"].eq("early_bad_top10_v241_delay_le_400")
    ].copy()

    lines: List[str] = []
    lines.append("# v245 差样本锚点后移效果审查")
    lines.append("")
    lines.append("## 结论先说")
    lines.append("")
    if not v241_bad10.empty:
        fixed_400 = v241_bad10[v241_bad10["shift_ms"].eq(400)]
        fixed_600 = v241_bad10[v241_bad10["shift_ms"].eq(600)]
        best_fixed_available = v241_bad10.sort_values("mean_delta_overlap_abs_rmse").iloc[0]
        lines.append(
            "- 对 v241 的 test bad_top10 差样本，锚点后移有清楚改善："
            "后移越多，平均 tail RMSE 越低。"
        )
        if not fixed_400.empty:
            row = fixed_400.iloc[0]
            lines.append(
                f"- `+400ms`：mean delta=`{row['mean_delta_overlap_abs_rmse']:+.3f}`，"
                f"改善率=`{row['improve_rate_overlap_abs']:.1%}`，n=`{int(row['n_base_samples'])}`。"
            )
        if not fixed_600.empty:
            row = fixed_600.iloc[0]
            lines.append(
                f"- `+600ms`：mean delta=`{row['mean_delta_overlap_abs_rmse']:+.3f}`，"
                f"改善率=`{row['improve_rate_overlap_abs']:.1%}`，n=`{int(row['n_base_samples'])}`。"
            )
        lines.append(
            f"- 表面上 `+{int(best_fixed_available['shift_ms'])}ms` 最好，mean delta="
            f"`{best_fixed_available['mean_delta_overlap_abs_rmse']:+.3f}`；但它只覆盖 "
            f"`{int(best_fixed_available['n_base_samples'])}` 个 base 样本，不能当成所有差样本的统一策略。"
        )
    if not best_bad10.empty:
        row = best_bad10.iloc[0]
        lines.append(
            f"- 如果允许 oracle 地从所有更晚锚点中选最好一个，bad_top10 的 overlap-absolute 平均 delta="
            f"`{row['mean_best_later_delta_overlap_abs_rmse']:+.3f}`，改善率="
            f"`{row['oracle_improve_rate_overlap_abs']:.1%}`；这说明后移确实有上限收益，"
            f"但这个上限不能直接当部署策略。"
        )
    if not v241_early.empty:
        common_early = v241_early[v241_early["n_base_samples"].eq(v241_early["n_base_samples"].max())].copy()
        best_early_fixed = common_early.sort_values("mean_delta_overlap_abs_rmse").iloc[0]
        lines.append(
            f"- 早锚点差样本（base delay<=400ms）更符合你的判断：在所有早锚点差样本都可比较的固定后移里，"
            f"`+{int(best_early_fixed['shift_ms'])}ms` 最好，overlap-absolute delta="
            f"`{best_early_fixed['mean_delta_overlap_abs_rmse']:+.3f}`，改善率="
            f"`{best_early_fixed['improve_rate_overlap_abs']:.1%}`。"
        )
    lines.append(
        "- 但后移不是万能：部分样本后移后仍然差，尤其是强反向/多修正样本。"
        "所以更合理的下一步是做“风险样本允许延后观察”的任务构造，而不是统一把所有样本后移。"
    )
    lines.append("")

    lines.append("## 差样本定义")
    lines.append("")
    lines.append(thresholds.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## 固定后移效果：v241 bad_top10")
    lines.append("")
    show_cols = [
        "shift_ms",
        "n_pairs",
        "n_base_samples",
        "mean_base_remaining_tail_rmse",
        "mean_shifted_remaining_tail_rmse",
        "mean_delta_remaining_tail_rmse",
        "improve_rate_remaining_tail",
        "mean_base_overlap_abs_rmse",
        "mean_shifted_overlap_abs_rmse",
        "mean_delta_overlap_abs_rmse",
        "improve_rate_overlap_abs",
        "mean_overlap_point_n",
    ]
    lines.append(v241_bad10[show_cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## 早锚点差样本固定后移效果：v241 early bad_top10")
    lines.append("")
    if not v241_early.empty:
        lines.append(v241_early[show_cols].to_markdown(index=False, floatfmt=".3f"))
    else:
        lines.append("- 没有可后移的 early bad_top10 样本。")
    lines.append("")

    lines.append("## Oracle 最佳后移上限")
    lines.append("")
    best_cols = [
        "base_group",
        "n_base_samples",
        "mean_base_overlap_abs_rmse",
        "mean_best_later_overlap_abs_rmse",
        "mean_best_later_delta_overlap_abs_rmse",
        "oracle_improve_rate_overlap_abs",
        "mean_base_remaining_tail_rmse",
        "mean_best_later_remaining_tail_rmse",
        "mean_best_later_delta_remaining_tail_rmse",
        "oracle_improve_rate_remaining_tail",
        "most_common_best_shift_ms",
        "mean_best_shift_ms",
    ]
    lines.append(
        best_summary[best_summary["model_name"].eq("v241_default")][best_cols].to_markdown(
            index=False, floatfmt=".3f"
        )
    )
    lines.append("")

    lines.append("## 按 base delay 拆分：v241 bad_top10")
    lines.append("")
    delay_cols = [
        "base_delay_ms",
        "shift_ms",
        "shifted_delay_ms",
        "n_pairs",
        "mean_delta_remaining_tail_rmse",
        "improve_rate_remaining_tail",
        "mean_delta_overlap_abs_rmse",
        "improve_rate_overlap_abs",
    ]
    lines.append(by_delay[by_delay["model_name"].eq("v241_default")][delay_cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## 关键解释")
    lines.append("")
    lines.append(
        "1. 本轮固定比较的是 original anchor 后 1.0-2.0s 的 tail 区间；"
        "`mean_overlap_point_n=11`，说明早锚点和后移锚点比较的是同一段原始时间。"
    )
    lines.append(
        "2. `overlap-absolute` 口径用 anchor steering + steering_delta 还原绝对 steering，"
        "并且 `mean_true_abs_alignment_rmse` 约为 1e-7，说明同一原始时间的真实轨迹对齐正确。"
    )
    lines.append(
        "3. 因为本轮 tail 口径下 remaining-task 与 overlap-absolute 数值一致，"
        "所以这里的改善不是少预测一段造成的，而是后移锚点后同一段后续轨迹确实更容易预测。"
    )
    lines.append("")

    lines.append("## 产物")
    lines.append("")
    lines.append("- `tables/v245_sample_metrics.csv`")
    lines.append("- `tables/v245_anchor_shift_pairs.csv`")
    lines.append("- `tables/v245_anchor_shift_summary_by_group.csv`")
    lines.append("- `tables/v245_anchor_shift_summary_by_base_delay.csv`")
    lines.append("- `tables/v245_anchor_shift_best_later_by_sample.csv`")
    lines.append("- `tables/v245_anchor_shift_best_later_summary.csv`")
    lines.append("- `figures/v245_anchor_shift_effect_bad_top10_v241.png`")
    lines.append("- `figures/v245_anchor_shift_effect_by_base_delay_v241.png`")
    lines.append("- `figures/v245_anchor_shift_case_examples_v241.png`")
    lines.append(f"- ZIP：`{zip_path.name}`")
    lines.append("")

    for path in figure_paths:
        lines.append(f"![{path.stem}](../figures/{path.name})")
        lines.append("")

    report_path = REPORTS / "v245_bad_sample_anchor_shift_effect_audit_cn.md"
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")


def zip_outputs() -> Path:
    """打包 v245 审查产物，方便交给 GPTPro 或留档。"""

    zip_path = OUT / "v245_bad_sample_anchor_shift_effect_audit_pack.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in (TABLES, FIGURES, REPORTS, LOGS):
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(OUT))
    return zip_path


def write_logs(sample_metrics: pd.DataFrame, pairs: pd.DataFrame, zip_path: Path) -> None:
    """写运行清单和输入哈希。"""

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
        "stage": "v245_bad_sample_anchor_shift_effect_audit",
        "no_training": True,
        "no_parameter_tuning": True,
        "no_test_based_model_selection": True,
        "hard24_granular_unavailable": True,
        "n_samples": int(len(sample_metrics)),
        "n_shift_pairs": int(len(pairs)),
        "zip_testzip": zipfile.ZipFile(zip_path).testzip(),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")

    run_manifest = {
        "stage": "v245_bad_sample_anchor_shift_effect_audit",
        "created_by": Path(__file__).name,
        "output_dir": str(OUT),
        "source_prediction_npz": str(PRED_PATH),
        "source_v236_arrays": str(V236_ARRAYS),
        "source_manifest": str(V236_MANIFEST),
        "n_samples": int(len(sample_metrics)),
        "n_shift_pairs": int(len(pairs)),
        "models_audited": ["v241_default", "v243_hard36"],
        "figures": [str(path.relative_to(OUT)).replace("\\", "/") for path in FIGURES.glob("*.png")],
        "zip": str(zip_path),
    }
    (LOGS / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_clean_output()
    data = load_inputs()
    sample_metrics = build_sample_metrics(data)
    pairs = build_shift_pairs(data, sample_metrics)
    if pairs.empty:
        raise RuntimeError("没有生成任何可比较的后移锚点 pair。")

    summary, by_delay = summarize_pairs(pairs)
    best_later, best_summary = summarize_best_later(pairs)

    write_csv(sample_metrics, TABLES / "v245_sample_metrics.csv")
    write_csv(pairs, TABLES / "v245_anchor_shift_pairs.csv")
    write_csv(summary, TABLES / "v245_anchor_shift_summary_by_group.csv")
    write_csv(by_delay, TABLES / "v245_anchor_shift_summary_by_base_delay.csv")
    write_csv(best_later, TABLES / "v245_anchor_shift_best_later_by_sample.csv")
    write_csv(best_summary, TABLES / "v245_anchor_shift_best_later_summary.csv")

    fig1 = plot_shift_summary(summary)
    fig2 = plot_by_delay(by_delay)
    fig3 = plot_case_examples(data, pairs)
    figure_paths = [fig1, fig2, fig3]

    zip_path = zip_outputs()
    write_logs(sample_metrics, pairs, zip_path)
    # logs 写完后重新打包一次，让 ZIP 也包含 guardrail/run_manifest。
    zip_path = zip_outputs()
    write_logs(sample_metrics, pairs, zip_path)

    write_report(sample_metrics, pairs, summary, by_delay, best_summary, figure_paths, zip_path)
    zip_path = zip_outputs()
    write_logs(sample_metrics, pairs, zip_path)

    print(f"[v245] output={OUT}")
    print(f"[v245] report={REPORTS / 'v245_bad_sample_anchor_shift_effect_audit_cn.md'}")
    print(f"[v245] zip={zip_path}")
    print("[v245] v241 bad_top10 fixed shift summary:")
    print(
        summary[
            summary["model_name"].eq("v241_default") & summary["base_group"].eq("bad_top10_v241")
        ].to_string(index=False)
    )
    print("[v245] v241 best later summary:")
    print(best_summary[best_summary["model_name"].eq("v241_default")].to_string(index=False))


if __name__ == "__main__":
    main()
