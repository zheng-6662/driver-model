# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
TRAINING_DIR = PROJECT_ROOT / "02_code" / "final_code" / "model" / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from event_conditioned_eval_support import annotate_event_meta, build_primary_selection_bundle
from run_event_conditioned_trajectory_baseline import DEFAULT_MANIFEST, build_sample_bundle_from_manifest
from run_g14_retrieval_reference import (
    add_physical_columns,
    build_available_feature_sets,
    build_neighbor_examples,
    df_to_markdown,
    group_rows,
    response_descriptor_features,
    split_indices,
    standardize_from_train,
    summarize_variant,
)


REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
OUT_DIR = REPORTS_DIR / "g15_retrieval_residual_20260512"
G11_CATALOG = REPORTS_DIR / "style_physio_eeg_g11_bad_case_attribution_20260509" / "bad_case_catalog.csv"
BASELINE_LOG = REPORTS_DIR / "current_model_version_result_log_20260509.csv"
RUN_META_PATH = OUT_DIR / "g15_run_meta.json"

K_VALUES = [1, 3, 5, 10, 20]
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
RESIDUAL_SCALES = [0.0, 0.25, 0.50, 0.75, 1.00]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def ensure_mask2d(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        return arr[:, :, 0].astype(np.float32)
    return arr.astype(np.float32)


def nearest_neighbors_between(
    z_features: np.ndarray,
    ref_idx: np.ndarray,
    query_idx: np.ndarray,
    max_k: int,
    exclude_self: bool = False,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    ref_feat = z_features[ref_idx].astype(np.float32)
    query_feat = z_features[query_idx].astype(np.float32)
    ref_pos = {int(idx): pos for pos, idx in enumerate(ref_idx.tolist())}
    keep = min(max_k, max(1, len(ref_idx) - (1 if exclude_self else 0)))
    all_indices: list[np.ndarray] = []
    all_dist: list[np.ndarray] = []
    for start in range(0, len(query_idx), chunk_size):
        end = min(len(query_idx), start + chunk_size)
        q = query_feat[start:end]
        dist = (
            np.sum(q * q, axis=1, keepdims=True)
            + np.sum(ref_feat * ref_feat, axis=1, keepdims=True).T
            - 2.0 * q @ ref_feat.T
        )
        dist = np.maximum(dist, 0.0)
        if exclude_self:
            for local_i, global_i in enumerate(query_idx[start:end].tolist()):
                pos = ref_pos.get(int(global_i))
                if pos is not None:
                    dist[local_i, pos] = np.inf
        kth = min(keep - 1, dist.shape[1] - 1)
        part = np.argpartition(dist, kth=kth, axis=1)[:, :keep]
        part_dist = np.take_along_axis(dist, part, axis=1)
        order = np.argsort(part_dist, axis=1)
        part = np.take_along_axis(part, order, axis=1)
        part_dist = np.take_along_axis(part_dist, order, axis=1)
        all_indices.append(ref_idx[part])
        all_dist.append(np.sqrt(part_dist).astype(np.float32))
    return np.concatenate(all_indices, axis=0), np.concatenate(all_dist, axis=0)


def predict_from_neighbors(
    y_pool: np.ndarray,
    query_idx: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    k: int,
) -> np.ndarray:
    keep = min(k, neighbor_indices.shape[1])
    idx = neighbor_indices[:, :keep]
    dist = neighbor_distances[:, :keep]
    weights = 1.0 / (dist + 1e-3)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    neighbors = y_pool[idx].astype(np.float32)
    pred = np.sum(neighbors * weights[:, :, None, None], axis=1)
    return pred.astype(np.float32)


def masked_rmse(pred: np.ndarray, true: np.ndarray, mask2d: np.ndarray) -> float:
    valid = mask2d > 0.5
    if not np.any(valid):
        return float("nan")
    diff = pred[:, :, 0] - true[:, :, 0]
    return float(np.sqrt(np.mean((diff[valid]) ** 2)))


def downsample_curve(curve: np.ndarray, points: int = 24) -> np.ndarray:
    arr = np.asarray(curve, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    idx = np.linspace(0, arr.shape[1] - 1, points).round().astype(np.int64)
    return arr[:, idx].astype(np.float32)


def weighted_neighbor_descriptors(
    descriptor_pool: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    k: int,
) -> np.ndarray:
    keep = min(k, neighbor_indices.shape[1])
    idx = neighbor_indices[:, :keep]
    dist = neighbor_distances[:, :keep]
    weights = 1.0 / (dist + 1e-3)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    desc = descriptor_pool[idx].astype(np.float32)
    return np.sum(desc * weights[:, :, None], axis=1).astype(np.float32)


def residual_features(
    z_features: np.ndarray,
    query_idx: np.ndarray,
    base_pred: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    k: int,
    descriptor_pool: np.ndarray,
) -> np.ndarray:
    keep = min(k, neighbor_distances.shape[1])
    dist = neighbor_distances[:, :keep].astype(np.float32)
    dist_stats = np.stack(
        [
            np.nanmin(dist, axis=1),
            np.nanmean(dist, axis=1),
            np.nanstd(dist, axis=1),
            np.nanmax(dist, axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    neighbor_desc = weighted_neighbor_descriptors(descriptor_pool, neighbor_indices, neighbor_distances, keep)
    pred_curve = downsample_curve(base_pred, points=24)
    return np.concatenate([z_features[query_idx].astype(np.float32), dist_stats, neighbor_desc, pred_curve], axis=1).astype(np.float32)


def standardize_matrix(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], dict[str, list[float]]]:
    x = np.asarray(train_x, dtype=np.float32)
    mu = np.nanmean(x, axis=0).astype(np.float32)
    sd = np.nanstd(x, axis=0).astype(np.float32)
    mu[~np.isfinite(mu)] = 0.0
    sd[~np.isfinite(sd)] = 1.0
    sd[sd < 1e-6] = 1.0

    def transform(arr: np.ndarray) -> np.ndarray:
        work = np.asarray(arr, dtype=np.float32).copy()
        bad = ~np.isfinite(work)
        if np.any(bad):
            rows, cols = np.where(bad)
            work[rows, cols] = mu[cols]
        return ((work - mu.reshape(1, -1)) / sd.reshape(1, -1)).astype(np.float32)

    return transform(x), [transform(arr) for arr in others], {"mean": mu.tolist(), "std": sd.tolist()}


def fit_ridge_multioutput(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xb = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float32), x.astype(np.float32)], axis=1)
    a = xb.T @ xb
    reg = np.eye(a.shape[0], dtype=np.float32) * float(alpha)
    reg[0, 0] = 0.0
    b = xb.T @ y.astype(np.float32)
    return np.linalg.solve(a + reg, b).astype(np.float32)


def predict_ridge(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    xb = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float32), x.astype(np.float32)], axis=1)
    return (xb @ weights).astype(np.float32)


def evaluate_prediction(
    model_id: str,
    feature_set: str,
    k: int,
    split_name: str,
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    ctx: np.ndarray,
    meta: pd.DataFrame,
    g11_keys: set[str],
    alpha: float | None = None,
    residual_scale: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    mask2d = ensure_mask2d(mask)
    bundle = build_primary_selection_bundle(
        pred=pred,
        true=true,
        mask=mask2d,
        ctx_raw=ctx,
        meta_df=meta,
        split_name=split_name,
        seed=2026,
    )
    sample_df = add_physical_columns(bundle["sample_df"], pred, true, mask2d, ctx)
    sample_df["model_id"] = model_id
    sample_df["feature_set"] = feature_set
    sample_df["k"] = int(k)
    row = summarize_variant(feature_set, k, sample_df, bundle["selection_summary"], g11_keys)
    row.update(
        {
            "model_id": model_id,
            "split": split_name,
            "alpha": float(alpha) if alpha is not None else np.nan,
            "residual_scale": float(residual_scale) if residual_scale is not None else np.nan,
        }
    )
    return row, sample_df, bundle["selection_summary"]


def load_prediction_map(pattern: str) -> dict[str, np.ndarray]:
    root = PROJECT_ROOT / "tmp" / "event_conditioned_runs"
    candidates = []
    for run_dir in root.glob(pattern):
        seq = run_dir / "prediction_figures" / "test" / "prediction_sequences.npz"
        if seq.exists():
            candidates.append(seq)
    if not candidates:
        return {}
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    data = np.load(candidates[0], allow_pickle=True)
    return {str(key): data["pred"][i].astype(np.float32) for i, key in enumerate(data["sample_key"].astype(str).tolist())}


def plot_selected_cases(
    plot_dir: Path,
    meta_test: pd.DataFrame,
    y_test: np.ndarray,
    ctx_test: np.ndarray,
    pred_map: dict[str, np.ndarray],
    selected_keys: list[str],
    baseline_maps: dict[str, dict[str, np.ndarray]],
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_test.reset_index(drop=True).copy()
    key_to_idx = {str(row.sample_key): int(i) for i, row in meta.iterrows()}
    keys = [key for key in selected_keys if key in key_to_idx][:12]
    if not keys:
        return
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True)
    axes = axes.reshape(-1)
    t = np.arange(y_test.shape[1]) * 2.0 / max(1, y_test.shape[1])
    for ax, key in zip(axes, keys):
        i = key_to_idx[key]
        anchor = float(ctx_test[i, 0])
        ax.plot(t, y_test[i, :, 0] + anchor, color="black", linewidth=2.0, label="真实")
        for label, bmap in baseline_maps.items():
            if key in bmap:
                ax.plot(t, bmap[key][:, 0] + anchor, linewidth=1.0, alpha=0.72, label=label)
        for label, pred in pred_map.items():
            ax.plot(t, pred[i, :, 0] + anchor, linewidth=1.4, alpha=0.90, label=label)
        row = meta.iloc[i]
        ax.set_title(f"{row.get('subj','?')} | {row.get('eval_morphology_label','?')}", fontsize=9)
        ax.axhline(0.0, color="#999999", linewidth=0.6)
        ax.grid(True, alpha=0.2)
    for ax in axes[len(keys) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(plot_dir / "g15_selected_g11_comparison.png", dpi=170)
    plt.close(fig)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        f = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(f):
        return "NA"
    return f"{f:.{digits}f}"


def build_report(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    chosen_df: pd.DataFrame,
    group_df: pd.DataFrame,
    g11_df: pd.DataFrame,
) -> str:
    baseline_text = "基准表读取失败或不存在。"
    if BASELINE_LOG.exists():
        base = pd.read_csv(BASELINE_LOG)
        keep = base[base["version"].astype(str).isin(["E2", "E5A", "E6", "E10C"])].copy()
        cols = [c for c in ["version", "test_rmse", "primary_rmse", "tail_rmse", "selection", "decision"] if c in keep.columns]
        baseline_text = df_to_markdown(keep[cols])

    val_cols = ["model_id", "feature_set", "k", "test_rmse", "tail_rmse", "selection_score", "alpha", "residual_scale"]
    val_show = val_df.sort_values(["selection_score", "test_rmse"]).head(12)
    test_cols = [
        "model_id",
        "feature_set",
        "k",
        "test_rmse",
        "tail_rmse",
        "selection_score",
        "g11_rmse",
        "large_rmse",
        "reverse_rmse",
        "multi_rmse",
        "severe_under_amp_rate",
        "opposite_peak_rate",
        "alpha",
        "residual_scale",
    ]
    chosen_text = df_to_markdown(chosen_df[[c for c in test_cols if c in chosen_df.columns]])

    g11_text = "无 G11 逐样本结果。"
    if not g11_df.empty:
        g11_summary = g11_df.groupby(["model_id", "feature_set", "k"]).agg(
            sample_count=("sample_key", "count"),
            rmse=("rmse_2s_abs_steer", "mean"),
            tail_rmse=("rmse_tail_abs_steer", "mean"),
            severe_under_amp_rate=("severe_under_amp", "mean"),
            opposite_peak_rate=("opposite_at_true_peak", "mean"),
        ).reset_index()
        g11_text = df_to_markdown(g11_summary)

    subj_text = "无分被试结果。"
    if not group_df.empty:
        subj = group_df[group_df["group_family"].eq("subj")].copy()
        subj = subj[subj["model_id"].isin(chosen_df["model_id"].astype(str).tolist())]
        subj_text = df_to_markdown(subj[["model_id", "group_label", "sample_count", "rmse", "tail_rmse", "severe_under_amp_rate", "opposite_peak_rate"]])

    return f"""# G15 路线1：相似历史事件检索与残差修正报告

## 1. 这轮为什么做

旧流程中最大问题不是只差一点 RMSE，而是很多预测图存在物理意义错误：真实大幅打方向被预测成轻微变化，反向修正和多段修正被平均成平滑轨迹，尾段回正或漂移不合理。

G14 已经说明：训练集中存在很多“如果选对就很像”的历史响应，但普通欧氏距离检索不能稳定找到困难样本对应的响应类型。因此 G15 路线1继续推进相似历史事件，但做两个改进：

1. **G15A：验证集选参的相似历史检索基线**。不再在测试集上挑 K，而是在验证集选择特征组和 K，再到测试集汇报。
2. **G15B：相似历史检索 + 残差修正**。先用相似历史事件得到一条参考轨迹，再用训练集学习一个小的残差修正，判断是否能修复普通检索的系统性偏差。

## 2. 公平边界

- 样本清单仍使用：`{DEFAULT_MANIFEST}`。
- 检索和残差修正只用训练集拟合。
- K、残差正则强度和残差缩放只用验证集选择。
- 测试集只用于最后汇报。
- 推理输入只包含触发前车辆/事件信息、连续驾驶风格、肌电；不使用真实未来响应标签。

## 3. 当前强基准

{baseline_text}

## 4. 验证集筛选前 12 个设置

{df_to_markdown(val_show[[c for c in val_cols if c in val_show.columns]])}

## 5. 验证集选出的版本在测试集上的结果

{chosen_text}

## 6. G11 困难样本

{g11_text}

## 7. 分被试结果

{subj_text}

## 8. 初步判断口径

如果 G15A 明显改善，说明“直接拿训练集相似历史响应做参考”就有价值，旧模型的问题可能是没有充分利用历史相似样本。

如果 G15B 比 G15A 明显改善，说明相似历史轨迹有用，但还需要一个可学习的偏差修正模块。

如果 G15A/G15B 全体 RMSE 好但 G11 困难样本仍差，说明相似检索对普通样本有帮助，但还没有解决真正的物理困难样本，下一步应转向“响应类型先判别”或“响应类型绑定候选轨迹”。

## 9. 输出文件

- `g15_validation_screening.csv`：验证集筛选结果；
- `g15_test_all_candidates.csv`：测试集全部候选结果，仅用于透明展示；
- `g15_test_chosen_by_validation.csv`：验证集选中版本的测试结果；
- `g15_group_summary.csv`：分被试、分响应类型、分幅值统计；
- `g15_g11_detail.csv`：G11 困难样本逐样本结果；
- `g15_neighbor_examples.csv`：代表性困难样本的相似训练事件；
- `g15_chosen_predictions_test.npz`：选中版本测试预测序列；
- `figures/g15_selected_g11_comparison.png`：困难样本对比图。
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("G15 route1: loading FAIR sample bundle...", flush=True)
    x_pool, y_pool, _curve_pool, ctx_pool, mask_pool, meta_df, dropped = build_sample_bundle_from_manifest(
        DEFAULT_MANIFEST,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        seed=2026,
    )
    train_idx, val_idx, test_idx = split_indices(meta_df)
    meta_annotated = annotate_event_meta(meta_df, y_pool, mask_pool)

    meta_train = meta_annotated.iloc[train_idx].reset_index(drop=True)
    meta_val = meta_annotated.iloc[val_idx].reset_index(drop=True)
    meta_test = meta_annotated.iloc[test_idx].reset_index(drop=True)

    y_train = y_pool[train_idx].astype(np.float32)
    y_val = y_pool[val_idx].astype(np.float32)
    y_test = y_pool[test_idx].astype(np.float32)
    mask_train = ensure_mask2d(mask_pool[train_idx])
    mask_val = ensure_mask2d(mask_pool[val_idx])
    mask_test = ensure_mask2d(mask_pool[test_idx])
    ctx_train = ctx_pool[train_idx].astype(np.float32)
    ctx_val = ctx_pool[val_idx].astype(np.float32)
    ctx_test = ctx_pool[test_idx].astype(np.float32)

    g11_catalog = pd.read_csv(G11_CATALOG) if G11_CATALOG.exists() else pd.DataFrame()
    g11_keys = set(g11_catalog["sample_key"].astype(str).tolist()) if not g11_catalog.empty else set()

    feature_sets, feature_names, context_meta = build_available_feature_sets(x_pool, ctx_pool, meta_annotated, train_idx)
    deployable_feature_sets = {
        key: value
        for key, value in feature_sets.items()
        if key in {"触发前车辆和事件信息", "触发前车辆事件加连续风格", "触发前车辆事件加连续风格和肌电"}
    }
    descriptor_pool, descriptor_names = response_descriptor_features(y_pool, ensure_mask2d(mask_pool))

    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    group_rows_all: list[dict[str, Any]] = []
    g11_frames: list[pd.DataFrame] = []
    neighbor_frames: list[pd.DataFrame] = []
    chosen_preds: dict[str, np.ndarray] = {}
    chosen_rows: list[dict[str, Any]] = []

    for feature_set, raw_features in deployable_feature_sets.items():
        print(f"G15 feature set: {feature_set}", flush=True)
        z, stat = standardize_from_train(raw_features, train_idx)
        save_json(
            OUT_DIR / f"feature_stats_{feature_set}.json",
            {
                "feature_set": feature_set,
                "feature_count": int(raw_features.shape[1]),
                "feature_names": feature_names.get(feature_set, []),
                "standardization": stat,
            },
        )
        max_k = max(K_VALUES)
        neigh_train, dist_train = nearest_neighbors_between(z, train_idx, train_idx, max_k=max_k, exclude_self=True)
        neigh_val, dist_val = nearest_neighbors_between(z, train_idx, val_idx, max_k=max_k, exclude_self=False)
        neigh_test, dist_test = nearest_neighbors_between(z, train_idx, test_idx, max_k=max_k, exclude_self=False)

        for k in K_VALUES:
            base_train = predict_from_neighbors(y_pool, train_idx, neigh_train, dist_train, k=k)
            base_val = predict_from_neighbors(y_pool, val_idx, neigh_val, dist_val, k=k)
            base_test = predict_from_neighbors(y_pool, test_idx, neigh_test, dist_test, k=k)

            val_row, _val_sample, _ = evaluate_prediction(
                "G15A_相似历史检索",
                feature_set,
                k,
                "val",
                base_val,
                y_val,
                mask_val,
                ctx_val,
                meta_val,
                set(),
            )
            test_row, test_sample, _ = evaluate_prediction(
                "G15A_相似历史检索",
                feature_set,
                k,
                "test",
                base_test,
                y_test,
                mask_test,
                ctx_test,
                meta_test,
                g11_keys,
            )
            val_rows.append(val_row)
            test_rows.append(test_row)
            group_rows_all.extend({**item, "model_id": "G15A_相似历史检索"} for item in group_rows(feature_set, k, test_sample))
            g11_part = test_sample[test_sample["sample_key"].astype(str).isin(g11_keys)].copy()
            if not g11_part.empty:
                g11_frames.append(g11_part)

            x_train = residual_features(z, train_idx, base_train, neigh_train, dist_train, k, descriptor_pool)
            x_val = residual_features(z, val_idx, base_val, neigh_val, dist_val, k, descriptor_pool)
            x_test = residual_features(z, test_idx, base_test, neigh_test, dist_test, k, descriptor_pool)
            x_train_z, others_z, _resid_stat = standardize_matrix(x_train, x_val, x_test)
            x_val_z, x_test_z = others_z
            y_resid = (y_train[:, :, 0] - base_train[:, :, 0]).astype(np.float32)

            best_cfg: dict[str, float] = {"alpha": ALPHAS[0], "scale": 0.0, "val_rmse": masked_rmse(base_val, y_val, mask_val)}
            best_val_pred = base_val
            best_test_pred = base_test
            for alpha in ALPHAS:
                weights = fit_ridge_multioutput(x_train_z, y_resid, alpha=alpha)
                val_delta = predict_ridge(x_val_z, weights)
                test_delta = predict_ridge(x_test_z, weights)
                for scale in RESIDUAL_SCALES:
                    pred_val = base_val.copy()
                    pred_val[:, :, 0] = pred_val[:, :, 0] + float(scale) * val_delta
                    val_rmse = masked_rmse(pred_val, y_val, mask_val)
                    if val_rmse < best_cfg["val_rmse"]:
                        pred_test = base_test.copy()
                        pred_test[:, :, 0] = pred_test[:, :, 0] + float(scale) * test_delta
                        best_cfg = {"alpha": float(alpha), "scale": float(scale), "val_rmse": float(val_rmse)}
                        best_val_pred = pred_val
                        best_test_pred = pred_test

            val_row_b, _val_sample_b, _ = evaluate_prediction(
                "G15B_检索加残差修正",
                feature_set,
                k,
                "val",
                best_val_pred,
                y_val,
                mask_val,
                ctx_val,
                meta_val,
                set(),
                alpha=best_cfg["alpha"],
                residual_scale=best_cfg["scale"],
            )
            test_row_b, test_sample_b, _ = evaluate_prediction(
                "G15B_检索加残差修正",
                feature_set,
                k,
                "test",
                best_test_pred,
                y_test,
                mask_test,
                ctx_test,
                meta_test,
                g11_keys,
                alpha=best_cfg["alpha"],
                residual_scale=best_cfg["scale"],
            )
            val_row_b["fast_val_rmse_used_for_residual_search"] = best_cfg["val_rmse"]
            val_rows.append(val_row_b)
            test_rows.append(test_row_b)
            group_rows_all.extend({**item, "model_id": "G15B_检索加残差修正"} for item in group_rows(feature_set, k, test_sample_b))
            g11_part_b = test_sample_b[test_sample_b["sample_key"].astype(str).isin(g11_keys)].copy()
            if not g11_part_b.empty:
                g11_frames.append(g11_part_b)

            if k == 5:
                neighbor_frames.append(
                    build_neighbor_examples(
                        meta_test=meta_test,
                        meta_all=meta_annotated,
                        result=type("RetrievalResult", (), {
                            "feature_set": feature_set,
                            "k": k,
                            "pred": base_test,
                            "neighbor_indices": neigh_test,
                            "neighbor_distances": dist_test,
                        })(),
                        feature_set=feature_set,
                        g11_keys=g11_keys,
                    )
                )

    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    group_df = pd.DataFrame(group_rows_all)
    g11_detail_df = pd.concat(g11_frames, ignore_index=True) if g11_frames else pd.DataFrame()
    neighbor_df = pd.concat(neighbor_frames, ignore_index=True) if neighbor_frames else pd.DataFrame()

    val_df.to_csv(OUT_DIR / "g15_validation_screening.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_DIR / "g15_test_all_candidates.csv", index=False, encoding="utf-8-sig")
    group_df.to_csv(OUT_DIR / "g15_group_summary.csv", index=False, encoding="utf-8-sig")
    g11_detail_df.to_csv(OUT_DIR / "g15_g11_detail.csv", index=False, encoding="utf-8-sig")
    neighbor_df.to_csv(OUT_DIR / "g15_neighbor_examples.csv", index=False, encoding="utf-8-sig")

    chosen_val_rows: list[pd.Series] = []
    for model_id in ["G15A_相似历史检索", "G15B_检索加残差修正"]:
        part = val_df[val_df["model_id"].astype(str).eq(model_id)].copy()
        part = part.sort_values(["test_rmse", "selection_score"], ascending=True)
        if not part.empty:
            chosen_val_rows.append(part.iloc[0])
    chosen_df_list: list[pd.DataFrame] = []
    for row in chosen_val_rows:
        hit = test_df[
            test_df["model_id"].astype(str).eq(str(row["model_id"]))
            & test_df["feature_set"].astype(str).eq(str(row["feature_set"]))
            & test_df["k"].astype(int).eq(int(row["k"]))
        ].copy()
        if not hit.empty:
            chosen_df_list.append(hit.head(1))
    chosen_df = pd.concat(chosen_df_list, ignore_index=True) if chosen_df_list else pd.DataFrame()
    chosen_df.to_csv(OUT_DIR / "g15_test_chosen_by_validation.csv", index=False, encoding="utf-8-sig")

    # Reconstruct chosen predictions for plotting and future comparisons.
    for _, row in chosen_df.iterrows():
        feature_set = str(row["feature_set"])
        k = int(row["k"])
        raw_features = deployable_feature_sets[feature_set]
        z, _stat = standardize_from_train(raw_features, train_idx)
        neigh_train, dist_train = nearest_neighbors_between(z, train_idx, train_idx, max_k=max(K_VALUES), exclude_self=True)
        neigh_val, dist_val = nearest_neighbors_between(z, train_idx, val_idx, max_k=max(K_VALUES), exclude_self=False)
        neigh_test, dist_test = nearest_neighbors_between(z, train_idx, test_idx, max_k=max(K_VALUES), exclude_self=False)
        base_train = predict_from_neighbors(y_pool, train_idx, neigh_train, dist_train, k=k)
        base_val = predict_from_neighbors(y_pool, val_idx, neigh_val, dist_val, k=k)
        base_test = predict_from_neighbors(y_pool, test_idx, neigh_test, dist_test, k=k)
        pred = base_test
        if str(row["model_id"]) == "G15B_检索加残差修正" and pd.notna(row.get("alpha")):
            x_train = residual_features(z, train_idx, base_train, neigh_train, dist_train, k, descriptor_pool)
            x_val = residual_features(z, val_idx, base_val, neigh_val, dist_val, k, descriptor_pool)
            x_test = residual_features(z, test_idx, base_test, neigh_test, dist_test, k, descriptor_pool)
            x_train_z, others_z, _ = standardize_matrix(x_train, x_val, x_test)
            _x_val_z, x_test_z = others_z
            weights = fit_ridge_multioutput(x_train_z, y_train[:, :, 0] - base_train[:, :, 0], alpha=float(row["alpha"]))
            delta = predict_ridge(x_test_z, weights)
            pred = base_test.copy()
            pred[:, :, 0] = pred[:, :, 0] + float(row["residual_scale"]) * delta
        chosen_preds[str(row["model_id"])] = pred
        chosen_rows.append(row.to_dict())

    if chosen_preds:
        pred_payload: dict[str, Any] = {
            "sample_key": meta_test["sample_key"].astype(str).to_numpy(dtype="<U512"),
            "true": y_test,
            "mask": mask_test,
            "ctx": ctx_test,
            "model_names": np.asarray(list(chosen_preds.keys()), dtype="<U128"),
        }
        for i, (_name, pred) in enumerate(chosen_preds.items()):
            pred_payload[f"pred_{i}"] = pred
        np.savez_compressed(
            OUT_DIR / "g15_chosen_predictions_test.npz",
            **pred_payload,
        )

    baseline_maps = {
        "E10C": load_prediction_map("RESTORE_E10C*seed2026*"),
        "E5A": load_prediction_map("RESTORE_E5A*seed2026*"),
        "E6": load_prediction_map("RESTORE_E6*seed2026*"),
    }
    selected_keys: list[str] = []
    if not g11_catalog.empty:
        order_col = "E10C_rmse_2s" if "E10C_rmse_2s" in g11_catalog.columns else "case_score"
        selected_keys = g11_catalog.sort_values(order_col, ascending=False)["sample_key"].astype(str).head(12).tolist()
    plot_selected_cases(OUT_DIR / "figures", meta_test, y_test, ctx_test, chosen_preds, selected_keys, baseline_maps)

    save_json(
        RUN_META_PATH,
        {
            "manifest": str(DEFAULT_MANIFEST),
            "dropped_samples": int(dropped),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "k_values": K_VALUES,
            "alphas": ALPHAS,
            "residual_scales": RESIDUAL_SCALES,
            "feature_sets": {name: int(arr.shape[1]) for name, arr in deployable_feature_sets.items()},
            "descriptor_names": descriptor_names,
            "context_meta": context_meta,
            "chosen_by_validation": chosen_rows,
        },
    )

    report = build_report(val_df, test_df, chosen_df, group_df, g11_detail_df)
    write_text(OUT_DIR / "g15_retrieval_residual_report_cn.md", report)
    print(f"done: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
